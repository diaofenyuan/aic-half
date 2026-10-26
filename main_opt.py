"""三气种资源预测与发电调度 MILP。

边界说明：
- 高炉气柜只有 2 号柜数据可用，焦炉/转炉没有独立柜位，不虚构库存。
- 官方提供参考热值，密度与效率未完整确定；合法训练期气电综合回归保留 inferred，
  不用质量比热值直接乘体积流量，也不把原始列缩放升级为官方常数。
- 已进入正式半决赛；输出字段已核对赛题，代理资源调度仍保留模拟标记。
"""
from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.linear_model import LinearRegression

import config as C
from data_cleaning import (PreparedData, build_base_features, resource_series,
                           target_calendar)


def load_resource_model(text: str):
    return lgb.Booster(model_str=text)


@dataclass
class ResourceBundle:
    models: dict
    final_models: dict
    feature_names: list[str]
    metrics: dict = field(default_factory=dict)
    train_rows: int = 0
    train_seconds: float = 0.0
    heat_rate: dict = field(default_factory=dict)
    final_heat_rate: dict = field(default_factory=dict)
    gas_caps: dict = field(default_factory=dict)
    final_gas_caps: dict = field(default_factory=dict)
    train_end: str | None = None
    final_train_end: str | None = None
    blocker: str | None = None
    selection: dict = field(default_factory=dict)
    dispatch_selection: dict = field(default_factory=dict)
    resource_nn: dict | None = None
    final_resource_nn: dict | None = None
    physical_validation: dict = field(default_factory=dict)
    final_physical_validation: dict = field(default_factory=dict)


def physical_submission_check(optimization: dict) -> dict:
    """代理计划校验与真实物理核验分开，单个成功标记不能替代约束证据。"""
    summary = optimization.get("summary", {})
    evidence = summary.get("physical_validation", {})
    checks = {
        "net_resource_boundary": summary.get("resource_mode") == "net_surplus_after_priority_users",
        "plan_constraints": summary.get("plan_validation_passed") is True,
        "physical_feasibility": summary.get("physical_feasibility_certified") is True,
        "daily_publication": (summary.get("dispatch_mode") == "daily_fixed"
                              and summary.get("intraday_refresh") is False),
        "datetime_semantics": (C.OPT_CONFIG["datetime_semantics_confirmed"] is True
                               and str(summary.get("time_semantics", "")).startswith(
                                   C.OPT_CONFIG["datetime_semantics"])),
    }
    descriptions = {
        "net_resource_boundary": "可用气仍为历史耗气代理包络，未核验扣除优先生产用户后的净资源边界",
        "plan_constraints": "优化计划约束核验未通过",
        "physical_feasibility": "真实气网物理可行性尚未核验",
        "daily_publication": "优化结果未按官方要求每日固定发布",
        "datetime_semantics": "优化 datetime 未按官方指令生效时刻生成",
        "gas_to_power": "训练期气电经验映射缺少通过的验证记录",
        "holder_mapping": "气柜列、容量与实际库存映射缺少核验依据",
        "converter_units": "转炉煤气单位与换算缺少核验依据",
        "priority_user_supply": "优先生产用户全时段供气约束缺少核验依据",
    }
    for name in ("gas_to_power", "holder_mapping", "converter_units", "priority_user_supply"):
        item = evidence.get(name, {})
        checks[name] = (isinstance(item, dict) and item.get("passed") is True
                        and isinstance(item.get("evidence"), str) and bool(item["evidence"].strip()))
        if isinstance(item, dict) and item.get("reason"):
            descriptions[name] = item["reason"]
    if evidence.get("official_constraints"):
        checks["official_constraints"] = evidence["official_constraints"].get("passed") is True
        descriptions["official_constraints"] = "当前配置与赛题已明确的硬约束不一致"
    blocked = [name for name, passed in checks.items() if not passed]
    return {"passed": not blocked, "checks": checks, "blocked_items": blocked,
            "evidence_basis": {name: evidence.get(name, {}).get("basis") for name in checks},
            "problems": [descriptions[name] for name in blocked]}


def _physical_stats(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {"rows": len(values), "valid_rows": len(finite),
        "valid_fraction": float(len(finite) / max(len(values), 1)),
        "min": float(finite.min()) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None}


def physical_gas_ledger(prepared, end_pos):
    """已知用户扣减账本；混气站进气扣一次，未知项不补零，不冒充完整净余气。"""
    prepared.require_training_data()
    if not 0 <= end_pos < len(prepared.index):
        raise ValueError("物理核验截止位置无效")
    frame = prepared.frame.iloc[:end_pos + 1]

    def values(columns):
        block = frame.reindex(columns=list(columns)).replace([np.inf, -np.inf], np.nan)
        block = block.where(block >= 0.)
        return block.sum(axis=1, min_count=len(columns))

    specifications = {
        "blast": {"generation": tuple(f"gas_blast_furnace_{i}" for i in range(1, 6)),
            "users": tuple(f"gas_air_heater_{i}" for i in range(1, 6)) +
                     tuple(f"gas_user_blast_furnace_user{i}" for i in range(1, 5)),
            "mixed": "gas_into_gas_mixed_blast_furnace"},
        "coke": {"generation": ("gas_coke_oven_1",), "users": (), "mixed": "gas_into_gas_mixed_coke"},
        "converter": {"generation": ("gas_converter_1",),
            "users": tuple(sorted({"gas_user_converter_user1", "gas_user_converter_user2"} |
                {c for c in frame if c.startswith("gas_user_converter_user")})),
            "mixed": "gas_into_gas_mixed_converter"},
    }
    ledger, details = pd.DataFrame(index=frame.index), {}
    for gas, specification in specifications.items():
        generation = values(specification["generation"])
        if gas == "converter":
            generation = generation * C.CONVERTER_GAS_SCALE
        # 焦炉原始表无独立生产用户列；只扣已知混气站进气，不宣称其它用户需求为零。
        priority = values((*specification["users"], specification["mixed"]))
        ledger[gas] = generation - priority
        required = (*specification["generation"], *specification["users"], specification["mixed"])
        missing = [c for c in required if c not in frame or not np.isfinite(frame[c].to_numpy(dtype=float)).any()]
        details[gas] = {"generation_columns": list(specification["generation"]),
            "priority_user_columns": list(specification["users"]), "mixed_station_column": specification["mixed"],
            "unavailable_columns": missing, "net_before_generation": _physical_stats(ledger[gas]),
            "negative_net_rows": int((ledger[gas] < 0.).sum()),
            "quantity_role": "remaining_after_known_deductions; not_certified_net_surplus",
            "all_users_enumerated": False,
            "unit_basis": "historical_inferred_scale" if gas == "converter" else "official_standard_m3h; raw_sensor_scaling_unverified"}
    return ledger, details


def build_physical_validation(prepared, end_pos, heat_rate):
    prepared.require_training_data()
    if not 0 <= end_pos < len(prepared.index):
        raise ValueError("物理核验截止位置无效")
    frame = prepared.frame.iloc[:end_pos + 1]
    end = str(frame.index[-1])
    calibration_end = heat_rate.get("calibration_end")
    if calibration_end is not None and pd.Timestamp(calibration_end) > frame.index[-1]:
        raise ValueError("气电标定截止晚于物理核验前缀，禁止使用未来拟合参数")
    source = "赛题《煤气发电量预测与发电优化-3》第5页、第7页、第9—10页；data_dictionary.xlsx 数据说明"
    evidence = {"training_end": end, "scope": "legal_training_prefix_only; no_official_test",
        "input_semantics": "causally_preprocessed_same_timestamp_grid; no_target_imputation",
        "data_sources": prepared.source_hashes, "test_data_used": False}
    rules = (tuple(C.OPT_CONFIG["unit_capacity_mw"]) == (50., 50., 50., 50., 120., 120.)
        and C.OPT_CONFIG["min_load_ratio"] == .6 and C.OPT_CONFIG["ramp_fraction_per_min"] == .01
        and C.OPT_CONFIG["ramp_scope"] == "aggregate_generator_1_and_all"
        and C.GAS_HOLDER["capacity_m3"] == 200000.
        and C.GAS_HOLDER["safe_low_ratio"] == .15 and C.GAS_HOLDER["safe_high_ratio"] == .9)
    evidence["official_constraints"] = {"passed": bool(rules), "basis": "official_task",
        "evidence": source + "；用户补充的2026-10-26赛事统一答复：爬坡修正为1%，启停按聚合负荷检查；其余容量、负荷和安全区沿用赛题。",
        "unified_reply": {"source": "用户补充的赛事统一答复（2026-10-25），AGENTS.md第11节",
            "rolling_information_boundary": "known_at_or_before_origin_for_forecast_and_dispatch",
            "annual_tariffs_known_in_advance": True, "optimization_covers_test_dates": True},
        "latest_reply": {"source": "用户补充的赛事统一答复（2026-10-26）",
            "datetime": "instruction_effective_time", "coverage": "align_with_long_forecast_96_steps",
            "gas_flow_standard_unit": "m3/h", "holder_capacity_by_column_m3": C.GAS_HOLDER["capacity_by_column_m3"],
            "ramp_fraction_per_min": .01, "startup_shutdown_scope": "aggregate_load",
            "ramp_time_base_source": C.OPT_CONFIG["ramp_time_base_source"],
            "coverage_interpretation": "same_datetime_as_long_forecast; one_fixed_publication_per_day",
            "test_generator_use": "first_48_hours_initial_reference; later_empty_by_design",
            "test_reference_boundary": "only_observed_prefix_for_operational_initialization; no_predictor_training_or_tuning",
            "heat_reference": {"blast": 750., "converter": 1250., "coke": 4150., "unit_as_published": "J/kg",
                "approximate_reference_only": True, "direct_m3h_to_MW_conversion": False,
                "reason": "质量比热值不能直接乘体积流量得到MW；缺少密度及机组效率，不擅改通知单位"}}}

    validation = heat_rate.get("prefix_validation", {})
    coefficients = heat_rate.get("coefficients_mw_per_m3h", {})
    positive = all(np.isfinite(coefficients.get(g, np.nan)) and coefficients.get(g, 0.) > 0.
                   for g in C.RESOURCE_NAMES)
    mapping_ok = bool(calibration_end is not None and heat_rate.get("ok") and validation.get("passed")
        and validation.get("r2", -np.inf) >= C.HEAT_RATE["min_r2"]
        and validation.get("wape", np.inf) <= C.HEAT_RATE["max_validation_wape"]
        and abs(validation.get("bias_ratio", np.inf)) <= C.HEAT_RATE["max_validation_abs_bias"] and positive)
    evidence["gas_to_power"] = {"passed": False, "basis": "training_calibrated_inferred",
        "evidence": "合法训练前缀内非负无截距气电回归，末7天留出验证；仅核验原始耗气输入到MW的经验映射。",
        "coefficients": coefficients, "prefix_validation": validation,
        "calibration_end": calibration_end,
        "official_heat_reference_provided": True, "official_efficiency_provided": False,
        "separate_heat_and_efficiency_identifiable": False,
        "flow_units_confirmed": True, "flow_unit_basis": "official_standard_m3_per_hour_for_generator_consumption",
        "usable_for_empirical_conversion": mapping_ok,
        "aggregate_mapping_passed": mapping_ok, "generator_1_mapping_passed": False,
        "official_model_trigger": "load_deviation_correction_or_missing_load_inference",
        "reason": ("总负荷经验映射已核验，但四台50MW机组的独立气电映射尚未通过跨期验证"
                   if mapping_ok else "经验气电映射未通过训练内留出核验")}

    holders = {}
    for col in sorted(c for c in frame if c.startswith("gas_holder_")):
        x = frame[col].to_numpy(dtype=float)
        valid = np.isfinite(x)
        capacity = C.GAS_HOLDER["capacity_by_column_m3"].get(col)
        low = capacity * C.GAS_HOLDER["safe_low_ratio"] if capacity is not None else None
        high = capacity * C.GAS_HOLDER["safe_high_ratio"] if capacity is not None else None
        holders[col] = {**_physical_stats(x), "capacity_m3": capacity,
            "safe_low_m3": low, "safe_high_m3": high, "deployed": col == C.GAS_HOLDER["column"],
            "inside_capacity_fraction": float(np.mean((x[valid] >= 0.) & (x[valid] <= capacity)))
            if valid.any() and capacity is not None else None,
            "inside_official_safety_fraction": float(np.mean((x[valid] >= low) & (x[valid] <= high)))
            if valid.any() and capacity is not None else None}
    column = C.GAS_HOLDER["column"]
    capacity = C.GAS_HOLDER["capacity_by_column_m3"].get(column)
    holder_ok = (column == "gas_holder_blast_furnace_gas_holder_2" and capacity == C.GAS_HOLDER["capacity_m3"]
        and holders.get(column, {}).get("valid_rows", 0) > 0
        and holders[column]["inside_capacity_fraction"] == 1.
        and 30000. <= holders[column]["median"] <= 180000.)
    evidence["holder_mapping"] = {"passed": False, "basis": "official_independent_holder_safety_requirement",
        "evidence": "2026-10-26赛事统一答复：1号柜30万m³，2号柜20万m³；实际部署2号柜按m³读取并复核数值范围。",
        "column": C.GAS_HOLDER["column"], "capacity_m3": C.GAS_HOLDER["capacity_m3"],
        "capacity_by_column_m3": C.GAS_HOLDER["capacity_by_column_m3"], "deployed_column_scale": 1.,
        "columns": holders, "deployed_column_mapping_passed": bool(holder_ok),
        "all_holder_safety_verified": False,
        "reason": ("2号柜映射已核验；1号柜单位与状态不可用，无法证明两座独立气柜均满足安全区间"
                   if holder_ok else "实际部署气柜的容量、量级或数值范围核验未通过"),
        "holder_1_scale_verified": False,
        "holder_1_deployed": False}

    _, ledger = physical_gas_ledger(prepared, end_pos)
    evidence["priority_user_supply"] = {"passed": False, "basis": "official_priority_rule_and_training_ledger",
        "evidence": source + "：先保生产，煤气发生量扣生产用户及混气站进气；混气站上游扣减不重复追加下游混气用户。",
        "accounting_implemented": True, "gas_ledger": ledger,
        "forecast_or_dispatch_replaced": False,
        "reason": "官方允许依据历史推测未观测用户；当前净资源候选尚未通过完整调度晋级，正式模型仍为耗气代理"}

    # 已知用量只构成所需发生量的下界，未知用户和放散使比例不能唯一确定量纲。
    known_users = [c for c in frame if c.startswith("gas_user_converter_user")
                   and np.isfinite(frame[c].to_numpy(dtype=float)).any()]
    required = [*known_users, "gas_into_gas_mixed_converter", "d_use_converter"]
    required_flow = frame.reindex(columns=required).where(lambda x: x >= 0.).sum(axis=1, min_count=len(required))
    raw = frame.get("gas_converter_1", pd.Series(np.nan, index=frame.index))
    ratio = required_flow / raw.where(raw > 0.)
    valid = ratio.notna()
    evidence["converter_units"] = {"passed": False, "basis": "historical_inferred_not_unique",
        "evidence": "2026-10-26赛事统一答复已明确标准m³/h；原始转炉发生量与已知用量仍不闭合，历史倍率不等于官方单位换算。",
        "standard_unit_confirmed": True, "standard_unit": "m3/h", "raw_converter_scale_verified": False,
        "inferred_scale": C.CONVERTER_GAS_SCALE, "known_consumption_scale_lower_bound": _physical_stats(ratio),
        "inferred_scale_below_known_demand_fraction": float((ratio[valid] > C.CONVERTER_GAS_SCALE).mean())
        if valid.any() else None,
        "reason": "标准m³/h已明确，但converter_1原始量级归一仍缺可闭合证据；7043.5保持历史推算标记"}
    evidence["datetime_semantics"] = {"passed": bool(C.OPT_CONFIG["datetime_semantics_confirmed"]),
        "basis": "official_reply", "evidence": "2026-10-26最新统一答复：优化时间列与长期预测一致，每天提交一次，日内不滚动刷新。",
        "prediction_origin_and_tail_confirmed": True, "optimization": C.OPT_CONFIG["datetime_semantics"],
        "reason": None if C.OPT_CONFIG["datetime_semantics_confirmed"] else "未启用官方指令生效时刻口径"}
    evidence["economic_scoring"] = {"passed": False, "basis": "partially_documented",
        "evidence": source + "：相对收益提升=(优化效益-基准效益)/基准效益；替代外购电收益需计放散惩罚。",
        "benefit_formula_confirmed": True, "release_penalty_coefficient_confirmed": False,
        "historical_release_observed": False, "research_penalty_is_official": False,
        "reason": "未提供官方放散罚金系数与真实历史放散，不能报告官方净收益提升"}
    return evidence


def refresh_physical_validation(bundle: ResourceBundle, prepared: PreparedData,
                                use_final: bool = False) -> dict:
    """补齐旧模型核验记录，复核只读训练前缀，不替换归档模型的实际系数。"""
    prepared.require_training_data()
    end = bundle.final_train_end if use_final else bundle.train_end
    positions = prepared.index.get_indexer([pd.Timestamp(end)])
    if positions[0] < 1:
        raise ValueError("实际资源模型的训练截止点不在合法训练网格")
    pos = int(positions[0])
    deployed = bundle.final_heat_rate if use_final else bundle.heat_rate
    heat = dict(deployed)
    mismatch = False
    if "calibration_end" not in heat or "bias_ratio" not in heat.get("prefix_validation", {}):
        audited = estimate_heat_rate(prepared, pos)
        # 旧归档缺少诊断时，只在参数一致的情况下关联新的前缀复核，禁止悄悄换系数。
        mismatch = not all(np.isclose(deployed.get("coefficients_mw_per_m3h", {}).get(g, np.nan),
            audited.get("coefficients_mw_per_m3h", {}).get(g, np.nan), rtol=1e-9, atol=1e-12)
            for g in C.RESOURCE_NAMES)
        if not mismatch:
            heat = audited
        else:
            heat.setdefault("calibration_end", str(prepared.index[pos]))
    validation = build_physical_validation(prepared, pos, heat)
    if mismatch:
        validation["gas_to_power"].update(passed=False, usable_for_empirical_conversion=False,
            reason="归档气电系数与当前训练前缀复核不同，未借用新参数的验证结论")
    validation["deployed_coefficients_unchanged"] = True
    if use_final:
        bundle.final_physical_validation = validation
    else:
        bundle.physical_validation = validation
    return validation


def _resource_base(prepared: PreparedData) -> pd.DataFrame:
    base = build_base_features(prepared, include_targets=False)
    # 包络观测视为外生代理输入；柜位属于控制状态，不能把历史调度柜位注入反事实控制。
    return base[[c for c in base if "holder" not in c]]


def _resource_design(prepared: PreparedData, base: pd.DataFrame,
                     positions: np.ndarray, horizons: tuple[int, ...]):
    cal = target_calendar(prepared, positions, horizons)
    names = list(base.columns) + list(cal.keys()) + ["h"]
    return cal, names


def _stack(prepared: PreparedData, base_values: np.ndarray, cal: dict,
           names: list[str], positions: np.ndarray, horizons: tuple[int, ...],
           series: np.ndarray, weights: np.ndarray | None = None
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(positions)
    width = base_values.shape[1] + len(cal) + 1
    X = np.empty((n * len(horizons), width), dtype=np.float32)
    y = np.empty(n * len(horizons), dtype=np.float64)
    w = np.ones(n * len(horizons), dtype=np.float32)
    for j, h in enumerate(horizons):
        lo = j * n
        parts = [base_values[positions]]
        for key in cal:
            parts.append(cal[key][:, j][:, None])
        parts.append(np.full((n, 1), float(h), dtype=np.float32))
        X[lo:lo + n] = np.hstack(parts)
        y[lo:lo + n] = series[positions + int(h)]
        if weights is not None:
            w[lo:lo + n] = weights[:, j]
    valid = np.isfinite(y)
    return X[valid], y[valid], w[valid]


def _fit_resource_models(prepared: PreparedData, positions: np.ndarray,
                         num_round: int, ref_pos: int,
                         objectives: dict | None = None,
                         horizons: tuple[int, ...] = C.HORIZONS_L,
                         gases: tuple[str, ...] = C.RESOURCE_NAMES) -> tuple[dict, list[str]]:
    prepared.require_training_data()
    if len(positions) == 0 or np.max(positions) + max(horizons) > ref_pos:
        raise ValueError("资源模型训练目标跨越冻结截止点")
    base = _resource_base(prepared)
    base_values = base.to_numpy(dtype=np.float32)
    cal, names = _resource_design(prepared, base, positions, horizons)
    series = resource_series(prepared)
    age = (ref_pos - positions[:, None] - np.asarray(horizons)[None, :]) / C.STEPS_PER_DAY
    decay = (np.power(0.5, np.maximum(age, 0.0) / float(C.DECAY_DAYS))
             if C.DECAY_DAYS else np.ones_like(age))
    decay = decay / max(float(decay.mean()), 1e-9)
    params = {**C.LGBM_BASE, "num_threads": C.LGBM_BASE["num_threads"]}
    models = {}
    for gas in C.progress(gases, desc="资源模型拟合", unit="气种"):
        X, y, weights = _stack(prepared, base_values, cal, names, positions,
                               horizons, series[gas].to_numpy(dtype=float), decay)
        gas_params = {**params}
        if objectives:
            gas_params["objective"] = objectives[gas]
        logging.getLogger("aic-half").info("资源拟合: gas=%s objective=%s rows=%d",
                                          gas, gas_params["objective"], len(y))
        models[gas] = lgb.train(gas_params, lgb.Dataset(
            X, label=y, weight=weights, feature_name=names, free_raw_data=True),
            num_boost_round=num_round)
    return models, names


def forecast_resources(bundle: ResourceBundle, prepared: PreparedData,
                       origins: pd.DatetimeIndex, use_final: bool = False) -> np.ndarray:
    positions = prepared.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("资源预测起报点不在数据网格内")
    base = _resource_base(prepared)
    base_values = base.to_numpy(dtype=np.float32)
    cal, names = _resource_design(prepared, base, positions, C.HORIZONS_L)
    if names != bundle.feature_names:
        raise ValueError("资源模型特征顺序不一致")
    n = len(positions)
    width = base_values.shape[1] + len(cal) + 1
    X = np.empty((n * len(C.HORIZONS_L), width), dtype=np.float32)
    for j, h in enumerate(C.HORIZONS_L):
        lo = j * n
        parts = [base_values[positions]]
        for key in cal:
            parts.append(cal[key][:, j][:, None])
        parts.append(np.full((n, 1), float(h), dtype=np.float32))
        X[lo:lo + n] = np.hstack(parts)
    out = np.empty((n, len(C.HORIZONS_L), len(C.RESOURCE_NAMES)))
    models = bundle.final_models if use_final else bundle.models
    for gi, gas in C.progress(enumerate(C.RESOURCE_NAMES), total=len(C.RESOURCE_NAMES),
                              desc="资源批量预测", unit="气种"):
        pred = models[gas].predict(X).reshape(len(C.HORIZONS_L), n).T
        out[:, :, gi] = np.maximum(pred, 0.0)
        head = models.get(gas + "__head")
        if head is not None:
            # 短资源头独立拟合八步，仅替换控制近期；96步尾部仍由长期模型提供。
            k = len(C.HORIZONS_S)
            weight = float(bundle.selection.get("head_selection", {}).get("weight", 1.))
            estimate = head.predict(X[:k * n]).reshape(k, n).T
            out[:, :k, gi] = (1. - weight) * out[:, :k, gi] + weight * np.maximum(estimate, 0.)
    adjustment = bundle.selection.get("availability_adjustment", {})
    if adjustment:
        heat = bundle.final_heat_rate if use_final else bundle.heat_rate
        expert = None
        if adjustment.get("method") == "positive_nn_peak":
            import main_nn
            pack = bundle.final_resource_nn if use_final else bundle.resource_nn
            if pack is None:
                raise ValueError("选定的资源网络缺失，禁止冒用持久性预测")
            expert = main_nn.predict_resource_nn(pack, base, resource_series(prepared), positions)
        out = _adjust_available_resources(out, prepared, origins, heat, adjustment, expert=expert)
    return out


def _adjust_available_resources(forecast: np.ndarray, prepared: PreparedData,
                                origins: pd.DatetimeIndex, heat: dict,
                                params: dict, prices: dict | None = None,
                                expert: np.ndarray | None = None) -> np.ndarray:
    """高价段仅用起报点供气修正短端低估；缺测时保留原预测，不构造实际供气。"""
    out = np.array(forecast, dtype=float, copy=True)
    if not params or not heat.get("ok"):
        return out
    if params.get("method") not in ("positive_persistence_peak", "positive_nn_peak"):
        raise ValueError("未知的资源修正策略")
    threshold, weight = float(params["threshold_ratio"]), float(params["weight"])
    if not 0. <= threshold <= 1. or not 0. <= weight <= 1.:
        raise ValueError("资源修正参数超出范围")
    positions = prepared.index.get_indexer(origins)
    if out.ndim != 3 or out.shape[0] != len(positions) or out.shape[2] != 3 or (positions < 0).any():
        raise ValueError("资源修正输入与起报点不一致")
    beta = np.array([heat["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    if not np.isfinite(beta).all() or (beta <= 0).any():
        raise ValueError("资源修正缺少有效气电标定")
    k = min(len(C.HORIZONS_S), out.shape[1])
    observed = resource_series(prepared).to_numpy(dtype=float)[positions]
    valid = np.isfinite(observed).all(axis=1) & (observed >= 0).all(axis=1)
    candidate = observed[:, None, :]
    if params["method"] == "positive_nn_peak":
        if expert is None or expert.shape != (len(positions), k, 3):
            raise ValueError("资源网络预测与短提前量不一致")
        candidate = expert
        valid &= np.isfinite(expert).all(axis=(1, 2)) & (expert >= 0).all(axis=(1, 2))
    # 先屏蔽缺测整行，避免未启用的分支仍因 0*NaN 污染预测。
    difference = np.where(valid[:, None, None], candidate - out[:, :k], 0.)
    energy = out[:, :k] @ beta
    times = pd.DatetimeIndex(origins).to_numpy()[:, None] + (
        np.arange(1, k + 1)[None, :] * np.timedelta64(C.STEP_MINUTES, "m"))
    tariff = prices if prices is not None else load_prices()
    flat = pd.DatetimeIndex(times.ravel())
    missing = sorted(set(flat.month) - set(tariff))
    if missing:
        raise ValueError(f"缺少资源高价段判断所需的官方电价月份: {missing}")
    high = np.array([tariff[t.month][t.hour * 4 + t.minute // 15] >= tariff[t.month].mean()
                     for t in flat]).reshape(len(positions), k)
    enabled = valid[:, None] & high & ((difference @ beta) > threshold * np.maximum(energy, 1.))
    out[:, :k] += weight * enabled[:, :, None] * np.maximum(difference, 0.)
    return out


def estimate_heat_rate(prepared: PreparedData, end_pos: int) -> dict:
    prepared.require_training_data()
    if not 0 < end_pos < len(prepared.index):
        raise ValueError("气电标定截止位置无效")
    frame = prepared.frame.iloc[:end_pos]
    y = prepared.labels["generator_all"].iloc[:end_pos].to_numpy(dtype=float)
    X = frame[["d_use_blast", "d_use_coke", "d_use_converter"]].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if int(valid.sum()) < 1000:
        return {"ok": False, "reason": "INSUFFICIENT_HISTORY"}
    model = LinearRegression(positive=True, fit_intercept=False).fit(X[valid], y[valid])
    pred = model.predict(X[valid])
    ss_res = float(np.sum((y[valid] - pred) ** 2))
    ss_tot = float(np.sum((y[valid] - y[valid].mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    coef = {gas: float(v) for gas, v in zip(C.RESOURCE_NAMES, model.coef_)}
    ok = bool(r2 >= C.HEAT_RATE["min_r2"] and min(coef.values()) > 1e-7)
    split = max(0, len(y) - 7 * C.STEPS_PER_DAY)
    fit_valid, hold_valid = valid.copy(), valid.copy()
    fit_valid[split:], hold_valid[:split] = False, False
    validation = {"scope": "last_7_days_within_training_prefix", "rows": int(hold_valid.sum()),
                  "start": str(prepared.index[split]), "end": str(prepared.index[end_pos - 1]),
                  "fit_end": str(prepared.index[split - 1]) if split else None}
    if fit_valid.sum() >= 1000 and hold_valid.sum() >= 96:
        prefix_model = LinearRegression(positive=True, fit_intercept=False).fit(X[fit_valid], y[fit_valid])
        hold_pred, hold_y = prefix_model.predict(X[hold_valid]), y[hold_valid]
        hold_r2 = 1 - np.sum((hold_pred - hold_y) ** 2) / max(np.sum((hold_y - hold_y.mean()) ** 2), 1e-9)
        validation.update(r2=float(hold_r2),
                          wape=float(np.abs(hold_pred - hold_y).sum() / max(hold_y.sum(), 1e-9)),
                          bias_ratio=float((hold_pred - hold_y).sum() / max(hold_y.sum(), 1e-9)),
                          passed=bool(hold_r2 >= C.HEAT_RATE["min_r2"]))
        ok = ok and validation["passed"]
    else:
        validation["passed"] = False
        ok = False
    return {"ok": ok, "coefficients_mw_per_m3h": coef, "r2": r2,
            "calibration_end": str(prepared.index[end_pos - 1]),
            "coefficient_unit_status": "empirical_MW_per_generator_m3h; raw_generation_scaling_checked_separately",
            "prefix_validation": validation,
            "valid_rows": int(valid.sum()), "method": C.HEAT_RATE["method"],
            "note": "合法训练前缀标定的气电综合系数，不分别识别热值与效率；标准耗气单位为m³/h，原始发生量缩放另行核验"}


def _gas_caps(prepared: PreparedData, end_pos: int) -> dict:
    prepared.require_training_data()
    history = resource_series(prepared).iloc[:end_pos + 1]
    return {gas: float(max(history[gas].quantile(C.OPT_CONFIG["gas_cap_quantile"])
                            * C.OPT_CONFIG["gas_cap_headroom"], 1.0))
            for gas in C.RESOURCE_NAMES}


def score_resources(forecast: np.ndarray, truth: np.ndarray) -> dict:
    """零气量计入绝对误差；累计量按实际预测跨度统计，兼容独立八步头。"""
    if forecast.shape != truth.shape or not np.isfinite(forecast).all():
        raise ValueError("资源预测维度错误或含非有限数")
    metrics = {}
    horizons = tuple(range(1, forecast.shape[1] + 1))
    hours = len(horizons) * C.STEP_MINUTES / 60.
    for gi, gas in enumerate(C.RESOURCE_NAMES):
        actual, estimate = truth[:, :, gi], forecast[:, :, gi]
        valid = np.isfinite(actual)
        positive = valid & (actual > 0)
        zero = valid & (actual == 0)
        error = estimate - actual
        denominator = float(actual[valid].sum())
        complete = valid.all(axis=1)
        cumulative = error[complete].sum(axis=1) * C.STEP_MINUTES / 60.0
        metrics[gas] = {
            "accuracy_1_mape": (float(100 * (1 - np.mean(np.abs(error[positive]) / actual[positive])))
                                 if positive.any() else None),
            "wape": float(np.abs(error[valid]).sum() / denominator) if denominator > 0 else None,
            "bias_ratio": float(error[valid].sum() / denominator) if denominator > 0 else None,
            "mae_m3h": float(np.abs(error[valid]).mean()) if valid.any() else None,
            "zero_fraction": float(zero.sum() / max(valid.sum(), 1)),
            "zero_false_supply_mae_m3h": float(estimate[zero].mean()) if zero.any() else None,
            f"cumulative_{hours:g}h_error_p10_m3": float(np.quantile(cumulative, .1)) if len(cumulative) else None,
            f"cumulative_{hours:g}h_error_p90_m3": float(np.quantile(cumulative, .9)) if len(cumulative) else None,
            "valid_cells": int(valid.sum()), "complete_paths": int(complete.sum()),
            "groups": {},
        }
        metrics[gas]["cumulative_paths"] = {}
        for count in (8, 32, 96):
            if count > forecast.shape[1]:
                continue
            good = valid[:, :count].all(axis=1)
            # 正号代表实际气量高于预测，直接对应实际库存减计划库存。
            residual = -error[good, :count].sum(axis=1) * .25
            metrics[gas]["cumulative_paths"][f"{count / 4:g}h"] = {
                "paths": int(good.sum()), "mean_m3": float(np.mean(residual)) if len(residual) else None,
                "p10_m3": float(np.quantile(residual, .1)) if len(residual) else None,
                "p90_m3": float(np.quantile(residual, .9)) if len(residual) else None,
                "min_m3": float(np.min(residual)) if len(residual) else None,
                "max_m3": float(np.max(residual)) if len(residual) else None}
        for name, lo, hi in C.LONG_GROUPS:
            idx = [j for j, h in enumerate(horizons) if lo <= h <= hi]
            if not idx:
                continue
            a, e = actual[:, idx], error[:, idx]
            ok = np.isfinite(a)
            denom = float(a[ok].sum())
            metrics[gas]["groups"][name] = {
                "wape": float(np.abs(e[ok]).sum() / denom) if denom > 0 else None,
                "bias_ratio": float(e[ok].sum() / denom) if denom > 0 else None}
    return metrics


def _select_dispatch(controls: list[dict], candidates: dict) -> dict:
    """以完整同窗回放晋级，收益改善不能以增加降载或代理释放换取。"""
    ranking = []
    for name, (steps, options) in candidates.items():
        gains, releases, passed = [], [], bool(controls)
        for item in controls:
            baseline, trial = item["comparison"]["daily"], item["comparison"][name]
            before, after = (x.get("proxy_inventory_adjusted_revenue_cny") for x in (baseline, trial))
            valid = (baseline.get("plan_validation_passed") and baseline.get("replay_complete")
                     and trial.get("plan_validation_passed") and trial.get("replay_complete")
                     and before is not None and before > 0 and after is not None)
            if not valid:
                passed = False
                continue
            gains.append(after / before - 1.)
            releases.append(trial["proxy_blast_vent_m3"])
            passed = (passed
                and trial["proxy_curtailed_generation_mwh"] <= baseline["proxy_curtailed_generation_mwh"] + 1e-5
                and trial["proxy_blast_vent_m3"] <= baseline["proxy_blast_vent_m3"] + 1.
                and (trial["price_weighted_load_ratio"] or 0.) >= (baseline["price_weighted_load_ratio"] or 0.) - .001)
        accepted = (name != "daily" and passed and len(gains) == len(controls)
                    and np.mean(gains) >= .002 and min(gains) >= -.01)
        ranking.append({"candidate": name, "accepted": bool(accepted),
                        "mean_gain_ratio": float(np.mean(gains)) if gains else None,
                        "worst_gain_ratio": float(min(gains)) if gains else None,
                        "mean_proxy_blast_release_m3": float(np.mean(releases)) if releases else None})
    accepted = [row for row in ranking if row["accepted"]]
    # 收益数值等价时优先少释放的方案，不靠未确认的罚金权重打破平局。
    winner = max(accepted, key=lambda row: (round(row["mean_gain_ratio"], 8),
                 -row["mean_proxy_blast_release_m3"]))["candidate"] if accepted else "daily"
    steps, options = candidates[winner]
    return {"chosen": "daily" if steps == C.STEPS_PER_DAY else "rolling_proxy",
            "candidate": winner, "replan_steps": steps, "control_options": options,
            "scope": "first_48h_per_tuning_fold_only", "oof": controls, "ranking": ranking,
            "promotion": "mean>=0.2%, every_fold>=-1%, no_fold_curtailment_or_release_increase, E_drop<=0.001"}


def _resource_oof(prepared: PreparedData, folds: list[dict], quick: bool, num_round: int) -> dict:
    """只在 A/B/C 前缀样本外选择逐气种损失函数，确认窗口不参加选择。"""
    results = {"regression_l1": [], "regression_l2": []}
    caches = []
    for fold in C.progress([f for f in folds if f["role"] == "tuning"],
                            desc="资源开发验证", unit="折"):
        masked = prepared.masked_copy(fold["start"], pd.Timestamp(fold["start"]) + pd.Timedelta(days=fold["days"]))
        positions, eval_positions, ref = _resource_positions(masked, fold, quick)
        truth = resource_series(masked).to_numpy()[eval_positions[:, None] + np.asarray(C.HORIZONS_L)]
        cached = {"fold": fold["name"], "prepared": masked,
                  "origins": masked.index[eval_positions], "predictions": {}, "ref": ref,
                  "truth": truth}
        baseline_models = None
        for objective in C.progress(results, desc=f"资源折 {fold['name']} 损失函数", unit="候选"):
            logging.getLogger("aic-half").info("资源开发折 %s: %s", fold["name"], objective)
            models, names = _fit_resource_models(masked, positions, num_round, ref,
                                                objectives={g: objective for g in C.RESOURCE_NAMES})
            bundle = ResourceBundle(models, {}, names)
            pred = forecast_resources(bundle, masked, masked.index[eval_positions])
            cached["predictions"][objective] = pred
            results[objective].append({"fold": fold["name"], "metrics": score_resources(pred, truth)})
            if objective == "regression_l1":
                baseline_models = models
        head_positions = _resource_head_positions(ref, quick)
        head_models, head_names = _fit_resource_models(masked, head_positions, num_round, ref,
            objectives={g: "regression_l1" for g in C.RESOURCE_NAMES}, horizons=C.HORIZONS_S)
        if head_names != names:
            raise ValueError("短资源头与长期资源特征顺序不一致")
        pool = {**baseline_models, **{gas + "__head": model for gas, model in head_models.items()}}
        head_prediction = forecast_resources(ResourceBundle(pool, {}, names), masked, masked.index[eval_positions])
        cached["head_prediction"] = head_prediction[:, :len(C.HORIZONS_S)]
        cached["head_metrics"] = score_resources(head_prediction, truth)
        cached["head_max_training_target"] = str(masked.index[head_positions[-1] + max(C.HORIZONS_S)])
        caches.append(cached)
    chosen = {}
    for gas in C.RESOURCE_NAMES:
        baseline = [r["metrics"][gas]["wape"] for r in results["regression_l1"]]
        alternative = [r["metrics"][gas]["wape"] for r in results["regression_l2"]]
        valid = all(v is not None for v in baseline + alternative)
        promote = (valid and np.mean(alternative) < np.mean(baseline)
                   and all(a <= b + .02 for a, b in zip(alternative, baseline)))
        chosen[gas] = "regression_l2" if promote else "regression_l1"
    head_gases = []
    for gas in C.RESOURCE_NAMES:
        baseline = [r["metrics"][gas]["groups"]["short"]["wape"] for r in results[chosen[gas]]]
        head = [item["head_metrics"][gas]["groups"]["short"]["wape"] for item in caches]
        if (all(v is not None for v in baseline + head)
                and np.mean(head) <= np.mean(baseline) - .005
                and all(a <= b + .02 for a, b in zip(head, baseline))):
            head_gases.append(gas)
    controls = []
    candidates = {"daily": (C.STEPS_PER_DAY, {}),
                  "hour_clip": (C.OPT_CONFIG["rolling_replan_steps"], {}),
                  "hour_substitute": (C.OPT_CONFIG["rolling_replan_steps"],
                                      {"execution_gas_mode": "substitute"}),
                  "hour_substitute_vent": (C.OPT_CONFIG["rolling_replan_steps"],
                                           {"execution_gas_mode": "substitute_vent"})}
    if head_gases:
        candidates["head_hour_substitute_vent"] = (C.OPT_CONFIG["rolling_replan_steps"],
                                                   {"execution_gas_mode": "substitute_vent"})
        candidates["head_half_hour_substitute_vent"] = (C.OPT_CONFIG["rolling_replan_steps"],
                                                        {"execution_gas_mode": "substitute_vent"})
        candidates["head_half_peak_persist15_substitute_vent"] = (1,
                                                        {"execution_gas_mode": "substitute_vent"})
        candidates["head_half_peak_nn15_substitute_vent"] = (1,
                                                        {"execution_gas_mode": "substitute_vent"})
        candidates["head_half_peak_nn03_15_substitute_vent"] = (1,
                                                        {"execution_gas_mode": "substitute_vent"})
    head_weights = {"head_hour_substitute_vent": 1., "head_half_hour_substitute_vent": .5,
                    "head_half_peak_persist15_substitute_vent": .5,
                    "head_half_peak_nn15_substitute_vent": .5,
                    "head_half_peak_nn03_15_substitute_vent": .5}
    adjustments = {"head_half_peak_persist15_substitute_vent": {
        "method": "positive_persistence_peak", "threshold_ratio": .03, "weight": .5},
        "head_half_peak_nn15_substitute_vent": {
        "method": "positive_nn_peak", "threshold_ratio": .05, "weight": 1.},
        "head_half_peak_nn03_15_substitute_vent": {
        "method": "positive_nn_peak", "threshold_ratio": .03, "weight": .5}}
    adjustment_metrics = {name: [] for name in adjustments}
    nn_training = []
    for item in C.progress(caches, desc="调度策略初筛", unit="折"):
        data, ref = item["prepared"], item["ref"]
        heat, caps = estimate_heat_rate(data, ref), _gas_caps(data, ref)
        frozen = ResourceBundle({}, {}, [], heat_rate=heat, gas_caps=caps)
        if head_gases and heat.get("ok"):
            import main_nn
            base = _resource_base(data)
            pack = main_nn.train_resource_nn(base, resource_series(data), ref, heat, load_prices(),
                                         quick)
            item["nn_forecast"] = main_nn.predict_resource_nn(pack, base, resource_series(data),
                                                               data.index.get_indexer(item["origins"]))
            nn_training.append({"fold": item["fold"], **{k: pack[k] for k in
                ("kind", "epochs", "inner_error", "train_end", "inner_train_end", "max_training_target", "seed")}})
        forecast = np.stack([item["predictions"][chosen[g]][:, :, gi]
                             for gi, g in enumerate(C.RESOURCE_NAMES)], axis=2)
        head_forecast = forecast.copy()
        for gi, gas in enumerate(C.RESOURCE_NAMES):
            if gas in head_gases:
                head_forecast[:, :len(C.HORIZONS_S), gi] = item["head_prediction"][:, :, gi]
        item["selected_forecast"], item["head_forecast"] = forecast, head_forecast
        comparisons = {}
        for name, (interval, options) in C.progress(candidates.items(),
                                                   desc=f"调度折 {item['fold']} 候选", unit="策略"):
            weight = head_weights.get(name, 0.)
            trial = (forecast * (1. - weight) + head_forecast * weight) if weight else forecast
            if name in adjustments:
                before = score_resources(trial, item["truth"])
                trial = _adjust_available_resources(trial, data, item["origins"], heat, adjustments[name],
                    expert=item.get("nn_forecast"))
                after = score_resources(trial, item["truth"])
                adjustment_metrics[name].append({"fold": item["fold"],
                    "before_short_wape": {g: before[g]["groups"]["short"]["wape"] for g in C.RESOURCE_NAMES},
                    "after_short_wape": {g: after[g]["groups"]["short"]["wape"] for g in C.RESOURCE_NAMES}})
            replay = optimize_rolling(trial,
                                      data, item["origins"], frozen,
                                      quick=True, replan_steps=interval, control_options=options)
            comparisons[name] = replay["summary"]
        controls.append({"fold": item["fold"], "comparison": comparisons})
    adjustment_allowed = {}
    for name, rows in adjustment_metrics.items():
        before = [r["before_short_wape"][g] for r in rows for g in C.RESOURCE_NAMES]
        after = [r["after_short_wape"][g] for r in rows for g in C.RESOURCE_NAMES]
        valid = bool(rows) and all(v is not None for v in before + after)
        adjustment_allowed[name] = bool(valid and np.mean(after) < np.mean(before)
            and all(a <= b + .02 for a, b in zip(after, before)))
    candidates = {name: spec for name, spec in candidates.items()
                  if name not in adjustments or adjustment_allowed[name]}
    dispatch = _select_dispatch(controls, candidates)
    confirmation = {}
    if not quick and head_gases:
        # 48小时只做初筛；资源头必须在完整开发折上避免增加已部署策略的降载。
        screen_ranking = dispatch["ranking"]
        raw_candidates = {name: spec for name, spec in candidates.items() if name not in head_weights}
        base_dispatch = _select_dispatch(controls, raw_candidates)
        screened = {row["candidate"] for row in dispatch["ranking"] if row["accepted"]}
        eligible = [name for name in head_weights if name in screened]
        full_controls = []
        base_name = base_dispatch["candidate"]
        references = (["head_half_hour_substitute_vent"]
                      if any(name in adjustments for name in eligible) else [])
        for item in C.progress(caches if eligible else [], desc="调度全窗口核验", unit="折"):
            comparisons = {}
            for name in C.progress(dict.fromkeys(["daily", base_name, *references, *eligible]),
                                    desc=f"调度折 {item['fold']} 全窗口候选", unit="策略"):
                interval, options = candidates[name]
                weight = head_weights.get(name, 0.)
                forecast = item["selected_forecast"]
                trial = (forecast * (1. - weight) + item["head_forecast"] * weight) if weight else forecast
                heat = estimate_heat_rate(item["prepared"], item["ref"])
                if name in adjustments:
                    trial = _adjust_available_resources(trial, item["prepared"], item["origins"],
                                                        heat, adjustments[name], expert=item.get("nn_forecast"))
                replay = optimize_rolling(trial, item["prepared"], item["origins"],
                    ResourceBundle({}, {}, [], heat_rate=heat,
                                   gas_caps=_gas_caps(item["prepared"], item["ref"])),
                    replan_steps=interval, control_options=options)
                comparisons[name] = replay["summary"]
            full_controls.append({"fold": item["fold"], "comparison": comparisons})
        full_candidates = {name: candidates[name] for name in dict.fromkeys(["daily", base_name, *references, *eligible])}
        checked = _select_dispatch(full_controls, full_candidates)
        accepted = []
        for row in checked["ranking"]:
            name = row["candidate"]
            if name not in eligible or not row["accepted"]:
                continue
            gains, safe = [], True
            for item in full_controls:
                baseline, trial = item["comparison"][base_name], item["comparison"][name]
                gains.append(trial["proxy_inventory_adjusted_revenue_cny"] /
                             baseline["proxy_inventory_adjusted_revenue_cny"] - 1.)
                safe &= trial["proxy_curtailed_generation_mwh"] <= baseline["proxy_curtailed_generation_mwh"] + 1e-5
            if name in adjustments:
                # 新修正还必须胜过已经选定的八步50%小时控制，不能增加其降载。
                previous_gains = []
                for item in full_controls:
                    previous = item["comparison"]["head_half_hour_substitute_vent"]
                    trial = item["comparison"][name]
                    previous_gains.append(trial["proxy_inventory_adjusted_revenue_cny"] /
                                          previous["proxy_inventory_adjusted_revenue_cny"] - 1.)
                    safe &= trial["proxy_curtailed_generation_mwh"] <= previous["proxy_curtailed_generation_mwh"] + 1e-5
                safe &= np.mean(previous_gains) >= .002 and min(previous_gains) >= -.01
                row["gain_vs_previous_head_control"] = previous_gains
            row["gain_vs_existing_control"] = gains
            row["head_gate_passed"] = bool(safe and np.mean(gains) >= .002 and min(gains) >= -.01)
            if row["head_gate_passed"]:
                accepted.append(row)
        dispatch = base_dispatch
        if accepted:
            winner = max(accepted, key=lambda row: row["mean_gain_ratio"])["candidate"]
            dispatch = {**dispatch, "candidate": winner, "chosen": "rolling_proxy",
                        "replan_steps": candidates[winner][0], "control_options": candidates[winner][1]}
        dispatch["scope"] = "first_48h_screen_then_full_development_head_gate"
        dispatch["ranking"] = screen_ranking
        confirmation = {"scope": "complete_development_folds_only; no_D_reselection",
                        "oof": full_controls, "ranking": checked["ranking"]}
    head_selection = {"gases": head_gases if dispatch["candidate"].startswith("head_") else [],
        "candidate_gases": head_gases, "horizons": list(C.HORIZONS_S),
        "weight": head_weights.get(dispatch["candidate"], 0.), "confirmation": confirmation,
        "method": "independent_L1_8_steps; short_WAPE_screen_then_joint_dispatch_gate",
        "oof": [{"fold": item["fold"], "metrics": item["head_metrics"],
                 "max_training_target": item["head_max_training_target"]} for item in caches]}
    return {"objectives": chosen, "oof": results, "selection_scope": "tuning_folds_only",
            "metric": "all_finite_wape_including_zero", "max_fold_wape_drop": .02,
            "dispatch_selection": dispatch, "head_selection": head_selection,
            "availability_adjustment": adjustments.get(dispatch["candidate"], {}),
            "availability_adjustment_selection": {"allowed": adjustment_allowed, "oof": adjustment_metrics,
                                                   "nn_training": nn_training}}


def _resource_head_positions(ref_pos: int, quick: bool) -> np.ndarray:
    stride = C.QUICK_TRAIN_STRIDE if quick else C.TRAIN_STRIDE_LONG
    return np.arange(C.PROFILE_DAYS * C.STEPS_PER_DAY,
                     max(ref_pos - max(C.HORIZONS_S) + 1, 0))[::stride]


def train_resources(prepared: PreparedData, folds: list[dict],
                    quick: bool = False) -> ResourceBundle:
    prepared.require_training_data()
    started = time.time()
    confirm = next(f for f in folds if f["role"] == "confirm")
    masked = prepared.masked_copy(
        confirm["start"],
        pd.Timestamp(confirm["start"]) + pd.Timedelta(days=int(confirm["days"])))
    positions, eval_positions, ref_pos = _resource_positions(masked, confirm, quick)
    num_round = C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG
    selection = _resource_oof(prepared, folds, quick, num_round)
    logging.getLogger("aic-half").info("资源头=%s，融合权重=%s，控制候选=%s",
        selection["head_selection"]["gases"], selection["head_selection"]["weight"],
        selection["dispatch_selection"]["candidate"])
    models, names = _fit_resource_models(masked, positions, num_round, ref_pos,
                                        objectives=selection["objectives"])

    final_positions = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY),
                                len(prepared.index) - max(C.HORIZONS_L))
    final_positions = final_positions[::C.QUICK_TRAIN_STRIDE if quick else C.TRAIN_STRIDE_LONG]
    final_ref = len(prepared.index) - 1
    final_models, _ = _fit_resource_models(prepared, final_positions, num_round, final_ref,
                                          objectives=selection["objectives"])
    head_gases = tuple(selection["head_selection"]["gases"])
    if head_gases:
        for pool, data, ref in ((models, masked, ref_pos),
                                (final_models, prepared, len(prepared.index) - 1)):
            head_models, _ = _fit_resource_models(data, _resource_head_positions(ref, quick),
                num_round, ref, objectives={g: "regression_l1" for g in head_gases},
                horizons=C.HORIZONS_S, gases=head_gases)
            pool.update({gas + "__head": model for gas, model in head_models.items()})

    bundle = ResourceBundle(models=models, final_models=final_models,
                            feature_names=names,
                            train_rows=int(len(positions) * len(C.HORIZONS_L)),
                            train_seconds=time.time() - started, selection=selection,
                            dispatch_selection=selection["dispatch_selection"],
                            heat_rate=estimate_heat_rate(prepared, ref_pos),
                            final_heat_rate=estimate_heat_rate(prepared, len(prepared.index) - 1))
    if selection.get("availability_adjustment", {}).get("method") == "positive_nn_peak":
        import main_nn
        bundle.resource_nn = main_nn.train_resource_nn(_resource_base(masked), resource_series(masked),
            ref_pos, bundle.heat_rate, load_prices(), quick)
        bundle.final_resource_nn = main_nn.train_resource_nn(_resource_base(prepared), resource_series(prepared),
            len(prepared.index) - 1, bundle.final_heat_rate, load_prices(), quick)
    origins = prepared.index[eval_positions]
    forecast = forecast_resources(bundle, masked, origins)
    truth = resource_series(masked).to_numpy(dtype=float)[
        np.asarray(eval_positions)[:, None] + np.asarray(C.HORIZONS_L)[None, :]]
    bundle.metrics = score_resources(forecast, truth)

    bundle.gas_caps = _gas_caps(prepared, ref_pos)
    bundle.final_gas_caps = _gas_caps(prepared, len(prepared.index) - 1)
    bundle.train_end = str(prepared.index[ref_pos])
    bundle.final_train_end = str(prepared.index[-1])
    refresh_physical_validation(bundle, prepared)
    refresh_physical_validation(bundle, prepared, use_final=True)
    bundle.train_seconds = time.time() - started
    if not bundle.heat_rate.get("ok"):
        bundle.blocker = "PHYSICAL_MAPPING_UNCERTAIN"
    if C.GAS_HOLDER["column"] not in prepared.frame.columns:
        bundle.blocker = "MISSING_GAS_HOLDER_MAPPING"
    return bundle


def _resource_positions(prepared: PreparedData, fold: dict,
                        quick: bool) -> tuple[np.ndarray, np.ndarray, int]:
    start = pd.Timestamp(fold["start"])
    train_end = pd.Timestamp(fold["train_end"])
    ref_pos = int(prepared.index.get_indexer([train_end])[0])
    positions = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY),
                          max(ref_pos - max(C.HORIZONS_L) + 1, 0))
    positions = positions[::C.QUICK_TRAIN_STRIDE if quick else C.TRAIN_STRIDE_LONG]
    end = start + pd.Timedelta(days=int(fold["days"])) - pd.Timedelta(minutes=C.STEP_MINUTES)
    origins = pd.date_range(start, end, freq=f"{C.STEP_MINUTES}min")
    eval_positions = prepared.index.get_indexer(origins)
    return positions, eval_positions, ref_pos


# ---------------------------------------------------------------------------
# 电价
# ---------------------------------------------------------------------------
def load_prices(config=C) -> dict[int, np.ndarray]:
    path = config.DATA_ROOT / config.OPT_CONFIG["price_file"]
    if not path.is_file():
        return {}
    raw = pd.read_excel(path, header=0)
    time_col = raw.columns[0]
    month_cols = {}
    for col in raw.columns[1:]:
        match = re.match(r"(\d+)月", str(col).strip())
        if match:
            month_cols[int(match.group(1))] = col
    prices = {}
    for month, col in month_cols.items():
        slots = np.full(96, np.nan)
        for _, row in raw.iterrows():
            text = str(row[time_col]).strip()
            start = text.split("-")[0]
            match = re.match(r"(\d+):(\d+)", start)
            if not match:
                continue
            slot = int(match.group(1)) * 4 + int(match.group(2)) // 15
            value = pd.to_numeric(row[col], errors="coerce")
            if np.isfinite(value):
                for k in range(2):
                    if 0 <= slot + k < 96:
                        slots[slot + k] = float(value)
        if np.isfinite(slots).all():
            prices[month] = slots
    return prices


def _price_vector(prices: dict[int, np.ndarray], start: pd.Timestamp, n: int) -> np.ndarray:
    times = pd.date_range(start, periods=n, freq=f"{C.STEP_MINUTES}min")
    values = []
    for t in times:
        if t.month not in prices:
            raise ValueError(f"缺少 {t.month} 月官方电价，禁止补零或沿用上月")
        value = prices[t.month][t.hour * 4 + t.minute // 15]
        if not np.isfinite(value):
            raise ValueError(f"电价无效: {t}")
        values.append(float(value))
    return np.asarray(values)


def _tou_periods(price: np.ndarray, config=C) -> list[str]:
    """按当日电价分档为谷/平/峰/尖；电价表只有两档时自动退化。"""
    levels = sorted(set(np.round(price[np.isfinite(price)], 6).tolist()))
    if not levels:
        return ["flat"] * len(price)
    labels = []
    n = len(levels)
    for value in np.round(price, 6):
        idx = levels.index(value) if value in levels else 0
        if n == 1:
            labels.append("flat")
        elif n == 2:
            labels.append("valley" if idx == 0 else "peak")
        elif n == 3:
            labels.append(("valley", "flat", "peak")[idx])
        else:
            # 四档及以上：最低谷、最高尖、中间对分平/峰。
            labels.append(("valley", "flat", "peak", "sharp")[min(idx, 3)])
    return labels


# ---------------------------------------------------------------------------
# MILP
# ---------------------------------------------------------------------------
def _last_level(prepared: PreparedData, target: str, pos: int) -> float:
    values = prepared.frame[C.TARGET_COLUMN[target]].iloc[:pos + 1].to_numpy(dtype=float)
    valid = np.where(np.isfinite(values))[0]
    return float(values[valid[-1]]) if len(valid) else float("nan")


def _holder_plan_bounds(n: int, config=C) -> tuple:
    capacity = config.GAS_HOLDER["capacity_m3"]
    low = np.full(n, capacity * (config.GAS_HOLDER["safe_low_ratio"] + config.OPT_CONFIG["holder_margin_ratio"]))
    high = np.full(n, capacity * (config.GAS_HOLDER["safe_high_ratio"] - config.OPT_CONFIG["holder_margin_ratio"]))
    return low, high


def _aggregate_ramp_groups(config=C):
    """按提交中的小机组聚合负荷和总负荷检查启停，避免单机最小负荷锁死开机。"""
    units = np.asarray(config.OPT_CONFIG["unit_capacity_mw"], dtype=float)
    if config.OPT_CONFIG["ramp_scope"] != "aggregate_generator_1_and_all":
        raise ValueError("未实现的聚合爬坡口径")
    fraction = config.OPT_CONFIG["ramp_fraction_per_min"] * config.STEP_MINUTES
    return [(np.arange(4), float(units[:4].sum() * fraction)),
            (np.arange(len(units)), float(units.sum() * fraction))]


def _allocate_aggregate_power(budget: float, previous: np.ndarray, config=C,
                              requested_power: np.ndarray | None = None):
    """回退与事后执行共用组合分配，硬约束冲突时返回失败，不放宽爬坡。"""
    units = np.asarray(config.OPT_CONFIG["unit_capacity_mw"], dtype=float)
    groups = _aggregate_ramp_groups(config)
    first_limit, total_limit = groups[0][1], groups[1][1]
    first_previous, total_previous = float(previous[:4].sum()), float(previous.sum())
    best = None
    for mask in range(1 << len(units)):
        on = np.array([(mask >> i) & 1 for i in range(len(units))], dtype=bool)
        lower = np.where(on, units * config.OPT_CONFIG["min_load_ratio"], 0.)
        upper = np.where(on, units, 0.)
        first_low = max(float(lower[:4].sum()), first_previous - first_limit)
        first_high = min(float(upper[:4].sum()), first_previous + first_limit)
        total_low = max(first_low + float(lower[4:].sum()), total_previous - total_limit)
        target = min(float(budget), first_high + float(upper[4:].sum()), total_previous + total_limit)
        if first_low > first_high + 1e-8 or total_low > target + 1e-8:
            continue
        feasible_first_low = max(first_low, target - float(upper[4:].sum()))
        feasible_first_high = min(first_high, target - float(lower[4:].sum()))
        if feasible_first_high < feasible_first_low - 1e-8:
            continue
        # 资源充足时保持已发布聚合负荷，避免重分配造成下一步爬坡冲突。
        wanted_first = (float(np.asarray(requested_power)[:4].sum()) if requested_power is not None
                        else feasible_first_high)
        first = min(max(wanted_first, feasible_first_low), feasible_first_high)
        power = lower.copy()
        for indices, wanted in ((range(4), first), (range(4, len(units)), target - first)):
            remaining = wanted - float(power[list(indices)].sum())
            for i in indices:
                extra = min(max(remaining, 0.), upper[i] - power[i])
                power[i] += extra
                remaining -= extra
        deviation = abs(first - wanted_first) if requested_power is not None else 0.
        key = (round(target, 6), -round(deviation, 6), -int(np.count_nonzero(on != (previous > 1e-5))))
        if best is None or key > best[0]:
            best = (key, power)
    return None if best is None else best[1]


def _solve_day(avail: np.ndarray, price: np.ndarray, holder0: float,
               initial_power: np.ndarray, caps: np.ndarray, heat_rate: dict,
               config=C) -> dict:
    """代理包络下的确定性 MILP；加权收益目标不等于真实气网可行性。"""
    n_h = len(price)
    units = config.OPT_CONFIG["unit_capacity_mw"]
    n_u = len(units)
    n_g = len(C.RESOURCE_NAMES)
    beta = np.array([heat_rate["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    capacity = config.GAS_HOLDER["capacity_m3"]
    real_low = capacity * config.GAS_HOLDER["safe_low_ratio"]
    real_high = capacity * config.GAS_HOLDER["safe_high_ratio"]
    margin = capacity * float(config.OPT_CONFIG["holder_margin_ratio"])
    low = real_low + margin
    high = real_high - margin
    lower_path, upper_path = _holder_plan_bounds(n_h, config)
    if (lower_path > upper_path).any():
        return {"status": "INFEASIBLE", "message": "计划安全边界无可行域"}
    if (not np.isfinite(avail).all() or not np.isfinite(price).all()
            or (avail < 0).any() or not real_low <= holder0 <= real_high):
        return {"status": "INVALID_INPUT", "message": "资源或初始柜位无效，禁止裁剪库存伪造可行性"}

    off_z, off_p = 0, n_u * n_h
    off_q = off_p + n_u * n_h
    off_v = off_q + n_g * n_h
    off_s = off_v + n_g * n_h
    # GPF：柜位对偏好带的双向偏离量，压在目标里软惩罚。
    off_du = off_s + n_h
    off_dd = off_du + n_h
    # P4：每机组每步的启停切换指示（连续 [0,1] 即可，取到 0/1 由最小化目标保证）。
    off_sw = off_dd + n_h
    n_var = off_sw + n_u * n_h
    lb = np.zeros(n_var)
    ub = np.full(n_var, np.inf)
    integrality = np.zeros(n_var)
    for i in range(n_u):
        for k in range(n_h):
            ub[off_z + i * n_h + k] = 1.0
            integrality[off_z + i * n_h + k] = 1
            ub[off_p + i * n_h + k] = units[i]
    for g in range(n_g):
        for k in range(n_h):
            ub[off_q + g * n_h + k] = caps[g]
    for k in range(n_h):
        lb[off_s + k] = lower_path[k]
        ub[off_s + k] = upper_path[k]
        # 偏离量上界取安全区宽度，足够覆盖极端柜位。
        ub[off_du + k] = capacity
        ub[off_dd + k] = capacity
    for i in range(n_u):
        for k in range(n_h):
            ub[off_sw + i * n_h + k] = 1.0

    rows: list[tuple[int, dict[int, float], float, float]] = []

    def add(coeffs: dict[int, float], lower: float, upper: float) -> None:
        rows.append((len(rows), coeffs, lower, upper))

    for i in range(n_u):
        cap = units[i]
        for k in range(n_h):
            add({off_p + i * n_h + k: 1.0, off_z + i * n_h + k: -cap}, -np.inf, 0.0)
            add({off_p + i * n_h + k: -1.0,
                 off_z + i * n_h + k: config.OPT_CONFIG["min_load_ratio"] * cap}, -np.inf, 0.0)
    # 仅在一步可跨额定容量时消除等价排列；较紧爬坡保留与初始状态匹配的组合。
    if config.OPT_CONFIG["ramp_fraction_per_min"] * config.STEP_MINUTES >= 1.0:
        for i in (0, 1, 2, 4):
            if units[i] == units[i + 1]:
                for k in range(n_h):
                    add({off_z + i * n_h + k: 1.0,
                         off_z + (i + 1) * n_h + k: -1.0}, 0.0, np.inf)
    for group, cap in _aggregate_ramp_groups(config):
        for k in range(n_h):
            delta = {off_p + int(i) * n_h + k: 1.0 for i in group}
            if k > 0:
                delta.update({off_p + int(i) * n_h + k - 1: -1.0 for i in group})
                add(delta, -cap, cap)
            else:
                prev = float(np.asarray(initial_power)[group].sum())
                add(delta, prev - cap, prev + cap)
    for k in range(n_h):
        coeffs = {off_s + k: 1.0}
        rhs = 0.25 * float(avail[k, 0])
        if k > 0:
            coeffs[off_s + k - 1] = -1.0
        else:
            rhs += holder0
        coeffs[off_q + 0 * n_h + k] = 0.25
        coeffs[off_v + 0 * n_h + k] = 0.25
        add(coeffs, rhs, rhs)
        # 气柜可提供的最大瞬时流量：不得超过当前库存高于下限的部分。
        draw = {off_q + 0 * n_h + k: 1.0, off_v + 0 * n_h + k: 1.0}
        if k > 0:
            draw[off_s + k - 1] = -4.0
            add(draw, -np.inf, float(avail[k, 0]) - 4.0 * lower_path[k])
        else:
            add(draw, -np.inf, float(avail[k, 0]) - 4.0 * lower_path[k] + 4.0 * holder0)
    for g in (1, 2):
        for k in range(n_h):
            add({off_q + g * n_h + k: 1.0, off_v + g * n_h + k: 1.0},
                -np.inf, float(avail[k, g]))
    pref_low = capacity * float(config.OPT_CONFIG["holder_pref_low_ratio"])
    pref_high = capacity * float(config.OPT_CONFIG["holder_pref_high_ratio"])
    for k in range(n_h):
        # s[k] + d_under[k] >= pref_low
        add({off_s + k: 1.0, off_dd + k: 1.0}, pref_low, np.inf)
        # s[k] - d_over[k] <= pref_high
        add({off_s + k: 1.0, off_du + k: -1.0}, -np.inf, pref_high)
    for i in range(n_u):
        for k in range(n_h):
            if k == 0:
                # 连续重规划也要计首步切换，否则每次换窗都能免费启停。
                initial_on = float(initial_power[i] > config.OPT_CONFIG["power_tolerance_mw"])
                add({off_sw + i * n_h: 1.0, off_z + i * n_h: -1.0}, -initial_on, np.inf)
                add({off_sw + i * n_h: 1.0, off_z + i * n_h: 1.0}, initial_on, np.inf)
                continue
            zk = off_z + i * n_h + k
            zk1 = off_z + i * n_h + k - 1
            sw = off_sw + i * n_h + k
            add({sw: 1.0, zk: -1.0, zk1: 1.0}, 0.0, np.inf)   # sw >= z_{k-1}-z_k
            add({sw: 1.0, zk: 1.0, zk1: -1.0}, 0.0, np.inf)   # sw >= z_k-z_{k-1}
    if config.OPT_CONFIG.get("terminal_holder") and low <= holder0 <= high:
        terminal_low = max(lower_path[-1], holder0)
        terminal_high = min(upper_path[-1], holder0)
        if terminal_low > terminal_high:
            return {"status": "INFEASIBLE", "message": "安全边界与末端库存约束冲突"}
        add({off_s + n_h - 1: 1.0}, terminal_low, terminal_high)
    for k in range(n_h):
        coeffs = {off_p + i * n_h + k: 1.0 for i in range(n_u)}
        for g in range(n_g):
            coeffs[off_q + g * n_h + k] = -float(beta[g])
        add(coeffs, 0.0, 0.0)

    matrix = lil_matrix((len(rows), n_var))
    lower = np.empty(len(rows))
    upper = np.empty(len(rows))
    for idx, coeffs, lo, hi in rows:
        for var, value in coeffs.items():
            matrix[idx, var] = value
        lower[idx] = lo
        upper[idx] = hi
    matrix = csr_matrix(matrix)
    constraints = [LinearConstraint(matrix, lower, upper)]
    options = {"time_limit": float(config.OPT_CONFIG["solver_time_limit_s"]),
               "mip_rel_gap": float(config.OPT_CONFIG["mip_rel_gap"]),
               "disp": False}

    objective = np.zeros(n_var)
    for k in range(n_h):
        for i in range(n_u):
            objective[off_p + i * n_h + k] = -float(price[k]) * 0.25 * 1000.0
    # 放散罚金：气种×分时差异化（Zhao 2015）。
    tou = _tou_periods(price, config)
    table = config.OPT_CONFIG["vent_penalty_table"]
    for g, gas_name in enumerate(C.RESOURCE_NAMES):
        base = float(table.get(gas_name, {}).get(
            "flat", config.OPT_CONFIG["vent_penalty_cny_per_m3"]))
        for k in range(n_h):
            pen = float(table.get(gas_name, {}).get(tou[k], base))
            objective[off_v + g * n_h + k] = 0.25 * pen
    objective[off_q:off_v] = 1e-8
    # GPF：柜位贴偏好带，避免顶到安全边界。
    hold_pen = float(config.OPT_CONFIG["holder_penalty_cny_per_m3"])
    objective[off_du:off_dd] = 0.25 * hold_pen
    objective[off_dd:off_sw] = 0.25 * hold_pen
    switch_pen = float(config.OPT_CONFIG.get("unit_switch_penalty_cny", 0.0))
    objective[off_sw:n_var] = switch_pen
    solution = milp(objective, integrality=integrality, bounds=Bounds(lb, ub),
                    constraints=constraints, options=options)
    if solution.x is None or solution.status not in (0, 1):
        status = {1: "TIMEOUT_NO_SOLUTION", 2: "INFEASIBLE", 3: "UNBOUNDED"}.get(
            solution.status, "SOLVER_FAILED")
        return {"status": status, "message": str(solution.message)}
    x = solution.x
    power = np.maximum(x[off_p:off_q].reshape(n_u, n_h), 0.0)
    gas = np.maximum(x[off_q:off_v].reshape(n_g, n_h), 0.0)
    vent = np.maximum(x[off_v:off_s].reshape(n_g, n_h), 0.0)
    holder = x[off_s:off_s + n_h]
    vent_total = float(np.sum(vent) * 0.25)
    holder_dev = (np.maximum(x[off_du:off_dd], 0.0)
                  + np.maximum(x[off_dd:off_sw], 0.0))
    result = {"status": "OPTIMAL" if solution.success else "FEASIBLE_TIMEOUT",
            "message": str(solution.message), "power": power, "gas": gas,
            "vent": vent, "holder": holder, "vent_total": vent_total,
            "holder_dev": holder_dev,
            "objective": float(solution.fun),
            "mip_gap": getattr(solution, "mip_gap", None)}
    result["validation"] = validate_day(result, avail, holder0, initial_power, caps, heat_rate, config)
    if not result["validation"]["passed"]:
        return {"status": "INVALID_SOLUTION", "message": str(result["validation"])}
    return result


def validate_day(result: dict, avail: np.ndarray, holder0: float,
                 initial_power: np.ndarray, caps: np.ndarray, heat_rate: dict,
                 config=C) -> dict:
    """求解与回退共用硬约束核验；结论仅限声明的耗气包络代理模型。"""
    power, gas, vent, holder = (np.asarray(result[k], dtype=float)
                               for k in ("power", "gas", "vent", "holder"))
    units = np.asarray(config.OPT_CONFIG["unit_capacity_mw"], dtype=float)
    n = len(avail)
    if (power.shape != (len(units), n) or gas.shape != (3, n)
            or vent.shape != (3, n) or holder.shape != (n,)
            or not all(np.isfinite(x).all() for x in (power, gas, vent, holder))):
        return {"passed": False, "problems": ["形状错误或存在非有限值"]}
    pt = config.OPT_CONFIG["power_tolerance_mw"]
    gt = config.OPT_CONFIG["gas_tolerance_m3h"]
    st = config.OPT_CONFIG["holder_tolerance_m3"]
    capacity = config.GAS_HOLDER["capacity_m3"]
    margin = capacity * config.OPT_CONFIG["holder_margin_ratio"]
    low = capacity * config.GAS_HOLDER["safe_low_ratio"] + margin
    high = capacity * config.GAS_HOLDER["safe_high_ratio"] - margin
    lower_path, upper_path = _holder_plan_bounds(n, config)
    beta = np.asarray([heat_rate["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    previous = np.r_[holder0, holder[:-1]]
    balance = holder - previous - 0.25 * (avail[:, 0] - gas[0] - vent[0])
    energy = power.sum(axis=0) - beta @ gas
    delta = np.diff(np.column_stack([initial_power, power]), axis=1)
    problems = []
    if (power < -pt).any() or (power > units[:, None] + pt).any():
        problems.append("功率越界")
    if ((power > pt) & (power < units[:, None] * config.OPT_CONFIG["min_load_ratio"] - pt)).any():
        problems.append("开机功率低于最小负荷")
    if any((np.abs(delta[group].sum(axis=0)) > cap + pt).any()
           for group, cap in _aggregate_ramp_groups(config)):
        problems.append("爬坡越界")
    if (gas < -gt).any() or (vent < -gt).any() or (gas > caps[:, None] + gt).any():
        problems.append("耗气或代理释放越界")
    if (gas[1:] + vent[1:] > avail[:, 1:].T + gt).any():
        problems.append("无柜气种超出资源包络")
    if np.max(np.abs(energy)) > pt:
        problems.append("燃料与功率不守恒")
    if (lower_path > upper_path).any() or np.max(np.abs(balance)) > st or (holder < lower_path - st).any() or (holder > upper_path + st).any():
        problems.append("代理库存平衡或安全边界违规")
    if (config.OPT_CONFIG["terminal_holder"] and low <= holder0 <= high
            and abs(holder[-1] - holder0) > st):
        problems.append("末端库存不一致")
    return {"passed": not problems, "problems": problems,
            "power_balance_max_mw": float(np.max(np.abs(energy))),
            "holder_balance_max_m3": float(np.max(np.abs(balance))),
            "scope": "historical_consumption_envelope_only"}


def _fallback_day(prepared: PreparedData, pos: int, avail: np.ndarray,
                  caps: np.ndarray, heat_rate: dict, config=C,
                  holder0: float | None = None, initial_power: np.ndarray | None = None) -> dict:
    """枚举六机组组合，按资源降载；剩余高炉包络释放以保持代理库存。"""
    units = np.asarray(config.OPT_CONFIG["unit_capacity_mw"], dtype=float)
    beta = np.asarray([heat_rate["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    if holder0 is None:
        history = prepared.frame[config.GAS_HOLDER["column"]].iloc[:pos + 1].dropna()
        if history.empty:
            return {"status": "BLOCKED_OPT", "message": "回退缺少初始柜位"}
        holder0 = float(history.iloc[-1])
    capacity = config.GAS_HOLDER["capacity_m3"]
    lower_path, upper_path = _holder_plan_bounds(len(avail), config)
    if (lower_path > upper_path).any():
        return {"status": "BLOCKED_OPT", "message": "风险裕度不可行，不能回退时放宽边界"}
    if not capacity * config.GAS_HOLDER["safe_low_ratio"] <= holder0 <= capacity * config.GAS_HOLDER["safe_high_ratio"]:
        return {"status": "BLOCKED_OPT", "message": "回退初始柜位越界"}
    initial_power = np.zeros(len(units)) if initial_power is None else np.asarray(initial_power)
    previous = initial_power.copy()
    power, gas, vent = np.zeros((6, len(avail))), np.zeros((3, len(avail))), np.zeros((3, len(avail)))
    holder = np.empty(len(avail))
    level = holder0
    for k, supply in enumerate(avail):
        available = np.minimum(supply, caps).copy()
        target_level = float(np.clip(level, lower_path[k], upper_path[k]))
        available[0] = min(available[0], max(0.0, supply[0] - (target_level - level) / 0.25))
        budget = float(beta @ available)
        p = _allocate_aggregate_power(budget, previous, config)
        if p is None:
            return {"status": "BLOCKED_OPT", "message": "资源不足且无合法降载组合"}
        target = float(p.sum())
        gas[:, k] = available * (target / budget if budget > 0 else 0.0)
        vent[0, k] = supply[0] - gas[0, k] - (target_level - level) / 0.25
        if vent[0, k] < -config.OPT_CONFIG["gas_tolerance_m3h"]:
            return {"status": "BLOCKED_OPT", "message": "资源不足以恢复代理柜位"}
        vent[0, k] = max(0.0, vent[0, k])
        level += 0.25 * (supply[0] - gas[0, k] - vent[0, k])
        power[:, k], holder[k], previous = p, level, p.copy()
    result = {"status": "FALLBACK_SAFE", "message": "MILP 降级为核验通过的资源约束规则分配",
              "power": power, "gas": gas, "vent": vent, "holder": holder,
              "objective": None}
    result["validation"] = validate_day(result, avail, holder0, initial_power, caps, heat_rate, config)
    if not result["validation"]["passed"]:
        return {"status": "BLOCKED_OPT", "message": str(result["validation"])}
    return result


def _execute_proxy_step(request_power: np.ndarray, request_gas: np.ndarray,
                        actual: np.ndarray, holder: float, previous: np.ndarray,
                        caps: np.ndarray, heat_rate: dict, config=C) -> dict:
    """到达目标时刻后按已观察到的代理包络降载，禁止把该信息提前交给规划器。"""
    units = np.asarray(config.OPT_CONFIG["unit_capacity_mw"])
    beta = np.array([heat_rate["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    low = config.GAS_HOLDER["capacity_m3"] * config.GAS_HOLDER["safe_low_ratio"]
    high = config.GAS_HOLDER["capacity_m3"] * config.GAS_HOLDER["safe_high_ratio"]
    if not np.isfinite(actual).all() or (actual < 0).any() or not low <= holder <= high:
        raise ValueError("执行回放的气量或连续柜位无效")
    available = actual.copy()
    available[0] += max(holder - low, 0) * 4.0
    execution_mode = config.OPT_CONFIG.get("execution_gas_mode", "clip")
    substitute = execution_mode in ("substitute", "substitute_vent")
    usable = np.minimum(available if substitute else np.minimum(request_gas, available), caps)
    budget = min(float(beta @ usable), float(np.sum(request_power)))
    power = _allocate_aggregate_power(budget, previous, config, requested_power=request_power)
    if power is None:
        raise ValueError("气量不足且执行层没有满足爬坡与最小负荷的降载组合")
    target = float(power.sum())
    if substitute:
        # 先保留可执行的原配比；缺气后仅利用当前已观察的其他气种补偿。
        gas = np.minimum(request_gas, usable).copy()
        energy = float(beta @ gas)
        if energy > target:
            gas *= target / energy
        else:
            remaining = target - energy
            overflow_flow = max(actual[0] + 4. * (holder - high), 0.)
            extra = min(max(min(overflow_flow, usable[0]) - gas[0], 0.), remaining / beta[0])
            gas[0] += extra
            remaining -= extra * beta[0]
            for g in (1, 2, 0):
                extra = min(max(usable[g] - gas[g], 0.), max(remaining, 0.) / beta[g])
                gas[g] += extra
                remaining -= extra * beta[g]
    else:
        energy = float(beta @ usable)
        gas = usable * (target / energy if energy > 0 else 0)
    if execution_mode == "substitute_vent":
        # 仅置换原本必须释放的高炉气，保持功率和释放后的柜位不变。
        # 无柜气种仍是供气上界，未用量另记，不能称为真实总放散减少。
        overflow_flow = max(actual[0] + 4. * (holder - high) - gas[0], 0.)
        other_energy = float(beta[1:] @ gas[1:])
        replacement = min(overflow_flow * beta[0], other_energy,
                          max(usable[0] - gas[0], 0.) * beta[0])
        if replacement > 0. and other_energy > 0.:
            gas[0] += replacement / beta[0]
            gas[1:] *= max(1. - replacement / other_energy, 0.)
    level = holder + .25 * (actual[0] - gas[0])
    vent = max(level - high, 0) * 4.0
    level -= .25 * vent
    if level < low - 1e-6 or abs(power.sum() - beta @ gas) > 1e-5:
        raise ValueError("执行层能量或库存校验失败")
    # 仅消除守恒计算产生的浮点尾差，避免下一步把 29999.999999999996 判成真实越界。
    level = max(low, min(high, level))
    return {"power": power, "gas": gas, "holder": level, "vent_blast_m3h": vent,
            "shortfall_mwh": max(float(request_power.sum() - target), 0) * .25}


def _dispatch_metrics(times: pd.DatetimeIndex, power: np.ndarray,
                      prices: dict[int, np.ndarray]) -> dict:
    """按自然日平均电价统计利用效率，残缺窗口不补造负荷或标签。"""
    power = np.asarray(power, dtype=float)
    times = pd.DatetimeIndex(times)
    if power.shape != (len(times),) or not np.isfinite(power).all():
        return {"complete": False, "generation_mwh": None,
                "generation_revenue_cny": None, "price_weighted_load_ratio": None}
    price = _price_vector(prices, times[0], len(times))
    mean_price = np.array([float(prices[t.month].mean()) for t in times])
    denominator = float(mean_price @ power)
    daily = []
    for day in times.normalize().unique():
        pick = times.normalize() == day
        denom = float(mean_price[pick] @ power[pick])
        daily.append({"day": str(day.date()), "steps": int(pick.sum()),
                      "generation_mwh": float(power[pick].sum() * .25),
                      "price_weighted_load_ratio": float(price[pick] @ power[pick] / denom)
                      if denom > 0 else None})
    return {"complete": True, "generation_mwh": float(power.sum() * .25),
            "generation_revenue_cny": float(price @ power * .25 * 1000.),
            "price_weighted_load_ratio": float(price @ power / denominator)
            if denominator > 0 else None, "daily": daily}


def optimize_rolling(resource_forecast: np.ndarray, prepared: PreparedData,
                     origins: pd.DatetimeIndex, resource_bundle: ResourceBundle,
                     quick: bool = False, use_final: bool = False,
                     replan_steps: int | None = None,
                     control_options: dict | None = None,
                     align_long_horizon: bool = False) -> dict:
    """连续柜位的代理 MPC；每小时重排 96 步，仅执行前四步。

    预测器只观察起报点及之前的历史包络；执行降载在目标时刻观察到资源之后发生。
    输出表保存提前做出的指令，执行收益单独作为历史代理回放诊断。
    """
    heat = resource_bundle.final_heat_rate if use_final else resource_bundle.heat_rate
    cap_info = resource_bundle.final_gas_caps if use_final else resource_bundle.gas_caps
    if not heat.get("ok"):
        return {"status": "BLOCKED_OPT", "schedule": None,
                "summary": {"reason": "PHYSICAL_MAPPING_UNCERTAIN"}}
    if resource_forecast.shape != (len(origins), len(C.HORIZONS_L), len(C.RESOURCE_NAMES)):
        raise ValueError("滚动资源预测维度错误")
    positions = prepared.index.get_indexer(origins)
    if (positions < 0).any() or len(positions) == 0 or (np.diff(positions) != 1).any():
        raise ValueError("滚动起报点必须连续且位于输入网格")
    steps = int(replan_steps or C.OPT_CONFIG["rolling_replan_steps"])
    if not 1 <= steps <= len(C.HORIZONS_L):
        raise ValueError("重规划间隔必须在 1~96 步内")
    cfg = SimpleNamespace(**vars(C))
    options = control_options or {}
    allowed = {"execution_gas_mode", "holder_penalty_cny_per_m3",
               "unit_switch_penalty_cny", "mip_rel_gap"}
    if set(options) - allowed:
        raise ValueError("不支持的控制参数: " + str(sorted(set(options) - allowed)))
    if options.get("execution_gas_mode", "clip") not in ("clip", "substitute", "substitute_vent"):
        raise ValueError("执行气种策略无效")
    cfg.OPT_CONFIG = {**C.OPT_CONFIG, **options,
                      "solver_time_limit_s": C.OPT_CONFIG["rolling_solver_time_limit_s"]}
    caps = np.array([cap_info.get(g, np.nan) for g in C.RESOURCE_NAMES])
    prices = load_prices()
    if not prices or not np.isfinite(caps).all():
        return {"status": "BLOCKED_OPT", "schedule": None,
                "summary": {"reason": "PRICE_OR_CAPS_UNAVAILABLE"}}
    pos0 = int(positions[0])
    observed = prepared.frame[C.GAS_HOLDER["column"]].iloc[:pos0 + 1].dropna()
    if observed.empty:
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": "NO_INITIAL_HOLDER"}}
    holder = holder0 = float(observed.iloc[-1])
    g1, total = _last_level(prepared, "generator_1", pos0), _last_level(prepared, "generator_all", pos0)
    if not np.isfinite([g1, total]).all():
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": "NO_INITIAL_POWER"}}
    previous = np.array([g1 / 4] * 4 + [max(total - g1, 0) / 2] * 2)
    actual = resource_series(prepared).to_numpy(dtype=float)
    limit = min(len(origins), 2 * C.STEPS_PER_DAY) if quick else len(origins)
    records, statuses, validations = [], [], []
    executed_power, unused = [], []
    gaps = []
    revenue, shortfall, vent_volume, realized_steps = 0., 0., 0., 0
    holder_min, holder_max = holder, holder
    started = time.time()
    rows = ([*range(0, limit - 1, steps), limit - 1] if align_long_horizon
            else range(0, limit, steps))
    for row in C.progress(rows,
                          desc=f"滚动调度 · 每 {steps * C.STEP_MINUTES} 分钟重规划",
                          unit="次"):
        origin = origins[row]
        avail = np.maximum(resource_forecast[row], 0)
        price = _price_vector(prices, origin + pd.Timedelta(minutes=C.STEP_MINUTES), len(C.HORIZONS_L))
        result = _solve_day(avail, price, holder, previous, caps, heat, cfg)
        if result["status"] not in ("OPTIMAL", "FEASIBLE_TIMEOUT"):
            result = _fallback_day(prepared, int(positions[row]), avail, caps, heat, cfg,
                                   holder0=holder, initial_power=previous)
        if result["status"] not in ("OPTIMAL", "FEASIBLE_TIMEOUT", "FALLBACK_SAFE"):
            return {"status": "BLOCKED_OPT", "schedule": None,
                    "summary": {"reason": result.get("message"), "failed_origin": str(origin)}}
        statuses.append(result["status"])
        if result.get("mip_gap") is not None:
            gaps.append(float(result["mip_gap"]))
        validations.append(result["validation"]["passed"])
        commit = (len(C.HORIZONS_L) if row == limit - 1 else min(steps, limit - 1 - row)) \
            if align_long_horizon else min(steps, limit - row)
        for k in range(commit):
            target = origin + pd.Timedelta(minutes=C.STEP_MINUTES * (k + 1))
            power, gas = result["power"][:, k], result["gas"][:, k]
            if records:
                prior_first, prior_total = records[-1][-2:]
                limits = _aggregate_ramp_groups(cfg)
                if (abs(float(power[:4].sum()) - prior_first) > limits[0][1] + cfg.OPT_CONFIG["power_tolerance_mw"]
                        or abs(float(power.sum()) - prior_total) > limits[1][1] + cfg.OPT_CONFIG["power_tolerance_mw"]):
                    return {"status": "BLOCKED_OPT", "schedule": None,
                        "summary": {"reason": "连续发布指令与执行反馈之间发生聚合爬坡冲突",
                                    "failed_origin": str(origin), "failed_target": str(target)}}
            records.append([target, *gas, power[:4].sum(), power.sum()])
            target_pos = int(positions[row]) + k + 1
            if target_pos < len(actual) and np.isfinite(actual[target_pos]).all():
                try:
                    executed = _execute_proxy_step(power, gas, actual[target_pos], holder, previous, caps, heat, cfg)
                except ValueError as exc:
                    return {"status": "BLOCKED_OPT", "schedule": None,
                        "summary": {"reason": str(exc), "failed_origin": str(origin),
                                    "failed_target": str(target), "constraint_conflict": True}}
                holder, previous = executed["holder"], executed["power"]
                revenue += price[k] * previous.sum() * .25 * 1000
                shortfall += executed["shortfall_mwh"]
                vent_volume += executed["vent_blast_m3h"] * .25
                realized_steps += 1
                executed_power.append(float(previous.sum()))
                unused.append(np.maximum(actual[target_pos, 1:] - executed["gas"][1:], 0.) * .25)
            else:
                # 无已发布资源时只接续预测计划，不能制造实测执行收益。
                holder, previous = float(result["holder"][k]), power.copy()
                executed_power.append(np.nan)
                unused.append(np.full(2, np.nan))
            holder_min, holder_max = min(holder_min, holder), max(holder_max, holder)
        if row % C.STEPS_PER_DAY == 0:
            logging.getLogger("aic-half").info("连续包络回放 %s，重规划间隔=%d步", origin, steps)
    schedule = pd.DataFrame(records, columns=["datetime", "opt_generator_use_blast_furnace_gas",
        "opt_generator_use_coke_gas", "opt_generator_use_converter_gas", "opt_generator_1", "opt_generator_all"])
    complete = realized_steps == len(schedule)
    prices_used = _price_vector(prices, schedule.datetime.iloc[0], len(schedule))
    planned_revenue = float(np.sum(prices_used * schedule.opt_generator_all) * .25 * 1000)
    # 防止以耗尽库存冒充经济收益，代理末端存气按平均电价折算单列。
    inventory_adjustment = ((holder - holder0) * heat["coefficients_mw_per_m3h"]["blast"]
                            * 1000 * float(prices_used.mean()))
    times = pd.DatetimeIndex(schedule.datetime)
    planned_metrics = _dispatch_metrics(times, schedule.opt_generator_all.to_numpy(), prices)
    executed_metrics = _dispatch_metrics(times, np.asarray(executed_power), prices)
    historical_metrics = _dispatch_metrics(times,
        prepared.labels["generator_all"].reindex(times).to_numpy(dtype=float), prices)
    historical_revenue = historical_metrics["generation_revenue_cny"]
    historical_gas = resource_series(prepared).reindex(times).to_numpy(dtype=float)
    beta = np.array([heat["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    converted_metrics = _dispatch_metrics(times, historical_gas @ beta, prices)
    converted_revenue = converted_metrics["generation_revenue_cny"]
    # 包络溢出只是代理释放；无柜气种的未用上界不是实际放散，不能一并计罚。
    penalty_sensitivity = []
    for penalty in C.OPT_CONFIG.get("evaluation_release_penalties", (0., .5, 1.)):
        cost = float(penalty) * vent_volume if complete else None
        penalty_sensitivity.append({"assumed_cny_per_m3": float(penalty),
            "proxy_blast_release_penalty_cny": cost,
            "proxy_net_benefit_cny": revenue - cost if complete else None,
            "proxy_inventory_adjusted_net_benefit_cny": revenue - cost + inventory_adjustment
            if complete else None})
    summary = {"status": "SIMULATED_NOT_SUBMITTABLE", "resource_mode": "historical_consumption_envelope",
        "availability_mode": "point", "dispatch_mode": "daily_proxy" if steps == C.STEPS_PER_DAY else "rolling_proxy", "replan_steps": steps,
        "control_options": options, "execution_gas_mode": cfg.OPT_CONFIG.get("execution_gas_mode", "clip"),
        "plan_feasible": all(validations), "plan_validation_passed": all(validations),
        "physical_feasibility_certified": False, "forecast_uses_historical_holder": False,
        "physical_validation": (resource_bundle.final_physical_validation if use_final
                                else resource_bundle.physical_validation),
        "replay_scope": "continuous_envelope_with_observed_resource_curtailment",
        "replay_complete": complete, "realized_steps": realized_steps, "planned_steps": len(schedule),
        "planned_revenue_cny": planned_revenue,
        "planned_generation_mwh": planned_metrics["generation_mwh"],
        "price_weighted_load_ratio": planned_metrics["price_weighted_load_ratio"],
        "planned_metrics": planned_metrics, "proxy_executed_metrics": executed_metrics,
        "historical_metrics": historical_metrics,
        "historical_generation_revenue_cny": historical_revenue,
        "same_conversion_historical_gas_metrics": converted_metrics,
        "same_conversion_historical_gas_revenue_cny": converted_revenue,
        "proxy_gain_vs_same_conversion_ratio": (revenue / converted_revenue - 1.)
        if complete and converted_revenue is not None and converted_revenue > 0 else None,
        "physical_parameter_sources": {
            "heat_rate": "training_prefix_positive_regression_inferred",
            "gas_caps": "training_prefix_quantile_inferred",
            "holder_capacity_and_safety": "official_task_200000m3_15_to_90_percent",
            "holder_column_mapping": "official_reply_holder_2_200000m3",
            "converter_scale": "historical_inferred_not_official_unit",
            "vent_switch_holder_penalties": "research_assumptions_not_official_costs"},
        "revenue_improvement_ratio": (planned_revenue / historical_revenue - 1.)
        if historical_revenue is not None and historical_revenue > 0 else None,
        "proxy_executed_revenue_improvement_ratio": (revenue / historical_revenue - 1.)
        if complete and historical_revenue is not None and historical_revenue > 0 else None,
        "revenue_basis": "gross_generation_value; historical_flare_penalty_unobservable",
        "proxy_release_penalty_sensitivity": penalty_sensitivity,
        "official_net_benefit_improvement_ratio": None,
        "official_net_benefit_unavailable_reason": "无真实净余气、历史放散量及官方罚金",
        "proxy_unused_no_holder_envelope_m3": {gas: float(np.asarray(unused)[:, gi].sum())
        if complete else None for gi, gas in enumerate(C.RESOURCE_NAMES[1:])},
        "proxy_executed_revenue_cny": revenue if complete else None,
        "proxy_inventory_adjusted_revenue_cny": revenue + inventory_adjustment if complete else None,
        "inventory_adjustment_cny": inventory_adjustment if complete else None,
        "proxy_curtailed_generation_mwh": shortfall, "proxy_blast_vent_m3": vent_volume,
        "initial_holder_m3": holder0, "terminal_holder_m3": holder,
        "proxy_holder_min_m3": holder_min, "proxy_holder_max_m3": holder_max,
        "solver_status_counts": {s: statuses.count(s) for s in set(statuses)},
        "solver_max_mip_gap": max(gaps) if gaps else None,
        "duration_seconds": time.time() - started,
        "time_semantics": "target_time; official_instruction_effective_time",
        "coverage_basis": "union_of_long_forecast_targets" if align_long_horizon else "committed_replay_steps_only",
        "official_test_gas_reference": "first_48_hours_only; later_empty_by_design; observed_at_or_before_origin",
        "note": "历史机组耗气是代理供气，执行降载发生在观察到资源之后；所有收益为模拟，不能证明真实气网可执行。"}
    return {"status": "SIMULATED_NOT_SUBMITTABLE", "submittable": False,
            "schedule": schedule, "summary": summary}


def optimize_daily(prepared: PreparedData, origins: pd.DatetimeIndex,
                   resource_bundle: ResourceBundle, quick: bool = False,
                   use_final: bool = False) -> dict:
    """每天提前一刻发布完整计划；日内观测只参与事后诊断，不刷新已发布值。"""
    grid = pd.DatetimeIndex(origins)
    if quick:
        grid = grid[:2 * C.STEPS_PER_DAY]
    if (not len(grid) or grid[0] != grid[0].normalize()
            or len(grid) % C.STEPS_PER_DAY or grid.has_duplicates
            or not grid.equals(pd.date_range(grid[0], periods=len(grid), freq="15min"))):
        return {"status": "BLOCKED_OPT", "schedule": None,
                "summary": {"reason": "每日固定发布要求完整、连续的自然日网格"}}
    days = grid[::C.STEPS_PER_DAY]
    issue_times = days - pd.Timedelta(minutes=C.STEP_MINUTES)
    positions = prepared.index.get_indexer(issue_times)
    if (positions < 0).any():
        return {"status": "BLOCKED_OPT", "schedule": None,
                "summary": {"reason": "缺少每日首点指令提前生成所需的历史起报点"}}
    heat = resource_bundle.final_heat_rate if use_final else resource_bundle.heat_rate
    caps = resource_bundle.final_gas_caps if use_final else resource_bundle.gas_caps
    if not heat.get("ok"):
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": "PHYSICAL_MAPPING_UNCERTAIN"}}
    forecast = forecast_resources(resource_bundle, prepared, issue_times, use_final=use_final)
    observed = prepared.frame[C.GAS_HOLDER["column"]].iloc[:positions[0] + 1].dropna()
    if observed.empty:
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": "NO_INITIAL_HOLDER"}}
    holder0 = holder = float(observed.iloc[-1])
    first = _last_level(prepared, "generator_1", int(positions[0]))
    total = _last_level(prepared, "generator_all", int(positions[0]))
    if not np.isfinite([first, total]).all():
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": "NO_INITIAL_POWER"}}
    previous = np.r_[[first / 4.] * 4, [max(total - first, 0.) / 2.] * 2]
    prices, records, plans = load_prices(), [], []
    cfg = SimpleNamespace(**vars(C))
    cfg.OPT_CONFIG = {**C.OPT_CONFIG, "solver_time_limit_s": 5.}
    cap_vector = np.array([caps[g] for g in C.RESOURCE_NAMES])
    for day, issue, pos, avail in zip(days, issue_times, positions, forecast):
        price = _price_vector(prices, day, C.STEPS_PER_DAY)
        result = _solve_day(avail, price, holder, previous, cap_vector, heat, cfg)
        if result["status"] not in ("OPTIMAL", "FEASIBLE_TIMEOUT"):
            result = _fallback_day(prepared, int(pos), avail, cap_vector, heat, cfg,
                                   holder0=holder, initial_power=previous)
        if result["status"] not in ("OPTIMAL", "FEASIBLE_TIMEOUT", "FALLBACK_SAFE"):
            return {"status": "BLOCKED_OPT", "schedule": None,
                    "summary": {"reason": result.get("message", result["status"]), "failed_day": str(day)}}
        plans.append({"day": str(day.date()), "issued_at": str(issue), "steps": C.STEPS_PER_DAY,
                      "validation": result["validation"], "solver_status": result["status"],
                      "initial_holder_m3": holder, "terminal_holder_m3": float(result["holder"][-1])})
        for k in range(C.STEPS_PER_DAY):
            power, gas = result["power"][:, k], result["gas"][:, k]
            if records:
                for value, prior, (_, limit) in zip((power[:4].sum(), power.sum()), records[-1][-2:], _aggregate_ramp_groups(cfg)):
                    if abs(float(value) - prior) > limit + cfg.OPT_CONFIG["power_tolerance_mw"]:
                        return {"status": "BLOCKED_OPT", "schedule": None,
                                "summary": {"reason": "每日发布计划的聚合爬坡衔接冲突", "failed_day": str(day)}}
            records.append([day + pd.Timedelta(minutes=k * C.STEP_MINUTES), *gas, power[:4].sum(), power.sum()])
        holder, previous = float(result["holder"][-1]), result["power"][:, -1].copy()
    schedule = pd.DataFrame(records, columns=["datetime", "opt_generator_use_blast_furnace_gas",
        "opt_generator_use_coke_gas", "opt_generator_use_converter_gas", "opt_generator_1", "opt_generator_all"])
    times = pd.DatetimeIndex(schedule.datetime)
    metrics = _dispatch_metrics(times, schedule.opt_generator_all.to_numpy(), prices)
    return {"status": "SIMULATED_NOT_SUBMITTABLE", "submittable": False, "schedule": schedule,
        "summary": {"status": "SIMULATED_NOT_SUBMITTABLE", "resource_mode": "historical_consumption_envelope",
            "dispatch_mode": "daily_fixed", "daily_publications": len(plans), "daily_plans": plans,
            "plan_feasible": True, "plan_validation_passed": True, "physical_feasibility_certified": False,
            "forecast_uses_historical_holder": False, "planned_steps": len(schedule),
            "initial_holder_m3": holder0, "terminal_holder_m3": holder,
            "planned_metrics": metrics, "planned_revenue_cny": metrics["generation_revenue_cny"],
            "planned_generation_mwh": metrics["generation_mwh"],
            "time_semantics": "target_time; daily instructions issued before the first effective timestamp",
            "coverage_basis": "same_datetime_as_long_forecast", "intraday_refresh": False,
            "physical_validation": (resource_bundle.final_physical_validation if use_final else resource_bundle.physical_validation),
            "note": "每日计划固定发布；库存连续接续预测计划，不把历史柜位重置为优化后的真实状态。"}}


def optimize(resource_forecast: np.ndarray, prepared: PreparedData,
             origins: pd.DatetimeIndex, long_pred: np.ndarray,
             resource_bundle: ResourceBundle, quick: bool = False,
             use_final: bool = False) -> dict:
    """按最新官方口径每天固定发布，时间列与长期预测一致。"""
    if C.OPT_CONFIG.get("availability_mode", "point") != "point":
        raise ValueError("正式调度只支持已验证的 point 模式")
    if resource_bundle.blocker and (not use_final or resource_bundle.blocker != "PHYSICAL_MAPPING_UNCERTAIN"):
        return {"status": resource_bundle.blocker, "submittable": False, "schedule": None,
                "summary": {"reason": resource_bundle.blocker}}
    resource_bundle.dispatch_selection = {"chosen": "daily_fixed", "replan_steps": C.STEPS_PER_DAY,
        "scope": "official_daily_publication_requirement_20261026", "intraday_refresh": False,
        "issue_lead_steps": 1, "coverage_basis": "same_datetime_as_long_forecast"}
    return optimize_daily(prepared, origins, resource_bundle, quick=quick, use_final=use_final)
