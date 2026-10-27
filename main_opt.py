"""三气种资源预测与发电调度 MILP。

边界说明：
- 1号柜早期不可用；双柜路径须有可计算初态并通过逐柜安全和资源误差核验。
- 焦炉/转炉没有独立柜位，不虚构库存。
- 官方提供参考热值，密度与效率未完整确定；合法训练期气电综合回归保留 inferred，
  不用质量比热值直接乘体积流量，也不把原始列缩放升级为官方常数。
- 已进入正式半决赛；输出字段已核对赛题，代理资源调度仍保留模拟标记。
"""
from __future__ import annotations

import copy
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


def daily_publication_times(origins, quick: bool = False, publication_time: str | None = None) -> pd.DatetimeIndex:
    """生效时刻归属最近一次已发布计划；发布当刻仍执行上一份计划的末步。"""
    grid = pd.DatetimeIndex(origins)
    if quick:
        grid = grid[:2 * C.STEPS_PER_DAY]
    clock = publication_time if publication_time is not None else C.OPT_CONFIG["publication_time"]
    if not isinstance(clock, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):(?:00|15|30|45)", clock):
        raise ValueError("每日发布时间必须为15分钟网格上的HH:MM")
    hour, minute = (int(v) for v in clock.split(":"))
    boundaries = grid.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    issued = boundaries.where(grid > boundaries, boundaries - pd.Timedelta(days=1))
    return issued.unique().sort_values()


def resource_publication_check(bundle, first_issue, use_final: bool = False) -> dict:
    """资源权重、耗气转换及网络必须在最早发布时刻前冻结，复用旧包同样检查。"""
    end = bundle.final_train_end if use_final else bundle.train_end
    heat = bundle.final_heat_rate if use_final else bundle.heat_rate
    neural = bundle.final_resource_nn if use_final else bundle.resource_nn
    sources = {"resource_model": end}
    if heat.get("calibration_end") is not None:
        sources["gas_to_power"] = heat["calibration_end"]
    if neural is not None:
        sources["resource_nn"] = neural.get("train_end")
    if use_final and bundle.final_net_supply is not None:
        sources["net_supply"] = bundle.final_net_supply.get("train_end")
    problems = []
    for name, value in sources.items():
        try:
            timestamp = pd.Timestamp(value) if value is not None else pd.NaT
            if pd.isna(timestamp) or timestamp > pd.Timestamp(first_issue):
                problems.append(f"{name}训练截止{value}晚于首份计划发布时刻{first_issue}，或缺少截止记录")
        except (TypeError, ValueError):
            problems.append(f"{name}训练截止记录无效: {value}")
    return {"passed": not problems, "first_issue": str(first_issue), "training_ends": sources, "problems": problems}


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
    final_net_supply: dict | None = None


def physical_submission_check(optimization: dict) -> dict:
    """代理计划校验与真实物理核验分开，单个成功标记不能替代约束证据。"""
    summary = optimization.get("summary", {})
    evidence = summary.get("physical_validation", {})
    checks = {
        "net_resource_boundary": summary.get("resource_mode") in (
            "net_surplus_after_priority_users", "inferred_net_surplus_after_priority_users"),
        "net_material_balance": False,
        "independent_holder_safety": False,
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
        "net_material_balance": "缺少覆盖全部计划时段的净资源守恒与流量单位核验",
        "independent_holder_safety": "缺少两座气柜分别覆盖初态及全部计划时段的安全、守恒和单位核验",
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
    # 必须有运行期逐步核验，不能用训练期量级检查或整体成功标记替代独立轨迹证据。
    steps = summary.get("planned_steps", 0)
    net = summary.get("net_resource_validation", {})
    independent = summary.get("independent_holder_validation", {})
    checks["net_material_balance"] = bool(steps > 0 and net.get("passed") is True
        and net.get("arithmetic_passed") is True and net.get("flow_units_verified") is True
        and net.get("rows") == steps and np.isfinite(net.get("balance_max_m3h", np.nan))
        and net["balance_max_m3h"] <= C.OPT_CONFIG["gas_tolerance_m3h"])
    holders = independent.get("holders", {})
    def holder_evidence_valid(name, capacity):
        item = holders.get(name, {})
        try:
            start, end, minimum, maximum, residual = [float(item.get(key, np.nan)) for key in (
                "initial_m3", "terminal_m3", "minimum_m3", "maximum_m3", "balance_max_m3")]
        except (TypeError, ValueError):
            return False
        low, high = capacity * C.GAS_HOLDER["safe_low_ratio"], capacity * C.GAS_HOLDER["safe_high_ratio"]
        tolerance = C.OPT_CONFIG["holder_tolerance_m3"]
        recovery = int(item.get("initial_outside_recovery_steps", 0) or 0)
        # 首态已在安全区外时，允许在恢复期内从实测值出发；恢复期后仍须回到安全区。
        low_bound = min(start, low) if recovery else low
        return (np.isfinite([start, end, minimum, maximum, residual]).all()
            and low_bound - tolerance <= minimum
            and maximum <= high + tolerance
            and -tolerance <= start <= capacity + tolerance
            and -tolerance <= end <= capacity + tolerance
            and 0. <= residual <= tolerance)
    checks["independent_holder_safety"] = bool(steps > 0 and independent.get("passed") is True
        and independent.get("rows") == steps and all(
            holders.get(name, {}).get("passed") is True
            and holders[name].get("units_verified") is True
            and holders[name].get("safety_passed") is True
            and holders[name].get("plan_margin_passed") is True
            and holders[name].get("balance_passed") is True
            and holders[name].get("capacity_m3") == capacity
            and holder_evidence_valid(name, capacity)
            for name, capacity in C.GAS_HOLDER["capacity_by_column_m3"].items()))
    if evidence.get("official_constraints"):
        checks["official_constraints"] = evidence["official_constraints"].get("passed") is True
        descriptions["official_constraints"] = "当前配置与赛题已明确的硬约束不一致"
    blocked = [name for name, passed in checks.items() if not passed]
    return {"passed": not blocked, "checks": checks, "blocked_items": blocked,
            "evidence_basis": {name: evidence.get(name, {}).get("basis") for name in checks},
            "problems": [descriptions[name] for name in blocked]}


def model_submission_check(optimization: dict, config=C) -> dict:
    """从逐步轨迹重算硬约束；默认假设的可提交性不等同于现场实测认证。"""
    summary = optimization.get("summary", {})
    evidence = summary.get("model_evidence", {})
    if not evidence and optimization.get("schedule") is None:
        return {"passed": False, "simulation_constraints_passed": False, "checks": {"schedule_present": False},
            "problems": [summary.get("reason") or "缺少优化调度及逐步核验轨迹"],
            "validation_basis": "declared_assumptions", "field_safety_certified": False}
    checks = {}
    try:
        n = int(summary.get("planned_steps", 0))
        stock = np.asarray(evidence["holders_m3"], float)
        available, gas, vent = [np.asarray(evidence[key], float) for key in ("available_m3h", "gas_m3h", "vent_m3h")]
        power = np.asarray(evidence["unit_power_mw"], float)
        previous = np.asarray(evidence["previous_unit_power_mw"], float)
        beta = np.array([evidence["coefficients_mw_per_m3h"][g] for g in config.RESOURCE_NAMES])
        arrays = (stock, available, gas, vent, power, previous, beta)
        checks["complete_finite_trajectories"] = bool(n > 0 and stock.shape == (n+1, 2)
            and all(a.shape == (n, 3) for a in (available, gas, vent)) and power.shape == (n, 6)
            and previous.shape == (6,) and beta.shape == (3,) and all(np.isfinite(a).all() for a in arrays))
        if not checks["complete_finite_trajectories"]:
            raise ValueError("缺少完整、有限的资源、库存或机组逐步轨迹")
        gt, pt, st = [config.OPT_CONFIG[key] for key in ("gas_tolerance_m3h", "power_tolerance_mw", "holder_tolerance_m3")]
        # 放散未出现在提交指令中，不能靠本地自由放散变量证明线上柜位安全。
        checks["submitted_controls_only"] = bool(np.max(np.abs(vent)) <= gt)
        capacity = np.array(list(config.GAS_HOLDER["capacity_by_column_m3"].values()))
        recovery = np.zeros(2, dtype=int)
        safety = np.ones(2, dtype=bool)
        for j in range(2):
            inside_initial = bool(.15*capacity[j] - st <= stock[0, j] <= .90*capacity[j] + st)
            recovery[j] = 0 if inside_initial else max(1, min(int(config.OPT_CONFIG.get("holder_recovery_steps", 4)), n))
            after = stock[recovery[j]:, j]
            safety[j] = bool(((after >= .15*capacity[j]-st) & (after <= .90*capacity[j]+st)).all()
                             and (stock[:, j] >= -st).all() and (stock[:, j] <= capacity[j]+st).all())
        checks["independent_holder_safety"] = bool(safety.all())
        supply = evidence.get("net_supply_model", {})
        uncertainty = net_supply_safety_check(supply, config)
        checks["historical_resource_uncertainty"] = uncertainty["passed"]
        checks["independent_holder_error_margin"] = False
        residual = available-gas-vent
        residual[:, 0] -= np.diff(stock, axis=0).sum(axis=1)/.25
        inlet = evidence.get("holder_inlet_m3h")
        exchange = evidence.get("holder_exchange_m3h")
        if inlet is not None and exchange is not None:
            inlet, exchange = np.asarray(inlet, float), np.asarray(exchange, float)
            holder_gas = inlet - exchange
            cap = np.asarray(list(config.GAS_HOLDER["capacity_by_column_m3"].values()), float)
            checks["holder_material_balance"] = bool(inlet.shape == (n, 2) and exchange.shape == (n, 2)
                and np.isfinite(inlet).all() and np.isfinite(exchange).all()
                and np.max(np.abs(np.diff(stock, axis=0)/.25 - exchange)) <= gt
                and np.max(np.abs(inlet-holder_gas-exchange)) <= gt
                and np.max(np.abs(holder_gas.sum(axis=1)-gas[:, 0])) <= gt
                and (inlet >= -gt).all() and (holder_gas >= -gt).all()
                and (np.abs(exchange) <= cap[None, :]/.25 + gt).all())
        checks["net_mass_balance"] = bool(np.max(np.abs(residual)) <= gt
            and (gas >= -gt).all() and (vent >= -gt).all() and checks.get("holder_material_balance", True))
        schedule = optimization.get("schedule")
        if schedule is not None:
            times = pd.DatetimeIndex(pd.to_datetime(schedule.datetime))
            state_times = pd.DatetimeIndex(pd.to_datetime(evidence.get("holder_state_times", [])))
            checks["inventory_interval_semantics"] = bool(summary.get("inventory_interval_semantics") == "command_start_to_next_step"
                and len(times) == n and state_times.equals(pd.date_range(times[0], periods=n+1,
                    freq=f"{config.STEP_MINUTES}min")))
        caps = np.asarray(config.OPT_CONFIG["unit_capacity_mw"], float)
        on = power > pt
        checks["unit_load_bounds"] = bool((power >= -pt).all() and (power <= caps+pt).all()
            and (power[on] >= np.broadcast_to(config.OPT_CONFIG["min_load_ratio"]*caps, power.shape)[on]-pt).all())
        checks["initial_unit_load_bounds"] = _valid_initial_unit_power(previous, config)
        path = np.vstack((previous, power))
        checks["aggregate_ramp"] = all(np.max(np.abs(np.diff(path[:, indices].sum(axis=1)))) <= limit+pt
            for indices, limit in _aggregate_ramp_groups(config))
        checks["gas_power_consistency"] = bool((beta > 0.).all() and np.max(np.abs(power.sum(axis=1)-gas@beta)) <= pt)
        plans = summary.get("daily_plans", [])
        checks["daily_publication_and_plan_validation"] = bool(plans and summary.get("dispatch_mode") == "daily_fixed"
            and summary.get("intraday_refresh") is False and summary.get("daily_publications") == len(plans)
            and sum(p.get("published_steps", 0) for p in plans) == n
            and all(p.get("steps") == 96 and p.get("validation", {}).get("passed") is True
                and p.get("model_constraints_passed") is True and p.get("group_allocation", {}).get("passed") is True for p in plans))
        priming = summary.get("first_effective_state", {})
        checks["first_instruction_state_primed_causally"] = bool(plans and priming.get("future_observations_used") is False
            and priming.get("interval_semantics") == "command_start_to_next_step"
            and priming.get("first_effective_at") == plans[0].get("effective_start")
            and priming.get("issued_at") == plans[0].get("issued_at")
            and priming.get("first_effective_holders_m3") == plans[0].get("initial_holders_m3"))
        if checks["first_instruction_state_primed_causally"]:
            names = list(config.GAS_HOLDER["capacity_by_column_m3"])
            before_bridge = np.asarray([priming["initial_holders_m3"][name] for name in names],float)
            after_bridge = np.asarray([priming["first_effective_holders_m3"][name] for name in names],float)
            bridge_flow = float(priming["predicted_net_blast_m3h"])-float(priming["previous_blast_use_m3h"])
            checks["first_instruction_stock_balance"] = bool(np.isfinite(before_bridge).all() and np.isfinite(after_bridge).all()
                and np.isfinite(bridge_flow) and abs((after_bridge-before_bridge).sum()-bridge_flow*config.STEP_MINUTES/60.) <= st
                and (before_bridge >= -st).all() and (before_bridge <= capacity+st).all()
                and (after_bridge >= -st).all() and (after_bridge <= capacity+st).all())
        if uncertainty["passed"] and plans and optimization.get("schedule") is not None:
            lo, hi = np.asarray(uncertainty["lower_error_m3"]), np.asarray(uncertainty["upper_error_m3"])
            times = pd.DatetimeIndex(pd.to_datetime(optimization["schedule"].datetime))
            first = pd.Timestamp(plans[0]["issued_at"]) + pd.Timedelta(minutes=config.STEP_MINUTES)
            offsets = np.asarray((times-first)/pd.Timedelta(minutes=config.STEP_MINUTES), dtype=int)
            days, steps = offsets//96, offsets%96
            lower = lo[steps] + days*lo[-1] + uncertainty["priming_error_lower_m3"]
            upper = hi[steps] + days*hi[-1] + uncertainty["priming_error_upper_m3"]
            checks["independent_holder_error_margin"] = bool((offsets >= 0).all()
                and (stock[1:] + lower[:, None] >= .15*capacity-st).all()
                and (stock[1:] + upper[:, None] <= .90*capacity+st).all())
        checks["training_before_publication"] = summary.get("resource_publication_check", {}).get("passed") is True
        defaults = evidence.get("declared_defaults", {})
        scales = defaults.get("holder_scales", {})
        checks["assumptions_explicit"] = bool(set(scales) == set(config.GAS_HOLDER["capacity_by_column_m3"])
            and all(np.isfinite(v.get("scale_to_m3", np.nan)) and v["scale_to_m3"] > 0. for v in scales.values())
            and defaults.get("reservation_role") and np.isfinite(defaults.get("converter_scale", np.nan))
            and defaults["converter_scale"] > 0. and defaults.get("field_safety_certified") is False)
        checks["net_resource_boundary"] = summary.get("resource_mode") == "inferred_net_surplus_after_priority_users"
        schedule = optimization.get("schedule")
        checks["schedule_present"] = schedule is not None
        if schedule is not None:
            values = schedule.iloc[:, 1:].to_numpy(float)
            expected = np.column_stack((gas, power[:, :4].sum(axis=1), power.sum(axis=1)))
            checks["schedule_matches_trajectories"] = bool(values.shape == expected.shape and np.isfinite(values).all()
                and np.max(np.abs(values[:, :3]-expected[:, :3])) <= gt
                and np.max(np.abs(values[:, 3:]-expected[:, 3:])) <= pt)
            times = pd.DatetimeIndex(pd.to_datetime(schedule.datetime))
            issues = daily_publication_times(times)
            checks["daily_time_semantics"] = bool(len(issues) == len(plans)
                and all(pd.Timestamp(p["issued_at"]) == issue
                    and pd.Timestamp(p["effective_start"]) == issue+pd.Timedelta(minutes=config.STEP_MINUTES)
                    and pd.Timestamp(p["effective_end"]) == issue+pd.Timedelta(days=1)
                    for p, issue in zip(plans, issues)))
        # 历史误差集合是额外风险核验；模拟表仍须通过其余所有逐步模型硬约束。
        risk_checks = {"historical_resource_uncertainty", "independent_holder_error_margin"}
        point_passed = all(value for name, value in checks.items() if name not in risk_checks)
        return {"passed": all(checks.values()), "simulation_constraints_passed": point_passed, "checks": checks,
            "problems": [name for name, valid in checks.items() if not valid],
            "net_balance_max_m3h": float(np.max(np.abs(residual))), "rows": n,
            "validation_basis": "declared_assumptions", "field_safety_certified": False}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return {"passed": False, "simulation_constraints_passed": False, "checks": checks, "problems": [str(exc)],
                "validation_basis": "declared_assumptions", "field_safety_certified": False}


def submission_readiness(optimization: dict, config=C) -> dict:
    physical = physical_submission_check(optimization)
    if config.OPT_CONFIG.get("submission_validation_basis") != "declared_assumptions":
        return {"passed": physical["passed"], "validation_basis": "strict_physical", "physical": physical,
                "problems": physical["problems"], "warnings": []}
    model = model_submission_check(optimization, config)
    # 声明的单位与流量假设可用于模拟，不能代替正式物理核验或赋予提交资格。
    return {"passed": bool(model["passed"] and physical["passed"]),
            "simulation_passed": model["simulation_constraints_passed"], "validation_basis": "declared_assumptions", "model": model,
            "physical": physical, "problems": list(dict.fromkeys(model["problems"] + physical["problems"])),
            "warnings": physical["problems"], "field_safety_certified": physical["passed"]}


def _physical_stats(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {"rows": len(values), "valid_rows": len(finite),
        "valid_fraction": float(len(finite) / max(len(values), 1)),
        "min": float(finite.min()) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None}


def net_resource_boundary(generation, priority_demand, inferred_reservation, *,
                          issued_at, training_end, flow_units_verified=False,
                          holder_inlet_m3h=None, forecast_components=None,
                          forecast_method=None) -> dict:
    """先保生产，再给发电和气柜分配余量；推测预留量不回写为用户实测需求。

    两柜进气按官方高炉分组路由单独给出；机组耗气仍由共享母管总耗气表示，
    逐柜出口在 MILP 内由独立库存平衡决定，不在这里预设固定分流比。
    """
    inputs = [np.asarray(x, dtype=float) for x in (generation, priority_demand, inferred_reservation)]
    shape = inputs[0].shape
    if len(shape) != 2 or shape[1] != len(C.RESOURCE_NAMES) or shape[0] == 0 or any(x.shape != shape for x in inputs):
        raise ValueError("发生量、已知用户和推测预留量必须为同形状的[时段,三气种]数组")
    problems = []
    if any(not np.isfinite(x).all() or (x < 0.).any() for x in inputs):
        problems.append("发生量或生产需求缺测、非有限或为负；不得补零构造净资源")
    inlet = None
    if holder_inlet_m3h is not None:
        inlet = np.asarray(holder_inlet_m3h, dtype=float)
        if inlet.shape != (shape[0], len(C.GAS_HOLDER["capacity_by_column_m3"])) or not np.isfinite(inlet).all():
            problems.append("两柜高炉进气路由缺测、形状错误或含非有限值")
            inlet = None
    try:
        issue, end = pd.Timestamp(issued_at), pd.Timestamp(training_end)
        if pd.isna(issue) or pd.isna(end) or end > issue:
            problems.append("净资源模型训练截止晚于发布时间，或缺少有效截止记录")
    except (TypeError, ValueError):
        problems.append("净资源模型训练截止或发布时间无效")
    net = inputs[0] - inputs[1] - inputs[2]
    return {"passed": not problems, "problems": problems, "available_m3h": net,
            "generation_m3h": inputs[0], "priority_demand_m3h": inputs[1],
            "inferred_reservation_m3h": inputs[2], "training_end": str(training_end),
            "issued_at": str(issued_at), "flow_units_verified": flow_units_verified is True,
            "resource_mode": "inferred_net_surplus_after_priority_users",
            "reservation_role": "inferred_users_and_unresolved_balance_reserve; not_observed_user_truth",
            "negative_net_rows_by_gas": {g: int((net[:, i] < 0.).sum()) for i, g in enumerate(C.RESOURCE_NAMES)},
            "holder_discharge_included": False, "mixed_station_boundary": "inlet_deducted_once",
            "holder_inlet_m3h": inlet,
            "holder_inlet_basis": ("blast_furnace_1_3_to_30e4_holder; blast_furnace_4_5_to_20e4_holder; "
                                   "generator_withdrawal_shared_by_independent_balances"
                                   if inlet is not None else "unavailable"),
            "forecast_components": forecast_components or {},
            "forecast_method": forecast_method or "declared_net_supply"}


def validate_net_dispatch(boundary: dict, result: dict, initial_holders: dict,
                          holder_units_verified: dict, config=C) -> dict:
    """逐柜守恒及独立安全；两柜总库存合规不能抵消其中一柜越界。"""
    names = tuple(config.GAS_HOLDER["capacity_by_column_m3"])
    problems = list(boundary.get("problems", []))
    available = np.asarray(boundary["available_m3h"], dtype=float)
    n = len(available)
    initial = np.array([initial_holders.get(name, np.nan) for name in names], dtype=float)
    levels = np.asarray(result.get("holders_m3", []), dtype=float)
    exchange = np.asarray(result.get("holder_exchange_m3h", []), dtype=float)
    inlet = np.asarray(result.get("holder_inlet_m3h", boundary.get("holder_inlet_m3h", [])), dtype=float)
    withdrawal = np.asarray(result.get("holder_withdrawal_m3h", []), dtype=float)
    release = np.asarray(result.get("holder_release_m3h", []), dtype=float)
    gas, vent = (np.asarray(result.get(key, []), dtype=float) for key in ("gas", "vent"))
    if (available.shape != (n, 3) or n == 0 or levels.shape != (n, len(names))
            or exchange.shape != levels.shape or inlet.shape != levels.shape
            or withdrawal.shape != levels.shape or release.shape != levels.shape
            or gas.shape != (3, n) or vent.shape != (3, n)
            or not all(np.isfinite(x).all() for x in (available, initial, levels, exchange, inlet, withdrawal, release, gas, vent))):
        return {"passed": False, "problems": problems + ["净资源或独立气柜轨迹缺测、形状错误"],
                "net_resource_validation": {"passed": False}, "independent_holder_validation": {"passed": False}}
    gt, st = config.OPT_CONFIG["gas_tolerance_m3h"], config.OPT_CONFIG["holder_tolerance_m3"]
    dt = config.STEP_MINUTES / 60.
    previous = np.vstack((initial, levels[:-1]))
    reconstructed = inlet - withdrawal - release
    stock_residual = levels - previous - dt * reconstructed
    material_residual = available - gas.T - vent.T
    material_residual[:, 0] -= reconstructed.sum(axis=1)
    expected = (np.asarray(boundary["generation_m3h"]) - np.asarray(boundary["priority_demand_m3h"])
                - np.asarray(boundary["inferred_reservation_m3h"]))
    boundary_consistent = bool(expected.shape == available.shape and np.allclose(expected, available, atol=gt, rtol=0.))
    allocation = result.get("holder_columns") == list(names)
    exchange_consistent = bool((np.abs(exchange - reconstructed) <= st).all())
    routing = bool((withdrawal >= -gt).all() and (release >= -gt).all()
                   and (reconstructed <= inlet + gt).all()
                   and (np.abs(withdrawal.sum(axis=1) - gas[0]) <= gt).all()
                   and (np.abs(release.sum(axis=1) - vent[0]) <= gt).all())
    holders = {}
    for i, name in enumerate(names):
        capacity = config.GAS_HOLDER["capacity_by_column_m3"][name]
        path = np.r_[initial[i], levels[:, i]]
        low, high = capacity * config.GAS_HOLDER["safe_low_ratio"], capacity * config.GAS_HOLDER["safe_high_ratio"]
        initial_inside = bool(low - st <= initial[i] <= high + st)
        recovery = max(1, min(int(config.OPT_CONFIG.get("holder_recovery_steps", 4)), n))
        start = 0 if initial_inside else recovery
        inside = bool(((levels[start:, i] >= low - st) & (levels[start:, i] <= high + st)).all()) if start < n else True
        margin = capacity * config.OPT_CONFIG["holder_margin_ratio"]
        planned_low, planned_high = _independent_holder_plan_bounds(initial, n, config)
        planned = bool(((levels[:, i] >= planned_low[i] - st) & (levels[:, i] <= planned_high[i] + st)).all())
        balanced = bool((np.abs(stock_residual[:, i]) <= st).all())
        target = config.OPT_CONFIG.get("terminal_inventory_by_holder_m3", {}).get(name,
            np.clip(initial[i], low + margin, high - margin))
        terminal = not config.OPT_CONFIG["terminal_holder"] or abs(levels[-1, i] - target) <= st
        unit_ok = holder_units_verified.get(name) is True
        holders[name] = {"passed": bool(inside and planned and balanced and terminal and allocation and unit_ok
                                        and routing and exchange_consistent),
            "capacity_m3": capacity, "initial_m3": float(initial[i]), "terminal_m3": float(levels[-1, i]),
            "minimum_m3": float(path.min()), "maximum_m3": float(path.max()),
            "safe_low_m3": low, "safe_high_m3": high, "units_verified": unit_ok,
            "initial_inside_safe_band": initial_inside,
            "initial_outside_recovery_steps": 0 if initial_inside else recovery,
            "safety_passed": inside, "plan_margin_passed": planned, "balance_passed": balanced,
            "terminal_passed": bool(terminal), "terminal_target_m3": float(target),
            "balance_max_m3": float(np.abs(stock_residual[:, i]).max())}
        if not holders[name]["passed"]:
            problems.append(f"{name}独立安全、单位或库存守恒未通过")
    nonnegative = bool((gas >= -gt).all() and (vent >= -gt).all())
    if np.max(np.abs(vent)) > gt:
        problems.append("提交路径依赖未提交的放散控制")
    closed = bool((np.abs(material_residual) <= gt).all() and nonnegative and boundary_consistent
                  and routing and exchange_consistent and boundary.get("passed") is True)
    if not closed:
        problems.append("扣除生产需求后，发电、放散与两柜交换不守恒")
    flow_verified = boundary.get("flow_units_verified") is True
    net_check = {"passed": bool(closed and flow_verified), "arithmetic_passed": closed,
        "rows": n, "balance_max_m3h": float(np.abs(material_residual).max()),
        "flow_units_verified": flow_verified, "inferred_demand_is_observed": False,
        "holder_routing_passed": routing, "holder_exchange_consistent": exchange_consistent,
        "withdrawal_matches_submitted_gas": bool((np.abs(withdrawal.sum(axis=1) - gas[0]) <= gt).all())}
    if not flow_verified:
        problems.append("净资源原始流量单位仍未核验")
    holder_check = {"passed": all(h["passed"] for h in holders.values()), "rows": n, "holders": holders}
    return {"passed": not problems, "problems": problems,
            "net_resource_validation": net_check, "independent_holder_validation": holder_check,
            "scope": "declared_net_supply_and_two_independent_holders; inferred_supply_not_field_certification"}


def solve_net_day(boundary: dict, price, initial_holders: dict, initial_power,
                  caps, heat_rate: dict, *, holder_units_verified: dict, config=C) -> dict:
    """在原MILP内直接约束两柜；缺少任一柜初态时不能借另一柜兜底。"""
    names = tuple(config.GAS_HOLDER["capacity_by_column_m3"])
    initial = np.array([initial_holders.get(name, np.nan) for name in names], dtype=float)
    capacities = np.array([config.GAS_HOLDER["capacity_by_column_m3"][name] for name in names])
    low = capacities * config.GAS_HOLDER["safe_low_ratio"]
    high = capacities * config.GAS_HOLDER["safe_high_ratio"]
    tolerance = config.OPT_CONFIG["holder_tolerance_m3"]
    if (boundary.get("passed") is not True or not np.isfinite(initial).all()
            or (initial < -tolerance).any() or (initial > capacities + tolerance).any()):
        return {"status": "BLOCKED_OPT", "message": "净资源边界或两柜初始库存、单位未通过；不虚构未知库存"}
    if any(config.OPT_CONFIG.get(key) is not None for key in ("terminal_inventory_m3", "planned_inventory_floor_m3")):
        return {"status": "BLOCKED_OPT", "message": "单柜库存目标不能直接用于两柜，须分别定义约束"}
    available = np.asarray(boundary["available_m3h"], dtype=float)
    prices = np.asarray(price, dtype=float)
    if prices.shape != (len(available),):
        return {"status": "BLOCKED_OPT", "message": "净资源时段与电价长度不一致"}
    result = _solve_day(available, prices, float(initial.sum()), np.asarray(initial_power, dtype=float),
                        np.asarray(caps, dtype=float), heat_rate, config, conserve_all_gases=True,
                        independent_holders=initial_holders,
                        holder_inlet=boundary.get("holder_inlet_m3h"))
    if result["status"] not in ("OPTIMAL", "FEASIBLE_TIMEOUT"):
        return result
    result["network_validation"] = validate_net_dispatch(boundary, result, initial_holders, holder_units_verified, config)
    network = result["network_validation"]
    model_constraints = network["net_resource_validation"].get("arithmetic_passed") and all(
        all(h.get(key) for key in ("safety_passed", "plan_margin_passed", "balance_passed", "terminal_passed"))
        for h in network["independent_holder_validation"].get("holders", {}).values())
    if not model_constraints:
        return {"status": "INVALID_SOLUTION", "message": str(result["network_validation"])}
    result["model_constraints_passed"] = model_constraints
    result["resource_mode"] = boundary["resource_mode"]
    result["physical_feasibility_certified"] = False
    result["topology_basis"] = ("blast_furnace_1_3_to_30e4_holder; blast_furnace_4_5_to_20e4_holder; "
                                "generator_withdrawal_shared_by_independent_balances; flow_limits_unprovided")
    return result


def physical_gas_ledger(prepared, end_pos):
    """已知用户扣减账本；混气站进气扣一次，未知项不补零，不冒充完整净余气。"""
    prepared.require_training_data()
    if not 0 <= end_pos < len(prepared.index):
        raise ValueError("物理核验截止位置无效")
    frame = prepared.frame.iloc[:end_pos + 1]

    def values(columns):
        block = prepared.observed_values(columns).iloc[:end_pos + 1].replace([np.inf, -np.inf], np.nan)
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
        state = prepared.observation_quality(required)
        complete = state["observed"].iloc[:end_pos + 1].all(axis=1)
        use_column = "load_generator_use_" + {"blast": "blast_furnace", "coke": "coke", "converter": "converter"}[gas] + "_gas"
        use = prepared.observed_values([use_column]).iloc[:end_pos + 1, 0]
        # 瞬时差额包含未观测用户、放散、计量误差与库存交换，不能伪称用户真值。
        unresolved = ledger[gas] - use if gas != "converter" else pd.Series(np.nan, index=frame.index)
        details[gas] = {"generation_columns": list(specification["generation"]),
            "priority_user_columns": list(specification["users"]), "mixed_station_column": specification["mixed"],
            "unavailable_columns": missing, "net_before_generation": _physical_stats(ledger[gas]),
            "negative_net_rows": int((ledger[gas] < 0.).sum()),
            "quantity_role": "remaining_after_known_deductions; not_certified_net_surplus",
            "input_semantics": "valid_raw_observations_only; no_causal_imputation",
            "observation_provenance_available": not prepared.provenance.empty,
            "raw_complete_rows": int(complete.sum()),
            "no_history_input_rows": int(state["no_observed_history"].iloc[:end_pos + 1].any(axis=1).sum()),
            "imputed_input_rows_excluded": int(state["imputed"].iloc[:end_pos + 1].any(axis=1).sum()),
            "unresolved_flow_proxy": {**_physical_stats(unresolved),
                "quantity_role": "signed_instantaneous_balance_difference; not_user_demand_or_true_net_resource",
                "unknown_components": ["unobserved_users", "vent", "inventory_exchange", "meter_error"],
                "negative_rows": int(unresolved.lt(0.).sum()),
                "negative_values_clipped": False, "raw_scale_verified": False,
                "unit_status": "standard_m3h_assumed_sensor_scale" if gas != "converter" else "raw_scale_unverified",
                "used_for_dispatch": False},
            "all_users_enumerated": False,
            "unit_basis": "historical_inferred_scale" if gas == "converter" else "official_standard_m3h; raw_sensor_scaling_unverified"}
    return ledger, details


def net_supply_safety_check(pack: dict, config=C) -> dict:
    """历史误差是风险集合，不是官方阈值；固定指令须能承受集合内的库存偏移。"""
    checks = {}
    try:
        calibration = pack["resource_error_calibration"]
        paths = np.asarray(calibration["cumulative_error_m3"], float)
        priming = np.asarray(calibration.get("priming_error_m3", []), float)
        issues = pd.DatetimeIndex(calibration["issued_at"])
        checks["complete_historical_days"] = bool(paths.ndim == 2 and paths.shape == (len(issues), 96)
            and priming.shape == (len(issues),2) and len(issues) >= 5
            and np.isfinite(paths).all() and np.isfinite(priming).all())
        checks["causal_training_calibration"] = bool(len(issues) and not issues.hasnans
            and not issues.has_duplicates and issues.is_monotonic_increasing
            and (issues+pd.Timedelta(days=1, minutes=config.STEP_MINUTES) <= pd.Timestamp(pack["train_end"])).all()
            and calibration["test_data_used"] is False
            and calibration.get("interval_semantics") == "command_start_to_next_step")
        if not checks["complete_historical_days"]:
            raise ValueError("缺少至少5个完整训练日的逐步资源误差，不能签发库存误差核验")
        lower, upper = np.minimum(paths.min(axis=0), 0.), np.maximum(paths.max(axis=0), 0.)
        initial_low, initial_high = min(float(priming.min()),0.), max(float(priming.max()),0.)
        width = upper-lower+initial_high-initial_low
        capacities = np.asarray(list(config.GAS_HOLDER["capacity_by_column_m3"].values()), float)
        safe_width = capacities*(config.GAS_HOLDER["safe_high_ratio"]-config.GAS_HOLDER["safe_low_ratio"])
        # 总柜位误差必须先合并同一日的两个初态误差，再与该日累计流量误差配对。
        aggregate_paths = paths + priming.sum(axis=1)[:, None]
        aggregate_width = (np.maximum(aggregate_paths.max(axis=0), 0.)
                           - np.minimum(aggregate_paths.min(axis=0), 0.))
        # 不能证明误差只进入某一柜，因此逐柜预留全部净流量误差；不借总量掩盖单柜风险。
        checks["independent_holder_error_capacity"] = bool((width[:, None] <= safe_width[None, :]).all())
        first_conflict = np.flatnonzero(width > safe_width.min())
        first_aggregate_conflict = np.flatnonzero(aggregate_width > safe_width.sum())
        return {"passed": all(checks.values()), "checks": checks,
            "problems": [k for k, v in checks.items() if not v],
            "historical_days": len(issues), "maximum_error_width_m3": float(width.max()),
            "safe_width_by_holder_m3": safe_width.tolist(),
            "aggregate_safe_width_m3": float(safe_width.sum()),
            "maximum_aggregate_error_width_m3": float(aggregate_width.max()),
            "even_aggregate_buffer_insufficient": bool(len(first_aggregate_conflict)),
            "minimum_extra_aggregate_buffer_m3": float(max(aggregate_width.max()-safe_width.sum(), 0.)),
            "first_aggregate_buffer_conflict_minutes": int((first_aggregate_conflict[0]+1)*config.STEP_MINUTES) if len(first_aggregate_conflict) else None,
            "first_independent_buffer_conflict_minutes": int((first_conflict[0]+1)*config.STEP_MINUTES) if len(first_conflict) else None,
            "lower_error_m3": lower.tolist(), "upper_error_m3": upper.tolist(),
            "priming_error_lower_m3": initial_low, "priming_error_upper_m3": initial_high,
            "scope": "historical_inferred_net_error_set; not_official_threshold_or_future_guarantee"}
    except (KeyError, TypeError, ValueError) as exc:
        return {"passed": False, "checks": checks, "problems": [str(exc)]}


def net_balance_training_labels(prepared, scales):
    """起点流量与后一个柜位差配对；该标签须等到t+15min才能用于训练。"""
    prepared.require_training_data()
    names = list(C.GAS_HOLDER["capacity_by_column_m3"])
    stock = prepared.observed_values(names).mul([
        np.nan if scales[n]["scale_to_m3"] is None else scales[n]["scale_to_m3"] for n in names], axis=1)
    gas = prepared.observed_values(["load_generator_use_blast_furnace_gas"]).iloc[:, 0]
    return gas + (stock.shift(-1)-stock).sum(axis=1, min_count=2)/(C.STEP_MINUTES/60.)


def _calibrate_net_error(prepared, observed, scales):
    """逐日只用当时历史预测，目标时刻柜位差额仅在训练期作事后误差标签。"""
    net_proxy = net_balance_training_labels(prepared, scales)
    names = list(C.GAS_HOLDER["capacity_by_column_m3"])
    stock = prepared.observed_values(names).mul([
        np.nan if scales[n]["scale_to_m3"] is None else scales[n]["scale_to_m3"] for n in names],axis=1)
    end = prepared.index[-1]
    candidates = pd.date_range((end-pd.Timedelta(days=21)).normalize(), end.normalize(), freq="D")
    hours, minutes = map(int, C.OPT_CONFIG["publication_time"].split(":"))
    paths, issues, priming_errors = [], [], []
    for day in candidates:
        issue = day + pd.Timedelta(hours=hours, minutes=minutes)
        if issue+pd.Timedelta(days=1, minutes=C.STEP_MINUTES) > end:
            continue
        times = pd.date_range(issue+pd.Timedelta(minutes=C.STEP_MINUTES), periods=96, freq="15min")
        actual = net_proxy.reindex(times).to_numpy(float)
        initial_states = stock.reindex([issue-pd.Timedelta(minutes=C.STEP_MINUTES),issue,times[0]]).to_numpy(float)
        past = observed.loc[issue-pd.Timedelta(days=28):issue]
        if not len(past) or not np.isfinite(actual).all() or not np.isfinite(initial_states).all():
            continue
        profile = past.groupby(past.index.hour*4+past.index.minute//15).median().reindex(range(96)).fillna(past.median())
        slots = times.hour*4+times.minute//15
        prediction = (profile.blast_generation-profile.blast_priority-profile.blast_residual.clip(lower=0.)).to_numpy()[slots]
        if np.isfinite(prediction).all():
            paths.append(np.cumsum((actual-prediction)*(C.STEP_MINUTES/60.)).tolist())
            issues.append(str(issue))
            priming_errors.append((initial_states[2]-(2*initial_states[1]-initial_states[0])).tolist())
    return {"issued_at": issues, "cumulative_error_m3": paths, "test_data_used": False,
        "priming_error_m3": priming_errors,
        "interval_semantics": "command_start_to_next_step",
        "quantity": "generator_use_plus_observed_two_holder_change; inferred_net_including_historical_uncontrolled_terms",
        "method": "causal_28_day_slot_median_daily_backtest", "window_days": 21}


def _slot_numbers(index):
    index = pd.DatetimeIndex(index)
    return index.hour * 4 + index.minute // 15


def _causal_component_forecast(frame: pd.DataFrame, column: str, slots: np.ndarray, *,
                               quantile: float | None = None, recent_steps: int = 8) -> np.ndarray:
    """只用发布时刻之前的历史：同日时段中位数 + 最近若干步偏移；不读未来。"""
    history = frame[column].dropna().astype(float)
    if history.empty:
        return np.full(len(slots), np.nan)
    by_slot = history.groupby(_slot_numbers(history.index)).median().reindex(range(96))
    by_slot = by_slot.fillna(float(history.median()))
    recent = history.iloc[-max(int(recent_steps), 1):]
    recent_slots = _slot_numbers(recent.index)
    offset = float(np.nanmedian(recent.to_numpy(float) - by_slot.reindex(recent_slots).to_numpy(float)))
    values = by_slot.reindex(slots).to_numpy(float) + offset
    if quantile is not None:
        high = history.groupby(_slot_numbers(history.index)).quantile(quantile).reindex(range(96))
        high = high.fillna(float(history.quantile(quantile)))
        values = np.maximum(values, high.reindex(slots).to_numpy(float) + offset)
    return values


def _net_supply_history(pack: dict, observed: pd.DataFrame | None = None,
                        issue=None) -> pd.DataFrame | None:
    payload = pack.get("history")
    if not payload or not payload.get("rows"):
        return None
    frame = pd.DataFrame(payload["rows"], columns=payload["columns"])
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.set_index("datetime").sort_index()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if observed is not None and len(observed):
        # 只用发布时刻之前的已观测测试前缀，且不得早于训练截止；不把未来行混入画像。
        lower = pd.Timestamp(pack["train_end"])
        upper = pd.Timestamp(issue) if issue is not None else observed.index.max()
        columns = [c for c in frame.columns if c in observed]
        extra = observed.loc[(observed.index >= lower) & (observed.index < upper), columns].copy()
        if len(extra):
            extra.index.name = "datetime"
            frame = pd.concat([frame, extra], axis=0).sort_index()
            frame = frame[~frame.index.duplicated(keep="last")]
    return frame



def fit_net_supply(prepared: PreparedData) -> dict:
    """用合法训练前缀构造因果净资源画像，并保留逐日误差做风险核验。

    关键修正：净资源标签不是“发生量-用户-历史发电耗气”，而是
    发生量-已知用户-(历史发电耗气+气柜库存变化)。差额只作为未观测用户/
    放散/计量残差的推测预留量，不生成用户真值。
    """
    prepared.require_training_data()
    _, ledger = physical_gas_ledger(prepared, len(prepared.index) - 1)
    observed, unknown = pd.DataFrame(index=prepared.index), {}
    for gas, item in ledger.items():
        generation_columns = item["generation_columns"]
        users = item["priority_user_columns"] + [item["mixed_station_column"]]
        present = [c for c in users if prepared.observed_values([c])[c].notna().any()]
        unknown[gas] = [c for c in users if c not in present]
        generation = prepared.observed_values(generation_columns).sum(axis=1, min_count=len(generation_columns))
        if gas == "converter":
            generation *= C.CONVERTER_GAS_SCALE
        priority = prepared.observed_values(present).sum(axis=1, min_count=len(present))
        observed[gas + "_generation"] = generation
        observed[gas + "_priority"] = priority

    names = list(C.GAS_HOLDER["capacity_by_column_m3"])
    scales = {}
    for name, capacity in C.GAS_HOLDER["capacity_by_column_m3"].items():
        raw = prepared.observed_values([name])[name].dropna()
        maximum = float(raw.max()) if len(raw) else np.nan
        candidates = [10. ** exponent for exponent in range(-3, 7)
                      if np.isfinite(maximum) and maximum > 0. and maximum * 10. ** exponent <= capacity]
        default = C.OPT_CONFIG.get("holder_scale_to_m3", {}).get(name)
        scale = default if default is not None else (max(candidates) if candidates else None)
        if scale is not None and np.isfinite(maximum) and maximum * scale > capacity:
            scale = max(candidates) if candidates else None
        scales[name] = {"scale_to_m3": scale, "status": "declared_default_inferred", "verified": False,
            "raw_max_in_training_prefix": maximum if np.isfinite(maximum) else None,
            "capacity_m3": capacity,
            "source": "official_capacity_and_legal_raw_prefix; not_sensor_unit_proof"}
    stock = prepared.observed_values(names).mul(
        [np.nan if scales[n]["scale_to_m3"] is None else scales[n]["scale_to_m3"] for n in names], axis=1)
    storage_change = (stock.shift(-1) - stock).sum(axis=1, min_count=2) / (C.STEP_MINUTES / 60.)
    observed["blast_storage_change_m3h"] = storage_change
    observed["blast_generation_13"] = prepared.observed_values(
        [f"gas_blast_furnace_{i}" for i in (1, 2, 3)]).sum(axis=1, min_count=3)
    observed["blast_generation_45"] = prepared.observed_values(
        [f"gas_blast_furnace_{i}" for i in (4, 5)]).sum(axis=1, min_count=2)
    use = resource_series(prepared)
    for gas in C.RESOURCE_NAMES:
        net_label = use[gas].astype(float)
        if gas == "blast":
            net_label = net_label + storage_change
        observed[gas + "_net_label"] = net_label
        # 未观测用户/放散/计量残差；负值表示表计或库存口径不一致，不能强行当零。
        observed[gas + "_residual"] = observed[gas + "_generation"] - observed[gas + "_priority"] - net_label

    window_days = int(C.OPT_CONFIG["net_supply_history_days"])
    recent = observed.loc[observed.index[-1] - pd.Timedelta(days=window_days):].copy()
    slots = _slot_numbers(recent.index)
    profile = recent.groupby(slots).median().reindex(range(96)).fillna(recent.median())
    quantile = float(C.OPT_CONFIG["net_supply_residual_quantile"])
    residual_high = {}
    for gas in C.RESOURCE_NAMES:
        column = gas + "_residual"
        series = recent[column].groupby(slots).quantile(quantile).reindex(range(96))
        residual_high[column] = series.fillna(float(recent[column].quantile(quantile))).tolist()
    history_columns = (["blast_generation_13", "blast_generation_45", "blast_storage_change_m3h"]
        + [gas + "_" + part for gas in C.RESOURCE_NAMES for part in ("generation", "priority", "residual")])
    history_columns = [c for c in history_columns if c in recent]
    history_frame = recent[history_columns].reset_index()
    history_frame.columns = ["datetime"] + history_columns
    history_rows = []
    for _, row in history_frame.iterrows():
        history_rows.append([str(row["datetime"])] + [
            None if not np.isfinite(float(row[c])) else float(row[c]) for c in history_columns])
    pack = {"method": "causal_component_slot_profile_q75", "train_end": str(prepared.index[-1]),
        "profile": {column: [float(v) if np.isfinite(v) else None for v in profile[column]]
                    for column in profile},
        "residual_high": residual_high,
        "history": {"columns": ["datetime"] + history_columns, "rows": history_rows},
        "observed_rows": {column: int(recent[column].notna().sum()) for column in recent},
        "unknown_users": unknown, "converter_scale": C.CONVERTER_GAS_SCALE,
        "flow_units_verified": False, "holder_scale_hypotheses": scales,
        "holder_inlet_basis": C.OPT_CONFIG["holder_inlet_split"],
        "reservation_role": "inferred_unobserved_users_vent_inventory_and_meter_reserve; not_observed_user_truth",
        "validation_scope": "known_component_A_B_C_coverage_and_Oct01_07_error_screen; not_true_net_resource_accuracy",
        "deployment_scope": "simulation_under_declared_inferred_units_and_demand",
        "literature": ["10.1016/j.compchemeng.2019.01.025", "10.1016/j.apenergy.2023.121603"]}
    pack["resource_error_calibration"] = _calibrate_net_error(prepared, observed, scales)
    pack["resource_error_validation"] = net_supply_safety_check(pack)
    return pack


def _observed_net_frame(prepared: PreparedData) -> pd.DataFrame:
    """按训练画像同一口径构造测试期已观测前缀，供每日因果更新使用。

    只允许调用方把 `issued_at` 之前的行并入；这里保留全窗口以便一次性构造，
    发布边界由 `_net_supply_history` 截断。
    """
    observed = pd.DataFrame(index=prepared.index)
    specifications = {
        "blast": (tuple(f"gas_blast_furnace_{i}" for i in range(1, 6)),
                  tuple(f"gas_air_heater_{i}" for i in range(1, 6)) +
                  tuple(f"gas_user_blast_furnace_user{i}" for i in range(1, 5)),
                  "gas_into_gas_mixed_blast_furnace"),
        "coke": (("gas_coke_oven_1",), (), "gas_into_gas_mixed_coke"),
        "converter": (("gas_converter_1",),
                      tuple(sorted({c for c in prepared.frame
                                    if c.startswith("gas_user_converter_user")})),
                      "gas_into_gas_mixed_converter"),
    }
    for gas, (generation_columns, user_columns, mixed) in specifications.items():
        present = [c for c in (*user_columns, mixed) if c in prepared.frame
                   and prepared.observed_values([c])[c].notna().any()]
        generation = prepared.observed_values(generation_columns).sum(axis=1, min_count=len(generation_columns))
        if gas == "converter":
            generation = generation * C.CONVERTER_GAS_SCALE
        priority = prepared.observed_values(present).sum(axis=1, min_count=len(present))
        observed[gas + "_generation"] = generation
        observed[gas + "_priority"] = priority
    names = list(C.GAS_HOLDER["capacity_by_column_m3"])
    scales = [C.OPT_CONFIG.get("holder_scale_to_m3", {}).get(name) for name in names]
    stock = prepared.observed_values(names).mul(scales, axis=1)
    storage_change = (stock.shift(-1) - stock).sum(axis=1, min_count=2) / (C.STEP_MINUTES / 60.)
    observed["blast_storage_change_m3h"] = storage_change
    observed["blast_generation_13"] = prepared.observed_values(
        [f"gas_blast_furnace_{i}" for i in (1, 2, 3)]).sum(axis=1, min_count=3)
    observed["blast_generation_45"] = prepared.observed_values(
        [f"gas_blast_furnace_{i}" for i in (4, 5)]).sum(axis=1, min_count=2)
    use = resource_series(prepared)
    for gas in C.RESOURCE_NAMES:
        net_label = use[gas].astype(float)
        if gas == "blast":
            net_label = net_label + storage_change
        observed[gas + "_net_label"] = net_label
        observed[gas + "_residual"] = observed[gas + "_generation"] - observed[gas + "_priority"] - net_label
    return observed


def forecast_net_supply(pack: dict, issued_at, observed: pd.DataFrame | None = None) -> dict:

    """因果净资源预测：同日时段画像 + 最近偏移 + 残差高分位。

    高炉进气按 1#/2#/3# 与 4#/5# 分组拆分；机组总耗气仍作为一个共享抽气量，
    两柜出口由独立库存平衡在 MILP 内决定。残差取高分位，避免把历史平均
    未观测用户全部当作可利用余气。
    """
    issue = pd.Timestamp(issued_at)
    effective = pd.date_range(issue + pd.Timedelta(minutes=C.STEP_MINUTES), periods=96, freq="15min")
    slots = effective.hour * 4 + effective.minute // 15
    if pack.get("method") != "causal_component_slot_profile_q75":
        # 兼容旧冻结画像：旧方法仍按原口径运行，但不作为新方案默认。
        blocks = []
        for component in ("generation", "priority", "residual"):
            blocks.append(np.column_stack([np.asarray(pack["profile"][gas + "_" + component], dtype=float)[slots]
                                           for gas in C.RESOURCE_NAMES]))
        boundary = net_resource_boundary(blocks[0], blocks[1], np.maximum(blocks[2], 0.), issued_at=issue,
            training_end=pack["train_end"], flow_units_verified=pack["flow_units_verified"])
        boundary["model"] = pack["method"]
        return boundary

    history = _net_supply_history(pack, observed=observed, issue=issue)
    if history is None or len(history) < 96:
        raise ValueError("净资源历史画像不足，不能构造因果预测")
    # 发布时刻之后的行即使在训练前缀内也不得进入本次画像或最近偏移。
    history = history.loc[history.index <= issue]
    if len(history) < 96:
        raise ValueError("发布时刻之前的净资源历史不足，不能构造因果预测")
    recent_steps = int(C.OPT_CONFIG["net_supply_recent_steps"])
    quantile = float(C.OPT_CONFIG["net_supply_residual_quantile"])
    components, low_net = {}, {}
    for gas in C.RESOURCE_NAMES:
        generation = _causal_component_forecast(history, gas + "_generation", slots, recent_steps=recent_steps)
        priority = _causal_component_forecast(history, gas + "_priority", slots, recent_steps=recent_steps)
        residual_mid = _causal_component_forecast(history, gas + "_residual", slots, recent_steps=recent_steps)
        residual_high = _causal_component_forecast(history, gas + "_residual", slots, quantile=quantile,
                                                   recent_steps=recent_steps)
        residual_mid = np.maximum(residual_mid, 0.)
        residual_high = np.maximum(residual_high, 0.)
        mid = generation - priority - residual_mid
        low = generation - priority - residual_high
        components[gas] = {"generation_m3h": generation, "priority_m3h": priority,
            "residual_mid_m3h": residual_mid, "residual_high_m3h": residual_high,
            "net_mid_m3h": mid, "net_low_m3h": low}
        low_net[gas] = low

    generation_13 = _causal_component_forecast(history, "blast_generation_13", slots, recent_steps=recent_steps)
    generation_45 = _causal_component_forecast(history, "blast_generation_45", slots, recent_steps=recent_steps)
    total_blast = generation_13 + generation_45
    share_13 = np.divide(generation_13, total_blast, out=np.full(len(slots), 0.6), where=total_blast > 0.)
    share_13 = np.clip(share_13, 0.50, 0.70)

    generation = np.column_stack((total_blast,
        components["coke"]["generation_m3h"], components["converter"]["generation_m3h"]))
    # 净资源画像未覆盖未观测用户，首日又缺少测试发生量观测；按可配置安全系数
    # 收紧高炉可用量，避免把历史高发生量直接当成可发电煤气抽空柜位。
    generation[:, 0] *= float(getattr(C, "OPT_CONFIG", {}).get("net_supply_safety_factor", 1.0))
    if issue <= pd.Timestamp(pack["train_end"]):
        # 首份计划完全没有测试期观测，且历史画像容易把近期高发生量外推；
        # 首日单独收紧，避免第一份指令就抽空低柜位。
        generation[:, 0] *= float(C.OPT_CONFIG.get("net_supply_first_issue_factor", 1.0))
    priority = np.column_stack([components[gas]["priority_m3h"] for gas in C.RESOURCE_NAMES])
    reservation = np.column_stack([components[gas]["residual_high_m3h"] for gas in C.RESOURCE_NAMES])
    # 无储气的气种只能先分配可供给的生产需求。分别保留请求量及供给缺口，
    # 防止独立预测的需求高于发生量时，把负净量直接送入发电求解器。
    for index in (1, 2):
        gas = C.RESOURCE_NAMES[index]
        requested_priority = priority[:, index].copy()
        requested_reservation = reservation[:, index].copy()
        priority[:, index] = np.minimum(requested_priority, generation[:, index])
        remainder = np.maximum(generation[:, index] - priority[:, index], 0.)
        reservation[:, index] = np.minimum(requested_reservation, remainder)
        components[gas].update(
            priority_shortfall_m3h=requested_priority - priority[:, index],
            reservation_unfunded_m3h=requested_reservation - reservation[:, index])
    # 两柜进气之和必须等于净资源边界中的高炉可用量，否则逐柜守恒会与总量矛盾。
    available_blast = generation[:, 0] - priority[:, 0] - reservation[:, 0]
    inlet_total = np.maximum(available_blast, 0.)
    inlet = np.column_stack((inlet_total * share_13, inlet_total * (1. - share_13)))
    boundary = net_resource_boundary(generation, priority, reservation, issued_at=issue,
        training_end=pack["train_end"], flow_units_verified=pack["flow_units_verified"],
        holder_inlet_m3h=inlet, forecast_method=pack["method"])
    boundary["model"] = pack["method"]
    boundary["forecast_components"] = {
        gas: {key: [float(v) for v in value] for key, value in item.items()}
        for gas, item in components.items()}
    boundary["holder_inlet_basis"] = "blast_furnace_1_3_to_30e4_holder; blast_furnace_4_5_to_20e4_holder"
    boundary["forecast_scope"] = "causal_slot_profile_with_recent_offset_and_residual_quantile"
    return boundary


def _prime_first_effective_state(prepared, position, pack, initial, observed=None):
    """首条指令生效前仍执行既有耗气；只按发布时刻已知量推进这一个时段。

    允许首条指令前的实测柜位已在安全区外，但必须在容量内；不把越界初值裁成
    安全值再求解，也不把“刚恢复”误报成历史合规。
    """
    names = list(C.GAS_HOLDER["capacity_by_column_m3"])
    issue = prepared.index[position]
    old_gas = prepared.observed_values(["load_generator_use_blast_furnace_gas"]).iloc[position, 0]
    if not np.isfinite(old_gas) or old_gas < 0.:
        raise ValueError("首条优化指令生效前缺少当前已知高炉耗气，不能构造桥接库存")
    boundary = forecast_net_supply(pack, issue, observed=observed)
    net = float(np.asarray(boundary["available_m3h"], float)[0, 0])
    inlet = boundary.get("holder_inlet_m3h")
    capacity = np.asarray(list(C.GAS_HOLDER["capacity_by_column_m3"].values()), float)
    states = np.asarray([initial[n] for n in names], float)
    low, high = capacity * C.GAS_HOLDER["safe_low_ratio"], capacity * C.GAS_HOLDER["safe_high_ratio"]
    if not np.isfinite(states).all() or (states < -C.OPT_CONFIG["holder_tolerance_m3"]).any() or (states > capacity + C.OPT_CONFIG["holder_tolerance_m3"]).any():
        raise ValueError("首次发布的原始柜位超出气柜容量")
    volume = (net - old_gas) * (C.STEP_MINUTES / 60.)
    basis = "causal_net_forecast_and_observed_committed_use; inferred_holder_inlet_split"
    pair = prepared.observed_values(names).iloc[max(position - 1, 0):position + 1]
    scales = np.array([pack["holder_scale_hypotheses"][n]["scale_to_m3"] for n in names], float)
    if len(pair) == 2 and pair.notna().all().all() and pair.index[1] - pair.index[0] == pd.Timedelta(minutes=C.STEP_MINUTES):
        # 只外推指令生效前的一步供需差；不把早前柜位变化配给未来新指令。
        allocation = pair.diff().iloc[-1].to_numpy(float) * scales
        volume = float(allocation.sum())
        net = old_gas + volume / (C.STEP_MINUTES / 60.)
        basis = "latest_complete_observed_holder_drift_persistence; inferred_one_step_state"
    else:
        if inlet is not None:
            weights = np.asarray(inlet[0], float)
            weights = np.maximum(weights, 0.) if volume >= 0. else np.maximum(-weights, 0.)
        else:
            weights = capacity.copy()
        if not np.isfinite(weights).all() or weights.sum() <= 0.:
            weights = capacity.copy()
        weights = weights / weights.sum()
        room = capacity - states if volume >= 0. else states
        desired = volume * weights
        allocation = np.minimum(desired, room) if volume >= 0. else -np.minimum(-desired, room)
        remaining = volume - allocation.sum()
        if abs(remaining) > C.OPT_CONFIG["holder_tolerance_m3"]:
            free = room - np.abs(allocation)
            if free.sum() <= C.OPT_CONFIG["holder_tolerance_m3"]:
                raise ValueError("指令生效前的既有耗气无法在气柜容量内维持")
            extra = remaining * free / free.sum()
            allocation = allocation + extra
    after = states + allocation
    if (after < -C.OPT_CONFIG["holder_tolerance_m3"]).any() or (after > capacity + C.OPT_CONFIG["holder_tolerance_m3"]).any():
        raise ValueError("首条指令生效前的一步库存预测越过气柜容量")
    inside_safe = bool(((states >= low - C.OPT_CONFIG["holder_tolerance_m3"]) & (states <= high + C.OPT_CONFIG["holder_tolerance_m3"])).all())
    after_safe = bool(((after >= low - C.OPT_CONFIG["holder_tolerance_m3"]) & (after <= high + C.OPT_CONFIG["holder_tolerance_m3"])).all())
    return dict(zip(names, after.tolist())), {"issued_at": str(issue),
        "first_effective_at": str(issue + pd.Timedelta(minutes=C.STEP_MINUTES)),
        "interval_semantics": "command_start_to_next_step", "previous_blast_use_m3h": float(old_gas),
        "predicted_net_blast_m3h": net, "initial_holders_m3": dict(initial),
        "first_effective_holders_m3": dict(zip(names, after.tolist())),
        "initial_inside_safe_band": inside_safe, "first_effective_inside_safe_band": after_safe,
        "holder_inlet_basis": boundary.get("holder_inlet_basis", "unavailable"),
        "basis": basis, "future_observations_used": False, "field_safety_certified": False}


def physical_input_audit(prepared: PreparedData, issued_at) -> dict:
    """逐字段核对原始观测和单位依据；发布时间之后的记录不参与初态判断。"""
    issue = pd.Timestamp(issued_at)
    stop = int(prepared.index.searchsorted(issue, side="right"))
    if stop == 0:
        return {"passed": False, "problems": ["发布时间之前没有历史输入"], "fields": {}, "holders": {}}
    prefix = prepared.prefix_copy(issue)
    columns = [name for name in C.OBSERVED_AT_T if not name.startswith("load_generator_")]
    columns += list(C.GAS_HOLDER["capacity_by_column_m3"])
    columns = list(dict.fromkeys(columns))
    raw = prefix.observed_values(columns)
    fields, holders = {}, {}
    for name in columns:
        values = raw[name].dropna()
        storage = name in C.GAS_HOLDER["capacity_by_column_m3"]
        last = values.index[-1] if len(values) else None
        fields[name] = {"raw_column": name.split("_", 1)[1],
            "standard_unit": "m3" if storage else "m3/h", "raw_unit": None,
            "scale_to_standard": None, "raw_unit_verified": False,
            "candidate_scale": C.CONVERTER_GAS_SCALE if name == "gas_converter_1" else 1.,
            "scale_basis": "historical_inferred" if name == "gas_converter_1" else "unverified_direct_read",
            "standard_unit_source": "official_reply_20261026; AGENTS.md section12",
            "raw_unit_source": "data/first_train/data_dictionary.xlsx; no_sensor_scale_specification",
            "first_observed_at": str(values.index[0]) if len(values) else None,
            "last_observed_at": str(last) if last is not None else None,
            "observation_age_steps": int((issue - last) / pd.Timedelta(minutes=C.STEP_MINUTES)) if last is not None else None,
            "valid_raw_rows": len(values), "raw_fraction": float(len(values) / stop),
            "source_role": "raw_observation; imputed_values_excluded"}
        if storage:
            holders[name] = {"capacity_m3": C.GAS_HOLDER["capacity_by_column_m3"][name],
                "initial_m3": None, "raw_value_at_issue": float(raw[name].iloc[-1]) if pd.notna(raw[name].iloc[-1]) else None,
                "raw_unit_verified": False, "initial_state_verified": False,
                "last_observed_at": fields[name]["last_observed_at"],
                "observation_age_steps": fields[name]["observation_age_steps"],
                "reason": "原始单位或换算依据未明确，不将读数换成确定库存"}
    return {"passed": False, "issued_at": str(issue), "read_through": str(prefix.index[-1]),
        "fields": fields, "holders": holders, "step_volume_conversion": "m3_per_step = m3_per_hour * 0.25",
        "problems": ["原始过程量和柜位缺少逐列换算依据", "两柜没有经单位核验的独立初态"],
        "future_rows_used": False, "unknown_state_filled": False}


def dispatch_readiness(bundle: ResourceBundle, prepared: PreparedData, first_issue,
                       use_final: bool = False) -> dict:
    """列出正式物理核验缺口；缺依据仍可按已声明假设继续代理模拟。"""
    audit = physical_input_audit(prepared, first_issue)
    heat = bundle.final_heat_rate if use_final else bundle.heat_rate
    publication = resource_publication_check(bundle, first_issue, use_final)
    physical = bundle.final_physical_validation if use_final else bundle.physical_validation
    ledger = physical.get("priority_user_supply", {}).get("gas_ledger", {})
    checks = {"training_before_first_issue": publication["passed"],
        "gas_to_power_validation": bool(heat.get("ok") and heat.get("prefix_validation", {}).get("passed")),
        "net_resource_model": bool(use_final and bundle.final_net_supply is not None),
        "upstream_units": all(item["raw_unit_verified"] for item in audit.get("fields", {}).values()) and bool(audit.get("fields")),
        "independent_holder_initials": all(item["initial_state_verified"] for item in audit.get("holders", {}).values()) and len(audit.get("holders", {})) == 2}
    reasons = {"training_before_first_issue": "; ".join(publication["problems"]),
        "gas_to_power_validation": "归档气电映射未通过合法训练期留出验证",
        "net_resource_model": "归档资源目标为历史机组耗气，缺少已验收的发生量、优先需求和推测预留量预测，不能作为净余气",
        "upstream_units": "原始过程量换算未逐列核验；converter_1倍率仍为inferred",
        "independent_holder_initials": "两柜独立初态与原始单位未知，禁止中点初始化、裁剪或仅控制二号柜后放行"}
    blocked = [name for name, passed in checks.items() if not passed]
    return {"passed": not blocked, "checks": checks, "blocked_items": blocked,
        "problems": [reasons[name] for name in blocked], "input_audit": audit,
        "resource_publication_check": publication,
        "resource_target_semantics": ("inferred_net_surplus_after_priority_users"
            if use_final and bundle.final_net_supply is not None else "historical_consumption_envelope"),
        "net_resource_identifiability": {gas: {key: item.get(key) for key in (
            "unavailable_columns", "raw_complete_rows", "all_users_enumerated", "unit_basis")}
            for gas, item in ledger.items()},
        "model_assumption_validation_passed": False, "field_safety_certified": False}


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
        "group_power_semantics": "controllable_group_gas_allocation; not_unique_regression_from_total_gas",
        "planned_allocation_validation": "recorded_per_daily_plan_after_solving",
        "separate_group_efficiency_measured": False,
        "official_model_trigger": "load_deviation_correction_or_missing_load_inference",
        "reason": ("总负荷经验映射已核验；分组供气按控制量闭合，两组独立效率缺少观测，仍为共享经验效率假设"
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
            "raw_unit_verified": False,
            "raw_unit_status": "assumed_m3_unverified" if col == C.GAS_HOLDER["column"] else "unidentified",
            "inside_capacity_fraction": float(np.mean((x[valid] >= 0.) & (x[valid] <= capacity)))
            if valid.any() and capacity is not None and col == C.GAS_HOLDER["column"] else None,
            "inside_official_safety_fraction": float(np.mean((x[valid] >= low) & (x[valid] <= high)))
            if valid.any() and capacity is not None and col == C.GAS_HOLDER["column"] else None}
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
        "columns": holders, "deployed_column_mapping_passed": False,
        "deployed_column_range_check_passed": bool(holder_ok),
        "all_holder_safety_verified": False,
        "reason": ("2号柜按m³读取的量级检查通过，尚非独立单位核验；1号柜单位与状态不可用，不能证明两柜安全"
                   if holder_ok else "实际部署气柜的容量、量级或数值范围核验未通过"),
        "holder_1_scale_verified": False,
        "holder_1_deployed": False,
        "holder_1_absolute_status": "unknown_unit; cannot_classify_as_observed_low_inventory",
        "holder_1_storage_offered_to_optimizer_m3": 0.}

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
        "dispatch_consumption_unit_confirmed": True,
        "empirical_conversion_uses_raw_generation_scale": False,
        "raw_generation_role": "inferred_scaled_sensor_feature_and_diagnostic_ledger; not_a_direct_flow_constraint",
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
           series: np.ndarray, weights: np.ndarray | None = None, design: np.ndarray | None = None
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from main_s import stack_features
    X = (stack_features(base_values, cal, list(cal), positions, horizons)
         if design is None else design)
    y = series[positions[None, :] + np.asarray(horizons)[:, None]].astype(np.float64).ravel()
    w = (np.ones(len(y), dtype=np.float32) if weights is None else weights.T.astype(np.float32).ravel())
    valid = np.isfinite(y)
    if valid.all():
        return X, y, w
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
    from main_s import fit_booster, stack_features
    design = stack_features(base_values, cal, list(cal), positions, horizons)
    models = {}
    for gas in C.progress(gases, desc="资源模型拟合", unit="气种"):
        X, y, weights = _stack(prepared, base_values, cal, names, positions,
                               horizons, series[gas].to_numpy(dtype=float), decay, design=design)
        gas_params = {**params}
        if objectives:
            gas_params["objective"] = objectives[gas]
        logging.getLogger("aic-half").info("资源拟合: gas=%s objective=%s rows=%d",
                                          gas, gas_params["objective"], len(y))
        # 资源损失保留原始时间权重，不能套用负荷的逆真值加权。
        models[gas] = fit_booster(X, y, names, gas_params, num_round, weights, mape_weight=False)
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
    heat = bundle.final_heat_rate if use_final else bundle.heat_rate
    if adjustment and heat.get("ok"):
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
    y = prepared.labels["generator_all"].iloc[:end_pos].to_numpy(dtype=float)
    columns = ["load_generator_use_blast_furnace_gas", "load_generator_use_coke_gas",
               "load_generator_use_converter_gas"]
    # 物理标定需要真实耗气与负荷配对，预测特征中的前向补值不能冒充实测。
    X = prepared.observed_values(columns).iloc[:end_pos].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1) & (X >= 0.).all(axis=1)
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
                          passed=bool(hold_r2 >= C.HEAT_RATE["min_r2"]
                              and np.abs(hold_pred - hold_y).sum() / max(hold_y.sum(), 1e-9) <= C.HEAT_RATE["max_validation_wape"]
                              and abs((hold_pred - hold_y).sum() / max(hold_y.sum(), 1e-9)) <= C.HEAT_RATE["max_validation_abs_bias"]))
        validation["load_groups"] = {}
        for name, lower, upper in (("zero", 0., 1e-9), ("low", 1e-9, 220.), ("high", 220., np.inf)):
            active = (hold_y >= lower) & (hold_y < upper)
            errors = hold_pred[active] - hold_y[active]
            validation["load_groups"][name] = {"rows": int(active.sum()),
                "mae_mw": float(np.abs(errors).mean()) if len(errors) else None,
                "bias_mw": float(errors.mean()) if len(errors) else None}
        ok = ok and validation["passed"]
    else:
        validation["passed"] = False
        ok = False
    observed = X[valid]
    shares = observed / np.maximum(observed.sum(axis=1, keepdims=True), 1e-9)
    domain = {"method": "observed_flow_and_mixture_ranges; necessary_not_sufficient_support",
        "flow_min_m3h": observed.min(axis=0).tolist(), "flow_max_m3h": observed.max(axis=0).tolist(),
        "share_min": shares.min(axis=0).tolist(), "share_max": shares.max(axis=0).tolist(),
        "rows": len(observed), "training_end": str(prepared.index[end_pos - 1]),
        "basis": "official_training_observations_only", "official_anomaly_threshold": False,
        "literature": "10.1007/s11081-021-09608-0"}
    return {"ok": ok, "coefficients_mw_per_m3h": coef, "r2": r2,
            "operating_domain": domain,
            "calibration_end": str(prepared.index[end_pos - 1]),
            "coefficient_unit_status": "empirical_MW_per_generator_m3h; raw_generation_scaling_checked_separately",
            "prefix_validation": validation,
            "valid_rows": int(valid.sum()), "method": C.HEAT_RATE["method"],
            "note": "合法训练前缀标定的气电综合系数，不分别识别热值与效率；标准耗气单位为m³/h，原始发生量缩放另行核验"}


def gas_power_domain_check(gas, heat_rate: dict, config=C) -> dict:
    """检查优化配气是否超出经验样本范围，不把范围内等同于物理关系已识别。"""
    flows = np.asarray(gas, dtype=float)
    domain = heat_rate.get("operating_domain", {})
    if (flows.ndim != 2 or flows.shape[0] != 3 or not np.isfinite(flows).all()
            or not all(key in domain for key in ("flow_min_m3h", "flow_max_m3h", "share_min", "share_max"))):
        return {"passed": False, "reason": "缺少可用的冻结气电适用范围或配气数组"}
    low, high, share_low, share_high = [np.asarray(domain[key], dtype=float)[:, None]
        for key in ("flow_min_m3h", "flow_max_m3h", "share_min", "share_max")]
    total = flows.sum(axis=0)
    shares = np.divide(flows, total, out=np.zeros_like(flows), where=total > 0.)
    tol = config.OPT_CONFIG["gas_tolerance_m3h"]
    outside_flow = ((flows < low - tol) | (flows > high + tol)).any(axis=0)
    outside_mix = ((((shares - share_low) * total) < -tol)
        | (((shares - share_high) * total) > tol)).any(axis=0)
    # 全关时气电关系为零恒等式，配比没有定义，不计算停机配比越界。
    outside_mix &= total > tol
    outside_flow &= total > tol
    return {"passed": bool(not (outside_flow | outside_mix).any()), "rows": flows.shape[1],
        "outside_flow_rows": int(outside_flow.sum()), "outside_mixture_rows": int(outside_mix.sum()),
        "scope": "necessary_empirical_support_only; not_official_model_or_anomaly_criterion"}


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
    first_issue = daily_publication_times(masked.index[eval_positions], quick=quick)[0]
    ref_pos = min(ref_pos, int(prepared.index.get_indexer([first_issue])[0]))
    if ref_pos < 0:
        raise ValueError("资源确认折缺少首份每日计划的发布时刻")
    positions = positions[positions + max(C.HORIZONS_L) <= ref_pos]
    num_round = C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG
    selection = copy.deepcopy(C.FROZEN_RESOURCE_RECIPE)
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


def train_final_resources(prepared: PreparedData, quick: bool = False) -> ResourceBundle:
    """只按已部署配方重训首份每日计划之前的资源模型，不重新选择小时控制策略。"""
    import main_nn
    prepared.require_training_data()
    started = time.time()
    net = fit_net_supply(prepared)
    if all(all(value is not None for value in row) for row in net["profile"].values()):
        ref = len(prepared.index) - 1
        heat = estimate_heat_rate(prepared, ref)
        bundle = ResourceBundle({}, {}, [], final_net_supply=net, heat_rate=heat, final_heat_rate=heat,
            final_gas_caps=_gas_caps(prepared, ref), train_end=str(prepared.index[-1]),
            final_train_end=str(prepared.index[-1]), train_seconds=time.time() - started,
            selection={"method": net["method"], "selection_scope": net["deployment_scope"]},
            dispatch_selection={**C.FROZEN_RESOURCE_RECIPE["dispatch_selection"], "publication_time": C.OPT_CONFIG["publication_time"]},
            metrics={"scope": "inferred_generation_priority_and_reservation; not_true_net_accuracy"})
        refresh_physical_validation(bundle, prepared, use_final=True)
        return bundle
    recipe = copy.deepcopy(C.FROZEN_RESOURCE_RECIPE)
    recipe["dispatch_selection"]["publication_time"] = C.OPT_CONFIG["publication_time"]
    ref = len(prepared.index) - 1
    positions = np.arange(C.PROFILE_DAYS * C.STEPS_PER_DAY, ref - max(C.HORIZONS_L) + 1,
                         C.QUICK_TRAIN_STRIDE if quick else C.TRAIN_STRIDE_LONG)
    if not len(positions):
        raise ValueError("资源最终训练前缀不足")
    rounds = C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG
    models, names = _fit_resource_models(prepared, positions, rounds, ref, objectives=recipe["objectives"])
    gases = tuple(recipe["head_selection"]["gases"])
    if gases:
        heads, head_names = _fit_resource_models(prepared, _resource_head_positions(ref, quick), rounds, ref,
            objectives={g: recipe["objectives"][g] for g in gases}, horizons=C.HORIZONS_S, gases=gases)
        if names != head_names:
            raise ValueError("资源长短头特征顺序不一致")
        models.update({gas + "__head": model for gas, model in heads.items()})
    heat = estimate_heat_rate(prepared, ref)
    neural = None
    if heat.get("ok") and recipe["availability_adjustment"].get("method") == "positive_nn_peak":
        neural = main_nn.train_resource_nn(_resource_base(prepared), resource_series(prepared),
                                          ref, heat, load_prices(), quick)
    bundle = ResourceBundle({}, models, names, final_heat_rate=heat, heat_rate=heat,
        final_gas_caps=_gas_caps(prepared, ref), selection=recipe,
        dispatch_selection=recipe["dispatch_selection"], final_resource_nn=neural,
        train_end=str(prepared.index[ref]), final_train_end=str(prepared.index[ref]),
        train_rows=len(positions) * len(C.HORIZONS_L), train_seconds=time.time() - started,
        metrics={"scope": "frozen_recipe_no_D_reassessment"})
    refresh_physical_validation(bundle, prepared, use_final=True)
    if not heat.get("ok"):
        bundle.blocker = "PHYSICAL_MAPPING_UNCERTAIN"
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
    floor = config.OPT_CONFIG.get("planned_inventory_floor_m3")
    if floor is not None:
        low = np.maximum(low, float(floor))
    return low, high


def _aggregate_ramp_groups(config=C):
    """按提交中的小机组聚合负荷和总负荷检查启停，避免单机最小负荷锁死开机。"""
    units = np.asarray(config.OPT_CONFIG["unit_capacity_mw"], dtype=float)
    if config.OPT_CONFIG["ramp_scope"] != "aggregate_generator_1_and_all":
        raise ValueError("未实现的聚合爬坡口径")
    fraction = config.OPT_CONFIG["ramp_fraction_per_min"] * config.STEP_MINUTES
    return [(np.arange(4), float(units[:4].sum() * fraction)),
            (np.arange(len(units)), float(units.sum() * fraction))]


def _valid_initial_unit_power(power, config=C):
    values = np.asarray(power, float)
    caps = np.asarray(config.OPT_CONFIG["unit_capacity_mw"], float)
    tolerance = config.OPT_CONFIG["power_tolerance_mw"]
    return bool(values.shape == caps.shape and np.isfinite(values).all()
        and (values >= -tolerance).all() and (values <= caps+tolerance).all()
        and ((values <= tolerance) | (values >= caps*config.OPT_CONFIG["min_load_ratio"]-tolerance)).all())


def _initial_unit_power(first, total, config=C):
    """聚合实测只确定组内总量；构造合法组合，不把单机开停状态冒充实测。"""
    if not np.isfinite([first, total]).all():
        raise ValueError("缺少有限的机组初始聚合负荷")
    units = np.asarray(config.OPT_CONFIG["unit_capacity_mw"], float)
    values = np.zeros(len(units))
    tolerance = config.OPT_CONFIG["power_tolerance_mw"]
    for indices, aggregate in ((np.arange(4), float(first)), (np.arange(4, 6), float(total-first))):
        if aggregate < -tolerance:
            raise ValueError("初始聚合负荷关系无效")
        if abs(aggregate) <= tolerance:
            continue
        feasible = False
        for mask in sorted(range(1, 1 << len(indices)), key=lambda x: (-x.bit_count(), x)):
            active = indices[[bool(mask & (1 << j)) for j in range(len(indices))]]
            low = units[active]*config.OPT_CONFIG["min_load_ratio"]
            capacity = units[active]
            if low.sum()-tolerance <= aggregate <= capacity.sum()+tolerance:
                share = (aggregate-low.sum())/(capacity-low).sum()
                values[active] = low+share*(capacity-low)
                feasible = True
                break
        if not feasible:
            raise ValueError("初始聚合负荷无法分解为满足额定容量和最低开机负荷的组合")
    if not _valid_initial_unit_power(values, config):
        raise ValueError("机组初态分解未通过逐台负荷核验")
    return values


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


def _independent_holder_plan_bounds(initial, steps, config=C):
    """逐柜计划裕度；初态已在安全区外时允许有限步恢复，但不裁剪实测初值。"""
    capacity = np.asarray(list(config.GAS_HOLDER["capacity_by_column_m3"].values()), float)
    margin = config.OPT_CONFIG["holder_margin_ratio"]
    lower = np.repeat((capacity * (config.GAS_HOLDER["safe_low_ratio"] + margin))[:, None], steps, axis=1)
    upper = np.repeat((capacity * (config.GAS_HOLDER["safe_high_ratio"] - margin))[:, None], steps, axis=1)
    if steps:
        lower[:, 0] = np.minimum(lower[:, 0], initial)
        upper[:, 0] = np.maximum(upper[:, 0], initial)
        safe_low = capacity * config.GAS_HOLDER["safe_low_ratio"]
        safe_high = capacity * config.GAS_HOLDER["safe_high_ratio"]
        recovery = max(1, min(int(config.OPT_CONFIG.get("holder_recovery_steps", 4)), steps))
        for j in range(len(capacity)):
            if initial[j] < safe_low[j]:
                # 恢复路径只承诺到计划裕度，不在第一步把实测低柜位“修”成安全值。
                path = np.linspace(initial[j], lower[j, -1], recovery + 1)[1:]
                lower[j, :recovery] = np.minimum(lower[j, :recovery], path)
                lower[j, :recovery] = np.maximum(lower[j, :recovery], 0.)
            elif initial[j] > safe_high[j]:
                path = np.linspace(initial[j], upper[j, -1], recovery + 1)[1:]
                upper[j, :recovery] = np.maximum(upper[j, :recovery], path)
                upper[j, :recovery] = np.minimum(upper[j, :recovery], capacity[j])
    error_low = np.asarray(config.OPT_CONFIG.get("holder_error_lower_m3", np.zeros(steps)), float)
    error_high = np.asarray(config.OPT_CONFIG.get("holder_error_upper_m3", np.zeros(steps)), float)
    if (error_low.shape != (steps,) or error_high.shape != (steps,)
            or not np.isfinite(error_low).all() or not np.isfinite(error_high).all()
            or (error_low > 0.).any() or (error_high < 0.).any()):
        raise ValueError("库存累计误差边界无效")
    recovering = np.zeros_like(lower, dtype=bool)
    if steps:
        for j in range(len(capacity)):
            if initial[j] < capacity[j] * config.GAS_HOLDER["safe_low_ratio"]:
                recovering[j, :recovery] = True
            elif initial[j] > capacity[j] * config.GAS_HOLDER["safe_high_ratio"]:
                recovering[j, :recovery] = True
    safe_low = capacity[:, None] * config.GAS_HOLDER["safe_low_ratio"] - error_low
    safe_high = capacity[:, None] * config.GAS_HOLDER["safe_high_ratio"] - error_high
    lower = np.where(recovering, lower, np.maximum(lower, safe_low))
    upper = np.where(recovering, upper, np.minimum(upper, safe_high))
    return lower, upper


def _solve_day(avail: np.ndarray, price: np.ndarray, holder0: float,
               initial_power: np.ndarray, caps: np.ndarray, heat_rate: dict,
               config=C, *, conserve_all_gases: bool = False,
               independent_holders: dict | None = None,
               holder_inlet: np.ndarray | None = None) -> dict:
    """确定性 MILP；净资源要求三气种完全守恒，包络模式保留既有代理语义。"""
    n_h = len(price)
    units = config.OPT_CONFIG["unit_capacity_mw"]
    n_u = len(units)
    n_g = len(C.RESOURCE_NAMES)
    beta = np.array([heat_rate["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    holder_names = list(config.GAS_HOLDER["capacity_by_column_m3"]) if independent_holders is not None else [config.GAS_HOLDER["column"]]
    stocks0 = np.array([independent_holders[name] for name in holder_names] if independent_holders is not None else [holder0], dtype=float)
    storage_caps = np.array([config.GAS_HOLDER["capacity_by_column_m3"][name] for name in holder_names]
        if independent_holders is not None else [config.GAS_HOLDER["capacity_m3"]], dtype=float)
    n_s = len(stocks0)
    capacity = float(storage_caps.sum())
    real_low = capacity * config.GAS_HOLDER["safe_low_ratio"]
    real_high = capacity * config.GAS_HOLDER["safe_high_ratio"]
    margin = capacity * float(config.OPT_CONFIG["holder_margin_ratio"])
    low = real_low + margin
    high = real_high - margin
    if independent_holders is None:
        lower_path, upper_path = _holder_plan_bounds(n_h, config)
        storage_low, storage_high = lower_path[None], upper_path[None]
    else:
        storage_low, storage_high = _independent_holder_plan_bounds(stocks0, n_h, config)
        lower_path, upper_path = storage_low.sum(axis=0), storage_high.sum(axis=0)
    if (lower_path > upper_path).any():
        return {"status": "INFEASIBLE", "message": "计划安全边界无可行域"}
    invalid_negative = (avail[:, 1:] < 0).any() if conserve_all_gases else (avail < 0).any()
    if (n_h == 0 or avail.shape != (n_h, n_g) or np.asarray(initial_power).shape != (n_u,)
            or np.asarray(caps).shape != (n_g,) or not np.isfinite(initial_power).all()
            or not _valid_initial_unit_power(initial_power, config)
            or not np.isfinite(caps).all() or (np.asarray(caps) < 0).any()
            or not np.isfinite(beta).all() or (beta <= 0.).any()
            or not np.isfinite(avail).all() or not np.isfinite(price).all()
            or invalid_negative
            or (not (real_low <= holder0 <= real_high) if independent_holders is None
                else (holder0 < -config.OPT_CONFIG["holder_tolerance_m3"]
                      or holder0 > capacity + config.OPT_CONFIG["holder_tolerance_m3"]))):
        return {"status": "INVALID_INPUT", "message": "资源或初始柜位无效，禁止裁剪库存伪造可行性"}

    off_z, off_p = 0, n_u * n_h
    off_q = off_p + n_u * n_h
    off_v = off_q + n_g * n_h
    off_s = off_v + n_g * n_h
    # GPF：柜位对偏好带的双向偏离量，压在目标里软惩罚。
    off_du = off_s + n_s * n_h
    off_dd = off_du + n_s * n_h
    # P4：每机组每步的启停切换指示（连续 [0,1] 即可，取到 0/1 由最小化目标保证）。
    off_sw = off_dd + n_s * n_h
    off_direction = off_sw + n_u * n_h
    # 逐柜抽气/放散变量：提交仍只有三个气种总量，内部按两柜独立平衡分配。
    off_eq = off_direction + (n_h if independent_holders is not None else 0)
    off_ev = off_eq + (n_s * n_h if independent_holders is not None else 0)
    n_var = off_ev + (n_s * n_h if independent_holders is not None else 0)
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
    if conserve_all_gases:
        # 提交只包含耗气与负荷，不靠未提交的自由放散变量构造可行性。
        ub[off_v:off_s] = 0.
    for j in range(n_s):
        for k in range(n_h):
            idx = j * n_h + k
            lb[off_s + idx] = storage_low[j, k]
            ub[off_s + idx] = storage_high[j, k]
            ub[off_du + idx] = storage_caps[j]
            ub[off_dd + idx] = storage_caps[j]
    if independent_holders is not None:
        ub[off_direction:off_eq] = 1.
        integrality[off_direction:off_eq] = 1
        if holder_inlet is None:
            # 旧冻结画像没有高炉分组路由信息；仍按两柜容量比例拆分总进气。
            # 净资源为负时保留负号，表示生产缺口由气柜库存补充，而不是把缺口抹成零。
            weights = storage_caps / storage_caps.sum()
            inlet = np.asarray(avail, float)[:, 0:1] * weights[None, :]
        else:
            inlet = np.asarray(holder_inlet, float)
            if inlet.shape != (n_h, n_s) or not np.isfinite(inlet).all() or (inlet < 0.).any():
                return {"status": "INVALID_INPUT", "message": "气柜进气拆分无效"}
        ub[off_eq:off_ev] = caps[0]
        ub[off_ev:] = caps[0]
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
        if independent_holders is None:
            coeffs = {off_s + j * n_h + k: 1.0 for j in range(n_s)}
            rhs = 0.25 * float(avail[k, 0])
            if k > 0:
                coeffs.update({off_s + j * n_h + k - 1: -1.0 for j in range(n_s)})
            else:
                rhs += holder0
            coeffs[off_q + 0 * n_h + k] = 0.25
            coeffs[off_v + 0 * n_h + k] = 0.25
            add(coeffs, rhs, rhs)
        else:
            # 每柜独立库存：进气按官方高炉分组路由，抽气/放散在柜内独立扣除。
            for j in range(n_s):
                coeffs = {off_s + j * n_h + k: 1.0, off_eq + j * n_h + k: 0.25,
                          off_ev + j * n_h + k: 0.25}
                rhs = 0.25 * float(inlet[k, j])
                if k > 0:
                    coeffs[off_s + j * n_h + k - 1] = -1.0
                else:
                    rhs += float(stocks0[j])
                add(coeffs, rhs, rhs)
            add({off_eq + j * n_h + k: 1.0 for j in range(n_s)} | {off_q + 0 * n_h + k: -1.0}, 0., 0.)
            add({off_ev + j * n_h + k: 1.0 for j in range(n_s)} | {off_v + 0 * n_h + k: -1.0}, 0., 0.)
        # 气柜可提供的最大瞬时流量：不得超过当前库存高于下限的部分。
        draw = {off_q + 0 * n_h + k: 1.0, off_v + 0 * n_h + k: 1.0}
        if k > 0:
            draw.update({off_s + j * n_h + k - 1: -4.0 for j in range(n_s)})
            add(draw, -np.inf, float(avail[k, 0]) - 4.0 * lower_path[k])
        else:
            add(draw, -np.inf, float(avail[k, 0]) - 4.0 * lower_path[k] + 4.0 * holder0)
    for g in (1, 2):
        for k in range(n_h):
            add({off_q + g * n_h + k: 1.0, off_v + g * n_h + k: 1.0},
                float(avail[k, g]) if conserve_all_gases else -np.inf, float(avail[k, g]))
    for j in range(n_s):
        pref_low = storage_caps[j] * float(config.OPT_CONFIG["holder_pref_low_ratio"])
        pref_high = storage_caps[j] * float(config.OPT_CONFIG["holder_pref_high_ratio"])
        for k in range(n_h):
            idx = j * n_h + k
            add({off_s + idx: 1.0, off_dd + idx: 1.0}, pref_low, np.inf)
            add({off_s + idx: 1.0, off_du + idx: -1.0}, -np.inf, pref_high)
            if independent_holders is not None:
                # 单管进出：d=1 表示净进气，d=0 表示净抽气。
                # 净交换 = 进气 - 抽气 - 放散；两柜各自方向，不再共享方向变量。
                # 方向大 M 用该柜容量而非当日耗气上限，否则资源为零时会把
                # “库存补生产缺口”误判为不可行。
                pipe_cap = float(max(storage_caps[j], caps[0], 1.0))
                net_flow = {off_eq + idx: 1., off_ev + idx: 1.,
                            off_direction + k: pipe_cap}
                # d=1: 抽气+放散 <= 进气；d=0: 抽气+放散 >= 进气。
                add(net_flow, -np.inf, float(inlet[k, j]) + pipe_cap)
                add(net_flow, float(inlet[k, j]), np.inf)
                # 单步净交换不超过管道容量。
                add({off_eq + idx: 1., off_ev + idx: 1.},
                    float(inlet[k, j]) - pipe_cap, float(inlet[k, j]) + pipe_cap)
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
    terminal_target = config.OPT_CONFIG.get("terminal_inventory_m3")
    if independent_holders is not None and config.OPT_CONFIG.get("terminal_holder"):
        targets = config.OPT_CONFIG.get("terminal_inventory_by_holder_m3", {})
        for j, name in enumerate(holder_names):
            # 裕度外合法初态保留原值，通过实际库存变化恢复，不能先裁剪再求解。
            target = float(targets.get(name, np.clip(stocks0[j], storage_low[j, -1], storage_high[j, -1])))
            add({off_s + j * n_h + n_h - 1: 1.}, target, target)
    elif config.OPT_CONFIG.get("terminal_holder") and (terminal_target is not None or low <= holder0 <= high):
        target = holder0 if terminal_target is None else float(terminal_target)
        terminal_low = max(lower_path[-1], target)
        terminal_high = min(upper_path[-1], target)
        if terminal_low > terminal_high:
            return {"status": "INFEASIBLE", "message": "安全边界与末端库存约束冲突"}
        add({off_s + n_h - 1: 1.0}, terminal_low, terminal_high)
    for k in range(n_h):
        coeffs = {off_p + i * n_h + k: 1.0 for i in range(n_u)}
        for g in range(n_g):
            coeffs[off_q + g * n_h + k] = -float(beta[g])
        add(coeffs, 0.0, 0.0)

    terminal_cap = config.OPT_CONFIG.get("terminal_power_cap_mw")
    if terminal_cap is not None:
        add({off_p + i * n_h + n_h - 1: 1. for i in range(n_u)}, -np.inf, float(terminal_cap))
    total_cap = config.OPT_CONFIG.get("total_power_cap_mw")
    if total_cap is not None:
        rate = _aggregate_ramp_groups(config)[1][1]
        for k in range(n_h):
            # 初始负荷过高时按合法爬坡逐步降至上限，不瞬间裁剪实际初态。
            ceiling = max(float(total_cap), float(np.sum(initial_power)) - (k + 1) * rate)
            add({off_p + i * n_h + k: 1. for i in range(n_u)}, -np.inf, ceiling)

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
    objective[off_sw:off_direction] = switch_pen
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
    stock_paths = x[off_s:off_du].reshape(n_s, n_h).T
    holder = stock_paths.sum(axis=1)
    vent_total = float(np.sum(vent) * 0.25)
    holder_dev = (np.maximum(x[off_du:off_dd], 0.0)
                  + np.maximum(x[off_dd:off_sw], 0.0)).reshape(n_s, n_h).sum(axis=0)
    result = {"status": "OPTIMAL" if solution.success else "FEASIBLE_TIMEOUT",
            "message": str(solution.message), "power": power, "gas": gas,
            "vent": vent, "holder": holder, "vent_total": vent_total,
            "holder_dev": holder_dev,
            "objective": float(solution.fun),
            "mip_gap": getattr(solution, "mip_gap", None)}
    aggregate_config = config
    if independent_holders is not None:
        aggregate_config = SimpleNamespace(**vars(config))
        aggregate_config.GAS_HOLDER = {**config.GAS_HOLDER, "capacity_m3": capacity}
        aggregate_config.OPT_CONFIG = {**config.OPT_CONFIG, "terminal_holder": False}
        result.update(holders_m3=stock_paths, holder_columns=holder_names,
            holder_exchange_m3h=np.diff(np.vstack((stocks0, stock_paths)), axis=0) / (config.STEP_MINUTES / 60.),
            holder_withdrawal_m3h=x[off_eq:off_ev].reshape(n_s, n_h).T,
            holder_release_m3h=x[off_ev:].reshape(n_s, n_h).T,
            holder_inlet_m3h=inlet,
            holder_recovery={name: {"initial_m3": float(stocks0[j]),
                "first_planned_m3": float(stock_paths[0, j]),
                "required": bool(stocks0[j] < storage_caps[j] * (config.GAS_HOLDER["safe_low_ratio"] + config.OPT_CONFIG["holder_margin_ratio"])
                    or stocks0[j] > storage_caps[j] * (config.GAS_HOLDER["safe_high_ratio"] - config.OPT_CONFIG["holder_margin_ratio"])),
                "initial_was_clipped": False} for j, name in enumerate(holder_names)},
            storage_formulation="independent_variables_in_joint_milp")
    result["validation"] = validate_day(result, avail, holder0, initial_power, caps, heat_rate, aggregate_config,
                                          conserve_all_gases=conserve_all_gases,
                                          allow_outside_initial=independent_holders is not None)
    if heat_rate.get("operating_domain"):
        result["domain_validation"] = gas_power_domain_check(gas, heat_rate, config)
    if not result["validation"]["passed"]:
        return {"status": "INVALID_SOLUTION", "message": str(result["validation"])}
    return result


def validate_day(result: dict, avail: np.ndarray, holder0: float,
                 initial_power: np.ndarray, caps: np.ndarray, heat_rate: dict,
                 config=C, *, conserve_all_gases: bool = False,
                 allow_outside_initial: bool = False) -> dict:
    """按调用方声明的包络或净资源语义核验，不签发真实气网安全结论。"""
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
    initial_inside = bool(low <= holder0 <= high)
    recovery = max(1, min(int(config.OPT_CONFIG.get("holder_recovery_steps", 4)), n))
    start = 0
    if allow_outside_initial and not initial_inside:
        lower_path, upper_path = lower_path.copy(), upper_path.copy()
        lower_path[:recovery] = np.minimum(lower_path[:recovery], holder0)
        upper_path[:recovery] = np.maximum(upper_path[:recovery], holder0)
        start = recovery
    beta = np.asarray([heat_rate["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    previous = np.r_[holder0, holder[:-1]]
    balance = holder - previous - 0.25 * (avail[:, 0] - gas[0] - vent[0])
    energy = power.sum(axis=0) - beta @ gas
    delta = np.diff(np.column_stack([initial_power, power]), axis=1)
    problems = []
    if not _valid_initial_unit_power(initial_power, config):
        problems.append("初始单机负荷越界或低于最低开机负荷")
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
    if conserve_all_gases and (np.abs(gas[1:] + vent[1:] - avail[:, 1:].T) > gt).any():
        problems.append("无柜气种净资源未全部分配至发电或放散")
    if np.max(np.abs(energy)) > pt:
        problems.append("燃料与功率不守恒")
    if (lower_path > upper_path).any() or np.max(np.abs(balance)) > st or (holder[start:] < lower_path[start:] - st).any() or (holder[start:] > upper_path[start:] + st).any():
        problems.append("代理库存平衡或安全边界违规")
    terminal_target = config.OPT_CONFIG.get("terminal_inventory_m3")
    if (config.OPT_CONFIG["terminal_holder"] and (terminal_target is not None or low <= holder0 <= high)
            and abs(holder[-1] - (holder0 if terminal_target is None else terminal_target)) > st):
        problems.append("末端库存不一致")
    terminal_cap = config.OPT_CONFIG.get("terminal_power_cap_mw")
    if terminal_cap is not None and power[:, -1].sum() > terminal_cap + pt:
        problems.append("末步负荷未留跨日可达裕度")
    total_cap = config.OPT_CONFIG.get("total_power_cap_mw")
    if total_cap is not None:
        rate = _aggregate_ramp_groups(config)[1][1]
        envelope = np.maximum(float(total_cap), np.sum(initial_power) - rate * np.arange(1, n + 1))
        if (power.sum(axis=0) > envelope + pt).any():
            problems.append("负荷超出声明的因果保守上限")
    return {"passed": not problems, "problems": problems,
            "power_balance_max_mw": float(np.max(np.abs(energy))),
            "holder_balance_max_m3": float(np.max(np.abs(balance))),
            "scope": "declared_net_supply" if conserve_all_gases else "historical_consumption_envelope_only"}


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
    try:
        previous = _initial_unit_power(g1, total, cfg)
    except ValueError as exc:
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": str(exc)}}
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


def allocate_group_gas(power: np.ndarray, gas: np.ndarray, heat_rate: dict) -> dict:
    """分组供气是控制量；在共享经验效率假设下闭合，不由总耗气强行回归唯一组负荷。"""
    power, gas = np.asarray(power, float), np.asarray(gas, float)
    beta = np.array([heat_rate["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    if (power.ndim != 2 or power.shape[0] != 6 or gas.shape != (3, power.shape[1])
            or not all(np.isfinite(x).all() for x in (power, gas, beta))
            or (power < -1e-6).any() or (gas < -1e-6).any() or (beta <= 0.).any()):
        raise ValueError("分组耗气分配输入无效")
    small, total = power[:4].sum(axis=0), power.sum(axis=0)
    if np.max(np.abs(total - beta @ gas)) > C.OPT_CONFIG["power_tolerance_mw"]:
        raise ValueError("总耗气与总功率未闭合，禁止生成分组供气")
    share = np.divide(small, total, out=np.zeros_like(total), where=total > 1e-9)
    grouped = np.stack((gas * share, gas * (1. - share)))
    generated = np.einsum("g,ugn->un", beta, grouped)
    residual = generated - np.stack((small, total - small))
    return {"group_gas_m3h": grouped,
            "validation": {"passed": bool(np.max(np.abs(residual)) <= C.OPT_CONFIG["power_tolerance_mw"]),
                "max_power_residual_mw": float(np.max(np.abs(residual))),
                "basis": "controllable_group_allocation_shared_empirical_conversion",
                "separate_group_efficiency_measured": False}}


def daily_plan_config(holder: float, observed_gas, heat_rate: dict, config=C):
    """发布时间只用当前已知耗气定上限；缺测时回退保守上限，不填造实际流量。"""
    cfg = SimpleNamespace(**vars(config))
    options = config.OPT_CONFIG
    beta = np.array([heat_rate["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    observed = np.asarray(observed_gas, float)
    if observed.shape != (3,):
        raise ValueError("每日耗气观测必须为三气种")
    valid = bool(np.isfinite(observed).all() and (observed >= 0.).all())
    power_cap = (float(np.clip(options["daily_observed_fuel_ratio"] * (observed @ beta),
                              options["daily_base_power_cap_mw"], options["daily_max_power_cap_mw"]))
                 if valid else float(options["daily_base_power_cap_mw"]))
    target = float(options["daily_terminal_inventory_m3"])
    cfg.OPT_CONFIG = {**options, "solver_time_limit_s": 2., "total_power_cap_mw": power_cap,
                      "terminal_power_cap_mw": options["daily_terminal_power_cap_mw"],
                      "terminal_inventory_m3": target, "planned_inventory_floor_m3": min(holder, target)}
    return cfg, {"total_power_cap_mw": power_cap, "terminal_power_cap_mw": options["daily_terminal_power_cap_mw"],
                 "terminal_inventory_m3": target, "observed_power_gas_available": valid,
                 "scope": "A_B_C_proxy_validated_control; not_official_operating_limits"}


def optimize_net_daily(prepared, grid, issue_times, positions, resource_bundle, readiness):
    """净资源和两柜实际接入每日链路；单位及需求假设未确认时仍明确标为模拟。"""
    pack, heat = resource_bundle.final_net_supply, resource_bundle.final_heat_rate
    uncertainty = net_supply_safety_check(pack)
    # 历史误差超出缓冲时保留完整点预测模拟，不把它当成误差集合通过或物理认证。
    use_error_margin = bool(uncertainty["passed"])
    names = list(C.GAS_HOLDER["capacity_by_column_m3"])
    initial, initialization = {}, {}
    for name in names:
        raw = prepared.observed_values([name])[name].iloc[:positions[0] + 1].dropna()
        hypothesis = pack["holder_scale_hypotheses"][name]
        scale = hypothesis["scale_to_m3"]
        if raw.empty or scale is None:
            return {"status": "BLOCKED_OPT", "schedule": None,
                "summary": {"reason": f"{name}无可计算的初态；模拟也不填造库存", "dispatch_readiness": readiness}}
        if raw.index[-1] != issue_times[0]:
            return {"status": "BLOCKED_OPT", "schedule": None,
                "summary": {"reason": f"{name}首次发布时刻缺少新鲜观测，不能把陈旧柜位当作当前状态",
                    "last_observed_at": str(raw.index[-1]), "required_at": str(issue_times[0])}}
        initial[name] = float(raw.iloc[-1] * scale)
        initialization[name] = {"estimated_m3": initial[name], "raw_observed_at": str(raw.index[-1]),
            "raw_value": float(raw.iloc[-1]), "scale_hypothesis": hypothesis,
            "observed_age_minutes": float((issue_times[0] - raw.index[-1]).total_seconds() / 60.),
            "state_role": "inferred_volume_under_unverified_sensor_scale; not_measured_m3"}
    try:
        previous = _initial_unit_power(_last_level(prepared, "generator_1", int(positions[0])),
            _last_level(prepared, "generator_all", int(positions[0])))
    except ValueError as exc:
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": str(exc)}}
    caps = np.array([resource_bundle.final_gas_caps[gas] for gas in C.RESOURCE_NAMES])
    units_verified = {name: pack["holder_scale_hypotheses"][name]["verified"] is True for name in names}
    try:
        observed_net = _observed_net_frame(prepared)
        holder, priming = _prime_first_effective_state(prepared, int(positions[0]), pack, initial,
                                                       observed=observed_net)
    except ValueError as exc:
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": str(exc)}}
    prices, plans, records, stock_paths = load_prices(), [], [], []
    submitted_initial = submitted_previous_power = None
    submitted_available, submitted_gas, submitted_vent, submitted_power = [], [], [], []
    submitted_inlet, submitted_exchange = [], []
    resource_seconds = milp_seconds = max_net_error = 0.
    point_only_days = []
    max_stock_error = dict.fromkeys(names, 0.)
    for day_number, (issue, pos) in enumerate(zip(issue_times, positions)):
        tick = time.perf_counter()
        boundary = forecast_net_supply(pack, issue, observed=observed_net)
        resource_seconds += time.perf_counter() - tick
        observed_gas = prepared.frame[["d_use_blast", "d_use_coke", "d_use_converter"]].iloc[pos].to_numpy(float)
        cfg, control = daily_plan_config(holder[names[1]], observed_gas, heat)
        first_capacity = C.GAS_HOLDER["capacity_by_column_m3"][names[0]]
        lower_error = (np.asarray(uncertainty["lower_error_m3"])+day_number*uncertainty["lower_error_m3"][-1]
            +uncertainty["priming_error_lower_m3"]) if use_error_margin else np.zeros(C.STEPS_PER_DAY)
        upper_error = (np.asarray(uncertainty["upper_error_m3"])+day_number*uncertainty["upper_error_m3"][-1]
            +uncertainty["priming_error_upper_m3"]) if use_error_margin else np.zeros(C.STEPS_PER_DAY)
        cfg.OPT_CONFIG.update(total_power_cap_mw=None, terminal_power_cap_mw=None,
            terminal_inventory_m3=None, planned_inventory_floor_m3=None,
            holder_error_lower_m3=lower_error.tolist(), holder_error_upper_m3=upper_error.tolist(),
            terminal_inventory_by_holder_m3={names[0]: float(np.clip(holder[names[0]],
                first_capacity * (C.GAS_HOLDER["safe_low_ratio"] + C.OPT_CONFIG["holder_margin_ratio"]),
                first_capacity * (C.GAS_HOLDER["safe_high_ratio"] - C.OPT_CONFIG["holder_margin_ratio"]))),
                                            names[1]: C.OPT_CONFIG["daily_terminal_inventory_m3"]})
        control.update(total_power_cap_mw=None, terminal_power_cap_mw=None,
            scope="official_unit_capacity_and_aggregate_ramp; no_proxy_power_cap")
        first = issue + pd.Timedelta(minutes=C.STEP_MINUTES)
        tariff = _price_vector(prices, first, C.STEPS_PER_DAY)
        tick = time.perf_counter()
        result = solve_net_day(boundary, tariff, holder, previous, caps, heat,
                              holder_units_verified=units_verified, config=cfg)
        if result["status"] not in ("OPTIMAL", "FEASIBLE_TIMEOUT") and use_error_margin:
            # 若额外风险裕度无解，仅回退到原有点预测模型；官方气柜、机组硬约束不放宽。
            cfg.OPT_CONFIG.update(holder_error_lower_m3=np.zeros(C.STEPS_PER_DAY).tolist(),
                                  holder_error_upper_m3=np.zeros(C.STEPS_PER_DAY).tolist())
            result = solve_net_day(boundary, tariff, holder, previous, caps, heat,
                                  holder_units_verified=units_verified, config=cfg)
            use_error_margin = False
        if not use_error_margin:
            point_only_days.append(str(issue))
        milp_seconds += time.perf_counter() - tick
        if result["status"] not in ("OPTIMAL", "FEASIBLE_TIMEOUT"):
            return {"status": "BLOCKED_OPT", "schedule": None, "summary": {
                "reason": result.get("message", result["status"]), "failed_issue": str(issue),
                "dispatch_readiness": readiness, "initialization": initialization}}
        allocation = allocate_group_gas(result["power"], result["gas"], heat)
        if not allocation["validation"]["passed"]:
            return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": "分组气电不守恒"}}
        effective = pd.date_range(first, periods=C.STEPS_PER_DAY, freq="15min")
        selected = (effective >= grid[0]) & (effective <= grid[-1])
        published = effective[selected]
        levels = result["holders_m3"]
        if submitted_initial is None:
            first_k = np.flatnonzero(selected)[0]
            # levels 保存每步结束状态；首个提交步的初态取前一步结束值。
            submitted_initial = dict(zip(names, levels[first_k - 1] if first_k else [holder[name] for name in names]))
            submitted_previous_power = (result["power"][:, first_k - 1] if first_k else previous).tolist()
        stock_paths.extend(levels[selected].tolist())
        submitted_available.extend(np.asarray(boundary["available_m3h"])[selected].tolist())
        inlet_source = boundary.get("holder_inlet_m3h")
        if inlet_source is None:
            inlet_source = result.get("holder_inlet_m3h")
        submitted_inlet.extend(np.asarray(inlet_source)[selected].tolist())
        submitted_exchange.extend(np.asarray(result["holder_exchange_m3h"])[selected].tolist())
        submitted_gas.extend(result["gas"][:, selected].T.tolist())
        submitted_vent.extend(result["vent"][:, selected].T.tolist())
        submitted_power.extend(result["power"][:, selected].T.tolist())
        network = result["network_validation"]
        max_net_error = max(max_net_error, network["net_resource_validation"]["balance_max_m3h"])
        for name in names:
            max_stock_error[name] = max(max_stock_error[name], network["independent_holder_validation"]["holders"][name]["balance_max_m3"])
        plans.append({"day": str(issue.date()), "issued_at": str(issue), "steps": 96,
            "effective_start": str(effective[0]), "effective_end": str(effective[-1]),
            "published_steps": len(published), "published_start": str(published[0]), "published_end": str(published[-1]),
            "validation": result["validation"], "solver_status": result["status"],
            "objective": result.get("objective"), "mip_gap": result.get("mip_gap"),
            "network_validation": network, "model_constraints_passed": result["model_constraints_passed"],
            "empirical_domain": result.get("domain_validation"),
            "group_allocation": allocation["validation"], "control": control,
            "initial_holders_m3": dict(holder), "terminal_holders_m3": dict(zip(names, levels[-1].tolist())),
            "holder_recovery": result["holder_recovery"], "resource_mode": boundary["resource_mode"],
            "nonstorage_demand_shortfall": {gas: {
                "priority_shortfall_m3": float(np.sum(component.get("priority_shortfall_m3h", [])) * C.STEP_MINUTES / 60.),
                "reservation_unfunded_m3": float(np.sum(component.get("reservation_unfunded_m3h", [])) * C.STEP_MINUTES / 60.)}
                for gas, component in boundary.get("forecast_components", {}).items() if gas != "blast"}})
        for k in np.flatnonzero(selected):
            power = result["power"][:, k]
            records.append([effective[k], *result["gas"][:, k], power[:4].sum(), power.sum()])
        holder, previous = dict(zip(names, levels[-1].tolist())), result["power"][:, -1].copy()
    schedule = pd.DataFrame(records, columns=["datetime", "opt_generator_use_blast_furnace_gas",
        "opt_generator_use_coke_gas", "opt_generator_use_converter_gas", "opt_generator_1", "opt_generator_all"])
    if not pd.DatetimeIndex(schedule.datetime).equals(grid):
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": "净资源每日计划时间覆盖错误"}}
    paths = np.vstack(([submitted_initial[name] for name in names], stock_paths))
    holders = {}
    network_holders = (result.get("network_validation", {}).get("independent_holder_validation", {})
                       if plans else {}).get("holders", {})
    for j, name in enumerate(names):
        detail = network_holders.get(name, {})
        capacity = C.GAS_HOLDER["capacity_by_column_m3"][name]
        low, high = (capacity * C.GAS_HOLDER["safe_low_ratio"],
                     capacity * C.GAS_HOLDER["safe_high_ratio"])
        first_initial = float(paths[0, j])
        first_inside = bool(low - C.OPT_CONFIG["holder_tolerance_m3"] <= first_initial
                            <= high + C.OPT_CONFIG["holder_tolerance_m3"])
        recovery = 0 if first_inside else max(1, min(int(C.OPT_CONFIG.get("holder_recovery_steps", 4)),
                                                    len(paths) - 1))
        after = paths[recovery:, j]
        safety = bool(((after >= low - C.OPT_CONFIG["holder_tolerance_m3"])
                       & (after <= high + C.OPT_CONFIG["holder_tolerance_m3"])).all())
        holders[name] = {"passed": bool(detail.get("passed", False)), "units_verified": units_verified[name],
            "capacity_m3": capacity,
            "initial_m3": first_initial, "terminal_m3": float(paths[-1, j]),
            "minimum_m3": float(paths[:, j].min()), "maximum_m3": float(paths[:, j].max()),
            "balance_max_m3": max_stock_error[name],
            "initial_inside_safe_band": first_inside,
            "initial_outside_recovery_steps": recovery,
            "safety_passed": safety,
            "plan_margin_passed": bool(detail.get("plan_margin_passed", False)),
            "balance_passed": bool(detail.get("balance_passed", False)),
            "terminal_passed": bool(detail.get("terminal_passed", False))}
    physical = copy.deepcopy(resource_bundle.final_physical_validation)
    physical.setdefault("priority_user_supply", {}).update(forecast_or_dispatch_replaced=True,
        reason="生产需求已扣减，未观测需求及残差预留仍为inferred，缺少真实净资源监督与原始单位确认")
    metrics = _dispatch_metrics(pd.DatetimeIndex(schedule.datetime), schedule.opt_generator_all.to_numpy(), prices)
    answer = {"status": "SIMULATED_NOT_SUBMITTABLE", "submittable": False, "schedule": schedule,
        "summary": {"status": "SIMULATED_NOT_SUBMITTABLE", "resource_mode": "inferred_net_surplus_after_priority_users",
            "dispatch_mode": "daily_fixed", "intraday_refresh": False, "publication_time": C.OPT_CONFIG["publication_time"],
            "publication_schedule_passed": True, "resource_publication_check": readiness["resource_publication_check"],
            "daily_control_policy": "daily_fixed_with_independent_holders_and_inferred_net_supply",
            "daily_control_parameters": {key: C.OPT_CONFIG[key] for key in (
                "daily_terminal_inventory_m3", "daily_terminal_power_cap_mw", "daily_base_power_cap_mw",
                "daily_max_power_cap_mw", "daily_observed_fuel_ratio")},
            "dispatch_readiness": readiness, "daily_publications": len(plans), "daily_plans": plans,
            "resource_error_validation": uncertainty,
            "point_only_publications": point_only_days,
            "uncertainty_scope": "unverified_when_point_only; historical_error_not_a_certificate",
            "initialization": initialization, "planned_steps": len(schedule), "plan_validation_passed": True,
            "first_effective_state": priming,
            "inventory_interval_semantics": "command_start_to_next_step",
            "unit_initialization": {"basis": "feasible_group_decomposition_inferred; actual_unit_switch_states_unknown",
                "initial_unit_power_mw": _initial_unit_power(_last_level(prepared, "generator_1", int(positions[0])),
                    _last_level(prepared, "generator_all", int(positions[0]))).tolist()},
            "group_allocation_passed": True, "plan_feasible": True, "physical_feasibility_certified": False,
            "forecast_uses_historical_holder": False, "physical_validation": physical,
            "independent_holder_validation": {"passed": False, "model_constraints_passed": True, "rows": len(grid), "holders": holders},
            "net_resource_validation": {"passed": False, "arithmetic_passed": True, "rows": len(grid),
                "balance_max_m3h": max_net_error, "flow_units_verified": False},
            "planned_metrics": metrics, "planned_revenue_cny": metrics["generation_revenue_cny"],
            "planned_generation_mwh": metrics["generation_mwh"],
            "time_semantics": "target_time; daily instructions issued before the first effective timestamp",
            "coverage_basis": "same_datetime_as_long_forecast", "replay_complete": False,
            "realized_net_resource_revenue_cny": None, "evidence_gaps_block_simulation": False,
            "simulation_assumptions": {"net_supply": pack["reservation_role"], "holder_units": "inferred_decimal_magnitude_hypotheses",
                "topology": "shared_blast_header_without_opposite_exchange; exchange_limits_unprovided",
                "gas_to_power": "shared_positive_empirical_conversion; holdout_metrics_retained"},
            "model_evidence": {"holders_m3": paths.tolist(), "available_m3h": submitted_available,
                "holder_inlet_m3h": submitted_inlet, "holder_exchange_m3h": submitted_exchange,
                "holder_state_times": [str(t) for t in pd.date_range(grid[0], periods=len(grid)+1, freq="15min")],
                "net_supply_model": pack,
                "gas_m3h": submitted_gas, "vent_m3h": submitted_vent, "unit_power_mw": submitted_power,
                "previous_unit_power_mw": submitted_previous_power,
                "coefficients_mw_per_m3h": heat["coefficients_mw_per_m3h"],
                "declared_defaults": {"holder_scales": pack["holder_scale_hypotheses"],
                    "converter_scale": pack["converter_scale"], "reservation_role": pack["reservation_role"],
                    "gas_to_power": "positive_empirical_inferred", "field_safety_certified": False}},
            "timings": {"resource_inference_seconds": round(resource_seconds, 4), "milp_seconds": round(milp_seconds, 4)}}}
    acceptance = submission_readiness(answer)
    answer["summary"]["submission_readiness"] = acceptance
    if acceptance["passed"]:
        answer.update(status="READY_WITH_ASSUMPTIONS", submittable=True)
        answer["summary"].update(status="READY_WITH_ASSUMPTIONS", validation_basis=acceptance["validation_basis"])
    elif acceptance.get("simulation_passed"):
        answer["summary"].update(validation_basis=acceptance["validation_basis"],
            reason=("调度仅通过点预测及声明假设下的模型硬约束；历史资源误差或正式物理核验未通过，仅可作为模拟"
                    if not acceptance["model"]["passed"] else
                    "调度通过声明假设下的模型核验；正式物理核验尚未通过，仅可作为模拟"))
    elif C.OPT_CONFIG.get("submission_validation_basis") == "declared_assumptions":
        # 导出资格与认证结果分开；保留实际求解结果，核验失败仍在摘要中如实记录。
        answer.update(status="UNVERIFIED_OPTIMIZATION", submittable=False)
        answer["summary"].update(status="UNVERIFIED_OPTIMIZATION", validation_basis="unverified_export",
            reason="已导出调度，待核验项: " + "; ".join(acceptance["problems"]))
    return answer


def optimize_daily(prepared: PreparedData, origins: pd.DatetimeIndex,
                   resource_bundle: ResourceBundle, quick: bool = False,
                   use_final: bool = False) -> dict:
    """按业务时刻发布96步计划，首尾裁取提交网格，日内不刷新指令。"""
    grid = pd.DatetimeIndex(origins)
    if quick:
        grid = grid[:2 * C.STEPS_PER_DAY]
    if (not len(grid) or grid[0] != grid[0].normalize()
            or len(grid) % C.STEPS_PER_DAY or grid.has_duplicates
            or not grid.equals(pd.date_range(grid[0], periods=len(grid), freq="15min"))):
        return {"status": "BLOCKED_OPT", "schedule": None,
                "summary": {"reason": "每日固定发布要求完整、连续的自然日网格"}}
    try:
        issue_times = daily_publication_times(grid)
    except ValueError as exc:
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": str(exc)}}
    positions = prepared.index.get_indexer(issue_times)
    if (positions < 0).any():
        return {"status": "BLOCKED_OPT", "schedule": None,
                "summary": {"reason": "缺少每日计划发布时刻所需的历史起报点"}}
    publication_check = resource_publication_check(resource_bundle, issue_times[0], use_final)
    if not publication_check["passed"]:
        return {"status": "BLOCKED_OPT", "schedule": None,
                "summary": {"reason": "; ".join(publication_check["problems"]),
                            "resource_publication_check": publication_check}}
    readiness = dispatch_readiness(resource_bundle, prepared, issue_times[0], use_final=use_final)
    heat = resource_bundle.final_heat_rate if use_final else resource_bundle.heat_rate
    caps = resource_bundle.final_gas_caps if use_final else resource_bundle.gas_caps
    beta = np.array([heat.get("coefficients_mw_per_m3h", {}).get(g, np.nan) for g in C.RESOURCE_NAMES])
    if not np.isfinite(beta).all() or (beta <= 0.).any():
        return {"status": "BLOCKED_OPT", "schedule": None,
            "summary": {"reason": "缺少有限正值气电系数，无法构造守恒方程", "dispatch_readiness": readiness}}
    if use_final and resource_bundle.final_net_supply is not None:
        return optimize_net_daily(prepared, grid, issue_times, positions, resource_bundle, readiness)
    resource_started = time.perf_counter()
    forecast = forecast_resources(resource_bundle, prepared, issue_times, use_final=use_final)
    resource_seconds = time.perf_counter() - resource_started
    observed = prepared.frame[C.GAS_HOLDER["column"]].iloc[:positions[0] + 1].dropna()
    if observed.empty:
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": "NO_INITIAL_HOLDER"}}
    holder0 = holder = float(observed.iloc[-1])
    first = _last_level(prepared, "generator_1", int(positions[0]))
    total = _last_level(prepared, "generator_all", int(positions[0]))
    if not np.isfinite([first, total]).all():
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": "NO_INITIAL_POWER"}}
    try:
        previous = _initial_unit_power(first, total)
    except ValueError as exc:
        return {"status": "BLOCKED_OPT", "schedule": None, "summary": {"reason": str(exc)}}
    prices, records, plans = load_prices(), [], []
    cap_vector = np.array([caps[g] for g in C.RESOURCE_NAMES])
    submitted_initial_holder = submitted_terminal_holder = None
    milp_started = time.perf_counter()
    for issue, pos, avail in zip(issue_times, positions, forecast):
        observed_gas = prepared.frame[["d_use_blast", "d_use_coke", "d_use_converter"]].iloc[pos].to_numpy(dtype=float)
        cfg, control = daily_plan_config(holder, observed_gas, heat)
        day = issue + pd.Timedelta(minutes=C.STEP_MINUTES)
        price = _price_vector(prices, day, C.STEPS_PER_DAY)
        result = _solve_day(avail, price, holder, previous, cap_vector, heat, cfg)
        if result["status"] not in ("OPTIMAL", "FEASIBLE_TIMEOUT"):
            result = _fallback_day(prepared, int(pos), avail, cap_vector, heat, cfg,
                                   holder0=holder, initial_power=previous)
        if result["status"] not in ("OPTIMAL", "FEASIBLE_TIMEOUT", "FALLBACK_SAFE"):
            return {"status": "BLOCKED_OPT", "schedule": None,
                    "summary": {"reason": result.get("message", result["status"]), "failed_day": str(day)}}
        effective = pd.date_range(day, periods=C.STEPS_PER_DAY, freq=f"{C.STEP_MINUTES}min")
        allocation = allocate_group_gas(result["power"], result["gas"], heat)
        published = effective[(effective >= grid[0]) & (effective <= grid[-1])]
        plans.append({"day": str(issue.date()), "issued_at": str(issue), "steps": C.STEPS_PER_DAY,
                      "effective_start": str(effective[0]), "effective_end": str(effective[-1]),
                      "published_steps": len(published),
                      "published_start": str(published[0]), "published_end": str(published[-1]),
                      "validation": result["validation"], "solver_status": result["status"],
                      "objective": result.get("objective"), "mip_gap": result.get("mip_gap"),
                      "empirical_domain": result.get("domain_validation"),
                      "control": control, "group_allocation": allocation["validation"],
                      "initial_holder_m3": holder, "terminal_holder_m3": float(result["holder"][-1])})
        for k in range(C.STEPS_PER_DAY):
            when = effective[k]
            if when < grid[0] or when > grid[-1]:
                continue
            power, gas = result["power"][:, k], result["gas"][:, k]
            if submitted_initial_holder is None:
                submitted_initial_holder = float(result["holder"][k - 1]) if k else holder
            submitted_terminal_holder = float(result["holder"][k])
            if records:
                for value, prior, (_, limit) in zip((power[:4].sum(), power.sum()), records[-1][-2:], _aggregate_ramp_groups(cfg)):
                    if abs(float(value) - prior) > limit + cfg.OPT_CONFIG["power_tolerance_mw"]:
                        return {"status": "BLOCKED_OPT", "schedule": None,
                                "summary": {"reason": "每日发布计划的聚合爬坡衔接冲突", "failed_day": str(day)}}
            records.append([when, *gas, power[:4].sum(), power.sum()])
        holder, previous = float(result["holder"][-1]), result["power"][:, -1].copy()
    schedule = pd.DataFrame(records, columns=["datetime", "opt_generator_use_blast_furnace_gas",
        "opt_generator_use_coke_gas", "opt_generator_use_converter_gas", "opt_generator_1", "opt_generator_all"])
    times = pd.DatetimeIndex(schedule.datetime)
    if not times.equals(grid):
        return {"status": "BLOCKED_OPT", "schedule": None,
                "summary": {"reason": "每日计划未完整覆盖长期预测时间网格"}}
    metrics = _dispatch_metrics(times, schedule.opt_generator_all.to_numpy(), prices)
    return {"status": "SIMULATED_NOT_SUBMITTABLE", "submittable": False, "schedule": schedule,
        "summary": {"status": "SIMULATED_NOT_SUBMITTABLE", "resource_mode": "historical_consumption_envelope",
            "dispatch_readiness": readiness, "evidence_gaps_block_simulation": False,
            "simulation_assumptions": {"resource": "historical_generation_consumption_is_only_a_proxy_envelope",
                "holder_2": "raw_volume_read_as_m3; independent_sensor_scale_not_certified",
                "holder_1": "not_controlled; absolute_safety_unknown",
                "gas_to_power": "positive_empirical_coefficients; failed_holdout_metrics_retained_as_diagnostics",
                "gas_to_power_holdout_passed": bool(heat.get("ok")),
                "resource_nn_adjustment_applied": bool(heat.get("ok") and resource_bundle.selection.get("availability_adjustment"))},
            "timings": {"resource_inference_seconds": round(resource_seconds, 4),
                        "milp_seconds": round(time.perf_counter() - milp_started, 4)},
            "dispatch_mode": "daily_fixed", "daily_publications": len(plans), "daily_plans": plans,
            "publication_time": C.OPT_CONFIG["publication_time"], "publication_schedule_passed": True,
            "resource_publication_check": publication_check,
            "daily_control_policy": "reserve140_terminal200_observed_fuel_cap240_to280",
            "daily_control_parameters": {key: C.OPT_CONFIG[key] for key in (
                "daily_terminal_inventory_m3", "daily_terminal_power_cap_mw", "daily_base_power_cap_mw",
                "daily_max_power_cap_mw", "daily_observed_fuel_ratio")},
            "proxy_execution_policy": C.OPT_CONFIG["execution_gas_mode"],
            "group_allocation_passed": all(p["group_allocation"]["passed"] for p in plans),
            "group_efficiency_basis": "shared_empirical_inferred; separate_group_efficiency_not_measured",
            "holder1_control": {"inventory_used": False, "controlled_exchange_m3h": 0.,
                                "absolute_safety_certified": False, "reason": "raw_unit_and_initial_volume_unidentified"},
            "plan_feasible": True, "plan_validation_passed": True, "physical_feasibility_certified": False,
            "forecast_uses_historical_holder": False, "planned_steps": len(schedule),
            "initial_holder_m3": submitted_initial_holder, "terminal_holder_m3": submitted_terminal_holder,
            "initial_planning_holder_m3": holder0, "terminal_planning_holder_m3": holder,
            "planned_metrics": metrics, "planned_revenue_cny": metrics["generation_revenue_cny"],
            "planned_generation_mwh": metrics["generation_mwh"],
            "time_semantics": "target_time; daily instructions issued before the first effective timestamp",
            "coverage_basis": "same_datetime_as_long_forecast", "intraday_refresh": False,
            "physical_validation": (resource_bundle.final_physical_validation if use_final else resource_bundle.physical_validation),
            "note": "每日计划固定发布；首尾窗口外计划只参与模型内状态衔接，不计提交收益；库存连续接续，不用历史柜位每日重置。"}}


def export_fallback_schedule(prepared: PreparedData, origins, resource_bundle: ResourceBundle,
                             failed: dict, quick: bool = False) -> dict:
    """完整导出兜底：按发布前的历史或冻结资源预测生成指令，未认证项不阻断归档。"""
    grid = pd.DatetimeIndex(origins)
    if quick:
        grid = grid[:2 * C.STEPS_PER_DAY]
    issues = daily_publication_times(grid)
    heat = resource_bundle.final_heat_rate or resource_bundle.heat_rate
    beta = np.array([heat.get("coefficients_mw_per_m3h", {}).get(g, np.nan)
                     for g in C.RESOURCE_NAMES], float)
    conversion_available = bool(np.isfinite(beta).all() and (beta > 0.).all())
    columns = ["load_generator_use_blast_furnace_gas", "load_generator_use_coke_gas",
               "load_generator_use_converter_gas"]
    history = prepared.observed_values(columns)
    pack = resource_bundle.final_net_supply
    records, plans, warnings = [], [], []
    try:
        observed = _observed_net_frame(prepared) if pack is not None else None
    except Exception as exc:
        observed = None
        warnings.append(f"实时资源画像不可用，仅复用发布前冻结历史: {exc}")
    first_pos = prepared.index.get_indexer([issues[0]])[0]
    try:
        previous = _initial_unit_power(_last_level(prepared, "generator_1", first_pos),
                                       _last_level(prepared, "generator_all", first_pos))
        initial_basis = "causal_observed_aggregate_load_decomposition"
    except (ValueError, IndexError):
        previous = np.zeros(len(C.OPT_CONFIG["unit_capacity_mw"]))
        initial_basis = "declared_off_state_for_export; actual_initial_units_unverified"
        warnings.append("缺少有效初始机组状态，兜底采用明确声明的停机初态")
    warnings.extend(["兜底指令未通过气柜库存及净资源硬约束认证",
                     "经验气电系数未通过的标定项仍保留，导出不代表正式核验通过"])
    if not conversion_available:
        warnings.append("没有有效气电系数，兜底生成零发电与零耗气的停机指令；首步爬坡待核验")
    capacity = float(sum(C.OPT_CONFIG["unit_capacity_mw"]))
    for issue in issues:
        effective = pd.date_range(issue + pd.Timedelta(minutes=C.STEP_MINUTES),
                                   periods=C.STEPS_PER_DAY, freq="15min")
        past = history.loc[:issue].tail(28 * C.STEPS_PER_DAY)
        reference = past.median().to_numpy(float)
        unknown = [gas for gas, value in zip(C.RESOURCE_NAMES, reference) if not np.isfinite(value)]
        if unknown:
            warnings.append(f"{issue}缺少{','.join(unknown)}耗气历史，未知气种不分配发电预算；未填造实际供气")
        reference = np.where(np.isfinite(reference) & (reference >= 0.), reference, 0.)
        basis = "causal_28_day_observed_consumption_median; inferred_dispatch_budget"
        budget = np.repeat(reference[None, :], C.STEPS_PER_DAY, axis=0)
        try:
            if pack is not None:
                boundary = forecast_net_supply(pack, issue, observed=observed)
                predicted = np.asarray(boundary["available_m3h"], float)
                if predicted.shape != budget.shape or not np.isfinite(predicted).all():
                    raise ValueError("冻结资源预测包含缺测或形状错误")
                budget = np.maximum(predicted, 0.)
                basis = "causal_frozen_net_forecast; nonnegative_dispatch_budget"
        except Exception as exc:
            warnings.append(f"{issue}资源预测失败，使用历史推测预算: {exc}")
        day_rows, excess = [], np.zeros(3)
        for when, available in zip(effective, budget):
            if conversion_available:
                wanted = min(float(available @ beta), capacity)
                power = _allocate_aggregate_power(wanted, previous)
                if power is None:
                    # 能源预算与爬坡冲突时保留可达机组状态，并明确记录额外用气，不能伪称资源可行。
                    power = previous.copy()
                mix = available.copy()
                if mix @ beta <= 0.:
                    mix = reference.copy()
                if mix @ beta <= 0.:
                    mix = np.array([1., 0., 0.])
                gas = mix * (float(power.sum()) / float(mix @ beta))
                excess += np.maximum(gas - available, 0.) * C.STEP_MINUTES / 60.
            else:
                power, gas = np.zeros_like(previous), np.zeros(3)
            day_rows.append([when, *gas, float(power[:4].sum()), float(power.sum())])
            previous = power
        selected = [row for row in day_rows if grid[0] <= row[0] <= grid[-1]]
        records.extend(selected)
        plans.append({"issued_at": str(issue), "steps": C.STEPS_PER_DAY,
            "effective_start": str(effective[0]), "effective_end": str(effective[-1]),
            "published_steps": len(selected), "published_start": str(selected[0][0]),
            "published_end": str(selected[-1][0]), "solver_status": "CAUSAL_EXPORT_FALLBACK",
            "forecast_basis": basis, "unobserved_budget_gases": unknown,
            "validation": {"passed": False, "reason": "export_only"},
            "extra_gas_above_inferred_budget_m3": dict(zip(C.RESOURCE_NAMES, excess.tolist()))})
    schedule = pd.DataFrame(records, columns=["datetime", "opt_generator_use_blast_furnace_gas",
        "opt_generator_use_coke_gas", "opt_generator_use_converter_gas", "opt_generator_1", "opt_generator_all"])
    return {"status": "UNVERIFIED_OPTIMIZATION", "submittable": False, "schedule": schedule,
        "summary": {"status": "UNVERIFIED_OPTIMIZATION", "validation_basis": "causal_fallback_export",
            "reason": "已生成因果兜底指令，原优化失败: " + str(failed.get("summary", {}).get("reason", failed.get("status"))),
            "original_optimization_failure": failed.get("summary", {}), "export_fallback_used": True,
            "export_warnings": list(dict.fromkeys(warnings)), "plan_validation_passed": False,
            "physical_feasibility_certified": False, "planned_steps": len(schedule),
            "resource_mode": "causal_inferred_export_budget", "dispatch_mode": "daily_fixed",
            "intraday_refresh": False, "publication_time": C.OPT_CONFIG["publication_time"],
            "daily_publications": len(plans), "daily_plans": plans, "initial_unit_basis": initial_basis,
            "time_semantics": "target_time; causal daily fallback instructions",
            "resource_publication_check": resource_publication_check(resource_bundle, issues[0], use_final=True),
            "physical_validation": resource_bundle.final_physical_validation}}


def optimize(resource_forecast: np.ndarray, prepared: PreparedData,
             origins: pd.DatetimeIndex, long_pred: np.ndarray,
             resource_bundle: ResourceBundle, quick: bool = False,
             use_final: bool = False) -> dict:
    """按最新官方口径每天固定发布，时间列与长期预测一致。"""
    if C.OPT_CONFIG.get("availability_mode", "point") != "point":
        raise ValueError("正式调度只支持已验证的 point 模式")
    if resource_bundle.blocker and resource_bundle.blocker != "PHYSICAL_MAPPING_UNCERTAIN":
        return {"status": resource_bundle.blocker, "submittable": False, "schedule": None,
                "summary": {"reason": resource_bundle.blocker}}
    resource_bundle.dispatch_selection = {"chosen": "daily_fixed", "replan_steps": C.STEPS_PER_DAY,
        "scope": "official_daily_publication_requirement_20261026", "intraday_refresh": False,
        "publication_time": C.OPT_CONFIG["publication_time"], "coverage_basis": "same_datetime_as_long_forecast"}
    return optimize_daily(prepared, origins, resource_bundle, quick=quick, use_final=use_final)
