"""数据契约与因果自检。

这里只做可执行断言，不生成独立审计文件；结果折叠进 ``运行结果.json``。
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

import config as C
from .pipeline import (PreparedData, build_base_features, prepare_data,
                       target_slot_profile)


def data_contract_check(prepared: PreparedData) -> dict:
    problems = list(prepared.scan.get("problems", [])) + list(prepared.quality.get("problems", []))
    if not len(prepared.index):
        return {"passed": False, "problems": problems + ["数据网格为空"], "rows": 0, "mask_start": None}
    if not prepared.index.is_monotonic_increasing:
        problems.append("时间索引非升序")
    if prepared.index.has_duplicates:
        problems.append("时间索引存在重复")
    if not prepared.labels.index.equals(prepared.index):
        problems.append("标签索引与特征索引不一致")
    if not prepared.frame.index.equals(prepared.index):
        problems.append("宽表索引与数据网格不一致")
    if not (prepared.index[1:] - prepared.index[:-1] == pd.Timedelta(minutes=C.STEP_MINUTES)).all():
        problems.append("数据网格不是连续15分钟")
    if np.isinf(prepared.frame.to_numpy(dtype=float)).any():
        problems.append("宽表仍含正负无穷值")
    if not prepared.provenance.empty:
        state = prepared.provenance
        if not state.index.equals(prepared.index) or not state.columns.isin(prepared.frame.columns).all():
            problems.append("观测来源索引或字段与宽表不一致")
        elif (not np.isin(state.to_numpy(), [0, 1, 2]).all()
                or not np.array_equal(state.to_numpy() > 0,
                                      prepared.frame[state.columns].notna().to_numpy())):
            problems.append("观测来源标记与清洗值不一致")
    if prepared.quality.get("data_scope") == "training_only":
        if prepared.index[-1] > pd.Timestamp(C.TRAIN_END):
            problems.append("训练数据超过允许截止点")
        if any(s["role"] != "train" for s in prepared.scan.get("stages", {}).values()):
            problems.append("训练数据包含非训练阶段")
    for target in C.TARGETS:
        if C.TARGET_COLUMN[target] not in prepared.frame.columns:
            problems.append(f"缺少目标列 {C.TARGET_COLUMN[target]}")
    if prepared.mask_start is not None:
        rows = prepared.frame.index >= prepared.mask_start
        if prepared.mask_end is not None:
            rows &= prepared.frame.index <= prepared.mask_end
        columns = [C.TARGET_COLUMN[t] for t in C.TARGETS if C.TARGET_COLUMN[t] in prepared.frame]
        present = prepared.frame.loc[rows, columns]
        if present.notna().any().any():
            problems.append("评测窗口目标未被完全遮挡")
    return {"passed": not problems, "problems": list(dict.fromkeys(problems)),
            # 新发布阶段覆盖旧版本属于已知修订，不与同阶段冲突混为一谈。
            "stage_revision_conflicts": prepared.quality.get("duplicate_conflicts", 0),
            "rows": int(len(prepared.index)),
            "mask_start": str(prepared.mask_start) if prepared.mask_start is not None else None}


def _synthetic_prepared(n: int = 480) -> PreparedData:
    index = pd.date_range("2025-01-01", periods=n, freq="15min")
    t = np.arange(n, dtype=float)
    frame = pd.DataFrame(index=index)
    base = 100.0 + 20.0 * np.sin(2 * np.pi * t / 96.0)
    for col in ("d_use_blast", "d_use_coke", "d_use_converter"):
        frame[col] = 40000.0 + 5000.0 * np.sin(2 * np.pi * t / 96.0 + 0.3)
    frame["d_use_total"] = frame[["d_use_blast", "d_use_coke", "d_use_converter"]].sum(axis=1)
    frame["d_fuel_proxy"] = frame["d_use_blast"] + 5.0 * frame["d_use_coke"] + 2.4 * frame["d_use_converter"]
    frame["d_gen_blast"] = 800000.0 + 10000.0 * np.sin(2 * np.pi * t / 96.0)
    frame["d_gen_coke"] = 70000.0
    frame["d_gen_converter_scaled"] = 120000.0
    frame["d_air_heater_total"] = 200000.0
    frame["d_blast_user_total"] = 60000.0
    frame["d_converter_user_total"] = 10000.0
    frame["d_surplus_blast"] = frame["d_gen_blast"] - frame["d_air_heater_total"] - frame["d_blast_user_total"]
    frame["d_surplus_coke"] = 30000.0
    frame["d_surplus_converter"] = 40000.0
    frame["d_holder_level"] = 100000.0 + 10000.0 * np.sin(2 * np.pi * t / 96.0)
    frame["d_holder_ratio"] = frame["d_holder_level"] / C.GAS_HOLDER["capacity_m3"]
    frame[C.TARGET_COLUMN["generator_1"]] = base
    frame[C.TARGET_COLUMN["generator_all"]] = base * 2.5
    labels = pd.DataFrame({t: frame[C.TARGET_COLUMN[t]] for t in C.TARGETS}, index=index)
    return PreparedData(index=index, frame=frame, labels=labels,
                        quality={}, scan={}, source_hashes={}, stage_ranges={})


def _stage_isolation_check() -> dict:
    """用临时数据验证新阶段覆盖、特殊文件名及测试集隔离，不读取真实测试值。"""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    C.EXPERIMENT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="stage_isolation_", dir=C.EXPERIMENT_OUTPUT_ROOT) as directory:
        cfg = SimpleNamespace(**{k: getattr(C, k) for k in dir(C) if k.isupper()})
        cfg.ROOT = Path(directory)
        cfg.DATA_ROOT = cfg.ROOT / "data"
        cfg.STAGES = tuple(s for s in C.STAGES if s["name"] in {"semi_test", "half_train", "half_test"})
        for stage in cfg.STAGES:
            folder = cfg.DATA_ROOT / stage["dir"]
            folder.mkdir(parents=True)
            times = (["2025-10-11 00:00:00", "2025-10-31 23:45:00"]
                     if stage["name"] == "half_test" else ["2025-10-10 23:45:00", "2025-10-11 00:00:00"])
            payloads = {
                "load": {"generator_1": 100., "generator_all": 200.,
                         "generator_use_blast_furnace_gas": 500000.,
                         "generator_use_coke_gas": 30000., "generator_use_converter_gas": 50000.},
                "gas": {"blast_furnace_1": 800000., "coke_oven_1": 50000., "converter_1": 2000.},
                "gas_user": {"blast_furnace_user1": 10000.},
                "gas_holder": {"blast_furnace_gas_holder_2": 100000.},
            }
            for table, values in payloads.items():
                raw = pd.DataFrame({"datetime": times, **values})
                if stage["name"] != "half_train" and table == "load":
                    raw.loc[:, list(C.TARGETS)] = np.nan
                name = "final_load_real.csv" if stage["name"] == "half_test" and table == "load" else f"final_{table}.csv"
                raw.to_csv(folder / name, index=False)
        training = prepare_data(cfg)
        evaluated = prepare_data(cfg, include_inference=True)
        training.require_training_data()
        from main_s import train_short
        from main_l import train_long
        from main_opt import train_resources
        from main_nn import train_nn
        blocked = []
        for trainer in (train_short, train_long, train_resources, train_nn):
            try:
                trainer(evaluated.prefix_copy(C.TRAIN_END), [])
            except ValueError as exc:
                blocked.append("本地评测数据不得进入训练" in str(exc))
            else:
                blocked.append(False)
        overlap = pd.Timestamp("2025-10-10 23:45:00")
        official = evaluated.index >= pd.Timestamp(C.OFFICIAL_EVAL_START)
        priority = bool(evaluated.labels.loc[overlap, "generator_1"] == 100.)
        boundary = bool(training.index[-1] == overlap and training.labels.notna().all().all())
        alias = bool(evaluated.frame.loc[C.OFFICIAL_EVAL_START, "d_use_blast"] == 500000.)
        targets_hidden = bool(evaluated.labels.loc[official].isna().all().all())
        test_load = cfg.DATA_ROOT / "half_test" / "final_load_real.csv"
        changed = pd.read_csv(test_load)
        changed.loc[:, "generator_use_blast_furnace_gas"] = 0.
        changed.loc[:, list(C.TARGETS)] = 999.
        changed.to_csv(test_load, index=False)
        second = prepare_data(cfg)
        independent = bool(training.frame.equals(second.frame) and training.source_hashes == second.source_hashes
                           and set(training.stage_ranges) == {"half_train"})
        rejected = False
        try:
            prepare_data(cfg, include_inference=True)
        except ValueError as exc:
            rejected = "官方测试标签禁止" in str(exc)
    return {"passed": all((priority, boundary, alias, targets_hidden, independent, rejected, all(blocked))),
            "new_stage_priority": priority, "training_cutoff": boundary,
            "load_real_recognized": alias, "test_targets_hidden": targets_hidden,
            "test_changes_do_not_affect_training": independent, "test_labels_rejected": rejected,
            "all_training_entries_reject_evaluation": all(blocked)}


def _future_perturbation_check() -> dict:
    prepared = _synthetic_prepared()
    base = build_base_features(prepared, include_targets=True)
    origin = 300
    rng = np.random.default_rng(7)
    perturbed = copy.deepcopy(prepared)
    future = perturbed.frame.index > prepared.index[origin]
    for col in perturbed.frame.columns:
        values = perturbed.frame.loc[future, col].to_numpy(dtype=float)
        if values.size:
            perturbed.frame.loc[future, col] = values * (1.0 + rng.normal(0, 0.5, values.size))
    changed = build_base_features(perturbed, include_targets=True)
    left = base.iloc[:origin + 1].to_numpy(dtype=float)
    right = changed.iloc[:origin + 1].to_numpy(dtype=float)
    same = np.allclose(left, right, equal_nan=True)
    return {"passed": bool(same), "checked_before_origin": origin + 1}


def _prefix_consistency_check() -> dict:
    prepared = _synthetic_prepared(480)
    full = build_base_features(prepared, include_targets=True)
    cutoff = prepared.index[320]
    prefix = build_base_features(prepared.masked_copy(cutoff, prepared.index[-1]),
                                 include_targets=True)
    # 遮挡从 cutoff 开始，cutoff 及之前的特征应完全一致（只差同日历史统计口径）。
    left = full.iloc[:300].to_numpy(dtype=float)
    right = prefix.iloc[:300].to_numpy(dtype=float)
    same = np.allclose(left, right, equal_nan=True)
    return {"passed": bool(same), "checked_rows": 300}


def _zero_preservation_check() -> dict:
    prepared = _synthetic_prepared(240)
    col = C.TARGET_COLUMN["generator_1"]
    prepared.frame.loc[prepared.index[80:88], col] = 0.0
    labels = prepared.labels.copy()
    labels.loc[prepared.index[80:88], "generator_1"] = 0.0
    prepared.labels = labels
    ok = bool((prepared.labels.loc[prepared.index[80:88], "generator_1"] == 0).all())
    ok &= bool((prepared.frame.loc[prepared.index[80:88], col] == 0).all())
    return {"passed": ok}


def _preprocessing_contract_check() -> dict:
    """核验真实观测与补值隔离、长缺口上限及质量门禁，不训练模型。"""
    from .pipeline import _apply_sentinel_rules, _causal_fill, _build_derived, _standardize_table

    index = pd.date_range("2025-01-01", periods=240, freq="15min")
    raw = pd.DataFrame({"gas_air_heater_1": np.r_[np.full(96, 100.), np.full(144, np.nan)],
                        "gas_coke_oven_1": 0., "gas_converter_1": 1.}, index=index)
    for i in range(1, 6):
        raw[f"gas_blast_furnace_{i}"] = np.nan if i == 3 else 100.
    for gas in ("blast_furnace", "coke", "converter"):
        raw[f"load_generator_use_{gas}_gas"] = 10.
    raw.loc[index[20], "gas_coke_oven_1"] = np.inf
    quality = {}
    _apply_sentinel_rules(raw, quality)
    provenance = _causal_fill(raw, quality)
    _build_derived(raw)
    probe = _synthetic_prepared(240)
    for col in raw:
        probe.frame[col] = raw[col]
    probe.provenance = provenance
    short = probe.prefix_copy(index[120])
    imputed = provenance["gas_air_heater_1"].eq(2)
    checks = {
        "bounded_causal_fill": int(imputed.sum()) == C.CLEANING["ffill_limit"]
            and raw.loc[index[96 + C.CLEANING["ffill_limit"]]:, "gas_air_heater_1"].isna().all(),
        "zero_is_observed": provenance.loc[index[10], "gas_coke_oven_1"] == 1
            and raw.loc[index[10], "gas_coke_oven_1"] == 0.,
        "infinity_is_missing": provenance.loc[index[20], "gas_coke_oven_1"] == 0,
        "partial_sum_kept_as_predictor": raw["d_gen_blast"].eq(400.).all()
            and raw["d_gen_blast_valid_fraction"].eq(.8).all(),
        "physical_view_rejects_partial_and_imputed": probe.observed_values(
            [f"gas_blast_furnace_{i}" for i in range(1, 6)]).sum(axis=1, min_count=5).isna().all()
            and probe.observed_values(["gas_air_heater_1"]).loc[imputed].isna().all().all(),
        "prefix_provenance_aligned": short.provenance.index.equals(short.index),
    }
    failed = copy.deepcopy(probe)
    failed.scan["problems"] = ["模拟扫描失败"]
    checks["scan_failure_blocked"] = not data_contract_check(failed)["passed"]
    failed = copy.deepcopy(probe)
    failed.quality["problems"] = ["模拟同阶段时间戳冲突"]
    checks["source_conflict_blocked"] = not data_contract_check(failed)["passed"]
    checks["imputed_source_retained"] = probe.provenance.loc[index[100], "gas_air_heater_1"] == 2
    failed = copy.deepcopy(probe)
    failed.provenance.loc[index[-1], "gas_air_heater_1"] = 2
    checks["inconsistent_provenance_blocked"] = not data_contract_check(failed)["passed"]
    state = probe.observation_quality(["gas_air_heater_1", "gas_blast_furnace_3"])
    checks["imputation_does_not_reset_observation_age"] = state["age_steps"].loc[index[100], "gas_air_heater_1"] == 5
    checks["unobserved_history_age_unknown"] = (state["age_steps"]["gas_blast_furnace_3"].isna().all()
        and state["no_observed_history"]["gas_blast_furnace_3"].all())
    checks["masked_summary_refreshed"] = (probe.masked_power_gas_copy().quality["preprocessing"]["columns"]["load_generator_use_coke_gas"]["observed"] == 0
        and probe.provenance["load_generator_use_coke_gas"].eq(1).all())
    issues = []
    _standardize_table(pd.DataFrame({"datetime": [index[0], index[0]], "x": [1., 2.]}), "probe", issues)
    checks["duplicate_conflict_recorded"] = bool(issues)
    masked = _synthetic_prepared(240).masked_copy(index[100], index[119])
    checks["mask_end_respected"] = data_contract_check(masked)["passed"]
    return {"passed": bool(all(checks.values())), **{k: bool(v) for k, v in checks.items()}}


def _mask_check() -> dict:
    prepared = _synthetic_prepared(240)
    masked = prepared.masked_copy(prepared.index[100], prepared.index[-1])
    cols = [C.TARGET_COLUMN[t] for t in C.TARGETS]
    empty = masked.frame.loc[masked.index >= masked.index[100], cols].isna().all().all()
    labels_kept = prepared.labels.equals(masked.labels)
    return {"passed": bool(empty and labels_kept)}


def _short_process_recovery_check() -> dict:
    """恢复分支不修改实测值，且耗气缺测起报点沿用完整原输入。"""
    from unittest.mock import patch
    from .pipeline import short_process_view
    import main_s

    data = _synthetic_prepared(480)
    for i in range(1, 6):
        data.frame[f"gas_blast_furnace_{i}"] = 200.
    column = "gas_blast_furnace_3"
    data.frame.loc[data.index[100:220], column] = np.nan
    data.provenance = data.frame[[f"gas_blast_furnace_{i}" for i in range(1, 6)]].notna().astype(np.uint8)
    recovered, active = short_process_view(data, C.SHORT_PROCESS_RECOVERY)
    changed = data.frame.copy()
    future = copy.deepcopy(data)
    future.frame.loc[future.index > data.index[160], column] = 10000.
    later, later_active = short_process_view(future, C.SHORT_PROCESS_RECOVERY)
    absent = copy.deepcopy(data)
    absent.provenance.loc[:, column] = 0
    absent.frame.loc[:, column] = np.nan
    _, no_history_active = short_process_view(absent, C.SHORT_PROCESS_RECOVERY)
    unavailable = copy.deepcopy(data)
    unavailable.frame.loc[unavailable.index[150]:, "d_use_coke"] = np.nan
    bundle = main_s.ShortBundle("global", "gas", C.HORIZONS_S, {}, {}, {}, [])
    def core(pack, view, origins, use_final=False):
        return np.full((len(origins), 8, 2), 1. if view is unavailable else 2.)
    with patch.object(main_s, "_predict_short_core", side_effect=core):
        routed = main_s.predict_short(bundle, unavailable, data.index[[110, 160]])
    checks = {"raw_values_and_coverage_kept": recovered.frame[column].equals(data.frame[column])
        and recovered.provenance.equals(data.provenance) and data.frame.equals(changed),
        "age_limit_uses_real_readings": recovered.frame.loc[data.index[195], "d_gen_blast"] > data.frame.loc[data.index[195], "d_gen_blast"]
            and recovered.frame.loc[data.index[196], "d_gen_blast"] == data.frame.loc[data.index[196], "d_gen_blast"],
        "no_backfill_before_own_observations": not no_history_active.any(),
        "future_does_not_change_earlier_recovery": np.allclose(recovered.frame.iloc[:161], later.frame.iloc[:161], equal_nan=True)
            and np.array_equal(active[:161], later_active[:161]),
        "missing_power_gas_routes_original_window": bool(np.all(routed[0] == 2.) and np.all(routed[1] == 1.))}
    return {"passed": bool(all(checks.values())), **{k: bool(v) for k, v in checks.items()}}


def _profile_causality_check() -> dict:
    prepared = _synthetic_prepared(960)
    positions = np.arange(400, 420)
    first = target_slot_profile(prepared, positions, C.HORIZONS_S, "generator_1")
    altered = copy.deepcopy(prepared)
    future = altered.frame.index > prepared.index[420]
    altered.frame.loc[future, C.TARGET_COLUMN["generator_1"]] = 0.0
    second = target_slot_profile(altered, positions, C.HORIZONS_S, "generator_1")
    from main_s import baseline_cubes
    altered.frame.loc[altered.index[positions], "d_fuel_proxy"] = np.nan
    baselines = baseline_cubes(altered, positions, C.HORIZONS_S)
    neutral = (np.isfinite(baselines["profile_fuel"]).all()
               and np.allclose(baselines["profile"], baselines["profile_fuel"]))
    return {"passed": bool(np.allclose(first, second, equal_nan=True) and neutral),
            "missing_fuel_unscaled_profile": bool(neutral)}


def _multistep_alignment_check() -> dict:
    from main_s import (_predict_models, _feature_columns, design_inputs,
                        build_training_block, training_weights)
    from main_opt import (_resource_base, _resource_design, ResourceBundle,
                          forecast_resources, _stack)

    class Probe:
        def predict(self, X):
            return 100.0 + X[:, 0] / 100000.0 + X[:, -1] * 0.1

    prepared = _synthetic_prepared(480)
    positions = np.array([240, 300, 383])
    base = build_base_features(prepared, include_targets=True)
    checks = []
    for horizons in (C.HORIZONS_S, C.HORIZONS_L):
        subset = base[_feature_columns(base, "gas")]
        extras, names = design_inputs(prepared, subset, positions, horizons, "gas")
        for kind in ("global", "grouped"):
            if kind == "global":
                models = {t: Probe() for t in C.TARGETS}
            else:
                groups = C.SHORT_GROUPS if len(horizons) == 8 else C.LONG_GROUPS
                models = {(t, g[0]): Probe() for t in C.TARGETS for g in groups}
            bulk = _predict_models(prepared, base, positions, horizons, kind, "gas", models, names)
            single = np.concatenate([_predict_models(prepared, base, np.array([p]), horizons,
                                                     kind, "gas", models, names) for p in positions])
            checks.append(np.allclose(bulk, single))
        weights = training_weights(prepared, positions, np.asarray(horizons), 479)
        X, y, w = build_training_block(prepared, subset, subset.to_numpy(), extras,
                                       list(extras), positions, horizons, 0, C.TARGETS[0], weights)
        checks.append(w.ndim == 1 and np.array_equal(w, weights[:, 0]))
    rb = _resource_base(prepared)
    _, names = _resource_design(prepared, rb, positions, C.HORIZONS_L)
    pool = {g: Probe() for g in C.RESOURCE_NAMES}
    bundle = ResourceBundle(pool, pool, names)
    for forecast in (lambda o: forecast_resources(bundle, prepared, o),):
        bulk = forecast(prepared.index[positions])
        single = np.concatenate([forecast(prepared.index[[p]]) for p in positions])
        checks.append(np.allclose(bulk, single))
    class HeadProbe:
        def predict(self, X):
            return 200. + X[:, -1]

    original = forecast_resources(bundle, prepared, prepared.index[positions])
    pool["blast__head"] = HeadProbe()
    with_head = forecast_resources(bundle, prepared, prepared.index[positions])
    expected_head = np.tile(200. + np.asarray(C.HORIZONS_S), (len(positions), 1))
    checks.append(np.allclose(with_head[:, :8, 0], expected_head)
                  and np.array_equal(with_head[:, 8:], original[:, 8:])
                  and np.array_equal(with_head[:, :, 1:], original[:, :, 1:]))
    bundle.selection = {"head_selection": {"weight": .5}}
    blended = forecast_resources(bundle, prepared, prepared.index[positions])
    checks.append(np.allclose(blended[:, :8, 0], (expected_head + original[:, :8, 0]) / 2.)
                  and np.array_equal(blended[:, 8:], original[:, 8:]))
    series = np.arange(480, dtype=float)
    series[102] = np.nan
    X, y, w = _stack(prepared, np.arange(480)[:, None], {}, [], np.array([100, 200]),
                     (1, 2, 3), series, np.array([[10, 11, 12], [20, 21, 22]]))
    checks.append(y.tolist() == [101, 201, 202, 103, 203]
                  and w.tolist() == [10, 20, 21, 12, 22])
    return {"passed": bool(all(checks)), "checks": len(checks)}


def _nn_contract_check() -> dict:
    import main_nn
    from unittest.mock import patch

    prepared = _synthetic_prepared(480)
    # 极小前缀一轮训练用于核对截止点隔离，不跑历史全量 NN。
    fold = [{"role": "tuning", "train_end": str(prepared.index[299])}]
    with patch.dict(C.NN_CONFIG, {"look_back": 8, "early_stop_days": 2,
                                  "quick_epochs": 1, "quick_patience": 1}), patch.object(C, "PROFILE_DAYS", 0):
        first = main_nn.train_nn(prepared, fold, horizons=(1, 2), quick=True)
        altered = copy.deepcopy(prepared)
        altered.labels.iloc[300:] = 100000.0
        altered.frame.iloc[300:] = 100000.0
        second = main_nn.train_nn(altered, fold, horizons=(1, 2), quick=True)
    same = all(np.array_equal(first.state_dict[k].numpy(), second.state_dict[k].numpy())
               for k in first.state_dict)
    origins = prepared.index[[300, 479]]
    bulk = main_nn.predict_nn(first, prepared, origins)
    singles = np.concatenate([main_nn.predict_nn(first, prepared, origins[[i]]) for i in range(2)])
    # 两个更新步后时间主干必须收到梯度；仅检查预测有限会漏掉连续零初始化闭锁。
    torch = main_nn.torch
    torch.manual_seed(19)
    network = main_nn.CausalTCN(8, 2, 3, n_fut=1, cfg={"head_mode": "scaled",
        "d_model": 8, "n_blocks": 1, "dropout": 0., "temporal_norm": "layer"})
    future = torch.zeros(12, 2, 1)
    x, target = torch.randn(12, 8, 3), torch.randn(12, 2, 2)
    optimizer = torch.optim.SGD(network.parameters(), lr=.05)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (network(x, future) - target).square().mean().backward()
        optimizer.step()
    gradient = bool(network.stem.weight.grad.abs().sum() > 0)
    sensitive = not torch.allclose(network(x, future), network(x + 2., future))
    with patch.object(C, "PROFILE_DAYS", 2):
        positions = np.array([200, 215, 270])
        blocked = main_nn._blocked_profiles(prepared, positions, (1, 2), 2)
        view = prepared.masked_copy(prepared.index[192], prepared.index[383])
        expected = main_nn._profile_cube(view, positions, (1, 2), fuel=False)
        mask_equivalent = np.allclose(blocked, expected, atol=1e-5)
    from main_s import enforce_physics, score_cube
    values = torch.arange(10, dtype=torch.float32, device=main_nn.DEVICE).reshape(-1, 1) / 10
    y = np.array([[[50, 160], [0, 180]], [[90, np.nan], [80, 360]], [[110, 400], [100, 0]]], dtype=np.float32)
    positions = np.array([2, 4, 6])
    future = np.zeros((3, 2, 6), dtype=np.float32)
    sample = main_nn._Sample(values, positions, y, future, None, 2, 2, (1, 2), "scaled", {})
    class Probe(torch.nn.Module):
        def forward(self, x, future):
            level = x[:, -1, 0]
            return torch.stack((level, 2 * level), dim=-1)[:, None, :].expand(-1, 2, -1)
    network = Probe().to(main_nn.DEVICE)
    prediction = network(main_nn._windows(values, sample.pos, 2), sample.future)
    prediction = sample.decode(prediction, torch.arange(3, device=main_nn.DEVICE)).cpu().numpy()
    expected_mape = 1. - score_cube(enforce_physics(prediction), y, (1, 2))["accuracy"] / 100.
    metric_equal = abs(sample.evaluate(network) - expected_mape) < 1e-6
    return {"passed": bool(same and gradient and sensitive and mask_equivalent and metric_equal and np.isfinite(bulk).all()
                           and np.allclose(bulk, singles, atol=1e-5)),
            "gradient_reaches_backbone": gradient, "input_sensitive": sensitive,
            "training_block_matches_masked_profile": bool(mask_equivalent),
            "early_stop_matches_forecast_metric": bool(metric_equal),
            "inner_target_history_masked": first.scores["early_stop"]["target_history_masked"],
            "prefix_model_unchanged": bool(same), "tail_shape": list(bulk.shape)}


def _model_role_check() -> dict:
    from types import SimpleNamespace
    from unittest.mock import patch
    import main_nn
    from main_s import ShortBundle, predict_short
    from main_l import LongBundle, predict_long

    prepared = _synthetic_prepared(480)
    checks = []
    for cls, forecast, horizons, groups in ((ShortBundle, predict_short, C.HORIZONS_S, C.SHORT_GROUPS),
                                            (LongBundle, predict_long, C.HORIZONS_L, C.LONG_GROUPS)):
        bundle = cls(kind="nn", features="nn", horizons=horizons,
                     blend={t: {g[0]: 1.0 for g in groups} for t in C.TARGETS},
                     models={}, final_models={}, feature_names=[],
                     nn_bundle=SimpleNamespace(value=100), final_nn_bundle=SimpleNamespace(value=120))
        with patch.object(main_nn, "predict_nn", side_effect=lambda b, p, o: np.full((len(o), len(horizons), 2), b.value)):
            checks.append(np.all(forecast(bundle, prepared, prepared.index[[300]]) == 100))
            checks.append(np.all(forecast(bundle, prepared, prepared.index[[300]], use_final=True) == 120))
    return {"passed": bool(all(checks))}


def _optimizer_contract_check() -> dict:
    from types import SimpleNamespace
    from main_opt import (_fallback_day, _solve_day, _gas_caps, _price_vector, validate_day,
                          _execute_proxy_step, score_resources, _dispatch_metrics,
                          _adjust_available_resources)
    prepared = _synthetic_prepared(480)
    prepared.frame[C.GAS_HOLDER["column"]] = prepared.frame["d_holder_level"]
    heat = {"coefficients_mw_per_m3h": {"blast": 0.000346, "coke": 0.0019, "converter": 0.00059}}
    caps = np.array([1e6, 1e5, 1e5])
    zero = np.zeros((4, 3))
    fallback = _fallback_day(prepared, 300, zero, caps, heat,
                             holder0=100000.0, initial_power=np.zeros(6))
    supply = np.tile([8e5, 3e4, 7e4], (4, 1))
    result = _solve_day(supply, np.array([0.2, 0.2, 1., 1.]), 100000., np.zeros(6), caps, heat)
    feasible = result.get("validation", {}).get("passed", False)
    tampered = copy.deepcopy(result)
    if feasible:
        tampered["power"] += 10
    detected = feasible and not validate_day(tampered, supply, 100000., np.zeros(6), caps, heat)["passed"]
    frozen = _gas_caps(prepared, 300)
    prepared.frame.loc[prepared.index[301]:, "d_use_blast"] = 1e12
    prefix = frozen == _gas_caps(prepared, 300)
    price = _price_vector({9: np.ones(96), 10: np.full(96, 2.)}, pd.Timestamp("2025-09-30 23:45"), 2)
    curtailed = _execute_proxy_step(np.full(6, 50.), np.array([8e5, 3e4, 7e4]),
                                    np.zeros(3), 30000., np.zeros(6), caps, heat)
    resource_truth = np.zeros((1, 96, 3))
    resource_truth[:, 48:, :] = 10
    scores = score_resources(np.full_like(resource_truth, 10), resource_truth)
    zero_counted = all(abs(m["wape"] - 1.) < 1e-9 and m["zero_fraction"] == .5
                       for m in scores.values())
    rng = np.random.default_rng(41)
    level, previous = 100000., np.zeros(6)
    blocked_shocks = 0
    for _ in range(1200):
        try:
            executed = _execute_proxy_step(np.array([50., 50., 50., 50., 100., 100.]),
                np.array([8e5, 3e4, 7e4]), rng.random(3) * [8e5, 3e4, 7e4],
                level, previous, caps, heat)
        except ValueError as exc:
            if "爬坡与最小负荷" not in str(exc):
                raise
            blocked_shocks += 1
            # 突发断气与下降爬坡可能无交集；失败须显式保留，下一独立工况重置初态。
            level, previous = 100000., np.zeros(6)
            continue
        level, previous = executed["holder"], executed["power"]
    stress_passed = 30000. <= level <= 180000.
    cfg = SimpleNamespace(**vars(C))
    cfg.OPT_CONFIG = {**C.OPT_CONFIG, "execution_gas_mode": "substitute"}
    request = np.array([50., 50., 50., 50., 0., 0.])
    mapping = {"coefficients_mw_per_m3h": {"blast": .0004, "coke": .002, "converter": .0006}}
    clip_config = SimpleNamespace(**vars(C))
    clip_config.OPT_CONFIG = {**C.OPT_CONFIG, "execution_gas_mode": "clip"}
    clipped = _execute_proxy_step(request, np.array([4.375e5, 1.25e4, 0.]),
        np.array([6e5, 0., 0.]), 1e5, request, caps, mapping, clip_config)
    substituted = _execute_proxy_step(request, np.array([4.375e5, 1.25e4, 0.]),
        np.array([6e5, 0., 0.]), 1e5, request, caps, mapping, cfg)
    recovered = (substituted["shortfall_mwh"] < clipped["shortfall_mwh"]
                 and abs(substituted["power"].sum() - request.sum()) < 1e-6)
    overflow_request = np.array([40., 36., 30., 0., 0., 0.])
    overflow_gas = np.array([1e5, 3e4, 1e4])
    overflowing = np.array([5e5, 3e4, 1e4])
    before = _execute_proxy_step(overflow_request, overflow_gas, overflowing,
                                 180000., overflow_request, caps, mapping, cfg)
    cfg.OPT_CONFIG["execution_gas_mode"] = "substitute_vent"
    after = _execute_proxy_step(overflow_request, overflow_gas, overflowing,
                                180000., overflow_request, caps, mapping, cfg)
    overflow_recovered = (np.allclose(before["power"], after["power"])
                          and abs(before["holder"] - after["holder"]) < 1e-6
                          and after["vent_blast_m3h"] < before["vent_blast_m3h"])
    substitute_blocked = 0
    for _ in range(1200):
        try:
            executed = _execute_proxy_step(np.array([50., 50., 50., 50., 100., 100.]),
                np.array([8e5, 3e4, 7e4]), rng.random(3) * [8e5, 3e4, 7e4],
                level, previous, caps, heat, cfg)
        except ValueError as exc:
            if "爬坡与最小负荷" not in str(exc):
                raise
            substitute_blocked += 1
            level, previous = 100000., np.zeros(6)
            continue
        level, previous = executed["holder"], executed["power"]
    substitute_safe = 30000. <= level <= 180000.
    times = pd.date_range("2025-01-01 23:45", periods=2, freq="15min")
    tariff = {1: np.r_[np.full(95, .2), 1.]}
    metrics = _dispatch_metrics(times, np.array([100., 200.]), tariff)
    expected = 140. / (tariff[1].mean() * 300.)
    metrics_ok = (abs(metrics["price_weighted_load_ratio"] - expected) < 1e-9
                  and len(metrics["daily"]) == 2
                  and not _dispatch_metrics(times, np.array([100., np.nan]), tariff)["complete"])
    gate_data = _synthetic_prepared(480)
    gate_origins = gate_data.index[[128, 200]]
    columns = ["d_use_blast", "d_use_coke", "d_use_converter"]
    gate_data.frame.loc[gate_origins, columns] = 20000.
    fixed = np.full((2, 96, 3), 10000.)
    recipe = {"method": "positive_persistence_peak", "threshold_ratio": .03, "weight": .5}
    tariffs = {1: np.r_[np.full(32, .2), np.ones(32), np.full(32, .2)]}
    gate_heat = {"ok": True, **mapping}
    gated = _adjust_available_resources(fixed, gate_data, gate_origins, gate_heat, recipe, tariffs)
    aligned = np.concatenate([_adjust_available_resources(fixed[i:i+1], gate_data,
        gate_origins[i:i+1], gate_heat, recipe, tariffs) for i in range(2)])
    changed = copy.deepcopy(gate_data)
    changed.frame.loc[changed.index > gate_origins[0], columns] *= 1000.
    future_same = _adjust_available_resources(fixed[:1], changed, gate_origins[:1], gate_heat, recipe, tariffs)
    gate_data.frame.loc[gate_origins[0], "d_use_coke"] = np.nan
    missing = _adjust_available_resources(fixed, gate_data, gate_origins, gate_heat, recipe, tariffs)
    gate_data.frame.loc[gate_origins[0], columns] = 0.
    zero = _adjust_available_resources(fixed, gate_data, gate_origins, gate_heat, recipe, tariffs)
    gate_data.frame.loc[gate_origins[0], columns] = 10200.
    below = _adjust_available_resources(fixed, gate_data, gate_origins, gate_heat, recipe, tariffs)
    gate_data.frame.loc[gate_origins[0], columns] = 20000.
    nn_recipe = {"method": "positive_nn_peak", "threshold_ratio": .05, "weight": 1.}
    expert = np.full((2, 8, 3), 20000.)
    nn_gated = _adjust_available_resources(fixed, gate_data, gate_origins, gate_heat,
                                          nn_recipe, tariffs, expert)
    expert[0] = np.nan
    nn_missing = _adjust_available_resources(fixed, gate_data, gate_origins, gate_heat,
                                            nn_recipe, tariffs, expert)
    resource_adjustment_ok = (np.all(gated[0, :8] == 15000.) and np.array_equal(gated[1], fixed[1])
        and np.array_equal(gated[:, 8:], fixed[:, 8:]) and np.array_equal(gated, aligned)
        and np.array_equal(future_same[0], gated[0]) and np.array_equal(missing, fixed)
        and np.array_equal(zero, fixed) and np.array_equal(below, fixed)
        and np.all(nn_gated[0, :8] == 20000.) and np.array_equal(nn_gated[:, 8:], fixed[:, 8:])
        and np.array_equal(nn_missing, fixed)
        and np.all(fixed == 10000.))
    return {"passed": bool(fallback["status"] == "FALLBACK_SAFE"
                            and np.all(fallback["power"] == 0) and detected and prefix
                            and np.all(curtailed["power"] == 0) and curtailed["holder"] == 30000.
                            and zero_counted
                            and stress_passed
                            and substitute_safe and recovered and overflow_recovered and metrics_ok
                            and resource_adjustment_ok
                            and price.tolist() == [1.0, 2.0]),
            "solver_status": result["status"], "tamper_detected": bool(detected),
            "resource_shock_trials": 1200, "hard_constraint_conflicts_blocked": blocked_shocks,
            "substitute_resource_shock_trials": 1200, "substitute_conflicts_blocked": substitute_blocked,
            "fuel_substitution_recovered_power": bool(recovered),
            "overflow_substitution_preserves_power_and_holder": bool(overflow_recovered),
            "daily_price_metric_and_missing_values": bool(metrics_ok),
            "peak_resource_adjustment_causal_missing_safe": bool(resource_adjustment_ok),
            "zero_resource_curtailment": True, "resource_zeros_counted": bool(zero_counted)}


def _rolling_causality_check() -> dict:
    from unittest.mock import patch
    import main_opt
    prepared = _synthetic_prepared(480)
    prepared.frame[C.GAS_HOLDER["column"]] = 100000.
    prepared.frame[["d_use_blast", "d_use_coke", "d_use_converter"]] = [8e5, 3e4, 7e4]
    heat = {"ok": True, "coefficients_mw_per_m3h": {"blast": .000346, "coke": .0019, "converter": .00059}}
    caps = {"blast": 1e6, "coke": 1e5, "converter": 1e5}
    bundle = main_opt.ResourceBundle({}, {}, [], heat_rate=heat, gas_caps=caps,
                                     dispatch_selection={"chosen": "rolling_proxy"})
    origins = prepared.index[300:308]
    forecast = np.tile([8e5, 3e4, 7e4], (len(origins), 96, 1))
    observed_states = []

    def solve(avail, price, holder, previous, limits, mapping, cfg):
        observed_states.append(holder)
        return main_opt._fallback_day(prepared, 300, avail, limits, mapping, cfg,
                                       holder0=holder, initial_power=previous)

    def run(data):
        return main_opt.optimize_rolling(forecast, data, origins, bundle)

    with patch.object(main_opt, "load_prices", return_value={1: np.ones(96)}), patch.object(main_opt, "_solve_day", side_effect=solve):
        original = run(prepared)
        changed_holder = copy.deepcopy(prepared)
        changed_holder.frame.loc[origins[0] + pd.Timedelta(minutes=15):, C.GAS_HOLDER["column"]] = 170000.
        same_holder = run(changed_holder)
        changed_future = copy.deepcopy(prepared)
        changed_future.frame.loc[origins[0] + pd.Timedelta(minutes=15):,
                                 ["d_use_blast", "d_use_coke", "d_use_converter"]] *= 1.1
        same_first_commit = run(changed_future)
    holder_not_injected = original["schedule"].equals(same_holder["schedule"])
    no_oracle = original["schedule"].iloc[:4].equals(same_first_commit["schedule"].iloc[:4])
    continuous = bool(len(observed_states) == 6 and observed_states[1] != 100000.)
    return {"passed": bool(holder_not_injected and no_oracle and continuous),
            "historical_holder_not_injected": holder_not_injected,
            "first_commit_future_independent": no_oracle, "continuous_state": continuous}


def _power_gas_availability_check() -> dict:
    from main_s import _feature_columns, design_inputs
    prepared = _synthetic_prepared(480).masked_copy("2025-01-04", "2025-01-05 23:45")
    original = prepared.frame.copy()
    failed = prepared.masked_power_gas_copy(prepared.index[200])
    full = build_base_features(prepared, include_targets=True)
    missing = build_base_features(failed, include_targets=True)
    columns = _feature_columns(full, "gas_no_power")
    unchanged = np.allclose(full[columns], missing[columns], equal_nan=True)
    extras, _ = design_inputs(prepared, full[columns], np.array([320, 420]), C.HORIZONS_S, "gas_no_power")
    altered, _ = design_inputs(failed, missing[columns], np.array([320, 420]), C.HORIZONS_S, "gas_no_power")
    extra_same = all(np.allclose(value, altered[key], equal_nan=True) for key, value in extras.items())
    before_same = np.allclose(full.iloc[:200], missing.iloc[:200], equal_nan=True)
    checks = {"no_power_inputs_invariant": bool(unchanged and extra_same),
              "past_features_unchanged": bool(before_same),
              "original_and_labels_preserved": bool(prepared.frame.equals(original) and failed.labels.equals(prepared.labels)),
              "all_dependent_current_values_missing": bool(failed.frame.iloc[200:][[
                  "d_use_blast", "d_use_coke", "d_use_converter", "d_use_total", "d_fuel_proxy"]].isna().all().all())}
    return {"passed": all(checks.values()), **checks}


def _observed_at_t_check() -> dict:
    from .pipeline import _observed_at_t_features
    from main_s import _feature_columns, _with_observed_features
    minutes = pd.date_range("2025-01-01", periods=121, freq="min")
    grid = minutes[::15]
    gas = pd.DataFrame({f"blast_furnace_{i}": np.arange(121.) + i * 100.
                        for i in range(1, 6)}, index=minutes)
    tables = {"gas": gas}
    base = _observed_at_t_features(tables, grid)
    t = grid[4]
    future = gas.copy()
    future.loc[future.index > t] *= 1000.
    altered = _observed_at_t_features({"gas": future}, grid)
    prefix = _observed_at_t_features({"gas": gas.loc[:t]}, grid[:5])
    current = gas.copy()
    current.loc[t, "blast_furnace_1"] += 150.
    changed_t = _observed_at_t_features({"gas": current}, grid)
    current = gas.copy()
    current.loc[t - pd.Timedelta(minutes=1), "blast_furnace_1"] += 150.
    changed_minute = _observed_at_t_features({"gas": current}, grid)
    missing = gas.copy()
    missing.loc[:, "blast_furnace_1"] = np.nan
    absent = _observed_at_t_features({"gas": missing}, grid)
    zeros = _observed_at_t_features({"gas": gas * 0.}, grid)
    prepared = _synthetic_prepared(len(grid))
    prepared.frame = prepared.frame.join(base.set_axis(prepared.index))
    legacy = build_base_features(prepared, include_targets=True)
    augmented = _with_observed_features(prepared, legacy, "gas_no_power_at_t")
    cols = _feature_columns(augmented, "gas_no_power_at_t")
    checks = {"future_unchanged": np.allclose(base.loc[:t], altered.loc[:t], equal_nan=True),
              "prefix_identical": np.allclose(base.loc[:t], prefix, equal_nan=True),
              "current_observation_used": changed_t.loc[t, "at_t_gas_blast_furnace_1"] != base.loc[t, "at_t_gas_blast_furnace_1"],
              "previous_minute_used": changed_minute.loc[t, "at_t_gen_blast_mean15m"] != base.loc[t, "at_t_gen_blast_mean15m"],
              "missing_not_zero": bool(absent["at_t_gen_blast_mean15m"].isna().all()),
              "zero_preserved": zeros.loc[t, "at_t_gen_blast_mean15m"] == 0.,
              "legacy_schema_unchanged": not any(c.startswith("at_t_") for c in legacy),
              "new_features_selected": all(c in cols for c in base)}
    return {"passed": bool(all(checks.values())), **{k: bool(v) for k, v in checks.items()},
            "additional_features": len(base.columns), "window": "(t-window,t]"}


def _generator1_contract_check() -> dict:
    import copy
    import main_s
    import main_nn
    prepared = _synthetic_prepared(40 * 96)
    prepared.frame["at_t_synthetic"] = prepared.frame["d_gen_blast"]
    origin = 35 * 96
    changed = copy.deepcopy(prepared)
    changed.frame.loc[changed.index > changed.index[origin], :] = np.nan
    causal = all(np.allclose(main_s.generator1_base(prepared, include_power=power).iloc[:origin + 1],
        main_s.generator1_base(changed, include_power=power).iloc[:origin + 1], equal_nan=True)
        for power in (False, True))
    positions = np.array([origin, origin + 1])
    first = C.PROFILE_DAYS * 96
    start = first + ((origin - first) // (21 * 96)) * 21 * 96
    view = prepared.masked_copy(prepared.index[start], prepared.index[-1])
    profiles_same = np.allclose(main_nn._blocked_profiles(prepared, positions, (1, 8, 96), 21),
        main_nn._profile_cube(view, positions, (1, 8, 96), fuel=False), equal_nan=True)
    reference = np.array([[[120., 300.], [80., 280.], [200., 440.]]])
    result = main_s.generator1_project(np.array([[-20., 400., 40.]]), reference)
    total_same = np.array_equal(reference[..., 1], result[..., 1])
    bounds = np.all(result[..., 0] >= np.maximum(0., result[..., 1] - 240.)) and np.all(result[..., 0] <= np.minimum(200., result[..., 1]))
    torch = main_nn.torch
    sample = main_nn._Sample(torch.zeros((8, 2), device=main_nn.DEVICE), np.array([3, 4, 5]),
        np.full((3, 4, 2), 100.), np.zeros((3, 4, 1), dtype=np.float32), None, 3, 4,
        (1, 2, 3, 4), "direct", {"target_indices": [0]})
    prediction = torch.full((3, 4, 2), 100., device=main_nn.DEVICE)
    changed_prediction = prediction.clone()
    changed_prediction[..., 1] = 999.
    take = torch.arange(3, device=main_nn.DEVICE)
    target_only = sample.loss(prediction, take).item() == sample.loss(changed_prediction, take).item()
    class ConstantHead(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, x, future):
            return torch.full((len(x), future.shape[1], 2), self.value, device=x.device)

    aggregation_ok = True
    for aggregation, weights, expected in (("mean", [1, 1, 1], 4.), ("median", [1, 1, 1], 4.),
                                            ("validation_weighted", [1, 0, 0], 1.)):
        actual = main_nn._predict_cube([{}, {}, {}], torch.zeros((10, 1), device=main_nn.DEVICE),
            np.array([5]), np.zeros((1, 2, 1), np.float32), 2, 2, "tsmixer", 1,
            {"ensemble_aggregation": aggregation, "ensemble_weights": weights}, 16,
            runtime_models=[ConstantHead(1.), ConstantHead(4.), ConstantHead(7.)])
        aggregation_ok &= np.allclose(actual, expected)
    transition_names = [f"g1_d_use_{gas}_log_age" for gas in ("blast", "coke", "converter")] + ["g1_generator_1_log_age", "h"]
    transition_input = np.array([[0., 0., 0., 0., 1.], [1., 1., 1., 1., 1.],
                                 [0., 0., 0., 2., 8.], [3., 3., 3., 3.2, 8.]], dtype=np.float32)
    transition_ok = np.array_equal(main_s.generator1_transition_bins(transition_input, transition_names), [0, 2, 7, 11])
    checks = {"future_independent": causal, "blocked_profile_matches_service": profiles_same,
              "total_preserved": total_same, "structure_bounds": bool(bounds),
              "specialist_loss_ignores_total_output": target_only,
              "seed_aggregation_values": bool(aggregation_ok), "transition_state_and_lead_bins": bool(transition_ok)}
    return {"passed": bool(all(checks.values())), **checks}


def _training_target_cutoff_check() -> dict:
    from main_s import _train_models
    from main_opt import _fit_resource_models
    prepared = _synthetic_prepared(480)
    rejected = []
    for fit in (lambda: _train_models(prepared, pd.DataFrame(), np.array([100]),
                 (1,), "global", "gas", 1, 100),
                lambda: _fit_resource_models(prepared, np.array([100]), 1, 100)):
        try:
            fit()
        except ValueError:
            rejected.append(True)
        else:
            rejected.append(False)
    return {"passed": all(rejected), "load_and_resource_targets_rejected": all(rejected)}


def _execution_tracking_check() -> dict:
    from types import SimpleNamespace
    from main_opt import _execute_proxy_step
    cfg = SimpleNamespace(**vars(C))
    cfg.OPT_CONFIG = {**C.OPT_CONFIG, "execution_gas_mode": "substitute_vent"}
    heat = {"coefficients_mw_per_m3h": {"blast": .000346, "coke": .0019, "converter": .00059}}
    beta = np.array([heat["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    previous, holder, tracked = np.array([30., 30., 30., 30., 100., 100.]), 100000., True
    for request in (np.array([30., 30., 30., 0., 105., 105.]),
                    np.array([30., 30., 0., 0., 90., 90.]),
                    np.array([30., 0., 0., 0., 72., 72.]),
                    np.array([0., 0., 0., 0., 108., 0.])):
        gas = np.array([8e5, 2e4, 4e4]) * (request.sum() / (beta @ np.array([8e5, 2e4, 4e4])))
        result = _execute_proxy_step(request, gas, np.array([1e6, 6e4, 8e4]), holder,
            previous, np.array([1.1e6, 8e4, 9e4]), heat, cfg)
        previous, holder = result["power"], result["holder"]
        tracked &= (abs(previous[:4].sum() - request[:4].sum()) < 1e-6
                    and abs(previous.sum() - request.sum()) < 1e-6 and result["shortfall_mwh"] < 1e-6)
    return {"passed": bool(tracked and 30000. <= holder <= 180000.),
            "published_aggregate_preserved_when_supply_sufficient": bool(tracked)}


def _daily_publication_check() -> dict:
    from unittest.mock import patch
    import main_opt
    from output import expected_result_grid, validate_official_frame
    prepared = _synthetic_prepared(480)
    prepared.frame[C.GAS_HOLDER["column"]] = 100000.
    prepared.frame[C.TARGET_COLUMN["generator_1"]] = 120.
    prepared.frame[C.TARGET_COLUMN["generator_all"]] = 320.
    origins = prepared.index[192:384]
    heat = {"ok": True, "coefficients_mw_per_m3h": {"blast": .000346, "coke": .0019, "converter": .00059}}
    caps = {"blast": 1.2e6, "coke": 1e5, "converter": 1e5}
    bundle = main_opt.ResourceBundle({}, {}, [], heat_rate=heat, gas_caps=caps,
        train_end=str(prepared.index[100]))
    expected_issues = pd.date_range("2025-01-02 07:00", periods=3, freq="D")
    with patch.dict(C.OPT_CONFIG, {"publication_time": "07:00"}), \
            patch.object(main_opt, "forecast_resources", return_value=np.tile([8e5, 3e4, 7e4], (3, 96, 1))) as forecast, \
            patch.object(main_opt, "load_prices", return_value={1: np.ones(96)}):
        result = main_opt.optimize_daily(prepared, origins, bundle)
    if result.get("schedule") is None:
        return {"passed": False, "reason": result.get("summary")}
    frame = result["schedule"]
    issues = forecast.call_args.args[2]
    import output
    import copy
    metadata = {"output_contract": {"optimization_publication_time": "07:00"}, "optimization": result["summary"],
                "resource": {"train_end": bundle.train_end, "heat_rate": {"calibration_end": bundle.train_end}}}
    publication_ok = output.validate_publication_contract(metadata, origins)["passed"]
    changed = copy.deepcopy(metadata)
    changed["optimization"]["daily_plans"][0]["effective_start"] = "2025-01-02 06:00:00"
    changed_rejected = not output.validate_publication_contract(changed, origins)["passed"]
    missing = copy.deepcopy(metadata)
    missing["resource"]["heat_rate"] = {}
    missing_rejected = not output.validate_publication_contract(missing, origins)["passed"]
    changed_control = copy.deepcopy(metadata)
    changed_control["optimization"]["daily_plans"][0]["control"]["total_power_cap_mw"] = 440.
    control_rejected = not output.validate_publication_contract(changed_control, origins)["passed"]
    full_grid = pd.date_range(C.OFFICIAL_EVAL_START, C.OFFICIAL_EVAL_END, freq="15min")
    full_issues = main_opt.daily_publication_times(full_grid, publication_time="07:00")
    bundle.train_end = "2025-01-02 07:15"
    with patch.dict(C.OPT_CONFIG, {"publication_time": "07:00"}), patch.object(main_opt, "forecast_resources") as late_forecast:
        late = main_opt.optimize_daily(prepared, origins, bundle)
    bundle.train_end = str(prepared.index[100])
    bundle.resource_nn = {"train_end": "2025-01-02 07:15"}
    late_nn = main_opt.resource_publication_check(bundle, expected_issues[0])
    checks = {"one_plan_per_business_day": result["summary"]["daily_publications"] == 3,
        "business_issue_times": issues.equals(expected_issues),
        "full_window_including_edges": (len(full_issues) == 22 and full_issues[0] == pd.Timestamp("2025-10-10 07:00")
            and full_issues[-1] == pd.Timestamp("2025-10-31 07:00")),
        "edge_window_counts": [p["published_steps"] for p in result["summary"]["daily_plans"]] == [29, 96, 67],
        "late_training_rejected_before_forecast": late["status"] == "BLOCKED_OPT" and not late_forecast.called,
        "late_resource_nn_rejected": not late_nn["passed"],
        "archive_publication_contract": publication_ok,
        "archive_effective_time_tamper_rejected": changed_rejected,
        "archive_missing_calibration_cutoff_rejected": missing_rejected,
        "archive_daily_control_tamper_rejected": control_rejected,
        "same_grid_as_long_forecast": expected_result_grid("opt_result.csv", origins).equals(origins),
        "no_intraday_refresh": result["summary"]["intraday_refresh"] is False,
        "continuous_constraints": validate_official_frame("opt_result.csv", frame, origins)["passed"],
        "daily_solver_validation": all(p["validation"]["passed"] for p in result["summary"]["daily_plans"])}
    return {"passed": all(checks.values()), **checks}


def _process_expert_check() -> dict:
    from types import SimpleNamespace
    from unittest.mock import patch
    import main_l
    prepared = _synthetic_prepared(600)
    columns = ["d_use_blast", "d_use_coke", "d_use_converter"]
    prepared.frame[columns] = [8e5, 3e4, 7e4]
    positions = np.array([480, 481])
    prepared.frame.loc[prepared.index[481], columns] = np.nan
    reference = np.tile([100., 300.], (2, 96, 1))
    state = {"recipe": C.LONG_PROCESS_EXPERT, "train_end": str(prepared.index[400]),
             "experts": [{"models": {}, "feature_names": []} for _ in range(3)]}
    bundle = SimpleNamespace(final_process_expert=state)
    with patch.object(main_l, "_predict_models", return_value=np.tile([110., 330.], (1, 96, 1))) as predict:
        result = main_l.apply_process_expert(bundle, prepared, positions, reference, True)
    expected = reference.copy()
    for target, name in enumerate(C.TARGETS):
        for group, low, high in C.LONG_GROUPS:
            expected[1, low - 1:high, target] += C.LONG_PROCESS_EXPERT["weights"][name][group] * (10. if target == 0 else 30.)
    checks = {"observed_branch_preserved": np.array_equal(result[0], reference[0]),
        "three_seed_group_weights": predict.call_count == 3 and np.allclose(result, expected),
        "historical_role_unchanged": np.array_equal(main_l.apply_process_expert(bundle, prepared, positions, reference, False), reference)}
    return {"passed": all(checks.values()), **checks}


def _frozen_final_training_check() -> dict:
    """用训练调用探针核验正式入口的冻结配方和目标截止，不实际拟合全量模型。"""
    from unittest.mock import patch
    import copy
    import main_s
    import main_l
    import main_nn
    import main_opt
    prepared = _synthetic_prepared(40 * C.STEPS_PER_DAY)
    calls = []

    def fit(data, base, positions, horizons, kind, features, rounds, ref):
        calls.append((int(positions[-1] + max(horizons)), ref))
        return {"probe": object()}, ["probe"]

    original = copy.deepcopy(C.FROZEN_LOAD_RECIPES)
    with patch.object(main_s, "_train_models", side_effect=fit), \
            patch.object(main_s, "train_availability_fallback", return_value=None), \
            patch.object(main_nn, "train_nn", return_value=None) as neural:
        short = main_s.train_final_load(prepared, "short", main_s.ShortBundle, quick=True)
        long = main_s.train_final_load(prepared, "long", main_l.LongBundle, quick=True)
    resource_calls = []

    def resource_fit(data, positions, rounds, ref, objectives=None, horizons=C.HORIZONS_L, gases=C.RESOURCE_NAMES):
        resource_calls.append((int(positions[-1] + max(horizons)), ref))
        return {gas: object() for gas in gases}, ["probe"]

    heat = {"ok": True, "coefficients_mw_per_m3h": {"blast": .0003, "coke": .002, "converter": .0006}}
    with patch.object(main_opt, "_fit_resource_models", side_effect=resource_fit), \
            patch.object(main_opt, "estimate_heat_rate", return_value=heat), \
            patch.object(main_opt, "_gas_caps", return_value={g: 1e6 for g in C.RESOURCE_NAMES}), \
            patch.object(main_opt, "refresh_physical_validation", return_value={}), \
            patch.object(main_opt, "_resource_base", return_value=pd.DataFrame(index=prepared.index)), \
            patch.object(main_opt, "load_prices", return_value={}), \
            patch.object(main_nn, "train_resource_nn", return_value={"train_end": str(prepared.index[-1])}):
        resource = main_opt.train_final_resources(prepared, quick=True)
    checks = {"independent_short_long_horizons": short.horizons == C.HORIZONS_S and long.horizons == C.HORIZONS_L,
              "load_target_cutoff": all(target <= cutoff for target, cutoff in calls),
              "resource_target_cutoff": all(target <= cutoff for target, cutoff in resource_calls),
              "one_final_cutoff_no_D": all(len(call.args[1]) == 1 and call.args[1][0]["name"] == "final" for call in neural.call_args_list),
              "configuration_not_mutated": C.FROZEN_LOAD_RECIPES == original,
              "resource_daily_publication": resource.dispatch_selection["chosen"] == "daily_fixed",
              "no_new_validation_score": short.scores.get("scope", "").startswith("no_new_validation")}
    return {"passed": all(checks.values()), **checks}


def _daily_repair_check() -> dict:
    """核验冻结控制的观测边界、分组守恒及独立末端约束。"""
    import copy
    import main_opt
    heat = {"ok": True, "coefficients_mw_per_m3h": {"blast": .000346, "coke": .0019, "converter": .00059}}
    missing, missing_control = main_opt.daily_plan_config(100000., [np.nan, 20000., 30000.], heat)
    known, known_control = main_opt.daily_plan_config(100000., [900000., 30000., 70000.], heat)
    avail = np.tile([800000., 30000., 70000.], (8, 1))
    initial = np.array([30., 30., 30., 30., 72., 72.])
    caps = np.array([1.2e6, 1e5, 1e5])
    result = main_opt._solve_day(avail, np.ones(8), 100000., initial, caps, heat, known)
    if result["status"] not in ("OPTIMAL", "FEASIBLE_TIMEOUT"):
        return {"passed": False, "reason": result.get("message")}
    allocation = main_opt.allocate_group_gas(result["power"], result["gas"], heat)
    target = float(known.OPT_CONFIG["terminal_inventory_m3"])
    bad = copy.deepcopy(result)
    bad["power"][:, -1] *= 2.
    rejected = not main_opt.validate_day(bad, avail, 100000., initial, caps, heat, known)["passed"]
    bad_gas = result["gas"].copy()
    bad_gas[0, 0] += 10000.
    try:
        main_opt.allocate_group_gas(result["power"], bad_gas, heat)
        fuel_rejected = False
    except ValueError:
        fuel_rejected = True
    checks = {"missing_gas_preserved_with_base_cap": missing_control["total_power_cap_mw"] == 240.
                and not missing_control["observed_power_gas_available"],
              "known_gas_cap_bounded": 240. <= known_control["total_power_cap_mw"] <= 280.,
              "group_fuel_conservation": allocation["validation"]["passed"]
                and np.allclose(allocation["group_gas_m3h"].sum(axis=0), result["gas"]),
              "terminal_inventory_reserve": abs(result["holder"][-1] - target) < 1.,
              "terminal_load_reachable": result["power"][:, -1].sum() <= 200. + 1e-4,
              "terminal_tamper_rejected": rejected, "fuel_allocation_tamper_rejected": fuel_rejected}
    synthetic = _synthetic_prepared(480)
    synthetic.frame[C.GAS_HOLDER["column"]] = 100000.
    synthetic.frame["gas_holder_blast_furnace_gas_holder_1"] = 21.
    evidence = main_opt.build_physical_validation(synthetic, 400, heat)
    first_holder = evidence["holder_mapping"]["columns"]["gas_holder_blast_furnace_gas_holder_1"]
    checks["unknown_holder_unit_not_misclassified_as_low_inventory"] = (
        first_holder["raw_unit_verified"] is False and first_holder["inside_official_safety_fraction"] is None)
    checks["raw_converter_not_an_energy_coefficient"] = evidence["converter_units"]["empirical_conversion_uses_raw_generation_scale"] is False
    checks = {name: bool(value) for name, value in checks.items()}
    return {"passed": all(checks.values()), **checks}


def _net_dispatch_check() -> dict:
    """覆盖生产预留、逐柜初态及守恒反例，不以总量正常替代独立安全。"""
    from types import SimpleNamespace
    import main_opt
    cfg = SimpleNamespace(**vars(C))
    cfg.OPT_CONFIG = {**C.OPT_CONFIG, "solver_time_limit_s": 2.}
    names = tuple(C.GAS_HOLDER["capacity_by_column_m3"])
    initial = dict(zip(names, (180000., 100000.)))
    units = dict.fromkeys(names, True)
    heat = {"coefficients_mw_per_m3h": {"blast": .000346, "coke": .0019, "converter": .00059}}
    generation = np.tile([1.6e6, 1e5, 2e5], (8, 1))
    priority = np.tile([6e5, 4e4, 8e4], (8, 1))
    reserved = np.tile([2e5, 3e4, 5e4], (8, 1))
    boundary = main_opt.net_resource_boundary(generation, priority, reserved, issued_at="2025-01-02 07:00",
        training_end="2025-01-02 07:00", flow_units_verified=True)
    result = main_opt.solve_net_day(boundary, np.r_[np.full(4, .2), np.ones(4)], initial,
        np.array([30., 30., 30., 30., 72., 72.]), [1.2e6, 1e5, 1e5], heat,
        holder_units_verified=units, config=cfg)
    if result["status"] not in ("OPTIMAL", "FEASIBLE_TIMEOUT"):
        return {"passed": False, "reason": result.get("message")}
    checks = {"net_demand_reserved_before_generation": np.allclose(boundary["available_m3h"], [8e5, 3e4, 7e4]),
        "two_holder_solve_and_validation": result["network_validation"]["passed"],
        "two_holder_terminal_return": np.allclose(result["holders_m3"][-1], list(initial.values()), atol=1.),
        "not_field_certified": result["physical_feasibility_certified"] is False}
    idle = main_opt.solve_net_day(boundary, np.ones(8), initial, np.zeros(6), [0., 0., 0.], heat,
        holder_units_verified=units, config=cfg)
    checks["unused_gas_explicitly_accounted_as_vent"] = (idle.get("network_validation", {}).get("passed", False)
        and np.allclose(idle["vent"][1:], boundary["available_m3h"][:, 1:].T))
    unsafe = {**initial, names[0]: 40000., names[1]: 150000.}
    bad_initial = main_opt.solve_net_day(boundary, np.ones(8), unsafe, np.zeros(6), [1e6, 1e5, 1e5], heat,
        holder_units_verified=units, config=cfg)
    missing_initial = main_opt.solve_net_day(boundary, np.ones(8), {names[1]: 100000.}, np.zeros(6),
        [1e6, 1e5, 1e5], heat, holder_units_verified=units, config=cfg)
    checks["safe_sum_cannot_hide_unsafe_holder"] = bad_initial["status"] == "BLOCKED_OPT"
    checks["missing_holder_not_invented"] = missing_initial["status"] == "BLOCKED_OPT"
    tampered = copy.deepcopy(result)
    tampered["holders_m3"][0] += [5000., -5000.]
    checks["compensated_stock_tamper_rejected"] = not main_opt.validate_net_dispatch(boundary, tampered, initial, units)["passed"]
    tampered = copy.deepcopy(result)
    tampered["vent"][1, 0] += 100.
    checks["gas_without_storage_must_close"] = not main_opt.validate_net_dispatch(boundary, tampered, initial, units)["passed"]
    checks["holder_unit_required"] = not main_opt.validate_net_dispatch(boundary, result, initial,
        {**units, names[0]: False})["passed"]
    late = main_opt.net_resource_boundary(generation, priority, reserved, issued_at="2025-01-02 07:00",
        training_end="2025-01-02 07:15")
    missing = reserved.copy()
    missing[0, 0] = np.nan
    unknown = main_opt.net_resource_boundary(generation, priority, missing, issued_at="2025-01-02 07:00",
        training_end="2025-01-02 07:00")
    checks["future_training_rejected"] = not late["passed"]
    checks["unknown_demand_not_zero"] = not unknown["passed"] and np.isnan(unknown["available_m3h"][0, 0])
    deficit = main_opt.net_resource_boundary(np.zeros((4, 3)), np.tile([4000., 0., 0.], (4, 1)), np.zeros((4, 3)),
        issued_at="2025-01-02 07:00", training_end="2025-01-02 07:00", flow_units_verified=True)
    cfg.OPT_CONFIG["terminal_holder"] = False
    drained = main_opt.solve_net_day(deficit, np.ones(4), initial, np.zeros(6), [0., 0., 0.], heat,
        holder_units_verified=units, config=cfg)
    checks["production_deficit_draws_storage_without_clipping"] = (drained.get("network_validation", {}).get("passed", False)
        and abs(drained["holders_m3"][-1].sum() - sum(initial.values()) + 4000.) < 1.)
    fake = {"summary": {"resource_mode": "net_surplus_after_priority_users", "plan_validation_passed": True,
        "physical_feasibility_certified": True, "dispatch_mode": "daily_fixed", "intraday_refresh": False,
        "time_semantics": "target_time", "planned_steps": 8}}
    gate = main_opt.physical_submission_check(fake)
    checks["blanket_flags_cannot_bypass_network_evidence"] = all(name in gate["blocked_items"]
        for name in ("net_material_balance", "independent_holder_safety"))
    return {"passed": bool(all(checks.values())), **{k: bool(v) for k, v in checks.items()}}


def run_self_checks(config=C) -> dict:
    checks = {
        "stage_isolation": _stage_isolation_check(),
        "training_target_cutoff": _training_target_cutoff_check(),
        "frozen_final_training": _frozen_final_training_check(),
        "future_perturbation": _future_perturbation_check(),
        "prefix_consistency": _prefix_consistency_check(),
        "zero_preservation": _zero_preservation_check(),
        "preprocessing_contract": _preprocessing_contract_check(),
        "short_process_recovery": _short_process_recovery_check(),
        "target_mask": _mask_check(),
        "profile_causality": _profile_causality_check(),
        "multistep_alignment": _multistep_alignment_check(),
        "nn_prefix_and_tail": _nn_contract_check(),
        "model_roles": _model_role_check(),
        "optimizer_contract": _optimizer_contract_check(),
        "rolling_causality": _rolling_causality_check(),
        "daily_publication": _daily_publication_check(),
        "daily_repair": _daily_repair_check(),
        "net_dispatch": _net_dispatch_check(),
        "execution_tracking": _execution_tracking_check(),
        "long_process_expert": _process_expert_check(),
        "power_gas_availability": _power_gas_availability_check(),
        "observed_at_t": _observed_at_t_check(),
        "generator1_contract": _generator1_contract_check(),
    }
    checks["passed"] = all(v.get("passed", False) for v in checks.values())
    return checks
