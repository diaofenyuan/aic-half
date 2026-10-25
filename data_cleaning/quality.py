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
    problems: list[str] = []
    if not prepared.index.is_monotonic_increasing:
        problems.append("时间索引非升序")
    if prepared.index.has_duplicates:
        problems.append("时间索引存在重复")
    if not prepared.labels.index.equals(prepared.index):
        problems.append("标签索引与特征索引不一致")
    if prepared.quality.get("data_scope") == "training_only":
        if prepared.index[-1] > pd.Timestamp(C.TRAIN_END):
            problems.append("训练数据超过允许截止点")
        if any(s["role"] != "train" for s in prepared.scan.get("stages", {}).values()):
            problems.append("训练数据包含非训练阶段")
    for target in C.TARGETS:
        if C.TARGET_COLUMN[target] not in prepared.frame.columns:
            problems.append(f"缺少目标列 {C.TARGET_COLUMN[target]}")
    if prepared.mask_start is not None:
        present = prepared.frame.loc[prepared.frame.index >= prepared.mask_start,
                                     [C.TARGET_COLUMN[t] for t in C.TARGETS]]
        if present.notna().any().any():
            problems.append("评测窗口目标未被完全遮挡")
    return {"passed": not problems, "problems": problems,
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


def _mask_check() -> dict:
    prepared = _synthetic_prepared(240)
    masked = prepared.masked_copy(prepared.index[100], prepared.index[-1])
    cols = [C.TARGET_COLUMN[t] for t in C.TARGETS]
    empty = masked.frame.loc[masked.index >= masked.index[100], cols].isna().all().all()
    labels_kept = prepared.labels.equals(masked.labels)
    return {"passed": bool(empty and labels_kept)}


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
                          forecast_resources, forecast_resources_quantile, _stack)

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
    bundle = ResourceBundle(pool, pool, names, quantile_models={"p10": pool})
    for forecast in (lambda o: forecast_resources(bundle, prepared, o),
                     lambda o: forecast_resources_quantile(bundle, prepared, o, "p10")):
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
    network = main_nn.DLinear(8, 2, 3, cfg={"head_mode": "residual"})
    x, target = torch.randn(12, 8, 3), torch.randn(12, 2, 2)
    optimizer = torch.optim.SGD(network.parameters(), lr=.05)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (network(x) - target).square().mean().backward()
        optimizer.step()
    gradient = bool(network.linear_trend.weight.grad.abs().sum() > 0)
    sensitive = not torch.allclose(network(x), network(x + 2.))
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
                          calibrate_resources_online, _adjust_available_resources)
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
    for _ in range(1200):
        executed = _execute_proxy_step(np.array([50., 50., 50., 50., 100., 100.]),
            np.array([8e5, 3e4, 7e4]), rng.random(3) * [8e5, 3e4, 7e4],
            level, previous, caps, heat)
        level, previous = executed["holder"], executed["power"]
    stress_passed = 30000. <= level <= 180000.
    cfg = SimpleNamespace(**vars(C))
    cfg.OPT_CONFIG = {**C.OPT_CONFIG, "execution_gas_mode": "substitute"}
    request = np.array([50., 50., 50., 50., 0., 0.])
    mapping = {"coefficients_mw_per_m3h": {"blast": .0004, "coke": .002, "converter": .0006}}
    clipped = _execute_proxy_step(request, np.array([2.5e5, 5e4, 0.]),
        np.array([6e5, 0., 0.]), 1e5, request, caps, mapping)
    substituted = _execute_proxy_step(request, np.array([2.5e5, 5e4, 0.]),
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
    for _ in range(1200):
        executed = _execute_proxy_step(np.array([50., 50., 50., 50., 100., 100.]),
            np.array([8e5, 3e4, 7e4]), rng.random(3) * [8e5, 3e4, 7e4],
            level, previous, caps, heat, cfg)
        level, previous = executed["holder"], executed["power"]
    substitute_safe = 30000. <= level <= 180000.
    times = pd.date_range("2025-01-01 23:45", periods=2, freq="15min")
    tariff = {1: np.r_[np.full(95, .2), 1.]}
    metrics = _dispatch_metrics(times, np.array([100., 200.]), tariff)
    expected = 140. / (tariff[1].mean() * 300.)
    metrics_ok = (abs(metrics["price_weighted_load_ratio"] - expected) < 1e-9
                  and len(metrics["daily"]) == 2
                  and not _dispatch_metrics(times, np.array([100., np.nan]), tariff)["complete"])
    paths = np.full((8, 96, 3), 100.)
    observations = np.full((12, 3), 120.)
    calibrated = calibrate_resources_online(paths, observations, np.arange(8))
    observations[4:] = 0.
    perturbed = calibrate_resources_online(paths, observations, np.arange(8))
    calibration_causal = (np.array_equal(calibrated[:4], perturbed[:4])
                          and calibrated[-1, 0, 0] > 100.)
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
                            and substitute_safe and recovered and overflow_recovered and metrics_ok and calibration_causal
                            and resource_adjustment_ok
                            and price.tolist() == [1.0, 2.0]),
            "solver_status": result["status"], "tamper_detected": bool(detected),
            "continuous_resource_shock_steps": 1200,
            "substitute_resource_shock_steps": 1200, "fuel_substitution_recovered_power": bool(recovered),
            "overflow_substitution_preserves_power_and_holder": bool(overflow_recovered),
            "daily_price_metric_and_missing_values": bool(metrics_ok),
            "online_resource_calibration_causal": bool(calibration_causal),
            "peak_resource_adjustment_causal_missing_safe": bool(resource_adjustment_ok),
            "zero_resource_curtailment": True, "resource_zeros_counted": bool(zero_counted)}


def _rolling_causality_check() -> dict:
    from unittest.mock import patch
    import main_opt
    prepared = _synthetic_prepared(480)
    prepared.frame[C.GAS_HOLDER["column"]] = 100000.
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
        return main_opt.optimize(forecast, data, origins, np.zeros((len(origins), 96, 2)), bundle)

    with patch.object(main_opt, "load_prices", return_value={1: np.ones(96)}), patch.object(main_opt, "_solve_day", side_effect=solve):
        original = run(prepared)
        changed_holder = copy.deepcopy(prepared)
        changed_holder.frame.loc[origins[0] + pd.Timedelta(minutes=15):, C.GAS_HOLDER["column"]] = 170000.
        same_holder = run(changed_holder)
        changed_future = copy.deepcopy(prepared)
        changed_future.frame.loc[origins[0] + pd.Timedelta(minutes=15):,
                                 ["d_use_blast", "d_use_coke", "d_use_converter"]] = 0.
        same_first_commit = run(changed_future)
    holder_not_injected = original["schedule"].equals(same_holder["schedule"])
    no_oracle = original["schedule"].iloc[:4].equals(same_first_commit["schedule"].iloc[:4])
    continuous = bool(len(observed_states) == 6 and observed_states[1] != 100000.)
    return {"passed": bool(holder_not_injected and no_oracle and continuous),
            "historical_holder_not_injected": holder_not_injected,
            "first_commit_future_independent": no_oracle, "continuous_state": continuous}


def _plan_research_check() -> dict:
    from types import SimpleNamespace
    import lightgbm as lgb
    from .pipeline import fit_profile_state, state_slot_profile
    from main_opt import (ZeroMixtureResource, load_resource_model, fit_inventory_risk,
                          _holder_plan_bounds, _solve_day, _fit_resource_models)
    from main_s import (_train_models, predict_allocation, design_inputs,
                        future_resource_features)
    prepared = _synthetic_prepared(40 * 96)
    prepared.frame["d_use_share_coke"] = prepared.frame.d_use_coke / prepared.frame.d_use_total
    prepared.frame["d_use_share_converter"] = prepared.frame.d_use_converter / prepared.frame.d_use_total
    recipe = fit_profile_state(prepared, 20 * 96)
    origin = 25 * 96 + 12
    masked = prepared.masked_copy(prepared.index[20 * 96], prepared.index[-1])
    profile = state_slot_profile(masked, np.array([origin]), (1, 8, 96), "generator_1", recipe)[0]
    altered = copy.deepcopy(masked)
    altered.frame.loc[prepared.index[origin + 1]:] = 1e9
    altered.labels.loc[prepared.index[20 * 96]:] = 1e12
    same = state_slot_profile(altered, np.array([origin]), (1, 8, 96), "generator_1", recipe)[0]
    future_independent = np.array_equal(profile, same)
    missing = copy.deepcopy(masked)
    missing.frame.loc[prepared.index[origin], recipe["columns"]] = np.nan
    fallback = state_slot_profile(missing, np.array([origin]), (1, 8, 96), "generator_1", recipe)[0]
    missing_safe = np.array_equal(fallback, target_slot_profile(missing, np.array([origin]), (1, 8, 96), "generator_1"))
    zero = copy.deepcopy(prepared)
    zero.frame[C.TARGET_COLUMN["generator_1"]] = 0.
    zeros_kept = np.all(state_slot_profile(zero, np.array([origin]), (1, 8, 96), "generator_1", recipe)[0] == 0.)
    X = np.arange(120., dtype=float).reshape(-1, 1)
    params = {**C.LGBM_BASE, "min_data_in_leaf": 5, "num_threads": 1}
    classifier = lgb.train({**params, "objective": "binary"}, lgb.Dataset(X, label=(X[:, 0] < 30)), num_boost_round=3)
    positive = lgb.train({**params, "objective": "regression_l2"}, lgb.Dataset(X, label=X[:, 0] + 1), num_boost_round=3)
    mixture = ZeroMixtureResource(classifier, positive)
    mixture_ok = np.array_equal(mixture.predict(X), load_resource_model(mixture.model_to_string()).predict(X))
    mixture_ok &= np.allclose(mixture.predict(X), (1. - classifier.predict(X)) * positive.predict(X))
    forecast = np.full((192, 96, 3), 100.)
    risk = fit_inventory_risk(forecast, forecast.copy(), "2025-05-30", "2025-05-31")
    impossible = fit_inventory_risk(forecast, forecast + 1e5, "2025-05-30", "2025-05-31")
    cfg = SimpleNamespace(**vars(C))
    cfg.OPT_CONFIG = {**C.OPT_CONFIG, "inventory_risk": impossible}
    low, high = _holder_plan_bounds(96, cfg)
    bad = _solve_day(forecast[0], np.ones(96), 1e5, np.zeros(6), np.full(3, 1e6),
        {"coefficients_mw_per_m3h": {g: .001 for g in C.RESOURCE_NAMES}}, cfg)
    risk_ok = risk["feasible"] and not impossible["feasible"] and (low > high).any() and bad["status"] == "RISK_INFEASIBLE"
    units = fit_inventory_risk(forecast, forecast + 100., "2025-05-30", "2025-05-31")
    risk_ok &= units["cumulative_diagnostics"]["blast"]["cumulative_paths"]["24h"]["mean_m3"] == 2400.
    positions = np.arange(origin, origin + 3)
    future = future_resource_features(masked, positions, forecast[:3], np.zeros_like(forecast[:3]))
    future_changed = future_resource_features(altered, positions[:1], forecast[:1], np.zeros_like(forecast[:1]))
    features_causal = all(np.array_equal(value[:1], future_changed[key], equal_nan=True) for key, value in future.items())
    base = pd.DataFrame({"x": np.zeros(len(prepared.index))}, index=prepared.index)
    _, names = design_inputs(masked, base, positions, (1, 8, 96), "gas")
    class ConstantModel:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value)
    protected = np.zeros((3, 3, 2))
    protected[:, :, 1] = np.array([0., 150., 440.])[:, None]
    allocation = predict_allocation(masked, base, positions, (1, 8, 96),
        {"allocation": ConstantModel(.9)}, names, protected)
    allocation_ok = (np.array_equal(allocation[:, :, 1], protected[:, :, 1]) and
        (allocation[:, :, 0] <= 200.).all() and (allocation[:, :, 1] - allocation[:, :, 0] <= 240.).all())
    import main_opt
    original_prices = main_opt.load_prices
    try:
        main_opt.load_prices = lambda: {1: np.ones(96)}
        resource_base = main_opt._resource_base(masked)
        _, resource_names = main_opt._resource_design(masked, resource_base, positions, C.HORIZONS_L)
        direct = {g: ConstantModel(1000.) for g in C.RESOURCE_NAMES}
        selection = {"availability_adjustment": {"method": "positive_persistence_peak", "threshold_ratio": .05, "weight": 1.}}
        heat = {"ok": True, "coefficients_mw_per_m3h": {g: .001 for g in C.RESOURCE_NAMES}}
        baseline = main_opt.forecast_resources(main_opt.ResourceBundle(direct, {}, resource_names,
            selection=selection, heat_rate=heat), masked, masked.index[positions])
        replacement = {**direct, "coke": ConstantModel(1e6), **{g + "__reference": m for g, m in direct.items()}}
        trial = main_opt.forecast_resources(main_opt.ResourceBundle(replacement, {}, resource_names,
            selection={**selection, "frozen_adjustment_reference": True}, heat_rate=heat), masked, masked.index[positions])
        blast_fixed = np.array_equal(baseline[:, :, 0], trial[:, :, 0]) and np.max(baseline[:, :8, 0]) > 1000.
    finally:
        main_opt.load_prices = original_prices
    rejected = []
    for call in (lambda: fit_inventory_risk(forecast, forecast, "2025-06-01", "2025-05-31"),
                 lambda: _fit_resource_models(prepared, np.array([100]), 1, 100),
                 lambda: _train_models(prepared, pd.DataFrame(), np.array([100]), (1,), "global", "gas", 1, 100)):
        try:
            call()
        except ValueError:
            rejected.append(True)
        else:
            rejected.append(False)
    checks = {"state_profile_future_independent": bool(future_independent),
              "state_profile_missing_fallback": bool(missing_safe), "state_profile_zero_preserved": bool(zeros_kept),
              "mixture_mean_and_reload": bool(mixture_ok), "risk_infeasible_keeps_hard_bounds": bool(risk_ok),
              "prefix_boundary_rejections": all(rejected), "future_resource_features_causal": bool(features_causal),
              "allocation_preserves_total": bool(allocation_ok), "resource_replacement_protects_blast": bool(blast_fixed)}
    return {"passed": all(checks.values()), **checks}


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


def run_self_checks(config=C) -> dict:
    checks = {
        "stage_isolation": _stage_isolation_check(),
        "future_perturbation": _future_perturbation_check(),
        "prefix_consistency": _prefix_consistency_check(),
        "zero_preservation": _zero_preservation_check(),
        "target_mask": _mask_check(),
        "profile_causality": _profile_causality_check(),
        "multistep_alignment": _multistep_alignment_check(),
        "nn_prefix_and_tail": _nn_contract_check(),
        "model_roles": _model_role_check(),
        "optimizer_contract": _optimizer_contract_check(),
        "rolling_causality": _rolling_causality_check(),
        "plan_research": _plan_research_check(),
        "power_gas_availability": _power_gas_availability_check(),
    }
    checks["passed"] = all(v.get("passed", False) for v in checks.values())
    return checks


def real_data_check(config=C) -> dict:
    """真实数据的前缀一致性检查：只取一段前缀，避免整表重复清洗过慢。"""
    prepared = prepare_data(config)
    cutoff = prepared.index[int(len(prepared.index) * 0.6)]
    prefix = prepare_data(config, cutoff=cutoff)
    left = build_base_features(prepared, include_targets=True).loc[:cutoff - pd.Timedelta(minutes=15)]
    right = build_base_features(prefix, include_targets=True)
    common = left.index.intersection(right.index)
    same = np.allclose(left.loc[common].to_numpy(dtype=float),
                       right.loc[common].to_numpy(dtype=float), equal_nan=True)
    return {"passed": bool(same), "cutoff": str(cutoff), "rows": int(len(common))}
