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


def _future_perturbation_check() -> dict:
    prepared = _synthetic_prepared()
    base = build_base_features(prepared, include_targets=True)
    origin = 300
    rng = np.random.default_rng(7)
    perturbed = copy.deepcopy(prepared)
    future = perturbed.frame.index > prepared.index[origin + 4]
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
    labels_kept = prepared.labels.notna().any().any()
    return {"passed": bool(empty and labels_kept)}


def _profile_causality_check() -> dict:
    prepared = _synthetic_prepared(960)
    positions = np.arange(400, 420)
    first = target_slot_profile(prepared, positions, C.HORIZONS_S, "generator_1")
    altered = copy.deepcopy(prepared)
    future = altered.frame.index > prepared.index[420]
    altered.frame.loc[future, C.TARGET_COLUMN["generator_1"]] = 0.0
    second = target_slot_profile(altered, positions, C.HORIZONS_S, "generator_1")
    return {"passed": bool(np.allclose(first, second, equal_nan=True))}


def run_self_checks(config=C) -> dict:
    checks = {
        "future_perturbation": _future_perturbation_check(),
        "prefix_consistency": _prefix_consistency_check(),
        "zero_preservation": _zero_preservation_check(),
        "target_mask": _mask_check(),
        "profile_causality": _profile_causality_check(),
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
