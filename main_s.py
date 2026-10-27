"""短期 2 小时预测：基线、Direct-H / Global-H LightGBM、折内选模与融合。

短期与长期各自训练、各自选参；本文件只提供无状态的拟合/设计工具供 main_l
复用，不共享任何已训练模型。
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import main_nn
import numpy as np
import pandas as pd

import config as C
from data_cleaning import (PreparedData, build_base_features, fuel_level_ratio,
                           last_valid_level, target_calendar,
                           target_slot_profile)


# ---------------------------------------------------------------------------
# 通用工具（main_l 复用）
# ---------------------------------------------------------------------------
def _mape_weights(y: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """把训练权重对齐 1-MAPE：小负荷样本误差占比大，应放大其权重。"""
    power = float(C.SAMPLE_WEIGHT_POWER or 0.0)
    base = np.ones(len(y), dtype=np.float32) if weights is None else weights.astype(np.float32)
    if power <= 0.0:
        return base
    scale = 1.0 / np.power(np.maximum(np.abs(y), 1.0), power)
    scale = scale / max(float(np.mean(scale)), 1e-9)
    return (base * scale).astype(np.float32)


def fit_booster(X: np.ndarray, y: np.ndarray, feature_names: list[str],
                params: dict, num_round: int, weights: np.ndarray | None = None,
                *, mape_weight: bool = True) -> lgb.Booster:
    import output
    if mape_weight:
        weights = _mape_weights(y, weights)
    cache = output.training_cache_path("boosters",
        {"params": params, "rounds": int(num_round), "features": feature_names,
         "weighted": weights is not None},
        arrays=(X, y) if weights is None else (X, y, weights))
    artifact = output.read_training_cache(cache)
    if artifact is not None:
        return lgb.Booster(model_str=artifact.read_text(encoding="utf-8"))
    dataset = lgb.Dataset(X, label=y, weight=weights, feature_name=feature_names,
                          free_raw_data=True)
    with C.progress(total=int(num_round), desc="LightGBM 迭代", unit="轮") as bar:
        def report_iteration(env):
            bar.update(1)

        model = lgb.train(params, dataset, num_boost_round=int(num_round),
                          callbacks=[report_iteration])
    def write_model(directory):
        path = directory / "model.txt"
        path.write_text(model.model_to_string(), encoding="utf-8")
        return path
    output.write_training_cache(cache, write_model)
    return model


def training_weights(prepared: PreparedData, positions: np.ndarray,
                     horizons: np.ndarray, ref_pos: int) -> np.ndarray:
    """按目标时刻到训练截止点的时间衰减；归一化到均值 1，避免改变梯度尺度。"""
    if C.DECAY_DAYS is None:
        return np.ones((len(positions), len(horizons)), dtype=np.float32)
    positions = np.asarray(positions, dtype=int)
    horizons = np.asarray(horizons, dtype=int)
    age_days = (ref_pos - (positions[:, None] + horizons[None, :])) / C.STEPS_PER_DAY
    weights = np.power(0.5, np.maximum(age_days, 0.0) / float(C.DECAY_DAYS))
    return (weights / max(float(np.mean(weights)), 1e-9)).astype(np.float32)


def design_inputs(prepared: PreparedData, base: pd.DataFrame,
                  positions: np.ndarray, horizons: tuple[int, ...],
                  features: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """构造与提前量相关的目标时刻日历、画像特征。"""
    extras = target_calendar(prepared, positions, horizons)
    # 少特征分支不得通过燃料缩放重新读到被排除的发电耗气。
    ratio = (np.ones(len(positions)) if features.removesuffix("_at_t") == "gas_no_power"
             else fuel_level_ratio(prepared, positions))
    extras["fuel_profile_ratio"] = np.repeat(ratio[:, None], len(horizons), axis=1)
    # 同槽位画像与最近可见水平都来自起报点之前的历史，部署时同样可得；
    # 它们不是未来真值，放入所有特征集可避免树模型只靠当前气量外推。
    for target in C.TARGETS:
        extras[f"profile_{target}"] = target_slot_profile(
            prepared, positions, horizons, target)
    names = list(base.columns) + list(extras.keys()) + ["h"]
    return extras, names


def _block(base_values: np.ndarray, positions: np.ndarray, extras: dict,
           extra_names: list[str], h: int, j: int) -> np.ndarray:
    parts = [base_values[positions]]
    for name in extra_names:
        parts.append(extras[name][:, j][:, None])
    parts.append(np.full((len(positions), 1), float(h), dtype=np.float32))
    return np.hstack(parts).astype(np.float32)


def build_training_block(prepared: PreparedData, base: pd.DataFrame,
                         base_values: np.ndarray, extras: dict, extra_names: list[str],
                         positions: np.ndarray, horizons: tuple[int, ...], j: int,
                         target: str, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = int(horizons[j])
    X = _block(base_values, positions, extras, extra_names, h, j)
    y = prepared.labels[target].to_numpy(dtype=float)[positions + h]
    valid = np.isfinite(y)
    return X[valid], y[valid], weights[:, j][valid]


def stack_features(base_values: np.ndarray, extras: dict, extra_names: list[str],
                   positions: np.ndarray, horizons: tuple[int, ...]) -> np.ndarray:
    """保留提前量优先的样本顺序，直接写入目标数组，避免逐步拼接与类型转换副本。"""
    n, m, width = len(positions), len(horizons), base_values.shape[1]
    blocks = np.empty((m, n, width + len(extra_names) + 1), dtype=np.float32)
    blocks[:, :, :width] = base_values[positions]
    for j, name in enumerate(extra_names):
        blocks[:, :, width + j] = extras[name].T
    blocks[:, :, -1] = np.asarray(horizons)[:, None]
    return blocks.reshape(n * m, -1)


def stack_training(prepared: PreparedData, base_values: np.ndarray, extras: dict,
                   extra_names: list[str], positions: np.ndarray,
                   horizons: tuple[int, ...], target: str,
                   weights: np.ndarray, design: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = (stack_features(base_values, extras, extra_names, positions, horizons)
         if design is None else design)
    y = prepared.labels[target].to_numpy(dtype=float)[
        positions[None, :] + np.asarray(horizons)[:, None]].ravel()
    w = weights.T.astype(np.float32).ravel()
    valid = np.isfinite(y)
    if valid.all():
        return X, y, w
    return X[valid], y[valid], w[valid]


def enforce_physics(cube: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """只修正不可能值并保证 all>=g1；不掩盖模型误差。"""
    cube = np.asarray(cube, dtype=float).copy()
    if fallback is not None:
        bad = ~np.isfinite(cube)
        if bad.any():
            cube[bad] = fallback[bad]
    cube[..., 0] = np.clip(cube[..., 0], 0.0, C.CAPACITY["generator_1"])
    cube[..., 1] = np.clip(cube[..., 1], 0.0, C.CAPACITY["generator_all"])
    cube[..., 1] = np.maximum(cube[..., 1], cube[..., 0])
    cube[..., 1] = np.minimum(cube[..., 1], cube[..., 0] + C.STRUCTURAL_CAPACITY["generator_big"])
    return cube


def score_cell(pred: np.ndarray, truth: np.ndarray, target_idx: int,
               horizon_indices: list[int]) -> float:
    actual = truth[:, horizon_indices, target_idx]
    estimate = pred[:, horizon_indices, target_idx]
    valid = np.isfinite(actual) & (actual > 0) & np.isfinite(estimate)
    if not valid.any():
        return float("nan")
    relative = np.abs(estimate[valid] - actual[valid]) / actual[valid]
    return float(100.0 * (1.0 - np.mean(relative)))


def score_cube(pred: np.ndarray, truth: np.ndarray,
               horizons: tuple[int, ...]) -> dict:
    """按赛题 1-MAPE 口径评分；零负荷真值单列，不参与 MAPE 分母。"""
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if pred.shape != truth.shape or not np.isfinite(pred).all():
        raise ValueError("评分预测形状不匹配或含非有限值，禁止通过丢弃预测缺失抬高分数")
    result = {"targets": {}, "per_horizon": {}}
    target_scores = []
    for ti, target in enumerate(C.TARGETS):
        actual = truth[:, :, ti]
        estimate = pred[:, :, ti]
        valid = np.isfinite(actual) & (actual > 0) & np.isfinite(estimate)
        all_valid = np.isfinite(actual) & np.isfinite(estimate)
        rel = np.abs(estimate - actual) / np.where(actual > 0, actual, np.nan)
        score = 100.0 * (1.0 - float(np.nanmean(rel)))
        zero_mask = np.isfinite(actual) & (actual == 0) & np.isfinite(estimate)
        result["targets"][target] = {
            "accuracy": score,
            "mae": float(np.nanmean(np.abs(estimate[valid] - actual[valid]))) if valid.any() else None,
            "zero_truth": int(zero_mask.sum()),
            # 零负荷不进 MAPE 分母，但单列绝对误差，避免模型在停机段失控。
            "zero_load_mae": (float(np.nanmean(np.abs(estimate[zero_mask])))
                              if zero_mask.any() else None),
            "valid_points": int(valid.sum()),
            "bias_ratio": float(np.sum(estimate[all_valid] - actual[all_valid]) / np.sum(actual[all_valid])) if valid.any() else None,
            "prediction_std_mw": float(np.std(estimate[all_valid])) if all_valid.any() else None,
            "truth_std_mw": float(np.std(actual[all_valid])) if all_valid.any() else None,
            "lead_groups": {name: score_cell(pred, truth, ti,
                [j for j, h in enumerate(horizons) if lo <= h <= hi])
                for name, lo, hi in _groups_for(horizons)},
        }
        target_scores.append(score)
        result["per_horizon"][target] = {
            str(h): score_cell(pred, truth, ti, [j]) for j, h in enumerate(horizons)}
    result["accuracy"] = float(np.mean(target_scores))
    result["horizons"] = [int(horizons[0]), int(horizons[-1])]
    return result


def truth_cube(prepared: PreparedData, positions: np.ndarray,
               horizons: tuple[int, ...]) -> np.ndarray:
    """按目标时刻 t+h 取标签，供 1-MAPE 评价。

    训练堆叠用的是 positions+h 的真值；评分若退回起报点标签会系统性
    扭曲选模，长周期尤其失真。越界提前量保持 NaN，由 score_cube 排除。
    """
    labels = prepared.labels[list(C.TARGETS)].to_numpy(dtype=float)
    n_pos = len(positions)
    out = np.full((n_pos, len(horizons), len(C.TARGETS)), np.nan, dtype=float)
    pos = np.asarray(positions, dtype=np.int64)
    for j, h in enumerate(horizons):
        idx = pos + int(h)
        ok = (idx >= 0) & (idx < len(labels))
        out[ok, j, :] = labels[idx[ok]]
    return out


def baseline_cubes(prepared: PreparedData, positions: np.ndarray,
                   horizons: tuple[int, ...]) -> dict[str, np.ndarray]:
    n, m = len(positions), len(horizons)
    raw_ratio = fuel_level_ratio(prepared, positions)
    # 燃料缺测时使用未缩放画像；只处理融合回退，不改模型输入或实际资源观测。
    ratio = np.where(np.isfinite(raw_ratio), raw_ratio, 1.0)[:, None]
    cubes = {"lastvalid": np.empty((n, m, 2)),
             "profile": np.empty((n, m, 2)),
             "profile_fuel": np.empty((n, m, 2))}
    for ti, target in enumerate(C.TARGETS):
        last = last_valid_level(prepared, positions, target)
        profile = target_slot_profile(prepared, positions, horizons, target)
        cubes["lastvalid"][:, :, ti] = np.repeat(last[:, None], m, axis=1)
        cubes["profile"][:, :, ti] = profile
        cubes["profile_fuel"][:, :, ti] = profile * ratio
    for name, cube in cubes.items():
        cubes[name] = enforce_physics(cube, fallback=cubes["profile"])
    return cubes


def blend_cube(pred: np.ndarray, profile: np.ndarray, blend: dict,
               horizons: tuple[int, ...], groups) -> np.ndarray:
    result = pred.copy()
    for ti, target in enumerate(C.TARGETS):
        for gname, lo, hi in groups:
            w = float(blend.get(target, {}).get(gname, 1.0))
            indices = [j for j, h in enumerate(horizons) if lo <= h <= hi]
            result[:, indices, ti] = (w * pred[:, indices, ti]
                                      + (1.0 - w) * profile[:, indices, ti])
    return enforce_physics(result, fallback=profile)


def select_blend(fold_preds: list[np.ndarray], fold_profiles: list[np.ndarray],
                 fold_truths: list[np.ndarray], horizons: tuple[int, ...],
                 groups) -> dict:
    blend: dict = {target: {} for target in C.TARGETS}
    grid = C.SELECTION["blend_grid"]
    for ti, target in enumerate(C.TARGETS):
        for gname, lo, hi in groups:
            indices = [j for j, h in enumerate(horizons) if lo <= h <= hi]
            best = (-np.inf, 1.0)
            reference = [score_cell(p[:, indices, ti][:, :, None],
                                    t[:, indices, ti][:, :, None], 0,
                                    list(range(len(indices))))
                         for p, t in zip(fold_preds, fold_truths)]
            for w in grid:
                scores = []
                for pred, profile, truth in zip(fold_preds, fold_profiles, fold_truths):
                    blended = pred[:, indices, ti] * w + profile[:, indices, ti] * (1.0 - w)
                    scores.append(score_cell(blended[:, :, None], truth[:, indices, ti][:, :, None],
                                             0, list(range(len(indices)))))
                mean = float(np.nanmean(scores))
                protected = all(a >= b - C.SELECTION["max_worst_fold_drop_pp"]
                                for a, b in zip(scores, reference))
                if protected and np.isfinite(mean) and mean > best[0]:
                    best = (mean, float(w))
            blend[target][gname] = best[1]
    return blend


def effective_model_name(kind: str, features: str, blend: dict) -> str:
    """融合权重为 0 时如实报告纯画像基线，不把未部署的树模型写成主模型。"""
    weights = [w for groups in blend.values() for w in groups.values()]
    if max(weights, default=0.0) <= 0.0:
        return "profile"
    return f"{kind}/{features}+profile"


def select_nn_tree_weights(neural, trees, truths, horizons, groups) -> dict:
    """所有开发折共用融合权重；以树为保护基线，避免总分掩盖单目标退化。"""
    weights = {t: {} for t in C.TARGETS}
    for ti, target in enumerate(C.TARGETS):
        for name, lo, hi in groups:
            indices = [j for j, h in enumerate(horizons) if lo <= h <= hi]
            baseline = [score_cell(p, y, ti, indices) for p, y in zip(trees, truths)]
            best = (float(np.mean(baseline)), 0.)
            for weight in (0.25, 0.50, 0.75, 1.0):
                scores = [score_cell(weight * a + (1 - weight) * b, y, ti, indices)
                          for a, b, y in zip(neural, trees, truths)]
                if (all(a >= b - C.SELECTION["max_worst_fold_drop_pp"] for a, b in zip(scores, baseline))
                        and float(np.mean(scores)) > best[0]):
                    best = (float(np.mean(scores)), weight)
            weights[target][name] = best[1]
    return weights


def select_candidate(candidates: list[dict], reference_kind: str) -> dict:
    """相对同一基线逐折保护，避免把最差折与最高平均分混为一谈。"""
    reference = next(c for c in candidates if c["kind"] == reference_kind)
    tolerance = C.SELECTION["max_worst_fold_drop_pp"]
    eligible = [c for c in candidates
                if c["mean"] >= reference["mean"] + C.SELECTION["min_mean_gain_pp"]
                and all(a["accuracy"] >= b["accuracy"] - tolerance
                        and all(a["targets"][t]["accuracy"] >= b["targets"][t]["accuracy"] - tolerance
                                for t in C.TARGETS)
                        for a, b in zip(c["scores"], reference["scores"]))]
    return max(eligible or [reference], key=lambda c: c["mean"])


def evaluate_blended_candidates(candidates: list[dict], profiles: list[np.ndarray],
                                horizons: tuple[int, ...], groups) -> None:
    """以实际部署的融合结果排名，避免先淘汰互补候选再给赢家单独融合。"""
    for candidate in C.progress(candidates, desc="候选画像融合评价", unit="候选"):
        candidate["raw_mean"] = candidate["mean"]
        blend = select_blend(candidate["preds"], profiles, candidate["truths"], horizons, groups)
        predictions = [blend_cube(p, prof, blend, horizons, groups)
                       for p, prof in zip(candidate["preds"], profiles)]
        scores = [score_cube(p, t, horizons) for p, t in zip(predictions, candidate["truths"])]
        candidate.update(blend=blend, scores=scores,
                         mean=float(np.mean([s["accuracy"] for s in scores])),
                         worst=float(min(s["accuracy"] for s in scores)))


# ---------------------------------------------------------------------------
# 折内训练与评估
# ---------------------------------------------------------------------------
SHORT_CANDIDATES = (
    {"kind": "direct", "features": "gas"},
    {"kind": "global", "features": "gas"},
)


def _gas_columns(base: pd.DataFrame) -> list[str]:
    return [c for c in base.columns if not c.startswith(("tgt_", "last_", "at_t_"))]


def _feature_columns(base: pd.DataFrame, features: str) -> list[str]:
    """按特征集取列。

    - gas：纯气侧/工况，最稳。
    - gaslast：气侧 + last_*（起报点前最后一次可见负荷水平与年龄）。
      不含 tgt_* 原值——评测窗口内目标被遮挡，原值会变成 NaN，
      训练/服务不一致，历史上 gaslast 的前身 target 集因此掉到 82 分。
    - target：全列，仅作对照基线，不参与部署。
    """
    if features.endswith("_at_t"):
        return _feature_columns(base, features.removesuffix("_at_t")) + [c for c in base if c.startswith("at_t_")]
    cols = list(base.columns)
    if features == "gas_no_power":
        return [c for c in _gas_columns(base) if not c.startswith((
            "d_use_", "d_fuel_proxy", "obs_d_use_", "obs_d_fuel_proxy",
            "ratio_d_use_", "load_generator_use_"))]
    if features == "gas":
        return _gas_columns(base)
    if features == "gaslast":
        return [c for c in cols
                if not c.startswith("tgt_") or c.endswith(("_last_level", "_age"))]
    return cols


def _with_observed_features(prepared: PreparedData, base: pd.DataFrame, features: str) -> pd.DataFrame:
    if not features.endswith("_at_t"):
        return base
    columns = [c for c in prepared.frame if c.startswith("at_t_")]
    if not columns:
        raise ValueError("at_t 模型缺少截至起报时刻的分钟特征，请重新读取官方原始数据")
    return base.join(prepared.frame[columns])


def _groups_for(horizons: tuple[int, ...]):
    return C.SHORT_GROUPS if max(horizons) <= max(C.HORIZONS_S) else C.LONG_GROUPS


def _train_models(prepared: PreparedData, base_all: pd.DataFrame,
                  positions: np.ndarray, horizons: tuple[int, ...],
                  kind: str, features: str, num_round: int,
                  ref_pos: int,
                  targets: tuple | None = None, params_override: dict | None = None) -> tuple[dict, list[str]]:
    prepared.require_training_data()
    if len(positions) == 0 or np.max(positions) + max(horizons) > ref_pos:
        raise ValueError("负荷训练目标跨越冻结截止点")
    base_all = _with_observed_features(prepared, base_all, features)
    columns = _feature_columns(base_all, features)
    base = base_all[columns]
    base_values = base.to_numpy(dtype=np.float32)
    extras, names = design_inputs(prepared, base, positions, horizons, features)
    extra_names = names[len(columns):-1]
    weights = training_weights(prepared, positions, horizons, ref_pos)
    models: dict = {}
    params = {**C.LGBM_BASE, "num_threads": C.LGBM_BASE["num_threads"], **(params_override or {})}
    targets = C.TARGETS if targets is None else targets
    if kind == "direct":
        for target in C.progress(targets, desc="Direct-H 训练目标", unit="目标"):
            for j, h in C.progress(enumerate(horizons), total=len(horizons),
                                   desc=f"{target} 提前量", unit="模型"):
                X, y, w = build_training_block(
                    prepared, base, base_values, extras, extra_names,
                    positions, horizons, j, target, weights)
                models[(target, int(h))] = fit_booster(X, y, names, params, num_round, w)
    elif kind in ("global", "grouped"):
        groups = _groups_for(horizons)
        # 同一前缀的多目标共用特征；缺测标签仍按各目标分别筛选。
        design = (stack_features(base_values, extras, extra_names, positions, horizons)
                  if kind == "global" else None)
        for target in C.progress(targets, desc=f"{kind} 训练目标", unit="目标"):
            if kind == "global":
                X, y, w = stack_training(prepared, base_values, extras, extra_names,
                                         positions, horizons, target, weights, design=design)
                models[target] = fit_booster(X, y, names, params, num_round, w)
            else:
                for gname, lo, hi in C.progress(groups, desc=f"{target} 提前量分组", unit="模型"):
                    subset = tuple(h for h in horizons if lo <= h <= hi)
                    jidx = [horizons.index(h) for h in subset]
                    sub_extras = {k: v[:, jidx] for k, v in extras.items()}
                    X, y, w = stack_training(prepared, base_values, sub_extras, extra_names,
                                             positions, subset, target, weights[:, jidx])
                    models[(target, gname)] = fit_booster(X, y, names, params, num_round, w)
    else:
        raise ValueError(f"未知模型类型 {kind}")
    return models, names


def _predict_models(prepared: PreparedData, base_all: pd.DataFrame,
                    positions: np.ndarray, horizons: tuple[int, ...],
                    kind: str, features: str, models: dict,
                    feature_names: list[str],
                    targets: tuple | None = None) -> np.ndarray:
    base_all = _with_observed_features(prepared, base_all, features)
    columns = _feature_columns(base_all, features)
    base = base_all[columns]
    base_values = base.to_numpy(dtype=np.float32)
    extras, names = design_inputs(prepared, base, positions, horizons, features)
    extra_names = names[len(columns):-1]
    if names != feature_names:
        raise ValueError("预测特征顺序与训练不一致")
    out = np.full((len(positions), len(horizons), 2), np.nan, dtype=float)
    targets = C.TARGETS if targets is None else targets
    indices = [(C.TARGETS.index(target), target) for target in targets]
    if kind == "direct":
        for ti, target in C.progress(indices, total=len(indices),
                                     desc="Direct-H 预测目标", unit="目标"):
            for j, h in C.progress(enumerate(horizons), total=len(horizons),
                                   desc=f"{target} 预测提前量", unit="模型"):
                X = _block(base_values, positions, extras, extra_names, int(h), j)
                out[:, j, ti] = models[(target, int(h))].predict(X)
    elif kind == "global":
        for ti, target in C.progress(indices, total=len(indices),
                                     desc="Global-H 批量预测", unit="目标"):
            X = np.empty((len(positions) * len(horizons), len(names)), dtype=np.float32)
            for j, h in enumerate(horizons):
                lo = j * len(positions)
                X[lo:lo + len(positions)] = _block(
                    base_values, positions, extras, extra_names, int(h), j)
            out[:, :, ti] = models[target].predict(X).reshape(len(horizons), len(positions)).T
    elif kind == "grouped":
        for ti, target in C.progress(indices, total=len(indices),
                                     desc="Grouped-H 预测目标", unit="目标"):
            for gname, lo_h, hi_h in C.progress(_groups_for(horizons),
                                               desc=f"{target} 预测分组", unit="模型"):
                subset = [j for j, h in enumerate(horizons) if lo_h <= h <= hi_h]
                X = np.empty((len(positions) * len(subset), len(names)), dtype=np.float32)
                for k, j in enumerate(subset):
                    lo = k * len(positions)
                    X[lo:lo + len(positions)] = _block(
                        base_values, positions, extras, extra_names,
                        int(horizons[j]), j)
                pred = models[(target, gname)].predict(X).reshape(len(subset), len(positions)).T
                for k, j in enumerate(subset):
                    out[:, j, ti] = pred[:, k]
    else:
        raise ValueError(f"未知模型类型 {kind}")
    return out


@dataclass
class ShortBundle:
    kind: str
    features: str
    horizons: tuple[int, ...]
    blend: dict
    models: dict
    final_models: dict
    feature_names: list[str]
    nn_bundle: object | None = None
    final_nn_bundle: object | None = None
    nn_weights: dict = field(default_factory=dict)
    lgb_kind: str | None = None
    lgb_features: str | None = None
    selection: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    train_rows: int = 0
    train_seconds: float = 0.0
    train_end: str | None = None
    final_train_end: str | None = None
    final_availability: dict | None = None
    final_generator1: dict | None = None
    process_recovery: dict = field(default_factory=lambda: dict(C.SHORT_PROCESS_RECOVERY))


def _fold_positions(prepared: PreparedData, fold: dict,
                    hmax: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    start = pd.Timestamp(fold["start"])
    end = start + pd.Timedelta(days=int(fold["days"])) - pd.Timedelta(minutes=C.STEP_MINUTES)
    origins = pd.date_range(start, end, freq=f"{C.STEP_MINUTES}min")
    eval_positions = prepared.index.get_indexer(origins)
    if (eval_positions < 0).any():
        missing = int((eval_positions < 0).sum())
        raise ValueError(f"折 {fold['name']} 有 {missing} 个起报点不在数据网格内")
    train_end = pd.Timestamp(fold["train_end"])
    ref_pos = int(prepared.index.get_indexer([train_end])[0])
    hmax = int(hmax or max(C.HORIZONS_L))
    limit = ref_pos - hmax
    positions = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY), max(limit + 1, 0))
    if len(positions) < 1000:
        raise ValueError(f"折 {fold['name']} 训练样本不足: {len(positions)}")
    return positions, eval_positions, ref_pos


def train_short(prepared: PreparedData, folds: list[dict], quick: bool = False) -> ShortBundle:
    prepared.require_training_data()
    tuning = [f for f in folds if f["role"] == "tuning"]
    if quick:
        tuning = tuning[:1]
    stride = C.TRAIN_STRIDE_SHORT if not quick else C.QUICK_TRAIN_STRIDE
    num_round = C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_SHORT
    started = time.time()
    fold_cache = []
    for fold in C.progress(tuning, desc="短期验证数据准备", unit="折"):
        masked = prepared.prediction_copy(
            fold["start"],
            pd.Timestamp(fold["start"]) + pd.Timedelta(days=int(fold["days"])))
        positions, eval_positions, ref_pos = _fold_positions(masked, fold, max(C.HORIZONS_S))
        positions = positions[::stride]
        base_all = build_base_features(masked, include_targets=True)
        baselines = baseline_cubes(masked, eval_positions, C.HORIZONS_S)
        fold_cache.append({"fold": fold, "prepared": masked, "base": base_all,
                           "positions": positions, "eval": eval_positions,
                           "ref_pos": ref_pos, "baselines": baselines})

    candidates: list[dict] = []
    rows_trained = 0
    # 短期 NN：与 LGBM 同折同窗口、只用该折 train_end 之前的数据独立训练，
    # 不与长期网络共享权重或选参。
    nn_started = time.time()
    nn_preds, nn_truths, nn_bundles = [], [], []
    for item in C.progress(fold_cache, desc="短期 NN 验证", unit="折"):
        fold = item["fold"]
        solo = [{**fold, "role": "tuning"},
                {**fold, "role": "confirm", "name": fold["name"] + "_val"}]
        nb = main_nn.train_nn(prepared, solo,
                              kind=str(C.NN_CONFIG.get("kind_short", "tcn_causal")),
                              horizons=C.HORIZONS_S, quick=quick,
                              n_seeds=int(C.NN_CONFIG.get("n_seeds", 1)))
        pred = main_nn.predict_nn(nb, item["prepared"], item["prepared"].index[item["eval"]])
        pred = enforce_physics(pred, fallback=item["baselines"]["profile"])
        nn_preds.append(pred)
        nn_truths.append(truth_cube(item["prepared"], item["eval"], C.HORIZONS_S))
        nn_bundles.append(nb)
    nn_scores = [score_cube(p, t, C.HORIZONS_S) for p, t in zip(nn_preds, nn_truths)]
    candidates.append({"kind": "nn", "features": "nn",
                       "preds": nn_preds, "truths": nn_truths,
                       "scores": nn_scores,
                       "mean": float(np.mean([s["accuracy"] for s in nn_scores])),
                       "worst": float(min(s["accuracy"] for s in nn_scores)),
                       "seconds": time.time() - nn_started,
                       "nn_bundle": nn_bundles[0]})
    for cand in C.progress(SHORT_CANDIDATES, desc="短期树模型选模", unit="候选"):
        started_cand = time.time()
        preds, truths = [], []
        models_store = None
        names_store = None
        for item in C.progress(fold_cache, desc=f"短期 {cand['kind']} 验证", unit="折"):
            masked = item["prepared"]
            logging.getLogger("aic-half").info("短期候选 %s/%s，折=%s", cand["kind"],
                                              cand["features"], item["fold"]["name"])
            models, names = _train_models(
                masked, item["base"], item["positions"], C.HORIZONS_S,
                cand["kind"], cand["features"], num_round, item["ref_pos"])
            pred = _predict_models(masked, item["base"], item["eval"], C.HORIZONS_S,
                                   cand["kind"], cand["features"], models, names)
            pred = enforce_physics(pred, fallback=item["baselines"]["profile"])
            preds.append(pred)
            truths.append(truth_cube(masked, item["eval"], C.HORIZONS_S))
            models_store = models
            names_store = names
            rows_trained += int(len(item["positions"]) * len(C.HORIZONS_S))
        scores = [score_cube(pred, truth, C.HORIZONS_S) for pred, truth in zip(preds, truths)]
        mean_score = float(np.mean([s["accuracy"] for s in scores]))
        worst_score = float(min(s["accuracy"] for s in scores))
        candidates.append({**cand, "preds": preds, "truths": truths,
                           "scores": scores, "mean": mean_score,
                           "worst": worst_score, "seconds": time.time() - started_cand})
        del models_store, names_store

    profiles = [item["baselines"]["profile"] for item in fold_cache]
    # NN+LGBM 融合候选：两者各自过 profile 后的凸组合，常优于单一模型。
    if any(c["kind"] == "nn" for c in candidates) and any(
            c["kind"] in ("direct", "global", "grouped") for c in candidates):
        nn_c = next(c for c in candidates if c["kind"] == "nn")
        lgb_c = max((c for c in candidates if c["kind"] in ("direct", "global", "grouped")),
                    key=lambda c: c["mean"])
        nn_weights = select_nn_tree_weights(nn_c["preds"], lgb_c["preds"], nn_c["truths"], C.HORIZONS_S, C.SHORT_GROUPS)
        ens_preds = [blend_cube(a, b, nn_weights, C.HORIZONS_S, C.SHORT_GROUPS)
                     for a, b in zip(nn_c["preds"], lgb_c["preds"])]
        ens_truths = nn_c["truths"]
        ens_scores = [score_cube(p, t, C.HORIZONS_S) for p, t in zip(ens_preds, ens_truths)]
        candidates.append({
            "kind": "nn_lgb_ens", "features": "hybrid",
            "nn_weights": nn_weights,
            "preds": ens_preds, "truths": ens_truths, "scores": ens_scores,
            "mean": float(np.mean([s["accuracy"] for s in ens_scores])),
            "worst": float(min(s["accuracy"] for s in ens_scores)),
            "seconds": 0.0, "nn_bundle": nn_c.get("nn_bundle"),
            "lgb_kind": lgb_c["kind"], "lgb_features": lgb_c["features"],
        })
    evaluate_blended_candidates(candidates, profiles, C.HORIZONS_S, C.SHORT_GROUPS)
    chosen = select_candidate(candidates, "direct")
    blend = chosen["blend"]
    blended_tuning = [blend_cube(p, prof, blend, C.HORIZONS_S, C.SHORT_GROUPS)
                      for p, prof in zip(chosen["preds"], profiles)]
    tuning_scores = [score_cube(p, t, C.HORIZONS_S)
                     for p, t in zip(blended_tuning, chosen["truths"])]

    confirm_fold = next(f for f in folds if f["role"] == "confirm")
    confirm_masked = prepared.prediction_copy(
        confirm_fold["start"],
        pd.Timestamp(confirm_fold["start"]) + pd.Timedelta(days=int(confirm_fold["days"])))
    c_positions, c_eval, c_ref = _fold_positions(
        confirm_masked, confirm_fold, max(C.HORIZONS_S))
    is_nn = chosen["kind"] in ("nn", "nn_lgb_ens")
    if is_nn:
        solo = [{**confirm_fold, "role": "tuning"},
                {**confirm_fold, "role": "confirm", "name": "val"}]
        nn_c = main_nn.train_nn(prepared, solo,
                                kind=str(C.NN_CONFIG.get("kind_short", "tcn_causal")),
                                horizons=C.HORIZONS_S, quick=quick,
                                n_seeds=int(C.NN_CONFIG.get("n_seeds", 1)))
        confirm_pred = main_nn.predict_nn(nn_c, confirm_masked,
                                          confirm_masked.index[c_eval])
        confirm_models, confirm_names = {}, []
        if chosen["kind"] == "nn_lgb_ens":
            c_positions_x, c_eval_x, c_ref_x = _fold_positions(
                confirm_masked, confirm_fold, max(C.HORIZONS_S))
            c_positions_x = c_positions_x[::stride]
            c_base_x = build_base_features(confirm_masked, include_targets=True)
            ens_models, ens_names = _train_models(
                confirm_masked, c_base_x, c_positions_x, C.HORIZONS_S,
                chosen.get("lgb_kind", "direct"), chosen.get("lgb_features", "gas"),
                num_round, c_ref_x)
            lgb_pred = _predict_models(
                confirm_masked, c_base_x, c_eval_x, C.HORIZONS_S,
                chosen.get("lgb_kind", "direct"), chosen.get("lgb_features", "gas"),
                ens_models, ens_names)
            lgb_pred = enforce_physics(lgb_pred)
            confirm_pred = blend_cube(confirm_pred, lgb_pred, chosen["nn_weights"], C.HORIZONS_S, C.SHORT_GROUPS)
            confirm_models, confirm_names = ens_models, ens_names
    else:
        c_positions = c_positions[::stride]
        c_base = build_base_features(confirm_masked, include_targets=True)
        confirm_models, confirm_names = _train_models(
            confirm_masked, c_base, c_positions, C.HORIZONS_S,
            chosen["kind"], chosen["features"], num_round, c_ref)
        confirm_pred = _predict_models(
            confirm_masked, c_base, c_eval, C.HORIZONS_S, chosen["kind"],
            chosen["features"], confirm_models, confirm_names)
    confirm_baseline = baseline_cubes(confirm_masked, c_eval, C.HORIZONS_S)
    confirm_pred = enforce_physics(confirm_pred, fallback=confirm_baseline["profile"])
    confirm_pred = blend_cube(confirm_pred, confirm_baseline["profile"], blend,
                              C.HORIZONS_S, C.SHORT_GROUPS)
    confirm_truth = truth_cube(confirm_masked, c_eval, C.HORIZONS_S)
    confirm_score = score_cube(confirm_pred, confirm_truth, C.HORIZONS_S)

    # 部署模型：用截至数据末尾的全部合法标签训练，供正式推理加载。
    if is_nn:
        legal_end = min(pd.Timestamp(C.TRAIN_END), prepared.index[-1])
        valid = prepared.labels.loc[:legal_end, list(C.TARGETS)].dropna()
        if valid.empty:
            raise ValueError("最终 NN 缺少合法训练标签")
        all_folds = [{"name": "final", "role": "tuning",
                      "train_end": str(valid.index[-1])}]
        nn_final = main_nn.train_nn(prepared, all_folds,
                                    kind=str(C.NN_CONFIG.get("kind_short", "tcn_causal")),
                                    horizons=C.HORIZONS_S, quick=quick,
                                    n_seeds=int(C.NN_CONFIG.get("n_seeds", 1)))
        final_models, final_names = {}, []
        if chosen["kind"] == "nn_lgb_ens":
            fp = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY),
                           len(prepared.index) - max(C.HORIZONS_S))[::stride]
            fb = build_base_features(prepared, include_targets=True)
            final_models, final_names = _train_models(
                prepared, fb, fp, C.HORIZONS_S,
                chosen.get("lgb_kind", "direct"), chosen.get("lgb_features", "gas"),
                num_round, len(prepared.index) - 1)
    else:
        final_positions = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY),
                                    len(prepared.index) - max(C.HORIZONS_S))
        final_positions = final_positions[::stride]
        final_base = build_base_features(prepared, include_targets=True)
        final_ref = len(prepared.index) - 1
        final_models, final_names = _train_models(
            prepared, final_base, final_positions, C.HORIZONS_S,
            chosen["kind"], chosen["features"], num_round, final_ref)
        if final_names != confirm_names:
            raise ValueError("最终模型特征与确认模型不一致")
        nn_final = None

    return ShortBundle(
        kind=chosen["kind"], features=chosen["features"], horizons=C.HORIZONS_S,
        blend=blend, models=confirm_models, final_models=final_models,
        feature_names=confirm_names,
        nn_bundle=nn_c if is_nn else None, final_nn_bundle=nn_final,
        nn_weights=chosen.get("nn_weights", {}),
        lgb_kind=chosen.get("lgb_kind"), lgb_features=chosen.get("lgb_features"),
        selection={"chosen": {"kind": chosen["kind"], "features": chosen["features"],
                              "effective": effective_model_name(
                                  chosen["kind"], chosen["features"], blend)},
                   "candidates": [{"kind": c["kind"], "features": c["features"],
                                   "mean_score": c["mean"], "worst_score": c["worst"],
                                   "scores": c["scores"], "seconds": c["seconds"]}
                                  for c in candidates],
                   "tuning_scores": tuning_scores},
        scores={"tuning_mean": float(np.mean([s["accuracy"] for s in tuning_scores])),
                "confirm": confirm_score},
        train_rows=rows_trained, train_seconds=time.time() - started,
        train_end=confirm_fold["train_end"],
        final_train_end=str(prepared.labels[list(C.TARGETS)].dropna().index[-1]),
        final_availability=train_availability_fallback(prepared, C.HORIZONS_S,
            C.AVAILABILITY_FALLBACK.get("short"), quick))


def train_final_load(prepared: PreparedData, span: str, bundle_type, quick: bool = False):
    """只拟合已冻结部署配方；短长任务各自建模，不读取开发或确认折分数。"""
    import copy
    prepared.require_training_data()
    started = time.time()
    recipe = copy.deepcopy(C.FROZEN_LOAD_RECIPES[span])
    horizons = C.HORIZONS_S if span == "short" else C.HORIZONS_L
    ref = len(prepared.index) - 1
    end = str(prepared.index[ref])
    stride = C.QUICK_TRAIN_STRIDE if quick else (C.TRAIN_STRIDE_SHORT if span == "short" else C.TRAIN_STRIDE_LONG)
    positions = np.arange(C.PROFILE_DAYS * C.STEPS_PER_DAY, ref - max(horizons) + 1, stride)
    if not len(positions):
        raise ValueError("最终训练前缀不足以构造合法目标")
    rounds = C.LGBM_TREES_QUICK if quick else (C.LGBM_TREES_SHORT if span == "short" else C.LGBM_TREES_LONG)
    models, names = _train_models(prepared, build_base_features(prepared, include_targets=True),
        positions, horizons, recipe["lgb_kind"], recipe["lgb_features"],
        rounds, ref)
    neural = main_nn.train_nn(prepared, [{"name": "final", "role": "tuning", "train_end": end}],
        kind=C.NN_CONFIG["kind_" + span], horizons=horizons, quick=quick, n_seeds=int(C.NN_CONFIG["n_seeds"]))
    return bundle_type(**recipe, horizons=horizons, models={}, final_models=models, feature_names=names,
        final_nn_bundle=neural, final_availability=train_availability_fallback(
            prepared, horizons, C.AVAILABILITY_FALLBACK[span], quick),
        selection={"chosen": copy.deepcopy(recipe), "selection_scope": "frozen_recipe_no_search"},
        scores={"scope": "no_new_validation; historical_scores_not_recomputed"},
        train_rows=len(positions) * len(horizons), train_seconds=time.time() - started,
        train_end=end, final_train_end=end)


def train_availability_fallback(prepared: PreparedData, horizons: tuple[int, ...],
                                recipe: dict | None, quick: bool = False) -> dict | None:
    """只按已冻结配方重训最终缺测分支，不重新访问开发折或确认折。"""
    if not recipe:
        return None
    if (recipe.get("retained_recipe") or {}).get("retained_recipe"):
        raise ValueError("缺测分支只支持一层已保留模型融合")
    prepared.require_training_data()
    ref = len(prepared.index) - 1
    training = prepared
    if recipe["features"].removesuffix("_at_t") == "gas":
        for offset in range(7 * C.STEPS_PER_DAY, ref, 21 * C.STEPS_PER_DAY):
            training = training.masked_power_gas_copy(training.index[offset],
                training.index[min(offset + 14 * C.STEPS_PER_DAY - 1, ref)])
    stride = C.TRAIN_STRIDE_SHORT if max(horizons) <= 8 else C.TRAIN_STRIDE_LONG
    positions = np.arange(C.PROFILE_DAYS * C.STEPS_PER_DAY, ref - max(horizons) + 1,
                          C.QUICK_TRAIN_STRIDE if quick else stride)
    models, names = _train_models(training, build_base_features(training, include_targets=True),
        positions, horizons, recipe["kind"], recipe["features"],
        C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG, ref)
    state = {**recipe, "models": models, "feature_names": names,
             "train_end": str(training.index[ref]), "role": "final"}
    if recipe.get("retained_recipe"):
        state["retained_state"] = train_availability_fallback(prepared, horizons, recipe["retained_recipe"], quick)
    return state


def apply_availability_fallback(bundle, prepared, positions, prediction, use_final):
    state = getattr(bundle, "final_availability", None)
    if not use_final or not state:
        return prediction
    if pd.Timestamp(state["train_end"]) >= prepared.index[positions.min()]:
        raise ValueError("缺测分支的训练截止进入推理窗口")
    current = prepared.frame[["d_use_blast", "d_use_coke", "d_use_converter"]]
    missing = ~np.isfinite(current.to_numpy(dtype=float)[positions]).all(axis=1)
    if not missing.any():
        return prediction
    selected = positions[missing]
    profile = baseline_cubes(prepared, selected, bundle.horizons)["profile"]
    raw = _predict_models(prepared, build_base_features(prepared, include_targets=True),
        selected, bundle.horizons, state["kind"], state["features"], state["models"], state["feature_names"])
    raw = enforce_physics(raw, fallback=profile)
    result = prediction.copy()
    candidate = blend_cube(raw, profile, state["blend"], bundle.horizons,
                            _groups_for(bundle.horizons))
    weight = float(state.get("fallback_weight", 1.))
    if not 0. <= weight <= 1.:
        raise ValueError("缺测分支融合权重超出范围")
    reference = result[missing]
    target_weights = state.get("fallback_weights")
    if (weight < 1. or target_weights) and state.get("reference_recipe"):
        recipe = state["reference_recipe"]
        if (bundle.lgb_kind or bundle.kind) != recipe["kind"]:
            raise ValueError("缺测分支的保护参考与实际树组件不匹配")
        raw_reference = _predict_models(prepared, build_base_features(prepared, include_targets=True),
            selected, bundle.horizons, recipe["kind"], recipe["features"], bundle.final_models, bundle.feature_names)
        reference = blend_cube(enforce_physics(raw_reference, fallback=profile), profile,
            recipe["blend"], bundle.horizons, _groups_for(bundle.horizons))
    result[missing] = (blend_cube(candidate, reference, target_weights, bundle.horizons,
                                  _groups_for(bundle.horizons)) if target_weights else
                       weight * candidate + (1. - weight) * reference)
    if state.get("retained_state"):
        from dataclasses import replace
        retained = apply_availability_fallback(replace(bundle, final_availability=state["retained_state"]),
                                               prepared, positions, prediction, use_final)
        mix = float(state["retained_mix_weight"])
        if not 0. <= mix <= 1.:
            raise ValueError("新旧缺测分支融合权重超出范围")
        result[missing] = enforce_physics(mix * result[missing] + (1. - mix) * retained[missing])
    return result


def predict_short(bundle: ShortBundle, prepared: PreparedData,
                  origins: pd.DatetimeIndex, use_final: bool = False) -> np.ndarray:
    from data_cleaning.pipeline import short_process_view
    recovered, active_all = short_process_view(prepared, bundle.process_recovery)
    positions = prepared.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("起报点不在数据网格内")
    active = active_all[positions]
    if not active.any():
        return _predict_short_core(bundle, prepared, origins, use_final)
    if active.all():
        return _predict_short_core(bundle, recovered, origins, use_final)
    result = np.empty((len(origins), len(C.HORIZONS_S), len(C.TARGETS)))
    # 两个分支的起报点互斥，耗气缺测专家不会读到恢复后的历史窗口。
    result[active] = _predict_short_core(bundle, recovered, origins[active], use_final)
    result[~active] = _predict_short_core(bundle, prepared, origins[~active], use_final)
    return result


def _predict_short_core(bundle: ShortBundle, prepared: PreparedData,
                        origins: pd.DatetimeIndex, use_final: bool = False) -> np.ndarray:
    positions = prepared.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("起报点不在数据网格内")
    baselines = baseline_cubes(prepared, positions, C.HORIZONS_S)
    if bundle.kind in ("nn", "nn_lgb_ens"):
        nn = bundle.final_nn_bundle if use_final else bundle.nn_bundle
        if nn is None:
            raise ValueError("缺少所选角色的 NN 模型")
        pred = main_nn.predict_nn(nn, prepared, origins)
        if bundle.kind == "nn_lgb_ens":
            models = bundle.final_models if use_final else bundle.models
            if not models:
                raise ValueError("融合缺少同角色 LightGBM 分量")
            base_all = build_base_features(prepared, include_targets=True)
            lgb_pred = _predict_models(
                prepared, base_all, positions, C.HORIZONS_S,
                bundle.lgb_kind or "direct", bundle.lgb_features or "gas",
                models, bundle.feature_names)
            lgb_pred = enforce_physics(lgb_pred, fallback=baselines["profile"])
            pred = (blend_cube(pred, lgb_pred, bundle.nn_weights, C.HORIZONS_S, C.SHORT_GROUPS)
                    if bundle.nn_weights else (pred + lgb_pred) / 2.0)
    else:
        base_all = build_base_features(prepared, include_targets=True)
        models = bundle.final_models if use_final else bundle.models
        pred = _predict_models(prepared, base_all, positions, C.HORIZONS_S,
                               bundle.kind, bundle.features, models,
                               bundle.feature_names)
    pred = enforce_physics(pred, fallback=baselines["profile"])
    result = blend_cube(pred, baselines["profile"], bundle.blend,
                        C.HORIZONS_S, C.SHORT_GROUPS)
    result = apply_availability_fallback(bundle, prepared, positions, result, use_final)
    return apply_generator1_specialist(bundle, prepared, positions, result, use_final)


def generator1_project(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """专项只调整小机组组，不能通过通用投影联动修改受保护的总负荷。"""
    result = np.asarray(reference, dtype=float).copy()
    candidate = np.asarray(candidate, dtype=float)
    if candidate.ndim == 3:
        candidate = candidate[..., 0]
    if candidate.shape != result.shape[:2] or not np.isfinite(candidate).all():
        raise ValueError("generator_1候选形状错误或存在非有限值")
    total = result[..., 1]
    result[..., 0] = np.clip(candidate, np.maximum(0., total - 240.), np.minimum(200., total))
    return result


def generator1_base(prepared: PreparedData, include_power: bool = False) -> pd.DataFrame:
    """专项特征单独构造，既有模型的列集合及数值保持不变。"""
    from data_cleaning.pipeline import _last_valid
    features = "gas_at_t" if include_power else "gas_no_power_at_t"
    base = _with_observed_features(prepared, build_base_features(prepared, True), features)
    base = base[_feature_columns(base, features)].copy()
    frame = prepared.frame
    for col in ("d_use_blast", "d_use_coke", "d_use_converter", "d_gen_blast", "d_gen_coke"):
        _, age = _last_valid(frame[col])
        base[f"g1_{col}_log_age"] = np.log1p(age / C.STEPS_PER_DAY)
        for window in (4, 96):
            base[f"g1_{col}_coverage{window}"] = frame[col].notna().rolling(window, min_periods=1).mean()
    for target in C.TARGETS:
        level, age = _last_valid(frame[C.TARGET_COLUMN[target]])
        base[f"g1_{target}_last_level"] = level
        base[f"g1_{target}_log_age"] = np.log1p(age / C.STEPS_PER_DAY)
    return base


def generator1_gate_matrix(prepared, positions, horizons, before, after):
    base = generator1_base(prepared)
    # 小门控只读取可解释的工况与专家分歧，不提供绝对日期或折编号。
    cols = [c for c in base if c.startswith("g1_") or c in
            ("d_gen_blast", "d_gen_coke", "d_gen_converter_scaled", "d_surplus_blast", "d_holder_level")]
    parts = []
    for j, h in enumerate(horizons):
        parts.append(np.column_stack([base[cols].to_numpy(dtype=np.float32)[positions],
            before[:, j], after[:, j], after[:, j] - before[:, j], np.full(len(positions), h)]))
    return np.concatenate(parts).astype(np.float32), cols + ["before", "after", "difference", "h"]


def generator1_transition_bins(X, names):
    columns = [names.index(f"g1_d_use_{gas}_log_age") for gas in ("blast", "coke", "converter")]
    age = np.nanmax(X[:, columns], axis=1)
    day = np.searchsorted(np.log1p([1., 7.]), age, side="right")
    lead = (X[:, names.index("h")] > 4).astype(int)
    result = day * 2 + lead
    target_age = np.expm1(X[:, names.index("g1_generator_1_log_age")])
    result += 6 * (target_age - np.expm1(age) >= 1.).astype(int)
    return result


def generator1_transition_weights(calibration, prepared, positions, horizons, before, after):
    X, names = generator1_gate_matrix(prepared, positions, horizons, before, after)
    if names != calibration["feature_names"]:
        raise ValueError("缺测阶段校准特征与归档顺序不一致")
    if calibration["family"] != "duration_state":
        raise ValueError("不支持的缺测阶段校准")
    table = np.asarray(calibration["table"], dtype=float)
    shrink = float(calibration["shrink"])
    if (len(table) != 12 or not np.isfinite(table).all()
            or np.any((table < 0.) | (table > .5)) or not 0. <= shrink <= 1.):
        raise ValueError("缺测阶段权重不合法")
    raw = table[generator1_transition_bins(X, names)]
    return np.clip(.25 + shrink * (raw.reshape(len(horizons), -1).T - .25), 0., .5)


def apply_generator1_specialist(bundle, prepared, positions, prediction, use_final):
    state = getattr(bundle, "final_generator1", None) if use_final else None
    if not state:
        return prediction
    if pd.Timestamp(state["train_end"]) >= prepared.index[np.min(positions)]:
        raise ValueError("generator_1专项训练截止进入推理窗口")
    recipe, horizons = state["recipe"], bundle.horizons
    if recipe["kind"] == "refined_nn":
        import copy
        if recipe.get("calibration"):
            calibration = state.get("calibration")
            if (not calibration or calibration.get("train_end") != state["train_end"]
                    or any(calibration.get(k) != recipe["calibration"][k] for k in ("family", "shrink", "mode"))):
                raise ValueError("冻结配方缺少同截止点、同参数的阶段校准")
        if state["base_state"]["recipe"]["kind"] == "refined_nn":
            raise ValueError("精修分支只保留一层，禁止不断堆叠历史模型")
        base_bundle = copy.copy(bundle)
        base_bundle.final_generator1 = state["base_state"]
        before = apply_generator1_specialist(base_bundle, prepared, positions, prediction, use_final)
        update = recipe["refinement"]
        observed = np.isfinite(prepared.frame[["d_use_blast", "d_use_coke", "d_use_converter"]].to_numpy()[positions]).all(axis=1)
        active = ~observed if update["mode"] == "missing" else observed if update["mode"] == "normal" else np.ones(len(positions), dtype=bool)
        if not active.any():
            return before
        estimate = main_nn.predict_nn(state["nn"], prepared, prepared.index[positions])[..., 0]
        weight = (generator1_transition_weights(state["calibration"], prepared, positions, horizons,
                  before[..., 0], estimate) if state.get("calibration") else update["weight"])
        result = generator1_project(before[..., 0] + weight * active[:, None] * (estimate - before[..., 0]), before)
        result[~active] = before[~active]
        return result
    active = np.ones(len(positions), dtype=bool)
    if recipe.get("missing_only", True):
        observed = prepared.frame[["d_use_blast", "d_use_coke", "d_use_converter"]].to_numpy()[positions]
        active = ~np.isfinite(observed).all(axis=1)
        if not active.any():
            return prediction
    if recipe["kind"] == "nn":
        expert = main_nn.predict_nn(state["nn"], prepared, prepared.index[positions])[..., 0]
    else:
        base = build_base_features(prepared, True)
        members = [np.clip(_predict_models(prepared, base, positions, horizons,
            state["expert_kind"], state["expert_features"], seed["models"], seed["feature_names"],
            targets=("generator_1",))[..., 0], 0., 200.) for seed in state.get("experts", [state])]
        expert = np.mean(members, axis=0)
        profile = target_slot_profile(prepared, positions, horizons, "generator_1")
        for group, lo, hi in _groups_for(horizons):
            js = [j for j, h in enumerate(horizons) if lo <= h <= hi]
            weight = state["expert_blend"]["generator_1"][group]
            expert[:, js] = weight * np.clip(expert[:, js], 0., 200.) + (1. - weight) * profile[:, js]
    weight = recipe.get("weight", 1.)
    if recipe.get("disagreement_margin_mw"):
        margin = recipe["disagreement_margin_mw"]
        weight = weight * np.clip((np.abs(expert - prediction[..., 0]) - margin) / margin, 0., 1.)
    if recipe.get("missing_only", True):
        weight = weight * active[:, None]
    result = generator1_project(prediction[..., 0] + weight * (expert - prediction[..., 0]), prediction)
    result[~active] = prediction[~active]
    return result


def save_generator1(state, directory):
    """仅保存当前专项实际使用的模型，不把内层折外模型或数组混入交付物。"""
    import json
    directory.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in state.items() if k not in ("nn", "models", "experts", "base_state")}
    if state.get("base_state"):
        save_generator1(state["base_state"], directory / "base")
        meta["base_path"] = "base"
    if state.get("experts"):
        meta["expert_paths"] = []
        for i, expert in enumerate(state["experts"]):
            name = f"seed_{i}"
            save_generator1(expert, directory / name)
            meta["expert_paths"].append(name)
    if "nn" in state:
        meta["nn_path"] = main_nn.save_nn(state["nn"], directory).split("\\")[-1].split("/")[-1]
    if "models" in state:
        meta["files"] = []
        for i, (key, model) in enumerate(state["models"].items()):
            filename = f"expert_{i}.txt"
            (directory / filename).write_text(model.model_to_string(), encoding="utf-8")
            meta["files"].append({"key": list(key) if isinstance(key, tuple) else key, "filename": filename})
    (directory / "state.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_generator1(directory):
    import json
    meta = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if meta.get("base_path"):
        meta["base_state"] = load_generator1(directory / meta["base_path"])
    if meta.get("expert_paths"):
        meta["experts"] = [load_generator1(directory / name) for name in meta["expert_paths"]]
    if "nn_path" in meta:
        meta["nn"] = main_nn.load_nn(directory / meta["nn_path"])
    if "files" in meta:
        meta["models"] = {(tuple(row["key"]) if isinstance(row["key"], list) else row["key"]):
            lgb.Booster(model_str=(directory / row["filename"]).read_text(encoding="utf-8")) for row in meta["files"]}
    return meta
