"""长期 24 小时预测：Global-H / Grouped-H LightGBM、日内画像与融合。

长期模型与短期模型完全独立训练；约束只保证物理一致性，不强制短期前 8 步
覆盖长期结果——官方模板是否要求一致尚待确认。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as C
from data_cleaning import (PreparedData, build_base_features, target_slot_profile)
import main_nn
from main_s import (_fold_positions, _predict_models, _train_models,
                    baseline_cubes, blend_cube, effective_model_name,
                    enforce_physics, score_cube, select_blend, truth_cube)


LONG_CANDIDATES = (
    {"kind": "global", "features": "gas"},
    {"kind": "grouped", "features": "gas"},
)


@dataclass
class LongBundle:
    kind: str
    features: str
    horizons: tuple[int, ...]
    blend: dict
    models: dict
    final_models: dict
    feature_names: list[str]
    # NN 支路：浅层 DLinear 为长周期主力候选；kind 含 'dlinear' 时启用。
    nn_bundle: object | None = None
    lgb_kind: str | None = None
    lgb_features: str | None = None
    selection: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    train_rows: int = 0
    train_seconds: float = 0.0
    train_end: str | None = None


def train_long(prepared: PreparedData, folds: list[dict], quick: bool = False) -> LongBundle:
    tuning = [f for f in folds if f["role"] == "tuning"]
    if quick:
        tuning = tuning[:1]
        stride = 4
        num_round = C.LGBM_TREES_QUICK
    else:
        stride = C.TRAIN_STRIDE_LONG
        num_round = C.LGBM_TREES_LONG
    started = time.time()
    fold_cache = []
    for fold in tuning:
        masked = prepared.masked_copy(
            fold["start"],
            pd.Timestamp(fold["start"]) + pd.Timedelta(days=int(fold["days"])))
        positions, eval_positions, ref_pos = _fold_positions(masked, fold, max(C.HORIZONS_L))
        positions = positions[::stride]
        base_all = build_base_features(masked, include_targets=True)
        baselines = baseline_cubes(masked, eval_positions, C.HORIZONS_L)
        fold_cache.append({"fold": fold, "prepared": masked, "base": base_all,
                           "positions": positions, "eval": eval_positions,
                           "ref_pos": ref_pos, "baselines": baselines})

    candidates = []
    rows_trained = 0
    # 浅层 DLinear：数据量有限时比 N-HiTS 稳，作为长周期主力候选之一。
    nn_started = time.time()
    nn_preds, nn_truths = [], []
    for item in fold_cache:
        fold = item["fold"]
        # 只用该折 train_end 之前的数据，避免串折泄漏
        solo = [{**fold, "role": "tuning"},
                {**fold, "role": "confirm", "name": fold["name"] + "_val"}]
        # DLinear 单折仅 ~15s，候选评估不走 quick，避免欠拟合误判。
        nb = main_nn.train_nn(prepared, solo, kind="dlinear",
                              horizons=C.HORIZONS_L, quick=False)
        pred = main_nn.predict_nn(nb, item["prepared"], item["prepared"].index[item["eval"]])
        pred = enforce_physics(pred, fallback=item["baselines"]["profile_fuel"])
        nn_preds.append(pred)
        nn_truths.append(truth_cube(item["prepared"], item["eval"], C.HORIZONS_L))
    nn_scores = [score_cube(p, t, C.HORIZONS_L) for p, t in zip(nn_preds, nn_truths)]
    candidates.append({"kind": "dlinear", "features": "nn",
                       "preds": nn_preds, "truths": nn_truths,
                       "scores": nn_scores,
                       "mean": float(np.mean([s["accuracy"] for s in nn_scores])),
                       "worst": float(min(s["accuracy"] for s in nn_scores)),
                       "seconds": time.time() - nn_started,
                       "nn_bundle": nb})

    for cand in LONG_CANDIDATES:
        started_cand = time.time()
        preds, truths = [], []
        for item in fold_cache:
            masked = item["prepared"]
            models, names = _train_models(
                masked, item["base"], item["positions"], C.HORIZONS_L,
                cand["kind"], cand["features"], num_round, item["ref_pos"])
            pred = _predict_models(masked, item["base"], item["eval"], C.HORIZONS_L,
                                   cand["kind"], cand["features"], models, names)
            pred = enforce_physics(pred, fallback=item["baselines"]["profile_fuel"])
            preds.append(pred)
            truths.append(truth_cube(masked, item["eval"], C.HORIZONS_L))
            rows_trained += int(len(item["positions"]) * len(C.HORIZONS_L))
        scores = [score_cube(pred, truth, C.HORIZONS_L) for pred, truth in zip(preds, truths)]
        candidates.append({**cand, "preds": preds, "truths": truths,
                           "scores": scores,
                           "mean": float(np.mean([s["accuracy"] for s in scores])),
                           "worst": float(min(s["accuracy"] for s in scores)),
                           "seconds": time.time() - started_cand})

    # NN+LGBM 融合候选：两者各自过 profile 后的凸组合，常优于单一模型。
    if any(c["kind"] == "dlinear" for c in candidates) and any(
            c["kind"] in ("global", "grouped") for c in candidates):
        nn_c = next(c for c in candidates if c["kind"] == "dlinear")
        # 取调参分最高的 LGBM
        lgb_c = max((c for c in candidates if c["kind"] in ("global", "grouped")),
                    key=lambda c: c["mean"])
        ens_preds = [(a + b) / 2.0 for a, b in zip(nn_c["preds"], lgb_c["preds"])]
        ens_truths = nn_c["truths"]
        ens_scores = [score_cube(p, t, C.HORIZONS_L) for p, t in zip(ens_preds, ens_truths)]
        candidates.append({
            "kind": "nn_lgb_ens", "features": "hybrid",
            "preds": ens_preds, "truths": ens_truths, "scores": ens_scores,
            "mean": float(np.mean([s["accuracy"] for s in ens_scores])),
            "worst": float(min(s["accuracy"] for s in ens_scores)),
            "seconds": 0.0, "nn_bundle": nn_c.get("nn_bundle"),
            "lgb_kind": lgb_c["kind"], "lgb_features": lgb_c["features"],
        })

    best_mean = max(c["mean"] for c in candidates)
    feasible = [c for c in candidates
                if c["worst"] >= best_mean - C.SELECTION["max_worst_fold_drop_pp"]]
    chosen = max(feasible or candidates, key=lambda c: (c["mean"], c["features"] == "gas"))
    profiles = [item["baselines"]["profile_fuel"] for item in fold_cache]
    blend = select_blend(chosen["preds"], profiles, chosen["truths"],
                         C.HORIZONS_L, C.LONG_GROUPS)
    blended_tuning = [blend_cube(p, prof, blend, C.HORIZONS_L, C.LONG_GROUPS)
                      for p, prof in zip(chosen["preds"], profiles)]
    tuning_scores = [score_cube(p, t, C.HORIZONS_L)
                     for p, t in zip(blended_tuning, chosen["truths"])]

    confirm_fold = next(f for f in folds if f["role"] == "confirm")
    confirm_masked = prepared.masked_copy(
        confirm_fold["start"],
        pd.Timestamp(confirm_fold["start"]) + pd.Timedelta(days=int(confirm_fold["days"])))
    c_positions, c_eval, c_ref = _fold_positions(
        confirm_masked, confirm_fold, max(C.HORIZONS_L))

    is_nn = chosen["kind"] in ("dlinear", "nn_lgb_ens")
    if is_nn:
        # NN：候选阶段已给出 preds/nn_bundle，这里只补确认折与最终包
        confirm_models, confirm_names = {}, []
        # fold_cache 只含调参折；确认折必须按协议单独训练后再评。
        solo = [{**confirm_fold, "role": "tuning"},
                {**confirm_fold, "role": "confirm", "name": "val"}]
        nn_c = main_nn.train_nn(prepared, solo, kind="dlinear",
                                horizons=C.HORIZONS_L, quick=False)
        nn_pred = main_nn.predict_nn(nn_c, confirm_masked,
                                     confirm_masked.index[c_eval])
        confirm_pred = nn_pred
        if chosen["kind"] == "nn_lgb_ens":
            # 训练同折 LGBM 分量再平均
            c_positions_x, c_eval_x, c_ref_x = _fold_positions(
                confirm_masked, confirm_fold, max(C.HORIZONS_L))
            c_base_x = build_base_features(confirm_masked, include_targets=True)
            ens_models, ens_names = _train_models(
                confirm_masked, c_base_x, c_positions_x, C.HORIZONS_L,
                chosen.get("lgb_kind", "global"), chosen.get("lgb_features", "gas"),
                300, c_ref_x)
            lgb_pred = _predict_models(
                confirm_masked, c_base_x, c_eval_x, C.HORIZONS_L,
                chosen.get("lgb_kind", "global"), chosen.get("lgb_features", "gas"),
                ens_models, ens_names)
            confirm_pred = (nn_pred + lgb_pred) / 2.0
        confirm_baseline = baseline_cubes(confirm_masked, c_eval, C.HORIZONS_L)
        confirm_pred = enforce_physics(confirm_pred, fallback=confirm_baseline["profile_fuel"])
        confirm_pred = blend_cube(confirm_pred, confirm_baseline["profile_fuel"], blend,
                                  C.HORIZONS_L, C.LONG_GROUPS)
        confirm_truth = truth_cube(confirm_masked, c_eval, C.HORIZONS_L)
        confirm_score = score_cube(confirm_pred, confirm_truth, C.HORIZONS_L)
        final_models, final_names = {}, []
        ens_final_models, ens_final_names = {}, []
        # 部署包：用全部有标签数据再训一次。验证窗必须有标签，
        # 否则 build_nn_dataset 会把样本全滤掉——取 semi_train 末尾 2 天。
        labels = prepared.labels[list(C.TARGETS)].to_numpy(dtype=float)
        valid_pos = np.where(np.isfinite(labels).all(axis=1))[0]
        val_end_pos = int(valid_pos[-1]) - max(C.HORIZONS_L) - 1
        val_start_pos = max(0, val_end_pos - 2 * C.STEPS_PER_DAY)
        val_start = prepared.index[val_start_pos]
        val_end = prepared.index[val_end_pos]
        all_folds = [
            {"name": "final", "role": "tuning", "start": str(prepared.index[0]),
             "days": 1, "train_end": str(val_start)},
            {"name": "final_val", "role": "confirm", "start": str(val_start),
             "days": 2, "train_end": str(val_end)},
        ]
        nn_final = main_nn.train_nn(prepared, all_folds, kind="dlinear",
                                    horizons=C.HORIZONS_L, quick=False)
        if chosen["kind"] == "nn_lgb_ens":
            # 融合还需要一个全量训练的 LGBM 分量
            fp = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY),
                           len(prepared.index) - max(C.HORIZONS_L))
            fp = fp[::stride]
            fb = build_base_features(prepared, include_targets=True)
            ens_final_models, ens_final_names = _train_models(
                prepared, fb, fp, C.HORIZONS_L,
                chosen.get("lgb_kind", "global"), chosen.get("lgb_features", "gas"),
                num_round, len(prepared.index) - max(C.HORIZONS_L))
    else:
        c_positions = c_positions[::stride]
        c_base = build_base_features(confirm_masked, include_targets=True)
        confirm_models, confirm_names = _train_models(
            confirm_masked, c_base, c_positions, C.HORIZONS_L,
            chosen["kind"], chosen["features"], num_round, c_ref)
        confirm_pred = _predict_models(
            confirm_masked, c_base, c_eval, C.HORIZONS_L, chosen["kind"],
            chosen["features"], confirm_models, confirm_names)
        confirm_baseline = baseline_cubes(confirm_masked, c_eval, C.HORIZONS_L)
        confirm_pred = enforce_physics(confirm_pred, fallback=confirm_baseline["profile_fuel"])
        confirm_pred = blend_cube(confirm_pred, confirm_baseline["profile_fuel"], blend,
                                  C.HORIZONS_L, C.LONG_GROUPS)
        confirm_truth = truth_cube(confirm_masked, c_eval, C.HORIZONS_L)
        confirm_score = score_cube(confirm_pred, confirm_truth, C.HORIZONS_L)

        final_positions = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY),
                                    len(prepared.index) - max(C.HORIZONS_L))
        final_positions = final_positions[::stride]
        final_base = build_base_features(prepared, include_targets=True)
        final_ref = len(prepared.index) - max(C.HORIZONS_L)
        final_models, final_names = _train_models(
            prepared, final_base, final_positions, C.HORIZONS_L,
            chosen["kind"], chosen["features"], num_round, final_ref)
        if final_names != confirm_names:
            raise ValueError("最终长期模型特征与确认模型不一致")
        nn_final = None

    return LongBundle(
        kind=chosen["kind"], features=chosen["features"], horizons=C.HORIZONS_L,
        blend=blend, models=confirm_models,
        final_models=(ens_final_models if chosen["kind"] == "nn_lgb_ens"
                      and ens_final_models else final_models),
        feature_names=(ens_final_names if chosen["kind"] == "nn_lgb_ens"
                       and ens_final_names else confirm_names),
        nn_bundle=nn_final,
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
        train_end=confirm_fold["train_end"])


def predict_long(bundle: LongBundle, prepared: PreparedData,
                 origins: pd.DatetimeIndex, use_final: bool = False) -> np.ndarray:
    positions = prepared.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("起报点不在数据网格内")
    baselines = baseline_cubes(prepared, positions, C.HORIZONS_L)
    if bundle.kind in ("dlinear", "nn_lgb_ens") and bundle.nn_bundle is not None:
        pred = main_nn.predict_nn(bundle.nn_bundle, prepared, origins)
        if bundle.kind == "nn_lgb_ens" and bundle.models:
            # LGBM 分量：final_models 存的是同折 LGBM，没有则退回纯 NN
            try:
                base_all = build_base_features(prepared, include_targets=True)
                lgb_pred = _predict_models(
                    prepared, base_all, positions, C.HORIZONS_L,
                    getattr(bundle, "lgb_kind", "global"),
                    getattr(bundle, "lgb_features", "gas"),
                    bundle.final_models or bundle.models,
                    bundle.feature_names)
                pred = (pred + lgb_pred) / 2.0
            except Exception:
                pass
    else:
        base_all = build_base_features(prepared, include_targets=True)
        models = bundle.final_models if use_final else bundle.models
        pred = _predict_models(prepared, base_all, positions, C.HORIZONS_L,
                               bundle.kind, bundle.features, models,
                               bundle.feature_names)
    pred = enforce_physics(pred, fallback=baselines["profile_fuel"])
    return blend_cube(pred, baselines["profile_fuel"], bundle.blend,
                      C.HORIZONS_L, C.LONG_GROUPS)
