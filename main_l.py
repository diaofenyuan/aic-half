"""长期 24 小时预测：Global-H / Grouped-H LightGBM、日内画像与融合。

长期模型与短期模型完全独立训练；约束只保证物理一致性，不强制短期前 8 步
覆盖长期结果——官方模板是否要求一致尚待确认。
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

import config as C
from data_cleaning import PreparedData, build_base_features
import main_nn
from main_s import (_fold_positions, _predict_models, _train_models,
                    baseline_cubes, blend_cube, effective_model_name,
                    enforce_physics, score_cube, select_candidate, truth_cube,
                    evaluate_blended_candidates, select_nn_tree_weights, apply_availability_fallback,
                    train_availability_fallback)


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
    # 长期网络按已晋级的因果 TCN 配方独立训练。
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
    final_process_expert: dict | None = None


def train_long(prepared: PreparedData, folds: list[dict], quick: bool = False) -> LongBundle:
    prepared.require_training_data()
    tuning = [f for f in folds if f["role"] == "tuning"]
    if quick:
        tuning = tuning[:1]
        stride = C.QUICK_TRAIN_STRIDE
        num_round = C.LGBM_TREES_QUICK
    else:
        stride = C.TRAIN_STRIDE_LONG
        num_round = C.LGBM_TREES_LONG
    started = time.time()
    fold_cache = []
    for fold in C.progress(tuning, desc="长期验证数据准备", unit="折"):
        masked = prepared.prediction_copy(
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
    for item in C.progress(fold_cache, desc="长期 NN 验证", unit="折"):
        fold = item["fold"]
        # 只用该折 train_end 之前的数据，避免串折泄漏
        solo = [{**fold, "role": "tuning"},
                {**fold, "role": "confirm", "name": fold["name"] + "_val"}]
        # 开发预算逐层传递；quick 的分数只用于链路和候选初筛。
        nb = main_nn.train_nn(prepared, solo,
                              kind=str(C.NN_CONFIG.get("kind_long", "tcn_causal")),
                              horizons=C.HORIZONS_L, quick=quick,
                              n_seeds=int(C.NN_CONFIG.get("n_seeds", 1)))
        pred = main_nn.predict_nn(nb, item["prepared"], item["prepared"].index[item["eval"]])
        pred = enforce_physics(pred, fallback=item["baselines"]["profile"])
        nn_preds.append(pred)
        nn_truths.append(truth_cube(item["prepared"], item["eval"], C.HORIZONS_L))
    nn_scores = [score_cube(p, t, C.HORIZONS_L) for p, t in zip(nn_preds, nn_truths)]
    candidates.append({"kind": "nn", "features": "nn",
                       "preds": nn_preds, "truths": nn_truths,
                       "scores": nn_scores,
                       "mean": float(np.mean([s["accuracy"] for s in nn_scores])),
                       "worst": float(min(s["accuracy"] for s in nn_scores)),
                       "seconds": time.time() - nn_started,
                       "nn_bundle": nb})

    for cand in C.progress(LONG_CANDIDATES, desc="长期树模型选模", unit="候选"):
        started_cand = time.time()
        preds, truths = [], []
        for item in C.progress(fold_cache, desc=f"长期 {cand['kind']} 验证", unit="折"):
            masked = item["prepared"]
            logging.getLogger("aic-half").info("长期候选 %s/%s，折=%s", cand["kind"],
                                              cand["features"], item["fold"]["name"])
            models, names = _train_models(
                masked, item["base"], item["positions"], C.HORIZONS_L,
                cand["kind"], cand["features"], num_round, item["ref_pos"])
            pred = _predict_models(masked, item["base"], item["eval"], C.HORIZONS_L,
                                   cand["kind"], cand["features"], models, names)
            pred = enforce_physics(pred, fallback=item["baselines"]["profile"])
            preds.append(pred)
            truths.append(truth_cube(masked, item["eval"], C.HORIZONS_L))
            rows_trained += int(len(item["positions"]) * len(C.HORIZONS_L))
        scores = [score_cube(pred, truth, C.HORIZONS_L) for pred, truth in zip(preds, truths)]
        candidates.append({**cand, "preds": preds, "truths": truths,
                           "scores": scores,
                           "mean": float(np.mean([s["accuracy"] for s in scores])),
                           "worst": float(min(s["accuracy"] for s in scores)),
                           "seconds": time.time() - started_cand})

    # 先按实际画像融合后的成绩挑选树组件，避免原始分高的树挤掉互补组件。
    profiles = [item["baselines"]["profile"] for item in fold_cache]
    evaluate_blended_candidates(candidates, profiles, C.HORIZONS_L, C.LONG_GROUPS)
    # NN+LGBM 融合候选保持原始预测输入，画像只在候选末端应用一次。
    if any(c["kind"] == "nn" for c in candidates) and any(
            c["kind"] in ("global", "grouped") for c in candidates):
        nn_c = next(c for c in candidates if c["kind"] == "nn")
        # 取调参分最高的 LGBM
        lgb_c = max((c for c in candidates if c["kind"] in ("global", "grouped")),
                    key=lambda c: c["mean"])
        nn_weights = select_nn_tree_weights(nn_c["preds"], lgb_c["preds"], nn_c["truths"], C.HORIZONS_L, C.LONG_GROUPS)
        ens_preds = [blend_cube(a, b, nn_weights, C.HORIZONS_L, C.LONG_GROUPS)
                     for a, b in zip(nn_c["preds"], lgb_c["preds"])]
        ens_truths = nn_c["truths"]
        ens_scores = [score_cube(p, t, C.HORIZONS_L) for p, t in zip(ens_preds, ens_truths)]
        candidates.append({
            "kind": "nn_lgb_ens", "features": "hybrid",
            "nn_weights": nn_weights,
            "preds": ens_preds, "truths": ens_truths, "scores": ens_scores,
            "mean": float(np.mean([s["accuracy"] for s in ens_scores])),
            "worst": float(min(s["accuracy"] for s in ens_scores)),
            "seconds": 0.0, "nn_bundle": nn_c.get("nn_bundle"),
            "lgb_kind": lgb_c["kind"], "lgb_features": lgb_c["features"],
        })

        evaluate_blended_candidates([candidates[-1]], profiles, C.HORIZONS_L, C.LONG_GROUPS)
    chosen = select_candidate(candidates, "global")
    blend = chosen["blend"]
    blended_tuning = [blend_cube(p, prof, blend, C.HORIZONS_L, C.LONG_GROUPS)
                      for p, prof in zip(chosen["preds"], profiles)]
    tuning_scores = [score_cube(p, t, C.HORIZONS_L)
                     for p, t in zip(blended_tuning, chosen["truths"])]

    confirm_fold = next(f for f in folds if f["role"] == "confirm")
    confirm_masked = prepared.prediction_copy(
        confirm_fold["start"],
        pd.Timestamp(confirm_fold["start"]) + pd.Timedelta(days=int(confirm_fold["days"])))
    c_positions, c_eval, c_ref = _fold_positions(
        confirm_masked, confirm_fold, max(C.HORIZONS_L))

    is_nn = chosen["kind"] in ("nn", "nn_lgb_ens")
    if is_nn:
        # NN：候选阶段已给出 preds/nn_bundle，这里只补确认折与最终包
        confirm_models, confirm_names = {}, []
        # fold_cache 只含调参折；确认折必须按协议单独训练后再评。
        solo = [{**confirm_fold, "role": "tuning"},
                {**confirm_fold, "role": "confirm", "name": "val"}]
        nn_c = main_nn.train_nn(prepared, solo,
                                kind=str(C.NN_CONFIG.get("kind_long", "tcn_causal")),
                                horizons=C.HORIZONS_L, quick=quick,
                                n_seeds=int(C.NN_CONFIG.get("n_seeds", 1)))
        nn_pred = main_nn.predict_nn(nn_c, confirm_masked,
                                     confirm_masked.index[c_eval])
        confirm_pred = nn_pred
        if chosen["kind"] == "nn_lgb_ens":
            # 训练同折 LGBM 分量再平均
            c_positions_x, c_eval_x, c_ref_x = _fold_positions(
                confirm_masked, confirm_fold, max(C.HORIZONS_L))
            c_positions_x = c_positions_x[::stride]
            c_base_x = build_base_features(confirm_masked, include_targets=True)
            ens_models, ens_names = _train_models(
                confirm_masked, c_base_x, c_positions_x, C.HORIZONS_L,
                chosen.get("lgb_kind", "global"), chosen.get("lgb_features", "gas"),
                num_round, c_ref_x)
            lgb_pred = _predict_models(
                confirm_masked, c_base_x, c_eval_x, C.HORIZONS_L,
                chosen.get("lgb_kind", "global"), chosen.get("lgb_features", "gas"),
                ens_models, ens_names)
            lgb_pred = enforce_physics(lgb_pred)
            confirm_pred = blend_cube(nn_pred, lgb_pred, chosen["nn_weights"], C.HORIZONS_L, C.LONG_GROUPS)
            confirm_models, confirm_names = ens_models, ens_names
        confirm_baseline = baseline_cubes(confirm_masked, c_eval, C.HORIZONS_L)
        confirm_pred = enforce_physics(confirm_pred, fallback=confirm_baseline["profile"])
        confirm_pred = blend_cube(confirm_pred, confirm_baseline["profile"], blend,
                                  C.HORIZONS_L, C.LONG_GROUPS)
        confirm_truth = truth_cube(confirm_masked, c_eval, C.HORIZONS_L)
        confirm_score = score_cube(confirm_pred, confirm_truth, C.HORIZONS_L)
        final_models, final_names = {}, []
        ens_final_models, ens_final_names = {}, []
        # 部署包只使用实际可见的合法标签，内部早停不触及未来推理窗口。
        legal_end = min(pd.Timestamp(C.TRAIN_END), prepared.index[-1])
        valid = prepared.labels.loc[:legal_end, list(C.TARGETS)].dropna()
        if valid.empty:
            raise ValueError("最终 NN 缺少合法训练标签")
        all_folds = [{"name": "final", "role": "tuning",
                      "train_end": str(valid.index[-1])}]
        nn_final = main_nn.train_nn(prepared, all_folds,
                                    kind=str(C.NN_CONFIG.get("kind_long", "tcn_causal")),
                                    horizons=C.HORIZONS_L, quick=quick,
                                    n_seeds=int(C.NN_CONFIG.get("n_seeds", 1)))
        if chosen["kind"] == "nn_lgb_ens":
            # 融合还需要一个全量训练的 LGBM 分量
            fp = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY),
                           len(prepared.index) - max(C.HORIZONS_L))
            fp = fp[::stride]
            fb = build_base_features(prepared, include_targets=True)
            ens_final_models, ens_final_names = _train_models(
                prepared, fb, fp, C.HORIZONS_L,
                chosen.get("lgb_kind", "global"), chosen.get("lgb_features", "gas"),
                num_round, len(prepared.index) - 1)
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
        confirm_pred = enforce_physics(confirm_pred, fallback=confirm_baseline["profile"])
        confirm_pred = blend_cube(confirm_pred, confirm_baseline["profile"], blend,
                                  C.HORIZONS_L, C.LONG_GROUPS)
        confirm_truth = truth_cube(confirm_masked, c_eval, C.HORIZONS_L)
        confirm_score = score_cube(confirm_pred, confirm_truth, C.HORIZONS_L)

        final_positions = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY),
                                    len(prepared.index) - max(C.HORIZONS_L))
        final_positions = final_positions[::stride]
        final_base = build_base_features(prepared, include_targets=True)
        final_ref = len(prepared.index) - 1
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
        final_availability=train_availability_fallback(prepared, C.HORIZONS_L,
            C.AVAILABILITY_FALLBACK.get("long"), quick))


def predict_long(bundle: LongBundle, prepared: PreparedData,
                 origins: pd.DatetimeIndex, use_final: bool = False) -> np.ndarray:
    positions = prepared.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("起报点不在数据网格内")
    baselines = baseline_cubes(prepared, positions, C.HORIZONS_L)
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
                prepared, base_all, positions, C.HORIZONS_L,
                bundle.lgb_kind or "global", bundle.lgb_features or "gas",
                models, bundle.feature_names)
            lgb_pred = enforce_physics(lgb_pred, fallback=baselines["profile"])
            pred = (blend_cube(pred, lgb_pred, bundle.nn_weights, C.HORIZONS_L, C.LONG_GROUPS)
                    if bundle.nn_weights else (pred + lgb_pred) / 2.0)
    else:
        base_all = build_base_features(prepared, include_targets=True)
        models = bundle.final_models if use_final else bundle.models
        pred = _predict_models(prepared, base_all, positions, C.HORIZONS_L,
                               bundle.kind, bundle.features, models,
                               bundle.feature_names)
    pred = enforce_physics(pred, fallback=baselines["profile"])
    result = blend_cube(pred, baselines["profile"], bundle.blend,
                        C.HORIZONS_L, C.LONG_GROUPS)
    from main_s import apply_generator1_specialist
    result = apply_availability_fallback(bundle, prepared, positions, result, use_final)
    result = apply_generator1_specialist(bundle, prepared, positions, result, use_final)
    return apply_process_expert(bundle, prepared, positions, result, use_final)


def train_process_expert(prepared: PreparedData, bundle: LongBundle, quick: bool = False) -> dict:
    """冻结配方只重训合法前缀，不重新选权重或查看确认窗口。"""
    import copy
    prepared.require_training_data()
    recipe = copy.deepcopy(C.LONG_PROCESS_EXPERT)
    end, ref = str(prepared.index[-1]), len(prepared.index) - 1
    existing = bundle.final_process_expert
    if existing and existing["recipe"] == recipe and existing["train_end"] == end:
        return {"reused": True, "train_end": end, "recipe": recipe}
    started = time.time()
    def fit_recipe(selected, previous):
        if previous and previous["recipe"] == selected and previous["train_end"] == end:
            return previous
        retained = None
        if selected.get("retained_recipe"):
            prior = previous.get("base_state") if previous else None
            if previous and previous["recipe"] == selected["retained_recipe"]:
                prior = previous
            retained = fit_recipe(selected["retained_recipe"], prior)
        view = process_training_view(prepared, selected)
        positions = np.arange(C.PROFILE_DAYS * C.STEPS_PER_DAY, ref - max(C.HORIZONS_L) + 1,
            C.QUICK_TRAIN_STRIDE if quick else selected.get("train_stride", C.TRAIN_STRIDE_LONG))
        experts, old_decay = [], C.DECAY_DAYS
        try:
            C.DECAY_DAYS = selected["decay_days"]
            base = build_base_features(view, True)
            for seed in range(selected["seeds"]):
                logging.getLogger("aic-half").info("长期过程专家最终训练 seed=%d，截止=%s，连续遮挡=%s天",
                    20261021 + seed, end, selected.get("history_mask_days", 0))
                models, names = _train_models(view, base, positions, C.HORIZONS_L,
                    selected["kind"], selected["features"], C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG,
                    ref, params_override={"seed": 20261021 + seed})
                experts.append({"models": models, "feature_names": names})
        finally:
            C.DECAY_DAYS = old_decay
        state = {"recipe": selected, "experts": experts, "train_end": end,
            "max_training_target": str(prepared.index[positions[-1] + max(C.HORIZONS_L)])}
        if retained:
            state["base_state"] = retained
        return state
    bundle.final_process_expert = fit_recipe(recipe, existing)
    bundle.scores["process_expert_development"] = recipe["development"]
    bundle.scores["base_scores_exclude_process_expert"] = True
    return {"reused": False, "train_end": end, "recipe": recipe, "seconds": time.time() - started}


def process_training_view(prepared: PreparedData, recipe: dict) -> PreparedData:
    """只遮挡训练输入的目标历史，保留原始监督标签，模拟持续缺测时的陈旧画像。"""
    missing = int(recipe.get("history_mask_days", 0))
    if not missing:
        return prepared
    visible = int(recipe["history_visible_days"])
    if missing <= 0 or visible <= 0:
        raise ValueError("连续缺测训练的可见天数与遮挡天数必须为正")
    frame, provenance = prepared.frame.copy(), prepared.provenance.copy()
    day = np.arange(len(prepared.index)) // C.STEPS_PER_DAY
    mask = day % (visible + missing) >= visible
    for column in C.TARGET_COLUMN.values():
        frame.loc[mask, column] = np.nan
        provenance.loc[mask, column] = 0
    return replace(prepared, frame=frame, provenance=provenance)


def apply_process_expert(bundle: LongBundle, prepared: PreparedData, positions: np.ndarray,
                         prediction: np.ndarray, use_final: bool) -> np.ndarray:
    state = bundle.final_process_expert if use_final else None
    if not state:
        return prediction
    if pd.Timestamp(state["train_end"]) >= prepared.index[positions.min()]:
        raise ValueError("长期过程专家的训练截止进入推理窗口")
    if state.get("base_state"):
        from copy import copy
        retained = copy(bundle)
        retained.final_process_expert = state["base_state"]
        prediction = apply_process_expert(retained, prepared, positions, prediction, use_final)
    current = prepared.frame[["d_use_blast", "d_use_coke", "d_use_converter"]].to_numpy(dtype=float)[positions]
    active = ~np.isfinite(current).all(axis=1)
    if not active.any():
        return prediction
    selected, recipe = positions[active], state["recipe"]
    base = build_base_features(prepared, True)
    estimates = [enforce_physics(_predict_models(prepared, base, selected, C.HORIZONS_L,
                  recipe["kind"], recipe["features"], expert["models"], expert["feature_names"]))
                 for expert in state["experts"]]
    expert = enforce_physics(np.mean(estimates, axis=0))
    weights = np.zeros((len(C.HORIZONS_L), 2))
    for target, name in enumerate(C.TARGETS):
        for group, low, high in C.LONG_GROUPS:
            weights[low - 1:high, target] = recipe["weights"][name][group]
    if not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
        raise ValueError("长期过程专家权重不合法")
    result = prediction.copy()
    result[active] = enforce_physics(prediction[active] + weights[None] * (expert - prediction[active]))
    return result
