"""独立负荷树模型：共享数值原语，短长期训练结果严格分离。"""
from dataclasses import dataclass, field
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import config as C
from data_cleaning.pipeline import build_features, horizon_block, process_series
from data_cleaning.quality import project_load, truth_cube, score_load


@dataclass
class LoadBundle:
    span: str
    horizons: list
    features: list
    models: list = field(default_factory=list)
    recipe: dict = field(default_factory=dict)
    train_end: str = ""
    diagnostics: list = field(default_factory=list)
    members: list = field(default_factory=list)
    experts: dict = field(default_factory=dict)


def fit_booster(X, y, names, times, *, loss="l1", rounds=320, seed=2026, leaves=31,
                weight_scale=None, original_y=None, decay_days=None, learning_rate=None,
                weight_power=1., weight_target=None):
    valid = np.isfinite(y)
    X, y, times = X[valid], y[valid], np.asarray(times)[valid]
    if len(y) < 100:
        raise ValueError(f"有效训练标签不足：{len(y)}")
    observed = y if original_y is None else np.asarray(original_y)[valid]
    weight = np.ones(len(y), dtype=float)
    if loss == "relative_l1":
        # 1MW以下分母截断仅用于训练稳定；评分仍使用非零真值的原始分母。
        denominator = observed if weight_target is None else np.asarray(weight_target)[valid]
        weight = 1./np.maximum(np.abs(denominator), 1.)**weight_power
        positive = denominator[denominator > 0.]
        weight[denominator == 0.] = 1./max(float(np.median(positive)) if len(positive) else 1., 1.)**weight_power
        if weight_scale is not None:
            weight *= np.asarray(weight_scale)[valid]
        if decay_days:
            # 近期权重仅改变已选负荷配方的训练损失，资源模型默认不启用。
            age=(pd.Timestamp(times.max())-pd.DatetimeIndex(times)).total_seconds()/86400.
            weight *= np.exp2(-np.asarray(age)/decay_days)
        weight /= weight.mean()
    split = pd.Timestamp(times.max()) - pd.Timedelta(days=7)
    valid_rows = times >= split.to_datetime64()
    train_rows = times < (split-pd.Timedelta(days=1)).to_datetime64()
    params = dict(C.TREE, objective="regression_l1" if loss != "l2" else "regression",
                  metric="l1", seed=seed, num_leaves=leaves)
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    best_round = rounds
    if train_rows.sum() > 200 and valid_rows.sum() > 100:
        dtrain = lgb.Dataset(X[train_rows], label=y[train_rows], weight=weight[train_rows], feature_name=names)
        dv = lgb.Dataset(X[valid_rows], label=y[valid_rows], weight=weight[valid_rows], reference=dtrain)
        early = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(35, verbose=False)])
        best_round = max(25, early.best_iteration)
        del dtrain, dv, early
    # 内层早停确定轮数后在合法前缀重拟合，不把外层验证用作早停标签。
    final = lgb.train(params, lgb.Dataset(X, label=y, weight=weight, feature_name=names),
                      num_boost_round=best_round)
    return final, {"rows": len(y), "rounds": best_round, "loss": loss,
                   "learning_rate": params["learning_rate"], "num_leaves": params["num_leaves"],
                   "weight_power":weight_power,"weight_target":"original" if weight_target is None else "score_aligned_auxiliary",
                   "zero_label_rows": int((observed == 0.).sum()),
                   "min_target_time": str(pd.Timestamp(times.min())),
                   "max_target_time": str(pd.Timestamp(times.max())), "inner_split": str(split)}


def load_recipe(recipe, span):
    result = dict(recipe)
    result.update(recipe.get("tasks", {}).get(span, {}))
    return result


def build_load_features(prepared, kind="base"):
    base = build_features(prepared)
    parts = set(kind.split("+"))
    if parts == {"base"}:
        return base
    f = prepared.frame
    raw = f[[x for x in f if x.startswith(("gas_", "gas_user_"))
             and not x.startswith("gas_holder_")]]
    extra = {}
    if "memory" in parts:
        for col in raw:
            v = raw[col]
            # 最近观测另设特征，原始缺测不回填；超过一天即停止沿用。
            extra[col+"_last96"] = v.ffill(limit=96)
            extra[col+"_valid96"] = v.notna().rolling(96, min_periods=1).mean()
    if "dynamics" in parts:
        series = process_series(prepared)
        # 差分保留两端缺测，分别提供聚合过程与单炉变化，不回填原始观测。
        for name in ("blast_13", "blast_45", "known_demand", "coke_available_proxy"):
            v = series[name]
            for lag in (1, 2, 4, 8, 16):
                extra[f"{name}_change{lag}"] = v-v.shift(lag)
            for lag in (2, 8):
                extra[f"{name}_lag{lag}"] = v.shift(lag)
        for col in (f"gas_blast_furnace_{i}" for i in range(1, 6)):
            for lag in (1, 4, 16):
                extra[f"{col}_change{lag}"] = raw[col]-raw[col].shift(lag)
    if "relative" in parts:
        series = process_series(prepared)
        for name in ("blast_13", "blast_45", "blast_observed_parts", "known_demand", "coke_available_proxy"):
            v = series[name]
            for window in (96, 672):
                level = v.rolling(window, min_periods=8).median()
                extra[f"{name}_relative{window}"] = v/level.where(level.abs() > 1.)
                extra[f"{name}_deviation{window}"] = v-level
        for unit in range(1, 6):
            gas, heater = raw[f"gas_blast_furnace_{unit}"], raw[f"gas_air_heater_{unit}"]
            # 成对收支只在同刻两项都有效时成立，缺测或无效分母保持缺失。
            net = gas-heater
            extra[f"furnace_{unit}_paired_net"] = net
            extra[f"furnace_{unit}_heater_ratio"] = gas/heater.where(heater > 1.)
            extra[f"furnace_{unit}_paired_med16"] = net.rolling(16, min_periods=1).median()
    if "mechanism" in parts:
        gas_cols=[f"gas_blast_furnace_{i}" for i in range(1,6)]
        heater_cols=[f"gas_air_heater_{i}" for i in range(1,6)]
        other_cols=[f"gas_user_blast_furnace_user{i}" for i in range(1,5)]+["gas_into_gas_mixed_blast_furnace"]
        other=raw.reindex(columns=other_cols).sum(axis=1,min_count=1)
        extra["mechanism_other_observed_demand"]=other
        extra["mechanism_other_demand_coverage"]=raw.reindex(columns=other_cols).notna().mean(axis=1)
        complete=raw.reindex(columns=gas_cols).sum(axis=1,min_count=5)
        for name,units in (("13",(1,2,3)),("45",(4,5))):
            gcols=[f"gas_blast_furnace_{i}" for i in units]
            hcols=[f"gas_air_heater_{i}" for i in units]
            gas=raw.reindex(columns=gcols).sum(axis=1,min_count=len(units))
            heater=raw.reindex(columns=hcols).sum(axis=1,min_count=len(units))
            paired=pd.DataFrame(raw.reindex(columns=gcols).to_numpy()-raw.reindex(columns=hcols).to_numpy(),index=f.index)
            extra[f"mechanism_{name}_paired_observed_net"]=paired.sum(axis=1,min_count=1)
            extra[f"mechanism_{name}_paired_coverage"]=paired.notna().mean(axis=1)
            # 热风炉逐炉抵扣，其他已观测需求仅按进气比例归属，避免重复扣气。
            share=gas/complete.where(complete>1.)
            extra[f"mechanism_{name}_observed_balance"]=gas-heater-share*other
            extra[f"mechanism_{name}_inlet_share"]=share
        columns=gas_cols+heater_cols+["gas_coke_oven_1","gas_into_gas_mixed_blast_furnace",
            "gas_into_gas_mixed_coke","gas_into_gas_mixed_converter"]
        for col in columns:
            v=raw.reindex(columns=[col])[col]
            for window in (4,16,96):
                extra[f"{col}_reference_mean{window}"]=v.rolling(window,min_periods=1).mean().shift(1)
            extra[f"{col}_reference_deviation96"]=v-v.rolling(96,min_periods=1).mean().shift(1)
    if "missingness" in parts:
        for col in raw:
            v=raw[col];age=base[col+"_age"]
            extra[col+"_is_missing"]=v.isna().astype(float)
            extra[col+"_never_seen"]=age.isna().astype(float)
            for window in (4,16,96):
                extra[f"{col}_coverage{window}"]=v.notna().rolling(window,min_periods=1).mean()
        cols=[f"gas_blast_furnace_{i}" for i in range(1,6)]
        gas=raw.reindex(columns=cols);valid=gas.notna()
        extra["blast_observed_zero_count"]=gas.eq(0.).sum(axis=1)
        for lag in (1,4,16):
            paired=gas-gas.shift(lag)
            # 只在两端均有观测的同一批高炉上计算变化，缺列不制造产气跳变。
            extra[f"blast_common_change{lag}"]=paired.sum(axis=1,min_count=1)
            extra[f"blast_common_count{lag}"]=paired.notna().sum(axis=1)
            extra[f"blast_newly_missing{lag}"]=(~valid & valid.shift(lag,fill_value=False)).sum(axis=1)
            extra[f"blast_newly_observed{lag}"]=(valid & ~valid.shift(lag,fill_value=False)).sum(axis=1)
    if not parts <= {"base", "memory", "dynamics", "relative", "mechanism", "missingness"}:
        raise ValueError(f"未知负荷特征：{kind}")
    return pd.concat((base, pd.DataFrame(extra, index=base.index)), axis=1).replace([np.inf, -np.inf], np.nan).astype("float32")


def load_horizon_block(prepared,base,positions,horizons,feature_set):
    X,names=horizon_block(prepared,base,positions,horizons)
    if "mechanism" not in feature_set.split("+"):
        return X,names
    target=pd.DatetimeIndex(np.tile(prepared.index[positions].to_numpy(),len(horizons)))+pd.to_timedelta(np.repeat(horizons,len(positions))*15,unit="min")
    slot=target.hour*4+target.minute//15;weekday=target.dayofweek
    extra=np.column_stack((np.sin(2*np.pi*slot/96),np.cos(2*np.pi*slot/96),
        np.sin(4*np.pi*slot/96),np.sin(2*np.pi*weekday/7),np.cos(2*np.pi*weekday/7)))
    return np.column_stack((X,extra)).astype('float32'),names+['target_slot_sin','target_slot_cos',
        'target_slot_sin2','target_week_sin','target_week_cos']


def horizon_groups(span, kind):
    if kind == "shared":
        return [list(range(1, 9 if span == "short" else 97))]
    if span == "long" and kind == "segmented":
        return [list(range(1, 9)), list(range(9, 33)), list(range(33, 97))]
    raise ValueError(f"不支持的提前量结构：{span}/{kind}")


def profile_pack(prepared, cutoff, days=28):
    history = prepared.labels.loc[:cutoff].tail(days*96)
    large = history.generator_all-history.generator_1
    labels = history.assign(generator_large=large)
    slots = labels.index.hour*4+labels.index.minute//15
    fallback = labels.median()
    values = labels.groupby(slots).median().reindex(range(96)).fillna(fallback)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("历史画像缺少有效支持")
    levels = fallback.to_numpy()
    enabled = levels > 1.
    shape = np.ones((96, 3))
    shape[:, enabled] = np.maximum(values.to_numpy()[:, enabled]/levels[enabled], .01)
    return {"values": values.to_numpy().tolist(), "shape": shape.tolist(),
            "shape_enabled": enabled.tolist(), "support": labels.groupby(slots).count().reindex(range(96), fill_value=0).to_numpy().tolist(),
            "cutoff": str(pd.Timestamp(cutoff)), "days": days,
            "source": "training_prefix_only", "zero_labels_retained": True}


def training_positions(index, days, stride, phase):
    start = max(96, len(index)-days*96)
    candidates = np.arange(start, len(index)-1)
    if phase == "fixed":
        return candidates[(candidates-start) % stride == 0]
    if phase != "rotating":
        raise ValueError("起报相位无效")
    dates = index[candidates].normalize().asi8 // pd.Timedelta(days=1).value
    slots = index[candidates].hour*4+index[candidates].minute//15
    selected = candidates[np.asarray(slots % stride == dates % stride)]
    count = len(candidates[::stride])
    return selected[:count]


def required_targets(spec, span):
    representation = spec.get("representation", "pair")
    if representation == "pair":
        targets = [0, 1]
    elif representation in ("reference_difference", "reference_ratio"):
        if not np.array_equal(spec.get("tree_weights", [1., 1.]), [1., 1.]):
            raise ValueError("参考水平结构必须保留全部预测头")
        return [0, 1, 2]
    elif representation == "group_difference":
        targets = [1, 2]
    elif representation == "group_ratio":
        targets = [1, 2]
    else:
        raise ValueError("未知机组组表示")
    weights = spec.get("tree_weights", [1., 1.])
    return [ti for ti in targets if ti == 2 or weights[ti] > 0.]


def train_load(prepared, span, recipe=None, quick=False, progress=None):
    recipe = dict(recipe or C.LOAD_RECIPE)
    spec = load_recipe(recipe, span)
    if spec.get("causal_selector"):
        return train_regime_load(prepared, span, recipe, quick, progress)
    if spec.get("anchor_transform"):
        return train_anchored_load(prepared, span, recipe, quick, progress)
    if spec.get("members"):
        prepared.require_training(prepared.index[-1])
        maximum = 8 if span == "short" else 96
        bundle = LoadBundle(span, list(range(1, maximum+1)), [], recipe=deepcopy_recipe(recipe),
                            train_end=str(prepared.index[-1]))
        bundle.recipe.update(load_version=C.LOAD_VERSION,
            frozen_profile=profile_pack(prepared, prepared.index[-1], spec.get("profile_days", 28)),
            effective_budget={"quick": bool(quick), "threads": C.CPU_THREADS, "kind": "fixed_ensemble"})
        weights = np.asarray([item["weight"] for item in spec["members"]], float)
        if len(weights) != 2 or not np.isfinite(weights).all() or (weights <= 0.).any() or not np.isclose(weights.sum(), 1.):
            raise ValueError("固定组合必须包含两个正权重分量，且权重和为1")
        for index, item in enumerate(spec["members"]):
            if item["options"].get("members"):
                raise ValueError("不允许嵌套研究组合")
            child_recipe = deepcopy_recipe(recipe)
            child_recipe["tasks"][span] = deepcopy_recipe(item["options"])
            child = train_load(prepared, span, child_recipe, quick, progress)
            bundle.members.append((float(item["weight"]), child))
            bundle.diagnostics.extend(dict(d, member_index=index) for d in child.diagnostics)
        return bundle
    if spec.get("window_level", False):
        return train_window_load(prepared, span, recipe, quick, progress)
    cutoff = prepared.index[-1]
    prepared.require_training(cutoff)
    base = build_load_features(prepared, spec.get("feature_set", "base"))
    maximum = 8 if span == "short" else 96
    stride = max(8, spec["stride"]) if quick else spec["stride"]
    days = min(56, spec["days"]) if quick else spec["days"]
    rounds = min(60, spec["rounds"]) if quick else spec["rounds"]
    pos = training_positions(prepared.index, days, stride, spec.get("phase", "fixed"))
    bundle = LoadBundle(span, list(range(1, maximum+1)), [], recipe=recipe, train_end=str(cutoff))
    bundle.recipe = dict(recipe, effective_budget={"quick": bool(quick), "days": days,
        "stride": stride, "origins": len(pos), "max_rounds": rounds, "threads": C.CPU_THREADS,
        "first_origin": str(prepared.index[pos[0]]), "last_origin": str(prepared.index[pos[-1]])})
    pack = profile_pack(prepared, cutoff, spec.get("profile_days", 28))
    bundle.recipe["frozen_profile"] = pack
    kinds = spec.get("target_kinds", ["direct", "direct"])
    kinds = kinds + ["direct"]*(3-len(kinds))
    targets = required_targets(spec, span)
    weights = spec.get("tree_weights", [1., 1.])
    bundle.recipe["load_version"] = C.LOAD_VERSION
    bundle.recipe["loss_definition"] = {
        "positive_weight": f"1/max(abs(weight_target_mw),1)**{spec.get('weight_power',1.)}",
        "zero_weight": f"1/max(median_positive_weight_target_mw,1)**{spec.get('weight_power',1.)}",
        "auxiliary_weight_target":spec.get('aux_weight_target','original'),
        "factor_conversion": "label=y_mw/S;weight=relative_weight*S",
        "auxiliary_target": "generator_all-generator_1;both_labels_required"}
    if spec.get("representation") in ("group_ratio", "reference_ratio"):
        bundle.recipe["loss_definition"].update(
            auxiliary_target="generator_1/generator_all; both labels required; both-zero maps to zero",
            ratio_conversion="relative g1 weight multiplied by true total MW during training only")
    if spec.get("process_scale"):
        bundle.recipe["loss_definition"].update(
            factor_conversion="label=y_mw/(shape*process_scale);weight=relative_weight*shape*process_scale",
            process_scale="causal four-furnace trailing 4-step median / 1e6; invalid or zero uses identity; raw fuel unchanged")
    if spec.get("representation") == "reference_ratio":
        bundle.recipe["loss_definition"].update(
            reference_heads="0=unscaled_total_reference;1=process_scaled_total;2=small_fraction;small=head0*head2",
            process_scale_targets=[1],reference_shape_target=1)
    shape_rows = np.ones((len(pos), 96, 3), dtype="float32")
    shape_sources = []
    if "factor" in kinds:
        first_day = prepared.index[pos[0]].normalize()
        block_ids = np.asarray((prepared.index[pos].normalize()-first_day).days//3)
        for block in np.unique(block_ids):
            rows = block_ids == block
            end = first_day+pd.Timedelta(days=int(block)*3)-pd.Timedelta(minutes=15)
            block_pack = profile_pack(prepared, end, spec.get("profile_days", 28))
            shape_rows[rows] = np.asarray(block_pack["shape"], "float32")
            shape_sources.append({"block_start": str(end+pd.Timedelta(minutes=15)), "shape_cutoff": str(end)})
    for group in horizon_groups(span, spec[span]):
        X, names = load_horizon_block(prepared,base,pos,group,spec.get('feature_set','base'))
        indices = np.concatenate([pos+h for h in group])
        available = indices < len(prepared.index)
        times = (pd.DatetimeIndex(np.tile(prepared.index[pos].to_numpy(), len(group)))+
                 pd.to_timedelta(np.repeat(group, len(pos))*15, unit="min")).to_numpy()
        labels = np.full((len(indices), 3), np.nan)
        labels[available, :2] = prepared.labels.iloc[indices[available]].to_numpy()
        labels[:, 2] = labels[:, 1]-labels[:, 0]
        original_small = labels[:, 0].copy()
        reference_mode = spec.get("representation") in ("reference_difference", "reference_ratio")
        ratio_mode = spec.get("representation") in ("group_ratio", "reference_ratio")
        if ratio_mode:
            with np.errstate(invalid="ignore", divide="ignore"):
                labels[:, 2] = labels[:, 0]/labels[:, 1]
            labels[(labels[:, 0] == 0.) & (labels[:, 1] == 0.), 2] = 0.
        if reference_mode:
            labels[:, 0] = labels[:, 1]
        target_slots = pd.DatetimeIndex(times).hour*4+pd.DatetimeIndex(times).minute//15
        for ti in targets:
            if ti < 2 and weights[ti] == 0.:
                continue
            tick = time.perf_counter()
            y = labels[:, ti]
            scale = np.tile(target_process_scale(prepared, spec, ti)[pos], len(group)) if not (ti == 2 and ratio_mode) else np.ones(len(y))
            if kinds[ti] == "factor":
                shape_ti = 1 if reference_mode and ti == 0 else ti
                scale *= shape_rows[np.tile(np.arange(len(pos)), len(group)), target_slots, shape_ti]
            original = original_small if ti == 2 and ratio_mode else y
            conversion = np.maximum(labels[:, 1], 1.) if ti == 2 and ratio_mode else 1.
            model, diag = fit_booster(X, y/scale, names, times, loss=spec["loss"], rounds=rounds,
                seed=spec["seed"], leaves=spec["leaves"], weight_scale=scale*conversion, original_y=original,
                decay_days=spec.get("decay_days"), learning_rate=spec.get("learning_rate"),
                weight_power=spec.get('weight_power',1.),
                weight_target=original_small if ti==2 and spec.get('aux_weight_target')=='small' else None)
            bundle.models.append((ti, group, model))
            bundle.features = names
            bundle.diagnostics.append(dict(diag, target=("generator_all_reference" if reference_mode and ti == 0 else "generator_1_fraction" if ti == 2 and ratio_mode
                else (*C.TARGETS, "generator_large")[ti]),
                horizons=group, structure=kinds[ti], shape_sources=shape_sources if kinds[ti] == "factor" else [],
                seconds=round(time.perf_counter()-tick, 3)))
            if progress:
                progress(f"{span} t={ti} {kinds[ti]} h={group[0]}..{group[-1]} {diag['rows']}行 {diag['rounds']}轮")
        del X
    return bundle


def train_window_load(prepared, span, recipe, quick=False, progress=None):
    """以完整未来窗口的相对损失中位水平作标签，降低逐提前量标签噪声。"""
    if span != "short":
        raise ValueError("窗口整体水平仅用于已采用的短期分量")
    spec = load_recipe(recipe, span)
    cutoff = prepared.index[-1]
    prepared.require_training(cutoff)
    base = build_load_features(prepared, spec.get("feature_set", "base"))
    stride = max(8, spec["stride"]) if quick else spec["stride"]
    days = min(56, spec["days"]) if quick else spec["days"]
    rounds = min(60, spec["rounds"]) if quick else spec["rounds"]
    pos = training_positions(prepared.index, days, stride, spec.get("phase", "fixed"))
    maximum = 8 if span == "short" else 96
    bundle = LoadBundle(span, list(range(1, maximum+1)), [], recipe=deepcopy_recipe(recipe), train_end=str(cutoff))
    bundle.recipe.update(load_version=C.LOAD_VERSION, frozen_profile=profile_pack(prepared, cutoff, spec.get("profile_days", 28)),
        effective_budget={"quick": bool(quick), "days": days, "stride": stride, "origins": len(pos),
                          "max_rounds": rounds, "threads": C.CPU_THREADS},
        loss_definition={"window_level": "weighted_median(y/scale); weights=scale/relative_denominator**power; scale=shape*causal_process_scale_if_enabled",
                         "zero_labels_retained": True, "partial_label_windows_used": False})
    if spec.get("representation") == "reference_difference":
        bundle.recipe["loss_definition"].update(
            reference_heads="0=unscaled_total_reference;1=process_scaled_total;2=unscaled_large;small=head0-head2",
            process_scale_targets=[1],reference_shape_target=1)
    shapes = np.ones((len(pos), 96, 3), dtype="float32")
    first_day = prepared.index[pos[0]].normalize()
    blocks = np.asarray((prepared.index[pos].normalize()-first_day).days//3)
    sources = []
    for block in np.unique(blocks):
        end = first_day+pd.Timedelta(days=int(block)*3)-pd.Timedelta(minutes=15)
        shapes[blocks == block] = np.asarray(profile_pack(prepared, end, spec.get("profile_days", 28))["shape"])
        sources.append({"block_start": str(end+pd.Timedelta(minutes=15)), "shape_cutoff": str(end)})
    targets = required_targets(spec, span)
    power = spec.get("weight_power", 1.)
    for group in horizon_groups(span, spec[span]):
        mid = group[(len(group)-1)//2]
        X, names = load_horizon_block(prepared, base, pos, [mid], spec.get("feature_set", "base"))
        ix = pos[:, None]+np.asarray(group)[None, :]
        available = ix < len(prepared.index)
        y = np.full((*ix.shape, 3), np.nan)
        y[available, :2] = prepared.labels.to_numpy()[ix[available]]
        y[..., 2] = y[..., 1]-y[..., 0]
        original_small = y[..., 0].copy()
        reference_mode = spec.get("representation") == "reference_difference"
        if reference_mode:
            y[..., 0] = y[..., 1]
        future = pd.DatetimeIndex(np.repeat(prepared.index[pos].to_numpy(), len(group)))+pd.to_timedelta(np.tile(group, len(pos))*15, unit="min")
        slots = (future.hour*4+future.minute//15).to_numpy().reshape(ix.shape)
        times = (prepared.index[pos]+pd.Timedelta(minutes=15*max(group))).to_numpy()
        for ti in targets:
            raw = y[..., ti]
            shape_ti = 1 if reference_mode and ti == 0 else ti
            scale = shapes[np.arange(len(pos))[:, None], slots, shape_ti]*target_process_scale(prepared, spec, ti)[pos, None]
            weight_target = original_small if ti == 2 and spec.get("aux_weight_target") == "small" else raw
            complete = np.isfinite(raw).all(axis=1) & np.isfinite(weight_target).all(axis=1)
            positive = weight_target[np.isfinite(weight_target) & (weight_target > 0.)]
            zero_denominator = max(float(np.median(positive)) if len(positive) else 1., 1.)
            denominator = np.where(weight_target == 0., zero_denominator, np.maximum(np.abs(weight_target), 1.))
            weights = scale/denominator**power
            levels = raw/scale
            order = np.argsort(levels, axis=1)
            ordered_values = np.take_along_axis(levels, order, axis=1)
            ordered_weights = np.take_along_axis(weights, order, axis=1)
            total_weight = ordered_weights.sum(axis=1)
            indices = np.argmax(np.cumsum(ordered_weights, axis=1) >= total_weight[:, None]/2., axis=1)
            target = ordered_values[np.arange(len(pos)), indices]
            original = raw.mean(axis=1)
            target[~complete] = np.nan
            original[~complete] = np.nan
            positive_mean = original[np.isfinite(original) & (original > 0.)]
            mean_zero = max(float(np.median(positive_mean)) if len(positive_mean) else 1., 1.)
            mean_denominator = np.where(original == 0., mean_zero, np.maximum(np.abs(original), 1.))
            conversion = total_weight*mean_denominator**power
            model, diag = fit_booster(X, target, names, times, loss=spec["loss"], rounds=rounds,
                seed=spec["seed"], leaves=spec["leaves"], original_y=original, weight_scale=conversion,
                weight_power=power, decay_days=spec.get("decay_days"), learning_rate=spec.get("learning_rate"))
            bundle.models.append((ti, group, model)); bundle.features = names
            bundle.diagnostics.append(dict(diag, target="generator_all_reference" if reference_mode and ti == 0 else (*C.TARGETS, "generator_large")[ti],
                horizons=group, structure="window_factor", shape_sources=sources,
                target_aggregation="full_window_relative_weighted_median"))
            if progress:
                progress(f"{span} t={ti} window_factor h={group[0]}..{group[-1]} {diag['rows']}行 {diag['rounds']}轮")
    return bundle


def target_process_scale(prepared, spec, target):
    """参考水平和机组组关系保持原尺度，过程修正仅作用于实际全厂目标。"""
    if spec.get("representation") in ("reference_difference", "reference_ratio") and target != 1:
        return np.ones(len(prepared.index))
    return process_scale(prepared, spec)


def process_scale(prepared, spec):
    """用起报点已知过程水平换算标签，缺测或零量只回退单位尺度，不补写燃料。"""
    mode = spec.get("process_scale")
    if not mode:
        return np.ones(len(prepared.index))
    if mode != "stable_four":
        raise ValueError("过程尺度配方不兼容")
    columns = [f"gas_blast_furnace_{i}" for i in (1,2,4,5)]
    values = prepared.frame[columns].sum(axis=1, min_count=len(columns))
    current = values.rolling(4, min_periods=4).median().to_numpy()
    # 百万立方米每小时只用于数值归一化，不是气电换算系数。
    return np.where(np.isfinite(current) & (current > 0.), current/1e6, 1.)


def deepcopy_recipe(recipe):
    from copy import deepcopy
    return deepcopy(recipe)


def reconcile(values, spec):
    pred = values[..., :2].copy()
    representation = spec.get("representation", "pair")
    if representation == "group_difference":
        pred[..., 0] = values[..., 1]-values[..., 2]
    elif representation == "reference_difference":
        pred[..., 0] = values[..., 0]-values[..., 2]
    elif representation == "reference_ratio":
        pred[..., 0] = values[..., 0]*np.clip(values[..., 2], 0., 1.)
    elif representation == "group_ratio":
        pred[..., 0] = values[..., 1]*np.clip(values[..., 2], 0., 1.)
    return pred


def consistent_short(raw, origins, alpha):
    if not alpha:
        return raw
    out = raw.copy()
    origins = pd.DatetimeIndex(origins)
    lookup = {t: i for i, t in enumerate(origins)}
    for i, origin in enumerate(origins):
        for h in range(7):
            prior, weights = [], []
            for k in range(1, 8-h):
                j = lookup.get(origin-pd.Timedelta(minutes=15*k))
                if j is not None:
                    prior.append(raw[j, h+k])
                    weights.append(1./k)
            if prior:
                out[i, h] = (1.-alpha)*raw[i, h]+alpha*np.average(prior, axis=0, weights=weights)
    return out


def predict_components(bundle, prepared, origins):
    if bundle.recipe.get("load_version") != C.LOAD_VERSION:
        raise ValueError("负荷模型版本不兼容，必须使用当前配方重新训练")
    origins = pd.DatetimeIndex(origins)
    if not len(origins) or origins.has_duplicates or not origins.is_monotonic_increasing:
        raise ValueError("起报点必须非空、递增且不重复")
    if pd.Timestamp(bundle.train_end) >= origins[0]:
        raise ValueError("模型标签截止必须早于起报窗口")
    spec = load_recipe(bundle.recipe, bundle.span)
    if spec.get("anchor_transform"):
        return predict_anchored_components(bundle, prepared, origins)
    base = build_load_features(prepared, spec.get("feature_set", "base"))
    positions = prepared.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("缺少请求的起报网格")
    pack = bundle.recipe["frozen_profile"]
    if pack["cutoff"] != bundle.train_end:
        raise ValueError("冻结画像截止与模型不一致")
    slots = np.stack([(origins+pd.Timedelta(minutes=15*h)).hour*4+
                      (origins+pd.Timedelta(minutes=15*h)).minute//15 for h in bundle.horizons], axis=1)
    baseline = np.asarray(pack["values"])[slots]
    values = baseline.copy()
    kinds = spec.get("target_kinds", ["direct", "direct"])
    kinds = kinds+["direct"]*(3-len(kinds))
    reference_mode = spec.get("representation") in ("reference_difference", "reference_ratio")
    for lo in range(0, len(positions), 256):
        pos = positions[lo:lo+256]
        for ti, group, model in bundle.models:
            actual_group = [group[(len(group)-1)//2]] if kinds[ti] == "window_factor" else group
            X, names = load_horizon_block(prepared,base,pos,actual_group,spec.get('feature_set','base'))
            if names != bundle.features or model.feature_name() != names:
                raise ValueError("模型特征顺序不兼容")
            pred = model.predict(X, num_threads=C.TREE["num_threads"]).reshape(len(actual_group), len(pos)).T
            if kinds[ti] == "window_factor":
                pred = np.repeat(pred, len(group), axis=1)
            if kinds[ti] in ("factor", "window_factor"):
                shape_ti = 1 if reference_mode and ti == 0 else ti
                pred *= np.asarray(pack["shape"])[slots[lo:lo+len(pos)][:, np.array(group)-1], shape_ti]
            if not (ti == 2 and spec.get("representation") in ("group_ratio", "reference_ratio")):
                pred *= target_process_scale(prepared, spec, ti)[pos, None]
            values[lo:lo+len(pos), np.array(group)-1, ti] = pred
    weights = np.asarray(spec.get("tree_weights", [1., 1.]))
    values[..., :2] = values[..., :2]*weights+(1.-weights)*baseline[..., :2]
    return values


def predict_load(bundle, prepared, origins):
    spec = load_recipe(bundle.recipe, bundle.span)
    origins = pd.DatetimeIndex(origins)
    if not len(origins) or origins.has_duplicates or not origins.is_monotonic_increasing:
        raise ValueError("起报点必须非空、递增且不重复")
    if origins[0] <= pd.Timestamp(bundle.train_end) or not origins.isin(prepared.index).all():
        raise ValueError("请求起报点必须位于训练截止之后且存在于输入网格")
    if bundle.recipe.get("load_version") != C.LOAD_VERSION:
        raise ValueError("负荷模型版本不兼容")
    validate_regime_bundle(bundle)
    if bundle.experts:
        predictions = {name:predict_load(child,prepared,origins) for name,child in bundle.experts.items()}
        sources = bundle.recipe["regime_selection"]["target_sources"]
        return project_load(np.stack([predictions[name][...,i] for i,name in enumerate(sources)],axis=-1))
    if bundle.members:
        if any(child.span != bundle.span or child.train_end != bundle.train_end for _, child in bundle.members):
            raise ValueError("组合分量跨度或训练截止不一致")
        result = sum(weight*predict_load(child, prepared, origins) for weight, child in bundle.members)
        return project_load(result)
    alpha = spec.get("consistency_alpha", 0.) if bundle.span == "short" else 0.
    all_origins = origins
    if alpha:
        # 每次从请求点的合法前七步补算原始预测，因此分批、单点调用具有相同结果。
        previous = [origins-pd.Timedelta(minutes=15*k) for k in range(1, 8)]
        all_origins = origins.union(pd.DatetimeIndex(np.concatenate([x.to_numpy() for x in previous])).unique()).sort_values()
        all_origins = all_origins[(all_origins > pd.Timestamp(bundle.train_end)) & all_origins.isin(prepared.index)]
    values = predict_components(bundle, prepared, all_origins)
    pred = reconcile(values, spec)
    pred = consistent_short(pred, all_origins, alpha)
    return project_load(pred[all_origins.get_indexer(origins)])


def frozen_profile(prepared, days=28):
    return project_load(np.asarray(profile_pack(prepared, prepared.index[-1], days)["values"])[:, :2])


def naive_profile(prepared, origins, horizons, cutoff):
    profile = frozen_profile(prepared.prefix(cutoff))
    rows = []
    for h in horizons:
        future = pd.DatetimeIndex(origins)+pd.Timedelta(minutes=15*h)
        rows.append(profile[future.hour*4+future.minute//15])
    return project_load(np.stack(rows, axis=1))


def train_short(prepared, recipe=None, quick=False, progress=None):
    return train_load(prepared, "short", recipe, quick, progress)


def predict_short(bundle, prepared, origins):
    if bundle.span != "short":
        raise ValueError("短期入口拒绝长期模型")
    return predict_load(bundle, prepared, origins)


def anchor_inputs(prepared):
    """过程上下文仅取完整分量的向后观测，不填造缺失燃料。"""
    f = prepared.frame
    blast = f[[f"gas_blast_furnace_{i}" for i in (1,2,4,5)]].sum(axis=1, min_count=4)
    heaters = f[[f"gas_air_heater_{i}" for i in range(1,6)]].sum(axis=1, min_count=5)
    coke = process_series(prepared).coke_available_proxy
    return pd.DataFrame({"blast":blast, "heaters":heaters, "coke":coke}).rolling(4,min_periods=4).median()


def anchor_pack(prepared, cutoff, days=7, inputs=None):
    pack = profile_pack(prepared, cutoff, days)
    history = (anchor_inputs(prepared) if inputs is None else inputs).loc[:cutoff].tail(days*96)
    references = history.median().where(history.count() >= 24)
    pack.update(anchor_inputs=references.to_numpy().tolist(), anchor_input_support=history.count().tolist(),
        anchor_input_basis="causal complete-component 4-step median; >=24 observations; no fuel imputation")
    return pack


def anchored_block(X, names, positions, origins, horizons, profiles, ages, inputs, references, target):
    """同一合法输入配不同已知目标截止点，学习目标长期不可见时的变化。"""
    n = len(positions); rows = np.tile(np.arange(n),len(horizons))
    future = pd.DatetimeIndex(np.tile(origins.to_numpy(),len(horizons)))+pd.to_timedelta(np.repeat(horizons,n)*15,unit="min")
    slots = future.hour*4+future.minute//15
    levels = np.maximum(profiles[rows,slots,target],1.)
    with np.errstate(divide="ignore",invalid="ignore"):
        ratios = inputs / references
    ratios[~np.isfinite(ratios) | (references <= 0.)] = np.nan
    extended = np.column_stack((X,levels,np.tile(ages,len(horizons)),ratios[rows])).astype("float32")
    return extended,names+["anchor_value_mw","anchor_age_days","anchor_ratio_blast","anchor_ratio_heaters","anchor_ratio_coke"],levels,future.to_numpy()


def train_anchored_load(prepared, span, recipe, quick=False, progress=None):
    spec = load_recipe(recipe,span); transform = spec["anchor_transform"]
    if span != "short" or transform != "delta" or spec.get("representation") != "pair":
        raise ValueError("负荷水平锚点配方不兼容")
    cutoff = prepared.index[-1]; prepared.require_training(cutoff)
    base = build_load_features(prepared,spec["feature_set"])
    process = anchor_inputs(prepared)
    stride = max(16,spec["stride"]) if quick else spec["stride"]
    days = min(56,spec["days"]) if quick else spec["days"]
    rounds = min(60,spec["rounds"]) if quick else spec["rounds"]
    initial = training_positions(prepared.index,days,stride,"rotating")
    initial = initial[initial >= 28*96]
    rng = np.random.default_rng(spec["seed"])
    gaps = (rng.integers(0,22,len(initial))[:,None]+np.array([0,5,11,16]))%22
    positions = np.repeat(initial,4); gaps = gaps.reshape(-1)
    origin = prepared.index[positions]
    anchor_cutoffs = origin.normalize()-pd.to_timedelta(gaps,unit="D")-pd.Timedelta(minutes=15)
    assert (anchor_cutoffs < origin).all()
    profiles = np.zeros((len(positions),96,3),dtype="float32")
    references = np.full((len(positions),3),np.nan)
    pack_records = []
    for end in anchor_cutoffs.unique():
        pack = anchor_pack(prepared,end,spec["profile_days"],process)
        mask = anchor_cutoffs == end
        profiles[mask] = np.asarray(pack["values"])
        references[mask] = np.asarray(pack["anchor_inputs"],float)
        pack_records.append(str(end))
    ages = np.asarray((origin-anchor_cutoffs).total_seconds()/86400.)
    current = process.to_numpy()[positions]
    maximum = 8
    bundle = LoadBundle(span,list(range(1,maximum+1)),[],recipe=deepcopy_recipe(recipe),train_end=str(cutoff))
    bundle.recipe.update(load_version=C.LOAD_VERSION,frozen_profile=anchor_pack(prepared,cutoff,spec["profile_days"],process),
        effective_budget={"quick":bool(quick),"days":days,"stride":stride,"origins":len(initial),
            "anchor_cases":len(positions),"max_rounds":rounds,"threads":C.CPU_THREADS},
        loss_definition={"transform":transform,"anchor_floor_mw":1.,"zero_labels_retained":True,
            "delta_conversion":"y-A; relative MW loss weight unchanged", "synthetic_gap_days":[0,21],
            "gap_sampling":"4 distinct uniformly shifted ages per origin, seed fixed",
            "anchor_cutoffs":pack_records,"outer_validation_anchor_frozen":True})
    for group in horizon_groups(span,spec[span]):
        horizons = group
        X,names = load_horizon_block(prepared,base,positions,horizons,spec["feature_set"])
        ix = np.concatenate([positions+h for h in horizons]);available = ix < len(prepared.index)
        raw = np.full((len(ix),2),np.nan);raw[available] = prepared.labels.to_numpy()[ix[available]]
        for target in (0,1):
            extended,features,scale,times = anchored_block(X,names,positions,origin,horizons,profiles,ages,current,references,target)
            y = raw[:,target]
            labels = y-scale
            model,diag = fit_booster(extended,labels,features,times,loss="relative_l1",rounds=rounds,
                seed=spec["seed"],leaves=spec["leaves"],original_y=y,
                weight_scale=np.ones(len(y)),
                decay_days=spec.get("decay_days"),learning_rate=spec.get("learning_rate"))
            bundle.models.append((target,group,model));bundle.features = features
            bundle.diagnostics.append(dict(diag,target=C.TARGETS[target],horizons=group,
                training_horizons=horizons,structure="frozen_absolute_level_"+transform,
                max_anchor_cutoff=str(anchor_cutoffs.max()),max_anchor_age_days=float(ages.max())))
            if progress:
                progress(f"{span} t={target} anchor_{transform} h={group[0]}..{group[-1]} {diag['rows']}行 {diag['rounds']}轮")
    return bundle


def predict_anchored_components(bundle,prepared,origins):
    spec = load_recipe(bundle.recipe,bundle.span);transform = spec["anchor_transform"]
    if bundle.span != "short" or transform != "delta":
        raise ValueError("负荷水平锚点变换错误")
    pack = bundle.recipe["frozen_profile"]
    if pack["cutoff"] != bundle.train_end or "anchor_inputs" not in pack:
        raise ValueError("负荷水平锚点来源或过程上下文缺失")
    positions = prepared.index.get_indexer(origins)
    if (positions < 0).any():raise ValueError("缺少起报网格")
    base = build_load_features(prepared,spec["feature_set"])
    process = anchor_inputs(prepared).to_numpy()
    values = np.zeros((len(origins),len(bundle.horizons),3))
    for lo in range(0,len(origins),256):
        pos = positions[lo:lo+256];origin = origins[lo:lo+256]
        profiles = np.broadcast_to(np.asarray(pack["values"]),(len(pos),96,3))
        references = np.broadcast_to(np.asarray(pack["anchor_inputs"],float),(len(pos),3))
        ages = np.asarray((origin-pd.Timestamp(bundle.train_end)).total_seconds()/86400.)
        for target,group,model in bundle.models:
            X,names = load_horizon_block(prepared,base,pos,group,spec["feature_set"])
            extended,features,scale,_ = anchored_block(X,names,pos,origin,group,profiles,ages,process[pos],references,target)
            if features != bundle.features or model.feature_name() != features:
                raise ValueError("锚点模型特征顺序不一致")
            pred = model.predict(extended,num_threads=C.CPU_THREADS)
            pred = pred+scale
            values[lo:lo+len(pos),np.asarray(group)-1,target] = pred.reshape(len(group),len(pos)).T
    return values


REGIME_ROUTES = (("core","core"),("anchor","core"),("core","anchor"),("anchor","anchor"))


def choose_regime(scores, minimum):
    valid = {"/".join(route):scores["/".join(route)] for route in REGIME_ROUTES
        if scores["/".join(route)].get("macro_score") is not None and
        all(scores["/".join(route)][t]["nonzero_cells"] >= minimum for t in C.TARGETS)}
    if not valid:
        return ["core","core"]
    best = max(valid,key=lambda key:valid[key]["macro_score"])
    return best.split("/")


def select_regime(prepared, span, recipe, quick=False, progress=None):
    """只在训练前缀最后21天回放，整段遮挡回放标签后决定两目标来源。"""
    if span != "short":raise ValueError("当前因果选择只用于独立短期链路")
    policy = deepcopy_recipe(load_recipe(recipe,span)["causal_selector"])
    if policy.get("version") != "prefix21d_targets_v2" or policy.get("history_days") != 21 or set(policy["experts"]) != {"core","anchor"}:
        raise ValueError("因果选择配方不兼容")
    cutoff = prepared.index[-1];prepared.require_training(cutoff)
    start = cutoff-pd.Timedelta(days=21)+pd.Timedelta(minutes=15)
    train = prepared.prefix(start-pd.Timedelta(minutes=15))
    origins = pd.date_range(start,cutoff,freq="15min")
    view = prepared.masked_copy(start,mask_use=True);view.labels=view.labels.copy();view.labels.loc[start:]=np.nan
    truth = truth_cube(prepared,origins,range(1,9))
    predictions = {}; models = {}
    for name in ("core","anchor"):
        child_recipe = deepcopy_recipe(recipe);child_recipe["tasks"][span] = deepcopy_recipe(policy["experts"][name])
        models[name] = train_load(train,span,child_recipe,quick,progress)
        predictions[name] = predict_load(models[name],view,origins)
    scores = {}
    for route in REGIME_ROUTES:
        pred = project_load(np.stack([predictions[name][...,i] for i,name in enumerate(route)],axis=-1))
        scores["/".join(route)] = score_load(pred,truth)
    sources = choose_regime(scores,policy["min_cells"])
    decision = {"version":policy["version"],"policy":policy,"target_sources":sources,"scores":scores,
        "inner_start":str(start),"inner_end":str(cutoff),"inner_train_end":str(train.index[-1]),
        "max_label_used_at":str(cutoff),"outer_targets_used":False,"inner_target_blackout":True,
        "future_targets_past_cutoff_excluded":True,"selection_performed_at":str(cutoff)}
    if progress:progress(f"short 因果目标选择：小机组={sources[0]}，全厂={sources[1]}")
    return decision,models


def train_regime_load(prepared, span, recipe, quick=False, progress=None):
    tick = time.perf_counter()
    decision,inner = select_regime(prepared,span,recipe,quick,progress)
    del inner
    experts = {}
    for name in sorted(set(decision["target_sources"])):
        resolved = deepcopy_recipe(recipe);resolved["tasks"][span] = deepcopy_recipe(decision["policy"]["experts"][name])
        experts[name] = train_load(prepared,span,resolved,quick,progress)
    if len(experts) == 1:
        bundle = next(iter(experts.values()))
        bundle.recipe["regime_selection"] = decision
    else:
        bundle = LoadBundle(span,list(range(1,9)),[],recipe=deepcopy_recipe(recipe),train_end=str(prepared.index[-1]),experts=experts)
        bundle.recipe.update(load_version=C.LOAD_VERSION,regime_selection=decision,
            frozen_profile=profile_pack(prepared,prepared.index[-1]),effective_budget={"quick":bool(quick),"threads":C.CPU_THREADS})
        bundle.diagnostics = [dict(d,selected_expert=name) for name,child in experts.items() for d in child.diagnostics]
    bundle.recipe["regime_training_seconds"] = time.perf_counter()-tick
    validate_regime_bundle(bundle)
    return bundle


def validate_regime_bundle(bundle):
    selection = bundle.recipe.get("regime_selection")
    if selection is None:
        if bundle.experts or load_recipe(bundle.recipe,bundle.span).get("causal_selector"):
            raise ValueError("缺少因果目标选择记录")
        return
    policy = selection.get("policy",{})
    if bundle.span != "short" or selection.get("version") != "prefix21d_targets_v2" or policy.get("version") != selection["version"]:
        raise ValueError("因果选择版本或任务错误")
    if set(policy.get("experts",{})) != {"core","anchor"} or policy.get("history_days") != 21 or policy.get("min_cells") != 100:
        raise ValueError("因果选择策略契约错误")
    cutoff = pd.Timestamp(bundle.train_end);start = pd.Timestamp(selection["inner_start"])
    if (pd.Timestamp(selection["inner_end"]) != cutoff or pd.Timestamp(selection["max_label_used_at"]) > cutoff or
        start != cutoff-pd.Timedelta(days=21)+pd.Timedelta(minutes=15) or
        pd.Timestamp(selection["inner_train_end"]) != start-pd.Timedelta(minutes=15) or
        selection.get("outer_targets_used") is not False or selection.get("inner_target_blackout") is not True or
        selection.get("future_targets_past_cutoff_excluded") is not True or
        pd.Timestamp(selection["selection_performed_at"]) != cutoff):
        raise ValueError("因果选择时间边界错误")
    sources = selection.get("target_sources")
    if set(selection.get("scores",{})) != {"/".join(route) for route in REGIME_ROUTES}:
        raise ValueError("因果回放缺少完整目标组合")
    if sources != choose_regime(selection["scores"],policy["min_cells"]):
        raise ValueError("因果目标来源与前缀回放分数不一致")
    if bundle.experts:
        if bundle.models or bundle.members or set(bundle.experts) != set(sources) or len(bundle.experts) != 2:
            raise ValueError("因果专家覆盖错误")
        children = bundle.experts
    else:
        if len(set(sources)) != 1:raise ValueError("混合目标缺少实际专家")
        children = {sources[0]:bundle}
    for name,child in children.items():
        if child.span != bundle.span or child.train_end != bundle.train_end:
            raise ValueError("因果专家训练截止不一致")
        actual = load_recipe(child.recipe,child.span)
        if any(actual.get(k) != v for k,v in policy["experts"][name].items()):
            raise ValueError("因果专家实际配方不一致")



def validate_deployment_pair(short,long):
    expected={"version":"short_level_v1","ratio_bounds":[.5,1.5]}
    if (short.span!="short" or long.span!="long" or short.train_end!=long.train_end or
        short.recipe.get("deployment")!=expected or long.recipe.get("deployment")!=expected):
        raise ValueError("长短期部署配方、跨度或训练截止不一致")


def align_long_level(long_pred,short_pred):
    """只校正全厂水平，保留长期形状和小机组预测；所有输入均为同起报点模型输出。"""
    short=np.asarray(short_pred,float);out=np.asarray(long_pred,float).copy()
    if (short.ndim!=3 or out.ndim!=3 or short.shape[1:]!=(8,2) or out.shape[1:]!=(96,2) or
        short.shape[0]!=out.shape[0] or not np.isfinite(short).all() or not np.isfinite(out).all()):
        raise ValueError("长短期预测形状错误或包含非有限值")
    reference=out[:,:,1].mean(axis=1)
    current=short[:,:,1].mean(axis=1)
    ratio=np.clip(current/np.maximum(reference,1e-6),.5,1.5)
    out[:,:,1]*=ratio[:,None]
    # 保留独立的小机组预测，只收敛全厂容量关系，避免用总量校正反向移动小机组。
    out[:,:,1]=np.clip(out[:,:,1],out[:,:,0],out[:,:,0]+sum(C.UNITS[4:]))
    return out


def predict_deployed(short,long,prepared,origins):
    """训练验证、正式推理、模型复用及逐格重推理共用同一部署出口。"""
    validate_deployment_pair(short,long)
    ps=predict_short(short,prepared,origins)
    pl=predict_load(long,prepared,origins)
    return ps,align_long_level(pl,ps)
