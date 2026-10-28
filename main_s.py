"""独立负荷树模型：共享数值原语，短长期训练结果严格分离。"""
from dataclasses import dataclass, field
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import config as C
from data_cleaning.pipeline import build_features, horizon_block, process_series
from data_cleaning.quality import project_load


@dataclass
class LoadBundle:
    span: str
    horizons: list
    features: list
    models: list = field(default_factory=list)
    recipe: dict = field(default_factory=dict)
    train_end: str = ""
    diagnostics: list = field(default_factory=list)


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
    elif representation == "group_difference":
        targets = [1, 2]
    else:
        raise ValueError("未知机组组表示")
    weights = spec.get("tree_weights", [1., 1.])
    return [ti for ti in targets if ti == 2 or weights[ti] > 0.]


def train_load(prepared, span, recipe=None, quick=False, progress=None):
    recipe = dict(recipe or C.LOAD_RECIPE)
    spec = load_recipe(recipe, span)
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
        target_slots = pd.DatetimeIndex(times).hour*4+pd.DatetimeIndex(times).minute//15
        for ti in targets:
            if ti < 2 and weights[ti] == 0.:
                continue
            tick = time.perf_counter()
            y = labels[:, ti]
            scale = np.ones(len(y))
            if kinds[ti] == "factor":
                scale = shape_rows[np.tile(np.arange(len(pos)), len(group)), target_slots, ti]
            model, diag = fit_booster(X, y/scale, names, times, loss=spec["loss"], rounds=rounds,
                seed=spec["seed"], leaves=spec["leaves"], weight_scale=scale, original_y=y,
                decay_days=spec.get("decay_days"), learning_rate=spec.get("learning_rate"),
                weight_power=spec.get('weight_power',1.),
                weight_target=labels[:,0] if ti==2 and spec.get('aux_weight_target')=='small' else None)
            bundle.models.append((ti, group, model))
            bundle.features = names
            bundle.diagnostics.append(dict(diag, target=(*C.TARGETS, "generator_large")[ti],
                horizons=group, structure=kinds[ti], shape_sources=shape_sources if kinds[ti] == "factor" else [],
                seconds=round(time.perf_counter()-tick, 3)))
            if progress:
                progress(f"{span} t={ti} {kinds[ti]} h={group[0]}..{group[-1]} {diag['rows']}行 {diag['rounds']}轮")
        del X
    return bundle


def reconcile(values, spec):
    pred = values[..., :2].copy()
    representation = spec.get("representation", "pair")
    if representation == "group_difference":
        pred[..., 0] = values[..., 1]-values[..., 2]
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
    for lo in range(0, len(positions), 256):
        pos = positions[lo:lo+256]
        for ti, group, model in bundle.models:
            X, names = load_horizon_block(prepared,base,pos,group,spec.get('feature_set','base'))
            if names != bundle.features or model.feature_name() != names:
                raise ValueError("模型特征顺序不兼容")
            pred = model.predict(X, num_threads=C.TREE["num_threads"]).reshape(len(group), len(pos)).T
            if kinds[ti] == "factor":
                pred *= np.asarray(pack["shape"])[slots[lo:lo+len(pos)][:, np.array(group)-1], ti]
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
