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
                weight_power=1., weight_target=None, state_target=None):
    valid = np.isfinite(y)
    X, y, times = X[valid], y[valid], np.asarray(times)[valid]
    if len(y) < 100:
        raise ValueError(f"有效训练标签不足：{len(y)}")
    observed = y if original_y is None else np.asarray(original_y)[valid]
    weight = np.ones(len(y), dtype=float)
    state_balance = None
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
        if state_target is not None:
            # 仅用内层训练期统计状态频次，限制稀少状态的放大倍数，避免借用留出标签。
            state = np.asarray(state_target)[valid]
            ids = np.digitize(state, [50., 100., 150.])
            inner_end = pd.Timestamp(times.max())-pd.Timedelta(days=8)
            available = np.isfinite(state) & (times < inner_end.to_datetime64())
            counts = np.bincount(ids[available], minlength=4)
            largest = max(int(counts.max()), 1)
            factors = np.ones(4)
            for i in range(4):
                if counts[i]:
                    factors[i] = min(2., np.sqrt(largest/counts[i]))
            weight *= np.where(np.isfinite(state), factors[ids], 1.)
            state_balance = {"edges_mw": [50., 100., 150.], "counts": counts.tolist(),
                             "factors": factors.tolist(), "max_factor": 2.,
                             "count_before": str(inner_end), "missing_state_factor": 1.}
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
                   "state_balance": state_balance,
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
    if kind == "segmented":
        if span == "short":
            # 短期近端与远端的误差结构和最优复杂度不同；实测两段优于共享、单步优于两段
            # （六窗 short macro 87.36→87.40→88.17，long 同步 87.58→87.68→87.91，
            #  w6 爬升窗 gall +0.99；代价是 w1/w4 的 g1 各 -0.5 左右）。
            return [[h] for h in range(1, 9)]
        return [list(range(1, 9)), list(range(9, 33)), list(range(33, 97))]
    raise ValueError(f"不支持的提前量结构：{span}/{kind}")


def _slot_profile(prepared, cutoff, days):
    """按日内时段取中位数，得到画像取值与整体水平。"""
    history = prepared.labels.loc[:cutoff].tail(days*96)
    large = history.generator_all-history.generator_1
    labels = history.assign(generator_large=large)
    slots = labels.index.hour*4+labels.index.minute//15
    fallback = labels.median()
    values = labels.groupby(slots).median().reindex(range(96)).fillna(fallback)
    return labels, values.to_numpy(), fallback.to_numpy()


def _slot_smooth(values, kernel):
    """环形平滑 96 个日内时段，降低小样本中位数的形状噪声，并保持整体水平不变。"""
    v = np.asarray(values, float)
    k = np.asarray(kernel, float)
    pad = len(k)//2
    padded = np.concatenate((v[-pad:], v, v[:pad]))
    smoothed = np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, padded)
    total = smoothed.sum(axis=0)
    return np.where(total > 0., smoothed*(v.sum(axis=0)/np.where(total > 0., total, 1.))[None, :], smoothed)


def profile_pack(prepared, cutoff, days=28, smooth=None):
    """画像包：最近 days 天的日内时段中位数与整体水平。"""
    labels, values, levels = _slot_profile(prepared, cutoff, days)
    if smooth:
        values = _slot_smooth(values, smooth)
    if not np.isfinite(values).all():
        raise ValueError("历史画像缺少有效支持")
    enabled = levels > 1.
    shape = np.ones((96, 3))
    shape[:, enabled] = np.maximum(values[:, enabled]/levels[enabled], .01)
    slots = labels.index.hour*4+labels.index.minute//15
    return {"values": values.tolist(), "shape": shape.tolist(),
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
    pack = profile_pack(prepared, cutoff, spec.get("profile_days", 28), spec.get("profile_smooth"))
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
            block_pack = profile_pack(prepared, end, spec.get("profile_days", 28), spec.get("profile_smooth"))
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
                weight_target=labels[:,0] if ti==2 and spec.get('aux_weight_target')=='small' else None,
                state_target=labels[:,0] if span=='short' else None)
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
    scale = spec.get("baseline_scale")
    if scale is not None:
        # 1-MAPE 对高估的惩罚重于低估：按赛题指标形状给冻结画像做轻微水平修正，
        # 系数在四窗历史验证上确定，只作用于该任务的输出，不改变训练与特征。
        values[..., :2] = values[..., :2]*np.asarray(scale, float)[:2]
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


def holder_anchor_factor(prepared, origins, cfg):
    """柜位平衡反推的发电水平相对 28 天同刻中位的比值（有界锚定系数）。

    只用起报点及以前已观测的发生量、热风炉、高炉用户、混合量与双柜变化：
    单柜净变化是已发生的观测量，不读未来。双柜任一缺测时比值不可用，回退为 1
    （不锚定、不插补）。系数只做有界缩放，不改变形状，也不写回任何输入。
    """
    if not cfg:
        return None
    f = prepared.frame
    blast_cols = [f"gas_blast_furnace_{i}" for i in range(1, 6)]
    heater_cols = [f"gas_air_heater_{i}" for i in range(1, 6)]
    user_cols = [f"gas_user_blast_furnace_user{i}" for i in range(1, 5)]
    blast = f[blast_cols].sum(axis=1, min_count=len(blast_cols))
    heater = f[heater_cols].sum(axis=1, min_count=len(heater_cols))
    user = f[user_cols].sum(axis=1, min_count=len(user_cols))
    mixed = f["gas_into_gas_mixed_blast_furnace"]
    hold = f[list(C.HOLDERS)].sum(axis=1, min_count=2)
    # 部分发生量不是完整供气；当前或上一柜位缺失时不能沿用滚动窗口中的旧值缩放。
    valid = blast.notna() & heater.notna() & user.notna() & mixed.notna() & hold.notna() & hold.shift(1).notna()
    balance = (blast-heater-user-mixed-hold.diff()/C.DT).where(valid)
    implied = balance.rolling(int(cfg.get("smooth", 16)), min_periods=4).mean().where(valid)
    ref = implied.rolling(int(cfg.get("days", 28))*96, min_periods=96).median()
    ratio = (implied/ref).reindex(pd.DatetimeIndex(origins)).to_numpy(float)
    ratio = np.where(np.isfinite(ratio), ratio, 1.)
    return np.clip(ratio, float(cfg["lo"]), float(cfg["hi"]))**float(cfg["gamma"])


def predict_deployed(short, long, prepared, origins, recipe=None):
    """正式推理、回测和自动选优共用同一出口，防止漏评已采用的后处理。"""
    if long.span != "long":
        raise ValueError("长期入口拒绝短期模型")
    recipe = C.LOAD_RECIPE if recipe is None else recipe
    ps = predict_short(short, prepared, origins)
    pl = predict_load(long, prepared, origins)
    spec_l = recipe.get("tasks", {}).get("long", {})
    spec_s = recipe.get("tasks", {}).get("short", {})
    pl = align_long_level(pl, ps, spec_l.get("short_level_alpha", 0.))
    ps = shrink_short_to_long(ps, pl, spec_s.get("shrink_to_long", 0.))
    guard = short.recipe.get("tasks", {}).get("short", {}).get("profile_upper_guard")
    if not isinstance(guard, dict) or not (guard.get("ratio", 0.) >= 1. and 0. < guard.get("scale", 0.) <= 1.):
        raise ValueError("短期模型缺少兼容的画像上沿校正参数")
    origin_index = pd.DatetimeIndex(origins)
    slots = np.stack([(origin_index + pd.Timedelta(minutes=15*h)).hour*4 +
                      (origin_index + pd.Timedelta(minutes=15*h)).minute//15
                      for h in range(1, 9)], axis=1)
    reference = np.asarray(short.recipe["frozen_profile"]["values"])[slots, 1]
    # 画像冻结于训练截止前；仅降低明显高于同槽画像的短期预测。
    ps = ps * np.where((reference > 0.) & (ps[:, :, 1] > guard["ratio"]*reference), guard["scale"], 1.)[:, :, None]
    # 柜位平衡锚：把当日“实际发出去的气”落实成水平系数，抑制整日状态级偏移。
    factor = holder_anchor_factor(prepared, origins, recipe.get("holder_anchor"))
    if factor is not None:
        cap_small, cap_large = float(sum(C.UNITS[:4])), float(sum(C.UNITS[4:]))
        for frame in (ps, pl):
            frame *= factor[:, None, None]
            frame[:, :, 0] = np.clip(frame[:, :, 0], 0., cap_small)
            frame[:, :, 1] = np.clip(frame[:, :, 1], frame[:, :, 0], frame[:, :, 0]+cap_large)
    # 分组份额校准放在最后：它按 gall 重构 g1，必须在所有水平/形状后处理之后执行，
    # 否则后续缩放会把目标份额重新拉开。
    ps, pl = calibrate_group_share(ps, pl, recipe.get("share_calibration"))
    return reconcile_groups(ps), pl


def shrink_short_to_long(short_pred, long_pred, weight):
    """把短期预测按提前量向长期（同起报点、同目标时刻）线性收缩。

    短期 8 步在远端已明显衰减（四窗实测 h=7、8 的误差贡献最大），而长期在同一起报点
    给出的是更稳的"形状×水平"估计；按 h=1..8 递增的权重做凸组合，相当于给短期
    远端加一个先验。权重为 0 时保持原状，为 1 时 h=8 完全采用长期值。
    """
    if not weight or weight <= 0.:
        return short_pred
    short=np.asarray(short_pred,float)
    long=np.asarray(long_pred,float)
    steps=short.shape[1]
    if long.shape[0]!=short.shape[0] or long.shape[2]!=short.shape[2] or long.shape[1]<steps:
        raise ValueError("短期与长期预测的起报点或目标维度不一致")
    ramp=np.linspace(float(weight)/steps,float(weight),steps)[None,:,None]
    return reconcile_groups(short*(1.-ramp)+long[:,:steps,:]*ramp)


def reconcile_groups(pred):
    """把预测约束回机组容量关系：小机组 0~200MW，总负荷不低于小机组且不超 +240MW。"""
    out=np.asarray(pred,float).copy()
    small=float(sum(C.UNITS[:4]))
    large=float(sum(C.UNITS[4:]))
    out[:, :, 0]=np.clip(out[:, :, 0],0.,small)
    out[:, :, 1]=np.clip(out[:, :, 1],out[:, :, 0],out[:, :, 0]+large)
    return out


def calibrate_group_share(short_pred,long_pred,cfg=None):
    """按目标分组份额重构 generator_1，抑制整段同向的机组组合偏差。

    模型学到的是训练期的平均份额（约 0.369），而低负荷时大机组优先满载、小机组
    少开，真实份额更低；起报后机组组合不可观测，树模型无法自行修正。这里按目标
    份额重算 g1，再用"大机组满载"下界 `gall-240` 保护物理可行性——高负荷时段大
    机组已满发，g1 无法继续下调，只能保持在下界。

    参数由 `config.SHARE_CALIBRATION` 给出，只做下调、全期单一常数，不按日期或
    时段细分，也不读取测试期标签。份额下调幅度受 `max_share_drop` 限制，避免配置
    误写把预测压到不合理区间。
    """
    cfg=C.SHARE_CALIBRATION if cfg is None else cfg
    out=[]
    for frame,span in ((short_pred,"short"),(long_pred,"long")):
        arr=np.asarray(frame,float).copy()
        target=(cfg or {}).get("target",{}).get(span)
        if not (cfg or {}).get("enabled") or target is None:
            out.append(arr)
            continue
        lower_cap=float(cfg.get("lower_capacity",240.))
        upper_cap=float(cfg.get("upper_capacity",200.))
        if not 0.15<=float(target)<=0.45:
            raise ValueError("分组份额目标超出合理区间，拒绝静默使用异常配置")
        gall=arr[:, :, 1]
        proposed=float(target)*gall
        lower=np.maximum(gall-lower_cap,0.)
        upper=np.minimum(gall,upper_cap)
        arr[:, :, 0]=np.clip(proposed,lower,upper)
        out.append(arr)
    return out[0],out[1]


def align_long_level(long_pred, short_pred, alpha):
    """用短期模型的当前水平校正长期 generator_all 的水平。

    长期是"形状×水平"结构，水平由 GBDT 从起报点特征给出；遇到供气或机组结构
    变化时反应滞后（四窗实测 w4 的长期 gall 比真值低约 30MW）。短期模型在同一
    因果边界内直接看到起报点前的气量与机组状态，因此用它的 8 步均值与长期 96 步
    均值之比做一次有界混合：alpha=0 保持原状，alpha=1 完全改用短期水平。
    只作用于 generator_all；generator_1 的冻结画像水平单独由 baseline_scale 控制。
    """
    if not alpha or alpha <= 0.:
        return long_pred
    short=np.asarray(short_pred,float)
    out=np.asarray(long_pred,float).copy()
    if short.shape[0]!=out.shape[0] or short.shape[2]!=out.shape[2]:
        raise ValueError("短期与长期预测的起报点或目标维度不一致")
    reference=out[:,:,1].mean(axis=1)
    current=short[:,:,1].mean(axis=1)
    ratio=np.clip(current/np.maximum(reference,1e-6),.5,1.5)
    out[:,:,1]*=(1.+float(alpha)*(ratio-1.))[:,None]
    # 缩放后必须仍满足机组容量与"总负荷不低于小机组负荷、大机组不超过 240MW"的
    # 硬关系，否则三表导出契约会拒绝（l_result 的 capacity_relations）。
    cap_small=float(sum(C.UNITS[:4]))
    cap_large=float(sum(C.UNITS[4:]))
    out[:,:,0]=np.clip(out[:,:,0],0.,cap_small)
    out[:,:,1]=np.clip(out[:,:,1],out[:,:,0],out[:,:,0]+cap_large)
    return out
