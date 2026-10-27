"""独立负荷树模型：共享数值原语，短长期训练结果严格分离。"""
from dataclasses import dataclass, field
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import config as C
from data_cleaning.pipeline import build_features, horizon_block
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


def fit_booster(X, y, names, times, *, loss="l1", rounds=320, seed=2026, leaves=31):
    valid = np.isfinite(y)
    X, y, times = X[valid], y[valid], np.asarray(times)[valid]
    if len(y) < 100:
        raise ValueError(f"有效训练标签不足：{len(y)}")
    weight = np.ones(len(y), dtype=float)
    if loss == "relative_l1":
        # 1MW以下分母截断仅用于训练稳定；评分仍使用非零真值的原始分母。
        weight = 1./np.maximum(y, 1.)
        positive = y[y > 0.]
        weight[y == 0.] = 1./max(float(np.median(positive)) if len(positive) else 1., 1.)
        weight /= weight.mean()
    split = pd.Timestamp(times.max()) - pd.Timedelta(days=7)
    valid_rows = times >= split.to_datetime64()
    train_rows = times < (split-pd.Timedelta(days=1)).to_datetime64()
    params = dict(C.TREE, objective="regression_l1" if loss != "l2" else "regression",
                  metric="l1", seed=seed, num_leaves=leaves)
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
                   "max_target_time": str(pd.Timestamp(times.max())), "inner_split": str(split)}


def horizon_groups(span, kind):
    full = list(range(1, 9 if span == "short" else 97))
    if kind == "shared":
        return [full]
    raise ValueError(f"正式训练仅允许已选共享提前量配方：{kind}")


def train_load(prepared, span, recipe=None, quick=False, progress=None):
    recipe = dict(recipe or C.LOAD_RECIPE)
    cutoff = prepared.index[-1]
    prepared.require_training(cutoff)
    base = build_features(prepared)
    kind = recipe[span]
    groups = horizon_groups(span, kind)
    maximum = 8 if span == "short" else 96
    stride = max(8, recipe["stride"]) if quick else recipe["stride"]
    days = min(56, recipe["days"]) if quick else recipe["days"]
    start = max(96, len(base)-days*96)
    pos = np.arange(start, len(base)-maximum, stride)
    bundle = LoadBundle(span, list(range(1,maximum+1)), [], recipe=recipe, train_end=str(cutoff))
    bundle.recipe = dict(recipe, effective_budget={"quick":bool(quick), "days":days,
        "stride":stride, "max_rounds":min(60,recipe["rounds"]) if quick else recipe["rounds"],
        "threads":C.CPU_THREADS})
    profile=frozen_profile(prepared,recipe.get("profile_days",28))
    bundle.recipe["frozen_profile"]={"values":profile.tolist(),"cutoff":str(cutoff),
                                      "days":recipe.get("profile_days",28)}
    weights=recipe.get("blend_weights",{}).get(span,[1.,1.])
    for group in groups:
        X, names = horizon_block(prepared, base, pos, group)
        target_positions = np.concatenate([pos+h for h in group])
        times = prepared.index[target_positions].to_numpy()
        for ti, name in enumerate(C.TARGETS):
            if weights[ti]==0.:
                bundle.diagnostics.append({"target":name,"basis":"frozen_causal_profile","tree_trained":False})
                continue
            tick = time.perf_counter()
            y = prepared.labels.iloc[target_positions, ti].to_numpy()
            model, diag = fit_booster(X, y, names, times, loss=recipe["loss"],
                                     rounds=min(60,recipe["rounds"]) if quick else recipe["rounds"],
                                     seed=recipe["seed"], leaves=recipe["leaves"])
            bundle.models.append((ti, group, model))
            bundle.features = names
            bundle.diagnostics.append(dict(diag, target=name, horizons=group, seconds=round(time.perf_counter()-tick,3)))
            if progress:
                progress(f"{span} {name} h={group[0]}..{group[-1]}，{diag['rows']}行，{diag['rounds']}轮")
        del X
    return bundle


def predict_load(bundle, prepared, origins):
    origins = pd.DatetimeIndex(origins)
    if pd.Timestamp(bundle.train_end) >= origins[0]:
        raise ValueError("模型标签截止必须早于起报窗口")
    base = build_features(prepared)
    positions = prepared.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("缺少请求的起报网格")
    profile_pack=bundle.recipe.get("frozen_profile")
    weights=np.array(bundle.recipe.get("blend_weights",{}).get(bundle.span,[1.,1.]))
    baseline=None
    if profile_pack:
        if profile_pack["cutoff"]!=bundle.train_end:
            raise ValueError("冻结画像截止与模型不一致")
        profile=np.asarray(profile_pack["values"],float)
        if profile.shape!=(96,2) or not np.isfinite(profile).all():
            raise ValueError("冻结画像不完整")
        baseline=np.stack([profile[(origins+pd.Timedelta(minutes=15*h)).hour*4+
                                   (origins+pd.Timedelta(minutes=15*h)).minute//15] for h in bundle.horizons],axis=1)
    pred = baseline.copy() if baseline is not None else np.full((len(origins), len(bundle.horizons), 2), np.nan)
    for lo in range(0, len(positions), 256):
        pos = positions[lo:lo+256]
        for ti, group, model in bundle.models:
            X, names = horizon_block(prepared, base, pos, group)
            if names != bundle.features:
                raise ValueError("模型特征顺序不兼容")
            values = model.predict(X, num_threads=C.TREE["num_threads"]).reshape(len(group),len(pos)).T
            pred[lo:lo+len(pos), np.array(group)-1, ti] = values
    if baseline is not None:
        pred=pred*weights+(1.-weights)*baseline
    return project_load(pred)


def frozen_profile(prepared,days=28):
    historical = prepared.labels.tail(days*96)
    slots = historical.index.hour*4+historical.index.minute//15
    profile = historical.groupby(slots).median().reindex(range(96)).fillna(historical.median())
    return project_load(profile.to_numpy())


def naive_profile(prepared, origins, horizons, cutoff):
    profile=frozen_profile(prepared.prefix(cutoff))
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
