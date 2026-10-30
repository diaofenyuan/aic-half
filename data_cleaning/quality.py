"""独立评分与物理核验，不调用调度求解器或其修复函数。"""
import numpy as np
import pandas as pd
import config as C


def project_load(pred):
    a = np.asarray(pred, float)
    if a.shape[-1] != 2 or not np.isfinite(a).all():
        raise ValueError("负荷预测形状错误或包含非有限值")
    s, t = a[..., 0], a[..., 1]
    e0 = np.stack((np.zeros_like(s), np.clip(t, 0., 240.)), axis=-1)
    e1 = np.stack((np.full_like(s, 200.), np.clip(t, 200., 440.)), axis=-1)
    p = np.clip((s+t)/2., 0., 200.)
    e2 = np.stack((p, p), axis=-1)
    p = np.clip((s+t-240.)/2., 0., 200.)
    e3 = np.stack((p, p+240.), axis=-1)
    edges = np.stack((e0, e1, e2, e3), axis=-2)
    choice = np.argmin(np.square(edges-a[..., None, :]).sum(axis=-1), axis=-1)
    projected = np.take_along_axis(edges, choice[..., None, None], axis=-2)[..., 0, :]
    legal = (s >= 0.) & (s <= 200.) & (t >= s) & (t <= s+240.)
    return np.where(legal[..., None], a, projected)


def truth_cube(data, origins, horizons, window_end=None):
    origins = pd.DatetimeIndex(origins)
    horizons = np.asarray(list(horizons))
    if horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all() or (horizons <= 0).any() or (horizons != np.floor(horizons)).any():
        raise ValueError("预测提前量必须为正整数")
    horizons = horizons.astype(int)
    pos = data.index.get_indexer(origins)
    if (pos < 0).any():
        raise ValueError("起报点超出输入网格")
    ix = pos[:, None] + horizons[None, :]
    y = np.full((*ix.shape, 2), np.nan)
    ok = ix < len(data.index)
    # 评分窗口由调用方声明，不能借用窗口结束后的已知历史标签。
    if window_end is not None:
        targets = origins.to_numpy()[:, None] + horizons[None, :] * np.timedelta64(C.STEP, "m")
        ok &= targets <= pd.Timestamp(window_end).to_datetime64()
    y[ok] = data.labels.to_numpy()[ix[ok]]
    return y


def score_load(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if pred.shape != truth.shape or pred.ndim == 0 or pred.shape[-1] != len(C.TARGETS):
        raise ValueError("负荷预测与真值形状不一致")
    # 缺失真值不参与评分，但预测缺失不能通过缩小分母制造虚假提升。
    if not np.isfinite(pred).all():
        raise ValueError("负荷预测包含非有限值，拒绝评分")
    out, scores = {}, []
    for i, name in enumerate(C.TARGETS):
        p, y = pred[..., i], truth[..., i]
        finite = np.isfinite(y)
        if (y[finite] < 0.).any():
            raise ValueError("负荷评分真值不得为负")
        valid, zero = finite & (y > 0.), finite & (y == 0.)
        score = float(100.*(1.-np.mean(np.abs(p[valid]-y[valid])/y[valid]))) if valid.any() else None
        out[name] = {"score": score, "nonzero_cells": int(valid.sum()), "zero_cells": int(zero.sum()),
                     "missing_truth_cells": int((~finite).sum()),
                  "total_cells": int(y.size),
                  "observed_fraction": float(finite.mean()) if y.size else None,
                  "score_status": "AVAILABLE" if valid.any() else "NO_POSITIVE_TRUTH" if finite.any() else "NO_OBSERVED_TRUTH",
                     "zero_mae": float(np.mean(np.abs(p[zero]))) if zero.any() else None,
                     "bias_mw": float(np.mean(p[finite]-y[finite])) if finite.any() else None}
        if score is not None:
            scores.append(score)
    out["macro_score"] = float(np.mean(scores)) if len(scores) == 2 else None
    return out


def score_horizons(pred, truth):
    """按提前量分组的得分，并给出该组占全部相对误差的比例；只看分组得分会漏掉误差集中度。"""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    h = pred.shape[1]
    groups = ((0, 4), (4, 8)) if h == 8 else ((0, 8), (8, 32), (32, 96))
    total = {}
    for index, name in enumerate(C.TARGETS):
        y = truth[..., index]
        valid = np.isfinite(y) & (y > 0.)
        total[name] = float(np.sum(np.abs(pred[..., index][valid] - y[valid]) / y[valid])) if valid.any() else 0.
    out = {}
    for a, b in groups:
        if a >= h:
            continue
        stop = min(b, h)
        row = score_load(pred[:, a:stop], truth[:, a:stop])
        shares = {}
        for index, name in enumerate(C.TARGETS):
            y, p = truth[:, a:stop, index], pred[:, a:stop, index]
            valid = np.isfinite(y) & (y > 0.)
            share = float(np.sum(np.abs(p[valid] - y[valid]) / y[valid])) if valid.any() else 0.
            shares[name] = float(share / total[name]) if total[name] > 0. else None
        row["error_share"] = shares
        out[f"{a+1}-{stop}"] = row
    return out


def score_prediction_csv(path, labels, origins, horizons, window_end=None):
    """独立从实际提交宽表的列名和时间戳重建目标时刻，不复用模型预测数组。"""
    origins=pd.DatetimeIndex(origins)
    if not isinstance(horizons,int) or horizons<=0 or not labels.index.is_unique:
        raise ValueError("CSV评分提前量或标签时间索引无效")
    f=pd.read_csv(path)
    expected=["datetime"]+[f"{t}_t+{h*C.STEP}_pred" for t in C.TARGETS for h in range(1,horizons+1)]
    if list(f.columns)!=expected:
        raise ValueError("预测CSV列名或提前量顺序不一致")
    times=pd.DatetimeIndex(pd.to_datetime(f["datetime"],errors="raise"))
    if not times.equals(origins):
        raise ValueError("预测CSV起报网格不一致")
    pred=np.stack([f[[f"{t}_t+{h*C.STEP}_pred" for h in range(1,horizons+1)]].to_numpy(float)
                   for t in C.TARGETS],axis=2)
    target_times=times.to_numpy()[:,None]+np.arange(1,horizons+1)[None,:]*np.timedelta64(C.STEP,"m")
    truth=labels.loc[:,list(C.TARGETS)].reindex(pd.DatetimeIndex(target_times.reshape(-1))).to_numpy(float).reshape(pred.shape)
    if window_end is not None:
        truth=np.where((target_times<=pd.Timestamp(window_end).to_datetime64())[...,None],truth,np.nan)
    return score_load(pred,truth),pred,truth


def score_origin_window(pred, truth, origins, start, end):
    """以起报点和目标时刻共同限定历史子窗口，避免尾部借用窗口外真值。"""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    origins = pd.DatetimeIndex(origins)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end or pred.shape != truth.shape or pred.ndim != 3 or pred.shape[0] != len(origins):
        raise ValueError("子窗口评分范围或预测维度无效")
    chosen = (origins >= start) & (origins <= end)
    if not chosen.any():
        return {"status": "NO_ORIGINS", "start": str(start), "end": str(end), "origins": 0}
    times = origins[chosen].to_numpy()[:, None] + np.arange(1, pred.shape[1]+1)[None, :]*np.timedelta64(C.STEP, "m")
    inside = (times >= start.to_datetime64()) & (times <= end.to_datetime64())
    labels = np.where(inside[..., None], truth[chosen], np.nan)
    forecast = pred[chosen]
    return {"status": "COMPLETE", "start": str(start), "end": str(end), "origins": int(chosen.sum()),
            "score": score_load(forecast, labels), "horizons": score_horizons(forecast, labels),
            "diagnostics": load_diagnostics(forecast, labels, origins[chosen]),
            "role": "historical_diagnostic_only_not_candidate_ranking"}


def load_diagnostics(pred, truth, origins):
    """按目标时刻和历史真实负荷区间解释误差，不参与模型输入。"""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    origins = pd.DatetimeIndex(origins)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[0] != len(origins) or pred.shape[2] != len(C.TARGETS):
        raise ValueError("负荷诊断维度与起报网格不一致")
    if not np.isfinite(pred).all():
        raise ValueError("负荷诊断拒绝非有限预测")
    targets=origins.to_numpy()[:,None]+np.arange(1,pred.shape[1]+1)[None,:]*np.timedelta64(C.STEP,"m")
    target_days=targets.astype("datetime64[D]")
    ranges=((0.,30.,60.,100.,150.,200.,np.inf),(0.,100.,200.,300.,440.,np.inf))
    result={}
    for i,name in enumerate(C.TARGETS):
        p,y=pred[...,i],truth[...,i]
        all_valid=np.isfinite(y)&(y > 0.)
        all_cells=int(all_valid.sum())
        all_error=float(np.sum(np.abs(p[all_valid]-y[all_valid])/y[all_valid]))
        def summarize(mask):
            valid=mask & np.isfinite(y) & (y > 0.)
            if not valid.any():
                return {"score":None,"cells":0,"bias_mw":None,
                        "fraction_of_scored_cells":0. if all_cells else None,
                        "fraction_of_total_relative_error":0. if all_error else None}
            error=np.abs(p[valid]-y[valid])/y[valid]
            return {"score":float(100.*(1.-np.mean(error))),
                    "cells":int(valid.sum()),"bias_mw":float(np.mean(p[valid]-y[valid])),
                    "fraction_of_scored_cells":float(valid.sum()/all_cells),
                    "fraction_of_total_relative_error":float(error.sum()/all_error) if all_error else None}
        bins={}
        for lo,hi in zip(ranges[i][:-1],ranges[i][1:]):
            label=f"({lo:g},{hi:g}]"
            bins[label]=summarize((y > lo) & (y <= hi))
        daily={}
        origin_daily={}
        if len(origins):
            for day in pd.date_range(origins.min().normalize(),origins.max().normalize(),freq="D"):
                daily[str(day.date())]=summarize(target_days == day.to_datetime64().astype("datetime64[D]"))
                origin_daily[str(day.date())]=summarize(np.broadcast_to((origins.normalize()==day)[:,None],y.shape))
        # 末尾截断使不同起报点的提前量数量不等；完整起报点指标仅作透明诊断。
        complete=np.isfinite(y).all(axis=1)
        complete_mask=np.broadcast_to(complete[:,None],y.shape)
        complete_score=summarize(complete_mask)
        complete_score.update(origins=int(complete.sum()),total_origins=len(origins))
        result[name]={"by_observed_target_mw":bins,"by_target_date":daily,
                      "by_origin_date":origin_daily,"complete_horizon_origins":complete_score,
                      "information_role":"historical_scoring_only"}
    return result


def _target_error(pred, truth, index):
    """单目标的有效评分单元与相对误差；零负荷真值单列，不进 MAPE 分母。"""
    p, y = np.asarray(pred, float)[..., index], np.asarray(truth, float)[..., index]
    valid = np.isfinite(y) & (y > 0.)
    err = np.where(valid, np.abs(p - y) / np.where(valid, y, 1.), np.nan)
    return p, y, valid, err


def _mean_or_none(values):
    a = np.asarray(values, float)
    return float(np.mean(a)) if a.size and np.isfinite(a).any() else None


def aggregation_variants(pred, truth, origins):
    """同一批预测在多种聚合口径下的得分，量化“窗口均值”对口径的敏感度。
    官方聚合方式未公开，这里只做敏感性对照，不参与候选排名。"""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    origins = pd.DatetimeIndex(origins)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[0] != len(origins):
        raise ValueError("聚合口径对照需要与起报网格一致的预测和真值")
    times = origins.to_numpy()[:, None] + np.arange(1, pred.shape[1] + 1)[None, :] * np.timedelta64(C.STEP, "m")
    out = {"role": "aggregation_sensitivity_diagnostic_only", "official_aggregation": "unknown", "targets": {}}
    for index, name in enumerate(C.TARGETS):
        p, y, valid, err = _target_error(pred, truth, index)
        row = {"cells": int(valid.sum()), "missing_truth": int((~np.isfinite(y)).sum()),
               "zero_truth": int((np.isfinite(y) & (y == 0.)).sum())}
        # 逐格等权即当前主指标；逐起报点、逐目标日等权模拟“按行或按日聚合”的官方口径。
        row["cell_mean"] = float(100. * (1. - np.mean(err[valid]))) if valid.any() else None
        per_origin = [np.mean(err[j][np.isfinite(err[j])]) if np.isfinite(err[j]).any() else np.nan
                      for j in range(len(origins))]
        row["origin_mean"] = float(100. * (1. - np.mean(np.asarray(per_origin, float)[np.isfinite(per_origin)]))) \
            if np.isfinite(per_origin).any() else None
        days = times.astype("datetime64[D]")
        daily = [float(100. * (1. - np.mean(err[days == day][np.isfinite(err[days == day])])))
                 for day in np.unique(days) if np.isfinite(err[days == day]).any()]
        row["day_equal_mean"] = _mean_or_none(daily)
        row["day_count"] = len(daily)
        if valid.any() and np.sum(y[valid]) > 0.:
            row["wape_score"] = float(100. * (1. - np.sum(np.abs(p[valid] - y[valid])) / np.sum(y[valid])))
            scale = np.abs(p[valid]) + y[valid]
            row["smape"] = float(100. * (1. - np.mean(2. * np.abs(p[valid] - y[valid]) / np.where(scale > 0., scale, 1.)))) \
                if np.all(scale > 0.) else None
        else:
            row["wape_score"] = row["smape"] = None
        # 末尾截断使各起报点提前量数量不等；完整提前量口径只保留全步都有真值的起报点。
        complete = np.isfinite(y).all(axis=1)
        complete_err = err[complete][np.isfinite(err[complete])]
        row["complete_horizon"] = float(100. * (1. - np.mean(complete_err))) if complete_err.size else None
        row["complete_horizon_origins"] = int(complete.sum())
        # 官方零值处理未公开：零负荷真值若按满误差计，分数变化单独报告，不改主指标。
        zero = np.isfinite(y) & (y == 0.)
        zero_flagged = int((np.abs(p[zero]) > 1e-9).sum())
        denominator = int(valid.sum()) + zero_flagged
        if denominator:
            total = float(np.sum(np.abs(p[valid] - y[valid]) / y[valid])) + float(zero_flagged)
            row["zero_as_full_error"] = float(100. * (1. - total / denominator))
            row["zero_flagged_cells"] = zero_flagged
        else:
            row["zero_as_full_error"] = row["zero_flagged_cells"] = None
        out["targets"][name] = row
    return out


def lead_profiles(pred, truth):
    """逐提前量得分，揭示被窗口均值掩盖的提前量结构差异。"""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError("提前量诊断需要形状一致的预测和真值")
    out = {"role": "lead_profile_diagnostic_only", "targets": {}}
    for index, name in enumerate(C.TARGETS):
        rows = {}
        for h in range(pred.shape[1]):
            _, _, valid, err = _target_error(pred[:, h:h + 1], truth[:, h:h + 1], index)
            rows[str(h + 1)] = {"cells": int(valid.sum()),
                                "score": float(100. * (1. - np.mean(err[valid]))) if valid.any() else None,
                                "mean_abs_error_mw": float(np.mean(np.abs(
                                    pred[:, h, index][valid[..., 0] & np.isfinite(pred[:, h, index])]
                                    - truth[:, h, index][valid[..., 0] & np.isfinite(pred[:, h, index])])))
                                if valid.any() else None}
        scores = [v["score"] for v in rows.values() if v["score"] is not None]
        out["targets"][name] = {"leads": rows, "worst": min(scores) if scores else None,
                                "best": max(scores) if scores else None}
    return out


def day_profiles(pred, truth, origins):
    """按目标时刻日期与起报日期分组的得分分布，说明均值背后有多少下尾日。"""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    origins = pd.DatetimeIndex(origins)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[0] != len(origins):
        raise ValueError("日级诊断需要与起报网格一致的预测和真值")
    times = origins.to_numpy()[:, None] + np.arange(1, pred.shape[1] + 1)[None, :] * np.timedelta64(C.STEP, "m")
    out = {"role": "day_profile_diagnostic_only", "targets": {}}
    for index, name in enumerate(C.TARGETS):
        _, _, valid, err = _target_error(pred, truth, index)
        days = times.astype("datetime64[D]")
        origin_days = np.broadcast_to(origins.normalize().to_numpy().astype("datetime64[D]")[:, None], err.shape)
        by_target, by_origin = {}, {}
        for day in np.unique(days):
            e = err[days == day]
            mask = np.isfinite(e)
            by_target[str(pd.Timestamp(day).date())] = {"cells": int((valid & (days == day)).sum()),
                                                        "score": float(100. * (1. - np.mean(e[mask]))) if mask.any() else None}
        for day in np.unique(origin_days):
            e = err[origin_days == day]
            mask = np.isfinite(e)
            by_origin[str(pd.Timestamp(day).date())] = {"cells": int((valid & (origin_days == day)).sum()),
                                                        "score": float(100. * (1. - np.mean(e[mask]))) if mask.any() else None}
        target_scores = [v["score"] for v in by_target.values() if v["score"] is not None]
        origin_scores = [v["score"] for v in by_origin.values() if v["score"] is not None]
        def quantiles(values):
            if not values:
                return {}
            return {"count": len(values), "p10": float(np.percentile(values, 10)), "p25": float(np.percentile(values, 25)),
                    "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
                    "min": float(np.min(values)), "max": float(np.max(values))}
        out["targets"][name] = {"by_target_date": by_target, "by_origin_date": by_origin,
                                "target_date_quantiles": quantiles(target_scores),
                                "origin_date_quantiles": quantiles(origin_scores)}
    return out


def split_decomposition(pred, truth, band_edges=(0., 250., 300., 330., 360., 400.)):
    """把 generator_1 误差分解为“总量项 + 分组份额项”，并给出分区间归因。
    恒等式 p1-y1 = (s_p-s_y)*p_all + s_y*(p_all-y_all)，只作事后诊断，不进入模型输入。"""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[2] != 2:
        raise ValueError("分解需要两目标的预测和真值")
    pa, ya = pred[..., 1], truth[..., 1]
    p1, y1 = pred[..., 0], truth[..., 0]
    ok = np.isfinite(pa) & (pa > 0.) & np.isfinite(ya) & (ya > 0.) & np.isfinite(p1) & np.isfinite(y1)
    s_p = np.where(ok, p1 / np.where(ok, pa, 1.), np.nan)
    s_y = np.where(ok, y1 / np.where(ok, ya, 1.), np.nan)
    share = np.where(ok, (s_p - s_y) * pa, np.nan)
    level = np.where(ok, s_y * (pa - ya), np.nan)

    def summarize(mask):
        n = int(mask.sum())
        if not n:
            return {"cells": 0, "share_fraction": None, "mean_abs_share_mw": None, "mean_abs_level_mw": None,
                    "share_attribution": None, "bias_share_mw": None, "bias_level_mw": None,
                    "generator_1_score": None, "truth_share_ratio": None}
        total = np.abs(share[mask]) + np.abs(level[mask])
        err = (np.abs(p1[mask] - y1[mask]) / np.where(y1[mask] > 0., y1[mask], 1.))
        return {"cells": n, "share_fraction": float(n / ok.sum()),
                "mean_abs_share_mw": float(np.mean(np.abs(share[mask]))),
                "mean_abs_level_mw": float(np.mean(np.abs(level[mask]))),
                "share_attribution": float(np.sum(np.abs(share[mask])) / np.sum(total)) if np.sum(total) > 0. else None,
                "bias_share_mw": float(np.mean(share[mask])), "bias_level_mw": float(np.mean(level[mask])),
                "generator_1_score": float(100. * (1. - np.mean(err))),
                "truth_share_ratio": float(np.mean(s_y[mask]))}

    edges = list(band_edges) + [np.inf]
    bands = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        bands[f"({lo:g},{hi:g}]"] = summarize(ok & (ya > lo) & (ya <= hi))
    # 份额差 |s_p-s_y| 是 generator_1 得分的主要控制量；分档曲线便于判断当前窗口处在哪一段。
    gap_edges = (0., .01, .02, .03, .05, .08, .12, np.inf)
    gap_bands = {}
    gap = np.abs(np.where(ok, s_p - s_y, np.nan))
    for lo, hi in zip(gap_edges[:-1], gap_edges[1:]):
        gap_bands[f"[{lo:.2f},{hi:.2f})"] = summarize(ok & (gap >= lo) & (gap < hi))
    return {"role": "error_decomposition_diagnostic_only", "identity": "p1-y1=(s_p-s_y)*p_all+s_y*(p_all-y_all)",
            "cells": int(ok.sum()), "overall": summarize(ok), "by_truth_generator_all_mw": bands,
            "by_abs_share_gap": gap_bands, "mean_abs_share_gap": float(np.mean(gap[ok])) if ok.any() else None}


def joint_error_bands(pred, truth, gap_edges=(0., .02, .04, .06, .08, .12, np.inf),
                      level_edges=(0., .05, .10, .15, .25, np.inf)):
    """份额差 × 总量相对误差的二维分档得分。
    单项分档会把相互抵消的份额项与总量项重复计入，联合表才能回答“本窗口的单元格实际落在
    哪一区、该区得分是多少”，只作事后诊断。最差档必须开口，否则会静默丢掉最差的单元格。"""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[2] != 2:
        raise ValueError("联合分档需要两目标的预测和真值")
    p1, pa, y1, ya = pred[..., 0], pred[..., 1], truth[..., 0], truth[..., 1]
    ok = np.isfinite(ya) & (ya > 0.) & np.isfinite(pa) & (pa > 0.) & np.isfinite(p1) & (p1 > 0.) & np.isfinite(y1) & (y1 > 0.)
    gap = np.where(ok, p1 / np.where(ok, pa, 1.) - y1 / np.where(ok, ya, 1.), np.nan)
    level = np.where(ok, pa / np.where(ok, ya, 1.) - 1., np.nan)
    err = np.where(ok, np.abs(p1 - y1) / np.where(ok, y1, 1.), np.nan)
    table = {}
    for glo, ghi in zip(gap_edges[:-1], gap_edges[1:]):
        for llo, lhi in zip(level_edges[:-1], level_edges[1:]):
            mask = ok & (np.abs(gap) >= glo) & (np.abs(gap) < ghi) & (np.abs(level) >= llo) & (np.abs(level) < lhi)
            n = int(mask.sum())
            key = f"gap[{glo:.2f},{ghi:.2f}) level[{llo:.2f},{lhi:.2f})"
            table[key] = {"cells": n, "cell_share": float(n / ok.sum()) if ok.any() else None,
                          "score": float(100. * (1. - np.mean(err[mask]))) if n else None,
                          "mean_abs_gap": float(np.mean(np.abs(gap[mask]))) if n else None,
                          "mean_abs_level_error": float(np.mean(np.abs(level[mask]))) if n else None}
    return {"role": "joint_error_bands_diagnostic_only", "cells": int(ok.sum()),
            "gap_edges": list(gap_edges), "level_edges": list(level_edges), "table": table}


def score_reference_pack(pred, truth, origins):
    """把一次历史回放的多口径敏感性、提前量、日级分布、误差分解与零值盲区汇总成参考包。
    主指标不变；这些量用于判断本地分数能代表哪种工况，官方口径仍未知。"""
    return {"role": "scoring_reference_diagnostic_only", "official_score_replicated": False,
            "aggregation": aggregation_variants(pred, truth, origins),
            "leads": lead_profiles(pred, truth),
            "days": day_profiles(pred, truth, origins),
            "decomposition": split_decomposition(pred, truth),
            "joint_bands": joint_error_bands(pred, truth)}


def compare_bands(pred, baseline, truth):
    """在同一批单元格上对比模型与冻结画像基线（朴素）的分档得分与误差贡献。
    1-MAPE 主要由低真值档驱动；只有分档对比才能看出模型在关键档是否真的优于基线。"""
    pred, baseline, truth = (np.asarray(x, float) for x in (pred, baseline, truth))
    if pred.shape != truth.shape or baseline.shape != truth.shape or pred.ndim != 3:
        raise ValueError("分档对比需要形状一致的模型、基线预测与真值")

    def rel(p, y, mask):
        return np.abs(p[mask] - y[mask]) / y[mask]

    ranges = ((0., 30., 60., 100., 150., 200., np.inf), (0., 100., 200., 300., 440., np.inf))
    out = {"role": "band_vs_baseline_diagnostic_only", "official_mapping": "unknown", "targets": {}}
    for index, name in enumerate(C.TARGETS):
        p, b, y = pred[..., index], baseline[..., index], truth[..., index]
        valid = np.isfinite(y) & (y > 0.)
        if not valid.any():
            out["targets"][name] = {"status": "NO_OBSERVED_TRUTH"}
            continue
        total_model = float(np.sum(rel(p, y, valid)))
        total_base = float(np.sum(rel(b, y, valid)))
        bands = {}
        for lo, hi in zip(ranges[index][:-1], ranges[index][1:]):
            mask = valid & (y > lo) & (y <= hi)
            if not mask.any():
                continue
            bands[f"({lo:g},{hi:g}]"] = {
                "cells": int(mask.sum()), "cell_share": float(mask.sum() / valid.sum()),
                "model_score": float(100. * (1. - np.mean(rel(p, y, mask)))),
                "baseline_score": float(100. * (1. - np.mean(rel(b, y, mask)))),
                "delta": float(100. * (np.mean(rel(b, y, mask)) - np.mean(rel(p, y, mask)))),
                "model_error_share": float(np.sum(rel(p, y, mask)) / total_model) if total_model else None,
                "baseline_error_share": float(np.sum(rel(b, y, mask)) / total_base) if total_base else None}
        out["targets"][name] = {
            "model_score": float(100. * (1. - np.mean(rel(p, y, valid)))),
            "baseline_score": float(100. * (1. - np.mean(rel(b, y, valid)))),
            "by_observed_target_mw": bands,
            "note": "基线为冻结同槽画像（不训练）；delta 为正表示模型该档优于基线"}
    return out


def score_uncertainty(pred, truth, origins, replicates=4000, seed=20261029):
    """按目标日块自助法估计本窗口得分的抽样不确定性。
    21 天窗口的日均分数波动较大，几分的窗口差异里有多少是抽样噪声必须量化，
    否则容易把噪声当成模型改进。块重采样保留日内相关性，固定种子保证可复现。"""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    origins = pd.DatetimeIndex(origins)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[0] != len(origins):
        raise ValueError("不确定性估计需要与起报网格一致的预测和真值")
    times = origins.to_numpy()[:, None] + np.arange(1, pred.shape[1] + 1)[None, :] * np.timedelta64(C.STEP, "m")
    days = times.astype("datetime64[D]")
    rng = np.random.default_rng(seed)
    out = {"role": "score_uncertainty_diagnostic_only", "method": "target_day_block_bootstrap",
           "replicates": int(replicates), "seed": int(seed), "targets": {}}
    for index, name in enumerate(C.TARGETS):
        y, p = truth[..., index], pred[..., index]
        valid = np.isfinite(y) & (y > 0.)
        err = np.where(valid, np.abs(p - y) / np.where(valid, y, 1.), 0.)
        sums, counts = [], []
        for day in np.unique(days):
            mask = (days == day) & valid
            if mask.any():
                sums.append(float(err[mask].sum()))
                counts.append(int(mask.sum()))
        if not sums:
            out["targets"][name] = {"status": "NO_OBSERVED_TRUTH"}
            continue
        sums, counts = np.asarray(sums), np.asarray(counts)
        draws = rng.integers(0, len(sums), size=(replicates, len(sums)))
        scores = 100. * (1. - sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
        out["targets"][name] = {"days": int(len(sums)),
                                "point": float(100. * (1. - sums.sum() / counts.sum())),
                                "standard_error": float(np.std(scores, ddof=1)),
                                "p2_5": float(np.percentile(scores, 2.5)),
                                "p97_5": float(np.percentile(scores, 97.5)),
                                "half_width": float((np.percentile(scores, 97.5) - np.percentile(scores, 2.5)) / 2.),
                                "note": "区分几分的窗口差异须先看该标准误；不是官网分数方差"}
    return out


def paired_delta_uncertainty(candidate, baseline, truth, origins, replicates=4000, seed=20261029):
    """候选与基线在同一批单元格上的配对差值及其按目标日块自助标准误。
    绝对水平的标准误不能用于判断改进量：同窗同格配对时日内噪声会相互抵消。"""
    candidate, baseline, truth = (np.asarray(x, float) for x in (candidate, baseline, truth))
    origins = pd.DatetimeIndex(origins)
    if candidate.shape != truth.shape or baseline.shape != truth.shape or candidate.ndim != 3:
        raise ValueError("配对差值需要形状一致的候选、基线与真值")
    times = origins.to_numpy()[:, None] + np.arange(1, candidate.shape[1] + 1)[None, :] * np.timedelta64(C.STEP, "m")
    days = times.astype("datetime64[D]")
    rng = np.random.default_rng(seed)
    out = {"role": "paired_delta_uncertainty_diagnostic_only", "method": "target_day_block_bootstrap",
           "replicates": int(replicates), "seed": int(seed), "targets": {}}
    for index, name in enumerate(C.TARGETS):
        y = truth[..., index]
        valid = np.isfinite(y) & (y > 0.)
        pc, pb = candidate[..., index], baseline[..., index]
        err_c = np.where(valid, np.abs(pc - y) / np.where(valid, y, 1.), 0.)
        err_b = np.where(valid, np.abs(pb - y) / np.where(valid, y, 1.), 0.)
        sums, counts = [], []
        for day in np.unique(days):
            mask = (days == day) & valid
            if mask.any():
                sums.append([float(err_c[mask].sum()), float(err_b[mask].sum())])
                counts.append(int(mask.sum()))
        if not sums:
            out["targets"][name] = {"status": "NO_OBSERVED_TRUTH"}
            continue
        sums, counts = np.asarray(sums), np.asarray(counts)
        draws = rng.integers(0, len(sums), size=(replicates, len(sums)))
        total = counts[draws].sum(axis=1)
        score_c = 100. * (1. - sums[draws, 0].sum(axis=1) / total)
        score_b = 100. * (1. - sums[draws, 1].sum(axis=1) / total)
        delta = score_c - score_b
        out["targets"][name] = {
            "delta_point": float(100. * (sums[:, 1].sum() - sums[:, 0].sum()) / counts.sum()),
            "delta_standard_error": float(np.std(delta, ddof=1)),
            "delta_p2_5": float(np.percentile(delta, 2.5)),
            "delta_p97_5": float(np.percentile(delta, 97.5)),
            "candidate_level_se": float(np.std(score_c, ddof=1)),
            "baseline_level_se": float(np.std(score_b, ddof=1)),
            "days": int(len(sums)),
            "note": "正差值表示候选优于基线；水平标准误与差值标准误不可混用"}
    return out


def score_resources(pred, truth):
    p, y = np.asarray(pred, float), np.asarray(truth, float)
    if p.shape != y.shape:
        raise ValueError("资源预测与真值形状不一致")
    observed = np.isfinite(y)
    missing_prediction = observed & ~np.isfinite(p)
    result = {"wape": None, "bias": None, "cells": int(observed.sum()),
              "missing_truth_cells": int((~observed).sum()),
              "missing_prediction_cells": int(missing_prediction.sum()),
              "score_valid": False, "zero_false_volume": None}
    # 评分范围只由真值决定，预测缺失不能缩小分母或改变候选排名。
    if missing_prediction.any():
        result["reason"] = "MISSING_PREDICTION_FOR_OBSERVED_TRUTH"
        return result
    if not observed.any():
        result["reason"] = "NO_OBSERVED_TRUTH"
        return result
    error = p[observed]-y[observed]
    denom = np.abs(y[observed]).sum()
    result.update(wape=float(np.abs(error).sum()/denom) if denom > 0 else None,
                  bias=float(error.sum()/denom) if denom > 0 else None,
                  zero_false_volume=float(np.abs(p[observed & (y == 0)]).sum()*C.DT),
                  score_valid=bool(denom > 0.))
    if denom <= 0.:
        result["reason"] = "ZERO_TRUTH_VOLUME"
    return result


def price_weighted_load(days):
    """独立计算各日计划的电价加权负荷比，不推定线上分数换算。"""
    daily = []
    numerator = denominator = 0.
    for day in days:
        prices = np.asarray(day["prices"], float)
        power = np.asarray(day["power"], float)
        if prices.ndim != 1 or power.shape != (len(prices), len(C.UNITS)) or not np.isfinite(prices).all() or not np.isfinite(power).all():
            raise ValueError("价格指标需要完整的计划电价和逐台负荷")
        load = power.sum(axis=1)
        num = float(prices @ load)
        den = float(prices.mean() * load.sum())
        ratio = num/den if den > 0. else None
        daily.append({"issued_at": str(day["issued_at"]), "ratio": ratio,
                      "price_mean": float(prices.mean()), "load_sum_mw": float(load.sum())})
        numerator += num
        denominator += den
    return {"ratio": numerator/denominator if denominator > 0. else None,
            "daily": daily, "basis": "published_plan_price_mean_over_96_steps",
            "official_score_mapping": "unknown", "is_executed_revenue": False}


def _optional_calibration_checks(day):
    """由归档原预测与实测前缀重算校准，不复用调度的修正函数。"""
    control=day.get('control_calibration',{})
    policy=control.get('optional_use_calibration')
    if policy is None and control.get('version')!=C.DISPATCH_CONTROL['version']:
        return {}
    checks={'optional_calibration_contract':policy==C.OPTIONAL_USE_CALIBRATION,
        'optional_calibration_causality':True,'optional_calibration_fidelity':True}
    trace=day.get('optional_consumption_calibration',{})
    expected=C.OPTIONAL_USE_CALIBRATION
    checks['optional_calibration_contract'] &= set(trace)==set(expected['components'])
    if not checks['optional_calibration_contract']:
        return checks
    issue=pd.Timestamp(day['issued_at']);steps=expected['history_steps']
    start=issue-pd.Timedelta(minutes=(steps-1)*15)
    caps=np.asarray(day['optional_caps'],float)
    for name in expected['components']:
        item=trace[name];observed=np.asarray(item['observed_values_m3h'],float)
        original=np.asarray(item['original_forecast_m3h'],float)
        checks['optional_calibration_causality'] &= bool(pd.Timestamp(item['observed_from'])==start and
            pd.Timestamp(item['observed_through'])==issue and not item['future_truth_used'] and not item['used_for_training'])
        valid=bool(observed.shape==(steps,) and np.isfinite(observed).all() and (observed>=0.).all())
        fidelity=bool(original.shape==(96,) and np.isfinite(original).all() and (original>=0.).all() and
            item['applied']==valid and not item['missing_consumption_filled'] and item['source']==expected['source'])
        reconstructed=original.copy()
        if valid and fidelity:
            current=float(np.median(observed));reference=float(np.median(original[:steps]));delta=current-reference
            reconstructed=np.maximum(original+delta*np.exp2(-np.arange(1,97)/expected['decay_steps']),0.)
            fidelity &= bool(abs(item['observed_median_m3h']-current)<1e-6 and
                abs(item['original_forecast_reference_m3h']-reference)<1e-6 and
                abs(item['initial_delta_m3h']-delta)<1e-6 and item['decay_steps']==expected['decay_steps'])
        elif not valid:
            fidelity &= item.get('reason')=='CURRENT_LEGAL_CONSUMPTION_OBSERVATION_UNAVAILABLE'
        j={'budget_coke':0,'budget_converter':1}[name]
        values=np.asarray(day['components'][name],float)
        checks['optional_calibration_fidelity'] &= bool(fidelity and caps.shape==(96,2) and
            np.allclose(values,reconstructed,atol=1e-6,rtol=0.) and
            np.allclose(caps[:,j],reconstructed,atol=1e-6,rtol=0.))
    return checks


def _validate_day(day):
    checks, details = {}, {}
    required = ("gas", "power", "previous_power", "available", "split", "stocks", "vent", "beta", "anchor")
    absent = [x for x in required if x not in day]
    if absent:
        return {"passed": False, "checks": {"trajectory_complete": False},
                "problems": ["MISSING_TRAJECTORY:" + ",".join(absent)]}
    q, p = np.asarray(day["gas"], float), np.asarray(day["power"], float)
    n = len(q)
    checks["shapes"] = q.shape == (n, 3) and p.shape == (n, 6) and n == 96
    if not checks["shapes"]:
        return {"passed": False, "checks": checks, "problems": ["TRAJECTORY_SHAPE"]}
    cap, tol = np.asarray(C.UNITS), C.POLICY["power_tolerance"]
    stocktol, gastol = C.POLICY["stock_tolerance"], C.POLICY["gas_tolerance"]
    checks["finite_dispatch"] = bool(np.isfinite(q).all() and np.isfinite(p).all())
    checks["normal_dispatch_plan"] = not bool(day.get("fallback_used",False))
    checks["initial_state_known"] = day.get("initial_state",{}).get("basis") not in (None,"UNVERIFIED_EXPORT_STATE")
    recovery=day.get('recovery_export',{})
    if recovery.get('active'):
        checks['normal_inventory_plan']=False
    relaxed=recovery.get('empirical_ramp_relaxed_prefix_steps',0)
    if relaxed:
        scenarios=day.get('joint_control',{}).get('ramp_scenarios',[])
        checks['recovery_export_disclosure']=bool(day.get('fallback_used') is True and recovery.get('active') is True and
            recovery.get('export_only') is True and recovery.get('terminal_command_fixed') is True and
            isinstance(relaxed,int) and 0<relaxed<=96 and len(scenarios)>0 and
            scenarios==recovery.get('original_ramp_scenarios'))
        terminal=recovery.get('fixed_terminal_command') or {}
        expected_p=np.asarray(terminal.get('power',[]),float);expected_q=np.asarray(terminal.get('gas',[]),float)
        checks['recovery_terminal_command']=bool(expected_p.shape==(6,) and expected_q.shape==(3,) and
            np.isfinite(expected_p).all() and np.isfinite(expected_q).all() and
            np.allclose(p[-1],expected_p,atol=1e-6,rtol=0.) and np.allclose(q[-1],expected_q,atol=1e-6,rtol=0.))
    checks["gas_nonnegative"] = bool((q >= -gastol).all())
    checks["unit_capacity"] = bool(((p >= -tol) & (p <= cap+tol)).all())
    checks["unit_minimum"] = bool(((p <= tol) | (p >= .6*cap-tol)).all())
    previous = np.asarray(day["previous_power"], float)
    prior = np.vstack((previous, p[:-1]))
    group = np.column_stack((p[:, :4].sum(axis=1), p.sum(axis=1)))
    pg = np.column_stack((prior[:, :4].sum(axis=1), prior.sum(axis=1)))
    delta = np.abs(group-pg)
    checks["aggregate_ramp"] = bool((delta <= np.array(C.POLICY["ramp"])+tol).all())
    # 过渡边界不被强行解释为已经稳定开机；后续两端均稳定时核验逐台连续爬坡。
    running = (p >= .6*cap-tol) & (prior >= .6*cap-tol)
    checks["running_unit_ramp"] = bool((~running | (np.abs(p-prior) <= cap*.01*15+tol)).all())
    beta = np.asarray(day["beta"])
    checks["gas_power"] = bool(np.isfinite(beta).all() and np.max(np.abs(q@beta-p.sum(axis=1))) <= .01)
    a, split = np.asarray(day["available"], float), np.asarray(day["split"], float)
    h, vent, anchor = np.asarray(day["stocks"], float), np.asarray(day["vent"], float), np.asarray(day["anchor"], float)
    checks["resource_shapes"] = a.shape == (96, 2) and split.shape == (96, 2)
    checks["net_resource_basis"] = day.get("resource_mode") == "net_inferred"
    checks["split"] = bool(np.isfinite(split).all() and (split >= 0.).all() and np.allclose(split.sum(axis=1), 1.))
    valid_stock = h.shape == (97, 2) and np.isfinite(h).all() and np.isfinite(anchor).all()
    checks["observed_anchor"] = bool(day.get("anchor_observed") and np.isfinite(anchor).all())
    checks["holder_trajectory"] = bool(valid_stock)
    checks["anchor_safety"] = bool(np.isfinite(anchor).all() and
        ((anchor >= np.array(C.CAPACITY)*.15-stocktol) & (anchor <= np.array(C.CAPACITY)*.90+stocktol)).all())
    checks["independent_holder_safety"] = bool(valid_stock and
        ((h >= np.array(C.CAPACITY)*.15-stocktol) & (h <= np.array(C.CAPACITY)*.90+stocktol)).all())
    # 07:00锚点及首条指令前的桥接由先验状态决定，与指令执行后的越界分别归因。
    # 这里只补充诊断，完整安全核验仍包含初态，不能据此给予恢复宽限或改写资格。
    safe_low, safe_high = np.array(C.CAPACITY)*.15-stocktol, np.array(C.CAPACITY)*.90+stocktol
    anchor_known = bool(np.isfinite(anchor).all())
    trajectory_known = bool(h.shape == (97, 2) and np.isfinite(h).all())
    stages = {"anchor_known":anchor_known,"pre_command_stock_known":trajectory_known,
              "post_execution_known_steps":96 if trajectory_known else 0,
              "post_execution_unknown_steps":0 if trajectory_known else 96,
              "anchor_unsafe_holders":None,"pre_command_unsafe_holders":None,
              "post_execution_unsafe_steps":None,"post_execution_unsafe_steps_by_holder":None}
    if anchor_known:
        stages["anchor_unsafe_holders"] = (np.flatnonzero((anchor<safe_low)|(anchor>safe_high))+1).tolist()
    if trajectory_known:
        stages["pre_command_unsafe_holders"] = (np.flatnonzero((h[0]<safe_low)|(h[0]>safe_high))+1).tolist()
        unsafe = (h[1:]<safe_low)|(h[1:]>safe_high)
        stages["post_execution_unsafe_steps"] = int(unsafe.any(axis=1).sum())
        stages["post_execution_unsafe_steps_by_holder"] = unsafe.sum(axis=0).astype(int).tolist()
    details["holder_boundary_diagnostics"] = stages
    # 诊断只解释不可行性，不豁免实测初态，也不把偏乐观上界当作可执行轨迹。
    optional = np.asarray(day.get("optional_caps",[]),float)
    bound_known = (trajectory_known and checks["resource_shapes"] and optional.shape==(96,2) and
        np.isfinite(a).all() and np.isfinite(split).all() and np.isfinite(optional).all() and
        np.isfinite(previous).all() and beta.shape==(3,) and np.isfinite(beta).all() and beta[0]>0. and
        (optional>=0.).all() and (beta>=0.).all() and (split>=0.).all() and
        np.allclose(split.sum(axis=1),1.) and checks["initial_state_known"])
    feasibility={"scope":"local_inferred_resource_model","immutable_anchor_or_bridge_failure":bool(
        stages["anchor_unsafe_holders"] or stages["pre_command_unsafe_holders"]),
        "optimistic_bound_available":bool(bound_known),"original_safety_checks_unchanged":True,
        "official_reconstruction_certified":False}
    if bound_known:
        # 允许立即用满其他气种，并忽略逐台启停等更紧约束，得到高炉耗气的合法下界。
        minimum_power=np.maximum(previous.sum()-C.POLICY["ramp"][1]*np.arange(1,97),0.)
        minimum_blast=np.maximum((minimum_power-optional@beta[1:])/beta[0],0.)
        upper_stock=h[0]+C.DT*np.cumsum(a-split*minimum_blast[:,None],axis=0)
        unavoidable_low=(upper_stock<safe_low).any(axis=1)
        feasibility.update(minimum_blast_extraction_bound_m3h=minimum_blast.tolist(),
            optimistic_stock_upper_bound_m3=upper_stock.tolist(),
            unavoidable_low_steps_lower_bound=int(unavoidable_low.sum()),
            unavoidable_low_effective_times=[str(day["times"][k]) for k in np.flatnonzero(unavoidable_low)],
            bound_basis="aggregate_ramp_and_maximum_optional_gas_only; ignored_constraints_make_bound_optimistic")
    details["feasibility_diagnostics"]=feasibility
    checks["holder_balance"] = False
    checks["bridge_balance"] = False
    checks["overflow_rule"] = False
    checks["production_supply"] = False
    checks["planning_bounds"] = False
    if valid_stock and checks["resource_shapes"] and vent.shape == (96, 2):
        raw = h[:-1] + C.DT*(a-split*q[:, :1])
        residual = h[1:]-raw+vent*C.DT
        checks["holder_balance"] = bool(np.max(np.abs(residual)) <= stocktol)
        expected_vent = np.maximum(raw-.90*np.array(C.CAPACITY), 0.)/C.DT
        checks["overflow_rule"] = bool(np.max(np.abs(vent-expected_vent)) <= gastol)
        checks["production_supply"] = bool(((a >= -gastol) | (h[1:] >= .15*np.array(C.CAPACITY)-stocktol)).all())
        checks["planning_bounds"] = bool(((h[1:] >= .20*np.array(C.CAPACITY)-stocktol) &
                                           (h[1:] <= .85*np.array(C.CAPACITY)+stocktol)).all())
        bridge = day.get("bridge", {})
        bridge_gas = np.asarray(bridge.get("gas_m3h", [np.nan]*3), float)
        bridge_available = np.asarray(bridge.get("available_m3h", [np.nan]*2), float)
        bridge_split = np.asarray(bridge.get("split",split[0]),float)
        raw_bridge = anchor + C.DT*(bridge_available-bridge_split*bridge_gas[0])
        expected_bridge = raw_bridge-np.maximum(raw_bridge-.90*np.array(C.CAPACITY),0.)
        checks["bridge_balance"] = bool(np.isfinite(expected_bridge).all() and np.max(np.abs(h[0]-expected_bridge)) <= stocktol)
        details["max_balance_error_m3"] = float(np.max(np.abs(residual)))
        details["holder_min_ratio"] = (h/np.array(C.CAPACITY)).min(axis=0).tolist()
        details["holder_max_ratio"] = (h/np.array(C.CAPACITY)).max(axis=0).tolist()
    bridge_source=day.get('bridge',{}).get('resource_components',{})
    if 'used_components_m3h' in bridge_source:
        observed=np.asarray(bridge_source['observed_components_m3h'],float)
        estimated=np.asarray(bridge_source['estimated_components_m3h'],float)
        used=np.asarray(bridge_source['used_components_m3h'],float)
        missing=~np.isfinite(observed)
        reconstructed=np.where(missing,estimated,observed)
        component_ok=bool(observed.shape==estimated.shape==used.shape==(5,) and np.isfinite(used).all() and
            np.allclose(used,reconstructed,atol=gastol,rtol=0.) and
            bridge_source['observed_furnaces']==(np.flatnonzero(~missing)+1).tolist() and
            bridge_source['estimated_furnaces']==(np.flatnonzero(missing)+1).tolist() and
            not np.isfinite(estimated[~missing]).any())
        g=np.array([used[:3].sum(),used[3:].sum()])
        recomputed=g-np.asarray(day['bridge']['split'])*(bridge_source['known_demand_m3h']+bridge_source['reserve_m3h'])
        checks['bridge_source_fidelity']=bool(component_ok and
            np.allclose(recomputed,day['bridge']['available_m3h'],atol=gastol,rtol=0.) and
            not bridge_source['raw_input_overwritten'] and not bridge_source['missing_fuel_zero_filled'])
        checks['bridge_source_causality']=pd.Timestamp(bridge_source['max_observed_at'])<=pd.Timestamp(day['issued_at'])
        if missing.any():
            checks['bridge_source_causality'] &= pd.Timestamp(bridge_source['process_train_end'])<pd.Timestamp(day['issued_at'])
    optional = np.asarray(day.get("optional_caps", np.full((96, 2), np.nan)), float)
    checks["optional_gas_caps"] = bool(optional.shape == (96, 2) and np.isfinite(optional).all() and
                                        (q[:, 1:] <= optional+gastol).all())
    checks.update(_optional_calibration_checks(day))
    checks["export_resource_shortfall"] = bool(np.max(np.asarray(day.get("export_shortfall_m3h",[0.]),float))<=gastol)
    checks["publication_causality"] = pd.Timestamp(day.get("max_observed_at", "2100-01-01")) <= pd.Timestamp(day["issued_at"])
    reserve=day.get("reserve",{})
    if "last_closed_at" in reserve:
        checks["residual_causality"] = pd.Timestamp(reserve["last_closed_at"]) <= pd.Timestamp(day["issued_at"])
        if 'observation_age_hours' in reserve:
            age=(pd.Timestamp(day['issued_at'])-pd.Timestamp(reserve['last_closed_at'])).total_seconds()/3600.
            checks['residual_age_fidelity']=bool(age>=0. and np.isclose(reserve['observation_age_hours'],age) and
                pd.Timestamp(reserve['available_as_of'])==pd.Timestamp(day['issued_at']))
    transition=day.get('state_transition')
    if transition is not None:
        checks['state_transition_contract']=bool(transition['inventory_source']=='observed_at_publication' and
            pd.Timestamp(transition['observed_at'])==pd.Timestamp(day['issued_at']) and
            transition['reset_is_material_flow'] is False and transition['reset_is_identified_resource_error'] is False and
            transition['historical_anchor_matches_optimized_execution'] is None)
    precheck=day.get('inventory_precheck',{})
    if precheck.get('bound_available'):
        # 独立从原始轨迹重算跳过正常求解的依据，不能信任求解器自报的无解标记。
        lower_power=np.maximum(previous.sum()-C.POLICY['generate_ramp'][1]*np.arange(1,97),0.)
        lower_blast=np.maximum((lower_power-optional@beta[1:])/beta[0],0.)
        upper=h[0]+C.DT*np.cumsum(a-split*lower_blast[:,None],axis=0)
        low=np.asarray(C.CAPACITY)*C.POLICY['plan'][0]-stocktol
        indices=np.flatnonzero((upper<low).any(axis=1)).tolist()
        checks['inventory_precheck_fidelity']=bool(
            np.allclose(precheck['optimistic_inventory_upper_m3'],upper,atol=stocktol,rtol=0.) and
            precheck['planning_low_step_indices']==indices and precheck['normal_plan_infeasible_proven']==bool(indices))
    if day.get('solver',{}).get('normal_solve_skipped'):
        checks['normal_solve_skip_proven']=bool(precheck.get('bound_available') and
            precheck.get('normal_plan_infeasible_proven') and checks.get('inventory_precheck_fidelity'))
    process=day.get('process_resource',{})
    if process.get('applied'):
        checks['process_resource_causality']=bool(pd.Timestamp(process['train_end'])<pd.Timestamp(day['issued_at']) and
            pd.Timestamp(process['max_observed_at'])<=pd.Timestamp(day['issued_at']))
        checks['missing_process_not_zero_filled']=bool(not process['missing_gas_is_zero'] and not process['raw_process_values_overwritten'])
    if "times" in day:
        times=pd.DatetimeIndex(day["times"])
        expected=pd.date_range(pd.Timestamp(day["issued_at"])+pd.Timedelta(minutes=15),periods=96,freq="15min")
        checks["effective_grid"]=bool(np.array_equal(times.to_numpy(),expected.to_numpy()))
    fit=day.get("gas_power_validation")
    if fit is not None:
        checks["gas_power_calibration"]=bool(fit["passed"])
    control=day.get('control_calibration')
    if control is not None:
        joint=day.get('joint_control',{})
        checks['joint_control_complete']=bool(joint)
        checks['control_calibration_causality']=pd.Timestamp(control['train_end'])<pd.Timestamp(day['issued_at'])
        risk=joint.get('reserve_risk')
        if risk is not None:
            target=np.asarray(risk.get('target_m3',[]),float)
            valid_target=target.shape==(2,) and np.isfinite(target).all()
            risk_end=pd.Timestamp(risk.get('max_observed_at'))
            checks['reserve_risk_causality']=bool(valid_target and pd.notna(risk_end) and
                risk_end<=pd.Timestamp(day['issued_at']))
            scenario=risk.get('scenario')
            if scenario is not None:
                closed=pd.Timestamp(scenario.get('history_end'))
                checks['reserve_risk_causality'] &= bool(pd.notna(closed) and
                    closed<=pd.Timestamp(day['issued_at']) and scenario.get('coverage_guarantee') is False)
        support=joint.get('empirical_gas_support')
        if support is not None:
            equations=np.asarray(support['equations'],float)
            energy=q*beta
            violation=energy[:,1:]@equations[:,:2].T+energy.sum(axis=1)[:,None]*equations[:,2]
            # 支持域不是官方配比或独立合格条件，域外情况保留诊断，不能据此否决结果。
            details['historical_gas_support']={'role':'diagnostic_only',
                'outside_steps':int((violation.max(axis=1)>tol).sum()),
                'maximum_energy_residual_mw':float(np.maximum(violation,0.).max()),
                'official_fixed_ratio':False}
        scenarios=joint.get('ramp_scenarios',[])
        previous_gas=np.asarray(day.get('previous_gas',[np.nan]*3),float)
        if len(scenarios):
            coefficients=np.asarray(scenarios,float)
            available=bool(coefficients.ndim==2 and coefficients.shape[1]==3 and
                           np.isfinite(coefficients).all() and np.isfinite(q).all())
            sensitivity={'role':'diagnostic_only','available':available}
            if available:
                mapped=q@coefficients.T
                initial_known=previous_gas.shape==(3,) and np.isfinite(previous_gas).all()
                history=np.vstack((previous_gas@coefficients.T,mapped)) if initial_known else mapped
                change=np.abs(np.diff(history,axis=0))
                # 经验映射用于识别敏感性，实际功率的硬约束已经独立核验。
                sensitivity.update(initial_gas_known=bool(initial_known),
                    mapped_capacity_exceedances=int((mapped>sum(C.UNITS)+tol).sum()),
                    mapped_ramp_exceedances=int((change>C.POLICY['ramp'][1]+tol).sum()),
                    max_mapped_power_mw=float(np.max(mapped)),
                    max_mapped_ramp_mw=float(np.max(change)) if change.size else None)
            details['mapping_sensitivity']=sensitivity
        frontier=joint.get('price_frontier')
        if frontier is not None:
            prices=np.asarray(day.get('prices',[]),float)
            checks['price_benefit_floor']=bool(prices.shape==(n,) and np.isfinite(prices).all() and
                prices@p.sum(axis=1)*1000*C.DT>=float(frontier['benefit_floor_cny'])-.01)
    refinement=day.get("refinement",{})
    if refinement.get("applied"):
        frozen_total=np.asarray(refinement["baseline_total_power"],float)
        frozen_end=np.asarray(refinement["baseline_terminal_unit_power"],float)
        checks["refinement_fixed_total_power"]=bool(np.max(np.abs(p.sum(axis=1)-frozen_total))<=1e-5)
        checks["refinement_fixed_terminal_state"]=bool(np.max(np.abs(p[-1]-frozen_end))<=1e-5)
        checks["group_mapping_causality"]=pd.Timestamp(day["group_mapping"]["train_end"])<pd.Timestamp(day["issued_at"])
    details["max_aggregate_change_mw"] = delta.max(axis=0).tolist()
    details["validation_basis"] = {
        "scope":"project_local_strict_validation",
        "holder_capacity_source":"data_dictionary.xlsx 数据说明!A21",
        "safety_ratio_source":"contest_problem_statement",
        "planning_margin_source":"project_policy",
        "anchor_disqualification_official_rule":"unspecified",
        "official_reconstruction_reproduced":False}
    boundary_names=("initial_state_known","observed_anchor","anchor_safety")
    model_names=("net_resource_basis","gas_power","gas_power_calibration","optional_gas_caps",
                 "optional_calibration_contract","optional_calibration_fidelity")
    details["local_check_groups"]={
        "observed_boundary_passed":all(checks.get(k,True) for k in boundary_names),
        "inferred_model_passed":all(checks.get(k,True) for k in model_names),
        "planning_margin_passed":checks["planning_bounds"],
        "post_execution_holder_safety_passed":bool(trajectory_known and stages["post_execution_unsafe_steps"]==0)}
    problems=[k for k,v in checks.items() if not v]
    initial={'initial_state_known','observed_anchor','anchor_safety'}
    direct={'finite_dispatch','gas_nonnegative','unit_capacity','unit_minimum','aggregate_ramp',
        'running_unit_ramp','publication_causality','effective_grid','process_resource_causality',
        'missing_process_not_zero_filled','control_calibration_causality','group_mapping_causality','reserve_risk_causality',
        'bridge_source_fidelity','bridge_source_causality','optional_calibration_causality'}
    direct.update(('recovery_export_disclosure','recovery_terminal_command','recovery_sensitivity_prefix_only',
                   'state_transition_contract','residual_age_fidelity','inventory_precheck_fidelity','normal_solve_skip_proven'))
    assessment={'scope':'local_strict_project_model','official_holder_validation':'UNVERIFIED',
        'official_anchor_boundary_treatment':'NOT_DISCLOSED',
        'failed_initial_conditions':[k for k in problems if k in initial],
        'failed_output_or_time_checks':[k for k in problems if k in direct],
        'failed_local_model_checks':[k for k in problems if k not in initial|direct],
        'post_execution_holder_safety_passed':bool(stages['post_execution_unknown_steps']==0 and
            stages['post_execution_unsafe_steps']==0),
        'planning_margin_source':'project_selected_20_to_85_percent',
        'local_failure_is_official_rejection':False}
    return {"passed": bool(all(checks.values())), "checks": checks,
            "problems": problems, "details": details,'assessment':assessment}


def validate_day(day):
    try:
        return _validate_day(day)
    except (KeyError,ValueError,TypeError,IndexError) as exc:
        return {"passed":False,"checks":{"trajectory_schema":False},
                "problems":["INVALID_TRAJECTORY:"+str(exc)]}


def validate_plan_sequence(days):
    """直接核对发布序列和跨日初态，不信任各日自行声明的上一条指令。"""
    checks={"daily_publication_grid":False,"publication_at_0700":False,"effective_grid":False,"previous_power_link":True,
            "previous_gas_link":True,"stock_reset_link":True}
    try:
        if not days:
            return {"passed":False,"checks":checks}
        issues=pd.DatetimeIndex([d["issued_at"] for d in days])
        checks["publication_at_0700"]=bool(issues.equals(issues.normalize()+pd.Timedelta(hours=7)))
        checks["daily_publication_grid"]=bool(issues.equals(pd.date_range(issues[0],periods=len(days),freq="1D")))
        times=pd.DatetimeIndex([t for d in days for t in d["times"]])
        expected=pd.date_range(issues[0]+pd.Timedelta(minutes=15),periods=len(days)*96,freq="15min")
        checks["effective_grid"]=bool(times.equals(expected))
        for prior,current in zip(days,days[1:]):
            power=np.asarray(prior["power"],float)[-1]
            gas=np.asarray(prior["gas"],float)[-1]
            checks["previous_power_link"] &= bool(np.allclose(power,current["previous_power"],atol=C.POLICY["power_tolerance"],rtol=0.))
            checks["previous_gas_link"] &= bool(np.allclose(gas,current["previous_gas"],atol=C.POLICY["gas_tolerance"],rtol=0.))
            expected_reset=np.asarray(current["anchor"],float)-np.asarray(prior["stocks"],float)[-2]
            actual_reset=np.asarray(current["state_reset_m3"],float)
            checks["stock_reset_link"] &= bool(np.allclose(expected_reset,actual_reset,atol=C.POLICY["stock_tolerance"],rtol=0.,equal_nan=True))
            transition=current.get('state_transition')
            if transition is not None:
                checks.setdefault('state_transition_link',True)
                checks['state_transition_link'] &= bool(transition['command_source']=='previous_published_command' and
                    np.allclose(transition['previous_plan_stock_at_issue_m3'],np.asarray(prior['stocks'])[-2],
                                atol=C.POLICY['stock_tolerance'],rtol=0.,equal_nan=True))
        return {"passed":bool(all(checks.values())),"checks":checks}
    except (KeyError,ValueError,TypeError,IndexError) as exc:
        return {"passed":False,"checks":checks,"problem":"INVALID_SEQUENCE:"+str(exc)}


def full_window_coverage(origins,effective,days):
    """正式范围由配置和已发布轨迹确定，不能用报告自报的完整标志代替。"""
    expected_origins=pd.date_range(C.TEST_START,C.TEST_END,freq="15min")
    expected_issues=pd.date_range(pd.Timestamp(C.TEST_START).normalize()+pd.Timedelta(hours=7),
        pd.Timestamp(C.TEST_END).normalize()+pd.Timedelta(hours=7),freq="1D")
    expected_effective=pd.date_range(expected_issues[0]+pd.Timedelta(minutes=15),
        expected_issues[-1]+pd.Timedelta(days=1),freq="15min")
    issues=pd.DatetimeIndex([d["issued_at"] for d in days])
    times=pd.DatetimeIndex([t for d in days for t in d["times"]])
    checks={"prediction_grid":pd.DatetimeIndex(origins).equals(expected_origins),
        "optimization_grid":pd.DatetimeIndex(effective).equals(expected_effective),
        "daily_publication_grid":issues.equals(expected_issues),
        "trajectory_effective_grid":times.equals(expected_effective)}
    return {"actual_full_window":bool(all(checks.values())),"checks":checks,
        "expected_prediction_rows":len(expected_origins),"expected_optimization_rows":len(expected_effective),
        "expected_daily_publications":len(expected_issues)}


def audit_resource_forecasts(days, truth, window_end=None):
    result={}
    names=sorted(set(truth.columns) | {name for day in days for name in day["components"]})
    for name in names:
        predicted,observed,daily_error=[],[],[]
        incomplete_days=0
        for day in days:
            times=pd.DatetimeIndex(day["times"])
            p=np.asarray(day["components"].get(name,np.full(len(times),np.nan)),float)
            if p.shape != (len(times),):
                raise ValueError("资源预测长度与发布时间网格不一致")
            y=truth.reindex(times)[name].to_numpy(float)
            if window_end is not None:
                y=np.where(times <= pd.Timestamp(window_end),y,np.nan)
            predicted.extend(p);observed.extend(y)
            if np.isfinite(y).all() and np.isfinite(p).all():
                daily_error.append(float(np.sum(p-y)*C.DT))
            else:
                incomplete_days+=1
        result[name]=dict(score_resources(predicted,observed),
                          cumulative_error_m3=daily_error,
                          incomplete_daily_error_count=incomplete_days,
                          target_time_end=str(pd.Timestamp(window_end)) if window_end is not None else None,
                          mean_abs_daily_error_m3=float(np.mean(np.abs(daily_error))) if daily_error else None)
    return result


def validate_result_files(root, origins, effective):
    from pathlib import Path
    root, checks = Path(root), {}
    specs = (("s_result.csv", origins, 8), ("l_result.csv", origins, 96), ("opt_result.csv", effective, None))
    for filename, grid, horizons in specs:
        path = root/filename
        if not path.exists():
            checks[filename] = {"passed": False, "reason": "MISSING_FILE"}
            continue
        f = pd.read_csv(path)
        expected = (["datetime"] + [f"{t}_t+{h*15}_pred" for t in C.TARGETS for h in range(1,horizons+1)]
                    if horizons else ["datetime"] + ["opt_generator_use_"+g for g in C.GASES] +
                    ["opt_generator_1", "opt_generator_all"])
        if list(f.columns) != expected:
            checks[filename]={"passed":False,"reason":"INVALID_COLUMNS"}
            continue
        shape = list(f.columns) == expected and len(f) == len(grid)
        times = pd.to_datetime(f["datetime"], errors="coerce")
        time_ok = len(f) == len(grid) and np.array_equal(times.to_numpy(), pd.DatetimeIndex(grid).to_numpy())
        numeric=f.iloc[:,1:].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        finite = bool(np.isfinite(numeric).all())
        nonnegative = bool((numeric >= -.001).all())
        if horizons:
            small,total=numeric[:,:horizons],numeric[:,horizons:]
        else:
            small,total=numeric[:,-2],numeric[:,-1]
        relations=bool(((small>=-.001)&(small<=200.001)&(total>=small-.001)&(total<=small+240.001)).all())
        checks[filename] = {"passed": bool(shape and time_ok and finite and nonnegative),
                            "rows": len(f), "columns": len(f.columns), "grid": bool(time_ok),
                            "capacity_relations":relations}
        checks[filename]["passed"] &= relations
        if not horizons:
            delta=np.abs(np.diff(np.column_stack((small,total)),axis=0))
            checks[filename]["adjacent_ramp_violations"]=int((delta>np.array(C.POLICY["ramp"])+.001).any(axis=1).sum())
            checks[filename]["passed"] &= checks[filename]["adjacent_ramp_violations"]==0
    return {"passed": all(x["passed"] for x in checks.values()), "files": checks}


def historical_replay(data,days):
    """独立事后差额代理；未来柜位/基准耗气仅在此评分器中读取。"""
    counts={"known_steps":0,"unknown_steps":0,"holder_low_steps":0,"holder_high_steps_before_release":0,
            "reference_holder_low_steps":0,"additional_holder_low_steps":0,
            "proxy_stock_shortfall_m3_hours":0.,"reference_stock_shortfall_m3_hours":0.,
            "extra_stock_shortfall_m3_hours":0.}
    reset=[]
    for day in days:
        issue=pd.Timestamp(day["issued_at"])
        # 当日模型以07:00实测重锚；首段实际差额在闭合后才能供评分使用。
        required=pd.date_range(issue,periods=98,freq="15min")
        f=data.frame.reindex(required)
        baseline=f[list(C.USE)].to_numpy(float)
        observed=f[list(C.HOLDERS)].to_numpy(float)
        state=observed[0].copy()
        split=np.asarray(day["split"],float)
        command=np.asarray(day["gas"],float)
        q0=np.asarray(day["bridge"]["gas_m3h"],float)
        valid_bridge=np.isfinite(observed[:2]).all() and np.isfinite(baseline[0,0]) and np.isfinite(q0[0])
        if valid_bridge:
            bridge_split=np.asarray(day["bridge"].get("split",split[0]),float)
            state=state+observed[1]-observed[0]+C.DT*bridge_split*(baseline[0,0]-q0[0])
            state=np.minimum(state,np.array(C.CAPACITY)*.9)
        else:
            state[:]=np.nan
        known=0
        for k in range(96):
            valid=np.isfinite(state).all() and np.isfinite(observed[k+1:k+3]).all() and np.isfinite(baseline[k+1,0])
            if not valid:
                counts["unknown_steps"]+=1
                state[:]=np.nan
                continue
            state=state+observed[k+2]-observed[k+1]+C.DT*split[k]*(baseline[k+1,0]-command[k,0])
            lower=.15*np.array(C.CAPACITY)
            low=state < lower-2.
            reference_low=observed[k+2] < lower-2.
            deficit=np.maximum(lower-state,0.)
            reference_deficit=np.maximum(lower-observed[k+2],0.)
            counts["holder_low_steps"]+=int(low.any())
            counts["reference_holder_low_steps"]+=int(reference_low.any())
            counts["additional_holder_low_steps"]+=int((low & ~reference_low).any())
            counts["proxy_stock_shortfall_m3_hours"]+=float(deficit.sum()*C.DT)
            counts["reference_stock_shortfall_m3_hours"]+=float(reference_deficit.sum()*C.DT)
            counts["extra_stock_shortfall_m3_hours"]+=float(np.maximum(deficit-reference_deficit,0.).sum()*C.DT)
            counts["holder_high_steps_before_release"]+=int((state > .90*np.array(C.CAPACITY)+2.).any())
            state=np.minimum(state,.90*np.array(C.CAPACITY))
            counts["known_steps"]+=1;known+=1
        reset.append({"issued_at":str(issue),"known_steps":known})
    counts.update(complete=counts["unknown_steps"]==0,basis="posthoc_historical_difference_proxy_no_pipeline_buffer",
                  official_reconstruction=False,executed_revenue=None,daily_coverage=reset,
                  scope="historical_proxy_diagnostic_not_official_violation_count",
                  shortfall_unit="m3_hours_of_stock_gap_not_gas_flow",
                  instantaneous_flow_assumption="left_endpoint_hold_over_15min")
    return counts
