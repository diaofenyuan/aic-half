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


def truth_cube(data, origins, horizons):
    pos = data.index.get_indexer(pd.DatetimeIndex(origins))
    if (pos < 0).any():
        raise ValueError("起报点超出输入网格")
    ix = pos[:, None] + np.array(horizons)[None, :]
    y = np.full((*ix.shape, 2), np.nan)
    ok = ix < len(data.index)
    y[ok] = data.labels.to_numpy()[ix[ok]]
    return y


def score_load(pred, truth):
    out, scores = {}, []
    for i, name in enumerate(C.TARGETS):
        p, y = np.asarray(pred)[..., i], np.asarray(truth)[..., i]
        finite = np.isfinite(y) & np.isfinite(p)
        valid, zero = finite & (y > 0.), finite & (y == 0.)
        score = float(100.*(1.-np.mean(np.abs(p[valid]-y[valid])/y[valid]))) if valid.any() else None
        out[name] = {"score": score, "nonzero_cells": int(valid.sum()), "zero_cells": int(zero.sum()),
                     "zero_mae": float(np.mean(np.abs(p[zero]))) if zero.any() else None,
                     "bias_mw": float(np.mean(p[finite]-y[finite])) if finite.any() else None}
        if score is not None:
            scores.append(score)
    out["macro_score"] = float(np.mean(scores)) if len(scores) == 2 else None
    return out


def score_horizons(pred, truth):
    h = pred.shape[1]
    return {f"{a+1}-{min(b,h)}": score_load(pred[:, a:min(b,h)], truth[:, a:min(b,h)])
            for a, b in ((0, 8), (8, 24), (24, 48), (48, 96)) if a < h}


def score_resources(pred, truth):
    p, y = np.asarray(pred), np.asarray(truth)
    valid = np.isfinite(p) & np.isfinite(y)
    if not valid.any():
        return {"wape": None, "cells": 0}
    error = p[valid]-y[valid]
    denom = np.abs(y[valid]).sum()
    return {"wape": float(np.abs(error).sum()/denom) if denom > 0 else None,
            "bias": float(error.sum()/denom) if denom > 0 else None, "cells": int(valid.sum()),
            "zero_false_volume": float(np.abs(p[valid & (y == 0)]).sum()*C.DT)}


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
    checks["initial_state_known"] = day.get("initial_state",{}).get("basis") not in (None,"UNVERIFIED_EXPORT_STATE")
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
    optional = np.asarray(day.get("optional_caps", np.full((96, 2), np.nan)), float)
    checks["optional_gas_caps"] = bool(optional.shape == (96, 2) and np.isfinite(optional).all() and
                                        (q[:, 1:] <= optional+gastol).all())
    checks["export_resource_shortfall"] = bool(np.max(np.asarray(day.get("export_shortfall_m3h",[0.]),float))<=gastol)
    checks["publication_causality"] = pd.Timestamp(day.get("max_observed_at", "2100-01-01")) <= pd.Timestamp(day["issued_at"])
    if "times" in day:
        times=pd.DatetimeIndex(day["times"])
        expected=pd.date_range(pd.Timestamp(day["issued_at"])+pd.Timedelta(minutes=15),periods=96,freq="15min")
        checks["effective_grid"]=bool(np.array_equal(times.to_numpy(),expected.to_numpy()))
    fit=day.get("gas_power_validation")
    if fit is not None:
        checks["gas_power_calibration"]=bool(fit["passed"])
    refinement=day.get("refinement",{})
    if refinement.get("applied"):
        frozen_total=np.asarray(refinement["baseline_total_power"],float)
        frozen_end=np.asarray(refinement["baseline_terminal_unit_power"],float)
        checks["refinement_fixed_total_power"]=bool(np.max(np.abs(p.sum(axis=1)-frozen_total))<=1e-5)
        checks["refinement_fixed_terminal_state"]=bool(np.max(np.abs(p[-1]-frozen_end))<=1e-5)
        checks["group_mapping_causality"]=pd.Timestamp(day["group_mapping"]["train_end"])<pd.Timestamp(day["issued_at"])
    details["max_aggregate_change_mw"] = delta.max(axis=0).tolist()
    return {"passed": bool(all(checks.values())), "checks": checks,
            "problems": [k for k, v in checks.items() if not v], "details": details}


def validate_day(day):
    try:
        return _validate_day(day)
    except (KeyError,ValueError,TypeError,IndexError) as exc:
        return {"passed":False,"checks":{"trajectory_schema":False},
                "problems":["INVALID_TRAJECTORY:"+str(exc)]}


def audit_resource_forecasts(days, truth):
    result={}
    names=sorted({name for day in days for name in day["components"]})
    for name in names:
        predicted,observed,daily_error=[],[],[]
        for day in days:
            if name not in day["components"]:
                continue
            p=np.asarray(day["components"][name],float)
            y=truth.reindex(pd.DatetimeIndex(day["times"]))[name].to_numpy(float)
            predicted.extend(p);observed.extend(y)
            if np.isfinite(y).all():
                daily_error.append(float(np.sum(p-y)*C.DT))
        result[name]=dict(score_resources(predicted,observed),
                          cumulative_error_m3=daily_error,
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
    return {"passed": all(x["passed"] for x in checks.values()), "files": checks}


def historical_replay(data,days):
    """独立事后差额代理；未来柜位/基准耗气仅在此评分器中读取。"""
    counts={"known_steps":0,"unknown_steps":0,"holder_low_steps":0,"holder_high_steps_before_release":0}
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
            counts["holder_low_steps"]+=int((state < .15*np.array(C.CAPACITY)-2.).any())
            counts["holder_high_steps_before_release"]+=int((state > .90*np.array(C.CAPACITY)+2.).any())
            state=np.minimum(state,.90*np.array(C.CAPACITY))
            counts["known_steps"]+=1;known+=1
        reset.append({"issued_at":str(issue),"known_steps":known})
    counts.update(complete=counts["unknown_steps"]==0,basis="posthoc_historical_difference_proxy_no_pipeline_buffer",
                  official_reconstruction=False,executed_revenue=None,daily_coverage=reset)
    return counts
