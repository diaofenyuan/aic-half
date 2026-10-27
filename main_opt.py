"""煤气资源、因果边界与每日固定 MILP；模型假设始终显式归档。"""
from dataclasses import dataclass, field
from copy import deepcopy
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
import config as C
from main_s import fit_booster
from data_cleaning.pipeline import build_features, horizon_block, process_series, prices_at
from data_cleaning.quality import validate_day, score_resources


@dataclass
class ResourceBundle:
    models: dict = field(default_factory=dict)
    features: list = field(default_factory=list)
    train_end: str = ""
    beta: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def resource_labels(data):
    out = process_series(data)[["blast_13", "blast_45", "known_demand"]].copy()
    for gas, column in zip(("budget_blast", "budget_coke", "budget_converter"), C.USE):
        out[gas] = data.frame[column]
    return out


def calibrate_power(data):
    q = data.frame[list(C.USE)].to_numpy(float)
    y = data.labels.generator_all.to_numpy(float)
    valid = np.isfinite(q).all(axis=1) & np.isfinite(y) & (q[:,0] > 0.) & (y > 0.)
    q, y = q[valid], y[valid]
    if len(y) < 200:
        raise ValueError("合法历史耗气/负荷对不足，无法标定气电映射")
    scale = np.maximum(np.median(q,axis=0), 1.)
    X = q/scale
    split = int(len(y)*.8)
    def fit(a,b):
        w = np.ones(len(b))
        coef = np.zeros(3)
        for _ in range(4):
            coef = lsq_linear(a*np.sqrt(w)[:,None], b*np.sqrt(w), bounds=(0., np.inf)).x
            err = np.abs(a@coef-b)
            threshold = max(float(np.median(err))*1.5, 1.)
            w = np.minimum(1., threshold/np.maximum(err,1e-8))
        return coef
    early = fit(X[:split],y[:split])
    hold = X[split:]@early
    diag = score_resources(hold,y[split:])
    diag["passed"] = bool(diag["wape"] is not None and diag["wape"]<=C.GAS_POWER_ACCEPTANCE["max_wape"]
                          and abs(diag["bias"])<=C.GAS_POWER_ACCEPTANCE["max_abs_bias"])
    diag["acceptance"] = C.GAS_POWER_ACCEPTANCE
    coef = fit(X,y)/scale
    if not np.isfinite(coef).all() or coef[0] <= 0.:
        raise ValueError("经验气电关系不可用")
    return coef.tolist(), {"basis": "training_prefix_inferred_no_intercept", "rows": len(y),
                           "holdout": diag, "coefficients_mw_per_m3h": coef.tolist(),
                           "gas_correlation": np.corrcoef(q.T).tolist(),
                           "per_unit_efficiency_identified": False}


def fit_resources(data, quick=False, progress=None):
    data.require_training(data.index[-1])
    base, labels = build_features(data), resource_labels(data)
    # 96步共享模型只在有限代表提前量训练，h作连续特征；推理仍完整输出96步。
    horizons = [1,4,8,16,24,36,48,60,72,84,96]
    stride = 16 if quick else 4
    pos = np.arange(max(96,len(data.index)-112*96),len(data.index)-96,stride)
    X,names = horizon_block(data,base,pos,horizons)
    ix = np.concatenate([pos+h for h in horizons])
    times = data.index[ix].to_numpy()
    b = ResourceBundle(features=names,train_end=str(data.index[-1]))
    b.diagnostics["training_budget"]={"quick":bool(quick),"threads":C.CPU_THREADS,
                                      "stride":stride,"horizons":horizons}
    b.beta,b.diagnostics["gas_power"] = calibrate_power(data)
    b.diagnostics["group_mapping"] = fit_group_mapping(data)
    for name in labels:
        y = labels[name].to_numpy()[ix]
        if np.isfinite(y).sum() < 100:
            b.diagnostics[name] = {"trained":False,"reason":"INSUFFICIENT_OBSERVATIONS"}
            continue
        model,diag = fit_booster(X,y,names,times,loss="l1",rounds=50 if quick else 240)
        b.models[name] = model
        b.diagnostics[name] = diag
        if progress:
            progress(f"resource {name}，{diag['rows']}行，{diag['rounds']}轮")
    return b


def closed_residual(data, issue):
    # 必须先截断，再访问下一柜位；残差在闭合端点而非区间起点可见。
    f = data.frame.loc[:issue]
    s = process_series(data.prefix(issue))
    delta = f[list(C.HOLDERS)].diff().sum(axis=1,min_count=2)/C.DT
    previous_net = (s.blast_13+s.blast_45-s.known_demand-f[C.USE[0]]).shift(1)
    r = previous_net-delta
    return r.where(np.isfinite(r))


def residual_profile(data, issue, times):
    r = closed_residual(data,issue).loc[pd.Timestamp(issue)-pd.Timedelta(days=28):]
    valid = r.dropna()
    if len(valid) < 24:
        return np.full(len(times),np.nan), {"closed_samples":len(valid),"basis":"UNSUPPORTED"}
    # 残差按区间起始时段归组，闭合时间仅决定什么时候可以使用该记录。
    begin = valid.index-pd.Timedelta(minutes=15)
    profile = valid.groupby(begin.hour*4+begin.minute//15).median()
    slot = pd.DatetimeIndex(times).hour*4+pd.DatetimeIndex(times).minute//15
    values = profile.reindex(slot).fillna(float(valid.median())).to_numpy()
    return values, {"closed_samples":len(valid),"slot_samples":valid.groupby(begin.hour*4+begin.minute//15).size().to_dict(),
                    "basis":"causal_28day_slot_median_inferred","last_closed_at":str(valid.index[-1])}


def forecast_resources(bundle,data,issue,base=None):
    issue = pd.Timestamp(issue)
    if pd.Timestamp(bundle.train_end) >= issue:
        raise ValueError("资源模型训练截止越过发布时间")
    pos = data.index.get_indexer([issue])
    if pos[0] < 0:
        raise ValueError("缺少发布时刻")
    base = build_features(data) if base is None else base
    X,names = horizon_block(data,base,pos,range(1,97))
    if names != bundle.features:
        raise ValueError("资源模型特征顺序不兼容")
    pred = {name:np.maximum(model.predict(X,num_threads=C.TREE["num_threads"]),0.)
            for name,model in bundle.models.items()}
    missing = [name for name in ("budget_blast","budget_coke","budget_converter") if name not in pred]
    if missing:
        raise ValueError("资源代理模型缺失："+",".join(missing))
    times = pd.date_range(issue+pd.Timedelta(minutes=15),periods=96,freq="15min")
    reserve,diag = residual_profile(data,issue,times)
    net = all(x in pred for x in ("blast_13","blast_45","known_demand")) and np.isfinite(reserve).all()
    if net:
        g = np.column_stack((pred["blast_13"],pred["blast_45"]))
        total = g.sum(axis=1)
        if (total <= 0.).any():
            net = False
    if net:
        split = g/total[:,None]
        available = g-split*(pred["known_demand"]+reserve)[:,None]
        mode = "net_inferred"
    else:
        # 仅作非认证代理调度；缺测发生量不被补写成真实煤气量。
        hist = process_series(data.prefix(issue)).tail(28*96)
        ratio = (hist.blast_13/(hist.blast_13+hist.blast_45)).dropna()
        ratio = float(ratio.median()) if len(ratio) else None
        if ratio is None:
            known = data.frame.loc[:issue].tail(28*96)
            a = known[["gas_blast_furnace_1","gas_blast_furnace_2"]].sum(axis=1,min_count=2)
            b = known[["gas_blast_furnace_4","gas_blast_furnace_5"]].sum(axis=1,min_count=2)
            inferred_ratio = (a/(a+b)).dropna()
            ratio = float(inferred_ratio.median()) if len(inferred_ratio) else .5
        split = np.tile([ratio,1.-ratio],(96,1))
        available = split*pred["budget_blast"][:,None]
        mode = "historical_consumption_proxy"
    optional = np.column_stack((pred["budget_coke"],pred["budget_converter"]))
    return {"issued_at":str(issue),"times":times,"available":available,"split":split,
            "optional_caps":optional,"budget_blast":pred["budget_blast"],"resource_mode":mode,
            "reserve":diag,"components":pred,"max_observed_at":str(issue)}


def aggregate_to_units(first,total):
    if not np.isfinite([first,total]).all() or not (0.<=first<=200.) or not (first<=total<=first+240.):
        raise ValueError("聚合初态超出容量或关系边界")
    values = np.zeros(6)
    for ids,amount in ((np.arange(4),first),(np.arange(4,6),total-first)):
        cap = np.array(C.UNITS)[ids]
        found = False
        for count in range(1,len(ids)+1):
            if .6*cap[:count].sum() <= amount <= cap[:count].sum():
                values[ids[:count]] = amount/count
                found = True
                break
        if not found and amount>0.:
            count = max(1,int(np.ceil(amount/cap[0])))
            values[ids[:count]] = amount/count
    return values


def initial_power(data,issue,bundle,previous=None):
    if previous is not None:
        return np.asarray(previous,float),{"basis":"previous_published_command","actual_unit_states_observed":False}
    row = data.frame.loc[issue]
    agg = row[["load_"+t for t in C.TARGETS]].to_numpy(float)
    if np.isfinite(agg).all():
        return aggregate_to_units(*agg),{"basis":"observed_aggregate_inferred_unit_decomposition"}
    gas = row[list(C.USE)].to_numpy(float)
    if not np.isfinite(gas).all():
        raise ValueError("首日缺少有效负荷和同刻耗气，不能假设全停初态")
    total = float(gas@np.array(bundle.beta))
    if total > 440. or total < 0.:
        raise ValueError("经验初态超过容量")
    past = data.frame.loc[:issue,["load_"+t for t in C.TARGETS]].tail(28*96)
    ratios = (past.iloc[:,0]/past.iloc[:,1]).replace([np.inf,-np.inf],np.nan).dropna()
    if not len(ratios):
        raise ValueError("缺少合法历史分组关系")
    first = np.clip(total*float(ratios.median()),max(0.,total-240.),min(200.,total))
    return aggregate_to_units(first,total),{"basis":"same_timestamp_gas_and_historical_group_share_inferred",
                                          "aggregate_mw":[float(first),total]}


class LinearRows:
    def __init__(self,n):
        self.n=n
        self.i,self.j,self.v,self.lb,self.ub=[],[],[],[],[]
    def add(self,terms,low=-np.inf,high=np.inf):
        k=len(self.lb)
        for index,value in terms:
            if value:
                self.i.append(k);self.j.append(index);self.v.append(value)
        self.lb.append(low);self.ub.append(high)
    def constraint(self):
        a=coo_matrix((self.v,(self.i,self.j)),shape=(len(self.lb),self.n)).tocsc()
        return LinearConstraint(a,np.array(self.lb),np.array(self.ub))


def solve_milp(inputs,initial,beta,stock0,prices,inventory=True,export_shortfall=False):
    if inventory and export_shortfall:
        raise ValueError("导出缺气松弛不得进入正式库存模型")
    n=96
    power=np.arange(6*n).reshape(n,6)
    on=np.arange(6*n,12*n).reshape(n,6)
    gas=np.arange(12*n,15*n).reshape(n,3)
    stock=np.arange(15*n,15*n+2*(n+1)).reshape(n+1,2)
    switch=np.arange(15*n+2*(n+1),21*n+2*(n+1)).reshape(n,6)
    slack=np.arange(int(switch.max()+1),int(switch.max()+1)+n) if export_shortfall else None
    size=int(slack[-1]+1) if export_shortfall else int(switch.max()+1)
    c=np.zeros(size);low=np.zeros(size);high=np.full(size,np.inf);integer=np.zeros(size)
    cap=np.array(C.UNITS)
    high[power]=cap
    high[on]=1.;integer[on]=1
    high[switch]=1.
    high[gas[:,1:]]=inputs["optional_caps"]
    high[gas[:,0]]=np.maximum(inputs["available"].sum(axis=1),0.)+sum(C.CAPACITY)/C.DT if inventory else inputs["budget_blast"]
    if export_shortfall:
        high[gas[:,0]]=sum(C.UNITS)/beta[0]
        # 先压低维持爬坡所必需的供气超额，避免整日照抄高负荷；超额仍是约束失败。
        c[slack]=max(float(np.max(prices)),.01)*1000.*C.DT*beta[0]*100.
    c[power]=-np.asarray(prices)[:,None]*1000.*C.DT
    c[gas]=1e-6
    c[switch]=C.POLICY["switch_penalty_cny"]
    if inventory:
        low[stock]=np.array(C.CAPACITY)*.20
        high[stock]=np.array(C.CAPACITY)*.85
        low[stock[0]]=stock0;high[stock[0]]=stock0
    else:
        high[stock]=0.
    rows=LinearRows(size)
    for k in range(n):
        if export_shortfall:
            rows.add([(gas[k,0],1.),(slack[k],-1.)],high=inputs["budget_blast"][k])
        for i in range(6):
            rows.add([(power[k,i],1.),(on[k,i],-cap[i])],high=0.)
            rows.add([(power[k,i],1.),(on[k,i],-.6*cap[i])],low=0.)
            prev_on=float(initial[i]>=.6*cap[i]-.001) if k==0 else None
            for sign in (1.,-1.):
                terms=[(power[k,i],sign),(on[k,i],cap[i])]
                bound=cap[i]*.01*15+2*cap[i]
                if k:
                    terms.extend([(power[k-1,i],-sign),(on[k-1,i],cap[i])])
                else:
                    bound+=sign*initial[i]-cap[i]*prev_on
                rows.add(terms,high=bound)
                terms=[(on[k,i],sign),(switch[k,i],-1.)]
                if k:
                    terms.append((on[k-1,i],-sign))
                    rows.add(terms,high=0.)
                else:
                    rows.add(terms,high=sign*prev_on)
        for ids,limit in ((range(4),30.),(range(6),60.)):
            terms=[(power[k,i],1.) for i in ids]
            if k:
                terms += [(power[k-1,i],-1.) for i in ids]
                baseline=0.
            else:
                baseline=float(initial[list(ids)].sum())
            rows.add(terms,low=baseline-limit,high=baseline+limit)
        rows.add([(power[k,i],1.) for i in range(6)]+[(gas[k,j],-beta[j]) for j in range(3)],low=0.,high=0.)
        if inventory:
            for g in range(2):
                rows.add([(stock[k+1,g],1.),(stock[k,g],-1.),
                          (gas[k,0],C.DT*inputs["split"][k,g])],
                         low=C.DT*inputs["available"][k,g],high=C.DT*inputs["available"][k,g])
    tick=time.perf_counter()
    # SciPy将额外选项原样传入HiGHS；只屏蔽这条已核实选项的转发提示。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="Unrecognized options detected:.*threads.*")
        result=milp(c,integrality=integer,bounds=Bounds(low,high),constraints=rows.constraint(),
                    options={"time_limit":C.POLICY["solver_seconds"],"mip_rel_gap":C.POLICY["mip_gap"],
                             "threads":C.CPU_THREADS})
    gap = getattr(result,"mip_gap",None)
    meta={"status_code":int(result.status),"message":str(result.message),
          "seconds":round(time.perf_counter()-tick,3),"mip_gap":None if gap is None else float(gap),
          "inventory_constraints":inventory,"export_resource_shortfall_allowed":export_shortfall}
    if result.x is None:
        return None,meta
    meta["status"]="OPTIMAL" if result.status==0 else "FEASIBLE_WITH_GAP"
    x=result.x
    return {"power":x[power],"gas":x[gas],"stocks":x[stock] if inventory else np.full((97,2),np.nan),
            "vent":np.zeros((96,2)),
            "export_shortfall_m3h":x[slack] if export_shortfall else np.zeros(96)},meta


def fallback_arrays(inputs,previous,beta):
    """最终导出分支保留真实失败；不把不确定柜位补成中位库存。"""
    p=np.tile(previous,(96,1))
    q=np.zeros((96,3))
    mix=np.column_stack((inputs["budget_blast"],inputs["optional_caps"]))
    for k in range(96):
        energy=float(mix[k]@beta)
        if energy>0.:
            q[k]=mix[k]*(p[k].sum()/energy)
        elif beta[0]>0.:
            q[k,0]=p[k].sum()/beta[0]
    return {"power":p,"gas":q,"stocks":np.full((97,2),np.nan),"vent":np.full((96,2),np.nan)}


def _replay_stocks(start,inputs,gas):
    if not np.isfinite(start).all():
        return np.full((97,2),np.nan),np.full((96,2),np.nan)
    stocks=[np.asarray(start,float)]
    vents=[]
    for k in range(96):
        raw=stocks[-1]+C.DT*(inputs["available"][k]-inputs["split"][k]*gas[k,0])
        vent=np.maximum(raw-.90*np.array(C.CAPACITY),0.)/C.DT
        stocks.append(raw-C.DT*vent);vents.append(vent)
    return np.array(stocks),np.array(vents)


def optimize_daily(bundle,data,origins,quick=False,progress=None):
    dates=pd.DatetimeIndex(origins).normalize().unique()
    if quick:
        dates=dates[:2]
    base=build_features(data)
    beta=np.asarray(bundle.beta)
    records,days=[],[]
    previous,previous_gas,previous_state=None,None,None
    for date in dates:
        issue=date+pd.Timedelta(hours=7)
        inputs=forecast_resources(bundle,data,issue,base)
        price=prices_at(data,inputs["times"])
        warnings=[]
        try:
            initial,initial_meta=initial_power(data,issue,bundle,previous)
        except ValueError as exc:
            # 首日无可辨识初态时仅作完整导出，并把未知初态作为验收失败。
            initial=np.zeros(6)
            initial_meta={"basis":"UNVERIFIED_EXPORT_STATE","reason":str(exc)}
            warnings.append(str(exc))
        row=data.frame.loc[issue]
        anchor=row[list(C.HOLDERS)].to_numpy(float)
        anchor_ok=bool(np.isfinite(anchor).all())
        q0=row[list(C.USE)].to_numpy(float) if previous_gas is None else previous_gas.copy()
        bridge_available=inputs["available"][0].copy()
        bridge_split=inputs["split"][0].copy()
        current=process_series(data.prefix(issue)).iloc[-1]
        if inputs["resource_mode"]=="net_inferred" and np.isfinite(current[["blast_13","blast_45","known_demand"]]).all():
            reserve,_=residual_profile(data,issue,[issue])
            g=current[["blast_13","blast_45"]].to_numpy(float)
            if g.sum()>0. and np.isfinite(reserve[0]):
                bridge_split=g/g.sum()
                bridge_available=g-bridge_split*(float(current.known_demand)+reserve[0])
        if np.isfinite(q0).all() and anchor_ok:
            raw=anchor+C.DT*(bridge_available-bridge_split*q0[0])
            bridge_vent=np.maximum(raw-.9*np.array(C.CAPACITY),0.)
            bridge=raw-bridge_vent
        else:
            bridge=np.full(2,np.nan)
            bridge_vent=np.full(2,np.nan)
            warnings.append("桥接缺少有效柜位或此前生效耗气")
        inventory=bool(np.isfinite(bridge).all() and inputs["resource_mode"]=="net_inferred")
        solution,solver=solve_milp(inputs,initial,beta,bridge,price,inventory=inventory)
        fallback=not inventory or solution is None or bool(warnings)
        if solution is None and inventory:
            warnings.append("库存约束模型无可行解，尝试保留机组与资源约束的导出分支")
            solution,secondary=solve_milp(inputs,initial,beta,bridge,price,inventory=False)
            solver["export_solver"]=secondary
        if solution is None:
            # 守住机组与爬坡，显式最小化资源冲突；不能把松弛解标为正常可行计划。
            solution,last_solver=solve_milp(inputs,initial,beta,bridge,price,inventory=False,export_shortfall=True)
            solver["shortfall_export_solver"]=last_solver
            if solution is not None:
                warnings.append("仅导出：首步资源与爬坡冲突，已最小化显式高炉耗气超额")
        if solution is None:
            solution=fallback_arrays(inputs,initial,beta)
            warnings.append("无可行导出解，采用明确标记的因果保持指令")
        if fallback:
            solution["stocks"],solution["vent"]=_replay_stocks(bridge,inputs,solution["gas"])
        day=dict(inputs,**solution,previous_power=initial,anchor=anchor,anchor_observed=anchor_ok,
                 bridge_stock=bridge,beta=beta,solver=solver,initial_state=initial_meta,
                 gas_power_validation=bundle.diagnostics["gas_power"]["holdout"],
                 bridge={"start":str(issue),"end":str(inputs["times"][0]),"gas_m3h":q0,
                         "available_m3h":bridge_available,"split":bridge_split,"vent_m3":bridge_vent},
                 state_reset_m3=anchor-previous_state if previous_state is not None else None,
                 warnings=warnings,fallback_used=fallback)
        day["validation"]=validate_day(day)
        mapping=bundle.diagnostics.get("group_mapping")
        if mapping is None:
            raise ValueError("归档缺少新版分组标定，不能静默迁移旧模型")
        if mapping["trusted"]:
            refined=refine_group_day(day,mapping,smooth=C.GROUP_DISPATCH["smooth_penalty"])
            if refined is not None:
                day=refined
                solution["power"]=day["power"]
            else:
                day["refinement"]={"applied":False,"reason":"fixed_trajectory_infeasible"}
        else:
            day["refinement"]={"applied":False,"reason":"mapping_holdout_gate_failed"}
        day["plan_revenue_cny"]=float(np.sum(price*solution["power"].sum(axis=1))*250.)
        day["planned_vent_m3"]=float(np.nansum(solution["vent"])*C.DT)
        day["export_resource_shortfall_m3"]=float(np.sum(solution.get("export_shortfall_m3h",0.))*C.DT)
        for k,when in enumerate(inputs["times"]):
            records.append([when,*solution["gas"][k],solution["power"][k,:4].sum(),solution["power"][k].sum()])
        days.append(day)
        previous=solution["power"][-1].copy()
        previous_gas=solution["gas"][-1].copy()
        # 下一日07:00重锚发生在本计划末条指令生效时，而非末条执行完之后。
        previous_state=solution["stocks"][-2].copy()
        if progress:
            progress(f"dispatch {issue.date()}：{solver.get('status',solver['message'])}，"
                     f"核验={'通过' if day['validation']['passed'] else ','.join(day['validation']['problems'])}")
    columns=["datetime"]+["opt_generator_use_"+g for g in C.GASES]+["opt_generator_1","opt_generator_all"]
    schedule=pd.DataFrame(records,columns=columns)
    failures=[{"issued_at":d["issued_at"],"problems":d["validation"]["problems"]} for d in days if not d["validation"]["passed"]]
    summary={"publication_time":"07:00","intraday_refresh":False,"daily_publications":len(days),
             "planned_steps":len(schedule),"constraint_validation_passed":not failures,
             "failed_publications":failures,"fallback_publications":sum(d["fallback_used"] for d in days),
             "group_refined_publications":sum(d.get("refinement",{}).get("applied",False) for d in days),
             "plan_revenue_cny":sum(d["plan_revenue_cny"] for d in days),
             "planned_vent_m3":sum(d["planned_vent_m3"] for d in days),
             "physical_model_basis":"adopted_project_policy_inferred_resources",
             "field_safety_certified":False,"replay_complete":False,
             "execution_revenue":None,"note":"计划轨迹不冒充官方柜位重建或真实执行收益"}
    return schedule,days,summary


def fit_group_mapping(data):
    data.require_training(data.index[-1])
    data=data.prefix(data.index[-1])
    start=data.index[-1]-pd.Timedelta(days=C.GROUP_DISPATCH["history_days"])
    data.frame=data.frame.loc[start:].copy()
    data.labels=data.labels.reindex(data.frame.index)
    frame=data.frame[list(C.USE)].to_numpy(float)
    y=data.labels.generator_1.to_numpy(float)
    valid=np.isfinite(frame).all(axis=1)&np.isfinite(y)&(frame[:,0]>0.)
    q,y=frame[valid],y[valid]
    times=data.index[valid]
    if len(y)<400:
        return {"trusted":False,"train_end":str(data.index[-1]),"rows":len(y),
                "source":"historical_inferred","reason":"INSUFFICIENT_GROUP_OBSERVATIONS"}
    scale=np.maximum(np.median(q,axis=0),1.)
    X=q/scale
    split=times.max()-pd.Timedelta(days=C.GROUP_DISPATCH["holdout_days"])
    before=times<split
    after=~before
    def fit(a,b):
        weight=np.ones(len(b))
        coef=np.zeros(3)
        for _ in range(5):
            coef=lsq_linear(a*np.sqrt(weight)[:,None],b*np.sqrt(weight),bounds=(0.,np.inf)).x
            err=np.abs(a@coef-b)
            cutoff=max(float(np.median(err))*1.5,1.)
            weight=np.minimum(1.,cutoff/np.maximum(err,1e-9))
        return coef
    if before.sum()<200 or after.sum()<100:
        return {"trusted":False,"train_end":str(data.index[-1]),"rows":len(y),
                "source":"historical_inferred","reason":"INSUFFICIENT_GROUP_HOLDOUT"}
    test_coef=fit(X[before],y[before])
    pred=X[after]@test_coef
    score=score_resources(pred,y[after])
    baseline=float(np.median(y[before]))
    ref=score_resources(np.full(after.sum(),baseline),y[after])
    beta=fit(X,y)/scale
    trusted=score["wape"] is not None and score["wape"]<=C.GROUP_DISPATCH["max_wape"] and abs(score["bias"])<=C.GROUP_DISPATCH["max_abs_bias"]
    return {"trusted":bool(trusted),"acceptance":C.GROUP_DISPATCH,"coefficients_mw_per_m3h":beta.tolist(),"train_end":str(data.index[-1]),
            "source":"causal_training_prefix_inferred_group_mapping","rows":len(y),
            "holdout_start":str(split),"holdout":score,"median_reference":ref,
            "official_composition_formula":False}


def refine_group_day(day,mapping,previous=None,smooth=0.,margin=.001):
    if pd.Timestamp(mapping["train_end"])>=pd.Timestamp(day["issued_at"]):
        raise ValueError("分组标定越过发布时刻")
    power=np.arange(576).reshape(96,6)
    on=np.arange(576,1152).reshape(96,6)
    error=np.arange(1152,1248)
    change=np.arange(1248,1344)
    size=1344
    cost=np.zeros(size);lower=np.zeros(size);upper=np.full(size,np.inf);integer=np.zeros(size)
    cap=np.array(C.UNITS)
    upper[power]=cap;upper[on]=1.;integer[on]=1.
    # 冻结末条逐台状态，使本日再分配不改变次日发布的可达边界。
    lower[power[-1]]=np.asarray(day["power"])[-1]
    upper[power[-1]]=np.asarray(day["power"])[-1]
    cost[error]=C.GROUP_DISPATCH["mapping_penalty"];cost[change]=smooth
    q=np.asarray(day["gas"])
    target=q@np.array(mapping["coefficients_mw_per_m3h"])
    previous=np.asarray(day["previous_power"] if previous is None else previous)
    total=np.asarray(day["power"]).sum(axis=1)
    rows=LinearRows(size)
    for k in range(96):
        rows.add([(power[k,i],1.) for i in range(6)],low=total[k],high=total[k])
        for i in range(6):
            rows.add([(power[k,i],1.),(on[k,i],-cap[i])],high=0.)
            rows.add([(power[k,i],1.),(on[k,i],-.6*cap[i])],low=0.)
            for sign in (1.,-1.):
                terms=[(power[k,i],sign),(on[k,i],cap[i])]
                bound=.15*cap[i]+2*cap[i]
                if k:
                    terms.extend([(power[k-1,i],-sign),(on[k-1,i],cap[i])])
                else:
                    bound+=sign*previous[i]-cap[i]*float(previous[i]>=.6*cap[i]-.001)
                rows.add(terms,high=bound)
        terms=[(power[k,i],1.) for i in range(4)]
        for sign in (1.,-1.):
            rows.add([(j,v*sign) for j,v in terms]+[(error[k],-1.)],high=sign*target[k])
        ramp=terms.copy()
        if k:
            ramp.extend((power[k-1,i],-1.) for i in range(4))
            prior=0.
        else:
            prior=float(previous[:4].sum())
        # 内部留极小数值裕度，不依赖放大验收容差跨越30MW边界。
        rows.add(ramp,low=prior-(30.-margin),high=prior+(30.-margin))
        for sign in (1.,-1.):
            rows.add([(j,v*sign) for j,v in ramp]+[(change[k],-1.)],high=sign*prior)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="Unrecognized options detected:.*threads.*")
        result=milp(cost,integrality=integer,bounds=Bounds(lower,upper),constraints=rows.constraint(),
                    options={"time_limit":10.,"mip_rel_gap":.005,"threads":C.CPU_THREADS})
    if result.x is None:
        if margin>0.:
            return refine_group_day(day,mapping,previous,smooth,margin=0.)
        return None
    out=deepcopy(day)
    out["power"]=result.x[power]
    out["power"][-1]=np.asarray(day["power"])[-1]
    out["previous_power"]=previous.copy()
    out["group_mapping"]=mapping
    out["refinement"]={"applied":True,"baseline_total_power":total.tolist(),"baseline_terminal_unit_power":np.asarray(day["power"])[-1].tolist(),"source":"fixed_gas_and_total_power_MILP","smooth_penalty":smooth,
                       "solver_status":int(result.status),"total_power_max_difference":
                       float(np.max(np.abs(out["power"].sum(axis=1)-total))),
                       "group_mae_mw":float(np.mean(np.abs(out["power"][:,:4].sum(axis=1)-target)))}
    out["validation"]=validate_day(out)
    required=("unit_capacity","unit_minimum","aggregate_ramp","running_unit_ramp","gas_power",
              "refinement_fixed_total_power","refinement_fixed_terminal_state","group_mapping_causality")
    if not all(out["validation"]["checks"].get(k,False) for k in required):
        return None
    return out
