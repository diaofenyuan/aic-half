"""煤气资源、因果边界与每日固定 MILP；模型假设始终显式归档。"""
from dataclasses import dataclass, field
from copy import deepcopy
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import lsq_linear, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull, QhullError
import config as C
from main_s import fit_booster
from data_cleaning.pipeline import build_features, horizon_block, process_series, prices_at
from data_cleaning.quality import validate_day, score_resources, validate_plan_sequence


@dataclass
class ResourceBundle:
    models: dict = field(default_factory=dict)
    features: list = field(default_factory=list)
    train_end: str = ""
    beta: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    process_model: object = None
    process_meta: dict = field(default_factory=dict)
    bootstrap: object = None


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
    b.diagnostics["dispatch_control"] = fit_dispatch_calibration(data, b.beta)
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
    b.process_model,b.process_meta=fit_process_resource(data)
    if progress:
        meta=b.process_meta
        progress(f"resource shared_process：{meta['training']['rows']}行，{meta['training']['rounds']}轮"
                 if meta['available'] else 'resource shared_process：'+meta['reason'])
    return b



def _process_feature_blocks(data,scales):
    result=[];f=data.frame;idx=f.index;positions=np.arange(len(idx))
    for unit in range(1,6):
        gas=f['gas_blast_furnace_'+str(unit)]/scales['gas'][unit-1]
        heater=f['gas_air_heater_'+str(unit)]/scales['heater'][unit-1]
        values={'unit':np.full(len(idx),unit),'slot':idx.hour*4+idx.minute//15,
            'weekday':idx.dayofweek,'month':idx.month,'gas_now':gas,'heater_now':heater}
        for prefix,series in [('gas',gas),('heater',heater)]:
            for lag in (1,4,16,96):values[prefix+'_lag'+str(lag)]=series.shift(lag)
            for window in (4,16,96):
                values[prefix+'_med'+str(window)]=series.rolling(window,min_periods=1).median()
                values[prefix+'_valid'+str(window)]=series.notna().rolling(window,min_periods=1).mean()
            values[prefix+'_std16']=series.rolling(16,min_periods=2).std()
            last=np.maximum.accumulate(np.where(series.notna(),positions,-1))
            values[prefix+'_age']=np.where(last>=0,positions-last,np.nan)
        last_nonzero=np.maximum.accumulate(np.where(heater.ne(0.).to_numpy(),positions,-1))
        values['heater_zero_duration']=np.where(heater.eq(0.),positions-last_nonzero,0.)
        values['gas_input_masked']=np.zeros(len(idx))
        block=pd.DataFrame(values,index=idx).astype('float32')
        result.append(block)
    return result


def _process_samples(data,blocks,positions,horizons,training=False):
    positions=np.asarray(positions);pieces=[];labels=[];times=[];names=None
    h=np.asarray(horizons)
    for unit,block in enumerate(blocks,1):
        a=np.tile(block.to_numpy()[positions],(len(h),1))
        origin=np.tile(positions,len(h))
        masked=data.frame['gas_blast_furnace_'+str(unit)].iloc[origin].isna().to_numpy()
        if training:
            masked|=((origin//C.PROCESS_RESOURCE['stride']+unit)%C.PROCESS_RESOURCE['dropout_modulo']==0)
        # 遮挡当前及陈旧的该炉发生量，模型必须使用关联过程量；原始表与标签均保持不变。
        gas_indices=[j for j,name in enumerate(block.columns) if name.startswith('gas_')]
        a[np.ix_(masked,gas_indices)]=np.nan
        a[:,block.columns.get_loc('gas_input_masked')]=masked.astype(float)
        targets=pd.DatetimeIndex(data.index[origin].to_numpy())+pd.to_timedelta(np.repeat(h,len(positions))*15,unit='min')
        extra=np.column_stack((np.repeat(h,len(positions)),targets.hour*4+targets.minute//15,
            targets.dayofweek,prices_at(data,targets)))
        pieces.append(np.column_stack((a,extra)).astype('float32'))
        labels.append(np.concatenate([positions+ahead for ahead in h]))
        times.append(targets.to_numpy())
        names=list(block.columns)+['horizon','target_slot','target_weekday','target_price']
    return np.concatenate(pieces),names,labels,np.concatenate(times)


def fit_dispatch_calibration(data, beta):
    """控制标定随冻结模型归档；窗口只含训练前缀，不重新拟合预测树。"""
    data.require_training(data.index[-1])
    end=data.index[-1]
    f=data.frame.loc[end-pd.Timedelta(days=C.DISPATCH_CONTROL['history_days']):,list(C.USE)]
    q=f.to_numpy(float)
    q=q[np.isfinite(q).all(axis=1)&(q[:,0]>0.)]
    support={'available':False,'rows':len(q),'train_end':str(end),
             'source':'historical_joint_energy_share_support','official_composition_formula':False}
    if len(q)>=C.DISPATCH_CONTROL['min_support_rows']:
        energy=q*np.asarray(beta)
        share=energy[:,1:]/energy.sum(axis=1)[:,None]
        try:
            hull=ConvexHull(share)
            support.update(available=True,equations=hull.equations.tolist())
        except QhullError:
            support['reason']='INSUFFICIENT_TWO_DIMENSIONAL_GAS_VARIATION'
    scenarios=[]
    for days in C.DISPATCH_CONTROL['sensitivity_days']:
        span=data.prefix(end)
        span.frame=span.frame.loc[end-pd.Timedelta(days=days):].copy()
        span.labels=span.labels.reindex(span.frame.index)
        q=span.frame[list(C.USE)].to_numpy()
        y=span.labels.to_numpy()
        y=np.column_stack((y[:,0],y[:,1]-y[:,0]))
        ok=np.isfinite(q).all(axis=1)&np.isfinite(y).all(axis=1)&(q[:,0]>0.)
        q,y=q[ok],y[ok]
        if len(y)<C.DISPATCH_CONTROL['min_support_rows']:
            continue
        scale=np.maximum(np.median(q,axis=0),1.)
        X=q/scale;coef=[]
        for j in range(2):
            w=np.ones(len(y))
            for _ in range(5):
                b=lsq_linear(X*np.sqrt(w)[:,None],y[:,j]*np.sqrt(w),bounds=(0.,np.inf)).x
                error=np.abs(X@b-y[:,j]);threshold=max(float(np.median(error))*1.5,1.)
                w=np.minimum(1.,threshold/np.maximum(error,1e-8))
            coef.append(b/scale)
        scenarios.append({'days':days,'rows':len(y),'train_end':str(end),
                          'coefficients_mw_per_m3h':np.asarray(coef).sum(axis=0).tolist(),
                          'source':'historical_mapping_sensitivity_only','point_mapping_replaced':False})
    return {'version':C.DISPATCH_CONTROL['version'],'recipe':deepcopy(C.DISPATCH_CONTROL),
            'train_end':str(end),'support':support,'mapping_scenarios':scenarios,
            'prediction_models_retrained':False,'probabilistic_coverage_claimed':False}


def fit_process_resource(data):
    data.require_training(data.index[-1]);policy=C.PROCESS_RESOURCE
    meta={'version':policy['version'],'recipe':deepcopy(policy),'train_end':str(data.index[-1]),
          'available':False,'dropout_is_input_only':True,'missing_labels_imputed':False,
          'zero_heater_means_zero_gas':False}
    scales={'gas':[],'heater':[]};counts={}
    for unit in range(1,6):
        for kind,column in [('gas','gas_blast_furnace_'),('heater','gas_air_heater_')]:
            values=data.frame[column+str(unit)].tail(policy['history_days']*96)
            positive=values[values>0.];counts[kind+'_'+str(unit)]=len(positive)
            if len(positive)<policy['min_scale_samples']:
                meta.update(reason='INSUFFICIENT_SHARED_PROCESS_LABEL_SUPPORT',normalization_support=counts)
                return None,meta
            scales[kind].append(float(positive.median()))
    blocks=_process_feature_blocks(data,scales)
    origins=np.arange(max(96,len(data.index)-policy['history_days']*96),len(data.index)-96,policy['stride'])
    X,names,targets,times=_process_samples(data,blocks,origins,policy['horizons'],training=True)
    y=np.concatenate([data.frame['gas_blast_furnace_'+str(unit)].iloc[positions].to_numpy()/scales['gas'][unit-1]
        for unit,positions in enumerate(targets,1)])
    valid=np.isfinite(y)
    params=dict(C.TREE,objective='quantile',alpha=policy['quantile'],num_leaves=policy['leaves'],seed=2026)
    # 缺测发生量估计采用已选下分位，以降低申报无法兑现的资源风险；不声称覆盖概率已校准。
    model=lgb.train(params,lgb.Dataset(X[valid],label=y[valid],feature_name=names),num_boost_round=policy['rounds'])
    diag={'rows':int(valid.sum()),'rounds':policy['rounds'],'loss':'quantile','alpha':policy['quantile'],
        'max_target_time':str(pd.Timestamp(times[valid].max())),'zero_label_rows':int((y[valid]==0.).sum())}
    meta.update(available=True,scales=scales,features=names,training=diag,normalization_support=counts)
    return model,meta


def validate_process_resource(bundle):
    meta=bundle.process_meta
    if (meta.get('version')!=C.PROCESS_RESOURCE['version'] or meta.get('recipe')!=C.PROCESS_RESOURCE or
        not isinstance(meta.get('available'),bool)):
        raise ValueError('资源过程模型契约不兼容；旧归档需显式补建参数，禁止静默迁移或补训')
    process_end=pd.Timestamp(meta.get('train_end'))
    if pd.isna(process_end) or process_end>pd.Timestamp(bundle.train_end):
        raise ValueError('资源过程模型训练截止无效或超过资源归档截止')
    if not meta['available']:
        if bundle.process_model is not None or not meta.get('reason'):
            raise ValueError('资源过程模型不支持状态与归档不一致')
        return
    scales=np.asarray([meta.get('scales',{}).get(k,[]) for k in ('gas','heater')],float)
    if scales.shape!=(2,5) or not np.isfinite(scales).all() or (scales<=0.).any():
        raise ValueError('资源过程模型归一化参数无效')
    if bundle.process_model is None or bundle.process_model.feature_name()!=meta.get('features'):
        raise ValueError('资源过程模型缺失或特征顺序不兼容')
    training=meta.get('training',{})
    label_end=pd.Timestamp(training.get('max_target_time'))
    if (training.get('loss')!='quantile' or training.get('alpha')!=C.PROCESS_RESOURCE['quantile'] or
        pd.isna(label_end) or label_end>process_end):
        raise ValueError('过程模型损失或训练标签时间契约不兼容')


def predict_process_resource(bundle,data,issue):
    validate_process_resource(bundle);meta=bundle.process_meta
    if not meta['available']:raise ValueError('资源过程模型缺少合法训练支持')
    if pd.Timestamp(meta['train_end'])>=pd.Timestamp(issue):raise ValueError('资源过程模型训练越过发布时刻')
    prefix=data.prefix(issue)
    blocks=_process_feature_blocks(prefix,meta['scales'])
    positions=prefix.index.get_indexer([pd.Timestamp(issue)])
    X,names,_,_=_process_samples(prefix,blocks,positions,range(1,97))
    if names!=meta['features']:raise ValueError('资源过程模型特征顺序改变')
    estimates=bundle.process_model.predict(X,num_threads=C.TREE['num_threads']).reshape(5,96).T
    return np.maximum(estimates,0.)*np.asarray(meta['scales']['gas'])[None,:]


def validate_dispatch_calibration(bundle):
    control=bundle.diagnostics.get('dispatch_control',{})
    if (control.get('version')!=C.DISPATCH_CONTROL['version'] or
        control.get('recipe')!=C.DISPATCH_CONTROL or control.get('train_end')!=bundle.train_end):
        raise ValueError('联合调度标定契约不兼容；须显式建立并归档新版控制参数，禁止静默迁移或补训')
    support=control.get('support',{})
    scenarios=control.get('mapping_scenarios')
    if not isinstance(scenarios,list) or support.get('train_end')!=bundle.train_end:
        raise ValueError('联合调度标定缺失或时间来源不一致')
    for item in scenarios:
        beta=np.asarray(item.get('coefficients_mw_per_m3h',[]),float)
        if item.get('train_end')!=bundle.train_end or beta.shape!=(3,) or not np.isfinite(beta).all() or (beta<0.).any():
            raise ValueError('气电敏感性标定参数无效')
    if support.get('available'):
        equations=np.asarray(support.get('equations',[]),float)
        if equations.ndim!=2 or equations.shape[1]!=3 or not len(equations) or not np.isfinite(equations).all():
            raise ValueError('历史耗气联合支持域参数无效')
    return control


def inventory_risk(data, issue):
    # 柜位只为控制风险定软价值，不进入资源预测特征；缺测窗口不插补。
    frame=data.frame.loc[:issue,list(C.HOLDERS)].tail(C.DISPATCH_CONTROL['history_days']*96)
    window=C.DISPATCH_CONTROL['reserve_hours']*4+1
    draw=frame.rolling(window,min_periods=window).max()-frame
    amount=draw.quantile(C.DISPATCH_CONTROL['reserve_quantile']).to_numpy()
    if not np.isfinite(amount).all():
        return None
    cap=np.asarray(C.CAPACITY)
    return {'source':'causal_observed_inventory_drawdown_soft_value',
            'target_m3':np.clip(.15*cap+amount,.2*cap,.85*cap).tolist(),
            'max_observed_at':str(issue),'official_penalty':False}


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


def calibrate_optional_prefix(pred,data,issue):
    """仅以发布前连续实测校准短时耗气水平；耗气缺失时保留原模型包络。"""
    issue=pd.Timestamp(issue);policy=C.OPTIONAL_USE_CALIBRATION;diagnostics={}
    for name in policy['components']:
        column={'budget_coke':C.USE[1],'budget_converter':C.USE[2]}[name]
        index=pd.date_range(end=issue,periods=policy['history_steps'],freq='15min')
        actual=data.frame.loc[:issue,column].reindex(index).to_numpy(float)
        original=np.asarray(pred[name],float).copy()
        detail={'source':policy['source'],'applied':False,'observed_from':str(index[0]),
            'observed_through':str(issue),'observed_values_m3h':actual,
            'original_forecast_m3h':original,'used_for_training':False,
            'future_truth_used':False,'missing_consumption_filled':False}
        if not np.isfinite(actual).all() or (actual<0.).any():
            diagnostics[name]=dict(detail,reason='CURRENT_LEGAL_CONSUMPTION_OBSERVATION_UNAVAILABLE');continue
        current=float(np.median(actual));reference=float(np.median(original[:policy['history_steps']]))
        delta=current-reference
        # 差额按四小时半衰期衰减；不将瞬时耗气扩张成全天真实可用余气。
        decay=np.exp2(-np.arange(1,len(original)+1)/policy['decay_steps'])
        pred[name]=np.maximum(original+delta*decay,0.)
        diagnostics[name]=dict(detail,applied=True,observed_median_m3h=current,
            original_forecast_reference_m3h=reference,initial_delta_m3h=delta,decay_steps=policy['decay_steps'])
    return diagnostics


def forecast_resources(bundle,data,issue,base=None,steps=96):
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
    if not 1<=steps<=163:
        raise ValueError("调度跨度必须在1至163步之间")
    times = pd.date_range(issue+pd.Timedelta(minutes=15),periods=steps,freq="15min")
    tail_diag={"applied":False}
    if steps>96:
        history=resource_labels(data.prefix(issue)).loc[issue-pd.Timedelta(days=C.OPT_TAIL['history_days']):]
        slot=history.index.hour*4+history.index.minute//15
        target=times[96:].hour*4+times[96:].minute//15
        tail_diag={"applied":True,"source":C.OPT_TAIL['source'],"first_horizon":97,
                   "last_horizon":steps,"max_observed_at":str(issue),"components":{}}
        for name,values in pred.items():
            valid=history[name].dropna()
            if valid.empty:
                raise ValueError("尾部画像缺少合法历史："+name)
            profile=history[name].groupby(slot).median()
            tail=profile.reindex(target).fillna(float(valid.median())).to_numpy()
            pred[name]=np.r_[values,tail]
            tail_diag['components'][name]={"history_start":str(valid.index[0]),
                "history_end":str(valid.index[-1]),"samples":len(valid),
                "slot_samples":history[name].groupby(slot).count().to_dict(),
                "profile_m3h":profile.fillna(float(valid.median())).to_dict(),"fallback_median_m3h":float(valid.median())}
    else:
        pred={name:values[:steps] for name,values in pred.items()}
    reserve,diag = residual_profile(data,issue,times)
    net = all(x in pred for x in ("blast_13","blast_45","known_demand")) and np.isfinite(reserve).all()
    if net:
        g = np.column_stack((pred["blast_13"],pred["blast_45"]))
        total = g.sum(axis=1)
        if (total <= 0.).any():
            net = False
    if net:
        process_diag={"applied":False,"reason":"complete_process_observations"}
        missing_furnaces=[i for i in (1,2,3) if not np.isfinite(data.frame.loc[issue,'gas_blast_furnace_'+str(i)])]
        if missing_furnaces and bundle.process_meta.get('available'):
            heaters=[data.frame.loc[issue,'gas_air_heater_'+str(i)] for i in missing_furnaces]
            if np.isfinite(heaters).all():
                inferred=predict_process_resource(bundle,data,issue)
                prior=pred['blast_13'].copy()
                pred['blast_13'][:min(steps,96)]=inferred[:min(steps,96),:3].sum(axis=1)
                g=np.column_stack((pred['blast_13'],pred['blast_45']))
                total=g.sum(axis=1)
                process_diag={"applied":True,"basis":"shared_process_LGBM_inferred", "missing_furnaces":missing_furnaces,
                    "train_end":bundle.process_meta['train_end'],"max_observed_at":str(issue),
                    "per_furnace_forecast_m3h":inferred,"original_blast13_forecast_m3h":prior,
                    "missing_gas_is_zero":False,"raw_process_values_overwritten":False}
            else:
                process_diag={"applied":False,"reason":"missing_related_heater_observations"}
        elif missing_furnaces:
            process_diag={"applied":False,"reason":bundle.process_meta.get('reason','missing_process_model_support')}
        split = g/total[:,None]
        available = g-split*(pred["known_demand"]+reserve)[:,None]
        mode = "net_inferred"
    else:
        process_diag={"applied":False,"reason":"net_resource_basis_unsupported"}
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
        split = np.tile([ratio,1.-ratio],(steps,1))
        available = split*pred["budget_blast"][:,None]
        mode = "historical_consumption_proxy"
    optional_diag=calibrate_optional_prefix(pred,data,issue)
    optional = np.column_stack((pred["budget_coke"],pred["budget_converter"]))
    return {"issued_at":str(issue),"times":times,"available":available,"split":split,
            "optional_caps":optional,"budget_blast":pred["budget_blast"],"resource_mode":mode,
            "reserve":diag,"components":pred,"max_observed_at":str(issue),"process_resource":process_diag,
            "optional_consumption_calibration":optional_diag,"tail_resource":tail_diag,
            "resource_train_end":bundle.train_end}


def bridge_resources(bundle,data,issue,inputs):
    issue=pd.Timestamp(issue);row=data.frame.loc[issue]
    available=np.asarray(inputs['available'][0],float).copy();split=np.asarray(inputs['split'][0],float).copy()
    meta={'basis':'first_forecast_fallback','max_observed_at':str(issue),
        'raw_input_overwritten':False,'missing_fuel_zero_filled':False}
    current=process_series(data.prefix(issue)).iloc[-1]
    reserve,_=residual_profile(data,issue,[issue])
    if inputs['resource_mode']!='net_inferred' or not np.isfinite(current.known_demand) or not np.isfinite(reserve[0]):
        return available,split,dict(meta,reason='CURRENT_NET_RESOURCE_BASIS_UNSUPPORTED')
    raw=row[['gas_blast_furnace_'+str(i) for i in range(1,6)]].to_numpy(float)
    missing=~np.isfinite(raw);estimates=np.full(5,np.nan)
    if missing.any():
        heaters=row[['gas_air_heater_'+str(i) for i in range(1,6)]].to_numpy(float)
        if not bundle.process_meta.get('available') or not np.isfinite(heaters[missing]).all():
            return available,split,dict(meta,reason='MISSING_PROCESS_ESTIMATION_SUPPORT')
        cached=np.asarray(inputs.get('process_resource',{}).get('per_furnace_forecast_m3h',[]),float)
        predicted=cached[0] if cached.shape==(96,5) else predict_process_resource(bundle,data,issue)[0]
        estimates[missing]=predicted[missing]
    used=np.where(missing,estimates,raw)
    if not np.isfinite(used).all():return available,split,dict(meta,reason='NONFINITE_CURRENT_PROCESS_ESTIMATE')
    g=np.array([used[:3].sum(),used[3:].sum()])
    # 只估计缺测分量，已知同刻值（包括零值）不能被整组未来预测覆盖。
    if g.sum()>0.:split=g/g.sum()
    available=g-split*(float(current.known_demand)+reserve[0])
    meta.update(basis='observed_current_components_with_explicit_missing_process_proxy',
        observed_components_m3h=raw,estimated_components_m3h=estimates,used_components_m3h=used,
        observed_furnaces=(np.flatnonzero(~missing)+1).tolist(),estimated_furnaces=(np.flatnonzero(missing)+1).tolist(),
        known_demand_m3h=float(current.known_demand),reserve_m3h=float(reserve[0]),
        split_basis='current_production' if g.sum()>0. else 'forecast_share_for_known_zero_production')
    if missing.any():meta.update(process_train_end=bundle.process_meta['train_end'],
        proxy_target_time=str(inputs['times'][0]),hold_assumption='first_15min_process_forecast_for_missing_current_components_only')
    return available,split,meta


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


def _primal_validation(x,lower,upper,integer,constraint):
    """超时返回值也须满足数学模型，不能仅凭求解器状态接收指令。"""
    x=np.asarray(x,float)
    if x.shape!=np.shape(lower) or not np.isfinite(x).all():
        return {'passed':False,'reason':'NONFINITE_OR_INVALID_SHAPE'}
    def violation(value,low,high):
        low,high=np.asarray(low),np.asarray(high)
        a=np.where(np.isfinite(low),np.maximum(low-value,0.)/(1.+np.abs(low)),0.)
        b=np.where(np.isfinite(high),np.maximum(value-high,0.)/(1.+np.abs(high)),0.)
        return float(max(np.max(a),np.max(b)))
    with np.errstate(invalid='ignore',over='ignore'):
        bound_error=violation(x,lower,upper)
        activity=constraint.A@x
        row_error=violation(activity,constraint.lb,constraint.ub) if np.isfinite(activity).all() else np.inf
    selected=x[np.asarray(integer)!=0]
    integer_error=float(np.max(np.abs(selected-np.rint(selected)))) if len(selected) else 0.
    return {'passed':bool(bound_error<=1e-6 and row_error<=1e-6 and integer_error<=1e-5),
        'max_scaled_bound_violation':bound_error,'max_scaled_row_violation':row_error,
        'max_integer_violation':integer_error}


def _solve_joint(inputs,initial,beta,stock0,prices,inventory=True,export_shortfall=False,
                mapping=None,support=None,scenarios=(),risk=None,price_ratio=None,benefit_floor=None,recovery_export=False,
                gas_ratio=True):
    if inventory and export_shortfall:
        raise ValueError('库存模型不得借用导出缺气松弛')
    if recovery_export and (not inventory or export_shortfall):
        raise ValueError('库存恢复只允许作为显式的库存轨迹导出分支')
    n=len(prices); cap=np.asarray(C.UNITS);beta=np.asarray(beta)
    cursor=0
    def allocate(shape):
        nonlocal cursor
        count=int(np.prod(shape));ids=np.arange(cursor,cursor+count).reshape(shape)
        cursor+=count
        return ids
    p=allocate((n,6));on=allocate((n,6));q=allocate((n,3));h=allocate((n+1,2))
    switch=allocate((n,6));err=allocate((n,));change=allocate((n,));tail=int(allocate((1,))[0])
    shortfall=allocate((n,)) if export_shortfall else None
    reserve=allocate((n,2)) if inventory and risk is not None else None
    recovery_slack=allocate((n,2)) if recovery_export else None
    unsafe_step=allocate((n,)) if recovery_export else None
    safety_gap=allocate((n,2)) if recovery_export else None
    recovery_vent=allocate((n,2)) if recovery_export else None
    vent_active=allocate((n,2)) if recovery_export else None
    size=cursor
    cost=np.zeros(size);lo=np.zeros(size);hi=np.full(size,np.inf);integer=np.zeros(size)
    hi[p]=cap;hi[on]=1.;integer[on]=1.;hi[switch]=1.
    hi[q[:,1:]]=inputs['optional_caps']
    hi[q[:,0]]=np.maximum(np.asarray(inputs['available']).sum(axis=1),0.)+sum(C.CAPACITY)/C.DT if inventory else inputs['budget_blast']
    cost[p]=-np.asarray(prices)[:,None]*1000*C.DT
    if price_ratio is not None:
        cost[p]=-(np.asarray(prices)-price_ratio)[:,None]*1000*C.DT
    cost[switch]=C.POLICY['switch_penalty_cny'];cost[q]=1e-6
    if inventory:
        lo[h]=np.asarray(C.CAPACITY)*.20;hi[h]=np.asarray(C.CAPACITY)*.85
        lo[h[0]]=stock0;hi[h[0]]=stock0
        if recovery_export:
            # 已知不安全初态可能无法在首步恢复；软变量只衡量失败，独立核验仍按原边界判定。
            lo[h[1:]]=-np.inf
            hi[h[1:]]=np.asarray(C.CAPACITY)*.90
            cost[recovery_slack]=max(float(np.max(prices)),.01)*1000.*beta[0]*n
            hi[unsafe_step]=1.;integer[unsafe_step]=1.
            hi[vent_active]=1.;integer[vent_active]=1.
            cost[recovery_vent]=C.POLICY['vent_penalty_cny_m3']*C.DT
            # 由最大电功率反推柜位下界；每步先考虑90%自动放散，再计算下一步边界。
            maximum_blast=sum(C.UNITS)/beta[0]
            minimum_stock=np.empty((n,2))
            previous_lower=np.asarray(stock0,float).copy()
            for k in range(n):
                previous_lower=np.minimum(previous_lower+C.DT*(np.asarray(inputs['available'])[k]-
                    np.asarray(inputs['split'])[k]*maximum_blast),.90*np.asarray(C.CAPACITY))
                minimum_stock[k]=previous_lower
            lo[h[1:]]=minimum_stock
            safety_big_m=np.maximum(.15*np.asarray(C.CAPACITY)-minimum_stock,0.)+C.POLICY['stock_tolerance']
            vent_big_m=np.maximum(np.asarray(inputs['available']),0.)+np.maximum(
                np.asarray(stock0)-.90*np.asarray(C.CAPACITY),0.)/C.DT
            hi[recovery_vent]=vent_big_m
    else: hi[h]=0.
    # 保持确定性资源可行域，期末仅计机会成本，不强加旧的期末气量或库存硬约束。
    cost[tail]=float(np.max(prices))*1000*C.DT
    if reserve is not None:
        # 以有限库存的发电机会价值定价风险；保持软目标，不修改逐柜安全边界。
        cost[reserve]=beta[0]*1000*float(np.max(prices))/n
    if export_shortfall:
        hi[q[:,0]]=440./beta[0]
        cost[shortfall]=max(float(np.max(prices)),.01)*1000*C.DT*beta[0]*100.
    trusted=mapping is not None and mapping.get('trusted',False)
    small_beta=np.asarray(mapping['coefficients_mw_per_m3h']) if trusted else None
    cost[err]=C.GROUP_DISPATCH['mapping_penalty'] if trusted else 0.
    cost[change]=C.GROUP_DISPATCH['smooth_penalty']
    rows=LinearRows(size)
    if benefit_floor is not None:
        rows.add([(p[k,i],float(prices[k])*250.) for k in range(n) for i in range(6)],low=benefit_floor)
    # 正常计划使用生成裕度，恢复分支用已采纳的硬上限；离散启停不能再扣除0.001MW。
    ramp_limits=C.POLICY['ramp'] if recovery_export else C.POLICY['generate_ramp']
    for k in range(n):
        if export_shortfall:
            rows.add([(q[k,0],1.),(shortfall[k],-1.)],high=inputs['budget_blast'][k])
        for i in range(6):
            rows.add([(p[k,i],1.),(on[k,i],-cap[i])],high=0.)
            rows.add([(p[k,i],1.),(on[k,i],-.6*cap[i])],low=0.)
            prior_on=float(initial[i]>=.6*cap[i]-.001)
            for sign in (-1.,1.):
                terms=[(p[k,i],sign),(on[k,i],cap[i])]
                bound=cap[i]*.15+2*cap[i]
                if k:
                    terms += [(p[k-1,i],-sign),(on[k-1,i],cap[i])]
                else: bound+=sign*initial[i]-cap[i]*prior_on
                rows.add(terms,high=bound)
                terms=[(on[k,i],sign),(switch[k,i],-1.)]
                if k: terms.append((on[k-1,i],-sign))
                rows.add(terms,high=0. if k else sign*prior_on)
        for ids,limit in ((range(4),ramp_limits[0]),(range(6),ramp_limits[1])):
            terms=[(p[k,i],1.) for i in ids]
            prior=0.
            if k: terms += [(p[k-1,i],-1.) for i in ids]
            else: prior=float(initial[list(ids)].sum())
            rows.add(terms,low=prior-limit,high=prior+limit)
            if len(ids)==4:
                for sign in (-1.,1.):
                    rows.add([(i,v*sign) for i,v in terms]+[(change[k],-1.)],high=sign*prior)
        rows.add([(p[k,i],1.) for i in range(6)]+[(q[k,j],-beta[j]) for j in range(3)],low=0.,high=0.)
        # 气种占比硬约束：官方 composition 按申报耗气重建机组组成校验，焦炉/转炉
        # 占比越界直接计违规。阈值 3%/11% 由线上 comp=6 反推（3% 以上 5 步、
        # 11% 以上 1 步，合计 6 步）；实测该约束把 load 从 5 降到 1。
        if gas_ratio:
            rows.add([(q[k,0],-.03),(q[k,1],.97),(q[k,2],-.03)],high=0.)
            rows.add([(q[k,0],-.11),(q[k,1],-.11),(q[k,2],.89)],high=0.)
        if trusted:
            terms=[(p[k,i],1.) for i in range(4)]+[(q[k,j],-small_beta[j]) for j in range(3)]
            for sign in (-1.,1.): rows.add([(i,v*sign) for i,v in terms]+[(err[k],-1.)],high=0.)
        # 历史映射情景只作敏感性诊断，不以推测映射替代决策功率的容量与爬坡约束。
        if inventory:
            for g in range(2):
                bound=C.DT*inputs['available'][k,g]
                balance=[(h[k+1,g],1.),(h[k,g],-1.),(q[k,0],C.DT*inputs['split'][k,g])]
                if recovery_export:
                    balance.append((recovery_vent[k,g],C.DT))
                rows.add(balance,low=bound,high=bound)
                if recovery_export:
                    rows.add([(h[k+1,g],1.),(recovery_slack[k,g],1.)],low=.20*C.CAPACITY[g])
                    rows.add([(h[k+1,g],1.),(unsafe_step[k],safety_big_m[k,g])],low=.15*C.CAPACITY[g])
                    rows.add([(h[k+1,g],1.),(safety_gap[k,g],1.)],low=.15*C.CAPACITY[g])
                    # 放散只可在柜位达到90%上限时发生，避免用人工放散伪造较低柜位。
                    rows.add([(recovery_vent[k,g],1.),(vent_active[k,g],-vent_big_m[k,g])],high=0.)
                    big_m=max(.90*C.CAPACITY[g]-minimum_stock[k,g],0.)+C.POLICY['stock_tolerance']
                    rows.add([(h[k+1,g],1.),(vent_active[k,g],-big_m)],low=.90*C.CAPACITY[g]-big_m)
                if reserve is not None:
                    rows.add([(h[k+1,g],1.),(reserve[k,g],1.)],low=risk['target_m3'][g])
    rows.add([(q[-1,0],beta[0]),(tail,-1.)],high=float(inputs['budget_blast'][-1])*beta[0])
    tick=time.perf_counter()
    stages=[]
    def optimize(objective):
        constraint=rows.constraint()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected:.*threads.*')
            result=milp(objective,integrality=integer,bounds=Bounds(lo,hi),constraints=constraint,
                options={'time_limit':C.POLICY['solver_seconds'],'mip_rel_gap':C.DISPATCH_CONTROL['mip_gap'],'threads':C.CPU_THREADS})
        if result.x is not None:
            result.primal_validation=_primal_validation(result.x,lo,hi,integer,constraint)
            if not result.primal_validation['passed']:
                result.x=None
                result.message=str(result.message)+'; rejected_invalid_primal_point'
        return result
    if recovery_export:
        objective=np.zeros(size);objective[unsafe_step]=1.
        first=optimize(objective)
        stages.append({'objective':'unsafe_steps','status_code':int(first.status),
            'mip_gap':getattr(first,'mip_gap',None),'value':float(first.fun) if first.x is not None else None})
        result=first
        if first.x is not None:
            step_limit=int(round(float(first.x[unsafe_step].sum())))
            rows.add([(int(i),1.) for i in unsafe_step],high=step_limit+1e-6)
            objective=np.zeros(size);objective[safety_gap]=1.
            second=optimize(objective)
            stages.append({'objective':'safety_stock_deficit_m3_steps','status_code':int(second.status),
                'mip_gap':getattr(second,'mip_gap',None),'value':float(second.fun) if second.x is not None else None})
            if second.x is not None:result=second
            debt=float(np.maximum(.15*np.asarray(C.CAPACITY)-result.x[h[1:]],0.).sum())
            rows.add([(int(i),1.) for i in safety_gap.ravel()],high=debt+.01)
            third=optimize(cost)
            stages.append({'objective':'existing_margin_and_revenue_cost','status_code':int(third.status),
                'mip_gap':getattr(third,'mip_gap',None),'value':float(third.fun) if third.x is not None else None})
            # 超时可返回较差可行解；优先保留满足最终约束的更优已有解。
            final_constraint=rows.constraint()
            incumbent_valid=_primal_validation(result.x,lo,hi,integer,final_constraint)['passed']
            if third.x is not None and (not incumbent_valid or cost@third.x < cost@result.x-1e-7):
                result=third
                incumbent_valid=True
            # 最终MILP限时未证最优时，固定已找到的离散状态细化连续量；不能放松前两级安全目标。
            if incumbent_valid and (third.x is None or third.status != 0):
                fixed_lo=lo.copy();fixed_hi=hi.copy()
                fixed=np.flatnonzero(integer)
                fixed_lo[fixed]=np.rint(result.x[fixed]);fixed_hi[fixed]=fixed_lo[fixed]
                revenue_vector=np.zeros(size)
                revenue_vector[p.ravel()]=np.repeat(np.asarray(prices)*250.,6)
                revenue_floor=float(revenue_vector@result.x)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',message='Unrecognized options detected:.*threads.*')
                    refined=milp(cost,integrality=np.zeros_like(integer),bounds=Bounds(fixed_lo,fixed_hi),
                        constraints=[final_constraint,LinearConstraint(revenue_vector,revenue_floor-.01,np.inf)],
                        options={'time_limit':C.POLICY['solver_seconds'],
                        'threads':C.CPU_THREADS})
                if refined.x is not None:
                    refined.primal_validation=_primal_validation(refined.x,lo,hi,integer,final_constraint)
                    if not refined.primal_validation['passed'] or revenue_vector@refined.x<revenue_floor-.02:
                        refined.x=None
                improved=refined.x is not None and cost@refined.x < cost@result.x-1e-7
                stages.append({'objective':'fixed_integer_economic_refinement','status_code':int(refined.status),
                    'value':float(cost@refined.x) if refined.x is not None else None,'improved':bool(improved)})
                if improved:result=refined
    else:
        result=optimize(cost)
    meta={'status_code':int(result.status),'message':str(result.message),'seconds':time.perf_counter()-tick,
          'inventory_constraints':inventory,'export_resource_shortfall_allowed':export_shortfall,
          'mip_gap':getattr(result,'mip_gap',None),'joint_dispatch':True}
    meta['recovery_export_only']=bool(recovery_export)
    meta['primal_validation']=getattr(result,'primal_validation',None)
    meta['generated_ramp_limits_mw']=list(ramp_limits)
    if recovery_export:meta['recovery_priority_stages']=stages
    if result.x is None: return None,meta
    meta['status']='OPTIMAL' if result.status==0 else 'FEASIBLE_WITH_GAP'
    if recovery_export:meta['status']='EXPORT_RECOVERY_WITH_REPORTED_VIOLATIONS'
    x=result.x
    if recovery_export:meta['recovery_economic_cost']=float(cost@x)
    out={'power':x[p],'gas':x[q],'stocks':x[h] if inventory else np.full((n+1,2),np.nan),
         'vent':x[recovery_vent] if recovery_export else np.zeros((n,2)),
         'export_shortfall_m3h':x[shortfall] if export_shortfall else np.zeros(n),
         'joint_control':{'terminal_resource_excess_mw':float(x[tail]),'group_mapping_used':trusted,
                          'empirical_gas_support':support,'ramp_scenarios':[list(b) for b in scenarios],
                          'reserve_risk':risk,'price_ratio_parameter':price_ratio,'benefit_floor_cny':benefit_floor}}
    if recovery_export:
        out['recovery_export']={'active':True,'export_only':True,
            'planning_inventory_shortfall_m3':x[recovery_slack],
            'objective_basis':'unsafe_steps_then_safety_deficit_then_existing_margin_and_revenue',
            'minimum_unsafe_steps_found':int(round(float(x[unsafe_step].sum()))),
            'safety_deficit_m3_steps':float(np.maximum(.15*np.asarray(C.CAPACITY)-x[h[1:]],0.).sum()),
            'priority_optimality_certified':bool(len(stages)>=2 and all(s['status_code']==0 and
                (s.get('mip_gap') or 0.)<=1e-9 for s in stages[:2])),
            'safe_bounds_unchanged':True,'slack_is_not_physical_gas_supply':True}
    return out,meta


def solve_milp(*args, **kwargs):
    fraction=C.DISPATCH_CONTROL["benefit_fraction"]
    best, meta = _solve_joint(*args, **kwargs)
    if best is None or fraction == 1. or kwargs.get("export_shortfall", False) or kwargs.get('recovery_export',False):
        return best, meta
    prices = np.asarray(args[4], float)
    def ratio(solution):
        power = solution["power"].sum(axis=1)
        return float(prices@power/power.sum()) if power.sum() > 1e-6 else 0.
    original_revenue = float(prices@best["power"].sum(axis=1)*250.)
    floor = original_revenue*fraction
    steps = [{"ratio": ratio(best), "revenue_cny": original_revenue, "status": meta["status"]}]
    for _ in range(C.DISPATCH_CONTROL["ratio_iterations"]):
        value = ratio(best)
        candidate, attempt = _solve_joint(*args, **kwargs, price_ratio=value, benefit_floor=floor)
        if candidate is None:
            break
        candidate_ratio = ratio(candidate)
        revenue = float(prices@candidate["power"].sum(axis=1)*250.)
        steps.append({"ratio": candidate_ratio, "revenue_cny": revenue, "status": attempt["status"]})
        if candidate_ratio <= value+1e-7 or revenue < floor-.01:
            break
        best, meta = candidate, attempt
        if candidate_ratio-value < 1e-5:
            break
    best["joint_control"]["price_frontier"] = {"benefit_fraction": fraction, "gross_reference_cny": original_revenue,
                                               "benefit_floor_cny": floor, "iterations": steps,
                                               "raw_ratio_is_official_price_eff": False}
    return best, meta


def fallback_arrays(inputs,previous,beta):
    """最终导出分支保留真实失败；不把不确定柜位补成中位库存。"""
    n=len(inputs["times"])
    p=np.tile(previous,(n,1))
    q=np.zeros((n,3))
    mix=np.column_stack((inputs["budget_blast"],inputs["optional_caps"]))
    for k in range(n):
        energy=float(mix[k]@beta)
        if energy>0.:
            q[k]=mix[k]*(p[k].sum()/energy)
        elif beta[0]>0.:
            q[k,0]=p[k].sum()/beta[0]
    return {"power":p,"gas":q,"stocks":np.full((n+1,2),np.nan),"vent":np.full((n,2),np.nan)}


def _replay_stocks(start,inputs,gas):
    n=len(gas)
    if not np.isfinite(start).all():
        return np.full((n+1,2),np.nan),np.full((n,2),np.nan)
    stocks=[np.asarray(start,float)]
    vents=[]
    for k in range(n):
        raw=stocks[-1]+C.DT*(inputs["available"][k]-inputs["split"][k]*gas[k,0])
        vent=np.maximum(raw-.90*np.array(C.CAPACITY),0.)/C.DT
        stocks.append(raw-C.DT*vent);vents.append(vent)
    return np.array(stocks),np.array(vents)


def optimize_daily(bundle,data,origins,quick=False,progress=None,output_window=None):
    control=validate_dispatch_calibration(bundle)
    validate_process_resource(bundle)
    mapping=bundle.diagnostics['group_mapping']
    support=control['support'] if control['support'].get('available') else None
    scenarios=[x['coefficients_mw_per_m3h'] for x in control['mapping_scenarios']]
    dates=pd.DatetimeIndex(origins).normalize().unique()
    if quick:
        dates=dates[:2]
    if output_window is not None:
        start,end=map(pd.Timestamp,output_window)
        first=start.normalize()+pd.Timedelta(hours=7)
        if first>=start:
            first-=pd.Timedelta(days=1)
        last=dates[-1]+pd.Timedelta(hours=7)
        dates=pd.date_range(first,last,freq='1D')-pd.Timedelta(hours=7)
        if bundle.bootstrap is None or pd.Timestamp(bundle.bootstrap.train_end)>=first:
            raise ValueError('2111行首端预案缺少发布时间之前训练的资源模型')
    base=build_features(data)
    beta=np.asarray(bundle.beta)
    records,days=[],[]
    previous,previous_gas,previous_state=None,None,None
    for date in dates:
        issue=date+pd.Timedelta(hours=7)
        active=bundle.bootstrap if output_window is not None and issue==first else bundle
        control=validate_dispatch_calibration(active)
        validate_process_resource(active)
        mapping=active.diagnostics['group_mapping']
        support=control['support'] if control['support'].get('available') else None
        scenarios=[x['coefficients_mw_per_m3h'] for x in control['mapping_scenarios']]
        beta=np.asarray(active.beta)
        steps=int((end-issue)/pd.Timedelta(minutes=15)) if output_window is not None and date==dates[-1] else 96
        inputs=forecast_resources(active,data,issue,base,steps=steps)
        price=prices_at(data,inputs["times"])
        warnings=[]
        try:
            initial,initial_meta=initial_power(data,issue,active,previous)
        except ValueError as exc:
            # 首日无可辨识初态时仅作完整导出，并把未知初态作为验收失败。
            initial=np.zeros(6)
            initial_meta={"basis":"UNVERIFIED_EXPORT_STATE","reason":str(exc)}
            warnings.append(str(exc))
        row=data.frame.loc[issue]
        anchor=row[list(C.HOLDERS)].to_numpy(float)
        anchor_ok=bool(np.isfinite(anchor).all())
        q0=row[list(C.USE)].to_numpy(float) if previous_gas is None else previous_gas.copy()
        bridge_available,bridge_split,bridge_source=bridge_resources(active,data,issue,inputs)
        if np.isfinite(q0).all() and anchor_ok:
            raw=anchor+C.DT*(bridge_available-bridge_split*q0[0])
            bridge_vent=np.maximum(raw-.9*np.array(C.CAPACITY),0.)
            bridge=raw-bridge_vent
        else:
            bridge=np.full(2,np.nan)
            bridge_vent=np.full(2,np.nan)
            warnings.append("桥接缺少有效柜位或此前生效耗气")
        inputs['previous_gas']=q0.copy()
        inputs['prices']=price.copy()
        inputs['control_calibration']={'version':control['version'],'train_end':control['train_end'],
            'optional_use_calibration':deepcopy(control['recipe']['optional_use_calibration'])}
        inventory=bool(np.isfinite(bridge).all() and inputs["resource_mode"]=="net_inferred")
        # 首端预案用更早的 bootstrap 快照，供气结构与本窗口不同，不加气种比例约束，
        # 否则 10-10 那次发布直接不可行；其余发布一律加硬约束。
        options=dict(mapping=mapping,support=support,scenarios=scenarios,risk=inventory_risk(data,issue),
                     gas_ratio=not (output_window is not None and issue==first))
        solution,solver=solve_milp(inputs,initial,beta,bridge,price,inventory=inventory,**options)
        fallback=not inventory or solution is None or bool(warnings)
        if solution is None and inventory:
            recovery_options=dict(options,risk=None)
            solution,recovery_solver=solve_milp(inputs,initial,beta,bridge,price,inventory=True,recovery_export=True,**recovery_options)
            solver['recovery_export_solver']=recovery_solver
            if solution is not None:
                warnings.append('仅导出：库存模型无解，优先恢复柜位；松弛量仅记录失败，不作为实际供气或可行计划')
        if solution is None and inventory:
            warnings.append("库存约束模型无可行解，尝试保留机组与资源约束的导出分支")
            solution,secondary=solve_milp(inputs,initial,beta,bridge,price,inventory=False,**options)
            solver["export_solver"]=secondary
        if solution is None:
            # 守住机组与爬坡，显式最小化资源冲突；不能把松弛解标为正常可行计划。
            solution,last_solver=solve_milp(inputs,initial,beta,bridge,price,inventory=False,export_shortfall=True,**options)
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
                 gas_power_validation=active.diagnostics["gas_power"]["holdout"],
                 bridge={"start":str(issue),"end":str(inputs["times"][0]),"gas_m3h":q0,
                         "available_m3h":bridge_available,"split":bridge_split,"vent_m3":bridge_vent,
                         'resource_components':bridge_source},
                 state_reset_m3=anchor-previous_state if previous_state is not None else None,
                 warnings=warnings,fallback_used=fallback)
        day['export_mask']=np.asarray((inputs['times']>=start)&(inputs['times']<=end) if output_window is not None else np.ones(steps,bool))
        if output_window is not None:
            day['export_window']={'start':str(start),'end':str(end)}
        day["validation"]=validate_day(day)
        mapping=active.diagnostics.get("group_mapping")
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
        exported=day['export_mask']
        day["plan_revenue_cny"]=float(np.sum(price[exported]*solution["power"][exported].sum(axis=1))*250.)
        day["planned_vent_m3"]=float(np.nansum(solution["vent"][exported])*C.DT)
        day["export_resource_shortfall_m3"]=float(np.sum(np.asarray(solution.get("export_shortfall_m3h",np.zeros(steps)))[exported])*C.DT)
        for k,when in enumerate(inputs["times"]):
            if not exported[k]:
                continue
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
    boundary=[d['validation'].get('details',{}).get('holder_boundary_diagnostics',{}) for d in days]
    sequence=validate_plan_sequence(days)
    summary={"publication_time":"07:00","intraday_refresh":False,"daily_publications":len(days),
             "planned_steps":len(schedule),"constraint_validation_passed":not failures and sequence["passed"],
             "sequence_validation":sequence,"validation_scope":"project_local_strict_validation",
             "official_acceptance_status":"unknown",
             "failed_publications":failures,"fallback_publications":sum(d["fallback_used"] for d in days),
             "group_refined_publications":sum(d.get("refinement",{}).get("applied",False) for d in days),
             "plan_revenue_cny":sum(d["plan_revenue_cny"] for d in days),
             "planned_vent_m3":sum(d["planned_vent_m3"] for d in days),
             "holder_boundary_diagnostics":{
                 "preexisting_unsafe_anchor_publications":sum(bool(x.get('anchor_unsafe_holders')) for x in boundary),
                 "unknown_anchor_publications":sum(not x.get('anchor_known',False) for x in boundary),
                 "unsafe_pre_command_publications":sum(bool(x.get('pre_command_unsafe_holders')) for x in boundary),
                 "post_execution_unsafe_publications":sum((x.get('post_execution_unsafe_steps') or 0)>0 for x in boundary),
                 "post_execution_unknown_steps":sum(x.get('post_execution_unknown_steps',96) for x in boundary),
                 "strict_safety_gate_unchanged":True},
             "dispatch_control":{'version':control['version'],'recipe':control['recipe'],
                                 'calibration_train_end':control['train_end']},
             "physical_model_basis":"adopted_project_policy_inferred_resources",
             "field_safety_certified":False,"replay_complete":False,
             "execution_revenue":None,"note":"计划轨迹不冒充官方柜位重建或真实执行收益"}
    summary['opt_result_reused']=False
    summary['output_window']=output_window
    summary['tail_resource_policy']=C.OPT_TAIL
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
    n=len(day["gas"])
    power=np.arange(6*n).reshape(n,6)
    on=np.arange(6*n,12*n).reshape(n,6)
    error=np.arange(12*n,13*n)
    change=np.arange(13*n,14*n)
    size=14*n
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
    for k in range(n):
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
    constraint=rows.constraint()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="Unrecognized options detected:.*threads.*")
        result=milp(cost,integrality=integer,bounds=Bounds(lower,upper),constraints=constraint,
                    options={"time_limit":10.,"mip_rel_gap":.005,"threads":C.CPU_THREADS})
    if result.x is None or not _primal_validation(result.x,lower,upper,integer,constraint)['passed']:
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
