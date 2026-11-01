"""统一入口：新版独立树模型、历史验证、冻结推理与每日调度。"""
import argparse
import json
import logging
from pathlib import Path
import sys
import time
import shutil
import numpy as np
import pandas as pd
import config as C
import main_s
import main_l
import main_opt
import output
from data_cleaning.pipeline import prepare_data, build_features
from data_cleaning.quality import (truth_cube, score_load, score_horizons, score_prediction_csv, score_origin_window, load_diagnostics, validate_day,
                                   validate_result_files, historical_replay, project_load, audit_resource_forecasts,
                                   price_weighted_load, score_resources)

LOG=logging.getLogger("aic-half")


def prediction_view(data,start):
    # 评分真值留在原始对象，预测视图只暴露窗口前的标签，避免后续特征误读未来值。
    view=data.masked_copy(start,mask_use=True)
    view.labels=data.labels.copy()
    view.labels.loc[view.labels.index>=pd.Timestamp(start),:]=np.nan
    return view


def setup_log(ctx):
    LOG.handlers.clear()
    LOG.setLevel(logging.INFO)
    formatter=logging.Formatter("%(asctime)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout),logging.FileHandler(ctx["root"]/"报告"/"运行日志.log",encoding="utf-8")):
        handler.setFormatter(formatter)
        LOG.addHandler(handler)


def evaluate_load(data,origins,recipe,quick=False):
    origins=pd.DatetimeIndex(origins)
    start,end=origins[0],origins[-1]
    train=data.prefix(start-pd.Timedelta(minutes=15))
    train.require_training(start-pd.Timedelta(minutes=15))
    short=main_s.train_short(train,recipe,quick,LOG.info)
    long=main_l.train_long(train,recipe,quick,LOG.info)
    view=prediction_view(data,start)
    ps,pl=main_s.predict_deployed(short,long,view,origins,recipe)
    # 历史评分按导出CSV的6位精度计算，避免与实际提交文件产生细微差异。
    ps,pl=np.round(ps,6),np.round(pl,6)
    # 各历史窗统一以声明窗口末端截断目标时刻；目标真值不进入预测视图。
    ys=truth_cube(data,origins,range(1,9),window_end=end)
    yl=truth_cube(data,origins,range(1,97),window_end=end)
    ns=main_s.naive_profile(train,origins,range(1,9),train.index[-1])
    nl=main_s.naive_profile(train,origins,range(1,97),train.index[-1])
    metrics={"short":score_load(ps,ys),"long":score_load(pl,yl),
             "short_horizons":score_horizons(ps,ys),"long_horizons":score_horizons(pl,yl),
             "naive_short":score_load(ns,ys),"naive_long":score_load(nl,yl),
             "short_diagnostics":load_diagnostics(ps,ys,origins),
             "long_diagnostics":load_diagnostics(pl,yl,origins),
             "score_contract":{"target_time_end":str(end),"tail":"within_window",
                 "zero_target":"separate_from_mape","target_weight":"equal_local_assumption",
                 "official_subscore_mapping":"unknown"},
             "max_training_target":str(train.index[-1]),"information":"process_only;validation_target_blackout"}
    metrics["available_history_tail_sensitivity"]={
        "short":score_load(ps,truth_cube(data,origins,range(1,9))),
        "long":score_load(pl,truth_cube(data,origins,range(1,97))),
        "role":"diagnostic_only_not_candidate_ranking"}
    # 最近的半决赛训练段单独报告，原窗口仍是固定主指标，测试集不参与选模。
    for stage,begin,stop,role in C.STAGES:
        if stage=="half_train" and role=="train":
            begin=max(start,pd.Timestamp(begin));stop=min(end,pd.Timestamp(stop))
            if begin<=stop:
                metrics["recent_training_window"]={"stage":stage,
                    "short":score_origin_window(ps,ys,origins,begin,stop),
                    "long":score_origin_window(pl,yl,origins,begin,stop)}
    metrics["local_prediction_objective"]=(2*metrics["short"]["macro_score"]+3*metrics["long"]["macro_score"])/5
    return short,long,ps,pl,metrics


def validate_prediction_csv_scores(root,data,origins,predictions,metrics):
    """从导出宽表独立重建标签时间并核对历史评分口径。"""
    origins=pd.DatetimeIndex(origins)
    results={}
    for span,filename,h,pred in (("short","s_result.csv",8,predictions[0]),
                                  ("long","l_result.csv",96,predictions[1])):
        score,csv_pred,csv_truth=score_prediction_csv(Path(root)/filename,data.labels,origins,h,origins[-1])
        expected_truth=truth_cube(data,origins,range(1,h+1),window_end=origins[-1])
        passed=(np.array_equal(csv_pred,pred) and np.array_equal(csv_truth,expected_truth,equal_nan=True) and
                all(score[t]["score"]==metrics[span][t]["score"] for t in C.TARGETS))
        if not passed:
            raise ValueError(f"{span}历史评分与导出CSV不一致")
        results[span]={"passed":True,"time_alignment":"origin_plus_column_offset",
                       "basis":"independent_csv_and_reference_labels","official_aggregation":"unknown"}
    return results


def run_backtest(ctx,quick=False):
    data=prepare_data()
    metrics=[]
    for spec in C.FOLDS[:2] if quick else C.FOLDS:
        origins=pd.date_range(spec["start"],spec["end"],freq="15min")
        LOG.info("历史折 %s：连续21天，开始新模型训练",spec["name"])
        short,long,ps,pl,score=evaluate_load(data,origins,C.LOAD_RECIPE,quick)
        train=data.prefix(origins[0]-pd.Timedelta(minutes=15))
        resource=main_opt.fit_resources(train,quick,LOG.info)
        view=prediction_view(data,origins[0])
        schedule,days,opt=main_opt.optimize_daily(resource,view,origins,quick,LOG.info)
        opt["price_weighted_load"]=price_weighted_load(days)
        replay=historical_replay(data.prefix(origins[-1]),days)
        metrics.append({"fold":spec["name"],**score,"optimization":opt,"historical_replay":replay,
                        "resource_validation":audit_resource_forecasts(days,main_opt.resource_labels(data),
                                                                       window_end=origins[-1])})
        fold_dir=ctx["root"]/spec["name"]
        (fold_dir/"模型").mkdir(parents=True)
        output.save_models(fold_dir/"模型",short,long,resource,{"role":"development","source":ctx["hashes"]})
        output.write_results(fold_dir,origins,ps,pl,schedule)
        metrics[-1]["csv_score_validation"]=validate_prediction_csv_scores(fold_dir,data,origins,(ps,pl),score)
        restored_s,restored_l,_,_=output.load_models(fold_dir)
        probe=origins[::251]
        restored_ps,restored_pl=main_s.predict_deployed(restored_s,restored_l,view,probe,C.LOAD_RECIPE)
        reload_error={"short":float(np.max(np.abs(restored_ps-ps[::251]))),
                      "long":float(np.max(np.abs(restored_pl-pl[::251])))}
        if max(reload_error.values())>1e-6:
            raise ValueError("历史验证模型重载不一致")
        metrics[-1]["reload_max_difference"]=reload_error
        metrics[-1]["coverage"]=validate_result_files(fold_dir,origins,schedule.datetime)
        report={"status":"DEVELOPMENT_COMPLETE","experimental":True,"full_window":False,
                "submittable":False,"validation":metrics,"daily_plans":days,
                "prediction_grid":[str(origins[0]),str(origins[-1])],
                "optimization_grid":[str(schedule.datetime.iloc[0]),str(schedule.datetime.iloc[-1])],
                "source":ctx["hashes"],"run_id":ctx["run_id"]}
        output.write_report(ctx,report)
    report.update(coverage={"passed":all(x["coverage"]["passed"] for x in metrics)},upload_ready=False,
                  note="历史折独立保存模型及三表；开发验证不组装为正式提交包")
    output.assert_source(ctx)
    output.write_report(ctx,report)
    return report


def reload_check(ctx,short,long,resource,data,origins,training_contract=None):
    inherited=training_contract or {}
    provenance=inherited.get("training_source",inherited.get("source",ctx["hashes"]))
    # 重载核验保留逐链路来源与实验角色，不能把复用的小预算候选重新标记为最终训练。
    contract=dict(inherited)
    contract.update(source=ctx['hashes'],policy=C.POLICY,training_source=provenance,
                    role=inherited.get('role','development' if ctx['experimental'] else 'final'))
    output.save_models(ctx["root"]/"模型",short,long,resource,contract)
    loaded_s,loaded_l,loaded_r,meta=output.load_models(ctx["root"])
    probe=pd.DatetimeIndex(origins)[::251]
    differences={}
    for name,b,restored,predict in (("short",short,loaded_s,main_s.predict_short),("long",long,loaded_l,main_l.predict_long)):
        a=predict(b,data,probe)
        z=predict(restored,data,probe)
        differences[name]=float(np.max(np.abs(a-z)))
        if differences[name]>.000001:
            raise ValueError(f"{name}保存重载不一致")
    before=main_s.predict_deployed(short,long,data,probe,C.LOAD_RECIPE)
    after=main_s.predict_deployed(loaded_s,loaded_l,data,probe,C.LOAD_RECIPE)
    differences["deployed_short"]=float(np.max(np.abs(before[0]-after[0])))
    differences["deployed_long"]=float(np.max(np.abs(before[1]-after[1])))
    if max(differences["deployed_short"],differences["deployed_long"])>.000001:
        raise ValueError("部署预测保存重载不一致")
    issue=pd.DatetimeIndex(origins)[0].normalize()+pd.Timedelta(hours=7)
    a=main_opt.forecast_resources(resource,data,issue)
    z=main_opt.forecast_resources(loaded_r,data,issue)
    differences["resource"]=float(np.max(np.abs(a["available"]-z["available"])))
    if differences["resource"]>.000001:
        raise ValueError("资源保存重载不一致")
    differences['optional_caps']=float(np.max(np.abs(a['optional_caps']-z['optional_caps'])))
    if differences['optional_caps']>.000001:
        raise ValueError('可选气种耗气包络保存重载不一致')
    if resource.process_meta.get('available'):
        a=main_opt.predict_process_resource(resource,data,issue)
        z=main_opt.predict_process_resource(loaded_r,data,issue)
        differences['process_resource']=float(np.max(np.abs(a-z)))
        if differences['process_resource']>.000001:raise ValueError('资源过程模型保存重载不一致')
    if resource.bootstrap is not None:
        issue=pd.Timestamp(C.OPT_START).normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=7)
        a=main_opt.forecast_resources(resource.bootstrap,data,issue)
        z=main_opt.forecast_resources(loaded_r.bootstrap,data,issue)
        differences['bootstrap_resource']=float(np.max(np.abs(a['available']-z['available'])))
        if differences['bootstrap_resource']>1e-6:raise ValueError('首端资源模型保存重载不一致')
    return differences


def infer(ctx,quick=False,reuse=None,opt_only=False,train_bootstrap=False):
    tick=time.perf_counter()
    if reuse:
        short,long,resource,meta=output.load_models(reuse)
        if not ctx["experimental"] and (meta.get("training_quick") or meta.get("contract",{}).get("role")!="final"):
            raise ValueError("正式运行不得复用quick或历史开发模型")
        LOG.info("复用兼容负荷与常规资源模型：%s",reuse)
    else:
        meta={}
        train=prepare_data(cutoff=C.TRAIN_END)
        LOG.info("最终训练：%s，纯LightGBM配方=%s",C.TRAIN_END,C.LOAD_RECIPE)
        short=main_s.train_short(train,quick=quick,progress=LOG.info)
        long=main_l.train_long(train,quick=quick,progress=LOG.info)
        resource=main_opt.fit_resources(train,quick,LOG.info)
    data=prepare_data(include_test=True)
    origins=pd.date_range(C.TEST_START,C.TEST_END,freq="15min")
    if quick:
        origins=origins[:192]
    if not origins.isin(data.index).all():
        raise ValueError("正式预测输入覆盖不完整")
    if data.labels.loc[C.TEST_START:C.TEST_END].notna().any().any():
        raise ValueError("测试标签隔离失败")
    view=prediction_view(data,C.TEST_START)
    bootstrap_issue=pd.Timestamp(C.OPT_START).normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=7)
    bootstrap_trained=False
    bootstrap_seconds=0.
    if resource.bootstrap is None:
        if opt_only and not train_bootstrap:
            raise ValueError('首端资源模型缺失；opt仅重算时需复用含首端模型的运行，首次可显式--train-bootstrap')
        cutoff=bootstrap_issue-pd.Timedelta(minutes=C.STEP)
        LOG.info('首端预案资源训练：截止%s；不重训负荷模型',cutoff)
        bootstrap_tick=time.perf_counter()
        resource.bootstrap=main_opt.fit_resources(prepare_data(cutoff=cutoff),quick,LOG.info)
        bootstrap_seconds=time.perf_counter()-bootstrap_tick
        resource.bootstrap.diagnostics['training_source']=ctx['hashes']
        bootstrap_trained=True
    if not ctx['experimental'] and resource.bootstrap.diagnostics['training_budget']['quick']:
        raise ValueError('正式运行不得复用quick首端资源模型')
    reload=reload_check(ctx,short,long,resource,view,origins,meta.get("contract"))
    if opt_only:
        source=Path(reuse)
        for name in ("s_result.csv","l_result.csv"):
            if quick:
                table=pd.read_csv(source/name)
                times=pd.to_datetime(table.datetime,errors="coerce")
                table=table.loc[times.isin(origins)]
                if len(table)!=len(origins):
                    raise ValueError("复用预测表不能覆盖quick请求网格")
                table.to_csv(ctx["root"]/name,index=False,float_format="%.6f",encoding="utf-8")
            else:
                shutil.copy2(source/name,ctx["root"]/name)
        ps=pl=None
    else:
        LOG.info("推理：%s个起报点；测试标签不计算本地分数",len(origins))
        ps,pl=main_s.predict_deployed(short,long,view,origins,C.LOAD_RECIPE)
    effective_end=origins[-1].normalize()+pd.Timedelta(days=1,hours=7) if quick else pd.Timestamp(C.OPT_END)
    effective=pd.date_range(C.OPT_START,effective_end,freq='15min')
    dispatch_tick=time.perf_counter()
    schedule,days,opt=main_opt.optimize_daily(resource,view,origins,quick,LOG.info,(C.OPT_START,str(effective_end)))
    dispatch_seconds=time.perf_counter()-dispatch_tick
    opt["price_weighted_load"]=price_weighted_load(days)
    if opt_only:
        s=schedule.copy()
        s.datetime=pd.to_datetime(s.datetime).dt.strftime("%Y-%m-%d %H:%M:%S")
        s.to_csv(ctx["root"]/"opt_result.csv",index=False,float_format="%.6f",encoding="utf-8")
    else:
        output.write_results(ctx["root"],origins,ps,pl,schedule)
    coverage=validate_result_files(ctx["root"],origins,effective)
    if not coverage["passed"]:
        raise ValueError("三表导出契约未通过："+json.dumps(coverage,ensure_ascii=False))
    full=bool(origins.equals(pd.date_range(C.TEST_START,C.TEST_END,freq='15min')) and
              pd.DatetimeIndex(schedule.datetime).equals(pd.date_range(C.OPT_START,C.OPT_END,freq='15min')))
    passed=bool(full and not ctx["experimental"] and coverage["passed"] and opt["constraint_validation_passed"])
    report={"run_id":ctx["run_id"],"schema":C.SCHEMA,"mode":"opt" if opt_only else "infer",
            "status":"COMPLETE" if passed else "COMPLETE_WITH_CONSTRAINT_FAILURES" if not opt["constraint_validation_passed"] else "EXPERIMENT_COMPLETE",
            "experimental":ctx["experimental"],"full_window":full,"submittable":passed,
            "validation_scope":"project_local_strict_validation","official_acceptance_status":"unknown",
            "submission_conditions":{"full_window":full,"non_experimental_run":not ctx["experimental"],
                "files_complete":coverage["passed"],"local_constraints_passed":opt["constraint_validation_passed"]},
            "threshold_basis":{"capacity":"data_dictionary.xlsx 数据说明!A21",
                "safety_ratio":"contest_problem_statement","planning_margin":"project_policy",
                "anchor_disqualification_official_rule":"unspecified"},
            "upload_ready":coverage["passed"],"training_performed":not bool(reuse) or bootstrap_trained,
            "load_training_performed":not bool(reuse),"bootstrap_training_performed":bootstrap_trained,
            "bootstrap_train_end":resource.bootstrap.train_end,"prediction_performed":not opt_only,
            "bootstrap_training_seconds":bootstrap_seconds,"dispatch_seconds":dispatch_seconds,
            "opt_result_reused":False,
            "reused_models_from":str(reuse) if reuse else None,"recipe":short.recipe,
            "share_calibration":C.SHARE_CALIBRATION,
            "prediction_score":{"status":"NO_OBSERVED_TEST_TRUTH","score":None,
                "reason":"测试期负荷标签为空，不能计算本地1-MAPE或推断官网分数"},
            "prediction_grid":[str(origins[0]),str(origins[-1])],
            "optimization_grid":[str(schedule.datetime.iloc[0]),str(schedule.datetime.iloc[-1])],
            "quality":data.quality,"data_sources":data.sources,"source":ctx["hashes"],
            "reload_max_abs_difference":reload,"optimization":opt,"daily_plans":days,
            "historical_replay":historical_replay(data,days),"coverage":coverage,
            "field_safety_certified":False,"official_score":None,"duration_seconds":round(time.perf_counter()-tick,3)}
    report['validation_scope']='local_strict_project_model'
    report['official_holder_validation']={'status':'UNVERIFIED','pipeline_buffer_model_known':False,
        'anchor_boundary_treatment':'NOT_DISCLOSED','local_failure_is_official_rejection':False}
    report["zip"]=output.build_zip(ctx)
    output.write_report(ctx,report)
    return report


def self_check(ctx):
    data=prepare_data(cutoff="2025-06-03 23:45:00")
    base=build_features(data)
    t=data.index[-97]
    altered=data.prefix(data.index[-1])
    cols=[x for x in altered.frame if x.startswith("gas_")]
    altered.frame.loc[altered.index>t,cols]=0.
    changed=build_features(altered)
    assert np.allclose(base.loc[:t],changed.loc[:t],equal_nan=True)
    cutoff=data.index[-192]
    masked=prediction_view(data,cutoff)
    assert np.allclose(base,build_features(masked),equal_nan=True)
    assert masked.labels.loc[cutoff:].isna().all().all()
    assert masked.labels.loc[:cutoff-pd.Timedelta(minutes=15)].equals(data.labels.loc[:cutoff-pd.Timedelta(minutes=15)])
    assert data.labels.loc[cutoff:].notna().any().any()
    assert not any("holder" in x or "generator" in x or "load_" in x for x in base.columns)
    observed_score=score_resources([20.,100.],[100.,100.])
    missing_score=score_resources([np.nan,100.],[100.,100.])
    assert observed_score['wape']==.4 and missing_score['wape'] is None
    assert not missing_score['score_valid'] and missing_score['cells']==2 and missing_score['missing_prediction_cells']==1
    zero_score=score_resources([10.,100.],[0.,100.])
    assert zero_score['wape']==.1 and zero_score['zero_false_volume']==2.5
    audit_times=data.index[-4:-2]
    audit_truth=pd.DataFrame({'budget_blast':[100.,100.]},index=audit_times)
    partial=[{'times':[audit_times[0]],'components':{'budget_blast':[100.]}},
             {'times':[audit_times[1]],'components':{}}]
    audit=audit_resource_forecasts(partial,audit_truth)['budget_blast']
    assert audit['wape'] is None and audit['missing_prediction_cells']==1
    bounded=audit_resource_forecasts(partial,audit_truth,window_end=audit_times[0])['budget_blast']
    assert bounded['wape']==0. and bounded['cells']==1 and bounded['missing_truth_cells']==1
    tiny_origins=data.index[-3:-1]
    limited=truth_cube(data,tiny_origins,[1,2],window_end=tiny_origins[-1])
    assert np.isnan(limited[0,1]).all() and np.isnan(limited[1]).all()
    price_day={'issued_at':'2025-06-01 07:00:00','prices':[1.,3.],
               'power':np.array([[50.,0.,0.,0.,0.,0.],[100.,0.,0.,0.,0.,0.]])}
    assert np.isclose(price_weighted_load([price_day])['ratio'],350./300.)
    sample=project_load(np.array([[-10.,300.],[210.,0.],[100.,150.],[0.,0.]]))
    assert (sample[:,0]>=0).all() and (sample[:,0]<=200).all() and (sample[:,1]>=sample[:,0]).all()
    # 分组份额校准：高负荷段被"大机组满载+峰值松弛"下界抬起，低负荷段按目标份额，关闭时逐值不变。
    # S2-G 配方（gall x0.975、short 0.3231、long 0.3128、松弛 8MW）下的期望值：
    # short: 200->0.3231*200=64.62（下界0）；300->96.93（下界76 不绑定）；440->clip(142.164,216,200)=200。
    probe=np.array([[[80.,200.],[120.,300.],[200.,440.]]])
    cal_s,cal_l=main_s.calibrate_group_share(probe,probe,C.SHARE_CALIBRATION)
    assert np.allclose(cal_s[:,:,0],[64.62,96.93,200.],atol=1e-6)
    assert (cal_s[:,:,0]<=cal_s[:,:,1]).all() and (cal_s[:,:,1]<=cal_s[:,:,0]+240.).all()
    off_s,off_l=main_s.calibrate_group_share(probe,probe,{"enabled":False})
    assert np.allclose(off_s,probe) and np.allclose(off_l,probe)
    inputs={"available":np.tile([450000.,450000.],(96,1)),"split":np.tile([.5,.5],(96,1)),
            "optional_caps":np.zeros((96,2)),"budget_blast":np.full(96,900000.)}
    previous=np.array([30.,30.,30.,30.,75.,75.])
    beta=np.array([.0003,.001,.0006])
    stock=np.array([150000.,100000.])
    # 历史组合可能未出现单一气种；合法可用量不应被经验最低份额拒绝。
    support={'available':True,'equations':[[-1.,0.,.02],[0.,-1.,.02],[1.,1.,-.15]],
        'source':'synthetic_historical_share','train_end':'2025-05-31 23:45:00'}
    # 核验仍按聚合硬上限 30/66；正常计划生成沿用 30MW 小机组边界（不排除合法启停），
    # 总负荷正常生成使用选定回档配方的 60MW 裕度。
    assert C.POLICY['ramp']==(30.,66.) and C.POLICY['generate_ramp']==(30.,60.)
    ramp_issue=pd.Timestamp('2025-06-01 07:00:00')
    ramp_prior=np.array([30.,30.,30.,30.,72.,72.])
    ramp_power=np.tile([37.5,37.5,37.5,37.5,90.,90.],(96,1))
    ramp_stock=np.tile(stock,(97,1))
    ramp_day={'gas':np.tile([1100000.,0.,0.],(96,1)),'power':ramp_power,
        'previous_power':ramp_prior,'available':np.tile([550000.,550000.],(96,1)),
        'split':np.tile([.5,.5],(96,1)),'stocks':ramp_stock,'vent':np.zeros((96,2)),
        'beta':beta,'anchor':stock,'anchor_observed':True,
        'initial_state':{'basis':'synthetic_observed_unit_state'},'resource_mode':'net_inferred',
        'issued_at':str(ramp_issue),'max_observed_at':str(ramp_issue),
        'times':pd.date_range(ramp_issue+pd.Timedelta(minutes=15),periods=96,freq='15min'),
        'optional_caps':np.zeros((96,2)),
        'bridge':{'gas_m3h':[1100000.,0.,0.],'available_m3h':[550000.,550000.],'split':[.5,.5]}}
    ramp_check=validate_day(ramp_day)
    assert ramp_check['passed'] and np.isclose(ramp_check['details']['max_aggregate_change_mw'][1],66.)
    from copy import deepcopy
    extended=deepcopy(ramp_day)
    for key in ('gas','power','available','split','vent','optional_caps'):
        extended[key]=np.tile(np.asarray(ramp_day[key])[0],(163,1))
    extended['stocks']=np.tile(stock,(164,1))
    extended['times']=pd.date_range(ramp_issue+pd.Timedelta(minutes=15),periods=163,freq='15min')
    assert validate_day(extended)['passed']
    broken=deepcopy(extended);broken['power'][96,0]+=10.
    assert not validate_day(broken)['passed']
    solved,solver=main_opt.solve_milp(inputs,previous,beta,stock,np.full(96,.53),
                                      support=support,scenarios=[beta*10.])
    if solved is None:
        raise AssertionError(solver)
    day=dict(inputs,**solved,previous_power=previous,beta=beta,anchor=stock,anchor_observed=True,
             initial_state={"basis":"synthetic_observed_unit_state"},
             resource_mode="net_inferred",issued_at="2025-06-01 07:00:00",max_observed_at="2025-06-01 07:00:00",
             bridge={"gas_m3h":[900000.,0.,0.],"available_m3h":[450000.,450000.]})
    check=validate_day(day)
    if not check["passed"]:
        raise AssertionError(check)
    off_inputs={"available":np.tile([150000.,150000.],(96,1)),"split":np.tile([.5,.5],(96,1)),
        "optional_caps":np.zeros((96,2)),"budget_blast":np.full(96,300000.),"previous_gas":np.zeros(3)}
    off,off_solver=main_opt._solve_joint(off_inputs,np.zeros(6),beta,stock,np.full(96,.53))
    if off is None:raise AssertionError('全停机组应可在30MW组爬坡上限内合法启动')
    off_day=dict(off_inputs,**off,previous_power=np.zeros(6),beta=beta,
        anchor=stock-C.DT*off_inputs['available'][0],anchor_observed=True,
        initial_state={'basis':'synthetic_observed_all_units_off'},resource_mode='net_inferred',
        issued_at='2025-06-01 07:00:00',max_observed_at='2025-06-01 07:00:00',
        bridge={'gas_m3h':np.zeros(3),'available_m3h':off_inputs['available'][0],'split':off_inputs['split'][0]})
    assert validate_day(off_day)['passed'] and (off['power'].sum(axis=1)>=30.-.001).any()
    overflow_inputs={'available':np.tile([0.,10000.],(96,1)),'split':np.tile([1.,0.],(96,1)),
        'optional_caps':np.zeros((96,2)),'budget_blast':np.zeros(96),'previous_gas':np.zeros(3)}
    overflow_stock=np.array([150000.,179000.])
    overflow,overflow_solver=main_opt._solve_joint(overflow_inputs,np.zeros(6),beta,overflow_stock,
                                                   np.zeros(96),recovery_export=True)
    if overflow is None:raise AssertionError('恢复分支应允许超过规划上限后按90%自动放散')
    replay_stock,replay_vent=main_opt._replay_stocks(overflow_stock,overflow_inputs,overflow['gas'])
    assert np.allclose(overflow['stocks'],replay_stock,atol=C.POLICY['stock_tolerance'],rtol=0.)
    assert np.allclose(overflow['vent'],replay_vent,atol=C.POLICY['gas_tolerance'],rtol=0.)
    assert np.isclose(overflow['stocks'][1,1],180000.) and overflow['vent'][0,1]>0.
    overflow_day=dict(overflow_inputs,**overflow,previous_power=np.zeros(6),beta=beta,
        anchor=overflow_stock-C.DT*overflow_inputs['available'][0],anchor_observed=True,
        initial_state={'basis':'synthetic_observed_all_units_off'},resource_mode='net_inferred',
        issued_at='2025-06-01 07:00:00',max_observed_at='2025-06-01 07:00:00',fallback_used=True,
        bridge={'gas_m3h':np.zeros(3),'available_m3h':overflow_inputs['available'][0],'split':[1.,0.]})
    overflow_check=validate_day(overflow_day)
    assert not overflow_check['passed'] and overflow_check['checks']['independent_holder_safety']
    assert overflow_check['checks']['holder_balance'] and overflow_check['checks']['overflow_rule']
    assert not overflow_check['checks']['planning_bounds']
    zero_view=data.prefix(data.index[-1]);zero_view.frame=zero_view.frame.tail(97).copy()
    zero_view.labels=zero_view.labels.reindex(zero_view.index)
    gas_columns=[name for name in zero_view.frame if name.startswith(('gas_','gas_user_'))]
    zero_view.frame.loc[:,gas_columns]=0.
    zero_view.frame.loc[:,list(C.USE)]=0.
    zero_view.frame.loc[:,list(C.HOLDERS)]=[150000.,100000.]
    zero_inputs=dict(inputs,resource_mode='net_inferred',
        times=pd.date_range(zero_view.index[-1]+pd.Timedelta(minutes=15),periods=96,freq='15min'))
    zero_supply,_,zero_source=main_opt.bridge_resources(main_opt.ResourceBundle(),zero_view,zero_view.index[-1],zero_inputs)
    assert np.array_equal(zero_supply,np.zeros(2)) and zero_source['estimated_furnaces']==[]
    from copy import deepcopy
    controlled=deepcopy(day)
    controlled['prices']=np.full(96,.53)
    controlled['previous_gas']=np.array([900000.,0.,0.])
    controlled['control_calibration']={'version':C.DISPATCH_CONTROL['version'],'train_end':'2025-05-31 23:45:00',
        'optional_use_calibration':deepcopy(C.OPTIONAL_USE_CALIBRATION)}
    optional_pred={'budget_converter':np.full(96,25000.)}
    trace=main_opt.calibrate_optional_prefix(optional_pred,zero_view,zero_view.index[-1])
    assert trace['budget_converter']['applied'] and trace['budget_converter']['observed_median_m3h']==0.
    controlled['issued_at']=controlled['max_observed_at']=str(zero_view.index[-1])
    controlled['components']=optional_pred
    controlled['optional_caps']=np.column_stack((np.zeros(96),optional_pred['budget_converter']))
    controlled['optional_consumption_calibration']=trace
    missing=zero_view.prefix(zero_view.index[-1]);missing.frame.loc[missing.index[-1],C.USE[2]]=np.nan
    unchanged={'budget_converter':np.full(96,25000.)}
    absent=main_opt.calibrate_optional_prefix(unchanged,missing,missing.index[-1])
    assert not absent['budget_converter']['applied'] and np.array_equal(unchanged['budget_converter'],np.full(96,25000.))
    # 收益优先配方不生成电价迭代记录；核验器用例显式声明收益底线，独立于当前配方。
    controlled['joint_control']['price_frontier']={
        'benefit_floor_cny':float(controlled['prices']@controlled['power'].sum(axis=1)*1000*C.DT)}
    if not validate_day(controlled)['passed']:
        raise AssertionError(validate_day(controlled))
    assert validate_day(controlled)['details']['historical_gas_support']['outside_steps']==96
    assert validate_day(controlled)['details']['mapping_sensitivity']['mapped_capacity_exceedances']>0
    controlled['joint_control']['reserve_risk']={'target_m3':[150000.,100000.],
        'max_observed_at':controlled['issued_at'],
        'scenario':{'history_end':controlled['issued_at'],'coverage_guarantee':False}}
    assert validate_day(controlled)['checks']['reserve_risk_causality']
    future_risk=deepcopy(controlled)
    future_risk['joint_control']['reserve_risk']['scenario']['history_end']=str(
        pd.Timestamp(controlled['issued_at'])+pd.Timedelta(minutes=15))
    assert not validate_day(future_risk)['checks']['reserve_risk_causality']
    bad_optional=deepcopy(controlled);bad_optional['optional_caps'][0,1]+=10.
    assert not validate_day(bad_optional)['checks']['optional_calibration_fidelity']
    future_optional=deepcopy(controlled)
    future_optional['optional_consumption_calibration']['budget_converter']['observed_through']='2100-01-01'
    assert not validate_day(future_optional)['checks']['optional_calibration_causality']
    controlled['process_resource']={'applied':True,'train_end':'2025-05-31 23:45:00',
        'max_observed_at':controlled['issued_at'],'missing_gas_is_zero':False,'raw_process_values_overwritten':False}
    assert validate_day(controlled)['passed']
    invalid_process=deepcopy(controlled)
    invalid_process['process_resource']['max_observed_at']=str(pd.Timestamp(controlled['issued_at'])+pd.Timedelta(minutes=15))
    assert not validate_day(invalid_process)['checks']['process_resource_causality']
    damaged=deepcopy(controlled)
    damaged['joint_control']['price_frontier']['benefit_floor_cny']=1e12
    assert not validate_day(damaged)['checks']['price_benefit_floor']
    boundary=deepcopy(day)
    prior_stock=float(boundary['stocks'][0,0])
    boundary['anchor'][0]=26388.22
    boundary['stocks'][0,0]=19215.77
    boundary['available'][0,0]+=(prior_stock-19215.77)/C.DT
    boundary['bridge']['available_m3h'][0]=450000.+(19215.77-26388.22)/C.DT
    boundary_check=validate_day(boundary)
    stages=boundary_check['details']['holder_boundary_diagnostics']
    assert not boundary_check['passed'] and stages['anchor_unsafe_holders']==[1]
    assert stages['pre_command_unsafe_holders']==[1] and stages['post_execution_unsafe_steps']==0
    assert boundary_check['checks']['bridge_balance'] and boundary_check['checks']['holder_balance']
    assert boundary_check['assessment']['post_execution_holder_safety_passed']
    assert boundary_check['assessment']['official_anchor_boundary_treatment']=='NOT_DISCLOSED'
    assert not boundary_check['assessment']['local_failure_is_official_rejection']
    mapping={"trusted":True,"train_end":"2025-05-31 23:45:00",
             "coefficients_mw_per_m3h":[.00015,0.,0.]}
    refined=main_opt.refine_group_day(day,mapping,smooth=C.GROUP_DISPATCH["smooth_penalty"])
    if refined is None:
        raise AssertionError("分组再分配合成用例无解")
    assert np.array_equal(refined["gas"],day["gas"])
    assert np.allclose(refined["power"].sum(axis=1),day["power"].sum(axis=1),atol=1e-6)
    assert np.array_equal(refined["power"][-1],day["power"][-1])
    day["stocks"]=solved["stocks"].copy()
    day["stocks"][4,0]+=100.
    assert not validate_day(day)["checks"]["holder_balance"]
    report={"status":"SELF_CHECK_PASSED","features":len(base.columns),"checks":[
        "future_perturbation","target_and_gas_blackout","no_holder_load_features","capacity_projection",
        "resource_missing_prediction_rejected","resource_missing_day_rejected","resource_zero_truth_false_volume","historical_target_window_end",
        "independent_price_weighted_load",
        "synthetic_milp","independent_balance_mutation","group_refinement_fixed_resources_and_terminal",
        "joint_control_validation","independent_benefit_floor_mutation","initial_boundary_failure_attribution",
        "mapping_sensitivity_diagnostic_only","legal_66mw_aggregate_ramp","recovery_ramp_66_generation_ramp_60",
        "reserve_risk_future_closure_rejected","recovery_automatic_overflow_and_failure_reporting",
        "process_resource_causality_mutation","no_fixed_gas_ratio","local_initial_condition_vs_official_unknown",
        "all_off_to_minimum_unit_startup","observed_zero_bridge_not_replaced_by_forecast",
        "optional_prefix_zero_vs_missing","optional_calibration_value_and_future_mutation"],
        "solver":solver,"synthetic_validation":check,"group_refinement":refined["refinement"]}
    output.assert_source(ctx)
    output.write_report(ctx,report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description="AIC半决赛：独立LightGBM与双柜每日MILP")
    p.add_argument("command",nargs="?",default="infer",choices=("infer","opt","run","doctor","self-check","verify"))
    p.add_argument("--quick",action="store_true")
    p.add_argument("--experiment",action="store_true")
    p.add_argument("--reuse-run",type=Path)
    p.add_argument("--train-bootstrap",action="store_true",help="首次生成2111行时，显式训练首端预案所需的更早资源模型")
    p.add_argument("--run-dir",type=Path)
    p.add_argument("--run-id")
    p.add_argument("--threads",type=int,default=C.CPU_THREADS,help="CPU线程总预算，默认16，避免嵌套过量并行")
    args=p.parse_args(argv)
    if not 1<=args.threads<=min(16,__import__("os").cpu_count() or 1):
        p.error("线程预算必须在1至本机可用16线程范围内")
    C.CPU_THREADS=args.threads
    C.TREE["num_threads"]=args.threads
    from threadpoolctl import threadpool_limits
    thread_limits=threadpool_limits(limits=args.threads)
    if args.command=="doctor":
        d=prepare_data()
        print(json.dumps(output.clean_json({"schema":C.SCHEMA,"quality":d.quality,"sources":d.sources,
              "features":len(build_features(d).columns),"nn_enabled":False}),ensure_ascii=False,indent=2))
        return 0
    if args.command=="verify":
        if args.run_dir is None:
            p.error("verify必须指定--run-dir")
        result=output.verify_run(args.run_dir)
        print(json.dumps(output.clean_json(result),ensure_ascii=False,indent=2))
        return 0 if result["archive_passed"] and result["submittable_claim_consistent"] else 2
    if args.command=="opt" and args.reuse_run is None:
        p.error("opt必须显式指定--reuse-run")
    if args.reuse_run and args.command not in ("infer","opt"):
        p.error("--reuse-run仅适用于infer/opt")
    experimental=args.quick or args.experiment or args.command in ("run","self-check")
    ctx=output.begin_run(args.command,experimental,args.run_id)
    setup_log(ctx)
    LOG.info("运行目录：%s；CPU线程预算=%s",ctx["root"],C.CPU_THREADS)
    try:
        if args.command=="self-check":
            report=self_check(ctx)
        elif args.command=="run":
            report=run_backtest(ctx,args.quick)
        else:
            report=infer(ctx,args.quick,args.reuse_run,args.command=="opt",args.train_bootstrap)
        LOG.info("完成：%s；本地可提交=%s",report["status"],report.get("submittable",False))
        return 2 if report.get("status")=="COMPLETE_WITH_CONSTRAINT_FAILURES" else 0
    except Exception as exc:
        LOG.exception("运行失败")
        previous=ctx["root"]/"报告"/"运行结果.json"
        failed=output.read_json(previous) if previous.is_file() else {}
        failed.update(status="FAILED",run_id=ctx["run_id"],error=str(exc),submittable=False)
        output.write_report(ctx,failed)
        return 1


if __name__=="__main__":
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
