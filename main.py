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
from data_cleaning.quality import (truth_cube, score_load, score_horizons, validate_day,
                                   validate_result_files, historical_replay, project_load, audit_resource_forecasts)

LOG=logging.getLogger("aic-half")


def setup_log(ctx):
    LOG.handlers.clear()
    LOG.setLevel(logging.INFO)
    formatter=logging.Formatter("%(asctime)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout),logging.FileHandler(ctx["root"]/"报告"/"运行日志.log",encoding="utf-8")):
        handler.setFormatter(formatter)
        LOG.addHandler(handler)


def evaluate_load(data,origins,recipe,quick=False):
    start=pd.DatetimeIndex(origins)[0]
    train=data.prefix(start-pd.Timedelta(minutes=15))
    train.require_training(start-pd.Timedelta(minutes=15))
    short=main_s.train_short(train,recipe,quick,LOG.info)
    long=main_l.train_long(train,recipe,quick,LOG.info)
    view=data.masked_copy(start,pd.DatetimeIndex(origins)[-1]+pd.Timedelta(days=1),mask_use=True)
    ps=main_s.predict_short(short,view,origins)
    pl=main_l.predict_long(long,view,origins)
    ys,yl=truth_cube(data,origins,range(1,9)),truth_cube(data,origins,range(1,97))
    ns=main_s.naive_profile(train,origins,range(1,9),train.index[-1])
    nl=main_s.naive_profile(train,origins,range(1,97),train.index[-1])
    metrics={"short":score_load(ps,ys),"long":score_load(pl,yl),
             "short_horizons":score_horizons(ps,ys),"long_horizons":score_horizons(pl,yl),
             "naive_short":score_load(ns,ys),"naive_long":score_load(nl,yl),
             "max_training_target":str(train.index[-1]),"information":"process_only;21day_target_blackout"}
    return short,long,ps,pl,metrics


def run_backtest(ctx,quick=False):
    data=prepare_data()
    metrics=[]
    for spec in C.FOLDS[:2] if quick else C.FOLDS:
        origins=pd.date_range(spec["start"],spec["end"],freq="15min")
        LOG.info("历史折 %s：连续21天，开始新模型训练",spec["name"])
        short,long,ps,pl,score=evaluate_load(data,origins,C.LOAD_RECIPE,quick)
        train=data.prefix(origins[0]-pd.Timedelta(minutes=15))
        resource=main_opt.fit_resources(train,quick,LOG.info)
        view=data.masked_copy(origins[0],origins[-1]+pd.Timedelta(days=1),mask_use=True)
        schedule,days,opt=main_opt.optimize_daily(resource,view,origins,quick,LOG.info)
        replay=historical_replay(data,days)
        metrics.append({"fold":spec["name"],**score,"optimization":opt,"historical_replay":replay,
                        "resource_validation":audit_resource_forecasts(days,main_opt.resource_labels(data))})
        fold_dir=ctx["root"]/spec["name"]
        (fold_dir/"模型").mkdir(parents=True)
        output.save_models(fold_dir/"模型",short,long,resource,{"role":"development","source":ctx["hashes"]})
        output.write_results(fold_dir,origins,ps,pl,schedule)
        restored_s,restored_l,_,_=output.load_models(fold_dir)
        probe=origins[::251]
        reload_error={"short":float(np.max(np.abs(main_s.predict_short(restored_s,view,probe)-ps[::251]))),
                      "long":float(np.max(np.abs(main_l.predict_long(restored_l,view,probe)-pl[::251])))}
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
    return differences


def infer(ctx,quick=False,reuse=None,opt_only=False):
    tick=time.perf_counter()
    if reuse:
        short,long,resource,meta=output.load_models(reuse)
        if not ctx["experimental"] and (meta.get("training_quick") or meta.get("contract",{}).get("role")!="final"):
            raise ValueError("正式运行不得复用quick或历史开发模型")
        LOG.info("严格复用新版兼容模型，不训练或重新标定：%s",reuse)
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
    view=data.masked_copy(C.TEST_START,C.TEST_END,mask_use=True)
    reload=reload_check(ctx,short,long,resource,view,origins,meta.get("contract"))
    copied_prediction_check=None
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
        source_report=output.read_json(source/"报告"/"运行结果.json")
        copied_prediction_check=output.verify_prediction_replay(ctx["root"],source_report,origins,short,long)
        if not copied_prediction_check["passed"]:
            raise ValueError("复用预测与归档模型或原始输入不一致，拒绝生成调度结果："+
                json.dumps(output.clean_json(copied_prediction_check),ensure_ascii=False))
        ps=pl=None
    else:
        LOG.info("推理：%s个起报点；测试标签不计算本地分数",len(origins))
        ps=main_s.predict_short(short,view,origins)
        pl=main_l.predict_long(long,view,origins)
    schedule,days,opt=main_opt.optimize_daily(resource,view,origins,quick,LOG.info)
    if opt_only:
        s=schedule.copy()
        s.datetime=pd.to_datetime(s.datetime).dt.strftime("%Y-%m-%d %H:%M:%S")
        s.to_csv(ctx["root"]/"opt_result.csv",index=False,float_format="%.6f",encoding="utf-8")
    else:
        output.write_results(ctx["root"],origins,ps,pl,schedule)
    coverage=validate_result_files(ctx["root"],origins,schedule.datetime)
    if not coverage["passed"]:
        raise ValueError("三表导出契约未通过："+json.dumps(coverage,ensure_ascii=False))
    full=len(origins)==2016 and len(schedule)==2016 and len(days)==21
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
            "upload_ready":coverage["passed"],"training_performed":not bool(reuse),"prediction_performed":not opt_only,
            "reused_models_from":str(reuse) if reuse else None,"recipe":short.recipe,
            "prediction_grid":[str(origins[0]),str(origins[-1])],
            "optimization_grid":[str(schedule.datetime.iloc[0]),str(schedule.datetime.iloc[-1])],
            "quality":data.quality,"data_sources":data.sources,"source":ctx["hashes"],
            "reload_max_abs_difference":reload,"optimization":opt,"daily_plans":days,
            "prediction_integrity_replay_on_reuse":copied_prediction_check,
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
    masked=data.masked_copy(data.index[-192],mask_use=True)
    assert np.allclose(base,build_features(masked),equal_nan=True)
    assert not any("holder" in x or "generator" in x or "load_" in x for x in base.columns)
    sample=project_load(np.array([[-10.,300.],[210.,0.],[100.,150.],[0.,0.]]))
    assert (sample[:,0]>=0).all() and (sample[:,0]<=200).all() and (sample[:,1]>=sample[:,0]).all()
    inputs={"available":np.tile([450000.,450000.],(96,1)),"split":np.tile([.5,.5],(96,1)),
            "optional_caps":np.zeros((96,2)),"budget_blast":np.full(96,900000.)}
    previous=np.array([30.,30.,30.,30.,75.,75.])
    beta=np.array([.0003,.001,.0006])
    stock=np.array([150000.,100000.])
    # 历史组合可能未出现单一气种；合法可用量不应被经验最低份额拒绝。
    support={'available':True,'equations':[[-1.,0.,.02],[0.,-1.,.02],[1.,1.,-.15]],
        'source':'synthetic_historical_share','train_end':'2025-05-31 23:45:00'}
    solved,solver=main_opt.solve_milp(inputs,previous,beta,stock,np.full(96,.53),support=support)
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
        raise AssertionError('联合控制独立核验未通过')
    assert validate_day(controlled)['details']['historical_gas_support']['outside_steps']==96
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
        "synthetic_milp","independent_balance_mutation","group_refinement_fixed_resources_and_terminal",
        "joint_control_validation","independent_benefit_floor_mutation","initial_boundary_failure_attribution",
        "process_resource_causality_mutation","no_fixed_gas_ratio","local_initial_condition_vs_official_unknown",
        "all_off_to_minimum_unit_startup","observed_zero_bridge_not_replaced_by_forecast",
        "optional_prefix_zero_vs_missing","optional_calibration_value_and_future_mutation"],
        "solver":solver,"synthetic_validation":check,"group_refinement":refined["refinement"]}
    output.write_report(ctx,report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description="AIC半决赛：独立LightGBM与双柜每日MILP")
    p.add_argument("command",nargs="?",default="infer",choices=("infer","opt","run","doctor","self-check","verify"))
    p.add_argument("--quick",action="store_true")
    p.add_argument("--experiment",action="store_true")
    p.add_argument("--reuse-run",type=Path)
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
            report=infer(ctx,args.quick,args.reuse_run,args.command=="opt")
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
