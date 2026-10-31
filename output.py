"""实际部署模型、冻结源码与三表归档；提交资格不与文件完整性混同。"""
from pathlib import Path
import json
import shutil
import re
import zipfile
import datetime
import importlib.metadata
import numpy as np
import pandas as pd
import lightgbm as lgb
import config as C
from main_s import LoadBundle, load_recipe, required_targets
from main_opt import ResourceBundle
from data_cleaning.pipeline import file_hash
from data_cleaning.quality import validate_result_files, validate_day, validate_plan_sequence


def clean_json(value):
    if isinstance(value,dict):
        return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value,(pd.Index,pd.Series)):
        return clean_json(value.tolist())
    if isinstance(value,(pd.Timestamp,Path,datetime.datetime)):
        return str(value)
    if isinstance(value,(np.integer,np.bool_)):
        return value.item()
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def begin_run(mode,experimental=True,run_id=None):
    rid=run_id or pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")+"_"+mode
    if not re.fullmatch(r"[A-Za-z0-9_-]+",rid):
        raise ValueError("运行标识只能包含字母、数字、下划线和连字符")
    root=(C.EXPERIMENT_ROOT if experimental else C.OUTPUT_ROOT)/rid
    if root.exists():
        raise ValueError(f"运行目录已存在，禁止覆盖：{root}")
    (root/"模型").mkdir(parents=True)
    (root/"报告").mkdir()
    source=(C.ROOT/"tmp_main"/rid/"源码") if experimental else (root/"源码")
    source.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name in C.SOURCE_FILES:
        src=C.CODE_ROOT/name
        if not src.is_file():
            raise ValueError(f"缺少源码：{src}")
        target=source/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,target)
        hashes[name]=file_hash(src)
    return {"run_id":rid,"root":root,"source":source,"hashes":hashes,
            "experimental":experimental,"mode":mode,"started":pd.Timestamp.now().isoformat()}


def assert_source(ctx):
    changed=[name for name,digest in ctx["hashes"].items() if file_hash(C.CODE_ROOT/name)!=digest or
             file_hash(ctx["source"]/name)!=digest]
    if changed:
        raise ValueError("运行期间源码变化，拒绝混装归档："+",".join(changed))


def _save_load_bundles(root,short,long):
    records_by_span={}
    for b in (short,long):
        if b.recipe.get("load_version")!=C.LOAD_VERSION:
            raise ValueError("负荷模型版本不兼容，禁止静默迁移")
        folder=root/b.span
        folder.mkdir(parents=True,exist_ok=True)
        records=[]
        for number,(ti,group,model) in enumerate(b.models):
            name=f"tree_{number:03d}.txt"
            (folder/name).write_text(model.model_to_string(),encoding="utf-8")
            records.append({"target_index":ti,"horizons":group,"file":f"{b.span}/{name}","sha256":file_hash(folder/name)})
        records_by_span[b.span]={"features":b.features,"horizons":b.horizons,"recipe":b.recipe,
                                "train_end":b.train_end,"models":records,"diagnostics":b.diagnostics}
    return records_by_span


def _load_bundles(root,meta):
    if meta.get("schema")!=C.SCHEMA or meta.get("load_version")!=C.LOAD_VERSION:
        raise ValueError("负荷模型版本不兼容；旧模型禁止自动迁移")
    if (meta.get("preprocessing_sha256")!=file_hash(C.CODE_ROOT/"data_cleaning/pipeline.py") or
        meta.get("load_implementation_sha256")!=file_hash(C.CODE_ROOT/"main_s.py")):
        raise ValueError("负荷特征或推理实现已改变，拒绝复用不兼容模型")
    bundles=[]
    for span in ("short","long"):
        m=meta["load"][span]
        b=LoadBundle(span,m["horizons"],m["features"],recipe=m["recipe"],train_end=m["train_end"],diagnostics=m["diagnostics"])
        expected=list(range(1,9 if span=="short" else 97))
        if b.horizons!=expected or b.recipe.get("load_version")!=C.LOAD_VERSION:
            raise ValueError("负荷模型提前量或版本契约错误")
        pack=b.recipe.get("frozen_profile",{})
        if pack.get("cutoff")!=b.train_end:
            raise ValueError("负荷模型画像截止不一致")
        for key in ("values","shape"):
            a=np.asarray(pack.get(key,[]),float)
            if a.shape!=(96,3) or not np.isfinite(a).all() or (key=="shape" and (a<=0.).any()):
                raise ValueError("负荷画像或形状契约错误")
        covered=np.zeros((len(expected),3),int)
        for record in m["models"]:
            path=(root/record["file"]).resolve()
            if not path.is_relative_to(root.resolve()) or file_hash(path)!=record["sha256"]:
                raise ValueError("模型文件路径或完整性错误")
            ti,group=record["target_index"],record["horizons"]
            if ti not in (0,1,2) or not group or len(set(group))!=len(group) or not set(group)<=set(expected):
                raise ValueError("负荷模型目标或提前量错误")
            model=lgb.Booster(model_str=path.read_text(encoding="utf-8"))
            if model.feature_name()!=b.features:
                raise ValueError("模型内特征顺序与元数据不一致")
            b.models.append((ti,group,model))
            covered[np.array(group)-1,ti]+=1
        needed=np.zeros(3,int)
        needed[required_targets(load_recipe(b.recipe,span),span)]=1
        if not (covered==needed[None,:]).all():
            raise ValueError("模型分量覆盖缺失、冗余或重复")
        bundles.append(b)
    return bundles


def save_load_models(root,short,long,contract=None):
    """预测专项保存两条真实链路，不伪造资源模型或完整提交包。"""
    root=Path(root)
    meta={"schema":C.SCHEMA,"load_version":C.LOAD_VERSION,
          "preprocessing_sha256":file_hash(C.CODE_ROOT/"data_cleaning/pipeline.py"),
          "load_implementation_sha256":file_hash(C.CODE_ROOT/"main_s.py"),
          "targets":C.TARGETS,"step_minutes":C.STEP,"contract":contract or {},
          "load":_save_load_bundles(root,short,long)}
    write_json(root/"load_meta.json",meta)
    return meta


def load_load_models(root):
    root=Path(root)
    meta=read_json(root/"load_meta.json")
    return (*_load_bundles(root,meta),meta)


def save_models(root,short,long,resource,contract=None):
    root=Path(root)
    meta={"schema":C.SCHEMA,"load_version":C.LOAD_VERSION,"holder_input_version":C.HOLDER_INPUT_VERSION,
          "targets":C.TARGETS,"step_minutes":15,"load":{},"resource":{},
          "preprocessing_sha256":file_hash(C.CODE_ROOT/"data_cleaning/pipeline.py"),
          "load_implementation_sha256":file_hash(C.CODE_ROOT/"main_s.py"),
          "training_quick":bool(short.recipe.get("effective_budget",{}).get("quick",False)),
          "runtime":{name:importlib.metadata.version(name) for name in ("numpy","pandas","lightgbm","scipy","scikit-learn")},
          "contract":contract or {}}
    meta["load"]=_save_load_bundles(root,short,long)
    meta["resource"]=save_resource(root,resource)
    if resource.bootstrap is not None:
        meta['bootstrap_resource']=save_resource(root/'bootstrap',resource.bootstrap)
    write_json(root/"model_meta.json",meta)
    return meta


def load_models(run):
    root=Path(run)/"模型"
    meta=read_json(root/"model_meta.json")
    if meta.get("schema")!=C.SCHEMA or meta.get("holder_input_version")!=C.HOLDER_INPUT_VERSION:
        raise ValueError("模型版本或单位契约不兼容；旧模型禁止自动迁移")
    if meta.get("preprocessing_sha256") not in (None,file_hash(C.CODE_ROOT/"data_cleaning/pipeline.py")):
        raise ValueError("预处理实现已改变，禁止以新特征变换加载旧模型")
    bundles=_load_bundles(root,meta)
    resource=load_resource(root,meta['resource'])
    if 'bootstrap_resource' in meta:
        resource.bootstrap=load_resource(root/'bootstrap',meta['bootstrap_resource'])
    return (*bundles,resource,meta)



def save_resource(root,resource):
    root=Path(root)
    folder=root/"resource"
    folder.mkdir(parents=True,exist_ok=True)
    records={}
    for name,model in resource.models.items():
        path=folder/(name+".txt")
        path.write_text(model.model_to_string(),encoding="utf-8")
        records[name]={"file":f"resource/{name}.txt","sha256":file_hash(path)}
    record={"features":resource.features,"train_end":resource.train_end,"beta":resource.beta,
                       "diagnostics":resource.diagnostics,"models":records}
    from main_opt import validate_process_resource
    validate_process_resource(resource)
    record['process_meta']=resource.process_meta
    if resource.process_meta['available']:
        path=folder/'shared_process.txt'
        path.write_text(resource.process_model.model_to_string(),encoding='utf-8')
        record['process_model']={'file':'resource/shared_process.txt','sha256':file_hash(path)}
    return record


def load_resource(root,r):
    root=Path(root)
    if "group_mapping" not in r["diagnostics"]:
        raise ValueError("新版归档必须包含分组气电标定和留出验证")
    resource=ResourceBundle(features=r["features"],train_end=r["train_end"],beta=r["beta"],diagnostics=r["diagnostics"])
    from main_opt import validate_dispatch_calibration
    validate_dispatch_calibration(resource)
    resource.process_meta=r.get('process_meta',{})
    if resource.process_meta.get('available'):
        record=r.get('process_model',{})
        if not record.get('file'):raise ValueError('归档缺少资源过程模型文件')
        path=(root/record['file']).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path)!=record.get('sha256'):
            raise ValueError('资源过程模型路径或完整性错误')
        resource.process_model=lgb.Booster(model_str=path.read_text(encoding='utf-8'))
    from main_opt import validate_process_resource
    validate_process_resource(resource)
    for name,record in r["models"].items():
        path=(root/record["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path)!=record["sha256"]:
            raise ValueError("资源模型文件不兼容")
        model=lgb.Booster(model_str=path.read_text(encoding="utf-8"))
        if model.feature_name()!=resource.features:
            raise ValueError("资源特征契约错误")
        resource.models[name]=model
    return resource

def prediction_frame(origins,pred):
    out={"datetime":pd.DatetimeIndex(origins).strftime("%Y-%m-%d %H:%M:%S")}
    for ti,t in enumerate(C.TARGETS):
        for h in range(pred.shape[1]):
            out[f"{t}_t+{(h+1)*15}_pred"]=pred[:,h,ti]
    return pd.DataFrame(out)


def write_results(root,origins,short,long,schedule):
    root=Path(root)
    prediction_frame(origins,short).to_csv(root/"s_result.csv",index=False,float_format="%.6f",encoding="utf-8")
    prediction_frame(origins,long).to_csv(root/"l_result.csv",index=False,float_format="%.6f",encoding="utf-8")
    f=schedule.copy()
    f.datetime=pd.to_datetime(f.datetime).dt.strftime("%Y-%m-%d %H:%M:%S")
    f.to_csv(root/"opt_result.csv",index=False,float_format="%.6f",encoding="utf-8")


def write_report(ctx,report):
    write_json(ctx["root"]/"报告"/"运行结果.json",report)


def build_zip(ctx):
    assert_source(ctx)
    target=ctx["root"]/"咕咕嘎嘎-半决赛.zip"
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        # 评分服务现阶段仅接收三表，源码与模型继续保留在本地归档。
        for name in ("s_result.csv","l_result.csv","opt_result.csv"):
            z.write(ctx["root"]/name,name)
    checked=verify_zip(ctx["root"])
    if not checked["passed"]:
        raise ValueError("提交ZIP完整性或三表白名单校验未通过")
    return checked


def verify_zip(root):
    root=Path(root)
    target=root/"咕咕嘎嘎-半决赛.zip"
    with zipfile.ZipFile(target) as z:
        names=z.namelist()
        expected={"s_result.csv","l_result.csv","opt_result.csv"}
        members_ok=len(names)==len(expected) and set(names)==expected
        crc_ok=z.testzip() is None
        csv_equal=members_ok and all(z.read(name)==(root/name).read_bytes() for name in names)
    return {"path":str(target),"crc_ok":crc_ok,"bytes":target.stat().st_size,"members":names,
            "members_ok":members_ok,"csv_bytes_equal":csv_equal,
            "passed":bool(members_ok and crc_ok and csv_equal)}


def verify_run(root):
    root=Path(root)
    report=read_json(root/"报告"/"运行结果.json")
    origins=pd.date_range(report["prediction_grid"][0],report["prediction_grid"][1],freq="15min")
    effective=pd.date_range(report["optimization_grid"][0],report["optimization_grid"][1],freq="15min")
    files=validate_result_files(root,origins,effective)
    days=report.get("daily_plans",[])
    constraints=[validate_day(d) for d in days]
    exported=pd.read_csv(root/"opt_result.csv")
    evidence=[]
    for day in days:
        p,q=np.asarray(day["power"]),np.asarray(day["gas"])
        mask=np.asarray(day.get('export_mask',np.ones(len(q),bool)),bool)
        evidence.extend(np.column_stack((q,p[:,:4].sum(axis=1),p.sum(axis=1)))[mask].tolist())
    linked=len(evidence)==len(exported) and np.allclose(exported.iloc[:,1:].to_numpy(),np.array(evidence),atol=.000002,rtol=0.)
    sequence=validate_plan_sequence(days)
    actual=bool(days) and all(x["passed"] for x in constraints) and sequence["passed"]
    metadata=load_models(root)[-1]
    fingerprint=report.get("source",{})
    source=(C.ROOT/"tmp_main"/report["run_id"]/"源码") if report["experimental"] else root/"源码"
    source_ok=all((source/name).is_file() and file_hash(source/name)==digest
                  for name,digest in fingerprint.items())
    checked=verify_zip(root)
    if report.get('full_window'):
        files['configured_full_window']=bool(origins.equals(pd.date_range(C.TEST_START,C.TEST_END,freq='15min')) and
            effective.equals(pd.date_range(C.OPT_START,C.OPT_END,freq='15min')))
        files['passed'] &= files['configured_full_window']
    return {"archive_passed":bool(files["passed"] and linked and source_ok and checked["passed"]),
            "source_passed":source_ok,"zip":checked,
            "constraints_passed":actual,"files":files,"csv_evidence_equal":linked,
            "constraint_scope":"local_strict_project_model","official_holder_validation":"UNVERIFIED",
            "validation_scope":"project_local_strict_validation","official_acceptance_status":"unknown",
            "sequence_validation":sequence,
            "submittable_claim_consistent":bool(report.get("submittable")==(
                actual and files["passed"] and not report["experimental"] and report.get("full_window",False))),
            "models_schema":metadata["schema"],"daily_constraints":constraints}
