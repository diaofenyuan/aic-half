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
from main_s import LoadBundle
from main_opt import ResourceBundle
from data_cleaning.pipeline import file_hash
from data_cleaning.quality import validate_result_files, validate_day


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


def save_models(root,short,long,resource,contract=None):
    root=Path(root)
    meta={"schema":C.SCHEMA,"holder_input_version":C.HOLDER_INPUT_VERSION,
          "targets":C.TARGETS,"step_minutes":15,"load":{},"resource":{},
          "preprocessing_sha256":file_hash(C.CODE_ROOT/"data_cleaning/pipeline.py"),
          "training_quick":bool(short.recipe.get("effective_budget",{}).get("quick",False)),
          "runtime":{name:importlib.metadata.version(name) for name in ("numpy","pandas","lightgbm","scipy","scikit-learn")},
          "contract":contract or {}}
    for b in (short,long):
        folder=root/b.span
        folder.mkdir(parents=True,exist_ok=True)
        records=[]
        for number,(ti,group,model) in enumerate(b.models):
            name=f"tree_{number:03d}.txt"
            (folder/name).write_text(model.model_to_string(),encoding="utf-8")
            records.append({"target_index":ti,"horizons":group,"file":f"{b.span}/{name}","sha256":file_hash(folder/name)})
        meta["load"][b.span]={"features":b.features,"horizons":b.horizons,"recipe":b.recipe,
                              "train_end":b.train_end,"models":records,"diagnostics":b.diagnostics}
    folder=root/"resource"
    folder.mkdir(parents=True,exist_ok=True)
    records={}
    for name,model in resource.models.items():
        path=folder/(name+".txt")
        path.write_text(model.model_to_string(),encoding="utf-8")
        records[name]={"file":f"resource/{name}.txt","sha256":file_hash(path)}
    meta["resource"]={"features":resource.features,"train_end":resource.train_end,"beta":resource.beta,
                       "diagnostics":resource.diagnostics,"models":records}
    write_json(root/"model_meta.json",meta)
    return meta


def load_models(run):
    root=Path(run)/"模型"
    meta=read_json(root/"model_meta.json")
    if meta.get("schema")!=C.SCHEMA or meta.get("holder_input_version")!=C.HOLDER_INPUT_VERSION:
        raise ValueError("模型版本或单位契约不兼容；旧模型禁止自动迁移")
    if meta.get("preprocessing_sha256") not in (None,file_hash(C.CODE_ROOT/"data_cleaning/pipeline.py")):
        raise ValueError("预处理实现已改变，禁止以新特征变换加载旧模型")
    bundles=[]
    for span in ("short","long"):
        m=meta["load"][span]
        b=LoadBundle(span,m["horizons"],m["features"],recipe=m["recipe"],train_end=m["train_end"],diagnostics=m["diagnostics"])
        expected=list(range(1,9 if span=="short" else 97))
        if b.horizons!=expected:
            raise ValueError("模型提前量契约错误")
        covered=np.zeros((len(expected),2),int)
        for record in m["models"]:
            path=(root/record["file"]).resolve()
            if not path.is_relative_to(root.resolve()) or file_hash(path)!=record["sha256"]:
                raise ValueError("模型文件路径或完整性错误")
            model=lgb.Booster(model_str=path.read_text(encoding="utf-8"))
            if model.feature_name()!=b.features:
                raise ValueError("模型内特征顺序与元数据不一致")
            b.models.append((record["target_index"],record["horizons"],model))
            covered[np.array(record["horizons"])-1,record["target_index"]]+=1
        needed=np.asarray(m["recipe"].get("blend_weights",{}).get(span,[1.,1.]))>0.
        if not (covered==needed[None,:].astype(int)).all():
            raise ValueError("模型分量覆盖缺失或重复")
        bundles.append(b)
    r=meta["resource"]
    if "group_mapping" not in r["diagnostics"]:
        raise ValueError("新版归档必须包含分组气电标定和留出验证")
    resource=ResourceBundle(features=r["features"],train_end=r["train_end"],beta=r["beta"],diagnostics=r["diagnostics"])
    for name,record in r["models"].items():
        path=(root/record["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path)!=record["sha256"]:
            raise ValueError("资源模型文件不兼容")
        model=lgb.Booster(model_str=path.read_text(encoding="utf-8"))
        if model.feature_name()!=resource.features:
            raise ValueError("资源特征契约错误")
        resource.models[name]=model
    return (*bundles,resource,meta)


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
        for name in ("s_result.csv","l_result.csv","opt_result.csv"):
            z.write(ctx["root"]/name,name)
        for folder,arc in ((ctx["source"],"源码"),(ctx["root"]/"模型","模型")):
            for p in sorted(folder.rglob("*")):
                if p.is_file():
                    z.write(p,str(Path(arc)/p.relative_to(folder)).replace("\\","/"))
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
    return {"path":str(target),"crc_ok":bad is None,"bytes":target.stat().st_size}


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
        evidence.extend(np.column_stack((q,p[:,:4].sum(axis=1),p.sum(axis=1))).tolist())
    linked=len(evidence)==len(exported) and np.allclose(exported.iloc[:,1:].to_numpy(),np.array(evidence),atol=.000002,rtol=0.)
    actual=bool(days) and all(x["passed"] for x in constraints)
    metadata=load_models(root)[-1]
    fingerprint=report.get("source",{})
    zip_ok=False
    with zipfile.ZipFile(root/"咕咕嘎嘎-半决赛.zip") as z:
        names=z.namelist()
        zip_ok=z.testzip() is None and not any(n.startswith(("data/","tmp/")) for n in names)
        for name,digest in fingerprint.items():
            if "源码/"+name not in names:
                zip_ok=False
                continue
            import hashlib
            zip_ok &= hashlib.sha256(z.read("源码/"+name)).hexdigest()==digest
        for name in ("s_result.csv","l_result.csv","opt_result.csv"):
            zip_ok &= z.read(name)==(root/name).read_bytes()
        for path in (root/"模型").rglob("*"):
            if path.is_file():
                name="模型/"+path.relative_to(root/"模型").as_posix()
                zip_ok &= name in names and z.read(name)==path.read_bytes()
    return {"archive_passed":bool(files["passed"] and linked and zip_ok),
            "constraints_passed":actual,"files":files,"csv_evidence_equal":linked,
            "submittable_claim_consistent":bool(report.get("submittable")==(
                actual and files["passed"] and not report["experimental"] and report.get("full_window",False))),
            "models_schema":metadata["schema"],"daily_constraints":constraints}
