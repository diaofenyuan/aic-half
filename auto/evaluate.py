"""固定历史评测器：真值、评分与硬约束核验独立于候选代码。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def clean(value):
    import numpy as np
    import pandas as pd
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (pd.Index, pd.Series)):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".new")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def hash_array(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def artifact_manifest(folder):
    return {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob("*") if path.is_file() and path.relative_to(folder).parts[0] != "报告"
            and path.suffix != ".new"}


def validate_artifacts(folder, manifest):
    for name, expected in manifest.items():
        path = (folder / name).resolve()
        if (not path.is_relative_to(folder.resolve()) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError("保存产物已改变或缺失，拒绝复用：" + name)


def persist(bundle, folder, kind):
    """只保存本链路的真实分量，出口验证不要求训练另一条负荷模型。"""
    folder.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in vars(bundle).items() if k not in ("models", "process_model")}
    entries = []
    if kind == "load":
        for i, (target, horizons, model) in enumerate(bundle.models):
            name = f"tree_{i:03d}.txt"
            (folder / name).write_text(model.model_to_string(), encoding="utf-8")
            entries.append({"target": target, "horizons": horizons, "file": name})
    else:
        for key, model in bundle.models.items():
            name = key + ".txt"
            (folder / name).write_text(model.model_to_string(), encoding="utf-8")
            entries.append({"key": key, "file": name})
        if bundle.process_model is not None:
            (folder / "process.txt").write_text(bundle.process_model.model_to_string(), encoding="utf-8")
    write(folder / "meta.json", {"kind": kind, "bundle": meta, "entries": entries,
                                 "hashes": {name: value for name, value in artifact_manifest(folder).items() if name != "meta.json"}})
    return restore(folder)


def restore(folder):
    import lightgbm as lgb
    import main_s
    import main_opt
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    validate_artifacts(folder, meta.get("hashes", {}))
    bundle = (main_s.LoadBundle if meta["kind"] == "load" else main_opt.ResourceBundle)(**meta["bundle"])
    for row in meta["entries"]:
        model = lgb.Booster(model_str=(folder / row["file"]).read_text(encoding="utf-8"))
        if model.feature_name() != bundle.features:
            raise ValueError("模型重载特征顺序不一致")
        if meta["kind"] == "load":
            bundle.models.append((row["target"], row["horizons"], model))
        else:
            bundle.models[row["key"]] = model
    if meta["kind"] == "resource" and bundle.process_meta.get("available"):
        bundle.process_model = lgb.Booster(model_str=(folder / "process.txt").read_text(encoding="utf-8"))
    return bundle


def protected_configuration(candidate, reference):
    names = ("SCHEMA", "LOAD_VERSION", "STEP", "DT", "TARGETS", "GASES", "USE", "HOLDERS", "CAPACITY", "UNITS",
             "TRAIN_END", "TEST_START", "TEST_END", "FOLDS", "STAGES", "TABLES", "HOLDER_INPUT_VERSION", "HOLDER_SCALES",
             "POLICY", "GAS_POWER_ACCEPTANCE", "OPTIONAL_USE_CALIBRATION", "SOURCE_FILES")
    for name in names:
        if clean(getattr(candidate, name)) != clean(getattr(reference, name)):
            raise ValueError("候选修改了冻结的规则或数据边界：" + name)
    for key, value in reference.GROUP_DISPATCH.items():
        if key not in ("smooth_penalty", "mapping_penalty") and candidate.GROUP_DISPATCH.get(key) != value:
            raise ValueError("候选修改了分组标定验收边界：" + key)


def training_boundary(bundle, cutoff):
    import pandas as pd
    if pd.Timestamp(bundle.train_end) > cutoff:
        return False
    def walk(value):
        if isinstance(value, dict):
            if "max_target_time" in value and pd.Timestamp(value["max_target_time"]) > cutoff:
                return False
            return all(walk(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(walk(v) for v in value)
        return True
    return walk(bundle.diagnostics) and walk(getattr(bundle, "process_meta", {}))


def _regime_coverage(truth):
    """窗口真值所在的负荷档位占比：说明窗口均分究竟覆盖了哪些工况。
    只描述评分真值，不进入模型输入，也不用于候选排名。"""
    import numpy as np
    y = np.asarray(truth, float)
    edges = {"generator_1": (0., 60., 100., 150., 200.), "generator_all": (0., 300., 330., 360., 400.)}
    out = {"role": "truth_regime_coverage_diagnostic_only", "targets": {}}
    for index, name in enumerate(("generator_1", "generator_all")):
        v = y[..., index]
        finite = np.isfinite(v)
        bounds = edges[name]
        shares = {}
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            shares[f"({lo:g},{hi:g}]"] = float((finite & (v > lo) & (v <= hi)).sum() / max(1, finite.sum()))
        shares["zero_truth"] = float((finite & (v == 0.)).sum() / max(1, finite.sum()))
        out["targets"][name] = {"observed_cells": int(finite.sum()), "shares": shares,
                                "mean_mw": float(np.mean(v[finite])) if finite.any() else None}
    return out


def reusable_load(job, span, current):
    parent = Path(job["parent_output"]) if job.get("parent_output") else None
    if parent is None or not (parent / "模型" / span / "meta.json").is_file():
        return None
    source = Path(job["source"])
    prior = Path(job["parent_source"])
    if any((source / name).read_bytes() != (prior / name).read_bytes()
           for name in ("main_s.py", "main_l.py", "data_cleaning/pipeline.py")):
        return None
    previous = read_module("auto_parent_config", prior / "config.py")
    previous.TREE["num_threads"] = current.TREE["num_threads"]
    def effective(config):
        recipe = dict(config.LOAD_RECIPE)
        recipe.update(recipe.pop("tasks", {}).get(span, {}))
        return clean({"recipe": recipe, "tree": config.TREE, "version": config.LOAD_VERSION})
    return restore(parent / "模型" / span) if effective(previous) == effective(current) else None


def evaluate(job, cache=None):
    sys.path.insert(0, job["source"])
    import numpy as np
    import pandas as pd
    import config as C
    import main_s
    import main_l
    import main_opt
    import output
    from threadpoolctl import threadpool_limits

    reference = Path(job["reference_source"])
    config_ref = read_module("auto_reference_config", reference / "config.py")
    protected_configuration(C, config_ref)
    # 数据读取和评分取任务开始时冻结的实现，候选只能改变特征与模型。
    pipeline = read_module("auto_reference_pipeline", reference / "data_cleaning" / "pipeline.py")
    quality = read_module("auto_reference_quality", reference / "data_cleaning" / "quality.py")
    contract = job["contract"]
    C.CPU_THREADS = contract["threads"]
    C.TREE["num_threads"] = contract["threads"]
    limits = threadpool_limits(limits=contract["threads"])
    cutoff, start, end = (pd.Timestamp(contract[k]) for k in ("train_end", "start", "end"))
    if cache is not None and "data" in cache:
        truth_data = cache["data"]
    else:
        truth_data = pipeline.prepare_data(cutoff=contract.get("data_end", end))
        if cache is not None:
            cache["data"] = truth_data
    train = truth_data.prefix(cutoff)
    train.require_training(cutoff)
    view = truth_data.masked_copy(start, None, mask_use=True)
    view.labels = view.labels.copy()
    view.labels.loc[start:] = np.nan
    origins = pd.date_range(start, end, freq="15min")
    if not origins.isin(view.index).all():
        raise ValueError("验证输入网格覆盖不完整")
    if contract.get("expected_origins") and len(origins) != contract["expected_origins"]:
        raise ValueError(f"脚本窗口起报点数量不匹配：期望{contract['expected_origins']}，实际{len(origins)}")
    quick = contract["training_budget"] == "quick"
    root = Path(job["output"])
    metrics, masks, checks = {}, {}, {"training_boundary": True, "coverage": True, "reload": True, "causality": True}
    primary = []
    selected = ("short", "long") if job["track"] in ("load", "g1") else (job["track"],)
    bundles, restored_bundles, reuse_info = {}, {}, {}
    if job["track"] in ("short", "long", "load", "g1"):
        # 单链路的主指标不变，但部署后处理有交叉依赖，不能用裸预测代替实际输出。
        for span, train_func in (("short", main_s.train_short), ("long", main_l.train_long)):
            existing = root / "模型" / span / "meta.json"
            resumed = existing.is_file()
            bundle = restore(existing.parent) if resumed else reusable_load(job, span, C)
            reused = bundle is not None
            if bundle is None:
                bundle = train_func(train, quick=quick, progress=print)
            checks["training_boundary"] &= training_boundary(bundle, cutoff)
            bundles[span] = bundle
            restored_bundles[span] = persist(bundle, existing.parent, "load")
            reuse_info[span] = {"reused_unchanged_model": reused, "resumed_saved_model": resumed}
        predictions = main_s.predict_deployed(bundles["short"], bundles["long"], view, origins, C.LOAD_RECIPE)
        reloaded_predictions = main_s.predict_deployed(restored_bundles["short"], restored_bundles["long"], view, origins, C.LOAD_RECIPE)
        checks["deployment_output"] = True
        metrics["dependency_models"] = reuse_info
        for when in (origins[0], origins[-1]):
            probe = pd.DatetimeIndex([when])
            full = main_s.predict_deployed(bundles["short"], bundles["long"], view, probe, C.LOAD_RECIPE)
            prefix = main_s.predict_deployed(bundles["short"], bundles["long"], view.prefix(when), probe, C.LOAD_RECIPE)
            checks["causality"] &= all(bool(np.allclose(a, b, atol=1e-6, rtol=0.)) for a, b in zip(full, prefix))
    for span in selected:
        if span not in ("short", "long"):
            continue
        bundle = bundles[span]
        index = 0 if span == "short" else 1
        pred = np.round(predictions[index], 6)
        h = 8 if span == "short" else 96
        expected_shape = (len(origins), h, 2)
        if pred.shape != expected_shape or not np.isfinite(pred).all():
            raise ValueError("预测必须完整且有限，禁止通过缺失输出缩小评分范围")
        if not np.allclose(pred, quality.project_load(pred), rtol=0., atol=1e-6):
            raise ValueError("预测不满足容量关系")
        truth = quality.truth_cube(truth_data, origins, range(1, h + 1))
        available_tail = quality.score_load(pred, truth)
        times = origins.to_numpy()[:, None] + np.arange(1, h + 1)[None, :] * np.timedelta64(15, "m")
        if contract.get("score_tail") != "script_available_history":
            truth[times > end.to_datetime64()] = np.nan
        masks[span] = {"truth_sha256": hash_array(truth), "positive_cells": int((np.isfinite(truth) & (truth > 0.)).sum())}
        score = quality.score_load(pred, truth)
        naive = np.round(main_s.naive_profile(train, origins, range(1, h + 1), cutoff), 6)
        metrics[span] = {"score": score, "horizons": quality.score_horizons(pred, truth),
                         "diagnostics": quality.load_diagnostics(pred, truth, origins),
                         "naive": quality.score_load(naive, truth), "training": bundle.diagnostics,
                         "available_history_tail_sensitivity": available_tail,
                         "effective_budget": bundle.recipe.get("effective_budget"), **reuse_info[span],
                         "target_dates": {}}
        # 评分参考包：同一批预测在多种聚合口径、提前量、日级分布与误差分解下的表现。
        # 主指标不变，只补齐判断“这个本地分数代表哪种工况”所需的诊断。
        metrics[span]["reference_pack"] = quality.score_reference_pack(pred, truth, origins)
        metrics[span]["regime_coverage"] = _regime_coverage(truth)
        # 与冻结画像基线（不训练）的分档对比：关键低真值档里模型是否真的优于基线。
        metrics[span]["naive_band_comparison"] = quality.compare_bands(pred, naive, truth)
        # 单窗得分的日块抽样不确定性：几分的窗口差异是否超出噪声必须先看这个。
        metrics[span]["score_uncertainty"] = quality.score_uncertainty(pred, truth, origins)
        # 与父方案的配对差值：只有父方案在同一窗口保存过预测时才可算，否则显式标注。
        paired = {"status": "NO_PARENT_PREDICTIONS", "role": "diagnostic_only"}
        parent_window = Path(job["parent_output"]) if job.get("parent_output") else None
        if parent_window is not None:
            saved_prediction = parent_window / f"{span}_predictions.npz"
            if saved_prediction.is_file():
                prior = np.load(saved_prediction, allow_pickle=False)
                prior_prediction = np.asarray(prior["prediction"], float)
                prior_truth = np.asarray(prior["truth"], float)
                if prior_prediction.shape != pred.shape:
                    paired = {"status": "SHAPE_MISMATCH", "role": "diagnostic_only",
                              "parent_shape": list(prior_prediction.shape)}
                elif not np.array_equal(prior_truth, truth, equal_nan=True):
                    paired = {"status": "TRUTH_MISMATCH", "role": "diagnostic_only"}
                else:
                    paired = quality.paired_delta_uncertainty(pred, prior_prediction, truth, origins)
        metrics[span]["paired_delta"] = paired
        # 尾部口径敏感性：本地默认在窗口末端截断目标时刻；若官方按整段提前量评分，
        # 差异必须单列。窗口紧邻数据末尾时无法检验，据实报告 0 个额外真值格。
        extended_truth = quality.truth_cube(truth_data, origins, range(1, h + 1))
        extended_score = quality.score_load(pred, extended_truth)
        metrics[span]["tail_boundary_sensitivity"] = {
            "role": "diagnostic_only_not_candidate_ranking",
            "extra_truth_cells": int(np.isfinite(extended_truth).sum() - np.isfinite(truth).sum()),
            "extended": {t: extended_score[t]["score"] for t in C.TARGETS},
            "delta": {t: (None if score[t]["score"] is None or extended_score[t]["score"] is None
                          else float(extended_score[t]["score"] - score[t]["score"])) for t in C.TARGETS},
            "note": "窗口后标签只用于事后评分，不进入训练、特征或选模"}
        for stage, begin, stop, role in C.STAGES:
            if stage == "half_train" and role == "train":
                begin, stop = max(start, pd.Timestamp(begin)), min(end, pd.Timestamp(stop))
                if begin <= stop:
                    metrics[span]["recent_training_window"] = {"stage": stage, **quality.score_origin_window(pred, truth, origins, begin, stop)}
        for date in pd.date_range(start.normalize(), end.normalize(), freq="1D"):
            mask = (times >= date.to_datetime64()) & (times < (date + pd.Timedelta(days=1)).to_datetime64())
            metrics[span]["target_dates"][str(date.date())] = quality.score_load(pred, np.where(mask[..., None], truth, np.nan))
        selected_score = score["generator_1"]["score"] if job["track"] == "g1" else score["macro_score"]
        if selected_score is None:
            raise ValueError("主指标缺少有效真值，拒绝评分")
        primary.append(selected_score)
        np.savez_compressed(root / (span + "_predictions.npz"), prediction=pred, truth=truth, origins=origins.to_numpy())
        csv_path=root / ("s_result.csv" if span == "short" else "l_result.csv")
        output.prediction_frame(origins, pred).to_csv(csv_path, index=False, float_format="%.6f")
        # 复评实际CSV而非内存数组，独立检查列偏移和目标时刻是否与冻结裁判一致。
        tail_end=None if contract.get("score_tail")=="script_available_history" else end
        csv_score,csv_pred,csv_truth=quality.score_prediction_csv(csv_path,truth_data.labels,origins,h,tail_end)
        csv_ok=(np.array_equal(csv_pred,pred) and np.array_equal(csv_truth,truth,equal_nan=True) and
                all(csv_score[t]["score"]==score[t]["score"] for t in C.TARGETS))
        checks["csv_scoring_alignment"]=checks.get("csv_scoring_alignment",True) and bool(csv_ok)
        metrics[span]["csv_score_validation"]={"passed":bool(csv_ok),"time_alignment":"origin_plus_column_offset",
            "basis":"independent_csv_and_reference_labels","official_aggregation":"unknown"}
        if not csv_ok:
            raise ValueError("CSV与内部评分数据不一致，拒绝使用本次分数")
        reloaded = reloaded_predictions[index]
        delta = float(np.max(np.abs(pred - reloaded)))
        checks["reload"] &= delta <= 1e-6
        metrics[span]["reload_max_difference"] = delta
    if job["track"] in ("short", "long", "load", "g1"):
        objective = float((2 * primary[0] + 3 * primary[1]) / 5 if len(primary) == 2 else primary[0])
    else:
        cache = Path(job["resource_cache"]) if job["resource_cache"] else None
        saved_resource = root / "模型" / "resource" / "meta.json"
        if saved_resource.is_file():
            bundle = restore(saved_resource.parent)
        elif job["track"] == "dispatch" and cache is not None:
            bundle = restore(cache)
            # 调度候选只重标定其决策参数，资源预测树始终复用本任务的冻结权重。
            bundle.diagnostics["dispatch_control"] = main_opt.fit_dispatch_calibration(train, bundle.beta)
        else:
            bundle = main_opt.fit_resources(train, quick=quick, progress=print)
        restored = persist(bundle, root / "模型" / "resource", "resource")
        checks["training_boundary"] &= training_boundary(bundle, cutoff)
        issues = origins.normalize().unique() + pd.Timedelta(hours=7)
        issues = issues[(issues >= start) & (issues <= end)]
        if not len(issues):
            raise ValueError("调度验证窗口不包含每日07:00发布点")
        for issue in (issues[0], issues[-1]):
            a = main_opt.forecast_resources(bundle, view, issue)
            b = main_opt.forecast_resources(restored, view, issue)
            prefix = main_opt.forecast_resources(bundle, view.prefix(issue), issue)
            for key in ("available", "optional_caps", "split", "budget_blast"):
                if not np.isfinite(a[key]).all():
                    checks["coverage"] = False
                checks["reload"] &= bool(np.allclose(a[key], b[key], rtol=0., atol=1e-6, equal_nan=True))
                checks["causality"] &= bool(np.allclose(a[key], prefix[key], rtol=0., atol=1e-6, equal_nan=True))
        # quick只缩小学习预算；不能让优化函数默默把三天评测裁成两天。
        schedule, days, opt = main_opt.optimize_daily(bundle, view, origins, quick=False, progress=print)
        independent = [quality.validate_day(day) for day in days]
        sequence = quality.validate_plan_sequence(days)
        checks["hard_constraints"] = bool(all(d["passed"] for d in independent) and sequence["passed"])
        checks["coverage"] &= bool(len(days) == len(issues) and len(schedule) == len(issues) * 96)
        labels = pipeline.process_series(truth_data)[["blast_13", "blast_45", "known_demand"]].copy()
        for name, column in zip(("budget_blast", "budget_coke", "budget_converter"), C.USE):
            labels[name] = truth_data.frame[column]
        for day in days:
            for values in day["components"].values():
                if np.asarray(values).shape != (96,) or not np.isfinite(values).all():
                    checks["coverage"] = False
        audit = quality.audit_resource_forecasts(days, labels, window_end=end)
        names = ("budget_blast", "budget_coke", "budget_converter")
        wapes = [audit[n]["wape"] for n in names]
        if any(w is None for w in wapes):
            raise ValueError("资源主指标缺少有效真值")
        masks["resource"] = {name: audit[name]["cells"] for name in names}
        for name in names:
            masks["resource"][name + "_truth_sha256"] = hash_array(labels[name].to_numpy())
        value, known = 0., []
        for day in days:
            times = pd.DatetimeIndex(day["times"])
            future = truth_data.frame.reindex(times)
            after = truth_data.frame.reindex(times + pd.Timedelta(minutes=15))
            mask = (np.isfinite(future[list(C.USE)]).all(axis=1).to_numpy() &
                    np.isfinite(future[list(C.HOLDERS)]).all(axis=1).to_numpy() &
                    np.isfinite(after[list(C.HOLDERS)]).all(axis=1).to_numpy() &
                    (times + pd.Timedelta(minutes=15) <= end))
            price = pipeline.prices_at(truth_data, times)
            power = np.asarray(day["power"]).sum(axis=1)
            vent = np.asarray(day["vent"]).sum(axis=1)
            if not np.isfinite(power).all() or not np.isfinite(vent).all():
                checks["coverage"] = False
            value += float(np.sum((price * power * 250. - C.POLICY["vent_penalty_cny_m3"] * vent * C.DT)[mask]))
            known.extend(mask.tolist())
        masks["dispatch"] = {"observed_steps": sum(known), "mask_sha256": hash_array(np.array(known, bool))}
        if job["track"] == "dispatch" and not sum(known):
            raise ValueError("调度主指标没有观测支持区间")
        metrics.update(resource=audit, optimization=opt, independent_days=independent, independent_sequence=sequence,
                       historical_replay=quality.historical_replay(truth_data.prefix(end), days),
                       observed_window_planned_net_value_cny=value,
                       revenue_basis="计划负荷与计划放散在有观测区间上的本地价值，不是执行收益或官网得分")
        objective = -float(np.mean(wapes)) if job["track"] == "resource" else value
        schedule.to_csv(root / "opt_result.csv", index=False, float_format="%.6f")
        write(root / "trajectory.json", days)
    checks["source_unchanged"] = all(hashlib.sha256((Path(job["source"]) / name).read_bytes()).hexdigest() == expected
                                     for name, expected in job["manifest"].items())
    comparison = {"contract": contract, "masks": masks, "data_sources": truth_data.sources}
    return {"status": "COMPLETE", "primary": objective, "checks": checks, "metrics": metrics,
            "comparison": comparison, "experimental": True, "submittable": False,
            "official_acceptance_status": "unknown", "field_safety_certified": False}


def window_score_sensitivity(reports):
    """单窗口敏感度：逐窗分数、去掉该窗后的均值和均值的最大偏移。
    基线任务没有对照包，只有再次比较候选时才能算配对收益，因此这里改用本次自身分数，
    回答“总分依赖哪一个窗口”，不改任何排名口径。"""
    import numpy as np
    out = {"role": "own_window_sensitivity_diagnostic_only", "spans": {}}
    for span in ("short", "long"):
        if not all(span in r.get("metrics", {}) for r in reports.values()):
            continue
        out["spans"][span] = {}
        for target in ("generator_1", "generator_all"):
            scores = {name: r["metrics"][span]["score"][target]["score"] for name, r in reports.items()}
            values = [v for v in scores.values() if isinstance(v, (int, float))]
            if len(values) < 2 or len(values) != len(scores):
                continue
            mean = float(np.mean(values))
            leave_out = {name: float(np.mean([v for k, v in scores.items() if k != name])) for name in scores}
            shifts = {name: leave_out[name] - mean for name in scores}
            out["spans"][span][target] = {
                "window_scores": scores, "mean": mean, "leave_one_out_mean": leave_out,
                "leave_one_out_shift": shifts,
                "max_abs_shift": float(max(abs(v) for v in shifts.values())),
                "most_influential_window": max(shifts, key=lambda k: abs(shifts[k])),
                "range": float(max(values) - min(values)),
                "worst_window": min(scores, key=scores.get), "best_window": max(scores, key=scores.get)}
    return out


def event_day_reference(reports):
    """事件日占比与窗口均值的条件参考：日级 1-MAPE<80 记为事件日（逐目标分别统计）。
    本地均值只有在事件日占比相近时才可比较；本块只描述工况条件，不改主指标，
    也不代表官方聚合或线上分数预测。"""
    out = {"role": "event_day_reference_diagnostic_only",
           "definition": "事件日=某起报日期整日池化 1-MAPE<80（本地约定，逐目标统计）",
           "caveat": "目标期事件日数量未知；本表只说明本地均值的工况条件，不预测线上分数",
           "spans": {}}
    for span in ("short", "long"):
        span_rows = {}
        for target in ("generator_1", "generator_all"):
            rows = []
            for name, report in reports.items():
                days = (report.get("metrics", {}).get(span, {}).get("reference_pack", {})
                        .get("days", {}).get("targets", {}).get(target, {})
                        .get("by_target_date") or {})
                scores = [item.get("score") for item in days.values()
                          if isinstance(item.get("score"), (int, float))]
                if len(scores) < 5:
                    continue
                rows.append({"window": name, "days": len(scores),
                             "event_share": sum(1 for s in scores if s < 80.) / len(scores),
                             "equal_day_mean": sum(scores) / len(scores),
                             "min_day": min(scores)})
            if rows:
                rows.sort(key=lambda r: r["event_share"])
                span_rows[target] = {"rows": rows}
        if span_rows:
            out["spans"][span] = span_rows
    return out


def reference_band(reports, sensitivity):
    """把逐窗分数、日级下尾、完整提前量覆盖与零值格数合成一条可直接引用的参考带。
    只描述本地历史分布，不是线上分数预测，也不替代主指标。"""
    import numpy as np
    out = {"role": "local_reference_band_diagnostic_only",
           "note": "本地历史窗口分布，不代表测试期或官网分数", "targets": {}}
    for span, targets in sensitivity.get("spans", {}).items():
        for target, row in targets.items():
            day_p10, day_median, complete, zero, cells = [], [], [], 0, 0
            for report in reports.values():
                pack = report.get("metrics", {}).get(span, {}).get("reference_pack", {})
                quant = pack.get("days", {}).get("targets", {}).get(target, {}).get("target_date_quantiles", {})
                if quant.get("p10") is not None:
                    day_p10.append(quant["p10"])
                    day_median.append(quant["median"])
                agg = pack.get("aggregation", {}).get("targets", {}).get(target, {})
                if agg.get("complete_horizon") is not None:
                    complete.append(agg["complete_horizon"])
                zero += int(agg.get("zero_truth") or 0)
                cells += int(agg.get("cells") or 0)
            scores = list(row["window_scores"].values())
            out["targets"][f"{span}:{target}"] = {
                "window_count": len(scores), "mean": row["mean"], "min": float(min(scores)),
                "max": float(max(scores)), "range": row["range"], "worst_window": row["worst_window"],
                "best_window": row["best_window"],
                "max_single_window_shift": row["max_abs_shift"],
                "most_influential_window": row["most_influential_window"],
                "day_p10_min": float(min(day_p10)) if day_p10 else None,
                "day_median_min": float(min(day_median)) if day_median else None,
                "complete_horizon_mean": float(np.mean(complete)) if complete else None,
                "zero_truth_cells": zero, "scored_cells": cells}
    return out


def paired_window_sensitivity(reports, reference):
    """揭示配对收益对单个历史窗口的依赖；只作诊断，不改变选模指标。"""
    import numpy as np
    if set(reports) != set(reference) or not reports:
        return {"status": "UNAVAILABLE_WINDOW_MISMATCH", "role": "diagnostic_only"}
    delta = {}
    for name, row in reports.items():
        prior = reference[name]
        left, right = row.get("comparison", {}), prior.get("comparison", {})
        if (left.get("contract") != right.get("contract") or not left.get("masks") or
                left.get("masks") != right.get("masks") or not left.get("data_sources") or
                left.get("data_sources") != right.get("data_sources")):
            return {"status": "UNAVAILABLE_CONTRACT_OR_TRUTH_MISMATCH", "role": "diagnostic_only"}
        delta[name] = float(row["primary"] - prior["primary"])
    mean = float(np.mean(list(delta.values())))
    if len(delta) < 2:
        return {"status": "NEEDS_MULTIPLE_WINDOWS", "role": "diagnostic_only", "mean_delta": mean,
                "window_deltas": delta}
    leave_out = {name: float(np.mean([v for k, v in delta.items() if k != name])) for name in delta}
    return {"status": "AVAILABLE", "role": "diagnostic_only", "mean_delta": mean,
            "window_deltas": delta, "leave_one_out_mean_delta": leave_out}


def historical_score_stability(reports):
    """比较仅用过去窗口得分估计下一窗口的误差，不预测或校准官网分数。"""
    from datetime import datetime
    import numpy as np
    if len(reports)<3:
        return {"status":"NEEDS_AT_LEAST_THREE_WINDOWS","role":"historical_diagnostic_only"}
    rows=[]
    try:
        for name,r in reports.items():
            c=r["comparison"]["contract"]
            start,end,cutoff=(datetime.fromisoformat(str(c[x])) for x in ("start","end","train_end"))
            if not cutoff<start<=end:
                return {"status":"INVALID_TRAINING_BOUNDARY","role":"historical_diagnostic_only"}
            rows.append((start,end,name,r))
        rows.sort(key=lambda x:x[0])
        if any(a[1]>=b[0] for a,b in zip(rows,rows[1:])):
            return {"status":"OVERLAPPING_WINDOWS","role":"historical_diagnostic_only"}
        groups={}
        for span in ("short","long"):
            for target in ("generator_1","generator_all"):
                if not all(span in row[3].get("metrics",{}) for row in rows):
                    continue
                scores=[row[3]["metrics"][span]["score"][target]["score"] for row in rows]
                if any(x is None or not np.isfinite(x) for x in scores):
                    continue
                comparisons=[]
                for i in range(1,len(rows)):
                    estimate=float(np.mean(scores[:i]));last=float(scores[i-1]);actual=float(scores[i])
                    comparisons.append({"target_window":rows[i][2],"previous_count":i,
                        "historical_mean":estimate,"previous_window":last,"actual_score":actual,
                        "mean_error":estimate-actual,"previous_error":last-actual})
                groups[span+"_"+target]={"count":len(comparisons),
                    "mean_estimate_mae":float(np.mean([abs(x["mean_error"]) for x in comparisons])),
                    "previous_estimate_mae":float(np.mean([abs(x["previous_error"]) for x in comparisons])),
                    "comparisons":comparisons}
        if not groups:
            return {"status":"NO_COMPARABLE_SCORES","role":"historical_diagnostic_only"}
        return {"status":"AVAILABLE","role":"historical_diagnostic_only_not_official_score",
                "independent_holdout":False,"window_count":len(rows),
                "window_order":[x[2] for x in rows],"groups":groups}
    except (KeyError,TypeError,ValueError):
        return {"status":"INVALID_WINDOW_OR_SCORE","role":"historical_diagnostic_only"}


def evaluate_windows(job):
    import numpy as np
    root = Path(job["output"])
    reports, cache = {}, {}
    source = Path(job["source"])
    if not all(hashlib.sha256((source / name).read_bytes()).hexdigest() == expected for name, expected in job["manifest"].items()):
        raise ValueError("实验源码已改变，拒绝混合续跑结果")
    windows = job["contract"].get("windows")
    if not windows:
        result = evaluate(job)
        result["artifacts"] = artifact_manifest(root)
        return result
    for window in windows:
        name = window["name"]
        part = dict(job)
        part["output"] = str(root / "窗口" / name)
        part["contract"] = {k: v for k, v in job["contract"].items() if k != "windows"}
        part["contract"].update(window)
        if job.get("parent_output"):
            part["parent_output"] = str(Path(job["parent_output"]) / "窗口" / name)
        if job.get("baseline_output"):
            part["resource_cache"] = str(Path(job["baseline_output"]) / "窗口" / name / "模型" / "resource")
        folder = Path(part["output"])
        folder.mkdir(parents=True, exist_ok=True)
        saved = folder / "报告" / "运行结果.json"
        if saved.is_file():
            old = json.loads(saved.read_text(encoding="utf-8"))
            if old.get("status") == "COMPLETE" and old.get("comparison", {}).get("contract") == part["contract"]:
                validate_artifacts(folder, old.get("artifacts", {}))
                print(f"窗口 {name}：接续已完成结果，不重复训练", flush=True)
                reports[name] = old
                continue
        print(f"窗口 {name}：{window['start']} 至 {window['end']}，训练截止 {window['train_end']}", flush=True)
        began = time.monotonic()
        result = evaluate(part, cache)
        result.update(window=name, seconds=time.monotonic() - began, artifacts=artifact_manifest(folder))
        write(saved, result)
        reports[name] = result
        print(f"窗口 {name} 完成：主指标={result['primary']:.6f}；检查={result['checks']}", flush=True)
    checks = {key: all(r["checks"].get(key) is True for r in reports.values())
              for key in set().union(*(r["checks"] for r in reports.values()))}
    summary = {"windows": {}, "mean": {}}
    for name, report in reports.items():
        row = {}
        for span in ("short", "long"):
            if span in report["metrics"]:
                score = report["metrics"][span]["score"]
                row[span] = {target: score[target]["score"] for target in ("generator_1", "generator_all")}
                row[span + "_macro"] = score["macro_score"]
                # 多口径与工况覆盖只进汇总表，便于逐窗对照；主指标仍是逐格 1-MAPE。
                pack = report["metrics"][span].get("reference_pack", {}).get("aggregation", {}).get("targets", {})
                if pack:
                    row[span + "_variants"] = {target: {k: v.get(k) for k in
                        ("cell_mean", "origin_mean", "day_equal_mean", "wape_score", "smape",
                         "complete_horizon", "zero_as_full_error", "zero_truth")}
                        for target, v in pack.items()}
                regime = report["metrics"][span].get("regime_coverage", {}).get("targets", {})
                if regime:
                    row[span + "_regime"] = {target: {"mean_mw": v.get("mean_mw"), **v.get("shares", {})}
                                             for target, v in regime.items()}
                share = report["metrics"][span].get("reference_pack", {}).get("decomposition", {})
                if share:
                    row[span + "_decomposition"] = {k: (share.get("overall") or {}).get(k) for k in
                        ("mean_abs_share_mw", "mean_abs_level_mw", "share_attribution", "bias_share_mw", "bias_level_mw")}
                    row[span + "_decomposition"]["mean_abs_share_gap"] = share.get("mean_abs_share_gap")
                tail = report["metrics"][span].get("tail_boundary_sensitivity", {})
                if tail:
                    row[span + "_tail"] = {"extra_truth_cells": tail.get("extra_truth_cells"),
                                           "delta": tail.get("delta")}
                vs = report["metrics"][span].get("naive_band_comparison", {})
                if vs.get("targets"):
                    row[span + "_vs_naive"] = {
                        target: {"model": v.get("model_score"), "baseline": v.get("baseline_score"),
                                 "band_delta": {b: x.get("delta") for b, x in (v.get("by_observed_target_mw") or {}).items()},
                                 "model_error_share": {b: x.get("model_error_share")
                                                       for b, x in (v.get("by_observed_target_mw") or {}).items()}}
                        for target, v in vs["targets"].items() if isinstance(v, dict) and v.get("model_score") is not None}
                unc = report["metrics"][span].get("score_uncertainty", {}).get("targets", {})
                if unc:
                    row[span + "_uncertainty"] = {
                        target: {"se": v.get("standard_error"), "half_width": v.get("half_width"), "days": v.get("days")}
                        for target, v in unc.items() if isinstance(v, dict) and v.get("standard_error") is not None}
                pair = report["metrics"][span].get("paired_delta", {})
                if pair.get("targets"):
                    row[span + "_paired"] = {
                        target: {"delta": v.get("delta_point"), "se": v.get("delta_standard_error"),
                                 "p2_5": v.get("delta_p2_5"), "p97_5": v.get("delta_p97_5")}
                        for target, v in pair["targets"].items() if isinstance(v, dict) and v.get("delta_point") is not None}
                elif pair.get("status"):
                    row[span + "_paired"] = {"status": pair["status"]}
        summary["windows"][name] = row
    for span in ("short", "long"):
        if all(span in row for row in summary["windows"].values()):
            for target, suffix in (("generator_1", "g1"), ("generator_all", "gall")):
                summary["mean"][span + "_" + suffix] = float(np.mean([row[span][target] for row in summary["windows"].values()]))
            for key in ("cell_mean", "origin_mean", "day_equal_mean", "wape_score", "smape", "complete_horizon"):
                values = [row[span + "_variants"][target][key] for row in summary["windows"].values()
                          for target in ("generator_1", "generator_all")]
                if all(v is not None for v in values):
                    summary["mean"][span + "_" + key + "_macro"] = float(np.mean(values))
    # 窗口均值的抽样不确定性：各窗是不同时段而非重复实验，按独立近似给出参考标准误。
    uncertainty_summary = {}
    for span in ("short", "long"):
        for target in ("generator_1", "generator_all"):
            key = span + "_uncertainty"
            ses = [row[key][target]["se"] for row in summary["windows"].values()
                   if key in row and isinstance(row[key].get(target), dict)]
            if ses and len(ses) == len(summary["windows"]):
                uncertainty_summary[f"{span}:{target}"] = {
                    "windows": len(ses), "mean_single_window_se": float(np.mean(ses)),
                    "se_of_window_mean": float(np.sqrt(np.mean(np.square(ses))) / np.sqrt(len(ses)))}
    summary["window_mean_uncertainty"] = uncertainty_summary
    baseline = job.get("baseline_output")
    if baseline:
        reference = {}
        for name in reports:
            path = Path(baseline) / "窗口" / name / "报告" / "运行结果.json"
            if path.is_file():
                reference[name] = json.loads(path.read_text(encoding="utf-8"))
        summary["paired_sensitivity"] = paired_window_sensitivity(reports, reference)
    else:
        # 基线任务没有对照包：显式标注，避免该字段静默缺失被误读成“无敏感性”。
        summary["paired_sensitivity"] = {"status": "NO_BASELINE_REFERENCE", "role": "diagnostic_only"}
    # 单窗口敏感度与参考带对任何一次回放都可算，直接由本次逐窗结果生成。
    sensitivity = window_score_sensitivity(reports)
    summary["own_window_sensitivity"] = sensitivity
    summary["reference_band"] = reference_band(reports, sensitivity)
    summary["event_day_reference"] = event_day_reference(reports)
    if job["track"] in ("short","long","load","g1"):
        summary["historical_score_stability"] = historical_score_stability(reports)
    artifacts = {}
    for name, report in reports.items():
        artifacts.update({f"窗口/{name}/{path}": value for path, value in report.get("artifacts", {}).items()})
        saved = root / "窗口" / name / "报告" / "运行结果.json"
        artifacts[f"窗口/{name}/报告/运行结果.json"] = hashlib.sha256(saved.read_bytes()).hexdigest()
    return {"status": "COMPLETE", "primary": float(np.mean([r["primary"] for r in reports.values()])),
            "artifacts": artifacts,
            "checks": checks, "metrics": {"windows": {k: r["metrics"] for k, r in reports.items()}},
            "window_summary": summary,
            "comparison": {"contract": job["contract"], "windows": {k: r["comparison"] for k, r in reports.items()}},
            "experimental": True, "submittable": False, "official_acceptance_status": "unknown",
            "field_safety_certified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    write(Path(job["output"]) / "报告" / "worker.json", {"pid": os.getpid()})
    tick = time.monotonic()
    try:
        report = evaluate_windows(job)
    except Exception as exc:
        traceback.print_exc()
        report = {"status": "FAILED", "error": str(exc), "experimental": True, "submittable": False}
    report.update(experiment_id=job["id"], seconds=time.monotonic() - tick,
                  job_sha256=hashlib.sha256(args.job.read_bytes()).hexdigest(),
                  source=job["source"], reference_source=job["reference_source"],
                  source_manifest=job["manifest"], training_budget=job["contract"]["training_budget"],
                  actual_threads=job["contract"]["threads"])
    write(Path(job["output"]) / "报告" / "运行结果.json", report)
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
