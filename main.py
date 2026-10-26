"""半决赛唯一日常入口：python main.py

默认 infer 按已冻结配方重训，对正式半决赛测试窗口生成预测与优化结果；
run 保留历史 21 天时序回测。测试标签为空，不计算测试集本地分数；
当次结果能否提交由窗口、运行范围和优化核验共同判定。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import main_l
import main_opt
import main_s
import output
from data_cleaning import data_contract_check, prepare_data, run_self_checks, scan_sources


LOGGER = logging.getLogger("aic-half")


class _ProgressLogHandler(logging.StreamHandler):
    def emit(self, record) -> None:
        try:
            C.tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def _setup_logger(log_path: Path | None = None) -> None:
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = _ProgressLogHandler(sys.stdout)
    console.setFormatter(fmt)
    LOGGER.addHandler(console)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        LOGGER.addHandler(file_handler)


def build_folds(prepared, config=C, quick: bool = False) -> list[dict]:
    folds = []
    specs = config.FOLDS
    if quick:
        # 开发反复运行只使用 A/B，不能把 D 冻结窗口当成日常调参反馈。
        specs = (config.FOLDS[0], {**config.FOLDS[1], "role": "confirm"})
    for spec in specs:
        start = pd.Timestamp(spec["start"])
        end = start + pd.Timedelta(days=int(spec["days"])) - pd.Timedelta(minutes=config.STEP_MINUTES)
        origins = pd.date_range(start, end, freq=f"{config.STEP_MINUTES}min")
        positions = prepared.index.get_indexer(origins)
        if (positions < 0).any():
            raise ValueError(f"折 {spec['name']} 起报点不在数据网格")
        truth_pos = positions[:, None] + np.asarray(config.HORIZONS_L)[None, :]
        if (truth_pos >= len(prepared.labels)).any():
            raise ValueError(f"折 {spec['name']} 的最长提前量超出合法标签范围")
        labels = prepared.labels[list(config.TARGETS)].to_numpy(dtype=float)[truth_pos]
        coverage = float(np.isfinite(labels).any(axis=2).mean())
        train_end_pos = int(prepared.index.get_indexer([pd.Timestamp(spec["train_end"])])[0])
        train_samples = max(0, train_end_pos - max(config.HORIZONS_L) - config.PROFILE_DAYS * config.STEPS_PER_DAY)
        if coverage < 0.95:
            raise ValueError(f"折 {spec['name']} 标签覆盖率不足: {coverage:.3f}")
        if train_samples < 1000:
            raise ValueError(f"折 {spec['name']} 训练样本不足: {train_samples}")
        folds.append({**spec, "origins": origins, "positions": positions,
                      "label_coverage": coverage, "train_samples": int(train_samples)})
    if not any(f["role"] == "tuning" for f in folds) or not any(f["role"] == "confirm" for f in folds):
        raise ValueError("缺少调参折或确认折")
    return folds


def _sources_unchanged(scan: dict) -> bool:
    for relative, info in C.progress(scan.get("files", {}).items(), desc="核验源文件", unit="文件"):
        path = C.ROOT / relative
        if not path.is_file():
            return False
        import hashlib
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != info["sha256"]:
            return False
    return True


def run_doctor() -> int:
    _setup_logger()
    scan = scan_sources()
    print("数据目录扫描（只读）")
    for stage, info in scan["stages"].items():
        print(f"\n[{stage}] role={info['role']} exists={info['exists']}")
        for table, detail in info["tables"].items():
            span = detail["time_range"] or ["-", "-"]
            targets = detail["target_valid"] or {}
            print(f"  {table:10s} files={len(detail['files'])} rows={detail['rows']:7d} "
                  f"{span[0]} -> {span[1]} target_valid={targets}")
    print("\n附加文件:")
    for name, info in scan["extras"].items():
        print(f"  {name}: exists={info['exists']}")
    if scan["problems"]:
        print("\n问题:")
        for problem in scan["problems"]:
            print(f"  - {problem}")
    return 0


def run_self_check() -> int:
    _setup_logger()
    checks = run_self_checks()
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks.get("passed") else 1


def run_verify(directory: Path | None = None) -> int:
    _setup_logger()
    from zipfile import ZipFile
    if directory is None:
        candidates = sorted(p for p in C.OUTPUT_ROOT.glob("*") if p.is_dir() and not p.name.startswith("_"))
        if not candidates:
            print("没有选定运行目录；实验结果请使用 verify --run-dir 指定")
            return 1
        directory = candidates[-1]
    directory = Path(directory)
    problems = []
    try:
        meta_path = directory / C.MODEL_DIR / "model_meta.json"
        report_path = directory / C.REPORT_DIR / C.REPORT_JSON
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        contract = meta.get("output_contract", {})
        quick = bool(meta.get("run", {}).get("quick", report.get("quick", False)))
        window = contract.get("window", report.get("window"))
        spec = contract.get("prediction")
        if window == "half_test" or (not contract and not report and meta.get("role") == "final"):
            origins = output.expected_result_grid("s_result.csv")
        else:
            span = ([spec["start"], spec["end"]] if spec else
                    report.get("origin_range", report.get("data", {}).get("eval_range")))
            if not span:
                raise ValueError("缺少可验证的输出时间契约")
            origins = pd.date_range(span[0], span[1], freq=f"{C.STEP_MINUTES}min")
        if meta.get("schema_version", 1) >= 2:
            if spec != output._grid_spec(origins):
                problems.append("模型元数据中的预测覆盖契约不一致")
            opt_spec = contract.get("optimization")
            if (directory / "opt_result.csv").is_file():
                if opt_spec != output._grid_spec(output.expected_result_grid("opt_result.csv", origins, quick)):
                    problems.append("模型元数据中的优化覆盖契约不一致")
                if contract.get("optimization_time_semantics") != C.OPT_CONFIG["datetime_semantics"]:
                    problems.append("优化时间语义与当前实现不一致")
                publication = output.validate_publication_contract(meta,
                    output.expected_result_grid("opt_result.csv", origins, quick))
                problems.extend(publication["problems"])
            elif opt_spec is not None:
                problems.append("缺少元数据中声明的优化结果")
        coverage = output.validate_result_files(directory, origins, quick)
        problems.extend(coverage["problems"])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"directory": str(directory), "passed": False,
                          "problems": [str(exc)]}, ensure_ascii=False, indent=2))
        return 1
    physical = main_opt.physical_submission_check({"summary": meta.get("optimization", {})})
    declared_submittable = report.get("submittable", report.get("output", {}).get("submittable", False))
    if declared_submittable and (quick or meta.get("run", {}).get("experimental") or window != "half_test"):
        problems.append("运行范围不是完整正式 half_test，不能声明可提交")
    if declared_submittable and not physical["passed"]:
        problems.append("运行摘要声称可提交，但完整物理核验未通过")
    zip_paths = [directory / name for name in {C.MOCK_ZIP_NAME, C.FORMAL_ZIP_NAME}
                 if (directory / name).is_file()]
    if not zip_paths:
        zip_paths = list(directory.glob("*评分包*.zip"))
    if not zip_paths:
        problems.append("缺少运行结果压缩包")
    else:
        try:
            with ZipFile(zip_paths[0]) as packed:
                if packed.testzip() is not None:
                    problems.append("ZIP CRC 检查失败")
                if f"{C.SOURCE_DIR}/main_nn.py" not in packed.namelist():
                    problems.append("ZIP 源码缺少 main_nn.py")
                if meta_path.is_file() and packed.read(f"{C.MODEL_DIR}/model_meta.json") != meta_path.read_bytes():
                    problems.append("ZIP 模型元数据与磁盘不一致")
                import hashlib
                for relative, digest in meta.get("source", {}).get("sha256", {}).items():
                    if hashlib.sha256(packed.read(f"{C.SOURCE_DIR}/{relative}")).hexdigest() != digest:
                        problems.append(f"ZIP 源码与启动快照不一致: {relative}")
                for name in ("s_result.csv", "l_result.csv", "opt_result.csv"):
                    if (directory / name).is_file() and packed.read(name) != (directory / name).read_bytes():
                        problems.append(f"ZIP 内 {name} 与磁盘不一致")
        except Exception as exc:
            problems.append(f"ZIP 读取失败: {exc}")
    print(json.dumps({"directory": str(directory), "passed": not problems,
                      "problems": problems, "coverage": coverage,
                      "submittable": bool(declared_submittable and physical["passed"] and not problems),
                      "submission_blockers": physical["problems"]}, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


def _inference_window(prepared) -> tuple[pd.DatetimeIndex, str]:
    """测试推理保留完整起报网格；已提供但不完整的官方输入直接报错。"""
    index = prepared.index
    half = [s for s in C.STAGES if s["name"] == "half_test"]
    info = prepared.scan.get("stages", {}).get("half_test", {})
    tables = info.get("tables", {})
    if half and any(t.get("files") for t in tables.values()):
        start = pd.Timestamp(C.OFFICIAL_EVAL_START)
        end = pd.Timestamp(C.OFFICIAL_EVAL_END)
        for table in C.TABLES:
            detail = tables.get(table, {})
            span = detail.get("time_range")
            if (not span or pd.Timestamp(span[0]) > start or pd.Timestamp(span[1]) < end):
                raise ValueError(f"half_test/{table} 缺失或未覆盖完整官方评测区间")
        origins = pd.date_range(start, end, freq=f"{C.STEP_MINUTES}min")
        if not origins.isin(index).all():
            raise ValueError("官方评测网格不完整，不得截短输出窗口")
        return origins, "half_test"
    stage = next(s for s in C.STAGES if s["name"] == "semi_test")
    start = pd.Timestamp(stage["period"][0] + " 00:00:00")
    end = pd.Timestamp(stage["period"][1] + " 23:45:00")
    origins = pd.date_range(start, end, freq=f"{C.STEP_MINUTES}min")
    origins = origins[origins.isin(index)]
    if not len(origins):
        raise ValueError("没有可用的推理起报点")
    return origins, "semi_test_dryrun"


def _inference_submission_check(window: str, optimization: dict, *,
                                quick: bool, experimental: bool,
                                coverage: dict | None = None, source: dict | None = None) -> dict:
    """正式参赛阶段与当次产物是否通过提交核验分别记录，避免沿用预开发总开关。"""
    problems = []
    if window != "half_test":
        problems.append("未使用完整 half_test 半决赛评测窗口")
    if quick:
        problems.append("quick 仅用于开发验证，调度未覆盖完整评测窗口")
    if experimental:
        problems.append("本次为开发实验，未归档为选定正式运行")
    if not coverage or not coverage.get("passed"):
        problems.append("缺少完整结果覆盖核验或核验未通过")
    if not source or source.get("freeze_stage") != "run_start" or not source.get("consistency_passed"):
        problems.append("缺少启动源码一致性核验或源码已变化")
    if not C.OPT_CONFIG["template_confirmed"]:
        problems.append("优化输出字段尚未核对官方要求")
    if optimization.get("schedule") is None:
        problems.append("缺少有效 opt_result.csv 调度结果")
    else:
        check = output.validate_official_frame("opt_result.csv", optimization["schedule"],
            output.expected_result_grid("opt_result.csv", quick=quick))
        problems.extend(check["problems"])
    opt_summary = optimization.get("summary", {})
    if not opt_summary.get("plan_validation_passed"):
        problems.append("优化计划约束核验未通过")
    physical = main_opt.physical_submission_check(optimization)
    problems.extend(physical["problems"])
    if not optimization.get("submittable", False):
        if optimization.get("status") == "SIMULATED_NOT_SUBMITTABLE":
            problems.append("发电优化仍为代理资源模拟计划，尚未通过正式物理约束核验")
        else:
            problems.append("优化提交核验未通过：" + str(
                opt_summary.get("reason") or optimization.get("status") or "未返回核验状态"))
    return {"passed": not problems, "problems": list(dict.fromkeys(problems)), "physical": physical}


def run_infer(quick: bool = False, experimental: bool = False,
              reuse_run: Path | None = None) -> int:
    """冻结模型后执行正式半决赛测试推理，并单独核验当次结果的提交条件。"""
    started = time.time()
    output.training_cache_stats(reset=True)
    ctx = output.begin_run(C, mode="infer", experimental=quick or experimental, quick=quick)
    _setup_logger(ctx.log_path)
    LOGGER.info("推理运行目录: %s", ctx.root)
    summary = {"run_id": ctx.run_id, "status": "FAILED", "mode": "INFER",
               "competition_stage": "SEMIFINAL",
               "experimental": bool(quick or experimental), "output_root": str(ctx.root),
               "quick": bool(quick), "validation_scope": "冻结配方重训；不重新选参或评估D", "errors": []}
    stages = C.progress(total=8, desc="推理 · 数据准备", unit="阶段", leave=True)
    try:
        scan = scan_sources()
        prepared = prepare_data(C, include_inference=True)
        contract = data_contract_check(prepared)
        if not contract["passed"]:
            raise ValueError("数据契约检查失败: " + "; ".join(contract["problems"]))
        if not _sources_unchanged(scan):
            raise ValueError("源数据在读取后被修改")

        origins, window = _inference_window(prepared)
        ctx.window = window
        # 训练重新读取独立白名单，测试期过程量也不能用于资源拟合或物理标定。
        training = prepare_data(C).prefix_copy(min(
            pd.Timestamp(C.TRAIN_END), origins[0] - pd.Timedelta(minutes=C.STEP_MINUTES)))
        LOGGER.info("训练白名单: %s .. %s；半决赛测试输入仅用于冻结模型推理",
                    training.index[0], training.index[-1])
        first_issue = main_opt.daily_publication_times(origins, quick=quick)[0]
        resource_training = training.prefix_copy(min(training.index[-1], first_issue))
        LOGGER.info("每日%s发布；资源训练截止=%s，首份计划发布=%s",
                    C.OPT_CONFIG["publication_time"], resource_training.index[-1], first_issue)
        stages.update(1)
        if reuse_run is None:
            stages.set_description("推理 · 短期训练")
            short_bundle = main_s.train_final_load(training, "short", main_s.ShortBundle, quick=quick)
            stages.update(1)
            stages.set_description("推理 · 长期训练")
            long_bundle = main_s.train_final_load(training, "long", main_l.LongBundle, quick=quick)
            stages.update(1)
            stages.set_description("推理 · 资源训练")
            resource_bundle = main_opt.train_final_resources(resource_training, quick=quick)
            stages.update(1)
        else:
            stages.set_description("推理 · 复用模型与冻结 NN")
            short_bundle, long_bundle, resource_bundle = output.load_final_models(reuse_run)
            for bundle in (short_bundle, long_bundle, resource_bundle):
                if pd.Timestamp(bundle.final_train_end) >= origins[0]:
                    raise ValueError("复用模型的训练截止点不得进入推理窗口")
            publication_check = main_opt.resource_publication_check(resource_bundle, first_issue, use_final=True)
            if not publication_check["passed"]:
                raise ValueError("复用资源模型不满足每日发布时间边界：" + "; ".join(publication_check["problems"]))
            summary["reused_models_from"] = str(reuse_run.resolve())
            short_bundle.process_recovery = dict(C.SHORT_PROCESS_RECOVERY)
            summary["preprocessing_recipe_updated"] = {"short_process_recovery": short_bundle.process_recovery}
            summary["validation_scope"] = "复用已有最终模型；神经网络按A/B/C冻结晋级，未评估D"
            stages.update(3)
            main_opt.refresh_physical_validation(resource_bundle, training, use_final=True)
        if C.GENERATOR1_SPECIALIST:
            summary["generator1"] = _train_generator1_final(training, short_bundle, long_bundle, quick=quick)
        summary["long_process_expert"] = main_l.train_process_expert(training, long_bundle, quick=quick)
        LOGGER.info("训练完成: short=%s/%s long=%s/%s heat_rate_ok=%s",
                    short_bundle.kind, short_bundle.features,
                    long_bundle.kind, long_bundle.features,
                    resource_bundle.heat_rate.get("ok"))

        stages.set_description("推理 · 批量预测")
        LOGGER.info("推理窗口: %s, 起报点=%d (%s .. %s)",
                    window, len(origins), origins[0], origins[-1])
        # 官方推理没有标签可见；评测窗口目标必须整体遮挡。
        mask_end = origins[-1] + pd.Timedelta(minutes=C.STEP_MINUTES)
        masked = prepared.masked_copy(origins[0], mask_end)
        short_pred = main_s.predict_short(short_bundle, masked, origins, use_final=True)
        long_pred = main_l.predict_long(long_bundle, masked, origins, use_final=True)
        prediction_report = output.write_predictions(ctx, origins, short_pred, long_pred)
        LOGGER.info("预测宽表: 短期 %s 列, 长期 %s 列",
                    prediction_report["short"]["columns"],
                    prediction_report["long"]["columns"])

        # 每日规划器只在提前发布时刻预测96步资源，不再构造日内刷新提交指令。
        resource_forecast = None
        stages.update(1)
        stages.set_description("推理 · 调度优化")
        optimization = main_opt.optimize(resource_forecast, masked, origins,
                                         long_pred, resource_bundle, quick=quick,
                                         use_final=True)
        optimization.setdefault("summary", {})["physical_validation"] = resource_bundle.final_physical_validation
        optimization["summary"]["submission_physical_check"] = main_opt.physical_submission_check(optimization)
        optimization_report = output.write_optimization(ctx, optimization.get("schedule"))
        stages.update(1)
        stages.set_description("推理 · 模型与结果归档")

        model_report = output.save_models(ctx, short_bundle, long_bundle,
                                          resource_bundle, optimization.get("summary", {}),
                                          use_final=True)
        calibration_report = output.write_calibration(
            ctx, short_bundle, long_bundle, resource_bundle, use_final=True)
        source_report = output.write_source_snapshot(ctx, C)
        submission_check = _inference_submission_check(
            window, optimization, quick=quick, experimental=experimental,
            coverage=output.validate_result_files(ctx.root, None if window == "half_test" else origins,
                                                  quick, require_optimization=True),
            source=source_report)
        submittable = submission_check["passed"]
        zip_report = output.build_zip(ctx, submittable=submittable, config=C)
        stages.update(1)
        stages.set_description("推理 · 保存摘要")

        summary.update({
            "status": "SUCCESS_INFER" if window == "half_test" else "SUCCESS_DRYRUN",
            "window": window,
            "data_usage": {"training_range": [str(training.index[0]), str(training.index[-1])],
                           "resource_training_range": [str(resource_training.index[0]), str(resource_training.index[-1])],
                           "first_dispatch_issue": str(first_issue),
                           "training_stages": list(training.stage_ranges),
                           "official_test_training_used": False,
                           "test_usage": ("frozen_model_official_inference" if window == "half_test"
                                          else "frozen_model_compatibility_dryrun")},
            "preprocessing": {"training": training.quality.get("preprocessing", {}),
                              "inference": prepared.quality.get("preprocessing", {})},
            "origins": int(len(origins)),
            "origin_range": [str(origins[0]), str(origins[-1])],
            "prediction_input_support": {
                "power_gas_missing_origins": int((~np.isfinite(masked.frame[["d_use_blast", "d_use_coke", "d_use_converter"]]
                    .to_numpy(dtype=float)[masked.index.get_indexer(origins)]).all(axis=1)).sum()),
                "short_fallback_active": bool(short_bundle.final_availability),
                "short_process_recovery": short_bundle.process_recovery,
                "long_fallback_active": bool(long_bundle.final_availability)},
            "short": {"scores": short_bundle.scores, "selection": short_bundle.selection,
                      "train_seconds": short_bundle.train_seconds},
            "long": {"scores": long_bundle.scores, "selection": long_bundle.selection,
                     "train_seconds": long_bundle.train_seconds},
            "prediction": prediction_report,
            "optimization": optimization_report,
            "optimization_summary": optimization.get("summary", {}),
            "resource_metrics": resource_bundle.metrics,
            "resource_selection": resource_bundle.selection,
            "models": model_report,
            "calibration": calibration_report,
            "source": source_report,
            "zip": zip_report,
            "submittable": submittable,
            "submission_check": submission_check,
            "reason": "; ".join(submission_check["problems"]) or None,
            "duration_seconds": round(time.time() - started, 2),
        })
        report_path = output.write_report(ctx, summary)
        if summary.get("generator1"):
            # 同时检查正常与缺测起报点，验证归档包含实际使用的全部分支。
            reloaded = output.load_final_models(ctx.root)
            sample = np.unique(np.array([0, min(96, len(origins) - 1), min(192, len(origins) - 1), len(origins) - 1]))
            short_full = main_s.predict_short(reloaded[0], masked, origins, use_final=True)
            long_full = main_l.predict_long(reloaded[1], masked, origins, use_final=True)
            differences = {"short": float(np.max(np.abs(short_full - short_pred))),
                           "long": float(np.max(np.abs(long_full - long_pred)))}
            if max(differences.values()) > 1e-5:
                raise ValueError("最终模型归档重载未复现包含缺测分支的推理")
            short_check = main_s.predict_short(reloaded[0], masked, origins[sample], use_final=True)
            long_check = main_l.predict_long(reloaded[1], masked, origins[sample], use_final=True)
            sampled = {"short": float(np.max(np.abs(short_check - short_full[sample]))),
                       "long": float(np.max(np.abs(long_check - long_full[sample])))}
            # FP32 在不同 GPU 批形状下可能换用卷积核；抽样误差与完整重载误差分开验收。
            tolerance = float(16 * np.finfo(np.float32).eps * max(C.CAPACITY.values()))
            if max(sampled.values()) > tolerance:
                raise ValueError("抽样与整批预测差异超过 FP32 数值预算")
            summary["availability_reload_check"] = {"passed": True, "origins": sample.tolist(),
                "max_abs_diff": differences, "sampled_batch_max_abs_diff": sampled,
                "sampled_tolerance_mw": tolerance}
            output.write_report(ctx, summary)
        stages.update(1)
        stages.set_description("推理完成")
        LOGGER.info("报告: %s", report_path)
        print("\n推理摘要")
        print("  阶段: 正式半决赛")
        print(f"  窗口: {window} | 起报点 {len(origins)}")
        print(f"  优化状态: {optimization.get('status')}")
        print(f"  输出目录: {ctx.root}")
        print(f"  半决赛结果包: {zip_report['name']}")
        print("  提交核验: " + ("通过" if submittable else "; ".join(submission_check["problems"])))
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("推理失败: %s", exc)
        LOGGER.error(traceback.format_exc())
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        summary["duration_seconds"] = round(time.time() - started, 2)
        try:
            output.write_report(ctx, summary)
        except Exception:
            pass
        return 1
    finally:
        stages.close()


def run_full(quick: bool = False, experimental: bool = False) -> int:
    started = time.time()
    output.training_cache_stats(reset=True)
    ctx = output.begin_run(C, mode="backtest", experimental=quick or experimental, quick=quick)
    _setup_logger(ctx.log_path)
    LOGGER.info("运行目录: %s", ctx.root)
    summary = {"run_id": ctx.run_id, "status": "FAILED", "mode": "BACKTEST",
               "experimental": bool(quick or experimental), "output_root": str(ctx.root),
               "quick": bool(quick), "validation_scope": "A/B开发验证" if quick else "历史开发验证，D已被多次查看", "errors": []}
    stages = C.progress(total=8, desc="回测 · 数据准备", unit="阶段", leave=True)
    try:
        scan = scan_sources(include_inference=False)
        LOGGER.info("扫描完成: %d 个阶段目录, %d 个文件",
                    len(scan["stages"]), len(scan["files"]))
        prepared = prepare_data(C)
        contract = data_contract_check(prepared)
        if not contract["passed"]:
            raise ValueError("数据契约检查失败: " + "; ".join(contract["problems"]))
        if not _sources_unchanged(scan):
            raise ValueError("源数据在读取后被修改")
        LOGGER.info("清洗完成: %s, rows=%d, 目标有效=%s",
                    prepared.quality["time_range"], prepared.quality["rows"],
                    prepared.quality["target_valid"])

        folds = build_folds(prepared, quick=quick)
        LOGGER.info("时序折: %s", [(f["name"], f["role"], round(f["label_coverage"], 3))
                                  for f in folds])
        stages.update(1)
        stages.set_description("回测 · 短期训练")
        short_bundle = main_s.train_short(prepared, folds, quick=quick)
        stages.update(1)
        stages.set_description("回测 · 长期训练")
        LOGGER.info("短期选模=%s/%s, 调参=%.3f, 确认=%.3f",
                    short_bundle.kind, short_bundle.features,
                    short_bundle.scores["tuning_mean"],
                    short_bundle.scores["confirm"]["accuracy"])
        long_bundle = main_l.train_long(prepared, folds, quick=quick)
        stages.update(1)
        stages.set_description("回测 · 批量预测")
        LOGGER.info("长期选模=%s/%s, 调参=%.3f, 确认=%.3f",
                    long_bundle.kind, long_bundle.features,
                    long_bundle.scores["tuning_mean"],
                    long_bundle.scores["confirm"]["accuracy"])

        confirm = next(f for f in folds if f["role"] == "confirm")
        mask_end = pd.Timestamp(confirm["start"]) + pd.Timedelta(days=int(confirm["days"]))
        masked = prepared.masked_copy(confirm["start"], mask_end)
        origins = pd.DatetimeIndex(confirm["origins"])
        short_pred = main_s.predict_short(short_bundle, masked, origins)
        long_pred = main_l.predict_long(long_bundle, masked, origins)
        prediction_report = output.write_predictions(ctx, origins, short_pred, long_pred)
        LOGGER.info("预测输出: %d 个起报点, 短期/长期校验通过", len(origins))
        stages.update(1)
        stages.set_description("回测 · 资源训练与预测")

        resource_bundle = main_opt.train_resources(prepared, folds, quick=quick)
        LOGGER.info("资源模型 WAPE: %s, heat_rate_ok=%s",
                    {g: round(v["wape"], 3) for g, v in resource_bundle.metrics.items()},
                    resource_bundle.heat_rate.get("ok"))
        resource_forecast = main_opt.forecast_resources(resource_bundle, masked, origins)
        stages.update(1)
        stages.set_description("回测 · 调度优化")
        optimization = main_opt.optimize(resource_forecast, masked, origins,
                                         long_pred, resource_bundle, quick=quick)
        optimization.setdefault("summary", {})["physical_validation"] = resource_bundle.physical_validation
        optimization["summary"]["submission_physical_check"] = main_opt.physical_submission_check(optimization)
        optimization_report = output.write_optimization(ctx, optimization.get("schedule"))
        stages.update(1)
        stages.set_description("回测 · 模型与结果归档")
        LOGGER.info("优化状态: %s, 原因=%s", optimization.get("status"),
                    optimization.get("summary", {}).get("reason"))
        opt_summary = optimization.get("summary", {})
        if opt_summary.get("revenue_improvement_ratio") is not None:
            LOGGER.info("优化回测: plan_feasible=%s, 计划收益对照=%.3f%%, 包络回放低柜步数=%s",
                        opt_summary.get("plan_feasible"),
                        100.0 * (opt_summary.get("revenue_improvement_ratio") or 0.0),
                        opt_summary.get("realized_holder_low_steps"))
        elif opt_summary.get("dispatch_mode") == "rolling_proxy":
            LOGGER.info("连续代理控制: 核验=%s，代理执行收益=%s，降载MWh=%.3f",
                        opt_summary.get("plan_validation_passed"), opt_summary.get("proxy_executed_revenue_cny"),
                        opt_summary.get("proxy_curtailed_generation_mwh", 0))

        model_report = output.save_models(ctx, short_bundle, long_bundle,
                                          resource_bundle, optimization.get("summary", {}))
        calibration_report = output.write_calibration(
            ctx, short_bundle, long_bundle, resource_bundle)
        source_report = output.write_source_snapshot(ctx, C)
        zip_report = output.build_zip(ctx, submittable=False, config=C)
        stages.update(1)
        stages.set_description("回测 · 保存摘要")

        summary.update({
            "status": "SUCCESS_MOCK",
            "data": {
                "available_data_range": prepared.quality["time_range"],
                "eval_range": [str(origins[0]), str(origins[-1])],
                "rows": prepared.quality["rows"],
                "target_valid": prepared.quality["target_valid"],
                "target_missing_ratio": prepared.quality["target_missing_ratio"],
                "target_invalid": prepared.quality["target_invalid"],
                "duplicate_conflicts": prepared.quality["duplicate_conflicts"],
                "key_missing_ratio": prepared.quality["key_missing_ratio"],
                "preprocessing": prepared.quality.get("preprocessing", {}),
                "causal_contract": {"passed": True, "target_mask": "整个开发验证窗口" if quick else "整个历史确认窗口",
                                    "future_dynamic_fields": "forbidden"},
                "sources_unchanged": True,
            },
            "short": {
                "selected_model": short_bundle.selection["chosen"].get(
                    "effective", f"{short_bundle.kind}/{short_bundle.features}"),
                "cv_score_1_minus_mape": short_bundle.scores["tuning_mean"],
                "confirm_score_1_minus_mape": short_bundle.scores["confirm"]["accuracy"],
                "targets": short_bundle.scores["confirm"]["targets"],
                "blend": short_bundle.blend,
                "train_seconds": round(short_bundle.train_seconds, 2),
                "train_rows": short_bundle.train_rows,
                "candidates": short_bundle.selection.get("candidates", []),
            },
            "long": {
                "selected_model": long_bundle.selection["chosen"].get(
                    "effective", f"{long_bundle.kind}/{long_bundle.features}"),
                "cv_score_1_minus_mape": long_bundle.scores["tuning_mean"],
                "confirm_score_1_minus_mape": long_bundle.scores["confirm"]["accuracy"],
                "targets": long_bundle.scores["confirm"]["targets"],
                "blend": long_bundle.blend,
                "train_seconds": round(long_bundle.train_seconds, 2),
                "train_rows": long_bundle.train_rows,
                "candidates": long_bundle.selection.get("candidates", []),
            },
            "optimization": optimization.get("summary", {"status": optimization.get("status")}),
            "resource_metrics": resource_bundle.metrics,
            "resource_selection": resource_bundle.selection,
            "output": {
                "prediction": prediction_report,
                "optimization": optimization_report,
                "models": model_report,
                "calibration": calibration_report,
                "source": source_report,
                "zip": zip_report,
                "submittable": False,
                "reason": "历史时序回测，分数和结果包不代表半决赛测试集结果",
            },
            "duration_seconds": round(time.time() - started, 2),
        })
        report_path = output.write_report(ctx, summary)
        stages.update(1)
        stages.set_description("回测完成")
        LOGGER.info("报告: %s", report_path)
        LOGGER.info("完成: %.1fs, ZIP=%s", summary["duration_seconds"], zip_report["name"])
        print("\n运行摘要")
        print(f"  短期 1-MAPE: 调参 {short_bundle.scores['tuning_mean']:.3f} | "
              f"确认 {short_bundle.scores['confirm']['accuracy']:.3f}")
        print(f"  长期 1-MAPE: 调参 {long_bundle.scores['tuning_mean']:.3f} | "
              f"确认 {long_bundle.scores['confirm']['accuracy']:.3f}")
        print(f"  优化状态: {optimization.get('status')} | "
              f"计划可行={optimization.get('summary', {}).get('plan_feasible')}（收益口径见 JSON）")
        print(f"  输出目录: {ctx.root}")
        print(f"  历史回测包: {zip_report['name']}（仅用于历史验证）")
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("运行失败: %s", exc)
        LOGGER.error(traceback.format_exc())
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        summary["duration_seconds"] = round(time.time() - started, 2)
        try:
            output.write_report(ctx, summary)
        except Exception:
            pass
        return 1
    finally:
        stages.close()


def _masked_fold_view(prepared, spec):
    horizons = np.asarray(C.HORIZONS_L)
    positions, evaluation, ref = main_s._fold_positions(prepared, spec, int(horizons.max()))
    if evaluation[-1] + horizons.max() >= len(prepared.index):
        raise ValueError("开发折标签延伸不完整")
    end = prepared.index[evaluation[-1] + horizons.max()]
    masked = prepared.masked_copy(spec["start"], end)
    return masked, positions, evaluation, ref


def _train_generator1_final(training, short_bundle, long_bundle, quick=False, recipes_override=None):
    import main_nn
    import copy
    recipes = copy.deepcopy(C.GENERATOR1_SPECIALIST if recipes_override is None else recipes_override)
    training.require_training_data()
    end, ref = str(training.index[-1]), len(training.index) - 1
    result = {}
    for span, bundle in (("short", short_bundle), ("long", long_bundle)):
        recipe = recipes.get(span)
        if not recipe:
            continue
        existing = getattr(bundle, "final_generator1", None)
        if existing and existing["recipe"] == recipe and existing["train_end"] == end:
            result[span] = {"recipe": recipe, "reused": True, "train_end": end}
            continue
        if (existing and existing["train_end"] == end and recipe["kind"] == existing["recipe"]["kind"] == "refined_nn"
                and recipe["base_recipe"] == existing["recipe"]["base_recipe"] and recipe["refinement"] == existing["recipe"]["refinement"]):
            state = dict(existing)
            state["recipe"] = recipe
            state.pop("calibration", None)
            if recipe.get("calibration"):
                state["calibration"] = _train_final_calibration(training, short_bundle, long_bundle, span, recipe, quick=quick)
            bundle.final_generator1 = state
            bundle.selection["generator1"] = {"candidate": recipe["candidate"], "source_study": recipe["source_study"]}
            result[span] = {"recipe": recipe, "train_end": end, "neural_components_reused": True,
                "calibration_fit_run": state.get("calibration", {}).get("fit_run"), "D_reassessed": False}
            continue
        if recipe["kind"] == "refined_nn":
            if recipe["base_recipe"]["kind"] == "refined_nn":
                raise ValueError("精修最终训练不允许多层嵌套")
            temporary = copy.copy(bundle)
            temporary.selection = copy.deepcopy(bundle.selection)
            pair = (temporary, long_bundle) if span == "short" else (short_bundle, temporary)
            _train_generator1_final(training, *pair, quick=quick,
                recipes_override={span: recipe["base_recipe"]})
            base_state = temporary.final_generator1
            update = recipe["refinement"]
            neural_recipe = {"kind": "nn", "nn_kind": update["nn_kind"], "config": update["config"],
                "seeds": update["seeds"], "model_stage": update["model_stage"], "weight": 1.,
                "missing_only": False, "candidate": recipe["candidate"], "source_study": recipe["source_study"]}
            temporary.final_generator1 = None
            _train_generator1_final(training, *pair, quick=quick, recipes_override={span: neural_recipe})
            bundle.final_generator1 = {"recipe": recipe, "train_end": end,
                "base_state": base_state, "nn": temporary.final_generator1["nn"]}
            if recipe.get("calibration"):
                bundle.final_generator1["calibration"] = _train_final_calibration(training, short_bundle, long_bundle, span, recipe, quick=quick)
            bundle.selection["generator1"] = {"candidate": recipe["candidate"], "source_study": recipe["source_study"]}
            result[span] = {"recipe": recipe, "train_end": end, "base_reused": existing is not None, "D_reassessed": False}
            continue
        if existing and existing["train_end"] == end and existing["recipe"]["kind"] == recipe["kind"] == "nn":
            before = dict(existing["recipe"]["config"])
            after = dict(recipe["config"])
            for key in ("ensemble_aggregation", "aggregation_power"):
                before.pop(key, None)
                after.pop(key, None)
            if before == after and existing["recipe"]["nn_kind"] == recipe["nn_kind"]:
                state = dict(existing)
                state["recipe"] = recipe
                state["nn"] = copy.copy(existing["nn"])
                state["nn"].config = {**existing["nn"].config, **recipe["config"]}
                bundle.final_generator1 = state
                result[span] = {"recipe": recipe, "reused": True, "train_end": end,
                    "refinement_operator_only": True, "D_reassessed": False}
                continue
        tick = time.time()
        LOGGER.info("冻结generator_1最终训练 %s/%s，截止=%s", span, recipe["kind"], end)
        state = {"recipe": recipe, "train_end": end}
        if recipe["kind"] == "nn":
            view = training
            if recipe["config"].get("power_mask_training", True):
                for offset in range(7 * C.STEPS_PER_DAY, ref, 21 * C.STEPS_PER_DAY):
                    view = view.masked_power_gas_copy(view.index[offset], view.index[min(offset + 14 * C.STEPS_PER_DAY - 1, ref)])
            state["nn"] = main_nn.train_nn(view, [{"name": "final", "role": "tuning", "train_end": end}],
                recipe["nn_kind"], bundle.horizons, quick=quick, n_seeds=recipe["seeds"], cfg=recipe["config"])
        else:
            stride = C.QUICK_TRAIN_STRIDE if quick else (1 if span == "short" else 2)
            positions = np.arange(C.PROFILE_DAYS * C.STEPS_PER_DAY, ref - max(bundle.horizons) + 1, stride)
            experts = []
            for seed in range(int(recipe.get("seeds", 1))):
                available = getattr(bundle, "final_availability", None) or {}
                if (seed == 0 and not quick and available.get("kind") == recipe["expert_kind"]
                        and available.get("features") == recipe["expert_features"]
                        and available.get("train_end") == end):
                    # 当前缺测专家已覆盖相同合法前缀，只复用同配方的首种子。
                    models = {k: v for k, v in available["models"].items()
                              if (k[0] if isinstance(k, tuple) else k) == "generator_1"}
                    names = available["feature_names"]
                    if not models:
                        raise ValueError("可复用的generator_1专家缺少目标模型")
                    state["reused_expert_seeds"] = [20261021]
                else:
                    models, names = main_s._train_models(training,
                        main_s.build_base_features(training, True), positions, bundle.horizons,
                        recipe["expert_kind"], recipe["expert_features"],
                        C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG, ref,
                        targets=("generator_1",), params_override={"seed": 20261021 + seed})
                experts.append({"models": models, "feature_names": names})
            if len(experts) == 1:
                state.update(experts[0])
            else:
                state["experts"] = experts
            state.update({k: recipe[k] for k in ("expert_kind", "expert_features", "expert_blend")})
        bundle.final_generator1 = state
        bundle.selection["generator1"] = {"candidate": recipe["candidate"], "source_study": recipe["source_study"]}
        result[span] = {"recipe": recipe, "train_end": end, "seconds": time.time() - tick,
            "reused_expert_seeds": state.get("reused_expert_seeds", []), "D_reassessed": False}
    return result




def _calibration_contract(template, specialist, quick):
    import main_nn
    neural = template.final_nn_bundle
    return {"primary": {k: getattr(template, k) for k in
            ("kind", "features", "blend", "lgb_kind", "lgb_features", "nn_weights")},
        "neural": {"kind": neural.kind, "config": neural.config, "seeds": len(neural.seed_states),
                   "runtime": main_nn.training_runtime()},
        "availability": {k: v for k, v in template.final_availability.items()
                         if k not in ("models", "retained_state", "feature_names", "train_end", "role")},
        "specialist": specialist, "quick": bool(quick)}


def _train_final_calibration(training, short_bundle, long_bundle, span, recipe, quick=False):
    import copy
    if span != "short":
        raise ValueError("正式阶段校准只启用已晋级的短期配方")
    training.require_training_data()
    templates = {"short": short_bundle, "long": long_bundle}
    if templates[span].kind != "nn_lgb_ens" or pd.Timestamp(templates[span].final_train_end) != training.index[-1]:
        raise ValueError("最终校准要求同截止点的完整融合主模型")
    cache = output.training_cache_path("calibration", _calibration_contract(templates[span], recipe, quick),
                                       frames=(training.frame, training.labels))
    artifact = output.read_training_cache(cache)
    if artifact is not None:
        return json.loads(artifact.read_text(encoding="utf-8"))
    ctx = output.begin_run(C, mode="final_calibration_" + span, experimental=True, quick=quick)
    ctx.log_path.write_text("最终阶段权重表训练；完整训练日志同时写入主推理日志。\n", encoding="utf-8")
    base_recipe = copy.deepcopy(recipe)
    base_recipe.pop("calibration", None)
    summary = {"run_id": ctx.run_id, "status": "RUNNING", "submittable": False, "quick": bool(quick),
        "data_sources": training.source_hashes, "base_recipes": {span: base_recipe},
        "mode": "final_calibration_fit_only; no_validation_reselection", "inner_provenance": []}
    end = str(training.index[-1])
    outer = {"name": "final", "start": str(training.index[-1] + pd.Timedelta(minutes=15)), "train_end": end}
    options = recipe["calibration"]
    started = time.time()
    try:
        _fit_calibration_prefixes(ctx, summary, training, [outer], templates, spans=(span,),
            offsets=tuple(options["final_offsets_days"]), days=options["final_window_days"], quick=quick)
        data = _calibration_training_rows(ctx, training, outer, span,
            offsets=tuple(options["final_offsets_days"]), days=options["final_window_days"])
        pack = _fit_transition_calibrator(ctx.model_dir / "calibration", data, options["family"], outer)
        calibration = {k: pack[k] for k in ("family", "feature_names", "train_end", "table", "provenance")}
        calibration.update(shrink=options["shrink"], mode=options["mode"], fit_run=str(ctx.root.resolve()))
        maximum = max(p["max_target"] for p in pack["provenance"])
        if pd.Timestamp(maximum) > training.index[-1]:
            raise ValueError("最终阶段校准使用了训练截止之后的标签")
        output.write_source_snapshot(ctx, C)
        summary.update(status="SUCCESS_FINAL_CALIBRATION", calibration=calibration,
            max_supervision_target=maximum, official_test_used=False, parameters_reselected=False)
        ctx.log_path.write_text("最终权重表已完成；只拟合冻结规则，不重新选参。\n最大监督目标=" + maximum + "\n", encoding="utf-8")
    except Exception as exc:
        summary.update(status="FAILED", errors=[str(exc)])
        raise
    finally:
        summary["duration_seconds"] = time.time() - started
        output.write_report(ctx, summary)
    def write_calibration(directory):
        path = directory / "calibration.json"
        path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    output.write_training_cache(cache, write_calibration)
    return calibration


def _load_calibration_bundle(directory, span):
    """重载阶段校准使用的完整预测组合。"""
    state = main_s.load_generator1(directory / "primary")
    recipe = state["recipe"]
    cls, horizons = main_s.ShortBundle, C.HORIZONS_S
    bundle = cls(**recipe, horizons=horizons, models={}, final_models=state["models"],
        final_nn_bundle=state.get("nn"), feature_names=state["feature_names"],
        train_end=state["train_end"], final_train_end=state["train_end"])
    def restore(item):
        if "base_state" in item:
            item["retained_state"] = restore(item.pop("base_state"))
        return item
    bundle.final_availability = restore(main_s.load_generator1(directory / "availability"))
    bundle.final_generator1 = main_s.load_generator1(directory / "specialist")
    return bundle


def _fit_calibration_bundle(directory, prepared, spec, span, templates, specialist_recipes, quick=False):
    import copy
    import shutil
    import main_nn
    training = prepared.prefix_copy(spec["train_end"])
    training.require_training_data()
    signature = output.training_fingerprint(_calibration_contract(templates[span], specialist_recipes[span], quick),
                                            frames=(training.frame, training.labels))
    if not (directory / "complete.json").exists():
        for path in directory.parent.glob("*/complete.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("signature") == signature:
                directory.mkdir(parents=True, exist_ok=True)
                for component in ("primary", "availability", "specialist"):
                    shutil.copytree(path.parent / component, directory / component, dirs_exist_ok=True)
                record["reused_from_identical_prefix"] = str(path.parent.resolve())
                (directory / "complete.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                break
    if (directory / "complete.json").exists():
        record = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
        if record.get("signature") != signature:
            raise ValueError("内层完整组合的数据、实现或冻结配方已变化，不能复用")
        return _load_calibration_bundle(directory, span)
    ref, end = len(training.index) - 1, str(training.index[-1])
    cls, horizons = main_s.ShortBundle, C.HORIZONS_S
    stride = C.QUICK_TRAIN_STRIDE if quick else C.TRAIN_STRIDE_SHORT
    template = templates[span]
    recipe = {k: copy.deepcopy(getattr(template, k)) for k in
              ("kind", "features", "blend", "lgb_kind", "lgb_features", "nn_weights")}
    LOGGER.info("完整组合内层训练 %s/%s，标签截止=%s", span, spec["name"], end)
    positions = np.arange(C.PROFILE_DAYS * C.STEPS_PER_DAY, ref - max(horizons) + 1, stride)
    models, names = main_s._train_models(training, main_s.build_base_features(training, True),
        positions, horizons, recipe["lgb_kind"], recipe["lgb_features"],
        C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG, ref)
    neural = main_nn.train_nn(training, [{**spec, "role": "tuning"}], template.final_nn_bundle.kind,
        horizons, quick=quick, n_seeds=len(template.final_nn_bundle.seed_states), cfg=template.final_nn_bundle.config)
    bundle = cls(**recipe, horizons=horizons, models={}, final_models=models,
        feature_names=names, final_nn_bundle=neural, train_end=end, final_train_end=end)
    fallback = {k: copy.deepcopy(v) for k, v in template.final_availability.items()
                if k not in ("models", "retained_state", "feature_names", "train_end", "role")}
    bundle.final_availability = main_s.train_availability_fallback(training, horizons, fallback, quick=quick)
    _train_generator1_final(training, bundle if span == "short" else None,
        bundle if span == "long" else None, quick=quick, recipes_override={span: specialist_recipes[span]})
    main_s.save_generator1({"recipe": recipe, "train_end": end, "feature_names": names,
        "models": models, "nn": neural}, directory / "primary")
    def transport(item):
        state = {k: v for k, v in item.items() if k != "retained_state"}
        if item.get("retained_state"):
            state["base_state"] = transport(item["retained_state"])
        return state
    main_s.save_generator1(transport(bundle.final_availability), directory / "availability")
    main_s.save_generator1(bundle.final_generator1, directory / "specialist")
    record = {"train_end": end, "max_training_target": str(training.index[positions[-1] + max(horizons)]),
              "specialist_recipe": specialist_recipes[span], "quick": bool(quick),
              "all_components_retrained_on_prefix": True, "signature": signature}
    (directory / "complete.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle


def _calibration_cases(prepared, spec):
    view, _, positions, _ = _masked_fold_view(prepared, spec)
    for scenario in ("normal", "all_missing", "outage_day7"):
        current = view if scenario == "normal" else view.masked_power_gas_copy(
            pd.Timestamp(spec["start"]) + pd.Timedelta(days=7 if scenario == "outage_day7" else 0))
        yield scenario, current, positions


def _calibration_components(bundle, view, positions, span):
    import copy
    import main_nn
    predictor = main_s.predict_short if span == "short" else main_l.predict_long
    prediction = predictor(bundle, view, view.index[positions], use_final=True)
    original = bundle.final_generator1
    if original["recipe"]["kind"] != "refined_nn":
        raise ValueError("状态融合要求已冻结的双网络精修组合")
    primary = copy.copy(bundle)
    primary.final_generator1 = original["base_state"]
    before = predictor(primary, view, view.index[positions], use_final=True)[..., 0]
    after = main_nn.predict_nn(original["nn"], view, view.index[positions])[..., 0]
    return prediction, before, after



def _calibration_training_rows(ctx, prepared, outer, span, offsets=(63, 35), days=21):
    horizons = C.HORIZONS_S
    Xs, ys, weights, befores, afters, missing_rows, provenance = [], [], [], [], [], [], []
    names = None
    for offset in offsets:
        start = pd.Timestamp(outer["start"]) - pd.Timedelta(days=offset)
        spec = {"name": f"{outer['name']}_inner{offset}", "start": str(start), "days": days,
                "train_end": str(start - pd.Timedelta(minutes=15))}
        for scenario, view, positions in _calibration_cases(prepared, spec):
            folder = ctx.model_dir / "inner_load" / span / spec["name"]
            with np.load(folder / f"{scenario}.npz") as data:
                if not np.array_equal(positions, data["positions"]):
                    raise ValueError("内层校正起报索引变化")
                # 起报点下采样只减少拟合冗余，不裁剪96步监督路径。
                take = np.arange(0, len(positions), 4)
                selected = positions[take]
                prediction, truth = data["prediction"][take], data["truth"][take]
                before, after = data["before"][take], data["after"][take]
            actual = main_s.truth_cube(prepared, selected, horizons)
            if not np.array_equal(actual, truth, equal_nan=True):
                raise ValueError("内层校正真值变化")
            X, columns = main_s.generator1_gate_matrix(view, selected, horizons, before, after)
            if names is not None and names != columns:
                raise ValueError("内层校正特征不一致")
            names = columns
            y = truth[..., 0].T.ravel()
            current = prediction[..., 0].T.ravel()
            visible = np.isfinite(view.frame[["d_use_blast", "d_use_coke", "d_use_converter"]].to_numpy()[selected]).all(axis=1)
            missing = np.tile(~visible, len(horizons))
            target_positions = (selected[:, None] + np.asarray(horizons)[None, :]).T.ravel()
            ref = prepared.index.get_loc(pd.Timestamp(outer["train_end"]))
            if target_positions.max() > ref:
                raise ValueError("校正监督标签跨入外层窗口")
            decay = np.power(.5, (ref - target_positions) / (60 * C.STEPS_PER_DAY))
            valid = np.isfinite(y) & np.isfinite(current)
            Xs.append(X[valid]); ys.append(y[valid]); weights.append(decay[valid])
            befores.append(before.T.ravel()[valid])
            afters.append(after.T.ravel()[valid]); missing_rows.append(missing[valid])
            provenance.append({"spec": spec, "scenario": scenario, "max_target": str(prepared.index[target_positions.max()]),
                               "rows": int(valid.sum()), "source": str(folder)})
    return {"X": np.concatenate(Xs), "y": np.concatenate(ys), "decay": np.concatenate(weights),
        "before": np.concatenate(befores), "after": np.concatenate(afters), "missing": np.concatenate(missing_rows),
        "feature_names": names, "provenance": provenance}


def _fit_transition_calibrator(directory, data, family, outer):
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "meta.json"
    if meta_path.exists():
        pack = json.loads(meta_path.read_text(encoding="utf-8"))
        if pack["train_end"] != outer["train_end"] or pack["feature_names"] != data["feature_names"]:
            raise ValueError("校正模型续跑契约变化")
        return pack
    X, y, before, after = (data[k] for k in ("X", "y", "before", "after"))
    pack = {"family": family, "feature_names": data["feature_names"], "train_end": outer["train_end"],
        "provenance": data["provenance"]}
    if family != "duration_state":
        raise ValueError("正式阶段校准只支持已晋级的 duration_state 配方")
    bins = main_s.generator1_transition_bins(X, data["feature_names"])
    table = []
    for group in range(12):
        valid = data["missing"] & (bins == group) & (y > 0)
        grid = (0., .125, .25, .375, .5)
        loss = [float(np.average(np.abs(before[valid] + w * (after[valid] - before[valid]) - y[valid]) /
                y[valid], weights=data["decay"][valid])) if valid.sum() >= 384 else abs(w - .25) for w in grid]
        table.append(grid[int(np.argmin(loss))])
    pack["table"] = table
    meta_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    return pack


def _fit_calibration_prefixes(ctx, summary, prepared, specs, templates, spans=("short",), offsets=(63, 35), days=21, quick=False):
    import main_nn
    for outer in specs:
        for offset in offsets:
            start = pd.Timestamp(outer["start"]) - pd.Timedelta(days=offset)
            spec = {"name": f"{outer['name']}_inner{offset}", "start": str(start), "days": days,
                    "train_end": str(start - pd.Timedelta(minutes=C.STEP_MINUTES)), "role": "tuning"}
            for span in spans:
                directory = ctx.model_dir / "inner_load" / span / spec["name"]
                horizons = C.HORIZONS_S
                bundle = _fit_calibration_bundle(directory, prepared, spec, span, templates, summary["base_recipes"], quick=quick)
                loaded = _load_calibration_bundle(directory, span)
                audits = []
                for scenario, view, positions in _calibration_cases(prepared, spec):
                    prediction, before, after = _calibration_components(bundle, view, positions, span)
                    truth = main_s.truth_cube(prepared, positions, horizons)
                    maximum = prepared.index[positions[-1] + max(horizons)]
                    if maximum > pd.Timestamp(outer["train_end"]):
                        raise ValueError("内层完整组合评分标签越过外层训练截止")
                    predictor = main_s.predict_short if span == "short" else main_l.predict_long
                    probe = np.arange(0, len(positions), 251)
                    restored = predictor(loaded, view, view.index[positions[probe]], use_final=True)
                    difference = float(np.max(np.abs(restored - prediction[probe])))
                    if difference > .002:
                        raise ValueError("内层完整组合重载差异超过0.002MW")
                    np.savez_compressed(directory / f"{scenario}.npz", prediction=prediction, before=before,
                        after=after, truth=truth, positions=positions)
                    audits.append({"scenario": scenario, "origins": len(positions), "max_target": str(maximum),
                                   "reload_max_abs_diff_mw": difference})
                record = {"outer": outer["name"], "span": span, "spec": spec, "scenarios": audits}
                summary.setdefault("inner_provenance", []).append(record)
                output.write_report(ctx, summary)
                del bundle, loaded
                main_nn.clear_cache()



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AIC 半决赛煤气发电预测与优化")
    parser.add_argument("command", nargs="?", default="infer",
                        choices=("run", "infer", "doctor", "self-check", "verify"),
                        help="默认 infer 执行半决赛测试推理；run 执行历史回测")
    parser.add_argument("--run-dir", type=Path, help="verify 指定选定运行或实验结果目录")
    parser.add_argument("--quick", action="store_true", help="单折、低迭代的开发模式")
    progress_args = parser.add_mutually_exclusive_group()
    progress_args.add_argument("--progress", dest="progress", action="store_true",
                               help="强制显示进度条（适用于 IDE 控制台）")
    progress_args.add_argument("--no-progress", dest="progress", action="store_false",
                               help="关闭进度条，仅保留日志")
    parser.set_defaults(progress=None)
    parser.add_argument("--experiment", action="store_true", help="全量开发实验写入 tmp/tmp_outputs")
    parser.add_argument("--reuse-run", type=Path, help="infer复用已归档的正式模型")
    args = parser.parse_args(argv)
    C.PROGRESS_ENABLED = args.progress
    if args.reuse_run and args.command != "infer":
        parser.error("--reuse-run 只用于 infer")
    if args.command == "doctor":
        return run_doctor()
    if args.command == "infer":
        return run_infer(quick=args.quick, experimental=args.experiment, reuse_run=args.reuse_run)
    if args.command == "self-check":
        return run_self_check()
    if args.command == "verify":
        return run_verify(args.run_dir)
    return run_full(quick=args.quick, experimental=args.experiment)


if __name__ == "__main__":
    # Windows 重定向输出可能默认 GBK；统一编码，避免 m³ 等单位使已完成运行报错。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.exit(main())
