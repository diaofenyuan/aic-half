"""半决赛唯一日常入口：python main.py

默认 infer 在历史验证和冻结训练后，对正式半决赛测试窗口生成预测与优化结果；
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
        truth_pos = np.clip(truth_pos, 0, len(prepared.labels) - 1)
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
    for name in ("s_result.csv", "l_result.csv", "opt_result.csv"):
        path = directory / name
        if not path.is_file():
            if name != "opt_result.csv":
                problems.append(f"缺少 {name}")
            continue
        frame = pd.read_csv(path)
        check = output.validate_official_frame(name, frame)
        problems.extend(check["problems"])
        if frame.empty:
            problems.append(f"{name} 为空")
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
                for name in ("s_result.csv", "l_result.csv", "opt_result.csv"):
                    if (directory / name).is_file() and packed.read(name) != (directory / name).read_bytes():
                        problems.append(f"ZIP 内 {name} 与磁盘不一致")
        except Exception as exc:
            problems.append(f"ZIP 读取失败: {exc}")
    print(json.dumps({"directory": str(directory), "passed": not problems,
                      "problems": problems}, ensure_ascii=False, indent=2))
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
                                quick: bool, experimental: bool) -> dict:
    """正式参赛阶段与当次产物是否通过提交核验分别记录，避免沿用预开发总开关。"""
    problems = []
    if window != "half_test":
        problems.append("未使用完整 half_test 半决赛评测窗口")
    if quick:
        problems.append("quick 仅用于开发验证，调度未覆盖完整评测窗口")
    if experimental:
        problems.append("本次为开发实验，未归档为选定正式运行")
    if not C.OPT_CONFIG["template_confirmed"]:
        problems.append("优化输出字段尚未核对官方要求")
    if optimization.get("schedule") is None:
        problems.append("缺少有效 opt_result.csv 调度结果")
    opt_summary = optimization.get("summary", {})
    if not opt_summary.get("plan_validation_passed"):
        problems.append("优化计划约束核验未通过")
    if not optimization.get("submittable", False):
        if optimization.get("status") == "SIMULATED_NOT_SUBMITTABLE":
            problems.append("发电优化仍为代理资源模拟计划，尚未通过正式物理约束核验")
        else:
            problems.append("优化提交核验未通过：" + str(
                opt_summary.get("reason") or optimization.get("status") or "未返回核验状态"))
    return {"passed": not problems, "problems": problems}


def run_infer(quick: bool = False, experimental: bool = False,
              reuse_run: Path | None = None, nn_studies: list[Path] | None = None) -> int:
    """冻结模型后执行正式半决赛测试推理，并单独核验当次结果的提交条件。"""
    started = time.time()
    ctx = output.begin_run(C, mode="infer", experimental=quick or experimental)
    _setup_logger(ctx.log_path)
    LOGGER.info("推理运行目录: %s", ctx.root)
    summary = {"run_id": ctx.run_id, "status": "FAILED", "mode": "INFER",
               "competition_stage": "SEMIFINAL",
               "experimental": bool(quick or experimental), "output_root": str(ctx.root),
               "quick": bool(quick), "validation_scope": "A/B开发验证" if quick else "历史开发验证，D已被多次查看", "errors": []}
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
        # 训练重新读取独立白名单，测试期过程量也不能用于资源拟合或物理标定。
        training = prepare_data(C).prefix_copy(min(
            pd.Timestamp(C.TRAIN_END), origins[0] - pd.Timedelta(minutes=C.STEP_MINUTES)))
        LOGGER.info("训练白名单: %s .. %s；半决赛测试输入仅用于冻结模型推理",
                    training.index[0], training.index[-1])
        stages.update(1)
        if reuse_run is None:
            folds = build_folds(training, quick=quick)
            stages.set_description("推理 · 短期训练")
            short_bundle = main_s.train_short(training, folds, quick=quick)
            stages.update(1)
            stages.set_description("推理 · 长期训练")
            long_bundle = main_l.train_long(training, folds, quick=quick)
            stages.update(1)
            stages.set_description("推理 · 资源训练")
            resource_bundle = main_opt.train_resources(training, folds, quick=quick)
            stages.update(1)
        else:
            stages.set_description("推理 · 复用模型与冻结 NN")
            import main_nn
            short_bundle, long_bundle, resource_bundle = output.load_final_models(reuse_run)
            for bundle in (short_bundle, long_bundle, resource_bundle):
                if pd.Timestamp(bundle.final_train_end) >= origins[0]:
                    raise ValueError("复用模型的训练截止点不得进入推理窗口")
            summary["reused_models_from"] = str(reuse_run.resolve())
            summary["validation_scope"] = "复用已有最终模型；神经网络按A/B/C冻结晋级，未评估D"
            summary["nn_promotion"] = []
            for study in nn_studies or []:
                result = json.loads((study / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
                if result.get("status") != "SUCCESS_NN_STUDY" or set(result["folds"]) != {"A", "B", "C"}:
                    raise ValueError("冻结晋级需要完整A/B/C成功专项结果")
                eligible = [s for s in result["selection"] if s.get("passed_promotion")]
                if not eligible:
                    summary["nn_promotion"].append({"study": str(study), "promoted": False})
                    continue
                chosen = max(eligible, key=lambda s: s["mean"])
                bundle = short_bundle if result["span"] == "short" else long_bundle
                horizon = C.HORIZONS_S if result["span"] == "short" else C.HORIZONS_L
                neural = main_nn.train_nn(training, [{"role": "tuning", "train_end": str(training.index[-1])}],
                    result["kind"], horizon, n_seeds=int(result.get("n_seeds",
                        len(result["results"][0]["training"]["early_stop"]["epochs"]))),
                    cfg=result["config_overrides"])
                old_kind, old_features = bundle.kind, bundle.features
                bundle.kind = "nn_lgb_ens" if chosen["kind"] == "ensemble" else "nn"
                bundle.features, bundle.blend = ("hybrid" if chosen["kind"] == "ensemble" else "nn"), chosen["blend"]
                bundle.nn_weights = chosen.get("nn_weights", {})
                bundle.final_nn_bundle = neural
                bundle.lgb_kind, bundle.lgb_features = old_kind, old_features
                if bundle.kind == "nn":
                    bundle.models, bundle.final_models, bundle.feature_names = {}, {}, []
                bundle.scores = {"tuning_mean": chosen["mean"], "confirm": {"accuracy": None,
                    "reason": "本轮未评估D"}, "development": chosen["scores"]}
                bundle.selection = {"chosen": {"kind": bundle.kind, "features": bundle.features},
                    "candidates": result["selection"], "frozen_study": str(study)}
                bundle.train_seconds = neural.train_seconds
                summary["nn_promotion"].append({"study": str(study), "promoted": True, "kind": bundle.kind})
            stages.update(3)
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
        if C.LONG_HEAD_FROM_SHORT:
            k = len(C.HORIZONS_S)
            long_pred[:, :k, :] = short_pred
        prediction_report = output.write_predictions(ctx, origins, short_pred, long_pred)
        LOGGER.info("预测宽表: 短期 %s 列, 长期 %s 列",
                    prediction_report["short"]["columns"],
                    prediction_report["long"]["columns"])

        resource_forecast = main_opt.forecast_resources(
            resource_bundle, masked, origins, use_final=True)
        stages.update(1)
        stages.set_description("推理 · 调度优化")
        optimization = main_opt.optimize(resource_forecast, masked, origins,
                                         long_pred, resource_bundle, quick=quick,
                                         use_final=True)
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
            window, optimization, quick=quick, experimental=experimental)
        submittable = submission_check["passed"]
        zip_report = output.build_zip(ctx, submittable=submittable, config=C)
        stages.update(1)
        stages.set_description("推理 · 保存摘要")

        summary.update({
            "status": "SUCCESS_INFER" if window == "half_test" else "SUCCESS_DRYRUN",
            "window": window,
            "data_usage": {"training_range": [str(training.index[0]), str(training.index[-1])],
                           "training_stages": list(training.stage_ranges),
                           "official_test_training_used": False,
                           "test_usage": ("frozen_model_official_inference" if window == "half_test"
                                          else "frozen_model_compatibility_dryrun")},
            "origins": int(len(origins)),
            "origin_range": [str(origins[0]), str(origins[-1])],
            "short": {"scores": short_bundle.scores, "selection": short_bundle.selection,
                      "train_seconds": short_bundle.train_seconds},
            "long": {"scores": long_bundle.scores, "selection": long_bundle.selection,
                     "train_seconds": long_bundle.train_seconds},
            "prediction": prediction_report,
            "optimization": optimization_report,
            "optimization_summary": optimization.get("summary", {}),
            "resource_metrics": resource_bundle.metrics,
            "resource_selection": resource_bundle.selection,
            "resource_quantiles": resource_bundle.quantile_metrics,
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
    ctx = output.begin_run(C, mode="backtest", experimental=quick or experimental)
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
        positions = prepared.index.get_indexer(origins)
        short_pred = main_s.predict_short(short_bundle, masked, origins)
        long_pred = main_l.predict_long(long_bundle, masked, origins)
        if C.LONG_HEAD_FROM_SHORT:
            # 重叠带用短期结果覆盖，见 config 注释。
            k = len(C.HORIZONS_S)
            long_pred[:, :k, :] = short_pred
            # 确认分数原口径在 train_long 内算的是覆盖前结果，这里补一个真值对照。
            truth = main_s.truth_cube(masked, positions, C.HORIZONS_L)
            rescored = main_s.score_cube(long_pred, truth, C.HORIZONS_L)
            LOGGER.info("长期覆盖前 8 步后重评: %.3f (原 %.3f)",
                        rescored["accuracy"],
                        long_bundle.scores["confirm"]["accuracy"])
            long_bundle.scores["confirm"] = rescored
        prediction_report = output.write_predictions(ctx, origins, short_pred, long_pred)
        LOGGER.info("预测输出: %d 个起报点, 短期/长期校验通过", len(origins))
        stages.update(1)
        stages.set_description("回测 · 资源训练与预测")

        resource_bundle = main_opt.train_resources(prepared, folds, quick=quick)
        LOGGER.info("资源模型 WAPE: %s, heat_rate_ok=%s",
                    {g: round(v["wape"], 3) for g, v in resource_bundle.metrics.items()},
                    resource_bundle.heat_rate.get("ok"))
        resource_forecast = main_opt.forecast_resources(resource_bundle, masked, origins)
        # P2：point 与 robust_low 同批对比；默认仍输出 point 调度表。
        quantile_low = None
        if resource_bundle.quantile_models:
            quantile_low = main_opt.forecast_resources_quantile(
                resource_bundle, masked, origins, "p10")
        stages.update(1)
        stages.set_description("回测 · 调度优化")
        optimization = main_opt.optimize(resource_forecast, masked, origins,
                                         long_pred, resource_bundle, quick=quick,
                                         availability_mode="point")
        robust_opt = main_opt.optimize(resource_forecast, masked, origins,
                                       long_pred, resource_bundle, quick=quick,
                                       availability_mode="robust_low",
                                       quantile_low=quantile_low)
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
            "optimization_robust_low": robust_opt.get("summary", {}),
            "resource_quantiles": resource_bundle.quantile_metrics,
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


def run_resource_nn_study(args) -> int:
    """资源时序候选与现有 MLP 同折比较；未过 WAPE 门槛不进入调度晋级。"""
    import main_nn
    from data_cleaning import resource_series

    names = [s.strip() for s in args.folds.split(",")]
    specs = [dict(f) for f in C.FOLDS if f["name"] in names and f["role"] == "tuning"]
    if {f["name"] for f in specs} != set(names):
        raise ValueError("资源专项只允许 A/B/C 开发折")
    cfg = json.loads(args.nn_config)
    ctx = output.begin_run(C, mode="nn_resource", experimental=True)
    _setup_logger(ctx.log_path)
    started = time.time()
    summary = {"status": "RUNNING", "submittable": False, "folds": names,
               "validation_scope": "仅开发折，D未评估", "results": [], "errors": []}
    try:
        output.write_source_snapshot(ctx, C)
        prepared = prepare_data(C).prefix_copy(max(pd.Timestamp(f["start"]) +
            pd.Timedelta(days=f["days"] + 1) for f in specs))
        base, resource = main_opt._resource_base(prepared), resource_series(prepared)
        prices = main_opt.load_prices()
        for spec in C.progress(specs, desc="资源 NN 专项验证", unit="折"):
            _, evaluation, ref = main_s._fold_positions(prepared, spec, 8)
            heat = main_opt.estimate_heat_rate(prepared, ref)
            truth = resource.to_numpy()[evaluation[:, None] + np.arange(1, 9)]
            item = {"fold": spec["name"], "models": {}}
            for kind in C.progress(("mlp", "tcn"), desc="资源 NN 网络比较", unit="候选"):
                main_nn.torch.cuda.reset_peak_memory_stats() if main_nn.DEVICE.type == "cuda" else None
                tick = time.perf_counter()
                pack = main_nn.train_resource_nn(base, resource, ref, heat, prices, args.quick,
                    use_history=False, network_kind=kind, max_epochs=int(cfg.get("epochs", 60)),
                    batch=int(cfg.get("batch_size", 512)))
                seconds = time.perf_counter() - tick
                directory = ctx.model_dir / spec["name"] / kind
                path = main_nn.save_resource_nn(pack, directory / "resource_nn.pt")
                prediction = main_nn.predict_resource_nn(pack, base, resource, evaluation)
                reloaded = main_nn.predict_resource_nn(main_nn.load_resource_nn(path), base, resource, evaluation)
                if not np.allclose(prediction, reloaded, rtol=1e-6, atol=1e-5):
                    raise ValueError("资源网络保存重载不一致")
                item["models"][kind] = {"metrics": main_opt.score_resources(prediction, truth),
                    "seconds": seconds, "epochs": pack["epochs"], "peak_memory_mb": main_nn.peak_memory_mb()}
                np.savez_compressed(directory / "development_predictions.npz", prediction=prediction,
                                    truth=truth, positions=evaluation)
                LOGGER.info("资源 %s/%s: WAPE=%s，训练=%.1fs，显存=%.0fMiB", spec["name"], kind,
                    {g: round(v["wape"], 4) for g, v in item["models"][kind]["metrics"].items()},
                    seconds, main_nn.peak_memory_mb())
            item["passed_wape_screen"] = all(item["models"]["tcn"]["metrics"][g]["wape"] <=
                item["models"]["mlp"]["metrics"][g]["wape"] for g in C.RESOURCE_NAMES)
            summary["results"].append(item)
            output.write_report(ctx, summary)
        summary["passed_wape_screen"] = all(r["passed_wape_screen"] for r in summary["results"])
        summary["dispatch_promotion"] = "pending_full_ABC_control_gate" if summary["passed_wape_screen"] else "not_eligible"
        summary["status"] = "SUCCESS_NN_STUDY"
        return 0
    except Exception as exc:
        summary["status"] = "FAILED"
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        LOGGER.exception("资源专项失败")
        return 1
    finally:
        summary["duration_seconds"] = time.time() - started
        output.write_report(ctx, summary)
        LOGGER.info("资源专项结果: %s", ctx.root)


def run_nn_reblend(directory: Path) -> int:
    """只重用已保存的开发预测选择统一融合系数，不重新训练也不接触新标签。"""
    import shutil
    started = time.time()
    original = json.loads((directory / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
    if original.get("status") != "SUCCESS_NN_STUDY" or set(original["folds"]) != {"A", "B", "C"}:
        raise ValueError("重选融合权重要求完整A/B/C成功专项结果")
    span = original["span"]
    horizons, groups = (C.HORIZONS_S, C.SHORT_GROUPS) if span == "short" else (C.HORIZONS_L, C.LONG_GROUPS)
    ctx = output.begin_run(C, mode=f"nn_fusion_{span}", experimental=True)
    _setup_logger(ctx.log_path)
    output.write_source_snapshot(ctx, C)
    neural, references, profiles, truths = [], [], [], []
    for fold in original["folds"]:
        folder = directory / C.MODEL_DIR / span / fold
        with np.load(folder / "development_predictions.npz") as data:
            neural.append(data["prediction"].copy())
            references.append(data["reference"].copy())
            profiles.append(data["profile"].copy())
            truths.append(data["truth"].copy())
        shutil.copytree(folder, ctx.model_dir / span / fold)
    weights = main_s.select_nn_tree_weights(neural, references, truths, horizons, groups)
    candidates = [
        {"kind": "nn", "preds": neural, "truths": truths, "mean": 0.},
        {"kind": "reference", "preds": references, "truths": truths, "mean": 0.},
        {"kind": "ensemble", "preds": [main_s.blend_cube(a, b, weights, horizons, groups)
            for a, b in zip(neural, references)], "truths": truths, "mean": 0., "nn_weights": weights}]
    main_s.evaluate_blended_candidates(candidates, profiles, horizons, groups)
    reference = candidates[1]
    selection = []
    for candidate in candidates:
        item = {k: candidate[k] for k in ("kind", "mean", "worst", "blend", "scores")}
        item["nn_weights"] = candidate.get("nn_weights", {})
        item["passed_promotion"] = (candidate["kind"] != "reference" and
            item["mean"] >= reference["mean"] + C.SELECTION["min_mean_gain_pp"] and
            all(a["targets"][t]["accuracy"] >= b["targets"][t]["accuracy"] - C.SELECTION["max_worst_fold_drop_pp"]
                for a, b in zip(item["scores"], reference["scores"]) for t in C.TARGETS))
        selection.append(item)
    original.update(selection=selection, training_source=str(directory.resolve()),
                    fusion_grid=[0., .25, .5, .75, 1.], submittable=False,
                    n_seeds=original.get("n_seeds", len(original["results"][0]["training"]["early_stop"]["epochs"])),
                    training_duration_seconds=original.get("duration_seconds"),
                    duration_seconds=time.time() - started, run_id=ctx.run_id)
    output.write_report(ctx, original)
    LOGGER.info("冻结融合: %s，结果=%s", weights, ctx.root)
    return 0


def run_nn_study(args) -> int:
    """独立神经网络开发入口：只访问指定开发折，不触发 D 或整条调度回放。"""
    import main_nn

    if args.nn_reblend:
        return run_nn_reblend(args.nn_reblend)

    if args.nn_span == "resource":
        return run_resource_nn_study(args)

    names = [s.strip() for s in args.folds.split(",")]
    specs = [dict(f) for f in C.FOLDS if f["name"] in names and f["role"] == "tuning"]
    if not names or {f["name"] for f in specs} != set(names):
        raise ValueError("NN 专项只允许 A/B/C 开发折，禁止把 D 用于调参")
    cfg = json.loads(args.nn_config)
    if not isinstance(cfg, dict):
        raise ValueError("--nn-config 必须是 JSON 对象")
    ctx = output.begin_run(C, mode=f"nn_{args.nn_span}_{args.nn_kind}", experimental=True)
    _setup_logger(ctx.log_path)
    started = time.time()
    summary = {"status": "RUNNING", "submittable": False, "folds": names,
               "kind": args.nn_kind, "span": args.nn_span, "config_overrides": cfg,
               "n_seeds": args.nn_seeds,
               "validation_scope": "历史开发折，D未评估", "results": [], "errors": []}
    try:
        output.write_source_snapshot(ctx, C)
        if args.nn_benchmark:
            summary["benchmark"] = main_nn.benchmark_engine(ctx.model_dir, args.nn_kind, cfg)
            summary["status"] = "SUCCESS_NN_BENCHMARK"
            return 0
        prepared = prepare_data(C)
        stop = max(pd.Timestamp(f["start"]) + pd.Timedelta(days=f["days"] + 1) for f in specs)
        prepared = prepared.prefix_copy(stop)
        horizons = C.HORIZONS_S if args.nn_span == "short" else C.HORIZONS_L
        preds, profiles, truths, references = [], [], [], []
        for spec in C.progress(specs, desc="负荷 NN 专项验证", unit="折"):
            LOGGER.info("NN 专项 %s/%s/%s，目录=%s", args.nn_span, args.nn_kind, spec["name"], ctx.root)
            positions, evaluation, ref = main_s._fold_positions(prepared, spec, max(horizons))
            masked = prepared.masked_copy(spec["start"],
                pd.Timestamp(spec["start"]) + pd.Timedelta(days=spec["days"]))
            profile = main_s.baseline_cubes(masked, evaluation, horizons)["profile"]
            truth = main_s.truth_cube(prepared, evaluation, horizons)
            bundle = main_nn.train_nn(prepared, [spec], args.nn_kind, horizons,
                                      quick=args.quick, n_seeds=args.nn_seeds, cfg=cfg)
            directory = ctx.model_dir / args.nn_span / spec["name"]
            path = main_nn.save_nn(bundle, directory)
            origins = prepared.index[evaluation]
            tick = time.perf_counter()
            prediction = main_nn.predict_nn(bundle, masked, origins)
            infer_seconds = time.perf_counter() - tick
            reloaded = main_nn.predict_nn(main_nn.load_nn(Path(path)), masked, origins)
            difference = float(np.max(np.abs(prediction - reloaded)))
            if not np.isfinite(prediction).all() or difference > 1e-5:
                raise ValueError("专项模型保存重载预测不一致")
            item = {"fold": spec["name"], "score": main_s.score_cube(prediction, truth, horizons),
                    "profile_score": main_s.score_cube(profile, truth, horizons),
                    "training": bundle.scores, "infer_seconds": infer_seconds,
                    "reload_max_abs_diff": difference, "feature_count": len(bundle.feature_names)}
            if args.nn_reference:
                kind = "direct" if args.nn_span == "short" else "global"
                base = main_s.build_base_features(masked, include_targets=True)
                stride = C.TRAIN_STRIDE_SHORT if args.nn_span == "short" else C.TRAIN_STRIDE_LONG
                models, feature_names = main_s._train_models(masked, base, positions[::stride],
                    horizons, kind, "gas", C.LGBM_TREES_SHORT if args.nn_span == "short" else C.LGBM_TREES_LONG, ref)
                reference = main_s.enforce_physics(main_s._predict_models(masked, base, evaluation,
                    horizons, kind, "gas", models, feature_names), fallback=profile)
                for key, model in models.items():
                    # Windows 原生 LightGBM 写中文路径不稳定，交由 Python 保存同一模型文本。
                    (directory / f"reference_{output._model_tag(key)}.txt").write_text(
                        model.model_to_string(), encoding="utf-8")
                references.append(reference)
                item["reference_score"] = main_s.score_cube(reference, truth, horizons)
            arrays = {"prediction": prediction, "profile": profile, "truth": truth, "positions": evaluation}
            if args.nn_reference:
                arrays["reference"] = references[-1]
            np.savez_compressed(directory / "development_predictions.npz", **arrays)
            preds.append(prediction)
            profiles.append(profile)
            truths.append(truth)
            summary["results"].append(item)
            output.write_report(ctx, summary)
            LOGGER.info("折 %s: NN=%.3f，画像=%.3f，训练=%.1fs，显存=%.0fMiB", spec["name"],
                        item["score"]["accuracy"], item["profile_score"]["accuracy"],
                        bundle.train_seconds, bundle.scores["peak_memory_mb"])
        groups = main_s._groups_for(horizons)
        candidates = [{"kind": "nn", "preds": preds, "truths": truths,
                       "mean": float(np.mean([r["score"]["accuracy"] for r in summary["results"]]))}]
        if references:
            candidates.append({"kind": "reference", "preds": references, "truths": truths,
                               "mean": float(np.mean([r["reference_score"]["accuracy"] for r in summary["results"]]))})
            candidates.append({"kind": "ensemble", "preds": [(a + b) / 2 for a, b in zip(preds, references)],
                               "truths": truths, "mean": 0.})
        main_s.evaluate_blended_candidates(candidates, profiles, horizons, groups)
        summary["selection"] = [{k: c[k] for k in ("kind", "mean", "worst", "blend", "scores")} for c in candidates]
        if references:
            baseline = candidates[1]
            for item in summary["selection"]:
                item["passed_promotion"] = (item["kind"] != "reference" and
                    item["mean"] >= baseline["mean"] + C.SELECTION["min_mean_gain_pp"] and
                    all(a["targets"][t]["accuracy"] >= b["targets"][t]["accuracy"] -
                        C.SELECTION["max_worst_fold_drop_pp"]
                        for a, b in zip(item["scores"], baseline["scores"]) for t in C.TARGETS))
        summary["status"] = "SUCCESS_NN_STUDY"
        return 0
    except Exception as exc:
        LOGGER.exception("神经网络专项失败")
        summary["status"] = "FAILED"
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        summary["duration_seconds"] = time.time() - started
        output.write_report(ctx, summary)
        LOGGER.info("NN 专项结果: %s", ctx.root)


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
    parser.add_argument("--fresh", action="store_true", help="保留参数，当前无训练缓存")
    parser.add_argument("--nn-only", action="store_true", help="只运行神经网络开发实验，不访问D或调度器")
    parser.add_argument("--folds", default="A,B", help="NN专项开发折，允许A/B/C")
    parser.add_argument("--nn-span", choices=("short", "long", "resource"), default="long")
    parser.add_argument("--nn-kind", choices=("dlinear", "tsmixer", "tcn_causal"), default="dlinear")
    parser.add_argument("--nn-seeds", type=int, default=1)
    parser.add_argument("--nn-config", default="{}", help="NN实验参数覆盖，JSON对象")
    parser.add_argument("--nn-reference", action="store_true", help="同折训练树模型用于受控比较")
    parser.add_argument("--nn-benchmark", action="store_true", help="固定更新步的合成GPU吞吐基准，不评价精度")
    parser.add_argument("--nn-reblend", type=Path, help="重用ABC开发预测选择NN与树的统一融合权重")
    parser.add_argument("--reuse-run", type=Path, help="infer复用本项目已归档的final模型")
    parser.add_argument("--nn-study-run", type=Path, action="append", help="infer读取已通过ABC门槛的冻结NN专项")
    args = parser.parse_args(argv)
    C.PROGRESS_ENABLED = args.progress
    if args.reuse_run and args.command != "infer":
        parser.error("--reuse-run 只用于 infer")
    if args.nn_only:
        if args.command != "run" or not args.experiment:
            parser.error("--nn-only 必须搭配 run --experiment")
        return run_nn_study(args)
    if args.command == "doctor":
        return run_doctor()
    if args.command == "infer":
        if args.nn_study_run and not args.reuse_run:
            parser.error("--nn-study-run 必须同时指定 --reuse-run")
        return run_infer(quick=args.quick, experimental=args.experiment,
                         reuse_run=args.reuse_run, nn_studies=args.nn_study_run)
    if args.command == "self-check":
        return run_self_check()
    if args.command == "verify":
        return run_verify(args.run_dir)
    return run_full(quick=args.quick, experimental=args.experiment)


if __name__ == "__main__":
    sys.exit(main())
