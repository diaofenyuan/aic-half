"""半决赛唯一日常入口：python main.py

默认执行当时能合法完成的完整链路：扫描→清洗→21 天时序折→短期/长期训练与
确认→资源预测→MILP→三份 CSV→模型/源码归档→模拟 ZIP。缺少正式评测输入或
优化模板时明确标记不可提交，不伪造正式成绩。
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


def _setup_logger(log_path: Path | None = None) -> None:
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
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
    for relative, info in scan.get("files", {}).items():
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
        problems.append("缺少模拟评分包")
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
    """选择官方推理起报点：优先 half_test，否则用 semi_test 做预开发干跑。"""
    index = prepared.index
    half = [s for s in C.STAGES if s["name"] == "half_test"]
    if half and (C.DATA_ROOT / half[0]["dir"]).is_dir():
        start = pd.Timestamp(C.OFFICIAL_EVAL_START)
        end = pd.Timestamp(C.OFFICIAL_EVAL_END)
        origins = pd.date_range(start, end, freq=f"{C.STEP_MINUTES}min")
        origins = origins[origins.isin(index)]
        if len(origins):
            return origins, "half_test"
    stage = next(s for s in C.STAGES if s["name"] == "semi_test")
    start = pd.Timestamp(stage["period"][0] + " 00:00:00")
    end = pd.Timestamp(stage["period"][1] + " 23:45:00")
    origins = pd.date_range(start, end, freq=f"{C.STEP_MINUTES}min")
    origins = origins[origins.isin(index)]
    if not len(origins):
        raise ValueError("没有可用的推理起报点")
    return origins, "semi_test_dryrun"


def run_infer(quick: bool = False, experimental: bool = False) -> int:
    """半决赛预开发：在官方特征窗口上滚动预测并生成提交格式文件。"""
    started = time.time()
    ctx = output.begin_run(C, mode="infer", experimental=quick or experimental)
    _setup_logger(ctx.log_path)
    LOGGER.info("推理运行目录: %s", ctx.root)
    summary = {"run_id": ctx.run_id, "status": "FAILED", "mode": "INFER",
               "experimental": bool(quick or experimental), "output_root": str(ctx.root),
               "quick": bool(quick), "validation_scope": "A/B开发验证" if quick else "历史开发验证，D已被多次查看", "errors": []}
    try:
        scan = scan_sources()
        prepared = prepare_data(C)
        contract = data_contract_check(prepared)
        if not contract["passed"]:
            raise ValueError("数据契约检查失败: " + "; ".join(contract["problems"]))
        if not _sources_unchanged(scan):
            raise ValueError("源数据在读取后被修改")

        origins, window = _inference_window(prepared)
        training = prepared.prefix_copy(origins[0] - pd.Timedelta(minutes=C.STEP_MINUTES))
        folds = build_folds(training, quick=quick)
        short_bundle = main_s.train_short(training, folds, quick=quick)
        long_bundle = main_l.train_long(training, folds, quick=quick)
        resource_bundle = main_opt.train_resources(training, folds, quick=quick)
        LOGGER.info("训练完成: short=%s/%s long=%s/%s heat_rate_ok=%s",
                    short_bundle.kind, short_bundle.features,
                    long_bundle.kind, long_bundle.features,
                    resource_bundle.heat_rate.get("ok"))

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
        optimization = main_opt.optimize(resource_forecast, masked, origins,
                                         long_pred, resource_bundle, quick=quick,
                                         use_final=True)
        optimization_report = output.write_optimization(ctx, optimization.get("schedule"))

        model_report = output.save_models(ctx, short_bundle, long_bundle,
                                          resource_bundle, optimization.get("summary", {}),
                                          use_final=True)
        calibration_report = output.write_calibration(
            ctx, short_bundle, long_bundle, resource_bundle, use_final=True)
        source_report = output.write_source_snapshot(ctx, C)
        official = False  # 当前用户仅授权预开发，不因目录或模板开关自动进入正式交付。
        zip_report = output.build_zip(ctx, submittable=bool(official), config=C)

        summary.update({
            "status": "SUCCESS_INFER" if official else "SUCCESS_DRYRUN",
            "window": window,
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
            "submittable": bool(official),
            "reason": None if official else
                      "缺少正式半决赛评测输入或官方 opt_result 模板，仅作预开发干跑",
            "duration_seconds": round(time.time() - started, 2),
        })
        report_path = output.write_report(ctx, summary)
        LOGGER.info("报告: %s", report_path)
        print("\n推理摘要")
        print(f"  窗口: {window} | 起报点 {len(origins)}")
        print(f"  优化状态: {optimization.get('status')}")
        print(f"  输出目录: {ctx.root}")
        print(f"  评分包: {zip_report['name']}")
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


def run_full(quick: bool = False, experimental: bool = False) -> int:
    started = time.time()
    ctx = output.begin_run(C, mode="backtest", experimental=quick or experimental)
    _setup_logger(ctx.log_path)
    LOGGER.info("运行目录: %s", ctx.root)
    summary = {"run_id": ctx.run_id, "status": "FAILED", "mode": "BACKTEST",
               "experimental": bool(quick or experimental), "output_root": str(ctx.root),
               "quick": bool(quick), "validation_scope": "A/B开发验证" if quick else "历史开发验证，D已被多次查看", "errors": []}
    try:
        scan = scan_sources()
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
        short_bundle = main_s.train_short(prepared, folds, quick=quick)
        LOGGER.info("短期选模=%s/%s, 调参=%.3f, 确认=%.3f",
                    short_bundle.kind, short_bundle.features,
                    short_bundle.scores["tuning_mean"],
                    short_bundle.scores["confirm"]["accuracy"])
        long_bundle = main_l.train_long(prepared, folds, quick=quick)
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
        optimization = main_opt.optimize(resource_forecast, masked, origins,
                                         long_pred, resource_bundle, quick=quick,
                                         availability_mode="point")
        robust_opt = main_opt.optimize(resource_forecast, masked, origins,
                                       long_pred, resource_bundle, quick=quick,
                                       availability_mode="robust_low",
                                       quantile_low=quantile_low)
        optimization_report = output.write_optimization(ctx, optimization.get("schedule"))
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
                "reason": "缺少正式半决赛评测输入与官方 opt_result 模板",
            },
            "duration_seconds": round(time.time() - started, 2),
        })
        report_path = output.write_report(ctx, summary)
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
        print(f"  评分包: {zip_report['name']}（不可提交）")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AIC 半决赛煤气发电预测与优化")
    parser.add_argument("command", nargs="?", default="run",
                        choices=("run", "infer", "doctor", "self-check", "verify"))
    parser.add_argument("--run-dir", type=Path, help="verify 指定选定运行或实验结果目录")
    parser.add_argument("--quick", action="store_true", help="单折、低迭代的开发模式")
    parser.add_argument("--experiment", action="store_true", help="全量开发实验写入 tmp/tmp_outputs")
    parser.add_argument("--fresh", action="store_true", help="保留参数，当前无训练缓存")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return run_doctor()
    if args.command == "infer":
        return run_infer(quick=args.quick, experimental=args.experiment)
    if args.command == "self-check":
        return run_self_check()
    if args.command == "verify":
        return run_verify(args.run_dir)
    return run_full(quick=args.quick, experimental=args.experiment)


if __name__ == "__main__":
    sys.exit(main())
