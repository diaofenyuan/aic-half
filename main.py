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
    report_path = directory / C.REPORT_DIR / C.REPORT_JSON
    if report_path.exists():
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        if saved.get("status") == "SUCCESS_PLAN_STUDY":
            return _verify_plan_study(directory, saved)
        if saved.get("status") == "SUCCESS_GENERATOR1_STUDY":
            return _verify_generator1_study(directory, saved)
        if saved.get("status") == "SUCCESS_GENERATOR1_REFINEMENT":
            return _verify_generator1_refinement(directory, saved)
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


def _train_frozen_availability_models(training, study: Path, quick=False):
    """从完成的开发专项冻结配方，只重训最终模型，不再评估 D。"""
    import main_nn
    study = study.resolve()
    if not study.is_relative_to(C.EXPERIMENT_OUTPUT_ROOT.resolve()):
        raise ValueError("冻结配方必须来自本项目实验目录")
    result = json.loads((study / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
    if (result.get("status") != "SUCCESS_PLAN_STUDY" or result.get("quick")
            or result.get("stages") != ["availability"]
            or not result.get("production_path_checks", {}).get("passed")):
        raise ValueError("最终冻结要求已完成的全量可用性专项")
    if result.get("observed_at_t") and not any(
            item.get("accepted") for item in result.get("retained_comparison", {}).values()):
        raise ValueError("当前时刻特征尚未通过相对已保留策略的完整 A/B/C 门槛")
    training.require_training_data()
    recipes = json.loads((study / C.MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    ref, end = len(training.index) - 1, str(training.index[-1])
    base = main_s.build_base_features(training, include_targets=True)
    bundles = []
    for span, horizons, cls, stride in (("short", C.HORIZONS_S, main_s.ShortBundle, C.TRAIN_STRIDE_SHORT),
                                       ("long", C.HORIZONS_L, main_l.LongBundle, C.TRAIN_STRIDE_LONG)):
        tick = time.time()
        recipe = dict(recipes[span])
        promotion = next((r for r in result.get("R0", []) if r.get("accepted")), None) if span == "long" else None
        if promotion:
            recipe.update(kind="nn_lgb_ens", features="hybrid", lgb_kind="global", lgb_features="gas",
                nn_weights=promotion["frozen_nn_weights"], blend=promotion["frozen_blend"])
        is_neural = recipe["kind"] in ("nn", "nn_lgb_ens")
        kind = recipe.get("lgb_kind") if is_neural else recipe["kind"]
        features = recipe.get("lgb_features") if is_neural else recipe["features"]
        positions = np.arange(C.PROFILE_DAYS * C.STEPS_PER_DAY, ref - max(horizons) + 1,
                              C.QUICK_TRAIN_STRIDE if quick else stride)
        models, names = ({}, [])
        LOGGER.info("冻结配方重训 final/%s，截止=%s", span, end)
        if recipe["kind"] != "nn":
            models, names = main_s._train_models(training, base, positions, horizons,
                kind or ("direct" if span == "short" else "global"), features or "gas",
                C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG, ref)
        neural = None
        if is_neural:
            nn_directory = (study / C.MODEL_DIR / "R0_long" / "A" if promotion else
                            study / C.MODEL_DIR / span / "baseline" / "A" / "nn")
            frozen_files = list(nn_directory.glob("*.pt"))
            if not frozen_files:
                raise ValueError("冻结配方缺少神经网络结构与预处理元信息")
            frozen_nn = main_nn.load_nn(frozen_files[0])
            neural = main_nn.train_nn(training, [{"name": "final", "role": "tuning", "train_end": end}],
                frozen_nn.kind, horizons, quick=quick,
                n_seeds=1 if quick else max(1, len(frozen_nn.seed_states)), cfg=frozen_nn.config)
        fallback = None
        selected = max(((stage, item) for stage, item in result["promotion"].get(span, {}).items()
                        if item["accepted"]), key=lambda pair: pair[1]["tests"]["all_missing"]["mean_gain_pp"], default=None)
        if selected:
            stage, item = selected
            model_stage = item.get("model_stage", stage)
            state_recipe = {"kind": "direct" if span == "short" else "global",
                "features": item.get("features", "gas_no_power" if model_stage == "R2" else "gas"), "blend": item["blend"],
                "fallback_weight": item.get("fallback_weight", 1.),
                "fallback_weights": item.get("fallback_weights"),
                "retained_recipe": item.get("retained_recipe"),
                "retained_mix_weight": item.get("retained_mix_weight"),
                "source_study": str(study), "stage": stage,
                "development_gain_pp": item["tests"]["all_missing"]["mean_gain_pp"]}
            if promotion and (state_recipe["fallback_weight"] < 1. or state_recipe["fallback_weights"]):
                original = recipes[span]
                state_recipe["reference_recipe"] = {"kind": original["kind"],
                    "features": original["features"], "blend": original["blend"]}
            fallback = main_s.train_availability_fallback(training, horizons, state_recipe, quick)
        scores = {**recipe["scores"], "confirm": {"accuracy": None, "reason": "冻结开发配方，仅训练final，未重评D"}}
        if promotion:
            scores["tuning_mean"] = promotion["mean_accuracy"]
        bundles.append(cls(kind=recipe["kind"], features=recipe["features"], horizons=horizons,
            blend=recipe["blend"], models={}, final_models=models, feature_names=names,
            final_nn_bundle=neural, nn_weights=recipe.get("nn_weights", {}), lgb_kind=recipe.get("lgb_kind"),
            lgb_features=recipe.get("lgb_features"), final_availability=fallback,
            train_end=end, final_train_end=end, scores=scores, train_seconds=time.time() - tick,
            selection={"chosen": {"kind": recipe["kind"], "features": recipe["features"]},
                       "frozen_availability_study": str(study), "D_reassessed": False}))
    recipe = recipes["resource"]
    positions = np.arange(C.PROFILE_DAYS * C.STEPS_PER_DAY, ref - max(C.HORIZONS_L) + 1,
                          C.QUICK_TRAIN_STRIDE if quick else C.TRAIN_STRIDE_LONG)
    rounds = C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG
    LOGGER.info("冻结配方重训 final/resource，截止=%s", end)
    models, names = main_opt._fit_resource_models(training, positions, rounds, ref,
        objectives=recipe["selection"]["objectives"])
    head_recipe = recipe.get("head_selection", {})
    if head_recipe.get("gases"):
        heads, _ = main_opt._fit_resource_models(training, main_opt._resource_head_positions(ref, quick),
            rounds, ref, objectives=recipe["selection"]["objectives"], horizons=C.HORIZONS_S,
            gases=tuple(head_recipe["gases"]))
        models.update({g + "__head": model for g, model in heads.items()})
    heat, caps = main_opt.estimate_heat_rate(training, ref), main_opt._gas_caps(training, ref)
    adjustment = recipe["selection"].get("availability_adjustment") or {}
    resource_nn = None
    if adjustment.get("method") == "positive_nn_peak":
        resource_nn = main_nn.train_resource_nn(main_opt._resource_base(training),
            main_opt.resource_series(training), ref, heat, main_opt.load_prices(), quick,
            use_history=C.OPT_CONFIG["resource_nn_history"])
    resource = main_opt.ResourceBundle(models={}, final_models=models, feature_names=names,
        metrics=recipe["metrics"], heat_rate=heat, final_heat_rate=heat, gas_caps=caps, final_gas_caps=caps,
        train_end=end, final_train_end=end, final_resource_nn=resource_nn,
        selection={**recipe["selection"], "head_selection": head_recipe},
        dispatch_selection=recipe["dispatch_selection"], blocker=recipe.get("blocker"))
    return bundles[0], bundles[1], resource


def _refresh_availability_recipe(short_bundle, long_bundle, study, training=None, quick=False):
    """复用主模型；新增特征分支只按已通过开发门槛的配方独立重训。"""
    root = study.resolve()
    if not root.is_relative_to(C.EXPERIMENT_OUTPUT_ROOT.resolve()):
        raise ValueError("联合配方必须来自本项目实验目录")
    result = json.loads((root / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
    if (result.get("status") != "SUCCESS_PLAN_STUDY" or result.get("quick")
            or not result.get("production_path_checks", {}).get("passed")):
        raise ValueError("联合配方尚未通过完整实际入口验证")
    if result.get("observed_at_t") and not any(
            item.get("accepted") for item in result.get("retained_comparison", {}).values()):
        raise ValueError("当前时刻特征尚未通过相对已保留策略的完整 A/B/C 门槛")
    originals = json.loads((root / C.MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    for span, bundle in (("short", short_bundle), ("long", long_bundle)):
        expected = dict(originals[span])
        promoted = next((r for r in result.get("R0", []) if r.get("accepted")), None) if span == "long" else None
        if promoted:
            expected.update(kind="nn_lgb_ens", features="hybrid", lgb_kind="global", lgb_features="gas",
                            blend=promoted["frozen_blend"], nn_weights=promoted["frozen_nn_weights"])
        if any(getattr(bundle, key) != expected.get(key) for key in
               ("kind", "features", "lgb_kind", "lgb_features", "blend", "nn_weights")):
            raise ValueError("复用运行主模型与冻结联合配方不一致，需要重训")
        selected = max(((s, item) for s, item in result["promotion"][span].items() if item["accepted"]),
            key=lambda pair: pair[1]["tests"]["all_missing"]["mean_gain_pp"], default=None)
        if not selected:
            continue
        stage, item = selected
        state = bundle.final_availability
        retained_state = state
        expected_features = item.get("features", "gas_no_power" if item.get("model_stage", stage) == "R2" else "gas")
        if not state or state["features"] != expected_features:
            if not result.get("observed_at_t") or training is None:
                raise ValueError("复用运行缺少对应缺测模型，需要按冻结配方重新训练")
            state = main_s.train_availability_fallback(training, bundle.horizons,
                {"kind": "direct" if span == "short" else "global", "features": expected_features,
                 "blend": item["blend"]}, quick)
            bundle.final_availability = state
            LOGGER.info("已按冻结配方重训 %s/%s 最终缺测分支", span, expected_features)
        if item.get("retained_recipe"):
            recipe = item["retained_recipe"]
            if retained_state is None or any(retained_state.get(k) != recipe.get(k) for k in
                    ("features", "blend", "fallback_weight", "fallback_weights")):
                retained_state = main_s.train_availability_fallback(training, bundle.horizons, recipe, quick)
            state.update(retained_state=retained_state, retained_recipe=recipe,
                         retained_mix_weight=item["retained_mix_weight"])
        state.update(blend=item["blend"], fallback_weight=item.get("fallback_weight", 1.),
            fallback_weights=item.get("fallback_weights"), stage=stage, source_study=str(root),
            development_gain_pp=item["tests"]["all_missing"]["mean_gain_pp"])
        if state["fallback_weight"] < 1. or state["fallback_weights"]:
            state["reference_recipe"] = {k: originals[span][k] for k in ("kind", "features", "blend")}
        bundle.selection["joint_availability_study"] = str(root)


def run_infer(quick: bool = False, experimental: bool = False,
              reuse_run: Path | None = None, nn_studies: list[Path] | None = None,
              availability_study: Path | None = None, generator1_study: Path | None = None) -> int:
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
        if availability_study is not None:
            if nn_studies:
                raise ValueError("可用性冻结推理不同时覆盖其他 NN 专项")
            if reuse_run is None:
                short_bundle, long_bundle, resource_bundle = _train_frozen_availability_models(training, availability_study, quick)
            else:
                short_bundle, long_bundle, resource_bundle = output.load_final_models(reuse_run)
                if any(pd.Timestamp(b.final_train_end) != training.index[-1]
                       for b in (short_bundle, long_bundle, resource_bundle)):
                    raise ValueError("复用模型没有覆盖当前合法训练截止点，必须重训")
                _refresh_availability_recipe(short_bundle, long_bundle, availability_study, training, quick)
                summary["reused_models_from"] = str(reuse_run.resolve())
            summary["availability_study"] = str(availability_study.resolve())
            summary["validation_scope"] = ("冻结A/B/C联合开发配方；D未重评；复用主模型，仅新增特征分支按需重训"
                                           if reuse_run else "冻结A/B/C开发配方；仅重训最终模型；D未重评")
            stages.update(3)
        elif reuse_run is None:
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
        if generator1_study is not None or C.GENERATOR1_SPECIALIST:
            summary["generator1"] = _train_generator1_final(training, short_bundle, long_bundle,
                generator1_study, quick=quick)
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
            "prediction_input_support": {
                "power_gas_missing_origins": int((~np.isfinite(masked.frame[["d_use_blast", "d_use_coke", "d_use_converter"]]
                    .to_numpy(dtype=float)[masked.index.get_indexer(origins)]).all(axis=1)).sum()),
                "short_fallback_active": bool(short_bundle.final_availability),
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
        if availability_study is not None or summary.get("generator1"):
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


def _verify_plan_study(directory: Path, summary: dict) -> int:
    problems, checked, diagnostics = [], 0, []
    prepared = prepare_data(C)
    for path in sorted((directory / C.MODEL_DIR).rglob("predictions.npz")):
        with np.load(path) as arrays:
            pred, truth, positions = arrays["prediction"], arrays["truth"], arrays["positions"]
            if pred.shape != truth.shape or not np.isfinite(pred).all() or not np.isfinite(truth).any():
                problems.append(f"预测形状或有限值错误: {path.relative_to(directory)}")
                continue
            if len(positions) != len(pred) or (np.diff(positions) != 1).any():
                problems.append(f"起报点不连续: {path.relative_to(directory)}")
            metadata_path = path.parent / "meta.json"
            if metadata_path.exists():
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (pd.Timestamp(meta["train_end"]) >= prepared.index[positions[0]] or
                    pd.Timestamp(meta.get("max_training_target", meta["train_end"])) > pd.Timestamp(meta["train_end"])):
                    problems.append(f"训练截止跨界: {path.relative_to(directory)}")
            else:
                problems.append(f"缺少训练截止元信息: {path.relative_to(directory)}")
                continue
            if pred.shape[2] == 2:
                physical = (pred[:, :, 0] >= -1e-6).all() and (pred[:, :, 0] <= 200. + 1e-6).all()
                physical &= (pred[:, :, 1] >= pred[:, :, 0] - 1e-6).all() and (pred[:, :, 1] - pred[:, :, 0] <= 240. + 1e-6).all()
                if not physical:
                    problems.append(f"组间容量关系错误: {path.relative_to(directory)}")
                h = C.HORIZONS_S if pred.shape[1] == 8 else C.HORIZONS_L
                name = "s_result.csv" if len(h) == 8 else "l_result.csv"
                frame = output._cube_to_frame(pred, prepared.index[positions], h)
                check = output.validate_official_frame(name, frame)
                problems.extend(check["problems"])
                groups = {}
                ref = int(prepared.index.get_indexer([pd.Timestamp(meta["train_end"])])[0])
                fuel_change = prepared.frame.d_fuel_proxy.diff(4).abs().to_numpy(dtype=float)
                change_threshold = float(np.nanquantile(fuel_change[:ref + 1], .75))
                observed_change = fuel_change[positions]
                process_missing = ~np.isfinite(prepared.frame.d_use_total.to_numpy(dtype=float)[positions])
                for ti, target in enumerate(C.TARGETS):
                    threshold = float(prepared.labels[target].iloc[:ref + 1].quantile(.25))
                    valid = np.isfinite(truth[:, :, ti]) & (truth[:, :, ti] > 0)
                    low = valid & (truth[:, :, ti] <= threshold)
                    groups[target] = {"low_threshold_from_training_mw": threshold,
                        "low_cells": int(low.sum()),
                        "low_accuracy": float(100 * (1 - np.mean(np.abs(pred[:, :, ti][low] - truth[:, :, ti][low]) / truth[:, :, ti][low]))) if low.any() else None,
                        "daily_accuracy": [main_s.score_cube(pred[mask], truth[mask], h)["targets"][target]["accuracy"]
                            for day in prepared.index[positions].normalize().unique()
                            if (mask := prepared.index[positions].normalize() == day).any()]}
                    groups[target]["process_change_threshold_from_training"] = change_threshold
                    groups[target]["process_regimes"] = {}
                    for label, rows in (("stable", observed_change <= change_threshold),
                            ("changing", observed_change > change_threshold), ("missing", process_missing)):
                        cells = valid & rows[:, None]
                        zero = np.isfinite(truth[:, :, ti]) & (truth[:, :, ti] == 0) & rows[:, None]
                        groups[target]["process_regimes"][label] = {
                            "positive_cells": int(cells.sum()), "zero_cells": int(zero.sum()),
                            "accuracy": float(100 * (1 - np.mean(np.abs(pred[:, :, ti][cells] - truth[:, :, ti][cells]) /
                                truth[:, :, ti][cells]))) if cells.any() else None,
                            "zero_mae": float(np.abs(pred[:, :, ti][zero]).mean()) if zero.any() else None}
                    groups[target]["all_finite_bias_and_variation"] = main_s.score_cube(pred, truth, h)["targets"][target]
                diagnostics.append({"archive": str(path.relative_to(directory)), "regimes": groups})
            elif pred.shape[2] != 3 or (pred < 0).any():
                problems.append(f"资源输出维度或非负约束错误: {path.relative_to(directory)}")
            checked += 1
    if not checked:
        problems.append("没有可验证的折外预测")
    if set(p.name for p in (directory / C.REPORT_DIR).iterdir()) != {C.REPORT_JSON, C.REPORT_LOG}:
        problems.append("报告目录包含额外产物")
    if any("half_test" in k or "semi_test" in k for k in summary.get("data_sources", {})):
        problems.append("研究数据来源包含测试输入")
    if summary.get("submittable") is not False:
        problems.append("研究产物不能声明可提交")
    if summary.get("stages") == ["availability"]:
        production = _verify_availability_production(directory, summary, prepared)
        summary["production_path_checks"] = production
        if not production["passed"]:
            problems.append("实际预测入口未复现冻结的可用性分支")
    result = {"passed": not problems, "archives": checked, "problems": problems,
              "regime_diagnostics": diagnostics, "reload_checked_during_training": True}
    summary["verification"] = result
    (directory / C.REPORT_DIR / C.REPORT_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "regime_diagnostics"}, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


def _verify_availability_production(directory, summary, prepared):
    import main_nn
    recipes = json.loads((directory / C.MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    results = []
    promoted_long = next((r for r in summary.get("R0", []) if r.get("accepted")), None)
    for span, horizons, cls, predictor in (("short", C.HORIZONS_S, main_s.ShortBundle, main_s.predict_short),
                                          ("long", C.HORIZONS_L, main_l.LongBundle, main_l.predict_long)):
        selected = max(((s, v) for s, v in summary["promotion"][span].items() if v["accepted"]),
            key=lambda pair: pair[1]["tests"]["all_missing"]["mean_gain_pp"], default=None)
        if not selected:
            continue
        stage, item = selected
        for spec in C.FOLDS:
            if spec["role"] != "tuning":
                continue
            root = directory / C.MODEL_DIR / span
            old = root / "baseline" / spec["name"]
            meta = json.loads((old / "meta.json").read_text(encoding="utf-8"))
            models = _study_load_models(old, meta["files"])
            normal = recipes[span]
            nn_files = list((old / "nn").glob("*.pt"))
            normal_archive = old / "predictions.npz"
            if span == "long" and promoted_long:
                normal = {**normal, "kind": "nn_lgb_ens", "features": "hybrid", "lgb_kind": "global",
                    "lgb_features": "gas", "nn_weights": promoted_long["frozen_nn_weights"],
                    "blend": promoted_long["frozen_blend"]}
                nn_files = list((directory / C.MODEL_DIR / "R0_long" / spec["name"]).glob("nn_*.pt"))
                normal_archive = directory / C.MODEL_DIR / "R0_long" / spec["name"] / "predictions.npz"
            candidate = root / item.get("model_stage", stage) / spec["name"]
            state = json.loads((candidate / "meta.json").read_text(encoding="utf-8"))
            state.update(models=_study_load_models(candidate, state["files"]), blend=item["blend"],
                         fallback_weight=item.get("fallback_weight", 1.),
                         fallback_weights=item.get("fallback_weights"), role="final")
            if item.get("retained_recipe"):
                retained_dir = root / "retained_model" / spec["name"]
                retained = json.loads((retained_dir / "meta.json").read_text(encoding="utf-8"))
                retained.update(item["retained_recipe"], models=_study_load_models(retained_dir, retained["files"]))
                state.update(retained_state=retained, retained_mix_weight=item["retained_mix_weight"])
            if span == "long" and promoted_long and (state["fallback_weight"] < 1. or state["fallback_weights"]):
                state["reference_recipe"] = {k: recipes[span][k] for k in ("kind", "features", "blend")}
            bundle = cls(kind=normal["kind"], features=normal["features"], horizons=horizons,
                blend=normal["blend"], models={}, final_models=models, feature_names=meta["feature_names"],
                final_nn_bundle=main_nn.load_nn(nn_files[0]) if nn_files else None,
                nn_weights=normal.get("nn_weights", {}), lgb_kind=normal.get("lgb_kind"),
                lgb_features=normal.get("lgb_features"), final_availability=state)
            masked, _, positions, _ = _study_view(prepared, spec)
            with np.load(normal_archive) as arrays:
                normal_prediction = arrays["prediction"].copy()
            for scenario in summary["contract"]["scenarios"]:
                start = (spec["start"] if scenario == "all_missing" else
                         pd.Timestamp(spec["start"]) + pd.Timedelta(days=7) if scenario == "outage_day7" else None)
                view = masked if start is None else masked.masked_power_gas_copy(start)
                missing = ~np.isfinite(view.frame[["d_use_blast", "d_use_coke", "d_use_converter"]]
                                      .to_numpy(dtype=float)[positions]).all(axis=1)
                expected = normal_prediction.copy()
                with np.load(root / stage / scenario / spec["name"] / "predictions.npz") as arrays:
                    expected[missing] = arrays["prediction"][missing]
                actual = predictor(bundle, view, view.index[positions], use_final=True)
                difference = float(np.max(np.abs(actual - expected)))
                with np.load(root / "R1" / scenario / spec["name"] / "predictions.npz") as arrays:
                    truth, baseline = arrays["truth"].copy(), arrays["prediction"].copy()
                if summary.get("observed_at_t"):
                    with np.load(root / "retained" / scenario / spec["name"] / "predictions.npz") as arrays:
                        baseline = arrays["prediction"].copy()
                after, before = main_s.score_cube(actual, truth, horizons), main_s.score_cube(baseline, truth, horizons)
                deltas = {t: after["targets"][t]["accuracy"] - before["targets"][t]["accuracy"] for t in C.TARGETS}
                results.append({"span": span, "fold": spec["name"], "scenario": scenario,
                    "max_abs_diff": difference, "accuracy": after["accuracy"],
                    "gain_pp": after["accuracy"] - before["accuracy"], "target_deltas_pp": deltas})
    return {"passed": all(r["max_abs_diff"] <= 1e-5 and min(r["target_deltas_pp"].values()) >= -.2 - 1e-8
                          for r in results), "comparisons": results}


def _study_archive_reference(directory: Path, reference: Path) -> None:
    import shutil
    if not directory.resolve().is_relative_to(C.EXPERIMENT_OUTPUT_ROOT.resolve()):
        raise ValueError("冻结参考模型只能归档到实验目录")
    for span in ("short", "long"):
        source = reference / C.MODEL_DIR / span / "baseline"
        target = directory / C.MODEL_DIR / span / "baseline"
        if source.is_dir() and not target.exists():
            shutil.copytree(source, target)


def _study_save_models(directory: Path, models: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, model in models.items():
        tag = output._model_tag(key)
        filename = tag + ".txt"
        (directory / filename).write_text(model.model_to_string(), encoding="utf-8")
        files[tag] = filename
    return files


def _study_load_models(directory: Path, files: dict) -> dict:
    models = {}
    for tag, name in files.items():
        key = tag
        if "__" in tag and tag.rsplit("__", 1)[1].isdigit():
            target, h = tag.rsplit("__", 1)
            key = (target, int(h))
        models[key] = main_opt.load_resource_model((directory / name).read_text(encoding="utf-8"))
    return models


def _study_view(prepared, spec):
    horizons = np.asarray(C.HORIZONS_L)
    positions, evaluation, ref = main_s._fold_positions(prepared, spec, int(horizons.max()))
    if evaluation[-1] + horizons.max() >= len(prepared.index):
        raise ValueError("开发折标签延伸不完整")
    end = prepared.index[evaluation[-1] + horizons.max()]
    masked = prepared.masked_copy(spec["start"], end)
    return masked, positions, evaluation, ref


def _study_load_stage(ctx, summary, prepared, specs, reference, stages, quick):
    import main_nn
    from data_cleaning.pipeline import fit_profile_state, state_slot_profile
    original = json.loads((reference / C.MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    stage_names = ["baseline"] if stages == {"baseline"} else [s for s in ("E1_21", "E1_45", "E2", "E3")
        if s.split("_")[0] in stages]
    for span, horizons, stride, kind in (("short", C.HORIZONS_S, C.TRAIN_STRIDE_SHORT, "direct"),
                                        ("long", C.HORIZONS_L, C.TRAIN_STRIDE_LONG, "global")):
        all_truths = []
        accepted = "baseline"
        for stage in stage_names:
            predictions, baselines = [], []
            half_life = float(stage.split("_")[1]) if stage.startswith("E1_") else 60.
            if stage in ("E2", "E3"):
                half_life = summary.get("selected_decay", {}).get(span, 60.)
            for spec in specs:
                LOGGER.info("方案实验 %s/%s/%s，half_life=%s", stage, span, spec["name"], half_life)
                masked, _, evaluation, ref = _study_view(prepared, spec)
                positions, _, _ = main_s._fold_positions(masked, spec, max(horizons))
                positions = positions[::C.QUICK_TRAIN_STRIDE if quick else stride]
                directory = ctx.model_dir / span / stage / spec["name"]
                base = main_s.build_base_features(masked, include_targets=True)
                profile = main_s.baseline_cubes(masked, evaluation, horizons)["profile"]
                truth = main_s.truth_cube(prepared, evaluation, horizons)
                source = reference / C.MODEL_DIR / span / "baseline" / spec["name"]
                if stage == "baseline":
                    baseline = None
                    neural = None
                    if span == "short":
                        pack = main_nn.train_nn(prepared, [{**spec, "role": "tuning"}], C.NN_CONFIG["kind_short"], horizons,
                            quick=quick, n_seeds=1 if quick else C.NN_CONFIG["n_seeds"])
                        path = main_nn.save_nn(pack, directory / "nn")
                        neural = main_nn.predict_nn(pack, masked, masked.index[evaluation])
                        reload_neural = main_nn.predict_nn(main_nn.load_nn(Path(path)), masked, masked.index[evaluation])
                        if not np.allclose(neural, reload_neural, rtol=0., atol=1e-5):
                            raise ValueError("基线 NN 重载不一致")
                else:
                    with np.load(source / "predictions.npz") as data:
                        baseline, neural = data["prediction"].copy(), data["neural"].copy()
                        if not np.array_equal(evaluation, data["positions"]):
                            raise ValueError("基线起报点不匹配")
                        if not np.allclose(truth, data["truth"], equal_nan=True):
                            raise ValueError("基线标签不匹配")
                    if stage in ("E2", "E3") and accepted != "baseline":
                        with np.load(ctx.model_dir / span / accepted / spec["name"] / "predictions.npz") as data:
                            baseline = data["prediction"].copy()
                recipe = fit_profile_state(masked, ref) if stage == "E2" else None
                previous_decay = C.DECAY_DAYS
                try:
                    C.DECAY_DAYS = half_life
                    if stage == "E3":
                        models, names = main_s.fit_allocation(masked, base, positions, horizons, ref,
                            C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG)
                        raw = main_s.predict_allocation(masked, base, evaluation, horizons,
                            models, names, baseline)
                    else:
                        models, names = main_s._train_models(masked, base, positions, horizons, kind, "gas",
                            C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG, ref, recipe)
                        raw = main_s._predict_models(masked, base, evaluation, horizons, kind,
                            "gas", models, names, recipe)
                finally:
                    C.DECAY_DAYS = previous_decay
                raw = main_s.enforce_physics(raw, fallback=profile)
                if stage == "E2":
                    profile = main_s.enforce_physics(np.stack([state_slot_profile(masked, evaluation,
                        horizons, target, recipe)[0] for target in C.TARGETS], axis=2), fallback=profile)
                def assemble(tree):
                    if stage == "E3":
                        return tree
                    tree = main_s.enforce_physics(tree, fallback=profile)
                    if span == "short":
                        tree = main_s.blend_cube(neural, tree, original[span]["nn_weights"], horizons, C.SHORT_GROUPS)
                    return main_s.blend_cube(tree, profile, original[span]["blend"], horizons, main_s._groups_for(horizons))
                prediction = assemble(raw)
                files = _study_save_models(directory, models)
                loaded = _study_load_models(directory, files)
                if stage == "E3":
                    reloaded = main_s.predict_allocation(masked, base, evaluation, horizons, loaded, names, baseline)
                else:
                    reloaded = main_s._predict_models(masked, base, evaluation, horizons, kind, "gas", loaded, names, recipe)
                difference = float(np.max(np.abs(prediction - assemble(reloaded))))
                if difference > 1e-8:
                    raise ValueError("候选模型重载预测不一致")
                meta = {"files": files, "feature_names": names, "kind": kind, "span": span,
                        "half_life_days": half_life, "profile_state": recipe,
                        "max_training_target": str(masked.index[max(positions) + max(horizons)]),
                        "train_end": spec["train_end"], "reload_max_abs_diff": difference,
                        "fixed_fusion": original[span]["blend"], "fixed_nn_weights": original[span]["nn_weights"]}
                (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                np.savez_compressed(directory / "predictions.npz", prediction=prediction, raw=raw,
                    truth=truth, profile=profile, neural=np.empty(0) if neural is None else neural, positions=evaluation)
                predictions.append(prediction)
                baselines.append(prediction if baseline is None else baseline)
                scores = main_s.score_cube(prediction, truth, horizons)
                summary["results"].append({"stage": stage, "span": span, "fold": spec["name"],
                    "score": scores, "max_training_target": meta["max_training_target"],
                    "reload_max_abs_diff": difference})
                LOGGER.info("完成 %s/%s/%s: %.6f", stage, span, spec["name"], scores["accuracy"])
                output.write_report(ctx, summary)
                if stage == stage_names[0]:
                    all_truths.append(truth)
            result = main_s.load_promotion(predictions, baselines, all_truths, horizons)
            summary.setdefault("promotion", {}).setdefault(span, {})[stage] = result
            if stage.startswith("E1_") and result["accepted"]:
                current = summary.get("promotion", {}).get(span, {}).get(accepted, {}).get("mean_accuracy", -np.inf)
                if result["mean_accuracy"] > current:
                    accepted = stage
                    summary.setdefault("selected_decay", {})[span] = half_life
            output.write_report(ctx, summary)


def _study_resource_bundle(prepared, spec, recipe, quick, two_stage=(), baseline=None):
    import copy
    import main_nn
    if recipe.get("two_stage_gases") and baseline is None:
        plain_recipe = {k: v for k, v in recipe.items() if k != "two_stage_gases"}
        reference_bundle, _, _, _ = _study_resource_bundle(prepared, spec, plain_recipe, quick)
        return _study_resource_bundle(prepared, spec, plain_recipe, quick,
            two_stage=tuple(recipe["two_stage_gases"]), baseline=reference_bundle)
    masked, _, evaluation, ref = _study_view(prepared, spec)
    positions, _, _ = main_opt._resource_positions(masked, spec, quick)
    rounds = C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG
    gases = tuple(two_stage) if baseline is not None else C.RESOURCE_NAMES
    models, names = main_opt._fit_resource_models(masked, positions, rounds, ref,
        objectives=recipe["selection"]["objectives"], gases=gases, two_stage_gases=tuple(two_stage))
    head_gases = tuple(g for g in gases if g in recipe["head_selection"].get("gases", []))
    heads = {}
    if head_gases:
        heads, _ = main_opt._fit_resource_models(masked, main_opt._resource_head_positions(ref, quick),
            rounds, ref, objectives={g: "regression_l1" for g in head_gases}, horizons=C.HORIZONS_S,
            gases=head_gases, two_stage_gases=tuple(two_stage))
    models.update({g + "__head": m for g, m in heads.items()})
    if baseline is not None:
        bundle = copy.copy(baseline)
        bundle.models = {**baseline.models, **models,
                         **{g + "__reference": m for g, m in baseline.models.items()}}
        bundle.selection = {**baseline.selection, "frozen_adjustment_reference": True}
    else:
        selection = {**recipe["selection"], "head_selection": recipe["head_selection"]}
        bundle = main_opt.ResourceBundle(models, {}, names, selection=selection,
            heat_rate=main_opt.estimate_heat_rate(masked, ref), gas_caps=main_opt._gas_caps(masked, ref),
            train_end=spec["train_end"], dispatch_selection=recipe["dispatch_selection"])
        if recipe["selection"].get("availability_adjustment", {}).get("method") == "positive_nn_peak":
            bundle.resource_nn = main_nn.train_resource_nn(main_opt._resource_base(masked),
                main_opt.resource_series(masked), ref, bundle.heat_rate, main_opt.load_prices(),
                quick, use_history=C.OPT_CONFIG["resource_nn_history"])
    forecast = main_opt.forecast_resources(bundle, masked, masked.index[evaluation])
    return bundle, masked, evaluation, forecast


def _study_save_resource(directory, bundle, prepared, evaluation, forecast):
    import main_nn
    files = _study_save_models(directory, bundle.models)
    metadata = {"files": files, "feature_names": bundle.feature_names, "selection": bundle.selection,
        "heat_rate": bundle.heat_rate, "gas_caps": bundle.gas_caps, "train_end": bundle.train_end,
        "dispatch_selection": bundle.dispatch_selection}
    if bundle.resource_nn:
        main_nn.save_resource_nn(bundle.resource_nn, directory / "resource_nn.pt")
    loaded = main_opt.ResourceBundle(_study_load_models(directory, files), {}, bundle.feature_names,
        selection=bundle.selection, heat_rate=bundle.heat_rate, gas_caps=bundle.gas_caps,
        resource_nn=main_nn.load_resource_nn(directory / "resource_nn.pt") if bundle.resource_nn else None)
    reloaded = main_opt.forecast_resources(loaded, prepared, prepared.index[evaluation])
    difference = float(np.max(np.abs(reloaded - forecast)))
    if difference > 1e-5:
        raise ValueError("资源候选保存重载预测不一致")
    metadata["reload_max_abs_diff"] = difference
    (directory / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _study_read_resource(directory, prepared, spec):
    import main_nn
    metadata_path, predictions_path = directory / "meta.json", directory / "predictions.npz"
    if not metadata_path.exists() or not predictions_path.exists():
        return None
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    masked, _, evaluation, _ = _study_view(prepared, spec)
    if pd.Timestamp(meta["train_end"]) != pd.Timestamp(spec["train_end"]):
        raise ValueError("续跑资源模型截止点与当前折不一致")
    bundle = main_opt.ResourceBundle(_study_load_models(directory, meta["files"]), {}, meta["feature_names"],
        selection=meta["selection"], heat_rate=meta["heat_rate"], gas_caps=meta["gas_caps"],
        train_end=meta["train_end"], dispatch_selection=meta["dispatch_selection"],
        resource_nn=main_nn.load_resource_nn(directory / "resource_nn.pt") if (directory / "resource_nn.pt").exists() else None)
    with np.load(predictions_path) as arrays:
        forecast, truth = arrays["prediction"].copy(), arrays["truth"].copy()
        if not np.array_equal(arrays["positions"], evaluation):
            raise ValueError("续跑资源起报点与当前折不一致")
    actual = main_opt.resource_series(prepared).to_numpy()[evaluation[:, None] + np.arange(1, 97)]
    if not np.allclose(truth, actual, equal_nan=True):
        raise ValueError("续跑资源评分标签与当前训练数据不一致")
    loaded = main_opt.forecast_resources(bundle, masked, masked.index[evaluation])
    if not np.allclose(forecast, loaded, rtol=0., atol=1e-5):
        raise ValueError("续跑资源权重与已存折外预测不一致")
    LOGGER.info("复用已完成资源模型 %s", directory.name)
    return bundle, masked, evaluation, forecast


def _study_inner_resources(ctx, prepared, outer, recipe, quick):
    forecasts, truths, positions, records = [], [], [], []
    # 完整内层误差路径的末目标必须留在外层训练前缀。
    for offset in (28, 14):
        start = pd.Timestamp(outer["start"]) - pd.Timedelta(days=offset)
        inner = {"name": f"{outer['name']}_inner_{offset}", "role": "tuning", "start": str(start),
            "train_end": str(start - pd.Timedelta(minutes=15)), "days": 13}
        LOGGER.info("前缀折外资源 %s，冻结截止=%s", inner["name"], inner["train_end"])
        directory = ctx.model_dir / "resource_oof" / inner["name"]
        cached = _study_read_resource(directory, prepared, inner)
        bundle, masked, evaluation, forecast = cached or _study_resource_bundle(prepared, inner, recipe, quick)
        truth = main_opt.resource_series(prepared).to_numpy()[evaluation[:, None] + np.arange(1, 97)]
        max_target = prepared.index[evaluation[-1] + 96]
        if max_target > pd.Timestamp(outer["train_end"]):
            raise ValueError("内层资源折外目标越过外层训练边界")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8")) if cached else _study_save_resource(directory, bundle, masked, evaluation, forecast)
        if not cached:
            np.savez_compressed(directory / "predictions.npz", prediction=forecast, truth=truth, positions=evaluation)
        forecasts.append(forecast)
        truths.append(truth)
        positions.append(evaluation)
        records.append({"train_end": inner["train_end"], "origins": [str(start), str(prepared.index[evaluation[-1]])],
                        "max_target": str(max_target), "reload_max_abs_diff": meta["reload_max_abs_diff"]})
    return np.concatenate(forecasts), np.concatenate(truths), np.concatenate(positions), records


def _study_resource_stage(ctx, summary, prepared, specs, reference, stages, quick):
    original = json.loads((reference / C.MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))["resource"]
    original["dispatch_selection"]["control_options"]["mip_rel_gap"] = .001
    summary["solver_settings"] = {"mip_rel_gap": .001,
        "rolling_time_limit_s": C.OPT_CONFIG["rolling_solver_time_limit_s"], "paired_same_settings": True}
    summary["resource_contract"] = {"mode": "historical_consumption_envelope",
        "zero_state": "zero_historical_power_consumption_not_zero_generation",
        "physical_feasibility_certified": False, "fixed_blast_forecast": True}
    caches, before_metrics, after_metrics = [], [], []
    for spec in specs:
        LOGGER.info("资源保护基线 %s", spec["name"])
        directory = ctx.model_dir / "resource" / "baseline" / spec["name"]
        cached = _study_read_resource(directory, prepared, spec)
        bundle, masked, evaluation, forecast = cached or _study_resource_bundle(prepared, spec, original, quick)
        truth = main_opt.resource_series(prepared).to_numpy()[evaluation[:, None] + np.arange(1, 97)]
        metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8")) if cached else _study_save_resource(directory, bundle, masked, evaluation, forecast)
        if not cached:
            np.savez_compressed(directory / "predictions.npz", prediction=forecast, truth=truth, positions=evaluation)
        row = {"fold": spec["name"], "baseline_metrics": main_opt.score_resources(forecast, truth),
            "train_end": bundle.train_end, "reload_max_abs_diff": metadata["reload_max_abs_diff"]}
        cache = {"spec": spec, "prepared": masked, "evaluation": evaluation, "bundle": bundle,
                 "baseline": forecast, "truth": truth}
        before_metrics.append(row["baseline_metrics"])
        if "E4" in stages:
            target = ctx.model_dir / "resource" / "E4" / spec["name"]
            trial_cached = _study_read_resource(target, prepared, spec)
            trial_bundle, _, _, trial = trial_cached or _study_resource_bundle(prepared, spec, original, quick,
                two_stage=("coke", "converter"), baseline=bundle)
            if not np.array_equal(trial[:, :, 0], forecast[:, :, 0]):
                raise ValueError("E4 不得改变固定高炉气预测")
            row["trial_metrics"] = main_opt.score_resources(trial, truth)
            after_metrics.append(row["trial_metrics"])
            trial_meta = json.loads((target / "meta.json").read_text(encoding="utf-8")) if trial_cached else _study_save_resource(target, trial_bundle, masked, evaluation, trial)
            row["trial_reload_max_abs_diff"] = trial_meta["reload_max_abs_diff"]
            row["two_stage_types"] = {g: type(trial_bundle.models[g]).__name__ for g in ("coke", "converter")}
            if not trial_cached:
                np.savez_compressed(target / "predictions.npz", prediction=trial, truth=truth, positions=evaluation)
            cache["trial"] = trial
            cache["trial_bundle"] = trial_bundle
        stage = "E4_resource" if "E4" in stages else "resource_baseline"
        summary["results"] = [r for r in summary["results"] if (r.get("stage"), r.get("fold")) != (stage, spec["name"])]
        summary["results"].append({"stage": stage, **row})
        LOGGER.info("资源折 %s: baseline=%s，trial=%s", spec["name"],
            {g: round(m["wape"], 5) for g, m in row["baseline_metrics"].items()},
            {g: round(m["wape"], 5) for g, m in row.get("trial_metrics", {}).items()})
        caches.append(cache)
        output.write_report(ctx, summary)
    if "E4" in stages:
        screen = main_opt.resource_promotion_screen(after_metrics, before_metrics)
        summary["E4"] = screen
        eligible = screen["eligible_gases"]
        if eligible:
            first_before, first_after = [], []
            for cache in caches:
                candidate = cache["baseline"].copy()
                for gas in eligible:
                    gi = C.RESOURCE_NAMES.index(gas)
                    candidate[:, :, gi] = cache["trial"][:, :, gi]
                cache["selected_trial"] = candidate
                for name, values, collection in (("baseline", cache["baseline"], first_before),
                                                  ("trial", candidate, first_after)):
                    result = main_opt.optimize_rolling(values, cache["prepared"],
                        cache["prepared"].index[cache["evaluation"]], cache["bundle"], quick=True,
                        replan_steps=original["dispatch_selection"]["replan_steps"],
                        control_options=original["dispatch_selection"]["control_options"])
                    collection.append(result["summary"])
            screen["screen_comparison"] = {"baseline": first_before, "trial": first_after}
            screen["dispatch_screen"] = main_opt.paired_dispatch_gate(first_before, first_after, full=False)
            output.write_report(ctx, summary)
            if screen["dispatch_screen"]["passed_screen"] and not quick:
                full_before, full_after = [], []
                for cache in caches:
                    for name, values, collection in (("baseline", cache["baseline"], full_before),
                                                     ("trial", cache["selected_trial"], full_after)):
                        result = main_opt.optimize_rolling(values, cache["prepared"],
                            cache["prepared"].index[cache["evaluation"]], cache["bundle"],
                            replan_steps=original["dispatch_selection"]["replan_steps"],
                            control_options=original["dispatch_selection"]["control_options"])
                        collection.append(result["summary"])
                        if result.get("schedule") is not None:
                            target = ctx.model_dir / "resource" / name / cache["spec"]["name"] / "dispatch.csv"
                            target.parent.mkdir(parents=True, exist_ok=True)
                            result["schedule"].to_csv(target, index=False)
                        output.write_report(ctx, summary)
                screen["full_comparison"] = {"baseline": full_before, "trial": full_after}
                screen["dispatch_full"] = main_opt.paired_dispatch_gate(full_before, full_after, full=True)
                screen["accepted"] = screen["dispatch_full"]["accepted"]
                if screen["accepted"]:
                    import copy
                    original["two_stage_gases"] = list(eligible)
                    for cache in caches:
                        chosen = copy.copy(cache["bundle"])
                        chosen.models = {**chosen.models, **{k: v for k, v in cache["trial_bundle"].models.items()
                            if k.endswith("__reference") or k.split("__")[0] in eligible}}
                        chosen.selection = cache["trial_bundle"].selection
                        cache["bundle"] = chosen
                        cache["baseline"] = main_opt.forecast_resources(chosen, cache["prepared"], cache["prepared"].index[cache["evaluation"]])
                        if not np.allclose(cache["baseline"], cache["selected_trial"], rtol=0., atol=1e-8):
                            raise ValueError("资源晋级组合与完整回放版本不一致")
        output.write_report(ctx, summary)
    if stages & {"E5", "E6"}:
        for cache in caches:
            forecast, truth, positions, records = _study_inner_resources(ctx, prepared, cache["spec"], original, quick)
            cache.update(inner_forecast=forecast, inner_truth=truth, inner_positions=positions, inner_records=records)
            maximum = max(r["max_target"] for r in records)
            risk = main_opt.fit_inventory_risk(forecast, truth, maximum, cache["spec"]["train_end"])
            summary["E6_calibration"] = [r for r in summary.get("E6_calibration", []) if r["fold"] != cache["spec"]["name"]]
            summary["E6_calibration"].append({"fold": cache["spec"]["name"], "blocks": records, **risk})
            cache["risk"] = risk
            output.write_report(ctx, summary)
        if "E5" in stages:
            _study_future_load(ctx, summary, prepared, caches, reference, quick)
        if "E6" in stages:
            _study_risk_dispatch(ctx, summary, caches, original, quick)


def _study_risk_dispatch(ctx, summary, caches, recipe, quick):
    if not all(c["risk"].get("feasible") for c in caches):
        summary["E6"] = {"accepted": False, "reason": "RISK_BOUNDS_HAVE_NO_FEASIBLE_DOMAIN",
            "action": "keep_verified_point_control; never_relax_15_to_90_percent",
            "infeasible_folds": [c["spec"]["name"] for c in caches if not c["risk"].get("feasible")]}
        output.write_report(ctx, summary)
        return
    comparisons = {}
    for full in (False, True):
        before, after = [], []
        for cache in caches:
            origins = cache["prepared"].index[cache["evaluation"]]
            for name, collection in (("baseline", before), ("risk", after)):
                options = dict(recipe["dispatch_selection"]["control_options"])
                if name == "risk":
                    options["inventory_risk"] = cache["risk"]
                result = main_opt.optimize_rolling(cache["baseline"], cache["prepared"], origins,
                    cache["bundle"], quick=not full, replan_steps=recipe["dispatch_selection"]["replan_steps"],
                    control_options=options)
                collection.append(result["summary"])
                if result.get("schedule") is not None:
                    directory = ctx.model_dir / "inventory_risk" / ("full" if full else "screen") / cache["spec"]["name"]
                    directory.mkdir(parents=True, exist_ok=True)
                    result["schedule"].to_csv(directory / f"{name}.csv", index=False)
        gate = main_opt.paired_dispatch_gate(before, after, full)
        comparisons["full" if full else "screen"] = {"baseline": before, "trial": after, "gate": gate}
        summary["E6"] = {"accepted": gate["accepted"], "comparisons": comparisons}
        output.write_report(ctx, summary)
        if not gate["passed_screen"] or quick:
            break


def _study_future_load(ctx, summary, prepared, caches, reference, quick):
    if not caches or not all(c.get("inner_records") for c in caches):
        summary["E5"] = {"accepted": False, "status": "NOT_STARTED_PREREQUISITE",
            "reason": "missing_frozen_training_prefix_OOF_resource_paths"}
        return
    original = json.loads((reference / C.MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))["long"]
    predictions, baselines, controls, truths = [], [], [], []
    provenance = []
    for cache in caches:
        positions = cache["inner_positions"]
        forecast, truth = cache["inner_forecast"], cache["inner_truth"]
        widths = np.full_like(forecast, np.nan)
        halfway = len(forecast) // 2
        residual = truth[:halfway] - forecast[:halfway]
        first_width = np.nanquantile(residual, .9, axis=0) - np.nanquantile(residual, .1, axis=0)
        widths[halfway:] = first_width
        outer_width = np.nanquantile(truth - forecast, .9, axis=0) - np.nanquantile(truth - forecast, .1, axis=0)
        records = cache["inner_records"]
        if pd.Timestamp(records[0]["max_target"]) >= pd.Timestamp(records[1]["origins"][0]):
            raise ValueError("区间宽度校准标签越过下个内层块起报点")
        extras = main_s.future_resource_features(cache["prepared"], positions, forecast, widths)
        evaluation = cache["evaluation"]
        outer_extras = main_s.future_resource_features(cache["prepared"], evaluation, cache["baseline"],
            np.broadcast_to(outer_width, cache["baseline"].shape))
        stride = C.QUICK_TRAIN_STRIDE if quick else C.TRAIN_STRIDE_LONG
        ref = int(prepared.index.get_indexer([pd.Timestamp(cache["spec"]["train_end"])])[0])
        base = main_s.build_base_features(cache["prepared"], include_targets=True)
        profile = main_s.baseline_cubes(cache["prepared"], evaluation, C.HORIZONS_L)["profile"]
        load_truth = main_s.truth_cube(prepared, evaluation, C.HORIZONS_L)
        for name, features, output_features in (("oof_control", None, None),
                ("E5", {k: v[::stride] for k, v in extras.items()}, outer_extras)):
            LOGGER.info("两阶段负荷 %s/%s，训练起报点=%d", name, cache["spec"]["name"], len(positions[::stride]))
            models, names = main_s._train_models(cache["prepared"], base, positions[::stride], C.HORIZONS_L,
                "global", "gas", C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG, ref,
                future_features=features)
            raw = main_s._predict_models(cache["prepared"], base, evaluation, C.HORIZONS_L,
                "global", "gas", models, names, future_features=output_features)
            prediction = main_s.blend_cube(main_s.enforce_physics(raw, profile), profile,
                original["blend"], C.HORIZONS_L, C.LONG_GROUPS)
            directory = ctx.model_dir / "long" / name / cache["spec"]["name"]
            files = _study_save_models(directory, models)
            reloaded = main_s._predict_models(cache["prepared"], base, evaluation, C.HORIZONS_L,
                "global", "gas", _study_load_models(directory, files), names, future_features=output_features)
            if not np.allclose(raw, reloaded, rtol=0., atol=1e-8):
                raise ValueError("两阶段负荷模型保存重载不一致")
            np.savez_compressed(directory / "predictions.npz", prediction=prediction, truth=load_truth, positions=evaluation)
            (directory / "meta.json").write_text(json.dumps({"files": files, "feature_names": names,
                "train_end": cache["spec"]["train_end"], "inner_blocks": records,
                "interval_width_source": "strictly_earlier_OOF_block; first_block_missing",
                "outer_interval_width": outer_width.tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")
            (predictions if name == "E5" else controls).append(prediction)
            LOGGER.info("完成两阶段负荷 %s/%s: %.6f", name, cache["spec"]["name"],
                main_s.score_cube(prediction, load_truth, C.HORIZONS_L)["accuracy"])
        with np.load(reference / C.MODEL_DIR / "long" / "baseline" / cache["spec"]["name"] / "predictions.npz") as data:
            baselines.append(data["prediction"].copy())
        truths.append(load_truth)
        provenance.append({"fold": cache["spec"]["name"], "blocks": records, "train_positions": len(positions[::stride])})
    gate = main_s.load_promotion(predictions, baselines, truths, C.HORIZONS_L)
    control_gate = main_s.load_promotion(predictions, controls, truths, C.HORIZONS_L)
    gate["accepted"] = bool(gate["accepted"] and control_gate["accepted"])
    summary["E5"] = {**gate, "matched_sample_control": control_gate, "provenance": provenance,
        "frozen_resource_version": "E4" if summary.get("E4", {}).get("accepted") else "protected_baseline"}


def _study_e0(ctx, summary, prepared, specs, reference):
    for spec in specs:
        ref = int(prepared.index.get_indexer([pd.Timestamp(spec["train_end"])])[0])
        summary.setdefault("E0_prefix", []).append({"fold": spec["name"],
            "ledger": main_opt.audit_resource_ledger(prepared, ref),
            "physical_comparison": main_opt.physical_prefix_comparison(prepared, ref)})
    original = json.loads((reference / C.MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    run = json.loads((reference / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
    optimization = run.get("optimization") or run.get("E0", {}).get("physical_parameters")
    if optimization:
        # D 仅重算既有回放的计值，不训练、不挑选新策略。
        history = prepare_data(C, cutoff="2025-09-27 00:00:00")
        times = pd.DatetimeIndex(pd.read_csv(reference / "opt_result.csv").datetime) if (reference / "opt_result.csv").exists() else None
        if times is None and run.get("reference"):
            source = Path(run["reference"])
            times = pd.DatetimeIndex(pd.read_csv(source / "opt_result.csv").datetime)
        if times is not None:
            gas = main_opt.resource_series(history).reindex(times).to_numpy(dtype=float)
            beta = np.array([original["resource"]["heat_rate"]["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
            converted = main_opt._dispatch_metrics(times, gas @ beta, main_opt.load_prices())
            actual = main_opt._dispatch_metrics(times, history.labels.generator_all.reindex(times).to_numpy(), main_opt.load_prices())
            executed = optimization["proxy_executed_revenue_cny"]
            summary["E0_value_comparison"] = {"scope": "existing_D_replay_diagnosis_only",
                "historical_generation_value_cny": actual["generation_revenue_cny"],
                "same_conversion_gas_value_cny": converted["generation_revenue_cny"],
                "proxy_executed_value_cny": executed,
                "gain_vs_actual_ratio": executed / actual["generation_revenue_cny"] - 1.,
                "gain_vs_same_conversion_ratio": executed / converted["generation_revenue_cny"] - 1.,
                "physical_feasibility_certified": False,
                "dictionary_conflict": "dictionary row21 mentions 200000/300000m3; task p5 uses 200000m3; holder2 mapping unresolved"}
    output.write_report(ctx, summary)


def _availability_fold_predictions(prepared, spec, horizons, directory):
    """重用冻结折模型，重新构造故障输入，避免仅修改已经缓存的预测。"""
    import main_nn
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    models = _study_load_models(directory, meta["files"])
    masked, _, positions, _ = _study_view(prepared, spec)
    neural_files = list((directory / "nn").glob("*.pt"))
    neural = main_nn.load_nn(neural_files[0]) if neural_files else None
    result = {}
    for scenario, start in (("normal", None), ("all_missing", spec["start"]),
                            ("outage_day7", pd.Timestamp(spec["start"]) + pd.Timedelta(days=7))):
        view = masked if start is None else masked.masked_power_gas_copy(start)
        base = main_s.build_base_features(view, include_targets=True)
        profile = main_s.baseline_cubes(view, positions, horizons)["profile"]
        pred = main_s._predict_models(view, base, positions, horizons,
            meta["kind"], "gas", models, meta["feature_names"])
        pred = main_s.enforce_physics(pred, fallback=profile)
        if neural is not None:
            pred = main_s.blend_cube(main_nn.predict_nn(neural, view, view.index[positions]),
                pred, meta["fixed_nn_weights"], horizons, main_s._groups_for(horizons))
        pred = main_s.blend_cube(pred, profile, meta["fixed_fusion"], horizons,
                                 main_s._groups_for(horizons))
        missing = ~np.isfinite(view.frame[["d_use_blast", "d_use_coke", "d_use_converter"]]
                              .to_numpy(dtype=float)[positions]).all(axis=1)
        result[scenario] = (pred, missing)
    return result, positions, main_s.truth_cube(prepared, positions, horizons), meta


def _availability_interval(predictions, baseline, truths):
    """按两天连续起报块重采样，重叠提前量不当作独立样本。"""
    rng = np.random.default_rng(20261025)
    differences = []
    for pred, before, truth in zip(predictions, baseline, truths):
        valid = np.isfinite(truth) & (truth > 0)
        relative = np.where(valid, (np.abs(before - truth) - np.abs(pred - truth)) /
                            np.where(valid, truth, 1.), np.nan)
        differences.append(100. * np.nanmean(relative, axis=(1, 2)))
    values = []
    width = 2 * C.STEPS_PER_DAY
    for _ in range(400):
        means = []
        for delta in differences:
            starts = rng.integers(0, len(delta), size=int(np.ceil(len(delta) / width)))
            indices = ((starts[:, None] + np.arange(width)) % len(delta)).ravel()[:len(delta)]
            means.append(float(np.mean(delta[indices])))
        values.append(np.mean(means))
    return {"block_steps": width, "replicates": 400,
            "gain_pp_95_interval": np.quantile(values, [.025, .975]).tolist(),
            "scope": "reused_development_windows_descriptive_only"}


def _availability_joint_mix(ctx, summary):
    """固定小机组保护权重，只比较总负荷权重；整条分支策略共同验收。"""
    if summary["quick"] or not any(r.get("accepted") for r in summary.get("R0", [])):
        return
    span, horizons, groups = "long", C.HORIZONS_L, C.LONG_GROUPS
    base = ctx.model_dir / span
    r2 = summary["promotion"][span]["R2"]
    conservative = summary["promotion"][span].get("R2_mix")
    if not conservative:
        return
    cases = {}
    specs = [s for s in C.FOLDS if s["role"] == "tuning"]
    for scenario in summary["contract"]["scenarios"]:
        rows = []
        for spec in specs:
            with np.load(base / "R1" / scenario / spec["name"] / "predictions.npz") as a:
                before, truth, missing, positions = (a[k].copy() for k in ("prediction", "truth", "missing", "positions"))
            with np.load(base / "R2" / scenario / spec["name"] / "predictions.npz") as a:
                candidate = a["prediction"].copy()
            with np.load(ctx.model_dir / "R0_long" / spec["name"] / "predictions.npz") as a:
                normal = a["prediction"].copy()
            rows.append((candidate, before, truth, missing, positions, normal))
        cases[scenario] = rows
    selected, search = None, []
    for all_weight in sorted(set([conservative["fallback_weight"], .25, .5, .75, 1.])):
        weights = {"generator_1": {g: conservative["fallback_weight"] for g, _, _ in groups},
                   "generator_all": {g: float(all_weight) for g, _, _ in groups}}
        tests, predictions = {}, {}
        for scenario, rows in cases.items():
            estimates = []
            for candidate, before, truth, missing, positions, normal in rows:
                mixed = main_s.blend_cube(candidate, before, weights, horizons, groups)
                estimates.append(np.where(missing[:, None, None], mixed, normal))
            predictions[scenario] = estimates
            tests[scenario] = main_s.load_promotion(estimates, [r[1] for r in rows], [r[2] for r in rows], horizons)
        protected = all(v >= -.2 for test in tests.values() for delta in test["fold_target_deltas_pp"] for v in delta.values())
        gain = tests["all_missing"]["mean_gain_pp"]
        passed = protected and gain >= .5 and tests["normal"]["mean_gain_pp"] >= .2
        search.append({"total_load_weight": all_weight, "mean_missing_gain_pp": gain, "passed": bool(passed)})
        if passed and (selected is None or gain > selected[0]):
            selected = (gain, weights, tests, predictions)
    summary["joint_recipe_search"] = {"small_group_weight_frozen": conservative["fallback_weight"],
                                       "candidates": search, "scope": "ABC_only_no_retraining"}
    if selected is None:
        raise ValueError("长期正常与缺测分支组合未通过保护门槛，禁止冻结部署")
    gain, weights, tests, predictions = selected
    for scenario, estimates in predictions.items():
        rows = cases[scenario]
        tests[scenario]["paired_block_interval"] = _availability_interval(estimates, [r[1] for r in rows], [r[2] for r in rows])
        for pred, row, spec in zip(estimates, rows, specs):
            target = base / "R2_target_mix" / scenario / spec["name"]
            target.mkdir(parents=True, exist_ok=True)
            meta = json.loads((base / "R2" / spec["name"] / "meta.json").read_text(encoding="utf-8"))
            meta.update(blend=r2["blend"], fallback_weights=weights, model_stage="R2")
            (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            np.savez_compressed(target / "predictions.npz", prediction=pred, truth=row[2], positions=row[4])
    summary["promotion"][span]["R2_target_mix"] = {"accepted": True, "model_stage": "R2",
        "blend": r2["blend"], "fallback_weights": weights, "tests": tests, "composition_includes_R0": True,
        "features": r2.get("features", "gas_no_power")}
    LOGGER.info("联合策略通过：缺测提升 %.4fpp，分目标权重=%s", gain, weights)


def _observed_retained_comparison(ctx, summary, reference, prior, specs):
    """额外对照已保留缺测策略，避免只相对早期失配基线报告改善。"""
    import shutil
    results = {}
    for span, horizons in (("short", C.HORIZONS_S), ("long", C.HORIZONS_L)):
        selected = max(((s, v) for s, v in summary["promotion"][span].items()
                        if not v.get("retained_without_new_promotion") and s != "R2_retained_mix"),
            key=lambda pair: pair[1]["tests"]["all_missing"]["mean_gain_pp"])
        old = max(((s, v) for s, v in prior["promotion"][span].items() if v["accepted"]),
            key=lambda pair: pair[1]["tests"]["all_missing"]["mean_gain_pp"])
        tests, cases = {}, {}
        for scenario in summary["contract"]["scenarios"]:
            after, before, truths = [], [], []
            for spec in specs:
                key = spec["name"]
                with np.load(reference / C.MODEL_DIR / span / "R1" / scenario / key / "predictions.npz") as data:
                    missing, truth, positions = (data[k].copy() for k in ("missing", "truth", "positions"))
                arrays = []
                for directory, stage, policy in ((ctx.root, selected[0], summary), (reference, old[0], prior)):
                    normal_path = (Path("R0_long") / key if span == "long" and
                        any(r.get("accepted") for r in policy.get("R0", [])) else Path(span) / "baseline" / key)
                    with np.load(directory / C.MODEL_DIR / normal_path / "predictions.npz") as data:
                        normal = data["prediction"].copy()
                    with np.load(directory / C.MODEL_DIR / span / stage / scenario / key / "predictions.npz") as data:
                        if (not np.array_equal(data["truth"], truth, equal_nan=True)
                                or not np.array_equal(data["positions"], positions)):
                            raise ValueError("当前时刻特征对照标签或起报点不一致")
                        arrays.append(np.where(missing[:, None, None], data["prediction"], normal))
                after.append(arrays[0])
                before.append(arrays[1])
                truths.append(truth)
            tests[scenario] = main_s.load_promotion(after, before, truths, horizons)
            cases[scenario] = (after, before, truths)
            for values, truth, spec in zip(before, truths, specs):
                target = ctx.model_dir / span / "retained" / scenario / spec["name"]
                target.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(reference / C.MODEL_DIR / span / old[0] / scenario / spec["name"] / "meta.json",
                                target / "meta.json")
                with np.load(reference / C.MODEL_DIR / span / "R1" / scenario / spec["name"] / "predictions.npz") as data:
                    np.savez_compressed(target / "predictions.npz", prediction=values, truth=truth, positions=data["positions"])
        candidates, chosen = [], None
        # 权重跨折、目标和提前量共用；只复用已有折外预测，不新增训练或查看确认折。
        for weight in (1., .5, .25, .1, .05):
            trial, predictions = {}, {}
            for scenario, (after, before, truths) in cases.items():
                predictions[scenario] = [main_s.enforce_physics(weight * a + (1. - weight) * b)
                                         for a, b in zip(after, before)]
                trial[scenario] = main_s.load_promotion(predictions[scenario], before, truths, horizons)
            protected = all(v >= -C.SELECTION["max_worst_fold_drop_pp"] for t in trial.values()
                            for delta in t["fold_target_deltas_pp"] for v in delta.values())
            gain = trial["all_missing"]["mean_gain_pp"]
            passed = protected and gain >= C.SELECTION["min_mean_gain_pp"]
            candidates.append({"weight": weight, "mean_gain_pp": gain, "passed": bool(passed)})
            if passed and (chosen is None or gain > chosen[0]):
                chosen = (gain, weight, trial, predictions)
        accepted = not summary["quick"] and chosen is not None
        if chosen:
            _, weight, tests, predictions = chosen
        else:
            weight = 1.
        results[span] = {"accepted": bool(accepted), "stage": selected[0], "tests": tests,
                         "reference_stage": old[0], "mix_weight": weight, "search": candidates}
        if not summary["quick"]:
            for stage, item in summary["promotion"][span].items():
                item["accepted"] = False
            model_stage = old[1].get("model_stage", old[0])
            source = reference / C.MODEL_DIR / span
            target = ctx.model_dir / span
            shutil.copytree(source / model_stage, target / "retained_model", dirs_exist_ok=True)
            meta = json.loads((source / model_stage / specs[0]["name"] / "meta.json").read_text(encoding="utf-8"))
            if accepted:
                originals = json.loads((reference / C.MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
                recipe = {"kind": meta["kind"], "features": meta["features"], "blend": old[1]["blend"],
                          "fallback_weight": old[1].get("fallback_weight", 1.),
                          "fallback_weights": old[1].get("fallback_weights")}
                if recipe["fallback_weight"] < 1. or recipe["fallback_weights"]:
                    recipe["reference_recipe"] = {k: originals[span][k] for k in ("kind", "features", "blend")}
                stage = "R2_retained_mix"
                summary["promotion"][span][stage] = {**selected[1], "accepted": True,
                    "model_stage": selected[1].get("model_stage", selected[0]), "retained_recipe": recipe,
                    "retained_mix_weight": weight, "tests": tests}
                for scenario, estimates in predictions.items():
                    for pred, truth, spec in zip(estimates, cases[scenario][2], specs):
                        directory = target / stage / scenario / spec["name"]
                        directory.mkdir(parents=True, exist_ok=True)
                        with np.load(target / selected[0] / scenario / spec["name"] / "predictions.npz") as data:
                            np.savez_compressed(directory / "predictions.npz", prediction=pred, truth=truth, positions=data["positions"])
                        metadata = json.loads((target / selected[0] / scenario / spec["name"] / "meta.json").read_text(encoding="utf-8"))
                        metadata.update(retained_mix_weight=weight, retained_recipe=recipe)
                        (directory / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                results[span]["stage"] = stage
            if not accepted:
                # 未过门槛的跨度完整保留旧分支，避免另一个跨度晋级时一起退回失配主模型。
                summary["promotion"][span]["retained"] = {**old[1], "accepted": True,
                    "model_stage": "retained_model", "features": meta["features"], "retained_without_new_promotion": True}
            results[span]["action"] = "promoted" if accepted else "retained"
        LOGGER.info("当前时刻特征相对保留策略 %s：缺测提升 %.4fpp，晋级=%s", span,
                    tests["all_missing"]["mean_gain_pp"], accepted)
    summary["retained_comparison"] = results


def run_availability_study(args) -> int:
    """输入可用性专项：冻结对照、全组缺测回放与独立短长期候选。"""
    import shutil
    from dataclasses import replace
    reference = args.study_reference.resolve()
    if not reference.is_relative_to(C.ROOT.resolve()):
        raise ValueError("可用性专项参考模型必须来自本项目")
    prior = json.loads((reference / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
    if prior.get("status") != "SUCCESS_PLAN_STUDY" or prior.get("quick"):
        raise ValueError("可用性专项要求完整的冻结开发基线")
    observed = args.plan_study == "observed"
    if observed:
        if (prior.get("stages") != ["availability"] or prior.get("observed_at_t")
                or not prior.get("production_path_checks", {}).get("passed")):
            raise ValueError("当前时刻特征专项必须对照已通过实际入口验证的可用性策略")
        if not args.nn_study_run:
            args.nn_study_run = [Path(p) for p in prior.get("nn_studies", [])]
    studies = [str(p.resolve()) for p in args.nn_study_run or []]
    if args.study_resume:
        root = args.study_resume.resolve()
        if not root.is_relative_to(C.EXPERIMENT_OUTPUT_ROOT.resolve()):
            raise ValueError("续跑只能使用本项目实验目录")
        summary = json.loads((root / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
        if (summary.get("reference") != str(reference) or summary.get("quick") != args.quick
                or summary.get("nn_studies") != studies or summary.get("stages") != ["availability"]
                or summary.get("observed_at_t", False) != observed):
            raise ValueError("可用性专项续跑参数不一致")
        ctx = output.RunContext(root.name, root, root / C.SOURCE_DIR, root / C.MODEL_DIR,
            root / C.REPORT_DIR, root / C.REPORT_DIR / C.REPORT_LOG, "plan_availability")
        summary.update(status="RUNNING", errors=[])
    else:
        ctx = output.begin_run(C, mode="plan_observed" if observed else "plan_availability", experimental=True)
        summary = {"status": "RUNNING", "submittable": False, "quick": args.quick,
            "reference": str(reference), "nn_studies": studies, "stages": ["availability"],
            "observed_at_t": observed,
            "validation_scope": ("AB_21_days_screen_only" if args.quick else "ABC_21_days_only")
                                + "; reused_development; no_D; no_test_inputs",
            "results": [], "errors": [], "promotion": {},
            "contract": {"scenarios": ["normal", "all_missing", "outage_day7"],
                "missing_gain_pp": .5, "fold_target_drop_pp": .2,
                "scenario_source": "predeclared_generic_failure_cases_not_test_masks"}}
    _setup_logger(ctx.log_path)
    started, elapsed = time.time(), summary.get("duration_seconds", 0.)
    try:
        output.write_source_snapshot(replace(ctx, source_dir=ctx.source_dir / "续跑")
                                     if args.study_resume else ctx, C)
        specs = [dict(f) for f in C.FOLDS if f["role"] == "tuning"][:2 if args.quick else 3]
        cutoff = max(pd.Timestamp(s["start"]) + pd.Timedelta(days=s["days"] + 1) for s in specs)
        prepared = prepare_data(C, cutoff=cutoff)
        prepared.require_training_data()
        if args.study_resume and summary.get("data_sources") != prepared.source_hashes:
            raise ValueError("训练来源已变化，禁止复用专项模型")
        summary["data_sources"] = prepared.source_hashes
        summary["data_contract"] = data_contract_check(prepared)
        if not summary["data_contract"]["passed"]:
            raise ValueError("可用性专项数据契约失败")
        _study_archive_reference(ctx.root, reference)
        shutil.copyfile(reference / C.MODEL_DIR / "model_meta.json", ctx.model_dir / "model_meta.json")
        summary["R0"] = []
        for study_name in studies:
            study = Path(study_name)
            if not study.is_relative_to(C.ROOT.resolve()):
                raise ValueError("冻结专项必须来自本项目")
            frozen = json.loads((study / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
            if frozen.get("span") != "long" or set(frozen.get("folds", [])) != {"A", "B", "C"}:
                raise ValueError("R0 仅接收完整 A/B/C 长期专项")
            chosen = next(c for c in frozen["selection"] if c["kind"] == "ensemble")
            predictions, baselines, truths, raw_diffs, pipeline_diffs = [], [], [], [], []
            for spec in specs:
                target = ctx.model_dir / "R0_long" / spec["name"]
                if not target.exists():
                    shutil.copytree(study / C.MODEL_DIR / "long" / spec["name"], target)
                with np.load(target / "development_predictions.npz") as a:
                    nn, tree, truth, profile = (a[k].copy() for k in ("prediction", "reference", "truth", "profile"))
                    positions = a["positions"].copy()
                old = ctx.model_dir / "long" / "baseline" / spec["name"]
                meta = json.loads((old / "meta.json").read_text(encoding="utf-8"))
                with np.load(old / "predictions.npz") as a:
                    if not np.array_equal(positions, a["positions"]) or not np.allclose(truth, a["truth"], equal_nan=True):
                        raise ValueError("长期融合对照时刻或标签不一致")
                    raw_diffs.append(float(np.max(np.abs(tree - a["raw"]))))
                    baselines.append(a["prediction"].copy())
                pred = main_s.blend_cube(nn, tree, chosen["nn_weights"], C.HORIZONS_L, C.LONG_GROUPS)
                pred = main_s.blend_cube(pred, profile, chosen["blend"], C.HORIZONS_L, C.LONG_GROUPS)
                predictions.append(pred)
                truths.append(truth)
                # 由真实预测入口重载组件复算，不在单折重新选择部署权重。
                import main_nn
                neural_path = next(target.glob("nn_*.pt"))
                probe_bundle = main_l.LongBundle(kind="nn_lgb_ens", features="hybrid", horizons=C.HORIZONS_L,
                    blend=chosen["blend"], models={}, final_models=_study_load_models(old, meta["files"]),
                    feature_names=meta["feature_names"], final_nn_bundle=main_nn.load_nn(neural_path),
                    nn_weights=chosen["nn_weights"], lgb_kind="global", lgb_features="gas")
                view, _, _, _ = _study_view(prepared, spec)
                reconstructed = main_l.predict_long(probe_bundle, view, view.index[positions], use_final=True)
                pipeline_diffs.append(float(np.max(np.abs(pred - reconstructed))))
                if pipeline_diffs[-1] > 1e-5:
                    raise ValueError("长期冻结融合未被真实预测入口复现")
                np.savez_compressed(target / "predictions.npz", prediction=pred, truth=truth,
                    positions=positions, profile=profile)
                (target / "meta.json").write_text(json.dumps({**meta, "nn_weights": chosen["nn_weights"],
                    "blend": chosen["blend"], "source_study": str(study)}, ensure_ascii=False, indent=2), encoding="utf-8")
            result = main_s.load_promotion(predictions, baselines, truths, C.HORIZONS_L)
            result.update(reference_raw_max_abs_diff=max(raw_diffs), operator_max_abs_diff=max(pipeline_diffs),
                reason="main_pipeline_ranked_tree_components_before_profile_blending",
                component_projection="before_ensemble_in_both_validation_and_inference",
                source_study=str(study), frozen_nn_weights=chosen["nn_weights"], frozen_blend=chosen["blend"])
            if max(raw_diffs) > 1e-8:
                result.update(accepted=False, reason="tree_reference_not_identical")
            summary["R0"].append(result)
        for span, horizons, stride, kind in (("short", C.HORIZONS_S, C.TRAIN_STRIDE_SHORT, "direct"),
                                            ("long", C.HORIZONS_L, C.TRAIN_STRIDE_LONG, "global")):
            cached = []
            for spec in specs:
                LOGGER.info("可用性回放 %s/%s，目录=%s", span, spec["name"], ctx.root)
                directory = ctx.model_dir / span / "baseline" / spec["name"]
                scenarios, evaluation, truth, meta = _availability_fold_predictions(prepared, spec, horizons, directory)
                with np.load(directory / "predictions.npz") as a:
                    difference = float(np.max(np.abs(scenarios["normal"][0] - a["prediction"])))
                if difference > 1e-5:
                    raise ValueError("正常输入未复现冻结基线")
                for name, (pred, missing) in scenarios.items():
                    target = ctx.model_dir / span / "R1" / name / spec["name"]
                    target.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target / "predictions.npz", prediction=pred, truth=truth,
                        positions=evaluation, missing=missing)
                    (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    item = {"stage": "R1", "span": span, "scenario": name, "fold": spec["name"],
                            "score": main_s.score_cube(pred, truth, horizons), "normal_reload_diff": difference}
                    summary["results"] = [r for r in summary["results"] if not all(r.get(k) == item[k]
                        for k in ("stage", "span", "scenario", "fold"))] + [item]
                cached.append({"spec": spec, "evaluation": evaluation, "truth": truth, "scenarios": scenarios})
                output.write_report(ctx, summary)
            for stage in (("R2",) if observed else ("R2", "R3")):
                if stage == "R3" and summary["promotion"][span]["R2"]["accepted"]:
                    break
                raws, profiles, truths = [], [], []
                scenario_raws = {s: [] for s in summary["contract"]["scenarios"]}
                feature_set = "gas_no_power" if stage == "R2" else "gas"
                if observed:
                    feature_set += "_at_t"
                for cache in cached:
                    spec, evaluation = cache["spec"], cache["evaluation"]
                    target = ctx.model_dir / span / stage / spec["name"]
                    masked, _, _, ref = _study_view(prepared, spec)
                    pristine = masked
                    positions, _, _ = main_s._fold_positions(masked, spec, max(horizons))
                    positions = positions[::C.QUICK_TRAIN_STRIDE if args.quick else stride]
                    if stage == "R3":
                        for offset in range(7 * C.STEPS_PER_DAY, ref, 21 * C.STEPS_PER_DAY):
                            masked = masked.masked_power_gas_copy(masked.index[offset],
                                masked.index[min(offset + 14 * C.STEPS_PER_DAY - 1, ref)])
                    inference = pristine.masked_power_gas_copy(spec["start"])
                    profile = main_s.baseline_cubes(inference, evaluation, horizons)["profile"]
                    base = main_s.build_base_features(masked, include_targets=True)
                    metadata_path = target / "meta.json"
                    if metadata_path.exists():
                        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                        if meta["train_end"] != spec["train_end"] or meta["features"] != feature_set:
                            raise ValueError("已有可用性模型契约不匹配")
                        models, names = _study_load_models(target, meta["files"]), meta["feature_names"]
                    else:
                        LOGGER.info("训练 %s/%s/%s，features=%s", stage, span, spec["name"], feature_set)
                        models, names = main_s._train_models(masked, base, positions, horizons, kind,
                            feature_set, C.LGBM_TREES_QUICK if args.quick else C.LGBM_TREES_LONG, ref)
                        files = _study_save_models(target, models)
                        meta = {"kind": kind, "features": feature_set, "feature_names": names, "files": files,
                            "span": span, "train_end": spec["train_end"],
                            "max_training_target": str(masked.index[max(positions) + max(horizons)]),
                            "block_mask_training_days": [14, 21] if stage == "R3" else None}
                        metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    raw = main_s._predict_models(inference, main_s.build_base_features(inference, include_targets=True),
                        evaluation, horizons, kind, feature_set, models, names)
                    loaded = _study_load_models(target, meta["files"])
                    probe = evaluation[::max(1, len(evaluation) // 9)]
                    reloaded = main_s._predict_models(inference, main_s.build_base_features(inference, include_targets=True),
                        probe, horizons, kind, feature_set, loaded, names)
                    reload_diff = float(np.max(np.abs(reloaded - raw[::max(1, len(evaluation) // 9)])))
                    if reload_diff > 1e-8:
                        raise ValueError("可用性模型重载不一致")
                    raw = main_s.enforce_physics(raw, fallback=profile)
                    for scenario in scenario_raws:
                        if stage == "R2" or scenario == "all_missing":
                            scenario_raws[scenario].append(raw)
                            continue
                        start = (pd.Timestamp(spec["start"]) + pd.Timedelta(days=7)
                                 if scenario == "outage_day7" else None)
                        view = pristine if start is None else pristine.masked_power_gas_copy(start)
                        extra_raw = main_s._predict_models(view,
                            main_s.build_base_features(view, include_targets=True), evaluation,
                            horizons, kind, feature_set, models, names)
                        scenario_raws[scenario].append(main_s.enforce_physics(extra_raw, fallback=profile))
                    raws.append(raw)
                    profiles.append(profile)
                    truths.append(cache["truth"])
                    meta["reload_max_abs_diff"] = reload_diff
                    metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    np.savez_compressed(target / "predictions.npz", prediction=raw, raw=raw,
                        profile=profile, truth=cache["truth"], positions=evaluation)
                    output.write_report(ctx, summary)
                blend = main_s.select_blend(raws, profiles, truths, horizons, main_s._groups_for(horizons))
                tests = {}
                for scenario in summary["contract"]["scenarios"]:
                    predictions = [main_s.blend_cube(p, b, blend, horizons, main_s._groups_for(horizons))
                                   for p, b in zip(scenario_raws[scenario], profiles)]
                    baseline = [c["scenarios"][scenario][0] for c in cached]
                    dispatched = [np.where(c["scenarios"][scenario][1][:, None, None], p, b)
                                  for p, b, c in zip(predictions, baseline, cached)]
                    test = main_s.load_promotion(dispatched, baseline, truths, horizons)
                    test["paired_block_interval"] = _availability_interval(dispatched, baseline, truths)
                    tests[scenario] = test
                    for pred, cache in zip(dispatched, cached):
                        target = ctx.model_dir / span / stage / scenario / cache["spec"]["name"]
                        target.mkdir(parents=True, exist_ok=True)
                        meta = json.loads((target.parent.parent / cache["spec"]["name"] / "meta.json").read_text(encoding="utf-8"))
                        meta["blend"] = blend
                        (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                        np.savez_compressed(target / "predictions.npz", prediction=pred, truth=cache["truth"],
                            positions=cache["evaluation"])
                protected = all(v >= -.2 for t in tests.values() for d in t["fold_target_deltas_pp"] for v in d.values())
                accepted = len(specs) == 3 and protected and tests["all_missing"]["mean_gain_pp"] >= .5
                summary["promotion"].setdefault(span, {})[stage] = {"accepted": bool(accepted),
                    "blend": blend, "tests": tests, "features": feature_set,
                    "normal_path": "original_unless_current_power_gas_missing"}
                LOGGER.info("完成 %s/%s：缺耗气提升 %.4fpp，晋级=%s", stage, span,
                    tests["all_missing"]["mean_gain_pp"], accepted)
                output.write_report(ctx, summary)
            if not any(item["accepted"] for item in summary["promotion"][span].values()):
                # 只重用同折预测试验一个共用权重，避免平均收益掩盖最差工况退化。
                r2 = summary["promotion"][span]["R2"]
                candidate_paths = {scenario: [] for scenario in summary["contract"]["scenarios"]}
                for scenario in candidate_paths:
                    for cache in cached:
                        path = ctx.model_dir / span / "R2" / scenario / cache["spec"]["name"] / "predictions.npz"
                        with np.load(path) as arrays:
                            candidate_paths[scenario].append(arrays["prediction"].copy())
                selected_mix = None
                for weight in C.SELECTION["blend_grid"]:
                    tests = {}
                    for scenario, values in candidate_paths.items():
                        before = [c["scenarios"][scenario][0] for c in cached]
                        estimates = [weight * a + (1. - weight) * b for a, b in zip(values, before)]
                        tests[scenario] = main_s.load_promotion(estimates, before, truths, horizons)
                    protected = all(v >= -.2 for t in tests.values() for d in t["fold_target_deltas_pp"] for v in d.values())
                    gain = tests["all_missing"]["mean_gain_pp"]
                    if protected and gain >= .5 and (selected_mix is None or gain > selected_mix[0]):
                        selected_mix = (gain, float(weight), tests)
                if selected_mix:
                    gain, weight, tests = selected_mix
                    for scenario, values in candidate_paths.items():
                        before = [c["scenarios"][scenario][0] for c in cached]
                        estimates = [weight * a + (1. - weight) * b for a, b in zip(values, before)]
                        tests[scenario]["paired_block_interval"] = _availability_interval(estimates, before, truths)
                        for pred, cache in zip(estimates, cached):
                            target = ctx.model_dir / span / "R2_mix" / scenario / cache["spec"]["name"]
                            target.mkdir(parents=True, exist_ok=True)
                            meta = json.loads((ctx.model_dir / span / "R2" / cache["spec"]["name"] / "meta.json").read_text(encoding="utf-8"))
                            meta.update(blend=r2["blend"], fallback_weight=weight, model_stage="R2")
                            (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                            np.savez_compressed(target / "predictions.npz", prediction=pred, truth=cache["truth"], positions=cache["evaluation"])
                    summary["promotion"][span]["R2_mix"] = {"accepted": len(specs) == 3,
                        "model_stage": "R2", "fallback_weight": weight, "blend": r2["blend"], "tests": tests,
                        "features": r2.get("features", "gas_no_power"),
                        "search_scope": "one_common_weight_from_existing_grid; frozen_ABC; no_D"}
                    LOGGER.info("完成 R2_mix/%s：共用权重 %.2f，提升 %.4fpp", span, weight, gain)
                    output.write_report(ctx, summary)
        _availability_joint_mix(ctx, summary)
        if observed:
            _observed_retained_comparison(ctx, summary, reference, prior, specs)
        summary["status"] = "SUCCESS_PLAN_STUDY"
        summary["R4_R5"] = {"status": "RESEARCH_ONLY_PREREQUISITES_UNRESOLVED",
            "reason": "historical_consumption_envelope; holder_mapping_and_real_net_gas_unverified; feedback_availability_requires_separate_validation"}
        return 0
    except Exception as exc:
        LOGGER.exception("可用性专项失败")
        summary["status"] = "FAILED"
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        summary["duration_seconds"] = elapsed + time.time() - started
        output.write_report(ctx, summary)
        LOGGER.info("可用性专项结果: %s", ctx.root)


def run_plan_study(args) -> int:
    """方案专项只访问开发折，不触发正式推理、D 重选或交付打包。"""
    if args.plan_study in ("availability", "observed"):
        return run_availability_study(args)
    stages = set(args.plan_study.split(","))
    allowed = {"baseline", "pressure", "E1", "E2", "E3", "E4", "E5", "E6"}
    if not stages or stages - allowed or (stages & {"baseline", "pressure"} and len(stages) != 1):
        raise ValueError("方案阶段仅允许 baseline、pressure 或 E1～E6 的组合")
    reference = args.study_reference.resolve()
    if not reference.is_relative_to(C.ROOT.resolve()):
        raise ValueError("研究基线必须来自本项目")
    if stages != {"baseline"}:
        previous = json.loads((reference / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
        if previous.get("status") != "SUCCESS_PLAN_STUDY" or previous.get("quick"):
            raise ValueError("候选研究要求已完成的全量开发基线")
    if args.study_resume:
        root = args.study_resume.resolve()
        if not root.is_relative_to(C.EXPERIMENT_OUTPUT_ROOT.resolve()) or stages - {"E4", "E5", "E6"}:
            raise ValueError("续跑仅支持 tmp/tmp_outputs 内的资源方案专项")
        summary = json.loads((root / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
        if (summary["reference"] != str(reference) or set(summary["stages"]) != stages or summary["quick"] != args.quick
                or summary["status"] == "SUCCESS_PLAN_STUDY"):
            raise ValueError("续跑参数不匹配或该专项已完成")
        ctx = output.RunContext(root.name, root, root / C.SOURCE_DIR, root / C.MODEL_DIR,
            root / C.REPORT_DIR, root / C.REPORT_DIR / C.REPORT_LOG, "plan_" + "_".join(sorted(stages)))
        summary["status"] = "RUNNING"
        summary.setdefault("resumes", []).append({"time": str(pd.Timestamp.now()), "previous_errors": summary["errors"][:]})
        summary["duration_scope"] = "recorded_completed_invocations_only; interrupted_process_time_unavailable"
        summary["errors"] = []
    else:
        ctx = output.begin_run(C, mode="plan_" + "_".join(sorted(stages)), experimental=True)
        summary = {"status": "RUNNING", "submittable": False, "quick": args.quick,
            "reference": str(reference), "stages": sorted(stages), "results": [], "errors": [],
            "validation_scope": "ABC_21_days_only; no_D; no_half_test; no_formal_delivery"}
    _setup_logger(ctx.log_path)
    started = time.time()
    previous_seconds = summary.get("duration_seconds", 0.)
    try:
        if args.study_resume:
            from dataclasses import replace
            output.write_source_snapshot(replace(ctx, source_dir=ctx.source_dir / "续跑"), C)
        else:
            output.write_source_snapshot(ctx, C)
        specs = [dict(f) for f in C.FOLDS if f["role"] == "tuning"][:2 if args.quick else 3]
        if stages == {"pressure"}:
            specs = [{"name": "P", "role": "diagnostic", "start": "2025-10-01 00:00:00",
                      "train_end": "2025-09-30 23:45:00", "days": 9}]
            summary["validation_scope"] = "recent_9_day_diagnosis_only; previously_viewed_distribution; no_reselection"
        prepared = prepare_data(C, cutoff=max(pd.Timestamp(f["start"]) + pd.Timedelta(days=f["days"] + 1) for f in specs))
        prepared.require_training_data()
        if args.study_resume and summary.get("data_sources") != prepared.source_hashes:
            raise ValueError("训练数据来源变化，禁止续用旧折外数组")
        summary["data_sources"] = prepared.source_hashes
        summary["data_contract"] = data_contract_check(prepared)
        if not summary["data_contract"]["passed"]:
            raise ValueError("方案专项数据契约未通过")
        if stages == {"baseline"}:
            import shutil
            shutil.copyfile(reference / C.MODEL_DIR / "model_meta.json", ctx.model_dir / "model_meta.json")
            baseline_summary = json.loads((reference / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
            summary["E0"] = {"source": str(reference), "physical_feasibility_certified": False,
                "resource_mode": "historical_consumption_envelope",
                "physical_parameters": baseline_summary["optimization"].get("summary", baseline_summary["optimization"])}
        else:
            import shutil
            shutil.copyfile(reference / C.MODEL_DIR / "model_meta.json", ctx.model_dir / "model_meta.json")
            _study_archive_reference(ctx.root, reference)
        if "E1" in stages or stages == {"baseline"}:
            _study_e0(ctx, summary, prepared, specs, reference)
        if stages & {"baseline", "E1", "E2", "E3"}:
            _study_load_stage(ctx, summary, prepared, specs, reference, stages, args.quick)
        if stages == {"pressure"}:
            _study_load_stage(ctx, summary, prepared, specs, reference, {"baseline"}, args.quick)
            summary["diagnostic_scores"] = summary.pop("promotion")
            for item in summary["diagnostic_scores"].values():
                item["baseline"]["scope"] = "recent_9_day_diagnosis_only_not_independent"
        if stages & {"E4", "E5", "E6"}:
            _study_resource_stage(ctx, summary, prepared, specs, reference, stages, args.quick)
        summary["status"] = "SUCCESS_PLAN_STUDY"
        return 0
    except Exception as exc:
        LOGGER.exception("方案专项失败")
        summary["status"] = "FAILED"
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        summary["duration_seconds"] = previous_seconds + time.time() - started
        output.write_report(ctx, summary)
        LOGGER.info("方案专项结果: %s", ctx.root)


def _generator1_evaluate(cases, predictions, horizons):
    """逐场景保护最差折，并独立检查总负荷，避免综合分掩盖专项退化。"""
    rows, tests, protected = [], {}, True
    for case, pred in zip(cases, predictions):
        before = main_s.score_cube(case["baseline"], case["truth"], horizons)
        after = main_s.score_cube(pred, case["truth"], horizons)
        same = np.array_equal(pred[..., 1], case["baseline"][..., 1])
        delta = after["targets"]["generator_1"]["accuracy"] - before["targets"]["generator_1"]["accuracy"]
        protected &= same and delta >= -C.GENERATOR1_GATE["max_fold_drop_pp"] - 1e-8 and np.isfinite(pred).all()
        rows.append({"fold": case["spec"]["name"], "scenario": case["scenario"],
            "accuracy": after["accuracy"], "generator_1": after["targets"]["generator_1"],
            "baseline_generator_1": before["targets"]["generator_1"]["accuracy"],
            "gain_pp": delta, "total_preserved": same})
    for scenario in ("normal", "all_missing", "outage_day7"):
        group = [r for r in rows if r["scenario"] == scenario]
        worst = min(r["generator_1"]["accuracy"] for r in group)
        before_worst = min(r["baseline_generator_1"] for r in group)
        protected &= worst >= before_worst - 1e-8
        tests[scenario] = {"mean_accuracy": float(np.mean([r["generator_1"]["accuracy"] for r in group])),
            "mean_gain_pp": float(np.mean([r["gain_pp"] for r in group])),
            "worst_fold": worst, "baseline_worst_fold": before_worst}
    accepted = protected and (tests["normal"]["mean_gain_pp"] >= C.GENERATOR1_GATE["normal_gain_pp"] or
                              tests["all_missing"]["mean_gain_pp"] >= C.GENERATOR1_GATE["missing_gain_pp"])
    return {"accepted": bool(accepted), "protected": bool(protected), "tests": tests, "rows": rows}


def run_generator1_study(args) -> int:
    """小机组组专项独立运行，全部开发产物留在实验根目录且不访问D或测试输入。"""
    stages = set(args.g1_stages.split(","))
    if not stages or stages - {f"G{i}" for i in range(8)}:
        raise ValueError("generator_1阶段必须为G0至G7的非空逗号列表")
    reference = args.study_reference.resolve()
    if not reference.is_relative_to(C.EXPERIMENT_OUTPUT_ROOT.resolve()):
        raise ValueError("generator_1参考必须来自本项目实验目录")
    prior = json.loads((reference / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
    if prior.get("status") != "SUCCESS_PLAN_STUDY" or prior.get("quick"):
        raise ValueError("generator_1要求完整开发配方")
    if args.study_resume:
        root = args.study_resume.resolve()
        if not root.is_relative_to(C.EXPERIMENT_OUTPUT_ROOT.resolve()):
            raise ValueError("generator_1续跑目录越界")
        summary = json.loads((root / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
        if summary.get("reference") != str(reference) or summary.get("quick") != args.quick:
            raise ValueError("generator_1续跑契约不一致")
        ctx = output.RunContext(root.name, root, root / C.SOURCE_DIR, root / C.MODEL_DIR,
            root / C.REPORT_DIR, root / C.REPORT_DIR / C.REPORT_LOG, "generator1")
    else:
        ctx = output.begin_run(C, mode="generator1", experimental=True)
        summary = {"reference": str(reference), "quick": args.quick, "submittable": False,
            "validation_scope": "AB_screen" if args.quick else "ABC_21_days_reused_development_no_D_no_test",
            "completed_stages": [], "candidates": {}, "errors": [],
            "contract": {"normal_gain_pp": .5, "missing_gain_pp": 1., "max_fold_drop_pp": .2,
                "total_load_frozen": True, "optimization_frozen": True,
                "stage_weights": "three_predeclared_scenarios_equal_weight"}}
    _setup_logger(ctx.log_path)
    started, elapsed = time.time(), summary.get("duration_seconds", 0.)
    if summary.get("errors"):
        summary.setdefault("recovered_failures", []).append({"errors": summary["errors"],
            "duration_seconds": summary.get("duration_seconds"), "action": "resume_completed_models_without_retraining"})
    if any(summary["contract"].get(k) != v for k, v in C.GENERATOR1_GATE.items()):
        summary.setdefault("policy_changes", []).append({"date": "2026-10-25",
            "source": "human_user_explicit_instruction", "reason": "平均提升门槛不用很高，差不多即可",
            "before": {k: summary["contract"].get(k) for k in C.GENERATOR1_GATE},
            "after": dict(C.GENERATOR1_GATE)})
        summary["contract"].update(C.GENERATOR1_GATE)
    for candidates in summary["candidates"].values():
        for result in candidates.values():
            result["accepted"] = bool(not args.quick and result["protected"] and (
                result["tests"]["normal"]["mean_gain_pp"] >= C.GENERATOR1_GATE["normal_gain_pp"] or
                result["tests"]["all_missing"]["mean_gain_pp"] >= C.GENERATOR1_GATE["missing_gain_pp"]))
    summary.update(status="RUNNING", errors=[])
    output.write_source_snapshot(ctx, C)
    output.write_report(ctx, summary)
    LOGGER.info("generator_1专项目录：%s", ctx.root)
    try:
        specs = [dict(f) for f in C.FOLDS if f["role"] == "tuning"][:2 if args.quick else 3]
        cutoff = max(pd.Timestamp(s["start"]) + pd.Timedelta(days=s["days"] + 1) for s in specs)
        prepared = prepare_data(C, cutoff=cutoff)
        prepared.require_training_data()
        if summary.get("data_sources") and summary["data_sources"] != prepared.source_hashes:
            raise ValueError("generator_1数据来源已变化")
        summary["data_sources"] = prepared.source_hashes
        summary["data_contract"] = data_contract_check(prepared)
        if not summary["data_contract"]["passed"]:
            raise ValueError("generator_1数据契约失败")
        pools = {}
        for span, horizons in (("short", C.HORIZONS_S), ("long", C.HORIZONS_L)):
            accepted = [(name, x) for name, x in prior["promotion"][span].items() if x.get("accepted")]
            stage, _ = max(accepted, key=lambda item: item[1]["tests"]["all_missing"]["mean_accuracy"])
            cases = []
            for spec in specs:
                view, _, positions, ref = _study_view(prepared, spec)
                for scenario, start in (("normal", None), ("all_missing", spec["start"]),
                    ("outage_day7", pd.Timestamp(spec["start"]) + pd.Timedelta(days=7))):
                    current = view if start is None else view.masked_power_gas_copy(start)
                    with np.load(reference / C.MODEL_DIR / span / stage / scenario / spec["name"] / "predictions.npz") as data:
                        baseline, truth = data["prediction"].copy(), data["truth"].copy()
                        if not np.array_equal(data["positions"], positions):
                            raise ValueError("generator_1起报点不一致")
                    with np.load(reference / C.MODEL_DIR / span / "R2" / scenario / spec["name"] / "predictions.npz") as data:
                        expert = data["prediction"].copy()
                    expected = main_s.truth_cube(prepared, positions, horizons)
                    if not np.array_equal(expected, truth, equal_nan=True):
                        raise ValueError("generator_1标签归档不一致")
                    missing = ~np.isfinite(current.frame[["d_use_blast", "d_use_coke", "d_use_converter"]].to_numpy()[positions]).all(axis=1)
                    case = {"span": span, "spec": spec, "scenario": scenario, "view": current,
                        "positions": positions, "ref": ref, "baseline": baseline, "truth": truth,
                        "expert": expert, "missing": missing}
                    directory = ctx.model_dir / span / "baseline" / scenario / spec["name"]
                    directory.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(directory / "predictions.npz", prediction=baseline, truth=truth,
                                        positions=positions, expert=expert, missing=missing)
                    cases.append(case)
            pools[span] = cases
            summary["candidates"].setdefault(span, {})
            summary["candidates"][span]["baseline"] = _generator1_evaluate(cases, [x["baseline"] for x in cases], horizons)
            if "G1" in stages:
                for weight in (.05, .1, .25, .5, 1.):
                    predictions = [main_s.generator1_project(x["baseline"][..., 0] +
                        weight * x["missing"][:, None] * (x["expert"][..., 0] - x["baseline"][..., 0]), x["baseline"]) for x in cases]
                    name = f"G1_fixed_{weight:g}"
                    result = _generator1_evaluate(cases, predictions, horizons)
                    result["accepted"] &= not args.quick
                    result["recipe"] = {"kind": "availability_expert", "weight": weight,
                        "missing_only": True, "source_stage": "R2"}
                    summary["candidates"][span][name] = result
                    for case, pred in zip(cases, predictions):
                        directory = ctx.model_dir / span / name / case["scenario"] / case["spec"]["name"]
                        directory.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(directory / "predictions.npz", prediction=pred,
                            truth=case["truth"], positions=case["positions"])
                    LOGGER.info("%s/%s 缺测增益 %.4fpp，保护=%s，晋级=%s", span, name,
                        result["tests"]["all_missing"]["mean_gain_pp"], result["protected"], result["accepted"])
                # 门控的低复杂度对照：只有专家分歧超过容量尺度的缓冲带才逐步修正。
                for margin in (5., 10., 20.):
                    predictions = []
                    for case in cases:
                        delta = case["expert"][..., 0] - case["baseline"][..., 0]
                        weight = np.clip((np.abs(delta) - margin) / margin, 0., 1.)
                        predictions.append(main_s.generator1_project(case["baseline"][..., 0] +
                            weight * case["missing"][:, None] * delta, case["baseline"]))
                    _generator1_record(ctx, summary, span, f"G2_disagreement_{margin:g}mw", cases,
                        predictions, horizons, {"kind": "availability_expert", "weight": 1.,
                        "missing_only": True, "source_stage": "R2", "disagreement_margin_mw": margin})
        summary["completed_stages"] = sorted(set(summary["completed_stages"]) | (stages & {"G0", "G1"}))
        output.write_report(ctx, summary)
        _generator1_train_stages(args, ctx, summary, prepared, specs, pools, stages)
        summary["status"] = "SUCCESS_GENERATOR1_STUDY"
    except Exception as exc:
        summary["status"] = "FAILED"
        summary["errors"].append(str(exc))
        LOGGER.exception("generator_1专项失败")
    summary["duration_seconds"] = elapsed + time.time() - started
    output.write_report(ctx, summary)
    print(f"generator_1专项：{summary['status']} | {ctx.root}")
    return 0 if summary["status"] == "SUCCESS_GENERATOR1_STUDY" else 1


def run_generator1_refinement(args) -> int:
    """第二轮直接对照已晋级策略，先复用既有权重，再比较不同信息和目标表示。"""
    import main_nn
    requested = set(args.g1_refine_stages.split(","))
    if not requested or requested - {"reuse", "train", "normal", "seeds", "combine"}:
        raise ValueError("精修阶段必须为reuse、train、normal、seeds、combine的非空列表")
    reference = args.study_reference.resolve()
    if not reference.is_relative_to(C.EXPERIMENT_OUTPUT_ROOT.resolve()):
        raise ValueError("精修参考目录必须来自本项目实验目录")
    prior = json.loads((reference / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
    if prior.get("status") != "SUCCESS_GENERATOR1_STUDY" or not prior.get("verification", {}).get("passed"):
        raise ValueError("精修要求已完成核验的全量专项")
    if args.study_resume:
        root = args.study_resume.resolve()
        if not root.is_relative_to(C.EXPERIMENT_OUTPUT_ROOT.resolve()):
            raise ValueError("精修续跑目录越界")
        summary = json.loads((root / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
        if summary["reference"] != str(reference) or summary["quick"] != args.quick:
            raise ValueError("精修续跑契约不一致")
        ctx = output.RunContext(root.name, root, root / C.SOURCE_DIR, root / C.MODEL_DIR,
            root / C.REPORT_DIR, root / C.REPORT_DIR / C.REPORT_LOG, "generator1_refine")
    else:
        ctx = output.begin_run(C, mode="generator1_refine", experimental=True)
        summary = {"reference": str(reference), "quick": args.quick, "submittable": False,
            "validation_scope": ("AB_screen_only" if args.quick else "ABC_reused_development") + "; current_promoted_baseline; no_D_no_test",
            "candidates": {}, "neural_recipes": {}, "completed_stages": [], "errors": [],
            "contract": {**C.GENERATOR1_GATE, "total_load_frozen": True,
                "normal_path_is_independently_evaluated": True,
                "hypotheses": ["seed_aggregation", "complementary_models", "observed_power_gas",
                               "profile_residual", "profile_ratio"]}}
    _setup_logger(ctx.log_path)
    tick, elapsed = time.time(), summary.get("duration_seconds", 0.)
    summary.setdefault("run_id", ctx.run_id)
    if "base_recipes" not in summary:
        # 冻结配置会随晋级更新，复现实验须读取参考专项自己的配方。
        summary["base_recipes"] = _generator1_study_recipes(reference, prior)
        if set(summary["base_recipes"]) != {"short", "long"}:
            raise ValueError("精修参考专项必须同时提供短期与长期已晋级配方")
    summary.update(status="RUNNING", errors=[])
    output.write_source_snapshot(ctx, C)
    output.write_report(ctx, summary)
    LOGGER.info("generator_1第二轮目录：%s", ctx.root)
    try:
        specs = [dict(f) for f in C.FOLDS if f["role"] == "tuning"][:2 if args.quick else 3]
        cutoff = max(pd.Timestamp(s["start"]) + pd.Timedelta(days=s["days"] + 1) for s in specs)
        prepared = prepare_data(C, cutoff=cutoff)
        prepared.require_training_data()
        if summary.get("data_sources") and summary["data_sources"] != prepared.source_hashes:
            raise ValueError("精修续跑训练来源已变化")
        summary["data_sources"] = prepared.source_hashes
        pools = {}
        for span, horizons in (("short", C.HORIZONS_S), ("long", C.HORIZONS_L)):
            cases, selected = [], prior["selection"][span]["chosen"]
            for spec in specs:
                view, _, positions, ref = _study_view(prepared, spec)
                for scenario, start in (("normal", None), ("all_missing", spec["start"]),
                    ("outage_day7", pd.Timestamp(spec["start"]) + pd.Timedelta(days=7))):
                    current = view if start is None else view.masked_power_gas_copy(start)
                    with np.load(reference / C.MODEL_DIR / span / selected / scenario / spec["name"] / "predictions.npz") as data:
                        baseline, truth = data["prediction"].copy(), data["truth"].copy()
                        if not np.array_equal(positions, data["positions"]):
                            raise ValueError("精修起报点与参考不一致")
                    if not np.array_equal(main_s.truth_cube(prepared, positions, horizons), truth, equal_nan=True):
                        raise ValueError("精修真值与参考不一致")
                    missing = ~np.isfinite(current.frame[["d_use_blast", "d_use_coke", "d_use_converter"]].to_numpy()[positions]).all(axis=1)
                    case = {"span": span, "spec": spec, "scenario": scenario, "view": current,
                        "positions": positions, "ref": ref, "baseline": baseline, "truth": truth, "missing": missing}
                    folder = ctx.model_dir / span / "baseline" / scenario / spec["name"]
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(folder / "predictions.npz", prediction=baseline, truth=truth, positions=positions)
                    cases.append(case)
            pools[span] = cases
            summary["candidates"].setdefault(span, {})
            summary["candidates"][span]["baseline"] = _generator1_evaluate(cases, [c["baseline"] for c in cases], horizons)
        if "reuse" in args.g1_refine_stages.split(","):
            _generator1_refine_reuse(ctx, summary, prior, pools)
            summary["completed_stages"] = sorted(set(summary["completed_stages"]) | {"reuse"})
        if set(args.g1_refine_stages.split(",")) & {"train", "normal"}:
            common = {"feature_set": "generator1", "head_mode": "scaled", "profile_base": "plain",
                "profile_conditions": True, "profile_conditions_scaled": True, "training_mask_days": 21,
                "load_history": False, "target_indices": [0], "n_seeds": 1}
            configs = [("H_observed", {**common, "feature_set": "generator1_observed"}),
                       ("H_residual", {**common, "head_mode": "residual", "residual_limit": .25}),
                       ("H_ratio", {**common, "head_mode": "ratio"}),
                       ("H_normal", {**common, "feature_set": "generator1_observed", "power_mask_training": False})]
            if "normal" in args.g1_refine_stages.split(",") and "train" not in args.g1_refine_stages.split(","):
                configs = [x for x in configs if x[0] == "H_normal"]
            for span, horizons in (("short", C.HORIZONS_S), ("long", C.HORIZONS_L)):
                for name, cfg in configs:
                    summary["neural_recipes"][name] = {"kind": "tsmixer", "config": cfg}
                    _generator1_neural_candidate(args, ctx, summary, prepared, specs, pools[span], span,
                                                 horizons, name, "tsmixer", cfg, False)
            summary["completed_stages"] = sorted(set(summary["completed_stages"]) | set(args.g1_refine_stages.split(",")) & {"train", "normal"})
        if "seeds" in args.g1_refine_stages.split(","):
            if not summary["neural_recipes"]:
                raise ValueError("种子复核需要先完成train或normal阶段")
            for span, horizons in (("short", C.HORIZONS_S), ("long", C.HORIZONS_L)):
                ranking = []
                for name, recipe in summary["neural_recipes"].items():
                    rows = [r for r in summary["candidates"][span].values() if r.get("recipe", {}).get("model_stage") == name]
                    if rows:
                        gain = max(np.mean([t["mean_gain_pp"] for t in r["tests"].values()]) for r in rows)
                        ranking.append((gain, name, recipe))
                for _, name, recipe in sorted(ranking, key=lambda x: x[0], reverse=True)[:2]:
                    _generator1_neural_candidate(args, ctx, summary, prepared, specs, pools[span], span,
                        horizons, name + "_3seeds", recipe["kind"], recipe["config"], True, name)
            summary["completed_stages"] = sorted(set(summary["completed_stages"]) | {"seeds"})
        if "combine" in requested:
            for span, horizons in (("short", C.HORIZONS_S), ("long", C.HORIZONS_L)):
                rows = summary["candidates"][span]
                score = lambda row: np.mean([x["mean_gain_pp"] for x in row["tests"].values()])
                aggregations = [(n, r) for n, r in rows.items() if r.get("accepted") and r.get("recipe", {}).get("kind") == "primary_aggregation"]
                networks = [(n, r) for n, r in rows.items() if r.get("accepted") and r.get("recipe", {}).get("kind") == "nn"
                            and r["recipe"].get("seeds") == 3 and r["recipe"]["weight"] < 1. and not r["recipe"].get("base_aggregation")]
                if not aggregations or not networks:
                    continue
                first, aggregate = max(aggregations, key=lambda x: score(x[1]))
                second, neural = max(networks, key=lambda x: score(x[1]))
                update, predictions = neural["recipe"], []
                for spec in specs:
                    bundle = main_nn.load_nn(ctx.model_dir / span / update["model_stage"] / spec["name"] / f"nn_{update['nn_kind']}.pt")
                    for case in [c for c in pools[span] if c["spec"]["name"] == spec["name"]]:
                        with np.load(ctx.model_dir / span / first / case["scenario"] / spec["name"] / "predictions.npz") as data:
                            before = data["prediction"].copy()
                        raw = main_nn.predict_nn(bundle, case["view"], case["view"].index[case["positions"]])[..., 0]
                        active = ~case["missing"] if update.get("normal_only") else case["missing"] if update.get("missing_only") else np.ones(len(raw), dtype=bool)
                        predictions.append(main_s.generator1_project(before[..., 0] + update["weight"] * active[:, None] * (raw - before[..., 0]), before))
                    del bundle
                    main_nn.clear_cache()
                _generator1_record(ctx, summary, span, "combined_" + second, pools[span], predictions, horizons,
                    {**update, "base_aggregation": aggregate["recipe"], "components": [first, second]})
            summary["completed_stages"] = sorted(set(summary["completed_stages"]) | {"combine"})
        summary["selection"] = {}
        for span, results in summary["candidates"].items():
            eligible = [(name, r) for name, r in results.items() if r.get("accepted")]
            chosen = max(eligible, key=lambda x: np.mean([t["mean_gain_pp"] for t in x[1]["tests"].values()]), default=None)
            summary["selection"][span] = {"chosen": chosen[0] if chosen else "baseline", "accepted": chosen is not None}
        summary["status"] = "SUCCESS_GENERATOR1_REFINEMENT"
    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["status"] = "FAILED"
        LOGGER.exception("generator_1第二轮失败")
    summary["duration_seconds"] = elapsed + time.time() - tick
    output.write_report(ctx, summary)
    print(f"generator_1第二轮：{summary['status']} | {ctx.root}")
    return 0 if summary["status"] == "SUCCESS_GENERATOR1_REFINEMENT" else 1


def _generator1_refine_reuse(ctx, summary, prior, pools):
    import main_nn
    reference = Path(summary["reference"])
    for span, horizons in (("short", C.HORIZONS_S), ("long", C.HORIZONS_L)):
        cases = pools[span]
        # 共用少量权重，验证互补性；不逐折、逐日或逐样本拟合权重。
        names = ["G6_price_mixer_3seeds_all", "G5_specialist_tcn_3seeds_all"] if span == "short" else ["G6_price_mixer_3seeds_all", "G6_specialist_mixer_3seeds_all"]
        for source in names:
            estimates = []
            for case in cases:
                with np.load(reference / C.MODEL_DIR / span / source / case["scenario"] / case["spec"]["name"] / "predictions.npz") as data:
                    estimates.append(data["prediction"][..., 0].copy())
            for weight in (.1, .25):
                for mode in ("missing", "normal", "all"):
                    predictions = []
                    for case, estimate in zip(cases, estimates):
                        active = case["missing"] if mode == "missing" else ~case["missing"] if mode == "normal" else np.ones(len(estimate), dtype=bool)
                        predictions.append(main_s.generator1_project(case["baseline"][..., 0] +
                            weight * active[:, None] * (estimate - case["baseline"][..., 0]), case["baseline"]))
                    _generator1_record(ctx, summary, span, f"reuse_{source}_{mode}_{weight:g}", cases, predictions,
                        horizons, {"kind": "secondary_blend", "source": source, "weight": weight, "mode": mode})
        if span == "short":
            source = prior["candidates"][span][prior["selection"][span]["chosen"]]["recipe"]["model_stage"]
            for aggregation, power in (("median", 1.), ("validation_weighted", 1.), ("validation_weighted", 2.)):
                predictions = []
                for spec in [dict(f) for f in C.FOLDS if f["role"] == "tuning"][:2 if summary["quick"] else 3]:
                    bundle = main_nn.load_nn(reference / C.MODEL_DIR / span / source / spec["name"] / "nn_tsmixer.pt")
                    bundle.config.update(ensemble_aggregation=aggregation, aggregation_power=power)
                    for case in [c for c in cases if c["spec"]["name"] == spec["name"]]:
                        estimate = main_nn.predict_nn(bundle, case["view"], case["view"].index[case["positions"]])[..., 0]
                        predictions.append(main_s.generator1_project(np.where(case["missing"][:, None],
                            estimate, case["baseline"][..., 0]), case["baseline"]))
                    del bundle
                    main_nn.clear_cache()
                _generator1_record(ctx, summary, span, f"aggregation_{aggregation}_{power:g}", cases, predictions,
                    horizons, {"kind": "primary_aggregation", "aggregation": aggregation, "power": power})
        else:
            original_reference = Path(prior["reference"])
            expert_values, originals = [], []
            for spec in [dict(f) for f in C.FOLDS if f["role"] == "tuning"][:2 if summary["quick"] else 3]:
                folder = original_reference / C.MODEL_DIR / span / "R2" / spec["name"]
                meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
                seed_models = [_study_load_models(folder, meta["files"])]
                for seed in (20261022, 20261023):
                    seed_folder = reference / C.MODEL_DIR / span / "G7_availability" / str(seed) / spec["name"]
                    other = json.loads((seed_folder / "meta.json").read_text(encoding="utf-8"))
                    seed_models.append(_study_load_models(seed_folder, other["files"]))
                for case in [c for c in cases if c["spec"]["name"] == spec["name"]]:
                    with np.load(reference / C.MODEL_DIR / span / "baseline" / case["scenario"] / spec["name"] / "predictions.npz") as data:
                        originals.append(data["prediction"].copy())
                    members = [np.clip(main_s._predict_models(case["view"], main_s.build_base_features(case["view"], True),
                        case["positions"], horizons, meta["kind"], meta["features"], models, meta["feature_names"],
                        targets=("generator_1",))[..., 0], 0., 200.) for models in seed_models]
                    profile = main_s.target_slot_profile(case["view"], case["positions"], horizons, "generator_1")
                    for values in members:
                        for group, lo, hi in main_s._groups_for(horizons):
                            js = [j for j, h in enumerate(horizons) if lo <= h <= hi]
                            w = C.GENERATOR1_SPECIALIST[span]["expert_blend"]["generator_1"][group]
                            values[:, js] = w * values[:, js] + (1. - w) * profile[:, js]
                    expert_values.append(members)
            for aggregation in ("mean", "median"):
                for margin in (16., 20., 24.):
                    predictions = []
                    for case, members, original in zip(cases, expert_values, originals):
                        expert = np.mean(members, axis=0) if aggregation == "mean" else np.median(members, axis=0)
                        delta = expert - original[..., 0]
                        w = case["missing"][:, None] * np.clip((np.abs(delta) - margin) / margin, 0., 1.)
                        predictions.append(main_s.generator1_project(original[..., 0] + w * delta, case["baseline"]))
                    _generator1_record(ctx, summary, span, f"tree_aggregation_{aggregation}_gate{margin:g}", cases,
                        predictions, horizons, {"kind": "primary_tree_gate", "expert_aggregation": aggregation,
                                               "disagreement_margin_mw": margin})


def _generator1_train_stages(args, ctx, summary, prepared, specs, pools, stages):
    import lightgbm as lgb
    for span, horizons in (("short", C.HORIZONS_S), ("long", C.HORIZONS_L)):
        cases = pools[span]
        if "G2" in stages:
            predictions, provenance = [], []
            for spec in specs:
                directory = ctx.model_dir / span / "G2_gate" / spec["name"]
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "gate.txt"
                if path.exists():
                    gate = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
                    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
                else:
                    gate, meta = _generator1_inner_gate(args, ctx, prepared, spec, span, horizons)
                    path.write_text(gate.model_to_string(), encoding="utf-8")
                    (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                provenance.append(meta)
                for case in [x for x in cases if x["spec"]["name"] == spec["name"]]:
                    before, after = case["baseline"][..., 0], case["expert"][..., 0]
                    X, names = main_s.generator1_gate_matrix(case["view"], case["positions"], horizons, before, after)
                    if names != meta["feature_names"]:
                        raise ValueError("门控特征顺序不匹配")
                    weight = np.clip(gate.predict(X).reshape(len(horizons), -1).T, 0., 1.)
                    pred = main_s.generator1_project(before + case["missing"][:, None] * weight * (after - before), case["baseline"])
                    predictions.append(pred)
            _generator1_record(ctx, summary, span, "G2_gate", cases, predictions, horizons,
                {"kind": "nested_gate", "provenance": provenance,
                 "inner_reference": "same_recipe_tree_proxy_without_neural; outer_reference_is_frozen_deployed"})
        for stage, variants in (("G3", (("direct", False, False), ("profile_residual", True, False))),
                                ("G4", (("blocked_direct", False, True), ("blocked_residual", True, True)))):
            if stage not in stages:
                continue
            for variant, residual, blocked in variants:
                raw_predictions, audits = [], []
                for spec in specs:
                    fold_cases = [x for x in cases if x["spec"]["name"] == spec["name"]]
                    directory = ctx.model_dir / span / f"{stage}_{variant}" / spec["name"]
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / "model.txt"
                    if path.exists():
                        pack = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
                        pack["model"] = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
                    else:
                        LOGGER.info("generator_1训练 %s/%s/%s", span, variant, spec["name"])
                        tick = time.time()
                        pack = main_s.generator1_fit(prepared, fold_cases[0]["ref"], horizons,
                            residual=residual, blocked=blocked, quick=args.quick)
                        pack["seconds"] = time.time() - tick
                        path.write_text(pack["model"].model_to_string(), encoding="utf-8")
                        (directory / "meta.json").write_text(json.dumps({k: v for k, v in pack.items() if k != "model"},
                            ensure_ascii=False, indent=2), encoding="utf-8")
                        LOGGER.info("generator_1完成 %s/%s/%s：%.1fs", span, variant, spec["name"], pack["seconds"])
                    audits.append({k: pack[k] for k in ("train_end", "max_training_target", "rows", "seconds")})
                    loaded = {**pack, "model": lgb.Booster(model_str=path.read_text(encoding="utf-8"))}
                    for case in fold_cases:
                        raw = main_s.generator1_predict(pack, case["view"], case["positions"], horizons)
                        probe = case["positions"][::251]
                        check = main_s.generator1_predict(loaded, case["view"], probe, horizons)
                        if not np.allclose(raw[::251], check, atol=1e-8, rtol=0.):
                            raise ValueError("generator_1树模型重载不一致")
                        raw_predictions.append(raw)
                for weight in (.25, .5, 1.):
                    predictions = [main_s.generator1_project(case["baseline"][..., 0] + weight * case["missing"][:, None] *
                        (raw - case["baseline"][..., 0]), case["baseline"]) for case, raw in zip(cases, raw_predictions)]
                    name = f"{stage}_{variant}_missing_w{weight:g}"
                    _generator1_record(ctx, summary, span, name, cases, predictions, horizons,
                        {"kind": "tree", "model_stage": f"{stage}_{variant}", "weight": weight,
                         "missing_only": True, "training": audits, "reload_passed": True})
                predictions = [main_s.generator1_project(raw, case["baseline"]) for case, raw in zip(cases, raw_predictions)]
                _generator1_record(ctx, summary, span, f"{stage}_{variant}_all", cases, predictions, horizons,
                    {"kind": "tree", "model_stage": f"{stage}_{variant}", "weight": 1.,
                     "missing_only": False, "training": audits, "reload_passed": True})
    completed = stages & {"G2", "G3", "G4"}
    summary["completed_stages"] = sorted(set(summary["completed_stages"]) | completed)
    output.write_report(ctx, summary)
    if stages & {"G5", "G6", "G7"}:
        _generator1_neural_stages(args, ctx, summary, prepared, specs, pools, stages)


def _generator1_record(ctx, summary, span, name, cases, predictions, horizons, recipe):
    result = _generator1_evaluate(cases, predictions, horizons)
    result["accepted"] &= not summary["quick"]
    result["recipe"] = recipe
    for case, pred in zip(cases, predictions):
        directory = ctx.model_dir / span / name / case["scenario"] / case["spec"]["name"]
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "predictions.npz", prediction=pred,
            truth=case["truth"], positions=case["positions"])
    summary["candidates"][span][name] = result
    output.write_report(ctx, summary)
    LOGGER.info("%s/%s 正常增益 %.3fpp，缺测增益 %.3fpp，保护=%s，晋级=%s", span, name,
        result["tests"]["normal"]["mean_gain_pp"], result["tests"]["all_missing"]["mean_gain_pp"],
        result["protected"], result["accepted"])


def _generator1_inner_gate(args, ctx, prepared, outer, span, horizons):
    """内层专家在更早前缀拟合，门控训练不能看到本外层窗口的任何标签。"""
    Xs, old, new, truths, provenance = [], [], [], [], []
    kind = "direct" if span == "short" else "global"
    for days in (63, 35):
        start = pd.Timestamp(outer["start"]) - pd.Timedelta(days=days)
        spec = {"name": f"{outer['name']}_inner{days}", "role": "tuning", "start": str(start),
                "days": 21, "train_end": str(start - pd.Timedelta(minutes=15))}
        directory = ctx.model_dir / span / "G2_inner" / spec["name"]
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / "predictions.npz"
        if archive.exists():
            with np.load(archive) as data:
                X, before, after, truth = [data[k].copy() for k in ("X", "before", "after", "truth")]
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            names = meta["feature_names"]
        else:
            LOGGER.info("门控内层专家 %s/%s，训练截止=%s", span, spec["name"], spec["train_end"])
            view, positions, evaluation, ref = _study_view(prepared, spec)
            view = view.masked_power_gas_copy(start)
            training = prepared.prefix_copy(spec["train_end"])
            positions = positions[::C.QUICK_TRAIN_STRIDE if args.quick else (1 if span == "short" else 2)]
            before, after = None, None
            for label, features in (("before", "gas"), ("after", "gas_no_power_at_t")):
                models, feature_names = main_s._train_models(training,
                    main_s.build_base_features(training, include_targets=True), positions, horizons,
                    kind, features, C.LGBM_TREES_QUICK if args.quick else C.LGBM_TREES_LONG, ref,
                    targets=("generator_1",))
                files = _study_save_models(directory / label, models)
                pred = main_s._predict_models(view, main_s.build_base_features(view, True), evaluation,
                    horizons, kind, features, models, feature_names, targets=("generator_1",))[..., 0]
                if label == "before":
                    before = np.clip(pred, 0., 200.)
                else:
                    after = np.clip(pred, 0., 200.)
                (directory / label / "meta.json").write_text(json.dumps({"features": features,
                    "kind": kind, "feature_names": feature_names, "files": files,
                    "train_end": spec["train_end"]}, ensure_ascii=False, indent=2), encoding="utf-8")
            truth = main_s.truth_cube(prepared, evaluation, horizons)[..., 0]
            X, names = main_s.generator1_gate_matrix(view, evaluation, horizons, before, after)
            meta = {"spec": spec, "feature_names": names,
                "max_training_target": str(prepared.index[positions[-1] + max(horizons)]),
                "max_gate_target": str(prepared.index[evaluation[-1] + max(horizons)])}
            if pd.Timestamp(meta["max_gate_target"]) > pd.Timestamp(outer["train_end"]):
                raise ValueError("门控标签跨入外层窗口")
            np.savez_compressed(archive, X=X, before=before, after=after, truth=truth)
            (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        Xs.append(X)
        old.append(before.T.ravel())
        new.append(after.T.ravel())
        truths.append(truth.T.ravel())
        provenance.append(meta)
    # 单列形状使训练展平顺序与按提前量堆叠的特征严格对应。
    gate = main_s.generator1_gate_fit(np.concatenate(Xs), np.concatenate(old)[:, None],
        np.concatenate(new)[:, None], np.concatenate(truths)[:, None], names)
    return gate, {"feature_names": names, "outer_train_end": outer["train_end"],
        "inner": provenance, "target": "prefix_oof_optimal_convex_weight"}


def _generator1_neural_stages(args, ctx, summary, prepared, specs, pools, stages):
    import main_nn
    configs = []
    common = {"feature_set": "generator1", "head_mode": "scaled", "profile_base": "plain",
        "profile_conditions": True, "profile_conditions_scaled": True,
        "training_mask_days": 21, "load_history": False, "n_seeds": 1}
    if "G5" in stages:
        configs += [("G5_joint_tcn", "tcn_causal", {**common, "target_indices": [0, 1]}),
                    ("G5_specialist_tcn", "tcn_causal", {**common, "target_indices": [0]})]
    if "G6" in stages:
        configs += [("G6_specialist_mixer", "tsmixer", {**common, "target_indices": [0]}),
                    ("G6_price_mixer", "tsmixer", {**common, "target_indices": [0], "future_price": True})]
    summary.setdefault("neural_recipes", {})
    for span, horizons in (("short", C.HORIZONS_S), ("long", C.HORIZONS_L)):
        cases = pools[span]
        for name, kind, cfg in configs:
            summary["neural_recipes"][name] = {"kind": kind, "config": cfg}
            _generator1_neural_candidate(args, ctx, summary, prepared, specs, cases, span,
                                         horizons, name, kind, cfg, False)
        if "G7" in stages:
            ranking = []
            for name, recipe in summary["neural_recipes"].items():
                matches = [r for n, r in summary["candidates"][span].items()
                           if r.get("recipe", {}).get("model_stage") == name]
                if matches:
                    best = max(matches, key=lambda r: np.mean([x["mean_gain_pp"] for x in r["tests"].values()]))
                    ranking.append((np.mean([x["mean_gain_pp"] for x in best["tests"].values()]), name, recipe))
            for _, name, recipe in sorted(ranking, key=lambda x: x[0], reverse=True)[:2]:
                _generator1_neural_candidate(args, ctx, summary, prepared, specs, cases, span,
                    horizons, name + "_3seeds", recipe["kind"], recipe["config"], True, name)
    summary["completed_stages"] = sorted(set(summary["completed_stages"]) | (stages & {"G5", "G6"}))
    if "G7" in stages:
        _generator1_availability_seeds(args, ctx, summary, prepared, specs, pools)
        summary["selection"] = {}
        for span in pools:
            accepted = [(n, r) for n, r in summary["candidates"][span].items() if r.get("accepted")]
            chosen = max(accepted, key=lambda item: np.mean([x["mean_gain_pp"] for x in item[1]["tests"].values()]), default=None)
            if chosen and chosen[1].get("recipe", {}).get("seeds", 1) == 1:
                robust = [(n, r) for n, r in accepted if r.get("recipe", {}).get("seeds") == 3 and
                    r["recipe"].get("seed_reference") == chosen[0]]
                if robust:
                    trial = max(robust, key=lambda item: np.mean([x["mean_gain_pp"] for x in item[1]["tests"].values()]))
                    if np.mean([x["mean_gain_pp"] for x in trial[1]["tests"].values()]) >= np.mean(
                            [x["mean_gain_pp"] for x in chosen[1]["tests"].values()]) - .2:
                        chosen = trial
            summary["selection"][span] = {"chosen": chosen[0] if chosen else "baseline",
                "accepted": chosen is not None, "reason": "all_scenario_guards_then_equal_scenario_gain"}
            if chosen:
                intervals = {}
                for scenario in ("normal", "all_missing", "outage_day7"):
                    cases = [x for x in pools[span] if x["scenario"] == scenario]
                    predictions = []
                    for case in cases:
                        with np.load(ctx.model_dir / span / chosen[0] / scenario / case["spec"]["name"] / "predictions.npz") as data:
                            pred = data["prediction"].copy()
                        # 描述性区间仅针对专项目标，保持现有成对块抽样实现。
                        predictions.append(np.repeat(pred[..., :1], 2, axis=-1))
                    intervals[scenario] = _availability_interval(predictions,
                        [np.repeat(c["baseline"][..., :1], 2, axis=-1) for c in cases],
                        [np.repeat(c["truth"][..., :1], 2, axis=-1) for c in cases])
                summary["selection"][span]["descriptive_intervals"] = intervals
        summary["completed_stages"] = sorted(set(summary["completed_stages"]) | {"G7"})
    output.write_report(ctx, summary)


def _generator1_availability_seeds(args, ctx, summary, prepared, specs, pools):
    for span, horizons in (("short", C.HORIZONS_S), ("long", C.HORIZONS_L)):
        eligible = [(n, r) for n, r in summary["candidates"][span].items() if r.get("accepted")
                    and r.get("recipe", {}).get("kind") == "availability_expert"
                    and r["recipe"].get("seeds", 1) == 1]
        selected = max(eligible, key=lambda item: np.mean([x["mean_gain_pp"] for x in item[1]["tests"].values()]), default=None)
        if not selected:
            continue
        original, result = selected
        cases, recipe = pools[span], result["recipe"]
        prior = json.loads((Path(summary["reference"]) / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
        blend = prior["promotion"][span]["R2"]["blend"]["generator_1"]
        seed_predictions = [[c["expert"][..., 0] for c in cases]]
        audits = []
        for seed in (20261022, 20261023):
            predictions = []
            for spec in specs:
                source = Path(summary["reference"]) / C.MODEL_DIR / span / "R2" / spec["name"]
                meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
                directory = ctx.model_dir / span / "G7_availability" / str(seed) / spec["name"]
                directory.mkdir(parents=True, exist_ok=True)
                saved = directory / "meta.json"
                if saved.exists():
                    saved_meta = json.loads(saved.read_text(encoding="utf-8"))
                    models = _study_load_models(directory, saved_meta["files"])
                    names = saved_meta["feature_names"]
                else:
                    training = prepared.prefix_copy(spec["train_end"])
                    ref = len(training.index) - 1
                    stride = 1 if span == "short" else 2
                    positions = np.arange(C.PROFILE_DAYS * C.STEPS_PER_DAY, ref - max(horizons) + 1, stride)
                    LOGGER.info("generator_1保留专家种子复核 %s/%s/%s", span, seed, spec["name"])
                    models, names = main_s._train_models(training, main_s.build_base_features(training, True),
                        positions, horizons, meta["kind"], meta["features"], C.LGBM_TREES_LONG, ref,
                        targets=("generator_1",), params_override={"seed": seed})
                    files = _study_save_models(directory, models)
                    saved_meta = {"files": files, "feature_names": names, "seed": seed,
                        "train_end": spec["train_end"], "max_training_target": str(training.index[positions[-1] + max(horizons)])}
                    saved.write_text(json.dumps(saved_meta, ensure_ascii=False, indent=2), encoding="utf-8")
                audits.append({k: saved_meta[k] for k in ("seed", "train_end", "max_training_target")})
                for case in [c for c in cases if c["spec"]["name"] == spec["name"]]:
                    raw = main_s._predict_models(case["view"], main_s.build_base_features(case["view"], True),
                        case["positions"], horizons, meta["kind"], meta["features"], models, names,
                        targets=("generator_1",))[..., 0]
                    raw = np.clip(raw, 0., 200.)
                    profile = main_s.target_slot_profile(case["view"], case["positions"], horizons, "generator_1")
                    for group, lo, hi in main_s._groups_for(horizons):
                        js = [j for j, h in enumerate(horizons) if lo <= h <= hi]
                        raw[:, js] = blend[group] * raw[:, js] + (1. - blend[group]) * profile[:, js]
                    predictions.append(raw)
            seed_predictions.append(predictions)
        ensembles = [np.mean([seed[i] for seed in seed_predictions], axis=0) for i in range(len(cases))]
        seed_checks = []
        for values in seed_predictions + [ensembles]:
            predictions = []
            for case, expert in zip(cases, values):
                delta = expert - case["baseline"][..., 0]
                weight = recipe["weight"]
                if recipe.get("disagreement_margin_mw"):
                    margin = recipe["disagreement_margin_mw"]
                    weight = weight * np.clip((np.abs(delta) - margin) / margin, 0., 1.)
                predictions.append(main_s.generator1_project(case["baseline"][..., 0] +
                    weight * case["missing"][:, None] * delta, case["baseline"]))
            seed_checks.append(_generator1_evaluate(cases, predictions, horizons))
        _generator1_record(ctx, summary, span, original + "_3seeds", cases, predictions, horizons,
            {**recipe, "seeds": 3, "seed_reference": original, "training": audits,
             "seed_checks": [{"protected": r["protected"], "tests": r["tests"]} for r in seed_checks]})


def _generator1_neural_candidate(args, ctx, summary, prepared, specs, cases, span,
                                 horizons, name, kind, cfg, ensemble, original=None):
    import main_nn
    raw_predictions, audits = [], []
    for spec in specs:
        directory = ctx.model_dir / span / name / spec["name"]
        path = directory / f"nn_{kind}.pt"
        if path.exists():
            bundle = main_nn.load_nn(path)
        else:
            training = prepared.prefix_copy(spec["train_end"])
            ref = len(training.index) - 1
            if cfg.get("power_mask_training", True):
                for offset in range(7 * C.STEPS_PER_DAY, ref, 21 * C.STEPS_PER_DAY):
                    training = training.masked_power_gas_copy(training.index[offset],
                        training.index[min(offset + 14 * C.STEPS_PER_DAY - 1, ref)])
            LOGGER.info("generator_1 NN %s/%s/%s开始", span, name, spec["name"])
            bundle = main_nn.train_nn(training, [spec], kind, horizons, quick=args.quick,
                n_seeds=2 if ensemble else 1, cfg={**cfg, "seed_base": 20261022 if ensemble else 20261021})
            if ensemble:
                first = main_nn.load_nn(ctx.model_dir / span / original / spec["name"] / f"nn_{kind}.pt")
                if first.feature_names != bundle.feature_names or not np.array_equal(first.feat_mean, bundle.feat_mean):
                    raise ValueError("多种子复用的预处理不一致")
                bundle.seed_states = first.seed_states + bundle.seed_states
                bundle.state_dict = bundle.seed_states[0]
                bundle.scores["early_stop"]["epochs"] = first.scores["early_stop"]["epochs"] + bundle.scores["early_stop"]["epochs"]
                bundle.scores["early_stop"]["loss"] = first.scores["early_stop"]["loss"] + bundle.scores["early_stop"]["loss"]
                bundle.scores["seconds"] = first.scores.get("seconds", 0.) + bundle.scores.get("seconds", 0.)
                bundle.config.update(seed_base=20261021, n_seeds=3)
            main_nn.save_nn(bundle, directory)
        audits.append({"fold": spec["name"], "train_end": bundle.train_end,
            "seeds": len(bundle.seed_states), "epochs": bundle.scores["early_stop"]["epochs"],
            "seconds": bundle.scores.get("seconds"), "peak_memory_mb": bundle.scores.get("peak_memory_mb")})
        loaded = main_nn.load_nn(path)
        for case in [x for x in cases if x["spec"]["name"] == spec["name"]]:
            raw = main_nn.predict_nn(bundle, case["view"], case["view"].index[case["positions"]])[..., 0]
            probe = case["positions"][::251]
            check = main_nn.predict_nn(loaded, case["view"], case["view"].index[probe])[..., 0]
            if not np.allclose(raw[::251], check, atol=.002, rtol=0.):
                raise ValueError("generator_1 NN重载预测差异超过0.002MW")
            raw_predictions.append(raw)
        del bundle, loaded
        main_nn.clear_cache()
    for weight in (.25, .5, 1.):
        predictions = [main_s.generator1_project(case["baseline"][..., 0] + weight * case["missing"][:, None] *
            (raw - case["baseline"][..., 0]), case["baseline"]) for case, raw in zip(cases, raw_predictions)]
        _generator1_record(ctx, summary, span, f"{name}_missing_w{weight:g}", cases, predictions, horizons,
            {"kind": "nn", "model_stage": name, "weight": weight, "missing_only": True,
             "nn_kind": kind, "config": cfg, "seeds": 3 if ensemble else 1,
             "training": audits, "reload_passed": True, "reload_tolerance_mw": .002})
        if summary.get("contract", {}).get("normal_path_is_independently_evaluated"):
            predictions = [main_s.generator1_project(case["baseline"][..., 0] + weight * (~case["missing"])[:, None] *
                (raw - case["baseline"][..., 0]), case["baseline"]) for case, raw in zip(cases, raw_predictions)]
            _generator1_record(ctx, summary, span, f"{name}_normal_w{weight:g}", cases, predictions, horizons,
                {"kind": "nn", "model_stage": name, "weight": weight, "normal_only": True,
                 "nn_kind": kind, "config": cfg, "seeds": 3 if ensemble else 1,
                 "training": audits, "reload_passed": True, "reload_tolerance_mw": .002})
    predictions = [main_s.generator1_project(raw, case["baseline"]) for case, raw in zip(cases, raw_predictions)]
    _generator1_record(ctx, summary, span, f"{name}_all", cases, predictions, horizons,
        {"kind": "nn", "model_stage": name, "weight": 1., "missing_only": False,
         "nn_kind": kind, "config": cfg, "seeds": 3 if ensemble else 1,
         "training": audits, "reload_passed": True, "reload_tolerance_mw": .002})


def _verify_generator1_study(directory, summary):
    import lightgbm as lgb
    from types import SimpleNamespace
    problems, archives = [], 0
    specs = [f for f in C.FOLDS if f["role"] == "tuning"]
    prepared = prepare_data(C, cutoff=max(pd.Timestamp(f["start"]) + pd.Timedelta(days=f["days"] + 1) for f in specs))
    diagnostics = {}
    if summary.get("quick") or not set(f"G{i}" for i in range(8)).issubset(summary.get("completed_stages", [])):
        problems.append("专项尚未完成全部完整阶段")
    for span, horizons in (("short", C.HORIZONS_S), ("long", C.HORIZONS_L)):
        chosen = summary.get("selection", {}).get(span, {}).get("chosen", "baseline")
        diagnostics[span] = []
        for spec in specs:
            prefix = prepared.labels.loc[:spec["train_end"], "generator_1"].dropna()
            low = float(prefix[prefix > 0.].quantile(.25))
            state = None
            if chosen != "baseline":
                recipe = summary["candidates"][span][chosen]["recipe"]
                state = {"recipe": recipe, "train_end": spec["train_end"]}
                if recipe["kind"] == "tree":
                    folder = directory / C.MODEL_DIR / span / recipe["model_stage"] / spec["name"]
                    state["tree"] = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
                    state["tree"]["model"] = lgb.Booster(model_str=(folder / "model.txt").read_text(encoding="utf-8"))
                elif recipe["kind"] == "nn":
                    import main_nn
                    state["nn"] = main_nn.load_nn(directory / C.MODEL_DIR / span / recipe["model_stage"] / spec["name"] / f"nn_{recipe['nn_kind']}.pt")
                else:
                    source = Path(summary["reference"]) / C.MODEL_DIR / span / "R2" / spec["name"]
                    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
                    prior = json.loads((Path(summary["reference"]) / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
                    experts = [{"models": _study_load_models(source, meta["files"]), "feature_names": meta["feature_names"]}]
                    for seed in range(1, int(recipe.get("seeds", 1))):
                        folder = directory / C.MODEL_DIR / span / "G7_availability" / str(20261021 + seed) / spec["name"]
                        other = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
                        experts.append({"models": _study_load_models(folder, other["files"]), "feature_names": other["feature_names"]})
                    state.update(experts=experts, expert_kind=meta["kind"], expert_features=meta["features"],
                        expert_blend=prior["promotion"][span]["R2"]["blend"])
                    if recipe["kind"] == "nested_gate":
                        state["gate"] = lgb.Booster(model_str=(directory / C.MODEL_DIR / span / "G2_gate" / spec["name"] / "gate.txt").read_text(encoding="utf-8"))
            for scenario in ("normal", "all_missing", "outage_day7"):
                folder = directory / C.MODEL_DIR / span
                with np.load(folder / "baseline" / scenario / spec["name"] / "predictions.npz") as data:
                    baseline, truth, positions = [data[k].copy() for k in ("prediction", "truth", "positions")]
                with np.load(folder / chosen / scenario / spec["name"] / "predictions.npz") as data:
                    pred = data["prediction"].copy()
                    if not np.array_equal(data["truth"], truth, equal_nan=True) or not np.array_equal(data["positions"], positions):
                        problems.append(f"{span}/{scenario}/{spec['name']}标签或起报点不一致")
                check = output.validate_cube(pred, len(horizons), 21 * 96)
                if not check["valid"] or not np.array_equal(pred[..., 1], baseline[..., 1]):
                    problems.append(f"{span}/{scenario}/{spec['name']}结构或总负荷保护失败")
                reload_diff = 0.
                if state:
                    view, _, _, _ = _study_view(prepared, spec)
                    if scenario != "normal":
                        view = view.masked_power_gas_copy(pd.Timestamp(spec["start"]) + pd.Timedelta(days=7 if scenario == "outage_day7" else 0))
                    reloaded = main_s.apply_generator1_specialist(SimpleNamespace(
                        final_generator1=state, horizons=horizons), view, positions, baseline, True)
                    reload_diff = float(np.max(np.abs(pred - reloaded)))
                    if reload_diff > (.002 if recipe["kind"] == "nn" else 1e-8):
                        problems.append(f"{span}/{scenario}/{spec['name']}实际专项入口未复现归档预测：{reload_diff}")
                target, forecast = truth[..., 0], pred[..., 0]
                positive = np.isfinite(target) & (target > 0.)
                zero = np.isfinite(target) & (target == 0.)
                low_mask = positive & (target <= low)
                diagnostics[span].append({"fold": spec["name"], "scenario": scenario,
                    "production_path_max_abs_diff_mw": reload_diff,
                    "low_threshold_from_training_mw": low, "low_cells": int(low_mask.sum()),
                    "low_accuracy": float(100. * (1. - np.mean(np.abs(forecast[low_mask] - target[low_mask]) / target[low_mask]))) if low_mask.any() else None,
                    "zero_cells": int(zero.sum()), "zero_mae": float(np.abs(forecast[zero]).mean()) if zero.any() else None,
                    "daily_accuracy": [main_s.score_cube(pred[k:k + 96], truth[k:k + 96], horizons)["targets"]["generator_1"]["accuracy"] for k in range(0, len(pred), 96)]})
                archives += 1
        for name, result in summary["candidates"][span].items():
            recipe = result.get("recipe", {})
            for row in recipe.get("training", []):
                if row.get("max_training_target") and pd.Timestamp(row["max_training_target"]) > pd.Timestamp(row["train_end"]):
                    problems.append(f"{span}/{name}训练目标越界")
            for row in recipe.get("provenance", []):
                for inner in row["inner"]:
                    if pd.Timestamp(inner["max_gate_target"]) > pd.Timestamp(row["outer_train_end"]):
                        problems.append(f"{span}/{name}门控标签越界")
    from data_cleaning.quality import _generator1_contract_check
    contract = _generator1_contract_check()
    if not contract["passed"]:
        problems.append("专项因果及投影自检失败")
    summary["verification"] = {"passed": not problems, "archives": archives,
        "problems": problems, "generator1_contract": contract,
        "official_test_used_for_selection": False, "independent_blind_test": False}
    summary["selected_regime_diagnostics"] = diagnostics
    (directory / C.REPORT_DIR / C.REPORT_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["verification"], ensure_ascii=False, indent=2))
    return 0 if not problems else 1


def _generator1_study_recipes(study, summary):
    """从已核验专项还原冻结配方，避免当前配置覆盖历史对照。"""
    import copy
    prior = json.loads((Path(summary["reference"]) / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
    recipes = {}
    for span, selected in summary["selection"].items():
        if not selected["accepted"]:
            continue
        result = summary["candidates"][span][selected["chosen"]]
        recipe = {k: copy.deepcopy(v) for k, v in result["recipe"].items() if k not in
                  ("training", "provenance", "inner_reference", "reload_passed", "reload_tolerance_mw", "seed_checks")}
        if recipe["kind"] in ("availability_expert", "nested_gate"):
            meta = json.loads((Path(summary["reference"]) / C.MODEL_DIR / span / "R2" / "A" / "meta.json").read_text(encoding="utf-8"))
            recipe.update(expert_kind=meta["kind"], expert_features=meta["features"],
                expert_blend=copy.deepcopy(prior["promotion"][span]["R2"]["blend"]))
        recipe.update(source_study=study.name, candidate=selected["chosen"], development=copy.deepcopy(result["tests"]))
        recipes[span] = recipe
    return recipes


def _generator1_refinement_recipes(summary):
    import copy
    recipes = copy.deepcopy(summary["base_recipes"])
    for span, selection in summary["selection"].items():
        if not selection["accepted"]:
            continue
        candidate = summary["candidates"][span][selection["chosen"]]
        update = candidate["recipe"]
        recipe = recipes[span]
        if update.get("base_aggregation"):
            if recipe["kind"] != "nn":
                raise ValueError("仅独立网络支持组合种子聚合精修")
            aggregation = update["base_aggregation"]
            recipe["config"].update(ensemble_aggregation=aggregation["aggregation"], aggregation_power=aggregation["power"])
        if update["kind"] == "primary_aggregation" and recipe["kind"] == "nn":
            recipe["config"].update(ensemble_aggregation=update["aggregation"], aggregation_power=update["power"])
        elif update["kind"] == "primary_tree_gate" and recipe["kind"] == "availability_expert":
            recipe.update(expert_aggregation=update["expert_aggregation"], disagreement_margin_mw=update["disagreement_margin_mw"])
        elif update["kind"] == "nn" and update.get("seeds") == 3 and update["weight"] == 1. and not update.get("normal_only"):
            recipe = {k: v for k, v in update.items() if k not in ("training", "reload_passed", "reload_tolerance_mw")}
            recipes[span] = recipe
        elif update["kind"] == "nn" and update.get("seeds") == 3:
            if recipe["kind"] == "refined_nn":
                raise ValueError("精修模型须先整合，不能无限嵌套")
            base = recipe
            mode = "normal" if update.get("normal_only") else "missing" if update.get("missing_only") else "all"
            recipe = {"kind": "refined_nn", "base_recipe": base,
                "refinement": {"nn_kind": update["nn_kind"], "config": update["config"],
                    "seeds": update["seeds"], "model_stage": update["model_stage"],
                    "weight": update["weight"], "mode": mode}}
            recipes[span] = recipe
        else:
            raise ValueError(f"{span}精修候选尚未完成相应的多种子或部署算子核验")
        recipe.update(previous_candidate=summary["base_recipes"][span]["candidate"],
            previous_source_study=summary["base_recipes"][span]["source_study"],
            source_study=summary["run_id"], candidate=selection["chosen"], development=candidate["tests"],
            comparison="previous_promoted_generator1_strategy")
    return recipes


def _generator1_reference_fold_state(reference, prior, span, spec, recipe):
    import main_nn
    state = {"recipe": recipe, "train_end": spec["train_end"]}
    if recipe["kind"] == "nn":
        state["nn"] = main_nn.load_nn(reference / C.MODEL_DIR / span / recipe["model_stage"] / spec["name"] / f"nn_{recipe['nn_kind']}.pt")
        if state["nn"].train_end != spec["train_end"]:
            raise ValueError("精修引用网络的训练截止不一致")
        state["nn"].config.update(recipe["config"])
    elif recipe["kind"] == "availability_expert":
        source = Path(prior["reference"]) / C.MODEL_DIR / span / "R2" / spec["name"]
        meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
        experts = [{"models": _study_load_models(source, meta["files"]), "feature_names": meta["feature_names"]}]
        for seed in range(1, recipe["seeds"]):
            folder = reference / C.MODEL_DIR / span / "G7_availability" / str(20261021 + seed) / spec["name"]
            other = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            experts.append({"models": _study_load_models(folder, other["files"]), "feature_names": other["feature_names"]})
        state.update(experts=experts, expert_kind=recipe["expert_kind"],
            expert_features=recipe["expert_features"], expert_blend=recipe["expert_blend"])
    else:
        raise ValueError("精修参考分支必须先整合为独立网络或树专家")
    return state


def _verify_generator1_refinement(directory, summary):
    import main_nn
    from types import SimpleNamespace
    summary.setdefault("run_id", directory.name)
    recipes = _generator1_refinement_recipes(summary)
    problems, checks = [], []
    if summary["quick"]:
        print(json.dumps({"passed": False, "problems": ["quick精修不能晋级"]}, ensure_ascii=False))
        return 1
    reference = Path(summary["reference"])
    prior = json.loads((reference / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
    specs = [dict(f) for f in C.FOLDS if f["role"] == "tuning"]
    prepared = prepare_data(C, cutoff=max(pd.Timestamp(s["start"]) + pd.Timedelta(days=s["days"] + 1) for s in specs))
    for span, horizons in (("short", C.HORIZONS_S), ("long", C.HORIZONS_L)):
        chosen = summary["selection"][span]["chosen"]
        predictions, baselines, truths, cases = [], [], [], []
        for spec in specs:
            state = None
            if chosen != "baseline":
                recipe = recipes[span]
                if recipe["kind"] == "refined_nn":
                    update = recipe["refinement"]
                    state = {"recipe": recipe, "train_end": spec["train_end"],
                        "base_state": _generator1_reference_fold_state(reference, prior, span, spec, recipe["base_recipe"]),
                        "nn": main_nn.load_nn(directory / C.MODEL_DIR / span / update["model_stage"] / spec["name"] / f"nn_{update['nn_kind']}.pt")}
                else:
                    model_root = directory if summary["candidates"][span][chosen]["recipe"]["kind"] == "nn" else reference
                    state = _generator1_reference_fold_state(model_root, prior, span, spec, recipe)
            for scenario in ("normal", "all_missing", "outage_day7"):
                folder = directory / C.MODEL_DIR / span
                with np.load(folder / "baseline" / scenario / spec["name"] / "predictions.npz") as data:
                    before, truth, positions = [data[k].copy() for k in ("prediction", "truth", "positions")]
                with np.load(folder / chosen / scenario / spec["name"] / "predictions.npz") as data:
                    pred = data["prediction"].copy()
                    if not np.array_equal(data["truth"], truth, equal_nan=True) or not np.array_equal(data["positions"], positions):
                        problems.append(f"{span}/{scenario}/{spec['name']}索引或真值不一致")
                if not output.validate_cube(pred, len(horizons), 21 * 96)["valid"] or not np.array_equal(pred[..., 1], before[..., 1]):
                    problems.append(f"{span}/{scenario}/{spec['name']}总负荷或结构校验失败")
                diff = 0.
                if state:
                    view, _, _, _ = _study_view(prepared, spec)
                    if scenario != "normal":
                        view = view.masked_power_gas_copy(pd.Timestamp(spec["start"]) + pd.Timedelta(days=7 if scenario == "outage_day7" else 0))
                    with np.load(reference / C.MODEL_DIR / span / "baseline" / scenario / spec["name"] / "predictions.npz") as data:
                        original = data["prediction"].copy()
                    actual = main_s.apply_generator1_specialist(SimpleNamespace(final_generator1=state, horizons=horizons),
                        view, positions, original, True)
                    diff = float(np.max(np.abs(actual - pred)))
                    if diff > .002:
                        problems.append(f"{span}/{scenario}/{spec['name']}实际入口复现差异{diff}")
                checks.append({"span": span, "fold": spec["name"], "scenario": scenario, "max_abs_diff_mw": diff})
                cases.append({"spec": spec, "scenario": scenario, "baseline": before, "truth": truth})
                predictions.append(pred)
                baselines.append(before)
                truths.append(truth)
            if state:
                del state
                main_nn.clear_cache()
        rescored = _generator1_evaluate(cases, predictions, horizons)
        if chosen != "baseline" and not rescored["accepted"]:
            problems.append(f"{span}精修重新评分未过门槛")
        summary["selection"][span]["descriptive_intervals"] = {}
        for scenario in ("normal", "all_missing", "outage_day7"):
            indices = [i for i, c in enumerate(cases) if c["scenario"] == scenario]
            summary["selection"][span]["descriptive_intervals"][scenario] = _availability_interval(
                [np.repeat(predictions[i][..., :1], 2, axis=-1) for i in indices],
                [np.repeat(baselines[i][..., :1], 2, axis=-1) for i in indices],
                [np.repeat(truths[i][..., :1], 2, axis=-1) for i in indices])
    summary["verification"] = {"passed": not problems, "archives": len(checks), "production_paths": checks,
        "problems": problems, "official_test_used_for_selection": False, "independent_blind_test": False}
    summary["frozen_recipes"] = recipes
    (directory / C.REPORT_DIR / C.REPORT_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["verification"], ensure_ascii=False, indent=2))
    return 0 if not problems else 1


def _train_generator1_final(training, short_bundle, long_bundle, study=None, quick=False, recipes_override=None):
    import main_nn
    import copy
    recipes = copy.deepcopy(C.GENERATOR1_SPECIALIST if recipes_override is None else recipes_override)
    if study is not None:
        study = study.resolve()
        if not study.is_relative_to(C.EXPERIMENT_OUTPUT_ROOT.resolve()):
            raise ValueError("generator_1冻结专项必须来自实验目录")
        summary = json.loads((study / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
        refinement = summary.get("status") == "SUCCESS_GENERATOR1_REFINEMENT"
        if summary.get("status") not in ("SUCCESS_GENERATOR1_STUDY", "SUCCESS_GENERATOR1_REFINEMENT") or not summary.get("verification", {}).get("passed"):
            raise ValueError("generator_1最终训练要求已核验的全量专项")
        recipes = _generator1_refinement_recipes(summary) if refinement else _generator1_study_recipes(study, summary)
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
        if recipe["kind"] == "tree":
            state["tree"] = main_s.generator1_fit(training, ref, bundle.horizons,
                residual="residual" in recipe["model_stage"], blocked=recipe["model_stage"].startswith("G4"), quick=quick)
        elif recipe["kind"] == "nn":
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
            if recipe["kind"] == "nested_gate":
                if study is None:
                    study = C.EXPERIMENT_OUTPUT_ROOT / recipe["source_study"]
                cache = output.RunContext(study.name, study, study / C.SOURCE_DIR,
                    study / C.MODEL_DIR / "final_refit", study / C.REPORT_DIR,
                    study / C.REPORT_DIR / C.REPORT_LOG, "generator1_final_fit")
                state["gate"], state["gate_provenance"] = _generator1_inner_gate(
                    argparse.Namespace(quick=quick), cache, training,
                    {"name": "final", "start": str(training.index[-1] + pd.Timedelta(minutes=15)),
                     "train_end": end}, span, bundle.horizons)
        bundle.final_generator1 = state
        bundle.selection["generator1"] = {"candidate": recipe["candidate"], "source_study": recipe["source_study"]}
        result[span] = {"recipe": recipe, "train_end": end, "seconds": time.time() - tick,
            "reused_expert_seeds": state.get("reused_expert_seeds", []), "D_reassessed": False}
    return result


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
    parser.add_argument("--availability-study-run", type=Path, help="infer按完整可用性专项冻结配方重训final，不重评D")
    parser.add_argument("--generator1-study-run", type=Path, help="infer按完整generator1专项冻结配方重训专项final")
    parser.add_argument("--plan-study", help="方案开发专项：baseline、pressure、availability、observed、generator1、generator1-refine 或 E1～E6")
    parser.add_argument("--g1-stages", default="G0,G1,G2,G3,G4,G5,G6,G7", help="generator1专项阶段，逗号分隔，可续跑")
    parser.add_argument("--g1-refine-stages", default="reuse,train,normal,seeds,combine", help="generator1-refine阶段：reuse、train、normal、seeds、combine")
    parser.add_argument("--study-reference", type=Path, help="原始运行或已生成的方案开发基线")
    parser.add_argument("--study-resume", type=Path, help="续跑已中断的资源方案专项，核验后复用已完成数组")
    args = parser.parse_args(argv)
    C.PROGRESS_ENABLED = args.progress
    if args.study_resume and not args.plan_study:
        parser.error("--study-resume 必须搭配 --plan-study")
    if args.plan_study:
        if args.command != "run" or not args.experiment or args.study_reference is None:
            parser.error("--plan-study 必须搭配 run --experiment --study-reference")
        if args.plan_study == "generator1-refine":
            return run_generator1_refinement(args)
        return run_generator1_study(args) if args.plan_study == "generator1" else run_plan_study(args)
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
                         reuse_run=args.reuse_run, nn_studies=args.nn_study_run,
                         availability_study=args.availability_study_run, generator1_study=args.generator1_study_run)
    if args.command == "self-check":
        return run_self_check()
    if args.command == "verify":
        return run_verify(args.run_dir)
    return run_full(quick=args.quick, experimental=args.experiment)


if __name__ == "__main__":
    sys.exit(main())
