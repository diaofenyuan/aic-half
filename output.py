"""输出契约：三份 CSV、模型归档、源码快照、模拟/正式 ZIP 与校验。

本模块不训练、不选模；所有结果只由已冻结的数组和模型落盘。
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C


@dataclass
class RunContext:
    run_id: str
    root: Path
    source_dir: Path
    model_dir: Path
    report_dir: Path
    log_path: Path
    mode: str


def _run_id(mode: str) -> str:
    return f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_{mode.lower()}"


def begin_run(config=C, mode: str = "backtest", experimental: bool = False) -> RunContext:
    run_id = _run_id(mode)
    # 实验复用完整归档链路，但整个目录必须隔离，避免干扰选定运行结果。
    base = config.EXPERIMENT_OUTPUT_ROOT if experimental else config.OUTPUT_ROOT
    root = base / run_id
    source = ((config.ROOT / "tmp_main" / run_id / config.SOURCE_DIR)
              if experimental else root / config.SOURCE_DIR)
    models = root / config.MODEL_DIR
    report = root / config.REPORT_DIR
    for path in (root, source, models, report):
        path.mkdir(parents=True, exist_ok=True)
    return RunContext(run_id=run_id, root=root, source_dir=source,
                      model_dir=models, report_dir=report,
                      log_path=report / config.REPORT_LOG, mode=mode)


def _short_columns() -> list[str]:
    return ["datetime", *[f"{t}_t+{15 * h}_pred" for t in C.TARGETS for h in C.HORIZONS_S]]


def _long_columns() -> list[str]:
    return ["datetime", *[f"{t}_t+{15 * h}_pred" for t in C.TARGETS for h in C.HORIZONS_L]]


def _cube_to_frame(cube: np.ndarray, grid: pd.DatetimeIndex,
                   horizons: tuple[int, ...]) -> pd.DataFrame:
    payload = {"datetime": pd.DatetimeIndex(grid).strftime("%Y-%m-%d %H:%M:%S")}
    for j, target in enumerate(C.TARGETS):
        for h in horizons:
            payload[f"{target}_t+{15 * h}_pred"] = cube[:, h - 1, j]
    return pd.DataFrame(payload)


def validate_cube(cube: np.ndarray, expected_h: int, expected_n: int | None = None) -> dict:
    problems = []
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3 or cube.shape[1] != expected_h or cube.shape[2] != 2:
        problems.append(f"预测形状错误: {cube.shape}")
        return {"valid": False, "problems": problems}
    if expected_n is not None and cube.shape[0] != expected_n:
        problems.append(f"起报点数量错误: {cube.shape[0]} != {expected_n}")
    if not np.isfinite(cube).all():
        problems.append("预测存在 NaN/Inf")
    if (cube < 0).any():
        problems.append("预测存在负值")
    g1 = cube[..., 0]
    allp = cube[..., 1]
    if (g1 > C.CAPACITY["generator_1"] + 1e-6).any():
        problems.append("generator_1 超出 200MW")
    if (allp > C.CAPACITY["generator_all"] + 1e-6).any():
        problems.append("generator_all 超出 440MW")
    if (allp + 1e-6 < g1).any():
        problems.append("generator_all 小于 generator_1")
    return {"valid": not problems, "problems": problems,
            "shape": list(cube.shape),
            "generator_1_range": [float(np.nanmin(g1)), float(np.nanmax(g1))],
            "generator_all_range": [float(np.nanmin(allp)), float(np.nanmax(allp))]}


def official_columns(kind: str) -> list[str]:
    """官方提交宽表列名；kind 取 s / l / opt。"""
    if kind == "s":
        return ["datetime", *[f"{t}_t+{15 * h}_pred"
                              for t in C.TARGETS for h in C.HORIZONS_S]]
    if kind == "l":
        return ["datetime", *[f"{t}_t+{15 * h}_pred"
                              for t in C.TARGETS for h in C.HORIZONS_L]]
    if kind == "opt":
        return ["datetime",
                "opt_generator_use_blast_furnace_gas",
                "opt_generator_use_coke_gas",
                "opt_generator_use_converter_gas",
                "opt_generator_1", "opt_generator_all"]
    raise ValueError(f"未知提交类型: {kind}")


def validate_official_frame(name: str, frame: pd.DataFrame) -> dict:
    """按赛题字段契约校验预测/优化宽表，问题不静默放过。"""
    kind = {"s_result.csv": "s", "l_result.csv": "l",
            "opt_result.csv": "opt"}[name]
    expected = official_columns(kind)
    problems = []
    if list(frame.columns) != expected:
        problems.append(f"{name} 列名与官方契约不一致")
    if "datetime" in frame.columns:
        times = pd.to_datetime(frame["datetime"], errors="coerce")
        if times.isna().any():
            problems.append(f"{name} datetime 无法解析")
        elif times.duplicated().any():
            problems.append(f"{name} datetime 重复")
    numeric = frame.drop(columns=["datetime"], errors="ignore")
    if len(numeric.columns):
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            problems.append(f"{name} 存在非有限值")
        if (values < 0).any():
            problems.append(f"{name} 存在负值")
    return {"passed": not problems, "problems": problems,
            "columns": list(frame.columns), "rows": int(len(frame))}


def write_predictions(ctx: RunContext, grid: pd.DatetimeIndex,
                      short_pred: np.ndarray, long_pred: np.ndarray) -> dict:
    grid = pd.DatetimeIndex(grid)
    short_check = validate_cube(short_pred, len(C.HORIZONS_S), len(grid))
    long_check = validate_cube(long_pred, len(C.HORIZONS_L), len(grid))
    if not short_check["valid"] or not long_check["valid"]:
        raise ValueError("预测校验失败: "
                         + "; ".join(short_check["problems"] + long_check["problems"]))
    short = _cube_to_frame(short_pred, grid, C.HORIZONS_S)[_short_columns()]
    long = _cube_to_frame(long_pred, grid, C.HORIZONS_L)[_long_columns()]
    # 官方契约校验放在落盘前，列名错位不能靠事后读回才发现。
    for name, frame in (("s_result.csv", short), ("l_result.csv", long)):
        check = validate_official_frame(name, frame)
        if not check["passed"]:
            raise ValueError(f"{name} 未通过官方契约: " + "; ".join(check["problems"]))
    short.to_csv(ctx.root / "s_result.csv", index=False, encoding="utf-8", float_format="%.6f")
    long.to_csv(ctx.root / "l_result.csv", index=False, encoding="utf-8", float_format="%.6f")

    for name, frame, columns, check in (("s_result.csv", short, _short_columns(), short_check),
                                        ("l_result.csv", long, _long_columns(), long_check)):
        reread = pd.read_csv(ctx.root / name, encoding="utf-8")
        if list(reread.columns) != columns:
            raise ValueError(f"{name} 落盘后列不一致")
        if len(reread) != len(grid):
            raise ValueError(f"{name} 行数与起报点不一致")
        times = pd.to_datetime(reread["datetime"], errors="coerce")
        if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
            raise ValueError(f"{name} datetime 非法、非递增或重复")
        if not np.isfinite(reread[columns[1:]].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} 落盘后存在非有限值")
        check["rows"] = int(len(reread))
        check["columns"] = len(columns)
    return {"short": short_check, "long": long_check, "origins": int(len(grid)),
            "start": str(grid[0]), "end": str(grid[-1])}


def write_optimization(ctx: RunContext, schedule: pd.DataFrame | None) -> dict:
    if schedule is None:
        return {"written": False, "reason": "NO_OPTIMIZATION_RESULT"}
    frame = schedule.copy()
    required = ["datetime", "opt_generator_use_blast_furnace_gas",
                "opt_generator_use_coke_gas", "opt_generator_use_converter_gas",
                "opt_generator_1", "opt_generator_all"]
    if list(frame.columns)[:len(required)] != required:
        raise ValueError("opt_result 列不符合当前模拟契约")
    frame = frame[required]
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    numeric = frame[required[1:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("opt_result 存在非有限值或负值")
    g1 = frame["opt_generator_1"].to_numpy(dtype=float)
    allp = frame["opt_generator_all"].to_numpy(dtype=float)
    if (g1 > C.CAPACITY["generator_1"] + 1e-6).any():
        raise ValueError("opt_generator_1 超容量")
    if (allp > C.CAPACITY["generator_all"] + 1e-6).any():
        raise ValueError("opt_generator_all 超容量")
    if (allp + 1e-6 < g1).any():
        raise ValueError("opt_generator_all 小于 opt_generator_1")
    check = validate_official_frame("opt_result.csv", frame)
    if not check["passed"]:
        raise ValueError("opt_result 未通过官方契约: " + "; ".join(check["problems"]))
    frame.to_csv(ctx.root / "opt_result.csv", index=False, encoding="utf-8",
                 float_format="%.6f")
    reread = pd.read_csv(ctx.root / "opt_result.csv", encoding="utf-8")
    return {"written": True, "rows": int(len(reread)), "columns": list(reread.columns),
            "generator_1_range": [float(g1.min()), float(g1.max())],
            "generator_all_range": [float(allp.min()), float(allp.max())]}


def _model_tag(key) -> str:
    if isinstance(key, tuple):
        return "__".join(str(part) for part in key)
    return str(key)


def load_final_models(directory: Path):
    """重载本项目已归档的最终模型，避免神经网络专项重复拟合未改变的树与资源模型。"""
    import main_nn
    from main_s import ShortBundle
    from main_l import LongBundle
    from main_opt import ResourceBundle

    directory = directory.resolve()
    if not directory.is_relative_to(C.ROOT.resolve()):
        raise ValueError("只允许复用本项目的本地运行模型")
    model_root = directory / C.MODEL_DIR
    meta = json.loads((model_root / "model_meta.json").read_text(encoding="utf-8"))
    report = json.loads((directory / C.REPORT_DIR / C.REPORT_JSON).read_text(encoding="utf-8"))
    if meta["role"] != "final":
        raise ValueError("推理只能复用 final 角色，禁止把确认模型冒充最终模型")
    bundles = []
    for name, cls, horizons in (("short", ShortBundle, C.HORIZONS_S), ("long", LongBundle, C.HORIZONS_L)):
        item, artifacts = meta[name], meta["artifacts"][name]
        models = {}
        for tag, filename in artifacts.get("final", {}).get("files", {}).items():
            key = tag
            if "__" in tag:
                target, part = tag.rsplit("__", 1)
                key = (target, int(part) if part.isdigit() else part)
            models[key] = lgb.Booster(model_str=(model_root / name / "final" / filename).read_text(encoding="utf-8"))
        neural = main_nn.load_nn(model_root / artifacts["nn"]["path"]) if "nn" in artifacts else None
        bundles.append(cls(kind=item["kind"], features=item["features"], horizons=horizons,
            blend=item["blend"], models={}, final_models=models, feature_names=item["feature_names"],
            nn_bundle=None, final_nn_bundle=neural, lgb_kind=item.get("lgb_kind"),
            nn_weights=item.get("nn_weights", {}),
            lgb_features=item.get("lgb_features"), selection=report[name]["selection"], scores=item["scores"],
            train_end=item["train_end"], final_train_end=item["train_end"]))
        if item.get("availability"):
            state = dict(item["availability"])
            directory = model_root / name / "final_availability"
            state["models"] = {}
            for tag, filename in artifacts["availability"]["files"].items():
                key = (tag.rsplit("__", 1)[0], int(tag.rsplit("__", 1)[1])) if "__" in tag else tag
                state["models"][key] = lgb.Booster(model_str=(directory / filename).read_text(encoding="utf-8"))
            if state.get("retained_state"):
                retained = state["retained_state"]
                retained["models"] = {}
                for tag, filename in artifacts["availability"]["retained"]["files"].items():
                    key = (tag.rsplit("__", 1)[0], int(tag.rsplit("__", 1)[1])) if "__" in tag else tag
                    retained["models"][key] = lgb.Booster(model_str=(directory / "retained" / filename).read_text(encoding="utf-8"))
            bundles[-1].final_availability = state
        if item.get("generator1"):
            from main_s import load_generator1
            bundles[-1].final_generator1 = load_generator1(model_root / item["generator1"]["path"])
    item, artifacts = meta["resource"], meta["artifacts"]["resource"]
    from main_opt import load_resource_model
    models = {tag: load_resource_model((model_root / "resource" / "final" / filename).read_text(encoding="utf-8"))
              for tag, filename in artifacts["final"]["files"].items()}
    neural = main_nn.load_resource_nn(model_root / artifacts["nn"]["path"]) if "nn" in artifacts else None
    resource = ResourceBundle(models={}, final_models=models,
        feature_names=next(iter(models.values())).feature_name(),
        metrics=item["metrics"], heat_rate=item["heat_rate"], final_heat_rate=item["heat_rate"],
        gas_caps=item["gas_caps"], final_gas_caps=item["gas_caps"],
        train_end=item["train_end"], final_train_end=item["train_end"],
        selection=report["resource_selection"], dispatch_selection=item["dispatch_selection"],
        resource_nn=None, final_resource_nn=neural, blocker=item.get("blocker"))
    bundles.append(resource)
    return bundles


def _save_boosters(directory: Path, prefix: str, models: dict,
                   feature_names: list[str], temp_dir: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    reload_ok = True
    probe = np.zeros((3, len(feature_names)), dtype=np.float32)
    for key, booster in C.progress(models.items(), desc=f"保存 {prefix} 模型", unit="模型"):
        tag = _model_tag(key)
        file_name = f"{prefix}_{tag}.txt"
        temp_path = temp_dir / file_name
        booster.save_model(str(temp_path))
        # LightGBM 的 C++ 层在 Windows 下无法直接读写中文路径；
        # 先写 ASCII 临时目录，再用 Python 文件系统移动并以 model_str 复载。
        from main_opt import load_resource_model
        loaded = load_resource_model(temp_path.read_text(encoding="utf-8"))
        same = np.allclose(booster.predict(probe), loaded.predict(probe), rtol=1e-6, atol=1e-6)
        reload_ok &= bool(same)
        shutil.move(str(temp_path), str(directory / file_name))
        index[tag] = file_name
    return {"files": index, "reload_ok": bool(reload_ok)}


def save_models(ctx: RunContext, short_bundle, long_bundle,
                resource_bundle, opt_summary: dict, use_final: bool = False) -> dict:
    """只归档实际用于本次结果的角色，候选与另一角色留在实验流程中。"""
    import main_nn

    role = "final" if use_final else "confirm"
    summary = {"role": role, "short": {}, "long": {}, "resource": {}, "reload_ok": True}
    temp_dir = ctx.root / "_model_tmp"
    for name, bundle in (("short", short_bundle), ("long", long_bundle),
                         ("resource", resource_bundle)):
        models = bundle.final_models if use_final else bundle.models
        prefix = "resource" if name == "resource" else f"{bundle.kind}_{bundle.features}"
        if models:
            info = _save_boosters(ctx.model_dir / name / role, prefix, models,
                                  bundle.feature_names, temp_dir)
            summary[name][role] = info
            summary["reload_ok"] &= info["reload_ok"]
        availability = getattr(bundle, "final_availability", None) if use_final else None
        if availability:
            info = _save_boosters(ctx.model_dir / name / "final_availability", "availability",
                availability["models"], availability["feature_names"], temp_dir)
            summary[name]["availability"] = info
            summary["reload_ok"] &= info["reload_ok"]
            if availability.get("retained_state"):
                retained = availability["retained_state"]
                retained_info = _save_boosters(ctx.model_dir / name / "final_availability" / "retained",
                    "availability", retained["models"], retained["feature_names"], temp_dir)
                info["retained"] = retained_info
                summary["reload_ok"] &= retained_info["reload_ok"]
        nn = getattr(bundle, "final_nn_bundle" if use_final else "nn_bundle", None)
        specialist = getattr(bundle, "final_generator1", None) if use_final else None
        if specialist:
            from main_s import save_generator1, load_generator1
            directory = ctx.model_dir / name / "final_generator1"
            save_generator1(specialist, directory)
            loaded = load_generator1(directory)
            same = (loaded["recipe"] == specialist["recipe"] and loaded["train_end"] == specialist["train_end"])
            summary[name]["generator1"] = {"path": str(directory.relative_to(ctx.model_dir)), "reload_ok": same}
            summary["reload_ok"] &= same
        if nn is not None:
            path = main_nn.save_nn(nn, ctx.model_dir / name / role)
            loaded = main_nn.load_nn(Path(path))
            same = (len(loaded.seed_states) == len(nn.seed_states)
                    and np.array_equal(loaded.feat_mean, nn.feat_mean)
                    and all(np.array_equal(loaded.state_dict[k].numpy(), v.numpy())
                            for k, v in nn.state_dict.items()))
            summary[name]["nn"] = {"path": str(Path(path).relative_to(ctx.model_dir)), "kind": nn.kind, "reload_ok": bool(same)}
            summary["reload_ok"] &= bool(same)
        if name == "resource":
            resource_nn = bundle.final_resource_nn if use_final else bundle.resource_nn
            if resource_nn is not None:
                path = main_nn.save_resource_nn(resource_nn, ctx.model_dir / name / role / "resource_nn.pt")
                loaded = main_nn.load_resource_nn(path)
                same = (loaded["feature_names"] == resource_nn["feature_names"]
                    and all(np.array_equal(loaded[k], resource_nn[k]) for k in ("center", "scale", "gas_scale"))
                    and all(np.array_equal(loaded["state_dict"][k].numpy(), value.cpu().numpy())
                            for k, value in resource_nn["state_dict"].items()))
                summary[name]["nn"] = {"path": str(path.relative_to(ctx.model_dir)),
                    "kind": resource_nn["kind"], "train_end": resource_nn["train_end"],
                    "max_training_target": resource_nn["max_training_target"], "reload_ok": bool(same)}
                summary["reload_ok"] &= bool(same)
    if temp_dir.exists():
        if not temp_dir.resolve().is_relative_to(ctx.root.resolve()):
            raise ValueError("临时模型目录超出当次运行路径")
        shutil.rmtree(temp_dir)
    if not summary["reload_ok"]:
        raise ValueError("实际模型保存后重载不一致")

    def forecast_meta(bundle):
        nn = getattr(bundle, "final_nn_bundle" if use_final else "nn_bundle", None)
        availability = getattr(bundle, "final_availability", None) if use_final else None
        state = {k: v for k, v in availability.items() if k != "models"} if availability else None
        if state and state.get("retained_state"):
            state["retained_state"] = {k: v for k, v in state["retained_state"].items() if k != "models"}
        return {"kind": bundle.kind, "features": bundle.features,
                "availability": state,
                "generator1": summary["short" if len(bundle.horizons) == 8 else "long"].get("generator1"),
                "blend": bundle.blend, "feature_names": bundle.feature_names,
                "lgb_kind": bundle.lgb_kind, "lgb_features": bundle.lgb_features,
                "nn_weights": bundle.nn_weights,
                "scores": bundle.scores,
                "train_end": nn.train_end if nn is not None else
                             (bundle.final_train_end if use_final else bundle.train_end)}

    meta = {
        "role": role, "artifacts": summary,
        "short": forecast_meta(short_bundle), "long": forecast_meta(long_bundle),
        "resource": {"metrics": resource_bundle.metrics,
                     "selection": {k: resource_bundle.selection.get(k)
                                   for k in ("objectives", "selection_scope", "availability_adjustment")},
                     "head_selection": {k: resource_bundle.selection.get("head_selection", {}).get(k)
                                        for k in ("gases", "weight", "horizons", "method")},
                     "dispatch_selection": {k: resource_bundle.dispatch_selection.get(k)
                                            for k in ("chosen", "candidate", "replan_steps",
                                                      "control_options", "scope", "promotion")},
                     "heat_rate": resource_bundle.final_heat_rate if use_final else resource_bundle.heat_rate,
                     "gas_caps": resource_bundle.final_gas_caps if use_final else resource_bundle.gas_caps,
                     "train_end": resource_bundle.final_train_end if use_final else resource_bundle.train_end,
                     "blocker": resource_bundle.blocker},
        "optimization": opt_summary,
        "data_contract": {"target_mask": "整个评测窗口", "step_minutes": C.STEP_MINUTES,
                          "future_dynamic_fields": "forbidden"},
    }
    (ctx.model_dir / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def _top_features(models: dict, names: list[str], top_k: int = 20) -> list[dict]:
    """汇总已保存 booster 的特征重要性；单模型直接取，多模型取均值。"""
    if not models:
        return []
    acc = {n: 0.0 for n in names}
    for booster in models.values():
        # LightGBM 默认按 split 次数统计，与 gain 相比更稳定、可跨折汇总。
        values = booster.feature_importance(importance_type="split")
        if len(values) != len(names):
            return []
        for name, value in zip(names, values):
            acc[name] += float(value)
    ranked = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    total = sum(v for _, v in ranked) or 1.0
    return [{"feature": name, "split_importance": round(v, 2),
             "share": round(v / total, 4)} for name, v in ranked]


def write_calibration(ctx: RunContext, short_bundle, long_bundle,
                      resource_bundle, use_final: bool = False) -> dict:
    """标定产物：模型选择、特征白名单、特征重要性、物理映射，供正式推理直接加载。"""
    directory = ctx.model_dir
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "short": {"kind": short_bundle.kind, "features": short_bundle.features,
                  "blend": short_bundle.blend, "feature_names": short_bundle.feature_names,
                  "top_features": _top_features(short_bundle.final_models if use_final else short_bundle.models,
                                                short_bundle.feature_names)},
        "long": {"kind": long_bundle.kind, "features": long_bundle.features,
                 "blend": long_bundle.blend, "feature_names": long_bundle.feature_names,
                 "top_features": _top_features(long_bundle.final_models if use_final else long_bundle.models,
                                               long_bundle.feature_names)},
        "resource": {"feature_names": resource_bundle.feature_names,
                     "heat_rate": resource_bundle.final_heat_rate if use_final else resource_bundle.heat_rate,
                     "metrics": resource_bundle.metrics,
                     "top_features": _top_features(resource_bundle.final_models if use_final else resource_bundle.models,
                                                   resource_bundle.feature_names)},
        "note": "LightGBM 树模型不做特征标准化；白名单即训练时实际列序。",
    }
    path = directory / "model_meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["calibration"] = payload
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return {"path": str(path), "short_features": len(short_bundle.feature_names),
            "long_features": len(long_bundle.feature_names)}


def write_source_snapshot(ctx: RunContext, config=C) -> dict:
    files = ["main.py", "main_s.py", "main_l.py", "main_nn.py", "main_opt.py", "config.py", "output.py",
             "requirements.txt", "README.md"]
    # 仓库保留 Git 忽略规则；提交源码快照不携带，并清理旧快照的遗留文件。
    (ctx.source_dir / ".gitignore").unlink(missing_ok=True)
    copied = []
    for name in C.progress(files, desc="归档源码", unit="文件"):
        source = config.ROOT / name
        if source.is_file():
            target = ctx.source_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(name)
    cleaning = config.ROOT / "data_cleaning"
    for path in sorted(cleaning.glob("*.py")):
        target = ctx.source_dir / "data_cleaning" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(f"data_cleaning/{path.name}")
    return {"files": copied, "count": len(copied), "directory": str(ctx.source_dir)}


def build_zip(ctx: RunContext, submittable: bool, config=C) -> dict:
    name = config.FORMAL_ZIP_NAME if submittable else config.MOCK_ZIP_NAME
    final_path = ctx.root / name
    temp_path = ctx.root / (name + ".tmp")
    members = ["s_result.csv", "l_result.csv"]
    if (ctx.root / "opt_result.csv").is_file():
        members.append("opt_result.csv")
    paths = [(ctx.root / member, member) for member in members]
    for name, folder in ((config.SOURCE_DIR, ctx.source_dir), (config.MODEL_DIR, ctx.model_dir)):
        paths.extend((path, f"{name}/{path.relative_to(folder).as_posix()}")
                     for path in sorted(folder.rglob("*")) if path.is_file() and path.name != ".gitignore")
    with ZipFile(temp_path, "w", ZIP_DEFLATED) as packed:
        for path, member in C.progress(paths, desc="生成压缩包", unit="文件"):
            packed.write(path, member)
    with ZipFile(temp_path) as packed:
        bad = packed.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC 校验失败: {bad}")
        names = packed.namelist()
        for member in members:
            if member not in names:
                raise ValueError(f"ZIP 缺少 {member}")
            if packed.read(member) != (ctx.root / member).read_bytes():
                raise ValueError(f"ZIP 内 {member} 与磁盘不一致")
    temp_path.replace(final_path)
    return {"name": name, "path": str(final_path), "members": len(names),
            "root_csv": members, "crc_ok": True, "submittable": bool(submittable)}


def write_report(ctx: RunContext, summary: dict) -> Path:
    path = ctx.report_dir / C.REPORT_JSON
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")
    return path
