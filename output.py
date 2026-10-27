"""输出契约：三份 CSV、模型归档、源码快照、模拟/正式 ZIP 与校验。

本模块不训练、不选模；所有结果只由已冻结的数组和模型落盘。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C


_TRAINING_CACHE_STATS = {"hits": {}, "misses": {}}


def training_fingerprint(contract: dict, *, arrays=(), frames=()) -> str:
    """按实际输入和实现生成缓存键，截止点相同但数据或配方不同也不得复用。"""
    digest = hashlib.sha256()
    settings = {k: v for k, v in vars(C).items() if k.isupper() and k != "PROGRESS_ENABLED"}
    digest.update(json.dumps({"schema": 1, "contract": contract, "config": settings,
        "numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lgb.__version__},
        sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    for relative in ("config.py", "main.py", "main_s.py", "main_l.py", "main_nn.py",
                     "main_opt.py", "data_cleaning/pipeline.py"):
        digest.update(relative.encode("utf-8"))
        digest.update((C.ROOT / relative).read_bytes())
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
        digest.update(memoryview(value).cast("B"))
    for frame in frames:
        digest.update(str([(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes)]).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def training_cache_path(kind: str, contract: dict, *, arrays=(), frames=()) -> Path | None:
    if os.environ.get("AIC_TRAINING_CACHE", "1") == "0":
        return None
    key = training_fingerprint(contract, arrays=arrays, frames=frames)
    return C.EXPERIMENT_OUTPUT_ROOT / "training_cache_v1" / C.MODEL_DIR / kind / key


def read_training_cache(directory: Path | None) -> Path | None:
    if directory is None:
        return None
    kind = directory.parent.name
    try:
        record = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
        if record["key"] != directory.name:
            raise ValueError("训练缓存请求标识不匹配")
        artifact = _artifact_path(directory, record["filename"])
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("训练缓存内容校验失败")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if (directory / "complete.json").exists():
            logging.getLogger("aic-half").warning("训练缓存失效，将重新拟合: %s", exc)
        bucket = _TRAINING_CACHE_STATS["misses"]
        bucket[kind] = bucket.get(kind, 0) + 1
        return None
    bucket = _TRAINING_CACHE_STATS["hits"]
    bucket[kind] = bucket.get(kind, 0) + 1
    logging.getLogger("aic-half").info("复用训练缓存: %s/%s", kind, directory.name[:12])
    return artifact


def write_training_cache(directory: Path | None, writer) -> None:
    """先完成独立临时文件，再发布内容与校验标记；中断产物不能作为成功缓存。"""
    if directory is None:
        return
    try:
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".writing-", dir=directory.parent) as scratch:
            artifact = Path(writer(Path(scratch)))
            record = {"key": directory.name, "filename": artifact.name,
                      "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
            marker = Path(scratch) / "complete.json"
            marker.write_text(json.dumps(record), encoding="utf-8")
            directory.mkdir(exist_ok=True)
            artifact.replace(directory / artifact.name)
            marker.replace(directory / marker.name)
    except OSError as exc:
        logging.getLogger("aic-half").warning("训练缓存写入失败，保留本次内存模型: %s", exc)


def training_cache_stats(*, reset: bool = False) -> dict:
    result = {key: dict(value) for key, value in _TRAINING_CACHE_STATS.items()}
    if reset:
        for value in _TRAINING_CACHE_STATS.values():
            value.clear()
    return result


@dataclass
class RunContext:
    run_id: str
    root: Path
    source_dir: Path
    model_dir: Path
    report_dir: Path
    log_path: Path
    mode: str
    quick: bool = False
    experimental: bool = False
    window: str | None = None
    prediction_grid: pd.DatetimeIndex | None = None
    optimization_grid: pd.DatetimeIndex | None = None
    source_report: dict = field(default_factory=dict)
    source_root: Path | None = None


def _run_id(mode: str) -> str:
    return f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_{mode.lower()}"


def begin_run(config=C, mode: str = "backtest", experimental: bool = False,
              quick: bool = False) -> RunContext:
    run_id = _run_id(mode)
    # 实验复用完整归档链路，但整个目录必须隔离，避免干扰选定运行结果。
    base = config.EXPERIMENT_OUTPUT_ROOT if experimental else config.OUTPUT_ROOT
    root = base / run_id
    source = ((config.ROOT / "tmp_main" / run_id / config.SOURCE_DIR)
              if experimental else root / config.SOURCE_DIR)
    models = root / config.MODEL_DIR
    report = root / config.REPORT_DIR
    root.mkdir(parents=True, exist_ok=False)
    for path in (source, models, report):
        path.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(run_id=run_id, root=root, source_dir=source,
                     model_dir=models, report_dir=report,
                     log_path=report / config.REPORT_LOG, mode=mode,
                     quick=quick, experimental=experimental)
    write_source_snapshot(ctx, config)
    return ctx


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


def expected_result_grid(name: str, origins=None, quick: bool = False) -> pd.DatetimeIndex:
    """预测表保留起报网格，优化表按业务发布对应的目标窗口覆盖。"""
    grid = (pd.date_range(C.OFFICIAL_EVAL_START, C.OFFICIAL_EVAL_END,
                          freq=f"{C.STEP_MINUTES}min")
            if origins is None else pd.DatetimeIndex(origins))
    if name == "opt_result.csv":
        from main_opt import optimization_effective_grid
        if C.OPT_CONFIG["datetime_semantics"] != "target_time":
            raise ValueError("未实现的优化时间语义，禁止静默平移结果")
        grid = optimization_effective_grid(grid, quick)
    return grid


def validate_publication_contract(meta: dict, grid) -> dict:
    """核对新归档的实际发布窗口；历史包未声明该扩展契约时保留兼容读取。"""
    clock = meta.get("output_contract", {}).get("optimization_publication_time")
    if clock is None:
        return {"passed": True, "checked": False, "problems": []}
    from main_opt import daily_publication_times
    problems = []
    constraint_warnings = []
    try:
        grid = pd.DatetimeIndex(grid)
        expected = daily_publication_times(grid, publication_time=clock)
        summary = meta.get("optimization", {})
        export_only = summary.get("validation_basis") in ("causal_fallback_export", "unverified_export", "adopted_policy_fallback")
        export_only = export_only or (summary.get("validation_basis") == "adopted_operating_policy"
                                     and summary.get("constraint_validation_passed") is False)
        constraint_problems = constraint_warnings if export_only else problems
        plans = summary.get("daily_plans", [])
        issues = pd.DatetimeIndex([p["issued_at"] for p in plans])
        if not issues.equals(expected) or summary.get("publication_time") != clock:
            problems.append("每日实际发布时间与归档契约不一致")
        for plan, issue in zip(plans, issues):
            effective = pd.date_range(issue + pd.Timedelta(minutes=C.STEP_MINUTES),
                periods=C.STEPS_PER_DAY, freq=f"{C.STEP_MINUTES}min")
            visible = effective.intersection(grid)
            if plan.get("steps") != C.STEPS_PER_DAY or plan.get("published_steps") != len(visible):
                problems.append("每日窗口长度或首尾提交范围不一致")
            if (not len(visible) or plan.get("effective_start") != str(effective[0])
                    or plan.get("effective_end") != str(effective[-1])
                    or plan.get("published_start") != str(visible[0])
                    or plan.get("published_end") != str(visible[-1])):
                problems.append("每日计划的生效边界与实际发布时间不一致")
            if plan.get("validation", {}).get("passed") is not True:
                constraint_problems.append("每日计划缺少通过的约束核验")
            if summary.get("daily_control_policy") == "reserve140_terminal200_observed_fuel_cap240_to280":
                control = plan.get("control", {})
                limits = summary.get("daily_control_parameters", {})
                cap = control.get("total_power_cap_mw", np.nan)
                if (not limits.get("daily_base_power_cap_mw", np.inf) <= cap <= limits.get("daily_max_power_cap_mw", -np.inf)
                        or control.get("terminal_power_cap_mw") != limits.get("daily_terminal_power_cap_mw")
                        or control.get("terminal_inventory_m3") != limits.get("daily_terminal_inventory_m3")):
                    problems.append("每日计划的保守负荷或末端状态与冻结控制参数不一致")
                if plan.get("group_allocation", {}).get("passed") is not True:
                    problems.append("每日计划缺少通过的分组供气守恒核验")
            elif summary.get("daily_control_policy"):
                if (plan.get("model_constraints_passed") is not True
                        or plan.get("group_allocation", {}).get("passed") is not True):
                    constraint_problems.append("净资源每日计划缺少通过的模型约束或分组供气核验")
        resource = meta["resource"]
        ends = [resource["train_end"], resource.get("heat_rate", {}).get("calibration_end")]
        recorded = summary.get("resource_publication_check", {})
        ends.extend(recorded.get("training_ends", {}).values())
        if (not len(expected) or not recorded.get("passed") or recorded.get("first_issue") != str(expected[0])
                or resource.get("train_end") is None or resource.get("heat_rate", {}).get("calibration_end") is None
                or any(pd.isna(pd.Timestamp(v)) or pd.Timestamp(v) > expected[0] for v in ends if v is not None)):
            problems.append("资源权重或标定参数晚于首份计划发布时刻")
    except (KeyError, TypeError, ValueError) as exc:
        problems.append("每日发布契约无效: " + str(exc))
    return {"passed": not problems, "checked": True, "problems": list(dict.fromkeys(problems)),
            "constraint_warnings": list(dict.fromkeys(constraint_warnings))}


def validate_official_frame(name: str, frame: pd.DataFrame, expected_grid=None) -> dict:
    """按赛题字段契约校验预测/优化宽表，问题不静默放过。"""
    kind = {"s_result.csv": "s", "l_result.csv": "l",
            "opt_result.csv": "opt"}[name]
    expected = official_columns(kind)
    problems = []
    if frame.empty:
        problems.append(f"{name} 为空")
    if list(frame.columns) != expected:
        problems.append(f"{name} 列名与官方契约不一致")
    if "datetime" in frame.columns:
        times = pd.to_datetime(frame["datetime"], errors="coerce")
        if times.isna().any():
            problems.append(f"{name} datetime 无法解析")
        else:
            times = pd.DatetimeIndex(times)
            if times.has_duplicates:
                problems.append(f"{name} datetime 重复")
            if not times.is_monotonic_increasing:
                problems.append(f"{name} datetime 非递增")
            grid = expected_result_grid(name) if expected_grid is None else pd.DatetimeIndex(expected_grid)
            if not times.equals(grid):
                problems.append(f"{name} 时间覆盖不一致：应有{len(grid)}行，实际{len(times)}行，"
                                f"缺失{len(grid.difference(times))}个，多出{len(times.difference(grid))}个时间点")
    numeric = frame.drop(columns=["datetime"], errors="ignore")
    if len(numeric.columns):
        try:
            values = numeric.to_numpy(dtype=float)
        except (ValueError, TypeError):
            problems.append(f"{name} 存在非数值字段")
        else:
            if not np.isfinite(values).all():
                problems.append(f"{name} 存在非有限值")
            if (values < 0).any():
                problems.append(f"{name} 存在负值")
            if kind == "opt" and {"opt_generator_1", "opt_generator_all"}.issubset(numeric.columns):
                first = numeric["opt_generator_1"].to_numpy(dtype=float)
                total = numeric["opt_generator_all"].to_numpy(dtype=float)
                if ((first > C.CAPACITY["generator_1"] + 1e-6).any()
                        or (total > C.CAPACITY["generator_all"] + 1e-6).any()
                        or (first > total + 1e-6).any()):
                    problems.append(f"{name} 聚合负荷超容量或总负荷小于小机组负荷")
                ramp = C.OPT_CONFIG["ramp_fraction_per_min"] * C.STEP_MINUTES
                tolerance = C.OPT_CONFIG["power_tolerance_mw"] + 1e-6
                if (np.abs(np.diff(first)) > C.CAPACITY["generator_1"] * ramp + tolerance).any() \
                        or (np.abs(np.diff(total)) > C.CAPACITY["generator_all"] * ramp + tolerance).any():
                    problems.append(f"{name} 连续指令聚合爬坡越界")
    return {"passed": not problems, "problems": problems,
            "columns": list(frame.columns), "rows": int(len(frame))}


def write_predictions(ctx: RunContext, grid: pd.DatetimeIndex,
                      short_pred: np.ndarray, long_pred: np.ndarray) -> dict:
    grid = pd.DatetimeIndex(grid)
    if grid.empty or grid.has_duplicates or not grid.is_monotonic_increasing:
        raise ValueError("预测起报网格为空、重复或非递增")
    if len(grid) > 1 and not np.all(np.diff(grid.asi8) == pd.Timedelta(minutes=C.STEP_MINUTES).value):
        raise ValueError("预测起报网格缺失时间点或不是15分钟连续网格")
    if ctx.window == "half_test" and not grid.equals(expected_result_grid("s_result.csv")):
        raise ValueError("预测起报网格未覆盖完整 half_test")
    ctx.prediction_grid = grid
    short_check = validate_cube(short_pred, len(C.HORIZONS_S), len(grid))
    long_check = validate_cube(long_pred, len(C.HORIZONS_L), len(grid))
    if not short_check["valid"] or not long_check["valid"]:
        raise ValueError("预测校验失败: "
                         + "; ".join(short_check["problems"] + long_check["problems"]))
    short = _cube_to_frame(short_pred, grid, C.HORIZONS_S)[_short_columns()]
    long = _cube_to_frame(long_pred, grid, C.HORIZONS_L)[_long_columns()]
    # 官方契约校验放在落盘前，列名错位不能靠事后读回才发现。
    for name, frame in (("s_result.csv", short), ("l_result.csv", long)):
        check = validate_official_frame(name, frame, grid)
        if not check["passed"]:
            raise ValueError(f"{name} 未通过官方契约: " + "; ".join(check["problems"]))
    short.to_csv(ctx.root / "s_result.csv", index=False, encoding="utf-8", float_format="%.6f")
    long.to_csv(ctx.root / "l_result.csv", index=False, encoding="utf-8", float_format="%.6f")

    for name, frame, columns, check in (("s_result.csv", short, _short_columns(), short_check),
                                        ("l_result.csv", long, _long_columns(), long_check)):
        reread = pd.read_csv(ctx.root / name, encoding="utf-8")
        reread_check = validate_official_frame(name, reread, grid)
        if not reread_check["passed"]:
            raise ValueError("; ".join(reread_check["problems"]))
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
    if list(frame.columns) != required:
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
    if ctx.prediction_grid is None:
        raise ValueError("优化归档缺少本次预测起报网格")
    grid = expected_result_grid("opt_result.csv", ctx.prediction_grid, ctx.quick)
    check = validate_official_frame("opt_result.csv", frame, grid)
    if not check["passed"]:
        raise ValueError("opt_result 未通过官方契约: " + "; ".join(check["problems"]))
    frame.to_csv(ctx.root / "opt_result.csv", index=False, encoding="utf-8",
                 float_format="%.6f")
    reread = pd.read_csv(ctx.root / "opt_result.csv", encoding="utf-8")
    check = validate_official_frame("opt_result.csv", reread, grid)
    if not check["passed"]:
        raise ValueError("; ".join(check["problems"]))
    ctx.optimization_grid = grid
    from main_opt import _aggregate_ramp_groups, _planning_ramp_groups
    submitted_load = reread[["opt_generator_1", "opt_generator_all"]].to_numpy(dtype=float)
    difference = np.abs(np.diff(submitted_load, axis=0))
    limits = np.array([limit for _, limit in _aggregate_ramp_groups(C)])
    excess = difference - limits
    ramp_audit = {"basis": C.OPT_CONFIG["ramp_time_base_source"],
        "scope": "adjacent_submitted_rows; initial_state_checked_in_dispatch; not_official_evaluator",
        "limit_mw_per_step": limits.tolist(),
        "generation_limit_mw_per_step": [limit for _, limit in _planning_ramp_groups(C)],
        "maximum_change_mw": difference.max(axis=0).tolist() if len(difference) else [0., 0.],
        "violating_rows_with_numeric_tolerance": int((excess > C.OPT_CONFIG["power_tolerance_mw"]).any(axis=1).sum()),
        "floating_boundary_rows": int(((excess > 0.).any(axis=1)
            & ~(excess > C.OPT_CONFIG["power_tolerance_mw"]).any(axis=1)).sum())}
    return {"written": True, "rows": int(len(reread)), "columns": list(reread.columns),
            "generator_1_range": [float(g1.min()), float(g1.max())],
            "generator_all_range": [float(allp.min()), float(allp.max())],
            "time_semantics": C.OPT_CONFIG["datetime_semantics"], "ramp_audit": ramp_audit,
            "start": str(grid[0]), "end": str(grid[-1]), "partial": bool(ctx.quick)}


def _model_tag(key) -> str:
    if isinstance(key, tuple):
        return "__".join(str(part) for part in key)
    return str(key)


def _grid_spec(grid) -> dict | None:
    if grid is None or len(grid) == 0:
        return None
    return {"start": str(grid[0]), "end": str(grid[-1]), "rows": len(grid),
            "step_minutes": C.STEP_MINUTES}


def validate_result_files(directory: Path, origins=None, quick: bool = False,
                          require_optimization: bool = False, optimization_grid=None) -> dict:
    problems, checks = [], {}
    for name in ("s_result.csv", "l_result.csv", "opt_result.csv"):
        path = directory / name
        if not path.is_file():
            if name != "opt_result.csv" or require_optimization:
                problems.append(f"缺少 {name}")
            continue
        try:
            frame = pd.read_csv(path)
            expected = (pd.DatetimeIndex(optimization_grid) if name == "opt_result.csv" and optimization_grid is not None
                        else expected_result_grid(name, origins, quick))
            check = validate_official_frame(name, frame, expected)
            checks[name] = check
            problems.extend(check["problems"])
        except (ValueError, TypeError, OSError) as exc:
            problems.append(f"{name} 无法校验: {exc}")
    return {"passed": not problems, "problems": problems, "files": checks}


def _artifact_path(root: Path, relative: str) -> Path:
    # Windows 归档路径也必须能在赛事 Linux 环境中重载。
    path = (root / str(relative).replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("模型文件路径超出归档目录")
    return path


_INFERENCE_SETTINGS = (
    "STEP_MINUTES", "STEPS_PER_DAY", "TARGETS", "CAPACITY", "STRUCTURAL_CAPACITY",
    "CLEANING", "CONVERTER_GAS_SCALE", "FUEL_PROXY_WEIGHTS", "PROFILE_DAYS",
    "PROFILE_RATIO_CLIP", "FEATURE_WINDOW_COLUMNS", "FEATURE_LAGS", "FEATURE_ROLLS",
    "FEATURE_STD_WINDOW", "OBSERVED_AT_T", "SHORT_GROUPS", "LONG_GROUPS",
    "AT_T_DEVICE_COLUMNS", "AT_T_MINUTE_GROUPS",
    "GAS_HOLDER", "OPT_CONFIG",
)


def inference_contract() -> dict:
    """只冻结影响推理语义的设置，训练批量等运行预算不影响归档兼容性。"""
    return json.loads(json.dumps({name: getattr(C, name) for name in _INFERENCE_SETTINGS}))


def _check_inference_contract(directory: Path, meta: dict) -> dict:
    archived = meta.get("inference_contract")
    legacy = archived is None
    literals = {}
    if legacy:
        # 旧包只解析本地源码中的常量，不执行归档代码，也不依赖开发缓存。
        import ast
        source = directory / C.SOURCE_DIR / "config.py"
        if not source.is_file():
            raise ValueError("旧归档缺少推理契约及源码/config.py，不能确认输入兼容性")
        for statement in ast.parse(source.read_text(encoding="utf-8-sig")).body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                try:
                    literals[statement.targets[0].id] = ast.literal_eval(statement.value)
                except (ValueError, TypeError):
                    pass
        current_tree = ast.parse((C.ROOT / "config.py").read_text(encoding="utf-8-sig"))
        expressions = {s.targets[0].id: ast.dump(s.value) for s in current_tree.body
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)}
        for statement in ast.parse(source.read_text(encoding="utf-8-sig")).body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                if name in _INFERENCE_SETTINGS and name not in literals and expressions.get(name) == ast.dump(statement.value):
                    literals[name] = getattr(C, name)
        archived = json.loads(json.dumps({name: literals.get(name) for name in _INFERENCE_SETTINGS}))
    current = inference_contract()
    optimizer_changed = archived.get("OPT_CONFIG") != current["OPT_CONFIG"]
    # 冻结预测权重允许采用本项目已选定的调度口径；预测输入与清洗配方仍须逐项兼容。
    optimizer_reconfigured = bool(optimizer_changed
        and C.OPT_CONFIG.get("holder_state_mode") == "daily_observed"
        and C.OPT_CONFIG.get("submission_validation_basis") == "adopted_operating_policy")
    def setting_value(settings, name):
        value = settings.get(name)
        if name == "GAS_HOLDER" and isinstance(value, dict):
            # 说明文字不影响数值语义；容量、边界和字段映射仍完整参与兼容检查。
            return {key: item for key, item in value.items() if key != "note"}
        return value
    mismatches = [name for name in _INFERENCE_SETTINGS if setting_value(archived, name) != setting_value(current, name)
                  and not (name == "OPT_CONFIG" and optimizer_reconfigured)]
    if mismatches:
        raise ValueError("归档推理设置与当前代码不兼容，禁止静默覆盖：" + ", ".join(mismatches))
    required = meta.get("required_components")
    if required is None:
        required = {name: {"generator1": bool(literals.get("GENERATOR1_SPECIALIST", {}).get(name)),
                           "availability": bool(literals.get("AVAILABILITY_FALLBACK", {}).get(name)),
                           "process_expert": name == "long" and bool(literals.get("LONG_PROCESS_EXPERT"))}
                    for name in ("short", "long")}
    for name, components in required.items():
        for component, needed in components.items():
            if needed and not meta.get(name, {}).get(component):
                raise ValueError(f"冻结归档缺少必要分量：{name}/{component}；不能自动重训或降级")
    return {"passed": True, "legacy_source_checked": legacy, "required_components": required,
            "optimization_policy_changed": optimizer_reconfigured,
            "descriptive_metadata_changed": archived.get("GAS_HOLDER", {}).get("note") != current["GAS_HOLDER"].get("note"),
            "optimization_policy_version": C.OPT_CONFIG.get("policy_version")}


def load_optimization_reference(directory: Path):
    """只读资源归档；opt 模式不加载或执行短期、长期网络。"""
    from main_opt import ResourceBundle
    directory = directory.resolve()
    meta = json.loads((directory / C.MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    if meta.get("role") != "final":
        raise ValueError("opt 只能复用 final 角色归档")
    compatibility = _check_inference_contract(directory, meta)
    item = meta["resource"]
    resource = ResourceBundle({}, {}, [], metrics=item.get("metrics", {}),
        heat_rate=item["heat_rate"], final_heat_rate=item["heat_rate"],
        gas_caps=item["gas_caps"], final_gas_caps=item["gas_caps"],
        train_end=item["train_end"], final_train_end=item["train_end"],
        selection=item.get("selection", {}), dispatch_selection=item.get("dispatch_selection", {}),
        final_physical_validation=item.get("physical_validation", {}), final_net_supply=item.get("net_supply"))
    return meta, resource, compatibility


def load_final_models(directory: Path):
    """完整重载实际部署的全部分量；缺失或不兼容时直接报错，不补训。"""
    import main_nn
    from main_s import ShortBundle
    from main_l import LongBundle
    from main_opt import ResourceBundle

    directory = directory.resolve()
    model_root = directory / C.MODEL_DIR
    meta = json.loads((model_root / "model_meta.json").read_text(encoding="utf-8"))
    if meta["role"] != "final":
        raise ValueError("推理只能复用 final 角色，禁止把确认模型冒充最终模型")
    compatibility = _check_inference_contract(directory, meta)
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
        neural = main_nn.load_nn(_artifact_path(model_root, artifacts["nn"]["path"])) if "nn" in artifacts else None
        bundles.append(cls(kind=item["kind"], features=item["features"], horizons=horizons,
            blend=item["blend"], models={}, final_models=models, feature_names=item["feature_names"],
            nn_bundle=None, final_nn_bundle=neural, lgb_kind=item.get("lgb_kind"),
            nn_weights=item.get("nn_weights", {}),
            lgb_features=item.get("lgb_features"),
            selection=item.get("selection", {"chosen": {"kind": item["kind"], "features": item["features"]}}),
            scores=item["scores"],
            train_end=item["train_end"], final_train_end=item["train_end"]))
        if name == "short":
            # 旧归档未记录此配方时恢复旧行为，不能暗中改变历史模型的输入语义。
            bundles[-1].process_recovery = item.get("process_recovery", {})
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
            bundles[-1].final_generator1 = load_generator1(_artifact_path(model_root, item["generator1"]["path"]))
        if item.get("process_expert"):
            from main_s import load_generator1
            bundles[-1].final_process_expert = load_generator1(_artifact_path(model_root, item["process_expert"]["path"]))
    item, artifacts = meta["resource"], meta["artifacts"]["resource"]
    from main_opt import load_resource_model
    models = {tag: load_resource_model((model_root / "resource" / "final" / filename).read_text(encoding="utf-8"))
              for tag, filename in artifacts.get("final", {}).get("files", {}).items()}
    neural = main_nn.load_resource_nn(_artifact_path(model_root, artifacts["nn"]["path"])) if "nn" in artifacts else None
    selection = {**item["selection"], "head_selection": item.get("head_selection", {})}
    resource = ResourceBundle(models={}, final_models=models,
        feature_names=next(iter(models.values())).feature_name() if models else [],
        metrics=item["metrics"], heat_rate=item["heat_rate"], final_heat_rate=item["heat_rate"],
        gas_caps=item["gas_caps"], final_gas_caps=item["gas_caps"],
        train_end=item["train_end"], final_train_end=item["train_end"],
        selection=selection, dispatch_selection=item["dispatch_selection"],
        final_physical_validation=item.get("physical_validation", {}),
        resource_nn=None, final_resource_nn=neural, blocker=item.get("blocker"),
        final_net_supply=item.get("net_supply"))
    bundles.append(resource)
    for name, bundle in zip(("short", "long", "resource"), bundles):
        if not bundle.final_models and ((name == "resource" and not bundle.final_net_supply)
                or (name != "resource" and bundle.kind != "nn")):
            raise ValueError(f"冻结归档缺少{name}树模型")
        if name != "resource":
            neural = bundle.final_nn_bundle
            if bundle.kind in ("nn", "nn_lgb_ens") and neural is None:
                raise ValueError(f"冻结归档缺少{name}神经网络")
            if neural is not None and (tuple(neural.horizons) != tuple(bundle.horizons)
                    or pd.Timestamp(neural.train_end) > pd.Timestamp(bundle.final_train_end)):
                raise ValueError(f"{name}网络跨度或训练截止与归档不一致")
            for state in (bundle.final_generator1, getattr(bundle, "final_process_expert", None), bundle.final_availability):
                _validate_frozen_state(state, bundle.final_train_end, name)
        bundle.archive_compatibility = compatibility
    return bundles


def _validate_frozen_state(state, cutoff, name):
    if not state:
        return
    end = state.get("train_end")
    if end is not None and (pd.isna(pd.Timestamp(end)) or pd.Timestamp(end) > pd.Timestamp(cutoff)):
        raise ValueError(f"{name}专项训练截止超过所属模型截止")
    recipe = state.get("recipe", {})
    if recipe.get("calibration") and not state.get("calibration"):
        raise ValueError(f"{name}冻结配方缺少校准分量，禁止自动补训")
    if recipe.get("kind") in ("nn", "refined_nn") and state.get("nn") is None:
        raise ValueError(f"{name}专项缺少网络权重")
    if recipe.get("kind") == "refined_nn" and not state.get("base_state"):
        raise ValueError(f"{name}专项缺少基础模型")
    neural = state.get("nn")
    if neural is not None and pd.Timestamp(neural.train_end) > pd.Timestamp(end or cutoff):
        raise ValueError(f"{name}专项网络训练截止不一致")
    for key in ("base_state", "retained_state", "calibration"):
        _validate_frozen_state(state.get(key), end or cutoff, name)


def _save_boosters(directory: Path, prefix: str, models: dict,
                    feature_names: list[str]) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    index = {}
    reload_ok = True
    probe = np.zeros((3, len(feature_names)), dtype=np.float32)
    for key, booster in C.progress(models.items(), desc=f"保存 {prefix} 模型", unit="模型"):
        tag = _model_tag(key)
        file_name = f"{prefix}_{tag}.txt"
        path = directory / file_name
        # 由 Python 写入模型文本，独立解压到中文目录后也不依赖 C++ 文件路径支持。
        path.write_text(booster.model_to_string(), encoding="utf-8")
        from main_opt import load_resource_model
        loaded = load_resource_model(path.read_text(encoding="utf-8"))
        same = np.allclose(booster.predict(probe), loaded.predict(probe), rtol=1e-6, atol=1e-6)
        reload_ok &= bool(same)
        index[tag] = file_name
    return {"files": index, "reload_ok": bool(reload_ok)}


def save_models(ctx: RunContext, short_bundle, long_bundle,
                resource_bundle, opt_summary: dict, use_final: bool = False) -> dict:
    """只归档实际用于本次结果的角色，候选与另一角色留在实验流程中。"""
    import main_nn

    role = "final" if use_final else "confirm"
    summary = {"role": role, "short": {}, "long": {}, "resource": {}, "reload_ok": True}
    for name, bundle in (("short", short_bundle), ("long", long_bundle),
                         ("resource", resource_bundle)):
        models = bundle.final_models if use_final else bundle.models
        if name == "resource" and use_final and bundle.final_net_supply is not None:
            # 净资源路径只使用画像、气电和容量参数，不归档未参与本次推理的旧包络树。
            models = {}
        prefix = "resource" if name == "resource" else f"{bundle.kind}_{bundle.features}"
        if models:
            info = _save_boosters(ctx.model_dir / name / role, prefix, models,
                                   bundle.feature_names)
            summary[name][role] = info
            summary["reload_ok"] &= info["reload_ok"]
        availability = getattr(bundle, "final_availability", None) if use_final else None
        if availability:
            info = _save_boosters(ctx.model_dir / name / "final_availability", "availability",
                availability["models"], availability["feature_names"])
            summary[name]["availability"] = info
            summary["reload_ok"] &= info["reload_ok"]
            if availability.get("retained_state"):
                retained = availability["retained_state"]
                retained_info = _save_boosters(ctx.model_dir / name / "final_availability" / "retained",
                    "availability", retained["models"], retained["feature_names"])
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
            summary[name]["generator1"] = {"path": directory.relative_to(ctx.model_dir).as_posix(), "reload_ok": same}
            summary["reload_ok"] &= same
        process = getattr(bundle, "final_process_expert", None) if use_final else None
        if process:
            from main_s import save_generator1, load_generator1
            directory = ctx.model_dir / name / "final_process_expert"
            save_generator1(process, directory)
            loaded = load_generator1(directory)
            same = loaded["recipe"] == process["recipe"] and loaded["train_end"] == process["train_end"]
            summary[name]["process_expert"] = {"path": directory.relative_to(ctx.model_dir).as_posix(), "reload_ok": same}
            summary["reload_ok"] &= same
        if nn is not None:
            path = main_nn.save_nn(nn, ctx.model_dir / name / role)
            loaded = main_nn.load_nn(Path(path))
            same = (len(loaded.seed_states) == len(nn.seed_states)
                    and np.array_equal(loaded.feat_mean, nn.feat_mean)
                    and all(np.array_equal(loaded.state_dict[k].numpy(), v.numpy())
                            for k, v in nn.state_dict.items()))
            summary[name]["nn"] = {"path": Path(path).relative_to(ctx.model_dir).as_posix(), "kind": nn.kind, "reload_ok": bool(same)}
            summary["reload_ok"] &= bool(same)
        if name == "resource":
            resource_nn = bundle.final_resource_nn if use_final else bundle.resource_nn
            if resource_nn is not None and not (use_final and bundle.final_net_supply is not None):
                path = main_nn.save_resource_nn(resource_nn, ctx.model_dir / name / role / "resource_nn.pt")
                loaded = main_nn.load_resource_nn(path)
                same = (loaded["feature_names"] == resource_nn["feature_names"]
                    and all(np.array_equal(loaded[k], resource_nn[k]) for k in ("center", "scale", "gas_scale"))
                    and all(np.array_equal(loaded["state_dict"][k].numpy(), value.cpu().numpy())
                            for k, value in resource_nn["state_dict"].items()))
                summary[name]["nn"] = {"path": path.relative_to(ctx.model_dir).as_posix(),
                    "kind": resource_nn["kind"], "train_end": resource_nn["train_end"],
                    "max_training_target": resource_nn["max_training_target"], "reload_ok": bool(same)}
                summary["reload_ok"] &= bool(same)
    if not summary["reload_ok"]:
        raise ValueError("实际模型保存后重载不一致")

    def forecast_meta(bundle):
        nn = getattr(bundle, "final_nn_bundle" if use_final else "nn_bundle", None)
        availability = getattr(bundle, "final_availability", None) if use_final else None
        state = {k: v for k, v in availability.items() if k != "models"} if availability else None
        if state and state.get("retained_state"):
            state["retained_state"] = {k: v for k, v in state["retained_state"].items() if k != "models"}
        return {"kind": bundle.kind, "features": bundle.features,
                "selection": {k: bundle.selection[k] for k in ("chosen", "generator1")
                              if k in bundle.selection},
                "availability": state,
                "generator1": summary["short" if len(bundle.horizons) == 8 else "long"].get("generator1"),
                "process_expert": summary["short" if len(bundle.horizons) == 8 else "long"].get("process_expert"),
                "blend": bundle.blend, "feature_names": bundle.feature_names,
                "lgb_kind": bundle.lgb_kind, "lgb_features": bundle.lgb_features,
                "nn_weights": bundle.nn_weights,
                **({"process_recovery": bundle.process_recovery} if hasattr(bundle, "process_recovery") else {}),
                "scores": bundle.scores,
                "train_end": nn.train_end if nn is not None else
                             (bundle.final_train_end if use_final else bundle.train_end)}

    meta = {
        "schema_version": 3, "role": role, "artifacts": summary,
        "inference_contract": inference_contract(),
        "required_components": {**{name: {"generator1": bool(bundle.final_generator1),
            "availability": bool(bundle.final_availability),
            "process_expert": bool(getattr(bundle, "final_process_expert", None))}
            for name, bundle in (("short", short_bundle), ("long", long_bundle))},
            "resource": {"net_supply": resource_bundle.final_net_supply is not None}} if use_final else {},
        "run": {"mode": ctx.mode, "quick": ctx.quick, "experimental": ctx.experimental},
        "output_contract": {"prediction": _grid_spec(ctx.prediction_grid),
                            "optimization": _grid_spec(ctx.optimization_grid),
                            "optimization_coverage_basis": "same_datetime_as_long_forecast; daily_fixed_publication",
                            "window": ctx.window,
                            "optimization_time_semantics": C.OPT_CONFIG["datetime_semantics"],
                            "optimization_publication_time": opt_summary.get("publication_time"),
                            "resource_publication_check": opt_summary.get("resource_publication_check")},
        "source": write_source_snapshot(ctx),
        "short": forecast_meta(short_bundle), "long": forecast_meta(long_bundle),
        "resource": {"metrics": resource_bundle.metrics,
                     "selection": {k: resource_bundle.selection.get(k)
                                   for k in ("objectives", "selection_scope", "availability_adjustment")},
                     "head_selection": {k: resource_bundle.selection.get("head_selection", {}).get(k)
                                        for k in ("gases", "weight", "horizons", "method")},
                     "dispatch_selection": {k: resource_bundle.dispatch_selection.get(k)
                                            for k in ("chosen", "candidate", "replan_steps",
                                                      "control_options", "scope", "promotion", "publication_time")},
                     "heat_rate": resource_bundle.final_heat_rate if use_final else resource_bundle.heat_rate,
                     "gas_caps": resource_bundle.final_gas_caps if use_final else resource_bundle.gas_caps,
                     "train_end": resource_bundle.final_train_end if use_final else resource_bundle.train_end,
                     "physical_validation": (resource_bundle.final_physical_validation if use_final
                                             else resource_bundle.physical_validation),
                     "net_supply": resource_bundle.final_net_supply if use_final else None,
                     "blocker": resource_bundle.blocker},
        "optimization": opt_summary,
        "data_contract": {"target_mask": "整个评测窗口", "step_minutes": C.STEP_MINUTES,
                          "prediction_power_gas_policy": C.PREDICTION_POWER_GAS_POLICY,
                          "future_dynamic_fields": "forbidden"},
    }
    if ctx.optimization_grid is not None:
        publication = validate_publication_contract(meta, ctx.optimization_grid)
        if not publication["passed"]:
            raise ValueError("模型归档的每日发布契约不一致: " + "; ".join(publication["problems"]))
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


def _source_files(root: Path) -> dict[str, Path]:
    files = ["main.py", "main_s.py", "main_l.py", "main_nn.py", "main_opt.py", "config.py", "output.py",
             "requirements.txt", "README.md", "data_cleaning/__init__.py",
             "data_cleaning/pipeline.py", "data_cleaning/quality.py"]
    files.extend(path.relative_to(root).as_posix() for path in (root / "data_cleaning").glob("*.py"))
    return {name: root / name for name in sorted(set(files))}


def write_source_snapshot(ctx: RunContext, config=C) -> dict:
    """首调冻结启动源码；后续仅核验，禁止用运行末尾的新代码覆盖快照。"""
    if ctx.source_report:
        files = _source_files(ctx.source_root)
        expected = ctx.source_report["sha256"]
        changed = sorted(set(files) ^ set(expected))
        for name, digest in expected.items():
            source, snapshot = files.get(name), ctx.source_dir / name
            if (source is None or not source.is_file() or not snapshot.is_file()
                    or hashlib.sha256(source.read_bytes()).hexdigest() != digest
                    or hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest):
                changed.append(name)
        ctx.source_report["consistency_passed"] = not changed
        if changed:
            ctx.source_report["changed_files"] = sorted(set(changed))
            raise ValueError("运行源码或冻结快照已变化，拒绝归档: " + "; ".join(sorted(set(changed))))
        return ctx.source_report
    ctx.source_root = config.ROOT.resolve()
    files = _source_files(ctx.source_root)
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise ValueError("缺少必要源码: " + "; ".join(missing))
    contents = {name: path.read_bytes() for name, path in files.items()}
    ctx.source_dir.mkdir(parents=True, exist_ok=True)
    (ctx.source_dir / ".gitignore").unlink(missing_ok=True)
    for name, content in C.progress(contents.items(), desc="冻结启动源码", unit="文件"):
        target = ctx.source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    ctx.source_report = {"files": list(contents), "count": len(contents), "directory": str(ctx.source_dir),
        "frozen_at": pd.Timestamp.now().isoformat(timespec="seconds"), "freeze_stage": "run_start",
        "sha256": {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}}
    return write_source_snapshot(ctx, config)


def build_zip(ctx: RunContext, submittable: bool, config=C) -> dict:
    source = write_source_snapshot(ctx, config)
    if ctx.prediction_grid is None:
        raise ValueError("缺少当次输出时间契约，拒绝打包")
    origins = None if ctx.window == "half_test" else ctx.prediction_grid
    if ctx.mode in ("infer", "opt") and not (ctx.root / "opt_result.csv").is_file():
        # 推理包缺少必交优化表时不制造同名的不完整ZIP。
        return {"name": None, "path": None, "members": 0, "root_csv": [], "crc_ok": False,
            "submittable": False, "complete_result_set": False, "missing_result_files": ["opt_result.csv"],
            "coverage": validate_result_files(ctx.root, origins, ctx.quick, require_optimization=True),
            "source_consistency_passed": True, "reason": "缺少有效优化表，未生成不完整推理包"}
    coverage = validate_result_files(ctx.root, origins, ctx.quick,
                                    require_optimization=submittable)
    if not coverage["passed"]:
        raise ValueError("结果覆盖校验失败: " + "; ".join(coverage["problems"]))
    if submittable:
        from main_opt import submission_readiness
        meta = json.loads((ctx.model_dir / "model_meta.json").read_text(encoding="utf-8"))
        readiness = submission_readiness({"summary": meta["optimization"],
            "schedule": pd.read_csv(ctx.root / "opt_result.csv")})
        if (ctx.mode not in ("infer", "opt") or ctx.window != "half_test" or ctx.quick or ctx.experimental
                or not ctx.prediction_grid.equals(expected_result_grid("s_result.csv"))):
            raise ValueError("非完整正式测试运行，拒绝标记为可提交")
        if not readiness["passed"]:
            raise ValueError("声明口径下的提交核验未通过: " + "; ".join(readiness["problems"]))
    name = config.FORMAL_ZIP_NAME if submittable else config.MOCK_ZIP_NAME
    final_path = ctx.root / name
    temp_path = ctx.root / (name + ".tmp")
    members = ["s_result.csv", "l_result.csv"]
    if (ctx.root / "opt_result.csv").is_file():
        members.append("opt_result.csv")
    paths = [(ctx.root / member, member) for member in members]
    for folder_name, folder in ((config.SOURCE_DIR, ctx.source_dir), (config.MODEL_DIR, ctx.model_dir)):
        paths.extend((path, f"{folder_name}/{path.relative_to(folder).as_posix()}")
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
        for relative, digest in source["sha256"].items():
            if hashlib.sha256(packed.read(f"{config.SOURCE_DIR}/{relative}")).hexdigest() != digest:
                raise ValueError(f"ZIP 源码与启动快照不一致: {relative}")
    write_source_snapshot(ctx, config)
    temp_path.replace(final_path)
    return {"name": name, "path": str(final_path), "members": len(names),
            "root_csv": members, "crc_ok": True, "submittable": bool(submittable),
            "complete_result_set": all(member in members for member in ("s_result.csv", "l_result.csv", "opt_result.csv")),
            "missing_result_files": [member for member in ("s_result.csv", "l_result.csv", "opt_result.csv") if member not in members],
            "coverage": coverage, "source_consistency_passed": True}


def write_report(ctx: RunContext, summary: dict) -> Path:
    summary.setdefault("source", ctx.source_report)
    summary["training_cache"] = training_cache_stats()
    path = ctx.report_dir / C.REPORT_JSON
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")
    return path
