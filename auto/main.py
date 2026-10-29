"""供本地 Agent 或调度器调用的隔离实验循环，不依赖特定 Agent 接口。"""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
if __package__:
    from . import search
else:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("auto_search", Path(__file__).with_name("search.py"))
    search = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(search)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tmp" / "tmp_outputs"
CODE = ROOT / "tmp_main"
POINTER = ROOT / "tmp" / "aic_auto_active.json"
DEFAULT_WINDOWS = {"w1": ("2025-09-20", "2025-10-10"), "w2": ("2025-08-30", "2025-09-19"),
                   "w3": ("2025-07-31", "2025-08-20"), "w4": ("2025-07-01", "2025-07-21")}
TRACKS = ("short", "long", "load", "g1", "resource", "dispatch")
EDITABLE = {"config.py", "main_s.py", "main_l.py", "main_opt.py", "data_cleaning/pipeline.py"}
OVERRIDES = ("LOAD_RECIPE.", "TREE.", "PROCESS_RESOURCE.", "DISPATCH_CONTROL.",
             "GROUP_DISPATCH.smooth_penalty", "GROUP_DISPATCH.mapping_penalty")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_once(path, row, key):
    """结算先写审计证据再提交状态；恢复时不重复追加同一条结果。"""
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get(key) == row[key]:
                return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
        raise ValueError("标识只允许字母、数字、下划线与连字符")
    return value


def alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            # 无权查询时保守视为存活，防止重复训练。
            return ctypes.get_last_error() == 5
        code = wintypes.DWORD()
        try:
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def lock(folder):
    path = folder / "controller.lock"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                owner = read(path)
            except (ValueError, OSError):
                raise ValueError("锁记录不可读，请检查正在运行的控制器；不能直接重复启动")
            if alive(owner.get("pid")):
                raise ValueError("已有控制器正在运行")
            path.unlink()
    else:
        raise ValueError("未能获取实验锁")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created": now()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def source_names(folder):
    tree = ast.parse((folder / "config.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCE_FILES" for t in node.targets):
            names = ast.literal_eval(node.value)
            for name in names:
                target = (folder / name).resolve()
                if not target.is_relative_to(folder.resolve()) or not target.is_file():
                    raise ValueError("源码白名单包含不存在或越界的文件")
            return tuple(names)
    raise ValueError("配置缺少 SOURCE_FILES 白名单")


def snapshot(origin, target, names):
    target.mkdir(parents=True, exist_ok=False)
    for name in names:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin / name, dest)
    return {name: digest(target / name) for name in names}


def seeds(track):
    if track == "g1":
        # g1 任务按已记录的误差机制提出候选，避免重复运行通用参数种子。
        return []
    if track in ("short", "long", "load", "g1"):
        spans = ("short", "long") if track in ("load", "g1") else (track,)
        choices = []
        for span in spans:
            for leaves in (15, 63):
                choices.append((f"{span}树叶数{leaves}", {f"LOAD_RECIPE.tasks.{span}.leaves": leaves}))
            key = "decay_days" if span == "short" else "profile_days"
            for days in ((14, 42) if span == "short" else (28, 56)):
                choices.append((f"{span}{key}设为{days}天", {f"LOAD_RECIPE.tasks.{span}.{key}": days}))
        return choices
    if track == "resource":
        return [(f"资源缺测模型分位数{q}", {"PROCESS_RESOURCE.quantile": q}) for q in (.1, .4)] + [
            ("资源树增加正则", {"TREE.lambda_l2": 5.}), ("资源树进一步增加正则", {"TREE.lambda_l2": 10.})]
    return [(f"调度收益底线{v}", {"DISPATCH_CONTROL.benefit_fraction": v}) for v in (1., .995)] + [
        (f"分组负荷平滑惩罚{v}", {"GROUP_DISPATCH.smooth_penalty": v}) for v in (5., 20.)]


def campaign_path(name):
    if name is None:
        if not POINTER.is_file():
            raise ValueError("尚无活动任务，先执行 python auto/main.py init")
        name = read(POINTER)["campaign"]
    return RESULTS / identifier(name)


def save_state(folder, state, event=None):
    state["updated"] = now()
    write(folder / "state.json", state)
    if event:
        with (folder / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": now(), **event}, ensure_ascii=False, allow_nan=False) + "\n")


def window_design(path, selected):
    """只提取窗口常量，不执行历史脚本里的训练、后处理或输出副作用。"""
    windows = DEFAULT_WINDOWS
    if path is not None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        windows = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "WINDOWS" for t in node.targets):
                windows = ast.literal_eval(node.value)
                break
    if not isinstance(windows, dict) or not windows:
        raise ValueError("窗口脚本缺少可直接读取的 WINDOWS 字典")
    keys = list(windows) if selected == "all" else selected.split(",")
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("窗口列表为空或重复")
    result = []
    for key in keys:
        identifier(key)
        begin, end = (datetime.fromisoformat(v) for v in windows[key])
        stop = end + timedelta(hours=23, minutes=45)
        if (begin.time() != datetime.min.time() or end.time() != datetime.min.time() or
                begin > end or stop >= datetime(2025, 10, 11)):
            raise ValueError("脚本窗口须为正式测试之前、首尾均按日期给出的完整自然日")
        expected_origins = int((stop - begin).total_seconds() // 900) + 1
        result.append({"name": key, "start": str(begin), "end": str(stop),
                       "train_end": str(begin - timedelta(minutes=15)), "expected_origins": expected_origins})
    return result


def initialize(args):
    name = identifier(args.campaign or "auto_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder, code = RESULTS / name, CODE / name
    if folder.exists() or code.exists():
        raise ValueError("任务目录已存在，禁止覆盖；续跑使用 step 或 run")
    if args.training_budget == "full" and not args.authorization:
        raise ValueError("全预算必须填写当前真人授权出处；记录本身不构成授权")
    custom = any((args.start, args.end, args.train_end))
    if custom:
        if not all((args.start, args.end, args.train_end)):
            raise ValueError("自定义单窗须同时提供start、end、train-end")
        start, stop, cutoff = (datetime.fromisoformat(v) for v in (args.start, args.end, args.train_end))
        if not cutoff < start <= stop or stop >= datetime(2025, 10, 11):
            raise ValueError("需要训练截止早于验证起点，且验证真值早于正式测试期")
        if any(v.second or v.minute % 15 for v in (start, stop, cutoff)):
            raise ValueError("训练与验证时刻必须位于15分钟网格")
        windows = [{"name": "custom", "train_end": str(cutoff), "start": str(start), "end": str(stop)}]
    else:
        windows = window_design(args.window_script.resolve() if args.window_script else None, args.windows)
    names = source_names(ROOT)
    folder.mkdir(parents=True)
    manifest = snapshot(ROOT, code / "源码", names)
    shutil.copy2(Path(__file__).with_name("evaluate.py"), code / "evaluate.py")
    if not custom and args.window_script:
        shutil.copy2(args.window_script, code / "window_design.py")
    state = {"schema": 5, "campaign": name, "track": args.track, "status": "ready", "created": now(),
             "contract": {"windows": windows, "window_source": str(args.window_script.resolve()) if not custom and args.window_script else None,
                          "window_source_sha256": digest(args.window_script) if not custom and args.window_script else None,
                          "data_end": "2025-10-10 23:45:00", "score_tail": "within_window",
                          "training_budget": args.training_budget, "threads": args.threads,
                          "primary": "1_mape_g1_short_40_long_60" if args.track == "g1" else
                              "1_mape_macro" if args.track in ("short", "long", "load") else
                              "negative_resource_macro_wape" if args.track == "resource" else "observed_window_planned_net_value_cny",
                          "direction": "maximize", "truth_mask": "fixed_reference_available_nonzero_targets",
                          "aggregation": "equal_window_mean_then_fixed_task_weight",
                          "input_boundary": "target_blackout;use_after48h_missing;process_prefix_only",
                          "load_output": "deployed_pair_v1",
                          "authorization": args.authorization, "holder_version": os.environ.get("AIC_HOLDER_VERSION", "historical_raw")},
             "budget": {"experiments": args.max_experiments, "stagnation": args.patience,
                        "timeout_seconds": args.timeout_seconds, "max_restarts": args.max_restarts},
             "source_manifest": manifest, "evaluator_sha256": digest(code / "evaluate.py"),
             "queue": [{"hypothesis": "冻结当前正式配方，建立同口径基线", "overrides": {}, "baseline": True}] +
                      [{"hypothesis": h, "overrides": p, "baseline": False} for h, p in seeds(args.track)],
             "records": [], "active": None, "best": None, "baseline": None, "stagnation": 0,
             "target": args.target, "skipped": []}
    for index, proposal in enumerate(state["queue"]):
        proposal["proposal_id"] = f"h{index:04d}"
    state["proposal_counter"] = len(state["queue"])
    save_state(folder, state, {"event": "initialized"})
    if args.campaign is None:
        write(POINTER, {"campaign": name})
    return state


def overrides_valid(patch):
    if not isinstance(patch, dict):
        raise ValueError("参数覆盖必须是JSON对象")
    json.dumps(patch, allow_nan=False)
    for key in patch:
        if not any(key.startswith(prefix) if prefix.endswith(".") else key == prefix for prefix in OVERRIDES):
            raise ValueError("不允许修改该配置或验收边界：" + key)
        if not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", part) for part in key.split(".")):
            raise ValueError("非法配置路径")


def apply_overrides(source, patch):
    overrides_valid(patch)
    if not patch:
        return
    with (source / "config.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# 本候选的配方覆盖仅存在于隔离源码，正式配置保持原样。\n")
        for key, value in patch.items():
            parts = key.split(".")
            expression = parts[0] + "".join(f"[{part!r}]" for part in parts[1:])
            stream.write(f"{expression} = {value!r}\n")


def manifest_errors(folder, manifest):
    """先验证路径与真实文件，再使用工作进程声称成功的结果。"""
    if not isinstance(manifest, dict) or not manifest:
        return ["缺少非空文件清单：" + str(folder)]
    errors = []
    for name, expected in manifest.items():
        try:
            path = (folder / name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(folder.resolve()) or not path.is_file() or digest(path) != expected:
                errors.append("文件缺失、改变或越界：" + str(name))
        except (OSError, ValueError, TypeError):
            errors.append("文件无法核验：" + str(name))
    return errors


def evidence_errors(root, report, expected_job):
    errors = []
    job_file = root / "job.json"
    try:
        if not expected_job or digest(job_file) != expected_job or report.get("job_sha256") != expected_job:
            return ["任务文件与完成报告未绑定或已改变"]
        job = read(job_file)
        if job.get("id") != root.name or report.get("experiment_id") != root.name or Path(job["output"]).resolve() != root.resolve():
            errors.append("实验标识或输出目录与任务不一致")
        source = Path(job["source"])
        if source.resolve() != (CODE / root.name / "源码").resolve():
            errors.append("候选源码目录与实验不一致")
        errors.extend(manifest_errors(source, job.get("manifest")))
        errors.extend(manifest_errors(root, report.get("artifacts")))
        # 必需出口不能因清单缺项而绕过核验；交叉依赖模型也必须随本次输出保存。
        spans = ("short", "long") if job["track"] in ("load", "g1") else (job["track"],)
        for window in job["contract"].get("windows") or [None]:
            prefix = "窗口/" + window["name"] + "/" if window else ""
            required = []
            if job["track"] in ("short", "long", "load", "g1"):
                required += ["s_result.csv" if s == "short" else "l_result.csv" for s in spans]
                models = ("short", "long") if job["contract"].get("load_output") else spans
                required += ["模型/" + s + "/meta.json" for s in models]
            else:
                required += ["opt_result.csv", "trajectory.json", "模型/resource/meta.json"]
            if window:
                required.append("报告/运行结果.json")
            for name in required:
                if prefix + name not in report.get("artifacts", {}):
                    errors.append("产物清单缺少必需出口：" + prefix + name)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append("结果证据无法核验：" + str(exc))
    return errors


def record_errors(record):
    root = RESULTS / record["id"]
    path = root / "报告" / "运行结果.json"
    try:
        report = read(path)
        errors = []
        if report.get("primary") != record.get("primary") or report.get("status") != record["status"]:
            errors.append("记录与报告内容不一致")
        if record.get("report_sha256") and digest(path) != record["report_sha256"]:
            errors.append("结算后的报告已改变")
        if record.get("job_sha256"):
            if report.get("status") == "COMPLETE":
                errors.extend(evidence_errors(root, report, record["job_sha256"]))
            else:
                job_file = root / "job.json"
                if digest(job_file) != record["job_sha256"] or report.get("job_sha256") != record["job_sha256"]:
                    errors.append("失败记录与任务不一致")
                job = read(job_file)
                errors.extend(manifest_errors(Path(job["source"]), job.get("manifest")))
                if report.get("artifacts"):
                    errors.extend(manifest_errors(root, report["artifacts"]))
        elif report.get("artifacts"):
            errors.extend(manifest_errors(root, report["artifacts"]))
        return errors
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return ["记录无法核验：" + str(exc)]


def finalize(folder, state, active):
    existing = next((r for r in state["records"] if r["id"] == active["id"]), None)
    if existing is not None:
        return existing
    report_path = RESULTS / active["id"] / "报告" / "运行结果.json"
    report = read(report_path) if report_path.is_file() else {"status": "FAILED", "error": "实验中断或超时，未生成结果"}
    if report.get("status") != "COMPLETE":
        report.setdefault("experiment_id", active["id"])
        report.setdefault("job_sha256", active.get("job_sha256"))
    score = report.get("primary")
    checks = report.get("checks", {})
    required = {"causality", "coverage", "reload", "training_boundary", "source_unchanged"}
    if state.get("schema", 0) >= 5 and state["track"] in ("short", "long", "load", "g1"):
        required.add("deployment_output")
    if state["track"] in ("resource", "dispatch"):
        required.add("hard_constraints")
    valid = report.get("status") == "COMPLETE" and isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)
    valid = valid and required.issubset(checks) and all(checks[k] is True for k in required)
    errors = []
    if state.get("schema", 0) >= 4 and report.get("status") == "COMPLETE":
        errors = evidence_errors(report_path.parent.parent, report, active.get("job_sha256"))
        checks["artifact_integrity"] = not errors
        valid = valid and not errors
    if state["baseline"]:
        baseline_record = next(r for r in state["records"] if r["id"] == state["baseline"])
        errors.extend(record_errors(baseline_record))
        reference = read(RESULTS / state["baseline"] / "报告" / "运行结果.json")
        valid = valid and not errors and report.get("comparison") == reference.get("comparison")
    report["settlement_validation"] = {"passed": not errors, "errors": errors}
    previous = state["best"]
    improved = valid and (previous is None or score > previous["primary"])
    reference_id = previous["id"] if previous else state["baseline"]
    deltas = []
    if reference_id and report.get("status") == "COMPLETE":
        reference_report = read(RESULTS / reference_id / "报告" / "运行结果.json")
        def compare(candidate, reference, path=""):
            if not isinstance(candidate, dict) or not isinstance(reference, dict):
                return
            for key, value in candidate.items():
                before = reference.get(key)
                label = path + "." + key if path else key
                if key in ("score", "macro_score", "zero_mae", "wape") and isinstance(value, (float, int)) and isinstance(before, (float, int)):
                    delta = value - before
                    deltas.append({"metric": label, "baseline": before, "candidate": value, "delta": delta,
                                   "regression": delta < 0 if key in ("score", "macro_score") else delta > 0})
                elif key not in ("training", "naive"):
                    compare(value, before, label)
        compare(report.get("metrics", {}), reference_report.get("metrics", {}))
    report["selection"] = {"eligible": bool(valid), "improved": bool(improved), "comparison_parent": reference_id,
                           "branch_parent": active.get("parent_id"),
                           "primary_delta": score - previous["primary"] if previous and isinstance(score, (float, int)) else None,
                           "slice_deltas": deltas}
    write(report_path, report)
    if report.get("window_summary"):
        append_once(folder / "windows.jsonl", {"tag": active["id"], **report["window_summary"],
                    "primary": score, "eligible": bool(valid), "improved": bool(improved)}, "tag")
    if active["baseline"]:
        # 基线约束失败仍保留原始证据，不能通过换窗口把失败隐藏掉。
        state["baseline"] = active["id"]
    if improved:
        state["best"] = {"id": active["id"], "primary": score}
        state["stagnation"] = 0
    elif not active["baseline"]:
        state["stagnation"] += 1
    row = {"id": active["id"], "hypothesis": active["hypothesis"], "baseline": active["baseline"],
           "proposal_id": active.get("proposal_id"),
           "parent_id": active.get("parent_id"), "candidate_signature": active.get("candidate_signature"),
           "rationale": active.get("rationale", ""), "refs": active.get("refs", []),
           "expected_gain": active.get("expected_gain"), "confidence": active.get("confidence"),
           "cost_minutes": active.get("cost_minutes"), "seconds": report.get("seconds"),
           "outcome": "kept" if improved else "rejected" if valid else "invalid" if report.get("status") == "COMPLETE" else "failed",
           "status": report.get("status"), "primary": score, "eligible": bool(valid), "improved": bool(improved),
           "checks": checks, "error": report.get("error"), "report": str(report_path),
           "attempts": active.get("attempts", 0), "elapsed_seconds": active.get("elapsed_seconds", 0.),
           "job_sha256": active.get("job_sha256"), "report_sha256": digest(report_path)}
    state["records"].append(row)
    state["active"] = None
    state["status"] = "ready"
    if active["baseline"] and not valid:
        state.update(status="stopped", stop_reason="baseline_failed" if report.get("status") != "COMPLETE" else "baseline_invalid")
    elif state.get("target") is not None and state.get("best") and state["best"]["primary"] >= state["target"]:
        state.update(status="stopped", stop_reason="target_reached")
    elif len(state["records"]) >= state["budget"]["experiments"]:
        state.update(status="stopped", stop_reason="budget_exhausted")
    elif state["stagnation"] >= state["budget"]["stagnation"]:
        state.update(status="stopped", stop_reason="no_improvement")
    elif not state["queue"]:
        state.update(status="needs_proposal", stop_reason="queue_empty")
    if (folder / "stop.json").is_file():
        state.update(status="stopped", stop_reason=read(folder / "stop.json")["reason"])
    append_once(folder / "history.jsonl", {"at": now(), "event_key": "finished:" + active["id"], "event": "finished", **row}, "event_key")
    save_state(folder, state)
    return row


def fail_active(folder, state, active, reason):
    path = RESULTS / active["id"] / "报告" / "运行结果.json"
    received = read(path) if path.is_file() else None
    write(path, {
        "status": "FAILED", "error": reason, "experiment_id": active["id"],
        "job_sha256": active.get("job_sha256"), "worker_report": received,
        "artifacts": received.get("artifacts", {}) if received else {},
        "experimental": True, "submittable": False})
    return finalize(folder, state, active)


def step(folder, state, dry_run=False, allow_full=False):
    if dry_run:
        return {"status": "dry_run", "contract": state["contract"], "active": state["active"],
                "next": search.ranked(state["queue"])[0] if state["queue"] else None}
    if state["active"]:
        if state["active"].get("phase") == "preparing":
            if (folder / "stop.json").is_file():
                return fail_active(folder, state, state["active"], "STOPPED_DURING_PREPARATION")
            if state["contract"]["training_budget"] == "full" and not allow_full:
                raise ValueError("接续全预算实验仍须当前授权与 --allow-full")
            return prepare_active(folder, state)
        report = RESULTS / state["active"]["id"] / "报告" / "运行结果.json"
        worker_meta = report.parent / "worker.json"
        pid = state["active"].get("pid")
        if not pid and worker_meta.is_file():
            pid = read(worker_meta).get("pid")
        if not report.is_file() and alive(pid):
            return {"status": "running", "active": state["active"]}
        active = state["active"]
        if active.get("attempt_started"):
            spent = (datetime.now(timezone.utc) - datetime.fromisoformat(active["attempt_started"])).total_seconds()
            if report.is_file():
                measured = read(report).get("seconds")
                if isinstance(measured, (int, float)) and math.isfinite(measured) and measured >= 0:
                    spent = measured
            active["elapsed_seconds"] = active.get("elapsed_seconds", 0.) + max(0., spent)
            active["attempt_started"] = None
            save_state(folder, state)
        job_file = report.parent.parent / "job.json"
        if not report.is_file() and job_file.is_file() and not (folder / "stop.json").is_file():
            if state["contract"]["training_budget"] == "full" and not allow_full:
                raise ValueError("续跑全预算实验仍须当前授权与 --allow-full")
            return execute_worker(folder, state, state["active"], read(job_file))
        return finalize(folder, state, state["active"])
    if (folder / "stop.json").is_file():
        state.update(status="stopped", stop_reason=read(folder / "stop.json")["reason"])
        save_state(folder, state)
    if state["status"] != "ready" or not state["queue"]:
        return {"status": state["status"], "reason": state.get("stop_reason")}
    if len(state["records"]) >= state["budget"]["experiments"]:
        state.update(status="stopped", stop_reason="budget_exhausted")
        save_state(folder, state)
        return {"status": "stopped", "reason": "budget_exhausted"}
    if state["contract"]["training_budget"] == "full" and not allow_full:
        raise ValueError("全预算须在每次调用中附加 --allow-full，并确认当前任务真人授权仍然适用")
    return launch(folder, state)


def launch(folder, state):
    while state["queue"]:
        proposal = search.ranked(state["queue"])[0]
        parent_record = search.resolve_parent(state, proposal.get("parent", "best"))
        references = {r["id"]: r for r in state["records"]
                      if r["id"] == state.get("baseline") or parent_record and r["id"] == parent_record["id"]}
        for record in references.values():
            errors = record_errors(record)
            if errors:
                raise ValueError("基线或父方案证据损坏，拒绝继续评测：" + "; ".join(errors))
        parent = CODE / (parent_record["id"] if parent_record else state["campaign"]) / "源码"
        origin = Path(proposal.get("source_dir") or parent).resolve()
        if not origin.is_relative_to(CODE.resolve()):
            raise ValueError("候选源码只能来自 tmp_main")
        if proposal.get("source_manifest") and any(digest(origin / n) != v for n, v in proposal["source_manifest"].items()):
            raise ValueError("入队后的候选源码已改变，请重新提出候选")
        fingerprint = search.signature(origin, EDITABLE, proposal["overrides"])
        reason = None
        if not proposal.get("baseline"):
            if fingerprint == search.signature(parent, EDITABLE):
                reason = "NO_CHANGE_FROM_PARENT"
            elif any(r.get("candidate_signature") == fingerprint for r in state["records"]):
                reason = "ALREADY_EVALUATED"
        if reason is None:
            break
        state["queue"].remove(proposal)
        skipped = {"proposal_id": proposal.get("proposal_id"), "hypothesis": proposal["hypothesis"], "reason": reason,
                   "parent_id": parent_record["id"] if parent_record else None}
        state.setdefault("skipped", []).append(skipped)
        save_state(folder, state, {"event": "skipped", **skipped})
    else:
        state.update(status="needs_proposal", stop_reason="queue_empty")
        save_state(folder, state)
        return {"status": "needs_proposal", "reason": "queue_empty"}
    name = identifier(f"{state['campaign']}_e{len(state['records']):03d}")
    source, result = CODE / name / "源码", RESULTS / name
    names = tuple(state["source_manifest"])
    for file in names:
        if file not in EDITABLE and digest(origin / file) != state["source_manifest"][file]:
            raise ValueError("候选修改了冻结的校验、入口或规则：" + file)
    worker = CODE / state["campaign"] / "evaluate.py"
    if digest(worker) != state["evaluator_sha256"]:
        raise ValueError("评估器在任务期间改变，请新建比较任务")
    if source.exists() or result.exists():
        raise ValueError("实验标识已存在但没有准备状态，拒绝覆盖孤立产物")
    state["queue"].remove(proposal)
    active = {"id": name, **proposal, "pid": None, "started": now(), "candidate_signature": fingerprint,
              "parent_id": parent_record["id"] if parent_record else None, "phase": "preparing",
              "origin": str(origin), "origin_manifest": {n: digest(origin / n) for n in names},
              "parent_source": str(parent), "prepare_attempts": 0, "attempts": 0, "elapsed_seconds": 0.}
    state.update(active=active, status="running")
    save_state(folder, state, {"event": "started", "id": name, "hypothesis": proposal["hypothesis"]})
    return prepare_active(folder, state)


def prepare_active(folder, state):
    active = state["active"]
    if active.get("prepare_attempts", 0) > state["budget"].get("max_restarts", 2):
        return fail_active(folder, state, active, "PREPARATION_RESTART_LIMIT")
    active["prepare_attempts"] = active.get("prepare_attempts", 0) + 1
    save_state(folder, state)
    source, result = CODE / active["id"] / "源码", RESULTS / active["id"]
    origin = Path(active["origin"])
    if any(digest(origin / n) != value for n, value in active["origin_manifest"].items()):
        return fail_active(folder, state, active, "PREPARATION_SOURCE_CHANGED")
    # 准备阶段尚未启动评测，重做复制可安全修复半写入快照，覆盖始终限于本实验。
    source.mkdir(parents=True, exist_ok=True)
    for name in active["origin_manifest"]:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin / name, target)
    apply_overrides(source, active["overrides"])
    manifest = {n: digest(source / n) for n in active["origin_manifest"]}
    result.joinpath("报告").mkdir(parents=True, exist_ok=True)
    job = {"id": active["id"], "track": state["track"], "contract": state["contract"], "source": str(source),
           "reference_source": str(CODE / state["campaign"] / "源码"), "output": str(result), "manifest": manifest,
           "parent_source": active["parent_source"],
           "parent_output": str(RESULTS / active["parent_id"]) if active.get("parent_id") else None,
           "baseline_output": str(RESULTS / state["baseline"]) if state["baseline"] else None,
           "resource_cache": str(RESULTS / state["baseline"] / "模型" / "resource") if state["baseline"] else None}
    write(result / "job.json", job)
    active.update(phase="prepared", job_sha256=digest(result / "job.json"))
    save_state(folder, state)
    return execute_worker(folder, state, active, job)


def execute_worker(folder, state, active, job):
    result = Path(job["output"])
    source = Path(job["source"])
    worker = CODE / state["campaign"] / "evaluate.py"
    if digest(worker) != state["evaluator_sha256"]:
        raise ValueError("冻结评估器改变，拒绝续跑")
    if active.get("attempts", 0) > state["budget"].get("max_restarts", 2):
        return fail_active(folder, state, active, "EXPERIMENT_RESTART_LIMIT")
    remaining = state["budget"]["timeout_seconds"] - active.get("elapsed_seconds", 0.)
    if remaining <= 0:
        return fail_active(folder, state, active, "EXPERIMENT_TIMEOUT")
    job_file = result / "job.json"
    if active.get("job_sha256") and digest(job_file) != active["job_sha256"]:
        return fail_active(folder, state, active, "JOB_CONTRACT_CHANGED")
    active.update(attempts=active.get("attempts", 0) + 1, attempt_started=now(), phase="launching")
    save_state(folder, state)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", AIC_DATA_ROOT=str(ROOT / "data"),
               AIC_HOLDER_VERSION=state["contract"]["holder_version"])
    env.update({k: str(state["contract"]["threads"]) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    tick = time.monotonic()
    with (result / "报告" / "运行日志.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(worker), "--job", str(result / "job.json")],
                                   cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        active["pid"] = process.pid
        active["phase"] = "running"
        save_state(folder, state)
        try:
            exit_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            active["elapsed_seconds"] += time.monotonic() - tick
            active["attempt_started"] = None
            return fail_active(folder, state, active, "EXPERIMENT_TIMEOUT")
    active["elapsed_seconds"] = active.get("elapsed_seconds", 0.) + time.monotonic() - tick
    active["attempt_started"] = None
    report_path = result / "报告" / "运行结果.json"
    if exit_code != 0 and report_path.is_file() and read(report_path).get("status") == "COMPLETE":
        return fail_active(folder, state, active, "WORKER_EXIT_" + str(exit_code))
    return finalize(folder, state, active)


def audit(folder, state):
    def matches(path, value):
        return path.is_file() and digest(path) == value
    checks = {"baseline_source": all(matches(CODE / state["campaign"] / "源码" / n, value) for n, value in state["source_manifest"].items()),
              "evaluator": matches(CODE / state["campaign"] / "evaluate.py", state["evaluator_sha256"])}
    rows = []
    for record in state["records"]:
        errors = record_errors(record)
        rows.append({"id": record["id"], "passed": not errors, "errors": errors})
    eligible = {r["id"]: r["primary"] for r in state["records"] if r.get("eligible")}
    checks["best_is_eligible"] = state["best"] is None or eligible.get(state["best"]["id"]) == state["best"]["primary"]
    passed = all(checks.values()) and all(r["passed"] for r in rows)
    return {"passed": passed, "status": "AUDIT_PASSED" if passed else "AUDIT_FAILED", "checks": checks,
            "records": rows, "active": state.get("active"), "training_started": False,
            "evidence_coverage": {"bound_records": sum(bool(r.get("job_sha256")) for r in state["records"]),
                                  "unbound_records": sum(not r.get("job_sha256") for r in state["records"])}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="AIC隔离自动优化：实验、评估与断点续跑")
    parser.add_argument("--campaign", help="指定任务；省略时使用活动任务")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="冻结当前正式源码并建立研究任务，不训练")
    init.add_argument("--track", choices=TRACKS, default="load")
    init.add_argument("--window-script", type=Path, help="可选的项目窗口定义；省略时使用本项目默认契约")
    init.add_argument("--windows", default="all", help="all或脚本窗口名的逗号分隔列表")
    init.add_argument("--train-end")
    init.add_argument("--start")
    init.add_argument("--end")
    init.add_argument("--training-budget", choices=("quick", "full"), default="quick")
    init.add_argument("--authorization", default="")
    init.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    init.add_argument("--max-experiments", type=int, default=12)
    init.add_argument("--patience", type=int, default=6)
    init.add_argument("--timeout-seconds", type=int, default=1800)
    init.add_argument("--target", type=float, help="可选的本地指标目标，不设置默认目标分数")
    init.add_argument("--max-restarts", type=int, default=2, help="同一实验中断后的最多恢复次数")
    sub.add_parser("status", help="查看任务、预算、最佳方案与下一候选")
    sub.add_parser("brief", help="提取供任意Agent接续的搜索状态、失败经验和主要退化")
    sub.add_parser("audit", help="独立核对任务、结果、源码与产物，不启动评测")
    one = sub.add_parser("step", help="完成一个实验或接收上次中断后的结果")
    one.add_argument("--dry-run", action="store_true")
    one.add_argument("--allow-full", action="store_true")
    run = sub.add_parser("run", help="在已设总预算内顺序执行")
    run.add_argument("--steps", type=int, default=1)
    run.add_argument("--allow-full", action="store_true")
    propose = sub.add_parser("propose", help="追加Agent提出的参数或代码候选")
    propose.add_argument("--hypothesis", required=True)
    propose.add_argument("--overrides", default="{}", help="点分隔配置路径对应的JSON对象")
    propose.add_argument("--source-dir", type=Path)
    propose.add_argument("--parent", default="best", help="best、baseline或本任务某个已完成实验标识")
    propose.add_argument("--rationale", default="", help="候选依据与可否证预期")
    propose.add_argument("--expected-gain", type=float, default=0.)
    propose.add_argument("--confidence", type=float, default=.5)
    propose.add_argument("--cost-minutes", type=float, default=1.)
    propose.add_argument("--ref", dest="refs", action="append", default=[])
    drop = sub.add_parser("drop", help="撤回尚未运行的候选，保留原因且不消耗评测次数")
    drop.add_argument("--id", required=True, help="brief或status中的proposal_id")
    drop.add_argument("--reason", required=True)
    stop = sub.add_parser("stop", help="停止继续启动实验，不终止现有进程")
    stop.add_argument("--reason", default="human_stopped")
    args = parser.parse_args(argv)
    if args.command == "init":
        if not 1 <= args.threads <= min(16, os.cpu_count() or 1) or min(args.max_experiments, args.patience, args.timeout_seconds) < 1 or args.max_restarts < 0:
            parser.error("线程或预算超出有效范围")
        if args.target is not None and not math.isfinite(args.target):
            parser.error("目标须为有限数")
        return initialize(args)
    folder = campaign_path(args.campaign)
    if args.command == "status":
        state = read(folder / "state.json")
        if (folder / "stop.json").is_file():
            state["stop_requested"] = read(folder / "stop.json")
        return state
    if args.command == "brief":
        state = read(folder / "state.json")
        latest = read(state["records"][-1]["report"]) if state["records"] else None
        return search.brief(state, latest)
    if args.command == "audit":
        with lock(folder):
            return audit(folder, read(folder / "state.json"))
    if args.command == "stop":
        write(folder / "stop.json", {"reason": args.reason, "requested": now()})
        return {"status": "stop_requested", "reason": args.reason}
    with lock(folder):
        state = read(folder / "state.json")
        if args.command == "drop":
            proposal = next((p for p in state["queue"] if p.get("proposal_id") == args.id), None)
            if proposal is None:
                raise ValueError("只能撤回队列中尚未运行的候选")
            state["queue"].remove(proposal)
            skipped = {"proposal_id": args.id, "hypothesis": proposal["hypothesis"], "reason": "WITHDRAWN: " + args.reason}
            state.setdefault("skipped", []).append(skipped)
            if not state["queue"] and state["status"] == "ready":
                state.update(status="needs_proposal", stop_reason="queue_empty")
            save_state(folder, state, {"event": "withdrawn", **skipped})
            return skipped
        if args.command == "propose":
            patch = json.loads(args.overrides)
            overrides_valid(patch)
            proposal = {"hypothesis": args.hypothesis, "overrides": patch, "baseline": False, "parent": args.parent,
                        "rationale": args.rationale, "expected_gain": args.expected_gain, "confidence": args.confidence,
                        "cost_minutes": args.cost_minutes, "refs": args.refs}
            search.priority(proposal)
            search.resolve_parent(state, args.parent)
            if args.source_dir:
                source = args.source_dir.resolve()
                if not source.is_relative_to(CODE.resolve()):
                    raise ValueError("代码候选须放 tmp_main 下")
                proposal["source_dir"] = str(source)
                proposal["source_manifest"] = {n: digest(source / n) for n in state["source_manifest"]}
            if any(p.get("overrides") == patch and p.get("source_dir") == proposal.get("source_dir")
                   and p.get("parent", "best") == args.parent for p in state["queue"]):
                raise ValueError("候选已在队列中")
            counter = state.get("proposal_counter", len(state["queue"]) + len(state["records"]) + len(state.get("skipped", [])))
            proposal["proposal_id"] = f"h{counter:04d}"
            state["proposal_counter"] = counter + 1
            state["queue"].append(proposal)
            if state["status"] == "needs_proposal":
                state.update(status="ready", stop_reason=None)
            save_state(folder, state, {"event": "proposed", **proposal})
            return proposal
        count = getattr(args, "steps", 1)
        if count < 1:
            raise ValueError("steps必须为正整数")
        results = []
        for _ in range(count):
            results.append(step(folder, state, getattr(args, "dry_run", False), args.allow_full))
            if state["status"] != "ready" or getattr(args, "dry_run", False):
                break
        return results


def cli():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        result = main()
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        rows = result if isinstance(result, list) else [result]
        failed = any(isinstance(row, dict) and (row.get("status") in ("AUDIT_FAILED", "FAILED")
                     or row.get("outcome") in ("failed", "invalid")) for row in rows)
        return 2 if failed else 0
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(cli())
