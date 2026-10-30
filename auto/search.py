"""独立于训练后端的候选排序、分支与去重，不导入项目模型。"""
import ast
from copy import deepcopy
import hashlib
import json
import math


def priority(proposal):
    gain = proposal.get("expected_gain", 0.)
    confidence = proposal.get("confidence", .5)
    cost = proposal.get("cost_minutes", 1.)
    if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in (gain, confidence, cost)) or gain < 0 or not 0 <= confidence <= 1 or cost <= 0):
        raise ValueError("预期改善须非负、信心在0至1之间、预计耗时须为正数，且均为有限数")
    return float(gain * confidence / cost)


def ranked(queue):
    # 基线必须先建立；相同优先级按提议顺序稳定执行。
    return sorted(queue, key=lambda p: (not p.get("baseline", False), -priority(p)))


def resolve_parent(state, selector="best"):
    if selector == "best":
        key = state["best"]["id"] if state.get("best") else state.get("baseline")
    elif selector == "baseline":
        key = state.get("baseline")
    else:
        key = selector
    if key is None:
        return None
    row = next((r for r in state["records"] if r["id"] == key), None)
    if row is None or row.get("status") != "COMPLETE":
        raise ValueError("父方案必须来自本任务已完成的实验")
    return row


def _path(target):
    keys = []
    while isinstance(target, ast.Subscript):
        if not isinstance(target.slice, ast.Constant) or not isinstance(target.slice.value, str):
            return None
        keys.insert(0, target.slice.value)
        target = target.value
    return [target.id, *keys] if isinstance(target, ast.Name) else None


def _update(dictionary, keys, value):
    if not isinstance(dictionary, ast.Dict):
        return False
    found = next((i for i, key in enumerate(dictionary.keys)
                  if isinstance(key, ast.Constant) and key.value == keys[0]), None)
    if len(keys) == 1:
        if found is None:
            dictionary.keys.append(ast.Constant(keys[0]))
            dictionary.values.append(deepcopy(value))
        else:
            dictionary.values[found] = deepcopy(value)
        return True
    return found is not None and _update(dictionary.values[found], keys[1:], value)


def _configuration(text, overrides):
    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            target = parts[0] + "".join(f"[{p!r}]" for p in parts[1:])
            text += f"\n{target} = {value!r}\n"
    tree = ast.parse(text)
    roots, body, observed = {}, [], set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                roots[target.id] = node.value
                observed.discard(target.id)
            elif isinstance(target, ast.Subscript):
                path = _path(target)
                try:
                    ast.literal_eval(node.value)
                    literal = True
                except (ValueError, TypeError):
                    literal = False
                # 已经被其他语句读取的配置不能前移覆盖，避免把有运行差异的程序误判重复。
                if path and path[0] in roots and path[0] not in observed and literal:
                    if _update(roots[path[0]], path[1:], node.value):
                        continue
        observed.update(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
        body.append(node)
    tree.body = body
    return ast.dump(tree, include_attributes=False)


def signature(source, names, overrides=None):
    """按语法与最终静态配置去重；不执行候选配置，不读取赛事数据。"""
    content = {}
    for name in sorted(names):
        text = (source / name).read_text(encoding="utf-8")
        content[name] = (_configuration(text, overrides or {}) if name == "config.py"
                         else ast.dump(ast.parse(text), include_attributes=False))
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def brief(state, latest_report=None):
    latest_report = latest_report or {}
    deltas = latest_report.get("selection", {}).get("slice_deltas", [])
    regressions = sorted((d for d in deltas if d.get("regression")), key=lambda d: -abs(d["delta"]))[:10]
    target = state.get("target")
    score = state["best"]["primary"] if state.get("best") else None
    if state["status"] == "stopped":
        action = "stop"
    elif state.get("active"):
        action = "inspect_or_resume"
    elif state["status"] == "needs_proposal":
        action = "propose"
    else:
        action = "step"
    return {"campaign": state["campaign"], "status": state["status"], "next_action": action,
            "contract": state["contract"], "remaining_experiments": max(0, state["budget"]["experiments"] - len(state["records"]) - int(bool(state.get("active")))),
            "best": state.get("best"), "target": target,
            "target_gap": max(0., target - score) if target is not None and score is not None else None,
            "stagnation": state.get("stagnation", 0), "stop_reason": state.get("stop_reason"),
            "active": state.get("active"), "next_proposals": ranked(state["queue"])[:5],
            "recent_experiments": state["records"][-5:],
            "known_rejections": [{"id": r["id"], "hypothesis": r["hypothesis"], "parent_id": r.get("parent_id"),
                                  "error": r.get("error"), "checks": r.get("checks"), "primary": r.get("primary")}
                                 for r in state["records"] if not r.get("improved")][-8:],
            "largest_slice_regressions": regressions, "skipped_proposals": state.get("skipped", [])[-8:]}
