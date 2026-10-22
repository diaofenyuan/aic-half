"""三气种资源预测与发电调度 MILP。

边界说明：
- 高炉气柜只有 2 号柜数据可用，焦炉/转炉没有独立柜位，不虚构库存。
- 热值/效率没有官方数值，只由历史 ``generator_all`` 与三气种耗气做正系数回归，
  标记为 inferred；正式提交前必须按官方数据字典复核。
- 当前没有官方 ``opt_result.csv`` 模板，因此只生成明确标记的模拟调度表。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix, vstack
from sklearn.linear_model import LinearRegression

import config as C
from data_cleaning import (PreparedData, build_base_features, resource_series,
                           target_calendar)


@dataclass
class ResourceBundle:
    models: dict
    final_models: dict
    feature_names: list[str]
    # P2：分位数资源模型，key 为分位标签（p10/p50），value 为 gas->booster。
    quantile_models: dict = field(default_factory=dict)
    final_quantile_models: dict = field(default_factory=dict)
    quantile_metrics: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    train_rows: int = 0
    train_seconds: float = 0.0
    heat_rate: dict = field(default_factory=dict)
    blocker: str | None = None


def _resource_base(prepared: PreparedData) -> pd.DataFrame:
    return build_base_features(prepared, include_targets=False)


def _resource_design(prepared: PreparedData, base: pd.DataFrame,
                     positions: np.ndarray, horizons: tuple[int, ...]):
    cal = target_calendar(prepared, positions, horizons)
    names = list(base.columns) + list(cal.keys()) + ["h"]
    return cal, names


def _stack(prepared: PreparedData, base_values: np.ndarray, cal: dict,
           names: list[str], positions: np.ndarray, horizons: tuple[int, ...],
           series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(positions)
    width = base_values.shape[1] + len(cal) + 1
    X = np.empty((n * len(horizons), width), dtype=np.float32)
    y = np.empty(n * len(horizons), dtype=np.float64)
    for j, h in enumerate(horizons):
        lo = j * n
        parts = [base_values[positions]]
        for key in cal:
            parts.append(cal[key][:, j][:, None])
        parts.append(np.full((n, 1), float(h), dtype=np.float32))
        X[lo:lo + n] = np.hstack(parts)
        y[lo:lo + n] = series[positions + int(h)]
    valid = np.isfinite(y)
    return X[valid], y[valid]


def _fit_resource_models(prepared: PreparedData, positions: np.ndarray,
                         num_round: int, ref_pos: int,
                         alpha: float | None = None) -> tuple[dict, list[str]]:
    base = _resource_base(prepared)
    base_values = base.to_numpy(dtype=np.float32)
    cal, names = _resource_design(prepared, base, positions, C.HORIZONS_L)
    series = resource_series(prepared)
    age = (ref_pos - positions[:, None]) / C.STEPS_PER_DAY
    decay = np.power(0.5, np.maximum(age, 0.0) / float(C.DECAY_DAYS or 60.0)).ravel()
    decay = decay / max(float(decay.mean()), 1e-9)
    params = {**C.LGBM_BASE, "num_threads": C.LGBM_BASE["num_threads"]}
    if alpha is not None:
        # 分位数回归：alpha 即目标分位；仍用 L1 损失族，便于复用同一特征栈。
        params = {**params, "objective": "quantile", "alpha": float(alpha),
                  "metric": "quantile"}
    models = {}
    for gas in C.RESOURCE_NAMES:
        X, y = _stack(prepared, base_values, cal, names, positions,
                      C.HORIZONS_L, series[gas].to_numpy(dtype=float))
        weights = np.repeat(decay, len(C.HORIZONS_L))
        # 与 X 的有效标签过滤保持同一顺序。
        mask = np.isfinite(series[gas].to_numpy(dtype=float)[
            positions[:, None] + np.asarray(C.HORIZONS_L)[None, :]]).ravel()
        models[gas] = lgb.train(params, lgb.Dataset(
            X, label=y, weight=weights[mask], feature_name=names, free_raw_data=True),
            num_boost_round=num_round)
    return models, names


def forecast_resources(bundle: ResourceBundle, prepared: PreparedData,
                       origins: pd.DatetimeIndex, use_final: bool = False) -> np.ndarray:
    positions = prepared.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("资源预测起报点不在数据网格内")
    base = _resource_base(prepared)
    base_values = base.to_numpy(dtype=np.float32)
    cal, names = _resource_design(prepared, base, positions, C.HORIZONS_L)
    n = len(positions)
    width = base_values.shape[1] + len(cal) + 1
    X = np.empty((n * len(C.HORIZONS_L), width), dtype=np.float32)
    for j, h in enumerate(C.HORIZONS_L):
        lo = j * n
        parts = [base_values[positions]]
        for key in cal:
            parts.append(cal[key][:, j][:, None])
        parts.append(np.full((n, 1), float(h), dtype=np.float32))
        X[lo:lo + n] = np.hstack(parts)
    out = np.empty((n, len(C.HORIZONS_L), len(C.RESOURCE_NAMES)))
    models = bundle.final_models if use_final else bundle.models
    for gi, gas in enumerate(C.RESOURCE_NAMES):
        pred = models[gas].predict(X).reshape(n, len(C.HORIZONS_L))
        out[:, :, gi] = np.maximum(pred, 0.0)
    return out


def forecast_resources_quantile(bundle: ResourceBundle, prepared: PreparedData,
                                origins: pd.DatetimeIndex, label: str,
                                use_final: bool = False) -> np.ndarray:
    """按分位标签预测三气种可用气，形状与 forecast_resources 一致。"""
    pool = bundle.final_quantile_models if use_final else bundle.quantile_models
    if label not in pool:
        raise KeyError(f"缺少分位模型: {label}")
    positions = prepared.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("资源预测起报点不在数据网格内")
    base = _resource_base(prepared)
    base_values = base.to_numpy(dtype=np.float32)
    cal, names = _resource_design(prepared, base, positions, C.HORIZONS_L)
    n = len(positions)
    width = base_values.shape[1] + len(cal) + 1
    X = np.empty((n * len(C.HORIZONS_L), width), dtype=np.float32)
    for j, h in enumerate(C.HORIZONS_L):
        lo = j * n
        parts = [base_values[positions]]
        for key in cal:
            parts.append(cal[key][:, j][:, None])
        parts.append(np.full((n, 1), float(h), dtype=np.float32))
        X[lo:lo + n] = np.hstack(parts)
    out = np.empty((n, len(C.HORIZONS_L), len(C.RESOURCE_NAMES)))
    calib = (bundle.quantile_metrics or {}).get(label, {}).get("scale", {})
    for gi, gas in enumerate(C.RESOURCE_NAMES):
        pred = pool[label][gas].predict(X).reshape(n, len(C.HORIZONS_L))
        scale = float(calib.get(gas, 1.0))
        out[:, :, gi] = np.maximum(pred * scale, 0.0)
    return out


def estimate_heat_rate(prepared: PreparedData, end_pos: int) -> dict:
    frame = prepared.frame.iloc[:end_pos]
    y = prepared.labels["generator_all"].iloc[:end_pos].to_numpy(dtype=float)
    X = frame[["d_use_blast", "d_use_coke", "d_use_converter"]].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if int(valid.sum()) < 1000:
        return {"ok": False, "reason": "INSUFFICIENT_HISTORY"}
    model = LinearRegression(positive=True, fit_intercept=False).fit(X[valid], y[valid])
    pred = model.predict(X[valid])
    ss_res = float(np.sum((y[valid] - pred) ** 2))
    ss_tot = float(np.sum((y[valid] - y[valid].mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    coef = {gas: float(v) for gas, v in zip(C.RESOURCE_NAMES, model.coef_)}
    ok = bool(r2 >= C.HEAT_RATE["min_r2"] and min(coef.values()) > 1e-7)
    return {"ok": ok, "coefficients_mw_per_m3h": coef, "r2": r2,
            "valid_rows": int(valid.sum()), "method": C.HEAT_RATE["method"],
            "note": "由历史运行标定，非官方热值/效率；正式提交前需复核"}


def train_resources(prepared: PreparedData, folds: list[dict],
                    quick: bool = False) -> ResourceBundle:
    started = time.time()
    confirm = next(f for f in folds if f["role"] == "confirm")
    masked = prepared.masked_copy(
        confirm["start"],
        pd.Timestamp(confirm["start"]) + pd.Timedelta(days=int(confirm["days"])))
    positions, eval_positions, ref_pos = _resource_positions(masked, confirm, quick)
    num_round = C.LGBM_TREES_QUICK if quick else C.LGBM_TREES_LONG
    models, names = _fit_resource_models(masked, positions, num_round, ref_pos)

    final_positions = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY),
                                len(prepared.index) - max(C.HORIZONS_L))
    final_positions = final_positions[::4 if quick else C.TRAIN_STRIDE_LONG]
    final_ref = len(prepared.index) - max(C.HORIZONS_L)
    final_models, _ = _fit_resource_models(prepared, final_positions, num_round, final_ref)

    bundle = ResourceBundle(models=models, final_models=final_models,
                            feature_names=names,
                            train_rows=int(len(positions) * len(C.HORIZONS_L)),
                            train_seconds=time.time() - started)
    origins = prepared.index[eval_positions]
    forecast = forecast_resources(bundle, masked, origins)
    truth = resource_series(masked).to_numpy(dtype=float)[
        np.asarray(eval_positions)[:, None] + np.asarray(C.HORIZONS_L)[None, :]]
    metrics = {}
    for gi, gas in enumerate(C.RESOURCE_NAMES):
        actual = truth[:, :, gi]
        estimate = forecast[:, :, gi]
        valid = np.isfinite(actual) & (actual > 0)
        rel = np.abs(estimate[valid] - actual[valid]) / actual[valid]
        wape = np.abs(estimate[valid] - actual[valid]).sum() / max(actual[valid].sum(), 1e-9)
        metrics[gas] = {"accuracy_1_mape": float(100 * (1 - np.mean(rel))) if valid.any() else None,
                        "wape": float(wape)}
    bundle.metrics = metrics

    # P2：在同一特征栈上拟合分位数模型；覆盖率应接近目标分位 alpha。
    q_round = max(80, num_round // 2)
    quantile_models, final_quantile_models, quantile_metrics = {}, {}, {}
    for alpha in (C.OPT_CONFIG.get("resource_quantiles") or (0.10,)):
        label = f"p{int(round(float(alpha) * 100)):02d}"
        q_models, _ = _fit_resource_models(
            masked, positions, q_round, ref_pos, alpha=float(alpha))
        q_final, _ = _fit_resource_models(
            prepared, final_positions, q_round, final_ref, alpha=float(alpha))
        quantile_models[label] = q_models
        final_quantile_models[label] = q_final
        bundle.quantile_models = quantile_models
        bundle.final_quantile_models = final_quantile_models
        q_pred = forecast_resources_quantile(bundle, masked, origins, label)
        coverage = {}
        scales = {}
        for gi, gas in enumerate(C.RESOURCE_NAMES):
            actual = truth[:, :, gi]
            estimate = q_pred[:, :, gi]
            valid = np.isfinite(actual) & np.isfinite(estimate)
            if not valid.any():
                continue
            coverage[gas] = float(np.mean(actual[valid] <= estimate[valid]))
            # 覆盖率偏差用分位数比值校准：真实 alpha 分位 / 预测 alpha 分位。
            q_true = float(np.quantile(actual[valid], float(alpha)))
            q_predv = float(np.quantile(estimate[valid], float(alpha)))
            scales[gas] = float(np.clip(q_true / max(q_predv, 1e-9), 0.5, 2.0))
        quantile_metrics[label] = {"alpha": float(alpha), "coverage": coverage,
                                   "scale": scales}
        # 校准后的覆盖率（仅诊断）
        for gi, gas in enumerate(C.RESOURCE_NAMES):
            if gas not in scales:
                continue
            a, e = truth[:, :, gi], q_pred[:, :, gi] * scales[gas]
            v = np.isfinite(a) & np.isfinite(e)
            if v.any():
                quantile_metrics[label].setdefault("coverage_calibrated", {})[gas] = (
                    float(np.mean(a[v] <= e[v])))
    bundle.quantile_models = quantile_models
    bundle.final_quantile_models = final_quantile_models
    bundle.quantile_metrics = quantile_metrics

    bundle.heat_rate = estimate_heat_rate(prepared, ref_pos)
    if not bundle.heat_rate.get("ok"):
        bundle.blocker = "PHYSICAL_MAPPING_UNCERTAIN"
    if C.GAS_HOLDER["column"] not in prepared.frame.columns:
        bundle.blocker = "MISSING_GAS_HOLDER_MAPPING"
    return bundle


def _resource_positions(prepared: PreparedData, fold: dict,
                        quick: bool) -> tuple[np.ndarray, np.ndarray, int]:
    start = pd.Timestamp(fold["start"])
    train_end = pd.Timestamp(fold["train_end"])
    ref_pos = int(prepared.index.get_indexer([train_end])[0])
    positions = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY),
                          max(ref_pos - max(C.HORIZONS_L) + 1, 0))
    positions = positions[::4 if quick else C.TRAIN_STRIDE_LONG]
    end = start + pd.Timedelta(days=int(fold["days"])) - pd.Timedelta(minutes=C.STEP_MINUTES)
    origins = pd.date_range(start, end, freq=f"{C.STEP_MINUTES}min")
    eval_positions = prepared.index.get_indexer(origins)
    return positions, eval_positions, ref_pos


# ---------------------------------------------------------------------------
# 电价
# ---------------------------------------------------------------------------
def load_prices(config=C) -> dict[int, np.ndarray]:
    path = config.DATA_ROOT / config.OPT_CONFIG["price_file"]
    if not path.is_file():
        return {}
    raw = pd.read_excel(path, header=0)
    time_col = raw.columns[0]
    month_cols = {}
    for col in raw.columns[1:]:
        match = re.match(r"(\d+)月", str(col).strip())
        if match:
            month_cols[int(match.group(1))] = col
    prices = {}
    for month, col in month_cols.items():
        slots = np.full(96, np.nan)
        for _, row in raw.iterrows():
            text = str(row[time_col]).strip()
            start = text.split("-")[0]
            match = re.match(r"(\d+):(\d+)", start)
            if not match:
                continue
            slot = int(match.group(1)) * 4 + int(match.group(2)) // 15
            value = pd.to_numeric(row[col], errors="coerce")
            if np.isfinite(value):
                for k in range(2):
                    if 0 <= slot + k < 96:
                        slots[slot + k] = float(value)
        if np.isfinite(slots).all():
            prices[month] = slots
    return prices


def _price_vector(prices: dict[int, np.ndarray], start: pd.Timestamp, n: int) -> np.ndarray:
    template = prices.get(start.month)
    if template is None:
        return np.zeros(n)
    slots = [((start + pd.Timedelta(minutes=C.STEP_MINUTES * k)).hour * 4
              + (start + pd.Timedelta(minutes=C.STEP_MINUTES * k)).minute // 15) for k in range(n)]
    return template[np.asarray(slots)]


def _tou_periods(price: np.ndarray, config=C) -> list[str]:
    """按当日电价分档为谷/平/峰/尖；电价表只有两档时自动退化。"""
    levels = sorted(set(np.round(price[np.isfinite(price)], 6).tolist()))
    if not levels:
        return ["flat"] * len(price)
    labels = []
    n = len(levels)
    for value in np.round(price, 6):
        idx = levels.index(value) if value in levels else 0
        if n == 1:
            labels.append("flat")
        elif n == 2:
            labels.append("valley" if idx == 0 else "peak")
        elif n == 3:
            labels.append(("valley", "flat", "peak")[idx])
        else:
            # 四档及以上：最低谷、最高尖、中间对分平/峰。
            labels.append(("valley", "flat", "peak", "sharp")[min(idx, 3)])
    return labels


def _recent_lower(actual: np.ndarray, origin_pos: int, n_steps: int) -> np.ndarray:
    """未来每一步同槽位、过去 N 天的低分位余气，仅使用起报点之前的数据。"""
    days = int(C.OPT_CONFIG["robust_history_days"])
    quantile = float(C.OPT_CONFIG["robust_low_quantile"])
    out = np.full(n_steps, np.nan)
    for k in range(n_steps):
        target = origin_pos + k + 1
        indices = target - C.STEPS_PER_DAY * np.arange(1, days + 1)
        indices = indices[indices >= 0]
        values = actual[indices]
        values = values[np.isfinite(values)]
        if len(values) >= 3:
            out[k] = float(np.quantile(values, quantile))
    return out


# ---------------------------------------------------------------------------
# MILP
# ---------------------------------------------------------------------------
def _last_level(prepared: PreparedData, target: str, pos: int) -> float:
    values = prepared.frame[C.TARGET_COLUMN[target]].iloc[:pos + 1].to_numpy(dtype=float)
    valid = np.where(np.isfinite(values))[0]
    return float(values[valid[-1]]) if len(valid) else float("nan")


def _solve_day(avail: np.ndarray, price: np.ndarray, holder0: float,
               initial_power: np.ndarray, caps: np.ndarray, heat_rate: dict,
               config=C) -> dict:
    """单日 96 步确定性 MILP；字典序：先最小放散，再最大化电价收益。"""
    n_h = len(price)
    units = config.OPT_CONFIG["unit_capacity_mw"]
    n_u = len(units)
    n_g = len(C.RESOURCE_NAMES)
    beta = np.array([heat_rate["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    capacity = config.GAS_HOLDER["capacity_m3"]
    real_low = capacity * config.GAS_HOLDER["safe_low_ratio"]
    real_high = capacity * config.GAS_HOLDER["safe_high_ratio"]
    margin = capacity * float(config.OPT_CONFIG["holder_margin_ratio"])
    low = real_low + margin
    high = real_high - margin
    holder0 = float(np.clip(holder0, real_low, real_high))

    off_z, off_p = 0, n_u * n_h
    off_q = off_p + n_u * n_h
    off_v = off_q + n_g * n_h
    off_s = off_v + n_g * n_h
    # GPF：柜位对偏好带的双向偏离量，压在目标里软惩罚。
    off_du = off_s + n_h
    off_dd = off_du + n_h
    # P4：每机组每步的启停切换指示（连续 [0,1] 即可，取到 0/1 由最小化目标保证）。
    off_sw = off_dd + n_h
    n_var = off_sw + n_u * n_h
    lb = np.zeros(n_var)
    ub = np.full(n_var, np.inf)
    integrality = np.zeros(n_var)
    for i in range(n_u):
        for k in range(n_h):
            ub[off_z + i * n_h + k] = 1.0
            integrality[off_z + i * n_h + k] = 1
            ub[off_p + i * n_h + k] = units[i]
    for g in range(n_g):
        for k in range(n_h):
            ub[off_q + g * n_h + k] = caps[g]
    for k in range(n_h):
        lb[off_s + k] = low
        ub[off_s + k] = high
        # 偏离量上界取安全区宽度，足够覆盖极端柜位。
        ub[off_du + k] = capacity
        ub[off_dd + k] = capacity
    for i in range(n_u):
        for k in range(n_h):
            ub[off_sw + i * n_h + k] = 1.0

    rows: list[tuple[int, dict[int, float], float, float]] = []

    def add(coeffs: dict[int, float], lower: float, upper: float) -> None:
        rows.append((len(rows), coeffs, lower, upper))

    for i in range(n_u):
        cap = units[i]
        for k in range(n_h):
            add({off_p + i * n_h + k: 1.0, off_z + i * n_h + k: -cap}, -np.inf, 0.0)
            add({off_p + i * n_h + k: -1.0,
                 off_z + i * n_h + k: config.OPT_CONFIG["min_load_ratio"] * cap}, -np.inf, 0.0)
    ramp = config.OPT_CONFIG["ramp_fraction_per_min"] * config.STEP_MINUTES
    for i in range(n_u):
        cap = units[i] * ramp
        for k in range(n_h):
            prev = initial_power[i] if k == 0 else None
            if k > 0:
                add({off_p + i * n_h + k: 1.0, off_p + i * n_h + k - 1: -1.0}, -np.inf, cap)
                add({off_p + i * n_h + k - 1: 1.0, off_p + i * n_h + k: -1.0}, -np.inf, cap)
            else:
                add({off_p + i * n_h: 1.0}, -np.inf, prev + cap)
                add({off_p + i * n_h: -1.0}, -np.inf, cap - prev)
    for k in range(n_h):
        coeffs = {off_s + k: 1.0}
        rhs = 0.25 * float(avail[k, 0])
        if k > 0:
            coeffs[off_s + k - 1] = -1.0
        else:
            rhs += holder0
        coeffs[off_q + 0 * n_h + k] = 0.25
        coeffs[off_v + 0 * n_h + k] = 0.25
        add(coeffs, rhs, rhs)
        # 气柜可提供的最大瞬时流量：不得超过当前库存高于下限的部分。
        draw = {off_q + 0 * n_h + k: 1.0, off_v + 0 * n_h + k: 1.0}
        if k > 0:
            draw[off_s + k - 1] = -4.0
            add(draw, -np.inf, float(avail[k, 0]) - 4.0 * low)
        else:
            add(draw, -np.inf, float(avail[k, 0]) - 4.0 * low + 4.0 * holder0)
    for g in (1, 2):
        for k in range(n_h):
            add({off_q + g * n_h + k: 1.0, off_v + g * n_h + k: 1.0},
                -np.inf, float(avail[k, g]))
    pref_low = capacity * float(config.OPT_CONFIG["holder_pref_low_ratio"])
    pref_high = capacity * float(config.OPT_CONFIG["holder_pref_high_ratio"])
    for k in range(n_h):
        # s[k] + d_under[k] >= pref_low
        add({off_s + k: 1.0, off_dd + k: 1.0}, pref_low, np.inf)
        # s[k] - d_over[k] <= pref_high
        add({off_s + k: 1.0, off_du + k: -1.0}, -np.inf, pref_high)
    for i in range(n_u):
        for k in range(n_h):
            if k == 0:
                # 首步相对初始状态的开关：这里从 0 计，避免依赖未给出的 z_{-1}
                continue
            zk = off_z + i * n_h + k
            zk1 = off_z + i * n_h + k - 1
            sw = off_sw + i * n_h + k
            add({sw: 1.0, zk: -1.0, zk1: 1.0}, 0.0, np.inf)   # sw >= z_{k-1}-z_k
            add({sw: 1.0, zk: 1.0, zk1: -1.0}, 0.0, np.inf)   # sw >= z_k-z_{k-1}
    if config.OPT_CONFIG.get("terminal_holder") and low <= holder0 <= high:
        add({off_s + n_h - 1: 1.0}, holder0, holder0)
    for k in range(n_h):
        coeffs = {off_p + i * n_h + k: 1.0 for i in range(n_u)}
        for g in range(n_g):
            coeffs[off_q + g * n_h + k] = -float(beta[g])
        add(coeffs, 0.0, 0.0)

    matrix = lil_matrix((len(rows), n_var))
    lower = np.empty(len(rows))
    upper = np.empty(len(rows))
    for idx, coeffs, lo, hi in rows:
        for var, value in coeffs.items():
            matrix[idx, var] = value
        lower[idx] = lo
        upper[idx] = hi
    matrix = csr_matrix(matrix)
    constraints = [LinearConstraint(matrix, lower, upper)]
    options = {"time_limit": float(config.OPT_CONFIG["solver_time_limit_s"]),
               "mip_rel_gap": float(config.OPT_CONFIG["mip_rel_gap"]),
               "disp": False}

    objective = np.zeros(n_var)
    for k in range(n_h):
        for i in range(n_u):
            objective[off_p + i * n_h + k] = -float(price[k]) * 0.25 * 1000.0
    # 放散罚金：气种×分时差异化（Zhao 2015）。
    tou = _tou_periods(price, config)
    table = config.OPT_CONFIG["vent_penalty_table"]
    for g, gas_name in enumerate(C.RESOURCE_NAMES):
        base = float(table.get(gas_name, {}).get(
            "flat", config.OPT_CONFIG["vent_penalty_cny_per_m3"]))
        for k in range(n_h):
            pen = float(table.get(gas_name, {}).get(tou[k], base))
            objective[off_v + g * n_h + k] = 0.25 * pen
    objective[off_q:off_v] = 1e-8
    # GPF：柜位贴偏好带，避免顶到安全边界。
    hold_pen = float(config.OPT_CONFIG["holder_penalty_cny_per_m3"])
    objective[off_du:off_dd] = 0.25 * hold_pen
    objective[off_dd:off_sw] = 0.25 * hold_pen
    switch_pen = float(config.OPT_CONFIG.get("unit_switch_penalty_cny", 0.0))
    objective[off_sw:n_var] = switch_pen
    solution = milp(objective, integrality=integrality, bounds=Bounds(lb, ub),
                    constraints=constraints, options=options)
    if not solution.success or solution.x is None:
        return {"status": "INFEASIBLE", "message": str(solution.message)}
    x = solution.x
    power = np.maximum(x[off_p:off_q].reshape(n_u, n_h), 0.0)
    gas = np.maximum(x[off_q:off_v].reshape(n_g, n_h), 0.0)
    vent = np.maximum(x[off_v:off_s].reshape(n_g, n_h), 0.0)
    holder = x[off_s:off_s + n_h]
    vent_total = float(np.sum(vent) * 0.25)
    holder_dev = (np.maximum(x[off_du:off_dd], 0.0)
                  + np.maximum(x[off_dd:off_sw], 0.0))
    return {"status": "OPTIMAL",
            "message": str(solution.message), "power": power, "gas": gas,
            "vent": vent, "holder": holder, "vent_total": vent_total,
            "holder_dev": holder_dev,
            "objective": float(solution.fun)}


def _fallback_day(prepared: PreparedData, pos: int, avail: np.ndarray,
                  caps: np.ndarray, heat_rate: dict, config=C) -> dict:
    units = np.asarray(config.OPT_CONFIG["unit_capacity_mw"], dtype=float)
    g1 = _last_level(prepared, "generator_1", pos)
    allp = _last_level(prepared, "generator_all", pos)
    if not np.isfinite(g1):
        g1 = 0.6 * config.CAPACITY["generator_1"]
    if not np.isfinite(allp):
        allp = 0.6 * config.CAPACITY["generator_all"]
    power = np.zeros((len(units), len(avail)))
    power[:4] = g1 / 4.0
    power[4:] = max(allp - g1, 0.0) / 2.0
    power = np.minimum(power, units[:, None])
    beta = np.array([heat_rate["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    gas = np.zeros((len(C.RESOURCE_NAMES), len(avail)))
    for k in range(len(avail)):
        need = float(power[:, k].sum())
        shares = avail[k] / max(float(avail[k].sum()), 1e-9)
        raw = shares * need / max(float(np.dot(shares, beta)), 1e-9)
        gas[:, k] = np.minimum(raw, avail[k]) * 0.98
    return {"status": "FALLBACK_HISTORICAL", "message": "MILP 未获可行解",
            "power": power, "gas": gas,
            "vent": np.maximum(avail.T - gas, 0.0),
            "holder": np.full(len(avail), np.nan), "vent_total": float("nan"),
            "objective": float("nan")}


def optimize(resource_forecast: np.ndarray, prepared: PreparedData,
             origins: pd.DatetimeIndex, long_pred: np.ndarray,
             resource_bundle: ResourceBundle, quick: bool = False,
             availability_mode: str | None = None,
             quantile_low: np.ndarray | None = None) -> dict:
    """按天滚动 24 小时调度；返回明确标记的模拟结果。

    ``availability_mode`` 覆盖全局配置，便于 point / robust_low 同批对比。
    ``quantile_low`` 为 p10 资源预测，robust_low 用它替代纯历史低分位。
    """
    if resource_bundle.blocker:
        return {"status": resource_bundle.blocker, "submittable": False,
                "schedule": None, "summary": {"reason": resource_bundle.blocker}}
    if not resource_bundle.heat_rate.get("ok"):
        return {"status": "BLOCKED_OPT", "submittable": False, "schedule": None,
                "summary": {"reason": "PHYSICAL_MAPPING_UNCERTAIN",
                            "heat_rate": resource_bundle.heat_rate}}
    prices = load_prices()
    if not prices:
        return {"status": "BLOCKED_OPT", "submittable": False, "schedule": None,
                "summary": {"reason": "PRICE_NOT_CONFIRMED"}}

    origin_positions = prepared.index.get_indexer(origins)
    position_to_row = {prepared.index[int(p)]: i for i, p in enumerate(origin_positions)}
    days = sorted(set(origins.normalize()))
    actual_series = resource_series(prepared).to_numpy(dtype=float)
    caps = np.array([
        float(prepared.frame[f"d_use_{gas}"].quantile(C.OPT_CONFIG["gas_cap_quantile"])
              * C.OPT_CONFIG["gas_cap_headroom"])
        for gas in C.RESOURCE_NAMES])
    caps = np.maximum(caps, 1.0)
    low = C.GAS_HOLDER["capacity_m3"] * C.GAS_HOLDER["safe_low_ratio"]
    high = C.GAS_HOLDER["capacity_m3"] * C.GAS_HOLDER["safe_high_ratio"]

    plans = {}
    records = []
    statuses = []
    for day in days:
        day_origin = pd.Timestamp(day)
        if day_origin not in position_to_row:
            continue
        row = position_to_row[day_origin]
        pos = int(origin_positions[row])
        recent = np.column_stack([
            _recent_lower(actual_series[:, g], pos, len(C.HORIZONS_L))
            for g in range(len(C.RESOURCE_NAMES))])
        mode = availability_mode or C.OPT_CONFIG.get("availability_mode")
        point = resource_forecast[row] * np.array(
            [C.OPT_CONFIG["gas_safety"][g] for g in C.RESOURCE_NAMES])
        if mode == "robust_low":
            # 下行边界优先用 p10 分位预测；其缺失槽位才退回历史同槽位低分位。
            # 不做双重 min：那会过度保守，把收益砍掉两成却换不来相应柜位安全。
            if quantile_low is not None:
                low_ref = np.where(np.isfinite(quantile_low[row]),
                                   quantile_low[row], recent)
            else:
                low_ref = recent
            avail = np.where(np.isfinite(low_ref),
                             np.minimum(point, low_ref * C.OPT_CONFIG["robust_headroom"]),
                             point)
        else:
            avail = point
        avail = np.maximum(avail, 0.0)

        target_start = day_origin + pd.Timedelta(minutes=C.STEP_MINUTES)
        price = _price_vector(prices, target_start, len(C.HORIZONS_L))
        last_g1 = _last_level(prepared, "generator_1", pos)
        last_all = _last_level(prepared, "generator_all", pos)
        if not np.isfinite(last_g1):
            last_g1 = 0.6 * C.CAPACITY["generator_1"]
        if not np.isfinite(last_all):
            last_all = 0.6 * C.CAPACITY["generator_all"]
        holder_series = prepared.frame[C.GAS_HOLDER["column"]]
        holder_value = holder_series.iloc[pos]
        if not np.isfinite(holder_value):
            valid_holder = holder_series.iloc[:pos + 1].dropna()
            holder_value = (float(valid_holder.iloc[-1]) if len(valid_holder)
                            else C.GAS_HOLDER["capacity_m3"] * 0.5)
        initial = np.array([last_g1 / 4.0] * 4 + [max(last_all - last_g1, 0.0) / 2.0] * 2)
        result = _solve_day(avail, price, float(holder_value),
                            initial, caps, resource_bundle.heat_rate)
        if result["status"] not in ("OPTIMAL", "FEASIBLE_STAGE1"):
            result = _fallback_day(prepared, pos, avail, caps, resource_bundle.heat_rate)
        statuses.append(result["status"])
        times = pd.date_range(target_start, periods=len(C.HORIZONS_L),
                              freq=f"{C.STEP_MINUTES}min")
        power = result["power"]
        gas = result["gas"]
        frame = pd.DataFrame({
            "datetime": times,
            "opt_generator_use_blast_furnace_gas": gas[0],
            "opt_generator_use_coke_gas": gas[1],
            "opt_generator_use_converter_gas": gas[2],
            "opt_generator_1": np.maximum(power[:4].sum(axis=0), 0.0),
            "opt_generator_all": np.maximum(power.sum(axis=0), 0.0),
        })
        plans[day_origin] = frame

        positions = prepared.index.get_indexer(times)
        # 官方窗口末端目标时刻会越出已发布网格；越界不参与回算，只保留可对齐步。
        aligned = positions >= 0
        actual = np.full((len(times), len(C.RESOURCE_NAMES)), np.nan)
        actual[aligned] = actual_series[positions[aligned]]
        excess = np.where(np.isfinite(actual), np.maximum(gas.T - actual, 0.0), 0.0)
        total_power = power.sum(axis=0)
        planned_revenue = float(np.nansum(price * total_power * 0.25 * 1000.0))
        label_col = prepared.labels["generator_all"]
        actual_total = label_col.reindex(times).to_numpy(dtype=float)
        real_revenue = float(np.nansum(price * actual_total * 0.25 * 1000.0))
        mean_price = float(np.mean(price)) if len(price) else 0.0
        weighted_ratio = (float(np.sum(price * total_power)
                                / max(mean_price * np.sum(total_power), 1e-9))
                          if mean_price > 0 else None)

        holder = float(holder_value)
        low_violations = 0
        overflow = 0.0
        holder_min, holder_max = holder, holder
        for k in range(len(times)):
            gas_b = gas[0, k]
            vent_b = result["vent"][0, k]
            flow = actual[k, 0] if np.isfinite(actual[k, 0]) else result["vent"][0, k] * 0.0
            holder += 0.25 * (flow - gas_b - vent_b)
            if holder > high:
                overflow += holder - high
                holder = high
            elif holder < low:
                low_violations += 1
            holder_min = min(holder_min, holder)
            holder_max = max(holder_max, holder)
        planned_holder = np.asarray(result["holder"], dtype=float)
        if np.isfinite(planned_holder).any():
            plan_violations = int(np.sum((planned_holder < low - 1e-6)
                                         | (planned_holder > high + 1e-6)))
            plan_min = float(np.nanmin(planned_holder))
            plan_max = float(np.nanmax(planned_holder))
        else:
            plan_violations = -1
            plan_min = plan_max = float("nan")
        above_by_gas = {gas: int(np.sum(excess[:, gi] > 1.0))
                        for gi, gas in enumerate(C.RESOURCE_NAMES)}
        records.append({
            "day": str(day.date()), "status": result["status"],
            "vent_m3": float(np.nansum(result["vent"]) * 0.25),
            "max_above_baseline_m3h": float(np.nanmax(excess)),
            "realized_above_baseline_cells": int(np.sum(excess > 1.0)),
            "realized_above_baseline_by_gas": above_by_gas,
            "plan_holder_violation_steps": plan_violations,
            "plan_holder_min_m3": plan_min,
            "plan_holder_max_m3": plan_max,
            "realized_holder_low_steps": int(low_violations),
            "realized_holder_overflow_m3": float(overflow),
            "realized_holder_min_m3": float(holder_min),
            "realized_holder_max_m3": float(holder_max),
            "planned_generation_mwh": float(np.sum(total_power) * 0.25),
            "baseline_generation_mwh": float(np.nansum(actual_total) * 0.25),
            "planned_revenue_cny": planned_revenue,
            "baseline_revenue_cny": real_revenue,
            "price_weighted_load_ratio": weighted_ratio,
        })

    if not plans:
        return {"status": "BLOCKED_OPT", "submittable": False, "schedule": None,
                "summary": {"reason": "NO_ORIGINS"}}
    schedule = pd.concat(plans.values()).sort_values("datetime").reset_index(drop=True)
    total_revenue = float(sum(r["planned_revenue_cny"] for r in records))
    baseline_revenue = float(sum(r["baseline_revenue_cny"] for r in records))
    summary = {
        "status": "SIMULATED_NOT_SUBMITTABLE",
        "days": len(records),
        "solver_status_counts": {s: statuses.count(s) for s in set(statuses)},
        "vent_m3": float(sum(r["vent_m3"] for r in records)),
        "above_baseline_cells": int(sum(r["realized_above_baseline_cells"]
                                          for r in records)),
        "above_baseline_by_gas": {
            gas: int(sum(r["realized_above_baseline_by_gas"][gas] for r in records))
            for gas in C.RESOURCE_NAMES},
        "max_gas_excess_m3h": float(max(r["max_above_baseline_m3h"] for r in records)),
        "plan_feasible": bool(all(r["plan_holder_violation_steps"] == 0 for r in records)),
        "plan_holder_violation_steps": int(sum(max(r["plan_holder_violation_steps"], 0)
                                                for r in records)),
        "plan_holder_min_m3": float(min(r["plan_holder_min_m3"] for r in records)),
        "plan_holder_max_m3": float(max(r["plan_holder_max_m3"] for r in records)),
        "realized_holder_low_steps": int(sum(r["realized_holder_low_steps"]
                                              for r in records)),
        "realized_holder_overflow_m3": float(sum(r["realized_holder_overflow_m3"]
                                                  for r in records)),
        "realized_holder_min_m3": float(min(r["realized_holder_min_m3"] for r in records)),
        "realized_holder_max_m3": float(max(r["realized_holder_max_m3"] for r in records)),
        "planned_generation_mwh": float(sum(r["planned_generation_mwh"] for r in records)),
        "baseline_generation_mwh": float(sum(r["baseline_generation_mwh"] for r in records)),
        "planned_revenue_cny": total_revenue,
        "baseline_revenue_cny": baseline_revenue,
        "revenue_improvement_ratio": (float((total_revenue - baseline_revenue)
                                            / baseline_revenue)
                                      if baseline_revenue > 0 else None),
        "price_weighted_load_ratio": (
            float(np.mean([r["price_weighted_load_ratio"] for r in records
                           if r["price_weighted_load_ratio"] is not None]))
            if any(r["price_weighted_load_ratio"] is not None for r in records) else None),
        "heat_rate": resource_bundle.heat_rate,
        "holder": {k: C.GAS_HOLDER[k] for k in ("capacity_m3", "safe_low_ratio",
                                                "safe_high_ratio")},
        "template_confirmed": bool(C.OPT_CONFIG["template_confirmed"]),
        "availability_mode": availability_mode or C.OPT_CONFIG.get("availability_mode"),
        "time_semantics": "target_time t+15min ... t+1440min",
        "note": "无官方 opt_result 模板与完整物理参数，结果为历史模拟原型；"
                "经济收益仅作回测诊断，未计放散罚金官方系数。",
        "daily": records,
    }
    return {"status": "SIMULATED_NOT_SUBMITTABLE", "submittable": False,
            "schedule": schedule, "summary": summary}
