"""负荷条件时序网络（TSMixer / 因果TCN / DLinear）与独立八步资源网络。

16GB GPU 专项优化要点：
1. 因果特征只算一次并常驻显存的 ``[T, F]`` 张量表，滑窗按批 gather，避免构造
   ``N×L×F`` 三重数组；同一折的多次训练/推理复用同一张表。
2. 训练走 bf16 自动混合精度，推理回到 fp32，保证批量与单点预测逐值一致可复验。
3. 支持容量归一化直接预测与画像残差对照；早停按原量纲物理投影后的 MAPE。
4. 内部归一化不使用早停段，验证目标历史连续遮挡，选轮和重训复用学习率轨迹。
"""
from __future__ import annotations

import math
import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from data_cleaning import PreparedData, build_base_features
from data_cleaning.pipeline import (_causal_slot_profiles, _day_reference,
                                    fuel_level_ratio, target_calendar)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 固定随机性：CUDA 卷积/矩阵乘若不确定，同参会差 0.2pp，掩盖真实收益。
torch.manual_seed(20261021)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# TF32 会改变 fp32 matmul 累加顺序，使“批量预测”与“单点预测”出现 1e-3 级差异，
# 破坏可复验契约；训练侧改用 bf16 autocast 提速，这里有理由保持关闭。
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# 各网络默认预测空间：比值空间只对条件网络启用，线性基线保留原始量纲便于对照。
DEFAULT_HEADS = {"dlinear": "direct", "nhits": "direct", "tcn": "direct",
                 "tsmixer": "ratio", "tcn_causal": "ratio"}


def _cfg(overrides: dict | None = None) -> dict:
    cfg = dict(C.NN_CONFIG)
    cfg.update(overrides or {})
    return cfg


def _amp_enabled(precision: str) -> bool:
    if precision not in ("fp32", "bf16"):
        raise ValueError("当前训练精度只支持 fp32/bf16，不能把其它模式静默当作 bf16")
    return precision == "bf16" and DEVICE.type == "cuda"


def peak_memory_mb() -> float:
    if DEVICE.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


# ---------------------------------------------------------------------------
# 独立八步资源网络（与负荷网络隔离，不共享标签或预测头）
# ---------------------------------------------------------------------------
class _ResourceResidualMLP(nn.Module):
    """资源联合预测与负荷网络隔离，只输出八步三气种的非共享预测头。"""
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Dropout(.10),
                                    nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 24))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x, anchor):
        return self.layers(x).reshape(-1, 8, 3) + anchor[:, None, :]


class _ResourceResidualTCN(nn.Module):
    """近期三气种观测和缺失标记经因果卷积编码，再与当前过程量共同预测。"""
    def __init__(self, width: int):
        super().__init__()
        self.current = nn.Sequential(nn.Linear(width, 64), nn.SiLU())
        self.stem = nn.Conv1d(6, 32, 1)
        self.blocks = nn.Sequential(*[_CausalBlock(32, 2 ** i, .10) for i in range(5)])
        self.head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 24))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, anchor, sequence):
        history = self.blocks(self.stem(sequence.transpose(1, 2)))
        context = torch.cat((self.current(x), history[:, :, -1], history.mean(-1)), dim=-1)
        return self.head(context).reshape(-1, 8, 3) + anchor[:, None, :]


def _resource_sequence(actual, positions, scale, look_back=96):
    selected = actual[positions[:, None] - np.arange(look_back - 1, -1, -1)]
    normalized = np.nan_to_num(np.clip(selected / scale, 0., 10.), nan=0.)
    return np.concatenate((normalized, np.isfinite(selected)), axis=-1).astype(np.float32)


def _resource_nn_matrix(values: np.ndarray, positions: np.ndarray,
                        center: np.ndarray, scale: np.ndarray,
                        history: np.ndarray | None = None,
                        history_center: np.ndarray | None = None,
                        history_scale: np.ndarray | None = None) -> np.ndarray:
    """当前过程量 + 缺失指示（+ 起报点前气量路径的固定窗口均值/标准差）。"""
    selected = values[positions]
    z = np.clip((selected - center) / scale, -10., 10.)
    # 标准化后的中性值与缺失指示同时输入，不把过程缺测伪装成真实零供气。
    parts = [np.nan_to_num(z, nan=0.), np.isfinite(selected).astype(np.float32)]
    if history is not None and history_center is not None:
        zh = np.clip((np.asarray(history, dtype=np.float32) - history_center) / history_scale,
                     -10., 10.)
        pooled = []
        for window in (8, 32):
            idx = positions[:, None] - np.arange(window)[None, :]
            block = zh[np.clip(idx, 0, len(zh) - 1)]
            block = np.where((idx >= 0)[:, :, None], block, np.nan)
            pooled.append(np.nan_to_num(np.nanmean(block, axis=1), nan=0.))
            pooled.append(np.nan_to_num(np.nanstd(block, axis=1), nan=0.))
        parts.append(np.concatenate(pooled, axis=1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def _resource_nn_features(values: np.ndarray, positions: np.ndarray, fit_end: int,
                          history: np.ndarray | None = None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    prefix = values[:fit_end + 1]
    center = np.nan_to_num(np.nanmedian(prefix, axis=0), nan=0.)
    scale = np.nanpercentile(prefix, 75, axis=0) - np.nanpercentile(prefix, 25, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-5), scale, 1.)
    history_center = history_scale = None
    if history is not None:
        # 历史窗口池化：只看起报点及以前的气量路径，给网络“近期水平/波动”上下文。
        series = np.asarray(history, dtype=np.float32)[:fit_end + 1]
        history_center = np.nan_to_num(np.nanmedian(series, axis=0), nan=0.)
        history_scale = (np.nanpercentile(series, 75, axis=0)
                         - np.nanpercentile(series, 25, axis=0))
        history_scale = np.where(np.isfinite(history_scale) & (history_scale > 1e-5),
                                 history_scale, 1.)
    X = _resource_nn_matrix(values, positions, center, scale, history,
                            history_center, history_scale)
    return X, center, scale, (history_center, history_scale)


def _fit_resource_nn(X, anchor, target, energy_weights, price_weights,
                     epochs: int, validation=None, seed: int = 73,
                     sequence=None, batch: int = 256):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = (_ResourceResidualTCN(X.shape[1]) if sequence is not None else
             _ResourceResidualMLP(X.shape[1])).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    tensors = [torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
               for a in (X, anchor, target, price_weights)]
    ew = torch.as_tensor(energy_weights, dtype=torch.float32, device=DEVICE)
    sequence = None if sequence is None else torch.as_tensor(sequence, device=DEVICE)
    best, best_epoch, stale = np.inf, 1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X), device=DEVICE)
        for start in range(0, len(X), batch):
            take = order[start:start + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=sequence is not None and DEVICE.type == "cuda"):
                prediction = (model(tensors[0][take], tensors[1][take], sequence[take])
                              if sequence is not None else model(tensors[0][take], tensors[1][take]))
            prediction = prediction.float()
            y = tensors[2][take]
            valid = torch.isfinite(y)
            safe = torch.nan_to_num(y, nan=0.)
            error = nn.functional.smooth_l1_loss(prediction, safe, reduction="none", beta=.2)
            weight = valid * tensors[3][take, :, None] * ew
            loss = (error * weight).sum() / weight.sum().clamp_min(1.)
            joint_valid = valid.all(dim=-1)
            joint_error = torch.abs(((prediction - safe) * ew).sum(dim=-1))
            loss += .25 * (joint_error * joint_valid * tensors[3][take]).sum() / joint_valid.sum().clamp_min(1.)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                vx, va, vy = validation[:3]
                prediction = model(vx, va, validation[3]) if sequence is not None else model(vx, va)
                mask = torch.isfinite(vy)
                error = torch.abs(prediction - torch.nan_to_num(vy, nan=0.)) * mask * ew
                # 与资源选模一致：逐气种 WAPE 含零真值误报，再等权聚合。
                if sequence is None:
                    score = float(error.sum() / (mask * ew).sum())
                else:
                    score = float(((torch.abs(prediction - torch.nan_to_num(vy, nan=0.)) * mask).sum((0, 1)) /
                                   (torch.nan_to_num(vy, nan=0.).abs() * mask).sum((0, 1)).clamp_min(1e-6)).mean())
            if score < best - 1e-5:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            if stale >= 10:
                break
    return model, best_epoch, best


def train_resource_nn(base: pd.DataFrame, resource: pd.DataFrame, end_pos: int,
                      heat: dict, prices: dict, quick: bool = False, seed: int = 73,
                       use_history: bool = True, network_kind: str | None = None,
                       max_epochs: int = 100, batch: int = 256) -> dict:
    """仅在起报前前缀选轮数，再以该轮数重训；不访问外层验证标签。"""
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    values = base.to_numpy(dtype=np.float32)
    actual = resource[list(C.RESOURCE_NAMES)].to_numpy(dtype=np.float32)
    network_kind = network_kind or C.OPT_CONFIG.get("resource_nn_kind", "mlp")
    if network_kind not in ("mlp", "tcn"):
        raise ValueError("资源网络仅支持 mlp/tcn")
    history = actual if use_history else None
    positions = np.arange(C.PROFILE_DAYS * C.STEPS_PER_DAY, end_pos - 8 + 1)
    if quick:
        positions = positions[::C.QUICK_TRAIN_STRIDE][-2048:]
    split = end_pos - 7 * C.STEPS_PER_DAY
    train, valid = positions + 8 < split, positions >= split
    if train.sum() < 128 or valid.sum() < 32:
        raise ValueError("资源网络的前缀训练或内部早停样本不足")
    gas_scale = np.maximum(np.nanmedian(actual[:end_pos + 1], axis=0), 1.)
    beta = np.array([heat["coefficients_mw_per_m3h"][g] for g in C.RESOURCE_NAMES])
    energy_weights = (gas_scale * beta) / (gas_scale @ beta)
    target = actual[positions[:, None] + np.arange(1, 9)] / gas_scale
    anchor = np.nan_to_num(actual[positions] / gas_scale, nan=0.)
    times = base.index.to_numpy()[positions[:, None] + np.arange(1, 9)]
    flat = pd.DatetimeIndex(times.ravel())
    price_weight = np.array([prices[t.month][t.hour * 4 + t.minute // 15] / prices[t.month].mean()
                             for t in flat]).reshape(len(positions), 8)
    price_weight = price_weight / np.sqrt(np.arange(1, 9))
    inner_scale = np.maximum(np.nanmedian(actual[:split], axis=0), 1.)
    inner_anchor = np.nan_to_num(actual[positions] / inner_scale, nan=0.)
    inner_target = actual[positions[:, None] + np.arange(1, 9)] / inner_scale
    inner_weights = (inner_scale * beta) / (inner_scale @ beta)
    inner_X, _, _, _ = _resource_nn_features(values, positions, split - 1, history)
    validation = tuple(torch.as_tensor(v, dtype=torch.float32, device=DEVICE)
                       for v in (inner_X[valid], inner_anchor[valid], inner_target[valid]))
    inner_sequence = _resource_sequence(actual, positions, inner_scale) if network_kind == "tcn" else None
    if inner_sequence is not None:
        validation += (torch.as_tensor(inner_sequence[valid], device=DEVICE),)
    _, epochs, error = _fit_resource_nn(inner_X[train], inner_anchor[train], inner_target[train],
        inner_weights if network_kind == "tcn" else energy_weights,
        price_weight[train], 2 if quick else max_epochs, validation, seed,
        sequence=None if inner_sequence is None else inner_sequence[train], batch=batch)
    X, center, scale, (history_center, history_scale) = _resource_nn_features(
        values, positions, end_pos, history)
    sequence = _resource_sequence(actual, positions, gas_scale) if network_kind == "tcn" else None
    model, _, _ = _fit_resource_nn(X, anchor, target, energy_weights, price_weight, epochs,
                                  seed=seed, sequence=sequence, batch=batch)
    return {"kind": "resource_tcn" if network_kind == "tcn" else "resource_mlp",
            "state_dict": model.cpu().state_dict(), "width": X.shape[1],
            "epochs": epochs, "inner_error": error, "feature_names": list(base.columns),
            "center": center, "scale": scale, "gas_scale": gas_scale,
            "use_history": bool(use_history),
            "history_center": history_center, "history_scale": history_scale,
            "train_end": str(base.index[end_pos]), "inner_train_end": str(base.index[split - 1]),
            "max_training_target": str(base.index[positions[-1] + 8]), "seed": seed}


def predict_resource_nn(pack: dict, base: pd.DataFrame, resource: pd.DataFrame,
                        positions: np.ndarray) -> np.ndarray:
    if list(base.columns) != pack["feature_names"]:
        raise ValueError("资源网络特征顺序不一致")
    model = pack.get("_runtime_model")
    if model is None:
        model = (_ResourceResidualTCN(pack["width"]) if pack["kind"] == "resource_tcn" else
                 _ResourceResidualMLP(pack["width"])).to(DEVICE)
        model.load_state_dict(pack["state_dict"])
        model.eval()
        pack["_runtime_model"] = model
    values = base.to_numpy(dtype=np.float32)
    actual = resource[list(C.RESOURCE_NAMES)].to_numpy(dtype=np.float32)
    history = actual if pack.get("use_history") else None
    X = _resource_nn_matrix(values, np.asarray(positions, dtype=int), pack["center"],
                            pack["scale"], history, pack.get("history_center"),
                            pack.get("history_scale"))
    anchor = np.nan_to_num(actual[positions] / pack["gas_scale"], nan=0.)
    sequence = (_resource_sequence(actual, positions, pack["gas_scale"])
                if pack["kind"] == "resource_tcn" else None)
    with torch.no_grad():
        estimates = []
        for start in range(0, len(X), 512):
            take = slice(start, start + 512)
            args = (torch.as_tensor(X[take], device=DEVICE), torch.as_tensor(anchor[take], device=DEVICE))
            if sequence is not None:
                args += (torch.as_tensor(sequence[take], device=DEVICE),)
            estimates.append(model(*args).cpu().numpy())
        output = np.concatenate(estimates)
    return np.maximum(output * pack["gas_scale"], 0.)


def save_resource_nn(pack: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in pack.items() if not k.startswith("_")}, path)
    return path


def load_resource_nn(path: Path) -> dict:
    pack = torch.load(path, map_location="cpu", weights_only=False)
    if pack.get("kind") not in ("resource_mlp", "resource_tcn"):
        raise ValueError("不是独立八步资源网络")
    return pack


# ---------------------------------------------------------------------------
# 因果特征张量表：每个 PreparedData 只算一次，滑窗在显存里按批 gather
# ---------------------------------------------------------------------------
@dataclass
class FeatureTable:
    """``values`` 形状 (T, F)：第 t 行的全部特征只依赖 t 时刻及以前。"""
    values: np.ndarray
    columns: list[str]
    index: pd.DatetimeIndex
    load_history: bool = False


_TABLE_CACHE: dict[tuple, tuple] = {}
_PROFILE_CACHE: dict[tuple, tuple] = {}
_CACHE_LIMIT = 4


def clear_cache() -> None:
    _TABLE_CACHE.clear()
    _PROFILE_CACHE.clear()


def _remember(cache: dict, key: tuple, holder, payload):
    if len(cache) >= _CACHE_LIMIT and key not in cache:
        cache.pop(next(iter(cache)))
    # 同时持有 frame 引用，杜绝 id() 复用造成的错表命中。
    cache[key] = (holder, payload)


def _load_history_columns(prepared: PreparedData) -> tuple[np.ndarray, list[str]]:
    """最后可见负荷水平与“距今多少天没见到标签”：遮挡后的陈旧程度可量化。"""
    labels = prepared.frame[[C.TARGET_COLUMN[t] for t in C.TARGETS]].to_numpy(dtype=float)
    valid = np.isfinite(labels)
    idx = np.arange(len(labels))
    last = np.maximum.accumulate(np.where(valid, idx[:, None], -1), axis=0)
    age = np.where(last >= 0, idx[:, None] - last, C.STEPS_PER_DAY * C.PROFILE_DAYS)
    # 前向填充只用过去可见标签；评测窗内标签整体遮挡时退化为起报前水平。
    forward = pd.DataFrame(labels).ffill().to_numpy(dtype=float)
    forward = np.nan_to_num(forward, nan=0.0)
    columns = ([f"nn_last_{t}_level" for t in C.TARGETS] +
               [f"nn_last_{t}_age_days" for t in C.TARGETS])
    values = np.concatenate([forward, np.clip(age, 0, C.PROFILE_DAYS * C.STEPS_PER_DAY) / C.STEPS_PER_DAY],
                            axis=1).astype(np.float32)
    return values, columns


def build_feature_table(prepared: PreparedData, load_history: bool = False,
                        feature_set: str = "engineered") -> FeatureTable:
    key = (id(prepared.frame), bool(load_history), feature_set)
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit[1]
    from main_s import _feature_columns

    base = build_base_features(prepared, include_targets=True)
    columns = list(_feature_columns(base, "gas"))
    if feature_set == "compact":
        columns = [c for c in columns if c.startswith(("cal_", "obs_")) or
                   (c.startswith("d_") and not any(s in c for s in ("_lag", "_ma", "_std"))
                    and not c.endswith(("_d1", "_d4")))]
    elif feature_set != "engineered":
        raise ValueError(f"未知神经网络特征集: {feature_set}")
    values = base[columns].to_numpy(dtype=np.float32)
    if load_history:
        extra, extra_columns = _load_history_columns(prepared)
        values = np.concatenate([values, extra], axis=1)
        columns = columns + extra_columns
    table = FeatureTable(values=values, columns=columns, index=prepared.index,
                         load_history=bool(load_history))
    _remember(_TABLE_CACHE, key, prepared.frame, table)
    return table


def _fit_normalizer(values: np.ndarray, fit_end: int) -> tuple[np.ndarray, np.ndarray]:
    history = values[:fit_end + 1]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(history, axis=0)
        std = np.nanstd(history, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std >= 1e-8), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _gpu_table(table: FeatureTable, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    z = (table.values - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.as_tensor(np.clip(z, -10.0, 10.0), dtype=torch.float32, device=DEVICE)


_OFFSET_CACHE: dict[int, torch.Tensor] = {}


def _offsets(look_back: int) -> torch.Tensor:
    offsets = _OFFSET_CACHE.get(look_back)
    if offsets is None or offsets.device != DEVICE:
        offsets = torch.arange(-(look_back - 1), 1, dtype=torch.long, device=DEVICE)
        _OFFSET_CACHE[look_back] = offsets
    return offsets


def _windows(table: torch.Tensor, positions: torch.Tensor, look_back: int) -> torch.Tensor:
    index = positions[:, None] + _offsets(look_back)[None, :]
    return table.index_select(0, index.reshape(-1)).view(positions.shape[0], look_back, -1)


# ---------------------------------------------------------------------------
# 画像比值目标与未来日历协变量
# ---------------------------------------------------------------------------
def _future_columns(prepared: PreparedData, positions: np.ndarray,
                     horizons: tuple[int, ...], cfg: dict | None = None) -> np.ndarray:
    cal = target_calendar(prepared, positions, horizons)
    n, m = len(positions), len(horizons)
    span = float(max(int(horizons[-1]), 1))
    frac = np.broadcast_to((np.asarray(horizons, dtype=np.float32) / span)[None, :], (n, m))
    parts = [cal["tgt_slot_sin"], cal["tgt_slot_cos"],
             cal["tgt_weekday"] / 6.0, cal["tgt_is_weekend"],
             (cal["tgt_month"] - 1.0) / 11.0, frac]
    if (cfg or {}).get("future_calendar") == "cyclic":
        # 不让有限月份的序号成为外推捷径；周内时段使用有界周期表示。
        weekday = cal["tgt_weekday"]
        parts = [cal["tgt_slot_sin"], cal["tgt_slot_cos"], np.sin(2 * np.pi * weekday / 7.),
                 np.cos(2 * np.pi * weekday / 7.), cal["tgt_is_weekend"], frac]
    return np.stack(parts, axis=-1).astype(np.float32)


def _profile_table(prepared: PreparedData,
                   horizons: tuple[int, ...]) -> np.ndarray:
    """全位置画像 (T, H, 2)：只用各位置之前已完整结束的自然日。"""
    max_h = int(max(horizons))
    key = (id(prepared.frame), max_h)
    hit = _PROFILE_CACHE.get(key)
    if hit is not None:
        return hit[1]
    periods = tuple(range(1, max_h + 1))
    steps = np.arange(len(prepared.index))
    ref_days = np.array([_day_reference(int(p)) for p in steps], dtype=int)
    slots = (steps[:, None] + np.asarray(periods)[None, :]) % C.STEPS_PER_DAY
    out = np.empty((len(steps), max_h, len(C.TARGETS)), dtype=np.float32)
    for ti, target in enumerate(C.TARGETS):
        series = prepared.frame[C.TARGET_COLUMN[target]].to_numpy(dtype=float)
        rows, _ = _causal_slot_profiles(series, ref_days, int(C.PROFILE_DAYS))
        last = pd.Series(series, index=prepared.index).ffill().to_numpy(dtype=float)
        last = np.nan_to_num(last, nan=0.0)
        filled = np.where(np.isfinite(rows), rows, last[:, None])
        out[:, :, ti] = filled[steps[:, None], slots]
    _remember(_PROFILE_CACHE, key, prepared.frame, out)
    return out


def _profile_cube(prepared: PreparedData, positions: np.ndarray,
                  horizons: tuple[int, ...], fuel: bool = True) -> np.ndarray:
    table = _profile_table(prepared, horizons)
    picks = [int(h) - 1 for h in horizons]
    cube = table[np.asarray(positions, dtype=int)][:, picks, :].astype(np.float64)
    if fuel:
        ratio = fuel_level_ratio(prepared, positions)
        ratio = np.where(np.isfinite(ratio), ratio, 1.0)
        cube *= ratio[:, None, None]
    # 画像比值侧的下界只防除零，不改变主流程已有的融合口径。
    cube = np.clip(cube, 1.0, None)
    for ti, target in enumerate(C.TARGETS):
        cube[:, :, ti] = np.minimum(cube[:, :, ti], float(C.CAPACITY[target]))
    return cube


def _profile_is_fuel(cfg: dict) -> bool:
    return str(cfg.get("profile_base", "fuel")) == "fuel"


def _profile_conditions(prepared, positions, future, profile, mask_days=0):
    """解码器必须知道被修正的画像和陈旧程度，否则同一工况对应的残差不唯一。"""
    values = prepared.frame[[C.TARGET_COLUMN[t] for t in C.TARGETS]].to_numpy(dtype=float)
    indices = np.arange(len(values))[:, None]
    last = np.maximum.accumulate(np.where(np.isfinite(values), indices, -1), axis=0)
    if mask_days > 0:
        first = C.PROFILE_DAYS * C.STEPS_PER_DAY
        starts = first + ((positions - first) // (mask_days * C.STEPS_PER_DAY)) * mask_days * C.STEPS_PER_DAY
        visible = last[np.maximum(starts - 1, 0)]
    else:
        visible = last[positions]
    ages = np.clip((positions[:, None] - visible) / (21. * C.STEPS_PER_DAY), 0., 2.)
    ages = np.broadcast_to(ages[:, None, :], profile.shape)
    capacity = np.array([C.CAPACITY[t] for t in C.TARGETS])
    return np.concatenate((future, profile / capacity, ages), axis=-1).astype(np.float32)


def _blocked_profiles(prepared: PreparedData, positions: np.ndarray,
                      horizons: tuple[int, ...], days: int) -> np.ndarray:
    """训练按连续块遮挡目标历史，画像可随窗口陈旧而变化，标签本身不被改写。"""
    result = np.empty((len(positions), len(horizons), 2), dtype=np.float32)
    block_steps = days * C.STEPS_PER_DAY
    first = C.PROFILE_DAYS * C.STEPS_PER_DAY
    starts = first + ((positions - first) // block_steps) * block_steps
    refs = np.array([_day_reference(int(p)) for p in positions])
    slots = (positions[:, None] + np.asarray(horizons)) % C.STEPS_PER_DAY
    for ti, target in enumerate(C.TARGETS):
        values = prepared.frame[C.TARGET_COLUMN[target]].to_numpy(dtype=float)
        matrix = values[:len(values) // C.STEPS_PER_DAY * C.STEPS_PER_DAY].reshape(-1, C.STEPS_PER_DAY)
        for start, ref in set(zip(starts.tolist(), refs.tolist())):
            take = (starts == start) & (refs == ref)
            lo, hi = max(0, ref - C.PROFILE_DAYS + 1), min(ref + 1, len(matrix))
            block = matrix[lo:hi].copy()
            times = np.arange(lo * C.STEPS_PER_DAY, hi * C.STEPS_PER_DAY).reshape(block.shape)
            block[times >= start] = np.nan
            prof = np.nanmedian(block, axis=0)
            prof = np.where((np.isfinite(block).sum(axis=0) >= 3) & np.isfinite(prof),
                            prof, np.nanmedian(prof))
            past = values[:start]
            last = past[np.isfinite(past)][-1] if np.isfinite(past).any() else 0.
            prof = np.where(np.isfinite(prof), prof, last)
            result[take, :, ti] = np.clip(prof[slots[take]], 1., C.CAPACITY[target])
    return result


# ---------------------------------------------------------------------------
# 网络结构
# ---------------------------------------------------------------------------
class DLinear(nn.Module):
    """序列线性分解（趋势 + 季节）后线性映射到多步，退化基线对照。"""

    def __init__(self, look_back: int, horizon: int, n_feat: int,
                 n_fut: int = 0, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}
        self.horizon = horizon
        self.trend_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.linear_trend = nn.Linear(look_back, horizon)
        self.linear_season = nn.Linear(look_back, horizon)
        self.mix = nn.Linear(n_feat, 2)
        if cfg.get("head_mode") in ("ratio", "residual", "scaled"):
            # 只清零末层；时间映射保留非零初始化，否则连续两层全零会永久阻断梯度。
            nn.init.zeros_(self.mix.weight)
            with torch.no_grad():
                self.mix.bias.fill_({"ratio": 1., "scaled": .5}.get(cfg.get("head_mode"), 0.))

    def forward(self, x: torch.Tensor, fut: torch.Tensor | None = None) -> torch.Tensor:
        series = x.transpose(1, 2)                            # (B, F, L)
        trend = self.trend_pool(series)                       # (B, F, L)
        season = series - trend
        out_t = self.linear_trend(trend)                      # (B, F, H)
        out_s = self.linear_season(season)
        merged = (out_t + out_s).transpose(1, 2)              # (B, H, F)
        return self.mix(merged)                               # (B, H, 2)


class _MixBlock(nn.Module):
    """TSMixer 单块：时间混合 + 特征混合，各自带残差与 LayerNorm。"""

    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.norm_time = nn.LayerNorm(d_model)
        self.time_mlp = nn.Sequential(nn.Linear(seq_len, seq_len), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(seq_len, seq_len))
        self.norm_feat = nn.LayerNorm(d_model)
        self.feat_mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_time(x)
        h = x + self.time_mlp(h.transpose(1, 2)).transpose(1, 2)
        return h + self.feat_mlp(self.norm_feat(h))


class _ConditionalHead(nn.Module):
    """目标时刻日历嵌入 + 窗口上下文 -> 每个提前量的双目标输出。"""

    def __init__(self, d_model: int, n_fut: int, dropout: float, init_level: float):
        super().__init__()
        self.future = nn.Sequential(nn.Linear(n_fut, d_model), nn.GELU(),
                                    nn.Linear(d_model, d_model))
        self.mix = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, 2))
        last = self.mix[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, init_level)

    def forward(self, context: torch.Tensor, fut: torch.Tensor) -> torch.Tensor:
        embedded = self.future(fut)
        joined = torch.cat([embedded, context.unsqueeze(1).expand(-1, embedded.shape[1], -1)], dim=-1)
        return self.mix(joined)


class TSMixer(nn.Module):
    """条件 TSMixer：交替混合时间/特征轴，再按目标时刻日历生成多步输出。"""

    def __init__(self, look_back: int, horizon: int, n_feat: int,
                 n_fut: int = 6, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}
        d_model = int(cfg.get("d_model", 64))
        n_blocks = int(cfg.get("n_blocks", 2))
        dropout = float(cfg.get("dropout", 0.15))
        init_level = {"ratio": 1., "scaled": .5}.get(cfg.get("head_mode"), 0.)
        self.proj = nn.Linear(n_feat, d_model)
        self.blocks = nn.ModuleList([_MixBlock(look_back, d_model, dropout)
                                     for _ in range(n_blocks)])
        self.context = nn.Linear(2 * d_model, d_model)
        self.head = _ConditionalHead(d_model, n_fut, dropout, init_level)

    def forward(self, x: torch.Tensor, fut: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj(x)
        for block in self.blocks:
            h = block(h)
        context = self.context(torch.cat([h[:, -1], h.mean(dim=1)], dim=-1))
        return self.head(context, fut)


class _CausalBlock(nn.Module):
    """左侧补零的膨胀卷积，严格因果；通道一致以便残差相加。"""

    def __init__(self, channels: int, dilation: int, dropout: float, normalization: str = "layer"):
        super().__init__()
        self.dilation = dilation
        self.kernel = 3
        self.conv = nn.Conv1d(channels, channels, kernel_size=self.kernel, dilation=dilation)
        self.norm = nn.LayerNorm(channels) if normalization == "layer" else nn.BatchNorm1d(channels)
        self.normalization = normalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 因果卷积要求左侧补 (k-1)*d 个零，补满后输出长度与输入一致才能做残差。
        h = F.pad(x, ((self.kernel - 1) * self.dilation, 0))
        h = self.conv(h)
        h = (self.norm(h.transpose(1, 2)).transpose(1, 2)
             if self.normalization == "layer" else self.norm(h))
        h = self.dropout(F.gelu(h))
        return x + h


class CausalTCN(nn.Module):
    """残差因果 TCN：短提前量主力，感受野覆盖整段回看窗口。"""

    def __init__(self, look_back: int, horizon: int, n_feat: int,
                 n_fut: int = 6, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}
        channels = int(cfg.get("d_model", 64))
        n_blocks = int(cfg.get("n_blocks", 3))
        dropout = float(cfg.get("dropout", 0.10))
        init_level = {"ratio": 1., "scaled": .5}.get(cfg.get("head_mode"), 0.)
        self.stem = nn.Conv1d(n_feat, channels, kernel_size=1)
        self.blocks = nn.ModuleList([_CausalBlock(channels, 2 ** i, dropout,
                                     str(cfg.get("temporal_norm", "batch")))
                                     for i in range(n_blocks)])
        self.norm = nn.LayerNorm(channels)
        self.context = nn.Linear(2 * channels, channels)
        self.head = _ConditionalHead(channels, n_fut, dropout, init_level)

    def forward(self, x: torch.Tensor, fut: torch.Tensor | None = None) -> torch.Tensor:
        h = self.stem(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        h = self.norm(h.transpose(1, 2))
        context = self.context(torch.cat([h[:, -1], h.mean(dim=1)], dim=-1))
        return self.head(context, fut)


class _NHiTSBlock(nn.Module):
    def __init__(self, look_back: int, horizon: int, n_feat: int,
                 d_model: int, pool_kernel: int):
        super().__init__()
        k = max(pool_kernel, 1)
        self.pool = nn.AvgPool1d(k, stride=k) if k > 1 else nn.Identity()
        self.proj_in = nn.Linear(n_feat, d_model)
        self.fc1 = nn.Linear((look_back // k) * d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, horizon * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.proj_in(x))
        z = self.pool(z.transpose(1, 2)).transpose(1, 2)
        z = z.reshape(z.size(0), -1)
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        return self.head(z).view(-1, self.head.out_features // 2, 2)


class NHiTS(nn.Module):
    """分层池化多块叠加，长外推对照（非完整 N-HiTS）。"""

    def __init__(self, look_back: int, horizon: int, n_feat: int,
                 n_fut: int = 0, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}
        d_model = int(cfg.get("d_model", 64))
        n_blocks = int(cfg.get("n_blocks", 3))
        kernels = [1, 2, 4][:n_blocks]
        self.blocks = nn.ModuleList([
            _NHiTSBlock(look_back, horizon, n_feat, d_model, k) for k in kernels])

    def forward(self, x: torch.Tensor, fut: torch.Tensor | None = None) -> torch.Tensor:
        out = 0.0
        for block in self.blocks:
            out = out + block(x)
        return out


class TCNBlock(nn.Module):
    """旧版对称 padding 卷积块，仅保留作历史对照。"""

    def __init__(self, n_feat: int, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(n_feat, channels, kernel_size=3,
                              dilation=dilation, padding=dilation)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class TCN(nn.Module):
    def __init__(self, look_back: int, horizon: int, n_feat: int,
                 n_fut: int = 0, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}
        channels = int(cfg.get("d_model", 32))
        n_layers = int(cfg.get("n_blocks", 3))
        layers = [TCNBlock(n_feat, channels, 1)]
        for i in range(1, n_layers):
            layers.append(TCNBlock(channels, channels, 2 ** i))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(channels * look_back, horizon * 2)

    def forward(self, x: torch.Tensor, fut: torch.Tensor | None = None) -> torch.Tensor:
        z = self.net(x.transpose(1, 2))
        return self.head(z.reshape(z.size(0), -1)).view(-1, self.head.out_features // 2, 2)


MODEL_ZOO = {"dlinear": DLinear, "tsmixer": TSMixer, "tcn_causal": CausalTCN,
             "nhits": NHiTS, "tcn": TCN}


def build_model(kind: str, look_back: int, horizon: int, n_feat: int,
                n_fut: int, cfg: dict) -> nn.Module:
    if kind not in MODEL_ZOO:
        raise ValueError(f"未知网络类型: {kind}")
    return MODEL_ZOO[kind](look_back, horizon, n_feat, n_fut=n_fut, cfg=cfg).to(DEVICE)


# ---------------------------------------------------------------------------
# 批量训练
# ---------------------------------------------------------------------------
class _Sample:
    """一组滑窗样本；目标、未来协变量、画像比值全部常驻显存。"""

    def __init__(self, table: torch.Tensor, positions: np.ndarray, targets: np.ndarray,
                 future: np.ndarray, profile: np.ndarray | None, batch: int,
                 look_back: int, horizons: tuple[int, ...], mode: str,
                 cfg: dict | None = None):
        cfg = cfg or {}
        self.table = table
        self.look_back = look_back
        self.batch = batch
        self.mode = mode
        self.residual_limit = float(cfg.get("residual_limit", 0.) or 0.)
        self.memory_limit = float(cfg.get("max_gpu_memory_gib", 15.)) * 1024 ** 3
        self.capacity = torch.tensor([C.CAPACITY[t] for t in C.TARGETS], device=DEVICE)
        self.positions = positions.astype(np.int64)
        self.count = len(self.positions)
        self.pos = torch.as_tensor(self.positions, dtype=torch.long, device=DEVICE)
        self.future = torch.as_tensor(future, dtype=torch.float32, device=DEVICE)
        y = torch.as_tensor(targets, dtype=torch.float32, device=DEVICE)
        self.y = y
        self.valid = torch.isfinite(y)
        if profile is not None:
            self.profile = torch.as_tensor(profile, dtype=torch.float32, device=DEVICE)
        else:
            self.profile = None
        weight = torch.as_tensor(np.asarray(horizons), dtype=torch.float32, device=DEVICE)
        self.h_weight = torch.where(weight > 32, torch.full_like(weight, float(cfg.get("far_weight", 1.))),
                                    torch.ones_like(weight)).view(1, -1, 1)
        # 主指标是 1-MAPE：损失按 1/目标量级 加权，否则低负荷时段会被画像尺度吞掉。
        floor_ratio = float(cfg.get("mape_floor_ratio", 0.02))
        self.floor = torch.as_tensor([floor_ratio * float(C.CAPACITY[t]) for t in C.TARGETS],
                                     dtype=torch.float32, device=DEVICE)
        # 增广只作用于历史输入，不改未来协变量与目标：样本量受“天”数限制，
        # 轻微噪声/通道丢弃能抑制 trunk 记住训练期工况。
        self.aug_noise = float(cfg.get("aug_noise", 0.0) or 0.0)
        self.aug_channel_drop = float(cfg.get("aug_channel_drop", 0.0) or 0.0)
        self.prior = float(cfg.get("prior_lambda", 0.0) or 0.0)
        half_life = float(cfg.get("sample_decay_days", 0.) or 0.)
        age = np.maximum(int(cfg.get("cutoff_position", self.positions.max())) - self.positions, 0)
        self.sample_weight = torch.as_tensor(np.power(2., -age / (half_life * C.STEPS_PER_DAY))
                            if half_life > 0 else np.ones_like(age, dtype=float),
                            dtype=torch.float32, device=DEVICE)

    def decode(self, prediction: torch.Tensor, take: torch.Tensor) -> torch.Tensor:
        prediction = prediction.float()
        if self.mode == "scaled":
            return prediction * self.capacity
        if self.mode == "ratio":
            return prediction * self.profile[take]
        if self.mode == "residual":
            if self.residual_limit > 0:
                prediction = self.residual_limit * torch.tanh(prediction)
            return self.profile[take] + prediction * self.capacity
        return prediction

    def loss(self, prediction: torch.Tensor, take: torch.Tensor) -> torch.Tensor:
        target = self.y[take]
        positive = self.valid[take] & (target > 0)
        zero = self.valid[take] & (target == 0)
        safe = torch.nan_to_num(target, nan=0.)
        error = (self.decode(prediction, take) - safe).abs()
        relative = error / safe.abs().clamp_min(self.floor)
        weights = self.h_weight * self.sample_weight[take, None, None]
        # 每个目标独立归约；零负荷保持合法标签，以容量归一化的绝对误差另行约束。
        loss = ((relative * positive * weights).sum((0, 1)) /
                (positive * weights).sum((0, 1)).clamp_min(1.)).mean()
        loss = loss + .1 * ((error / self.capacity * zero).sum((0, 1)) /
                           zero.sum((0, 1)).clamp_min(1.)).mean()
        if self.prior > 0.0 and self.mode == "ratio":
            # 向“完全信任画像”收缩：开发折显示柔性网络会把训练期工况当成规律，
            # 遇到换季后的新工况反而低于画像，用显式先验压住这类偏移。
            loss = loss + self.prior * ((prediction.float() - 1.0) ** 2).mean()
        return loss

    def epoch(self, model: nn.Module, optimizer, amp: bool) -> float:
        model.train()
        order = torch.randperm(self.count, device=DEVICE)
        total = torch.zeros((), device=DEVICE)
        for start in range(0, self.count, self.batch):
            take = order[start:start + self.batch]
            x = _windows(self.table, self.pos[take], self.look_back)
            if self.aug_noise > 0 or self.aug_channel_drop > 0:
                x = x.clone()
                if self.aug_noise > 0:
                    x = x + torch.randn_like(x) * self.aug_noise
                if self.aug_channel_drop > 0:
                    keep = (torch.rand((x.shape[0], 1, x.shape[2]), device=x.device)
                            >= self.aug_channel_drop).to(x.dtype)
                    x = x * keep
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                prediction = model(x, self.future[take])
            loss = self.loss(prediction, take)
            if not torch.isfinite(loss):
                raise ValueError("NN 训练损失出现非有限值")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total = total + loss.detach() * len(take)
        if DEVICE.type == "cuda" and torch.cuda.max_memory_allocated() > self.memory_limit:
            raise RuntimeError("神经网络显存超过本次配置预算，请减小实际 batch")
        return float(total / max(self.count, 1))

    def evaluate(self, model: nn.Module) -> float:
        model.eval()
        total = torch.zeros(2, dtype=torch.float64, device=DEVICE)
        count = torch.zeros(2, dtype=torch.float64, device=DEVICE)
        with torch.inference_mode():
            for start in range(0, self.count, self.batch):
                take = torch.arange(start, min(start + self.batch, self.count), device=DEVICE)
                x = _windows(self.table, self.pos[take], self.look_back)
                prediction = model(x, self.future[take])
                estimate = self.decode(prediction, take)
                small = estimate[..., 0].clamp(0., C.CAPACITY["generator_1"])
                allp = estimate[..., 1].clamp(0., C.CAPACITY["generator_all"])
                allp = torch.maximum(small, torch.minimum(allp, small + C.STRUCTURAL_CAPACITY["generator_big"]))
                estimate = torch.stack((small, allp), dim=-1)
                valid = self.valid[take] & (self.y[take] > 0)
                safe = torch.nan_to_num(self.y[take], nan=1.).clamp_min(1e-12)
                relative = (estimate - safe).abs() / safe
                total += (relative * valid).sum((0, 1), dtype=torch.float64)
                count += valid.sum((0, 1))
        return float((total / count.clamp_min(1.)).mean())


def _make_sample(prepared: PreparedData, table: torch.Tensor, positions: np.ndarray,
                 horizons: tuple[int, ...], cfg: dict, mode: str,
                 batch: int, look_back: int) -> _Sample:
    positions = np.asarray(positions, dtype=np.int64)
    labels = prepared.labels[list(C.TARGETS)].to_numpy(dtype=np.float64)
    index = positions[:, None] + np.asarray(horizons, dtype=np.int64)[None, :]
    inside = index < len(labels)
    targets = labels[np.where(inside, index, 0)]
    targets[~inside] = np.nan
    keep = np.isfinite(targets).any(axis=(1, 2))
    if not keep.any():
        raise ValueError("没有同时满足回看窗口与目标可见的训练样本")
    positions, targets = positions[keep], targets[keep]
    future = _future_columns(prepared, positions, horizons, cfg)
    profile = None
    if mode in ("ratio", "residual"):
        mask_days = int(cfg.get("training_mask_days", 0))
        profile = (_blocked_profiles(prepared, positions, horizons, mask_days)
                   if mask_days > 0 else _profile_cube(prepared, positions, horizons, fuel=_profile_is_fuel(cfg)))
        if cfg.get("profile_conditions"):
            future = _profile_conditions(prepared, positions, future, profile, mask_days)
    return _Sample(table, positions, targets, future, profile, batch, look_back,
                   horizons, mode, cfg)


@dataclass
class NNBundle:
    kind: str
    look_back: int
    horizons: tuple[int, ...]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    state_dict: dict
    # 多种子：多个 state_dict，预测时平均，压训练随机性。
    seed_states: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    train_seconds: float = 0.0
    train_rows: int = 0
    train_end: str | None = None
    feature_names: list[str] = field(default_factory=list)
    n_fut: int = 0
    head_mode: str = "direct"
    load_history: bool = False
    config: dict = field(default_factory=dict)


def train_nn(prepared: PreparedData, folds: list[dict], kind: str = "dlinear",
             horizons: tuple[int, ...] | None = None, quick: bool = False,
             n_seeds: int = 1, cfg: dict | None = None) -> NNBundle:
    """单个训练截止点拟合：前缀内早停选轮数，再按固定轮数重训合法前缀。"""
    prepared.require_training_data()
    horizons = tuple(horizons or C.HORIZONS_L)
    cfg = _cfg(cfg)
    look_back = int(cfg["look_back"])
    epochs = int(cfg["quick_epochs"] if quick else cfg["epochs"])
    patience = int(cfg["quick_patience"] if quick else cfg["patience"])
    batch = int(cfg["quick_batch_size"] if quick else cfg["batch_size"])
    stride = C.QUICK_TRAIN_STRIDE if quick else int(cfg.get("train_stride", 1))
    mode = str(cfg.get("head_mode") or DEFAULT_HEADS.get(kind, "direct"))
    if mode in ("ratio", "residual") and int(C.PROFILE_DAYS) <= 0:
        # 合成自检把画像窗口裁成 0，此时退回原始量纲，保证契约检查可跑通。
        mode = "direct"
    cfg["head_mode"] = mode
    load_history = bool(cfg.get("load_history", False))
    tuning = [f for f in folds if f["role"] == "tuning"]
    if not tuning:
        tuning = [f for f in folds if f["role"] == "confirm"]
    if len(tuning) != 1:
        raise ValueError("NN 每次仅拟合一个训练截止点，跨折评估由调用方组织")
    cutoff = int(prepared.index.get_indexer([pd.Timestamp(tuning[0]["train_end"])])[0])
    if cutoff < 0:
        raise ValueError("NN 训练截止点不在数据网格")
    prepared = prepared.prefix_copy(tuning[0]["train_end"])
    logger = logging.getLogger("aic-half")
    logger.info("NN %s 开始，截止=%s，预算=%d轮，batch=%d，精度=%s，目标空间=%s", kind,
                tuning[0]["train_end"], epochs, batch, cfg.get("precision", "fp32"), mode)
    started = time.time()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hmax = max(horizons)
    first = max(look_back - 1, C.PROFILE_DAYS * C.STEPS_PER_DAY)
    window_days = int(cfg.get("train_window_days", 0) or 0)
    if window_days:
        first = max(first, cutoff - window_days * C.STEPS_PER_DAY + 1)
    cfg["cutoff_position"] = cutoff
    val_start = cutoff - int(cfg["early_stop_days"] * C.STEPS_PER_DAY) + 1
    train_pos = np.arange(first, val_start - hmax, stride)
    val_pos = np.arange(max(first, val_start), cutoff - hmax + 1, stride)
    if not len(train_pos) or not len(val_pos):
        raise ValueError("NN 训练前缀不足以构造带目标间隔的内部早停集")

    feature_set = str(cfg.get("feature_set", "engineered"))
    if load_history and int(cfg.get("training_mask_days", 0)) > 0:
        raise ValueError("连续遮挡训练暂不启用额外负荷历史，避免输入可见性不一致")
    table = build_feature_table(prepared, load_history, feature_set)
    mean, std = _fit_normalizer(table.values, val_start - 1)
    gpu = _gpu_table(table, mean, std)
    fit_sample = _make_sample(prepared, gpu, train_pos, horizons, cfg, mode, batch, look_back)
    validation = prepared.masked_copy(prepared.index[val_start], prepared.index[cutoff])
    val_table = build_feature_table(validation, load_history, feature_set)
    val_gpu = _gpu_table(val_table, mean, std)
    val_sample = _make_sample(validation, val_gpu, val_pos, horizons,
                              {**cfg, "training_mask_days": 0}, mode, batch, look_back)
    n_feat = gpu.shape[1]
    n_fut = fit_sample.future.shape[-1]
    amp = _amp_enabled(str(cfg.get("precision", "fp32")))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    def make_model(seed: int) -> nn.Module:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return build_model(kind, look_back, len(horizons), n_feat, n_fut, cfg)

    def make_schedule(optimizer):
        # 两阶段复用同一学习率前缀，不按选定轮数重新压缩余弦周期。
        return (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                if cfg.get("scheduler") == "cosine" else None)

    seeds = [20261021 + i for i in range(max(1, n_seeds))]
    selected_epochs, val_losses = [], []
    curves = {"train": [], "val": []}
    for seed in seeds:
        model = make_model(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        schedule = make_schedule(optimizer)
        # 未训练画像也是合法候选，不能强迫选中一个比初始基线更差的更新步。
        best = val_sample.evaluate(model) if mode in ("ratio", "residual") else math.inf
        best_epoch, bad = 0, 0
        for epoch in range(1, epochs + 1):
            train_value = fit_sample.epoch(model, optimizer, amp)
            value = val_sample.evaluate(model)
            if schedule is not None:
                schedule.step()
            curves["train"].append(round(float(train_value), 6))
            curves["val"].append(round(float(value), 6))
            if not math.isfinite(value):
                raise ValueError("NN 内部早停损失出现非有限值")
            if value < best - 1e-5:
                best, best_epoch, bad = value, epoch, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        selected_epochs.append(best_epoch)
        val_losses.append(best)
        logger.info("NN %s 内部早停完成: seed=%d, 选定%d轮, loss=%.5f", kind, seed, best_epoch, best)
    del model, optimizer, fit_sample, val_sample, gpu, val_gpu
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # 只按已冻结轮数拟合合法前缀；确认窗口不参与梯度、早停或归一化。
    full_pos = np.arange(first, max(cutoff - hmax + 1, first + 1), stride)
    mean, std = _fit_normalizer(table.values, cutoff)
    gpu = _gpu_table(table, mean, std)
    full_sample = _make_sample(prepared, gpu, full_pos, horizons, cfg, mode, batch, look_back)
    states = []
    for seed, rounds in zip(seeds, selected_epochs):
        model = make_model(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        schedule = make_schedule(optimizer)
        for _ in range(rounds):
            full_sample.epoch(model, optimizer, amp)
            if schedule is not None:
                schedule.step()
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    seconds = time.time() - started
    memory = peak_memory_mb()
    rows = int(full_sample.count)
    logger.info("NN %s 训练完成: %.1fs, 行数=%d, 峰值显存=%.0fMB", kind, seconds, rows, memory)
    del gpu, full_sample
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return NNBundle(kind=kind, look_back=look_back, horizons=horizons,
                    feat_mean=mean, feat_std=std,
                    state_dict=states[0], seed_states=states,
                    scores={"early_stop": {"origin_start": str(prepared.index[val_pos[0]]),
                                           "max_target_time": str(prepared.index[cutoff]),
                                           "metric": "mape_after_physics",
                                           "normalizer_end": str(prepared.index[val_start - 1]),
                                           "target_history_masked": True,
                                           "scheduler_horizon": epochs,
                                           "epochs": selected_epochs, "loss": val_losses, "curves": curves},
                            "seconds": seconds, "peak_memory_mb": memory,
                             "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / 1024 ** 2) if DEVICE.type == "cuda" else 0.,
                             "parameter_count": sum(v.numel() for v in states[0].values()),
                             "device": str(DEVICE), "precision": cfg.get("precision", "fp32"),
                             "effective_precision": "bf16" if amp else "fp32",
                             "batch_size": batch, "feature_count": n_feat,
                            "train_rows": rows},
                    train_seconds=seconds, train_rows=rows,
                    train_end=str(prepared.index[cutoff]), feature_names=list(table.columns),
                    n_fut=n_fut, head_mode=mode, load_history=load_history,
                    config=dict(cfg))


def _predict_cube(model_states: list[dict], table: torch.Tensor, positions: np.ndarray,
                  future: np.ndarray, look_back: int, horizon: int, kind: str,
                   n_fut: int, cfg: dict, batch: int, runtime_models=None) -> np.ndarray:
    # 推理固定 fp32：bf16 在不同批形状下会换内核，批量与单点预测会出现可达 1e-2 的差异，
    # 破坏“同一权重逐值可复验”的契约；训练侧仍用 bf16 提速。
    pos = torch.as_tensor(positions, dtype=torch.long, device=DEVICE)
    fut = torch.as_tensor(future, dtype=torch.float32, device=DEVICE)
    total = torch.zeros((len(positions), horizon, 2), dtype=torch.float32, device=DEVICE)
    if runtime_models is None:
        runtime_models = []
        for state in model_states:
            model = build_model(kind, look_back, horizon, table.shape[1], n_fut, cfg)
            model.load_state_dict(state)
            runtime_models.append(model)
    for model in runtime_models:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(positions), max(1, batch)):
                take = slice(start, start + max(1, batch))
                x = _windows(table, pos[take], look_back)
                total[take] += model(x, fut[take]).float()
    return (total / len(model_states)).cpu().numpy()


def predict_nn(bundle: NNBundle, prepared: PreparedData,
               origins: pd.DatetimeIndex) -> np.ndarray:
    """输出 (n_origins, n_horizons, 2)；越界起报点保持 NaN 由调用方回退画像。"""
    from main_s import enforce_physics

    positions = prepared.index.get_indexer(origins)
    n, m = len(origins), len(bundle.horizons)
    out = np.full((n, m, 2), np.nan, dtype=float)
    valid = (positions >= 0) & (positions - bundle.look_back + 1 >= 0)
    if not valid.any():
        return out
    table = build_feature_table(prepared, bundle.load_history,
                                str(bundle.config.get("feature_set", "engineered")))
    if list(table.columns) != list(bundle.feature_names):
        raise ValueError("NN 特征列顺序与训练不一致")
    sub = positions[valid]
    gpu = _gpu_table(table, bundle.feat_mean, bundle.feat_std)
    cfg = dict(bundle.config)
    cfg["head_mode"] = bundle.head_mode
    future = _future_columns(prepared, sub, bundle.horizons, cfg)
    profile = None
    if bundle.head_mode in ("ratio", "residual"):
        profile = _profile_cube(prepared, sub, bundle.horizons, fuel=_profile_is_fuel(cfg))
        if cfg.get("profile_conditions"):
            future = _profile_conditions(prepared, sub, future, profile)
    states = bundle.seed_states or [bundle.state_dict]
    runtime = getattr(bundle, "_runtime_models", None)
    if runtime is None:
        runtime = []
        for state in states:
            model = build_model(bundle.kind, bundle.look_back, m, gpu.shape[1], bundle.n_fut, cfg)
            model.load_state_dict(state)
            runtime.append(model)
        bundle._runtime_models = runtime
    cube = _predict_cube(states, gpu, sub, future, bundle.look_back, m, bundle.kind,
                         bundle.n_fut, cfg, int(cfg.get("predict_batch", 1024)), runtime_models=runtime)
    if bundle.head_mode == "ratio":
        cube = cube * profile
    elif bundle.head_mode == "residual":
        limit = float(cfg.get("residual_limit", 0.) or 0.)
        if limit > 0:
            cube = limit * np.tanh(cube)
        cube = profile + cube * np.array([C.CAPACITY[t] for t in C.TARGETS])
    elif bundle.head_mode == "scaled":
        cube = cube * np.array([C.CAPACITY[t] for t in C.TARGETS])
    cube = np.clip(cube, 0.0, None)
    out[valid] = cube
    for row in np.nonzero(~valid)[0]:
        out[row] = np.nan
    return enforce_physics(out, fallback=None)


def save_nn(bundle: NNBundle, directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"nn_{bundle.kind}.pt"
    torch.save({
        "kind": bundle.kind,
        "look_back": bundle.look_back,
        "horizons": list(bundle.horizons),
        "feat_mean": bundle.feat_mean,
        "feat_std": bundle.feat_std,
        "state_dict": bundle.state_dict,
        "seed_states": bundle.seed_states,
        "scores": bundle.scores,
        "train_end": bundle.train_end,
        "feature_names": bundle.feature_names,
        "n_fut": bundle.n_fut,
        "head_mode": bundle.head_mode,
        "load_history": bundle.load_history,
        "config": bundle.config,
    }, path)
    return str(path)


def load_nn(path: Path) -> NNBundle:
    """仅重载本工程保存的本地权重，CPU 端可复验实际多种子预测。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return NNBundle(kind=payload["kind"], look_back=payload["look_back"],
                    horizons=tuple(payload["horizons"]),
                    feat_mean=payload["feat_mean"], feat_std=payload["feat_std"],
                    state_dict=payload["state_dict"], seed_states=payload.get("seed_states", []),
                    scores=payload.get("scores", {}), train_end=payload.get("train_end"),
                    feature_names=payload.get("feature_names", []),
                    n_fut=int(payload.get("n_fut", 0)),
                    head_mode=payload.get("head_mode", "direct"),
                    load_history=bool(payload.get("load_history", False)),
                    config=payload.get("config", {}))


def benchmark_engine(directory: Path, kind: str, cfg: dict) -> list[dict]:
    """固定更新步的合成算子基准，不产生预测分数，权重不得用于赛事模型。"""
    if DEVICE.type != "cuda":
        raise RuntimeError("GPU 基准需要可用 CUDA")
    directory.mkdir(parents=True, exist_ok=True)
    cfg = {**_cfg(cfg), "head_mode": "scaled", "aug_noise": 0., "aug_channel_drop": 0.}
    length, horizon, features = int(cfg["look_back"]), 8, 192
    results = []
    for precision in ("fp32", "bf16"):
        for batch in (128, 512, 1024, 2048, 4096):
            if batch == 4096 and precision == "fp32":
                continue
            torch.manual_seed(73)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = build_model(kind, length, horizon, features, 6, cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            x = torch.randn(batch, length, features, device=DEVICE)
            future = torch.zeros(batch, horizon, 6, device=DEVICE)
            target = torch.full((batch, horizon, 2), .6, device=DEVICE)
            item = {"batch": batch, "precision": precision, "steps": 12,
                    "synthetic_only": True, "kind": kind, "look_back": length}
            prediction = loss = None
            try:
                for step in range(15):
                    if step == 3:
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                        prediction = model(x, future)
                    loss = (prediction.float() - target).abs().mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                item.update(seconds=seconds, step_ms=seconds / 12 * 1000,
                            samples_per_second=batch * 12 / seconds,
                            peak_allocated_mb=peak_memory_mb(),
                            peak_reserved_mb=torch.cuda.max_memory_reserved() / 1024 ** 2)
                torch.save({"purpose": "synthetic_benchmark_only", "state_dict": model.cpu().state_dict()},
                           directory / f"{kind}_{precision}_{batch}.pt")
            except torch.cuda.OutOfMemoryError:
                item["status"] = "OOM_not_selected"
            results.append(item)
            logging.getLogger("aic-half").info("GPU基准: %s", item)
            del model, optimizer, x, future, target, prediction, loss
            torch.cuda.empty_cache()
    return results
