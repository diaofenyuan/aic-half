"""负荷 DLinear / N-HiTS / TCN 与独立八步资源残差网络。

设计要点：
1. 负荷网络输出两目标；资源网络独立输出八步三气种，不共用标签或预测头。
2. 归一化只用训练段统计量（RevIN 可选），避免评测窗信息泄漏。
3. 负荷采用 MAPE 加权 L1；资源兼顾逐气种与联合能量误差，并按已知电价加权。
4. 实验完整写入 tmp/tmp_outputs/；只有通过开发折晋级的实际模型进入运行归档。
"""
from __future__ import annotations

import json
import math
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
from data_cleaning import PreparedData, build_base_features


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 固定随机性：CUDA 卷积/矩阵乘若不确定，同参会差 0.2pp，掩盖真实收益。
torch.manual_seed(20261021)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


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


def _resource_nn_features(values: np.ndarray, positions: np.ndarray,
                          fit_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = values[:fit_end + 1]
    center = np.nan_to_num(np.nanmedian(history, axis=0), nan=0.)
    scale = np.nanpercentile(history, 75, axis=0) - np.nanpercentile(history, 25, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-5), scale, 1.)
    selected = values[positions]
    z = np.clip((selected - center) / scale, -10., 10.)
    # 标准化后的中性值与缺失指示同时输入，不把过程缺测伪装成真实零供气。
    X = np.concatenate([np.nan_to_num(z, nan=0.), np.isfinite(selected).astype(np.float32)], axis=1)
    return X.astype(np.float32), center, scale


def _fit_resource_nn(X, anchor, target, energy_weights, price_weights,
                     epochs: int, validation=None, seed: int = 73):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _ResourceResidualMLP(X.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    tensors = [torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
               for a in (X, anchor, target, price_weights)]
    ew = torch.as_tensor(energy_weights, dtype=torch.float32, device=DEVICE)
    best, best_epoch, stale = np.inf, 1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X), device=DEVICE)
        for start in range(0, len(X), 256):
            take = order[start:start + 256]
            prediction = model(tensors[0][take], tensors[1][take])
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
                vx, va, vy = validation
                prediction = model(vx, va)
                mask = torch.isfinite(vy)
                error = torch.abs(prediction - torch.nan_to_num(vy, nan=0.)) * mask * ew
                score = float(error.sum() / (mask * ew).sum())
            if score < best - 1e-5:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            if stale >= 10:
                break
    return model, best_epoch, best


def train_resource_nn(base: pd.DataFrame, resource: pd.DataFrame, end_pos: int,
                       heat: dict, prices: dict, quick: bool = False, seed: int = 73) -> dict:
    """仅在起报前前缀选轮数，再以该轮数重训；不访问外层验证标签。"""
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    values = base.to_numpy(dtype=np.float32)
    actual = resource[list(C.RESOURCE_NAMES)].to_numpy(dtype=np.float32)
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
    inner_X, _, _ = _resource_nn_features(values, positions, split - 1)
    validation = tuple(torch.as_tensor(v, dtype=torch.float32, device=DEVICE)
                       for v in (inner_X[valid], inner_anchor[valid], inner_target[valid]))
    _, epochs, error = _fit_resource_nn(inner_X[train], inner_anchor[train], inner_target[train],
        energy_weights, price_weight[train], 2 if quick else 100, validation, seed)
    X, center, scale = _resource_nn_features(values, positions, end_pos)
    model, _, _ = _fit_resource_nn(X, anchor, target, energy_weights, price_weight, epochs, seed=seed)
    return {"kind": "resource_mlp", "state_dict": model.cpu().state_dict(), "width": X.shape[1],
            "epochs": epochs, "inner_error": error, "feature_names": list(base.columns),
            "center": center, "scale": scale, "gas_scale": gas_scale,
            "train_end": str(base.index[end_pos]), "inner_train_end": str(base.index[split - 1]),
            "max_training_target": str(base.index[positions[-1] + 8]), "seed": seed}


def predict_resource_nn(pack: dict, base: pd.DataFrame, resource: pd.DataFrame,
                         positions: np.ndarray) -> np.ndarray:
    if list(base.columns) != pack["feature_names"]:
        raise ValueError("资源网络特征顺序不一致")
    model = pack.get("_runtime_model")
    if model is None:
        model = _ResourceResidualMLP(pack["width"]).to(DEVICE)
        model.load_state_dict(pack["state_dict"])
        model.eval()
        pack["_runtime_model"] = model
    values = base.to_numpy(dtype=np.float32)[positions]
    z = np.clip((values - pack["center"]) / pack["scale"], -10., 10.)
    X = np.concatenate([np.nan_to_num(z, nan=0.), np.isfinite(values).astype(np.float32)], axis=1).astype(np.float32)
    actual = resource[list(C.RESOURCE_NAMES)].to_numpy(dtype=np.float32)[positions]
    anchor = np.nan_to_num(actual / pack["gas_scale"], nan=0.)
    with torch.no_grad():
        output = model(torch.as_tensor(X, device=DEVICE),
                       torch.as_tensor(anchor, device=DEVICE)).cpu().numpy()
    return np.maximum(output * pack["gas_scale"], 0.)


def save_resource_nn(pack: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in pack.items() if not k.startswith("_")}, path)
    return path


def load_resource_nn(path: Path) -> dict:
    pack = torch.load(path, map_location="cpu", weights_only=False)
    if pack.get("kind") != "resource_mlp":
        raise ValueError("不是独立八步资源网络")
    return pack


# ---------------------------------------------------------------------------
# 特征张量化
# ---------------------------------------------------------------------------
@dataclass
class NNTensor:
    """滑窗样本：X (n, look_back, n_feat)，Y (n, n_horizon, 2)，含起报点位置。"""
    X: np.ndarray
    Y: np.ndarray
    positions: np.ndarray
    feat_mean: np.ndarray
    feat_std: np.ndarray


def _normalize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std >= 1e-8), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def build_nn_dataset(prepared: PreparedData, positions: np.ndarray,
                     horizons: tuple[int, ...], look_back: int,
                     mean: np.ndarray | None = None,
                     std: np.ndarray | None = None,
                      require_target: bool = True,
                      feature_names: list[str] | None = None) -> NNTensor:
    """把起报点转成滑窗样本；mean/std 不传则用本集合统计（仅训练时合法）。

    require_target=False 用于推理：目标可为 NaN，Y 用 0 占位，只取 X。
    """
    from main_s import _feature_columns

    base = build_base_features(prepared, include_targets=True)
    cols = _feature_columns(base, "gas")
    if feature_names and cols != feature_names:
        raise ValueError("NN 特征列顺序与训练不一致")
    values = base[cols].to_numpy(dtype=np.float32)
    labels = prepared.labels[list(C.TARGETS)].to_numpy(dtype=float)

    n = len(positions)
    h_max = max(horizons)
    # 只保留 look_back 与目标都可见的位置
    pos_list, xs, ys = [], [], []
    for p in positions:
        p = int(p)
        if p - look_back + 1 < 0 or p >= len(values):
            continue
        if require_target and p + h_max >= len(labels):
            continue
        window = values[p - look_back + 1: p + 1]
        target = (labels[p + np.asarray(horizons, dtype=int)] if require_target
                  else np.zeros((len(horizons), len(C.TARGETS)), dtype=float))
        # 缺失值在标准化后填零，即训练均值；避免先补原始零污染均值和方差。
        window = np.where(np.isfinite(window), window, np.nan)
        if require_target and not np.isfinite(target).all():
            continue
        if not require_target:
            target = np.nan_to_num(target, nan=0.0)
        pos_list.append(p)
        xs.append(window)
        ys.append(target)

    if not pos_list:
        raise ValueError("没有合法的 NN 训练样本，请检查 look_back/horizons")
    X = np.stack(xs).astype(np.float32)
    Y = np.stack(ys).astype(np.float32)

    if mean is None or std is None:
        flat = X.reshape(-1, X.shape[-1])
        mean, std = _normalize_fit(flat)
    X = (X - mean) / std
    X = np.nan_to_num(X, nan=0.0)
    return NNTensor(X=X, Y=Y, positions=np.asarray(pos_list, dtype=np.int64),
                    feat_mean=mean, feat_std=std)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class DLinear(nn.Module):
    """序列线性分解（趋势 + 季节）后分别线性映射到多步，长周期强基线。

    多特征情形：对时间维做分解与 L->H 线性映射，再把特征维混合到双目标。
    """

    def __init__(self, look_back: int, horizon: int, n_feat: int):
        super().__init__()
        self.horizon = horizon
        self.trend_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.linear_trend = nn.Linear(look_back, horizon)
        self.linear_season = nn.Linear(look_back, horizon)
        self.mix = nn.Linear(n_feat, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        series = x.transpose(1, 2)                            # (B, F, L)
        trend = self.trend_pool(series)                       # (B, F, L)
        season = series - trend
        out_t = self.linear_trend(trend)                      # (B, F, H)
        out_s = self.linear_season(season)
        merged = (out_t + out_s).transpose(1, 2)              # (B, H, F)
        return self.mix(merged)                               # (B, H, 2)


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
        # x: (B, L, F) -> 特征压缩到 d_model，再时间池化后进 MLP
        z = torch.relu(self.proj_in(x))                        # (B, L, D)
        z = self.pool(z.transpose(1, 2)).transpose(1, 2)       # (B, L/k, D)
        z = z.reshape(z.size(0), -1)
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        return self.head(z).view(-1, self.head.out_features // 2, 2)


class NHiTS(nn.Module):
    """分层插值：多块不同池化粒度，叠加残差，专攻长外推。"""

    def __init__(self, look_back: int, horizon: int, n_feat: int,
                 d_model: int = 64, n_blocks: int = 3):
        super().__init__()
        kernels = [1, 2, 4][:n_blocks]
        self.blocks = nn.ModuleList([
            _NHiTSBlock(look_back, horizon, n_feat, d_model, k) for k in kernels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0.0
        for blk in self.blocks:
            out = out + blk(x)
        return out


class TCNBlock(nn.Module):
    def __init__(self, n_feat: int, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(n_feat, channels, kernel_size=3,
                              dilation=dilation, padding=dilation)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class TCN(nn.Module):
    """膨胀因果卷积，短周期/资源可选。"""

    def __init__(self, look_back: int, horizon: int, n_feat: int,
                 channels: int = 32, n_layers: int = 3):
        super().__init__()
        layers = [TCNBlock(n_feat, channels, 1)]
        for i in range(1, n_layers):
            layers.append(TCNBlock(channels, channels, 2 ** i))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(channels * look_back, horizon * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)          # (B, F, L)
        z = self.net(z)
        z = z.reshape(z.size(0), -1)
        return self.head(z).view(-1, self.head.out_features // 2, 2)


MODEL_ZOO = {"dlinear": DLinear, "nhits": NHiTS, "tcn": TCN}


# ---------------------------------------------------------------------------
# 训练与推理
# ---------------------------------------------------------------------------
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


def _mape_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                  horizons: tuple[int, ...]) -> torch.Tensor:
    """按 1/|y| 加权 L1，远端提前量加权 1.5x。"""
    err = torch.abs(pred - target)
    weight = 1.0 / torch.clamp(target.abs(), min=1.0)
    h_arr = torch.tensor(horizons, dtype=torch.float32, device=pred.device)
    h_w = torch.where(h_arr > 32, 1.5, 1.0).view(1, -1, 1)
    return (err * weight * h_w).mean()


def train_nn(prepared: PreparedData, folds: list[dict], kind: str = "dlinear",
             horizons: tuple[int, ...] | None = None, quick: bool = False,
             n_seeds: int = 1) -> NNBundle:
    """单个训练截止点拟合：前缀内早停选轮数，再按固定轮数重训合法前缀。"""
    horizons = tuple(horizons or C.HORIZONS_L)
    cfg = C.NN_CONFIG
    look_back = int(cfg["look_back"])
    epochs = int(cfg["quick_epochs"] if quick else cfg["epochs"])
    patience = int(cfg["quick_patience"] if quick else cfg["patience"])
    batch = int(cfg["quick_batch_size"] if quick else cfg["batch_size"])
    stride = C.QUICK_TRAIN_STRIDE if quick else 1
    started = time.time()
    tuning = [f for f in folds if f["role"] == "tuning"]
    if len(tuning) != 1:
        raise ValueError("NN 每次仅拟合一个训练截止点，跨折评估由调用方组织")
    cutoff = int(prepared.index.get_indexer([pd.Timestamp(tuning[0]["train_end"])])[0])
    logger = logging.getLogger("aic-half")
    logger.info("NN %s 开始，截止=%s，预算=%d轮，设备=%s", kind,
                tuning[0]["train_end"], epochs, DEVICE)
    if cutoff < 0:
        raise ValueError("NN 训练截止点不在数据网格")
    hmax = max(horizons)
    first = max(look_back - 1, C.PROFILE_DAYS * C.STEPS_PER_DAY)
    val_start = cutoff - int(cfg["early_stop_days"] * C.STEPS_PER_DAY) + 1
    train_pos = np.arange(first, val_start - hmax, stride)
    val_pos = np.arange(max(first, val_start), cutoff - hmax + 1, stride)
    if not len(train_pos) or not len(val_pos):
        raise ValueError("NN 训练前缀不足以构造带目标间隔的内部早停集")
    ds = build_nn_dataset(prepared, train_pos, horizons, look_back)
    val_ds = build_nn_dataset(prepared, val_pos, horizons, look_back,
                              ds.feat_mean, ds.feat_std)
    n_feat = ds.X.shape[-1]

    def make_model(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return MODEL_ZOO[kind](look_back, len(horizons), n_feat).to(DEVICE)

    def epoch_fit(model, opt, X, Y):
        model.train()
        perm = torch.randperm(len(X), device=DEVICE)
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            loss = _mape_l1_loss(model(X[idx]), Y[idx], horizons)
            if not torch.isfinite(loss):
                raise ValueError("NN 训练损失出现非有限值")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    X = torch.from_numpy(ds.X).to(DEVICE)
    Y = torch.from_numpy(ds.Y).to(DEVICE)
    VX = torch.from_numpy(val_ds.X).to(DEVICE)
    VY = torch.from_numpy(val_ds.Y).to(DEVICE)
    seeds = [20261021 + i for i in range(max(1, n_seeds))]
    selected_epochs, val_losses = [], []
    for seed in seeds:
        model = make_model(seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, best_epoch, bad = math.inf, 1, 0
        for epoch in range(epochs):
            epoch_fit(model, opt, X, Y)
            model.eval()
            total = 0.0
            with torch.no_grad():
                for i in range(0, len(VX), batch):
                    loss = _mape_l1_loss(model(VX[i:i + batch]), VY[i:i + batch], horizons)
                    total += float(loss) * len(VX[i:i + batch])
            value = total / len(VX)
            if not math.isfinite(value):
                raise ValueError("NN 内部早停损失出现非有限值")
            if value < best - 1e-5:
                best, best_epoch, bad = value, epoch + 1, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        selected_epochs.append(best_epoch)
        val_losses.append(best)
        logger.info("NN %s 内部早停完成: seed=%d, 选定%d轮, loss=%.5f", kind, seed, best_epoch, best)
    del X, Y, VX, VY, model, opt

    # 只按已冻结轮数拟合，确认窗口既不参与梯度，也不参与早停或归一化。
    full_pos = np.arange(first, cutoff - hmax + 1, stride)
    full_ds = build_nn_dataset(prepared, full_pos, horizons, look_back)
    X = torch.from_numpy(full_ds.X).to(DEVICE)
    Y = torch.from_numpy(full_ds.Y).to(DEVICE)
    states = []
    for seed, rounds in zip(seeds, selected_epochs):
        model = make_model(seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(rounds):
            epoch_fit(model, opt, X, Y)
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    from main_s import _feature_columns
    names = _feature_columns(build_base_features(prepared, include_targets=True), "gas")
    return NNBundle(kind=kind, look_back=look_back, horizons=horizons,
                    feat_mean=full_ds.feat_mean, feat_std=full_ds.feat_std,
                    state_dict=states[0], seed_states=states,
                    scores={"early_stop": {"origin_start": str(prepared.index[val_pos[0]]),
                                           "max_target_time": str(prepared.index[cutoff]),
                                           "epochs": selected_epochs, "loss": val_losses}},
                    train_seconds=time.time() - started, train_rows=len(full_ds.positions),
                    train_end=str(prepared.index[cutoff]), feature_names=names)


def predict_nn(bundle: NNBundle, prepared: PreparedData,
               origins: pd.DatetimeIndex) -> np.ndarray:
    """输出 (n_origins, n_horizons, 2)；越界起报点自动跳过并回退 profile。"""
    from main_s import enforce_physics

    positions = prepared.index.get_indexer(origins)
    n, m = len(origins), len(bundle.horizons)
    out = np.full((n, m, 2), np.nan, dtype=float)
    valid_mask = (positions >= 0) & (positions - bundle.look_back + 1 >= 0)
    if not valid_mask.any():
        return out
    ds = build_nn_dataset(prepared, positions[valid_mask], bundle.horizons,
                          bundle.look_back, bundle.feat_mean, bundle.feat_std,
                          require_target=False, feature_names=bundle.feature_names)
    n_feat = ds.X.shape[-1]
    states = bundle.seed_states or [bundle.state_dict]
    pred = None
    for st in states:
        model = MODEL_ZOO[bundle.kind](bundle.look_back, m, n_feat).to(DEVICE)
        model.load_state_dict({k: v.to(DEVICE) for k, v in st.items()})
        model.eval()
        with torch.no_grad():
            pr = model(torch.from_numpy(ds.X).to(DEVICE)).cpu().numpy()
        pred = pr if pred is None else pred + pr
    pred = pred / len(states)
    pred = enforce_physics(pred, fallback=None)
    # 按实际保留位置回填，不能假设张量构造永远保留所有候选行。
    slots = {int(p): row for row, p in enumerate(ds.positions)}
    for row, p in enumerate(positions):
        if int(p) in slots:
            out[row] = pred[slots[int(p)]]
    return out


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
                    feature_names=payload.get("feature_names", []))
