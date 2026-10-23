"""神经网络支路：DLinear / N-HiTS / TCN，主攻长周期外推。

设计要点：
1. 与树模型共用 15min 宽表与折协议；输出同为 (n_origins, n_horizons, 2) 立方体。
2. 归一化只用训练段统计量（RevIN 可选），避免评测窗信息泄漏。
3. 损失为 MAPE 加权 L1，远端提前量加权，直接对齐 1-MAPE 口径。
4. 一切实验写 tmp/nn/；只有调参折达标的模型才进正式归档。
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
