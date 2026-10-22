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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
from data_cleaning import PreparedData, build_base_features
from main_s import _feature_columns, score_cube, truth_cube, enforce_physics


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
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def build_nn_dataset(prepared: PreparedData, positions: np.ndarray,
                     horizons: tuple[int, ...], look_back: int,
                     mean: np.ndarray | None = None,
                     std: np.ndarray | None = None,
                     require_target: bool = True) -> NNTensor:
    """把起报点转成滑窗样本；mean/std 不传则用本集合统计（仅训练时合法）。

    require_target=False 用于推理：目标可为 NaN，Y 用 0 占位，只取 X。
    """
    base = build_base_features(prepared, include_targets=True)
    cols = _feature_columns(base, "gas")
    values = base[cols].to_numpy(dtype=np.float32)
    labels = prepared.labels[list(C.TARGETS)].to_numpy(dtype=float)

    n = len(positions)
    h_max = max(horizons)
    # 只保留 look_back 与目标都可见的位置
    pos_list, xs, ys = [], [], []
    for p in positions:
        p = int(p)
        if p - look_back + 1 < 0 or p + h_max >= len(labels):
            continue
        window = values[p - look_back + 1: p + 1]
        target = labels[p + np.asarray(horizons, dtype=int)]
        if not np.isfinite(window).all():
            # 窗口内 NaN 用列均值临时填；正式归一化后再填一次。
            window = np.nan_to_num(window, nan=0.0)
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
    horizons = tuple(horizons or C.HORIZONS_L)
    look_back = 96
    epochs = 40 if quick else 80
    lr = 1e-3
    batch = 64 if not quick else 256
    started = time.time()

    tuning = [f for f in folds if f["role"] == "tuning"]
    confirm = next(f for f in folds if f["role"] == "confirm")

    # 用第一折训练段拟合归一化
    f0 = tuning[0]
    train_end = int(prepared.index.get_indexer([pd.Timestamp(f0["train_end"])])[0])
    train_pos = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY),
                          max(train_end - max(horizons), 0))
    if quick:
        train_pos = train_pos[::8]
    ds0 = build_nn_dataset(prepared, train_pos, horizons, look_back)
    mean, std = ds0.feat_mean, ds0.feat_std

    datasets = []
    for fold in tuning:
        end = int(prepared.index.get_indexer([pd.Timestamp(fold["train_end"])])[0])
        pos = np.arange(max(0, C.PROFILE_DAYS * C.STEPS_PER_DAY),
                        max(end - max(horizons), 0))
        if quick:
            pos = pos[::8]
        datasets.append(build_nn_dataset(prepared, pos, horizons, look_back, mean, std))

    # 验证集：确认折起报点（仅用于早停参考，不参与调参决策）
    c_start = pd.Timestamp(confirm["start"])
    c_end = c_start + pd.Timedelta(days=int(confirm["days"]))
    c_origins = pd.date_range(c_start,
                              c_end - pd.Timedelta(minutes=C.STEP_MINUTES),
                              freq=f"{C.STEP_MINUTES}min")
    c_pos = prepared.index.get_indexer(c_origins)
    c_pos = c_pos[c_pos >= 0]
    if quick:
        c_pos = c_pos[::8]
    val_ds = build_nn_dataset(prepared, c_pos, horizons, look_back, mean, std)

    n_feat = ds0.X.shape[-1]

    def to_t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).to(DEVICE)

    def train_one(seed: int) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 31 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = MODEL_ZOO[kind](look_back, len(horizons), n_feat).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_val = math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        patience, bad = 20, 0
        for epoch in range(epochs):
            model.train()
            for ds in datasets:
                X, Y = to_t(ds.X), to_t(ds.Y)
                perm = torch.randperm(len(X), device=DEVICE)
                for i in range(0, len(X), batch):
                    idx = perm[i:i + batch]
                    pred = model(X[idx])
                    loss = _mape_l1_loss(pred, Y[idx], horizons)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = _mape_l1_loss(model(to_t(val_ds.X)), to_t(val_ds.Y), horizons)
            v = float(val_loss)
            if v < best_val - 1e-5:
                best_val = v
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        model.load_state_dict(best_state)
        return {k: v.cpu() for k, v in best_state.items()}

    seeds = [20261021 + i for i in range(max(1, n_seeds))]
    seed_states = [train_one(s) for s in seeds]
    rows = sum(len(ds.X) for ds in datasets)

    # 确认折评分（只作记录）：多种子预测平均
    def predict_states(states, X):
        acc = None
        for st in states:
            m = MODEL_ZOO[kind](look_back, len(horizons), n_feat).to(DEVICE)
            m.load_state_dict({k: v.to(DEVICE) for k, v in st.items()})
            m.eval()
            with torch.no_grad():
                pr = m(to_t(X)).cpu().numpy()
            acc = pr if acc is None else acc + pr
        return acc / len(states)

    pred = predict_states(seed_states, val_ds.X)
    pred = enforce_physics(pred, fallback=None)
    sc = score_cube(pred, val_ds.Y, horizons)
    return NNBundle(kind=kind, look_back=look_back, horizons=horizons,
                    feat_mean=mean, feat_std=std,
                    state_dict=seed_states[0],
                    seed_states=seed_states,
                    scores={"confirm": sc},
                    train_seconds=time.time() - started,
                    train_rows=rows)


def predict_nn(bundle: NNBundle, prepared: PreparedData,
               origins: pd.DatetimeIndex) -> np.ndarray:
    """输出 (n_origins, n_horizons, 2)；越界起报点自动跳过并回退 profile。"""
    positions = prepared.index.get_indexer(origins)
    n, m = len(origins), len(bundle.horizons)
    out = np.full((n, m, 2), np.nan, dtype=float)
    valid_mask = (positions >= 0) & (positions - bundle.look_back + 1 >= 0)
    if not valid_mask.any():
        return out
    ds = build_nn_dataset(prepared, positions[valid_mask], bundle.horizons,
                          bundle.look_back, bundle.feat_mean, bundle.feat_std,
                          require_target=False)
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
    out[valid_mask] = pred
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
        "scores": bundle.scores,
    }, path)
    return str(path)
