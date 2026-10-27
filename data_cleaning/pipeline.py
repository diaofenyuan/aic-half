"""只读同刻采样、来源记录及不依赖目标负荷的因果特征。"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import config as C


@dataclass
class PreparedData:
    frame: pd.DataFrame
    labels: pd.DataFrame
    sources: list
    quality: dict
    scope: str = "train"
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def index(self):
        return self.frame.index

    def prefix(self, cutoff):
        end = pd.Timestamp(cutoff)
        return PreparedData(self.frame.loc[:end].copy(), self.labels.loc[:end].copy(),
                            self.sources, self.quality, self.scope, self.prices)

    def masked_copy(self, start, end=None, mask_use=False):
        f = self.frame.copy()
        rows = f.index >= pd.Timestamp(start)
        if end is not None:
            rows &= f.index <= pd.Timestamp(end)
        f.loc[rows, ["load_" + x for x in C.TARGETS]] = np.nan
        if mask_use:
            f.loc[f.index >= pd.Timestamp(start) + pd.Timedelta(hours=48), list(C.USE)] = np.nan
        return PreparedData(f, self.labels, self.sources, self.quality, self.scope, self.prices)

    def require_training(self, cutoff):
        if self.scope != "train" or self.index.max() > pd.Timestamp(cutoff):
            raise ValueError("训练入口拒绝测试视图或超过标签截止的数据")


def read_csv(path):
    for enc in ("utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"数据编码无法识别：{path}")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load_prices():
    raw = pd.read_excel(C.DATA_ROOT / "first_train" / "price.xlsx")
    a = np.full((12, 96), np.nan)
    for _, row in raw.iterrows():
        hour, minute = map(int, str(row.iloc[0]).split("-")[0].strip().split(":"))
        slot = hour * 4 + minute // 15
        a[:, slot:slot+2] = pd.to_numeric(row.iloc[1:13]).to_numpy()[:, None]
    if not np.isfinite(a).all():
        raise ValueError("全年电价覆盖不完整")
    return pd.DataFrame(a)


def prices_at(prepared, times):
    t = pd.DatetimeIndex(times)
    return prepared.prices.to_numpy()[t.month-1, t.hour*4+t.minute//15]


def prepare_data(include_test=False, cutoff=None):
    if C.HOLDER_INPUT_VERSION not in C.HOLDER_SCALES:
        raise ValueError("必须声明有效的柜位输入版本")
    pieces = {k: [] for k in C.TABLES}
    sources, invalid = [], {}
    for stage, begin, stop, role in C.STAGES:
        if role == "test" and not include_test:
            continue
        if cutoff is not None and pd.Timestamp(begin) > pd.Timestamp(cutoff):
            continue
        for table in C.TABLES:
            suffixes = ("load.csv", "load_real.csv") if table == "load" else (table + ".csv",)
            hits = [p for p in (C.DATA_ROOT/stage).glob("*.csv") if p.name.lower().endswith(suffixes)]
            if len(hits) != 1:
                raise ValueError(f"{stage}/{table}：需要唯一数据表，实际{len(hits)}份")
            p = hits[0]
            raw = read_csv(p)
            if "datetime" not in raw:
                raise ValueError(f"{p.name}缺少datetime")
            raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
            if raw.datetime.isna().any():
                raise ValueError(f"{p.name}存在非法时间戳")
            raw = raw.set_index("datetime").sort_index().loc[begin:stop]
            if cutoff is not None:
                raw = raw.loc[:pd.Timestamp(cutoff)]
            # 同刻采样先于统计，不将后续分钟值反向赋给当前起点。
            raw = raw[(raw.index.minute % 15 == 0) & (raw.index.second == 0)]
            if raw.index.has_duplicates:
                if raw.groupby(level=0).nunique(dropna=False).gt(1).any().any():
                    raise ValueError(f"{p.name}同时间戳观测冲突")
                raw = raw[~raw.index.duplicated()]
            raw = raw.apply(pd.to_numeric, errors="coerce")
            if role == "test" and table == "load" and raw.reindex(columns=C.TARGETS).notna().any().any():
                raise ValueError("测试标签必须为空，禁止读取隐藏真值")
            bad = ~np.isfinite(raw) | (raw.abs() >= 1e29) | (raw < 0.)
            invalid[f"{stage}/{table}"] = int(bad.to_numpy().sum())
            raw = raw.mask(bad)
            if table == "gas_holder":
                for col, scale in zip((h.removeprefix("gas_holder_") for h in C.HOLDERS),
                                      C.HOLDER_SCALES[C.HOLDER_INPUT_VERSION]):
                    raw[col] = raw[col] * scale
            pieces[table].append(raw.add_prefix(table + "_"))
            sources.append({"path": str(p.relative_to(C.DATA_ROOT)), "role": role,
                            "sha256": file_hash(p), "grid_rows": len(raw)})
    tables = []
    for table in C.TABLES:
        f = pd.concat(pieces[table]).sort_index()
        if f.index.has_duplicates:
            raise ValueError(f"{table}阶段时间范围发生重叠")
        tables.append(f)
    frame = pd.concat(tables, axis=1)
    index = pd.date_range(frame.index.min(), frame.index.max(), freq="15min", name="datetime")
    frame = frame.reindex(index)
    for name, cap in zip(C.TARGETS, (200., 440.)):
        col = "load_" + name
        frame.loc[frame[col] > cap + .01, col] = np.nan
    diff = frame.load_generator_all - frame.load_generator_1
    inconsistent = diff.notna() & ((diff < -.01) | (diff > 240.01))
    frame.loc[inconsistent, ["load_" + t for t in C.TARGETS]] = np.nan
    labels = frame[["load_" + t for t in C.TARGETS]].copy()
    labels.columns = list(C.TARGETS)
    quality = {"invalid_values": invalid, "holder_input_version": C.HOLDER_INPUT_VERSION,
               "holder_scales_applied_once": C.HOLDER_SCALES[C.HOLDER_INPUT_VERSION],
               "same_timestamp_sampling": True, "rows": len(index),
               "target_valid": labels.notna().sum().to_dict(),
               "holder_valid": frame[list(C.HOLDERS)].notna().sum().to_dict()}
    return PreparedData(frame, labels, sources, quality, "test" if include_test else "train", load_prices())


def process_series(prepared):
    f = prepared.frame
    out = pd.DataFrame(index=f.index)
    for name, cols in (
        ("blast_13", [f"gas_blast_furnace_{i}" for i in (1, 2, 3)]),
        ("blast_45", [f"gas_blast_furnace_{i}" for i in (4, 5)]),
        ("known_demand", [f"gas_air_heater_{i}" for i in range(1, 6)] +
         [f"gas_user_blast_furnace_user{i}" for i in range(1, 5)] + ["gas_into_gas_mixed_blast_furnace"]),
    ):
        a = f.reindex(columns=cols)
        out[name] = a.sum(axis=1, min_count=len(cols) if name.startswith("blast") else 1)
        out[name + "_coverage"] = a.notna().mean(axis=1)
    out["blast_observed_parts"] = f[[f"gas_blast_furnace_{i}" for i in range(1, 6)]].sum(axis=1, min_count=1)
    out["coke_available_proxy"] = f.gas_coke_oven_1 - f.gas_into_gas_mixed_coke
    return out


def build_features(prepared):
    """整表运算仅含向后窗口；负荷标签、发电耗气、柜位永不进入特征。"""
    f, idx = prepared.frame, prepared.index
    cols = [x for x in f if x.startswith(("gas_", "gas_user_")) and not x.startswith("gas_holder_")]
    raw, series = f[cols], process_series(prepared)
    out = {"slot": idx.hour*4+idx.minute//15, "weekday": idx.dayofweek,
           "month": idx.month, "price": prices_at(prepared, idx)}
    positions = np.arange(len(idx))
    for col in raw:
        v = raw[col]
        last = np.maximum.accumulate(np.where(v.notna(), positions, -1))
        out[col] = v
        out[col+"_age"] = np.where(last >= 0, positions-last, np.nan)
    for col in ("blast_13", "blast_45", "blast_observed_parts", "known_demand", "coke_available_proxy"):
        v = series[col]
        out[col] = v
        for lag in (1, 4, 16, 96):
            out[f"{col}_lag{lag}"] = v.shift(lag)
        for window in (4, 16, 96):
            out[f"{col}_med{window}"] = v.rolling(window, min_periods=1).median()
            out[f"{col}_valid{window}"] = v.notna().rolling(window, min_periods=1).mean()
        out[col+"_std16"] = v.rolling(16, min_periods=2).std()
    for col in series:
        if col.endswith("coverage"):
            out[col] = series[col]
    return pd.DataFrame(out, index=idx).replace([np.inf, -np.inf], np.nan).astype("float32")


def horizon_block(prepared, base, positions, horizons):
    pos, h = np.asarray(positions, int), np.asarray(horizons, int)
    n, m = len(pos), len(h)
    a = np.tile(base.to_numpy()[pos], (m, 1))
    target = pd.DatetimeIndex(np.tile(prepared.index[pos].to_numpy(), m)) + pd.to_timedelta(np.repeat(h, n)*15, unit="min")
    extra = np.column_stack((np.repeat(h, n), target.hour*4+target.minute//15,
                             target.dayofweek, prices_at(prepared, target)))
    return np.column_stack((a, extra)).astype("float32"), list(base.columns) + ["horizon", "target_slot", "target_weekday", "target_price"]
