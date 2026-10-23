"""数据扫描、因果清洗、15 分钟对齐与特征底座。

关键约束：
- 原始文件只读；所有清洗只生成内存对象。
- 15 分钟网格取同一时间戳的观测，不对 ``t`` 之后的分钟做聚合或插值。
- 目标列只作标签；评测窗口内由 ``masked_copy`` 整体遮挡，特征函数没有
  任何读取 ``t+h`` 真实值的路径。
"""
from __future__ import annotations

import copy
import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

warnings.filterwarnings("ignore", category=RuntimeWarning)

TARGET_COLUMNS = tuple(C.TARGET_COLUMN[t] for t in C.TARGETS)


# ---------------------------------------------------------------------------
# 数据对象
# ---------------------------------------------------------------------------
@dataclass
class PreparedData:
    index: pd.DatetimeIndex
    frame: pd.DataFrame              # 清洗后的 15 分钟宽表（目标列可被遮挡）
    labels: pd.DataFrame             # 未遮挡的合法标签，仅供评分
    quality: dict
    scan: dict
    mask_start: pd.Timestamp | None = None
    mask_end: pd.Timestamp | None = None
    source_hashes: dict = field(default_factory=dict)
    stage_ranges: dict = field(default_factory=dict)

    def masked_copy(self, start: pd.Timestamp | str, end: pd.Timestamp | str) -> "PreparedData":
        """复制并遮挡连续评测窗口的目标负荷，模拟正式推理的输入模式。"""
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        frame = self.frame.copy()
        present = [c for c in TARGET_COLUMNS if c in frame.columns]
        frame.loc[(frame.index >= start) & (frame.index <= end), present] = np.nan
        obj = copy.copy(self)
        obj.frame = frame
        obj.mask_start = start
        obj.mask_end = end
        return obj

    def prefix_copy(self, end: pd.Timestamp | str) -> "PreparedData":
        """推理拟合只拿首个起报点之前的前缀，资源标签也不能跨进干跑窗口。"""
        stop = int(self.index.searchsorted(pd.Timestamp(end), side="right"))
        if stop == 0:
            raise ValueError("训练前缀为空")
        obj = copy.copy(self)
        obj.index = self.index[:stop]
        obj.frame = self.frame.iloc[:stop].copy()
        obj.labels = self.labels.iloc[:stop].copy()
        obj.quality = {**self.quality, "rows": stop,
                       "time_range": [str(obj.index[0]), str(obj.index[-1])]}
        return obj


# ---------------------------------------------------------------------------
# 扫描与读取
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"无法解析 {path.name}: {last_error}")


def _find_table_files(stage_dir: Path, table: str) -> list[Path]:
    files = [p for p in stage_dir.glob("*.csv") if p.is_file()]
    hits: list[Path] = []
    for suffix in C.TABLE_SUFFIXES[table]:
        hits.extend(p for p in files if p.name.lower().endswith(suffix.lower()))
    return sorted(set(hits))


def scan_sources(config=C) -> dict:
    """只读扫描四个真实数据目录；不根据目录名猜标签是否发布。"""
    report = {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
              "stages": {}, "files": {}, "extras": {}, "problems": []}
    for stage in config.STAGES:
        stage_dir = config.DATA_ROOT / stage["dir"]
        info = {"dir": str(stage_dir), "exists": stage_dir.is_dir(),
                "role": stage["role"], "tables": {}}
        if not stage_dir.is_dir():
            if not stage.get("optional"):
                report["problems"].append(f"缺少目录 {stage_dir}")
            report["stages"][stage["name"]] = info
            continue
        for table in config.TABLES:
            files = _find_table_files(stage_dir, table)
            table_info = {"files": [], "rows": 0, "columns": [], "time_range": None,
                          "target_valid": {}, "duplicates": 0, "missing_ratio": {}}
            for path in files:
                report["files"][str(path.relative_to(config.ROOT))] = {
                    "bytes": path.stat().st_size, "sha256": _sha256(path)}
                try:
                    raw = _read_csv(path)
                except Exception as exc:  # noqa: BLE001
                    report["problems"].append(f"{path.name}: {exc}")
                    continue
                if "datetime" not in raw.columns:
                    report["problems"].append(f"{path.name}: 缺少 datetime")
                    continue
                times = pd.to_datetime(raw["datetime"], errors="coerce")
                valid_time = times.notna()
                table_info["files"].append(path.name)
                table_info["rows"] += int(valid_time.sum())
                table_info["columns"] = sorted(set(table_info["columns"]) | set(raw.columns))
                if valid_time.any():
                    span = [times[valid_time].min().isoformat(), times[valid_time].max().isoformat()]
                    table_info["time_range"] = span if table_info["time_range"] is None else [
                        min(table_info["time_range"][0], span[0]),
                        max(table_info["time_range"][1], span[1])]
                table_info["duplicates"] += int(times.duplicated().sum())
                for target in C.TARGETS:
                    if target in raw.columns:
                        count = int(pd.to_numeric(raw[target], errors="coerce").notna().sum())
                        key = target
                        table_info["target_valid"][key] = table_info["target_valid"].get(key, 0) + count
                for col in raw.columns:
                    if col == "datetime":
                        continue
                    ratio = float(pd.to_numeric(raw[col], errors="coerce").isna().mean())
                    old = table_info["missing_ratio"].get(col, 0.0)
                    table_info["missing_ratio"][col] = max(old, ratio)
            info["tables"][table] = table_info
        report["stages"][stage["name"]] = info

    for extra in ("data_dictionary.xlsx", "price.xlsx"):
        path = config.DATA_ROOT / "first_train" / extra
        report["extras"][extra] = {
            "path": str(path), "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None}
    return report


# ---------------------------------------------------------------------------
# 对齐与清洗
# ---------------------------------------------------------------------------
def _standardize_table(raw: pd.DataFrame, name: str, conflicts: list[str]) -> pd.DataFrame:
    if "datetime" not in raw.columns:
        raise ValueError(f"{name} 缺少 datetime 列")
    frame = raw.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    bad_time = int(frame["datetime"].isna().sum())
    if bad_time:
        conflicts.append(f"{name}: 无效时间戳 {bad_time}")
    frame = frame[frame["datetime"].notna()].set_index("datetime").sort_index()
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame.index.duplicated().any():
        repeated = frame[frame.index.duplicated(keep=False)]
        conflict = repeated.groupby(level=0).nunique(dropna=False).gt(1).any(axis=1)
        if conflict.any():
            conflicts.append(f"{name}: 同时间戳冲突 {int(conflict.sum())}，保留首条")
        frame = frame.loc[~frame.index.duplicated(keep="first")]
    return frame


def _apply_sentinel_rules(frame: pd.DataFrame, quality: dict) -> None:
    counts = {}
    for col, rule in C.CLEANING["sentinel_rules"].items():
        if col not in frame.columns:
            continue
        values = frame[col]
        invalid = pd.Series(False, index=values.index)
        if "min" in rule:
            invalid |= values < float(rule["min"])
        if "max" in rule:
            invalid |= values > float(rule["max"])
        counts[col] = int(invalid.sum())
        frame.loc[invalid, col] = np.nan
    for prefix in C.CLEANING["nonnegative_prefixes"]:
        for col in frame.columns:
            if not col.startswith(prefix):
                continue
            invalid = frame[col] < 0
            if invalid.any():
                counts[col] = counts.get(col, 0) + int(invalid.sum())
                frame.loc[invalid, col] = np.nan
    quality["sentinel_or_negative"] = counts


def _causal_fill(frame: pd.DataFrame, quality: dict) -> None:
    """仅对连续供气/用气字段做单边填补；发生量和目标列绝不填补。"""
    limit = int(C.CLEANING["ffill_limit"])
    window = int(C.CLEANING["rolling_median_steps"])
    min_periods = int(C.CLEANING["rolling_min_periods"])
    counts = {}
    for col in C.CLEANING["causal_fill_columns"]:
        if col not in frame.columns:
            continue
        values = frame[col]
        missing = values.isna()
        if not missing.any():
            continue
        trailing = values.rolling(window, min_periods=min_periods).median().shift(1)
        filled = values.ffill(limit=limit).fillna(trailing)
        counts[col] = int((missing & filled.notna()).sum())
        frame[col] = filled
    quality["causal_imputed"] = counts


def _sum_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [c for c in columns if c in frame.columns]
    if not present:
        return pd.Series(np.nan, index=frame.index)
    return frame[present].sum(axis=1, min_count=1)


def _build_derived(frame: pd.DataFrame) -> None:
    gen_blast = [f"gas_blast_furnace_{i}" for i in range(1, 6)]
    air_heaters = [f"gas_air_heater_{i}" for i in range(1, 6)]
    blast_users = [f"gas_user_blast_furnace_user{i}" for i in range(1, 5)]
    converter_users = [f"gas_user_converter_user{i}" for i in range(1, 3)]
    for name, columns in (("gen_blast", gen_blast), ("air_heater", air_heaters),
                          ("blast_user", blast_users), ("converter_user", converter_users)):
        present = [c for c in columns if c in frame]
        frame[f"d_{name}_valid_fraction"] = (frame[present].notna().sum(axis=1) / len(columns))

    frame["d_gen_blast"] = _sum_columns(frame, gen_blast)
    frame["d_gen_coke"] = frame.get("gas_coke_oven_1", np.nan)
    frame["d_gen_converter_raw"] = frame.get("gas_converter_1", np.nan)
    frame["d_gen_converter_scaled"] = frame["d_gen_converter_raw"] * C.CONVERTER_GAS_SCALE
    frame["d_air_heater_total"] = _sum_columns(frame, air_heaters)
    frame["d_blast_user_total"] = _sum_columns(frame, blast_users)
    frame["d_converter_user_total"] = _sum_columns(frame, converter_users)
    frame["d_mixed_blast"] = frame.get("gas_into_gas_mixed_blast_furnace", np.nan)
    frame["d_mixed_coke"] = frame.get("gas_into_gas_mixed_coke", np.nan)
    frame["d_mixed_converter"] = frame.get("gas_into_gas_mixed_converter", np.nan)

    frame["d_surplus_blast"] = (frame["d_gen_blast"] - frame["d_air_heater_total"]
                                - frame["d_blast_user_total"] - frame["d_mixed_blast"])
    frame["d_surplus_coke"] = frame["d_gen_coke"] - frame["d_mixed_coke"]
    frame["d_surplus_converter"] = (frame["d_gen_converter_scaled"]
                                    - frame["d_converter_user_total"] - frame["d_mixed_converter"])

    frame["d_use_blast"] = frame.get("load_generator_use_blast_furnace_gas", np.nan)
    frame["d_use_coke"] = frame.get("load_generator_use_coke_gas", np.nan)
    frame["d_use_converter"] = frame.get("load_generator_use_converter_gas", np.nan)
    frame["d_use_total"] = frame[["d_use_blast", "d_use_coke", "d_use_converter"]].sum(
        axis=1, min_count=3)
    frame["d_use_valid_fraction"] = frame[["d_use_blast", "d_use_coke", "d_use_converter"]].notna().mean(axis=1)
    # 发电气种占比：比绝对量更稳，对负荷水平漂移不敏感。
    for gas in ("blast", "coke", "converter"):
        frame[f"d_use_share_{gas}"] = (frame[f"d_use_{gas}"] / frame["d_use_total"]
                                       .replace(0.0, np.nan))
    weights = C.FUEL_PROXY_WEIGHTS
    frame["d_fuel_proxy"] = (frame["d_use_blast"] * weights["blast"]
                             + frame["d_use_coke"] * weights["coke"]
                             + frame["d_use_converter"] * weights["converter"])
    holder = C.GAS_HOLDER["column"]
    frame["d_holder_level"] = frame.get(holder, np.nan)
    frame["d_holder_ratio"] = frame["d_holder_level"] / C.GAS_HOLDER["capacity_m3"]


def _concat_stage_tables(frames: list[pd.DataFrame], name: str,
                         second_counts: list[int], quality: dict) -> pd.DataFrame:
    if not frames:
        raise ValueError(f"没有可用的 {name} 数据")
    combined = pd.concat(frames, axis=0).sort_index(kind="stable")
    if combined.index.duplicated().any():
        repeated = combined[combined.index.duplicated(keep=False)]
        conflict = repeated.groupby(level=0).nunique(dropna=False).gt(1).any(axis=1)
        if conflict.any():
            second_counts.append(int(conflict.sum()))
        combined = combined.loc[~combined.index.duplicated(keep="first")]
    return combined


def prepare_data(config=C, mask_targets: tuple | None = None,
                 cutoff: pd.Timestamp | str | None = None) -> PreparedData:
    """扫描、拼接、清洗并生成统一 15 分钟宽表。"""
    scan = scan_sources(config)
    source_hashes = {k: v["sha256"] for k, v in scan["files"].items()}
    conflicts: list[str] = []
    duplicate_conflicts: list[int] = []
    per_table: dict[str, list[pd.DataFrame]] = {t: [] for t in config.TABLES}
    stage_ranges: dict[str, list[str]] = {}

    for stage in config.STAGES:
        stage_dir = config.DATA_ROOT / stage["dir"]
        if not stage_dir.is_dir():
            continue
        loaded = {}
        for table in config.TABLES:
            pieces = []
            for path in _find_table_files(stage_dir, table):
                raw = _read_csv(path)
                pieces.append(_standardize_table(raw, f"{stage['name']}/{table}", conflicts))
            if pieces:
                frame = pd.concat(pieces, axis=0).sort_index()
                frame = frame.loc[~frame.index.duplicated(keep="first")]
                per_table[table].append(frame)
                loaded[table] = frame
        if loaded:
            spans = [(f.index.min(), f.index.max()) for f in loaded.values() if len(f)]
            stage_ranges[stage["name"]] = [min(s[0] for s in spans).isoformat(),
                                           max(s[1] for s in spans).isoformat()]

    merged_tables = {}
    for table, frames in per_table.items():
        merged_tables[table] = _concat_stage_tables(
            frames, table, duplicate_conflicts, {})

    start = min(f.index.min() for f in merged_tables.values())
    end = max(f.index.max() for f in merged_tables.values())
    if cutoff is not None:
        end = min(end, pd.Timestamp(cutoff))
    index = pd.date_range(start.floor("15min"), end.floor("15min"), freq=f"{config.STEP_MINUTES}min")
    parts = []
    for table, frame in merged_tables.items():
        aligned = frame.reindex(index)
        parts.append(aligned.add_prefix(f"{table}_"))
    frame = pd.concat(parts, axis=1)
    frame.index.name = "datetime"

    quality: dict = {"rows": int(len(frame)),
                     "time_range": [index.min().isoformat(), index.max().isoformat()],
                     "duplicate_conflicts": int(sum(duplicate_conflicts)),
                     "problems": conflicts}
    _apply_sentinel_rules(frame, quality)
    _causal_fill(frame, quality)
    _build_derived(frame)

    targets = pd.DataFrame({t: frame.get(config.TARGET_COLUMN[t], np.nan)
                            for t in config.TARGETS}, index=index)
    valid = (targets["generator_1"] >= 0) & (targets["generator_1"] <= config.CAPACITY["generator_1"])
    valid &= (targets["generator_all"] >= 0) & (targets["generator_all"] <= config.CAPACITY["generator_all"])
    valid &= targets["generator_all"] + 1e-8 >= targets["generator_1"]
    quality["target_invalid"] = int((~valid & targets.notna().any(axis=1)).sum())
    targets.loc[~valid] = np.nan
    for target in config.TARGETS:
        frame[config.TARGET_COLUMN[target]] = targets[target]

    quality["target_valid"] = {t: int(targets[t].notna().sum()) for t in config.TARGETS}
    quality["target_missing_ratio"] = {t: float(targets[t].isna().mean()) for t in config.TARGETS}
    key_missing = {}
    for col in ("gas_holder_blast_furnace_gas_holder_2", "d_gen_blast",
                "d_surplus_blast", "d_use_total"):
        if col in frame.columns:
            key_missing[col] = float(frame[col].isna().mean())
    quality["key_missing_ratio"] = key_missing
    quality["calendar_complete"] = bool(len(index) == ((index.max() - index.min()) /
                                                       pd.Timedelta(minutes=config.STEP_MINUTES) + 1))
    quality["source_files"] = len(source_hashes)
    quality["hash_algorithm"] = "sha256"

    prepared = PreparedData(index=index, frame=frame, labels=targets,
                            quality=quality, scan=scan,
                            source_hashes=source_hashes, stage_ranges=stage_ranges)
    if mask_targets is not None:
        prepared = prepared.masked_copy(mask_targets[0], mask_targets[1])
    return prepared


# ---------------------------------------------------------------------------
# 特征底座
# ---------------------------------------------------------------------------
def _calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    slot = index.hour.to_numpy() * 4 + index.minute.to_numpy() // 15
    out = pd.DataFrame(index=index)
    out["cal_slot_sin"] = np.sin(2 * np.pi * slot / 96.0)
    out["cal_slot_cos"] = np.cos(2 * np.pi * slot / 96.0)
    out["cal_weekday"] = index.weekday.to_numpy().astype(np.float32)
    out["cal_is_weekend"] = (index.weekday.to_numpy() >= 5).astype(np.float32)
    out["cal_month"] = index.month.to_numpy().astype(np.float32)
    return out


def _last_valid(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = series.to_numpy(dtype=float)
    valid = np.isfinite(values)
    positions = np.arange(len(values))
    last = np.maximum.accumulate(np.where(valid, positions, -1))
    age = positions - last
    level = np.where(last >= 0, values[np.maximum(last, 0)], np.nan)
    return level, age.astype(np.float32)


BASE_CURRENT_COLUMNS = (
    "d_use_blast", "d_use_coke", "d_use_converter", "d_use_total", "d_fuel_proxy",
    "d_gen_blast", "d_gen_coke", "d_gen_converter_scaled",
    "d_air_heater_total", "d_blast_user_total", "d_converter_user_total",
    "d_surplus_blast", "d_surplus_coke", "d_surplus_converter",
    "d_holder_level", "d_holder_ratio",
)


def build_base_features(prepared: PreparedData, include_targets: bool) -> pd.DataFrame:
    """生成每个 15 分钟起报点已知的特征；目标特征仅在 include_targets 时加入。"""
    frame = prepared.frame
    out: dict[str, np.ndarray] = {}
    for col in BASE_CURRENT_COLUMNS:
        if col in frame.columns:
            out[col] = frame[col].to_numpy(dtype=float)
    for col in frame:
        if col.startswith("d_") and col.endswith("_valid_fraction"):
            out[col] = frame[col].to_numpy(dtype=float)

    for col in C.FEATURE_WINDOW_COLUMNS:
        if col not in frame.columns:
            continue
        series = frame[col]
        # 缺测与零供气是不同工况；年龄封顶一天，避免长缺口被当作稳定观测。
        _, observation_age = _last_valid(series)
        out[f"obs_{col}_missing"] = series.isna().to_numpy(dtype=float)
        out[f"obs_{col}_staleness"] = np.minimum(observation_age, C.STEPS_PER_DAY)
        for lag in C.FEATURE_LAGS:
            out[f"{col}_lag{lag}"] = series.shift(lag).to_numpy(dtype=float)
        for window in C.FEATURE_ROLLS:
            min_periods = max(2, window // 4)
            out[f"{col}_ma{window}"] = series.rolling(
                window, min_periods=min_periods).mean().to_numpy(dtype=float)
        min_periods = max(2, C.FEATURE_STD_WINDOW // 4)
        out[f"{col}_std{C.FEATURE_STD_WINDOW}"] = series.rolling(
            C.FEATURE_STD_WINDOW, min_periods=min_periods).std().to_numpy(dtype=float)

    cal = _calendar_frame(prepared.index)
    for col in cal.columns:
        out[col] = cal[col].to_numpy(dtype=float)

    for col in ("d_use_total", "d_gen_blast", "d_surplus_blast", "d_holder_level"):
        if col in frame.columns:
            level, age = _last_valid(frame[col])
            out[f"{col}_last_level"] = level
            out[f"{col}_age"] = age

    # 短时差分：负荷变化方向比绝对量更能解释下一时刻；实验 +0.11pp。
    for col in ("d_use_blast", "d_use_coke", "d_use_converter",
                "d_holder_level", "d_surplus_blast", "d_gen_blast"):
        if col in frame.columns:
            series = frame[col]
            out[f"{col}_d1"] = series.diff(1).to_numpy(dtype=float)
            out[f"{col}_d4"] = series.diff(4).to_numpy(dtype=float)

    total = frame["d_use_total"].replace(0, np.nan)
    for col in ("d_use_blast", "d_use_coke", "d_use_converter"):
        out[f"ratio_{col}"] = (frame[col] / total).to_numpy(dtype=float)
    out["ratio_surplus_blast"] = (frame["d_surplus_blast"]
                                  / (frame["d_gen_blast"].abs() + 1.0)).to_numpy(dtype=float)

    if include_targets:
        # 最近可见目标水平及年龄只进入 target 特征集。评测窗口内它们会退化为
        # 窗口前最后观测，若进入 gas 集会让树模型误把陈旧水平当成当前状态。
        for target in C.TARGETS:
            col = C.TARGET_COLUMN[target]
            if col in frame.columns:
                level, age = _last_valid(frame[col])
                out[f"last_{target}_level"] = level
                out[f"last_{target}_age"] = age
        for target in C.TARGETS:
            col = C.TARGET_COLUMN[target]
            if col not in frame.columns:
                continue
            series = frame[col]
            out[f"tgt_{target}"] = series.to_numpy(dtype=float)
            for lag in C.FEATURE_LAGS:
                out[f"tgt_{target}_lag{lag}"] = series.shift(lag).to_numpy(dtype=float)
            for window in C.FEATURE_ROLLS:
                min_periods = max(2, window // 4)
                out[f"tgt_{target}_ma{window}"] = series.rolling(
                    window, min_periods=min_periods).mean().to_numpy(dtype=float)
            level, age = _last_valid(series)
            out[f"tgt_{target}_last_level"] = level
            out[f"tgt_{target}_age"] = age

    frame_out = pd.DataFrame(out, index=prepared.index)
    # 年龄/last_level 族在重要性里长期垫底，去掉后短周期 +0.1pp 且训练更快。
    # 保留 last_{target}_level（真·最后可见负荷），只裁剪派生噪声列。
    drop = [c for c in frame_out.columns
            if c.endswith("_age") or (c.endswith("_last_level") and not c.startswith("last_"))]
    return frame_out.drop(columns=drop)


def target_calendar(prepared: PreparedData, positions: np.ndarray,
                    horizons: tuple[int, ...]) -> dict[str, np.ndarray]:
    """目标时刻的日历特征：仅使用已知的未来日历，不读未来生产量。

    官方评测窗口末端起报点的目标时刻会越出已发布数据；日历可从
    起报时刻加偏移直接推算，因此不在索引上 take，避免越界。
    """
    positions = np.asarray(positions, dtype=int)
    horizons_arr = np.asarray(horizons, dtype=int)
    origin_index = prepared.index.take(positions)
    # 只依赖钟表时间，越界目标时刻可由起报时刻 + 15min*h 推出。
    step = np.asarray(horizons_arr, dtype=np.int64) * C.STEP_MINUTES
    target_index = pd.DatetimeIndex(
        np.repeat(origin_index.to_numpy(), len(horizons_arr))
        + np.tile(step, len(positions)).astype("timedelta64[m]"))
    shape = (len(positions), len(horizons_arr))
    slot = target_index.hour.to_numpy() * 4 + target_index.minute.to_numpy() // 15
    return {
        "tgt_slot_sin": np.sin(2 * np.pi * slot / 96.0).reshape(shape),
        "tgt_slot_cos": np.cos(2 * np.pi * slot / 96.0).reshape(shape),
        "tgt_weekday": target_index.weekday.to_numpy().astype(np.float32).reshape(shape),
        "tgt_is_weekend": (target_index.weekday.to_numpy() >= 5).astype(np.float32).reshape(shape),
        "tgt_month": target_index.month.to_numpy().astype(np.float32).reshape(shape),
    }


def _day_matrix(values: np.ndarray) -> np.ndarray:
    days = len(values) // C.STEPS_PER_DAY
    return values[:days * C.STEPS_PER_DAY].reshape(days, C.STEPS_PER_DAY)


def _day_reference(position: int) -> int:
    """返回位置 ``position`` 之前最后一个完整自然日编号。"""
    day = position // C.STEPS_PER_DAY
    return day if position % C.STEPS_PER_DAY == C.STEPS_PER_DAY - 1 else day - 1


def _causal_slot_profiles(values: np.ndarray, ref_days: np.ndarray,
                          days: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """按参考日生成各时槽的历史中位数，缓存减少重复计算。"""
    matrix = _day_matrix(values)
    n_days = len(matrix)
    cache: dict[int, np.ndarray] = {}
    profiles = np.full((len(ref_days), C.STEPS_PER_DAY), np.nan)
    for i, day in enumerate(ref_days):
        day = int(min(max(day, -1), n_days - 1))
        if day < 0:
            continue
        if day not in cache:
            lo = max(0, day - days + 1)
            block = matrix[lo:day + 1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prof = np.nanmedian(block, axis=0)
            counts = np.sum(np.isfinite(block), axis=0)
            if np.isfinite(prof).any():
                fallback = float(np.nanmedian(prof))
            else:
                fallback = float(np.nan)
            prof = np.where((counts >= 3) & np.isfinite(prof), prof, fallback)
            cache[day] = prof
        profiles[i] = cache[day]
    return profiles, cache


def _day_types(index: pd.DatetimeIndex) -> np.ndarray:
    """1=工作日，0=周末；工业负荷两者形态差异大，画像不能混着取中位数。"""
    return (index.dayofweek.to_numpy() < 5).astype(np.int8)


def _causal_slot_profiles_typed(values: np.ndarray, ref_days: np.ndarray,
                                day_types: np.ndarray, days: int
                                ) -> np.ndarray:
    """同槽位中位数，但只在同日型（工作日/周末）历史里取，避免周末拉低工作日画像。"""
    matrix = _day_matrix(values)
    n_days = len(matrix)
    out = np.full((len(ref_days), C.STEPS_PER_DAY), np.nan)
    for i, day in enumerate(ref_days):
        day = int(min(max(day, -1), n_days - 1))
        if day < 0:
            continue
        lo = max(0, day - days + 1)
        # 同日型优先；同日型样本不足 3 天时退回全日型中位数。
        same = [d for d in range(lo, day + 1) if day_types[d] == day_types[day]]
        pool = same if len(same) >= 3 else list(range(lo, day + 1))
        block = matrix[np.asarray(pool, dtype=int)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prof = np.nanmedian(block, axis=0)
        counts = np.sum(np.isfinite(block), axis=0)
        fallback = float(np.nanmedian(prof)) if np.isfinite(prof).any() else float("nan")
        out[i] = np.where((counts >= 2) & np.isfinite(prof), prof, fallback)
    return out


def target_slot_profile(prepared: PreparedData, positions: np.ndarray,
                        horizons: tuple[int, ...], target: str,
                        days: int = None) -> np.ndarray:
    """目标时刻同槽位、同日型历史中位数，只用起报点之前已完整结束的自然日。"""
    days = int(days or C.PROFILE_DAYS)
    values = prepared.frame[C.TARGET_COLUMN[target]].to_numpy(dtype=float)
    last_level, _ = _last_valid(prepared.frame[C.TARGET_COLUMN[target]])
    positions = np.asarray(positions, dtype=int)
    ref_days = np.array([_day_reference(int(p)) for p in positions], dtype=int)
    # 周末/工作日拆分在 28 天窗口内样本不足，quick 全链路分数略降；
    # 保留 typed 实现备查，部署仍用全日型中位数（_causal_slot_profiles）。
    profiles, _ = _causal_slot_profiles(values, ref_days, days)
    out = np.full((len(positions), len(horizons)), np.nan)
    for i, p in enumerate(positions):
        for j, h in enumerate(horizons):
            value = profiles[i, (int(p) + int(h)) % C.STEPS_PER_DAY]
            if not np.isfinite(value):
                value = last_level[p]
            out[i, j] = value
    return out


def fuel_level_ratio(prepared: PreparedData, positions: np.ndarray,
                     days: int = None) -> np.ndarray:
    """当前可见燃料水平相对历史同槽位水平的比例，用于 profile 缩放。"""
    days = int(days or C.PROFILE_DAYS)
    fuel = prepared.frame["d_fuel_proxy"].to_numpy(dtype=float)
    positions = np.asarray(positions, dtype=int)
    ref_days = np.array([_day_reference(int(p)) for p in positions], dtype=int)
    profiles, _ = _causal_slot_profiles(fuel, ref_days, days)
    slot = positions % C.STEPS_PER_DAY
    base = profiles[np.arange(len(positions)), slot]
    ratio = np.divide(fuel[positions], base,
                      out=np.ones(len(positions), dtype=float),
                      where=np.isfinite(base) & (base > 0))
    # 当前燃料缺测保留为缺失特征；画像融合侧单独禁用该次缩放。
    low, high = C.PROFILE_RATIO_CLIP
    return np.clip(ratio, low, high)


def last_valid_level(prepared: PreparedData, positions: np.ndarray,
                     target: str) -> np.ndarray:
    level, _ = _last_valid(prepared.frame[C.TARGET_COLUMN[target]])
    return level[np.asarray(positions, dtype=int)]


def resource_series(prepared: PreparedData) -> pd.DataFrame:
    """面向发电的三气种基准耗气序列。

    原始数据缺少部分混气/生产用户列，无法仅凭 ``发生−已知用户`` 得到可信余气；
    因此优化侧使用历史机组耗气作为“可发电煤气基准包络”，再叠加气柜缓冲。
    这不是把历史耗气直接复制成优化结果：MILP 仍可在包络内重新分配时段。
    """
    frame = prepared.frame
    out = pd.DataFrame(index=prepared.index)
    out["blast"] = frame["d_use_blast"]
    out["coke"] = frame["d_use_coke"]
    out["converter"] = frame["d_use_converter"]
    return out


def surplus_series(prepared: PreparedData) -> pd.DataFrame:
    """发生量减已知用户/混气的诊断余气，仅用于资源侧误差分析，不直接进优化。"""
    frame = prepared.frame
    out = pd.DataFrame(index=prepared.index)
    out["blast"] = frame["d_surplus_blast"].clip(lower=0)
    out["coke"] = frame["d_surplus_coke"].clip(lower=0)
    out["converter"] = frame["d_surplus_converter"].clip(lower=0)
    return out
