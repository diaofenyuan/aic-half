"""半决赛唯一配置源。

设计原则：
1. 日期、阶段、字段角色、可见性与物理常量只在此处定义，子模块不硬编码。
2. 动态生产字段一律按 ``available_time <= origin`` 使用；本工程采用
   “15 分钟网格上同一时间戳的瞬时观测”，不做向后聚合，避免目标时刻污染特征。
3. 尚无法从官方数据字典确认的优化参数保持 ``None`` 或明确标记为
   ``inferred``，由 ``main_opt.py`` 返回阻断原因，而不是编造可提交结果。
"""
import os
import sys
from pathlib import Path

from tqdm import tqdm


PROGRESS_ENABLED: bool | None = None


def progress(iterable=None, *, desc: str, **kwargs):
    """统一控制终端进度；非交互输出默认静默，避免污染重定向日志。"""
    options = {"dynamic_ncols": True, "mininterval": .5, "leave": False, "file": sys.stdout,
               "disable": None if PROGRESS_ENABLED is None else not PROGRESS_ENABLED,
               "unit": "项"}
    options.update(kwargs)
    return tqdm(iterable, desc=desc, **options)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs"
# 实验/调参/临时模型只允许放 tmp；不得写入 models/ 或正式 outputs 归档。
TMP_ROOT = ROOT / "tmp"
EXPERIMENT_OUTPUT_ROOT = TMP_ROOT / "tmp_outputs"

# ---------------------------------------------------------------------------
# 阶段与文件
# 半决赛训练标签覆盖 2025-10-01~10-10，评测区间为 2025-10-11~10-31。
# 阶段按发布顺序排列；重叠时间戳以较新发布的数据为准。
# ---------------------------------------------------------------------------
STEPS_PER_DAY = 96
STEP_MINUTES = 15

STAGES = (
    {"name": "first_train", "dir": "first_train", "role": "train",
     "labels": "present", "period": ("2025-01-01", "2025-04-30")},
    {"name": "first_test", "dir": "first_test", "role": "train",
     "labels": "present", "period": ("2025-05-01", "2025-05-02")},
    {"name": "semi_train", "dir": "semi_train", "role": "train",
     "labels": "present", "period": ("2025-05-03", "2025-09-30")},
    {"name": "semi_test", "dir": "semi_test", "role": "inference_features",
     "labels": "masked", "period": ("2025-10-01", "2025-10-10")},
    {"name": "half_train", "dir": "half_train", "role": "train",
     "labels": "present", "period": ("2025-10-01", "2025-10-10"),
     "optional": True},
    # 半决赛测试输入已接入，仅用于冻结模型推理，不参与训练、选参或标定。
    {"name": "half_test", "dir": "half_test", "role": "official_inference",
     "labels": "forbidden", "period": ("2025-10-11", "2025-10-31"),
     "optional": True},
)

# 表名 -> 文件名后缀。先匹配长后缀，避免 gas_user 被 gas 规则误吞。
TABLE_SUFFIXES = {
    "gas_holder": ("gas_holder.csv",),
    "gas_user": ("gas_user.csv",),
    "gas": ("gas.csv",),
    "load": ("load.csv", "load_real.csv"),
}
TABLES = ("load", "gas", "gas_user", "gas_holder")

TARGETS = ("generator_1", "generator_all")
CAPACITY = {"generator_1": 200.0, "generator_all": 440.0}
STRUCTURAL_CAPACITY = {"generator_big": 240.0}  # all - g1 对应 2×120MW
HORIZONS_S = tuple(range(1, 9))
HORIZONS_L = tuple(range(1, 97))
PROBE_HORIZONS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96)

TRAIN_START = "2025-01-01 00:00:00"
TRAIN_END = "2025-10-10 23:45:00"
# 官方测试范围仅用于推理起报网格与输出覆盖校验，不提供本地评分标签。
OFFICIAL_EVAL_START = "2025-10-11 00:00:00"
OFFICIAL_EVAL_END = "2025-10-31 23:45:00"

# ---------------------------------------------------------------------------
# 时序折：前三个用于选模，D 只确认一次。
# 每折 21 天连续起报点，训练标签的最大目标时刻必须落在 train_end 之前。
# ---------------------------------------------------------------------------
FOLDS = (
    {"name": "A", "role": "tuning", "start": "2025-06-01 00:00:00", "days": 21,
     "train_end": "2025-05-31 23:45:00"},
    {"name": "B", "role": "tuning", "start": "2025-07-05 00:00:00", "days": 21,
     "train_end": "2025-07-04 23:45:00"},
    {"name": "C", "role": "tuning", "start": "2025-08-02 00:00:00", "days": 21,
     "train_end": "2025-08-01 23:45:00"},
    {"name": "D", "role": "confirm", "start": "2025-09-05 00:00:00", "days": 21,
     "train_end": "2025-09-04 23:45:00"},
)

# ---------------------------------------------------------------------------
# 字段角色与可见性
# ---------------------------------------------------------------------------
TARGET_COLUMN = {t: f"load_{t}" for t in TARGETS}
OBSERVED_AT_T = (
    "load_generator_use_blast_furnace_gas",
    "load_generator_use_coke_gas",
    "load_generator_use_converter_gas",
    "gas_blast_furnace_1", "gas_blast_furnace_2", "gas_blast_furnace_3",
    "gas_blast_furnace_4", "gas_blast_furnace_5",
    "gas_coke_oven_1", "gas_converter_1",
    "gas_air_heater_1", "gas_air_heater_2", "gas_air_heater_3",
    "gas_air_heater_4", "gas_air_heater_5",
    "gas_into_gas_mixed_blast_furnace", "gas_into_gas_mixed_coke",
    "gas_into_gas_mixed_converter",
    "gas_user_blast_furnace_user1", "gas_user_blast_furnace_user2",
    "gas_user_blast_furnace_user3", "gas_user_blast_furnace_user4",
    "gas_user_converter_user1", "gas_user_converter_user2",
    "gas_holder_blast_furnace_gas_holder_1",
    "gas_holder_blast_furnace_gas_holder_2",
)
# 目标列只允许作标签或起报点之前的历史；本工程默认在评测窗口整体遮挡。
KNOWN_IN_ADVANCE = ("calendar", "price")
FORBIDDEN_FUTURE = ("target_at_t_plus_h", "future_gas_truth")

# ---------------------------------------------------------------------------
# 清洗与物理约束
# ---------------------------------------------------------------------------
CLEANING = {
    # 传感器哨兵/不可识别量：只标记为 NaN，不猜真实值。
    "sentinel_rules": {
        "gas_air_heater_4": {"min": 0.0},
        "gas_user_converter_user1": {"min": -100.0},
        "gas_holder_blast_furnace_gas_holder_1": {"min": 0.0, "max": 400000.0},
    },
    # 煤气发生/消耗为负值时按无效观测处理；派生平衡量允许为负。
    "nonnegative_prefixes": ("gas_blast_furnace_", "gas_coke_oven_", "gas_converter_",
                             "gas_air_heater_", "gas_into_gas_mixed_",
                             "gas_user_blast_furnace_user", "gas_user_converter_user"),
    # 这些列由模拟量求和/供暖连续性要求，允许单边因果填补；发生量不做长缺口填补。
    "causal_fill_columns": (
        "gas_air_heater_1", "gas_air_heater_2", "gas_air_heater_3",
        "gas_air_heater_4", "gas_air_heater_5",
        "gas_user_blast_furnace_user1", "gas_user_blast_furnace_user2",
        "gas_user_blast_furnace_user3", "gas_user_blast_furnace_user4",
        "gas_user_converter_user1", "gas_user_converter_user2",
    ),
    "ffill_limit": 8,
    "rolling_median_steps": 96,
    "rolling_min_periods": 24,
}
# 数据字典明确：高炉煤气柜容量为 20 万和 30 万 m³。1 号柜有效样本不足且出现
# 1e30 哨兵，不能作为库存状态；优化只使用 2 号柜并保守限制在 15%~90%。
GAS_HOLDER = {
    "column": "gas_holder_blast_furnace_gas_holder_2",
    "capacity_m3": 200000.0,
    "safe_low_ratio": 0.15,
    "safe_high_ratio": 0.90,
    "note": "1号柜数据不可用；焦炉/转炉无独立柜位字段，优化中不虚构库存。",
}
# 数据字典未给热值；仅用三维线性回归从历史发电机出力标定，属于 inferred。
HEAT_RATE = {"method": "nnls_like_positive_regression", "min_r2": 0.60,
             "source": "load.csv 历史 generator_all 与三气种发电耗气"}

# 燃料水平画像的近似能量权重（LHV 相对比例），只用于 profile_fuel 的缩放，
# 不作为优化热值；目的是避免把 1 m³ 焦炉气与 1 m³ 高炉气当同质。
FUEL_PROXY_WEIGHTS = {"blast": 1.0, "coke": 5.0, "converter": 2.4}

# ---------------------------------------------------------------------------
# 特征与模型
# ---------------------------------------------------------------------------
# 需要做滞后/滚动统计的关键列；其余列只取当前值，控制训练内存。
FEATURE_WINDOW_COLUMNS = (
    "d_use_total", "d_use_blast", "d_use_coke", "d_use_converter",
    "d_surplus_blast", "d_surplus_coke", "d_surplus_converter",
    "d_holder_level", "d_gen_blast", "d_gen_coke", "d_gen_converter_scaled",
    "d_fuel_proxy",
    "d_use_share_blast", "d_use_share_coke", "d_use_share_converter",
)
FEATURE_LAGS = (1, 4, 16, 96)
FEATURE_ROLLS = (4, 16, 96)
FEATURE_STD_WINDOW = 96
# 新特征只供显式选择的 at_t 分支使用，保证旧模型重载时的特征口径不变。
AT_T_DEVICE_COLUMNS = (
    *(f"gas_blast_furnace_{i}" for i in range(1, 6)),
    *(f"gas_air_heater_{i}" for i in range(1, 6)),
    *(f"gas_user_blast_furnace_user{i}" for i in range(1, 5)),
    *(f"gas_user_converter_user{i}" for i in range(1, 3)),
    "gas_into_gas_mixed_blast_furnace", "gas_into_gas_mixed_coke",
    "gas_into_gas_mixed_converter",
)
AT_T_MINUTE_GROUPS = {
    "gen_blast": tuple(f"gas_blast_furnace_{i}" for i in range(1, 6)),
    "gen_coke": ("gas_coke_oven_1",),
    "gen_converter": ("gas_converter_1",),
    "air_heater": tuple(f"gas_air_heater_{i}" for i in range(1, 6)),
    "blast_user": tuple(f"gas_user_blast_furnace_user{i}" for i in range(1, 5)),
    "converter_user": tuple(f"gas_user_converter_user{i}" for i in range(1, 3)),
    "mixed_blast": ("gas_into_gas_mixed_blast_furnace",),
    "mixed_coke": ("gas_into_gas_mixed_coke",),
    "mixed_converter": ("gas_into_gas_mixed_converter",),
    "holder": ("gas_holder_blast_furnace_gas_holder_2",),
}
PROFILE_DAYS = 28
PROFILE_RATIO_CLIP = (0.5, 2.0)

DECAY_DAYS = 60.0          # 样本时间衰减半衰期；None 表示不加权
# 评分是 1-MAPE，训练按 1/|y| 加权对齐指标；power=1 即逆真值加权。
SAMPLE_WEIGHT_POWER = 1.0

LGBM_BASE = {
    "objective": "regression_l1",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "verbosity": -1,
    "num_threads": min(16, os.cpu_count() or 1),
    "seed": 20261021,
}
LGBM_TREES_SHORT = 450
LGBM_TREES_LONG = 450
LGBM_TREES_QUICK = 180
TRAIN_STRIDE_SHORT = 1
TRAIN_STRIDE_LONG = 2
QUICK_TRAIN_STRIDE = 8
NN_CONFIG = {"look_back": 192, "epochs": 80, "quick_epochs": 8,
             "patience": 12, "quick_patience": 3, "early_stop_days": 7,
             "batch_size": 512, "quick_batch_size": 256,
             # 16GB GPU 引擎：按批 gather 滑窗、bf16 训练、fp32 推理。
             "precision": "bf16", "predict_batch": 1024, "train_stride": 1,
             "d_model": 64, "n_blocks": 5, "dropout": 0.15,
             "lr": 1e-3, "weight_decay": 1e-4, "scheduler": "none",
             # A/B/C 开发验证后保留容量归一化因果 TCN；是否部署仍由逐目标融合门槛决定。
             "kind_long": "tcn_causal", "kind_short": "tcn_causal",
             "head_mode": "scaled", "load_history": False,
             "feature_set": "engineered", "training_mask_days": 0, "far_weight": 1.0,
             "profile_conditions": True,
             "temporal_norm": "layer", "future_calendar": "cyclic",
             "max_gpu_memory_gib": 15.0,
             # 三种子聚合降低训练随机性；实际聚合算子随模型保存。
             "profile_base": "plain", "mape_floor_ratio": 0.02, "n_seeds": 3,
             }
# 完整 A/B/C 缺测情景通过后冻结；配方随源码交付，不依赖 tmp 中的实验文件。
AVAILABILITY_FALLBACK = {
    "short": {"kind": "direct", "features": "gas_no_power", "fallback_weight": 1.,
              "blend": {"generator_1": {"head": .9, "tail": .6},
                        "generator_all": {"head": .4, "tail": .4}},
              "stage": "R2", "source_study": "20261025_182342_plan_availability"},
    "long": {"kind": "global", "features": "gas_no_power", "fallback_weight": 1.,
             "fallback_weights": {
                 "generator_1": {"short": .1, "medium": .1, "long": .1},
                 "generator_all": {"short": 1., "medium": 1., "long": 1.}},
             "blend": {"generator_1": {"short": .9, "medium": .9, "long": .9},
                       "generator_all": {"short": 0., "medium": 0., "long": 0.}},
             "reference_recipe": {"kind": "global", "features": "gas",
                 "blend": {"generator_1": {"short": 1., "medium": 1., "long": 1.},
                           "generator_all": {"short": .2, "medium": .1, "long": 0.}}},
             "stage": "R2_target_mix", "source_study": "20261025_182342_plan_availability"},
}
# A/B/C 上以旧保留策略为对照通过保护门槛；分钟分支仅占 5%，不强行替换稳定分量。
AVAILABILITY_FALLBACK["long"] = {
    "kind": "global", "features": "gas_no_power_at_t", "fallback_weight": 1.,
    "blend": {"generator_1": {"short": .8, "medium": .7, "long": .6},
              "generator_all": {"short": .2, "medium": .1, "long": .1}},
    "retained_mix_weight": .05, "retained_recipe": AVAILABILITY_FALLBACK["long"],
    "stage": "R2_retained_mix", "source_study": "20261025_192733_plan_observed",
}
LONG_GROUPS = (("short", 1, 8), ("medium", 9, 32), ("long", 33, 96))
SHORT_GROUPS = (("head", 1, 4), ("tail", 5, 8))

# 选模门槛（百分点）。仅作为预防单折噪声的工程阈值，不冒充官方标准。
SELECTION = {
    "min_mean_gain_pp": 0.20,
    "max_worst_fold_drop_pp": 0.20,
    "confirm_drop_tolerance_pp": 1.5,
    "blend_grid": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
}

# 用户于2026-10-25放宽专项平均增益要求；折间保护与总负荷约束保持独立。
GENERATOR1_SPECIALIST = {'short': {'kind': 'refined_nn',
           'base_recipe': {'kind': 'nn',
                           'model_stage': 'G6_specialist_mixer_3seeds',
                           'weight': 1.0,
                           'missing_only': True,
                           'nn_kind': 'tsmixer',
                           'config': {'feature_set': 'generator1',
                                      'head_mode': 'scaled',
                                      'profile_base': 'plain',
                                      'profile_conditions': True,
                                      'profile_conditions_scaled': True,
                                      'training_mask_days': 21,
                                      'load_history': False,
                                      'n_seeds': 1,
                                      'target_indices': [0],
                                      'ensemble_aggregation': 'validation_weighted',
                                      'aggregation_power': 2.0},
                           'seeds': 3,
                           'source_study': '20261025_202459_generator1',
                           'candidate': 'G6_specialist_mixer_3seeds_missing_w1',
                           'development': {'normal': {'mean_accuracy': 90.06247010582729,
                                                      'mean_gain_pp': 0.0,
                                                      'worst_fold': 86.1063217540028,
                                                      'baseline_worst_fold': 86.1063217540028},
                                           'all_missing': {'mean_accuracy': 88.31532951414306,
                                                           'mean_gain_pp': 1.55304119650323,
                                                           'worst_fold': 83.65952309883055,
                                                           'baseline_worst_fold': 78.94438501759404},
                                           'outage_day7': {'mean_accuracy': 89.19598004405042,
                                                           'mean_gain_pp': 1.822611610281508,
                                                           'worst_fold': 85.04576173500845,
                                                           'baseline_worst_fold': 80.20936716197376}}},
           'refinement': {'nn_kind': 'tsmixer',
                          'config': {'feature_set': 'generator1_observed',
                                     'head_mode': 'scaled',
                                     'profile_base': 'plain',
                                     'profile_conditions': True,
                                     'profile_conditions_scaled': True,
                                     'training_mask_days': 21,
                                     'load_history': False,
                                     'target_indices': [0],
                                     'n_seeds': 1,
                                     'power_mask_training': False},
                          'seeds': 3,
                          'model_stage': 'H_normal_3seeds',
                          'weight': 0.25,
                          'mode': 'missing'},
           'previous_candidate': 'combined_H_normal_3seeds_missing_w0.25',
           'previous_source_study': '20261025_214751_generator1_refine',
           'source_study': '20261025_225217_deep_study',
           'candidate': 'duration_state_missing_0.375',
           'development': {'normal': {'mean_accuracy': 90.06247010582729,
                                      'mean_gain_pp': 0.0,
                                      'worst_fold': 86.1063217540028,
                                      'baseline_worst_fold': 86.1063217540028},
                           'all_missing': {'mean_accuracy': 90.03896059236563,
                                           'mean_gain_pp': 0.24705992006307062,
                                           'worst_fold': 87.72524812149841,
                                           'baseline_worst_fold': 86.84938891815538},
                           'outage_day7': {'mean_accuracy': 90.09723089192511,
                                           'mean_gain_pp': 0.16369961125159213,
                                           'worst_fold': 87.30284673809983,
                                           'baseline_worst_fold': 86.84060349028677}},
           'comparison': 'previous_promoted_generator1_strategy',
           'calibration': {'family': 'duration_state',
                           'shrink': 0.375,
                           'mode': 'missing',
                           'final_offsets_days': [42, 21],
                           'final_window_days': 20}},
 'long': {'kind': 'availability_expert',
          'weight': 1.0,
          'missing_only': True,
          'source_stage': 'R2',
          'disagreement_margin_mw': 20.0,
          'seeds': 3,
          'seed_reference': 'G2_disagreement_20mw',
          'expert_kind': 'global',
          'expert_features': 'gas_no_power_at_t',
          'expert_blend': {'generator_1': {'short': 0.8, 'medium': 0.7, 'long': 0.6},
                           'generator_all': {'short': 0.2, 'medium': 0.1, 'long': 0.1}},
          'source_study': '20261025_202459_generator1',
          'candidate': 'G2_disagreement_20mw_3seeds',
          'development': {'normal': {'mean_accuracy': 87.35255057282806,
                                     'mean_gain_pp': 0.0,
                                     'worst_fold': 83.08015714818647,
                                     'baseline_worst_fold': 83.08015714818647},
                          'all_missing': {'mean_accuracy': 85.69861099137687,
                                          'mean_gain_pp': 13.526537770450874,
                                          'worst_fold': 81.33915109377075,
                                          'baseline_worst_fold': 51.156890373904396},
                          'outage_day7': {'mean_accuracy': 86.30850995503857,
                                          'mean_gain_pp': 8.64070111460716,
                                          'worst_fold': 81.83947748060507,
                                          'baseline_worst_fold': 64.9909496605128}}}}

# ---------------------------------------------------------------------------
# 资源预测与发电优化
# ---------------------------------------------------------------------------
RESOURCE_NAMES = ("blast", "coke", "converter")
# converter_1 的量纲与其它流量不一致；用历史“用户+混气+发电耗气 / converter_1”
# 的中位数反推比例。该值明确标记为 inferred，正式数据字典确认后必须复核。
CONVERTER_GAS_SCALE = 7043.5
OPT_CONFIG = {
    "unit_capacity_mw": (50.0, 50.0, 50.0, 50.0, 120.0, 120.0),
    "unit_groups": (("generator_1", 4), ("generator_all", 6)),
    "min_load_ratio": 0.60,
    "ramp_fraction_per_min": 0.10,
    "availability_safety": 0.80,   # 预测余气打折，抵消未建模用户与计量误差
    "availability_mode": "point",
    "gas_safety": {"blast": 1.0, "coke": 1.0, "converter": 1.0},
    # 计划柜位相对安全区间再留 5% 裕度；回测仍按官方 15%~90% 检查。
    "holder_margin_ratio": 0.05,
    # 日末回到初始柜位，避免计划靠净放空/净充满气柜获得虚假收益。
    "terminal_holder": True,
    "execution_gas_mode": "clip",  # 替代配比须经开发折验证后晋级
    # 仅用于代理溢出的敏感性评价，不能冒充官方放散罚金。
    "evaluation_release_penalties": (0.0, 0.5, 1.0),
    "gas_cap_quantile": 0.995,
    "gas_cap_headroom": 1.10,
    "solver_time_limit_s": 15,
    "rolling_replan_steps": 4,
    "rolling_solver_time_limit_s": 1.0,
    "mip_rel_gap": 0.01,
    "power_tolerance_mw": 1e-4,
    "gas_tolerance_m3h": 1.0,
    "holder_tolerance_m3": 1.0,
    "vent_tolerance_m3": 1.0,      # 兼容旧参数；当前用放散罚金替代字典序
    # 官方未给放散罚金。基线 0.5 元/m³ 使高炉气在峰段发电收益仍高于放散，
    # 同时允许谷段把气存入气柜；正式提交前必须按官方口径复核。
    # P1（Zhao 2015）：按气种×分时差异化——焦炉气热值高、放散代价大，
    # 尖峰时段外购电替代价值高，放散的机会成本也更高。
    "vent_penalty_cny_per_m3": 0.50,
    "vent_penalty_table": {
        #              谷     平     峰     尖
        "blast":     {"valley": 0.5, "flat": 0.8, "peak": 1.0, "sharp": 1.2},
        "coke":      {"valley": 1.0, "flat": 1.6, "peak": 2.0, "sharp": 2.4},
        "converter": {"valley": 0.7, "flat": 1.1, "peak": 1.4, "sharp": 1.7},
    },
    # 气柜 GPF：让计划柜位尽量贴在偏好带中部，偏离带外按距离线性加罚，
    # 避免 MILP 把柜位顶到安全边界上换取短期收益（Zhao 2015 思想）。
    "holder_pref_low_ratio": 0.35,
    "holder_pref_high_ratio": 0.65,
    "holder_penalty_cny_per_m3": 0.02,
    # P4：启停切换惩罚，压机组频繁开关（元/次·台）。
    "unit_switch_penalty_cny": 150.0,
    "max_daily_milp": 31,
    # 已核对本地赛题 PDF 第十三节（三）的五个 opt_ 字段；物理可行性另行核验。
    "template_confirmed": True,
    "price_file": "first_train/price.xlsx",
    "price_unit": "CNY_per_kWh",
}

# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
TEAM_CODE = "AIC-2026-68001720"
TEAM_NAME = "咕咕嘎嘎"
TEAM_LEADER = "朱泓宇"
# 包名按队伍约定统一；模拟/正式状态由运行摘要和 submittable 字段表达。
MOCK_ZIP_NAME = f"{TEAM_NAME}-半决赛.zip"
FORMAL_ZIP_NAME = MOCK_ZIP_NAME
REPORT_JSON = "运行结果.json"
REPORT_LOG = "运行日志.log"
SOURCE_DIR = "源码"
MODEL_DIR = "模型"
REPORT_DIR = "报告"
