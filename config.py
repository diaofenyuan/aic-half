"""集中定义数据契约、模型配方、调度参数及其来源。"""
from pathlib import Path
import os

CODE_ROOT = Path(__file__).resolve().parent
ROOT = next((p for p in (CODE_ROOT, *CODE_ROOT.parents) if (p / "data").is_dir()), CODE_ROOT)
DATA_ROOT = Path(os.environ.get("AIC_DATA_ROOT", ROOT / "data"))
OUTPUT_ROOT = ROOT / "outputs"
EXPERIMENT_ROOT = ROOT / "tmp" / "tmp_outputs"
SCHEMA = "lgbm_core_v3"
LOAD_VERSION = "load_group_shape_v4"
CPU_THREADS = min(16, os.cpu_count() or 1)
STEP = 15
DT = .25
TARGETS = ("generator_1", "generator_all")
GASES = ("blast_furnace_gas", "coke_gas", "converter_gas")
USE = tuple("load_generator_use_" + g for g in GASES)
HOLDERS = ("gas_holder_blast_furnace_gas_holder_1", "gas_holder_blast_furnace_gas_holder_2")
CAPACITY = (300000., 200000.)
UNITS = (50., 50., 50., 50., 120., 120.)
TRAIN_END = "2025-10-10 23:45:00"
TEST_START = "2025-10-11 00:00:00"
TEST_END = "2025-10-31 23:45:00"
OPT_START = "2025-10-11 00:15:00"
OPT_END = "2025-11-01 23:45:00"
# 超出已训练96步的部分先使用合法历史画像，不能静默外推树模型。
OPT_TAIL = {"history_days": 28, "source": "causal_slot_median_tail_v1"}
FOLDS = (
    {"name": "A", "start": "2025-06-01", "end": "2025-06-21 23:45:00"},
    {"name": "B", "start": "2025-07-05", "end": "2025-07-25 23:45:00"},
    {"name": "C", "start": "2025-08-02", "end": "2025-08-22 23:45:00"},
)
STAGES = (
    ("first_train", "2025-01-01", "2025-04-30 23:59:00", "train"),
    ("first_test", "2025-05-01", "2025-05-02 23:59:00", "train"),
    ("semi_train", "2025-05-03", "2025-09-30 23:59:00", "train"),
    ("half_train", "2025-10-01", "2025-10-10 23:59:00", "train"),
    ("half_test", TEST_START, "2025-10-31 23:59:00", "test"),
)
TABLES = ("load", "gas", "gas_user", "gas_holder")
# 倍率由数据版本声明，禁止根据量级猜测并重复换算。
HOLDER_INPUT_VERSION = os.environ.get("AIC_HOLDER_VERSION", "historical_raw")
HOLDER_SCALES = {"historical_raw": (10000., 1.), "official_m3": (1., 1.)}
LOAD_RECIPE = {"short": "shared", "long": "shared", "loss": "relative_l1",
               "days": 112, "stride": 2, "rounds": 320, "leaves": 31, "seed": 2026,
               # 采用近期同日内时段画像，平衡生产水平变化与样本覆盖。
               "profile_days": 14,
               # 环形平滑降低同槽中位数的采样噪声，保持日内形状连续。
               "profile_smooth": [.25, .5, .25],
               # 水平后处理使用起报点已观测的过程收支与双柜变化，执行有界缩放。
               # 该估计不作为真实发电耗气；双柜观测无效时系数回退为 1，不插补库存。
               "holder_anchor": {"lo": .90, "hi": 1.10, "gamma": .8, "smooth": 16, "days": 28,
                                 "source": "user_authorized_2026-10-29_time_holdout_w1"},
               # 两条链路统一缩放全厂负荷，再由分组校准重构小机组组负荷。
               "gall_level_scale": .975,
               "tasks": {
                   # 短期逐提前量训练，使用近期过程水平、动态变化与缺失状态特征。
                   "short": {"short": "segmented", "feature_set": "memory+dynamics+missingness+relative",
                             "phase": "fixed", "days": 84, "stride": 1, "leaves": 63,
                             "representation": "group_difference", "consistency_alpha": .3,
                             # 短期远端向同起报点长期预测收缩（权重按步递增 1/8..1）。
                             "shrink_to_long": 1.0,
                             # 仅在总负荷预测高于训练截止前同槽画像时作单向下调。
                             "profile_upper_guard": {"ratio": 1.02, "scale": 0.97},
                             "decay_days": 28, "learning_rate": .015, "rounds": 1600, "aux_weight_target":"small"},
                   # 长期使用等权 L1，动态水平与冻结日内形状分离建模。
                   "long": {"long": "segmented", "feature_set": "base+relative+mechanism", "phase": "rotating", "stride": 1,
                            "weight_power": 0.0,
                            # 冻结画像仅形成中间基线，部署出口继续执行分组份额重构。
                            "baseline_scale": [0.95, 1.0],
                            # 长期全厂水平由短期均值有界校正，保留长期日内形状。
                            "short_level_alpha": 1.0,
                            "target_kinds": ["direct", "factor"], "tree_weights": [0., 1.]}}}
TREE = {"learning_rate": .05, "num_leaves": 31, "min_data_in_leaf": 80,
        "feature_fraction": .9, "lambda_l2": 2., "verbosity": -1,
        "num_threads": CPU_THREADS, "deterministic": True, "force_col_wise": True,
        "use_missing": True, "zero_as_missing": False}
# 分组份额校准在最终出口重构小机组组负荷，参数不按测试日期或时段细分。
SHARE_CALIBRATION = {
    "enabled": True,
    # 目标份额作用于水平校准后的全厂负荷，随后按大小机组容量限制取值。
    "target": {"short": .3231, "long": .3128},
    "lower_capacity": 240.,   # 2×120MW 大机组满载：g1 不得低于 gall-240
    "upper_capacity": 200.,   # 4×50MW 小机组容量：g1 不得超过 200
    # 经验校正将小机组组下界抬升固定功率；该数值不代表额外装机容量。
    "peak_slack_mw": 8.,
    # 目标份额超出 [0.15, 0.45] 时推理直接报错，避免配置误写把预测压到不合理区间。
    "source": "s2g_final_20261101_online_validated_74_6568",
}
POLICY = {"source": "adopted_project_policy", "publication": "07:00", "availability": "point",
          # 聚合硬上限遵守当前项目选定的 30/66MW；正常生成采用 30/60MW 裕度。
          # 历史线上触发数不能用于反推官方阈值。
          "safe": (.15, .90), "plan": (.20, .85), "ramp": (30., 66.),
          "generate_ramp": (30., 60.), "running_ramp_per_min": .01,
          "min_load": .60, "solver_seconds": 10., "mip_gap": .01,
          "power_tolerance": .001, "gas_tolerance": 1., "stock_tolerance": 2.,
          "residual_days": 28, "vent_penalty_cny_m3": .10,
          # 启停成本抑制由微小收益差异引起的机组状态频繁切换。
          "switch_penalty_cny": 60., "terminal_constraint": False}
GAS_POWER_ACCEPTANCE={"max_wape":.10,"max_abs_bias":.05,"source":"local_engineering_gate"}
GROUP_DISPATCH={"history_days":28,"holdout_days":7,"max_wape":.15,"max_abs_bias":.05,
                "mapping_penalty":10.,"smooth_penalty":10.,"source":"historical_inferred_soft_objective"}
OPTIONAL_USE_CALIBRATION={"version":"optional_prefix_level_v1","components":["budget_converter"],
    "history_steps":4,"decay_steps":16,"require_current_observation":True,
    "source":"inferred_short_term_consumption_level_calibration"}
DISPATCH_CONTROL={"version":"joint_optional_prefix_v4","history_days":28,"min_support_rows":400,
                  "sensitivity_days":[28,56],"reserve_hours":4,"reserve_quantile":.9,
                  "benefit_fraction":.999,"ratio_iterations":3,"mip_gap":.005,
                  "gas_support_role":"diagnostic_only",
                  "optional_use_calibration":OPTIONAL_USE_CALIBRATION,
                  "source":"historical_inferred_joint_control"}
PROCESS_RESOURCE={"version":"shared_process_q25_v2","history_days":112,"stride":32,
    "horizons":[1,4,8,16,24,48,72,96],"rounds":80,"leaves":15,"dropout_modulo":2,"quantile":.25,
    "min_scale_samples":100,"source":"inferred_shared_furnace_process_model"}
SOURCE_FILES = ("main.py", "main_s.py", "main_l.py", "main_opt.py", "main_nn.py",
                "config.py", "output.py", "data_cleaning/__init__.py",
                "data_cleaning/pipeline.py", "data_cleaning/quality.py",
                "requirements.txt", "README.md", "AGENTS.md")
