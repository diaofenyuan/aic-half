"""新版配置：赛事事实、项目选择与模型配方集中定义。"""
from pathlib import Path
import os

CODE_ROOT = Path(__file__).resolve().parent
ROOT = next((p for p in (CODE_ROOT, *CODE_ROOT.parents) if (p / "data").is_dir()), CODE_ROOT)
DATA_ROOT = Path(os.environ.get("AIC_DATA_ROOT", ROOT / "data"))
OUTPUT_ROOT = ROOT / "outputs"
EXPERIMENT_ROOT = ROOT / "tmp" / "tmp_outputs"
SCHEMA = "lgbm_core_v3"
LOAD_VERSION = "load_group_shape_v3"
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
               # 画像窗口 14 天：六窗（含 05-06 月爬升窗）实测，窗口越短水平越贴当前，
               # 14 天长期 g1 86.72 vs 28 天 84.33、42 天 78.77，10 天起开始变差。
               "profile_days": 14,
               # 画像时段环形平滑：96 个时段的中位数只有 14 个样本，形状噪声会直接进入
               # 长期 g1 与 factor 形状。3 点核 0.25/0.5/0.25 在六窗上对 long g1 全窗为正
               # （86.714→86.753、macro 87.777→87.806）；5 点核过平滑，短期回落。
               "profile_smooth": [.25, .5, .25],
               # 柜位平衡锚（用户 2026-10-29 明确授权放宽本条信息边界）：用起报点前已观测的
               # 发生量-热风炉-高炉用户-混合量与双柜净变化反推“当前发出去的气”，相对其
               # 28 天滚动中位做有界缩放，抑制整日状态级偏移。w1 时间留出（前段定参、
               # 后 11 天验证）：short macro 88.456→89.817、long 87.526→87.990。
               # 双柜缺测时系数回退为 1，不锚定、不插补；w2~w6 因 1 号柜缺失实际不触发。
               "holder_anchor": {"lo": .90, "hi": 1.10, "gamma": .8, "smooth": 16, "days": 28,
                                 "source": "user_authorized_2026-10-29_time_holdout_w1"},
               "tasks": {
                   # 短期按 1-4 / 5-8 分段并加入相对水平特征：六窗 short g1 84.99→85.42、
                   # gall 89.74→90.17，long 同步 87.58→87.71。相对特征把水平绑到近期气量
                   # 而不是历史画像，避免起报后水平漂移时整段错位；采用依据为本地六窗对照。
                   # 短期训练窗口 112→84 天（六窗 short macro 87.79→88.04，5/6 窗改善，
                   # w4 水平偏低探针 +0.6、w6 爬升窗 +1.0；91 天出现 w4 崩到 78.6 的敏感点，
                   # 故取 12 周整数窗口而不追 77 天峰值）。长期仍用 112 天，水平由短期对齐。
                   "short": {"short": "segmented", "feature_set": "memory+dynamics+missingness+relative",
                             "phase": "fixed", "days": 84,
                             "representation": "group_difference", "consistency_alpha": .3,
                             # 短期远端向同起报点长期预测收缩（权重按步递增 1/8..1）。
                             # 长期水平已完全取自短期（short_level_alpha=1.0），不再有
                             # “被长期更差的水平拖住”的耦合风险；权重从 0.8 加到 1.0
                             # 单调改善（六窗 g1 85.61→85.75→85.81→85.87），故取满。
                             "shrink_to_long": 1.0,
                             # 仅在总负荷预测高于训练截止前同槽画像时作单向下调。
                             "profile_upper_guard": {"ratio": 1.02, "scale": 0.97},
                             "decay_days": 28, "learning_rate": .03, "rounds": 640, "aux_weight_target":"small"},
                   # 长期全程遮挡下相对误差权重会系统性压低水平（四窗实测 w4 少 85MW），改用等权 L1。
                   "long": {"long": "segmented", "feature_set": "base+relative+mechanism", "phase": "rotating",
                            "weight_power": 0.0,
                            # 1-MAPE 对高估更敏感：长期 g1 冻结画像整体下调 5%，
                            # 四窗实测 85.11→85.95（w1/w2/w3 改善，w4 水平偏高下跌 3.4）。
                            "baseline_scale": [0.95, 1.0],
                            # 长期 generator_all 的水平完全取短期模型（形状仍用长期），
                            # 六窗 gall 89.60→90.23；短期水平本身已被画像锚定。
                            "short_level_alpha": 1.0,
                            "target_kinds": ["direct", "factor"], "tree_weights": [0., 1.]}}}
TREE = {"learning_rate": .05, "num_leaves": 31, "min_data_in_leaf": 80,
        "feature_fraction": .9, "lambda_l2": 2., "verbosity": -1,
        "num_threads": CPU_THREADS, "deterministic": True, "force_col_wise": True,
        "use_missing": True, "zero_as_missing": False}
POLICY = {"source": "adopted_project_policy", "publication": "07:00", "availability": "point",
          # 本地对照中统一 60MW 后，10 个 |Δgall|>60 步消除，计划收益未下降。
          # 保留当前项目配方；异常线上评分不能证明官方采用 60MW 阈值。
          "safe": (.15, .90), "plan": (.20, .85), "ramp": (30., 60.),
          "generate_ramp": (30., 60.), "running_ramp_per_min": .01,
          "min_load": .60, "solver_seconds": 10., "mip_gap": .01,
          "power_tolerance": .001, "gas_tolerance": 1., "stock_tolerance": 2.,
          "residual_days": 28, "vent_penalty_cny_m3": .10,
          "switch_penalty_cny": 30., "terminal_constraint": False}
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
