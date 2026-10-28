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
               "profile_days": 28,
               "tasks": {
                   "short": {"feature_set": "memory+dynamics+missingness", "phase": "fixed",
                             "representation": "group_difference", "consistency_alpha": .6,
                             "decay_days": 28, "learning_rate": .03, "rounds": 640, "aux_weight_target":"small"},
                   "long": {"long": "segmented", "feature_set": "base+relative+mechanism", "phase": "rotating",
                            "weight_power":1.45,
                            "target_kinds": ["factor", "factor"], "tree_weights": [1., 1.]}}}
TREE = {"learning_rate": .05, "num_leaves": 31, "min_data_in_leaf": 80,
        "feature_fraction": .9, "lambda_l2": 2., "verbosity": -1,
        "num_threads": CPU_THREADS, "deterministic": True, "force_col_wise": True,
        "use_missing": True, "zero_as_missing": False}
POLICY = {"source": "adopted_project_policy", "publication": "07:00", "availability": "point",
          "safe": (.15, .90), "plan": (.20, .85), "ramp": (30., 66.),
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
