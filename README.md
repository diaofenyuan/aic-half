# 煤气发电量预测与发电优化

本项目面向 AIC 半决赛煤气发电量预测与发电优化任务，使用赛事提供的生产、煤气发生量、用户耗气、气柜及电价数据，实现短期与长期负荷预测，并生成每日固定发布的发电调度计划。学习模型采用 LightGBM，调度采用混合整数线性规划（MILP），在 CPU 环境下运行。

| 任务 | 预测或决策对象 | 时间分辨率 | 输出跨度 |
|---|---|---|---|
| 短期预测 | 小机组组 `generator_1`、全厂 `generator_all` | 15 分钟 | 8 步，即未来 2 小时 |
| 长期预测 | 同上 | 15 分钟 | 96 步，即未来 24 小时 |
| 发电优化 | 三类煤气耗气量、小机组组及全厂负荷 | 15 分钟 | 每日 07:00 发布，常规计划为 96 步，首尾补齐提交窗口 |

负荷单位为 MW，煤气流量为 m³/h，气柜库存为 m³。`generator_1` 表示四台 50 MW 机组的合计负荷，`generator_all` 表示全部六台机组的合计负荷。

## 1. 环境准备

已核对的本地环境为 Windows 64 位、Python **3.10.11**。依赖精确版本见 [requirements.txt](requirements.txt)，包括 NumPy、pandas、SciPy、scikit-learn、LightGBM、openpyxl 和 threadpoolctl。无需 GPU、PyTorch 或外部模型服务；MILP 通过 SciPy 调用 HiGHS，无需单独安装商业求解器。

获取完整源码后，在项目根目录使用 PowerShell 执行：

```powershell
python --version
python -m venv .venv
$Python = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $Python -m pip install -r requirements.txt
```

`python --version` 应为 Python 3.10 系列。后续示例沿用同一终端中的 `$Python` 变量，通过虚拟环境解释器运行，无需激活脚本。其他操作系统需相应调整解释器路径，本项目未提供跨平台复现保证。

下载受限时，可将安装命令替换为：

```powershell
& $Python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

离线部署时，先在具有相同操作系统、体系结构与 Python 版本的联网环境中准备依赖，再将源码、依赖目录及获授权的赛事数据转移至目标环境：

```powershell
# 联网环境：下载依赖及其传递依赖。
& $Python -m pip download -r requirements.txt --dest tmp/wheelhouse

# 离线环境：在已创建的虚拟环境中安装。
& $Python -m pip install --no-index --find-links tmp/wheelhouse -r requirements.txt
```

## 2. 数据准备

赛事数据不随代码分发，也不包含在提交 ZIP 中。复现者须自行取得使用权限，并保留官方表头与原始文件内容。默认数据根目录为项目下的 `data/`。

| 阶段目录 | 负荷表 | 其余三类表 |
|---|---|---|
| `first_train/` | `Pre_load.csv` | `Pre_gas.csv`、`Pre_gas_user.csv`、`Pre_gas_holder.csv` |
| `first_test/` | `Pre_test_load.csv` | `Pre_test_gas.csv`、`Pre_test_gas_user.csv`、`Pre_test_gas_holder.csv` |
| `semi_train/` | `Semi_load.csv` | `Semi_gas.csv`、`Semi_gas_user.csv`、`Semi_gas_holder.csv` |
| `half_train/` | `final_load.csv` | `final_gas.csv`、`final_gas_user.csv`、`final_gas_holder.csv` |
| `half_test/` | `final_load_real.csv` | `final_test_gas.csv`、`final_test_gas_user.csv`、`final_test_gas_holder.csv` |

以上目录均位于数据根目录下。`first_train/` 还需包含全年电价文件 `price.xlsx`；同目录的 `data_dictionary.xlsx` 为字段及单位参考材料。程序按 `config.py` 的阶段定义读取数据，当前加载器不读取 `semi_test/`。

加载器按文件名后缀识别四类 CSV，每阶段每类必须恰有一份匹配文件。不要在同一阶段目录放置以 `load.csv`、`load_real.csv`、`gas.csv`、`gas_user.csv` 或 `gas_holder.csv` 结尾的备份文件。CSV 必须包含 `datetime` 列及对应官方字段，支持 UTF-8（含 BOM）和 GBK 编码。

运行前显式设置数据位置与柜位输入版本：

```powershell
$env:AIC_DATA_ROOT = (Resolve-Path .\data).Path
$env:AIC_HOLDER_VERSION = "historical_raw"
```

外部数据目录可直接将 `AIC_DATA_ROOT` 设置为其绝对路径，该目录下仍须包含上述阶段子目录。

| 柜位输入版本 | 柜 1 换算倍率 | 柜 2 换算倍率 | 适用数据 |
|---|---|---|---|
| `historical_raw`（默认） | 10000 | 1 | 本项目采用的历史原始柜位版本 |
| `official_m3` | 1 | 1 | 已统一为 m³ 的官方版本 |

输入已经统一为 m³ 时，应改为 `$env:AIC_HOLDER_VERSION = "official_m3"`。程序按版本仅换算一次，不自动推断单位；错误版本会改变库存轨迹和约束结论。

## 3. 复现流程

### 3.1 检查环境与数据

```powershell
& $Python main.py --help
& $Python main.py doctor
& $Python main.py self-check --threads 16 --run-id reproduce_check_01
```

`doctor` 检查训练数据与电价，输出数据来源、有效值统计、单位版本及基础特征数量。`self-check` 使用历史数据视图及合成用例检查因果特征、评分、预测容量关系和调度约束，并将检查记录写入实验目录。`infer` 自动执行输入覆盖与模型保存重载检查。

`--threads` 的有效范围为 1 至 `min(16, 本机逻辑处理器数)`；默认取该上限，并统一限制 LightGBM、求解器和数值库的线程预算。处理器数量不足时，请降低示例中的数值。

### 3.2 小预算验证

先验证数据读取、训练、推理、调度和归档链路：

```powershell
& $Python main.py infer --quick --experiment --threads 16 --run-id reproduce_quick_01
```

`--quick` 缩减训练样本、轮数与推理范围，用于检查运行链路。结果写入 `tmp/tmp_outputs/reproduce_quick_01/`，源码快照写入 `tmp_main/reproduce_quick_01/源码/`；性能评估与完整导出应使用完整预算运行。

### 3.3 从原始数据完整训练与导出

完成检查后，按当前固定配方重新训练短期、长期和资源模型，生成完整预测及每日调度：

```powershell
& $Python main.py --threads 16 --run-id reproduce_full_01
```

省略子命令等价于 `infer`。数据阶段、训练截止、预测范围与调度范围统一由 `config.py` 定义；首端预案使用独立的较早资源模型。默认流程按固定配方训练，不自动执行历史 A/B/C 回测或重新搜索参数。

该命令将最终运行归档至 `outputs/reproduce_full_01/`。需要完整预算实验但不进行正式归档时，添加 `--experiment`，结果转入 `tmp/tmp_outputs/`。每次运行必须使用新的 `--run-id`，标识仅允许字母、数字、下划线和连字符，已有目录不会被覆盖。完整训练耗时取决于硬件和输入数据，实际计时写入运行记录。

### 3.4 独立校验结果

```powershell
& $Python main.py verify --run-dir outputs/reproduce_full_01
```

重点核对 `报告/运行结果.json` 和 `verify` 输出中的以下字段：

| 字段 | 含义 |
|---|---|
| `upload_ready` | 运行生成的三张结果表通过覆盖检查；需同时查看 `zip.passed` 确认打包校验 |
| `full_window` | 预测及优化均覆盖完整正式时间网格 |
| `submittable` | 非实验运行、完整窗口、文件检查及本地严格约束全部满足 |
| `archive_passed` | `verify` 确认三表、ZIP、源码快照及调度证据一致 |
| `constraints_passed` | `verify` 独立重算逐日与跨日约束的结果 |
| `submittable_claim_consistent` | 报告中的可提交结论与重新核验结果一致 |

`infer` / `opt` 正常完成返回 0，完成导出但存在调度约束失败时返回 2，执行异常返回 1。`verify` 在归档完整且可提交声明一致时返回 0，否则返回 2；因此 **`verify` 返回 0 不等于约束通过**，仍须查看 `constraints_passed` 与 `submittable`。

求解无解时，程序仍尽可能保留三表与 ZIP，记录恢复或兜底分支，并将本地可提交状态标记为 false。不得通过人工改写 CSV 消除失败。归档完整、本地可提交、官方接受及现场安全认证是不同结论。

### 3.5 复用归档模型

已有兼容的完整模型时，可以重新推理或仅重新生成调度。以下为隔离复现示例：

```powershell
# 复用短期、长期和资源模型，重新生成预测与调度。
& $Python main.py infer --reuse-run outputs/reproduce_full_01 --experiment --threads 16 --run-id reproduce_infer_01

# 保留原短期、长期预测表，仅重新生成调度。
& $Python main.py opt --reuse-run outputs/reproduce_full_01 --experiment --threads 16 --run-id reproduce_opt_01
```

常规模型不会因复用而重新训练或标定。若旧归档缺少首端资源模型，`infer` 会补训该模型；`opt` 默认拒绝，只有显式添加 `--train-bootstrap` 才会准备首端模型。因此，严格的无训练复现应使用包含首端模型的完整兼容归档。正式运行不能复用 quick 或历史开发模型。

加载器检查模型版本、柜位单位版本、特征顺序、必要标定参数和文件完整性，并对 `main_s.py`、`data_cleaning/pipeline.py` 检查源码摘要。复现既有归档时，应使用该运行 `源码/` 中的完整源码及对应依赖和数据版本；不要混用当前源码与历史模型。即使只修改相关文件的注释，也可能触发源码摘要不兼容。

从原始数据重新训练与加载既有模型是两种复现方式。模型保存重载在采样起报点按最大绝对差 `1e-6` 检查；调度求解采用时间限制与相对最优间隙，同版本环境下仍应以实际约束和数值核验为准，不承诺不同硬件上的 MILP 指令逐字节相同。

## 4. 输出格式与归档

| 结果文件 | 内容 | 列数 | `datetime` 含义 |
|---|---|---|---|
| `s_result.csv` | 两个目标各 8 步预测 | 17 | 预测起报时刻 |
| `l_result.csv` | 两个目标各 96 步预测 | 193 | 预测起报时刻 |
| `opt_result.csv` | 三气种耗气量及两组负荷 | 6 | 调度指令生效时刻 |

三表均为 UTF-8 CSV，时间格式为 `YYYY-MM-DD HH:MM:SS`，数值保留六位小数。预测列按目标依次排列，命名为 `generator_1_t+15_pred`、`generator_all_t+15_pred` 等；其中偏移量以分钟计，短期至 `t+120`，长期至 `t+1440`。

优化表按顺序包含 `datetime`、`opt_generator_use_blast_furnace_gas`、`opt_generator_use_coke_gas`、`opt_generator_use_converter_gas`、`opt_generator_1` 和 `opt_generator_all`。

```text
outputs/<run_id>/
├── s_result.csv
├── l_result.csv
├── opt_result.csv
├── 咕咕嘎嘎-半决赛.zip
├── 源码/
├── 模型/
└── 报告/
    ├── 运行结果.json
    └── 运行日志.log
```

源码在运行开始时冻结，运行中修改源码会导致归档核验失败。`模型/model_meta.json` 及关联文件保存实际训练模型、特征顺序、冻结画像、校准参数、训练预算和运行依赖版本；JSON 与日志记录数据来源、源码摘要、逐日计划、约束、重载误差及耗时。

提交 ZIP 的根目录仅包含三张结果 CSV，不包含数据、源码、模型或报告；打包器检查成员白名单、CRC 以及与外部 CSV 的字节一致性。重载和独立校验需要完整本地归档，单独的提交 ZIP 不具备模型复现所需信息。

实验结果遵循同一三表命名，模型与运行记录位于 `tmp/tmp_outputs/<experiment_id>/`，源码快照单独位于 `tmp_main/<experiment_id>/源码/`。历史 `run` 回测则按开发折分别存放模型和结果，不合并为正式提交 ZIP。

## 5. 算法概述

### 5.1 数据处理与特征构造

数据按 15 分钟同时间戳采样，不使用后续分钟聚合、双向插值或向后填充。无效数值转为缺失，合法零负荷保留，标签不插补。过程特征包括煤气发生量、已观测需求、滞后、向后滚动统计、观测年龄、有效覆盖率及相对历史水平；长期模型增加分组收支代理和目标日历特征。未来电价作为官方预先公布信息使用。

训练与推理按时间顺序分离，各起报点仅使用当时已知的观测。负荷树和资源树不直接使用发电耗气或柜位作为输入特征；部署后处理使用已观测的双柜变化构造有界水平系数，双柜缺失时回退为 1。

### 5.2 短期与长期负荷预测

短期模型为每个提前量独立训练 LightGBM，分别预测全厂负荷与大机组组负荷，再由差值得到小机组组负荷。训练采用相对误差加权的 L1 损失，结合近期样本衰减与稀少负荷状态加权；早停仅使用训练前缀内部的时间留出，确定轮数后重新拟合合法训练样本。

长期模型按 **1—8、9—32、33—96 步**分别训练，使用“动态水平 × 冻结日内形状”表示全厂负荷。日内画像来自训练截止前 14 天的同时间段中位数，经环形平滑后保存；训练因子标签使用各训练块之前的画像，推理时保持画像固定。小机组组的中间基线采用冻结画像，最终输出经分组份额校准重构。

两条链路独立训练与保存，部署时通过统一出口协调：长期全厂水平由同起报点短期均值有界校正，短期远端按提前量递增权重向长期形状收缩，随后进行画像上沿校正、柜位水平缩放、全厂水平校准和分组份额重构，并满足机组容量关系。

| 当前固定参数 | 配置值 |
|---|---|
| 短期训练窗口与起报采样 | 84 天，15 分钟 |
| 短期树参数 | 63 叶，学习率 0.015，最多 1600 轮 |
| 长期训练窗口与起报采样 | 112 天，15 分钟 |
| 长期树参数 | 31 叶，学习率 0.05，最多 320 轮 |
| 最终全厂水平系数 | 0.975 |
| 小机组组目标份额 | 短期 0.3231，长期 0.3128；容量下界附加 8 MW 校正 |

以上为当前源码的部署配置，具体取值以 [config.py](config.py) 和模型归档为准。最终 `generator_1` 由校准后的全厂负荷按固定目标份额重构，并受容量上下界限制；独立树输出、中间画像与最终提交预测应分别理解。复现过程直接使用固定参数，不重新搜索或逐条修改输出。

### 5.3 资源预测与每日调度

资源链路独立预测两组高炉发生量、已观测生产需求及历史发电耗气包络。未观测用户、放散和计量差异作为整体残差，根据发布前 28 天、96 个日内时段的中位数估计，涉及下一柜位的记录仅在区间闭合后可用。缺失高炉发生量在具有相关过程观测和训练支持时，使用共享 LightGBM 的 0.25 分位模型估计；该分位值不表示经过校准的概率保证。

调度联合决定三类耗气量、六台机组启停与出力、两个独立高炉气柜的库存轨迹。气电关系由合法历史前缀的非负回归标定；分组映射作为软目标，模型总功率与耗气通过经验总量映射相连。目标综合计划发电收益、启停、平滑及库存风险成本，并在保留参考计划 99.9% 毛收益的条件下迭代改善电价加权负荷比。该比值不等同于官方线上评分。

主要项目约束和建模口径如下：

- 机组容量为 4×50 MW + 2×120 MW，开机区间为额定容量的 60%—100%；小机组组和总负荷分别核验 30、66 MW/15min 聚合爬坡，正常生成采用 30、60 MW 裕度，持续开机单机按 1%/min 检查。
- 两柜容量分别为 30 万、20 万 m³，规划区间为 20%—85%，本地安全区间为 15%—90%；高炉 1—3、4—5 分别对应两组管网，发电抽气按预测进气比例分摊，不跨柜调气。
- 每日 07:00 以同刻有效实测双柜重锚，单独记录首条指令前桥接和复位差额，不设置跨日硬期末库存。瞬时流量在 15 分钟区间内按持平假设积分，即流量乘以 0.25 小时。
- 计算库存超过 90% 的富余按本地自动放散模型记账，焦炉气和转炉气不设置虚构库存；经验资源、用户分摊和放散均保留推测来源。
- 首端使用独立预案，末端按输出范围延长计划；超过 96 步的资源使用发布前的历史分时画像估计。

**当前实现与规则的差异：** `main_opt.py` 在常规发布中额外限制焦炉气、转炉气占三气种总体积流量的比例分别不超过 3% 和 11%，首端预案不启用。该限制与 [AGENTS.md](AGENTS.md) 中不设置固定煤气配比的约定不一致，也不是官方已公布阈值。该差异作为当前实现的限制保留，复现结果应按实际约束解释。

独立校验器按原始轨迹重新检查容量、爬坡、逐柜初末态、物料守恒、发布时间及跨日衔接。求解器返回可行解不能替代独立核验，本地代理计划也不等于官方柜位重建或真实气网安全认证。

## 6. 历史评估与工程索引

需要评估历史预测和调度时，单独执行回测；该步骤不是正式训练的前置条件：

```powershell
& $Python main.py run --quick --threads 16 --run-id reproduce_backtest_quick_01
& $Python main.py run --threads 16 --run-id reproduce_backtest_full_01
```

完整 `run` 使用 A/B/C 三个连续 21 天历史窗口，`run --quick` 使用 A/B 并降低训练预算，结果始终写入实验目录。负荷主指标为 `100 × (1 − MAPE)`，零负荷样本单独报告绝对误差；资源按 WAPE、偏差和累计量误差评价，零气量样本上的误报计入分子。

| 文件或目录 | 职责 |
|---|---|
| `main.py` | 统一命令入口、训练编排、历史回测、推理与检查 |
| `config.py` | 数据路径、字段、时间范围、模型配方和调度配置 |
| `main_s.py`、`main_l.py` | 短长期独立模型训练、画像及部署后处理 |
| `main_opt.py` | 资源预测、气电标定、状态估计与每日 MILP |
| `data_cleaning/pipeline.py` | 只读数据加载、单位转换、信息可见性与因果特征 |
| `data_cleaning/quality.py` | 历史评分、独立约束核验与事后代理回放 |
| `output.py` | 模型序列化、源码冻结、CSV 导出及 ZIP 校验 |
| `main_nn.py` | 停用占位，不加载神经网络依赖或权重 |
| `learn.md` | 历史实验结论与证据索引，历史结果不代表当前实现 |
| `AGENTS.md` | 持续有效的工程、数据合规与本地验收约束 |
