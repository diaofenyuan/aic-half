# AGENTS.md — 项目永久记忆

本文件是本项目对后续所有 Agent/开发者的常驻约束。开始任何任务前先读本文件；
除非用户明确要求，不要删除或弱化其中的规则。

## 1. 项目与数据

- 项目：`D:\python_play_do\aic-half`，AIC 半决赛“煤气发电量预测与发电优化”。
- 唯一日常入口：`python main.py`；开发子命令：`doctor`、`infer`、`self-check`、
  `verify`、`--quick`。`infer` 在官方特征窗口（缺 half_test 时用 semi_test）
  干跑提交链路，产物仍是模拟包。
- 数据目录 `data/first_train`、`first_test`、`semi_train`、`semi_test` 只读，
  不改名、不覆盖、不上传、不打进 ZIP/Git。
- **当前没有正式半决赛评测数据**（2025-10-11～10-31）。`data/half_test/` 只是
  预留目录；缺失时走历史 21 天盲测，产物只能是
  `模拟评分包_不可提交.zip`，禁止标记 `SUBMITTABLE`。
- 禁止使用参考时刻之后的动态生产量、目标真值、测试未来耗气；评测窗口目标
  必须整体遮挡。未来日历和官方提前公布的电价可按定义使用。

## 2. 工程结构

只允许：

```text
main.py  main_s.py  main_l.py  main_opt.py  main_nn.py  config.py  output.py
data_cleaning/{__init__.py,pipeline.py,quality.py}
requirements.txt  README.md  .gitignore  AGENTS.md
data/  outputs/  tmp/
```

不得新增 `src/`、`utils/`、`scripts/`、`tests/`、notebook、多套 YAML/JSON
配置树。短期与长期必须独立训练、独立选参、独立保存；不得把 96 步结果裁成
8 步冒充短期模型。子模块不得反向 import `main.py`。

## 3. 模型与实验产物（用户明确要求）

- **所有实验、调参、消融、临时预测、临时日志、实验模型权重一律写入 `tmp/`。**
- **禁止把实验模型写入 `models/`。** 本项目不建立 `models/` 目录。
- 只有 `output.py` 归档的、本次真正部署的最终模型，才允许写入
  `outputs/<run_id>/模型/`；`模型/` 是提交交付物，不是实验垃圾区。
- `tmp/` 已被 `.gitignore` 忽略；实验产物不得进入源码快照、模型归档或 ZIP。
- 正式运行的 `源码/` 快照必须是当次磁盘真实代码；缓存/实验模型不得混入。

## 4. 数据契约

- 15 分钟网格按**同一时间戳**取值，不对 `t` 之后分钟做聚合、插值、`bfill`。
- 目标列只作标签或起报点之前的历史；`PreparedData.masked_copy` 负责整段遮挡。
- 目标标签绝不插补；0 负荷是合法工况，不得当异常删除或替换成历史均值。
- 1 号高炉气柜数据大量缺失且含 1e30 哨兵，视为不可用；焦炉/转炉无独立柜位，
  不得虚构气柜。
- 转炉煤气 `converter_1` 量纲与其它流量不一致，`CONVERTER_GAS_SCALE` 为历史
  反推值（inferred），正式数据字典确认前不得当作官方常数。
- 原始数据缺少部分混气/生产用户列，优化侧使用历史机组耗气作为“可发电煤气
  包络”，MILP 仍须在包络内重排时段，不得直接复制历史耗气作为优化结果。

## 5. 选模与评分

- 时序折：A/B/C 调参折，D 冻结确认折；每折连续 21 天，训练标签最大目标时刻
  必须早于验证窗口。
- 主指标 `1-MAPE`；零负荷真值单列，不参与 MAPE 分母。
- 选模必须看目标、提前量分组、最差折；确认折只评一次，不得反复调参。
- 无正式评测数据时只报告历史盲测分数，不冒充官网成绩。

## 6. 发电优化

- MILP：4×50MW + 2×120MW；开机 60%~100% 额定；爬坡 10%额定/分钟换算到
  15 分钟；气柜 15%~90% 安全区（计划留 5% 裕度）；江苏分时电价。
- 缺官方 `opt_result.csv` 模板、热值/效率、柜位映射时，只生成模拟调度表，
  不得声称可提交。热值/效率由历史回归标定，标记为 inferred。
- 默认 `availability_mode=point`（确定性 O1）；`robust_low` 是研究模式，
  历史回测显示其显著牺牲收益且不能消除预测误差风险，未经重新验证不得默认启用。
- `opt_result.csv` 的 `datetime` 当前为目标时刻语义；官方模板发布后必须重新
  对照行列、单位、起报点/目标时刻定义。

## 7. Git 与协作

- 远端提交不得添加 Codex、ChatGPT、Claude 等 AI 为 contributor / co-author，
  提交信息不强调 AI 生成。
- 严格控制修改范围：不做无关重构、全仓库格式化、过度设计。
- 代码注释用中文，解释设计原因与关键逻辑，不写教学式长注释。
- 不生成过多 Markdown/报告/图档；测试聚焦本次改动的关键链路。

## 8. 训练与运行纪律

- **禁止 Agent 自行发起全量训练**（`python main.py run` / `infer` 不带 `--quick`
  的完整训练）。全量训练必须由真人明确发起；Agent 只允许 `--quick`、`doctor`、
  `self-check`、`verify` 等轻量命令，或在真人已启动的训练旁做只读检查。
- 需要全量结果时，向用户说明命令与预计耗时，由用户决定何时执行。

## 9. 当前已知限制（预开发状态）

- LightGBM 短期/长期与 profile_fuel 融合已跑通；TCN/DLinear/N-HiTS 尚未实现。
- 资源模型对焦炉气（零值多）误差较大，转炉气约 35% WAPE；仅作优化输入，
  不冒充高精度资源预测。
- 日尺度开环调度下，真实气量误差可能使实际柜位偏离计划；报告同时给出
  `plan_feasible`（计划自身可行）与 `realized_holder_low_steps`（实测压力诊断）。
- 正式半决赛数据到位后：重新训练最终模型、适配官方起报点与 `opt_result`
  模板、核对单位/柜位/电价，再允许 `SUBMITTABLE`。
