# AIC 半决赛：煤气发电量预测与发电优化

## 运行

```powershell
python main.py            # 正式半决赛完整测试推理（默认 infer）——须由真人启动或明确授权
python main.py infer      # 与默认入口相同，训练冻结后生成半决赛结果包
python main.py run --quick # A/B 历史开发验证
python main.py infer --quick
python main.py run --experiment # 获准后的全量开发实验，写入 tmp/tmp_outputs
python main.py doctor     # 只读扫描数据目录
python main.py self-check # 因果清洗与特征自检
python main.py verify     # 校验最近一次输出
python main.py verify --run-dir tmp/tmp_outputs/<run_id> # 校验指定实验目录
python main.py run --experiment --nn-only --nn-span short --nn-kind tcn_causal --folds A,B
python main.py run --experiment --nn-only --nn-span resource --folds A,B
```

`--nn-only` 是隔离的神经网络开发入口，只允许 A/B/C，不运行 D 或调度器。
可通过 `--nn-seeds 3` 复验随机性、`--nn-reference` 同折训练树模型，
`--nn-config` 接收 JSON 对象覆盖本次实验参数。资源专项比较 MLP 与历史序列 TCN；
通过误差初筛后仍须完成 A/B/C 控制回放，不能自动替换部署模型。

2026-10-25 方案专项同样从唯一入口运行：

```powershell
python main.py run --experiment --plan-study baseline --study-reference outputs/<原始基线>
python main.py run --experiment --plan-study E1,E2,E3 --study-reference tmp/tmp_outputs/<方案基线>
python main.py run --experiment --plan-study E4,E5,E6 --study-reference tmp/tmp_outputs/<方案基线>
python main.py run --experiment --plan-study pressure --study-reference tmp/tmp_outputs/<方案基线>
python main.py run --experiment --plan-study availability --study-reference tmp/tmp_outputs/<方案基线> --nn-study-run tmp/tmp_outputs/<长期融合专项>
python main.py run --quick --experiment --plan-study observed --study-reference tmp/tmp_outputs/<已验证的可用性专项>
python main.py infer --experiment --availability-study-run tmp/tmp_outputs/<全量可用性专项>
```

`baseline` 按原归档的结构和融合权重重建 A/B/C 折外预测；E1 比较树模型
21/45 天半衰期，冻结三种子 TCN；E2 比较工况条件画像；E3 只改变小机组组分配，
保护总负荷。E4 固定高炉预测和原资源修正动作，比较零状态概率与正气量条件均值。
E5 在资源版本冻结且前缀折外路径齐备后推进，训练特征来自更早冻结的内层资源模型；
E6 用训练前缀内完整折外误差路径校准库存裕度，不可行时保留 point。
所有模型、必要参数、折外数组及精简摘要均留在当次 `tmp/tmp_outputs/`，
专项不推理 half_test、不重选 D、不打正式交付包，也不自动部署未晋级候选。

`availability` 固定正常输入、整窗缺发电耗气和第七天中断三种情景，重算全部耗气
派生特征，比较独立少特征分支与必要的连续块遮挡对照。只用 A/B/C 选择全折共用
的画像/分支权重，保持逐折逐目标保护；可通过 `--study-resume` 复用已保存模型。
`--availability-study-run` 只接收已完成的全量专项，按冻结配方重训最终模型，
不重新评估 D；实际使用的缺测分支及必要参考配方一并归档，可离线重载。
`observed` 为独立缺测候选补充起报点设备明细，以及 `(t-15min,t]`、
`(t-60min,t]` 的真实分钟统计和有效比例；原始 15 分钟取值不变，不补造缺测量。
旧模型不读取这些新增列。专项同时比较已保留策略，quick 只做 A/B 筛选，不能晋级。
新增分支可与已保留分支按全折共用的小比例融合，仍须逐折逐目标通过保护门槛；
两组实际模型及融合参数共同归档，重载后不依赖专项目录。
已通过完整门槛的短期少特征分支和长期分目标保护融合也写入唯一配置源，默认
正式训练会独立拟合最终缺测分支；完整观测仍走选定的主模型。
若只更新已通过联合验证的分支权重，可同时传 `--reuse-run <最终模型运行>` 与
`--availability-study-run <专项>`；要求复用模型已训练至当前合法截止点，不重复训练。
对通过完整门槛的 `observed` 专项，同一入口复用主模型，只独立重训缺少的新增特征分支。
`pressure` 只诊断 10 月 1～9 日起报点，训练截止 9 月 30 日；该已查看分布的
九天窗口不参加晋级，也不称为独立盲测。
资源专项中断后，可在同一命令追加 `--study-resume tmp/tmp_outputs/<中断目录>`；
续跑核对原数据来源、训练截止、标签索引和模型重载预测，只补尚未完成的模型，
原源码快照保留，续跑源码归档到 `源码/续跑/`。

`--nn-benchmark` 在固定更新步数下比较 FP32/BF16 与 batch，使用合成输入，
不产生预测分数。`--nn-reblend <专项目录>` 只重用已保存的 A/B/C 开发预测，
选择全折共用的分目标/提前量融合系数；门槛固定为均分提升至少 0.20 个百分点，
任一折任一目标下降不超过 0.20 个百分点。

`infer --experiment --reuse-run <本项目final运行> --nn-study-run <冻结专项>`
可复用未改变的最终树/资源模型，仅重训已通过完整 A/B/C 门槛的 NN；
模型截止点必须早于推理窗口，冻结专项可按短期、长期重复传入。
运行摘要明确记录模型来源和本轮未评估 D，不把既有诊断当作新确认。

全量 `run` / `infer` 训练耗时较长，必须由真人发起或明确授权；
2026-10-23 的历史单轮授权不自动延续，后续任务按 `AGENTS.md` §8 判断授权范围。

`--quick` 的完整运行产物写入 `tmp/tmp_outputs/<run_id>/`。开发实验也必须使用
该目录；`outputs/` 只保留用户明确归档的选定最终运行。

当前已进入正式半决赛阶段。`python main.py` 默认使用完整 `half_test` 窗口；
`python main.py run` 保留历史回测用途，生成的回测包不能作为半决赛测试结果。
本次结果是否满足提交条件单独记录于 `submission_check` 和 `submittable`，
未通过时终端与运行摘要列出具体原因，不再由固定的预开发状态统一判定。

## 数据

所有原始数据目录只读。`half_train` 已接入，新增 2025-10-01～10-10 的完整训练标签；
与旧 `semi_test` 重叠时，以新发布的 `half_train` 记录为准。

训练数据准备默认只读取 `role=train` 阶段，截止 2025-10-10 23:45。
`half_test`（2025-10-11～10-31）已接入，只用于冻结模型后的测试推理，
不参与负荷/资源训练、选参、融合、早停或物理标定；训练扫描也不读取该目录。
短期、长期、资源及 NN 训练入口会拒绝本地评测对象，即使传入其测试期之前的前缀。
测试推理显式开启测试输入，支持 `final_load_real.csv`，保持完整 2016 个起报点。
该文件的两个目标负荷当前全为空，无法计算官方测试集的本地 1-MAPE；
已发布耗气仅到 10 月 12 日，后续保持缺测，不补零伪造资源。

## 方法

- 15 分钟网格按同一时间戳取值，不使用未来分钟聚合；评测窗口目标整体遮挡。
- 短期 8 步、长期 96 步分别训练与选参：LastValid、历史同槽位画像、
  profile_fuel、Direct-H/Global-H LightGBM、Grouped-H 长期模型，按 1-MAPE 在
  三个 21 天调参折上选模，确认折只评一次。
- 原始数据缺少部分混气/生产用户列，无法仅凭“发生−已知用户”得到可信余气；
  优化侧以历史机组耗气作为可发电煤气包络，再叠加 2 号高炉气柜缓冲。MILP
  仍可在包络内重排各时段发电，而不是复制历史耗气作为结果。
- MILP 使用 4×50MW + 2×120MW、开机 60%~100% 额定、爬坡 10%额定/分钟、
  气柜 15%~90% 安全区（计划再留 5% 裕度）、江苏分时电价。焦炉/转炉没有
  独立柜位，不虚构库存。`availability_mode=point` 为默认确定性方案，
  `robust_low` 为保留的低分位研究模式。
- 资源评价将真实零气量的误报计入 WAPE，并报告偏差和 24 小时累计误差。
  A/B/C 前缀样本外逐气种选择 L1/L2；不在 D 窗口标定分位数或选择控制策略。
- 小时控制另训练独立八步资源头，按短提前量 WAPE 筛选，再与原资源模型进行
  联合调度晋级比较；全量模式在 A/B/C 完整 21 天上核验直接替换及 50% 融合，
  不允许增加已有控制的降载。长期尾部保持独立模型，只有胜出的头模型随本次归档。
- 独立八步资源残差网络仅用起报前前缀训练、内部早停后重训，缺失指示与归一化
  参数随模型保存。高价段的持久性/网络正向分歧修正与 15 分钟重规划均为候选，
  还须通过短端 WAPE、完整 A/B/C 收益、电价效率及已有控制降载门槛；推理只读取
  起报点及以前的过程量和提前公布电价。缺测时保持原资源预测，96 步尾部不改。
  仅归档本次真正选中的资源网络，未胜出的网络不进入模拟包。
- 连续代理回放比较每日与每小时重排，均从同一初始库存接续。目标时刻观察到
  资源后才执行降载；代理库存变化计入收益调整。控制策略先在开发折前 48 小时
  初筛，资源头和高价段修正再以完整 A/B/C 各 21 天确认，并设收益、降载、代理
  释放和电价利用率晋级门槛。气种补偿只在目标时刻观察
  到资源后进行，提前计划与执行结果分开统计；每次重规划的首步也计启停切换。
  高炉气触顶时可在保持功率、库存不变的条件下置换部分其他气种，降低高炉气
  代理释放；这不表示真实气网总放散减少。
  柜位不作为代理资源模型输入，避免注入历史控制状态。
  所有执行收益仍是历史耗气包络仿真，不是完整气网或实际生产收益证明。
- 优化摘要分别保存计划、代理执行和历史发电的收入/电量/按日平均电价计算的
  加权负荷比；代理高炉气释放按多个假设罚金做敏感性评价。无柜气种的未用包络
  不作为实际放散，缺真实净余气、历史放散及官方罚金时不计算官方净效益提升率。
- 压缩包统一命名为 `咕咕嘎嘎-半决赛.zip`。五个优化输出字段已核对本地赛题
  PDF 第十三节（三）；物理口径仍须核验，代理优化保留模拟标记。
  正式测试推理成功记录为 `SUCCESS_INFER`，提交核验结果单独记录。

## 输出

`outputs/<run_id>/`：`s_result.csv`、`l_result.csv`、`opt_result.csv`（有效模拟解才生成）、
`咕咕嘎嘎-半决赛.zip`、`源码/`、`模型/`、`报告/运行结果.json`、
`报告/运行日志.log`。

标定、特征顺序和融合参数合并在 `模型/model_meta.json`，只归档本次实际用于
预测的确认或最终模型；源码包含 `main_nn.py`。`--quick` 使用 A/B 开发折，NN
最多 8 轮内部早停试验并按选定轮数重训，优化仅抽取前 2 天，不代表全量结果。
历史多步错位修复前的分数不作新基线；D 窗口已参与历史开发，不声称独立盲测。

当前 `infer` 使用已接入的 `data/half_test` 四表，对完整官方评测区间进行推理。
若在其他环境中未放置该输入，兼容分支仍可使用 `semi_test`
（2025-10-01～10-10）做链路干跑；这不代表当前项目缺少半决赛数据。
已提供但不完整的官方四表会报错，不静默缩短评测窗口或退回旧窗口。

`opt_result.csv` 的 `datetime` 采用调度目标时刻（t+15min…t+1440min）。五个
优化输出字段已按赛题要求实现；提交前仍须核验覆盖范围、单位和时间语义。

datetime 格式按赛题第十三节要求使用 `YYYY-MM-DD HH:MM:SS`，带空格。

## 环境

Python 3.10，依赖见 `requirements.txt`；全部计算可离线运行。

终端默认显示分阶段及嵌套 `tqdm` 进度，覆盖数据读取、候选/时序折、LightGBM 迭代、
NN 训练轮次与批次、批量预测、调度重规划、模型归档及打包；进度中的剩余时间为估计值。
重定向输出时自动关闭进度条，运行日志保持纯文本。IDE 控制台可加 `--progress`
强制显示；加 `--no-progress` 只保留日志，例如 `python main.py infer --experiment --progress`。
