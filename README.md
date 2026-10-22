# AIC 半决赛：煤气发电量预测与发电优化

## 运行

```powershell
python main.py            # 完整链路（历史 21 天盲测）——须由真人启动
python main.py infer      # 半决赛特征窗口干跑——须由真人启动
python main.py --quick    # 单折开发模式
python main.py infer --quick
python main.py doctor     # 只读扫描数据目录
python main.py self-check # 因果清洗与特征自检
python main.py verify     # 校验最近一次输出
```

全量 `run` / `infer` 训练耗时较长，**必须由真人发起**；Agent 侧只使用
`--quick` 与诊断命令。详见 `AGENTS.md` §8。

## 数据

`data/first_train`、`data/first_test`、`data/semi_train`、`data/semi_test` 只读。
半决赛正式评测区间 2025-10-11～10-31 的输入尚未发布；`data/half_test/` 为预留目录。
程序不会根据目录名假定标签存在，扫描结果决定可用标签。

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
- 缺少官方 `opt_result.csv` 模板，当前只生成 `模拟评分包_不可提交.zip`。

## 输出

`outputs/<run_id>/`：`s_result.csv`、`l_result.csv`、`opt_result.csv`（模拟）、
`标定/资源标定.json`（模型选择、特征白名单、Top 特征重要性、热值映射）、
`模拟评分包_不可提交.zip`、`源码/`、`模型/`、`报告/运行结果.json`、
`报告/运行日志.log`。

`infer` 在 `data/half_test` 就绪时对准官方评测窗；否则用 `semi_test`
（2025-10-01～10-10）做提交链路干跑，产物同样标记不可提交。

`opt_result.csv` 的 `datetime` 采用调度目标时刻（t+15min…t+1440min）。官方
模板尚未发布，正式提交前必须按官方行列语义重新适配。

datetime 格式与原始 CSV 一致（`YYYY-MM-DD HH:MM:SS`，带空格）。赛题 PDF
文本写作 `YYYY-MM-DDHH:MM:SS`，疑似排版丢失空格；若官方校验器要求无空格，
只改 `output.py` 的 strftime 一行即可。

## 环境

Python 3.10，依赖见 `requirements.txt`；全部计算可离线运行。
