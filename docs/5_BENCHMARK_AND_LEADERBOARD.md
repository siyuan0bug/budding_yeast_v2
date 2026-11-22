# 实验结果与比较入口

## 当前结果位置

结果根：`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/`。

- [结果导航](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/index.html)：模型、池、算法的目录/指标/预测图/日志。
- [全部当前指标](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/all_current/report.md)。
- [选定baseline与三个ours](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/selected_baselines_three_ours_20261119/report.md)。
- `batches/`保存任务清单、资源、迁移和清理审计；正式成绩位于模型/算法的eval目录。

## 已有证据的范围

冻结协议的65项完整结果已核对，包含不同baseline、对照、ours及历史变体，不是65个独立同协议算法。
当前比较主要为FNO、seed42；普通训练预算2k/4k/6k/8k/10k，100/100/100/100/200epoch，具体以原config为准。
mutation-Power的real MAE约0.15030、MSE2.25805；当前ours尚未表现出全面稳定优势。
HGGS历史成绩属于旧控制流程，修正版本没有被本次目录治理重新训练。
主Relative与其他轨迹诊断定义见[评测指南](3_LOSSES_AND_PHYSICS_CONSTRAINTS.md)。

## 公平比较与更新方式

不同持续/生成/一次性方法的标签访问、历史训练曝光和优化步数不同，须连同同流程随机对照、时间与资源报告。
不得只报最佳seed，不得将旧定义Relative、错误版本或不同预算混入同一主排名。
迁移不改变模型/轨迹/评测值；路径元数据与校验更新需要审计。
新实验完成后运行汇总工具，更新此处的当前结论和报告入口；详细表格、图、日志留在外部结果目录，不逐次向本文追加流水账。
历史比较及旧版模型成绩保留在[整理前快照](history/GUIDE_SNAPSHOT_5_BENCHMARK_AND_LEADERBOARD.md)，不能自动视为当前协议成绩。

当前FNO三seed终评分析入口：[多seed原始指标与汇总](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/multiseed_20261122/report.md)、[Ours诊断与后续验证计划](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/multiseed_20261122/ours_improvement.md)。仅七类baseline具备三个seed，现有ours仍为单seed；不得混为多seed优胜结论。

三种Adaptive生成方法的42/123/1024终评：[全指标与目录](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/adaptive_multiseed_20261122/report.md)、[机制诊断](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/adaptive_multiseed_20261122/findings.md)。九项均完成，当前结果尚未稳定超过Power；无锚点固定分数及实际响应覆盖不足须先验证修正。
