# 实验结果与比较入口

## 当前结果位置

结果根：`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/`。

跨会话/平台接手使用[研究交接入口](research/README.md)，最新研究状态以它指向的研究记录及原始完成凭据为准。R16响应缺失生成原型已[筛选阴性](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/fno/mutation/ours_response_omission_probe/diagnostic_2k_plus256_steps400/seed_42/findings.md)，不自动开展确认或正式长跑。

[R17生成范式证据审计](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/generation_paradigm_evidence_v1/report.md)核对42项原始终评、21组同方法/seed配对及两个10万池输入：mutation的MAE/MSE/Corr在21组均优于LHS，但主Relative仅7组优于LHS。90维背景域差异确认，不能将生成协议总体优势单独归因于mutation联合结构；下一步控制尚未启动，原池规则保持不变。

- [结果导航](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/index.html)：模型、池、算法的目录/指标/预测图/日志。
- [采样idea研究记录](research/journal.md)：按R01起的固定编号追踪假设、对照、负结果、证据与方向调整；当前下一方向为合成训练侧机制错误驱动选样，尚未验证。后续沿同一记录维护，不为每次讨论新建平行记录。
- [R13初值与演化分解](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/scalar_mechanism_diagnostic_v1/report.md)：四个既有模型、366条既有合成反馈、三条独立合成—降解方程；CPU只读推理，无新训练或ODE。揭示初值识别与后续演化的不同错误，尚未构成有效采样算法。
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

Response V2与路径随机对照三seed已完成：[全指标和目录](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/response_v2_results_20261122/report.md)、[负结果及覆盖/变量诊断](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/response_v2_results_20261122/findings.md)。响应机制运行到位，但未取得训练收益，不能将评分相关性当作算法胜出。

跨背景干预迁移已完成seed42、四组等400步的[实际学习收益探针](/home/users/zsy/mydata/budding_yeast/resource_profiles/background_transfer_probe_20261122/report.md)。相对同生成器随机组归一化Relative仅改善约0.29%，未通过预定2%条件；不作为正式10k结果，不据此启动多seed长跑。探针不使用real或正式holdout标签，见[实现边界](reference/BACKGROUND_TRANSFER_PROBE.md)。

训练侧[随机／真实高误差／反馈试训选批诊断](/home/users/zsy/mydata/budding_yeast/resource_profiles/sampling_utility_probe_20261122/report.md)已完成，同一划分下用训练RNG42/123复核。高误差追加相对随机的归一化Relative平均仅改善约0.14%，分组bootstrap区间跨0；反馈选批选中了同一高误差批次。该400步增量诊断不支持单纯困难度带来明显跨背景收益，也不替代完整冷启动重训；未读real/正式holdout标签。见[诊断协议](reference/SAMPLING_UTILITY_DIAGNOSTIC.md)。

[随机1万/3万数据量参考](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/data_volume_10k_30k_seed42_20261122/report.md)已完成seed42，两个内部探针MAE改善约43%～49%。固定best后的126株real补充终评：MAE 0.17746→0.11934、Relative 12.36778→0.53508，MSE 3.44796→3.89046变差；逐株MAE改善120/126。3万参考同时将更新数从15800增至47000，不能单独归因数据量或混入1万预算采样排名。近零分解、逐株CSV及后续等更新控制建议见[结果解释](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/data_volume_10k_30k_seed42_20261122/findings.md)。

[近似等更新次数控制](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/matched_updates_10k_30k_seed42_20261122/report.md)已完成：1万595epoch=47005步的real MAE/MSE/Relative为0.16867/4.83186/0.65011；3万200epoch=47000步为0.11934/3.89046/0.53508。训练充分能消除大部分近零Relative异常，更多唯一输入在近似等更新下仍有收益；但单seed且训练归一化各自拟合，不是纯数据量因果证据。旧结果保留。

[条件动力学七组研究对照](reference/CONDITIONAL_DYNAMICS_STUDY.md)的21项训练已全部完成并核验：seed42/123/1024，一次性1万条/47005步，best为468—594epoch。本批用时4小时9分、自有PSS峰值52.12GiB，系统有效余量最低631.04GiB；无工作流失败，1次ODE失败已补足。Real平均MAE：池内随机0.15911、轨迹覆盖0.16026、仅条件覆盖0.16659、机制覆盖0.15422、组合迁移0.19156；随机生成0.15016、机制缺口生成0.15594。机制覆盖对池内随机只赢1/3seed，不能宣称稳定胜出；合成测试MAE三seed均改善7.31%。组合迁移与生成版暂不支持扩大。Relative仍有精确零通道主导的seed波动，不删除异常指标。本批不是旧五轮协议，不能直接与旧Power/LCMD排名混排。完整指标、配对变化、选择背景分布及完成核验见[结果报告](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/conditional_dynamics_v1/report.md)。

[干预阶数校准](reference/INTERVENTION_ORDER_CALIBRATION.md)已在seed42筛选阴性后按规则停止：3模型595epoch/47005步，用时52分56秒，无任务失败；123/1024确认跳过，独立审计/test/real未开放。均衡/低阶优先虽使共同坐标反馈Relative改善9.58%/11.58%，但均衡物理MAE变差24.49%/21.58%，物理Relative约为随机8倍。L1 MAE改善约42%，L2/L3受损，不能将L3视为无用；只有单seed且归一化各自拟合，不是纯L因果证据。不继续搜索比例或把固定quota称为创新。详见[分层复核报告](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/intervention_order_calibration_v1/report.md)。

反馈定点复核没有新训练或ODE调用：两种quota的L2 MAE净增量约91%/86%来自ORI；28条解析条件保证CDH1T恒零的反馈样本，随机模型平均预测0.00758，两种quota约0.99990/0.99973，且各自训练中仍有924/760条该通道全零轨迹。相同0↔1误判在当前归一化Relative下惩罚明显不对称，与物理Relative的权重方向不同；这是目标错位的证据，尚非归一化变化导致全部退化的因果证明。详表在同报告的反馈定点排查及feedback_diagnosis目录；独立审计、正式test和real仍未开放。
