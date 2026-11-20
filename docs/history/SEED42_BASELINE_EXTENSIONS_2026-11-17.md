# seed42 补充 baseline 实验（2026-11-17）

**11月18日结束核查**：30项中25项完成、5项失败，tmux已退出，首尾墙钟8小时47分17秒。失败为两池HGGS、两池Lev和LHS Physics-PRE-Power，不得把部分结果当终评。PEAKS同流程随机对照小幅改善；ERACS两池各39次检查但0次重选；原mutation-Power仍具最好的综合表现。完整分析及适配边界见[11月18日报告](../../analysis/sampling_extensions_20261118/report.md)。本轮仅分析，没有自动补跑。

## 当前批次

入口仍为 `bash run_sampling_budding.sh extensions --directory <队列目录>`，后台 tmux 会话为 `budding_extensions_seed42_20261117`。

队列目录：`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling_extensions/seed42_20261117/`。

其中 `queue.log` 是调度日志，`tasks.json` 给出逐项状态、PID、GPU、实验目录和输出验证结果，`status.json` 是当前汇总，`logs/<pool>_<method>.log` 是子程序输出。资源采样在 `resources.jsonl`。这些是队列文件，不是重复的实验结果或轨迹。

共30项，仅FNO、seed42，mutation和LHS各15项；原已完成的8种固定池方法不重跑、不覆盖。

|类别|方法|
|---|---|
|固定池补充|low_rank、lev_input、clustering_sensitivity、clustering_sensitivity_wor、physics_pre_topk、physics_pre_power|
|连续训练与动态子集|peaks_regression、eracs_regression|
|开放生成|hggs_projected、oed_temporal、ogas_async_projected|
|对应流程随机对照|random_generate、random_async、random_stream、random_dynamic|

前四项依次为mutation的Low-Rank、PEAKS、ERACS、HGGS，然后是LHS对应项，余项排队。结果不改变公共流程的目录：固定池在 `experiments/budding_sampling/seed42_a866aef5d988/<pool>/<method>/seed_42/`；生成式在 `experiments/budding_generation/<protocol>_gen<hash>/` 或 `_online<hash>/`；动态方法在 `experiments/budding_dynamic/<protocol>_dynamic<hash>/`。具体路径以 `tasks.json` 的 `result_directory` 为准。

## Low-Rank的还原边界

[原论文](https://proceedings.mlr.press/v306/cohen-addad26a.html)合成实验使用平方范数代价；另一实验用逐样本梯度的**范数**代替loss。当前实现将FNO输出头梯度投影为特征，并用平方范数 `ell(v)=||v||²` 作为选择代价。它结合了原文已有的构件，但不等于上述任何一组实验的逐项照搬。

Algorithm 2的低秩基方向项、投影残差项、常数平滑项、有放回抽样与重要性权重均保留；正式表述为 **Low-Rank-GradientEnergy任务适配**，不宣称原Relative L2的理论保证。10000个抽样槽可能对应更少唯一输入，pilot、标签访问、重复和实际训练曝光分别记录。FNO训练仍使用共同的Relative L2，不以选择代价替换训练loss。

PEAKS的回归参考梯度评分、ERACS的梯度投影/分组OMP、HGGS的合法提案投影也都有适配，详见 `REGRESSION_SAMPLING_BASELINES_2026-11-17.md`。不能把所有方法解释为完全相同训练日程或仅查询10000标签；各自流程及随机对照保留、成本另报。

## 数据和训练协议

复用 `benchmarks/budding_sampling/seed42_a866aef5d988` 的冻结修复池、划分和真实突变体参考。固定池五轮为2000/4000/6000/8000/10000条、100/100/100/100/200 epoch，batch128、LR5e-4、width192、blocks4、modes24。每轮训练集refit，按同池归一化验证loss选best；保留last。

连续训练类保留自己的persistent optimizer与初始归一化冻结规则，详见各自config；不冒充固定池五轮协议。real、validation和test不参与采样评分或梯度。允许监督候选筛选的方法只访问训练侧，并记录实际真值访问量。

LHS生成复用原池保存的同一规格字节；队列的 `frozen_lhs/` 仅补建已核对池指纹的provenance sidecar，不更改边际和生物规则。本轮启动前82个Budding/原始生成器文件SHA全部一致。

## 并发和启动验证

仅GPU0～3。先4项、每卡1项；15分钟后，已有至少2项实际进入epoch且资源允许时，每90秒增加一个槽，最多8项、每卡2项。系统可用内存保留100GiB，不设自有PSS硬上限；新项准入还要求124GiB可用、对应GPU空余12GiB、数据盘空余50GiB。CPU和持续磁盘压力保护只管理本批自有进程组，不终止其他用户任务。

重加载并发4、评分并发2、生成每项ODE worker2。原始轨迹RAM块缓存上限1024块，减少随机访问反复解压，不重复保存轨迹到磁盘。修复动态初始化和训练adapter的嵌套I/O锁，避免全部槽被外层占满后等待内层锁。

W&B强制offline。队列固定源代码指纹；修改算法源码后拒绝混用同批配置。子程序退出0还须核对completion状态、126株real指标与CSV、同池test CSV和126幅图片，才记为完成。

启动前测试：最终队列专项6项通过；此前相关组合22项通过（有重叠，不累加为独立数量）。实际FNO192/4/24、batch128、T500的GPU核查完成一次前向/反向和新增选择内核：allocated峰0.863GiB、reserved1.039GiB。该测试仅128条训练侧样本，不代表完整实验或90k梯度评分的主机内存峰值；正式队列仍逐步观察资源。测试不提供性能优劣结论。

20:56启动复核：4项全部已进入真实训练，Low-Rank pilot记录46个epoch、HGGS pilot29个epoch、PEAKS/ERACS各3个epoch；26项等待，失败0。当前本批PSS约47.57GiB、系统可用471.84GiB，守护未触发；这只是启动阶段观测，不是全程峰值。4项W&B均offline，217个运行源码及82个生物/旧生成器文件指纹无变化。证据保存在队列目录的 `startup_verification.json`。
