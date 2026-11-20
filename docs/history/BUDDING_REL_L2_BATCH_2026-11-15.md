# Budding纯Relative L2重跑（2026-11-15）

用户要求后续实验统一采用现有Relative L2训练损失，归档原结果，GPU0–3后台运行更多已实现baseline，同时严格控制内存和I/O。

## 损失与公平比较

原批次为`hybrid_mse_rel`：归一化MSE + 0.2×RelativeL2。新批次为`rel_l2`，直接使用`utils/training/losses.py:RelativeL2Loss`，没有改其实现或原默认eps=0.1、max_clamp=10：

\[
L=\frac1{BV}\sum_{b,v}\min\left(\frac{\|\hat z_{bv}-z_{bv}\|_2}{\sqrt{\|z_{bv}\|_2^2+0.1^2}},10\right).
\]

z是log1p后Z-score的变量轨迹。上限10约束的是损失比值，不是变量值或仿真轨迹。模型仍为FNO width192/modes24/blocks4，batch128，LR5e-4→1e-5，AdamW weight_decay1e-4，5轮100/100/100/100/200 epochs、warmup5/5/5/5/10。每轮冷启动、仅按累计合成训练集重新拟合归一化。

新批次train_loss/val_synth/test_synth/test_real统一使用同一定义的归一化Relative L2；历史物理test_real_denorm口径保持不变，checkpoint仍由同池验证的物理macro trajectory RelL2选择。归一化MSE保存在history.csv的独立诊断字段`val_mse_norm/test_synth_mse_norm/test_real_mse_norm`；最终物理MAE/MSE/Relative L2/Corr继续报告。测试仍只读监控，不参与选样/梯度/停止条件。

BenchmarkConfig、train.py入口默认损失更新为rel_l2；run_sampling_budding.sh历史在线入口默认也采用rel_l2（显式LOSS_TYPE可覆盖）。未删除其他损失实现，历史checkpoint带自身loss_type保持可辨识。

## 数据复用

- 两个100k完整源池不重生成，real reference不重仿真。
- 新协议：`/home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_bb035a147b38`。
- 由原`seed42_18bf9a3289fa`仅修改loss派生，四个input/split NPZ均硬链接同一inode、同一SHA；初始2k、validation/test及全部候选ID完全一致。
- `derive_training_protocol`拒绝seed/budgets/holdout/validation_scope变化的复用，验证源池定义和每个缓存哈希，不解压完整源池来重建相同划分。
- 五轮累计训练量2k/4k/6k/8k/10k，每次新增2k，seed42。

## 批次及队列

8种baseline × mutation/LHS两池，共16项；均项目内独立实现：Random、Uncertainty TopK、Power、预测轨迹MaxDist、隐藏特征CoreSet、隐藏特征LCMD、隐藏特征BAIT、VeSSAL penultimate。

| GPU | 按顺序运行的完整实验 |
|---|---|
| 0 | mutation/random → mutation/coreset_ll → lhs/random → lhs/coreset_ll |
| 1 | mutation/topk → mutation/lcmd_ll → lhs/topk → lhs/lcmd_ll |
| 2 | mutation/power → mutation/bait_ll → lhs/power → lhs/bait_ll |
| 3 | mutation/max_dist_pred → mutation/vessal_penultimate → lhs/max_dist_pred → lhs/vessal_penultimate |

主入口仍为run_sampling_budding.sh；使用neuralop解释器。GPU4–7不使用。每张卡只运行一项本批任务，同卡其他实验排队；W&B强制offline。普通实验失败会记录退出码后继续下一项，资源门控仍优先。

## 资源保护

- 全局最多4项实验、每卡1项。
- 跨进程最多2项重加载/解压/归一化；绘图每项最多2个CPU worker。
- OMP/MKL每进程2线程、OpenBLAS1线程，DataLoader0 worker；子任务nice10/ionice2,n7，父调度器不重复降nice。
- 新任务与重I/O等待MemAvailable≥128GiB、load≤1.5×64、memory PSI≤1%、IO PSI≤5%。
- 10秒间隔监控自有批次整个进程树PSS；PSS>96GiB或有效可用内存<128GiB时停止自有批次进程组，不影响其他用户任务。
- 这些是协作式门控与watchdog，未设置系统cgroup硬限额；不使用会干扰CUDA虚拟地址空间的ulimit -v，不清理全机page cache。
- 所有训练清单引用源池，不重复保存轨迹；同文件系统归档不复制数据。

## 结果与归档

新结果根：`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/seed42_bb035a147b38`。
每项为`<pool>/<method>/seed_42`，train.log/五轮loss/每轮eval/私有_state结构保持最新规范。batch/launch_plan.json列出16项、配置和限制，launcher.log记录调度，resource_samples.jsonl记录资源。

原8项已完成结果原地重命名归档到：
`/home/users/zsy/mydata/budding_yeast/experiments/archive/budding_sampling_hybrid_mse_rel_20261115/seed42_18bf9a3289fa`。
校验5542个普通文件inode/大小均不变，约5.03GiB，所有最终eval软链接仍有效。archive_manifest.json记录原/新位置，不修改历史checkpoint、指标或代码来源记录。


## 完成核查与结果（2026-11-15）

16项全部完成，最后结束时间为北京时间12:13:44，退出码全0、无Traceback，五轮完整。已核对80份每轮历史/评测、对应逐样本CSV行数及10080幅real图的数量；批次进程树PSS峰值40.73GiB，未触发内存保护。当前无训练任务或等待任务。

只按旧批次同样存在的Random/Uncertainty/LCMD/BAIT进行8组配对。mutation均值MAE降低10.72%、MSE降低29.33%，但历史RelativeL2升高27.75%、全局Correlation降低0.04855；LHS综合收益不稳定。训练RelativeL2与历史物理评测口径不同，不能仅因名称相同就预计后者必然降低。完整对比报告见`/home/users/zsy/mydata/budding_yeast/analysis/sampling_loss_comparison_20261115/report.md`，主文档同步于docs/5_BENCHMARK_AND_LEADERBOARD.md。


## 2026-11-16 新批次准备（尚未启动正式训练）

此前章节是已完成的历史批次；当前配置以本节及损失文档第12/13节为准。

- Relative L2：训练与物理评测均sqrt(norm²+0.01²)、无loss cap、梯度范数裁剪1；训练归一化空间、评测原始物理空间。新物理版本`physical_sample_channel_rel_l2_smooth_eps001_v3`。
- best checkpoint：同池合成validation的`val_sample_channel_rel_l2`；与主评测公式和推理精度一致，real/test不参与选择。
- 新协议：`/home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_02eebd57592f`。四个输入/切分缓存与旧协议同inode，pool和初始样本/切分完全复用。
- 新结果：`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/seed42_02eebd57592f/<pool>/<method>/seed_42/`；train.log及rounds/round_XX/{history.csv,loss.png,model.ckpt,eval/}不变。旧协议结果留在原处。
- 训练超参数维持FNO192/4blocks/modes24、batch128、LR5e-4、minLR1e-5、weight decay1e-4、五轮100/100/100/100/200epoch和5/5/5/5/10warmup，累计2k/4k/6k/8k/10k。
- 默认8baseline×两池，共16项：random/topk/power/max_dist_pred/coreset_ll/lcmd_ll/bait_ll/vessal_penultimate。此次默认不包含ours。
- 总并发从4改为6、每卡最多2项、数据加载/解压仍2项。批入口自动记录batch/resource_samples.jsonl；owned PSS>96GiB或有效可用内存<128GiB时停止本批已注册隔离子进程并停止追加任务；普通实验报错仍按CONTINUE_ON_ERROR继续。协作式监控不是OS硬限额，不停止别人的作业。

资源实测：2/4/6项完整预处理后的PSS为13.689/27.019/40.396GiB；六项同时训练/评测时峰值67.744GiB，最低可用内存645.465GiB，各进程GPU allocated峰值0.902GiB。六项均完成、退出码0；memory PSI和IO full PSI未触发压力。资源测试使用GPU2/3各三进程，正式默认每卡最多2项；没有使用4--7。此为warm-cache、两epoch资源验证，没有比较完整4/6吞吐，也未同时压测6项90k acquisition/ensemble辅助训练及全部绘图。资源报告位于`resource_profiles/concurrency_2_4_6_smooth_eps001_20261116/`。

已核对dry-run显示16任务、总6并发/每卡2；SKIP_PREPARE路径解析到新协议，不做ODE生成。正式训练尚未启动，计划由用户运行：

```bash
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2
SAMPLING_MODE=offline SKIP_PREPARE=1 bash run_sampling_budding.sh batch --gpus 0 1 2 3
```

建议在tmux中执行。W&B强制本地offline。新loss/指标下的完整精度、收敛及旧checkpoint重新评测尚未开展，旧JSON分数不可与新版本直接混算。


## 2026-11-16正式启动

用户确认新目录不覆盖旧结果后授权启动。实际启动时间：2026-11-16T02:40:43.672579+08:00（Asia/Shanghai）；tmux：`budding_sampling_eps001_20261116`，使用neuralop环境。新目录`seed42_02eebd57592f`启动前不存在，与旧`seed42_bb035a147b38`真实路径不同；旧16份实验config的SHA256及inode已保存并在启动后复核一致。

16项任务已提交，最多6项并行，重加载2项；GPU0--3，每卡最多2项。显式指定冻结protocol和real_reference，跳过池/初始集/real生成；W&B强制offline。初期WAIT shared holdout/queried trajectory loading是读取限流，不是重复仿真。

计划与命令：`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/seed42_02eebd57592f/batch/launch_plan.json`、`command.txt`；启动日志`batch/launcher.log`；自动资源监控`batch/resource_samples.jsonl`。生产代码按source_inventory白名单另存`batch/source_snapshot.tar.gz`，不含数据与run_pipeline.sh。各项日志`<pool>/<method>/seed_42/train.log`。本节记录启动状态，不代表已训练完成。

启动后已验证6份新config均eps0.01及sample/channel validation；6份W&B均offline、无Traceback，LCMD与MaxDist已有实际epoch及checkpoint，其他4份按CPU/IO门控等待。代码source_inventory与启动snapshot一致，旧16份配置哈希/inode仍一致。详情见batch/startup_verification.json；仍为运行中状态，非完成结果。

## 2026-11-16完成及新旧统一评测

最新16项全部完成，最晚完成时间2026-11-16T08:17:49.612372+08:00（Asia/Shanghai）；16项退出码均0、无Traceback，80个新round均100/100/100/100/200epoch完整、最终10000唯一训练输入、无新增ODE。连同旧批共32项/160轮、20160幅real图数量检查通过。当前无运行/等待任务。

旧批是pure Relative L2（训练eps0.1、cap10、整轨迹macro validation选best）；新批为eps0.01、不截断、物理sample/channel validation选best。验证推理精度和并发也改变。两批old/new原JSON的Relative L2不可直接比较，因此32个最终checkpoint均使用同一real参考、原batch128、float32推理、物理smooth eps0.01重算；新模型重算与已保存主指标一致。没有重写历史结果。

|候选池|指标|旧八方法均值|新八方法均值|相对变化|改善方法数|
|---|---|---:|---:|---:|---:|
|mutation|MAE|0.195921|0.227082|+15.90%|1/8|
|mutation|MSE|5.697612|7.690987|+34.99%|2/8|
|mutation|unified_RelL2|7.046660|8.473850|+20.25%|4/8|
|mutation|macro_RelL2|0.341143|0.374917|+9.90%|1/8|
|mutation|pattern_accuracy|0.549603|0.464286|-15.52%|1/8|
|lhs|MAE|0.447290|0.449294|+0.45%|3/8|
|lhs|MSE|34.005737|33.133017|-2.57%|4/8|
|lhs|unified_RelL2|1.861933|1.855541|-0.34%|4/8|
|lhs|macro_RelL2|0.603064|0.620799|+2.94%|2/8|
|lhs|pattern_accuracy|0.281746|0.296627|+5.28%|7/8|

结论：mutation整体退化；LHS混合、总体接近，不构成全面升级。mutation/Uncertainty的MAE/MSE改善，其余7方法MAE均变差；LHS/MaxDist改善明显。
Random五轮训练ID及顺序相同，最终训练归一化逐项一致，模型前向/编码代码哈希相同；退化不能只归因于选样变化。自适应方法后续数据变化，比较的是整条流程，而非纯loss单因素。
新mutation/Random的4914个样本/变量项中，仅244项（4.97%）目标时间范数≤0.01，却贡献96.68%的主Relative L2；CDH1T单变量贡献84.20%。这包含真正cdh1Δ/APC-ts的错误表达，有生物意义，不能简单忽略或加大eps掩盖。剔除近零项作事后诊断时，Random其余项也从0.4060退化为0.4469，因此并非单纯指标口径错觉。
新Random和Power虽各训练200epoch，最终best分别来自56/36epoch（旧176/197）；新Random选中epoch训练loss约0.1803，末epoch约0.0901。说明checkpoint选择可能牺牲后期的其它改善，但没有存每个epoch/最后权重，尚未做同一训练按两种validation准则选best的反事实，不能把所有退化断言为选择规则导致。
训练归一化Relative与物理Relative对近零的定义不同，公式相同并不自动对齐权重。应先用固定Random数据，在同一训练中保存两种validation准则best及last，分离checkpoint影响；只按预定义合成validation选择，不以real挑epoch。此对照尚未启动。
新批PSS峰值59.49GiB，无内存保护触发；批次监控墙钟旧约8.87h、新约5.62h，非严格吞吐对照。只有seed42，不作统计显著性推广。
报告及16对明细：`/home/users/zsy/mydata/budding_yeast/analysis/sampling_protocol_comparison_20261116/report.md`、comparison.png、paired_comparison.csv；逐株/逐变量、每池top20、checkpoint选择和资源核查均有CSV/JSON。只保存标量统计，无重复轨迹。


## 2026-11-16：提高并发前的IO复测

旧批`seed42_02eebd57592f`实际从02:40:47.635运行到08:17:50.930，墙钟5:37:03；16项累计任务墙钟29:19:38，单项平均1:49:59，均退出0。峰值6并发，平均5.22并发，自有PSS峰59.493GiB，系统可用内存最低665.615GiB。逐项时长见`analysis/sampling_batch_resources_20261116/report.md`。累计任务墙钟不能当成GPU计算小时。

已将固定GPU子队列改为共用待运行队列，有空位即可领取下一项，训练/IO准入限制仍保留。BatchMemoryGuard新增持续压力保护：IO full PSI avg10>5%或memory some PSI avg10>1%持续20秒，会终止仅本批登记的进程组；原96GiB owned PSS/128GiB可用底线保留。22项CPU回归通过，包括1/2/3个加载槽、调度、压力触发、监控父进程退出时资源子进程跟随退出。

资源诊断已完成，报告位于`/home/users/zsy/mydata/budding_yeast/resource_profiles/io_concurrency_comparison_20261116/report.md`。IO2下8项完成每项10000行的2epoch训练及90k LCMD/BAIT/双成员分歧推理，全部退出0；PSS峰91.608GiB、最低可用578.126GiB、每进程GPU allocated峰1.481GiB。未覆盖辅助成员训练或完整五轮，96GiB自有内存保护仍须保留。

|重加载并发|8项全部加载就绪|加载阶段PSS峰|加载阶段IO full PSI峰|加载阶段memory PSI峰|
|---|---:|---:|---:|---:|
|2|739.23秒|54.95GiB|1.96%|0.33%|
|3|573.33秒|53.90GiB|2.37%|0.36%|

IO3加载耗时缩短22.44%，正式默认据此改为总8项／每卡2项／IO3，所有入口及锁默认同步。IO3为加载完成即主动停止的诊断，worker退出-15符合设计，不能把它的加载内存与IO2完整训练峰值直接比较。完整IO2阶段memory PSI曾短暂达到2.35%，没有持续20秒触发保护。测试不清理系统page cache，不碰GPU4--7；绝大部分读取命中缓存，不能宣称已证明冷缓存机械盘安全或完整批次提速22%。这是软保护，不是cgroup硬隔离；共享服务器突发负载仍可能使任务被安全停止。

tmux诊断已退出且探针进程已释放，未启动新协议的正式训练。上一个对话中断导致监控父进程退出的测试已标记interrupted并停止6个等待中的自有子进程，未冒称压测完成；新增parent-death信号避免资源子进程失去监控后继续驻留。

### 用户后续要求：取消自有内存限额，继续IO4/6/8

已移除生产BatchMemoryGuard的96GiB PSS停止条件、构造参数及旧环境变量作用。PSS仅记录；监测实际有效可用内存（包含可回收缓存并考虑可识别cgroup限制），128GiB可用底线及持续PSI保护保留。生产入口不再导出SAMPLING_BATCH_MAX_PSS_GIB，资源探针/独立监控也不再默认限额。模拟自有PSS200GiB但剩余400GiB时继续执行，低可用内存仍停止自有进程树，22项CPU回归通过。

本轮统一8项一开始准入，每项10000条、GPU0–3每卡2项：

|重加载并发|8项加载就绪|加载PSS峰|加载最低可用内存|
|---|---:|---:|---:|
|4|379.43秒|53.92GiB|639.49GiB|
|6|299.19秒|53.89GiB|643.48GiB|
|8|211.40秒|54.28GiB|649.19GiB|

IO8比IO4加载用时缩短44.28%，比IO6缩短29.34%，默认总8项／每卡2项／IO8，所有生产入口/锁一致更新。IO4随后完成8项2epoch训练及90k LCMD/BAIT/双模型分歧推理，全部退出0；完整资源峰90.22GiB，最低可用600.24GiB。IO6/8在加载完成后主动停止（-15），未伪称完成训练；均未触发保护。IO6启动时有短暂IO PSI 6.98%尖峰，未持续20秒，不能断言由该并发级别引起。

报告`/home/users/zsy/mydata/budding_yeast/resource_profiles/io_scaling_available_memory_20261116/report.md`与comparison.json。单次warm-cache共享服务器测试，不以44%推断完整五轮提速，也没有冷缓存结论；前一轮IO3按6→8分阶段准入，不能直接作纯单因素对照。本条更新上面的IO3/96GiB历史默认，探针已全部释放，尚未启动正式五轮重训。

### 2026-11-17：历史故障并非已确认的正常训练内存耗尽

按用户更正追查11月14日中午12点前后记录，发现旧scratch/concurrent.py遮蔽Python标准库，PyTorch导入时触发其模块顶层双子进程启动；每个resources.py子进程又导入PyTorch，形成递归链。父探针90秒超时没有清理后代，11:52启动/11:54父超时，后代错误文件最后写到13:14。14:53记录仍有682GiB可用内存但load1=162.18；这只是故障后快照，不能排除12点曾OOM，也不能据此认定递归探针是唯一原因。kernel/syslog轮转文件仍在，账号无权读取，缺少内核确证。

历史脚本按相同字节/哈希保全并隔离为concurrent_probe_20261114.py.disabled，避免再次遮蔽；没有重放风险链或修改生产训练/采样。当前正常8项的资源测试无法代表历史失控后代数量。证据、时间线及管理员只读日志提取命令位于`/home/users/zsy/mydata/budding_yeast/analysis/incident_20261114_1200/report.md`。

### 2026-11-17：扩大并发，等待共享IO恢复后自动重测

用户要求以约100GiB有效可用内存为余量，生产入口、资源锁、BatchMemoryGuard和探针默认已统一100GiB，不设自有PSS限额。当前默认队列16项，原计划依次测12/16 worker（每卡3/4项、IO12/16），每项10000条、2epoch及90k评分，不分配无用张量填满RAM。只用GPU0–3，nice10/ionice7；补充清理进程退出竞态，24项CPU回归通过（含有效可用110GiB时继续运行）。

首轮12项/IO12在151.4秒因global IO full PSI持续超5%达20秒而自动停止，未进入训练。峰PSS31.58GiB、最低可用676.91GiB，memory PSI为0；自有物理读取累计约1.28MiB。不能把全局IO压力归因于本批读取：12个worker全退出后IO PSI继续升至19–21%，5秒设备样本中/home/users底层设备约99.8%忙、/data无明显IO，同期当前用户无超过0.05MiB/s的活跃物理IO进程。未定位具体来源，未终止其他用户任务。

重测改为12/16项、IO均8。tmux `sampling_concurrency_12_16_20261117`先等待global IO PSI<1%、memory PSI<0.5%、有效可用>=120GiB持续30秒后才启动，每档运行仍保留100GiB和持续压力保护；首档失败则取消升级，等待6小时不恢复则退出留痕。当前没有新增GPU测试进程，正式默认并发保留8/每卡2/IO8，12/16尚未通过。初始父准入曾沿用128GiB而worker/停止线100GiB，测试始终>676GiB；重测父子已统一100GiB。

输出`/home/users/zsy/mydata/budding_yeast/resource_profiles/higher_concurrency_12_16_20261117/`，`retry_status.json`为实时状态，`retry_launcher.log`为日志，`host_cooldown.jsonl`记录背景压力，`initial_attempt_summary.json`保留首次中止数据。源池/正式结果未覆盖，未启动正式五轮重训。不能据此称12项安全或不安全，待稳定宿主环境下完成测试再定。


#### 01:33重测中止（覆盖上述等待状态）

12项/IO8于01:26:33自动启动，加载294.29秒，主模型2epoch训练和独立评测全部返回；12项均进入90k预测特征提取，391.34秒时global IO PSI持续超限触发停止，最终23.64%。内存峰130.85GiB、最低可用576.43GiB，停止时memory PSI=0。自有物理读取总计76.68MiB，350秒后该计数不增；尚缺盘级及进程写入同步记录，不能认定为本批大量读盘或保护误判。全局PSI不提供本批责任归因，参考https://docs.kernel.org/accounting/psi.html。

退出码均-15，来自保护；完整候选评分未完成，不报告资源吞吐排名。16项自动取消，tmux已结束，没有新GPU测试进程。正式并发仍8项、每卡2、IO8、有效可用余量100GiB。后续需要同步磁盘/自有读写采样及单独的候选推理并发对照，不能只提高阈值重试。最新报告同目录report.md与retry_attempt_summary.json。


### 2026-11-17：评分限4与设备联合IO保护对照（进行中）

新增独立评分槽与进程读写/同步diskstats采样，34项CPU检查通过（跨进程评分槽、盘字段单位、拥塞判定、放宽诊断IO准入时保留内存/CPU保护）。首个strict试验在42.8秒、仅10个worker时中止，尚未到评分；PSS峰8.23GiB、最低可用700.16GiB、global PSI14.06%，数据盘忙峰47.72%（分区51.07%）、队列4.18，自有物理读292.71MiB/写9.21MiB。不能评价评分4效果；表明仅global PSI>5%终止的规则可能过于敏感，设备数据不支持当时已经严重拥塞。

新试验保留12→16项、IO8、scoring4、100GiB余量及所有模型/数据规则，显式`--io-policy disk_correlated`。该诊断中global IO PSI>5%须与设备忙>=90%且平均队列>=16或请求await>=250ms同时持续60秒才停止。设备样本缺失回退strict20秒；memory PSI>1%20秒和可用<100GiB停止仍保留。这些是待测工程阈值，不是文献规定或绝对安全保证。诊断IO准入设warn避免正常读盘等待反复终止，仍限制IO8/评分4和CPU/内存准入；正式BatchMemoryGuard及生产默认strict没有改，也未给正式runner包裹评分槽。

新12项于01:58启动。初期样本global IO PSI14.44%，数据盘忙36.56%/队列6.22/await3.24ms，可用692.95GiB，未出现严重设备拥塞，测试继续推进。需要完成全套候选评分后再判断，不能将当前加载推进当成完整通过。tmux `sampling_scoring_12_16_20261117`；`/home/users/zsy/mydata/budding_yeast/resource_profiles/scoring_limited_12_16_20261117/status.json`实时状态，strict_报告保存早先试验，raw样本与最后comparison/report自动生成。只用GPU0–3，尚未开始正式五轮重训。


## 2026-11-17：资源对照通过，采用16并发正式运行

12项与16项诊断均完整完成并全部退出0；12项14分21.60秒，PSS峰131.86GiB、最低可用571.57GiB；16项17分21.31秒，PSS峰178.76GiB、最低可用522.32GiB。每项10000训练样本、2epoch、90k候选特征及LCMD/BAIT/双模型推理；未覆盖完整五轮、辅助成员训练或收敛，不能据此保证所有负载绝对安全。16项memory_policy=available，12项strict，不能视作严格单因素吞吐实验。

用户授权测试后正式启动；离线batch默认总16／每卡4／重加载8／候选评分4，GPU仅0–3。按系统有效可用内存100GiB保底，取消自有内存配额；memory PSI仅预警。IO保护采用全局PSI>5%并与设备busy>=90%且queue>=16或await>=250ms持续60秒联合判定，设备数据缺失回退strict20秒。保留CPU准入、进程隔离与只停止本批子进程的规则。候选评分等待单独记入selection.json，不改变选样公式。

正式协议seed42_a866aef5d988，输出experiments/budding_sampling/seed42_a866aef5d988/，与旧目录分开；两池各8项FNO，seed42、五轮2000→4000→6000→8000→10000，100/100/100/100/200epochs，逐轮训练集refit，normalized validation-best、eps0.01 smooth/no cap、保留last不变。W&B仅本地offline。资源原始报告：/home/users/zsy/mydata/budding_yeast/resource_profiles/scoring_limited_12_16_20261117/report.md。


2026-11-17 02:51正式批次已在tmux `budding_sampling_formal_20261117`启动，输出 `/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/seed42_a866aef5d988/`；批日志 `batch/launcher.log`，单项 `<pool>/<method>/seed_42/train.log`。45项资源CPU回归和3项小流程回归通过，dry-run16项及参考SHA核对通过。

02:54复核：16个正式run进程均存活且WANDB_MODE=offline、GPU仅0–3；mutation/VeSSAL已完成首轮至少16epoch，loss有限，其余正在加载/预处理。启动核查记录batch/startup_verification.json。尚不能声称整批五轮完成或所有后续评分均通过。


## 2026-11-17 完整批次结果：normalized-best与修正池

协议seed42_a866aef5d988，FNO、两池8baseline共16项全部完成/退出0，80轮、10080幅轮次real图；预算2000→4000→6000→8000→10000，epochs100/100/100/100/200、batch128、lr5e-4、smooth eps0.01无cap，每轮train-only refit，以同池val_loss_synth选best并保留last。逐轮ID与holdout无交集、同池初始有序2000一致、best与history最低值一致。02:51:08至06:08:11，墙钟3:17:04，任务累计46:28:51；PSS峰151.96GiB，最低可用538.82GiB，16并发未触发守护。

mutation真实评测：Power MAE0.15030/MSE2.25805/Relative1.05698/Corr0.90974；Random0.17746/3.44796/12.36778/0.89962；LCMD0.16444/2.65932/12.88390/0.89108；Uncertainty0.21613/18.05836/1.26566/0.90313。Power相对Random的MAE/MSE下降15.30%/34.51%；LCMD六分组准确率最高68.25%，Power63.49%、Random65.87%。LHS的Power绝对误差最好0.41464/24.90815，但VeSSAL的Relative最低1.60615，不能单指标宣布全胜。

关键诊断：CPU重推mutation Random/Power/LCMD与LHS Random的126株，总Relative与GPU结果相差<0.1%。Random/LCMD总Relative约88.90%/88.21%来自CDH1T通道；#37真值0，Random预测约0.97、LCMD约1、Power接近0，说明零通道关闭确有失败，也因小分母被放大。25株real的该通道近零；训练中Random/LCMD/Power/BAIT对应零初值样本1229/1364/1918/1863，不能仅用数量解释差距。Uncertainty #124贡献整体MSE78.84%。末轮后20epoch验证loss接近平台，best规则核对正确；尚不证明改变训练预算无收益。

与旧02eebd批次，mutation Power MAE0.27613→0.15030、MSE10.21753→2.25805；新配置同时改数值标签与checkpoint规则/选择路径，不能单独归因。当前结果仅seed42且只含baseline，没有ours成绩；应固定协议，以合成已查询反馈和系统已知关闭/初值/ODE规律设计采集，不使用real结果打分。

完整报告：/home/users/zsy/mydata/budding_yeast/analysis/sampling_results_20261117/report.md，含全部16项、Top20坏例、逐轮图、收敛窗口、逐通道/初值覆盖/划分审计与旧批次对照。复用分析入口scripts/analysis/analyze_complete_sampling_batch.py（argparse，CPU可重推126株，不加载整池轨迹、不改训练）。详见报告的跨池OOD失败解释；LHS额外背景扰动与mutation不同，不能把生成范式差距全归因选样算法。


### 2026-11-17多seed前补充核查

总墙钟3:17:04，主模型80轮+辅助16轮完成；所有辅助best也与归一化validation最低值核对一致。主模型mutation前四轮best97～100，末轮183～200；LHS首轮八方法均54，末轮123～164，第二轮还有64/76。确有几十epoch的best，但不是早前物理指标选best的口径问题。best_epochs.csv已保存全部80行。

新增诊断：VeSSAL随机补齐在mutation为299/8000=3.74%，LHS为5274/8000=65.93%（逐次1347/1333/1271/1323）；作者query支持fill_random，不据此认定bug，但论文报告需注明penultimate适配及补齐比例，多seed前建议对账该分支。未擅自修改算法或重训。详见docs/history/SAMPLING_BASELINE_ROADMAP_2026-11-17.md及vessal_fill_summary.json。

补充复现建议：已读10篇本地PDF与14个参考仓库，优先MaxDet-LL、k-means++-LL及KMeans/Herding；现有Physics-PRE/Lev无需重复实现；监督筛选方向优先GradMatch。PEAKS需回归推导，HGGS/OED/OGAS保持生成流程，CREST/ERACS保持动态重选，不假装固定池五轮原版。完整清单见[SAMPLING_BASELINE_ROADMAP_2026-11-17.md](SAMPLING_BASELINE_ROADMAP_2026-11-17.md)。本轮没有实现新baseline或启动多seed。
