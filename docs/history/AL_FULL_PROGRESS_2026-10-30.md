# 长程FNO主动学习：10月30日上午进度与结果

北京时间2026-10-30约10:48检查：14项中2项完整完成、6项仍运行、6项异常退出。本轮只读取结果并生成分析文件，没有重启任务、改动训练代码或修改正式数据。

分析目录：`/home/users/zsy/mydata/budding_yeast/analysis/al_full_progress_20261030_1050/`。其中REPORT.md、status.csv、round_metrics.csv、per_mutant.csv、convergence.csv、query_design_summary.csv、completed_final_mutant_comparison.csv、random_final_top20_all_methods.csv保留完整统计，准确快照时间以snapshot.json为准。复用入口：`scripts/analysis/snapshot_al_full_batch.py --manifest <启动manifest> --health <启动health> --output <分析目录>`。

## 进度与失败

|状态|方法|
|---|---|
|完成四轮|Random、MaxDist|
|仍运行|ours_response_marginal、ours_event、ours_response_feedback、ours_error、ours_interaction、ours_coverage|
|异常退出|Uncertainty、Power、LCMD、BAIT、首版ours_response、ours_invariance|

检查时marginal在第二次追加ODE、feedback在第三次追加ODE；event第二轮绘图；error/coverage第三轮辅助ensemble；interaction第三轮约95/100 epoch。辅助ensemble每10epoch输出变化率，几分钟无新行不直接判为阻塞。全部任务已有启动记录，没有未启动队列。

Uncertainty、Power在07:44第三次追加触发父进程35秒无结果保护，退出1。这是整个进程池无进展，不能直接等同于单条ODE超时；公共资源压力、worker死亡、进程池阻塞的具体链路尚未确定。有界提交保护没有消除全部故障。

LCMD（05:54）、BAIT（07:21）、ours_response及ours_invariance（07:42）退出137，批日志为Killed，无CUDA OOM traceback。当前tmux所属session-1699.scope记录累计oom_kill=11，内存峰值750576140288字节，约699GiB；服务器物理内存约755GiB。主机RAM压力是重要嫌疑，但内核日志权限不足，累计计数可能包含更早任务，不能逐个证明SIGKILL和watchdog的完整因果链。显存足够不保证八项实验能持续并行。检查时RAM可用约360GiB，数据盘可用约102GiB，未观察到磁盘写满错误。没有删除实验或操作其他用户任务。

## 已完成最终结果

均为seed42、70k合成训练、100/100/100/200 epochs、FNO192/4/24、batch512、最低LR1e-5。RelL2/Corr为JSON整体聚合口径，非逐突变体算术平均。

|方法|Real MAE|Real MSE|Real RelL2|Real Corr|墙钟用时|
|---|---:|---:|---:|---:|---:|
|Random|0.098193|1.374911|0.169334|0.963811|6:10:45|
|MaxDist|0.097129|1.519672|0.134503|0.978758|7:00:15|

用时读取END experiment的elapsed_s，包含加载、训练、采集、ODE、评测和保存，不含输入预生成或排队；不同并行负载不能作为纯算法速度比较。

MaxDist相对Random整体RelL2降低约20.6%，MAE仅降低1.1%，MSE增加10.5%。126例中，MaxDist只有29例逐样本RelL2更低、97例更高；中位数Random=0.0744、MaxDist=0.1016。收益集中于部分难例，普通样本存在退步，不能说普遍变好。

|real编号|Random逐样本RelL2|MaxDist逐样本RelL2|
|---|---:|---:|
|15|6.9159|1.0310|
|22|0.5130|0.2551|
|42|0.5140|0.3058|
|76|0.4738|0.1308|
|64|0.0774|0.7050|
|124|0.1016|0.1777|

#15仍未解决，MaxDist也超过1；#64、#124有反向权衡。本轮只核对已有逐样本误差，没有重新推理或作新的轨迹机制归因。

对比旧50/50/50/100同范围协议：Random MAE降低19.1%、MSE降低51.0%，RelL2增加3.0%；MaxDist MAE降低20.7%、MSE降低36.1%、RelL2降低13.3%。不能全部归因于epoch，最低LR、round seed及后续采集数据也改变；需固定相同dataset才能隔离训练预算效应。

## ours阶段比较

仅比较已完成40k训练评测的方法，均100epochs，不混入未完成70k轮次。

|方法|Real MAE|Real RelL2|
|---|---:|---:|
|Power|0.144223|0.178220|
|Random|0.146244|0.193238|
|LCMD|0.147925|0.176651|
|MaxDist|0.149890|0.178117|
|ours_response_feedback|0.157299|0.195023|
|ours_invariance|0.157859|0.217001|
|首版ours_response|0.159217|0.206766|
|ours_coverage|0.176696|0.208515|
|ours_error|0.183790|0.237180|

feedback比首版ours略好，仍未优于Random/Power。20k预算下interaction MAE/RelL2=0.193892/0.220821，Random=0.177596/0.220913，尚无全面优势。marginal/event/interaction最终结果未齐，不能据中途结果宣布失效或胜出。

选样审计确认新机制生效：marginal首次追加1000单点、8000响应端点、1000随机；feedback第一次相同，第二次已经拟合前1万合成查询的响应误差。反馈留出MSE=0.02681，常数预测0.04485，但它是已查询分布的随机留出，非无偏pool评估，也不直接代表real提升。error/coverage反馈拟合同样启用。

invariance三次追加的真正对照端点成本为8000、338、33，其余大量point_fallback。限定TEM1的合法对照很快耗尽，后续不能称为大量机制对照；这是选样审计支持的可扩展性限制，尚非性能差异的唯一原因。

## loss变化率

|完成实验|末10epoch训练loss下降趋势|验证MSE下降趋势|连续平台检查|
|---|---:|---:|---:|
|Random|0.0758%/epoch|0.0454%/epoch|1/3|
|MaxDist|0.0520%/epoch|0.0025%/epoch|1/3|

相比此前约0.2%/epoch的训练变化率，末段下降已变慢。最后一次满足0.1%阈值，但未连续三次，因此监控仍为still_changing。更接近平台不等于证明严格收敛，也不能据字符串认定必须再次翻倍epoch；小LR也会压低变化率。本轮未改变预算或自动早停。

全部仅一个seed，没有跨seed稳定性或显著性结论；real指标仅事后报告，没有用于在线选择、停止训练或挑checkpoint。

## 短程/长程与收敛补查

逐项重读两种协议的最终round_03/training_history.csv及rounds_summary.json，新增分析文件short_full_convergence_comparison.csv。短程为50/50/50/100，长程100/100/100/200，均70k最终训练量。

|方法/协议|Real MAE|Real MSE|Real RelL2|Real Corr|
|---|---:|---:|---:|---:|
|Random短程|0.121305|2.807405|0.164360|0.968588|
|Random长程|0.098193|1.374911|0.169334|0.963811|
|MaxDist短程|0.122449|2.376790|0.155186|0.971815|
|MaxDist长程|0.097129|1.519672|0.134503|0.978758|

末10epoch变化率统一对11个loss值的log做线性回归，转成每epoch下降百分比；不是逐步变化绝对值均值。

|方法/协议|训练loss下降趋势(%/epoch)|验证MSE下降趋势(%/epoch)|训练loss末10epoch净降幅|
|---|---:|---:|---:|
|Random短程|0.2266|0.0524|2.258%|
|Random长程|0.0758|0.0454|0.799%|
|MaxDist短程|0.2187|0.1300|2.247%|
|MaxDist长程|0.0520|0.0025|0.541%|

当前规则为train/val趋势绝对值都低于0.1%/epoch且连续3次10epoch检查。长程两项在180/190epoch仍因训练loss下降超过阈值未通过，直到200epoch首次通过，均只有1/3。结论应表述为“接近当前学习率计划下的平台，尚未按当前规则确认稳定收敛”，不应把still_changing当成严格证明未收敛。

MaxDist的验证MSE最佳在178epoch，最终值仅比最佳高约0.463%；Random最佳在196epoch，最终值比最佳高约0.298%。结合低验证趋势，没有证据据此立即再次翻倍epoch。长短程都是各轮冷启动，长程并非在旧100epoch checkpoint上直接续训；协议和采集数据差异仍限制纯epoch归因。
