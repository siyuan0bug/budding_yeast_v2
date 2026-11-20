# 长程FNO主动学习：10月30日16:56阶段结果

本次14项已有7项完整完成，ours_event仍在最终轮196/200 epoch；此前异常的6项未重启。完成定义为四轮评测保存且程序退出0，不把训练结束但评测未完成的实验列作完成。该状态为时间快照，不是自动更新页面。

本轮仅分析已有结果，没有更改训练、数据、采集配置或启动新任务。分析目录：`/home/users/zsy/mydata/budding_yeast/analysis/al_full_progress_20261030_1655/`。复用`scripts/analysis/snapshot_al_full_batch.py`，核对49个完成轮次的126株real指标、历史长度和checkpoint，全部通过；其中7项是最终70k完整结果，其余是未完成实验的中间轮次。

## 最终70k结果

统一seed42、FNO width192/blocks4/modes24、batch512，四轮100/100/100/200 epochs、warmup5/5/5/10、LR1e-3至1e-5、逐轮冷启动和累计合成训练集重归一化。需ensemble的方法使用两个成员。表中RelL2、Corr采用已有JSON整体汇总口径，不能当成逐mutant算术平均。

|方法|Real MAE|Real MSE|Real RelL2|Real Corr|用时约|
|---|---:|---:|---:|---:|---:|
|MaxDist|0.097129|1.519672|0.134503|0.978758|7:00|
|Random|0.098193|1.374911|0.169334|0.963811|6:11|
|ours_interaction|0.103874|2.068136|0.155149|0.970675|6:46|
|ours_response_feedback|0.105677|2.449735|0.137153|0.977592|7:38|
|ours_response_marginal|0.109749|2.445821|0.145404|0.974720|7:30|
|ours_coverage|0.115056|2.100165|0.168578|0.965781|6:57|
|ours_error|0.133048|2.510365|0.191238|0.957351|6:52|

用时来自END experiment的elapsed_s，包含程序加载、四轮训练、必要的辅助ensemble、选样、ODE、绘图和保存，不含共享输入预生成及GPU排队；不同并行负载，不能当作纯采集计算成本。精确秒数见status.csv。

已完成方法中，MaxDist的MAE、整体RelL2、Corr最好，Random的MSE最好。没有ours在这四项指标上全面超过这两个baseline。Power/Uncertainty/LCMD/BAIT缺少本批最终结果，旧短程指标不得混入该表证明优劣。

interaction在ours中MAE最低，但比Random仍高5.8%，MSE高50.4%。feedback相对marginal的MAE降低3.7%、整体RelL2降低5.7%，MSE基本相同；这是单seed的一组消融观察，支持反馈有一定作用的假设，但不能认为跨seed稳定优越。error当前最弱；coverage没有显示整体优势。

## 难例改善与代价

逐样本RelL2与整体RelL2不是同一个统计量。

|real编号|Random|MaxDist|interaction|feedback|marginal|
|---|---:|---:|---:|---:|---:|
|15|6.916|1.031|0.145|0.139|0.202|
|20|0.482|0.421|0.558|0.495|0.315|
|22|0.513|0.255|0.319|0.213|0.253|
|42|0.514|0.306|0.349|0.325|0.325|
|64|0.077|0.705|0.095|0.185|0.314|
|76|0.474|0.131|0.412|0.157|0.664|
|124|0.102|0.178|0.285|0.155|0.117|

多个ours确实缓解#15，但error仍为6.878。interaction还改善#22/#42，相对Random对#20/#64仍退步。feedback对#22/#76较好；不能把针对几个难例的收益推广到全部126株。相比Random，interaction有43/126株MAE更低、40/126株逐样本RelL2更低；feedback为31/126和30/126。

#124的绝对误差特别关键：其MSE分别为Random9.984、MaxDist20.894、interaction114.234、feedback95.781、marginal49.924、coverage107.720、error34.912。不能只看该例RelL2相对较小就忽略绝对误差。

作为事后贡献诊断，单独拿掉#124后剩余125株：Random MAE/MSE为0.095939/1.306040，interaction为0.092937/1.170811；保留完整126株时interaction反而更差。说明interaction相对Random的全体MAE/MSE劣势可被#124的额外误差解释，其他样本合计有净收益，但不代表大多数样本改善，也不代表#124是唯一问题。不能真的移除benchmark样本后宣称方法获胜。feedback拿掉#124后仍为MAE0.096306/MSE1.703087，不能套用相同结论。

以上为指标贡献分解，不是参数覆盖或模型机制的因果证明；本轮未重新推理、重算ODE或查看全量轨迹来确认#124的具体错误通道。完整贡献见mutant124_metric_contribution.csv；所有逐样本指标见per_mutant.csv。

## 收敛与运行限制

7项最终轮末10epoch训练loss下降趋势为0.052%–0.090%/epoch，验证趋势绝对值0.002%–0.063%/epoch，均小于0.1%；但各自convergence.json仍标still_changing，尚未连续满足三次检查。更接近当前LR计划下的平台，不等于严格收敛；也没有证据据此直接再次翻倍epoch。不能跨方法直接比较归一化loss的绝对值，因为训练数据和拟合尺度不同。

仍失败：Uncertainty、Power第三次追加触发进程池watchdog；LCMD、BAIT、首版ours_response、ours_invariance退出137。原因边界见上午审计，本轮未修复或重跑。数据盘检查约剩41GiB，未删除文件或改变保存规则。

当前支持继续重点研究interaction及response_feedback，但没有最终赢家结论。应保留统一完整126株评测，补齐失败baseline并做多seed，才能判断方法收益是否稳定。real仅用于本轮事后分析，没有用于在线选样或checkpoint选择。
