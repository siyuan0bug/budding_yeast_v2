> 后续排查已补充满管道证据及受控死锁复现，早期“具体调用未完全确认”的状态已更新；同时完成train.log时间日志。[后续修复记录](AL_LOGGING_AND_PROCESS_POOL_FIX_2026-10-29.md)。

# 新采样范围与主动学习实验核对（2026-10-29）

本次读取已有实验，不启动正式训练，不终止现有进程。统计快照及 CSV、图表：
`/home/users/zsy/mydata/budding_yeast/analysis/al_range_comparison_2026-10-29/`。
可复用脚本：`scripts/analysis/dataset_diagnosis/compare_al_range_experiments.py`。

## 当前结论

新 gene_ranges 已缓解大值样本问题，并在相同训练协议、相同预算的 Random 对照中改善总体 real 指标。
不能据此声称所有 badcase 已解决，也不能将收益完全归因于 MC 上限：两批池的合法模式、有限容量配额、具体选点和归一化统计也一起变化。
只有 seed42，尚无多 seed 显著性结论。

新批次 Power、LCMD、MaxDist、BAIT 已完成四轮且批处理退出码为 0。
Random、Uncertainty 完成前三轮评测，第三次追加没有完成；检查时日志分别自 03:55、04:59 后未推进，主进程及部分 worker 在 pipe_write/futex 等待，不能视作正常训练。
ours_response 完成首轮评测及首次 10,000 条追加仿真，保存审计 JSON 时失败；20,000 条累计训练集已保存在 round_01/dataset.npz，但没有第二轮训练结果。

## 同协议 Random：新范围是否有效

两组均 FNO width192/modes24/4 blocks，batch512，LR1e-3；50/50/50/100 epoch，warmup4/4/4/8；每轮冷启动并只用累计合成训练数据拟合归一化。
真实测试均为同一126株；不把两批不同合成测试集的指标变化当作同测试集上的精度提升。

|训练量|旧 balanced MAE|新 gene_ranges MAE|旧/新总体 Rel-L2|旧/新 MSE|
|---:|---:|---:|---:|---:|
|10,000|0.405350|0.330454|0.359381 / 0.327653|22.789558 / 13.602991|
|20,000|0.274958|0.253151|0.284868 / 0.265002|11.798286 / 9.145644|
|40,000|0.213509|0.180065|0.233944 / 0.221874|6.099851 / 3.300605|

40k 时 MAE 降约15.66%，MSE 降约45.89%，总体 Rel-L2 降约5.16%。
逐株 MAE：88株改善、38株变差；逐株 Rel-L2：96株改善、30株变差。
首轮所有新策略的指标完全相同，符合共享初始训练、验证、测试集的设计；此时 AL 尚未产生影响，不能算七次独立重复。

“大值”仅作为诊断：任意非MASS变量峰值 > max(2×该变量real峰值, 2)，或 MASS>10；本次不据此删除数据。

|Random训练量|旧大值占比|新大值占比|
|---:|---:|---:|
|10,000|19.55%|11.79%|
|20,000|17.84%|11.13%|
|40,000|16.71%|10.84%|

新范围不是消灭所有大值，而是减少远离目标任务的过宽强度覆盖。真实的突变组合仍可产生高浓度和强阻滞。

## 新批次 AL 方法比较

以下均为最终70k训练、100epoch的126株real结果；未完成方法不列入最终排行榜。

|方法|MAE↓|MSE↓|总体 Rel-L2↓|Corr↑|训练集大值占比|
|---|---:|---:|---:|---:|---:|
|Power|**0.105360**|**1.619516**|0.148046|0.973145|16.03%|
|LCMD|0.112627|2.222827|**0.146436**|**0.975239**|17.80%|
|MaxDist|0.122449|2.376790|0.155186|0.971815|23.97%|
|BAIT|0.138255|2.145770|0.180925|0.961344|25.32%|

Power 和 LCMD 是目前更好的两种选择，评价侧重不同。BAIT 并非因为被称为基线就必然有效。
新 Random 的70k尚未评测，不能宣布某方法已胜过同分布最终 Random。

相同40k下：Random MAE/Rel-L2=0.180065/0.221874；Uncertainty=0.226397/0.243068。
Uncertainty 数据的大值占比27.13%，高于 Random 的10.84%；其 CLB2 峰值P95约54.57，高于Random约15.21。
这支持“当前 Top-K 分歧选择集中于高幅值区域”的诊断，但没有完成删除大值后的重训，不能声称大值是唯一因果原因。
Power 保留加权随机探索，实测比纯Top-K更稳；LCMD覆盖预测输出空间，不能解释成已覆盖所有生物条件背景。

新增样本明显偏向 L3：LCMD/MaxDist/BAIT 各6万追加中，L3分别59,031/59,412/59,461；Power为55,883。
候选池的模块集合均衡，并不要求 AL 最终选出的训练集继续均衡；也不能把 L3 总数当作 badcase 的局部有效支撑数。

## 至少20个固定 badcase

从旧 balanced Random 最终70k的逐株 MAE和Rel-L2百分位平均排名，固定选出：
76、20、22、54、124、15、41、16、46、31、53、66、59、52、33、57、23、42、60、24。
原始名称及所有轮次逐株结果分别见 reference_top20.csv、fixed_top20_errors.csv；
这种按旧结果选出的集合带有选择偏差，跨批改善应结合全部126株一起看。

|样本|Power最终 Rel-L2|LCMD|MaxDist|BAIT|
|---|---:|---:|---:|---:|
|#15|0.4138|0.9252|0.3515|0.3056|
|#22|0.5047|0.2885|0.2611|0.2725|
|#76|0.1745|0.2052|0.2012|0.7292|
|#124|0.1555|0.2091|0.1855|0.2612|

固定20株的平均逐株 Rel-L2：旧 balanced Random最终为0.4601；新Power/LCMD/MaxDist/BAIT为0.2922/0.3091/0.3288/0.3934。
这里同时改变了采样域和AL策略，不能当作纯范围消融。
#15 在新Random40k仍为6.6553（旧为7.4925），#22新Random40k从0.4860变差至0.5215；缩窄范围并没有自动学会条件突变响应。
#42仍是既定L≤3采样域之外的四模块组合，不能承诺仅靠当前候选池彻底补齐。

总体 Rel-L2 是先将所有样本沿同一变量拼接后计算，再对变量平均；逐株 Rel-L2 是每株独立计算。
两者不能直接取均值互换。MAE/MSE可以按等长样本平均；详见 utils/metrics.py。

## ours_response 报错与修复

报错位置：write_response_audit → json.dump，TypeError: Object of type int64 is not JSON serializable。
CandidatePool.keep 用 NumPy mask.sum()得到池大小；对照端点是训练锚点时，training_row=i-pool_size也变成np.int64。
此前测试覆盖了选样和审计，但没有将“裁剪后NumPy池大小＋实际训练锚点＋写审计”连起来，这次补上该回归。

修复：training_row及timings中的整数显式转换为Python int；JSON先完整序列化，再写临时文件并原子替换，避免写出半截报告。
10项response测试通过，包括真实选样到审计写出的组合及序列化失败时保留旧文件。
失败实验已选出10k、构建3,709,878条合法对照边；不是未形成对照，也不是新的float64错误。
原 response_pairs.json 只有半截，无法据此恢复完整动作清单，不修改它来冒充成功。没有后续训练，不能评价ours效果。

## 另外发现的并行执行风险

bounded_ordered_map 原先先提交全部任务，再开始消费和35秒父级监控。
第三次追加一次提交30,000任务，存在Python3.10进程池管道/唤醒队列阻塞风险；父级监控在提交期间尚未生效。
已改为最多2×worker数量的滑动窗口，按原输入顺序返回，仍保留求解和父级超时。
新测试覆盖30,000个候选的有界提交、提前达到配额关闭、真实spawn、异常和原有watchdog。
独立10,000个spawn任务探针完成。旧线上进程的Python栈因ptrace限制无法读取，确切阻塞调用尚未证实；
因此这是一项已验证的排队保护修复，不能据此声称已恢复两个停滞进程。现存进程不会自动载入修复。

## 数值边界与下一步

TEM1>Tem1T 的可疑轨迹仍存在：新Random40k有2条；新Power/LCMD/MaxDist/BAIT最终70k分别3/7/10/20条；
新Uncertainty40k有10条；ours已保存20k为0条。CDC15/ESP1未发现同类超总量。
这是有界蛋白总量约束，与“超过real最大值两倍”的经验阈值不同，不能混为合法大值问题。
跨新策略去重共26个可疑候选，用 rtol=1e-7、atol=1e-9、max_step=0.25 重算：25个成功且全部回到总量以内，1个在现有超时保护下失败。
这确认至少25个原始超总量峰值属于数值伪影；不能用这种TEM1峰值推断其GAL分布应该收窄。
逐候选严格重算见 invariant_strict_rechecks.json；本次不修改生产ODE、超参数、采样规则或现有数据。

正式结论应在Random/Uncertainty恢复完成后，用相同最终预算比较，并至少增加独立seed。
ours需用修复后代码重启或另行实现断点恢复；原20k快照仍保留，直接重跑批脚本会拒绝已有实验目录。
旧v2批次的batch1024、100/100/100/200epoch与本次不同，只供历史背景，不能作为范围的严格对照。

复现：
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle \
/home/users/zsy/.conda/envs/neuralop/bin/python \
scripts/analysis/dataset_diagnosis/compare_al_range_experiments.py \
  --output /home/users/zsy/mydata/budding_yeast/analysis/al_range_comparison_2026-10-29 \
  --scan-trajectories --strict-recheck-invariants
```

验证记录：response测试10项通过；并行窗口及原watchdog共6项通过；dtype/package/parallel组合57项通过、1项旧测试fixture因缺少新的范围指纹而失败，补齐fixture元数据后该项真实spawn集成回归通过。未放宽生产缓存校验。
