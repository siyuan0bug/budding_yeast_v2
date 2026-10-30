# gene_ranges 未完成实验重跑与已完成用时

2026-10-29 15:33:49（Asia/Shanghai）按用户要求重启三个未完成实验。

## 重跑范围

|方法|旧状态|新安排|
|---|---|---|
|Random|完成10k/20k/40k评测，第三次追加时进程池死锁|GPU0|
|Uncertainty|完成10k/20k/40k评测，第三次追加时进程池死锁|GPU1|
|ours_response|完成10k评测并保存20k累计数据；写JSON失败|GPU0，与Random同卡并行|

仅终止经PID、UID、完整train.py路径和exp_dir核对的两个阻塞训练进程及其子进程。
没有终止tmux会话、其他用户进程、其他GPU任务或已完成结果。

使用 neuralop 解释器 `/home/users/zsy/.conda/envs/neuralop/bin/python`，通过原 `run_al_budding.sh` 的 EXPERIMENTS_TEXT 过滤重跑三项。
后台tmux为 `al_budding_retry_20261029`。GPU0最多2项并行，GPU1一项；不使用正在被其他任务使用的GPU2/3。

复用既有 million pool 和初始数据，不重新生成、不重新划分：

- `/home/users/zsy/mydata/budding_yeast/budding_pool_1000000_seed42_gene_ranges.npz`
- `/home/users/zsy/mydata/budding_yeast/budding_initial_train10000_seed42_gene_ranges.npz`

从初始10k重新执行完整四轮，非断点恢复。FNO width192、modes24、4blocks、batch512、LR1e-3；epoch50/50/50/100，warmup4/4/4/8；原归一化、候选预算和seed42保持。
W&B强制offline，默认每实验4个ODE worker；已包含JSON/池大小、float32推理、有界进程池和时间日志修复。

## 结果与旧失败记录

新实验回到原标准路径：
`/home/users/zsy/mydata/budding_yeast/experiments/al_budding/fno/hybrid_mse_rel_<strategy>/w192_n4_m24_b512_e50-100_lr1e-3_wm4-8_al4_refit_gene_ranges/seed_42/`

三个旧未完成目录已整体移动保留到：
`/home/users/zsy/mydata/budding_yeast/experiments/al_budding/failed_runs/gene_ranges_20261029_153315/<strategy>/seed_42/`

每项有 RELOCATION.json；旧config/summary内部记录的绝对路径仍是当时原路径，查旧文件时需按此映射定位。没有改写旧实验内容来冒充新结果。
原40k/20k快照、checkpoint、错误日志和W&B离线数据仍保留。四项成功目录保持原位。

调度日志、停止进程清单、目录映射、输入文件身份、代码hash、重跑脚本副本和用时CSV：
`/home/users/zsy/mydata/budding_yeast/analysis/al_rerun_2026-10-29/`。

## 已完成的新范围实验用时

|方法|程序开始|正常结束|总墙钟时间（约）|
|---|---|---|---|
|MaxDist|02:30:50|06:06:53.806|3小时36分04秒|
|LCMD|02:30:50|06:10:04.496|3小时39分14秒|
|BAIT|02:30:50|06:51:08.712|4小时20分19秒|
|Power|02:30:50|06:52:56.258|4小时22分06秒|

均为2026-10-29，北京时间；四轮评测齐全且批处理退出码0。
开始使用train.log文件创建时间，结束使用本地W&B debug.log的exitcode0时间，因此是近似程序墙钟，包含初始化、训练、选样、ODE、评测绘图和保存，不含共享pool/初始轨迹预生成或GPU排队。
Power额外训练ensemble成员，方法间用时差异不能全部解释成选样算法耗时。

来源：`scripts/analysis/summarize_al_runtime.py`，重跑前快照位于上述分析目录的 `runtimes_before/experiment_runtimes.csv` 和 `.json`。

float64适合ODE与参数/初值的高精度存档，约15–16位有效数字；float32约6–7位、每元素4字节，是float64的一半，适合模型计算。
项目在存档边界使用float64、模型输入边界统一float32，训练仍使用既定混合精度；这次重跑不更改精度协议。

启动验收：截至15:42，三项均已进入第一轮并连续完成多个epoch，最新train_loss为有限值；时间前缀、离线W&B、原超参数及GPU映射核对通过。实时详情见 training_started_verified.json。完整四轮仍在后台运行。
