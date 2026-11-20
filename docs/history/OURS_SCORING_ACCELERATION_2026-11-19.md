# Ours 评分执行提速与后台恢复（2026-11-19）

用户授权的是执行提速。候选范围、mutation分布、ODE、事件处理、采样公式、seed、FNO超参及训练预算均保持不变。

## 原因与修改

- 原流程每次预测128条，但随后逐候选在主进程执行有限差分Jacobian、saltation和隐式输运。两个90k级评分各占住一个全局评分锁，普通方法因此额外等待约4小时47分。
- `ours/mechanistic/parallel.py`：使用spawn启动独立CPU worker，主进程继续负责FNO预测。一个预测batch是全部在途任务上限；结果按输入顺序收集，不改变排序、平分处理或随机探索。CPU worker每个单线程，不使用CUDA，不读候选真值或real目录；异常直接传回，不零分补位。
- 正式队列每个长评分使用8核，同时最多2项长评分；普通评分仍最多2项，使用独立锁。整体仍最多4实验、每卡最多2项，仅GPU0/1。启动器跳过暂时无法准入的长任务，让待运行的普通任务先开始，避免四个实验位置都被长评分占住。
- 生成流程同一轮内缓存训练集RMS和等价证据索引，新轮次refresh重建。缓存绑定训练对象及系统/RHS身份；不缓存未知候选标签，不改变参照集。
- I/O并发仍4，系统可用内存底线仍100GiB，保留批次进程树与磁盘压力监测。没有提升GPU实验总并发，没有复制池轨迹。

## 独立测量

使用已有真实FNO checkpoint，仅从训练侧候选输入取样；参考真值仅取已允许的初始训练样本。每组128候选，以下并行时间是持久worker预热后时间，不含一次性启动。并发共享机器有波动，不能当作整场实验加速比。

|池|方法|原串行秒|8核秒|评分加速|
|---|---|---:|---:|---:|
|mutation|Dynamics Transport|15.43|3.38|4.56×|
|mutation|Mechanism Witness|16.73|2.67|6.26×|
|LHS|Dynamics Transport|18.81|3.58|5.25×|
|LHS|Mechanism Witness|20.43|2.52|8.11×|

上述512候选的分数及响应特征逐位一致；另外12组串行/并行/等价证据分支检查中，3种新方法的入选ID与算法诊断一致。还检查Budding/Fission的事件路径、重复调用、异常传播、生成轮次缓存失效、长短任务准入及恢复流程。最终相关回归共64项全部通过。以scratch/probes/ours_speed_20261119中的profile.json、selection_check.json和tests_verified.log为证。

## 已有进度如何保留

先停止旧队列及其明确拥有的4个子进程，再修改生产源码。旧source快照、manifest、日志和切换前状态留在原队列目录；旧guard的“Batch interrupted”是本次计划切换，不是内存不足。

两项已完成的Pattern Context/Support实验不重跑。其余4个已有实验仅更新config的source身份，并保存明确的execution_migrations审计；没有放宽通用身份校验。36份已保存模型、历史和选样/数据引用文件在切换前后逐一核对SHA。已完成轮次复用checkpoint，已保存selection复用ID；尚在内存中扫描、未保存的本轮评分重新计算。恢复可能重做部分只读评测，但不重新训练已完成的轮次。

结果根目录不变：

`/home/users/zsy/mydata/budding_yeast/experiments/budding_ours/seed42_20261119/`

新队列记录：

`/home/users/zsy/mydata/budding_yeast/experiments/budding_ours_queue/seed42_20261119_accelerated/`

tmux会话沿用`budding_ours_seed42_20261119`。实验train.log沿用原日志并追加恢复标记。新队列的execution_migration.json记录旧队列和每项此前耗时，分析总用时时应合并切换前后，不能只用新PID运行时长。

正式速度、进度与内存以新队列status.json、resources.jsonl和train.log的PROGRESS为准；没有承诺评分的4–8倍可等比例缩短ODE生成、训练或绘图。
