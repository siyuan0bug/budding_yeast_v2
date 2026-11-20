# Ours 队列：6→8→10 并发验证

用户授权仅使用GPU0/1提高整批吞吐，继续保留100GiB系统可用内存底线。训练超参、seed、数据、归一化、采样公式、评分候选范围均不改。

## 为什么显存低还会慢

长评分在CPU进行。目前实际两个8-worker评分约38候选/秒，84k候选约36.8分钟。四次全池评分合计约2.5小时，另有训练、归一化准备及逐轮评测。因此新机制从头运行仍是小时级；旧版本累计耗时包含原串行评分和此前等待，不能直接当作本次执行配置的预测耗时。

普通Pattern Context/Support已完成的6小时51分中，约4小时47分是旧全局评分锁等待，扣除等待的实际处理约2小时。新的长评分方法不能用这个处理时长直接替代其自身计算成本。生成型方法另有8000条新增ODE查询，耗时取决于真实成功率和求解时长。

增加并发针对整批完成时间；单项可能因共享CPU/GPU变慢。当前阶段按实时速率估计，mutation Witness/Transport各剩约1.5–2小时；提高并发后需要重新估算。

## 调度与保护

- 总并发先6，健康时每120秒加2，目标上限10；GPU0/1各最多5实验进程。GPU4–7独占规则及授权范围不变。
- 长任务总在途最多5。继承的两个CPU评分继续各用8 worker；新增长任务每项4 worker；评分worker每个单线程。最多约28个长评分CPU worker同时计算，不把10个实验都配置为8核。
- 重加载I/O分两个准入域：继承4项保留原I/O上限4，新加入任务的域上限4，合计最多8。不能把新增域4误解成整个队列仍只能4个I/O，也不能宣称新旧任务都已改了环境变量。
- 分域还防止新任务占住旧任务只能访问的两个CPU评分锁，从而使旧任务再次长时间等待。
- 总进程调度/可用内存和磁盘关联压力守护由新主管统一覆盖所有继承和新增任务。仅将自有PSS作为监测值，没有自有内存硬配额；系统MemAvailable底线100GiB，扩容需至少200GiB可用。
- W&B offline，原数据文件和结果根目录沿用，不复制轨迹。

## 如何保留正在计算的进度

新模块`scripts/launch/queue_handover.py`使用UID、独立进程组、PID启动时刻、GPU环境及旧主管身份校验。只停止已核实的旧主管，不停止实验进程/CPU workers。新主管读取不可变handover receipt，接管同一批PID及其内存守护。

接管后的非亲生子进程无法取得OS waitpid退出码，因此账本显式标记`exit_code_source=verified_completion_after_adoption`：必须在进程结束后通过原有完成状态、126株real结果/图片和合成test CSV等校验，才能标为完成；不凭PID消失猜测成功。PID复用或不属于自己的进程直接拒绝。

5项定向接管测试、以及与旧队列组合的16项测试通过，包括真实小主管退出后子进程持续运行、PID复用拒绝、输出不全判失败、10并发边界、独占卡拒绝。没有修改`utils/models/data`等冻结的实验源码；仅扩展启动器与接管工具，不需要迁移/重写实验config。

## 查看位置

- tmux：`budding_ours_seed42_20261119`，monitor窗口跟踪新queue.log。
- 当前队列：`/home/users/zsy/mydata/budding_yeast/experiments/budding_ours_queue/seed42_20261119_c10/`。
- 结果：`/home/users/zsy/mydata/budding_yeast/experiments/budding_ours/seed42_20261119/`（不变）。
- 旧主管接管回执：`seed42_20261119_accelerated/handover.json`；旧任务PID不变。
- 加压记录：`scratch/probes/ours_concurrency_20261119/ramp_observations.jsonl`、`ramp_summary.json`；正式资源记录在新队列`resources.jsonl`。

是否已实际达到10并发，以及当前训练/评分阶段，必须以status.json和逐项日志为准，不能仅根据最大值配置宣称已经完成测试。

## 现场验收（10:58，北京时间）

已从6升至8再升至10个实际运行进程，GPU0/1各5个；2完成、10运行、5排队、0失败。原4个PID、启动时刻均不变。新增LHS Witness已记录训练epoch，其他新增项处于加载、归一化准备或评测。

加压观察到的自有PSS峰约135GiB、系统可用最低约550GiB；I/O full PSI最高0.05%，未触发可用内存/磁盘关联守护。曾有短暂memory PSI约3%，当时仍有约550GiB可用内存，不能据此说内存已满；同时也不宣称任何压力指标始终为零。CPU负载约64，接近64个逻辑CPU，因此后续吞吐主要受CPU及训练/绘图阶段重叠影响，不能仅按显存闲置继续无限扩容。

上述是实际启动/进入训练的加压窗口，不是覆盖全部五轮与所有生成ODE的长期峰值保证。原资源守护继续监测，结果/算法源码身份保持不变。

7分钟加压记录结束：42次采样，最高10并发，自有PSS峰137.22GiB、系统可用最低547.68GiB，I/O full PSI峰0.05%，0失败、无守护中止；结束时仍10项运行。该7分钟包含升到10之前的阶段。

## 11:41追加启动剩余5项（用户明确要求）

采用同一PID接管机制将10项现有实验移交给`seed42_20261119_all15`主管；原进度没有重新计算。解除整项heavy任务准入，max_jobs=15、每卡最多8，实际GPU0为8项、GPU1为7项。新启动mutation×Witness/Transport/Closure，以及LHS×Witness/Transport，共5项生成式实验；总计2完成、15运行、0待启动。

为保持已有资源约束，增加`--added-resource-dir`仅允许复用经过原manifest核实的c10资源域；新5项和原B域任务共用I/O4、CPU评分5项×4worker，没有创建第三套锁来绕过限流。旧A域I/O4和两项8-worker评分保持原值，因此整体I/O上界仍8，最高同时CPU评分worker上界由28变为36；ODE每个生成实验仍2worker。守护继续看全队列的系统可用内存100GiB底线，而不是自有内存硬配额。

`--admit-all-pending`表示不再等完整实验结束才启动；加载、评分和求解阶段仍可短时等待共享资源。启动时剩余内存约496GiB，0失败；尚不等于所有新增项已完成首轮训练。17项接管/队列回归通过，实验源码身份及101个生物/data根目录文件SHA保持不变。

当前queue.log/status.json/tasks.json、统一resources.jsonl及all15_verification.json都位于：
`/home/users/zsy/mydata/budding_yeast/experiments/budding_ours_queue/seed42_20261119_all15/`。结果根和tmux名字不变。
