# AL 时间日志、ours JSON 与进程池阻塞修复

## 已修复的 ours 错误

`CandidatePool.keep()` 原来把 `mask.sum()` 的 `np.int64` 保存为池大小。
选到已有训练锚点时，`training_row=i-pool_size` 也变为 `np.int64`，JSON 无法编码。

当前池大小在源头转为 Python `int`；response 的 training_row、timings 继续显式转为 int，覆盖外部调用传入 NumPy 整数的情况。
response 审计先完整序列化，再通过临时文件原子替换，避免留下半截 JSON。
测试覆盖“池裁剪＋训练锚点＋选择＋审计保存”，不只是单独测选样。

现有失败实验的20k累计数据仍保留；没有重写其半截 response_pairs.json，也没有把缺少后续训练的结果标为完成。

## Random、Uncertainty 为何卡住

本次对 PID 2142133、2142142 进行只读检查，没有向其管道读取消息或发送信号：

- 主线程均为 `pipe_write`，进程池管理线程等待 futex，部分 worker 写结果管道。
- 两个进程的 executor 唤醒管道（FD65/66）均填满 **65536/65536 bytes**。
- 结果管道也被大块轨迹结果占据，worker 无法继续回传。

本机 Python 3.10 的 `concurrent/futures/process.py` 显示：

1. `submit()` 持有 `_shutdown_lock`，向唤醒管道 `send_bytes(b'')`。
2. 管道填满后，提交线程等待读端清空，但仍持有该锁。
3. 管理线程也要先取得 `_shutdown_lock`，才能 `thread_wakeup.clear()`。
4. 两者互相等待；返回轨迹的 worker 随之阻塞。

旧 `bounded_ordered_map()` 在开始消费结果和35秒父级监控之前，先提交全部30,000个任务，因此会进入这个死锁窗口。不是某个 ODE 一直计算，也不是GPU显存不足。

为验证机制，在独立子进程中只给首次管理线程调度增加1秒延迟：旧式20,000次连续submit稳定重现；
faulthandler明确显示主线程卡在 `submit→wakeup→send_bytes`，管理线程卡在取得同一锁。
相同调度延迟下，当前有界提交20,000任务在 **7.15秒** 完成，返回顺序完全一致。
线上 Python 栈受 ptrace 权限限制无法直接读取；管道满载及线程等待状态与独立重现相符。

当前实现最多保留 **2×worker数** 个提交或已完成未消费的任务；默认4worker对应8条任务，防止唤醒管道和内存缓冲被海量提交填满。
保留原候选顺序、5秒单样本保护和35秒无完成结果保护；watchdog触发会记录待完成数、worker PID及超时阈值。

没有修改系统 Python，也没有向线上满管道“抽取几条消息”来尝试恢复。
已挂起的进程仍载入旧实现，新代码不会热替换；本轮未终止或重启正式实验。

## train.log 的新格式

时间统一为 `Asia/Shanghai`，含毫秒与明确 `+08:00` 偏移。
`train.py` 对 stdout/stderr 的常规文本输出逐行加时间；保留原 `[Epoch ...]` 等内容，旧损失曲线解析仍可使用。
`run_al_budding.sh` 将启动、退出码、GPU/模型/策略/seed和结束时间同时写入对应 train.log。
失败仍返回原退出码，W&B仍离线，不新增第二份训练日志。

```text
[2026-10-29 15:08:18.965+08:00] [START] AL round | round=1 rounds=4 strategy=ours_response
[2026-10-29 15:08:18.965+08:00] [START] main model training | round=1 train_samples=10000 epochs=50
[时间] [Epoch 001/050] train_loss=... | val_synth=... | ...
[时间] [END] main model training | round=1 train_samples=10000 epochs=50 elapsed_s=...
[时间] [START] candidate scoring and selection | strategy=... acquisition=1 pool_size=987500 budget=10000
[时间] [PROGRESS] ensemble uncertainty inference | members=2 completed=... total=... elapsed_s=... rate_per_s=...
[时间] [START] ODE process pool | candidates=10000 workers=4
[时间] [PROGRESS] ODE simulation | workers=4 completed=... total=10000 elapsed_s=... rate_per_s=...
[时间] [END] acquisition snapshot and audit saving | ... elapsed_s=...
```

示例是格式演示，不是新正式实验的测量结果。

阶段包括：初始数据载入/归一化、候选池准备、每轮开始/结束、累计数据归一化重拟合、主模型训练、checkpoint保存、真实/合成测试评测及绘图、汇总及loss图、额外ensemble成员训练、候选评分选择、ODE/过滤、有效配额进度、训练数据追加、快照与审计保存。
候选推理和ODE每约30秒有新的完成结果时记录一次进度，结束时也记录；这不是在无进展时伪造“仍然正常”的心跳。
异常会记录 `[FAIL]`、阶段名、耗时与异常类型，随后保留原 traceback。

新进程启动时还会注册 `SIGUSR1` 的全部Python线程栈转储，便于以后不停止训练地排查。
**仅对启动日志明确显示 `stack_dump_SIGUSR1=True` 的新进程使用该信号；旧进程未注册，不能直接发送。**
本轮没有向旧实验发送该信号。旧日志不补造历史时间，新格式从之后启动的进程生效。

## TEM1 的“数值伪影”是什么意思

TEM1是活性蛋白量，`Tem1T`是该样本设定的蛋白总量；当前方程下，合法初值的活性量应保持在总量内。
例如候选47184：总量7.6725，原求解峰值11.8037；收紧容差并限制步长后，峰值7.6479。
这表示默认精度下的数值步骤越过了物理边界，不能把11.8037当作这个参数组合的可靠真实输出。

方程的激活项包含 `(Tem1T-TEM1)/(jatem+Tem1T-TEM1)`，越界后还可能靠近分母为零的位置，进一步放大伪峰。
这与CLB2等变量因过表达/降解缺陷产生合法高浓度不同，也不是按real峰值的倍数来判断。
跨新策略的26个可疑候选，严格设置 `rtol=1e-7, atol=1e-9, max_step=.25` 后25个成功且均回到总量内，1个超时。
既有数据、生产ODE与求解容差本轮未改变；这些数据的修复/重生成需要作为独立数据版本操作，不能悄悄改正在比较的训练集。

## 验证与证据

94项相关测试通过，覆盖日志碎片/多行/ stderr、阶段成功失败、进度、ours审计、dtype、池裁剪、有界提交、真实spawn与AL训练追加快照、批脚本及W&B键。
日志示例、管道快照、受控复现前后栈、测试输出保存在：
`/home/users/zsy/mydata/budding_yeast/analysis/al_logging_hang_2026-10-29/`。
对比结果见 [范围与AL实验审计](AL_RANGE_RESULTS_AUDIT_2026-10-29.md)。

追加验证：6项时间日志测试（含现有loss曲线解析器）通过，批脚本失败退出码/时间头尾及两个watchdog共3项通过；独立新进程注册SIGUSR1后成功输出Python栈并继续执行。没有向旧实验发送信号。
