# 2026-10-30 全部FNO主动学习长程批次

> **10:48进度更新**：2项完整完成、6项仍运行、6项异常退出。Random/MaxDist最终MAE为0.098193/0.097129。失败原因、同预算ours和loss变化率见[上午进度审计](AL_FULL_PROGRESS_2026-10-30.md)。下方为02:27启动记录。

用户授权后，本批已于北京时间10月30日02:27在独立tmux后台启动。默认脚本已切为research全部14项、full长程；screen档位保留但本批没有使用。正式运行状态应以启动目录的health和各实验train.log为准；尚无本批完整结果或收敛结论。

## 训练协议

FNO width192、blocks4、modes24、batch512；四轮100/100/100/200 epoch，warmup5/5/5/10，峰值LR1e-3、最低LR1e-5。四轮冷启动，seed42，按固定round seed规则训练；需ensemble的方法使用两个独立成员且辅助成员预算一致。最终训练量10k→20k→40k→70k，原pool/初始集复用，没有重新生成。

W&B强制offline，使用neuralop环境。保留loss变化率监控，不自动早停。

## 四卡队列

|GPU|队列中的完整实验|总项数|
|---|---|---:|
|0|Random、MaxDist、ours_response_marginal、ours_event|4|
|1|Uncertainty、LCMD、ours_response_feedback、ours_error|4|
|2|Power、首版ours_response、ours_interaction|3|
|3|BAIT、ours_invariance、ours_coverage|3|

每卡最多同时2项，因此首批8项，后续6项自动补位。失败记录退出码并继续队列，现有旧结果不覆盖。GPU4–7未使用、未操作其他用户进程。

## 后台与路径

tmux会话：`al_budding_full_20261030_022728`。

```bash
tmux attach -t al_budding_full_20261030_022728
# Ctrl+b，再按d：仅脱离，不停止训练
```

启动记录：`/home/users/zsy/mydata/budding_yeast/analysis/al_full_launch_20261030_022728/`。

- `manifest.json`：14项、GPU、精确输出路径、显式环境配置和源码指纹。
- `plan.txt`、`launch.sh`：实际启动计划和调用，核心入口仍为run_al_budding.sh。
- `launcher.log`：批处理启动/结束信息；详细训练日志在各实验目录。
- `startup_health.json`：tmux的status窗口每30秒刷新的状态快照；关闭status窗口后保留最后一次结果，更新时间见文件内checked。
- `startup_verification.json`：启动验收快照，确认首批8项已训练、其余6项排队，以及数据/归一化/seed/offline检查结果。
- `source_code.tar.gz`：本次启动时的源码备份；实际从工作区运行。
- `batch_exit_code.txt`：批次全部结束后写入的退出码，运行期间不存在。

实验结果：

`/home/users/zsy/mydata/budding_yeast/experiments/al_budding/fno/hybrid_mse_rel_<strategy>/w192_n4_m24_b512_e100-200_lr1e-3_lmin1e-5_wm5-10_al4_refit_gene_ranges_research_v1_full/seed_42/`

每项仍有config、train.log、rounds/checkpoint/dataset/归一化/loss图/real评测、acquisitions和summary。新变体额外审计见[策略说明](AL_QUERY_VARIANTS_2026-10-30.md)。

关闭IDE不会终止该tmux任务；不要重复运行同一批次。脚本会拒绝已有config的目标目录。

tmux窗口0为批任务，窗口1/status为只读状态页。status退出不会终止窗口0的训练；状态页不负责自动修复任务错误。

## 启动前检查

确认GPU0–3当时均仅2MiB显存、无训练进程，已有pool和initial存在；默认队列/调度56项回归通过。按历史每项13–14GiB估算，本批约190GiB；启动时机械盘可用230.5GiB。其余用户后续磁盘写入不可预先保证；未删除历史数据或改变数据保存内容。

截至02:38:55，首批8项均完成加载并训练至第一轮第32–46个epoch，6项排队，无Traceback或配置不一致。八项均核实train/val/test/real=10000/1250/1250/126、39维float64物理初值、500时间点/210min、精确L元数据、train_synthetic/log1p归一化、固定seed42，以及各1个W&B offline目录。GPU0–3各有两个训练进程，检查时每卡约9GiB显存。启动验收不代表后续所有轮次、采集算法和14项实验均已完成；以实时状态及最终退出码为准。
