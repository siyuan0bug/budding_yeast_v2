# 2026-10-31 optimized 批次完整性与 Event 修复

检查对象：2026-10-30 22:22 开始的 optimized_v2_full 共11项。结果为10项正常完成、1项Event首次选样失败。原批次退出码1由该单项失败导致，不表示10项完整结果失败。

原批次开始到最后一个任务退出约 9小时36分19秒（昨晚22:22:13至今天07:58:32）。四GPU并行，各卡内部串行；正常单项约2小时47分至3小时33分。

| 方法 | 完整用时 | MAE | MSE | RelL2 | 状态 |
|---|---:|---:|---:|---:|---|
| ours_global_interaction | 3小时21分53秒 | 0.108465 | 2.911264 | 0.154655 | 四轮完整 |
| al4pde_uncertainty | 2小时53分38秒 | 0.109821 | 2.146834 | 0.152717 | 四轮完整 |
| ours_event | 0小时16分23秒 | — | — | — | 首次选样失败 |
| ours_global_response | 3小时31分08秒 | 0.101676 | 1.748053 | 0.146930 | 四轮完整 |
| al4pde_power | 2小时48分32秒 | 0.085997 | 1.329895 | 0.131711 | 四轮完整 |
| ours_invariance | 2小时55分14秒 | 0.103247 | 1.642369 | 0.141069 | 四轮完整 |
| al4pde_lcmd | 2小时47分38秒 | 0.092770 | 1.537290 | 0.134808 | 四轮完整 |
| ours_global_tail | 3小时06分58秒 | 0.093763 | 1.629908 | 0.129411 | 四轮完整 |
| ours_response | 2小时49分00秒 | 0.109470 | 2.359423 | 0.146629 | 四轮完整 |
| al4pde_bait | 3小时33分07秒 | 0.135454 | 3.061644 | 0.173774 | 四轮完整 |
| ours_global_calibrated | 3小时08分42秒 | 0.113085 | 2.605886 | 0.156262 | 四轮完整 |

## 已完成实验的实证检查

10项均退出码0、四轮训练历史分别100/100/100/200 epochs，总500。累计合成train分别10000/20000/40000/70000；固定val/test/real为1250/1250/126，跨轮/跨方法身份顺序一致且切分不重叠。

逐轮检查checkpoint global_step为1900/3900/7800/27200，与生产DataLoader的batch512、drop_last=True完全一致。最后一轮27200次优化器更新，不是只打印200epoch。需要ensemble的方法前三轮独立成员的checkpoint训练步数也匹配，辅助训练合计300epoch；LCMD/BAIT使用预测轨迹特征，不额外训练ensemble。

每项三次追加10000/20000/30000条，合计60000，候选ID不重复且不与初始train/val/test候选交叉。逐轮manifest来源SHA及读取的字段摘要通过；未重新加载全部大轨迹做逐点数值审计。各轮归一化为train_synthetic/log1p，训练样本数匹配、统计量有限、尺度为正。

各轮物理metrics JSON有126株real和1250合成测试，指标有限、126幅real PNG及loss曲线齐全。10项日志没有本轮检索的异常、超时重建或CUDA OOM。主模型训练共约70–88分钟/项；ensemble约24–30分钟/需要项，余下时间为候选预测/选样、ODE及I/O/评测。全部完整任务合计约30.87 GPU小时，四卡重叠后约9.6小时墙钟时间合理。百万pool只有输入而非百万ODE轨迹；每项只在线新增6万条仿真。

完成预算不等于严格收敛。10项最终convergence.json仍为still_changing，当前0.1%/epoch、连续3次的判据仅通过最后1次；本轮未提前停止，也未擅自改变训练预算。

## Event 故障、修复及重新启动

失败时间05:06:50，首次选样调用event_disagreement，报 `ValueError: The truth value of an array with more than one element is ambiguous`。事件去抖结果改为NumPy数组后，`if divisions`在两个及以上有效分裂时失效。修为`if len(divisions)`，事件阈值和去抖规则不变。之前的回归只涵盖零个或一个有效分裂，漏掉该分支，这是此次遗漏。

新测试覆盖0/1/2/3次分裂、相同预测零分歧和成员不同预测的非零分歧，相关36项回归通过。使用失败任务保存的两个真实FNO重放512个训练输入，包含33条多分裂成员预测，评分全部有限、无异常。原始pool/初始集、FNO、损失和训练预算不变。

新tmux：`al_budding_eventfix_20261031_102144`；新结果：`/home/users/zsy/mydata/budding_yeast/experiments/al_budding/fno/hybrid_mse_rel_ours_event/w192_n4_m24_b512_e100-200_lr1e-3_lmin1e-5_wm5-10_al4_refit_gene_ranges_optimized_v2_eventfix_full/seed_42`。从round0冷启动，原失败目录保留。W&B强制offline，每卡1项，GPU0，ODE workers4。

启动清单：`/home/users/zsy/mydata/budding_yeast/analysis/al_eventfix_launch_20261031_102144/manifest.json`。原批次manifest增加followup_retries引用，保留原失败状态，不覆盖历史审计。

## 复查命令与资料

`scripts/analysis/audit_al_batch_completion.py --manifest <launch/manifest.json> --output <analysis_dir>`。

本次机器可读结果：`/home/users/zsy/mydata/budding_yeast/analysis/al_optimized_completion_20261031/integrity.json`、`completion.csv`、`event_replay.json`。单seed42的性能表只作描述性结果，不能代表跨seed显著性。

启动后代码快照对比仅多出其他模型注册改动（marker_closure）和本次Event修复，FNO前向、损失和训练流程与启动快照一致；未覆盖用户的其他模型更改。

## 附加日志问题与重跑健康状态

10:28补查发现EpochSummaryCallback在on_validation_epoch_end读取训练epoch聚合值，第一轮显示nan，其后train_loss滞后一轮。训练CSV和convergence在on_train_epoch_end更新，41个已结束轮次的training_history.csv中训练loss全部有限；权重、曲线和收敛统计不受该日志显示问题影响。摘要改到on_train_epoch_end；真实Lightning小型单元回归确认首epoch即打印正确聚合值，预算/离线日志相关36项测试通过（与前述Event相关36项为不同测试组）。此修改在以后启动的进程生效，当前Event进程保持运行，分析其训练loss以CSV为准。

Event重跑于10:28进入正式训练，10:29已到第一轮约33/100epoch，GPU0、进程649009，W&B离线，无新异常。尚未完成重跑，首次生产选样仍在后面；多分裂评分已用保存FNO的512输入验证。
