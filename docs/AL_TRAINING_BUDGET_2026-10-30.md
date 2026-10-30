# Budding AL：快速初筛与长预算复核

保留screen/full两档预算。用户最新要求全部切换长程，当前默认 `TRAINING_PROFILE=full`、`AL_SUITE=research`，14项正式批次已在GPU0–3启动；screen保留为显式选项。保留FNO容量、batch、峰值LR、数据和采集机制。启动详情见[本批记录](AL_FULL_LAUNCH_2026-10-30.md)，尚无本批完整精度或已收敛结论。

## 当前设置

|项目|screen：显式初筛|full：当前默认长程|
|---|---|---|
|前三轮epoch|50/50/50|100/100/100|
|最后一轮epoch|100|200|
|warmup|4/4/4/8|5/5/5/10|
|峰值LR|1e-3|1e-3|
|余弦下限|1e-5|1e-5|
|batch|512|512|
|FNO width/blocks/modes|192/4/24|192/4/24|
|训练量|10k→20k→40k→70k|相同|

原50/50/50/100 epoch结果的loss仍在下降，因此screen只比较有限计算预算下的表现，不宣称充分收敛。根据合成验证表现、训练趋势和开销筛选有潜力的机制，再对少数候选连同Random/Power等baseline采用相同full协议；不将两档结果直接混排。少epoch和充分优化可能有冲突，降低预算没有消除这个问题。

两档保留AdamW、weight_decay1e-4、梯度裁剪、bf16、冷启动及逐轮train-only归一化。下限均为峰值的1%，不同于历史1e-6下限。full提供更长的有效学习阶段，但也不保证200 epoch必然充分收敛。各方法和其辅助ensemble采用同档位预算。

drop_last=True，每epoch分别19/39/78/136次更新：screen主模型总更新数20,400，full为40,800。screen相比全14项都用full，训练更新数减半；仿真/选样/评测开销不会因此减半，所以不能保证总墙钟减半。需要ensemble的方法，其辅助成员采用相同档位。两档均查询至70k有效训练轨迹。

## 使用方式与结果目录

```bash
cd budding_yeast_v2
AL_SUITE=research DRY_RUN=1 bash run_al_budding.sh
# 正式实验由用户启动
AL_SUITE=research SKIP_PREPARE=1 MAX_JOBS_PER_GPU=2 bash run_al_budding.sh
# 可选：仅选部分方法运行full（本次实际启动了全部14项）
AL_SUITE=research TRAINING_PROFILE=full SKIP_PREPARE=1 \
  STRATEGIES="al4pde_random al4pde_power ours_response_feedback" bash run_al_budding.sh
```

`research`仍为6个baseline、首版ours、7个新变体，总共14项。直接运行脚本默认research全部14项、full预算；AL_SUITE=original可只选原七项，TRAINING_PROFILE=screen可显式缩短预算。非法档位在生成数据前报错。GPU0/1/2/3队列4/4/3/3、每卡最多2项、W&B强制offline。

复用现有 `budding_pool_1000000_seed42_gene_ranges.npz` 和 `budding_initial_train10000_seed42_gene_ranges.npz`；SKIP_PREPARE只执行已有数据一致性检查，不重新仿真初始集。

结果目录现在明确记录最低LR：

screen：`/home/users/zsy/mydata/budding_yeast/experiments/al_budding/fno/hybrid_mse_rel_<strategy>/w192_n4_m24_b512_e50-100_lr1e-3_lmin1e-5_wm4-8_al4_refit_gene_ranges_research_v1_screen/seed_42/`

full：对应 `e100-200`、`wm5-10`、`_research_v1_full`。两档目录分开，已有结果不改写。full从共享初始集重新运行完整AL闭环，包含各轮训练、采集及ODE；不是将screen最后一轮checkpoint续训冒充完整full实验。

原结果原地保留。`ROUND_EPOCHS`等显式覆盖仍高于档位默认值，实际设置以dry-run和config为准。`MIN_LR`可显式覆盖；若要精确复跑历史预算，可指定 `ROUND_EPOCHS=50 FINAL_EPOCHS=100 ROUND_WARMUP=4 FINAL_WARMUP=8 MIN_LR=1e-6`，并核对研究组的固定round seed和历史legacy seed差异。

模型/静态CLI的 `min_lr` 默认仍为1e-6，Fission脚本未改。本次只有Budding脚本显式传1e-5；老checkpoint缺少该字段时仍按1e-6构造调度。余弦进度在完整周期后固定为1，避免意外超出原训练周期后学习率回升；这不是完整AL断点续跑功能。

## Loss变化率监控

每轮主模型新增 `convergence.csv` 和 `convergence.json`；辅助成员保存至 `rounds/round_XX/ensemble/member_01/` 等对应目录。主模型的round summary和本地W&B同时记录设置与变化率，train.log每10个epoch打印一次。

监控在训练epoch最终聚合之后读取训练loss与合成验证MSE，和现有training_history相同的epoch口径，不使用real或合成测试指标。每个10-epoch窗口计算：

- 净下降率：`100*[1-(L_end/L_start)^(1/10)]`。
- log(loss)线性拟合的趋势下降率：`100*[1-exp(slope)]`，避免仅凭首尾两个点判断。
- 每步绝对相对变化率均值，显示波动是否被净下降率掩盖。

每10个epoch检查一次：训练loss和合成验证MSE的绝对趋势均低于0.1%/epoch，连续3次，则标记 `plateau_candidate`；否则为 `still_changing`。历史不足或指标缺失分别标记，JSON不写NaN。相邻检查窗口仅共用边界点。

0.1%是本次明确记录的诊断阈值，不是通用收敛定理；小学习率仍可能让loss变平。**监控不会提前停止训练或据real指标调整预算**，不改变各方法的固定epoch公平性。训练结束仍在变化时，会如实保留状态；达到预算不自动等于已收敛。

## 档位验证与产物

本次档位改动55项检查通过，覆盖默认screen、显式full、两档14种方法一致性、显式参数覆盖、离线W&B与调度；正式训练未启动。以下134项为此前调度/监控实现的回归记录，不是新的训练精度结果。

134项回归通过，包括调度旧值兼容、最低LR持久化配置、整个新周期及周期后不回升、非法下限拒绝、变化率公式、零/缺失loss、连续窗口且不早停、实际Lightning当前epoch聚合、14项脚本参数一致性、候选采集/反馈、数据追加、dtype、ensemble与离线W&B。17条warning为已有/CPU测试提示，没有失败项。CPU小测试不代表正式FNO已完成训练。

同时通过shell语法和git diff检查，已生成新旧学习率曲线及计划；未运行正式重训。

相关实现：`train.py`、`models/yeast_lit_module.py`、`utils/active_learning/convergence.py`、`run_al_budding.sh`。

分析产物：`/home/users/zsy/mydata/budding_yeast/analysis/al_training_budget_2026-10-30/`，包括 `protocol.json`、`research_plan.txt`、`learning_rate_schedules.csv`、`learning_rate_schedules.png/pdf`。曲线来自实际模型构造出的scheduler，不是拟合的训练loss。

方法枚举与预算规则分别见[研究变体](AL_QUERY_VARIANTS_2026-10-30.md)、[收敛数据](AL_CONVERGENCE_AND_NEXT_IDEAS_2026-10-29.md)。
