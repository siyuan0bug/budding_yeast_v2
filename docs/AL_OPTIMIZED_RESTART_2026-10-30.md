# 2026-10-30：全局覆盖与局部响应的优化批次

本轮按用户指示实现新采集方法，并从 round 0 重跑此前未完成的实验。旧结果保留，新的结果使用 `gene_ranges_optimized_v2_full` 目录后缀。新方法尚无完整实验结果，不能据选样验证宣称优于基线。

## 四个新方法

旧 interaction 改善 #15/#22/#50，却恶化 #62/#124；局部对照不能替代全局覆盖。本轮将“先覆盖预测轨迹空间、再查询局部响应”作为共同框架：

| 策略 | 全局覆盖 | 局部查询 | 随机探索 | 额外机制 |
|---|---:|---:|---:|---|
| `ours_global_interaction` | 50% | 40% 四端点相互作用 | 10% | 保留 MC/GAL-MC × KO 的合法四点对照 |
| `ours_global_response` | 50% | 40% 同背景强度响应 | 10% | 沿用合成节点响应误差反馈作为对照 |
| `ours_global_tail` | 30% | 40% 同背景强度响应 | 10% | 20% 按变量平衡的预测大幅值分层查询 |
| `ours_global_calibrated` | 50% | 40% 同背景强度响应 | 10% | 最近一次完整合成边的响应误差校准 |

比例按本次新增的、去重后的未知端点数计算。已在训练集或本批中选择的端点不重复计费；失败端点不计有效样本预算。局部动作不足时以全局 MaxDist 补足，而不是由单点不确定性占满剩余对照预算。尾部分层不足时余额也归全局。每次追加仍填满 1万、2万、3万条有效轨迹。

全局距离使用两个独立 FNO 的平均预测：32 个固定时间点 × 全部变量的归一化预测，共享确定性 CountSketch 压到 128 维。复用官方 bmdal MaxDist 选择器，以累计合成训练集的**模型预测**及本批已选点为锚。它与原 `al4pde_max_dist` 的 512 维特征设置不同，是本轮 ours 的表示选择，不宣称与基线逐项等价。投影与选择使用局部随机状态，不改变训练 RNG。

局部响应沿用合法背景图：模块、模式、环境、初值及固定参数约束一致，在既定强度分布的 CDF 坐标内连边；可包含多个连续强度，不需要另造池外候选。响应评分为 ensemble 对轨迹变化量的分歧，除以 `1 + CDF距离`，并由方向覆盖的边际收益抑制重复查询。四点动作使用二阶差分；两个独立成员的中心化对比分量可精确压成一个，保留原方差。

尾部分层对每个变量分别计算候选的预测峰值 p90/p99，建立 p90–p99 与 p99+ 两档，在各档按该变量的物理预测方差排序，跨变量/档轮流取点并去重。分位数只是**选样分组**，不截断轨迹，不删除大值，不用 real 极值设阈值。

校准版在查询前保存 17 维边特征：单点/响应分歧、CDF 距离、预测峰值与端点差异、固定输入差异、模块/模式背景等。查询后仅用两个端点都已知的合成对照计算物理 `log1p` 响应 MSE；失败端点、不完整对照和单点回退标签不参与边校准。只保留最近一次追加（含同次补采批次）的边训练 ExtraTrees；目标是 `log(真实响应误差+ε) - log(预测log空间分歧+ε)`。下一次采集按预测校正量的排名乘以 1～2 的有界权重，不直接乘无界误差。历史 80/20 随机拆分只提供局部校准诊断，不是全池无偏评估，也不保证模型跨轮保持校准。

评分不读取 real 或固定验证/测试的真轨迹。训练锚点的真值只在**查询后的合成误差反馈**中使用。real badcase 仅用于此前事后诊断，算法没有 #15/#124 身份特判。

## 六项重跑与事件修正

重跑 `al4pde_uncertainty`、`al4pde_power`、`al4pde_lcmd`、`al4pde_bait`、`ours_response`、`ours_invariance`。全部冷启动，不混用旧中断目录的数据或 checkpoint。

同时重跑 `ours_event`：原事件描述符在去抖时比较“上一次保留跌落”，与中性六分类比较“上一次检测跌落”不同；如 5/11/17 步连续跌落，原描述符误计两次，规范规则计一次。本次统一到六分类规则，不改真实标签与 BIO/ABC 定义。

## 资源修复和公平协议

旧四项退出码 137；另两项第三次 ODE 同时超过 35 秒无完成结果。进程组有累计 OOM kill 记录，但不能据此逐项证明死因。新默认每 GPU 一项、四 GPU 并行，Loader workers=0、每项 ODE workers=4。

- Worker 不再传回父进程未使用的巨大虚拟 X；轨迹在 IPC 前转为最终存储所用的 float32，参数和实际初值仍 float64，ODE 仍双精度。训练输入仍 float32。
- 追加后的归一化分块写入，释放仿真临时结果；下一轮重载累计数据之前主动释放旧 Lightning 对象及数据循环引用。
- 有界进程池若 Timeout/BrokenProcessPool，最多重建一次，从尚未交付结果的候选继续；已经交付的结果不重算、不乱序。不把基础设施失败标为生物不可存活。重复故障明确报错退出。
- 求解器 5 秒限制、父进程 35 秒无结果保护、轨迹有效性检查和有限负值/大幅值保留规则不变。

所有方法共用同一个百万 pool、初始 10000/1250/1250+126 real、seed42，FNO width192/blocks4/modes24/batch512，LR=1e-3→1e-5，100/100/100/200 epochs，warmup5/5/5/10，逐轮冷启动及累计合成 train-only 归一化重拟合，ensemble 两个独立成员。方法不涉及分布、ODE 或损失修改。

## 启动与结果

入口仍为 `run_al_budding.sh`。`AL_SUITE=optimized` 为默认新批次；`research` 保留旧 14 方法清单，`original` 保留原基线清单。`MAX_JOBS_PER_GPU=1` 为默认值。四卡队列：

| GPU | 依次运行 |
|---|---|
| 0 | global_interaction → Uncertainty → 修正 event |
| 1 | global_response → Power → Invariance |
| 2 | LCMD → global_tail → Response |
| 3 | BAIT → global_calibrated |

只使用 GPU0–3，占用中的卡等待空闲。W&B 强制 offline，日志带时间。结果在 `/home/users/zsy/mydata/budding_yeast/experiments/al_budding/fno/hybrid_mse_rel_<strategy>/w192_n4_m24_b512_e100-200_lr1e-3_lmin1e-5_wm5-10_al4_refit_gene_ranges_optimized_v2_full/seed_42/`。

继续使用共享初始数据＋追加块＋逐轮 `dataset.json`，不重复保存累计轨迹。校准版追加目录增加 `edge_feedback.npz` 和 `edge_feedback_schema.json`；原 prequery/query_design 审计保留。

验证脚本：`scripts/analysis/verify_al_hybrid_integration.py`，用旧已保存的真实 FNO 及独立成员验证新策略选样、真实 ODE、入库、manifest 重载、标签/dtype、跨轮 float32 输入刷新与近期边反馈，不做新模型短训。检查结果和启动状态放在机械盘 `analysis/al_optimized_*` 目录。正式训练完成后需再比较同预算整体指标、坏例和计算成本。

## 本次验收与后台状态（22:06）

- 专项回归115项通过；另一次相关回归54项通过（有重叠，不相加计独立用例）。先前包/预处理/存储等100项检查通过。Bash语法及git diff空白检查通过。
- 保存的真实FNO与独立成员：4个新方法均从4096候选正确选择256个唯一输入，未改共享pool。原生ODE追加256/256通过，累计10256训练样本、固定1250/1250/126验证/合成测试/real、完整标签和float64参数/初值及float32轨迹均验收通过。近期边反馈97条，能够完成拟合；这不是新AL完整精度结果。
- pool及初始集SHA256与旧批次完全一致。
- 后台tmux：`al_budding_optimized_20261030_220508`，batch/status两窗口。启动清单与代码快照：`/home/users/zsy/mydata/budding_yeast/analysis/al_optimized_launch_20261030_220508/`。
- 批队列日志：`/home/users/zsy/mydata/budding_yeast/experiments/al_budding/batch_logs/20261030_220539_77023/`。
- 启动检查时0–3号卡均被其他用户占用，四个worker均已打印WAIT GPU；11项在排队，**尚未正式训练**。卡内存低于512MiB后，每30秒检查的后台队列自动启动。未更换GPU、未抢占其他进程；关闭IDE不终止tmux。
- `tmux attach -t al_budding_optimized_20261030_220508` 查看状态；`Ctrl-b`再按`d`脱离。
