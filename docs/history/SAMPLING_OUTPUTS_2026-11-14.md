# 完整池采样实验输出（2026-11-14）

2026-11-16绘图调整：删除loss图左下角的数值摘要框，避免遮挡曲线。每轮图及多轮总览共用此设置；曲线、双轴、图例和验证最优epoch标线保留，精确值仍可查history.csv及train.log。历史结果不批量改写。

本次仅整理训练、恢复、评测和绘图接口；不改变源池、划分、预算、选择算法、归一化拟合范围或训练目标。

正式结果位于：
`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/<protocol_id>/<mutation或lhs>/<method>/seed_42/`。

```text
seed_42/
├── config.json                 # 模型/训练配置、源代码哈希、真实评测参考
├── train.log                   # 带时间的阶段/epoch日志
├── loss.png                    # 已完成各轮按顺序排列；同一绘图函数
├── performance.csv             # 各轮real与两个来源测试指标
├── performance.png             # 多轮时生成；仅real的MAE/MSE/RelL2/Corr
├── eval -> rounds/round_04/eval # 指向最近完成轮，结束后为最后轮；不复制
├── rounds/
│   ├── round_00/
│   │   ├── model.ckpt          # 验证集最优模型+完整归一化契约，无optimizer
│   │   ├── history.csv         # 每个epoch的真实记录
│   │   ├── loss.png
│   │   ├── dataset.json        # 源池路径/哈希、有序行号和必要权重，无轨迹
│   │   ├── selection.json      # 下一轮新增行号、来源及选择诊断；末轮无此文件
│   │   └── eval/
│   │       ├── global_metrics_denorm.json
│   │       ├── per_mutant_metrics.csv
│   │       ├── per_synth_test_metrics.csv
│   │       └── real_mutants/   # 126幅完整39变量预测图
│   └── round_01…round_04/
└── _state/                     # 恢复、审计和辅助模型，不是额外数据集
    ├── state.json              # progress/oracle/completion三个独立部分
    ├── round_00/primary/
    │   ├── training.json       # 已完成训练身份、步数、耗时、loss变化率
    │   └── resume.ckpt         # 仅未完成训练保留；完整optimizer/scheduler/RNG
    ├── round_00/ensemble_member_1/
    └── wandb/                 # offline，本地保存，不上传
```

`eval`在每轮评测完成后原子更新；运行中可能暂时指向上一轮。单轮诊断没有学习曲线，因此不生成performance.png。历史结果不批量改写。

## 评测JSON

顶层为`round/train_pool/train_samples`以及：

- `real_mutants`：126株真实定义的ODE参考。
- `synth_test`：与训练来源相同的固定合成测试集，例如LHS实验对应LHS test。
- `cross_pool_test`：另一个来源的固定测试集，只作为跨域诊断。

每部分含`count`、`all_39_vars`、`core_9_vars`、`trajectory_metrics`和`pattern_metrics`。合成部分另有`source`。Fission使用`all_16_vars`，不添加Budding core9或六分类。

`all_39_vars`保留历史`MAE/MSE/Relative L2/Correlation`定义。`trajectory_metrics.macro_trajectory_rel_l2`是逐样本整条轨迹的Frobenius相对误差均值，与历史按变量、eps=0.1的`Relative L2`不同，两者不能混称。计算函数没有改变，导出仅移除重复的all_variables别名。

## 每条测试样本CSV

real表包含名称、ABC、观察窗口六分组、分裂次数以及MAE/MSE/rel_l2/correlation/trajectory_rel_l2。

同来源合成测试表另外包含`source/pool_row/source_candidate_id`和源池实际保存的生物标签、L/模块数、mutation描述（有该字段时）。CSV行顺序严格对应该来源固定test划分。LHS没有mutation标签时不会人为生成mutation身份；没有生物标签时也不由窗口六分类推断生物存活性。

## 图片与loss

Budding每图5行×8列展示39变量；Fission按自身16变量布局。标题包含样本、实际可得Pattern/窗口分组、seed/模型/方法/轮次/训练量、四项物理误差和规范突变参数。Budding使用突变体编号匹配规范规则，避免文件名空格转下划线后误标WT；未知身份明确显示无法匹配。

每轮loss复用原有TrainingCurveLogger的双轴、颜色、图例和统计框：左轴归一化训练目标及合成验证MSE，右轴物理验证macro trajectory RelL2；附学习率小图、最佳验证epoch和warmup区间。测试只在轮末评测，没有伪造逐epoch的real/synthetic test曲线，也不增加测试反馈到选样流程。根loss将实际各轮以相同风格按顺序排列。

## 恢复与存储

周期性resume保存完整训练状态；完成训练时先写history/图/模型及原子training.json，成功后才删除完整resume。已完成轮由轻量model和history恢复，不重新训练。选样清单在揭示新增真值之前保存，重启沿同一有序ID继续；源代码/协议/输入/权重身份变化会拒绝混跑。

轨迹仅在两个源池中保存一次。训练清单、selection和最终eval链接都不复制样本；辅助ensemble仅保存在_state。静态实验和旧在线流程仍保留其历史输出格式；公共图形渲染与指标函数共用。

## 本次验证

已通过74项不同的训练/接口/采样/资源及输出回归：包括五轮冷启动、全部已实现方法的小规模流程、DeepONet通用训练与eval.py标准checkpoint重载、独立baseline及资源锁、Fission指标隔离、离线W&B、逐样本CSV、五轮图和单轮不生成性能图。恢复测试在第6轮epoch开始时主动中断，继续后最终权重、optimizer、scheduler、全部epoch历史与不中断训练逐值一致；完成轮没有resume也不再重训。绘图和接口最后补正又做了定向复核。

真实结果格式预览：
`/home/users/zsy/mydata/budding_yeast/resource_profiles/output_redesign_20261114_preview/`。

该预览复用**之前3 epoch的资源测试模型**，并非正式训练成绩；历史探针当时仍采用both-pool验证，预览loss保留其真实记录，不能当作当前source-only协议的收敛曲线。已检查126张PNG均完整、126行real CSV、5000行mutation test CSV的来源及唯一行号、三类测试JSON以及无轨迹副本。原资源测试及旧静态实验目录保留原样。

## 结果字段与日志阅读（本批完成后的展示调整）

本批8项实验均已完成。40份逐轮JSON统一排序为：轮次/训练来源/训练量 → real_mutants → synth_test → cross_pool_test。仅调整键顺序，不修改任何指标值。

- `real_mutants`：126株规范突变体的参考ODE轨迹。
- `synth_test`：与训练来源相同的独立5000条测试轨迹；不参与训练、初始化选样或checkpoint选择。
- `cross_pool_test`：另一个池独立5000条测试轨迹，用于跨域诊断；mutation模型评LHS，LHS模型评mutation，不是合并训练后的测试。
- `all_39_vars`：全部变量的历史指标；`core_9_vars`是MASS/CLN2/CLB2/CLB5/SIC1/CDC20/CDH1/ORI/SPN。
- `trajectory_metrics.macro_trajectory_rel_l2`：先对每个样本计算整条39×T轨迹的相对L2，再让每个样本等权平均。macro不表示宏观变量。`all_39_vars.Relative L2`则先把每个变量的全部样本和时间拼接计算误差，再平均39个变量，并采用历史eps=0.1能量下限；二者不能混称。
- `median_rel_l2/p90_rel_l2/p95_rel_l2`：上述逐样本误差的中位/90分位/95分位；`worst_10pct_mean_rel_l2`为误差最大的10%样本的均值。`global_frobenius_rel_l2`先整体拼接再作范数比，较大幅值的轨迹权重更高。
- `valid_channel_correlation`：逐样本逐变量、沿时间的相关系数，仅对非退化通道平均。`undefined_channel_correlations`统计参考或预测变化近零、相关系数不可靠的样本—变量通道数量，不是失败样本数；它与历史按变量整体拼接的Correlation口径不同。
- `pattern_metrics.six_group_confusion`：固定顺序P1/P2/P3/P4/P5a/P5b；行是参考轨迹分组，列是预测轨迹分组。由轨迹套用窗口规则得到，不是分类头概率或生物存活性标签。对角线表示分组一致。
- `pattern_metrics.division_count_mae`：按MASS下跳>0.1、相邻检测点间隔>10步的既定规则识别分裂，计算逐样本预测与参考分裂次数的绝对误差均值，单位是“次”。

本批mutation/random最终轮示例：real MAE=0.1909165、MSE=5.3475738、历史Relative L2=0.2263703、Correlation=0.9391748；逐轨迹macro RelL2=0.3438502。六分组一致65/126=51.59%，分裂次数MAE=0.5634921。连续值总体指标与事件/阈值分组衡量的侧重点不同。

该模型跨LHS的MAE约1.206e6，属于严重跨域失效。只读核对最终checkpoint及LHS测试输入：mutation训练统计有90个近常量参数（p_std约1e-6），LHS同时扰动这些参数，标准化后的测试输入最大绝对值约2.50e7。这是分布与标准化支撑域不匹配的直接证据，跨池分数应与同池及real分开解释；本次没有更改归一化或重训。

日志默认只给START/END/FAIL/WAIT/PROGRESS/COMPLETE等显式阶段事件加时间戳；普通库输出和EPOCH不自动加时间。epoch打印6位小数、学习率科学计数，去掉val_error和validation_metric的重复显示，CSV/W&B仍保存原字段与精度：

```text
[EPOCH] model training | round=0 member=0 epoch=4/100 train_loss=1.168341 val_macro_rel_l2=0.832071 val_mse_norm=1.003208 lr=4.00e-04 optimizer_steps=64
```

round=0为第一轮、member=0为主模型；train_loss为归一化空间训练混合目标，val_mse_norm为归一化验证MSE，val_macro_rel_l2为物理空间逐样本轨迹相对L2均值。optimizer_steps是本轮累计参数更新次数，示例2000条、batch128共16步/epoch，4个epoch为64步；不是64个样本。

本批8份完成日志已同步精简，5600条epoch记录保留；逐份原始日志以`_state/train.original.log.gz`无损压缩保留，格式化前后哈希和JSON变更账本存于`_state/report_display_changes.json`。历史config内源代码哈希仍对应实际执行时版本，不伪造为新代码执行结果。13项针对日志、报告与恢复的回归通过。

## 2026-11-15：恢复逐epoch的完整测试监控与五条loss曲线

用户要求参照旧MaxDist实验的all_rounds_loss.png。11月14日版本只有训练/验证逐epoch、测试仅轮末；**从本次修改起，主模型每个epoch评测同池固定test以及冻结real reference**。跨池test仍只在轮末评测。下列规则替代前文“没有逐epoch测试曲线”的历史描述。

只读监控独立位于`utils/training/monitoring.py`。流程适配层提供同池test和real的CPU引用，分batch编码，不创建另一份整池轨迹/编码缓存。没有梯度，保存并恢复模型模式以及Python/NumPy/Torch/CUDA随机状态。归一化仍只由累计合成训练集拟合；测试指标不会参与梯度、学习率、停止条件、checkpoint选择或采样评分。辅助ensemble成员不重复评测这两组test，以控制成本。主模型checkpoint仍按同池验证macro trajectory RelL2选择。

每epoch格式如下（数值仅用于展示格式，非新实验成绩）：

```text
[Epoch 036/100] train_loss=0.089688 | val_synth=0.103367 | test_synth=0.083166 | test_real=0.201523 | test_real_denorm=0.310472 | val_macro_rel_l2=0.295314 | lr=6.94e-04 | 33.0s | round=00 member=0 steps=576
```

- `train_loss`：训练所用混合目标，归一化空间；保持现有目标不变。
- `val_synth`：同池验证集的归一化MSE。
- `test_synth`：同池固定测试集的归一化MSE。
- `test_real`：126株冻结real参考的归一化MSE。
- `test_real_denorm`：物理尺度的历史Relative L2，先逐变量汇总所有样本和时间再平均，保留eps=0.1能量下限。不是MSE，也不是逐轨迹macro口径；在同一epoch模型上与JSON的all_39_vars.Relative L2一致。
- `val_macro_rel_l2`：实际用于选checkpoint的物理验证指标；保留它是为了说明模型选择依据。
- LR、epoch耗时（包括验证和只读test监控）、round/member及累计optimizer steps保留。仅重要阶段有时间戳。

history.csv和本地W&B新增/保存`test_loss_synth/test_loss_real/test_real_denorm_rel_l2`。标量汇总不保存逐epoch预测数组或额外checkpoint；相对L2先汇总各batch的每变量误差/参考能量，再统一计算，避免直接平均batch比例造成尾batch权重错误。

单轮loss.png恢复旧图的五条主线：蓝色训练目标、绿色验证MSE、黄绿色同池test MSE、橙色real MSE共用左侧归一化对数轴；红色real物理Relative L2使用右轴（无量纲）。沿用原来的线型、marker、合并图例、左下统计框；学习率小图不再覆盖主曲线，LR继续完整记录在日志/CSV。标题含round/训练样本数、模型/损失/来源/方法/seed和每轮归一化说明。

统计框包含：最佳验证MSE及epoch、最后训练目标、最后同池test MSE、最后real MSE、最后real物理Relative L2。新完整池流程还显示真正的checkpoint选择指标（validation macro RelL2）及其epoch，并以细虚线标出。最佳验证MSE与实际checkpoint标准可能不同，均如实标注，不以最低real误差选择模型。

根loss.png按round_00→round_04排列五个同风格子图；每轮自己的loss.png相同。测试监控每epoch多做同池5000条+real126条推理，因此耗时会增加；不逐epoch画126幅预测图，不评跨池test，不对辅助成员重复这些测试。

**历史数据限制**：已完成的这批8项没有逐epoch test值，也没有每epoch checkpoint，不能回填真实测试曲线。本次不重跑、不伪造、不覆盖其history.csv。绘图预览位于`/home/users/zsy/mydata/budding_yeast/resource_profiles/epoch_monitor_preview_20261115/`，使用用户指定旧MaxDist实验的真实training_history.csv重绘；它有4轮，预览也保持4轮。新协议实际输出5轮。provenance.json记录来源及哈希，预览不代表新训练成绩。

后续重新跑正式实验时使用新的结果输出根目录（例如`--output /home/users/zsy/mydata/budding_yeast/experiments/budding_sampling_epoch_monitor`），继续引用原冻结protocol/pool/split；不要在既有完成结果目录混用新旧代码来源。本次没有启动正式重跑。

验证结果：29项针对监控、日志、图形、标准checkpoint及恢复的检查通过；另10项batch/多轮流程检查通过（含四/五轮Random、五轮Uncertainty及LCMD/BAIT等选择流程）。其中显式验证：16/39变量、float32/float64参考轨迹、短尾batch能量汇总、监控前后模型模式和三种RNG恢复、主模型训练参数/验证历史/最佳checkpoint逐值不变；aux不评test；五轮图各有五条主线且红线取real历史Relative L2，不再用验证macro替代。预览PNG完整性检查通过。

注意：日志和曲线的每epoch点对应当时epoch的模型；统计框Final为末epoch。最终eval JSON加载的是验证集选出的最优checkpoint，因此末epoch的real误差不保证等于最终JSON。比较口径时须对齐同一checkpoint/epoch。
# 2026-11-16：新增末epoch纯权重

用户确认每轮除归一化validation-best外保留末epoch权重。公开`rounds/round_XX/model.ckpt`仍为best；额外末权重位于`_state/round_XX/primary/last_weights.ckpt`，辅助成员使用`_state/round_XX/ensemble_member_N/last_weights.ckpt`。独立调用共享Trainer而未指定state_dir时为`<round>/_state/last_weights.ckpt`。

末权重可由公共checkpoint loader及eval.py读取，保存实际该轮预处理统计，无optimizer/scheduler/恢复callback状态；不用于热恢复，临时resume仍在完成后删除。训练元数据包含best_epoch和末权重路径/epoch/SHA256，复用完整轮次前检查末文件存在且哈希一致。现有已完成历史实验的last无法凭空补出，新实验开始生效。当前FNO每份约28MiB，五轮主模型约额外140MiB；ensemble方法额外保存辅助成员末权重。没有重复保存样本或轨迹。
