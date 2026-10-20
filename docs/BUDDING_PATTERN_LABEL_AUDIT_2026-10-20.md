# Budding Pattern 标签核验：窗口分类、机制证据与 20 个 Badcase

> **后续采用方案（2026-10-21）**：用户确认保留BIO，六分类作为中性的补充分组接入生成、读取、AL快照和评测旁表。
> ABC继续兼容现有训练；未启用新六分类头、分组补采或临界标记。以下“不改代码”的表述描述当时的分析阶段。
> 现行字段与兼容边界见 [数据集标签说明](4_DATASETS_AND_NORMALIZATION.md)。


本轮只新增可复用分析工具与统计旁文件，不改原始 NPZ、生成规则、训练标签或模型。结论优先于用户提供的分类器注释：**有序决策树的形式完备性成立，但其中一些动力学命名和分岔解释不成立。**

## 1. 当前已经在用什么标签

来源为 `data/systems/budding/phenotypes.py`，继承 `mutant_aware.py`：

| 字段 | 现有取值 | 含义 |
|---|---|---|
| `labels` | `Viable`；`Inviable: G1 Arrest`；`Inviable: M-phase Arrest`；`Inviable: Unbudded Division`；`Inviable: Premature Division`；`Inviable: Sequence Failure` | 依据事件顺序、出芽、分裂质量等规则判定；与动力学分组是两个维度 |
| `pattern_labels` | `Pattern_A: Multi-Division (>=2)`；`Pattern_B: Single-Division (==1)`；`Pattern_C: Exponential-Arrest (==0)` | 观察窗内质量跌落次数；C 的名称本身也不证明渐近阻滞 |

`Inviable: Numerical Failure`、`Discarded_Mass` 属于求解/过滤状态，新导出流程不会把失败样本当成正常轨迹写入。真实基准是 **126 个基因型/培养条件的模型仿真轨迹**，不是 126 条直接测得的实验时间曲线。新六分类本轮尚未替换 DataModule 的 A/B/C 编码及分类头。

## 2. 如何复现

在项目父目录运行，解释器使用 `neuralop` 环境：

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/users/zsy/.conda/envs/neuralop/bin/python
OUT=budding_yeast_v2/scripts/analysis/outputs/six_pattern_audit_2026-10-20

$PY budding_yeast_v2/scripts/analysis/dataset_diagnosis/audit_six_patterns.py \
  --data-dir /home/users/zsy/mydata/budding_yeast --output-dir "$OUT" --workers 8
$PY budding_yeast_v2/scripts/analysis/dataset_diagnosis/probe_six_pattern_horizons.py \
  --output-dir "$OUT/horizon_probes" --workers 6
$PY budding_yeast_v2/scripts/analysis/dataset_diagnosis/probe_pattern_training_neighbors.py \
  --audit-dir "$OUT/mutant_aware_pure_random_100000_210m_500s" \
  --output-dir "$OUT/training_neighbor_probes"
$PY budding_yeast_v2/scripts/analysis/dataset_diagnosis/summarize_six_pattern_audit.py \
  --audit-dir "$OUT" --output-dir "$OUT/badcases_hybrid_mse_rel_seed42" --plot
```

默认 badcase 来源为用户指定的 `biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/per_mutant_metrics.csv`。可用 `--eval-csv` 显式切换；不能把其他 loss 的数值混入这一排名。已有统计可用 `--resume` 复用，原文件路径、大小、修改时间、分类器版本、拆分 seed 必须一致；评测 CSV 改变会重新按名字关联。

工具直接流式读取 ZIP 中的 `data.npy`，并检查全部 39 维是否有限；`np.load(..., mmap_mode='r')` 不会使压缩 NPZ 的 `data` 成为真正的内存映射。支持 N×39×T 与 N×T×39，拒绝维度歧义；NaN/Inf 单列为 INVALID，不允许兜底成 P3/P4。

每次质量跌落比较的是**上一个检测到的跌落**，即使那个跌落被 10 步去抖抑制，也更新参考位置。这与原生产代码、用户附件一致。旧分析脚本 `analyze_dynamic_phases.py` 的去抖参考更新位置不同，本次不使用它，也不假定 real 总在文件前 126 行。real 优先读 `is_real`，缺失时按数字编号识别；评测按名称一对一关联。

## 3. 六分类的形式正确，不等于生物学解释正确

本次执行的是用户最后给出的**有序决策树**，版本 `user_six_patterns_v1_2026-10-20`，不擅自把前面文字列表的不同条件混进来。

| 编码 | 目前能够确认的观察含义 | 暂不应宣称的含义 |
|---|---|---|
| P1 | 当前窗口检测到 ≥2 次分裂 | 已证明自持稳定极限环 |
| P2 | 当前窗口检测到 1 次分裂 | 单次分裂后衰竭、母体蛋白耗尽 |
| P3 | 零分裂且未触发优先的 Clb2/Esp1 分支，ORI_max<0.5 | 全系统休眠、不动点、永久 G1 阻滞 |
| P4 | 同上，但 ORI_max≥0.5 | 一定已启动/完成 DNA 复制、一定 SPN<0.2 |
| P5a | 零分裂、命中 Clb2_end/Esp1_end 优先条件且不是 P5b | 一定已经进入 M 期且退出受阻 |
| P5b | 零分裂且 Clb2_max>10 或 Clb2_end>5 | 数学有限时奇点、该组一定主导总误差 |

对维度正确、数值有限的输入，该 if/else 树每行只返回一类，因此互斥、完备。**前面的独立文字谓词并不 MECE**：P5b 与 P5a 原始谓词重叠；低 Esp1 还可同时满足 P3/P4 的文字条件。当前 126 个 real 中文字谓词有 22 个重叠，虽然没有空缺；决策优先级将这些重叠隐藏了。

在合成数据中，树的兜底 P4 还会收进 SPN≥0.2 或过去 Clb2≥0.35 的样本。统计中保留 `p4_prose_inconsistent`、`prose_rule_matches`，逐行区分“执行代码分到 P4”和“满足 P4 文字机制”。例如 #26 的 Clb2_max≈0.1469、SPN_max≈0.0764，仅因终点 Esp1≈0.0114 被分为 P5a；不能据此证明它曾进入有丝分裂。

## 4. 论文实际支持哪些阈值

- **Chen 2004，Table 1、pp.3847–3848 的 Timing of Cell Cycle Events / Assay of Model Robustness**：ORI=1 表示 DNA 合成起始；SPN=1 表示染色体在纺锤体上排列完成；BUD=1 表示出芽。ORI 是辅助累积变量，不是 DNA 含量或复制完成率。Clb2 向下跨 0.3 触发退出/分裂，Clb2+Clb5 向下跨 0.2 触发起点重新许可。
- **Chen 2004 补充材料 p.4–5、p.10 附近的 C 节**：Esp1=0.1 讨论的是**分裂时刻**是否充分激活，作者还讨论了这一存活标准可能过严。它不是“任意 210 分钟终点 Esp1<0.1 就已进入 M 期”的证明。
- **Chen 2000 的 Discussion 与 checkpoint 机制**：支持双稳态开关与 START/FINISH 的机制背景，但不提供当前六个轨迹分箱及全部数值门槛。
- **Ahmadian 补充材料、`full_supplementary_info.md`**：Table S5 可核对基因型、条件和存活结果；混合随机模型包含额外 mRNA/总量变量和分子数换算，不能将其分子数门槛直接套到当前 39 维浓度轨迹上。

因此 0.35、ORI 的 0.5–1.5、SPN 的 0.2、Clb2 的 5/10 等，当前应记录为**待校准的工程阈值**，不能标注成论文已经验证的普适相态边界。论文对近周期性还要求连续分裂状态的 RMS 差异足够小（<0.05），没有采用“分裂两次即证明极限环”的推理。

资料：[Chen 2004 作者 PDF](https://jigcell.cs.vt.edu/Pubs/BudYeast.pdf)、[Chen 2000 作者 PDF](https://jigcell.cs.vt.edu/Pubs/Chen2000.pdf)、[本地 Chen 补充材料](../../papers/e03-11-0794_chen_suppl.pdf)、[Ahmadian 2020](https://doi.org/10.1038/s41540-020-0126-z)。

## 5. 延长观察窗：P2 不能叫“衰竭”

对全部 126 个 canonical MA 条件，使用相同参数、初值处理、ODE、BDF 容差、事件重置和 5 秒单条超时，分别积分 210/420/630 分钟，共 378 次；保持相同采样时间间隔，全部完成。质量跌落计数与求解器 event1 数目在每次积分中均一致。

| 条件 | 210 分钟 | 420 分钟 | 630 分钟 | 直接证据 |
|---|---|---|---|---|
| WT 葡萄糖 | P1，2 次 | P1，4 次 | P1，6 次 | 正常参考 |
| WT 半乳糖 | P2，1 次 | P1，2 次 | P1，3 次 | 第二次分裂在 291.19 min |
| #22 | P2，1 次 | P1，3 次 | P1，5 次 | 第二次分裂在 305.79 min |
| #42 | P2，1 次 | P1，2 次 | P1，3 次 | 第二次分裂在 320.47 min |

210 分钟时的 **47 个 P2，到 630 分钟全部成为 P1**。其中 45 个长时最大质量仍≤10；另外 2 个超过论文的质量存活上限，不能将数值分裂解释为生物可存活。即使只看 WT、#22、#42，也足以否定“P2 必然衰竭”。

#15 在 210 min 是 P3、到 420 min 数值分类变为 P5b；420 min 时质量约30.6，已超过生物有效范围，这一延长结果只能说明窗口依赖性，不能据此更改它的存活真值。#124 的 Clb2 与 ORI 随质量继续增大，并没有从这些有限数值观察到数学奇点。

## 6. “临界”标记是启发式，不是分岔或梯度的证明

完全按用户 START/FINISH 公式计算。canonical 126 条中仅 **7 条为 True**；按本轮三个误差指标综合挑选的 20 个 badcase 中仅 **1 条为 True**，#15/#22/#42/#124 均为 False。它可与 pattern 交叉统计，但“正交”只表示另设一个字段，不意味着统计独立或无需验证的物理真实性。

将全部分箱/标记门槛同时上下浮动 10% 作稳定性检查：126 条 real 的临界标记分别有 20/3 条翻转。特别是默认初始 Clb2≈0.1469227 非常靠近人为下限0.15，使用全时最大值容易受初值残留影响；必须同时记录初值、动态峰值和去掉初始过渡后的结果。

浓度落在区间内不等于处于鞍结分岔。要证明分岔，需要选定控制参数、追踪平衡/周期解及其稳定性；增长质量和事件重置还要求区分冻结质量子系统与混合系统的返回映射。Jacobian 特征值趋零是特定平衡点分岔的证据之一，不是这些阈值可直接推出的结论。还存在 Hopf 等其他分岔。[原始分岔分析](https://arxiv.org/abs/q-bio/0404006) 对 Chen 2000 及简化模型采用了延拓、稳定性分析；不能把它当作当前39维模型这些区间的验证。

布尔标签也不会自动“瞬间激活相变梯度”。硬阈值对输入几乎处处梯度为0，在边界不可导。若它来自真值并用于加权损失，例如 `(1+λ·flag)·L`，只能改变已有损失梯度的权重；若需要可微边界学习，应另外设计连续边界距离/敏感度损失并验证。

## 7. 主动学习建议

保留三类信息，避免把一种划分承担所有任务：

1. **观察事实**：窗口长度、培养生长速率、分裂数/分裂时间、ORI/SPN/BUD 真实事件、事件顺序、分裂时 Esp1；P1/P2 先改用中性的“窗口多次/单次分裂”名称。
2. **难度与覆盖**：Clb2/ORI 等逐变量幅值、增长率、尾段变化、训练输入邻域、精确零参数面的覆盖。高幅值是所有分裂类别的独立标记，不能只在 v=0 的 P5b 分支发现。
3. **待验证机制**：延长观察后是否恢复分裂、近周期状态回归、参数邻域扰动是否改变事件时序/序列/分支。把未验证的临界程度作为候选评分，避免将其当成真值物理证书。

类别非空不等于具体基因型/初值邻域有覆盖；“口径3零条”必须先明确是哪个参数支持集与容差，不能由某个大类占比推出。输出同时给出每个 real 的同 pattern 样本数、同 pattern×临界标记样本数、全输入最近邻、动力学特征最近邻和精确零参数面支持数。

本轮只是对既有评测做事后诊断。用这126条真实测试坏例设计 AL 策略之后，应另设独立验证/测试条件；否则后续提高只能解释为针对该基准的适配。尚未仿真的候选池没有真值轨迹，不能凭此脚本直接知道候选的真值 pattern；需要代理预测+不确定性筛选，再仿真回填标签。

全量计数、20条明细、原文件清单和逐行统计见 `scripts/analysis/outputs/six_pattern_audit_2026-10-20/`；20个 badcase 的排名取 MAE、MSE、Relative_L2 各自从差到好排名的等权平均，Correlation 只展示。保留各指标单独Top20，防止绝对误差相关性或综合排名掩盖单项极端值。

## 8. 本次完整统计与局部覆盖证据

已完成 **30 个非 LHS 轨迹文件，共 2,609,797 行**，其中合成样本 2,606,017 行；跨文件重复样本没有去重。清单另排除23个LHS文件、1个只有参数的候选池。包括旧 `real_mutants_only` 在内的历史文件也单列统计，不能用其中旧生物规则替代当前 MA 基准。

所有源文件读前/读后身份一致，未覆盖任何原始数据。非有限轨迹为0，旧A/B/C标签重算不一致为0。历史合成轨迹中有1,143行出现低于−1e−4的数值，已标记而未静默剔除；旁文件字段 `negative_below_minus_1e4` 的实际阈值是−0.0001。六分类统计仍包含这些有限数值行，不能把形式可分类当成通过物理有效性验证。

各文件同编号real的严格float32轨迹指纹并非全部相同；指纹不一致只证明数值未逐位相同，不能直接判定ODE错误。本轮保留每个文件实际轨迹分别统计，未据指纹差异重新生成旧数据；badcase关联仅使用当前pure-random的canonical目标，不将重复的real当成30份独立真值。

| Pattern | 当前 pure-random 100k | mutant-aware 100k | MASS random 100k | canonical real 126 |
|---|---:|---:|---:|---:|
| P1 | 6,329 | 6,711 | 6,039 | 21 |
| P2 | 29,124 | 29,052 | 33,530 | 47 |
| P3 | 16,788 | 18,389 | 13,968 | 7 |
| P4 | 9,313 | 6,849 | 6,154 | 5 |
| P5a | 30,748 | 30,341 | 32,412 | 33 |
| P5b | 7,698 | 8,658 | 7,897 | 13 |

pure-random合成样本中临界标记为12,304条；P4中1,331条不满足文字描述的SPN/Clb2条件。将全部非LHS合成行合计，文字谓词有35,421个空缺、426,310个重叠；这些是跨文件行数，不是独立生物样本数。决策树靠优先分支/兜底消除分类空缺，不代表文字机制对所有样本成立。

在当前 seed42 的79,998条合成训练拆分中：

| 条件 | 同pattern训练数 | 满足该real全部零参数坐标的训练数 |
|---|---:|---:|
| #15 / P3 | 13,430 | 83 |
| #22 / P2 | 23,298 | 122 |
| #42 / P2 | 23,298 | 0 |
| #124 / P5b | 6,164 | 0 |

这里“零参数面”只要求该real为0的全部坐标也为0，不要求其余参数、初值、轨迹接近。#15这83条中，另要求 `ksn2pp∈[0.6,0.9]` 的只有1条；该区间只是显式诊断口径，不能冒充已经验证的相变边界。

为避免把低维特征近似当成完整轨迹近似，额外从源文件提取了20条badcase各自的两个训练邻居、比较全部39维轨迹：

- #15的动力学特征邻居 `RANDOM_CLN2[ts]+CLN3[KO]+BCK2[KO]_L3_66736`，完整轨迹MAE≈0.000089、逐变量带0.1分母地板的平均relL2≈0.000355，与其输出轨迹非常接近。
- #15的全输入诊断距离邻居 `RANDOM_CLN2[MC]_L1_20428`，同一轨迹relL2≈8.855，动力学分支差异很大。这与“输出流形从未出现”不同，更值得检查参数/初值到轨迹的局部映射、分支区分与模型约束；尚不能据此证明具体误差因果。
- 上述邻居来自**当前重建**的训练拆分。没有历史训练成员记录时，不能证明它当时确实被该冻结模型读到。

邻居距离的定义：动力学为 `log1p(max(v,0),max(Clb2),end(Clb2),end(Esp1),max(ORI),max(SPN),end(MASS))` 的逐项变换、再按当前训练标准差标准化后的欧氏距离；全输入为原始参数与原始39维初值拼接、按当前训练标准差标准化后的欧氏距离。后者是**物理输入诊断度量**，不等于历史网络归一化后的实际输入距离；两者均不宣称全轨迹全局最近邻。

主20条排名为 **#22、#15、#124、#23、#20、#59、#26、#54、#42、#44、#62、#53、#33、#46、#24、#75、#66、#122、#60、#57**。其中P1有2条、P2有9条、P3有1条、P4为0、P5a有4条、P5b有4条；只有#23被临界标记命中。

对同模型已保存重放预测，126条目标按名称及float32整条轨迹SHA256与当前NPZ完全一致。逐通道误差证据：#124的ORI占总平方误差99.45%，#22的ORI占98.50%；#15的CLN2/ORI/BUD/Vi20约占35.11%/34.98%/12.89%/10.24%。这是独立重放预测的诊断数值，主排名仍以用户指定的原始CSV为准。

产物：[20条完整明细](../scripts/analysis/outputs/six_pattern_audit_2026-10-20/badcases_hybrid_mse_rel_seed42/top20_badcases.csv)、[全部30文件合成计数](../scripts/analysis/outputs/six_pattern_audit_2026-10-20/badcases_hybrid_mse_rel_seed42/all_datasets_synthetic_counts.csv)、[详细统计报告](../scripts/analysis/outputs/six_pattern_audit_2026-10-20/badcases_hybrid_mse_rel_seed42/REPORT.md)、[重点轨迹图](../scripts/analysis/outputs/six_pattern_audit_2026-10-20/badcases_hybrid_mse_rel_seed42/key_badcase_trajectories.png)。

验证：22项针对性测试通过，覆盖边界、去抖链、float32门槛、转置、无效数值、旧拆分一致性、按名称关联、误差排名、目标指纹和不改原文件；另与用户附件逐条对照512条实际轨迹的返回结果一致。
