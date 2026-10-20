# 主动学习实验中的 Pattern 标签比较

> **后续采用方案（2026-10-21）**：用户确认保留BIO，六分类作为中性的补充分组接入生成、读取、AL快照和评测旁表。
> ABC继续兼容现有训练；未启用新六分类头、分组补采或临界标记。以下“不改代码”的表述描述当时的分析阶段。
> 现行字段与兼容边界见 [数据集标签说明](4_DATASETS_AND_NORMALIZATION.md)。


**六分类对绝对误差风险有诊断价值，但没有全面优于现有标签；当前临界标记没有显示额外收益。保留原有生物存活标签。**

本轮读取 `experiments/al_pure_random_1m/biom_refiner_plus` 的现存结果，未修改实验、数据、生成、采样、损失或训练代码，也没有重新训练模型。

## 1. 比较范围

- 10种策略目录，其中第1轮有8种，第2轮有7种可比较。`bifurcation`只有第1轮，`event_entropy`、`fisher_oed`只有共同的第0轮。
- 10份第0轮评测按名称对齐后数值完全相同，只保留1份；7份最终评测与轮次评测重复，不重复计数。排除 `seed_42_failed_v1`，主分析共16份非重复评测。
- 主分析采用7种策略的第2轮，分别为random、uncertainty、diversity、novelty、residual、residual_diversity、bifurcation_entropy。
- 实际合成训练量为29,247–37,996，不是同预算算法竞赛；全部为seed42，共享初始模型与126个真实条件，不能当作独立seed重复试验。
- 全量读取25个物理训练快照，共465,598行；累计快照有重复前缀，跨策略也共享样本，这不是独立样本数。每个快照的126条real目标按名称与整条轨迹SHA256核对，均与canonical MA完全相同，且不计入训练覆盖数。
- 用逐mutant CSV的MAE/MSE/Relative_L2，未将JSON的pooled relL2/correlation混成逐mutant平均。BIO标签取canonical MA；AL快照本身只存有原pattern字段。

## 2. 六分类在哪些方面更好

只用标签预测一个真实条件的误差风险：每次评测内将MAE、MSE、relL2各转为0–1名次，三者等权平均为综合指标。留出目标AL算法，同时按突变体身份做5折留出、重复10次；测试突变体在其他算法中的误差也不得参与拟合。

采用组均值及固定的父组收缩；另检查0/2/5/10收缩强度。未细分的P1/P2严格继承A/B预测，防止额外正则化伪造细分类收益。比较ABC、BIO、ABC×BIO、六分类、六分类×BIO，以及附加临界标记。

下面是**误差名次预测RMSE的改善百分比**，正数更好，负数更差；**不是代理模型重新训练后的误差下降**。

| 比较 | MAE风险 | MSE风险 | relL2风险 | 综合风险 |
|---|---:|---:|---:|---:|
| 六分类 vs ABC | +5.43% | +9.16% | −1.60% | +2.09% |
| 六分类×BIO vs ABC×BIO | +3.13% | +5.61% | −1.58% | +0.66% |
| 六分类+临界标记 vs 六分类 | −1.01% | −0.79% | −1.21% | −1.11% |

126个突变体聚类重采样得到的条件95%区间：

- 六分类 vs ABC：MAE改善1.51%–9.44%，MSE改善3.85%–14.50%；综合为−0.91%至+5.05%，不能确定总体获益。relL2恶化0.68%–2.57%。
- 保留BIO再更换pattern：MSE改善1.39%–10.12%；综合为−1.79%至+3.09%，仍不能确定总体获益，relL2恶化0.59%–2.54%。
- 六分类单独对现有ABC×BIO，综合差0.17%，区间−3.79%至+3.20%，不能证明胜出。

在ABC内随机打乱六分类、保持组大小的499次条件置换中，MAE/MSE的p均为0.002，主综合p=0.014，说明绝对误差诊断收益不只是任意增加类别。保留BIO的补充对照，综合p=0.092。第1轮也表现为绝对误差风险改善、relL2风险变差。

这些区间条件于当前7种策略与交叉验证预测，没有重新训练bootstrap模型；分类阈值又是在同一126基准上提出的，因此不能当成独立外部验证。主比较为六分类对ABC的综合风险，其余指标和保留BIO组合为补充分析。

此外，仅靠六分类在严格留出检验中挑20条，平均预期命中真实Top20约3.39条；ABC约2.27条；均匀随机挑20条的预期为3.17条。**分类能解释部分组间差异，定位具体坏例的能力仍然有限。**

## 3. 哪些坏例始终没解决

按每种策略的MAE、MSE、relL2平均名次取20条，再看跨策略频次：

| 条件 | 第2轮进入Top20的策略数 | 六分类 | relL2中位数 |
|---|---:|---|---:|
| #15 | 7/7 | P3 | 8.1107 |
| #124 | 7/7 | P5b | 0.1621 |
| #22 | 6/7 | P2 | 0.6318 |
| #42 | 6/7 | P2 | 0.4969 |
| #16 | 5/7 | P4 | 0.4152 |

#15的relL2范围为6.11–9.00，但同P3另6个条件的各自中位relL2只有0.0089–0.0898。各策略已有1,696–7,393条P3，其精确零参数面支持数15–62；不能简单归因为“没有P3数据”，也不能把P3都当成一样困难。

#124的精确零参数面支持数在7种策略中全部为0。residual有9,457条P5b、random有2,978条，前者#124的MSE却为43.15、后者9.51。这里只说明类计数不足以描述局部覆盖，不能据预算不齐的观察推断算法因果优劣。缺少精确0不等于近零样本无用，仍需看参数/初值距离与扰动响应。

新P2与原B完全相同，无法细分#22/#42与其他单次分裂条件；P1也与原A相同。P5b在这些评测中的中位MSE为1.414、P5a为0.032，但P5b中位relL2只有0.0367，说明它主要刻画大幅值绝对误差风险。

临界标记仅7个real为True，7种策略的140个“策略×Top20”记录命中10个（7.14%）。覆盖率低本身不证明标记物理错误，因为标记人群也只占7/126；更直接的问题是附加该标记后风险预测恶化。不要将这个新布尔标记与现有名字含bifurcation的AL算法混淆，历史实验没有使用本轮新标记。

## 4. 生物学解释需要修正

- P2先叫“观察窗口内单次分裂”。上轮47个P2延长至630分钟后全部再分裂，其中45个质量仍≤10；“单次后衰竭”不成立。
- P4的5个real中，#37/#77/#110的ORI_max仅0.712/0.644/0.540，未达到模型的DNA起始事件阈值1；不能叫“DNA已复制”。#16/#35确实越过ORI=1。起始也不等于复制完成。
- #26仅因终点Esp1低被归P5a，但Clb2最大仅约0.147；不能据此断言它已进入M期并退出失败。
- P1的21条中有3条原事件顺序失败、6条未出芽分裂；“多次分裂”不能替代存活判定，BIO字段应保留。

论文阈值与长时积分依据见 [先前生物学核验](BUDDING_PATTERN_LABEL_AUDIT_2026-10-20.md)。当前更适合将六类视为明确阈值的**窗口行为分组**，另外记录真实ORI/SPN/BUD事件、事件时序/缺失、幅值、尾段变化和输入邻域覆盖。高Clb2/高ORI应是跨所有分裂类别的附加标记。

对#15这类输出与简单G1相近、输入条件却难以学习的样本，还应检查参数条件化、激活门控和局部敏感性。不能靠不断拆真实输出类别来保证解决输入到轨迹的映射问题。

## 5. 复现与证据入口

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/users/zsy/.conda/envs/neuralop/bin/python
OUT=budding_yeast_v2/scripts/analysis/outputs/al_pattern_utility_2026-10-20
$PY budding_yeast_v2/scripts/analysis/dataset_diagnosis/audit_al_training_patterns.py \
  --output-dir "$OUT/training_snapshots" --workers 4
$PY budding_yeast_v2/scripts/analysis/dataset_diagnosis/compare_al_pattern_labels.py \
  --output-dir "$OUT/label_comparison" --permutations 499 --repeats 10 --bootstraps 5000
$PY budding_yeast_v2/scripts/analysis/dataset_diagnosis/summarize_al_pattern_utility.py \
  --analysis-dir "$OUT" --plot
```

- [完整报告](../scripts/analysis/outputs/al_pattern_utility_2026-10-20/REPORT.md)
- [跨策略稳定Top20](../scripts/analysis/outputs/al_pattern_utility_2026-10-20/label_comparison/persistent_top20.csv)
- [每次评测各自的Top20](../scripts/analysis/outputs/al_pattern_utility_2026-10-20/label_comparison/top20_each_evaluation.csv)
- [重点坏例的训练覆盖与误差](../scripts/analysis/outputs/al_pattern_utility_2026-10-20/key_badcase_coverage_and_error.csv)
- [新旧标签比较图](../scripts/analysis/outputs/al_pattern_utility_2026-10-20/label_utility.png)
- [训练覆盖和#15误差图](../scripts/analysis/outputs/al_pattern_utility_2026-10-20/training_coverage_and_mutant15.png)

训练覆盖均为快照行数，未按完全相同的参数/初值去重；新增行由累计快照前缀一致性核验后统计。具体来源/目标指纹、比较方法、逐策略结果、收缩敏感性、条件区间和同大小类别置换均保留在输出目录。

本轮使用真值pattern作分组，尚未计入未仿真候选的pattern预测误差。要证明它真正提高AL效果，还需相同初始集/候选池、相同成功ODE标注预算、相同归一化政策的标签对照训练，多seed及未参与标签设计的独立条件；现有结果不能代替这个实验。

29项针对性测试通过，覆盖算法/突变体同时留出、父子分组继承、稀疏组回退、同分随机处理、共同初始模型去重、失败运行排除，以及原六分类边界和数据只读契约。
