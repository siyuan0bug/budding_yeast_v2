# Budding real-mutant badcase：跨静态数据集与主动学习审计（2026-10-27）

> **后续核验与结论限定**：用户提供#7–10的好例后，补做同一Random训练集对照：#8/#10严格局部计数0仍预测好；#15已有几乎相同的39维阻滞轨迹仍预测差。固定背景ODE与冻结模型对照直接显示#15的条件参数响应错误。因此下文“局部支撑不足”应理解为部分案例的关联与候选瓶颈，不能作为每个坏例的唯一原因，也不能等同于没见过类似输出。补充证据见 [好例/坏例对照](GOOD_BAD_CASE_COMPARISON_2026-10-27.md)。

本次结论是：**badcase 不能用“高阶突变难学”或“缺少某个 pattern”单独解释。更直接的证据是特定基因组合、突变机制、碳源与初值背景下的动力学支撑不足；同时存在假启动、单次分裂后衰竭、高幅值连续误差以及旧评测口径不一致。**

本报告分析已有权重、数据和评测；补做的 ODE 仿真仅用于固定背景敏感度诊断。没有重训模型、追加正式训练数据或修改采样规则。文中的 real-mutant 轨迹是相应突变体参数下保存的 **ODE 基准轨迹**，不是直接测量的 39 变量实验时序。

## 1. 分析范围与可比性

- 主分析读取 **148 份静态 per_mutant_metrics.csv + 27 份当前 AL 逐轮 metrics_denorm.json**，覆盖 28 个静态目录、26 个实际 NPZ 文件。全量逐样本指标保存在 `all_mutant_metrics.csv`。
- 每个实际静态数据文件固定选择一个代表实验，优先 BioRefinerPlus / hybrid_mse_rel / seed42 / log1p，不依据测试误差选模型；同配置优先原名目录。全部其他模型/损失实验仍分别导出 Top20，没有丢弃。
- 深度轨迹分析针对 **26 个静态代表 + 8 个当前 AL 实验最新可用轮次**。32/34 个 checkpoint 重载指标与归档相符，其中24个静态、8个AL；18个静态还具有相同架构与训练超参数。复现采用归档图结构，部分历史实验需要原 Git 版本模型代码。通过标准是归档指标与重载指标在声明容差内一致，并非逐位一致：MAE/MSE/RelL2的最大绝对差分别不超过max(0.005/0.02/0.005，各项归档最大值的1%)，Correlation不超过0.02；此1%按全体最大指标而非逐例定义，精确差值保存在replay_validation.csv。
- `rolemode`、`leaklevel` 两个静态代表的旧指标尚不能由现有 checkpoint、数据及历史代码可靠复现；保留其归档指标，但不使用重载轨迹做机理归因。
- 另外 **44 份旧 al_pure_random_1m CSV** 单独汇总为历史附录，不混入静态数据集排名；2份Fission CSV排除。即194份CSV中192份Budding均有归档统计，另加当前27份AL JSON。
- 静态数据集间并非全部同预算、同损失、同归一化，也有少数基准真值版本差异；下表不能当成严格采样算法排行榜。旧静态划分多数按原代码/seed重建，当前AL则直接使用checkpoint保存的划分与归一化。

**坏例选择：** 每个实验分别对MAE、RelL2排序，二者平均名次形成综合难度；各静态数据文件等权汇总，得到反复困难的20例。MSE、MAE、RelL2各自Top20也单独导出，因此不会只关注一个量纲。20例是代表性诊断集，不代表每一份实验里同样20例都最差。

**指标解释：** MAE/MSE是在反归一化轨迹上计算。当前归档的RelL2是逐通道 `sqrt(MSE_channel / max(mean(y_channel²),0.01))` 后再平均39通道；它不是把39通道拼接后算一次L2。故低信号通道会显著影响RelL2，大幅ORI则会支配物理MSE。下面的“相对误差贡献”和“SSE贡献”是两种不同量。

## 2. 20 个重点 badcase

为便于逐行比较，下面的数值全部来自同一份纯随机100k / BioRefinerPlus / hybrid_mse_rel / seed42 实验；Top20频次才是跨26个静态代表统计。模块数按模型功能模块定义，`cln1Δ cln2Δ`作为同一个CLN2模块，不能按名称里出现几个基因机械计数。

| ID | 基因型（文件原名） | 模块数 | 参考MAE | 参考MSE | 参考RelL2 | Top20出现/26 | 主要SSE通道 |
|---|---|---|---|---|---|---|---|
| 42 | 42_sic1Δ_cdc6Δ_2-49_cdh1Δ_GAL-CDC20 | 4 | 0.2138 | 0.6954 | 0.4436 | 26/26 | ORI |
| 124 | 124_APC-A_GAL-CLB2 | 2 | 3.304 | 557.4 | 0.3249 | 25/26 | ORI |
| 22 | 22_cln1Δ_cln2Δ_cln3Δ_multi-copy_CLB5 | 3 | 0.463 | 26.49 | 0.4776 | 24/26 | ORI |
| 20 | 20_cln1Δ_cln2Δ_cln3Δ_sic1Δ | 3 | 0.2025 | 2.603 | 0.3417 | 25/26 | ORI |
| 33 | 33_GAL-SIC1_GAL-CLN2_cln1Δ_cln2Δ_cdh1Δ | 3 | 0.1169 | 1.265 | 0.3076 | 22/26 | ORI |
| 59 | 59_CLB2-dbΔ_multi-copy_CDC6 | 2 | 0.204 | 2.31 | 0.3353 | 20/26 | ORI |
| 46 | 46_swi5Δ_cdh1Δ_GAL-SIC1 | 3 | 0.149 | 0.9386 | 0.2979 | 17/26 | ORI |
| 111 | 111_TAB6-1_CLB1_clb2Δ | 2 | 0.09721 | 0.476 | 0.1554 | 17/26 | ORI |
| 62 | 62_GAL-CLB2-dbΔ | 1 | 0.3824 | 9.9 | 0.08413 | 17/26 | ORI |
| 15 | 15_cln3Δ_bck2Δ_multi-copy_CLN2 | 3 | 0.6029 | 4.824 | 7.536 | 20/26 | ORI |
| 24 | 24_cln1Δ_cln2Δ_cln3Δ_multi-copy_BCK2 | 3 | 0.09431 | 1.146 | 0.2718 | 19/26 | ORI |
| 53 | 53_GAL-CLB2_sic1Δ | 2 | 0.1269 | 0.942 | 0.3162 | 16/26 | ORI |
| 36 | 36_sic1Δ_cdh1Δ | 2 | 0.06092 | 0.1009 | 0.1233 | 11/26 | ORI |
| 76 | 76_cdc20Δ_pds1Δ_clb5Δ | 3 | 0.08786 | 0.4665 | 0.2493 | 17/26 | ORI |
| 23 | 23_cln1Δ_cln2Δ_cln3Δ_GAL-CLB5 | 3 | 0.335 | 8.342 | 0.4702 | 18/26 | ORI |
| 16 | 16_cln3Δ_bck2Δ_sic1Δ | 3 | 0.03761 | 0.02102 | 0.1105 | 15/26 | Vi20 |
| 58 | 58_CLB2-dbΔ_GAL-SIC1 | 2 | 0.07473 | 0.572 | 0.142 | 13/26 | ORI |
| 52 | 52_CLB1_clb2Δ_pds1Δ | 2 | 0.06513 | 0.4351 | 0.1224 | 10/26 | ORI |
| 122 | 122_APC-A_cdh1Δ_multi-copy_CDC20 | 3 | 0.117 | 1.052 | 0.146 | 9/26 | CDC20T |
| 54 | 54_GAL-CLB2_cdh1Δ | 2 | 0.2599 | 2.29 | 0.199 | 9/26 | ORI |

### 2.1 每例的局部支撑与错误类型

下表仍使用上述统一参考实验。局部计数为参数盒诊断，近20同组比例则同时考虑参数及初值的近邻；它们不是同一个指标。主要SSE通道是逐例重载预测计算，不凭基因名称猜测。下列解释除专门进行ODE干预的#15/#22/#124等外，属于支持程度不同的诊断假设，不能当成已完成的因果消融。

| ID | 真/预测分组 | 同模块训练数 | 严格局部训练数 | 近20同组比例 | 首要SSE占比 | 观察解释 |
|---|---|---|---|---|---|---|
| 42 | P2/P4 | 2 | 0 | 0.45 | 0.3766 | L4超出新pool范围；多KO与初值联合外推 |
| 124 | P5b/P5b | 68 | 0 | 0.2 | 0.9945 | 高幅值或稳定性背景下ORI累计量误差；分类正确仍可错 |
| 22 | P2/P2 | 5 | 0 | 0.2 | 0.9849 | 单次分裂/衰竭瞬态与阶段切换；ORI驱动或时序误差 |
| 20 | P2/P1 | 7 | 0 | 0.75 | 0.8633 | 单次分裂/衰竭瞬态与阶段切换；ORI驱动或时序误差 |
| 33 | P2/P2 | 6 | 0 | 0.1 | 0.9278 | 单次分裂/衰竭瞬态与阶段切换；ORI驱动或时序误差 |
| 59 | P2/P1 | 58 | 0 | 0.15 | 0.8706 | 高幅值或稳定性背景下ORI累计量误差；分类正确仍可错 |
| 46 | P2/P2 | 4 | 0 | 0.05 | 0.9067 | 联合突变条件覆盖不足的关联证据；主要误差通道ORI |
| 111 | P2/P2 | 51 | 1 | 0.75 | 0.7381 | 联合突变条件覆盖不足的关联证据；主要误差通道ORI |
| 62 | P5b/P5b | 866 | 2 | 0.9 | 0.9936 | 单模块内GAL+db复合突变；高驱动/稳定性与ORI误差 |
| 15 | P3/P2 | 5 | 0 | 0 | 0.3511 | START假激活；低信号多通道相对误差；条件邻域稀少 |
| 24 | P2/P2 | 5 | 0 | 0.05 | 0.9077 | 单次分裂/衰竭瞬态与阶段切换；ORI驱动或时序误差 |
| 53 | P2/P2 | 78 | 0 | 0.9 | 0.8197 | 联合突变条件覆盖不足的关联证据；主要误差通道ORI |
| 36 | P1/P1 | 67 | 0 | 0.15 | 0.4946 | 联合突变条件覆盖不足的关联证据；主要误差通道ORI |
| 76 | P2/P5a | 7 | 0 | 0.05 | 0.7534 | 联合突变条件覆盖不足的关联证据；主要误差通道ORI |
| 23 | P5a/P4 | 5 | 0 | 0.05 | 0.9436 | 单次分裂/衰竭瞬态与阶段切换；ORI驱动或时序误差 |
| 16 | P4/P2 | 12 | 0 | 0 | 0.5779 | START/S-G2解耦条件覆盖与动力学混淆 |
| 58 | P2/P2 | 78 | 1 | 0.75 | 0.9649 | 高幅值或稳定性背景下ORI累计量误差；分类正确仍可错 |
| 52 | P2/P2 | 63 | 1 | 0.45 | 0.9216 | 联合突变条件覆盖不足的关联证据；主要误差通道ORI |
| 122 | P1/P1 | 9 | 0 | 0.8 | 0.9397 | 联合突变条件覆盖不足的关联证据；主要误差通道CDC20T |
| 54 | P5a/P2 | 60 | 0 | 0 | 0.9842 | 高幅值或稳定性背景下ORI累计量误差；分类正确仍可错 |

所有20例各有一张9变量时序图，位于 `figures/cases/case_<ID>.png`。图中黑色为同一版本ODE真值，红/绿为两份数据训练后的较差/较好预测；两份实验按该例结果事后挑选，用于展示可学习性，不能拿它估计无偏改进幅度。精确配对见 `same_truth_plot_pairs.csv`。

## 3. #15：主要是错误启动，不是“CLN2 必然很大”

该例为 `cln3Δ bck2Δ multi-copy CLN2`。当前参数C0=B0=0、ksn2pp=0.75（5拷贝）。方程为：

`dCLN2/dt = (ksn2p + ksn2pp*SBF)*MASS - kdn2*CLN2`。

多拷贝增加的是受SBF调控的合成项，不能等同于GAL的组成型表达。Cln3/Bck2与SBF/START的关系也与 [Chen 2004 原模型](https://pmc.ncbi.nlm.nih.gov/articles/PMC491841/) 的网络一致。**名义上有multi-copy并不保证此背景下CLN2大量表达。**

统一参考模型RelL2=7.536：ODE真实轨迹处于低信号阻滞，预测却生成明显CLN2、ORI信号和虚假分裂。在24份可复现静态模型中，#15预测分组有12份P1、6份P2，只有4份P3，另有P4/P5a各1份；低RMS(<0.1)通道贡献的相对误差比例中位数为91.75%。这不是仅一个大变量把总误差拉高。

固定双KO背景，单独将CLN2拷贝从2、3、4、5逐步增至20，ODE仍为零分裂P3；30拷贝时变为P5a。因此**没有证据把名义5拷贝点直接称为局部分岔刀刃**。该例更多体现未学好“相同上游关闭背景下，增加受控合成系数仍可能不启动”。

最有说服力的对照：

| 数据集 | #15 RelL2 | 全局P3训练数 | 同模块训练数 | 严格局部训练数 | 近20同组比例 |
|---|---|---|---|---|---|
| bams_50000_210m_500s | 7.247 | 5546 | 0 | 0 | 0 |
| bams_ablation_hard_50000_210m_500s | 0.008656 | 5726 | 1588 | 48 | 1 |
| mutant_aware_pure_random_100000_210m_500s | 7.536 | 13430 | 5 | 0 | 0 |
| bams_ablation_hypo_50000_210m_500s | 0.01258 | 5563 | 1578 | 0 | 1 |

BAMS50k与hard/hypo50k的全局P3数量相近，#15误差却相差约三个数量级；hard训练集中同背景及局部支撑显著增加，且没有与该real完全相同的参数+初值训练行。说明“补P3总量”不如“补正确背景的P3”直接。

**对照的边界：** `data/generate_bams_ablation_datasets.py` 的 TARGETED_SKELETONS 明确包含M15/M22/M42/M124，这是依据已知测试坏例设计的定向消融；它说明该动力学可被学会，却不能作为盲测AL优越性的证据。hypo还包含历史部分LOF模式，不能不经说明移植成当前合法突变规则。

![#15同真值对比](/home/users/zsy/mydata/budding_yeast/analysis/badcase_audit_2026-10-27/figures/cases/case_15.png)

## 4. #20/#22/#23/#24/#33：一次分裂后的衰竭与切换

这些案例共享G1启动/补偿相关背景，但不能仅因都属于P2就认为轨迹相同。初始残留分子、补偿方式（MC与GAL）、进入START的时间与退出后能否再启动，都影响210分钟窗口内的结果。

#22固定CLN2与CLN3关闭背景，CLB5从2/3拷贝增至4拷贝，ODE观察由零分裂P5a切换为单分裂P2；名义real为4拷贝，确实接近该条参数扫描上的有限时窗分界。**这是有限时窗行为切换证据，尚未计算雅可比或做连续分岔分析，不能据此证明鞍结分岔。**

24份可复现静态模型中，#20与#22物理SSE落在最后10%时间段的比例中位数约64.5%与61.3%。尾段ORI/退出时序是重要误差源；单靠峰值幅度或ABC粗分组会遗漏。

#33还包含GAL-SIC1与GAL-CLN2及CDH1背景；当前池同模块样本并不等于真正相近参数组合，严格局部盒内数量为0。#23的GAL-CLB5与#22的MC-CLB5也不能合并为同一表达规则。这些案例适合检查同背景强度扫描及尾段事件，而不是把所有P2无差别加权。

## 5. #124及#59/#62等：分类正确，连续轨迹仍错误

#124是APC-A + GAL-CLB2。在24个可复现静态代表里，真值与预测均被分为P5b，但物理SSE的ORI占比中位数为 **99.16%**。统一参考模型MAE=3.304、MSE=557.4、RelL2=0.325。因此“P5b识别准确”远不足以代表预测准确。

ORI方程是 `dORI/dt = ksori*(eorib5*CLB5 + eorib2*CLB2) - kdori*ORI`。它是受驱动并衰减的模型标记，不能把大ORI值直接解释为几百份完整基因组。ORI的SSE占比也受物理尺度影响，不等于它在生物学上比其他变量更重要。CLB2/CLB5幅值及时序误差可以持续传到ORI；独立轨迹输出头本身也可能不满足此方程。应进一步用预测残差区分驱动偏差与输出一致性，现有SSE分解不能单独证明哪一项占主因。

#62虽然仅L1，却是同模块内GAL+db双机制，绝对误差明显而RelL2相对较低；#59/#58涉及稳定性与第二模块补偿。**突变模块阶数L不编码突变机制复杂度，也不编码量纲风险。**

log1p是可逆的，不能据此声称模型“分不清大值和小值”。归一化训练目标与物理SSE关注点可能不同，但本轮没有对loss梯度做受控消融，暂不把归一化宣判为根因，也没有理由恢复大值截断。

## 6. 为什么另一些样本预测好？

在每份可复现静态实验内比较最好20例与最差20例：最近输入距离的逐实验中位数再取中位数，分别约 **0.0120 vs 0.5776**；最好20例中精确参数+编码后初值重复训练输入的数量中位数为1，最差20例为0。接近已有输入、动力学类型相符的案例通常容易；部分好结果还含训练输入重复，不能全部解释成强外推能力。输入重复只是一项下界证据，不自动证明时间开关等隐藏条件或整条输出也完全相同。

逐实验相关性汇总如下；完整p值及BH校正见CSV：

| 诊断量 | 与RelL2的逐实验Spearman中位数 |
|---|---|
| input_near20_same_pattern_fraction | -0.5768 |
| input_nn_distance | 0.4895 |
| param_nn_distance | 0.4908 |
| train_local25 | -0.4232 |
| train_same_module_set | -0.4392 |
| train_same_pattern | 0.008368 |
| trajectory_feature_nn_distance | 0.6923 |

最醒目的对照是：**全局同六分类样本数与RelL2几乎无关联，而局部相同动力学支撑关联更强。** 但这属于关联诊断，类别自身难度、参数范围及网络容量都会混杂；不能把相关系数换算成采样方法的预期收益。

轨迹特征距离使用真实ODE轨迹，属于事后oracle诊断，不能直接拿去给未仿真的AL候选打分。输入距离采用公共审计表示（参数Z-score + 初值log1p/Z），历史asinh等模型的实际编码另由checkpoint重载恢复；不能把此距离误称为所有模型真实隐空间距离。

## 7. 当前AL：候选供给问题与选择问题必须分开

### 7.1 100万池的真实局部数量

为跨当前AL实验统一口径，参数盒定义为每个非零参数：

`abs(P-P_real) <= max(0.25*abs(P_real), 0.05*abs(P_WT))`，且mdt相同。

严格盒要求real为0的坐标也严格为0；残余盒只将这些坐标放宽至1%WT。该1%是敏感度诊断口径，不是经过全系统证明的统一生物学容差；参数盒也未自动保证初值或真实轨迹相近。静态支撑表的局部盒基准宽度使用各自合成参数均值，与此池统一WT盒略有不同，不应跨表混用计数。

| ID | pool同模块数 | 其中同碳源 | 严格局部25% | 允许1%残余局部25% |
|---|---|---|---|---|
| 124 | 2963 | 967 | 0 | 6 |
| 42 | 0 | 0 | 0 | 0 |
| 22 | 578 | 154 | 0 | 0 |
| 20 | 564 | 157 | 0 | 5 |
| 15 | 549 | 185 | 1 | 1 |

例：APC-A在mutant-aware规则里本来是WT的0–1%连续残余，精确0概率为0是规则结果；#15涉及的KO则有离散零点。**当前池不是141维完全连续均匀池，不能使用“所有多KO都是零测度、永远抽不到”概括。**

固定#124背景，将APC残余从0扫到1%WT，始终P5b，CLB2峰值31.95→30.99，ORI峰值445.08→431.63。#20将CLN2残余从0扫到20%WT仍为P2。由此可见至少这些扫描中有限邻域能提供有用支持，精确零端点缺失不等于整片动力学真空。

#42是唯一L4组合，在26/26个静态数据文件中均进入综合Top20；当前L≤3池中同组合为0，这是已选择的研究域边界。加L3占比或换评分器都不能补出L4。保持现行规则时，应单列OOD结果；要纳入同域覆盖需后续明确允许L4，**本轮未改该决定**。

### 7.2 最终训练集究竟拿到了多少邻近样本？

| 策略 | 训练数量 | ID | 已选严格局部 | 已选允许1%残余局部 |
|---|---|---|---|---|
| al4pde_bait | 40000 | 124 | 0 | 0 |
| al4pde_power | 70000 | 124 | 0 | 1 |
| al4pde_random | 70000 | 124 | 0 | 0 |
| al4pde_uncertainty | 70000 | 124 | 0 | 0 |
| al4pde_bait | 40000 | 42 | 0 | 0 |
| al4pde_power | 70000 | 42 | 0 | 0 |
| al4pde_random | 70000 | 42 | 0 | 0 |
| al4pde_uncertainty | 70000 | 42 | 0 | 0 |
| al4pde_bait | 40000 | 22 | 0 | 0 |
| al4pde_power | 70000 | 22 | 0 | 0 |
| al4pde_random | 70000 | 22 | 0 | 0 |
| al4pde_uncertainty | 70000 | 22 | 0 | 0 |
| al4pde_bait | 40000 | 20 | 0 | 0 |
| al4pde_power | 70000 | 20 | 0 | 0 |
| al4pde_random | 70000 | 20 | 0 | 0 |
| al4pde_uncertainty | 70000 | 20 | 0 | 0 |
| al4pde_bait | 40000 | 15 | 0 | 0 |
| al4pde_power | 70000 | 15 | 1 | 1 |
| al4pde_random | 70000 | 15 | 0 | 0 |
| al4pde_uncertainty | 70000 | 15 | 1 | 1 |

Uncertainty70k中#15严格局部1条，#22为0条；#20的5条残余邻居和#124的6条残余邻居均未进入该训练集。Power获得#124残余邻居1条。既有“池本来就极稀疏”，也有“池有但未选”两种情况。单一邻居不等于已掌握局部响应，更不能根据这些计数断言某一条候选被选后的因果收益。

### 7.3 不确定性不是在#15上始终很低

重载FNO已保存的独立ensemble成员，对126个real输入进行事后分歧评估。Uncertainty在round0/1/2中，#15分歧名次分别25/5/1，但对应误差仍约7.29/8.09/7.32；Power名次25/12/3。它能逐渐“看到风险”，却不保证候选池中存在足量、同背景的可选支撑。

这里的名次仅是在126个诊断输入内排序，**不是100万池排序**。旧selected.csv没有保存完整池得分及所有未选点排名，无法复原“为什么每个邻近候选当时没选上”。不能把缺少记录补成一个未经验证的评分失败故事。

当前FNO三种策略均完成70k训练时：

| mutant_id | power | random | uncertainty |
|---|---|---|---|
| 124 | 0.2541 | 0.28 | 0.2616 |
| 15 | 5.242 | 6.801 | 3.284 |
| 20 | 0.2421 | 0.522 | 0.4224 |
| 22 | 0.58 | 0.5449 | 0.2146 |
| 42 | 0.4038 | 0.4907 | 0.4495 |

全体real的平均RelL2 Random/Uncertainty/Power约0.1782/0.1794/0.1765，几乎接近，但#15和#22差别很大：平均值确实掩盖尾部。

**当前完成度：** 只有FNO Random/Uncertainty/Power及DeepONet Random完成4轮；FNO BAIT和DeepONet Uncertainty/Power最新为round2（40k），UNet Random为round1（20k）。BAIT不能用40k直接对比别人最终70k后宣称优劣。目录有结果不等于整个批实验已跑完，本轮未恢复训练。

## 8. 历史口径错误：先隔离，才能讨论采样优劣

1. **旧MASS系real初值：** 多个旧文件的#88–91 TEM1初值约0.90389693，而当前标准为0；参数全部126例一致但初值只122例一致。以当前ODE轨迹为参照，#88/#89/#90/#91目标自身RelL2差异约12.07/10.97/2.09/0.069。它们不能直接跨版本归因采样效果。BAMS、pure-random与当前AL真实轨迹一致；Gibbs/kDPP约1e-7级差异属于数值精度量级。
2. **旧小数ID识别：** 原Git版本 `^\d+_` 只认整数ID，漏掉2.1、2.2、61.1、61.2，旧WB、WB_s5与mass_random_l只评122例；这4例被纳入“synthetic”归一化，部分进入训练。原版本恢复后3份实验均可复现。当前AL使用明确is_real与存储划分，不受该旧规则影响。
3. **两份未复现：** rolemode与leaklevel继续作为归档排名资料，不在32份可复现的轨迹证据中。不能声称已排除全部历史训练/评估缺陷。

完整对应关系见 `replay_validation.csv`、`dataset_truth_consistency_summary.csv` 与 `real_truth_versions.csv`。旧文件均未改写。

## 9. 对后续主动学习的建议顺序

1. **先改进可选候选的条件覆盖，再讨论换哪一个官方评分器。** 保持已批准的合法模块/突变模式，审计模块组合×模式组合×碳源×精确KO/有限残余结构的覆盖，不只看总L1/L2/L3比例。像#15/#22池里仅0–1条的条件应补足合法有限强度邻域，而非直接复制real测试点。
2. **同背景采一组强度点，兼顾不启动平台与有限时窗切换。** #15证明平坦阻滞区也要学，不能只追“疑似临界”；#22提示同基因背景的强度扫描可能比更换背景更加有用。新候选机制与现有分布不同须单独标明实验臂。
3. **保留官方ensemble评分作为基线，验证覆盖探索与它的互补。** 在未仿真时只能使用参数、初值、模块规则及模型预测/分歧/预测物理残差，不能使用本报告的真实轨迹距离或真实六分类。覆盖辅助方案目前只是待验证假设，未实现、没有收益数值。
4. **六分类作为辅助诊断，仍需连续和事件指标。** 按case记录首次分裂、分裂次数、尾段ORI/CDC20T、逐通道物理误差与低信号相对误差；#124与#62证明同组不代表同轨迹，#16/122提醒不能只盯ORI。ABC仍可用于粗汇总，但对定向消除这些失败信息不足。
5. **区分可采域与模型/损失瓶颈。** 部分#59/#62已有局部候选进入训练仍然误差较大，不能把所有问题归结于缺样本。下一步需要同模型/预算/归一化/真值的受控采样对照，并查看训练误差与预测方程残差；本轮不能从评估CSV唯一识别容量、优化与损失各自贡献。
6. **把#42单列OOD或明确扩大研究域。** 这一选择要显式做，不能通过统计解释假装L3覆盖了L4。
7. **补充选择可追踪性。** 每轮保存候选ID、可复现打分或分批排序、选中原因、模块/模式桶、失败/拒绝原因；才有能力区分供给、打分、预算和仿真失败造成的缺口。
8. **避免测试集调参闭环。** 126例已用于本轮诊断，BAMS-hard也是test-informed。后续以它们调新方法只能报告development结果；另留未参与定向规则设计的基因型/强度或独立测试集合，才能说明泛化提升。

本轮不支持预先指定“80/20混合比例”、恢复变量上限或把所有badcase都贴成分岔临界。这些都需要新的受控实验，而非由本次关联表直接推出。

## 10. 每种静态数据集的关键坏例结果

下列每个文件均有独立Top20、逐样本指标与数据来源。MAE/RelL2/MSE独立排名可在 `per_experiment/static/` 查看；表中只压缩列出5例RelL2。跨行配置不同须结合前面的可比性说明。

| 数据集 | 模型/变量归一化 | 复现核对 | #15 | #20 | #22 | #42 | #124 |
|---|---|---|---|---|---|---|---|
| bams_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 4.673 | 0.4744 | 0.5448 | 0.2522 | 0.1311 |
| bams_50000_210m_500s | biom_refiner_plus/log1p | 通过 | 7.247 | 0.296 | 0.533 | 0.5005 | 0.1909 |
| bams_ablation_hard_50000_210m_500s | biom_refiner_plus/log1p | 通过 | 0.008656 | 0.4591 | 0.1595 | 0.4022 | 0.1059 |
| bams_ablation_hypo_50000_210m_500s | biom_refiner_plus/log1p | 通过 | 0.01258 | 0.3873 | 0.05007 | 0.3877 | 0.1131 |
| bams_motif_topology_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 3.902 | 0.4483 | 0.4344 | 0.3617 | 0.3123 |
| distribution_twin_a_direct_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 1.13 | 0.3359 | 0.1932 | 0.542 | 0.4052 |
| distribution_twin_b_continuous_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 0.6962 | 0.4676 | 0.2936 | 0.4325 | 0.4317 |
| gibbs_boltzmann_100000_210m_500s | biom_refiner_plus/asinh_z | 通过 | 4.551 | 0.3961 | 0.6219 | 0.5873 | 0.4012 |
| gibbs_sobol_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 6.928 | 0.1828 | 0.3786 | 0.3019 | 0.2353 |
| kdpp_curv_gated_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 7.256 | 0.1401 | 0.5729 | 0.3798 | 0.2034 |
| kdpp_gated_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 9.42 | 0.5837 | 0.356 | 0.5385 | 0.211 |
| mass_100000_dataset_210min_500steps_dual_labels | biom_refiner_plus/asinh_scale | 通过 | 0.02881 | 0.485 | 0.03819 | 0.3876 | 0.1589 |
| mass_100000_random_dataset_210min_500steps_dual_labels | biom_refiner_plus/log1p | 通过 | 7.886 | 0.1368 | 0.4426 | 0.3716 | 0.2243 |
| mass_100000_random_l_dataset_210min_500steps_dual_labels | biom_refiner_plus/log1p | 通过 | 6.7 | 0.3248 | 0.4403 | 0.2982 | 0.1915 |
| mass_50000_dataset_210min_500steps_dual_labels | biom_refiner_curvature/log1p | 通过 | 0.02201 | 0.3691 | 0.1402 | 0.3073 | 0.06323 |
| mass_random_dist_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 7.221 | 0.2683 | 0.4927 | 0.3373 | 0.2861 |
| mutant_aware_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 3.046 | 0.3501 | 0.3824 | 0.4501 | 0.2622 |
| mutant_aware_100000_210m_500s_rolemode | biom_refiner_plus/log1p | 未通过 | 7.038 | 0.3854 | 0.5796 | 0.5046 | 0.2177 |
| mutant_aware_100000_leaklevel_210m_500s | biom_refiner_plus/log1p | 未通过 | 6.37 | 0.4973 | 0.5334 | 0.3969 | 0.245 |
| mutant_aware_pure_random_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 7.536 | 0.3417 | 0.4776 | 0.4436 | 0.3249 |
| mutant_aware_pure_random_ko_interval_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 6.538 | 0.4908 | 0.4916 | 0.436 | 0.3948 |
| mutant_aware_pure_random_rescue_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 7.264 | 0.4274 | 0.4981 | 0.5046 | 0.2349 |
| pure_random_distribution_twin_a_direct_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 0.1462 | 0.3703 | 0.5378 | 0.3919 | 0.4116 |
| pure_random_distribution_twin_b_continuous_100000_210m_500s | biom_refiner_plus/log1p | 通过 | 0.1773 | 0.459 | 0.4017 | 0.4481 | 0.4144 |
| wb_mass_100000_dataset_210min_500steps_dual_labels | biom_refiner_plus/log1p | 通过 | 7.207 | 0.3621 | 0.6269 | 0.4293 | 0.2325 |
| wb_mass_100000_dataset_210min_500steps_dual_labels_s5mode | biom_refiner_plus/log1p | 通过 | 4.743 | 0.4146 | 0.4229 | 0.4885 | 0.1138 |

## 11. 旧AL附录

44份旧CSV全部统计，见 `legacy_al_archived_runs.csv` 与 `legacy_al_archived_metrics.csv`。下表仅展示本次20例在历史结果中的复现频度，包含不同模型/归一化/预算与重复轮次，不是44个独立重复实验，也不用于采样因果比较。没有可靠重建这些旧AL所有候选池和逐轮训练集，故不替它们给出确定性选点根因。

| mutant_id | 文件数 | Top20次数 | RelL2中位数 | MAE中位数 |
|---|---|---|---|---|
| 22 | 44 | 42 | 0.7441 | 0.9895 |
| 20 | 44 | 38 | 0.5297 | 0.324 |
| 15 | 44 | 37 | 8.156 | 0.7009 |
| 24 | 44 | 37 | 0.5074 | 0.2547 |
| 54 | 44 | 36 | 0.8263 | 0.8948 |
| 23 | 44 | 36 | 0.5487 | 0.4043 |
| 33 | 44 | 30 | 0.5097 | 0.2338 |
| 111 | 44 | 25 | 0.4263 | 0.295 |
| 76 | 44 | 24 | 0.4089 | 0.1699 |
| 16 | 44 | 24 | 0.4152 | 0.1857 |
| 42 | 44 | 24 | 0.5141 | 0.2481 |
| 124 | 44 | 23 | 0.1579 | 0.5758 |
| 122 | 44 | 22 | 0.3749 | 0.2886 |
| 53 | 44 | 21 | 0.5021 | 0.2287 |
| 58 | 44 | 20 | 0.3538 | 0.2547 |
| 59 | 44 | 19 | 0.2782 | 0.1875 |
| 36 | 44 | 16 | 0.3019 | 0.2064 |
| 46 | 44 | 15 | 0.321 | 0.1931 |
| 52 | 44 | 4 | 0.3137 | 0.2071 |
| 62 | 44 | 2 | 0.0963 | 0.6512 |

## 12. 文件与复现

输出根目录：`/home/users/zsy/mydata/budding_yeast/analysis/badcase_audit_2026-10-27/`。

- `twenty_badcase_evidence.csv`：20例跨数据集概况与统一参考指标。
- `all_mutant_metrics.csv` / `top20_per_run.csv`：175份主分析结果的全量指标/Top20。
- `per_experiment/`：每个静态实验/AL轮次的综合、MAE、MSE、RelL2各Top20。
- `static_dataset_overview.csv` / `matched_training_cohort_ranking.csv`：静态概览/18份同配置子集。
- `validated_case_evidence.csv` / `validated_channel_errors.csv`：32份通过重载核对的局部支撑、时序与逐通道错误证据。
- `support/<run_id>/neighbors.csv`：每例最近训练样本的名字、原始行号、距离与动力学摘要，可回溯到原NPZ。
- `pool_vs_acquisition_support.csv`：候选供给与实际训练覆盖；`real_ensemble_calibration.csv`：保存ensemble的事后分歧。
- `fixed_background_ode_probes.csv/.npz`：固定背景参数扫描及轨迹；未追加正式数据集。
- `figures/badcase_dataset_heatmap.png`、`figures/fno_badcases_across_al_rounds.png`、`figures/fixed_background_ode_probes.png`及20张逐例图。
- `replay_validation.csv`、各replay目录的`provenance.json`、`real_truth_versions.csv`：核对容差、未通过项、版本限制。
- `raw_manifest.json` / `feature_manifest.json`：源结果与数据缓存身份（路径/大小/mtime），不包含全源文件内容哈希证明。

分析工具位于 `scripts/analysis/dataset_diagnosis/`。在项目目录使用neuralop环境，按以下依赖顺序执行；输出默认就在上述机械盘目录，不进行训练：

```bash
python scripts/analysis/dataset_diagnosis/cross_dataset_badcases.py --phase inventory
python scripts/analysis/dataset_diagnosis/cross_dataset_badcases.py --phase features --workers 8
python scripts/analysis/dataset_diagnosis/reconstruct_legacy_scale.py
python scripts/analysis/dataset_diagnosis/badcase_support.py --workers 4
python scripts/analysis/dataset_diagnosis/replay_badcase_models.py --device cuda:5
# 旧WB及mass_random_l需原版本；重载产物与默认重载分开保存
python scripts/analysis/dataset_diagnosis/replay_badcase_models.py --device cuda:5 --historical-model-code --only wb_mass
python scripts/analysis/dataset_diagnosis/replay_badcase_models.py --device cuda:5 --historical-model-code --only mass_100000_random_l
python scripts/analysis/dataset_diagnosis/probe_badcase_manifolds.py --workers 8
python scripts/analysis/dataset_diagnosis/audit_pool_badcase_coverage.py
python scripts/analysis/dataset_diagnosis/audit_badcase_uncertainty.py --device cuda:5
python scripts/analysis/dataset_diagnosis/summarize_cross_dataset_badcases.py
```

以上是事后统计与机理探针，不是已验证有效的新AL算法。未经新的受控训练对照，不承诺对20例全部改进。
