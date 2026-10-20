# 突变强度、参数赋值与初值的文献审计（2026-10-20）

## 结论与适用范围

比较对象是当前 `mass_dataset_generator.py`（MASS）和 `mutant_aware.py`（MA）的
`sample_scaling_factor`、`apply_gene_mutation`、突变初值覆盖及相关模拟器初始化；
不把基因/模式选择概率、motif、图距离、分组配额混入本轮“谁的突变机制更好”的判断。

**没有一份文件可以原样同时被认证为“最符合所有所列论文、严格不预知真实测试参数、预测所有 real mutants 最好”。**

- 忠实复现论文指定的完全失活或稳定化等位基因时，MASS 的若干精确零/固定赋值更接近原定义。
- MC 的整数剂量形式、TEM1 总池为零时的活性初值清零，MA 更合理。
- MA 的近零缓冲、锚点包络是额外训练/不确定性建模假设，不自动比原始零值更有生物学依据。
- 一些规则完全相同；两者还共享初值可行域和合成代理机制问题。
- 历史预测结果按家族混合，不能把单次训练差异归因于某一个采样分布。不能从报告的家族平均分反向按 real 参数设计训练邻域。

本轮不修改生成器、ODE、基准、采样范围、图、指标或训练配置，不重新训练/评估模型；
仅添加分析报告和隔离在 scratch 的诊断工具。

## 1. 文献依据与不能混用的模型

### 1.1 Chen 2004 是当前 ODE 的直接基准

[Chen 2004 主文](<../../papers/Chen 等 - 2004 - Integrative analysis of cell cycle control in budding yeast.pdf>)：
Table 1 为浓度 ODE；Table 2 为 WT 参数与新生子细胞初值；第 3848 页 Computer Simulation 说明：

- 基因删除通常通过关闭对应蛋白的合成表示。
- 多个内源启动子整合拷贝通过整数倍修改对应合成项。
- GAL 构建体改变组成型合成，并按培养基改变生长速率；各构建体的数值是为重现实验表型选定的校准值，不是已测得的总体随机分布。
- 限制温度下 ts 的相关活性可为零；中间温度下合成致死可用 WT 活性的一个比例表示。
- 初值应对应实验协议。论文没有给出“所有 KO 必须保留 WT 初值”或“所有 KO 必须清零全部状态”的统一规则。

[期刊全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC491841/) 可用于定位上述 Methods 说明。
当前两份代码的 WT 参数、39 维初值与连续 RHS 经 AST/数值比较一致。

### 1.2 Chen 2000 不应作为替换 2004 参数值的表

[Chen 2000](<../../papers/Chen 等 - 2000 - Kinetic Analysis of a Molecular Model of the Budding Yeast Cell Cycle.pdf>)
是较早且不同的模型版本（例如 Hct1/Cdh1 表示方式、状态与某些参数定义不同）。
Table 3 注释对 GAL 所说的“uniform value”是统一固定数值，不是从均匀概率分布抽样。
不能从早期模型同名参数的数值，直接否定/覆盖 2004 的浓度参数。

### 1.3 两份不同的补充材料

- [e03-11-0794_chen_suppl.pdf](../../papers/e03-11-0794_chen_suppl.pdf) 是 Chen 2004 的 Online Supplement；
  第 11–12 页 E 节讨论 CLB2-db 和其他突变的参数脆弱性。
- [Ahmadian 补充 PDF](<../../papers/Ahmadian 等 - Supplementary Information A Hybrid Stochastic Model of the Budding Yeast Cell Cycle.pdf>)
  是后续混合随机模型：增加 mRNA/蛋白动态与分子数换算，Table S1/S4 数值不是当前浓度 ODE 的同一套变量。
- [full_supplementary_info.md](../../papers/full_supplementary_info.md)
  是后一份补充材料的整理，并非第五份独立实验验证。其 Table S5 可核对等位干预，但不能忽略单位、版本和表格自身的不一致。
- [Ahmadian 2020 正文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7101447/)
  区分随机模型的实验表型比较与模拟结果；不能用其分裂概率直接当作当前确定性轨迹的实测浓度。

### 1.4 “real-mutant 轨迹”与生物实验真值要分开

当前 NPZ 的这些轨迹来自 REAL_MUTANTS 参数字典调用 ODE，实验论文主要提供基因型/表型等依据；
它们不是每个浓度通道的真实实验时序测量。
低 MAE/MSE 表示更好逼近该参考模拟器，不自动证明更接近真实细胞。
Chen 2004 主文本身也讨论了 131 个表型中 11 个不一致，不能要求神经算子仅靠训练采样消除原机理模型与生物实验的差异。

## 2. 判断“更优”的三个层次

1. **干预语义**：是否改对反应/总池？完全删除、特定等位基因和泛化低功能变体不能混为一类。
2. **概率先验**：已知一个构建体的固定值，不足以推出 LogU、Beta、Normal 的形状或上下限。
3. **预测证据**：需要固定组合规则、样本预算、切分、归一化、模型、损失和随机种子，才能归因强度/初值变化。

依据等级在本报告中为：论文明确机制/表格值；由当前方程推出的物理可行性；
以及工程采样假设/尚待实验的泛化建议。代码注释自称“文献一致”不作为独立证据。

## 3. 有差异的核心规则：如何选

### R1. MC：MA 的整数形式更符合原模型，但不等于概率已经正确

MASS 为连续 LogU[1.5,30]；MA 为整数 n=2,…,30，P(n)∝1/n。
两者修改的参数位置基本相同，均按各自 WT 成比例。

对“内源启动子整合的基因拷贝数”这一论文定义，MA 的整数剂量更直接。
但论文未给出 1/n 先验或上限 30。连续有效表达倍数也可作为另一种不确定性模型，
只是不能把它直接解释为整数拷贝数。
MA 注释中的“到 50”与实际 `np.arange(2,31)` 不同；本报告按实际 2–30 判断。

### R2. GAL：多数相同；CLN2 新下限不能仅靠测试锚点认定更生物合理

两者均把相应 basal synthesis（或在固定总池模型中的总池）设为 gene-specific LogU 绝对值，
通常不是把 WT 参数乘一个通用 GAL 倍数。
CLN2 额外将受调控合成项设为零，对应具体替换启动子/删除内源背景的建模约定。

唯一主要采样差异：CLN2 从 LogU[0.03,0.30] 改为 LogU[0.06,0.30]。
MA 注释明确参考 real GAL-CLN2 的 0.12 构建体。
这不是论文测得的随机 GAL 强度分布，不能因新范围更靠近该测试构建体而宣布更生物合理。
其他 GAL 区间相同，也不能据此认定它们都具有严格独立于测试的来源。

### R3. 普通 KO：两者相同；不要把“合成零”和“初始蛋白零”等同

CLN2/CLB2/CLB5/SIC1/CDC6/SWI5/CDC20/PDS1/PPX 等主要关闭合成；
CLN3/BCK2 等关闭代数表达驱动。
CDH1、TEM1、CDC15、BUB2 的相应模式另外有总池/初值清零。

- 急性关闭合成后，已有蛋白继续衰减可以是合理实验协议。
- 稳定遗传删除株从出生开始的状态，不能无条件用 WT 含蛋白/复合物的初值来代表。
- 清除一个结合蛋白时，还须考虑其复合物内组分与其他结合伙伴；不能简单只清一个游离通道。

因此，之前将“普通 KO 一定要保留 WT 初值”理解为硬性论文要求是不准确的。
当前两份代码没有提供可切换的统一实验协议，不能靠更换强度分布补足这一点。

### R4. CLB5/PDS1/SIC1-db、APC-A：精确复现时偏 MASS

这些特定论文干预把对应识别/降解或激活项设为精确零：
例如 CLB5-db 的 kdb5pp，PDS1-db 的 kd2pdspp/kd3pdspp，GAL-SIC1db 的 kd3c1，
APC-A 的 ka20pp。MA 改成 WT×U[0,0.01]。

MASS 更接近这些指定等位基因的理想化模型定义；
MA 对“有少量残余功能的其他等位/未知构建体”可作为假设，但不能仍声称精确复现原始完全失活株。
连续近零范围不等于有显式零点概率质量，尤其不能因为 real 碰撞去重困难，就以“生物学修复”为名把结构零全部去掉。

### R5. CLB2-db：机制相同，范围没有全面胜者

两者均 kdb2P=0，且保留较弱 CDH1/KEN 降解：
MASS 的实际 kdb2pp∈[0.016,0.060]；
MA 的实际 kdb2pp∈[0.0085,0.068]。

Chen 2004 Online Supplement E 节强调这一表型对剩余降解参数很敏感，
并讨论 0.025/0.03/0.05 带来的不同结果。
这支持“不能随意混同所有 db 变体”，而不支持“更宽或更靠近测试锚点的均匀分布必然更优”。
对未知等位变体，两种范围都缺少总体频率证据；对一个已定义构建体，应有构建体/实验条件而非任意再抽样。

### R6. ts：特定限制温度等位与泛化 hypomorph 要分开

- **CDC20-ts**：论文指定合成零；MASS 用 0.2×Beta(0.5,2)，MA 用 U[0,0.01]。
  MA 更接近强失活，但两者都不严格等于零，也都不涵盖完整温度—活性关系。
- **CDC15-ts / APC_Core-ts**：MASS 的相应零值更接近所列限制温度模拟；MA 微小残余是扩展假设。
- **CDC14-ts**：MASS 关闭分支使用 ks14=0；MA 使用小非零值。
  两者另有四个催化参数同时衰减的分支，与部分活性表型的思路相容；
  但 50/50 分配及 U[0.05,0.35] 不是上述文献给出的总体概率规律。
- **NET1-ts / TAB6-1**：特定表格为共同缩放 f=0.05；
  MASS 的 TAB6-1 固定值较接近该定义，其 NET1-ts 的通用 Beta 则也不是固定该等位。
  MA 两者都随机落在 LogU[0.0125,0.10]，该范围由已知测试锚点构造。
- **ESP1-ts**：MASS 使用固定 kasesp=0.1、kdiesp=0.001；
  MA 以共同 f∈LogU[0.0005,0.004] 缩放二者。固定等位与连续等位是不同问题；另见第 6 节原文数值不一致。
- **TEM1-ts**：MA 初值更一致，但其随机 ka15pp 还有动力学无效的问题，见第 4 节。
- **PPX-ts / Mcm1-ts**：所列文献不足以验证当前固定常数/通用 Beta 就是对应命名等位的分布。
- **CLN2/CLB2-ts**：两者都通过减少合成来代理温敏，不能未经对应等位机制说明就等同于所有 ts 的催化失活。

若目标是通用训练覆盖，可考虑把完全失活、部分失活、许可温度分开声明，
但模式权重和范围应事先定义，并用合成留出验证；本报告不根据 real 参数给出定向区间。

### R7. DAmP、pLOF：相同，概率仍属先验；部分 DAmP 初值不合法

两者 DAmP 都使用 Normal(0.2,0.05²) 后裁剪到[0.05,0.40]；
pLOF 都是 U[0.10,0.50]。
减少对应合成/总池可以表达部分表达损失，但这些概率和参数并非所列论文给出的实测分布。
TEM1/CDC15 的 DAmP 在两者中都使活性初值超过总池，不能原样作为物理可行状态。

### R8. 相同的 CA、检查点、代理模式不意味着都已被论文证实

MAD2-CSL、BUB2-CKO、CDH1-CA 在该模型语义下基本相同。
CLB2-CA、SCC1-RR、CDC5_Proxy、CDC28 及 ORI/BUD/SPN 的代理映射应分别看待，
不能因为同一论文中提到这些基因就视为完整机制复现，详见第 5 节与附录。

## 4. 初值和动态有效性的实际验证

对27模块、84合法模式逐项执行64个固定种子的小规模参数探针。
这些不是重新生成数据集，也不测量训练性能；报告的是代码行为与初值不等式。

| 引擎 | 探针发现 | 次数/64 |
|---|---|---:|
| MASS | TEM1-ts：Tem1T=0，但Tem1(0)=0.9038969 | 64 |
| MA | 上述零总池问题已清零修正 | 0 |
| MASS、MA | TEM1-DAmP：Tem1T≤0.4，Tem1(0)=0.9038969 | 各64 |
| MASS、MA | CDC15-DAmP：Cdc15T≤0.4，Cdc15(0)=0.6565329 | 各64 |

代码支持的总池非负约束应包括 0≤Tem1≤Tem1T、0≤Cdc15≤Cdc15T、
0≤Esp1≤ESP1T、0≤Cdh1≤Cdh1T，以及 RENTP/NET1P 非负。
表中验证仅覆盖探针的单项设置，不是全部组合与全过程不变量的证明。
不能据此断言所有旧 NPZ 已损坏或这些问题已经解释了全部 badcase。

### TEM1-ts 的“随机强度”并不一定有效

MA 在此模式设定 Tem1T=0、Tem1(0)=0。
当前 Tem1 方程保持 Tem1=0；Cdc15 激活项中的 ka15pp×Tem1 始终为零。
因此仅改变此模式抽取的 ka15pp，并没有增加相应轨迹的有效多样性。

人工探针取 alpha=0.0005 与0.004，均按当前 MA 规则模拟210分钟、500步：
两次模拟成功，Tem1通道始终为0，完整轨迹逐点最大差为 **0**。
这不应被解释为“较窄锚点包络已被证明解决 TEM1 家族的物理差异”。

## 5. 比新旧强度更基础的共同代理机制风险

### 5.1 CLB2-CA 实际改的是抑制复合物解离

两者均将 kdib2 设为零。该项控制 CLB2:SIC1 复合物的解离，而不是直接把 Clb2 激酶变成组成型活跃。
在 WT 初值处，人工比较该单项修改使 dCLB2/dt 减少0.011920235，
并增加 C2 形成方向的净滞留。这是瞬时 RHS 的代码事实，不是已测得的长期表型方向。
至少不能把它未经检验就标成“抗CKI抑制/组成型活性增强”的生物实现。

### 5.2 SCC1-RR 没有在当前模型中实现不可裂解底物

两者均把 kasesp=0；它实际上关闭 PDS1 与 ESP1 的结合。
当前状态中没有显式SCC1或其裂解反应，因此该操作不等价于让 cohesin 变得不可被 separase 裂解。
WT初值探针中 dESP1/dt 反而增加0.3858663316795。
两者都需要额外机制或明确的表型代理限定，不能单靠换α分布认证正确。

### 5.3 CDC5_Proxy 与匿名PPX不能直接等同

当前 CDC5_Proxy 使用 PPX 的合成/稳定性参数；
PPX 在方程中控制 Net1 去磷酸化，Cdc5 则是论文中有不同调控角色的蛋白。
Chen 2004 还明确指出未充分包括 Cdc5/FEAR 等作用。
此外，旧 PPX-ts 注释把“#112”称为 cdc5-ts，但当前 REAL_MUTANTS #112 是 mad2Δ，不能用该注释作证。
本报告不宣称 PPX 的合成实验无用，只是不把它认证为真实 CDC5 敲除的完整分子实现。

### 5.4 CDC28 与其他合成代理

CDC28 模式只缩放所列一组效率参数，未显式重建全部 Cdc28 依赖反应；
SBF_MBF 是合并TF，ORI/BUD/SPN 是事件辅助变量。
这些可以作为有明确标签的数学扰动，但需要与“已有论文验证的具体等位突变”区分。
两者共有这些问题，因此不能凭某一份文件名称宣布它全面符合生物学。

## 6. 原始表格与当前基准的数值不一致：不能自动选一个覆盖另一个

1. CLB2-db：Chen 2004 补充E和Ahmadian Table S5印为 kdb2pp=0.03，
   后者同一行又写“8.5%”；若WT=0.4，8.5%对应0.034而非0.03。
   当前 REAL_MUTANTS 与 MA 锚点使用0.034。
2. ESP1-ts：Ahmadian Table S5 第26页印 kdiesp=0.002，又标两项为1/500×WT；
   WT kdiesp=0.5 时1/500为0.001。两份当前代码及基准使用0.001。

已对原始PDF第23、26页做页面图像核对，上述差异不是本地Markdown独有的转录错误。
这不能直接证明当前值错误，更不能当作 MA 范围合理性的独立证据；
需要明确采用的模型/代码版本，并在必要时核对作者代码或勘误。
本轮未修改 REAL_MUTANTS 或任何标签。

## 7. 历史预测证据：哪些结论有支持

以下均为 BioRefinerPlus / hybrid_mse_rel / log1p / width192 / modes24 / blocks4 /
batch256 / epochs100 / lr0.00088 / seed42，real数均126。
指标是原有存档，不是本轮修复后的重训结果。

| 历史数据集/规则 | MAE | MSE | pooled relL2 | pooled Corr | 训练提交 |
|---|---:|---:|---:|---:|---|
| [MASS-random（旧规则，组合加权）](../experiments/mass_100000_random_dataset_210min_500steps_dual_labels/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json) | 0.063306 | 1.130352 | 0.124909 | 0.979102 | f0375fc |
| [MASS-random-dist（新幅度＋旧赋值）](../experiments/mass_random_dist_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json) | 0.086977 | 1.554738 | 0.180318 | 0.956659 | 5027906 |
| [mutant-aware（新规则＋结构采样）](../experiments/mutant_aware_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json) | 0.050027 | 0.740209 | 0.098039 | 0.988552 | e79ca36 |
| [pure-random（新规则＋随机骨架）](../experiments/mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json) | 0.086912 | 5.184606 | 0.151820 | 0.972276 | 64f2608 |
| [pure Twin-A（部分直接赋值规则）](../experiments/pure_random_distribution_twin_a_direct_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json) | 0.089117 | 2.388915 | 0.156859 | 0.968697 | eb4bdc8 |
| [pure Twin-B（连续包络/缓冲）](../experiments/pure_random_distribution_twin_b_continuous_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json) | 0.082969 | 2.084944 | 0.158334 | 0.969458 | eb4bdc8 |

### 7.1 为何不能只看哪个数据集总分更好

- mutant-aware 数据集的存档优于 MASS-random，但其组合选择也变了，不能归因于本轮参数规则。
- pure-random 使用 MA 规则却并非全部优于 MASS-random；这也不是公平地只切换底层规则。
- MASS-random-dist 只混入新幅度、保留旧赋值，也没有整体收益。
  其2026-09-08训练提交中的幅度函数相对旧MASS主要只变了MC与GAL-CLN2；不能用这组旧指标证明当前所有新增缓冲都不好。
- 更接近对照的是 pure Twin-A/B：同一训练提交eb4bdc8，A仅把一部分目标模式改回直接赋值。
  MC、普通KO、GAL等仍共享MA设置，模拟器两侧也都来自MA，因此A不是一份完整的原始MASS规则数据集。

pure Twin-B 相比 A：MAE降低约6.90%，MSE降低约12.72%，但pooled relL2增加约0.94%；
9个核心变量的relL2与Corr也不是全面改善。不存在“所有指标、所有突变体都更好”的统一胜者。

### 7.2 家族内表现（只能作观察证据）

下表是同一对pure Twin存档中，按所列ID选出的评估样本平均值，不是按家族单独训练/调参；
家族可重叠，不能将这些均值再直接平均为全局指标。
此处的平均relL2是逐mutant平均，与上一表pooled relL2不是同一统计量。

| 评估家族 | 样本数 | A 平均MAE | B 平均MAE | A 平均MSE | B 平均MSE | A 逐样本平均relL2 | B 逐样本平均relL2 | B 的MAE更低例数 | mutant IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CLB2-db | 9 | 0.255007 | 0.124394 | 5.609177 | 2.019177 | 0.242079 | 0.130233 | 6/9 | 55, 56, 57, 58, 59, 60, 61.1, 61.2, 62 |
| CLB5-db | 5 | 0.206444 | 0.115347 | 9.012667 | 2.420710 | 0.090574 | 0.043409 | 4/5 | 68, 69, 70, 71, 72 |
| PDS1-db | 3 | 0.007108 | 0.013135 | 0.001380 | 0.015309 | 0.008569 | 0.020257 | 0/3 | 82, 83, 84 |
| SIC1-db | 1 | 0.007262 | 0.020080 | 0.001293 | 0.011125 | 0.008922 | 0.011294 | 0/1 | 29 |
| CDC20-ts | 5 | 0.063024 | 0.064652 | 0.557267 | 0.537798 | 0.019703 | 0.018749 | 3/5 | 73, 78, 79, 85, 102 |
| CDC14-ts | 6 | 0.029765 | 0.026685 | 0.048503 | 0.033271 | 0.043207 | 0.044790 | 5/6 | 99, 103, 104, 105, 106, 107 |
| TEM1-ts | 4 | 0.018648 | 0.017650 | 0.016775 | 0.022234 | 0.023436 | 0.024728 | 2/4 | 88, 89, 90, 91 |
| CDC15-ts | 3 | 0.030755 | 0.029949 | 0.047457 | 0.044448 | 0.066061 | 0.062651 | 1/3 | 95, 96, 109 |
| ESP1-ts | 2 | 0.007745 | 0.019118 | 0.001841 | 0.019934 | 0.009321 | 0.033812 | 0/2 | 81, 84 |
| NET1/TAB6 | 8 | 0.029558 | 0.050989 | 0.171033 | 0.429757 | 0.039157 | 0.064866 | 1/8 | 90, 95, 97, 102, 108, 109, 110, 111 |
| APC-A | 10 | 0.162800 | 0.187497 | 8.052897 | 13.537511 | 0.080989 | 0.085016 | 2/10 | 115, 116, 117, 118, 119, 120, 121, 122, 123, 124 |
| mutant15 | 1 | 0.066815 | 0.054588 | 0.077998 | 0.057476 | 0.146169 | 0.177321 | 1/1 | 15 |

### 7.3 对这组证据的限制

1. 两侧是同时更换一组规则，而非每次只更换一个模块或一个数值。
2. 这些家族经常包含多突变组合，不能把“该家族分数变好”理解为某一个α先验的因果贡献。
3. 仅一个训练seed，无多seed不确定性估计；“0/2改善”也不是显著性检验。
4. [先前参数审计](SAMPLING_DISTRIBUTION_AUDIT_2026-10-20.md)发现：
   pure Twin 在29223条不含目标模式的合成行中，仍有1898条参数不同，差异涉及ksbud/ksori/ksspn/mdt。
   因而这不是严格只改变9类目标模式的单因素实验。
5. 本轮额外只读取当前两份NPZ各自前126条真实基准轨迹（每份9,828,000字节），其内容与名称哈希相同。
   这确认当前文件的两侧真实轨迹一致，但不等于证明它们与9月训练时的文件逐字节相同；
   原数据缺少完整的生成manifest/绑定哈希。
6. 这些实验早于10月的归一化、图、运行链路等修复；未在当前统一版本下重训，不能宣称当前方案的最佳性能。
7. 所有家族指标只是该ODE基准轨迹的误差，不是对论文实验表型正确性的再次验证。

观察上：CLB2-db、CLB5-db倾向B；
PDS1-db、ESP1-ts、NET1/TAB6、APC-A倾向A；
CDC20/CDC14/TEM1/CDC15的不同指标又有取舍。
这可用于提出后续合成控制实验的假设，但不授权按每个real的参数创建专属训练邻域。

## 8. 对用户两个要求的最终建议

### 8.1 要求一：生物/论文优先

不整份照搬“更新的”文件。更合理的概念组合是：

- 保留MASS中与论文指定完全失活/去识别一致的结构零语义；
- 采用MA的整数MC表示，但把整数支持集/概率标为设计先验，而非论文实测结论；
- 采用MA在零总池时保持初值可行的原则，并补齐两者共同的低总池初值约束；
- 保留有依据的部分活性建模，但把特定等位基因与泛化hypomorph分开命名/配置；
- 不把连续缓冲当成删除/失活的生物学必然改进；
- 先审查CLB2-CA、SCC1-RR、CDC5_Proxy等映射，再决定是否纳入声称“文献支持”的训练数据。

这是后续设计建议，不是本轮已经实现的新采样器。

### 8.2 严格不预知real参数的边界

论文公开的通用反应规律、WT尺度和遗传干预类别，与给定测试株的具体校准常数不是同一层信息。
如果按用户要求禁止预知real具体参数，则：

- MA显式围绕已知real值构造包络，不能原样被认证为盲态。
- MASS的若干固定非零等位常数与GAL区间也需要做来源审计，不能因其更旧就认定完全无此信息。
- 除法则明确规定的结构零等之外，不应通过查找测试株参数来收窄/居中训练区间。
- 文献复现用的“已知特定构建体配置”和盲态泛化训练配置应分开，且记录依据。
- 未知范围和分布形状应预先声明，并用合成侧的物理/留出诊断选择；不能先看某个real再给它定制周边样本。

### 8.3 要求二：预测100多个real更好

现有证据不足以认证任一文件的全部规则最优。
应先修物理不可行/无效/错误代理，再在相同随机骨架、合成切分、训练预算和代码下逐项比较；
所有非目标随机值也应真正共享，而不是只共享一次α。
这些对照必须与原存档分开命名，不能用重生成数据覆盖旧NPZ再沿用旧成绩。
论文更忠实、参数覆盖更广、单次MAE更低是三个不同属性，需要分别报告。

## 9. 本轮方法与可复核材料

- 当前HEAD：17be4b4b1038a6ca5a5711c4091a10167cd65eae。
- MASS源码SHA256：9494e9a888bca2ebbb9a28612396e493f03da545efcb28c11cf6c5546e63aa7f。
- MA源码SHA256：72f9b3df8537b21f6bfa1109eed276f34edfc0d7ee5d2f77d7499909af9551c7。
- 诊断脚本：[scratch/probes/audit_mutation_rules_chen_20261020.py](../scratch/probes/audit_mutation_rules_chen_20261020.py)。
- 每项64个种子；覆盖84个合法模块—模式，不改变全局训练RNG或写入数据池。
- 人工TEM1-ts两个210分钟ODE积分，不是新增real采样或训练样本。
- WT参数、初始向量及RHS的AST比较一致。
- 未执行训练/GPU推理，未改生成器/模型/基准/指标；不能把参数探针当作新模型实验。

复核命令（仅诊断输出）：

```bash
/home/users/zsy/.conda/envs/neuralop/bin/python -B \
  scratch/probes/audit_mutation_rules_chen_20261020.py \
  --draws 64 --simulate-tem1 --include-experiments --check-twin-prefix --compact
```

当前pure Twin真实轨迹前缀SHA256：
`95e58f2be9cd4928f14795e4ad5a51624a14093e5ee0b668ed6960b9740491a1`（两侧相同）。
名称前缀SHA256：
`f97599014d1dabddc3e4ee02697f6a7ba4dcd5a597753071e044ead022d7dacf`（两侧相同）。

## 附录 A：27 模块、84 个合法模块—模式逐项清单

`MASS` 指 `mass_dataset_generator.py`；`MA` 指 `mutant_aware.py`。本表只比较单项参数采样/赋值，不比较组合选择。
`WT×f` 对每个参数使用其各自WT值；`LogU` 为对数均匀，`U` 为均匀，`N` 的第二参数在本表中表示方差。
某些分支抽取了α但不使用，表中写的是实际生效规则。普通KO的蛋白/复合物起始状态还须由实验操作定义。

| 模块 | 模式 | MASS 实际规则 | MA 实际规则 | 文献/机制判断 |
|---|---|---|---|---|
| CLN2 | KO | ksn2pp=0 | ksn2pp=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| CLN2 | GAL-OE | ksn2p/ksn2pp；绝对值α~LogU[0.03,0.3]；ksn2pp=0 | ksn2p/ksn2pp；绝对值α~LogU[0.06,0.3]；ksn2pp=0 | 机制相同；新下限参考real锚点，不足以证明更生物合理或严格盲态 |
| CLN2 | MC | ksn2pp=WT×n；n~LogU[1.5,30]（连续倍数） | ksn2pp=WT×n；n∈{2,…,30}，P(n)∝1/n | MA的整数剂量形式更接近Chen 2004；2–30及1/n概率不是论文实测分布 |
| CLN2 | ts | ksn2p/ksn2pp=WT×0.2×Beta(0.5,2) | ksn2p/ksn2pp=WT×0.2×Beta(0.5,2) | 两者相同；以合成减弱代理温敏，未给温度/等位机制，不能视为通用催化失活定律 |
| CLN3 | KO | C0=0 | C0=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| CLN3 | GAL-OE | C0；绝对值α~LogU[1,5] | C0；绝对值α~LogU[1,5] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| CLN3 | MC | C0=WT×n；n~LogU[1.5,30]（连续倍数） | C0=WT×n；n∈{2,…,30}，P(n)∝1/n | MA的整数剂量形式更接近Chen 2004；2–30及1/n概率不是论文实测分布 |
| CLB2 | KO | ksb2p/ksb2pp=0 | ksb2p/ksb2pp=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| CLB2 | GAL-OE | ksb2p；绝对值α~LogU[0.002,1.5] | ksb2p；绝对值α~LogU[0.002,1.5] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| CLB2 | MC | ksb2p/ksb2pp=WT×n；n~LogU[1.5,30]（连续倍数） | ksb2p/ksb2pp=WT×n；n∈{2,…,30}，P(n)∝1/n | MA的整数剂量形式更接近Chen 2004；2–30及1/n概率不是论文实测分布 |
| CLB2 | ts | ksb2p/ksb2pp=WT×0.2×Beta(0.5,2) | ksb2p/ksb2pp=WT×0.2×Beta(0.5,2) | 两者相同；以合成减弱代理温敏，未给温度/等位机制，不能视为通用催化失活定律 |
| CLB2 | db | kdb2P=0；kdb2pp~U[0.016,0.060] | kdb2P=0；kdb2pp~U[0.0085,0.068] | 两者都保留正确的CDC20去识别+CDH1残余方向；无证据新宽包络更优，见正文版本差异 |
| CLB2 | pLOF | ksb2p/ksb2pp=WT×U[0.1,0.5] | ksb2p/ksb2pp=WT×U[0.1,0.5] | 两者相同；可表示部分合成保留，但U[0.1,0.5]并非CLB1残余量的已测分布 |
| CLB2 | CA | kdib2=0（减少CLB2:SIC1复合物解离） | kdib2=0（减少CLB2:SIC1复合物解离） | 两者均待重审：降低抑制复合物解离不直接实现组成型激活 |
| CLB5 | KO | ksb5p/ksb5pp=0 | ksb5p/ksb5pp=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| CLB5 | GAL-OE | ksb5p；绝对值α~LogU[0.004,0.03] | ksb5p；绝对值α~LogU[0.004,0.03] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| CLB5 | MC | ksb5p/ksb5pp=WT×n；n~LogU[1.5,30]（连续倍数） | ksb5p/ksb5pp=WT×n；n∈{2,…,30}，P(n)∝1/n | MA的整数剂量形式更接近Chen 2004；2–30及1/n概率不是论文实测分布 |
| CLB5 | db | kdb5pp=0 | kdb5pp=WT×U[0,0.01] | MASS精确零更接近文献指定db缺失株；MA非零缓冲属于hypomorph扩展 |
| SIC1 | KO | ksc1p/ksc1pp=0 | ksc1p/ksc1pp=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| SIC1 | GAL-OE | ksc1p；绝对值α~LogU[0.03,0.36] | ksc1p；绝对值α~LogU[0.03,0.36] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| SIC1 | MC | ksc1p/ksc1pp=WT×n；n~LogU[1.5,30]（连续倍数） | ksc1p/ksc1pp=WT×n；n∈{2,…,30}，P(n)∝1/n | MA的整数剂量形式更接近Chen 2004；2–30及1/n概率不是论文实测分布 |
| SIC1 | db | kd3c1=0 | kd3c1=WT×U[0,0.01] | MASS精确零更接近文献指定db缺失株；MA非零缓冲属于hypomorph扩展 |
| SIC1 | PM | kd2c1=kd3c1=0 | kd2c1=kd3c1=0 | 两者相同；多个稳定化参数同时置零，不能无条件等同于完全不可磷酸化蛋白 |
| CDC6 | KO | ksf6p/ksf6pp/ksf6ppp=0 | ksf6p/ksf6pp/ksf6ppp=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| CDC6 | GAL-OE | ksf6p；绝对值α~LogU[0.1,0.6] | ksf6p；绝对值α~LogU[0.1,0.6] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| CDC6 | MC | ksf6p/ksf6pp/ksf6ppp=WT×n；n~LogU[1.5,30]（连续倍数） | ksf6p/ksf6pp/ksf6ppp=WT×n；n∈{2,…,30}，P(n)∝1/n | MA的整数剂量形式更接近Chen 2004；2–30及1/n概率不是论文实测分布 |
| CDC6 | DAmP | ksf6p/ksf6pp/ksf6ppp=WT×clip(N(0.2,0.05²),0.05,0.4)；初值不变 | ksf6p/ksf6pp/ksf6ppp=WT×clip(N(0.2,0.05²),0.05,0.4)；初值不变 | 相同的合成缩减代理；N(0.2,0.05²)和裁剪阈值为采样假设 |
| BCK2 | KO | B0=0 | B0=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| BCK2 | MC | B0=WT×n；n~LogU[1.5,30]（连续倍数） | B0=WT×n；n∈{2,…,30}，P(n)∝1/n | MA的整数剂量形式更接近Chen 2004；2–30及1/n概率不是论文实测分布 |
| SWI5 | KO | ksswip/ksswipp=0 | ksswip/ksswipp=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| SWI5 | GAL-OE | ksswip；绝对值α~LogU[0.02,0.1] | ksswip；绝对值α~LogU[0.02,0.1] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| CDC20 | KO | ks20p/ks20pp=0 | ks20p/ks20pp=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| CDC20 | GAL-OE | ks20p；绝对值α~LogU[0.5,12] | ks20p；绝对值α~LogU[0.5,12] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| CDC20 | MC | ks20p/ks20pp=WT×n；n~LogU[1.5,30]（连续倍数） | ks20p/ks20pp=WT×n；n∈{2,…,30}，P(n)∝1/n | MA的整数剂量形式更接近Chen 2004；2–30及1/n概率不是论文实测分布 |
| CDC20 | ts | ks20p/ks20pp=WT×0.2×Beta(0.5,2) | ks20p/ks20pp=WT×U[0,0.01] | 两者均不精确复现论文的零；MA更靠近强失活，不能替代显式完全失活模式 |
| CDC20 | APC-A | ka20pp=0 | ka20pp=0.2×U[0,0.01] | MASS精确零更接近论文APC-A定义；MA表示微弱残余而非原始完全去磷酸识别 |
| CDC20 | DAmP | ks20p/ks20pp=WT×clip(N(0.2,0.05²),0.05,0.4)；初值不变 | ks20p/ks20pp=WT×clip(N(0.2,0.05²),0.05,0.4)；初值不变 | 相同的合成缩减代理；N(0.2,0.05²)和裁剪阈值为采样假设 |
| CDH1 | KO | init_CDH1/init_CDH1T/kscdh=0 | init_CDH1/init_CDH1T/kscdh=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| CDH1 | CA | kicdhpp=0；kscdh=0.03 | kicdhpp=0；kscdh=0.03 | 两者相同，与所列CDH1-CA实现相近；培养基/启动协议另行保持一致 |
| CDC14 | ts | 50%:ks14=0；50%:ks14=WT，四个催化参数×U[0.05,0.35] | 50%:ks14=0.2×U[0,0.01]；50%:ks14=WT，四个催化参数×U[0.05,0.35] | MASS关闭分支更接近特定文献株；催化衰减可有依据，但50/50混合及范围需独立声明 |
| CDC14 | GAL-OE | ks14；绝对值α~LogU[0.3,1.2] | ks14；绝对值α~LogU[0.3,1.2] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| CDC14 | MC | ks14=WT×n；n~LogU[1.5,30]（连续倍数） | ks14=WT×n；n∈{2,…,30}，P(n)∝1/n | MA的整数剂量形式更接近Chen 2004；2–30及1/n概率不是论文实测分布 |
| CDC14 | DAmP | ks14=WT×clip(N(0.2,0.05²),0.05,0.4)；初值不变 | ks14=WT×clip(N(0.2,0.05²),0.05,0.4)；初值不变 | 相同的合成缩减代理；N(0.2,0.05²)和裁剪阈值为采样假设 |
| NET1 | TAB6-1 | kasrent=10；kasrentp=0.05 | kasrent=200f；kasrentp=f；f~LogU[0.0125,0.1] | 固定特定TAB6-1更接近MASS；随机NET1弱化可用于新等位假设但不能仍称原始固定株 |
| NET1 | GAL-OE | ksnet；绝对值α~LogU[0.15,0.6] | ksnet；绝对值α~LogU[0.15,0.6] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| NET1 | ts | kasrent=200f；kasrentp=f；f~0.2×Beta(0.5,2) | kasrent=200f；kasrentp=f；f~LogU[0.0125,0.1] | 都未固定论文特定f=0.05；新窄包络显式用real锚点，不能认证为盲态最优 |
| NET1 | DAmP | ksnet=WT×clip(N(0.2,0.05²),0.05,0.4)；初值不变 | ksnet=WT×clip(N(0.2,0.05²),0.05,0.4)；初值不变 | 相同的合成缩减代理；N(0.2,0.05²)和裁剪阈值为采样假设 |
| TEM1 | KO | Tem1T/init_Tem1/ka15pp=0 | Tem1T/init_Tem1/ka15pp=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| TEM1 | GAL-OE | Tem1T；绝对值α~LogU[2,8] | Tem1T；绝对值α~LogU[2,8] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| TEM1 | MC | Tem1T=WT×n；n~LogU[1.5,30]（连续倍数） | Tem1T=WT×n；n∈{2,…,30}，P(n)∝1/n | MA的整数剂量形式更接近Chen 2004；2–30及1/n概率不是论文实测分布 |
| TEM1 | ts | Tem1T=0；ka15pp=0.002；保留WT活性初值 | Tem1T=init_Tem1=0；ka15pp~LogU[0.0005,0.004] | MA初值更自洽；但清零Tem1后该ka15pp随机化不改变轨迹，须重审有效机制 |
| TEM1 | DAmP | Tem1T=WT×clip(N(0.2,0.05²),0.05,0.4)；初值不变 | Tem1T=WT×clip(N(0.2,0.05²),0.05,0.4)；初值不变 | 两者都有总量小于活性初值的问题，不能直接采用 |
| CDC15 | KO | Cdc15T/init_Cdc15/kkpnetpp=0 | Cdc15T/init_Cdc15/kkpnetpp=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| CDC15 | GAL-OE | Cdc15T；绝对值α~LogU[3,20] | Cdc15T；绝对值α~LogU[3,20] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| CDC15 | MC | Cdc15T=WT×n；n~LogU[1.5,30]（连续倍数） | Cdc15T=WT×n；n∈{2,…,30}，P(n)∝1/n | MA的整数剂量形式更接近Chen 2004；2–30及1/n概率不是论文实测分布 |
| CDC15 | ts | kkpnetpp=0；抽出的α不使用 | kkpnetpp=0.6×U[0,0.01] | MASS精确零更接近论文限制温度模拟；MA是另加的残余活性假设 |
| CDC15 | DAmP | Cdc15T=WT×clip(N(0.2,0.05²),0.05,0.4)；初值不变 | Cdc15T=WT×clip(N(0.2,0.05²),0.05,0.4)；初值不变 | 两者都有总量小于活性初值的问题，不能直接采用 |
| PPX | KO | ksppx=0 | ksppx=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| PPX | GAL-OE | ksppx；绝对值α~LogU[0.2,0.6] | ksppx；绝对值α~LogU[0.2,0.6] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| PPX | ts | kppnetpp=0.05；抽出的α不使用 | kppnetpp=3×0.2×Beta(0.5,2) | 未在所列文献找到对应命名突变的概率分布；固定常数和通用Beta均需额外依据 |
| PPX | db | kdppxpp=0 | kdppxpp=WT×U[0,0.01] | 两者都是对匿名PPX稳定性的合成扩展，未有特定等位分布证据 |
| PDS1 | KO | ks1pdspp/ks2pdspp=0 | ks1pdspp/ks2pdspp=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| PDS1 | GAL-OE | kspdsp；绝对值α~LogU[0.02,0.25] | kspdsp；绝对值α~LogU[0.02,0.25] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| PDS1 | db | kd2pdspp/kd3pdspp=0；抽出的U[0,0.05]不使用 | kd2pdspp/kd3pdspp=WT×U[0,0.01] | MASS精确零更接近文献指定db缺失株；MA非零缓冲属于hypomorph扩展 |
| ESP1 | ts | kasesp=0.1；kdiesp=0.001；抽出的α不使用 | kasesp=50f；kdiesp=0.5f；f~LogU[0.0005,0.004] | 固定特定等位基因与未知连续等位基因是不同任务；原文kdiesp数值还存在内部不一致 |
| ESP1 | GAL-OE | ESP1T；绝对值α~LogU[1.5,6] | ESP1T；绝对值α~LogU[1.5,6] | 赋值相同；GAL基础表达/总池上调是模型近似，精确LogU范围不是论文给出的总体分布 |
| MAD2 | CSL | Vi20=Vi20_active=0.01 | Vi20=Vi20_active=0.01 | 两者相同，符合此模型中关闭相应检查点的规则；CKO包括初值清零 |
| BUB2 | CKO | kbub2=BUB2_active=init_BUB2=0 | kbub2=BUB2_active=init_BUB2=0 | 两者相同，符合此模型中关闭相应检查点的规则；CKO包括初值清零 |
| SBF_MBF | KO | kasbf/ksb5pp/ksn2pp=0 | kasbf/ksb5pp/ksn2pp=0 | 同等：联合TF失活代理，需明确所指基因背景 |
| SBF_MBF | CA | is_sbf_ca=1；kasbf=0.38；kisbfpp=0 | is_sbf_ca=1；kasbf=0.38；kisbfpp=0 | 两者相同，模型TF固定激活代理，不是全部SBF/MBF等位机制 |
| Mcm1 | ts | kamcm=0.13；抽出的α不使用 | kamcm=1×0.2×Beta(0.5,2) | 未在所列文献找到对应命名突变的概率分布；固定常数和通用Beta均需额外依据 |
| Mcm1 | KO | kamcm=0 | kamcm=0 | 同等：关闭合成/有效总池；普通KO初值应由实验协议决定 |
| APC_Core | APC-A | ka20pp=0 | ka20pp=0.2×U[0,0.01] | MASS精确零更接近论文APC-A定义；MA表示微弱残余而非原始完全去磷酸识别 |
| APC_Core | ts | ks20p/ks20pp/kscdh=0；init_CDH1T=init_CDH1=0 | ks20p/ks20pp/kscdh=WT×U[0,0.01]；init_CDH1T=init_CDH1=0 | MASS精确零更接近论文限制温度模拟；MA是另加的残余活性假设 |
| ORI_BUD_SPN | perturb | ksori/ksbud/ksspn分别乘clip(N(1,0.2²),0.1,3) | ksori/ksbud/ksspn分别乘clip(N(1,0.2²),0.1,3) | 两者相同；为辅助事件变量的表型代理，不是这些基因的完整分子机制 |
| ORI_BUD_SPN | cdc24Δ | ksbud=0 | ksbud=0 | 两者相同；为辅助事件变量的表型代理，不是这些基因的完整分子机制 |
| ORI_BUD_SPN | cdc45Δ | ksori=0 | ksori=0 | 两者相同；为辅助事件变量的表型代理，不是这些基因的完整分子机制 |
| ORI_BUD_SPN | spindle_defect | ksspn=0 | ksspn=0 | 两者相同；为辅助事件变量的表型代理，不是这些基因的完整分子机制 |
| MDT | MDT | mdt∈{90,150,160}，各1/3 | mdt∈{90,150,160}，P={0.75,0.18,0.07} | 三个培养基对应的MDT值可用；等概率或75/18/7为实验设计先验，非生物实测频率 |
| CDC5_Proxy | KO | ksppx=0 | ksppx=0 | 两者均待重审：匿名PPX不是显式CDC5，不能据此认证CDC5等位突变 |
| CDC5_Proxy | db | kdppxpp=0 | kdppxpp=WT×U[0,0.01] | 两者均待重审：匿名PPX不是显式CDC5，不能据此认证CDC5等位突变 |
| CDC28 | ts | ebudb5/ebudn2/ebudn3/ec1b2/ec1b5/ec1k2/ec1n2/ec1n3/ef6b2/ef6b5/ef6k2/ef6n2/ef6n3/eicdhb2/eicdhb5/eicdhn2/eicdhn3/eorib2/eorib5/esbfb5/esbfn2/esbfn3=WT×0.2×Beta(0.5,2) | ebudb5/ebudn2/ebudn3/ec1b2/ec1b5/ec1k2/ec1n2/ec1n3/ef6b2/ef6b5/ef6k2/ef6n2/ef6n3/eicdhb2/eicdhb5/eicdhn2/eicdhn3/eorib2/eorib5/esbfb5/esbfn2/esbfn3=WT×0.2×Beta(0.5,2) | 两者相同但为部分通道代理；未覆盖全部Cdc28依赖反应，范围/概率未经文献验证 |
| CDC28 | pLOF | ebudb5/ebudn2/ebudn3/ec1b2/ec1b5/ec1k2/ec1n2/ec1n3/ef6b2/ef6b5/ef6k2/ef6n2/ef6n3/eicdhb2/eicdhb5/eicdhn2/eicdhn3/eorib2/eorib5/esbfb5/esbfn2/esbfn3=WT×U[0.1,0.5] | ebudb5/ebudn2/ebudn3/ec1b2/ec1b5/ec1k2/ec1n2/ec1n3/ef6b2/ef6b5/ef6k2/ef6n2/ef6n3/eicdhb2/eicdhb5/eicdhn2/eicdhn3/eorib2/eorib5/esbfb5/esbfn2/esbfn3=WT×U[0.1,0.5] | 两者相同但为部分通道代理；未覆盖全部Cdc28依赖反应，范围/概率未经文献验证 |
| SCC1 | scc1-RR | kasesp=0（实际关闭PDS1:ESP1结合） | kasesp=0（实际关闭PDS1:ESP1结合） | 两者均待重审：未显式表示SCC1裂解；关闭抑制复合物结合不等价于不可裂解cohesin |
