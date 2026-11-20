# 多seed前核查与sampling baseline补充清单

> 后续实现更新：四类方法已接入明确命名的回归/动态/生成适配入口，详细机制和未继承的原文假设见[新实现说明](REGRESSION_SAMPLING_BASELINES_2026-11-17.md)。下方“尚未完成”是更改前盘点，不能替代最新状态；尚无新方法正式性能成绩。

日期：2026-11-17。这里只整理方案，没有实现新baseline、修改训练协议或启动多seed。**最新用户要求：至少2～3项2025～2026近期方法，允许HGGS/OED/OGAS直接生成新样本。第10节调整优先级，覆盖前文仅因固定池约束而暂缓的建议。**

## 1. 本批耗时与运行问题

结果协议：`seed42_a866aef5d988`。

- 北京时间02:51:08.255至06:08:11.951，墙钟 **3小时17分04秒**，是两池16项全部完成的时间。
- 16个任务各自约2小时39分～3小时17分；累计任务时间46小时28分51秒，不能与并发墙钟混淆。
- 16项退出码均0，主模型80轮、辅助ensemble16轮完成；未检出Traceback、OOM、RuntimeError、FloatingPointError或守护中止记录，最终保存指标有限。守护未abort，PSS峰151.96 GiB、系统最低可用538.82 GiB。
- 主模型五轮预算2000/4000/6000/8000/10000，训练100/100/100/100/200epoch；按本轮训练集重拟合归一化，best由同池归一化`val_loss_synth`最低决定。全部80个best与history最低点一致；同池初始有序ID一致，训练与合成holdout无交集。
- 当前仍有预测短板：CDH1T关闭通道错误、高幅值#124、mutation模型跨LHS测试严重OOD失效；这些不是进程崩溃，但不能称“所有科学问题都解决”。详见[本批分析报告](../../../../../../../../data/zsy/budding_yeast/analysis/sampling_results_20261117/report.md)。

### VeSSAL需要单独注明的情况

检查四次追加`selection.json`：

|池|每次追加2000条中的随机补齐数|合计/8000|
|---|---|---:|
|mutation|0、299、0、0|3.74%|
|LHS|1347、1333、1271、1323|65.93%|

作者`VeSSAL/query_strategies/vessal.py`的query确有`fill_random`分支，项目也保存了这一诊断，因此不能仅凭补齐认定实现bug。但LHS结果明显混入大量随机补齐，不能解释为完全由体积规则挑出的8000条。当前用的是作者支持的penultimate特征分支、随机流顺序和项目固定池/五轮协议，并非默认分类gradient-embedding整套实验。

多seed前建议对VeSSAL做小规模确定性公式对账与拒绝/补齐原因诊断：核查特征秩/尺度、协方差更新、有限性、参考分支设置；不能为了提高real成绩直接修改阈值或隐去补齐。没有必要因此重跑其余已经正确完成的方法；若改变VeSSAL规则，另设方法/版本并保留当前结果。

## 2. best具体取在哪个epoch

以下是**主模型**，不是辅助ensemble。每一列来自各轮`training.json`与`history.csv`核对。

|池|方法|R0/100|R1/100|R2/100|R3/100|R4/200|
|---|---|---:|---:|---:|---:|---:|
|mutation|Random|100|98|100|99|200|
|mutation|Uncertainty|100|99|99|98|192|
|mutation|Power|100|99|99|100|200|
|mutation|MaxDist|100|100|100|99|183|
|mutation|CoreSet|100|99|100|98|192|
|mutation|LCMD|100|99|99|100|197|
|mutation|BAIT|100|98|100|99|200|
|mutation|VeSSAL|100|97|100|100|200|
|LHS|Random|54|76|87|96|154|
|LHS|Uncertainty|54|100|87|96|123|
|LHS|Power|54|87|93|93|164|
|LHS|MaxDist|54|87|95|100|158|
|LHS|CoreSet|54|89|98|92|140|
|LHS|LCMD|54|92|97|100|136|
|LHS|BAIT|54|86|97|99|140|
|LHS|VeSSAL|54|64|89|98|159|

结论：mutation几乎都在尾声；LHS确实出现几十epoch的best，首轮均54，第二轮还有64/76。LHS最终轮没有36/56这样的早期best，而是123～164。相同池首轮数据、主模型seed与训练规则相同，因此八个方法首轮同为54或100正常。LHS首轮辅助成员也有51epoch，不应混作主模型best。

这次是归一化validation本身在这些epoch最好，并非继续用物理Relative选best。训练loss下降不要求独立validation同步下降；后期平台或轻微回升与这张表一致。不要为让best看起来靠后而改成last。具体数值见[best_epochs.csv](../../../../../../../../data/zsy/budding_yeast/analysis/sampling_results_20261117/best_epochs.csv)。

## 3. 阅读范围和实现原则

本地阅读了`papers/baselines/sampling/`的10篇PDF，包含AL4PDE、Physics-PRE、HGGS、PEAKS、Clustering Sensitivity、Low-Rank、Lev、ERACS、OGAS和OED；并检查`baselines/sampling/`14个参考仓库的README及相关选择内核。区分作者代码、作者框架中的变体和第三方合集，不因为出现在DeepCore/DISTIL中就称为对应原论文的作者实现。

**下面“难度”是对本项目适配工作的判断，不是论文结论，也不承诺GPU开销很低。**参考代码只用于开发核对；新生产实现继续放`utils/sampling/baseline/`，不import/exec/运行时读取参考仓库，也不改参考仓库文件。一个新算法一个独立模块，通过已有`select(context,budget)`契约注册，不再扩张一个巨大分派文件。

## 4. 已有实现，暂未在本批运行

|方法/注册名|当前状态|是否值得先补跑|需注意|
|---|---|---|---|
|Physics-PRE：`physics_pre_topk/power`|项目有独立ODE适配|**高优先级**，与细胞周期任务直接相关|原论文PDE→39维ODE，需核对物理时间、变量尺度、分裂事件mask、训练参考残差；不是CP-PRE所有UQ功能的复现|
|Lev：`lev_input`|独立实现输入特征分支|中优先级，廉价几何对照|原论文LLM嵌入→参数+初值Z-score，当前one-shot；不能假称与五轮模型驱动流程完全同构|
|Clustering Sensitivity：`clustering_sensitivity`|论文重实现，有放回、重要性权重|中优先级，覆盖代表性/损失方向|需要代表点真值，保留权重及实际标签访问数；不是只揭示最终10000标签的AL|
|`clustering_sensitivity_wor`|无放回、无权重适配|可作单独适配对照|不能继承有放回重要性估计的无偏结论|
|`lcmd_pred/bait_pred`|已有预测轨迹特征变体|可选特征消融|与已跑LL版本同算法，不能计作两个新算法家族|
|`residual_topk`|已实现真预测误差排序|诊断对照|只是HGGS的残差思路适配，不是完整HGGS；访问候选真轨迹|
|`true_pattern_random`|已实现真实Pattern分层|诊断对照|不是文献baseline；访问训练侧未选样本Pattern也要记账|

已有完整结果的8种为Random、Uncertainty、Power、MaxDist-Pred、CoreSet-LL、LCMD-LL、BAIT-LL、VeSSAL-penultimate。不要重复把它们列为“尚未实现”。

## 5. 建议新增的方法

|优先级|方法|选择逻辑/补充价值|参考位置|本项目难度|未选候选是否需真轨迹|
|---|---|---|---|---|---|
|P1|**MaxDet-LL**|贪心增加正则化核行列式，覆盖互补信息方向；与BAIT的信息矩阵目标不同|`bmdal_reg/bmdal_reg/bmdal/selection.py`的MaxDet两种内核|中|否|
|P1|**k-means++-LL**|按到已覆盖点的平方距离随机选，而非CoreSet总选最远点；易复用现有特征|同文件KmeansppSelectionMethod|低|否|
|P1/P2|**KMeans-LL**|聚类后每簇选最近的真实候选，代表密集区域|`distil/.../kmeans_sampling.py`|低～中，完整KMeans计算可能较重|否|
|P2|**Herding-LL**|逐步让已选特征均值接近全候选均值，补充均值匹配对照|`DeepCore/deepcore/methods/herding.py`|低～中|否|
|P2|**MaxDiag-LL**|按条件核对角线/后验方差单点排序，单模型核不确定性对照|BMDAL MaxDiagSelectionMethod|低～中|否；需明确核/条件化，不能把裸特征范数叫不确定性|
|P1（监督筛选方向）|**GradMatch**|用OMP选有权重子集，逼近全训练池梯度|`cords/.../SL/gradmatchstrategy.py`及作者论文|中～高|通常是|
|P2|**CRAIG**|在梯度表示上做代表点覆盖并赋代表权重|`cords/.../SL/craigstrategy.py`|中～高|非凸梯度版通常是|
|P2|**GraNd**|按早期单样本梯度范数衡量学习难度|`DeepCore/.../grand.py`|中；原版多次早期训练成本要计入|是|
|P2|**Facility Location**|选点共同覆盖训练侧候选的相似度|DeepCore submodular_function.py、DISTIL submod_sampling.py|中，资源风险需先处理|用预测/隐藏特征则否|
|P3|**ACS-FW/Frank–Wolfe**|核/特征空间匹配整体嵌入的迭代采集，已有回归框架|BMDAL FrankWolfeSelectionMethod|中|否|
|P3|**GLISTER**|用候选梯度的一步训练效果改善合成验证目标|`cords/.../SL/glisterstrategy.py`|中～高|需候选真值及验证真值|

### 适配细节不能省略

1. **MaxDet不等于直接跑无噪声det。**512维投影与每轮2000点预算组合会有秩问题；应保留参考框架的posterior、noise、训练锚点模式和稳定更新，所有超参只用合成开发数据确定。优先特征空间实现，不分配90k×90k全核。
2. k-means++是随机距离选点；KMeans是拟合中心再选最近候选；LCMD是动态簇覆盖；CoreSet是最远点。这四者不能混称“都是KMeans”或换名字重复计数。MiniBatchKMeans/流式版本若替代全KMeans要显式命名近似。
3. Herding可不按类别分层；不需要为39维回归强造分类标签。但隐藏特征提取、投影/归一化、已选点是否作为锚点必须固定。
4. GradMatch/CRAIG/GraNd从分类迁移到回归，不是把num_classes改成39即可。当前FNO末层为共享时间的`Conv1d(128,39,1)`；每个样本的参数梯度要对时间求和。按当前归一化Relative L2及其eps求导，不能沿用one-hot、softmax或分类outer-product公式。先以小样本autograd对账，再分块/投影降维。
5. 本地CORDS有回归任务配置和可替换loss的梯度计算入口，但公开实现仍有分类分支/张量形状假设；不能声称开箱即用39×500多输出回归。
6. Facility Location等参考实现可能分配N²矩阵：90k候选float32全矩阵约30.2 GiB；多个并发会把资源风险重新引入。分块本身可保持同一目标，稀疏化/随机贪心则要报告近似变化，不可静默替换算法。
7. GLISTER使用同池合成validation属于其算法设定，需记录额外信息访问；可用现有验证集但不能碰synth test或real。它不再是与其他方法完全相同的验证信息使用方式。

## 6. 阅读后建议暂缓的论文/方法

|方法|原因与后续可行路线|
|---|---|
|**PEAKS**|值得研究，但不是“容易原样复现”。原论文/作者公开代码是分类IDS，使用真实one-hot、softmax误差和类别logit/kernel，公开主实验还是预训练ViT。应先推导轨迹回归下误差×核相互作用及流式阈值，标为PEAKS-regression适配。不能直接将Pattern作为类别替换后，声称完全复现了轨迹选择方法。|
|**HGGS完整方法**|与生物振荡最相关，但原任务主要预测振荡频率，包含GF过滤与MGS遗传生成，不是当前固定池选点。只实现残差排序不算复现；应独立设计保持合法mutation约束的生成基准并记ODE成本。|
|**CREST**|动态训练子集和重选择是方法本体，最终驻留10000不代表历史只训练10000。需要训练调度/累计训练暴露账本，不能硬塞到只追加五轮而称原版。|
|**ERACS**|论文用梯度SNR监测代表性退化并触发OMP重选择，先有GradMatch/OMP基础更顺。当前本地有论文、未发现同名作者仓库；不宣称网上不存在。应保留动态重选语义，独立比较训练总步数/历史输入暴露。|
|**Low-Rank Active Learning**|论文Algorithm1查询子空间基向量的loss；基向量可能不是合法酵母参数/初值，无法直接ODE仿真。需先定义合法映射，不能默默用SVD得分替代整个方法。|
|**Xu等OED**|论文Algorithm1先LHS仿真，再用物理解的信息量做Bayesian后验/UCB自适应生成。不是简单Fisher打分；需定义适合有分裂跳变ODE的方差/能量/频谱信息量。原PDE物理量不能机械搬用；本地未找到对应作者代码，论文虽然称提供实现。适合后续独立生成范式。|
|**OGAS**|原版为连续在线仿真/训练及条件扩散生成，依赖Melissa等；当前固定池+混合离散mutation域差异大。应单独做开放生成实验，不值得为补一项baseline立刻重构训练系统。|
|**DiSF**|LLM文本块去相关选择可改用FNO特征，但任务距离远、逐候选协方差成本较高；本地入口还有导入语法问题，不宜直接运行。已有Lev及MaxDet更适合作为本轮几何对照。|
|**DSIR**|原本匹配目标语料分布，需要参考目标数据与文本特征。若拿126株real作为目标来拟合会污染评测；若只用合成训练参考，须另外说明密度比目标。当前不优先。|
|分类Entropy/BALD/BatchBALD/BADGE等|原版类别概率不等于轨迹随时间的方差/熵。BMDAL有自己的回归解释/变体，可优先沿该框架定义MaxDiag/MaxDet/kmeans++；不能把分类公式直接套39个物种并称原版。|

## 7. 建议实施顺序

1. 先整理VeSSAL补齐诊断与方法名称，冻结当前8方法协议；其余正确完成方法不因这个问题回炉。
2. 优先新增MaxDet-LL和k-means++-LL；补充一个密度代表性方法KMeans-LL或Herding-LL，避免只堆高度相似的几何baseline。
3. 现有Physics-PRE先核对适配后补跑，Lev作低成本输入几何对照。这两项不需要“再实现一次”。
4. 监督完整池筛选另优先实现GradMatch，然后考虑CRAIG/GraNd；它们需要候选真值，必须报告实际标签访问数，不能与只揭示10000真值的方法混称相同标注预算。
5. PEAKS放在第二阶段做明确回归推导；CREST/ERACS及HGGS/OED/OGAS分别保留动态/生成协议，不为凑baseline数量破坏其核心流程。

以上是按相关性、覆盖互补性和实现成本排序，不保证新增方法一定比已有Power好。

## 8. 多seed建议

- 建议先使用42、2026、123作为共同实验seed，不重新生成两个100k池；generation seed仍固定42，冻结修补视图和126株real参考。
- 每个seed重新产生其自己的训练侧/validation/test划分、随机初始2000和随机序列；同seed下全部方法共享相同划分/初始集。模型和选样的子seed从该base seed确定性派生，避免混成每个组件都字面使用相同随机流。
- 新seed必须准备/引用对应协议；不能对seed42的显式协议仅加`--seed 2026`绕过一致性检查。每轮归一化继续只用该轮已查询训练样本拟合。
- 参数、epoch、best规则、预算和指标定义全部固定，不根据这次real分数逐seed调参。主报各seed指标的均值、标准差和逐seed配对差值；三个seed不能支持很强的统计显著性结论。
- 不创建A/B目录层级，但在方法配置中记录是否访问未选候选真轨迹、累计标签访问数、历史训练暴露数和额外模型训练成本。
- 16并发只是本批实测安全；未来多seed任务加入队列，总活跃16、加载8、评分4不乘以seed数，仍按系统剩余内存保留100GiB，W&B仅本地offline。

## 9. 参考依据与本地定位

- [BMDAL作者代码](https://github.com/dholzmueller/bmdal_reg)：回归框架、MaxDet/MaxDiag/FW/kmeans++内核。
- [DISTIL KMeans源码](https://github.com/decile-team/distil/blob/main/distil/active_learning_strategies/kmeans_sampling.py)。
- [DeepCore](https://github.com/PatrickZH/DeepCore)：Herding/GraNd/CRAIG等合集，不一概等同各原作者实现。
- [GradMatch论文](https://proceedings.mlr.press/v139/killamsetty21a.html)、[CORDS](https://github.com/decile-team/cords)：OMP梯度匹配及其实现。
- [PEAKS论文](https://proceedings.mlr.press/v267/gurbuz25a.html)、本地`PEAKS/Source/acquisition.py:482`。
- [ERACS论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/92a035beb61e68f7a688467e5103a17c-Abstract-Conference.html)：动态SNR触发机制，Algorithm1。
- [OED论文](https://www.sciencedirect.com/science/article/pii/S0045782525009478)：两阶段参数生成，Algorithm1。
- [CP-PRE作者仓库](https://github.com/gitvicky/CP-PRE)；本地Physics-based AL论文Eq.5及项目`baseline/physics_pre.py`。

本地PDF正文提取、仓库origin/commit清单保存在`scratch/probes/baseline_review_20261117/`；没有运行官方训练、安装依赖、修改参考代码或占用GPU。下一步独立复现要以固定commit小样本对账为准，不能只比方法名称。

## 10. 最新调整：近期方法、跳跃适配和开放生成

### 10.1 LL具体表示什么

LL指Last Layer，本文方法名里的含义是“输出线性层之前的最后隐藏特征”。当前FNO先以width192运行主干，再投影到每时间点128通道的隐藏特征，最后输出39个变量。当前选择器将128×500的时间特征展开并随机投影到512维，再交给LCMD/BAIT等选择内核。`_pred`版本则使用39变量预测轨迹的表示。LL本身不需要真实候选轨迹，也不是单独训练的模型。

### 10.2 近期方法的主清单

|位置|方法|年份和来源|原方法核心|本项目工作|
|---|---|---|---|---|
|近期主对照1|Physics-PRE|2026，ICLR AI&PDE Workshop|根据预测轨迹的物理残差采集|验证现有分段连续ODE适配，尤其事件附近处理；只能称任务适配，不能宣称已复现混杂系统理论|
|近期主对照2|OED（Xu等）|2026，CMAME|物理信息量的Bayesian建模与自适应参数生成|独立实现其后验/采集规则，定义适合酵母跳跃轨迹的信息量；允许新输入ODE查询|
|近期主对照3|OGAS|2026，作者论文及代码；本次核对arXiv v2于11月12日更新|条件扩散按学习难度在线生成输入，耦合仿真和训练|保留条件生成/难度反馈核心，适配合法参数域与mutation约束；评估在线流程及额外训练成本|
|近期备选/补充|PEAKS|2025，ICML|预测误差×kernel作用的流式增量筛选|需明确轨迹回归推导，原分类one-hot/logit不能直接搬用；不是因为年份满足就忽略适配工作|

上述三项优先考虑任务相关的科学计算方法，PEAKS补充2025年数据选择方向。若OGAS适配验证尚未完成，不能提前声称已有三个近期方法结果；可优先完成PEAKS-regression作为第三项正式对照，OGAS继续开发。HGGS保留为生物振荡相关生成对照；其本地PDF为2024预印本，正式发表为AAAI2025，符合近期方法年份要求，但尚未实现完整GF/MGS，不能计作已有结果。AL4PDE发表于2025不等于其中所有经典采集公式都是2025新提出，文中应分别标注原方法出处；其Uncertainty明确沿用Seung等COLT1992的QBC，参见[出处及复现程度更正](BASELINE_ATTRIBUTION_AND_FIDELITY_2026-11-17.md)。

### 10.3 跳跃系统上的Physics-PRE

把分裂前后两点直接差分会产生真实跳跃除以dt的尖峰，这不是连续ODE区间的求解误差。应区分：

1. 两次事件之间：只在同一连续段计算`dy/dt-f(y,p)`，差分模板不能跨事件。
2. 事件附近：不能要求连续ODE残差为0；排除对应窗口或使用正确的事件条件/重置映射单独评价。
3. 初值：独立检查预测与输入实际初值的一致性。某条错误常值轨迹也可能连续残差小，因而残差低不能保证预测准确。

当前`baseline/physics_pre.py`有基于预测的CLB2/CLB5/ORI/SPN阈值穿越、MASS下降和重置标志检测，并扩展±4步mask；目前只累计mask外的连续残差，没有完整的跳跃映射评分。这是已有工程适配，尚不能据此声称完整通过了事件机制验证。

验证应至少包含：正确的分裂轨迹、正确阻滞轨迹、同输入严格ODE真值残差基线、故意错误的连续段、错误初值、漏掉事件、重置幅值错误。严格ODE真值在平滑段应有低残差，正确跳跃不应只因差分尖峰获高分，错误模型也不能靠大量mask把残差降为0。当前代码在无有效区间时报错，不得静默给0分。

实际采集时，mask/事件检测只能使用预测、候选参数/初值和已知事件规则；不能读取未选候选的真实分裂时刻来帮模型。基于重置映射的额外打分若超出原论文，应单独命名为事件增强变体/消融；不要悄悄加到baseline里。当前mask范围和各变量残差尺度也需用合成数据核查，不以real成绩选取。

### 10.4 允许生成类方法保留原流程

用户已明确允许生成新样本；固定池不再作为拒绝HGGS/OED/OGAS的理由。生成器继续复用当前生物模型、实际初值、ODE事件、real精确碰撞排除、数值重算/失败补采与完整样本元数据保存；不引入未经确认的mutation类型。保持合法强度支持范围和生物映射，允许算法改变查询落点概率。

- OED保留信息量后验与自适应生成，不用单一Fisher分数冒充论文方法。
- OGAS保留难度条件生成器与在线反馈；若改为五轮批量冷启动，要明示是额外适配版本，不能同时声称原版连续在线流程完全未变。
- HGGS保留GF与遗传多网格生成，明确从频率预测迁移到轨迹任务的目标；不能只给现有residual_topk换名。
- 新生成样本可能与固定pool训练点接近，应允许合法相近点，仍排除重复输入、holdout/real精确输入及按既定规则救援；real性能/身份不作为条件生成信号。
- 数据组织继续使用共享源块＋每轮ID清单，新增轨迹只存一次；累计保存输入、mutation、实际初值、轨迹、时间网格和各类标签。
- 既报告最终10000唯一训练样本，也报告所有ODE尝试/成功/失败、访问标签数、历史训练输入、训练步数及生成器额外开销。不能用最终10000掩盖额外仿真或反复训练成本。
- 生成类需配套相同在线训练流程的随机生成对照，才能区分生成策略收益与训练流程变化；固定池方法继续保持共同协议、共同同池holdout和共同real参考。无需增加A/B目录层级。

执行建议：先验证Physics-PRE的事件处理，再分别完成OED和OGAS的独立实现与资源验证；PEAKS作为近期筛选补充。现有经典方法继续提供稳定参照。仅补充了清单，本次未开始实现/仿真/多seed。

参考：[Physics-PRE 2026及Workshop说明](https://arxiv.org/abs/2605.21348)、[OED期刊论文](https://doi.org/10.1016/j.cma.2025.118675)、[OGAS最新作者版本](https://arxiv.org/abs/2606.09949)、[PEAKS ICML2025](https://proceedings.mlr.press/v267/gurbuz25a.html)。


### 2026-11-17后续实施记录

用户已授权开始实现，并要求生物规则只复用不修改。基础实现与18项新增CPU测试现已完成，状态和具体适配见RECENT_BASELINE_IMPLEMENTATION_2026-11-17.md；前文“本次仅方案”是此前阶段。OGAS当前是显式轮次式合法投影版本，原版连续在线流程仍未接入；不得将本次小规模测试计作三项完整大规模论文复现结果。


## 2026-11-17 后续实现：OGAS在线链路

已接入`ogas_async_projected`和`random_async`，支持持续模型/optimizer、后台CPU条件扩散、异步canonical ODE与完整轨迹replay。在线默认对照OGAS v2，原round明确v1。仅为混合生物域Projected适配，原生Melissa自回归时间片/1.8 batches速度限制器及OGAS-Unc没有宣称完成。33项CPU测试通过，生物规则不改，无正式GPU结果。细节见[完整核查表](BASELINE_FIDELITY_AND_OGAS_ONLINE_2026-11-17.md)。
