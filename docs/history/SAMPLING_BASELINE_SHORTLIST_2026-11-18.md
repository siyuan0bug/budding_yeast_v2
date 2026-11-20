# 面向酵母轨迹代理模型的采样 baseline 清单

> **源码一致性核查补充**：[详细报告](../../../papers/baselines/sampling/BASELINE_SOURCE_FIDELITY_AUDIT_2026-11-18.md)。Power/LCMD核心对照通过；HGGS论文与作者代码的公式不一致，项目采用论文式；OGAS的CFG参数约定和密度校正上限已按后续要求[修复](OGAS_ALIGNMENT_FIX_2026-11-18.md)，旧实验结果保留。主清单不能解释为全部已通过原作者实现等价认证，任务适配仍需注明；未启动正式重跑。

> **最新整合选择**：主对照为Random、Power、LCMD-LL、PEAKS-Regression-Reference、Low-Rank-GradientEnergy、Physics-PRE-Power、HGGS-Projected、OGAS-Loss-Projected，共8项。BAIT保留补充，PICore保留后续实现计划；当前不将未实现PICore作为主表前置条件。完整论文、作者代码、适配边界和公平协议见[最终报告](../../../papers/baselines/sampling/BASELINE_REVIEW_AND_FINAL_SELECTION_2026-11-18.md)。本文下方6项、原8项及OED替换建议是讨论记录，以最终报告为当前规划依据；本次未改生产配置或启动实验。

> 全部10篇本地论文的机制及证据见[逐篇盘点](SAMPLING_PAPERS_FULL_AUDIT_2026-11-18.md)。OGAS已有作者源码与实验流程；PRE已核实相关作者残差/AL示例，不能等同专用完整发布。Low-Rank尚未核实专用作者代码，采用明确命名的论文独立实现；PEAKS公开主代码是分类，当前回归版本为项目适配。

核查日期：2026-11-18。本轮只阅读本地论文、参考仓库、生产接口和账本，并核实在线一手来源。没有修改正在运行的算法、训练设置、生物规则或数据池，没有启动额外实验。下文的难度/推荐顺序是针对本课题的判断。

## 1. 结论

现有库已具备足够数量、2025–2026方法及固定池/生成两种覆盖，不需要为了“至少6个”继续堆积不相关方法。但尚不能把全部入口统一称为作者原方法的完整复现，也不能宣称全部实验采用相同初始集。

建议主表先采用6个方法家族：**Random、Ensemble Uncertainty、LCMD、Physics-PRE、HGGS、OGAS**。前三项提供随机、不确定性和几何多样性，后三项提供物理残差、生物边界生成和学习式生成。Power、BAIT已有结果，值得作为扩展对照保留；不通过删除强对照控制表格数量。两种池都运行同一组适用方法，LHS/mutation属于数据生成范式，不计作两种采样baseline。

若要求三项近期方法全部来自主会/正式期刊，可考虑以尚未实现的**PICore（TMLR2025）**替换主表Physics-PRE，后者留作附录。实现与事件适配验收完成前，不能把PICore算作已有结果。

## 2. 两类方法怎样称呼

|分类|后续训练输入来自哪里|内部可以做什么|
|---|---|---|
|固定池选样/数据选择|冻结训练侧候选输入集合|逐轮追加、一次选子集、流式接受、动态替换、带权重采样|
|Query Synthesis / 生成式查询|算法提出新的合法求解器输入，查询ODE|遗传搜索、Bayesian采集、扩散条件生成、异步在线采样|

Query Synthesis本身也可以属于主动学习。固定池方法只有在未选候选真输出不可见时，才能按普通pool-based AL解释；PEAKS/ERACS及部分coreset会读取未选候选真值，应称监督数据选择/子集训练并报告实际访问量。这里是信息条件，不要求恢复目录中的A/B划分。

生成方法可以使用共用起始训练数据和验证集；“不从固定池选后续输入”不等于从头到尾不使用任何池数据。输入与旧训练池偶合不构成自动错误，但必须统计查询/重复输入。real与合成holdout继续遵守既有排除规则。

当前HGGS/OGAS/OED也会用临时的新合法提案库；该库每次重新生成，不是从固定100k训练侧池取行。OGAS/HGGS的期望坐标映射到现有合法提案，是本项目的Projected适配，不能写成完全无约束的原始连续生成。

## 3. 六个主对照及一手出处

|家族|来源/年份|本地代码/结果状态|本任务适配判断|
|---|---|---|---|
|Random|标准随机对照，无新近论文年份|`random`、`random_generate`、`random_async`等已实现并有结果|低难度；各流程使用匹配的随机对照，不能把变体重复计作多个近期算法|
|Ensemble Uncertainty|QBC经典思想；当前回归公式参考AL4PDE，不能把AL4PDE2025当作该思想提出年份|`topk`已实现、两池已完成；`power`另保留|实现容易；独立模型训练成本需计费|
|LCMD|Holzmüller等，*A Framework and Benchmark for Deep Batch Active Learning for Regression*，JMLR2023|`lcmd_ll/lcmd_pred`已实现，LL已有两池结果；参考`bmdal_reg`|原任务就是回归，适配较小；LL/Pred特征替换须说明，不与原sketched NTK配置混称|
|Physics-PRE|Polanska等，*Data-Efficient Neural Operator Training via Physics-Based Active Learning*，2026 ICLR AI/PDE Workshop|`physics_pre_topk/power`已实现；事件mask故障已修，本次LHS-Power补跑完成|物理代理模型相关性高、实现中等；连续残差不能替代完整事件/初值正确性检查|
|HGGS|Rao等，*Hierarchical Gradient-Based Genetic Sampling for Accurate Prediction of Biological Oscillations*，AAAI2025|`hggs_projected`已实现GF/MGS/ODE；本次两池仍补跑中|生物ODE最相关，但频率输出→完整轨迹、连续遗传域→合法提案投影，适配较大；GF初始集不相同|
|OGAS|Cesar等，*Learning Where to Simulate: Generative Active Sampling for Online PDE Surrogate Training*，NeurIPS2026|`ogas_async_projected`已实现并有两池结果；仅Loss方向；分轮版另保留|同为代理模型选仿真输入，相关性高；异步架构、混合域投影和生成器规模使原样复现难度较高|

来源：[LCMD论文](https://jmlr.org/papers/v24/22-0937.html)及[作者代码](https://github.com/dholzmueller/bmdal_reg)；[Physics-PRE论文](https://arxiv.org/abs/2605.21348)及[作者主页的Workshop归属](https://alicjaap.github.io/publications/)；[HGGS正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/34920)及[作者代码](https://github.com/hengrao02/HGGS)；[OGAS论文v2](https://arxiv.org/abs/2606.09949)及[作者发布代码](https://github.com/melissa-sa/ogas)。

Physics-PRE论文明确引用AL4PDE和`gitvicky/CP-PRE`残差代码。CP-PRE是作者相关的残差/UQ代码来源，不能把其README当成这篇AL短文整套实验的复现证据。论文Eq.5及后面的最近训练参考归一化可直接对账；事件mask和Z-score距离是本项目适配。

**近期性：HGGS2025、Physics-PRE2026、OGAS2026，共3项。Physics-PRE须明确写Workshop，不冒充ICLR主会。**OGAS的NeurIPS2026归属已由作者v2、公开代码和NeurIPS2026列表交叉核实。不能用QBC/LCMD/BAIT因被AL4PDE收录就重标为2025新方法。

“有官方代码”提供对照依据，不等于本项目应运行时import该仓库。继续遵守用户要求：生产实现独立放在`utils/sampling/baseline/`，参考库只用于开发核查。

## 4. 已实现、但不优先放入六项主表的内容

|方法|当前状态|暂不优先的原因|
|---|---|---|
|Power、BAIT、CoreSet/MaxDist|已实现并完成两池旧批次|与主表互补，可保留扩展；不是2025新算法。Power已有很强结果，不应因不利于ours就删掉|
|VeSSAL|penultimate分支已实现并完成|LHS随机补齐占65.93%，应公开；分类梯度分支未迁移|
|PEAKS-Regression-Reference，ICML2025来源|已实现并完成|作者公开主代码分类，本项目改为回归参考梯度kernel；读取约6.8万训练侧标签，且持续训练，与普通10k查询AL不同|
|ERACS-Regression，NeurIPS2025来源|已实现并完成|此前两池均0次OMP触发；有机制测试不等于正式规模重选有效，需要合成开发数据校准|
|Low-Rank-GradientEnergy，ICML2026来源|已实现并完成|basis-loss被实例化为梯度平方能量，不是原轨迹Relative loss；理论适用与有放回权重需单列，适配较大|
|Lev-Input，2026预印本来源|已实现；修复后两池补跑完成|原LLM语料嵌入任务较远，当前换成参数+初值；适合廉价几何附录，不为近期性优先使用|
|Clustering Sensitivity|有放回带权及WOR适配均已实现并完成|通用监督coreset，需访问真值；WOR不继承原重要性无偏结论|
|OED-Temporal，CMAME2026|已实现并有两池结果|原生算子学习相关性高，适合作生成对照；当前信息量/输入字典为任务适配，尚未找到对应作者完整代码，不能称原代码移植|
|真Pattern分层、residual_topk|已有实现|诊断对照，不能当作新文献baseline；residual_topk也不等于完整HGGS|

OED出处：Xu、Guilleminot、Tarokh，*Data generation with optimal experimental design for operator learning*，[CMAME450（2026）118675](https://www.sciencedirect.com/science/article/pii/S0045782525009478)。正式卷期为2026，DOI中2025不代表应该把卷期年份改成2025。作者称开源，但本轮未核实可执行完整代码入口，因此不满足“已确认官方代码可用”的强声明。

## 5. 哪些还没做

### 最值得新补充：PICore

Satheesh等，*PICore: Physics-Informed Unsupervised Coreset Selection for Data Efficient Neural Operator Training*，**TMLR2025**。[正式论文PDF](https://openreview.net/pdf?id=l0VSewTJCI)、[作者代码](https://github.com/Asatheesh6561/PICore)。当前两个本地参考目录及生产注册表均未包含它，本轮只查阅，没有实现。

它在未标注输入上用物理loss预热，计算物理loss/梯度，再调用coreset选择，之后只模拟选中点并监督训练。直接研究FNO/UNO，适合本课题。它不是简单换名的PRE Top-K；需要保留其预热、具体coreset内核及权重。作者实验另有重置模型初始化的设置，适配时需明确。

难度判断为中等偏上：可复用canonical RHS和物理时间；仍要处理混合ODE的事件、初值、残差可微性及39变量尺度。已有PRE推理mask不等于已经完成物理梯度训练。不能承诺无需修改即可运行，也不能把一个壳接口计作完成。

### 已有本地参考，但还未形成独立完整生产入口

|方法|参考位置|优先级判断|
|---|---|---|
|MaxDet、k-means++|`bmdal_reg`|易～中；若缺低成本几何对照可补，不是近期性补充|
|KMeans、Herding|DISTIL/DeepCore|易～中；注意90k上KMeans计算成本，不与LCMD混称同算法|
|GradMatch|CORDS|中；回归梯度和完整选择/权重流程仍需独立实现，已有ERACS OMP不等于完整GradMatch入口|
|CRAIG、GraNd、GLISTER|CORDS/DeepCore|中～高；信息访问、近似和验证使用需明确；通用类任务，非当前最缺口|
|CREST|CREST作者仓库（ICML2023）|中～高；动态子集/学习排除/训练调度未接入，不优于先完成ERACS机制验收|
|ACS-FW|BMDAL回归实现|中；可用于机制补充，非必需|

此外，OGAS-Uncertainty、完整原生Melissa配置、VeSSAL分类gradient分支、Low-Rank原loss理论的逐项复现仍未完成。它们是已有家族的变体/复现缺口，不计作多个新的baseline。

DiSF/DSIR等语料筛选不作为当前主表优先项；分类Entropy/BatchBALD等也不能直接把39变量当39类别。并非这些方法没有价值，而是已有更贴近轨迹回归/ODE任务的选项。

## 6. 共同初始集：当前确实存在例外

普通增量Random/QBC/Power/LCMD/BAIT/PRE从同池共同2000条出发。以下现有原生/任务适配流程不能混称相同：

- HGGS保留GF：从10000条coarse训练侧真轨迹筛2000条，并额外训练pilot。当前补跑的初始集与共用2000条仅重叠mutation380、LHS407。GF修改初始集是该流程的一部分，不能因为都叫round_00就当成同一批数据。
- ERACS从共用2000条再随机扩充到10000条resident后持续训练；并非与普通增量AL相同的2k起步。
- Lev/部分coreset一次选择最终子集，不能把它的训练流程标成五轮共同起始集。

若论文主比较强制共同起始2000：可以把完整HGGS保留作原生流程补充对照；另实现显式命名的“HGGS-MGS，共同起始集适配”用于受控主比较，但不能把它冒充包含原GF的完整HGGS。另一种选择是允许方法原生初始化不同并据实报告，从标签/算力总预算比较。当前运行保持原设定，本轮未擅自切换。

## 7. 正式比较必须保留的信息

同一范式内共享源池、seed/split、固定holdout、模型/训练损失和评测口径。生成方法只能复用既有合法规则、实际初值和ODE，不能额外扩大突变支持。混合域投影、暖启动/持续训练、逐轮冷启动等改动公开写清。

分别记录最终训练子集、累计唯一训练曝光、读取真值数量、所有ODE尝试/失败/成功、总训练更新和墙钟；预先模拟100k池的准备成本另报。不能把PEAKS/HGGS读过更多真值的结果写成“只访问10000标签”。当前real已反复用于开发诊断，不应再宣称它是完全未观察过的最终盲测集；新增调参继续仅用训练/合成验证。

两大类均建议有流程匹配的Random；不为压成统一五轮而删掉原方法关键在线机制，也不把不同训练流程产生的全部收益都归给采样打分。随机对照的同流程变体不计作新的算法家族。

## 8. 这次核查时的后台状态

修复补跑队列：`experiments/budding_sampling_extensions/seed42_repairs_20261118/`。读取时3完成/2运行/0失败：两池Lev及LHS Physics-PRE-Power完成；两池HGGS仍在运行。历史完成结果和当前补跑状态不能混合为“所有baseline正式实验都已完成”。本轮未改动后台程序。

## 9. 后续注释：AL4PDE分类、8项主对照与生成成本

用户同意按特征类/不确定性类分别选1～2个代表，并希望增加其他论文的对照。AL4PDE第4.2/4.3节确实按这些机制组织；它收录和适配算法，不为这些算法重新赋予2025年的提出年份。[原文](https://arxiv.org/html/2408.01536v2)

本课题建议不确定性类优先**Power/SBAL**，Top-K作为已有补充；特征类优先**LCMD**，再加已有**BAIT**。LCMD代表距离覆盖，BAIT代表Fisher/信息矩阵互补，区别于再加一个最远点变体。按AL4PDE分组把BAIT放特征实现组，不表示其信息论目标与不确定性毫无关系。

年份必须分别写：LCMD为JMLR2023；Power所依托的Stochastic Batch Acquisition正式发表于TMLR2023（2021有预印本）；BAIT为NeurIPS2021。因此它们是成熟代表，不计作导师要求的2025–2026新方法。[LCMD](https://jmlr.org/papers/v24/22-0937.html)、[Power/SBA](https://arxiv.org/abs/2106.12059)、[BAIT](https://proceedings.neurips.cc/paper_files/paper/2021/file/4afe044911ed2c247005912512ace23b-Paper.pdf)

### 9.1 建议8项

|方法|类型|当前状态|选择理由|
|---|---|---|---|
|Random|共同参照|已有两类匹配随机流程|不把随机流程变体计为多个新算法|
|Power/SBAL|不确定性|已有实现/两池结果|保留ensemble分歧，带随机性的批采集；无需新迁移|
|LCMD|特征覆盖|已有实现/两池结果|原生回归、作者代码可核查|
|BAIT|特征/信息矩阵|已有实现/两池结果|加入与LCMD不同的批次信息互补目标|
|Physics-PRE|物理残差选样|已有实现，两池补跑已完成|2026 Workshop、与代理模型相关，事件处理明确适配|
|PICore（固定具体选择内核）|物理梯度coreset|尚未实现|TMLR2025、FNO/UNO和官方代码，是最值得增加的新方法|
|HGGS-Projected|GF筛选＋MGS生成|两池修复补跑完成|AAAI2025、生物ODE任务最贴近|
|OGAS-Loss-Projected|生成式查询|已有实现/两池结果|NeurIPS2026，目标包含困难轨迹和误差尾部；当前继续使用的新增工程成本低|

相对上一版：用Power作为主要不确定性代表，加入BAIT与PICore，Top-K保留补充。四个近期来源包括一个Workshop；排除Workshop后仍有PICore2025/HGGS2025/OGAS2026三项正式来源。PICore当前只是拟新增，不提前标记已复现。实现它之前应先验证事件残差/初值、可微梯度与物理预热；具体选哪个作者支持的coreset内核要写清，不能把PRE Top-K换个名字叫PICore。

如果只需要7个主对照，可把BAIT移至补充；若只愿使用已经完成的方法，可暂以已有OED-Temporal替换未实现PICore，并明确其代码证据和任务适配边界。现有PEAKS/ERACS/Low-Rank/Lev结果保留，不因主表名单变化删除。

### 9.2 生成方法的难度和贴合度

以下是本项目的工程判断，不是原文对所有任务的难度保证。

|方法|从头实现/适配|贴合budding的部分|主要额外困难|建议|
|---|---|---|---|---|
|HGGS，AAAI2025|中等|生物振荡ODE、参数边界敏感性|原频率标签改成全轨迹；GF改变初始集；MGS需合法混合域投影|最直接的领域生成对照|
|OED，CMAME2026|中等；小型Bayes/UCB内核较简单|为神经算子选择新的仿真参数|ODE信息量定义、离散/连续合法搜索；完整作者代码入口未核实|可以作为额外或替补生成对照，不能称作者代码原样移植|
|OGAS，NeurIPS2026|较高|难例/尾部误差、在线代理训练、最高308维配置的原实验|DDPM、密度比、异步队列、replay、混合域合法投影|从零开始不是最省事；当前已完成工程和结果，可低新增成本继续保留|
|Solver-integrated adversarial sampling，2025预印本、2026修订|高|直接搜索模型/求解器误差大的输入|理想方案需要求解器导数；当前混合BDF/重置事件和整数突变使梯度搜索困难|不作为“容易复现”主对照|
|DAS-PINNs，JCP2023|原算法有明确流程；迁移目标不一致|都依据物理残差|它生成时空collocation点来训练PINN，不是挑一组参数并调用ODE获得完整轨迹|不直接纳入第二层轨迹采样主表|

OGAS当时被选中的依据是相关性和作者代码，不是因为从头移植便宜。此前把“原平台复现成本”和“本项目现在继续使用的增量成本”混在一起容易误导，应以上表区分。

现有记录也表明工程复杂不等于运行慢：上批OGAS两池完整任务约3.44/3.58小时，OED约6.23/6.56小时；本次修复HGGS约3.01/3.08小时。它们处于不同并发批次、训练安排和排队条件，不能当独占硬件效率排名；只能说明当前逐条OED查询不比异步OGAS更快。选样算子耗时、ODE时间和训练耗时应分别报告。

其他文献的证据：[OED论文](https://www.sciencedirect.com/science/article/pii/S0045782525009478)、[OGAS原文及代码入口](https://arxiv.org/abs/2606.09949)、[Solver-Integrated Adversarial Attacking and Training of Neural Operators，当前v4](https://arxiv.org/abs/2510.18989)、[DAS-PINNs](https://www.sciencedirect.com/science/article/pii/S0021999122009317)。原2025 PGD论文在2026修订后已改标题，不拿旧搜索标题冒充新版本。

另查到2026 BMC Bioinformatics的[生物ODE合成数据/迁移学习研究及作者代码](https://github.com/DILiS-lab/opt-synthdata-4tl)，主题贴近，但主要比较合成数据特征和迁移学习，不自动等同本课题第二层的迭代查询算法，宜作相关工作而不是强行增加一个AL baseline。

### 9.3 不能省略的共同条件

该建议按机制和任务匹配选取，不依据real成绩删除对照。PICore/OED等新增适配只用训练/合成开发数据调参；生成方法只复用当前生物规则和ODE。当前HGGS的GF初始集例外、OGAS持续训练与冻结统计例外都要继续披露；需要严格共同初始化的受控版本必须另命名，不能与原生结果混写。

本次仅读取与更新文档。修复队列最新状态5完成/0失败，用户本轮未要求重新评测或增加新后台实验，因此没有启动新的训练。
