# 本地采样论文逐篇盘点与主对照范围

> **逐源码核查更新**：见[原文／源码一致性报告](../../../papers/baselines/sampling/BASELINE_SOURCE_FIDELITY_AUDIT_2026-11-18.md)。新增HGGS论文与源码公式差异、OGAS的CFG及密度上限未对齐问题；OGAS随后已[修复](OGAS_ALIGNMENT_FIX_2026-11-18.md)，旧结果未重跑。不能把已有实验结果等同为原样复现。Power/LCMD同输入选择内核分别通过90/45组对照。

> **当前最终清单与证据整合**见[论文目录下的报告](../../../papers/baselines/sampling/BASELINE_REVIEW_AND_FINAL_SELECTION_2026-11-18.md)：Random、Power、LCMD、PEAKS、Low-Rank、Physics-PRE、HGGS、OGAS，共8项，均须使用报告注明的适配名称。BAIT转补充，PICore仍保留实现计划；本次未切换脚本或启动训练。下文各阶段主表与替换意见保留为核查过程，不与最终清单并列执行。

核查日期：2026-11-18。较早讨论中用户接受的是 Random、Power、LCMD、BAIT、Physics-PRE、PICore、HGGS、OGAS 这8项主对照；后续又讨论PEAKS和Low-Rank，现已形成上述整合报告。此表补全本地论文阅读范围，不自动增加实验、不改变现有算法、数据和生物规则。

## 1. 核查范围与此前展示的不足

`papers/baselines/sampling/` 当前有10篇PDF，共256页，均已提取全文，并逐篇核对任务、主要算法、实验协议及本项目适配边界。本次核查不是对每篇定理的独立证明，也不意味着所有作者实现均已逐行对账。

上一次生成方法比较表不是这个目录的完整盘点：遗漏解释了 Clustering Sensitivity、PEAKS、ERACS、Low-Rank 和 Lev，又穿插了目录外的 Solver-integrated/DAS-PINNs。应分别说明本地论文与外部候选，不能让用户据此判断“目录只有这几种”。

全文提取、页数及源PDF SHA256清单：`scratch/probes/sampling_papers_inventory_20261118/inventory.json`（忽略目录中的本次核查证据；本文是持久结论）。HGGS文件名标2024预印本，正式发表按AAAI2025；OGAS本地PDF是早期版本，项目异步适配另参考在线v2。

## 2. 全部10篇论文

|作者、论文|年份与来源|选择/生成什么|当前项目状态与取舍|
|---|---|---|---|
|Axiotis等，*Data-Efficient Learning via Clustering-Based Sensitivity Sampling: Foundation Models and Beyond*|ICML 2024|以聚类代表点损失和簇内距离构造敏感度，重要性抽样并赋权；论文也包含线性回归，不仅是分类/大模型|已有 `clustering_sensitivity` 与无放回适配 `_wor`。与本任务有关，保留补充；需披露代表点真值访问、有放回权重及无放回版本区别|
|Cesar等，*Learning Where to Simulate: Generative Active Sampling for Online PDE Surrogate Training*（OGAS）|NeurIPS 2026|条件扩散生成新的仿真参数；难度反馈、历史缓冲、密度修正和异步在线训练|已有 `ogas_async_projected`，本批有结果；主对照。当前仅Loss信号分支、合法域投影及酵母完整轨迹适配，不能称Melissa/PDE系统原样复现|
|Cheng等，*Efficient Representativeness-Aware Coreset Selection*（ERACS）|NeurIPS 2025|训练中以梯度SNR判断子集是否失去代表性，触发带权OMP重新选子集|已有 `eracs_regression`，保留补充。旧正式结果每池39次检查均未触发OMP，机制收益尚未被该批结果验证，不能用“训练完成”代替触发有效性验证|
|Cohen-Addad等，*Active Learning with Low-Rank Structure for Data Selection*|ICML 2026|低秩结构、投影残差及基方向loss构造敏感度，带权抽样|已有 `low_rank`，保留补充。基方向采用梯度能量定义是任务扩展；不能称原Relative L2目标的完整理论保证。抽象基方向查询不等于生成新的合法ODE输入|
|Gurbuz等，*PEAKS: Selecting Key Training Examples Incrementally via Prediction Error Anchored by Kernel Similarity*|ICML 2025|对带标签数据流评分、百分位接受、缓存刷新，并混合新旧样本训练|已有 `peaks_regression`，保留补充。回归用实际损失梯度与参考kernel替代分类logit评分；不是因为原实验是分类就排除。正式结果曾访问约6.8万标签，不能计作只查询1万的普通AL|
|Ma等，*Towards Scalable Data Diversification for Language Model Pretraining via Leverage Score Sampling*（Lev）|2026 arXiv预印本|固定语料嵌入的leverage/diversity选择，不生成新的仿真条件|已有 `lev_input`，修复后两池完成；保留补充。将文本嵌入换成参数与初值，是跨任务适配，贴合程度低于专门的物理代理模型方法；leverage思想本身不是2026首次提出|
|Musekamp等，*Active Learning for Neural PDE Solvers*（AL4PDE）|ICLR 2025|统一比较不确定性和特征选择，以及新输入生成/仿真流程|参考框架；当前有Random、TopK、Power、LCMD、BAIT等独立实现。不能将所收录的经典算法都标为2025新方法，也不能把benchmark另算一种采样算法|
|Polanska等，*Data-Efficient Neural Operator Training via Physics-Based Active Learning*（Physics-PRE）|ICLR 2026 Workshop on AI and PDEs|以预测物理残差及已查询参考残差构造评分，再TopK/概率选样|已有 `physics_pre_topk/power`，事件处理修复及补跑完成；主对照。时间差分只在连续区间使用，跳变附近屏蔽；这不等于已加入完整事件重置残差。Workshop不可写成ICLR主会|
|Rao等，*Hierarchical Gradient-Based Genetic Sampling for Accurate Prediction of Biological Oscillations*（HGGS）|AAAI 2025；本地文件名为2024预印本|GF邻域梯度/边界过滤，GMM分层，MGS遗传交叉/变异并查询新ODE输入|已有 `hggs_projected`，修复后两池完成；主对照。生物振荡ODE最贴近本系统；频率改为完整轨迹、连续遗传目标投影到合法提案是适配。原GF粗池预读及初始子集不同须计费和披露|
|Xu等，*Data generation with optimal experimental design for operator learning*（OED）|CMAME 450 (2026), 118675|根据配置的信息量构造Bayes后验和UCB，提出新的仿真配置|已有 `oed_temporal`，保留生成类补充/替补。酵母信息量定义、有限特征字典与统一初始集是适配；尚未核实完整作者公开代码入口，不能写成作者源码移植|

这10篇中，直接以新仿真输入为目标的是 **HGGS、OGAS、OED**。PEAKS的“流式”是带标签样本逐步到达，不自动等于Query Synthesis；ERACS是动态子集；Low-Rank的代数方向也不自动是一个合法生物样本。AL4PDE是统一评测框架，内部覆盖多种机制。

## 3. 每篇需要保留的核心和适配边界

### Clustering Sensitivity

Algorithm 1保留代表点loss与距离项、抽样概率和逆概率权重；Algorithm 2说明其并非只适用于分类。原有放回版本与本项目无放回版本不能合并报告。真值访问发生在训练侧仍须统计，不能只报最终子集大小。

### OGAS

核心是学习条件生成分布并以新反馈更新，配合缓冲与历史偏置修正，不是“随机生成后按loss排序”。从零复现Melissa异步系统成本高；当前已有适配后继续使用的增量成本较低。生成的最终参数、初值和事件设定继续由现有生物规则决定，禁止为方便生成器修改规则。

### ERACS

以Algorithm 1及§4.2的SNR定义为准：SNR下降并低于阈值时触发重新选择。论文实验说明段中另有“exceeds”的措辞，不能绕过算法直接反转不等号。本项目回归梯度、投影及Batch-OMP是显式适配；0次触发须报告并用训练侧开发数据核查阈值与尺度，而非看real测试成绩调触发。

### Low-Rank

需要明确基方向loss如何定义，并保留投影残差、敏感度、抽样和权重。项目使用梯度能量，不可只说“做了SVD所以完全复现”。这些代数向量不要求或保证是合法mutation/初值，不应为它们随意查询ODE真值。

### PEAKS

保留初始训练、流式顺序、kernel锚定评分、百分位接受、缓存刷新和新旧混合训练；不能只改成一次TopK。论文的回归适用性不等于分类评分可以不加推导地原样搬入轨迹回归。标签访问成本和持续训练曝光量需另报。

### Lev

保留分块、随机起点、Gram/正则项、leverage评分与逐批更新；不能简化成输入范数。项目以物理输入特征替代文本嵌入，属于多样性选择的任务迁移，不是对LLM预训练实验的复现。

### AL4PDE

统一框架可以作为接口/协议参照，但Power、LCMD、BAIT分别有自己的来源；其benchmark年份不等于算法提出年份。本文已接受的主表不再重复计TopK与Power为两项近年创新。

### Physics-PRE

预测残差与参考残差归一化是方法核心。酵母轨迹含分裂跳变，不能跨跳变直接用连续ODE差分。安全区不足时缩小屏蔽窗口也必须继续排除真正跨事件的模板；不能用未查询真实轨迹定位事件。

### HGGS

需要完整GF和MGS而不是只按残差排序。原任务输出振荡频率，本项目输出完整轨迹；新的遗传提议只能经当前合法规则实现。GF读取的粗池标签、试训和初始集变化必须披露，不能并入严格共同2k起始AL排名而不说明。

### OED

保留后验更新、探索与利用和新配置查询。酵母完整轨迹信息量是任务定义，不能将任意预测方差TopK命名为OED原方法；实际顺序ODE调用的墙钟成本也应独立报告。

## 4. 用户接受的8项主表

|主对照|作用|实现状态|
|---|---|---|
|Random|随机参照|已有|
|Power/SBAL|不确定性概率批采集；TMLR2023|已有|
|LCMD|特征覆盖；JMLR2023|已有|
|BAIT|特征信息矩阵；NeurIPS2021|已有|
|Physics-PRE|物理残差；2026 Workshop|已有酵母事件适配|
|PICore|物理梯度coreset；TMLR2025|**尚未实现**，不提前计为已复现或有结果|
|HGGS-Projected|生物ODE边界与遗传生成；AAAI2025|已有任务适配|
|OGAS-Loss-Projected|学习式生成查询；NeurIPS2026|已有异步任务适配|

PICore不在当前10篇本地PDF中，是此前外部补查的候选。正式来源见[论文](https://openreview.net/pdf?id=l0VSewTJCI)与[作者代码](https://github.com/Asatheesh6561/PICore)。需落实具体coreset内核、物理预训练/梯度及事件适配，不能将Physics-PRE改名代替。

“已有”表示项目有实现，不代表与作者原系统完全等价；详细差异见第2、3节及已有fidelity文档。Clustering Sensitivity、PEAKS、ERACS、Low-Rank、Lev、OED结果保留为补充，不因为主表选择而删除，也不根据real成绩决定去留。

## 5. 沿OGAS引用新补查的Query Synthesis候选

这些不在当前10篇本地PDF内，不能混进本地计数。

- **PaPQS，NeurIPS2025**：Wang等，*A Plug-and-Play Query Synthesis Active Learning Framework for Neural PDE Solvers*。直接优化新仿真条件的期望信息增益与批多样性，论文用SWAG近似模型后验并训练critic估计信息量；不是只按候选分歧排序。正式[论文页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6a1b224b153e55c40a6359f9c9fb9d8c-Abstract-Conference.html)、[论文PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/6a1b224b153e55c40a6359f9c9fb9d8c-Paper-Conference.pdf)、[作者仓库](https://github.com/QianLab/PaPQS)均已找到，但本次打开仓库明确显示 **“This repository is empty”**，因此不能宣称已有可用作者源码。目标与本任务接近，值得继续核查代码发布或独立实现成本；当前未实现。LHS连续域较易迁移，mutation含离散模式/拷贝数和联动初值，仍需合法域适配，不能提前称“直接可用、低成本”。优化神经代理的输入不要求对真实ODE求解器反传，但不能任意破坏生物规则。
- **MelissaDL × Breed，SC-W2024**：Dymchenko等，以训练loss引导Adaptive Multiple Importance Sampling，在线生成多参数代理模型训练数据。作者[预印本](https://arxiv.org/abs/2410.05860)、[Melissa官方文档](https://melissa.gitlabpages.inria.fr/melissa/)可查。目标相关，可作为生成方向的后续候选；异步接口工程成本与OGAS重叠，且不是2025–2026近期条目。目前未实现。

PaPQS补查来自OGAS本地PDF参考文献[47]，Breed来自[12]。上述成本和优先级是针对酵母混合输入域的判断，不是作者对本项目的结论；本次不据此更换已接受的8项主清单。

## 6. 本次操作边界

只新增本盘点和更新清单导航，未修改生产算法、参数分布、ODE、损失、归一化或训练预算，也未启动新训练。正式代码仍须在项目内独立实现，作者库只用于开发核查，不直接运行时import参考仓库。

## 7. 后续注释：官方源码、适配成本与替换建议

2026-11-18 再次在线核查。用户指出原8项中2026只有2项：这个数量正确，分别是Physics-PRE和OGAS；但“都没有官方代码”不正确。**OGAS已有可读取的作者源码与实验启动流程**；Physics-PRE需要区别作者相关残差/AL示例和该2026论文整套实验发布版。

原8项中的2025–2026来源共4项：PICore2025、HGGS2025、Physics-PRE2026 Workshop、OGAS2026；排除Workshop仍有3项正式来源，但PICore仍待本项目实现，不能说已经完成3项。此前要求是至少2～3项来自2025–2026，并非必须3项都来自2026。若现阶段进一步要求“3项2026＋均有已确认完整官方源码＋容易适配”，当前清单还不能同时满足，不能以年份重标或拿LLM方法凑数。

### 7.1 源码核查证据

这里“有代码”表示确实读到源码/配置文件，不是只看见论文标题或空仓库，也不等于已经在原平台执行作者全部实验。

|方法|本次核实到的官方资料|可以支持的声明|
|---|---|---|
|OGAS|[作者仓库](https://github.com/melissa-sa/ogas)实际有`ogas/sampler/`、`ogas/models/`、`ogas/scenarios/`、`scripts/`和`snakemake_setup/`；README给出Melissa、MPI依赖、配置生成和训练启动流程；[论文v2](https://arxiv.org/abs/2606.09949)直接链接该仓库|有作者公开实现与实验流程；从零工程成本高，当前已有适配后的新增成本较低|
|PICore|[作者仓库](https://github.com/Asatheesh6561/PICore)有`main.py/train.py/coreset.py/pi_losses.py`及配置；README提供FNO＋CRAIG命令|有作者实现；项目当前未实现，事件和物理梯度适配仍需验收|
|HGGS|[作者仓库](https://github.com/hengrao02/HGGS)有`src/`、数据及运行说明，本地已有参考副本|有作者实现；频率→完整轨迹、GF预算及混合域投影是明确适配|
|PEAKS|[作者仓库](https://github.com/BurakGurbuz97/PEAKS)有`incremental.py/main.py/Source/`等；README明确公开主ViT分类实验，并提示其他实验/Jacobian版本可联系作者|有作者主实验代码，但不是现成的39变量回归实现；本项目回归kernel需单独说明|
|Lev|[作者仓库](https://github.com/mazailin/LevDataDiversity)有`Lev.py/embed.py/select_file_gpu.py`|有作者选择器；嵌入换成酵母输入是迁移，不是原文本实验复现|
|AL4PDE|[作者仓库](https://github.com/dmusekamp/al4pde)|有benchmark实现；不能另算一个2025新采样算法|
|Physics-PRE|[论文§2.2](https://arxiv.org/html/2605.21348v1)引用[作者相关CP-PRE](https://github.com/gitvicky/CP-PRE)；该库实际有残差工具和`Active_Learning/Burgers_AL_Joint.py`等示例，但README对应2025 conformal-UQ论文|有相关作者源码及AL示例；尚未核实2026短文的完整专用实验发布版，不能直接等号|
|OED|[期刊论文](https://www.sciencedirect.com/science/article/pii/S0045782525009478)声明开源；本地全文没有找到具体源码URL，本次题名/作者/仓库搜索仍未定位完整入口|尚未核实官方可执行代码，不能声称没有，也不能声称已找到|
|Low-Rank|[正式论文页](https://proceedings.mlr.press/v306/cohen-addad26a.html)和本地PDF可核对算法，未找到专用作者代码入口|当前项目是论文重实现及梯度能量实例化，不是作者源码移植|
|ERACS|[正式论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/92a035beb61e68f7a688467e5103a17c-Abstract-Conference.html)及本地PDF，未找到对应作者公开仓库|当前是论文重实现；正式结果0触发仍需训练侧机制校准|
|Clustering Sensitivity|[正式论文](https://proceedings.mlr.press/v235/axiotis24a.html)可核查公式；未核实该算法专用作者发布实现，文中引用DISTIL不等于提供了本方法源码|按论文独立实现，不能把通用参考库说成作者该论文完整实现|

对无仓库结果使用“未核实”，不由搜索无结果推断绝对不存在。PaPQS作者仓库当前为空，继续不计入“可用官方代码”。

### 7.2 对本课题的工程判断

以下成本指**从头忠实实现并适配39变量混合ODE**，不是已写完代码后的继续运行成本。原方法计算代价和代码开发难度也不同。

|方法|任务匹配|从头适配成本|最大的适配点|是否替换原8项|
|---|---|---|---|---|
|HGGS|很高，生物振荡ODE|中～高|标量频率改全轨迹；GF初始与查询计费；合法离散/连续生成|保留主表，不优先换掉|
|OGAS|高，仿真参数→代理轨迹|高|异步、扩散、replay/历史修正；合法域投影|已有适配，保留主表|
|OED|高，FNO/DeepONet数据生成|中|酵母信息量定义和合法输入优化，官方入口尚待确认|若更看重2026覆盖，最值得替换BAIT，BAIT保留补充|
|PICore|高，物理神经算子coreset|中～高|物理预热、每样本梯度、事件/初值及变量尺度|官方代码明确，仍优先补实现|
|Physics-PRE|高，物理代理选样|中|跳变不能按连续ODE差分，跨参数残差归一化|已有适配，保留；不冒充完整事件误差|
|PEAKS|中，通用监督数据选择且有回归分析|中～高|分类参考代码转实际回归梯度kernel；流式和持续训练成本|若优先已确认作者源码和2025主会，可替BAIT；需单列真值访问量|
|Clustering Sensitivity|中，通用监督coreset并支持线性回归|中|嵌入、代表点可查询性、loss距离与重要性权重|可替BAIT作补充方向；不是补2026的选择|
|ERACS|中，动态加速训练而非纯少标签AL|高|SNR尺度、OMP带权重选与持续训练回调|先验证触发机制，暂不替换稳定主对照|
|Low-Rank|中，通用数据选择|高（忠于理论目标）|基方向loss及理论条件；梯度能量是任务实例化|不为2026年份优先换入主表|
|Lev|中～低；几何选择器可迁移但原任务是文本|低～中|选择合适输入/隐藏嵌入，不能宣称保留文本语义实验|可作廉价多样性补充，不替代物理或生成代表|
|AL4PDE|高，参考实验框架|具体算法分别判断|统一接口和任务，不把框架当新算法|不额外占主表名额|

### 7.3 两个可选主清单，不自动切换

1. **优先任务匹配并增加2026覆盖**：Random / Power / LCMD / Physics-PRE / PICore / HGGS / OGAS / **OED**，用OED替BAIT。2026为PRE、OGAS、OED三项，其中PRE为Workshop；必须承认OED完整官方代码未核实。此为当前更契合用户目标的建议，原BAIT结果继续保留。
2. **优先已确认作者实现**：保留原清单，或以PEAKS替BAIT增加2025正式会议对照；不能把PEAKS改标2026，也不能把其访问约6.8万标签的监督选择当只查询1万的AL。若完全要求专用官方发布，可将PRE列补充、PEAKS列主表，但这样不是增加2026数量。

不建议为了“全是2026”把Low-Rank梯度能量实例化或Lev文本迁移包装成更贴合本课题的强对照。上述建议按任务与复现证据选择，不按real成绩淘汰强baseline。未更改当前八项配置、未移除BAIT、未启动实现或训练。

## 8. 后续澄清：PEAKS/Low-Rank可用性与Lev的机制重合

用户明确当前两个候选池均完整标注，因此读取训练侧候选真轨迹不是排除PEAKS、监督coreset方法的理由。继续只隔离合成holdout和real；标签访问、历史曝光和训练计算量是实验描述/成本，而非禁止使用。以下是候选方法评估修正，不自动覆盖已接受8项清单，也未改生产代码或开实验。

### 8.1 PEAKS：论文允许回归，公开主代码仍为分类

本地PDF第2节（印刷第2页）明确说研究的是分类，但问题定义和后续数学分析也适用于回归。**该声明不等于论文展示了39变量完整轨迹Relative L2的回归实验。**

再次读取[官方README](https://github.com/BurakGurbuz97/PEAKS)：公开主实验是DINO预训练ViT-B/16；其他架构、精确Jacobian等版本可联系作者。实际[训练代码](https://github.com/BurakGurbuz97/PEAKS/blob/main/Source/trainer.py)使用`CrossEntropyLoss`；[评分代码](https://github.com/BurakGurbuz97/PEAKS/blob/main/Source/acquisition.py)的PEAKS分支使用`softmax`、`one_hot`、真实类别logit及类别计数。未见可直接替换运行的官方轨迹回归入口。

回归可沿其一阶参数更新/NTK分析改写：候选一步训练对参考预测的变化约为`-eta J_ref J_candidate^T grad_output(loss_candidate)`。若把目标定义为降低参考回归loss，分数对应`grad_theta(loss_ref) dot grad_theta(loss_candidate)`。实际Relative L2的导数、参考集合和输出层近似都须明确，不能仅把交叉熵改个名字，也不能把Pattern分类当完整轨迹回归。

当前 `baseline/peaks.py` 采用均值参考梯度与候选梯度内积，并保留流式百分位接受、缓存刷新和新旧混合训练。因此可以作为 **PEAKS-Regression-Reference** baseline，但应称“基于原文的回归适配”，不是“作者提供的官方回归版”。参考来自训练侧已选样本，是当前项目选择，不等同原分类的同类平均logit目标。

### 8.2 Low-Rank：可以作为baseline，区分论文与当前目标实例化

先前“改动较大、不优先”不等于不能作为baseline。原框架接受一般loss和嵌入；本地正文§4既有平方范数loss实验，也有以每样本梯度范数为替代目标的实验，并非只可用于分类标签。

需要说明的具体问题是：SVD基方向通常不是合法酵母样本；即使候选池全标注，也不能据此得到任意基方向的真实ODE输出。原Algorithm 1用真实数据行张成的子空间缓解这类查询问题；Algorithm 2允许非数据基方向时仍需定义其loss。可以用论文支持的代理目标实例化，不能无定义地给基方向制造真值。

当前 `baseline/low_rank.py` 实际使用：

- 嵌入：模型输出头的逐样本Relative L2梯度的随机投影。
- 方向目标：`ell(v)=||v||²`，是梯度平方能量，不是轨迹Relative L2。
- 保留Algorithm 2的低秩投影、残差、敏感度概率、有放回抽样和逆概率权重。
- 最终神经网络仍按原Relative L2训练，并未把训练目标换成梯度能量。

因此 **Low-Rank-GradientEnergy** 可以纳入2026 baseline，不能宣称直接继承原轨迹loss保真的定理。原文替代目标段写的是梯度“范数”，本项目是“范数平方”，不应称完全相同。独立复现可接受，但若要求更接近原文，应增加对应目标对照并在训练/合成开发侧核查，不根据real测试结果选版本。

### 8.3 Lev：共享模型特征不等于相同选择器，确实属于成熟信息几何家族

[作者Lev.py](https://github.com/mazailin/LevDataDiversity/blob/main/Lev.py)对已选集合累计Gram矩阵。固定一批时，候选分数为：

`q(x) = phi(x)^T [ (1/|S|) sum_{i in S} phi_i phi_i^T + lambda I ]^{-1} phi(x)`。

然后选该批高分、更新Gram，并分块执行。原文文本嵌入可以替成FNO隐藏特征；这是特征迁移，不自动改变选择公式。当前项目入口仍是 `lev_input`（参数＋初值），**尚未切为模型隐藏特征版**。

|选择器|相同特征上的核心区别|
|---|---|
|Lev|逆Gram二次型，优先当前已选集合覆盖不足的特征方向|
|LCMD|距离加权的候选簇覆盖，选优先簇中的远点；不是逆协方差打分|
|CoreSet/MaxDist|与已覆盖集合的最小距离，再选最远点；不是Gram信息量|
|BAIT|以目标特征集的预测方差/信息矩阵目标作前向选取和后向删减；并非仅取最大leverage|
|MaxDet/D-optimal（更广泛的同族方法）|单点加入时logdet增益为`log(1+phi^T G^-1 phi)`，与相同G下的leverage排序一致|

所以Lev不应简单说成“等于LCMD/BAIT”，但也不能因名称/预印本年份不同就称全新的独立采样思想。归一化、正则随集合大小变化、块内批量更新、是否逐点重算和初始化等差异决定与某个MaxDet实现是否真正等价；不笼统断言整套算法完全相同。

本地AL4PDE默认配置包含`max_dist/lcmd/bait/coreset_maxdist/top_k/power`等，没有独立`maxdet.yaml`；不要把BMDAL更广泛的选择器集合全称为AL4PDE默认已评测方法。Lev与已有特征家族有明显机制重合，可保留低成本补充，但其2026预印本不能被当成leverage思想的首次提出。

### 8.4 当前评估建议

PEAKS与Low-Rank均具备纳入baseline的理由，可在现有方法上补足适配核查并独立命名，而非因任务原来包含分类就排除。若保持8项，PEAKS可替BAIT，Low-Rank可作为PICore尚未实现时的已有2026适配替补；这只是清单建议，不表示两者覆盖相同机制或PICore无需完成。Lev属于与已有信息几何对照较重合的补充，不优先单独占近期方法主名额。原有BAIT/PICore计划及全部结果继续保留。
