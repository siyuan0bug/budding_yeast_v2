# Budding采样Benchmark与Baseline实施对比方案

初稿：2026-11-13；实现更新：2026-11-14（Asia/Shanghai）。状态：**P0 工程协议与 P1/P2 方法接口已落地；正式训练校准及算法排名尚未执行，P3 仍待明确适配**。首次交付见第13节，最新职责解耦与扩展接口见第14节。

**最新轮次以第16节为准**：增量方法五轮2k/4k/6k/8k/10k，每次新增2k，epochs100/100/100/100/200；逐轮训练集归一化保留。第13–15节中的四轮验收、协议路径和更新次数是修改前记录。

已完整阅读项目根目录的`参考实施方案`全部606行，核对`papers/baselines/sampling/`的10篇论文主要算法、实验与相关附录，检查`baselines/sampling/`的14个本地仓库及现有项目接口。原参考文件、两个完整池和历史结果保持不变；本轮在用户确认后迁移工具代码并实现离线协议。以下原方案中的未来时描述以第13节实际状态为准。

## 先看这张表

| 问题 | 本方案建议 |
|---|---|
| 比较什么？ | 第一层比较LHS与mutation两种生成范式；第二层在同一来源内比较选择器。两层组成交叉实验，不把LHS当作与LCMD同层的算法 |
| 使用什么模型？ | 主实验只有现有FNO；模型创新与采样创新分开。其他骨干仅作后续迁移验证 |
| 训练多少数据？ | 固定池方法最终10000条不同有效输入；增量检查点2000→4000→6000→8000→10000，每次追加2000 |
| seed如何设置？ | 生成数据固定42；本轮实验统一experiment/split/initial/selection/model seed=42，未来统一换成2026或123；不同用途派生独立随机流 |
| 初始训练集如何选？ | 从每池划分后的90k可选集合均匀无放回随机取2000条，同pool/seed下所有通用增量方法共享 |
| 候选有完整轨迹和标签吗？ | 有。实际保存完整，但不同方法按原算法需要访问字段；使用候选真值的方法可以比较，须明确其标签成本 |
| 哪些方法优先？ | Random、CoreSet/MaxDist、LCMD、TopK、Power、BAIT、VeSSAL；加入物理残差TopK/Power、Lev和已标注误差选择形成第二批 |
| HGGS的VeSSAL能直接当作者原版吗？ | 不能；其嵌入使用真实预测误差，与作者默认分类实现不同。优先复用作者采样内核，公开FNO适配方式 |
| 所有方法放在同一个排行榜吗？ | 按信息条件分表：固定池模拟AL、已标注数据筛选、开放生成。可以共同展示最终预测质量，但不能混称等标注/等仿真预算 |
| 是否重写一套工程？ | 现有工具迁入`utils/sampling/`，baseline与ours分目录；复用FNO与既有方法，补离线oracle、权限视图、预算账本和作者代码适配器 |
| 是否开始运行？ | 已完成接口回归与小型CPU训练、共享划分/real reference准备；没有启动正式长实验或新的候选池仿真 |

## 1. 相对参考方案保留和调整的内容

| 参考内容 | 处理及原因 |
|---|---|
| 两个完整池各100k、最终10k、独立real评测 | 保留，补充两池参数域与求解精度差异的披露 |
| 固定池与HGGS/OGAS生成式方法分轨 | 保留；生成式方法不仅多了仿真，也改变了可访问输入域 |
| 按来源保留原算法、适配版分别命名 | 保留，并冻结到具体commit和实验配置，不能只写“基于某仓库” |
| 增量预算2k→4k→8k→10k | **不采用**；沿用用户已确认的2k→4k→7k→10k。原论文翻倍日程另作复现配置，不能偷偷替换主协议 |
| 原始训练结果＋最终统一FNO重训 | 保留思路；若最后一轮已经完全满足统一冷启动协议，直接复用该权重，避免无意义重复训练 |
| BAIT、LCMD都按最后层特征理解 | 细分基础配置与论文实验配置。当前项目是预测轨迹分支；论文主要配置的latent分支仍需提供接口 |
| 选择Row Subset版本即可解决Low-Rank非法基向量 | 不成立：选真实行构造子空间后，正交基仍通常不是合法ODE输入，公式中的基向量loss仍需定义 |
| 全方法立即全部移植 | 调整为分阶段验收；没有明确回归适配或作者实现的项目先列入候选清单，不用简化公式冒充完整复现 |
| 默认batch64、LR3e-4、300epochs、纯MSE | 不直接照搬。这是参考建议，不是所有论文的共同标准；训练配置需围绕当前FNO与2k–10k任务统一校准 |

## 2. 数据底座和固定划分

### 2.1 已有数据，不重新生成

根目录：`/home/users/zsy/mydata/budding_yeast/offline_pools/`。

| 来源 | 读取入口 | 当前事实 |
|---|---|---|
| mutation | `mutation_100000_seed42/dataset.json` | 100000条、19模块、1159种合法L1–L3集合均有覆盖；最终L1/L2/L3=1017/12972/86011 |
| LHS | `lhs_matched_100000_seed42/dataset.json` | 100000条；45模型参数匹配mutation提案边际，90背景参数±50%，另6控制参数；无mutation身份 |

每条物理输入包含141维参数、39维实际初值及相关事件/时间控制；输出为39×500、210min轨迹，包含BIO、ABC和中性六分组。metadata是否可用由方法能力声明决定，不能用LHS的`L=-1`当作真实突变阶数。

两池内部均通过唯一有效输入、real排除、初值、标签和归档检查。**跨池重复检查还没有执行，必须在划分时补齐，不能把池内检查当成跨池检查。**

两种范式并非“只改变LHS还是随机”的控制实验：mutation的90背景参数固定WT，LHS扰动；mutation的碰撞救援和去重也改变了最终接受边际。论文称为“生成范式对比”，不能把差异全部归因于生物模块关系。若后续需要识别单一因素，再增设共同参数域消融，不覆盖用户当前两池。

### 2.2 建议从每池固定留出5%验证、5%测试

| 每池用途 | 数量 | 是否可被选择 |
|---|---:|---|
| Selection universe | 90000 | 可以，最终从中形成10000条训练输入 |
| Synthetic validation | 5000 | 不可以，只供共同的开发与checkpoint选择 |
| Synthetic test | 5000 | 不可以，只做冻结后的评测 |

这是本方案的新划分建议，**尚未实际写出split文件**。100k仍是源池总量，不能把扣除holdout后的90k称为100k可选训练候选。

执行顺序：先计算跨池有效输入身份，再将完全重复输入作为同组，以`split_seed=experiment_seed`随机分配；本轮为42。重复组不得跨任一来源的train/val/test边界。若组约束使精确5000不可达，保存实际计数，不拆组凑数。两池之间的训练输入重复若存在，可以同属训练侧；必须避免A池训练输入在B池测试里出现。

主划分继续**按seed随机，不按Pattern或mutation家族分层**。家族OOD/邻域隔离只作独立补充实验，不改变主划分。无论哪种方法，不得访问共同holdout的逐样本真值、标签或特征用于采集评分。

固定两套验证与两套测试，所有方法都在两域上评价。建议checkpoint指标为两域validation的物理空间macro RelL2等权平均，不按real误差选checkpoint。这样每个方法使用相同的验证目标；共同验证集的10000条标签也必须列为共享开发成本，不包含在选择训练10k内。

### 2.3 初始化、种子和预算

- **用户确认：本轮只用统一实验seed=42，不同时启动3个seed。** 未来可将统一seed改成2026或123，完成对应的整套方法对照。
- `generation_seed=42`确实是数据集生成种子。两个现有源池固定复用，未来换实验seed也不重新生成20万轨迹。
- `experiment_seed=split_seed=initial_seed=selection_seed=model_seed=s`；本轮`s=42`。这些名字只区分随机操作的用途，并不要求数值不同。原方案把split单独固定成日期种子是可选设计，现按用户意见取消。
- 每个s先产生对应的随机split，再从该split的90k可选输入均匀无放回取2000条，形成共同D0。该pool/s的所有通用增量方法共享相同split、D0和每轮主模型初始化；不能每个方法自行重抽初始集。不同方法在后续采集中可以选出不同样本。
- 实现时从统一s按阶段、pool、轮次、ensemble成员等固定标识派生独立随机流，并保存派生规则/状态。不能共享一个会被其他阶段消耗的全局RNG，也不能让ensemble两成员采用完全相同的初始化流。**统一用户seed不等于所有随机操作复用同一串随机数。**
- 换s会同时改变split、D0、选样和模型随机性，因此未来跨seed统计反映这些变化的综合影响。公平性要求同一s内各方法共享对应条件，不要求42和2026拥有相同holdout；跨seed不迁移已训练权重或用另一seed测试表现调参。
- 初始集正式采用随机选取，替代旧在线路径的KMeans初始化。KMeans只保留为未来可选消融，不是本轮默认。
- 四轮累计2000/4000/7000/10000，即新增2000/3000/3000。所选训练样本单调累积、不可重复；D0计入10k。
- 特殊一次性/批内初始化方法另设native配置，不能称为遵循共同D0的增量AL。2k/5k/10k静态随机子集可由同一排列嵌套获得，5k不是主AL的第五个检查点。

已核对的作者代码支持随机初始集的选择：

- AL4PDE `scripts/gen_data.py:generate_data`通过Task的参数/初值生成器生成初始样本再求解，`main`还把离线数据生成seed固定为42。它是随机生成初始数据，不是从预计算90k里做KMeans；本项目从已有池随机抽取是离线固定池适配。
- VeSSAL `run.py`的“generate initial labeled pool”直接shuffle全部索引，取前`NUM_INIT_LB`条。
- PEAKS `Source/data_handler.py:initialize_T_s`用`random.sample`选初始样本。

因此共同随机2000条适用于本轮通用增量基线；HGGS的GF、Lev的批内随机起点等特殊native步骤仍单独处理，不能据此声称所有论文都使用同一种初始化。

### 2.4 数值质量的前置处理

保持用户既定规则：不按大值或有限负值删除，不截断原始物理轨迹。两池当前BDF为rtol=1e-4、atol=1e-6；LHS有4条最低值低于-1e-4。12条诊断轨迹以两档严格容差复算均成功、六分类不变，但原解到严格解的完整轨迹RelL2最大约2.30%。

正式冻结前做共同的数值质量评估：对两池固定seed随机抽样，加上预先定义的事件/负值诊断样本，收紧容差复核轨迹与分裂时间。按两池同一规则决定是否采用高精度修正覆盖层；当前池不覆盖。若修正，参数和样本身份不变，但轨迹、所有派生标签和指纹必须同步更新。不能仅给某个方法挑中的样本“提精度”，也不能只处理某类badcase。Real参考轨迹亦需记录求解协议。

本方案没有把12条复算外推为20万轨迹已经严格精度验收。数值协议冻结与后续训练收敛校准是不同检查。

## 3. AL4PDE究竟能看到哪些真实轨迹？

**AL4PDE选样时不能看到尚未选中候选的真实未来轨迹，也不能看由这些真轨迹计算的BIO/ABC/六分类标签。它当然可以看到已经选入训练集的真实轨迹，否则无法训练模型。** 候选参数、初值和时间网格是允许访问的输入；初值不是未知的未来输出。

用每池90k可选输入、初始2000条举例：

1. 按seed随机选出2000条，把它们的真实ODE轨迹交给FNO训练。
2. 对剩余88000条，选样算法只能读取参数/初值。Uncertainty/Power计算模型预测分歧；LCMD/BAIT计算预测轨迹或隐藏特征。**此时不读取这88000条的真实轨迹或真实Pattern。**
3. 挑出本轮新增的2000个ID之后，才从已生成数据文件中读取这些ID的真实轨迹及派生标签，加入训练集。
4. 用累计4000条真实轨迹重新训练，再继续选择下一批。

读取预存真值代替了原在线流程的“选中后调用ODE”，省掉重复仿真，不改变先选后揭示的次序。用于评分的预测不是拿来替代真实训练目标的伪标签；实际训练样本仍使用已生成的真实ODE输出。

| 数据对象/阶段 | 可读参数和初值？ | AL4PDE可读真实未来轨迹及其Pattern？ |
|---|---|---|
| 已选2000条，用于训练 | 可以 | 可以 |
| 剩余88000条，正在评分和选择 | 可以 | **不可以**；只可计算模型预测/特征 |
| 刚选出的新增2000条，选择结束后入库 | 可以 | 可以，从预计算文件读取 |
| 共同validation | 可供独立验证器 | 真值仅用于共同验证指标，不能直接传给采集评分函数 |
| synthetic test和real benchmark | 可供独立评测器 | 真值仅用于评测，不用于训练或选样 |

下面的A/B/C分类是为了另外安置“原本就要读取候选真误差”的数据筛选方法和“生成新输入”的方法，**不是允许AL4PDE改用候选真值评分**。这里的“label”同时包括完整轨迹和它派生的类别。

| 轨道 | 可访问的信息 | 可以回答的问题 | 主预算 |
|---|---|---|---|
| A：AL4PDE等固定池模拟AL | 已选训练样本真值可见；未选候选真值不可见，只读其输入并计算预测 | 在限制已揭示输出的条件下，如何更有效增加训练数据？ | 10000个唯一查询训练输入＋共同验证成本 |
| B：完整标注池筛选 | 90k训练侧候选的真实轨迹/标签可以参与评分；holdout仍不可见 | 已有100k完整仿真数据，如何挑出更有用的10k训练子集？ | 最终10k，另报实际访问标签数，可达90k |
| C：开放生成 | 按方法生成新的合法输入并查询ODE；可用起始数据按实际记录 | 固定查询/算力成本下，生成式策略是否优于固定池选择？ | 全部ODE尝试、成功解、唯一标签、最终训练输入分别统计 |

三种问题都合理。**B轨道读取训练侧候选真值不是测试泄漏，但它不是仅标注10k的AL。** A轨道虽然磁盘有全部轨迹，程序仍须经oracle显式揭示。使用预计算真值筛选Pattern的ours应进入B；只用已查询标签训练Pattern预测器的ours进入A，名称和账本必须区分。

例如HGGS的`UncertaintySampling.py`读取未选候选的`target`计算真实预测误差，才属于B；AL4PDE的`uncertainty_based.py`只由输入调用预测不确定性，属于A。原方案并未建议将两种机制合并，现显式写出以免把“已选样本有真值”误读为“全部候选真值可见”。

同一pool/split的合法候选mask全方法一致；新离线协议不自动继承旧六种启发式策略的合成速率门控或其他方法专属过滤。若算法确有论文规定的域限制，另报其可选域与拒绝数，不能继续声称与其余方法在同一候选集合比较。评分异常、NaN或特征失败须有统一且留痕的处理，不能把失败点当0分或静默删除。

动态替换/重采样另加`training_exposure=dynamic`：最终驻留10k不代表历史只训练过10k。CREST、ERACS、HGGS-WRS等此类方法报告累计见过的不同输入及训练总步数，不能挤进“查询10k”排名。

## 4. Baseline清单：来源、适配和优先级

优先级：P1建立可靠主比较；P2扩充对照；P3需要专项适配后再决定进入正式排名。优先级不表示方法优劣。

| 方法/建议实验名 | 优先使用的来源 | 本任务机制及信息条件 | 优先级 |
|---|---|---|---|
| Random | AL4PDE `pool_random`等价机制 | 90k均匀无放回，静态10k/增量版；A/B共同底线 | P1 |
| MaxDist-Pred；CoreSet-LL | AL4PDE＋bmdal_reg | 预测轨迹/最后层特征的最大最小距离；A，两个表示分别命名 | P1 |
| LCMD-LL；LCMD-Pred | AL4PDE论文配置`ll_mean`；bmdal_reg | 距离加权簇代价＋簇内最远点；A，保留当前Pred作适配消融 | P1 |
| TopK-Uncertainty | AL4PDE `uncertainty_based.py` | 两个独立成员的预测分歧，取TopK；A | P1 |
| Power/SBAL | 同上 | 同一不确定性，概率∝score^β、无放回；β=1为官方默认起点；A | P1 |
| BAIT-LL；BAIT-Pred | AL4PDE实验配置`bait`＋bmdal_reg | 特征缩放、posterior变换及前向/后向信息选择；A | P1 |
| VeSSAL-Penultimate；VeSSAL-Reg | **asaran/VeSSAL** | 保留流式接受率和体积更新内核；前者用作者支持的隐藏特征分支，后者需回归梯度适配；A | P1/P2 |
| Physics-PRE-TopK / Power | Polanska论文Eq.5＋作者链接CP-PRE/AL4PDE | 预测轨迹ODE残差、最近已标注训练参考归一化；A | P2 |
| Lev-Input / Lev-LL | mazailin/LevDataDiversity | 按作者分块、随机起点、正则Gram/Cholesky、top-b更新；A/一次性选择 | P2 |
| HGGS-Residual-TopK | **HGGS自己的**`UncertaintySampling.py` | 候选真实预测残差排序；B。公开名称不误叫普通uncertainty | P2 |
| Clustering-Sensitivity（Axiotis2024） | 论文Alg.1/附录C；未定位到作者专属实现 | 聚类代表点loss＋距离敏感度、概率及权重；B，或对额外查询计费的A适配 | P2 |
| LowRank-Sensitivity2026 | 论文Alg.1/相关附录；未定位到作者专属实现 | 低秩、投影loss及残差敏感度、加权有放回抽样；基向量loss尚需适配 | P3 |
| PEAKS-Reg / Pattern-Transfer | BurakGurbuz97/PEAKS | 原版分类流式误差＋kernel规则；回归或分类迁移须另行定义；B | P3 |
| HGGS-Continuous / Mutation-Constrained | HGGS作者仓库 | GF＋残差GMM分层＋Multigrid Genetic Sampling，新增真实ODE；C | P3 |
| OGAS-Continuous / Mixed | 论文v2作者链接melissa-sa/ogas | 条件生成器＋历史难度缓冲＋密度比纠偏＋基础提案混合；C | P3 |

参考方案的13类全部保留在候选清单，补VeSSAL和PEAKS；不承诺15类都能无需适配直接运行。LL接口补齐后可以恢复此前未实现的`coreset_maxdist`，但历史结果不能回溯改名为LL版本。

其他本地仓库的定位：`distil`、`DeepCore`、`cords`是算法集合/工具库，其实现只有在与目标论文和版本核对后才采用；不能因为Axiotis引用DISTIL就称DISTIL含其作者官方算法。`CREST`对应另一篇ICML2023论文，不是本地Cheng2025 ERACS的官方实现。`DiSF`与`dsir`主要是文本数据场景，当前不列为强制酵母对照。Xu2026 OED论文可作开放生成扩展，但其方差/耗散/频谱指标需重新解释为细胞周期目标，且本地尚未定位到相应官方仓库，暂不列入P1。

## 5. 必须落实的原版差异

### 5.1 VeSSAL：作者版与HGGS版不能混用

作者版：[`query_strategies/vessal.py`](https://github.com/asaran/VeSSAL/blob/f506fe67106df14e3fe478e6ff37b4af2de91158/query_strategies/vessal.py)的`streaming_sampler`维护当前协方差及其逆，通过体积增益、剩余预算/剩余流长度的目标接受率和Woodbury更新逐点决定。`strategy.py:get_exp_grad_embedding`默认基于分类预测概率和假设标签构造期望梯度嵌入；不是先看候选真实分类误差。

HGGS版：`Baseline_Training_code/VeSSAL.py`借用了体积选择框架，但`yu/nn/model.py:get_last_layer_grad`计算的是`||prediction-target||₂ × penultimate_features`，读取候选真值；又由均匀生成器先生成并求出候选输出。这是**HGGS的回归改写版**，不是作者默认的无标签梯度嵌入实现。若保留，命名`HGGS-VeSSAL-Residual`并进入B/C相应成本表。

推荐先做`VeSSAL-Penultimate`：保留作者明确支持的`embs=penultimate`分支，只把特征提供器替换为FNO隐藏特征；它不含默认分类梯度的不确定性项，需如实命名。`VeSSAL-Reg`作为后续适配：使用归一化轨迹平方损失的最后层Jacobian，以及仅由已查询数据拟合的预测分布构造期望梯度；不能把回归标签直接设成自身预测，否则平方损失梯度恒为0。预测分布、随机秩/投影及相关代价都是适配的一部分，不称为未经修改的作者实验。

固定seed打乱候选流；`single_pass`、`early_stop`、初始逆协方差尺度、候选不足时的random filler按选定作者配置记录。**不得默认假设正好选满k**：先验收边界情况，使用作者已有fill_random分支的适配要报告填充数量，不能静默补样。固定池多轮重扫与真正单遍流式实验也应分开。

### 5.2 AL4PDE：特征由完整配置决定

官方基础`config/acquisition/lcmd.yaml`、`bait.yaml`使用`use_latent_space:false`，但论文实验覆盖配置`experiment/sub_exp/ll_mean.yaml`、`bait.yaml`设为true并指定`features:last_layer`。[LCMD实验配置](https://github.com/dmusekamp/al4pde/blob/c183431e43c122717beaf23cbd7f77aebcaaf5c2/config/experiment/sub_exp/ll_mean.yaml)、[BAIT实验配置](https://github.com/dmusekamp/al4pde/blob/c183431e43c122717beaf23cbd7f77aebcaaf5c2/config/experiment/sub_exp/bait.yaml)。

因此主复现优先`LCMD-LL/BAIT-LL/CoreSet-LL`；现有`features.py`从预测轨迹生成共享sketch的路径继续保留为`*-Pred`，不冒充LL。FNO只新增可选特征接口，不改预测主干、前向输出或训练损失：建议返回最终输出投影前、128通道、500时间点的隐藏特征；时间不是PDE空间轴，不能把`spatial_mean`翻译成时间平均。官方降到ODE的单空间点适配应保留时间轴，然后按固定同一sketch映射投影。

BAIT须保留train/pool同一个特征缩放、posterior变换、`sigma=1e-2`、`noise_sigma=1e-2`和`sel_with_train=true`的选定配置。所用512维随机sketch及种子同时记录。简化为原始参数KMeans或单次距离排序均不是BAIT。

TopK/Power共同使用2个独立训练成员，按选定官方`loss_mse`/方差分支一致聚合，不用单模型沿时间方差冒充不确定性；最终主预测使用第一成员，ensemble均值单列，避免与单模型基线暗中改变最终预测器。成员2只在后续需要选样的轮次训练，预算和耗时单独列出。

### 5.3 物理残差：不仅是把旧ours_residual改名

Polanska论文Eq.5采用平均绝对PRE，并用参数空间最近**训练样本真实轨迹的残差**归一化，处理不同参数下残差量纲/条件差异。该训练参考已被查询，所以A轨道允许；不能读取最近候选的未查询真轨迹。CP-PRE是论文脚注指向的残差工具来源，但其旧conformal/筛选示例不是完整神经算子Benchmark的直接替代。

适配本系统使用物理单位的39维ODE RHS，连续段计算`dŷ/dt-f(ŷ,p,t)`；事件条件、重置和#104定时干预必须一致。普通有限差分不能跨分裂跳变，候选事件位置只能由预测或独立预测事件服务获得，不能偷读未查询真值事件；训练参考可用已知事件。事件检测失败/无有效差分区间须记录，不当作残差0。近零分母采用预先冻结的floor并报告敏感性，不能让高精度训练参考的接近零残差使排序全被数值精度支配。ODE事件适配改变了原PDE流程，命名`Physics-PRE-ODE`，单独保留无归一化消融。

### 5.4 Axiotis、Low-Rank、Lev不是同一种“importance”

**Axiotis2024**：用实际样本代表聚类中心，计算中心loss，按`loss(center)+Λ_cluster×distance^z`建立概率；保留论文指定权重及敏感度估计。中心不能是无法查询ODE的任意质心。共同D0、额外中心查询和最终采样三部分分别计费，中心标签看过即计入，不能只数最后10k。参考附录的“20%初始化＋聚类/抽样”属于实验适配，不是所有任务必须的一条定理。未找到作者专属仓库，因此标`paper_reimplementation`，不标`official_code`。

**Low-Rank2026**：局部PDF Alg.1的分数同时含基向量上的loss和投影残差，附录亦有相关敏感度算法。即使子空间由真实候选行张成，正交基也未必是合法参数/初值，不能直接查询其ODE标签；仅做SVD残差排序不足以复现。需要先定义论文loss如何延拓到表示空间，或采用论文明确的可实现经验版本并披露loss/梯度代理。无法完成该定义时先不进入正式排名；`LowRank-Residual`可以作为几何启发式，但不得声称是完整论文方法。当前未定位到作者专属实现，不断言其在全网不存在。

**Lev2026**：复用作者`Lev.py:LevSelector`的分块贪心机制。保留每块随机初始化、`G=(XᵀX)/n+λI`、Cholesky三角求解、最高分top-b和Gram增量；不是简单按输入范数或SVD leverage一次抽样。源码在函数内固定`seed=0`，对照复现保留，主多seed适配显式改成传入的选择种子并记录差异；处理尾块和ceil配额后须正好输出目标数。主特征建议共同Z-score输入，不依赖mutation标签；LL表示另作变体。原文的文本embedding换为酵母输入/模型特征属于任务适配。

### 5.5 PEAKS与HGGS-US必须保留真实误差信息声明

HGGS-US源码按`||2(pred-target)||₂`排序，不是ensemble disagreement。建议名`HGGS-Residual-TopK-Aligned`，在本方案四轮上运行；原版5k起步、250×20轮只在锁定具体配置文件后列为native协议，不从名字猜预算。

PEAKS作者代码当前提供分类ViT实验，涉及真实类、预测误差、kernel/类别统计及流式训练。完整轨迹回归不能机械把39变量当39类。可研究输出向量损失与Jacobian对应的回归kernel适配，或明示`PEAKS-Pattern-Transfer`：先按真实Pattern分类目标选样，再训练统一轨迹FNO。后者是目标迁移变体，B轨道需记录Pattern可见的全部候选数量，不称为原版回归PEAKS。

`importance`不是一个足够明确的算法名。HGGS的`IS.py`是在已标注训练集合内按误差更新有放回训练索引；`WeightedReservoirSampling.py`还会替换驻留集。前者并不自动从90k选出新的唯一10k，后者历史接触量可超过最终驻留量。若纳入，分别作为`HGGS-IS-TrainingSampler`、`HGGS-WRS-Dynamic`，不要与Power或Axiotis混为一个baseline。

### 5.6 HGGS与OGAS开放生成

HGGS保留GF、残差GMM三分区、MGS探索/利用及多轮模型更新。原论文是系数→振荡频率，本任务改为完整39维轨迹；GF距离/响应敏感度的输出度量必须预先确定，并在补充材料说明。先做连续域适配；mutation域的交叉/变异必须通过现有模块、模式、强度及环境合法性校验，不能把KO和GAL的参数逐维插值后仍叫合法mutation。所有real碰撞继续排除，已有holdout输入同样不能成为新训练查询。

特别记录参考集成本：如果先读10000条标签筛出5000，再新增5000条训练轨迹，最终训练10000，但至少已经访问15000条不同标签；不是10k标注预算。不得把GF丢掉的标签从账本删除。

OGAS需保留训练难度条件生成、历史缓冲、更新时序、基础提案混合和密度比纠偏，不能用固定池真实误差TopK代替。2026-11-13核实[arXiv官方v2](https://arxiv.org/abs/2606.09949v2)于2026-11-12发布，当前本地PDF是v1；[新版作者代码入口](https://github.com/melissa-sa/ogas)与本地GitLab来源需核对版本。已检查v2 Algorithm1仍含缓冲/条件生成/密度比步骤。当前本地`melissa`子模块未初始化，缺少其实际采样实现，不能宣布OGAS已经可直接运行。

OGAS在本任务分两级：先定义可采样的连续参考域，完成其内核复现；混合mutation域再适配离散模块、模式、零原子、整数拷贝与连续强度。若基础参考从均匀连续域改成当前mutation提案π，纠偏目标亦应相应为`π/p_buffer`并重新验收；随意裁剪/最近邻吸附会改变生成分布。最终统一FNO重训可比较所生成10k数据；原版异步在线训练的native结果单列，不假装四轮暂停重训是OGAS原始运行时序。

## 6. 唯一样本、抽样次数、训练权重和实际成本

每项至少同时记录：最终唯一输入数、抽样槽位数、历史训练过的唯一输入数、选样阶段读取过的唯一真值数、标签读取次数、新ODE尝试/成功/失败数、主/辅助训练步数、选样耗时、总GPU小时及峰值RAM/VRAM。

A轨道只揭示已选点时，固定池10k最终训练数通常等于10k查询数；B轨道扫描90k误差时，标签访问数就是90k，尽管额外ODE调用为0。20万轨迹预计算成本作为公共固定成本另报，“0新ODE”不表示未付出数据生成成本。

### 6.1 有放回概率算法：不能同时偷改抽样和权重

原版有放回抽取m次可重复，必须保存每次p_i、重复次数c_i和权重。以n点均值风险为例，其估计为`Σ_draw loss_i/(m·n·p_i)`；合并重复行后累计权重为`c_i/(m·n·p_i)`。重复可只存一份物理轨迹，但不能丢掉重数。确定性包含的中心/初始集与后续抽样总体也须区分，不能重复计入同一总体。

严格“10000个不同输入”与原版“10000次有放回抽样”是两种协议。本方案保留native weighted结果，额外提供`*-WOR10k`统一适配；无放回权重需要其包含概率或明确采用启发式权重。不能在改为无放回后继续宣称`1/p_i`保持原理论无偏性。若做unweighted消融，另起名称，不覆盖主版本。

### 6.2 当前实现的具体缺口

`utils/losses.py:HybridMSERelLoss.forward`虽然接受`batch_weights`，实际返回未加权的MSE和RelL2；目前不能直接承载weighted coreset。未来应先把统一基础loss实现为逐样本reduction，再由训练器按方法声明的权重聚合；验证/测试仍按目标总体普通均值。不要每个mini-batch临时把权重除以其随机总和并声称是原版无偏估计。增加“手算小数据加权风险”“合并重复行前后等价”和“全1权重等于现有损失”验收。

## 7. FNO、训练协议与归一化

### 7.1 固定架构；配置先校准再冻结

保留现有FNO：width=192总隐藏通道、modes=24、4个主干块、现有完整初值＋参数＋时间输入、39×500输出，不为了某个采样方法修改主干。新增特征返回是只读接口，不增加预测容量。

**建议配置草案（尚未改脚本）**：AdamW、weight_decay=1e-4、四轮100/100/100/200epochs、warmup5/5/5/10、cosine且min_lr=1e-5。主损失先保持现有`hybrid_mse_rel`的MSE＋0.2×RelL2定义，精确冻结eps/clip和实现hash；增加归一化MSE-only作为训练损失消融。原算法选样专用的MSE/梯度度量仍按各自定义，不能因主损失统一而静默替换其采集公式。

小样本预算需要重新检查batch：旧512在2k/4k/7k/10k下每epoch仅4/8/14/20次更新，100/100/100/200共6600步；原70k实验是否收敛不能证明新任务也收敛。**建议把128、LR5e-4列为新任务的优先校准起点**，同样epoch下约26100步；旧512、LR1e-3作为共同校准候选。只用Random和固定synthetic validation比较，再冻结所有方法共同配置；不依据任何real测试指标选择。这里是供审阅的设置建议，不是宣称已经得到最优超参数。

各方法主模型同架构、相同每轮epoch/更新规则、优化器和checkpoint选择；额外ensemble、分类器、生成器的费用另计。收敛报告最近10/20epochs训练与验证error变化率、最佳epoch、末段LR、总更新步数，不能仅凭cosine末端曲线平坦宣称收敛。是否延长训练在正式批次前统一决定，不能单独为ours加训练。

### 7.2 两种输出分别保存，避免重复训练

- **Aligned最终模型**：固定池方法按共同四轮冷启动训练；最后一轮从头训练10k、使用统一配置，是主要数据质量比较。最后轮已满足条件则直接作为final，不再另训一遍。
- **Native模型**：有特殊在线/流式/动态更新机制的方法保留其原生训练结果，附实际训练步数；可以再用最终子集训练相同Aligned FNO，区分选择质量与运行时训练机制。
- 原native final train已接触额外数据的影响不会因重训消失；额外标签/计算成本照实保留。不同方法对齐的只有声明的预算维度，不能声称所有成本都完全相同。

### 7.3 保留当前逐轮train-only归一化

沿用用户已确认规则：每轮在**当前累计合成训练集**重新拟合参数逐维Z-score；轨迹采用`log1p(max(y,0))`后逐变量Z-score。模型输入float32，物理参数/实际初值原始存档float64。物理负值仍保留在源池；截零只是当前训练编码行为，须在实验配置披露，不能冒称ODE本来非负。

不得使用90k候选真实输出、val/test/real拟合归一化。带权重方法的归一化默认按选中唯一训练行做普通train-only拟合，与是否重采样槽位无关；若方法要求weighted normalization，应另列消融并记录，不静默改变。

输入几何算法可在可见90k候选的**输入**上拟合Z-score，属于无标签transductive预处理；这一尺度与模型每轮Z_train不同，但变换都为逐维Z-score，分别保存。零方差坐标处理固定且不靠holdout决定。模型特征sketch只拟合/生成一次共同映射，train/pool同时使用，不能各自独立投影。固定D0输出归一化可作为后续消融，不在本轮悄悄替换已确认的逐轮refit。

## 8. 评测、badcase与论文能支持的结论

### 8.1 必须分清reference的性质

当前126条“Real Mutants”对应文献的真实突变体定义，但本项目评测轨迹由canonical ODE仿真得到，**不是126条独立湿实验测量时间曲线**。论文应称literature-defined mutant simulation benchmark；表型文献依据与轨迹监督来源分开写，避免把结果直接称为已证明sim-to-experimental-real泛化。

126条历史结果和badcase已被多次查看并用于idea设计，不能事后声明它们是从未接触的盲测。新协议冻结后不再用其结果调参或选择ours版本；透明报告其历史开发用途。真正未用于新方案开发的synthetic holdout，以及可新增的预注册机制/家族留出测试，提供独立证据；不能只靠输入去重就宣称排除了历史测试集适应。

### 8.2 统一物理空间指标

主指标建议为逐轨迹`r_i=||ŷ_i-y_i||₂/(||y_i||₂+ε)`的均值，遍历39×500，ε固定并入配置；同时保留全局Frobenius比值，二者分别命名，不互相替换。所有预测先正确逆归一化。

每项输出MAE、MSE、macro RelL2、median/p90/p95 RelL2、最差10%均值（126条时13条）、Corr及其常量通道无定义计数；补9核心变量与逐变量结果。宏观相态错误另报ABC/六分组混淆矩阵、分裂次数误差及可匹配分裂时间误差；不能让整体RelL2掩盖“轨迹看着接近但分裂次数错误”。

BIO、ABC、六分组维持各自含义，P1/P2只是有限210min窗口分裂统计，不据此直接宣称长期极限环/衰竭或真实分岔。LHS按轨迹分组评价，mutation额外按实际模块数、KO/GAL/MC/复合背景分组；跨pool比较只用双方共有字段。

### 8.3 结果表与统计

1. A轨道：每个方法×两种训练来源；分别列LHS-Test、Mutation-Test、126参考指标及2k/4k/7k/10k曲线。
2. B轨道：相同形式，明确候选真值可见，标签访问数与最终10k分列。
3. C轨道：Native/Aligned质量，ODE尝试/成功/失败及CPU/GPU成本；不能与A共用“仅10k仿真”的标题。
4. 统一预算账本：预生成公共成本、标签权限、历史训练暴露、选择/主训练/辅助训练耗时、存储峰值。

本轮只跑统一seed42，报告单seed原值，不伪造跨seed标准差或声称跨seed稳定性。未来增加统一seed2026、123后，按同pool/seed配对比较并报告mean±std；由于split也随s变化，应说明统计包含划分、选择和训练变化。3个seed也不夸大显著性。统计bootstrap时保留种子/突变背景相关结构，不把39×500的时间点当独立实验重复。2k–10k曲线可报告按训练样本数积分的面积，注明非等计算成本。

历史Top20及#15/#22/#42/#50/#62/#124仅用于事后机制解释；新方法输入不能包含real ID、距离real的奖励或从real误差拟合的权重。真实参考已用于范围设计和碰撞排除这一事实也须披露。所有方法可以使用相同公开ODE与合法域先验，但如果只有ours使用mutation标签/图结构，应声明为先验贡献，增加去除该先验消融。

B轨道建议增设强而简单的`TruePattern-Stratified-Random`：使用训练候选真实六分组进行固定配额随机选样，再训练FNO；它是有标签筛选baseline，不叫无标签AL。它能检验复杂ours是否仅靠类别再平衡获益。A轨道对应预测Pattern版本，辅助分类器只在已揭示标签训练，二者不混报。

## 9. 工程落地：复用现有包，补明确接口

建议仍由`run_sampling_budding.sh`作为用户批实验入口，新增显式`benchmark_mode=offline_pool`及独立结果命名空间；默认旧实验保持可复现，不能直接传dataset.json就以为已有在线engine会正确改成离线选择。

| 现有位置 | 建议后续职责/修改边界 |
|---|---|
| `data/dataset_archive.py` | 复用manifest/chunks读取；新增按ID分块读取或轻量离线索引适配，不复制完整数据 |
| `utils/active_learning/pool.py`、`oracle.py` | 增加完整池输入视图＋离线真值查询；已仿真池追加时只揭示/读取，不再调用ODE |
| `contracts.py`、`registry.py` | 扩展方法能力、信息权限、weighted selection结果及来源；不强迫动态/生成式方法伪装成一次返回ID |
| `features.py`、`backends.py` | 作者特征约定、LL/Pred区分、sketch及bmdal适配；移除仅靠名称“官方”的歧义 |
| `strategies/`及建议`adapters/` | 策略逻辑与作者代码窄接口；不导入旧archive启发式实现充当新基线 |
| `models/baselines/fno.py` | 可选返回指定隐藏层特征，默认forward签名及输出完全兼容 |
| `train.py`、`yeast_datamodule.py` | 显式固定split/ID训练集、逐轮refit、逐样本weighted loss与统一checkpoint；禁止重新自动切分已固定holdout |
| `scripts/analysis/`、`scripts/eval/` | 来源核对、协议/预算验收、统一多表统计；主指标与当前metrics兼容导出 |

不必建立参考方案那样庞大的全新顶层`benchmark/`。可以把少量声明式协议放在`configs/benchmark/`，共享服务仍归现有包；是否新增该配置目录在实现时按规模决定。

建议接口语义：

```text
MethodSpec: source/commit/config, information_access, representation,
            selection_mode, supports_weights, native_training_protocol
PoolInputView: stable IDs, physical P/y0/controls, permitted metadata
LabelOracle.reveal(ids, purpose): exact stored trajectories/labels + access ledger
SelectionResult: IDs, probabilities/inclusion_probabilities, weights,
                 multiplicities, generated_input_refs, costs, diagnostics
EvaluationView: holdout only; never supplied to selector
```

接口能力检查在启动时执行：LHS不能要求真实mutation类别；A轨道不能注入全部真标签；缺LL接口不能回退Pred却沿用LL名称；不支持weights的loss不能接受weighted算法；C轨道不允许用最近邻池内标签冒充新参数的ODE输出。给选择器的只是受限视图，不能同时传含真实输出的整个DataModule再期待算法自觉不读。

## 10. 文件布局与存储

继续全部放机械盘，例如：

```text
/home/users/zsy/mydata/budding_yeast/
├── offline_pools/                         # 现有两个完整池，仅引用
├── benchmarks/budding_sampling/<protocol_id>/
│   ├── protocol.json / sources.json
│   ├── splits/                           # 各来源固定IDs、input signatures
│   └── initial/                          # 各selection seed的共同D0 IDs
└── experiments/budding_sampling/<protocol_id>/
    └── <pool>/<method>/seed_<seed>/
        ├── config.json / sources.json / train.log
        ├── selection/round_*/             # 入选IDs、p/weights、分数与预算账本
        ├── rounds/round_*/                # 数据引用清单、normalizer、checkpoint
        │   ├── loss.csv / loss.png
        │   └── eval/real_mutants/         # 预测图＋逐株CSV＋metrics_denorm.json
        ├── summary/                       # 全轮loss大图、曲线CSV/PNG、最终指标
        └── wandb/                         # 强制offline，不上传
```

`protocol_id`由域/求解覆盖层、split、预算、FNO、训练/归一化规范等指纹生成，使用简短可读前缀；不要求文件名堆满所有超参数。每轮只保存有序ID引用、切分、权重和归一化，不复制完整累计轨迹。生成式方法额外保存其唯一新轨迹块和来源（父代/生成器状态/求解日志），其他轮次仍只引用。

内存也按块控制：90k×39×500个float32预测约6.54 GiB，禁止每个并发进程都长期缓存全池完整预测/真实轨迹。预测特征分批生成并直接投影到例如512维（90k约176 MiB），原轨迹只在查询或训练小批量时读取；不要直接让现有DataModule把整个100k池展开成常驻训练张量。

目录是建议，不是本轮已经创建的正式结果目录。历史`experiments/al_budding`保留，不能覆盖或把旧1M在线结果并入新100k完整池主表。

## 11. 作者实现来源锁定表

下表是2026-11-13本地实际HEAD；正式实现还需记录tracked diff、实际import路径及依赖版本。**已克隆不等于已适配或已通过回归测试。**

| 本地仓库（`baselines/sampling/`下） | remote/定位 | 本地HEAD |
|---|---|---|
| al4pde | [dmusekamp/al4pde](https://github.com/dmusekamp/al4pde) | `c183431e43c122717beaf23cbd7f77aebcaaf5c2` |
| bmdal_reg | [dholzmueller/bmdal_reg](https://github.com/dholzmueller/bmdal_reg) | `08c28e131d9dc49cf1e47ddb80f7230ca4e10d72` |
| VeSSAL | [asaran/VeSSAL](https://github.com/asaran/VeSSAL) | `f506fe67106df14e3fe478e6ff37b4af2de91158` |
| HGGS | [hengrao02/HGGS](https://github.com/hengrao02/HGGS)，README另有旧clone地址 | `abf9387f2f6843d6be0ac30ca09f7dbf5feae5cc` |
| LevDataDiversity | [mazailin/LevDataDiversity](https://github.com/mazailin/LevDataDiversity) | `1aca946b23175bc71881ee422c01d016587830e5` |
| PEAKS | [BurakGurbuz97/PEAKS](https://github.com/BurakGurbuz97/PEAKS) | `58c581df7f89a27d41b3210cc98719c91b448072` |
| CP-PRE | [gitvicky/CP-PRE](https://github.com/gitvicky/CP-PRE)，物理残差工具来源 | `0a4719ea596750eb7219bc2da7f144c9f3f415dc` |
| ogas_experiments | [GitLab旧入口](https://gitlab.inria.fr/melissa/ogas_experiments)，[新版作者入口](https://github.com/melissa-sa/ogas) | `631b5cc366ff4650129b43202a8ae57f1bf1bbc2` |
| CREST | [YuYang0901/CREST](https://github.com/YuYang0901/CREST) | `04ec1d7748d7c1545e67aacd39d1bef96718537d` |
| DeepCore | [PatrickZH/DeepCore](https://github.com/PatrickZH/DeepCore) | `74ec709f5ccba3b1e921b06c6e312d57741de3f8` |
| cords | [decile-team/cords](https://github.com/decile-team/cords) | `8d10c7f5d96e071f98c20e4e9ff4c41c2c4ea2af` |
| distil | [decile-team/distil](https://github.com/decile-team/distil) | `d46f0b4e930f7ae70ff0b5bfacf1d7da93c1b9b4` |
| DiSF | [MediaBrain-SJTU/DiSF](https://github.com/MediaBrain-SJTU/DiSF) | `c17a3c2eb32cb3722b47e5994dd89b812d7543d8` |
| dsir | [p-lambda/dsir](https://github.com/p-lambda/dsir) | `4802a1c71cdb1ae3c8367e20df64fe14da6d597d` |

根目录`al4pde/`与`baselines/sampling/al4pde/`的tracked HEAD相同；前者有本地未跟踪说明文档。未来只选一处为生产来源，防止import顺序误用另一份代码。OGAS本地子模块记录`be91967d3af39da5d1a56b58f8eeb7b89763a42c`但目录为空，需在实际移植阶段补齐并冻结。

每种方法附一张实现卡：论文版本/页码/算法号、官方仓库与commit、选定配置、与原算法的逐项差异、表示和标签权限、概率与权重、随机流、模型训练要求、复杂度与实际测试记录。来源类型明确`official_code_adapter`、`paper_reimplementation`、`task_adaptation`、`project_heuristic`，不统一写成official。

### 论文依据与版本

- AL4PDE：[Active Learning for Neural PDE Solvers](https://arxiv.org/abs/2408.01536)，本地文件名年份2025对应ICLR2025，不能仅按arXiv首发年判断版本。
- VeSSAL：[ICML2023作者论文](https://proceedings.mlr.press/v202/saran23a.html)及上述作者仓库。
- Axiotis2024：[ICML2024论文](https://proceedings.mlr.press/v235/axiotis24a.html)，Alg.1/2与附录C。论文引用工具库不等同于作者算法已发布在其中。
- Low-Rank2026：[ICML2026论文页](https://proceedings.mlr.press/v306/cohen-addad26a.html)与[arXiv版本](https://arxiv.org/abs/2606.16045)，实现前锁定本地PDF和采用的算法编号，避免不同排版版本混用。
- Lev2026：[arXiv:2609.32484](https://arxiv.org/abs/2609.32484)，Alg.1及作者Lev.py；当前按预印本表述，不自行补会议录用结论。
- Polanska2026：[arXiv:2605.21348](https://arxiv.org/abs/2605.21348)，Eq.5及其后归一化说明；属于ICLR2026 AI/PDE workshop，不写成未经核实的主会。
- PEAKS：[ICML2025作者论文](https://arxiv.org/abs/2504.05250)及作者代码；分类到回归适配单列。
- HGGS：[作者论文](https://arxiv.org/abs/2409.12816)，本地文件首发年份2024；重点Alg.1、GF/MGS与官方实现，不将其baseline改写当对应算法作者原版。
- OGAS：[v2](https://arxiv.org/abs/2606.09949v2)与[作者代码](https://github.com/melissa-sa/ogas)；本地v1保留作版本差异证据。
- Cheng2025 ERACS：本地论文Alg.1；与CREST区分，本轮未确认作者代码入口，不纳入必选首批。
- Xu2026 OED：[期刊原文](https://doi.org/10.1016/j.cma.2025.118675)；开放生成备选，不冒充现有启发式`fisher_oed`已实现该论文。

## 12. 实施顺序和通过条件

| 阶段 | 做什么 | 通过后才进行什么 |
|---|---|---|
| P0 数据与协议 | 跨池重复、固定split/D0、数值协议、权限视图、ID读取、损失权重、输入dtype、训练Random校准与收敛日志 | 冻结protocol与源池指纹，再做算法排名 |
| P1 作者内核 | Random、TopK、Power、MaxDist/CoreSet、LCMD、BAIT、VeSSAL-Penultimate | 同一小数据/特征/随机流与作者选择器一致，LL和Pred分别验收 |
| P2 扩展对照 | Physics-PRE、Lev、HGGS-Residual、Axiotis；ours A/B明确版本 | 信息/权重/查询计费通过后进入共同FNO比较 |
| P3 专项研究 | Low-Rank、PEAKS-Reg、HGGS完整、OGAS、动态训练采样 | 先解决算法定义和合法域，再独立native/aligned实验 |
| 汇总 | 同一协议多seed、质量/尾部/效率/成本、badcase归因 | 文档与论文只使用验收通过且完整的实验 |

至少验收以下会改变科学结论的行为：

1. 同输入特征和随机数下，对照作者选择内核的索引/概率/矩阵更新；任务适配只影响声明的接口，不偷改算法。
2. 给未查询池真值/Pattern置换后，A轨道选样结果不变；置换测试集/real标签后，所有轨道采集结果不变。验证集允许按共同协议选择checkpoint，因此验证标签可能间接改变后续预测/采集；它不得直接成为候选打分输入。测试视图不可到达评分函数。
3. ID稳定、不重复选点、预算精确、split不漂移；B轨道全池真值访问全部计费，生成式拒绝/失败也计费。
4. float64物理输入进入模型前统一float32，恢复checkpoint带完整normalizer和字段顺序；复核先前LCMD dtype故障路径。
5. weighted loss和重复行合并的数学等价；不能只检查文件里存在weights字段。
6. 主模型冷启动、优化器/调度器重置、ensemble独立且预算对应；native例外公开。
7. 选样中断能从已提交ID/权重/随机状态恢复，日志有阶段时间；W&B离线；结果只引用源数据，无累计重复轨迹副本。

初稿阶段只交付方案；用户确认后的实际实施如下。

## 13. 2026-11-14 首次实施交付（目录/训练器随后按第14节归并）

### 13.1 目录和接口

`utils/active_learning`迁为`utils/sampling`。基线集中在`sampling/baseline`，项目方法在`sampling/ours`，离线协议在`sampling/benchmark`。loss、物理、评测和日志分别迁到`training/physics/evaluation/experiments`；旧模块名集中用`_compat.py`延迟别名，不复制实现。迁移前备份在`scratch/probes/sampling_migration_20261113/utils_before.tar.gz`，既有`archive`保留。

完整结构、命令和旧接口兼容规则见[`utils/README.md`](../../utils/README.md)。历史在线实验依然可用`SAMPLING_MODE=online`运行；`run_sampling_budding.sh`默认转入新的完整池离线CLI。Fission没有加入新Budding Benchmark，其原在线接口仍保留并有回归覆盖。

### 13.2 当前已实现的方法

| 方法名 | 轨道/实现来源 | 必须披露的适配 |
|---|---|---|
| random、topk、power | A；复用既有AL4PDE对齐的池选择 | 固定池Random；两独立成员的MSE分歧；Power β=1 |
| max_dist_pred、coreset_ll | A；bmdal作者后端 | 分别预测轨迹/最后隐藏层，共用512维随机投影 |
| lcmd_ll、lcmd_pred、bait_ll、bait_pred | A；bmdal作者后端 | LL/Pred明确分名，不能宣称是完全相同的特征配置 |
| vessal_penultimate | A；固定VeSSAL commit的作者类 | 最后隐藏层投影；CUDA调用改为通用device；流顺序seed打乱；补齐行为记入diagnostics |
| lev_input | A；固定Lev commit的作者类 | 单次输入空间选择，参数与初值Z-score；block配额精确分配，不伪装成四轮AL |
| physics_pre_topk、physics_pre_power | A；论文规则的ODE任务适配 | 当前canonical RHS、物理时间中心差分、分裂/事件邻域掩码、训练近邻归一化；不是原PDE算例直接复现 |
| residual_topk | B；HGGS真实预测误差的固定池适配 | 会访问未选候选真轨迹，不能叫HGGS完整GF/MGS |
| true_pattern_random | B；项目诊断对照 | 真六分类水位分层抽取，显式计费真标签 |
| clustering_sensitivity | B；Axiotis论文重实现 | 原生有放回draw-slot预算，重复ID聚合并使用重要性权重；最终不同输入可少于10k，不纳入unique-10k主表 |
| clustering_sensitivity_wor | B；明确的无放回适配 | 原生中心误差+输入距离分数；无放回、无权训练，不宣称保留原生无偏估计量 |
| ours_pattern_balance/context/support | A；原项目Pattern方法的离线适配 | 分类器只用已查询训练标签；候选真Pattern不可见；LHS允许没有mutation描述 |
| ours_sensitivity | A；原项目敏感性方法离线适配 | 仅mutation池；合法强度方向的预测探针与ODE RHS一致性，不查询探针真实轨迹 |

共21个已实现配置。默认批量排除两个额外Pred变体和原生带权有放回方法，避免无意将不同预算合并；`list`展示完整能力。Low-Rank、PEAKS-Reg、HGGS完整、OGAS四项仍显式不可用，原因分别是非法基向量loss、分类到回归内核、混合域生成/频率表示、缺失子模块和先验定义。它们不会静默回退到Random。

### 13.3 已冻结的共享数据协议

现有两个100k完整池未改写；共享协议目录为：

```text
/home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_37748cb66b1a/
  protocol.json
  mutation_inputs.npz / lhs_inputs.npz
  mutation_split.npz / lhs_split.npz
```

各池90000可选、5000验证、5000合成测试；各自从90000中均匀随机取2000初始训练输入。实际检查跨池相同有效输入为0；代码对存在重复的情况仍保证同输入不跨角色。输入缓存与划分共约77MiB，没有复制39×500轨迹。生成seed固定42，实验根seed当前42，split/initial/selection/model通过不同purpose派生随机流，未来只需统一修改根seed。

冻结real reference为`benchmarks/budding_sampling/references/real_reference_b227aee93901.npz`，126株仿真全部成功；只用于评测，不用于选择器或normalizer。主模型按两个合成验证域的宏平均trajectory RelL2选择checkpoint；测试域和real不选择epoch。

当前起始训练配置：FNO width192、modes24、blocks4、batch128、AdamW lr5e-4、weight_decay1e-4、min_lr1e-5；100/100/100/200 epochs，warmup5/5/5/10、余弦衰减、每轮冷启动，参数Z-score、变量log1p后Z-score均仅按累计训练集重拟合。**这些是待正式Random校准的起始配置，不代表已经证明收敛。**覆盖配置通过`--config`传JSON，整个实验协议共同生效，不按方法偷偷调参。

### 13.4 输出与断点

```text
experiments/budding_sampling/<protocol_id>/<mutation或lhs>/<method>/seed_42/
  config.json / sources.json / train.log / label_access.json / progress.json
  rounds/round_00/...round_03/
    dataset.json                         # 只引用源池有序IDs
    primary/{best.pt,last.pt,loss.csv,loss.png,training_summary.json}
    ensemble_member_1/                  # 仅需要ensemble且还有下轮选择时
    eval/metrics_denorm.json
    eval/real_mutants/{*.png,per_mutant_metrics.csv,metrics_denorm.json}
  selection/after_round_00/...02/
    result.json / selected.csv / added_samples.json
  summary/{rounds_summary.json,all_rounds_loss.png,al_performance_curve.csv,
           al_performance_curve.png,final_metrics_denorm.json,completion.json}
  wandb/offline-run-*/
```

单次方法只有最终round_03；原生clustering-sensitivity额外保存pilot。公共数据reader识别`yeast_sampling_selection`清单，并保留所有已选行为训练，不再次拆分；如有weights，会传到loss，不仅存个字段。新离线checkpoint由`utils.training.offline_fno.load_fno`读取，与旧Lightning checkpoint格式区分。

恢复时核对源代码、源池、split、行顺序与权重指纹；checkpoint保留优化器及随机状态，选样决定先原子保存后揭示真值。相同目录不能混入修改后的源码或协议。批量日志有阶段/epoch时间，默认失败后继续，W&B强制本地offline。

`completion.json`分别统计primary/auxiliary/pilot训练步数和秒数、选样秒数、不同训练输入、不同访问标签、draw slots、固定验证测试规模、新ODE调用0。`invocation_seconds`仅本次调用的墙钟时间，恢复前耗时不能据此忽略；共享池预仿真、real reference和验证成本另行报告。尚未实现完整CPU/GPU峰值资源采样，不应声称已完成全部论文成本审计。

### 13.5 验证与下一步

回归覆盖旧导入/多进程pickle、Budding/Fission调用、float32模型输入、原生作者内核接入、源池ID读取、权限账本、无重复预算、四轮冷启动、带权损失数学等价、预测图与checkpoint重载。测试使用小型CPU诊断数据，不作为正式126株性能或收敛结果。具体最新通过数量见核心文档5的实施记录。

接下来先校准Random的训练曲线，再在同一协议上比较算法；尚无新协议正式排行榜。两个源池和mutation/LHS分布未在这次工具整理中重新生成或改变。

## 14. 2026-11-14 职责解耦与去冗余（当前实现）

用户进一步明确：按职责独立组织、逻辑只实现一次，不为减少文件数而把所有功能塞进同一个文件。第13节记录的是首次实现；下面覆盖其目录、训练器和checkpoint部分。

### 14.1 稳定的层次与接口

```text
数据生成/读取(data) → 当前实验流程(sampling/benchmark)
                           ├─ select_samples(context,budget) → baseline/ours
                           ├─ LabelOracle → 访问授权和查询账本
                           ├─ 流程数据适配(rounds.py) → prepared DataModule
                           │                               ↓
                           │                   通用训练(training/runner.py)
                           │                               ↓
                           └─ 公共推理、指标与报告(evaluation)
```

* `models/`继续唯一负责模型结构；`models/yeast_lit_module.py`提供公共训练step、loss、AdamW、调度和物理解码。
* `utils/training/runner.py`提供`make_trainer`与`fit_model_round(model_factory, datamodule, ...)`。不导入采样流程、候选池或CLI，不维护FNO专用梯度循环；验证指标通过`monitor`指定。
* `sampling/benchmark/rounds.py`约50行，只核对已选清单/权重、准备当前流程的DataModule/验证域和模型factory，再调用训练服务。
* `sampling/benchmark/runner.py`编排当前完整池的轮次、查询和评测。新流程不依赖`train.py/eval.py`的命令行模块。
* `sampling/contracts.py`定义输入/输出视图，`sampling/query.py`定义查询服务；它们与模型训练相互独立。旧在线契约移入`archive/online/contracts.py`，旧导入通过延迟属性兼容。
* `evaluation/checkpoints.py`统一加载和推理，`metrics.py`统一指标，`reporting.py`负责CSV/JSON和轨迹图；`experiments/curves.py`负责训练/性能曲线。
* `data/normalization.py:NormalizationStats`负责唯一的统计拟合与编码，原四种基础变换函数保持不变；`data/dataset_archive.py`负责公共存储与行读取。

专用`offline_fno.py`、重复的`benchmark/model.py`、`benchmark/storage.py`、`benchmark/contracts.py`、`benchmark/methods.py`、`evaluation/sampling.py`和`experiments/sampling_plots.py`已移除。必要的公共能力按以上职责归并。历史在线引擎/参数池/ODE查询/追加/兼容入口隔离到`archive/online`，Fission与历史重放继续保留。当前非归档Python文件从78个降到63个（含`__init__`）；这个数量只作审计记录，维护边界才是验收重点。

### 14.2 下一版如何扩展

新增选择规则只需一个`baseline`或`ours`算法文件，以及`registry.METHODS`中的一条`Method(..., implementation='my_rule:select')`登记。新入口接收`SamplingContext,budget`并返回`SamplingResult`，不会改动训练、评测、当前流程runner或旧`offline.py`适配器。

修改轮次/预算/epochs/warmup通常只修改当前流程配置。换初始化/split策略修改当前`protocol.py`；新数据追加流程增加独立编排/适配模块，继续提供准备好的DataModule给公共训练服务。若改变输入输出契约，如池外生成，则显式新增相应契约，不用伪造池内ID绕过检查。详见[`utils/README.md`](../../utils/README.md)。

### 14.3 训练、数据与兼容边界

新轮次统一保存标准Lightning `primary/best.ckpt`、`last.ckpt`；归一化统计在`preprocessing_contract`中，训练恢复信息在`training_run`中。`eval.py`与Benchmark共用loader，并支持`--real_reference <冻结参考.npz>`直接按checkpoint统计评测。短暂存在的`yeast_sampling_fno_v1`文件只保留读取兼容，不再创建此格式。

此次整理将新训练执行规则对齐原主流程，包括公共warmup/余弦端点和梯度裁剪1.0。配置中的模型容量、LR及轮次预算保留；不能宣称整理前专用训练器的诊断权重与新实现逐位相同。正式长实验尚未启动。

当前共享协议目录为`/home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_c0d0d9a0169c`，新增显式模型配置后具有新指纹；与旧`seed42_37748cb66b1a`逐项比较，两个池的candidate/validation/test/initial IDs完全一致。只准备了输入缓存与协议，没有重新生成100k源池、改变mutation/LHS分布或ODE。原data根目录生成器保留原始字节。

### 14.4 验证范围

跨模型FNO/DeepONet训练、标准checkpoint独立评测、四种归一化共享统计、原指标兼容、Fission标签、实际ODE旧接口、候选真值权限和预算均纳入回归。新增扩展性检查：只登记独立算法即可调用、配置将四轮改为两轮、另一验证指标复用相同训练服务。中断恢复通过人为中止验证，连续/恢复训练的参数、历史与优化步数逐项相同。

本地W&B与四轮报告已重新验证：父环境设为online时仍强制offline，8张诊断预测图成功输出，标准checkpoint可重载。诊断不作为正式126株性能或收敛结论。最终回归数量记录在核心文档5及`scratch/probes/redundancy_cleanup_20261114/`日志。

本轮最终联合回归：**492项通过**，含新增扩展性、跨模型与逐位恢复检查；另验证8张诊断轨迹图、本地offline W&B和旧诊断PT读取。`train/eval/tune --help`、批量dry-run、AST及shell检查通过。没有启动正式长实验。日志：`scratch/probes/redundancy_cleanup_20261114/{final_regression.log,output_smoke.log}`。

## 15. 2026-11-14 作者实现的训练预算与归一化核对

本节为源码核对和建议，**不改变当前已确认的epochs、归一化或源池**。不同任务的作者配置不能视为酵母FNO的收敛证据；下面列的是具体入口/默认值，不代表仓库所有实验均相同。

### 15.1 作者训练协议

| 作者仓库/方法族 | 已核对的原始训练规则 | 与本项目的区别 |
|---|---|---|
| AL4PDE：Random、Uncertainty、Power等 | `config/main.yaml`默认每轮500 epoch；`models/wrapper.py:init_training`每轮重建模型、优化器和scheduler | 没有统一的“最后一轮翻倍”规则；默认短时间窗训练，一条轨迹可贡献许多窗口，epoch不等于本项目整条轨迹的epoch |
| BMDAL：LCMD、BAIT、CoreSet等 | `bmdal_reg/train.py:fit_model`默认256 epoch、batch256，每轮重建模型；训练完成后恢复验证集最优权重 | 最终轮默认沿用相同配置；原任务为表格回归，不能直接照搬其学习率/epoch数 |
| 作者VeSSAL | `query_strategies/strategy.py:train`重置网络，直到训练准确率达到阈值；CLI默认99% | 虽读取`n_epoch`，实际while循环并不使用它限制轮数；分类准确率停止规则不适用于FNO轨迹回归 |
| HGGS | `src/yu/tasks/train_code/HSECC/HGGS.sh`示例为首训2000、后续3000 epoch、batch40960，并有早停选项 | `last_epoch_n`在这里表示首次训练；`Genetic_Training`沿用模型/优化器，属于继续训练，不能当成四轮冷启动 |
| CP-PRE | `Active_Learning/Wave_AL_Joint.py`先训练500 epoch，随后每次追加继续训练100 epoch | 每种采集方法从共同初始checkpoint出发，方法内部继续训练；不是每轮冷启动 |
| DeepCore | `main.py`默认selection_epochs=40、最终子集训练epochs=200 | 筛选预训练和最终训练是两个预算，不是四轮AL各训练200 epoch |
| distil | CIFAR10随机采样示例`n_epoch=300,isreset=true`；`train_helper.py`还检查准确率与饱和条件 | 300是上限配置，不能保证每轮实际训练满300 epoch |
| CREST、PEAKS、LevDataDiversity | CREST默认总训练200 epoch；PEAKS以总优化更新数计费；Lev作者仓库提供嵌入/筛选内核 | 不是共同的四轮冷启动协议，不能为它们虚构同一套原生epoch设置 |

以上源码路径均相对于项目根`baselines/sampling/`下对应仓库。主实验应固定同一FNO训练协议比较选择规则；另将作者原生动态训练/曝光预算列为独立轨道，不能把两种问题的结果混排。

当前主模型batch128、drop_last=False，不含辅助ensemble和pilot时：

| 轮次 | 训练样本 | epoch | 每epoch更新数 | 本轮更新数 |
|---|---:|---:|---:|---:|
| 0 | 2000 | 100 | 16 | 1600 |
| 1 | 4000 | 100 | 32 | 3200 |
| 2 | 7000 | 100 | 55 | 5500 |
| 3 | 10000 | 200 | 79 | 15800 |
| 合计 | — | — | — | 26100 |

100/100/100/200是本课题的待校准预算，不是官方要求，也尚无新100k完整池协议下的正式收敛证据。旧70k训练实验不能替代当前2k/10k规模的检查。末轮加长只能改善最终拟合，不能补救早轮模型训练不足造成的选样偏差。

建议先固定Random的样本ID，在2k和10k两个规模比较当前预算与更长训练，仅用共同合成验证集判断是否仍有实质收益；再检查一个模型驱动选择方法。记录末段训练loss/物理验证RelL2的窗口变化、最优epoch和优化步数。余弦末段变平不能独立证明收敛，阈值应预先约定；不能用real/test结果反复调整训练预算。

### 15.2 归一化并无统一的作者默认

| 作者实现 | 统计来源与是否逐轮重拟合 |
|---|---|
| AL4PDE | `config/prob_model/model_wrapper/base.yaml`默认`norm_mode:first`，初始训练集拟合后冻结；`models/wrapper.py:97–106`也明确支持`every`逐轮按当前训练集重拟合。`TaskNormalizer`为均值/标准差归一化，没有本项目log1p变换 |
| BMDAL | 实验入口`run_experiments.py:run_experiments`默认使用初始训练+未标注池的输入X拟合一次，然后固定；可改为只用初始训练X。`data.py:TaskSplit`在Z-score后还采用`5*tanh(0.2*X)`，该处没有使用未查询输出y拟合统计 |
| VeSSAL | 图像任务使用固定的MNIST/SVHN/CIFAR通道均值和标准差，不按每轮新查询标签重算 |
| CP-PRE Wave | 初始训练数据拟合Min-Max输入/输出变换，后续训练复用已有normalizer |

本项目当前保留用户已确认的`every`：每轮仅按累计已查询合成训练集拟合P的逐维Z-score、Y的逐变量log1p后Z-score。验证/test/real/未查询候选轨迹不参与拟合；每轮冷启动不存在旧权重直接接入新坐标的问题。它有AL4PDE可选实现依据，但不能称作AL4PDE的默认做法。

逐轮拟合可以适应新增训练数据的幅值分布，但不同方法/轮次得到不同Y标准差，也会改变标准化MSE对变量的有效权重。因此不能用归一化loss数值直接跨轮/跨方法排名，继续使用共同物理空间验证与测试指标。冻结初始输出统计`first`可作为后续对照，用于检验是否因尺度变化改变结论；当前没有切换。输入池X统计和输出Y统计需分开讨论，允许访问未标注输入不代表允许访问其输出。

### 15.3 存储约束继续有效

完整轨迹仅保存在冻结源池分块中。每轮`dataset.json`、追加的`added_samples.json`保存源池位置/哈希、有序行ID和必要权重；不新增累计NPZ或复制样本轨迹。归一化统计、checkpoint、指标和曲线属于实验状态，可以逐轮保存。

## 16. 2026-11-14 五轮等量追加与冻结统计的边界

按用户最新要求，`BenchmarkConfig`默认改为五轮；仍由`run_sampling_budding.sh`的离线分支读取统一配置，不在各算法内部另写日程。

| 轮次 | 累计训练样本 | epoch | warmup | 训练后追加 | 主模型更新次数（batch128） |
|---|---:|---:|---:|---:|---:|
| round_00 | 2000 | 100 | 5 | 2000 | 1600 |
| round_01 | 4000 | 100 | 5 | 2000 | 3200 |
| round_02 | 6000 | 100 | 5 | 2000 | 4700 |
| round_03 | 8000 | 100 | 5 | 2000 | 6300 |
| round_04 | 10000 | 200 | 10 | 0 | 15800 |

主模型合计31600次更新，比原四轮26100次多约21.1%；不含ensemble、pilot与选样。需要ensemble的增量方法在前四轮提供辅助成员，最终轮仍只训练主模型。一次性选择方法仍只训练最终10k，没有虚构四次追加。历史显式四轮配置仍可读取，新默认不会覆盖旧协议/结果，协议指纹随配置变化。

归一化暂时保留已确认的逐轮重算。AL4PDE的`norm_mode=every`确实在每次`init_training`时用当前train_loader更新统计，默认则是`first`。二者都是可选设计，不等于作者已经证明其中一个在本任务更优。

冻结统计不是固定数值上下界。对正值变量，变换`z=(log1p(y)-mu0)/sigma0`仍能编码初始范围以外的值；参数的Z-score同理，不要求落在[-1,1]。39个变量和141个参数维度没有增加，变化的是取值分布。主要风险是初始集合某一维恒定/方差很小，后续新值经小标准差被放大，以及极值在标准化MSE中占据较大权重；逐轮重算则会改变模型坐标和不同变量的有效loss权重。

如后续考虑`first`，应只用已查询训练数据检查每轮相对初始统计的标准化分位数、极值、初始小方差维度及各变量loss贡献；不能据此读取未查询轨迹来拟合归一化。输入P可见性和输出Y可见性不同。冻结尺度本身不丢失大值；是否影响本课题预测必须用固定合成验证集对照，不以real/test调参。尚未据这些讨论切换训练配置。

完整轨迹继续只保存在源池；五轮训练和四次追加使用JSON引用清单。共享split和初始2k由相同seed确定，修改后续预算不会改变这些ID或重做ODE。

已准备新共享协议`/home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_ea54a862dd85`，与原`seed42_c0d0d9a0169c`核对：源池定义相同，mutation/LHS全部candidate/validation/test/initial/random_order数组逐项相同；每池仍为90000/5000/5000。只创建协议、输入缓存和划分，无新ODE或正式训练。

针对性集成检查3项通过：历史四轮Random、新五轮Random、新五轮ensemble TopK；验证每轮嵌套训练ID、等额追加、最终唯一预算、重复调用恢复、无在线ODE、JSON存储不复制轨迹，以及前四轮辅助成员/最终轮不训辅助成员。日志在`scratch/probes/redundancy_cleanup_20261114/five_round_validation.log`；诊断不构成正式收敛证据。

### 16.1 结果目录取消A/B层级

按用户要求，批量启动器和单实验runner统一使用`experiments/budding_sampling/<protocol_id>/<pool>/<method>/seed_<seed>/`，不再把A/B放进目录。当前五轮例如`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/seed42_ea54a862dd85/mutation/random/seed_42/`。访问条件仍写入方法配置与标签访问账本，不改变选择器可访问真值的范围，也不改变共享数据划分、源池和训练预算。当前没有这套协议的正式结果目录，因此无需搬迁已有正式实验。

4项针对性回归通过：五轮Random训练/恢复、读取候选真值的方法执行、oracle权限检查、批量与单实验路径一致性（覆盖两种访问条件）。日志：`scratch/probes/redundancy_cleanup_20261114/flat_result_paths.log`。没有启动正式实验。

## 17. 2026-11-14 正式实验前状态与复现口径复核

当前注册25项，21个可运行配置、4个显式不可运行占位。21项中17项位于baseline目录、4项为ours；17项中`true_pattern_random`是项目诊断对照，其余也包含同一算法的表示/抽样变体，不能写成17种独立官方算法。默认批量18配置，在两个池上共35实验（ours_sensitivity仅支持mutation池）；没有自动启动。

当前两源池、五轮共享协议和real参考已具备。当前canonical定义对应`references/real_reference_b227aee93901.npz`，本次验证文件存在且与其JSON哈希一致，无需重新仿真参考。环境中实际加载的BMDAL `selection.py/features.py/feature_maps.py`逐字节与`baselines/sampling/bmdal_reg`对应作者文件相同。VeSSAL/Lev适配器另对作者commit和工作区文件强制校验。

复现边界仍按13.2的方法表：TopK/Power保留AL4PDE分歧与抽样公式，但新训练任务/日程为本项目统一协议；固定池Random对应pool_random，不是开放生成Random。LCMD/BAIT/CoreSet使用作者选择内核，输入特征则为本项目FNO的LL或Pred与512维投影；不能因此声称完整复现原PDE或表格实验。VeSSAL使用作者penultimate分支，公开流顺序与不足预算时的随机补齐；Lev使用作者内核但改为参数+初值输入特征和精确分块预算。Physics-PRE是含事件的ODE任务适配，HGGS-Residual只是其残差排序适配，Axiotis是论文重实现而非作者代码移植。

显式未完成：LowRank的非物理基向量loss定义、PEAKS分类到轨迹回归适配、完整HGGS的GF/MGS/生成流程、OGAS依赖子模块与混合域生成/概率纠偏定义。原方案中的VeSSAL-Reg、Lev-LL也未注册；CREST、ERACS、HGGS-IS/WRS动态训练不是当前已实现方法。无需为了第一批固定池主实验先补齐全部开放生成/动态训练方法。

仍缺正式实验层面的证据：

1. 新2k至10k协议的共同FNO训练预算校准；目前100/100/100/100/200未证明收敛，不以旧70k结果替代。可先运行Random作为第一项正式校准，只用固定合成验证集调整，再冻结共同配置。
2. 90k候选规模下各选择器的耗时/内存/GPU可用性；已有诊断是小数据规模，新五轮重点覆盖Random及ensemble/统一追加分支，尚未逐方法完成正式规模五轮。
3. 论文比较需要逐项公开来源和任务差异、标签访问成本及唯一训练输入；原生有放回clustering_sensitivity不保证10k不同输入，应单列。不能把作者内核一致、固定特征测试一致和完整论文实验复现混为一谈。
4. 当前工程有优化步数/训练与选样秒数账本，但未实现完整CPU/GPU峰值资源采样；需要峰值效率结论时补测。seed42用于首批，最终稳健性结论仍需其他统一seed复核，不作为首批启动硬阻塞。

主实验可优先选择Random、TopK、Power、MaxDist-Pred、CoreSet-LL、LCMD-LL、BAIT-LL、VeSSAL-Penultimate及ours；扩展对照按各自信息与预算条件明确报告。没有因本次盘点改变算法、归一化、采样分布或启动正式训练。

## 18. 独立基线实现与服务器负载保护（2026-11-14）

用户已确认首批FNO实验：Random、Uncertainty、LCMD-LL、BAIT-LL × mutation/LHS，共8项，仍使用当前五轮、128 batch、5e-4峰值LR和逐轮重拟合。**正式实验尚未启动**。

按用户最新约束，正式路径移除作者包/仓库运行依赖：`baseline/geometry.py`独立实现Gaussian投影和MaxDist/LCMD，`baseline/bait.py`独立实现带posterior的BAIT信息矩阵更新（含可选前向/后向），`baseline/authors.py`现为项目内VeSSAL/Lev实现，不再动态读取/执行参考源码。历史命名`_bmdal_*`仅保留接口兼容；第13/17节“运行时作者内核”属于更改前记录。参考仓库不修改，仅用于开发对照。8个随机种子共56组对照通过（距离选择、BAIT含前后向、VeSSAL、Lev及Gaussian投影）；无作者import的运行检查通过。尚不能据此宣称所有正式规模训练均已验收。

已完成的形状资源探针：128×39×500 FP32 FNO训练，PyTorch峰值allocated 0.926GiB、reserved 1.039GiB；90k×512特征、8k训练锚点、选择2k的BAIT探针allocated 1.427GiB、reserved 1.871GiB。这些不包含正式数据加载RAM峰值，不是完整实验显存上界。LCMD/ensemble/并发探针及联合回归没有最终完成报告，不能声称通过；后续因服务器负载问题停止推进重资源验证。

14:53左右排查时未发现本轮探针/正式实验仍在运行；MemAvailable约682GiB、load1约162、64可用CPU，memory PSI当时为0。此快照不足以确定此前SSH登录失败的原因，也不能归因给某个进程。此前仅按显存估计并发是不充分的，需要同时限制CPU、主存、I/O和解压。

本次实现的保护：

| 项目 | 默认行为 |
|---|---|
| 全局并行实验 | `SAMPLING_MAX_TOTAL_JOBS=2`；批量队列与单实验跨进程锁共同限制，跨启动器也共享 |
| 重I/O/解压/预处理 | `SAMPLING_IO_CONCURRENCY=1`；共享留出集、oracle读取与DataModule预处理共用跨进程锁 |
| 内存余量 | 有效可用内存低于min(128GiB,有效总内存20%)时等待；考虑user/session cgroup上限，可用`SAMPLING_MIN_AVAILABLE_GIB`调整 |
| CPU负载 | load1>1.5×可用CPU数时等待；64核对应96，不将高负载时的空闲GPU视为可直接启动依据 |
| 系统压力 | memory PSI some avg10>1%或IO PSI full avg10>5%时等待 |
| 调度优先级 | 批量子进程nice=10、ionice class2 level7（命令存在时）；BLAS默认2线程、DataLoader workers=0 |
| 减少重复解压 | 同一源池验证/测试ID合并读取一次，再以无复制视图保留不同角色；训练/标签权限与数值不变 |

等待发生在启动和下一次重加载之前，不会自动暂停一个已经运行中的GPU训练步骤。不能阻止其他用户同时消耗资源，也不能代替cgroup硬限制。若需要硬保障，应由服务器资源配置在独立实验cgroup中限制整批MemoryHigh/MemoryMax、CPUQuota和IOWeight，并将SSH/IDE会话留在该cgroup之外；本次没有设置硬限额，不清理全机cache、不终止其他用户任务。

长期可将压缩轨迹改为可映射/按块读取的存储，减少反复解压与重复RAM张量，但必须逐值验证和保持有序ID/权限/哈希语义；本次不重写源池，也不额外复制整份轨迹。正式8项仍需完成受控资源验证后启动。

负载保护8项测试通过（含跨进程重加载互斥、8任务/4卡/每卡2项仍受全局2项限制、内存/CPU/PSI拒绝及异常释放），五轮训练/恢复、权限和批量目录3项集成复测通过；均为隐藏CUDA、CPU单线程的诊断，不触发正式数据加载。日志在`independent_baselines_20261114/{resource_guards_final.log,resource_integration.log}`。复查load1约152、64可用CPU时准入返回等待，未启动正式实验。

## 19. 单项完整规模资源实测与首批配置（2026-11-14，当前）

用户同意先实测一项以决定并发，并授权6号卡做单进程资源测试；另明确4–7号卡每卡只能有一个GPU计算进程。资源测试已成功退出并释放6号卡，不能将它当作正式精度结果。

测试入口`python scripts/analysis/profile_sampling_resources.py --help`；报告位于`/home/users/zsy/mydata/budding_yeast/resource_profiles/single_n10000_20261114/`。复用源池和输入缓存，训练ID为来源池随机顺序前10000；正式FNO192/4/24、batch128、500时间点、双池完整验证/测试、126张真实突变体预测图，训练缩为3epoch。另在同一进程测90k候选特征、LCMD/BAIT（8k锚点、选2k）、双模型预测。模型精度、长程收敛及五轮内存漂移不由此证明。

| 阶段 | 进程树PSS峰值GiB |
|---|---:|
| 加载两池留出集 | 2.36 |
| 加载10000条训练轨迹 | 3.19 |
| 数据预处理 | 8.63 |
| 主模型训练 | 9.36 |
| 评测及CPU并发绘图 | **13.89** |
| 90k特征提取 | 10.39 |
| LCMD / BAIT | 10.37 / 10.38 |

0.5秒采样的进程树RSS峰值13.90GiB、PSS峰值13.89GiB，含W&B/绘图子进程；PyTorch GPU allocated峰值1.48GiB、reserved峰值2.26GiB，后者不等于NVML驱动总占用。总测试1181.7秒，首次加载留出集296.7秒。保护阈值为进程树PSS32GiB、主机有效可用内存128GiB及30分钟超时，本次未触发。测试期间仅测试主进程占用6号卡，绘图进程不初始化CUDA。

附加受控压缩块解压对照：同32块609.375MiB原始轨迹，1读者4.795秒/127.09MiB每秒；2读者2.379秒/256.12MiB每秒，总峰值RSS约0.0865GiB，校验结果相同。此为OS缓存已热的读/解压测量，不能宣称冷机械盘吞吐翻倍；未清理全机cache。结果在上述目录`io_one/summary.json`和`io_two/summary.json`。

据此将临时限制调整为**全局4项实验、最多2项读取/解压/预处理**，先按每项24GiB预算估计4项约96GiB；不是直接允许32/48等极端并发。保留128GiB余量、CPU和PSI门限、低优先级，后续按正式运行持续观察。4–7号卡每卡单GPU进程独立于这些限制，批量槽位和跨启动器锁/已占用PID检查均遵守；当前正式8项仍使用0–3号卡。

资源测试还揭示验证口径问题：mutation训练有90维参数标准差<1e-5，其归一化下mutation验证输入|Z|的99%分位3.32、最大15.64；LHS验证输入99%分位约7.4e6、最大约2.5e7。不能用跨域验证异常主导mutation模型的checkpoint。用户已确认改为**各自来源池验证**，交叉来源仅作为测试报告。`BenchmarkConfig.validation_scope='source'`明确进入指纹；旧存储协议缺失此字段时仍按历史`both`解释。

新正式共享协议`/home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_18bf9a3289fa`；四个split/input文件与原`seed42_ea54a862dd85`硬链接同一inode，全部哈希/有序ID不变，没有重生成数据或复制轨迹。当前默认首批`random/topk/lcmd_ll/bait_ll`×两个来源共8项，其他方法显式指定。池内/双池验证及五轮流程3项复测通过；独立选择内核11项回归、独占GPU等资源规则10项检查通过，日志位于`scratch/probes/independent_baselines_20261114/`。全规模资源报告是诊断文件，不进入正式排行榜。

后台会话`budding_sampling_8_seed42_20261114`包含`experiments`和`resources`两个窗口；正式结果根目录`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/seed42_18bf9a3289fa/`。首批仅0–3号卡，每卡1项、4并发，两波完成8项。`batch/launcher.log`记录准入/任务日志，`launch_plan.json`记录审阅过的配置，`resource_samples.jsonl`每10秒记录进程树PSS/RSS、每个任务PID、主机可用内存与load。监控无CUDA，可在整批PSS>160GiB或主机可用内存<64GiB时对经过UID/PID起始时间/进程组校验的本批发SIGTERM；不能称为cgroup硬限额。独立检查确认仅停止目标测试进程组，不影响另一进程。

首个队列在数据加载阶段重启一次以纠正父子nice优先级叠加，未开始epoch，源数据/实验配置不变；当前父进程nice0、训练子进程nice10，磁盘优先级保持较低。当前队列受CPU负载门限约束，不能把进程已建立或WAIT当成正式训练已经开始。并发限制和独立选择内核13项联合复测通过，另3项验证范围/五轮集成、1项监控退出范围检查通过。
