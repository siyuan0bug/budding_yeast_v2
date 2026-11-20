> **已实施：MA精确关闭（CDC14不变）**：CLB5/SIC1/PDS1-db、CDC20/CDC15/APC_Core-ts和APC-A对应通道改为确定性0，随机/逆CDF/赋值/容量/救援同步；MASS及其他分布保持。LHS边际已重编译，旧池版本失效，未生成新大型池。[修改前后与验证](BUDDING_EXACT_ZERO_UPDATE_2026-11-12.md)。

> **当前规则更新（2026-11-12，禁用三模块后）**：PPX/SBF_MBF/Mcm1停止合成采样；19个活动遗传模块、64个单独模式、4个有real依据的同基因双模式，另保留CLB2专门GAL-MC。未覆盖模型参数由85增至90，均为±50%；混合边际已按新规则重算。下方22模块/71模式/10复合/85参数及旧概率为调整前记录。[当前配置、real复合明细与零值诊断](BUDDING_REAL_SUPPORTED_MODULES_AND_COMPOUNDS_2026-11-12.md)。

> **当前选择更新**：混合边际LHS、mutation边际权重和85参数的0.5～1.5WT正值窗口已确认；下文“区间/权重待选”为初始讨论记录。当前配置和新增零点审计以[已选策略](BUDDING_MIXED_LHS_ZERO_POLICY_2026-11-12.md)为准，新增零质量与联合采样尚未完成。

> **参数域细化更新**：未被mutation改变的85个普通模型参数也按用户要求纳入LHS讨论；已分开多模式支持/离散零点、计算随机提案概率，列出两套待讨论倍率范围及2个当前未被RHS/solver读取的参数。[详细概率与范围](BUDDING_MUTATION_PROBABILITIES_AND_LHS_DOMAINS_2026-11-12.md)。

# Budding两类完整候选池与池内选样方案（2026-11-12）

状态：需求与参数范围审计已完成；生产生成器和AL入口尚未切换，未生成新10万条轨迹。用户已确定：LHS与mutation两类完整池，各10万；mutation模块/模式/强度全部随机、取消强制均衡配额、保留强度分布；四轮累计训练2000→4000→7000→10000。LHS上下界先由现有mutation规则整理，再审阅确定。

## 两个独立维度

| 数据生成范式 | 池内样本 | 第二层选择器 | 最终合成train |
|---|---|---|---:|
| LHS参数设计 | 参数、实际初值、完整ODE轨迹、真实BIO/ABC/六分类；无mutation标签 | Random / Uncertainty / Importance / BAIT / LCMD等 | 10000 |
| 随机mutation设计 | 上述全部内容，加实际gene/mode/强度/组合标签 | 同一组选择器 | 10000 |

每一种选择器分别在两个池上运行；LHS与mutation不再和Uncertainty/BAIT/LCMD写在同一层策略枚举里。每池仅生成一次，所有方法复用同一源池、初始集与固定holdout；real mutant单独评测，不进入10万合成池或train。

## 完整池意味着什么

先完成全池数值仿真和标签计算。第二层选中ID后直接读取已存轨迹，不再调用ODE、不再变更候选参数或临时救援。每池10万是最终有效、去重后合成轨迹数量；失败尝试和拒绝数量另存日志。训练阶段离线取数，但选样仍可逐轮依赖当前代理模型。

Pattern参与选样时读取仿真得到的真标签，旧输入→Pattern辅助预测器不作为新池标签来源。Uncertainty仍按独立ensemble预测分歧评分；BAIT/LCMD仍可按模型特征选样，这与拿预测Pattern冒充真实标签不同。每种算法必须声明可访问参数、mutation、真实Pattern或完整目标轨迹哪些字段。

若Importance通过全池真实轨迹计算误差，或某个方法使用全池真实轨迹描述，应明确标为相应的label-aware数据选择设置，不能将其隐去并称作只有参数的无标签AL。完整仿真成本是每池10万条，最终训练预算1万不等于只用了1万次ODE。Importance尚无确认的具体评分定义，不能直接恢复归档is近似或仅注册一个名称冒充官方实现。

## 已确认的随机mutation要求

- 合法模块范围仍为L1–L3，当前catalog合法单模式及复合模式均保留；KO、ts、MC、GAL等参数修改、初值和事件规则来自当前canonical系统。
- 删除Sobol点生成与逆CDF坐标流作为生产抽样路径，直接调用`BuddingSystem.sample_scaling_factor(...,rng=...)`及`apply_gene_mutation(...,rng=...)`。普通均匀随机是随机数源，不意味着把log-uniform、1/N整数MC、Beta、裁剪Normal等全部改成数值均匀分布。
- 模块集合和完整合法模式组合随机选择，不再先给每个集合/模式分配固定数量；禁止沿用旧配额规划器再只替换Sobol。
- 概率细化建议：从1772个合法模块集合等概率随机选择，再从该集合合法完整等位组合中随机选择。由此L1/L2/L3提案概率约1.24%/12.98%/85.78%，只是建议的随机概率律，不是用户另行确认的固定层级配额；最终保存频率还受有限容量、去重和数值成功条件影响。也不能将“全部随机”自动等同L1/L2/L3各1/3。
- 保留已批准强度分布：分基因MC整数上限与1/N权重、GAL对数均匀、GAL-MC等；有GAL时mdt=150，无GAL时90/150/160按80%/15%/5%。旧`samplers/random.py`仍有75%/18%/7%，不能不加核对直接复用。
- 碰撞检查/保持模块的mutation形式救援/去重在生成期完成，救援不能修改mdt来逃避碰撞。输出记录救援前后模式、来源与重抽次数。数值失败/非有限/不完整/超时不入池；保留有限高幅值及有限负值，不增加Pattern或viability筛选。

## LHS参数范围审计

本次直接调用现有随机强度与mutation函数到达支持边界，并遍历CDC14-ts等内部随机分支；没有用Sobol、没有调用ODE。记录141个模型输入坐标、71条单gene/mode规则。55个坐标在mutation规则下可变化，其中50个为模型参数、2个为事件活性、3个为初值覆盖；mdt三种环境单独列出。其余在当前mutation规则下未变化的坐标不能凭空指定范围。

边界并集含WT，不等于某一种mutation的强度分布，更不等于已经批准的新LHS独立参数域。结构性零、受约束总量/活性、事件开关和初值需要单独处理。例如TEM1/CDC15总量与实际初始活性不能任意独立取值；无法简单将141维全部做连续LHS。

| 参数 | WT | mutation规则边际包络（含WT） |
|---|---:|---:|
| ksb2p | 0.001 | 0～1.2 |
| ksb2pp | 0.04 | 0～0.32 |
| kdb2pp | 0.4 | 0.017～0.4 |
| ksn2pp | 0.15 | 0～1.5 |
| ks20p | 0.006 | 0～12 |
| ks14 | 0.2 | 0～2 |
| kasrentp | 1 | 0.025～1 |
| init_CDH1T | 1 | 0～1 |

例如kdb2pp边际包络上限0.4来自WT；CLB2-db本身仍是0.017～0.068，没有被改成0.017～0.4。所有CSV的`lhs_lower/lhs_upper`当前留空，避免把审计包络误当正式LHS上下界。

对71条规则各随机检查512次，共36352次参数修改，所有数值都在审计边界内；这只是生产规则边界核对，不是LHS联合生物可行性或求解成功率证明。

- 完整141维清单：`analysis/offline_pool_plan_20261112/parameter_domain_inventory.csv`。
- 55个变化坐标：`analysis/offline_pool_plan_20261112/variable_parameter_inventory.csv`。
- 71条mutation规则对应强度与实际修改参数包络：`analysis/offline_pool_plan_20261112/mutation_support_inventory.csv`。
- 环境/非向量开关、fingerprint与校验：`inventory.json`、`inventory_validation.json`。上述analysis根目录为`/home/users/zsy/mydata/budding_yeast/analysis/`。

## 每条记录的统一契约

两类池均需pool ID＋稳定candidate ID、141维参数/名称、39维实际初值/名称、time_grid、完整轨迹、数值状态、BIO/ABC/六分类和分裂次数、生成范式/seed/范围配置/求解配置。mutation池另有最终gene/mode列表、实际强度/拷贝数、L/module_count、overrides和救援记录。LHS用`mutation_available=false`、空pairs、无模块阶数的明确标记，不能伪造KO/GAL或将“未知mutation”当WT。

保留参数/初值float64、轨迹float32；模型输入仍转float32。范围内不同值和实际初值不能从float32轨迹首点反推。标签来自actual simulation，不记录成predicted label。

## 初始集、holdout与预算

已确认四轮synthetic train为2000/4000/7000/10000，追加量2000/3000/3000；不能沿用当前initial×1/2/3得到14000。最终1万仅计训练样本。

建议沿用初始80/10/10口径：每池先用输入Z-score后的KMeans选择2500条初始代表，再按seed随机拆为2000train/250val/250synthetic test，不按Pattern分层。500条holdout从后续选择域永久剔除。第一轮后剩余可选97500条，最后剩余89500条；每池总量仍严格10万。该holdout/初始化细化是方案建议，尚未生成或写入正式split。

同一池跨方法共用初始与holdout；不同池仅共用切分协议及seed，不能声称它们拥有相同物理样本。模型每轮仅用累计已选synthetic train拟合参数Z-score与轨迹log1p+Z-score，禁止因为全池已有轨迹而在全池拟合训练归一化。KMeans输入距离的Z-score与模型train-only尺度用途/拟合范围应各自记录。

当前生产仍为210min/500点；11月11日的400点审计没有修改默认值。新池正式输出长度应在生成前显式写入配置，并保证两池和real评测同网格；若改400点，须同步原有500形状检查与按10步去重的标签定义，不能只改导出点数。训练epoch/batch需按最终1万数据重新检查，不把旧7万实验的收敛或用时直接迁移过来。

## 实现迁移位置与验收

1. `data/systems/budding/`新增清晰的随机mutation完整池与独立参数LHS入口；公共solver/初值/标签复用当前canonical源，不导入data根目录或旧LHS文件来继承整套模型。
2. `utils/active_learning/`增加完整池读取后端，`query(selected_ids)`只取已存数据；不再走`oracle.py`在线ODE，也不能用当前要求mutation字段的参数池验证器加载LHS。Pattern方法切换到真实分组字段。
3. 选择器使用显式累计预算列表，输入/标签可访问范围明确；`importance`先给出可审阅公式再注册。baseline遵守原算法的特征/ensemble定义。
4. `run_sampling_budding.sh`继续作为统一批实验入口，第一层循环pool_kind=lhs/mutation，第二层循环选择器。结果放在机械盘，路径显式带pool_kind，记录源池SHA、选中ID与split，逐轮引用共享源数据，不重复复制轨迹。
5. 数据加载不能靠名字中的`_LHS_`推断real或旧母体家族划分；以显式is_real、sampling_paradigm、split为准。两种池都要支持相同无mutation编码的基础接口，不能强迫LHS进入L1–L3字段验证。
6. 验收包括两个小完整池、候选/轨迹/标签/overrides对齐、无Sobol调用、随机分布与当前函数一致、稳定ID/去重/real碰撞、失败补足、holdout隔离、离线query中solver调用次数为0、最终恰好1万train、同池跨方法共用初始split、W&B offline。

当前已交付的是这份方案及范围清单。LHS具体上下界、Importance评分及初始化细化尚需明确，不能宣称旧生产入口已经迁移，也没有启动20万轨迹生成。
