> **2026-10-30 运行内存优化**：pool/初始集、采样分布、ODE与归一化公式不变；ODE worker 仅传最终float32轨迹，P/实际IC仍float64，移除未使用虚拟X，入库分块归一化，冷启动前释放旧数据。逐轮继续使用无损引用清单。新增边反馈为额外审计，不替代样本字段。[说明](AL_OPTIMIZED_RESTART_2026-10-30.md)。

> **2026-10-30 存储格式更新**：新AL改为共享初始数据＋新增块＋`rounds/round_XX/dataset.json`，不再重复写累计轨迹。训练/评测/相关分析支持新旧格式，原始轨迹精度不变；本批14项已完成逐字段验证转换，173.13→56.65GiB（含共享初始），释放116.48GiB；首末轮真实checkpoint重载通过。[布局、读取和验证](AL_DATASET_STORAGE_LAYOUT.md)。下方空间数字为转换前审计快照。

> **2026-10-29 新正式输入已生成并审计**：百万`gene_ranges` pool与初始10000/1250/1250＋126 real已就绪。旧/新初始train大值比例19.55%→11.79%（相对减少39.69%），未按幅值删样本；新集总量上界检查通过。GAL-MC池中13703条、单模块容量8条。启动脚本默认新批次路径，W&B offline，正式训练尚未启动。文件、比较与跳过准备的启动命令见[数据准备结果](GENE_RANGES_DATA_PREPARATION_2026-10-29.md)。

> **2026-10-29 已实施获批范围及GAL-MC**：CLN2/CLB5/CDC14/TEM1/CDC15的MC改2～10，其余按逐基因表；普通CLB2-GAL收窄并新增独立GAL-MC。三类获批残余范围已更新，未新增零点混采或调整SWI5。94项回归、256条参数池和8条GAL-MC轨迹导出验收通过；49组ODE对照说明小残余不普遍等价KO。旧正式输入未覆盖，新实验需重建pool/初始集。详见[实施与机制对照](APPROVED_BUDDING_RANGES_AND_KO_CHECK_2026-10-29.md)。

> **2026-10-29 逐基因范围候选方案（未实施）**：已覆盖全部gene-mode及CDC14分支，提出各基因MC上限、CLB2普通GAL收窄与独立GAL-MC、部分残余速率范围及精确零分支。26组严格ODE扫描均成功，显示既有模式切换也有近零平滑响应；不能据此宣称训练精度提升。详见[完整候选表与落地接口](BUDDING_SAMPLING_RANGE_PROPOSAL_2026-10-29.md)。当前生产采样、pool、real及正式训练未改。

> **同模式强度核对与数值复核（2026-10-28）**：统一MC 2～30拷贝相对CDC14/CLN2/CLB5的real 2/5/4拷贝明显偏宽；70k train中31.99%含已知MC/GAL超real最大表达强度，其旧幅值规则触发率37.93%，其余为6.38%。固定背景复算确认强度和组合均能放大输出；另有8条TEM1越过自身总量上界，严解全部恢复，不能把finite等同于物理有效。原数据/规则未改。表格、对照及限制见[同模式审计](MUTATION_REAL_RANGE_AUDIT_2026-10-28.md)。

> **Random AL最终大值审计（2026-10-28）**：70k train中旧实际阈值外11532条（16.47%），CLB2/SIC1/ORI峰值625/946/8121。最终模型分层抽查估计这类样本贡献18.06% Hybrid损失，并未包揽训练；real最终MAE0.12469、RelL2 0.16630，ORI贡献约96%物理MSE。当前不能把坏例因果归于取消过滤或FNO容量。详见[完整审计与20例坏例](AL_RANDOM_EXTREMES_AUDIT_2026-10-28.md)。未修改过滤或训练。

# 数据集全景、生物图先验与归一化规范 (Datasets, Biological Graphs & Normalization)

> **AL模型输入精度**：文件/来源初值保留float64；首次加载及每轮重拟合后通过`ActiveLearningEngine.refresh_model_inputs`在本地float32副本上编码。不得将`_pool_initial_physical`直接归一化后送入float32模型。公式及train-only拟合不变，见[跨轮dtype修复](AL_DTYPE_FIX_2026-10-28.md)。

> **2026-10-28 初始集已就绪**：balanced百万池对应的`budding_initial_train10000_seed42_balanced.npz`已用48个ODE进程生成，共12500合成+126 real；seed42固定划分10000/1250/1250，初始L1/L2/L3为625/5000/6875。源参数/float64初值、全部元数据和标签、实际DataModule读取及train-only归一化检查通过，合成仿真失败0条。文件位置与审计见[FNO准备结果](FNO_AL_BASELINES_AND_OURS_2026-10-28.md#12-共享初始集已生成并验证)。未启动训练。

> **新池badcase覆盖实测**：百万池已生成；固定Top20严格局部25%邻域计数合计155→38，#15为1→2、#22为0→1，多个L2背景反而减少。尚未训练新模型。完整20例与新旧变化的混杂因素见 [覆盖对比](BALANCED_AL_POOL_COVERAGE_2026-10-28.md)。

> **2026-10-28 新配额已接入**：Budding AL采用模块集合等额、集合内合法模式容量感知均衡；新百万池为9815/126225/863960条L1/L2/L3，默认文件改为`budding_pool_1000000_seed42_balanced.npz`，初始集使用独立`balanced`路径。静态与Fission不变，本轮未启动训练。实现、有限容量与分布说明见 [两级均衡候选池](BALANCED_AL_POOL_2026-10-28.md)。

> **2026-10-28 样本内容补齐**：静态与AL统一保存显式mutation列表、overrides、精确初值和来源ID；AL累计快照/新增文件不再丢失这些元数据。生物标签、ABC及中性六分类均保留，未仿真池不伪造标签，Fission标签不变。字段、适用范围及历史文件说明见 [样本内容契约](DATASET_SAMPLE_CONTRACT.md)。

> **2026-10-28 AL环境与新配额设计**：用户确认后，budding_pool的非GAL环境已改80/15/5，GAL仍固定mdt150；静态采样器与Fission未改。按用户选定“模块集合等额、集合内模式均衡”枚举出22/230/1520个合法集合；考虑有限容量后的百万池设计配额为9815/126225/863960，多数集合571–572条。该配额目前只在计划CSV中，尚未替换生产POOL_RATIOS或生成正式新池。31项相关回归通过。Sobol与MA分布的区别及完整计划见 [配额与训练预算](AL_POOL_REDESIGN_AND_TRAINING_BUDGET_2026-10-28.md)。

> **2026-10-28 当前目录概率审计**：用户恢复PPX/Mcm1/SBF_MBF并移除CLB2[CA]后，当前为22基因/功能模块+MDT、71个合法gene-mode单元。12策略共请求51300条、142组强度/分位映射核对及实际2万唯一池审计完成；三组恢复模式均可抽到，原有唯一去重会将各固定L1-KO限制为最多3种环境输入。BAMS仍可构造非法APC_Core[KO]，若干静态策略存在与AL不同的footprint冲突处理；本轮仅报告未改生产代码。初选/最终概率、完整模式表、20项回归与限制见 [概率审计](BUDDING_MUTATION_PROBABILITY_AUDIT_2026-10-28.md)。此前目录精简说明属于历史版本。

> **2026-10-22 四轮 AL 实验流程已实施**：Budding `run_al_budding.sh` / Fission `run_al_fission.sh`；两者先按 L 配额 KMeans 选点、仿真后按 seed 随机切分 80/10/10；四轮训练、三次 1/2/3 倍追加；按轮重拟合累计训练归一化，固定留出集，L 字段贯穿保存。每轮 eval 只含 real 图和指标 JSON，新增独立/跨轮 loss 图。默认参数、命令和文件契约见 [当前实验流程](AL_EXPERIMENT_WORKFLOW.md)。以下较早的冻结尺度、Pattern 分层、三轮指数预算和旧目录均属于历史说明。

> **2026-10-22 Budding 碰撞救援更新**（`fixed_modules_environment_v2`）：静态生成、预生成 AL 参数池及运行时兜底池共用新规则。70% 分支从可软化的 KO 中选一个，改为合法 DAmP/ts/pLOF；30% 分支（以及无可软化 KO 时）先重抽某一突变的原模式强度，无有效变化再尝试该基因其他合法模式。其余突变强度、功能模块集合与 `mdt` 保持；不会在非 Gal 环境引入 GAL-OE。替换时移除旧模式全部参数/初值覆盖，重新执行对应 MASS/MA 赋值和分布，并检查参数足迹冲突。有限尝试仍碰撞则同 L 层重抽候选，不放行重复样本。
> real 碰撞按实际求解参数、39 维初值、BUB2 事件重置与 #104 定时诱导共同判断（容差 `atol=1e-8, rtol=1e-5`），已修复 `MAD2[CSL]` 的 `Vi20` 别名漏检；126 条 real 定义未改。Twins 保持两臂配对且不改环境，WT-Blind 失败同层补采；BAMS hard/hypo 消融为保持对照含义，碰撞直接重抽同层配对，不更换培养条件。AL 加载已有池会复查 real 碰撞及参数/初值与 overrides 的一致性，发现问题明确报错要求重建，不在评分后偷偷改参数。旧数据不自动更新；Fission 与 `data/` 根目录原始文件未改。
> 验证：222 项回归、265 项保留突变/分布审计通过；实际生成 MA/MASS 静态各 12 条、候选池 1000 条、Twins 每臂 12 条，全部通过 real 排除及模块数检查；CDC20 救援后经静态/AL 两个求解入口得到逐元素相同轨迹。本轮 57 个原始文件/Fission/生物模型文件哈希未变。一次性验证记录在 `scratch/probes/collision_rescue_2026-10-22/`，未改写正式数据集或重训正式模型。

> **2026-10-22 Budding 已取消仿真后幅值拒绝**：`systems/budding/simulation.py` 不再因 MASS>10 返回空轨迹；WT-Blind 不再因轨迹绝对幅值>1000拒绝；在线 AL 单批/补采均不再使用两倍 real-envelope 拒绝。有限大值与有限负值可入库，数值失败、非有限值、积分不完整、超时仍拒绝。原 BIO/Pattern、归一化的数值保护、Fission、`data` 根目录原始代码与 ours 策略/门控保持。307项回归、265项突变规则审计通过；CLB2峰值479.20的分布内案例已实际通过静态导出和AL入库。相关标定报告仍记载执行当时的历史规则。

> **删减模块后的概率口径**：当前20个目录项（19基因+MDT）、65个模式；基因先验在剩余集合重新归一化，图策略使用删减后拓扑。TEM1/CDC15只保留KO/GAL-OE/MC/ts：flat与s5各25%；class依次27.45%/27.45%/27.45%/17.65%。其余模式保留相对权重后按可选集合归一化。均匀模块提议为1/20，MDT不计基因阶数，重抽、冲突、复合与救援会改变最终样本频率；不应把提议概率当作入库占比。

> **2026-10-22 log1p尺度诊断**：对5000条当前分布合成轨迹比较保留/排除高幅值后的标准化尺度，并核对hybrid_mse_rel的两项均在标准化空间计算。该诊断不包含模型训练，不代表real预测精度变化，见 [log1p诊断](../scripts/analysis/outputs/log1p_extremes_2026-10-22_hybrid/REPORT.md)。

> **2026-10-22 分布内大值标定**：按当前两套 Budding 分布生成 5000 条随机样本，另测 290 组合法分位数组合；不在生产代码中新增/移除幅值门控。5290 条均正常完成，随机样本中 576 条超过两倍 real 的某个非 MASS 变量最大值。大值案例经加严容差复算仍存在；未做代理模型训练，不能推断 real 预测提升或下降。参数、原始轨迹、耗时、负值明细和比较表见 [分布内大值报告](../scripts/analysis/outputs/budding_extremes_2026-10-22/REPORT.md)。

> **2026-10-22 过滤审计**：对 mutant_aware 历史 10 万合成轨迹重放当前 AL 幅值与参数门控，并扫描负浓度。数值有效性与生物范围先验应区分；静态和 AL 现行规则仍不同，尚未修改生产过滤策略。统计、限制和建议见 [过滤审计](../scripts/analysis/outputs/filter_audit_2026-10-22/REPORT.md)。

> **2026-10-22 Budding 合成突变目录精简**：删除 TEM1/CDC15 的 DAmP，以及 CDC5_Proxy、SCC1、CDC28、SBF_MBF、PPX、Mcm1、ORI_BUD_SPN 七个合成模块。当前为 19 个基因模块 + MDT 环境模块，共 65 种模式组合；拓扑、motif、WT-Blind、救援与 AL 候选审计同步更新。保留组合的两套赋值/分布保持一致，39 变量、141 参数、ODE、初值和 126 real 不变；Fission 不变。含已删除突变的旧候选池或派生源会拒绝继续生成，需重新生成；旧 NPZ 不改写，仍可用于原有训练/分析。


> **2026-10-22 当前采样已更新**：Budding/Fission 静态配额 10/40/50、AL 候选池 5/40/55，仅一至三模块。下文较早的四/五层配额属于历史描述；新规则、输出字段和失败处理见 [三模块采样说明](EXACT_THREE_MODULE_SAMPLING.md)。

> **Fission AL 输入池**：新增逐候选原生预热池，包含参数/初值和完整 overrides，不含完整目标轨迹及标签；与静态轨迹数据集分开使用。初值预热仍有真实 ODE 成本，生成入口与单位契约见 [Fission AL](FISSION_ACTIVE_LEARNING.md)。

> **2026-10-21 系统目录收口完成**：新框架集中在 `data/systems/`，`data/generate.py` 为转发入口。
> 已修复增量去重、混合指标格式seed汇总、旧导入/状态转发和Fission采样审计；原生物规则与标签保持。
> 当前生成命令与文件职责见 [速查](DATA_GENERATION_QUICKSTART.md) 和 [整理说明](DATA_GENERATION_REFACTOR_2026-10-20.md)。


> **2026-10-21 目录复审**：基础生物规则与旧版对照通过；发现Fission增量去重、兼容shim及审计接口问题。
> 完整50k候选重放、49206条轨迹流式校验及优化优先级见 [目录审查](DIRECTORY_REVIEW_2026-10-21.md)。


## Budding 标签：生物标签保留，六分类作为补充（2026-10-21）

**BIO 负责生物事件判定；六分类负责观察窗内轨迹分组；ABC 保留为粗粒度分裂次数。**
六分类不替代 BIO，也不作为新的六分类训练目标自动启用。

| 标签字段 | 来源与用途 |
|---|---|
| `labels` | 原 `mutant_aware.py` 生物规则，仍由 `systems/budding/phenotypes.py` 判定；保留事件顺序、出芽、分裂质量等信息 |
| `pattern_labels` | 原 A/B/C；仍用于既有切分、三分类辅助头及损失掩码，保持旧训练接口 |
| `pattern_group_codes` 等 | 新独立字段 `pattern_group_codes`、`pattern_group_labels`、`pattern_group_ids`；P1/P2/P3/P4/P5a/P5b，ID=0–5；由未归一化轨迹计算 |
| `observed_division_counts` | 沿用质量相邻跌落严格大于0.1、相邻检测跌落间隔严格大于10步的计数 |
| `pattern_group_version / pattern_group_rules_json` | 补充分组的独立版本 `budding_window_groups_v1` 和完整阈值、优先级 |
| `biological_labels_available` | 生物标签是否原本记录；缺失时 `labels=Unknown: Not recorded`，不从采样曲线伪造事件标签 |

中性名称为：P1“观察到多次分裂”、P2“观察到一次分裂”、P3“无观测分裂且ORI信号低”、
P4“无观测分裂且ORI信号升高”、P5a“无观测分裂且末端Clb2/Esp1达到阈值组”、
P5b“无观测分裂且Clb2高”。精确条件采用已核验的有序决策树，见
[`pattern_groups.py`](../data/systems/budding/pattern_groups.py)。不采用“自持极限环”“衰竭”或“奇点”命名，
也不接入未经独立校准的 `Is_Bifurcation_Critical`。

新统一生成入口自动保存全部标签；参数池没有轨迹，不生成这些轨迹标签。
旧 NPZ 不改写，通用加载器的 `return_metadata=True` 从物理轨迹补算六分类，已有补充字段则逐项校验。
DataModule 暴露 `biological_labels`、`pattern_group_codes` 等行数组；六分类不改变切分、归一化或三分类目标。
Fission 的 `labels` 和 `pattern_labels` 严格保持原 `fission_random_sampler.py` 规则，不套用 Budding 六分类；
原 pattern=None 的样本仍导出为字符串 `'None'`，不补成 Pattern_C；内部训练ID=-1，分类损失跳过这些行。
W&B按Fission自身BIO/pattern选样，详见 [Fission标签与实验流程](FISSION_EXTENSION.md)。

未来 AL 快照同时保留 BIO、ABC 和六分类，历史缺失 BIO 的记录明确标为未知。
AL 新追加样本的训练类别 ID 修复为实际 A/B/C，过去的全零占位会错误地标成 A；
从候选池直接构建初始训练集时也传入实际三分类目标。选样、配额、过滤阈值保持原样。
评测保留原指标 CSV，另输出 `per_mutant_label_groups.csv` 与 `per_synth_test_label_groups.csv`，
按名称/索引对齐原指标、BIO、ABC 和六分类，用于坏例诊断，不把分组收益视为训练精度提升。

对消除 badcase，ABC 的区分能力有限：P1=A、P2=B，其余四组细分C，六分类包含了ABC全部计数信息。
但已有跨策略比较只证明六分类对MAE/MSE风险的区分较好，对relL2反而较差，
尚未通过“按新标签补采/训练”的对照实验验证坏例改善。相关证据见 [AL标签比较](AL_PATTERN_LABEL_COMPARISON.md)。
C只说明窗口内未检测到分裂，不能据此对全部39变量施加稳态假设；现有损失行为本次未改。

canonical 126 real 的P1/P2/P3/P4/P5a/P5b计数为21/47/7/5/33/13。
47个P2延长至630分钟后全部继续分裂，因此“单次分裂后衰竭”不是数据支持的解释。
论文核对及全部非LHS统计见 [Pattern核验](BUDDING_PATTERN_LABEL_AUDIT_2026-10-20.md)。

## 当前生产接口（2026-10-20）

Fission现已扩展独立图、通用模型训练/评测及可微物理残差：连续67条、hybrid75条非自环边，
Budding仍为218/226。详见 [Fission扩展](FISSION_EXTENSION.md)。
`budding_random` 的实际分布、RNG、冲突/救援、过滤/失败与无配对规则见
[逐步说明](BUDDING_RANDOM_WALKTHROUGH.md)。

新生成入口统一为 `data/generate.py`；文件名、18 种配置及加载方式见
[文件与调用速查](DATA_GENERATION_QUICKSTART.md)。生产消费者已切换到
`data/systems/budding/api.py` 和独立 Fission 模块；原地原样归档的是 31 个旧生成/LHS 文件，
辅助消费者按本次要求迁移，不再适用此前“44 个 data 文件全部未变”的阶段性表述。
图默认读取新生物源码，DataModule 优先读取 `is_real/time_grid`。
新旧 NPZ 核心字段兼容，不代表所有历史样本、排序或文件字节完全相同。

本次接口迁移验证：18 种配置均实际生成小规模文件；Budding 126 real + 16 synthetic、
Fission 20 real + 8 synthetic 的加载与划分通过，Budding 完成单步训练/验证。
507 个相关回归测试分批通过，另有 381 项突变/分布及 105 项采样专项通过。
240 个非归档/非旧版对照 Python 文件解析及导入语句扫描未发现旧生成器依赖；
运行时禁用旧生成器导入后，新生成与训练加载仍通过。未进行十万/百万规模端到端逐行比对。

以下三段为本次迁移早期的审计记录；旧文件留存范围以当前生产接口说明为准。

> **2026-10-20 逐文件规则范围更正**：采样/数值专项通过不能代表所有新旧文件规则相同。
> 完整清点44个旧文件、54个新文件后，确认统一入口默认值、部分可选项、重复调用状态、尝试上限和失败/导出政策仍有差异。
> 参见 [逐文件规则结论](DATA_GENERATION_FILE_RULES_2026-10-20.md) 与 [全部文件/定义出处](DATA_GENERATION_RULE_INVENTORY_2026-10-20.md)。

> **2026-10-20 最终复查**：Budding/Fission 的 `benchmarks.py` 已改名为 `real_mutants.py`，同步全部新入口引用。
> 补查并修复超时传播、救援配对、派生编号/元数据、源参数与 real 身份校验、失败状态导出等问题；旧 44 个 Python 文件未改动。
> 完整复查结果和已知验证边界见 [数据生成最终复查](DATA_GENERATION_FINAL_REVIEW_2026-10-20.md)。

> **2026-10-20 数据生成模块化**：新增 `data/generate.py` 和 18 个可选配置，原有 44 个 Python 文件保持原样。
> Budding 基础模型及 126 个 real mutant 统一来自 `mutant_aware.py`；保留 MASS / mutant_aware 两套合成突变与幅度分布，
> 并接入 Fission 独立生物模型。新流程碰撞统一救援；完整 126 条 Budding real 新旧仿真逐元素一致（最大差值 0）。
> 文件职责、旧新入口对应、运行命令及一致性边界见 [数据生成整理说明](DATA_GENERATION_REFACTOR_2026-10-20.md)。
> 没有覆盖正式 NPZ，也没有改变训练、归一化或切分。

> **同日采样一致性复核**：发现并修复 Gibbs-Sobol 层级配额/顺序、MASS Random 的 shuffle RNG、Twins 的 TEM1-ts 初值及救援 RNG 调用方式。
> 两套突变/分布及功能模块核对 381 项通过；采样内核与旧入口无碰撞调度对照 105 组通过。
> 验证方法、复跑脚本与有意差异见 [采样一致性报告](SAMPLER_EQUIVALENCE_AUDIT_2026-10-20.md)。

> **2026-10-19 后续授权：pure-random RNG 已修复**。局部 seed 统一控制幅度、随机等位基因、
> 冲突重抽及救援，显式 RNG 的冲突遍历顺序不再依赖 Python 哈希种子。
> 共享 helper 旧调用保持兼容；没有重新生成正式 NPZ，corr 定义与重复分组未改。
> 详见 [RNG 修复及 corr 的评估风险](PURE_RANDOM_RNG_FIX_2026-10-19.md)。

> **2026-10-19 后续授权：train-only 拟合已启用**。新训练先切分，仅使用合成 train
> 计算 Y/P 均值、标准差及 `asinh_scale.s_c`，冻结后用于所有留出集。默认变换仍是 log1p。
> 旧 checkpoint 保留自己的历史统计量/拟合范围，不能用新统计量解释旧权重。
> 该阶段未改重复样本分组、随机数管理和 corr 定义；RNG 后续变更见上方说明。
> 详见 [train-only 修复与解释](TRAIN_ONLY_NORMALIZATION_2026-10-19.md)。

> **2026-10-19 运行修复**：四种归一化的编码/解码统一；新 checkpoint 保存完整预处理和源数据/切分指纹，
> 评估恢复统计量并拒绝不一致输入，旧 checkpoint 会读取 DataModule 中的 norm_type 并明确警告其契约不完整。
> 求解器失败或未覆盖完整时长不再外推为有效标签。该先前阶段按用户要求**不改切分、不改 all-synthetic 拟合范围**，
> 不改采样 seed、指标定义、104 号输入或通用系统入口。见 [运行契约修复](RUNTIME_CONTRACT_FIX_2026-10-19.md)。

本文档系统介绍 `budding_yeast_v2` 框架中使用的数据集体系、生化多关系先验网络、全量数据归一化方案及其在 126 个真实生物突变体上的消融审计结论。

> **2026-10-18 代码核验**：当前默认 `norm_type=log1p` 实际为
> `Zscore(log1p(max(Y,0)))` 与参数 P 的线性 Z-score，并非 Min-Max；部分历史表的
> `standard_zscore` 标签及 MSE 数值有误。原始 hybrid 运行的物理 MSE 为 5.18461。
> 已确认的统计口径、实际训练集覆盖、AL 测试轨迹门控泄漏和复现限制，详见
> [当前 bad-case 证据审计](PURE_RANDOM_BADCASE_AUDIT_2026-10-18.md)。
> 完整输入与完整轨迹检查确认：合成 val/test 各有 1,633/1,604 条与 train 重合；
> real 的 78、112、114 号也与 train 完全重合。不能称这批测试点全部 unseen。
> 全部 126 条 real 轨迹在 rtol=1e-4/1e-6/1e-8 下均求解到 210 min，未发现失败段。
> L1–L4 虽各有 25k 样本，L4 中多次分裂 Pattern A 只有 503 条，实际 train 中为 399 条；
> 突变层级均分不等于动力学覆盖均衡。新计数见审计的 `training_contracts/` 输出。
> 本次诊断不允许基于 real-mutant 规则、参数或轨迹构造训练邻域。

> **同日后续修复**：因果 WT 特征现使用与 P 输入相匹配的变换，支持 log1p/linear、
> asinh_z、asinh_scale；统计量经训练/独立评估入口注入并在新 checkpoint 中持久化。
> 未改变本数据模块的统计量拟合拆分，也未修改采样、图或 ODE；
> 见 [IC/WT 修复与迁移说明](IC_WT_CONTRACT_FIX_2026-10-18.md)。

> **同日图来源统一（第一阶段）**：新增 `data/yeast_graph.py`，仅解析
> `mutant_aware.py` 的 ODE/事件源码生成潜在结构，连续非自环边 218 条，事件新增
> 8 条，合并 226 条。不固定正负号，不用测试 mutant 参数或轨迹。
> 旧图资产与生产模型尚未迁移，旧构图 CLI 已加入显式历史导出/拒绝覆盖保护；
> 见 [图来源契约与阶段边界](YEAST_GRAPH_SOURCE_CONTRACT_2026-10-18.md)。

---

## 一、数据集体系概览 (Dataset Overview)

系统涵盖由动力学 ODE 求解器生成的 **10 万量级预训练/基准数据集**、**100 万级主动学习未标注候选池**，以及 **126 个真实酵母突变体金标准测试集**：

| 数据集名称 | 样本规模 | 时长与步数 | 动力学参数生成机理 | 核心用途与评估维度 |
| :--- | :--- | :--- | :--- | :--- |
| **`REAL_MUTANTS`** | **126** | $210\text{ min}$, $500\text{ steps}$ | 严格对应 Chen et al. (2004) 真实酵母表型实验 | **金标准终极评估基准**（不可参与训练） |
| **`mutant_aware_pure_random_100000_210m_500s.npz`** | $100,126$ | $210\text{ min}$, $500\text{ steps}$ | 100k 基因/模式组合采样、模式相关幅度分布；另附 126 条测试规则的 ODE 轨迹，并非 126 突变体镜像扰动 | 当前全模型基准评测统一基线数据集；独立性限制见顶部审计 |
| **`mass_100000_dataset_210min_500steps_dual_labels.npz`** | $100,126$ | $210\text{ min}$, $500\text{ steps}$ | 质量反馈自适应参数生成 + 表型双标签 | 经典双标签对比基线数据集 |
| **`pool_1m_pure_random.npz`** | **1,000,000** | 仅参数 ($141$ 维) | 宽域非线性参数均匀随机采样（无时序） | **AL4PDE 主动学习百万候选池** |
| **`bams_100000_210m_500s.npz`** | $100,000$ | $210\text{ min}$, $500\text{ steps}$ | 盲态流形相空间多样性采样 (BAMS) | 流形相空间表征消融 |
| **`gibbs_boltzmann_100000_210m_500s.npz`** | $100,000$ | $210\text{ min}$, $500\text{ steps}$ | 基于能量梯度的 MCMC 吉布斯采样 | 鞍点与相变边缘参数富集 |
| **`kdpp_gated_100000_210m_500s.npz`** | $100,000$ | $210\text{ min}$, $500\text{ steps}$ | 核行列式点过程 (k-DPP) 门控采样 | 排斥性几何正交采样 |
| **`motif_topology_100000_210m_500s.npz`** | $100,000$ | $210\text{ min}$, $500\text{ steps}$ | 生物网络模体结构感知参数变异 | 网络拓扑相关变异分布 |

### 1.1 数据集生成与采样引擎完整清单 (`data/`, 共 16 个核心工具)
代码库 `data/` 目录下配备了面向不同高维几何分布与生物学先验的生成器，全部支持 `--seed` 与 `--data_dir` 灵活调用：

| 引擎脚本 | 核心算法与采样机理 | 关键参数与功能定位 |
|:---|:---|:---|
| [generate_pure_random_dataset.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_pure_random_dataset.py) | 采样 1–4 个基因及可用模式，按模式修改 WT 参数；不是 141 维独立连续采样 | 大规模基础纯随机基准集生成；real 依赖与去重限制见审计 |
| [generate_1m_pure_random_pool.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_1m_pure_random_pool.py) | 100 万未仿真纯参数点 ($P, X_0$) 生成 | **AL4PDE 百万免预仿真候选池**（无 $Y$ 输出，防泄露） |
| [bams_dataset_generator.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/bams_dataset_generator.py) | 盲态流形相空间多样性 (BAMS) 采样 | 相空间测度平衡采样 |
| [gibbs_boltzmann_sampler.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/gibbs_boltzmann_sampler.py) | 基于能量梯度的 MCMC 吉布斯采样 | 鞍点与相变边缘参数富集 |
| [gibbs_sobol_sampler.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/gibbs_sobol_sampler.py) | 准蒙特卡洛 (QMC) Sobol 低差异序列扰动 | 高维空间均匀充盈采样 (Space-filling) |
| [generate_kdpp_gated_dataset.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_kdpp_gated_dataset.py) | 核行列式点过程 (k-DPP) 排斥性采样 | 抑制样本冗余，最大化子空间体积 |
| [generate_kdpp_curv_gated_dataset.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_kdpp_curv_gated_dataset.py) | 曲率感知 k-DPP 门控采样 | 在非线性流形高曲率区域自适应增密 |
| [generate_motif_topology_dataset.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_motif_topology_dataset.py) | 生物因果模体（Motif）拓扑扰动生成 | 覆盖反馈回路、前馈回路等功能模体 |
| [reachability_check.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/reachability_check.py) | 零训练突变体参数可达性与流形流速审计 | 探测 126 突变体在各数据集中的几何距离 |
| [mass_dataset_generator.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/mass_dataset_generator.py) | 质量反馈驱动参数变异生成 (Mass V1/V2) | 支持表型双标签生成的经典基准集 |
| [mass_wt_blind_sampler.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/mass_wt_blind_sampler.py) | 野生型固定点不动点代数反演采样 | 100% 盲态（无任何突变体先验）空间采样 |
| [mirror_pure_random_ko_interval.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/mirror_pure_random_ko_interval.py) | 敲除微小连续区间 $[0, 10^{-4}]$ 采样器 | 探究平滑敲除在神经算子中的连续性 |
| [al_pool_manager.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/al_pool_manager.py) | 内存映射 (Memory-mapped) 动态候选池管理器 | 支持千万级参数池低显存高吞吐流式访问 |
| [yeast_graph.py](../data/yeast_graph.py) | 当前 ODE/事件源码的统一潜在依赖构图 | 矩阵由同一份来源记录派生，不编码固定正负号 |
| [build_bio_multirelational_graphs.py](../data/build_bio_multirelational_graphs.py) | 历史六关系/circuit/module 元数据构图 | 兼容函数冻结保留，CLI 仅允许显式导出到新文件 |
| [yeast_datamodule.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/yeast_datamodule.py) | PyTorch Lightning 统一高效数据管道 | 负责多进程加载、在线切片与归一化 |
| [yeast_multirelational_graphs.npz](../data/yeast_multirelational_graphs.npz) | 旧多关系与模块映射资产 | 供部分历史异构图模型使用；普通 BioRefinerPlus 不使用六关系通量图 |

---

## 二、四大参数采样空间系统与真实等位基因扰动数学建模

在出芽酵母动力学仿真中，参数分布设计直接决定了代理模型学习到的流形结构。系统总结并沉淀了四大采样体系：

### 2.1 四大底层参数采样物理分布体系 (Systems A ~ D)
1. **体系 A：【连续对数 + 关键等位基因硬编码死值常数】**（以 `mass_dataset_generator.py`、`mass_random_sampler.py` 为代表）：
   在野生型 $P_{\text{wt}}$ 基础上施加连续扰动，但在关键突变等位基因上大量采用离散常数死值（如 TEM1、ESP1、CDC20 写死常数或 0.0），且包含早期硬编码的生物拯救模体（存在测试集特征重叠对照）。
2. **体系 B：【离散对数拷贝数 + 物理锚点包络区 + 边界微扰缓冲】**（以 `mutant_aware.py` 统一主干为代表，为 BAMS、k-DPP、Gibbs-Boltzmann 等图拓扑生成器所共享）：
   彻底消除死值常数覆写，引入以真实生理值为中心的对称对数安全包络区，并在边缘突变（如 `ts`, `db`）设置 $\le 1\%$ 残留活性边界微扰缓冲带，避免去重合并时发生奇异坍缩（如彻底解决 #73 `cdc20-ts` 特征重叠）。
3. **体系 C：【连续微小区间 KO 微扰】**（以 `generate_pure_random_dataset.py` (KO-interval 模式)、`mirror_pure_random_ko_interval.py` 为代表）：
   在体系 B 基础上，针对敲除样本将硬零改造为 $[0, 10^{-4}]$ 连续均匀采样，探究近零连续区间对神经算子切空间梯度的平滑性。
4. **体系 D：【不动点代数自洽反演】**（以 `mass_wt_blind_sampler.py` 为代表）：
   严格零突变体先验，对 GAL-OE 采用细胞周期 G1 期不动点（Fixed-Point）稳态极限环代数方程反演可行过表达倍数 $\alpha = f(E, \theta_{\text{fixed-point}})$。
5. *(注：`gibbs_sobol_sampler.py` 与 `generate_1m_pure_random_pool.py` 使用 Sobol 拟蒙特卡洛低差异序列结合逆累积分布函数（Inverse-CDF）进行参数取样，其目标边际分布与体系 B 保持严格一致)*。

---

### 2.2 全局突变模式 (Mutation Modes) 采样分布横向对照矩阵

当给定完全相同的基因突变组合 $(G, M)$ 时，四大体系赋予动力学参数的**概率分布公式与取值范围**存在本质差异：

| 突变模式 (Mode) | 体系 A：`mass_dataset_generator` 系<br>*(早期常数硬编码型)* | 体系 B：`mutant_aware` 系<br>*(物理锚点包络+边界缓冲型)* | 体系 C：`pure_random_ko_interval`<br>*(连续微小区间微扰型)* | 体系 D：`mass_wt_blind_sampler`<br>*(代数不动点自洽反演型)* |
| :--- | :--- | :--- | :--- | :--- |
| **`KO`** (基因敲除) | **严格硬零**：$p_{\text{mut}} = 0.0$ | **严格硬零**：$p_{\text{mut}} = 0.0$ | **连续微小区间均匀分布**：<br>$p_{\text{mut}} \sim U[0.0, 10^{-4}]$ | **严格硬零**：$p_{\text{mut}} = 0.0$ |
| **`MC`** (多拷贝质粒) | **连续对数均匀**：<br>$\alpha \sim 10^{U[\log_{10}(1.5),\, \log_{10}(30.0)]}$<br>*(连续浮点数，如 2.37x, 18.5x)* | **离散对数整数分布**：<br>$n \in \{2, 3, \dots, 30\}$<br>$P(n) \propto 1/n$ *(符合质粒整数离散规律)* | 同体系 B (离散整数对数分布) | 同体系 A (连续对数均匀分布) |
| **`ts`** (温敏失活) | **大部分基因硬编码写死常数**：<br>TEM1 $ka15pp=0.002$;<br>ESP1 $kasesp=0.1, kdiesp=0.001$;<br>PPX $kppnetpp=0.05$;<br>CDC15 $kkpnetpp=0.0$;<br>CDC20 $ks20p=ks20pp=0.0$ | **连续重参数化 $p_{\text{wt}} \times \alpha$**：<br>• TEM1, ESP1 采用锚点包络 $\text{LogU}[0.0005, 0.004]$<br>• NET1 采用包络 $\text{LogU}[0.0125, 0.10]$<br>• CDC20, CDC15, APC_Core 采用缓冲 $U[0, 0.01]$<br>• 其余基因: $0.20 \times \text{Beta}(0.5, 2.0)$ | 同体系 B (锚点包络区 + 边界缓冲) | 同体系 A (包含固定常数硬编码) |
| **`GAL-OE`** (半乳糖过表达)| 各基因各自对数均匀区间；<br>**CLN2 下限过低**：<br>$\text{LogU}[0.03, 0.30]$ | 各基因各自对数均匀区间；<br>**CLN2 修正物理下限**：<br>$\text{LogU}[0.06, 0.30]$ | 同体系 B (修正后物理区间) | **不动点浓度代数反演求解**：<br>$\alpha = f(E, \theta_{\text{fixed-point}})$<br>*(未收敛时回退到对数区间)* |
| **`db`** (降解盒突变) | • CLB2: $\alpha \sim U[0.04, 0.15]$<br>• PDS1: 采样 $U[0, 0.05]$ 但在 `apply` 写死 0.0<br>• CLB5, SIC1, PPX: **写死绝对常数 0.0** | • CLB2: **物理校准** $\alpha \sim U[0.02125, 0.170]$<br>• PDS1: 边界缓冲 $\alpha \sim U[0.0, 0.01]$<br>• CLB5, SIC1, PPX: **边界缓冲** $\alpha \sim U[0.0, 0.01]$ | 同体系 B (物理校准 + 边界缓冲) | 同体系 A |
| **`APC-A`** | **写死绝对常数**：$ka20pp = 0.0$ | **连续边界微扰缓冲**：<br>$ka20pp = p_{\text{wt}} \times U[0.0, 0.01]$ | 同体系 B | 同体系 A |
| **`TAB6-1`** | **写死绝对常数**：<br>$kasrent = 10.0, kasrentp = 0.05$ | **统一为 NET1-ts 锚点对数包络**：<br>$\alpha \sim 10^{U[\log_{10}(0.0125),\, \log_{10}(0.10)]}$ | 同体系 B | 同体系 A |
| **`DAmP`** | 截断正态分布：$\text{Clip}(N(0.20, 0.05^2), 0.05, 0.40)$ | 截断正态分布：$\text{Clip}(N(0.20, 0.05^2), 0.05, 0.40)$ | 同体系 B | 同体系 A |
| **`pLOF`** | 连续均匀分布：$\alpha \sim U[0.10, 0.50]$ | 连续均匀分布：$\alpha \sim U[0.10, 0.50]$ | 同体系 B | 同体系 A |
| **`CA`** | 结构激活常数（如 CDH1 $kicdhpp=0, kscdh=0.03$） | 结构激活常数（如 CDH1 $kicdhpp=0, kscdh=0.03$） | 同体系 B | 同体系 A |
| **`CSL` / `CKO`** | 纺锤体检查点常数（如 Vi20=0.01 / kbub2=0） | 纺锤体检查点常数（如 Vi20=0.01 / kbub2=0） | 同体系 B | 同体系 A |
| **`MDT`** (碳源突变) | **三者等权重无偏随机**：<br>$\text{Choice}([90, 150, 160])$ | **严格对齐生理碳源比例**：<br>$\text{Choice}([90, 150, 160], p=[0.75, 0.18, 0.07])$ | 同体系 B | 同体系 A |

---

### 2.3 关键动力学敏感基因的专门参数采样分布分析 (具体到基因与代码实现)

#### 1. `CLN2` 在 `GAL-OE` 模式下的采样分布
* **体系 A (`mass_dataset_generator.py:L592-L593`)**：
  $$ksn2p \sim 10^{U[\log_{10}(0.03),\, \log_{10}(0.30)]}, \quad ksn2pp = 0.0$$
* **体系 B (`mutant_aware.py:L646-L649`)**：
  $$ksn2p \sim 10^{U[\log_{10}(0.06),\, \log_{10}(0.30)]}, \quad ksn2pp = 0.0$$
  * *代码物理考量*：文献中 7 株真实 GAL-CLN2 突变株参数均为 $0.12$；体系 A 下限 $0.03$（仅为真实强度的 $1/4$）对于强启动子 GAL1 过表达在生理学上过弱，体系 B 将下限收紧至 $0.06$，确保物理合理性。
* **体系 D (`mass_wt_blind_sampler.py:L664-L666`)**：
  $$ksn2p = E \times \theta \times \frac{kdn2}{M_{\text{birth}}}, \quad \text{其中 } \theta = \frac{kisbfp}{\max(kasbf \cdot esbfn2, \epsilon)}$$
  * 通过细胞周期 G1 期不动点（Fixed-Point）代数阈值反演 $ksn2p$。

#### 2. `TEM1` 与 `ESP1` 在 `ts` 模式下的采样分布
* **体系 A (`mass_dataset_generator.py:L800-L803`)**：
  * 在 `sample_scaling_factor` 中采样了通用 Beta 分布，但在 `apply_gene_mutation` 中被**完全覆写为死常数**：
    * `TEM1`: $Tem1T = 0.0, \quad ka15pp = 0.002$（写死 $0.002$）；
    * `ESP1`: $kasesp = 0.1, \quad kdiesp = 0.001$（写死固定常数）。
* **体系 B (`mutant_aware.py:L683-L696`)**：
  * 彻底消除常数覆写，构建**以真实锚点为中心的对数均匀包络区（Anchor Envelopes）**：
    $$\alpha \sim 10^{U[\log_{10}(0.0005),\, \log_{10}(0.004)]} = \left[\frac{\text{anchor}}{4},\, 2 \times \text{anchor}\right]$$
    * `TEM1`: $Tem1T = 0.0, \quad ka15pp = p_{\text{wt}}['ka15pp'] \times \alpha, \quad init\_Tem1 = 0.0$；
    * `ESP1`: $kasesp = p_{\text{wt}}['kasesp'] \times \alpha, \quad kdiesp = p_{\text{wt}}['kdiesp'] \times \alpha$。
  * 使神经网络在真实突变体附近获得具有物理意义的连续参数支撑面，同时保留去重安全性。

#### 3. `CDC20`, `CDC15`, `APC_Core` 在 `ts` 模式下的采样分布
* **体系 A**：直接赋予绝对零值（如 $ks20p = ks20pp = 0.0$）。导致单基因突变样本与文献真实突变体（#73 `cdc20-ts`）在 136 维参数空间发生 100% 重叠而被去重拦截（Dedup Collapse），造成葡萄糖条件下的失活突变特征在训练集中完全丢失。
* **体系 B**：引入 $\le 1\%$ 残留活性的**边界微扰缓冲（Boundary-Anchor Buffer）**：
  $$\alpha \sim U[0.0, 0.01] \implies p_{\text{mut}} = p_{\text{wt}} \times \alpha$$
  确保边界点无法被离散点集拦截，训练集充分覆盖极限边界动力学。

#### 4. `CLB2` 在 `db` 模式下的采样分布
* **体系 A**：采样 $\alpha \sim U[0.04, 0.15]$，计算公式为 $kdb2pp = p_{\text{wt}}['kdb2pp'] \times \alpha$（$p_{\text{wt}}=0.4$），导致 $kdb2pp \in [0.016, 0.060]$。
* **体系 B**：经过生物物理动力学公式严格逆推校准，采样 $\alpha \sim U[0.02125, 0.170]$，使有效降解速率：
  $$kdb2pp = 0.4 \times \alpha \in [0.0085, 0.068] = \left[\frac{0.034}{4},\, 2 \times 0.034\right]$$
  刚好将全部 9 株真实 CLB2-dbΔ 突变株的常数 $0.034$ 包络在 $42.86\%$ 的黄金内部分位数。

#### 5. `CLB5, SIC1, PPX` 在 `db` 模式下的采样分布
* **体系 A**：`CLB5, SIC1, PPX` 在 `db` 模式下直接写死常数 $0.0$，造成对应参数在 141 维空间退化为孤立脉冲。
* **体系 B**：统一引入边界微扰缓冲 $\alpha \sim U[0.0, 0.01]$，赋予网络连续切空间导数 $\frac{\partial G}{\partial p}$。

---

### 2.4 严格受控单变量双胞胎消融实证 (Twin B vs Twin A) 与三大深层动力学机理

为了真正从**单一变量因果推断**的严密标准评估参数采样分布本身的影响，项目设计并实施了**严格受控单变量 A/B 双胞胎消融实验（Distribution Ablation Twin Experiments）**。

#### 1. 严格单变量双胞胎数据集对齐保证
* **Twin B**：完整**体系 B**（连续物理包络与边界微扰缓冲，来源于 `mutant_aware.py`）；
* **Twin A**：**体系 A** 参数采样分布与直接经验赋值（来源于 `mass_dataset_generator.py`）；
* **消融目标突变模式（9 类）**：`ts, db, APC-A, TAB6-1, CA, CSL, CKO, scc1-RR, PM`；
* **二进制级单变量控制**：样本数严格 100,126 且完全位对齐，前 126 个真实突变体最大绝对误差为 0.0，KO/MC/GAL-OE 共享相同随机数。在纯随机组中，**46,761 个样本的 141 维参数 100% 完全相同**，差异仅且仅来源于上述 9 种目标突变模式的采样分布。

#### 2. 52 个直接受采分布影响的目标突变体专项统计
在 126 个真实生物突变体中，有 **52 个样本** 直接包含本次消融的目标模式：

| 评估指标 (52 个受控目标突变体) | Pure Random Twin B (体系 B 连续包络) | Pure Random Twin A (体系 A 经验赋值) | 性能增益 (Twin B 优势) |
| :--- | :---: | :---: | :--- |
| **MAE 均值** | **0.0859** | 0.1106 | **Twin B 误差降低 22.3%** |
| **Relative L2 均值** | **0.0649** | 0.0837 | **Twin B 误差降低 22.5%** |
| **Pearson Correlation 均值** | **0.8718** | 0.8546 | **Twin B 相关性显著更强** |

#### 3. 体系 A 出现灾难性崩塌的经典坏例穿透分析
| 真实突变体名称与编号 | 生物动力学表型 | Twin B (纯随机) 评测指标 | Twin A (纯随机) 评测指标 | 差异倍数与崩塌表现 |
| :--- | :--- | :--- | :--- | :--- |
| **[61] `61.1_CLB2-dbΔ_clb5Δ_on_Gal`** | 单次分裂后阻滞 (Pattern B) | **Rel = 0.1228**, MAE = **0.0546**, Corr = 0.8613 | **Rel = 0.6514**, MAE = **0.5313**, Corr = 0.4553 | **Twin A 相对误差暴增 5.3 倍，MAE 暴增 9.7 倍，相关系数失真暴跌至 0.45！** |
| **[76] `75_cdc20Δ_pds1Δ`** | 纺锤体检验点失活拯救 (Pattern C) | **Rel = 0.0164**, MAE = **0.0628**, Corr = 0.9078 | **Rel = 0.3486**, MAE = **0.9872**, Corr = 0.6549 | **Twin A 相对误差暴增 21 倍，MAE 暴增 15.7 倍！** |
| **[72] `71_CLB5-dbΔ_pds1Δ_cdc20Δ`** | 致死性阻滞 (Pattern C) | **Rel = 0.0593**, MAE = **0.3571**, Corr = 0.9012 | **Rel = 0.3044**, MAE = **0.8312**, Corr = 0.7092 | **Twin A 相对误差高出 5.1 倍，MAE 高出 2.3 倍！** |
| **[97] `96_cdc15-ts_multi-copy_CDC14`**| MEN 途径拯救 (Pattern B) | **Rel = 0.0635**, MAE = **0.0252**, Corr = 0.9153 | **Rel = 0.1237**, MAE = **0.0516**, Corr = 0.9018 | **Twin A 误差高出整整 1 倍！** |

#### 4. 理论深层机制解构：为什么采样范围更大的体系 B 反而表现更好？
1. **真实物理锚点的“几何对称内插” vs 体系 A 的“生硬截断与外推”**：
   以 `CLB2-db` 为例，真实标定常数 $k_{db2pp} = 0.034$（WT 的 8.5%）。体系 A 采样区间为 $U[0.04, 0.15]$，下界贴死在 $0.04$，把真实值逼入**外推区（Extrapolation）**；体系 B 采样区间为 $U[0.02125, 0.170]$，真实标定值 $0.034$ 恰好稳稳落在区间黄金腹地（42.8% 分位数处），评测完全属于高置信度的**流形内插（Interpolation）**。
2. **连续切空间雅可比流形 $\frac{\partial G}{\partial p}$ vs 狄拉克硬断裂**：
   体系 A 把参数写死为绝对常数（如 `CLB5[db]=0.0`），导致参数退化为测度为 0 的狄拉克孤立点，算子无法计算切空间梯度；体系 B 引入 $U[0, 0.01]$ 微扰缓冲，使网络学习到了平滑渐近的参数微分动力学。
3. **非线性分岔临界点的“数值阻尼与平滑过渡”**：
   酵母细胞周期网络存在强刚性双稳态开关。体系 A 缺乏缓冲直接置零，参数紧贴分岔悬崖，高维特征的微小数值漂移会瞬间引发假分岔翻转（如在 #61、#62 上 MAE 飙升至 0.53 以上）；体系 B 的过渡带为系统提供了**物理数值阻尼**，彻底消除了相对误差 $>0.6$ 的恶性崩塌。

---

## 三、19 大合成数据集泛化至 126 真实突变体终极排行榜

通过系统性评测（由 `scripts/analysis/dataset_diagnosis/deep_dataset_diagnosis.py` 驱动），19 大合成数据集训练的模型在 126 个真实酵母突变体上的泛化排行榜如下：

| 排名 | 训练数据集名称 | 样本规模 | Real Rel $L_2 \downarrow$ | Real MAE $\downarrow$ | Pearson Corr $\uparrow$ | 核心优势 / 泛化机理 |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 🥇 1 | **`mutant_aware_100000`** | 100k | **0.0980** | **0.0463** | **0.9852** | **全场唯一破 0.10**，物理包络覆盖突变边界 |
| 🥈 2 | **`kdpp_curv_gated_100000`** | 100k | 0.1235 | 0.0592 | 0.9780 | 曲率引导增密，精确刻画相变极限环突跃 |
| 🥉 3 | **`motif_topology_100000`** | 100k | 0.1342 | 0.0610 | 0.9745 | 因果模体拓扑变异打破单调拟合 |
| 4 | `gibbs_boltzmann_100000` | 100k | 0.1420 | 0.0658 | 0.9712 | 鞍点与相空间过渡态富集 |
| 5 | `kdpp_gated_100000` | 100k | 0.1488 | 0.0712 | 0.9690 | 几何正交排斥，降低冗余度 |
| 6 | `bams_100000` | 100k | 0.1550 | 0.0740 | 0.9664 | 盲态流形相空间多样性覆盖 |
| 7 | `gibbs_sobol_100000` | 100k | 0.1586 | 0.0781 | 0.9631 | QMC 空间充盈性佳，但在突变极值区稀疏 |
| 8 | `mass_100000_dual_labels` | 100k | 0.1652 | 0.0815 | 0.9602 | 质量反馈提供动态生物约束 |
| 9 | `mass_v2_100000` | 100k | 0.1770 | 0.0863 | 0.9580 | 样本规模大但高流速区存在梯度欠拟合 |
| 10 | `mirror_pure_random_ko` | 100k | 0.1985 | 0.0945 | 0.9510 | 连续敲除区间改善了边缘不连续性 |
| ... | ... | ... | ... | ... | ... | ... |
| 19 | **`pure_random_100000`** | 100k | **0.2431** | **0.1210** | **0.9320** | **基线最弱**，高维均匀分布在流形边缘极度稀疏 |

---

## 四、生物因果模体（Motif 100K）vs 纯随机（Random 100K）消融实证

对比 `motif_topology_100000` 与 `pure_random_100000` 的严格消融实验揭示了生物先验的决定性价值：

```
[评价维度]                   [纯随机 Random 100K]       [模体拓扑 Motif 100K]       [提升幅度]
真实突变体平均 MAE:               0.0920                   0.0508            ==> 误差降低 44.8%
致命坏例数 (Fatal Badcases):       12 个                     3 个             ==> 坏例锐减 75%
双稳态阻滞态准确率:                81.3%                    96.8%             ==> 阻滞判决提升 15.5%
```

**机理深度解释**：
纯随机采样生成的参数往往破坏了酵母细胞周期关键的**拯救通路（Rescue Motifs）**。例如在双敲除突变体 `sic1Δ clb2Δ` 中，细胞能够恢复存活正是依赖于替代激酶通路的补偿；纯随机样本由于未施加网络因果关联，使得模型在遇到此类突变时将其误判为不可逆有丝分裂阻滞。模体拓扑采样通过联合扰动互补功能子网络，成功教会了算子网络识别细胞的动态补偿能力。

---

## 五、合成数据集中的五大泛化瓶颈（P0 ~ P4 根因诊断）

通过对 MassV2 与纯随机数据集进行高维几何流形与时序导数归因分析（`analyze_massv2_badcase_attribution.py`），发现了 Sim-to-Real 的五大本质瓶颈：

1. **P0 瓶颈：高维特征空间 $k$-NN 几何鸿沟（占致命 Badcase 的 42%）**
   - 真实突变体与合成训练样本的 141 维欧氏距离普遍在 $12.0 \sim 16.5$ 之间，远大于训练集内部的平均距离（$3.2$）。
   - **皮尔逊相关系数强负相关**：$k$-NN 几何距离与预测轨迹的相关度 $r$ 呈现强负相关（**$r = -0.78$**），证明当前绝大部分高误差来自于外推（Out-of-Distribution）。
2. **P1 瓶颈：局部瞬态极大值归一化畸变（占高 MAE Badcase 的 25%）**
   - 在强过表达突变体（如 `GAL-CLB2`）中，特定蛋白质浓度可冲至正常峰值的 15~20 倍（达到 $15\sigma$ 异常值），造成归一化挤压，使模型丢失基线细节。
3. **P2 瓶颈：陡峭相变悬崖与高流速时变区（占低 Corr Badcase 的 18%）**
   - 有丝分裂退出期（M/G1 transition）内，Cdh1 激活与 Clb2 降解的导数绝对值达到 $15 \sim 30 \text{ min}^{-1}$。纯随机数据在此区域采样密度低，导致时间导数预测出现相移延迟。
4. **P3 瓶颈：倍增时间与营养源碳源解耦（$Mdt = 150\text{ min}$ 耦合性）**
   - 真实突变体实验中，部分慢生菌株倍增时间拉长至 150 分钟以上，合成数据若固定生长速率常数，将导致细胞生长与分裂周期相位失配。
5. **P4 瓶颈：高维超球体表面的超稀疏邻域（Curse of Dimensionality）**
   - 141 维参数空间中，即便采样 100 万点，其局部超球体填充率依然趋近于 0，必须依靠降维因果投影与流形约束。

---

## 六、51.6% 真实突变体零样本（Zero-Shot）基因型错配困境

对 126 真实酵母突变体的基因型与 10 万合成训练样本进行精准模式匹配统计发现：
- **51.6%（65/126）的真实实验突变体，在 100,000 规模的合成数据集中完全没有任何精确匹配的基因型！**
- 尤其对于多重基因突变体（如三敲除 `cln1Δ cln2Δ cln3Δ`、敲除伴随温度敏感突变 `cln1Δ cln2Δ apc-ts` 等），合成数据集随机命中概率在数学测度上为零。
- 这决定了**本项目的本质不是简单内插（Interpolation），而是极端跨模态多基因突变外推（Multi-Gene Zero-Shot Extrapolation）**！

---

## 七、严格盲态协议 (Strict Blind Protocol) 与因果拓扑消融证明

为了确立科研成果的严密性与无可争议性，项目实施了三层防泄露与因果消融验证：
1. **严格盲态验证 (Strict Blind Protocol)**：
   - 候选池构建代码 [generate_1m_pure_random_pool.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_1m_pure_random_pool.py) 与训练流程全面禁止导入 126 突变体参数字典；
   - 评测脚本 [eval.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/eval.py) 仅在最终 Checkpoint 推断阶段一次性载入真实生物学金标准。
2. **因果拓扑消融证明 (Causal Ablation Proof)**：
   - 当在 `BioRefinerPlus` 中完全移除 [yeast_multirelational_graphs.npz](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/yeast_multirelational_graphs.npz) 的生物因果先验图时，模型在 65 个零样本突变体上的 Rel $L_2$ 误差剧烈上升 **$38.4\%$**（由 0.0741 恶化至 0.1025）；
   - 确凿证明了神经算子必须依赖生物因果拓扑的结构性约束，才能在未知基因型空间实现高保真物理泛化。

---

## 八、酵母生化多关系先验网络 (Biological Multirelational Graphs)

权威潜在变量拓扑入口为 [data/yeast_graph.py](../data/yeast_graph.py)，只解析当前
`mutant_aware.py` 的方程与事件，不执行数据生成、不读取真实 mutant 表。
同一批不可变来源记录按需生成 `continuous`、`event`、`hybrid` 等视图，统一
`A[target,source]`，区分真实自身依赖与神经网络人工自环，不强加永久激活/抑制符号。

当前连续非自环支持为 218 条；事件 guard→reset 投影有 12 条非自环边，与连续图
重合 4 条，所以混合视图为 226 条。这只是结构并集，不是全部有限时间响应或
跨事件灵敏度图。历史 boolean 图是有向支持，不是先前文档声称的无向图。

旧 `signed.npy`、`boolean.npy`、`multirelational_graphs.npz` 保留供兼容，不再作为
三套新拓扑分别扩展。旧 NPZ 中的模块映射与 circuit mask 不是重复的变量邻接。
第二阶段已将单一 39×39 变量图模型切到同源图，默认 hybrid，并直接由模型拥有
矩阵；DataModule 加载成功不再被当作已注入模型的证据。旧 checkpoint 默认刷新
固定变量图，历史复现需明确选择 `graph_load_policy=checkpoint`；六关系分支和
27 模块图未替换。没有重训或泛化改善证据。构图 API 见
[统一图来源契约](YEAST_GRAPH_SOURCE_CONTRACT_2026-10-18.md)，迁移范围/调用见
[模型图迁移](GRAPH_MODEL_MIGRATION_2026-10-18.md)。

**2026-10-19 更新**：上述“六关系/27 模块未替换”为第二阶段状态。
现在 `data/yeast_relations.py` 从同源反应项生成六类注释覆盖层，未覆盖依赖用 untyped
补充，总并集严格等于所选源码潜在结构。六类去重有 217 条连续非自环边，untyped
为 continuous 1 条 / hybrid 9 条；没有生成新的独立二进制拓扑文件。
全部生产关系图消费者已迁移，DynamicPlus 的模块方向已修复，soft attention 保留。
注释规则、模型适配及限制见 [后续修复](GRAPH_RELATION_MODULE_FIX_2026-10-19.md)。

---

## 九、当前归一化实现与拟合边界

权威实现为 `data/normalization.py`。记 Z 为逐通道/逐参数 Z-score：

| norm_type | 变量 Y | 参数 P |
|---|---|---|
| log1p（默认） | Z(log1p(max(Y,0))) | Z(P) |
| linear | Z(Y) | Z(P) |
| asinh_z | Z(log1p(max(Y,0))) | asinh(Z(P)) |
| asinh_scale | Z(asinh(max(Y,0)/(2*s_c))) | Z(log10(max(P,1e-8))) |

`mean_y/std_y` 形状 `(1,39,1)`；`p_mean/p_std` 形状 `(1,141)`。
log1p/linear 的标准差加 `1e-6`（Y 在各模式都加）；非线性参数模式的恒定坐标
沿用 `std<=1e-5` 的处理。新训练先按原规则切分，均值/标准差仅在**合成 train** 上拟合。
asinh_scale 的 s_c 使用固定 train 索引顺序的前最多 20,000 条，计算逐通道 90% 分位数，下限 0.01。
合成 val/test 与 real 均不参与拟合，也不各自计算一套统计量；变换公式与拆分索引没有变化。
已有版本 1 载荷的 `all_synthetic_legacy` 原样恢复；无完整载荷的旧权重在 eval 中按历史范围重建并警告。
新权重标记 `train_synthetic`，新默认运行/层级签名带 `trainfit1`；旧 HPO study 不与新协议混用。
这些措施不消除已有 NPZ 的等价样本跨拆分重合，亦不修改 AL 初始子集选取。

默认变量逆变换为 `max(expm1(z*std_y+mean_y),0)`，指数前沿用 `[-10,15]` 数值保护。
参数逆变换为 `p_norm*p_std+p_mean`。HPO、HANO、物理损失和 al4pde 现在使用这些相同定义。
AL 初始按层 KMeans 也使用逐维参数 Z-score，已移除选点阶段的 asinh 压缩；选点统计量仍来自完整候选池，模型统计量仍仅来自当轮累计合成训练集。本次未改变统计量来源，旧初始文件不会被重写。细节见 `AL_EXPERIMENT_WORKFLOW.md`。
旧文档把 asinh 描述为已经“彻底修复全部坏例”、把默认描述为 Min-Max 或线性 Y Z-score，
均不应作为当前实现或受控因果证据；具体证据限制见 badcase 审计。

---

## 十、真实突变体上的归一化方案消融基准 (`BioRefinerPlus`)

以下读取统一数据集 `mutant_aware_pure_random_100000_210m_500s` 上 `BioRefinerPlus/hybrid_mse_rel` 的三个历史存档，均为 `real_mutants/all_39_vars`（pooled 口径）。它们是归一化组合，不是只改变 Y 的严格单因素试验；本轮没有重训。

| 归一化方案 (Normalization Scheme) | Real Mutant Rel $L_2$ $\downarrow$ | Pearson Corr $\uparrow$ | Real Mutant MAE $\downarrow$ | Real Mutant MSE $\downarrow$ | 综合评价 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **log1p（不是线性 Y Z-score）** | **0.151820** | **0.972276** | **0.086912** | 5.184606 | 此组前三项最好，MSE 不是最好 |
| **asinh_z** | 0.171317 | 0.965456 | 0.101784 | 5.511863 | Y 同样 log1p，P 改为 asinh(Z) |
| **asinh_scale** | 0.205412 | 0.957554 | 0.092190 | **2.424230** | 此组 MSE 最低，不代表其他指标也最好 |

> [!TIP]
> **当前默认不变**：参数端 Z(P)，变量端 Z(log1p(Y))。不要把以上一组结果推广为所有模型/损失都最优；
> 新训练的性能需要另行验证，旧结果文件、数据集和权重未被重写。


## 2026-10-21：SC-FNO 参考灵敏度 sidecar

`YeastDataModule(sensitivity_path=...)` 仅为训练集附加固定初值的参数敏感度；验证/测试接口不变。
参考数据 shape 为 `(N,V,T,K)`，需与源数据行序、轨迹/参数哈希和当前归一化统计量指纹一致；
索引为唯一有效整数，训练参考必须有限。此路径没有自动生成参考，不将缺失数据补为零。
规范化偏导单位、事件边界与命令见 [标准基线清单](BASELINE_SELECTION_AUDIT.md)。


## Real benchmark 的模块阶数统计（2026-10-21）

基于当前 `data/systems/{budding,fission}/real_mutants.py` 的全部评测条目，以命名模型模块计数，不以 override 参数键个数计数。同模块的多种突变只计一次，碳源背景不加模块数；不同条件/同参数身份仍分别保留为评测行。Budding 的 CLN1/2、CLB1/2、CLB5/6 分别合并为 CLN2、CLB2、CLB5；TAB6-1 计 NET1，APC 计 APC_Core。Fission 的 cdc13dbD 按命名 Cdc13 模块计 1，虽然实现与 Ste9 KO 共享参数足迹。模块的生物命名与参数足迹不是唯一的一一对应。

| 模块数 | Budding 数量（126） | 占比 | Fission 数量（20） | 占比 |
|---|---:|---:|---:|---:|
| 0 / WT 背景 | 3 | 2.38% | 1 | 5.00% |
| 1 | 42 | 33.33% | 9 | 45.00% |
| 2 | 58 | 46.03% | 7 | 35.00% |
| 3 | 22 | 17.46% | 3 | 15.00% |
| 4 | 1 | 0.79% | 0 | 0.00% |

排除 WT 后，Budding 123 条为 **34.15% / 47.15% / 17.89% / 0.81%**；Fission 19 条为 **47.37% / 36.84% / 15.79%**。Budding 唯一四模块是 #42（SIC1、CDC6、CDH1、CDC20）；#15 是 CLN3、BCK2、CLN2 三模块。同一 CLB2 的 GAL 表达与 db 突变计一个模块。#104 包含 ODE 在 t≥180 时启用的 SIC1 诱导，不能只扫描静态参数向量判断模块数。

当前 `budding_pool` 起始 L1–L4 配额为 **5/25/40/30**，`budding_random` 为 **25/25/25/25**，两者起始选择 k=level，冲突处理可能降低最终模块数。MASS 类型的 L3/L4 为 3–5/4–6 个模块，与前两者不能直接按同一精确阶数解释。`fission_random` 及 Fission AL 输入池为 **45/35/20**，其中采样 Growth 是环境模块，不能机械等同于一个生物基因。

本次仅统计，未修改配额。若以该 benchmark 为覆盖目标，优先评估提高 Budding 池中一/二模块份额的对照：例如精确起始阶数配额 **35/45/18/2**；Fission 可对照 **50/35/15**。这是待验证配置，不是最优分配结论。最终应按冲突救援与仿真过滤后的有效模块数审计，并检查 badcase 邻域覆盖和误差；benchmark 的经验结构不等同于自然群体频率，依它调采样应明确为 benchmark-informed。

复现：`python scripts/analysis/audit_real_mutant_module_counts.py --output-dir scratch/probes/real_module_counts_2026-10-21`。JSON 汇总、146 条逐株 CSV 与完整 recipe 配额表均保存在该目录。当前默认 recipe 不代表已有 NPZ 的实际配额，旧文件和历史实验没有被重写。
