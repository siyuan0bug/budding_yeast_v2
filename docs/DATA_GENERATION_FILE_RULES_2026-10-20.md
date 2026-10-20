# 每个文件的新旧规则是否一致

> 阶段性审计记录。后续已迁移生产消费者、恢复部分旧 CLI 默认值和可选项；
> 当前文件名、接口和留存范围见 [当前速查](DATA_GENERATION_QUICKSTART.md)。
> 以下“44 文件未变/54 新文件/未改生产规则”描述的是此次消费者迁移之前的状态。

## 结论

**目前不能判定所有文件的新旧规则一致；已经确认存在完整流程、默认配置和可用选项的差异。**

此前 105 组采样和 381 项分布/模块检查证明的是特定条件下的核心规则一致：相同配置、相同初始状态，并且采样调度对照绕过碰撞。它们不能证明默认命令、重复调用、所有失败分支及整个文件等价。此前把这些检查概括为整体一致，范围过宽。

本轮只做只读核对、增加审计脚本与文档，未继续改变生产规则。以下分别列明用户指定的统一事项、当前额外差异，以及尚未获得全分支等价证明的部分；不把此前新增的修复自动当作用户指定的等价例外。

## 核对覆盖

- 原始 44 个 Python 文件全部纳入清单，SHA256 均未变。
- 其中 17 个文件是本次迁移的生成器；14 个为用户明确不迁移的 LHS/real_mutants_only 文件；13 个为继续使用的其他原模块。
- 新增 54 个 Python 文件全部列出来源，包括包初始化和没有一对一旧文件的公共执行模块。
- 迁移生成器共 169 个顶层函数/类：125 个移除文档、后端注入和可选 RNG 接口后结构匹配，8 个存在结构差异，36 个主流程/包装函数被公共代码替代。**125 个定义相同不等于 17 个文件所有规则相同。**

每个文件、每个定义的旧行号、新位置和结构结论见 [完整出处清单](DATA_GENERATION_RULE_INVENTORY_2026-10-20.md)。原函数默认值、CLI 默认表达式、常量及 NPZ 键集合还保存在 `scratch/probes/file_rule_inventory.json`。

## 逐生成文件结论

表中的“核心一致”只指该列明确列出的规则。所有生成器还受到后文公共执行与导出差异影响。

| 旧文件 | 新位置 / 配置 | 核心规则结论 | 尚有的新旧差异 |
| --- | --- | --- | --- |
| `mutant_aware.py` | `systems/budding/`、`sampling/mass.py`、`budding_mutant_aware` | 参数、初值、ODE、标签、real、突变/分布、结构选择和模式协调沿用 MA；初始化拆成单独函数 | 一次救援后丢弃/重抽改为有限重复救援；max_attempts 语义改变；重复调用不再沿用 L1/配额状态 |
| `mass_dataset_generator.py` | `mutations/mass.py`、`distributions/mass.py`、`sampling/mass_motifs.py`、`budding_mass` | MASS 的突变赋值、幅度分布和原定向 motif 保留 | 基础模型/real 初值统一 MA；目录增加两个安全初值字段；原丢弃碰撞改为救援；尝试上限与执行/导出改变 |
| `mass_random_sampler.py` | `sampling/random.py`、`budding_mass_random` | 加权基因、模式、复合突变、幅度与赋值核心一致 | 旧 Python 函数默认比例为 .25/.35/.25/.15，旧 CLI 为 .40/.30/.20/.10；新配置选了 CLI 比例，因此不等于旧函数无参调用；碰撞改救援 |
| `mass_random_mutant_dist.py` | `sampling/random.py`、`budding_mass_random_dist` | MASS 赋值 + MA 分布组合、层级 rounding 和顺序一致 | 原碰撞直接丢弃；新救援。旧 worker 引用 MASS 生物模型，新统一 MA |
| `generate_pure_random_dataset.py` | `sampling/random.py`、`budding_random` / `budding_random_ko_interval` | 均匀基因/模式、复合突变、KO 两种形式、分布及显式 RNG 核心一致 | 旧函数可关闭 collision rescue；新入口未暴露该开关，固定救援。旧一次救援失败重抽，新重复救援耗尽报错；默认 workers 由 50 改为统一值 |
| `gibbs_boltzmann_sampler.py` | `sampling/gibbs.py`、`budding_gibbs` | Gibbs 核、温度 .8、层级 floor/shuffle 和候选赋值一致 | 旧碰撞丢弃改救援；旧函数默认 10k，新公共 API 100k；fast-param-only 的 real、假 data/标签与新参数池式输出不同；重复调用状态不同 |
| `gibbs_sobol_sampler.py` | `sampling/gibbs.py`、`sampling/sobol.py`、`budding_gibbs_sobol` | Gibbs 核、16 维 Sobol、逆 CDF、rounding 差额补 L1、层级顺序一致 | 同上：碰撞、函数默认样本数、fast-param-only 与重复调用语义不同 |
| `generate_kdpp_gated_dataset.py` | `sampling/kdpp.py`、`budding_kdpp` | diffusion 核、tau、谱采样、模式协调和候选赋值一致 | 旧一次救援失败重抽，新最多 30 次救援后报错；因此不是所有碰撞后的序列一致 |
| `generate_kdpp_curv_gated_dataset.py` | `sampling/kdpp.py`、`budding_kdpp_curvature` | signed 核、1.35 候选队列、曲率公式、.01 阈值、.20 flat 概率及正常配额接纳逻辑一致 | 碰撞救援、失败处理及公共输入校验/导出不同；完整异常分支不等价 |
| `generate_motif_topology_dataset.py` | `sampling/motif.py`、`budding_motif` | 模体目录、选基因/模式、扰动、L4/L5 的 70/30 拆分一致 | 旧拒绝 real 碰撞，新救援；原函数接收 4 个比例，新接口要求 5 个有效层级比例；公共入口会设置全局 NumPy seed，旧函数本身只创建局部 RNG |
| `bams_dataset_generator.py` | `sampling/bams.py`、`budding_bams` | signed walk、细化区间、3 步二分、jitter 和最终参数回传一致 | **新入口缺少旧 `--no_bisection` / enable_bisection=False**；新 worker 随机流按 sample_id 初始化；全局状态、救援重试和总尝试上限不同 |
| `mass_wt_blind_sampler.py` | `sampling/wt_blind.py`、`wt_pipeline.py`、`budding_wt_blind` | 46 边专用图、QSS、GAL 逆解、log bands、ledger 和相同显式配置下的接纳循环一致 | **旧 CLI pilot=20000，新默认=1000**，会改变分箱和接纳结果；旧 nan_cap_frac、backlog_thresh、max_attempts_mult 可配置，新前两项写死；旧函数 fold 默认 flat、新配置 tier2（旧 CLI 也是 tier2）；重复调用状态及失败后的 ledger rollback 不同 |
| `generate_1m_pure_random_pool.py` | `sampling/sobol.py`、`budding_pool` | 8 维 Sobol、基因/模式、逆 CDF、幅度与覆盖值核心一致 | **旧默认 1,000,000，新 CLI 默认 100,000**；旧池不做 real 碰撞过滤，新公共流程做救援；新 y_init 使用 canonical 组装且存为 float64，旧为 float32；元数据不同 |
| `generate_distribution_ablation_twins.py` | `sampling/twins.py`、`generation/derived.py`、`budding_twins` | 共享骨架/非目标剂量和两套目标赋值核心一致；两侧共同成功才输出的规则保留 | 旧只检测 B 侧 real 碰撞并丢弃，新检测两侧并救援；旧总尝试上限为 10×样本数，新按有限救援；执行方式和来源元数据不同 |
| `mirror_pure_random_ko_interval.py` | `generation/derived.py`、`budding_ko_mirror` | KO 的 U[0,1e-4] 与 active≤total、正常非 KO 行复用、KO 行重算规则一致 | 旧固定前 126 行为 real，新按身份判断；旧重算失败仍可能留下新参数+旧轨迹，新报错；新增加碰撞救援、源校验和 no-real 处理；默认 workers 不同 |
| `generate_bams_ablation_datasets.py` | `sampling/targeted_ablation.py`、`generation/derived.py`、`budding_bams_ablation` | 定向 skeleton、hard/hypo 参数与配对剂量核心一致；配置 50k 总数时 40k 基底/10k 目标比例一致 | **旧固定 control=50k、base=40k、target=10k、seed=20260915；新默认总数100k、目标20k、seed42**；旧两臂分别打包、失败可零填充，新共同成功过滤；源校验和编号不同 |
| `fission_random_sampler.py` | `systems/fission/`、`sampling/fission_random.py`、`fission_random` | 16 变量、58 参数、ODE、事件、预热、标签、20 个 real、模块与幅度核心一致 | **旧默认1000/seed20000，新默认100000/seed42**；旧碰撞丢弃改救援；重复运行重建网格；超时传播和完整300min检查更严格 |

## 用户已指定的统一事项与其他改变不能混称

用户已经明确指定的规则是：

1. Budding 基础模型和 real 以 MA 为准，不沿用旧 MASS 有问题的 real 初值。
2. 保留 MASS、MA 两套合成突变与分布，保留各采样算法。
3. 碰撞全部救援。
4. LHS/废弃 real 脚本不迁移；Fission 有独立模型。

这些要求本身就意味着新版本不能逐文件复现每个旧版本的全部行为。但是 pilot 默认数、遗漏开关、统一 CLI 样本数、重复调用状态、公共 max_attempts 语义等，并不能仅凭这些要求被说成“与原规则一致”。

本次审计没有将这些差异默默改掉，也没有把它们归类为已全部获得用户确认的例外。

## 公共执行、失败与导出差异

这些差异适用于多份文件，逐文件表不重复展开。

- **默认命令**：新 CLI 统一 samples=100000、seed=42、workers=min(60,cpu_count)，并要求明确输出路径。旧脚本各有自己的默认样本数、worker 数、路径和文件名。
- **参数默认与 CLI 默认不总相同**：如旧 Gibbs 函数默认10k而 CLI 默认100k，WT 函数默认200、fold=flat而 CLI 默认100k、fold=tier2。对照时必须说明参照哪个入口。
- **尝试上限**：旧 MASS/BAMS/Fission 等的 max_attempts 限制整个候选抽取循环（常见 max(50000,10×N)）；新值主要限制单候选救援次数，WT 还复用为总接纳尝试上限。相同参数名不能视为相同规则。
- **比例输入**：新公共层统一归一化比例并拒绝部分小样本负配额；旧不同函数有的归一化、有的直接按给定比例计算。默认有效比例匹配不代表任意比例输入都等价。
- **失败分支**：新层统一拒绝非有限/不完整轨迹、失败 status、重复 ID、错误源元数据和空结果；旧不同导出函数的接受、零填充、忽略异常方式不相同。
- **输出**：主要 data/params/names/labels 保留；新增加 y_init、is_real、sample_id、pair_id、系统版本、配置、覆盖字典和报告。参数池无假轨迹，覆盖默认拒绝、原子发布，都是导出行为差异。
- **可选后处理**：`clean_wb_dataset.py` 仍是原工具，没有被等价合并到新生成入口中；不能把其参数阈值、轨迹阈值清洗自动视为新生成流程已执行。

## 重复调用差异已实际复现

不启用碰撞、seed=42、每次32个候选，对 MA MASS、BAMS、Gibbs、Gibbs-Sobol 分别连续调用两次：

| 对照 | 4 种算法结果 |
| --- | --- |
| 第一次 old vs new | 全部一致 |
| 旧版本第一次 vs 第二次 | 全部不同（延续旧状态） |
| 新版本第一次 vs 第二次 | 全部相同（每次重置） |
| 第二次 old vs new | 全部不同 |

证据：`scratch/probes/repeated_driver_rule_differences.json`。因此此前审计为了隔离公式而主动 reset 旧状态的做法，不能用于证明同进程重复运行也完全相同。

## 8 个结构差异的具体解释

| 旧定义 | 源码结构差异 | 是否可直接说整个函数规则相同 |
| --- | --- | --- |
| MA `simulate_mutant` | 原初始化块提取为 prepare_initial_conditions | 初始化及后续公式按 MA 保留，126 条对照支持；这里是函数拆分 |
| MASS `simulate_mutant` | 改为 MA 的模型/初值处理 | 否，这是用户指定的 canonical 统一 |
| Fission `run_event_sim` | info 新增 completed_time | 数值公式保留，返回结构不同 |
| Fission `mutant_birth_state` | TimeoutError 不再按一般异常回退 WT | 正常预热规则保留，超时分支不同 |
| Fission `simulate_mutant` | 新增初值/完成时长元数据，继续抛出超时 | 正常数值规则保留，返回及超时分支不同 |
| 旧 pool `sample_scaling_factor_sobol` | return表达式 vs 赋给局部val再return | 此处表达式及返回值相同，无额外随机抽取 |
| Twins `apply_gene_mutation_A` | 目标模式手写分支改为 MASS 委托+TEM1-ts初值补充 | 合法目标模式调用域已有对照；不承诺旧函数未处理的模式输入也具有相同副作用 |
| WT `build_candidate` | 新增按 backend.distribution_profile 选择分布 | 默认 mass 的同配置调用匹配；显式改成 MA 时本来就选择另一套规则 |

其余36个没有一对一同名匹配的定义，主要是各生成主函数、main、worker、pack 和救援包装，由公共 generation/CLI 替换；必须按逐文件表和公共差异解释，不能算作结构匹配。

## 保留工具与图来源边界

13 个未迁移模块包括：包入口、AL pool manager、构图工具、WT cleaner、dataset_utils、normalization、parse_target、preprocessing_contract、reachability、simulation_guards、yeast_datamodule、yeast_graph、yeast_relations。它们没有被改写，因此其自身规则仍是旧规则。

但原图工具/训练链路仍可读取原 `mutant_aware.py`，新生成器读取 `systems/budding/`。当前公式对照一致，不代表将来只修改新 ODE 后，所有旧图工具也会自动跟着更新。这是迁移连接范围，不能用“源文件留存”证明全仓库已只剩一个生物规则入口。

## 复跑清单审计

```bash
PYTHONDONTWRITEBYTECODE=1 /home/users/zsy/.conda/envs/neuralop/bin/python \
  scripts/analysis/audit_rule_inventory.py \
  --output scratch/probes/file_rule_inventory.json \
  --markdown-output docs/DATA_GENERATION_RULE_INVENTORY_2026-10-20.md
```

该脚本只读取源码。它不会把没有结构匹配的函数自动判为等价，也不会将测试通过数量换算成整文件规则一致。
