# 实验采样分布与 real-mutant 指标审计（2026-10-20）

## 1. 结论和审计边界

若仅排除 motif/拓扑定向组合，现有记录支持把**原始 pure-random + BioRefinerPlus + triple_dynamic**作为优先基线；
若允许基因/模式有非均匀权重，MASS-random 同样值得保留。
这不是已经找到与模型、损失、代码版本无关的“最优采样分布”。

若要求训练设计**完全不利用 real-mutant 参数/轨迹来设采样范围**，当前 pure-random 也不能直接认证合格：
它依赖的 mutant_aware 幅度区间中有明确 real-anchor 包络，并用 real 规则检查碰撞。
本轮只做归因、排名与报告，不修改这些采样规则。

- 只读扫描 experiments 中 **192** 份 global_metrics_denorm.json，涉及 **29** 个有指标的一级目录。
- 全部记录的 seed 都是 **42**；目录含版本副本、AL 各轮及最终摘要，不能当作 192 次独立复现实验。
- **14** 份指标只含 122 个 real mutants，其他为 126；不同集合不混排。
- 所有 192 份的 CSV 行数与 real count 一致；逐样本 MAE/MSE 平均与 JSON 对应值在容差内一致。
- relL2/Corr 以 JSON 的 real_mutants/all_39_vars 为准，**pooled**，不与逐 mutant 算术平均混用。
- 匹配组 real-mutant 名称集合相同（除排除的 1mass 旧归档外），但这不证明不同生成版本的真值数值完全相同。
- 代码版本存在差异；均为历史存档，并非当前图修复、train-only 归一化和 RNG 修复版本的统一重训。
- 未重新推理、训练、积分真实 mutant 或全量读取轨迹；抽查当前同名 NPZ 的头部/样本名，并逐行比较两对 Twin 的参数。
- 原始 pure-random 既有完整哈希审计确认 train 与 val/test 有重合、real 有3条重合；本次未重扫轨迹。
  其他数据集未完成同等级查重，不能把下列存档成绩直接视为严格独立泛化。
- 新建本报告与审计 CSV；没有改动生产代码、原 NPZ、checkpoint 或原评估文件。

完整原始清单：[run_inventory.csv](../scripts/analysis/outputs/sampling_distribution_audit_2026-10-20/run_inventory.csv)。

## 2. 主要同配置历史比较

固定 BioRefinerPlus / hybrid_mse_rel / log1p / one_time / AL=none / seed42；
width192、modes24、blocks4、batch256、epochs100、lr0.00088、warmup8。
下表是 100k 级文件名组（不是所有文件的最终保留样本数都已复核），按 MAE 排序。
模型/loss/超参一致仍不能消除跨 git 版本的混杂；不称严格单因素消融。

| MAE 排名 | 数据集简称 | relL2↓ | MAE↓ | MSE↓ | pooled Corr↑ |
|---:|---|---:|---:|---:|---:|
| 1 | mutant_aware | 0.098039 | 0.050027 | 0.740209 | 0.988552 |
| 2 | kDPP-curvature | 0.132905 | 0.060365 | 0.929716 | 0.978078 |
| 3 | 旧 Twin-B（骨架待追溯） | 0.105124 | 0.060714 | 1.861685 | 0.987266 |
| 4 | 旧 Twin-A（骨架待追溯） | 0.114764 | 0.061351 | 1.425010 | 0.984237 |
| 5 | kDPP | 0.143608 | 0.062236 | 0.936842 | 0.973928 |
| 6 | MASS-random | 0.124909 | 0.063306 | 1.130352 | 0.979102 |
| 7 | Gibbs-Sobol | 0.137604 | 0.064339 | 1.291347 | 0.976075 |
| 8 | pure-random-rescue | 0.147512 | 0.069624 | 1.070362 | 0.972160 |
| 9 | BAMS | 0.132467 | 0.077018 | 1.783581 | 0.977215 |
| 10 | pure-random-KO-interval | 0.152796 | 0.081037 | 1.987756 | 0.971966 |
| 11 | pure-random Twin-B | 0.158334 | 0.082969 | 2.084944 | 0.969458 |
| 12 | 原始 pure-random | 0.151820 | 0.086912 | 5.184606 | 0.972276 |
| 13 | MASS-random-dist | 0.180318 | 0.086977 | 1.554738 | 0.956659 |
| 14 | pure-random Twin-A | 0.156859 | 0.089117 | 2.388915 | 0.968697 |
| 15 | motif-topology | 0.221407 | 0.132510 | 17.181028 | 0.951995 |

[机器可读同配置表](../scripts/analysis/outputs/sampling_distribution_audit_2026-10-20/matched_hybrid_log1p.csv)。

### 未混入上表的记录

- Gibbs-Boltzmann 无后缀目录的 config 实际是 **asinh_scale**，不是 log1p：
  MAE=0.062351，MSE=1.163715，relL2=0.123592，Corr=0.981364。
  不能因为目录未写 asinh 就把它当同归一化对照。
- 50k BAMS / hard / hypo 的 MAE 分别为 0.091945 / 0.076336 / 0.085686，样本量不同，单独比较。
- 旧 MASS 100k 的最小 MAE 0.034662 来自 BioRefinerGraph，宽度/损失/模型与主表不同；
  还有明确 rescue-motif 模板，不能据此选为纯随机分布冠军。
- 旧版/编号目录代表历史代码或数据副本，不当作额外 seed。
  1mass_random_dist 与 mass_random_dist 配置指向同一源文件名，real 名称集合却不同，不认证为两个独立分布。
- rolemode / leaklevel 负对照有巨大误差，见完整表；不当作普通随机候选。
- mass_100000_random_level... 目录未找到 global_metrics_denorm.json，不能由目录名补造结果。
- 旧 distribution_twin_A/B 缺对应生成脚本快照，当前脚本默认只生成 pure_random 前缀那一对。
  旧双胞胎的离散骨架来源未确证，不能标为无拓扑均匀随机。
- AL 最终摘要中，hybrid_mse_rel 的 uncertainty 为 MAE=0.076520、MSE=0.960852、
  relL2=0.141739、Corr=0.976027；random 为 MAE=0.082654、MSE=1.292552、
  relL2=0.184788、Corr=0.957003。相应末轮目录分别标记 train35036/train33733。
  这是不同主动选样和预算下的记录，不是相同离线样本预算的 iid 分布对比。

## 3. 同代码版本下的纯随机系列

三组配置、训练 git=64f2608、模型/loss/归一化/超参相同；只读存档。triple_dynamic 结果为：

| MAE 排名 | 数据集简称 | relL2↓ | MAE↓ | MSE↓ | pooled Corr↑ |
|---:|---|---:|---:|---:|---:|
| 1 | 原始 pure-random | 0.119775 | 0.056734 | 1.002128 | 0.982488 |
| 2 | pure-random-rescue | 0.137694 | 0.062521 | 1.231576 | 0.976900 |
| 3 | pure-random-KO-interval | 0.138310 | 0.067408 | 1.110944 | 0.976727 |

原始 pure-random 在这组 **四项指标都优于 rescue 和 KO-interval**。
但 hybrid_mse_rel 中 rescue 的 MAE/MSE 更低、relL2/Corr 没有同步变好。
因此不支持“救援必然优于原始随机”或“KO 加连续扰动必然更好”。

Pure-random Twin-B 相对 Twin-A：
MAE 0.082969 vs 0.089117、MSE 2.084944 vs 2.388915、relL2 0.158334 vs 0.156859、
Corr 0.969458 vs 0.968697。
B 的 MAE/MSE 更优，但 relL2 略差，不能说全部指标均获益。
这一对修改的是一组目标模式的分布/赋值包，不是只改一个数值参数，也不证明当前幅度区间全局最优。

### 新核查：现存 Twin 也不是严格只变目标模式

对当前同名 NPZ 参数逐行比较（没有读轨迹）：两对各100000条合成样本的名称、参数名及顺序均一致。

| 配对 | 参数完全相同的合成行 | 名称中不含9种目标模式的行 | 其中参数仍不同的行 |
|---|---:|---:|---:|
| pure_random Twin-A/B | 46752 | 29223 | **1898** |
| 旧 distribution Twin-A/B | 51073 | 39687 | **1894** |

这些非目标行的差异列为 `ksbud, ksori, ksspn, mdt`。
当前生成器在“共享非目标模式”分支分别调用两次 apply_gene_mutation，
但 ORI_BUD_SPN[perturb] / MDT[MDT] 内部会重新随机抽值，因此“共享alpha”并未保证整个override相同。
源码行为与该现象相符；由于缺生成快照，不宣称已还原每条历史行的全部来源。
所以不能把 A/B 指标差全部归因于那9种目标模式的分布。本轮按分析范围只报告，没有修生成器或重生成数据。

## 4. 采样器到底是否有“指定采样”

以下源码结论对应当前仓库；历史文件缺生成 manifest 时，不冒称精确还原其生成快照。

| 家族 | 基因/模式选择 | 幅度/保留机制 | 符合“无 motif/拓扑定向”吗 |
|---|---|---|---|
| 原始 pure-random | 初始27基因无放回均匀；各基因可用模式初始均匀；L1–L4 等配额 | 同基因25%复合、冲突重抽；模式相关幅度；real 碰撞拒绝 | 初始骨架符合；不是无条件 iid 或无 real 先验 |
| pure-random-rescue | 同上 | 与 real 碰撞后，70%连续模式切换、30%碳源漂移或回退 | 无预设 motif，但有 real 条件改样 |
| KO-interval | 同上；当前 mirror 脚本保持对应样本 | 把 KO 目标参数改为绝对区间 U(0,1e-4)，不是统一相对 fold | 无 motif；是显式边界设计，不是已证实最优 |
| pure-random Twin A/B | 当前脚本使用均匀基因/初始模式；配额25/25/25/25 | 试图共享非目标模式但实际存在额外随机差异；目标模式按A直接赋值/B连续包络分叉 | 骨架无 motif，但幅度有指定包络/常数 |
| MASS-random | 预设 P_bio_G 加权无放回；模式按预设非均匀权重；L1/L2固定1/2，L3为3–5，L4为4–6 | 旧 MASS 幅度/等位基因规则；碳源、复合、real拒绝 | 无 rescue motif，但不是等概率随机 |
| MASS-random-dist | MASS-random 的加权骨架 | 仅幅度函数换 mutant_aware；mutation 应用仍用旧 MASS | 无 motif，不是完整替换新动力学干预规则 |
| MASS 标准版 | L3/L4 中有50%概率进入人工 rescue-motif 分支 | 旧模式与幅度规则 | 不符合 |
| mutant_aware | 当前含50%结构配额骨架+50%随机基因，模式class等配额；有汇聚/级联/星形等 | 模式相关连续包络、冲突/筛选 | 不符合 |
| BAMS | 拮抗/级联/汇聚等定向骨架 | 稳态响应反演/细化、拒绝与救援 | 不符合 |
| motif-topology | 连通二/三/四元组及master-switch五元组 | 预设模式偏好、筛选 | 不符合 |
| kDPP / curvature | 基因图核的排斥/多样性选择，curvature另有符号与fanin规则 | curvature下采样平直轨迹（默认保留概率0.2、阈值0.01） | 不符合 |
| Gibbs-Boltzmann | 按图距离/边/fanin构造能量的分类分布抽组合，并非此脚本中运行长MCMC链 | 模式相关幅度 | 不符合 |
| Gibbs-Sobol | 离散骨架仍用图能量Gibbs | 连续幅度用Sobol逆CDF | 不符合；Sobol不是“取消拓扑” |
| WB-MASS / S5mode | 图结构骨架、mode配置、覆盖账本接受 | 稳态代数坐标/效应反演和分箱定额 | 不符合；现有若干存档只有122 real |
| AL1M | 候选池与AL策略共同决定最终训练分布 | kmeans初始子集、主动选样、real-envelope/flux筛选 | 不作为纯 iid 离线采样对照 |

### 初始均匀不等于最终均匀

pure-random 的冲突处理调用 _sample_mode；当前默认 MODE_FREQ=class，按模式类别配额重抽，
不是再次调用主函数的均匀模式抽取。再加上拒绝、复合、质量/求解失败过滤，最终保留分布会偏离初始提议分布。

“避开不合法干预冲突”可以是必要机制约束，但必须如实记录其改变了最终分布；
本轮不修改这项设计。

### 原始 pure-random 的当前幅度分布

| 模式 | 当前分布摘要 | 注意 |
|---|---|---|
| KO | 目标敲除参数为0 | IC清零等机制规则另行应用，不是所有参数归零 |
| MC | 整数 n=2..30，P(n)正比于1/n | 以代码 arange(2,31) 为准，不沿用注释中的2..50 |
| GAL-OE | 每个基因指定范围内的对数均匀 | 许多项是绝对合成率/总量直接赋值，不都是乘WT fold |
| 通用ts | 0.2*Beta(0.5,2) | CDC14、TEM1/ESP1、NET1、CDC20/15/APC等有独立分支 |
| DAmP | Normal(0.2,0.05) 截断到[0.05,0.40] | 随后按基因规则缩放 |
| pLOF | U(0.10,0.50) | 按基因规则缩放 |
| db | 因基因而异，包含近零Uniform/指定区间 | 不是全局共享一个db分布 |
| 环境 | 含GAL-OE时mdt=150；否则通常90/150/160按75/18/7抽 | MDT干预及救援会覆盖常规规则 |

对数均匀表示对数坐标均匀，不等于原始浓度坐标均匀，定义见
[SciPy loguniform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html)。
Sobol 是低差异准随机序列，不等于独立伪随机抽样，见
[SciPy Sobol](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)。
这两条数学定义不证明任一分布更适合本任务。

### real 信息依赖的明确证据

- mutant_aware.sample_scaling_factor 的 TEM1/ESP1/NET1 ts 和 CLB2-db 注释/范围明确按 real anchor 包络设置。
- pure-random 的 is_duplicate_real_mutant 读取 real 参数规则决定拒绝/救援。
- MASS-random 的旧引擎还有固定等位基因赋值、预设不均匀模式概率；不能认证为无测试信息先验。
- 这些问题不因“概率为0，不直接抽中精确测试点”而消失，也不因文件名叫pure_random而消失。
- 按用户之前“不提前知道real具体参数”的约束，不能把这些anchor值提炼成新的推荐采样范围。
  本报告的建议只是保留无拓扑基因骨架作基线，幅度来源仍须独立于real参数规则清理。

## 5. 当前同名 NPZ 抽查

读取头部/样本名，并额外比较两对 Twin 参数；不读完整轨迹。抽查的9份当前文件均为100126×39×500，126 real+100000 synthetic，均无额外生成manifest字段。

- mass_100000_random...：L1/L2/L3/L4 = 25000/35000/25000/15000。
- 原始 pure-random / rescue / KO-interval：25000/25000/25000/25000。
- pure-random Twin-A/B：25000/25000/25000/25000。
- 旧 distribution Twin-A/B 与 mutant_aware_100000...：25000/35000/25000/15000。
- 名称的 Level 是生成分层标签，不一定等于冲突处理后的最终有效基因数。
- 缺少生成版本/内容哈希的历史训练记录，无法只凭同名路径证明当时使用的文件就是当前文件。

## 6. 最佳存档成绩：只作检索，不用于证明采样因果优势

每个一级数据集目录选126-real记录中最小MAE；模型、loss、norm、宽度/代码版本可能不同。
被搜索的组合越多越容易得到较低最小值，不是公平“分布排行榜”。
只搜索过一次与搜索过几十模型的目录，不能直接比较极小值作为科学结论。

| 排名 | 数据集目录 | 模型 / loss / norm | relL2↓ | MAE↓ | MSE↓ | pooled Corr↑ |
|---:|---|---|---:|---:|---:|---:|
| 1 | `mass_100000_dataset_210min_500steps_dual_labels` | biom_refiner_graph / hybrid_mse_rel / log1p | 0.057939 | 0.034662 | 0.284278 | 0.995881 |
| 2 | `mutant_aware_100000_210m_500s` | biom_refiner_causal_plus / triple_dynamic / log1p | 0.082199 | 0.043661 | 0.499283 | 0.991748 |
| 3 | `mass_50000_dataset_210min_500steps_dual_labels` | biom_refiner_curvature / triple_dynamic / log1p | 0.071218 | 0.047625 | 0.589747 | 0.994392 |
| 4 | `gibbs_boltzmann_100000_210m_500s` | biom_refiner_plus / triple_dynamic / log1p | 0.120973 | 0.053357 | 0.765689 | 0.981541 |
| 5 | `mutant_aware_pure_random_100000_210m_500s` | biom_refiner_plus / triple_dynamic / log1p | 0.119775 | 0.056734 | 1.002128 | 0.982488 |
| 6 | `gibbs_sobol_100000_210m_500s` | biom_refiner_latch_causal / hybrid_mse_rel / log1p | 0.121895 | 0.058117 | 0.968443 | 0.981047 |
| 7 | `kdpp_gated_100000_210m_500s` | biom_refiner_causal_plus / hybrid_mse_rel / log1p | 0.146161 | 0.058318 | 0.841948 | 0.971832 |
| 8 | `kdpp_curv_gated_100000_210m_500s` | biom_refiner_causal_plus / hybrid_mse_rel / log1p | 0.132728 | 0.058489 | 1.113823 | 0.977750 |
| 9 | `distribution_twin_b_continuous_100000_210m_500s` | biom_refiner_plus / hybrid_mse_rel / log1p | 0.105124 | 0.060714 | 1.861685 | 0.987266 |
| 10 | `distribution_twin_a_direct_100000_210m_500s` | biom_refiner_plus / hybrid_mse_rel / log1p | 0.114764 | 0.061351 | 1.425010 | 0.984237 |
| 11 | `mass_100000_random_dataset_210min_500steps_dual_labels` | biom_refiner_curvature / triple_dynamic / log1p | 0.122844 | 0.061438 | 0.787777 | 0.980640 |
| 12 | `mutant_aware_pure_random_rescue_100000_210m_500s` | biom_refiner_plus / triple_dynamic / log1p | 0.137694 | 0.062521 | 1.231576 | 0.976900 |
| 13 | `bams_100000_210m_500s` | biom_refiner_causal_plus / hybrid_mse_rel / log1p | 0.136286 | 0.067161 | 1.083938 | 0.975993 |
| 14 | `mutant_aware_pure_random_ko_interval_100000_210m_500s` | biom_refiner_plus / triple_dynamic / log1p | 0.138310 | 0.067408 | 1.110944 | 0.976727 |
| 15 | `bams_ablation_hard_50000_210m_500s` | biom_refiner_plus / hybrid_mse_rel / log1p | 0.119574 | 0.076336 | 1.377369 | 0.983733 |
| 16 | `pure_random_distribution_twin_b_continuous_100000_210m_500s` | biom_refiner_plus / hybrid_mse_rel / log1p | 0.158334 | 0.082969 | 2.084944 | 0.969458 |
| 17 | `bams_ablation_hypo_50000_210m_500s` | biom_refiner_plus / hybrid_mse_rel / log1p | 0.125657 | 0.085686 | 2.542206 | 0.983487 |
| 18 | `mass_random_dist_100000_210m_500s` | biom_refiner_plus / hybrid_mse_rel / log1p | 0.180318 | 0.086977 | 1.554738 | 0.956659 |
| 19 | `1mass_random_dist_100000_210m_500s` | biom_refiner_plus / hybrid_mse_rel / log1p | 0.181145 | 0.088908 | 2.168105 | 0.950447 |
| 20 | `pure_random_distribution_twin_a_direct_100000_210m_500s` | biom_refiner_plus / hybrid_mse_rel / log1p | 0.156859 | 0.089117 | 2.388915 | 0.968697 |
| 21 | `bams_50000_210m_500s` | biom_refiner_plus / hybrid_mse_rel / log1p | 0.170434 | 0.091945 | 1.404750 | 0.963028 |
| 22 | `bams_motif_topology_100000_210m_500s` | biom_refiner_plus / triple_dynamic / log1p | 0.192575 | 0.113574 | 15.287204 | 0.961895 |
| 23 | `mutant_aware_100000_210m_500s旧版本` | hano_dual / hybrid_mse_rel / log1p | 0.215571 | 0.119607 | 2.977331 | 0.943166 |
| 24 | `mutant_aware_100000_leaklevel_210m_500s` | biom_refiner_plus / hybrid_mse_rel / log1p | 3303.554199 | 71.901497 | 2.1770e+8 | 0.901338 |
| 25 | `mutant_aware_100000_210m_500s_rolemode` | biom_refiner_plus / hybrid_mse_rel / log1p | 4361.755371 | 139.503448 | 4.5201e+8 | 0.923464 |

[完整最佳存档CSV](../scripts/analysis/outputs/sampling_distribution_audit_2026-10-20/dataset_best_mae.csv)。

## 7. 推荐的下一步，而不是未经验证的“最优范围”

1. 更严格的初始骨架可从 K均匀取1..4、27基因无放回均匀、可用模式均匀开始；它是透明基线，不是已证实最优。
2. 不依据real参数/轨迹制定新包络，不给已知坏例的离散组合定向加密。
3. 原始硬KO优先保留为基线；现有triple_dynamic三组比较不支持无条件替换为KO-interval/rescue。
4. 若允许非均匀基因/模式先验，把MASS-random列为另一个候选，不把它命名为等概率随机。
5. 固定当前代码、分组隔离的合成拆分、train-only拟合、训练预算与loss，先比较少数候选再用多个seed复核。
   不把“换loss后的收益”算作“换采样分布的收益”。
6. 可以按用户允许的real开发指标比较开发方案，但不能由此声称新的独立盲测结果；
   若有独立的最终评估集，应与开发选择区分。
7. 旧存档无法唯一确定最优K配比、基因权重、模式权重、连续幅度边界及其联合分布；
   更不能从一个seed反推出唯一最佳概率密度。

## 8. 同配置表的逐条来源

- **mutant_aware**：`mutant_aware_100000_210m_500s`；[metrics](../experiments/mutant_aware_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/mutant_aware_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=e79ca36。
- **kDPP-curvature**：`kdpp_curv_gated_100000_210m_500s`；[metrics](../experiments/kdpp_curv_gated_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/kdpp_curv_gated_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=e79ca36。
- **旧 Twin-B（骨架待追溯）**：`distribution_twin_b_continuous_100000_210m_500s`；[metrics](../experiments/distribution_twin_b_continuous_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/distribution_twin_b_continuous_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=eb4bdc8。
- **旧 Twin-A（骨架待追溯）**：`distribution_twin_a_direct_100000_210m_500s`；[metrics](../experiments/distribution_twin_a_direct_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/distribution_twin_a_direct_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=eb4bdc8。
- **kDPP**：`kdpp_gated_100000_210m_500s`；[metrics](../experiments/kdpp_gated_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/kdpp_gated_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=e79ca36。
- **MASS-random**：`mass_100000_random_dataset_210min_500steps_dual_labels`；[metrics](../experiments/mass_100000_random_dataset_210min_500steps_dual_labels/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/mass_100000_random_dataset_210min_500steps_dual_labels/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=f0375fc。
- **Gibbs-Sobol**：`gibbs_sobol_100000_210m_500s`；[metrics](../experiments/gibbs_sobol_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/gibbs_sobol_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=0d3fe85。
- **pure-random-rescue**：`mutant_aware_pure_random_rescue_100000_210m_500s`；[metrics](../experiments/mutant_aware_pure_random_rescue_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/mutant_aware_pure_random_rescue_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=64f2608。
- **BAMS**：`bams_100000_210m_500s`；[metrics](../experiments/bams_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/bams_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=e79ca36。
- **pure-random-KO-interval**：`mutant_aware_pure_random_ko_interval_100000_210m_500s`；[metrics](../experiments/mutant_aware_pure_random_ko_interval_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/mutant_aware_pure_random_ko_interval_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=64f2608。
- **pure-random Twin-B**：`pure_random_distribution_twin_b_continuous_100000_210m_500s`；[metrics](../experiments/pure_random_distribution_twin_b_continuous_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/pure_random_distribution_twin_b_continuous_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=eb4bdc8。
- **原始 pure-random**：`mutant_aware_pure_random_100000_210m_500s`；[metrics](../experiments/mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=64f2608。
- **MASS-random-dist**：`mass_random_dist_100000_210m_500s`；[metrics](../experiments/mass_random_dist_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/mass_random_dist_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=5027906。
- **pure-random Twin-A**：`pure_random_distribution_twin_a_direct_100000_210m_500s`；[metrics](../experiments/pure_random_distribution_twin_a_direct_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/pure_random_distribution_twin_a_direct_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=eb4bdc8。
- **motif-topology**：`bams_motif_topology_100000_210m_500s`；[metrics](../experiments/bams_motif_topology_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/global_metrics_denorm.json)；[config](../experiments/bams_motif_topology_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/config.json)；git=64f2608。

## 9. 主要源码入口

- [pure-random](../data/generate_pure_random_dataset.py)
- [幅度、冲突重抽、结构骨架](../data/mutant_aware.py)
- [MASS-random](../data/mass_random_sampler.py)
- [MASS-random-dist](../data/mass_random_mutant_dist.py)
- [旧MASS与rescue模板](../data/mass_dataset_generator.py)
- [当前纯随机Twin生成器](../data/generate_distribution_ablation_twins.py)
- [KO镜像生成器](../data/mirror_pure_random_ko_interval.py)
- [Gibbs图能量](../data/gibbs_boltzmann_sampler.py)
- [Gibbs-Sobol](../data/gibbs_sobol_sampler.py)
- [kDPP](../data/generate_kdpp_gated_dataset.py)
- [signed kDPP与曲率门控](../data/generate_kdpp_curv_gated_dataset.py)
- [BAMS](../data/bams_dataset_generator.py)
- [motif-topology](../data/generate_motif_topology_dataset.py)
- [WB-MASS](../data/mass_wt_blind_sampler.py)
