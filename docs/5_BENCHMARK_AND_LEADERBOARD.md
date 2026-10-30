> **2026-10-30 22:06 后台已安排**：11项optimized批次已挂到tmux `al_budding_optimized_20261030_220508`；当前GPU0–3被其他用户占用，队列等待空闲，尚未开始正式训练。115项专项回归及真实FNO选样→256条ODE→入库/重载验证通过。[记录](AL_OPTIMIZED_RESTART_2026-10-30.md)。

> **2026-10-30 优化批次准备/启动记录**：旧8项完整结果仍按原表保留；六项中断实验将从round0独立重跑，另有四新ours及修正event。统一full预算，不能将新方法选样验收当作预测提升。[协议与新目录](AL_OPTIMIZED_RESTART_2026-10-30.md)。

> **2026-10-30 ours完整诊断**：目前8项完成、6项中断。interaction显著改善#15/#22/#50，但#62/#124退步；CPU重载显示物理MSE约93%–97%来自ORI。对照预算后期回退、跨轮历史误差失配和过窄MC×KO动作均有审计证据。优先验证全局覆盖＋局部响应，再独立加入合成/降解联合响应与反馈校准。[结果、波形和方案](AL_OURS_DIAGNOSIS_2026-10-30.md)。仅分析，未启动新实验。

> **2026-10-30 存储格式更新**：新AL改为共享初始数据＋新增块＋`rounds/round_XX/dataset.json`，不再重复写累计轨迹。训练/评测/相关分析支持新旧格式，原始轨迹精度不变；本批14项已完成逐字段验证转换，173.13→56.65GiB（含共享初始），释放116.48GiB；首末轮真实checkpoint重载通过。[布局、读取和验证](AL_DATASET_STORAGE_LAYOUT.md)。下方空间数字为转换前审计快照。

> **2026-10-30 17:14更新**：event完成，最终MAE/MSE/RelL2/Corr=0.115966/2.217988/0.177114/0.960494；本批8完成/6失败。最终200epoch训练变化率0.052%–0.107%/epoch，未连续三次通过当前收敛判据。本批目录173.13GiB，尚未清理。[收敛、每项空间及无损优化方案](AL_STORAGE_AND_CONVERGENCE_2026-10-30.md)。

> **2026-10-30 16:56长程7项完整结果**：MaxDist MAE/RelL2=0.097129/0.134503，Random MSE=1.374911；interaction MAE=0.103874，feedback RelL2=0.137153。ours改善部分难例但尚未赢得整体比较，interaction的总MAE/MSE劣势受#124显著影响。event未完成、6失败，不混合旧预算补榜。[完整表格与限制](AL_FULL_RESULTS_2026-10-30.md)。

> **2026-10-30 10:48长程阶段结果**：14项中2完成、6运行、6异常。70k最终Random/MaxDist的MAE为0.098193/0.097129、MSE为1.374911/1.519672、整体RelL2为0.169334/0.134503。单seed，ours尚无完整最终结果；不能混合40k/70k排名。逐样本权衡、历史预算比较和loss变化率见[阶段审计](AL_FULL_PROGRESS_2026-10-30.md)。

> **2026-10-30 正式长程批次已启动**：默认切换为research全部14项FNO＋full档（100/100/100/200 epochs、warmup5/5/5/10），GPU0–3队列4/4/3/3，每卡最多2项。保留screen作为显式选项。tmux后台与结果位置见[本批启动记录](AL_FULL_LAUNCH_2026-10-30.md)。此条表示已启动，不表示实验已完成。

> **2026-10-30 最新预算档位**：默认screen采用50/50/50/100 epochs和warmup4/4/4/8枚举方法；显式TRAINING_PROFILE=full采用100/100/100/200和5/5/5/10复核少数候选。两档均保留峰值LR1e-3、最低LR1e-5、FNO192/4/24、batch512和loss变化率监控，不自动早停，不保证充分收敛。尚未正式训练。[设置、目录与验证](AL_TRAINING_BUDGET_2026-10-30.md)。下方较早配置与结果保留为历史记录。

> 2026-10-30 loss变化率补查：ours末10轮训练loss平均下降0.203%/epoch、验证MSE下降0.098%/epoch；七项训练loss仍平均下降0.185%–0.228%/epoch。下降减缓，不等于已满足严格收敛标准。保持超参，未加训。[定义与统计](AL_CONVERGENCE_AND_NEXT_IDEAS_2026-10-29.md)。

> 2026-10-30 研究变体准备完成：保持训练超参，新增7个独立采集方法，与6个baseline/首版ours统一比较。全池输入几何核验完成，但尚未正式训练，不能声称精度提升。[策略枚举与验证记录](AL_QUERY_VARIANTS_2026-10-30.md)。

> 2026-10-29 收敛补查：核对七方法四轮日志/CSV，最终100 epoch接近当前schedule的末段平台，尚不能证明充分收敛；没有新增模型结果。详细统计与后续设计见[收敛与idea](AL_CONVERGENCE_AND_NEXT_IDEAS_2026-10-29.md)。

> 2026-10-29 完整七方法结果：全部完成四轮。Power最终MAE=0.105360，LCMD Rel-L2=0.146436；ours MAE=0.136249/Rel-L2=0.165282，首版未显示优势。已完成配对动作、20个badcase、CPU ensemble重放及218伙伴hub缓存轨迹诊断。[完整结果与优化建议](AL_FINAL_COMPARISON_AND_RESPONSE_DIAGNOSIS_2026-10-29.md)。

> 2026-10-29 15:33：Random、Uncertainty、ours_response已按修复后的代码后台重跑，旧失败记录整体保留。MaxDist/LCMD/BAIT/Power已完成，用时约3:36/3:39/4:20/4:22。[本次重跑与用时记录](AL_GENE_RANGES_RESTART_2026-10-29.md)。

> 2026-10-29 AL结果审计：新 gene_ranges 同协议 Random 在40k训练时，real MAE 0.213509→0.180065、MSE 6.099851→3.300605；已完成最终70k的Power MAE=0.105360、LCMD Rel-L2=0.146436。Random/Uncertainty停滞，ours保存审计失败；详见 [逐轮、20个badcase与修复记录](AL_RANGE_RESULTS_AUDIT_2026-10-29.md)。

> **2026-10-29 新正式输入已生成并审计**：百万`gene_ranges` pool与初始10000/1250/1250＋126 real已就绪。旧/新初始train大值比例19.55%→11.79%（相对减少39.69%），未按幅值删样本；新集总量上界检查通过。GAL-MC池中13703条、单模块容量8条。启动脚本默认新批次路径，W&B offline，正式训练尚未启动。文件、比较与跳过准备的启动命令见[数据准备结果](GENE_RANGES_DATA_PREPARATION_2026-10-29.md)。

> **2026-10-29 已实施获批范围及GAL-MC**：CLN2/CLB5/CDC14/TEM1/CDC15的MC改2～10，其余按逐基因表；普通CLB2-GAL收窄并新增独立GAL-MC。三类获批残余范围已更新，未新增零点混采或调整SWI5。94项回归、256条参数池和8条GAL-MC轨迹导出验收通过；49组ODE对照说明小残余不普遍等价KO。旧正式输入未覆盖，新实验需重建pool/初始集。详见[实施与机制对照](APPROVED_BUDDING_RANGES_AND_KO_CHECK_2026-10-29.md)。

> **2026-10-29 采样域诊断补充**：26组同背景强度/残余扫描完成，全部成功；#22的CLB5 2→4拷贝出现ABC变化，#124的APC近零扫描则保持平滑。仅为ODE响应检查，无新模型指标。范围提议、证据边界与复现产物见[候选方案](BUDDING_SAMPLING_RANGE_PROPOSAL_2026-10-29.md)。

> **2026-10-28 强度与组合归因补充**：百万pool和Random最终70k的同基因/同模式参数全量核对、17组诊断ODE复算见[强度审计](MUTATION_REAL_RANGE_AUDIT_2026-10-28.md)。MC对部分基因明显宽于real；同拷贝数仍可因背景产生极大输出。另确认8条TEM1数值越界。没有重训、改写正式数据或新增模型精度结论。

# 全量模型基准评测榜单与坏 Case 深度剖析 (Benchmark Leaderboard & Badcase Analysis)

> **2026-10-28 AL用时核对**：16项计划中8项有启动记录，4项完成、4项中断未完成。完整墙钟用时：FNO Random约4小时51分、Uncertainty约6小时54分、Power约6小时16分，DeepONet Random约4小时06分；实际epoch100/100/100/200，含选样/在线仿真/评测保存，不含预生成与排队。FNO BAIT的10小时38分为中断前运行窗口，不能列作完成用时。来源与精确计时表见 [候选池与用时](AL_POOL_AND_RUNTIME_2026-10-28.md)。本轮未训练或改写结果。

> **2026-10-28 采样概率审计**：恢复目录的142组强度/分位核对、12策略候选及2万唯一池统计见 [审计报告](BUDDING_MUTATION_PROBABILITY_AUDIT_2026-10-28.md)。发现BAMS非法模式和若干静态冲突规则差异，未修生产实现。本轮未积分ODE或训练，不更新模型精度榜；2026-10-27坏例报告和现有百万池对应恢复之前的目录，不能套用为新分布实测结果。

> **2026-10-27 好例反证补充**：同一FNO Random最终70k中，#8/#10严格局部参数盒计数均0，RelL2仍仅0.092/0.078；#7/#9则有极近输入支撑。#15的训练集中已存在完整39变量轨迹RelL2差0.000297的阻滞样本。8组固定背景ODE与冻结FNO对照显示：双KO下CLN2从1→5拷贝，真实CLN2峰值仍0.065，预测却0.725→14.107。因而“类似样本少→预测差”不是充分解释，需检查模型是否学会条件参数依赖。16组诊断、邻居拟合误差、完整计数及限制见 [好例/坏例对照](GOOD_BAD_CASE_COMPARISON_2026-10-27.md)。未重训、未改采样规则，6项分析回归通过。

> **2026-10-27 跨数据集 badcase 审计**：统计148份静态CSV、27份当前AL逐轮JSON及另列的44份旧AL CSV；深度核对26个静态数据文件与8个AL最新轮次，32/34个checkpoint可复现。20例逐样本指标、局部训练支撑、通道误差、固定背景ODE扫描及当前100万池供给见 [完整报告](BADCASE_SAMPLING_AUDIT_2026-10-27.md)。#15的主要现象是假START，而非真实CLN2必然很大；#124在24份静态复现中六分类均正确但ORI主导SSE。旧MASS的#88–91初值差异、3份旧实验小数real ID漏识别及2份未复现项已隔离。BAMS-hard属于test-informed定向消融，不能作为盲测AL收益。当前只有4个AL任务完成四轮，不能把不同预算直接排名。本轮未训练或修改生产规则。

> **2026-10-22 AL 工程流程验收**：Budding/Fission 新四轮闭环及逐轮重归一化已实施，357 项回归通过，并完成两种系统的小规模真实仿真、四轮训练及八个 checkpoint 重载。初始生成验证使用完整原生 real 基准；四轮训练验收使用每系统 2 株 real 子集和小型 MLP，不作为正式模型精度结果，不更新排行榜。配置、目录与验证范围见 [AL 实验流程](AL_EXPERIMENT_WORKFLOW.md)。

> **2026-10-22 log1p尺度诊断**：对5000条当前分布合成轨迹比较保留/排除高幅值后的标准化尺度，并核对hybrid_mse_rel的两项均在标准化空间计算。该诊断不包含模型训练，不代表real预测精度变化，见 [log1p诊断](../scripts/analysis/outputs/log1p_extremes_2026-10-22_hybrid/REPORT.md)。

> **2026-10-22 分布内大值标定**：按当前两套 Budding 分布生成 5000 条随机样本，另测 290 组合法分位数组合；不在生产代码中新增/移除幅值门控。5290 条均正常完成，随机样本中 576 条超过两倍 real 的某个非 MASS 变量最大值。大值案例经加严容差复算仍存在；未做代理模型训练，不能推断 real 预测提升或下降。参数、原始轨迹、耗时、负值明细和比较表见 [分布内大值报告](../scripts/analysis/outputs/budding_extremes_2026-10-22/REPORT.md)。

> **2026-10-21 baseline 清理**：下方旧实现的指标只保留为历史记录，不代表新标准 baseline 的性能。
> 当前标准入口和来源见 [基线清单](BASELINE_SELECTION_AUDIT.md)；本次仅有数值/接口回归验证，没有重跑论文比较。


> **2026-10-21 汇总兼容修复**：按seed规范化all_variables/all_16_vars/all_39_vars别名后再汇总，
> 防止漏掉旧格式seed；每项指标记录实际有效n_seeds，并拒绝混合16/39变量系统或冲突的别名。
> 本次未改写历史实验指标或seed_summary；现有单seed Fission榜单不受影响。


> **2026-10-21 Fission 50k实测审查**：同seed42、100epoch，FNO1d/BioRefinerPlus的real MAE为0.031449/0.026654，
> pooled relL2为0.137331/0.136807；只含一个seed，wee1ts cdc25D仍为主要坏例。
> 实际有效synthetic为49186条，另有20 real；数据/指标口径及代码问题见 [目录与实验审查](DIRECTORY_REVIEW_2026-10-21.md)。


> **2026-10-21 Fission流程复核**：实际跑通W&B离线训练、checkpoint恢复、20 real/8 synth评测与一轮受控HPO。
> 采用复制合成轨迹的小规模冒烟数据，仅验证接口，不更新模型精度榜单。
> 原Fission标签保持不变，详细范围见 [Fission实验流程](FISSION_EXTENSION.md)。


> **2026-10-21 中性标签补充**：未来评测另存 `per_mutant_label_groups.csv` 与 `per_synth_test_label_groups.csv`，
> 将BIO、ABC、六分类和原指标逐行对齐；原指标CSV格式、指标定义、历史文件和排行榜不变。
> 本次没有训练六分类辅助头或实施分组补采，不能据此宣称消除了badcase。


> **AL跨策略标签诊断补充**：完成第2轮的7种策略中，#15/#124均为7/7策略的Top20，#22/#42为6/7。
> 同时留出算法及mutant的风险预测检验：六分类对ABC的MAE/MSE名次预测RMSE改善5.43%/9.16%，
> relL2名次预测恶化1.60%；这些是**标签诊断指标，不是新模型精度提升**。
> 保留BIO更换pattern，综合风险预测改善0.66%，条件区间跨0；目前不能宣称全面胜出。
> 逐策略坏例、预算差异、指纹与验证边界见 [完整标签比较](AL_PATTERN_LABEL_COMPARISON.md)。

> **2026-10-20 Pattern/Badcase核验**：本轮按用户指定的BioRefinerPlus / hybrid_mse_rel / seed_42
> 的`eval/per_mutant_metrics.csv`，用MAE、MSE、Relative_L2的平均名次挑选20个badcase。
> #15的Relative_L2=7.53625，#124的MSE=557.40076；不能混入triple_dynamic的指标。
> 新临界标记只命中这20条中的1条，#15/#22/#42/#124均未命中；六分类完备不等于机制已验证。
> 真实条件长时积分、完整名单及逐变量诊断见 [Pattern核验与统计入口](BUDDING_PATTERN_LABEL_AUDIT_2026-10-20.md)。
> 本轮未训练新模型、未修改旧指标或排行榜。

> **2026-10-20接口验证，不纳入精度榜单**：小规模Fission数据完成CLI单epoch训练、checkpoint恢复、
> 20 real评测和16变量绘图，文件位于`scratch/probes/fission_cli_train`与`fission_cli_eval`。
> 这仅验证系统接入，不能与下述充分训练的Budding精度结果比较。见 [扩展与验证范围](FISSION_EXTENSION.md)。

本文档汇集了在统一基准数据集 `mutant_aware_pure_random_100000_210m_500s` 上，覆盖全部神经算子模型对 **126 个真实酵母突变体金标准**（Chen et al., 2004）的最终评测表现、关键细胞周期核心蛋白精度拆解以及极端坏 Case 的生化机理归因。

> **2026-10-19 corr 口径再核查（只读历史存档，未重新推理）**：
> BioRefinerPlus / hybrid_mse_rel / log1p / seed_42 同一运行的 `eval/global_metrics_denorm.json`
> 中 real pooled corr 为 **0.972276**，`eval/per_mutant_metrics.csv` 的 126 行 Correlation
> 算术平均为 **0.846339**（中位数 0.890521）。这两者不是同一统计量，不能混作平均逐样本质量。
> 单条 corr 还含恒定通道返回 0 的约定，不能仅据低 corr 宣称波形错误。
> 本轮不改旧指标文件、指标定义或排名，完整路径及核验边界见
> [RNG 修复与 corr 风险说明](PURE_RANDOM_RNG_FIX_2026-10-19.md)。

> **2026-10-18 复核说明**：下方历史表含 pooled 与逐 mutant 平均口径混用，部分
> loss/MSE/模型对应关系不能由现存结果复现；请勿直接据此认定模型排名或修复效果。
> 已核实结果与原因分析见 [当前 bad-case 证据审计](PURE_RANDOM_BADCASE_AUDIT_2026-10-18.md)。
> 例如同一 triple_dynamic 运行的 pooled corr 为 0.98249，逐 mutant 平均 corr 为
> 0.86051；MSE 为 1.00213。新增 Attractor 的初值布局缺陷也已通过标识值探针复现。
> 新增全 126 样本联合诊断：固定四指标规则标记 22 个，其中 16 个由 ORI 主导平方误差，
> 3 个仅 corr 触发；另有 18 个样本的预测事件数量与 ODE 记录不符。
> 合成测试有 1,604 条完整输入/标签与 train 重合，real 的 78/112/114 也有完全重合，
> 因而下方“完全未见过”与“独立泛化”的历史表述不成立。
> 本次只运行冻结模型重放与数值负对照，没有新的训练改进结果。
> 进一步核对历史 checkpoint：自动选权监控的是合成 `val_loss_lhs`；
> DTW 时域取样止于约 202 min，15 号约 58.3% 的平方误差在其后。
> 这是损失分项的监督范围诊断，不是新模型结果，也不表示整个 loss 忽略末端。
> 两个主力 BioRefinerPlus checkpoint 的图偏置均确认为单位矩阵，未保存预期跨变量
> 生化图连接；不能将这些历史结果当成“动态多关系通量图已验证有效”的证据。

> **同日后续实现修复**：已修正 IC 读取与 WT 差分特征，完成相关回归测试及旧权重
> 严格加载冒烟检查；**没有运行训练或新的 real-mutant 指标评测**，不更新排行榜。
> 旧权重可加载，但修复改变了前向语义，不能宣称直接复现旧输出或已获泛化提升。
> 细节见 [IC/WT 修复说明](IC_WT_CONTRACT_FIX_2026-10-18.md)。

---

## 一、真实生物突变体天梯榜 (Real Mutants Leaderboard)

评测协议：统一采用训练集统计量反归一化至真实物理浓度空间，计算 126 个真实生物突变体在 39 个变量、500 个时间步（210 分钟）上的平均相对 $L_2$ 误差、Pearson 相关系数、平均绝对误差（MAE）及 5 大核心细胞周期蛋白（Clb2, Clb5, Sic1, Cdh1, Cdc20）的子集指标：

| 排名 | 模型架构 (Model) | 损失函数配置 (Loss) | 突变体相对 $L_2$ $\downarrow$ | Pearson 相关系数 $\uparrow$ | 突变体 MAE $\downarrow$ | 突变体 MSE $\downarrow$ | 核心蛋白 Rel $L_2$ $\downarrow$ | 架构流派 / 关键技术 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 👑 **历史最优** | **`biom_refiner_plus` (Ours)** | **`triple_dynamic`** | **0.0741** | **0.8605** | **0.0567** | **1.0021** | **0.0812** | **全时期综合总榜第一 (All-time SOTA)** |
| 🥇 **1** | **`biom_refiner_attractor_plus`** | `hybrid_mse_rel` | **0.0825** | 0.8510 | 0.0612 | 1.1205 | 0.0895 | 吸引子相空间锁定 + 因果拓扑增强 |
| 🥈 **2** | **`biom_refiner_causal_reach`** | `hybrid_mse_rel` | **0.0894** | 0.8432 | 0.0645 | 1.1830 | 0.0942 | 因果单向可达性门控网络 |
| 🥉 **3** | **`biom_refiner_plus` (Ours)** | `hybrid_mse_rel` | **0.1198** | **0.9825** | 0.0720 | 0.0631 | 0.1367 | 标准混合损失基线 |
| 4 | **`bregman_wno`** | `hybrid_mse_rel` | 0.1310 | 0.9821 | 0.0912 | 0.0782 | 0.1604 | 布雷格曼小波算子 |
| 5 | **`biom_refiner_event_hybrid`** | `hybrid_mse_rel` | 0.1351 | 0.9785 | 0.0708 | 0.0811 | 0.1811 | 事件跳跃图混合算子 |
| 6 | **`konode`** | `hybrid_mse_rel` | 0.1526 | 0.9737 | 0.1123 | 0.0942 | 0.1899 | 库普曼神经常微分方程 |
| 7 | **`icon`** | `hybrid_mse_rel` | 0.1528 | 0.9743 | 0.1188 | 0.0950 | 0.1960 | 算子连续学习网络 |
| 8 | **`unet1d`** | `hybrid_mse_rel` | 0.1549 | 0.9726 | 0.0964 | 0.0981 | 0.2061 | 经典多尺度时序 U-Net |
| 9 | **`dpot1d`** | `hybrid_mse_rel` | 0.1559 | 0.9732 | 0.1072 | 0.1012 | 0.2207 | 去噪预训练算子 Transformer |
| 10 | **`mwt1d`** | `hybrid_mse_rel` | 0.1573 | 0.9721 | 0.1096 | 0.1040 | 0.1903 | 多小波变换神经算子 |
| 11 | **`gnot`** | `hybrid_mse_rel` | 0.1586 | 0.9727 | 0.1130 | 0.1065 | 0.2007 | 通用神经算子 Transformer |
| 12 | **`cno1d`** | `hybrid_mse_rel` | 0.1621 | 0.9699 | 0.1235 | 0.1102 | 0.2371 | 连续神经算子 |
| 13 | **`transolver1d`** | `hybrid_mse_rel` | 0.1647 | 0.9691 | 0.1070 | 0.1135 | 0.2384 | 物理几何解算 Transformer |
| 14 | **`bregman_fno`** | `hybrid_mse_rel` | 0.1650 | 0.9685 | 0.1025 | 0.1140 | 0.2275 | 布雷格曼傅里叶算子 |
| 15 | **`transolver3_1d`** | `hybrid_mse_rel` | 0.1655 | 0.9694 | 0.1247 | 0.1148 | 0.2232 | 进阶物理解算器 |
| 16 | **`gfno`** | `hybrid_mse_rel` | 0.1680 | 0.9668 | 0.1134 | 0.1189 | 0.2164 | 群等变傅里叶神经算子 |
| 17 | **`fno1d`** | `hybrid_mse_rel` | 0.1697 | 0.9671 | 0.1180 | 0.1205 | 0.2169 | 标准一维傅里叶算子 |
| 18 | **`iganet_transformer`** | `hybrid_mse_rel` | 0.1704 | 0.9669 | 0.1250 | 0.1221 | 0.2439 | 等几何分析 Transformer |
| 19 | **`reformer`** | `hybrid_mse_rel` | 0.1714 | 0.9681 | 0.1374 | 0.1240 | 0.2025 | 局部敏感哈希 Transformer |
| 20 | **`galerkin_transformer1d`**| `hybrid_mse_rel` | 0.1767 | 0.9634 | 0.1254 | 0.1310 | 0.2622 | 伽辽金积分 Transformer |
| 21 | **`ncflow`** | `hybrid_mse_rel` | 0.1778 | 0.9629 | 0.1409 | 0.1332 | 0.2128 | 神经连续流模型 |
| 22 | **`vanilla_transformer`** | `hybrid_mse_rel` | 0.1781 | 0.9656 | 0.1409 | 0.1340 | 0.2193 | 标准时序 Transformer |
| 23 | **`sc_fno`** | `hybrid_mse_rel` | 0.1785 | 0.9627 | 0.1186 | 0.1345 | 0.2489 | 谱卷积傅里叶算子 |
| 24 | **`flowformer`** / `flashformer`| `hybrid_mse_rel` | 0.1820 | 0.9619 | 0.1298 | 0.1390 | 0.2487 | 线性流注意力 Transformer |
| 25 | **`mpno1d`** | `hybrid_mse_rel` | 0.1852 | 0.9613 | 0.1493 | 0.1430 | 0.2242 | 多项式积分神经算子 |
| 26 | **`resnet_film`** | `hybrid_mse_rel` | 0.1926 | 0.9575 | 0.1189 | 0.1512 | 0.2603 | 经典 FiLM 条件卷积残差网络 |
| 27 | **`deeponet`** | `hybrid_mse_rel` | 0.2072 | 0.9511 | 0.1896 | 0.1690 | 0.2533 | 经典 DeepONet (Branch+Trunk) |
| 28 | **`pyraformer`** | `hybrid_mse_rel` | 0.2087 | 0.9517 | 0.1775 | 0.1710 | 0.2530 | 金字塔注意力 Transformer |
| 29 | **`informer`** | `hybrid_mse_rel` | 0.2177 | 0.9479 | 0.1902 | 0.1825 | 0.2544 | 稀疏 ProbSparse Transformer |
| 30 | **`fedformer`** | `hybrid_mse_rel` | 0.2252 | 0.9420 | 0.1859 | 0.1910 | 0.2554 | 频域增强时序 Transformer |
| 31 | **`itransformer`** | `hybrid_mse_rel` | 0.2298 | 0.9367 | 0.2345 | 0.1980 | 0.2299 | 变量倒置 Transformer |
| 32 | **`neural_ode`** | `hybrid_mse_rel` | 0.2896 | 0.9069 | 0.2826 | 0.2540 | 0.3536 | 连续神经伴随 ODE |
| 33 | **`koopman_dmd`** | `hybrid_mse_rel` | 0.4855 | 0.8819 | 0.4794 | 0.4210 | 0.8748 | 动态模态分解 (线性算子) |
| 34 | **`dlinear` / `nlinear`** | `hybrid_mse_rel` | 0.5622 ~ 0.6586 | 0.6253 ~ 0.6404 | 0.6613 ~ 0.6904 | 0.55 ~ 0.68 | 0.6322 ~ 0.6447 | 线性时序投影基线 |

### 1.1 损失函数驱动的消融对比与“多任务梯度冲突”深层机理
在主力架构 `BioRefinerPlus` 下，对比不同损失函数配置在 126 真实突变体上的泛化表现：

| 损失函数配置 (`loss_type`) | 突变体 Rel $L_2 \downarrow$ | 突变体 MAE $\downarrow$ | Pearson Corr $\uparrow$ | 物理机理与梯度交互分析 |
|:---|:---:|:---:|:---:|:---|
| 👑 **`triple_dynamic`** | **0.0741** | **0.0567** | **0.8605** | $\text{MSE} + \text{RelL2} + \lambda \text{Deriv}$，导数项有效锁定了相变极限环的瞬态转折速度 |
| 🥇 **`hybrid_mse_rel`** | **0.1198** | 0.0720 | **0.9825** | 标准两阶段平衡，波形与绝对尺度兼顾 |
| 🥈 **`phase_derivative`** | 0.1280 | 0.0754 | 0.9760 | 单独一阶导数惩罚能够消除高频虚假振荡 |
| 🥉 **`mse_only`** | 0.1450 | 0.0821 | 0.9680 | 仅优化绝对均方误差，低丰度周期素（如 Cdc20）被高浓度分子掩盖 |
| 4. **`hybrid_all`** | 0.1820 | 0.0984 | 0.9540 | **负向退化**：同时叠加 DTW、PINN 残差与 Sobolev 梯度，产生严重的多任务梯度方向冲突（Gradient Conflict），需配合 `GradNorm` 动态平衡 |

---

## 二、合成测试集独立分布泛化天梯榜 (10,002 Unseen Synthetic Points)
为了解耦真实突变体上的 Sim-to-Real 外推误差与代理模型在流形内部的纯内插拟合能力，框架评测了在 10,002 个完全未见过的合成动力学点上的泛化精度：

| 排名 | 模型架构 (Model) | 合成集 Rel $L_2 \downarrow$ | 合成集 MAE $\downarrow$ | 合成集 MSE $\downarrow$ | 合成集 Corr $\uparrow$ | 内插拟合能力评语 |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 🥇 1 | **`biom_refiner_plus`** | **0.0215** | **0.0142** | **0.0031** | **0.9988** | 极限环与瞬态特征几近完美重构 |
| 🥈 2 | **`bregman_wno`** | 0.0284 | 0.0185 | 0.0048 | 0.9975 | 小波多尺度局域重构优异 |
| 🥉 3 | **`cno1d`** | 0.0351 | 0.0221 | 0.0062 | 0.9960 | 连续算子抗混叠能力强 |
| 4 | **`fno1d`** | 0.0512 | 0.0310 | 0.0098 | 0.9930 | 标准频域全局滤波，高频瞬态存在吉布斯振荡 |
| 5 | **`deeponet`** | 0.0845 | 0.0520 | 0.0210 | 0.9850 | Trunk-Branch 基函数表征能力受限 |
| 6 | **`neural_ode`** | 0.1850 | 0.1120 | 0.0650 | 0.9620 | 伴随状态积分误差累积 |
| 7 | **`koopman_dmd`** | 0.3820 | 0.2450 | 0.1820 | 0.9100 | 无法线性化非线性相变 |


---

## 二、核心评测指标公式 (Metric Formulations)

对突变体 $m \in \{1, \dots, 126\}$，真实时序 $Y_m \in \mathbb{R}^{V \times T}$，预测时序 $\hat{Y}_m \in \mathbb{R}^{V \times T}$（$V=39, T=500$）：

1. **真实物理相对 $L_2$ 误差 (Real Mutant Rel $L_2$)**：
   $$\text{Rel } L_2(m) = \frac{1}{V} \sum_{v=1}^{V} \frac{\| \hat{Y}_{m, v} - Y_{m, v} \|_2}{\| Y_{m, v} \|_2 + 10^{-4}}$$
2. **Pearson 相关系数 (Pearson Correlation)**：
   $$\text{Corr}(m) = \frac{1}{V} \sum_{v=1}^{V} \frac{\sum_{t} (\hat{Y}_{m, v, t} - \bar{\hat{Y}}_{m, v})(Y_{m, v, t} - \bar{Y}_{m, v})}{\sqrt{\sum_{t} (\hat{Y}_{m, v, t} - \bar{\hat{Y}}_{m, v})^2 \sum_{t} (Y_{m, v, t} - \bar{Y}_{m, v})^2}}$$
3. **核心蛋白子集指标 (Core Protein Sub-metrics)**：
   指定 $V_{\text{core}} = \{\text{Clb2}, \text{Clb5}, \text{Sic1}, \text{Cdh1}, \text{Cdc20}\}$，单独计算其在细胞周期调控开关处的分岔保真度。

---

## 三、极端坏 Case 生化机理解析与修复归因

> **历史解释待核正**：下列早期“120倍CLN2”“归一化修复90%”及表中机理/基因型不作为当前已验证结论。2026-10-27重新核对保存的参数、轨迹和checkpoint后，#15真实为低信号阻滞，且MC受SBF控制；不存在本轮证实的统一“极端值→分岔→坏例”因果链。当前逐例证据与限制以 [新审计报告](BADCASE_SAMPLING_AUDIT_2026-10-27.md) 为准。

在大规模神经算子评估中，我们系统识别并攻克了两类由于参数极端变异引发的物理分岔失效坏 Case：

### 1. 强启动子多拷贝过表达型坏 Case（如 15 号、44 号）
- **代表突变体**：
  - **15 号 (`cln3Δ bck2Δ multi-copy CLN2`)**：G1 细胞周期蛋白 Cln2 表达量激增至野生型的 120 倍以上。
  - **44 号 (`swi5Δ GAL-CLB2`)**：强半乳糖启动子驱动 M 期周期蛋白 Clb2 持续过量合成。
- **历史失效机理**：
  旧版线性归一化下，120 倍参数导致其 Z-score 达到 $+65 \sim +600$。注入主干 FiLM 层后，仿射激活因子 $(1 + \gamma(p))$ 发生剧烈饱和溢出，输出发散。
- **修复方案**：统一采用 $\text{asinh}(Z)$ 压缩，将 $+600$ 严格平滑压缩至 $+3.85$，误差下降超过 **90%**。

### 2. 高阶多重协同敲除型坏 Case（如 19 号、76 号）
- **代表突变体**：
  - **19 号 (`cln1Δ cln2Δ cln3Δ GAL-CLN3`)**：三重 G1 启动蛋白同时敲除并外源诱导。
  - **76 号 (`cdc20Δ pds1Δ clb5Δ`)**：后期促进复合物及抑制因子三重缺失。
- **历史失效机理**：
  对数截断方案引入 $\log_{10}(10^{-8})$，单个敲除参数产生 $-7.0$。当同时敲除 6 个关联生化反应参数时，参数输入向量的 $L_2$ 模长由正常的 $6.5$ 暴增到 $17.1 \sim 76.11$，强行将网络深层特征通道置零（Shutdown），引发不可逆的假分岔阻滞。
- **修复方案**：$\text{asinh}(Z)$ 将敲除点平滑限定在 $-1.80$，多重敲除总模长完全受控在 12.99 以内，保证了复杂多基因突变下的网络响应线性度。

### 3. Top-20 代表性坏 Case 基因型与数据根因全景审计表
基于 `scripts/analysis/dataset_diagnosis/generate_multimetric_top20_badcases.py` 对模型多维误差进行综合打分，沉淀出最具挑战性的 Top-20 突变体及其底层数据根因：

| 突变体 ID | 突变体基因型 (Genotype) | 表型模式 | 关键受损通道 | $k$-NN 几何距离 | 核心失效机理 / 数据瓶颈归因 |
|:---|:---|:---:|:---:|:---:|:---|
| **#121** | `cln1Δ cln2Δ cln3Δ apc-ts` | Pattern C | Cln2, Clb2, Cdh1 | 16.42 | P0: 四重极端突变，连续参数空间测度为 0，外推距离极远 |
| **#19** | `cln1Δ cln2Δ cln3Δ GAL-CLN3` | Pattern C | MASS, Cln2, Clb5 | 15.80 | P0+P1: 三重敲除 + 强半乳糖外源挽救，相变阈值极窄 |
| **#76** | `cdc20Δ pds1Δ clb5Δ` | Pattern C | Pds1, Esp1, Clb2 | 14.95 | P0: 纺锤体检验点失活拯救突变，合成数据中无匹配模体 |
| **#44** | `swi5Δ GAL-CLB2` | Pattern B/C | Clb2, Sic1, Cdh1 | 13.82 | P1: Clb2 启动子超强持续激活，峰值浓度高达 $18\sigma$ |
| **#15** | `cln3Δ bck2Δ multi-copy CLN2` | Pattern A | Cln2, Sic1 | 14.10 | P1: 双阻滞蛋白缺失伴随高拷贝补偿，初速度相移 |
| **#73** | `cdc20-ts` (非限制性温度) | Pattern A | Cdc20, Clb2, Pds1 | 12.30 | P2: 温度敏感残余活性边界效应，易触发假双稳态阻滞 |
| **#88** | `clb2-dbΔ sic1Δ` | Pattern C | Clb2, Sic1 | 13.65 | P2: 降解框缺失导致无法泛素化，形成不可逆死锁极限环 |
| **#102** | `GAL-SIC1 (0-100m) OFF` | Pattern B | Sic1, Clb2 | 12.90 | P2: 脉冲式半乳糖瞬态诱导，时间导数出现陡峭阶跃 |
| **#61** | `clb1Δ clb2Δ GAL-CLB1` | Pattern C | Clb2, Clb5 | 14.20 | P0: 主控周期素全敲除 + 异源旁系同源补偿 |
| **#94** | `net1-ts cdc14-ts` | Pattern C | RENT, Net1, Cdc14 | 15.10 | P0: 核仁释放网络双重敏感突变，相空间处于鞍点边缘 |
| **#33** | `cdh1Δ clb5Δ` | Pattern B | Cdh1, Clb5, Clb2 | 11.80 | P4: 退出有丝分裂双重补偿，局部超球体邻域稀疏 |
| **#52** | `sic1-ts cdh1-ts` | Pattern C | Sic1, Cdh1 | 13.40 | P2: 双制动阀同步减活，导致重置检查点跳变失灵 |
| **#81** | `esp1-ts pds1Δ` | Pattern C | Esp1, Pds1, Spn | 14.70 | P0: 分离酶复合体功能反向解偶联 |
| **#110** | `GAL-CDC20 cdc20Δ` | Pattern B | Cdc20, Clb2 | 12.10 | P1: 启动子置换导致周期性节律转变为稳态平台 |
| **#27** | `clb5Δ clb6Δ` | Pattern B | Clb5, Clb2, Ori | 11.20 | P4: S 期周期素双缺失，DNA 复制起点延迟触发 |
| **#49** | `mad2Δ bub2Δ` | Pattern B | BUB2, Clb2 | 12.60 | P0: 纺锤体组装检验点双重缺陷 |
| **#67** | `tem1-ts lte1Δ` | Pattern C | Tem1, Cdc15, Cdc14 | 15.30 | P0: MEN 途径双突变，有丝分裂退出不可逆锁定 |
| **#84** | `apc-ts clb2Δ` | Pattern C | APC, Clb2, Sic1 | 14.85 | P0: 泛素连接酶变异伴随主周期素缺失 |
| **#99** | `GAL-CLB5 cdh1Δ` | Pattern B | Clb5, Cdh1 | 13.20 | P1: S 期周期素超量表达并缺乏降解机器 |
| **#118** | `cln2Δ cdc28-ts` | Pattern A | Cln2, Clb2 | 14.50 | P2: 激酶催化活性受抑伴随周期素缺失 |

---

## 四、评测复现命令 (Reproduction CLI)

使用内置评估脚本直接加载指定实验目录进行 126 突变体全量评估与可视化：

```bash
# 评估指定模型在 126 个真实突变体上的完整指标
python eval.py \
  --exp_dir experiments/mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/triple_dynamic_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42

# 使用分析脚本横向对比实验 badcase
python scripts/analysis/analyze_badcases.py
```
# 2026-10-23 显存测量补充

Budding 进一步补测 batch1024：FNO192／DeepONet256／UNet32／UNet64 的 allocator 峰值分别为8.10／2.60／2.12／3.97 GiB，四项均完成两次合成训练步。原始数据见 `scripts/analysis/outputs/gpu_memory_2026-10-23_larger_batch/`。这只验证当前形状下的显存可行性，不是收敛和精度结论；正式脚本未改batch。

用户授权后在 RTX A6000（物理 GPU 5）进行十项资源探针：batch512、500步、BF16 mixed、hybrid_mse_rel、真实训练分支与两次 AdamW 更新。全部完成，无 OOM；这不是精度对照或完整 AL 实验。Budding FNO192 的 allocator 峰值为 4.10 GiB、Transformer192 为 10.69 GiB、BioRefinerPlus192 为 34.65 GiB；Fission FNO128 为 2.62 GiB、BioRefinerPlus128 为 17.96 GiB。完整表和限制见 [AL 实验流程](AL_EXPERIMENT_WORKFLOW.md)，原始结果在 `scripts/analysis/outputs/gpu_memory_2026-10-23/summary.json`。


## 2026-10-28：均衡百万池 FNO＋Random AL 完成

配置：FNO width192、modes24、4块、batch512，50/50/50/100 epoch；累计合成train10k→20k→40k→70k，seed42、逐轮冷启动与train-only重拟合。最终126 real：MAE **0.124689**、MSE **2.145700**、RelL2 **0.166301**、Correlation **0.967073**；中位逐株MAE0.06153、RelL2 0.10845。指标来自round_03/eval/metrics_denorm.json，重算通过；与历史不同模型/采样域配置不作直接因果排名。其余六个模型打分策略曾因dtype错误失败，不算完成实验。

数据极值、归一化、实际损失分层抽查、Top20和局部覆盖见[Random大值审计](AL_RANDOM_EXTREMES_AUDIT_2026-10-28.md)。本轮未执行过滤后重训或模型容量消融，未恢复阈值。
