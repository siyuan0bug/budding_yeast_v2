> 历史内容快照，不代表全部规则仍然有效；当前说明以docs根目录对应指南为准。

> **2026-11-18全部指标统一入口**：机械盘`experiments/budding_sampling_summary/seed42_a866aef5d988/`的[全指标报告](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling_summary/seed42_a866aef5d988/all_metrics.md)包含46项×real/同池/跨池共138组终评，全部39/核心9指标、轨迹统计、分裂误差及36格混淆矩阵；同目录all_metrics.csv/json保留原值、all_round_metrics.csv汇总438组逐轮评测、all_training_history.csv汇总逐epoch记录、experiment_index.csv给出48项完整目录。修复后两项OGAS仍进行中，不以监控值代替终评。

> **2026-11-18 46项完成结果重验收与OGAS修复重跑**：原16项＋扩展25项＋修复5项全部完成标记／126real CSV／5000同池test CSV／126图验收通过，逐项JSON为相同物理指标口径。mutation Power为MAE/MSE/Relative综合最好，HGGS补跑MAE/MSE较好但Relative高；旧OGAS仅LHS相对同流程Random有收益，不能当作修复后性能。完整表与公平边界见[新汇总](../analysis/baseline_status_20261118_ogas_alignment/report.md)。本次仅两池OGAS在tmux `budding_ogas_seed42_20261118`重新启动，结果根`experiments/budding_ogas_alignment/seed42_20261118/generation/`，沿用同一seed42协议，不覆盖旧结果。新实验尚未完成，不提前纳入排名。

> **2026-11-18 修复补跑输出盘点**：两池HGGS、两池Lev及LHS Physics-PRE-Power共5项均完成、退出0，126real CSV行数及物理MAE/MSE/Relative/Correlation有限检查通过；队列状态5完成/0失败，未触发资源守护。仅做完成盘点，不是新的多seed结论，也未以这些成绩调新baseline。逐项指标和路径见`analysis/sampling_repairs_20261118/completion_summary.json`。主对照扩为8项的讨论见[最新清单第9节](SAMPLING_BASELINE_SHORTLIST_2026-11-18.md#9-后续注释al4pde分类8项主对照与生成成本)。

> **2026-11-15 cap10回放已完成**：16项纯RelativeL2实验×5轮共80个主模型，240组train/real/合成test回放。训练保存状态27/80出现超限，real为2/80；最终16个模型的real均无超限，最终mutation训练均无超限，最终LHS训练5/8有零星超限（合计23个样本×变量检查项）。显著异常是mutation＋BAIT第二轮CDC14T：3544/4000训练样本触发，cap前/后均值6.51581/0.433437。历史没有逐步cap计数，不能还原全过程或据此单独归因最终badcase。160组历史test监控交叉核验最大差1.44e-8；RSS峰值4.741GiB，单GPU审计已退出。未修改生产损失、评测或checkpoint选择。[说明与评测口径建议](3_LOSSES_AND_PHYSICS_CONSTRAINTS.md#6-cap10-的实际-checkpoint-回放证据)；完整报告位于机械盘analysis/relative_l2_cap_audit_20261115/report.md。

> **2026-11-15 损失比较解释补充**：训练的逐样本／逐变量归一化 Relative L2、物理空间拼接样本的历史 `Relative L2`、物理空间逐完整轨迹的 `macro_trajectory_rel_l2` 是三个不同定义。相近的 HyperNOs 生物 ODE 与 GNOT 多输出路径支持逐样本／通道相对损失；不能仅凭历史 pooled 指标变差否定整个 Relative L2 训练方向。硬 cap10 在超限通道上产生零梯度，实际checkpoint检查见上方更新，对最终badcase的因果影响尚未证明。没有更改生产指标、损失或重跑实验。详见[定义、官方源码证据与后续建议](3_LOSSES_AND_PHYSICS_CONSTRAINTS.md#2026-11-15相近多输出任务的-relative-l2-源码核对)。

> **2026-11-14 职责解耦验证，非正式性能结果**：训练、统计、数据读取、指标与报告均改为公共服务；旧在线/Fission兼容保留，新方法通过注册接入。新增诊断涵盖跨模型训练、标准checkpoint独立评测、真实中断恢复、替换验证指标及变更轮次预算。正式长实验没有启动；具体通过数量与边界见Benchmark方案第14节。

> **2026-11-14 资源验证与首批队列，非准确率结果**：单项10k训练/完整留出集/126图/90k选样资源探针完成，进程树PSS峰值13.89GiB；默认4实验/2重加载，4–7号卡每卡单GPU进程。Random/TopK/LCMD-LL/BAIT-LL × mutation/LHS共8项已建tmux后台队列，按主机负载准入。用户确认checkpoint仅用来源池validation，协议`seed42_18bf9a3289fa`；归一化算法和源池不变。3epoch探针不是收敛或正式排行榜数据。[报告与边界](BUDDING_SAMPLING_BENCHMARK_PLAN_2026-11-13.md#19-单项完整规模资源实测与首批配置2026-11-14当前)。
本轮最终联合回归：**492项通过**，含新增扩展性、跨模型与逐位恢复检查；另验证8张诊断轨迹图、本地offline W&B和旧诊断PT读取。`train/eval/tune --help`、批量dry-run、AST及shell检查通过。没有启动正式长实验。日志：`scratch/probes/redundancy_cleanup_20261114/{final_regression.log,output_smoke.log}`。


> **2026-11-14 Benchmark工程验收，非正式排行榜**：目录迁移、完整池离线oracle、共享split/初始随机2k、四轮冷启动与方法适配已实现；小型CPU诊断验证训练→选样→读取→评测，并验证带权损失、checkpoint重载、本地W&B和预测图输出。没有启动正式长实验，不能报告新的126株准确率或收敛结论。主指标新增`macro_trajectory_rel_l2`，保留旧`all_variables.Relative L2`及逐株`rel_l2`，二者定义不同不得混排。[具体实现与待完成项目](BUDDING_SAMPLING_BENCHMARK_PLAN_2026-11-13.md#13-2026-11-14-实施交付与边界)。

工程验证记录：联合回归241项通过（7项既有Lightning提示，含空诊断loader与worker数量）；随后针对最终有效初值覆盖补测，ours/随机四轮与探针7项复测通过（含一个新增回归）。独立CPU输出测试验证8张诊断预测图、4轮loss/指标、本地offline W&B以及最终checkpoint重载；父环境设online仍未上传。`train.py/eval.py/tune.py --help`、公共sampling导入、批量dry-run、Python编译和shell语法检查通过。原两个池清单与共享split/input缓存哈希保持不变。日志在`scratch/probes/sampling_migration_20261113/{final_regression.log,ours_final.log,output_smoke_final.log}`。这些诊断数据不是正式126株评测，不能进入方法榜单。

> **2026-11-13扩展mutation机制验证**：329次诊断＋20次更严容差复算全部成功；当前SIC1-PM有稳定化/阻断START作用，但不等于完全不可磷酸化。四种DAmP的表达缩放方向与量级可解释，CDC6/NET1对诊断初值较敏感。生产规则未修改，未开展训练精度比较。[方程、实测表格及曲线](BUDDING_EXTENSION_VALIDATION_2026-11-13.md)。

> **2026-11-12零/残余轨迹实测**：8机制×3背景×5强度，三档容差共360次仿真全部成功；严格结果的各机制最大整体RelL2为0.50%–13.26%，96个配对未出现分裂次数/Pattern/生物标签切换。更严容差复核最大差0.00341%；默认容差少数案例差达8.39%，生产容差未改。已输出全24组曲线与PDF。[实验、数值限制及图表](BUDDING_ZERO_RESIDUAL_TRAJECTORIES_2026-11-12.md)。

> **2026-11-12数据规则诊断（非训练性能结果）**：三模块禁用/同基因复合允许列表收紧后，新增转入LHS的5个参数各自置零，WT背景原生210min/500点BDF均成功；其余10个控制也成功。单点可求解不代表联合生物合理，新增零概率未赋值，未开展新模型训练。[规则与诊断](BUDDING_REAL_SUPPORTED_MODULES_AND_COMPOUNDS_2026-11-12.md)。

> **全部126 real配置分辨率验证**：126/126求解成功，300/350/400/500点逐株比较已完成；400点全变量log重构均值2.225%（500点1.994%），最坏核心最高峰损失为幅度0.612%，连续核心通道显著峰数无变化。敏感株包括#118/#110，短暂初始弛豫局部有明显差异；未做新模型训练，未改默认网格。[全体指标、敏感图与126页PDF](BUDDING_ALL_REAL_TIME_RESOLUTION_2026-11-11.md)。

> **300–400点补测**：按用户认为256偏少的偏好，优先建议400点（dt≈0.526min），未改生产默认。本轮300/350/400/500点FNO更新为15.75/22.28/21.18/28.98ms；示例全程曲线接近，单seed学习pilot不能证明精度等价。[完整记录与图](BUDDING_TIME_RESOLUTION_2026-11-11.md#300400点补测同日按用户希望增加点数)。

> **时间分辨率目标澄清**：用户以210min图上的可见波形和实际训练效果为主。新增WT及三突变体曲线叠加，256/500全程波形基本重合、差异集中于跳变局部；按此目标优先验证256点，不以event同区间统计单独要求500点。未改生产默认。[图像、局限与修订建议](BUDDING_TIME_RESOLUTION_2026-11-11.md#用户目标澄清后的判断以可见波形和实际预测效果为主)。

> **2026-11-11 分辨率小实验**：254/254轨迹成功；标准FNO width192/batch128/A6000的500→256点更新耗时24.43→15.71ms、峰值allocated显存914→530MiB；event重构误差上升。96/32局部合成样本、单seed小模型80epoch仅为pilot，不纳入正式模型排行榜。[完整结果与复现](BUDDING_TIME_RESOLUTION_2026-11-11.md)。

> **2026-11-02完整对比**：同FNO/数据/预算的22方法中，Pattern Context MSE最低0.945067（比Power低28.94%），但MAE高1.26%、汇总RelL2高10.28%，仍未全面胜出。逐株80/126 RelL2胜Power，少数badcase退步；MSE净改善主要由#50/#22/#62贡献。Balance和当前Sensitivity/Event无收益，Support未超过Context。[结果、badcase及采集诊断](AL_MECHANISM_COMPARISON_2026-11-02.md)。

> **2026-10-31 16:34完成核查**：四项mechanism与Event重跑全部四轮完整、主模型500epoch/40800次更新、每项有效追加6万条；单项3小时16分至4小时31分，无本轮检出的训练错误。context最终MAE=0.087077/MSE=0.945067/RelL2=0.145252；support为0.092608/0.979995/0.153649。原启动器exit=1来自有意取消等待worker，独立sensitivity退出0。[完整证据、耗时与结果](AL_MECHANISM_COMPLETION_2026-10-31.md)。

> **2026-10-31 新机制批次**：Pattern三个消融及ODE响应一致性已接入，结果使用`mechanism_v1_full`后缀；目前没有正式最终指标，不能混入已完成排行榜。126 real的开发使用历史继续披露。[实现与验证](AL_MECHANISM_IMPLEMENTATION_2026-10-31.md)。

> **2026-10-31 评测独立性说明**：126 real的逐株误差已多次用于设计/诊断，后续论文应将其使用历史作为开发评测披露；采集器不读test/real真值与完全独立盲评是不同要求。新增Pattern条件采集已实现并启动验证实验，尚无新性能结果。[方法与无泄漏协议](AL_PATTERN_CONDITIONAL_RESEARCH_DESIGN_2026-10-31.md)。

> **2026-10-31 ours与完整baseline比较**：global_tail整体RelL2=0.129411、Corr=0.980956略好于Power，但MAE/MSE高9.03%/22.56%，逐株RelL2仅26/126胜出；Power综合和中间预算更强。固定Top20有收益，#124 ORI失分抵消部分收益；建议验证Power为主体、小比例tail/response。[完整比较与待测消融](AL_OPTIMIZED_RESULTS_AND_NEXT_STEPS_2026-10-31.md)。仅分析，未启动新方法。

> **10:29重跑更新**：Event已在GPU0进入正式训练，约第33/100epoch。另修复epoch摘要loss滞后一轮/首epoch显示nan的问题；实际41轮CSV loss均有限，曲线和收敛统计正确。当前进程继续运行，新日志hook以后续启动生效。详见[本批审计](AL_OPTIMIZED_COMPLETION_AND_EVENT_FIX_2026-10-31.md)。

> **2026-10-31 optimized批次检查**：10项完整、1项Event首次选样失败；完整项均实际500个主模型epochs、40800次更新及6万条追加。Event多分裂NumPy布尔判断已修复，通过36项回归和真实FNO512样本重放，单独后台冷启动重跑。整批原用时9小时36分，四卡并行；完成预算不等于严格收敛。[完整性、用时及结果](AL_OPTIMIZED_COMPLETION_AND_EVENT_FIX_2026-10-31.md)。

> **2026-10-30 模型设计分析（无正式重训）**：读取快照为192份Budding CSV＋130份AL逐轮JSON；
> 同一纯随机100k数据集54份非备份历史配置，指标内容去重50份。按参考Plus/hybrid三指标平均名次
> 选20例，新增驱动/日程2×2 oracle诊断。#124主要瓶颈为上游cyclin驱动；#42本次冻结模型主要SSE
> 通道为Vi20。冻结Plus替换三标记物读出，MAE 0.087015→0.078941、MSE 5.183655→4.722494，
> 平均RelL2 0.142708→0.148833，存在退化；这些是确定性读出替换，不是新模型重训结果。
> 已实现 `biom_refiner_marker_closure` 消融和15项相关验证；完整ICHFO仍为研究设计。
> 数据含ODE生成的real-mutant条件轨迹，不能直接当作39变量实测时序。
> [完整报告、20例、局限与复现](OPERATOR_DESIGN_AND_BADCASES_2026-10-30.md)。

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
# 2026-11-13：完整候选池生成验收（非训练排名）

新mutation/LHS生成器各256样本的1进程/8进程独立运行，在参数、实际初值、轨迹、BIO/ABC/六分组及来源元数据上逐元素一致；DataModule读取与seed随机切分、train-only归一化通过，相关34项回归通过。正式两池现已各完成100000条，保存输入无重复或real碰撞，初值/标签/哈希及LHS全部参数坐标核对通过。mutation最终L1/L2/L3=1017/12972/86011，目录5.54 GiB；LHS目录6.18 GiB。LHS12条负值样本24次严格复算均成功、六分组不变，支持数值误差解释，但原解对严格解最大RelL2约2.30%，不能据此宣称全池严格精度验收。未覆盖原池，尚无新模型训练或泛化精度结论。[详细验收与数值证据](BUDDING_COMPLETE_POOLS_GENERATION_2026-11-13.md)。

2026-11-14：完整池采样实验输出已精简并验证，新增同来源test逐样本CSV及原风格完整变量图/双轴loss。格式预览复用3-epoch资源探针，不能作为算法成绩排名；当前8项正式FNO实验恢复运行。规范与验证见[SAMPLING_OUTPUTS_2026-11-14.md](SAMPLING_OUTPUTS_2026-11-14.md)。

2026-11-15：原hybrid_mse_rel八项已完成结果归档，不改成绩；新16项rel_l2批次单独保存，不将两种训练目标的曲线混作同一损失。 详见[BUDDING_REL_L2_BATCH_2026-11-15.md](BUDDING_REL_L2_BATCH_2026-11-15.md)。


### 2026-11-15：16项纯Relative L2完整批次及配对结果

16/16完成，全部退出码0、80轮完整评测、10080幅real预测图数量检查通过；最后任务在北京时间2026-11-15 12:13:44结束。自有批次PSS峰值40.73GiB，watchdog未触发，没有待运行任务。

完整对比产物：`/home/users/zsy/mydata/budding_yeast/analysis/sampling_loss_comparison_20261115/`，含report.md、comparison.png、final_metrics.csv、paired_comparison.csv、per_mutant_changes.csv及audit.json。复现脚本为scripts/analysis/compare_sampling_losses.py，不加载轨迹或checkpoint，仅扫描JSON/CSV/目录。

旧hybrid只存在4方法×2池，因此以下只对这8组配对；新方法不混入loss前后均值。配置差异只有loss；冻结real参考及划分相同，两池Random五轮有序训练ID完全一致。自适应方法D0相同，后续选样会随模型改变。

| 池（4种配对方法均值） | MAE 旧→新 | MSE 旧→新 | 历史Relative L2 旧→新 | 历史Correlation 旧→新 | macro轨迹RelL2 旧→新 |
|---|---:|---:|---:|---:|---:|
| mutation | 0.20326→0.18147 | 6.24009→4.40982 | 0.21808→0.27859 | 0.94341→0.89486 | 0.34178→0.33142 |
| LHS | 0.45285→0.44721 | 31.85812→33.33682 | 0.40579→0.46874 | 0.80534→0.76411 | 0.60696→0.61231 |

mutation：MAE平均降低10.72%、MSE降低29.33%、macro轨迹RelL2降低3.03%，但历史Relative L2升高27.75%，历史Correlation下降0.04855。LHS：MAE降低1.25%、MSE升高4.64%、历史Relative L2升高15.52%，整体没有稳定综合改善。历史Correlation全部下降，但逐样本非退化通道相关性两池均改善，不能等同于每条波形更差。

新mutation+LCMD的MAE=0.15397、MSE=1.66161，是新批次mutation中的最低值；历史RelativeL2=0.27074仍高于旧LCMD的0.20981。#124的单样本MSE从317.12降至23.89，但#22从15.87升至21.26、#42从0.482升至1.452；仅60/126株的MSE改善，均值改善集中在部分大误差突变体。Random固定训练集对照中#15 MAE从0.2311降至0.1296，#42/#124则退化。

结论：纯Relative L2不是全面升级。mutation绝对误差/部分动力学表现受益，原Hybrid在当前历史Relative L2和全局Correlation上更稳。训练的归一化逐样本逐变量截断RelativeL2、物理评测的全样本逐变量RelativeL2和逐轨迹macro定义不同；删去MSE改变优化权重。机制解释需与观测区分。仅seed42，不构成多seed显著性结论；不根据新增无配对方法推断loss优劣。

## 2026-11-15：新增物理主Relative L2与训练cap恢复对照

用户确认最终eval只报告物理空间精度，新增优先的`sample_channel_rel_l2`（逐样本／变量的时间范数比，eps=.1保护，无cap）；旧pooled Relative L2保留。JSON/CSV/预测图/performance与本地W&B新字段已接入，独立eval直接使用源ODE真值。62项相关回归通过。此变更没有覆盖旧正式结果，也没有修改既有validation选checkpoint口径。[具体公式与接口](3_LOSSES_AND_PHYSICS_CONSTRAINTS.md#7-用户最终确认正式-eval-新增物理空间不截断的主-relative-l2已实施)。

另完成两臂诊断：从同一mutation＋BAIT第二轮异常checkpoint继续，各20epoch/640步，同4000训练行、固定统计、batch128、LR1e-4、warmup2、minLR1e-5、梯度范数裁剪1；仅训练比值cap10与None不同。固定512合成validation选择best，512合成test及126 real仅事后评测，两臂before完全一致。

|固定20epoch后的last指标|cap10|不截断|
|---|---:|---:|
|合成validation，新物理RelL2|0.736101|1.022140|
|合成validation，MSE|39.499632|55.817971|
|合成test，新物理RelL2|0.792796|1.065693|
|real，新物理RelL2|1.093951|1.205166|
|real，MAE|0.351466|0.359928|
|real，MSE|17.697548|22.663455|
|real，CDC14T MAE|1.140966|0.157075|

cap分支每轮都有3544个超限训练项；uncapped末轮仅2项，说明硬cap确实影响局部恢复。但取消后validation整体指标退化，real有31/39变量的新RelL2变差；修好CDC14T不能代表整体升级。因此训练默认cap10暂保留，评测始终不cap。此处是单seed、已训练状态的短程恢复，不是冷启动正式方法排名，不以real/test选择训练超参数。[实验细节及官方实现区别](3_LOSSES_AND_PHYSICS_CONSTRAINTS.md#8-训练cap10不截断的实际小对照已完成)。

全部文件：`/home/users/zsy/mydata/budding_yeast/analysis/relative_l2_training_cap_probe_20261115/{report.md,comparison.png,results.json,comparison.csv,real_per_variable.csv}`。训练和诊断均已结束；未重启正式批次、未复制轨迹。

## 2026-11-15后续：只用新Relative L2与60epoch冷启动

最新生产评测只输出新物理sample/channel Relative L2，旧pooled定义移入历史归档，不再双列或在图中展示Legacy。原字段Relative L2/rel_l2现在是新定义，新JSON/CSV/history有`physical_sample_channel_rel_l2_v1`版本；旧文件不覆盖，多seed/历史loss汇总有混用保护。注意不仅平均顺序不同，零附近分母保护也不同；仅单株图中高信号变量两种数值可以很接近。[例子与迁移说明](3_LOSSES_AND_PHYSICS_CONSTRAINTS.md#9-仅采用新物理relative-l2旧定义移出生产路径)。

按用户要求另完成60epoch从头训练对照（不沿用20epoch恢复权重）：seed42，两组相同随机初始化、4k训练集合与统计；batch128，LR7.5e-4，warmup4，minLR1e-5，梯度范数裁剪1。各1920步，全部loss/梯度有限。cap/uncapped训练loss均从1.061545下降至0.258658/0.258306；新物理RelL2在validation为1.776419/1.771358，在real为2.097527/2.097539。cap只触发24/936万次训练检查，收益或影响在本次设置中很小；不截断能够正常结束。末10轮平均每epoch训练loss下降0.3041%/0.2968%，不能等同严格收敛。正式默认cap未改，不用单seed微小差异断言普遍优越性。

目录：`/home/users/zsy/mydata/budding_yeast/analysis/relative_l2_cap_60epoch_coldstart_20261115/`，含report.md、comparison.png、results.json、stability_summary.json、metric_examples.json。RSS峰值3.044GiB、CUDA allocated峰值0.902GiB，后台任务正常结束。64个不同相关测试通过（两批有交叉，非68项）。

用户随后明确采用不截断训练：当前RelativeL2默认cap已关闭，正常模型／Hybrid相对项不再隐式cap10；分母eps=.1、梯度范数裁剪1、非有限loss检查和既有编码保护保留。旧pooled评测的归档实现与Python引用也已彻底删除。29项相关检查通过，另验证新checkpoint记录实际损失定义；无正式重训，旧实验数字不改写。此为用户最终配置选择，不能将既有cap10实验追溯标为不截断训练。[最终规则](3_LOSSES_AND_PHYSICS_CONSTRAINTS.md#10-用户确认训练默认不截断旧指标实现完全删除)。


## 2026-11-16：max分母与epsilon对照完成

按用户确认，生产Relative L2训练与物理评测均改为 `max(||target||_time, eps)`，无loss cap，梯度范数裁剪1。新物理指标版本为 `physical_sample_channel_rel_l2_floor_v2`，不与smooth v1或旧pooled数字混算，历史结果不改写。

同一mutation/BAIT的4k已查询训练集合、同一train-only归一化和seed42随机初始权重，三组分别训练eps0.03/0.1/0.3；均60epoch/1920步，batch128，LR7.5e-4，warmup4，cosine最低1e-5。所有物理评测eps固定0.1。最后epoch合成validation物理RelL2分别1.906388/1.779945/1.536348，因此先选0.3，再评测test/real。合成test RelL2为2.078948/1.957462/1.710899；real RelL2为2.272277/2.224718/2.006778，real MAE为0.375326/0.373068/0.342805，real MSE为19.174719/18.294945/14.722330。全部loss/梯度有限，35项相关回归通过。

单seed小对照支持下一步优先考虑训练eps0.3，不能当作完整采样benchmark排名或跨seed最优证明。当前生产默认eps仍暂保留0.1；正式重训未启动。RSS峰值3.032GiB，CUDA allocated峰值0.902GiB，后台诊断正常结束。

结果：`/home/users/zsy/mydata/budding_yeast/analysis/relative_l2_epsilon_floor_20261116/`；[完整规则与实验边界](3_LOSSES_AND_PHYSICS_CONSTRAINTS.md#11-2026-11-16max分母及epsilon受控对照)。


## 2026-11-16最终设置：smooth eps0.01

按用户最新选择，训练与物理评测默认分母均为sqrt(||target||_time²+0.01²)，不截断，梯度范数裁剪1；训练/物理评测空间仍保持各自定义。新物理版本 `physical_sample_channel_rel_l2_smooth_eps001_v3`、输出schema v5，旧max/smooth0.1/pooled结果保持原样，不允许直接混算。历史checkpoint的实际损失规则保留，新训练默认smooth0.01；没有启动新的正式重训。本次仅实现及回归验证，不把此前max epsilon对照冒充当前新配置的性能证据。规则及测试见损失文档第12节。

此次实现通过37项损失、评测、checkpoint及采样流程回归检查（含8种baseline五轮小型集成）；没有新的正式性能结果。


## 2026-11-16：新验证协议与六并发就绪

用户确认checkpoint选用与评测一致的物理sample/channel Relative L2。新协议`seed42_02eebd57592f`复用旧源池、初始样本、切分和四个缓存inode；新完整训练尚未启动。新结果将写入`experiments/budding_sampling/seed42_02eebd57592f`，旧结果不覆盖。

资源诊断逐步2→4→6预处理，并六项重叠训练/评测：owned PSS峰值67.744GiB、最低可用645.465GiB；六项均正常完成，各GPU allocated峰值0.902GiB。默认总并发6、每卡最多2、IO并发2；自动watchdog保留96GiB owned PSS上限/128GiB可用底线。两epoch、warm-cache、非正式性能结果，不证明吞吐最优或模型收敛。[资源报告](/home/users/zsy/mydata/budding_yeast/resource_profiles/concurrency_2_4_6_smooth_eps001_20261116/report.md)。

正式批次随后已获用户授权，于2026-11-16 02:40在tmux budding_sampling_eps001_20261116启动：16项、最多6并发，结果写入新协议seed42_02eebd57592f，旧16份配置启动后哈希/inode一致。暂无完成结果，不作精度排名。启动计划、source快照和资源日志集中在新结果根的batch/。

## 2026-11-16完成及新旧统一评测

最新16项全部完成，最晚完成时间2026-11-16T08:17:49.612372+08:00（Asia/Shanghai）；16项退出码均0、无Traceback，80个新round均100/100/100/100/200epoch完整、最终10000唯一训练输入、无新增ODE。连同旧批共32项/160轮、20160幅real图数量检查通过。当前无运行/等待任务。

旧批是pure Relative L2（训练eps0.1、cap10、整轨迹macro validation选best）；新批为eps0.01、不截断、物理sample/channel validation选best。验证推理精度和并发也改变。两批old/new原JSON的Relative L2不可直接比较，因此32个最终checkpoint均使用同一real参考、原batch128、float32推理、物理smooth eps0.01重算；新模型重算与已保存主指标一致。没有重写历史结果。

|候选池|指标|旧八方法均值|新八方法均值|相对变化|改善方法数|
|---|---|---:|---:|---:|---:|
|mutation|MAE|0.195921|0.227082|+15.90%|1/8|
|mutation|MSE|5.697612|7.690987|+34.99%|2/8|
|mutation|unified_RelL2|7.046660|8.473850|+20.25%|4/8|
|mutation|macro_RelL2|0.341143|0.374917|+9.90%|1/8|
|mutation|pattern_accuracy|0.549603|0.464286|-15.52%|1/8|
|lhs|MAE|0.447290|0.449294|+0.45%|3/8|
|lhs|MSE|34.005737|33.133017|-2.57%|4/8|
|lhs|unified_RelL2|1.861933|1.855541|-0.34%|4/8|
|lhs|macro_RelL2|0.603064|0.620799|+2.94%|2/8|
|lhs|pattern_accuracy|0.281746|0.296627|+5.28%|7/8|

结论：mutation整体退化；LHS混合、总体接近，不构成全面升级。mutation/Uncertainty的MAE/MSE改善，其余7方法MAE均变差；LHS/MaxDist改善明显。
Random五轮训练ID及顺序相同，最终训练归一化逐项一致，模型前向/编码代码哈希相同；退化不能只归因于选样变化。自适应方法后续数据变化，比较的是整条流程，而非纯loss单因素。
新mutation/Random的4914个样本/变量项中，仅244项（4.97%）目标时间范数≤0.01，却贡献96.68%的主Relative L2；CDH1T单变量贡献84.20%。这包含真正cdh1Δ/APC-ts的错误表达，有生物意义，不能简单忽略或加大eps掩盖。剔除近零项作事后诊断时，Random其余项也从0.4060退化为0.4469，因此并非单纯指标口径错觉。
新Random和Power虽各训练200epoch，最终best分别来自56/36epoch（旧176/197）；新Random选中epoch训练loss约0.1803，末epoch约0.0901。说明checkpoint选择可能牺牲后期的其它改善，但没有存每个epoch/最后权重，尚未做同一训练按两种validation准则选best的反事实，不能把所有退化断言为选择规则导致。
训练归一化Relative与物理Relative对近零的定义不同，公式相同并不自动对齐权重。应先用固定Random数据，在同一训练中保存两种validation准则best及last，分离checkpoint影响；只按预定义合成validation选择，不以real挑epoch。此对照尚未启动。
新批PSS峰值59.49GiB，无内存保护触发；批次监控墙钟旧约8.87h、新约5.62h，非严格吞吐对照。只有seed42，不作统计显著性推广。
报告及16对明细：`/home/users/zsy/mydata/budding_yeast/analysis/sampling_protocol_comparison_20261116/report.md`、comparison.png、paired_comparison.csv；逐株/逐变量、每池top20、checkpoint选择和资源核查均有CSV/JSON。只保存标量统计，无重复轨迹。
# 2026-11-16 补充诊断：较小的平滑epsilon

固定mutation已查询训练4000条、validation512条、相同随机初始权重/批顺序，训练epsilon0.01/0.003/0.001各60epoch（1920步）。所有物理评测统一smooth eps0.01，末轮物理validation Relative L2为8.377018/8.628620/8.526945；validation MAE为0.494015/0.503551/0.501721。三组在保留梯度范数裁剪1的条件下全部有限完成，未证实减小epsilon带来验证收益。

归一化训练目标范数最小0.076561，epsilon0.01对原始单项Relative的最大/平均改动为0.842249%/0.019237%；此子集没有极近零归一化范数，不能推广为所有KO或后续训练集的稳定性保证。本次仅诊断，不加入正式五轮方法排名、不据real挑epsilon、未改生产配置。物理best与归一化best的作者代码差异和受控实验详情见[损失文档第15节](3_LOSSES_AND_PHYSICS_CONSTRAINTS.md)。输出：`analysis/relative_l2_smooth_epsilon_20261116/report.md`（mydata目录下）。

## 2026-11-16：Power固定最终轮的物理多指标补查

用户要求比较物理MAE/MSE/Corr，而非只看Relative。正式Power最终round04只保留epoch36 best，因此复用同一10000条训练数据、全部5000条同池validation、原模型seed及训练协议，诊断重跑一次200epoch。四个归一化张量完全一致；epoch36参数SHA与原模型一致；200epoch全部既有history数值与原训练完全一致。诊断observer不改变训练随机状态，相关共享训练/监控回归18项通过。未重启采样、未改原正式文件、未用real选择权重。

|固定epoch|real物理MAE↓|real物理MSE↓|real物理Relative L2↓|real Corr↑|
|---:|---:|---:|---:|---:|
|36（原物理validation-best）|0.276126|10.217533|12.866506|0.799023|
|50|0.233186|5.656558|12.682310|0.829627|
|100|0.187816|3.007052|12.756761|0.877653|
|150|0.172359|2.422906|12.591127|0.886759|
|200（last、归一化validation-best）|0.171135|2.441152|12.552222|0.887489|

36→200的MAE/MSE分别下降38.02%/76.11%，Corr增加0.08847，Relative只下降2.44%。当前物理Relative选best牺牲了后期的部分绝对误差/波形改善；这与实现选错checkpoint不同。MAE/MSE也不能代替近零通道诊断；应在正式下一协议前用合成validation定义选择目标，不从real结果反向挑epoch。150→200的MAE改善仅0.71%、MSE小幅增加，不能将200epoch当作所有指标都最优。

仅对一个方法的固定最终轮做检查，不作为新采样方法的五轮完整结果或普遍因果结论。完整指标、逐株表、固定epoch模型和对照图位于`/home/users/zsy/mydata/budding_yeast/analysis/power_final_round_physical_replay_20261116/`；`reproduction_audit.json`保存精确复现证据，没有重复保存源轨迹。

同池5000条validation的物理MAE/MSE从0.418817/128.481277降至0.275642/56.005394，而Relative略升0.261%，即相同分歧也存在于合成验证集。real有111/126株MAE改善、105/126株MSE改善；整体MSE降幅约93.6%来自ORI通道，末轮Relative约90.6%来自CDH1T，不能把任何单一汇总量当成全部预测目标。逐株、逐变量与同池validation完整表见该目录的`validation_report.md`及相关CSV；后续选择协议应先明确目标，再统一用于所有方法。

## 2026-11-16：归一化验证最优协议已准备，尚未启动

用户确认新的主/辅助模型均按同池合成validation的归一化Relative L2最低epoch保存和使用，monitor=`val_loss_synth`。每轮新增后按累计train-only重新拟合P/Y统计，轮内冻结。正式eval仍报告物理四指标；不根据real选择权重。

协议`seed42_b5bff21cb8de`从`seed42_02eebd57592f`派生，只有validation_metric配置变化。输入、划分、初始ID通过四个硬链接缓存保持一致，无新增仿真或轨迹副本。新结果将位于`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/seed42_b5bff21cb8de/`，不覆盖上午16项。

16项首轮两种best均不同：mutation为99/100epoch、LHS为83/54epoch（物理/归一化）。14项模型相关采样实验要从首轮重跑才能代表新流程；两项Random可复用全部选样ID，只重训要报告的轮次。旧结果有效但属于旧checkpoint协议，不应重命名成新结果。检查表见mydata的`analysis/normalized_validation_readiness_20261116/checkpoint_choice_audit.csv`。本次仅准备代码/协议与CPU回归，没有启动正式批次，也不宣称新协议精度提升。

## 2026-11-16：正式重跑前复查

数据隔离、初始公平性、逐轮train-only重拟合、normalized validation-best及恢复链通过核查。两池20万输入无重复或real碰撞，同池初始统计下最大输入Z约15/22，不存在已发现的稀有参数标准化爆炸。新增raw非有限预测/validation评分保护并纠正实际LR日志，共45项CPU回归通过；未改变训练损失、epochs或采样评分。

全池轨迹只读扫描和11条严格复算发现少数数值伪影，应先定点修正再冻结长跑：7条TEM1越总量、4条明显负尾，全部复算成功且Pattern组未变。均不在固定validation/test，最严重例子未被上午任何方法训练；不可用它们解释上午整体退化。源池和旧结果保持，未启动重训。

论文须区分程序上的无泄漏与反复查看126株real后的开发反馈；单seed不能给稳定排名，cross_pool属于分布外诊断，LHS背景参数±50%与mutation固定背景也构成比较差异。详见[全流程审计](EXPERIMENT_FLOW_AUDIT_2026-11-16.md)。

用户随后确认保留末epoch纯权重，已在共享训练接入`state_dir/last_weights.ckpt`，不替代best或保留optimizer恢复文件；新增最后权重、完成缓存完整性和绘图中断恢复检查，26项相关CPU回归通过。源池数值疑点重新比对生成时哈希和实际磁盘条目，确认参数范围/初值约束存在，但仿真出口缺乏逐时间不变量检查；没有修改采样分布或启动新正式实验。


## 2026-11-16：数值修复已完成，旧池保留

已对mutation 3条、LHS 8条确认数值伪影进行同输入严格复算并以11行NPZ修补视图发布；两池仍各100000条，输入/ID/顺序/切分未变。新默认池目录增加`_numerical_qc`，协议`seed42_a866aef5d988`，共享四个输入/划分缓存和原126株real参考，原池/旧结果保留。以后生产求解默认BDF 1e-8/1e-10，检查TEM1已知总量约束并按需重试1e-10/1e-12，失败记录且补采至足量；负值继续保留、未恢复real×2幅值过滤。17项CPU回归通过，没有启动正式重训。旧16项中10项用过7条错误监督数据，首次受影响轮及重训边界详见[数值修复记录](NUMERICAL_POOL_REPAIR_2026-11-16.md)。本条更新早前“源池暂未修正”的阶段记录。

2026-11-16并发资源复测完成：旧16项批次墙钟5:37:03，累计任务墙钟29:19:38。新8项短训练与90k选样推理PSS峰91.61GiB；重加载并发2→3使8项加载就绪耗时739.23→573.33秒（-22.44%）。默认更新总8项／每卡2项／重加载3项，并增加持续IO/内存压力停止保护；22项CPU回归通过。这是warm-cache资源诊断，不是新的准确率/完整训练吞吐结果；尚未启动新协议正式五轮训练。详见[资源测量与边界](BUDDING_REL_L2_BATCH_2026-11-15.md)。

用户后续要求已执行：取消固定自有PSS限额，只依有效剩余内存/压力保护；同流程IO4/6/8加载379.43/299.19/211.40秒，默认提高至总8项／每卡2项／IO8。IO4完整资源测试8项退出0，PSS峰90.22GiB、最低可用600.24GiB；IO6/8仅完成加载即停止，均为warm-cache单次测量。22项CPU回归含PSS200GiB/剩余400GiB继续、剩余不足仅停止自有任务的验证；128GiB可用余量及持续PSI保护保留。本条覆盖此前96GiB/IO3默认，不改变准确率排行榜或声明正式重训已启动。详见同一资源测量文档末节。

2026-11-17历史故障核查：发现11月14日旧concurrent.py探针遮蔽标准库，PyTorch导入链触发模块顶层递归Popen，90秒父超时未清理后代；已保全并隔离该脚本，当前标准库解析恢复。后代错误日志写到13:14，14:53可用682GiB/load1=162.18；缺少12点系统指标及内核日志权限，不能确认为OOM或唯一原因。没有重放探针/修改排行榜。详情见[资源与历史故障核查](BUDDING_REL_L2_BATCH_2026-11-15.md)末节及analysis/incident_20261114_1200/report.md。


2026-11-17扩大并发正在排队：12项/IO12因持续全局IO压力中止（151.4秒，PSS峰31.58GiB、最低可用676.91GiB），未完成训练，不进入性能排行。停止所有worker后全局IO PSI仍高、/home设备忙而/data无明显IO，未归因具体用户。已设置等待恢复后自动重测12/16项、IO8，测试及生产余量统一100GiB，无自有内存限额；24项CPU检查通过。正式并发仍8项，不能宣称12/16已通过。实时状态resource_profiles/higher_concurrency_12_16_20261117/retry_status.json，详见资源文档末节。


2026-11-17 01:33资源重测已停止：12项/IO8主模型短训练/独立评测全完成，但90k候选特征提取阶段触发global IO PSI保护（23.64%），未完成全资源验证。PSS峰130.85GiB、最低可用576.43GiB；不是100GiB余量触底。读盘累计76.68MiB在后段不增，不能归因本批数据读取或确认为保护误判。16项未启动，tmux结束，正式并发保留8；完整吞吐与准确率排行不更新。详情见资源文档末节及resource_profiles/higher_concurrency_12_16_20261117/retry_attempt_summary.json。


## 2026-11-17：资源对照通过，采用16并发正式运行

12项与16项诊断均完整完成并全部退出0；12项14分21.60秒，PSS峰131.86GiB、最低可用571.57GiB；16项17分21.31秒，PSS峰178.76GiB、最低可用522.32GiB。每项10000训练样本、2epoch、90k候选特征及LCMD/BAIT/双模型推理；未覆盖完整五轮、辅助成员训练或收敛，不能据此保证所有负载绝对安全。16项memory_policy=available，12项strict，不能视作严格单因素吞吐实验。

用户授权测试后正式启动；离线batch默认总16／每卡4／重加载8／候选评分4，GPU仅0–3。按系统有效可用内存100GiB保底，取消自有内存配额；memory PSI仅预警。IO保护采用全局PSI>5%并与设备busy>=90%且queue>=16或await>=250ms持续60秒联合判定，设备数据缺失回退strict20秒。保留CPU准入、进程隔离与只停止本批子进程的规则。候选评分等待单独记入selection.json，不改变选样公式。

正式协议seed42_a866aef5d988，输出experiments/budding_sampling/seed42_a866aef5d988/，与旧目录分开；两池各8项FNO，seed42、五轮2000→4000→6000→8000→10000，100/100/100/100/200epochs，逐轮训练集refit，normalized validation-best、eps0.01 smooth/no cap、保留last不变。W&B仅本地offline。资源原始报告：/home/users/zsy/mydata/budding_yeast/resource_profiles/scoring_limited_12_16_20261117/report.md。


2026-11-17 02:51正式批次已在tmux `budding_sampling_formal_20261117`启动，输出 `/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/seed42_a866aef5d988/`；批日志 `batch/launcher.log`，单项 `<pool>/<method>/seed_42/train.log`。45项资源CPU回归和3项小流程回归通过，dry-run16项及参考SHA核对通过。

02:54复核：16个正式run进程均存活且WANDB_MODE=offline、GPU仅0–3；mutation/VeSSAL已完成首轮至少16epoch，loss有限，其余正在加载/预处理。启动核查记录batch/startup_verification.json。尚不能声称整批五轮完成或所有后续评分均通过。


## 2026-11-17 完整批次结果：normalized-best与修正池

协议seed42_a866aef5d988，FNO、两池8baseline共16项全部完成/退出0，80轮、10080幅轮次real图；预算2000→4000→6000→8000→10000，epochs100/100/100/100/200、batch128、lr5e-4、smooth eps0.01无cap，每轮train-only refit，以同池val_loss_synth选best并保留last。逐轮ID与holdout无交集、同池初始有序2000一致、best与history最低值一致。02:51:08至06:08:11，墙钟3:17:04，任务累计46:28:51；PSS峰151.96GiB，最低可用538.82GiB，16并发未触发守护。

mutation真实评测：Power MAE0.15030/MSE2.25805/Relative1.05698/Corr0.90974；Random0.17746/3.44796/12.36778/0.89962；LCMD0.16444/2.65932/12.88390/0.89108；Uncertainty0.21613/18.05836/1.26566/0.90313。Power相对Random的MAE/MSE下降15.30%/34.51%；LCMD六分组准确率最高68.25%，Power63.49%、Random65.87%。LHS的Power绝对误差最好0.41464/24.90815，但VeSSAL的Relative最低1.60615，不能单指标宣布全胜。

关键诊断：CPU重推mutation Random/Power/LCMD与LHS Random的126株，总Relative与GPU结果相差<0.1%。Random/LCMD总Relative约88.90%/88.21%来自CDH1T通道；#37真值0，Random预测约0.97、LCMD约1、Power接近0，说明零通道关闭确有失败，也因小分母被放大。25株real的该通道近零；训练中Random/LCMD/Power/BAIT对应零初值样本1229/1364/1918/1863，不能仅用数量解释差距。Uncertainty #124贡献整体MSE78.84%。末轮后20epoch验证loss接近平台，best规则核对正确；尚不证明改变训练预算无收益。

与旧02eebd批次，mutation Power MAE0.27613→0.15030、MSE10.21753→2.25805；新配置同时改数值标签与checkpoint规则/选择路径，不能单独归因。当前结果仅seed42且只含baseline，没有ours成绩；应固定协议，以合成已查询反馈和系统已知关闭/初值/ODE规律设计采集，不使用real结果打分。

完整报告：/home/users/zsy/mydata/budding_yeast/analysis/sampling_results_20261117/report.md，含全部16项、Top20坏例、逐轮图、收敛窗口、逐通道/初值覆盖/划分审计与旧批次对照。复用分析入口scripts/analysis/analyze_complete_sampling_batch.py（argparse，CPU可重推126株，不加载整池轨迹、不改训练）。详见报告的跨池OOD失败解释；LHS额外背景扰动与mutation不同，不能把生成范式差距全归因选样算法。


### 2026-11-17多seed前补充核查

总墙钟3:17:04，主模型80轮+辅助16轮完成；所有辅助best也与归一化validation最低值核对一致。主模型mutation前四轮best97～100，末轮183～200；LHS首轮八方法均54，末轮123～164，第二轮还有64/76。确有几十epoch的best，但不是早前物理指标选best的口径问题。best_epochs.csv已保存全部80行。

新增诊断：VeSSAL随机补齐在mutation为299/8000=3.74%，LHS为5274/8000=65.93%（逐次1347/1333/1271/1323）；作者query支持fill_random，不据此认定bug，但论文报告需注明penultimate适配及补齐比例，多seed前建议对账该分支。未擅自修改算法或重训。详见docs/history/SAMPLING_BASELINE_ROADMAP_2026-11-17.md及vessal_fill_summary.json。

补充复现建议：已读10篇本地PDF与14个参考仓库，优先MaxDet-LL、k-means++-LL及KMeans/Herding；现有Physics-PRE/Lev无需重复实现；监督筛选方向优先GradMatch。PEAKS需回归推导，HGGS/OED/OGAS保持生成流程，CREST/ERACS保持动态重选，不假装固定池五轮原版。完整清单见[SAMPLING_BASELINE_ROADMAP_2026-11-17.md](SAMPLING_BASELINE_ROADMAP_2026-11-17.md)。本轮没有实现新baseline或启动多seed。


2026-11-17清单更新：用户要求至少2～3项2025～2026近期baseline，并允许生成新样本。近期主对照调整为Physics-PRE2026、OED2026、OGAS2026，PEAKS2025为补充/备选；HGGS2024保留生物相关性，不占近期名额。Physics-PRE现有预测事件±4步mask尚非完整混杂系统验证，新增jump/IC评分要单独命名；生成类保持原流程及实际ODE/标签/训练成本账本，并配同在线流程Random生成对照。本轮仅更新方案，没有实现或启动实验。详见SAMPLING_BASELINE_ROADMAP_2026-11-17.md第10节。


## 2026-11-17近期baseline基础实现完成

已接入Physics-PRE差分模板保护、OED后验/UCB和`oed_temporal`生成适配、OGAS条件DDPM+CFG/密度比及`ogas_round_projected`轮次式合法投影适配，并有`random_generate`共同生成流程对照。统一入口`bash run_sampling_budding.sh generate ...`，不自动启动长实验。原版OGAS异步连续在线流程、HGGS及PEAKS回归版仍未完成，不能写成已完整复现作者全部实验。

18项新增CPU测试全部通过，另通过原有Physics-PRE两变体与冷启动多轮路径定向回归（当时23项合计通过）；实际canonical ODE→小FNO训练→评测→恢复闭环已验证，含失败补采、冻结源覆盖引用、CFG、密度比、SHA损坏拒绝。默认1000扩散步/50采样步单独有限性测试通过。82个既有Budding/data根目录Python文件实现前后逐字节一致；两种生产池catalog/范围和LHS原配置SHA验证通过。没有使用GPU或开展正式多seed。

参数、初值、ODE、事件、分布、real碰撞救援及数值政策均通过现有提案器/仿真器调用；算法不得提交任意生成的物理向量。新增成功轨迹仅存一次，初始数据和每轮快照用源引用；生成器状态仅保留两恢复点；ODE开始/结束和内部重试计费。大规模生成流程资源开销尚未测量，不直接套用固定池16并发。详见[近期baseline实现](RECENT_BASELINE_IMPLEMENTATION_2026-11-17.md)。


## 2026-11-17：新生成baseline的声明口径

新增OGAS-Online-Projected（`ogas_async_projected`）及同流程`random_async`，输出独立在线hash；不可直接冒充固定池五轮完全同训练预算结果。报告全部ODE尝试/严格复算、成功标签、唯一训练曝光、训练步数和墙钟，明确best实际见过的输入。Physics-PRE是事件ODE适配，OED-Temporal的信息量/高维字典/合法域最大化是任务适配；OGAS合法投影和CPU执行也非作者原实验。尚无这些新方法的正式实验成绩/生产资源峰值，CPU小闭环不作为leaderboard结果。版本及差异见[baseline核查表](BASELINE_FIDELITY_AND_OGAS_ONLINE_2026-11-17.md)。


## 2026-11-17：新增回归方法的排名边界

PEAKS-Regression-Reference、ERACS-Regression-BatchOMP、Low-Rank-GradientEnergy与HGGS-Projected已接入可执行工作流，目前只有CPU小规模功能/恢复验证，无正式成绩。PEAKS/ERACS与random_stream/random_dynamic同流程对比；Low-Rank报告有放回draw slots、唯一行、pilot及历史训练输入，不能冒充唯一查询10000；HGGS报告coarse bank访问/pilot训练与所有新增ODE尝试。原有seed42成绩和五轮协议不改写，不直接混排不同训练日程/访问条件。详见[实现、核查和费用](REGRESSION_SAMPLING_BASELINES_2026-11-17.md)。

## 2026-11-19：初始化与收敛再次核查

只读核查当前协议`seed42_a866aef5d988`相关LCMD、Low-Rank、HGGS修复版、PEAKS、OGAS对齐版共10项已完成结果，未更改后台实验或启动训练。完整统计与精确目录见[核查报告](/home/users/zsy/mydata/budding_yeast/analysis/baseline_initialization_convergence_20261119/report.md)，同目录`audit.json`与`convergence.csv`保留原始统计。

HGGS先由共同随机2000加训练侧另8000形成10000粗池，再按GF选1000边界、辅助试训、另1000高残差形成初始2000；mutation/LHS与共同初始仅重叠380/407，粗池与验证/测试重叠均0。GF后另生成8000，因此训练侧标签访问至少18000，不能称所有方法共用相同初始或查询总成本10000。LCMD与HGGS主训练是五轮100/100/100/100/200，合计31600步；HGGS另有100epoch/800步GF辅助试训。Low-Rank实际是共同2000试训100epoch/1600步后一次性有放回带权选样、最终200epoch；唯一样本mutation8794、LHS8394，主训练13800/13200步。PEAKS和异步OGAS为连续200epoch/15800步，初始统计冻结，不能套用五轮refit口径。

以191–200相对181–190均值变化率衡量最终阶段：mutation LCMD/HGGS/Low-Rank/PEAKS/OGAS训练loss分别−0.676/−0.706/−1.075/−1.299/−0.744%，验证loss分别−0.085/−0.089/−0.164/−0.150/−0.042%；LHS对应训练−0.803/−0.820/−0.751/−0.889/−1.276%，验证+0.005/+0.005/−0.013/−0.017/−0.002%。验证接近平台不等于优化充分收敛，尤其余弦尾部LR接近1e-5；不能只凭200epoch保证收敛，也不据real结果决定训练时长。

新增待复核问题：mutation HGGS的GF辅助试训恢复到第1epoch的validation-best，该epoch实际LR=0，train loss后续由1.0243降至0.2663，但公共合成验证由70.315升至1398.296；因此第二阶段残差补选不能描述为使用充分训练的辅助模型。原作者GF代码使用训练中的当前模型，没有公共fit_round的自动best恢复；这是辅助checkpoint协议与边界分布/归一化需要进一步审计的点，不能把最终主模型best=193epoch与该辅助异常混淆。本轮仅记录问题，未热改源代码或重跑HGGS。

## 2026-11-19：HGGS作者训练控制复核及隔离修正

详细报告：[HGGS及三方法训练流程复核](../../papers/baselines/sampling/HGGS_TRAINING_FIDELITY_REVIEW_2026-11-19.md)。作者源码明确GF结束后重新初始化一次MGS主模型，然后各MGS迭代持续使用同一模型/优化器/调度器；不能误写为GF权重一路继承，也不能把项目每轮冷启动称为作者原流程。GF内部使用当前试训模型，不恢复公共validation-best；作者GF全局补选默认还分五次穿插训练，项目一次性补选也是简化。当前旧HGGS结果仍为适配版，不能宣称完整源码训练流程一致。

实测mutation GF checkpoint的边界拟合尺度使kaswi/kppc1/kppf6验证输入绝对Z达到1896123.875/3792247.75/3792247.75，三列训练标准差均约1e-6；以共同初始训练2000为参考，整个验证输入最大绝对Z为14.8246。未读取holdout真值拟合参考统计。该尺度失配是直接观察到的异常因素，尚未用完整受控重训证明唯一因果。

GF定点修正在`scratch/probes/hggs_fidelity_20261119/staged/`：内部反馈加载末epoch并核对SHA/epoch/统计，固定已揭示初始训练参考统计、重新编码训练及验证、记录参考行指纹和GF版本，拒绝旧缓存。6项CPU回归通过，包含实际小FNO试训与checkpoint重载/holdout隔离/恢复；补丁`hggs_gf.patch`通过git apply --check。后台仍有ours运行，遵从源码冻结要求，**未合入生产、未改正在运行的源码身份、未实现MGS持续训练、未启动正式重跑**。三个相关生产文件SHA复核不变。

注释涉及的具体轮数也已更正：Low-Rank100+200为项目预算；PEAKS作者默认100初始更新/2000总更新，项目持续200epoch=15800更新，实际约epoch18已补齐10000；OGAS论文主2D实验实际18000更新，项目15800更新，原仓库scheduler的200000不能当作已运行步数。持续或一次性结构有来源，不等于这些数值是作者原设定。

## 2026-11-19：ours17项完成，基线及全部评测指标重新汇总

本批ours17项已全部完成，队列0运行/0排队/17完成/0失败，最后LHS Pattern Support于北京时间17:44:50结束，内存守护未中止。对应用户此前询问的剩余12项/生成项均已结束；队列以`experiments/budding_ours_queue/seed42_20261119_all15/tasks.json`为准。

新工具`scripts/analysis/summarize_sampling_results.py`只读当前冻结协议`seed42_a866aef5d988`，纳入65项完整实验（48份基线/对照含OGAS两版本＋17份ours），另将5项历史未完成目录单列。报告[全部结果汇总](/home/users/zsy/mydata/budding_yeast/analysis/sampling_baselines_ours_20261119/report.md)；同目录`all_final_metrics.csv`为195组real/同池test/跨池test终评、全部71列（含来源字段）；`all_metrics.json`保留全部源嵌套评测字段，`all_round_metrics.csv`有735组逐轮/持续终评。130份逐样本CSV均值与JSON核对、8190张final real图、50份训练侧标签账本域检查、65项best/history一致性检查通过。不加载大轨迹或checkpoint，不重训，不改旧结果。

主要real结果（39变量物理空间；主Relative为逐样本逐变量、eps0.01、无上限）：

|池|方法/流程|MAE↓|MSE↓|Relative↓|Corr↑|
|---|---|---:|---:|---:|---:|
|mutation|Power/固定池|0.15030|2.25805|1.05698|0.90974|
|mutation|LCMD/固定池|0.16444|2.65932|12.88390|0.89108|
|mutation|ours Closure/固定池|0.16641|2.73847|1.37550|0.87655|
|mutation|ours Sensitivity/固定池|0.16173|3.09815|1.57301|0.89021|
|mutation|ours Context/固定池|0.18173|7.55667|1.46042|0.91469|
|mutation|Low-Rank/带权一次性|0.32224|14.20894|13.79462|0.77337|
|mutation|PEAKS/持续|0.19209|5.20494|12.75508|0.89922|
|mutation|OGAS公式修正后/持续生成|0.21762|7.86289|13.44570|0.87741|
|lhs|Power/固定池|0.41464|24.90815|1.96470|0.79346|
|lhs|ours Pattern Balance/固定池|0.40791|28.84106|2.61005|0.78881|
|lhs|ours Witness/固定池|0.46813|25.24701|1.56309|0.78586|
|lhs|ours Transport/生成|0.42858|32.69011|2.11357|0.78020|
|lhs|Low-Rank/带权一次性|0.52917|36.04224|2.09261|0.70018|
|lhs|PEAKS/持续|0.41310|27.50178|1.52318|0.79199|
|lhs|OGAS公式修正后/持续生成|0.44463|31.33072|2.28106|0.79140|

结论：当前ours没有全面胜过Power。mutation Context的Corr0.91469及完整轨迹均值0.28381有优势；逐株79/126的MAE、72/126的MSE优于Power，但#62/#124/#50的MSE增量合计解释93.59%的整体净MSE差额。Closure将#15的MAE从Power0.1377降至0.06317、Relative从7.420降至0.8513，并改善#124；但#78/#75/#59等退步，整体相对Power的MAE/MSE仍高10.72%/21.28%。不能仅凭几个badcase收益宣布整体创新有效。

同池mutation synthetic test上Pattern Support的Relative1.27052、Corr0.84984均是当前固定池方法最优，Context Relative1.31220也优于Power1.38114，但real未同步获胜。LHS Pattern Balance的real MAE比Power低1.62%，MSE/Relative却高15.79%/32.85%。生成Transport在LHS相对同流程Random-Generate的MAE/MSE低10.18%/14.78%，Relative高20.69%；mutation MSE高138.77%。因此有局部/部分指标收益，尚无跨池、跨指标稳定优势。所有对照仅seed42，不作统计显著性声明。

旧HGGS仍有上述GF及持续训练差异，GF隔离补丁未合入；Low-Rank为GradientEnergy适配、PEAKS为Regression-Reference，不能冒充作者原实验。OGAS修正前后均提取但分名，主对照采用公式修正后；不因修正后成绩未提高恢复错误公式。不同流程标签访问/历史曝光/训练步数不同，配对随机对照和精确目录在报告CSV。上述real诊断不进入采样代码，未来分数校准应使用合成train/validation。


## 2026-11-19后续：HGGS生产修正与Closure收益归因更正

前述“GF隔离补丁未合入”已被本次生产修正取代：GF分批持续训练、主模型只重置一次、MGS持续训练，冻结已揭示共同初始参考尺度。默认GF粗池4000→保留2000；GF六阶段合计600epoch，主阶段600epoch，分别报告成本。原生物规则、ODE及既有结果不改，新增实验将采用独立`_hggs<hash>`目录；本次没有新的正式精度成绩。

固定池Closure实际新增8000条中，仅8条来自完整四角，7992条为单点误差补位。此前#15指标改善的数值仍成立，但不能归因为四角局部响应已得到验证。生成Closure的8007次端点尝试中8004来自四角、3为补位，成功8000，整体指标更差，不能混用两者机制归因。候选真值接口明确拒绝、训练标签账本holdout重叠0；人工反复用real badcase改进方法不等于完全盲测，须在研究报告中说明。

实现、作者流程差异与测试见[HGGS持续训练修正与Closure信息审计](HGGS_CONTINUOUS_FIX_2026-11-19.md)。


## 2026-11-19：导师汇报用三个Ours与八个Baseline全指标导出

只读提取Sensitivity/Transport/Closure及Random、Power/SBAL、LCMD、PRE-Power、旧HGGS、Low-Rank-GradientEnergy、PEAKS-Regression、公式修正后OGAS；含池选/生成实际已完成组合，共主清单23项，另附PRE-TopK两池2项，总25项。报告[完整评测指标](/home/users/zsy/mydata/budding_yeast/analysis/selected_baselines_three_ours_20261119/report.md)，同目录ZIP/原嵌套JSON/75组终评71列CSV/303组逐轮指标/3150条real逐株指标/实验和预测图索引。25份终评JSON逐份核对，全部75个混淆矩阵计数核对，ZIP完整性检查通过；不加载轨迹，不重新训练，不改变旧结果。不存在的LHS Sensitivity/Closure不填0，HGGS新持续训练版暂无正式成绩。导出工具`scripts/analysis/export_selected_sampling_metrics.py`。


## 2026-11-20：统一结果目录与迁移

所有当前采样流程共用`utils/sampling/layout.py`生成`<模型>/<mutation或lhs>/<算法>/<可读超参>/seed_<主seed>`，结果根为`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling`；固定池、持续训练、生成式不再分不同主目录。Transport/Closure的pool与generate变体分名，旧HGGS归入hggs_legacy、旧OGAS归入ogas_legacy。哈希保留在科学元数据，目录不再生成哈希名。

共同输入/划分已移到`shared/pool100k/seed_42`；生成seed仍各42，实验主seed及派生流在`seed_manifest.json`。不同训练配置可使用`configurations/<可读超参>`协议视图，输入和划分以硬链接共用，无第二份轨迹。已有协议的科学指纹和内容不改；共享池或划分设置发生变化须使用新的共享数据名称，不能覆盖。

最新冻结协议全部65项完成实验已同盘rename迁移；原配置/权重/指标字节不改，旧路径留兼容软链接，避免历史绝对引用失效。35,907个常规文件inode/大小/mtime逐项一致，341个链接及231份逐轮数据引用可访问。更早其他协议保留历史状态，没有混入本次最新65项。队列归入batches，汇总归入summary；直接打开[结果导航](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/index.html)或[路径索引](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/experiments.csv)。迁移审计`batches/layout_migration_20261120/verification.json`。

统一选定方法批量入口（预览，不启动训练）：

```bash
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2
bash run_sampling_budding.sh extensions --suite all \
  --directory /home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/next_seed42 \
  --gpus 0 1 2 3 --dry-run
```

共23项=八个baseline×两池16项＋三个ours的实际适用池/流程7项。去掉dry-run才会启动；已有同名实验身份不匹配会拒绝覆盖。此轮没有启动正式训练。

验收补充：覆盖80项不同CPU检查通过（71项原检查直接通过，5项旧路径断言更新后通过，另4项新增目录检查；最后19项复测包含重叠，不重复累加）。全部65份原始嵌套指标逐项完全相同；101个受保护生物/data文件SHA变化0。正式训练未启动。


### 2026-11-21：采样脚本更名及历史目录归档

生产入口已更名为`run_sampling_budding.sh`和`run_sampling_fission.sh`，调度器、README、文档和测试引用同步更新，不更改各系统现有算法/训练设置。Budding仍以`extensions --suite all`预览/启动统一任务清单；Fission仍为原有流程，更名不代表已接入Budding的新完整池协议。

当前结果根的三个旧seed哈希目录不是重复空目录：02e约9.921GiB，bb0约9.795GiB，a866残留约0.576GiB。已同盘移至`experiments/archive/sampling_legacy_paths/`；原位置为兼容链接，不能在仍有历史绝对路径引用时当垃圾直接删除。归档22,663个常规文件inode/大小/mtime保持，65项新结果及旧路径均可访问；未删除实验数据，也没有释放约20GiB空间。清单`batches/layout_migration_20261120/archive_cleanup.json`。


### 2026-11-21：按授权精简历史结果并移除旧哈希入口

已删除三个归档批次中的旧权重和预测图（21,126个文件，逻辑大小21,434,413,555字节，约19.96GiB），保留配置、指标JSON/CSV、训练历史、日志与数据文件。当前结果371份JSON中的路径已重定位，272个符号链接已规范化，25份完成记录的受影响校验值同步更新；修改前原文保存在`batches/history_prune_20261121/before`。模型权重、轨迹与评测值不更改。三个旧seed哈希根入口已移除；历史归档保留精简记录，旧目录不再用于访问当前结果。

根目录现在仅batches/fno/shared/summary，后续其他模型与fno平行。Budding统一入口的run/dynamic/generate/batch以及extensions默认均写入该根；显式覆盖output/results-root或选择历史online模式仍可指定其他位置。清理审计`batches/history_prune_20261121/verification.json`，当前汇总`summary/all_current/report.md`，导航`summary/index.html`。汇总工具已改为按迁移清单发现真实新路径，不再依赖已移除的旧路径。

清理后验证：231份逐轮选择清单均通过源哈希和首行参数/初值读取；Power和生成式Closure实际CPU重载权重并前向输出有限；65项原始嵌套指标逐项一致，8190幅当前real图仍在。指定baseline/ours报告和ZIP已用新真实路径重新导出，未启动训练。
