> **2026-11-19 后续HGGS修正已合入生产**：旧ours 17项已全部完成，确认没有运行中源码冻结冲突后，接入GF五批补选与持续辅助训练、GF后主模型只重置一次、MGS持续优化器/调度器训练；参考尺度固定为已揭示共同初始训练统计。新目录使用独立`_hggs<hash>`，旧结果不动，尚未正式GPU重跑。Closure固定池实际只有8/8000个新增端点来自完整四角，其余单点补位，不能将#15收益归因于四角机制；无候选/测试真值用于其评分的证据。详见[修正与信息审计](docs/HGGS_CONTINUOUS_FIX_2026-11-19.md)。下方运行中/未合入状态为历史记录。

> **2026-11-19 用户要求剩余5生成项立即启动**：当前主管`experiments/budding_ours_queue/seed42_20261119_all15/`，沿用tmux与结果根。接管原10项PID/内存中进度不变，5生成项已全部启动；2完成/15运行/0待启动/0失败，GPU0/1各8/7项。启动器admit-all-pending只解除整项准入等待；新增任务复用c10的同一resource domain，不增加第三套I/O/CPU锁。原A域I/O4＋B域I/O4总最多8；B域重评分仍5项×4worker，旧A两项各8worker。100GiB系统可用底线及统一内存守护不变。只改scripts/launch启动逻辑，冻结utils/models/data sources不变，101个生物/根data文件SHA核对不变，17项定向回归通过；实际阶段可能等待共享I/O/评分槽位，不能宣称15项同时GPU训练。验证all15_verification.json，详见docs/OURS_CONCURRENCY_2026-11-19.md末节。

> **2026-11-19 用户授权提高至10并发，已实际到10**：新主管队列`experiments/budding_ours_queue/seed42_20261119_c10/`，沿用tmux及结果根。通过handover接管原4个实验，PID/内存中评分不变，不重训/不重算。仅改scripts/launch启动/接管工具，utils/models/data的sources身份保持原冻结值。6→8→10逐步扩容，GPU0/1每卡最多5；长任务总最多5，原2项各8核/新增各4核，合计最多28个评分CPU worker；继承/新增重加载域各4、合计I/O最多8。统一100GiB系统可用底线，无自有PSS硬上限。16项接管/队列检查通过；实际2完成/10运行/5排队/0失败，新增LHS已训练；加压PSS约135GiB/系统可用约550GiB/IO full PSI≤0.05%，CPU负载约64。此为当前加压窗口，不是所有后续峰值保证。详见新status/resources与docs/OURS_CONCURRENCY_2026-11-19.md。

> **2026-11-19 ours执行提速切换**：已停止旧队列后切换ordered spawn CPU评分，正式每长任务8核/单worker单线程，同时最多2长任务；普通评分独立锁/最多2项，总4实验仅GPU0/1、I/O4、系统可用100GiB底线不变。新队列`experiments/budding_ours_queue/seed42_20261119_accelerated/`，原tmux/结果根沿用。旧2项完成不动，4项未完成仅执行代码身份做定点审计迁移，36份模型/历史/选样文件SHA核对保留；未保存的部分评分重算。生成轮内缓存训练统计，新轮refresh失效。64项相关回归通过；已测512候选分数/特征逐位一致，12组选样一致，评分预热后4.6–8.1倍；不是整场实验加速承诺。详情docs/OURS_SCORING_ACCELERATION_2026-11-19.md。启动后继续冻结sources，不得热改；当前进度看新队列status.json。

> **2026-11-19 ours正式队列已启动**：tmux `budding_ours_seed42_20261119`，仅GPU0/1、每卡2项/总4项，17任务=当前7个ours的12项适用池内组合＋新3个ours的5项生成组合；原FNO/seed42/2k→4k→6k→8k→10k/100-100-100-100-200配置复用。结果`/home/users/zsy/mydata/budding_yeast/experiments/budding_ours/seed42_20261119/`，队列`experiments/budding_ours_queue/seed42_20261119/`的tasks.json及startup_verification.json记录精确路径和已核实epoch。W&B offline、100GiB系统可用底线/IO4/评分2；普通失败继续排队。固定池train.log由队列侧publish_logs.py链接，不复制日志、不修改运行中source。启动前14项定向检查及两池5项真实checkpoint数值探针通过。已有4项实际GPU训练，未宣称全部完成；禁止运行中再改sources身份代码。

> **2026-11-19 Pattern-free ours 首版**：`ours_mechanism_witness`、`ours_dynamics_transport`、`ours_intervention_closure` 接入当前 run/generate；模块在 sampling/ours/mechanistic，四角合法编译在 generation/controlled。共51个不同相关检查通过，101个原生/data根目录文件SHA不变；正式FNO的8候选CPU评分约1.1–1.7秒/RSS0.85GiB，不是90k吞吐结论。旧Pattern/sensitivity明确Budding-only；通用内核已验证Fission RHS/单位/模式，完整池及生成实验服务仍Budding，不能宣称Fission新CLI已贯通。旧TEM1 invariance与四点交互已存在，不重复算新贡献。详情docs/OURS_IMPLEMENTATION_AND_PORTABILITY_2026-11-19.md。本轮未启动长实验；此前OGAS两池队列均已complete，11月18日进行中记录是历史状态。

> **2026-11-18 OGAS修复后已授权tmux重跑**：`budding_ogas_seed42_20261118`，仅mutation/LHS×ogas_async_projected、FNO/seed42、原冻结协议。新根`experiments/budding_ogas_alignment/seed42_20261118/generation/`，队列`experiments/budding_sampling_extensions/seed42_ogas_alignment_20261118/`；0–3卡准入、实际1与0、2并发/每卡1、100GiB系统可用底线、IO4/评分2、W&B offline。不得运行中再改sources身份代码；目录和进度以tasks.json为准。原46项标量/CSV/图片重新验收通过，报告analysis/baseline_status_20261118_ogas_alignment/report.md。下方“未启动”为修复阶段历史状态。

> **2026-11-18 OGAS作者源码对齐修复**：共用CFG公式改为`eps_u+w*(eps_c-eps_u)`；v2密度校正区分判别器类别先验0.5与反馈小批实际reference比例，按作者`max(pi_u,1/m)`上限并单位均值归一化。布尔来源从随机探索/引导/随机补位经candidate metadata、仿真、replay、后台反馈/FIFO/恢复贯通；旧无来源controller状态明确拒绝恢复。生物/data根目录原文件不改，不覆盖历史结果、不启动正式训练。详见docs/OGAS_ALIGNMENT_FIX_2026-11-18.md及papers目录源码审计；其他baseline的论文/源码及任务适配差异不能因此统称已消除。

> **2026-11-18 用户注释后的8项baseline建议**：固定池按机制选代表：Power/SBAL(TMLR2023)、LCMD(JMLR2023)、BAIT(NeurIPS2021)，不能因AL4PDE收录重标2025。主清单建议Random/Power/LCMD/BAIT/PRE/PICore/HGGS/OGAS，PICore仍待实现；OED为生成替补。区分OGAS从零工程成本高与当前已实现后增量成本低，实际OED逐条查询比异步OGAS更慢；DAS-PINNs采的是collocation点，不能直接当参数到全轨迹查询。详细比较docs/SAMPLING_BASELINE_SHORTLIST_2026-11-18.md第9节。修复队列5完成/0失败，基本终评盘点见analysis/sampling_repairs_20261118/completion_summary.json；本轮未改生产或启动新训练。

> **2026-11-18 baseline主表核查**：现有数量/近年性/两类覆盖足够，建议Random/QBC/LCMD/PRE/HGGS/OGAS约6家族；PRE为2026 ICLR AI/PDE Workshop，不能称主会；HGGS为AAAI2025、OGAS为NeurIPS2026。未实现PICore（TMLR2025，作者代码）列为最高相关性新补充；MaxDet/kmeans++/GradMatch/CRAIG/CREST等仍无完整独立入口。HGGS当前GF初始与共同2k仅重叠mutation380/LHS407、预读10k；ERACS从2k扩成10k resident，不能宣称所有流程同起始。报告docs/SAMPLING_BASELINE_SHORTLIST_2026-11-18.md；本轮只读核查和文档，未改生产/运行实验。

> **2026-11-18 13:30:22 修复后5项已在tmux正式补跑**：`budding_repairs_seed42_20261118`；队列`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling_extensions/seed42_repairs_20261118/`，新结果`experiments/budding_sampling_repairs/seed42_20261118/`。仅旧失败HGGS×2/Lev×2/LHS PRE-Power，FNO/seed42/原协议不变，新目录重跑，旧25项不动。GPU0–3最多5并发、每卡2、加载4/评分2、系统100GiB可用底线及磁盘关联IO守护、W&B offline。启动器失败筛选7项回归通过；正式完成状态看tasks.json，不能提前称完成。下方“未正式补跑”为此前修复时状态。

> **2026-11-18 baseline故障已定向修复**：HGGS跨轮target覆盖及旧错误缓存迁移、Lev浮点矩阵求解/重复ID、Physics-PRE过度事件padding已修复。真实90k/池重放：旧Lev唯一9996/9946→新各10000；失败LHS PRE的82000候选重放选出2000，行96195缩窗至0保留22个安全模板，未跨跳变求导。ERACS增加threshold/trigger_rule/机制汇总及0触发警告，默认0.1及不等号不改；正式规模阈值校准仍待做，不能称旧0触发结果验证了OMP。48项修复及动态/生成恢复回归通过；82生物文件及两池manifest SHA不变。未正式补跑、未覆盖旧结果；不得绕过源代码身份校验复用旧失败目录。详见docs/BASELINE_REPAIRS_2026-11-18.md。下方“未修生产代码”为此前分析时状态。

> **2026-11-18 补跑队列已结束并只读分析**：30项中25完成/5失败，墙钟8:47:17，PSS峰120.20GiB、最低系统可用409.97GiB，守护未中止。失败为mutation/LHS HGGS（refresh的target变量覆盖）、两池Lev（ID/预算合同；CPU另见Cholesky不正定，未证明同根因）、LHS Physics-PRE-Power（无有限连续残差区间）。未修生产代码、未自动补跑。PEAKS同流Random的MAE改善mutation3.41%/LHS2.96%，但访问约6.8万标签；ERACS39次检查/池均0重选，不能视为OMP动态机制有效性验证。OGAS仅LHS优于异步Random，OED无稳定收益；原mutation-Power综合仍最好。25项终评及原16项输出/best核对通过，17份候选标签账本无训练侧之外ID；5模型126real CPU复核匹配，CDH1T贡献PEAKS/ERACS Relative约88%/84%。生物82文件SHA变化0。分析文件analysis/sampling_extensions_20261118/report.md及CSV；下方启动中为历史状态。

> **2026-11-17 20:47 seed42补跑已启动**：tmux `budding_extensions_seed42_20261117`；队列`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling_extensions/seed42_20261117/`，`tasks.json`给出逐项结果目录。30项=两池各11种方法/变体+4个流程随机对照，旧完成8方法不重跑；FNO/seed42、0–3号卡先4后最多8、每卡最多2、系统100GiB可用底线、自有PSS不设硬上限、W&B offline。实际GPU小探针及最终6项队列测试通过；不代表所有长流程已完成。Low-Rank平方梯度能量是Algorithm2任务适配，不是原文梯度范数实验原样复现。生物82文件SHA变化0；有界RAM缓存减轻重复解压，动态入口嵌套I/O锁已修复。详见docs/SEED42_BASELINE_EXTENSIONS_2026-11-17.md。下方未启动等为历史状态。

> **2026-11-17 四类回归baseline已接入**：`dynamic --method peaks_regression/eracs_regression`、`run --method low_rank`、`generate --method hggs_projected`。PEAKS先初始训练再流式percentile/cache/new+old，回归selected-reference kernel替代分类logit；ERACS谱协方差SNR<tau触发带权Batch-OMP动态替换；Low-Rank Algorithm2在gradient-energy定义basis-loss，保留有放回重要性权重，不冒充唯一10k；HGGS有GF/pilot/global保留及GMM/MGS，目标投影后仅提交canonical原提案。独立代码，不执行参考库。新增17专项通过（两池/随机对照/真实ODE/训练及eval恢复）；此前联合48和旧流程6通过，存在重叠不可相加。生物82文件SHA变化0，未用GPU、未开正式实验。动态初始归一化冻结/持续optimizer，旧固定池五轮refit不改；源引用/新块一次/标签和历史曝光计费。详见docs/REGRESSION_SAMPLING_BASELINES_2026-11-17.md。不能套用旧16并发结论，下方“尚未完成四方法”为历史盘点。

> **2026-11-17 baseline出处更正**：Uncertainty是QBC（Seung/Opper/Sompolinsky，COLT1992）＋Top-K，AL4PDE2025是参考框架，不能当作新提出的2025方法。HGGS正式AAAI2025，2024仅本地预印本；符合近期要求但完整GF/MGS未实现。PEAKS/ERACS/LowRank仍待完整回归/动态/basis-loss适配，不能只因不同于五轮流程就拒绝；五项新增方法仅可按核查过的核心及适配范围表述，不能统称完整作者复现。详见docs/BASELINE_ATTRIBUTION_AND_FIDELITY_2026-11-17.md。本次仅核查与文档更正，没有修改算法/生物规则或启动实验。

> **2026-11-17 OGAS在线适配已接入**：新增`ogas_async_projected`与`random_async`，同次训练前向难度→有界CPU生成器线程→新合法提案投影→异步canonical ODE→FIFO replay；生物82文件SHA变化0，禁止改规则。在线依据11月12日OGAS v2修正密度权重/AdamW/日程，旧round入口明确v1。33项近期/在线CPU测试通过（含真实ODE、训练/评测中断恢复和best同步）；初始统计冻结、persistent训练、validation-best及合法投影均为明确适配，非Melissa/PDE完整复现，不套用固定池16并发结论。旧固定池五轮逐轮refit不变；未用GPU、未开正式实验。详见docs/BASELINE_FIDELITY_AND_OGAS_ONLINE_2026-11-17.md。下方“OGAS仅round/尚未接入online”为此前状态。

> **2026-11-17近期baseline已开始落地**：用户明确生物规则只复用不能修改；82个Budding/data根目录旧Python文件前后SHA一致。新增baseline/oed.py与ogas.py（Bayes后验/UCB、条件DDPM+CFG、密度比/反馈/探索）；generation/独立CLI/只读domain/policy/store/cohorts/runner，入口run_al_budding.sh generate。18项新增CPU测试通过，另有旧Physics-PRE/多轮定向回归通过；实际小规模canonical ODE→FNO→eval→恢复闭环、失败补采、QC源引用、损坏拒绝与默认1000/50扩散采样已验证。两池catalog/range与原LHS配置SHA相符；未用GPU、未启动正式多seed。OGAS仅`ogas_round_projected`轮次合法投影适配，不是原版异步在线全复现；HGGS/PEAKS回归版未完成。新增轨迹分块仅存一次，初始/累计用引用，政策恢复点最多2，ODE开始/结束/重试单独计费。不可直接复用固定池16并发结论。说明docs/RECENT_BASELINE_IMPLEMENTATION_2026-11-17.md；开题时间建议docs/BIBM_RESEARCH_SCHEDULE_2026-11-17.md（2027日程未公布，9月是目标）。

> **2026-11-17 最新baseline范围**：用户要求至少2～3项2025～2026近期方法，明确允许HGGS/OED/OGAS生成新输入，不能继续只因固定池限制拒绝。路线第10节以Physics-PRE2026/OED2026/OGAS2026为近期主对照、PEAKS2025补充；尚未实现/验证的不能提前算作有结果。LL=输出前隐藏特征，当前FNO128×T投影512；Physics-PRE已预测事件mask±4步，需分连续段/事件重置/初值核查，不能用未选真值事件时刻。新增事件增强项须独立命名，不冒充论文原版。生成类复用合法规则并报告全部ODE成本/在线训练对照。本轮仅更新方案，未启动仿真/训练。

> **2026-11-17 多seed前核查与baseline清单**：主80/辅助16轮全部完成且best正确；mutation best前四轮97～100、末轮183～200，LHS首轮54、末轮123～164，不得说所有池都在尾声或没有几十epoch的best。VeSSAL LHS追加中65.93%随机补齐（作者允许fill_random），mutation3.74%，需注明/进一步对账，不能称纯VeSSAL筛选。已读本地10PDF/14参考仓库，路线docs/SAMPLING_BASELINE_ROADMAP_2026-11-17.md：优先独立复现MaxDet/kmeans++/KMeans-Herding，现有Physics-PRE/Lev补跑，GradMatch作监督池方向；不import官方参考、不改变论文原流程冒充复现。本轮仅阅读/计划与CPU核查，未启动多seed或新增算法。

> **2026-11-17 批次已全部完成并分析**：seed42_a866aef5d988两池16项全部退出0，80轮、10080幅real图，best/history/ID切分核对通过。墙钟3:17:04，PSS峰151.96GiB、最低可用538.82GiB，守护未中止。mutation Power MAE0.15030/MSE2.25805/Relative1.05698最优；Random0.17746/3.44796/12.36778。Random/LCMD的Relative约89%/88%来自CDH1T零目标预测错误，不能当作全变量同比退化；Uncertainty #124贡献MSE78.84%。同池初始2000一致，末轮验证平台且best正确；LHS跨域输入背景常量Z-score外推异常必须单列OOD。仅seed42、没有ours结果，不因本次分析改变协议。新工具scripts/analysis/analyze_complete_sampling_batch.py；报告analysis/sampling_results_20261117/report.md及docs/5最新节。下方02:51启动中等为历史记录。

> **2026-11-17 最新资源验证及正式启动配置**：12/16项完整规模资源测试全部退出0；每项10000训练样本、2epoch及90k候选LCMD/BAIT/双模型推理。12项861.60秒/PSS131.86GiB/最低可用571.57GiB；16项1041.31秒/PSS178.76GiB/最低可用522.32GiB。16项采用available内存策略，12项为strict，非纯单因素比较；不代表五轮收敛或辅助ensemble训练已验证。正式离线batch默认总16／GPU0–3每卡4／加载8／评分4，系统有效可用100GiB底线，无自有PSS上限。全局IO PSI>5%还须设备busy>=90%且queue>=16或await>=250ms持续60秒才停止；设备信息缺失回退strict20秒。memory PSI仅预警，低于可用100GiB停止；CPU负载准入保护保留。runner候选评分已接入公共槽，等待耗时独立记录。W&B离线，协议seed42_a866aef5d988（修复池、逐轮refit、normalized-best、保留last）；旧结果不覆盖。资源报告resource_profiles/scoring_limited_12_16_20261117/report.md。下方更低并发及“未接入/运行中”均为历史记录。

> **2026-11-17 追查11月14日中午12点左右异常（用户更正）**：确认历史scratch探针concurrent.py遮蔽标准库，PyTorch→asyncio→concurrent.futures误执行该文件；模块顶层Popen两个resources.py子进程，子进程再次import torch形成递归启动，父进程90秒超时无清理。11:52启动、11:54父退出，后代日志最后写到13:14；14:53快照可用682GiB但load1=162.18。该缺陷是强过载解释，但没有12点进程/内存采样，且账号无权读取kern.log.1/syslog.1，不能断言触发OOM或唯一根因。原脚本已按原字节/哈希保全并隔离为concurrent_probe_20261114.py.disabled，静态/真实标准库导入恢复；未重放探针或启动GPU。报告analysis/incident_20261114_1200/report.md。临时入口不要使用concurrent.py/asyncio.py/multiprocessing.py等遮蔽标准库的名字，子进程启动须有main guard和异常清理；禁止通过重放递归链验证。

> **2026-11-16 用户新要求已完成：按剩余内存控制**：已取消BatchMemoryGuard固定96GiB自有内存上限，PSS仅记录，旧SAMPLING_BATCH_MAX_PSS_GIB也不再影响生产；可用内存128GiB底线及持续IO/内存PSI保护保留。模拟PSS200GiB/可用400GiB继续的回归通过，总22项通过。IO4/6/8同流程8项加载379.43/299.19/211.40秒，IO8比IO4缩短44.28%；默认总8项／每卡2项／IO8。IO4完整资源阶段含2epoch及90k评分，8项退出0，PSS峰90.22GiB、最低可用600.24GiB；IO6/8仅加载，按设计停止。全部是warm-cache单次诊断，无冷缓存/五轮吞吐结论；探针已释放，未启动正式重训。报告resource_profiles/io_scaling_available_memory_20261116/report.md。下方96GiB及IO3是此前历史规则，不能重新引入。

> **2026-11-16 并发复测完成**：新离线默认总8项／每卡2项／重加载3项，共享任务队列。IO2下8项完整规模2epoch训练及90k LCMD/BAIT/ensemble推理退出全0，PSS峰91.61GiB、最低可用578.13GiB；IO3加载比IO2缩短22.44%（739.23→573.33秒），仅warm-cache诊断，未覆盖IO3完整五轮或辅助训练。守护保留96GiB自有PSS/128GiB可用底线，新增IO full PSI>5%或memory some PSI>1%持续20秒只停止登记进程组。22项CPU回归通过；未启动正式重训，不保证冷缓存或其他用户突发负载下绝对安全。报告resource_profiles/io_concurrency_comparison_20261116/report.md及docs/BUDDING_REL_L2_BATCH_2026-11-15.md。

> **2026-11-16 数值修复已完成**：生产BDF默认1e-8/1e-10，已知TEM1总量异常同输入1e-10/1e-12重试，每次5秒最多两次；失败补采至足量，有限负值不丢弃。mutation3/LHS8共11条已发布`_numerical_qc/dataset.json`修补视图，原池不动，不复制全轨迹。新默认协议seed42_a866aef5d988，四缓存硬链接、输入/ID/切分及原126real参考保持；10项旧实验曾用错误标签，需区分数据修复与normalized-best迁移的重训范围。17项CPU回归通过，未启动正式训练。详见docs/NUMERICAL_POOL_REPAIR_2026-11-16.md。

> **2026-11-16 用户确认保留末epoch纯权重**：共享训练每轮最后epoch额外保存一次state_dir/last_weights.ckpt，含实际预处理契约、无optimizer/scheduler/callback状态；best仍rounds/round_XX/model.ckpt并用于后续选样。完成后仍删resume，meta保存best_epoch与last路径/epoch/哈希；缺失/改动last拒绝直接复用完成缓存。26项定向CPU回归通过，未给历史实验伪造last，未启动正式训练。复核生成哈希及磁盘数据：TEM1总量GAL参数范围[2,8]、部分初值约束确实存在；完整池仿真出口没有逐时间总量约束检查。异常为求解精度问题，源池暂未修改。

> **2026-11-16 最新全流程审计**：20万轨迹逐块审计发现11条数值疑点（mutation 3条TEM1明显越总量、LHS 4条越总量和4条明显负尾），严格复算11/11恢复且Pattern组未变。最严重mutation row38053原TEM1峰125.75→严格4.17、总量4.19，未被上午任何方法训练；11条均不在固定val/test，7条曾被部分方法选入。建议定点修正后冻结长跑，当前源池未改、未启动正式训练。已补raw预测/validation非有限保护及真实epoch LR日志（lr_next另存），未改优化/损失/预算，45项CPU回归通过。详见docs/EXPERIMENT_FLOW_AUDIT_2026-11-16.md；不得把schema审计通过等同全部轨迹无数值伪影，也不得宣称历史real主结果被Inf污染。记录126real已被反复观察用于开发反馈、单seed等方法学边界。

> **2026-11-16 最新checkpoint协议**：用户确认改用同池合成validation的归一化Relative L2最低值（`validation_metric=normalized_loss`、monitor=`val_loss_synth`），应用于主模型及辅助ensemble、逐轮评测和后续采样。每轮仍仅以累计已查询训练集重拟合P/Y归一化，轮内冻结并重新编码固定validation；物理验证Relative留作诊断，正式eval仍为物理MAE/MSE/Relative/Corr。epsilon0.01、无loss上限、梯度裁剪1及100/100/100/100/200epoch均保留。新协议`seed42_b5bff21cb8de`已从`seed42_02eebd57592f`派生，四个输入/切分缓存硬链接复用；新目录不覆盖上午16项。已核查两池首轮best均不同，模型相关方法要完整重跑；Random可复用原样本ID补训。本次只准备并验证代码/协议，没有启动整批正式训练。

> **2026-11-16 上午旧协议完成**：此前用户选择物理逐样本/变量Relative L2选best（`val_sample_channel_rel_l2`），只看同池合成validation；验证推理float32与独立评测一致，训练仍bf16。协议`seed42_02eebd57592f`，eps0.01、相同源池/初始集/切分，四个缓存硬链接复用旧协议。offline批次默认总6项、每卡最多2项、IO2；实测2→4→6预处理后6项训练/评测PSS峰值67.744GiB，最低可用645.465GiB。入口内置owned-child内存watchdog，96GiB PSS/128GiB可用底线；仅信号停止本批隔离进程组，普通失败按CONTINUE_ON_ERROR处理。GPU4--7独占规则保留。正式16项已全部完成、退出码全0，tmux已退出；新旧32项/160轮/20160幅图数量核查通过。统一重算real的smooth eps0.01 Relative L2后，mutation新配置平均MAE/MSE/RelL2分别退化15.9%/35.0%/20.3%，LHS基本持平。新Random/Power最终best选56/36epoch；小分母物理通道主导评分，不能把变差单独归因于loss。结果及边界见analysis/sampling_protocol_comparison_20261116/report.md。

> **2026-11-16 最新：恢复平滑分母，eps=0.01**：用户最终选择训练与物理评测均用 `sqrt(||target||_time²+0.01²)`，默认无loss上限、梯度范数裁剪1。训练仍在归一化空间，正式eval保留原始物理空间，数值相同不代表尺度相同。指标版本 `physical_sample_channel_rel_l2_smooth_eps001_v3`，输出schema v5；不与smooth eps0.1或max历史分数混算。checkpoint记录实际eps/规则，显式历史floor checkpoint仍可准确读取；新建训练均默认smooth eps0.01。此前max epsilon对照仅为历史诊断，不作为当前规则的验证。详见损失文档第12节。

> **2026-11-15 最终指令：训练与评测均默认不截断**：RelativeL2Loss及其生产复用项默认max_clamp=None；eps=.1、梯度范数裁剪1、非有限loss检查和既有归一化保护保留。旧pooled指标的归档实现与Python引用均已删除，历史结果不改写。历史诊断可显式指定cap，不能把它用于新训练默认。新checkpoint记录training_objective_definition实际eps/cap/空间。下方“训练cap未改/旧指标归档”均为此前阶段记录；当前规则见损失文档第10节。

> **2026-11-15 最新：仅用新物理Relative L2**：旧pooled指标已从生产计算／展示移除，仅归档历史复现。公共relative_l2_loss、Relative L2字段、CSV rel_l2和逐epoch物理监控统一为新sample/channel定义，不截断；新JSON/CSV/history有physical_sample_channel_rel_l2_v1版本，禁止与旧结果混算。训练默认cap10未动；用户要求的60epoch冷启动两臂对照已完整完成且两臂loss/梯度全部有限，详情见损失文档第9节。下方“旧pooled保留”是此前阶段记录，不代表当前生产实现。

> **2026-11-15 eval最终口径与cap对照**：用户明确最终eval精度指标只在物理空间计算。新增优先指标`sample_channel_rel_l2`：逐样本／变量沿时间求`||error||/sqrt(||truth||²+.1²)`再平均，无cap；旧pooled Relative L2保留。JSON/CSV/预测图/performance/W&B已接入；独立eval直接读原始真值，不用编码／解码重建标签。归一化训练loss只作训练诊断，不混作物理指标。训练默认cap10未改：同一BAIT异常状态的两分支20epoch恢复实验显示取消cap修复CDC14T，却使合成validation整体物理误差退化，不能直接全局移除。详见docs/3_LOSSES_AND_PHYSICS_CONSTRAINTS.md第7/8节。未覆盖旧正式结果、未重启正式批次。

> **2026-11-15纯Relative L2批次**：用户要求后续默认训练损失rel_l2（原hybrid=MSE+0.2rel）。新日志train/val/test归一化loss统一现有RelativeL2Loss，MSE另存诊断，物理评测/验证选checkpoint口径保留。GPU0–3共16项8baseline×2pool，最多4并行、2重加载、2绘图worker/项；watchdog自有进程树PSS上限96GiB、有效MemAvailable底线128GiB。源池不重生成，新协议seed42_bb035a147b38硬链接旧输入/划分，旧8项归档experiments/archive/budding_sampling_hybrid_mse_rel_20261115。详见docs/BUDDING_REL_L2_BATCH_2026-11-15.md。

> **2026-11-15 逐epoch监控**：用户要求恢复完整旧式loss信息。主模型每epoch只读评测同池test和real，日志含train/val_synth/test_synth/test_real/test_real_denorm及验证macro、LR/耗时。测试不参与选checkpoint/梯度/采样；checkpoint仍由验证macro选择。只读监控独立在utils/training/monitoring.py，分batch、不复制轨迹并恢复RNG；aux成员不重复评测。旧8项缺失的逐epoch测试值不能补造；仅新训练完整记录。详见docs/SAMPLING_OUTPUTS_2026-11-14.md末节。

> **日志与指标展示**：完整池结果排序real_mutants→synth_test→cross_pool_test；默认仅log_event阶段事件有时间戳，普通输出和EPOCH不自动加。EPOCH去重复别名、简化小数，CSV/W&B精度不改。已完成日志原文压缩保留于_state/train.original.log.gz；展示整理不得改指标数值或冒充重训。

> **2026-11-14 输出整理**：完整池benchmark每轮输出model.ckpt/history.csv/loss.png/dataset.json/selection.json/eval；eval含real及同来源合成test逐样本CSV。恢复/辅助模型/W&B集中_state，轨迹仅引用源池；不得恢复重复快照和summary JSON。原样式双轴loss准确区分validation与test。详见docs/SAMPLING_OUTPUTS_2026-11-14.md。

> **2026-11-14 用户新增基线约束**：正式采样baseline必须在项目内独立实现，不得import、动态exec或读取`baselines/sampling`参考代码执行；也不通过已安装作者包绕过此要求。官方论文/代码仅用于开发对照验证，参考源码不得随本轮修改。将来新增baseline同样遵守。当前正式实验已确认：FNO下Random/Uncertainty/LCMD-LL/BAIT-LL × mutation/LHS，共8项；验证显存、并发和实现后再启动。

> **2026-11-14 共享服务器保护**：用户报告并发I/O/内存压力影响登录。已在用户授权的6号GPU完成单进程完整规模资源探针：PSS峰值13.89GiB、GPU allocated峰值1.48GiB；6号卡已释放。默认从临时2/1调整到全局4实验、2重加载跨进程锁，动态检查有效MemAvailable、load、memory/IO PSI；批量子进程降低nice/ionice优先级。不得仅因显存够用增加并发，不清理全机page cache或终止其他用户任务。该限制是协作式调度，不能宣称已设置cgroup硬限额。诊断默认CPU单线程、CUDA不可见；GPU验证仅使用明确空闲且授权的卡。

> **2026-11-14 用户确认验证范围**：mutation模型仅以mutation validation选checkpoint，LHS模型仅以LHS validation选checkpoint，测试仍报告两个来源。避免mutation训练中90维固定背景导致LHS输入Z约2.5e7而支配共同验证均值。当前协议`seed42_18bf9a3289fa`，原split/initial/input缓存与上一协议硬链接同一inode，源池/归一化算法不改。

> **GPU独占规则（用户明确要求）**：4、5、6、7号卡每卡最多一个GPU计算进程，不能因显存空余启动第二个。同一进程内多模型可以顺序执行；绘图/监控子进程不能初始化CUDA。当前用户仅额外授权6号卡进行单项资源测试，正式实验仍按0–3号卡的既有约定。不要把6号卡用于同卡并发测试。

> **2026-11-14 解耦规范**：遵从用户最新要求，按职责独立组织而非追求文件最少。通用训练为`utils/training/runner.py`，当前流程适配为`utils/sampling/benchmark/rounds.py`；候选生成、算法、流程、训练、评测相互分离。不要新建模型专用训练器，也不要把全部服务塞进CLI入口。旧在线引擎在`utils/archive/online/`，通过兼容别名支持Fission/历史重放。当前接口以`utils/README.md`为准。

> **2026-11-14当前组织**：用户已确认统一采用`utils/sampling/`，作者基线放`baseline/`，项目方法放`ours/`。其余工具按`training/physics/evaluation/experiments`归类，旧Python模块路径由`utils/_compat.py`延迟路由，不再新增散落的旧模块。`run_al_budding.sh`默认完整池离线Benchmark，`SAMPLING_MODE=online`仅用于历史在线实验。下文1M池与旧实验默认值属于历史记录；当前协议及实现状态以`docs/BUDDING_SAMPLING_BENCHMARK_PLAN_2026-11-13.md`为准。

# AGENTS.md — 出芽酵母细胞周期神经算子项目 AI Agent 工作指南

> **适用对象**：面向所有介入本项目的 AI Agent（如 Claude, Gemini, GPT 等）与核心研发人员。
> **核心使命**：本指南确立了代码库的**绝对工程铁律**、**全类型代码存放规范**、**系统架构认知字典**与**防坑避雷指南**。任何 Agent 在执行增删改查任务前，必须严格遵守本文档所立之规范。

---

## 目录
- [一、AI Agent 开发铁律与全类型代码归宿规范 (MANDATORY)](#一ai-agent-开发铁律与全类型代码归宿规范-mandatory)
  - [1.1 绝对工程铁律：严禁在仓库根目录乱放临时脚本](#11-绝对工程铁律严禁在仓库根目录乱放临时脚本)
  - [1.2 全类型代码存放字典 (Where Code Belongs)](#12-全类型代码存放字典-where-code-belongs)
- [二、科学背景与预测任务定义](#二科学背景与预测任务定义)
  - [2.1 输入输出张量契约](#21-输入输出张量契约)
  - [2.2 核心挑战：Sim-to-Real 泛化与测试基准](#22-核心挑战sim-to-real-泛化与测试基准)
- [三、运行环境与执行指令 (Environment & CLI)](#三运行环境与执行指令-environment--cli)
  - [3.1 Python 解释器与关键依赖](#31-python-解释器与关键依赖)
  - [3.2 必须设置的 PYTHONPATH](#32-必须设置的-pythonpath)
  - [3.3 核心生产入口与流水线](#33-核心生产入口与流水线)
- [四、模型架构体系 (`models/`)](#四模型架构体系-models)
  - [4.1 统一 Forward 接口契约](#41-统一-forward-接口契约)
  - [4.2 标准基线模型体系 (`models/baselines/`)](#42-40-基线模型体系-modelsbaselines)
  - [4.3 30+ 原创生物算子体系 (`models/ours/`)](#43-30-原创生物算子体系-modelsours)
- [五、损失函数与物理约束体系 (`utils/training/losses.py`)](#五损失函数与物理约束体系-utilslossespy)
- [六、主动学习 (Active Learning) 体系 (`utils/archive/online/legacy.py`)](#六主动学习-active-learning-体系-utilsal_callbackpy)
  - [6.1 1M 免预仿真纯参数空间候选池架构](#61-1m-免预仿真纯参数空间候选池架构)
  - [6.2 当前在线策略与官方对齐状态](#62-当前在线策略与官方对齐状态)
- [七、数据生成、归一化与生物网络图谱 (`data/`)](#七数据生成归一化与生物网络图谱-data)
  - [7.1 归一化规范与最新实测证据](#71-归一化规范与最新实测证据)
  - [7.2 数据集生成引擎清单](#72-数据集生成引擎清单)
  - [7.3 多关系生物因果拓扑图](#73-多关系生物因果拓扑图)
- [八、关键避坑指南与科学细节 (Gotchas & Best Practices)](#八关键避坑指南与科学细节-gotchas--best-practices)
- [九、权威专著文档导航 (`docs/`)](#九权威专著文档导航-docs)

---

## 一、AI Agent 开发铁律与全类型代码归宿规范 (MANDATORY)

### 1.1 绝对工程铁律：严禁在仓库根目录乱放临时脚本
> [!CAUTION]
> **绝对禁止 AI Agent 在仓库根目录下直接创建任何临时探针、测试或调试 `.py` 文件！**（如 `_cap_probe_tmp.py`, `inspect_xxx.py`, `temp_test.py` 等）。
> 仓库根目录只允许保留核心生产入口（`train.py`, `eval.py`, `tune.py`）、批量运行脚本（`run_pipeline.sh`, `run_al_budding.sh`, `run_tune.sh`）以及权威文档（`README.md`, `AGENTS.md`）。

### 1.2 全类型代码存放字典 (Where Code Belongs)

当 AI Agent 根据用户需求编写不同类型的代码时，**必须严格存放在对应专属目录**：

| 代码类型 | 必须存放的目标目录 | 示例 / 说明 | 注册与合规要求 |
|:---|:---|:---|:---|
| **神经网络模型** | `models/baselines/` 或 `models/ours/` | 如新基线 `mamba_no.py`，新原创 `bio_refiner_v3.py` | 必须在 [models/yeast_lit_module.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/models/yeast_lit_module.py) 的 `MODEL_REGISTRY` 中注册，并提供标准 `forward_ic_time` 接口。 |
| **损失函数 / 物理约束** | `utils/training/losses.py` | 如新损失 `AdaptivePhaseLoss` | 必须在 [utils/training/losses.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/training/losses.py) 的 `LOSS_REGISTRY` 中注册，支持课程学习权重调度。 |
| **主动学习策略 / 物理残差** | `utils/sampling/` / `utils/physics/residuals.py` | 如新 AL4PDE 策略实现 | 作者基线在 `utils/sampling/baseline/`，项目方法在 `utils/sampling/ours/`；方法统一登记到 `sampling/registry.py`；当前选择器返回 SamplingResult，查询由 query.LabelOracle负责，流程调用公共训练/评测服务；历史 SelectionPlan/engine 仅在archive/online。`al_callback.py` 只保留兼容导入。 |
| **数据生成器 / 空间采样器** | `data/` | 如 `gibbs_sobol_sampler.py`, `bams_dataset_generator.py` | 保持模块化独立运行，支持 `--seed` 与 `--data_dir` 参数。 |
| **持久生产级分析 / 评测工具** | `scripts/analysis/` 或 `scripts/eval/` | 如 `analyze_experiments.py`, `eval_ensemble_benchmark.py` | 必须使用 `argparse` 提供标准命令行参数，长期保留供科研人员复用。 |
| **自动化单元测试 / 回归测试** | `tests/` | 如 `test_bio_refiner_causal.py`, `verify_training_loop.py` | 供开发验证与 CI 冒烟测试使用，已被 `.gitignore` 忽略。 |
| **AI 临时探针 / 调试沙盒 / 一次性脚本** | `scratch/` 或 `scratch/probes/` | 如 `_cap_probe_tmp.py`, `test_batch_residual.py` | **所有临时、探针、排查脚本强制存入此处，随用随弃**，已被 `.gitignore` 忽略。 |

### 1.3 绝对工程铁律：同步维护权威文档 (Mandatory Synchronous Documentation)
> [!IMPORTANT]
> **每当向代码库新增功能或产生新的实验结果时，必须在对应的 `docs/` 文档中同步更新：**
> 1. **新增模型/模块**：在 `models/` 编写并注册新模型后，必须同步在 [docs/1_ARCHITECTURE_AND_MODELS.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/1_ARCHITECTURE_AND_MODELS.md) 补充模型架构说明、输入输出、参数量与调用命令。
> 2. **新增主动学习策略**：在 `utils/sampling/ours/` 实现并注册新策略后，必须同步在 [docs/2_ACTIVE_LEARNING_GUIDE.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/2_ACTIVE_LEARNING_GUIDE.md) 补充数学设计、采样目标函数与实测性能。
> 3. **新增损失函数/物理约束**：在 `utils/training/losses.py` 新增损失后，必须同步在 [docs/3_LOSSES_AND_PHYSICS_CONSTRAINTS.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/3_LOSSES_AND_PHYSICS_CONSTRAINTS.md) 补充数学公式与梯度特性。
> 4. **新增数据生成器/归一化**：在 `data/` 编写新生成引擎或验证归一化后，必须同步在 [docs/4_DATASETS_AND_NORMALIZATION.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/4_DATASETS_AND_NORMALIZATION.md) 补充采样空间分布与物理基准。
> 5. **实验完成/评测出炉**：任何批次实验（如 `run_pipeline.sh`, `run_al_budding.sh` 等）运行完毕后，必须运行分析脚本提取关键指标，并同步更新至 [docs/5_BENCHMARK_AND_LEADERBOARD.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/5_BENCHMARK_AND_LEADERBOARD.md)（与对应的专门指南中），严禁只在终端汇报而导致专著文档脱节！

### 1.4 AI 专用分析、探针与评测工具库导航 (AI Tooling Navigation Guide)
代码库为 AI Agent 和科研人员配备了多层次的诊断工具链，当遇到性能瓶颈、收敛异常或需开展归因分析时，**优先复用以下现有成熟工具，避免重复造轮子**：

#### 1. 深度实验分析工具库 (`scripts/analysis/`)
- [analyze_experiments.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/analyze_experiments.py)：自动化多实验横向对比，可自动遍历多级目录提取 `global_metrics_denorm.json`，生成 Markdown 排行榜表格与 CSV 矩阵。
- [test_ko_perturbation_sensitivity.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/test_ko_perturbation_sensitivity.py)：基因敲除参数微扰敏感度分析器，探测模型在敲除边界条件（如连续 $[0, 10^{-4}]$ 微扰）下的梯度平滑性与鲁棒性。
- [evaluate_sampling_efficacy.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/evaluate_sampling_efficacy.py)：定量评测采样策略在高维参数空间中的空间覆盖率（Space-Filling Index）、均匀度与多样性。

#### 2. 数据集高维几何诊断与 Badcase 根因归因库 (`scripts/analysis/dataset_diagnosis/`)
- [analyze_pure_random_badcases.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/dataset_diagnosis/analyze_pure_random_badcases.py)：纯随机训练集泛化至 126 突变体的 Badcase 详细逐通道时序对比与根因诊断。
- [analyze_massv2_badcase_attribution.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/dataset_diagnosis/analyze_massv2_badcase_attribution.py)：MassV2 等大样本数据集的坏例数据归因分析（分析特征邻域、支撑集稀疏度与相空间流速）。
- [deep_dataset_diagnosis.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/dataset_diagnosis/deep_dataset_diagnosis.py)：深入诊断训练集参数流形覆盖与支撑集完整度。
- [deep_genotype_distribution_analysis.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/dataset_diagnosis/deep_genotype_distribution_analysis.py)：对比合成数据集与 126 真实实验突变体的基因型频率、敲除模式重叠率与 Sim-to-Real 覆盖盲区。
- [generate_multimetric_top20_badcases.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/dataset_diagnosis/generate_multimetric_top20_badcases.py)：多指标综合打分生成 Top-20 最具代表性的真实突变体坏例，并输出高分辨时序曲线与误差热图。

#### 3. 评测与突变体验证工具库 (`scripts/eval/`)
- [eval_ensemble_benchmark.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/eval/eval_ensemble_benchmark.py)：多模型集成（Ensemble）推断与评测，对比单模型 vs 集成在 126 突变体上的方差缩减与鲁棒性提升。

#### 4. 自动化单元测试与算子验证 (`tests/`)
- [test_bio_refiner_causal.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/tests/test_bio_refiner_causal.py) 等：涵盖因果掩码单向可达性验证、维度对齐校验、前向与反向梯度流通测试。

#### 5. AI 临时探针与调试沙盒 (`scratch/` & `scratch/probes/`)
- 当 AI Agent 需要针对特定模块编写单次排查脚本时，统一写入 `scratch/probes/`，支持快速验证，不会污染版本控制。

---

## 二、科学背景与预测任务定义

这是一个**基于神经算子（Neural Operator）的代理模型（surrogate model）项目**，用于高精度预测出芽酵母（*Saccharomyces cerevisiae*）细胞周期动态演化。

### 2.1 输入输出张量契约
- **输入参数 $P$**：141 维连续/离散动力学参数向量（覆盖 141 个反应速率常数、合成速率、降解速率及基因敲除/过表达覆盖），形状 `(B, 141)`。
- **输入初始条件 $X$**：39 维分子物种的初始浓度及其一阶时序差分，形状 `(B, 39, 2)`。
- **输出轨迹 $Y$**：39 维分子浓度在 500 个时间步（跨度 210 min）上的演化轨迹，形状 `(B, 39, 500)`。
- **39 个分子物种 (`VARIABLE_NAMES`)**：
  `MASS`, `CLN2`, `CLB2`, `CLB5`, `SIC1`, `CDC6`, `C2`, `C5`, `F2`, `F5`, `SIC1P`, `C2P`, `C5P`, `CDC6P`, `F2P`, `F5P`, `SWI5T`, `SWI5`, `IEP`, `CDC20T`, `CDC20`, `CDH1T`, `CDH1`, `CDC14T`, `CDC14`, `NET1T`, `NET1`, `RENT`, `TEM1`, `CDC15`, `PPX`, `PDS1`, `ESP1`, `ORI`, `BUD`, `SPN`, `Vi20`, `lte1`, `BUB2`。
- **9 个核心评估变量 (`CORE_9_INDICES`)**：`[0, 1, 2, 3, 4, 20, 22, 33, 35]`（对应 MASS, CLN2, CLB2, CLB5, SIC1, CDC20, CDH1, ORI, SPN）。

### 2.2 核心挑战：Sim-to-Real 泛化与测试基准
- **训练集**：由 ODE 数值积分器（`solve_ivp`）在纯随机/准随机空间填充分布上生成的数值仿真样本（数万至数十万量级）。
- **测试基准（Ground Truth）**：**126 个具有生物学真值实验验证的酵母突变体（Real Mutants）轨迹**。
- **核心矛盾**：模型只能在合成参数分布上训练，但必须在 126 个真实实验突变体上精准泛化（解决 Sim-to-Real Gap、相变悬崖与双稳态阻滞伪振荡）。

---

## 三、运行环境与执行指令 (Environment & CLI)

### 3.1 Python 解释器与关键依赖
- **Conda 环境**：`neuralop` (Python 3.10)
- **解释器绝对路径**：`/home/users/zsy/.conda/envs/neuralop/bin/python`
- **核心依赖**：`torch`, `pytorch_lightning`, `scipy`, `scikit-learn`, `optuna`, `wandb`, `ptwt` (小波), `torchdiffeq`。

### 3.2 必须设置的 PYTHONPATH
所有内部模块均采用 `budding_yeast_v2.` 前缀规范导入，因此执行脚本前**必须确保父目录在 `PYTHONPATH` 中**：
```bash
export PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle:$PYTHONPATH
```
*注：[train.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/train.py)、[eval.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/eval.py) 及所有 shell 脚本已内置头部自动探测修正逻辑。*

### 3.3 核心生产入口与流水线
```bash
# 1. 单模型基线训练 (例如 BioRefinerPlus + HybridMSERelLoss)
python train.py \
    --model biom_refiner_plus \
    --loss_type hybrid_mse_rel \
    --dataset_name /home/users/zsy/mydata/budding_yeast/gibbs_boltzmann_100000_210m_500s.npz \
    --modes 24 --width 192 --batch_size 256 --max_epochs 100 \
    --lr 8.8e-4 --warmup_epochs 8 --norm_type log1p --al_strategy none

# 2. 1M 纯随机池 AL4PDE 闭环主动学习批量运行
bash run_al_budding.sh

# 3. 评测与 126 真实突变体轨迹绘制
python eval.py \
    --ckpt experiments/al_pure_random_1m/biom_refiner_plus/hybrid_mse_rel_al4pde_bifurcation/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/checkpoints/best.ckpt \
    --dataset_name /home/users/zsy/mydata/budding_yeast/gibbs_boltzmann_100000_210m_500s.npz \
    --save_dir experiments/eval_output

# 4. 全局实验指标解析与排行榜生成
python scripts/analysis/analyze_experiments.py --exp_dir experiments/al_pure_random_1m
```

---

## 四、模型架构体系 (`models/`)

### 4.1 统一 Forward 接口契约
所有模型在 [models/yeast_lit_module.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/models/yeast_lit_module.py) 中由 `forward_ic_time(x, p)` 统一调度：
- `ic_time_grid: (B, 39, 2, 500)`，包含初始浓度与时间网格。
- `params: (B, 141)`，默认是原始动力学参数逐参数 Z-score，**不是 Min-Max**。
- 返回预测张量：`pred_norm: (B, 39, 500)`。

### 4.2 标准 baseline 体系 (`models/baselines/`)
- 只保留用户采纳的标准骨干：`mlp`, `resnet`, `unet`, `kan`, `transformer`, `mamba`, `deeponet`, `fno`, `wno`, `mwt`, `gnot`, `transolver`, `lno`, `neural_ode`, `sc_fno`, `bregman_fno`, `lamo`, `konode`。
- 禁止给 baseline 添加原方法没有的参数调制、图先验、额外跨注意力等模块；只做参数、完整初值和查询时间的输入输出适配。保留原方法自身的注意力、专家和状态空间机制。
- 已删除的 `pure_*` / `*_ref` / `fno1d` / 时序预测等旧入口不再兼容；不得重新引入别名把修改版冒充标准实现。
- SC-FNO 必须使用 `sc_sensitivity` 及严格匹配数据/归一化的参考灵敏度文件；无该监督时不得以 SC-FNO 训练。
- ours 的原 WNO/LNO 使用 `cross_wno` / `cross_lno`。来源和验证见 [BASELINE_SELECTION_AUDIT.md](docs/BASELINE_SELECTION_AUDIT.md)。

### 4.3 30+ 原创生物算子体系 (`models/ours/`)
- **`BioRefinerPlus` (`biom_refiner_plus`)**：分层生物通量感知 + 多关系因果图卷积细化器（当前 SOTA 主力）。
- **`BioRefinerAttractor` & `BioRefinerAttractorPlus`**：相空间吸引子流形锁定算子（解决阻滞态伪振荡）。
- **`BioRefinerCausalReach` & `BioRefinerCausal`**：生物因果单向可达性门控网络。
- **`HANO` 系列**：分层自适应神经算子（多尺度时空谱图分解）。
- **`BioMassActionRefiner`**：质量作用定律（Mass Action Law）物理通量引导细化器。

---

## 五、损失函数与物理约束体系 (`utils/training/losses.py`)

[utils/training/losses.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/training/losses.py) 注册了 17 种现代损失函数（统一通过 `--loss_type` 调用）：

| 损失名称 (`loss_type`) | 对应实现类 | 核心组成与设计目的 |
|:---|:---|:---|
| `mse_only` | `MSEOnlyLoss` | 纯均方误差基线。 |
| `rel_l2` | `RelativeL2Loss` | 相对 $L_2$ 误差损失（规避大动态范围主导）。 |
| `hybrid_mse_rel` | `HybridMSERelLoss` | $\text{MSE} + \lambda_{\text{rel}} \text{RelL2}$（兼顾绝对量纲与相对波形）。 |
| `gradnorm_hybrid` | `GradNormHybridLoss` | 基于梯度范数的自适应多任务动态平衡损失。 |
| `soft_dtw` | `SoftDTWLoss` | 可微动态时间规整损失（对齐相位偏移与时滞）。 |
| `phase_derivative` | `PhaseDerivativeLoss` | 相平面一阶时间导数对齐损失（抑制高频伪振荡）。 |
| `triple_dynamic` | `TripleDynamicLoss` | $\text{MSE} + \text{RelL2} + \text{Derivative}$ 三位一体综合动态损失。 |
| `hybrid_bifurcate` | `HybridBifurcateLoss` | 分岔敏感损失（针对双稳态相变区域施加自适应权重）。 |
| `pinn_residual` | `PINNResidualLoss` | 向量化 ODE 物理微分残差 + 软事件掩码约束。 |
| `h1` | `H1Loss` | 频域 Sobolev $H^1$ 梯度平滑惩罚。 |
| `pino` | `PINOLoss` | PINN 物理残差 + 频域 $H^1$ 联合约束。 |
| `hybrid_pino_collocation`| `HybridPINOCollocationLoss` | 配点自适应物理残差损失。 |
| `hybrid_all` | `HybridAllLoss` | 数据驱动 + 导数 + DTW + 物理残差全能复合损失。 |

---

## 六、主动学习 (Active Learning) 体系 (`utils/archive/online/legacy.py`)

**当前组织**：`utils/sampling/`只保留算法、契约、注册和流程；旧engine/pool/ODE查询器及兼容入口隔离在`utils/archive/online`。新规则通过统一受限视图接入，不在算法中调用训练、写数据或查询real。以下6.1–6.2主要描述历史在线协议。`tune.py` 当前明确仅支持 `--al_strategy none`。接口、归档与验证见 [拆包说明](docs/AL_PACKAGE_REFACTOR.md)。

### 6.1 1M 免预仿真纯参数空间候选池架构

**2026-10-30 存储更新**：新AL采用`shared/datasets/<sha>.npz`共享初始块、每次`added_samples.npz`及每轮`dataset.json`清单，不再保存重复累计NPZ。轨迹float32、物理参数/实际初值float64和训练输入float32/bf16保持；数据读取使用`data.dataset_archive.open_dataset`，不要直接对清单调用`np.load`。checkpoint仍严格检查同一数据哈希/切分/归一化；迁移必须逐字段验证一致后才释放旧快照。详见[存储布局](docs/AL_DATASET_STORAGE_LAYOUT.md)。下方旧快照描述保留为历史。
AL 与静态批训练的新实验结果默认根目录为机械盘 `/home/users/zsy/mydata/budding_yeast/experiments/`，两系统 AL 分别使用其下 `al_budding/`、`al_fission/`。`EXP_ROOT` 可覆盖，不把新结果默认写回项目盘。共享输入路径仍由 `DATA_DIR` 控制，Budding默认文件名为`_balanced.npz`，Fission为`_al.npz`，规则版本保留在内部元数据；用户授权删除的旧 AL 根目录输入不再使用。
当前 `run_al_budding.sh` 用 `data/generate.py --recipe budding_pool` 生成参数/初值池，`data/systems/al_initial.py` 按层 KMeans 选取并仿真初始集；之后由 `train.py` 驱动选样、在线 ODE 仿真和外层冷启动训练。两个 AL 脚本均内置 `SEEDS` 和 `EXPERIMENTS` 清单，每行指定模型、损失、策略、超参数和 GPU；Budding有界同卡并行、Fission同卡串行，均跨卡并行，先准备一个共享池及每个 seed 的初始数据，再启动 worker。Budding当前默认FNO的6个基线（Random/Uncertainty/Power/MaxDist/LCMD/BAIT）加ours_response，共7项，GPU0–3任务数2/2/2/1，MAX_JOBS_PER_GPU默认2（设1则串行），Fission仍同卡串行；Fission默认标准fno四策略4项（Random/Uncertainty/Power/BAIT，GPU0–3）；AL W&B 强制本地离线，池大小为 100 万/50 万。Budding候选池使用budding/al_pool.py两级容量配额和逐模式Sobol流（百万池L1/L2/L3=9815/126225/863960）；Fission候选池多核提议仍使用parallel_pool.py（固定256条块、SeedSequence、全局去重）；SIM_WORKERS默认32，新并行RNG契约区别于旧串行流，详见 AL_EXPERIMENT_WORKFLOW.md。Budding当前FNO统一50/50/50/100 epoch、warmup4/4/4/8、batch512、LR1e-3；其他Budding模型和AL方法以此为参考，容量仍按架构定义。Fission默认60/60/60/120 epoch、warmup6/6/6/12不变。标准 FNO width 为 Budding192/Fission128、modes24、4主干块；当前默认清单仅FNO，不以cross_fno的每变量width推导其容量。批次 `tasks.tsv` 按完成顺序记录成功失败，失败默认继续下一项，退出码仍为非零；CONTINUE_ON_ERROR=0不再启动该卡后续项，已运行项正常收尾。Budding ODE默认每实验4进程，可用AL_SIM_WORKERS覆盖。
Budding 仿真后幅值拒绝已取消，旧六种启发式策略的参数门控保持；新增ours_response不加该门控。当前四轮、初始 KMeans + seed 随机切分、逐轮累计 train 重拟合和快照 L 字段契约以 [AL_EXPERIMENT_WORKFLOW.md](docs/AL_EXPERIMENT_WORKFLOW.md) 为准；旧三轮/Pattern 分层/冻结尺度说明不适用于新 AL。Fission AL 已支持 Random/Uncertainty/Power/BAIT/LCMD 五种通用策略；其余 Budding 专用策略明确拒绝。候选初值必须逐株预热，入口与成本边界见 [Fission AL](docs/FISSION_ACTIVE_LEARNING.md)。

### 6.2 当前在线策略与官方对齐状态

**2026-10-31机制批次**：新增`ours_pattern_balance/context/support`及`ours_sensitivity`，辅助分类器在`utils/sampling/ours/auxiliary/`；核心只接收synthetic train和无标签pool视图。默认`AL_SUITE=mechanism`四项、`mechanism_v1_full`独立目录；旧清单可显式选择。四项mechanism及Event重跑已在2026-10-31完整完成，预算/辅助模型/审计核对通过，见[用时与结果](docs/AL_MECHANISM_COMPLETION_2026-10-31.md)及[实现与验证](docs/AL_MECHANISM_IMPLEMENTATION_2026-10-31.md)。
保留原 12 种在线策略：`al4pde_random`、`al4pde_uncertainty`、`al4pde_diversity`、`maal`、`al4pde_mass`、`al4pde_novelty`、`al4pde_residual`、`al4pde_residual_diversity`、`al4pde_bifurcation`、`al4pde_bifurcation_entropy`、`al4pde_event_entropy`、`al4pde_fisher_oed`；四个短别名继续兼容。

新增 `al4pde_power`、`al4pde_bait`、`al4pde_lcmd`，新增预测轨迹`al4pde_max_dist`后加上`ours_response`当前合计17种在线策略；Power 复用独立 ensemble，BAIT/LCMD 使用预测轨迹共享投影及实际 bmdal 选择器；LCMD 无 posterior transform。`coreset_maxdist` 尚未实现，不能用输出轨迹替代 latent 冒充支持。官方 8 个预置配置及未移植项见 [清单](docs/AL_POWER_BAIT_AND_OFFICIAL_INVENTORY.md)。

`random/us/is/wrs/vessal/hggs/pial` 已归档至 `utils/archive/al_callback_legacy.py`，主文件和训练入口不再支持这些旧轨迹池策略。归档包含完整原文件，生产代码不可导入它来绕过限制。

当前 uncertainty 已按官方两个独立训练成员和 MSE disagreement/原始方差分支对齐，见 [Ensemble 对齐](docs/AL_ENSEMBLE_UNCERTAINTY.md)；它是项目训练适配，未直接实例化官方 Task/ProbModel。Random 保持固定池。diversity 使用自定义 332 维特征及本地 MaxDist；Fisher 名称对应特征正交化而非 Fisher 信息矩阵；分岔和事件策略是启发式扩展。不要把名称当作官方算法等价性证明。

归档、数值审计与未解决问题见 [官方核对](docs/AL_ARCHIVAL_AND_OFFICIAL_ALIGNMENT_2026-10-21.md)。复查使用 `scripts/analysis/audit_al4pde_alignment.py --output scratch/probes/al_cleanup_2026-10-21/alignment.json`。

---

## 七、数据生成、归一化与生物网络图谱 (`data/`)

### 7.1 归一化规范与最新实测证据
- **当前默认 `norm_type=log1p`**：$Y$ 先 `log1p(max(Y,0))`，再逐变量跨样本/时间 Z-score；$P$ 为原始参数逐坐标 Z-score，**不是 Min-Max，也不是参数取 log**。
- 所有工程路径共用 `data/normalization.py`。`asinh_z/asinh_scale/linear` 为显式选项，禁止在消费者内部写死另一种逆变换。
- **2026-10-19 用户后续授权 train-only**：新训练先按原规则切分，仅在合成 train 拟合 Y/P 均值、标准差与 `asinh_scale.s_c`；统一冻结用于 val/test/real 和 AL 入库。新 checkpoint 标记 `train_synthetic`。旧载荷保留 `all_synthetic_legacy`，无完整载荷的旧权重仅按历史范围重建，禁止默配新统计量。重复样本分组和指标口径仍未修改；pure-random 生成 RNG 的后续修复见 7.2。详见 [train-only 修复](docs/TRAIN_ONLY_NORMALIZATION_2026-10-19.md)。
- **最新实测证据（2026年9月基准）**：
  在 `mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/` 下的三种归一化对比实测证明：
  - **log1p + Y/P Z-score**：Real Rel L2 = **0.1518**, Real Corr = **0.9723**, Real MAE = **0.0869**（这组三项最好，但 MSE 不是最好）。
  - **Asinh Scale**：Real Rel L2 = 0.2054, Real Corr = 0.9576。
  - **Asinh Z-Score**：Real Rel L2 = 0.1713, Real Corr = 0.9655。
  这是历史同组存档，不是本轮重训结果。物理 MSE 分别为 log1p 5.18461、asinh_z 5.51186、asinh_scale 2.42423。
  checkpoint/启动/自回归/求解保护等当前契约见 [运行链路修复](docs/RUNTIME_CONTRACT_FIX_2026-10-19.md)。

### 7.2 数据集生成引擎清单

**2026-10-22**：当前 systems 静态10/40/50；Budding AL已在2026-10-28改为模块集合/模式两级均衡，不再用5/40/55（Fission仍用原规则），仅精确L1–L3，原始生成器仍保留。修改采样前必须阅读 [三模块规则](docs/EXACT_THREE_MODULE_SAMPLING.md)，不得重新引入独立兜底配额或默默截断旧高阶数据。
**2026-10-19 RNG 后续修复**：`generate_pure_random_dataset.py` 的局部 `RandomState(seed)`
现显式传入幅度采样、随机等位基因处理、冲突重抽和碰撞救援；显式 RNG 的冲突键按固定顺序遍历。
新 NPZ 记录 `sampling_seed` / `sampling_rng_contract=pure_random_single_stream_v1`。
共享辅助函数省略 rng 时仅为兼容旧调用方而保留全局行为；新/迁移的生成器应显式传入同一 rng，
不能据此宣称其他生成器也已完成迁移。ODE、采样范围/概率权重、real 筛选及旧数据未修改。
详见 [RNG 修复与 corr 风险说明](docs/PURE_RANDOM_RNG_FIX_2026-10-19.md)。

`data/` 下的所有数据生成与采样器已全部得到保护与留存：
- `generate_1m_pure_random_pool.py`：1M 纯随机参数候选池生成器。
- `gibbs_sobol_sampler.py` & `gibbs_boltzmann_sampler.py`：加扰 Sobol 与能量分布准随机采样。
- `bams_dataset_generator.py` & `mass_dataset_generator.py`：生物学感知与分层变异采样。
- `generate_kdpp_gated_dataset.py`, `generate_kdpp_curv_gated_dataset.py`, `generate_motif_topology_dataset.py` 等。
- `reachability_check.py`：可达性与零训练探针引擎（支持任意目录执行，默认输出至 `data/reachability/`）。

### 7.3 多关系生物因果拓扑图
当前权威潜在结构由 `data/yeast_graph.py` 从 `mutant_aware.py` 的 ODE/事件源码派生，
统一 `A[target,source]`。关系图由 `data/yeast_relations.py` 的反应项注释覆盖层与
untyped 补充组成，不从旧 NPZ 读取连边；默认 hybrid 为 226 条非自环边。
旧 `data/yeast_multirelational_graphs.npz` 保留供历史复现和 circuit/module 元数据，
不能称为完整 ODE 图。DynamicPlus 模块方向已修复，soft attention 保留。
固定图默认 current 刷新，历史图复现显式使用 checkpoint 策略。完整范围与限制见
[docs/GRAPH_RELATION_MODULE_FIX_2026-10-19.md](docs/GRAPH_RELATION_MODULE_FIX_2026-10-19.md)。

---

## 八、关键避坑指南与科学细节 (Gotchas & Best Practices)

1. **`PARAM_KEYS` 顺序绝对不可更改**：
   `al_callback.py`、`physics_eqns_torch.py`、`mass_dataset_generator.py` 均严格依赖 `sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])` 得到的 141 维特征顺序。
2. **ODE BDF 积分器三层超时保护**：
   在调用 `solve_ivp(method='BDF')` 仿真复杂突变体时，病态参数可能导致底层 Fortran 循环挂起。必须保证：
   - 内置 `SIGALRM` 5秒硬超时中断；
   - Python 父进程 35 秒无完成结果时终止自有 worker；AL 最多重建一次未交付任务，重复失败则中止生成（不是对已完成 future 调用 timeout）；
   - `try...except` 捕获异常返回 `None`。
3. **日志输出与实验管理**：
   所有实验日志唯一写入各自的 `experiments/.../train.log`，不再在根目录生成 `logs/` 或 `logs_v3/` 软链接。
4. **Git 仓库管理**：
   所有大文件（`*.npz`, `*.npy`, `*.ckpt`, `*.zip`）、实验输出（`experiments/`, `eval_result/`）、临时沙盒（`scratch/`）、测试输出（`tests/`）已全部被 `.gitignore` 纳管，严禁提交至版本库。

---

## 九、权威专著文档导航 (`docs/`)

如需查阅更详细的数学公式推导、模型内部实现与评测结果，请直接阅读 `docs/` 下的 5 大专题文档：
- 📘 [docs/1_ARCHITECTURE_AND_MODELS.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/1_ARCHITECTURE_AND_MODELS.md) — 系统架构与 70+ 模型全景速查字典
- 📗 [docs/2_ACTIVE_LEARNING_GUIDE.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/2_ACTIVE_LEARNING_GUIDE.md) — 全量主动学习算法与 AL4PDE 体系详解
- 📙 [docs/3_LOSSES_AND_PHYSICS_CONSTRAINTS.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/3_LOSSES_AND_PHYSICS_CONSTRAINTS.md) — 17 种损失函数、GradNorm 梯度平衡与 PINO 约束
- 📕 [docs/4_DATASETS_AND_NORMALIZATION.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/4_DATASETS_AND_NORMALIZATION.md) — 数据集生成引擎、三大归一化实证与生物图谱
- 📓 [docs/5_BENCHMARK_AND_LEADERBOARD.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/5_BENCHMARK_AND_LEADERBOARD.md) — 126 真实突变体基准评测排行榜与 Badcase 审计

AL精度边界：`_pool_initial_physical`用于精确来源保存（Budding float64），模型缓存由`refresh_model_inputs(dm)`统一生成float32；train.py每轮必须调用它，时间轴匹配IC dtype。不能直接normalize_y原始float64初值后推理；此次六策略首轮故障及真实checkpoint验证见docs/AL_DTYPE_FIX_2026-10-28.md。

2026-10-30 新AL优化：默认 AL_SUITE=optimized，四个 global_* 策略＋六项中断重跑＋修正event；full预算不变，GPU0–3各一项，旧结果不可覆盖。方法/内存修复/验证见 docs/AL_OPTIMIZED_RESTART_2026-10-30.md。


2026-11-17 02:51正式批次已在tmux `budding_sampling_formal_20261117`启动，输出 `/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/seed42_a866aef5d988/`；批日志 `batch/launcher.log`，单项 `<pool>/<method>/seed_42/train.log`。45项资源CPU回归和3项小流程回归通过，dry-run16项及参考SHA核对通过。

02:54复核：16个正式run进程均存活且WANDB_MODE=offline、GPU仅0–3；mutation/VeSSAL已完成首轮至少16epoch，loss有限，其余正在加载/预处理。启动核查记录batch/startup_verification.json。尚不能声称整批五轮完成或所有后续评分均通过。
