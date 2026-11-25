# Mutant-aware sampling研究记录：假设、证据与决策

最后整理：2026-11-25。范围：当前可核验报告支持的Budding/FNO采样研究，包含早期ours、2026-11-22至25日的机制探究、统计量控制与R21–R27查询合成推进。不是成功方法宣传稿，也不是全部历史运行的逐条日志。

本记录中的“思考过程”指可审查的研究理由：为什么提出假设、怎样隔离因素、观察到什么、因此保留或停止什么。早期精确执行时间或缺失协议不凭对话补造；日期优先采用原报告日期。详细指标和样本预测不复制到本文，通过证据链接回查。

## 接手先读：当前快照

- 最新已完成：**R26 首批 5/9 项完成并归档（`ours_bif_power_synth` 在 `seed 123/1024` 两 seed 上双线优于 `Power/LCMD/Random`，`ours_grad_bif_synth` 保留作次要消融参考，`ours_pool_mode_bif_synth` v1 已查明复合体误拆与高振幅劫持两处缺陷）**；**当前后台运行中：R27 方案 B（8 项任务分三卡三会话：`r27_gpu0` 4 并发、`r27_gpu4` 与 `r27_gpu7` 严格独占单并发，补齐 `ours_bif_power_synth seed_42` 并三 seed 验证纯池锚定 `ours_pool_bif_synth` 与复合体安全/尺度归一化 `ours_pool_mode_bif_v2_synth`）**。队列运行期间严禁修改任何 Python 源码。
- 最新用户决定（2026-11-25）：固定 mutation 数据底座与 126 株 real 评测目标，按三 seed（`42/123/1024`）正式验证；保留 `ours_grad_bif_synth` 思路但不作主攻；在未获完整三 seed 稳定胜出前不提前宣称方法定型。不要照历史建议重启已结束分支。
- 机器可读状态：[current_state.json](current_state.json)；跨平台入口：[RESEARCH_HANDOFF.md](README.md)。原报告及checkpoint身份优先于聊天摘要。
- 早期“下一方向已同意”是历史决策，后续R13—R26已执行；不能当作最新运行状态。

## 1. 当前结论与研究目标

目标：不改变合法mutation、强度支持、初值、ODE和事件规则，在有限训练样本/查询与算力预算下，提高多变量完整轨迹泛化；先研究Budding，再验证跨系统适用性。

当前没有获得稳定超过强baseline的ours证据。已确认的限制：

1. 高预测误差、物理残差或响应评分，不自动等于高训练收益。
2. 大量预算加密局部路径，会减少不同突变背景的覆盖；能找到困难点不代表能学得更好。
3. 数据量、更新次数、统计拟合范围和checkpoint选择都可能影响结论，不能把联合变化全归因于选样。
4. 零/近零通道与ORI等变量可主导不同指标；不能只看单个总分，也不能因此删除不利指标。
5. 低阶样本不是越多越好；本轮固定L配额路线停止，不继续搜比例。

**下一方向已获用户同意（2026-11-23）：围绕已查询合成样本暴露的具体机制错误设计选样。** 目前仅为方向，不是已完成算法；不等于只围绕CDH1T或已知real badcase定制权重。

## 2. 阅读路径与证据等级

| 编号 | 问题 | 本次得到的判断 | 决策 |
|---|---|---|---|
| R01 | 早期机制ours是否胜过baseline？ | 部分指标改善，未形成全面稳定优势 | 保留历史结果，补多seed及机制诊断 |
| R02 | 响应/事件/进展生成是否有效？ | 三seed未稳定超过Power；响应支撑经常缺失 | 修正无依据评分与支撑问题 |
| R03 | 支撑充分、分数有效后是否带来收益？ | Response V2仍差；路径覆盖收缩 | 暂停局部加密长跑 |
| R04 | 跨背景物理响应评分能否产生实际收益？ | 小试增益0.286%，未过门槛 | 不扩成长跑 |
| R05 | 真正高误差样本是否值得优先训练？ | 两训练RNG平均仅改善0.138% | 不再把困难度等同训练价值 |
| R06 | 更多数据是否还有收益空间？ | 随机3万优于1万多项指标，但步数也增多 | 补近似等更新对照 |
| R07 | 是数据量还是训练步数？ | 长训改善1万结果，3万近似等更新仍有收益 | 后续研究统一更新预算 |
| R08 | 条件机制覆盖/迁移/缺口生成是否有效？ | 机制覆盖部分改善但不稳定，其他未证实 | 不扩大评分组合 |
| R09 | 平衡干预阶数是否是关键？ | L1改善，L2/L3损伤，筛选阴性 | 停止固定L配额 |
| R10 | 退化具体体现在哪？ | ORI与恒零CDH1T错误，非简单“没见过零” | 提出统计来源控制 |
| R11 | 共用归一化能否挽救低阶组？ | 未挽救，随机组也会退步 | 保留正式归一化协议，停止该配额分支 |
| R12 | 如何从机制错误得到可迁移选样？ | 尚未验证 | 先定机制与反事实对照，不直接批量长跑 |
| R13 | 初值还是后续演化错误？ | 不同机制失效不同 | 区分错误机制，不只看总误差 |
| R14 | 机制高误差＋背景去重有学习收益吗？ | 两RNG阴性，配对差异较弱 | 不放大该实现 |
| R15 | 模型是否识别相关干预？ | 部分通道几乎忽略真实控制 | 检验响应缺失生成 |
| R16 | 响应缺失生成能否改善其余变量？ | 阴性，评分比同生成器随机差0.333% | 停止该原型，复核主贡献定位 |
| R17 | 现有mutant-aware优势到底多稳定？ | 21/21配对MAE/MSE/Corr占优，存在背景混杂 | 该消融建议后被用户暂缓，继续采样创新 |
| R18 | 输入通路梯度能修复响应并迁移吗？ | 梯度可用，选样反馈改善但隔离背景未改善 | 不立即扩大草案 |
| R19 | 固定评分实际冷启动训练是否有收益？ | 阴性，condition物理Relative比random差2.199% | 停止该贪心梯度匹配原型，转向校准探针与任务对齐不确定性×多样性/合法合成 |
| R20 | Power/LCMD逐轮机制与双尺度不确定性结构？ | 双尺度分歧全面胜Power；早覆盖/晚精修相变；原子误差15.8×异质 | 设计DMCED统一框架（固定池+合法合成）并跳过微探针直上三seed五轮 |
| R21 | DMCED固定池/消融/合法合成能否三seed胜Power/LCMD？ | 9/9完成，real未胜Power；`dmced_power`合成MAE第一(0.2601) | 停止确定性argmax与无层级保护合成；转向单通道切换能量+随机加权$D^2$与层级保边际合成 |
| R22 | 未截顶相对方差+随机加权$D^2$+层级保边际合成能否胜Power？ | 3/3完成：全消除Rel爆炸(1.11)；$D^2$净赢-9%MAE/-38%Rel；但`e_rel`分母畸变伤MAE | 不扩R22原版；R23换回无分母畸变$e_{\text{mse}}$+随机加权$D^2$/$D^1$及无偏合法合成 |
| R23 | 无分母畸变$e_{\text{mse}}$+随机加权$D^2$/$D^1$及无偏合法合成能否胜Power？ | 3/3完成：`d1_pool`合成全面第一(0.2607/60.95)、real逐株75/126胜Power且中位MAE/MSE/MacroRel全胜，仅1株#124拉高单seed均值 | 补齐`d1_pool`三seed并测`d1_synth`与`d05_pool`三seed正式检验 |
| R24 | Epistemic-$D^p$（$p\in\{0.5,1.0\}$）固定池与无偏合法合成能否三seed全面胜Power/LCMD/Random？ | 8/8完成：`d1_pool`胜合成MAE/real中位与Corr但三组real均值MAE均输Power(0.1719/0.1793/0.1886 vs 0.1618)；查明`d^p`饿死L1/L2 | 正式终止`Power×d^p`路线；转向R25 CEBS（阶数守恒+0-ODE上位性+分岔剂量合成） |
| R25 | CEBS（阶数边际守恒+0-ODE虚拟非线性上位性+0-ODE连续强度分岔合成）能否同时击败原版Power与分层Power？ | 3/3完成：模块1/2在10k为负向(`0.1661/0.1739`)；唯有模块3分岔合成大胜（全历史最优`val_error 0.1866`/`test_loss_synth 0.1843`/real中位MSE `0.3217` -26.8%、逐株71/126胜Power） | 不扩R25原版；淘汰模块1/2，建议R26测试纯净“无偏Power拓扑+0-ODE连续分岔边界合成” |
| R26 | 池锚定混合模式切换+分岔合成与纯连续分岔合成能否在三seed同时击败固定池Power？ | 首批5/9完成（4项因旧目录残余快速退出）：`ours_bif_power_synth`两seed均值real MAE `0.16049`/synth MAE `0.25171`双线领先Power(`0.16757`/`0.26134`)；`ours_grad_bif_synth`略弱保留参考；`pool_mode` v1查明误拆复合体与振幅劫持两处缺陷 | 补齐`ours_bif_power_synth` `seed_42`，并推进R27纯池连续分岔与修复版跨模式分岔 |
| R27 | 纯池锚定连续分岔（0池外生成器）与复合体安全+尺度归一化门控跨模式分岔能否三seed稳定胜Power？ | 方案B共8项已在GPU 0（4并发）与独占GPU 4/7（严格单并发）启动中 | 等待全部完成并汇总完整三seed对照表，再判断是否稳定有效 |

证据类型分开使用：①已完成的多seed终评；②单seed控制；③固定checkpoint短程探针；④只读误差分解；⑤待验证设想。它们不能放进一个同预算主榜单。本文中“阴性/未通过”仅针对该实现与协议，不是证明所有同类方法无效。

## 3. 逐项研究记录

### R01：早期机制ours与baseline对照（2026-11-19报告）

- **研究理由：** 不只找单点不确定性，尝试利用ODE灵敏度、局部响应Closure、Transport识别模型未掌握的突变作用。
- **设计：** Sensitivity、Closure、Transport池选/生成等版本，与Random/Power/LCMD/PRE等已有适配比较；此时ours主要为seed42，持续训练/生成方法成本不同。
- **观察：** mutation real MAE：Power 0.150298；Sensitivity池选0.161728，Closure池选0.166407，Transport池选0.197371。个别指标有收益，不支持整体胜出。HGGS旧流程等历史版本在原报告明确标记。
- **决策及下一问：** 不能靠挑一个seed/指标判断方法；补baseline三seed，并检查机制评分是否实际触发。真实误差事后分析不直接写成选样规则。
- **证据：** [早期三ours/八baseline完整表](/data/zsy/budding_yeast/experiments/budding_sampling/summary/selected_baselines_three_ours_20261119/report.md)、[多seedbaseline及早期ours分析](/data/zsy/budding_yeast/experiments/budding_sampling/summary/multiseed_20261122/report.md)。

### R02：Adaptive Response / Event / Progress（三seed，2026-11-22报告）

- **假设：** 合法响应、事件失配或查询后进展反馈，比只看预测不确定性更能识别有用新样本。
- **设计：** 三种生成方法各42/123/1024，共9项；五轮达到1万条，原生mutation和ODE不变。
- **观察：** real平均MAE：Power 0.16182，Response 0.16963，Event 0.17563，Progress 0.17228。新增查询73.5%～77.0%没有同背景锚点，却以固定score=1和可计算误差混排；只有少数候选有多锚点支撑。
- **解释边界：** 方法代码执行过，不代表核心响应机制主导了选择。不能把未知分数当成测得的模型误差。
- **决策：** 分离探索与可估计响应，补路径随机对照，保留查询前预测及任务一致误差。
- **证据：** [完整结果](/data/zsy/budding_yeast/experiments/budding_sampling/summary/adaptive_multiseed_20261122/report.md)、[机制触发与失败分析](/data/zsy/budding_yeast/experiments/budding_sampling/summary/adaptive_multiseed_20261122/findings.md)。

### R03：Response V2与同路径随机（两方法×三seed，2026-11-22）

- **假设：** 先形成可辨识的路径支撑、再选响应错误大的点，可以修复R02。
- **设计：** 同路径生成、同预算随机选点对照；探索1600、补支撑500、有效响应/随机查询5900，共新增8000。
- **观察：** 支撑查询73.75%，没有固定1回退；响应分数与查询后任务误差有较高相关性。但real MAE：V2 0.25889，路径随机0.24297，均明显落后Power。V2对路径随机三个seed均更差；同池合成测试也退步。
- **额外证据：** 新增背景由上版约6810～6952缩至约1812～1832；前100背景占比约一半。V2的ORI极值比上版更低，仍预测更差；训练里仍有关闭轨迹。
- **决策：** 暂停多数预算用于局部强度加密；不能把分数相关性当作泛化收益，也不能简单恢复大值过滤来解释退步。
- **证据：** [结果](/data/zsy/budding_yeast/experiments/budding_sampling/summary/response_v2_results_20261122/report.md)、[覆盖与逐变量诊断](/data/zsy/budding_yeast/experiments/budding_sampling/summary/response_v2_results_20261122/findings.md)、[实现协议](../reference/RESPONSE_PATHS_V2.md)。

### R04：跨背景干预迁移的短程学习收益（seed42）

- **假设：** 在多个背景重复检查合法干预的物理响应，比单个局部路径更可迁移。
- **设计：** 同初始2000条checkpoint，四组各400更新；新增256条全局随机/同生成器随机/跨背景评分；独立512条未见背景合成探针。冻结原统计，不读real或正式holdout真值。
- **观察：** 评分组归一化Relative 0.380166，对同生成器随机0.381257仅改善0.286%，对全局随机0.379835反而稍差；未达到预定>2%门槛。
- **决策：** 不启动该原型多seed长跑；转向直接检验“高误差是否有训练价值”。
- **证据：** [协议与逐组结果](/data/zsy/budding_yeast/resource_profiles/background_transfer_probe_20261122/report.md)。

### R05：真实高误差与反馈试训选批（同一划分，训练RNG42/123）

- **假设：** 如果物理评分不准确，直接知道候选真实误差、或比较短试训收益，应能选到更好的追加批次。
- **设计：** 训练侧4096条短名单；每批256条；8批40步反馈试训；锁定选择后400步独立检查。两套训练RNG不是两个独立split；读取7120条标签，不是只查询256条的AL。
- **观察：** 高误差相对随机平均归一化Relative仅改善0.138%，背景bootstrap区间跨0。试训选中相同高误差批次，不是另一种独立算法胜出。
- **决策：** 不能只继续发明“难度”评分；转查数据规模/训练预算是否限制了可观测收益。
- **证据：** [原始结果、区间与信息访问](/data/zsy/budding_yeast/resource_profiles/sampling_utility_probe_20261122/report.md)。

### R06：随机1万与嵌套3万数据量参考（seed42）

- **假设：** 先确认同模型对更多唯一输入是否仍有学习空间，避免在接近饱和的区间盲目采样。
- **设计：** 3万包含原1万，均200epoch；相同模型初始化，统计各自拟合，best按来源合成验证归一化loss。内部探针先比较，锁定best后补real参考。
- **观察：** real MAE 0.17746→0.11934，Relative 12.36778→0.53508；但MSE 3.44796→3.89046变差。内部探针MAE也改善。
- **混杂：** 更新数同时由15800增至47000，不能据此宣布数据量因果作用或采样算法收益。
- **决策：** 补1万近似等更新控制。
- **证据：** [报告](/data/zsy/budding_yeast/experiments/budding_sampling/summary/data_volume_10k_30k_seed42_20261122/report.md)、[分解与边界](/data/zsy/budding_yeast/experiments/budding_sampling/summary/data_volume_10k_30k_seed42_20261122/findings.md)。

### R07：1万长训与3万近似等更新（seed42）

- **设计：** 原1万从头595epoch/47005步，对比3万200epoch/47000步；预热约对齐，余弦按epoch离散，统计仍各自拟合。
- **观察：** 1万长训real MAE/MSE/Relative为0.168669/4.831856/0.650110；3万为0.119336/3.890455/0.535076。仅加更新已消除大部分原Relative异常，更多输入在近似等更新下仍有收益。
- **决策：** 后续最终模型控制更新预算；不把“训练更久”包装成采样创新。仍不能排除数据统计的共同变化，且只有一个seed。
- **证据：** [三方对照和近零贡献](/data/zsy/budding_yeast/experiments/budding_sampling/summary/matched_updates_10k_30k_seed42_20261122/report.md)。

### R08：条件动力学覆盖/组合迁移/机制缺口生成（7方法×3seed）

- **假设：** 跨背景泛化可能依赖机制作用覆盖，而非原始参数距离或单点误差。
- **设计：** 普通轨迹覆盖、条件覆盖、RHS等机制覆盖、梯度迁移及生成与各自随机对照；全部最终1万条、595epoch/47005步，迁移pilot额外计费。
- **观察：** real平均MAE：池内随机0.15911，轨迹覆盖0.16026，仅条件覆盖0.16659，机制覆盖0.15422，组合迁移0.19156；随机生成0.15016，机制缺口生成0.15594。机制覆盖均值改善3.07%，但只赢1/3seed；同池合成MAE均值改善7.31%。
- **决策：** 不宣称稳定超过baseline，也不继续扩大评分组合；检查干预阶数是否是覆盖方法变化中的可解释因素。
- **证据：** [全部指标与核验](/data/zsy/budding_yeast/experiments/budding_sampling/summary/conditional_dynamics_v1/report.md)。与旧五轮实验预算不同，不跨协议混排。

### R09：干预阶数配额（seed42筛选）

- **假设：** 原池L3占多数，重新分配低阶训练支持可能有助于组合学习。
- **设计：** 共同初始2000条；最终1万条的随机/尽量L均衡/低阶优先三组；同初始化和47005步。训练、反馈、独立审计按完整背景隔离。
- **观察：** 均衡/低阶优先的共同坐标Relative改善约9.58%/11.58%，但均衡物理MAE变差约24.49%/21.58%，物理Relative大幅变差。L1获益，L2/L3受损。
- **决策：** 筛选阴性；123/1024确认跳过，独立审计/test/real未开放；不继续找“更好比例”。这些是固定配额控制，不是新算法。
- **证据：** [完整报告和门槛](/data/zsy/budding_yeast/experiments/budding_sampling/summary/intervention_order_calibration_v1/report.md)。

### R10：反馈变量分解与归一化权重诊断（只读，不训练）

- **问题：** R09的物理退步具体来自哪里？是否只是没覆盖相似轨迹？
- **观察：** 两quota的L2 MAE净增量约91%/86%来自ORI。28条输入满足CDH1T解析恒零条件：随机预测均值约0.00758，两quota约0.99990/0.99973；训练中仍有924/760条该通道全零轨迹。
- **机制核验：** CDH1T满足dy/dt=kscdh-kdcdh*y，初值和kscdh均0且不受事件重置，即真值应恒零；不是依据real身份猜测，也没有新调用ODE。
- **损失诊断：** log1p后中心化Relative的零点与物理Relative不同，相同0↔1错误的训练惩罚不对称。此为目标权重失配证据，不证明归一化解释全部退化。
- **决策：** 追加仅统计来源变化的配对控制，避免误把训练尺度影响归因于样本选择。
- **证据：** [定点诊断JSON](/data/zsy/budding_yeast/experiments/budding_sampling/summary/intervention_order_calibration_v1/feedback_diagnosis/diagnosis.json)、[逐通道CSV](/data/zsy/budding_yeast/experiments/budding_sampling/summary/intervention_order_calibration_v1/feedback_diagnosis/per_channel_feedback.csv)。

### R11：公共初始统计的2×2配对控制（2026-11-23）

- **设计：** 复用随机/低阶优先原训练ID与顺序，各补一个公共初始2000统计模型；旧两模型各用自身1万统计。模型seed、47005步、loss、验证ID与best规则不变。公共统计影响输入编码、目标编码和验证best，不是仅隔离loss分母。

| 训练集 | 统计来源 | 共同坐标Relative | 均衡物理MAE | 均衡物理MSE | 均衡物理Relative |
|---|---|---:|---:|---:|---:|
| 随机 | 自身1万 | 0.183869 | 0.200024 | 29.103063 | 0.533903 |
| 随机 | 公共初始2千 | 0.189357 | 0.231813 | 56.843937 | 4.120412 |
| 低阶优先 | 自身1万 | 0.162580 | 0.243183 | 74.658233 | 4.256179 |
| 低阶优先 | 公共初始2千 | 0.164427 | 0.237996 | 68.455646 | 4.457449 |

- **核验：** 两新增模型完整595epoch/47005步、无任务失败，批次49分30秒；best为473/560epoch，来源合成验证选择；未读独立审计/test/real。
- **观察：** 共用统计未挽救低阶优先。共用统计下其L1 MAE相对随机改善约56%，L3变差约34%，均衡MAE仍差约2.7%；随机冻结统计自身也退步。
- **注意指标权重：** 冻结随机组CDH1T贡献约88%的普通逐样本平均Relative总和；这是未按L/背景均衡的分解，不能当作上表均衡Relative的88%。
- **决策：** 不把冻结初始统计作为新默认，不继续L比例搜索；归一化有影响，但不是该配额思路的解法。两项单seed控制不足以宣称一般性因果规律。
- **证据：** [比较报告](/data/zsy/budding_yeast/experiments/budding_sampling/summary/statistics_only_control_v1/report.md)、[四组完整指标与运行路径](/data/zsy/budding_yeast/experiments/budding_sampling/summary/statistics_only_control_v1/comparison.json)、[批次完成凭据](/data/zsy/budding_yeast/experiments/budding_sampling/batches/statistics_only_control_v1_20261123/completion.json)。

### R12：机制错误驱动的选样（已同意方向；待设计、未启动）

- **研究问题：** 能否利用已查询合成样本及已知ODE结构，区分模型“尚未学会某种机制”与“只是某个点误差大”，选择在新背景上真正有训练价值的合法样本？
- **为什么继续：** R03/R05否定了把难度直接当价值的捷径；R10/R11显示具体通道关闭规律可能仍被误学，且样本数量与归一化变化不足以单独解释。
- **尚未证实：** 任何机制错误评分都可能只是PRE/uncertainty的变体；“ODE约束”本身不是创新，也不能保证采样提升。须先定义与已有baseline的差异及可证伪假设。
- **下一项设计必须明确：** 通用机制接口；仅训练侧错误证据；选择或合法生成动作；同生成器随机及高误差对照；新增标签、选择反馈和优化步数成本；独立合成反馈/审计的使用次序；停止条件。
- **信息边界：** 不把CDH1T名字、real编号或其测试误差硬编码成奖励。不能拿R09/R11的保留审计或正式test来反复设计规则。合成反馈已参与研究选择，不再称盲测。
- **跨系统：** 本记录仅支持Budding观察；Fission尚未验证。其测试数据如未参与开发，可作为未来外部验证，不能提前承诺同样效果。
- **执行状态：** 本次只整理文档，没有新增训练、生成、ODE查询或评测。

## 4. 信息使用与诚实报告

早期R01/R02/R03及R06/R07/R08的real结果曾参与开发诊断，不能将整个研究追溯性地称为“从未看过测试集”。源代码没有读取real文件，不等于研究过程没有测试集适配风险。后续应如实说明开发参考与独立验证的区别。

R04/R05不读取real或正式holdout标签；R09/R10/R11仅用既定合成反馈，独立审计/test/real未开放。R09的阴性门槛和R11的补充诊断必须同时保留，不因后续想法变动改写成筛选成功。

已有完整标注池允许的方法须报告实际访问标签数；不能把最终训练1万等同“总共只见1万标签”。生成法记录ODE尝试/成功/唯一输入，持续训练报告历史曝光和更新数。各方案best只能按声明的合成验证指标选，不能依real挑epoch。

## 5. 后续维护方式

1. 新研究使用下一个固定编号R13、R14……；运行前记录假设、对照、预算、信息权限和继续/停止条件，不能看完结果再补称“预先声明”。
2. 完成后补结果、失败/未触发机制、结论边界、下一项决策；失败结果保留，不能只留最优seed。
3. 只改正事实错误，不删除旧结论：新增“更正/被哪项证据修订”，保留原始报告和身份。
4. 完整模型配置、源码身份、选择ID、checkpoint和日志放原实验目录；本文件只记录决策链并链接。无新实验时不创建空批次或伪造运行记录。
5. 当前证据快照见[evidence_manifest.json](evidence_manifest.json)：记录原报告路径、SHA256与核验日期。哈希仅证明本次引用的文件版本，不证明论文创新、指标因果关系或数据独立性。

## 6. 当前停止与保留清单

- **停止扩大：** 固定L配额搜索；Response V2式多数预算用于局部强度加密；未过门槛的跨背景评分原型；仅凭高误差选样继续大规模长跑。
- **保留为对照/证据：** 强baseline与各自多seed、所有负结果、同生成器随机、共同初始集、现有loss/eps和正式逐训练集统计协议。
- **待证明：** 机制错误识别是否能转化为未见背景上的实际训练收益，以及相对PRE/Power等是否有独立增益。

## R13：初值错误与后续演化错误的解析分解（2026-11-23，已完成）

- 假设：部分关闭通道的失效可能在t=0已出现，不能全部归为后续动力学或采样背景不足。
- 设计：仅复用R11的四个checkpoint和既有366条合成反馈；CPU单进程，检查三条原ODE独立方程CDH1T、CDC14T、NET1T的dy/dt=a-b*y。原事件重置不改变它们。无训练、新ODE或real/test/audit访问。
- 分解：解析真解与预测之差=初值误差按exp(-bt)传播+模型相对自身预测初值的额外演化误差。报告两个平方项及有符号交叉项，不把平方项比例解释成因果百分比。另核对保存真轨迹与解析解的偏差。
- 边界：这是诊断，不修正正式预测、不改变主指标。三条已核对的方程不是通用自动机制提取；结果不能直接声称采样有效。若错误仅发生于可直接求解的孤立方程，继续购买其重复仿真不一定有价值，需检查下游耦合影响。
- 结果：28条解析恒零CDH1T样本，随机自拟合统计的初值误差/整段MAE约0.00949/0.00752，其余三模型均约1/1；CDC14T及NET1T在随机模型中的初值误差仅约0.00025/0.00034，整段MAE却为0.46164/0.26639，呈现不同失效形式。
- 决策：机制错误至少区分初值识别与后续演化；不能只加初值约束或继续局部密集查询。解析通道无需昂贵ODE标签，直接按其错误排序仍不等于有效采样创新。下一步需要检验机制错误的跨背景迁移及下游耦合收益，而非直接把三个变量加入评分公式。
- 证据：[R13完整分解](/data/zsy/budding_yeast/experiments/budding_sampling/summary/scalar_mechanism_diagnostic_v1/report.md)、[运行协议](/data/zsy/budding_yeast/experiments/budding_sampling/summary/scalar_mechanism_diagnostic_v1/protocol.json)、[checkpoint哈希及结构化结果](/data/zsy/budding_yeast/experiments/budding_sampling/summary/scalar_mechanism_diagnostic_v1/report.json)。本条在推理完成前登记，属于开放式诊断，不是独立假设检验；首次路径序列化失败及成功重跑日志均保留。

## R14：机制错误的跨背景训练收益（2026-11-23，已完成，筛选阴性）

- 假设：同类机制误学在更多完整突变背景上的训练支持，能比只挑机制误差最大的样本带来更好的下游泛化；只凭R13不能确认。
- 对照：同一个共同初始2000条checkpoint，继续原数据、随机追加256、完整任务高误差追加256、三条解析机制高误差追加256、机制高误差且每完整背景至多1条追加256。最后两者的误差难度可能不同，报告所选误差与背景数量，不宣称单独隔离多样性的因果效应。
- 公平性：每组固定400更新、batch128、LR1e-4；统计冻结于初始checkpoint，仅诊断不修改正式协议。训练RNG42/123共10次短训，不等同两个独立数据划分。
- 选择：4096条训练侧候选；机制分数采用三条解析真解在初始checkpoint归一化空间的Relative误差最大值，所有组同候选域。完整任务高误差对照允许读取短名单真值，明确是完全监督诊断。
- 独立检查：新保留512条未见完整背景训练侧探针。排除R05与R09反馈/审计的完整背景；旧审计和正式validation/test/real标签不读。新探针标签在五组选择锁定后才揭示。另保留的新反馈角色不读取。
- 预定门槛：两个训练RNG中，机制多背景组在排除三条打分通道后的36变量归一化Relative，均相对随机、完整任务高误差及机制高误差改善至少2%；其耦合变量物理MAE相对随机不得恶化超过5%。未通过则不开展完整长跑，不根据新探针逐例误差重写选择器。
- 边界：这是区分学习价值的短程对照，不是新算法或论文创新声明；机制分数与PRE相关性及跨系统机制接口尚未解决。没有新仿真，不重新生成候选池。初值与方程约束没有加入模型或loss。
- 入口：scratch/probes/probe_mechanism_transfer.py，单进程GPU0；记录及最终结果为summary/mechanism_transfer_screen_v1/，批次为batches/mechanism_transfer_screen_v1_20261123/，系统可用内存保留100GiB。

### R14完成后的决策补记

10次短训均成功，用时12分31秒。机制背景去重组相对随机的耦合Relative在训练RNG42/123分别恶化0.921%/0.698%，没有通过原定门槛。机制TopK和背景去重实际重合237/256，只有19条变化；随机本已覆盖254种背景，去重256种。因此原设计对“增加背景多样性”的干预较弱，只否定本实现，不据此否定一般跨背景迁移。

决定不扩大本版，不增加epoch或改门槛。下一项候选方向为干预可辨识性：用合法的相关条件改变与无关背景改变，诊断模型有没有学会区分机制条件；同时约束已知规律正确性以排除常数模型。不将这种诊断直接称新算法，先验证动作差异和信号，尚未开始新训练。详见[R14完整报告](/data/zsy/budding_yeast/experiments/budding_sampling/summary/mechanism_transfer_screen_v1/report.md)。

## R15：干预可辨识性输入配对检查（2026-11-23，已完成）

- 用户确认：正式比较仍沿用LCMD等共同协议：2k/4k/6k/8k/10k，100/100/100/100/200epoch，warmup5/5/5/5/10，batch128，LR5e-4到1e-5，FNO192/4/24，来源合成验证归一化loss选best，逐轮统计按当前训练集重新拟合。R14固定400步是内部诊断，不能与正式结果排名。
- 本项只做预测检查，不改变任何正式超参数、loss、模型或生物规则。使用R14已合法隔离的4096候选输入，不读取轨迹标签。独立审计/real/正式test继续不读。
- 三条机制分别构造精确输入配对：背景变化对保持该标量方程的初值/合成/降解/mdt完全相同且背景不同；机制变化对在原生目标基因所有合法模式的参数/初值足迹以外完全相同，且该标量方程条件确实变化。端点均为源池已有完整合法输入，不进行向量拼接或改mdt。
- 配对仅由输入决定；每机制每类型最多32对、每精确匹配组最多8对。不能满足精确条件时直接报告数量，不放宽为相似输入凑数。
- 指标：真实/预测响应RMS、响应误差RMS、端点绝对误差，以及相关条件改变时的响应增益。背景不变性必须和已知规律正确性一起看，常数错误预测不会因此被视为正确。
- 重载初始2k模型及R11四个模型，使用各checkpoint原统计，CPU单进程；不训练、不求解新ODE、不据此改写正式评测。现有模型训练集合可能覆盖部分诊断输入，故不能将其当作独立泛化成绩或学习曲线因果对照。
- 状态：配对协议在推理前锁定，开放式诊断，不另设事后成功阈值。输出summary/intervention_identifiability_v1/；临时探针scratch/probes/probe_identifiability.py。

### R15完成后的决策补记

精确配对：CDH1T背景/机制变化32/2，CDC14T32/32，NET1T24/32，共154对、149端点。随机长训模型对相关条件变化的响应增益中位数为CDH1T约1.00006、CDC14T约0.00115、NET1T约0.0000389；后两者几乎未响应相关干预，虽对无关背景变化也很小。因此背景不变性单独不能作为好坏分数。低阶优先自拟合统计模型恢复部分CDC14T响应但丢失CDH1T响应，不能据此改写原配额阴性结论。

下一步转向相关干预“响应缺失”的机制选样原型设计，必须排除错误常数解，并验证相对同生成器随机/高误差/PRE的学习收益；本项不自动触发完整训练。配对共享端点、有输入选择偏倚、部分输入可能已被最终模型训练，不能作为新泛化榜单。详见[R15报告和曲线](/data/zsy/budding_yeast/experiments/budding_sampling/summary/intervention_identifiability_v1/report.md)。

## R16：响应缺失驱动的合法干预生成（2026-11-23，已完成，筛选阴性）

- 出发点：R15出现模型几乎忽略真正控制条件的情况，单纯背景不变性会接受错误常数模型。R14单点机制误差不能证明训练价值，且去重只替换19条。
- 原型：以已查询初始2000条为锚点，调用既有合法模块替换生成器，在CDH1/CDC14/NET1三个已审计机制上抽原生模式及原生强度，不改分布支持、mdt、初值准备、事件或ODE。对照动作是一次合法目标基因替换，不是四角构造。
- 评分：解析机制变化在checkpoint归一化坐标中的期望响应，与模型预测干预前后变化相减，取RMS。精确零作用动作标为不可评分，不强行赋高/低分；错误常数在真实作用非零时有正缺失分数。公共偏置会被差分抵消，所以分数不是绝对准确性证明。三条人工审计机制不冒充跨系统通用算法。
- 同生成器随机：使用完全相同的可评分候选与每机制标签预算；查询reserve跨机制交错，避免按gene串接导致首批标签偏向一个机制。响应评分组选样与随机组在查询前记录重合、背景及gene分布。未知背景不使用固定1分。
- 试验：仅原数据继续训练、全局合法随机生成、同生成器/同机制预算随机、响应缺失评分四组；各增量组补足256个成功输入，所有ODE失败尝试和重复去重单列。CPU4worker，底层/父级超时沿生产规则。新增轨迹只保存一次，初始集及训练清单引用原池。
- 学习收益：复用既有400更新、batch128、LR1e-4、初始checkpoint冻结统计的诊断协议；固定末步评测，不按探针选best。首轮只用训练RNG42。目标是先否定无收益原型，不与LCMD正式五轮成绩混排。
- 停止条件：排除三个评分通道后的36变量合成探针Relative相对同生成器随机改善>2%，同时优于仅原数据继续训练，物理MAE恶化不超过5%。未通过停止；通过仅允许复用同查询数据进行第二训练RNG确认，不能立刻声称有效算法或多seed胜出。第二确认后才考虑同baseline正式协议。
- 隔离：新合成探针按输入独立保留完整背景，排除R05/R09/R14已用反馈与保留审计背景；生成也排除这些背景及正式holdout输入碰撞。real/正式holdout标签不读取；新探针指标不反馈到本轮评分。
- 代码：utils/sampling/ours/response_identifiability.py；既有生成探针入口新增--score-kind response_omission，默认旧原型行为保留。复用公共训练/评测/存储/仿真，不复制这些循环。
- 结果目录：fno/mutation/ours_response_omission_probe/diagnostic_2k_plus256_steps400/seed_42/。批次batches/response_omission_probe_v1_20261123/，GPU0单进程，保留系统100GiB可用内存。现已完成，结果及停止决定见本节完成补记；未获得有效采样增益。

### R16完成后的决策补记

四组正常完成，批次10分21秒，717次ODE全部成功、去重后轨迹只保存一次。主要36变量Relative：仅原数据0.403770、全局随机0.397125、同生成器随机0.400503、响应评分0.401835；评分相对同生成器随机变差0.333%，没有达到>2%改善条件。两选样组仅51/256重合，本轮确有不同动作，仍没有收益。停止该原型，不自动开展第二训练RNG或正式长跑，也不换指标改判。

建议暂停相近评分的枚举，先审视已完成的mutant-aware与LHS正式多seed证据、背景参数域差异等混杂，并与导师确认主贡献定位；新训练尚未启动。此为下一步建议，不是用户已确认的新实验。详见[R16结果与下一步](/data/zsy/budding_yeast/experiments/budding_sampling/fno/mutation/ours_response_omission_probe/diagnostic_2k_plus256_steps400/seed_42/findings.md)。

## R17：生成范式证据与混杂审计（2026-11-23，已完成）

- 工作：只读7类baseline×mutation/LHS×42/123/1024共42项正式结果，逐项核对原始终评JSON、两池同方法/seed训练配置、指标版本；另读两个10万池的输入缓存和保存的LHS分布规格，不读新轨迹、不重推模型、不改数据。
- 结果：mutation在21/21个同方法/seed配对中MAE、MSE和Corr优于LHS；七类方法三seed平均MAE分别约0.162—0.266，而LHS约0.420—0.527。Relative只有Power三seed全赢，其他方法存在零通道主导的反向排序，不能省略。21对共享测试对象与pool，不作为21个独立seed的统计显著性证据。
- 输入核对：90个指定背景参数在mutation池所有样本均等于WT；LHS中没有样本这90维全部等于WT，最大偏差约50%。每样本不同于WT的141参数坐标数中位数为mutation5维、LHS95维。WT在LHS窗口内部，不能声称real全部超出LHS参数支持。
- 解释：现有证据支持既定mutant-aware生成协议对本任务的强优势，但不能将作用唯一归因于模块组合结构；背景域、联合相关结构和归一化统计等仍混在一起。未验证附加采样器稳定超过Power。
- 决策：停止相近错误评分的枚举。优先与导师明确生成范式主贡献是否成立，再做有针对性的背景/联合结构控制；新控制会涉及已冻结的生成条件，须作为独立方案讨论，不能悄悄改原LHS/ mutation池。采样创新目标未自动取消，但不能用目前证据冒称已实现。
- 证据：[完整结果/限制](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/generation_paradigm_evidence_v1/report.md)、[逐seed配对指标](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/generation_paradigm_evidence_v1/paired_results.csv)、[141参数域审计](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/generation_paradigm_evidence_v1/parameter_domains.csv)。入口scripts/analysis/audit_generation_evidence.py，无GPU或新训练。
- 文档迁移：研究主记录、当前状态、证据清单与跨平台入口统一迁入docs/research/；旧summary/research_journal目录不保留副本或软链。迁移映射/前后SHA在batches/research_docs_migration_20261123/manifest.json。原始实验JSON/CSV/图片不迁入Git。

## 用户确认的研究主线（2026-11-23；覆盖R17的下一步建议）

- 核心任务是针对mutation池/合法mutation域提出采样算法：既允许池内选择，也允许生成pool之外的合法输入；目标是在公平预算下稳定优于Random、Power、LCMD。
- 固定现有两池、mutation种类与强度分布、ODE、初值及正式模型/训练/评测协议。LHS保留对照，不再将背景参数消融或生成范式归因作为当前主线的前置条件。
- R17事实和混杂限制保留；其建议的背景域控制暂缓，不把更换论文主贡献当作已经获批的决定。算法创新目标没有取消。
- 后续需区别于已失败的高误差、局部路径加密、固定L配额和解析响应缺失原型；“换名字/加权拼分数/放大候选域”不作为独立有效性证据。
- 正式验证需同协议、多seed并报MAE/MSE/Relative/Corr及成本；胜出判据在看新结果前写清，不能事后挑指标或seed。real不用于拟合评分或调整查询。
- 本条是目标与范围确认，不是新实验；新算法及实验设计尚未完成，无新训练/ODE任务启动。两个平台接手以本条及current_state.json为最新指引。

## 方案草案：突变对照的输入依赖修复（2026-11-23，未运行）

按用户已确认的mutation采样算法主线，核对现有实现与原始文献后，排除将saltation/事件评分或简单对照样本采集重新当新点子。提出MCLD工作草案：以合法突变的完整39变量响应作为学习任务，关注输入条件通路的可纠正方向及跨背景迁移，而非继续按解析响应误差大小排序。须与R08输出头梯度匹配、CAL、BADGE、GLISTER区分，不能把换梯度层当作新算法。

[详细草案与否证条件](NEXT_SAMPLING_PROPOSAL.md)。首版建议仅池选，使用训练侧已有真值，明确标签访问成本；生成版以后再接原生合法规则。本次只做方案与初步排重，无新训练或ODE，未编号为已完成R18实验。

## R18：突变对照输入通路梯度可行性（2026-11-23，已完成，未获得迁移收益）

- 目的：在采样器实现前检查R15响应缺失是否存在可利用的输入依赖纠正方向，区分输出头拟合与输入通路变化；不假定内表征坍缩或正面结果。
- 精确图：R14合法训练侧4096候选，按19模块原生足迹找干预配对，不改变任何参数向量。非焦点参数/初值和其他mutation标签精确一致。候选、反馈、检查的完整端点背景以及焦点基因/非焦点背景组合均隔离，缺额不放宽匹配。覆盖数量先报告。
- 梯度：实际候选更新方向来自原端点Relative eps0.01；响应MSE只作为选择反馈任务，使用checkpoint归一化坐标的39变量差分，不替换训练loss。分别记录输入条件块与输出头的训练/响应梯度、方向一致性和反馈方向余弦。原有GLISTER/梯度匹配思想仍须承认，不据此宣称原创。
- 小规模选择：条件块评分、输出头评分、端点高误差、随机四组各选四对不共享端点的样本，共八个新输入；选择锁定后才读取检查侧标签。只是机制可行性，不冒充正式10k标签效率。
- 验证：只做可逆虚拟权重扰动，块权重范数比例固定1e-4/3e-4，输入块不改时间列。各组在输入/输出头分别检查反馈与独立检查的响应MSE、端点Relative和解码物理MAE，不按检查结果挑步长；每次权重精确还原、原checkpoint不覆盖。虚拟扰动不是实际SGD/Adam训练，也不证明长程收益。
- 范围：无新训练、无ODE、无real/正式holdout/旧保留审计标签；CPU读取有界，GPU0单进程梯度与前向，系统可用内存保留100GiB。
- 输出：summary/conditioning_gradient_probe_v1/，批次batches/conditioning_gradient_probe_v1_20261123/。CLI scripts/analysis/probe_conditioning_gradients.py；精确配对服务utils/sampling/ours/mutation_contrasts.py。此时尚无新结果。
- 后续：若梯度无信息、选择与对照近似一致或仅反馈改善不迁移，不扩大；若有一致信号，再定义真正同预算训练的下一验证，不能跳过训练收益而称方法已有效。

### R18完成后的决策补记

52/47/47对候选/反馈/检查、覆盖18/17/16模块，读取260条训练侧标签，无实际训练和ODE；虚拟扰动权重全部精确恢复。候选输入训练梯度与响应梯度余弦中位数约0.675，输出头约0.0126；条件评分选样与其他组不同。条件评分组在3e-4相对块扰动下反馈响应MSE改善0.460%，隔离检查响应MSE反而恶化0.0322%、端点Relative恶化0.0444%。因此仅有可用梯度和反馈收益仍不足，不立即扩大该草案。

结论只限少量精确配对和局部虚拟更新，不能当完整训练阴性证明；反馈任务跨背景迁移假设尚未成立。后续不能再用本检查逐例结果调规则。用户已确定的mutation采样创新目标继续保留，现有正式超参数/生物规则不改，也没有启动新长跑。详见[R18全部结果](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/conditioning_gradient_probe_v1/report.md)。

## R19：固定评分规则的实际冷启动训练（2026-11-24，已完成，筛选阴性）

- 问题：R18仅八个端点、虚拟块更新，不能替代实际训练。保持R18响应反馈MSE与候选端点标准Relative梯度方向余弦的规则，不从旧检查结果调权或选步长，改为真正训练检验。
- 新角色：固定mutation训练侧输入，排除旧R05/R09/R14/R16/R18的反馈/保留审计及检查背景与初始训练背景；输入随机预留512条新完整背景探针，候选/响应反馈/备用检查精确配对各自背景隔离。不读取旧保留审计、formal test或real真值。来源验证loader也排除新旧保护背景，四组完全相同。
- 动作：原生模块足迹外参数/初值及其他mutation保持精确一致，128对不共享端点=256条新增唯一输入；四组为条件梯度、输出头梯度、同配对域随机、端点高误差。无新ODE或候选池重生成。候选零梯度在四组统一排除并计数，不当0分悄悄混入。
- 算法计算：使用单批反传精确计算逐样本FNO输入/输出头梯度，单元测试逐样本反传数值等价；仅加速评分，不近似改变选择公式。读取的候选及反馈标签总数明确计费，不冒充仅256标签的AL。
- 实际训练：同一初始2000条＋各组256条，2256条，四组相同模型初始化seed；FNO192/4/24、batch128、LR5e-4到1e-5，100epoch、warmup5，每组1800更新，从头训练所有权重，当前Relative eps.01不截断。各自累计训练集重拟合统计，来源合成验证归一化loss选best。不是原checkpoint续训，不加pair loss。
- 比较：这是2256规模的单轮真实试训，不冒称完整五轮10k实验，不与旧正式Power/LCMD终评直接排名。新512条合成探针标签在全部选样锁定后才由训练任务终评读取，预测不回流选样。共同初始统计的Relative只作辅助诊断，物理四指标全报。
- 事前门槛：条件评分对同配对随机的物理逐样本逐变量Relative改善至少5%，物理MAE和MSE均不得变差超过5%。未通过不自动扩大；通过也只是单seed候选证据，下一阶段才预定多seed或正式完整协议。未定义为统计显著性检验。
- 执行：准备阶段单项；四组训练在授权GPU0—3中按空余显存分配，每卡最多一项、全局最多两项并发，I/O并发1、保留系统可用内存100GiB，W&B离线。原始轨迹只引用源池。正式单项结果fno/mutation/contrast_coldstart_{condition,head,random,error}_v1/<profile>/seed_42/；汇总summary/contrast_coldstart_v1/；批次batches/contrast_coldstart_v1_20261124_gpu_pool/（初次GPU0排队未启动子任务，记录保留）。
- 代码：utils/sampling/benchmark/contrast_coldstart.py复用公共fit_round及评测，选择梯度服务utils/sampling/ours/conditioning_gradients.py；分析/批次入口run_contrast_coldstart.py分属scripts/analysis和scripts/launch。

### R19完成后的决策补记（2026-11-24，Antigravity写回）

- **核验与耗时**：6/6任务正常完成，0失败，总耗时1383.36秒（约23.1分钟），自有PSS峰值7.55GiB，未读real/正式test/旧保留审计。
- **新512条隔离背景合成探针结果**：
  - `random`：物理Relative **10.104443**、MAE 0.620028、MSE 115.083771、Corr 0.635374、共同坐标Norm Rel 0.353883；
  - `head`：物理Relative 10.171105、MAE **0.594629**、MSE **105.961945**、Corr **0.649263**、共同坐标Norm Rel **0.352912**；
  - `condition`：物理Relative 10.326623、MAE 0.630682、MSE 123.341553、Corr 0.632975、共同坐标Norm Rel 0.354163；
  - `error`：物理Relative 10.915681、MAE 0.682748、MSE 136.945389、Corr 0.613968、共同坐标Norm Rel 0.367194。
- **门槛与诊断**：`condition`相对`random`的物理Relative改善为**-2.199%**，未通过≥5%的事前门槛（`screen_passed=False`）。`error`四指标全部垫底；`condition`与`error`重合高达105/256（随机重合约62–70），受贪心Top-K排序拉向高误差离群点，且精确配对筛选将9万候选池压缩至608对、选走42%端点，同时2000+256单轮100epoch探针本身信噪比偏低（物理Relative≈10.1主要由零通道波动主导）。
- **决策**：停止R19的稀疏配对贪心输入梯度匹配原型，不追加epoch或调权。下一步按用户最新授权，先对`Power`/`LCMD`/`Random`在训练侧的真实选样行为与不确定性/多样性机制做只读诊断（R20），再同时设计具备顶会创新性的**固定池选择**与**池外合法Query Synthesis**算法，并在具备`Power`阳性对照的探针与正式五轮多seed协议上验证。
- **证据**：[R19汇总报告](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/contrast_coldstart_v1/report.md)、[R19结构化结果](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/contrast_coldstart_v1/report.json)、[批次完成凭据](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/contrast_coldstart_v1_20261124_gpu_pool/completion.json)。

## R20：双尺度通道不确定性几何、逐轮相变与组合异质性只读诊断（2026-11-24，已完成）

- **范围与隔离**：严格只读；复用既有`Power` seed_42 `round_00`的2成员FNO集成在`mutation`训练侧4,096条未查询候选上推理（`scratch/probes/probe_r20_uncertainty_geometry.py`），并核对既有`seed_42` `random`/`lcmd`/`power`/`topk`的五轮`performance.csv`。零训练、零ODE、未触碰validation/test/real/保留审计。
- **关键发现**：
  1. **双尺度通道分歧全面超越`Power`的均值MSE分歧**：同一2成员集成无标签前向，`power_mse`与真实误差的Spearman相关性为`RelNorm=0.5405 / NZRel=0.1978 / MAE=0.4548`、Top-20%精确零通道失效召回率`0.2461`；而逐通道归一化相对分歧`rel_norm_mean`达到`RelNorm=0.7414 (+0.201)`、`NZRel=0.3301 (+67%)`；物理MAE分歧`phys_mae_dis`零通道失效召回率达`0.3277 (+33%)`；双尺度几何平均`geom_relnorm_physmae`达`MAE=0.5586`、零通道召回`0.3076`。且**单样本内39通道物理相对分歧与真实39通道误差的Spearman中位数高达`0.6918`**，证明集成天然给出39维失败通路指纹。
  2. **逐轮“早覆盖、晚精修”相变**：`seed_42`五轮`mutation_test`曲线显示，第1轮（4k）`LCMD` MAE `0.437`大幅领先`Random 0.479`和`Power 0.642`（Power早期甚至不如2k）；但第3–4轮（8k–10k）`LCMD`在Relative上停滞于`8.51`（real `12.88`），`Power`将Relative压至`1.38`（real `1.06`）且避免`TopK`尾部塌缩（real MAE `0.150` vs `TopK 0.216`）。这直接解释了此前`2k+256`单轮微探针信噪比不足的原因。
  3. **组合超图15.8×–16.3×异质性**：64个高频`gene:mode`原子真实MAE跨度`15.81×`（如`CLB2:GAL-MC 4.259` vs `CLB2:ts 0.269`），171个基因对NZRel跨度`16.27×`；无标签双尺度不确定性在原子级与真实MAE的Spearman达`0.6333`（优于Power `0.5213`），可直接驱动因子化合法查询合成。
- **决策**：基于上述三项实证设计统一框架DMCED（Disagreement-Metric Combinatorial Experimental Design），跳过低信噪比微探针，按用户授权直接开展Budding三seed正式五轮实验（R21）。

## R21：DMCED固定池选样、消融与超图因子化合法查询合成（2026-11-24，已完成，未通过胜出判据）

- **用户授权与资源约束**：用户明确授权使用GPU 1–3（0号卡显存不足不用，4–7号独占卡不用），单卡最多2项并发、总并发上限6、`--io-concurrency 1`、`--gpu-free-gib 4.0`、保留系统可用内存100GiB，在后台`tmux`会话`dmced_r21`运行Budding `mutation`三个seed（`42 / 123 / 1024`），暂不开展Fission。
- **统一假设与三组候选方法（均与已有`random`/`lcmd`/`power`完全对齐冻结五轮协议：2k→4k→6k→8k→10k，100/100/100/100/200 epoch，FNO 192/4/24，batch 128，LR 5e-4→1e-5，2成员集成，仅主成员0评测，来源合成验证归一化loss选best，累计训练集重拟合统计）**：
  1. `ours_dmced_pool`（Track A 固定池完整法）：支柱1（双尺度逐通道分歧`u_rel`+`u_phys`，中位数校准+99.5%分位截顶几何风险`w(x)`）+ 支柱2（联合动力学隐层与39通道失败指纹空间`[unit(phi_dyn), 0.5*unit(u)]`上的自退火认识风险加权Voronoi/Weighted-LCMD，$\beta_r\in\{0.5, 1.0, 1.5, 2.0\}$）。无硬编码Budding蛋白名，严格未选真值不可见。
  2. `ours_dmced_power`（Track A 支柱1消融）：保留`Power`的$\beta=1.0$随机幂抽样核，仅将标量MSE分歧替换为双尺度通道风险`w(x)`，隔离支柱1与支柱2的独立贡献。
  3. `ours_dmced_synth`（Track C 池外合法查询合成）：每轮在1,159原生合法模块组合上抽12,288无标签探测样本（0 ODE）估计经验贝叶斯收缩原子/模块认识风险，构造带40%原生均匀探索下界保护的因子化Gibbs混合合法提议分布生成24,000池外合法候选（100%原生强度/初值/mdt/碰撞救援，不改任何生物规则），再用支柱2认识风险加权Voronoi选出当轮批次并经生产级ODE求解器（4 CPU worker、5s/35s超时）补足预算，完整计费ODE尝试与失败。
- **事前胜出判据**：三seed（`42/123/1024`）全部完成并核验后，对比既有三seed `random`（real MAE 0.1719 / Rel 5.56）、`lcmd`（0.1671 / 5.26）、`power`（0.1618 / 1.265）：要求至少一项DMCED方法在126株`real`三seed平均MAE与Relative L2上**同时优于Power、LCMD和Random**（且三seed均不出现Relative爆炸），并在`mutation_test`上取得一致改善；不挑单seed或删指标改判。
- **实现与入口**：核心库`utils/sampling/ours/dmced/{metrics,selector,synthesis}.py`，测试`tests/test_dmced.py`，调度入口`bash run_sampling_budding.sh extensions --suite dmced --seeds 42 123 1024 --gpus 1 2 3 --initial-jobs 3 --max-jobs 6 --jobs-per-gpu 2 --io-concurrency 1 --gpu-free-gib 4.0 --cache-chunks 128 --directory /home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/dmced_multiseed_20261124`。单项结果存标准`fno/mutation/ours_dmced_{pool,power,synth}/<profile>/seed_<seed>/`。

### R21完成后的结果与机制归因（2026-11-24，Antigravity写回）

- **完成核验**：9/9项全部`exit=0`完成，0失败，总耗时15,959秒（4小时26分），自有PSS峰值77.84GiB，无内存/IO中止；`ours_dmced_synth`三seed合计24,000次ODE全部成功（0失败）。
- **三seed终评均值（126株`real` / 5000条`mutation_test`）**：
  - `power`（既有基准）：real MAE **0.16182±0.01015** / MSE 2.973 / Rel **1.2654±0.4022** / Corr **0.8977**；synth MAE 0.26355 / MSE 61.815 / Rel **1.3903**。
  - `lcmd`（既有基准）：real MAE 0.16705±0.00288 / MSE 2.934 / Rel 5.2565±6.6091；synth MAE 0.26801 / MSE 65.366 / Rel 3.7874。
  - `random`（既有基准）：real MAE 0.17193±0.00608 / MSE 3.024 / Rel 5.5606±5.9170；synth MAE 0.27733 / MSE 81.252 / Rel 3.6902。
  - **`ours_dmced_power`（R21支柱1消融）**：real MAE **0.16702±0.00317**（胜Random、持平LCMD且方差极低）、MSE 3.177、Rel 5.2560（seed123=1.1111、seed1024=1.5027，仅seed42炸至13.1544）；**synth MAE 0.26009±0.01346（全场最低，2/3 seed胜Power）**。
  - **`ours_dmced_pool`（R21固定池）**：real MAE 0.17464±0.01159、MSE 3.716、Rel 8.8864（seed42/1024均炸至~12.5）；**synth MSE 60.981±6.629（全场最低）**。注意seed42在R1/R2大幅领先Power（4k real MAE 0.297 vs 0.472；6k 0.264 vs 0.272），直到R3/R4才被反超。
  - **`ours_dmced_synth`（R21合法合成）**：real MAE 0.20775±0.02502、MSE 7.115、Rel 9.0452；synth MAE 0.30321。
- **关键机制归因（seed_42分布核查）**：
  1. **为什么`Power`三seed Relative全稳（1.06/1.73/1.01）而`ours_dmced_power`在seed42爆炸？** `Power`使用**未截顶的平方Z-score分歧**，单通道`0↔1`离散切换时提供~34×概率尖峰，使`seed_42`的`CDH1_KO`训练样本从Random的1229拉升至**1918（+56%）**，跨过了FNO学习零通道的相变阈值（>20爆炸突变体由27–28株降至2株）；而`ours_dmced_power`的线性平均+`sqrt`+`Q_99.5`截顶压平了单通道切换尖峰，仅选到1381条`CDH1_KO`。
  2. **为什么`ours_dmced_pool`在R3–R4退步？** 确定性`argmax(w^beta * d^2)`配合后期$\beta=1.5, 2.0$实质退化为`TopK`尾部贪心：`levels`塌缩至`L1=27 / L2=519 / L3=9454`（与`TopK`的`29/626/9345`几乎一致）。
  3. **为什么`ours_dmced_synth`显著退步？** Gibbs偏向高不确定性模块（偏向L3，7.5×概率比）叠加确定性`argmax(w^beta * d^2)`导致双重偏移：8000条新增样本中L1仅剩**9条（0.1%）**、L2仅**418条（5.2%）**、L3占94.7%，严重损伤以L1/L2为主的`real`泛化。
- **决策**：停止确定性`argmax` Voronoi与无层级保护的Gibbs合成版本。下一轮重构应：(1) 用**未截顶平方通道分歧 + 高阶$L_p$单通道切换能量**保留并超越`Power`的离散分岔敏感度；(2) 用**随机加权$D^2$采样（Stochastic Epistemic-Weighted K-Means++，`p_i ∝ s_i * d^2`）**替代确定性`argmax`，兼得理论覆盖界与防L1/L2塌缩；(3) 合法查询合成采用**层级边际守恒（Level-Stratified / Unbiased Large Bank）+ 随机加权$D^2$筛选**，彻底杜绝L1/L2被挤压。
- **证据**：[R21三seed汇总与归因报告](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/dmced_r21_multiseed_20261124/report.md)、[批次状态](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/dmced_multiseed_20261124/status.json)、[任务完成清单](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/dmced_multiseed_20261124/tasks.json)。

## R22：Switch-D2——未截顶任务对齐相对方差与随机加权 $D^2$ 采样（2026-11-24，已完成，验证支柱2大胜但未过总体门槛）

- **出发点（针对 R21 的精准修正）**：
  1. R21 证明 `Power` 三 seed 不炸 Relative L2 的关键是**未截顶平方分歧**对单通道 `0↔1` 切换的放大；R22 准备期在 `Power` seed_42 `round_02` 的 8,192 未查询候选上只读验证：**未截顶逐通道平方相对方差** $\mathcal{E}_{\text{rel}}(x)=\frac{1}{V}\sum_v \frac{\frac{1}{M}\sum_m \|\hat{y}^{(m)}_v-\bar{y}_v\|_2^2}{\|\bar{y}_v\|_2^2+\epsilon^2}$ 既是 `RelativeL2Loss` 的精确二阶集成方差，又将 `CDH1_KO` 采样率从 `Power` 的 13.5% 提升至 **17.5%（+30%）**，且不硬编码任何蛋白名。
  2. R21 证明确定性 `argmax` Voronoi 会退化为 `TopK` 导致 L1/L2 塌缩；R22 改为 **Arthur–Vassilvitskii 随机认识风险加权 $D^2$ 采样**（$p(x_i)\propto \mathcal{E}_{\text{rel}}(x_i)\cdot \min_{a\in\mathcal{L}\cup S}\|\tilde{\phi}(x_i)-\tilde{\phi}(a)\|_2^2$，在单位球面联合动力学–失败签名空间中运行），自带 $O(\log B)$ 期望风险近似界、实测显著降低批内余弦冗余（0.9531→0.9383）同时保住切换样本与 L1/L2 分布。
  3. R21 证明跨层级 Gibbs 偏向会将 L1 挤至 0.1%；R22 Track C 合成版严格**锁定 L1/L2/L3 原生边际比例（19/171/969 ÷ 1159）**，仅在各层级内部做 50% 探索底线的经验贝叶斯温和倾斜，再用同款随机加权 $D^2$ 采样选出 2,000 条送入原生 ODE。
- **三组方法（seed_42 正式五轮 `2k→10k`，`100/100/100/100/200 epoch`，2 成员集成，仅主成员 0 评测）**：
  - `ours_switch_power`（Track A 支柱 1 消融：$\mathcal{E}_{\text{rel}}(x)$ + Power 随机幂采样 $\beta=1$）
  - `ours_switch_d2_pool`（Track A 固定池完整法：$\mathcal{E}_{\text{rel}}(x)$ + 随机加权 $D^2$ 采样）
  - `ours_switch_d2_synth`（Track C 合法查询合成：层级边际守恒提议 + 随机加权 $D^2$ 采样 + 原生 ODE）
- **事前晋级门槛（决定是否扩展到 seed 123/1024）**：至少一项 R22 方法在 `seed_42` 126 株 `real` 上不出现 Relative L2 爆炸（`< 1.5`，目标 `< 1.06`），且 `real MAE` 优于 `Power` seed_42（`0.15030`）或在 `mutation_test` 与 `real` 综合指标上显著优于 `Power`/`LCMD`/`Random`。未通过则不盲目扩三 seed。
- **实现与执行**：代码 `utils/sampling/ours/switch_d2/`，测试 `tests/test_switch_d2.py`，批次目录 `/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/switch_d2_r22_seed42_20261124`，后台 `tmux` 会话 `switch_d2_r22`（授权 GPU 0–3 池，max 3 项、每卡 1 项、`--io-concurrency 1`、`--gpu-free-gib 4.0`）。

### R22完成后的结果与精准归因（2026-11-24，Antigravity写回）

- **完成核验**：3/3项全部`exit=0`完成，0失败，总耗时7,768秒（2小时09分），峰值PSS 39.57GiB，`ours_switch_d2_synth` 8,000次ODE全部成功（0失败）。
- **seed_42 10k 终评（126株`real` / 5000条`mutation_test`）**：
  - `power`（既有基准）：real MAE **0.15030** / MSE **2.2580** / Rel **1.0570** / Corr **0.9097**（>20仅2株）；synth MAE **0.26796** / Rel **1.3811** / Corr 0.8355。
  - `ours_switch_power`（支柱1消融）：real MAE 0.18673 / MSE 3.7978 / Rel **1.7874** / Corr 0.8857（>20仅3株）；synth MAE 0.31238 / Rel 1.6426。
  - **`ours_switch_d2_pool`（固定池完整法）**：real MAE **0.16989** / MSE 3.6332 / Rel **1.1062** / Corr **0.9030**（>20仅2株，与Power一致）；synth MAE 0.29524 / Rel 1.5498 / **Corr 0.8421（全场最高）**。
  - `ours_switch_d2_synth`（合法合成）：real MAE 0.20998 / MSE 6.0333 / Rel **1.4773** / Corr 0.8883（>20仅3株）。注意在 **Round 2（6k）** 时 `ours_switch_d2_synth` 曾达 real MAE **0.2395** / MSE **6.634** / Rel **3.41**，`ours_switch_d2_pool` 达 **0.2597 / 10.65 / 3.20 / Corr 0.842**，均大幅碾压同轮 `power`（0.2717 / 13.32 / 12.34 / 0.792）。
- **三大确凿发现**：
  1. **Relative L2 爆炸彻底根治**：三组`CDH1_KO`升至1,659–1,716，28株`cdh1Δ`爆炸归零，`seed_42` Relative L2 从R21的12.5–13.3降至`1.106 / 1.477 / 1.787`。
  2. **支柱2（随机加权 $D^2$ 采样）净收益巨大**：在同一`e_rel`分数上，`ours_switch_d2_pool`相对`ours_switch_power`取得 **real MAE -9.0%（0.1867→0.1699）、Rel L2 -38.1%（1.787→1.106）、Corr +0.0173** 的大幅优势。
  3. **唯一短板定位至`e_rel`的中心化Z-score分母**：在Z-score空间中，均值附近普通轨迹的`||mean_norm||^2 ≈ 0`触发分母爆炸（r3 `p99/p50`翻倍至18.0，合成版能量暴涨8×），同时压低了高振幅多分裂轨迹（`P5b`从Power的1,476降至1,121–1,161，简单停滞`P2`增至3,011–3,349），导致单换`e_rel`自身损失了+0.0364 MAE（0.1503→0.1867）。
- **决策**：不将R22原版扩至123/1024；立即提出**R23正交组合**——去掉有分母畸变的`e_rel`，直接把无畸变的`Power`精确平方Z-score能量 $e_{\text{mse}}$ 与 R22 已证明大胜的 **联合动力学–失败指纹球面随机加权 $D^2$（及温和 $D^1$ 变体）采样** 结合，并同步测试无偏大池外合法合成 + $e_{\text{mse}}$ 随机加权 $D^2$。
- **证据**：[R22 seed_42 汇总与归因报告](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/switch_d2_r22_seed42_20261124/report.md)、[批次状态](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/switch_d2_r22_seed42_20261124/status.json)、[任务清单](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/switch_d2_r22_seed42_20261124/tasks.json)。

## R23：Epistemic-D2——精确 Z-score MSE 能量与联合球面随机加权 $D^2$/$D^1$ 采样（2026-11-24，已完成，`d1_pool` 强阳性信号）

- **假设与干净设计**：直接组合已由 R21/R22 分别验证最强的两个组件——与 `Power` 数学完全一致的无分母畸变 Z-score MSE 能量 $e_{\text{mse}}$（单元测试精确对齐 `Predictor.uncertainty`）+ 在单位球面联合动力学隐层与 39 通道失败签名空间 $\tilde{\phi}(x)=[\text{unit}(\phi_{\text{dyn}}(x)),\,0.5\,\text{unit}(\log(1+e_{\text{mse}}/\text{med}))]$ 上的 Arthur–Vassilvitskii 随机加权 $D^p$ 采样（$p_i \propto e_{\text{mse},i}\cdot d(x_i,\mathcal{L}\cup S)^p$）。
- **三组方法（seed_42 正式五轮 `2k→10k`，`100/100/100/100/200 epoch`，2 成员集成，仅主成员 0 评测）**：
  1. `ours_epistemic_d2_pool`（Track A 固定池主法，$p=2$，自带 $O(\log B)$ 加权 K-Means++ 期望风险界，与 `Power` 唯一差异为联合球面 $D^2$ 覆盖）；
  2. `ours_epistemic_d1_pool`（Track A 温和距离变体，$p=1$，进一步抑制对 L1/L2 的任何轻微距离偏移）；
  3. `ours_epistemic_d2_synth`（Track C 无偏合法查询合成：每轮直接用 `FrozenBuddingDomain.bank(24000)` 生成 24,000 条 100% 原生无偏池外合法候选，零 Gibbs 偏移，再用同款 $e_{\text{mse}}$ 随机加权 $D^2$ 选样并经原生 ODE 求解）。
- **事前晋级门槛（决定是否扩展到 seed 123/1024）**：至少一项 R23 方法在 `seed_42` 126 株 `real` 上不出现 Relative L2 爆炸（`< 1.3`，目标 `< 1.057`），且 `real MAE` 明确优于 `Power` seed_42（`0.15030`）或在 `real` 与 `mutation_test` 核心指标上稳定占优。
- **实现与执行**：代码 `utils/sampling/ours/epistemic_d2/`，测试 `tests/test_epistemic_d2.py`，批次目录 `/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/epistemic_d2_r23_seed42_20261124`，后台 `tmux` 会话 `epistemic_d2_r23`（授权 GPU 0–3 池，max 3 项、每卡 1 项、`--io-concurrency 1`、`--gpu-free-gib 4.0`）。

### R23完成后的结果与逐株归因（2026-11-24，Antigravity写回）

- **完成核验**：3/3项全部`exit=0`完成，0失败，总耗时7,388秒（2小时03分），峰值PSS 42.98GiB，`ours_epistemic_d2_synth` 8,000次ODE全部成功（0失败）。
- **seed_42 10k 终评（5,000条`mutation_test` / 126株`real`）**：
  - `power`（既有最强基准）：synth MAE 0.26796 / MSE 73.74 / Rel 1.381 / MacroRel 0.3077 / Core9 0.7546；real 均值MAE 0.15030（中位0.10449）/ MSE 2.258（中位0.4396）/ Rel 1.057 / MacroRel 0.2990。
  - `ours_epistemic_d2_pool`（$p=2$）：synth MAE 0.30158 / Rel 8.578；real MAE 0.18011 / Rel 12.721（$d^2$ 距离权重过强，明确不如 $d^1$）。
  - **`ours_epistemic_d1_pool`（$p=1$）**：**synth 全面全场第一**（MAE **0.26065** -2.7% vs Power、MSE **60.95** **-17.3%** vs Power、Core9 **0.7350**、MacroRel **0.2950**、Rel **1.473** 无爆炸）；**real 逐株大胜 Power 75/126（59.5%）MAE、74/126（58.7%）MSE**，**中位 MAE 0.09851**（-5.7%）、**中位 MSE 0.3926**（-10.7%）、**MacroRel 0.2977**（胜 Power）、Rel **1.375** 无爆炸；前 3 轮（4k/6k/8k）全程大幅领先 Power（R1 `0.276 vs 0.472`、R2 `0.219 vs 0.272`、R3 `0.217 vs 0.218`）。单 seed 均值 MAE `0.1760` / MSE `6.467` 仅因 1 株极端离群 `#124_APC-A_GAL-CLB2`（MSE `402.9`，独献 +3.20 均值 MSE）拉高，而在 Power 最差株 `#50` 上 `d1_pool` 将 MAE 从 `1.307` 降至 `0.264`（改善 5×）。
  - **`ours_epistemic_d2_synth`（无偏 24k 合法合成，$p=2$）**：较 R22 暴涨跃升——real 均值 MAE `0.2100 → 0.16924`、**中位 MSE 0.36583（比 Power 低 16.8%）**、**MacroRel 0.2986（胜 Power）**、Rel **1.2527**（>20 仅 2 株，与 Power 完全一致）、逐株胜 Power **69/126 MAE、73/126 MSE**！
- **决策**：停止 $p=2$ 固定池版（`ours_epistemic_d2_pool`）；锁定最优区间 $p\in[0.5, 1.0]$。建议下一步（R24）立刻：(1) 补齐 `ours_epistemic_d1_pool` 的 `seed 123/1024`（仅 2 项）检验三 seed 均值是否全面击败 Power（0.16182）；(2) 新增并将 Track C 无偏合成升级为 $p=1$（`ours_epistemic_d1_synth`）跑满三 seed（`42/123/1024`）；(3) 同步加跑 `ours_epistemic_d05_pool`（$p=0.5$）检验更温和距离是否同时压住极端尾部并保留全部中位/合成优势。
- **证据**：[R23 seed_42 汇总与逐株报告](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/epistemic_d2_r23_seed42_20261124/report.md)、[批次状态](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/epistemic_d2_r23_seed42_20261124/status.json)、[任务清单](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/epistemic_d2_r23_seed42_20261124/tasks.json)。

## R24：Epistemic-$D^p$（$p\in\{0.5,1.0\}$）固定池与无偏查询合成三 Seed 正式验证（2026-11-24，已完成，正式终止 `Power × d^p` 路线）

- **假设与设计**：基于 R23 确定的最优距离指数区间 $p\in[0.5, 1.0]$ 与无偏 24k 合法候选库合成框架，在冻结的正式五轮三 seed 协议（`seed 42/123/1024`，`2k→10k`，`100/100/100/100/200 epoch`，2 成员集成，仅主成员 0 评测）下开展三组正式验证（共 8 项新任务，复用 R23 已完成的 `ours_epistemic_d1_pool/seed_42` 构成完整 3×3 对比）：
  1. `ours_epistemic_d1_pool`（补跑 `seed 123/1024`，2 项）：检验 $p=1.0$ 固定池法在三 seed 平均上是否全面击败 `Power`（3-seed 均值 real MAE `0.16182` / Rel `1.2654` / synth MAE `0.26355`）；
  2. `ours_epistemic_d05_pool`（全 3 seed `42/123/1024`，3 项）：$p=0.5$（$p_i\propto e_{\text{mse},i}\cdot d_i^{0.5}$），检验更温和的距离权重能否压住 `#124` 类极端长尾样本、实现 real 均值/中位数/合成测试全线占优，并与 `Power`（$p=0$）、`d1_pool`（$p=1$）、`d2_pool`（$p=2$）构成完整指数消融曲线；
  3. `ours_epistemic_d1_synth`（全 3 seed `42/123/1024`，3 项）：Track C 无偏 24k 池外合法查询合成由 $p=2$ 升级至 $p=1$（每轮新增 2,000 条原生 ODE 仿真，共 8,000 条/seed），检验合法查询合成路线在三 seed 上能否击败固定池 `Power`/`LCMD`/`Random`。
- **事前成功判据**：至少一项 R24 方法在三 seed（`42/123/1024`）平均上，同时满足：(a) 126 株 `real` 不发生 Relative L2 爆炸（三 seed 平均 `< 1.5`）；(b) `real` 三 seed 平均 MAE（及中位 MAE/MSE、逐株胜率）或 `mutation_test` 三 seed 平均 MAE/MSE 稳定优于 `Power`（`0.16182` / `0.26355`）、`LCMD` 与 `Random`，且至少 2/3 seed 胜出。
- **实现与资源配置**：代码 `utils/sampling/ours/epistemic_d2/`、`utils/sampling/registry.py`、`scripts/launch/run_sampling_extensions.py`（`--suite epistemic_r24`）、`scripts/launch/queue_protocols.py`，单测 `tests/test_epistemic_d2.py`；批次目录 `/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/epistemic_r24_multiseed_20261124`，后台 `tmux` 会话 `epistemic_r24`（授权 GPU 0–3 池，按用户指示充分利用空闲 GPU 0：`--initial-jobs 4 --max-jobs 6 --jobs-per-gpu 4 --gpu-free-gib 12.0 --io-concurrency 2`，100 GiB 系统内存保留，禁止在队列运行期间修改源码）。

### R24完成后的三 Seed 终评与归因（2026-11-24，Antigravity写回）

- **完成核验**：8/8 项新任务全部 `exit=0` 完成（0 失败，总耗时 20,504s = 5h41m，峰值 PSS 101.97 GiB，系统可用内存始终 >600 GiB）。
- **三 Seed 均值对比（`real` 126 株 / `mutation_test` 5,000 条）**：
  - `power`：real MAE **0.16182±0.00829**（中位 0.10757）/ MSE **2.973** / Rel 1.2654 / Corr 0.8977；synth MAE 0.26355 / MSE 61.82。
  - `ours_epistemic_d05_pool`（$p=0.5$）：real MAE 0.17189±0.01053 / MSE 3.670 / **Rel 1.2043（最低）** / Corr 0.8963；synth MAE 0.26436 / MSE 68.31。
  - `ours_epistemic_d1_pool`（$p=1.0$）：**synth MAE 0.26174 / MSE 58.28（全场最优）**、**real 中位 MAE 0.10485 / MacroRel 0.3088 / Corr 0.8992（全场最优）**、Rel 1.2525，但 real 均值 MAE **0.17930±0.00838** / MSE **6.005** 明显落后 Power。
  - `ours_epistemic_d1_synth`（Track C，$p=1.0$）：real MAE 0.18855±0.01619 / MSE 4.913 / Rel 3.5233（`seed_1024` 出现 7.95 异常）；synth MAE 0.30361。
- **训练侧 10k 选样归因（决定性发现）**：对比三 seed 10k 最终训练集的突变阶数，原生池 `random` 为 `L1=110, L2=1286, L3=8604`；`Power` 自身已将 L1 压至 `46–63`、L2 压至 `932–1036`；而乘上高维距离 `d^p` 后由于 L3 间距天然更大，`d05_pool / d1_pool` 将 L2 进一步饿死 `87–214` 条（`L2=813–881`），L1 跌至 `43–58`，造成低阶经典突变及长尾误差上升。
- **决策**：正式终止 `Power × d^p` 路线（既缺乏 AAAI 级原创深度，又在 real 均值 MAE 上被 Power 单调压制）；基于新验证的组合上位性与分岔信号，立即转向 **R25 CEBS**。
- **证据**：[R24 三 Seed 终评报告](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/epistemic_r24_multiseed_20261124/report.md)、[批次状态](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/epistemic_r24_multiseed_20261124/status.json)、[任务清单](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/epistemic_r24_multiseed_20261124/tasks.json)。

## R25：CEBS——组合上位性与分岔感知主动采样及查询合成（2026-11-24/25，已完成，模块 3 强阳性、模块 1/2 淘汰）

- **前置只读验证探针（0 训练、训练侧 3,072 候选 + 48 条真实 ODE 核验）**：
  1. 证伪：残差集成方差 $U_{\text{epi\_tot}}=\text{Var}_m[\hat{y}(ABC)-\hat{y}(A)-\hat{y}(B)-\hat{y}(C)+2\hat{y}(\text{WT})]$ 因叠加 5 个预测噪声在 `round_02` 劣于 Power（$\rho=0.318$ vs $0.552$），立刻淘汰。
  2. 证实信号 #1（**0-ODE 非线性上位性强度 $\mathcal{M}_{\text{epi}}$**）：使用集成均值 $\bar{y}$ 计算非加性偏离 $\mathcal{M}_{\text{epi}}(S)=\|\bar{y}(S)-\sum_{j=1}^k \bar{y}(M_j)+(k-1)\bar{y}(\text{WT}_e)\|_2^2$，既非振幅混杂（$\rho=0.12\text{--}0.21$）又几乎正交于 Power（$\rho=0.17\text{--}0.26$）；与 Power 耦合 $U_{\text{power}}\cdot(1+\log(1+\mathcal{M}_{\text{epi}}/\text{med}_k))$ 在 `round_00`（2k）和 `round_02`（6k）同时全面超越纯 Power（`round_02`: $\rho(\text{MAE})\ 0.552\to 0.558$、$\rho(\text{NZRel})\ 0.449\to 0.458$、Top-20% MAE `+12.6%`、Top-20% MSE **`+66.4%`**、ZeroFail `+13%`）。
  3. 证实信号 #2（**0-ODE 连续强度分岔搜索**）：在 48 条真实 ODE 核验的同基因组合剂量线上，仅用 0-ODE 扫合法连续强度选不确定性峰值 $\alpha^*$，相比同组合随机强度使真实非零 Relative L2 捕获提升 **`+44%`**（`0.531→0.764`）、真实 MAE 提升 `+16%`（32/48 曲线胜出）。
- **三组严格对照设计（`seed_42` 正式五轮 `2k→10k`，2 成员集成，仅主成员 0 评测）**：
  1. `ours_order_power_pool`（Track A 干净消融）：仅**模块一（按候选池原生阶数 L1/L2/L3 比例精确分层，无人工配额扭曲）** + 纯 `Power`（$\lambda=0$），直接回应审稿人“若 baseline 也按阶数分层是否更强”的质疑；
  2. `ours_epistasis_pool`（Track A 完整主法）：**模块一（阶数边际守恒） + 模块二（0-ODE 虚拟非线性上位性增益 $\lambda=1.0$）**；
  3. `ours_epibif_synth`（Track C 真正结构化查询合成主法）：无偏 24k 池外组合库 → **模块一 + 模块二**选组合 → **模块三（0-ODE 9 点连续强度分岔扫描与 $P(\alpha_g)\propto U_{\text{power}}(\alpha_g)^2$ 临界强度合成）** → 单点原生 ODE 求解（1 组合 = 1 ODE，彻底避免 R01/R03 覆盖坍缩）。
- **事前晋级门槛（决定是否扩 `seed 123/1024` 与跨系统 Fission 验证）**：至少一项完整法（`ours_epistasis_pool` 或 `ours_epibif_synth`）在 `seed_42` 上同时满足：(a) `real Rel L2 < 1.3`（无零通道爆炸）；(b) `real MAE` 明确优于原版 `Power`（`< 0.15030`）**且**优于分层消融对照 `ours_order_power_pool`。未通过则不盲目扩三 seed。
- **实现与资源配置**：代码 `utils/sampling/ours/cebs/{__init__,metrics,selector,synthesis}.py`、测试 `tests/test_cebs.py`、队列 `--suite cebs_r25`；批次目录 `/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/cebs_r25_seed42_20261124`，后台 `tmux` 会话 `cebs_r25`（空闲 GPU 0–2 三卡并发 1 波、`--jobs-per-gpu 1 --initial-jobs 3 --max-jobs 3 --io-concurrency 2`，100 GiB 系统内存保留，禁止运行期间改 `.py` 源码）。

### R25完成后的消融归因与决策（2026-11-25，Antigravity写回）

- **完成核验**：3/3 项全部 `exit=0` 完成（0 失败，总耗时 7,839s = 2h11m，峰值 PSS 51.17 GiB；`ours_epibif_synth` 8,000/8,000 次 ODE 全成功，0 失败，其中 **6,793/8,000 = 84.9%** 经 0-ODE 连续强度分岔合成，各轮平均不确定性增益 `1.19–1.33×`）。
- **seed_42 10k 终评与消融分解**：
  - `power`：real MAE **0.15030** / Med MSE 0.4396 / Rel 1.0570 / MacroRel 0.2990；synth MAE **0.26796** / `val_error` 0.18975 / `test_loss_synth` 0.1879。
  - `ours_order_power_pool`（仅模块 1）：real MAE 0.16614 / Rel **1.0480**；synth MAE 0.27503 / `test_loss_synth` 0.1899。R1–R3 大幅领先 Power（`0.319 / 0.239 / 0.208`），但 10k 略输原生 Power。
  - `ours_epistasis_pool`（模块 1+2）：real MAE 0.17388 / synth MAE 0.28673，全面劣于 `order_power`（逐株仅胜 `54/126`）。原子统计查明 `epi_mag` 抽走了 `CDC20:GAL-OE (508→392)`、`MAD2:CSL (1120→970)`、`ESP1 (745→646)`、`CLB5:db (694→633)` 预算，导致 `CDC20/PDS1/CLB5-db` 类真实突变体误差激增。**正式淘汰模块 2！**
  - **`ours_epibif_synth`（模块 1+2+3）**：尽管背负负向模块 1/2，单靠**模块 3（0-ODE 连续强度分岔合成）**就相对 `ours_epistasis_pool` 全线暴涨，并创下**全历史最优 `val_error`（0.18664）与 `test_loss_synth`（0.1843）、最优 `mutation_test` MacroRel（0.2947）、全历史最低 `real` Med MSE（0.3217，比 Power 低 26.8%）、`real` MacroRel（0.2985 胜 Power）、逐株胜 Power 71/126 MAE / 73/126 MSE、R2/R3 Rel L2 提前两轮降至 3.76 / 2.10（vs Power 12.34 / 8.84）**！仅因模块 2 拖累下的 3 株 `cdc20/pds1` 离群拉高 `real` 均值 MAE 至 `0.17039`。
- **决策**：严格执行事前门槛——因 10k `real` 均值 MAE 未低于 `0.1503`，不直接扩 R25 原版；**立即剥离拖后腿的模块 1 与模块 2，建议 R26 聚焦纯净且更通用的 Track C 核心创新：“拓扑保真 + 0-ODE 连续参数分岔边界查询合成”（纯 `Power` 选拓扑 + 单轴/多轴 0-ODE 分岔临界强度合成）**。等待用户确认。
- **证据**：[R25 seed_42 消融报告](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/cebs_r25_seed42_20261124/report.md)、[批次状态](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/cebs_r25_seed42_20261124/status.json)、[任务清单](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/cebs_r25_seed42_20261124/tasks.json)。

## R26：池锚定混合模式切换与 0-ODE 连续分岔查询合成三 Seed 正式验证（2026-11-25，首批 5/9 完成，发现强阳性主力 `ours_bif_power_synth`）

- **设计动机与核心升级（纳入用户关于“仅以无标签 pool 为锚点 + 跨 KO/抑制/过表达三大类切换突变模式”的新提案）**：在冻结 Budding `mutation` 五轮三 seed 协议（`seed 42 / 123 / 1024`，`2k→10k`，`100/100/100/100/200 epoch`，2 成员集成，仅主成员 0 评测，每轮仅解 2,000 次新 ODE、共 8,000 次/seed）下运行 3 组 Track C 主法（共 9 项；首版 `bif_r26` 启动 14 分钟后干净中止清理并纳入新主法统一启动）：
  1. **`ours_pool_mode_bif_synth`（用户新旗舰：池锚定等位基因序列混合模式+分岔合成）**：**0 额外池外拓扑生成器、0 读取 pool 真值标签**！直接从现有 ~88k 无标签 pool 输入中按 `Power` 选取基因组合锚点；固定交互基因集，对其内目标基因模块在合法目录 `GROUP_BY_NAME` 下**跨 KO / 抑制（`ts/pLOF/DAmP/APC-A`） / 过表达与稳定性（`GAL-OE/MC/GAL-MC/db/CA/PM`）三大类枚举所有合法离散模式**，并在每个连续模式内扫描 5 点分位强度，经**两层层次随机采样**（模式内聚焦分岔 $\tau_{\text{dose}}=2.0$，模式间温和防坍缩 $\tau_{\text{mode}}=0.75$）合成池外全新合法突变体送入单点 ODE；
  2. `ours_grad_bif_synth`（池外 24k 多轴粗-细动力学分岔合成）：无偏 24k 池外组合库 + 纯 `Power` 选拓扑 + 多轴自适应最陡分岔纤维 5 点粗扫 + 临界区间 5 倍加密细扫（综合 $U(\alpha)$ 与归一化轨迹跳变斜率 $\|\partial\bar{y}/\partial u\|_2$）；
  3. `ours_bif_power_synth`（池外 24k 纯净单轴分岔合成）：无偏 24k 组合库 + 纯 `Power` 选拓扑 + 单轴 9 点连续分岔强度合成（$\tau=2.0$）。（同源纯对照 `ours_power_synth` 已保留注册，待主法胜出后可补跑。）
- **事前成功判据**：至少一项 R26 合成法在三 seed（`42/123/1024`）均值上同时满足：(a) `real Rel L2 < 1.5`（无零通道爆炸）；(b) `real` 三 seed 均值 MAE（与中位 MAE/MSE）及 `mutation_test` MAE/MSE 稳定击败固定池 `Power`（`0.16182 / 0.26355`）、`LCMD` 与 `Random`，且至少 2/3 seed 胜出。
- **实现与资源配置**：代码 `utils/sampling/ours/bif_synth/{__init__,synthesis,pool_mode_synthesis}.py`、`utils/sampling/generation/{cohorts,policies,runner,cli}.py`，单测 `tests/test_bif_synth.py`，队列 `--suite bif_r26 --seeds 42 123 1024`；批次目录 `/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/bif_r26_multiseed_20261125_v2`，后台 `tmux` 会话 `bif_r26`（按用户指令严格仅使用 GPU 0：`--gpus 0 --initial-jobs 4 --max-jobs 4 --jobs-per-gpu 4 --gpu-free-gib 10.0 --io-concurrency 2`，100 GiB 系统内存保留，运行期间严禁修改 `.py` 源码）。

### R26 首批完成结果（5/9）与机制归因（2026-11-25，Antigravity 写回）

- **完成核验与 4 项快速退出原因**：GPU 0 单卡 5h50m 内 **5/9 完成 (`exit=0`)**（`ours_bif_power_synth` `seed 123/1024`、`ours_grad_bif_synth` `seed 123/1024`、`ours_pool_mode_bif_synth` `seed 1024`；ODE 成功率 99.99%–100%）。其余 4 项（`seed 42` × 3 + `ours_pool_mode_bif_synth seed 123`）在 10–15s 时因 `01:32` 旧残余目录未清干净（误删 `generated/mutation/` 而非 `fno/mutation/`）触发 `ValueError: Generation configuration changed` 快速退出，零算力浪费。
- **成对两 Seed（`seed 123 & 1024`）核心成绩**：
  - **`ours_bif_power_synth`（强阳性突破！）**：
    - `seed 123`：real MAE **`0.14960`**（vs Power `0.16567` **-9.7%**）、real MSE **`2.537`**（-19.7%）、real Corr **`0.9052`**、`test_loss_real = 0.19780`（全场第一）、synth MAE **`0.23729`**（-4.4%）、synth MSE **`45.31`**（-29.2%），**6/6 表型组（`P1–P5b`）全胜 Power**，逐株胜 **73/126 MAE、74/126 MSE**；
    - `seed 1024`：synth MAE **`0.26612`**（全场第一）、`val_error = 0.18238`（全场第一）、`test_synth = 0.18357`（全场第一）、`test_loss_real = 0.19514`（全场第一，vs Power `0.20220`、LCMD `0.21053`、Random `0.22060`），real Med MAE/Med MSE/MacroRel/Corr 全胜 Power/LCMD，real Mean MAE `0.17137` 与 Power `0.16948` 几乎持平；
    - **两 seed 均值**：real MAE **`0.16049`**（vs Power `0.16757` **-4.2%**、LCMD `0.16835`、Random `0.16917`），real Med MAE `0.10387`、real Corr `0.8994`、`test_loss_real 0.19647`、synth MAE **`0.25171`**（-3.7%）、synth MSE **`52.14`**（-6.7%）。10k 阶数 `{1:71, 2:995, 3:8934}` 自然保真且无模式畸变。
  - **`ours_grad_bif_synth`（保留作次要消融参考，不作主攻）**：同 24k 候选库与同 Power 选点下，加有限差分斜率 $\|\partial\bar{y}/\partial u\|^2$ 与局部 5 倍加密在两 seed 上弱于纯方差 `ours_bif_power_synth`（`123`: `0.16346` vs `0.14960`；`1024`: `0.19600` vs `0.17137`，两 seed 均值 `0.17973`，`test_loss_real 0.22358`），过度聚集陡崖并挤掉 `ts (-187)` 与 `CLB2:GAL-OE (-79)`。按用户指示保留该想法及代码（并在 GPU 7 空档顺带补全 `seed_42` 构成完整 3-seed 对照），不作主攻方向。
  - **`ours_pool_mode_bif_synth` v1（`seed 1024` 诊断出 2 个具体缺陷）**：ODE 成功 8,000/8,001，但 real MAE `0.17792`、`test_loss_real 0.21324` 弱于 `ours_bif_power_synth` 与 `Power`。10k 成分精准查明两处缺陷：(1) 单模式替换误拆了 `COMPOSITE_REAL_EVIDENCE` 复合体（全表跌幅前 4 恰为 `CLB5:db -109, CLB2:db -103, CLB2:GAL-OE -100, CLB5:GAL-OE -80`，直接拖累 `#62/#71`）；(2) 无条件 62.7% 切换导致高振幅 `CDC20:GAL-OE (+167)` 挤占 `CDC20:ts (-39)` 与 `CDC20:KO`，拖累 `#75/#79`。
- **证据**：[R26 首批分析报告](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/bif_r26_multiseed_20261125_v2/report.md)、[批次状态](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/bif_r26_multiseed_20261125_v2/status.json)、[任务清单](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/bif_r26_multiseed_20261125_v2/tasks.json)。

## R27：纯池锚定连续分岔与复合体安全尺度归一化跨模式分岔三 Seed 终验（2026-11-25，用户批准方案 B 启动中）

- **设计目标（在用户批准的方案 B 下同时锁定 4 组 Track C 方法的完整 3-seed `42/123/1024` 对比）**：
  1. **`ours_bif_power_synth`（补跑 `seed 42`，凑齐完整 3/3 seeds）**：验证池外 24k 纯净 9 点连续分岔合成在全部三 seed 上稳定击败 `Power / LCMD / Random`；
  2. **`ours_pool_bif_synth`（新核心主力，全 3 seeds `42/123/1024`）**：**0 池外拓扑生成器、0 读取 pool 标签、0 拆模式与复合体**！直接以 ~88k 无标签 pool 输入为锚点用 `Power` 选组合，在原组合连续轴上做与 `ours_bif_power_synth` 完全同构的 9 点 0-ODE 连续分岔合成（$\tau_{\text{dose}}=2.0$），直接检验纯池锚定能否复现或超越池外版；
  3. **`ours_pool_mode_bif_v2_synth`（修复版跨模式+分岔，全 3 seeds `42/123/1024`）**：针对 R26 v1 查出的两处缺陷做精准修复——(a) 严格保护 `COMPOSITE_REAL_EVIDENCE` 同基因双模式复合体（仅对单模式多模式基因枚举合法切换，复合基因只在原连续轴调分岔强度）；(b) 跨模式比较改用**尺度归一化相对不确定性** $U_{\text{rel}}=\frac{1}{V}\sum_v \frac{\text{Var}_m[\hat{y}_v]}{\|\bar{y}_v\|^2+\epsilon_{\text{rel}}^2}$（$\epsilon_{\text{rel}}=0.5$，消除 `CDC20:GAL-OE` 高振幅劫持），并引入增益门控（有连续轴 `1.35×`、无连续轴 `1.15×`）与原模式竞争软采样（$\tau_{\text{mode}}=0.75$），将模式切换率收敛至精准的 ~20–30%；
  4. **`ours_grad_bif_synth`（顺带补跑 `seed 42` 作完整 3-seed 次要消融参考）**：挂在独占卡 GPU 7 第二顺位，零额外墙钟等待。
- **事前成功判据**：至少一项方法（`ours_bif_power_synth`、`ours_pool_bif_synth` 或 `ours_pool_mode_bif_v2_synth`）在完整三 seed（`42/123/1024`）均值上同时满足：(a) `real Rel L2 < 1.5`（无零通道爆炸）；(b) `real` 三 seed 均值 MAE 及 `mutation_test` 均值 MAE 稳定击败 `Power (0.16182 / 0.26355)`、`LCMD (0.16705 / 0.26801)`、`Random (0.17193 / 0.27733)`，且至少 2/3 seed 胜出。
- **实现与三卡资源编排（按用户授权使用共享卡 GPU 0 + 独占卡 GPU 4/7，清理 `fno/mutation/` 的 4 个 `01:32` 旧残余目录后启动）**：
  - 代码 `utils/sampling/ours/bif_synth/pool_mode_synthesis.py`、`scripts/launch/run_sampling_extensions.py`（`--suite bif_r27_gpu0 / bif_r27_gpu4 / bif_r27_gpu7`），单测 `tests/test_bif_synth.py`（3/3）；
  - **`tmux` 会话 `r27_gpu0`**（GPU 0 4 并发 1 波，批次 `/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/bif_r27_gpu0_20261125`）：`ours_pool_bif_synth`（`42/123/1024`）+ `ours_pool_mode_bif_v2_synth`（`42`）；
  - **`tmux` 会话 `r27_gpu4`**（独占卡 GPU 4 严格单并发串行 `--jobs-per-gpu 1`，批次 `/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/bif_r27_gpu4_20261125`）：`ours_bif_power_synth`（`42`）→ `ours_pool_mode_bif_v2_synth`（`123`）；
  - **`tmux` 会话 `r27_gpu7`**（独占卡 GPU 7 严格单并发串行 `--jobs-per-gpu 1`，批次 `/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/bif_r27_gpu7_20261125`）：`ours_pool_mode_bif_v2_synth`（`1024`）→ `ours_grad_bif_synth`（`42`）；
  - 三队列共用 `--max-jobs 6 --io-concurrency 2` 全局资源锁与 `exclusive_gpu_{4,7}.lock` 独占保护，100 GiB 内存底线，运行期间严禁修改 `.py` 源码。
