# 2026-11-16：正式重训前的全流程审计

审计对象为`seed42_b5bff21cb8de`准备协议及上午`seed42_02eebd57592f`的16项实验。没有启动正式训练、改写旧结果或修改源候选池。两池共20万轨迹已逐块扫描，只保留统计；11条疑点另外做严格ODE复算。报告不是模型精度或收敛保证。

## 1. 优先处理的发现

|事项|证据与影响|处理|
|仿真有限但不满足数值质量要求|mutation有3条、LHS有4条TEM1明显超过总量；LHS另有4条最低值小于−1e-4。严格复算11/11成功，TEM1恢复总量内、明显负尾恢复近零；Pattern组均未变|建议只定点修正这些轨迹及相关派生字段，保持输入、采样分布和切分不变；本次未修改池。无需重采样20万条|
|无穷预测可能被解码保护掩盖|`expm1(clamp(+Inf,max=15))`会返回有限数；独立Predictor原先未检查解码前输出|已加解码前及解码/类型转换后的有限值检查；异常直接报错，不当正常轨迹评测|
|验证异常缺少即时终止|归一化验证loss若NaN/Inf可能拖到训练末才发现无有效best|已在记录checkpoint指标前检查形状、预测/目标及验证loss是否有限|
|学习率日志错位|Lightning调度后在epoch结束读取optimizer LR，得到的是下一epoch值|history/log的`lr`改为epoch开始时实际使用值；`lr_next`另存下轮值。优化器、调度公式和训练预算不变|

### 数值质量问题的具体范围

|全池检查|mutation 100000|LHS 100000|
|---|---:|---:|
|NaN/Inf样本|0|0|
|初值与轨迹首点不一致（按float32容差）|0|0|
|非正MASS|0|0|
|目标超过解码上限expm1(15)|0|0|
|出现任意负值|18061|6551|
|最低值小于−1e-4|0|4|
|TEM1超总量>1e-3（审计标记）|19|91|
|TEM1超总量>max(0.01,1%总量)|3|4|
|全池最低值|−1.6408e-6|−0.0312475|
|全池最大值|5570.7573|19758.2617|

这些阈值只用于发现疑点，不是新增过滤策略。保留数值零附近的负数；明显负值采用同输入严格复算，不按“只要负数就删”。大值不因超过real包络而剔除。

最严重的是mutation池0-based row 38053：Tem1T=4.19007，原TEM1峰值125.74828；rtol/atol收紧到1e-8/1e-10后峰值4.16580，整个39维轨迹相对差异0.6810。进一步1e-10/1e-12复算峰值4.165797913，两种严格解相对差异1.39e-5，支持原解为数值伪影。其它10条原/严格整轨迹差异不超过约0.318%，但特定通道仍有可见误差。

11条全都属于candidate，不在初始2000、固定validation或test中；其中7条曾被上午部分方法选入训练。最严重的row 38053未被任何上午实验选入。不能据这11条推断它们导致了上午整体精度问题，也不应因此把全部历史实验判为无效。未来应避免把明确数值伪影作为监督真值。若修池，应保留旧池，新建带修正来源的视图/少量修正块并核对输入和切分一致性；不得原地偷偷改已有哈希。

## 2. 核心链路核对

|环节|核对结果|
|---|---|
|冻结池/输入|两池各100000，协议缓存哈希一致；有效输入无重复、无real精确碰撞、无跨池重复|
|划分|每池90000 candidate / 5000 validation / 5000 test；初始2000属于candidate；同一有效输入不会跨holdout角色|
|选样信息|当前8项基线只访问候选输入、已查询训练轨迹和模型预测；oracle拒绝holdout/越域ID；主流程不把real/test传入selector|
|初始公平性|同池所有方法共用初始行ID、split、seed与逐轮派生初始化；Random使用独立selection随机流|
|归一化|每轮按累计已查询训练集重拟合P逐维Z-score、Y log1p+逐变量Z-score；轮内冻结；固定validation重新编码|
|实际初轮参数统计|上午checkpoint中的P均值/标准差与对应初始2000输入逐位一致|
|输入支撑|首轮统计下，同池candidate最大绝对Z为mutation 14.8246、LHS 21.9229；同池无>100坐标；real最大为12.1561、18.1413；无“全池变化但初始标准差近零”的参数|
|训练与验证loss|当前FNO/rel_l2使用同一逐样本、逐变量时间Relative；smooth eps0.01、无ratio截断、梯度范数裁剪1；训练bf16、验证float32|
|新best规则|同池`val_loss_synth`最低；主/辅助模型一致；加载best后才做轮末评测和下一次选择|
|物理评测|使用对应checkpoint的统计逆变换，真实目标保持原始数值；MAE/MSE/Relative/Corr分开报告，归一化loss不冒充物理误差|
|ensemble|TopK/Power使用两个独立成员，当前采集分数保留MSE预测分歧；统一训练Relative不等于自动改写采集公式；最终比较仍用主模型，额外训练成本单列|
|选择内核|本地独立实现，无生产作者仓库import；CoreSet/LCMD/BAIT的LL/Pred特征版本需按实际命名，不宣称与作者整个PDE实验完全一致|
|恢复|模型、optimizer、scheduler、loader/Python/NumPy/Torch随机状态随恢复保存；monitor及源代码变更会拒绝混用旧状态；中断恢复回归与连续训练一致|
|存储/资源|每轮只存有序ID引用，无重复轨迹；全局6任务、每卡最多2、重加载2、96GiB自有PSS/128GiB可用底线不变，GPU4--7仍独占|

上午80份history中的训练与归一化验证loss全部有限。上午主流程逐epoch的real/同池test监控原本已检查raw预测有限值；本次新增的Predictor保护不意味着历史主指标已被Inf污染。它堵住独立评测等入口的漏洞，不能解释正常有限输出上的精度变化。

## 3. 需要在论文和实验解释中固定的边界

1. **126株real已参与研究开发反馈。** 程序上没有进入梯度、best或采样评分，但我们已经反复观察real badcase并据此讨论设计/协议。不能把这126株描述为完全未接触的最终blind test。应将其定位为公开固定基准，从现在起仅按synthetic validation决定超参/改进；独立未接触数据与多seed用于更强的泛化结论。事后重新拆分已看过的real不能消除开发反馈。
2. **单seed只够筛查，不能稳定排名。** 同一个seed统一所有方法是公平起点，但不能通过不断重跑42选择最有利配置来代替多seed。可冻结当前规则后再用42/2026/123作共同种子，池生成seed42保持固定。
3. **LHS与mutation不只差在组合相关性。** LHS对背景参数也扰动±50%，mutation保留固定背景。这是用户/导师确定的生成范式比较；可以回答两种范式在当前任务上谁更好，不能只凭这组实验把性能差异归因于模块相关性。cross_pool_test属于分布外诊断，不与同池test/real合成一个排名。
4. **Relative权重偏好保持透明。** 物理近零通道仍可主导Relative，MAE/MSE又可能被ORI等大通道主导；各轮归一化变化后，归一化loss不能直接跨轮/跨方法排名。沿用当前已确认公式，按预先规定的物理多指标/逐株结果汇报，不再根据real临时换epsilon或best准则。
5. **预算是最终10000训练输入，不是只花10000次ODE。** 两池各预计算100000完整轨迹，另有共同validation成本；本项目是固定完整池上的选择比较。报告选入训练数、实际可见真值、预计算成本及额外ensemble训练成本，勿把预计算隐藏成本抹掉。
6. **全局Corr与逐株Corr平均不同。** 当前JSON的Correlation先将每个变量跨mutant×time拼接计算Pearson，再平均变量；CSV逐株Corr是在每株内平均变量。这两种量不应混用，尤其恒定或近零轨迹的Pearson并不稳定。已有valid_channel_correlation/undefined数量用于诊断；本次不更换Corr定义或用它选best。

## 4. 减少再次重跑和额外耗时

- 已按用户确认保留每轮末epoch的weights-only状态，放在`_state`；当前FNO每份约27.86MiB，五轮主模型额外约139.3MiB/实验，远小于轨迹。它能避免再次为best/last固定数据诊断重训；改变早期checkpoint仍可能改变选样，不能拿last权重冒充另一条完整AL路径。
- **用户随后确认，已实施末epoch权重保留**：每轮额外保存`_state/round_XX/primary/last_weights.ckpt`，辅助成员对应`ensemble_member_N/last_weights.ckpt`。纯权重附带预处理契约，无optimizer/scheduler/callback恢复状态；训练完成仍删除临时resume。best继续是`rounds/round_XX/model.ckpt`，评测与下一轮选样仍用best。训练元数据记录best_epoch、末epoch路径/哈希，缺失或被改动则拒绝把该轮当完整缓存复用。未给既有历史实验伪造last。末权重/中断恢复等26项定向CPU回归通过。
- 合成test5000每epoch前向仅用于日志，不影响best；600epoch合计300万条次推理。若以后需要省时，可降低test日志频率并保留逐epochvalidation、real和轮末完整test。当前遵从用户要求，未降低测试曲线密度。
- 当前warmup第一epoch LR=0，Adam状态仍更新，参数不更新；是既有调度定义。已纠正日志，不借此再改训练协议。100/100/100/100/200epoch、batch128、峰值LR5e-4、minLR1e-5、warmup5/5/5/5/10保持。
- 新任务应使用准备协议与独立结果目录。修改源代码后不要往旧运行目录硬续；先完成定点数值修正与核查，再冻结代码/配置，统一比较所有方法，避免再混换loss、best和数据后归因于某个因素。
- 批次还应保存源码快照和依赖版本。现有source/config指纹会拒绝代码变化后的旧目录混用，但不代表跨PyTorch/CUDA版本、不同推理后端都能逐位复现。本次另保存execution_environment.json供协议冻结时核对。

## 5. 验证与产物

45项CPU回归通过：异常验证/Inf解码防护、实际LR日志、归一化best与重拟合、恢复、监控、独立选择内核、内存保护、8种基线五轮和物理指标/旧调度兼容。没有GPU正式训练。正常有限数值的优化器和loss未改变。

持久工具：`scripts/analysis/audit_sampling_experiment_flow.py`、`audit_frozen_pool_numerics.py`、`recheck_pool_numerical_flags.py`。前者只用缓存输入和checkpoint统计，后者逐块扫描两池，再复算11条固定输入。

外部产物目录：`/home/users/zsy/mydata/budding_yeast/analysis/experiment_flow_audit_20261116/`。

- `input_and_split_audit.json`：哈希、划分、输入碰撞、初轮统计/输入Z范围、80轮history。
- `full_pool_numerical_qc.json`：两池20万轨迹逐块数值审计。
- `strict_solver_rechecks.json/.png`：11条复算、原/严格曲线和训练曝光记录。
- `worst_case_tolerance_convergence.json`：最严重例子的严格/超严格容差一致性。

结论：当前训练/选择主链具备完整可执行的已验证接口，不建议再随意改loss或epochs。先定点处理已证实的少量数值伪影，然后冻结协议；仍需如实说明开发反馈、多seed和两种生成域的比较边界。此审计不能保证所有未检查的未来配置或生物模型扩展都正确。

## 6. 总量限制与数值越界的复核

重新比对池生成run_config的source_hashes：simulation.py、offline_pool.py、simulation_guards.py、ode.py及分布ranges.py均与生成时相同。TEM1 GAL范围为Tem1T~LogU(2,8)，sample row38053实际为APC_Core[ts]+CDC15[KO]+TEM1[GAL-OE]，候选ID41199，Tem1T=4.1900696104、初始TEM1=0.9038969。直接核对源param_names与variable_names，确认参数列21为Tem1T、变量下标28为TEM1，原磁盘轨迹峰125.7482758。

采样参数范围和部分采样路径的初始活性≤总量约束确实存在（samplers/random.py:49、derived.py:258），但它们不等同于整条仿真轨迹的约束检查。完整池调用offline_pool.simulate→BuddingSystem.simulate_mutant→simulation.simulate_mutant，BDF rtol1e-4/atol1e-6；出口只检查求解成功、时间完整、形状和有限值，没有逐时间TEM1≤Tem1T检查。方程含Tem1T−TEM1项不保证数值积分自动保持边界。本例初值/参数均合规，却产生了错误数值轨迹，收紧容差后恢复，证据保存在row38053_disk_evidence.json和strict_solver_rechecks.json。未修改采样分布或源池。


## 2026-11-16：数值修复已完成，旧池保留

已对mutation 3条、LHS 8条确认数值伪影进行同输入严格复算并以11行NPZ修补视图发布；两池仍各100000条，输入/ID/顺序/切分未变。新默认池目录增加`_numerical_qc`，协议`seed42_a866aef5d988`，共享四个输入/划分缓存和原126株real参考，原池/旧结果保留。以后生产求解默认BDF 1e-8/1e-10，检查TEM1已知总量约束并按需重试1e-10/1e-12，失败记录且补采至足量；负值继续保留、未恢复real×2幅值过滤。17项CPU回归通过，没有启动正式重训。旧16项中10项用过7条错误监督数据，首次受影响轮及重训边界详见[数值修复记录](NUMERICAL_POOL_REPAIR_2026-11-16.md)。本条更新早前“源池暂未修正”的阶段记录。
