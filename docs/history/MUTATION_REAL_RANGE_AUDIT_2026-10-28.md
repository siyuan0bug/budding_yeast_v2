# 当前突变强度与 real 条件对照（2026-10-28）

**结论：部分 MC 强度相对 real 同基因/模式明显偏宽，是大值来源之一；组合反馈与稳定性修改也会放大输出。另有少量数值越界，不能把所有有限大值都视为正确解。以上均不能单独证明 FNO 的 badcase 是大值训练样本造成的。** 本轮只读审计及诊断复算，没有更改生产采样分布、mutation 类型、过滤、ODE、正式数据或训练。

## 比较口径

- 当前 AL 实际使用 `systems/budding/distributions/mutant_aware.py`、`distributions/quantiles.py` 和 `mutations/mutant_aware.py`，不是 MASS。基准为 `systems/budding/real_mutants.py` 的126个条件。
- 对照同一基因、同一模式，CDC14-ts 另分合成/活性两分支。比较的是**实际生效的参数**，不是笼统的 proposal alpha，也不是输出浓度。real 数值是模型对这些实验条件的参数设定，不是实验直接测定的普适生理边界。
- 全量读取 balanced 百万候选的参数和 mutation 列表、Random 最终70000条合成 train 的输入，复用上一轮的全部轨迹峰值统计。不把1250 validation、1250 synthetic test或126 real计入train频率。
- 按生效强度检查当前代码的各模式定义区间，并核对MC为整数：百万pool与70k train的越界数均为0。这里的问题是设计范围与real条件的差异，而不是发现了Sobol抽到定义区间之外。
- 这里“超过 real 上界”只是一维表达强度对照；其他参数、环境、初值可能不同。126个条件不能代表所有合理突变。

## 1. MC：统一2～30拷贝，确实没有按各基因的 real 强度缩放

基础提议为整数 `N=2,...,30`，`P(N)∝1/N`，Sobol经逆CDF取得N。下表最后一列是**70000条train中含该基因MC的样本里**，超过对应real最大拷贝数的实际比例，不能解读为全train比例。

| 基因 | 生效参数（乘WT） | real MC拷贝数 | 当前上限/real最大值 | train中超过real最大拷贝数 |
|---|---|---:|---:|---:|
| CLN2 | ksn2pp | 5 | 6倍 | 63.53% |
| CLB5 | ksb5p、ksb5pp | 4 | 7.5倍 | 63.44% |
| CDC14 | ks14 | 2 | 15倍 | 86.04% |
| CDC15 | Cdc15T | 5 | 6倍 | 61.09% |
| TEM1 | Tem1T | 5 | 6倍 | 61.55% |
| CDC6 | ksf6p、ksf6pp、ksf6ppp | 5、10 | 3倍 | 40.86% |
| BCK2 | B0 | 5、10 | 3倍 | 37.77% |
| SIC1 | ksc1p、ksc1pp | 10、15 | 2倍 | 24.47% |
| CDC20 | ks20p、ks20pp | 25 | 1.2倍 | 7.12% |
| CLN3 | C0 | 无同模式条目 | 无法比较 | — |
| CLB2 | ksb2p、ksb2pp | 无普通MC条目 | 无法比较 | — |

CLB2 的 #50 是 **multi-copy GAL-CLB2**：ksb2p=0.96，相当于常见GAL设定0.12的8倍；不是普通MC同时倍增两项原生合成参数。不能把它当成普通CLB2-MC的真实8拷贝校准点。CLN3的GAL与MC也不因控制同一个参数就变成同一个命名突变实验。

两级配额均衡的是模块集合和完整模式组合，**不表示最终强度边际严格保持基础1/N**。有限离散单元按权重无放回抽取，饱和后会包含高拷贝；全局唯一去重、real排除和初始化KMeans也会改变分布。比如CLN2-MC基础提议中大于5拷贝约57.16%，实际百万池59.81%，最终train63.53%。这不是Sobol把MC变成线性均匀，但也不能把最终pool频率说成严格1/N。

## 2. GAL：逐基因的绝对参数范围，多数比MC温和

全部为对应参数的对数均匀范围，**不是WT倍数**：CLN2等的WT基础项为0，不能用除WT解释GAL强度。

| 基因 | 参数 | 当前范围 | real同类型设定 |
|---|---|---|---|
| CLN2 | ksn2p | 0.06～0.30 | 0.12 |
| CLN3 | C0 | 1～5 | 2 |
| CLB2 | ksb2p | 0.002～1.5 | 普通GAL：0.002、0.12、0.48；#50多拷贝GAL：0.96 |
| CLB5 | ksb5p | 0.004～0.030 | 0.008、0.012 |
| SIC1 | ksc1p | 0.03～0.36 | 0.12、0.24 |
| CDC6 | ksf6p | 0.10～0.60 | 0.36 |
| CDC20 | ks20p | 0.5～12 | 4、6、10 |
| CDC14 | ks14 | 0.3～1.2 | 0.6、0.8 |
| NET1 | ksnet | 0.15～0.60 | 0.252、0.42 |
| TEM1 | Tem1T | 2～8 | 5 |
| CDC15 | Cdc15T | 3～20 | 15 |
| ESP1 | ESP1T | 1.5～6 | 3 |
| PDS1 | kspdsp | 0.02～0.25 | 0.10 |
| SWI5 | ksswip | 0.02～0.10 | 无同类型real；WT为0.005 |
| PPX | ksppx | 0.2～0.6 | 无同类型real |

GAL-CLN2还把ksn2pp设为0；GAL固定mdt150。#104是180min后的定时GAL-SIC1诱导，本表不把它混作恒定GAL表达。CLB2上限1.5是普通GAL最大0.48的3.125倍，或包含多拷贝GAL后最大0.96的1.5625倍；只和常见0.12比较会得到12.5倍，三种口径不能混用。

## 3. 稳定性/失活强度也需按生效参数比较

| 模式 | 当前MA生效值 | real对应值及解释 |
|---|---|---|
| CLB2-db | kdb2P=0；kdb2pp=0.0085～0.068 | kdb2pp=0.034；当前为其0.25～2倍，较低降解率可能增加蓄积 |
| SIC1-PM | kd2c1=kd3c1=0 | 无直接同模式real；#29的SIC1-db只把kd3c1置0，不等于PM |
| CLB5/SIC1/PDS1-db | 对应降解通道残余0～1% WT | 对应real为0；合成残余通常比real完全阻断更弱，而非更强 |
| CDC20/APC_Core-ts、APC-A、CDC15-ts | 对应通道残余0～1% WT | 对应real为0；不能把数值高于0解释为突变更严重 |
| CDC14-ts合成分支 | ks14残余0～1% WT | real为0 |
| CDC14-ts活性分支 | 四项活性共同残余5～35% WT | real为10%、16%、30%；#107是10%，不是20% |
| TEM1/ESP1-ts | 对应活性残余0.0005～0.004 WT；TEM1另清总量/初值 | real为0.002 WT |
| NET1-ts、TAB6-1 | 两项结合速率残余0.0125～0.1 WT | real为0.05 WT；TAB6-1的强度在mutation函数内抽取 |
| CLB2-pLOF | 两项合成速率残余10～50% WT | CLB1保留、clb2删除的模型合并设定为33% WT |
| DAmP | clipped Normal(0.2,0.05)，残余5～40% WT | 当前126条无直接同模式校准；TEM1/CDC15无DAmP |

KO及CA/CSL/CKO等固定修改也已经逐项匹配。MAD2的Vi20是别名，真正生效为Vi20_active，比较时已消除别名误判。PPX、Mcm1、SBF_MBF，以及SWI5-GAL等缺少直接同模式real校准，不能据此判为非法；它们是比本benchmark更宽的探索域。CDC20-APC-A作为APC_Core-APC-A的参数等效代理单独标注，未冒充新的命名real。

## 4. 全量关联统计：表达强度偏高与大值明显相关，但不充分

百万pool中312558条（31.26%）至少有一项**有real同模式参照的MC/GAL**表达强度超过其real最大值；最终train为22391/70000（31.99%）。没有参照的模式不算“超上限”。残余降解/活性高于real不混入表达强度标记。

| train中有已知MC/GAL超过real最大值 | 未触发旧幅值规则 | 触发旧幅值规则 | 触发率 |
|---|---:|---:|---:|
| 是 | 13897 | 8494 | 37.93% |
| 否 | 44571 | 3038 | 6.38% |

旧幅值规则指非MASS峰值>max(2×real逐变量最大值,2)，或MASS>10。11532条大值里8494条（73.66%）带有已知表达强度超界；其余3038条仍可能来自组合效应或无同模式real参照的探索。是关联统计，不能据此推断删掉这些样本会改善real预测。

## 5. 固定背景复算：既有强度放大，也有组合放大

同一ODE与初始化/事件规则，全部诊断用rtol=1e-7、atol=1e-9、max_step=0.25，仿真210min/500点；只在本次审计进程应用。每个对照明确修改输入，未回写数据。

| 原始训练样本背景（最终快照row） | 原峰值 | 只降低MC | 再恢复一个组合因素 |
|---|---:|---|---|
| CLB2-KO + CLN2-MC30 + MAD2-CSL（18714） | CLN2=214.18 | MC5（real #15用过的拷贝数）：35.69 | 这是另一突变背景，不能把35.69解释成real #15的预期；#15实际CLN2峰值0.065 |
| SIC1-MC30+PM + CDC14-GAL + SWI5-GAL（42231） | SIC1=946.00 | MC15：472.61 | 在MC15下将SWI5恢复WT：81.14；再恢复PM两项降解：70.99 |
| CDC6-MC23 + BUB2-CKO + SWI5-GAL（58774） | CDC6=433.52 | MC10：181.30 | 在MC10下将SWI5恢复WT：27.02 |
| CLB2-MC30+db + SWI5-KO + TEM1-KO（35537） | CLB2=624.71 | MC2（范围下限，非real校准）：39.55 | 在MC2下恢复db两项降解：3.45 |

SIC1例中仅在MC15基础上取消PM，仍有435.06；本背景的SWI5-GAL贡献比PM更显著。因此不能凭“带PM”就将该极值单因果归到PM。CDC14-GAL、背景反馈和有限时间积累仍在，恢复SWI5后也不必回到real范围。

ODE直接包含 `(ksn2p+ksn2pp*SBF)*MASS`、`ksc1pp*SWI5`、`ksf6pp*SWI5` 等项。多拷贝放大合成速率，其他突变再改变转录因子、游离/结合分配和降解；无分裂时还会持续积累。逐个参数范围不离谱，不代表任意组合的轨迹落在126个real的峰值包络内。

## 6. 独立数值问题：8条TEM1越过自身总量上界

在合法初始状态和非负lte1/BUB2下，ODE在TEM1=Tem1T边界导数非正，TEM1应≤Tem1T。扫描70000 train，用超出`1e-3*max(1,Tem1T)`为越界，发现8条（0.0114%）；CDC15/ESP1按各自总量检查无此量级越界。8条严格复算全部回到TEM1≤Tem1T。例如row2705中Tem1T=16，保存峰值124.0806，复算峰值15.9752。

原TEM1方程在TEM1=Tem1T+jatem有分母奇点；默认BDF大步越过该处与落入错误分支的现象相容。当前已经确认默认结果违反模型上界、收紧容差/步长能消除；尚未追踪每一个内部Newton/接受步，不能声称完成底层求解器根因修复。

CLN2/CLB2/SIC1/CDC6四个最大样本严格复算分别为214.1663/624.6762/945.9988/433.5356，和保存结果几乎相同。8条TEM1问题不能解释11532条大值，但说明仅检查finite和积分完整还不足以保证所有轨迹物理有效。正式数据和原有实验未被覆盖，这8条需要另行修复与补采/复算。

## 7. 对后续采样的含义

如果目标更侧重这些real条件，优先考虑**保留mutation类型，按基因重新设计MC的主概率质量与探索尾部**，而不是所有基因统一用30拷贝上限，也不是按输出2倍real一刀切。CDC14、CLN2、CLB5与CDC20的证据明显不同。模式均衡、强度分布和组合实验域是三个不同设计问题。

以上是待讨论的采样方向，不是本轮改动。126个real中的强度只能提供锚点，不能直接当普适合法范围；如果继续依据该benchmark调分布，后续收益应另有独立条件验证。当前证据也不能判定过滤大值或更换FNO一定改善badcase，仍需固定数据/预算的对照。

## 产物与复现

产物目录：`/home/users/zsy/mydata/budding_yeast/analysis/mutation_real_range_audit_2026-10-28/`。

- `real_components.csv`：逐real身份、基因、模式、分支、强度及命名/代理口径。
- `nominal_vs_real.csv`：71行（70个gene-mode加CDC14-ts第二分支），全部模式逐项比较。qmin/qmax是4096个等概率中点的值，**不是连续分布严格端点**；正文范围来自源代码。
- `empirical_pool_train.csv`：实际百万pool/70k train的强度、超real比例和大值关联；固定模式strength=1只是类别匹配占位，不是物理倍数。
- `sample_parameter_flags.csv`、`summary.json`：逐行标记和全量交叉统计；不同标记允许重叠。
- `mass_profile_differences.csv`：辅助列出另一静态MASS profile的有效控制差异，不代表当前AL采用MASS。
- `solver_recheck.json`、`solver_recheck_trajectories.npz`：5个最大样本原/严解对照；`controlled_reruns.json`及轨迹文件：8条TEM1复核和9个固定背景对照，全部成功。
- `tem1_bound_recheck.csv`：8条原始TEM1峰值、该样本总量与严格复算峰值。

生产分析入口：`scripts/analysis/dataset_diagnosis/compare_mutation_real_ranges.py --pool ... --dataset ... --moments ... --output ...`。一次性ODE探针位于`scratch/probes/recheck_extreme_solver.py`与`range_counterfactuals.py`。4项回归检查了MC/GAL命名区分、定时诱导排除、CDC14生效分支、MAD2别名及内部随机控制。没有新跑训练或修改生物规则。

## 2026-10-29 补充：同一基因MC＋GAL，以及CDC20数值来源

用户确认后续MC应按基因分别标定，认可现有GAL-CLN2/CLB5/CDC14范围；本次补充仅核对，尚未调整分布。

真实条目#50为multi-copy GAL-CLB2。Ahmadian补充材料PDF的Table S5明确注明8 copies of GAL-CLB2，与`papers/full_supplementary_info.md`第513行、当前real表和原始`data/mutant_aware.py`一致。它相对#49普通GAL-CLB2把ksb2p从0.12提高到0.96，ksb2pp仍为WT的0.04，mdt150，仍计一个CLB2模块。普通CLB2-MC8则将两项原生合成参数同时乘8，得到ksb2p=0.008、ksb2pp=0.32，二者不是同一输入。

当前`quotas.allele_options('CLB2')`返回KO、GAL-OE、MC、ts、db、pLOF、GAL-OE+db、MC+db八类，没有GAL-OE+MC；两种表达模式修改ksb2p的足迹重叠，不能直接叠加原赋值函数。当前MA（及MASS）GAL-CLB2以ksb2p的LogUniform(0.002,1.5)覆盖0.96附近，ksb2pp保持0.04；MA普通MC仍为整数2～30、基础权重1/N。也就是说目前覆盖其有效参数附近，但没有显式的“GAL基底×拷贝数”生成机制或独立复合模式元数据。基础GAL提议中ksb2p>0.96概率约6.74%，不是最终pool实测比例。若未来分开GAL强度和拷贝数，需要联合设计有效强度分布，不能简单把现有GAL上限1.5再乘普通MC上限30。

CDC20的real参数经Ahmadian补充PDF Table S5复核：#37 ks20p=10、#42=4、#77=6，和当前real表、原始mutant_aware一致；普通MC条目#122为ks20p=0.15、ks20pp=15（两项WT的25倍），没有GAL。ODE为dCDC20T/dt=ks20p+ks20pp*MCM1−kd20*CDC20T，WT参数0.006、0.6、0.3；GAL修改恒定合成项ks20p，保留调控项ks20pp。因此6不是“6倍WT”或“浓度6”。它相对WT很低的恒定项0.006是1000倍，但不能据此说总合成率/浓度是WT的1000倍。Chen2004正文参数讨论明确采用有效速率和可能因蛋白不同的任意浓度单位；不能将不同基因合成参数的绝对数值当作相同生物尺度比较。这些特定背景为何分别拟合4/6/10，现有核对只能确认模型设定，不能反推实测启动子倍数。

原始PDF已用pdftotext读取，提取文本位于`scratch/probes/mutation_ranges_2026-10-29/`。本次没有改采样器、real参数或正式数据。
