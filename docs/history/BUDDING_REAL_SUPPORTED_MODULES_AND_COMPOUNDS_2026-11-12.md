> **已实施：MA精确关闭（CDC14不变）**：CLB5/SIC1/PDS1-db、CDC20/CDC15/APC_Core-ts和APC-A对应通道改为确定性0，随机/逆CDF/赋值/容量/救援同步；MASS及其他分布保持。LHS边际已重编译，旧池版本失效，未生成新大型池。[修改前后与验证](BUDDING_EXACT_ZERO_UPDATE_2026-11-12.md)。

> **参数级复核补充（2026-11-12）**：两套profile的19模块投影38项、64模式足迹128项、real同基因复合完整输入重构12项完成。未发现额外整模块缺失；SIC1-PM的kd2c1是唯一未曾被real改变的活动突变坐标。MA三种GAL+db残余分布不含精确零原子，不能把重构边界当作实际采样覆盖。[逐模块参数与复合数值证据](BUDDING_PARAMETER_LEVEL_REAL_AUDIT_2026-11-12.md)。

# 当前有效模块、同基因复合模式及LHS范围

本次按用户要求禁用PPX、SBF_MBF、Mcm1的合成mutation，并将同基因复合模式收紧为canonical real表有依据的规则。旧审计中的22模块、71单独模式、10复合状态、85未覆盖模型参数均为调整前数据。

## 模块禁用与生物系统保留

逐项核对`real_mutants.REAL_MUTANTS`的126个名称及override：没有以PPX、SBF_MBF、Mcm1为靶点的real突变定义。这不等于它们没有参与模型动力学；PPX状态以及SBF/MBF、Mcm1辅助表达式仍是原ODE的一部分。

`catalog.DISABLED_FUNCTIONAL_MODULES`明确禁用这三者，完整定义保留在`DISABLED_BINDING_GROUPS`，供以后显式恢复。活动catalog、L1轮转、图采样节点、两套mutation/distribution入口均不再接受它们；禁用入口在消耗随机数或修改参数前报错。采样图过滤这三个节点，分子关系图/ODE/状态定义不改。

当前活动遗传模块19个，另保留MDT环境项。单独gene/mode为64种，允许的同基因双模式状态4种，共68个局部状态。剩余10种没有real类型对应的单独模式本轮没有删改。

## real中的同基因复合情况

| 同基因干预 | canonical real编号 | 实际参数意义 | 当前处理 |
|---|---|---|---|
| CLB2 GAL-MC | #50 | `ksb2p=0.12×8=0.96` | 专门的GAL-MC模式，整数拷贝2–10，保留既有1/N分布 |
| CLB2 GAL+db | #62 | `ksb2p=0.12`，`kdb2P=0`，`kdb2pp=0.034` | 允许GAL-OE与db组合 |
| CLB5 GAL+db | #72 | `ksb5p=0.012`，`kdb5pp=0` | 允许GAL-OE与db组合 |
| SIC1 GAL+db | #29 | `ksc1p=0.12`，`kd3c1=0` | 允许GAL-OE与db组合 |
| PDS1 GAL+db | #83/#84 | `kspdsp=0.10`，`kd2pdspp=kd3pdspp=0` | 允许GAL-OE与db组合；#84另有跨基因ESP1-ts |

GAL在这里是提高/替换表达启动子的合成通道；MC按拷贝数倍增指定合成参数；db破坏降解相关机制、降低降解通道，并不是让蛋白表达量按倍率变小。因此GAL与db可以分别作用于合成和降解，real表确实包含这样的组合。连续合成db残余分布仍保持前次批准范围，类型存在并不表示所有取值都与real完全相同。

另有GAL-CLN2的原生合成通道关闭情形（#4/#7/#14/#18/#31/#33/#107，`ksn2pp=0`而`ksn2p>0`）。当前GAL-CLN2单模式本来已包含这个替换规则，无须额外生成KO+GAL，也不能据此允许其他模块任意KO+GAL。

同基因复合允许列表位于`catalog.COMPOSITE_REAL_EVIDENCE`，选择器统一调用`expand_composites()`，枚举器用相同列表；冲突重抽、碰撞救援及最终候选校验也检查同一规则。MC+db、MC+PM、GAL+PM等不再因为touch集合不相交就获准。CLB2的GAL-MC仍是一个模块且不再叠加db。

该限制针对同一个基因/模块的复合模式，不是把跨基因探索限制为126个已知基因型。跨基因L1–L3探索仍按已有机制进行；real原始规则（包括四模块#42、特殊#104）独立保留。

## LHS对应变化

取消三模块的mutation派生区间后，以下5个参数失去mutation修改来源，转入U(0.5WT,1.5WT)：

| 参数 | WT | 新正值范围 |
|---|---:|---:|
| ksppx | 0.1 | 0.05–0.15 |
| kdppxpp | 2 | 1–3 |
| kppnetpp | 3 | 1.5–4.5 |
| kasbf | 0.38 | 0.19–0.57 |
| kamcm | 1 | 0.5–1.5 |

`ksn2pp/ksb5pp`仍有CLN2/CLB5等活动mutation来源，不能因删除SBF_MBF就删除它们的KO/GAL等分布。总输入仍141维＝45个mutation可变模型参数＋90个未被mutation覆盖的模型参数＋2个事件活动＋3个初值覆盖＋1个环境mdt。90个正值窗口全部为±50%。LHS不生成PPX等gene/mode标签，也不沿用其被禁用的强干预区间。

重新枚举后有19/171/969个合法L1/L2/L3模块集合。当前讨论中的“随机均匀抽合法集合，再随机抽该集合的合法完整模式”提案，层概率相应为1.6393%/14.7541%/83.6066%，不是固定配额；所有混合边际权重按新catalog重新计算。非GAL条件环境80/15/5规则不改，但GAL总体出现率变化使无条件mdt边际随之改变。

最新边际配置249个分量，目录`/home/users/zsy/mydata/budding_yeast/analysis/offline_pool_plan_20261112_real_supported/`。旧`offline_pool_plan_20261112/`保留为前次审计，不能复用其中旧的概率/CDF验证来证明本次新权重已完成经验验证。新配置做了边界、权重和枚举一致性检查，没有生成两个10万完整轨迹池，约束联合LHS生产采样仍待实现。

## 精确零值能否纳入LHS

可以单独设计`p0·δ(0)+(1−p0)·U(0.5WT,1.5WT)`。正值分支仍为±50%，精确零是额外离散原子；只从连续U(0,a)抽样不能保证产生精确零。`p0`必须单独确定，不能将所有参数统一归零或把90维下界改0。

这次对新增5个参数做了独立置零诊断：其他参数及初值为canonical WT，210min/500点、原BDF容差和超时，5/5数值成功。沿用该脚本的其余10个控制也成功。

| 置零参数 | 生物诊断标签 | 六分类 | 解释 |
|---|---|---|---|
| ksppx | Viable | P2 | 停止合成，仍保留非零初始PPX；不是完整PPX-KO |
| kdppxpp | M-phase Arrest | P5a | 删除受调控的降解贡献，不是降低PPX表达 |
| kppnetpp | Viable | P2 | 关闭PPX到NET1的一项调控贡献 |
| kasbf | Unbudded Division | P2 | 关闭SBF激活贡献，不等于有real校准的等位突变 |
| kamcm | M-phase Arrest | P5a | 关闭Mcm1激活贡献，不等于有real校准的等位突变 |

这些只证明WT背景下单参数关闭能够求解，不证明90维联合扰动的成功率或生物合理性。现有90参数包含72个可进一步审计的反应/耦合关闭候选、16个应保持正值的阈值/分母参数、2个当前不被RHS/事件使用的输入。边界测试已显示某些J阈值置零可能产生0/0。当前新增零质量仍为0（under_review），没有擅自采用5%或10%的关闭概率；已有活动mutation本来包含的精确零点继续保留。

## 验证与保护范围

9项专门回归验证禁用前置检查、复合扩展/枚举、候选拒绝、三类冲突处理及救援；另有68项范围/救援/各采样器精确模块数回归和21项图表/样本元数据回归通过。一次历史保护哈希测试失败：2026-10-22的本地manifest与当前`simulation_guards.py`不同；该文件当前SHA与本次开始时的git HEAD完全相同，本次未改。没有更改历史manifest掩盖这个差异。

`CATALOG_VERSION`已更新，使旧catalog的AL池无法静默作为新规则缓存复用。data根目录原始生成文件、canonical ODE/real/参数/初值及Fission源码均未修改；没有启动正式训练或重写旧数据。

复现脚本：`scripts/analysis/plan_budding_lhs_domain.py`、`audit_budding_mutation_real_coverage.py`、`audit_lhs_zero_parameters.py`。real核对结果含`real_coverage/real_compound_records.csv`（逐条原始override），零值轨迹及标签见`zero_audit/`。
