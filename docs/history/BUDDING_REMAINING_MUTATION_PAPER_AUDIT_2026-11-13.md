# 其他mutation的论文依据与合成扩展（2026-11-13）

本轮只审计，不调整生产mutation、分布、概率、real或ODE。范围是19个活动遗传模块、64个单独gene/mode、4个同基因双模式；CLB2的GAL-MC已计入64种。PPX/SBF_MBF/Mcm1和TEM1/CDC15-DAmP未重新启用。

## 证据不能混为一谈

64种单独模式中54种在当前126条canonical real中有基因型或功能机制映射，另10种没有。后者不能一律称为“原论文没有”：CLB2/CLN3普通MC在Chen 2004预测表出现过，Sic1磷酸化位点缺失在Chen 2000讨论过。类型/机制有出处，也不等于当前随机分布、概率、初值或任意多基因背景得到验证。

这不是针对全部原始引文的系统文献综述。“未找到直接定义”不等于整个实验生物学领域没有这种突变。

## 额外或需要限定解释的模式

| 模式 | 当前参数实现 | 依据判断 |
|---|---|---|
| CDC6-DAmP | ksf6p/ksf6pp/ksf6ppp乘α | 无当前real对应；未找到原模型论文中的基因特定规则 |
| CDC20-DAmP | ks20p/ks20pp乘α | 同上；不能与合成关闭的CDC20-ts混同 |
| CDC14-DAmP | ks14乘α | 同上；不能与CDC14-ts两分支混同 |
| NET1-DAmP | ksnet乘α | 同上；不是NET1-ts降低结合速率的机制 |
| CLN2-ts | 合成项乘β，有效变化是ksn2pp（WT的ksn2p为0） | 无当前real对应；降低合成不能直接视为已校准的温敏活性失活 |
| CLB2-ts | ksb2p/ksb2pp乘β | 同上 |
| SWI5-GAL | ksswip~LogUniform(0.02,0.10)，保留ksswipp | 无当前real对应；该特定数值规则是扩展 |
| SIC1-PM | kd2c1=kd3c1=0 | 有相关生物概念依据；当前2004模型两项同时置零尚未得到直接验证 |
| CLB2-pLOF | ksb2p/ksb2pp乘U(0.1,0.5) | #48/#51/#52/#111的CLB1 clb2Δ保留0.33WT是参照；通用pLOF命名和整个分布属于合成概括 |
| CLB2/CLN3普通MC | 整数2–8，P(N)∝1/N；分别缩放CLB2两项合成、CLN3的C0 | 不在当前126条内，但Chen 2004 Table 4有2拷贝预测；不是“论文从未提过” |

DAmP的α=`clip(Normal(0.2,0.05),0.05,0.4)`，是裁剪正态，不是超界重抽的截断正态。四基因共同的5%–40%合成残余是项目采样方案。β=`0.2*Beta(0.5,2)`，范围0–20%WT。

DAmP本身有独立实验技术依据：Breslow等2008通过扰动3′UTR降低mRNA稳定性。当前ODE没有显式mRNA，把它近似为合成速率降低可以作为建模扩展；该文不直接验证这四基因共同采用上述α、蛋白残余恰等于α，或当前初值已被校准。[原始研究](https://static1.squarespace.com/static/595ea62ae4fcb5c36ab6c982/t/59b9ee51935d81c62ec7c21b/1505357394642/Breslow_NatMethods_2008.pdf)。

SIC1-PM须特别区分：Chen 2000讨论去除Sic1磷酸化位点使蛋白稳定，但当前2004模型的`Vkpc1=kd1c1+kd2c1*(CDK加权项)/(Jd2c1+SIC1T)`。置零kd2c1仍保留kd1c1贡献，再置零kd3c1同时删除磷酸化Sic1降解通道。因此不能未经推导就声称当前规则精确代表“完全不可磷酸化”或sic1-9F。real #29的db只改kd3c1，未改kd2c1。

## 其余有对应类型的模式

- KO保留模型的基因组/功能映射：CLN2对应CLN1/2集合，CLB2对应CLB1/2，CLB5对应CLB5/6；CDC6删除的参照为cdc6Δ2–49，不代表消除真实CDC6所有功能。
- GAL与MC提高/倍增合成的机制有论文依据；逐基因范围、LogUniform、1/N拷贝概率和抽样权重仍是设计选择。
- CDH1-CA对应#35：kicdhpp=0、kscdh=0.03。
- MAD2-CSL/BUB2-CKO是检查点缺失的模型代理；MAD2采用Vi20_active=0.01，不是所有速率归零；BUB2涉及活动量、初值及事件重置。
- APC-A两入口共享ka20pp=0，是同一个机制，不能当两份独立证据。
- 当前同基因复合仅CLB2 GAL-MC（#50）以及CLB2/CLB5/SIC1/PDS1 GAL+db（#62/#72/#29/#83/#84）。GAL提高合成、db削弱降解，作用于不同通道；不因此允许任意MC+db或GAL+PM。

最近已实施的精确关闭见[前后对比](BUDDING_EXACT_ZERO_UPDATE_2026-11-12.md)。CDC14-ts按用户要求两分支都不变。数值相同不自动证明基因型相同；完整输入碰撞继续按原排除/救援规则处理。

## 使用建议与材料

建议区分论文复现核心与合成扩展。四种DAmP、两种通用ts、SWI5-GAL如保留，应明确标为扩展；SIC1-PM需另行推导；CLB2-pLOF应解释其CLB1残余表达参照。CLB2/CLN3普通MC有论文预测依据，不能写成当前Table S5已经实测。本轮没有禁用或改名这些类型。

原始材料：

- Chen 2004，本地PDF第8页Methods给出突变映射原则，第15页Table 4(vii)列出net1ts背景2×CLB2、2×CLN3等模型预测。[作者PDF](https://jigcell.cs.vt.edu/Pubs/BudYeast.pdf)。
- Chen 2000，本地PDF第13页/印刷页381，讨论Sic1磷酸化位点缺失。旧模型参数不能未经推导映射到2004的39维模型。
- Ahmadian补充Table S5与当前real_mutants.py用于126条具体数值交叉核对，与Chen原文分别标识。
- papers/Mutant_Aware_Sampling_Scheme*.md是项目采样方案，不能代替原论文证据。

复现：

```bash
/home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/audit_budding_mutation_real_coverage.py \
  --output /home/users/zsy/mydata/budding_yeast/analysis/remaining_mutation_paper_audit_20261113
```

CSV新增paper_evidence_scope，区分Table S5对应、Table 4预测、相关概念与扩展。64单模式分为53个常规real映射、1个pLOF机制映射、2个预测表对应、1个相关概念、4个DAmP扩展、3个其他未见直接规则；4个双模式均有对应。交叉表为人工整理的代表条目，不自动证明所有参数/文献一致性；未读取real轨迹或评测误差来优化采样。
