> **2026-11-13证据层级补充**：当前64单模式/4双模式的Table S5对应、原论文预测和概念依据已分开标记；普通MC-CLB2/CLN3不能因未收录126条就称论文没有。此前残余分布现已按精确关闭更新。[最新审计](BUDDING_REMAINING_MUTATION_PAPER_AUDIT_2026-11-13.md)。

> **参数级复核补充（2026-11-12）**：两套profile的19模块投影38项、64模式足迹128项、real同基因复合完整输入重构12项完成。未发现额外整模块缺失；SIC1-PM的kd2c1是唯一未曾被real改变的活动突变坐标。MA三种GAL+db残余分布不含精确零原子，不能把重构边界当作实际采样覆盖。[逐模块参数与复合数值证据](BUDDING_PARAMETER_LEVEL_REAL_AUDIT_2026-11-12.md)。

> **当前规则更新（2026-11-12，禁用三模块后）**：PPX/SBF_MBF/Mcm1停止合成采样；19个活动遗传模块、64个单独模式、4个有real依据的同基因双模式，另保留CLB2专门GAL-MC。未覆盖模型参数由85增至90，均为±50%；混合边际已按新规则重算。下方22模块/71模式/10复合/85参数及旧概率为调整前记录。[当前配置、real复合明细与零值诊断](BUDDING_REAL_SUPPORTED_MODULES_AND_COMPOUNDS_2026-11-12.md)。

# 当前合成mutation与126条real规则的类型覆盖（2026-11-12）

核对来源为当前`data/systems/budding/catalog.py`、`quotas.allele_options()`、`mutations/mutant_aware.py`与`real_mutants.REAL_MUTANTS`。没有读取real轨迹、预测误差或训练任何模型，也没有修改catalog、mutation强度、ODE或采集规则。本表是事后类型清单，不参与拟合候选池概率或选样得分。

口径：real表的某个多基因背景包含该gene/mode，也算该类型出现过；按具体基因型及对应功能机制核对，不能只搜索模式字符串，也不能仅因参数碰巧相同就视为同一等位突变。以下“未出现”仅指当前126条canonical记录，不表示文献中不存在。

当前71个单独gene/mode中，有17个没有对应real条目；54个有基因型或功能机制对应。另有10个合法同基因复合状态，其中6个没有对应real条目。这里没有枚举全部跨基因组合；已出现过的单独模式组合起来，也可能形成real表未见的新背景。

| 模块 | 未见于当前real表的单独模式 | 数量 |
|---|---|---:|
| CLN2 | ts | 1 |
| CLN3 | 普通MC | 1 |
| CLB2 | 普通MC、ts | 2 |
| SIC1 | PM | 1 |
| CDC6 | DAmP | 1 |
| SWI5 | GAL-OE | 1 |
| CDC20 | DAmP | 1 |
| CDC14 | DAmP | 1 |
| NET1 | DAmP | 1 |
| PPX | KO、GAL-OE、ts、db | 4 |
| SBF_MBF | KO | 1 |
| Mcm1 | KO、ts | 2 |
| 合计 | | 17 |

当前TEM1和CDC15均没有DAmP，未恢复这两种已删除模式。

同基因复合状态未出现的6种：CLB2[MC+db]、CLB5[MC+db]、SIC1[GAL-OE+PM]、SIC1[MC+db]、SIC1[MC+PM]、PPX[GAL-OE+db]。四种有对应条目的复合状态为CLB2[GAL-OE+db]（#62）、CLB5[GAL-OE+db]（#72）、SIC1[GAL-OE+db]（#29）和PDS1[GAL-OE+db]（#83/#84）。

## 容易混淆的对应关系

- CLB2[GAL-MC]有#50（multi-copy GAL-CLB2）作为依据；它不等于普通MC。#57是CLB2-db加多拷贝SIC1，也不是MC-CLB2。
- SIC1[PM]同时清零kd2c1和kd3c1；#29的SIC1-db仅清零kd3c1，因此不能把PM算成已出现的db。
- CDC20[APC-A]和APC_Core[APC-A]共享#115等的APC-A机制依据，是同一通路机制的两个catalog入口，不能算两种独立有实验证据的基因等位突变。real的ka20pp=0与当前synthetic微小残余并不完全相等。
- CLB2[pLOF]对应#48/#51/#52/#111的CLB1保留、clb2删除后部分合成机制；不是real名字中明确写着pLOF。
- APC_Core[ts]对应#26的apc-ts功能机制。CDC6[KO]对应cdc6Δ 2-49的模型处理。MAD2[CSL]/BUB2[CKO]分别对应mad2Δ/bub2Δ。
- PPX、Mcm1、SBF_MBF目前都是方程上的功能干预扩展，没有当前real表的等位突变校准。

有对应类型，不代表整段合成强度、初值规则和任意多基因背景都经过real校准；本次也不把连续弱残余近似当作精确KO等价证明。未出现的类型保持不变，本轮没有删除授权。

## 复现与输出

```bash
/home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/audit_budding_mutation_real_coverage.py \
  --output /home/users/zsy/mydata/budding_yeast/analysis/offline_pool_plan_20261112/real_coverage
```

`mutation_real_coverage.csv`保存全部81个局部状态及代表real编号、名称、参数touch集合和解释；`mutations_absent_from_real.csv`保存23个未对应状态；`mutation_real_coverage.json`保存统计和口径。对应关系是脚本内显式人工核对表，不是由预测性能推导的结果；脚本检查引用编号及模式仍合法，但catalog或real定义变动后仍需复核语义。
