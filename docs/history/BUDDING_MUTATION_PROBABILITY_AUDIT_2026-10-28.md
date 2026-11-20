# Budding mutation采样概率审计（2026-10-28）

> **后续改动说明**：本报告2万池及候选频率实测使用当时非GAL的75/18/7。用户随后确认80/15/5，已修改AL pool提议环境；新增边界/计数验证与均衡配额计划见 [后续设计](AL_POOL_REDESIGN_AND_TRAINING_BUDGET_2026-10-28.md)。本报告保留为该次抽样的快照，不将其频率冒充改动后的重测。

结论：**恢复的PPX/Mcm1/SBF_MBF均可抽到，两个强度分布及Sobol映射接入正常；但当前不能认定整个采样系统没有问题。** BAMS存在非法模式报错，部分静态策略的冲突处理不一致，最终概率还会被real救援和去重明显改变。本轮保留用户当前mutation类型和所有生产文件，没有调整概率或重建正式数据集。

## 1. 审计范围与正常项

- 当前目录有22个基因/功能模块，加MDT环境模块，共23组、71个合法gene-mode单元（基因70、环境1）。新增3组共7个单元；此次工作区同时去掉了CLB2[CA]。TEM1/CDC15的DAmP仍未启用。
- 原始参数、变量、ODE和real表不在此次目录恢复的修改范围；本轮亦未改它们。
- 检查flat/class/role/s5四种初选权重，均非负且归一化后和为1。CLB2原始模式权重因去掉CA仅和为0.97，但实际使用时重新归一化，不是漏掉3%概率。
- 两套profile×71单元，每单元2048次直接抽样并对照2048个分位点，共142组强度检查；全数有限，最大经验CDF差0.03125。此为数值一致性核对，不把确定分位点当IID样本做显著性结论；Sobol端点截到[1e-6,1-1e-6]保留原实现。
- 每单元另做32次实际赋值，共4544次；均有参数变化，未发现空赋值或写到TOUCH声明之外的键。两套profile允许有不同赋值：例如MASS的Mcm1-ts固定kamcm=0.13，而MA按残余分布缩放；不是漏接分布。
- 12种策略请求51300条，审计器得到51298条；BAMS的2次异常被审计器记录后继续。**生产流程没有这种继续约定，真实批量生成可能直接退出。**
- 另用实际8-worker生产路径准备20000条唯一参数候选，配额1000/8000/11000准确完成。未积分ODE，未测最终有效轨迹接受率，未做模型训练。
- 现有相关回归20项通过；旧test_pruned_budding_catalog.py仍断言PPX/Mcm1/SBF_MBF必须删除，已与用户新要求冲突，未将其过时断言作为当前目录依据。

## 2. 当前AL的“初选概率”到底是什么

`budding_pool`默认mutation/distribution均为mutant_aware。模块先从23组中均匀无放回选L个；这一步包含MDT，单槽每组1/23。MDT不算基因模块，最终严格L检查会重采含MDT的候选；尚不考虑其他冲突时，纯基因集合相当于在22模块中选择。L1/L2/L3配额仍为5%/40%/55%。

**选中一个模块后，AL初选在它的合法模式中均匀取一种**，不是直接使用catalog的S5权重：

| 功能模块 | AL初选合法模式 | 每种模式初选概率 |
|---|---|---|
| APC_Core | APC-A、ts | 50% |
| BCK2 | KO、MC | 50% |
| BUB2 | CKO | 100% |
| CDC14 | ts、GAL-OE、MC、DAmP | 25% |
| CDC15 | KO、GAL-OE、MC、ts | 25% |
| CDC20 | KO、GAL-OE、MC、ts、APC-A、DAmP | 16.7% |
| CDC6 | KO、GAL-OE、MC、DAmP | 25% |
| CDH1 | KO、CA | 50% |
| CLB2 | KO、GAL-OE、MC、ts、db、pLOF | 16.7% |
| CLB5 | KO、GAL-OE、MC、db | 25% |
| CLN2 | KO、GAL-OE、MC、ts | 25% |
| CLN3 | KO、GAL-OE、MC | 33.3% |
| ESP1 | ts、GAL-OE | 50% |
| MAD2 | CSL | 100% |
| Mcm1 | ts、KO | 50% |
| NET1 | TAB6-1、GAL-OE、ts、DAmP | 25% |
| PDS1 | KO、GAL-OE、db | 33.3% |
| PPX | KO、GAL-OE、ts、db | 25% |
| SBF_MBF | KO | 100% |
| SIC1 | KO、GAL-OE、MC、db、PM | 20% |
| SWI5 | KO、GAL-OE | 50% |
| TEM1 | KO、GAL-OE、MC、ts | 25% |

这是第一次选模式的条件概率，不是“最终整个池有多少比例属于这一模式”。之后会发生复合扩展、冲突重采、real救援以及全池去重，不能将这些百分比当成最终出现率。

## 3. 恢复三组的明确概率

class是部分结构采样器的默认模式权重，flat列代表AL/纯随机的初选。复合列只包含25%复合扩展的影响，尚未计入冲突、救援和去重；同一模块可有两个模式，故包含率之和可以超过100%。

| 模块 | 模式 | 选法 | 初选概率 | 复合扩展后包含该模式的概率 |
|---|---|---|---|---|
| Mcm1 | ts | flat | 50.000% | 50.000% |
| Mcm1 | KO | flat | 50.000% | 50.000% |
| Mcm1 | ts | class | 39.130% | 39.130% |
| Mcm1 | KO | class | 60.870% | 60.870% |
| PPX | KO | flat | 25.000% | 25.000% |
| PPX | GAL-OE | flat | 25.000% | 31.250% |
| PPX | ts | flat | 25.000% | 25.000% |
| PPX | db | flat | 25.000% | 31.250% |
| PPX | KO | class | 29.787% | 29.787% |
| PPX | GAL-OE | class | 29.787% | 35.106% |
| PPX | ts | class | 19.149% | 19.149% |
| PPX | db | class | 21.277% | 28.723% |
| SBF_MBF | KO | flat | 100.000% | 100.000% |
| SBF_MBF | KO | class | 100.000% | 100.000% |

特别是PPX同时有GAL-OE与db，自动进入COMPOSITE_ELIGIBLE：

- AL初选KO、GAL、ts、db各25%。
- 选GAL后25%加db，选db后25%加GAL。
- 最终这一步的互斥组合为KO25%、ts25%、单GAL18.75%、单db18.75%、GAL+db12.5%。
- 因而“包含GAL”和“包含db”各31.25%，不是各25%。这来自已有通用复合规则，不是新分布数值错误。

Mcm1在AL初选KO/ts各50%；class下为KO60.87%、ts39.13%。SBF_MBF只有KO，选中后概率100%。class权重是**逐模式按所属类别赋权后归一化**；同类别有多个模式时会重复获得权重，不能将MODE_CLASS_QUOTA误读为最终精确类别配额。

不同静态策略也不应混为一套概率：mass_random使用P_bio_G和catalog模式权重；mutant_aware/图采样部分路径用class/role/s5；L1轮转还会按gene-mode单元轮转；motif/BAMS另有结构与方向偏置；WT-Blind还有专门强度/ledger规则。本轮未要求把这些策略强行统一。

## 4. 正常但很大的概率变化：碰撞、冲突与唯一去重

### 4.1 全池唯一去重使固定模式在L1出现率大幅下降

真实生成的2万池中，L1结果为：

| 模块 | 模式 | 最终L1数量 | L1总数 |
|---|---|---|---|
| Mcm1 | ts | 31 | 1000 |
| Mcm1 | KO | 3 | 1000 |
| PPX | KO | 3 | 1000 |
| PPX | GAL-OE | 23 | 1000 |
| PPX | ts | 22 | 1000 |
| PPX | db | 18 | 1000 |
| SBF_MBF | KO | 3 | 1000 |

SBF-KO、Mcm1-KO、PPX-KO在L1只有固定参数+三个mdt环境，且初值固定，**每种最多3个不同输入**。实测各保留3条，其余重复被丢弃并重新抽候选；所以SBF在L1只有0.3%。如果生成100万池、L1为5万，单SBF-KO最多3/50000=0.006%的L1样本。这是唯一输入规则造成的上限，不是随机数故障。

多模块时叠加其他连续突变，可有更多不同输入，不能把L1上限套到L2/L3。这个规则是否满足训练需要属于设计选择；若保持不重复的输入池，就不能同时要求固定KO单模块无限按等比例复制。

实际2万池共记录1353次重复输入、377次real碰撞、230次成功救援、147次救援耗尽后重采。救援会优先将可软化KO转换成合法DAmP/ts/pLOF，因此最终KO比例不等于初选比例；70%是尝试分支概率，不是最终70%的碰撞一定软化成功。

### 4.2 恢复SBF之后部分组合概率为0

SBF_MBF-KO写kasbf、ksn2pp、ksb5pp，而CLN2各模式都写ksn2pp。按当前common的同参数冲突禁令，AL/Random不能保留SBF_MBF与CLN2同模块集合，丢一个后又不满足精确L，于是整条重采。SBF与部分CLB5模式也冲突。它们的组合并不服从独立均匀选择；不能用1/22×1/22计算最终组合概率。

这是当前冲突政策的直接结果。本轮不放宽政策，也不把所有参数重叠都判定为生物学非法；如未来允许重叠组合，应先显式定义联合赋值语义。

## 5. 需要修正的实现问题（不涉及新增/删除mutation类型）

### 5.1 BAMS可能构造APC_Core[KO]并终止生成

`samplers/bams.py:150–152`：当某条级联需要表达型模式而目标没有GAL-OE/MC时，直接fallback到字符串KO；APC_Core只有APC-A/ts，没有KO。新require_mutation严格检查后报错。

主审计3000次BAMS请求中2次异常；另一次独立进程复查记录4次APC_Core[KO]。两次采样受历史集合遍历等随机流细节影响，不把此次数解释成严格频率估计。直接原生L2采样在seed46、523、700也触发同一ValueError（见reproducer JSON）。

应修级联fallback，只从该模块合法模式中选或重新选候选，不应为消除错误给APC_Core添加KO。本轮仅报告，未改生产代码。

### 5.2 部分策略绕过共同的参数冲突规则

主审计接受后的footprint重叠样本：

- MASS random：71/3000。
- MASS random_dist：56/3000。
- motif：263/3000。
- BAMS：161/2998。
- AL pool、普通Random、mutant_aware、Gibbs、kDPP等本次样本未发现接受后重叠。

重叠既可能写相同值、表现为冗余，也可能写不同值、导致顺序覆盖；不能把所有重叠直接称为错误ODE。确定的问题是**策略间执行了不同的联合赋值政策**，而公共valid_candidate只检查合法类型和模块数，不检查重叠。

例如BAMS接受 `SBF_MBF[KO]+CLN2[GAL-OE]+CLB5[MC]`，最终ksb5pp被CLB5-MC写成0.01，覆盖了SBF-KO先写的0；同类组合在AL pool会被冲突规则重采。新功能模块恢复后，这种差异更容易发生。完整override和重叠键见strategy_reports.json。

### 5.3 db的方向分类尚未同步完成

本次改动已将若干LOF候选列表中的db删去，但BAMS顶层LOF_MODES仍包含db，get_mode_polarity('db')仍为-1；motif也把db放在dec_avail。db降低降解通常增加稳定性，不能普遍等同于丧失蛋白功能。这里会影响方向引导和模式分配；具体网络效应需按模块判断，不可把所有db硬编码成同一网络效应。

这些问题优先影响相应静态策略；当前Random/AL pool初选没有调用BAMS的方向分类，不能把BAMS异常说成AL也在报同一个错误。

## 6. 现有百万池仍是旧目录，不能代表本次恢复

实际检查 `budding_pool_1000000_seed42_al.npz`：PPX/Mcm1/SBF_MBF各0条；同时包含 **22176条CLB2[CA]**，而当前catalog已移除CA。

因此改Python目录不会自动改变旧池的概率。按当前AL加载器合法性验证，该旧池的CLB2[CA]会被拒绝，不能直接当成完整新目录的候选池继续开展新实验。新的候选池与初始数据需另行生成，保留旧池用于旧实验复现；本轮未删除或重建它们。

CATALOG_VERSION虽然在catalog中定义，目前NPZ/初始缓存契约未完整校验这个目录版本，只有sampling_policy等标记不足以证明分布相同。建议后续把目录版本与源分布身份纳入产物契约；本轮不更改接口。

## 7. 各策略主审计汇总

proposal指策略完成内部模式扩展/冲突处理后返回的提议；accepted指通过公共精确L与real救援的候选，**不是完成ODE积分后的有效轨迹**。不含仿真失败、BAMS二分细化、WT-Blind ledger正式接受造成的进一步概率变化。2万unique pool另外执行了生产去重路径。

| recipe | 请求数 | 接受数 | 模块数重采 | real碰撞 | 救援成功 | 救援耗尽重采 | 接受后footprint重叠 | 异常 |
|---|---|---|---|---|---|---|---|---|
| budding_gibbs | 3000 | 3000 | 159 | 77 | 67 | 10 | 0 | 0 |
| budding_mutant_aware | 3000 | 3000 | 356 | 73 | 56 | 17 | 0 | 0 |
| budding_gibbs_sobol | 3000 | 3000 | 171 | 63 | 52 | 11 | 0 | 0 |
| budding_mass_random | 3000 | 3000 | 290 | 166 | 128 | 38 | 71 | 0 |
| budding_kdpp | 3000 | 3000 | 209 | 76 | 63 | 13 | 0 | 0 |
| budding_mass_random_dist | 3000 | 3000 | 328 | 173 | 123 | 50 | 56 | 0 |
| budding_motif | 3000 | 3000 | 5 | 82 | 66 | 16 | 263 | 0 |
| budding_bams | 3000 | 2998 | 83 | 91 | 79 | 12 | 161 | 2 |
| budding_kdpp_curvature | 3000 | 3000 | 256 | 85 | 70 | 15 | 0 | 0 |
| budding_wt_blind | 300 | 300 | 42 | 5 | 4 | 1 | 0 | 0 |
| budding_pool | 12000 | 12000 | 1544 | 204 | 122 | 82 | 0 | 0 |
| budding_random | 12000 | 12000 | 1503 | 334 | 192 | 142 | 0 | 0 |

所有恢复模式在12条路径的accepted候选中均有出现。WT-Blind仅抽300条作接入核对，其余通常3000，Random与AL pool各12000；不能把这些不同样本量行当成等精度估计。

## 8. 文件与复现

输出根目录：`/home/users/zsy/mydata/budding_yeast/analysis/mutation_probability_audit_2026-10-28`。

- theoretical_mode_probabilities.csv：71单元×4种模式权重及通用复合包含概率。
- proposal_and_accepted_modes.csv：12策略、L1–L3、提议/接受分阶段计数与每候选包含率。
- strength_distributions.csv / mutation_assignments.csv：两profile的强度分布及赋值检查。
- unique_pool_modes.csv / unique_pool_report.json：真实2万唯一池模式计数与重采统计。
- strategy_reports.json / bams_error_recheck.json / bams_invalid_mode_reproducer.json：异常、参数重叠实例及复查。
- regression.log：20通过、20未选择；未运行与新目录冲突的旧prune断言。
- provenance.json：生产文件审计前后SHA256一致及参数。当前正式数据、权重、原始data根目录代码未改。

命令（neuralop环境，在budding_yeast_v2目录）：

```bash
python scripts/analysis/dataset_diagnosis/audit_budding_mutation_probabilities.py   --workers 8 --random-samples 12000 --samples 3000   --strength-samples 2048 --pool-size 20000
```

是否在某一新突变体上改善预测，需要之后重新生成数据并做训练对照；本审计不提供精度收益结论。
