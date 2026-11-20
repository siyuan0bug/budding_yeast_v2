> **精确零与残余分布讨论（2026-11-12，未实施）**：找到9个gene/mode分支入口、合并8类机制，将real中的精确关闭改为连续残余；不能只关注三种db，也不能将所有ts归零。[逐项范围、非零特例及建议](BUDDING_ZERO_RESIDUAL_POLICY_DISCUSSION_2026-11-12.md)。

# 参数级核对：活动模块与real复合干预（2026-11-12）

本轮只核对，不修改采样、catalog、ODE、real定义或分布。核对源为当前两套mutation/distribution实际函数、canonical初值处理器和126条real override；没有使用real轨迹、预测误差或训练模型。此前按名称/机制整理的对应表不再单独作为参数一致性的证据。

## 核对方法与结果

对19个活动遗传模块各选一个数值见证，执行mutation后调用`prepare_initial_conditions()`，与real记录在该次干预完整改变的参数、实际初值及事件控制上比较。其他基因背景不纳入该模块投影比较；另对两套规则全部64种模式遍历端点和CDC14分支，记录实际override与生效坐标，避免把写入但等于WT的值当作变化。

两套profile共38个模块投影检查通过，128个模式足迹完成。19个活动模块均有参数级对应，未发现额外整个模块无依据的情况。APC_Core是功能集合，不是额外的独立基因；其ts由#26提供依据，其APC-A与CDC20[APC-A]共享ka20pp机制。

## 所有活动模块的参数见证

表中数值为该见证产生的变化后数值，已经与real逐坐标核对。`p:`是生效参数，`y0:`是经过初始化后的真实状态，`control:`是事件开关/重置。GAL模块的环境约束在后面的完整复合重构中单独检查。

| 模块 | 见证模式 | real编号 | 完整变化投影（合成值＝real值） |
|---|---|---|---|
| APC_Core | ts | #26 | p:ks20p=0；p:ks20pp=0；p:kscdh=0；y0:CDH1=0；y0:CDH1T=0 |
| BCK2 | KO | #10 | p:B0=0 |
| BUB2 | CKO | #113 | control:BUB2_loss=1；control:BUB2_reset=0；p:BUB2_active=0；y0:BUB2=0 |
| CDC14 | GAL-OE | #100 | p:ks14=0.8 |
| CDC15 | KO | #92 | p:Cdc15T=0；p:kkpnetpp=0；y0:CDC15=0 |
| CDC20 | KO | #74 | p:ks20p=0；p:ks20pp=0 |
| CDC6 | KO | #38 | p:ksf6p=0；p:ksf6pp=0；p:ksf6ppp=0 |
| CDH1 | KO | #34 | p:kscdh=0；y0:CDH1=0；y0:CDH1T=0 |
| CLB2 | KO | #47 | p:ksb2p=0；p:ksb2pp=0 |
| CLB5 | KO | #63 | p:ksb5p=0；p:ksb5pp=0 |
| CLN2 | KO | #3 | p:ksn2pp=0 |
| CLN3 | KO | #8 | p:C0=0 |
| ESP1 | ts | #81 | p:kasesp=0.1；p:kdiesp=0.001 |
| MAD2 | CSL | #112 | control:MAD2_loss=1；p:Vi20_active=0.01 |
| NET1 | ts | #97 | p:kasrent=10；p:kasrentp=0.05 |
| PDS1 | KO | #80 | p:ks1pdspp=0；p:ks2pdspp=0 |
| SIC1 | KO | #27 | p:ksc1p=0；p:ksc1pp=0 |
| SWI5 | KO | #43 | p:ksswip=0；p:ksswipp=0 |
| TEM1 | KO | #86 | p:Tem1T=0；p:ka15pp=0；y0:TEM1=0 |

APC_Core-ts的mutant-aware见证用alpha=0重构#26的精确关闭；这只是参数边界验证。正常MA的U(0,0.01)残余分布没有正概率的零点原子，不能宣称实际随机抽样会以正概率抽到该精确值。MASS-ts规则直接关闭这些通道。

禁用模块的特异干预参数`ksppx/kdppxpp/kppnetpp`（PPX）、`kasbf`（SBF_MBF）、`kamcm`（Mcm1）均未被canonical real改变。虽然SBF_MBF还触及ksn2pp/ksb5pp，而它们在CLN2/CLB5 real中变化过，这种部分重叠不构成SBF_MBF干预本身的依据。

## 模块有依据，不等于所有模式有依据

下列10个单独gene/mode没有对应real基因型。即使作用于同一个参数、甚至可得到相同数值，也不能据此判为同一种生物等位突变：

| 模块 | 未见的模式 | 实际改变的生效坐标（两端/分支的并集） |
|---|---|---|
| CDC14 | DAmP | p:ks14 |
| CDC20 | DAmP | p:ks20p；p:ks20pp |
| CDC6 | DAmP | p:ksf6p；p:ksf6pp；p:ksf6ppp |
| CLB2 | MC | p:ksb2p；p:ksb2pp |
| CLB2 | ts | p:ksb2p；p:ksb2pp |
| CLN2 | ts | p:ksn2pp |
| CLN3 | MC | p:C0 |
| NET1 | DAmP | p:ksnet |
| SIC1 | PM | p:kd2c1；p:kd3c1 |
| SWI5 | GAL-OE | p:ksswip |

其中SIC1[PM]修改`kd2c1`和`kd3c1`。全126条real从未修改kd2c1（始终为WT=1）；#29的db只将kd3c1置0。这是当前活动mutation中唯一“新增到real未曾改变过的生效坐标”的项，不能把PM与db等同。其他未见模式通常改变real也改变过的参数，但幅度/联合修改关系或基因型不同。

例如CLN3[MC]和GAL-CLN3都修改C0，5拷贝时C0=0.4×5=2，与#9的GAL值相同；这证明模型输入可能退化为相同，不能证明real记录包含普通MC-CLN3。实际完整输入重合仍由碰撞排除处理。

## 五种同基因复合干预的数值复核

独立组装mutation覆盖值，设对应mdt=150，再比较整个生效输入：全部基础参数、39维实际初值、checkpoint/reset和#104时序开关。以下6条real记录×2套profile共12项完整输入重构通过，绝对容差1e-12、相对容差0。它们是定点重构检查，不是随机抽样命中统计。

| 复合干预 | real编号 | 必须同时满足的real参数 |
|---|---|---|
| CLB2 GAL-MC | #50 | ksb2p=0.12×8=0.96；ksb2pp保持WT=0.04 |
| CLB2 GAL+db | #62 | ksb2p=0.12；kdb2P=0；kdb2pp=0.034 |
| CLB5 GAL+db | #72 | ksb5p=0.012；kdb5pp=0 |
| SIC1 GAL+db | #29 | ksc1p=0.12；kd3c1=0；kd2c1保持WT=1 |
| PDS1 GAL+db | #83/#84 | kspdsp=0.10；kd2pdspp=kd3pdspp=0；#84另有kasesp=0.1、kdiesp=0.001 |

机制允许列表与这5种对应一致。但MA中CLB5/SIC1/PDS1-db采连续微小残余，没有精确零点原子；定点重构使用alpha=0，不能把这个结果说成现有随机池覆盖了real的精确关闭。MASS在这三种db上直接写0。CLB2-db保持kdb2P=0并抽kdb2pp=0.017–0.068，real的0.034位于其内。GAL-MC是专门乘法规则，没有同时给普通MC再次乘到原生ksb2pp上。

所有这些复合重构的初值处理均与对应real一致。GAL-CLN2关闭原生通道的情况仍是其内置模式，不新增任意KO+GAL规则。跨基因组合仍是探索域，不受这张同基因允许表限制。

## 可复现文件

```bash
/home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/audit_budding_parameter_evidence.py \
  --output /home/users/zsy/mydata/budding_yeast/analysis/budding_parameter_evidence_20261112
```

输出：`module_parameter_witnesses.csv`（38项）、`mode_parameter_inventory.csv`（128项）、`coordinate_real_coverage.csv`（逐坐标）、`compound_reconstructions.csv`（12项）、`parameter_evidence_summary.json`。基因型命名/机制对应仍需人工语义核对，代码不能仅凭参数共享推断它们相同；这些审计输出不进入选样打分或概率拟合。
