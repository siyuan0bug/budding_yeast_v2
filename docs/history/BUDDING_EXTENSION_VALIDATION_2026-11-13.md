# SIC1-PM、DAmP等扩展mutation的机制与仿真验证

## 结论

现有SIC1-PM和四种DAmP可以保留为有机制解释的合成干预；这次检查没有发现必须因此删除它们的数值或反应方向错误。它们仍不是原论文逐基因校准过的等位突变。PM适合称为“Sic1稳定化并削弱CDK相关磷酸化”，不能精确等同于“完全不可磷酸化”。DAmP适合表示合成降低后的动态过程，当前初值并不代表已预培养至低表达稳态。

本轮没有修改生产源码、概率、分布、初值、ODE、real或正式数据；data目录全部Python源文件的运行前后SHA256一致。没有训练模型，不能从本次仿真推断real预测精度会改善。没有用real轨迹、坏例预测误差或真实测试身份调整分布。

## 实验范围

主实验329次求解全部成功：DAmP 228次、PM 63次、其他扩展38次。12个CPU进程，生产事件/重置/初始化代码，210分钟500点；另45次长时对照为840分钟2000点。生产容差rtol=1e-4/atol=1e-6和严格容差1e-8/1e-10形成142个配对。严格求解诊断超时30秒，不修改生产5秒超时。

四种DAmP各在三个固定背景测试：

| 靶基因 | 背景 |
|---|---|
| CDC6 | WT、CLN3/BCK2双KO、CDH1-KO |
| CDC20 | WT、CDH1-KO、PDS1-KO |
| CDC14 | WT、CDH1-KO、NET1-TAB6 |
| NET1 | WT、TEM1-KO、CDC15-KO |

强度α=0/0.05/0.1/0.2/0.3/0.4/1。0和1是关闭/原表达边界对照，不属于DAmP抽样值。PM在WT、CLN3-MC2、CLB5-KO背景比较六种参数定义。另在WT背景扫描CLN2/CLB2-ts、CLB2-pLOF及SWI5-GAL；后者所有点固定mdt=150，不能直接与mdt=90的WT对照解释介质外的单因素效应。这部分仅是小范围一致性检查。

## SIC1-PM：作用方向合理，但不等于db或完全去磷酸化位点

记S为SIC1/C2/C5/SIC1P/C2P/C5P六种形式总和。由原方程逐项相加得到：

`dS/dt = ksc1p + ksc1pp*SWI5 - kd3c1*(SIC1P+C2P+C5P)`。

600个状态/参数组合的恒等式最大绝对残差1.94e-16。当前PM关闭kd3c1，持续合成仍在，因此总Sic1积累；同时关闭kd2c1，削弱CDK驱动的磷酸化。但kd1c1=0.01仍在，不能说所有磷酸化消失。

WT背景，210分钟，严格容差：

| 定义 | 参数变化 | Sic1总量终值 | 终点磷酸化形式占比 | ORI最大值 | 分裂事件 |
|---|---|---:|---:|---:|---:|
| WT | 无 | 0.9980 | 3.25% | 24.349 | 2 |
| 仅db | kd3c1=0 | 5.8426 | 42.66% | 0.092 | 0 |
| 当前PM | kd2c1=kd3c1=0 | 5.8423 | 7.51% | 0.091 | 0 |
| 仅关闭CDK相关磷酸化 | kd2c1=0 | 2.1369 | 0.91% | 1.675 | 0 |
| 关闭全部磷酸化 | kd1c1=kd2c1=0，保留kd3c1 | 5.8291 | 约0 | 0.095 | 0 |

当前PM、仅db、全部磷酸化关闭在三个背景的210分钟窗口均表现为复制起始未触发、无分裂，原生物分类为G1 Arrest。表型接近不代表分子机制相同：WT背景当前PM与db完整39变量RelL2差19.66%，与全部磷酸化关闭差5.05%；差异以各自对照轨迹为分母。延长840分钟，后者差异仍约6.45%，并非210分钟末巧合。只关闭kd2c1的对照允许ORI越过1，说明残留的基础磷酸化/降解确实可以改变动力学。

替代PM定义只用于机制诊断，未加入采样模式，不是新增同基因复合规则。清除初始磷酸化形式时，将对应量转回未磷酸化的游离/结合形式，保持Sic1和结合Cyclin总量；WT背景当前PM全轨迹RelL2变化仅约0.026%。这也不能代替实际等位突变数据。

Chen 2000正文（本地PDF第13页）支持Sic1磷酸化位点缺失会稳定蛋白并阻碍S期；本次仿真支持当前规则作为这种现象的近似，而不证明其精确对应sic1-9F。曲线：[PM重点对照](../../../../../../../../data/zsy/budding_yeast/analysis/extension_validation_20261113/PM_key_comparison.png)、[六种定义完整曲线](../../../../../../../../data/zsy/budding_yeast/analysis/extension_validation_20261113/SIC1_PM__WT.png)。

## DAmP：表达降低的方向与量级可解释

四种DAmP仅将对应合成速率乘α，不将蛋白催化活性乘α。固定同一个状态时，RHS关于该表达缩放满足严格仿射关系，最大残差2.22e-16；实际初值均与原初值逐元素一致。CDC14T、NET1T还分别满足独立线性方程：

`dCDC14T/dt=α*ks14_WT-kd14*CDC14T`；`dNET1T/dt=α*ksnet_WT-kdnet*NET1T`。

α=0.2时210分钟末，两者总量分别为0.4000、0.5641，对照为2.0000、2.8000，与预期量级一致。整个扫描的严格轨迹与解析总量解最大差6.48e-5，包含输出网格的线性插值误差。

WT背景的几个代表结果：

| 模式 | α | 210分钟内分裂数 | CLB2峰值 | 原生物分类 |
|---|---:|---:|---:|---|
| WT表达对照 | 1 | 2 | 1.427 | Viable |
| CDC6-DAmP | 0.2 | 2 | 1.429 | Viable |
| CDC20-DAmP | 0.2 | 0 | 4.847 | M-phase Arrest |
| CDC14-DAmP | 0.2 | 0 | 1.818 | M-phase Arrest |
| NET1-DAmP | 0.2 | 0 | 0.317 | M-phase Arrest |
| CDC20-DAmP | 0.05 | 0 | 12.177 | M-phase Arrest |

不同基因的表达削弱不应产生相同表型；同一基因蛋白的终点量也不一定是WT终点量的α倍，因为WT和突变体可能位于不同周期相位，调控反馈也不同。低CDC20下CLB2较高符合减弱降解通路的模型解释，不能因“大值”就认定mutation错了。CDC6这里属于既有模型中CDC6调控/抑制功能的干预，不代表真实CDC6所有复制功能都被完整建模。

实际DAmP抽样100000次：平均α=0.20007，标准差0.04998，5%–95%分位为0.11783–0.28253；2.28%低于0.1。Breslow等报告典型mRNA量降低2–10倍，即残余约0.1–0.5，与当前主要质量区域相容；但这是mRNA尺度，不能自动等同于四个基因的蛋白尺度或验证裁剪正态的形状。[原始DAmP论文第2页](https://static1.squarespace.com/static/595ea62ae4fcb5c36ab6c982/t/59b9ee51935d81c62ec7c21b/1505357394642/Breslow_NatMethods_2008.pdf)。没有据此裁掉低于0.1的尾部。

曲线：[四种DAmP强度扫描](../../../../../../../../data/zsy/budding_yeast/analysis/extension_validation_20261113/DAmP_overview.png)。

## 初值是一个需要明确任务定义的问题

现有规则在t=0沿用原蛋白初值、降低之后的合成速率，可以解释为表达干预后转变的动态过程，不等于低表达株已经预培养到新状态。为检查敏感性，另构造α=0.2的初始耗减对照：同时调整靶蛋白游离/结合/磷酸化形式，释放相关结合伙伴，维持隐含物种和总量关系。它不是实测初值，也不是经过预培养得到的稳态，因此不能宣称更正确。

WT背景210分钟，耗减初值相对当前初值的39变量RelL2：CDC6 75.27%、CDC20 2.29%、CDC14 4.16%、NET1 37.00%。CDC6依然分裂两次但出现Unbudded Division标签；NET1的原生物分类也变化。部分差异包含周期相位变化，不能把75%解释为所有变量幅值都错误75%。这说明若以后任务转为“持续低表达株的培养后动力学”，需要定义统一的预培养协议，不能只缩放一个游离蛋白，也不能把本诊断初值直接作为生产默认。当前统一初值约定本轮保留。

[初值敏感性840分钟曲线](../../../../../../../../data/zsy/budding_yeast/analysis/extension_validation_20261113/DAmP_initial_state_sensitivity.png)。长时积累/分类只用于模型敏感性检查，不是对840分钟真实细胞命运的验证。

## 数值、标签与适用边界

329次结果显式变量最小-2.67e-9，隐含RENTP/NET1P最小-1.79e-9，未出现明显负物种。生产与严格容差142组比较，整体RelL2中位0.0378%、最大9.46%，但本次分裂数/六分组/生物标签均未变。较大差异可能混有事件时序和插值影响，不能把默认容差视为所有轨迹均精确。

挑出较大差异及关键PM/DAmP案例，再做20次rtol=1e-10/atol=1e-12复算，20/20成功，严格与更严格的最大RelL2仅0.00270%，分裂数/六分组一致。因此正文定量比较使用严格轨迹。没有修改生产求解容差，349次求解不等于349个独立基因型。

现有中性六分组在PM明显G1阻滞时可给出P5a（低ESP1触发），不能把P5a的旧英文名称当作实际M期机制证据。此处判断采用实际ORI事件、Sic1形式与原生物标签；没有更改标签树，也没有用六分类反推机制合理性。

其他扩展的小扫描全部数值成功，CLB2-pLOF的0.33点保留分裂，较低表达趋于阻滞；CLN2/CLB2通用ts仍只支持“部分表达降低”的解释，SWI5-GAL仅证明模型对额外基础合成有响应。这些不补足等位突变或强度分布的实验校准。

## 复现与输出

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/validate_budding_extension_mechanisms.py \
  --workers 12 --output /home/users/zsy/mydata/budding_yeast/analysis/extension_validation_20261113
# 关键案例更严容差复核；--plot-only只重绘
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/validate_budding_extension_mechanisms.py \
  --workers 8 --verify-ultra --output /home/users/zsy/mydata/budding_yeast/analysis/extension_validation_20261113
```

目录含runs.json/csv、comparisons.csv、summary.json（含源码SHA256）、ultra_verification.csv、trajectories.npz、15组背景曲线PDF及重点图。替代初值/替代PM参数均有显式标记，不会混入生产数据。脚本在scripts/analysis长期保存，数据输出位于机械盘。
