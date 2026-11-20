> 后续状态：当前新分布的百万pool和共享初始集已生成，通过契约与幅值审计；19.55%→11.79%为初始train的大值比例变化，不是模型精度。详见[新输入验收](GENE_RANGES_DATA_PREPARATION_2026-10-29.md)。

# 已实施采样范围与“接近KO”对照（2026-10-29）

## 当前实施范围

用户确认后的范围已落地。MC在默认MA中为整数、基础权重1/N；MASS保留连续对数均匀的等效剂量，但上下界更新为同一逐基因范围，未悄悄四舍五入。

| 基因 | MC范围 |
|---|---|
| CLN2、CLB5、CDC14、TEM1、CDC15 | **2～10**（按最新批注） |
| CDC6、BCK2 | 2～15 |
| SIC1 | 2～20 |
| CDC20 | 2～30 |
| CLN3、CLB2 | 2～8，仍属缺少普通MC real校准的探索设定 |

CLB2普通GAL的ksb2p改为LogUniform(0.002,0.6)。新增独立`CLB2[GAL-MC]`：GAL基底0.12，N为2～10整数，权重1/N，最终ksb2p=0.12N，ksb2pp保持WT=0.04，mdt150。它计一个模块；metadata中的`mutation_components`保存`gal_base_strength`、`copy_number`、`effective_parameter`、`effective_strength`。没有自动添加GAL-MC+db。L1仅有9种完整输入，排除#50后剩8种，按有限容量处理。

获批的三个非MC修改：

- CLB2-db：kdb2P仍0；kdb2pp改为LogUniform(0.017,0.068)，对应WT残余比例0.0425～0.17。
- TEM1/ESP1-ts：残余活性比例LogUniform(0.001,0.004)，保留原总量、初值和联动规则。
- NET1-ts/TAB6-1：两项结合速率共同乘LogUniform(0.025,0.1)。

两套profile均采用上述获批范围；为使MASS的TEM1/ESP1-ts与TAB6-1也实际生效，原来固定的这些合成赋值已改为按新范围缩放/抽取。没有将其余MASS机制全部统一为MA：例如其CDC14合成失活/APC-A/部分db仍有原来的精确零赋值。

此前建议的CDC20/CDC15/PDS1/SWI5 GAL范围缩窄**未实施**；SWI5、DAmP、PPX等保持原方案。新增“50%精确零＋50%近零”**未实施**。当前MA原本已有的某些0～1%残余近似也没有被冒充为实验证实的通用规律；应进一步按各等位型的机制决定，而非统一给KO/real零值加抖动。real定义、KO及PM固定规则未修改。

## 小残余值能否模拟KO：49组诊断

同一39维ODE、初始化和事件；四个CPU进程，rtol1e-7、atol1e-9、max_step0.25，210min/500点；49组均成功返回，没有做FNO训练。ESP1总量/初值均为0的输入仅为数学对照，不是新增real或生产突变类型。NET1的参照则为canonical real的结合比例0.05，不冒称完整NET1基因删除。

### TEM1-ts：在当前模型里，整个区间都与KO相同

WT、CDC14-MC2、CDC15-MC5三个背景，各扫描ka15pp=0、0.0005、0.001、0.002、0.004。相同背景的全部39变量轨迹与ka15pp=0的TEM1-KO对照逐元素一致，RelL2=0。

原因是当前TEM1-ts赋值已令Tem1T=0、init_Tem1=0；ODE保持TEM1=0，而ka15pp在CDC15激活项中乘TEM1。于是该残余参数在这些条件下失去作用。**这不是原下界特别小的功劳，也不是扩大/缩窄范围能解决的问题。** 要让ts与KO呈现不同机制，须另行审查原模型参数化。本次没有修改canonical real或ODE来制造这种区别；原始mutant_aware也具有相同规则。

当前去重按完整生效参数/初值/事件输入比较，不做条件依赖的ODE等价类合并。因此不同ka15pp仍可能是不同输入、却有同一轨迹；输入数量不能直接解释为动力学多样性。本次没有据此删除TEM1-ts类型或改写real。

### CLB2-db：降解通道失活不是CLB2基因失活

| WT背景条件 | CLB2峰值 |
|---|---:|
| CLB2-KO | 0.146923（初值残留；原模型不清此初值） |
| db，kdb2pp=0.0085（旧下界） | 17.2452 |
| db，kdb2pp=0.017（新下界） | 12.6133 |
| db，kdb2pp=0.034（real） | 5.4537 |
| db，kdb2pp=0 | 19.9612 |

越小的降解参数可令蛋白更稳定，和删除CLB2合成的方向不同。在CDH1-KO背景，kdb2pp乘CDH1，扫描的这些值甚至得到相同轨迹，进一步说明不能脱离背景解释参数大小。

### ESP1-ts：两个速率趋零，也没有删除现存蛋白

WT背景下，将kasesp/kdiesp共同残余比例从0.0005降到0，free ESP1峰值约0.8834→0.99995；总量/初值均0的数学对照峰值才是0。现有初始游离/结合池及其他释放通道仍在，零反应速率不能自动替代基因删除。加入PDS1-GAL-db背景也未变成蛋白删除。不能以某一个总体轨迹误差较小就断言ESP1-KO被正确模拟。

### NET1-ts/TAB6-1：改变的是结合，不是简单降低NET1含量

WT和CDC15-ts两个背景扫描结合比例0、0.0125、0.025、0.05、0.1，出现明显不同的游离CDC14和下游轨迹。canonical real为0.05；趋零可用于结合能力完全丧失的数学对照，但不能据此将其直接标成某个新的真实KO构建体。当前real中的NET1删除代理也是模型级映射，应保留其原定义。

## SWI5那条建议是什么意思

WT的ksswip=0.005。当前GAL范围0.02～0.10，即恒定合成项为WT的4～20倍；提议的0.01～0.03是2～6倍，目的是减少合成放大。它不是修改SWI5-KO，也不是抑制变量输出。因无直接同模式real校准，这一缩窄仍需验证，**本次未采用**。DAmP/通用ts/PPX的意思也是保留既定类型与当前参数规则，不是假称它们有同模式real校准。

## 上下游同步与验证

- 共用范围定义：`data/systems/budding/distributions/ranges.py`。随机与Sobol同时读取；TAB6-1内部抽样同步更新。
- 新模式接入静态各策略、Twins、碰撞救援及WT-Blind；保留原模块选择算法。BAMS细化不再越过当前基因范围，默认MA的MC细化保持整数。
- pool有限枚举与容量乘积改为各基因支持集，不再硬编码29；GAL-MC只枚举mdt150。百万池**规划**L1/L2/L3=9777/126089/864134；并未生成新百万池。
- ours响应选样使用新离散CDF，支持GAL-MC的0.12N生效强度。文件新增范围指纹，初始集复用也会检查源pool契约，避免新配置误用旧缓存。
- **94项相关回归通过**，覆盖两套profile、随机/逆CDF、静态策略、有限容量、模块计数、救援、AL响应特征及缓存拒绝。另完成256条真实新参数池和8条非real GAL-MC轨迹的实际导出/重读，参数、float64初值、mutation组件、生物/ABC/六分类标签齐全。验证脚本最初误写字段单数`pattern_group_code`，改成既有契约复数`pattern_group_codes`后通过；生产标签字段没有改变。
- 本轮之前保存的60个原始data根目录/Fission/核心生物模型文件哈希全部未变。原始生成器、正式pool/初始集/训练结果未覆盖。没有启动正式训练。

**后续正式实验必须重新准备pool和初始集。** 旧文件仍是旧规则的数据，新代码会拒绝将其作为当前候选缓存；不能只换配置继续声称已采用新分布。新输入准备也应与当前已完成实验分开留存来源。

诊断产物：`/home/users/zsy/mydata/budding_yeast/analysis/weak_loss_vs_ko_2026-10-29/`内results.json、summary.csv、trajectories.npz、weak_loss_vs_ko.png/pdf。实际接口验收：`/home/users/zsy/mydata/budding_yeast/analysis/approved_ranges_validation_2026-10-29/verification.json`及两个小型NPZ。上述均是诊断文件，不是正式AL输入。
