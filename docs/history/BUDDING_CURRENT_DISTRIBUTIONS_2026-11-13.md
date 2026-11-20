# 当前Budding采样分布总表（2026-11-13）

用户确认保留现有扩展，包括四种DAmP、CLN2/CLB2通用ts、SWI5-GAL、SIC1-PM、CLB2-pLOF及普通MC。此处记录实际代码，不更改生产规则。以下主表为mutant-aware profile；MASS与策略专用规则另外说明。范围指条件于选定该模式之后的强度，不是最终样本频率。

`U(a,b)`为数值均匀；`LogU(a,b)`为log坐标均匀，其密度与1/x成比例；clip为裁剪，不是超界重抽。WT为当前canonical参数。

## 活动类型

| 模块 | 合法单独模式 |
|---|---|
| CLN2 | KO、GAL-OE、MC、ts |
| CLN3 | KO、GAL-OE、MC |
| CLB2 | KO、GAL-OE、MC、GAL-MC、ts、db、pLOF |
| CLB5 | KO、GAL-OE、MC、db |
| SIC1 | KO、GAL-OE、MC、db、PM |
| CDC6 | KO、GAL-OE、MC、DAmP |
| BCK2 | KO、MC |
| SWI5 | KO、GAL-OE |
| CDC20 | KO、GAL-OE、MC、ts、APC-A、DAmP |
| CDH1 | KO、CA |
| CDC14 | ts、GAL-OE、MC、DAmP |
| NET1 | TAB6-1、GAL-OE、ts、DAmP |
| TEM1 | KO、GAL-OE、MC、ts |
| CDC15 | KO、GAL-OE、MC、ts |
| PDS1 | KO、GAL-OE、db |
| ESP1 | ts、GAL-OE |
| MAD2 | CSL |
| BUB2 | CKO |
| APC_Core | APC-A、ts |

共19模块、64单独模式。PPX/SBF_MBF/Mcm1禁用；TEM1/CDC15没有DAmP。MDT为单独环境控制，不增加遗传模块数。

## MC与GAL-MC

普通MC抽整数N，`P(N)=(1/N)/sum(1/n)`，不是整数等概率。

| 模块 | N支持 |
|---|---|
| CLN2、CLB5、CDC14、TEM1、CDC15 | 2–10 |
| CDC6、BCK2 | 2–15 |
| SIC1 | 2–20 |
| CDC20 | 2–30 |
| CLN3、CLB2 | 2–8 |

MC缩放通道：CLN2的ksn2pp；CLN3的C0；BCK2的B0；CLB2的ksb2p/ksb2pp；CLB5的ksb5p/ksb5pp；SIC1的ksc1p/ksc1pp；CDC6的ksf6p/ksf6pp/ksf6ppp；CDC20的ks20p/ks20pp；CDC14的ks14；TEM1的Tem1T；CDC15的Cdc15T。同模块被缩放的多项共用N。

CLB2 GAL-MC：整数N=2–10，同样1/N权重；`ksb2p=0.12*N`，ksb2pp保持WT=0.04，mdt=150；仍是一个模块，不再叠加普通MC或db。

## GAL-OE

以下为直接赋给目标参数的物理值，不是WT倍率；均按LogU抽样。

| 模块 | 参数 | 范围 |
|---|---|---:|
| CLN2 | ksn2p | 0.06–0.30 |
| CLN3 | C0 | 1–5 |
| CLB2 | ksb2p | 0.002–0.60 |
| CLB5 | ksb5p | 0.004–0.030 |
| SIC1 | ksc1p | 0.03–0.36 |
| CDC6 | ksf6p | 0.10–0.60 |
| SWI5 | ksswip | 0.02–0.10 |
| CDC20 | ks20p | 0.50–12 |
| CDC14 | ks14 | 0.30–1.20 |
| NET1 | ksnet | 0.15–0.60 |
| TEM1 | Tem1T | 2–8 |
| CDC15 | Cdc15T | 3–20 |
| PDS1 | kspdsp | 0.02–0.25 |
| ESP1 | ESP1T | 1.5–6 |

GAL-CLN2另关闭原生调控合成ksn2pp=0；其他未指定参数保留原规则。含GAL的候选mdt=150。

## DAmP、pLOF与非零ts

| 模式 | 分布和实际修改 |
|---|---|
| CDC6/CDC20/CDC14/NET1-DAmP | α=clip(N(0.2,0.05²),0.05,0.40)，相应合成参数同时乘α |
| CLB2-pLOF | α~U(0.10,0.50)，ksb2p/ksb2pp同时乘α |
| CLN2-ts | α=0.2*Beta(0.5,2)，ksn2p/ksn2pp乘α，前者WT为0 |
| CLB2-ts | 同上，ksb2p/ksb2pp乘α |
| TEM1-ts | α~LogU(0.001,0.004)，Tem1T和init_Tem1固定0，ka15pp=WT*α |
| ESP1-ts | α~LogU(0.001,0.004)，kasesp=50α、kdiesp=0.5α |
| NET1-ts | α~LogU(0.025,0.10)，kasrent=200α、kasrentp=α |
| NET1-TAB6-1 | 与NET1-ts相同的结合速率分布，独立名称保留 |

DAmP对应：CDC6三项ksf6p/ksf6pp/ksf6ppp；CDC20两项ks20p/ks20pp；CDC14的ks14；NET1的ksnet。它们不额外缩放蛋白初值。

CDC14-ts保留50/50混合：

- 合成分支：ks14~U(0,0.002)，其余活性参数不改。
- 活性分支：ks14=0.2；β~U(0.05,0.35)，(kppc1,kppf6,kaswi,kacdhpp)=(4β,4β,2β,0.8β)。

均匀连续分布不是精确零原子；合成分支没有新增正概率的零点。

## db与固定关闭

| 模式 | 最终生效规则 |
|---|---|
| CLB2-db | kdb2P=0；kdb2pp~LogU(0.017,0.068) |
| CLB5-db | kdb5pp=0 |
| SIC1-db | kd3c1=0 |
| PDS1-db | kd2pdspp=kd3pdspp=0 |
| CDC20-ts | ks20p=ks20pp=0 |
| CDC15-ts | kkpnetpp=0 |
| APC_Core-ts | ks20p=ks20pp=kscdh=0，init_CDH1T=init_CDH1=0 |
| APC-A（CDC20/APC_Core入口） | ka20pp=0 |
| SIC1-PM | kd2c1=kd3c1=0；保留kd1c1 |
| CDH1-CA | kicdhpp=0、kscdh=0.03 |
| MAD2-CSL | Vi20/Vi20_active=0.01 |
| BUB2-CKO | kbub2/BUB2_active/init_BUB2=0 |
| KO | 该模块既有删除规则对应通道固定0；不是所有初值/所有相关参数都归零 |

同基因双模式允许CLB2/CLB5/SIC1/PDS1的GAL-OE+db，GAL强度按上表抽、db按各自规则执行。CLB2 GAL-MC为独立单模式。不允许任意MC+db、GAL+PM等组合。

## 模块/模式频率、环境与实现状态

各入口的模块/模式选择不统一，不能把catalog.modes里的参考数字当作所有入口的实际概率：

- 普通pure_random在给定L后先均匀无放回抽模块、再均匀抽基础模式，合法复合另有25%扩展分支；冲突/精确模块数检查/救援会改变最终频率。
- 图策略共用_sample_mode时默认MODE_FREQ=class，先给类别loss/expression/activity/stability/structure/event权重0.28/0.28/0.18/0.20/0.03/0.03，再在该基因合法模式上归一化。部分策略可使用flat/role/s5或自身配额，不能统一描述成等概率。
- MASS随机策略保留catalog的模块/模式加权。WT-Blind等专用采样还含自身的effect-space强度设计，本表不把其声明为标准MA随机强度。
- 旧AL budding_pool生产入口仍是模块集合/模式的容量感知配额＋Sobol逆CDF，不是已切换的全随机完整池。其非GAL环境为90/150/160按80%/15%/5%。
- 普通旧静态random的非GAL环境仍75%/18%/7%，且MDT显式模式和专用策略还可影响环境。含GAL时固定150。新完整mutation池的已讨论环境为80%/15%/5%，生产迁移待完成。

新完整随机mutation池的提案律：1159个合法模块集合（L1/L2/L3=19/171/969）等概率抽一个，然后从其合法完整模式组合中等概率抽一个；强度直接使用上述MA随机分布，不用Sobol、不设集合固定配额。因此L先验约1.6393%/14.7541%/83.6066%，是配置提案而非当前生产实测频率。去重、real碰撞救援和仿真失败补采之后频率还会改变。不要把“全随机已获用户要求”表述成“生产已迁移”。

## MASS profile差异

比较最终参数而非只比较sample_scaling_factor返回值：

| 内容 | mutant-aware | MASS |
|---|---|---|
| 普通MC | 整数2至各基因上限，1/N权重 | 相同范围的连续LogU剂量 |
| GAL-CLN2 | LogU(0.06,0.30) | LogU(0.03,0.30) |
| CDC20-ts | 两项合成精确0 | 两项合成乘0.2*Beta(0.5,2) |
| CDC14-ts合成分支 | U(0,0.002) | 精确0 |

MASS的PDS1-db标量接口仍可抽U(0,0.05)，但赋值阶段两降解参数固定0，因此不能误报最终仍有残余。最近的精确关闭更新仅改MA；其他差异没有擅自合并。

## LHS：边际配置，不是mutation选择

已批准的混合边际包含WT原子、已有mutation带来的零原子/连续分布/整数原子；权重匹配上述新随机mutation提案的单参数边际。90个未被活动mutation覆盖的模型参数为U(0.5WT,1.5WT)，未分配额外零点概率。LHS不是把参数所有支持合并成一个min/max盒后均匀抽值，也不会生成mutation标签。参数间联合物理约束、生产LHS抽样和两个新10万完整轨迹池尚未完成。

边际JSON：/home/users/zsy/mydata/budding_yeast/analysis/offline_pool_plan_20261112_exact_zero/mixed_lhs_domain.json。

源码：distributions/mutant_aware.py、distributions/ranges.py、mutations/mutant_aware.py；活动模式catalog.py。与MASS、策略专用扰动及历史数据分开解释；旧NPZ不会自动继承新分布。
