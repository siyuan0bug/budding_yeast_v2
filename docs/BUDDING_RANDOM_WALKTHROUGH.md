# budding_random：从随机数到 NPZ 的实际流程

以下对应默认配置 `budding_random_100000_seed42_modular.npz`，未覆盖两个 profile。
文件名只是约定；真实配置在 NPZ 的 `generation_config` 中。

## 来源

| 内容 | 新模块 | 原始来源 |
| --- | --- | --- |
| 变量、默认参数、初值、ODE、事件、表型、126 real | systems/budding/ | mutant_aware.py |
| 初始基因与模式挑选 | sampling/random.py: sample_pure_random_mutant | generate_pure_random_dataset.py |
| 合法模式、touch set、冲突处理 | systems/budding/catalog.py、sampling/common.py | mutant_aware.py |
| 突变强度 | systems/budding/distributions/mutant_aware.py | mutant_aware.py |
| 参数与初值修改 | systems/budding/mutations/mutant_aware.py | mutant_aware.py |
| 碰撞救援 | generation/policies.py | MA 的 KO 弱化/碳源漂移规则 |

本配置不使用 MASS profile，也不使用图、motif、Gibbs、Sobol 或 BAMS 二分细化。

## 随机数、Level、基因和模式

`RandomState(42)` 建立 MT19937 随机流。同一条流用于基因、初始模式、复合模式、冲突重抽、
幅度、突变实现内部的随机分支、碳源和 real 碰撞救援。不是逐个样本重新 seed。
生成入口暂存并恢复 NumPy/Python 全局 RNG；本配置没有依赖跨运行的 L1 网格或配额历史。
seed 不是 ODE 参数，也不是数据拆分/模型初始化 seed；改变它改变合成候选，126 real 不变。
相同软件、配置和 seed 可复现候选流；机器负载导致的超时可能改变最终保留集合。

100000 个候选按固定配额生成：L1/L2/L3/L4 各 25000，顺序为 L1→L4，不打乱层级。
Level k 最初从 27 个基因/功能组均匀无放回选择 k 个，然后对每组的合法模式等概率抽一种。
初选不使用 `P_bio_G` 或模式字典中的频率权重。

CLB2、CLB5、PDS1、PPX、SIC1 可以在 25% 概率下尝试叠加一种互补模式：
表达类 GAL-OE/MC 与结构类 db/PM/CA 互补；只在对应合法集合存在时添加。
之后用 touch set 排除多个修改覆盖同一参数的冲突：尝试重抽本方或对方模式，
最多 30 次，无法消解时删除冲突项；全部为空则重新抽此候选。
**冲突重抽默认采用 MA 的 class 权重，而不是初选的等概率模式。**
因此 Level 是最初计划层级，最终基因数可能减少；叠加模式不增加被选基因数。

## 强度与参数修改

LogU(a,b) 表示 log10 空间均匀，不是线性空间均匀。

| 模式 | 实际强度分布 |
| --- | --- |
| GAL-OE | 基因专用 LogU 区间，如 CLN2 [.06,.30]、CLB2 [.002,1.50]、CDC20 [.50,12]；这是参数赋值范围，不能统一解释为 WT 倍数 |
| MC | 整数 n=2…30，概率正比 1/n；按 MA 规则放大相应表达参数。代码实际上限为30，旧注释中的50不是当前实现 |
| ts | 通常 .20×Beta(.5,2)；CDC14 为 U(.05,.35)，TEM1/ESP1 为 LogU(.0005,.004)，NET1 为 LogU(.0125,.10)，CDC20/CDC15/APC_Core 为 U(0,.01) |
| DAmP | N(.20,.05²)，越界值裁剪到 [.05,.40]，不是截断后重抽 |
| pLOF | U(.10,.50) |
| db | CLB2 为 U(.02125,.170)；PDS1/CLB5/SIC1/PPX 等为 U(0,.01)，再按各自降解参数语义赋值 |
| KO / 其他结构模式 | 调用 MA 专用规则；KO 不是全部初值归零。部分其他模式内部仍有随机分支，如 TAB6-1、APC-A |

所有 GAL 区间及每种模式的具体参数键，以两个 profile 模块为准。
例如 CLB2-db 将 kdb2P=0、kdb2pp=WT×alpha；不是将全部 CLB2 参数统一乘 alpha。
TEM1-KO 会联动 Tem1T、ka15pp 和 init_Tem1；CDC20/PDS1-KO 保留需要自然衰减的初值。

有 GAL-OE 时 mdt=150；否则在未被其他规则设定时，90/150/160 的概率为 .75/.18/.07。

## real 碰撞、过滤与失败

候选与 126 real 比较：非 init 参数用 MA 的容差，init_* 覆盖字典要求相等。
这沿用原规则，并不是对所有等价初值表示进行数学上的完全去重。
普通 Budding Random 不做合成样本之间的全局去重。

碰撞后尝试救援：70% 分支尝试把一个合法 KO 改成 DAmP/ts/pLOF 并重建修改；
没有可改的 KO 或分支未选中时，将 mdt 从150/160等概率重选。仍碰撞则再试。
默认最多30次，耗尽会中止生成，不保存碰撞候选；因此最终被接受分布有条件偏移。

初值处理和事件求解使用 MA：BDF，rtol=1e-4、atol=1e-6，0–210分钟，500个均匀输出点。
事件按原规则重置并继续积分，最多300轮；不外推未积分的尾段。
单样本5秒硬超时；并行池35秒没有任何任务完成时终止本池并中止生成。

求解失败、超时、未完整到达210分钟、非有限轨迹或错误形状不入库；max(MASS)>10也不入库。
普通 Random 不使用曲率门控、WT-Blind轨迹上限或只保留 viable 的筛选。
有效求解出的 Inviable / Pattern_C 样本会保留。求解失败和过滤不会自动补足100000个。
默认126 real 必须全部有效，否则拒绝导出不完整测试集。

## 排序、配对和保存

没有 Twins 配对、母体关系或双臂同步淘汰；`pair_id=-1`。
并行结果按候选原始顺序收集，real 按 canonical 顺序在前，之后是保留下来的合成候选。
未过滤时共100126行；过滤后少于这个数，Level比例也可能变化。

轨迹 float32、参数 float64，形状分别为 (N,39,500)、(N,141)。
141维参数不包含所有可能的额外初值键，因此还保存完整 `y_init` 和 `overrides_json`。
`is_real` 明确标记每行身份；`time_grid` 保存实际物理时间；样本编号可因过滤而不连续。
生成报告记录碰撞、救援、失败与实际保存数。原子写出，默认不覆盖已有文件。
