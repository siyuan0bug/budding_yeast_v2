# 从 real-mutant 失败机制到神经算子设计：2026-10-30

这次建议把研究主线定为 **Intervention-Conditioned Hybrid Flow Operator（ICHFO，暂用名）**：学习给定干预背景下的局部连续流，通过可重复的物理事件和重置连接轨迹，并用合成的同背景干预对监督条件响应。优先解决错误启动和上游驱动；标记变量的解析闭合是一个可独立验证的消融。

当前已经完成代码阅读、历史指标重新汇总、20 个坏例逐通道诊断、冻结模型的读出替换实验，以及一个可训练的 `biom_refiner_marker_closure` 实现。**完整 ICHFO 尚属设计；没有开展正式训练、没有新 SOTA 结论。** 本文将归档观测、新诊断、待验证假设明确区分。

## 1. 任务和代码的实际含义

| 路径 | 作用与本次核对重点 |
|---|---|
| `data/systems/budding/parameters.py` | 141 维导出顺序；包含 `init_BUB2/init_CDH1/init_CDH1T`，并非全部都是反应速率 |
| `states.py` | 39 变量及标准初值 |
| `ode.py` | Chen 型质量作用、GK 超敏开关、四个方向性事件 |
| `simulation.py` | BDF 分段积分、状态重置、统一网格插值 |
| `real_mutants.py` | 生物学指定的 mutant 参数规则、有效输入去重 |
| `recipes.py`、`specification.py`、`adapter.py` | 合成采样配方、MA/MASS 后端、仿真与保存契约 |
| `samplers/`、`mutations/`、`distributions/` | 模块组合、KO/MC/GAL 等机制与幅度分布 |
| `data/yeast_datamodule.py`、`normalization.py` | synthetic 划分、train-only 拟合、物理空间解码 |
| `models/yeast_lit_module.py` | 注册、统一 forward、coarse/final 联合训练、checkpoint 统计量 |
| `models/ours/bio_refiner_plus.py` | 全时域粗预测与局部残差修正；曲率、多尺度、图注意力 |
| `bio_event_hybrid_operator.py`、`bio_event_refiner.py` | 旧事件历史与相区特征；见第 6 节的限制 |
| `models/baselines/`、`../baselines/` | 当前标准适配与原始论文代码；不能把旧实现指标改名归给当前基线 |

输入契约为 `X:(B,39,2,T)`，两个特征是**重复的初始浓度和查询时间**；不是过去两帧或初值导数。参数 `P:(B,141)`，输出 `Y:(B,39,T)`。默认 Y 为 `log1p + per-variable Z-score`，P 是原始参数的逐坐标 Z-score；参数没有随 Y 一起取 log。

当前 real-mutant 的 39×500 轨迹也是由 ODE 求解器产生，`adapter.py` 对 real 与 synthetic 都调用仿真。准确的现有任务表述是：**从合成干预分布泛化到生物学指定的 mutant 条件**。实验文献为 mutant/表型提供生物依据，并不使保存的完整轨迹变成实测浓度序列。论文若使用严格的“sim-to-real trajectory prediction”表述，需要新增真正实测的时序/表型验证。[Chen et al., 2004](https://pmc.ncbi.nlm.nih.gov/articles/PMC491841/) 是机制背景；本文的指标数值来自本地存档。

现有代码足以确定系统具有刚性、多时间尺度、超敏反馈和混合事件；未找到对这些具体 mutant 进行混合系统 Lyapunov 指数估计的证据。“耦合混沌”应作为需要验证的动力学命题。周期振荡、阻滞和事件时间敏感本身不证明混沌。

## 2. 分析范围、指标和可比性

本次保存的读取快照包含 **192 份 Budding CSV + 130 份当前 AL 逐轮 JSON，共 322 份评测**；2 份 Fission CSV 排除。AL 实验仍在运行，今后重新扫描总数可能变化。所有原始来源路径和 CSV/JSON 的 SHA256 均保存在 `runs.csv`，原实验文件未改写。

重点横向比较固定数据集 `mutant_aware_pure_random_100000_210m_500s` 的 **54 份非备份历史配置**，全部是 seed42。其中文件内容去重后只有 50 份：Flowformer/Flashformer 一组，以及四个 i-变体一组，指标 CSV 分别逐字节相同。它们保留在档案表中，综合坏例投票只给相同 SHA256 一票。

同数据集不等于严格公平：loss、normalization、参数量和历史代码版本仍不同；100 epoch 也不等于相同计算预算。当前标准 baselines 在 2026-10-21 重构过，旧 `fno1d/unet1d/mwt1d/...` 的数值只能描述旧实现。当前 AL 的标准 FNO、DeepONet、U-Net 与静态历史结果分别归档；不同采样池、初始集、轮次和训练预算不直接混排。

逐 mutant 的 RelL2 为

\[
 r_i=\frac1{39}\sum_v
 \sqrt{\frac{\operatorname{mean}_t(\hat y_{ivt}-y_{ivt})^2}
 {\max(\operatorname{mean}_t y_{ivt}^2,0.1^2)}}.
\]

`per_mutant_metrics.csv` 的均值是 `mean_i r_i`；全局 JSON 的 pooled RelL2 先把全部样本/时间合并到每个变量，再取比值和平均，二者不同。MAE/MSE 在等长样本下可以先算逐 mutant 再平均。相关系数对恒定通道返回接近 0，不能把单条相关性低直接判定为动力学失败。

本次分析的新入口为 `scripts/analysis/analyze_operator_failure_modes.py`。旧审计的训练支撑和固定背景 ODE/FNO 对照是**已存在证据的复用**，见 [2026-10-27 好例/坏例对照](GOOD_BAD_CASE_COMPARISON_2026-10-27.md)。本次新增的是跨模型重新统计、三标记物的 2×2 驱动/事件日程替换诊断、实现和验证。

## 3. 不同模型究竟好在哪里、差在哪里

下表全部从相同数据集的 per-mutant CSV 重新计算，RelL2 是逐 mutant 平均。所有数值都是历史单 seed 归档，不是当前代码重训。

| 历史模型 | loss | MAE | MSE | mean RelL2 | 最差20条 RelL2 均值 |
|---|---|---:|---:|---:|---:|
| BioRefinerPlus | triple_dynamic | 0.05673 | 1.00213 | 0.07408 | 0.28608 |
| EventHybrid | triple_dynamic | 0.07076 | 1.97730 | 0.11937 | 0.56144 |
| CausalReach | hybrid_bifurcate | 0.07836 | 3.28737 | 0.14867 | 0.74562 |
| BioRefinerPlus | hybrid_mse_rel | 0.08691 | 5.18461 | 0.14265 | 0.68618 |
| Bregman-WNO | hybrid_mse_rel | 0.09116 | 3.09348 | 0.11674 | 0.42158 |
| U-Net1d | hybrid_mse_rel | 0.09635 | 5.15945 | 0.14857 | 0.64825 |
| Transolver1d | hybrid_mse_rel | 0.10701 | 8.00400 | 0.14901 | 0.60980 |
| MWT1d | hybrid_mse_rel | 0.10962 | 2.93948 | 0.16162 | 0.58465 |
| AttractorPlus | hybrid_bifurcate | 0.11202 | 7.81337 | 0.13414 | 0.57358 |
| KoNODE | hybrid_mse_rel | 0.11226 | 2.55590 | 0.15610 | 0.53438 |
| GNOT | hybrid_mse_rel | 0.11301 | 5.96140 | 0.15985 | 0.61620 |
| FNO1d | hybrid_mse_rel | 0.11799 | 7.19826 | 0.13356 | 0.51125 |
| DeepONet | hybrid_mse_rel | 0.18957 | 9.08480 | 0.21231 | 0.55968 |
| LaMO | hybrid_mse_rel | 0.20049 | 6.37591 | 0.29393 | 0.89452 |
| Neural ODE | hybrid_mse_rel | 0.28264 | 14.92526 | 0.39976 | 1.24365 |

几个比“哪个模型均值最好”更有用的结果：

1. **Plus+hybrid 的多数样本更好，但尾部更脆弱。** 逐例 RelL2 比较，历史 FNO 只在 7/126 条上优于它，FNO 的平均 RelL2 却更小（0.13356 vs 0.14265）。Plus 的 median RelL2 是 0.03357，FNO 为 0.05806；最差20条均值则相反（0.68618 vs 0.51125）。架构设计应直接评估尾部，不能只优化总体或中位数。
2. **小波值得保留为强基线。** Bregman-WNO 的 MAE 比 Plus+hybrid 略差，MSE、RelL2 和尾部更好；它只在 23/126 条的 RelL2 上胜出，优势集中在较难案例。局部多尺度表示是合理候选机制，但这组统计没有证明其因果来源。
3. **triple_dynamic 是值得重新验证的低成本对照。** 相对 Plus+hybrid，MAE/RelL2 分别改善 86/126、91/126 条。#15 RelL2 从 7.536 降至 0.704，#124 MSE 从 557.40 降至 57.19；#22 RelL2 却从 0.478 升至 0.508。它缓解部分失败，没有普遍解决事件/条件泛化。
4. **复杂机制模块已有负例。** 同用 triple_dynamic 的 EventHybrid 总体逊于 Plus；#15 RelL2 为 6.105。CausalReach 的 #42 较好（0.303），#15 却为 9.307。AttractorPlus 的 #124 MSE=876.06。模块名字和解释不能替代逐例证据。
5. **连续潜在动力学不自动处理状态跳跃。** Neural ODE/KoNODE 的连续演化结构可用于局部流；要表达 MASS 和外部开关的瞬时重置，还需显式事件连接。KoNODE 当前归档 MSE 较好并不证明其离散事件已正确。

从现有代码和 `../baselines/` 可借鉴的具体内容：

| 家族 | 可借鉴部分 | 在本任务需要验证的缺口 |
|---|---|---|
| FNO/Bregman-FNO | 全局连续流表示、残差更新、推理吞吐 | 跳跃附近误差、错误事件词、极端条件响应 |
| WNO/MWT/U-Net | 局部突变和时间尺度分解 | 分辨率一致性、长程周期和跨事件泛化 |
| DeepONet/GNOT/LNO/LaMO | 查询坐标解码、条件分支、低维潜在场 | 模式组合外推、事件定位；时序预测论文结果不可直接套用 |
| Neural ODE/KoNODE | 局部状态演化和可组合性 | 刚性求解成本、离散重置、事件时刻梯度 |
| NCFlow | 同背景下参数/上下文响应的正则化思路 | 原方法涉及新环境适应；本任务零样本条件下不能使用 real 真值适应后声称 zero-shot |
| SC-FNO | solver-derived Jacobian 监督 | 跨事件敏感度和近切换区不可直接套平滑导数；必须计入附加 oracle 成本 |
| BioRefinerPlus | 已有很好的中位数拟合、coarse/final 训练接口 | 未强制反应条件乘积、缺少事件局部流契约、尾部脆弱 |

以上缺口中，跨案例统计是观测，架构归因仍要依靠固定数据/预算的消融。

## 4. 20 个代表性 badcase

选择规则预先固定为：在 **Plus / hybrid_mse_rel / log1p / seed42** 参考实验内，分别对 MAE、MSE、RelL2 降序排名，取三者平均名次最小的20条。原始指标、综合名次、各指标独立 Top20 和跨50份非重复结果的共识排序均单独导出。下面的三项数值来自归档 CSV；通道占比来自已匹配归档的冻结重放，有微小重放误差。

| ID | 主要干预背景 | MAE | MSE | RelL2 | 冻结重放主要 SSE 通道/占比 | 首要诊断方向 |
|---|---|---:|---:|---:|---|---|
| 22 | CLN2/CLN3 关闭 + MC-CLB5 | .4630 | 26.4902 | .4776 | ORI 98.49% | 补偿启动、尾部和退出时刻 |
| 15 | cln3Δ bck2Δ + MC-CLN2 | .6029 | 4.8241 | 7.5363 | CLN2 35.11% | 错误条件响应和假 START |
| 124 | APC-A + GAL-CLB2 | 3.3037 | 557.4008 | .3249 | ORI 99.45% | 上游持续驱动幅度/时序 |
| 23 | CLN2/CLN3 关闭 + GAL-CLB5 | .3350 | 8.3415 | .4702 | ORI 94.36% | GAL 与 MC 的作用方式、尾段事件 |
| 20 | cln1/2Δ cln3Δ sic1Δ | .2025 | 2.6029 | .3417 | ORI 86.33% | 补偿路径、分裂后再启动 |
| 59 | CLB2-dbΔ + MC-CDC6 | .2040 | 2.3105 | .3353 | ORI 87.06% | 稳定性与 CDC6 背景、事件日程 |
| 26 | cln1/2Δ cln3Δ apc-ts | .1310 | 1.2422 | .9354 | ORI 82.51% | 启动/降解耦合、低信号相对误差 |
| 54 | GAL-CLB2 cdh1Δ | .2599 | 2.2900 | .1990 | ORI 98.42% | CLB2 驱动偏差与累积读出 |
| 42 | sic1Δ cdc6Δ cdh1Δ GAL-CDC20 | .2138 | .6954 | .4436 | Vi20 37.66% | 四模块组合外推、开关时程 |
| 44 | swi5Δ GAL-CLB2 | .2399 | 1.4877 | .2252 | ORI 94.42% | 反馈/退出与重置时刻 |
| 62 | GAL-CLB2-dbΔ | .3824 | 9.9000 | .0841 | ORI 99.36% | 同模块双机制、高幅值误差 |
| 33 | GAL-SIC1/GAL-CLN2 + cdh1Δ | .1169 | 1.2651 | .3076 | ORI 92.78% | 多条补偿路径与事件时序 |
| 53 | GAL-CLB2 sic1Δ | .1269 | .9420 | .3162 | ORI 81.97% | 抑制背景下的 CLB2 和重置 |
| 46 | swi5Δ cdh1Δ GAL-SIC1 | .1490 | .9386 | .2979 | ORI 90.67% | 后期退出/尾段误差 |
| 24 | CLN2/CLN3 关闭 + MC-BCK2 | .0943 | 1.1457 | .2718 | ORI 90.77% | 补偿后的事件时刻 |
| 75 | cdc20Δ pds1Δ | .3000 | 8.1916 | .0511 | ORI 99.40% | 高量纲误差与相对指标脱耦 |
| 66 | GAL-CLB5 sic1Δ | .1256 | .7563 | .1753 | ORI 97.45% | 受驱动标记物输出一致性 |
| 122 | APC-A cdh1Δ + MC-CDC20 | .1170 | 1.0522 | .1460 | CDC20T 93.97% | 总量驱动/降解方程 |
| 60 | CLB2-dbΔ clb5Δ | .1669 | 1.6267 | .0845 | ORI 99.42% | CLB2 驱动与标记物闭合 |
| 57 | CLB2-dbΔ + MC-SIC1 | .0843 | .5271 | .2539 | ORI 83.92% | 稳定性补偿、末端退出 |

“首要诊断方向”不是已经证明的唯一根因。20例中的17例由 ORI 主导物理 SSE；#15、#42、#122 分别是 CLN2、Vi20、CDC20T。某些旧审计使用不同模型/重放得到 #42 的主通道为 ORI；这里以本次冻结轨迹计算为准，不能跨模型套通道归因。

每条样本都有 MASS、CLN2、CLB2、CLB5、ORI、SPN 六变量对比图。完整产物目录见第10节。

## 5. 三个决定模型方向的案例

### 5.1 #15：需要学会条件参数响应

方程为

\[
 \dot {CLN2}=(k_{sn2p}+k_{sn2pp}\,SBF)MASS-k_{dn2}CLN2.
\]

MC 改变的是 `ksn2pp`，其效应受 SBF 控制；GAL 则可以改变组成型项。用一个扁平 MLP 把“更大 ksn2pp”编码为“更强 CLN2”，可能在缺少上游启动的背景下产生错误响应。

本次冻结图显示，参考模型在约70–130分钟产生不应有的 CLN2/CLB2/CLB5 激活，MASS 还出现虚假分裂。真实 CLN2 峰值约0.06525，模型峰值超过16；参考 RelL2=7.536。替换 ORI 读出不能消除已经错误的 cyclin 驱动。

复用的固定背景对照更有说服力：此前 FNO 在双 KO 背景将 CLN2 从1拷贝变为5拷贝，预测 CLN2 峰值0.725→14.107，而 ODE 峰值保持约0.06525。训练集中还有一条完整39变量轨迹与 #15 RelL2 差仅0.000297的阻滞样本。因此该例支持“需要学习跨输入的条件关系”，不能概括成“没有见过相似输出”。这也不能硬编码成 `C0=B0=0` 永不启动；已有更强拷贝数的 ODE 反例。

### 5.2 #124：上游驱动是主要瓶颈

ORI 满足

\[
 \dot O=k_{sori}(e_{orib5}CLB5+e_{orib2}CLB2)-k_{dori}O.
\]

本次增加了驱动/重置日程的2×2替换。事件日程由各自500点 cyclin 轨迹的方向性阈值交叉估计；标记物在间隔内采用线性驱动解析积分。涉及真实驱动或真实日程的行只用于诊断。

| #124 ORI 方案 | ORI MSE |
|---|---:|
| 原始独立预测头 | 21614.922 |
| 预测驱动 + 预测日程的闭合 | 20686.832 |
| 预测驱动 + 真实日程（oracle） | 20687.012 |
| 真实驱动 + 预测日程（oracle） | 0.03512 |
| 真实驱动 + 真实日程（数值自检） | 0.00116 |

实际曲线中，CLB2 被明显低估且启动偏晚，ORI 随之被低估。保持预测驱动时只修日程几乎无效；换驱动后误差显著下降。**该冻结模型中上游 cyclin 驱动更值得优先修复。** 这些 MSE 不是可相加的误差分解，也不能据此指定某一个训练参数是唯一原因。

其他样本不同：#59 的 ORI MSE 从原始78.42，变为预测驱动/预测日程99.05；采用真实日程可降至5.69。#22 的对应值为1017.61→312.20，真实日程269.23，真实驱动/预测日程150.30。因此需要同时测上游流和事件，而不是给所有坏例套一个机制。

### 5.3 #42：组合外推和外部开关

#42 有四个功能模块；当前 L≤3 的生成域无法直接给出相同四模块组合。它适合单列为组合 OOD，不能通过增大 L3 比例声称已经覆盖。

本次参考模型的主要 SSE 通道是 Vi20（37.66%），反映复制/纺锤体事件后 checkpoint 开关的时程错误也必须检查。Biological pattern 分类和连续轨迹准确性是两个任务。

## 6. 现有 ours 的结构限制

1. **事件历史只有全程锁存。** `CausalODEEventDetectors` 对每个事件作全时间段 `cummax`。一旦进入 Exit，其历史不会在下次周期复位。这种表征可作为第一次事件历史，不能等同于完整的多周期混合自动机。
2. **物理 guard 没有被保留。** 它对标准化 CLB2、CLB5 直接求和，再由参数网络学习阈值。一般 `z_CLB2+z_CLB5` 不等于物理 `CLB2+CLB5` 的固定阈值坐标；网络可拟合替代特征，却没有保证与原方程的0.2一致。
3. **增加了求解器没有的固定事件前置约束。** 当前事件模块强制 E3→E4→E1→E2，原生求解器在每个分段同时注册四个方向性 guard。异常 mutant 可能绕过正常事件顺序，直接把典型生物顺序当硬约束会缩小可表达范围。
4. **图注意力是软偏置。** `softmax(QK + 0.5*graph_bias)` 允许图外通道通信；潜在通道还被重新映射成“物种”特征。它不是反应化学计量矩阵，也没有强制 source×rate 的乘积。FluxEncoder 的学习权重也不能直接称为测量的反应通量。
5. **差分是采样索引差分。** Plus 的 dy/d2y 没有除以物理 Δt；改变查询分辨率时特征尺度变化。可作为粗轨迹形状特征，不能直接称为同量纲物理导数。
6. **coarse→refine 不能保证纠正错误拓扑。** 如果 coarse 已假启动，最终的小残差容易沿错误分支修波形；必须单独监督事件、响应和局部流，不能只依赖最终 MSE。

全时域 FNO 读取未来的查询坐标/自身预测不构成未来真值泄漏；代理模型也不一定需要按查询点严格因果。这里的问题是没有实现所声称的物理自动机、重置和条件依赖，不应把注意力方向性等同于完整因果识别。

## 7. 新设计：ICHFO

### 7.1 先使输入决定问题

保留完整 IC 与参数，额外显式提供外源干预程序 `u(t)`，例如开关类型、开始时间和幅度。不能从 mutant ID 推断该程序。

本次原生 ODE 对照验证：#104 带/不带 `is_mutant_104` 时，导出的141维参数和39维初值完全一致；实际轨迹却不同（全轨迹 MSE≈0.0001845、最大绝对差≈0.2721）。该 flag 控制180分钟后的 SIC1 合成，未进入现有 P。若未来数据同时包含这两种条件，现有输入无法唯一决定输出。这个对照说明输入契约的限制；未宣称当前存档必然含这两个相同输入的训练行。

新增程序输入应版本化；旧 checkpoint 按原接口复现。三标记闭合消融仍使用原接口，没有暗中为 #104 添加 ID 信息。

### 7.2 用反应局部条件表示代替无约束的全局干预效应

构建物种—反应二部表示，分别绑定相应参数、反应底物/激活量和完整初值。已知的源项、降解项、GK 激活和结合项作为明确的 feature；可学习部分负责有限时间流和未闭合修正。既要输入 `ksn2pp`，也要表达它与 SBF/MASS 的联合条件。

优先使用来源于源码的 ODE potential graph；软图偏置只作参照消融。不要对固定 double-KO 背景硬编码“不启动”，也不要把所有边的符号固定为不随状态变化的全局因果结论。

为保留速度，主网络学习局部解映射，而不是重新逐步调用完整刚性 RHS 积分器：

\[
 \mathcal F_\theta:(y_k,p,q_k,u_k,s)\longmapsto \tilde y_k(s),\quad s\in[0,h_k].
\]

`q_k` 是 Vi20/lte1/BUB2 等离散/分段常值记忆，`s` 是段内物理时间。使用坐标解码器支持不同查询网格；FNO、小波、GNOT 可作为段内骨干消融。起点通过显式残差构造满足 `F(y,p,q,u,0)=y`，无需额外惩罚才能记住 IC。

### 7.3 事件调度和连续流分开，但相互闭环

每个段同时评估四个 guard 的**首个有方向交叉**；取最早者，施加原系统 reset 后重新预测下一段。允许任意次数的 E1/E2/E3/E4，允许一段没有事件，也允许 mutant 的异常顺序。细胞周期仅作为可观察事件模式，不预先强制固定状态路径。

| 事件 | guard/方向 | 必须实现的主要动作 |
|---|---|---|
| E1 | CLB2−0.3，下降 | MASS 分配；BUD/SPN 清零；lte1→0.1 |
| E2 | CLB2+CLB5−0.2，下降 | ORI 清零 |
| E3 | ORI−1，上升 | Vi20/BUB2 按对应 mutant 参数置活跃状态 |
| E4 | SPN−1，上升 | Vi20→0.01；lte1→1；BUB2 按初始设定/KO规则复位 |

MASS 的 E1 比例应复现当前代码：`exp(-1.026 + 32*log(2)/mdt)`，不能用任意0.5或0.3代替。原生 guard 变量的 `1e-6` 防重复触发偏移属于数值实现约定，训练/评测需要统一。外部定时操作作为独立已知事件排入调度。

训练数据应额外保存 solver 的准确 `event_type/event_time/y_minus/y_plus`；不能把500点曲线的阈值估计当成精确事件时间。两侧极限和 guard crossing 比常规网格导数更适合监督跳跃。没有事件的样本需要右删失/无事件监督，避免强迫每条轨迹都产生一个“下一次分裂”。

### 7.4 同背景干预对：直接约束 #15 类条件响应

在**只使用合成训练域**的背景 p、相同实际初值下，选择合法同模块强度 `a,b` 仿真成对样本：

\[
 L_{pair}=\left\|
 [\hat Y(p_b)-\hat Y(p_a)]-[Y(p_b)-Y(p_a)]
 \right\|^2_{W}.
\]

W 从 synthetic train 的固定变量尺度构造。它监督真实的条件效应，不假定所有 KO 附近效应为0。初值随突变变化的自然效应与固定初值的参数偏导要分别命名、分别评测。

同事件词、远离 grazing 的小扰动可以加事件时间差/方向 JVP 监督；跨事件数变化用有限幅度响应和事件词监督，避免用普通平滑 Jacobian 强行穿过拓扑变化。所有附加求解次数、事件标注和敏感度计算计入 oracle 预算；普通 baseline 可获得同样轨迹数据。

若已有大批训练集，先从真实存在的同背景邻域组成对；需新增仿真时按预先定义的 synthetic 规则采样，而不是围绕这20个 real 参数定向补点后声称盲测提升。已存在的 ours_response 等 AL 工作是相关思路，不能把“同背景配对”本身再作为独立新颖性声明。

建议的最小训练目标为 `L_flow + λ_event L_event + λ_pair L_pair`，先逐项消融。
`L_flow` 在真实事件分段内监督轨迹，起点包含 synthetic 的真实重置后状态；`L_event` 同时监督
下一事件类型、时间与无事件删失。随后在 synthetic 上进行多段闭环训练，验证时始终从原始 IC
完整 rollout，避免只在真实分段初值上表现好。需要组合性时再单独加入
`F(F(y,p,h1),p,h2)≈F(y,p,h1+h2)` 的段内一致性项；该式只能用于同一事件段或包含正确 reset
的组合，不能跨跳跃直接套用普通半群公式。阶段划分和权重只由 synthetic 开发集选择。

### 7.5 可独立启用的解析读出

ORI/BUD/SPN 为线性衰减 + 已知驱动，段内可用解析卷积；独立总量变量如 CDC20T 也有可利用的结构。复杂耦合变量由局部流网络预测，标记变量由物理读出产生，并将其事件反馈给下一个段。这样避免独立输出头违反明确的驱动关系。

**本轮只实现了这个读出消融，完整局部流、事件调度、反事实训练损失均尚未实施。** 解析读出能够消除一部分输出不一致，但无法挽救错误的上游流；第8节给出了反例。

### 7.6 理论方向：事件时间误差和条件响应的一致性

在固定事件词、guard 横截、`|g_t+∇g·f^-|≥γ>0`、流和 reset 局部光滑时，事件时刻一阶误差满足

\[
 \delta\tau\approx-
 \frac{\nabla g\cdot\delta y+g_p\delta p}
 {g_t+\nabla g\cdot f^-}.
\]

混合流的状态敏感度在事件处由 saltation matrix 更新：

\[
 S=D_yR+
 \frac{(f^+-D_yR f^- -R_t)\nabla g^\top}
 {g_t+\nabla g^\top f^-}.
\]

对参数求导还需显式的 `R_p/g_p` 项，不能只乘 S。隐式求根反向传播和显式 saltation 校正应选择一致实现，防止重复计算修正。

可尝试建立“段内流误差 + 首次交叉时刻误差 + reset 误差”的组合界，并说明反事实监督如何约束相同背景的局部响应。这个界需要给出事件数量、γ、流/reset Lipschitz 条件及累积放大项；不能声称跨 grazing 或事件词变化仍全局稳定。理论尚未证明，saltation 本身是已有工具。[Kong et al.](https://arxiv.org/abs/2306.06862) 提供相关基础。

还需区分误差范数：一个幅度为 J 的孤立跳跃错位 δτ，会产生约 `||J||²|δτ|` 的时间积分平方误差，
因而 L2 误差通常按 `sqrt(|δτ|)` 缩放。不能把段内的光滑 Lipschitz 界直接推广为跨跳跃的全时域
L2 Lipschitz 界。目标理论可在事件词固定、事件间隔分离的条件下控制平方误差中的
`Σ ||J_k||²|δτ_k|`，同时给出连续流与 reset 的误差项。saltation 监督应在事件两侧/光滑分段内
定义，不应未经处理就把移动跳跃当成全时域 L2 中处处存在的普通 Jacobian。

## 8. 已实现和实际测得的消融

新模型文件：`models/ours/bio_refiner_marker_closure.py`；注册名 `biom_refiner_marker_closure`。继承 Plus，保留 coarse/final 接口，只将 ORI/BUD/SPN 改为物理读出，不改变其余36个输出，也不修改 baseline。

实现采用间隔内线性驱动的解析积分与 O(log T) 深度的 affine scan：

`y[k+1] = a[k]*y[k] + b[k]`。

guard 用物理 CLB2=0.3、CLB2+CLB5=0.2 的下降穿越；每次交叉均可重置。区间内插值估计交叉时刻，重置前后的强迫项分开积分。事件存在性是离散选择，梯度流过已选区间内的交叉时刻和连续驱动。它没有实现完整 saltation-aware 学习，没有恢复 E3/E4 对上游状态的闭环作用；同一输出间隔内部多个未被采样解析的事件也可能漏检。

使用 DataModule/checkpoint 明确绑定统计量；支持现有四种归一化，AMP 中读出用 float32。没有统计量时明确拒绝推理。模型不接收真实事件日程；`reset_override` 只是分析函数的 oracle 钩子。

在126条已匹配归档的冻结 Plus+hybrid 轨迹上，替换三标记读出得到：

| 物理指标，逐 mutant 平均 | 替换前 | 替换后 | 逐例改善数量 |
|---|---:|---:|---:|
| MAE | 0.087015 | 0.078941 | 95/126 |
| MSE | 5.183655 | 4.722494 | 86/126 |
| RelL2 | 0.142708 | 0.148833 | 80/126 |

平均 MAE/MSE 分别下降约9.28%/8.90%，平均 RelL2 **升高约4.29%**。ORI 单通道 MSE 198.620→180.604（83/126条改善），BUD 和 SPN 的均值 MSE 反而升高。不能根据多数样本改善或物理闭合就宣布全面收益。

这些是**冻结输出上的确定性替换**；不含重训收益，也不能归因到新模型训练后的泛化能力。重放与归档有小差异，例如参考归档 MSE=5.184605、冻结重放为5.183655；完整差异和上游重放容差已保存，所有前后比较使用同一冻结预测。

验证范围：

- affine scan 与独立顺序递推的值/梯度一致，包括精确归零重置。
- 线性驱动积分与 SciPy quad 对照，包含零衰减、近零衰减和大衰减。
- 重复周期交叉、初始位于阈值下不误触发、IC 保留。
- 四种归一化下前向/反向与严格 state_dict 重载。
- 新模型和原 Plus 在 width192/modes24/n4 下均为 **9,321,111 个可训练参数元素**，未增加可训练参数；复数元素按 numel 计1。
- 真实 ODE 数据前384条（126 real+258 synthetic）的工程冒烟：仅206条 synthetic train 拟合归一化并训练2个优化步，bf16 CPU 混合精度；checkpoint 重载预测最大差0。没有将 real 加入训练，冒烟不是精度实验。
- 针对本次实现及标准 baseline 注册隔离的15项回归通过。

## 9. 顶会研究定位和最小实验闭环

“加图 + 加事件 + 加物理 loss”已有大量先例。Neural Event ODE 已研究可学习事件和跨事件微分，[原文](https://arxiv.org/abs/2011.03902)；Neural Hybrid Automata 已研究多模式混合动力学，[NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5291822d0636dc429e80e953c58b6a76-Abstract.html)。[NCFlow](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f3398b76d17792893ce6d4f660546353-Abstract-Conference.html) 和 [SC-FNO](https://arxiv.org/abs/2505.08740) 分别涉及上下文调制与灵敏度。2026年的 [Cut-DeepONet](https://arxiv.org/abs/2605.19823) 和 [φ-DeepONet](https://arxiv.org/abs/2604.08076) 也明确处理不连续表示。这是定向文献核对，并非穷尽新颖性检索。

因此候选贡献应具体到：**同背景干预响应与重复事件的流映射同时学习，并在事件词外推/近切换处给出可检验的一致性结果。** 解析读出是便于归因的辅助结构，不宜单独包装成核心算法创新。能否形成顶会贡献取决于新实验证据和更完整文献比较，当前归档还不足以支持。

建议按下列顺序实施，阶段间根据合成开发集表现推进：

1. **重建可信对照。** 当前标准 FNO/WNO/MWT/DeepONet/KoNODE + 当前 Plus；同时比较 hybrid 与 triple_dynamic。固定相同样本、split、train-only 统计量、每模型 HPO 预算，至少3个训练/数据种子；不要只给 ours 附加更好的 loss。
2. **先验证最小改动。** Plus 与 marker_closure 同预算重训，确认冻结读出改善能否在端到端学习后兼顾 RelL2。再做反应条件输入、局部流、循环事件、pair loss 的逐项和组合消融，保留负结果。
3. **设计真正的泛化划分。** 基因型/模块组合分组去重；分别测组合留出、强度范围留出、KO 边界、IC 留出、事件词留出和长时程。既有126条已多次用于设计，属于开发基准；论文要增加未参与设计的组合/系统评测，#15 定向改善不能代替无偏结果。
4. **控制先验和 oracle 成本。** 同样的成对轨迹供所有比较方法使用；事件/敏感度额外成本单列。增加已知机制的简单指数积分器/较松容差 BDF，以及部分解析闭合的基线。报告相同物理误差下端到端延迟，包含 root finding/reset/预处理，防止“把原求解器装进网络”却没有速度收益。
5. **同时报连续和离散质量。** per-mutant mean/median/P90/worst20、core9、各通道尺度化误差、事件次数/类型序列、错误启动率、匹配事件时刻误差、闭合残差。跨样本整体 pooled 指标保留，但不能替代尾部。事件未发生时不以随意填充时间计算 MAE。
6. **跨系统验证。** 利用现有 Fission 流水线，再加至少一个带清楚 reset 的混合系统；先验不同的设置分别报告。实测 yeast 数据另作为外部验证。单一系统上的逐 mutant 自举区间不能替代多个训练种子的方差。

应以可证伪的里程碑判断方案：如果同预算重训后 pair loss 没改善未见背景的响应方向，或循环事件模型只改善训练事件词，核心假设就尚未获得支持。如果全部收益可由解析读出或补数据解释，应相应收窄论文贡献。

## 10. 产物和复现

输出目录：

`/home/users/zsy/mydata/budding_yeast/analysis/operator_design_2026-10-30/`

主要文件：`runs.csv`、`historical_cohort.csv`、`paired_against_reference.csv`、`top20_reference.csv`、`top20_evidence.csv`、`consensus_badcases.csv`、`duplicate_cohort_metrics.csv`、`frozen_channel_errors.csv`、`frozen_closure_diagnosis.csv`、`frozen_full_metrics_before_after.csv`、`frozen_provenance.json`、`badcase_model_heatmap.png`、`cases/case_<ID>.png`、`hidden_intervention_104.json`。被排除文件及原因见 `skipped.json`。

分析依赖2026-10-27已有重放/特征缓存，路径和来源均显式记录；没有把旧 checkpoint 用当前错误统计量重新解释。analysis 输出不提交到代码仓库。

```bash
cd /home/users/zsy/projects/budding_yeast_cell_cycle
export PYTHONPATH="$PWD"
PY=/home/users/zsy/.conda/envs/neuralop/bin/python

$PY budding_yeast_v2/scripts/analysis/analyze_operator_failure_modes.py \
  --probe-hidden-intervention

$PY -m pytest budding_yeast_v2/tests/test_marker_closure.py \
  budding_yeast_v2/tests/test_standard_baselines.py::test_selected_directory_and_registry_have_no_modified_baseline_aliases -q
```

正式消融训练示例（命令尚未执行）：

```bash
CUDA_VISIBLE_DEVICES=5 $PY budding_yeast_v2/train.py \
  --model biom_refiner_marker_closure --loss_type hybrid_mse_rel \
  --dataset_name /home/users/zsy/mydata/budding_yeast/mutant_aware_pure_random_100000_210m_500s.npz \
  --width 192 --modes 24 --n_blocks 4 --batch_size 256 \
  --max_epochs 100 --lr 8.8e-4 --warmup_epochs 8 \
  --norm_type log1p --al_strategy none --seed 42 --no_wandb \
  --exp_root /home/users/zsy/mydata/budding_yeast/experiments/operator_design_ablation_2026-10-30
```

完整 ICHFO 的训练命令需等局部流/事件数据与程序输入实现后提供；不能将这个标记读出模型的命令冒充完整方案。
