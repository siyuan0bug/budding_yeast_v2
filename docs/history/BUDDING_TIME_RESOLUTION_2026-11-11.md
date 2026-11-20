# Budding 时间采样分辨率小实验（2026-11-11）

## 代码结论

`data/systems/budding/simulation.py::simulate_mutant(tf=210, ns=500)` 的 `ns` 是输出点数。
最后执行 `interp1d(tr, yr)(np.linspace(0, tf, ns))`，包含首尾，因此 500 点为
499 个间隔，`dt=210/499=0.4208416834 min`。不是每分钟一点，也不是积分器只走500步。
`specification.py`、`adapter.py`、`utils/active_learning/oracle.py` 还固定使用500点。

积分器是自适应 BDF，`rtol=1e-4, atol=1e-6`，四个终止event各自进行零点定位，
触发后重置部分状态并重新启动积分。`ns` 只在完成积分后参与插值，降低它不会减少积分器工作。
WT实测128/256/500点均为696个内部步、2151次右端函数调用，8个event时间完全一致。
SciPy关于输出时刻、自适应误差控制、event根定位的定义见
[solve_ivp官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)。

当前event拼接有一个独立精度问题：保留旧段的跳变前端点，再追加新段的 `sol.t[1:]` / `sol.y[:,1:]`，
跳变后的初始点被跳过。全局线性插值会在event与下一内部步之间产生一段斜坡。
它的宽度由BDF重启后的第一个内部步决定，不是由输出 `dt` 决定；不能把整段采样误差都归因于此。
后续修正宜采用分段dense output，明确event时刻取右极限；不应把重复时间戳直接交给全局线性插值。
此次只审计，没有修改生产求解器、标签、采样规则、默认网格或已有实验。

## 实验协议与复现

脚本：`scripts/analysis/benchmark_budding_time_resolution.py`。
原始JSON和日志：`scripts/analysis/outputs/time_resolution_2026-11-11/`。

```bash
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2
export PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/users/zsy/.conda/envs/neuralop/bin/python \
  scripts/analysis/benchmark_budding_time_resolution.py \
  --output scripts/analysis/outputs/time_resolution_2026-11-11 --device cuda:2
OPENBLAS_NUM_THREADS=1 /home/users/zsy/.conda/envs/neuralop/bin/python \
  scripts/analysis/benchmark_budding_time_resolution.py --event-only \
  --output scripts/analysis/outputs/time_resolution_2026-11-11
```

- 轨迹重构：全部126个真实突变体配置，加128条局部合成参数轨迹。后者独立扰动
  `mdt,ksn2pp,ksb2pp,ksb5pp,ksc1pp,ks20pp,kscdh,kdb2pp`，倍率在0.7–1.3内对数均匀分布，seed42。
  这是局部诊断分布，不代表百万AL池或完整KO/GAL分布。
- 参考：同一次生产容差的求解轨迹，在0.01min、21001点网格评估。
  各候选网格直接从原始内部步插值取值，再线性重构到参考网格。
  这测量采样/重构误差，不是相对解析解或更严格容差真解的ODE误差，也不是模型预测误差。
- 全局误差：`||log1p(max(y_recon,0))-log1p(max(y_ref,0))||₂ / ||log1p(max(y_ref,0))||₂`，
  逐轨迹计算再汇总；另报告CORE9同口径误差。
- event误差：任一event前后1min的时间点，按每轨迹每变量
  `max(轨迹峰峰值,0.1)` 归一化后计算RMSE，39变量等权；平滑区为其余点。
  该指标仍会受变量平均影响，不能作为最坏单通道跳变误差保证。
- event独立诊断：22个显式配置中的相邻事件是否落入同一采样区间；另检查短暂LTE1高平台是否完全漏采。
  同一区间表示输出的时间分辨率受限，不能解释成求解器漏event，也不能解释成模型必然分类错误。
- FNO训练计时：项目现有标准FNO，width192/modes24/4blocks/padding8，batch128，
  BF16混合精度（FFT仍float32），Adam，5次预热后20次前向/反向/更新的中位耗时及峰值allocated显存。
  GPU为RTX A6000，使用当时空闲GPU2；不含数据加载、验证、AL选样/ODE和checkpoint成本。
- 额外小规模学习实验：128条局部合成轨迹，前75%训练、后25%测试，seed42，
  width64/modes24/4blocks的FNO，FP32，Adam/lr1e-3，80epoch，batch32，比较128/256/500点。
  参数和逐通道log轨迹Z-score仅由训练子集拟合；测试输出插值到同一0.01min网格后比较。
  单seed、局部分布、小模型，仅用于初筛，不足以改变正式AL设置。

## 迁移约束

FNO的FFT/逐点卷积激活量随点数增加，训练通常会变慢；真实速度还取决于频率截断、padding、
FFT长度因子分解、批量大小、显存和GPU利用率。不能把500→256等同于严格1.95倍提速。
普通全注意力模型的注意力矩阵对时间长度呈二次增长；其他模型要按各自实现评估。

正式降采样不是只改 `simulate_mutant(ns=...)`：还需统一SPEC、静态/real导出、AL oracle形状检查与入库、
时间轴、模型长度和checkpoint契约。`yeast_datamodule.py`会从数据读取长度，但不能据此认为整个AL流程已支持变长。
`phenotypes.py::analyze_mass_pattern` 的 `dd-ld>10` 是按采样索引设置的去重间隔，
改变网格会改变它对应的物理时间，标签必须保留原判定或改为按event/物理时间计算后再验证。
归一化统计和评测时间轴也需要一致，旧checkpoint不能直接冒用新预处理。

均匀网格仍适合当前FNO输入；不要直接把各样本不同的event自适应非均匀网格送入其FFT。
可以保留500点主数据，在训练入口统一下采样，并在统一细网格评测；但这仍需要AL/评测适配。
更强调event的后续方案是额外记录event时间及重置前后状态，采用辅助监督或分段模型。

## 实测结果与建议

![采样误差与FNO开销](../../scripts/analysis/outputs/time_resolution_2026-11-11/tradeoff.png)

254/254条轨迹求解成功，无排除样本。内部积分步数均值540.5、最大819。

| 点数 | dt/min | 全局log相对L2均值 | event邻域NRMSE均值 | 平滑区NRMSE均值 | FNO更新/ms | 峰值allocated/MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 3.3333 | 7.36% | 12.61% | 1.696% | 17.93 | 228 |
| 128 | 1.6535 | 5.13% | 9.42% | 0.479% | 14.72 | 335 |
| 211 | 1.0000 | 4.03% | 7.29% | 0.162% | 16.01 | 503 |
| 256 | 0.8235 | 3.68% | 6.70% | 0.115% | 15.71 | 530 |
| 500 | 0.4208 | 2.58% | 4.76% | 0.034% | 24.43 | 914 |
| 1000 | 0.2102 | 1.83% | 3.29% | 0.010% | 34.88 | 1700 |

所有误差都是相对细网格的重构指标；表中不是模型预测误差。event邻域与平滑区的缩放口径见协议，不能直接与log相对L2互换。

单seed小模型实际训练（96训练、32测试、80epoch）在相同细网格上的测试log相对L2均值：
- 128点：11.98%，P95=15.09%。
- 256点：13.41%，P95=17.68%。
- 500点：12.92%，P95=15.91%。

该pilot没有显示点数与泛化误差的单调关系，学习/优化误差明显超过重构误差。不能将128点该次较好解释为普遍更优。首个pilot耗时还包含冷启动，不使用pilot总耗时比较分辨率速度，采用上表预热后的width192更新计时。

500→256时，单步速度约1.55倍，时间减少35.7%，峰值allocated显存减少42.0%，轨迹本体存储减少48.8%。端到端AL加速会更小，因为ODE、采集和其他固定成本不随输出长度同比下降。

22配置event诊断：全部成功，26个完整LTE1高平台在各网格均至少命中两个点；未发现该类平台完全漏采。99对相邻event中，64/128/211/256/500/1000点分别有20/10/6/10/1/0对落在同一区间。最小event间隔0.3113min。211比256的此项表现更好是网格对齐造成，不代表较粗采样普遍更准。

原生产插值相对同容差分段dense output的全局log相对L2均值为0.0695%，P95=0.3659%。这同时包含普通段内线性插值误差，不能完全归为event拼接误差。event后首步宽度均值0.0004911min，最大0.0054680min；该窄斜坡远小于0.421min，不是本次降采样误差的主要来源。0.01min评估网格也可能漏掉这类极窄斜坡，不作为其最坏局部幅度保证。

**此前按event时间精度优先给出的建议（用户目标现已澄清，见下文修订）：正式实验500点，快速筛选256点。此次不修改默认值。** 若明确接受全局重构均值约4%以内、event邻域均值约7%以内，256点在这组样本上满足该示例预算；若要求全局均值3%以内且event邻域均值5%以内，500点是此次候选中最少满足者。这些预算是解释选择的示例，不是通用生物学阈值或逐株最坏误差保证。

不建议未经完整验证就将128点作为统一正式网格，也没有本次证据支持全面升级到1000点。若需要event时间精度达到0.1min，应增加明确的event时间/左右状态监督或采用分段表示；均匀采样中点估计的最坏量化半宽为dt/2，500点约0.2104min，不能保证0.1min。

正式迁移前建议在同一合成train/val与126配置开发评测上，对256和500做同数据预算、同更新次数、至少3seed的正式容量模型比较，同时检查event时间、MASS/ORI/SPN等跳变通道与平滑通道误差，再决定是否降低全流程采样率。126配置已用于本次分辨率诊断，应披露其开发使用，不能当作未接触盲测。

## 用户目标澄清后的判断：以可见波形和实际预测效果为主

用户关心0–210min曲线的直观变化与训练实验结果，不要求区分图上不可见的短间隔event。
因此，不能仅凭相邻event落在同一采样区间就认定降采样对任务有影响；此前用它支持统一保留500点，证据不足。

补充绘图使用同一次内部积分结果，直接采128/256/500均匀输出点，物理量坐标、不做平滑。
检查WT核心9通道，并比较#1/#36/#80/#104的MASS、CLB2、SPN。
在这些示例的210min全图中，256与500点的整体波形、周期与幅值基本重合，128点整体也接近。
将WT首次分裂放大到6min窗口后，MASS/ORI/SPN的跳变斜坡宽度和位置可分辨，128点最明显，256与500也有局部差异。
这说明差异主要集中于很短的跳变邻域，不能将较大的event局部误差直接描述成整条曲线明显失真。
仅检查四个配置的图，不构成全部突变体或训练池的视觉等价保证。

按这一目标，**256点、dt≈0.824min应作为效率与可见波形保真的优先候选**；不再仅因event同区间统计坚持500点。
训练单步减少约36%、峰值allocated显存减少约42%沿用前述测量。
小模型pilot中256与500点测试误差为13.41%与12.92%，差0.49个百分点，但单seed不足以证明等价或显著退化。
图像接近并不自动保证训练指标不变，正式采用仍应在相同细网格评测实际预测；本次没有新增大规模重训，也未改变生产默认。

绘图脚本：`scripts/analysis/plot_budding_time_resolution.py`，复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle \
/home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/plot_budding_time_resolution.py \
  --output scripts/analysis/outputs/time_resolution_2026-11-11
```

![WT全程简图](../../scripts/analysis/outputs/time_resolution_2026-11-11/wt_overview.png)

[WT核心9变量](../../scripts/analysis/outputs/time_resolution_2026-11-11/wt_full.png)；
[四配置全程叠加](../../scripts/analysis/outputs/time_resolution_2026-11-11/mutants_full.png)；
[WT跳变局部放大](../../scripts/analysis/outputs/time_resolution_2026-11-11/wt_event_zoom.png)。

## 300–400点补测（同日，按用户希望增加点数）

复用原先128条合成轨迹及96/32切分，seed42，重新运行小模型80epoch；标准FNO计时仍为width192、batch128、A6000 GPU2，BF16混合精度，5次预热后20次更新中位数。表内全部为同轮计时，不把前轮500点24.43ms与本轮300点混算；本轮500点为28.98ms，体现运行时波动。

| 点数 | 间隔/min | 更新/ms | 相对同轮500点耗时减少 | allocated峰值/MiB |
|---:|---:|---:|---:|---:|
| 256 | 0.8235 | 17.87 | 38.3% | 529 |
| 300 | 0.7023 | 15.75 | 45.7% | 601 |
| 350 | 0.6017 | 22.28 | 23.1% | 678 |
| 384 | 0.5483 | 19.23 | 33.7% | 729 |
| 400 | 0.5263 | 21.18 | 26.9% | 754 |
| 500 | 0.4208 | 28.98 | 0.0% | 911 |

小模型pilot实际测试log相对L2均值：
- 300点：13.05%。
- 350点：13.20%。
- 400点：13.58%。
- 500点：12.92%。

以上单seed局部pilot不能推断300/350/400点与500点等价或谁的正式训练精度更好。384仅补测计时，未增加384点学习pilot或曲线图。

绘图继续使用WT与#36/#80/#104内部积分轨迹，直接取500/400/350/300点。四种网格在这些示例全程图中的主要波形基本重合；WT六分钟局部图中差异集中于跳变边缘。局部相位对齐可能让某一次较粗网格看起来更好，不说明其整体更准。

**根据用户认为256偏少的偏好，更新建议为400点，dt=210/399≈0.5263min。** 300点也是有效效率候选（dt≈0.7023min）；400保留原500点的80%输出点，间隔更细，本轮训练单步约减少27%，峰值allocated显存约减少17%。350并未比400更快，不因数字居中而优先选择；384本轮也有较好计时，作为备选保留。未修改生产网格或AL流程。

复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle \
/home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/benchmark_budding_time_resolution.py \
  --benchmark-only --pilot-lengths 300 350 400 500 --lengths 256 300 350 384 400 500 \
  --report-name fno_intermediate.json --output scripts/analysis/outputs/time_resolution_2026-11-11
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle \
/home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/plot_budding_time_resolution.py \
  --lengths 500 400 350 300 --output scripts/analysis/outputs/time_resolution_2026-11-11/intermediate
```

![300至500点WT全程曲线](../../scripts/analysis/outputs/time_resolution_2026-11-11/intermediate/wt_overview.png)

[跳变局部放大图](../../scripts/analysis/outputs/time_resolution_2026-11-11/intermediate/wt_event_zoom.png)；[四配置全程图](../../scripts/analysis/outputs/time_resolution_2026-11-11/intermediate/mutants_full.png)。

## 全部real-mutant配置的后续验证

已完成全部126配置的300/350/400/500点逐株审计，并输出126页CORE9图。400点全变量log重构误差均值2.225%，500点1.994%；核心通道最高峰损失最坏为幅度0.612%，六个平滑核心通道显著峰数无改变。最敏感差异集中在#118/#110等初始亚分钟快速变化和分裂跳变邻域。[详细协议、逐株结果和全部图像](BUDDING_ALL_REAL_TIME_RESOLUTION_2026-11-11.md)。
