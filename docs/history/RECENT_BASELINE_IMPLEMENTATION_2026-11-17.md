# 近期baseline实现：范围、边界及验证

本次用户授权开始实现；明确要求生成方法只能调用现有生物/突变规则，不能更改它们。本轮没有启动正式多seed或大规模数据生成。

## 1. 接入内容

|入口/名称|内容|复现边界|
|---|---|---|
|`run --method physics_pre_topk`或`physics_pre_power`|已有Physics-PRE，新增输入/物理时间检查及完整中心差分模板mask|连续ODE残差任务适配；事件重置/初值评分没有偷偷加进baseline|
|`generate --method oed_temporal`|Xu2026 Eq.5共轭Bayes更新、Eq.8 UCB、逐个真实ODE反馈|初始集与现有方法共用；高维输入的有界投影特征字典；ODE时间方差/频谱信息量。没有把轨迹粗糙度叫作生物能量耗散。不是原论文PDE实验逐项复现|
|`generate --method ogas_round_projected`|OGAS条件DDPM、CFG、FIFO反馈、EMA难度、密度比修正、偏重高难度的目标分布、探索混合|明确是轮次式代理模型训练＋合法提案投影版；不是Melissa异步连续在线原版|
|`generate --method random_generate`|同生物提案器、同生成/训练流程的随机查询|用于区分生成策略与流程差异|

不能仅因“核心代码已实现”就宣称三个近期方法均已完成大规模复现，或将轮次投影版写成原版OGAS。OGAS原生连续在线训练、HGGS和PEAKS回归版仍是后续工作。本次没有通过改变原生物参数支持、添加mutation或扩大范围来弥补适配难点。

## 2. 规则冻结

`generation/domain.py`只包装已有`data.systems.budding.offline_pool.Proposals`与`simulate`：

- 模块/模式、强度分布、环境、参数映射、初值和real碰撞救援，全部由原提案器完成。
- 仿真直接复用当前严格BDF容差、事件处理、已知总量约束重算及超时政策；不增加变量上限、不改变负值接受政策。
- OED对新合法提案打分；OGAS生成的是“期望输入特征”，随后取新合法提案库中的最近唯一候选。提交ODE的是提案器原样产生的完整对象，不能将扩散生成的任意141维向量直接当作物理参数。
- 投影/特征标准化是算法内部几何处理，不裁剪实际ODE参数、初值或真轨迹。
- real沿用原碰撞容差和救援；两池合成validation/test的输入签名也被排除。算法不读取real标签、未查询真值或未查询真实事件时间。
- 运行前记录生物源文件SHA，生成中继续核查；源代码变化会拒绝继续，而不是静默换规则。

本轮另对82个既有Budding及data根目录Python源文件保存实现前SHA，结束时逐字节核对；没有编辑这些文件。新增代码均在utils采样/实验工具层，原始生成文件保留。

## 3. 文件边界

```text
utils/sampling/baseline/
  physics_pre.py          连续残差与事件mask
  oed.py                  Bayes后验、UCB、信息量
  ogas.py                 条件扩散、CFG、密度比及反馈buffer
utils/sampling/generation/
  cli.py                  参数检查与执行入口
  domain.py               冻结生物提案/仿真适配及合法投影
  policies.py             采集决策和已查询反馈
  storage.py              不可变新增块＋初始源引用
  cohorts.py              共用holdout、real与公共评测
  runner.py               轮次编排、补采、恢复和成本账本
```

公共`fit_round`、YeastLitModule、归一化、checkpoint、CSV、预测图和loss图继续复用。没有模型专用训练器，没有import/exec/运行时读取官方参考仓库。

## 4. 公式和算法设置

OED支持noise/prior variance、投影维数与exploration设置。默认从初始已查询训练数据拟合输入几何统计及通道尺度，然后冻结。信息量为按初始通道尺度缩放的时间方差`log1p`与高频能量比例之和；Bayes后验仅由初始及后来真实查询的合成轨迹更新。默认一次查询1条，先观察再更新；`--query-batch-size>1`明确属于批量更新适配。

OGAS包含：

- 平衡分类器估计`q_reference(x)/p_buffer(x)`，方向为`D/(1-D)`，权重限制在[0.1,10]，用于生成器目标，不截断FNO Relative loss。
- 条件噪声预测和CFG；条件dropout=0.1，guidance=5；FiLM调制的两个残差MLP块、可学习null条件、sinusoidal时间编码；DDPM线性噪声日程。
- 默认width128、diffusion1000步、采样最多50步、每周期50更新；与论文主2D配置width1024不同，是显式的任务/资源配置，不能隐藏。
- 目标难度范围[0,5]的密度正比于难度，即`5*sqrt(U)`；10%探索来自原合法提案分布。
- 本课题混合离散/连续域，以原合法提案分布替代连续box uniform参考，并投影回合法提案。该变化已写入采集诊断和结果身份。
- 代理模型仍按共同冷启动轮次训练；生成器跨采集周期保留状态。新增样本的难度由查询前的模型与返回的合成真值计算，测试/real不进入buffer。

Physics-PRE的mask覆盖预测事件邻域，且中心差分三个点都必须在连续区间。正常跳跃不应因差分尖峰获高残差；连续残差也可能漏掉错误的跳跃幅度/错误初值，不能当作完整轨迹正确性的证书。

## 5. 数据、资源和恢复

- 初始训练集引用冻结协议的原始行，完全不复制其轨迹。
- 新轨迹集中成最多约128个成功样本的不可变NPZ块；不是每个OED顺序查询保存一份累计数据。
- 每轮dataset.json引用源块及行ID；初始数据若有数值QC覆盖，也保留其逐字段覆盖来源，不回退旧轨迹。
- 保留参数、实际初值、mutation/overrides、L及各类标签、轨迹和时间轴等原schema字段。
- 每次ODE开始和返回都有账本，内部严格容差重试也计费。失败后继续补到本轮目标，达到尝试上限明确报错，不输出假完成。
- 状态按批次原子提交，生成器只留最近两个恢复点。异常丢失的未提交查询可能重新计算，但开始/完成账本仍保留这些成本；有未完成查询时`solver_attempts_exact=false`。
- 进程池使用spawn、有界提交、35秒父级无进展保护和Linux父进程死亡信号；不会递归启动探针。默认4个ODE worker，加载和评分使用公共资源槽。
- formal必须冻结real参考，使用同池validation选best，逐轮train-only refit；W&B只允许offline。
- 当前入口是单项`generate`，不自动塞入现有16并发队列。大规模运行前仍需单项资源测量，不能把此前固定池的16并发数据直接当成生成流程资源验证。
- 完成缓存核对源身份、输入/切分与保存的模型、history、数据块等SHA；不能仅凭completion.json跳过损坏文件。

## 6. 调用示例

先只检查，不训练、不创建输出：

```bash
bash run_sampling_budding.sh generate \
  --protocol /home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_a866aef5d988 \
  --pool mutation --method oed_temporal --dry-run
```

单项正式调用（示例，本轮没有执行）：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_sampling_budding.sh generate \
  --protocol /home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_a866aef5d988 \
  --pool mutation --method ogas_round_projected --device cuda:0 \
  --workers 4 --query-batch-size 32 --plots --wandb
```

输出使用独立`experiments/budding_generation/<protocol>_gen<hash>/<pool>/<method>/seed_42/`，不覆盖已完成固定池结果。目录包含train.log、config、rounds、eval、loss/performance，data只存新增源块及引用。

LHS生成需提供与原池一致的已审核混合边际配置；其文件SHA必须与原池`generation_config.lhs_spec_sha256`完全一致，catalog和range指纹也必须相符。不接受通过算法config夹带生物参数override。`--oed-config/--ogas-config`只接收各算法dataclass的已知字段。

## 7. 验证范围

测试文件`tests/test_recent_sampling_baselines.py`：Bayes闭式解对账、CFG/null条件、密度比方向、DDPM训练与RNG恢复、合法提案一致性/holdout排除、投影唯一性、跳跃mask及其盲区、不可变源引用和数值QC覆盖、失败补采与完整成本账本，以及三个生成方法的实际canonical ODE→小FNO训练→评测→恢复CPU闭环。

旧固定池Physics-PRE的两种选样与既有多轮冷启动路径也纳入回归。测试不证明大规模收敛、优于baseline或原版OGAS异步吞吐；没有读取real来挑算法超参数。最终测试数量与源文件SHA核对见本节后续完成记录。


### 完成核查记录

2026-11-17：18项新增测试通过（49.95秒）；此前与原Physics-PRE和多轮路径的定向回归23项通过（94.40秒）。默认1000步噪声日程、50步CFG采样的补充CPU测试通过。编译、bash语法、git diff --check通过，生成CLI两方法dry-run未创建正式输出。82个冻结文件变化数0；两池生产catalog/range与LHS原配置SHA核对通过。日志位于scratch/probes/baseline_implementation_20261117/，测试输出为pytest临时目录，不是正式实验成绩。


### 后续更新：在线接入与原文版本核对

此前“OGAS异步尚未接入”仅描述当时round版。现已新增`ogas_async_projected`/`random_async`，完成持续代理/后台生成器/异步ODE/replay及恢复；新在线依据11月12日v2数值配置，旧round版明确保留v1。33项新增及近期CPU测试通过，含实际ODE和FNO、训练周期恢复的matching-best检查；82个生物及原始源文件SHA无变化。仍有合法投影、完整轨迹、CPU生成器等显式适配，非Melissa完整复现，未开展正式实验。详细[对应关系与限制](BASELINE_FIDELITY_AND_OGAS_ONLINE_2026-11-17.md)。
