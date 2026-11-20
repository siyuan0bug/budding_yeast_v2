# 新baseline与原文的对应关系，以及OGAS在线接入

> **2026-11-18后续修复**：CFG参数约定及v2密度上限已对照作者源码修正，来源标记贯穿查询、训练反馈和恢复。下文为接入时说明，最新行为见[修复记录](OGAS_ALIGNMENT_FIX_2026-11-18.md)。旧结果不会自动变为修复后结果。

方法出处补充更正：Uncertainty沿用QBC（Seung等，COLT1992），AL4PDE2025是当前评分实现的参考框架；HGGS正式发表于AAAI2025，本地文件为2024预印本。新增五类方法的逐项差异和四项待完成原因见[出处与复现程度核查](BASELINE_ATTRIBUTION_AND_FIDELITY_2026-11-17.md)。

用户要求：生成式方法可以查询旧池之外的新输入，但只能复用现有生物/突变规则、初值和ODE，不能修改它们。作者代码仅用于开发核对，不在实验运行时import、exec或读取执行。

## 1. 先明确复现程度

|项目入口|保留的论文核心|本课题的适配|应如何命名|
|---|---|---|---|
|`physics_pre_topk` / `physics_pre_power`|预测轨迹的平均绝对方程残差；最近已查询真轨迹的残差归一化；TopK/Power选样|PDE换为既有39维ODE；参数距离使用训练Z-score；中心差分避开预测的分裂/重置事件|Physics-PRE，事件ODE适配|
|`oed_temporal`|线性高斯信息量模型、共轭后验更新、UCB及递减探索系数、查询后更新|共用随机初始集；高维投影特征；时间方差/频谱信息量；合法新提案库中近似最大化UCB|OED-Temporal，任务适配|
|`ogas_round_projected`|v1条件DDPM、CFG、EMA、FIFO难度反馈、密度比、偏向高难度条件、探索|分轮冷启动；使用已查询轨迹的任务Relative反馈；合法域投影|OGAS-Round-Projected，不能称原版在线OGAS|
|`ogas_async_projected`（本次新增）|持续训练反馈、后台生成器、异步ODE、有限buffer、条件扩散及v2密度比稳定化|完整轨迹替代自回归时间片；合法提案分布替代连续box；投影；CPU生成器；初始统计冻结、项目验证best及固定收尾训练|OGAS-Online-Projected，在线任务适配|
|`random_generate` / `random_async`|合法提案随机查询|分别复用分轮/在线流程|生成流程对照，不是新增论文方法|

**不能把这些适配都写成“作者实验完整复现”。** 核心公式对账和代码能运行，只证明实现的对应关系；尚未证明大规模效果、与作者同吞吐、或在此生物混合域具有原文全部理论性质。

## 2. Physics-PRE如何做

论文为Polanska等《Data-Efficient Neural Operator Training via Physics-Based Active Learning》，[arXiv:2605.21348](https://arxiv.org/abs/2605.21348)，ICLR 2026 AI/PDE Workshop。

每个候选只有参数和实际初值进入打分器，不揭示未查询轨迹：

1. 当前FNO预测完整轨迹，逆变换到物理空间。
2. 计算`r(t)=dŷ/dt−f(ŷ,p)`。`f`来自现有规范ODE的公共向量化实现，不改任何反应项。
3. 对39个变量和可计算时间点取平均绝对残差。
4. 找参数空间中最近的一条已查询训练输入，以其真实轨迹的离散残差作归一化参考；参考残差设置数值下限`1e-6`。
5. 根据分数TopK，或按原Power规则抽样。

这里的事件处理是必要适配：质量在细胞分裂时瞬间下降，直接用跨跳跃的差分会产生很大伪残差。因此只根据**候选预测自身**识别阈值跨越/重置，屏蔽附近±4步，而且中心差分模板的三个点都必须可用。不读取未查询候选的真实事件时刻。

局限：它测量连续段方程残差，不能保证初值和跳跃幅度正确。例如错误重置后恰好仍满足连续方程，可能残差很小；本次没有偷偷加入新的事件损失或初值项冒充作者方法。

实现：`utils/sampling/baseline/physics_pre.py`。

## 3. OED如何做

论文为Xu等《Data generation with optimal experimental design for operator learning》，CMAME 450 (2026) 118675，[DOI](https://doi.org/10.1016/j.cma.2025.118675)。本地PDF已逐项核对第3节、Algorithm 1；未找到可直接对账的完整作者生成代码，属于依据论文独立实现。

它学习的是“输入的轨迹信息量”，不是FNO的不确定性：

```text
已查询的合法输入及轨迹
→ 计算轨迹信息量Y
→ 更新 输入特征→信息量 的Bayes模型
→ 产生一批新的合法输入，计算UCB
→ 查询最高分输入的ODE真轨迹
→ 更新Bayes模型，继续
```

信息量回归模型为`Y=φ(x)ᵀβ+noise`，采用论文Eq.5的高斯共轭更新。默认逐条更新，分数为论文Eq.8/16的形式：

`φ(x)ᵀμ + λ₀ sqrt(log(k)/k) sqrt(φ(x)ᵀΣφ(x))`。

当前默认投影16维，特征字典97维，包括常数、1/2/3次项、sin/cos和Gaussian项。相比原文更全面的字典，省略了多频率/对数等分量；有界投影是本课题高维几何的适配，不能称作者原字典。

原论文有解的方差、高频含量和能量耗散等物理信息量。这里没有擅自定义“酵母能量耗散”，而是：

- 用初始训练轨迹拟合的各变量尺度作缩放，随后冻结；
- 时间方差取`log1p`；
- rFFT高频功率占比按变量平均；
- 默认两项权重均为1。

因此`oed_temporal`是明确的信息量适配，尤其不能把这个新的信息量公式说成原论文原式。原文LHS初始化也被共用的随机初始集替代，以便本项目控制初始数据差异。每次在新合法提案库中寻找UCB最大者，是对连续域优化的离散近似。

实现：`utils/sampling/baseline/oed.py`，流程由`generation/policies.py`和`runner.py`调用。

## 4. OGAS是不是生成池外样本

**是。** [OGAS原文](https://arxiv.org/abs/2606.09949)属于query-synthesis：生成新的求解器输入，再调用求解器取得监督数据。它不是要求先有一个完整固定池，再从中挑选。

本项目在线版只从现有100k池引用共用初始训练集和固定验证/测试集，之后调用现有提案器产生新的合法输入。不能保证某个离散输入在整个旧训练侧池中从未出现；但不从旧池读取它的标签，仍实际调用ODE并计费。已查询输入、real碰撞和合成holdout按现有规则处理。

### 4.1 一条在线链路

```text
FNO训练一个已查询批次
   ├─ 原训练Relative L2 → 更新FNO
   └─ 同次前向预测 → 每条样本难度
                  ↓
        (参数＋实际初值，难度) FIFO反馈
                  ↓
       后台CPU线程训练条件DDPM和密度比分类器
                  ↓
       要求生成“当前比较难”的输入坐标
                  ↓
       映射到新抽出的合法突变/参数提案
                  ↓
       CPU ODE worker异步仿真
                  ↓
       有效轨迹进入replay，FNO继续训练
```

难度来自**训练中的同次前向**，不是real/test误差，也不是读遍未查询真值。默认难度采用论文Eq.4的完整轨迹适配：归一化空间的逐样本MSE除以该样本目标轨迹L2范数加`1e-8`。这只是采样器的信号；FNO训练损失仍是现有的逐样本/逐变量Relative L2，epsilon及无截断规则没有变化。也可显式配置`signal=relative_l2`，该分支直接使用FNO criterion的epsilon。

### 4.2 为什么需要合法域投影

论文主要实验是连续PDE参数域；本项目有精确KO、整数MC、合法复合模式、参数耦合和初值依赖。Gaussian DDPM输出的任意180维期望坐标不一定代表合法突变，不能直接交给ODE。

所以生成器只提供“期望方向”，程序另从现有`offline_pool.Proposals`抽出新合法提案，在算法标准化空间找到最近且未重复的合法点；最终提交的是**提案器原样输出的对象**。没有裁成某个新参数范围，没有单独改初值，没有新增mutation。

论文v2附录B.2也指出混合/离散空间可采用混合类型生成模型，但**本文作者没有把这里的最近合法点投影作为原方法**。本项目因此必须保留Projected后缀。基础强度分布、合法范围及参数映射不改；AL最终选中哪些强度/模块的频率当然会改变，否则无法自适应。

### 4.3 原文版本差异不能忽略

本地原PDF是2026年6月v1；作者2026年11月12日的[v2 PDF](https://arxiv.org/pdf/2606.09949v2)更改了若干实现细节。本次额外下载核对，证据保存在`scratch/probes/baseline_implementation_20261117/ogas_v2.txt`。

|设置|此前分轮版默认，明确保留v1|新在线版默认，依据v2|
|---|---|---|
|噪声步数|1000|100|
|采样步数|50|100，完整日程|
|生成器Adam学习率|1e-4|2e-4|
|判别器|Adam、普通BCE|AdamW lr1e-4/wd1e-2，BCE＋0.03×logit平方|
|密度比稳定化|`D/(1-D)`限制[0.1,10]|log-ratio限制[-5,5]，平衡先验上界2，再按批次归一到均值1|
|生成比例|默认90%，属于已有配置|前期线性增加至70%；余下30%来自合法参考分布|
|难度条件|EMA标准化，目标`5×sqrt(U)`|相同|
|CFG|dropout .1，guidance5，可学习null条件|相同|

在线混合warmup默认100个主模型更新，是项目资源配置，不是声称原文统一规定100。新在线生成器width仍为128，论文Table 1是1024；两个FiLM残差块等结构保留。CPU后台执行是资源适配，不能宣称拥有作者辅助GPU上的开销/吞吐成绩。

密度比是在纠正“过去过度采样区域”对生成器的偏置，不是在截断FNO训练loss。参考分布从论文连续box uniform改为本项目合法提案分布，其比值目标也随之改变；原文基于连续uniform的性质不能不加说明地照搬。

### 4.4 已接入的在线机制和仍然不同的地方

已经接入：

- 独立CPU生成器线程及有界反馈/目标缓存；常规FNO批次不等待DDPM周期完成。
- spawn ODE进程与训练重叠，按已完成future接收结果，不被最早提交的慢任务阻塞。
- 同一代理模型/优化器持续训练，不再按五轮反复冷启动；FIFO replay随机取完整轨迹。
- 查询失败补采至固定有效样本目标；开始、返回、内部严格重算和未完成查询分别记账。
- sampler未及时产生目标时的随机参考回退单独计数；反馈队列过期淘汰也计数，不宣称100%生成器引导。
- checkpoint时同步有限队列，保存源块引用、replay RNG/曝光次数、生成器/判别器/优化器、主训练恢复状态。恢复时还保存/还原与该恢复点匹配的best，避免误用中断后期的未提交best。

与作者实验仍不同：

- 本课题是完整210分钟轨迹回归；不是作者的PDE自回归时间片流式训练。
- 在线全过程用共用初始训练集拟合一次统计并冻结，避免持续更新统计后模型坐标突变。**原固定池五轮实验仍逐轮refit，完全不改。**
- 为复用本项目模型选择标准，继续以独立同池归一化validation loss选best；另保存last，并记录best实际见过多少唯一训练输入。
- 默认总200epoch、前120epoch在线补采、后80epoch收尾；每epoch默认`ceil(最终样本数/batch)`个更新。到预设边界会补齐尚缺查询，再继续训练。这是明确的固定预算实验适配，尚未复现Melissa的1.8 batches/trajectory双向速度限制器。
- 保留新轨迹的不可变落盘块，以便复现和后续分析；作者完全内存流式流程的无磁盘I/O优势不能照搬。
- 只接入OGAS-Loss方向，未把OGAS-Unc也宣称完成。

因此当前已是可执行的异步在线**任务适配版**，不是作者Melissa/PDE平台的逐项完整复现。固定池baseline与在线方法的训练安排不同，先与`random_async`同流程对照，分别报告全部ODE尝试、成功标签、训练步数、唯一训练曝光和墙钟时间。

## 5. 文件及入口

```text
utils/sampling/baseline/ogas.py      条件DDPM及v1/v2密度比
utils/sampling/generation/
  background.py                    后台生成器线程、有限队列、异常传递
  async_queries.py                 ODE异步future、超时、成本账本
  replay.py                        原始完整轨迹的有界训练视图
  online.py                        批次反馈和在线采集回调
  online_runner.py                 共用训练/评测服务的独立流程
```

没有新增FNO专用模型，也没有复制另一套optimizer/loss/eval。`benchmark/rounds.py`仅增加默认不启用的数据适配hook，旧固定池调用行为不变。

只检查入口，不启动训练：

```bash
bash run_sampling_budding.sh generate \
  --protocol /home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_a866aef5d988 \
  --pool mutation --method ogas_async_projected --dry-run
```

单项正式运行示例（本次未执行）：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
CUDA_VISIBLE_DEVICES=0 bash run_sampling_budding.sh generate \
  --protocol /home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_a866aef5d988 \
  --pool mutation --method ogas_async_projected \
  --device cuda:0 --workers 4 --bank-size 512 --plots --wandb
```

`--online-config`仅控制在线epoch/replay/更新频率等；`--ogas-config`仅控制生成器配置。LHS仍需原审核配置的精确SHA。当前正式适配验证对象为FNO＋rel_l2，未知组合拒绝运行。

新输出：

```text
experiments/budding_generation/<protocol>_online<hash>/<pool>/<method>/seed_42/
  config.json, train.log, performance.csv, loss.png
  stream/
    model.ckpt, best_exposure.json, history.csv, loss.png
    initial_dataset.json            仅用于初始统计拟合的源引用
    dataset.json                    最终全部已接受样本的源引用
    eval/                           公共指标、逐样本CSV、real图片
  data/chunks/                      新轨迹只存一次
  eval/                             公共工具的最终结果汇总
  _state/                           恢复状态、last、ODE/生成器账本、offline W&B
```

不改原实验目录，不复制初始/每轮累计轨迹，不启动新的大批并发。常规已有资源槽继续使用；有界queue不等于已证明生产内存峰值，不能直接套用固定池16并发结论。先做单项生产规模资源验证，再安排正式比较。

## 6. 验证范围

CPU检查包括公式、v2权重与优化器、后台非阻塞及恢复、异常传播、队列上限/完成乱序、replay容量/精度/RNG、真实canonical ODE→FNO训练→评测、评测中断恢复、训练周期checkpoint恢复及损坏缓存拒绝。测试使用很小的诊断模型/样本，不代表正式收敛或方法优劣。

完成结果：近期及新增在线测试33项通过（122.84秒）；原固定池分轮/恢复与事件mask定向回归4项通过（40.79秒）。另以默认在线width128、100步噪声/100步采样完成CPU有限值检查（仅1次生成器更新，不代表收敛）。82个冻结生物及原始生成文件SHA变化数0；14个相关模块编译、CLI dry-run、bash语法及`git diff --check`通过。

旧流程CPU小回归最初因全局memory PSI准入等待，停止该测试进程后，在独立测试锁目录以available内存政策重跑；仍保留100GiB系统可用底线，未改变生产资源设置。新增测试均为很小的诊断数据，不是正式资源并发验证。

完整日志：`scratch/probes/baseline_implementation_20261117/online_and_recent_tests.log`及`legacy_rounds_regression.log`。汇总：同目录`online_verification.json`。本次未使用GPU，未启动正式训练，未覆盖旧实验结果。
