# Ours 的现状、通用实现与运行边界

日期：2026-11-19。旧方法与新原型同时保留。本文区分源代码存在、当前流程接入、机制验证与正式效果，不把它们混为一谈。

## 1. 原有 ours 到底有哪些

### 1.1 已接入当前完整池 benchmark 的四个原方法

|方法|实际机制|当前完整池入口|跨系统情况|
|---|---|---|---|
|`ours_pattern_balance`|以已查询合成训练数据拟合辅助分类器，再按预测 Pattern 配额选择|`run --method ours_pattern_balance`|描述器和编码仍绑定 Budding|
|`ours_pattern_context`|Pattern 加突变背景的条件分配|`run --method ours_pattern_context`|同上|
|`ours_pattern_support`|Pattern 加局部支撑不足程度|`run --method ours_pattern_support`|同上|
|`ours_sensitivity`|合法强度探针下，模型响应与 ODE 切向/割线响应的一致性；只在连续区间计算|`run --pool mutation --method ours_sensitivity`|39/141 维、210 分钟、Budding guard 等仍硬绑定|

注册见 `utils/sampling/registry.py` 的 `METHODS`，调用适配见 `ours/offline.py`。原辅助编码 `auxiliary/views.py` 明确使用 39 通道，不能仅改 system_id 就声称支持 Fission。

三个 Pattern 方法需要额外 ensemble；sensitivity 使用主模型。原完整池测试曾覆盖它们的流程，本次还会复核；这不等于新候选池和 10k 预算下已有正式效果。

### 1.2 主要仍在旧在线流程中的方法

|方法组|名称|当前默认完整池流程|
|---|---|---|
|对照响应|`ours_response`、`ours_response_marginal`、`ours_response_feedback`|未迁移这些名称的完整入口|
|机制/交互|`ours_interaction`、`ours_invariance`|未迁移这些名称的完整入口|
|反馈与事件|`ours_event`、`ours_error`、`ours_coverage`|未迁移这些名称的完整入口|
|混合研究变体|`ours_global_interaction`、`ours_global_response`、`ours_global_tail`、`ours_global_calibrated`|未迁移这些名称的完整入口|

这些名称在历史 `SPECS` 中，并依赖归档 online engine 的候选池、provenance、ensemble 和反馈状态。旧流程可通过 `SAMPLING_MODE=online` 显式进入，但不是本次完整池/10k 协议。禁止为了“能跑”让当前 CLI 静默切换协议。

另有 `maal`、`al4pde_mass/novelty/residual/residual_diversity/bifurcation/bifurcation_entropy/event_entropy/fisher_oed` 等更早研究启发式。它们放在 ours 目录不表示是官方基线，也不建议直接作为新论文主线。

## 2. 旧方法效果如何理解

2026-10-31 旧在线协议为百万参数池、最终 70k train、FNO192/4/24、batch512、100/100/100/200 epochs。历史报告给出的部分 real MAE/MSE 如下：

|方法|MAE|MSE|
|---|---:|---:|
|Power|0.085997|1.329895|
|ours_global_tail|0.093763|1.629908|
|ours_response|0.109470|2.359423|
|ours_response_feedback|0.105677|2.449735|
|ours_invariance|0.103247|1.642369|
|ours_sensitivity|0.113218|2.020053|
|ours_pattern_context|0.087077|0.945067|
|ours_pattern_support|0.092608|0.979995|

来源：`AL_OPTIMIZED_RESULTS_AND_NEXT_STEPS_2026-10-31.md` 与 `AL_MECHANISM_COMPLETION_2026-10-31.md`。表内数值是旧报告记录，不是本次重新评测。

结论：旧 ours 有运行结果和可复用机制，尚不能据此宣称稳定战胜 Power；部分指标有收益，部分更差。旧 Relative L2 是已经弃用的历史定义，本表不将它拿来与当前新 Relative L2 混排。70k 与当前 10k 的预算、数据分布、loss、checkpoint 等协议差异也不能忽略。

还需纠正原创性口径：TEM1 条件不变性在旧 `ours_invariance` 中**已经实现过**，四端点交互与边际成本在旧 `query_design.py` 中也已经存在。新实现是面向独立接口、误差传播和池外实验设计的研究原型，不能把旧 TEM1 例子或四元组本身重复包装成新贡献。

## 3. 本轮新增的三个可独立运行原型

统一代码在 `utils/sampling/ours/mechanistic/`；不读取 Pattern 或 real 表格进行评分，不引入新的训练模型/损失。

|入口名|具体实现|诚实的适用范围|
|---|---|---|
|`ours_mechanism_witness`|用已查询训练轨迹作锚点，估计合法干预后的响应，再查找代理预测与机制响应不一致的输入；经过严格校验的等价关系优先直接比较锚点真值|目前仅 TEM1 不活跃通道有精确等价证据，其余是局部线性响应估计，不能称严格下界|
|`ours_dynamics_transport`|从初值和时间离散方程缺陷出发，用隐式线性传播估计完整轨迹误差；孤立横截事件使用 saltation 更新|一阶离散估计，有有限差分/网格误差；不是已证明的神经算子误差上界|
|`ours_intervention_closure`|寻找精确的双干预强度矩形，根据估计交互预测误差及新增端点成本补全；未知交互不预设为零|固定池中精确矩形不足时明确记录单点补齐；生成模式主动提出合法四角|

三个方法只用一个当前代理模型，不需要额外 ensemble。前两个保留 10% 显式随机探索，第三个按实验组成本与余量补齐。探索只是预算安排，不是论文创新点。

### 3.1 模块职责

- `systems.py`：只读系统适配、原 RHS 的张量化、单位转换、guard/reset，以及保守的条件等价校验。
- `transport.py`：连续误差/响应传播、事件时间敏感性更新和数值诊断。
- `actions.py`：精确干预矩形、共享端点与新增查询成本。
- `selectors.py`：批量预测、train-only 尺度、候选分数与统一 `SamplingResult`。
- `generation.py`：同一选择逻辑作用于新合法提案，保存/恢复算法计数；不保存第二份轨迹或模型。
- `sampling/generation/controlled.py`：生成服务调用原 mutation 规则组织四角，并负责碰撞、去重和保留集合隔离；选择器只看到接受的输入。

### 3.2 生物规则与信息隔离

参数、初值、mutation 支持与强度范围、环境绑定、ODE、real 定义及 data 根目录原文件均不在本次修改范围。新方法改变自适应入选分布，不修改合法支持。

固定池方法只获得输入与已查询训练轨迹，候选真实输出不传入选择器。Pattern 元数据不会被使用。生成服务继续负责 real 精确碰撞排除以及冻结 holdout 的隔离，策略不读取保留集合名单或距离。

Fission 的 MASS 输出和存储初值使用不同质量尺度，新适配器分别转换；`kS` 不在导出参数中，按复制状态为当前连续段恢复原模式。初值直接使用原流程给出的实际出生状态，不自行固定为 WT，不另造初值。

### 3.3 数值范围和仍需研究的内容

默认在 33 个粗网格点基础上保留检测到的每个事件跨越的两端，再进行传播。数值 Jacobian 使用中心差分，隐式步保证比显式 Euler 更适于刚性衰减，但不保证任意非线性预测都可靠。

孤立横截事件使用 saltation；擦边事件显式记录 `grazing_updates` 并使用 reset Jacobian 近似；同一区间多个事件记为 `unresolved_event_intervals`。此类结果不能作为严格事件误差证据。模型完全漏掉事件、粗输出遗漏触发以及长距离干预导致线性化失效仍是局限。

TEM1 精确证据只在已核对的 RHS/重置源哈希、正的分母常量、Tem1T=0 和 TEM1(0)=0 等条件成立时启用；源码变化后禁用该证据，不能继续沿用证明标签。

`selected_scores`、精确/近似响应数量、事件退化计数、闭包端点与 `point_error_fill` 都写入现有 acquisition 记录。没有精确四角时不应把单点补齐的实验称为“全部交互采样”。

## 4. 可以怎么运行

先使用已经存在的 neuralop 环境或脚本默认 Python。从 `budding_yeast_v2/` 目录运行；以下仅是命令示例，本轮不自动启动正式训练。

### 4.1 当前完整池实验

```bash
CUDA_VISIBLE_DEVICES=0 bash run_sampling_budding.sh run \
  --protocol /home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_a866aef5d988 \
  --pool mutation --method ours_dynamics_transport --device cuda:0 \
  --output /home/users/zsy/mydata/budding_yeast/experiments/budding_ours_portable \
  --plots --wandb
```

`--method` 可换成另外两个新入口。旧四个已迁移 ours 也使用 `run`，但不代表其适用 Fission。closure 只接受 mutation 池；无 mutation 语义的 LHS 不虚构功能模块。

复用冻结池、相同随机初始 2k、2k/4k/6k/8k/10k 五轮、train-only 逐轮重新拟合归一化、原 normalized validation-best。新输出根避免覆盖既有实验。W&B 仍由现有入口强制 offline。

### 4.2 池外合法生成实验

```bash
CUDA_VISIBLE_DEVICES=0 bash run_sampling_budding.sh generate \
  --protocol /home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_a866aef5d988 \
  --pool mutation --method ours_intervention_closure --device cuda:0 \
  --output /home/users/zsy/mydata/budding_yeast/experiments/budding_ours_portable_generation \
  --workers 4 --bank-size 64 --query-batch-size 8 --plots --wandb
```

三个新方法都可进入 `generate`。前两个选择新生成的原规则提案；closure 由生成服务先组织合法的双轴四角，再用同一选择器计分。新轨迹分块仅写一次，各轮保存源引用，不重复保存累计轨迹。失败仍由现有流程补采，ODE 尝试与成功数继续记账。

这些方法涉及 Jacobian 和隐式求解，不能把旧 Random/LCMD 的资源吞吐结论直接套过来。正式大批并发前应先测单次选样耗时与内存；已有的系统可用内存/I/O准入保护继续有效。

### 4.3 Fission 的支持范围

通用选择内核接收 `SamplingContext(system_id='fission')`，使用真实的 16 状态、58 参数、300 分钟和原 guard/reset。基础接口和 RHS/模式/单位已安排独立验证，不需要 Budding Pattern。

**当前完整池 benchmark 的 protocol、数据划分及生成 runner 仍是 Budding 服务。不能声称直接运行 `run_sampling_fission.sh --method <新方法>` 就能完成新协议正式实验。**Fission 的完整实验服务接入及其原预热成本账本是后续明确工作，不靠给旧 engine 添加一个别名假装已完成。

## 5. 验证与结果解释

专项测试：`tests/test_portable_ours.py`。包含两个原系统的接口、原生 Fission RHS 对账、动态模式、单位转换、闭式解传播、一阶初值响应、事件时间敏感性、标签隔离、Pattern 扰动不变、精确四角与预算，以及真实 canonical ODE 的生成/恢复链路。

测试里的两轮小 FNO 是功能验证，不是正式准确率实验，也不构成新方法优于 baseline 的证据。正式新候选池下的性能、90k 扫描成本、跨系统预测收益、多 seed 稳定性及理论新颖性尚待验证。

本轮完成的检查：

- 新原型、旧机制与既有完整池 ours 两轮链路联合检查 45 项通过。
- 原 Random/OED/OGAS 生成、失败补采和合法域回归 5 项通过。
- 最后 5 项针对性检查覆盖稳定候选 ID 和闭包端点审计，其中 1 项为新增独立检查；总计 51 个不同相关检查通过。
- 修正一条旧测试：APC-A 现在是精确零固定等位规则，没有合法连续强度探针；旧测试仍期望生成一个探针。只更新测试预期，不改 mutation 规则。
- 用已有正式 FNO192 checkpoint、8 个训练侧候选与 8 个已查询锚点做 CPU 评分诊断：witness/transport/closure 分别约 1.695/1.130/1.142 秒，进程 RSS 峰值约 0.852 GiB。不是完整 90k 性能测量，不能按此承诺正式并发吞吐。
- 101 个原生生物系统及 data 根目录文件前后哈希一致，未修改既有数据池，未启动新的正式长训练。

诊断记录：`scratch/probes/ours_portable_20261119/{verification,profile}.json`。未给历史实验补造新指标或声称新原型已经优于 baseline。

## 6. 用户授权后的正式启动

已在 tmux `budding_ours_seed42_20261119` 启动17项（12项池内＋5项生成），GPU0/1每卡2项、总4项并发。严格复用原FNO/seed42协议，未重新生成底座。结果根 `/home/users/zsy/mydata/budding_yeast/experiments/budding_ours/seed42_20261119/`；队列、任务状态、日志链接发布器和启动核查在 `experiments/budding_ours_queue/seed42_20261119/`。阶段进度以 tasks.json 和实际 train.log 为准。启动后禁止修改生产source身份；上文“未启动”描述实现验证阶段历史状态。

## 7. 2026-11-19 上午运行性能审计

两项新方法已运行约7小时24分仍在第三次选样，约88%时间用于CPU机制评分；Pattern Context/Support已完成，但各有约4小时47分在等共享评分名额。此为吞吐及调度问题，快照无报错/OOM/死锁证据。完整只读计时见[分析报告](../../../../../../../../data/zsy/budding_yeast/analysis/ours_runtime_20261119/report.md)。本轮未热修改生产源码或重启现有实验。
