# 目录整理与 Fission 50k 实验审查（2026-10-21）

> **后续修复已落地**：本报告是修复前审查快照。列出的四类缺陷现已修复，旧三个框架目录已收口到systems，
> 当前路径和兼容边界以 [整理说明](DATA_GENERATION_REFACTOR_2026-10-20.md) 为准；下方反例保留作为历史证据。


目录拆分方向合理，核心生物规则未发现漂移，但存在需要修复的边界错误。此次仅审查，未修改实现、正式数据、checkpoint 或历史评测；探针及证据放在 `scratch/probes/directory_review_2026-10-21/`。

## 已复现的问题

| 优先级 | 问题及触发条件 | 影响与建议 |
|---|---|---|
| P1 | `data/systems/fission/samplers/references.py:34`：`add()` 后 `recent` 尚未达到32条时，没有 recent_tree，collides() 也没有直接检查缓冲。每次主树重建后重复出现。合法 Growth:scale 向量添加后立即查询同一向量，实际False，应为True | 后续生成可能漏掉重复候选；主树和增量缓冲必须共同覆盖全部引用。用最近缓冲直接检查，或维护已索引长度并扫描尾部；按批次构建增量树。当前写法达到32后每条add都重建recent_tree，也有额外开销 |
| P2 | `scripts/analysis/analyze_experiments.py:197`：某seed只有all_variables，另一个有all_16_vars时，全局移除别名，但没有把每个seed的block归一化 | MAE分别1和9的两个seed被汇总成9，仍显示n_seeds=2；正确为5。应逐seed规范键名再汇总，并记录每项有效seed数 |
| P2 | `data/sampling/factory.py:4`及其他兼容shim：globals().update只复制当前对象，不代理赋值；旧Sampler类也没有转发 | 原from data.sampling.factory import Sampler已失败；给sampling.common.MODE_FREQ赋flat，真实系统模块仍为class。对外承诺兼容时应保留旧符号和状态语义，或明确废弃旧模块 |
| P2 | `scripts/analysis/audit_sampler_refactor.py:109`：审计直接读sampler.kernel，新FissionSampler没有此属性 | 原样运行105项只通过99项，6项Fission报AttributeError。不是6项规则不一致；审计端把可选kernel按None处理后，6项候选/RNG比较全部通过。应修复审计接口，并覆盖新adapter和兼容边界 |

另外，Fission救援存在真实规则改动，不能称为纯目录搬迁：`systems/fission/samplers/rescue.py:21` 对有替代模式的模块也有50%概率保留原模式，仅重抽强度。可复现Rum1:OP救援后仍是Rum1:OP。原提交前实现优先切换模式，新逻辑意在避免切到真实KO后反复碰撞，但与此前“换一种突变形式”的约定有差别，需要明确策略并独立版本化。基础强度分布本身未变。

## 生物与采样一致性证据

- 203项相关pytest通过，覆盖系统、标签、模块化生成、派生配对、读取和Fission监控。通过现有测试并未覆盖上述增量索引、混合格式seed和shim赋值边界。
- 两套合成突变赋值/幅度分布及功能模块审计381/381通过；包含32次/单元、seed0/7/42、模式权重和RNG状态对照。MASS基础生物与real仍统一取MA；其TEM1/CDC15安全初值是此前授权统一，不回退旧MASS real初值。
- 所有126条Budding real重新比较旧MA与当前模拟器：126/126逐元素一致，标签一致，最大绝对差0。
- 采样审计原样99/105；6项Fission均为缺kernel的工具错误。仅在独立探针提供可选kernel=None后，6/6亦完全一致。Budding Random、MASS、Gibbs、Sobol、k-DPP、Motif、BAMS、WT-Blind、Twins、targeted的审计比较通过。
- 当前Fission数据的20条real及11条分散抽取的synthetic重新用旧fission_random_sampler和当前模拟器运行：31/31的轨迹、参数、标签与文件逐元素一致，最大绝对差0。原始变量/参数/ODE、预热与标签协议未发现漂移。
- 重放完整50000个Fission候选，不做ODE：文件保留下来的49186条synthetic的参数与名称全部一致；6216次碰撞、6216次救援、12536次救援尝试也与生成报告完全一致。
- 31个原始归档生成器的SHA256保持一致。

这些结论不等于所有策略的大规模完整输出与最旧脚本逐字节相同：统一MA real初值、碰撞救援、超时及失败导出策略具有此前或本次明确的差异。

## 本次 Fission 数据与实验

文件为 `/data/zsy/budding_yeast/fission_random_50000_seed20000_modular.npz`。生成seed20000；两个训练实验均为seed42。

- 请求50000条synthetic，实际49186条synthetic＋20条real，合计49206。少814条均记录为Discarded_Mass，没有数值失败；这是过滤后的实际数量，不能按文件名声称有50000条有效synthetic。
- 全量流式检查49206×16×500个float64数据：无非有限轨迹，每条y_init与轨迹首点一致，维度16/58、时长300分钟。
- BIO总计：Viable34867、G2 Arrest7789、M-phase Arrest4237、Too Small2001、cut312；原pattern为B25403、A9464、None14339，无Budding六分类。
- 精确参数去重仅发现real基准的ste9D/cdc13dbD一对相同参数，为原始定义中的别名，保留。未发现额外synthetic精确重复，未发现synthetic与任一real在allclose阈值下碰撞；增量索引漏查窗口内也未发现synthetic近重复。故没有证据要求仅因该去重漏洞作废当前实验；漏洞仍需在下一轮生成前修复。
- 两个实验使用相同train39346、val4920、synth-test4920、real20，train_synthetic归一化，均完成100epoch并恢复验证最优checkpoint。BioRefinerPlus本身为通用卷积/时间特征/图注意力实现，本次允许16/58维是合理的；其图provenance来自Fission。

以下是global_metrics_denorm.json的real指标，relL2与Corr为先合并同变量全部样本/时间后计算的口径，不是逐mutant CSV平均值：

| 模型 | MAE | MSE | pooled relL2 | pooled Corr |
|---|---:|---:|---:|---:|
| FNO1d | 0.031449 | 0.024279 | 0.137331 | 0.958521 |
| BioRefinerPlus | 0.026654 | 0.024104 | 0.136807 | 0.960068 |

BioRefinerPlus的MAE改善15.25%，MSE改善0.72%，pooled relL2改善0.38%；只有一个seed，不是稳定优劣结论。最差wee1ts cdc25D的relL2反而从FNO的0.5213升至0.5591，Corr从0.3389降至0.3029，不能据平均指标认定坏例解决。

日志中两者第1epoch的train_loss=nan出现在EpochSummaryCallback读取尚未聚合的train_loss时；后续训练和最终指标有限，现有证据不支持宣称训练发散。该回调及曲线记录应在正确的训练epoch聚合阶段读取指标，避免第一轮缺值或滞后一轮。CSV中的字面None若用pandas默认参数读成NaN，也不是原Fission标签被重写；按字符串读取应使用keep_default_na=False。

## 目录优化优先级

1. 先修上述索引、聚合、兼容和审计问题，不再进行大规模目录搬迁。生物模型与采样分开、各系统维护各自samplers、公共runner调度的结构已经基本到位。
2. 将SystemSpec、adapter、sampler的系统分派集中在一个注册表。目前base.get_adapter、registry.get_system_spec和factory.make_sampler均有硬编码系统分支；文档“只注册registry即可添加系统”并不成立。补充明确的Sampler协议和参数/返回类型。
3. 将MODE_FREQ、L1轮转、QuotaPicker等可变状态收归sampler实例。当前factory仍通过模块全局状态设置，两个不同配置实例交错使用会相互影响；单次顺序生成通过不代表并发隔离。
4. 对救援、去重、超时和标签分别记规则版本；run元数据记录dirty patch/source hash。当前相同model_version/rng_contract无法区分所有执行策略改变。
5. 提交时完整带入24个当前untracked的data/systems实现文件，不能只提交git diff中的薄shim；否则其他机器会缺少真实实现。原始31个生成器继续原地归档，不改写。过时的默认参数说明同步修正：recipe默认已分开，Fission为1000/seed20000，并非文档所称全部100000/42。

主要证据：`mutations.json`、`samplers.json`、`fission_sampler_controlled_audit.json`、`budding_real_parity.json`、`fission_simulation_parity.json`、`fission_replay.json`、`stream_fission_report.json`、`reference_and_shim_failures.json`、`counterexamples.json`；均位于审查探针目录。
