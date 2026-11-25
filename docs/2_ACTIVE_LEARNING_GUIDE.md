# 采样实验运行指南

R19突变对照试训：`scripts/launch/run_contrast_coldstart.py --directory <外部批次目录> --dry-run`预览准备、四组真实冷启动和汇总。它是2256条/100epoch的单轮研究试训，不在正式suite注册、不与完整五轮结果混排；复用公共训练loss、逐训练集归一化及合成验证best。新探针在全部选择锁定后评测，不读real/test或旧保留审计。设计见[研究记录](research/journal.md)。默认只用已授权0—3卡、每卡一项、全局最多两项、加载I/O并发1。

响应缺失生成原型：`scripts/analysis/probe_background_transfer.py --score-kind response_omission`仅为学习收益诊断，未加入正式suite。使用三个已人工核对的独立标量机制、合法模块替换和同生成器/同机制预算随机对照；公共训练loss不变，诊断冻结统计、固定400步。主收益检查排除三个打分通道，避免将解析通道自身收益冒充耦合变量泛化。设计和停止条件见[研究记录R16](research/journal.md)；默认`integral_defect`保留原探针行为，不能混用或覆盖旧结果。

## 当前入口和输出

```bash
bash run_sampling_budding.sh extensions --suite all \
  --directory /home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/preview \
  --dry-run
```

该预览列出选定八类baseline和三类ours的23项适用池/流程组合；去掉dry-run才会启动。已有不同身份的结果不得覆盖。
无参数运行默认是固定池batch，不是全部23项。run、dynamic、generate分别调用独立流程。
统一路径由`utils/sampling/layout.py`生成：`experiments/budding_sampling/<model>/<pool>/<method>/<profile>/seed_<主seed>`。
`batches`为队列与维护审计，`shared`为共同数据协议，`summary`为跨实验汇总；模型目录才保存权重、loss、预测图与eval。

## 多seed与ensemble执行

`scripts/launch/run_multiseed_sampling.py`使用既有pool为多个实验主seed建立共享协议，分seed调用同一有界队列；每个seed内部共享划分/初始集。
显式启用`SAMPLING_ENSEMBLE_PARALLEL=1`时，普通固定池ensemble的主成员和辅助成员在独立spawn进程训练，复用同一训练快照和验证集，随机流独立；辅助日志在各成员目录。
多seed调度器支持`--jobs 1..4`、`--io-concurrency 1..4`；当前默认4项、I/O并发2，仅使用GPU0，保留系统约100GiB可用内存准入与运行保护。`--adopt-first-queue`通过已校验的交接凭据保留首个seed的在跑任务；交接时旧/新任务各自保留资源锁域，首个seed两域各1个加载槽，总计最多2个，后续seed统一2个加载槽。
GPU4～7禁止多进程ensemble。执行并行度不改变成员数量、训练epoch、损失或采样公式；正式启动后禁止热改sources范围内代码。

## 方法与运行库

- 作者baseline独立实现在`utils/sampling/baseline/`，项目方法在`ours/`。
- 完整池协议/普通多轮流程在`benchmark/`；原生持续选样在`dynamic/`；合法新输入生成在`generation/`。
- 共同接口是SamplingContext/SamplingResult、LabelOracle及公共训练/评测服务；算法不直接管理第二套训练循环。
- 必须区分未选真值不可见的AL与可读训练侧真值的数据选择。都不能读取测试/real真值选样。
- 生成式算法只复用现有合法参数、mutation、初值与ODE规则，不修改生物域；全部尝试/成功/历史曝光分别计费。

## 在线流程和归档

仍服务旧参数池工作流及Fission的在役实现位于`utils/sampling/online/`。
`utils/archive/online/`保留原始快照但不再作为生产依赖，当前入口和兼容映射指向在役实现。
Fission入口仍是其原在线流程；完整池新CLI的跨系统支持不能仅由改名推断。

## 原型与结果边界

Sensitivity、Transport、Closure已有Budding初步实验，不代表稳定优于baseline。
固定池Closure仅8/8000条新增端点来自完整四角，不能把其badcase收益归因于四角机制。
HGGS旧流程标为hggs_legacy；持续训练修正版本须单独正式运行，不可复用旧成绩。
Low-Rank梯度能量、PEAKS回归评分、PRE事件屏蔽及OGAS合法投影均需披露任务适配。
详见[现有原型与局限](history/OURS_IMPLEMENTATION_AND_PORTABILITY_2026-11-19.md)、[HGGS修正](history/HGGS_CONTINUOUS_FIX_2026-11-19.md)。

三种非四角反馈生成原型使用显式`--suite adaptive`，不改变旧suite默认任务；方法、实际实现限制和命令见[反馈生成原型](reference/ADAPTIVE_MUTATION_GENERATION.md)。

Response V2与同路径随机对照使用`--suite response_v2`，与旧`adaptive`三方法分开。未知误差不再赋固定高分，即时使用已查询支撑，并记录共享训练Relative诊断。见[Response路径V2](reference/RESPONSE_PATHS_V2.md)。

跨背景干预迁移暂只提供独立的实际学习收益探针，不加入正式suite。使用训练侧保留探针、同checkpoint及同更新预算比较评分/随机生成，见[跨背景学习收益验证](reference/BACKGROUND_TRANSFER_PROBE.md)。

“随机／真实高误差／反馈试训选批”机制诊断只访问训练侧已仿真数据，区分找准错误与实际训练收益；反馈和最终检查背景独立。另提供随机1万/3万数据量参考、近似等更新次数控制及近零通道分解，见[采样机制诊断协议](reference/SAMPLING_UTILITY_DIAGNOSTIC.md)。这些不是生产采样算法或同预算AL成绩。

条件机制覆盖、组合迁移及生成式缺口补充的三seed研究协议见[条件动力学选择对照](reference/CONDITIONAL_DYNAMICS_STUDY.md)。七组都使用最终1万条、近似47k次更新，池选择与生成的信息访问/额外pilot和ODE成本分别记录；一次性选择协议不与历史五轮榜单混排。

后续[干预阶数分布校准检验](reference/INTERVENTION_ORDER_CALIBRATION.md)只比较三个固定quota控制，先用seed42反馈筛选，有阳性信号才确认123/1024并开放独立审计；最多7个最终模型。阴性即停止，不用real调比例，不把改变L配额本身称为新算法。

配额筛选之后的统计来源控制只补跑seed42随机/低阶优先两项，沿用原有训练ID和47005步预算，公共初始2000条拟合统计；不开放独立审计/test/real，也不新增采样算法。入口`python scripts/launch/run_normalization_control.py --directory <外部批次目录>`，支持`--dry-run`。设计及归因边界见[干预阶数控制](reference/INTERVENTION_ORDER_CALIBRATION.md)。

DMCED统一采样与合法查询合成（R21）：实现在`utils/sampling/ours/dmced/`，使用`bash run_sampling_budding.sh extensions --suite dmced --seeds 42 123 1024`调度三方法（Track A 固定池`ours_dmced_pool`、支柱1消融`ours_dmced_power`、Track C 池外合法合成`ours_dmced_synth`）×三seed，沿用冻结五轮协议（2k→10k、100/100/100/100/200 epoch、2成员集成，评测仅用主成员0）。`ours_dmced_synth`仅使用原生1,159合法模块组合、原生强度/初值/碰撞救援与ODE求解器，全程计费ODE尝试与失败。设计、假设与结果状态见[研究记录R20–R21](research/journal.md)。

Switch-D2（R22）：实现在`utils/sampling/ours/switch_d2/`，使用`--suite switch_d2`调度`ours_switch_d2_pool`（未截顶逐通道平方相对方差+随机加权$D^2$采样）、`ours_switch_power`（支柱1消融）及`ours_switch_d2_synth`（层级边际守恒合法查询合成），先在`seed_42`正式五轮验证，通过门槛后再扩展多seed。见[研究记录R22](research/journal.md)。

Epistemic-D2（R23–R24）：实现在`utils/sampling/ours/epistemic_d2/`，使用`--suite epistemic_d2` / `epistemic_r24`调度`ours_epistemic_d2_pool`、`ours_epistemic_d1_pool`、`ours_epistemic_d05_pool`与`ours_epistemic_d1_synth` / `ours_epistemic_d2_synth`。R24三seed终评证实距离项`d^p`隐式饿死低阶L1/L2，该分支已终止扩大。见[研究记录R23–R24](research/journal.md)。

CEBS（R25）：实现在`utils/sampling/ours/cebs/`，使用`--suite cebs_r25`调度`ours_order_power_pool`、`ours_epistasis_pool`及`ours_epibif_synth`。`seed_42`消融证实仅模块三（0-ODE连续分岔强度合成）具强正收益，模块一/二已淘汰。见[研究记录R25](research/journal.md)。

分岔与池锚定混合查询合成（R26–R27）：实现在`utils/sampling/ours/bif_synth/`（池外分岔`synthesis.py`、纯无标签池锚定分岔与跨模式合成`pool_mode_synthesis.py`），无标签池锚点加载接口在`utils/sampling/generation/cohorts.py`（仅读输入参数与元数据，严禁读取轨迹标签），严格复用原生`catalog`/`ranges`/`apply_gene_mutation`/`valid_candidate`生物规则。队列入口：`--suite bif_r26`、`bif_r27_gpu0`、`bif_r27_gpu4`、`bif_r27_gpu7`；`run_sampling_extensions.py`对独占卡GPU 4–7强制约束`--jobs-per-gpu 1`并配合`exclusive_gpu_{4,7}.lock`。详见[研究记录R26–R27](research/journal.md)。
