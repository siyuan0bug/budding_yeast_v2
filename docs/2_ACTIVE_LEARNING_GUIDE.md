# 采样实验运行指南

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
