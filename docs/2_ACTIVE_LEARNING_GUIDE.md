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
