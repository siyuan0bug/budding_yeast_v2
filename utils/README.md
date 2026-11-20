# 公共运行库

utils提供可复用服务，不是脚本和备份的混合目录。开发规则见[AGENTS](../AGENTS.md)，使用入口见[文档导航](../docs/README.md)。

|目录|职责|
|---|---|
|training|训练器、损失、历史与checkpoint|
|evaluation|加载、指标、逐样本CSV及绘图|
|experiments|路径、日志、资源准入和内存保护|
|physics|ODE物理评分与变量约束|
|sampling/baseline|项目内独立baseline实现|
|sampling/ours|项目方法及辅助模型|
|sampling/benchmark|完整池固定协议和普通多轮流程|
|sampling/dynamic|持续训练中选样|
|sampling/generation|合法新输入生成、仿真与计费|
|sampling/online|仍在役的参数池在线工作流及Fission适配|
|archive|保留的退役源码快照，不得被当前入口和兼容层调用|

依赖方向是CLI/调度器调用运行库；运行库不得反向import scripts工具。
采样算法通过统一上下文、结果和查询接口接入，流程调用公共训练、归一化和评测。
新增算法不要复制训练器、FNO、checkpoint加载或指标公式。
正式分析、评测和维护CLI分别放scripts/analysis、scripts/eval、scripts/maintenance。
