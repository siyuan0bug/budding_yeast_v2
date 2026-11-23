# 项目文档

日常开发只需从以下五份现行指南进入。规则以根目录[AGENTS.md](../AGENTS.md)为准。

科研跨平台接手先读[RESEARCH_HANDOFF](research/README.md)：定位唯一研究记录、当前状态与原始证据，避免依赖Codex或Antigravity某一平台的聊天历史。

|指南|内容|
|---|---|
|[模型与架构](1_ARCHITECTURE_AND_MODELS.md)|模型入口、输入输出、特征和公共训练接口|
|[采样实验](2_ACTIVE_LEARNING_GUIDE.md)|选择/生成流程、运行命令、支持范围|
|[损失与评测](3_LOSSES_AND_PHYSICS_CONSTRAINTS.md)|当前Relative定义、best选择、物理评测|
|[数据与归一化](4_DATASETS_AND_NORMALIZATION.md)|两个候选池、划分、seed、数据存储|
|[结果与比较](5_BENCHMARK_AND_LEADERBOARD.md)|当前结果入口、比较条件与结论范围|

- `reference/`：仍有参考价值的详细契约；与源代码核对后使用。
- `history/`：历史方案、修复记录及整理前五份指南的内容快照。历史说明不自动代表当前实现。
- `research/`：跨平台研究记录、接手规则、当前状态与证据索引；实验原始产物仍留在外部结果目录。
- 实验报告、图、CSV、JSON和每次排障记录保存在外部实验目录，禁止继续向本目录堆放运行产物。
- 修改现行规则应更新指南正文；实验状态不再逐条置顶追加。
