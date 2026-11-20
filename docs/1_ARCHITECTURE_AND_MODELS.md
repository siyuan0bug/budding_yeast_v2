# 模型与架构

## 当前职责

模型定义在`models/baselines/`与`models/ours/`，由`models/yeast_lit_module.py`注册和调度。
`train.py`、`eval.py`、`tune.py`为CLI入口，复用`utils/training`、`utils/evaluation`服务；采样不拥有第二套FNO或训练器。
scripts是工具入口层，不是被运行库反向导入的公共API。

## 输入输出

- Budding：实际初值与时间网格`(B,39,2,T)`，归一化参数`(B,141)`，输出`(B,39,T)`。
- Fission使用自身16变量、58参数和实际初值，不能套用Budding维度、标签或预热规则。
- `forward_ic_time(x,p)`统一主模型调用；特征选择通过模型的显式特征接口，不在各算法中重写模型。
- 模型输入采用统一编码后的float32；存储物理初值的float64不代表模型以float64训练。

## 扩展要求

标准baseline只允许必要任务输入输出适配；不能加入原方法没有的机制后仍使用原名。
SC-FNO要求严格匹配的灵敏度监督。具体清单见[模型基线核查](reference/BASELINE_SELECTION_AUDIT.md)。
新模型需注册，验证形状、前向、损失、保存与重载；Budding/Fission支持范围分别验证。
checkpoint加载共用`utils/evaluation/checkpoints.py`，不能绕过统计、维度及模型键校验。

[详细Fission契约](reference/FISSION_EXTENSION.md)。整理前的架构、研究模型及历史测量保存在[历史快照](history/GUIDE_SNAPSHOT_1_ARCHITECTURE_AND_MODELS.md)，不是新的实验成绩。
