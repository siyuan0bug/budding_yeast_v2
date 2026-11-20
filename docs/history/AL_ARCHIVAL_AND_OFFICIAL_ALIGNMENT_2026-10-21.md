# 主动学习归档与官方实现差异核对（2026-10-21）

旧轨迹池策略已退出生产入口。本文记录归档阶段的差异；随后 uncertainty 已按官方 ensemble 默认 MSE 分歧和 raw 方差分支修正，见 [当前 Ensemble 对齐](AL_ENSEMBLE_UNCERTAINTY.md)。其余差异仍保留，历史实验不因新代码变更而重新标为官方结果。

## 本轮变更

- `utils/archive/al_callback_legacy.py`：归档前 `al_callback.py` 的完整原样快照，保留旧七种策略及共享依赖，供显式历史复现。生产代码不导入此文件。
- `utils/al_callback.py`：移除旧七种策略的 19 个方法，包括初始化、epoch 内触发、预生成轨迹迁移及专属评分辅助方法。保留 12 种在线策略和四个别名。
- `train.py`：CLI 仅接受 `none`、在线 `al4pde_*` 策略和 `maal`，删除单次训练中的旧回调挂载。直接构造 callback 时，旧名称和未知名称明确报错。
- `run_pipeline.sh`：移除旧策略专属分支。旧构造参数和部分 CLI 参数保留以兼容配置；帮助文本说明哪些参数已不生效。
- 澄清 uncertainty 注释与日志：历史方法名保留，实际计算为参数扰动预测方差。

归档采用完整快照，而不是只摘录函数，以保留当时的导入、全局常量和共享方法。后续复现需显式导入归档回调；原训练 CLI 不再接受旧策略。

## 归档阶段与本地官方库的对照（Uncertainty 已在后续修正）

对照源码目录为 `../al4pde/al4pde/`。

| 能力 | 官方本地实现 | 当前项目实现 | 判定 |
|---|---|---|---|
| Uncertainty | `UncertaintyBased` 调用 `ProbModel`；`Ensemble` 在同一输入上计算模型成员预测分歧，也支持不同 rollout 模式 | 单模型保持 eval，对归一化参数加标准差 0.02 的高斯噪声，重复五次并平均预测方差 | 实质不同；当前衡量局部输入敏感性，不是 MC Dropout，也不等同于后验不确定性 |
| Top-K | `uncertainty_based.top_k` | NumPy 排序截取 | 同一组无并列分数时选中集合一致；分数来源不同 |
| Random | `Random` 每轮生成新参数；`pool_random` 从固定参数/初值池随机选 | 固定候选池随机抽取 | 更接近官方 pool_random；不是官方 fresh-draw Random |
| Diversity | `DistancePoolBased`，配置轨迹/latent 特征、bmdal 选择器和可选 posterior transform | 自定义 332 维形状与参数特征，项目 PyTorch MaxDist | 有相同的贪心距离选择思想，但完整特征、变换和调用链不同，不能声称整体等价 |
| 指数调度 | `first_batch_num × 2^iteration`，再乘 batch size 得到样本量 | `initial_train_size × 2^iteration`，池大小截断 | 转换为相同样本量单位、未触发截断时公式一致；配置接口不同 |
| Power / LCMD / BAIT | 官方有对应 acquisition 配置和实现 | 未接入 | 缺少官方策略 |
| MAAL / MASS / Novelty / Residual / Bifurcation / Event / Fisher | 没有这些同名项目扩展 | 项目专用评分及组合选择 | 属于扩展，不能当作官方基线 |

`al4pde_fisher_oed` 当前正交化的是残差加权轨迹特征，没有计算参数梯度或 Fisher 信息矩阵。`al4pde_bifurcation` 是振幅和漂移启发式，没有求解分岔点。`al4pde_bifurcation_entropy` 使用 ABC 分类熵，六分类仅保存为补充元数据。

## 尚未修改的流程差异与问题

1. 公共入库仍用真实突变体测试轨迹包络设置幅值阈值；部分评分提前使用同样包络，通量门控也含 Table S5 先验。不能称为完全测试盲态。
2. Random / Uncertainty / Diversity / MAAL / MASS / Novelty 为单批仿真，失败后不继续补足；Residual 及后面几种使用排序队列补采，候选耗尽时仍可能不足。比较时必须同时记录入库数和实际仿真预算。
3. 全池预测特征缓存跨模型轮次保留，只删除候选行，下一轮可能继续使用旧模型预测。该问题已定位，本轮为保持在线算法行为未修改。
4. 外部新版 `budding_pool` 与 callback 无预存池时的 MASS 即时池，分布和 Level 比例不同。
5. Fission 仍在训练入口禁止主动学习；本轮没有解除此保护或借用 Budding oracle。

## 验证与复查

- 231 项回归测试通过，覆盖旧策略拒绝、归档可构造、全部在线名称/别名分发、Random/Uncertainty/Diversity 同种子选择一致、AL 标签快照、归一化和仿真保护、Fission 监控。
- AST 排除文档和日志后，12 个在线 acquisition 方法及其他保留的数值辅助方法与归档一致；删除了一处 return 后不可达的重复语句。
- 人工输入反例：参数敏感的确定性模型，项目预测方差为正；两个相同的官方 ensemble 成员方差为 0；常数预测 0 和 2 的两个成员方差为 2。
- 官方函数从本地源码 AST 隔离加载，仅运行数值方法，不初始化 Hydra/Task 或真实仿真。这是源码及数值审计，不是官方库端到端集成验证。
- 已有数据集、实验结果和官方库源码均未修改。

可重复审计：

```bash
python scripts/analysis/audit_al4pde_alignment.py \
  --output scratch/probes/al_cleanup_2026-10-21/alignment.json
```

后续仍需完善官方 Task/ProbModel 接入和其他官方选择器，统一仿真预算、解决缓存及测试包络依赖；当前 uncertainty 已按上述后续说明完成数值和训练组织适配。
