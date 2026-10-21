# Power、BAIT、LCMD 接入与官方 acquisition 清单

> 当前实现位置为 `utils/active_learning/strategies/`；统一接口、兼容路径和后续缓存/元数据修复见 [模块整理](AL_PACKAGE_REFACTOR.md)。

本地 `al4pde/config/acquisition/` 有 8 个具体预置配置，不只是 Power、LCMD、BAIT 三个待接入项。`base.yaml` 是公共配置，`data_schedule/` 是预算调度，不是新的选样算法。

## 当前对应关系

| 官方配置 | 本项目当前状态 |
|---|---|
| `random` | 尚未接入每轮生成新参数的独立入口；按用户要求保留固定池随机方式 |
| `pool_random` | `al4pde_random` 保留固定池随机语义及原 NumPy 抽样逻辑 |
| `top_k` | `al4pde_uncertainty`，独立 ensemble 评分和官方 Top-K 语义的 ODE 适配 |
| `power` | 新增 `al4pde_power`，使用相同 ensemble 分数及幂加权无放回抽样 |
| `max_dist` | 已有 `al4pde_diversity` 使用自定义 332 维特征与本地 MaxDist，不能声称完整等价于官方默认特征流程 |
| `coreset_maxdist` | 尚未接入；官方在模型 latent 表示上执行 MaxDist，需要模型提供隐特征接口 |
| `lcmd` | 新增 `al4pde_lcmd`，预测轨迹共享投影 + 实际 bmdal LCMD；平方距离加权，不做 posterior transform |
| `bait` | 新增 `al4pde_bait`，使用官方同一 `bmdal_reg` BAIT 算法与缩放/posterior 处理 |

官方还支持不同不确定性来源、轨迹变换、posterior 选项等组合，不应把这些选项误计为已完整移植的独立策略。全部官方 Task/ProbModel PDE 框架仍未直接实例化。

## 没有仿真轨迹，如何使用 LCMD / BAIT？

官方 `PoolBased.generate` 先调用 `select_next`，再调用 `simulate_all`。`DistancePoolBased.create_features_pool` 使用 `prob_model.roll_out` 得到的是**代理模型预测**；真实 simulator 在选中后才运行。

```text
候选参数 + 初始状态
  → 当前代理模型预测轨迹（便宜推理）
  → 提取特征，在特征空间按 LCMD / BAIT 选样
  → 仅对选中的候选运行真实 ODE
  → 将真实轨迹加入训练集
```

训练样本已有真实轨迹，用于拟合模型；但官方这几项配置的 `predict_train: true` 还会用模型预测训练样本的特征，保证训练与候选特征按同一方式构造。LCMD 的“簇”是特征空间中按最近代表点分配的区域，不要求提前知道候选真实轨迹，也不是一次 KMeans 聚类。

## 新增入口（LCMD 已在后续接入）

在原训练命令中选用：

```bash
# Power 默认 beta=1，复用两个独立成员的 ensemble 评分
--al_strategy al4pde_power --al_power_beta 1 --al_ensemble_size 2

# BAIT 默认 512 维共享投影，与官方设置一致
--al_strategy al4pde_bait --al_trajectory_sketch_dim 512

# LCMD 使用相同预测特征；不需要 latent 接口
--al_strategy al4pde_lcmd --al_trajectory_sketch_dim 512
```

`--al_trajectory_sketch_dim` 为 BAIT/LCMD 的共同参数，旧名称 `--al_bait_sketch_dim` 继续兼容。LCMD 直接调用 `LargestClusterMaxDistSelectionMethod(sel_with_train=True, dist_weight_mode="sq-dist")`，按官方 `lcmd.yaml` 不应用 BAIT 的 scale/posterior 流程；原 BAIT 的结果保持一致。

Power：正常非退化输入沿用官方 `weights = uncertainty ** beta`、归一化和 `torch.multinomial(replacement=False)`。全零分数采用均匀抽样；正权重不足时先抽正权重候选，再从零权重候选补足。对负值、NaN、Inf 分数报错，对数值幂溢出/下溢做等比例稳定处理。beta 必须为有限正数。

BAIT：

- 训练和候选两侧都用当前模型预测，重新计算每轮特征，不复用旧模型预测缓存。
- 出芽酵母没有空间网格，将 `(B,V,T)` 映射为 `(B,1,T,V)` 后使用官方 `spatial_mean` 语义；它只消去单例空间轴，保留时间和变量，随后共享随机投影到 512 维。不是将时间均值压成 39 维，也不是原自定义 332 维形状特征。
- 与官方一致，训练特征构造 scale transform，随后 `posterior_tfm(sigma=1e-2, allow_kernel_space_posterior=False)`，再调用 `BaitFeatureSpaceSelectionMethod(sel_with_train=True, noise_sigma=1e-2)`，保留其前向选取、后向删减和内置退化处理。
- `utils/active_learning/backends.py` 从已安装的原始 bmdal 源文件加载私有模块，为该模块提供 `np.Inf` 的局部兼容视图。未修改源码、site-packages、原 selection 模块或全局 NumPy；不是自行重写 BAIT 评分。
- 最终统一走既有 ODE 标注与过滤流程，原 BIO/ABC/补充分组元数据保持原规则；Fission 已开放五种通用策略；候选须逐株预热，使用原生标签和过滤，见 [Fission AL](FISSION_ACTIVE_LEARNING.md)。

大型池会增加预测特征内存与批量选择计算量，本轮未执行 100 万池的完整训练或性能基准。BAIT 的 512 维随机投影对 100 万候选仅特征矩阵约需 2 GB float32 内存，选择过程还会创建额外矩阵。

## 验证

273 项测试通过：增加 LCMD 与官方无 posterior 管线的逐项选中索引对照、零预算和池大小截断、退化/非有限特征、先预测后仿真、空池和 NumPy 2 隔离验证；其余覆盖包括新增入口、Power 与官方同分数同 RNG 一致、零分/极值边界、BAIT 与官方 posterior + 选择源码结果一致、共享投影保留时序信息、训练特征不读取真实目标、选择前不触发 ODE、跨轮预测更新、NumPy 兼容隔离，以及 ensemble/标签/归一化/Fission 回归。

持久审计：

```bash
python scripts/analysis/audit_al4pde_alignment.py \
  --output scratch/probes/al_lcmd_2026-10-21/alignment.json
```

原始归档、既有数据集和历史实验结果未修改；新入口不会补跑或重标历史实验。


## coreset_maxdist 的设计说明（仅分析，尚未实现）

官方配置是在模型隐特征上做 MaxDist。普通轨迹多样性比较“预测出的浓度曲线”；latent coreset 比较“模型输出前形成的内部表示”。它仍然先推理、选样，再运行真实 ODE，不能保证一定优于输出轨迹特征。

不必给每个模型文件都添加接口，但每个希望支持 latent 选样的模型/模型家族，需要一个明确的特征提取适配。可以这样组织：

1. 在 `YeastLitModule` 提供统一外部入口，例如 `extract_al_features(x,p)`。这是拟议接口，本轮未添加。
2. 建立按模型类型注册的适配：指定提取层、批维恢复、时序聚合和可选投影。比如 BioRefinerPlus 的 `stage2_refiner.head` 前特征，PureFNO 的 `q` 前特征；它们的形状并不相同。
3. 原型可以用临时 forward hook/pre-hook，在 `try/finally` 中释放，避免改模型文件。长期需要更多语义信息时，只给少数共享模型基类增加可选 `return_features=True`，派生模型自动继承，不必逐个改全部模型。
4. 优先支持当前主力 `biom_refiner_plus` 和 FNO 家族；未注册模型明确报不支持，不把预测轨迹偷偷当成 latent。每种模型自身 train/pool 特征维数须一致，不要求所有模型拥有相同原始隐维数。
5. 一次性预测与自回归的特征聚合需要分别约定，不能只保留最后一次 hook 调用；换轮或换模型后训练和候选两侧都重新提取。验证需确保读取特征不改变原预测、权重、梯度或训练模式。

当前主力模型只对外返回轨迹（部分变体另有 coarse/pattern 返回），`YeastLitModule.forward_ic_time` 也未提供统一 latent 返回。官方 FNO wrapper 的源码同样说明并非所有模型实现了 `return_features`；因此“只有支持 latent 的模型启用该策略”是合理的能力边界。用户尚未要求实现 latent coreset，本轮没有修改模型文件。
