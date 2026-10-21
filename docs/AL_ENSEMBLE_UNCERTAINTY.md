# Ensemble uncertainty 与官方 AL4PDE 对齐

> 后续 `al4pde_power` 已复用此 ensemble 训练与评分，BAIT 也已接入；当前状态以 [Power/BAIT 清单](AL_POWER_BAIT_AND_OFFICIAL_INVENTORY.md) 为准。下文“尚未新增 Power/BAIT”的描述记录此前阶段。

当前 `al4pde_uncertainty` 使用独立训练成员在同一输入上的预测分歧，不再使用参数加噪评分。`al4pde_random` 继续从固定参数池随机选样，数值逻辑未改。

## 训练和选样

- 默认 `--al_ensemble_size 2`，对应本地官方 `config/experiment/sub_exp/topk.yaml` 的两个成员。
- 每个需要选样的轮次：主模型按原训练流程训练；辅助成员以独立 seed、独立优化器和 Trainer 在相同累积数据上冷启动训练。归一化统计量共用并冻结。
- 辅助成员保存至 `checkpoints/round_<r>_ensemble_member_<i>.ckpt`，成员列表和 seed 记录在 `eval/ensemble_round_<r>.json`。
- 评测和最终模型继续使用第一个成员，对应官方 `first_model_prediction: true`；没有把单模型评测悄悄换成 ensemble 平均。
- 最后一轮不选样，因此只训练主模型，对应官方非同步训练路径的最后一轮行为。
- 参数与初值对所有成员完全相同，推理使用 eval/no_grad。没有训练 ensemble 时明确报错，不退回输入扰动。
- 辅助模型不写主模型的 W&B 曲线。全部成员权重保存为独立 checkpoint；本轮没有新增 AL 全状态断点恢复功能。

## 官方评分的两个分支

本地 `al4pde/prob_models/ensemble.py` 中，`uncertainty()` 给出样本方差；默认 `unc_independent_rollout()` 的 `adapt_unc_to_loss=True` 则计算相对 ensemble 均值的逐成员损失。默认 wrapper loss 是 MSE。

设 K 个预测为 y_k，均值为 m：

- 默认 `--al_ensemble_uncertainty loss_mse`：`mean_k((y_k-m)^2)`，再平均变量和时间，匹配官方默认 MSE disagreement 分支。
- 可选 `--al_ensemble_uncertainty sample_variance`：`var_k(y_k, correction=1)`，再平均变量和时间，匹配官方 raw variance 分支。
- 两者对固定 K 只差常数 K/(K-1)，Top-K 排序通常相同，但记录的分数不同。
- Top-K 使用 `torch.argsort(descending=True)`，连并列分数的处理也沿用官方函数。

这里沿用项目公共归一化坐标计算 ODE 全轨迹分歧。官方 PDE 的自回归 rollout 不直接用于当前全轨迹代理模型；MSE 分歧与当前训练用的 hybrid 等损失分开定义，不能把 hybrid 损失直接当成官方默认 MSE。

运行示例（附加到原训练命令）：

```bash
--al_strategy al4pde_uncertainty \
--al_ensemble_size 2 \
--al_ensemble_uncertainty loss_mse
```

这是官方评分语义和训练组织的项目适配，没有直接实例化官方 Task/ProbModel 全套类。当前环境整体导入该库仍缺 `hydra`、`tensordict`、`pdearena` 等依赖；本轮未安装或修改这些依赖，也没有修改官方源码。数值测试隔离执行本地官方函数对照项目结果，不能称为官方端到端 PDE 流程已接入。

## Power、LCMD、BAIT 的移植判断

| 算法 | 官方实际行为 | 在当前项目的接入条件 |
|---|---|---|
| Power | 按 `P(i) ∝ uncertainty(i)^beta` 加权无放回抽样；beta 越大越偏向高分 | 可复用当前 ensemble 分数，改选择步骤；需处理全零分数和正权重候选少于预算的边界 |
| LCMD | 把候选分到最近的训练/已选代表点；默认按平方距离累计各簇权重，优先最大加权簇，再选其中最远点 | 可复用预测轨迹特征和官方 bmdal 选择器；需控制候选规模与迭代成本。它不是一次 KMeans 后取中心 |
| BAIT | 基于特征协方差和信息准则的批量选择，使用前向选取、后向删减；官方配置含 posterior transform | 可移植官方 bmdal 实现，必须保留特征、posterior transform 和噪声设置；不能用现有 Gram–Schmidt“Fisher”替代 |

当前 `bmdal_reg` 类可导入，但该安装版本多处使用 NumPy 2 已删除的 `np.Inf`；需要在正式接入前做局部兼容处理及数值测试。不要全局修改 `numpy` 来掩盖此问题。本轮只检查可行性，未新增 Power/LCMD/BAIT 策略入口。

## 验证范围

240 项测试通过：包含两个模型的实际 CPU 训练、独立权重、checkpoint 读取、输入无噪声、原训练模式恢复、官方两个评分分支及 Top-K 对照、旧归档保留、其余算法与标签/仿真链路回归。

审计命令：

```bash
python scripts/analysis/audit_al4pde_alignment.py \
  --output scratch/probes/al_ensemble_2026-10-21/alignment.json
```

未运行生产规模主动学习或重写历史实验结果。real-envelope 依赖、跨轮特征缓存和不同策略补采预算差异仍是独立待处理事项；Fission AL 仍保持禁用。
