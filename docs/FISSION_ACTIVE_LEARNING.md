# Fission 主动学习：入口、规则与运行

> **2026-10-22 当前采样已更新**：Budding/Fission 静态配额 10/40/50、AL 候选池 5/40/55，仅一至三模块。下文较早的四/五层配额属于历史描述；新规则、输出字段和失败处理见 [三模块采样说明](EXACT_THREE_MODULE_SAMPLING.md)。

实验入口是 `train.py`，按轮训练、评测和在线标注；`run_al_pure_random_1m.sh` 仍是 Budding 专用批量脚本。Fission 新增 `run_al_fission.sh`，调用同一训练入口，不复制训练流程。策略清单与系统能力由 `utils/active_learning/registry.py` 管理。

## 当前支持范围

| 策略 | Fission 怎么选候选 |
|---|---|
| `al4pde_random` | 在固定未标注池中无放回随机选择，对应 pool_random 语义 |
| `al4pde_uncertainty` | 独立训练的 ensemble 对同一输入预测，以分歧评分，再 Top-K |
| `al4pde_power` | 使用同一 ensemble 分歧，按不确定性幂权重无放回抽取 |
| `al4pde_lcmd` | 代理预测完整轨迹，使用共享投影和实际 bmdal LCMD 选择器 |
| `al4pde_bait` | 代理预测轨迹、共享投影、官方 scale/posterior 顺序与实际 bmdal BAIT 选择器 |

这五种复用既有策略函数。Uncertainty/Power 默认 MSE disagreement，仍支持 sample_variance。它们是官方数值规则的项目适配，没有直接实例化整个 AL4PDE Task/ProbModel。`coreset_maxdist` 未实现；其余十种策略含 Budding 专用通路、变量索引、形状特征或生物学先验，本次不开放给 Fission，训练和引擎初始化均显式拒绝。

## 候选与仿真边界

1. `data/systems/fission/al_pool.py` 使用原 FissionSampler：L1/L2/L3 为 45%/35%/20%，L1 网格轮转，L2/L3 均匀挑模块，模式按 class/flat 规则抽取，强度沿用原 distributions。real/池内碰撞沿用 FissionReferenceIndex 与救援算法；seed 控制完整抽样随机状态。
2. Fission 候选必须知道突变体自己的初值。先推导 WT 出生态，再逐候选执行原 `mutant_birth_state` 的 1000 min 预热和原回退规则。质量按 MASS_UNIT 导出，保存参数、真实预热初值、完整 overrides 和来源元数据。**不保存完整 300 min 目标轨迹或标签，但预热本身需要真实 ODE 积分，必须计入成本。** 不得用 WT 初值占位来冒充正确的 Fission 候选输入。
3. 新池必须是带逐株预热协议的 Fission parameter_pool，16 个变量、58 个参数和顺序/版本/时间网格均校验。加载时额外剔除与初始 NPZ 全部行（train/val/test/real）参数重合的候选；同时隔离原 20 个 real 并去池内重复。pool 自带均值不用于训练归一化。
4. acquisition 使用模型预测，选样期间不生成完整真实目标轨迹。选中后，`utils/active_learning/fission.py` 的顶层 spawn worker 调用原 `simulation.simulate_mutant`，包括预热、300 min 主轨迹、需要时的 1050 min 稳态出生质量续跑。成功须完成完整时域、状态 End、有限值并通过原质量上限。保留原 cut/Too Small/G2/M arrest 等标签；不用 Budding 的 real-envelope 阈值过滤。逐样本 30 s 截止保留；Fission AL 的父进程无进展 watchdog 为 60 s，给 spawn 导入留出余量（Budding 仍为原 35 s）。
5. 同一候选的 oracle 初值必须与选样输入一致，否则报错。失败候选被消耗但不入库；不可存活但数值有效的轨迹仍入库。五种通用策略沿用 single_batch，不保证请求数量全部成功，也不会静默补采。池耗尽后不追加。
6. 原生 Pattern_A/B/C 映射 0/1/2；原 `None` 保留为 -1，BIO 保留，不生成 Budding 六分组。新增行只使用初始 train 拟合并冻结的归一化统计量。每轮保存 native 标签摘要供 W&B 和本地记录使用；空测试拆分在 AL 曲线中记 null/CSV 空值，日志显示 N/A。

Fission 的变量、参数、初值、ODE、事件、生物标签均继承 `data/systems/fission/`，对应保留的原 `data/fission_random_sampler.py`。本次未改动这些原始生物规则，也未改动既有静态数据集或历史实验。

## 使用示例

在 `budding_yeast_v2` 目录执行，父目录需可导入：

```bash
export PYTHONPATH="$(dirname "$PWD")${PYTHONPATH:+:$PYTHONPATH}"

# 小规模原生轨迹初始集（含原 20 real）；训练仍按 DataModule 规则切分。
python data/generate.py --recipe fission_random --samples 1000 --seed 20001 \
  --workers 4 --output /path/to/fission_initial.npz

# 候选池：只做逐株预热，不生成完整带标签目标；默认拒绝覆盖。
python -m budding_yeast_v2.data.systems.fission.al_pool \
  --samples 10000 --seed 20002 --workers 4 \
  --output /path/to/fission_al_inputs.npz

# 单个策略；支持 Fission 的模型，例如 cross_fno。
python train.py --system fission --model cross_fno \
  --dataset_name /path/to/fission_initial.npz \
  --al_pool_dataset /path/to/fission_al_inputs.npz \
  --al_strategy al4pde_power --num_al_rounds 3 \
  --al_initial_train_size 1000 --max_epochs 100 \
  --loss_type hybrid_mse_rel --norm_type log1p

# 五种策略顺序运行；可用 STRATEGIES 只选其中一部分。
INIT_DATASET=/path/to/fission_initial.npz \
POOL_DATASET=/path/to/fission_al_inputs.npz \
PYTHON_BIN=python bash run_al_fission.sh
```

`--al_initial_train_size` **不会裁小已经提供的训练集**，而是缺省初始化目标及指数追加基数。例中三轮训练之间请求追加 1000、2000 条；初始训练数量由初始 NPZ 和拆分决定，失败会使实际增量减少。每轮都冷启动主模型；Uncertainty/Power 额外独立训练 ensemble 成员，最后一轮只需训练主模型。脚本默认模型 cross_fno、log1p、hybrid_mse_rel，可用环境变量或追加 CLI 参数调整。

产物沿用 `checkpoints/round_*.ckpt`、`final-last-epoch.ckpt`、`datasets/al_iter<r>_train<n>.npz`、`eval/round_*`、选样 CSV、性能曲线及 ensemble 记录；新增 `eval/dataset_labels_round_<r>.json`。`tune.py` 仍只支持静态 HPO。

## 验证范围

Fission 专项覆盖五种选样、逐株初值、原生缺失标签、错误/跨系统池拒绝、失败消耗、去重和真实 spawn/训练/重载。另以实际 `train.py` 对原生 Fission 轨迹执行 Random/Power 各两轮：训练集 7→8，保留 20 real，Power 独立成员 checkpoint 存在，空合成测试集正常记 null。关闭绘图与远程 W&B；这是小规模工程冒烟，不是主动学习效果或大规模性能对照。

Budding 15 策略 × 3 seed 的有序候选与 NumPy/Torch/Python RNG 对照仍通过。官方审计单独校验新增的系统分派与动态变量宽度，Budding 输入张量逐值不变。验证产物放在 `scratch/probes/fission_al_2026-10-21/`。

最终合并回归 **420 项全部通过**，其中 Fission AL 专项 28 项；超时余量调整后再次独立运行这 28 项也全部通过。官方数值审计与 Budding 45 项选样/RNG 对照通过，31 个受保护的原始生成文件 hash 不变。完整工程记录：`scratch/probes/fission_al_2026-10-21/verification.json`。
