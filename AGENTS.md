# AGENTS.md

面向 AI Agent 与开发者的出芽酵母细胞周期仿真项目工作指南。本文件旨在让任何新介入者（人或 Agent）仅凭此文档即可建立对代码库的完整心智模型，正确运行实验、修改模型、扩展功能而不踩坑。

---

## 1. 项目本质与科学背景

这是一个**基于神经算子（Neural Operator）的代理模型（surrogate model）项目**，用于预测出芽酵母（*Saccharomyces cerevisiae*）细胞周期动力学。

### 1.1 预测任务
- **输入**：141 维动力学参数向量（对应 ODE 模型的速率常数/初始条件，可视为"突变扰动"）。
- **输出**：39 个分子物种在 500 个时间步（跨度 210 min）上的浓度轨迹，张量形状 `(B, 39, 500)`。
- **39 个变量**（`VARIABLE_NAMES`）：MASS, CLN2, CLB2, CLB5, SIC1, CDC6, C2, C5, F2, F5, SIC1P, C2P, C5P, CDC6P, F2P, F5P, SWI5T, SWI5, IEP, CDC20T, CDC20, CDH1T, CDH1, CDC14T, CDC14, NET1T, NET1, RENT, TEM1, CDC15, PPX, PDS1, ESP1, ORI, BUD, SPN, Vi20, lte1, BUB2。
- **9 个核心变量**（`CORE_9_INDICES`）：评估时单独报告指标（CLN2, CLB2, CLB5, SIC1, CDC6, CDC20, CDH1, CDC14, ORI 等关键调控因子）。

### 1.2 核心挑战：Sim-to-Real 迁移
- **训练集** = LHS（拉丁超立方采样）生成的 ODE 仿真样本（数万量级，覆盖参数空间的"母体+扰动"）。
- **测试集** = 126 个**真实实验突变体**轨迹 + 部分 LHS 留出样本。
- 模型在仿真分布上训练，但必须在真实实验分布上泛化。这是本项目所有设计决策（归一化方案、数据划分、主动学习、PINN 损失）的根本驱动力。

### 1.3 数据集命名约定
- `lhs_cheat_origin_210min_500steps_dual_labels.npz` — 主数据集（"cheat"= 训练/测试同分布可见，"dual_labels"= 含生物活性 + Pattern 双重标签）。
- `lhs_blind_210min_500steps_dual_labels.npz` — 盲测数据集。
- **Pattern 标签**：A/B/C 三类，表示细胞周期行为模式（如正常分裂/异常/不可逆 arrest），用于分层划分与稀有流形保护。

### 1.4 项目形态
这是一个**研究型 Python 项目**，不是可安装的包——没有 `setup.py`/`pyproject.toml`，所有脚本在仓库根目录直接用 `python xxx.py` 运行。基于 **PyTorch Lightning** 框架。

---

## 2. 运行环境（必须遵守）

### 2.1 Python 环境
- **conda env**：`neuralop`（Python 3.10）。解释器路径：`/home/users/zsy/.conda/envs/neuralop/bin/python`
- **关键依赖**：`torch`, `pytorch_lightning`, `wandb`, `optuna`, `scikit-learn`, `ptwt`（小波）, `torchdiffeq`（NeuralODE）, `scipy`（solve_ivp）。

### 2.2 PYTHONPATH（关键陷阱）
代码内部用 `budding_yeast_v2.` 前缀导入（如 `from budding_yeast_v2.models.yeast_lit_module import YeastLitModule`），因此必须把**父目录**加入 PYTHONPATH：
```bash
export PYTHONPATH=$(pwd)/..:$PYTHONPATH
```
所有 shell 脚本已内置此行；直接 `python train.py` 从仓库根目录运行时也需要设置，否则导入失败。

### 2.3 GPU 与分布式
- 通过 `CUDA_VISIBLE_DEVICES` 指定单卡；多卡时设 `NCCL_P2P_DISABLE=1`。
- 默认 `precision="bf16-mixed"` 加速。
- 多卡 strategy：`ddp_find_unused_parameters_true`（因部分模型有未使用参数）。

### 2.4 数据位置
数据集**不在仓库内**（被 `.gitignore` 忽略），位于 `/home/users/zsy/mydata/budding_yeast/`。邻接矩阵 `yeast_signed_adjacency_matrix.npy` 在 `mydata/`。

### 2.5 W&B 日志
- 默认在线上传到 project `budding_yeast_v2_active`；`--no_wandb` 切离线模式，日志存 `wandb/`，用 `wandb sync wandb/offline-run-*` 恢复上传。
- `run_pipeline.sh` 顶部有 `WANDB_API_KEY` 和 `NO_WANDB` 开关。

---

## 3. 入口脚本

| 脚本 | 用途 |
|------|------|
| `train.py` | 训练入口。组装 LightningModule + DataModule + Callbacks（含 AL + 可视化）→ `trainer.fit` |
| `eval.py` | 加载 checkpoint，评估 real mutants + LHS 测试集，输出 JSON/CSV/39 变量轨迹图 |
| `tune.py` | Optuna 超参搜索（目标：Real Mutants MAE），结果存 `tune_result/best_params.json` |
| `train_eval_filtered.py` | 仅用差母体（MAE > 阈值）的 LHS 子样本训练，验证数据分布假设 |
| `analyze_seeds.py` | 汇总多 seed 实验的均值±标准差 |
| `run_pipeline.sh` | **主批量实验脚本**：读 `EXPERIMENTS` 数组，逐个训练+评估 |
| `run_filtered_experiment.sh` | 过滤训练集实验的封装 |
| `run_tune.sh` | Optuna 搜索封装 |

### 3.1 run_pipeline.sh 实验配置格式
`EXPERIMENTS` 数组每行 16 列（空格分隔）：
```
MODEL LOSS_TYPE USE_ADJ DATASET_PATH DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE
```
脚本逻辑：编辑 `EXPERIMENTS` 数组 → 自动 `train.py` → 找 `train_result/<run_name>/*.ckpt` → `eval.py`。脚本内含 PINN 专属参数自动注入（`LOSS_TYPE=pinn_residual` 时加 `--lambda_phys 0.01 --lambda_jump 0.001 ...`）和 PIAL 推迟触发逻辑（`pial` → `TRIGGER_EPOCH=50`，其余 → 10）。

### 3.2 常用命令
```bash
# 基线训练（无 AL）
python train.py --model hyper_fno --loss_type mse_only \
    --dataset_name /home/users/zsy/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels.npz \
    --modes 32 --width 32 --batch_size 64 --max_epochs 200 --lr 6e-4 --warmup_epochs 10 \
    --al_strategy none

# 评估
python eval.py --ckpt train_result/<run_name>/best-*.ckpt \
    --dataset_name <dataset.npz> --save_dir ./eval_result/<run_name>

# 批量实验（编辑 EXPERIMENTS 数组后）
bash run_pipeline.sh

# Optuna 超参搜索
python tune.py --model hyper_fno --al_strategy none --dataset_name <tag> --n_trials 100
```

---

## 4. 代码架构

```
budding_yeast_v2/
├── train.py / eval.py / tune.py / train_eval_filtered.py / analyze_seeds.py   # 入口
├── run_pipeline.sh / run_filtered_experiment.sh / run_tune.sh                # 批量脚本
├── models/
│   ├── yeast_lit_module.py             # ⭐ LightningModule：模型注册表 + 训练/验证/优化器 + forward_ic_time
│   └── components/                      # 各模型实现
│       ├── pure_fno.py                  # PureFNO_ICLR2021 (FNO 基线)
│       ├── cross_fno.py                 # NIPSYeastFNO (Cross-Attn 注入参数)
│       ├── hyper_fno.py                 # HyperFNO (动态投影 + 超网络)
│       ├── hp_fno.py                    # hpYeastFNO (耦合谱卷积 + HyperNet shift)
│       ├── ultimate_fno.py              # UltimateFNO (因果矩阵掩码, 双输入接口)
│       ├── wno_cross.py                 # NIPSYeastWNO (小波卷积版)
│       ├── lno_cross.py                 # NIPSYeastLNO (Laplace 卷积版)
│       ├── neural_ode.py                # NeuralODEModel (torchdiffeq)
│       └── patchtst.py                  # PatchTST (ICLR2023, FiLM 注入参数)
├── data/
│   ├── yeast_datamodule.py             # ⭐ DataModule：加载/划分/归一化/画图样本
│   ├── dataset_utils.py                # npz 加载器
│   └── lhs_v1_origin.py                # ⭐ ODE 真值引擎（solve_ivp + LHS 采样 + 事件重置）
└── utils/
    ├── al_callback.py                  # ⭐ 主动学习回调（9 种策略）
    ├── losses.py                       # 损失注册表 + curriculum 权重
    ├── metrics.py                      # MAE/MSE/Relative L2/Correlation
    └── physics_eqns_torch.py           # ODE 方程的 PyTorch 可微版（供 PINN 残差用）
```

### 4.1 三个核心注册表

- **模型注册表**（[yeast_lit_module.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/models/yeast_lit_module.py)）：`MODEL_REGISTRY` 映射 `model_name → ModelClass`。新增模型须在此注册，并在 `__init__` 中加对应构造分支。当前支持 9 种（见下表）。
- **损失注册表**（[losses.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/losses.py)）：`LOSS_REGISTRY` 映射 `loss_type → LossClass`。
- **AL 策略**（[al_callback.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：`--al_strategy` 选项。

---

## 5. 模型清单（forward 接口约定）

**统一接口**（除 ultimate_fno 外）：`forward(ic_time_grid, params)`
- `ic_time_grid: (B, V, 2, T)`，dim 2 = [初始条件 IC, 时间网格 t_grid]
- `params: (B, 141)`
- 返回 `(B, V, T)`

| 模型名 | 类 | 架构特点 | 参数注入方式 |
|--------|----|----|----|
| `pure_fno` | `PureFNO_ICLR2021` | 标准 FNO（4 层谱卷积块），变量独立处理 | 拼接到输入通道 |
| `cross_fno` | `NIPSYeastFNO` | IC 编码器 + ParamToICCrossAttention + NIPSYeastBlock（Fourier+Attn+Local），支持邻接矩阵物理偏置 | Cross-Attention |
| `hyper_fno` | `HyperFNO` | 超网络：参数编码为 `lambda_prime`，通过 `DynamicProjection` 调制每层权重（R0/W0 动态） | Hypernetwork 动态投影 |
| `hp_fno` | `hpYeastFNO` | 耦合谱卷积（变量间混频）+ HyperNet 生成 per-layer shift | HyperNet shift |
| `ultimate_fno` | `UltimateFNO` | **特殊双输入接口** `forward(ic, t_grid, param_cond)`，因果矩阵稀疏注意力混合器，`param_dim=42`，不支持自回归滚动 | 拼接到 cond |
| `wno` | `NIPSYeastWNO` | 用小波卷积（`ptwt`）替换谱卷积的 cross_fno 变体 | Cross-Attention |
| `lno` | `NIPSYeastLNO` | 用 Laplace 卷积（指数包络+FFT）替换谱卷积 | Cross-Attention |
| `neural_ode` | `NeuralODEModel` | `DerivativeMLP` + `torchdiffeq.odeint`（dopri5），需 `torchdiffeq` | 拼接到导数网络 |
| `patchtst` | `PatchTST` | 通道独立 + Patching + Transformer（ICLR2023），`modes` 参数未用（兼容接口） | FiLM 调制每层 |

**自回归滚动**：`forward_ic_time` 根据 `rollout_type` 选择 `one_time`（一次性预测，默认）或 `autoregressive`（逐步反馈）。`ultimate_fno` 不支持自回归。autoregressive 模式显存约是 one_time 的 T 倍。

---

## 6. 损失函数清单

`LOSS_REGISTRY`（[losses.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/losses.py)）。所有损失在**归一化空间**计算。统一签名：`forward(pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth)`。

| loss_type | 类 | 组成 |
|-----------|----|----|
| `mse_only` | `MSEOnlyLoss` | 纯 MSE（最常用基线） |
| `mse_negpen` | `MSENegPenLoss` | MSE + 负值惩罚（`relu(bound - pred)²`，bound=`-mean_y/std_y`） |
| `mse_smooth` | `MSESmoothLoss` | MSE + 频谱平滑（`|FFT(pred)-FFT(target)|²·k²`） |
| `physics_informed` | `PhysicsInformedLoss` | MSE + 负值惩罚 + 频谱平滑 |
| `t_smooth` | `DilatedSobolevLoss` | MSE + 膨胀跳跃检测 + 一阶/二阶平滑（跳变点用 `dilation_window` 卷积膨胀） |
| `pinn_residual` | `PINNResidualLoss` | ⭐ MSE + 负值惩罚 + ODE 残差 + 跳跃条件（见下） |

### 6.1 PINNResidualLoss（物理约束损失，v2）
- **Curriculum learning**：内部 `set_epoch` 管理。前 `warmup_epochs=30` 轮纯 MSE，之后 30 轮线性增长到目标权重（`lambda_phys`/`lambda_jump`）。
- **ODE 残差**：反归一化到真实空间 → 中心差分算 `dû/dt` → 向量化算 `f(û, p)` → 残差归一化回归一化空间 → 用**真值检测的 event mask** 软掩码（连续区间权重 1.0，event 区间权重 `event_weight`）。
- **跳跃条件 v2**：用**真值变化方向**约束预测（`relu(-sign(d_target)·d_pred)²`），v1 用 event 类型取最小违反量导致约束失效。
- **关键修复**：v1 的 event mask 基于预测（早期错误→梯度爆炸），v2 改用真值；v1 `lambda_phys=1.0` 导致 val_loss 灾难性恶化（0.930 vs baseline 0.037），v2 降到 0.01。
- `residual_subsample` 控制残差计算的时间步子采样率（默认 1=全部，10=隔 10 步，降低计算量）。

### 6.2 get_curriculum_weights（外部 curriculum）
`PhysicsInformedLoss`/`mse_negpen`/`mse_smooth` 用外部 curriculum：epoch<15 返回 (0,0)；15-60 线性增长到 (10.0, 0.5)；60+ 返回 (10.0, 0.5)。`pinn_residual` 不用此函数（内部自管）。

---

## 7. 数据流水线

### 7.1 数据生成（lhs_v1_origin.py）
- `eqns(t, y, p)`：39 维 ODE 右端函数（出芽酵母细胞周期模型，~140 个参数）。
- `simulate_mutant(name, param_overrides)`：用 `scipy.integrate.solve_ivp` 积分，处理 4 个 terminal events（见第 9 节），返回 `(name, y_uniform, bio_label, pattern_label)`。
- `generate_lhs_tasks(n)` / `generate_local_lhs_tasks(target_mutants, perturbation)`：LHS 采样生成参数组合。
- `mutant_rules`：突变体规则字典（参数覆盖）。
- `get_default_params()`：默认参数，`PARAM_KEYS = sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])` 得到 141 维。

### 7.2 归一化方案（详见 [normalization_scheme.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/normalization_scheme.md)）
| 数据 | 算法 | 统计量来源 |
|------|------|-----------|
| Y 轨迹 (39×500) | Z-score | **仅 LHS 样本** |
| Params (141) | Min-Max → [0,1] | **仅 LHS 样本** |
| 时间 t | 无（已 [0,1]） | — |
| IC | = `Y_norm[:, :, 0]` | — |

- `mean_y/std_y` 形状 `(1, 39, 1)`；`p_mean/p_std` 形状 `(1, 141)`。
- **为什么 Y 用 Z-score**：Loss-评估对齐度最高（Spearman=0.48）。zscore 后各变量方差≈1，MSE 隐式权重与大值变量（ORI, CDC20T, CLB2）的评估贡献正相关。log_mm 反而负相关（-0.13），minmax 极端放大（13×）。
- 归一化在 `YeastDataModule.setup` 中完成并缓存为模块属性；`PINNResidualLoss` 在 `on_train_start` 中用 datamodule 真实统计量重建损失（替换 `__init__` 里的占位实例）。
- 评估/画图时需手动反归一化：`pred * std_y + mean_y`。

### 7.3 数据划分：两级分层 + 孤本保护
在 [yeast_datamodule.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/yeast_datamodule.py) 中：
1. 先按**突变体家族**（`_LHS_` 前缀，母体名 = `split('_LHS_')[0]`）分组——同一母体的所有 LHS 扰动必须同进同出，避免数据泄露。
2. 再按 **Pattern**（A/B/C）子组分组。
3. 每个子组内独立切 10% val / 10% test / 80% train。
4. **孤本保护**：子组 ≤2 样本全部进训练集；≤4 样本不抽测试集。修改切分逻辑要保留此保护，否则稀有 Pattern 流形会丢失。

DataModule 提供 3 个验证/测试 DataLoader：`val_loader_lhs`、`test_loader_lhs`、`test_loader_real`（dataloader_idx 0/1/2 对应 `val_loss_lhs`/`test_loss_lhs`/`test_loss_real`）。

---

## 8. 训练流水线（LightningModule）

### 8.1 优化器与调度
- **AdamW**（`lr`, `weight_decay`）。
- **Cosine + Warmup**：前 `warmup_epochs` 轮线性 0→1，之后 cosine 退火到 `eta_min=1e-6`。
- 调度器 `interval='epoch'`。

### 8.2 训练步
- `training_step`：`forward_ic_time(x, p)` → 根据 `loss_type` 调用 criterion（pinn_residual 需 `set_epoch`）→ log `train_loss` + 各分量。
- `validation_step`：3 个 dataloader_idx 对应 3 个 val/test 指标（归一化空间 MSE）。
- **Checkpoint**：monitor `val_loss_lhs`，`mode=min`，`save_top_k=1`。

### 8.3 回调
- `ModelCheckpoint`：存 `train_result/<run_name>/best-{epoch:02d}-{val_loss_lhs:.4f}.ckpt`。
- `LearningRateMonitor`。
- `SimToRealVisualizerCallback`：每 10 epoch 在固定画图样本上预测并上传 W&B（39 变量网格图，反归一化）。
- `ActiveLearningCallback`（见第 10 节）。

---

## 9. ODE 事件系统（关键科学细节）

出芽酵母 ODE 有 **4 种 terminal events**（[lhs_v1_origin.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/lhs_v1_origin.py)）：
- `event1`：CLB2 下降穿过 0.3（分裂）
- `event2`：CLB2+CLB5 下降穿过 0.2（S期退出）
- `event3`：ORI 上升穿过 1.0（DNA复制）
- `event4`：SPN 上升穿过 1.0（纺锤体通过）

触发时变量被**硬重置**（如 BUD→0、SPN→0、MASS 衰减、ORI→0）。这导致：
- 插值后突变被"涂抹"到小区间，数值微分产生虚假巨大导数。
- PINN 残差/PIAL 计算时必须用 `_detect_event_mask` / `detect_event_mask_torch` 构建时间掩码，只在连续区间算残差。
- [physics_eqns_torch.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/physics_eqns_torch.py) 的 `apply_event_torch` 实现了可微的事件重置规则，`eqns_torch` 是 ODE 方程的 PyTorch 可微版。

`analyze_cell_viability_unified` 根据 event 历史判断生物活性（Viable / Inviable: G1 Arrest / Inviable: M-phase Arrest）。

---

## 10. 主动学习（AL）系统

### 10.1 触发机制
- `ActiveLearningCallback` 在 `on_train_start` 用 K-Means（或 random）从全量 LHS 训练集训出初始子集（默认 `initial_train_size=2000/5000`），剩余进未标注池。
- 每隔 `trigger_every_n_epochs` 触发一次采样。
- **Trainer 必须设 `reload_dataloaders_every_n_epochs=trigger_epoch`**，否则 AL 新增数据不会生效。此逻辑已在 `train.py`/`tune.py` 中根据 `al_strategy != 'none'` 自动开启。
- AL 仅在 `trainer.is_global_zero` 进程执行采样，再 `broadcast` 到其他进程。

### 10.2 策略清单（`--al_strategy`）

| 策略 | 范式 | 机制 |
|------|------|------|
| `none` | — | 不启用 AL，全量数据锁定 |
| `random` | 池化 | 随机采样 |
| `us` | 池化 | Uncertainty Sampling（MC Dropout 预测方差/熵） |
| `is` | 池化 | Importance Sampling = 不确定性 × 多样性（`diversity_weight`） |
| `wrs` | 池化 | 加权蓄水池采样 |
| `vessal` | 池化 | VeSSAL（不确定性+多样性） |
| `hggs` | 池化 | HGGS 流形采样 |
| `rgs` | **查询合成** | Real-Guided Sampling：评估 Real Mutant MAE → top-K 差样本邻域局部 LHS 扰动（90% 预算）+ 全局 LHS（10%）→ ODE `solve_ivp` 物理验证当"标注" → 用真实 MAE 选取最难的 `num_add` 个 |
| `pial` | 池化 | Physics-Informed AL：三信号融合 = MC Dropout 不确定性（主导）+ PCA 轨迹多样性 + ODE 残差（辅助）。**推迟到 epoch 50 触发**（早期模型预测差，ODE 残差无意义） |

### 10.3 RGS 细节
- `perturbation`：连续参数局部扰动幅度（0.05=±5%）。
- `mae_threshold`：MAE > 此阈值的 Real Mutant 优先密集采样。
- `_validate_param_ranges`：ODE 验证前剔除异常参数（非负、关键参数下界如 `mdt>10`、上界 5×默认）。
- `_physics_validate_and_prune`：用 ODE 重新积分验证候选，剔除数值失败样本。
- `_select_by_real_mae`：用模型预测 vs ODE 真值的真实 MAE 选取最难样本。

### 10.4 PIAL 细节
- `_compute_trajectory_diversity_pca`：PCA 降到 20 维解决维度灾难（原版 39×500=19500 维距离趋同），PCA 拟合在训练集上避免泄露。
- 不确定性为主信号，ODE 残差为辅助（因残差计算昂贵且早期不准）。

---

## 11. 评估流水线（eval.py）

- 加载 checkpoint → `dm.setup('test')` → 评估 3 个子集：`real_mutants`、`lhs_test`、`all_test`。
- 每个子集报告 `all_39_vars` 和 `core_9_vars` 的 MAE/MSE/Relative L2/Correlation。
- 输出：
  - `global_metrics_denorm.json`：汇总指标。
  - `per_mutant_metrics.csv`：每个突变体的指标（供 `train_eval_filtered.py` 读 MAE 筛差母体）。
  - `real_mutants/*.png`、`lhs_test/*.png`：39 变量轨迹对比图（反归一化）。
- `tune.py` 目标：Real Mutants MAE（加载 best ckpt → 在 `test_dataset_real` 上算 MAE）。

---

## 12. 关键约定与陷阱（修改前必读）

### 12.1 参数键顺序（PARAM_KEYS）—— 切勿改动
`al_callback.py`、`physics_eqns_torch.py`、`lhs_v1_origin.py` 三处都用相同的 `PARAM_KEYS = sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])`。这 141 维参数向量的顺序必须与 `np.savez` 时的特征顺序**完全一致**，否则模型预测与 ODE 真值会错位。修改任一文件的参数处理逻辑时，三者必须同步。

### 12.2 归一化统计量来源
`mean_y/std_y/p_mean/p_std` **只用 LHS 样本**计算（避免稀有 Pattern 缺席导致偏差）。`PINNResidualLoss` 必须在 `on_train_start` 用 datamodule 真实统计量重建，不能用 `__init__` 的占位实例。

### 12.3 数据划分孤本保护
修改 `YeastDataModule` 切分逻辑时必须保留两级分层 + 孤本保护，否则稀有 Pattern 流形丢失、同母体泄露。

### 12.4 AL 生效条件
`reload_dataloaders_every_n_epochs` 必须设为 `al_trigger_epoch`（已自动化）。`pial` 推迟到 epoch 50。AL 采样仅在 global_zero 进程执行。

### 12.5 ultimate_fno 特殊性
双输入接口 `forward(ic, t_grid, param_cond)`，`param_dim=42`，**不支持自回归滚动**，在 `forward_ic_time` 中单独分支处理。

### 12.6 PINN 损失的 v1/v2 教训
v1 的三个致命问题（`lambda_phys=1.0` 灾难性恶化、`lambda_penalty=10.0` 压制 MSE、event mask 基于预测导致梯度爆炸）已在 v2 修复。修改 PINN 损失时不要倒退回 v1 设定。

### 12.7 forward 接口一致性
新增模型必须遵守 `forward(ic_time_grid, params) → (B, V, T)` 接口（除非像 ultimate_fno 一样在 `forward_ic_time` 加特殊分支）。

### 12.8 注入 PYTHONPATH
直接 `python train.py`（非通过 shell 脚本）时，若从仓库根目录运行，仍需 `export PYTHONPATH=$(pwd)/..:$PYTHONPATH`，否则 `budding_yeast_v2.` 导入失败。

---

## 13. 扩展指南

### 13.1 新增模型
1. 在 `models/components/` 创建 `xxx.py`，实现 `forward(ic_time_grid, params) → (B, V, T)`。
2. 在 [yeast_lit_module.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/models/yeast_lit_module.py) 顶部 import，加入 `MODEL_REGISTRY`，在 `__init__` 加构造分支。
3. 若接口特殊（如 ultimate_fno），在 `forward_ic_time` 加分支。

### 13.2 新增损失
1. 在 [losses.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/losses.py) 实现 `nn.Module`，签名 `forward(pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth)`。
2. 加入 `LOSS_REGISTRY`。
3. 若需 datamodule 统计量，在 `on_train_start` 加重建逻辑（参考 `pinn_residual`）。

### 13.3 新增 AL 策略
1. 在 [al_callback.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py) 的采样分发处加分支，实现 `_xxx_sample` 方法。
2. 在 `train.py` 的 `--al_strategy` choices 加入。
3. 若是查询合成（如 rgs），用 `lhs_v1_origin.py` 引擎生成 + ODE 验证。

### 13.4 新增实验
编辑 `run_pipeline.sh` 的 `EXPERIMENTS` 数组，按 16 列格式添加一行即可。

---

## 14. 输出目录约定
- `train_result/<run_name>/best-*.ckpt` — checkpoint
- `eval_result/<run_name>/` — `global_metrics_denorm.json`、`per_mutant_metrics.csv`、`real_mutants/*.png`、`lhs_test/*.png`
- `tune_result/best_params.json` + `optuna_study.db` + `checkpoints/trial_*/`
- `wandb/`、`lightning_logs/`、`*_result/` 均被 `.gitignore` 忽略

`<run_name>` 格式：`${MODEL}_${AL_STRATEGY}_${LOSS_TYPE}_${DATASET_TAG}_modes${MODES}_seed${SEED}`

---

## 15. 代码风格
- 注释与日志大量使用中文；保持新增代码注释语言一致（中文）。
- 关键参数与设计决策在代码中有详细 docstring，修改前先读注释。
- `VARIABLE_NAMES`（39 项）与 `CORE_9_INDICES`（9 个核心变量索引）在多处复用，保持一致。
- `.gitignore` 已忽略 `*.npz *.npy *.csv *.png` 及各类结果目录——不要提交数据/产物文件。
- 参考文档：[normalization_scheme.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/normalization_scheme.md)（归一化方案对比）、[总结.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/总结.md)（v2 cheat 数据集说明）。
