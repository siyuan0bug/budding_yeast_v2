# Budding Yeast v2 - Architecture & Framework Overview

该文档梳理 `budding_yeast_v2` 项目的整体架构：目录结构、代码模块及其作用。

## 1. 整体架构概览 (High-level Architecture)

本项目是一个**基于神经算子（Neural Operator）的代理模型（surrogate model）项目**，用于预测出芽酵母细胞周期动力学。采用 **PyTorch Lightning** 框架，核心范式是**经典训练 + 内嵌主动学习（Callback 模式）**：

- **经典训练范式**：直接使用预先生成好的大规模静态数据集（LHS 采样 + target 定向采样），训练算子模型预测 39 变量 × 500 时间步的细胞周期轨迹。支持 PINN 物理残差约束。
- **内嵌主动学习范式**：通过 PyTorch Lightning 的 Callback 机制（`ActiveLearningCallback`）在训练过程中持续扩容训练集。模型不重训，新数据通过 `torch.cat` 拼接进现有 `train_dataset_lhs`，下一 epoch 立即生效。共 13 种 AL 策略（含 al4pde 框架池化策略）。

> **注意**：自回归（AR）+ 外层迭代重训式 AL 框架（移植自 al4pde 项目）在**独立的 `al4pde4yeast/` 项目**中实现，不在 `budding_yeast_v2` 内。详见 [AL_VS_AL4PDE.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/AL_VS_AL4PDE.md) 的两项目对比。

---

## 2. 目录结构梳理 (Directory Structure)

```text
budding_yeast_v2/
├── train.py                     # ⭐ 训练入口：组装 LightningModule + DataModule + Callbacks → trainer.fit
├── eval.py                      # 评估入口：加载 checkpoint，评估 real mutants + LHS 测试集
├── tune.py                      # Optuna 超参搜索（目标：Real Mutants MAE）
├── analyze_seeds.py             # 多 seed 实验汇总（均值±标准差）
├── analyze_data_distribution.py # 数据分布探索分析
├── run_pipeline.sh              # ⭐ 主批量实验脚本：17 列配置，并行执行 train + eval
├── run_tune.sh                  # Optuna 搜索封装
├── run_v3_pipeline.sh           # V3 数据集专用流水线
│
├── models/                      # 【模型层】
│   ├── yeast_lit_module.py      # ⭐ LightningModule：模型注册表 + 训练/验证/优化器 + forward_ic_time
│   └── components/              # 各模型实现
│       ├── pure_fno.py          # PureFNO_ICLR2021 (FNO 基线)
│       ├── cross_fno.py         # NIPSYeastFNO (Cross-Attn 注入参数)
│       ├── hyper_fno.py         # ⭐ HyperFNO (动态投影 + 超网络，主力模型)
│       ├── hp_fno.py            # hpYeastFNO (耦合谱卷积 + HyperNet shift)
│       ├── ultimate_fno.py      # UltimateFNO (因果矩阵掩码, 双输入接口)
│       ├── wno_cross.py         # NIPSYeastWNO (小波卷积版)
│       ├── lno_cross.py         # NIPSYeastLNO (Laplace 卷积版)
│       ├── neural_ode.py        # NeuralODEModel (torchdiffeq)
│       └── patchtst.py          # PatchTST (ICLR2023, FiLM 注入参数)
│
├── data/                        # 【数据与生成层】
│   ├── yeast_datamodule.py      # ⭐ PyTorch Lightning DataModule：加载/划分/归一化/画图样本
│   ├── dataset_utils.py         # npz 加载器
│   ├── lhs_v1_origin.py         # ⭐ ODE 真值引擎（solve_ivp + LHS 采样 + 事件重置）
│   ├── lhs_v1_origin_v2.py      # V2 数据集生成（+ 三大家族定向扩展）
│   ├── lhs_v1_origin_v3.py      # V3 数据集生成（+ 26 个 MAE>0.3 突变体扩展）
│   ├── lhs_v1_origin_v4.py      # V4 数据集生成（+ 12 类扩展家族）
│   ├── lhs_v1_origin_v5.py      # ⭐ V5 数据集生成（+ 7 项定向修复，17 类家族）
│   ├── lhs_v1_origin_v6.py      # ⭐ V6 数据集生成（+ 6 个修复点，历史最优）
│   ├── lhs_v0.py / lhs_v1.py    # 早期 LHS 采样脚本
│   └── DATASET_GENERATION_GUIDE.md # 数据集生成指南
│
├── utils/                       # 【工具与约束层】
│   ├── al_callback.py           # ⭐ 主动学习回调（13 种策略：池化/查询合成/al4pde 框架）
│   ├── losses.py                # 损失注册表 + curriculum 权重（含 PINNResidualLoss v2）
│   ├── metrics.py               # MAE/MSE/Relative L2/Correlation
│   └── physics_eqns_torch.py    # ODE 方程的 PyTorch 可微版（供 PINN 残差用）
│
├── eval_result/                 # 评估结果输出（global_metrics_denorm.json + per_mutant_metrics.csv + 轨迹图）
├── train_result/                # 训练 checkpoint 输出
├── tune_result/                 # Optuna 搜索结果
├── logs/                        # 并行实验日志（每个实验一个 .log 文件）
├── wandb/                       # W&B 日志（在线/离线）
│
└── Markdown 文档
    ├── AGENTS.md                # ⭐ 项目工作指南（AI Agent 与开发者入口）
    ├── ALGORITHMS_AL.md         # 主动学习算法详解（13 种策略）
    ├── ALGORITHMS_PINN_RESIDUAL.md # PINN 残差损失详解
    ├── AL_VS_AL4PDE.md          # budding_yeast_v2 vs al4pde 项目 AL 对比
    ├── ARCHITECTURE_OVERVIEW.md # 本文档
    ├── normalization_scheme.md  # 归一化方案对比
    ├── data_distribution_analysis.md # 数据分布分析
    └── 总结.md                  # v2 cheat 数据集说明
```

---

## 3. 核心工作流

### 工作流：经典训练 + 内嵌主动学习

**定位**：在大规模静态数据集上拟合，通过 Callback 在训练过程中持续扩容训练集。

**调用链**：
```
train.py
  ├─ YeastLitModule (models/yeast_lit_module.py)
  │    ├─ MODEL_REGISTRY → 选择模型 (hyper_fno / cross_fno / patchtst / ...)
  │    ├─ LOSS_REGISTRY → 选择损失 (mse_only / physics_informed / pinn_residual / ...)
  │    ├─ forward_ic_time(x, p) → one_time 或 autoregressive 预测
  │    └─ Cosine + Warmup 调度器
  │
  ├─ YeastDataModule (data/yeast_datamodule.py)
  │    ├─ 加载 npz 数据集
  │    ├─ 两级分层划分（家族 × Pattern）+ 孤本保护
  │    ├─ Z-score 归一化（仅 LHS 样本统计量）
  │    └─ 3 个 DataLoader: val_lhs / test_lhs / test_real
  │
  └─ Callbacks
       ├─ ModelCheckpoint (monitor val_loss_lhs)
       ├─ LearningRateMonitor
       ├─ SimToRealVisualizerCallback (每 10 epoch 画 39 变量轨迹图)
       ├─ EpochSummaryCallback (每 epoch print 摘要行到日志)
       └─ ActiveLearningCallback (见下)
```

**ActiveLearningCallback 流程**（[utils/al_callback.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：
```
on_train_start:
  ├─ 标准池化策略: K-Means 选初始训练子集 (默认 5000)
  ├─ 查询合成策略 (rgs/mutant_aware): 用全量 LHS 数据
  └─ al4pde 框架策略: K-Means 选 15000 (v1) + 加载 v5 候选池 (162,008)

Lightning 正常训练
  └─ on_train_epoch_end (每 trigger_epoch 轮):
      ├─ 到达触发 epoch?
      │   ├─ pial → 50, mutant_aware → 100, al4pde_*/maal → 4, 其余 → 10
      │   ├─ 评估池子集 (不确定性/残差/多样性/Real MAE)
      │   ├─ 按策略选 num_add 个样本
      │   │   ├─ 标准池化: 从池迁移 (零成本)
      │   │   ├─ 查询合成: LHS 生成 + ODE 验证 (solve_ivp)
      │   │   └─ al4pde 框架: 从 v5 候选池选取 (零成本)
      │   ├─ torch.cat 扩容 dm.train_dataset_lhs
      │   └─ DDP broadcast 同步
      └─ 下一 epoch 用新数据继续训练（不重训）

训练结束 → eval.py 独立评估
```

---

## 4. 三大注册表

### 4.1 模型注册表 (MODEL_REGISTRY)
位于 [models/yeast_lit_module.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/models/yeast_lit_module.py)，映射 `model_name → ModelClass`。当前支持 9 种模型（见目录结构）。统一接口：`forward(ic_time_grid, params) → (B, V, T)`（ultimate_fno 除外，用双输入接口）。

### 4.2 损失注册表 (LOSS_REGISTRY)
位于 [utils/losses.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/losses.py)，映射 `loss_type → LossClass`。所有损失在**归一化空间**计算。当前支持 6 种：`mse_only` / `mse_negpen` / `mse_smooth` / `physics_informed` / `t_smooth` / `pinn_residual`。

### 4.3 AL 策略
位于 [utils/al_callback.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)，通过 `--al_strategy` 选择。当前支持 13 种 + none：
- **标准池化**（6 种）：random / us / is / wrs / vessal / hggs
- **查询合成**（2 种）：rgs / mutant_aware
- **物理信息**（1 种）：pial
- **al4pde 框架**（4 种）：al4pde_random / al4pde_uncertainty / al4pde_diversity / maal

---

## 5. 关键设计决策

### 5.1 防爆栈 (Defense Mechanisms)
训练 ODE 代理模型时必须内置防溢出约束，修改时不要去掉：
- **AL Callback** 中的 `_validate_param_ranges`：ODE 验证前剔除异常参数（非负、物理下界、相对上界）。
- **PINNResidualLoss** 的 `nan_to_num`：早期模型预测可能产生 inf/nan，全程防御。
- **Event mask 基于真值**：v1 基于预测导致梯度爆炸，v2 改用 `detect_event_mask_torch(target_real)`。

### 5.2 物理约束 (PINN) 的准确度
[losses.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/losses.py) 的事件检测绑定**真值 (Ground Truth)**，代表模型的物理 Loss（如 Jump Condition）被精确约束在真实发生突变的时刻。Curriculum Learning 前 30 轮纯 MSE，之后线性增长物理权重。

### 5.3 PCA 降维对于 AL 的重要性
在 39 变量 × 500 步高维特征场景下，不使用 PCA 直接求距离会导致**距离趋同现象**（高维空间任意两点距离几乎相同），主动学习退化为随机采样。PIAL 的 `_compute_trajectory_diversity_pca` 降到 20 维，保留 95%+ 方差。

### 5.4 al4pde 框架的内存优化
V5 候选池完整加载需 ~35 GB。v2 内存优化只存 `P_norm` + `Y_IC` + `mutant_names`（~0.12 GB），完整 `Y` 按需从磁盘加载（`_load_full_Y_from_disk`），每次仅 ~78 MB。

### 5.5 并行实验执行
[run_pipeline.sh](file:///home/users/zsy/trae_workspace/budding_yeast_v2/run_pipeline.sh) 用后台子 shell `{ ... } &` 并行执行所有实验，每个实验通过 `CUDA_VISIBLE_DEVICES=$GPU_ID` 独占一张 GPU。日志重定向到 `logs/${RUN_NAME}.log`，`PYTHONUNBUFFERED=1` 确保进度条立即写入。

---

## 6. 相关项目：al4pde4yeast

自回归（AR）+ 外层迭代重训式 AL 框架在**独立的 `al4pde4yeast/` 项目**中实现（路径：`/home/users/zsy/trae_workspace/al4pde4yeast/`），不在 `budding_yeast_v2` 内。该项目的特点：
- ARModelWrapper（绕开 Lightning，纯 PyTorch 多步自回归训练）
- Teacher Forcing Schedule（前 50% epoch 喂真值，防早期 AR 发散）
- 外层迭代重训式 AL（每轮从头重训 ensemble，数据量指数增长）
- 候选池来自 `bmdal_reg` 库的 BAIT/LCMD/MaxDist 距离类策略

两项目的详细对比见 [AL_VS_AL4PDE.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/AL_VS_AL4PDE.md)。
