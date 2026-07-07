# Budding Yeast v2 vs AL4PDE 主动学习过程宏观对比

> 本文档从宏观流程层面对比 `budding_yeast_v2`（出芽酵母细胞周期 ODE 系统）与 `al4pde`（PDE 方程系统）两个项目的主动学习（Active Learning, AL）实现，不涉及具体数据集差异，聚焦于架构、循环结构、数据流与策略范式的对比。

---

## 目录

- [1. 总览](#1-总览)
- [2. Budding Yeast v2 AL 流程详解](#2-budding-yeast-v2-al-流程详解)
  - [2.1 整体范式](#21-整体范式)
  - [2.2 初始训练集](#22-初始训练集)
  - [2.3 触发时机](#23-触发时机)
  - [2.4 选样策略](#24-选样策略)
  - [2.5 标注方式](#25-标注方式)
  - [2.6 数据池管理](#26-数据池管理)
  - [2.7 终止条件](#27-终止条件)
  - [2.8 循环结构](#28-循环结构)
- [3. AL4PDE AL 流程详解](#3-al4pde-al-流程详解)
  - [3.1 整体范式](#31-整体范式)
  - [3.2 初始训练集](#32-初始训练集)
  - [3.3 触发时机](#33-触发时机)
  - [3.4 选样策略](#34-选样策略)
  - [3.5 标注方式](#35-标注方式)
  - [3.6 数据池管理](#36-数据池管理)
  - [3.7 终止条件](#37-终止条件)
  - [3.8 循环结构](#38-循环结构)
- [4. 核心区别对比](#4-核心区别对比)
- [5. 本质差异分析](#5-本质差异分析)

---

## 1. 总览

| 项目 | 领域 | AL 范式 | 模型架构 | 数据生成 |
|------|------|---------|----------|----------|
| **budding_yeast_v2** | 系统生物学 ODE（39 维细胞周期） | 训练内嵌式增量 AL | 单模型（PyTorch Lightning） | LHS 拉丁超立方 + 物理保护 |
| **al4pde** | PDE 方程（Burgers/NS/KS 等） | 外层迭代式重训 AL | Ensemble 集成 | `torch.rand` 纯均匀随机 |

两个项目都用主动学习从大池子中选高价值样本加入训练集，但**范式截然不同**：前者把 AL 嵌入训练 callback 中持续喂新数据；后者把 AL 作为外层循环，每轮从头重训 ensemble。

---

## 2. Budding Yeast v2 AL 流程详解

### 2.1 整体范式

**训练内嵌式增量主动学习（Callback 模式）**：一次训练全程持续扩容训练集，模型不重训，新数据通过 `torch.cat` 拼接进现有 `train_dataset_lhs`，下一 epoch 立即生效。

核心实现：[utils/al_callback.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py) 中的 `ActiveLearningCallback`，通过 PyTorch Lightning 的 `on_train_start` / `on_train_epoch_end` 钩子介入训练流程。

### 2.2 初始训练集

按策略分流（[al_callback.py:148-245](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L148-L245)）：

- **池化策略**（us / is / random / wrs / vessal / hggs / pial）：
  从 `dm.train_dataset_lhs`（全量 LHS 训练集，含 X/P/Y）中通过 **K-Means 聚类**选种子
  - 簇数 = `initial_train_size`（默认 5000）
  - 在 141 维参数空间聚类，选离簇心最近的样本（[_kmeans_initial_selection](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L247-L288)）
  - 目的：保证初始集在参数空间均匀覆盖，避免随机选带来的聚集

- **查询合成策略**（rgs / mutant_aware）：
  **跳过 K-Means**，直接用全量 LHS 数据开始训练（[al_callback.py:187-207](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L187-L207)）
  - 原因：合成策略本身会定向采样，初始集不必特意选种子

### 2.3 触发时机

每 `trigger_every_n_epochs` 个 epoch 触发一次（epoch 0 跳过），在 `on_train_epoch_end` 钩子中执行（[al_callback.py:290-313](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L290-L313)）。

不同策略不同触发周期（[run_pipeline.sh:130-138](file:///home/users/zsy/trae_workspace/budding_yeast_v2/run_pipeline.sh#L130-L138)）：

| 策略 | trigger_every_n_epochs |
|------|------------------------|
| pial | 50 |
| mutant_aware | 100 |
| 其他（us/is/random/wrs/vessal/hggs/rgs） | 10 |

**嵌入训练过程内**，不重训。

### 2.4 选样策略

每轮选 `num_add` 个样本（默认 5000，v2 mutant_aware 为 500）。在 [_standard_al_sample](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L343-L408) 中分发：

| 策略 | 原理 |
|------|------|
| `random` | 池中随机抽 |
| `us` | top-K 不确定性 |
| `is` | 不确定性 × 多样性（权重 `diversity_weight`） |
| `wrs` | 加权蓄水池采样 |
| `vessal` | 动态权重（随训练进度增多样性权重） |
| `hggs` | K-Means 分层纯多样性 |
| `pial` | MC Dropout 不确定性(α=0.6→0.5) + PCA 轨迹多样性(β=0.3→0.4) + ODE 残差(γ=0.1)（[L731-820](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L731-L820)） |
| `rgs` / `mutant_aware` | 查询合成分支，走定向采样 |

详见 [ALGORITHMS_AL.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/ALGORITHMS_AL.md)。

### 2.5 标注方式

按策略分两种（[al_callback.py:1192-1229](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L1192-L1229) 与 [L2105-2158](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L2105-L2158)）：

- **池化策略**：标签**已预计算**（ODE 提前算好存池）
  - "标注" = `_move_samples_to_training_set` 从池迁移到训练集
  - 零成本，直接 `torch.cat` 拼接

- **合成策略**（rgs/mutant_aware）：实时解 ODE 标注
  - 先 [_validate_param_ranges](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L1656) 剔异常参数
  - 再 [_physics_validate_and_prune](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L2105-L2158) 用 `solve_ivp` 解 ODE 生成真实轨迹作为"标注"
  - 最后 [_select_by_real_mae](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L1837-L1917) 按"模型预测 vs ODE 真值"的真实 MAE 选取最难样本

### 2.6 数据池管理

仅池化策略有池（[al_callback.py:131-138, L222-238](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L131-L138)）：

- **池的来源**：在 `on_train_start` 中由「全量 LHS − 初始训练集」剩余样本构成 `_pool_X / _pool_P / _pool_Y`
- **池的更新**：被选中样本通过 mask **从池中移除**（[L1206-1216](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L1206-L1216)），池单调递减
- **池耗尽处理**：池空时跳过本轮采样（[L364-366](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L364-L366)）

**查询合成策略无池**：每轮用 LHS 引擎新合成候选，用完即弃。

### 2.7 终止条件

**无显式终止条件**：训练跑到 `max_epochs`（默认 200）。池化策略在池耗尽后自动停采，训练继续到 max_epochs。

### 2.8 循环结构

```
on_train_start:
    ├─ 池化策略: K-Means 选初始训练子集
    └─ 合成策略: 用全量 LHS 数据

Lightning 正常训练若干 epoch
    │
    └─ on_train_epoch_end:
        ├─ 到达触发 epoch?
        │   ├─ 是: 评估池子集不确定性 / Real Mutant MAE
        │   │       → 按策略选 num_add 个
        │   │       → 池化迁移 or 合成+ODE 验证
        │   │       → torch.cat 扩容 dm.train_dataset_lhs
        │   │       → DDP broadcast 同步
        │   └─ 否: 跳过
        └─ Trainer 下个 epoch 用新数据继续训练（不重训）

循环直到 max_epochs
最后 eval.py 独立评估
```

---

## 3. AL4PDE AL 流程详解

### 3.1 整体范式

**外层迭代式重训主动学习（Ensemble 模式）**：训练与 AL 严格交替，每轮从头重训 ensemble，数据量按指数增长。

主入口：[scripts/run_active_learning.py](file:///home/users/zsy/trae_workspace/al4pde/scripts/run_active_learning.py)（Hydra 配置 `config/main.yaml`，默认 `num_al_iter: 5`、`num_epochs: 500`）。

核心组件：
- `Task`（[al4pde/tasks/task.py](file:///home/users/zsy/trae_workspace/al4pde/al4pde/tasks/task.py)）：封装 PDE、参数生成器、IC 生成器、Oracle 数值求解器
- `ProbModel`（[al4pde/prob_models/ensemble.py](file:///home/users/zsy/trae_workspace/al4pde/al4pde/prob_models/ensemble.py)）：集成模型，每个 AL 迭代重新初始化
- `BatchSelection`（[al4pde/acquisition/batch_selection.py](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/batch_selection.py)）：采集策略基类

### 3.2 初始训练集

**纯随机种子**，无 K-Means 聚类（[batch_selection.py:30-32](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/batch_selection.py#L30-L32)）：

通过 `acq_strat.generate_initial()` 调用 `scripts/gen_data.py:generate_data`，随机抽取 `cfg.task.data_gen.num_initial_batches` 批参数，每批 `batch_size` 个，送 Oracle 求解器演化成轨迹存盘。

`gen_data.py` 还会单独生成 val/test 集（不参与训练）。

### 3.3 触发时机

**每轮 AL 训练完全结束后触发**（[run_active_learning.py:81](file:///home/users/zsy/trae_workspace/al4pde/scripts/run_active_learning.py#L81)）：

主循环 `for al_iter in range(num_al_iter)`：
1. 先 `prob_model.train_n_epoch` 跑完 `num_epochs=500` 个 epoch
2. 再 `acq_strat.generate()` 采集下一批

即「训练 → 采集」严格交替。

### 3.4 选样策略

**数量**由 `DataSchedule` 决定（[acquisition/data_schedule.py](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/data_schedule.py)）：

| 调度类型 | 公式 |
|----------|------|
| `Linear` | 固定数量 |
| `Exponential`（默认） | `first_batch_num × 2^al_iter`（如 128→256→512→1024→2048） |
| `Fixed` | 手动列表 |

**策略**分两类：

- **非池类**（[Random](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/random.py)）：每轮直接随机生成新参数，无候选池

- **池类**（[PoolBased](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/pool_based.py)）：预生成 `pool_size`（默认 10 万）候选参数
  - `UncertaintyBased`（[uncertainty_based.py](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/uncertainty_based.py)）：含 `top_k` / `power` / `random` 三种 selection_mode
  - `DistancePoolBased`（[distance_pool_based.py](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/distance_pool_based.py)）：基于 `bmdal_reg` 库的批量距离/coreset 选择，如 LCMD、MaxDist、BAIT（配置见 `config/acquisition/{bait,lcmd,max_dist,coreset_maxdist}.yaml`）

**注意**：项目里**未见 VeSSAL** 实现，常见策略为 Random、US(top_k/power)、Coreset/Distance 系列。

### 3.5 标注方式

**始终在线标注**：调用 Oracle（JAX 求解器）实时演化出真实轨迹（[batch_selection.py:83](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/batch_selection.py#L83)）：

```
simulate_all → task.save_trajectories → .npy 追加写到 task.traj_save_path
```

下一轮 `init_training` 重新加载该目录所有数据（[task.py:49](file:///home/users/zsy/trae_workspace/al4pde/al4pde/tasks/task.py#L49) `train_data_folders=[traj_save_path]`）。

### 3.6 数据池管理

**池类**（[pool_based.py:14-43](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/pool_based.py#L14-L43)）：
- 一次性预生成 `pool_size=10万` 候选参数（ic + pde_params + pool_mask 全 True）
- 每轮 `generate`（[pool_based.py:69](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/pool_based.py#L69)）调用 `select_next` 选索引，然后 `pool_mask[sel_idx]=False`（L82-84）将选中样本从池中移除
- `prepare_data` 每次只对未被 mask 掉的子集评估
- 池空会抛异常

**非池类**（Random）：无池概念，每轮独立采样。

### 3.7 终止条件

**显式 `num_al_iter`**（默认 5），最后一轮 `is_last=True` 时跳过采集（[run_active_learning.py:82, 96](file:///home/users/zsy/trae_workspace/al4pde/scripts/run_active_learning.py#L82)），仅训练。

### 3.8 循环结构

```
generate_initial:
    └─ 随机种子 → Oracle 模拟 → 存盘

for al_iter in 0..num_al_iter-1:
    ├─ set_seed
    ├─ prob_model.train_n_epoch(al_iter, 500 epochs)  # 从头重训 ensemble
    ├─ save_checkpoint; log num_data
    ├─ if not is_last:
    │   ├─ set_seed
    │   ├─ acq_strat.generate(prob_model, al_iter, train_loader)  # 选参数→Oracle模拟→追加存盘
    │   └─ save_checkpoint; end_of_al_iter_plots
    └─ next iter
```

训练集随轮次指数增长（如 128→256→512→1024→2048）。

---

## 4. 核心区别对比

| 对比项 | Budding Yeast v2 | AL4PDE |
|--------|------------------|--------|
| **AL 与训练的关系** | 训练**内嵌** AL（Callback），单次训练全程增量扩容 | 训练与 AL **外层交替**，每轮**从头重训** ensemble |
| **模型架构** | 单模型（PyTorch Lightning） | Ensemble 集成（多模型平均不确定性） |
| **初始集选择** | K-Means 聚类选种子（多样性优先） | 纯随机（简单） |
| **数据量增长** | 固定每轮 +5000（线性） | 指数增长 2×（默认）或线性或手动 |
| **标注成本** | 池化：零成本（预计算）；合成：`solve_ivp` 较慢 | 始终调 Oracle（JAX 求解器）实时算 |
| **数据池来源** | 全量 LHS 预生成（确定性） | 池类预生成 10 万；非池每轮新生成 |
| **数据生成方法** | `qmc.LatinHypercube` 拉丁超立方 + 物理保护 | `torch.rand` 纯均匀随机 |
| **物理保护** | 离散真值图鉴、成对锁定、克隆拦截、高斯微扰 | 无（仅 min/max 边界） |
| **策略丰富度** | 8 种（含 PIAL 物理信息、RGS 查询合成） | ~6 种（含 BAIT/LCMD 等 coreset 距离类） |
| **特色策略** | PIAL（MC Dropout+PCA+ODE 残差）、RGS（Real Mutant 引导合成） | BAIT/LCMD/MaxDist（基于 `bmdal_reg` 库的批量距离选择） |
| **终止** | 隐式（跑到 max_epochs=200） | 显式（num_al_iter=5） |
| **数据存储** | 内存中 `torch.cat` 扩容 | 磁盘 `.npy` 追加 + 每轮重新加载 |
| **分布式** | DDP broadcast 同步 | 单卡 / Ensemble 并行 |
| **不确定性来源** | MC Dropout（参数加噪声多次前向） | Ensemble（多模型独立训练的不一致性） |
| **多样性策略** | PCA 降维 + 参数空间 K-Means | `bmdal_reg` 库的批量距离/coreset（LCMD/MaxDist/BAIT） |

---

## 5. 本质差异分析

### 5.1 增量 vs 重训

- **Budding Yeast v2** 是「一次训练、持续喂新数据」的**增量学习**范式
  - 适合数据量大、训练成本高的场景
  - 优势：训练效率高，避免重复计算
  - 风险：可能因早期数据偏差陷入局部最优

- **AL4PDE** 是「多轮独立训练、每轮数据翻倍」的**重训式 AL**
  - 适合数据量小、训练快、需要严格对比每轮性能的场景
  - 优势：每轮独立评估，便于 ablation 研究
  - 风险：训练成本随轮次指数增长

### 5.2 标签获取

- **Budding Yeast v2** 通过**预计算池**规避标注成本
  - 池化策略：标签已预计算，迁移即用
  - 合成策略：仅 RGS/mutant_aware 需实时算（`solve_ivp` 解 ODE）
  - 设计哲学：用 LHS 提前生成大量候选，AL 阶段仅做选择

- **AL4PDE** 每轮都调 Oracle 实时算
  - 始终在线标注：选参数 → JAX 求解器演化 → 存盘
  - 设计哲学：生成与选择解耦，参数生成纯随机，靠 AL 策略弥补覆盖度

### 5.3 不确定性来源

- **Budding Yeast v2**：MC Dropout
  - 对参数加高斯噪声 `N(0, 0.02²)`，前向 10 次计算预测方差
  - 单模型即可，计算成本低
  - 适合参数空间高维（141 维）的场景

- **AL4PDE**：Ensemble
  - 多个独立模型训练，用模型间不一致性估不确定性
  - 计算成本高（N 倍训练），但估计更稳健
  - 适合 PDE 参数空间低维的场景

### 5.4 多样性策略

- **Budding Yeast v2**：PCA 降维 + 参数空间 K-Means
  - PCA 将轨迹降到 20 维避免维度灾难
  - 在低维空间算最小欧氏距离
  - 直接衡量"行为是否已被覆盖"

- **AL4PDE**：`bmdal_reg` 库的批量距离/coreset
  - LCMD（Log Determinant of Covariance）
  - MaxDist（最大最小距离）
  - BAIT（Batch Active learning via Information Theory）
  - 基于 kernel 或特征空间的批量选择，理论性强

### 5.5 设计哲学对比

| 维度 | Budding Yeast v2 | AL4PDE |
|------|------------------|--------|
| **生成阶段** | LHS 强制均匀覆盖 + 物理保护 | 纯随机，靠 AL 弥补 |
| **AL 阶段** | 定向优化（基于 Real Mutant 引导） | 严格基于不确定性/距离 |
| **训练阶段** | 增量扩容，持续学习 | 重训式，独立评估 |
| **数据规模** | 大（80000 样本 LHS 池） | 中（10万池，但实际只用几千） |
| **研究导向** | 工程优化（数据效率） | 学术对比（策略 ablation） |

---

## 参考文献

- Budding Yeast v2：
  - [ALGORITHMS_AL.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/ALGORITHMS_AL.md)：主动学习算法详解
  - [ALGORITHMS_PINN_RESIDUAL.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/ALGORITHMS_PINN_RESIDUAL.md)：PINN 残差损失详解
  - [ARCHITECTURE_OVERVIEW.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/ARCHITECTURE_OVERVIEW.md)：架构总览
  - [utils/al_callback.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)：AL 核心实现
  - [train.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/train.py)：训练入口
  - [run_pipeline.sh](file:///home/users/zsy/trae_workspace/budding_yeast_v2/run_pipeline.sh)：实验编排
  - [data/lhs_v0.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/lhs_v0.py)：LHS v0 数据生成
  - [data/lhs_v1.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/lhs_v1.py) / [lhs_v1_origin.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/lhs_v1_origin.py)：LHS v1 数据生成

- AL4PDE：
  - [scripts/run_active_learning.py](file:///home/users/zsy/trae_workspace/al4pde/scripts/run_active_learning.py)：主循环
  - [al4pde/tasks/task.py](file:///home/users/zsy/trae_workspace/al4pde/al4pde/tasks/task.py)：Task 封装
  - [al4pde/tasks/param_gen.py](file:///home/users/zsy/trae_workspace/al4pde/al4pde/tasks/param_gen.py)：PDE 参数生成器
  - [al4pde/acquisition/pool_based.py](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/pool_based.py)：池基类
  - [al4pde/acquisition/uncertainty_based.py](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/uncertainty_based.py)：不确定性选择
  - [al4pde/acquisition/distance_pool_based.py](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/distance_pool_based.py)：距离/coreset 选择
  - [al4pde/acquisition/data_schedule.py](file:///home/users/zsy/trae_workspace/al4pde/al4pde/acquisition/data_schedule.py)：数据量调度
  - [pipeline.md](file:///home/users/zsy/trae_workspace/al4pde/pipeline.md)：流程文档
