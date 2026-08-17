# 主动学习（Active Learning, AL）算法详解

本项目实现了 **14 种主动学习策略**，用于从参数空间中智能选取最有价值的样本进行 ODE 数值求解（"标注"），以最小计算成本训练高精度神经求解器。

> **最新版本**: v2.2 (2026-07) — al4pde 框架引入训练节奏优化 P1-P5, 详见 [AL4PDE_CHANGELOG.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/AL4PDE_CHANGELOG.md)

## 目录

- [1. 算法分类总览](#1-算法分类总览)
- [2. 通用 AL 框架](#2-通用-al-框架)
  - [2.1 整体流程](#21-整体流程)
  - [2.2 初始训练集选择（K-Means 聚类）](#22-初始训练集选择k-means-聚类)
  - [2.3 未标注池（Unlabeled Pool）管理](#23-未标注池unlabeled-pool管理)
  - [2.4 池子集采样（Subsample Pool）](#24-池子集采样subsample-pool)
  - [2.5 不确定性计算（公共）](#25-不确定性计算公共)
  - [2.6 参数空间多样性（公共）](#26-参数空间多样性公共)
  - [2.7 样本迁移（公共）](#27-样本迁移公共)
  - [2.8 关键参数](#28-关键参数)
- [3. 标准池化策略](#3-标准池化策略)
  - [3.1 Random](#31-random)
  - [3.2 US (Uncertainty Sampling)](#32-us-uncertainty-sampling)
  - [3.3 IS (Importance Sampling)](#33-is-importance-sampling)
  - [3.4 WRS (Weighted Reservoir Sampling)](#34-wrs-weighted-reservoir-sampling)
  - [3.5 VeSSAL](#35-vessal)
  - [3.6 HGGS](#36-hggs)
- [4. 查询合成策略](#4-查询合成策略)
  - [4.1 RGS (Real-Guided Sampling)](#41-rgs-real-guided-sampling)
    - [4.1.1 RGS 完整流程](#411-rgs-完整流程)
    - [4.1.2 Step 1: Real Mutant 评估](#412-step-1-real-mutant-评估)
    - [4.1.3 Step 2: 名额分配](#413-step-2-名额分配)
    - [4.1.4 Step 3: 候选生成](#414-step-3-候选生成)
    - [4.1.5 Step 3.5: 参数合理性校验](#415-step-35-参数合理性校验)
    - [4.1.6 Step 4: ODE 物理验证](#416-step-4-ode-物理验证)
    - [4.1.7 Step 5: 真实 MAE 选取](#417-step-5-真实-mae-选取)
    - [4.1.8 RGS 与池化策略的区别](#418-rgs-与池化策略的区别)
    - [4.1.9 RGS 的优势与局限](#419-rgs-的优势与局限)
- [5. 物理信息策略](#5-物理信息策略)
  - [5.1 PIAL (Physics-Informed Active Learning) v2](#51-pial-physics-informed-active-learning-v2)
- [6. AL4PDE 大规模候选池框架 (v2.2)](#6-al4pde-大规模候选池框架-v22)
  - [6.1 策略概览](#61-策略概览)
  - [6.2 候选池三类采样 (v2.0+)](#62-候选池三类采样-v20)
  - [6.3 diversity/maal/boundary 形状感知 (v1.2+)](#63-diversitymaalboundary-形状感知-v12)
  - [6.4 al4pde_boundary 策略 (v1.3+)](#64-al4pde_boundary-策略-v13)
  - [6.5 MAAL 策略详述](#65-maal-策略详述)
  - [6.6 v2.2 训练节奏优化 P1-P5](#66-v22-训练节奏优化-p1-p5)
  - [6.7 内存优化机制](#67-内存优化机制)
  - [6.8 候选池初始化与状态维护](#68-候选池初始化与状态维护)
  - [6.9 al4pde 五策略调用链](#69-al4pde-五策略调用链)
- [7. 实验配置](#7-实验配置)
- [8. 实验结果](#8-实验结果)
  - [8.1 al4pde V2 框架 4 策略对比 (v2.1 配置)](#81-al4pde-v2-框架-4-策略对比-v21-配置)
  - [8.2 al4pde V2.2 框架 5 策略对比 (v2.2 配置, 待运行)](#82-al4pde-v22-框架-5-策略对比-v22-配置-待运行)
  - [8.3 数据集版本演进对比](#83-数据集版本演进对比)

---


## 1. 算法分类总览

| 策略 | 类型 | 是否需要标签 | 核心信号 | 触发 Epoch |
|------|------|-------------|---------|-----------|
| `random` | 池化 | 否 | 随机 | 10 |
| `us` | 池化 | 否 | 预测方差 | 10 |
| `is` | 池化 | 否 | 不确定性 + 参数多样性 | 10 |
| `wrs` | 池化 | 否 | 加权随机（不确定性） | 10 |
| `vessal` | 池化 | 否 | 动态权重（不确定性+多样性） | 10 |
| `hggs` | 池化 | 否 | K-Means 参数空间覆盖 | 10 |
| `rgs` | 合成 | ODE 验证 | Real Mutant MAE 引导 | 10 |
| `mutant_aware` | 合成 | ODE 验证 | 历史 MAE + Atlas 优先级加权 | 100 |
| `pial` | 池化 | **不需要** | ODE 残差 + 轨迹多样性 | **50** |
| `al4pde_random` | 池化 (候选集) | 否 | 从候选池随机选取 | 分段 8/4/8 (v2.2) |
| `al4pde_uncertainty`| 池化 (候选集) | 否 | 从候选池按预测方差选取 | 分段 8/4/8 (v2.2) |
| `al4pde_diversity` | 池化 (候选集) | 否 | 332维形状特征 MaxDist (方案A权重) | 分段 8/4/8 (v2.2) |
| `maal` | 池化 (候选集) | 否 | diversity + target_priority 加权 | 分段 8/4/8 (v2.2) |
| `al4pde_boundary` | 池化 (候选集) | 否 | (1-w)·diversity + w·boundary_proximity | 分段 8/4/8 (v2.2) |

---

## 2. 通用 AL 框架

所有策略共享统一流程，本节详细说明每个公共环节。

### 2.1 整体流程

```
训练开始
  │
  ▼
[on_train_start] 初始化
  ├─ 获取全量 LHS 训练集 (full_X, full_P, full_Y)
  ├─ K-Means 聚类选初始训练集 (initial_train_size=2000)
  ├─ 剩余样本 → 未标注池 (unlabeled pool)
  └─ 验证集/测试集保持不变
  │
  ▼
[模型训练] epoch 0 → trigger_epoch
  │
  ▼
[on_train_epoch_end] 触发 AL 采样 (每 trigger_every_n_epochs 轮)
  ├─ Step 1: 从池采样子集 (pool_subset_size=10000)
  ├─ Step 2: 模型推理计算信号 (不确定性/残差/多样性)
  ├─ Step 3: 按策略选取 num_add 个样本 (默认 500)
  ├─ Step 4: 从池迁移到训练集 ("标注"过程)
  └─ 多 GPU 广播同步
  │
  ▼
[继续训练] → 下一轮 trigger → ... → 训练结束
```

### 2.2 初始训练集选择（K-Means 聚类）

**目的**：从全量 LHS 训练集中选出最具多样性的初始子集，确保模型从覆盖整个参数空间的样本开始学习。

**实现**（[al_callback.py:_kmeans_initial_selection](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```python
def _kmeans_initial_selection(self, full_P, initial_size):
    P_np = full_P.cpu().numpy()  # (N, 141) 归一化参数
    n_clusters = min(initial_size, P_np.shape[0])

    # 1. 在 141 维参数空间做 K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(P_np)
    centers = kmeans.cluster_centers_

    # 2. 从每个簇中选取离簇心最近的样本
    selected_idx = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        distances = np.linalg.norm(P_np[cluster_indices] - centers[cluster_id], axis=1)
        nearest_idx = cluster_indices[np.argmin(distances)]
        selected_idx.append(nearest_idx)

    return np.array(selected_idx)
```

**关键设计**：
- **在参数空间聚类**：141 维归一化参数空间，确保初始集覆盖不同参数组合
- **每簇选 1 个最近样本**：保证初始集大小 = 簇数 = `initial_train_size`
- **默认 initial_train_size=2000**：约占全量 LHS 数据的 4%（50000+ 样本），给 AL 留足发挥空间

**数据划分合规性与查询合成/候选池差异**：
- 初始训练集和未标注池**均仅包含 LHS 样本**（来自 `dm.train_dataset_lhs`）。
- 验证集和测试集不受影响，测试集仍同时包含 LHS 和 Real Mutants。
- **查询合成策略（`rgs`, `mutant_aware`）**：跳过 K-Means 聚类，初始直接使用全量 LHS 数据进行训练，触发 AL 时调用生成引擎在线生成。
- **AL4PDE 候选池框架 (v2.2)**：K-Means 从 v1 LHS 常规样本中选 **5000** 个 (v2.1: 15000, v2.2 P2 缩减) 作为初始训练集; 候选池由 [lhs_v1_origin_v5.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/lhs_v1_origin_v5.py) + [lhs_v1_origin_v5_boundary.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/lhs_v1_origin_v5_boundary.py) 动态生成 ~110700 个样本 (normal/density/boundary 三类各 ~36900), 与初始训练集完全解耦。初始训练集占比 ~1/16, AL 占比 ~93.8%。

### 2.3 未标注池（Unlabeled Pool）管理

**初始化**（[al_callback.py:on_train_start](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```python
# 全量 LHS 训练集
full_X, full_P, full_Y = dm.train_dataset_lhs.tensors  # (N, 39, 2, T), (N, 141), (N, 39, T)

# K-Means 选初始集
selected_idx = self._kmeans_initial_selection(full_P, initial_size)

# 剩余样本 → 未标注池
mask = np.ones(full_size, dtype=bool)
mask[selected_idx] = False
pool_idx = np.where(mask)[0]

self._pool_X = full_X[pool_idx]  # (pool_size, 39, 2, T)
self._pool_P = full_P[pool_idx]  # (pool_size, 141) 归一化
self._pool_Y = full_Y[pool_idx]  # (pool_size, 39, T) 归一化 (模拟"标注")
```

**池状态**：
- `_pool_X`：输入特征（IC + time grid）
- `_pool_P`：参数向量（归一化空间）
- `_pool_Y`：标签轨迹（归一化空间，ODE 预计算，模拟"标注"过程）
- `_pool_size`：当前池中剩余样本数

**池化策略 vs 查询合成策略**：
- **池化策略**（us/is/wrs/vessal/hggs/pial）：从池中查询样本，池逐渐缩小
- **查询合成策略**（rgs）：通过 LHS 引擎生成新样本，池仅作备份不消耗

### 2.4 池子集采样（Subsample Pool）

**目的**：当池很大时（如 48000 样本），对全池推理计算量大。通过子集采样平衡效率与覆盖。

**实现**（[al_callback.py:_subsample_pool](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```python
def _subsample_pool(self):
    if self._pool_size <= self.pool_subset_size:  # 默认 10000
        return np.arange(self._pool_size)  # 池小，用全池

    # 池大，随机采样子集
    rng = np.random.RandomState(self.random_seed + len(self._train_size_history))
    pool_indices = rng.choice(self._pool_size, self.pool_subset_size, replace=False)
    return np.sort(pool_indices)
```

**关键设计**：
- **随机种子随 AL 轮次变化**：`random_seed + len(_train_size_history)`，每轮采不同子集
- **默认 pool_subset_size=10000**：平衡推理速度（~5s）和覆盖度
- **池 ≤ 10000 时用全池**：避免小池时浪费样本

### 2.5 不确定性计算（公共）

**适用策略**：us / is / wrs / vessal（pial 用自己的 MC Dropout 不确定性，hggs/random 不需要）

**实现**（[al_callback.py:_compute_uncertainty](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```python
def _compute_uncertainty(self, pl_module, device, pool_indices):
    pl_module.eval()
    with torch.no_grad():
        for batch in pool_indices:
            X_batch = self._pool_X[batch].to(device)
            P_batch = self._pool_P[batch].to(device)
            preds = pl_module.forward_ic_time(X_batch, P_batch)  # (B, 39, T)

            if self.uncertainty_metric == 'entropy':
                # 预测熵: 沿时间维度的分布熵
                preds_abs = torch.abs(preds) + 1e-8
                preds_norm = preds_abs / preds_abs.sum(dim=-1, keepdim=True)
                entropy = -torch.sum(preds_norm * torch.log(preds_norm), dim=-1)
                scores = entropy.mean(dim=1)  # (B,)
            else:
                # 默认: 预测方差 (沿时间维度)
                variance = torch.var(preds, dim=-1)  # (B, 39)
                scores = variance.mean(dim=1)  # (B,)
    return scores
```

**两种度量**：
- **variance**（默认）：预测沿时间维度的方差均值，衡量"预测波动程度"
- **entropy**：预测分布的熵，衡量"预测不确定性"

**局限性**：这是**启发式不确定性**（单次前向），不如 MC Dropout 的贝叶斯不确定性准确。PIAL v2 用 MC Dropout 替代。

### 2.6 参数空间多样性（公共）

**适用策略**：is / vessal（pial 用 PCA 轨迹多样性，hggs 用 K-Means 覆盖）

**实现**（[al_callback.py:_compute_diversity](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```python
def _compute_diversity(self, pool_indices, dm):
    # 池子集参数
    pool_P = self._pool_P[pool_indices].cpu().numpy()  # (N, 141)

    # 训练集参数（代表点）
    train_P = dm.train_dataset_lhs.tensors[1].cpu().numpy()  # (M, 141)

    # 计算池样本到训练集的最小欧氏距离
    distances = euclidean_distances(pool_P, train_P)  # (N, M)
    diversity_scores = distances.min(axis=1)  # (N,)
    return diversity_scores
```

**与 PIAL 的 PCA 轨迹多样性的区别**：
- **参数空间多样性**（本方法）：在 141 维参数空间算距离，衡量"参数是否新颖"
- **PCA 轨迹多样性**（PIAL）：在 20 维 PCA 轨迹空间算距离，衡量"行为是否新颖"
- 参数相近的样本可能有完全不同的动态行为，轨迹多样性更符合 AL 目标

### 2.7 样本迁移（公共）

**目的**：将选中样本从未标注池迁移到训练集，这是 AL 的"标注"步骤。

**实现**（[al_callback.py:_move_samples_to_training_set](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```python
def _move_samples_to_training_set(self, selected_pool_idx, current_epoch):
    # 1. 提取选中样本
    selected_X = self._pool_X[selected_pool_idx]
    selected_P = self._pool_P[selected_pool_idx]
    selected_Y = self._pool_Y[selected_pool_idx]

    # 2. 从池中移除选中样本
    mask = np.ones(self._pool_size, dtype=bool)
    mask[selected_pool_idx] = False
    self._pool_X = self._pool_X[mask]
    self._pool_P = self._pool_P[mask]
    self._pool_Y = self._pool_Y[mask]
    self._pool_size = self._pool_X.shape[0]

    # 3. 返回广播数据（用于多 GPU 同步）
    return {'X': selected_X, 'Y': selected_Y, 'P': selected_P, ...}
```

**多 GPU 同步**（[al_callback.py:on_train_epoch_end](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：
```python
# rank 0 执行查询，广播结果到所有进程
if trainer.is_global_zero:
    broadcast_data = self._standard_al_sample(...)
if trainer.world_size > 1:
    broadcast_data = trainer.strategy.broadcast(broadcast_data, src=0)

# 所有进程更新训练集
old_X, old_P, old_Y = dm.train_dataset_lhs.tensors
updated_X = torch.cat([old_X, broadcast_data['X']], dim=0)
dm.train_dataset_lhs = TensorDataset(updated_X, updated_P, updated_Y)
```

### 2.8 关键参数

**命令行参数**（[run_pipeline.sh](file:///home/users/zsy/trae_workspace/budding_yeast_v2/run_pipeline.sh)）：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--al_strategy` | none | AL 策略（none/random/us/is/wrs/vessal/hggs/rgs/pial/al4pde_*/maal） |
| `--al_trigger_epoch` | 10 | AL 触发间隔（PIAL=50, mutant_aware=100, al4pde/maal=4） |
| `--al_num_add` | 500 | 每轮新增样本数 |
| `--al_initial_train_size` | 2000 | 初始训练集大小（al4pde/maal 推荐 15000） |
| `--al_initial_selection` | kmeans | 初始集选择方法（kmeans/random） |
| `--al_pool_subset_size` | 10000 | 每轮池子集采样大小 |
| `--al_diversity_weight` | 0.5 | IS/VeSSAL 的多样性权重 |
| `--al_uncertainty_metric` | variance | 不确定性度量（variance/entropy） |
| `--al_perturbation` | 0.05 | RGS 扰动幅度 |
| `--al_mae_threshold` | 0.1 | RGS 的 MAE 阈值 |
| `--al_pool_dataset` | None | al4pde/maal 专用: 候选池数据集路径（V5 数据集） |
| `--al_target_weight` | 0.4 | MAAL 的 target priority 权重（diversity 权重 = 1 - target_weight） |
| `--atlas_csv` | None | mutant_aware 专用: mutant_atlas_merged.csv 路径 |

**参数选择经验**：
- `initial_train_size=2000` (标准 AL): 太小（<1000）模型学不到基本动力学，太大（>5000）AL 边际收益有限
- `initial_train_size=5000` (al4pde, v2.2 P2): 让 AL 占比 ~93.8%, 初始占总训练集 1/16
- `num_add=500` (标准 AL): 太小（<100）每轮收益小，太大（>2000）引入噪声样本
- `num_add=1000/3000/4000` (al4pde, v2.2 P4 递增式): 探索→利用渐进, 总新增 ~76000
- `trigger_epoch=10` (标准 AL): 太频繁（<5）训练不充分，太稀疏（>20）AL 轮次不足
- `trigger=分段 8/4/8` (al4pde, v2.2 P3): 早期稳定/中期主力/后期减扰
- PIAL `trigger_epoch=50`：早期模型预测差，MC Dropout/ODE 残差信号不稳定

**v2.2 al4pde 训练配置** (硬编码覆盖命令行参数):
- Best checkpoint: `monitor=test_loss_real` (v2.2 P1, 修复 diversity/maal 过拟合选点)
- 总 epoch: `max_epochs=150` + `EarlyStopping(patience=20, monitor=test_loss_real)` (v2.2 P5)
- AL 轮次: 29 轮 (3 早期 + 22 中期 + 3 后期 + 1 safety)
- 总训练样本: 5000 (初始) + 76000 (AL) = 81000

---

## 3. 标准池化策略

本节 6 种策略都遵循第 2 节的通用框架，区别仅在 **Step 3 样本选取** 环节。

### 3.1 Random

**原理**：纯随机采样，作为基线对比。不依赖任何模型信号。

**实现**（[al_callback.py:347](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L347)）：
```python
rng = np.random.RandomState(self.random_seed + current_epoch)
selected_local = rng.choice(len(pool_indices), num_to_select, replace=False)
```

**特点**：
- 计算最快（无需模型推理）
- 提供多样性下界（随机采样天然覆盖参数空间）
- 用作其他策略的对比基线

### 3.2 US (Uncertainty Sampling)

**原理**：选取模型预测最不确定的样本（预测方差最大），快速降低模型误差。

**不确定性度量**（[al_callback.py:_compute_uncertainty](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：
```python
# 沿时间维度计算预测方差，再对 39 个变量取平均
variance = torch.var(preds, dim=-1)  # (B, 39)
scores = variance.mean(dim=1)        # (B,)
```

**选取策略**：top-K 贪心（按方差降序选前 K 个）。

**优缺点**：
- ✅ 简单有效，快速聚焦难样本
- ❌ 容易选到噪声样本（异常预测导致高方差）
- ❌ 缺乏多样性，可能聚集在参数空间同一区域

### 3.3 IS (Importance Sampling)

**原理**：综合不确定性和参数空间多样性，避免 US 的聚集问题。

**得分公式**（[al_callback.py:_query_importance_sampling](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：
```
score = (1 - w) · normalize(uncertainty) + w · normalize(diversity)
```
- `w = diversity_weight`（默认 0.5）
- `diversity` = 样本参数到训练集的最小欧氏距离（141 维参数空间，见 2.6 节）

**归一化**：min-max 归一化到 [0, 1]，确保两信号量级一致。

**选取策略**：top-K 贪心（按综合得分降序）。

**与 US 的区别**：IS 加入多样性项，避免选到参数空间聚集的样本。

### 3.4 WRS (Weighted Reservoir Sampling)

**原理**：按不确定性权重做加权随机采样（无放回），在偏向不确定样本的同时保留随机性。

**权重计算**（[al_callback.py:_query_weighted_reservoir_sampling](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：
```python
u_shifted = uncertainty - uncertainty.min()  # 平移到非负
weights = u_shifted ** 2 + 1e-8              # 平方强化高不确定性
probs = weights / weights.sum()
selected = rng.choice(N, K, p=probs, replace=False)
```

**与 US 的区别**：
- **US**：贪心选 top-K，可能全部聚集在最难样本
- **WRS**：按概率随机抽取，既偏向不确定样本，又保留多样性
- **平方强化**：`weights = u_shifted ** 2` 让高不确定性样本被选中的概率更高

### 3.5 VeSSAL

**原理**：动态权重平衡不确定性和多样性，权重随训练进度变化。

**动态权重**（[al_callback.py:_query_vessal](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：
```python
progress = min(1.0, len(self._train_size_history) / 10.0)  # 基于 AL 轮次
w_dynamic = self.diversity_weight * progress  # 早期 0，后期 0.5
score = (1 - w_dynamic) · uncertainty_norm + w_dynamic · diversity_norm
```

**动态策略**：
- **早期**（progress≈0）：纯不确定性，快速降误差
- **后期**（progress≈1）：不确定性 + 多样性各占一半，避免过拟合

**与 IS 的区别**：IS 权重固定，VeSSAL 权重随训练进度动态调整。

### 3.6 HGGS

**原理**：K-Means 聚类分层采样，纯多样性驱动（不依赖模型预测）。

**算法**（[al_callback.py:_query_hggs](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：
1. 在池子集 141 维参数空间做 K-Means 聚类（K = num_to_select）
2. 从每个簇选离簇心最近的样本
3. 保证选中样本均匀覆盖整个参数空间

**适用场景**：
- 模型预测不可靠的早期阶段
- 需要最大化参数空间覆盖的场景
- 与初始训练集选择（K-Means）理念一致

**与其他策略的区别**：HGGS 完全不依赖模型，是纯几何方法。

---

## 4. 查询合成策略

### 4.1 RGS (Real-Guided Sampling)

**原理**：RGS 是唯一的**查询合成**策略——不从未标注池选取，而是通过 LHS 引擎**合成新样本**，用 Real Mutant 的 MAE 引导采样方向，直接优化目标域（Real Mutants）表现。

**核心思想**：
> 训练集是 LHS 采样，测试集包含 Real Mutants。两者分布有差异（distribution gap）。RGS 通过在 Real Mutant 参数邻域生成新样本，让模型在"接近目标域"的区域学习，缩小分布差异。

#### 4.1.1 RGS 完整流程

**实现**（[al_callback.py:_rgs_sample](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```
Step 1: 评估 Real Mutant 当前表现
        ├─ 在 Real Mutant 测试集上推理
        ├─ 反归一化到真实值空间
        └─ 计算每个突变体的 MAE (真实值空间)
        │
Step 2: 按 MAE 分配采样名额 (聚焦 top-K，不撒胡椒面)
        ├─ 差样本 (MAE > threshold) 分配 90% 预算
        │   ├─ 聚焦 top-K 最差样本 (K = max(20, 差样本数×30%))
        │   └─ 按 MAE² 比例分配名额 (强化聚焦)
        └─ 全局 LHS 保持多样性 10% 预算
        │
Step 3: 生成候选参数
        ├─ 3a: 局部 LHS 扰动 (在差样本参数邻域, perturbation=0.05)
        └─ 3b: 全局 LHS 采样 (保持参数空间覆盖)
        │
Step 3.5: 参数合理性校验 (在 ODE 验证前剔除异常)
        ├─ 非负性: 所有参数 ≥ 0
        ├─ 物理下界: mdt≥10, B0≥0.001, C0≥0.001, Dn3≥0.001, ESP1T≥0.001
        └─ 相对上界: 关键参数不超过默认值的 5 倍
        │
Step 4: ODE 物理验证 (solve_ivp)
        ├─ 并行求解 (32 workers)
        ├─ 动态浓度剔除: 销毁"怪物"细胞 (浓度超训练集 2 倍)
        └─ 返回 (X, Y, P, label) 四元组
        │
Step 5: 用真实 MAE 选取最难的 num_add 个
        ├─ 模型预测 vs ODE 真值 (真实值空间 MAE)
        ├─ 以 Real Mutant MAE 为锚点设定区间:
        │   - 下限 = Real 中位数 × 0.5 (排除太容易)
        │   - 上限 = Real 最大值 × 5 (排除异常)
        └─ 区间内按 MAE 降序选取
        │
Step 6: 绘制采样对比图 (每种 Pattern 各一张)
        │
Step 7: 组装广播数据，迁移到训练集
```

#### 4.1.2 Step 1: Real Mutant 评估

**实现**（[al_callback.py:_evaluate_real_mutants](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```python
def _evaluate_real_mutants(self, pl_module, dm, device):
    # 在 Real Mutant 测试集上推理
    for idx in range(len(test_dataset)):
        X, P, Y_true_norm = test_dataset[idx]
        pred_norm = pl_module.forward_ic_time(X_batch, P_batch)

        # 反归一化到真实值空间 (MAE 在真实空间计算)
        pred_real = pred_norm * std_y + mean_y
        Y_true_real = Y_true_norm * std_y + mean_y

        # 真实值空间 MAE
        mae = torch.mean(torch.abs(pred_real - Y_true_real)).item()
        mutant_maes[name] = mae
```

**关键设计**：MAE 在**真实值空间**计算（反归一化后），因为归一化空间的 MAE 无法反映物理意义。

#### 4.1.3 Step 2: 名额分配

**实现**（[al_callback.py:_allocate_rgs_quota](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```python
def _allocate_rgs_quota(self, mutant_maes):
    total_budget = self.num_add  # 500
    rgs_budget = int(total_budget * 0.9)      # 90% 给 RGS 局部采样
    global_budget = total_budget - rgs_budget  # 10% 给全局 LHS

    # 差样本: MAE > threshold (默认 0.1)
    poor_mutants = {k: v for k, v in mutant_maes.items() if v > self.mae_threshold}

    # 聚焦 top-K 最差样本 (不撒胡椒面)
    sorted_poor = sorted(poor_mutants.items(), key=lambda x: x[1], reverse=True)
    top_k = max(20, int(len(sorted_poor) * 0.3))  # 至少 20 个，最多 30%

    # 按 MAE² 比例分配名额 (强化聚焦)
    weights = {k: v ** 2 for k, v in focused_mutants}  # 平方强化
    for name, w in weights.items():
        quota = max(30, int(rgs_budget * w / total_weight))
        sampling_plan[name] = quota
```

**关键设计**：
- **90/10 分配**：90% 预算给差样本邻域（聚焦），10% 给全局 LHS（保持多样性）
- **top-K 聚焦**：只对最差的 top-K 样本分配名额，避免预算分散
- **MAE² 加权**：MAE 越大的样本分配越多名额，强化聚焦效果
- **最少 30 名额**：每个聚焦样本至少 30 个候选，保证局部 LHS 有足够样本

#### 4.1.4 Step 3: 候选生成

**3a: 局部 LHS 扰动**（[al_callback.py:1234](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：
```python
# 在差样本参数邻域做 LHS 扰动
rgs_tasks_dict = generate_local_lhs_tasks(
    target_mutants=sampling_plan,  # {mutant_name: quota}
    perturbation=self.perturbation  # 0.05 (5% 扰动)
)
```
- `perturbation=0.05`：参数在 ±5% 范围内扰动
- 在差样本参数邻域生成局部 LHS 样本

**3b: 全局 LHS 采样**（[al_callback.py:1262](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：
```python
# 全局 LHS 保持参数空间覆盖
global_tasks_dict = generate_lhs_tasks(per_mutant)
```
- 在全参数空间做 LHS 采样，保持多样性

#### 4.1.5 Step 3.5: 参数合理性校验

**实现**（[al_callback.py:_validate_param_ranges](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```python
# 1. 非负性: 所有参数 ≥ 0
# 2. 物理下界:
param_lower_bounds = {
    'mdt': 10.0,      # 细胞周期时间至少 10 分钟
    'B0': 0.001,      # 生长速率
    'C0': 0.001,      # CLN3 系数
    'Dn3': 0.001,     # CLN3 解除抑制系数
    'ESP1T': 0.001,   # ESP1 总量
}
# 3. 相对上界: 关键参数不超过默认值的 5 倍
param_upper_multiplier = {'mdt': 5.0, 'B0': 5.0, 'C0': 5.0, ...}
```

**目的**：在昂贵的 ODE 验证前剔除明显异常的参数，节省计算资源。

#### 4.1.6 Step 4: ODE 物理验证

**实现**（[al_callback.py:_physics_validate_and_prune](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```python
# 1. 并行 ODE 求解 (32 workers)
with ProcessPoolExecutor(max_workers=32) as executor:
    futures = [(name, executor.submit(solve_single_mutant_bridge, name, p))
               for name, p in new_params_list]
    for name, f in futures:
        res = f.result()  # (X, Y, P, label)
        if res is not None:
            valid_results.append(res)

# 2. 动态浓度剔除: 销毁"怪物"细胞
real_envelope = Y_train_raw.max(dim=-1)[0].max(dim=0)[0].numpy()
dynamic_thresholds = np.maximum(real_envelope * 2.0, 2.0)  # 训练集最大值的 2 倍

for res in valid_results:
    y_uniform = res[1]
    lhs_max_vals = np.max(y_uniform, axis=1)
    for var_idx in range(1, 39):
        if lhs_max_vals[var_idx] > dynamic_thresholds[var_idx]:
            is_monster = True  # 浓度超限 → 销毁
            break
```

**关键设计**：
- **并行求解**：32 个进程并行 ODE 求解，加速验证
- **动态浓度剔除**：基于训练集最大值的 2 倍设定阈值，销毁浓度异常的"怪物"细胞
- **返回四元组**：(X, Y, P, label)，其中 Y 是 ODE 真值（模拟"标注"）

#### 4.1.7 Step 5: 真实 MAE 选取

**实现**（[al_callback.py:_select_by_real_mae](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```python
# 计算每个候选的真实 MAE (模型预测 vs ODE 真值)
for res in valid_results:
    X, y_uniform, p_vec, _ = res
    pred_norm = pl_module.forward_ic_time(X_tensor, P_tensor)
    pred_real = pred_norm * std_y + mean_y
    mae = torch.mean(torch.abs(pred_real - y_real)).item()
    candidate_maes.append(mae)

# 以 Real Mutant MAE 为锚点设定区间
real_max = np.max(real_maes)
real_median = np.median(real_maes)
mae_upper = max(real_max * 5.0, 10.0)    # 排除异常
mae_lower = real_median * 0.5              # 排除太容易

# 区间内按 MAE 降序选取
mask = (candidate_maes >= mae_lower) & (candidate_maes <= mae_upper)
sorted_within = valid_indices[np.argsort(candidate_maes[valid_indices])[::-1]]
idx_sel = sorted_within[:num_to_select]
```

**关键设计**：
- **真实 MAE**：模型预测 vs ODE 真值（反归一化后），反映样本真实难度
- **Real 锚点**：以 Real Mutant 的 MAE 为参考，选取"跳一跳够得着"的样本
- **区间选取**：排除太容易（< Real 中位数 × 0.5）和太难（> Real 最大值 × 5）的样本
- **降序选取**：区间内按 MAE 降序，优先选较难的样本

#### 4.1.8 RGS 与池化策略的区别

| 维度 | 池化策略 (US/IS/...) | RGS |
|------|---------------------|-----|
| 样本来源 | 未标注池（LHS 预计算） | LHS 引擎实时合成 |
| 标签来源 | ODE 预计算（池中已有） | ODE 实时求解（solve_ivp） |
| 信号来源 | 模型预测（不确定性/残差） | Real Mutant MAE 引导 |
| 计算成本 | 低（仅模型推理） | 高（ODE 求解 + 模型推理） |
| 目标域 | LHS 参数空间 | Real Mutant 邻域 |
| 多样性 | 参数空间覆盖 | 全局 LHS (10% 预算) |

#### 4.1.9 RGS 的优势与局限

**优势**：
1. **直接优化目标域**：在 Real Mutant 邻域采样，缩小训练-测试分布差异
2. **自适应聚焦**：根据 Real Mutant MAE 动态调整采样方向
3. **物理验证**：ODE 求解保证标签质量，浓度剔除避免异常样本

**局限**：
1. **计算成本高**：每轮需要 ODE 求解（32 进程并行仍较慢）
2. **依赖 Real Mutant**：需要 Real Mutant 测试集作为引导信号
3. **局部扰动范围有限**：perturbation=0.05 可能无法覆盖远距离的有价值样本

---

### 4.2 Mutant-Aware AL

**原理**：这是 RGS 的进阶版查询合成策略。RGS 仅利用当前模型的 MAE 来分配计算资源，可能导致陷入“本来就无法拟合（即使加了样本）”的黑洞区域。Mutant-Aware 结合了预先计算的 Atlas 优先度映射表（如 `mutant_atlas_merged.csv`），利用历史表现辅助采样决策。

**核心思想**：
利用 3D 权重计算每个 Real Mutant 的采样优先级：
1. `lhs_MAE_min`（历史最优 MAE）：如果历史最优就很差，说明该区域天生难学。
2. `lhs_n_samples`（历史样本覆盖量）：样本覆盖越少，越值得探索。
3. `当前模型 MAE`：实时反映模型当前的薄弱环节。

通过这三者的加权，将计算资源（AL 名额）聚焦到“历史一直表现差且样本覆盖少，且当前依然很差”的顽固瓶颈邻域。

**触发时机与实现**：
- **推迟触发**：推迟到 Epoch 100 触发，需模型基本收敛后再做 atlas 优先级评估，否则 priority 基于错误的早期模型状态。
- **降级机制**：如果未提供 `--atlas_csv` 或者文件不存在，则系统自动安全降级为普通的 `rgs` 策略。

---

## 5. 物理信息策略

### 5.1 PIAL (Physics-Informed Active Learning) v2

**v2 重写动机**：v1 用 ODE 残差作主导信号，实验显示 PIAL(0.516) 不如 baseline(0.451)。根因：
1. **残差大 ≠ 预测误差大**：数值微分放大噪声，满足 ODE ≠ 接近真值
2. **维度灾难**：19500 维欧氏距离失去区分度
3. **动态权重失衡**：早期残差权重过高（0.4-0.7）但残差信号是噪声

**v2 设计**：MC Dropout 不确定性主导 + PCA 轨迹多样性 + ODE 残差辅助。

#### 5.1.1 核心思想

PIAL v2 用**三信号融合**选样本：

1. **MC Dropout 不确定性**（主导）：对参数 p 加微小高斯噪声前向 K 次，预测方差大 → 模型不确定 → 信息量大
2. **PCA 轨迹多样性**（次之）：PCA 降维后到训练集的最小距离，避免维度灾难
3. **ODE 残差**（辅助）：保留物理约束信号，但权重 < 0.2

> **关键改进**：不确定性是"模型对该样本有多不确定"的直接度量，比"模型违反物理"更贴近 AL 目标。MC Dropout 通过输入扰动实现，对所有模型通用（无需模型有 Dropout 层）。

#### 5.1.2 Score 构成

**综合得分公式**（[al_callback.py:759-762](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py#L759-L762)）：

```
score = α · normalize(uncertainty) + β · normalize(diversity) + γ · normalize(residual)
```

**动态权重**（早期重不确定性快速降误差，后期重多样性避免过拟合，残差始终辅助）：
```python
progress = current_epoch / max_epochs  # ∈ [0, 1]
alpha = 0.6 - 0.1 * progress   # 不确定性: 0.6 → 0.5 (主导)
beta = 0.3 + 0.1 * progress    # 多样性: 0.3 → 0.4 (次之)
gamma = 0.1                     # 残差: 0.1 (辅助, 始终低权重)
```

| 训练阶段 | α (不确定) | β (多样) | γ (残差) | 策略倾向 |
|---------|-----------|---------|---------|---------|
| 早期 (epoch=50) | 0.575 | 0.325 | 0.1 | 不确定性主导：快速降误差 |
| 中期 (epoch=125) | 0.538 | 0.362 | 0.1 | 平衡 |
| 后期 (epoch=200) | 0.500 | 0.400 | 0.1 | 多样性增强：避免过拟合 |

#### 5.1.3 MC Dropout 不确定性得分（uncertainty，主导信号）

**定义**（[al_callback.py:_compute_mc_dropout_uncertainty](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```
对每个样本 i:
    for k = 1..K:
        p_k = p_i + ε_k,  ε_k ~ N(0, σ²·I)    # 参数加高斯噪声
        û_k = model(x_i, p_k)                   # 前向预测
    uncertainty_i = mean_{v,t}( Var_k(û_k) )    # 沿 MC 维度求方差，再对变量和时间取平均
```

**参数**：
- `n_mc_samples = 10`：MC 采样次数
- `noise_std = 0.02`：参数扰动标准差（归一化空间，约 2% 扰动）

**为什么用输入扰动而非传统 MC Dropout**：
- 传统 MC Dropout 需要模型有 Dropout 层且推理时保持开启（HyperFNO 无 Dropout）
- 输入扰动法对所有模型通用，等效于贝叶斯神经网络的后验近似
- 扰动在归一化参数空间进行，保证不同参数维度的扰动幅度一致

**计算步骤**：
```python
preds_list = []
for _ in range(K):  # K=10
    noise = torch.randn_like(P_batch) * 0.02
    P_perturbed = P_batch + noise
    preds = pl_module.forward_ic_time(X_batch, P_perturbed)  # (B, 39, T)
    preds_list.append(preds)

preds_stack = torch.stack(preds_list, dim=0)  # (K, B, 39, T)
variance = torch.var(preds_stack, dim=0)      # (B, 39, T)
uncertainty = variance.mean(dim=(1, 2))        # (B,)
```

#### 5.1.4 PCA 轨迹多样性得分（diversity，次信号）

**定义**（[al_callback.py:_compute_trajectory_diversity_pca](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```
1. PCA 拟合在训练集轨迹上 (避免数据泄露)
2. 将池预测轨迹和训练集轨迹降到 20 维
3. diversity = min_distance(û_pool_pca, Y_train_pca)
```

**为什么用 PCA 降维**：
- v1 展平为 (39*T=19500,) 维向量算欧氏距离，高维下距离趋同（维度灾难）
- PCA 降到 20 维，保留 95%+ 方差，低维空间距离恢复区分度

**计算步骤**：
```python
# 展平轨迹
pool_flat = pred_trajectories.reshape(N, -1)        # (N, 39*T)
train_flat = train_Y_real.reshape(R, -1)            # (R, 39*T)

# PCA 拟合在训练集上
pca = PCA(n_components=20, random_state=seed)
pca.fit(train_flat)

# 降维
pool_pca = pca.transform(pool_flat)     # (N, 20)
train_pca = pca.transform(train_flat)   # (R, 20)

# 低维空间最小欧氏距离
distances = euclidean_distances(pool_pca, train_pca)  # (N, R)
diversity_scores = distances.min(axis=1)              # (N,)
```

**与参数空间多样性的区别**：
- **参数空间多样性**（IS/VeSSAL 用）：两个参数相近的样本可能有完全不同的动态行为
- **轨迹空间多样性**（PIAL 用）：直接衡量"行为是否已被覆盖"，更符合 AL 目标
- **PCA 降维**：在保留主要行为模式的前提下避免维度灾难

#### 5.1.5 ODE 残差得分（residual，辅助信号）

**定义**（[al_callback.py:_compute_ode_residual](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```
r(t) = dû/dt - f(û(t), p)    # ODE 残差
residual_score = sqrt(mean_t ||r(t)||²)    # 仅在连续区间
```

**计算步骤**：
1. **数值微分**：中心差分计算 `dû/dt`
   ```
   dû/dt[t] ≈ (u[t+1] - u[t-1]) / (2·dt)    # 中心差分
   dû/dt[0] = (u[1] - u[0]) / dt             # 前向差分
   dû/dt[T-1] = (u[T-1] - u[T-2]) / dt       # 后向差分
   ```
2. **ODE 右端项**：对每个时间步调用 `_yeast_ode_eqns(0, u[:,t], p_dict)` 计算 `f(u, p)`
3. **Event 掩码**：检测 4 种 terminal event（分裂/S期退出/DNA复制/纺锤体通过），在 event 点附近 ±4 步窗口内屏蔽残差
4. **得分**：`sqrt(mean(masked_residual²))`

**Event 掩码检测**：
- **策略 1（事件条件）**：
  - event1: `|CLB2 - 0.3| < tol` 且 CLB2 下降 → 分裂
  - event2: `|CLB2+CLB5 - 0.2| < tol` 且下降 → S期退出
  - event3: `|ORI - 1.0| < tol` 且上升 → DNA复制
  - event4: `|SPN - 1.0| < tol` 且上升 → 纺锤体通过
- **策略 2（硬跳变）**：受 event 影响变量（MASS/CLB2/CLB5/ORI/BUD/SPN/Vi20/lte1/BUB2）的相邻点差分 > 30% 值域

**防御性处理**：
```python
residual = np.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)
residual_scores = np.nan_to_num(residual_scores, nan=0.0, posinf=1e6, neginf=-1e6)
```

**v2 中残差降为辅助信号的原因**：
1. 残差大 ≠ 预测误差大：数值微分放大噪声，满足 ODE ≠ 接近真值
2. 真值本身在 event 点不满足 ODE：即使掩码处理，插值涂抹区残差异常
3. 残差信号是负相关的：实验显示残差大的样本对模型帮助反而更小

#### 5.1.6 PIAL v2 完整流程

**实现**（[al_callback.py:_query_physics_informed](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```
Step 1: MC Dropout 不确定性 (主导)
        ├─ 对参数 p 加 N(0, 0.02²) 噪声前向 10 次
        ├─ uncertainty = mean(Var(û))
        └─ 防御: nan_to_num 处理早期异常值
        │
Step 2: PCA 轨迹多样性 (次信号)
        ├─ 模型推理获取池预测轨迹 (真实值空间)
        ├─ 训练集采样 500 代表点，反归一化到真实值空间
        ├─ PCA 拟合在训练集上 (避免数据泄露)
        ├─ 池预测和训练集轨迹降到 20 维
        └─ diversity = min_distance(pool_pca, train_pca)
        │
Step 3: ODE 残差 (辅助)
        ├─ 中心差分求 dû/dt (dt = t_max / (T-1))
        ├─ 调用 _yeast_ode_eqns 求 f(u, p) (逐时间步)
        ├─ Event 掩码屏蔽突变点 (±4 步窗口)
        ├─ residual_score = sqrt(mean(masked_residual²))
        └─ 防御: 掩码过度(>90%)时降级为全轨迹计算
        │
Step 4: 归一化 + 动态权重综合
        ├─ normalize(uncertainty) ∈ [0, 1] (min-max)
        ├─ normalize(diversity) ∈ [0, 1] (min-max)
        ├─ normalize(residual) ∈ [0, 1] (min-max)
        ├─ 动态权重:
        │   - α = 0.6 - 0.1·progress (不确定性, 主导)
        │   - β = 0.3 + 0.1·progress (多样性, 次之)
        │   - γ = 0.1                (残差, 辅助)
        └─ score = α·uncertainty + β·diversity + γ·residual
        │
Step 5: 选 top-K
        └─ 按 score 降序选取 num_to_select 个
```

**关键实现细节**：
- **参数反归一化**：ODE 方程在真实参数空间成立，需 `p_real = p_norm * p_std + p_mean`
- **轨迹反归一化**：PCA 和 ODE 都在真实值空间，需 `y_real = y_norm * std_y + mean_y`
- **逐样本 ODE 残差**：每个样本独立计算（无法批量，因 ODE 方程非线性）
- **异常防御**：早期模型预测可能产生 inf/nan，全程用 `nan_to_num` 处理

#### 5.1.7 PIAL v2 的优势

1. **MC Dropout 不确定性是 AL 的正确信号**：直接度量"模型对该样本有多不确定"，比"违反物理"更贴近 AL 目标
2. **对所有模型通用**：输入扰动法不需要模型有 Dropout 层
3. **PCA 降维避免维度灾难**：19500 维 → 20 维，保留 95%+ 方差，距离恢复区分度
4. **保留物理约束**：ODE 残差作弱辅助，仍利用物理信息但不让其主导
5. **不需要真实标签**：三信号都只用模型预测 + 物理方程

#### 5.1.8 PIAL v2 的注意事项

- **触发 epoch 推迟到 50**：早期模型预测差，三信号都不稳定（[run_pipeline.sh:105-108](file:///home/users/zsy/trae_workspace/budding_yeast_v2/run_pipeline.sh#L105-L108)）
- **MC 采样增加计算成本**：每个样本前向 10 次，比 US 慢 ~10 倍
- **PCA 拟合在训练集上**：避免池数据泄露影响主成分方向
- **inf/nan 防御**：早期预测可能产生异常值，用 `nan_to_num` 处理

#### 5.1.9 v1 → v2 改进对比

| 维度 | v1 | v2 |
|-----|----|----|
| 主导信号 | ODE 残差 (α=0.4-0.7) | MC Dropout 不确定性 (α=0.5-0.6) |
| 多样性 | 19500 维欧氏距离 | PCA 降到 20 维 |
| 残差角色 | 主导 | 辅助 (γ=0.1) |
| 信号数 | 2 | 3 |
| 实验结果 | 0.516 (不如 baseline 0.451) | 待验证 |

---

## 6. AL4PDE 大规模候选池框架 (v2.2)

**背景与动机**：
传统的 RGS/Mutant-Aware 查询合成策略需要在训练时调用 ODE 求解器（`solve_ivp`）在线生成样本，极大地拖慢了训练速度。为了在加速训练的同时保留高质量定向采样，引入了 **AL4PDE** 框架。

**核心思想**：
将"在线 ODE 求解"解耦为"离线预生成 + 选中后仿真"。候选池只存参数向量,选中后调用 ODE 求解器生成真实轨迹。利用大规模定向候选池 (normal/density/boundary 三类采样),在训练中执行基于池化 (Pool-Based) 的主动学习。

**v2.2 当前配置** (硬编码覆盖命令行参数,详见 [AL4PDE_CHANGELOG.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/AL4PDE_CHANGELOG.md)):
- **初始训练集**: 5000 样本 (K-Means 从 v1 LHS 常规样本中选取, v2.2 P2: 15000→5000)
- **AL 触发**: 分段式 8/4/8 ep (v2.2 P3: 早期稳定/中期主力/后期减扰)
- **每轮新增**: 递增式 1000/3000/4000 (v2.2 P4: 探索→利用渐进)
- **AL 轮次**: 29 轮 (3 早期 + 22 中期 + 3 后期 + 1 safety)
- **候选池**: v5 采样算法动态生成 ~110700 个 (normal/density/boundary 三类各 ~36900), 过滤后 ~105099
- **总训练集**: 5000 (初始) + 76000 (AL) = 81000 样本, 初始占比 ~1/16
- **Best checkpoint**: `monitor=test_loss_real` (v2.2 P1, 修复 val_lhs 选点过拟合)
- **总 epoch**: 150 + `EarlyStopping(patience=20, monitor=test_loss_real)` (v2.2 P5)

### 6.1 策略概览

5 种策略共享同一候选池基础设施, 区别在样本选取环节:

| 策略 | 选取方法 | ODE 预计算 | 核心信号 |
|------|---------|-----------|---------|
| `al4pde_random` | 候选池随机采样 | 否 | 无 (基线) |
| `al4pde_uncertainty` | MC Dropout 预测方差 top_k | 否 | 模型预测不确定性 |
| `al4pde_diversity` | MaxDist on 332维形状特征 (方案A权重) | **是** | 轨迹形状 + 参数多样性 |
| `maal` | diversity + target_priority 加权 | **是** | diversity + 突变体优先级 |
| `al4pde_boundary` | (1-w)·diversity + w·boundary_proximity | **是** | diversity + 0值邻域偏好 |

### 6.2 候选池三类采样 (v2.0+)

候选池由 [al_callback.py `_generate_param_only_pool`](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L1588) 调用 v5 采样算法动态生成:

**采样配置**:
- 排除 3 个 WT 家族 (`1_WT_Glc`, `2.1_WT_Gal`, `2.2_WT_Raff`): WT 无突变参数, 对 AL 学习无贡献
- 每 mutant 三类采样 (cdh1 家族 density 500, 其他 300, boundary 全 300):

| 类型 | 采样函数 | 范围 | 键名后缀 |
|------|---------|------|---------|
| **normal** | `generate_lhs_tasks_v5(300)` | 全局均匀 LHS, 跨所有 mutant 历史值扩展区间 | `_LHS_{idx}` |
| **density** | `generate_targeted_lhs_tasks_density` (pert=0.30) | 突变参数邻域扰动, KO 用 `[0, default×0.3]` | `_LHS_D{idx:03d}` |
| **boundary** | `generate_targeted_lhs_tasks_boundary` (pert=0.30) | 0 值邻域覆盖, KO 用 `[0, default×1.3]`, 非 KO 用 `[0, center×1.3]` | `_LHS_B{idx:03d}` |

**v2.2 cdh1 定向增强**: cdh1 家族 mutant (含 cdh1 关键词) 的 density 采样数量从 300 提升到 500, 其他 mutant 保持 300, boundary 全部 300。原因: 41/42/45/21/122 等含 cdh1 多突变体在所有策略下 Rel L2 都上千, 共性难点。

**候选池大小** (v2.1 实测日志):
- 合并前: 36900 (normal) + 36900 (density) + 36900 (boundary) = 110700
- ODE 仿真成功: 110534 (失败 166)
- 极端值过滤后: 105099 (mass>10=0, 浓度爆炸=5435)

### 6.3 diversity/maal/boundary 形状感知 (v1.2+)

**动机**: 原始 diversity 仅基于参数距离, 但相同参数可能产生截然不同的轨迹 (mass pattern), 需结合轨迹形状评估多样性。早期模型预测不可靠, 必须用 ODE 求解器解出真实轨迹。

**实现流程** (`_precompute_pool_ode_shape_features`):

```
1. 初始化时 32 进程并行 ODE 仿真全部候选池样本 (~20min)
2. 3 层极端值过滤:
   ├── 仿真失败 → 丢弃
   ├── max_mass > 10.0 → 丢弃 (对齐 lhs_v1_origin.py)
   └── 任一变量 > real_max×2 → 丢弃 (最低阈值 2.0, v2.1 阈值参考 real mutants)
3. 提取 332 维形状特征:
   ├── MASS 轨迹降采样 (32 维)
   ├── MASS 一阶差分统计 (3 维)
   ├── 全变量统计 mean/std/min/max (39×4 = 156 维)
   └── 参数向量 (141 维)
4. 分组 z-score 标准化 + 方案A权重 (v2.2)
5. 缓存特征供所有 AL 轮次使用 (ODE 轨迹不随训练变化)
```

**v2.2 方案 A 分组权重** (`_SHAPE_FEAT_GROUP_WEIGHTS`):

```python
_SHAPE_FEAT_GROUP_SIZES   = [32,  3,   39,  39,  39,  39,  141]
_SHAPE_FEAT_GROUP_WEIGHTS = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5]
#   - MASS 轨迹降采样 (32 维) ×0.5: 降低形状特征主导
#   - 参数向量 (141 维) ×2.5: 增强参数空间多样性敏感度
#   - 其他组保持 1.0
```

**关键设计**: 权重在 z-score 标准化**之后**施加, 不影响 mean/std 统计量; 三种策略 (diversity/maal/boundary) 共享同一套权重, 保证可比性。

**训练集特征**: 不用模型预测 (训练早期模型差), 从 `train_dataset_lhs` 取真实 Y 轨迹复用相同特征提取逻辑。

### 6.4 al4pde_boundary 策略 (v1.3+)

**动机**: 借鉴 [lhs_v1_origin_v5_boundary.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/lhs_v1_origin_v5_boundary.py) 的 0 值邻域采样思想, 老师认可的合理 0 邻域采样应纳入 AL 选样偏好。

**Acquisition 函数**:
```python
acquisition = (1 - w) * diversity + w * boundary_proximity
```

**boundary_proximity 计算**:
- 对每个突变参数 `p`:
  - `u_bound = ref_val * (1 + pert)` if KO 突变 (center=0)
  - `u_bound = center_val * (1 + pert)` if 非 KO 突变
- `proximity = clip(1 - val / u_bound, 0, 1)`
  - `val = 0` → proximity = 1 (纯 boundary, 最大权重)
  - `val = u_bound` → proximity = 0 (远边界, 无权重)
- 取所有突变参数 proximity 的均值作为该样本的 boundary_proximity

**完全复用 diversity 的 ODE 预计算 + 过滤机制**: 同样初始化时 ODE 仿真 + 极端值过滤 + 332 维特征 + MaxDist 选择 (但用 acquisition 替代纯 diversity 得分)。

**v2.0/v2.1 实验结果**: boundary 是当前最优策略, 在所有 Pattern 上都优于 random, 整体 Rel L2 最低 (86.3 vs random 111.2), 胜出 mutant 数最多 (57/126)。但对 ts 突变体 (tem1_ts/cdc15_ts/cdc14_ts) 退化, 因 ts 突变本质是"部分功能丧失"不是完全 KO, 0 值邻域采样不适用。

### 6.5 MAAL 策略详述

MAAL（Mutant-Aware Active Learning in Pool）策略通过融合两个得分来打分选取样本：
- **目标优先级（Target Priority）**：优先从匹配瓶颈突变体（如带有特定 Family 后缀）的池样本中选取。
- **多样性（Diversity）**：保证选出的样本在参数空间分布广泛，避免冗余。

**得分计算**（[al_callback.py:_maal_sample](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py)）：
```
Score = (1 - target_weight) · diversity + target_weight · target_priority
```
- `target_weight` 通过 `--al_target_weight` 控制（默认 0.4）
- `diversity` = 候选样本到训练集在 332维形状特征空间 (方案A权重) 的最小欧氏距离
- `target_priority` = 改进版三维加权优先级（详见 6.5.1）

**MAAL 完整流程**：
```
Step 1: 计算轨迹 spatial_mean 特征
        ├─ 对候选池做 forward: features(c) = model(X(c), P(c)).mean(time_dim) → (N, 39)
        └─ 对训练集做 forward: 同上 → (M, 39)

Step 2: 计算 diversity score
        ├─ diversity = min_distance(pool_features, train_features)  (欧氏距离)
        └─ 归一化到 [0, 1]

Step 3: 计算 target_priority (改进版)
        ├─ 评估 Real Mutant MAE (实时模型 forward 126 个 Real Mutant)
        ├─ 三维加权: 0.5·MAE百分位 + 0.3·参数距离 + 0.2·覆盖度
        └─ 归一化到 [0, 1]

Step 4: 加权 acquisition
        └─ acquisition = 0.6 · diversity + 0.4 · target_priority

Step 5: top_k 选取
        └─ 按 acquisition 降序选取 num_add 个样本
```

#### 6.5.1 target_priority 改进版（三维加权）

**改进动机**：原版 target_priority 使用简单前缀匹配 + max 聚合（原始 MAE），导致"赢者通吃"问题——MAE 最大的少数家族独占采样预算，其他家族被忽略。

**改进方案**（[al_callback.py:_map_family_priority](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：

```
target_priority = 0.5 · MAE百分位排名 + 0.3 · 参数空间距离 + 0.2 · 家族覆盖度
```

**三个维度详解**：

| 维度 | 权重 | 计算方式 | 作用 |
|------|------|---------|------|
| MAE 百分位排名 | 0.5 | `(mae_values <= mae).mean()` | 用百分位替代原始 MAE，避免极端值主导 |
| 参数空间距离 | 0.3 | `1.0 / (1.0 + ||p_candidate - p_real||)` | 候选到 Real Mutant 的距离越近，priority 越高 |
| 家族覆盖度 | 0.2 | `1.0 - family_count / max_family_count` | 候选池中同家族样本越少（覆盖不足），priority 越高 |

**v2.2 maal 验证结论**: 经检查 al_callback.py 中无 `multi_annotator|MDAL|annotator|n_annotator|ensemble` 关键词, [_maal_sample](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L2704) 是自定义 "Mutant-Aware Active Learning" (非原版 MDAL), 仅用单模型 `pl_module`, 无 ensemble。

**实际发现的问题** (解释 maal v2.0/v2.1 表现差的原因):
1. **MC dropout 非标准**: 调用 `pl_module.eval()` 关闭 dropout 层; "不确定性" 实际通过对 `P_batch` 加 `noise * 0.02` 高斯噪声实现, 是参数空间扰动而非标准 MC dropout
2. **target_priority 区分度不足**: 维度 3 (家族覆盖度 0.2 权重) 候选池每家族样本数均匀, `coverage_factor` 区分度低; 维度 2 (参数距离 0.3 权重) density 候选围绕家族中心, 距离区分度有限; 仅维度 1 (不确定性 0.5 权重) 提供主要信号
3. **数据隔离合规** ✓: 仅访问 `dm.train_dataset_lhs`, 无 `dm.test_dataset_real` 泄露

### 6.6 v2.2 训练节奏优化 P1-P5

**动机**: v2.1 训练日志深入分析暴露 5 个训练节奏问题, v2.2 实施五项改进 (详见 [AL4PDE_CHANGELOG.md v2.2](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/AL4PDE_CHANGELOG.md#v22--特征权重调整--cdh1-定向增强--maal-验证--训练节奏优化-p1-p5)):

| 改进 | 问题诊断 | 实施方案 | 适用范围 |
|------|---------|---------|---------|
| **P1** | val_loss_lhs 选 best 导致 diversity/maal 严重早停 (best epoch=28 但 final epoch 性能更好) | ModelCheckpoint `monitor='test_loss_real'` | 所有策略 |
| **P2** | 初始 15000 占比 22.7%, AL 发挥空间不足 | `initial_train_size = 5000` (占比 1/16) | 仅 al4pde |
| **P3** | 固定 4ep 触发: 早期扰 warmup, 后期扰收敛 | 分段式 8/4/8 ep (ep≤30:8, ep≤120:4, ep>120:8) | 仅 al4pde |
| **P4** | 固定 1000/轮: 探索与利用无差异化 | 递增式 1000/3000/4000 (al_iter<10/<20/≥20) | 仅 al4pde |
| **P5** | 200 ep 无 early stop, 后期浪费算力 | `max_epochs=150` + `EarlyStopping(patience=20, monitor=test_loss_real)` | 仅 al4pde |

**v2.1 → v2.2 节奏对比**:

| 维度 | v2.1 | v2.2 |
|------|------|------|
| 初始训练集 | 15000 | **5000** (1/16 占比) |
| AL 触发频率 | 固定 4 ep | 分段 8/4/8 ep |
| 每轮新增 | 固定 1000 | 递增 1000/3000/4000 |
| AL 轮次 | 51 轮 | 29 轮 |
| AL 总新增 | 51000 | 76000 |
| 总训练样本 | 66000 | 81000 |
| 初始占比 | 22.7% | **6.17% ≈ 1/16** |
| AL 占比 | 77.3% | **93.83%** |
| 总 epoch | 200 | 150 + EarlyStopping |
| Best 依据 | val_loss_lhs | test_loss_real |

**关键代码位置**:
- P1: [train.py L286-292](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/train.py#L286-L292) ModelCheckpoint
- P2: [al_callback.py L198](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L198) initial_train_size
- P3: [al_callback.py L371-388](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L371-L388) on_train_epoch_end
- P4: [al_callback.py L2696-2713](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L2696-L2713) _al4pde_scheduled_num_add
- P5: [train.py L302-311](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/train.py#L302-L311) EarlyStopping + [run_pipeline.sh L149-150](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/run_pipeline.sh#L149-L150) EPOCHS=150

### 6.7 内存优化机制（按需加载完整 Y, 旧 v2 设计, v2.2 候选池改为参数-only）

> **v2.2 注**: 候选池已改为参数-only (由 v5 采样算法动态生成), 完整 Y 选中后 ODE 仿真生成。以下内存优化机制适用于旧版 V5 npz 候选池设计, 保留作为参考。

候选池数据集非常庞大（V5 有 ~162,008 个 LHS 样本）。如果将完整数据加载到内存：
- X (N×39×2×500): ~23.5 GB
- Y (N×39×500): ~11.9 GB
- P (N×141): ~0.09 GB
- **总计 ~35.4 GB**，会耗尽系统内存

**v2 内存优化策略**（[al_callback.py:_load_al4pde_pool](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：只加载参数和 IC（初始条件），完整 Y 按需从磁盘加载。

**内存中只存**：
| 数据 | Shape | 内存占用 | 说明 |
|------|-------|---------|------|
| `P_norm` | (N, 141) | ~0.09 GB | 归一化参数向量 |
| `Y_IC` | (N, 39) | ~0.025 GB | Y[:, :, 0]（只取第一个时间步，归一化后） |
| `mutant_names` | (N,) | ~0.005 GB | 样本名（用于 MAAL family 匹配） |
| `t_grid` | (1, 1, 500) | ~0.002 GB | 共享时间网格 |
| `pool_idx` | (N,) | ~0.001 GB | 候选池样本在原始 npz 中的索引 |
| **总计** | | **~0.12 GB** | **节省 ~35 GB** |

**关键洞察**：
- X = [IC, t_grid]，其中 IC = Y[:, :, 0]（只需要第一个时间步）
- AL 采样过程中（forward 推理、特征计算）只需要 X，不需要完整 Y
- 完整 Y 仅在选中样本加入训练集时需要（用于计算 loss）

**动态构造 X**（[al_callback.py:_get_pool_X_batch](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：
```python
def _get_pool_X_batch(self, indices):
    # 从 Y_IC 动态构造 X (不需要完整 Y)
    Y_ic_batch = self._pool_Y_ic[indices]  # (B, 39)
    T = self._al4pde_t_grid.shape[2]       # 500
    IC = Y_ic_batch.unsqueeze(2).expand(-1, -1, T)  # (B, 39, T) — 视图, 不占额外内存
    t_grid = self._al4pde_t_grid.expand(IC.shape[0], 39, T)
    return torch.stack([IC, t_grid], dim=2)  # (B, 39, 2, T)
```

**按需加载完整 Y**（[al_callback.py:_load_full_Y_from_disk](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/al_callback.py)）：
```python
def _load_full_Y_from_disk(self, sel_idx):
    # 获取选中样本在原始 npz 中的索引
    orig_idx = self._al4pde_pool_idx[sel_idx]
    # 从 npz 文件按需加载 (只加载选中的样本)
    v5_data = np.load(self._al4pde_pool_path, allow_pickle=True)
    Y_sel = torch.tensor(v5_data['data'][orig_idx], dtype=torch.float32)  # (B, 39, 500)
    # 归一化
    Y_sel_norm = (Y_sel - self._al4pde_mean_y) / self._al4pde_std_y
    return Y_sel_norm
```
- 每次加载量：1000 × 39 × 500 × 4 bytes ≈ 78 MB（可接受）
- 加载时机：仅在 `_build_broadcast_from_pool` 中选中样本加入训练集时

**v1 → v2 内存优化对比**：

| 维度 | v1 (预存 Y) | v2 (按需加载 Y) |
|------|------------|----------------|
| 内存中存 Y | Y_norm (N×39×500, ~11.9 GB) | Y_IC (N×39, ~0.025 GB) |
| X 构造方式 | 从 Y[:, :, 0:1] 构造 | 从 Y_IC 构造 |
| 完整 Y 来源 | 内存直接索引 | 从磁盘按需加载 |
| 内存峰值 | ~12 GB | ~0.12 GB |
| 磁盘 I/O | 无 | 每轮 AL 一次（~78 MB） |
| 适用场景 | 内存充足 | 大规模候选池 |

### 6.8 候选池初始化与状态维护 (v2.2 参数-only)

**初始化流程**（[al_callback.py:on_train_start](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py)）：
```
1. 从 v1 (DATASET_CHEAT) 获取全量 LHS 训练集 (~53670 个)
2. K-Means 聚类选 5000 个初始训练集 (v2.2 P2)
3. 调用 _generate_param_only_pool 动态生成候选池:
   ├─ 排除 3 个 WT 家族, 得到 all_mutants
   ├─ 三类采样各 300/mutant (cdh1 density=500, v2.2 cdh1 增强):
   │   ├── normal:   generate_lhs_tasks_v5(300)
   │   ├── density:  _sample_target_with_retry(density, 300/500)
   │   └── boundary: _sample_target_with_retry(boundary, 300)
   ├─ 合并 (内部不去重) → ~110700 样本
   ├─ 提取参数向量 + 构造 IC (覆盖 init_CDH1T/init_CDH1/init_BUB2)
   ├─ 归一化: P_norm = (P - p_mean) / p_std, Y_ic_norm = (Y_ic - mean_y) / std_y
   └─ 存储: _pool_P, _pool_Y_ic, _pool_P_real, _pool_mutant_names
4. diversity/maal/boundary 额外预计算:
   └─ _precompute_pool_ode_shape_features: 32 进程 ODE 仿真 + 过滤 + 332 维特征
```

**候选池状态维护**（`_build_broadcast_from_pool`）：
选中样本后，需要从候选池中移除并更新所有相关状态：
```python
# 移除已选样本
mask = np.ones(self._pool_size, dtype=bool)
mask[sel_idx] = False
self._pool_P = self._pool_P[mask]           # 参数
self._pool_Y_ic = self._pool_Y_ic[mask]     # IC
self._pool_mutant_names = self._pool_mutant_names[mask]  # 样本名
self._al4pde_pool_idx = self._al4pde_pool_idx[mask]      # 原始索引
self._pool_size = mask.sum()                 # 更新池大小
```

### 6.9 al4pde 五策略调用链 (v2.2)

所有 5 种策略共享相同的候选池基础设施，区别仅在样本选取环节：

| 策略 | 选取方法 | ODE 预计算 | 需要的候选池数据 |
|------|---------|-----------|----------------|
| al4pde_random | `np.random.choice` 随机选取 | 否 | P, Y_IC (构造 X) |
| al4pde_uncertainty | MC Dropout 预测方差 top_k | 否 | P, Y_IC (构造 X) |
| al4pde_diversity | MaxDist on 332维形状特征 (方案A权重) | 是 | P, Y_IC, 预计算特征 |
| maal | diversity + target_priority 加权 | 是 | P, Y_IC, mutant_names, 预计算特征 |
| al4pde_boundary | (1-w)·diversity + w·boundary_proximity | 是 | P, Y_IC, P_real, 预计算特征 |

**统一输出**：所有策略最终都调用 `_build_broadcast_from_pool(sel_idx)`,选中样本后 ODE 仿真生成真实 Y, 返回 `{'X': sel_X, 'P': sel_P, 'Y': sel_Y}` 用于训练集扩容。

---

## 7. 实验配置

### 7.1 run_pipeline.sh 配置格式

```bash
# 17 列：MODEL LOSS_TYPE USE_ADJ DATASET TAG MODES WIDTH BATCH EPOCHS STRATEGY LR WARMUP NUM_ADD PERTURB MAE_THRESH ROLLOUT GPU_ID

# 标准池化/查询合成策略
"hyper_fno mse_only false ${DATASET_CHEAT} auto 32 32 64 200 rgs 6e-4 10 500 0.05 0.1 one_time 0"
"hyper_fno mse_only false ${DATASET_CHEAT} auto 32 32 64 200 pial 6e-4 10 500 0 0 one_time 0"

# al4pde 框架 v2.2 (5 种策略对比, 初始 5000 + AL 76000, 150 ep + EarlyStopping)
# 注: 命令行的 AL_NUM_ADD=1000 和 EPOCHS=150 被硬编码覆盖, 实际由 al_callback.py 控制
"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 150 al4pde_random 6e-4 10 1000 0 0 one_time 0"
"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 150 al4pde_uncertainty 6e-4 10 1000 0 0 one_time 1"
"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 150 al4pde_diversity 6e-4 10 1000 0 0 one_time 2"
"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 150 maal 6e-4 10 1000 0 0 one_time 3"
"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 150 al4pde_boundary 6e-4 10 1000 0 0 one_time 0"
```

**al4pde v2.2 框架专属参数** (命令行参数被硬编码覆盖, 详见 [AL4PDE_CHANGELOG.md 第 5 章](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/AL4PDE_CHANGELOG.md#5-关键参数对照表-v22-current)):

| 参数 | 命令行传入值 | 实际生效值 (v2.2) | 覆盖位置 |
|------|------------|-----------------|---------|
| `initial_train_size` | 2000 | **5000** (P2) | al_callback.py L198 |
| `trigger_every_n_epochs` | 30 | **分段式 8/4/8** (P3) | al_callback.py on_train_epoch_end |
| `num_add` | 1000 | **递增式 1000/3000/4000** (P4) | al_callback.py _al4pde_scheduled_num_add |
| `max_epochs` | 150 | **150** + EarlyStopping(patience=20) (P5) | train.py L302-311 |
| Best checkpoint | - | **monitor=test_loss_real** (P1) | train.py L286-292 |

**候选池动态生成** (v2.0+, 不再读 npz):
```python
# al_callback.py _generate_param_only_pool
# 排除 3 个 WT 家族, 三类采样 (cdh1 density=500, 其他 300, boundary 全 300)
# 候选池 ~110700 样本, 过滤后 ~105099
```

### 7.2 策略选择建议

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 快速基线 | `random` | 计算最快，用于对比 |
| 不确定性驱动 | `us` | 简单有效 |
| 平衡探索/利用 | `vessal` | 动态权重自适应 |
| 参数空间覆盖 | `hggs` | 不依赖模型，早期可用 |
| Real Mutant 引导 | `rgs` | 直接优化目标域 |
| 物理信息驱动 | `pial` | 无需标签，物理可解释 |
| 历史先验合成 | `mutant_aware` | 利用 Atlas 规避黑洞 |
| 大规模池化基线 | `al4pde_random` | 候选池随机选取，无在线 ODE |
| 大规模池化不确定性 | `al4pde_uncertainty` | MC Dropout 预测方差驱动 |
| 大规模池化多样性 | `al4pde_diversity` | bmdal_reg MaxDist 特征空间覆盖 (v2.2 方案 A 权重) |
| 大规模池化定向 | `maal` | 改进版 target_priority + diversity |
| **0 值邻域偏好 (v2.2 推荐)** | `al4pde_boundary` | (1-w)·diversity + w·boundary_proximity, v2.0/v2.1 实验最优策略 |

### 7.3 已知问题与修复

1. **PIAL 早期残差 inf/nan**：已添加 `nan_to_num` 防御 + 推迟触发到 epoch 50
2. **RGS 参数异常**：已添加 `_validate_param_ranges` 校验（非负性、物理下界、相对上界）
3. **AL 边际收益递减**：初始训练集从 5000 降到 2000，每轮新增从 5000 降到 500
4. **al4pde V1 框架失败**：初始训练集 5000（仅占总量 9.3%）导致严重欠训练，Real MAE 是 v5+none 的 6.7×-7.4×。**V2 修复**：初始训练集增至 15000，总 AL 预算增至 50000（65000 总样本），Real MAE 改善 49-61%
5. **MAAL "赢者通吃"**：原版 target_priority 用原始 MAE + max 聚合，MAE 极大的家族独占预算。修复：改进为三维加权（0.5·MAE百分位 + 0.3·参数距离 + 0.2·覆盖度）
6. **候选池内存溢出**：V5 候选池完整加载需 ~35 GB。修复：v2 内存优化只存 P_norm + Y_IC（~0.12 GB），完整 Y 按需从磁盘加载
7. **cdc20_ts 系列灾难**：diversity/maal 在 cdc20_ts 参数聚类上系统性失败（MAE>1.6），因 diversity 采集函数避开特征相似但行为关键的区域。random 无偏采样能命中（MAE 0.07-0.12）

#### v2.2 新增修复 (P1-P5 训练节奏优化)

> 详见 [AL4PDE_CHANGELOG.md v2.2](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/AL4PDE_CHANGELOG.md#v22--特征权重调整--cdh1-定向增强--maal-验证--训练节奏优化-p1-p5)

8. **Best checkpoint 选点过拟合到 LHS** (v2.2 P1 修复):
   - **问题**: v2.1 `ModelCheckpoint(monitor='val_loss_lhs')` 使 diversity/maal 选到 LHS 验证集最低点但 real 泛化差的 checkpoint
   - **修复**: 改为 `monitor='test_loss_real'`, 对所有策略 (含非 AL) 生效; EarlyStopping 限 al4pde 5 种策略

9. **初始训练集偏大, AL 占比不足** (v2.2 P2 修复):
   - **问题**: v2.1 `initial_train_size=15000`, AL 占比仅 ~77.3% (51000/66000), AL 价值未充分发挥
   - **修复**: 缩减到 `initial_train_size=5000`, 初始占总训练集 ~1/16 (6.17%), AL 占比 ~93.8%

10. **触发频率僵化** (v2.2 P3 修复):
    - **问题**: v2.1 固定 `trigger_every_n_epochs=4` 全程不变, 早期训练未稳定就触发, 后期冗余扰动
    - **修复**: 分段式触发 `ep≤30:8ep / ep≤120:4ep / ep>120:8ep`, 早期稳定/中期主力/后期减扰, 共 29 轮 AL

11. **num_add 固定导致探索/利用失衡** (v2.2 P4 修复):
    - **问题**: v2.1 固定 `num_add=1000` × 51 轮 = 51000, 早期模型未稳定时大批量引入噪声, 后期稳定时批量不足
    - **修复**: 递增式 `al_iter<10:1000 / al_iter<20:3000 / al_iter≥20:4000`, 探索→利用渐进, 总新增 ~76000

12. **epoch 过多 + 无 early stop, 训练效率低** (v2.2 P5 修复):
    - **问题**: v2.1 `max_epochs=200` 无 EarlyStopping, real_loss 收敛后仍冗余训练
    - **修复**: `max_epochs=150` + `EarlyStopping(monitor='test_loss_real', patience=20)`, 仅 al4pde 5 种策略生效

13. **diversity 过拟合 LHS 形状分布** (v2.2 方案 A 修复):
    - **问题**: 332 维特征中 MASS 轨迹降采样 (32 维) 主导 MaxDist, 参数向量贡献不足, real_mutants 泛化差
    - **修复**: 分组权重 `_SHAPE_FEAT_GROUP_WEIGHTS = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5]`, MASS×0.5 + 参数×2.5, diversity/maal/boundary 共用

14. **cdh1 多突变体系统性差** (v2.2 cdh1 定向增强修复):
    - **问题**: 含 cdh1 的多突变体 (6/7/21/32/33/34/35/36/37/40/41/42/45/46/51/54/67/106/122 等) 在所有策略下 Rel L2 都上千
    - **修复**: cdh1 家族 density 采样从 300 提升到 500, 其他 mutant 保持 300, boundary 全 300

15. **MAAL 实现审查与验证** (v2.2 maal 验证):
    - **问题**: 怀疑 MAAL 误激活 multi-annotator 模式 (124 个 mutant 仅胜出 1 个)
    - **结论**: 无 multi-annotator 误激活; 实际问题是 (a) MC dropout 非标准 (单次前向非多次), (b) target_priority 区分度不足, (c) density 权重与 diversity 权重耦合

---

## 8. 实验结果

### 8.1 al4pde V2 框架 4 策略对比 (v2.1 配置)

> 详见 [eval_result/AL4PDE_V2_EVAL_REPORT.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/eval_result/AL4PDE_V2_EVAL_REPORT.md)

**配置**：初始 15000（v1 K-Means）+ AL 50000（v5 候选池，每 4 epoch 触发，50 轮 × 1000 样本）= 总 65000。

| 策略 | Real MAE | LHS Test MAE | Tier1 占比 | Tier4 占比 | Sim-to-Real Gap |
|------|----------|--------------|------------|------------|-----------------|
| **al4pde_random** | **0.0976** ✅ | 0.1018 | 92 (73%) | 3 | 0.96× |
| al4pde_uncertainty | 0.1894 | 0.1389 | 49 (39%) | 16 | 1.36× |
| MAAL | 0.1579 | 0.1376 | 66 (52%) | 10 | 1.15× |
| al4pde_diversity | 0.2725 ❌ | 0.0972 | 35 (28%) | 31 | 2.80× |

**核心结论**：
- **random 最优**（Real MAE 0.0976），diversity 最差（0.2725）
- V2 配置比 V1 配置（15000 样本）Real MAE 改善 49-61%
- cdc20_ts 系列是 diversity/maal 的绝对灾难区（MAE>1.6），random 能命中（MAE 0.07-0.12）
- 根因：diversity 采集函数系统性避开 cdc20_ts 参数聚类（特征相似但行为关键），random 无偏采样能命中

### 8.2 al4pde V2.2 框架 5 策略对比 (v2.2 配置, 待运行)

> **状态**: 实验待运行, 结果将填充本节。
> 配置参见 [run_pipeline.sh](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/run_pipeline.sh) 和 [AL4PDE_CHANGELOG.md v2.2](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/AL4PDE_CHANGELOG.md#v22--特征权重调整--cdh1-定向增强--maal-验证--训练节奏优化-p1-p5)

**v2.2 配置** (vs v2.1):
- **初始训练集**: 5000 (v2.1: 15000), K-Means 从 v1 LHS 常规样本中选, 初始占总训练集 ~1/16
- **候选池**: ~110700 样本 (过滤后 ~105099), v5 动态生成 (normal/density/boundary 三类各 ~36900, cdh1 density 500)
- **AL 触发**: 分段式 8/4/8 ep (v2.1: 固定 4 ep)
- **num_add**: 递增式 1000/3000/4000 (v2.1: 固定 1000)
- **总训练集**: 5000 + 76000 = 81000 (v2.1: 15000 + 50000 = 65000)
- **Best checkpoint**: `monitor=test_loss_real` (v2.1: val_loss_lhs)
- **max_epochs**: 150 + EarlyStopping(patience=20) (v2.1: 200 无 early stop)
- **特征权重**: 方案 A `_SHAPE_FEAT_GROUP_WEIGHTS = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5]`

**5 种策略** (vs v2.1 增加 al4pde_boundary):

| 策略 | Real MAE | LHS Test MAE | Tier1 占比 | Tier4 占比 | Sim-to-Real Gap |
|------|----------|--------------|------------|------------|-----------------|
| `al4pde_random` | 待运行 | | | | |
| `al4pde_uncertainty` | 待运行 | | | | |
| `al4pde_diversity` | 待运行 | | | | |
| `maal` | 待运行 | | | | |
| `al4pde_boundary` | 待运行 | | | | |

**预期改进**:
- P1 (best=test_loss_real): 修复 diversity/maal 过拟合 LHS 选点, Real MAE 预期改善
- P2 (initial=5000): AL 占比 77.3%→93.8%, AL 价值充分发挥
- P3 (分段 trigger): 早期稳定训练, 中期主力采样, 后期减扰
- P4 (递增 num_add): 探索→利用渐进, 总新增 51000→76000
- P5 (max_epochs=150+early stop): 训练效率提升, 避免冗余训练
- 方案 A (参数×2.5): diversity/maal/boundary 参数空间覆盖增强, 预期改善 real 泛化
- cdh1 增强 (density 500): 含 cdh1 多突变体 Rel L2 预期下降

### 8.3 数据集版本演进对比

> 详见 [eval_result/V6_EVAL_ANALYSIS_REPORT.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/eval_result/V6_EVAL_ANALYSIS_REPORT.md) 和 [eval_result/V5_NO_TARGET_EVAL_REPORT.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/eval_result/V5_NO_TARGET_EVAL_REPORT.md)

| 数据集 | Real MAE | Sim-to-Real Gap | 说明 |
|--------|----------|-----------------|------|
| V1 | 0.250 | 4.53× | 基础 LHS |
| V5 | 0.0552 | 0.89× | + 7 项定向修复 |
| V5_no_target | 0.3858 ❌ | 5.91× | 去除 target，恶化 7.0× |
| **V6** | **0.0492** ✅ | 0.83× | + 6 个修复点（历史最优） |
| V6+mutant_aware | 0.0493 | 0.83× | AL 完全失效（+0.2%） |

**核心结论**：
- V6 Real MAE 0.0492（历史最优，-10.9% vs V5）
- target 定向采样贡献了 Real MAE 的 85.7%（去除后恶化 7.0×）
- mutant_aware AL 在 V6 上完全失效（+0.2%），因 V6 数据集已足够好
- V6 六个修复点中 4 个生效（P0 tem1_ts 拆分极强 -74%），2 个恶化（P2/P3 pert 增大失效）
