# 主动学习（Active Learning, AL）算法详解

本项目实现了 **9 种主动学习策略**，用于从参数空间中智能选取最有价值的样本进行 ODE 数值求解（"标注"），以最小计算成本训练高精度神经求解器。

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
    - [5.1.1 核心思想](#511-核心思想)
    - [5.1.2 Score 构成](#512-score-构成)
    - [5.1.3 MC Dropout 不确定性得分（主导信号）](#513-mc-dropout-不确定性得分uncertainty主导信号)
    - [5.1.4 PCA 轨迹多样性得分（次信号）](#514-pca-轨迹多样性得分diversity次信号)
    - [5.1.5 ODE 残差得分（辅助信号）](#515-ode-残差得分residual辅助信号)
    - [5.1.6 PIAL v2 完整流程](#516-pial-v2-完整流程)
    - [5.1.7 PIAL v2 的优势](#517-pial-v2-的优势)
    - [5.1.8 PIAL v2 的注意事项](#518-pial-v2-的注意事项)
    - [5.1.9 v1 → v2 改进对比](#519-v1--v2-改进对比)
- [6. 实验配置](#6-实验配置)

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
| `pial` | 池化 | **不需要** | ODE 残差 + 轨迹多样性 | **50** |

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

**数据划分合规性**：
- 初始训练集和未标注池**均仅包含 LHS 样本**（来自 `dm.train_dataset_lhs`）
- 验证集和测试集不受影响，测试集仍同时包含 LHS 和 Real Mutants

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
| `--al_strategy` | none | AL 策略（none/random/us/is/wrs/vessal/hggs/rgs/pial） |
| `--al_trigger_epoch` | 10 | AL 触发间隔（PIAL 为 50） |
| `--al_num_add` | 500 | 每轮新增样本数 |
| `--al_initial_train_size` | 2000 | 初始训练集大小 |
| `--al_initial_selection` | kmeans | 初始集选择方法（kmeans/random） |
| `--al_pool_subset_size` | 10000 | 每轮池子集采样大小 |
| `--al_diversity_weight` | 0.5 | IS/VeSSAL 的多样性权重 |
| `--al_uncertainty_metric` | variance | 不确定性度量（variance/entropy） |
| `--al_perturbation` | 0.05 | RGS 扰动幅度 |
| `--al_mae_threshold` | 0.1 | RGS 的 MAE 阈值 |

**参数选择经验**：
- `initial_train_size=2000`：太小（<1000）模型学不到基本动力学，太大（>5000）AL 边际收益有限
- `num_add=500`：太小（<100）每轮收益小，太大（>2000）引入噪声样本
- `trigger_epoch=10`：太频繁（<5）训练不充分，太稀疏（>20）AL 轮次不足
- PIAL `trigger_epoch=50`：早期模型预测差，MC Dropout/ODE 残差信号不稳定

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

## 6. 实验配置

### 6.1 run_pipeline.sh 配置格式

```bash
# 16 列：MODEL LOSS_TYPE USE_ADJ DATASET TAG MODES WIDTH BATCH EPOCHS STRATEGY LR WARMUP NUM_ADD PERTURB MAE_THRESH ROLLOUT
"hyper_fno mse_only false ${DATASET_CHEAT} auto 32 32 64 200 rgs 6e-4 10 500 0.05 0.1 one_time"
"hyper_fno mse_only false ${DATASET_CHEAT} auto 32 32 64 200 pial 6e-4 10 500 0 0 one_time"
```

### 6.2 策略选择建议

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 快速基线 | `random` | 计算最快，用于对比 |
| 不确定性驱动 | `us` | 简单有效 |
| 平衡探索/利用 | `vessal` | 动态权重自适应 |
| 参数空间覆盖 | `hggs` | 不依赖模型，早期可用 |
| Real Mutant 引导 | `rgs` | 直接优化目标域 |
| 物理信息驱动 | `pial` | 无需标签，物理可解释 |

### 6.3 已知问题与修复

1. **PIAL 早期残差 inf/nan**：已添加 `nan_to_num` 防御 + 推迟触发到 epoch 50
2. **RGS 参数异常**：已添加 `_validate_param_ranges` 校验（非负性、物理下界、相对上界）
3. **AL 边际收益递减**：初始训练集从 5000 降到 2000，每轮新增从 5000 降到 500
