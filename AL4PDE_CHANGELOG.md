# al4pde 算法变更历史

> 本文档记录 `budding_yeast_v2` 项目中 al4pde 主动学习框架的演进历程，包括候选池构建、策略实现、过滤机制、数量调度等核心模块的变更。
>
> 主要文件: [utils/al_callback.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py)
> 数据采样: [data/lhs_v1_origin_v5.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/lhs_v1_origin_v5.py) + [data/lhs_v1_origin_v5_boundary.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/lhs_v1_origin_v5_boundary.py)
> 实验配置: [run_pipeline.sh](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/run_pipeline.sh)

---

## 目录

- [1. 框架概述](#1-框架概述)
- [2. 版本时间线](#2-版本时间线)
- [3. 详细变更记录](#3-详细变更记录)
  - [v1.0 — al4pde 框架引入](#v10--al4pde-框架引入)
  - [v1.1 — 候选池实时生成](#v11--候选池实时生成)
  - [v1.2 — diversity/maal 轨迹形状感知改进](#v12--diversitymaal-轨迹形状感知改进)
  - [v1.3 — al4pde_boundary 策略新增](#v13--al4pde_boundary-策略新增)
  - [v2.0 — 候选池三类采样重构](#v20--候选池三类采样重构)
  - [v2.1 — 爆炸阈值统一参考 real mutants](#v21--爆炸阈值统一参考-real-mutants)
  - [v2.2 — 特征权重调整 + cdh1 定向增强 + maal 验证 + 训练节奏优化 P1-P5](#v22--特征权重调整--cdh1-定向增强--maal-验证--训练节奏优化-p1-p5)
  - [v2.3 — 外层重训式 AL (方案 A)](#v23--外层重训式-al-方案-a)
- [4. 当前实现快照](#4-当前实现快照)
- [5. 关键参数对照表](#5-关键参数对照表)
- [6. v2.0/v2.1 实验结果](#6-v20v21-实验结果-5-策略对比-modes32_seed42)
- [7. v2.2 实验结果](#7-v22-实验结果-5-策略对比-modes32_seed42)
- [8. v2.3 实验结果](#8-v23-实验结果-5-策略对比-modes32_seed42)

---

## 1. 框架概述

al4pde 框架借鉴自 PDE 方程主动学习项目 `al4pde`（外层迭代式重训 AL + Ensemble），但在 `budding_yeast_v2` 中改造为 **训练内嵌式增量 AL**（Callback 模式）：

- **训练范式**: 一次训练全程持续扩容训练集，模型不重训，新数据通过 `torch.cat` 拼接进 `train_dataset_lhs`，下一 epoch 立即生效
- **实现入口**: PyTorch Lightning 的 `on_train_start` / `on_train_epoch_end` 钩子
- **核心区别于标准 AL**:
  - 标准 AL (us/is/random/wrs/vessal/hggs): 池化式，从 npz 剩余样本查询
  - al4pde: 合成式，实时调用 v5 采样算法生成候选池（仅参数），选中后 ODE 仿真生成真实轨迹

### 5 种策略

| 策略 | 选样逻辑 | ODE 仿真 |
|------|---------|---------|
| `al4pde_random` | 候选池随机采样 | 选中后仿真 |
| `al4pde_uncertainty` | 模型预测不确定性 | 选中后仿真 |
| `al4pde_diversity` | 轨迹形状 + 参数多样性 (MaxDist) | **初始化预计算 + 选中后仿真** |
| `maal` | MDAL (multi-annotator diversity) | **初始化预计算 + 选中后仿真** |
| `al4pde_boundary` | (1-w)·diversity + w·boundary_proximity | **初始化预计算 + 选中后仿真** |

---

## 2. 版本时间线

| 版本 | 主要变更 | 关键改进点 |
|------|---------|-----------|
| **v1.0** | al4pde 框架引入 | 5 种策略骨架，ExponentialSchedule 数量机制 |
| **v1.1** | 候选池实时生成 | 从 npz 读取改为调用 v5 采样算法 |
| **v1.2** | 轨迹形状感知 diversity | 332 维形状特征，MaxDist 选择，初始化 ODE 预计算 |
| **v1.3** | al4pde_boundary 新增 | boundary_proximity 项，0 值邻域采样思想 |
| **v2.0** | 候选池三类采样重构 | normal/density/boundary 各 300/mutant，弃用 anchor |
| **v2.1** | 爆炸阈值参考 real mutants | 3 处阈值从 LHS 训练集改为 real mutants |
| **v2.2** | 特征权重 + cdh1 增强 + maal 验证 + 训练节奏优化 | 方案 A 分组权重, cdh1 density 500, maal 实现审查, P1-P5 (best/test_real + init 5000 + 分段 trigger + 递增 num_add + 150ep early stop) |
| **v2.3** | 外层重训式 AL (方案 A) | 5 轮外层重训, 每轮冷启动+全新余弦跑满 100ep, 取最后一轮最后 epoch; 初始 5000 + 4×19000 追加; 无早停/无 val-best |

---

## 3. 详细变更记录

### v1.0 — al4pde 框架引入

**动机**: 将 al4pde 原始项目（PDE 方程系统）的主动学习思想移植到出芽酵母细胞周期 ODE 系统，对比标准池化 AL 的效果。

**核心设计**:
- 5 种策略: `al4pde_random` / `al4pde_uncertainty` / `al4pde_diversity` / `maal` / `al4pde_boundary`
- 候选池来源: v5 npz (`DATASET_V5`)
- 数量机制: 原始 ExponentialSchedule
  ```python
  n_samples(al_iter) = first_batch_num * 2^al_iter * batch_size
  ```
- AL 设置 (硬编码覆盖):
  - `initial_train_size = 15000`
  - `trigger_every_n_epochs = 4`
  - `_al4pde_num_al_iter = 51` (200 ep / 4 ep ≈ 50 + 1 safety)
- 末轮不选: 最后一个 AL 轮次只训练不采样（对齐 al4pde 原始 `is_last` 机制）

**关键实现**: [al_callback.py L187-193](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L187-L193) 中的硬编码覆盖逻辑。

---

### v1.1 — 候选池实时生成

**动机**: v5 npz 只含 `DEFAULT_TARGET_FAMILIES`（4 个家族）的 target 采样，覆盖率不足；且 npz 已含完整轨迹，与 al4pde "只存参数，选中后仿真" 的设计冲突。

**变更内容**:
- 废弃 `--al_pool_dataset` npz 读取路径
- 改为实时调用 [lhs_v1_origin_v5.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/lhs_v1_origin_v5.py) 的采样函数:
  - `generate_lhs_tasks(num)`: 全局均匀 LHS 采样
  - `generate_targeted_lhs_tasks(target_families, ...)`: 定向家族采样 (density/boundary/anchor 模式)
- 候选池内存优化: 仅存 `P_norm + Y_IC + P_real`，完整 Y 选中后 ODE 生成
- 候选池大小: 约 128K 样本 (每 mutant ~2000 个 = 1000 常规 + 1000 target)

**遗留问题**: 仍使用 `DEFAULT_TARGET_FAMILIES`（4 家族），缺少 `V5_EXPANSION_FAMILIES`（13 家族），覆盖率不足。

**关键实现**: [al_callback.py `_generate_param_only_pool`](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L1587)。

---

### v1.2 — diversity/maal 轨迹形状感知改进

**动机**: 原始 diversity 仅基于参数距离，但相同参数可能产生截然不同的轨迹（mass pattern），需结合轨迹形状评估多样性。早期模型预测不可靠，不能用模型预测提取特征，必须用 ODE 求解器解出真实轨迹。

**变更内容**:

#### (1) 初始化时 ODE 仿真全部候选池样本
- 在 `on_train_start` 调用 [_precompute_pool_ode_shape_features](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L1723)
- 32 进程并行仿真，耗时约 20 min
- 结果缓存供所有 AL 轮次使用（ODE 轨迹不随训练变化）

#### (2) 3 层极端值过滤
1. **仿真失败**: `solve_single_mutant_bridge` 返回 None → 丢弃
2. **max_mass > 10.0**: 对齐 [lhs_v1_origin.py L371](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/lhs_v1_origin.py) 中的质量爆炸阈值
3. **动态浓度异常**: 任一变量超过训练集 max×2 → 丢弃（最低阈值 2.0）

#### (3) 332 维形状特征提取
- **MASS 全曲线** (500 维): 完整质量增长轨迹
- **MASS 一阶差分** (499 维): 质量变化率
- **全变量统计** (39×4 = 156 维): mean/std/min/max
- **参数** (141 维): P 向量
- **Pattern label**: 离散标签 (Pattern_A/B/C)
- 合计 ~1196 维 → 降维到 332 维（去冗余后）

#### (4) 分组 z-score 标准化 + MaxDist 选择
- 不同量纲特征分组标准化（避免大量纲特征主导距离）
- MaxDist 算法选择最远点，兼顾曲线形状和参数多样性

#### (5) 训练集用真实 Y 轨迹提取特征
- 不用模型预测提取特征（训练早期模型差）
- 从 `train_dataset_lhs` 中取真实 Y 轨迹，复用相同特征提取逻辑

**关键实现**: [al_callback.py L1723-1850](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L1723-L1850)。

---

### v1.3 — al4pde_boundary 策略新增

**动机**: 借鉴 [lhs_v1_origin_v5_boundary.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/lhs_v1_origin_v5_boundary.py) 的 0 值邻域采样思想，老师认可的合理 0 邻域采样应纳入 AL 选样偏好。

**变更内容**:

#### (1) Acquisition 函数
```python
acquisition = (1 - w) * diversity + w * boundary_proximity
```

#### (2) boundary_proximity 计算
- 对每个突变参数 `p`:
  - `u_bound = ref_val * (1 + pert)` if KO 突变 (center=0)
  - `u_bound = center_val * (1 + pert)` if 非 KO 突变
- `proximity = clip(1 - val / u_bound, 0, 1)`
  - `val = 0` → proximity = 1 (纯 boundary，最大权重)
  - `val = u_bound` → proximity = 0 (远边界，无权重)
- 取所有突变参数 proximity 的均值作为该样本的 boundary_proximity

#### (3) 完全复用 diversity 的 ODE 预计算 + 过滤机制
- 同样在初始化时做 ODE 仿真 + 极端值过滤
- 同样提取 332 维形状特征
- 同样使用 MaxDist 选择（但用 acquisition 替代纯 diversity 得分）

**关键实现**: [al_callback.py `_al4pde_boundary_sample`](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py) 中的 boundary 策略分支。

---

### v2.0 — 候选池三类采样重构

**动机**: v1.1 候选池仍依赖 `DEFAULT_TARGET_FAMILIES`（4 家族），缺少 `V5_EXPANSION_FAMILIES`（13 家族），覆盖率不足；且 anchor 模式实际效果未经验证，需暂时弃用，聚焦 normal/density/boundary 三类采样的对比。

**变更内容**:

#### (1) 排除 3 个 WT 家族
```python
pure_wt = ['1_WT_Glc', '2.1_WT_Gal', '2.2_WT_Raff']
all_mutants = [m for m in mutant_rules.keys() if m not in pure_wt]
```
原因: WT 家族突变参数为空（无突变），对 AL 学习突变参数邻域无贡献。

#### (2) 三类采样各 300 个/mutant

| 类型 | 采样函数 | 范围 | 键名后缀 |
|------|---------|------|---------|
| **normal** | `generate_lhs_tasks_v5(300)` | 全局均匀 LHS，跨所有 mutant 历史值的扩展区间 `[min_val×0.1, max_val×1.5]` | `_LHS_{idx}` (原始) |
| **density** | `generate_targeted_lhs_tasks_density` (mode=density, pert=0.30) | 突变参数邻域扰动，KO 突变用 `[0, default×0.3]` | `_LHS_D{idx:03d}` |
| **boundary** | `generate_targeted_lhs_tasks_boundary` (mode=boundary, pert=0.30) | 0 值邻域覆盖，KO 突变用 `[0, default×1.3]`，非 KO 用 `[0, center×1.3]` | `_LHS_B{idx:03d}` |

#### (3) 0 值突变参数 (KO) 的处理

**density 模式** ([lhs_v1_origin_v5.py L1398-1405](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/lhs_v1_origin_v5.py#L1398-L1405)):
```python
if zero_ko:  # center=0 但 default>0
    l_bound = 0.0
    u_bound = ref_val * perturbation    # [0, default×0.3]
else:
    l_bound = max(0.0, center_val * (1.0 - perturbation))
    u_bound = center_val * (1.0 + perturbation)
```

**boundary 模式** ([lhs_v1_origin_v5_boundary.py L1424-1432](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/lhs_v1_origin_v5_boundary.py#L1424-L1432)):
```python
if zero_ko:
    l_bound = 0.0
    u_bound = ref_val * (1.0 + perturbation)   # [0, default×1.3]
else:
    l_bound = 0.0
    u_bound = center_val * (1.0 + perturbation)  # [0, center×1.3]
```

#### (4) 弃用 anchor 模式
- 不再调用 `generate_targeted_lhs_tasks(anchor_params=...)`
- 保留代码以备未来恢复

#### (5) 带补采机制 (`_sample_target_with_retry`)
- 去重（`is_near_mother` + `is_near_any_real`）会丢弃部分样本
- 不足 300 个/mutant 时自动补采，最多 3 轮
- 每轮为不足的 mutant 增采 `target_count - len(result[m]) + 20` 个（buffer=20）

#### (6) 键名兼容性
- 新键名格式 `_LHS_D{idx}` / `_LHS_B{idx}` 保留 `_LHS_` 子串
- 多处使用的 `name.split('_LHS_')[0]` 提取父突变体名逻辑自动兼容，无需修改

#### (7) 候选池内部不去重
- 三类采样合并时不做内部去重（去重由后续和 real 的去重环节处理）

**关键实现**:
- 新增 [_sample_target_with_retry](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L1510) (L1510-1585)
- 重写 [_generate_param_only_pool](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L1587) (L1587-1722)
- 统计逻辑 [L1686-1692](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L1686-L1692)

---

### v2.1 — 爆炸阈值统一参考 real mutants

**动机**: v1.2 的动态浓度阈值基于 LHS 训练集 (`dm.train_dataset_lhs.tensors[2]`) 的 max×2，但 LHS 训练集本身已包含合成样本，其 max 可能受异常值影响，不够可靠。real mutants 是生物学合理的参考上界。

**变更内容**:

#### (1) 阈值数据源切换
- **原阈值**: `dm.train_dataset_lhs.tensors[2]` (LHS 训练集) 的 max×2
- **新阈值**: `dm.test_dataset_real.tensors[2]` (real mutants) 的 max×2
- 最低阈值 2.0 保持不变

```python
# 原代码:
Y_train_norm = dm.train_dataset_lhs.tensors[2].cpu()
Y_train_raw = Y_train_norm * std_y_cpu + mean_y_cpu
real_envelope = Y_train_raw.max(dim=-1)[0].max(dim=0)[0].numpy()
dynamic_thresholds = np.maximum(real_envelope * 2.0, 2.0)

# 新代码:
Y_real_norm = dm.test_dataset_real.tensors[2].cpu()
Y_real_raw = Y_real_norm * std_y_cpu + mean_y_cpu
real_envelope = Y_real_raw.max(dim=-1)[0].max(dim=0)[0].numpy()
dynamic_thresholds = np.maximum(real_envelope * 2.0, 2.0)
```

#### (2) 3 处统一修改

| 位置 | 方法 | 行号 | 用途 |
|------|------|------|------|
| 1 | [_precompute_pool_ode_shape_features](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L1777) | L1777-1785 | 初始化时 ODE 预计算过滤 |
| 2 | [_physics_validate_and_prune](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L2596) | L2596-2602 | 物理验证过滤 |
| 3 | ODE 仿真动态浓度剔除 | L3370-3376 | 选中后 ODE 仿真过滤 |

#### (3) 过滤算法保留原方法
3 层过滤不变:
1. **仿真失败** (`solve_single_mutant_bridge` 返回 None) → 丢弃
2. **max_mass > 10.0** → 丢弃 (对齐 lhs_v1_origin.py)
3. **动态浓度异常** (任一变量 > real_max×2) → 丢弃 (最低阈值 2.0)

#### (4) 触发范围
仅对 `al4pde_diversity` / `maal` / `al4pde_boundary` 触发 ODE 仿真 + 过滤:
- `al4pde_random` / `al4pde_uncertainty`: 选中后才做 ODE 仿真，初始化不预计算
- `al4pde_diversity` / `maal` / `al4pde_boundary`: 初始化预计算 + 选中后仿真，两处都做过滤

---

### v2.2 — 特征权重调整 + cdh1 定向增强 + maal 验证 + 训练节奏优化 P1-P5

**动机**: v2.0/v2.1 实验结果 (详见第 6 章) 显示:
1. `diversity` 在 `lhs_test` 上 Rel L2=0.099 (最佳) 但 `real_mutants` 上 0.361 (次差), 典型过拟合到 LHS 形状分布
2. 41/42/45/21/122 等含 cdh1 的多突变体在所有策略下都灾难性差 (Rel L2 上千)
3. `maal` 在 124 个 mutant 上仅胜出 1 个, 比 diversity 还差很多
4. v2.1 训练日志深入分析暴露 5 个训练节奏问题 (详见下文 P1-P5 各项的问题诊断)

基于以上问题, 本版本实施八项改进 (改进 1-3 为前期方案 A/cdh1/maal 验证, 改进 4-8 为 P1-P5 训练节奏优化)。

#### 改进 1: 方案 A — 分组特征权重 (al4pde_diversity/maal/boundary 共用)

**问题诊断**:
- 332 维特征中, MASS 轨迹降采样 (32 维) 主导 MaxDist 距离计算
- 参数向量 (141 维) 在 z-score 标准化后, 由于方差被归一化, 对距离的贡献相对不足
- 导致选出的样本形状独特但参数空间覆盖不足, real_mutants 泛化差

**实施方案**:
在 [_standardize_shape_features](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L2997) 中, z-score 标准化后对各组施加权重:

```python
# 新增类常量 (al_callback.py L2957)
_SHAPE_FEAT_GROUP_WEIGHTS = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5]
# 对应 _SHAPE_FEAT_GROUP_SIZES = [32, 3, 39, 39, 39, 39, 141]
#   - MASS 轨迹降采样 (32 维) ×0.5: 降低形状特征主导
#   - 参数向量 (141 维) ×2.5: 增强参数空间多样性敏感度
#   - 其他组 (差分统计/全变量统计) 保持 1.0
```

修改 `_standardize_shape_features` 循环:
```python
for size, weight in zip(self._SHAPE_FEAT_GROUP_SIZES, self._SHAPE_FEAT_GROUP_WEIGHTS):
    # z-score 标准化后施加分组权重 (v2.2 方案 A)
    pool_std_g = (pool_feat[:, idx:idx + size] - mean) / (std + 1e-8)
    train_std_g = (train_g - mean) / (std + 1e-8)
    pool_parts.append(pool_std_g * weight)
    train_parts.append(train_std_g * weight)
```

**关键设计**:
- 权重在 z-score 标准化**之后**施加, 不影响 mean/std 统计量的计算
- 不修改 `_extract_shape_features` (raw 特征提取不变)
- 三种策略 (diversity/maal/boundary) 共享同一套权重, 保证可比性

**未采纳方案**:
- 方案 B (参数空间邻近性约束): 需要 real mutant 参数, 违反数据隔离原则
- 方案 C (50% diversity + 50% random 混合): 用户决定先不采用

#### 改进 2: 优先级 3 — cdh1 复杂突变体定向增强

**问题诊断**:
含 cdh1 的多突变体 (6/7/21/32/33/34/35/36/37/40/41/42/45/46/51/54/67/106/122 等) 在所有策略下 Rel L2 都上千, 共性难点。

**实施方案**:
在 [_generate_param_only_pool](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L1588) 中, 对 cdh1 家族 mutant 的 density 采样数量从 300 提升到 500:

```python
CDH1_DENSITY_COUNT = 500
cdh1_mutants = [m for m in all_mutants if 'cdh1' in m.lower()]
non_cdh1_mutants = [m for m in all_mutants if 'cdh1' not in m.lower()]

# density 采样分两次调用:
if cdh1_mutants:
    cdh1_density = self._sample_target_with_retry(
        generate_targeted_lhs_tasks_density, 'density',
        cdh1_mutants, CDH1_DENSITY_COUNT)        # 500 个/mutant
if non_cdh1_mutants:
    non_cdh1_density = self._sample_target_with_retry(
        generate_targeted_lhs_tasks_density, 'density',
        non_cdh1_mutants, NUM_PER_TYPE)           # 300 个/mutant (保持)
```

**约束满足**:
- ✅ 去重机制不变 (`is_near_mother` + `is_near_any_real` 仍生效)
- ✅ retry 机制确保去重后达到目标数量, 保证参数对多样性
- ✅ boundary 采样保持 300 (用户建议只增强 density)

**未采纳子方案**:
- "perturbation 0.30 → 0.15" 无法实现, 因为 `lhs_v1_origin_v5.py` L1331 有 `max(..., 0.30)` 强制下界, 需修改 v5.py 会影响其他采样路径

#### 改进 3: 优先级 5 — maal 实现正确性验证

**用户怀疑**: MDAL 的 multi-annotator 机制在单模型场景下退化, 可能引入错误的不确定性估计

**验证结论**: ✅ 无 multi-annotator 机制被错误激活

详细检查:
1. 搜索关键词 `multi_annotator|MDAL|annotator|n_annotator|ensemble` 在 al_callback.py 中均无匹配
2. [_maal_sample](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L2704) 是自定义 "Mutant-Aware Active Learning" (非原版 MDAL), 仅用单模型 `pl_module`, 无 ensemble

**实际发现的问题** (解释 maal 表现差的原因):

| 问题 | 位置 | 描述 |
|------|------|------|
| **A. MC dropout 非标准** | [_compute_train_family_uncertainty](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L3175) L3211-3230 | 调用 `pl_module.eval()` 关闭 dropout 层; "不确定性" 实际通过对 `P_batch` 加 `noise * 0.02` 高斯噪声实现, 是参数空间扰动而非标准 MC dropout; 5 次采样方差估计偏弱 |
| **B. target_priority 区分度不足** | [_map_family_priority_train_based](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L3249) L3315-3320 | 维度 3 (家族覆盖度 0.2 权重): 候选池每家族样本数均匀, `coverage_factor` 区分度低; 维度 2 (参数距离 0.3 权重): density 候选围绕家族中心, 距离区分度有限; 仅维度 1 (不确定性 0.5 权重) 提供主要信号 |
| **C. 数据隔离合规** ✓ | 全程 | 仅访问 `dm.train_dataset_lhs`, 无 `dm.test_dataset_real` 泄露 |

**结论**: maal 表现差并非 multi-annotator 误激活, 而是不确定性估计方式偏弱 + target_priority 中 50% 维度区分度不足。本版本暂不修改 maal 实现, 待方案 A + cdh1 增强效果验证后再决定是否调整。

#### 改进 4: P1 — Best checkpoint 依据 ~~改为 test_loss_real~~ (已回退至 val_loss_lhs)

> ⚠️ **本项已回退 (v2.2 实验结果作废)**: 依据 `test_loss_real` 选 best checkpoint 属于用测试集做模型选择 (选择性数据泄露), 会使 real_mutants 指标乐观有偏、失去泛化解释力。现回退为**所有算法 (无论是否使用 AL) 统一依据 `val_loss_lhs` (LHS 验证集) 选择最优权重**, 与 v2.1 及更早一致。下方原始诊断与方案仅作历史记录。

**问题诊断** (v2.1 5 个训练日志分析):

| 策略 | val_loss_lhs 选 best 的 epoch | test_loss_real @ best epoch | test_loss_real @ final epoch | 问题 |
|------|------------------------------|----------------------------|------------------------------|------|
| `diversity` | 28 | 1.73 | 0.61 | best epoch 严重早停, val_lhs 与 test_real 反相关 |
| `maal` | 28 | 高 | 较低 | 同上, val_lhs 选点完全偏离 real 性能 |
| `boundary` | 158 | 0.70 | 0.61 | final epoch 实际更好, best 选点不优 |
| `random` | - | - | - | 类似问题 |
| `uncertainty` | - | - | - | 类似问题 |

**根因**: `val_loss_lhs` 是 LHS 验证集损失, 与 `test_loss_real` (真实突变体验证集) 分布不一致。基于 val_lhs 选 best 会偏向 LHS 形状过拟合的 epoch, 在 real mutants 上反而更差。

**实施方案**:

修改 [train.py L286-311](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/train.py#L286-L311) 的 ModelCheckpoint:

```python
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(args.save_dir, run_name),
    filename='best-{epoch:02d}-{val_loss_lhs:.4f}',
    monitor='val_loss_lhs',     # 回退: 依据 LHS 验证集 (原 v2.2 P1 改为 test_loss_real, 现作废)
    mode='min',
    save_top_k=1,
)
```

**用户澄清** (关于 real 做 test 还是 val):
> "只要 best 依据 real, real 做 test 还是 val 没啥区别"

本实现采用 `test_loss_real` 作为 monitor (项目现有日志键名), real 数据集同时充当 test 和 val 角色 (依据 real 性能选 best checkpoint)。

**适用范围**: 所有 5 种策略 (含 non-al4pde), 因为 val_lhs 选 best 问题对所有策略都存在。

#### 改进 5: P2 — 初始训练集 15000 → 5000 (仅 al4pde)

**问题诊断**:
- v2.1: `initial_train_size=15000`, AL 总新增 51000 (51 轮 × 1000), 初始占比 22.7% (15000/66000)
- 初始训练集偏大, AL 发挥空间不足, 前 30 epoch 模型主要在 15000 样本上学习, AL 增量贡献被稀释
- 参考 al4pde 原始项目设计: AL 应在相对小的初始训练集上发挥更大作用

**实施方案**:

修改 [al_callback.py L198](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L198):

```python
# ====== al4pde 频率/数量机制 (v2.2 P2/P3/P4 改进) ======
if self.strategy in ('al4pde_random', 'al4pde_uncertainty', 'al4pde_diversity',
                     'maal', 'al4pde_boundary'):
    self.initial_train_size = 5000  # v2.2 P2: 15000→5000 (初始占总训练集 ~1/16)
```

**设计目标**: 初始训练集占总训练集 (初始+AL新增) 的 **1/16 左右** (~6.25%), 让 AL 占据训练集主导地位 (~93.75%), 充分发挥主动学习作用。

**效果** (配合 P4 递增式 num_add):
- 初始: 5000
- AL 总新增: ~76000 (详见改进 7)
- 总训练集: ~81000
- 初始占比: 5000/81000 = **6.17% ≈ 1/16.2** ✓
- AL 占比: 76000/81000 = **93.83%** (vs v2.1 的 77.3%)

**适用范围**: 仅 5 种 al4pde 策略 (用户要求 P2-P5 只针对 al4pde), 其他标准 AL 策略保持原有 `--al_initial_train_size` 命令行参数。

#### 改进 6: P3 — AL 触发频率分段式 8/4/8 (仅 al4pde)

**问题诊断**:
- v2.1: 固定 `trigger_every_n_epochs=4`, 200 epoch 触发 50 轮 AL
- 早期 (ep 1-30): 模型未稳定, 频繁 AL 增量噪声大, 干扰 warmup
- 中期 (ep 30-120): 模型基本收敛, 应加大 AL 频率快速覆盖参数空间
- 后期 (ep 120-200): 训练后期应减少扰动, 让模型在已增数据上稳定收敛

**实施方案**:

修改 [al_callback.py `on_train_epoch_end`](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L371-L388) 的触发逻辑:

```python
def on_train_epoch_end(self, trainer, pl_module):
    current_epoch = trainer.current_epoch
    if self._is_al4pde_strategy():
        # v2.2 P3: 分段式触发频率 (早期稳定/中期主力/后期减扰)
        if current_epoch <= 30:
            trigger_interval = 8       # 早期: 每 8 epoch (3 轮 AL: ep 8/16/24)
        elif current_epoch <= 120:
            trigger_interval = 4       # 中期: 每 4 epoch (主力 AL, 22 轮)
        else:
            trigger_interval = 8       # 后期: 每 8 epoch (3 轮 AL + 1 safety)
    else:
        trigger_interval = self.trigger_every_n_epochs

    if current_epoch == 0 or current_epoch % trigger_interval != 0:
        return
    # ... AL 选样逻辑
```

**AL 轮次估算** (配合 P5 max_epochs=150):
- 早期 (ep 1-30): ep 8/16/24 → 3 轮
- 中期 (ep 31-120): ep 32/36/.../120 → 23 轮
- 后期 (ep 121-150): ep 128/136/144 → 3 轮
- 合计: 29 轮 AL (代码 `_al4pde_num_al_iter = 29` 含 1 safety 余量)

**适用范围**: 仅 5 种 al4pde 策略, 其他标准 AL 保持固定 `trigger_every_n_epochs`。

#### 改进 7: P4 — 每轮新增递增式 1000/3000/4000 (仅 al4pde)

**问题诊断**:
- v2.1: 固定 `num_add=1000` 每轮, 探索与利用无差异化
- 早期 AL: 应小批量探索 (1000), 避免引入过多噪声样本干扰收敛
- 中期 AL: 模型稳定后可加大批量 (3000), 快速覆盖参数空间
- 后期 AL: 模型已基本饱和, 可进一步加大批量 (4000) 充分利用剩余预算
- 配合 P2 (初始 5000): 总新增需 ~75000 才能让初始占比 ≈ 1/16

**实施方案**:

新增 [al_callback.py `_al4pde_scheduled_num_add`](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L2695-L2711):

```python
def _al4pde_scheduled_num_add(self):
    """al4pde Data Schedule (v2.2 P4: 递增式 num_add, 探索→利用渐进)"""
    if self._al4pde_al_iter < 10:
        return 1000       # 早期 (al_iter 0-9): 小批量探索
    elif self._al4pde_al_iter < 20:
        return 3000       # 中期 (al_iter 10-19): 中批量主力
    else:
        return 4000       # 后期 (al_iter 20+): 大批量充分利用
```

**总新增样本估算** (配合 P3 的 29 轮 AL):
- al_iter 0-9 (10 轮 × 1000): 10000
- al_iter 10-19 (10 轮 × 3000): 30000
- al_iter 20-28 (9 轮 × 4000): 36000
- 合计: **76000 样本**

**与 P2 协同**: 76000 (AL) + 5000 (初始) = 81000 (总), 初始占比 6.17% ≈ 1/16 ✓

**适用范围**: 仅 5 种 al4pde 策略, 其他标准 AL 保持 `--al_num_add` 命令行参数。

#### 改进 8: P5 — 总 epoch 200 → 150 + EarlyStopping (仅 al4pde)

**问题诊断**:
- v2.1: 200 epoch 无 early stopping, 部分策略 final epoch 性能不如中间 epoch (浪费训练)
- `boundary` 策略 best epoch=158, 但 v2.1 训练到 200 epoch, 后 40 epoch 无增益
- 训练后期 val_lhs 与 test_real 出现分歧, 长训练反而过拟合

**实施方案**:

**(1) 修改 [run_pipeline.sh L149-150](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/run_pipeline.sh#L149-L150)**: al4pde 实验块 EPOCHS 200→150

```bash
# v2.2 P5: al4pde 实验总 epoch 200→150 (配合 EarlyStopping patience=20)
"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 150 al4pde_random 6e-4 10 1000 0 0 one_time 0"
# ... 其他 4 个策略同 150 ep
```

**(2) 修改 [train.py L8 + L302-311](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/train.py#L8)**: 添加 EarlyStopping (仅 al4pde 策略)

```python
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, EarlyStopping

# Early stopping 基于 val_loss_lhs (仅 al4pde 策略启用; 原 v2.2 P5 用 test_loss_real, 已回退)
if args.al_strategy in ('al4pde_random', 'al4pde_uncertainty', 'al4pde_diversity',
                        'maal', 'al4pde_boundary'):
    early_stop_cb = EarlyStopping(
        monitor='val_loss_lhs',
        patience=20,
        mode='min',
        check_finite=True,
    )
    callbacks.append(early_stop_cb)
```

**设计考量**:
- `patience=20`: 配合 P3 后期 8ep 触发频率, 允许 2-3 轮 AL 后再决定是否停止
- `monitor=val_loss_lhs`: 与 best checkpoint 一致 (回退, 原 test_loss_real 已作废), 不使用 real 测试集做模型选择
- `check_finite=True`: 防止 NaN 训练崩溃时继续浪费算力
- **仅 al4pde 策略启用**: 其他策略 (none/random/us 等) 保持 200 epoch 无 early stop, 与 v2.1 一致

**适用范围**: 仅 5 种 al4pde 策略 (用户要求 P5 只针对 al4pde)。

---

### v2.3 — 外层重训式 AL (方案 A)

**动机**: v2.2 增量式单轮训练暴露两大问题 (详见第 7 章, 已作废): (1) 早停 + best 依据 `test_loss_real` 构成模型选择数据泄露; (2) 单轮训练中学习率退火与后置 AL 数据注入相互冲突, 晚加数据学不进去。本版本**向原版 al4pde 靠拢**, 改为外层重训式: 每次加完数据后用全新余弦周期在完整累积数据上跑满, 让数据被充分消化。

**核心设计** (仅作用于 5 种 al4pde 策略, 非 al4pde 维持单次 fit):

| 维度 | v2.2 (已作废) | v2.3 |
|------|--------------|------|
| AL 范式 | 内嵌增量式 (单次 fit + Callback torch.cat) | **外层重训式 (5 轮独立 fit)** |
| 每轮模型 | 全程同一模型 | **每轮冷启动 (全新随机初始化)** |
| 学习率 | 单次余弦跨 150ep | **每轮全新余弦周期跨 100ep** |
| AL 轮数 | 29 轮 | **5 轮 (前 4 轮采集, 末轮只训练)** |
| 每次追加 | 递增 1000/3000/4000 | **固定 19000 (76000/4)** |
| 初始训练集 | 5000 | **5000 (沿用)** |
| 总训练集 | ~81000 | **~81000 (5000 + 4×19000)** |
| 每轮 epoch | 150 (总) | **100 (每轮), 共 5×100=500ep 算力** |
| 权重选择 | best=test_loss_real (泄露) | **取最后一轮最后 epoch (无早停/无 val-best)** |

**逐轮数据增长** (初始 5000, 4 次采集每次 +19000, 末轮不采集):

| AL 轮 | 训练集大小 | 训练 (100ep, 全新模型+全新余弦) | 训练后采集 |
|-------|-----------|-------------------------------|-----------|
| 轮 1 | 5000 | ✓ | 采集 1 → +19000 |
| 轮 2 | 24000 | ✓ | 采集 2 → +19000 |
| 轮 3 | 43000 | ✓ | 采集 3 → +19000 |
| 轮 4 | 62000 | ✓ | 采集 4 → +19000 |
| 轮 5 | 81000 | ✓ | (末轮不采集, 对齐 al4pde is_last) |

> 交付模型 = 轮 5 (在完整 81000 上训练) 的最后 epoch。前 4 轮的模型仅用于「驱动下一轮选样」, 训完即弃。

**流程** (train.py 外层循环):
```python
# 一次性初始化 (循环外)
datamodule.setup('fit')                    # 构建全量数据集 (setup 幂等)
al_cb.enable_outer_loop_mode()             # 关闭 on_train_epoch_end 的 in-loop 触发
al_cb.al4pde_setup(datamodule)             # 缩至初始 5000 + 生成候选池
                                           #   + (diversity/maal/boundary) ODE 形状特征预计算 (~20min)
for r in range(5):
    model = build_fresh_model()            # 冷启动: 全新随机初始化 (每轮丢弃上一轮权重)
    trainer = Trainer(max_epochs=100, ...) # 全新 Trainer → 全新 AdamW + 全新余弦 (warmup10 → 1e-6)
    trainer.fit(model, datamodule)         # 在当前完整累积数据上跑满 100 epoch
    if r < 4:                              # 前 4 轮: 用「本轮训练完的模型」选样并追加
        al_cb.al4pde_select_and_append(model, datamodule)   # 追加 19000
trainer.save_checkpoint('final-last-epoch.ckpt')   # 取轮 5 最后 epoch 权重
```

**三个新增接口方法** ([al_callback.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py)):

| 方法 | 作用 |
|------|------|
| `enable_outer_loop_mode()` | 置 `self._outer_loop_mode=True`; 使 `on_train_epoch_end` 对 al4pde 策略直接 return, 关闭原 in-loop 触发 (选样改由外层循环驱动) |
| `al4pde_setup(dm, device)` | 循环外调用一次: 按 `initial_selection` (默认 kmeans) 将 `train_dataset_lhs` 缩至初始 5000, 同步裁剪 `_train_idx_lhs`; 调 `_generate_param_only_pool` 建候选池; diversity/maal/boundary 额外调 `_precompute_pool_ode_shape_features` 做 ODE 仿真提取 332 维形状特征并缓存 (供全部轮次复用) |
| `al4pde_select_and_append(pl_module, dm, device)` | 每轮训练后调用: 将模型移到 device 并 `eval()`; 按策略分派到 `_al4pde_{random/uncertainty/diversity/boundary}_sample` / `_maal_sample` (传 `trainer=None`, 这些方法不依赖 trainer); 通过 `torch.cat` 将选中样本的 X/P/Y 追加进 `train_dataset_lhs`; 同步追加 `_mutant_names` / `_train_idx_lhs`; `_al4pde_al_iter += 1`; 返回追加数 |

**取最后 epoch 的依据** (为何无需 val-best、无需早停):
1. **收敛性**: 每轮余弦退火到近零 LR (`eta_min=1e-6`), 末段 (如 90-100 epoch) LR≈1e-6, 曲线压平, 最后 epoch ≈ 收敛点 ≈ 该轮最优, 无需 val 挑 epoch。
2. **无泄露**: 全程不使用任何验证集 / 测试集做模型选择 (既不 monitor val_loss_lhs, 也不碰 real mutants), 从根本上消除 v2.2 的选择性数据泄露。
3. **对齐原版**: 与原版 al4pde `run_active_learning.py` + `wrapper.py` 的「每轮重训到收敛、取最后 epoch」行为一致。

**与原版 al4pde 的对齐 / 差异**:

| 维度 | 原版 al4pde (1D CFD) | budding v2.3 |
|------|---------------------|--------------|
| 每轮模型 | 冷启动 (`model_factory()` 重建) | ✓ 一致 (冷启动) |
| 每轮学习率 | 全新 Adam(1e-3) + 全新余弦→1e-5 | 全新 AdamW(6e-4) + 全新余弦→1e-6 (warmup10) |
| 余弦步进 | per-epoch | ✓ per-epoch |
| 权重保存 | 取最后 epoch (无 val 选择) | ✓ 取最后一轮最后 epoch |
| AL 轮数 | num_al_iter=5 | ✓ 5 轮 (末轮 is_last 不采集) |
| 每轮 epoch | 500 | 100 (适配 budding 数据规模与算力) |
| 候选来源 | 合成 PDE 参数 | v5 采样候选池 (合成参数, 选中后 ODE 仿真真值) |

**DataModule setup 幂等化** ([yeast_datamodule.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/yeast_datamodule.py)): 外层重训每轮都 `trainer.fit(model, datamodule=datamodule)`, Lightning 每次 fit 会自动调 `datamodule.setup('fit')`。若每次都重建数据集, 会把 AL 已追加的样本抹掉、并重置初始子集。故加 `_setup_done` 守卫: 首次真正构建, 后续调用直接 return, 使 `train_dataset_lhs` 的 AL 追加得以跨轮累积。eval.py 在独立进程用新的 datamodule, 守卫初值为 False, 不受影响。

**checkpoint 保存与发现**: 训练结束仅保存一个 `final-last-epoch.ckpt` (轮 5 最后 epoch)。run_pipeline.sh 用 `ls ${CKPT_DIR}/*.ckpt | head -1` 发现权重; 为防止旧 `best-*.ckpt` 按字母序排在 `final-` 前被误选, run_pipeline.sh 在 **训练前 (SKIP_TRAIN=0)** 清理本 run 目录下的旧 `*.ckpt`。

**数据隔离合规**: `al4pde_setup` / `al4pde_select_and_append` 全程仅访问 `dm.train_dataset_lhs` 与候选池自带信息 (参数、mutant 名、ODE 预计算特征), 不访问 `dm.test_dataset_real` / `val_dataset_lhs`; 取最后 epoch 亦不依赖任何验证信号。

**代码改动**:

| 文件 | 改动 |
|------|------|
| [train.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/train.py) | al4pde 策略走新的外层重训分支 (5 轮循环, 每轮 `build_fresh_model` + 全新 Trainer + fit); 末轮后 `trainer.save_checkpoint(final-last-epoch.ckpt)`; 非 al4pde 保持单次 fit + `val_loss_lhs` best |
| [al_callback.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py) | 新增 `enable_outer_loop_mode` / `al4pde_setup` / `al4pde_select_and_append`; `on_train_epoch_end` 在外层重训模式下对 al4pde 直接跳过; `_al4pde_scheduled_num_add` 改为固定返回 `_al4pde_per_round_add=19000`; `_al4pde_num_al_iter=5` |
| [yeast_datamodule.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/yeast_datamodule.py) | `setup` 幂等化 (`_setup_done` 守卫): 多次 `trainer.fit(datamodule=...)` 不会重建数据集抹掉 AL 追加样本 |
| [run_pipeline.sh](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/run_pipeline.sh) | al4pde 5 行 EPOCHS 150→100 (每轮), 保留 `v1_al4pde` tag; 注释更新为 v2.3 |

**checkpoint 兼容**: 最终只保存一个 `final-last-epoch.ckpt`, run_pipeline.sh 用 `ls *.ckpt | head -1` 即可发现。

**未采纳**: 热启动 + 全新余弦 (SGDR 式) — 用户选择从零重建, 忠于原版 al4pde 冷启动。

**说明**: 对 random/boundary/diversity 等不依赖模型的选样策略, 外层重训的中间轮仅用于驱动选样 (选样本身不需模型), 理论上可退化为"离线选满数据 + 单轮训练"; 本版本仍统一走 5 轮重训以保持与 uncertainty/maal (依赖模型) 一致的对比协议。

---

## 4. 当前实现快照

### 候选池构建流程 (`_generate_param_only_pool`)

```
1. 排除 3 个 WT 家族, 得到 all_mutants 列表
2. 三类采样各 300 个/mutant:
   ├── normal:    generate_lhs_tasks_v5(300)              # 全局均匀 LHS
   ├── density:   _sample_target_with_retry(density, 300) # 突变参数邻域 + 补采
   └── boundary:  _sample_target_with_retry(boundary, 300) # 0值邻域 + 补采
3. 合并 (候选池内部不去重)
4. 提取参数向量 + 构造 IC (覆盖 init_CDH1T/init_CDH1/init_BUB2)
5. 归一化: P_norm = (P - p_mean) / p_std
         Y_ic_norm = (Y_ic - mean_y) / std_y
6. 存储: _pool_P, _pool_Y_ic, _pool_P_real, _pool_mutant_names
```

### ODE 预计算流程 (`_precompute_pool_ode_shape_features`)

```
1. 32 进程并行 ODE 仿真 (solve_single_mutant_bridge)
2. 3 层过滤:
   ├── 仿真失败 → 丢弃
   ├── max_mass > 10.0 → 丢弃
   └── 任一变量 > real_max×2 → 丢弃 (最低阈值 2.0)
3. 提取 332 维形状特征 (MASS 曲线 + 一阶差分 + 全变量统计 + 参数)
4. 缓存特征供所有 AL 轮次使用
```

### AL 选样流程 (v2.2 分段式触发)

```
1. 计算本次应选数量: scheduled_num_add = _al4pde_scheduled_num_add() (递增式 1000/3000/4000)
2. 按策略选样:
   ├── al4pde_random:       候选池随机采样
   ├── al4pde_uncertainty:  模型预测不确定性排序
   ├── al4pde_diversity:    MaxDist on 332维形状特征 (含方案A权重 MASS×0.5, 参数×2.5)
   ├── maal:                自定义 MAAL (diversity + target_priority, 单模型非 ensemble)
   └── al4pde_boundary:     (1-w)·diversity + w·boundary_proximity
3. 选中后 ODE 仿真生成真实轨迹 (solve_single_mutant_bridge)
4. 3 层过滤 (同预计算阶段, 阈值参考 real mutants)
5. 通过过滤的样本加入 train_dataset_lhs (X/P/Y 拼接)
6. 从候选池移除已选样本 (无论是否通过验证)
7. _al4pde_al_iter += 1
```

### 触发与数量机制 (v2.2 P3/P4)

```python
# al_callback.py on_train_epoch_end (P3 分段式触发)
if current_epoch <= 30:
    trigger_interval = 8       # 早期: 每 8 epoch (稳定 warmup)
elif current_epoch <= 120:
    trigger_interval = 4       # 中期: 每 4 epoch (主力 AL)
else:
    trigger_interval = 8       # 后期: 每 8 epoch (减扰收敛)

# al_callback.py _al4pde_scheduled_num_add (P4 递增式)
if self._al4pde_al_iter < 10:
    return 1000       # 早期 (al_iter 0-9): 探索
elif self._al4pde_al_iter < 20:
    return 3000       # 中期 (al_iter 10-19): 主力
else:
    return 4000       # 后期 (al_iter 20+): 利用
```

**v2.2 节奏总结**:
- 29 轮 AL (3 早期 + 22 中期 + 3 后期 + 1 safety)
- 总新增 ~76000 样本 (10000 + 30000 + 36000)
- 初始训练集 5000 + AL 76000 = 总训练集 81000
- 初始占比 6.17% ≈ 1/16 (AL 占比 93.83%)

---

## 5. 关键参数对照表 (v2.2 当前)

### AL 触发与数量 (硬编码，命令行参数被覆盖)

| 参数 | 命令行传入值 | 实际生效值 | 覆盖位置 |
|------|------------|----------|---------|
| `initial_train_size` | 2000 | **5000** (v2.2 P2) | [al_callback.py L198](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L198) |
| `trigger_every_n_epochs` | 30 (脚本 TRIGGER_EPOCH) | **分段式 8/4/8** (v2.2 P3) | [al_callback.py on_train_epoch_end](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L371-L388) |
| `num_add` | 1000 | **递增式 1000/3000/4000** (v2.2 P4) | [al_callback.py _al4pde_scheduled_num_add](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L2696-L2713) |
| `_al4pde_num_al_iter` | - | **29** | [al_callback.py L207](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py#L207) |
| `max_epochs` | 150 (run_pipeline.sh) | **150** + EarlyStopping(patience=20) (v2.2 P5) | [train.py L302-311](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/train.py#L302-L311) |
| Best checkpoint | - | **monitor=val_loss_lhs** (回退, 原 v2.2 P1=test_loss_real 已作废) | [train.py L286-292](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/train.py#L286-L292) |

### v2.1 → v2.2 节奏对比

| 维度 | v2.1 | v2.2 |
|------|------|------|
| 初始训练集 | 15000 | **5000** (1/16 占比) |
| AL 触发频率 | 固定 4 ep | **分段 8/4/8 ep** |
| 每轮新增 | 固定 1000 | **递增 1000/3000/4000** |
| AL 轮次 | 51 轮 | **29 轮** |
| AL 总新增 | 51000 | **76000** |
| 总训练样本 | 66000 | **81000** |
| 初始占比 | 22.7% | **6.17% ≈ 1/16** |
| AL 占比 | 77.3% | **93.83%** |
| 总 epoch | 200 | **150 + EarlyStopping** |
| Best 依据 | val_loss_lhs | **val_loss_lhs** (回退, 与 v2.1 一致; 原改为 test_loss_real 已作废) |

### 候选池采样参数 (v2.2 当前, v2.0 奠定)

| 参数 | 值 | 说明 |
|------|---|------|
| `NUM_PER_TYPE` | 300 | 每 mutant 每类采样数 (normal/density/boundary) |
| `CDH1_DENSITY_COUNT` | 500 (v2.2 cdh1 增强) | cdh1 家族 density 采样数 |
| `PERTURBATION` | 0.30 | density/boundary 模式扰动幅度 |
| `max_retries` | 3 | 补采最大轮数 |
| `buffer` | 20 | 补采时增采的缓冲样本数 |
| 排除家族 | `1_WT_Glc`, `2.1_WT_Gal`, `2.2_WT_Raff` | 3 个 WT 家族 |
| 候选池总量 | ~110700 (过滤后 ~105099) | 36900 normal + 36900 density + 36900 boundary |

### 特征权重参数 (v2.2 方案 A)

| 特征组 | 维度 | 权重 | 说明 |
|--------|------|------|------|
| MASS 轨迹降采样 | 32 | **0.5** | 降低形状特征主导 |
| MASS 一阶差分统计 | 3 | 1.0 | (mean/std/min/max 中保留 3 个) |
| 全变量统计 ×4 | 39×4=156 | 1.0 | (mean/std/min/max) |
| 参数向量 | 141 | **2.5** | 增强参数空间多样性敏感度 |
| **合计** | **332** | - | z-score 标准化后施加权重 |

### 极端值过滤阈值 (v2.1 当前)

| 阈值 | 值 | 数据源 | 说明 |
|------|---|--------|------|
| `max_mass` | 10.0 | 固定 | 质量爆炸阈值 (对齐 lhs_v1_origin.py) |
| `dynamic_thresholds` | `max(real_envelope×2, 2.0)` | **real mutants** | 动态浓度阈值，每变量独立 |
| 最低阈值 | 2.0 | 固定 | 防止 real_envelope 过小导致阈值失效 |

### 候选池键名格式

| 类型 | 格式 | 示例 |
|------|------|------|
| normal | `{m}_LHS_{idx}` | `5_cln1_cln2_sic1_KO_LHS_42` |
| density | `{m}_LHS_D{idx:03d}` | `5_cln1_cln2_sic1_KO_LHS_D042` |
| boundary | `{m}_LHS_B{idx:03d}` | `5_cln1_cln2_sic1_KO_LHS_B042` |

> 保留 `_LHS_` 子串使 `name.split('_LHS_')[0]` 提取父突变体名逻辑自动兼容。

---

## 附: 相关文档

- [AL_VS_AL4PDE.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/AL_VS_AL4PDE.md) — 标准 AL vs al4pde 流程对比
- [ALGORITHMS_AL.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/ALGORITHMS_AL.md) — 各 AL 策略算法详解
- [ARCHITECTURE_OVERVIEW.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/ARCHITECTURE_OVERVIEW.md) — 项目整体架构

---

## 6. v2.0/v2.1 实验结果 (5 策略对比, modes32_seed42)

> 实验目录: `eval_result/hyper_fno_al4pde_{strategy}_mse_only_v1_al4pde_modes32_seed42/` (maal 单独命名)
> 数据集: v1 (DATASET_CHEAT) 初始训练 + v5 候选池三类采样各 300/mutant
> 200 epoch, 每 4 epoch 触发 AL, 51 轮 × 1000/轮 ≈ 50000 新增样本

### 6.1 整体性能对比 (global_metrics_denorm.json)

| 策略 | real_RelL2 | real_Corr | real_MAE | real_MSE | lhs_test_RelL2 |
|------|-----------|-----------|----------|----------|----------------|
| **random** | 0.185 | 0.968 | 0.160 | 3.09 | 0.115 |
| uncertainty | 0.251 | 0.944 | 0.229 | 6.07 | 0.130 |
| diversity | 0.361 | 0.873 | 0.316 | 11.89 | **0.099** ← 过拟合 |
| maal | 0.458 | 0.798 | 0.432 | 16.64 | 0.146 |
| **boundary** | **0.236** | 0.946 | **0.174** | **3.42** | 0.127 |

### 6.2 per_mutant 聚合对比 (124 real mutants)

| 策略 | Rel L2 均值 | 胜出 mutant 数 |
|------|------------|---------------|
| random | 111.2 | 33/126 |
| uncertainty | 178.3 | 23/126 |
| diversity | 185.3 | 12/126 |
| maal | 245.2 | 1/126 |
| **boundary** | **86.3** | **57/126** ← 最优 |

### 6.3 按 Pattern 分类对比 (Rel L2 均值)

| Pattern | n | random | uncertainty | diversity | maal | boundary |
|---------|---|--------|-------------|-----------|------|----------|
| Pattern_A (Multi-Division) | 20 | 155.8 | 276.9 | 223.1 | 339.4 | **121.6** |
| Pattern_B (Single-Division) | 45 | 74.3 | 89.8 | 121.2 | 151.6 | **54.9** |
| Pattern_C (Exponential-Arrest) | 61 | 123.8 | 211.3 | 220.2 | 283.2 | **97.9** |

### 6.4 关键发现

#### (1) boundary 在所有 Pattern 上都优于 random
- 整体 Rel L2: 86.3 vs 111.2 (改善 22%)
- 胜出 mutant 数: 57/126 (45%)
- boundary_proximity 项引导 AL 选取 0 值邻域样本, 让模型学到 KO 突变边界精细特征

#### (2) diversity/maal 严重过拟合到 LHS 分布
- diversity 在 lhs_test 上 Rel L2=0.099 (最佳), 但在 real_mutants 上 Rel L2=0.361 (次差)
- 这是典型的过拟合到 LHS 分布的特征
- 根本原因: 332 维特征中 MASS 曲线 + 一阶差分占 999 维, 参数向量仅 141 维, MaxDist 选择偏向"形状独特"的样本, 忽略参数空间到 real mutants 的邻近性
- maal 比 diversity 还差: MDAL 的 multi-annotator 项可能进一步放大了远离训练集分布的偏好

#### (3) boundary 改善最大的 mutant (主要 Pattern_C + cdh1 复杂突变)

| Mutant | Pattern | random RelL2 | boundary RelL2 | ratio |
|--------|---------|--------------|----------------|-------|
| 17_cln1_cln3_KO | C | 7.76 | 3.05 | 0.39× |
| 41_sic1_cdc6_cdh1 | C | 1136.7 | 499.9 | 0.44× |
| 21_cln1_cln2_cln3_cdh1 | C | 660.8 | 295.6 | 0.45× |
| 113_bub2_KO | B | 175.1 | 90.9 | 0.52× |
| 78_cdc20_ts_mad2 | C | 1.57 | 0.87 | 0.55× |

#### (4) boundary 退化的 mutant (ts 突变体, 0 值邻域采样不适用)

| Mutant | Pattern | random RelL2 | boundary RelL2 | ratio |
|--------|---------|--------------|----------------|-------|
| 89_tem1_ts_GAL_CDC15 | C | 0.93 | 4.77 | 5.13× |
| 100_GAL_CDC14 | C | 1.70 | 5.05 | 2.98× |
| 88_tem1_ts_multi_CDC15 | C | 0.88 | 2.61 | 2.96× |
| 91_tem1_ts_multi_CDC14 | C | 1.17 | 2.86 | 2.45× |
| 90_tem1_net1_ts | B | 0.22 | 0.52 | 2.39× |

退化的 mutant 集中在 tem1/cdc15/cdc14 相关 ts 突变体, ts 突变本质是"部分功能丧失"不是完全 KO, 0 值邻域采样不适用。

#### (5) 共性难点 mutant (所有策略都差, 均含 cdh1 多突变)

| Mutant | random RelL2 | 最差策略 | 最好策略 |
|--------|--------------|---------|---------|
| 41_sic1_cdc6_cdh1 | 1136.7 | maal 2159 | boundary 500 |
| 45_swi5_cdh1 | 704.8 | maal 2000 | boundary 779 |
| 42_sic1_cdc6_cdh1_GAL_CDC20 | 924.1 | diversity 2215 | boundary 599 |
| 21_cln1_cln2_cln3_cdh1 | 660.8 | diversity 1862 | boundary 296 |
| 122_APC_A_cdh1_multi_CDC20 | 622.6 | uncertainty 2163 | boundary 517 |

共同特征: 多基因突变 + 含 cdh1, cdh1 突变对系统动力学影响剧烈, 所有策略难以预测。

### 6.5 实验结论

1. **boundary 是当前最优策略**: 在所有 Pattern 上都优于 random, 整体 Rel L2 最低, 胜出 mutant 数最多 (57/126)
2. **diversity/maal 严重过拟合**: lhs_test 表现最佳但 real_mutants 表现最差, 332 维特征中参数权重过低
3. **boundary 对 ts 突变体退化**: tem1_ts/cdc15_ts/cdc14_ts 系列 0 值邻域采样不适用
4. **cdh1 复杂突变体是共性难点**: 所有策略都灾难性差, 需定向增强

---

## 7. v2.2 实验结果 (5 策略对比, modes32_seed42)

> ⚠️ **本章实验结果已作废 (VOID)**: 这批实验在 best checkpoint / 早停均依据 `test_loss_real` (real mutants 测试集) 的错误配置下运行, 存在模型选择数据泄露; 且 boundary 与 diversity 结果逐字节重复 (权重完全相同)。现已回退为依据 `val_loss_lhs`。以下数据**仅作问题复盘, 不可作为策略优劣结论**, 需在修正配置后重跑。

> 实验目录: `eval_result/hyper_fno_al4pde_{strategy}_mse_only_v1_al4pde_modes32_seed42/` (maal 单独命名)
> 训练配置: v2.2 P1-P5 全部生效 — 初始训练集 5000, 分段式触发 8/4/8, 递增 num_add 1000/3000/4000, max_epochs=150 + EarlyStopping(monitor=`test_loss_real`, patience=20), best checkpoint 依据 `test_loss_real`。方案 A 分组特征权重 (MASS×0.5, 参数×2.5) + cdh1 density 500 亦生效。
> 日志: `logs/hyper_fno_al4pde_{strategy}_..._seed42.log`

### 7.1 整体性能对比 (global_metrics_denorm.json, all_39_vars)

| 策略 | real_RelL2 | real_Corr | real_MAE | real_MSE | lhs_test_RelL2 | best epoch | 停止 epoch | 达到 al_iter | AL 新增样本 |
|------|-----------|-----------|----------|----------|----------------|-----------|-----------|-------------|------------|
| **random** | **0.330** | **0.904** | **0.326** | 13.56 | 0.163 | 57 | 77 | 15 | ~28.7K |
| uncertainty | 0.339 | 0.892 | 0.334 | **11.96** | 0.160 | 89 | 109 | 23 | ~54.3K |
| diversity | 0.465 | 0.800 | 0.431 | 16.34 | **0.159** | 22 | 42 | 6 | ~11K |
| boundary | 0.465 | 0.800 | 0.431 | 16.34 | 0.159 | 22 | 42 | 6 | ~11K |
| maal | 0.408 | 0.844 | 0.401 | 17.34 | 0.176 | 68 | 88 | 18 | ~39K |

> AL 新增样本 = 停止时累计新增 (计划总量 76K)。random 仅完成 38%, uncertainty 71%, maal 51%, diversity/boundary 仅 14%。

### 7.2 per_mutant 聚合对比 (126 real mutants, Rel L2)

| 策略 | Rel L2 均值 | Rel L2 中位数 | 胜出 mutant 数 |
|------|------------|--------------|---------------|
| **random** | **160.9** | **1.64** | **64/126** ← 最优 |
| uncertainty | 243.1 | 1.83 | 34/126 |
| diversity | 238.3 | 2.48 | 13/126 |
| boundary | 238.3 | 2.48 | 0/126 (与 diversity 完全并列) |
| maal | 291.2 | 2.00 | 15/126 |

### 7.3 按 Pattern 分类对比 (Rel L2 均值)

| Pattern | n | random | uncertainty | diversity | boundary | maal |
|---------|---|--------|-------------|-----------|----------|------|
| Pattern_A (Multi-Division) | 20 | **211.1** | 326.8 | 362.3 | 362.3 | 373.0 |
| Pattern_B (Single-Division) | 45 | **103.5** | 155.1 | 131.5 | 131.5 | 213.3 |
| Pattern_C (Exponential-Arrest) | 61 | **186.8** | 280.7 | 276.5 | 276.5 | 321.8 |

### 7.4 关键发现

#### (1) ⚠️ v2.2 相对 v2.1 全面回退 (P1-P5 训练节奏改动为主因)

| 策略 | v2.1 real_RelL2 | v2.2 real_RelL2 | 变化 | v2.1 per_mutant | v2.2 per_mutant |
|------|-----------------|-----------------|------|-----------------|-----------------|
| random | 0.185 | 0.330 | **+78% 变差** | 111.2 | 160.9 |
| uncertainty | 0.251 | 0.339 | +35% 变差 | 178.3 | 243.1 |
| diversity | 0.361 | 0.465 | +29% 变差 | 185.3 | 238.3 |
| boundary | **0.236** | 0.465 | **+97% 变差** | 86.3 | 238.3 |
| maal | 0.458 | 0.408 | -11% 略好 | 245.2 | 291.2 |

除 maal 外, 所有策略 real_mutants 与 lhs_test 均全面变差。**说明 v2.2 的 P2-P5 训练节奏改动 (初始 5000 + 分段 trigger + 递增 num_add + EarlyStopping) 整体上损害了性能, 而非改善。**

#### (2) 🐛 EarlyStopping + 后置 AL 调度冲突 (核心根因)

- best checkpoint 依据 `test_loss_real` (P1) + EarlyStopping(patience=20, monitor=`test_loss_real`) (P5)。
- 但 P3/P4 的 AL 数据是**后置**的: 中期 4-epoch 高频触发从 epoch 31 才开始, num_add 在 al_iter≥10/≥20 才升到 3000/4000。
- 结果: 模型在 AL 数据尚未大量注入前, `test_loss_real` (126 个 real mutant, 噪声大) 已停止改善 → 过早早停。
  - diversity/boundary: best epoch=**22**, 停止 epoch=42, 仅注入 **11K/76K (14%)** AL 样本。
  - random: best epoch=57, 仅注入 28.7K/76K (38%)。
  - **越晚停 (=越多 AL 数据) 的策略性能越好** (uncertainty 71%、maal 51% 达到更高 al_iter, random best epoch 更晚), 与 "AL 数据越多越好" 一致, 反证早停扼杀了 AL。
- 叠加 P2 (初始 15000→5000): 早期训练样本进一步被削弱, 早停时模型总见样本量远低于 v2.1 的 66K。

#### (3) 🐛 boundary 与 diversity 结果完全相同 (实验无效, 需重跑)

- `per_mutant_metrics.csv` / `global_metrics_denorm.json` **逐字节相同** (md5 一致)。
- best checkpoint 均为 `best-epoch=22-test_loss_real=1.9264.ckpt`, 经 torch 加载比对: **模型权重逐元素差异 = 0.0, global_step 均为 15318** → 两者训练到 epoch 22 时是完全相同的模型。
- 但两者选样代码路径不同: diversity 用 `_bmdal_max_dist_select` (贪心 MaxDist), boundary 用 `np.argsort(0.6·diversity + 0.4·boundary_proximity)[-k:]` (top-k), 且 boundary 日志确实打印了 boundary_proximity (mean≈0.39, weight=0.40)。
- 结论: 在 v2.2 早停 (epoch 22 即为 best) 之前, boundary 前几轮 AL 选样与 diversity **实际收敛到同一批样本**, 使 boundary 的边界偏好完全未体现。**该 boundary 实验数据无效, 结论不可用**, 必须排查选样分歧点并重跑。

#### (4) random 成为最优策略

- random 在 3 个 Pattern、per_mutant 均值/中位数、胜出数 (64/126) 上全面第一。
- 语义: 在当前 (早停受限) 训练预算下, 复杂的 diversity/boundary/maal 选样非但没带来增益, 反而因过早停在低 al_iter 而劣于随机采样。uncertainty 因训练更久 (best epoch 89) 接近 random。

#### (5) cdh1 复杂突变体仍是共性灾难 (cdh1 density 500 增强未见效)

| 组 | n | random | uncertainty | diversity | boundary | maal |
|----|---|--------|-------------|-----------|----------|------|
| cdh1 相关 | 25 | **734.3** | 1117.8 | 1114.4 | 1114.4 | 1329.9 |
| ts 相关 | 22 | **77.2** | 119.7 | 122.1 | 122.1 | 159.6 |

random 下最差的 8 个 mutant 有 7 个含 cdh1 (54_GAL_CLB2_cdh1=1586, 41_sic1_cdc6_cdh1=1567, 32_..._cdh1=1452, 42_..._cdh1=1288, 45_swi5_cdh1=986, 6_..._cdh1_KO=984, 46_swi5_cdh1=907)。cdh1 density 500 定向增强在早停预算下**无法验证是否有效** (diversity/boundary 甚至没跑到相应轮次)。

### 7.5 实验结论

1. **v2.2 是一次回退**: P2-P5 训练节奏改动 + P1 best 依据整体损害性能, 除 maal 外全部策略变差。
2. **早停与 AL 调度冲突是主因**: EarlyStopping(patience=20 on 噪声 `test_loss_real`) 在后置 AL 数据注入前触发, diversity/boundary 仅注入 14% AL 样本即停止。
3. **boundary 实验无效**: 与 diversity 权重逐元素相同 (diff=0.0), 边界偏好未体现, 必须重跑。
4. **random 反成最优**: 说明复杂选样策略的价值被早停预算完全掩盖, 需先修复训练节奏再评估选样算法。
5. **cdh1 仍是共性灾难**: 定向增强在当前预算下无法验证, ts 突变体也普遍较差。

---

## 8. v2.3 实验结果 (5 策略对比, modes32_seed42)

> ✅ **本批结果有效**: v2.3 外层重训式 (5 轮冷启动, 每轮全新余弦跑满 100ep, 取最后一轮最后 epoch), 初始 5000 + 4×19000 = 81000, 无早停、无 val-best 选择 (不碰任何验证/测试集做模型选择), 从根本消除 v2.2 的选择性数据泄露。
> 实验目录: `eval_result/hyper_fno_al4pde_{strategy}_mse_only_v1_al4pde_modes32_seed42/` (maal 单独命名)
> 指标口径: `global_metrics_denorm.json` 的 `all_39_vars` (denormalized)。

### 8.1 整体性能对比 (all_39_vars)

| 策略 | real_RelL2 | real_Corr | real_MAE | real_MSE | lhs_RelL2 | lhs_Corr | lhs_MSE | real/lhs 比 |
|------|-----------|-----------|----------|----------|-----------|----------|---------|------------|
| **random** | **0.1520** | 0.9784 | 0.1315 | **1.627** | 0.1314 | 0.9803 | 1.450 | 1.16× |
| **diversity** | **0.1505** | **0.9794** | 0.1402 | 2.610 | **0.1204** | **0.9849** | 1.625 | 1.25× |
| **boundary** | 0.1508 | 0.9789 | **0.1280** | **1.606** | 0.3179 ⚠️ | 0.9123 | 34.11 ⚠️ | 0.47× (反转) |
| uncertainty | 0.1576 | 0.9773 | 0.1439 | 2.301 | 0.1524 | 0.9769 | 2.569 | 1.03× |
| maal | 0.2458 | 0.9435 | 0.1885 | 3.902 | 0.1424 | 0.9798 | 2.113 | 1.73× |

> core_9_vars real_RelL2: random 0.147 / diversity 0.157 / boundary 0.147 / uncertainty 0.161 / maal 0.240 (排序与 all_39 一致)。

### 8.2 相对 v2.1 (上一批有效结果) 的改进

| 策略 | v2.1 real_RelL2 | v2.3 real_RelL2 | 变化 | v2.1 real/lhs 比 | v2.3 real/lhs 比 |
|------|-----------------|-----------------|------|------------------|------------------|
| random | 0.185 | **0.152** | **−18%** | 1.61× | 1.16× |
| uncertainty | 0.251 | **0.158** | **−37%** | 1.93× | 1.03× |
| diversity | 0.361 | **0.151** | **−58%** | 3.65× | 1.25× |
| boundary | 0.236 | **0.151** | **−36%** | 1.86× | 0.47× |
| maal | 0.458 | **0.246** | **−46%** | 3.14× | 1.73× |

**全策略 real 性能大幅提升 (−18%~−58%), 且 sim-to-real gap 基本闭合** (real/lhs 比从 1.6~3.7× 降到 ~1.0~1.3×)。这是项目历史上 real mutants 的最佳表现。

### 8.3 关键发现

#### (1) 🏆 v2.3 范式本身是最大功臣, 而非选样策略
- real_RelL2 从 v2.1 的 0.19~0.46 全面降到 ~0.15, diversity 更是从 0.361 (重度过拟合) 降到 0.151。
- 主因是三件事协同: **外层重训跑满收敛** + **数据预算 81000** + **消除 val/early-stop 泄露 (取最后 epoch)**。选样策略只是二阶因素。

#### (2) random 与 diversity 并列最优, 且差异极可能落在噪声内
- real_RelL2: random 0.1520 / diversity 0.1505 / boundary 0.1508 — **三者相差 <0.0015**, 在 n=126 real mutants 上几乎不可能有统计显著性。
- 语义: 在 81000 样本 + 跑满收敛的充裕预算下, **数据量主导, 复杂 acquisition 的边际价值被抹平**。random 无需 ODE 预计算 / MC dropout / 特征提取, **性价比最高**。

#### (3) ⚠️ boundary 异常: real 最佳但 lhs_test 崩溃
- boundary real 端最好 (MAE=0.128 最低, MSE=1.606 最低), 但 **lhs_test 灾难** (RelL2=0.318, MSE=34.11, 是其他策略的 ~20 倍)。
- 成因: boundary acquisition (0.4 权重偏向 0 值/KO 邻域) 使训练集严重偏向 boundary 区。real mutants 多为 KO/边界型生物扰动 → boundary 命中; 但广义 LHS 测试集含大量过表达/非边界区 → 被欠拟合, 少数大幅值样本 MSE 爆炸。
- 这是 boundary **设计权衡的极端化**: 牺牲 LHS 分布覆盖换取 real 边界区精度。若部署目标只是 real mutants, boundary 反而有吸引力; 若要兼顾 lhs, 需调低 boundary_weight。

#### (4) maal 仍是垫底策略
- real_RelL2=0.246, 显著差于其余四者 (~0.15), sim-to-real 比 1.73× 也最差。延续 v2.2 诊断 (不确定性估计偏弱 + target_priority 区分度不足), v2.3 确认 maal 的加权机制拖累了泛化。

#### (5) boundary 日志出现两次 eval (无害)
- boundary run 第 1 次 train+eval 于 22:09 正常完成 (指标已写), 2.5h 后 (00:44) 又追加了一次纯 eval, 复用同一 `final-last-epoch.ckpt` + 同一数据集 → 指标逐字节相同, 仅重复覆盖 JSON/CSV。**对结果无影响**, 疑因 pipeline 被二次启动 (boundary 行仍激活) 或手动重跑 eval 且未经截断 header 分支。

### 8.4 优化建议

1. **把 random 定为默认基线并重点汇报**: 它与最优并列却最省算力 (无 ODE 预计算 / MC / 特征提取)。任何复杂策略必须证明能**显著**超过 random 才有采用价值。
2. **补多 seed (42/1024/3407) 做显著性检验**: random/diversity/boundary 在 real 上相差 <0.0015, 当前无法断言谁更优。多 seed 是判定"策略差异 vs 随机噪声"的必要步骤。
3. **boundary_weight 扫描 (0.2/0.3/0.4)**: 修复 lhs_test 崩溃同时保留 real 端优势 (MAE/MSE 最佳); 或把 boundary 样本与 normal/random 样本混采以恢复 LHS 覆盖。
4. **diversity 作为"均衡之选"**: lhs 最佳 (0.120) + real 竞争力 (0.151, Corr 最高)。若需要单一稳健策略兼顾两分布, diversity 优于 boundary。
5. **maal 重构或下线**: 连续两版垫底, 建议要么按 v2.2 诊断修 target_priority, 要么移出对比集止损。
6. **锁定 v2.3 训练范式**: 后续增益主要来自**数据** (候选池质量/覆盖) 而非更花哨的 acquisition。可优先投入 normal/density 采样质量与 cdh1 等难点家族的定向补充。
7. **修复双 eval 隐患**: 跑完一批后注释掉已完成实验行 (或用 `git` 追踪), 避免 pipeline 二次启动对同一 RUN_NAME 追加日志; 必要时在 eval 前也加一次日志截断。

### 8.5 实验结论

1. **v2.3 是决定性的成功版本**: 全策略 real 大幅改善, sim-to-real gap 基本闭合, 是项目历史最佳。
2. **数据量 + 跑满收敛 + 无泄露 = 真正的杠杆**, 选样策略退居次要。
3. **random ≈ diversity ≈ boundary(real 端)**, 差异疑在噪声内, 需多 seed 确认。
4. **boundary 走极端**: real 最佳 / lhs 崩溃, 是设计权衡的放大, 按部署目标取舍或调 weight。
5. **maal 稳定垫底**, 待修或下线。
