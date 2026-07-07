# LHS 数据集生成引擎说明文档

> **位置**: `/home/users/zsy/trae_workspace/budding_yeast_v2/data/`
> **维护规则**: 每次对数据生成代码做改进，必须同步更新本文档对应版本章节与"更新日志"
> **当前版本**: V5（基于 V4 eval 残留瓶颈的 7 项定向修复 + 下游分析 regex bug 修复）
> **维护人**: zsy

---

## 一、文件清单

| 文件 | 版本 | 状态 | 说明 |
|------|------|------|------|
| `lhs_v0.py` | V0 | 仅留作历史参考 | 早期版本，已被 V1 取代 |
| `lhs_v1_origin.py` | V1 | 当前基线 | 基础 LHS 采样引擎，124 突变体 + 1000 LHS/突变体 |
| `lhs_v1_origin_v2.py` | V2 | 已发布 | V1 + 三大家族定向扩展（cdc20_ts/CLB1_clb2/cdh1）+ N_LHS 补足 |
| `lhs_v1_origin_v3.py` | V3 | 已生成+已分析 | V2 + 残留瓶颈定向扩展（27 个 MAE>0.3 突变体，5 类家族） |
| `lhs_v1_origin_v4.py` | V4 | 已生成 | V3 + 邻域污染修复 + 覆盖盲区扩展（12 类家族，67 突变体，~44K 定向样本） |
| `lhs_v1_origin_v5.py` | V5 | 已生成+已分析 | V4 + 7 项定向修复（17 类家族，88 突变体，~54K 定向样本） |
| `DATASET_GENERATION_GUIDE.md` | - | 本文件 | 数据生成引擎说明与版本变更记录 |

---

## 二、整体采样流程（所有版本通用）

数据集生成遵循 **5 阶段流水线**，从突变体规则字典出发，最终输出 `.npz` 压缩文件。

### 阶段 1: 任务组装（Task Assembly）

```python
all_tasks = {}
all_tasks.update(mutant_rules)             # ① 124 个真实突变体（含 3 个 WT）
all_tasks.update(generate_lhs_tasks(N))    # ② 全局 LHS 采样（每非-WT 突变体 N 个样本）
# V2 新增：
all_tasks.update(generate_targeted_lhs_tasks())                      # ③ 三大家族定向扩展
# V3 新增（V4 中被替代，不再调用）：
# all_tasks.update(generate_targeted_lhs_tasks(V3_EXPANSION_FAMILIES))
# V4 新增（替代 V3，V5 中被替代）：
# all_tasks.update(generate_targeted_lhs_tasks(V4_EXPANSION_FAMILIES))
# V5 新增（替代 V4）：
all_tasks.update(generate_targeted_lhs_tasks(V5_EXPANSION_FAMILIES))  # ④ 7 项定向修复 + 17 类家族
```

**键名约定**（用于下游 `_LHS_` 子串匹配）：
- 真实突变体: `{id}_{name}`（如 `73_cdc20_ts`）
- 全局 LHS: `{mutant_name}_LHS_{idx:03d}`（如 `73_cdc20_ts_LHS_042`）
- 定向 LHS: `{mutant_name}_LHS_T{idx:03d}`（T=Targeted，如 `73_cdc20_ts_LHS_T007`）
- RGS 局部: `{mutant_name}_RGS_{idx:03d}`（仅 al_callback 动态生成时使用）

### 阶段 2: ODE 并行求解

- **求解器**: `scipy.integrate.solve_ivp`，方法 `BDF`（刚性系统）
- **精度**: `rtol=1e-4, atol=1e-6`
- **事件检测**: 4 个 `terminal` 事件
  - `event1`: `CLB2 - 0.3`（方向 -1）→ 触发细胞分裂
  - `event2`: `CLB2 + CLB5 - 0.2`（方向 -1）→ 触发 ORI 复位
  - `event3`: `ORI - 1.0`（方向 +1）→ 触发 Vi20/BUB2 激活
  - `event4`: `SPN - 1.0`（方向 +1）→ 触发后期事件
- **超时保护**: `SIM_TIMEOUT = 5.0s`（防 Fortran 死锁）
- **并行**: `ProcessPoolExecutor(max_workers=cpu_count//4)`（绅士模式，留 3/4 核给 GPU）
- **Mass 限制**: `SIM_TF <= 210` 且 `max_mass > 10.0` 时丢弃样本

### 阶段 3: 双重打标系统

每个成功求解的样本生成两个标签：

**生物学机理标签**（`bio_label`，由 `analyze_cell_viability_unified` 生成）：
- `Viable`
- `Inviable: G1 Arrest`
- `Inviable: M-phase Arrest`
- `Inviable: Unbudded Division`
- `Inviable: Premature Division`
- `Inviable: Sequence Failure`

**视觉形态学标签**（`pattern_label`，由 `analyze_mass_pattern` 生成）：
- `Pattern_A: Multi-Division (>=2)` — 至少 2 次有效 Mass 跳变
- `Pattern_B: Single-Division (==1)` — 仅 1 次有效 Mass 跳变
- `Pattern_C: Exponential-Arrest (==0)` — 无有效跳变（指数增长后停滞）

### 阶段 4: 动态浓度异常剔除

第三级防御，剔除 ODE 发散的 LHS 样本：
```python
real_envelope = max(所有 Real Mutant 的逐变量极值)        # 39 维
dynamic_thresholds = max(real_envelope * 2.0, 2.0)        # 双门限
for lhs_sample in LHS_samples:
    if any(lhs_max_vals[var] > dynamic_thresholds[var]):
        delete(lhs_sample)                                  # 怪物细胞拦截
```

### 阶段 5: 对齐与保存

- **参数键序**: `param_keys = sorted([k for k in defaults if k != 'is_mutant_104'])`（共 141 维）
- **输出路径**: `/home/users/zsy/trae_workspace/mydata/budding_yeast/lhs_cheat_origin_{ver}_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`
- **保存内容**:
  ```python
  np.savez_compressed(DATASET_PATH,
      data=all_tensors,           # (N, 39, 500) float — 轨迹
      params=all_params,          # (N, 141) float — 参数向量
      param_names=param_keys,     # (141,) str — 参数名
      mutant_names=final_names,   # (N,) str — 样本名
      labels=all_bio_labels,      # (N,) str — 生物学标签
      pattern_labels=all_pattern_labels)  # (N,) str — 形态学标签
  ```

---

## 三、核心采样引擎

### 3.1 全局 LHS 采样 `generate_lhs_tasks(num_lhs_samples)`

**职责**: 为每个非-WT 突变体生成 `num_lhs_samples` 个全局 LHS 扰动样本。

**采样范围**（连续参数）:
```python
l_bound = 0.0 if min_val <= 0 else min_val * 0.1
u_bound = 0.01 if max_val <= 0 else max_val * 1.5
# 全局覆盖 [0, max_val × 1.5]
```

**关键特性**:
- 离散参数从 `single_pools` 随机选（由 `build_discrete_pools` 自动构建）
- `CDH1T/CDH1` 强绑定元组（满足 `活性 = 总数 × 比例` 生物学约束）
- `BUB2_active/init_BUB2` 同频连续缩放（coeff ∈ [0, 1.5]）
- 等价碰撞拦截器（`np.isclose` 精度对齐，丢弃与母体一致的克隆）
- 纯离散突变体智能截断（防止生成数百个完全相同的克隆）

### 3.2 RGS 局部密集采样 `generate_local_lhs_tasks(target_mutants, ...)`

**职责**: AL 触发时基于 Real Mutant MAE 反馈做局部密集采样。

**与 `generate_lhs_tasks` 的唯一区别**:
```python
# generate_lhs_tasks: 全局范围 [0, max_val × 1.5]
# generate_local_lhs_tasks: 局部范围 [center × (1 - pert), center × (1 + pert)]
```

**键名后缀**: `_RGS_{idx:03d}`（区别于全局 LHS 的 `_LHS_`）

### 3.3 定向扩展采样 `generate_targeted_lhs_tasks(target_families, ...)`

**职责**: 按 `target_families` 字典配置，对指定家族做定向密集采样，补齐 LHS 训练分布的覆盖盲区。

#### 3.3.1 关键变量与术语解释

在深入三种采样模式之前，先澄清公式中出现的几个关键变量：

| 变量名 | 含义 | 来源 | 具体数值例子 |
|--------|------|------|-------------|
| `center` / `center_val` | 突变体规则中该连续参数的**原始突变值**，作为定向采样的中心点 | `original_rules[p_name]`（即 `mutant_rules[mutant_name][param]`） | `73_cdc20_ts` 的 `ks20p=0.006`；`47_clb1_clb2_KO` 的 `ksb2p=0.001`；`34_cdh1_KO` 的 `init_CDH1T=0.0` |
| `pert` / `perturbation` | **扰动幅度**（相对比例），控制采样范围围绕 center 的扩展宽度。`pert=0.1` 表示 ±10% | 家族配置中的 `perturbation` 字段 | V2 cdc20_ts 家族 `pert=0.1`；V2 CLB1_clb2 家族 `pert=0.2`；V3 contamination_fix 家族 `pert=0.08` |
| `b_val` | **boundary 模式的目标边界值**，即希望采样范围覆盖到的"彻底敲除"目标值（通常为 0.0）。**仅在 boundary 模式且该参数出现在 `boundary_params` 字典中时才使用** | 家族配置中的 `boundary_params[p_name]` | V2 cdc20_ts 家族 `boundary_params={'ks20p': 0.0, 'ks20pp': 0.0}` → `b_val=0.0`；V3 KO_family `boundary_params={'Tem1T': 0.0, 'Cdc15T': 0.0}` → `b_val=0.0` |
| `ε` (epsilon) | **小偏移量**（代码中硬编码为 `1e-6`），防止 `u_bound` 精确等于 `b_val` 导致 LHS 采样退化到单点 | 代码常量 | `b_val + 1e-6 = 0.000001` |
| `anchor_params` | **anchor 模式的强制锚点值字典**，直接指定 CDH1T/CDH1 的取值（绕过 cdh1_pairs 随机选择），用于强制实现 CDH1T/CDH1=(0,0) 的彻底敲除锚点 | 家族配置中的 `anchor_params` 字段 | V2 cdh1 家族 `anchor_params={'init_CDH1T': 0.0, 'init_CDH1': 0.0}` → 所有 500 样本的 init_CDH1T/init_CDH1 都被强制设为 0.0 |
| `cdh1_pairs` | **CDH1T/CDH1 强绑定元组列表**，由 `build_discrete_pools(mutant_rules)` 自动扫描所有突变体规则构建。每个元组 `(init_CDH1T, init_CDH1)` 满足生物学约束 `活性 = 总数 × 比例`。**density/boundary 模式下**，若突变体涉及 CDH1T/CDH1 突变，则从该列表随机选一个元组赋值（保留生物学约束） | `build_discrete_pools()` 返回值 | V2/V3 数据集中 `cdh1_pairs = [(1.0, 0.9304992), (0.0, 0.0)]`（WT 元组 + cdh1_KO 元组）→ density/boundary 模式下随机选其一 |
| `single_pools` | **离散参数的合法真值集合字典**，由 `build_discrete_pools` 构建。对 `mdt`/`BUB2_active`/`ESP1T` 等离散参数，从对应 pool 随机选 | `build_discrete_pools()` 返回值 | `single_pools['mdt'] = [90.0, 150.0, 160.0]`（WT 90 + 两个突变体）；`single_pools['ESP1T'] = [1.0, 3.0]` |

#### 3.3.2 三种采样模式详解

**家族配置 schema**:
```python
{
    'family_name': {
        'mutants': ['mutant_id1', 'mutant_id2', ...],
        'num_per_mutant': int,             # 每突变体采样数
        'perturbation': float,             # 扰动幅度 (0.1 = ±10%)
        'mode': 'density' | 'boundary' | 'anchor',
        'boundary_params': {param: val},   # mode='boundary' 时必填
        'anchor_params': {param: val}      # mode='anchor' 时必填
    }
}
```

##### 模式 1: `density` — 邻域密度增强（最安全）

**连续参数采样范围**: `[max(0, center × (1-pert)), center × (1+pert)]`

**CDH1T/CDH1 处理**: 从 `cdh1_pairs` 随机选一个元组（保留生物学约束）

**设计意图**: 在原参数邻域做局部 LHS 扰动，增加采样密度，**不引入新的边界值**。最安全，不会引发邻域污染。

**具体数值例子**（V3 contamination_fix 家族，`76_cdc20_pds1_clb5` 突变体）:
```python
# 家族配置: perturbation=0.08, mode='density'
# 突变体 76_cdc20_pds1_clb5 的 original_rules 包含 ks20p=0.006, ks20pp=0.6
# 对 ks20p 参数:
center_val = 0.006
pert = 0.08
l_bound = max(0.0, 0.006 × (1 - 0.08)) = max(0.0, 0.00552) = 0.00552
u_bound = 0.006 × (1 + 0.08) = 0.00648
# LHS 在 [0.00552, 0.00648] 范围内采样 800 个样本
# → 采样范围极窄 (±8%)，精准修复 V2 boundary 邻域污染

# 对 ks20pp 参数:
center_val = 0.6
l_bound = max(0.0, 0.6 × 0.92) = 0.552
u_bound = 0.6 × 1.08 = 0.648
# LHS 在 [0.552, 0.648] 范围内采样
```

##### 模式 2: `boundary` — 边界扩展到 0（覆盖彻底敲除）

**连续参数采样范围**: `[max(0, min(b_val, center × (1-pert))), max(center × (1+pert), b_val + 1e-6)]`

**CDH1T/CDH1 处理**: 从 `cdh1_pairs` 随机选一个元组

**设计意图**: 扩展连续参数的采样范围到 0 边界，覆盖 Real Mutant 的彻底敲除参数（如 `ks20p=0`）。**适用于 KO 突变体**，但可能引入邻域污染（因采样范围跨越多个数量级）。

**具体数值例子**（V2 cdc20_ts 家族，`73_cdc20_ts` 突变体）:
```python
# 家族配置: perturbation=0.1, mode='boundary', boundary_params={'ks20p': 0.0, 'ks20pp': 0.0}
# 突变体 73_cdc20_ts 的 original_rules 包含 ks20p=0.006, ks20pp=0.6
# 对 ks20p 参数 (出现在 boundary_params 中):
center_val = 0.006
b_val = 0.0  # 来自 boundary_params['ks20p']
pert = 0.1
l_bound = max(0.0, min(0.0, 0.006 × 0.9)) = max(0.0, 0.0) = 0.0  # 扩展到 0 边界
u_bound = max(0.006 × 1.1, 0.0 + 1e-6) = max(0.0066, 0.000001) = 0.0066
# LHS 在 [0.0, 0.0066] 范围内采样 500 个样本
# → 覆盖了从 0 (彻底敲除) 到 0.0066 (过表达) 的完整范围

# 对 ks20pp 参数 (同样出现在 boundary_params 中):
center_val = 0.6
b_val = 0.0
l_bound = max(0.0, min(0.0, 0.54)) = 0.0
u_bound = max(0.66, 1e-6) = 0.66
# LHS 在 [0.0, 0.66] 范围内采样
```

**V3 KO_family 数值例子**（`86_tem1_KO` 突变体）:
```python
# 家族配置: perturbation=0.1, mode='boundary', boundary_params={'Tem1T': 0.0, 'Cdc15T': 0.0}
# 突变体 86_tem1_KO 的 original_rules 包含 Tem1T=1.0
# 对 Tem1T 参数:
center_val = 1.0
b_val = 0.0
pert = 0.1
l_bound = max(0.0, min(0.0, 0.9)) = 0.0  # 扩展到 0
u_bound = max(1.1, 1e-6) = 1.1
# LHS 在 [0.0, 1.1] 范围内采样 500 个样本 → 覆盖 Tem1T=0 彻底敲除
```

##### 模式 3: `anchor` — CDH1T/CDH1 强制锚点（高风险高收益）

**连续参数采样范围**: `[max(0, center × (1-pert)), center × (1+pert)]`（与 density 相同）

**CDH1T/CDH1 处理**: **直接使用 `anchor_params` 字典中的值**（绕过 cdh1_pairs 随机选择），强制实现 CDH1T/CDH1=(0,0) 锚点

**设计意图**: 已知 CDH1T/CDH1 强绑定约束时，强制所有样本的 init_CDH1T=0/init_CDH1=0，生成纯 CDH1T=0 的 LHS 样本。**高风险**：邻域污染风险高（V2 的 4 个 cdh1 突变体因 anchor 模式表现不佳，V3 改用 density 修复）。

**具体数值例子**（V2 cdh1 家族，`34_cdh1_KO` 突变体）:
```python
# 家族配置: perturbation=0.1, mode='anchor', anchor_params={'init_CDH1T': 0.0, 'init_CDH1': 0.0}
# 突变体 34_cdh1_KO 的 original_rules 包含 init_CDH1T=0.0, init_CDH1=0.0, kscdh=0.0, kdcdh=0.0
# 对 init_CDH1T (在 anchor_params 中):
# → 不走 cdh1_pairs 随机选择，直接赋值 sample_dict['init_CDH1T'] = 0.0
# → 所有 500 个样本的 init_CDH1T 都被强制设为 0.0
# 对 init_CDH1 (在 anchor_params 中):
# → 直接赋值 sample_dict['init_CDH1'] = 0.0
# 对 kscdh (连续参数, 不在 anchor_params 中):
center_val = 0.0  # original_rules['kscdh'] = 0.0
pert = 0.1
l_bound = max(0.0, 0.0 × 0.9) = 0.0
u_bound = 0.0 × 1.1 = 0.0
# u_bound == l_bound, 触发 fallback: u_bound = l_bound + 1e-6 = 1e-6
# LHS 在 [0.0, 1e-6] 范围内采样 → 实际上 kscdh 几乎全为 0
```

#### 3.3.3 三种模式对比汇总

| 模式 | 连续参数采样范围 | CDH1T/CDH1 处理 | 风险等级 | 适用场景 |
|------|------------------|------------------|----------|----------|
| `density` | `[max(0, center×(1-pert)), center×(1+pert)]` | 从 `cdh1_pairs` 随机选 | 🟢 低（最安全） | 邻域密度增强、邻域污染修复 |
| `boundary` | `[max(0, min(b_val, center×(1-pert))), max(center×(1+pert), b_val+1e-6)]` | 从 `cdh1_pairs` 随机选 | 🟡 中（可能邻域污染） | KO 突变体、彻底敲除覆盖 |
| `anchor` | `[max(0, center×(1-pert)), center×(1+pert)]` | 直接用 `anchor_params` 强制赋值 | 🔴 高（邻域污染风险高） | CDH1T/CDH1 强制 (0,0) 锚点 |

**键名后缀**: `_LHS_T{idx:03d}`（T=Targeted，保持 `_LHS_` 子串匹配兼容）

---

## 四、版本演进

### V1（`lhs_v1_origin.py`）— 基础 LHS 采样引擎

**核心改进**（相对于 V0）:
1. 引入 `DISCRETE_STRICT_PARAMS` 严格离散保护名单，废弃 `endswith('T')` 字符串匹配
2. 实现 `CDH1T/CDH1` 强绑定元组机制，保障 `活性 = 总数 × 比例` 生物学约束
3. `BUB2_active/init_BUB2` 同频连续缩放系数 ∈ [0, 1.5]
4. 等价碰撞拦截器（`np.isclose` 精度对齐，rtol=1e-5, atol=1e-8）
5. 纯离散突变体智能截断（防止无意义克隆）
6. 双重打标系统（bio_label + pattern_label）
7. 动态浓度异常剔除系统（双门限：相对 2.0× + 绝对 2.0）
8. `eqns` 修复 `is_mutant_104` 的 180min 时序条件分支

**采样规模**:
- 124 突变体规则（3 WT + 121 非-WT）
- 1000 LHS 样本/突变体 → ~121,000 LHS + 124 Real = ~121,124 总样本

**输出**: `lhs_cheat_origin_v1_210min_500steps_dual_labels.npz`（即原始无 `_v1` 后缀的 `lhs_cheat_origin_210min_500steps_dual_labels.npz`）

---

### V2（`lhs_v1_origin_v2.py`）— 三大家族定向扩展（Sim-to-Real Gap 缓解）

**核心改进**（相对于 V1）:
1. **新增 `generate_targeted_lhs_tasks` 函数**，支持 density / boundary / anchor 三种采样模式（详见第 3.3 节）
2. **新增 `DEFAULT_TARGET_FAMILIES` 字典**，4 类家族定向扩展（cdc20_ts / CLB1_clb2 / cdh1 / low_lhs_count）
3. **引入 `boundary_params` 字段**：boundary 模式下指定目标边界值（如 `ks20p=0.0`），将采样范围扩展到 0
4. **引入 `anchor_params` 字段**：anchor 模式下指定 CDH1T/CDH1 强制锚点值（如 `init_CDH1T=0.0`）

**家族配置**（`DEFAULT_TARGET_FAMILIES`）:

| 家族 | 突变体数 | num_per_mutant | mode | 扰动 | boundary_params / anchor_params | 理论采样总数 |
|------|---------|----------------|------|------|----------------------------------|------------|
| `cdc20_ts` | 8 | 500 | boundary | 0.1 | `{'ks20p': 0.0, 'ks20pp': 0.0}` | 4,000 |
| `CLB1_clb2` | 5 | 500 | boundary | 0.2 | `{'ksb2p': 0.0, 'ksb2pp': 0.0}` | 2,500 |
| `cdh1` | 10 | 500 | anchor | 0.1 | `{'init_CDH1T': 0.0, 'init_CDH1': 0.0}` | 5,000 |
| `low_lhs_count` | 4 | 300 | density | 0.1 | 无（density 模式不需要） | 1,200 |
| **合计** | **27** | — | — | — | — | **12,700** |

**各家族设计动机详解**:

1. **cdc20_ts 家族**（boundary 模式）:
   - **问题**: V1 全局 LHS 在 `[0, 0.009]` 范围采样 ks20p，但 Real Mutant `73_cdc20_ts` 的 `ks20p=0.006`，且 cdc20_ts 突变体在 ts（temperature-sensitive）表型下 `ks20p` 趋近于 0
   - **解决**: boundary 模式将 ks20p 采样范围扩展到 `[0.0, 0.0066]`，覆盖彻底敲除（ks20p=0）到轻度过表达
   - **典型突变体**: `73_cdc20_ts`, `78_cdc20_ts_mad2`, `102_net1_cdc20_ts` 等 8 个

2. **CLB1_clb2 家族**（boundary 模式）:
   - **问题**: V1 全局 LHS 在 `[0, 0.0015]` 范围采样 ksb2p，但 Real Mutant `47_clb1_clb2_KO` 的 `ksb2p=0.001`，且 clb1/clb2 双敲除需要 `ksb2p=0`/`ksb2pp=0`
   - **解决**: boundary 模式将 ksb2p 采样范围扩展到 `[0.0, 0.0012]`（扰动 0.2 → `0.001 × 1.2 = 0.0012`），覆盖彻底敲除
   - **典型突变体**: `47_clb1_clb2_KO`, `48_CLB1_clb2`, `51_CLB1_clb2_cdh1` 等 5 个

3. **cdh1 家族**（anchor 模式）:
   - **问题**: V1 全局 LHS 中 CDH1T=0 样本数为 0，但 Real Mutant 中 24 个 cdh1 家族突变体的 `init_CDH1T=0.0`/`init_CDH1=0.0`，导致 Sim-to-Real Gap 的核心缺陷
   - **解决**: anchor 模式强制所有 5,000 个样本的 `init_CDH1T=0.0`/`init_CDH1=0.0`，生成纯 CDH1T=0 的 LHS 样本
   - **典型突变体**: `34_cdh1_KO`, `36_sic1_cdh1`, `41_sic1_cdc6_cdh1` 等 10 个

4. **low_lhs_count 家族**（density 模式）:
   - **问题**: V1 中 4 个突变体的常规 LHS 样本数不足（<300），训练覆盖不够
   - **解决**: density 模式在原参数邻域扰动（±10%），增加采样密度
   - **典型突变体**: `88_tem1_ts_multi_CDC15`, `89_tem1_ts_GAL_CDC15`, `15_cln3_bck2_multi_CLN2`, `53_GAL_CLB2_sic1`

**实验效果**（V2+none vs V1 baseline）:
- Real MAE: 0.3645 → 0.1926（**-47%**）
- cdc20_ts 家族 MAE: -97%
- Pattern C（指数停滞）MAE: -59%
- Sim-to-Real Gap: -34%
- LHS 中 CDH1T=0 样本数: 0 → 5,252（新增）
- CDH1T 平坦阶跃度: 57.75% → 99.90%（转为纯开关型）
- CDH1 (idx 22) 归一化路由: log_mm → zscore（适配开关特性）

**已知遗留问题**（V3 修复目标）:
- 26 个突变体 MAE > 0.3（残留瓶颈）
- V2 boundary 采样引入 9 个新恶化突变体（邻域污染，如 `76_cdc20_pds1_clb5` MAE 从 0.36 → 1.39）
- V2 anchor 邻域污染：4 个 cdh1 突变体仍表现不佳（`116_APC_A_cdh1`, `46_swi5_cdh1_GAL_SIC1` 等）
- V2 未采样家族：CLB2_db 家族（7 个突变体，MAE 均值 0.685）、cln 三突家族（4 个突变体，MAE 均值 0.617）、KO 家族（Tem1T=0/Cdc15T=0，3 个突变体）

**输出**: `lhs_cheat_origin_v2_210min_500steps_dual_labels.npz`

---

### V3（`lhs_v1_origin_v3.py`）— 残留瓶颈定向扩展

**核心改进**（相对于 V2）:
1. **新增 `V3_EXPANSION_FAMILIES` 字典**，针对 V2+none eval 中 26 个 MAE > 0.3 突变体，分 5 类家族定向修复
2. **修复 V2 boundary 邻域污染**：对 V1→V2 恶化的 9 个突变体用 density + 小扰动（0.08）+ 翻倍采样数（800）修复
3. **修复 V2 anchor 邻域污染**：对 cdh1 已采样但仍差的 4 个突变体改用 density 模式（扰动 0.10，采样数翻倍至 800）
4. **采样策略改进**：未采样家族用 density 模式（避免引入新的邻域污染），KO 突变体用 boundary 模式（覆盖到 0 边界）
5. **新增 KO 家族覆盖**：针对 V2 未采样的 Tem1T=0/Cdc15T=0 彻底敲除突变体，用 boundary 模式扩展

**家族配置**（`V3_EXPANSION_FAMILIES`）:

| 家族 | 突变体数 | num_per_mutant | mode | 扰动 | 理论采样总数 | 设计理由 |
|------|---------|----------------|------|------|------------|----------|
| `CLB2_db_family` | 7 | 500 | density | 0.15 | 3,500 | V2 未采样，MAE 均值 0.685，最大 1.84 |
| `cln_triple` | 4 | 500 | density | 0.15 | 2,000 | V2 未采样，MAE 均值 0.617 |
| `cdh1_contamination_fix` | 4 | 800 | density | 0.10 | 3,200 | V2 anchor 邻域污染，改用 density 修复 |
| `KO_family` | 3 | 500 | boundary | 0.10 | 1,500 | V2 未采样 KO（Tem1T=0, Cdc15T=0） |
| `contamination_fix` | 9 | 800 | density | 0.08 | 7,200 | V2 boundary 邻域污染，小扰动修复 |
| **合计** | **27** | — | — | — | **17,400** | — |

**设计动机**:
- V2 在三大家族上取得 -47% Real MAE 提升，但残留 26 个瓶颈突变体
- 假设性分析：若这 26 个突变体 MAE 都降到 0.1，Real MAE 会从 0.1926 降到 0.0999（-48%），接近 LHS 测试集水平
- V3 针对 5 类瓶颈分别设计修复策略

**采样策略选择原则**（基于 V2 教训的精细化设计）:
- **未采样家族**（CLB2_db / cln_triple）→ density 模式 + 大扰动 0.15（避免 boundary 邻域污染，同时保证足够的采样宽度）
- **已采样但邻域污染**（cdh1_contamination_fix / contamination_fix）→ density 模式 + 翻倍采样数（500→800）+ 小扰动（0.08-0.10，精准修复）
- **KO 突变体**（KO_family）→ boundary 模式（覆盖到 0 边界，因 Tem1T=0/Cdc15T=0 是 KO 突变体的核心特征，必须覆盖）
- **CDH1T/CDH1** → 不用 anchor 模式（避免邻域污染），改用 density 模式 + 从 cdh1_pairs 随机选

**V3 数据集分析结果**（基于 `analysis_output_v3/`，对比 V2）:

| 指标 | v2 | v3 | 变化 |
|------|----|----|------|
| 总样本数 | 120,017 | 137,406 | +17,389 (1.14×) |
| 文件大小 | 14,345 MB | 16,418 MB | 1.14× |
| LHS CDH1T=0 样本数 | 5,252 (4.4%) | 6,826 (5.0%) | +1,574 |
| CDH1T LHS 均值 | 1.055 | 1.029 | 更接近 Real=0.814 |
| CDH1T 异常漂移率 | 19.1% | 17.9% | -1.2pp |
| CDC14T 异常漂移率 | 5.6% | 4.9% | 已低于 Real 5.6% |
| NET1T 异常漂移率 | 1.7% | 1.5% | 已与 Real 1.6% 对齐 |
| TEM1 LHS Std | 2.353 | 3.692 | +57%（更接近 Real 6.863） |
| CLN2 LHS 均值漂移 | 1.874 | 1.725 | 缓解 (-0.149) |
| SIC1 LHS 均值漂移 | 1.843 | 1.644 | 缓解 (-0.199) |
| MASS Z-score 偏度 | 1.67 | 1.61 | 持续改善 |
| 浓度爆炸样本 | 0 | 0 | 安全 |
| 最高浓度极值 | 906.81 | 908.92 | +0.23%（可控） |
| Pattern C 占比 | 36.3% | 37.9% | +1.6pp（KO 引入 arrest，预期） |

**关键发现**:
- **V2 邻域污染成功修复**: CLN2/SIC1 均值漂移显著缓解，CDC14T/NET1T 异常漂移率已低于或对齐 Real
- **KO 家族效果显著**: TEM1 标准差提升 57%，覆盖了 V2 完全缺失的 Tem1T=0/Cdc15T=0 边界
- **CDH1T=0 覆盖持续增加**: 即使改用 density 模式，cdh1_contamination_fix 仍贡献了 +1,574 个 CDH1T=0 样本（因 density 扰动触及边界）
- **副作用可控**: Pattern C 仅上升 1.6pp（远低于 V1→V2 的 +3.4pp），density 小扰动策略生效

**预期效果**:
- V3+none 预期 Real MAE ≈ 0.10-0.12（残留瓶颈修复）
- V3+mutant_aware AL 预期 Real MAE ≈ 0.08-0.10（接近 LHS 测试集水平 0.05-0.08）

**输出**: `lhs_cheat_origin_v3_210min_500steps_dual_labels.npz`

**与 V2 的兼容性**:
- V3 完整保留 V2 的 `DEFAULT_TARGET_FAMILIES` 与 `generate_targeted_lhs_tasks` 函数
- `V3_EXPANSION_FAMILIES` 与 `DEFAULT_TARGET_FAMILIES` 无重叠突变体，键名不会冲突
- 主程序在 V2 之后再追加 V3 扩展任务

**V3 Eval 结果**（详见 `eval_result/V3_EVAL_ANALYSIS_REPORT.md` 与 `eval_result/V4_OPTIMIZATION_PLAN.md`）:
- Real MAE: 0.1926 → 0.1013（-47%）
- Real MSE: 4.43 → 0.442（-90%）
- Sim-to-Real Gap: 4.63× → 1.89×
- Tier 4 (MAE>0.4): 26 → 5
- **残留瓶颈**: 25 个 MAE > 0.15，三类问题：
  - 邻域污染（5 个）：KO_family boundary 模式污染 tem1_ts（90_tem1_net1_ts +177%）
  - 覆盖盲区（18 个）：V3 仅覆盖 54/126（42.9%），cln1_/cdc14_ts/swi5_cdh1 等未覆盖
  - 采样强度不足（2 个）：CLB1_clb2 仅覆盖 2 个、CLB2_db pert/num 偏小

---

### V4（`lhs_v1_origin_v4.py`）— 邻域污染修复 + 覆盖盲区扩展

**核心改进**（相对于 V3）:
1. **修复 KO_family 邻域污染**：boundary 模式 → density 模式，去除 `boundary_params`（不再将 Tem1T/Cdc15T 扩展到 0），pert 0.10→0.05
2. **新增 tem1_ts_family**：专覆盖 88/89/90/91 四个 tem1_ts 突变体（V3 被污染且未覆盖），density + pert=0.08
3. **新增 cdc14_ts_family**：覆盖全部 6 个 cdc14_ts 相关突变体（99/103/104/105/106/107），density + pert=0.08
4. **新增 swi5_cdh1_family**：覆盖 43/44/45/46（V3 全部未覆盖），density + pert=0.12
5. **新增 cln1_cln2_KO_family**：覆盖 9 个 cln1_cln2 双 KO 突变体（V3 全部未覆盖），density + pert=0.12
6. **新增 sic1_KO_family**：覆盖 27/28/29（V3 全部未覆盖），density + pert=0.10
7. **新增 GAL_CLB2_cdh1_family**：覆盖 33/51/54（V3 全部未覆盖），density + pert=0.10
8. **扩展 cln_triple**：4 → 11 个突变体（覆盖全部 cln1_cln2_cln3 前缀突变体）
9. **扩展 CLB1_clb2**：2 → 4 个突变体，pert 0.2→0.3（覆盖 48/51/52/111）
10. **增强 CLB2_db**：7 → 10 个突变体（新增 57/58/59），pert 0.15→0.20，num 500→800
11. **保留 cdh1_contamination_fix / contamination_fix**：V3 已验证有效，配置不变

**家族配置**（`V4_EXPANSION_FAMILIES`）:

| 家族 | 类别 | 突变体数 | num | mode | pert | 样本数 | 设计理由 |
|------|------|---------|-----|------|------|--------|---------|
| `KO_family_fixed` | 修复 | 3 | 500 | density | 0.05 | 1,500 | 修复 V3 boundary 污染 tem1_ts |
| `tem1_ts_family` | 新增 | 4 | 800 | density | 0.08 | 3,200 | 覆盖 88/89/90/91（V3 被污染） |
| `cdc14_ts_family` | 新增 | 6 | 800 | density | 0.08 | 4,800 | 覆盖全部 cdc14_ts（V3 未覆盖） |
| `swi5_cdh1_family` | 新增 | 4 | 500 | density | 0.12 | 2,000 | 覆盖 43/44/45/46（V3 未覆盖） |
| `cln1_cln2_KO_family` | 新增 | 9 | 400 | density | 0.12 | 3,600 | 覆盖 9 个 cln1_cln2 KO（V3 未覆盖） |
| `sic1_KO_family` | 新增 | 3 | 400 | density | 0.10 | 1,200 | 覆盖 27/28/29（V3 未覆盖） |
| `GAL_CLB2_cdh1_family` | 新增 | 3 | 500 | density | 0.10 | 1,500 | 覆盖 33/51/54（V3 未覆盖） |
| `cln_triple_extended` | 扩展 | 11 | 500 | density | 0.15 | 5,500 | 全部 cln1_cln2_cln3 突变体 |
| `CLB1_clb2_extended` | 扩展 | 4 | 500 | boundary | 0.30 | 2,000 | V2 的 2→4，pert 0.2→0.3 |
| `CLB2_db_enhanced` | 增强 | 10 | 800 | density | 0.20 | 8,000 | V3 的 7→10，pert/num 提升 |
| `cdh1_contamination_fix` | 保留 | 4 | 800 | density | 0.10 | 3,200 | V3 配置不变 |
| `contamination_fix` | 保留 | 9 | 800 | density | 0.08 | 7,200 | V3 配置不变 |
| **合计** | — | **67** | — | — | — | **43,700** | — |

**设计动机**:
- V3+none Real MAE=0.1013（-47%），但仍残留 25 个 MAE > 0.15 瓶颈
- 逐突变体三方对比（V1/V2/V3）暴露三类系统性问题：邻域污染 + 覆盖盲区 + 采样强度不足
- V4 通过 12 类家族（67 突变体，~44K 定向样本）系统性修复全部三类问题

**采样策略选择原则**（基于 V3 教训的精细化设计）:
- **KO 突变体**（KO_family_fixed）→ density 模式 + 极小扰动（0.05），**不再用 boundary**（V3 教训：boundary 扩展到 0 会污染 tem1_ts）
- **ts 突变体**（tem1_ts/cdc14_ts）→ density 模式 + 小扰动（0.08），专覆盖温度敏感突变的局部参数空间
- **未采样家族**（swi5_cdh1/cln1_cln2_KO/sic1_KO/GAL_CLB2_cdh1）→ density 模式 + 中等扰动（0.10-0.12）
- **cln_triple 扩展** → density 模式 + pert=0.15（与 V3 一致），覆盖全部 11 个 cln1_cln2_cln3 突变体
- **CLB1_clb2 扩展** → boundary 模式（ksb2p/ksb2pp=0 是真值）+ pert=0.30（增大覆盖范围）
- **CLB2_db 增强** → density 模式 + pert=0.20 + num=800（提升采样强度）
- **CDH1T/CDH1** → 不用 anchor 模式（避免邻域污染），density 模式从 cdh1_pairs 随机选

**与 V3 的关键差异**:
1. **KO_family 模式变更**：boundary（Tem1T/Cdc15T→0）→ density（pert=0.05），彻底切断 tem1_ts 污染源
2. **V4 完全替代 V3**：`__main__` 不再调用 `V3_EXPANSION_FAMILIES`，改调用 `V4_EXPANSION_FAMILIES`
3. **V4 包含 V3 所有家族的修复/增强/保留版**：12 个家族 = 1 修复 + 6 新增 + 2 扩展 + 1 增强 + 2 保留
4. **V2 的 DEFAULT_TARGET_FAMILIES 保留不变**：cdc20_ts/cdh1/low_lhs_count 仍有效；CLB1_clb2 被 V4 覆盖（键名相同，V4 pert 更大）

**预期效果**（基于 `eval_result/V4_OPTIMIZATION_PLAN.md` 估算）:
- Real MAE: 0.1013 → 0.080–0.090（-11~21%）
- Real MSE: 0.442 → 0.25–0.32（-28~43%）
- Sim-to-Real Gap: 1.89× → 1.45–1.55×
- Tier 4 (MAE>0.4): 5 → 0–1
- MAE > 0.15 突变体: 25 → 8–10
- 覆盖率: 42.9% → 75%+

**输出**: `lhs_cheat_origin_v4_210min_500steps_dual_labels.npz`

**与 V3 的兼容性**:
- V4 完整保留 V3 的 `V3_EXPANSION_FAMILIES` 字典定义（作为历史参考），但 `__main__` 不再调用它
- V4 完整保留 V2 的 `DEFAULT_TARGET_FAMILIES` 与 `generate_targeted_lhs_tasks` 函数
- V4 的 `V4_EXPANSION_FAMILIES` 与 V2 的 `DEFAULT_TARGET_FAMILIES` 有 9 个重叠突变体（CLB1_clb2 家族），V4 后执行会覆盖 V2（pert 更大）

---

### V5（`lhs_v1_origin_v5.py`）— 7 项定向修复（基于 V4 eval 残留瓶颈）

**核心改进**（相对于 V4）:
1. **修复 CLB1_clb2_extended**：boundary 模式 → density 模式，pert 0.30→0.50，num 500→1000（V4 boundary 模式导致 CLB1_clb2_extended 家族预测失败）
2. **新增 cdc6_family**：覆盖 5 个 cdc6 相关突变体（38/39/40/41/42），density + pert=0.10，600 样本/突变体
3. **增强 tem1_ts_family**：num 800→1200，pert 0.08→0.12（V4 中 tem1_ts MAE 仍偏高，提升采样强度）
4. **新增 CLB2_db_clb5_Raff_family**：覆盖 61.2（V4 未采样），density，400 样本
5. **新增 APC_A_GAL_CLB2_family**：覆盖 124（V4 未采样），density，400 样本
6. **新增 cdc15_net1_ts_family**：覆盖 95/109（V4 未采样），density，400 样本/突变体
7. **新增 sic1_cdh1_family**：覆盖 36/37（V4 未采样），density，400 样本/突变体

**家族配置**（`V5_EXPANSION_FAMILIES`，17 类家族）:

| 家族 | 类别 | 突变体数 | num | mode | pert | 理论样本数 | 设计理由 |
|------|------|---------|-----|------|------|----------|---------|
| `CLB1_clb2_fixed` | 修复 | 4 | 1000 | density | 0.50 | 4,000 | boundary→density 修复 V4 失败 |
| `cdc6_family` | 新增 | 5 | 600 | density | 0.10 | 3,000 | 覆盖 cdc6_42_49 全家族（V4 未覆盖） |
| `tem1_ts_enhanced` | 增强 | 4 | 1200 | density | 0.12 | 4,800 | V4 tem1_ts MAE 偏高，提升采样强度 |
| `CLB2_db_clb5_Raff_family` | 新增 | 1 | 400 | density | 0.10 | 400 | 覆盖 61.2（V4 未覆盖） |
| `APC_A_GAL_CLB2_family` | 新增 | 1 | 400 | density | 0.10 | 400 | 覆盖 124（V4 未覆盖） |
| `cdc15_net1_ts_family` | 新增 | 2 | 400 | density | 0.10 | 800 | 覆盖 95/109（V4 未覆盖） |
| `sic1_cdh1_family` | 新增 | 2 | 400 | density | 0.10 | 800 | 覆盖 36/37（V4 未覆盖） |
| *（V4 保留的 10 类家族）* | 保留 | 67 | — | — | — | ~39,700 | KO_fixed/tem1_ts(原)/cdc14_ts/swi5_cdh1/cln1_cln2_KO/sic1_KO/GAL_CLB2_cdh1/cln_triple_ext/CLB2_db_enh/cdh1_contam_fix/contam_fix |
| **合计** | — | **88** | — | — | — | **~54,000** | — |

**设计动机**:
- V4+none eval Real MAE=0.0598（-41% vs V3），但仍存在 CLB1_clb2_extended 失败（boundary 模式不匹配）与 38_cdc6_42_49 退化（+56%）
- V5 通过 7 项定向修复，针对性解决 V4 的两类残留问题

**V5 数据集分析结果**（基于 `analysis_output_v5/`，对比 V4）:

| 指标 | v4 | v5 | 变化 |
|------|----|----|------|
| 总样本数 | 157,599 | 162,134 | +4,535 (1.03×) |
| 文件大小 | 18,829 MB | 19,379 MB | +550 MB |
| 定向样本 (_LHS_T) | 37,582 | 42,117 | +4,535 |
| Viable 占比 | 47.4% | 48.7% | +1.3pp |
| Pattern C (Arrest) | 40.0% | 38.5% | -1.5pp |
| CLB2 mean drift (P0 fix) | — | — | -16.8% |
| TEM1 mean drift (P2 enhance) | — | — | -10.0% |

**关键发现**:
- CLB1_clb2_extended 修复生效：boundary→density 模式使 Viable 回升 +1.3pp，Pattern C 下降 1.5pp
- cdc6_family 新增覆盖生效：cdc6 突变体 Pattern B 增加，表型分布更均衡
- tem1_ts 增强生效：TEM1 mean drift 改善 10.0%

**输出**: `lhs_cheat_origin_v5_210min_500steps_dual_labels.npz`

**与 V4 的兼容性**:
- V5 完整保留 V4 的 `V4_EXPANSION_FAMILIES` 字典定义（作为历史参考），但 `__main__` 不再调用它
- V5 完整保留 V2 的 `DEFAULT_TARGET_FAMILIES` 与 `generate_targeted_lhs_tasks` 函数
- V5 的 `V5_EXPANSION_FAMILIES` 替代 V4 的全部定向扩展（含修复/新增/增强/保留）

---

## 五、全局配置参数

```python
SIM_TF = 210            # 模拟总时长 (min)，可通过环境变量 SIM_TF 覆盖
SIM_STEPS = 500          # 采样总步数，可通过环境变量 SIM_STEPS 覆盖
NUM_LHS_SAMPLES = 1000  # 每个非-WT 突变体的全局 LHS 采样数量
SIM_TIMEOUT = 5.0       # 单个细胞求解超时阈值 (秒)
```

**离散保护名单**（`DISCRETE_STRICT_PARAMS`）:
```python
['init_CDH1T', 'init_CDH1', 'mdt', 'BUB2_active', 'init_BUB2', 'is_mutant_104', 'ESP1T']
```

**输出路径**:
- 目录: `/home/users/zsy/trae_workspace/mydata/budding_yeast/`
- 文件名: `lhs_cheat_origin_{ver}_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`

---

## 六、运行方式

```bash
cd /home/users/zsy/trae_workspace/budding_yeast_v2/data

# 生成 V2 数据集
PYTHONPATH=/home/users/zsy/trae_workspace/mydata/budding_yeast:$PYTHONPATH \
    /home/users/zsy/.conda/envs/neuralop/bin/python lhs_v1_origin_v2.py

# 生成 V3 数据集
PYTHONPATH=/home/users/zsy/trae_workspace/mydata/budding_yeast:$PYTHONPATH \
    /home/users/zsy/.conda/envs/neuralop/bin/python lhs_v1_origin_v3.py

# 生成 V4 数据集
PYTHONPATH=/home/users/zsy/trae_workspace/mydata/budding_yeast:$PYTHONPATH \
    /home/users/zsy/.conda/envs/neuralop/bin/python lhs_v1_origin_v4.py

# 生成 V5 数据集
PYTHONPATH=/home/users/zsy/trae_workspace/mydata/budding_yeast:$PYTHONPATH \
    /home/users/zsy/.conda/envs/neuralop/bin/python lhs_v1_origin_v5.py
```

**耗时**: V2 约 2-3 小时（~125,000 ODE 求解），V3 约 3-4 小时（~133,000 ODE 求解），V4 约 4-5 小时（~165,000 ODE 求解），V5 约 4-5 小时（~170,000 ODE 求解）

---

## 七、更新日志

> **维护规则**: 每次对数据生成代码做改进（新增家族、调整采样参数、修改采样逻辑等），必须在此追加一条记录，格式：
> ```
> ### [YYYY-MM-DD] vX.Y - 简述
> **文件**: `lhs_v1_origin vX.py`
> **改动**: ...
> **原因**: ...
> **效果**: ...
> ```

### [2026-07-06] V5 - 7 项定向修复（基于 V4 eval 残留瓶颈）
**文件**: `lhs_v1_origin_v5.py`（新建）
**改动**:
- 复制自 `lhs_v1_origin_v4.py`
- 修改 `DATASET_NAME` 加 `_v5` tag
- 新增 `V5_EXPANSION_FAMILIES` 字典（17 类家族，88 突变体，~54K 定向样本）
  - 修复类（1 个）：CLB1_clb2_fixed（boundary→density，pert 0.30→0.50，num 500→1000）
  - 新增类（5 个）：cdc6_family / CLB2_db_clb5_Raff_family / APC_A_GAL_CLB2_family / cdc15_net1_ts_family / sic1_cdh1_family
  - 增强类（1 个）：tem1_ts_enhanced（num 800→1200，pert 0.08→0.12）
- 修改 `__main__` 块：不再调用 `V4_EXPANSION_FAMILIES`，改调用 `V5_EXPANSION_FAMILIES`

**原因**: V4+none eval Real MAE=0.0598（-41% vs V3），但残留两类问题
1. CLB1_clb2_extended 预测失败（boundary 模式不匹配，38_cdc6_42_49 退化 +56%）
2. 覆盖盲区（cdc6/CLB2_db_clb5_Raff/APC_A_GAL_CLB2/cdc15_net1_ts/sic1_cdh1 未采样）

**效果**（基于 `analysis_output_v5/` 数据集分析）:
- 总样本: 157,599 → 162,134（+4,535，1.03×）
- 定向样本: 37,582 → 42,117（+4,535）
- Viable 占比: 47.4% → 48.7%（+1.3pp，CLB1_clb2 修复生效）
- Pattern C 占比: 40.0% → 38.5%（-1.5pp）
- CLB2 mean drift: -16.8%（P0 修复）
- TEM1 mean drift: -10.0%（P2 增强）
- 训练效果（V5+none / V5+mutant_aware）待 run_pipeline.sh 执行后评估

### [2026-07-06] 下游分析 regex bug 修复
**文件**: `mydata/budding_yeast/dataset_analysis/real_vs_lhs_csv.py`、`visualization/plot_real_vs_lhs.py`、`visualization/plot_all_cheat_samples.py`
**改动**:
- 3 个脚本共 4 处正则 `^[_LHS0-9\-]+$` → `^[_LHST0-9\-]+$`（字符类新增 `T`）
  - `real_vs_lhs_csv.py` 第 113 行
  - `plot_real_vs_lhs.py` 第 151、255 行
  - `plot_all_cheat_samples.py` 第 124 行
**原因**: 定向样本命名为 `{mutant}_LHS_T{idx}`（如 `73_cdc20_ts_LHS_T007`），替换 real_name 后 remainder 为 `_LHS_T007`，其中 `T` 不在原字符类中，导致正则不匹配。V5 中 54,619 个定向样本（33.7%）全部被排除出 `Sensitivity_Atlas.csv`，进而使 `correlation_analysis.py`、`plot_real_vs_lhs.py` 的 LHS 敏感度统计缺失定向样本。
**效果**:
- V5 Sensitivity_Atlas.csv 从 107,389 行 → 162,008 行（+54,619 定向样本纳入）
- correlation_analysis_v5.csv 重新生成，相关性结构剧变：Top 1 从 lhs_MSE_min (Spearman=+0.8488) 变为 lhs_MAE_std (Spearman=+0.6389)
- plot_real_vs_lhs.py 重跑完成（126 PNG + Plotted_Summary.csv 含 88 个 _LHS_T 定向样本，初版仅含常规 LHS）
- DATASET_ANALYSIS_REPORT_V5.md 更新（第 10/12/13 节）
- 影响 V2~V5 所有含定向样本的数据集分析（V2 起 `_LHS_T` 后缀即存在）

---

### [2026-07-05] V4 - 邻域污染修复 + 覆盖盲区扩展
**文件**: `lhs_v1_origin_v4.py`（新建）
**改动**:
- 复制自 `lhs_v1_origin_v3.py`
- 修改 `DATASET_NAME` 加 `_v4` tag
- 新增 `V4_EXPANSION_FAMILIES` 字典（12 类家族，67 突变体，~44K 定向样本）
  - 修复类（1 个）：KO_family_fixed（boundary→density，去除 boundary_params）
  - 新增类（6 个）：tem1_ts/cdc14_ts/swi5_cdh1/cln1_cln2_KO/sic1_KO/GAL_CLB2_cdh1
  - 扩展类（2 个）：cln_triple_extended（4→11 突变体）、CLB1_clb2_extended（2→4 突变体，pert 0.2→0.3）
  - 增强类（1 个）：CLB2_db_enhanced（7→10 突变体，pert 0.15→0.20，num 500→800）
  - 保留类（2 个）：cdh1_contamination_fix、contamination_fix（V3 配置不变）
- 修改 `__main__` 块：不再调用 `V3_EXPANSION_FAMILIES`，改调用 `V4_EXPANSION_FAMILIES`
- 修复突变体名错误：`106_cdc14_ts` → `106_cdc14_ts_cdh1`，删除不存在的 `108_cdc14_ts_multi_TEM1`
- cdc14_ts_family 扩展为覆盖全部 6 个 cdc14_ts 相关突变体（99/103/104/105/106/107）

**原因**: V3+none eval 逐突变体分析暴露三类系统性问题
1. 邻域污染（5 个恶化）：V3 KO_family boundary 模式污染 tem1_ts（90_tem1_net1_ts +177%）
2. 覆盖盲区（18 个未覆盖且 MAE>0.15）：V3 仅覆盖 54/126（42.9%），cln1_/cdc14_ts/swi5_cdh1 等未覆盖
3. 采样强度不足：CLB1_clb2 仅覆盖 2 个、CLB2_db pert/num 偏小

**预期效果**（详见 `eval_result/V4_OPTIMIZATION_PLAN.md`）:
- Real MAE: 0.1013 → 0.080–0.090（-11~21%）
- Real MSE: 0.442 → 0.25–0.32（-28~43%）
- Sim-to-Real Gap: 1.89× → 1.45–1.55×
- Tier 4 (MAE>0.4): 5 → 0–1
- MAE > 0.15 突变体: 25 → 8–10
- 覆盖率: 42.9% → 75%+
- 训练效果（V4+none / V4+mutant_aware）待 run_pipeline.sh 执行后评估

---

### [2026-07-05] V3 - 残留瓶颈定向扩展
**文件**: `lhs_v1_origin_v3.py`（新建）
**改动**:
- 复制自 `lhs_v1_origin_v2.py`
- 修改 `DATASET_NAME` 加 `_v3` tag
- 新增 `V3_EXPANSION_FAMILIES` 字典（5 类家族，27 个残留瓶颈突变体）
- 修改 `__main__` 块在 V2 扩展后追加 V3 扩展调用

**原因**: V2 残留 26 个 MAE > 0.3 突变体，需定向修复以接近 LHS 测试集水平
**效果**（基于 `analysis_output_v3/` 数据集分析）:
- 总样本: 120,017 → 137,406（+17,389，1.14×）
- CDH1T=0 样本: 5,252 → 6,826（+1,574）
- CDH1T LHS 均值: 1.055 → 1.029（更接近 Real=0.814）
- CDC14T 异常漂移率: 5.6% → 4.9%（已低于 Real 5.6%）
- NET1T 异常漂移率: 1.7% → 1.5%（已与 Real 1.6% 对齐）
- TEM1 LHS Std: 2.353 → 3.692（+57%，更接近 Real 6.863）
- CLN2/SIC1 均值漂移显著缓解（V2 boundary 邻域污染修复）
- 浓度爆炸样本: 0（安全）
- Pattern C 占比: 36.3% → 37.9%（+1.6pp，KO 引入 arrest，预期）
- MASS Z-score 偏度: 1.67 → 1.61（持续改善）
- **V2 邻域污染成功修复，KO 家族覆盖生效，副作用可控**
- 训练效果（V3+none / V3+mutant_aware）待 run_pipeline.sh 执行后评估

---

### [2026-07-05] V2 - 三大家族定向扩展
**文件**: `lhs_v1_origin_v2.py`（新建）
**改动**:
- 复制自 `lhs_v1_origin.py`
- 修改 `DATASET_NAME` 加 `_v2` tag
- 新增 `DEFAULT_TARGET_FAMILIES` 字典（4 类家族：cdc20_ts/CLB1_clb2/cdh1/low_lhs_count）
- 新增 `generate_targeted_lhs_tasks` 函数（支持 density/boundary/anchor 三种模式）
- 修改 `__main__` 块在 LHS 采样后追加定向扩展调用

**原因**: V1 全局 LHS 在边界覆盖不足，导致 Sim-to-Real Gap
**效果**:
- Real MAE: 0.3645 → 0.1926（-47%）
- cdc20_ts 家族 MAE: -97%
- Pattern C MAE: -59%
- Sim-to-Real Gap: -34%
- 已知遗留: 26 个 MAE > 0.3 残留瓶颈（由 V3 修复）

---

### [2026-07-05] V1 - 基础 LHS 采样引擎
**文件**: `lhs_v1_origin.py`（V0 → V1 重构）
**改动**:
- 引入 `DISCRETE_STRICT_PARAMS` 严格离散保护名单
- 实现 `CDH1T/CDH1` 强绑定元组机制
- `BUB2_active/init_BUB2` 同频连续缩放
- 等价碰撞拦截器（`np.isclose`）
- 纯离散突变体智能截断
- 双重打标系统（bio_label + pattern_label）
- 动态浓度异常剔除系统

**原因**: V0 采样逻辑存在离散参数泄漏、CDH1 比例约束违反、克隆冗余等问题
**效果**: 作为后续版本的基线，Real MAE = 0.3645

---

## 八、下游依赖

| 下游模块 | 依赖文件 | 依赖内容 |
|---------|---------|----------|
| `utils/al_callback.py` | `lhs_v1_origin.py` | `simulate_mutant`, `get_default_params`, `generate_lhs_tasks`, `generate_local_lhs_tasks`, `mutant_rules`, `eqns` |
| `train.py` | npz 数据集 | `data`, `params`, `mutant_names`, `labels`, `pattern_labels` |
| `eval/` 评估脚本 | npz 数据集 | 同上 |
| `run_pipeline.sh` | npz 数据集 | `DATASET_V2`, `DATASET_V3`, `DATASET_V4`, `DATASET_V5` 变量 |
| `mydata/budding_yeast/dataset_analysis/` | npz 数据集 + `mutant_names` | `real_vs_lhs_csv.py` 依赖 `_LHS_`/`_LHS_T` 子串匹配区分 Real 与 LHS 样本（regex 已修复）；`correlation_analysis.py` 依赖 `Sensitivity_Atlas.csv` |
| `mydata/budding_yeast/run_analysis_v5.py` | npz 数据集 | 14 个分析脚本流水线（含 `real_vs_lhs_csv` → `correlation_analysis` 依赖链） |

**重要约束**（来自 `project_memory.md`）:
- `lhs_v1_origin.py` 的 `generate_local_lhs_tasks` 函数（RGS 专属算法）必须在同步到 `mydata/budding_yeast/` 时保留
- v2 数据集生成代码必须包含 `os.makedirs(OUTPUT_DIR, exist_ok=True)`
- v2 数据集必须使用 `np.savez_compressed` 保存完整路径
- 生成后必须打印完整输出路径

---

## 九、设计原则与经验教训

### 9.1 采样模式选择

- **density 模式**：局部邻域扰动，**最安全**，不会引入新的边界样本污染
- **boundary 模式**：扩展到 0 边界，**适用于彻底敲除突变体**（如 KO），但可能引入邻域污染
- **anchor 模式**：CDH1T/CDH1 强制锚点，**适用于已知强绑定约束**，但邻域污染风险高

### 9.2 邻域污染问题

**V2 教训**: boundary 采样（ks20p=0, ksb2pp=0）虽然解决了 cdc20_ts/CLB1_clb2 家族，但引入了 7 个新的 V1→V2 恶化突变体（如 `76_cdc20_pds1_clb5` MAE 从 0.36 → 1.39）。

**V3 修复策略**: 对 V2 邻域污染突变体用 density 模式 + 小扰动（0.08）+ 翻倍采样数（800）精准修复，避免引入新污染。

### 9.3 Sim-to-Real Gap 缓解策略

1. **数据层**（V2/V3）：定向扩展覆盖 Real Mutant 边界
2. **算法层**（mutant_aware AL）：基于 atlas 优先级加权采样
3. **联合方案**（V3 + mutant_aware AL）：数据层补盲 + 算法层定向

### 9.4 采样规模权衡

- 单突变体 400 样本：基础覆盖，适合未采样家族的初步扩展（V4 cln1_cln2_KO/sic1_KO）
- 单突变体 500 样本：基础扩展，适合一般残留瓶颈（V2/V3/V4 默认值）
- 单突变体 800 样本：精准修复，适合邻域污染修复与瓶颈突变体（V3/V4 contamination_fix/cdc14_ts/tem1_ts）
- 单突变体 1000 样本（NUM_LHS_SAMPLES）：全局基线覆盖
- 总规模上限：~165,000（V4，含 124 Real + 121K LHS + 44K 定向），单次生成耗时 4-5 小时

---

## 十、待办与未来改进方向

- [x] 生成 V3 数据集并完成综合分析（`analysis_output_v3/`）
- [x] 运行 V3+none / V3+mutant_aware 实验并完成 eval 分析（`V3_EVAL_ANALYSIS_REPORT.md`）
- [x] V3 逐突变体完整分析与 V4 优化方案（`V4_OPTIMIZATION_PLAN.md`）
- [x] 生成 V4 数据集生成代码（`lhs_v1_origin_v4.py`）
- [x] 生成 V5 数据集生成代码（`lhs_v1_origin_v5.py`）并完成综合分析（`analysis_output_v5/`）
- [x] 运行 V4+none / V4+mutant_aware 实验（`run_pipeline.sh` 配置已就绪，V4 eval 完成）
- [x] 修复 `real_vs_lhs_csv.py` 正则 bug（`[_LHS0-9\-]` → `[_LHST0-9\-]`，V5 定向样本 54,619 个已纳入 Sensitivity_Atlas.csv）
- [ ] 运行 V5+none / V5+mutant_aware 实验（待 run_pipeline.sh 执行）
- [ ] 评估 V5 训练效果，验证 7 项定向修复效果
- [ ] 评估是否需要修复 `metrics.py` 的 Rel_L2 数值爆炸问题（CDH1T=0 导致真值方差接近 0）
- [ ] 考虑引入 V6：基于 V5 eval 残留瓶颈的进一步定向扩展（如果 V5 仍有 MAE > 0.15 突变体）

---

**文档结束** | 最后更新: 2026-07-06 | 维护人: zsy
