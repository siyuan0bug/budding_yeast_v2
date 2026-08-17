# LHS 数据集生成引擎说明文档

> **位置**: `/home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/`
> **维护规则**: 每次对数据生成代码做改进，必须同步更新本文档对应版本章节与"更新日志"
> **当前版本**: V6（基于 V5 eval 残留瓶颈的 P0-P5 修复）
> **维护人**: zsy
> **去重算法统一**: 2026-07-14 起，除 V0 外所有脚本均已升级为 rtol=0.08 + 全局跨检查 + 0 值修复的完整防泄漏算法

---

## 一、文件清单

| 文件 | 版本 | 状态 | 去重算法 | 说明 |
|------|------|------|----------|------|
| `lhs_v0.py` | V0 | 历史参考 | 旧 (rtol=1e-5) | 全局盲抽 LHS 引擎，80K 样本，14 个离散保护参数 |
| `lhs_v1.py` | V1-cheat | 历史参考 (有 bug) | 新 (rtol=0.08) ⚠️ | 家族式 LHS，10K 样本/突变体，14 个离散保护参数。**注意**: `generate_lhs_tasks` 中引用了 `real_mutant_full` 但从未调用 `build_real_mutant_full()` 构建，直接运行会触发 `NameError`。仅 `get_default_params` 等工具函数可被外部安全导入 |
| `lhs_v1_origin.py` | V1 | 当前基线 | 新 (rtol=0.08) | 基础 LHS 采样引擎，124 突变体 + 1000 LHS/突变体 |
| `lhs_v1_origin_v2.py` | V2 | 已发布 | 新 (rtol=0.08) | V1 + 三大家族定向扩展（cdc20_ts/CLB1_clb2/cdh1）+ N_LHS 补足 |
| `lhs_v1_origin_v3.py` | V3 | 已生成+已分析 | 新 (rtol=0.08) | V2 + 残留瓶颈定向扩展（27 个 MAE>0.3 突变体，5 类家族） |
| `lhs_v1_origin_v4.py` | V4 | 已生成 | 新 (rtol=0.08) | V3 + 邻域污染修复 + 覆盖盲区扩展（12 类家族，67 突变体，~44K 定向样本） |
| `lhs_v1_origin_v5.py` | V5 | 已生成+已分析 | 新 (rtol=0.08) | V4 + 7 项定向修复（17 类家族，88 突变体，~54K 定向样本） |
| `lhs_v1_origin_v6.py` | V6 | 待生成 | 新 (rtol=0.08) | V5 + P0-P5 定向修复（修复 tem1_ts, CLB2_db_Gal 盲区等） |
| `lhs_v1_origin_guaranteed.py` | 实验 | 对照实验 | 新 (rtol=0.08) | 保证产量版，Phase A-G 补采机制，rtol=0.08 + 全局跨检查 |
| `lhs_v1_origin_no_filter.py` | 实验 | 对照实验 | 新 (rtol=0.08) | 无过滤版，移除浓度修剪与智能截断，保留完整防泄漏去重 |
| `test_lhs_filter_analysis.py` | 工具 | 诊断工具 | 旧 (基于 V1_origin 旧逻辑) | LHS 采样过滤阶段损失分析，追踪 3 阶段样本损失链路。**注意**: 代码仍使用旧的 rtol=1e-5 + candidate_count=target×10，未跟进新算法 |
| `run_lhs_batch.sh` | 工具 | 批量运行 | — | 批量运行 v2-v6 数据生成脚本，含 BLAS 死锁修复 |
| `DATASET_GENERATION_GUIDE.md` | - | 本文件 | — | 数据生成引擎说明与版本变更记录 |

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
# V5 新增（替代 V4，V6 中被替代）：
# all_tasks.update(generate_targeted_lhs_tasks(V5_EXPANSION_FAMILIES))
# V6 新增（替代 V5）：
all_tasks.update(generate_targeted_lhs_tasks(V6_EXPANSION_FAMILIES))  # ④ P0-P5 定向修复
```

**键名约定**（用于下游 `_LHS_` 子串匹配）:
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
- **超时保护**: `SIM_TIMEOUT = 5.0s`，通过 SIGALRM 硬超时兜底（详见 11.4.2 节）。原 `eqns_with_timeout` 的 Python 层 `time.time()` 检查在 BDF 卡于 Fortran 层时无效，SIGALRM 信号能中断系统调用，确保单个病态样本不会拖死整个进程池
- **并行**:
  - V0: `ProcessPoolExecutor(max_workers=os.cpu_count())`（使用全部核心）
  - V1 及之后: `ProcessPoolExecutor(max_workers=cpu_count//4)`（绅士模式，留 3/4 核给 GPU）
- **BLAS 死锁修复**（`run_lhs_batch.sh` 中设置）:
  ```bash
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  ```
  `ProcessPoolExecutor` 用 `fork` 创建子进程时，OpenBLAS 线程池的锁状态不一致会导致死锁。设为 1 后每个工作进程单线程做线性代数，16 进程 × 1 线程 = 16 核，匹配 `safe_cores`。
- **Mass 限制**: `SIM_TF <= 210` 且 `max_mass > 10.0` 时丢弃样本

### 阶段 3: 双重打标系统

每个成功求解的样本生成两个标签：

**生物学机理标签**（`bio_label`，由 `analyze_cell_viability_unified` 生成）:
- `Viable`
- `Inviable: G1 Arrest`
- `Inviable: M-phase Arrest`
- `Inviable: Unbudded Division`
- `Inviable: Premature Division`
- `Inviable: Sequence Failure`

**视觉形态学标签**（`pattern_label`，由 `analyze_mass_pattern` 生成）:
- `Pattern_A: Multi-Division (>=2)` — 至少 2 次有效 Mass 跳变
- `Pattern_B: Single-Division (==1)` — 仅 1 次有效 Mass 跳变
- `Pattern_C: Exponential-Arrest (==0)` — 无有效跳变（指数增长后停滞）

### 阶段 4: 动态浓度异常剔除

第三级防御，剔除 ODE 发散的 LHS 样本:
```python
real_envelope = max(所有 Real Mutant 的逐变量极值)        # 39 维
dynamic_thresholds = max(real_envelope * 2.0, 2.0)        # 双门限
for lhs_sample in LHS_samples:
    if any(lhs_max_vals[var] > dynamic_thresholds[var]):
        delete(lhs_sample)                                  # 怪物细胞拦截
```

> **注意**: `lhs_v1_origin_no_filter.py` 移除了此阶段，用于对照实验。

### 阶段 5: 对齐与保存

- **参数键序**: `param_keys = sorted([k for k in defaults if k != 'is_mutant_104'])`（共 141 维）
- **输出路径**: `/home/users/zsy/mydata/budding_yeast/lhs_cheat_origin_{ver}_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`
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

## 三、去重与防泄漏机制

### 3.1 设计背景

数据集生成的核心矛盾：LHS 样本需要覆盖真实突变体（Real Mutant）的参数邻域以缩小 Sim-to-Real Gap，但又不能与真实突变体过于接近，否则会导致训练集混入测试集（数据泄漏）。

**数据泄漏的三个来源**:
1. **母体克隆**: LHS 样本与生成它的母体突变体参数完全一致
2. **跨家系污染**: 某家族 A 的 LHS 样本恰好接近另一家族 B 的真实突变体
3. **WT 污染**: LHS 样本的参数接近野生型（WT）默认值

### 3.2 各版本去重逻辑对比

> **2026-07-14 更新**: 除 V0 外，所有脚本（含 V1/V1-cheat/V1_origin/guaranteed/no_filter/V2-V6）均已升级为统一的 rtol=0.08 + 全局跨检查 + 0 值修复算法。V0 保留旧算法作为历史对照。

| 脚本 | 母体检查 rtol | 0 值修复 | 全局跨检查 | 内部去重范围 | MIN_PERTURBATION | 浓度修剪 | 冗余倍数 |
|------|--------------|----------|-----------|-------------|------------------|----------|----------|
| `lhs_v0.py` | 1e-5 (126 指纹) | ❌ | ❌ (仅完全一致) | 全局 5 位小数 | ❌ | ✅ | 20x attempts |
| `lhs_v1.py` ⚠️ | **0.08** | ✅ | ✅ (126 个) | 仅纯离散 | ❌ | ✅ | 10x |
| `lhs_v1_origin.py` | **0.08** | ✅ | ✅ (126 个) | 仅纯离散 | **0.30** | ✅ | 10x |
| `lhs_v1_origin_v2~v6.py` | **0.08** | ✅ | ✅ (126 个) | 仅纯离散 | **0.30** | ✅ | 10x |
| `lhs_v1_origin_guaranteed.py` | **0.08** | ✅ | ✅ (126 个) | 仅纯离散 | **0.30** | ✅ | 10x+3x 补采 |
| `lhs_v1_origin_no_filter.py` | **0.08** | ✅ | ✅ (126 个) | 仅纯离散 | **0.30** | ❌ | 10x |

> **注意 — `lhs_v1.py` 的 bug**: 该文件虽内联了新去重函数，但 `generate_lhs_tasks` 在调用 `build_cross_check_candidates(real_mutant_full, ...)` 时引用了未定义的 `real_mutant_full` 变量（从未调用 `build_real_mutant_full()` 构建）。直接运行该脚本会触发 `NameError`。下游代码（如 `analyze_samples.py`）仅导入其 `get_default_params` 等工具函数，不受影响。

> **注意 — 冗余倍数**: 所有新算法脚本当前使用 `candidate_count = actual_samples * 10`。project_memory 中要求 15x 冗余以补偿去重损失，但代码尚未跟进。如遇某些家族因去重严格导致采样不足，可考虑提升至 15x。

### 3.3 统一三层去重防泄漏逻辑（V1 及之后所有版本）

> **2026-07-14 更新**: 此逻辑原为 V2-V6 独有，现已推广至 V1/V1-cheat/V1_origin/guaranteed/no_filter。除 V0 外所有脚本的三个采样函数（`generate_lhs_tasks`, `generate_local_lhs_tasks`, `generate_targeted_lhs_tasks`）均实施以下三层去重。lhs_v1.py 例外：仅有 `generate_lhs_tasks`（无 local/targeted），且因 bug 无法直接运行。

#### 第一层: 母体近重复拦截 (rtol=0.08)

```python
is_identical = True
for p_name in mutated_params:
    _orig = original_rules[p_name]
    # 0 值修复：母体值=0 时 rtol 失效，改以 defaults[p_name]*8% 作为 atol
    _eff_atol = max(1e-6, defaults.get(p_name, 0.0) * 0.08) if _orig == 0.0 else 1e-6
    if not np.isclose(sample_dict[p_name], _orig, rtol=0.08, atol=_eff_atol):
        is_identical = False
        break
if is_identical:
    continue  # 剔除与母体参数偏差 <8% 的近重复样本
```

**关键点**:
- `rtol=0.08` 而非 `1e-5`：拦截"接近"的近重复，而非仅"完全一致"的克隆
- **0 值修复**: KO 突变体（参数=0）在 `rtol` 下 `np.isclose` 完全失效——任何 0 值样本都会被误判为与 0 值母体"完全一致"。改用 `atol = max(1e-6, defaults[p_name]*0.08)` 以默认值的 8% 作为绝对阈值

#### 第二层: 全局跨检查（核心防泄漏）

**预计算**: 构建全部 126 个真实突变体的完整参数向量:
```python
real_mutant_full = {}
for _rm_name, _rm_rules in mutant_rules.items():
    _rm_full = dict(defaults)
    _rm_full.update(_rm_rules)
    real_mutant_full[_rm_name] = _rm_full
```

**预筛优化** — 减少计算量:
```python
# 若突变体 B 的非共享突变参数远离 default，则样本（该参数=default）不可能接近 B
_mutated_set = set(mutated_params)
cross_check_candidates = []
for _rm_name, _rm_full in real_mutant_full.items():
    _is_cand = True
    for _rp, _rv in mutant_rules[_rm_name].items():
        if _rp in _mutated_set:  # 跳过共享参数
            continue
        _def = defaults.get(_rp, 0.0)
        _ea = max(1e-6, _def * 0.08) if _def == 0.0 else 1e-6
        if not np.isclose(_def, _rv, rtol=0.08, atol=_ea):
            _is_cand = False  # 非共享参数远离 default，不可能接近
            break
    if _is_cand:
        cross_check_candidates.append((_rm_name, _rm_full))
```

**逐候选检查**:
```python
_near_real = False
for _rm_name, _rm_full in cross_check_candidates:
    _near_this = True
    for p_name in mutated_params:
        _rv = _rm_full[p_name]
        _ea = max(1e-6, defaults.get(p_name, 0.0) * 0.08) if _rv == 0.0 else 1e-6
        if not np.isclose(sample_dict[p_name], _rv, rtol=0.08, atol=_ea):
            _near_this = False
            break
    if _near_this:
        _near_real = True
        break
if _near_real:
    continue  # 拦截跨家系污染和 WT 污染
```

**预筛的精巧之处**: 全局跨检查本需对每个样本 × 126 个突变体 × 全部参数做比对，计算量大。预筛通过"非共享参数远离 default 的突变体不可能被接近"这一逻辑，大幅缩减候选集。

#### 第三层: 纯离散内部去重

```python
if not continuous_mutated and not has_bub2:
    config_tuple = tuple(sample_dict.get(p) for p in sorted(mutated_params))
    if config_tuple in seen_configs:
        continue
    seen_configs.add(config_tuple)
```

仅对纯离散突变体（如 `mdt` 突变体）做配置元组去重，防止生成数百个完全一致的内部样本。

### 3.4 其他版本的去重逻辑

#### V0 (`lhs_v0.py`) — 全局盲抽（旧算法，未升级）

**三级去重**:
1. **内部去重** (`seen_lhs_configs`): 全部 141 维参数四舍五入到 5 位小数构建 tuple，全局去重
2. **真实克隆拦截**: `np.isclose(rtol=1e-5, atol=1e-8)` 逐参数比对全部 126 个真实突变体指纹，**完全一致**才丢弃
3. **最大突变数限制**: 单样本突变参数不超过真实突变体历史最大值

**特殊机制**: 高斯微扰涂抹 ±2% — 非突变连续参数加微小高斯噪声消除 Dirac Delta 尖峰

**缺陷**: 虽覆盖 126 个突变体，但阈值过严 (`rtol=1e-5`)，只能拦截"完全一致"的克隆，无法拦截"接近"的近重复。**保留旧算法作为历史对照，不再用于生成新数据集**。

#### V1-cheat (`lhs_v1.py`) — 家族式（新算法，但有 bug）

> **2026-07-14 更新**: 已内联新去重函数（`build_real_mutant_full`, `is_near_mother`, `is_near_any_real`），但存在 bug。

**去重逻辑**（设计意图）:
1. 母体近重复拦截: `rtol=0.08` + 0 值修复
2. 全局跨检查: 对 126 个真实突变体预筛 + 逐候选检查
3. 纯离散内部去重

**⚠️ 已知 bug**: `generate_lhs_tasks` 函数在循环体内调用了 `build_cross_check_candidates(real_mutant_full, ...)`，但 `real_mutant_full` 变量从未被构建（缺少 `real_mutant_full = build_real_mutant_full(mutant_rules, defaults)` 这一行）。直接运行会触发 `NameError: name 'real_mutant_full' is not defined`。

**当前用途**: 仅作为工具函数库被下游导入（`from lhs_v1 import get_default_params`），不直接运行生成数据集。如需修复，在 `generate_lhs_tasks` 函数体的 `dynamic_c_bounds` 循环后添加 `real_mutant_full = build_real_mutant_full(mutant_rules, defaults)` 即可。

#### V1 / V1-origin (`lhs_v1_origin.py`) — 基础采样引擎（新算法）

> **2026-07-14 更新**: 已升级为完整的 rtol=0.08 + 全局跨检查 + 0 值修复算法。

**三层去重**（与 3.3 节统一逻辑一致）:
1. 母体近重复拦截: `rtol=0.08` + 0 值修复
2. 全局跨检查: 对 126 个真实突变体预筛 + 逐候选检查
3. 纯离散内部去重

**采样函数**: `generate_lhs_tasks` + `generate_local_lhs_tasks`（perturbation=0.3）。无 `generate_targeted_lhs_tasks`。

#### Guaranteed (`lhs_v1_origin_guaranteed.py`) — 保证产量版（新算法）

> **2026-07-14 更新**: 已升级为完整防泄漏算法。perturbation 从 0.1 更新为 0.3。

**三层去重**（与 3.3 节统一逻辑一致）:
1. 母体近重复拦截: `rtol=0.08` + 0 值修复
2. 全局跨检查: 对 126 个真实突变体
3. 纯离散内部去重

**Phase A-G 保证产量机制**:
- Phase A: 生成冗余 LHS 任务 (`SURPLUS_FACTOR=3` 倍)
- Phase B: 模拟全部
- Phase C: 计算动态浓度阈值
- Phase D: 浓度剔除
- Phase E: 统计存活数量
- Phase F: 补采循环（最多 `MAX_TOPUP_ROUNDS=5` 轮，每轮缺口 × 3 倍冗余）
- Phase G: 裁剪到恰好目标数量

#### No-filter (`lhs_v1_origin_no_filter.py`) — 无过滤版（新算法）

> **2026-07-14 更新**: 已升级为完整防泄漏算法。内部去重仍为纯离散范围（非全局）。

**三层去重**（与 3.3 节统一逻辑一致）:
1. 母体近重复拦截: `rtol=0.08` + 0 值修复
2. 全局跨检查: 对 126 个真实突变体
3. 纯离散内部去重（`if not continuous_mutated and not has_bub2:` 条件下的 `config_tuple` 去重）

**移除的过滤**:
- ❌ 移除智能截断逻辑（强制所有突变体采样数 = `num_lhs_samples`）
- ❌ 移除动态浓度修剪（第三级防御）

**用途**: 用于对照实验，验证浓度修剪和智能截断策略的效果。防泄漏去重逻辑完整保留。

### 3.5 关键经验教训

1. **除 V0 外所有脚本已统一防泄漏逻辑**: 全局跨检查 (126 个真实突变体) + 0 值修复 + 8% 近重复阈值。V0 保留旧算法作为历史对照
2. **0 值修复是关键**: KO 突变体 (参数=0) 在 `rtol=1e-5` 下 `np.isclose` 完全失效，会误判所有 0 值样本为重复
3. **V0 的克隆拦截虽覆盖 126 个突变体，但阈值过严**: 只能拦截"完全一致"的克隆，无法拦截"接近"的近重复（如参数偏差 3% 的样本会漏网）
4. **lhs_v1.py 的 bug 提醒**: 批量升级去重算法时，`generate_lhs_tasks` 中引用 `real_mutant_full` 前必须先调用 `build_real_mutant_full()` 构建。其他脚本均已正确构建

---

## 四、核心采样引擎

### 4.1 全局 LHS 采样 `generate_lhs_tasks(num_lhs_samples)`

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
- 等价碰撞拦截器（V2-V6 使用 rtol=0.08 + 0 值修复，详见第三章）
- 纯离散突变体智能截断（防止生成数百个完全相同的克隆）

### 4.2 RGS 局部密集采样 `generate_local_lhs_tasks(target_mutants, ...)`

**职责**: AL 触发时基于 Real Mutant MAE 反馈做局部密集采样。

**与 `generate_lhs_tasks` 的唯一区别**:
```python
# generate_lhs_tasks: 全局范围 [0, max_val × 1.5]
# generate_local_lhs_tasks: 局部范围 [center × (1 - pert), center × (1 + pert)]
```

**参数差异**:
- 所有新算法脚本（V1/V1_origin/guaranteed/no_filter/V2-V6）: `perturbation=0.3` (±30%)
- V0: 无此函数（全局盲抽，不基于 Real Mutant 局部采样）

**键名后缀**: `_RGS_{idx:03d}`（区别于全局 LHS 的 `_LHS_`）

### 4.3 定向扩展采样 `generate_targeted_lhs_tasks(target_families, ...)`

**职责**: 按 `target_families` 字典配置，对指定家族做定向密集采样，补齐 LHS 训练分布的覆盖盲区。

#### 4.3.1 关键变量与术语解释

| 变量名 | 含义 | 来源 | 具体数值例子 |
|--------|------|------|-------------|
| `center` / `center_val` | 突变体规则中该连续参数的**原始突变值**，作为定向采样的中心点 | `original_rules[p_name]` | `73_cdc20_ts` 的 `ks20p=0.006`；`34_cdh1_KO` 的 `init_CDH1T=0.0` |
| `pert` / `perturbation` | **扰动幅度**（相对比例），控制采样范围围绕 center 的扩展宽度 | 家族配置中的 `perturbation` 字段 | V2 cdc20_ts 家族 `pert=0.1`；V3 contamination_fix 家族 `pert=0.08` |
| `b_val` | **boundary 模式的目标边界值**（通常为 0.0）。仅在 boundary 模式且该参数在 `boundary_params` 中时使用 | 家族配置中的 `boundary_params[p_name]` | V2 cdc20_ts `boundary_params={'ks20p': 0.0}` → `b_val=0.0` |
| `ε` (epsilon) | **小偏移量**（硬编码 `1e-6`），防止 `u_bound` 精确等于 `b_val` 导致 LHS 退化 | 代码常量 | `b_val + 1e-6 = 0.000001` |
| `anchor_params` | **anchor 模式的强制锚点值字典**，直接指定 CDH1T/CDH1 的取值 | 家族配置中的 `anchor_params` 字段 | V2 cdh1 家族 `anchor_params={'init_CDH1T': 0.0, 'init_CDH1': 0.0}` |
| `cdh1_pairs` | **CDH1T/CDH1 强绑定元组列表**，由 `build_discrete_pools` 自动构建 | `build_discrete_pools()` 返回值 | `[(1.0, 0.9304992), (0.0, 0.0)]` |
| `single_pools` | **离散参数合法真值集合字典** | `build_discrete_pools()` 返回值 | `single_pools['mdt'] = [90.0, 150.0, 160.0]` |

#### 4.3.2 三种采样模式详解

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

##### 模式 2: `boundary` — 边界扩展到 0（覆盖彻底敲除）

**连续参数采样范围**: `[max(0, min(b_val, center × (1-pert))), max(center × (1+pert), b_val + 1e-6)]`

**CDH1T/CDH1 处理**: 从 `cdh1_pairs` 随机选一个元组

**设计意图**: 扩展连续参数的采样范围到 0 边界，覆盖 Real Mutant 的彻底敲除参数（如 `ks20p=0`）。**适用于 KO 突变体**，但可能引入邻域污染。

##### 模式 3: `anchor` — CDH1T/CDH1 强制锚点（高风险高收益）

**连续参数采样范围**: `[max(0, center × (1-pert)), center × (1+pert)]`（与 density 相同）

**CDH1T/CDH1 处理**: **直接使用 `anchor_params` 字典中的值**（绕过 cdh1_pairs 随机选择），强制实现 CDH1T/CDH1=(0,0) 锚点

**设计意图**: 强制所有样本的 init_CDH1T=0/init_CDH1=0，生成纯 CDH1T=0 的 LHS 样本。**高风险**：邻域污染风险高（V2 的 4 个 cdh1 突变体因 anchor 模式表现不佳，V3 改用 density 修复）。

#### 4.3.3 三种模式对比汇总

| 模式 | 连续参数采样范围 | CDH1T/CDH1 处理 | 风险等级 | 适用场景 |
|------|------------------|------------------|----------|----------|
| `density` | `[max(0, center×(1-pert)), center×(1+pert)]` | 从 `cdh1_pairs` 随机选 | 🟢 低（最安全） | 邻域密度增强、邻域污染修复 |
| `boundary` | `[max(0, min(b_val, center×(1-pert))), max(center×(1+pert), b_val+1e-6)]` | 从 `cdh1_pairs` 随机选 | 🟡 中（可能邻域污染） | KO 突变体、彻底敲除覆盖 |
| `anchor` | `[max(0, center×(1-pert)), center×(1+pert)]` | 直接用 `anchor_params` 强制赋值 | 🔴 高（邻域污染风险高） | CDH1T/CDH1 强制 (0,0) 锚点 |

**键名后缀**: `_LHS_T{idx:03d}`（T=Targeted，保持 `_LHS_` 子串匹配兼容）

---

## 五、版本演进

### V0（`lhs_v0.py`）— 全局盲抽 LHS

**核心特征**:
- 80,000 个全局盲抽样本，命名 `LHS_Sample_{idx:05d}`
- 概率稀疏突变: `p_mutate = 0.35 + 0.5*(count/max_count)`
- 14 个离散保护参数（比 V1 多 7 个: `kscdh`, `kdcdh`, `ks14`, `kd14`, `ksnet`, `kdnet`, `Vi20_active`）
- 高斯微扰涂抹 ±2%（非突变连续参数加微小噪声消除 Dirac Delta 尖峰）
- 使用全部 CPU 核心（无绅士模式）

**去重逻辑**: 见第三章 3.4 节

**输出**: `lhs_blind_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`

---

### V1-cheat（`lhs_v1.py`）— 家族式 LHS

**核心特征**:
- 每个突变体 10,000 个样本，命名 `{mutant}_LHS_{idx:03d}`
- 14 个离散保护参数（与 V0 相同）

**去重逻辑**: 见第三章 3.4 节

**输出**: `lhs_cheat_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`

---

### V1（`lhs_v1_origin.py`）— 基础 LHS 采样引擎

**核心改进**（相对于 V0/V1-cheat）:
1. 引入 `DISCRETE_STRICT_PARAMS` 严格离散保护名单（7 个参数），废弃 V0/V1-cheat 的 14 个参数名单
2. 实现 `CDH1T/CDH1` 强绑定元组机制，保障 `活性 = 总数 × 比例` 生物学约束
3. `BUB2_active/init_BUB2` 同频连续缩放系数 ∈ [0, 1.5]
4. 等价碰撞拦截器（原始 rtol=1e-5，**2026-07-14 升级为 rtol=0.08 + 0 值修复 + 全局跨检查**）
5. 纯离散突变体智能截断（防止无意义克隆）
6. 双重打标系统（bio_label + pattern_label）
7. 动态浓度异常剔除系统（双门限：相对 2.0× + 绝对 2.0）
8. `eqns` 修复 `is_mutant_104` 的 180min 时序条件分支
9. 绅士模式: `safe_cores = max(1, cpu_count // 4)`

**采样规模**:
- 124 突变体规则（3 WT + 121 非-WT）
- 1000 LHS 样本/突变体 → ~121,000 LHS + 124 Real = ~121,124 总样本

**去重逻辑**: 见第三章 3.4 节

**输出**: `lhs_cheat_origin_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`（即原始无版本后缀的文件）

---

### V2（`lhs_v1_origin_v2.py`）— 三大家族定向扩展 + 防泄漏完善

**核心改进**（相对于 V1）:
1. **新增 `generate_targeted_lhs_tasks` 函数**，支持 density / boundary / anchor 三种采样模式
2. **新增 `DEFAULT_TARGET_FAMILIES` 字典**，4 类家族定向扩展
3. **引入 `boundary_params` / `anchor_params` 字段**
4. **完整的防泄漏逻辑**: 全局跨检查 + 0 值修复 + rtol=0.08（详见第三章）

**家族配置**（`DEFAULT_TARGET_FAMILIES`）:

| 家族 | 突变体数 | num_per_mutant | mode | 扰动 | boundary_params / anchor_params | 理论采样总数 |
|------|---------|----------------|------|------|----------------------------------|------------|
| `cdc20_ts` | 8 | 500 | boundary | 0.1 | `{'ks20p': 0.0, 'ks20pp': 0.0}` | 4,000 |
| `CLB1_clb2` | 5 | 500 | boundary | 0.2 | `{'ksb2p': 0.0, 'ksb2pp': 0.0}` | 2,500 |
| `cdh1` | 10 | 500 | anchor | 0.1 | `{'init_CDH1T': 0.0, 'init_CDH1': 0.0}` | 5,000 |
| `low_lhs_count` | 4 | 300 | density | 0.1 | 无 | 1,200 |
| **合计** | **27** | — | — | — | — | **12,700** |

**实验效果**（V2+none vs V1 baseline）:
- Real MAE: 0.3645 → 0.1926（**-47%**）
- cdc20_ts 家族 MAE: -97%
- Pattern C（指数停滞）MAE: -59%
- Sim-to-Real Gap: -34%

**输出**: `lhs_cheat_origin_v2_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`

---

### V3（`lhs_v1_origin_v3.py`）— 残留瓶颈定向扩展

**核心改进**（相对于 V2）:
1. **新增 `V3_EXPANSION_FAMILIES` 字典**，针对 V2+none eval 中 26 个 MAE > 0.3 突变体，分 5 类家族定向修复
2. **修复 V2 boundary 邻域污染**：对 V1→V2 恶化的 9 个突变体用 density + 小扰动（0.08）+ 翻倍采样数（800）修复
3. **修复 V2 anchor 邻域污染**：对 cdh1 已采样但仍差的 4 个突变体改用 density 模式
4. **新增 KO 家族覆盖**：boundary 模式扩展到 Tem1T=0/Cdc15T=0

**家族配置**（`V3_EXPANSION_FAMILIES`）:

| 家族 | 突变体数 | num_per_mutant | mode | 扰动 | 理论采样总数 | 设计理由 |
|------|---------|----------------|------|------|------------|----------|
| `CLB2_db_family` | 7 | 500 | density | 0.15 | 3,500 | V2 未采样，MAE 均值 0.685 |
| `cln_triple` | 4 | 500 | density | 0.15 | 2,000 | V2 未采样，MAE 均值 0.617 |
| `cdh1_contamination_fix` | 4 | 800 | density | 0.10 | 3,200 | V2 anchor 邻域污染修复 |
| `KO_family` | 3 | 500 | boundary | 0.10 | 1,500 | V2 未采样 KO（Tem1T=0, Cdc15T=0） |
| `contamination_fix` | 9 | 800 | density | 0.08 | 7,200 | V2 boundary 邻域污染修复 |
| **合计** | **27** | — | — | — | **17,400** | — |

**V3 Eval 结果**:
- Real MAE: 0.1926 → 0.1013（-47%）
- Real MSE: 4.43 → 0.442（-90%）
- Sim-to-Real Gap: 4.63× → 1.89×
- Tier 4 (MAE>0.4): 26 → 5

**输出**: `lhs_cheat_origin_v3_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`

---

### V4（`lhs_v1_origin_v4.py`）— 邻域污染修复 + 覆盖盲区扩展

**核心改进**（相对于 V3）:
1. **修复 KO_family 邻域污染**：boundary → density，pert 0.10→0.05
2. **新增 6 个家族**：tem1_ts/cdc14_ts/swi5_cdh1/cln1_cln2_KO/sic1_KO/GAL_CLB2_cdh1
3. **扩展 2 个家族**：cln_triple（4→11）、CLB1_clb2（2→4，pert 0.2→0.3）
4. **增强 CLB2_db**：7→10 突变体，pert 0.15→0.20，num 500→800

**家族配置**（`V4_EXPANSION_FAMILIES`，12 类家族，67 突变体，~44K 定向样本）

**预期效果**:
- Real MAE: 0.1013 → 0.080–0.090
- 覆盖率: 42.9% → 75%+

**输出**: `lhs_cheat_origin_v4_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`

---

### V5（`lhs_v1_origin_v5.py`）— 7 项定向修复

**核心改进**（相对于 V4）:
1. **修复 CLB1_clb2_extended**：boundary → density，pert 0.30→0.50，num 500→1000
2. **新增 cdc6_family**：覆盖 5 个 cdc6 相关突变体
3. **增强 tem1_ts_family**：num 800→1200，pert 0.08→0.12
4. **新增 4 个家族**：CLB2_db_clb5_Raff/APC_A_GAL_CLB2/cdc15_net1_ts/sic1_cdh1

**家族配置**（`V5_EXPANSION_FAMILIES`，17 类家族，88 突变体，~54K 定向样本）

**V5 数据集分析结果**:

| 指标 | v4 | v5 | 变化 |
|------|----|----|------|
| 总样本数 | 157,599 | 162,134 | +4,535 (1.03×) |
| Viable 占比 | 47.4% | 48.7% | +1.3pp |
| Pattern C (Arrest) | 40.0% | 38.5% | -1.5pp |

**输出**: `lhs_cheat_origin_v5_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`

---

### V6（`lhs_v1_origin_v6.py`）— P0-P5 定向修复

**核心改进**（相对于 V5）:
1. P0: 拆分 tem1_ts_family，新增 tem1_ts_multi_family（窄扰动 pert=0.05 专治 88/89）
2. P1: 新增 CLB2_db_Gal_family（覆盖 mdt 参数盲区，专治 56/61.1）
3. P2: 增强 CLB2_db_clb5_Raff_family（pert 0.25→0.30, num 400→600）
4. P3: 增强 APC_A_GAL_CLB2_family（pert 0.15→0.20, num 400→600）
5. P4: 修复 sic1_cdh1_family（纳入 36，pert 0.12→0.10）
6. P5: 新增 clb5_family（覆盖 63/72 残留盲区）

**输出**: `lhs_cheat_origin_v6_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`

---

## 六、实验对照脚本

### 6.1 `lhs_v1_origin_guaranteed.py` — 保证产量版

**用途**: 通过 Phase A-G 补采机制保证每个突变体恰好达到目标采样数，用于验证"保证产量"对模型训练的影响。

> **2026-07-14 更新**: 已升级为完整防泄漏算法（rtol=0.08 + 全局跨检查 + 0 值修复）。perturbation 从 0.1 更新为 0.3。

**与 V1 的关键差异**:
- `NUM_LHS_SAMPLES=500`（V1 为 1000）
- `SURPLUS_FACTOR=3`（初始生成 3 倍冗余）
- `MAX_TOPUP_ROUNDS=5`（最多 5 轮补采）
- `perturbation=0.3`（RGS 扰动，与 V1/V2-V6 一致）
- 母体近重复拦截: `rtol=0.08` + 0 值修复（与 V1/V2-V6 一致）
- **全局跨检查: ✅**（对 126 个真实突变体）

**Phase A-G 流程**:
```
Phase A: 生成冗余 LHS 任务 (NUM_LHS_SAMPLES × SURPLUS_FACTOR)
Phase B: 模拟全部 (Real Mutant + LHS)
Phase C: 计算动态浓度阈值 (从 Real Mutant)
Phase D: 对 LHS 样本执行浓度剔除
Phase E: 统计每个 mutant 的存活 LHS 数量
Phase F: 补采循环 — 为不足的 mutant 补充样本 (最多 MAX_TOPUP_ROUNDS 轮)
Phase G: 裁剪到每个 mutant 恰好 NUM_LHS_SAMPLES 个
```

**输出**: `lhs_cheat_origin_guaranteed_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`

---

### 6.2 `lhs_v1_origin_no_filter.py` — 无过滤版

**用途**: 移除浓度修剪和智能截断，用于验证这两种过滤策略对模型训练的必要性。防泄漏去重逻辑完整保留。

> **2026-07-14 更新**: 已升级为完整防泄漏算法（rtol=0.08 + 全局跨检查 + 0 值修复）。内部去重仍为纯离散范围。

**与 V1 的关键差异**:
- `NUM_LHS_SAMPLES=500`（V1 为 1000）
- **移除智能截断逻辑**（强制所有突变体采样数 = `num_lhs_samples`）
- **移除动态浓度修剪**（第三级防御）
- 纯离散内部去重（`if not continuous_mutated and not has_bub2:` 条件下，与 V1 一致）
- 母体近重复拦截: `rtol=0.08` + 0 值修复（与 V1/V2-V6 一致）
- **全局跨检查: ✅**（对 126 个真实突变体）

**输出**: `lhs_cheat_origin_no_filter_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`

---

### 6.3 `test_lhs_filter_analysis.py` — 过滤阶段损失分析

**用途**: 追踪每个过滤操作对各家族 LHS 样本的影响，诊断采样损失瓶颈。

> **2026-07-14 注意**: 此脚本基于 `lhs_v1_origin.py` 的**旧逻辑副本**（`generate_lhs_with_stats` 复制了原始 `generate_lhs_tasks` 并插桩），仍使用 `rtol=1e-5` 等价碰撞判定和 `candidate_count = target × 10`，未跟进新算法。分析结论中的损失率可能与新算法脚本的实际损失不同。如需精确诊断新算法的损失分布，需同步更新此脚本的去重逻辑。

**追踪的 3 个过滤阶段**:
1. **Stage 1**: 等价碰撞拦截 + 纯离散去重（LHS 生成阶段）
2. **Stage 2**: 超时 + Discarded_Mass（ODE 模拟阶段）
3. **Stage 3**: 动态浓度剔除（2× Real Mutant）（模拟后处理阶段）

**基于 `lhs_v1_origin.py`**（旧 V1 逻辑副本），支持通过命令行参数自定义离散参数名单:
```bash
python test_lhs_filter_analysis.py                                # 默认 (1000样本/突变体)
python test_lhs_filter_analysis.py --remove ESP1T                 # ESP1T 变为连续参数
python test_lhs_filter_analysis.py --remove init_CDH1T init_CDH1   # CDH1 对绑定解除
python test_lhs_filter_analysis.py --num-samples 200              # 小规模采样
python test_lhs_filter_analysis.py --skip-simulation              # 只分析 LHS 生成阶段
```

**输出**: 每阶段每家族的损失统计表 + 完整损失链路汇总（目标采样数 → 候选生成 → 等价碰撞拦截 → 纯离散去重 → 有效 LHS → ODE 超时 → ODE Mass 丢弃 → ODE 失败 → ODE 成功 → 浓度剔除 → 最终数据集）

---

### 6.4 `run_lhs_batch.sh` — 批量运行脚本

**用途**: 批量运行 v2-v6 数据生成脚本。

**特性**:
- 支持 BLAS 死锁修复（`OMP_NUM_THREADS=1` 等）
- 支持命令行参数指定版本: `./run_lhs_batch.sh 5 6`（仅运行 v5, v6）
- 默认版本顺序: `5 6 4 3 2`（先生成最新版本）
- 日志保存到 `/home/users/zsy/mydata/budding_yeast/logs/batch_lhs_{TIMESTAMP}.log`

```bash
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data
./run_lhs_batch.sh           # 运行 v5,v6,v4,v3,v2
./run_lhs_batch.sh 2 4       # 仅运行 v2,v4
SIM_TF=300 ./run_lhs_batch.sh  # 自定义模拟时长
```

---

## 七、全局配置参数

```python
SIM_TF = 210            # 模拟总时长 (min)，可通过环境变量 SIM_TF 覆盖
SIM_STEPS = 500          # 采样总步数，可通过环境变量 SIM_STEPS 覆盖
NUM_LHS_SAMPLES = 1000  # 每个非-WT 突变体的全局 LHS 采样数量 (V0=80000, V1-cheat=10000, guaranteed/no_filter=500)
SIM_TIMEOUT = 5.0       # 单个细胞求解超时阈值 (秒)
```

**离散保护名单**（`DISCRETE_STRICT_PARAMS`）:

V0/V1-cheat（14 个参数）:
```python
['init_CDH1T', 'init_CDH1', 'kscdh', 'kdcdh',
 'ks14', 'kd14', 'ksnet', 'kdnet',
 'mdt', 'Vi20_active', 'BUB2_active', 'init_BUB2', 'is_mutant_104', 'ESP1T']
```

V1 及之后（7 个参数，精简版）:
```python
['init_CDH1T', 'init_CDH1',
 'mdt', 'BUB2_active', 'init_BUB2', 'is_mutant_104', 'ESP1T']
```

> **差异说明**: V0/V1-cheat 将 `kscdh/kdcdh/ks14/kd14/ksnet/kdnet/Vi20_active` 也视为离散参数保护。V1-origin 发现这些参数作为连续参数采样效果更好，精简为 7 个。

**输出路径**:
- 目录: `/home/users/zsy/mydata/budding_yeast/`
- 文件名: `lhs_cheat_origin_{ver}_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`

---

## 八、运行方式

```bash
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data

# 单个版本生成
PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast:$PYTHONPATH \
    /home/users/zsy/.conda/envs/neuralop/bin/python lhs_v1_origin_v5.py

# 批量生成 v2-v6（推荐，含 BLAS 死锁修复）
./run_lhs_batch.sh

# 批量生成指定版本
./run_lhs_batch.sh 5 6

# 实验对照脚本
PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast:$PYTHONPATH \
    /home/users/zsy/.conda/envs/neuralop/bin/python lhs_v1_origin_guaranteed.py
PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast:$PYTHONPATH \
    /home/users/zsy/.conda/envs/neuralop/bin/python lhs_v1_origin_no_filter.py

# 过滤损失分析
/home/users/zsy/.conda/envs/neuralop/bin/python test_lhs_filter_analysis.py --num-samples 200 --skip-simulation
```

**耗时**: V2 约 2-3 小时（~125,000 ODE 求解），V3 约 3-4 小时（~133,000 ODE 求解），V4 约 4-5 小时（~165,000 ODE 求解），V5 约 4-5 小时（~170,000 ODE 求解）

---

## 九、更新日志

> **维护规则**: 每次对数据生成代码做改进（新增家族、调整采样参数、修改采样逻辑等），必须在此追加一条记录。

### [2026-07-17] ODE BDF 死锁修复 — SIGALRM 硬超时兜底
**文件**: `lhs_v0.py`, `lhs_v1.py`, `lhs_v1_origin.py`, `lhs_v1_origin_no_filter.py`, `lhs_v1_origin_guaranteed.py`, `lhs_v1_origin_v2.py` ~ `lhs_v1_origin_v6.py`（共 10 个）
**改动**:
- 在 `simulate_mutant` 函数内引入 SIGALRM 硬超时兜底，替代原 `eqns_with_timeout` 的 Python 层 `time.time()` 检查
- 模式: `signal.alarm(SIM_TIMEOUT)` + `try/finally` 确保 `alarm(0)` 取消与 `_old_handler` 恢复
- `test_lhs_filter_analysis.py` 无需修改（通过 `from lhs_v1_origin import simulate_mutant` 间接获得保护）
**原因**: 病态 LHS 参数导致 BDF 求解器在 Fortran 层陷入步长无限缩减循环，Python 层超时检查永不执行，`future.result(timeout=)` 只计数不杀 worker，16 个 worker 陆续耗尽后 `as_completed` 永久阻塞
**效果**: 病态样本 5 秒后被 SIGALRM 中断丢弃，worker 释放继续处理下一个 task；正常样本（<3 秒）零影响
**详见**: 11.4.2 节

### [2026-07-14] 文档同步更新 — 去重算法统一 + lhs_v1.py bug 记录
**文件**: `DATASET_GENERATION_GUIDE.md`
**改动**:
- 第一章文件清单表新增"去重算法"列，标注各文件使用新/旧算法
- 标注 `lhs_v1.py` 存在 bug：`generate_lhs_tasks` 引用未定义的 `real_mutant_full`，直接运行会触发 `NameError`
- 标注 `test_lhs_filter_analysis.py` 基于旧 V1 逻辑副本，未跟进新算法
- 第三章去重对比表（3.2 节）全面修正：V1/V1-cheat/V1_origin/guaranteed/no_filter 的 rtol 从 1e-5 更正为 0.08，全局跨检查从 ❌ 更正为 ✅，0 值修复从 ❌ 更正为 ✅
- 3.3 节从"V2-V6 的三层去重"改名为"统一三层去重防泄漏逻辑"（V1 及之后所有版本）
- 3.4 节修正各版本描述：V1/V1_origin/guaranteed/no_filter 均已升级为新算法；no_filter 内部去重更正为"纯离散范围"（非全局）
- 4.2 节 perturbation 更正：所有新算法脚本统一为 0.3（原文档说 V1/guaranteed/no_filter 为 0.1）
- 6.1 节 guaranteed 更正：perturbation=0.3，有全局跨检查
- 6.2 节 no_filter 更正：纯离散内部去重，有全局跨检查
- 6.3 节 test_lhs_filter_analysis 添加注意：基于旧逻辑副本
- 11.3 节数据泄漏问题更新：V1 系列已修复
- 新增冗余倍数注意事项：当前代码用 10x，project_memory 要求 15x
**原因**: 2026-07-14 批量升级去重算法后，文档未同步更新，仍记录旧算法参数

### [2026-07-12] 文档全面更新 — 去重防泄漏机制 + 路径修复 + 实验脚本文档
**文件**: `DATASET_GENERATION_GUIDE.md`
**改动**:
- 新增第三章"去重与防泄漏机制"，完整记录 10 个数据生成脚本的去重逻辑对比
- 新增第六章"实验对照脚本"，记录 guaranteed/no_filter/test_lhs_filter_analysis/run_lhs_batch.sh
- 修复所有文件路径: `/home/users/zsy/trae_workspace/` → `/home/users/zsy/projects/budding_yeast_cell_cycle/`
- 修复输出路径: `/home/users/zsy/mydata/budding_yeast/`
- 修复 `DISCRETE_STRICT_PARAMS` 文档: 区分 V0/V1-cheat (14 个) 与 V1-origin (7 个)
- 补充 BLAS 死锁修复说明 (`OMP_NUM_THREADS=1` 等)
- 文件清单扩展至 13 个文件（原 7 个）
**原因**: 上下文丢失后重新分析全部 10 个数据生成脚本的去重逻辑，发现文档严重滞后

### [2026-07-12] BLAS 死锁修复
**文件**: `run_lhs_batch.sh`
**改动**: 新增 4 个 BLAS 线程限制环境变量
```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```
**原因**: `ProcessPoolExecutor` 用 `fork` 创建子进程时，OpenBLAS 线程池的锁状态不一致导致死锁。症状：所有进程 0% CPU，main 进程 `wchan=pipe_write`，worker 进程 `wchan=futex_wait_queue`
**效果**: v5/v6 批量生成不再卡死，16 进程 × 1 线程 = 16 核，匹配 `safe_cores`

### [2026-07-07] V6 - 修复 P0-P5 残留瓶颈
**文件**: `lhs_v1_origin_v6.py`（新建）
**改动**:
- 复制自 `lhs_v1_origin_v5.py`，修改 `DATASET_NAME` 加 `_v6` tag
- 新增 `V6_EXPANSION_FAMILIES` 字典（P0-P5 修复）
  - P0: 拆分 tem1_ts_family，新增 tem1_ts_multi_family (pert=0.05 专治 88/89)
  - P1: 新增 CLB2_db_Gal_family (覆盖 mdt 参数盲区，专治 56/61.1)
  - P2: 增强 CLB2_db_clb5_Raff_family (pert 0.25→0.30, num 400→600)
  - P3: 增强 APC_A_GAL_CLB2_family (pert 0.15→0.20, num 400→600)
  - P4: 修复 sic1_cdh1_family (纳入 36，pert 0.12→0.10)
  - P5: 新增 clb5_family (覆盖 63/72 残留盲区)
- 修改 `__main__` 块：不再调用 `V5_EXPANSION_FAMILIES`，改调用 `V6_EXPANSION_FAMILIES`

### [2026-07-06] V5 - 7 项定向修复
**文件**: `lhs_v1_origin_v5.py`（新建）
**改动**:
- 复制自 `lhs_v1_origin_v4.py`，新增 `V5_EXPANSION_FAMILIES`（17 类家族，88 突变体，~54K 定向样本）
- 修复 CLB1_clb2_extended（boundary→density），新增 cdc6/CLB2_db_clb5_Raff/APC_A_GAL_CLB2/cdc15_net1_ts/sic1_cdh1 家族
- 增强 tem1_ts_enhanced（num 800→1200，pert 0.08→0.12）

### [2026-07-06] 下游分析 regex bug 修复
**文件**: `mydata/budding_yeast/dataset_analysis/real_vs_lhs_csv.py`、`visualization/plot_real_vs_lhs.py`、`visualization/plot_all_cheat_samples.py`
**改动**: 3 个脚本共 4 处正则 `^[_LHS0-9\-]+$` → `^[_LHST0-9\-]+$`（字符类新增 `T`）
**原因**: 定向样本命名为 `{mutant}_LHS_T{idx}`，`T` 不在原字符类中导致正则不匹配，V5 中 54,619 个定向样本被排除

### [2026-07-05] V4 - 邻域污染修复 + 覆盖盲区扩展
**文件**: `lhs_v1_origin_v4.py`（新建）
**改动**:
- 新增 `V4_EXPANSION_FAMILIES`（12 类家族，67 突变体，~44K 定向样本）
- 修复 KO_family（boundary→density），新增 6 个家族，扩展 2 个家族

### [2026-07-05] V3 - 残留瓶颈定向扩展
**文件**: `lhs_v1_origin_v3.py`（新建）
**改动**: 新增 `V3_EXPANSION_FAMILIES`（5 类家族，27 突变体，17,400 定向样本）

### [2026-07-05] V2 - 三大家族定向扩展
**文件**: `lhs_v1_origin_v2.py`（新建）
**改动**: 新增 `DEFAULT_TARGET_FAMILIES`（4 类家族）+ `generate_targeted_lhs_tasks` 函数 + 完整防泄漏逻辑

### [2026-07-05] V1 - 基础 LHS 采样引擎
**文件**: `lhs_v1_origin.py`
**改动**: 引入 `DISCRETE_STRICT_PARAMS`（7 个）、CDH1 强绑定、BUB2 同频缩放、等价碰撞拦截器、智能截断、双重打标、动态浓度剔除

---

## 十、下游依赖

| 下游模块 | 依赖文件 | 依赖内容 |
|---------|---------|----------|
| `utils/al_callback.py` | `lhs_v1_origin.py` | `simulate_mutant`, `get_default_params`, `generate_lhs_tasks`, `generate_local_lhs_tasks`, `mutant_rules`, `eqns` |
| `train.py` | npz 数据集 | `data`, `params`, `mutant_names`, `labels`, `pattern_labels` |
| `eval/` 评估脚本 | npz 数据集 | 同上 |
| `run_pipeline.sh` | npz 数据集 | `DATASET_V2`, `DATASET_V3`, `DATASET_V4`, `DATASET_V5` 变量 |
| `mydata/budding_yeast/dataset_analysis/` | npz 数据集 + `mutant_names` | `real_vs_lhs_csv.py` 依赖 `_LHS_`/`_LHS_T` 子串匹配区分 Real 与 LHS 样本（regex 已修复）；`correlation_analysis.py` 依赖 `Sensitivity_Atlas.csv` |
| `mydata/budding_yeast/run_analysis.py` | npz 数据集 | 通用分析流水线（10 个原子分析脚本） |

**重要约束**（来自 `project_memory.md`）:
- `lhs_v1_origin.py` 的 `generate_local_lhs_tasks` 函数（RGS 专属算法）必须在同步到 `mydata/budding_yeast/` 时保留
- v2 数据集生成代码必须包含 `os.makedirs(OUTPUT_DIR, exist_ok=True)`
- v2 数据集必须使用 `np.savez_compressed` 保存完整路径
- 生成后必须打印完整输出路径
- 所有突变体采样必须使用严格参数去重 (rtol=0.08, atol=1e-6 for non-zero)
- 零值突变参数去重 atol 必须设为 `max(1e-6, default_value * 0.08)`
- 所有 LHS 采样函数必须对全部 126 个真实突变体做全局跨检查
- 所有 LHS 采样函数必须包含显式 WT 检查
- 目标采样最小扰动从 0.1 增至 0.30 以避免接近母体突变体

---

## 十一、设计原则与经验教训

### 11.1 采样模式选择

- **density 模式**：局部邻域扰动，**最安全**，不会引入新的边界样本污染
- **boundary 模式**：扩展到 0 边界，**适用于彻底敲除突变体**（如 KO），但可能引入邻域污染
- **anchor 模式**：CDH1T/CDH1 强制锚点，**适用于已知强绑定约束**，但邻域污染风险高

### 11.2 邻域污染问题

**V2 教训**: boundary 采样虽然解决了 cdc20_ts/CLB1_clb2 家族，但引入了 7 个新的 V1→V2 恶化突变体。

**V3 修复策略**: 对 V2 邻域污染突变体用 density 模式 + 小扰动（0.08）+ 翻倍采样数（800）精准修复。

### 11.3 数据泄漏问题

> **2026-07-14 更新**: V1 系列已修复，不再有此问题。

**历史根因**: V1/V1_origin/guaranteed/no_filter 原先只检查母体克隆（rtol=1e-5），不检查与其他 125 个真实突变体的接近性。一个 cdc14_ts 家族的 LHS 样本如果参数恰好接近 cdc20_ts 家族的真实突变体，完全不会被拦截。

**修复策略（已实施）**: 除 V0 外所有脚本均已升级为全局跨检查（126 个真实突变体）+ rtol=0.08 近重复阈值 + 0 值修复。

**0 值修复的必要性**: KO 突变体（参数=0）在 `rtol=1e-5` 下 `np.isclose` 完全失效——`np.isclose(x, 0, rtol=1e-5)` 等价于 `abs(x) < 1e-8`，几乎拦截所有 0 值样本。改用 `atol = max(1e-6, defaults[p_name]*0.08)` 以默认值的 8% 作为绝对阈值，使 KO 突变体也能正常采样。

**lhs_v1.py 遗留 bug**: 该文件虽内联了新去重函数，但 `generate_lhs_tasks` 中 `real_mutant_full` 变量未构建，直接运行会 `NameError`。其他脚本均已正确构建。

### 11.4 死锁问题

数据生成管线存在两类死锁，需分别修复：

#### 11.4.1 BLAS 线程死锁

**症状**: `ProcessPoolExecutor` + `fork` 启动子进程时，OpenBLAS 线程池的锁状态不一致导致所有进程 0% CPU 死锁（`wchan=futex_wait_queue`）。

**修复**: 在 `run_lhs_batch.sh` 中设置 `OMP_NUM_THREADS=1` 等环境变量，禁用 OpenBLAS 多线程。

#### 11.4.2 ODE BDF 求解器死锁

**症状**: 病态 LHS 参数导致 `solve_ivp(method='BDF')` 在 Fortran/C 层陷入步长无限缩减循环。原 `eqns_with_timeout` 的 `time.time()` 检查在 BDF 卡于 Fortran 层时**永不执行**（Python 字节码无法调度），`future.result(timeout=)` 也只计数不杀 worker，最终 16 个 worker 全部耗尽，`as_completed` 永久阻塞。

**修复**: 在 10 个 lhs 脚本（`lhs_v0.py`, `lhs_v1.py`, `lhs_v1_origin.py`, `lhs_v1_origin_no_filter.py`, `lhs_v1_origin_guaranteed.py`, `lhs_v1_origin_v2.py` ~ `lhs_v1_origin_v6.py`）的 `simulate_mutant` 函数内引入 SIGALRM 硬超时兜底：

```python
def _sigalrm_handler(signum, frame):
    raise TimeoutError("ODE BDF Timeout (SIGALRM)")
_old_handler = signal.signal(signal.SIGALRM, _sigalrm_handler)
signal.alarm(max(1, int(SIM_TIMEOUT)))
try:
    while t0 < tf and loop_count < 300:
        sol = solve_ivp(...)  # BDF 求解
        ...
finally:
    signal.alarm(0)                          # 必须取消 alarm，否则误杀 worker 复用的下一个 task
    signal.signal(signal.SIGALRM, _old_handler)  # 恢复默认 handler，避免污染 worker 后续 task
```

**关键点**:
- SIGALRM 是 Unix 信号，能中断 Fortran/C 层的系统调用与步长缩减循环，而 Python 层的 `time.time()` 检查做不到
- `signal.alarm(0)` 与恢复 `_old_handler` 必须放在 `finally`，保证 `ProcessPoolExecutor` worker 复用时不被污染
- `test_lhs_filter_analysis.py` 无需修改：它 `from lhs_v1_origin import simulate_mutant`，已通过 import 间接获得保护（fork 时子进程继承已修复的模块）

**影响评估**:
- 正常样本（0.1–3 秒完成）：零影响，SIGALRM 不会触发，`finally` 取消闹钟
- 病态样本（>5 秒）：从"死锁卡死整个进程池"变为"5 秒后丢弃，worker 释放继续下一个 task"。这些样本本就是参数空间病态点（BDF 步长无限缩减），即使强行算出轨迹也不可信，丢弃是正确行为
- 数据质量：只会提升不会下降

### 11.5 Sim-to-Real Gap 缓解策略

1. **数据层**（V2/V3）：定向扩展覆盖 Real Mutant 边界
2. **算法层**（mutant_aware AL）：基于 atlas 优先级加权采样
3. **联合方案**（V3 + mutant_aware AL）：数据层补盲 + 算法层定向

### 11.6 采样规模权衡

- 单突变体 400 样本：基础覆盖，适合未采样家族的初步扩展
- 单突变体 500 样本：基础扩展，适合一般残留瓶颈
- 单突变体 800 样本：精准修复，适合邻域污染修复与瓶颈突变体
- 单突变体 1000 样本（NUM_LHS_SAMPLES）：全局基线覆盖
- 总规模上限：~170,000（V5，含 124 Real + 121K LHS + 54K 定向），单次生成耗时 4-5 小时

---

## 十二、待办与未来改进方向

- [x] 生成 V3-V5 数据集并完成综合分析
- [x] 运行 V3/V4 eval 实验并完成分析报告
- [x] 修复 `real_vs_lhs_csv.py` 正则 bug
- [x] 修复 BLAS 死锁问题（`run_lhs_batch.sh` 添加环境变量）
- [x] 完整记录 10 个数据生成脚本的去重防泄漏逻辑（本文档第三章）
- [x] 修复 ODE BDF 求解器死锁 — SIGALRM 硬超时兜底（2026-07-17，详见 11.4.2 节）
- [x] 统一去重算法：V1/V1_origin/guaranteed/no_filter 升级为 rtol=0.08 + 全局跨检查（2026-07-14）
- [ ] 修复 `lhs_v1.py` 的 `real_mutant_full` 未构建 bug（添加 `real_mutant_full = build_real_mutant_full(mutant_rules, defaults)`）
- [ ] 更新 `test_lhs_filter_analysis.py` 去重逻辑以匹配新算法（rtol=0.08 + 全局跨检查）
- [ ] 评估是否将 `candidate_count` 从 10x 提升至 15x（project_memory 要求）
- [ ] 运行 V5+none / V5+mutant_aware 实验
- [ ] 运行 V6 数据集生成并评估
- [ ] 评估是否需要修复 `metrics.py` 的 Rel_L2 数值爆炸问题（CDH1T=0 导致真值方差接近 0）

---

**文档结束** | 最后更新: 2026-07-17 | 维护人: zsy
