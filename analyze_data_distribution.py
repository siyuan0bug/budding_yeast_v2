"""
数据分布特征全面分析脚本
分析内容：
  1. 数据集总体概况
  2. 跨样本分布特征（LHS vs Real, Pattern A/B/C）
  3. 跨变量分布特征（39个轨迹变量的统计量、偏度、峰度、值域）
  4. 跨参数分布特征（141个参数的统计量、离散性、固定值比例）
  5. 变量间相关性分析
  6. 时间维度特征分析
  7. 归一化方法适用性分析
"""

import numpy as np
import json
import time
from scipy import stats as scipy_stats

DATASET_PATH = "/home/users/zsy/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels.npz"

VARIABLE_NAMES = [
    "MASS", "CLN2", "CLB2", "CLB5", "SIC1", "CDC6", "C2", "C5", "F2", "F5",
    "SIC1P", "C2P", "C5P", "CDC6P", "F2P", "F5P", "SWI5T", "SWI5", "IEP", "CDC20T",
    "CDC20", "CDH1T", "CDH1", "CDC14T", "CDC14", "NET1T", "NET1", "RENT", "TEM1", "CDC15",
    "PPX", "PDS1", "ESP1", "ORI", "BUD", "SPN", "Vi20", "lte1", "BUB2"
]

EPS = 1e-8

print("=" * 80)
print("出芽酵母细胞周期模拟数据分布特征全面分析")
print("=" * 80)
print(f"数据集路径: {DATASET_PATH}")
print(f"加载时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================
# 1. 加载数据
# ============================================================
print("正在加载数据集...")
t0 = time.time()
dataset = np.load(DATASET_PATH, allow_pickle=True)
data = dataset['data']              # (N, 39, 500)
params = dataset['params']          # (N, 141)
names = dataset.get('mutant_names', [f"Sample_{i:05d}" for i in range(data.shape[0])])
pattern_labels = dataset['pattern_labels']
load_time = time.time() - t0
print(f"加载完成! 耗时: {load_time:.1f}s")

N, V, T = data.shape
P = params.shape[1]
print(f"样本数 N={N}, 变量数 V={V}, 时间步 T={T}, 参数数 P={P}")
print()

# 清洗
data = np.nan_to_num(data, nan=0.0, posinf=1e4, neginf=0.0)
params = np.nan_to_num(params, nan=0.0, posinf=1e4, neginf=0.0)

# 划分 LHS / Real
lhs_indices = [i for i, n in enumerate(names) if '_LHS_' in str(n)]
real_indices = [i for i, n in enumerate(names) if '_LHS_' not in str(n)]
print(f"LHS 样本: {len(lhs_indices)} | Real 样本: {len(real_indices)}")

# Pattern 分布
pattern_counts = {}
for i in range(N):
    pat = str(pattern_labels[i]).split(':')[0]
    pattern_counts[pat] = pattern_counts.get(pat, 0) + 1
print(f"Pattern 分布: {pattern_counts}")
print()

# ============================================================
# 2. 跨样本分布特征
# ============================================================
print("=" * 80)
print("2. 跨样本分布特征分析")
print("=" * 80)

# 2.1 LHS vs Real 的样本级统计
lhs_data = data[lhs_indices]   # (N_lhs, 39, 500)
real_data = data[real_indices] # (N_real, 39, 500)

# 每个样本的全局统计（跨所有变量和时间步）
lhs_sample_means = lhs_data.mean(axis=(1, 2))
real_sample_means = real_data.mean(axis=(1, 2))
lhs_sample_stds = lhs_data.std(axis=(1, 2))
real_sample_stds = real_data.std(axis=(1, 2))
lhs_sample_max = lhs_data.max(axis=(1, 2))
real_sample_max = real_data.max(axis=(1, 2))

print("\n2.1 样本级全局统计 (跨所有39变量×500时间步)")
print(f"{'指标':<20} {'LHS 均值':<12} {'LHS 标准差':<12} {'LHS 最小':<12} {'LHS 最大':<12} {'Real 均值':<12} {'Real 标准差':<12} {'Real 最小':<12} {'Real 最大':<12}")
for label, lhs_vals, real_vals in [
    ("样本均值", lhs_sample_means, real_sample_means),
    ("样本标准差", lhs_sample_stds, real_sample_stds),
    ("样本最大值", lhs_sample_max, real_sample_max),
]:
    print(f"{label:<20} {np.mean(lhs_vals):<12.4f} {np.std(lhs_vals):<12.4f} {np.min(lhs_vals):<12.4f} {np.max(lhs_vals):<12.4f} {np.mean(real_vals):<12.4f} {np.std(real_vals):<12.4f} {np.min(real_vals):<12.4f} {np.max(real_vals):<12.4f}")

# 2.2 Pattern 分组的样本级统计
print("\n2.2 按 Pattern 分组的样本级统计")
for pat in ['Pattern_A', 'Pattern_B', 'Pattern_C']:
    pat_mask = np.array([str(pattern_labels[i]).startswith(pat) for i in range(N)])
    pat_data = data[pat_mask]
    if len(pat_data) == 0:
        continue
    pat_means = pat_data.mean(axis=(1, 2))
    pat_stds = pat_data.std(axis=(1, 2))
    pat_max = pat_data.max(axis=(1, 2))
    print(f"\n  {pat} (n={len(pat_data)}):")
    print(f"    样本均值: mean={np.mean(pat_means):.4f}, std={np.std(pat_means):.4f}, range=[{np.min(pat_means):.4f}, {np.max(pat_means):.4f}]")
    print(f"    样本标准差: mean={np.mean(pat_stds):.4f}, std={np.std(pat_stds):.4f}, range=[{np.min(pat_stds):.4f}, {np.max(pat_stds):.4f}]")
    print(f"    样本最大值: mean={np.mean(pat_max):.4f}, std={np.std(pat_max):.4f}, range=[{np.min(pat_max):.4f}, {np.max(pat_max):.4f}]")

# 2.3 LHS vs Real 分布差异 (KS检验)
print("\n2.3 LHS vs Real 分布差异 (KS检验)")
lhs_flat = lhs_data.flatten()
real_flat = real_data.flatten()
ks_stat, ks_p = scipy_stats.ks_2samp(lhs_flat[::100], real_flat[::100])  # 采样加速
print(f"  KS 统计量: {ks_stat:.6f}, p-value: {ks_p:.2e}")
print(f"  LHS 全局: mean={np.mean(lhs_flat):.4f}, std={np.std(lhs_flat):.4f}, median={np.median(lhs_flat):.4f}")
print(f"  Real 全局: mean={np.mean(real_flat):.4f}, std={np.std(real_flat):.4f}, median={np.median(real_flat):.4f}")
print()

# ============================================================
# 3. 跨变量分布特征
# ============================================================
print("=" * 80)
print("3. 跨变量分布特征分析 (39个轨迹变量)")
print("=" * 80)

# 使用 LHS 数据计算变量统计量
lhs_data_clamped = np.clip(lhs_data, 0, None)

var_stats = []
print(f"\n{'ID':<4} {'变量名':<10} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12} {'中位数':<12} {'偏度':<10} {'峰度':<10} {'值域跨度':<12} {'CV':<10}")
print("-" * 130)

for v in range(V):
    var_vals = lhs_data_clamped[:, v, :].flatten()
    mean_v = np.mean(var_vals)
    std_v = np.std(var_vals)
    min_v = np.min(var_vals)
    max_v = np.max(var_vals)
    median_v = np.median(var_vals)
    skew_v = scipy_stats.skew(var_vals)
    kurt_v = scipy_stats.kurtosis(var_vals)
    range_ratio = max_v / (min_v + EPS) if min_v > EPS else float('inf')
    cv = std_v / (mean_v + EPS) if mean_v > EPS else float('inf')

    var_stats.append({
        'id': v, 'name': VARIABLE_NAMES[v],
        'mean': mean_v, 'std': std_v, 'min': min_v, 'max': max_v,
        'median': median_v, 'skew': skew_v, 'kurtosis': kurt_v,
        'range_ratio': range_ratio, 'cv': cv
    })

    rr_str = f"{range_ratio:.1f}x" if range_ratio < 1e6 else f"{range_ratio:.2e}"
    print(f"{v:<4} {VARIABLE_NAMES[v]:<10} {mean_v:<12.4f} {std_v:<12.4f} {min_v:<12.4f} {max_v:<12.4f} {median_v:<12.4f} {skew_v:<10.2f} {kurt_v:<10.2f} {rr_str:<12} {cv:<10.2f}")

# 3.1 变量分类
print("\n3.1 变量分类统计")
high_skew = [s for s in var_stats if abs(s['skew']) > 2]
low_skew = [s for s in var_stats if abs(s['skew']) <= 2]
high_kurt = [s for s in var_stats if abs(s['kurtosis']) > 10]
large_range = [s for s in var_stats if s['range_ratio'] > 100]
small_range = [s for s in var_stats if s['range_ratio'] <= 100]

print(f"  高偏度 (|skew|>2): {len(high_skew)} 个变量")
for s in high_skew:
    print(f"    {s['name']}: skew={s['skew']:.2f}, range={s['range_ratio']:.1f}x")
print(f"  低偏度 (|skew|<=2): {len(low_skew)} 个变量")
print(f"  高峰度 (|kurt|>10): {len(high_kurt)} 个变量")
for s in high_kurt:
    print(f"    {s['name']}: kurt={s['kurtosis']:.2f}, skew={s['skew']:.2f}")
print(f"  大值域 (range>100x): {len(large_range)} 个变量")
print(f"  小值域 (range<=100x): {len(small_range)} 个变量")

# 3.2 变量值域分布
print("\n3.2 变量值域分布")
mean_values = [s['mean'] for s in var_stats]
std_values = [s['std'] for s in var_stats]
print(f"  变量均值范围: [{min(mean_values):.6f}, {max(mean_values):.4f}]")
print(f"  变量均值中位数: {np.median(mean_values):.4f}")
print(f"  变量标准差范围: [{min(std_values):.6f}, {max(std_values):.4f}]")
print(f"  变量标准差中位数: {np.median(std_values):.4f}")
print(f"  均值最大/最小比: {max(mean_values)/(min(mean_values)+EPS):.1f}x")
print(f"  标准差最大/最小比: {max(std_values)/(min(std_values)+EPS):.1f}x")

# 3.3 变量方差比（归一化重要性的关键指标）
print("\n3.3 变量方差比（决定 Z-score 归一化下 loss 权重）")
var_vars = [s['std']**2 for s in var_stats]
var_vars_sorted = sorted(var_vars, reverse=True)
print(f"  方差最大: {var_vars_sorted[0]:.4f} ({VARIABLE_NAMES[var_stats[np.argmax([s['std']**2 for s in var_stats])]['id']]})")
print(f"  方差最小: {var_vars_sorted[-1]:.4f} ({VARIABLE_NAMES[var_stats[np.argmin([s['std']**2 for s in var_stats])]['id']]})")
print(f"  方差比 (max/min): {var_vars_sorted[0]/(var_vars_sorted[-1]+EPS):.1f}x")
print(f"  方差中位数: {np.median(var_vars):.4f}")
print(f"  方差变异系数 (CV): {np.std(var_vars)/np.mean(var_vars):.2f}")
print()

# ============================================================
# 4. 跨参数分布特征
# ============================================================
print("=" * 80)
print("4. 跨参数分布特征分析 (141个参数)")
print("=" * 80)

lhs_params = params[lhs_indices]

param_stats = []
n_fixed = 0
n_bool = 0
n_discrete = 0
n_continuous = 0

print(f"\n{'ID':<5} {'均值':<14} {'标准差':<14} {'最小值':<14} {'最大值':<14} {'CV':<10} {'类型':<12}")
print("-" * 90)

for p in range(P):
    p_vals = lhs_params[:, p]
    mean_p = np.mean(p_vals)
    std_p = np.std(p_vals)
    min_p = np.min(p_vals)
    max_p = np.max(p_vals)
    cv = std_p / (abs(mean_p) + EPS) if abs(mean_p) > EPS else 0

    # 判断参数类型
    unique_vals = np.unique(p_vals)
    if len(unique_vals) <= 1:
        ptype = "固定值"
        n_fixed += 1
    elif len(unique_vals) == 2:
        ptype = "布尔型"
        n_bool += 1
    elif len(unique_vals) <= 10:
        ptype = "离散型"
        n_discrete += 1
    else:
        ptype = "连续型"
        n_continuous += 1

    param_stats.append({
        'id': p, 'mean': mean_p, 'std': std_p, 'min': min_p, 'max': max_p,
        'cv': cv, 'type': ptype, 'n_unique': len(unique_vals)
    })

    if p < 30 or ptype != "固定值":  # 只打印前30个或非固定值
        print(f"{p:<5} {mean_p:<14.6f} {std_p:<14.6f} {min_p:<14.6f} {max_p:<14.6f} {cv:<10.4f} {ptype:<12}")

print(f"\n4.1 参数类型统计:")
print(f"  固定值 (1个唯一值): {n_fixed} 个 ({n_fixed/P*100:.1f}%)")
print(f"  布尔型 (2个唯一值): {n_bool} 个 ({n_bool/P*100:.1f}%)")
print(f"  离散型 (3-10个唯一值): {n_discrete} 个 ({n_discrete/P*100:.1f}%)")
print(f"  连续型 (>10个唯一值): {n_continuous} 个 ({n_continuous/P*100:.1f}%)")

# 4.2 非固定参数的分布
non_fixed = [s for s in param_stats if s['type'] != '固定值']
if non_fixed:
    print(f"\n4.2 非固定参数分布 (n={len(non_fixed)}):")
    cvs = [s['cv'] for s in non_fixed]
    print(f"  CV 范围: [{min(cvs):.4f}, {max(cvs):.4f}]")
    print(f"  CV 中位数: {np.median(cvs):.4f}")
    print(f"  CV 均值: {np.mean(cvs):.4f}")

    # 高变异参数
    high_cv = sorted(non_fixed, key=lambda x: x['cv'], reverse=True)[:10]
    print(f"\n  变异系数最高的10个参数:")
    for s in high_cv:
        print(f"    Param[{s['id']}]: CV={s['cv']:.4f}, mean={s['mean']:.6f}, range=[{s['min']:.6f}, {s['max']:.6f}]")

# 4.3 参数分布形态
print(f"\n4.3 参数分布形态 (非固定参数)")
for s in non_fixed[:20]:
    p_vals = lhs_params[:, s['id']]
    skew_p = scipy_stats.skew(p_vals)
    kurt_p = scipy_stats.kurtosis(p_vals)
    print(f"  Param[{s['id']}]: skew={skew_p:.2f}, kurt={kurt_p:.2f}, n_unique={s['n_unique']}")
print()

# ============================================================
# 5. 变量间相关性分析
# ============================================================
print("=" * 80)
print("5. 变量间相关性分析")
print("=" * 80)

# 计算变量间 Pearson 相关系数（使用时间均值）
print("\n5.1 变量间 Pearson 相关系数 (基于时间均值)")
lhs_var_means = lhs_data.mean(axis=2)  # (N_lhs, 39)
corr_matrix = np.corrcoef(lhs_var_means.T)  # (39, 39)

# 找出高相关变量对
high_corr_pairs = []
for i in range(V):
    for j in range(i+1, V):
        if abs(corr_matrix[i, j]) > 0.8:
            high_corr_pairs.append((i, j, corr_matrix[i, j]))

high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print(f"  高相关变量对 (|r|>0.8): {len(high_corr_pairs)} 对")
for i, j, r in high_corr_pairs[:15]:
    print(f"    {VARIABLE_NAMES[i]} <-> {VARIABLE_NAMES[j]}: r={r:.4f}")

# 找出低相关变量对
low_corr_pairs = []
for i in range(V):
    for j in range(i+1, V):
        if abs(corr_matrix[i, j]) < 0.1:
            low_corr_pairs.append((i, j, corr_matrix[i, j]))
print(f"  低相关变量对 (|r|<0.1): {len(low_corr_pairs)} 对")

# 相关性矩阵的整体特征
off_diag = corr_matrix[np.triu_indices(V, k=1)]
print(f"\n5.2 相关性矩阵整体特征")
print(f"  相关系数均值: {np.mean(off_diag):.4f}")
print(f"  相关系数标准差: {np.std(off_diag):.4f}")
print(f"  正相关对数 (r>0.3): {np.sum(off_diag > 0.3)}")
print(f"  负相关对数 (r<-0.3): {np.sum(off_diag < -0.3)}")
print()

# ============================================================
# 6. 时间维度特征分析
# ============================================================
print("=" * 80)
print("6. 时间维度特征分析")
print("=" * 80)

# 6.1 变量的时间变化模式
print("\n6.1 变量时间变化模式")
print(f"{'ID':<4} {'变量名':<10} {'t=0均值':<12} {'t=T/2均值':<12} {'t=T均值':<12} {'时间标准差':<12} {'振荡频率':<12} {'类型':<10}")
print("-" * 100)

for v in range(V):
    var_traj = lhs_data[:, v, :]  # (N_lhs, T)
    t0_mean = var_traj[:, 0].mean()
    tmid_mean = var_traj[:, T//2].mean()
    tend_mean = var_traj[:, -1].mean()

    # 时间维度的标准差（跨样本平均）
    time_std = var_traj.std(axis=0).mean()

    # 振荡频率：计算平均轨迹的过零次数
    mean_traj = var_traj.mean(axis=0)
    centered = mean_traj - mean_traj.mean()
    zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)

    # 判断类型
    if time_std < 0.01 * abs(mean_traj.mean()):
        vtype = "平稳"
    elif zero_crossings > 20:
        vtype = "振荡"
    elif abs(tend_mean - t0_mean) > 2 * time_std:
        vtype = "趋势"
    else:
        vtype = "波动"

    print(f"{v:<4} {VARIABLE_NAMES[v]:<10} {t0_mean:<12.4f} {tmid_mean:<12.4f} {tend_mean:<12.4f} {time_std:<12.4f} {zero_crossings:<12} {vtype:<10}")

# 6.2 时间自相关
print("\n6.2 时间自相关分析 (lag=1, 10, 50)")
for v in [0, 2, 4, 9, 19, 22, 33]:  # 代表性变量
    var_traj = lhs_data[:, v, :]  # (N_lhs, T)
    mean_traj = var_traj.mean(axis=0)
    centered = mean_traj - mean_traj.mean()

    for lag in [1, 10, 50]:
        if len(centered) > lag:
            ac = np.corrcoef(centered[:-lag], centered[lag:])[0, 1]
            print(f"  {VARIABLE_NAMES[v]} (lag={lag}): autocorr={ac:.4f}")
    print()

# ============================================================
# 7. 归一化方法适用性分析
# ============================================================
print("=" * 80)
print("7. 归一化方法适用性分析")
print("=" * 80)

# 7.1 Z-score 归一化后的变量方差
print("\n7.1 Z-score 归一化后的变量方差 (应接近1)")
zscore_vars = []
for v in range(V):
    var_vals = lhs_data_clamped[:, v, :].flatten()
    mu = np.mean(var_vals)
    sigma = np.std(var_vals) + EPS
    normed = (var_vals - mu) / sigma
    zscore_vars.append(np.var(normed))

print(f"  方差范围: [{min(zscore_vars):.6f}, {max(zscore_vars):.6f}]")
print(f"  方差均值: {np.mean(zscore_vars):.6f}")
print(f"  方差标准差: {np.std(zscore_vars):.6f}")
print(f"  → Z-score 后所有变量方差≈1，loss 权重均衡")

# 7.2 Z-score 归一化后的值域
print("\n7.2 Z-score 归一化后的值域")
zscore_ranges = []
for v in range(V):
    var_vals = lhs_data_clamped[:, v, :].flatten()
    mu = np.mean(var_vals)
    sigma = np.std(var_vals) + EPS
    normed = (var_vals - mu) / sigma
    zscore_ranges.append((np.min(normed), np.max(normed), np.percentile(np.abs(normed), 99)))

print(f"{'ID':<4} {'变量名':<10} {'zscore_min':<12} {'zscore_max':<12} {'99%|z|':<12} {'是否需clamp':<10}")
for v in range(V):
    zmin, zmax, p99 = zscore_ranges[v]
    need_clamp = "是" if (zmax > 10 or zmin < -10) else "否"
    print(f"{v:<4} {VARIABLE_NAMES[v]:<10} {zmin:<12.2f} {zmax:<12.2f} {p99:<12.2f} {need_clamp:<10}")

# 7.3 Z-score vs Log+MM 的 loss 权重对比
print("\n7.3 Z-score vs Log+MM 的 loss 权重对比 (Spearman相关)")
# Z-score 的隐式权重 = 原始方差 σ²
zscore_weights = np.array([s['std']**2 for s in var_stats])
# Log+MM 的隐式权重 = 归一化后方差
logmm_weights = []
for v in range(V):
    var_vals = lhs_data_clamped[:, v, :].flatten()
    log_vals = np.log(var_vals + EPS)
    log_min = np.min(log_vals)
    log_max = np.max(log_vals)
    normed = (log_vals - log_min) / (log_max - log_min + EPS)
    logmm_weights.append(np.var(normed))
logmm_weights = np.array(logmm_weights)

# 评估权重 = 原始空间 MAE 贡献 (用均值近似)
eval_weights = np.array([s['mean'] for s in var_stats])

# Spearman 相关
rho_zscore, _ = scipy_stats.spearmanr(zscore_weights, eval_weights)
rho_logmm, _ = scipy_stats.spearmanr(logmm_weights, eval_weights)

print(f"  Z-score 权重 vs 评估权重 Spearman: {rho_zscore:.4f}")
print(f"  Log+MM 权重 vs 评估权重 Spearman: {rho_logmm:.4f}")
print(f"  → Z-score 的 loss 权重与评估权重{'正' if rho_zscore > 0 else '负'}相关")
print(f"  → Log+MM 的 loss 权重与评估权重{'正' if rho_logmm > 0 else '负'}相关")
print()

# ============================================================
# 8. 跨 Mutant 分布稳定性分析
# ============================================================
print("=" * 80)
print("8. 跨 Mutant 分布稳定性分析")
print("=" * 80)

# 按突变体家族分组
family_groups = {}
for idx in lhs_indices:
    name = str(names[idx])
    family = name.split('_LHS_')[0] if '_LHS_' in name else name
    family_groups.setdefault(family, []).append(idx)

print(f"\n8.1 突变体家族数: {len(family_groups)}")
print(f"  每家族样本数: min={min(len(v) for v in family_groups.values())}, max={max(len(v) for v in family_groups.values())}, median={np.median([len(v) for v in family_groups.values()]):.0f}")

# 8.2 跨家族的变量统计量稳定性
print(f"\n8.2 跨家族的变量统计量稳定性 (变异系数 CV)")
print(f"{'ID':<4} {'变量名':<10} {'均值CV':<12} {'标准差CV':<12} {'范围比':<12} {'稳定性':<10}")
print("-" * 70)

for v in range(V):
    family_means = []
    family_stds = []
    family_ranges = []

    for fam, indices in family_groups.items():
        if len(indices) < 5:
            continue
        fam_data = data[indices, v, :].flatten()
        family_means.append(np.mean(fam_data))
        family_stds.append(np.std(fam_data))
        family_ranges.append(np.max(fam_data) - np.min(fam_data))

    if len(family_means) < 3:
        continue

    mean_cv = np.std(family_means) / (np.mean(family_means) + EPS)
    std_cv = np.std(family_stds) / (np.mean(family_stds) + EPS)
    range_ratio = max(family_ranges) / (min(family_ranges) + EPS)

    stability = "高" if mean_cv < 0.3 else ("中" if mean_cv < 0.7 else "低")
    print(f"{v:<4} {VARIABLE_NAMES[v]:<10} {mean_cv:<12.4f} {std_cv:<12.4f} {range_ratio:<12.1f} {stability:<10}")

print()
print("=" * 80)
print("分析完成!")
print("=" * 80)
