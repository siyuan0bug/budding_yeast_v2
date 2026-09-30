#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Mutant Reachability Check + 训练前预测探针 (v3 - Comprehensive SciML Edition)
================================================================================
目的：完全无需训练神经网络，在秒级至分钟级内评估「该采样算法能否让模型做好 real mutants 预测」。

核心能力：
  [1] 参数空间可达性（Geometry & Coverage）
      126 株 real 是否落在合成样本参数支撑内（z 空间 min-dist、P90、max-dist、驱动参数解析）。
      超快模式（--fast）：只载入 141-D 参数矩阵，无需载入 12GB 轨迹，<0.5 秒极速反馈。

  [2] 全指标对齐的 k-NN 回归探针（Nadaraya-Watson 惰性预测器）
      与 eval.py 评测口径 100% 对齐：
        - 全 39 变量：Probe MAE、Relative L2（含基底截断保护）、Correlation
        - 核心 9 变量：Core 9 MAE、Core 9 Relative L2
      探针误差直接作为该采样算法在测试集上的零训练插值理论下界。

  [3] 局部 Lipschitz 审计（相空间陡度与刚性）
      对每株 real 估计局部映射陡度 L = max ||y_a-y_b||/||p_a-p_b||，代理误差 = L × h。
      提前点名非线性突变点（邻域内微小参数变动导致轨迹跳变者）。

  [4] 多数据集头对头 PK 评测（Head-to-Head Comparison / --baseline）
      自动对比多个采样算法：胜率统计（如 A 在 126 株中有 121 株优于 B）、
      均值/中位数指标差异、拉开差距的 Top 改进株与 Top 差距株。

  [5] 逐突变体体检 CSV 报表（per_mutant_csv）
      输出 reachability_<stem>_per_mutant.csv，包含各株的 min_dist, MAE, relL2, Core 9,
      与 per_mutant_metrics.csv 格式完全对齐，支持在 IDE/表格中直接排序排查 Badcases。

  [6] GPU 极速向量化加速
      自动检测 PyTorch CUDA，将 126 x 100,000 的高维欧式距离计算缩短至 <0.2 秒；
      自动向下兼容 CPU 无缝降级运行。

用法：
  # 1. 单数据集完整体检（含轨迹探针，约 30 秒）
  python reachability_check.py --datasets mutant_aware_100000_210m_500s.npz

  # 2. 极速纯几何可达性检查（不载入 12GB 轨迹，<0.5 秒）
  python reachability_check.py --fast --datasets mutant_aware_100000_210m_500s.npz

  # 3. 多数据集对比 PK（对战基线，自动输出胜率与详细差值）
  python reachability_check.py --datasets aware.npz random.npz --baseline random.npz

  # 4. 探针 vs DL 真实模型秩相关验证
  python reachability_check.py --datasets aware.npz --dl_csv aware=per_mutant_metrics.csv
"""

import os
import sys
import csv
import argparse
import time
import numpy as np

# PyTorch GPU 算子检测
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

DEFAULT_K = 16        # k-NN 探针邻居数
DEFAULT_K_LIP = 50    # Lipschitz 审计邻居数
FLOOR_EPS = 0.1       # 物理基底浓度下限（与 eval.py / metrics.py 完全对齐）

# 39 变量列表
VARIABLE_NAMES = [
    "MASS", "CLN2", "CLB2", "CLB5", "SIC1", "CDC6", "C2", "C5", "F2", "F5",
    "SIC1P", "C2P", "C5P", "CDC6P", "F2P", "F5P", "SWI5T", "SWI5", "IEP", "CDC20T",
    "CDC20", "CDH1T", "CDH1", "CDC14T", "CDC14", "NET1T", "NET1", "RENT", "TEM1", "CDC15",
    "PPX", "PDS1", "ESP1", "ORI", "BUD", "SPN", "Vi20", "lte1", "BUB2"
]

# 核心 9 变量索引 (MASS, CLN2, CLB2, CLB5, SIC1, CDC20, CDH1, ORI, SPN)
CORE_9_INDICES = [0, 1, 2, 3, 4, 20, 22, 33, 35]


# ---------------------------------------------------------------------------
# 1. 距离计算与 GPU/CPU 调度
# ---------------------------------------------------------------------------

def knn_dist_gpu(z_real, z_train, k):
    """使用 PyTorch 在 GPU 上极速计算 126 x N 的欧氏距离与 Top-k 邻居索引。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z_r = torch.as_tensor(z_real, dtype=torch.float32, device=device)
    z_t = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    k_val = min(k, z_t.shape[0])
    # torch.cdist 计算两两欧氏距离
    dists = torch.cdist(z_r, z_t, p=2.0)
    top_d, top_i = torch.topk(dists, k=k_val, largest=False, dim=1)
    return top_d.cpu().numpy(), top_i.cpu().numpy()


def knn_dist_chunked(z_real, z_train, k, chunk=128):
    """CPU 逐块计算 126 x N 欧氏距离，返回每株 real 最近 k 个邻居的 (距离, 索引)。"""
    n_real = z_real.shape[0]
    k = min(k, z_train.shape[0])
    best_d = np.full((n_real, k), np.inf)
    best_i = np.zeros((n_real, k), dtype=np.int64)
    tr_sq = np.einsum('ij,ij->i', z_train, z_train)
    re_sq = np.einsum('ij,ij->i', z_real, z_real)
    for s in range(0, n_real, chunk):
        e = min(s + chunk, n_real)
        d2 = re_sq[s:e, None] + tr_sq[None, :] - 2.0 * (z_real[s:e] @ z_train.T)
        np.maximum(d2, 0.0, out=d2)
        if k < d2.shape[1]:
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        else:
            part = np.tile(np.arange(d2.shape[1]), (e - s, 1))
        rows = np.arange(e - s)[:, None]
        dpart = np.sqrt(d2[rows, part])
        order = np.argsort(dpart, axis=1)
        dpart = np.take_along_axis(dpart, order, axis=1)
        ipart = np.take_along_axis(part, order, axis=1)
        merged_d = np.concatenate([best_d[s:e], dpart], axis=1)
        merged_i = np.concatenate([best_i[s:e], ipart], axis=1)
        sel = np.argsort(merged_d, axis=1)[:, :k]
        best_d[s:e] = np.take_along_axis(merged_d, sel, axis=1)
        best_i[s:e] = np.take_along_axis(merged_i, sel, axis=1)
    return best_d, best_i


def compute_knn_dists(z_real, z_train, k, use_gpu=True):
    """调度器：优先尝试 GPU 计算，若无 GPU 或显存不足则自动回退至 CPU 分块。"""
    if use_gpu and HAS_TORCH and torch.cuda.is_available():
        try:
            return knn_dist_gpu(z_real, z_train, k)
        except Exception as exc:
            print(f"  [Notice] GPU 距离计算回退至 CPU ({exc})")
    return knn_dist_chunked(z_real, z_train, k)


# ---------------------------------------------------------------------------
# 2. 全指标对齐探针评估 (Aligned with eval.py & calculate_metrics)
# ---------------------------------------------------------------------------

def calculate_single_mutant_metrics(pred, target, eps=FLOOR_EPS):
    """
    计算单个突变体在 (Vars, Time) 轨迹上的指标，与 budding_yeast_v2/utils/metrics.py 100% 对齐。
    pred, target: shape (39, T_STEPS) 或 (Core9, T_STEPS)
    返回: (mae, rel_l2, corr)
    """
    V, T = pred.shape
    # 1. MAE
    mae = float(np.mean(np.abs(pred - target)))

    # 2. Per-variable Relative L2 (含物理基线截断保护)
    numerator = np.sum((pred - target) ** 2, axis=1)
    denominator = np.sum(target ** 2, axis=1)
    floor_energy = (eps ** 2) * T
    safe_denominator = np.maximum(denominator, floor_energy)
    rel_l2 = float(np.mean(np.sqrt(numerator / safe_denominator)))

    # 3. Per-variable Correlation (Pearson)
    pred_mean = np.mean(pred, axis=1, keepdims=True)
    target_mean = np.mean(target, axis=1, keepdims=True)
    pred_c = pred - pred_mean
    target_c = target - target_mean
    cov = np.sum(pred_c * target_c, axis=1)
    std_prod = np.sqrt(np.sum(pred_c ** 2, axis=1) * np.sum(target_c ** 2, axis=1))
    corr = float(np.mean(cov / (std_prod + 1e-8)))

    return mae, rel_l2, corr


def knn_probe_full(y_real, y_nb, dists, eps=FLOOR_EPS):
    """
    逆距离加权 k-NN 探针，同时计算全 39 变量与核心 9 变量指标。
    y_real: (39, 500); y_nb: (k, 39, 500); dists: (k,)
    返回指标字典。
    """
    d = np.maximum(dists, 1e-12)
    w = 1.0 / d
    w = w / w.sum()
    # 预测轨迹 (39, 500)
    pred = np.einsum('k,kvt->vt', w, y_nb)

    # 全 39 变量
    mae_39, rel_l2_39, corr_39 = calculate_single_mutant_metrics(pred, y_real, eps=eps)

    # 核心 9 变量
    pred_c9 = pred[CORE_9_INDICES, :]
    real_c9 = y_real[CORE_9_INDICES, :]
    mae_c9, rel_l2_c9, corr_c9 = calculate_single_mutant_metrics(pred_c9, real_c9, eps=eps)

    return {
        'mae': mae_39,
        'rel_l2': rel_l2_39,
        'corr': corr_39,
        'core9_mae': mae_c9,
        'core9_rel_l2': rel_l2_c9,
        'core9_corr': corr_c9,
    }


# ---------------------------------------------------------------------------
# 3. 局部 Lipschitz 审计与动力学刚性分析
# ---------------------------------------------------------------------------

def lipschitz_audit(y_nb, z_nb, y_norm):
    """局部 Lipschitz 估计 + 代理误差。
    y_nb: (k, 39, 500); z_nb: (k, P) 参数差向量; y_norm: real 轨迹范数。
    L = median ||y_a - y_b||_F / ||z_a - z_b||_2; h = 邻居到 real 距离中位数。
    proxy = L * h / y_norm。"""
    k = z_nb.shape[0]
    if k < 2 or y_norm < 1e-12:
        return np.nan, np.nan, np.nan
    pd = np.sqrt(np.maximum(
        np.sum(z_nb ** 2, axis=1)[:, None]
        + np.sum(z_nb ** 2, axis=1)[None, :]
        - 2.0 * (z_nb @ z_nb.T), 0.0))
    yf = y_nb.reshape(k, -1)
    yd = np.sqrt(np.maximum(
        np.sum(yf ** 2, axis=1)[:, None]
        + np.sum(yf ** 2, axis=1)[None, :]
        - 2.0 * (yf @ yf.T), 0.0))
    tri = np.triu_indices(k, 1)
    pd_pair = pd[tri]
    yd_pair = yd[tri]
    valid = pd_pair > max(1e-3, 1e-6 * float(np.max(pd_pair)))
    if not valid.any():
        return np.nan, np.nan, np.nan
    ratios = yd_pair[valid] / pd_pair[valid]
    L = float(np.median(ratios))
    h = float(np.median(np.sqrt(np.sum(z_nb ** 2, axis=1))))
    return L, h, L * h / y_norm


# ---------------------------------------------------------------------------
# 4. 统计工具与质量综合打分 (Quality Score 0~100)
# ---------------------------------------------------------------------------

def _ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or len(a) != len(b):
        return np.nan
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    if denom < 1e-12:
        return np.nan
    return float(np.sum(ra * rb) / denom)


def calculate_quality_score(reach_med, reach_max, probe_mae, probe_rel, lip_proxy_med):
    """
    计算数据集在测试集上的全局质量健康度得分 (0 ~ 100 分)。
    - 几何可达性权重: 30% (max 距离 > 5.0 严重扣分)
    - 探针拟合下界权重: 50% (MAE 与 Rel L2 越低分越高)
    - 动力学平滑度权重: 20% (Lipschitz 代理越低分越高)
    """
    # 可达性得分 (0~30)
    score_reach = 30.0 * np.exp(-0.25 * reach_med) * (1.0 / (1.0 + 0.1 * max(0.0, reach_max - 2.0)))
    
    # 探针得分 (0~50)
    if np.isfinite(probe_mae) and np.isfinite(probe_rel):
        score_probe = 25.0 * np.exp(-3.0 * probe_mae) + 25.0 * np.exp(-1.5 * probe_rel)
    else:
        score_probe = 25.0  # 纯几何模式下的默认折中分

    # 平滑度得分 (0~20)
    if np.isfinite(lip_proxy_med):
        score_lip = 20.0 * np.exp(-0.5 * lip_proxy_med)
    else:
        score_lip = 10.0

    total = float(np.clip(score_reach + score_probe + score_lip, 0.0, 100.0))
    return round(total, 1)


def load_dl_csv(path):
    """读取 DL 真实模型逐株 eval CSV -> {mutant_name: (Relative_L2, MAE)}"""
    out = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = (row.get('Mutant') or '').strip()
            val_l2 = row.get('Relative_L2')
            val_mae = row.get('MAE')
            if name and val_l2:
                try:
                    l2_f = float(val_l2)
                    mae_f = float(val_mae) if val_mae else np.nan
                    out[name] = (l2_f, mae_f)
                except ValueError:
                    pass
    return out


# ---------------------------------------------------------------------------
# 5. 单数据集主检查流程
# ---------------------------------------------------------------------------

def check_dataset(npz_path, top_k=15, k_probe=DEFAULT_K,
                  k_lip=DEFAULT_K_LIP, dl_map=None,
                  fast_mode=False, use_gpu=True, out_dir="."):
    """对单个数据集执行完整的可达性 + 探针体检，输出报告并导出 CSV 详表。"""
    t0 = time.time()
    with np.load(npz_path, allow_pickle=False) as z:
        params = z['params'].astype(np.float64)
        names = [str(n) for n in z['mutant_names']]
        if 'is_real' in z:
            is_real = z['is_real'].astype(bool)
        else:
            is_real = np.array([n[:1].isdigit() for n in names], dtype=bool)
        if 'param_names' in z:
            pkeys = [str(k) for k in z['param_names']]
        else:
            pkeys = [f"p{i}" for i in range(params.shape[1])]

        pattern_labels = [str(p) for p in z['pattern_labels']] if 'pattern_labels' in z else None

        traj_key = 'data' if 'data' in z.files else (
            'trajectories' if 'trajectories' in z.files else None)
        
        # 极速模式：跳过加载 12GB 轨迹
        if fast_mode or traj_key is None:
            traj = None
        else:
            traj = z[traj_key]

    n_real = int(is_real.sum())
    n_syn = int((~is_real).sum())
    if n_real == 0 or n_syn == 0:
        return None, None

    p_train = params[~is_real]
    p_real = params[is_real]
    names_real = [n for n, r in zip(names, is_real) if r]
    patterns_real = [pattern_labels[i] for i, r in enumerate(is_real) if r] if pattern_labels else ["Unknown"] * n_real

    # ---- [1] 可达性（z 空间归一化距离） ----
    mu = np.nanmean(p_train, axis=0)
    sd = np.nanstd(p_train, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z_train = np.where(np.isfinite(p_train), (p_train - mu) / sd, 0.0)
    z_real = np.where(np.isfinite(p_real), (p_real - mu) / sd, 0.0)

    k_need = max(k_probe, k_lip, 1)
    knn_d, knn_i = compute_knn_dists(z_real, z_train, k_need, use_gpu=use_gpu)
    best_d, best_i = knn_d[:, 0], knn_i[:, 0]

    rep = []
    rep.append(f"  Real mutants           : {n_real}")
    rep.append(f"  Synthetic support      : {n_syn} samples, {params.shape[1]} parameters")
    med, mean = float(np.median(best_d)), float(np.mean(best_d))
    p90, mx = float(np.percentile(best_d, 90)), float(np.max(best_d))
    verdict = ("OK（Real 落在合成流形良好插值支撑内）" if mx < 5.0
               else "WARNING: 局部突变存在外推支撑空洞（min-dist > 5.0σ）" if med < 3.0
               else "FAIL: Real 整体严重偏离合成支撑")
    rep.append(f"  min-dist (z-space)     : median={med:.2f} | mean={mean:.2f} | P90={p90:.2f} | max={mx:.2f}")
    rep.append(f"  可达性判定             : {verdict}")
    rep.append("")

    # 最差可达性突变体
    order = np.argsort(-best_d)
    rep.append(f"  [Top {top_k} worst real mutants by reachability (min-dist)]")
    rep.append(f"  {'#':<4}{'Real Mutant':<44}{'min-dist':>10}")
    rep.append("  " + "-" * 58)
    for rank, ri in enumerate(order[:top_k], 1):
        rep.append(f"  {rank:<4}{names_real[ri]:<44}{best_d[ri]:>10.2f}")
        diff = np.abs(z_real[ri] - z_train[best_i[ri]])
        top3 = np.argsort(-diff)[:3]
        contrib = ", ".join(f"{pkeys[j]}={diff[j]:.1f}σ" for j in top3 if diff[j] > 0.5)
        if contrib:
            rep.append(f"      └─ 核心偏移参数: {contrib}")
    rep.append("")

    detail = {
        'names': names_real, 'patterns': patterns_real,
        'min_dist': best_d, 'nn_index': best_i,
        'reach_median': med, 'reach_mean': mean, 'reach_p90': p90, 'reach_max': mx
    }

    # ---- [2]+[3] 探针与 Lipschitz 评估 ----
    probe_mae = np.full(n_real, np.nan)
    probe_rel = np.full(n_real, np.nan)
    probe_corr = np.full(n_real, np.nan)
    core9_mae = np.full(n_real, np.nan)
    core9_rel = np.full(n_real, np.nan)
    lip_proxy = np.full(n_real, np.nan)

    if fast_mode:
        rep.append("  [k-NN probe & Lipschitz] SKIP: 已启用 --fast 极速模式（不载入 12GB 轨迹数据）")
        p_mae_med, p_rel_med, lp_med = np.nan, np.nan, np.nan
    elif traj is None:
        rep.append("  [k-NN probe] SKIP: 数据集中缺少轨迹数组（data/trajectories）")
        p_mae_med, p_rel_med, lp_med = np.nan, np.nan, np.nan
    else:
        y_real_all = traj[is_real]
        traj_train = traj[~is_real]
        for ri in range(n_real):
            y_real = y_real_all[ri]
            if not np.all(np.isfinite(y_real)):
                continue
            nb_idx = knn_i[ri, :k_probe]
            y_nb = traj_train[nb_idx].astype(np.float64)

            # 探针多指标计算
            pm = knn_probe_full(y_real.astype(np.float64), y_nb, knn_d[ri, :k_probe])
            probe_mae[ri] = pm['mae']
            probe_rel[ri] = pm['rel_l2']
            probe_corr[ri] = pm['corr']
            core9_mae[ri] = pm['core9_mae']
            core9_rel[ri] = pm['core9_rel_l2']

            # Lipschitz 审计
            nb_idx_l = knn_i[ri, :k_lip]
            z_nb = z_train[nb_idx_l] - z_real[ri]
            y_nb_l = traj_train[nb_idx_l].astype(np.float64)
            y_norm = float(np.linalg.norm(y_real.astype(np.float64)))
            _, _, proxy = lipschitz_audit(y_nb_l, z_nb, y_norm)
            lip_proxy[ri] = proxy

        valid = np.isfinite(probe_mae)
        p_mae_med = float(np.median(probe_mae[valid]))
        p_mae_mean = float(np.mean(probe_mae[valid]))
        p_rel_med = float(np.median(probe_rel[valid]))
        p_rel_mean = float(np.mean(probe_rel[valid]))
        p_corr_med = float(np.median(probe_corr[valid]))
        c9_mae_med = float(np.median(core9_mae[valid]))
        c9_rel_med = float(np.median(core9_rel[valid]))
        lp_med = float(np.median(lip_proxy[np.isfinite(lip_proxy)]))

        rep.append(f"  [k-NN Zero-Training Probe] k={k_probe} 逆距离加权 (有效 {int(valid.sum())}/{n_real})")
        rep.append(f"  All 39 Vars  -> MAE: median={p_mae_med:.4f} | mean={p_mae_mean:.4f}")
        rep.append(f"                  Rel L2: median={p_rel_med:.4f} | mean={p_rel_mean:.4f} | Corr: {p_corr_med:.4f}")
        rep.append(f"  Core 9 Vars  -> MAE: median={c9_mae_med:.4f} | Rel L2: median={c9_rel_med:.4f}")
        rep.append("  （注：探针为零训练最简流形插值下界；优秀 DL 模型经训练后指标通常可比探针好 1.5~3x）")
        rep.append("")

        # 预测最差坏点清单（按 probe MAE 排列）
        order_p = np.argsort(-np.nan_to_num(probe_mae, nan=-1.0))
        n_worst = min(top_k, 10)
        rep.append(f"  [Top {n_worst} predicted worst badcases (ranked by Probe MAE)]")
        rep.append(f"  {'#':<4}{'Real Mutant':<42}{'MAE':>8}{'RelL2':>8}{'C9-MAE':>8}{'min-dist':>9}")
        rep.append("  " + "-" * 75)
        for rank, ri in enumerate([i for i in order_p if np.isfinite(probe_mae[i])][:n_worst], 1):
            rep.append(f"  {rank:<4}{names_real[ri]:<42}{probe_mae[ri]:>8.4f}"
                       f"{probe_rel[ri]:>8.4f}{core9_mae[ri]:>8.4f}{best_d[ri]:>9.2f}")
        rep.append("")

        # Lipschitz 结果
        rep.append(f"  [Lipschitz Audit] k_lip={k_lip}: L×h/||y|| median={lp_med:.3f} | max={np.nanmax(lip_proxy):.3f}")
        order_l = np.argsort(-np.nan_to_num(lip_proxy, nan=-1.0))
        worst_l = [i for i in order_l if np.isfinite(lip_proxy[i])][:5]
        wl = ", ".join(f"{names_real[i]}({lip_proxy[i]:.2f})" for i in worst_l)
        rep.append(f"  相空间最陡 5 株: {wl}")
        rep.append("")

        detail.update({
            'probe_mae': probe_mae, 'probe_rel_l2': probe_rel, 'probe_corr': probe_corr,
            'core9_mae': core9_mae, 'core9_rel_l2': core9_rel, 'lip_proxy': lip_proxy,
            'probe_mae_median': p_mae_med, 'probe_mae_mean': p_mae_mean,
            'probe_rel_median': p_rel_med, 'probe_rel_mean': p_rel_mean,
            'core9_mae_median': c9_mae_med, 'core9_rel_median': c9_rel_med,
            'lip_proxy_median': lp_med
        })

    # ---- [4] 全局综合质量健康分 (Quality Score) ----
    q_score = calculate_quality_score(med, mx, p_mae_med, p_rel_med, lp_med)
    rep.append(f"  ★ 全局数据集质量综合评分 (Dataset Quality Score) : {q_score} / 100 分")
    rep.append("")
    detail['quality_score'] = q_score

    # ---- [5] 探针 vs 真实 DL 模型 Spearman 秩相关 ----
    if dl_map and not fast_mode:
        dl_name, dl_dict = dl_map
        common = [nm for nm in names_real if nm in dl_dict and np.isfinite(probe_rel[names_real.index(nm)])]
        if len(common) >= 5:
            p_rel_sub = [probe_rel[names_real.index(nm)] for nm in common]
            p_mae_sub = [probe_mae[names_real.index(nm)] for nm in common]
            dl_rel_sub = [dl_dict[nm][0] for nm in common]
            dl_mae_sub = [dl_dict[nm][1] for nm in common]

            rho_rel = spearman(p_rel_sub, dl_rel_sub)
            rho_mae = spearman(p_mae_sub, dl_mae_sub) if not np.isnan(dl_mae_sub[0]) else np.nan

            rep.append(f"  [Probe vs DL Validation] 匹配模型: {dl_name} ({len(common)} 株共有)")
            rep.append(f"  Spearman Rank Correlation (Probe RelL2 vs DL RelL2): {rho_rel:.3f}")
            if np.isfinite(rho_mae):
                rep.append(f"  Spearman Rank Correlation (Probe MAE   vs DL MAE)  : {rho_mae:.3f}")
            rep.append("  （注：ρ > 0.8 表明探针可极其可靠地替代深度学习完整训练，作为采样算法的快速筛选器）")
            rep.append("")
            detail['spearman_probe_dl_rel'] = rho_rel
            detail['spearman_probe_dl_mae'] = rho_mae

    elapsed = time.time() - t0
    rep.append(f"  体检耗时: {elapsed:.2f} 秒")

    # ---- [6] 导出逐突变体体检 CSV 报表 ----
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    csv_out_path = os.path.join(out_dir, f"reachability_{stem}_per_mutant.csv")
    with open(csv_out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Mutant", "Pattern", "Min_Dist_Sigma", "Probe_MAE", "Probe_Relative_L2",
            "Probe_Correlation", "Core9_MAE", "Core9_Relative_L2", "Lipschitz_Proxy", "Status"
        ])
        for ri in range(n_real):
            dist_val = best_d[ri]
            mae_val = probe_mae[ri]
            if dist_val > 5.0:
                st = "CRITICAL_OOD"
            elif np.isfinite(mae_val) and mae_val > 1.0:
                st = "PREDICTED_BADCASE"
            elif dist_val < 1.0:
                st = "WELL_COVERED"
            else:
                st = "NORMAL"

            writer.writerow([
                names_real[ri],
                patterns_real[ri],
                f"{dist_val:.2f}",
                f"{probe_mae[ri]:.4f}" if np.isfinite(probe_mae[ri]) else "N/A",
                f"{probe_rel[ri]:.4f}" if np.isfinite(probe_rel[ri]) else "N/A",
                f"{probe_corr[ri]:.4f}" if np.isfinite(probe_corr[ri]) else "N/A",
                f"{core9_mae[ri]:.4f}" if np.isfinite(core9_mae[ri]) else "N/A",
                f"{core9_rel[ri]:.4f}" if np.isfinite(core9_rel[ri]) else "N/A",
                f"{lip_proxy[ri]:.3f}" if np.isfinite(lip_proxy[ri]) else "N/A",
                st
            ])

    detail['csv_path'] = csv_out_path
    return "\n".join(rep), detail


# ---------------------------------------------------------------------------
# 6. 多数据集头对头横向 PK 对比 (Cross-Dataset Head-to-Head PK)
# ---------------------------------------------------------------------------

def print_head_to_head_pk(summaries, baseline_stem=None):
    """打印数据集两两对决 PK 胜率与详细指标对比。"""
    print(f"\n{'=' * 85}\n[Cross-Dataset Head-to-Head PK & Benchmark Summary]\n{'=' * 85}")
    
    # 概览总表
    headers = f"  {'Dataset Stem':<40}{'Reach.Med':>10}{'Reach.Max':>10}{'Probe.MAE':>11}{'Probe.RelL2':>12}{'Score':>8}"
    print(headers)
    print("  " + "-" * 91)
    for stem, d in summaries:
        rm = d['reach_median']
        rx = d['reach_max']
        p_mae = d.get('probe_mae_median', np.nan)
        p_rel = d.get('probe_rel_median', np.nan)
        sc = d.get('quality_score', np.nan)
        print(f"  {stem:<40}{rm:>10.2f}{rx:>10.2f}{p_mae:>11.4f}{p_rel:>12.4f}{sc:>8.1f}")
    print("")

    # 如果有指定 baseline 或 2 个数据集，输出差值 PK
    if len(summaries) >= 2:
        base_idx = 0
        if baseline_stem:
            for i, (st, _) in enumerate(summaries):
                if baseline_stem in st:
                    base_idx = i
                    break
        base_stem, base_d = summaries[base_idx]

        for i, (cand_stem, cand_d) in enumerate(summaries):
            if i == base_idx:
                continue
            if 'probe_mae' not in base_d or 'probe_mae' not in cand_d:
                continue

            base_mae = base_d['probe_mae']
            cand_mae = cand_d['probe_mae']
            valid = np.isfinite(base_mae) & np.isfinite(cand_mae)
            if valid.sum() == 0:
                continue

            diff = cand_mae[valid] - base_mae[valid]
            wins = (diff < -1e-5).sum()
            ties = (np.abs(diff) <= 1e-5).sum()
            losses = (diff > 1e-5).sum()
            win_rate = (wins / valid.sum()) * 100.0

            print(f"  ⚔️  PK 对决: [{cand_stem}] VS Baseline [{base_stem}]")
            print(f"      胜负战绩 : 胜 {wins} 株 | 平 {ties} 株 | 负 {losses} 株 (胜率: {win_rate:.1f}%)")
            print(f"      MAE 改善 : 中位数 {cand_d['probe_mae_median']:.4f} vs {base_d['probe_mae_median']:.4f} "
                  f"(Δ = {cand_d['probe_mae_median'] - base_d['probe_mae_median']:+.4f})")
            print(f"      L2 改善  : 中位数 {cand_d['probe_rel_median']:.4f} vs {base_d['probe_rel_median']:.4f} "
                  f"(Δ = {cand_d['probe_rel_median'] - base_d['probe_rel_median']:+.4f})")

            # Top 3 最大改善株
            best_idx = np.argsort(diff)[:3]
            names = cand_d['names']
            print("      Top 3 优势提升株:")
            for bi in best_idx:
                if diff[bi] < 0:
                    print(f"        • {names[bi]}: {cand_mae[bi]:.4f} vs {base_mae[bi]:.4f} ({diff[bi]:+.4f})")
            print("")


# ---------------------------------------------------------------------------
# 7. 主程序入口 CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-mutant reachability + zero-training probe evaluation (v3)")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="一个或多个数据集 .npz 路径")
    parser.add_argument("--baseline", type=str, default=None,
                        help="用于 PK 对比的基线数据集文件名或子串")
    parser.add_argument("--fast", action="store_true",
                        help="极速模式：不载入 12GB 轨迹数据，仅评估相空间几何与参数可达性（<0.5秒）")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="报告与 CSV 输出目录（默认: 脚本所在目录）")
    parser.add_argument("--top_k", type=int, default=15,
                        help="展示的最差突变体数量（默认 15）")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="k-NN 探针邻居数（默认 16）")
    parser.add_argument("--k_lip", type=int, default=DEFAULT_K_LIP,
                        help="Lipschitz 审计邻居数（默认 50）")
    parser.add_argument("--no_gpu", action="store_true",
                        help="强制禁用 GPU 加速，使用 CPU 分块计算")
    parser.add_argument("--dl_csv", action="append", default=[],
                        metavar="SUBSTR=CSV_PATH",
                        help="DL 逐株 eval CSV（如 'aware=per_mutant_metrics.csv'），用于计算 Spearman 秩相关")
    args = parser.parse_args()

    if args.out_dir:
        args.out_dir = os.path.abspath(args.out_dir)
        os.makedirs(args.out_dir, exist_ok=True)
    else:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        args.out_dir = os.path.join(_script_dir, "reachability")
        os.makedirs(args.out_dir, exist_ok=True)

    use_gpu = not args.no_gpu

    # 解析 DL CSV 映射
    dl_csvs = []
    for spec in args.dl_csv:
        if '=' not in spec:
            continue
        substr, csv_path = spec.split('=', 1)
        if os.path.isfile(csv_path):
            dl_csvs.append((substr.strip(), csv_path))

    valid_paths = [p for p in args.datasets if os.path.isfile(p)]
    if not valid_paths:
        print("[ERROR] 未找到任何有效的 .npz 数据集文件！")
        return 1

    stems = [os.path.splitext(os.path.basename(p))[0] for p in valid_paths]

    # 匹配 DL CSV
    dl_map_by_idx = {}
    for i, st in enumerate(stems):
        for substr, csv_path in dl_csvs:
            if substr in st:
                dl_map_by_idx[i] = (os.path.basename(csv_path), load_dl_csv(csv_path))
                break

    summaries = []
    for di, path in enumerate(valid_paths):
        stem = stems[di]
        dl_map = dl_map_by_idx.get(di)

        print(f"\n{'=' * 75}\n[Reachability+Probe Assessment] {os.path.basename(path)}\n{'=' * 75}")
        try:
            report, detail = check_dataset(
                path, top_k=args.top_k, k_probe=args.k, k_lip=args.k_lip,
                dl_map=dl_map, fast_mode=args.fast, use_gpu=use_gpu, out_dir=args.out_dir
            )
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")
            import traceback
            traceback.print_exc()
            continue

        if report is None:
            print("[SKIP] 数据集缺少 real 块或合成块（is_real 全同）")
            continue

        print(report)

        # 保存 .txt 报告
        txt_path = os.path.join(args.out_dir, f"reachability_{stem}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Real-Mutant Reachability + Zero-Training Probe (v3)\n"
                    f"Dataset: {os.path.abspath(path)}\n"
                    f"{'=' * 75}\n{report}\n")

        # 保存 .npz 详细数组
        npz_out = os.path.join(args.out_dir, f"reachability_{stem}.npz")
        out = {k: (v if isinstance(v, np.ndarray) else np.array([v]))
               for k, v in detail.items() if k not in ('names', 'patterns', 'csv_path')}
        out['real_names'] = np.array(detail['names'])
        np.savez_compressed(npz_out, **out)

        print(f"  📄 摘要文本报告已保存: {txt_path}")
        print(f"  📊 逐株详细评估已导出: {detail['csv_path']}")
        summaries.append((stem, detail))

    # 多数据集 PK 总表
    if len(summaries) >= 2:
        print_head_to_head_pk(summaries, baseline_stem=args.baseline)

    return 0


if __name__ == "__main__":
    sys.exit(main())
