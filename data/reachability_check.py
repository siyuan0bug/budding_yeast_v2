"""
Real-Mutant Reachability Check + 训练前预测探针 (v2)
================================================================================
目的：不训练任何模型，评估「该采样算法能否让模型做好 real mutants 预测」。

三层诊断（均为零训练）：

  [1] 可达性（原有功能）
      126 株 real 是否落在合成样本参数支撑内（z 空间 min-dist）。
      支撑空洞（>50σ）→ 任何模型物理上无法插值（v3.7 教训：459σ）。

  [2] k-NN 回归探针（Nadaraya-Watson 惰性学习器）
      每株 real 用参数空间最近 k 个合成样本的轨迹（逆距离加权）平均作为
      「预测」，直接算 relL2。同时捕获局部密度与局部映射平滑性：
        探针误差小 → 邻域密且轨迹平缓 → 好模型可做好（数据侧无天花板）
        探针误差大 → 数据侧已有天花板 → 换模型/训练配置难救
      注意：DL 模型通常应显著优于探针；DL ≈ 探针 说明瓶颈在模型侧。

  [3] 局部 Lipschitz 审计（纯几何，无拟合）
      对每株 real 取最近 k_lip 个邻居，估计局部映射陡度
      L = max ||y_a-y_b||/||p_a-p_b||，代理误差 = L × h（h=邻域填充距离）。
      可在训练前逐株点名「必差」的突变体（邻域内轨迹剧烈变化者）。

  [4] 探针 vs DL 秩相关验证（可选，--dl_csv）
      若提供同数据集的 DL 逐株 eval CSV（Mutant,Pattern,MAE,MSE,
      Relative_L2,Correlation），计算探针 relL2 与 DL relL2 的 Spearman
      秩相关。秩相关高（>0.8）→ 探针可作为采样算法的快速筛选器：
      生成 → 探针（分钟级）→ 达标才进 DL 训练（小时级）。

用法：
  python reachability_check.py --datasets a.npz b.npz
  python reachability_check.py --datasets a.npz --k 16 --k_lip 50
  python reachability_check.py --datasets a.npz --dl_csv wb=blind_per_mutant_metrics.csv
  （每个数据集输出 reachability_<数据集文件名>.txt / .npz，逐株明细在 npz）

内存提示：探针需加载全量轨迹，100k 数据集约 8-16GB RAM。
"""

import os
import sys
import csv
import argparse
import numpy as np

DEFAULT_K = 16        # k-NN 探针邻居数
DEFAULT_K_LIP = 50    # Lipschitz 审计邻居数


# ---------------------------------------------------------------------------
# 距离计算
# ---------------------------------------------------------------------------

def knn_dist_chunked(z_real, z_train, k, chunk=128):
    """逐块计算 126 x N 欧氏距离，返回每株 real 最近 k 个邻居的
    (距离, 索引)，按距离升序。距离 = ||a||^2+||b||^2-2ab（clamp>=0）。"""
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
        # 每块取最小的 k 个
        if k < d2.shape[1]:
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        else:
            part = np.tile(np.arange(d2.shape[1]), (e - s, 1))
        rows = np.arange(e - s)[:, None]
        dpart = np.sqrt(d2[rows, part])
        order = np.argsort(dpart, axis=1)
        dpart = np.take_along_axis(dpart, order, axis=1)
        ipart = np.take_along_axis(part, order, axis=1)
        # 与已有 best 归并取前 k
        merged_d = np.concatenate([best_d[s:e], dpart], axis=1)
        merged_i = np.concatenate([best_i[s:e], ipart], axis=1)
        sel = np.argsort(merged_d, axis=1)[:, :k]
        best_d[s:e] = np.take_along_axis(merged_d, sel, axis=1)
        best_i[s:e] = np.take_along_axis(merged_i, sel, axis=1)
    return best_d, best_i


# ---------------------------------------------------------------------------
# 探针
# ---------------------------------------------------------------------------

def knn_probe(y_real, y_nb, dists):
    """逆距离加权 k-NN 探针。
    y_real: (39,500) 真实轨迹; y_nb: (k,39,500) 邻居轨迹; dists: (k,)
    返回 relL2 = ||pred-y_real||_F / ||y_real||_F。"""
    d = np.maximum(dists, 1e-12)
    w = 1.0 / d
    w = w / w.sum()
    pred = np.einsum('k,kvt->vt', w, y_nb)
    denom = np.linalg.norm(y_real)
    if denom < 1e-12:
        return np.nan
    return float(np.linalg.norm(pred - y_real) / denom)


def lipschitz_audit(y_nb, z_nb, y_norm):
    """局部 Lipschitz 估计 + 代理误差。
    y_nb: (k,39,500); z_nb: (k,P) z 空间参数; y_norm: real 轨迹 Frobenius 范数。
    L = median ||y_a-y_b||_F / ||z_a-z_b||_2; h = 邻居 z 距离中位数。
    proxy = L×h/||y_real|| —— 与 relL2 同量纲，跨株可比（arrest 株 ||y|| 小，
    未归一化的 L×h 会系统性低估其相对误差）。"""
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
    # 鲁棒估计：过滤近重合点对（pd→0 时比率数值爆炸；事件驱动的
    # 不连续性本身是真实信号，由 k-NN 探针捕获，此处只需局部陡度）
    valid = pd_pair > max(1e-3, 1e-6 * float(np.max(pd_pair)))
    if not valid.any():
        return np.nan, np.nan, np.nan
    ratios = yd_pair[valid] / pd_pair[valid]
    L = float(np.median(ratios))
    h = float(np.median(np.sqrt(np.sum(z_nb ** 2, axis=1))))  # 到 real 的距离
    return L, h, L * h / y_norm


# ---------------------------------------------------------------------------
# 秩相关（无 scipy 依赖）
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
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # 平均秩（并列）
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


# ---------------------------------------------------------------------------
# 数据集主检查
# ---------------------------------------------------------------------------

def load_dl_csv(path):
    """读取 DL 逐株 eval CSV -> {mutant_name: Relative_L2}"""
    out = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = (row.get('Mutant') or '').strip()
            val = row.get('Relative_L2')
            if name and val:
                try:
                    out[name] = float(val)
                except ValueError:
                    pass
    return out


def check_dataset(npz_path, top_k=15, k_probe=DEFAULT_K,
                  k_lip=DEFAULT_K_LIP, dl_map=None):
    """对单个数据集执行可达性 + 探针检查。返回 (report_text, detail_dict)。"""
    with np.load(npz_path, allow_pickle=False) as z:
        params = z['params'].astype(np.float64)
        names = [str(n) for n in z['mutant_names']]
        if 'is_real' in z:
            is_real = z['is_real'].astype(bool)
        else:
            # 旧格式：real 样本名以数字开头，合成样本名不以数字开头
            is_real = np.array([n[:1].isdigit() for n in names], dtype=bool)
        if 'param_names' in z:
            pkeys = [str(k) for k in z['param_names']]
        else:
            pkeys = [f"p{i}" for i in range(params.shape[1])]
        traj_key = 'data' if 'data' in z.files else (
            'trajectories' if 'trajectories' in z.files else None)
        traj = z[traj_key] if traj_key else None

    n_real = int(is_real.sum())
    n_syn = int((~is_real).sum())
    if n_real == 0 or n_syn == 0:
        return None, None

    p_train = params[~is_real]
    p_real = params[is_real]
    names_real = [n for n, r in zip(names, is_real) if r]

    # ---- [1] 可达性（z 空间由合成块定义） ----
    mu = np.nanmean(p_train, axis=0)
    sd = np.nanstd(p_train, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z_train = np.where(np.isfinite(p_train), (p_train - mu) / sd, 0.0)
    z_real = np.where(np.isfinite(p_real), (p_real - mu) / sd, 0.0)

    k_need = max(k_probe, k_lip, 1)
    knn_d, knn_i = knn_dist_chunked(z_real, z_train, k_need)
    best_d, best_i = knn_d[:, 0], knn_i[:, 0]

    rep = []
    rep.append(f"  Real mutants           : {n_real}")
    rep.append(f"  Synthetic support      : {n_syn} samples, "
               f"{params.shape[1]} params")
    med, mean = float(np.median(best_d)), float(np.mean(best_d))
    p90, mx = float(np.percentile(best_d, 90)), float(np.max(best_d))
    verdict = ("OK（real 落在训练支撑内）" if mx < 50
               else "WARNING: 存在支撑空洞，训练前需修复生成侧" if med < 50
               else "FAIL: real 整体偏离训练支撑")
    rep.append(f"  min-dist (z-space)     : median={med:.2f} | mean={mean:.2f} "
               f"| P90={p90:.2f} | max={mx:.2f}")
    rep.append(f"  判定                   : {verdict}")
    rep.append("")

    # ---- 可达性明细 ----
    order = np.argsort(-best_d)
    rep.append(f"  [Top {top_k} worst real mutants (reachability)]")
    rep.append(f"  {'#':<4}{'real mutant':<44}{'min-dist':>10}")
    rep.append("  " + "-" * 58)
    for rank, ri in enumerate(order[:top_k], 1):
        rep.append(f"  {rank:<4}{names_real[ri]:<44}{best_d[ri]:>10.2f}")
        diff = np.abs(z_real[ri] - z_train[best_i[ri]])
        top3 = np.argsort(-diff)[:3]
        contrib = ", ".join(
            f"{pkeys[j]}={diff[j]:.1f}σ" for j in top3 if diff[j] > 0.5)
        if contrib:
            rep.append(f"      └─ 驱动参数: {contrib}")
    rep.append("")

    detail = {'names': names_real, 'min_dist': best_d, 'nn_index': best_i,
              'median': med, 'mean': mean, 'p90': p90, 'max': mx}

    # ---- [2]+[3] 探针与 Lipschitz（需要轨迹） ----
    probe_rel = np.full(n_real, np.nan)
    lip_L = np.full(n_real, np.nan)
    lip_h = np.full(n_real, np.nan)
    lip_proxy = np.full(n_real, np.nan)

    if traj is None:
        rep.append("  [k-NN probe] SKIP: 数据集无轨迹数组（data/trajectories）")
    else:
        y_real_all = traj[is_real]
        skipped = 0
        for ri in range(n_real):
            y_real = y_real_all[ri]
            if not np.all(np.isfinite(y_real)):
                skipped += 1
                continue
            nb_idx = knn_i[ri, :k_probe]
            y_nb = traj[nb_idx].astype(np.float64)
            probe_rel[ri] = knn_probe(
                y_real.astype(np.float64), y_nb, knn_d[ri, :k_probe])
            # Lipschitz: 邻居相对 real 的 z 距离；proxy 按该株轨迹范数归一
            nb_idx_l = knn_i[ri, :k_lip]
            z_nb = z_train[nb_idx_l] - z_real[ri]
            y_nb_l = traj[nb_idx_l].astype(np.float64)
            y_norm = float(np.linalg.norm(y_real.astype(np.float64)))
            L, h, proxy = lipschitz_audit(y_nb_l, z_nb, y_norm)
            lip_L[ri], lip_h[ri], lip_proxy[ri] = L, h, proxy
        valid = np.isfinite(probe_rel)
        if valid.sum() == 0:
            rep.append("  [k-NN probe] SKIP: 无有效 real 轨迹")
        else:
            pv = probe_rel[valid]
            p_med = float(np.median(pv))
            p_mean = float(np.mean(pv))
            p_p90 = float(np.percentile(pv, 90))
            p_max = float(np.max(pv))
            rep.append(f"  [k-NN probe] k={k_probe} 逆距离加权 "
                       f"(有效 {int(valid.sum())}/{n_real}"
                       + (f", 跳过 {skipped} 株非有限轨迹" if skipped else "")
                       + ")")
            rep.append(f"  probe relL2            : median={p_med:.3f} | "
                       f"mean={p_mean:.3f} | P90={p_p90:.3f} | max={p_max:.3f}")
            rep.append("  （探针=零训练插值下界代理；DL 应显著优于探针，"
                       "DL≈探针 说明瓶颈在模型侧）")
            rep.append("")
            # 最差株（探针）
            n_worst = min(top_k, 10)
            order_p = np.argsort(-np.nan_to_num(probe_rel, nan=-1.0))
            rep.append(f"  [Top {n_worst} worst real mutants (probe)]")
            rep.append(f"  {'#':<4}{'real mutant':<44}{'probe':>8}{'min-dist':>10}")
            rep.append("  " + "-" * 66)
            for rank, ri in enumerate(
                    [i for i in order_p if np.isfinite(probe_rel[i])][:n_worst], 1):
                rep.append(f"  {rank:<4}{names_real[ri]:<44}"
                           f"{probe_rel[ri]:>8.3f}{best_d[ri]:>10.2f}")
            rep.append("")
            # Lipschitz 汇总
            lv = np.isfinite(lip_proxy)
            if lv.sum() > 0:
                lp = lip_proxy[lv]
                rep.append(f"  [Lipschitz audit] k_lip={k_lip}: "
                           f"L×h median={np.median(lp):.3f} | "
                           f"max={np.max(lp):.3f}")
                order_l = np.argsort(-np.nan_to_num(lip_proxy, nan=-1.0))
                worst_l = [i for i in order_l if np.isfinite(lip_proxy[i])][:5]
                wl = ", ".join(f"{names_real[i]}({lip_proxy[i]:.2f})"
                               for i in worst_l)
                rep.append(f"  最陡 5 株: {wl}")
                rep.append("")
            detail.update({'probe_rel_l2': probe_rel, 'lip_L': lip_L,
                           'lip_h': lip_h, 'lip_proxy': lip_proxy,
                           'probe_median': p_med, 'probe_mean': p_mean,
                           'probe_max': p_max})

    # ---- [4] 探针 vs DL 秩相关 ----
    if dl_map:
        dl_name, dl_vals = dl_map
        common, p_list, d_list = [], [], []
        for ri, nm in enumerate(names_real):
            if nm in dl_vals and np.isfinite(probe_rel[ri]):
                common.append(nm)
                p_list.append(probe_rel[ri])
                d_list.append(dl_vals[nm])
        if len(common) >= 5:
            rho_probe = spearman(p_list, d_list)
            rho_lip = spearman(
                [lip_proxy[names_real.index(nm)] for nm in common
                 if np.isfinite(lip_proxy[names_real.index(nm)])],
                [dl_vals[nm] for nm in common
                 if np.isfinite(lip_proxy[names_real.index(nm)])])
            rep.append(f"  [Probe vs DL] {dl_name}: {len(common)} 株共有 "
                       f"(DL CSV {len(dl_vals)} 株)")
            rep.append(f"  Spearman(probe relL2, DL relL2) = {rho_probe:.3f}")
            if np.isfinite(rho_lip):
                rep.append(f"  Spearman(L×h proxy,   DL relL2) = {rho_lip:.3f}")
            rep.append(f"  均值: probe {np.mean(p_list):.3f} vs "
                       f"DL {np.mean(d_list):.3f} "
                       f"(DL/probe = {np.mean(d_list) / max(np.mean(p_list), 1e-9):.2f}x)")
            rep.append("  （ρ>0.8 → 探针可作为采样算法的快速筛选器，"
                       "免训练迭代采样设计）")
            rep.append("")
            detail['spearman_probe_dl'] = rho_probe
            detail['spearman_lip_dl'] = rho_lip
        else:
            rep.append(f"  [Probe vs DL] {dl_name}: 仅 {len(common)} 株共有，"
                       f"不足以计算秩相关（检查名称匹配）")
            rep.append("")

    return "\n".join(rep), detail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-mutant reachability + zero-training probe check")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="一个或多个数据集 .npz 路径")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="报告输出目录（默认: 脚本所在目录）")
    parser.add_argument("--top_k", type=int, default=15,
                        help="可达性明细表中展示最差 real 突变体的数量")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="k-NN 探针邻居数（默认 16）")
    parser.add_argument("--k_lip", type=int, default=DEFAULT_K_LIP,
                        help="Lipschitz 审计邻居数（默认 50）")
    parser.add_argument("--dl_csv", action="append", default=[],
                        metavar="SUBSTR=CSV_PATH",
                        help="DL 逐株 eval CSV，形如 'wb=blind_per_mutant_metrics.csv'。"
                             "SUBSTR 用于匹配数据集文件名（包含即匹配），可多次指定")
    args = parser.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    else:
        args.out_dir = os.path.dirname(os.path.abspath(__file__))

    # 解析 DL CSV 映射
    dl_csvs = []
    for spec in args.dl_csv:
        if '=' not in spec:
            print(f"[WARN] --dl_csv 格式应为 SUBSTR=CSV_PATH，忽略: {spec}")
            continue
        substr, csv_path = spec.split('=', 1)
        if not os.path.isfile(csv_path):
            print(f"[WARN] DL CSV 不存在，忽略: {csv_path}")
            continue
        dl_csvs.append((substr.strip(), csv_path))

    summaries = []
    valid_paths = [p for p in args.datasets if os.path.isfile(p)]
    stems = [os.path.splitext(os.path.basename(p))[0] for p in valid_paths]

    # 解析 DL CSV 映射（两阶段）：
    #   Pass 1: 精确 stem 认领（最可靠）；
    #   Pass 2: 未认领的数据集用最长子串匹配，且跳过已被精确认领的 spec
    #           （避免前缀撞车：flat 的 stem 是 s5mode stem 的前缀，
    #             flat 的 CSV spec 不应再匹配 s5 数据集）
    dl_map_by_idx = {}
    claimed = set()
    for i, st in enumerate(stems):
        for substr, csv_path in dl_csvs:
            if substr == st:
                dl_map_by_idx[i] = (os.path.basename(csv_path),
                                    load_dl_csv(csv_path))
                claimed.add(csv_path)
                break
    for i, st in enumerate(stems):
        if i in dl_map_by_idx:
            continue
        cands = [(s, p) for s, p in dl_csvs
                 if p not in claimed and s in st]
        if cands:
            cands.sort(key=lambda t: -len(t[0]))
            dl_map_by_idx[i] = (os.path.basename(cands[0][1]),
                                load_dl_csv(cands[0][1]))

    for di, path in enumerate(valid_paths):
        stem = stems[di]
        dl_map = dl_map_by_idx.get(di)

        print(f"\n{'=' * 70}\n[Reachability+Probe] {os.path.basename(path)}\n"
              f"{'=' * 70}")
        try:
            report, detail = check_dataset(
                path, top_k=args.top_k, k_probe=args.k,
                k_lip=args.k_lip, dl_map=dl_map)
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")
            continue
        if report is None:
            print("[SKIP] 数据集缺少 real 块或合成块（is_real 全同）")
            continue
        print(report)

        # 输出文件名携带数据集文件名
        txt_path = os.path.join(args.out_dir, f"reachability_{stem}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Real-Mutant Reachability + Zero-Training Probe\n"
                    f"Dataset: {os.path.abspath(path)}\n"
                    f"{'=' * 70}\n{report}\n")
        npz_out = os.path.join(args.out_dir, f"reachability_{stem}.npz")
        out = {k: (v if isinstance(v, np.ndarray) else np.array([v]))
               for k, v in detail.items() if k != 'names'}
        out['real_names'] = np.array(detail['names'])
        np.savez_compressed(npz_out, **out)
        print(f"  报告已保存: {txt_path}")
        print(f"  明细已保存: {npz_out}")
        summaries.append((stem, detail))

    # 多数据集对比总表
    if len(summaries) > 1:
        print(f"\n{'=' * 70}\n[Cross-dataset summary]\n{'=' * 70}")
        has_probe = any('probe_median' in d for _, d in summaries)
        if has_probe:
            print(f"  {'dataset':<44}{'reach.max':>10}{'probe.med':>10}"
                  f"{'probe.max':>10}{'L×h.max':>10}")
            print("  " + "-" * 84)
            for stem, d in summaries:
                pm = d.get('probe_median', np.nan)
                pmax = d.get('probe_max', np.nan)
                lmax = np.nanmax(d['lip_proxy']) if 'lip_proxy' in d else np.nan
                print(f"  {stem:<44}{d['max']:>10.2f}{pm:>10.3f}"
                      f"{pmax:>10.3f}{lmax:>10.3f}")
        else:
            print(f"  {'dataset':<52}{'median':>8}{'P90':>8}{'max':>8}")
            print("  " + "-" * 76)
            for stem, d in summaries:
                print(f"  {stem:<52}{d['median']:>8.2f}{d['p90']:>8.2f}"
                      f"{d['max']:>8.2f}")


if __name__ == "__main__":
    sys.exit(main())
