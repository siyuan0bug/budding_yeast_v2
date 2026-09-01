"""
Real-Mutant Reachability Check (参数空间可达性检查)
================================================================================
目的：训练前验证「126 株 real 测试突变体是否落在训练集（合成样本）的参数
支撑内」。支撑空洞（real 距训练集几百个标准差）对 max|params|/std 类 sanity
检查完全不可见，只有显式计算 min z-distance 才能暴露（v3.7 教训：
GAL-PDS1-dbΔ 距训练集 459σ → 模型对该突变体物理上无法插值）。

方法：
  1. 用合成样本（is_real=False 块）的均值/标准差定义 z 空间；
  2. 每株 real 突变体标准化后，计算其到每条合成样本的欧氏距离，取最小值；
  3. 输出 min-dist 分布（中位/均值/P90/max）+ 最差的 real 突变体明细
     （含驱动距离的前 3 个参数，便于定位支撑空洞在哪个参数上）。

判读参考（来自本项目历史测量）：
  min-dist 中位 ~1-10  : real 落在训练分布内部，可插值（random 集 ~1.0）
  min-dist > 50        : 存在支撑空洞，需在生成侧修复（如 v3.7 的 459）

用法：
  python reachability_check.py --datasets a.npz b.npz
  python reachability_check.py --datasets a.npz --out_dir ./reach_reports
  （每个数据集输出一份 reachability_<数据集文件名>.txt / .npz）
"""

import os
import sys
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# 核心计算
# ---------------------------------------------------------------------------

def min_dist_chunked(z_real, z_train, chunk=2048):
    """逐块计算 126 x N 的欧氏距离矩阵，返回每株 real 的 min-dist 与
    最近邻索引。距离 = ||a||^2 + ||b||^2 - 2ab（数值下溢 clamp 到 0）。"""
    n_real = z_real.shape[0]
    best_d = np.full(n_real, np.inf)
    best_i = np.zeros(n_real, dtype=np.int64)
    tr_sq = np.einsum('ij,ij->i', z_train, z_train)
    re_sq = np.einsum('ij,ij->i', z_real, z_real)
    for s in range(0, n_real, chunk):
        e = min(s + chunk, n_real)
        # (e-s, N) = re_sq + tr_sq - 2 * z_real @ z_train.T
        d2 = re_sq[s:e, None] + tr_sq[None, :] \
            - 2.0 * (z_real[s:e] @ z_train.T)
        np.maximum(d2, 0.0, out=d2)
        idx = np.argmin(d2, axis=1)
        rows = np.arange(e - s)
        dmin = np.sqrt(d2[rows, idx])
        upd = dmin < best_d[s:e]
        best_i[s:e] = np.where(upd, idx, best_i[s:e])
        best_d[s:e] = np.where(upd, dmin, best_d[s:e])
    return best_d, best_i


def check_dataset(npz_path, top_k=15):
    """对单个数据集执行可达性检查。返回 (report_text, detail_dict)。"""
    with np.load(npz_path, allow_pickle=False) as z:
        params = z['params'].astype(np.float64)
        names = [str(n) for n in z['mutant_names']]
        if 'is_real' in z:
            is_real = z['is_real'].astype(bool)
        else:
            # 旧格式（mass/random 采样器）无 is_real 字段：real 样本名以数字
            # 开头（"83_GAL-PDS1-dbΔ"），合成样本名不以数字开头
            # （"MASS_..." / "WB-MASS_..."）。
            is_real = np.array([n[:1].isdigit() for n in names], dtype=bool)
        if 'param_names' in z:
            pkeys = [str(k) for k in z['param_names']]
        else:
            pkeys = [f"p{i}" for i in range(params.shape[1])]

    n_real = int(is_real.sum())
    if n_real == 0:
        return None, None
    if int((~is_real).sum()) == 0:
        return None, None

    p_train = params[~is_real]
    p_real = params[is_real]
    names_real = [n for n, r in zip(names, is_real) if r]

    # z 空间：由合成块定义（nan-safe，NaN 视作该参数未编辑 -> 填 0 即均值）
    mu = np.nanmean(p_train, axis=0)
    sd = np.nanstd(p_train, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)          # 恒定参数不参与距离
    z_train = np.where(np.isfinite(p_train), (p_train - mu) / sd, 0.0)
    z_real = np.where(np.isfinite(p_real), (p_real - mu) / sd, 0.0)

    best_d, best_i = min_dist_chunked(z_real, z_train)

    # 每株 real 的参数级 z 贡献（相对其最近邻）：定位空洞在哪个参数上
    order = np.argsort(-best_d)
    lines = []
    lines.append(f"  {'#':<4}{'real mutant':<44}{'min-dist':>10}")
    lines.append("  " + "-" * 58)
    for rank, ri in enumerate(order[:top_k], 1):
        lines.append(f"  {rank:<4}{names_real[ri]:<44}{best_d[ri]:>10.2f}")
        diff = np.abs(z_real[ri] - z_train[best_i[ri]])
        top3 = np.argsort(-diff)[:3]
        contrib = ", ".join(
            f"{pkeys[j]}={diff[j]:.1f}σ" for j in top3 if diff[j] > 0.5)
        if contrib:
            lines.append(f"      └─ 驱动参数: {contrib}")

    med, mean = float(np.median(best_d)), float(np.mean(best_d))
    p90, mx = float(np.percentile(best_d, 90)), float(np.max(best_d))
    verdict = ("OK（real 落在训练支撑内）" if mx < 50
               else "WARNING: 存在支撑空洞，训练前需修复生成侧" if med < 50
               else "FAIL: real 整体偏离训练支撑")

    rep = []
    rep.append(f"  Real mutants           : {n_real}")
    rep.append(f"  Synthetic support      : {int((~is_real).sum())} samples, "
               f"{params.shape[1]} params")
    rep.append(f"  min-dist (z-space)     : median={med:.2f} | mean={mean:.2f} "
               f"| P90={p90:.2f} | max={mx:.2f}")
    rep.append(f"  判定                   : {verdict}")
    rep.append("")
    rep.append(f"  [Top {top_k} worst real mutants]")
    rep.extend(lines)
    detail = {'names': names_real, 'min_dist': best_d,
              'nn_index': best_i, 'median': med, 'mean': mean,
              'p90': p90, 'max': mx}
    return "\n".join(rep), detail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-mutant parameter-space reachability check")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="一个或多个数据集 .npz 路径")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="报告输出目录（默认: 脚本所在目录）")
    parser.add_argument("--top_k", type=int, default=15,
                        help="明细表中展示最差 real 突变体的数量")
    args = parser.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    else:
        args.out_dir = os.path.dirname(os.path.abspath(__file__))

    summaries = []
    for path in args.datasets:
        if not os.path.isfile(path):
            print(f"[SKIP] {path} (file not found)")
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{'=' * 70}\n[Reachability] {os.path.basename(path)}\n{'=' * 70}")
        try:
            report, detail = check_dataset(path, top_k=args.top_k)
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
            f.write(f"Real-Mutant Reachability Check\n"
                    f"Dataset: {os.path.abspath(path)}\n"
                    f"{'=' * 70}\n{report}\n")
        npz_out = os.path.join(args.out_dir, f"reachability_{stem}.npz")
        np.savez_compressed(
            npz_out,
            real_names=np.array(detail['names']),
            min_dist=detail['min_dist'],
            nn_index=detail['nn_index'])
        print(f"\n  报告已保存: {txt_path}")
        print(f"  明细已保存: {npz_out}")
        summaries.append((stem, detail))

    # 多数据集对比总表
    if len(summaries) > 1:
        print(f"\n{'=' * 70}\n[Cross-dataset summary]\n{'=' * 70}")
        print(f"  {'dataset':<52}{'median':>8}{'P90':>8}{'max':>8}")
        print("  " + "-" * 76)
        for stem, d in summaries:
            print(f"  {stem:<52}{d['median']:>8.2f}{d['p90']:>8.2f}"
                  f"{d['max']:>8.2f}")


if __name__ == "__main__":
    sys.exit(main())
