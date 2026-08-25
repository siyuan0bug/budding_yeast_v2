#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Seed Aggregator & Experiment Leaderboard (多 Seed 聚合与实验排行榜分析工具)
=============================================================================
功能:
  1. 自动递归扫描 experiments/ 分层目录，计算多 Seed 的指标均值 ± 标准差 (Mean ± Std)。
  2. 自动在超参签名目录下生成 `seed_summary.json`。
  3. 兼容扫描历史扁平目录 (如 eval_results_v3/, eval_result/)。
  4. 生成美观的 Markdown 实验排行榜 (Leaderboard) 与 CSV 导出。

用法示例:
  python analyze_experiments.py --exp_root ./experiments
  python analyze_experiments.py --exp_root ./experiments --dataset mass_v3 --sort_by real_mae
  python analyze_experiments.py --legacy_dir ./eval_results_v3
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from typing import Dict, List, Any, Optional


def load_metrics_from_folder(folder_path: str) -> Optional[Dict[str, Any]]:
    """从评估目录或实验子目录加载 global_metrics_denorm.json"""
    candidates = [
        os.path.join(folder_path, "eval", "global_metrics_denorm.json"),
        os.path.join(folder_path, "global_metrics_denorm.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
            except Exception:
                pass
    return None


def scan_hierarchical_experiments(exp_root: str) -> List[Dict[str, Any]]:
    """扫描 experiments/ 树形目录"""
    if not os.path.exists(exp_root):
        return []

    results = []
    # 结构: exp_root / dataset / model / loss_strategy / hparam_sig / seed_*
    for root, dirs, files in os.walk(exp_root):
        # 寻找包含 seed_* 文件夹的父级 (即 hparam_sig 目录)
        seed_dirs = [d for d in dirs if d.startswith("seed_")]
        if not seed_dirs:
            continue

        hparam_dir = root
        rel_path = os.path.relpath(hparam_dir, exp_root)
        path_parts = rel_path.split(os.sep)

        dataset_tag = path_parts[0] if len(path_parts) > 0 else "unknown"
        model_name = path_parts[1] if len(path_parts) > 1 else "unknown"
        loss_strat = path_parts[2] if len(path_parts) > 2 else "unknown"
        hparam_sig = path_parts[3] if len(path_parts) > 3 else "unknown"

        seed_data_list = []
        for sd in sorted(seed_dirs):
            seed_path = os.path.join(hparam_dir, sd)
            metrics = load_metrics_from_folder(seed_path)
            if metrics is not None:
                seed_data_list.append({
                    "seed": sd.replace("seed_", ""),
                    "path": seed_path,
                    "metrics": metrics
                })

        if seed_data_list:
            results.append({
                "type": "hierarchical",
                "hparam_dir": hparam_dir,
                "dataset_tag": dataset_tag,
                "model_name": model_name,
                "loss_strategy": loss_strat,
                "hparam_sig": hparam_sig,
                "seeds": seed_data_list
            })

    return results


def scan_legacy_experiments(legacy_dir: str) -> List[Dict[str, Any]]:
    """扫描历史扁平目录 (如 eval_results_v3)"""
    if not os.path.exists(legacy_dir):
        return []

    groups = {}
    for folder in os.listdir(legacy_dir):
        full_path = os.path.join(legacy_dir, folder)
        if not os.path.isdir(full_path):
            continue

        metrics = load_metrics_from_folder(full_path)
        if metrics is None:
            continue

        # 解析旧式名字: e.g. biom_refiner_plus_none_triple_dynamic_massv3_biom_refiner_plus_triple_width128_modes32_seed42
        if "_seed" in folder:
            prefix, seed_str = folder.rsplit("_seed", 1)
        else:
            prefix, seed_str = folder, "0"

        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append({
            "seed": seed_str,
            "path": full_path,
            "metrics": metrics
        })

    results = []
    for prefix, seed_list in groups.items():
        results.append({
            "type": "legacy",
            "hparam_dir": os.path.join(legacy_dir, prefix),
            "dataset_tag": "legacy",
            "model_name": prefix.split("_")[0],
            "loss_strategy": "legacy",
            "hparam_sig": prefix,
            "seeds": seed_list
        })
    return results


def extract_submetrics(d: Any) -> Dict[str, float]:
    """递归提取包含 MAE/Relative L2/Correlation 的最底层字典"""
    if not isinstance(d, dict):
        return {}
    if "MAE" in d:
        return d
    if "all_39_vars" in d and isinstance(d["all_39_vars"], dict):
        return d["all_39_vars"]
    if "real_mutants" in d:
        return extract_submetrics(d["real_mutants"])
    return {}


def aggregate_group_metrics(group_entry: Dict[str, Any]) -> Dict[str, Any]:
    """计算单个超参组多 Seed 的均值与标准差"""
    seeds = group_entry["seeds"]
    n_seeds = len(seeds)

    real_maes, real_l2s, real_corrs = [], [], []
    lhs_maes, lhs_l2s = [], []

    for s in seeds:
        m = s["metrics"]
        real_dict = extract_submetrics(m.get("real_mutants", m))
        if real_dict and "MAE" in real_dict:
            real_maes.append(float(real_dict.get("MAE", float("nan"))))
            real_l2s.append(float(real_dict.get("Relative L2", float("nan"))))
            real_corrs.append(float(real_dict.get("Correlation", float("nan"))))

        lhs_dict = extract_submetrics(m.get("lhs_test", {}))
        if lhs_dict and "MAE" in lhs_dict:
            lhs_maes.append(float(lhs_dict.get("MAE", float("nan"))))
            lhs_l2s.append(float(lhs_dict.get("Relative L2", float("nan"))))

    summary = {
        "dataset": group_entry["dataset_tag"],
        "model": group_entry["model_name"],
        "loss_strategy": group_entry["loss_strategy"],
        "hparam_signature": group_entry["hparam_sig"],
        "n_seeds": n_seeds,
        "seeds_evaluated": [s["seed"] for s in seeds],
        "real_mae_mean": float(np.nanmean(real_maes)) if real_maes else float("nan"),
        "real_mae_std": float(np.nanstd(real_maes)) if len(real_maes) > 1 else 0.0,
        "real_l2_mean": float(np.nanmean(real_l2s)) if real_l2s else float("nan"),
        "real_l2_std": float(np.nanstd(real_l2s)) if len(real_l2s) > 1 else 0.0,
        "real_corr_mean": float(np.nanmean(real_corrs)) if real_corrs else float("nan"),
        "real_corr_std": float(np.nanstd(real_corrs)) if len(real_corrs) > 1 else 0.0,
    }

    if lhs_maes:
        summary["lhs_mae_mean"] = float(np.nanmean(lhs_maes))
        summary["lhs_mae_std"] = float(np.nanstd(lhs_maes)) if len(lhs_maes) > 1 else 0.0

    # 若是新版分层目录，将 summary 保存到 hparam_dir 下
    if group_entry["type"] == "hierarchical" and os.path.exists(group_entry["hparam_dir"]):
        sum_path = os.path.join(group_entry["hparam_dir"], "seed_summary.json")
        try:
            with open(sum_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Aggregator & Leaderboard")
    parser.add_argument("--exp_root", type=str, default="./experiments", help="分层实验根目录")
    parser.add_argument("--legacy_dir", type=str, default=None, help="历史扁平结果目录 (如 ./eval_results_v3)")
    parser.add_argument("--dataset", type=str, default=None, help="按数据集标签过滤")
    parser.add_argument("--model", type=str, default=None, help="按模型名称过滤")
    parser.add_argument("--sort_by", type=str, default="real_mae", choices=["real_mae", "real_l2", "real_corr"],
                        help="排行榜排序指标 (默认 real_mae 升序)")
    parser.add_argument("--export_csv", type=str, default=None, help="导出 CSV 排行榜路径")
    args = parser.parse_args()

    groups = []
    # 扫描分层目录
    if os.path.exists(args.exp_root):
        groups.extend(scan_hierarchical_experiments(args.exp_root))

    # 扫描旧目录
    if args.legacy_dir and os.path.exists(args.legacy_dir):
        groups.extend(scan_legacy_experiments(args.legacy_dir))
    elif not groups and os.path.exists("./eval_results_v3"):
        # 默认回退检查旧版
        groups.extend(scan_legacy_experiments("./eval_results_v3"))

    if not groups:
        print(f"⚠️ 未在 {args.exp_root} 或 {args.legacy_dir} 找到任何评估结果。")
        return

    summaries = []
    for g in groups:
        if args.dataset and args.dataset.lower() not in g["dataset_tag"].lower():
            continue
        if args.model and args.model.lower() not in g["model_name"].lower():
            continue
        s = aggregate_group_metrics(g)
        summaries.append(s)

    if not summaries:
        print("⚠️ 过滤后无匹配的实验结果。")
        return

    # 排序
    reverse_sort = True if args.sort_by == "real_corr" else False
    key_fn = lambda x: x.get(f"{args.sort_by}_mean", float("inf") if not reverse_sort else -float("inf"))
    summaries.sort(key=key_fn, reverse=reverse_sort)

    # 打印 Leaderboard
    print(f"\n{'='*110}")
    print(f"🏆 Budding Yeast 细胞周期模型实验排行榜 (Total Runs: {len(summaries)} | Sorted by: {args.sort_by})")
    print(f"{'='*110}")

    headers = ["Rank", "Dataset", "Model", "Loss/Strategy", "Hparams", "Seeds", "Real MAE ↓", "Real Rel L2 ↓", "Real Corr ↑"]
    row_fmt = "{:<5} {:<10} {:<20} {:<22} {:<24} {:<7} {:<18} {:<18} {:<15}"
    print(row_fmt.format(*headers))
    print("-" * 110)

    for rank, s in enumerate(summaries, 1):
        mae_str = f"{s['real_mae_mean']:.4f} ± {s['real_mae_std']:.4f}" if s['n_seeds'] > 1 else f"{s['real_mae_mean']:.4f}"
        l2_str = f"{s['real_l2_mean']:.4f} ± {s['real_l2_std']:.4f}" if s['n_seeds'] > 1 else f"{s['real_l2_mean']:.4f}"
        corr_str = f"{s['real_corr_mean']:.4f} ± {s['real_corr_std']:.4f}" if s['n_seeds'] > 1 else f"{s['real_corr_mean']:.4f}"

        # 截断过长签名保证对齐
        hparam_short = s['hparam_signature'][:22] + ".." if len(s['hparam_signature']) > 24 else s['hparam_signature']
        strat_short = s['loss_strategy'][:20] + ".." if len(s['loss_strategy']) > 22 else s['loss_strategy']

        print(row_fmt.format(
            f"#{rank}",
            s['dataset'][:10],
            s['model'][:20],
            strat_short,
            hparam_short,
            str(s['n_seeds']),
            mae_str,
            l2_str,
            corr_str
        ))

    print(f"{'='*110}\n")

    # 导出 CSV
    if args.export_csv:
        import csv
        with open(args.export_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"📊 榜单已成功导出至: {args.export_csv}")


if __name__ == "__main__":
    main()
