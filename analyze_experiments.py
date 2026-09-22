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
    """
    计算单个超参组多 Seed 的完整统计均值与标准差。
    全面覆盖 real_mutants, lhs_test, all_test 以及 all_39_vars 与 core_9_vars 的所有指标。
    """
    seeds = group_entry["seeds"]
    n_seeds = len(seeds)

    # 1. 结构化汇总全部指标
    # 结构: stats[split][var_set][metric] = [val_seed1, val_seed2, ...]
    splits = ["real_mutants", "synth_test", "all_test"]
    var_subsets = ["all_39_vars", "core_9_vars"]
    metric_keys = ["MAE", "Relative L2", "Correlation", "MSE"]

    collected_metrics = {sp: {vs: {mk: [] for mk in metric_keys} for vs in var_subsets} for sp in splits}
    sample_counts = {sp: None for sp in splits}

    for s in seeds:
        m = s["metrics"]
        # 平滑兼容：若旧实验只有 lhs_test，则映射为 synth_test
        if "synth_test" not in m and "lhs_test" in m:
            m["synth_test"] = m["lhs_test"]
        for sp in splits:
            if sp in m and isinstance(m[sp], dict):
                if sample_counts[sp] is None and "count" in m[sp]:
                    sample_counts[sp] = m[sp]["count"]
                for vs in var_subsets:
                    if vs in m[sp] and isinstance(m[sp][vs], dict):
                        for mk in metric_keys:
                            if mk in m[sp][vs]:
                                try:
                                    val = float(m[sp][vs][mk])
                                    collected_metrics[sp][vs][mk].append(val)
                                except (ValueError, TypeError):
                                    pass

    # 2. 生成结构化 summary_metrics 字典
    summary_metrics = {}
    for sp in splits:
        summary_metrics[sp] = {}
        for vs in var_subsets:
            summary_metrics[sp][vs] = {}
            for mk in metric_keys:
                vals = collected_metrics[sp][vs][mk]
                if vals:
                    summary_metrics[sp][vs][mk] = {
                        "mean": float(np.nanmean(vals)),
                        "std": float(np.nanstd(vals)) if len(vals) > 1 else 0.0
                    }
        if sample_counts[sp] is not None:
            summary_metrics[sp]["count"] = sample_counts[sp]

    # 3. 构造扁平快捷指标 (方便排行榜与快速索引)
    real_39 = summary_metrics.get("real_mutants", {}).get("all_39_vars", {})
    real_c9 = summary_metrics.get("real_mutants", {}).get("core_9_vars", {})
    synth_39 = summary_metrics.get("synth_test", {}).get("all_39_vars", {}) or summary_metrics.get("lhs_test", {}).get("all_39_vars", {})
    all_39 = summary_metrics.get("all_test", {}).get("all_39_vars", {})

    summary = {
        "dataset": group_entry["dataset_tag"],
        "model": group_entry["model_name"],
        "loss_strategy": group_entry["loss_strategy"],
        "hparam_signature": group_entry["hparam_sig"],
        "n_seeds": n_seeds,
        "seeds_evaluated": [s["seed"] for s in seeds],
        
        # Real Mutants (全部 39 变量)
        "real_mae_mean": real_39.get("MAE", {}).get("mean", float("nan")),
        "real_mae_std": real_39.get("MAE", {}).get("std", 0.0),
        "real_l2_mean": real_39.get("Relative L2", {}).get("mean", float("nan")),
        "real_l2_std": real_39.get("Relative L2", {}).get("std", 0.0),
        "real_corr_mean": real_39.get("Correlation", {}).get("mean", float("nan")),
        "real_corr_std": real_39.get("Correlation", {}).get("std", 0.0),
        "real_mse_mean": real_39.get("MSE", {}).get("mean", float("nan")),
        "real_mse_std": real_39.get("MSE", {}).get("std", 0.0),

        # Real Mutants (核心 9 变量)
        "real_core9_mae_mean": real_c9.get("MAE", {}).get("mean", float("nan")),
        "real_core9_mae_std": real_c9.get("MAE", {}).get("std", 0.0),
        "real_core9_l2_mean": real_c9.get("Relative L2", {}).get("mean", float("nan")),
        "real_core9_l2_std": real_c9.get("Relative L2", {}).get("std", 0.0),
        "real_core9_corr_mean": real_c9.get("Correlation", {}).get("mean", float("nan")),
        "real_core9_corr_std": real_c9.get("Correlation", {}).get("std", 0.0),

        # Synthetic Test & All Test 关键指标
        "synth_mae_mean": synth_39.get("MAE", {}).get("mean", float("nan")),
        "synth_mae_std": synth_39.get("MAE", {}).get("std", 0.0),
        "all_test_mae_mean": all_39.get("MAE", {}).get("mean", float("nan")),
        "all_test_mae_std": all_39.get("MAE", {}).get("std", 0.0),

        # 完整分层结构化指标
        "detailed_metrics": summary_metrics
    }

    # 若是分层目录，写入到 Level 4 超参根目录下
    if group_entry["type"] == "hierarchical" and os.path.exists(group_entry["hparam_dir"]):
        sum_path = os.path.join(group_entry["hparam_dir"], "seed_summary.json")
        try:
            with open(sum_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 保存 {sum_path} 失败: {e}")

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
