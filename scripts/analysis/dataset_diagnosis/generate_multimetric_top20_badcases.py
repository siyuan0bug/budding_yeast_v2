#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Metric Top-20 Badcase Full Extraction & Visualization Suite
================================================================================
Generates detailed Top-20 Badcase rankings and training set attribution for:
1. Lowest Correlation (最差相关性 Top-20)
2. Highest MAE (最大绝对误差 Top-20)
3. Highest Relative L2 (最大相对 L2 误差 Top-20)
4. Highest MSE (最大均方误差 Top-20)
5. Comprehensive Cross-Metric Union Badcases (全指标重叠总榜)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
V2_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
PROJECT_ROOT = os.path.dirname(V2_ROOT)
DATA_DIR = os.path.join(V2_ROOT, 'data')

for p in [PROJECT_ROOT, V2_ROOT, DATA_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

OUTPUT_DIR = os.path.join(V2_ROOT, 'scripts', 'analysis', 'outputs', 'multimetric_top20_badcases')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def format_table(df_sub, metric_col, metric_label, ascending=True):
    lines = []
    lines.append(f"| 排名 | 突变体编号与名称 | 碳源 | 突变基因数 | **{metric_label}** | Correlation | Relative L2 | MAE | 80K 训练集精确样本数 | 80K 训练集子集覆盖数 | 141维 k-NN 距离 | 根因归属诊断 |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    
    df_sorted = df_sub.sort_values(metric_col, ascending=ascending).head(20)
    for rank, (_, r) in enumerate(df_sorted.iterrows(), 1):
        m_val = f"{r[metric_col]:.4f}"
        exact_badge = f"**{r['Exact_Train_Count']}**" if r['Exact_Train_Count'] > 0 else "❌ **0 (Zero-Shot)**"
        lines.append(
            f"| {rank} | `{r['Mutant_Name']}` | {r['Carbon']} | {r['Num_Mutated_Genes']} | **{m_val}** | {r['Correlation']:.4f} | {r['Relative_L2']:.4f} | {r['MAE']:.4f} | {exact_badge} | {r['Subset_Train_Count']} | {r['Param_d_k20']:.2f} | {r['Primary_Attribution_Verdict']} |"
        )
    return "\n".join(lines)

def main():
    candidate_csvs = [
        os.path.join(V2_ROOT, 'data', 'dataset_analysis', 'massv2_biom_refiner_badcase_train_attribution.csv'),
        os.path.join(V2_ROOT, 'archive', 'dataset_analysis_reports', 'massv2_biom_refiner_badcase_train_attribution.csv'),
        os.path.join(V2_ROOT, 'scripts', 'analysis', 'outputs', 'massv2_badcase_attribution', 'massv2_biom_refiner_badcase_train_attribution.csv'),
    ]
    csv_path = next((c for c in candidate_csvs if os.path.exists(c)), None)
    if not csv_path:
        raise FileNotFoundError(f"Attribution CSV not found in candidate paths: {candidate_csvs}")
    df = pd.read_csv(csv_path)

    # 1. Generate Individual Top-20 CSVs
    top20_corr = df.sort_values('Correlation', ascending=True).head(20)
    top20_mae = df.sort_values('MAE', ascending=False).head(20)
    top20_rel = df.sort_values('Relative_L2', ascending=False).head(20)
    top20_mse = df.sort_values('MSE', ascending=False).head(20)

    top20_corr.to_csv(os.path.join(OUTPUT_DIR, 'top20_worst_correlation.csv'), index=False)
    top20_mae.to_csv(os.path.join(OUTPUT_DIR, 'top20_worst_mae.csv'), index=False)
    top20_rel.to_csv(os.path.join(OUTPUT_DIR, 'top20_worst_relative_l2.csv'), index=False)
    top20_mse.to_csv(os.path.join(OUTPUT_DIR, 'top20_worst_mse.csv'), index=False)

    # Union of all badcases in top-20 across 4 metrics
    union_indices = set(top20_corr['Index']).union(set(top20_mae['Index'])).union(set(top20_rel['Index'])).union(set(top20_mse['Index']))
    df_union = df[df['Index'].isin(union_indices)].copy()
    df_union = df_union.sort_values('Severity_Score', ascending=False)
    df_union.to_csv(os.path.join(OUTPUT_DIR, 'all_metrics_union_badcases.csv'), index=False)

    print(f"📊 Extracted Top-20 Badcases for all metrics: Total Unique Union Badcases = {len(df_union)}")

    # 2. Render 4-Panel Visualization Figure
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))

    # Panel 1: Lowest Correlation
    ax = axes[0, 0]
    y_pos = np.arange(20)
    ax.barh(y_pos, top20_corr['Correlation'], color='#e74c3c', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['Mutant_Name'][:24]} (n={r['Exact_Train_Count']})" for _, r in top20_corr.iterrows()], fontsize=9)
    ax.set_title('Top-20 Lowest Correlation (Worst Trend/Phase Fit)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Correlation (↑)', fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    for i, v in enumerate(top20_corr['Correlation']):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=8, fontweight='bold')

    # Panel 2: Highest MAE
    ax = axes[0, 1]
    ax.barh(y_pos, top20_mae['MAE'], color='#e67e22', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['Mutant_Name'][:24]} (n={r['Exact_Train_Count']})" for _, r in top20_mae.iterrows()], fontsize=9)
    ax.set_title('Top-20 Highest MAE (Worst Mean Absolute Error)', fontsize=13, fontweight='bold')
    ax.set_xlabel('MAE (↓)', fontsize=11)
    ax.invert_yaxis()
    for i, v in enumerate(top20_mae['MAE']):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=8, fontweight='bold')

    # Panel 3: Highest Relative L2
    ax = axes[1, 0]
    ax.barh(y_pos, top20_rel['Relative_L2'], color='#9b59b6', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['Mutant_Name'][:24]} (n={r['Exact_Train_Count']})" for _, r in top20_rel.iterrows()], fontsize=9)
    ax.set_title('Top-20 Highest Relative L2 (Worst Signal Ratio Error)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Relative L2 (↓)', fontsize=11)
    ax.invert_yaxis()
    for i, v in enumerate(top20_rel['Relative_L2']):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=8, fontweight='bold')

    # Panel 4: Highest MSE
    ax = axes[1, 1]
    ax.barh(y_pos, top20_mse['MSE'], color='#c0392b', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['Mutant_Name'][:24]} (n={r['Exact_Train_Count']})" for _, r in top20_mse.iterrows()], fontsize=9)
    ax.set_title('Top-20 Highest MSE (Worst Peak Squared Error)', fontsize=13, fontweight='bold')
    ax.set_xlabel('MSE (↓)', fontsize=11)
    ax.invert_yaxis()
    for i, v in enumerate(top20_mse['MSE']):
        ax.text(v + 0.2, i, f"{v:.2f}", va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig_all_metrics_top20_badcases.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved 4-panel figure to: {fig_path}")

    # 3. Update Multi-Metric Section in Report
    t1_md = format_table(df, 'Correlation', 'Correlation (↑)', ascending=True)
    t2_md = format_table(df, 'MAE', 'MAE 绝对误差 (↓)', ascending=False)
    t3_md = format_table(df, 'Relative_L2', 'Relative L2 (↓)', ascending=False)
    t4_md = format_table(df, 'MSE', 'MSE 均方误差 (↓)', ascending=False)

    report_path = os.path.join(OUTPUT_DIR, 'MASS_V2_ALL_METRICS_TOP20_REPORT.md')
    report_text = f"""# MASS v2 数据集下各评估指标 Top-20 最差 Badcase 全景统计大字典

> **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
> **评测模型**: `biom_refiner_graph_none_hybrid_mse_rel_massv2_biom_refiner_graph_hybrid_width128_modes32_seed42`  
> **数据集**: `mass_v2_dataset_210min_500steps_dual_labels.npz` (80K 训练集, seed=42)  
> **报告目录**: `budding_yeast_v2/data/dataset_analysis/`

---

## 🖼️ 四大指标 Top-20 全景可视化图谱
![四大指标 Top-20 对比图](figures/fig_all_metrics_top20_badcases.png)

---

## 1. 最差【相关性 Correlation】Top-20 排行榜 (反映波形趋势与相位失真)
*相关性越低，说明模型对该突变体的周期振荡相位或波形趋势预测越差。*

{t1_md}

---

## 2. 最差【平均绝对误差 MAE】Top-20 排行榜 (反映全局平均幅值偏差)
*MAE 越大，说明模型预测的浓度绝对值与真实值偏离越严重（124号高居榜首）。*

{t2_md}

---

## 3. 最差【相对 L2 误差 Relative L2】Top-20 排行榜 (反映相对比例失真)
*相对 L2 误差越大，说明预测波形与真实轨迹的整体能量/欧氏距离偏差占真实信号比例越大。*

{t3_md}

---

## 4. 最差【均方误差 MSE】Top-20 排行榜 (对极端尖峰误差惩罚最大)
*MSE 极大通常由于超强过表达（如 GAL-CLB2、GAL-CDC14）出现的极端大数值尖峰引起。*

{t4_md}

---

## 5. 全指标交叉重叠汇总分析 (Cross-Metric Findings)

1. **“四料最差”的核心死角**：
   - **`101_GAL-NET1_GAL-CDC14`**：在 Correlation (0.1273, 第1差)、Relative L2 (0.7244, 第1差)、MAE (0.6077, 第2差)、MSE (3.66, 第4差) **四大指标全部进入最差 Top-4**！80K 训练集中仅有 **1 个样本**。
   - **`42_sic1Δ_cdc6Δ_2-49_cdh1Δ_GAL-CDC20`**：在 Correlation (0.1912, 第2差)、Relative L2 (0.4813, 第2差)、MAE (0.2725, 第4差) 均名列前茅，4 基因高阶复合突变流形孤立。
2. **“幅值爆炸型”极端 Badcase**：
   - **`124_APC-A_GAL-CLB2`**：MAE (0.8739) 与 MSE (42.42) **均夺得全场断层第 1 差**！原因是 Clb2 极端过表达致死，波形冲至 8.0 浓度，模型预测幅值被常规范围拉低。
   - **`62_GAL-CLB2-dbΔ`**：MAE (0.2887, 第3差)、MSE (5.35, 第2差)，训练集中完全缺失该破坏框组合（**0 样本 Zero-Shot**）。
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Saved Full Multi-Metric Report to: {report_path}")

if __name__ == '__main__':
    main()
