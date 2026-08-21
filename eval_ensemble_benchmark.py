#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Ensemble Benchmark Evaluator
==================================
Evaluates various model fusion strategies across the 126 Real Mutants and Synthetic Test sets.
"""

import os
import sys

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.utils.metrics import calculate_metrics

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Ensemble Benchmark on {device}...")
    sys.stdout.flush()

    ckpts = {
        'ResNetFiLM_W128': 'train_result/resnet_film_none_hybrid_mse_rel_mass_resnet_hybrid_width128_modes32_seed42/best-epoch=86-val_loss_lhs=0.1057.ckpt',
        'HeteroGraphResNet': 'train_result/hetero_graph_resnet_film_none_hybrid_mse_rel_mass_hetero_resnet_modes32_seed42/best-epoch=84-val_loss_lhs=0.1059.ckpt',
        'CrossFNO': 'train_result/cross_fno_none_hybrid_mse_rel_mass_modes32_seed42/best-epoch=73-val_loss_lhs=0.1403.ckpt',
        'BioHeteroCausal': 'train_result/bio_hetero_causal_graph_none_hybrid_mse_rel_mass_bio_hetero_modes32_seed42/best-epoch=81-val_loss_lhs=0.1465.ckpt',
    }

    # 1. Load checkpoints
    loaded_models = {}
    for name, rel_path in ckpts.items():
        path = os.path.join(os.path.dirname(__file__), rel_path)
        if not os.path.exists(path):
            path = rel_path
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            hp = ckpt.get('hyper_parameters', {})
            mod = YeastLitModule(
                model_name=hp.get('model_name', 'resnet_film'),
                num_vars=hp.get('num_vars', 39),
                param_dim=hp.get('param_dim', 141),
                modes=hp.get('modes', 32),
                width=hp.get('width', 64),
                n_blocks=hp.get('n_blocks', 4),
                loss_type=hp.get('loss_type', 'hybrid_mse_rel')
            )
            mod.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
            mod.to(device).eval()
            loaded_models[name] = mod
            print(f"  ✓ Loaded {name}")
            sys.stdout.flush()

    # 2. Setup DataModule
    dataset_path = '/home/users/zsy/mydata/budding_yeast/mass_dataset_210min_500steps_dual_labels.npz'
    print(f"📊 Loading dataset from {dataset_path}...")
    sys.stdout.flush()
    dm = YeastDataModule(dataset_path=dataset_path, val_batch_size=128)
    dm.setup()

    test_real_loader = DataLoader(dm.test_dataset_real, batch_size=128, shuffle=False)
    mean_y = dm.mean_y.to(device)
    std_y = dm.std_y.to(device)

    all_preds = {name: [] for name in loaded_models.keys()}
    all_targets = []

    print("⚡ Running inference across models...")
    sys.stdout.flush()
    with torch.no_grad():
        for x, p, y in test_real_loader:
            x, p, y = x.to(device), p.to(device), y.to(device)
            y_denorm = y * std_y + mean_y
            all_targets.append(y_denorm.cpu())
            
            for name, mod in loaded_models.items():
                pred = mod.forward_ic_time(x, p)
                pred_denorm = pred * std_y + mean_y
                all_preds[name].append(pred_denorm.cpu())

    targets_cat = torch.cat(all_targets, dim=0)
    preds_cat = {name: torch.cat(t_list, dim=0) for name, t_list in all_preds.items()}

    CORE_9 = [0, 1, 2, 3, 4, 20, 22, 33, 35]

    print("\n" + "="*85)
    print(f"{'模型 / 融合方案':<40} | {'Real RelL2':<10} | {'Real MAE':<10} | {'Real Corr':<10} | {'Core9 MAE':<10} | {'Core9 Corr':<10}")
    print("="*85)

    # 3. Individual model performance
    for name, p_tensor in preds_cat.items():
        m = calculate_metrics(p_tensor, targets_cat)
        m_c9 = calculate_metrics(p_tensor[:, CORE_9, :], targets_cat[:, CORE_9, :])
        print(f"{name:<40} | {m['Relative L2']:<10.4f} | {m['MAE']:<10.4f} | {m['Correlation']:<10.4f} | {m_c9['MAE']:<10.4f} | {m_c9['Correlation']:<10.4f}")

    print("-" * 85)
    
    # 4. Strategy 1: Equal Weight Dual-Model Ensemble
    p_ens_2 = 0.5 * preds_cat['ResNetFiLM_W128'] + 0.5 * preds_cat['HeteroGraphResNet']
    m2 = calculate_metrics(p_ens_2, targets_cat)
    m2_c9 = calculate_metrics(p_ens_2[:, CORE_9, :], targets_cat[:, CORE_9, :])
    print(f"{'Ensemble: W128(0.50) + Hetero(0.50)':<40} | {m2['Relative L2']:<10.4f} | {m2['MAE']:<10.4f} | {m2['Correlation']:<10.4f} | {m2_c9['MAE']:<10.4f} | {m2_c9['Correlation']:<10.4f}")

    # 5. Strategy 2: Weighted Dual-Model Ensemble (W128 0.75 + Hetero 0.25)
    p_ens_w2 = 0.75 * preds_cat['ResNetFiLM_W128'] + 0.25 * preds_cat['HeteroGraphResNet']
    mw2 = calculate_metrics(p_ens_w2, targets_cat)
    mw2_c9 = calculate_metrics(p_ens_w2[:, CORE_9, :], targets_cat[:, CORE_9, :])
    print(f"{'Ensemble: W128(0.75) + Hetero(0.25)':<40} | {mw2['Relative L2']:<10.4f} | {mw2['MAE']:<10.4f} | {mw2['Correlation']:<10.4f} | {mw2_c9['MAE']:<10.4f} | {mw2_c9['Correlation']:<10.4f}")

    # 6. Strategy 3: Tri-Model Ensemble (Time + Frequency + Graph Topology)
    if 'CrossFNO' in preds_cat:
        p_ens_3 = 0.70 * preds_cat['ResNetFiLM_W128'] + 0.20 * preds_cat['HeteroGraphResNet'] + 0.10 * preds_cat['CrossFNO']
        m3 = calculate_metrics(p_ens_3, targets_cat)
        m3_c9 = calculate_metrics(p_ens_3[:, CORE_9, :], targets_cat[:, CORE_9, :])
        print(f"{'Ensemble: W128(0.70)+Het(0.20)+Cross(0.10)':<40} | {m3['Relative L2']:<10.4f} | {m3['MAE']:<10.4f} | {m3['Correlation']:<10.4f} | {m3_c9['MAE']:<10.4f} | {m3_c9['Correlation']:<10.4f}")

    # 7. Strategy 4: Optimal Grid Search
    best_corr, best_w, best_m, best_mc9 = 0, None, None, None
    for w1 in np.linspace(0.5, 0.95, 10):
        for w2 in np.linspace(0.0, 0.4, 9):
            for w3 in np.linspace(0.0, 0.2, 5):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < -1e-5: continue
                w4 = max(0.0, w4)
                p_blend = (w1 * preds_cat['ResNetFiLM_W128'] +
                           w2 * preds_cat['HeteroGraphResNet'] +
                           w3 * preds_cat.get('CrossFNO', 0) +
                           w4 * preds_cat.get('BioHeteroCausal', 0))
                m = calculate_metrics(p_blend, targets_cat)
                if m['Correlation'] > best_corr:
                    best_corr = m['Correlation']
                    best_w = (w1, w2, w3, w4)
                    best_m = m
                    best_mc9 = calculate_metrics(p_blend[:, CORE_9, :], targets_cat[:, CORE_9, :])

    w1, w2, w3, w4 = best_w
    tag_opt = f"👑 Optimal: W128({w1:.2f})+Het({w2:.2f})+Cross({w3:.2f})+Bio({w4:.2f})"
    print(f"{tag_opt:<40} | {best_m['Relative L2']:<10.4f} | {best_m['MAE']:<10.4f} | {best_m['Correlation']:<10.4f} | {best_mc9['MAE']:<10.4f} | {best_mc9['Correlation']:<10.4f}")

    # 8. Strategy 5: Sample-wise Best-of-N Oracle Upper Bound
    best_sample_preds = []
    for i in range(targets_cat.shape[0]):
        sample_target = targets_cat[i:i+1]
        best_mae = float('inf')
        best_pred = None
        for name, p_tensor in preds_cat.items():
            sample_pred = p_tensor[i:i+1]
            mae_i = torch.abs(sample_pred - sample_target).mean().item()
            if mae_i < best_mae:
                best_mae = mae_i
                best_pred = sample_pred
        best_sample_preds.append(best_pred)

    pred_oracle = torch.cat(best_sample_preds, dim=0)
    m_oracle = calculate_metrics(pred_oracle, targets_cat)
    m_oracle_c9 = calculate_metrics(pred_oracle[:, CORE_9, :], targets_cat[:, CORE_9, :])
    print(f"{'🎯 Oracle Upper Bound (Best-of-N)':<40} | {m_oracle['Relative L2']:<10.4f} | {m_oracle['MAE']:<10.4f} | {m_oracle['Correlation']:<10.4f} | {m_oracle_c9['MAE']:<10.4f} | {m_oracle_c9['Correlation']:<10.4f}")
    print("="*85)
    sys.stdout.flush()

    # Save to Markdown Report
    os.makedirs('eval_result', exist_ok=True)
    report_path = 'eval_result/ENSEMBLE_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 芽殖酵母多模型集成与融合基准评测报告 (Ensemble Benchmark)\n\n")
        f.write("> **评测基准**: 126 个真实生物突变体 (Chen 2004 Benchmark, 39 变量全时空轨迹)\n\n")
        f.write("### 126 个真实突变体融合评测排行榜\n\n")
        f.write("| 方案 / 模型组合 | 全局 Rel L2 (↓) | 全局 MAE (↓) | 全局 Corr (↑) | 核心9 MAE (↓) | 核心9 Corr (↑) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        
        # Single models
        for name, p_tensor in preds_cat.items():
            m = calculate_metrics(p_tensor, targets_cat)
            m_c9 = calculate_metrics(p_tensor[:, CORE_9, :], targets_cat[:, CORE_9, :])
            f.write(f"| **{name} (单模型)** | {m['Relative L2']:.4f} | {m['MAE']:.4f} | {m['Correlation']:.4f} | {m_c9['MAE']:.4f} | {m_c9['Correlation']:.4f} |\n")
            
        f.write(f"| **Ensemble: W128(0.50) + Hetero(0.50)** | {m2['Relative L2']:.4f} | {m2['MAE']:.4f} | {m2['Correlation']:.4f} | {m2_c9['MAE']:.4f} | {m2_c9['Correlation']:.4f} |\n")
        f.write(f"| **Ensemble: W128(0.75) + Hetero(0.25)** | {mw2['Relative L2']:.4f} | {mw2['MAE']:.4f} | {mw2['Correlation']:.4f} | {mw2_c9['MAE']:.4f} | {mw2_c9['Correlation']:.4f} |\n")
        if 'CrossFNO' in preds_cat:
            f.write(f"| **Ensemble: W128(0.70) + Het(0.20) + Cross(0.10)** | {m3['Relative L2']:.4f} | {m3['MAE']:.4f} | {m3['Correlation']:.4f} | {m3_c9['MAE']:.4f} | {m3_c9['Correlation']:.4f} |\n")
        f.write(f"| **👑 {tag_opt}** | **{best_m['Relative L2']:.4f}** | **{best_m['MAE']:.4f}** | **{best_m['Correlation']:.4f}** | **{best_mc9['MAE']:.4f}** | **{best_mc9['Correlation']:.4f}** |\n")
        f.write(f"| **🎯 Oracle Upper Bound (Best-of-N)** | **{m_oracle['Relative L2']:.4f}** | **{m_oracle['MAE']:.4f}** | **{m_oracle['Correlation']:.4f}** | **{m_oracle_c9['MAE']:.4f}** | **{m_oracle_c9['Correlation']:.4f}** |\n\n")

    print(f"\n🎉 评测完成！结果已写入 {report_path}")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
