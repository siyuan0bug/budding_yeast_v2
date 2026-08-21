#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find and categorize badcases across 126 Real Mutants for top models.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.utils.metrics import calculate_metrics

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Starting Badcase Mining on {device}...", flush=True)

    import glob
    model_patterns = {
        'ResNetFiLM_W128': 'train_result/resnet_film_*width128*/*.ckpt',
        'ST_HyperNO': 'train_result/st_hyperno_*width128*/*.ckpt',
        'DyRFO': 'train_result/dyrfo_*width128*/*.ckpt',
        'HeteroResNet': 'train_result/hetero_graph_resnet_film_*/*.ckpt',
        'CrossFNO': 'train_result/cross_fno_*/*.ckpt',
    }

    # 1. Load models
    loaded_models = {}
    for name, pat in model_patterns.items():
        files = glob.glob(pat)
        if not files:
            files = glob.glob(os.path.join(os.path.dirname(__file__), pat))
        if files:
            path = files[0]
            ckpt = torch.load(path, map_location=device, weights_only=False)
            hp = ckpt.get('hyper_parameters', {})
            mod = YeastLitModule(
                model_name=hp.get('model_name', 'resnet_film'),
                num_vars=hp.get('num_vars', 39),
                param_dim=hp.get('param_dim', 141),
                modes=hp.get('modes', 32),
                width=hp.get('width', 128 if 'width128' in path else 64),
                n_blocks=hp.get('n_blocks', 4),
                loss_type=hp.get('loss_type', 'hybrid_mse_rel')
            )
            mod.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
            mod.to(device).eval()
            loaded_models[name] = mod
            print(f"  ✓ Loaded {name} from {os.path.basename(path)}", flush=True)

    dataset_path = '/home/users/zsy/mydata/budding_yeast/mass_dataset_210min_500steps_dual_labels.npz'
    raw_data = np.load(dataset_path, allow_pickle=True)
    mutant_names = raw_data['mutant_names']
    phenotypes = raw_data['phenotypes'] if 'phenotypes' in raw_data else np.array(['Unknown'] * len(mutant_names))
    real_mask = np.array([not str(name).startswith('MASS_') for name in mutant_names])
    real_names = mutant_names[real_mask]
    real_phenos = phenotypes[real_mask] if 'phenotypes' in raw_data else np.array(['Unknown'] * len(real_names))

    print(f"📊 Setting up datamodule (Real Mutants count: {len(real_names)})...", flush=True)
    dm = YeastDataModule(dataset_path=dataset_path, val_batch_size=128)
    dm.setup()
    test_real_loader = DataLoader(dm.test_dataset_real, batch_size=128, shuffle=False)
    mean_y = dm.mean_y.to(device)
    std_y = dm.std_y.to(device)

    all_preds = {name: [] for name in loaded_models.keys()}
    all_targets = []

    print("⚡ Running inference...", flush=True)
    with torch.no_grad():
        for x, p, y in test_real_loader:
            x, p, y = x.to(device), p.to(device), y.to(device)
            y_denorm = y * std_y + mean_y
            all_targets.append(y_denorm.cpu())
            for name, mod in loaded_models.items():
                pred = mod.forward_ic_time(x, p)
                pred_denorm = pred * std_y + mean_y
                all_preds[name].append(pred_denorm.cpu())

    targets = torch.cat(all_targets, dim=0) # (126, 39, 500)
    preds = {name: torch.cat(t_list, dim=0) for name, t_list in all_preds.items()}

    CORE_9 = [0, 1, 2, 3, 4, 20, 22, 33, 35]

    # Sample-wise metrics
    records = []
    for i in range(len(real_names)):
        name_i = str(real_names[i])
        pheno_i = str(real_phenos[i])
        t_i = targets[i:i+1] # (1, 39, 500)
        
        rec = {'Index': i, 'Mutant_Name': name_i, 'Phenotype': pheno_i}
        for m_name, p_tensor in preds.items():
            p_i = p_tensor[i:i+1]
            m = calculate_metrics(p_i, t_i)
            rec[f'{m_name}_Corr'] = m['Correlation']
            rec[f'{m_name}_MAE'] = m['MAE']
            rec[f'{m_name}_RelL2'] = m['Relative L2']
            
            # Core 9
            m_c9 = calculate_metrics(p_i[:, CORE_9, :], t_i[:, CORE_9, :])
            rec[f'{m_name}_C9_Corr'] = m_c9['Correlation']
            rec[f'{m_name}_C9_MAE'] = m_c9['MAE']
        
        corrs = [rec[f'{m_name}_Corr'] for m_name in loaded_models.keys()]
        maes = [rec[f'{m_name}_MAE'] for m_name in loaded_models.keys()]
        rel_l2s = [rec[f'{m_name}_RelL2'] for m_name in loaded_models.keys()]
        rec['Avg_Corr'] = float(np.mean(corrs))
        rec['Avg_MAE'] = float(np.mean(maes))
        rec['Avg_RelL2'] = float(np.mean(rel_l2s))
        records.append(rec)

    df = pd.DataFrame(records)
    
    # Save results to json & csv
    os.makedirs('eval_result', exist_ok=True)
    df.to_csv('eval_result/real_mutants_badcases.csv', index=False)
    with open('eval_result/real_mutants_badcases.json', 'w') as f:
        json.dump(records, f, indent=2)

    print("\n" + "="*110, flush=True)
    print("      【126 个真实突变体 全模型平均表现最差的 Top-25 Badcases (按 Avg_Corr 升序排序)】", flush=True)
    print("="*110, flush=True)
    cols_show = ['Index', 'Mutant_Name', 'Avg_Corr', 'ResNetFiLM_W128_Corr', 'ST_HyperNO_Corr', 'DyRFO_Corr', 'Avg_MAE', 'ResNetFiLM_W128_MAE', 'Phenotype']
    df_worst_corr = df.sort_values(by='Avg_Corr', ascending=True)
    print(df_worst_corr[cols_show].head(25).to_string(index=False), flush=True)

    print("\n" + "="*110, flush=True)
    print("      【126 个真实突变体 全模型平均误差最大的 Top-25 Badcases (按 Avg_MAE 降序排序)】", flush=True)
    print("="*110, flush=True)
    cols_show_mae = ['Index', 'Mutant_Name', 'Avg_MAE', 'ResNetFiLM_W128_MAE', 'ST_HyperNO_MAE', 'DyRFO_MAE', 'Avg_RelL2', 'Avg_Corr', 'Phenotype']
    df_worst_mae = df.sort_values(by='Avg_MAE', ascending=False)
    print(df_worst_mae[cols_show_mae].head(25).to_string(index=False), flush=True)

    print("\n🎉 分析完成！Badcase 详细数据已持久化至 eval_result/real_mutants_badcases.csv", flush=True)

if __name__ == '__main__':
    main()
