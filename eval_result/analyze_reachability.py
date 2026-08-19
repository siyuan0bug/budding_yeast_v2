import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from budding_yeast_v2.data.mass_dataset_generator import REAL_MUTANTS, BINDING_GROUPS, get_default_params

df = pd.read_csv('budding_yeast_v2/eval_result/graph_resnet_film_none_hybrid_mse_rel_mass_graph_resnet_hybrid_modes32_seed42/per_mutant_metrics.csv')

# Criteria for bad cases: MAE > 0.15 OR Correlation < 0.80 OR Relative_L2 > 0.30
bad_df = df[(df['MAE'] > 0.15) | (df['Correlation'] < 0.80) | (df['Relative_L2'] > 0.30)].copy()
bad_df['Rank_Score'] = bad_df['MAE'] * 2.0 + (1.0 - bad_df['Correlation']) + bad_df['Relative_L2']
bad_df = bad_df.sort_values(by='Rank_Score', ascending=False)

print(f'Total bad cases identified: {len(bad_df)} / 126\n')

for idx, row in bad_df.iterrows():
    m_full = row['Mutant']
    m_num = m_full.split('_')[0]
    
    real_def = None
    for k, (name, overrides) in REAL_MUTANTS.items():
        if k == m_num or m_full.startswith(f"{k}_"):
            real_def = (name, overrides)
            break
            
    print(f"==================================================================================")
    print(f"🔬 Mutant #{m_num}: {m_full}")
    print(f"   Metrics: MAE={row['MAE']:.4f}, Corr={row['Correlation']:.4f}, Rel_L2={row['Relative_L2']:.4f}, Pattern={row['Pattern']}")
    if real_def:
        print(f"   Biological Overrides: {real_def[1]}")
    else:
        print(f"   Biological Overrides: Not found in REAL_MUTANTS")
