import numpy as np
import pandas as pd
import os

# Define file paths
npz_full_path = '/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast/lhs_cheat_origin_v5_210min_500steps_dual_labels.npz'
npz_no_target_path = '/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast/lhs_cheat_origin_v5_no_target_210min_500steps_dual_labels.npz'

csv_full_path = '/home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/eval_result/hyper_fno_none_mse_only_v5_modes32_seed42/per_mutant_metrics.csv'
csv_no_target_path = '/home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/eval_result/hyper_fno_none_mse_only_v5_no_target_modes32_seed42/per_mutant_metrics.csv'

output_path = '/home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/v5_dataset_comparison.csv'

# Read CSV files
df_full = pd.read_csv(csv_full_path)
df_no_target = pd.read_csv(csv_no_target_path)

# Build a dictionary of real mutants and their MSE
full_mse = dict(zip(df_full['Mutant'], df_full['MSE']))
no_target_mse = dict(zip(df_no_target['Mutant'], df_no_target['MSE']))

# Function to parse mutant names and group into families
def get_family_counts(mutant_names):
    normal_lhs_counts = {}
    target_lhs_counts = {}
    
    for name in mutant_names:
        if '_LHS_T' in name:
            base_name = name.split('_LHS_T')[0]
            target_lhs_counts[base_name] = target_lhs_counts.get(base_name, 0) + 1
        elif '_LHS_' in name:
            base_name = name.split('_LHS_')[0]
            normal_lhs_counts[base_name] = normal_lhs_counts.get(base_name, 0) + 1
            
    return normal_lhs_counts, target_lhs_counts

# Load npz files
print("Loading npz files...")
data_full = np.load(npz_full_path, allow_pickle=True)
data_no_target = np.load(npz_no_target_path, allow_pickle=True)

names_full = data_full['mutant_names']
names_no_target = data_no_target['mutant_names']

# Get counts
full_normal, full_target = get_family_counts(names_full)
no_target_normal, no_target_target = get_family_counts(names_no_target)

# Get all real mutants
all_real_mutants = sorted(list(set(df_full['Mutant']).union(set(df_no_target['Mutant']))), key=lambda x: int(x.split('_')[0].split('.')[0]) if x[0].isdigit() else 999)

# Build the final table
rows = []
for mutant in all_real_mutants:
    # Full dataset counts
    f_norm = full_normal.get(mutant, 0)
    f_targ = full_target.get(mutant, 0)
    f_total = f_norm + f_targ
    full_count_str = f"{f_total} (Target: {f_targ})" if f_targ > 0 else str(f_total)
    
    # No target dataset counts
    nt_norm = no_target_normal.get(mutant, 0)
    nt_targ = no_target_target.get(mutant, 0)
    nt_total = nt_norm + nt_targ
    nt_count_str = str(nt_total)
    
    # Eval results
    f_eval = f"{full_mse.get(mutant, 'N/A'):.6f}" if mutant in full_mse else "N/A"
    nt_eval = f"{no_target_mse.get(mutant, 'N/A'):.6f}" if mutant in no_target_mse else "N/A"
    
    rows.append({
        'real_mutant样本名称': mutant,
        '全量样本的家族数量 (含Target)': full_count_str,
        'no_target的家族样本数量': nt_count_str,
        '全量样本的eval结果(MSE)': f_eval,
        'no_target的eval结果(MSE)': nt_eval
    })

df_output = pd.DataFrame(rows)
df_output.to_csv(output_path, index=False)
print(f"Table saved to {output_path}")

# Also output markdown table to stdout for the chat response
print("\n" + df_output.to_markdown(index=False))
