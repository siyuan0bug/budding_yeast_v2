import os
import glob
import json
import pandas as pd
import numpy as np

base_dir = "/home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/experiments/mutant_aware_pure_random_100000_210m_500s"

results = []

for model in sorted(os.listdir(base_dir)):
    model_path = os.path.join(base_dir, model)
    if not os.path.isdir(model_path):
        continue
    
    # search for global_metrics_denorm.json
    metric_files = glob.glob(os.path.join(model_path, "**/eval/global_metrics_denorm.json"), recursive=True)
    if not metric_files:
        continue
    
    # choose the metric file
    metric_file = sorted(metric_files)[-1]
    eval_dir = os.path.dirname(metric_file)
    exp_dir = os.path.dirname(eval_dir)
    
    try:
        with open(metric_file, "r") as f:
            data = json.load(f)
            
        real_m = data.get("real_mutants", {})
        real_all = real_m.get("all_39_vars", {})
        real_core = real_m.get("core_9_vars", {})
        
        synth_m = data.get("lhs_test", data.get("synth_test", {}))
        synth_all = synth_m.get("all_39_vars", {})
        synth_core = synth_m.get("core_9_vars", {})
        
        # Check if per_mutant_metrics.csv exists
        per_mutant_csv = os.path.join(eval_dir, "per_mutant_metrics.csv")
        has_per_mutant = os.path.exists(per_mutant_csv)
        
        # check number of real_mutant images
        real_mutant_img_dir = os.path.join(eval_dir, "real_mutants")
        n_imgs = len(os.listdir(real_mutant_img_dir)) if os.path.exists(real_mutant_img_dir) else 0
        
        results.append({
            "model": model,
            "real_rel_l2": real_all.get("Relative L2", np.nan),
            "real_mae": real_all.get("MAE", np.nan),
            "real_corr": real_all.get("Correlation", np.nan),
            "real_core_rel_l2": real_core.get("Relative L2", np.nan),
            "real_core_mae": real_core.get("MAE", np.nan),
            "real_core_corr": real_core.get("Correlation", np.nan),
            "synth_rel_l2": synth_all.get("Relative L2", np.nan),
            "synth_mae": synth_all.get("MAE", np.nan),
            "synth_corr": synth_all.get("Correlation", np.nan),
            "n_real_imgs": n_imgs,
            "per_mutant_csv": per_mutant_csv if has_per_mutant else None,
            "exp_dir": exp_dir
        })
    except Exception as e:
        print(f"Error reading {metric_file}: {e}")

df = pd.DataFrame(results)
df_sorted = df.sort_values(by="real_rel_l2")
df_sorted.to_csv("benchmark_49_models_full.csv", index=False)

print("\n=== TOP 25 MODELS BY REAL MUTANTS RELATIVE L2 (ALL 39 VARS) ===")
cols_show = ["model", "real_rel_l2", "real_mae", "real_corr", "real_core_rel_l2", "synth_rel_l2", "synth_mae", "n_real_imgs"]
print(df_sorted[cols_show].head(25).to_string(index=False))

print("\n=== BOTTOM 15 MODELS BY REAL MUTANTS RELATIVE L2 ===")
print(df_sorted[cols_show].tail(15).to_string(index=False))
