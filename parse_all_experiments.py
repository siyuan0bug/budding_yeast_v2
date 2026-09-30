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
    exp_dir = os.path.dirname(os.path.dirname(metric_file))
    
    try:
        with open(metric_file, "r") as f:
            data = json.load(f)
            
        real_m = data.get("real_mutants", {})
        synth_m = data.get("synth_test", {})
        
        # Check if per_mutant_metrics.csv exists
        per_mutant_csv = os.path.join(os.path.dirname(metric_file), "per_mutant_metrics.csv")
        has_per_mutant = os.path.exists(per_mutant_csv)
        
        # check number of real_mutant images
        real_mutant_img_dir = os.path.join(os.path.dirname(metric_file), "real_mutants")
        n_imgs = len(os.listdir(real_mutant_img_dir)) if os.path.exists(real_mutant_img_dir) else 0
        
        results.append({
            "model": model,
            "real_mae": real_m.get("MAE", np.nan),
            "real_rel_l2": real_m.get("Relative L2", np.nan),
            "real_corr": real_m.get("Correlation", np.nan),
            "real_mse": real_m.get("MSE", np.nan),
            "synth_mae": synth_m.get("MAE", np.nan),
            "synth_rel_l2": synth_m.get("Relative L2", np.nan),
            "synth_corr": synth_m.get("Correlation", np.nan),
            "synth_mse": synth_m.get("MSE", np.nan),
            "n_real_imgs": n_imgs,
            "per_mutant_csv": per_mutant_csv if has_per_mutant else None,
            "exp_dir": exp_dir
        })
    except Exception as e:
        print(f"Error reading {metric_file}: {e}")

df = pd.DataFrame(results)
print(f"Loaded {len(df)} evaluated models.")
# Sort by real_rel_l2
df_sorted = df.sort_values(by="real_rel_l2")
print("\n=== TOP 20 MODELS BY REAL MUTANTS RELATIVE L2 ===")
print(df_sorted[["model", "real_rel_l2", "real_mae", "real_corr", "synth_rel_l2", "synth_mae", "n_real_imgs"]].head(20).to_string())

print("\n=== BOTTOM 10 MODELS BY REAL MUTANTS RELATIVE L2 ===")
print(df_sorted[["model", "real_rel_l2", "real_mae", "real_corr", "synth_rel_l2", "synth_mae", "n_real_imgs"]].tail(10).to_string())

# Save summary to csv
df_sorted.to_csv("all_models_summary.csv", index=False)
