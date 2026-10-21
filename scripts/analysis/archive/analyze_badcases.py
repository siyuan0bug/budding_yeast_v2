import os
import glob
import pandas as pd
import numpy as np

base_dir = "/home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/experiments/mutant_aware_pure_random_100000_210m_500s"

key_models = [
    "biom_refiner_plus",
    "bregman_wno",
    "konode",
    "icon",
    "unet1d",
    "fno1d",
    "gfno",
    "bregman_fno",
    "ncflow",
    "sc_fno",
    "autoformer",
    "wdno",
    "noble",
    "linear"
]

dfs = {}
for m in key_models:
    files = glob.glob(os.path.join(base_dir, m, "**/eval/per_mutant_metrics.csv"), recursive=True)
    if files:
        f = sorted(files)[-1]
        df_m = pd.read_csv(f)
        dfs[m] = df_m

print(f"Successfully loaded per-mutant metrics for {len(dfs)} key models.")

# Combine Relative_L2 for each mutant across models
mutants = dfs["biom_refiner_plus"]["Mutant"].tolist()
patterns = dfs["biom_refiner_plus"]["Pattern"].tolist()

combined_l2 = pd.DataFrame({"Mutant": mutants, "Pattern": patterns})

for m, df_m in dfs.items():
    # map by Mutant name
    m_dict = dict(zip(df_m["Mutant"], df_m["Relative_L2"]))
    combined_l2[m] = combined_l2["Mutant"].map(m_dict)

# Calculate mean error across models for each mutant
model_cols = list(dfs.keys())
combined_l2["Mean_Rel_L2"] = combined_l2[model_cols].mean(axis=1)
combined_l2["Median_Rel_L2"] = combined_l2[model_cols].median(axis=1)
combined_l2["BiomRefiner_Rel_L2"] = combined_l2["biom_refiner_plus"]
combined_l2["BregmanWNO_Rel_L2"] = combined_l2["bregman_wno"]
combined_l2["BestBaseline_Rel_L2"] = combined_l2[["bregman_wno", "konode", "icon", "unet1d", "fno1d"]].min(axis=1)

# Sort by Mean_Rel_L2 to find universally hardest mutants
sorted_hardest = combined_l2.sort_values(by="Mean_Rel_L2", ascending=False)

print("\n" + "="*80)
print("TOP 15 UNIVERSALLY HARDEST BAD CASES (Highest Mean Relative L2 across models):")
print("="*80)
cols_to_print = ["Mutant", "Pattern", "Mean_Rel_L2", "biom_refiner_plus", "bregman_wno", "konode", "unet1d", "fno1d", "autoformer"]
print(sorted_hardest[cols_to_print].head(15).to_string(index=False))

print("\n" + "="*80)
print("ERROR BY BIOLOGICAL PATTERN (Pattern A: Multi-div, Pattern B: Single-div, Pattern C: Arrest):")
print("="*80)
pattern_group = combined_l2.groupby("Pattern")[model_cols].mean()
print(pattern_group.T.to_string())

# Find mutants where biom_refiner_plus strongly outperforms baselines
combined_l2["Refiner_Advantage"] = combined_l2["BestBaseline_Rel_L2"] - combined_l2["BiomRefiner_Rel_L2"]
top_advantage = combined_l2.sort_values(by="Refiner_Advantage", ascending=False)

print("\n" + "="*80)
print("TOP 10 MUTANTS WHERE BIOM_REFINER_PLUS MOST STRONGLY OUTPERFORMS BASELINES:")
print("="*80)
print(top_advantage[["Mutant", "Pattern", "Refiner_Advantage", "BiomRefiner_Rel_L2", "BestBaseline_Rel_L2"]].head(10).to_string(index=False))

# Find mutants where even biom_refiner_plus has high error (True system-level hard cases)
biom_badcases = combined_l2.sort_values(by="BiomRefiner_Rel_L2", ascending=False)
print("\n" + "="*80)
print("TOP 10 BAD CASES SPECIFICALLY FOR BIOM_REFINER_PLUS:")
print("="*80)
print(biom_badcases[["Mutant", "Pattern", "BiomRefiner_Rel_L2", "bregman_wno", "konode", "unet1d", "Mean_Rel_L2"]].head(10).to_string(index=False))

combined_l2.to_csv("mutant_cross_model_comparison.csv", index=False)
