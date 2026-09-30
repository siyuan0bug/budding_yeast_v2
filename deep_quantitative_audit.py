import numpy as np
import pandas as pd

# Load dataset
d = np.load("/home/users/zsy/mydata/budding_yeast/mutant_aware_pure_random_100000_210m_500s.npz")
all_params = d["params"] # (100126, 141)
mutant_names = [str(x) for x in d["mutant_names"]]
pattern_labels = [str(x) for x in d["pattern_labels"]]

n_real = 126
real_params = all_params[:n_real]
synth_params = all_params[n_real:]

# Load cross model comparison
df_cross = pd.read_csv("mutant_cross_model_comparison.csv")

print(f"Total samples: {len(all_params)}, Real: {n_real}, Synth: {len(synth_params)}")

# Normalize parameters across dataset to compute fair Euclidean distance
p_mean = synth_params.mean(axis=0)
p_std = synth_params.std(axis=0)
p_std[p_std < 1e-6] = 1.0

synth_p_norm = (synth_params - p_mean) / p_std
real_p_norm = (real_params - p_mean) / p_std

# Compute distance from each real mutant to synthetic dataset
# To make it fast, subsample synth_p_norm by 10,000 for kNN distance
sub_idx = np.random.RandomState(42).choice(len(synth_p_norm), size=10000, replace=False)
synth_sub = synth_p_norm[sub_idx]

dists = []
for i in range(n_real):
    diff = synth_sub - real_p_norm[i:i+1] # (10000, 141)
    d = np.linalg.norm(diff, axis=1)
    min_dist = np.min(d)
    k10_dist = np.mean(np.partition(d, 10)[:10])
    dists.append({
        "Mutant": mutant_names[i],
        "Pattern": pattern_labels[i],
        "min_dist": min_dist,
        "k10_dist": k10_dist,
    })

df_dist = pd.DataFrame(dists)

# Merge with model performance
merged = pd.merge(df_cross, df_dist, on="Mutant")

print("\n=== CORRELATION BETWEEN PARAMETER DISTANCE TO SYNTHETIC DATA AND MODEL ERROR ===")
corr_mean = merged["k10_dist"].corr(merged["Mean_Rel_L2"])
corr_biom = merged["k10_dist"].corr(merged["biom_refiner_plus"])
corr_unet = merged["k10_dist"].corr(merged["unet1d"])
corr_breg = merged["k10_dist"].corr(merged["bregman_wno"])
print(f"Corr(k10_dist, Mean_Rel_L2): {corr_mean:.4f}")
print(f"Corr(k10_dist, biom_refiner_plus): {corr_biom:.4f}")
print(f"Corr(k10_dist, unet1d): {corr_unet:.4f}")
print(f"Corr(k10_dist, bregman_wno): {corr_breg:.4f}")

# Look at specific contrast pairs
test_mutants = [
    "10_bck2Δ", "13_cln3Δ_bck2Δ", "15_cln3Δ_bck2Δ_multi-copy_CLN2", "14_cln3Δ_bck2Δ_GAL-CLN2_cln1Δ_cln2Δ",
    "3_cln1Δ_cln2Δ", "17_cln1Δ_cln2Δ_cln3Δ", "18_cln1Δ_cln2Δ_cln3Δ_GAL-CLN2", "20_cln1Δ_cln2Δ_cln3Δ_sic1Δ", "22_cln1Δ_cln2Δ_cln3Δ_multi-copy_CLB5",
    "50_multi-copy_GAL-CLB2", "53_GAL-CLB2_sic1Δ", "54_GAL-CLB2_cdh1Δ"
]

print("\n=== CONTRASTIVE PAIRS: PARAMETER DISTANCE & ERROR ===")
cols = ["Mutant", "min_dist", "k10_dist", "biom_refiner_plus", "bregman_wno", "unet1d", "Mean_Rel_L2"]
print(merged[merged["Mutant"].isin(test_mutants)][cols].to_string(index=False))
