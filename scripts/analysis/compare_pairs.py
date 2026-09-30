import pandas as pd
import numpy as np

# Load per-mutant metrics
df_refiner = pd.read_csv("experiments/mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/per_mutant_metrics.csv")
df_bregman = pd.read_csv("experiments/mutant_aware_pure_random_100000_210m_500s/bregman_wno/hybrid_mse_rel_none/w192_lev3_haar_b256_ep100_lr8.8e-4_wm8/seed_42/eval/per_mutant_metrics.csv")
df_unet = pd.read_csv("experiments/mutant_aware_pure_random_100000_210m_500s/unet1d/hybrid_mse_rel_none/w64_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/per_mutant_metrics.csv")
df_cross = pd.read_csv("mutant_cross_model_comparison.csv")

# Look at contrastive pairs:
pairs = [
    # Pair 1: cln3Δ bck2Δ family
    ["10_bck2Δ", "13_cln3Δ_bck2Δ", "15_cln3Δ_bck2Δ_multi-copy_CLN2", "14_cln3Δ_bck2Δ_GAL-CLN2_cln1Δ_cln2Δ", "16_cln3Δ_bck2Δ_sic1Δ"],
    # Pair 2: cln1Δ cln2Δ cln3Δ family
    ["3_cln1Δ_cln2Δ", "17_cln1Δ_cln2Δ_cln3Δ", "18_cln1Δ_cln2Δ_cln3Δ_GAL-CLN2", "20_cln1Δ_cln2Δ_cln3Δ_sic1Δ", "22_cln1Δ_cln2Δ_cln3Δ_multi-copy_CLB5"],
    # Pair 3: GAL-CLB2 family
    ["51_GAL-CLB2", "50_multi-copy_GAL-CLB2", "53_GAL-CLB2_sic1Δ", "54_GAL-CLB2_cdh1Δ", "55_GAL-CLB2_clb5Δ"],
    # Pair 4: APC-A family
    ["115_APC-A", "116_APC-A_cdh1Δ", "122_APC-A_cdh1Δ_multi-copy_CDC20", "123_APC-A_sic1Δ", "124_APC-A_GAL-CLB2"]
]

print("=== CONTRASTIVE PAIRS ANALYSIS ===")
for p_group in pairs:
    print("\n" + "="*80)
    sub = df_cross[df_cross["Mutant"].isin(p_group)][["Mutant", "Pattern", "Mean_Rel_L2", "biom_refiner_plus", "bregman_wno", "unet1d"]]
    print(sub.to_string(index=False))
