import pandas as pd
import numpy as np

df_full = pd.read_csv("benchmark_49_models_full.csv")
comp = pd.read_csv("mutant_cross_model_comparison.csv")

print("=== TOTAL MODELS BENCHMARKED ===", len(df_full))
print("\n--- MODEL CATEGORIES ---")

# Let's categorize models:
# 1. Biological Hybrid / Ours: biom_refiner_plus, biom_refiner_event_hybrid
# 2. Advanced Neural Operators (ICML/ICLR 2024-2025): bregman_wno, bregman_fno, sc_fno, pno1d, mpno1d, ncflow, iganet_transformer, konode, transolver3_1d
# 3. Classic Neural Operators: fno1d, cno1d, mwt1d, dpot1d, transolver1d, galerkin_transformer1d, deeponet, pure_fno, pure_wno, wdno, gfno, noble
# 4. Graph & Operator: gnot, gcgnet, icon, bampno, resnet_film, latent_lno, lamo, koopman_dmd, neural_ode, unet1d
# 5. Time Series Transformers & Linear: autoformer, flowformer, iflowformer, informer, iinformer, reformer, ireformer, flashformer, iflashformer, vanilla_transformer, fedformer, pyraformer, itransformer, dlinear, nlinear, linear

def get_cat(name):
    if "biom" in name:
        return "1. Bio-Mechanistic Hybrid (Ours)"
    elif name in ["bregman_wno", "bregman_fno", "sc_fno", "pno1d", "mpno1d", "ncflow", "iganet_transformer", "konode", "transolver3_1d"]:
        return "2. Frontier Neural Operators (2024-2025)"
    elif name in ["fno1d", "cno1d", "mwt1d", "dpot1d", "transolver1d", "galerkin_transformer1d", "deeponet", "pure_fno", "pure_wno", "wdno", "gfno", "noble"]:
        return "3. Classical Neural Operators"
    elif name in ["gnot", "gcgnet", "icon", "bampno", "resnet_film", "latent_lno", "lamo", "koopman_dmd", "neural_ode", "unet1d"]:
        return "4. Graph/Latent/ODE Architectures"
    else:
        return "5. Deep Sequence / Transformer / Linear"

df_full["category"] = df_full["model"].apply(get_cat)

cat_summary = df_full.groupby("category")[["real_rel_l2", "real_mae", "real_corr", "real_core_rel_l2", "synth_rel_l2"]].agg(["mean", "min", "max"])
print(cat_summary.to_string())

# Top 5 in each category
print("\n" + "="*80)
print("TOP MODELS BY CATEGORY (Real Mutants Rel-L2):")
print("="*80)
for cat, group in df_full.groupby("category"):
    print(f"\n--- {cat} ---")
    g_sorted = group.sort_values(by="real_rel_l2")
    print(g_sorted[["model", "real_rel_l2", "real_mae", "real_corr", "real_core_rel_l2", "synth_rel_l2"]].head(5).to_string(index=False))

# Core 9 variables breakdown
# Let's inspect which core 9 variables are tracked
print("\n" + "="*80)
print("CORE 9 VARIABLES CORRELATION AND RELATIVE L2:")
print("="*80)
print(df_full.sort_values(by="real_core_rel_l2")[["model", "real_core_rel_l2", "real_core_mae", "real_core_corr", "real_rel_l2"]].head(15).to_string(index=False))
