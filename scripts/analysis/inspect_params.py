import numpy as np
import re

# Load dataset
d = np.load("/home/users/zsy/mydata/budding_yeast/mutant_aware_pure_random_100000_210m_500s.npz")
mutant_names = [str(x) for x in d["mutant_names"]]
param_names = [str(x) for x in d["param_names"]] if "param_names" in d else None

def find_mutant(query):
    for i, name in enumerate(mutant_names):
        if query in name:
            return i, name
    return None, None

wt_idx, _ = find_mutant("1_Wild-type")
wt_p = d["params"][wt_idx]

queries = [
    "10_bck2Δ", "13_cln3Δ_bck2Δ", "14_cln3Δ_bck2Δ", "15_cln3Δ_bck2Δ", "16_cln3Δ_bck2Δ",
    "3_cln1Δ", "17_cln1Δ", "18_cln1Δ", "20_cln1Δ", "22_cln1Δ"
]

for q in queries:
    idx, name = find_mutant(q)
    if idx is not None:
        p = d["params"][idx]
        y = d["data"][idx] # (39, 500)
        diff_idx = np.where(np.abs(p - wt_p) > 1e-4)[0]
        print(f"[{idx}] {name}")
        print(f"    Trajectory max: {y.max():.2f}, min: {y.min():.2f}, mean: {y.mean():.4f}")
        print(f"    Params diff from WT: {len(diff_idx)} params changed:")
        for di in diff_idx:
            pname = param_names[di] if param_names else f"param_{di}"
            print(f"      [{di}] {pname}: WT={wt_p[di]:.4f} -> Mut={p[di]:.4f}")
