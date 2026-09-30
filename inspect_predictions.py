import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Load dataset
d = np.load("/home/users/zsy/mydata/budding_yeast/mutant_aware_pure_random_100000_210m_500s.npz")
mutant_names = [str(x) for x in d["mutant_names"]]

target_names = [
    "13_cln3Δ_bck2Δ",
    "14_cln3Δ_bck2Δ_GAL-CLN2_cln1Δ_cln2Δ",
    "15_cln3Δ_bck2Δ_multi-copy_CLN2",
    "18_cln1Δ_cln2Δ_cln3Δ_GAL-CLN2",
    "20_cln1Δ_cln2Δ_cln3Δ_sic1Δ",
    "22_cln1Δ_cln2Δ_cln3Δ_multi-copy_CLB5"
]

for t in target_names:
    idx = [i for i, n in enumerate(mutant_names) if t in n][0]
    name = mutant_names[idx]
    y = d["data"][idx] # (39, 500)
    
    # Let's inspect: What is the L2 norm of y?
    norm_y = np.linalg.norm(y)
    
    # Check max value of Cln2 (idx 0), Clb2 (idx 1), Sic1 (idx 2)
    # Check how many peaks (divisions) in Clb2
    clb2 = y[1]
    peaks = len([k for k in range(1, len(clb2)-1) if clb2[k] > clb2[k-1] and clb2[k] > clb2[k+1] and clb2[k] > 0.2])
    print(f"[{idx}] {name}")
    print(f"    Norm: {norm_y:.3f}, Max: {y.max():.3f}, Clb2 Max: {clb2.max():.3f}, Clb2 Peaks: {peaks}")
    print(f"    Final step sum: {y[:, -1].sum():.3f}, Initial step sum: {y[:, 0].sum():.3f}")
