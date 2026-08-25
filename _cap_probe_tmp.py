"""容量探测: 各 Level 在 8% 池内去重下的真实容量 (不含 D_real 检查, 纯空间测量)"""
import sys
sys.path.insert(0, '/home/users/zsy/projects/budding_yeast_cell_cycle')
import numpy as np
from budding_yeast_v2.data.mass_random_sampler import sample_mass_random

np.random.seed(2024)

for level in [1, 2, 3, 4]:
    dedup_buckets = {}
    n_accepted = 0
    attempts = 200000
    for _ in range(attempts):
        ovr, desc = sample_mass_random(level)
        key_sig = tuple(sorted(ovr.keys()))
        if key_sig:
            pert_vals = np.array([ovr[k] for k in key_sig], dtype=np.float64)
            is_dup = False
            for prev_vals in dedup_buckets.get(key_sig, ()):
                arr_prev = np.array([prev_vals[k] for k in key_sig], dtype=np.float64)
                if np.all(np.abs(pert_vals - arr_prev) <=
                          0.08 * np.maximum(np.abs(arr_prev), 1e-12)):
                    is_dup = True
                    break
            if is_dup:
                continue
            dedup_buckets.setdefault(key_sig, []).append(ovr)
        n_accepted += 1
    print(f"L{level}: {attempts} 次尝试 → {n_accepted} 接受 "
          f"({100*n_accepted/attempts:.1f}%) | 去重桶数={len(dedup_buckets)}")
