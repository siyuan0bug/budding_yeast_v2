"""诊断: random-only 采样的 real-dup 拦截原因"""
import sys
sys.path.insert(0, '/home/users/zsy/projects/budding_yeast_cell_cycle')
import numpy as np
from budding_yeast_v2.data.mass_random_sampler import sample_mass_random
from budding_yeast_v2.data.mass_dataset_generator import (
    is_duplicate_real_mutant, get_default_params, get_cached_real_params, REAL_MUTANTS)

np.random.seed(123)
defaults = get_default_params()
defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

n_real_dup = 0
dup_examples = []
n = 3000
for i in range(n):
    ovr, desc = sample_mass_random(1 + (i % 4))  # 轮流 4 个 level
    if is_duplicate_real_mutant(ovr, defaults):
        n_real_dup += 1
        if len(dup_examples) < 15:
            dup_examples.append((1 + (i % 4), desc, sorted(ovr.keys())))

print(f"real-dup 拦截率: {n_real_dup}/{n} = {100*n_real_dup/n:.1f}%\n")
for lv, desc, keys in dup_examples:
    print(f"  L{lv} | {desc} | keys={keys}")

# 对照: ovr 全等于 WT 的样本 (纯碳源) 是否存在
n_carbon_only = 0
for i in range(2000):
    ovr, desc = sample_mass_random(1)
    non_mdt = {k: v for k, v in ovr.items() if k != 'mdt'}
    if not non_mdt:
        n_carbon_only += 1
print(f"\nL1 纯碳源样本 (ovr 仅含 mdt): {n_carbon_only}/2000")
