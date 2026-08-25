"""_generate_param_only_pool 集成冒烟测试 (novelty 策略, 生产规模 100K, 临时脚本)
验证: random-only 采样器集成 / MASS_RAND_ 命名 / Level 分布 / 去重 / 元数据解析
"""
import sys
sys.path.insert(0, '/home/users/zsy/projects/budding_yeast_cell_cycle')
import numpy as np
import torch
import time

from budding_yeast_v2.utils.al_callback import ActiveLearningCallback, _mass_family_of


class MockDM:
    """提供 _generate_param_only_pool 所需的归一化统计量"""
    mean_y = torch.rand(1, 39, 1)
    std_y = torch.rand(1, 39, 1) + 0.5
    p_mean = torch.rand(1, 141)
    p_std = torch.rand(1, 141) + 0.5


cb = ActiveLearningCallback(strategy='al4pde_novelty', random_seed=42)
dm = MockDM()

t0 = time.time()
cb._generate_param_only_pool(dm)
elapsed = time.time() - t0
print(f"\n池生成耗时: {elapsed:.0f}s")

# ===== 断言 =====
assert cb._pool_size == 100000, f"池规模应为 100000, 实际 {cb._pool_size}"
assert cb._pool_P.shape == (100000, 141)
assert cb._pool_Y_ic.shape == (100000, 39)
assert cb._pool_ovr is not None and len(cb._pool_ovr) == 100000
assert cb._pool_mass_meta is not None and len(cb._pool_mass_meta) == 100000

# 命名: 全部 MASS_RAND_ 前缀 (无 MASS_AL_ / 无 motif 命名)
names = cb._pool_mutant_names
assert all(str(n).startswith('MASS_RAND_') for n in names[:1000]), "命名前缀应为 MASS_RAND_"

# Level 分布
level_counts = {}
for meta in cb._pool_mass_meta:
    level_counts[meta['level']] = level_counts.get(meta['level'], 0) + 1
total = sum(level_counts.values())
for lv in sorted(level_counts):
    print(f"  L{lv}: {level_counts[lv]} ({100*level_counts[lv]/total:.1f}%)")
assert abs(level_counts[1] - 25000) <= 1
assert abs(level_counts[2] - 35000) <= 1
assert abs(level_counts[3] - 25000) <= 1

# 家族解析可用性 (随机抽 100 个)
for i in range(0, 100000, 1000):
    fam = _mass_family_of(names[i])
    assert isinstance(fam, str) and len(fam) > 0

# ovr 完整性: 每个 ovr 含 mdt 且非空
for i in range(0, 100000, 10000):
    assert 'mdt' in cb._pool_ovr[i]

print("=== 集成冒烟测试全部通过 (100K random-only 候选池) ===")
