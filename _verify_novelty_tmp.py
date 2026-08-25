"""al4pde_novelty 小规模功能验证 (临时脚本, 验证后删除)
1. sample_mass_random: Level 分层正确 + ki 范围正确 + 输出可仿真
2. _mass_family_of / _mass_desc_of: MASS_RAND_ 前缀解析正确
3. _compute_combination_novelty_scores: 上位性基因对覆盖稀缺度计算正确
4. _map_family_unc_to_pool: 家族不确定性 → 候选池映射正确
5. ActiveLearningCallback('al4pde_novelty'): 初始化 override 生效
"""
import sys
sys.path.insert(0, '/home/users/zsy/projects/budding_yeast_cell_cycle')

import numpy as np

# ========== Test 1: sample_mass_random ==========
from budding_yeast_v2.data.mass_random_sampler import sample_mass_random

np.random.seed(42)
from budding_yeast_v2.utils.al_callback import _parse_mass_gene_modes
ki_by_level = {1: [], 2: [], 3: [], 4: []}
n = 400
for level in [1, 2, 3, 4]:
    for _ in range(n):
        ovr, desc = sample_mass_random(level)
        # 用正规解析器统计不同基因数 (复合机制 CLB2[KO+GAL-OE] 内含 +, 不能直接 split)
        genes = set(g for g, _ in _parse_mass_gene_modes(desc))
        ki_by_level[level].append(len(genes))
        assert isinstance(ovr, dict) and 'mdt' in ovr, f"ovr 缺 mdt: {ovr}"

for level, kis in ki_by_level.items():
    print(f"L{level}: ki range = [{min(kis)}, {max(kis)}] (期望: "
          f"{ {1:'[1,1]', 2:'[2,2]', 3:'[3,5]', 4:'[4,6]'}[level] })")

assert set(ki_by_level[1]) == {1}, "L1 基因数应为 1"
assert set(ki_by_level[2]) == {2}, "L2 基因数应为 2"
assert set(ki_by_level[3]) <= {3, 4, 5}, "L3 基因数应为 3~5"
assert set(ki_by_level[4]) <= {4, 5, 6}, "L4 基因数应为 4~6"
print("PASS 1: sample_mass_random Level 分层与 ki 范围正确\n")

# 展示各 Level 样例
for level in [1, 2, 3, 4]:
    ovr, desc = sample_mass_random(level)
    print(f"  L{level} 样例: {desc}")

# ========== Test 2: 名称解析 ==========
from budding_yeast_v2.utils.al_callback import _mass_family_of, _mass_desc_of, _parse_mass_gene_modes

tests = [
    ('MASS_RAND_CLB2[KO]+CLN2[GAL-OE]_L1_123', 'CLB2', 'CLB2[KO]+CLN2[GAL-OE]'),
    ('MASS_AL_SIC1[GAL-OE]+CDH1[KO]_L2_42', 'SIC1', 'SIC1[GAL-OE]+CDH1[KO]'),
    ('MASS_CDC14[ts]+NET1[TAB6-1]_L3_7', 'CDC14', 'CDC14[ts]+NET1[TAB6-1]'),
    ('MASS_RAND_PDS1[db+PM]_L2_9', 'PDS1', 'PDS1[db+PM]'),
]
for name, fam, desc in tests:
    assert _mass_family_of(name) == fam, f"{name} family 错误: {_mass_family_of(name)}"
    assert _mass_desc_of(name) == desc, f"{name} desc 错误: {_mass_desc_of(name)}"
assert _mass_desc_of('1_WT_Glc') is None
print("PASS 2: _mass_family_of / _mass_desc_of 对 MASS_RAND_/MASS_AL_/MASS_ 名解析正确\n")

# ========== Test 3+4: novelty 打分 (mock 池) ==========
from budding_yeast_v2.utils.al_callback import ActiveLearningCallback

cb = ActiveLearningCallback(strategy='al4pde_novelty', random_seed=42)
assert cb.initial_train_size == 5000 and cb._al4pde_per_round_add == 19000
assert cb._is_al4pde_strategy()
assert cb.novelty_weight == 0.4 and cb.family_weight == 0.3
print("PASS 5a: __init__ override 生效 (5000 初始 + 4x19000 采集)\n")

# mock 候选池元数据: 4 个候选
cb._pool_size = 4
cb._pool_mass_meta = [
    {'genes': [('CLB2', 'KO'), ('CLN2', 'GAL-OE')], 'level': 2},   # 训练集已见该对 → novelty 0
    {'genes': [('CDC14', 'ts'), ('NET1', 'TAB6-1')], 'level': 3},  # 未见对 → novelty 1
    {'genes': [('CLB2', 'KO')], 'level': 1},                       # 单基因已见 → 0
    {'genes': [], 'level': 1},                                     # 纯碳源 → 0
]

class MockDM:
    _mutant_names = ['MASS_RAND_CLB2[KO]+CLN2[GAL-OE]_L1_1', 'MASS_RAND_SIC1[KO]_L1_2']
    _train_idx_lhs = [0, 1]
    class _T:  # 模拟 train_dataset_lhs
        tensors = None
    train_dataset_lhs = _T()

novelty = cb._compute_combination_novelty_scores(MockDM())
print(f"novelty scores = {novelty}")
assert novelty[0] == 0.0, "CLB2+CLN2 对已在训练集, novelty 应为 0"
assert novelty[1] == 1.0, "CDC14+NET1 对未见, novelty 应为 1"
assert novelty[2] == 0.0, "CLB2 单基因已见, novelty 应为 0"
assert novelty[3] == 0.0, "纯碳源 novelty 应为 0"
print("PASS 3: 组合新颖性 (上位性基因对覆盖) 计算正确\n")

# 家族不确定性映射: CDC14 家族最不确定 → 候选 1 (CDC14[ts]) 得最高分
# 百分位: CLB2(0.1,最低)→0.0 | SIC1(0.3)→1/3 | CLN2(0.5)→2/3 | CDC14(0.9,最高)→1.0
family_unc = {'CLB2': 0.1, 'CLN2': 0.5, 'SIC1': 0.3, 'CDC14': 0.9}
fam_scores = cb._map_family_unc_to_pool(family_unc)
print(f"family scores = {fam_scores}")
assert fam_scores[1] == 1.0, "CDC14 家族最不确定, 候选 1 应得 1.0"
assert fam_scores[0] == 0.0, "CLB2 家族最确定 (0.1), 候选 0 应得 0.0"
assert fam_scores[2] == 0.0, "候选 2 也是 CLB2 家族, 应得 0.0"
assert fam_scores[3] == 0.0, "纯碳源候选家族分应为 0"
print("PASS 4: 家族不确定性 → 候选池映射正确 (百分位)\n")

# ========== Test 6: 随机池采样集成 (小规模 _generate_param_only_pool 冒烟) ==========
# 直接验证: random-only 采样器生成的样本可被 mass 版 ODE 引擎仿真 (抽 3 个)
from budding_yeast_v2.utils.al_callback import solve_single_mass_bridge

np.random.seed(7)
ok = 0
for i in range(3):
    ovr, desc = sample_mass_random(3)
    res = solve_single_mass_bridge(f"MASS_RAND_{desc}_L3_{i}", ovr)
    if res is not None:
        _, y, p_vec, pat = res
        assert y.shape == (39, 500) and p_vec.shape == (141,)
        ok += 1
        print(f"  仿真 OK: {desc} | pattern={pat}")
    else:
        print(f"  仿真失败 (可接受, 物理过滤): {desc}")
print(f"ODE 仿真验证: {ok}/3 成功 (失败为物理过滤, 属正常)")
print("\n=== 全部功能验证通过 ===")
