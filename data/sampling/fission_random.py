"""Original Fission L1 rotation and random module selection."""
import numpy as np
from ..systems.fission.catalog import SAMPLING_MODULES, MODE_CLASS, MODE_CLASS_QUOTA
from ..systems.fission.distributions import _strength

MODE_FREQ = 'class'        # 由 --mode_freq 覆盖 (class = 配额; flat = 均匀)

_L1_GRID = [(m, mo) for m in SAMPLING_MODULES for mo in SAMPLING_MODULES[m]['modes']]

_L1_IDX = {'i': 0}

def next_L1_cell():
    cell = _L1_GRID[_L1_IDX['i'] % len(_L1_GRID)]
    _L1_IDX['i'] += 1
    return cell

def _sample_mode(module):
    """按 --mode_freq 抽取该模块的一个合法模式。"""
    modes = SAMPLING_MODULES[module]['modes']
    if MODE_FREQ == 'class':
        w = np.array([MODE_CLASS_QUOTA.get(MODE_CLASS.get(m, 'event'), 1.0)
                      for m in modes], dtype=float)
        return str(np.random.choice(modes, p=w / w.sum()))
    return str(np.random.choice(modes))          # flat: 均匀 1/k

def sample_random_mutant(level, rng):
    """按分层 level 抽一个合成突变体, 返回 (overrides, name)。
    level=1 -> 1 个模块 (L1 网格轮转); level=2 -> 2 个; level=3 -> 3 个。
    (模块数上限 3, 与论文 Table 2 的最大突变基因数一致。) 所有模块均均匀随机抽取。"""
    if level == 1:
        m, mode = next_L1_cell()                  # 网格轮转 -> 覆盖保证
        pairs = [(m, mode)]
    else:
        ki = level                                # L2 -> 2, L3 -> 3
        mods = rng.choice(list(SAMPLING_MODULES.keys()), size=ki, replace=False)
        pairs = [(str(m), _sample_mode(str(m))) for m in mods]
    ovr, parts = {}, []
    for m, mode in pairs:
        for key in SAMPLING_MODULES[m]['params']:
            ovr[key] = _strength(mode, key, rng)
        parts.append(f'{m}:{mode}')
    return ovr, '+'.join(parts)
