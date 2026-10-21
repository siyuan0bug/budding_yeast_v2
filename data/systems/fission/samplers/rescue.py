"""Fission collision rescue: change one legal module mode or redraw its strength."""
import numpy as np


def rescue_fission(overrides, description, rng):
    """Change one legal module mode, retaining the Fission strength semantics."""
    from ..catalog import SAMPLING_MODULES
    from ..distributions import _strength
    pairs = [tuple(part.split(':',1)) for part in description.split('+')]
    eligible = [i for i,(module,mode) in enumerate(pairs) if any(m!=mode for m in SAMPLING_MODULES[module]['modes'])]
    if not eligible:
        # Single-mode modules are continuously resampled in their existing band.
        eligible = list(range(len(pairs)))
    if not eligible:
        raise ValueError('No module available for collision rescue')
    i = int(rng.choice(eligible))
    module, old = pairs[i]
    modes = [m for m in SAMPLING_MODULES[module]['modes'] if m != old]
    # 有时保留原模式、只重抽连续强度；否则像 Rum1:OP 只剩 KO，而 KO 恰好等于
    # real rum1D 时会永远碰撞，30 次救援必然耗尽。
    mode = str(rng.choice(modes)) if (modes and rng.random() < .5) else old
    pairs[i] = (module,mode)
    new = dict(overrides)
    for key in SAMPLING_MODULES[module]['params']:
        new[key] = _strength(mode,key,rng)
    return new, '+'.join(f'{g}:{m}' for g,m in pairs)
