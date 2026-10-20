"""Original L1 rotation, mode weighting and conflict resolution shared by graph strategies."""
import numpy as np
from ..systems.budding.catalog import GROUP_BY_NAME, GENE_NAMES, MODE_CLASS, MODE_CLASS_QUOTA, GENE_ROLE, ROLE_AFFINITY, touch_set
MODE_FREQ = "class"

_L1_GRID = [(g, m) for g in GENE_NAMES for m in GROUP_BY_NAME[g]['modes']]

np.random.RandomState(20260831).shuffle(_L1_GRID)

_L1_IDX = {'i': 0}

def next_L1_cell():
    cell = _L1_GRID[_L1_IDX['i'] % len(_L1_GRID)]
    _L1_IDX['i'] += 1
    return cell

def _sample_mode(gene, *, rng=None):
    rng = np.random if rng is None else rng
    modes = list(GROUP_BY_NAME[gene]['modes'].keys())
    if MODE_FREQ == 's5':   # Ablation A: Table-S5-fitted mode weights (leaks by design)
        w = np.array([GROUP_BY_NAME[gene]['modes'][m] for m in modes], dtype=float)
        return str(rng.choice(modes, p=w / w.sum()))
    if MODE_FREQ == 'role':  # Ablation B: role-aware affinity weights (Tier 2/3)
        aff = ROLE_AFFINITY.get(GENE_ROLE.get(gene, 'event'), None)
        if aff is not None:
            w = np.array([aff.get(MODE_CLASS.get(m, 'event'), 1.0)
                          for m in modes], dtype=float)
            return str(rng.choice(modes, p=w / w.sum()))
        # event-role genes (or unmapped): flat, same RNG call as the flat arm
    if MODE_FREQ == 'class':  # v5.1 main arm: mode-class quotas (R1)
        w = np.array([MODE_CLASS_QUOTA.get(MODE_CLASS.get(m, 'event'), 1.0)
                      for m in modes], dtype=float)
        return str(rng.choice(modes, p=w / w.sum()))
    return str(rng.choice(modes))   # flattened uniform (anti-leak, §3.1.4)

def _resolve_conflicts(pairs, *, rng=None):
    """Same-parameter dual-mode ban + cross-gene overlap ban (§3.1.6).
    On collision between pair i and pair j: resample i's mode; if NO mode of
    gene_i can avoid the collision (e.g. CDH1 KO/CA both touch kscdh, which
    APC_Core-ts also touches), resample the PARTNER j instead; if neither
    side has a clean mode, drop pair i. After any drop the whole list is
    re-checked — a drop can expose a second, previously masked conflict
    (found by smoke test A2, 2026-09-06). Output is always conflict-free."""
    # Explicit streams also fix set traversal order across PYTHONHASHSEEDs.
    # Keep the omitted-rng path unchanged for existing sampler callers.
    pairs = list(pairs)
    rounds = 30
    while len(pairs) > 1:
        touched = {}
        conflict = None
        for i, (g, m) in enumerate(pairs):
            ts = touch_set(g, m)
            for pk in (sorted(ts) if rng is not None else ts):
                if pk in touched:
                    conflict = (i, touched[pk])
                    break
                touched[pk] = i
            if conflict is not None:
                break
        if conflict is None:
            return pairs
        i, j = conflict
        def _other_keys(idx):
            ks = [touch_set(g, m) for k, (g, m) in enumerate(pairs) if k != idx]
            return set().union(*ks) if ks else set()
        gi, gj = pairs[i][0], pairs[j][0]
        i_ok = any(not (touch_set(gi, mm) & _other_keys(i))
                   for mm in GROUP_BY_NAME[gi]['modes'])
        j_ok = any(not (touch_set(gj, mm) & _other_keys(j))
                   for mm in GROUP_BY_NAME[gj]['modes'])
        if rounds > 0 and i_ok:
            pairs[i] = (gi, _sample_mode(gi, rng=rng)); rounds -= 1
        elif rounds > 0 and j_ok:
            pairs[j] = (gj, _sample_mode(gj, rng=rng)); rounds -= 1
        else:
            # neither side can redraw clean (or redraw budget spent): drop
            # pair i and re-check the remaining list (loop continues)
            pairs.pop(i)
    return pairs
class QuotaPicker:
    """Exact quota categorical picker (equalized counts)."""
    def __init__(self, names, quotas):
        self.names = list(names)
        self.quotas = np.asarray(quotas, dtype=float)
        self.counts = np.zeros(len(self.names), dtype=int)
    def pick(self, weights=None):
        if weights is None:
            deficit = self.quotas * max(1, self.counts.sum()) - self.counts
            w = np.maximum(deficit, 1e-9)
        else:
            w = np.asarray(weights, dtype=float) + 1e-9
        i = int(np.random.choice(len(self.names), p=w / w.sum()))
        self.counts[i] += 1
        return self.names[i]
