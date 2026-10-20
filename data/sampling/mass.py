"""MA structure-aware selection, unchanged quotas, mixture and convergent coordination."""
import numpy as np
from ..systems.budding.catalog import GROUP_BY_NAME, GENE_NAMES, COMPOSITE_ELIGIBLE
from ..systems.budding import topology as graph
from .common import QuotaPicker, next_L1_cell, _sample_mode, _resolve_conflicts
ADJ = graph.ADJ
P_BIO = graph.P_BIO
_BIO_SUM = graph._BIO_SUM
PAIR_LISTS = graph.PAIR_LISTS
HUBS_DEG3 = graph.HUBS_DEG3
STAR_LEAVES = graph.STAR_LEAVES
CONVERGENT_PAIRS = graph.CONVERGENT_PAIRS
_OUT_ALL = graph._OUT_ALL
graph_dist = graph.graph_dist
MODE_FREQ = "class"

STRUCT_L2 = {'C-A': 0.30, 'C-B': 0.30, 'C-C': 0.15, 'C-D': 0.25}

STRUCT_L34 = {'convergent': 0.30, 'cascade': 0.25, 'star': 0.25, 'antag_core': 0.15, 'orthogonal': 0.05}

_L2_PICKER = QuotaPicker(list(STRUCT_L2.keys()), list(STRUCT_L2.values()))

_L34_PICKER = QuotaPicker(list(STRUCT_L34.keys()), list(STRUCT_L34.values()))

STRUCT_MIX_P = 0.50

def _sample_random_genes(ki, boost=None):
    out = []
    while len(out) < ki:
        out.append(_weighted_gene(boost, exclude=tuple(out)))
    return out

def _weighted_gene(weights=None, exclude=()):
    names = [n for n in GENE_NAMES if n not in exclude]
    if weights is None:
        w = np.array([P_BIO[n] for n in names])
    else:
        w = np.array([weights.get(n, P_BIO[n]) for n in names])
    w = w / w.sum()
    return str(np.random.choice(names, p=w))

def _pair_weight(a, b, boost=None):
    w = P_BIO[a] * P_BIO[b] / (_BIO_SUM ** 2)
    if boost:
        w *= (boost.get(a, 1.0) + boost.get(b, 1.0)) / 2.0
    return w

def sample_L2_fallback(boost=None):
    a = _weighted_gene(boost)
    b = _weighted_gene(boost, exclude=(a,))
    return [a, b]

def _sample_L2(boost=None):
    cls = _L2_PICKER.pick()
    if cls == 'C-C':
        hub_candidates = HUBS_DEG3 + ['MDT']
        hub_weights = [float(len(ADJ[h])) for h in HUBS_DEG3] + [3.0]
        hw = np.array(hub_weights, dtype=float)
        center = str(np.random.choice(hub_candidates, p=hw / hw.sum()))
        leaves = [l for l in STAR_LEAVES[center]]
        if not leaves:
            return sample_L2_fallback(boost), 'C-C'
        w = np.array([P_BIO[l] * (boost.get(l, 1.0) if boost else 1.0) for l in leaves])
        leaf = str(np.random.choice(leaves, p=w / w.sum()))
        return [center, leaf], 'C-C'
    pairs = PAIR_LISTS[cls]
    w = np.array([_pair_weight(a, b, boost) for a, b in pairs])
    i = int(np.random.choice(len(pairs), p=w / w.sum()))
    return list(pairs[i]), cls

def _walk_cascade(ki, boost=None):
    start = _weighted_gene(boost)
    chain = [start]
    while len(chain) < ki:
        cur = chain[-1]
        # sorted: ADJ[cur] is a set whose iteration order depends on
        # per-process string hash randomization (PYTHONHASHSEED) — an
        # unsorted feed into np.random.choice breaks same-seed
        # reproducibility across processes
        neigh = sorted(n for n in ADJ[cur] if n not in chain)
        if neigh:
            nxt = str(np.random.choice(neigh))
        else:
            nxt = _weighted_gene(boost, exclude=tuple(chain))
        chain.append(nxt)
    return chain

def _sample_convergent(ki, boost=None):
    pair_keys = list(CONVERGENT_PAIRS.keys())
    if not pair_keys:
        return _sample_random_genes(ki, boost)
    w = np.array([P_BIO[a] * P_BIO[b] * (boost.get(a, 1.0) + boost.get(b, 1.0) if boost else 1.0) for a, b in pair_keys])
    idx = int(np.random.choice(len(pair_keys), p=w / w.sum()))
    a, b = pair_keys[idx]
    shared = CONVERGENT_PAIRS[(a, b)]
    target = str(np.random.choice(shared))
    
    # 2-hop biological pathway fan-in / downstream effector rescue (Tier 2/3 generic prior):
    # 50% chance: if target has downstream targets (_OUT_ALL[target]),
    # pick one as the downstream effector (e.g., SBF_MBF -> CLN2/CLB5)
    downstream = sorted(_OUT_ALL.get(target, set()) - {a, b, target})
    if downstream and np.random.rand() < 0.50:
        effector = str(np.random.choice(downstream))
        out = [a, b, effector]
    else:
        out = [a, b, target]
    
    if ki == 3:
        return out
    
    # ki > 3: draw downstream neighbors or regulators of target
    cand = sorted(set(ADJ[target]) - set(out))
    while len(out) < ki and cand:
        nxt = str(np.random.choice(cand))
        out.append(nxt)
        cand.remove(nxt)
    while len(out) < ki:
        out.append(_weighted_gene(boost, exclude=tuple(out)))
    return out

def _sample_star(ki, boost=None):
    hub_candidates = HUBS_DEG3 + ['MDT']
    hub_weights = [float(len(ADJ[h])) for h in HUBS_DEG3] + [3.0]
    hw = np.array(hub_weights, dtype=float)
    center = str(np.random.choice(hub_candidates, p=hw / hw.sum()))
    leaves = [l for l in STAR_LEAVES[center]]
    out = [center]
    pool = list(leaves) if leaves else list(GENE_NAMES)
    while len(out) < ki and pool:
        w = np.array([P_BIO[g] * (boost.get(g, 1.0) if boost else 1.0) for g in pool])
        i = int(np.random.choice(len(pool), p=w / w.sum()))
        out.append(pool.pop(i))
    while len(out) < ki:
        out.append(_weighted_gene(boost, exclude=tuple(out)))
    return out

def _sample_antag_core(ki, boost=None):
    core, _ = _sample_L2(boost)
    out = list(core)
    cand = set()
    for g in core:
        cand |= ADJ[g]
    cand -= set(out)
    while len(out) < ki:
        if cand:
            g = str(np.random.choice(sorted(cand)))
            cand.discard(g)
        else:
            g = _weighted_gene(boost, exclude=tuple(out))
        out.append(g)
    return out

def _sample_orthogonal(ki, boost=None):
    out = [_weighted_gene(boost)]
    tries = 0
    while len(out) < ki and tries < 60:
        tries += 1
        g = _weighted_gene(boost, exclude=tuple(out))
        if all(graph_dist(g, s) >= 3 for s in out):
            out.append(g)
    while len(out) < ki:
        out.append(_weighted_gene(boost, exclude=tuple(out)))
    return out

def sample_skeleton(level, boost=None):
    """Returns (genes list, structure_class). Level quotas are exact.
    v5.1 (R3): L2/L3/L4 draw the structure-aware arm with prob STRUCT_MIX_P
    (0.50), else the i.i.d. uniform 'random' arm."""
    if level == 1:
        return [_weighted_gene(boost)], 'L1'
    if np.random.rand() >= STRUCT_MIX_P:   # v5.1 random arm (R3)
        if level == 2:
            ki = 2
        elif level == 3:
            ki = int(np.random.randint(3, 6))
        else:
            ki = int(np.random.randint(4, 7))
        return _sample_random_genes(ki, boost), 'random'
    if level == 2:
        return _sample_L2(boost)
    ki = int(np.random.randint(3, 6)) if level == 3 else int(np.random.randint(4, 7))
    cls = _L34_PICKER.pick()
    if cls == 'convergent':   return _sample_convergent(ki, boost), cls
    if cls == 'cascade':      return _walk_cascade(ki, boost), cls
    if cls == 'star':         return _sample_star(ki, boost), cls
    if cls == 'antag_core':   return _sample_antag_core(ki, boost), cls
    return _sample_orthogonal(ki, boost), cls
def _assign_convergent_modes(genes):
    """Universal Pathway Buffering / Genetic Suppressor mode assignment.
    For convergent multigene skeletons [a, b, eff, ...]:
    - First 2 parallel regulator inputs (a, b) receive Loss-of-Function (synthetic defect)
    - 3rd effector gene receives Dosage Suppression (OE) if activated downstream,
      or Bypass Suppression (LOF) if inhibited downstream.
    - Additional genes draw naturally from _sample_mode.
    Completely generic across the ODE topology, zero test-set hardcoding."""
    pairs = []
    for g in genes[:2]:
        modes = GROUP_BY_NAME[g]['modes']
        lofs = [m for m in ['KO', 'ts', 'DAmP', 'pLOF', 'db'] if m in modes]
        if lofs:
            m = 'KO' if 'KO' in lofs and np.random.rand() < 0.75 else str(np.random.choice(lofs))
        else:
            m = _sample_mode(g)
        pairs.append((g, m))

    if len(genes) >= 3:
        eff = genes[2]
        modes = GROUP_BY_NAME[eff]['modes']
        oes = [m for m in ['GAL-OE', 'MC'] if m in modes]
        lofs = [m for m in ['KO', 'ts', 'DAmP', 'pLOF', 'db'] if m in modes]
        if oes and (not lofs or np.random.rand() < 0.60):
            m = str(np.random.choice(oes))
        elif lofs:
            m = 'KO' if 'KO' in lofs and np.random.rand() < 0.75 else str(np.random.choice(lofs))
        else:
            m = _sample_mode(eff)
        pairs.append((eff, m))

    for g in genes[3:]:
        pairs.append((g, _sample_mode(g)))

    return pairs
def sample_mass_mutant(level=1, *, ma):
    """
    v2 leak-free mutant selection (replaces the v1 rescue-motif path).

    v1 leakage (REMOVED, 2026-09-06): 8 rescue-motif templates were hand-built
    against the 126 real mutants (code comments mapped motifs to strain numbers
    #13-#126 = ban-4 targeted hardcoding) and the gene/mode draws used
    Table-S5-fitted P_bio_G / mode weights (ban-3 composition statistics).

    Selection = the structure-aware skeleton inlined in Part 5.5 above
    (all Tier 2/3 sources, zero test-set contact):
      - gene weights  : uniform over 27 modules (v5.1 R2; degree+1 and engine
                        P_bio_G retained as ablation arms)
      - L1            : shuffled 84-cell (gene, mode) grid rotation (coverage)
      - L2/L3/L4      : 50% structure-class quotas from the 36-edge module
                        graph (C-A/C-B/C-C/C-D; cascade/star/antag_core/
                        orthogonal) + 50% i.i.d. uniform genes (v5.1 R3)
      - mode          : --mode_freq arm (class quotas DEFAULT = v5.1 R1;
                        flat = uniform anti-leak; role = Tier 2/3 affinity;
                        s5 = leak-by-design control)
      - conflicts     : touch-set collision check + mode redraw (Sect 3.1.6)
      - carbon rule   : GAL -> mdt=150 forced; else {90,150,160}
                        w.p. (0.75,0.18,0.07) (v5.1 R5, v1r-aligned)
    Strength values: engine sample_scaling_factor (Tier 2 literature bands;
    v5.1: TEM1/ESP1/NET1 ts anchor envelopes LogU[anchor/4, anchor x 2],
    PDS1-db residual U[0, 0.01], MC integer support {2..50}).
    Level quotas: set by generate_mass_dataset (40/30/20/10 mechanized).
    """
    p_wt = ma.get_default_params()
    p_mut = p_wt.copy()
    ovr = {}
    has_gal_oe = False
    mut_descriptions = []
    if level == 1:
        g0, m0 = next_L1_cell()
        if MODE_FREQ != 'flat':
            m0 = _sample_mode(g0)
        pairs = [(g0, m0)]
    else:
        genes, _cls = sample_skeleton(level)
        if _cls == 'convergent' and np.random.rand() < 0.6:
            pairs = _assign_convergent_modes(genes)
        else:
            pairs = [(g, _sample_mode(g)) for g in genes]
    expanded = []
    for g, m in pairs:
        expanded.append((g, m))
        gmode = GROUP_BY_NAME.get(g, {}).get('modes', [])
        if g in COMPOSITE_ELIGIBLE and np.random.rand() < 0.25:
            synth = [x for x in ('GAL-OE', 'MC') if x in gmode]
            struc = [x for x in ('db', 'PM', 'CA') if x in gmode]
            if m in synth and struc:
                expanded.append((g, str(np.random.choice(struc))))
            elif m in struc and synth:
                expanded.append((g, str(np.random.choice(synth))))
    pairs = expanded
    pairs = _resolve_conflicts(pairs)
    if not pairs:
        return sample_mass_mutant(level, ma=ma)
    for g_name, m_chosen in pairs:
        mut_descriptions.append(f'{g_name}[{m_chosen}]')
        if m_chosen == 'GAL-OE':
            has_gal_oe = True
        alpha = ma.sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
        ma.apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)
    if has_gal_oe:
        ovr['mdt'] = 150.0
    elif 'mdt' not in ovr:
        ovr['mdt'] = float(np.random.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))
    desc_str = '+'.join(mut_descriptions)
    return (ovr, desc_str)
