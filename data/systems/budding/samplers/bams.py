"""Original signed walks and bifurcation bisection; refined overrides remain paired with their trajectory."""
import numpy as np
from ..catalog import GENE_NAMES, GROUP_BY_NAME
from ..topology import MODULE_EDGES

ADJ = {n: set() for n in GENE_NAMES}
EDGES_OUT = {n: [] for n in GENE_NAMES}
EDGES_IN = {n: [] for n in GENE_NAMES}
for u, v, s in MODULE_EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)
    EDGES_OUT[u].append((v, s))
    EDGES_IN[v].append((u, s))
CONVERGENT_TARGETS = {}
for i, a in enumerate(GENE_NAMES):
    for b in GENE_NAMES[i + 1:]:
        t_a = {v for v, _ in EDGES_OUT[a]}
        t_b = {v for v, _ in EDGES_OUT[b]}
        common = t_a & t_b
        if common:
            CONVERGENT_TARGETS[a, b] = sorted(common)
ANTAG_PAIRS = [(u, v) for u, v, s in MODULE_EDGES if s == '-']
LOF_MODES = {'KO', 'ts', 'DAmP', 'pLOF', 'db'}
GOF_MODES = {'GAL-OE', 'MC', 'CA', 'APC-A', 'TAB6-1'}

def get_mode_polarity(mode):
    if mode in LOF_MODES:
        return -1
    if mode in GOF_MODES:
        return +1
    return 0

def touch_set(g_name, m_chosen, *, ma):
    ovr = {}
    p_wt = ma.get_default_params()
    p_wt.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    ma.apply_gene_mutation(g_name, m_chosen, 1.0, p_wt, {}, ovr)
    return set(ovr.keys())

def resolve_pair_conflicts(pairs, *, ma):
    pairs = list(pairs)
    rounds = 20
    while len(pairs) > 1 and rounds > 0:
        rounds -= 1
        touched = {}
        conflict = None
        for i, (g, m) in enumerate(pairs):
            ts = touch_set(g, m, ma=ma)
            for pk in ts:
                if pk in touched:
                    conflict = (i, touched[pk])
                    break
                touched[pk] = i
            if conflict is not None:
                break
        if conflict is None:
            break
        i, j = conflict
        gj = pairs[j][0]
        modes_j = list(GROUP_BY_NAME[gj]['modes'].keys())
        pairs[j] = (gj, str(np.random.choice(modes_j)))
    return pairs

def fast_eval_division_status(ovr, tf=180, *, ma):
    """
    Lightweight ODE execution (~12ms) to check whether cell divides or arrests.
    Returns: True if cell divides (Pattern_A/B), False if arrested (Pattern_C).
    """
    _, yu, _, pat = ma.simulate_mutant('bisect_probe', ovr, tf=tf, ns=150)
    if yu is None or pat == 'Pattern_C: Exponential-Arrest (==0)':
        return False
    return True

def bisect_bifurcation_dosage(base_ovr, target_gene, mode, alpha_min, alpha_max, n_steps=3, *, ma):
    """
    3-step log-scale numerical bisection on ODE solver.
    Locates the knife-edge threshold alpha* where the network transitions
    between arrest and division.
    """
    p_wt = ma.get_default_params()
    p_wt.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    ovr_lo = dict(base_ovr)
    ma.apply_gene_mutation(target_gene, mode, alpha_min, p_wt, {}, ovr_lo)
    div_lo = fast_eval_division_status(ovr_lo, ma=ma)
    ovr_hi = dict(base_ovr)
    ma.apply_gene_mutation(target_gene, mode, alpha_max, p_wt, {}, ovr_hi)
    div_hi = fast_eval_division_status(ovr_hi, ma=ma)
    if div_lo == div_hi:
        return None
    lo, hi = (alpha_min, alpha_max)
    for _ in range(n_steps):
        mid = np.sqrt(lo * hi)
        ovr_mid = dict(base_ovr)
        ma.apply_gene_mutation(target_gene, mode, mid, p_wt, {}, ovr_mid)
        div_mid = fast_eval_division_status(ovr_mid, ma=ma)
        if div_mid == div_lo:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)
_L1_GRID = [(g, m) for g in GENE_NAMES for m in GROUP_BY_NAME[g]['modes']]
np.random.RandomState(20260831).shuffle(_L1_GRID)
_L1_IDX = {'i': 0}

def get_next_L1_cell():
    cell = _L1_GRID[_L1_IDX['i'] % len(_L1_GRID)]
    _L1_IDX['i'] += 1
    return cell

def sample_bams_skeleton_and_modes(level, *, ma):
    """
    Samples multi-gene skeleton and coordinated modes based on signed topology.
    Level:
      1: L1 grid rotation (1 gene)
      2: Direct topological edge coupling (2 genes)
      3: Flux-balance motif walk (3~5 genes)
      4: Cross-pathway extended walk (4~6 genes)
    """
    if level == 1:
        g, m = get_next_L1_cell()
        return ([(g, m)], 'L1_Grid', None)
    g0 = str(np.random.choice(GENE_NAMES))
    modes0 = list(GROUP_BY_NAME[g0]['modes'].keys())
    m0 = str(np.random.choice(modes0))
    s0 = get_mode_polarity(m0)
    if s0 == 0:
        s0 = -1
    pairs = [(g0, m0)]
    bisection_spec = None
    strat_desc = ''
    if level == 2:
        sub_type = np.random.choice(['antag', 'cascade', 'convergent', 'random'], p=[0.35, 0.35, 0.2, 0.1])
        strat_desc = f'L2_{sub_type}'
        if sub_type == 'antag' and ANTAG_PAIRS:
            cand_antag = [v for u, v, s in MODULE_EDGES if u == g0 and s == '-'] + [u for u, v, s in MODULE_EDGES if v == g0 and s == '-']
            if cand_antag:
                g1 = str(np.random.choice(cand_antag))
            else:
                u, v = ANTAG_PAIRS[np.random.choice(len(ANTAG_PAIRS))]
                g0, g1 = (u, v)
                m0 = 'KO' if 'KO' in GROUP_BY_NAME[g0]['modes'] else list(GROUP_BY_NAME[g0]['modes'].keys())[0]
                pairs[0] = (g0, m0)
            m1_pool = [m for m in ['KO', 'ts', 'db', 'GAL-OE', 'MC'] if m in GROUP_BY_NAME[g1]['modes']]
            m1 = str(np.random.choice(m1_pool)) if m1_pool else list(GROUP_BY_NAME[g1]['modes'].keys())[0]
            pairs.append((g1, m1))
        elif sub_type == 'cascade':
            neighbors = [(v, s) for v, s in EDGES_OUT[g0]] + [(u, s) for u, s in EDGES_IN[g0]]
            if neighbors:
                g1, sign = neighbors[np.random.choice(len(neighbors))]
                if s0 == -1 and sign == '+':
                    m1_pool = [m for m in ['GAL-OE', 'MC'] if m in GROUP_BY_NAME[g1]['modes']]
                    m1 = str(np.random.choice(m1_pool)) if m1_pool else 'KO'
                    if m1 in ['MC', 'GAL-OE']:
                        bisection_spec = (g1, m1, 1.0 if m1 == 'MC' else 0.01, 30.0 if m1 == 'MC' else 12.0)
                else:
                    m1_pool = list(GROUP_BY_NAME[g1]['modes'].keys())
                    m1 = str(np.random.choice(m1_pool))
                pairs.append((g1, m1))
            else:
                g1 = str(np.random.choice([g for g in GENE_NAMES if g != g0]))
                pairs.append((g1, str(np.random.choice(list(GROUP_BY_NAME[g1]['modes'].keys())))))
        elif sub_type == 'convergent' and CONVERGENT_TARGETS:
            matching_pairs = [(a, b) for a, b in CONVERGENT_TARGETS.keys() if a == g0 or b == g0]
            if matching_pairs:
                a, b = matching_pairs[np.random.choice(len(matching_pairs))]
                g1 = b if a == g0 else a
            else:
                a, b = list(CONVERGENT_TARGETS.keys())[np.random.choice(len(CONVERGENT_TARGETS))]
                g0, g1 = (a, b)
                pairs[0] = (g0, 'KO' if 'KO' in GROUP_BY_NAME[g0]['modes'] else list(GROUP_BY_NAME[g0]['modes'].keys())[0])
            m1_lofs = [m for m in ['KO', 'ts', 'db'] if m in GROUP_BY_NAME[g1]['modes']]
            m1 = str(np.random.choice(m1_lofs)) if m1_lofs else list(GROUP_BY_NAME[g1]['modes'].keys())[0]
            pairs.append((g1, m1))
        else:
            g1 = str(np.random.choice([g for g in GENE_NAMES if g != g0]))
            pairs.append((g1, str(np.random.choice(list(GROUP_BY_NAME[g1]['modes'].keys())))))
    else:
        ki = 3 if level == 3 else 3
        strat_desc = f'L{level}_FluxWalk'
        visited = {g0}
        walk_history = [(g0, s0)]
        if s0 == 1 and np.random.rand() < 0.7:
            lofs = [m for m in ['KO', 'ts', 'db'] if m in GROUP_BY_NAME[g0]['modes']]
            if lofs:
                m0 = 'KO' if 'KO' in lofs else lofs[0]
                pairs[0] = (g0, m0)
                s0 = -1
                walk_history[0] = (g0, s0)
        while len(pairs) < ki:
            frontier = []
            for u in visited:
                for v, s in EDGES_OUT[u]:
                    if v not in visited:
                        frontier.append((u, v, s, 'out'))
                for v, s in EDGES_IN[u]:
                    if v not in visited:
                        frontier.append((u, v, s, 'in'))
            if frontier:
                u, v, sign, direction = frontier[np.random.choice(len(frontier))]
                visited.add(v)
                u_polar = next((pol for g_u, pol in walk_history if g_u == u), -1)
                is_compensation = np.random.rand() < 0.6
                if is_compensation and u_polar == -1 and (sign == '+'):
                    oes = [m for m in ['MC', 'GAL-OE'] if m in GROUP_BY_NAME[v]['modes']]
                    if oes:
                        mv = str(np.random.choice(oes))
                        if bisection_spec is None:
                            bisection_spec = (v, mv, 1.0 if mv == 'MC' else 0.01, 30.0 if mv == 'MC' else 12.0)
                    else:
                        mv = str(np.random.choice(list(GROUP_BY_NAME[v]['modes'].keys())))
                elif is_compensation and u_polar == -1 and (sign == '-'):
                    lofs = [m for m in ['KO', 'ts', 'db'] if m in GROUP_BY_NAME[v]['modes']]
                    mv = str(np.random.choice(lofs)) if lofs else list(GROUP_BY_NAME[v]['modes'].keys())[0]
                else:
                    mv = str(np.random.choice(list(GROUP_BY_NAME[v]['modes'].keys())))
                v_polar = get_mode_polarity(mv)
                walk_history.append((v, v_polar))
                pairs.append((v, mv))
            else:
                unvisited = [g for g in GENE_NAMES if g not in visited]
                if not unvisited:
                    break
                v = str(np.random.choice(unvisited))
                visited.add(v)
                mv = str(np.random.choice(list(GROUP_BY_NAME[v]['modes'].keys())))
                pairs.append((v, mv))
    COMPOSITE_ELIGIBLE = [n for n in GENE_NAMES if any((m in GROUP_BY_NAME[n]['modes'] for m in ('GAL-OE', 'MC'))) and any((m in GROUP_BY_NAME[n]['modes'] for m in ('db', 'PM', 'CA')))]
    expanded = []
    for g, m in pairs:
        expanded.append((g, m))
        gmode = GROUP_BY_NAME.get(g, {}).get('modes', {})
        if g in COMPOSITE_ELIGIBLE and np.random.rand() < 0.25:
            synth = [x for x in ('GAL-OE', 'MC') if x in gmode]
            struc = [x for x in ('db', 'PM', 'CA') if x in gmode]
            if m in synth and struc:
                expanded.append((g, str(np.random.choice(struc))))
            elif m in struc and synth:
                expanded.append((g, str(np.random.choice(synth))))
    pairs = resolve_pair_conflicts(expanded, ma=ma)
    return (pairs, strat_desc, bisection_spec)

def sample_bams_mutant_task(level=2, defaults=None, *, ma):
    """
    Fast task constructor (Phase 2). Returns initial overrides, descriptor, and optional bisection spec.
    """
    if defaults is None:
        defaults = ma.get_default_params()
        defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    p_wt = defaults
    ovr = {}
    pairs, strat_desc, bisection_spec = sample_bams_skeleton_and_modes(level, ma=ma)
    desc_parts = []
    has_gal = False
    for g, m in pairs:
        if m == 'GAL-OE':
            has_gal = True
        alpha = ma.sample_scaling_factor(m, g)
        ma.apply_gene_mutation(g, m, alpha, p_wt, {}, ovr)
        desc_parts.append(f'{g}[{m}]')
    if has_gal:
        ovr['mdt'] = 150.0
    else:
        ovr['mdt'] = float(np.random.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))
    desc_str = '+'.join(desc_parts) + f'_{strat_desc}'
    return (ovr, desc_str, bisection_spec)

def execute_simulation_task(task_spec, *, ma):
    """
    Executed inside each worker process:
    1. If bisection_spec is attached, runs 3-step ODE bisection to cluster dosage around alpha*.
    2. Simulates full 210 min / 500 steps trajectory.
    """
    name, ovr, bisection_spec, tf, ns = task_spec
    defaults = ma.get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    if bisection_spec is not None:
        target_g, target_m, a_min, a_max = bisection_spec
        base_ovr = dict(ovr)
        try:
            alpha_star = bisect_bifurcation_dosage(base_ovr, target_g, target_m, a_min, a_max, n_steps=3, ma=ma)
            if alpha_star is not None:
                jitter = float(np.random.choice([0.85, 0.95, 1.05, 1.2]))
                refined_alpha = float(np.clip(alpha_star * jitter, a_min, a_max))
                ma.apply_gene_mutation(target_g, target_m, refined_alpha, defaults, {}, ovr)
        except Exception:
            pass
    return (ma.simulate_mutant(name, ovr, tf=tf, ns=ns), ovr)
