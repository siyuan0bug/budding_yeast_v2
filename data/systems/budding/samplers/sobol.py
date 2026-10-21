"""Original Sobol allocation for Gibbs and parameter pools; quantiles provided by selected profile."""
import numpy as np
from .gibbs import GibbsBoltzmannGraphSampler
from ..distributions.quantiles import sample_scaling_factor_sobol

def sample_gibbs_sobol_mutant(level, gibbs_sampler, sobol_vec, rng=np.random, *, ma):
    """
    Constructs a valid cell-cycle mutant parameter override dictionary using
    Gibbs sampling for gene topology and Sobol coordinates for continuous parameters.
    sobol_vec: array of shape (16,) containing low-discrepancy coordinates in [0, 1].
    """
    p_wt = ma.get_default_params()
    p_mut = p_wt.copy()
    ovr = {}
    has_gal_oe = False
    mut_descriptions = []
    if level == 1:
        g0, m0 = ma.next_L1_cell()
        if ma.MODE_FREQ != 'flat':
            m0 = ma._sample_mode(g0)
        pairs = [(g0, m0)]
    else:
        genes = gibbs_sampler.sample_genes(level, rng=rng)
        pairs = [(g, ma._sample_mode(g)) for g in genes]
    expanded = []
    for g, m in pairs:
        expanded.append((g, m))
        gmode = ma.GROUP_BY_NAME.get(g, {}).get('modes', [])
        if g in ma.COMPOSITE_ELIGIBLE and rng.rand() < 0.25:
            synth = [x for x in ('GAL-OE', 'MC') if x in gmode]
            struc = [x for x in ('db', 'PM', 'CA') if x in gmode]
            if m in synth and struc:
                expanded.append((g, str(rng.choice(struc))))
            elif m in struc and synth:
                expanded.append((g, str(rng.choice(synth))))
    pairs = expanded
    pairs = ma._resolve_conflicts(pairs)
    if not pairs:
        rolled_vec = np.roll(sobol_vec, 1)
        return sample_gibbs_sobol_mutant(level, gibbs_sampler, rolled_vec, rng, ma=ma)
    coord_idx = 0
    for g_name, m_chosen in pairs:
        mut_descriptions.append(f'{g_name}[{m_chosen}]')
        if m_chosen == 'GAL-OE':
            has_gal_oe = True
        if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db']:
            u_coord = sobol_vec[coord_idx % len(sobol_vec)]
            coord_idx += 1
            alpha = ma.sample_scaling_factor_sobol(m_chosen, g_name, u_coord)
        else:
            alpha = 1.0
        ma.apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)
    if has_gal_oe:
        ovr['mdt'] = 150.0
    elif 'mdt' not in ovr:
        u_mdt = sobol_vec[-1]
        if u_mdt < 0.75:
            ovr['mdt'] = 90.0
        elif u_mdt < 0.93:
            ovr['mdt'] = 150.0
        else:
            ovr['mdt'] = 160.0
    desc_str = '+'.join(mut_descriptions)
    return (ovr, desc_str)
def sample_pure_random_candidate(level, sobol_vec, rng, *, ma):
    """
    Generates a single pure random mutant with zero graph priors.
    - Level k selects k genes uniformly at random without replacement.
    - Mode chosen uniformly from gene's available modes.
    - Continuous multipliers sampled via Sobol low-discrepancy coordinates.
    """
    p_wt = ma.get_default_params()
    p_wt.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    p_mut = p_wt.copy()
    ovr = {}
    ki = int(level)
    selected_genes = list(rng.choice(ma.GENE_NAMES, size=ki, replace=False))
    pairs = []
    for g in selected_genes:
        modes = sorted(ma.GROUP_BY_NAME[g]['modes'].keys())
        m = str(rng.choice(modes))
        pairs.append((g, m))
    expanded = []
    for g, m in pairs:
        expanded.append((g, m))
        gmode = ma.GROUP_BY_NAME.get(g, {}).get('modes', {})
        if g in ma.COMPOSITE_ELIGIBLE and rng.rand() < 0.25:
            synth = [x for x in ('GAL-OE', 'MC') if x in gmode]
            struc = [x for x in ('db', 'PM', 'CA') if x in gmode]
            if m in synth and struc:
                expanded.append((g, str(rng.choice(struc))))
            elif m in struc and synth:
                expanded.append((g, str(rng.choice(synth))))
    pairs = expanded
    pairs = ma._resolve_conflicts(pairs)
    if not pairs:
        return sample_pure_random_candidate(level, np.roll(sobol_vec, 1), rng, ma=ma)
    has_gal_oe = False
    mut_descriptions = []
    coord_idx = 0
    for g_name, m_chosen in pairs:
        mut_descriptions.append(f'{g_name}[{m_chosen}]')
        if m_chosen == 'GAL-OE':
            has_gal_oe = True
        if m_chosen == 'KO':
            alpha = 0.0
        elif m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db']:
            u_coord = sobol_vec[coord_idx % len(sobol_vec)]
            coord_idx += 1
            alpha = ma.sample_scaling_factor_sobol(m_chosen, g_name, u_coord)
        else:
            alpha = 1.0
        ma.apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)
    if has_gal_oe:
        ovr['mdt'] = 150.0
    elif 'mdt' not in ovr:
        u_mdt = sobol_vec[-1]
        if u_mdt < 0.75:
            ovr['mdt'] = 90.0
        elif u_mdt < 0.93:
            ovr['mdt'] = 150.0
        else:
            ovr['mdt'] = 160.0
    desc_str = '+'.join(mut_descriptions)
    return (ovr, desc_str, pairs)
