"""Original paired skeleton/shared-dose construction; two profiles share canonical biology."""
import numpy as np
from ..systems.budding.distributions.mass import sample_scaling_factor as sample_scaling_factor_A
from ..systems.budding.mutations.mass import apply_gene_mutation as _apply_mass_mutation


def apply_gene_mutation_A(g_name, m_chosen, alpha, p_wt, p_mut, ovr):
    """Original twin A uses MASS rules plus its explicit TEM1-ts initial reset."""
    _apply_mass_mutation(g_name,m_chosen,alpha,p_wt,p_mut,ovr)
    if g_name=='TEM1' and m_chosen=='ts':
        ovr['init_Tem1']=0.0

TARGET_MODES = {'ts', 'db', 'APC-A', 'TAB6-1', 'CA', 'CSL', 'CKO', 'scc1-RR', 'PM'}

def sample_twin_mutant_pair(level, defaults, *, ma):
    """
    Generates a strictly paired candidate mutant representation:
    1. Sample gene skeleton and modes using Regime B.
    2. Shared variables (MC discrete copies, GAL-OE values, KO zeros, mdt) are identical.
    3. Target modes (ts, db, APC-A, TAB6-1, etc.) branch into:
       - ovr_B: Regime B physical envelopes and boundary buffers.
       - ovr_A: Regime A sampling distributions and direct assignments.
    Returns: (sample_name, ovr_B, ovr_A, desc_str)
    """
    p_wt = defaults.copy()
    ki = int(level)
    selected_genes = list(np.random.choice(ma.GENE_NAMES, size=ki, replace=False))
    pairs = []
    for g in selected_genes:
        modes = sorted(ma.GROUP_BY_NAME[g]['modes'].keys())
        m = str(np.random.choice(modes))
        pairs.append((g, m))
    expanded = []
    for g, m in pairs:
        expanded.append((g, m))
        gmode = ma.GROUP_BY_NAME.get(g, {}).get('modes', [])
        if g in ma.COMPOSITE_ELIGIBLE and np.random.rand() < 0.25:
            synth = [x for x in ('GAL-OE', 'MC') if x in gmode]
            struc = [x for x in ('db', 'PM', 'CA') if x in gmode]
            if m in synth and struc:
                expanded.append((g, str(np.random.choice(struc))))
            elif m in struc and synth:
                expanded.append((g, str(np.random.choice(synth))))
    pairs = expanded
    pairs = ma._resolve_conflicts(pairs)
    if not pairs:
        return sample_twin_mutant_pair(level, defaults, ma=ma)
    has_gal_oe = any((m == 'GAL-OE' for _, m in pairs))
    if has_gal_oe:
        mdt_shared = 150.0
    else:
        mdt_shared = float(np.random.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))
    mut_descriptions = [f'{g}[{m}]' for g, m in pairs]
    desc_str = '+'.join(mut_descriptions)
    ovr_B = {'mdt': mdt_shared}
    ovr_A = {'mdt': mdt_shared}
    p_mut_B = p_wt.copy()
    p_mut_A = p_wt.copy()
    for g_name, m_chosen in pairs:
        if m_chosen not in TARGET_MODES:
            alpha_shared = ma.sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'DAmP', 'pLOF'] else 1.0
            ma.apply_gene_mutation(g_name, m_chosen, alpha_shared, p_wt, p_mut_B, ovr_B)
            ma.apply_gene_mutation(g_name, m_chosen, alpha_shared, p_wt, p_mut_A, ovr_A)
        else:
            alpha_B = ma.sample_scaling_factor(m_chosen, g_name) if m_chosen in ['ts', 'db'] else 1.0
            ma.apply_gene_mutation(g_name, m_chosen, alpha_B, p_wt, p_mut_B, ovr_B)
            alpha_A = sample_scaling_factor_A(m_chosen, g_name) if m_chosen in ['ts', 'db'] else 1.0
            apply_gene_mutation_A(g_name, m_chosen, alpha_A, p_wt, p_mut_A, ovr_A)
    return (ovr_B, ovr_A, desc_str)
