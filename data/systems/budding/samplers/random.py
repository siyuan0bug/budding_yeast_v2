"""Original uniform and weighted random selection algorithms; explicit backend argument."""
import numpy as np

def sample_pure_random_mutant(level, rng, enable_ko_interval=False, ko_interval_max=0.0001, *, ma):
    """
    Samples a synthetic mutant candidate with zero graph topology.
    - Level k initially selects k genes; conflict resolution may remove a pair.
    - Genes are chosen uniformly at random without replacement.
    - Modes are chosen uniformly at random from the gene's available modes.
    - All downstream random draws use this same caller-owned rng.
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
        if g in ma.COMPOSITE_ELIGIBLE and rng.random() < 0.25:
            synth = [x for x in ('GAL-OE', 'MC') if x in gmode]
            struc = [x for x in ('db', 'PM', 'CA') if x in gmode]
            if m in synth and struc:
                expanded.append((g, str(rng.choice(struc))))
            elif m in struc and synth:
                expanded.append((g, str(rng.choice(synth))))
    pairs = expanded
    pairs = ma._resolve_conflicts(pairs, rng=rng)
    if not pairs:
        return sample_pure_random_mutant(level, rng, enable_ko_interval=enable_ko_interval, ko_interval_max=ko_interval_max, ma=ma)
    has_gal_oe = False
    mut_descriptions = []
    for g_name, m_chosen in pairs:
        mut_descriptions.append(f'{g_name}[{m_chosen}]')
        if m_chosen == 'GAL-OE':
            has_gal_oe = True
        if m_chosen == 'KO':
            if enable_ko_interval:
                temp_ovr = {}
                ma.apply_gene_mutation(g_name, 'KO', 0.0, p_wt, p_mut, temp_ovr, rng=rng)
                for k in temp_ovr:
                    if k in ('init_CDH1', 'init_Tem1', 'init_Cdc15'):
                        continue
                    val = float(rng.uniform(0.0, ko_interval_max))
                    ovr[k] = val
                    p_mut[k] = val
                if 'init_CDH1T' in temp_ovr:
                    t_val = ovr['init_CDH1T']
                    ovr['init_CDH1'] = float(rng.uniform(0.0, t_val))
                    p_mut['init_CDH1'] = ovr['init_CDH1']
                if 'Tem1T' in temp_ovr:
                    t_val = ovr['Tem1T']
                    ovr['init_Tem1'] = float(rng.uniform(0.0, t_val))
                    p_mut['init_Tem1'] = ovr['init_Tem1']
                if 'Cdc15T' in temp_ovr:
                    t_val = ovr['Cdc15T']
                    ovr['init_Cdc15'] = float(rng.uniform(0.0, t_val))
                    p_mut['init_Cdc15'] = ovr['init_Cdc15']
            else:
                ma.apply_gene_mutation(g_name, 'KO', 0.0, p_wt, p_mut, ovr, rng=rng)
        elif m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db']:
            alpha = ma.sample_scaling_factor(m_chosen, g_name, rng=rng)
            ma.apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr, rng=rng)
        else:
            ma.apply_gene_mutation(g_name, m_chosen, 1.0, p_wt, p_mut, ovr, rng=rng)
    if has_gal_oe:
        ovr['mdt'] = 150.0
    elif 'mdt' not in ovr:
        ovr['mdt'] = float(rng.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))
    desc_str = '+'.join(mut_descriptions)
    return (ovr, desc_str)
def sample_mass_random(level=1, *, ma):
    """
    Generate a synthetic mutant sample by PURE RANDOM exploratory sampling
    (no rescue motif / directed sampling), keeping the original Level tiers.

    Level -> number of mutated binding groups (ki), identical to the standard
    exploratory branch of mass_dataset_generator.sample_mass_mutant:
      L1: ki=1 | L2: ki=2 | L3: ki=3; higher levels disabled by the public sampler

    Returns:
        (ovr_dict, mutant_description_str) — 与 sample_mass_mutant 返回格式严格一致
    """
    p_wt = ma.get_default_params()
    p_mut = p_wt.copy()
    ovr = {}
    has_gal_oe = False
    mut_descriptions = []
    if level == 1:
        ki = 1
    elif level == 2:
        ki = 2
    elif level == 3:
        ki = 3
    else:
        ki = 3
    group_keys = list(ma.BINDING_GROUPS.keys())
    group_weights = [ma.BINDING_GROUPS[k]['P_bio_G'] for k in group_keys]
    selected_g_keys = np.random.choice(group_keys, size=min(ki, len(group_keys)), replace=False, p=group_weights)
    for g_key in selected_g_keys:
        g_info = ma.BINDING_GROUPS[g_key]
        g_name = g_info['name']
        modes = list(g_info['modes'].keys())
        m_weights = [g_info['modes'][m] for m in modes]
        m_weights = np.array(m_weights) / sum(m_weights)
        m_chosen = np.random.choice(modes, p=m_weights)
        comp_mode = None
        if g_name in ['CLB2', 'CLB5', 'PDS1', 'SIC1', 'CDH1']:
            synth_modes = [m for m in ['GAL-OE', 'MC'] if m in modes]
            struc_modes = [m for m in ['db', 'PM', 'CA'] if m in modes]
            if m_chosen in synth_modes and struc_modes and (np.random.rand() < 0.25):
                comp_mode = np.random.choice(struc_modes)
            elif m_chosen in struc_modes and synth_modes and (np.random.rand() < 0.25):
                comp_mode = np.random.choice(synth_modes)
        if comp_mode:
            mut_descriptions.append(f'{g_name}[{m_chosen}+{comp_mode}]')
            if m_chosen == 'GAL-OE' or comp_mode == 'GAL-OE':
                has_gal_oe = True
            alpha1 = ma.sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            ma.apply_gene_mutation(g_name, m_chosen, alpha1, p_wt, p_mut, ovr)
            alpha2 = ma.sample_scaling_factor(comp_mode, g_name) if comp_mode in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            ma.apply_gene_mutation(g_name, comp_mode, alpha2, p_wt, p_mut, ovr)
        else:
            mut_descriptions.append(f'{g_name}[{m_chosen}]')
            if m_chosen == 'GAL-OE':
                has_gal_oe = True
            alpha = ma.sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            ma.apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)
    if has_gal_oe:
        ovr['mdt'] = 150.0
    elif 'mdt' not in ovr:
        carbon_choice = np.random.choice(['Glc', 'Gal', 'Raff'], p=[0.75, 0.18, 0.07])
        if carbon_choice == 'Gal':
            ovr['mdt'] = 150.0
        elif carbon_choice == 'Raff':
            ovr['mdt'] = 160.0
        else:
            ovr['mdt'] = 90.0
    desc_str = '+'.join(mut_descriptions)
    return (ovr, desc_str)
def sample_mass_random_dist(level=1, *, ma):
    """
    Generate a synthetic mutant sample by pure random exploratory sampling
    (No Rescue Motifs), identical to mass_random_sampler.py, but using
    sample_scaling_factor from mutant_aware.py.
    """
    p_wt = ma.get_default_params()
    p_mut = p_wt.copy()
    ovr = {}
    has_gal_oe = False
    mut_descriptions = []
    if level == 1:
        ki = 1
    elif level == 2:
        ki = 2
    elif level == 3:
        ki = 3
    else:
        ki = 3
    group_keys = list(ma.BINDING_GROUPS.keys())
    group_weights = [ma.BINDING_GROUPS[k]['P_bio_G'] for k in group_keys]
    group_weights = np.array(group_weights) / sum(group_weights)
    selected_g_keys = np.random.choice(group_keys, size=min(ki, len(group_keys)), replace=False, p=group_weights)
    for g_key in selected_g_keys:
        g_info = ma.BINDING_GROUPS[g_key]
        g_name = g_info['name']
        modes = list(g_info['modes'].keys())
        m_weights = [g_info['modes'][m] for m in modes]
        m_weights = np.array(m_weights) / sum(m_weights)
        m_chosen = np.random.choice(modes, p=m_weights)
        comp_mode = None
        if g_name in ['CLB2', 'CLB5', 'PDS1', 'SIC1', 'CDH1']:
            synth_modes = [m for m in ['GAL-OE', 'MC'] if m in modes]
            struc_modes = [m for m in ['db', 'PM', 'CA'] if m in modes]
            if m_chosen in synth_modes and struc_modes and (np.random.rand() < 0.25):
                comp_mode = np.random.choice(struc_modes)
            elif m_chosen in struc_modes and synth_modes and (np.random.rand() < 0.25):
                comp_mode = np.random.choice(synth_modes)
        if comp_mode:
            mut_descriptions.append(f'{g_name}[{m_chosen}+{comp_mode}]')
            if m_chosen == 'GAL-OE' or comp_mode == 'GAL-OE':
                has_gal_oe = True
            alpha1 = ma.sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            ma.apply_gene_mutation(g_name, m_chosen, alpha1, p_wt, p_mut, ovr)
            alpha2 = ma.sample_scaling_factor(comp_mode, g_name) if comp_mode in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            ma.apply_gene_mutation(g_name, comp_mode, alpha2, p_wt, p_mut, ovr)
        else:
            mut_descriptions.append(f'{g_name}[{m_chosen}]')
            if m_chosen == 'GAL-OE':
                has_gal_oe = True
            alpha = ma.sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            ma.apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)
    if has_gal_oe:
        ovr['mdt'] = 150.0
    elif 'mdt' not in ovr:
        carbon_choice = np.random.choice(['Glc', 'Gal', 'Raff'], p=[0.75, 0.18, 0.07])
        if carbon_choice == 'Gal':
            ovr['mdt'] = 150.0
        elif carbon_choice == 'Raff':
            ovr['mdt'] = 160.0
        else:
            ovr['mdt'] = 90.0
    desc_str = '+'.join(mut_descriptions)
    return (ovr, desc_str)
