"""Original MASS rescue-motif selection algorithm, retained as an explicitly named strategy."""
import numpy as np

def sample_mass_mutant(level=1, *, ma):
    """
    Generate a synthetic mutant sample by selecting binding groups and applying
    Section 7 master table lookups, Section 2.3 biophysical distributions,
    Section 2.4 carbon decoupling & Section 2.4.2 biological rescue motifs.
    Returns (ovr_dict, mutant_description_str).
    """
    p_wt = ma.get_default_params()
    p_mut = p_wt.copy()
    ovr = {}
    has_gal_oe = False
    mut_descriptions = []
    use_rescue_motif = level in [3, 4] and np.random.rand() < 0.5
    if use_rescue_motif:
        motif_type = np.random.choice(['M_EXIT', 'G1_S_RESCUE', 'SHIRAYAMA_BYPASS', 'APC_A_INTERACTION', 'ANTAGONISTIC_TUG', 'ANAPHASE_SEPARASE', 'MEN_TELOPHASE', 'CDC14_SUBSTRATE'])
        selected_mut_pairs = []
        if motif_type == 'M_EXIT':
            driver_choices = [('CLB2', 'db'), ('CLB2', 'GAL-OE'), ('CDC14', 'ts'), ('TEM1', 'ts')]
            driver = driver_choices[np.random.choice(len(driver_choices))]
            suppressors_pool = [('SIC1', np.random.choice(['MC', 'GAL-OE'])), ('CLB5', 'KO'), ('CDC6', np.random.choice(['MC', 'GAL-OE'])), ('CDC15', 'GAL-OE'), ('NET1', 'TAB6-1')]
            k_supp = np.random.randint(1, min(level, len(suppressors_pool) + 1))
            chosen_indices = np.random.choice(len(suppressors_pool), size=k_supp, replace=False)
            selected_mut_pairs = [driver] + [suppressors_pool[idx] for idx in chosen_indices]
            if np.random.rand() < 0.3:
                ovr['mdt'] = float(np.random.choice([150.0, 160.0]))
        elif motif_type == 'G1_S_RESCUE':
            if np.random.rand() < 0.5:
                base_drivers = [('CLN3', 'KO'), ('BCK2', 'KO')]
                suppressors_pool = [('CLN2', np.random.choice(['MC', 'GAL-OE'])), ('CLB5', np.random.choice(['MC', 'GAL-OE'])), ('SIC1', 'KO'), ('CDH1', 'KO')]
            else:
                base_drivers = [('CLN2', 'KO'), ('CLN3', 'KO')]
                suppressors_pool = [('BCK2', 'MC'), ('CLB5', np.random.choice(['MC', 'GAL-OE'])), ('CLB2', 'GAL-OE'), ('SIC1', 'KO'), ('CDH1', 'KO'), ('APC_Core', 'ts')]
            k_supp = min(max(1, level - len(base_drivers)), len(suppressors_pool))
            chosen_indices = np.random.choice(len(suppressors_pool), size=k_supp, replace=False)
            selected_mut_pairs = base_drivers + [suppressors_pool[idx] for idx in chosen_indices]
        elif motif_type == 'SHIRAYAMA_BYPASS':
            driver = ('CDC20', np.random.choice(['KO', 'ts']))
            suppressors_pool = [('PDS1', np.random.choice(['KO', 'db'])), ('CLB5', np.random.choice(['KO', 'db'])), ('SIC1', 'KO'), ('MAD2', 'CSL')]
            k_supp = np.random.randint(1, min(level, len(suppressors_pool) + 1))
            chosen_indices = np.random.choice(len(suppressors_pool), size=k_supp, replace=False)
            selected_mut_pairs = [driver] + [suppressors_pool[idx] for idx in chosen_indices]
        elif motif_type == 'APC_A_INTERACTION':
            driver = ('APC_Core', 'APC-A')
            suppressors_pool = [('CDH1', 'KO'), ('CLB2', 'GAL-OE'), ('SIC1', np.random.choice(['MC', 'GAL-OE', 'KO'])), ('CDC6', np.random.choice(['MC', 'GAL-OE'])), ('CDC20', 'MC')]
            k_supp = np.random.randint(1, min(level, len(suppressors_pool) + 1))
            chosen_indices = np.random.choice(len(suppressors_pool), size=k_supp, replace=False)
            selected_mut_pairs = [driver] + [suppressors_pool[idx] for idx in chosen_indices]
            if any((g == 'CDH1' for g, _ in selected_mut_pairs)) and np.random.rand() < 0.4:
                ovr['mdt'] = 150.0
        elif motif_type == 'ANTAGONISTIC_TUG':
            r_pick = np.random.rand()
            if r_pick < 0.3:
                base_drivers = [('SIC1', 'GAL-OE'), ('CLN2', np.random.choice(['GAL-OE', 'KO']))]
                suppressors_pool = [('CDH1', 'KO'), ('SWI5', 'KO'), ('CLB2', 'GAL-OE')]
            elif r_pick < 0.55:
                base_drivers = [('SIC1', 'KO'), ('CDH1', 'KO')]
                suppressors_pool = [('CDC6', 'pDEL'), ('CDC20', np.random.choice(['GAL-OE', 'MC'])), ('SWI5', 'KO')]
            elif r_pick < 0.8:
                base_drivers = [('SWI5', 'KO'), ('CDH1', 'KO')]
                suppressors_pool = [('SIC1', 'GAL-OE'), ('CLN2', 'KO'), ('CDC20', 'MC')]
            else:
                base_drivers = [('SIC1', 'GAL-OE'), ('CDH1', 'KO')]
                suppressors_pool = [('CLN2', 'KO'), ('SWI5', 'KO'), ('CDC6', 'MC')]
            k_supp = np.random.randint(0, min(max(1, level - len(base_drivers)) + 1, len(suppressors_pool) + 1))
            if k_supp > 0:
                chosen_indices = np.random.choice(len(suppressors_pool), size=k_supp, replace=False)
                selected_mut_pairs = base_drivers + [suppressors_pool[idx] for idx in chosen_indices]
            else:
                selected_mut_pairs = list(base_drivers)
            if any((g in ['CDH1', 'SIC1'] for g, _ in selected_mut_pairs)) and np.random.rand() < 0.4:
                ovr['mdt'] = 150.0
        elif motif_type == 'ANAPHASE_SEPARASE':
            driver = ('PDS1', 'db') if np.random.rand() < 0.5 else ('CDC20', 'ts')
            suppressors_pool = [('ESP1', np.random.choice(['GAL-OE', 'MC', 'ts'])), ('NET1', np.random.choice(['ts', 'TAB6-1'])), ('MAD2', 'CSL'), ('BUB2', 'CKO')]
            k_supp = np.random.randint(1, min(level, len(suppressors_pool) + 1))
            chosen_indices = np.random.choice(len(suppressors_pool), size=k_supp, replace=False)
            selected_mut_pairs = [driver] + [suppressors_pool[idx] for idx in chosen_indices]
        elif motif_type == 'MEN_TELOPHASE':
            driver = ('TEM1', 'ts') if np.random.rand() < 0.6 else ('CDC15', 'ts')
            suppressors_pool = [('CDC14', np.random.choice(['MC', 'GAL-OE'])), ('CDC15', np.random.choice(['MC', 'GAL-OE'])), ('TEM1', 'MC'), ('NET1', np.random.choice(['ts', 'TAB6-1']))]
            k_supp = np.random.randint(1, min(level, len(suppressors_pool) + 1))
            chosen_indices = np.random.choice(len(suppressors_pool), size=k_supp, replace=False)
            selected_mut_pairs = [driver] + [suppressors_pool[idx] for idx in chosen_indices]
        elif np.random.rand() < 0.35:
            selected_mut_pairs = [('NET1', 'GAL-OE'), ('CDC14', 'GAL-OE')]
        else:
            driver = ('CDC14', 'ts')
            suppressors_pool = [('SIC1', np.random.choice(['KO', 'GAL-OE'])), ('CDH1', 'KO'), ('CLN2', 'GAL-OE'), ('SWI5', 'KO'), ('NET1', np.random.choice(['TAB6-1', 'ts']))]
            k_supp = np.random.randint(1, min(level, len(suppressors_pool) + 1))
            chosen_indices = np.random.choice(len(suppressors_pool), size=k_supp, replace=False)
            selected_mut_pairs = [driver] + [suppressors_pool[idx] for idx in chosen_indices]
        for g_name, m_chosen in selected_mut_pairs:
            mut_descriptions.append(f'{g_name}[{m_chosen}]')
            if m_chosen == 'GAL-OE':
                has_gal_oe = True
            alpha = ma.sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            ma.apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)
    else:
        if level == 1:
            ki = 1
        elif level == 2:
            ki = 2
        elif level == 3:
            ki = np.random.randint(3, 6)
        else:
            ki = np.random.randint(4, 7)
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
