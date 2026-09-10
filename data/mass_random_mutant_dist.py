"""
MASS Random-Selection + Mutant-Aware Parameter Distribution Hybrid Generator
================================================================================
Ablation Dataset Generator:
1. Everything (ODE engine, REAL_MUTANTS, BINDING_GROUPS, apply_gene_mutation,
   Level selection logic, carbon decoupling, deduplication, simulation) is
   100% identical to and imported from mass_random_sampler.py / mass_dataset_generator.py.
2. The ONLY component inherited from mutant_aware.py is sample_scaling_factor
   (the parameter scaling distributions, e.g. v5.1/v5.2 anchors, continuous distributions).
"""

import os
import sys
import math
import time
import warnings
import concurrent.futures
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

warnings.filterwarnings("ignore")

# Ensure current directory and workspace root are in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
for _p in [_CURRENT_DIR, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 1. Import all base mechanics, constants, engine, and selection logic from mass_dataset_generator
try:
    from mass_dataset_generator import (
        REAL_MUTANTS,
        get_default_params,
        _defaults,
        PARAM_KEYS,
        BINDING_GROUPS,
        apply_gene_mutation,
        simulate_mutant,
        is_duplicate_real_mutant,
    )
except ImportError:
    from budding_yeast_v2.data.mass_dataset_generator import (
        REAL_MUTANTS,
        get_default_params,
        _defaults,
        PARAM_KEYS,
        BINDING_GROUPS,
        apply_gene_mutation,
        simulate_mutant,
        is_duplicate_real_mutant,
    )

# 2. ONLY import sample_scaling_factor from mutant_aware
try:
    from mutant_aware import sample_scaling_factor
except ImportError:
    from budding_yeast_v2.data.mutant_aware import sample_scaling_factor


def sample_mass_random_dist(level=1):
    """
    Generate a synthetic mutant sample by pure random exploratory sampling
    (No Rescue Motifs), identical to mass_random_sampler.py, but using
    sample_scaling_factor from mutant_aware.py.
    """
    p_wt = get_default_params()
    p_mut = p_wt.copy()
    ovr = {}

    has_gal_oe = False
    mut_descriptions = []

    # Standard unconstrained exploratory sampling (from mass_random_sampler.py)
    if level == 1:
        ki = 1
    elif level == 2:
        ki = 2
    elif level == 3:
        ki = int(np.random.randint(3, 6))   # ki in {3, 4, 5}
    else:
        ki = int(np.random.randint(4, 7))   # ki in {4, 5, 6}

    group_keys = list(BINDING_GROUPS.keys())
    group_weights = [BINDING_GROUPS[k]['P_bio_G'] for k in group_keys]
    group_weights = np.array(group_weights) / sum(group_weights)

    selected_g_keys = np.random.choice(
        group_keys, size=min(ki, len(group_keys)), replace=False, p=group_weights
    )

    for g_key in selected_g_keys:
        g_info = BINDING_GROUPS[g_key]
        g_name = g_info['name']

        modes = list(g_info['modes'].keys())
        m_weights = [g_info['modes'][m] for m in modes]
        m_weights = np.array(m_weights) / sum(m_weights)
        m_chosen = np.random.choice(modes, p=m_weights)

        comp_mode = None
        if g_name in ['CLB2', 'CLB5', 'PDS1', 'SIC1', 'CDH1']:
            synth_modes = [m for m in ['GAL-OE', 'MC'] if m in modes]
            struc_modes = [m for m in ['db', 'PM', 'CA'] if m in modes]
            if m_chosen in synth_modes and struc_modes and np.random.rand() < 0.25:
                comp_mode = np.random.choice(struc_modes)
            elif m_chosen in struc_modes and synth_modes and np.random.rand() < 0.25:
                comp_mode = np.random.choice(synth_modes)

        if comp_mode:
            mut_descriptions.append(f"{g_name}[{m_chosen}+{comp_mode}]")
            if m_chosen == 'GAL-OE' or comp_mode == 'GAL-OE':
                has_gal_oe = True

            alpha1 = sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            apply_gene_mutation(g_name, m_chosen, alpha1, p_wt, p_mut, ovr)
            alpha2 = sample_scaling_factor(comp_mode, g_name) if comp_mode in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            apply_gene_mutation(g_name, comp_mode, alpha2, p_wt, p_mut, ovr)
        else:
            mut_descriptions.append(f"{g_name}[{m_chosen}]")
            if m_chosen == 'GAL-OE':
                has_gal_oe = True

            alpha = sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)

    # Environmental Carbon Source Decoupling (from mass_dataset_generator.py)
    if has_gal_oe:
        ovr['mdt'] = 150.0
    else:
        if 'mdt' not in ovr:
            carbon_choice = np.random.choice(['Glc', 'Gal', 'Raff'], p=[0.75, 0.18, 0.07])
            if carbon_choice == 'Gal':
                ovr['mdt'] = 150.0
            elif carbon_choice == 'Raff':
                ovr['mdt'] = 160.0
            else:
                ovr['mdt'] = 90.0

    desc_str = "+".join(mut_descriptions)
    return ovr, desc_str


# ==============================================================================
# Full Dataset Generator Pipeline & Multiprocessing
# ==============================================================================

def _worker_task_executor(task_tuple):
    idx, name, ovr = task_tuple
    mname, y_sim, bio_l, pat_l = simulate_mutant(name, ovr, tf=210, ns=500)
    return (idx, mname, y_sim, bio_l, pat_l, ovr)


def generate_mass_random_dist_dataset(n_synthetic=100000, level_ratios=[0.25, 0.35, 0.25, 0.15],
                                       max_attempts=None, num_workers=60,
                                       output_name=None, output_dir=None):
    """
    Generate complete dataset (126 Real Mutants + n_synthetic Synthetic Random MASS Mutants)
    and export into .npz file.
    """
    if output_name is None:
        output_name = f"mass_random_dist_{n_synthetic}_210m_500s.npz"

    if output_dir is None:
        if os.name == 'nt':
            output_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            output_dir = "/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast"

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass

    output_path = os.path.join(output_dir, output_name)

    print("=====================================================================================")
    print(f" MASS Random + Mutant-Aware Parameter Distribution Dataset Generator")
    print(f" Output Target: {output_path}")
    print(f" Target Synthetic Samples: {n_synthetic:,} | CPU Workers: {num_workers}")
    print(f" Level Distribution: L1={level_ratios[0]:.0%}, L2={level_ratios[1]:.0%}, L3={level_ratios[2]:.0%}, L4={level_ratios[3]:.0%}")
    print("=====================================================================================")

    p_wt_default = _defaults.copy()

    # --------------------------------------------------------------------------
    # Phase 1: Real Mutants (126 Strains)
    # --------------------------------------------------------------------------
    print("\n[Phase 1/4] Loading 126 Chen (2004) Appendix B Real Mutants (D_real)...")
    tasks_to_simulate = []
    task_idx = 0

    for mid, (mname, movr) in REAL_MUTANTS.items():
        clean_name = f"{mid}_{mname.replace(' ', '_')}"
        tasks_to_simulate.append((task_idx, clean_name, movr))
        task_idx += 1

    n_real = len(tasks_to_simulate)
    print(f" -> Successfully prepared {n_real} Real Mutants.")

    # --------------------------------------------------------------------------
    # Phase 2: Synthetic Mutants (Level Stratified Random Selection)
    # --------------------------------------------------------------------------
    print("\n[Phase 2/4] Sampling Synthetic Candidates with Real Mutant Deduplication...")
    level_counts = [int(round(n_synthetic * r)) for r in level_ratios]
    diff = n_synthetic - sum(level_counts)
    level_counts[1] += diff

    n_dedup_blocked = 0
    t_start_sample = time.time()

    for lvl_idx, (lvl, target_k) in enumerate(zip([1, 2, 3, 4], level_counts), 1):
        generated_k = 0
        while generated_k < target_k:
            ovr, desc_str = sample_mass_random_dist(level=lvl)

            if is_duplicate_real_mutant(ovr, p_wt_default):
                n_dedup_blocked += 1
                continue

            synth_name = f"MASS_{desc_str}_L{lvl}_{generated_k+1}"
            tasks_to_simulate.append((task_idx, synth_name, ovr))
            task_idx += 1
            generated_k += 1

        print(f"    * Level {lvl} (k_{lvl}): {target_k:,} samples selected.")

    total_tasks = len(tasks_to_simulate)
    print(f" -> Completed sampling in {time.time()-t_start_sample:.2f}s! Total tasks: {total_tasks:,} (D_real deduplication blocked {n_dedup_blocked:,} candidates)")

    # --------------------------------------------------------------------------
    # Phase 3: Parallel ODE Simulation (60 Workers)
    # --------------------------------------------------------------------------
    print(f"\n[Phase 3/4] Launching High-Throughput ODE Solver ({num_workers} parallel workers)...")
    t_sim_start = time.time()

    data_list = []
    params_list = []
    mutant_names_list = []
    labels_list = []
    pattern_labels_list = []

    n_discarded_mass = 0
    n_failed_ode = 0
    completed_count = 0

    chunksize = min(200, max(1, total_tasks // (num_workers * 4)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, mname, y_sim, bio_l, pat_l, ovr in executor.map(_worker_task_executor, tasks_to_simulate, chunksize=chunksize):
            completed_count += 1

            if y_sim is None:
                if bio_l == "Discarded_Mass":
                    n_discarded_mass += 1
                else:
                    n_failed_ode += 1
                continue

            data_list.append(y_sim)
            mutant_names_list.append(mname)
            labels_list.append(bio_l)
            pattern_labels_list.append(pat_l)

            p_vec = [float(ovr.get(k, _defaults.get(k, 0.0))) for k in PARAM_KEYS]
            params_list.append(p_vec)

            if completed_count % 2500 == 0 or completed_count == total_tasks:
                elapsed = time.time() - t_sim_start
                rate = completed_count / max(elapsed, 0.001)
                pct = completed_count / total_tasks * 100.0
                print(f"  |-- [{completed_count:6d}/{total_tasks:6d}] ({pct:5.1f}%) | Valid: {len(data_list):6d} | Discarded(Mass>10): {n_discarded_mass:4d} | Speed: {rate:5.1f} samples/s")

    elapsed_sim = time.time() - t_sim_start
    print(f" -> Simulation Completed in {elapsed_sim:.2f}s ({completed_count/elapsed_sim:.1f} samples/s).")

    # --------------------------------------------------------------------------
    # Phase 4: Packaging & Saving .npz Array Files
    # --------------------------------------------------------------------------
    print("\n[Phase 4/4] Packaging & Saving Dataset Array Files...")
    data_arr = np.array(data_list, dtype=np.float32)
    params_arr = np.array(params_list, dtype=np.float32)
    param_names_arr = np.array(PARAM_KEYS, dtype=object)
    mutant_names_arr = np.array(mutant_names_list, dtype=object)
    labels_arr = np.array(labels_list, dtype=object)
    pattern_labels_arr = np.array(pattern_labels_list, dtype=object)

    np.savez_compressed(
        output_path,
        data=data_arr,
        params=params_arr,
        param_names=param_names_arr,
        mutant_names=mutant_names_arr,
        labels=labels_arr,
        pattern_labels=pattern_labels_arr
    )

    file_size_gb = os.path.getsize(output_path) / (1024 ** 3)

    # Sync to secondary output path if available
    alt_dir = "/home/users/zsy/mydata/budding_yeast"
    if os.path.exists(alt_dir) and output_dir != alt_dir:
        try:
            alt_path = os.path.join(alt_dir, output_name)
            if not os.path.exists(alt_path):
                import shutil
                shutil.copy2(output_path, alt_path)
                print(f" -> Synchronized to secondary path: {alt_path}")
        except Exception as e:
            print(f" -> Note on secondary sync: {e}")

    print("=====================================================================================")
    print(f" MASS Random + Mutant-Aware Distribution Dataset Generation Summary")
    print("=====================================================================================")
    print(f" Dataset Export Path               : {output_path}")
    print(f" File Size                         : {file_size_gb:.2f} GB")
    print(f" Data Tensor Shape (N, Vars, Steps): {data_arr.shape}")
    print(f" Params Matrix Shape (N, P_dim)    : {params_arr.shape}")
    print(f" Total Tasks Simulated             : {completed_count}")
    print(f" Total Validated Samples Saved     : {len(data_arr)} ({len(data_arr)/completed_count*100:.2f}% yield)")
    print(f" Deduplication Blocked (Real)      : {n_dedup_blocked}")
    print(f" Discarded Divergent (Mass > 10)   : {n_discarded_mass}")
    print("=====================================================================================")
    print("\n[SUCCESS] Generator Execution Finished Cleanly!")
    return output_path


if __name__ == '__main__':
    NUM_SAMPLES = 100000
    NUM_WORKERS = min(60, os.cpu_count() or 4)
    LEVEL_RATIOS = [0.25, 0.35, 0.25, 0.15]
    DATASET_NAME = f"mass_random_dist_{NUM_SAMPLES}_210m_500s.npz"

    if os.name == 'nt':
        OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    else:
        OUTPUT_DIR = "/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast"

    import argparse
    parser = argparse.ArgumentParser(description="MASS Random + Mutant-Aware Parameter Distribution Generator")
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES, help="Number of synthetic samples to generate")
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help="Parallel CPU workers")
    parser.add_argument('--level_ratios', type=float, nargs=4, default=LEVEL_RATIOS, help="Proportions for Level 1, 2, 3, 4 tiers")
    parser.add_argument('--output_name', type=str, default=None, help="Custom output npz filename")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()

    final_output_name = args.output_name if args.output_name is not None else (
        DATASET_NAME if args.samples == NUM_SAMPLES else f"mass_random_dist_{args.samples}_210m_500s.npz"
    )

    generate_mass_random_dist_dataset(
        n_synthetic=args.samples,
        level_ratios=args.level_ratios,
        num_workers=args.workers,
        output_name=final_output_name,
        output_dir=args.output_dir
    )
