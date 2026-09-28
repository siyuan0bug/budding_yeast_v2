#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_distribution_ablation_twins.py
================================================================================
Strictly Controlled Single-Variable A/B Twin Dataset Generator:
Comparing Parameter Sampling Distribution Regimes:
- Twin B: Complete Regime B (Continuous Physical Envelopes & Boundary Buffers, mutant_aware.py v5.2)
- Twin A-Assign: Regime B Skeletons/MC/KO/GAL-OE, but Regime A Sampling & Assignment for:
  ts, db, APC-A, TAB6-1, CA, CSL, CKO, scc1-RR, PM.

Core Design (Strict 1-to-1 Paired Control):
1. In each generation loop, the mutated gene list, mutation modes, carbon source (mdt),
   and shared parameters (MC discrete counts, GAL-OE continuous values, KO hard zeros)
   are 100% IDENTICAL across Twin B and Twin A-Assign.
2. The ONLY difference between the two datasets:
   For [ts, db, APC-A, TAB6-1, CA, CSL, CKO, scc1-RR, PM]:
   - Twin B applies Regime B continuous physical envelopes and boundary buffers.
   - Twin A-Assign applies Regime A sampling distributions and direct assignments.
3. Both datasets are evaluated through identical Chen (2004) BDF ODE simulations
   (210 min, 500 steps, exact discrete events and viability classifications).
================================================================================
"""

import os
import sys
import time
import math
import argparse
import concurrent.futures
import numpy as np

# Ensure current directory is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import mutant_aware as ma

TARGET_MODES = {'ts', 'db', 'APC-A', 'TAB6-1', 'CA', 'CSL', 'CKO', 'scc1-RR', 'PM'}


# ==============================================================================
# Regime A: Sampling & Assignment Engine (from mass_dataset_generator.py)
# ==============================================================================

def sample_scaling_factor_A(mode, g_name=None):
    """Exact sampling distribution from mass_dataset_generator.py."""
    if mode == 'ts':
        if g_name == 'CDC14':
            return float(np.random.uniform(0.05, 0.35))
        u = np.random.beta(a=0.5, b=2.0)
        return float(0.20 * u)
    elif mode == 'db':
        if g_name == 'CLB2':
            return float(np.random.uniform(0.04, 0.15))
        elif g_name == 'PDS1':
            return float(np.random.uniform(0.0, 0.05))
        elif g_name in ('CLB5', 'SIC1'):
            return 0.0
        else:
            return 0.0
    return 1.0


def apply_gene_mutation_A(g_name, m_chosen, alpha, p_wt, p_mut, ovr):
    """
    Exact parameter modification engine for target modes from mass_dataset_generator.py.
    """
    cdk_eps = ['ec1k2', 'ec1n2', 'ec1b5', 'ec1b2', 'ec1n3', 'ef6k2', 'ef6n2', 'ef6b5', 'ef6b2', 'ef6n3',
               'eicdhn2', 'eicdhn3', 'eicdhb5', 'eicdhb2', 'esbfn2', 'esbfn3', 'esbfb5', 'eorib5', 'eorib2',
               'ebudn2', 'ebudn3', 'ebudb5']

    if m_chosen == 'ts':
        if g_name == 'CLN2':
            ovr['ksn2p'] = p_wt['ksn2p'] * alpha
            ovr['ksn2pp'] = p_wt['ksn2pp'] * alpha
        elif g_name == 'CLB2':
            ovr['ksb2p'] = p_wt['ksb2p'] * alpha
            ovr['ksb2pp'] = p_wt['ksb2pp'] * alpha
        elif g_name == 'CDC20':
            ovr['ks20p'] = p_wt['ks20p'] * alpha
            ovr['ks20pp'] = p_wt['ks20pp'] * alpha
        elif g_name == 'CDC14':
            if np.random.rand() < 0.50:
                ovr['ks14'] = 0.0
            else:
                ovr['ks14'] = p_wt['ks14']
                ovr['kppc1'] = p_wt['kppc1'] * alpha
                ovr['kppf6'] = p_wt['kppf6'] * alpha
                ovr['kaswi'] = p_wt['kaswi'] * alpha
                ovr['kacdhpp'] = p_wt['kacdhpp'] * alpha
        elif g_name == 'NET1':
            ovr['kasrent'] = p_wt['kasrent'] * alpha
            ovr['kasrentp'] = p_wt['kasrentp'] * alpha
        elif g_name in ['PPX', 'CDC5_Proxy']:
            ovr['kppnetpp'] = 0.05
        elif g_name == 'CDC15':
            ovr['kkpnetpp'] = 0.0
        elif g_name == 'TEM1':
            ovr['Tem1T'] = 0.0
            ovr['ka15pp'] = 0.002
            ovr['init_Tem1'] = 0.0  # safe initialization protection
        elif g_name == 'ESP1':
            ovr['kasesp'] = 0.1
            ovr['kdiesp'] = 0.001
        elif g_name == 'APC_Core':
            ovr['ks20p'] = 0.0
            ovr['ks20pp'] = 0.0
            ovr['kscdh'] = 0.0
            ovr['init_CDH1T'] = 0.0
            ovr['init_CDH1'] = 0.0
        elif g_name == 'Mcm1':
            ovr['kamcm'] = 0.13
        elif g_name == 'CDC28':
            for ep in cdk_eps:
                ovr[ep] = p_wt[ep] * alpha

    elif m_chosen == 'db':
        if g_name == 'CLB2':
            ovr['kdb2P'] = 0.0
            ovr['kdb2pp'] = p_wt['kdb2pp'] * alpha
        elif g_name == 'CLB5':
            ovr['kdb5pp'] = 0.0
        elif g_name == 'PDS1':
            ovr['kd2pdspp'] = 0.0
            ovr['kd3pdspp'] = 0.0
        elif g_name == 'SIC1':
            ovr['kd3c1'] = 0.0
        elif g_name in ['PPX', 'CDC5_Proxy']:
            ovr['kdppxpp'] = 0.0

    elif m_chosen == 'CA':
        if g_name == 'CDH1':
            ovr['kicdhpp'] = 0.0
            ovr['kscdh'] = 0.03
        elif g_name == 'CLB2':
            ovr['kdib2'] = 0.0
        elif g_name == 'SBF_MBF':
            ovr['is_sbf_ca'] = 1
            ovr['kasbf'] = 0.38
            ovr['kisbfpp'] = 0.0

    elif m_chosen == 'CSL':
        ovr['Vi20'] = 0.01
        ovr['Vi20_active'] = 0.01

    elif m_chosen == 'CKO':
        ovr['kbub2'] = 0.0
        ovr['BUB2_active'] = 0.0
        ovr['init_BUB2'] = 0.0

    elif m_chosen == 'APC-A' or (g_name in ['CDC20', 'APC_Core'] and m_chosen == 'APC-A'):
        ovr['ka20pp'] = 0.0

    elif m_chosen == 'TAB6-1':
        ovr['kasrent'] = 10.0
        ovr['kasrentp'] = 0.05

    elif m_chosen == 'scc1-RR':
        ovr['kasesp'] = 0.0

    elif m_chosen == 'PM':
        ovr['kd2c1'] = 0.0
        ovr['kd3c1'] = 0.0


# ==============================================================================
# Twin Sample Construction: Strictly Paired Parameter Assignment
# ==============================================================================

def sample_twin_mutant_pair(level, defaults):
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

    # Step 1: 100% Pure Random Sampling of Genes and Modes (Zero Topology, Zero Motifs)
    # - 27 genes drawn completely uniformly at random without replacement: P(g) = 1/27
    # - Mutation modes drawn completely uniformly at random from available set: P(m|g) = 1/k_g
    ki = int(level)
    selected_genes = list(np.random.choice(ma.GENE_NAMES, size=ki, replace=False))
    pairs = []
    for g in selected_genes:
        modes = sorted(ma.GROUP_BY_NAME[g]['modes'].keys())
        m = str(np.random.choice(modes))
        pairs.append((g, m))

    # Same-gene composite: expression x structure dual mode (25% chance if eligible)
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

    # Same-parameter touch-set conflict resolution
    pairs = ma._resolve_conflicts(pairs)
    if not pairs:
        return sample_twin_mutant_pair(level, defaults)

    # Step 2: Build Descriptions & Shared Carbon Source
    has_gal_oe = any(m == 'GAL-OE' for _, m in pairs)
    if has_gal_oe:
        mdt_shared = 150.0
    else:
        mdt_shared = float(np.random.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))

    mut_descriptions = [f"{g}[{m}]" for g, m in pairs]
    desc_str = "+".join(mut_descriptions)

    ovr_B = {'mdt': mdt_shared}
    ovr_A = {'mdt': mdt_shared}

    p_mut_B = p_wt.copy()
    p_mut_A = p_wt.copy()

    # Step 3: Apply Mutations across Twin B and Twin A
    for g_name, m_chosen in pairs:
        # Branch 1: Invariant Non-Target Modes (MC, GAL-OE, KO, DAmP, pLOF, perturb, etc.)
        if m_chosen not in TARGET_MODES:
            # Sample alpha ONCE so both Twin B and Twin A share the exact same numeric value!
            alpha_shared = ma.sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'DAmP', 'pLOF'] else 1.0
            ma.apply_gene_mutation(g_name, m_chosen, alpha_shared, p_wt, p_mut_B, ovr_B)
            ma.apply_gene_mutation(g_name, m_chosen, alpha_shared, p_wt, p_mut_A, ovr_A)
        else:
            # Branch 2: Target Modes (ts, db, APC-A, TAB6-1, CA, CSL, CKO, scc1-RR, PM)
            # Twin B: Regime B (continuous envelopes & boundary buffers)
            alpha_B = ma.sample_scaling_factor(m_chosen, g_name) if m_chosen in ['ts', 'db'] else 1.0
            ma.apply_gene_mutation(g_name, m_chosen, alpha_B, p_wt, p_mut_B, ovr_B)

            # Twin A-Assign: Regime A (sampling distributions & direct assignments)
            alpha_A = sample_scaling_factor_A(m_chosen, g_name) if m_chosen in ['ts', 'db'] else 1.0
            apply_gene_mutation_A(g_name, m_chosen, alpha_A, p_wt, p_mut_A, ovr_A)

    return ovr_B, ovr_A, desc_str


# ==============================================================================
# Simulation & Packaging Pipeline
# ==============================================================================

def simulate_twin_task(task_tuple):
    """Worker task simulating both Twin B and Twin A overrides for a given mutant name."""
    name, ovr_B, ovr_A = task_tuple

    # Simulate Twin B
    res_B = ma.simulate_mutant(f"{name}_B", ovr_B, tf=210, ns=500)
    # Simulate Twin A
    res_A = ma.simulate_mutant(f"{name}_A", ovr_A, tf=210, ns=500)

    return name, res_B, res_A


def generate_twins(num_samples=100000,
                   level_ratios=(0.25, 0.25, 0.25, 0.25),
                   output_dir="/home/users/zsy/mydata/budding_yeast",
                   output_name_B="pure_random_distribution_twin_b_continuous_100000_210m_500s.npz",
                   output_name_A="pure_random_distribution_twin_a_direct_100000_210m_500s.npz",
                   num_workers=60,
                   seed=42):
    """
    Simultaneously produces Twin B (Continuous Envelopes) and Twin A (Direct Assignments)
    with 100% matched mutant names, gene combinations, and shared parameter draws.
    """
    np.random.seed(seed)

    print("=" * 85)
    print("Strictly Controlled Single-Variable A/B Twin Dataset Generator")
    print("Comparing Parameter Distribution Regimes:")
    print("  - Twin B: Complete Regime B (Continuous Physical Envelopes & Boundary Buffers)")
    print("  - Twin A: Regime A Sampling Distributions & Direct Assignments for ts/db/APC-A/TAB6-1")
    print(f"Target Samples per Dataset: {num_samples} (+ 126 Real Mutants)")
    print(f"Output Directory: {output_dir}")
    print(f"Twin B Destination: {os.path.join(output_dir, output_name_B)}")
    print(f"Twin A Destination: {os.path.join(output_dir, output_name_A)}")
    print("=" * 85, flush=True)

    os.makedirs(output_dir, exist_ok=True)
    defaults = ma.get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    param_keys = sorted([k for k in defaults.keys() if k != 'is_mutant_104'])

    # Phase 1: Real Mutants (Both datasets contain identical 126 Real Mutants)
    print("\n[Phase 1/4] Preparing 126 Real Mutants...")
    real_tasks = {}
    real_names = []
    for mid, (mname, movr) in ma.REAL_MUTANTS.items():
        sample_name = f"{mid}_{mname}".replace(" ", "_")
        real_names.append(sample_name)
        real_tasks[sample_name] = movr

    # Phase 2: Sample Skeletons & Construct Twin Override Dictionaries
    print(f"\n[Phase 2/4] Sampling {num_samples} Matched Synthetic Mutants...")
    r1, r2, r3, r4 = level_ratios
    tot = r1 + r2 + r3 + r4
    n_l1 = int(math.floor((r1 / tot) * num_samples))
    n_l2 = int(math.floor((r2 / tot) * num_samples))
    n_l3 = int(math.floor((r3 / tot) * num_samples))
    n_l4 = num_samples - n_l1 - n_l2 - n_l3

    levels = [1] * n_l1 + [2] * n_l2 + [3] * n_l3 + [4] * n_l4
    np.random.shuffle(levels)

    twin_tasks = {}  # name -> (ovr_B, ovr_A)
    ordered_synth_names = []

    success = 0
    attempts = 0
    max_attempts = num_samples * 10

    while success < num_samples and attempts < max_attempts:
        attempts += 1
        lvl = levels[success]
        ovr_B, ovr_A, desc_str = sample_twin_mutant_pair(lvl, defaults)

        # Check deduplication against D_real (check ovr_B)
        if ma.is_duplicate_real_mutant(ovr_B, defaults):
            continue

        success += 1
        name = f"MASS_{desc_str}_L{lvl}_{success}".replace(" ", "_")
        ordered_synth_names.append(name)
        twin_tasks[name] = (ovr_B, ovr_A)

    print(f" -> Successfully sampled {success} matched twin tasks in {attempts} attempts.")

    # Phase 3: Parallel ODE Simulation for Both Twins
    print(f"\n[Phase 3/4] Running Parallel ODE Simulations on {num_workers} Workers...")
    t_start = time.time()

    all_names = real_names + ordered_synth_names

    # Collect simulation tasks:
    # Real mutants share identical ovr for both B and A
    tasks_to_run = []
    for name in real_names:
        tasks_to_run.append((name, real_tasks[name], real_tasks[name]))
    for name in ordered_synth_names:
        ovr_B, ovr_A = twin_tasks[name]
        tasks_to_run.append((name, ovr_B, ovr_A))

    results_B = {}
    results_A = {}
    log_step = max(50, len(tasks_to_run) // 40)

    import multiprocessing
    with multiprocessing.Pool(processes=num_workers) as pool:
        for idx, (name, res_B, res_A) in enumerate(pool.imap_unordered(simulate_twin_task, tasks_to_run, chunksize=25)):
            try:
                # res is (name, yu, bio_l, pat_l)
                _, yu_B, bio_B, pat_B = res_B
                _, yu_A, bio_A, pat_A = res_A

                if yu_B is not None and yu_A is not None:
                    results_B[name] = (yu_B, bio_B, pat_B)
                    results_A[name] = (yu_A, bio_A, pat_A)
            except Exception:
                pass

            if (idx + 1) % log_step == 0 or (idx + 1) == len(tasks_to_run):
                elapsed = max(0.001, time.time() - t_start)
                speed = (idx + 1) / elapsed
                valid_count = len(results_B)
                pct = ((idx + 1) / len(tasks_to_run)) * 100
                print(f"  |-- [{idx + 1:6d}/{len(tasks_to_run)}] ({pct:5.1f}%) | "
                      f"Both Valid: {valid_count:6d} | Speed: {speed:5.1f} twin-samples/s", flush=True)

    elapsed_total = time.time() - t_start
    print(f" -> Simulations Completed in {elapsed_total:.2f}s ({len(tasks_to_run) / max(0.001, elapsed_total):.1f} samples/s).")

    # Phase 4: Symmetrical Packaging
    print(f"\n[Phase 4/4] Symmetrically Packaging Both Datasets...")

    # Filter strictly for samples valid in BOTH datasets to maintain 100% 1-to-1 match
    valid_common_names = [n for n in all_names if n in results_B and n in results_A]
    print(f" -> Common 1-to-1 Matched Valid Samples: {len(valid_common_names)}")

    for twin_tag, res_dict, out_fname in [('B_Continuous', results_B, output_name_B),
                                          ('A_DirectAssign', results_A, output_name_A)]:
        out_path = os.path.join(output_dir, out_fname)
        data_tensors = []
        final_params = []
        bio_labels = []
        pattern_labels = []

        for name in valid_common_names:
            yu, bio_l, pat_l = res_dict[name]
            data_tensors.append(yu)
            bio_labels.append(bio_l)
            pattern_labels.append(pat_l)

            # Build param vector
            if name in real_names:
                ovr = real_tasks[name]
            else:
                ovr_B_cand, ovr_A_cand = twin_tasks[name]
                ovr = ovr_B_cand if twin_tag == 'B_Continuous' else ovr_A_cand

            p_vec = [ovr.get(k, defaults[k]) for k in param_keys]
            final_params.append(p_vec)

        np.savez_compressed(
            out_path,
            data=np.array(data_tensors, dtype=np.float32),
            params=np.array(final_params, dtype=np.float64),
            param_names=np.array(param_keys),
            mutant_names=np.array(valid_common_names),
            labels=np.array(bio_labels),
            pattern_labels=np.array(pattern_labels)
        )
        print(f" -> Saved Twin Dataset [{twin_tag}]: {out_path} ({len(valid_common_names)} samples)")

    print("\n✅ All Twin Datasets successfully generated and packaged!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Strictly Matched A/B Twin Datasets")
    parser.add_argument('--num_samples', type=int, default=100000, help="Number of synthetic samples (default 100000)")
    parser.add_argument('--num_workers', type=int, default=60, help="CPU workers for parallel simulation")
    parser.add_argument('--output_dir', type=str, default="/home/users/zsy/mydata/budding_yeast", help="Output directory")
    parser.add_argument('--smoke_test', action='store_true', help="Run fast smoke test with 50 samples")
    args = parser.parse_args()

    n_samples = 50 if args.smoke_test else args.num_samples
    out_b = f"smoke_twin_b_{n_samples}.npz" if args.smoke_test else "pure_random_distribution_twin_b_continuous_100000_210m_500s.npz"
    out_a = f"smoke_twin_a_{n_samples}.npz" if args.smoke_test else "pure_random_distribution_twin_a_direct_100000_210m_500s.npz"
    workers = 8 if args.smoke_test else args.num_workers

    generate_twins(
        num_samples=n_samples,
        output_dir=args.output_dir if not args.smoke_test else os.path.join(_current_dir, "smoke_test_twins"),
        output_name_B=out_b,
        output_name_A=out_a,
        num_workers=workers
    )
