#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_pure_random_dataset.py
================================================================================
Pure Random Sampling Dataset Generator based on mutant_aware.py.

Core Principles:
1. Everything (biophysical ODEs, 27 genes, mutation types, parameter distributions,
   conflict resolution, canonical ODE solver, and 126 Real Mutants) is strictly
   imported from mutant_aware.py.
2. ZERO TOPOLOGY:
   - No MODULE_EDGES, no networkx graph, no distance/subgraph metrics.
   - Genes are drawn completely uniformly at random: P(g) = 1/27.
   - Mutation types (modes) are drawn completely uniformly at random from each
     gene's available mode set: P(m|g) = 1/k_g.
3. Level Quotas (Confirmed Option 2: Equal Weights):
   - Level 1 (k=1): 25% (25,000 samples)
   - Level 2 (k=2): 25% (25,000 samples)
   - Level 3 (k=3): 25% (25,000 samples)
   - Level 4 (k=4): 25% (25,000 samples)
4. Physiological Carbon Decoupling:
   - If mutant contains GAL-OE -> mdt = 150.0 (Galactose).
   - Otherwise -> Glc (90.0): 75%, Gal (150.0): 18%, Raff (160.0): 7%.
5. Strict Deduplication against all 126 Real Mutants via is_duplicate_real_mutant.
6. Standard Output Packaging:
   - data           : (100126, 39, 500) float32
   - params         : (100126, 141) float64
   - param_names    : (141,) string array
   - mutant_names   : (100126,) string array (126 Real Mutants first, then synthetic)
   - labels         : (100126,) viability labels
   - pattern_labels : (100126,) growth pattern labels
================================================================================
"""

import os
import sys
import time
import argparse
import concurrent.futures
import numpy as np

# Ensure current directory and project root are in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import mutant_aware as ma


def sample_pure_random_mutant(level, rng, enable_ko_interval=False, ko_interval_max=1e-4):
    """
    Samples a synthetic mutant candidate with zero graph topology.
    - Level k in {1, 2, 3, 4} dictates exactly k mutated genes.
    - Genes are chosen uniformly at random without replacement.
    - Modes are chosen uniformly at random from the gene's available modes.
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

    # Same-gene composite: expression x structure dual mode (25% chance if eligible)
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

    # Resolve parameter collision conflicts
    pairs = ma._resolve_conflicts(pairs)
    if not pairs:
        return sample_pure_random_mutant(level, rng, enable_ko_interval=enable_ko_interval, ko_interval_max=ko_interval_max)

    has_gal_oe = False
    mut_descriptions = []

    for g_name, m_chosen in pairs:
        mut_descriptions.append(f"{g_name}[{m_chosen}]")
        if m_chosen == 'GAL-OE':
            has_gal_oe = True

        # Knockout handling: Hard Zero (default) or Interval Sampling [0, ko_interval_max]
        if m_chosen == 'KO':
            if enable_ko_interval:
                # Sample within [0, ko_interval_max] strictly bounded <= 1e-4
                temp_ovr = {}
                ma.apply_gene_mutation(g_name, 'KO', 0.0, p_wt, p_mut, temp_ovr)
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
                ma.apply_gene_mutation(g_name, 'KO', 0.0, p_wt, p_mut, ovr)
        elif m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db']:
            alpha = ma.sample_scaling_factor(m_chosen, g_name)
            ma.apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)
        else:
            ma.apply_gene_mutation(g_name, m_chosen, 1.0, p_wt, p_mut, ovr)

    # Carbon source handling: aligned with native physiological distribution
    if has_gal_oe:
        ovr['mdt'] = 150.0
    elif 'mdt' not in ovr:
        ovr['mdt'] = float(rng.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))

    desc_str = "+".join(mut_descriptions)
    return ovr, desc_str


def generate_pure_random_dataset(num_samples=100000,
                                 num_workers=50,
                                 level_ratios=(0.25, 0.25, 0.25, 0.25),
                                 output_name=None,
                                 output_dir="/home/users/zsy/mydata/budding_yeast",
                                 seed=42,
                                 enable_collision_rescue=True,
                                 enable_ko_interval=False,
                                 ko_interval_max=1e-4):
    """Generates 100,000 purely random synthetic samples + 126 Real Mutants."""
    if ko_interval_max > 1e-4:
        raise ValueError(f"ko_interval_max ({ko_interval_max}) must not exceed 1e-4!")

    rng = np.random.RandomState(seed)
    if output_name is None:
        if enable_ko_interval:
            if enable_collision_rescue:
                output_name = f"mutant_aware_pure_random_rescue_koint_{num_samples}_210m_500s.npz"
            else:
                output_name = f"mutant_aware_pure_random_ko_interval_{num_samples}_210m_500s.npz"
        elif enable_collision_rescue:
            output_name = f"mutant_aware_pure_random_rescue_{num_samples}_210m_500s.npz"
        else:
            output_name = f"mutant_aware_pure_random_{num_samples}_210m_500s.npz"

    dataset_path = os.path.join(output_dir, output_name)

    print("\n" + "=" * 85)
    print(f" PURE RANDOM DATASET GENERATOR (Target: {num_samples} samples, Seed: {seed})")
    print(f" Output Target File: {dataset_path}")
    print(f" Level Quotas (25/25/25/25): {level_ratios}")
    print(f" Collision Rescue Switch: {'ENABLED' if enable_collision_rescue else 'DISABLED'}")
    print(f" KO Interval Sampling Switch: {'ENABLED' if enable_ko_interval else 'DISABLED'} (U[0, {ko_interval_max}])")
    print("=" * 85, flush=True)

    # Calculate level quotas
    n_l1 = int(round(num_samples * level_ratios[0]))
    n_l2 = int(round(num_samples * level_ratios[1]))
    n_l3 = int(round(num_samples * level_ratios[2]))
    n_l4 = num_samples - (n_l1 + n_l2 + n_l3)

    quota_plan = [
        (1, n_l1, "Level 1: Single Mutants (k=1) Uniform Random"),
        (2, n_l2, "Level 2: Double Mutants (k=2) Uniform Random"),
        (3, n_l3, "Level 3: Triple Mutants (k=3) Uniform Random"),
        (4, n_l4, "Level 4: Quadruple Mutants (k=4) Uniform Random"),
    ]

    print("[Phase 1/4] Generating Task Definitions & Enforcing Zero-Leakage Deduplication...")
    all_tasks = {}
    ordered_names = []

    # 1. Real Mutants (First 126 entries)
    for m_id, (m_name, p_dict) in ma.REAL_MUTANTS.items():
        t_name = f"{m_id}_{m_name.replace(' ', '_')}"
        all_tasks[t_name] = p_dict
        ordered_names.append(t_name)
    print(f"  -> Added {len(ordered_names)} Real Mutants as gold benchmark entries.", flush=True)

    # 2. Synthetic Mutants according to Equal Quota Plan
    duplicates_blocked = 0
    duplicates_rescued = 0
    t0_prep = time.time()

    p_wt_default = ma.get_default_params()
    p_wt_default.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

    for level, quota, desc in quota_plan:
        print(f"  -> Generating {desc}: {quota} samples...", flush=True)
        count = 0
        while count < quota:
            ovr, desc_str = sample_pure_random_mutant(
                level, rng,
                enable_ko_interval=enable_ko_interval,
                ko_interval_max=ko_interval_max
            )

            # Strict 141-D Zero-Leakage Deduplication against Real Mutants
            if ma.is_duplicate_real_mutant(ovr, p_wt_default):
                if enable_collision_rescue:
                    # 碰撞拯救：70% 概率连续微衰减 (DAmP/ts/pLOF)，30% 概率碳源漂移 (Gal/Raff)
                    rescued_ovr, rescued_desc = ma.rescue_collision_candidate(ovr, desc_str, p_wt_default)
                    if rescued_ovr is not None and not ma.is_duplicate_real_mutant(rescued_ovr, p_wt_default):
                        ovr = rescued_ovr
                        desc_str = rescued_desc
                        duplicates_rescued += 1
                    else:
                        duplicates_blocked += 1
                        continue
                else:
                    duplicates_blocked += 1
                    continue

            sample_id = len(all_tasks) - len(ma.REAL_MUTANTS) + 1
            task_name = f"RANDOM_{desc_str}_L{level}_{sample_id}"
            all_tasks[task_name] = ovr
            ordered_names.append(task_name)
            count += 1

    print(f"  -> Task Generation completed in {time.time() - t0_prep:.2f}s.")
    print(f"  -> Total Tasks: {len(all_tasks)} (Real: {len(ma.REAL_MUTANTS)}, Synthetic: {num_samples})")
    print(f"  -> KO Interval Sampling Switch    : {'ENABLED' if enable_ko_interval else 'DISABLED'} (U[0, {ko_interval_max}])")
    print(f"  -> Collision Rescue Switch        : {'ENABLED' if enable_collision_rescue else 'DISABLED'}")
    print(f"  -> Rescued Real-Mutant Collisions : {duplicates_rescued} candidates.")
    print(f"  -> Blocked Real-Mutant Collisions : {duplicates_blocked} candidates.\n", flush=True)

    # Phase 2: Parallel ODE Simulation
    print(f"[Phase 2/4] Simulating {len(ordered_names)} Tasks with {num_workers} Parallel Workers...", flush=True)
    t_start_sim = time.time()
    all_results = {}
    bio_labels_dict = {}
    pattern_labels_dict = {}

    sim_counts = {
        'real_success': 0, 'synth_success': 0,
        'discarded_real': 0, 'discarded_synth': 0,
        'failed_real': 0, 'failed_synth': 0
    }

    log_step = max(500, len(ordered_names) // 50)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_name = {
            executor.submit(ma.simulate_mutant, name, all_tasks[name], 210, 500): name
            for name in ordered_names
        }
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_name)):
            name = future_to_name[future]
            is_synth = "_L" in name
            try:
                m_name, y_uniform, m_bio_label, m_pattern_label = future.result(timeout=25.0)
                if y_uniform is not None:
                    all_results[m_name] = y_uniform
                    bio_labels_dict[m_name] = m_bio_label
                    pattern_labels_dict[m_name] = m_pattern_label
                    if is_synth: sim_counts['synth_success'] += 1
                    else: sim_counts['real_success'] += 1
                elif m_bio_label == "Discarded_Mass":
                    if is_synth: sim_counts['discarded_synth'] += 1
                    else: sim_counts['discarded_real'] += 1
                else:
                    if is_synth: sim_counts['failed_synth'] += 1
                    else: sim_counts['failed_real'] += 1
            except Exception:
                if is_synth: sim_counts['failed_synth'] += 1
                else: sim_counts['failed_real'] += 1

            if (idx + 1) % log_step == 0 or (idx + 1) == len(ordered_names):
                elapsed = max(0.001, time.time() - t_start_sim)
                speed = (idx + 1) / elapsed
                valid_total = sim_counts['real_success'] + sim_counts['synth_success']
                pct = ((idx + 1) / len(ordered_names)) * 100
                print(f"  |-- [{idx + 1:6d}/{len(ordered_names)}] ({pct:5.1f}%) | "
                      f"Valid: {valid_total:6d} | Discarded: {sim_counts['discarded_synth']:4d} | "
                      f"Failed: {sim_counts['failed_synth']:4d} | Speed: {speed:5.1f} samples/s", flush=True)

    elapsed_time = time.time() - t_start_sim
    print(f" -> Simulation Completed in {elapsed_time:.2f}s ({len(ordered_names) / max(0.001, elapsed_time):.1f} samples/s).\n", flush=True)

    # Phase 3: Packaging & .npz Export
    print("[Phase 3/4] Packaging & Exporting .npz Array...", flush=True)
    defaults = ma.get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    param_keys = sorted([k for k in defaults.keys() if k != 'is_mutant_104'])

    ordered_tensors = []
    final_names = []
    final_params = []
    final_bio_labels = []
    final_pattern_labels = []

    real_final_names = []
    synth_final_names = []

    for name in ordered_names:
        if name in all_results:
            ordered_tensors.append(all_results[name])
            final_names.append(name)
            final_bio_labels.append(bio_labels_dict[name])
            final_pattern_labels.append(pattern_labels_dict[name])

            if "_L" in name:
                synth_final_names.append(name)
            else:
                real_final_names.append(name)

            overrides = all_tasks[name]
            p_vec = [overrides.get(k, defaults[k]) for k in param_keys]
            final_params.append(p_vec)

    all_tensors = np.array(ordered_tensors, dtype=np.float32)
    all_params = np.array(final_params, dtype=np.float64)
    all_bio_labels = np.array(final_bio_labels)
    all_pattern_labels = np.array(final_pattern_labels)

    os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)
    np.savez_compressed(dataset_path,
                        data=all_tensors,
                        params=all_params,
                        param_names=np.array(param_keys),
                        mutant_names=np.array(final_names),
                        labels=all_bio_labels,
                        pattern_labels=all_pattern_labels)

    total_valid = len(final_names)
    print("=" * 85)
    print(f" PURE RANDOM DATASET EXPORT REPORT")
    print("=" * 85)
    print(f"Dataset Export Path               : {dataset_path}")
    print(f"Data Tensor Shape (N, Vars, Steps): {all_tensors.shape}")
    print(f"Params Matrix Shape (N, P_dim)    : {all_params.shape}")
    print(f"Total Parameter Keys Count        : {len(param_keys)}")
    print(f"Real Mutants Validated            : {len(real_final_names)} / {len(ma.REAL_MUTANTS)}")
    print(f"Synthetic Mutants Validated       : {len(synth_final_names)} / {num_samples}")
    print(f"Total Validated Samples Saved     : {total_valid}")
    print("-" * 85)

    # Pattern Distribution Breakdown
    pattern_categories = [
        "Pattern_A: Multi-Division (>=2)",
        "Pattern_B: Single-Division (==1)",
        "Pattern_C: Exponential-Arrest (==0)"
    ]
    for pat in pattern_categories:
        c_r = list(final_pattern_labels[:len(real_final_names)]).count(pat)
        c_s = list(final_pattern_labels[len(real_final_names):]).count(pat)
        c_t = list(final_pattern_labels).count(pat)
        print(f"  {pat:<36} | Real: {c_r:3d} | Synth: {c_s:6d} ({c_s/len(synth_final_names)*100:4.1f}%) | Total: {c_t:6d}")
    print("=" * 85 + "\n", flush=True)

    return dataset_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure Random Dataset Generator (mutant_aware baseline)")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of synthetic samples (default: 100,000)")
    parser.add_argument("--num_workers", type=int, default=50, help="ProcessPoolExecutor workers (default: 50)")
    parser.add_argument("--output_name", type=str, default=None, help="Output npz filename (defaults dynamically based on rescue switch)")
    parser.add_argument("--output_dir", type=str, default="/home/users/zsy/mydata/budding_yeast")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_collision_rescue", dest="enable_collision_rescue", action="store_true", default=True,
                        help="Enable collision rescue for real-mutant collisions (default: True)")
    parser.add_argument("--disable_collision_rescue", dest="enable_collision_rescue", action="store_false",
                        help="Disable collision rescue and drop collisions directly")
    parser.add_argument("--enable_ko_interval", dest="enable_ko_interval", action="store_true", default=False,
                        help="Enable interval sampling for KO mutations (default: False)")
    parser.add_argument("--disable_ko_interval", dest="enable_ko_interval", action="store_false",
                        help="Disable interval sampling for KO mutations (use hard zero 0.0)")
    parser.add_argument("--ko_interval_max", type=float, default=1e-4,
                        help="Upper bound for KO interval sampling (must <= 1e-4, default: 1e-4)")
    args = parser.parse_args()

    generate_pure_random_dataset(
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        level_ratios=(0.25, 0.25, 0.25, 0.25),
        output_name=args.output_name,
        output_dir=args.output_dir,
        seed=args.seed,
        enable_collision_rescue=args.enable_collision_rescue,
        enable_ko_interval=args.enable_ko_interval,
        ko_interval_max=args.ko_interval_max
    )
