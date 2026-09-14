#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_kdpp_gated_dataset.py
================================================================================
MASS (Mutant-Aware Stratified Sampling with k-DPP Graph Diffusion Engine)
Generates full 100,000 synthetic sample dataset + 126 Real Mutants.

Replaces heuristic quotas with:
  - Graph Laplacian Diffusion k-DPP Skeleton Sampler (pure NumPy)
  - Strict preservation of all biological invariants, 141 parameter keys,
    and 126 Appendix B real mutants.
================================================================================
"""

import os
import sys
import time
import math
import argparse
import concurrent.futures
import numpy as np
from scipy.linalg import expm

# Setup paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
_up_data = os.path.abspath(os.path.join(_script_dir, '..', '..'))
if _up_data not in sys.path:
    sys.path.insert(0, _up_data)

import mutant_aware as ma


# ==============================================================================
# Graph Diffusion k-DPP Sampler
# ==============================================================================

class GraphDiffusionKDPP:
    """k-DPP Sampler based on 54-edge regulatory network diffusion kernel."""
    def __init__(self, tau=0.5, max_k=6, seed=42):
        self.rng = np.random.RandomState(seed)
        self.gene_names = list(ma.GENE_NAMES)
        self.N = len(self.gene_names)
        self.idx_map = {g: i for i, g in enumerate(self.gene_names)}
        self.max_k = max_k
        self.tau = tau

        # Adjacency matrix
        self.A = np.zeros((self.N, self.N), dtype=float)
        for u, v, _ in ma.MODULE_EDGES:
            if u in self.idx_map and v in self.idx_map:
                i, j = self.idx_map[u], self.idx_map[v]
                self.A[i, j] = 1.0
                self.A[j, i] = 1.0

        deg = np.sum(self.A, axis=1)
        d_inv_sqrt = np.array([1.0 / np.sqrt(d + 0.5) for d in deg])
        L_norm = np.eye(self.N) - d_inv_sqrt[:, None] * self.A * d_inv_sqrt[None, :]

        # Heat diffusion kernel
        K_diff = expm(-self.tau * L_norm)
        d_s = np.sqrt(np.maximum(np.diag(K_diff), 1e-12))
        self.S = K_diff / (d_s[:, None] * d_s[None, :])

        # Mild degree prior (alpha=0.15)
        q = (deg + 1.0) ** 0.15
        self.L = q[:, None] * self.S * q[None, :]

        self.eigvals, self.eigvecs = np.linalg.eigh(self.L)
        self.eigvals = np.maximum(self.eigvals, 1e-10)

        # Elementary symmetric polynomials
        self.E = np.zeros((self.max_k + 1, self.N + 1), dtype=float)
        self.E[0, :] = 1.0
        for l in range(1, self.max_k + 1):
            for n in range(1, self.N + 1):
                self.E[l, n] = self.E[l, n - 1] + self.eigvals[n - 1] * self.E[l - 1, n - 1]

    def sample(self, k):
        if k <= 0:
            return []
        if k >= self.N:
            return list(self.gene_names)

        selected_vecs = []
        rem_k = k
        for n in range(self.N, 0, -1):
            if rem_k == 0:
                break
            denom = self.E[rem_k, n]
            p = (self.eigvals[n - 1] * self.E[rem_k - 1, n - 1] / denom) if denom > 1e-14 else 0.0
            p = min(max(p, 0.0), 1.0)
            if self.rng.rand() < p:
                selected_vecs.append(n - 1)
                rem_k -= 1

        V = self.eigvecs[:, selected_vecs].copy()
        Y = []
        for _ in range(k):
            p = np.sum(V**2, axis=1)
            p_sum = np.sum(p)
            if p_sum <= 1e-12:
                unchosen = [i for i in range(self.N) if i not in Y]
                item = self.rng.choice(unchosen)
            else:
                p_norm = p / p_sum
                item = self.rng.choice(self.N, p=p_norm)
            Y.append(item)

            v_item = V[item, :].copy()
            j = np.argmax(np.abs(v_item))
            col = V[:, j].copy()
            V = V - np.outer(col, v_item) / (v_item[j] + 1e-12)
            if V.shape[1] > 1:
                q_proj, _ = np.linalg.qr(np.delete(V, j, axis=1))
                V = q_proj

        return [self.gene_names[i] for i in Y]


# ==============================================================================
# Sample Generator
# ==============================================================================

def sample_kdpp_mutant(level, kdpp_sampler, rng=np.random):
    """
    Draws a mutant using k-DPP for levels 2..4, and Latin-square grid for level 1.
    Fully adheres to physical scaling laws and collision filters.
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
        genes = kdpp_sampler.sample(level)
        pairs = [(g, ma._sample_mode(g)) for g in genes]

    # Composite expansion (expression x structure dual mode, 25%)
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

    # Touch-set conflict resolution
    pairs = ma._resolve_conflicts(pairs)
    if not pairs:
        return sample_kdpp_mutant(level, kdpp_sampler, rng)

    for g_name, m_chosen in pairs:
        mut_descriptions.append(f"{g_name}[{m_chosen}]")
        if m_chosen == 'GAL-OE':
            has_gal_oe = True
        alpha = ma.sample_scaling_factor(m_chosen, g_name) \
            if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
        ma.apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)

    # Carbon source rule
    if has_gal_oe:
        ovr['mdt'] = 150.0
    elif 'mdt' not in ovr:
        ovr['mdt'] = float(rng.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))

    desc_str = "+".join(mut_descriptions)
    return ovr, desc_str


# ==============================================================================
# Simulation Worker Wrapper
# ==============================================================================

def run_simulation_task(task_tuple):
    name, ovr = task_tuple
    return ma.simulate_mutant(name, ovr, tf=210, ns=500)


# ==============================================================================
# Main Dataset Generation Pipeline
# ==============================================================================

def generate_kdpp_dataset(output_path, n_synthetic=100000, num_workers=60, seed=42, ratios=(0.25, 0.25, 0.25, 0.25)):
    print("=" * 85)
    print(" MASS v5.2 + Graph Diffusion k-DPP DATASET GENERATION")
    print("=" * 85)
    print(f"Target Output Path       : {output_path}")
    print(f"Target Synthetic Samples : {n_synthetic:,}")
    print(f"Random Seed              : {seed}")
    print(f"Parallel Workers         : {num_workers}")
    print(f"Level Ratios (L1~L4)     : {ratios}")
    print("=" * 85, flush=True)

    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    defaults = ma.get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    param_keys = sorted([k for k in defaults.keys() if k != 'is_mutant_104'])

    all_tasks = {}
    ordered_names = []

    # Phase 1: Real Mutants
    print("\n[Phase 1/4] Preparing D_real (126 Appendix B Table S5 Real Mutants)...")
    for mid, (mname, movr) in ma.REAL_MUTANTS.items():
        sample_name = f"{mid}_{mname}".replace(" ", "_")
        ordered_names.append(sample_name)
        all_tasks[sample_name] = movr
    print(f" -> Loaded {len(ma.REAL_MUTANTS)} Real Mutants.", flush=True)

    # Phase 2: Synthetic Mutants (k-DPP Engine)
    print(f"\n[Phase 2/4] Sampling {n_synthetic:,} Synthetic Mutants via Graph Diffusion k-DPP...")
    kdpp = GraphDiffusionKDPP(tau=0.5, seed=seed)

    # Level quotas: user hypothesis testing [0.25, 0.25, 0.25, 0.25]
    r1, r2, r3, r4 = ratios
    n_l1 = int(math.floor(r1 * n_synthetic))
    n_l2 = int(math.floor(r2 * n_synthetic))
    n_l3 = int(math.floor(r3 * n_synthetic))
    n_l4 = n_synthetic - n_l1 - n_l2 - n_l3
    print(f" -> Level Quotas: L1={n_l1:,} ({r1*100:.1f}%), L2={n_l2:,} ({r2*100:.1f}%), L3={n_l3:,} ({r3*100:.1f}%), L4={n_l4:,} ({r4*100:.1f}%)", flush=True)

    level_assignments = [1]*n_l1 + [2]*n_l2 + [3]*n_l3 + [4]*n_l4
    rng.shuffle(level_assignments)

    synth_success = 0
    attempts = 0
    duplicates_blocked = 0
    duplicates_rescued = 0

    t_sample_start = time.time()
    while synth_success < n_synthetic:
        attempts += 1
        level = level_assignments[synth_success]
        ovr, desc_str = sample_kdpp_mutant(level, kdpp, rng=rng)

        # Collision check against D_real
        if ma.is_duplicate_real_mutant(ovr, defaults):
            rescued_ovr, rescued_desc = ma.rescue_collision_candidate(ovr, desc_str, defaults)
            if rescued_ovr is not None and not ma.is_duplicate_real_mutant(rescued_ovr, defaults):
                ovr = rescued_ovr
                desc_str = rescued_desc
                duplicates_rescued += 1
            else:
                duplicates_blocked += 1
                continue

        synth_success += 1
        sample_name = f"KDPP_{desc_str}_L{level}_{synth_success}".replace(" ", "_")
        ordered_names.append(sample_name)
        all_tasks[sample_name] = ovr

        if synth_success % 20000 == 0 or synth_success == n_synthetic:
            print(f"  * Generated {synth_success:,} / {n_synthetic:,} tasks ({synth_success/n_synthetic*100:.1f}%) in {time.time()-t_sample_start:.1f}s", flush=True)

    print(f" -> Sampling Completed: {synth_success:,} synthetic tasks generated in {attempts:,} attempts.")
    print(f"    - Filter [D_real Deduplication]: {duplicates_blocked} blocked, {duplicates_rescued} rescued.")
    print(f" -> Total Simulation Queue: {len(ordered_names):,} tasks ({len(ma.REAL_MUTANTS)} Real + {synth_success:,} Synthetic).", flush=True)

    # Phase 3: Parallel ODE Simulation Engine
    print(f"\n[Phase 3/4] Simulating {len(ordered_names):,} Mutant Trajectories on {num_workers} Workers...")
    all_results = {}
    bio_labels_dict = {}
    pattern_labels_dict = {}

    sim_counts = {
        'real_success': 0, 'synth_success': 0,
        'discarded_real': 0, 'discarded_synth': 0,
        'failed_real': 0, 'failed_synth': 0
    }

    t_sim_start = time.time()
    log_step = max(50, len(ordered_names) // 40)

    task_list = [(name, all_tasks[name]) for name in ordered_names]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, (m_name, y_uniform, m_bio_label, m_pattern_label) in enumerate(
            executor.map(run_simulation_task, task_list, chunksize=16)
        ):
            is_synth = "_L" in m_name
            if y_uniform is not None:
                all_results[m_name] = y_uniform
                bio_labels_dict[m_name] = m_bio_label
                pattern_labels_dict[m_name] = m_pattern_label
                if is_synth:
                    sim_counts['synth_success'] += 1
                else:
                    sim_counts['real_success'] += 1
            elif m_bio_label == "Discarded_Mass":
                if is_synth:
                    sim_counts['discarded_synth'] += 1
                else:
                    sim_counts['discarded_real'] += 1
            else:
                if is_synth:
                    sim_counts['failed_synth'] += 1
                else:
                    sim_counts['failed_real'] += 1

            if (idx + 1) % log_step == 0 or (idx + 1) == len(ordered_names):
                elapsed = max(0.001, time.time() - t_sim_start)
                speed = (idx + 1) / elapsed
                valid_total = sim_counts['real_success'] + sim_counts['synth_success']
                disc_total = sim_counts['discarded_real'] + sim_counts['discarded_synth']
                fail_total = sim_counts['failed_real'] + sim_counts['failed_synth']
                pct = ((idx + 1) / len(ordered_names)) * 100
                print(f"  |-- [{idx + 1:6d}/{len(ordered_names):6d}] ({pct:5.1f}%) | Valid: {valid_total:6d} (Real:{sim_counts['real_success']:3d}, Synth:{sim_counts['synth_success']:6d}) | Discarded(Mass>10): {disc_total:4d} | Failed: {fail_total:3d} | Speed: {speed:5.1f} samples/s", flush=True)

    elapsed_sim = time.time() - t_sim_start
    print(f" -> Simulation Completed in {elapsed_sim:.1f}s ({len(ordered_names) / max(0.001, elapsed_sim):.1f} samples/s).", flush=True)

    # Phase 4: Format Packaging & .npz Export
    print("\n[Phase 4/4] Packaging & Saving Dataset Array Files (.npz)...", flush=True)
    t_pack_start = time.time()

    ordered_tensors = []
    final_names = []
    final_params = []
    final_bio_labels = []
    final_pattern_labels = []

    for name in ordered_names:
        if name in all_results:
            ordered_tensors.append(all_results[name])
            final_names.append(name)
            final_bio_labels.append(bio_labels_dict[name])
            final_pattern_labels.append(pattern_labels_dict[name])

            overrides = all_tasks[name]
            p_vec = [overrides.get(k, defaults[k]) for k in param_keys]
            final_params.append(p_vec)

    all_tensors = np.array(ordered_tensors, dtype=np.float64)
    all_params = np.array(final_params, dtype=np.float64)
    all_bio_labels = np.array(final_bio_labels)
    all_pattern_labels = np.array(final_pattern_labels)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f" -> Compressing and writing {output_path} via np.savez_compressed...", flush=True)
    np.savez_compressed(
        output_path,
        data=all_tensors,
        params=all_params,
        param_names=np.array(param_keys),
        mutant_names=np.array(final_names),
        labels=all_bio_labels,
        pattern_labels=all_pattern_labels
    )

    pack_time = time.time() - t_pack_start
    total_valid = len(final_names)
    file_size_gb = os.path.getsize(output_path) / (1024 ** 3)

    print("=" * 85)
    print(" K-DPP DATASET GENERATION SUMMARY REPORT")
    print("=" * 85)
    print(f"Dataset Export Path               : {output_path}")
    print(f"Export File Size                  : {file_size_gb:.2f} GB")
    print(f"Data Tensor Shape (N, Vars, Steps): {all_tensors.shape}")
    print(f"Params Matrix Shape (N, P_dim)    : {all_params.shape}")
    print(f"Total Validated Samples Saved     : {total_valid:,} / {len(ordered_names):,} ({total_valid / len(ordered_names) * 100:.2f}% yield)")
    print(f"Total Packaging Time              : {pack_time:.1f}s")
    print(f"Total Pipeline Runtime            : {time.time() - t_sample_start:.1f}s")
    print("=" * 85 + "\n", flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate 100k Dataset via Graph Diffusion k-DPP Engine")
    parser.add_argument('--output', type=str, default="/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast/kdpp_gated_uniform25_100000_210m_500s.npz")
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratios', type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25],
                        help="Level quotas L1, L2, L3, L4 (default: 0.25 0.25 0.25 0.25)")
    args = parser.parse_args()

    generate_kdpp_dataset(output_path=args.output, n_synthetic=args.n_samples, num_workers=args.workers,
                          seed=args.seed, ratios=tuple(args.ratios))


if __name__ == '__main__':
    main()
