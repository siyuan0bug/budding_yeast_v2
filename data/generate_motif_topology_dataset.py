#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_motif_topology_dataset.py
================================================================================
Motif-Topology High-Dimensional Manifold Dataset Generator for Budding Yeast Cell Cycle.
Optimized Option A: Combines biophysical ODEs from mutant_aware.py with
Graph-Theoretic Connected Subgraphs, Parallel Checkpoint Pairs, and Hard-Zero KO.

Key Features:
  1. Zero-Leakage Topology-Driven:
     - 47 Direct Epistatic Pairs
     - 177 Connected Triplet Motifs (cFFL, feedback loops)
     - 651 Connected Quadruplet Motifs
     - Parallel Checkpoint Pairs (MAD2+BUB2, NET1 cross-talks, CLN3/BCK2+SIC1)
     - Master Switch 5-Order Deep Exploration (Start & Exit Clusters)
     - Zero hardcoded mutant strains or test labels.
  2. Strict Alignment with mutant_aware.py:
     - Reuses canonical Chen (2004) ODE solver, Goldbeter-Koshland functions, events.
     - 141 parameter keys, default WT values, and dynamic scaling distributions.
     - Hard Zeros (0.0) strictly enforced for KO mutations (validated by Ablation Arm A).
     - Full 141-D deduplication against all 126 real mutants via is_duplicate_real_mutant.
  3. Standard Output Packaging:
     - data           : (N, 39, 500) float32 array
     - params         : (N, 141) float64 array
     - param_names    : (141,) string array
     - mutant_names   : (N,) string array (126 real mutants first, then synthetic)
     - labels         : (N,) biological viability labels
     - pattern_labels : (N,) mass growth pattern labels
================================================================================
"""

import os
import sys
import time
import argparse
import itertools
import concurrent.futures
import numpy as np
import networkx as nx

# Setup path to import mutant_aware
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import mutant_aware as ma

# ==============================================================================
# Part 1: Topological Network & Motif Mining
# ==============================================================================

# Chen (2004) 54-edge directed regulatory network
MODULE_EDGES = [
    ('CLN3', 'SBF_MBF', '+'), ('BCK2', 'SBF_MBF', '+'), ('CLN2', 'SBF_MBF', '+'),
    ('CLB5', 'SBF_MBF', '+'), ('CLB2', 'SBF_MBF', '-'),
    ('SBF_MBF', 'CLN2', '+'), ('SBF_MBF', 'CLB5', '+'), ('SBF_MBF', 'CDC6', '+'),
    ('CLB2', 'Mcm1', '+'), ('Mcm1', 'CLB2', '+'), ('Mcm1', 'SWI5', '+'),
    ('Mcm1', 'CDC20', '+'), ('Mcm1', 'PDS1', '+'),
    ('CLB2', 'CDC20', '+'), ('CDC20', 'CLB2', '-'), ('CDC20', 'CLB5', '-'),
    ('CDC20', 'PDS1', '-'),
    ('CDH1', 'CLB2', '-'), ('CDH1', 'PDS1', '-'),
    ('CDC14', 'CDH1', '+'), ('CDC14', 'SIC1', '+'), ('CDC14', 'CDC6', '+'),
    ('CLB2', 'CDH1', '-'), ('CLB5', 'CDH1', '-'), ('CLN2', 'CDH1', '-'),
    ('CLN3', 'CDH1', '-'),
    ('SIC1', 'CLB2', '-'), ('SIC1', 'CLB5', '-'),
    ('CDC6', 'CLB2', '-'), ('CDC6', 'CLB5', '-'),
    ('SWI5', 'SIC1', '+'), ('SWI5', 'CDC6', '+'),
    ('NET1', 'CDC14', '-'), ('PPX', 'NET1', '+'), ('CDC15', 'NET1', '+'),
    ('TEM1', 'CDC15', '+'), ('BUB2', 'TEM1', '-'), ('MAD2', 'CDC20', '-'),
    ('PDS1', 'ESP1', '-'),
    ('CDC28', 'SIC1', '+'), ('CDC28', 'CDC6', '+'), ('CDC28', 'CDH1', '+'),
    ('CDC28', 'SBF_MBF', '+'), ('CDC28', 'ORI_BUD_SPN', '+'),
    ('CLB2', 'ORI_BUD_SPN', '+'), ('CLB5', 'ORI_BUD_SPN', '+'),
    ('APC_Core', 'CLB2', '-'), ('APC_Core', 'PDS1', '-'),
    ('CDC20', 'APC_Core', '+'), ('CDH1', 'APC_Core', '+'),
    ('CLB2', 'SWI5', '-'), ('CLB2', 'SIC1', '-'), ('CLB5', 'SIC1', '-'), ('CLN2', 'SIC1', '-'),
]

# Distal / Parallel Functional Pairs (Known biophysical dual-gates)
PARALLEL_DISTAL_PAIRS = [
    # 1. Spindle assembly checkpoint dual-branch
    ('MAD2', 'BUB2'),
    # 2. Mitotic exit & G1/S cross-phase couplings
    ('NET1', 'CLB2'),
    ('NET1', 'CLB5'),
    ('NET1', 'CDC20'),
    ('NET1', 'CDC14'),
    # 3. G1/S Starter-Brake direct couplings
    ('CLN3', 'SIC1'),
    ('BCK2', 'SIC1'),
    ('CLN3', 'CDH1'),
    ('BCK2', 'CDH1')
]

# Master Switch Clusters for Order-5 Exploration
MASTER_SWITCH_CLUSTERS = [
    # G1/S Start Switch cluster (Cln3/Bck2/Cln2/Sic1/Cdh1/SBF)
    ['CLN2', 'CLN3', 'BCK2', 'SIC1', 'CDH1', 'SBF_MBF'],
    # Anaphase/Exit Switch cluster (Clb2/Cdc20/Pds1/Cdc6/Sic1/APC_Core)
    ['CLB2', 'CDC20', 'PDS1', 'CDC6', 'SIC1', 'APC_Core']
]


class MotifTopologySampler:
    """Extracts and manages topological connected motifs for zero-leakage generation."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.G = nx.Graph()
        for u, v, s in MODULE_EDGES:
            self.G.add_edge(u, v)

        # Filter nodes present in ma.GENE_NAMES
        self.active_nodes = [n for n in self.G.nodes() if n in ma.GENE_NAMES]

        # 1. Direct Edges (Level 2 pairs)
        self.l2_motifs = []
        for u, v in itertools.combinations(self.active_nodes, 2):
            if self.G.has_edge(u, v):
                self.l2_motifs.append((u, v))
        # Add parallel distal pairs
        for u, v in PARALLEL_DISTAL_PAIRS:
            if u in ma.GENE_NAMES and v in ma.GENE_NAMES:
                self.l2_motifs.append((u, v))
        self.l2_motifs = list(set(self.l2_motifs))

        # 2. Connected Triplets (Level 3)
        self.l3_motifs = []
        for triplet in itertools.combinations(self.active_nodes, 3):
            subG = self.G.subgraph(triplet)
            if nx.is_connected(subG):
                self.l3_motifs.append(tuple(sorted(triplet)))
        self.l3_motifs = list(set(self.l3_motifs))

        # 3. Connected Quadruplets & Quints (Level 4 & 5)
        self.l4_motifs = []
        for quad in itertools.combinations(self.active_nodes, 4):
            subG = self.G.subgraph(quad)
            if nx.is_connected(subG):
                self.l4_motifs.append(tuple(sorted(quad)))
        self.l4_motifs = list(set(self.l4_motifs))

        # Master switch 5-order subsets
        self.l5_motifs = []
        for cluster in MASTER_SWITCH_CLUSTERS:
            for quint in itertools.combinations(cluster, 5):
                self.l5_motifs.append(tuple(sorted(quint)))
        self.l5_motifs = list(set(self.l5_motifs))

        # L1 Grid
        self.l1_grid = [(g, m) for g in ma.GENE_NAMES for m in ma.GROUP_BY_NAME[g]['modes']]
        self.rng.shuffle(self.l1_grid)
        self.l1_idx = 0

        print("=" * 80)
        print(" [MotifTopologySampler] Initialized Topological Motifs:")
        print(f"  * Level 1 Cells (Gene x Mode)         : {len(self.l1_grid)}")
        print(f"  * Level 2 Pairs (Direct + Distal)     : {len(self.l2_motifs)}")
        print(f"  * Level 3 Triplets (Connected SubG)   : {len(self.l3_motifs)}")
        print(f"  * Level 4 Quadruplets (Connected SubG): {len(self.l4_motifs)}")
        print(f"  * Level 5 Master Switch Quints        : {len(self.l5_motifs)}")
        print(f"  * Total Discrete Motifs Available     : {len(self.l2_motifs) + len(self.l3_motifs) + len(self.l4_motifs) + len(self.l5_motifs)}")
        print("=" * 80)

    def sample_motif(self, level):
        """Samples a gene combination based on hierarchical level."""
        if level == 1:
            cell = self.l1_grid[self.l1_idx % len(self.l1_grid)]
            self.l1_idx += 1
            return [cell[0]], [cell[1]]

        elif level == 2:
            idx = self.rng.randint(len(self.l2_motifs))
            genes = list(self.l2_motifs[idx])
        elif level == 3:
            idx = self.rng.randint(len(self.l3_motifs))
            genes = list(self.l3_motifs[idx])
        elif level == 4:
            idx = self.rng.randint(len(self.l4_motifs))
            genes = list(self.l4_motifs[idx])
        else:  # Level 5
            idx = self.rng.randint(len(self.l5_motifs))
            genes = list(self.l5_motifs[idx])

        # Assign modes for each gene in the selected motif
        modes = []
        for g in genes:
            avail = list(ma.GROUP_BY_NAME[g]['modes'].keys())
            # Balanced mode preference: 45% KO, 25% ts/db/pLOF, 30% GOF (MC/GAL-OE)
            ko_avail = [m for m in avail if m == 'KO']
            dec_avail = [m for m in avail if m in ('ts', 'db', 'pLOF', 'DAmP', 'CSL', 'CKO')]
            gof_avail = [m for m in avail if m in ('MC', 'GAL-OE', 'CA')]

            r = self.rng.rand()
            if r < 0.45 and ko_avail:
                mode = 'KO'
            elif r < 0.70 and dec_avail:
                mode = str(self.rng.choice(dec_avail))
            elif gof_avail:
                mode = str(self.rng.choice(gof_avail))
            else:
                mode = str(self.rng.choice(avail))
            modes.append(mode)

        return genes, modes


# ==============================================================================
# Part 2: Biophysical Perturbation Application (Following mutant_aware.py)
# ==============================================================================

def build_perturbed_sample(genes, modes, rng):
    """
    Constructs a complete parameter override dict for the chosen (genes, modes):
    - Strict Hard Zero (0.0) for KO mutations.
    - Continuous dynamic range scaling for ts, db, MC, GAL-OE.
    - Proper carbon source (mdt) decoupling.
    """
    p_wt = ma.get_default_params()
    p_wt.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    ovr = {}

    has_gal = False
    for g, m in zip(genes, modes):
        if m == 'GAL-OE':
            has_gal = True

        # Case 1: Pure Knockout -> Strict Hard Zero (Ablation Arm A validated)
        if m == 'KO':
            ma.apply_gene_mutation(g, 'KO', 0.0, p_wt, {}, ovr)
        # Case 2: Continuous Hypomorphic, GOF, or Special alleles
        else:
            alpha = ma.sample_scaling_factor(m, g)
            ma.apply_gene_mutation(g, m, alpha, p_wt, {}, ovr)

    # Carbon source handling
    if has_gal:
        ovr['mdt'] = 150.0
    else:
        # Glc: 90.0 (75%), Gal: 150.0 (18%), Raff: 160.0 (7%)
        ovr['mdt'] = float(rng.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))

    return ovr


# ==============================================================================
# Part 3: Dataset Generation Engine
# ==============================================================================

def generate_motif_dataset(num_samples=100000,
                           num_workers=60,
                           level_ratios=(0.10, 0.30, 0.30, 0.30),
                           output_name=None,
                           output_dir="/home/users/zsy/mydata/budding_yeast",
                           seed=42):
    """Generates 100,000 synthetic samples + 126 Real Mutants using Motif-Topology Sampling."""
    rng = np.random.RandomState(seed)
    sampler = MotifTopologySampler(seed=seed)

    if output_name is None:
        output_name = f"bams_motif_topology_{num_samples}_210m_500s.npz"
    dataset_path = os.path.join(output_dir, output_name)

    print("\n" + "=" * 85)
    print(f" MOTIF-TOPOLOGY DATASET GENERATOR (Target: {num_samples} samples, Seed: {seed})")
    print(f" Target File: {dataset_path}")
    print("=" * 85, flush=True)

    # Calculate level quotas
    n_l1 = int(round(num_samples * level_ratios[0]))
    n_l2 = int(round(num_samples * level_ratios[1]))
    n_l3 = int(round(num_samples * level_ratios[2]))
    n_l45 = num_samples - (n_l1 + n_l2 + n_l3)
    n_l4 = int(round(n_l45 * 0.70))
    n_l5 = n_l45 - n_l4

    quota_plan = [
        (1, n_l1, "Level 1: Single Mutants Grid Rotation"),
        (2, n_l2, "Level 2: Pairwise Connected & Distal Checkpoints"),
        (3, n_l3, "Level 3: Triplet Connected Motifs (cFFL & Loops)"),
        (4, n_l4, "Level 4: Quadruplet Connected Motifs"),
        (5, n_l5, "Level 5: Master Switch 5-Order Deep Exploration")
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

    # 2. Synthetic Mutants according to Quota Plan
    duplicates_blocked = 0
    t0_prep = time.time()

    p_wt_default = ma.get_default_params()
    p_wt_default.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

    for level, count, desc in quota_plan:
        print(f"  * Constructing {desc} ({count} samples)...", flush=True)
        accepted = 0
        while accepted < count:
            genes, modes = sampler.sample_motif(level)
            ovr = build_perturbed_sample(genes, modes, rng)

            # Strict 141-D Deduplication against all 126 real mutants
            if ma.is_duplicate_real_mutant(ovr, p_wt_default):
                duplicates_blocked += 1
                continue

            # Descriptive task name
            g_str = "+".join(f"{g}_{m}" for g, m in zip(genes, modes))
            t_name = f"Synth_L{level}_{accepted:05d}_{g_str}"

            all_tasks[t_name] = ovr
            ordered_names.append(t_name)
            accepted += 1

    print(f" -> Task construction finished in {time.time() - t0_prep:.2f}s.")
    print(f"    Total candidates created: {len(ordered_names)} | Exact Real Duplicates Filtered: {duplicates_blocked}")

    # 3. Parallel Simulation via ProcessPoolExecutor
    print(f"\n[Phase 2/4] Executing Parallel ODE Simulations ({num_workers} Workers)...", flush=True)
    t_sim_start = time.time()

    all_results = {}
    bio_labels_dict = {}
    pattern_labels_dict = {}

    sim_counts = {
        'real_success': 0, 'synth_success': 0,
        'discarded_real': 0, 'discarded_synth': 0,
        'failed_real': 0, 'failed_synth': 0
    }
    log_step = max(100, len(ordered_names) // 50)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_name = {
            executor.submit(ma.simulate_mutant, name, all_tasks[name], 210, 500): name
            for name in ordered_names
        }
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_name)):
            name = future_to_name[future]
            is_synth = name.startswith("Synth_")
            try:
                m_name, y_uniform, m_bio_label, m_pattern_label = future.result(timeout=30.0)
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
                elapsed = max(0.001, time.time() - t_sim_start)
                speed = (idx + 1) / elapsed
                valid_total = sim_counts['real_success'] + sim_counts['synth_success']
                pct = ((idx + 1) / len(ordered_names)) * 100
                print(f"  |-- [{idx + 1:6d}/{len(ordered_names)}] ({pct:5.1f}%) | Valid: {valid_total:6d} | Speed: {speed:5.1f} samples/s", flush=True)

    print(f" -> ODE Simulations Completed in {time.time() - t_sim_start:.2f}s.", flush=True)

    # 4. Packaging and NPZ Export
    print("\n[Phase 3/4] Packaging Standardized Array Structures...", flush=True)
    defaults = ma.get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    param_keys = ma.PARAM_KEYS

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

    all_tensors = np.array(ordered_tensors, dtype=np.float32)
    all_params = np.array(final_params, dtype=np.float64)
    all_bio_labels = np.array(final_bio_labels)
    all_pattern_labels = np.array(final_pattern_labels)

    os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)
    print(f"[Phase 4/4] Writing compressed archive: {dataset_path} ...", flush=True)
    np.savez_compressed(dataset_path,
                        data=all_tensors,
                        params=all_params,
                        param_names=param_keys,
                        mutant_names=final_names,
                        labels=all_bio_labels,
                        pattern_labels=all_pattern_labels)

    print("=" * 85)
    print(" MOTIF-TOPOLOGY GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 85)
    print(f"Dataset Output File        : {dataset_path}")
    print(f"Data Tensor Shape (N, V, T): {all_tensors.shape}")
    print(f"Params Shape (N, P_dim)    : {all_params.shape}")
    print(f"Total Validated Yield      : {len(final_names)} / {len(ordered_names)} ({len(final_names)/len(ordered_names)*100:.2f}%)")
    print(f"Real Mutants Included      : {sim_counts['real_success']} / {len(ma.REAL_MUTANTS)}")
    print(f"Synthetic Mutants Generated: {sim_counts['synth_success']}")
    print("=" * 85 + "\n", flush=True)

    return dataset_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Motif-Topology Manifold Dataset Generator")
    parser.add_argument("--samples", type=int, default=100000, help="Number of synthetic samples to generate")
    parser.add_argument("--workers", type=int, default=60, help="Number of parallel CPU worker processes")
    parser.add_argument("--output_name", type=str, default=None, help="Export .npz filename")
    parser.add_argument("--output_dir", type=str, default="/home/users/zsy/mydata/budding_yeast", help="Output directory path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_motif_dataset(num_samples=args.samples,
                           num_workers=args.workers,
                           output_name=args.output_name,
                           output_dir=args.output_dir,
                           seed=args.seed)
