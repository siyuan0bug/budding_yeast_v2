#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASS v6.5: Signed Directed Graph Laplacian k-DPP + Phase-Space Curvature Adaptive Gating
========================================================================================
Key Innovations (Zero Data Leakage, 100% Derived from 54-Edge ODE Topology & Dynamics):
  [1] Signed Directed Graph Laplacian Sensitivity Kernel (方案 A):
      - Uses 54 directed signed edges from Chen (2004) ODE system.
      - Computes Dynamical Sensitivity Index (DSI) for k-DPP quality vector:
        Accounts for 2-cycle mutual toggle switches (e.g. Clb2-Sic1, Clb2-Cdh1, Clb5-Sic1)
        and convergent fan-in hubs (e.g. SBF/MBF, Sic1, Cdh1, Cdc20, APC_Core).
      - Topologically guided mode coordination: Pairs spanning antagonistic toggle switches
        or converging inputs are coordinated to explore true epistatic bifurcations.

  [2] Phase-Space Dynamic Curvature Adaptive Gating (方案 B):
      - Evaluates trajectory second-order phase acceleration norm K(y).
      - Suppresses degenerate flatline arrests (K < 0.01, downsampled to p=0.20),
        preventing dataset capacity from being wasted on trivial zero-flatlines.
      - Prioritizes trajectories exhibiting active non-linear phase transitions
        (mitotic triggers, delayed arrests, limit cycle oscillations).

Output:
  - 100,000 synthetic samples + 126 real mutants in standard .npz format.
"""

import os
import sys
import time
import math
import argparse
import concurrent.futures
import numpy as np
from scipy.linalg import expm

# Invariant: Import baseline mutant_aware without any modifications
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutant_aware as ma


# ==============================================================================
# 1. Signed Directed Graph Laplacian & Topological Motif k-DPP Sampler
# ==============================================================================

class SignedDirectedGraphKDPP:
    """
    k-DPP Sampler incorporating directed regulatory feedback loops and convergent fan-in.
    Purely derived from the 54-edge ODE topology without any test-set awareness.
    """
    def __init__(self, tau=0.5, max_k=6, seed=42):
        self.rng = np.random.RandomState(seed)
        self.gene_names = list(ma.GENE_NAMES)
        self.N = len(self.gene_names)
        self.idx_map = {g: i for i, g in enumerate(self.gene_names)}
        self.max_k = max_k
        self.tau = tau

        # Directed and signed adjacency matrices
        self.A_dir = np.zeros((self.N, self.N), dtype=float)
        self.A_sign = np.zeros((self.N, self.N), dtype=float)
        self.edge_types = {}  # (u, v) -> sign

        for u, v, s in ma.MODULE_EDGES:
            if u in self.idx_map and v in self.idx_map:
                i, j = self.idx_map[u], self.idx_map[v]
                self.A_dir[i, j] = 1.0
                sign_val = 1.0 if s == '+' else -1.0
                self.A_sign[i, j] = sign_val
                self.edge_types[(u, v)] = s

        # Symmetrized connectivity for diffusion kernel
        A_sym = np.maximum(self.A_dir, self.A_dir.T)
        deg_sym = np.sum(A_sym, axis=1)
        d_inv_sqrt = np.array([1.0 / np.sqrt(d + 0.5) for d in deg_sym])
        L_norm = np.eye(self.N) - d_inv_sqrt[:, None] * A_sym * d_inv_sqrt[None, :]

        # Symmetrized Heat Diffusion Similarity Kernel
        K_diff = expm(-self.tau * L_norm)
        d_s = np.sqrt(np.maximum(np.diag(K_diff), 1e-12))
        self.S = K_diff / (d_s[:, None] * d_s[None, :])

        # Dynamical Sensitivity Index (DSI) from directed topology
        in_deg = np.sum(self.A_dir, axis=0)
        out_deg = np.sum(self.A_dir, axis=1)
        two_cycles = np.diag(self.A_dir @ self.A_dir)
        three_cycles = np.diag(self.A_dir @ self.A_dir @ self.A_dir)

        # Quality weights: High sensitivity for toggle switches & convergent hubs
        alpha_loop = 0.50
        alpha_fanin = 0.40
        gamma = 0.30
        self.q = (1.0 + alpha_loop * (two_cycles + 0.5 * three_cycles) +
                  alpha_fanin * in_deg + 0.20 * out_deg) ** gamma

        # L-ensemble kernel L = q * S * q^T
        self.L = self.q[:, None] * self.S * self.q[None, :]

        # Eigendecomposition
        self.eigvals, self.eigvecs = np.linalg.eigh(self.L)
        self.eigvals = np.maximum(self.eigvals, 1e-10)

        # Precompute Elementary Symmetric Polynomials
        self.E = np.zeros((self.max_k + 1, self.N + 1), dtype=float)
        self.E[0, :] = 1.0
        for l in range(1, self.max_k + 1):
            for n in range(1, self.N + 1):
                self.E[l, n] = self.E[l, n - 1] + self.eigvals[n - 1] * self.E[l - 1, n - 1]

        # Pre-index fan-in parent pairs for mode coordination
        self.fanin_parents = {}
        for w in range(self.N):
            parents = np.where(self.A_dir[:, w] > 0)[0]
            if len(parents) >= 2:
                for a in range(len(parents)):
                    for b in range(a + 1, len(parents)):
                        p_pair = tuple(sorted([self.gene_names[parents[a]], self.gene_names[parents[b]]]))
                        self.fanin_parents.setdefault(p_pair, []).append(self.gene_names[w])

    def sample_genes(self, k):
        """Exact k-DPP projection sampling."""
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

    def coordinate_modes(self, genes):
        """
        Topologically guided mode assignment based on signed edges and fan-in.
        Zero test-set hardcoding: derives 100% from network topology.
        """
        pairs = []
        assigned = {}

        # 1. Check for mutual antagonistic edges (toggle switches like CLB2-SIC1, CLB2-CDH1)
        for i, g1 in enumerate(genes):
            for g2 in genes[i + 1:]:
                # If negative interaction exists between g1 and g2
                has_neg = (self.edge_types.get((g1, g2)) == '-' or self.edge_types.get((g2, g1)) == '-')
                if has_neg and self.rng.rand() < 0.50:
                    # Coordinate opposing modes to excite the switch
                    m1_modes = list(ma.GROUP_BY_NAME[g1]['modes'].keys())
                    m2_modes = list(ma.GROUP_BY_NAME[g2]['modes'].keys())
                    lofs_1 = [m for m in ['KO', 'ts', 'DAmP', 'pLOF', 'db'] if m in m1_modes]
                    oes_2 = [m for m in ['GAL-OE', 'MC'] if m in m2_modes]
                    if lofs_1 and oes_2:
                        assigned[g1] = str(self.rng.choice(lofs_1))
                        assigned[g2] = str(self.rng.choice(oes_2))

        # 2. Check for convergent fan-in pairs (e.g. g1, g2 both regulate target w)
        for i, g1 in enumerate(genes):
            for g2 in genes[i + 1:]:
                pair_key = tuple(sorted([g1, g2]))
                if pair_key in self.fanin_parents and self.rng.rand() < 0.50:
                    # Coordinate synthetic epistatic defect (co-loss or co-gain)
                    if g1 not in assigned:
                        m1_modes = list(ma.GROUP_BY_NAME[g1]['modes'].keys())
                        lofs_1 = [m for m in ['KO', 'ts', 'DAmP'] if m in m1_modes]
                        if lofs_1:
                            assigned[g1] = 'KO' if 'KO' in lofs_1 and self.rng.rand() < 0.70 else str(self.rng.choice(lofs_1))
                    if g2 not in assigned:
                        m2_modes = list(ma.GROUP_BY_NAME[g2]['modes'].keys())
                        lofs_2 = [m for m in ['KO', 'ts', 'DAmP'] if m in m2_modes]
                        if lofs_2:
                            assigned[g2] = 'KO' if 'KO' in lofs_2 and self.rng.rand() < 0.70 else str(self.rng.choice(lofs_2))

        # 3. Fill remaining unassigned genes with standard mode sampling
        for g in genes:
            if g in assigned:
                pairs.append((g, assigned[g]))
            else:
                pairs.append((g, ma._sample_mode(g)))

        return pairs


# ==============================================================================
# 2. Sample Generation Function
# ==============================================================================

def sample_signed_kdpp_mutant(level, kdpp_sampler, rng=np.random):
    """
    Samples a candidate mutant using Signed Graph Diffusion k-DPP.
    Fully adheres to biological physical scaling laws and collision filters.
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
        genes = kdpp_sampler.sample_genes(level)
        pairs = kdpp_sampler.coordinate_modes(genes)

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
        return sample_signed_kdpp_mutant(level, kdpp_sampler, rng)

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
# 3. Trajectory Dynamic Curvature Calculator & Simulation Worker
# ==============================================================================

def compute_trajectory_curvature(y):
    """
    Computes trajectory dynamic phase acceleration norm K(y).
    y: (39, 500) array.
    K(y) = (1/T) sum_t ||y(t+1) - 2y(t) + y(t-1)||_2
    """
    if y is None or y.shape[1] < 3:
        return 0.0
    a = y[:, 2:] - 2.0 * y[:, 1:-1] + y[:, :-2]
    return float(np.mean(np.sqrt(np.sum(a ** 2, axis=0))))


def run_simulation_task(task_tuple):
    """Worker task executing ODE simulation and returning results + curvature."""
    name, ovr = task_tuple
    m_name, y_uniform, m_bio, m_pat = ma.simulate_mutant(name, ovr, tf=210, ns=500)
    curv = compute_trajectory_curvature(y_uniform) if y_uniform is not None else 0.0
    return m_name, y_uniform, m_bio, m_pat, curv


# ==============================================================================
# 4. Main Dataset Generation Pipeline
# ==============================================================================

def generate_kdpp_curv_gated_dataset(output_path, n_synthetic=100000, num_workers=60,
                                     seed=42, ratios=(0.20, 0.40, 0.25, 0.15),
                                     curv_thresh=0.01, flat_prob=0.20):
    print("=" * 85)
    print(" MASS v6.5: Signed Graph Diffusion k-DPP + Phase-Space Curvature Gating")
    print("=" * 85)
    print(f"Target Output Path       : {output_path}")
    print(f"Target Synthetic Samples : {n_synthetic:,}")
    print(f"Random Seed              : {seed}")
    print(f"Parallel Workers         : {num_workers}")
    print(f"Level Quotas (L1~L4)     : {ratios}")
    print(f"Curvature Gating         : thresh={curv_thresh}, flatline_keep_prob={flat_prob}")
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

    # Phase 2: Synthetic Mutants (Signed k-DPP Engine)
    print(f"\n[Phase 2/4] Sampling Synthetic Tasks via Signed Graph Diffusion k-DPP...")
    kdpp = SignedDirectedGraphKDPP(tau=0.5, seed=seed)

    r1, r2, r3, r4 = ratios
    n_l1 = int(math.floor(r1 * n_synthetic))
    n_l2 = int(math.floor(r2 * n_synthetic))
    n_l3 = int(math.floor(r3 * n_synthetic))
    n_l4 = n_synthetic - n_l1 - n_l2 - n_l3

    # Generate an over-sampled candidate queue (1.35x) to allow curvature gating
    queue_multiplier = 1.35
    total_queue = int(math.ceil(n_synthetic * queue_multiplier))
    q_l1 = int(math.floor(r1 * total_queue))
    q_l2 = int(math.floor(r2 * total_queue))
    q_l3 = int(math.floor(r3 * total_queue))
    q_l4 = total_queue - q_l1 - q_l2 - q_l3

    level_assignments = [1] * q_l1 + [2] * q_l2 + [3] * q_l3 + [4] * q_l4
    rng.shuffle(level_assignments)

    synth_candidates = []
    attempts = 0
    duplicates_blocked = 0
    duplicates_rescued = 0

    t_sample_start = time.time()
    for level in level_assignments:
        attempts += 1
        ovr, desc_str = sample_signed_kdpp_mutant(level, kdpp, rng=rng)

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

        sample_name = f"MASS_{desc_str}_L{level}_{len(synth_candidates) + 1}".replace(" ", "_")
        synth_candidates.append((sample_name, ovr))
        all_tasks[sample_name] = ovr

    print(f" -> Generated {len(synth_candidates):,} candidates in {attempts:,} draws "
          f"({duplicates_blocked} blocked, {duplicates_rescued} rescued).", flush=True)

    # Phase 3: Parallel ODE Simulation & Curvature-Adaptive Gating
    print(f"\n[Phase 3/4] Simulating Trajectories with Curvature & Phenotype Gating on {num_workers} Workers...")
    all_results = {}
    bio_labels_dict = {}
    pattern_labels_dict = {}
    curvatures_dict = {}

    sim_counts = {
        'real_success': 0, 'synth_success': 0,
        'flatline_filtered': 0, 'discarded': 0, 'failed': 0
    }

    t_sim_start = time.time()
    
    # Run Real Mutants first
    real_task_list = [(name, all_tasks[name]) for name in ordered_names if not name.startswith("MASS_")]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_workers, 16)) as executor:
        for m_name, y_uniform, m_bio, m_pat, curv in executor.map(run_simulation_task, real_task_list):
            if y_uniform is not None:
                all_results[m_name] = y_uniform
                bio_labels_dict[m_name] = m_bio
                pattern_labels_dict[m_name] = m_pat
                curvatures_dict[m_name] = curv
                sim_counts['real_success'] += 1

    print(f" -> Real Mutants Simulated: {sim_counts['real_success']} / {len(ma.REAL_MUTANTS)} OK.", flush=True)

    # Run Synthetic Queue with Dynamic Curvature Gating
    final_synthetic_names = []
    level_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    target_levels = {1: n_l1, 2: n_l2, 3: n_l3, 4: n_l4}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, (m_name, y_uniform, m_bio, m_pat, curv) in enumerate(
            executor.map(run_simulation_task, synth_candidates, chunksize=16)
        ):
            if len(final_synthetic_names) >= n_synthetic:
                break

            if y_uniform is None:
                if m_bio == "Discarded_Mass":
                    sim_counts['discarded'] += 1
                else:
                    sim_counts['failed'] += 1
                continue

            # Parse sample level
            lvl = 1
            for cand_lvl in (4, 3, 2, 1):
                if f"_L{cand_lvl}_" in m_name:
                    lvl = cand_lvl
                    break

            # If quota for this level is already reached, skip unless needed
            if level_counts[lvl] >= target_levels[lvl] and len(final_synthetic_names) < n_synthetic - 100:
                continue

            # Curvature Gating Policy:
            # If trajectory is flatline (curv < curv_thresh), downsample with flat_prob
            if curv < curv_thresh and rng.rand() > flat_prob:
                sim_counts['flatline_filtered'] += 1
                continue

            # Accept Sample
            all_results[m_name] = y_uniform
            bio_labels_dict[m_name] = m_bio
            pattern_labels_dict[m_name] = m_pat
            curvatures_dict[m_name] = curv
            final_synthetic_names.append(m_name)
            level_counts[lvl] += 1
            sim_counts['synth_success'] += 1

            if len(final_synthetic_names) % 5000 == 0 or len(final_synthetic_names) == n_synthetic:
                elapsed = max(0.001, time.time() - t_sim_start)
                speed = len(final_synthetic_names) / elapsed
                print(f"  |-- Accepted [{len(final_synthetic_names):6d}/{n_synthetic:6d}] "
                      f"(L1:{level_counts[1]}, L2:{level_counts[2]}, L3:{level_counts[3]}, L4:{level_counts[4]}) | "
                      f"Flatlines Downsampled: {sim_counts['flatline_filtered']:5d} | "
                      f"Speed: {speed:5.1f} samples/s", flush=True)

    elapsed_sim = time.time() - t_sim_start
    print(f" -> Simulation & Gating Completed in {elapsed_sim:.1f}s.", flush=True)

    # Phase 4: Format Packaging & .npz Export
    print("\n[Phase 4/4] Packaging & Saving Dataset Array Files (.npz)...", flush=True)
    t_pack_start = time.time()

    all_sample_names = [n for n in ordered_names if n in all_results] + final_synthetic_names
    ordered_tensors = [all_results[n] for n in all_sample_names]
    final_bio_labels = [bio_labels_dict[n] for n in all_sample_names]
    final_pattern_labels = [pattern_labels_dict[n] for n in all_sample_names]

    final_params = []
    for name in all_sample_names:
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
        mutant_names=np.array(all_sample_names),
        labels=all_bio_labels,
        pattern_labels=all_pattern_labels
    )

    pack_time = time.time() - t_pack_start
    file_size_gb = os.path.getsize(output_path) / (1024 ** 3)

    print("=" * 85)
    print(" SIGNED K-DPP + CURVATURE GATED DATASET GENERATION REPORT")
    print("=" * 85)
    print(f"Dataset Export Path               : {output_path}")
    print(f"Export File Size                  : {file_size_gb:.2f} GB")
    print(f"Data Tensor Shape (N, Vars, Steps): {all_tensors.shape}")
    print(f"Params Matrix Shape (N, P_dim)    : {all_params.shape}")
    print(f"Total Real Mutants Saved          : {sim_counts['real_success']}")
    print(f"Total Synthetic Mutants Saved     : {len(final_synthetic_names):,}")
    print(f"Level Distribution Achieved       : L1={level_counts[1]}, L2={level_counts[2]}, L3={level_counts[3]}, L4={level_counts[4]}")
    print(f"Flatline Degenerates Filtered     : {sim_counts['flatline_filtered']:,}")
    print(f"Packaging Time                    : {pack_time:.1f}s")
    print(f"Total Pipeline Runtime            : {time.time() - t_sample_start:.1f}s")
    print("=" * 85 + "\n", flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate 100k Dataset via Signed k-DPP + Curvature Gating")
    parser.add_argument('--output', type=str, default="/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast/kdpp_curv_gated_100000_210m_500s.npz")
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratios', type=float, nargs=4, default=[0.20, 0.40, 0.25, 0.15],
                        help="Level quotas L1, L2, L3, L4 (default: 0.20 0.40 0.25 0.15)")
    parser.add_argument('--curv_thresh', type=float, default=0.01,
                        help="Curvature threshold below which trajectories are treated as flatlines (default: 0.01)")
    parser.add_argument('--flat_prob', type=float, default=0.20,
                        help="Downsample keep-probability for flatline trajectories (default: 0.20)")
    args = parser.parse_args()

    generate_kdpp_curv_gated_dataset(
        output_path=args.output,
        n_synthetic=args.n_samples,
        num_workers=args.workers,
        seed=args.seed,
        ratios=tuple(args.ratios),
        curv_thresh=args.curv_thresh,
        flat_prob=args.flat_prob
    )


if __name__ == '__main__':
    main()
