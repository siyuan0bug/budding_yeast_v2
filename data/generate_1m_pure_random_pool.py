#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1,000,000 Pure Random Candidate Pool Generator with Sobol Inverse-CDF Scaling
================================================================================
Generates 1,000,000 parameter combinations and corresponding initial states y0
strictly for Active Learning without expensive ODE simulation.

Features:
- Direct dependency on budding_yeast_v2.data.mutant_aware
- Level distribution: L1 (5%, 50k) / L2 (25%, 250k) / L3 (40%, 400k) / L4 (30%, 300k)
- Sobol quasi-random low-discrepancy coordinates mapped via Inverse-CDF quantile functions
- Output: 141-dim physical parameters P and 39-dim initial state conditions y0
- Strictly aligned with mutant_aware.py export formats
"""

import os
import sys
import math
import time
import argparse
import numpy as np
from scipy.stats import qmc, beta, norm

# Ensure project root is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_v2_root = os.path.dirname(_script_dir)
_repo_root = os.path.dirname(_v2_root)
for p in [_script_dir, _v2_root, _repo_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import budding_yeast_v2.data.mutant_aware as ma
except ImportError:
    try:
        import data.mutant_aware as ma
    except ImportError:
        import mutant_aware as ma

# Precomputed discrete CDF for Multi-Copy (MC) integer range 2..30 with P(n) ~ 1/n
_MC_NS = np.arange(2, 31)
_MC_WEIGHTS = 1.0 / _MC_NS
_MC_PROBS = _MC_WEIGHTS / _MC_WEIGHTS.sum()
_MC_CDF = np.cumsum(_MC_PROBS)

# Mapping of init_* keys in ovr to initial condition indices
_INIT_Y0_MAP = {
    'init_CDH1T': ma.nCDH1T, 'init_CDH1': ma.nCDH1, 'init_BUB2': ma.nBUB2,
    'init_CLN2': ma.nCLN2, 'init_CLB2': ma.nCLB2, 'init_CLB5': ma.nCLB5,
    'init_SIC1': ma.nSIC1, 'init_CDC6': ma.nCDC6, 'init_CDC14': ma.nCDC14,
    'init_NET1': ma.nNET1, 'init_CDC20': ma.nCDC20, 'init_Tem1': ma.nTEM1,
    'init_Cdc15': ma.nCDC15, 'init_PPX': ma.nPPX, 'init_PDS1': ma.nPDS1,
    'init_ESP1': ma.nESP1,
}


def sample_scaling_factor_sobol(mode, g_name, u):
    """
    Maps a low-discrepancy uniform coordinate u in (0, 1) to the exact physiological
    parameter scaling factor via Inverse-CDF (Quantile Transformation).
    Matches mutant_aware.py distributions 100% identically.
    """
    u = np.clip(u, 1e-6, 1.0 - 1e-6)

    if mode == 'GAL-OE':
        if g_name == 'CDC20':
            lo, hi = np.log10(0.50), np.log10(12.0)
        elif g_name == 'CLB2':
            lo, hi = np.log10(0.002), np.log10(1.50)
        elif g_name == 'SIC1':
            lo, hi = np.log10(0.03), np.log10(0.36)
        elif g_name == 'CDC6':
            lo, hi = np.log10(0.10), np.log10(0.60)
        elif g_name == 'CLB5':
            lo, hi = np.log10(0.004), np.log10(0.030)
        elif g_name == 'PDS1':
            lo, hi = np.log10(0.02), np.log10(0.25)
        elif g_name == 'CLN2':
            lo, hi = np.log10(0.06), np.log10(0.30)
        elif g_name == 'CLN3':
            lo, hi = np.log10(1.0), np.log10(5.0)
        elif g_name == 'TEM1':
            lo, hi = np.log10(2.0), np.log10(8.0)
        elif g_name == 'CDC15':
            lo, hi = np.log10(3.0), np.log10(20.0)
        elif g_name == 'ESP1':
            lo, hi = np.log10(1.5), np.log10(6.0)
        elif g_name == 'CDC14':
            lo, hi = np.log10(0.30), np.log10(1.20)
        elif g_name == 'NET1':
            lo, hi = np.log10(0.15), np.log10(0.60)
        elif g_name in ['PPX', 'CDC5', 'CDC5_Proxy']:
            lo, hi = np.log10(0.20), np.log10(0.60)
        elif g_name == 'SWI5':
            lo, hi = np.log10(0.02), np.log10(0.10)
        else:
            lo, hi = np.log10(2.0), np.log10(30.0)
        return float(10.0 ** (lo + u * (hi - lo)))

    elif mode == 'MC':
        idx = np.searchsorted(_MC_CDF, u)
        return float(_MC_NS[min(idx, len(_MC_NS) - 1)])

    elif mode == 'ts':
        if g_name == 'CDC14':
            return float(0.05 + u * (0.35 - 0.05))
        if g_name in ('TEM1', 'ESP1'):
            lo, hi = np.log10(0.0005), np.log10(0.004)
            return float(10.0 ** (lo + u * (hi - lo)))
        if g_name == 'NET1':
            lo, hi = np.log10(0.0125), np.log10(0.10)
            return float(10.0 ** (lo + u * (hi - lo)))
        if g_name in ('CDC20', 'CDC15', 'APC_Core'):
            return float(0.0 + u * 0.01)
        return float(0.20 * beta.ppf(u, 0.5, 2.0))

    elif mode == 'DAmP':
        val = float(norm.ppf(u, loc=0.20, scale=0.05))
        return float(np.clip(val, 0.05, 0.40))

    elif mode == 'pLOF':
        return float(0.10 + u * (0.50 - 0.10))

    elif mode == 'db':
        if g_name == 'CLB2':
            return float(0.02125 + u * (0.170 - 0.02125))
        elif g_name == 'PDS1':
            return float(0.0 + u * 0.01)
        elif g_name in ('CLB5', 'SIC1', 'PPX', 'CDC5_Proxy'):
            return float(0.0 + u * 0.01)
        else:
            return 0.0

    return 1.0


def sample_pure_random_candidate(level, sobol_vec, rng):
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
        # Fallback with rolled coordinate
        return sample_pure_random_candidate(level, np.roll(sobol_vec, 1), rng)

    has_gal_oe = False
    mut_descriptions = []
    coord_idx = 0

    for g_name, m_chosen in pairs:
        mut_descriptions.append(f"{g_name}[{m_chosen}]")
        if m_chosen == 'GAL-OE':
            has_gal_oe = True

        if m_chosen == 'KO':
            alpha = 0.0
        elif m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db']:
            u_coord = sobol_vec[coord_idx % len(sobol_vec)]
            coord_idx += 1
            alpha = sample_scaling_factor_sobol(m_chosen, g_name, u_coord)
        else:
            alpha = 1.0

        ma.apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)

    # Carbon source handling with Sobol mapping
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

    desc_str = "+".join(mut_descriptions)
    return ovr, desc_str, pairs


def generate_1m_pure_random_pool(n_total=1000000,
                                 output_path="/home/users/zsy/mydata/budding_yeast/pure_random_1m_pool.npz",
                                 level_ratios=(0.05, 0.25, 0.40, 0.30),
                                 seed=42):
    """
    Generates 1,000,000 pure random parameter sets without solving ODEs.
    """
    print("=" * 80)
    print(f"1,000,000 PURE RANDOM CANDIDATE POOL GENERATOR (Sobol Inverse-CDF)")
    print(f"Target Samples : {n_total}")
    print(f"Output File    : {output_path}")
    print(f"Level Quotas   : L1={level_ratios[0]*100:.1f}%, L2={level_ratios[1]*100:.1f}%, "
          f"L3={level_ratios[2]*100:.1f}%, L4={level_ratios[3]*100:.1f}%")
    print(f"Seed           : {seed}")
    print("=" * 80, flush=True)

    defaults = ma.get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    param_keys = sorted([k for k in defaults.keys() if k != 'is_mutant_104'])
    n_params = len(param_keys)

    # Target allocations
    n_l1 = int(round(n_total * level_ratios[0]))
    n_l2 = int(round(n_total * level_ratios[1]))
    n_l3 = int(round(n_total * level_ratios[2]))
    n_l4 = n_total - (n_l1 + n_l2 + n_l3)

    levels = np.array([1]*n_l1 + [2]*n_l2 + [3]*n_l3 + [4]*n_l4, dtype=np.int16)
    rng = np.random.RandomState(seed)
    rng.shuffle(levels)

    # Scrambled Sobol quasi-random low-discrepancy generator (dimension=8)
    sobol_dim = 8
    sobol_gen = qmc.Sobol(d=sobol_dim, scramble=True, seed=seed)
    
    # Pre-allocate output arrays
    print(f"\n[1/3] Pre-allocating memory for {n_total} candidates...")
    params_mat = np.zeros((n_total, n_params), dtype=np.float64)
    y_init_mat = np.zeros((n_total, 39), dtype=np.float32)
    mutant_names = []

    y0_base = ma.initial_conds()  # (39,) base initial condition

    # Batch generation for high throughput
    batch_size = 65536
    t0 = time.time()
    
    print(f"[2/3] Generating candidates with Sobol Inverse-CDF sampling...")
    for start_idx in range(0, n_total, batch_size):
        end_idx = min(start_idx + batch_size, n_total)
        cur_batch_size = end_idx - start_idx
        
        # Draw Sobol coordinates for current batch
        sobol_points = sobol_gen.random(cur_batch_size)

        for i in range(cur_batch_size):
            global_idx = start_idx + i
            lvl = int(levels[global_idx])
            s_vec = sobol_points[i]

            ovr, desc_str, pairs = sample_pure_random_candidate(lvl, s_vec, rng)

            # Assemble 141-dim physical parameter vector
            p_vec = [ovr.get(k, defaults[k]) for k in param_keys]
            params_mat[global_idx, :] = p_vec

            # Assemble 39-dim initial condition vector
            y0 = y0_base.copy()
            for k, val in ovr.items():
                if k in _INIT_Y0_MAP:
                    y0[_INIT_Y0_MAP[k]] = float(val)
            y_init_mat[global_idx, :] = y0.astype(np.float32)

            name = f"MASS_RAND_{desc_str}_L{lvl}_{global_idx + 1}"
            mutant_names.append(name)

        pct = (end_idx / n_total) * 100
        elapsed = time.time() - t0
        speed = end_idx / max(0.001, elapsed)
        print(f"  |-- Processed [{end_idx:7d}/{n_total:7d}] ({pct:5.1f}%) | Speed: {speed:6.1f} samples/s", flush=True)

    print(f"\n[3/3] Computing global statistics and saving to disk...")
    p_mean = np.mean(params_mat, axis=0)
    p_std = np.std(params_mat, axis=0)
    p_std = np.where(p_std < 1e-8, 1.0, p_std)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        output_path,
        params=params_mat,
        param_names=np.array(param_keys),
        mutant_names=np.array(mutant_names),
        y_init=y_init_mat,
        levels=levels,
        p_mean=p_mean,
        p_std=p_std,
    )

    total_time = time.time() - t0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("=" * 80)
    print(f" CANDIDATE POOL GENERATION COMPLETED SUCCESSFULLY")
    print(f" Total Time    : {total_time:.2f} seconds ({n_total / total_time:.1f} samples/s)")
    print(f" Export Path   : {output_path}")
    print(f" File Size     : {file_size_mb:.2f} MB")
    print(f" Params Shape  : {params_mat.shape}")
    print(f" y_init Shape  : {y_init_mat.shape}")
    print(f" Number Keys   : {len(param_keys)}")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate 1M Pure Random Candidate Pool with Sobol Inverse-CDF")
    parser.add_argument('--n_samples', type=int, default=1000000, help="Total candidate pool size (default: 1,000,000)")
    parser.add_argument('--output', type=str, default="/home/users/zsy/mydata/budding_yeast/pure_random_1m_pool.npz",
                        help="Target output .npz path")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_1m_pure_random_pool(n_total=args.n_samples, output_path=args.output, seed=args.seed)
