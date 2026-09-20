#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BAMS 50,000 Ablation Dataset Generator
================================================================================
Generates 3 rigorously aligned datasets (50,000 samples each):
1. Control (0-targeted): bams_50000_210m_500s.npz
   - 126 Real Mutants + 50,000 pure unbiased BAMS samples.
2. Arm A (Hard Zeros): bams_ablation_hard_50000_210m_500s.npz
   - 126 Real Mutants + 40,000 base BAMS samples + 10,000 targeted badcase skeletons with Hard Zeros (0.0).
3. Arm B (Continuous Hypomorphic): bams_ablation_hypo_50000_210m_500s.npz
   - 126 Real Mutants + 40,000 base BAMS samples + 10,000 targeted badcase skeletons with Continuous Hypomorphic (ts-like decay).

Key Biophysical Design:
- Base 40,000 samples across Arm A and Arm B are 100% BIT-FOR-BIT IDENTICAL to the first 40,000 samples of Control.
- The 10,000 targeted skeletons across Arm A and Arm B share EXACT SAME skeleton pairing and GOF dosages.
- GOF dosages 100% strictly sampled via sample_scaling_factor(mode, g_name).
- Hypomorphic decays strictly aligned with calibrated ts Beta(0.5, 2.0) and Uniform(0.0, 0.01) envelopes.
- Complete 141-D deduplication against all 126 Real Mutants.
"""

import os
import sys
import time
import math
import random
import warnings
import numpy as np
import concurrent.futures

warnings.filterwarnings("ignore")

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from data.mutant_aware import (
    get_default_params, PARAM_KEYS,
    simulate_mutant, is_duplicate_real_mutant,
    sample_scaling_factor, apply_gene_mutation
)

TARGETED_SKELETONS = [
    # 1. M15: cln3Δ bck2Δ multi-copy CLN2
    {'desc': 'M15_mc', 'lofs': ['CLN3', 'BCK2'], 'gofs': [('CLN2', 'MC')]},
    {'desc': 'M15_gal', 'lofs': ['CLN3', 'BCK2'], 'gofs': [('CLN2', 'GAL-OE')]},
    # 2. M16: cln3Δ bck2Δ sic1Δ
    {'desc': 'M16', 'lofs': ['CLN3', 'BCK2', 'SIC1'], 'gofs': []},
    # 3. M13: cln3Δ bck2Δ
    {'desc': 'M13', 'lofs': ['CLN3', 'BCK2'], 'gofs': []},
    # 4. M22: cln1Δ cln2Δ cln3Δ multi-copy CLB5
    {'desc': 'M22_mc', 'lofs': ['CLN2', 'CLN3'], 'gofs': [('CLB5', 'MC')]},
    # 5. M23: cln1Δ cln2Δ cln3Δ GAL-CLB5
    {'desc': 'M23_gal', 'lofs': ['CLN2', 'CLN3'], 'gofs': [('CLB5', 'GAL-OE')]},
    # 6. M24: cln1Δ cln2Δ cln3Δ multi-copy BCK2
    {'desc': 'M24_mc', 'lofs': ['CLN2', 'CLN3'], 'gofs': [('BCK2', 'MC')]},
    # 7. M20: cln1Δ cln2Δ cln3Δ sic1Δ
    {'desc': 'M20', 'lofs': ['CLN2', 'CLN3', 'SIC1'], 'gofs': []},
    # 8. M33: GAL-SIC1 GAL-CLN2 cln1Δ cln2Δ cdh1Δ
    {'desc': 'M33', 'lofs': ['CLN2', 'CDH1'], 'gofs': [('SIC1', 'GAL-OE'), ('CLN2', 'GAL-OE')]},
    # 9. M31: GAL-SIC1 GAL-CLN2 cln1Δ cln2Δ
    {'desc': 'M31', 'lofs': ['CLN2'], 'gofs': [('SIC1', 'GAL-OE'), ('CLN2', 'GAL-OE')]},
    # 10. M42: sic1Δ cdc6Δ 2-49 cdh1Δ GAL-CDC20
    {'desc': 'M42', 'lofs': ['SIC1', 'CDC6', 'CDH1'], 'gofs': [('CDC20', 'GAL-OE')]},
    # 11. M58: CLB2-dbΔ GAL-SIC1
    {'desc': 'M58', 'lofs': [('CLB2', 'db')], 'gofs': [('SIC1', 'GAL-OE')]},
    # 12. M53: GAL-CLB2 sic1Δ
    {'desc': 'M53', 'lofs': ['SIC1'], 'gofs': [('CLB2', 'GAL-OE')]},
    # 13. M124: APC-A GAL-CLB2
    {'desc': 'M124', 'lofs': [], 'gofs': [('APC_Core', 'APC-A'), ('CLB2', 'GAL-OE')]},
    # 14. M76: cdc20Δ pds1Δ clb5Δ
    {'desc': 'M76', 'lofs': ['CDC20', 'PDS1', 'CLB5'], 'gofs': []}
]


def sample_ts_alpha(g_name):
    """
    Calibrated ts decay factor matching mutant_aware.py:
    - Special genes (CDC20, PDS1, CLB5): <=1% boundary buffer U[0, 0.01]
    - CLB2-db: U[0.02125, 0.170]
    - Non-essential genes (CLN3, BCK2, CLN2, SIC1, CDH1, CDC6):
        50% generic ts: 0.20 * Beta(0.5, 2.0) (support (0, 0.20], mean 0.04)
        50% boundary buffer: U[0.0, 0.01] (support (0, 0.01], mean 0.005)
    """
    if g_name in ('CDC20', 'PDS1', 'CLB5'):
        return float(np.random.uniform(0.0, 0.01))
    if g_name == 'CLB2_db':
        return float(np.random.uniform(0.02125, 0.170))
    if np.random.rand() < 0.50:
        return float(0.20 * np.random.beta(0.5, 2.0))
    else:
        return float(np.random.uniform(0.0, 0.01))


def apply_targeted_mutation(g_entry, is_hypo, p_wt, ovr, decay_scalars):
    """
    Applies mutation according to biophysical categories:
    1. 纯敲除基因 (mode == 'KO'):
       - is_hypo = False (Arm A): 严格赋硬零点 0.0 (敲除基因不变，保持二值断崖)
       - is_hypo = True (Arm B): 赋连续弱化过渡态 (ts/Beta 衰减，消除硬断崖)
    2. 减弱基因 (mode in ('db', 'ts', 'DAmP', 'pLOF')):
       - 在 Arm A 与 Arm B 中，减弱倍数均动态改变 (从物理分布中抽取，不等于 real mutant 固定标称值)
       - Arm A 与 Arm B 共享同一个抽取的减弱倍数，确保严格的单变量控制！
    """
    if isinstance(g_entry, tuple):
        g_name, mode = g_entry
    else:
        g_name, mode = g_entry, 'KO'

    # Case 2: 减弱基因 (非纯敲除，属于衰弱/减弱突变) -> Arm A 和 Arm B 都改变衰弱倍数！
    if mode in ('db', 'ts', 'DAmP', 'pLOF'):
        alpha = decay_scalars.get((g_name, mode), sample_scaling_factor(mode, g_name))
        apply_gene_mutation(g_name, mode, alpha, p_wt, {}, ovr)
        return

    # Case 1: 纯敲除基因 (mode == 'KO')
    if not is_hypo:
        # Arm A (Hard Zeros): 敲除基因数值严格不变，赋精确 0.0
        if g_name == 'CLN3':
            ovr['C0'] = 0.0
        elif g_name == 'BCK2':
            ovr['B0'] = 0.0
        elif g_name == 'CLN2':
            ovr['ksn2pp'] = 0.0
        elif g_name == 'SIC1':
            ovr['ksc1p'] = 0.0; ovr['ksc1pp'] = 0.0
        elif g_name == 'CDH1':
            ovr['kscdh'] = 0.0; ovr['init_CDH1T'] = 0.0; ovr['init_CDH1'] = 0.0
        elif g_name == 'CDC6':
            ovr['ksf6p'] = 0.0; ovr['ksf6pp'] = 0.0; ovr['ksf6ppp'] = 0.0
        elif g_name == 'CLB5':
            ovr['ksb5p'] = 0.0; ovr['ksb5pp'] = 0.0
        elif g_name == 'CDC20':
            ovr['ks20p'] = 0.0; ovr['ks20pp'] = 0.0
        elif g_name == 'PDS1':
            ovr['kspdsp'] = 0.0
        elif g_name == 'CLB2':
            ovr['ksb2p'] = 0.0; ovr['ksb2pp'] = 0.0
    else:
        # Arm B (Continuous Hypomorphic): 敲除基因赋连续弱等位基因过渡态 (消除断崖)
        alpha = sample_ts_alpha(g_name)
        if g_name == 'CLN3':
            ovr['C0'] = p_wt['C0'] * alpha
        elif g_name == 'BCK2':
            ovr['B0'] = p_wt['B0'] * alpha
        elif g_name == 'CLN2':
            ovr['ksn2pp'] = p_wt['ksn2pp'] * alpha
        elif g_name == 'SIC1':
            ovr['ksc1p'] = p_wt['ksc1p'] * alpha; ovr['ksc1pp'] = p_wt['ksc1pp'] * alpha
        elif g_name == 'CDH1':
            ovr['kscdh'] = p_wt['kscdh'] * alpha
            ovr['init_CDH1T'] = p_wt.get('init_CDH1T', 1.0) * alpha
            ovr['init_CDH1'] = p_wt.get('init_CDH1', 0.9304992) * alpha
        elif g_name == 'CDC6':
            ovr['ksf6p'] = p_wt['ksf6p'] * alpha
            ovr['ksf6pp'] = p_wt['ksf6pp'] * alpha
            ovr['ksf6ppp'] = p_wt['ksf6ppp'] * alpha
        elif g_name == 'CLB5':
            ovr['ksb5p'] = p_wt['ksb5p'] * alpha; ovr['ksb5pp'] = p_wt['ksb5pp'] * alpha
        elif g_name == 'CDC20':
            ovr['ks20p'] = p_wt['ks20p'] * alpha; ovr['ks20pp'] = p_wt['ks20pp'] * alpha
        elif g_name == 'PDS1':
            ovr['kspdsp'] = p_wt['kspdsp'] * alpha
        elif g_name == 'CLB2':
            ovr['ksb2p'] = p_wt['ksb2p'] * alpha; ovr['ksb2pp'] = p_wt['ksb2pp'] * alpha


def generate_targeted_pairs(n_samples=10000, seed=20260915):
    """
    Generates n_samples paired specifications for Arm A (Hard) and Arm B (Hypo).
    Both share EXACT same skeleton and GOF multipliers.
    """
    np.random.seed(seed)
    random.seed(seed)
    p_wt = get_default_params()
    p_wt.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

    tasks_A = []
    tasks_B = []

    print(f"Generating {n_samples} matched targeted skeleton pairs...", flush=True)
    count = 0
    attempts = 0

    while count < n_samples:
        attempts += 1
        sk = TARGETED_SKELETONS[count % len(TARGETED_SKELETONS)]
        
        has_gal = any(m == 'GAL-OE' for g, m in sk['gofs'])
        mdt = 150.0 if has_gal else float(np.random.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))

        # Pre-sample GOF multipliers so both arms are 100% matched
        gof_scalars = {}
        for g, m in sk['gofs']:
            gof_scalars[(g, m)] = sample_scaling_factor(m, g)

        # Pre-sample decay multipliers for any decay/ts/db/pLOF genes so both arms match
        decay_scalars = {}
        for lof in sk['lofs']:
            if isinstance(lof, tuple):
                g, m = lof
                if m in ('db', 'ts', 'DAmP', 'pLOF'):
                    decay_scalars[(g, m)] = sample_scaling_factor(m, g)

        # Arm A (Hard)
        ovr_A = {'mdt': mdt}
        for lof in sk['lofs']:
            apply_targeted_mutation(lof, is_hypo=False, p_wt=p_wt, ovr=ovr_A, decay_scalars=decay_scalars)
        for (g, m), alpha in gof_scalars.items():
            apply_gene_mutation(g, m, alpha, p_wt, {}, ovr_A)

        # Arm B (Hypo)
        ovr_B = {'mdt': mdt}
        for lof in sk['lofs']:
            apply_targeted_mutation(lof, is_hypo=True, p_wt=p_wt, ovr=ovr_B, decay_scalars=decay_scalars)
        for (g, m), alpha in gof_scalars.items():
            apply_gene_mutation(g, m, alpha, p_wt, {}, ovr_B)

        # Zero-leakage deduplication
        if is_duplicate_real_mutant(ovr_A, p_wt) or is_duplicate_real_mutant(ovr_B, p_wt):
            continue

        name_A = f"BAMS_TARGET_HARD_{count}_{sk['desc']}"
        name_B = f"BAMS_TARGET_HYPO_{count}_{sk['desc']}"

        tasks_A.append((name_A, ovr_A, None, 210, 500))
        tasks_B.append((name_B, ovr_B, None, 210, 500))
        count += 1

    print(f"Successfully constructed {n_samples} matched pairs in {attempts} attempts.", flush=True)
    return tasks_A, tasks_B


def _worker_sim(task):
    name, ovr, _, tf, ns = task
    res = simulate_mutant(name, ovr, tf=tf, ns=ns)
    return res, ovr


def run_parallel_simulation(tasks, max_workers=32):
    print(f"Simulating {len(tasks)} mutants using {max_workers} parallel workers...", flush=True)
    results = []
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, res in enumerate(executor.map(_worker_sim, tasks, chunksize=50)):
            results.append(res)
            if (i + 1) % 2500 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t0
                print(f"  Progress: {i+1}/{len(tasks)} completed ({elapsed:.1f}s)", flush=True)
    return results


def pack_and_save_npz(base_data, sim_results, out_path, num_base=40000):
    """
    Packs 126 Real + num_base base BAMS + len(sim_results) targeted synthetic samples.
    """
    p_wt = get_default_params()
    p_wt.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    param_keys = PARAM_KEYS

    # Base slices
    real_count = 126
    total_base_keep = real_count + num_base

    base_data_arr = base_data['data'][:total_base_keep]
    base_params_arr = base_data['params'][:total_base_keep]
    base_mutants_arr = base_data['mutant_names'][:total_base_keep]
    base_labels_arr = base_data['labels'][:total_base_keep]
    base_patterns_arr = base_data['pattern_labels'][:total_base_keep]

    # Targeted data
    n_new = len(sim_results)
    new_data = np.zeros((n_new, 39, 500), dtype=np.float32)
    new_params = np.zeros((n_new, len(param_keys)), dtype=np.float64)
    new_mutants = []
    new_labels = []
    new_patterns = []

    for idx, ((name, yu, labels, pat), ovr) in enumerate(sim_results):
        if yu is None:
            new_data[idx] = 0.0
        else:
            new_data[idx] = yu.astype(np.float32)

        p_full = p_wt.copy()
        for k, v in ovr.items():
            if not k.startswith('init_'):
                p_full[k] = v
        for p_i, pk in enumerate(param_keys):
            new_params[idx, p_i] = p_full[pk]

        new_mutants.append(name)
        new_labels.append(labels if labels else "Unknown")
        new_patterns.append(pat if pat else "Pattern_C: Exponential-Arrest (==0)")

    # Concatenate
    final_data = np.concatenate([base_data_arr, new_data], axis=0)
    final_params = np.concatenate([base_params_arr, new_params], axis=0)
    final_mutants = np.concatenate([base_mutants_arr, np.array(new_mutants, dtype=object)], axis=0)
    final_labels = np.concatenate([base_labels_arr, np.array(new_labels, dtype=object)], axis=0)
    final_patterns = np.concatenate([base_patterns_arr, np.array(new_patterns, dtype=object)], axis=0)

    print(f"Saving {final_data.shape[0]} samples to {out_path}...", flush=True)
    np.savez_compressed(
        out_path,
        data=final_data,
        params=final_params,
        param_names=base_data['param_names'],
        mutant_names=final_mutants,
        labels=final_labels,
        pattern_labels=final_patterns
    )
    print(f"Successfully saved {out_path} (size: {os.path.getsize(out_path) / 1e9:.2f} GB).", flush=True)


def main():
    data_dir = "/home/users/zsy/mydata/budding_yeast"
    src_100k_path = os.path.join(data_dir, "bams_100000_210m_500s.npz")
    
    if not os.path.exists(src_100k_path):
        print(f"Error: Base file {src_100k_path} not found!", file=sys.stderr)
        sys.exit(1)

    print(f"Loading source dataset: {src_100k_path} ...", flush=True)
    bams_100k = np.load(src_100k_path, allow_pickle=True)

    # 1. Generate Control Dataset: bams_50000_210m_500s.npz
    out_ctrl = os.path.join(data_dir, "bams_50000_210m_500s.npz")
    if not os.path.exists(out_ctrl):
        print("=== Step 1: Exporting 50k Control Dataset (0-targeted) ===", flush=True)
        total_ctrl = 126 + 50000
        np.savez_compressed(
            out_ctrl,
            data=bams_100k['data'][:total_ctrl],
            params=bams_100k['params'][:total_ctrl],
            param_names=bams_100k['param_names'],
            mutant_names=bams_100k['mutant_names'][:total_ctrl],
            labels=bams_100k['labels'][:total_ctrl],
            pattern_labels=bams_100k['pattern_labels'][:total_ctrl]
        )
        print(f"Control dataset saved: {out_ctrl} ({os.path.getsize(out_ctrl)/1e9:.2f} GB)")
    else:
        print(f"Control dataset already exists: {out_ctrl}")

    # 2. Construct 10,000 Matched Targeted Skeletons
    tasks_A, tasks_B = generate_targeted_pairs(n_samples=10000, seed=20260915)

    # 3. Simulate Arm A (Hard Zeros)
    out_hard = os.path.join(data_dir, "bams_ablation_hard_50000_210m_500s.npz")
    if not os.path.exists(out_hard):
        print("\n=== Step 2: Simulating Arm A (10k Targeted Hard Zeros) ===", flush=True)
        results_A = run_parallel_simulation(tasks_A, max_workers=32)
        pack_and_save_npz(bams_100k, results_A, out_hard, num_base=40000)
    else:
        print(f"Arm A dataset already exists: {out_hard}")

    # 4. Simulate Arm B (Continuous Hypomorphic)
    out_hypo = os.path.join(data_dir, "bams_ablation_hypo_50000_210m_500s.npz")
    if not os.path.exists(out_hypo):
        print("\n=== Step 3: Simulating Arm B (10k Targeted Continuous Hypomorphic) ===", flush=True)
        results_B = run_parallel_simulation(tasks_B, max_workers=32)
        pack_and_save_npz(bams_100k, results_B, out_hypo, num_base=40000)
    else:
        print(f"Arm B dataset already exists: {out_hypo}")

    print("\n===========================================================")
    print(" ALL 3 ABLATION DATASETS READY!")
    print(f" 1. Control (0-targeted) : {out_ctrl}")
    print(f" 2. Arm A (Hard Zeros)   : {out_hard}")
    print(f" 3. Arm B (Hypomorphic)  : {out_hypo}")
    print("===========================================================")


if __name__ == "__main__":
    main()
