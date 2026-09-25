#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active Learning Candidate Pool Manager & Oracle Simulation Engine
================================================================================
Handles:
1. Fast MiniBatchKMeans representative clustering on candidate pool
2. Multiprocess ODE simulation of selected candidate samples
3. Full dataset packaging strictly adhering to mutant_aware.py format (including 126 real mutants)
"""

import os
import sys
import time
import argparse
import concurrent.futures
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Ensure import paths
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


def select_initial_kmeans(pool_path, n_samples=10000, seed=42):
    """
    Selects n_samples representatives from the candidate pool using MiniBatchKMeans
    in the asinh(Z) normalized parameter space.
    """
    print(f"\n[AL Pool Manager] Loading candidate pool from {pool_path}...")
    t0 = time.time()
    pool_data = np.load(pool_path, allow_pickle=True)
    params = pool_data['params']  # (N, 141)
    p_mean = pool_data['p_mean']  # (141,)
    p_std = pool_data['p_std']    # (141,)
    n_pool = params.shape[0]

    print(f"  |-- Pool loaded: {n_pool} samples, {params.shape[1]} parameters ({time.time() - t0:.2f}s)")
    print(f"  |-- Normalizing parameter space with asinh(Z)...")
    # asinh(Z) normalization
    z = (params - p_mean) / p_std
    p_norm = np.arcsinh(z)

    print(f"  |-- Running MiniBatchKMeans (K={n_samples} clusters, batch_size=10000, seed={seed})...")
    t_km = time.time()
    mbk = MiniBatchKMeans(
        n_clusters=n_samples,
        batch_size=min(10000, n_pool),
        random_state=seed,
        n_init=3,
        max_iter=100,
        verbose=0
    )
    cluster_labels = mbk.fit_predict(p_norm)
    centers = mbk.cluster_centers_  # (n_samples, 141)
    print(f"  |-- MiniBatchKMeans converged in {time.time() - t_km:.2f}s.")

    print(f"  |-- Finding nearest candidate to each cluster center...")
    # Find nearest candidate per cluster
    selected_indices = np.zeros(n_samples, dtype=np.int64)
    # Block distance evaluation for speed and memory efficiency
    chunk_size = 500
    for c_start in range(0, n_samples, chunk_size):
        c_end = min(c_start + chunk_size, n_samples)
        sub_centers = centers[c_start:c_end]  # (B, 141)
        
        # For each cluster, find sample in that cluster nearest to its center
        for i, c_idx in enumerate(range(c_start, c_end)):
            members = np.where(cluster_labels == c_idx)[0]
            if len(members) > 0:
                dist = np.linalg.norm(p_norm[members] - sub_centers[i], axis=1)
                selected_indices[c_idx] = members[np.argmin(dist)]
            else:
                # Fallback if cluster has 0 members
                dist_all = np.linalg.norm(p_norm - sub_centers[i], axis=1)
                selected_indices[c_idx] = np.argmin(dist_all)

    # Deduplicate in case of duplicate assignments
    unique_selected = np.unique(selected_indices)
    if len(unique_selected) < n_samples:
        n_missing = n_samples - len(unique_selected)
        remaining = np.setdiff1d(np.arange(n_pool), unique_selected)
        rng = np.random.RandomState(seed)
        extra = rng.choice(remaining, size=n_missing, replace=False)
        selected_indices = np.concatenate([unique_selected, extra])
    else:
        selected_indices = unique_selected[:n_samples]

    print(f"  |-- Selected {len(selected_indices)} unique initial samples (total time: {time.time() - t0:.2f}s).")
    return selected_indices


def _simulate_single_task(task_tuple):
    """Worker task wrapper for ProcessPoolExecutor"""
    name, ovr = task_tuple
    try:
        m_name, y_uniform, bio_label, pattern_label = ma.simulate_mutant(name, ovr, 210, 500)
        return m_name, y_uniform, bio_label, pattern_label, ovr
    except Exception:
        return name, None, "Solver_Error", "Pattern_C: Exponential-Arrest (==0)", ovr


def simulate_selected_tasks(selected_indices, pool_path, num_workers=40):
    """
    Solves stiff ODE trajectories for the selected candidate pool indices.
    """
    print(f"\n[AL Pool Manager] Preparing ODE simulation for {len(selected_indices)} selected candidates...")
    pool_data = np.load(pool_path, allow_pickle=True)
    all_names = pool_data['mutant_names']
    all_params = pool_data['params']
    param_keys = pool_data['param_names']
    defaults = ma.get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

    tasks = []
    for idx in selected_indices:
        name = str(all_names[idx])
        p_vec = all_params[idx]
        # Build override dict for mutant_aware
        ovr = {param_keys[i]: float(p_vec[i]) for i in range(len(param_keys))}
        tasks.append((name, ovr))

    print(f"  |-- Simulating {len(tasks)} tasks on {num_workers} CPU workers...")
    t0 = time.time()
    results = {}
    bio_labels = {}
    pattern_labels = {}
    final_params = {}
    valid_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_simulate_single_task, t): t[0] for t in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            name = futures[future]
            try:
                m_name, y_uniform, bio_label, pattern_label, ovr = future.result(timeout=25.0)
                if y_uniform is not None:
                    results[m_name] = y_uniform
                    bio_labels[m_name] = bio_label
                    pattern_labels[m_name] = pattern_label
                    final_params[m_name] = [ovr.get(k, defaults[k]) for k in param_keys]
                    valid_count += 1
            except Exception:
                pass

            if (i + 1) % max(1, len(tasks) // 20) == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t0
                pct = ((i + 1) / len(tasks)) * 100
                speed = (i + 1) / max(0.001, elapsed)
                print(f"  |-- [{i+1:5d}/{len(tasks):5d}] ({pct:5.1f}%) | Valid: {valid_count} | Speed: {speed:5.1f} samples/s", flush=True)

    print(f"  |-- ODE Simulation completed: {valid_count}/{len(tasks)} valid trajectories in {time.time() - t0:.2f}s.")
    return results, bio_labels, pattern_labels, final_params, param_keys


def package_and_export_al_dataset(output_path, synth_results, num_workers=40):
    """
    Packages the simulated synthetic samples together with the 126 Real Mutants gold standard test set,
    strictly aligned with mutant_aware.py format.
    """
    synth_trajs, synth_bio, synth_patterns, synth_params, param_keys = synth_results
    defaults = ma.get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

    print(f"\n[AL Pool Manager] Simulating 126 Real Mutants benchmark test set...")
    real_tasks = []
    for mid, (mname, movr) in ma.REAL_MUTANTS.items():
        sample_name = f"{mid}_{mname}".replace(" ", "_")
        real_tasks.append((sample_name, movr))

    real_trajs = {}
    real_bio = {}
    real_patterns = {}
    real_params = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(40, len(real_tasks))) as executor:
        futures = {executor.submit(_simulate_single_task, t): t[0] for t in real_tasks}
        for future in concurrent.futures.as_completed(futures):
            m_name, y_uniform, bio_label, pattern_label, ovr = future.result()
            if y_uniform is not None:
                real_trajs[m_name] = y_uniform
                real_bio[m_name] = bio_label
                real_patterns[m_name] = pattern_label
                real_params[m_name] = [ovr.get(k, defaults[k]) for k in param_keys]

    print(f"  |-- Real Mutants ready: {len(real_trajs)} valid benchmark strains.")

    # Combine: Real mutants first (for standardized index mapping), then synthetic mutants
    all_names = list(real_trajs.keys()) + list(synth_trajs.keys())
    all_data = [real_trajs[n] for n in real_trajs] + [synth_trajs[n] for n in synth_trajs]
    all_p = [real_params[n] for n in real_trajs] + [synth_params[n] for n in synth_trajs]
    all_labels = [real_bio[n] for n in real_trajs] + [synth_bio[n] for n in synth_trajs]
    all_patterns = [real_patterns[n] for n in real_trajs] + [synth_patterns[n] for n in synth_trajs]

    data_arr = np.array(all_data, dtype=np.float32)
    params_arr = np.array(all_p, dtype=np.float64)
    param_names_arr = np.array(param_keys)
    names_arr = np.array(all_names)
    labels_arr = np.array(all_labels)
    pattern_labels_arr = np.array(all_patterns)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[AL Pool Manager] Exporting packaged dataset to {output_path}...")
    np.savez_compressed(
        output_path,
        data=data_arr,
        params=params_arr,
        param_names=param_names_arr,
        mutant_names=names_arr,
        labels=labels_arr,
        pattern_labels=pattern_labels_arr
    )

    print("=" * 80)
    print(f" DATASET PACKAGING COMPLETED")
    print(f" Export Path        : {output_path}")
    print(f" Total Samples      : {len(all_names)} (126 Real + {len(synth_trajs)} Synthetic)")
    print(f" Trajectory Shape   : {data_arr.shape}")
    print(f" Parameters Shape   : {params_arr.shape}")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Active Learning Candidate Pool Manager")
    parser.add_argument('--action', type=str, required=True, choices=['init_kmeans', 'simulate_and_package'],
                        help="Action to perform")
    parser.add_argument('--pool_path', type=str, default="/home/users/zsy/mydata/budding_yeast/pure_random_1m_pool.npz",
                        help="Candidate pool .npz path")
    parser.add_argument('--n_initial', type=int, default=10000, help="Initial training size via K-Means")
    parser.add_argument('--output_dataset', type=str, default="/home/users/zsy/mydata/budding_yeast/al_iter0_10000_210m_500s.npz",
                        help="Output packaged dataset path")
    parser.add_argument('--num_workers', type=int, default=40, help="Number of CPU workers for ODE simulation")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.action == 'init_kmeans':
        sel_idx = select_initial_kmeans(args.pool_path, n_samples=args.n_initial, seed=args.seed)
        sim_res = simulate_selected_tasks(sel_idx, args.pool_path, num_workers=args.num_workers)
        package_and_export_al_dataset(args.output_dataset, sim_res, num_workers=args.num_workers)
