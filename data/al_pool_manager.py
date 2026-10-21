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
import json
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

from budding_yeast_v2.data.systems.budding import api as ma


def select_initial_kmeans(pool_path, n_samples=10000, seed=42, _indices=None):
    """
    Selects n_samples representatives from the candidate pool using MiniBatchKMeans
    in the asinh(Z) normalized parameter space.
    """
    print(f"\n[AL Pool Manager] Loading candidate pool from {pool_path}...")
    t0 = time.time()
    pool_data = np.load(pool_path, allow_pickle=True)
    if _indices is None:
        from budding_yeast_v2.data.systems.generation import level_schedule
        from budding_yeast_v2.data.systems.sampling_policy import STATIC_RATIOS, VERSION
        if str(pool_data.get('sampling_policy','')) != VERSION or 'module_counts' not in pool_data:
            raise ValueError('Regenerate the pool with exact L1/L2/L3 metadata before initial selection')
        counts = pool_data['module_counts']
        pool_data.close()
        quotas = np.bincount(level_schedule(n_samples,STATIC_RATIOS,np.random.RandomState(seed)),minlength=4)
        selected=[]
        for level in (1,2,3):
            indices=np.flatnonzero(counts==level)
            if len(indices)<quotas[level]:raise ValueError(f'Insufficient L{level} candidates for initial dataset')
            if quotas[level]:
                selected.extend(indices[select_initial_kmeans(pool_path,int(quotas[level]),seed+level,_indices=indices)])
        return np.asarray(selected,dtype=np.int64)
    params = pool_data['params']  # (N, 141)
    p_mean = pool_data['p_mean']  # (141,)
    p_std = pool_data['p_std']    # (141,)
    if _indices is not None:params=params[_indices]
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
    # NPZ indexing decompresses a whole member; load it once, not once per selection.
    all_overrides = pool_data['overrides_json'] if 'overrides_json' in pool_data else None
    if 'system_id' in pool_data and str(pool_data['system_id'])!='budding':
        raise ValueError('The AL oracle currently supports only Budding pools')
    if list(param_keys)!=list(ma.PARAM_KEYS):
        raise ValueError('Pool parameter names/order do not match the Budding oracle')
    defaults = ma.get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

    tasks = []
    for idx in selected_indices:
        name = str(all_names[idx])
        p_vec = all_params[idx]
        # Build override dict for mutant_aware
        ovr = (json.loads(str(all_overrides[idx])) if all_overrides is not None
               else {param_keys[i]: float(p_vec[i]) for i in range(len(param_keys))})
        tasks.append((name, ovr))
    pool_data.close()

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
    ordered=[name for name,_ in tasks if name in results]
    return ({name:results[name] for name in ordered}, bio_labels, pattern_labels, final_params, param_keys)


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

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_workers, len(real_tasks))) as executor:
        futures = {executor.submit(_simulate_single_task, t): t[0] for t in real_tasks}
        for future in concurrent.futures.as_completed(futures):
            m_name, y_uniform, bio_label, pattern_label, ovr = future.result()
            if y_uniform is not None:
                real_trajs[m_name] = y_uniform
                real_bio[m_name] = bio_label
                real_patterns[m_name] = pattern_label
                real_params[m_name] = [ovr.get(k, defaults[k]) for k in param_keys]

    print(f"  |-- Real Mutants ready: {len(real_trajs)} valid benchmark strains.")
    if len(real_trajs)!=len(real_tasks):
        raise RuntimeError('Incomplete real-mutant benchmark; refusing to export a partial real test set')
    # Completion order must not reorder the canonical real benchmark.
    real_trajs={name:real_trajs[name] for name,_ in real_tasks}

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

    # Preserve the historical direct packaging API; tag auditable current inputs.
    from budding_yeast_v2.data.systems.sampling_policy import VERSION, valid_candidate
    from budding_yeast_v2.data.systems.contracts import Candidate
    sampling_fields={}
    checked=[]
    for name in synth_trajs:
        try:level=int(str(name).rsplit('_L',1)[1].split('_')[0])
        except (IndexError,ValueError):break
        candidate=valid_candidate(Candidate(1,str(name),{},level),'budding')
        if candidate is None:break
        checked.append(candidate)
    else:
        levels=[row.level for row in checked]
        sampling_fields=dict(sampling_policy=np.array(VERSION),
            levels=np.array([0]*len(real_trajs)+levels,dtype=np.int16),
            module_counts=np.array([-1]*len(real_trajs)+levels,dtype=np.int16))
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
        pattern_labels=pattern_labels_arr,
        is_real=np.array([True]*len(real_trajs)+[False]*len(synth_trajs)),
        y_init=data_arr[:,:,0].astype(np.float64),
        system_id=np.array('budding'),
        variable_names=np.array(ma.VARIABLE_NAMES),
        time_grid=np.linspace(0,210,500),
        artifact_kind=np.array('trajectory_dataset'), **sampling_fields
    )

    print("=" * 80)
    print(f" DATASET PACKAGING COMPLETED")
    print(f" Export Path        : {output_path}")
    print(f" Total Samples      : {len(all_names)} ({len(real_trajs)} Real + {len(synth_trajs)} Synthetic)")
    print(f" Trajectory Shape   : {data_arr.shape}")
    print(f" Parameters Shape   : {params_arr.shape}")
    print("=" * 80, flush=True)


def build_initial_dataset(pool_path, output_path, n_initial=10000, seed=42, num_workers=40):
    """Stratified KMeans, same-level failure replacement, one final real benchmark."""
    from collections import Counter
    from budding_yeast_v2.data.systems.generation import level_schedule
    from budding_yeast_v2.data.systems.sampling_policy import STATIC_RATIOS
    selected=select_initial_kmeans(pool_path,n_initial,seed)
    with np.load(pool_path,allow_pickle=True) as archive:
        counts=archive['module_counts'];names=archive['mutant_names']
    targets=Counter(level_schedule(n_initial,STATIC_RATIOS,np.random.RandomState(seed)).tolist())
    mapping={str(name):int(level) for name,level in zip(names,counts)}
    used=set();combined=({}, {}, {}, {})
    rng=np.random.RandomState(seed)
    while True:
        used.update(map(int,selected))
        rows=simulate_selected_tasks(selected,pool_path,num_workers)
        for destination,source in zip(combined,rows):destination.update(source)
        got=Counter(mapping[name] for name in combined[0])
        missing={level:targets[level]-got[level] for level in targets if got[level]<targets[level]}
        if not missing:break
        if len(used)>=30*n_initial:raise RuntimeError(f'Initial valid quota exhausted: {missing}')
        selected=[]
        for level,number in missing.items():
            available=np.array([i for i in np.flatnonzero(counts==level) if int(i) not in used])
            if len(available)<number:raise RuntimeError(f'Initial pool exhausted at L{level}: missing {number}')
            selected.extend(rng.choice(available,number,replace=False))
    return package_and_export_al_dataset(output_path,combined,num_workers=num_workers)


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
        build_initial_dataset(args.pool_path,args.output_dataset,args.n_initial,args.seed,args.num_workers)
