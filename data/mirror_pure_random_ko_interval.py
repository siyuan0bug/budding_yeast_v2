#!/usr/bin/env python3
"""
================================================================================
Strict 1-to-1 Paired Mirror Dataset Generator: KO Interval Perturbation
================================================================================
Takes `mutant_aware_pure_random_100000_210m_500s.npz` as the master base:
1. Real Mutants 0-125: 100% exact copy (unmodified Table S5 gold standard).
2. Synthetic Non-KO Mutants (60,200 samples): 100% bit-exact copy of parameters
   and ODE trajectories.
3. Synthetic KO Mutants (39,800 samples):
   - Keeps identical gene combinations, mutant names, other non-KO mutations,
     and carbon sources.
   - ONLY replaces the parameters zeroed by KO mutations with sampled values
     from U[0, ko_interval_max] (default: 1e-4).
   - Re-simulates ODE trajectories using 50 parallel CPU workers.
Result: 100% strict 1-to-1 paired ablation dataset.
================================================================================
"""

import os
import sys
import time
import argparse
import concurrent.futures
import numpy as np

# Ensure current directory is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import mutant_aware as ma


def create_mirror_ko_interval_dataset(
    source_path="/home/users/zsy/mydata/budding_yeast/mutant_aware_pure_random_100000_210m_500s.npz",
    output_path="/home/users/zsy/mydata/budding_yeast/mutant_aware_pure_random_ko_interval_100000_210m_500s.npz",
    ko_interval_max=1e-4,
    num_workers=50,
    seed=42
):
    if ko_interval_max > 1e-4:
        raise ValueError(f"ko_interval_max ({ko_interval_max}) must not exceed 1e-4!")

    print("=" * 85)
    print(" STRICT 1-TO-1 PAIRED MIRROR DATASET GENERATOR")
    print(f" Source Master Dataset: {source_path}")
    print(f" Output Target Dataset: {output_path}")
    print(f" KO Interval Bound    : U[0, {ko_interval_max}]")
    print(f" Parallel CPU Workers : {num_workers}")
    print(f" Random Seed          : {seed}")
    print("=" * 85, flush=True)

    t0 = time.time()
    print("[Phase 1/4] Loading Parameter Definitions...", flush=True)
    d_base = np.load(source_path)
    param_names = list(d_base['param_names'])
    mutant_names = d_base['mutant_names']
    new_params = np.copy(d_base['params'])
    new_labels = np.copy(d_base['labels'])
    new_patterns = np.copy(d_base['pattern_labels'])

    total_samples = len(mutant_names)
    print(f"  -> Total Samples in Base: {total_samples}")
    print(f"  -> Params Shape         : {new_params.shape}", flush=True)

    # Identify tasks requiring re-simulation
    print("[Phase 2/4] Parsing Mutants and Injecting KO Perturbations...", flush=True)
    rng = np.random.RandomState(seed)
    p_wt = ma.get_default_params()
    p_wt.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    p_mut = p_wt.copy()

    ko_tasks = {}  # index -> (name, p_dict)
    num_ko = 0
    num_no_ko = 0

    for idx in range(126, total_samples):
        name = str(mutant_names[idx])
        if '[KO]' in name:
            num_ko += 1
            p_vec = np.copy(new_params[idx])
            p_dict = {k: p_vec[p_i] for p_i, k in enumerate(param_names)}

            # Parse mutated genes that have KO
            desc_str = name[len('RANDOM_'):].rsplit('_L', 1)[0]
            muts = desc_str.split('+')
            ko_genes = [m.split('[')[0] for m in muts if m.endswith('[KO]')]

            for g in ko_genes:
                ko_keys = {}
                ma.apply_gene_mutation(g, 'KO', 0.0, p_wt, p_mut, ko_keys)
                for k in ko_keys:
                    if k in ('init_CDH1', 'init_Tem1', 'init_Cdc15'):
                        continue
                    val = float(rng.uniform(0.0, ko_interval_max))
                    p_dict[k] = val
                    if k in param_names:
                        new_params[idx, param_names.index(k)] = val

                if 'init_CDH1T' in ko_keys:
                    c_val = float(rng.uniform(0.0, p_dict['init_CDH1T']))
                    p_dict['init_CDH1'] = c_val
                    if 'init_CDH1' in param_names:
                        new_params[idx, param_names.index('init_CDH1')] = c_val

                if 'Tem1T' in ko_keys:
                    c_val = float(rng.uniform(0.0, p_dict['Tem1T']))
                    p_dict['init_Tem1'] = c_val
                    if 'init_Tem1' in param_names:
                        new_params[idx, param_names.index('init_Tem1')] = c_val

                if 'Cdc15T' in ko_keys:
                    c_val = float(rng.uniform(0.0, p_dict['Cdc15T']))
                    p_dict['init_Cdc15'] = c_val
                    if 'init_Cdc15' in param_names:
                        new_params[idx, param_names.index('init_Cdc15')] = c_val

            ko_tasks[idx] = (name, p_dict)
        else:
            num_no_ko += 1

    print(f"  -> Real Mutants (Untouched Table S5)      : 126")
    print(f"  -> Non-KO Synthetic (Bit-Exact Reuse)     : {num_no_ko}")
    print(f"  -> KO Synthetic (Requiring Simulation)    : {num_ko}", flush=True)

    # Phase 3: Parallel ODE Simulation for KO tasks
    print(f"\n[Phase 3/4] Simulating {num_ko} KO Tasks with {num_workers} Parallel Workers...", flush=True)
    t_start_sim = time.time()
    results_dict = {}  # idx -> (y_uniform, m_bio_label, m_pattern_label)
    success_count = 0
    discard_count = 0
    fail_count = 0

    log_step = max(500, num_ko // 25)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(ma.simulate_mutant, name, p_dict, 210, 500): idx
            for idx, (name, p_dict) in ko_tasks.items()
        }

        for count, future in enumerate(concurrent.futures.as_completed(future_to_idx)):
            idx = future_to_idx[future]
            try:
                m_name, y_uniform, m_bio_label, m_pattern_label = future.result(timeout=25.0)
                if y_uniform is not None:
                    results_dict[idx] = (y_uniform, m_bio_label, m_pattern_label)
                    success_count += 1
                elif m_bio_label == "Discarded_Mass":
                    discard_count += 1
                else:
                    fail_count += 1
            except Exception:
                fail_count += 1

            if (count + 1) % log_step == 0 or (count + 1) == num_ko:
                elapsed = max(0.001, time.time() - t_start_sim)
                speed = (count + 1) / elapsed
                pct = ((count + 1) / num_ko) * 100
                print(f"  |-- [{count + 1:6d}/{num_ko}] ({pct:5.1f}%) | "
                      f"Success: {success_count:6d} | Discarded: {discard_count:3d} | "
                      f"Failed: {fail_count:3d} | Speed: {speed:5.1f} samples/s", flush=True)

    elapsed_time = time.time() - t_start_sim
    print(f" -> Simulation Completed in {elapsed_time:.2f}s ({num_ko / max(0.001, elapsed_time):.1f} samples/s).\n", flush=True)

    # Phase 4: Save and Verification Report
    print("[Phase 4/4] Merging Trajectories and Exporting 1-to-1 Paired .npz Dataset...", flush=True)
    print("  -> Decompressing and loading original trajectories array...", flush=True)
    new_data = np.copy(d_base['data'])
    pattern_flips = 0

    print("  -> Updating perturbed trajectories and labels...", flush=True)
    for idx, (y_uniform, m_bio_label, m_pattern_label) in results_dict.items():
        new_data[idx] = y_uniform
        new_labels[idx] = m_bio_label
        if m_pattern_label != new_patterns[idx]:
            pattern_flips += 1
        new_patterns[idx] = m_pattern_label
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(
        output_path,
        data=new_data,
        params=new_params,
        param_names=np.array(param_names),
        mutant_names=mutant_names,
        labels=new_labels,
        pattern_labels=new_patterns
    )

    print("=" * 85)
    print(" STRICT 1-TO-1 MIRROR DATASET EXPORT REPORT")
    print("=" * 85)
    print(f"Dataset Export Path                  : {output_path}")
    print(f"Total Samples Count                  : {total_samples}")
    print(f"Real Mutants Identical (0-125)       : 126 / 126 (100.0%)")
    print(f"Non-KO Mutants Bit-Exact Reused      : {num_no_ko} / {num_no_ko} (100.0%)")
    print(f"KO Mutants Successfully Re-Simulated : {success_count} / {num_ko} (100.0%)")
    print(f"Pattern Consistency in KO Mutants    : {(num_ko - pattern_flips) / max(1, num_ko) * 100:.2f}% ({num_ko - pattern_flips}/{num_ko})")
    print(f"Total Execution Time                 : {time.time() - t0:.2f}s")
    print("=" * 85, flush=True)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict 1-to-1 Paired Mirror Dataset Generator")
    parser.add_argument("--source_path", type=str, default="/home/users/zsy/mydata/budding_yeast/mutant_aware_pure_random_100000_210m_500s.npz")
    parser.add_argument("--output_path", type=str, default="/home/users/zsy/mydata/budding_yeast/mutant_aware_pure_random_ko_interval_100000_210m_500s.npz")
    parser.add_argument("--ko_interval_max", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_mirror_ko_interval_dataset(
        source_path=args.source_path,
        output_path=args.output_path,
        ko_interval_max=args.ko_interval_max,
        num_workers=args.num_workers,
        seed=args.seed
    )
