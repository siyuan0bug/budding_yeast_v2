"""
Random-only MASS Sampler & Complete Dataset Generator (无定向 Motif 纯随机版)
================================================================================
本模块提供无任何生物学定向基序 (No Rescue Motif) 的纯随机分层采样与全量数据集生成功能:

1. 核心单样本生成器函数 (供 al_callback.py 等外部模块直接 import):
   - sample_mass_random(level=1)
   - 保证外部接口与返回值签名 (ovr_dict, desc_str) 100% 不变。

2. 全量数据集生成与 .npz 文件导出管道 (支持 CLI 直接运行):
   - generate_mass_random_dataset(n_synthetic=100000, num_workers=60)
   - 包含 Phase 1 (126 真实测试突变体加载) -> Phase 2 (纯随机多阶采样与去重) 
     -> Phase 3 (并行 BDF 刚性 ODE 求解) -> Phase 4 (.npz 格式打包与质量审计)。
   - 生成格式严格对齐 lhs_v1_origin.py / mass_dataset_generator.py 标准。

Level 分层与突变基因数 ki (与原版完全对齐):
  - L1 (单突变): ki = 1       (目标占比: 25%)
  - L2 (双突变): ki = 2       (目标占比: 35%)
  - L3 (多突变): ki = 3 ~ 5   (目标占比: 25%)
  - L4 (高阶突变): ki = 4 ~ 6 (目标占比: 15%)
"""

import os
import sys
import math
import time
import concurrent.futures
import numpy as np

# Ensure current directory and workspace root are in sys.path when script is executed directly
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
for _p in [_CURRENT_DIR, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 从 mass_dataset_generator 共享核心常数、机制先验与物理仿真引擎
try:
    from mass_dataset_generator import (
        REAL_MUTANTS,
        get_default_params,
        BINDING_GROUPS,
        sample_scaling_factor,
        apply_gene_mutation,
        simulate_mutant,
        is_duplicate_real_mutant,
    )
except ImportError:
    from budding_yeast_v2.data.mass_dataset_generator import (
        REAL_MUTANTS,
        get_default_params,
        BINDING_GROUPS,
        sample_scaling_factor,
        apply_gene_mutation,
        simulate_mutant,
        is_duplicate_real_mutant,
    )

# 默认导出目录
if os.name == 'nt':
    DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    DEFAULT_OUTPUT_DIR = "/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast"

DEFAULT_DATASET_NAME = "mass_random_dataset_210min_500steps_dual_labels.npz"


def sample_mass_random(level=1):
    """
    Generate a synthetic mutant sample by PURE RANDOM exploratory sampling
    (no rescue motif / directed sampling), keeping the original Level tiers.

    Level -> number of mutated binding groups (ki), identical to the standard
    exploratory branch of mass_dataset_generator.sample_mass_mutant:
      L1: ki=1 | L2: ki=2 | L3: ki=3~5 | L4: ki=4~6

    Returns:
        (ovr_dict, mutant_description_str) — 与 sample_mass_mutant 返回格式严格一致
    """
    p_wt = get_default_params()
    p_mut = p_wt.copy()
    ovr = {}

    has_gal_oe = False
    mut_descriptions = []

    # Standard unconstrained exploratory sampling (唯一分支, 彻底移除 rescue motif)
    if level == 1: ki = 1
    elif level == 2: ki = 2
    elif level == 3: ki = np.random.randint(3, 6)   # ki in {3, 4, 5}
    else: ki = np.random.randint(4, 7)              # ki in {4, 5, 6}

    group_keys = list(BINDING_GROUPS.keys())
    group_weights = [BINDING_GROUPS[k]['P_bio_G'] for k in group_keys]
    selected_g_keys = np.random.choice(
        group_keys, size=min(ki, len(group_keys)), replace=False, p=group_weights
    )

    for g_key in selected_g_keys:
        g_info = BINDING_GROUPS[g_key]
        g_name = g_info['name']

        modes = list(g_info['modes'].keys())
        m_weights = [g_info['modes'][m] for m in modes]
        m_weights = np.array(m_weights) / sum(m_weights)
        m_chosen = np.random.choice(modes, p=m_weights)

        comp_mode = None
        if g_name in ['CLB2', 'CLB5', 'PDS1', 'SIC1', 'CDH1']:
            synth_modes = [m for m in ['GAL-OE', 'MC'] if m in modes]
            struc_modes = [m for m in ['db', 'PM', 'CA'] if m in modes]
            if m_chosen in synth_modes and struc_modes and np.random.rand() < 0.25:
                comp_mode = np.random.choice(struc_modes)
            elif m_chosen in struc_modes and synth_modes and np.random.rand() < 0.25:
                comp_mode = np.random.choice(synth_modes)

        if comp_mode:
            mut_descriptions.append(f"{g_name}[{m_chosen}+{comp_mode}]")
            if m_chosen == 'GAL-OE' or comp_mode == 'GAL-OE':
                has_gal_oe = True

            alpha1 = sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            apply_gene_mutation(g_name, m_chosen, alpha1, p_wt, p_mut, ovr)
            alpha2 = sample_scaling_factor(comp_mode, g_name) if comp_mode in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            apply_gene_mutation(g_name, comp_mode, alpha2, p_wt, p_mut, ovr)
        else:
            mut_descriptions.append(f"{g_name}[{m_chosen}]")
            if m_chosen == 'GAL-OE':
                has_gal_oe = True

            alpha = sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)

    # Environmental Carbon Source Decoupling (与原版保持一致)
    if has_gal_oe:
        ovr['mdt'] = 150.0
    else:
        if 'mdt' not in ovr:
            carbon_choice = np.random.choice(['Glc', 'Gal', 'Raff'], p=[0.75, 0.18, 0.07])
            if carbon_choice == 'Gal':
                ovr['mdt'] = 150.0
            elif carbon_choice == 'Raff':
                ovr['mdt'] = 160.0
            else:
                ovr['mdt'] = 90.0

    desc_str = "+".join(mut_descriptions)
    return ovr, desc_str


def generate_mass_random_dataset(n_synthetic=100000, max_attempts=None, num_workers=60,
                                  output_name=DEFAULT_DATASET_NAME, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Generate complete PURE RANDOM dataset (126 Real Mutants + n_synthetic Random MASS Mutants)
    and export into a single .npz file with exact field alignment to mass_dataset_generator.py.
    """
    print("=" * 85)
    print("MASS PURE RANDOM Dataset Generator Protocol (No-Motif Exploratory Sampling)")
    print("=" * 85)

    if max_attempts is None:
        max_attempts = max(50000, n_synthetic * 10)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass

    dataset_path = os.path.join(output_dir, output_name)

    defaults = get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    param_keys = sorted([k for k in defaults.keys() if k != 'is_mutant_104'])

    all_results = {}
    all_tasks = {}
    bio_labels_dict = {}
    pattern_labels_dict = {}
    ordered_names = []

    # Phase 1: Prepare 126 Real Mutants (Ground Truth Benchmark)
    print(f"\n[Phase 1/4] Preparing D_real (126 Historical Real Mutants from Appendix B)...")
    for mid, (mname, movr) in REAL_MUTANTS.items():
        sample_name = f"{mid}_{mname}".replace(" ", "_")
        ordered_names.append(sample_name)
        all_tasks[sample_name] = movr
    print(f" -> Loaded {len(REAL_MUTANTS)} Real Mutants.")

    # Phase 2: Sample Pure Random Synthetic Mutants (No Motif)
    print(f"\n[Phase 2/4] Sampling {n_synthetic} Pure Random Mutants (No Rescue Motifs)...")
    n_l1 = int(math.floor(0.25 * n_synthetic))
    n_l2 = int(math.floor(0.35 * n_synthetic))
    n_l3 = int(math.floor(0.25 * n_synthetic))
    n_l4 = n_synthetic - n_l1 - n_l2 - n_l3

    print(f" -> Target Level Distribution: L1 (ki=1): {n_l1} | L2 (ki=2): {n_l2} | L3 (ki=3~5): {n_l3} | L4 (ki=4~6): {n_l4}")

    level_assignments = [1]*n_l1 + [2]*n_l2 + [3]*n_l3 + [4]*n_l4
    np.random.shuffle(level_assignments)

    attempts = 0
    synth_success = 0
    duplicates_blocked = 0

    while synth_success < n_synthetic and attempts < max_attempts:
        attempts += 1
        level = level_assignments[synth_success]
        ovr, desc_str = sample_mass_random(level=level)

        # 136-dim Parameter Deduplication Check against D_real (Strict Zero Leakage)
        if is_duplicate_real_mutant(ovr, defaults):
            duplicates_blocked += 1
            continue

        synth_success += 1
        sample_name = f"MASS_{desc_str}_L{level}_{synth_success}".replace(" ", "_")
        ordered_names.append(sample_name)
        all_tasks[sample_name] = ovr

    print(f" -> Sampling Done: {synth_success} synthetic tasks generated in {attempts} attempts.")
    print(f"    - Filter [D_real Deduplication]: {duplicates_blocked} candidates blocked (duplicate with real mutants).")
    print(f" -> Total Simulation Queue: {len(ordered_names)} tasks ({len(REAL_MUTANTS)} Real + {synth_success} Synthetic).")

    # Phase 3: Parallel ODE Simulation Engine
    if num_workers is None:
        num_workers = min(60, os.cpu_count() or 4)

    print(f"\n[Phase 3/4] Simulating {len(ordered_names)} Mutant Trajectories on {num_workers} CPU Workers...")
    t_start = time.time()
    sim_counts = {
        'real_success': 0, 'synth_success': 0,
        'discarded_real': 0, 'discarded_synth': 0,
        'failed_real': 0, 'failed_synth': 0
    }

    log_step = max(50, len(ordered_names) // 40)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_name = {
            executor.submit(simulate_mutant, name, all_tasks[name], 210, 500): name
            for name in ordered_names
        }
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_name)):
            name = future_to_name[future]
            is_synth = "_L" in name
            try:
                m_name, y_uniform, m_bio_label, m_pattern_label = future.result(timeout=20.0)
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
                elapsed = max(0.001, time.time() - t_start)
                speed = (idx + 1) / elapsed
                valid_total = sim_counts['real_success'] + sim_counts['synth_success']
                disc_total = sim_counts['discarded_real'] + sim_counts['discarded_synth']
                fail_total = sim_counts['failed_real'] + sim_counts['failed_synth']
                pct = ((idx + 1) / len(ordered_names)) * 100
                print(f"  |-- [{idx + 1:5d}/{len(ordered_names)}] ({pct:5.1f}%) | Valid: {valid_total:5d} (Real:{sim_counts['real_success']:3d}, Synth:{sim_counts['synth_success']:5d}) | Discarded(Mass>10): {disc_total:4d} | Failed: {fail_total:3d} | Speed: {speed:5.1f} samples/s", flush=True)

    elapsed_time = time.time() - t_start
    print(f" -> Simulation Completed in {elapsed_time:.2f}s ({len(ordered_names) / max(0.001, elapsed_time):.1f} samples/s).", flush=True)

    # Phase 4: Packaging & .npz Export
    print(f"\n[Phase 4/4] Packaging & Saving Dataset Array Files...", flush=True)
    ordered_tensors = []
    final_names = []
    final_params = []
    final_bio_labels = []
    final_pattern_labels = []

    real_final_names = []
    synth_final_names = []
    real_bio_labels = []
    synth_bio_labels = []
    real_pattern_labels = []
    synth_pattern_labels = []
    synth_levels = []

    for name in ordered_names:
        if name in all_results:
            ordered_tensors.append(all_results[name])
            final_names.append(name)
            final_bio_labels.append(bio_labels_dict[name])
            final_pattern_labels.append(pattern_labels_dict[name])

            if "_L" in name:
                synth_final_names.append(name)
                synth_bio_labels.append(bio_labels_dict[name])
                synth_pattern_labels.append(pattern_labels_dict[name])
                try:
                    lvl = int(name.split("_L")[1].split("_")[0])
                    synth_levels.append(lvl)
                except Exception:
                    synth_levels.append(-1)
            else:
                real_final_names.append(name)
                real_bio_labels.append(bio_labels_dict[name])
                real_pattern_labels.append(pattern_labels_dict[name])

            overrides = all_tasks[name]
            p_vec = [overrides.get(k, defaults[k]) for k in param_keys]
            final_params.append(p_vec)

    all_tensors = np.array(ordered_tensors)
    all_params = np.array(final_params)
    all_bio_labels = np.array(final_bio_labels)
    all_pattern_labels = np.array(final_pattern_labels)

    os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)
    np.savez_compressed(dataset_path,
                        data=all_tensors,
                        params=all_params,
                        param_names=param_keys,
                        mutant_names=final_names,
                        labels=all_bio_labels,
                        pattern_labels=all_pattern_labels)

    total_valid = len(final_names)
    total_disc = sim_counts['discarded_real'] + sim_counts['discarded_synth']
    total_fail = sim_counts['failed_real'] + sim_counts['failed_synth']

    print("=" * 85, flush=True)
    print(f" MASS PURE RANDOM DATASET GENERATION SUMMARY REPORT", flush=True)
    print("=" * 85, flush=True)
    print(f"Dataset Export Path               : {dataset_path}", flush=True)
    print(f"Data Tensor Shape (N, Vars, Steps): {all_tensors.shape}", flush=True)
    print(f"Params Matrix Shape (N, P_dim)    : {all_params.shape}", flush=True)
    print(f"Total Parameter Keys Count        : {len(param_keys)}", flush=True)
    print(f"Total Tasks Simulated             : {len(ordered_names)}", flush=True)
    print(f"Total Validated Samples Saved     : {total_valid} ({total_valid / len(ordered_names) * 100:.2f}% yield)", flush=True)
    print("-" * 85, flush=True)

    # Quality Audit & Sample Composition Breakdown
    print(f"\n[Quality Audit]")
    print(f"  * D_real Deduplication Blocked : {duplicates_blocked:5d} candidates")
    print(f"  * Discarded (Mass > 10.0)      : {total_disc:5d} tasks")
    print(f"  * Failed Solvers               : {total_fail:5d} tasks")
    print(f"  * Valid Saved Real Mutants     : {len(real_final_names):5d} / {len(REAL_MUTANTS)}")
    print(f"  * Valid Saved Synthetic Samples: {len(synth_final_names):5d} / {synth_success}")

    if synth_levels:
        u_lvl, c_lvl = np.unique(synth_levels, return_counts=True)
        for l_idx, count in zip(u_lvl, c_lvl):
            target_n = [n_l1, n_l2, n_l3, n_l4][l_idx-1] if 1 <= l_idx <= 4 else -1
            print(f"    - Level {l_idx} Random Mutants       : {count:5d} / {target_n} ({count/target_n*100:.1f}%)")

    print("\n" + "=" * 85, flush=True)
    print("Sample Mutant Names Demonstration:")
    for name in real_final_names[:3]:
        print(f"  - [Real]  {name}", flush=True)
    for name in synth_final_names[:5]:
        print(f"  - [Synth] {name}", flush=True)
    print("=" * 85 + "\n", flush=True)

    return dataset_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="MASS Pure Random Dataset Generator (No-Motif)")
    parser.add_argument("--samples", type=int, default=100000, help="Number of synthetic samples to generate")
    parser.add_argument("--workers", type=int, default=60, help="Number of parallel CPU worker processes")
    parser.add_argument("--output_name", type=str, default=DEFAULT_DATASET_NAME, help="Export dataset .npz filename")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Export directory path")
    args = parser.parse_args()

    print(f"Executing MASS Pure Random Dataset Generator (Samples: {args.samples}, Workers: {args.workers})...", flush=True)
    generate_mass_random_dataset(
        n_synthetic=args.samples,
        num_workers=args.workers,
        output_name=args.output_name,
        output_dir=args.output_dir
    )
    print("\n[SUCCESS] MASS Random Dataset Generator Execution Finished Cleanly!", flush=True)
