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


def generate_mass_random_dataset(n_synthetic=100000, level_ratios=[0.25, 0.35, 0.25, 0.15],
                                  max_attempts=None, num_workers=60,
                                  output_name=None, output_dir=None):
    """
    Generate complete PURE RANDOM dataset (126 Real Mutants + n_synthetic Random MASS Mutants)
    and export into a single .npz file with exact field alignment to mass_dataset_generator.py.
    """
    if output_name is None:
        output_name = f"mass_{n_synthetic}_random_dataset_210min_500steps_dual_labels.npz"

    if output_dir is None:
        if os.name == 'nt':
            output_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            output_dir = "/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast"

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass

    dataset_path = os.path.join(output_dir, output_name)

    print("=" * 85)
    print("MASS PURE RANDOM Dataset Generator Protocol (No-Motif Exploratory Sampling)")
    print(f"Target Output: {dataset_path}")
    print("=" * 85)

    if max_attempts is None:
        max_attempts = max(50000, n_synthetic * 10)

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
    r1, r2, r3, r4 = level_ratios
    total_r = r1 + r2 + r3 + r4
    r1, r2, r3, r4 = r1 / total_r, r2 / total_r, r3 / total_r, r4 / total_r

    n_l1 = int(math.floor(r1 * n_synthetic))
    n_l2 = int(math.floor(r2 * n_synthetic))
    n_l3 = int(math.floor(r3 * n_synthetic))
    n_l4 = n_synthetic - n_l1 - n_l2 - n_l3

    print(f" -> Target Level Distribution: L1 (ki=1): {n_l1} ({r1*100:.1f}%) | L2 (ki=2): {n_l2} ({r2*100:.1f}%) | L3 (ki=3~5): {n_l3} ({r3*100:.1f}%) | L4 (ki=4~6): {n_l4} ({r4*100:.1f}%)")

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

    # 1. Filtering & Quality Control Audit
    print("\n[1. Filtering & Quality Control Audit]", flush=True)
    print(f"  * Filter 1 [D_real Deduplication Blocked] : {duplicates_blocked:5d} synthetic candidates")
    print(f"  * Filter 2 [Mass Divergence (Mass > 10)]  : {total_disc:5d} tasks (Real: {sim_counts['discarded_real']}, Synth: {sim_counts['discarded_synth']})")
    print(f"  * Filter 3 [ODE Solver Failures]          : {total_fail:5d} tasks (Real: {sim_counts['failed_real']}, Synth: {sim_counts['failed_synth']})")
    print(f"  * Valid Sample Yield Rate                 : {total_valid:5d} / {len(ordered_names)} ({total_valid / len(ordered_names) * 100:.1f}%)")

    # 2. Sample Composition Breakdown
    print("\n[2. Sample Composition Breakdown]", flush=True)
    print(f"  * Real Mutants (Appendix B Table S5)      : {len(real_final_names):5d} / {len(REAL_MUTANTS)} ({len(real_final_names)/len(REAL_MUTANTS)*100:.1f}%)")
    print(f"  * Synthetic Mutants (MASS Random Multi-Level): {len(synth_final_names):5d} / {synth_success} ({len(synth_final_names)/synth_success*100:.1f}%)")
    if synth_levels:
        u_lvl, c_lvl = np.unique(synth_levels, return_counts=True)
        for l_idx, count in zip(u_lvl, c_lvl):
            target_n = [n_l1, n_l2, n_l3, n_l4][l_idx-1] if 1 <= l_idx <= 4 else -1
            print(f"    - Level {l_idx} Random Mutants         : {count:5d} / {target_n} ({count/target_n*100:.1f}%)")

    # 3. Mass Growth Pattern Distribution (Real vs Synth vs Total)
    print("\n[3. Mass Growth Pattern Labels Distribution (Real vs Synth)]", flush=True)
    pattern_categories = [
        "Pattern_A: Multi-Division (>=2)",
        "Pattern_B: Single-Division (==1)",
        "Pattern_C: Exponential-Arrest (==0)"
    ]
    header = f"  {'Pattern Label':<36} | {'Real Mutants':<15} | {'Synthetic Mutants':<18} | {'Overall Dataset':<18}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for pat in pattern_categories:
        c_r = real_pattern_labels.count(pat)
        pct_r = (c_r / max(1, len(real_pattern_labels))) * 100
        c_s = synth_pattern_labels.count(pat)
        pct_s = (c_s / max(1, len(synth_pattern_labels))) * 100
        c_t = list(all_pattern_labels).count(pat)
        pct_t = (c_t / max(1, len(all_pattern_labels))) * 100
        print(f"  {pat:<36} | {c_r:5d} ({pct_r:5.1f}%)   | {c_s:7d} ({pct_s:5.1f}%)    | {c_t:7d} ({pct_t:5.1f}%)")

    # 4. Biological Viability Labels Distribution (Real vs Synth vs Total)
    print("\n[4. Biological Viability Labels Distribution (Real vs Synth)]", flush=True)
    all_bio_unique = sorted(list(set(all_bio_labels)))
    v_header = f"  {'Viability Label':<32} | {'Real Mutants':<15} | {'Synthetic Mutants':<18} | {'Overall Dataset':<18}"
    print(v_header)
    print("  " + "-" * (len(v_header) - 2))
    for bio in all_bio_unique:
        c_r = real_bio_labels.count(bio)
        pct_r = (c_r / max(1, len(real_bio_labels))) * 100
        c_s = synth_bio_labels.count(bio)
        pct_s = (c_s / max(1, len(synth_bio_labels))) * 100
        c_t = list(all_bio_labels).count(bio)
        pct_t = (c_t / max(1, len(all_bio_labels))) * 100
        print(f"  {bio:<32} | {c_r:5d} ({pct_r:5.1f}%)   | {c_s:7d} ({pct_s:5.1f}%)    | {c_t:7d} ({pct_t:5.1f}%)")

    print("\n" + "=" * 85, flush=True)
    print("Sample Mutant Names Demonstration:", flush=True)
    for name in real_final_names[:3]:
        print(f"  - [Real]  {name}", flush=True)
    for name in synth_final_names[:5]:
        print(f"  - [Synth] {name}", flush=True)
    print("=" * 85 + "\n", flush=True)

    return dataset_path


if __name__ == '__main__':
    # =========================================================================
    # 用户可调参数配置区 (随时在此处直接修改，或通过命令行参数指定)
    # =========================================================================
    NUM_SAMPLES = 100000                           # 生成的采样样本数量
    NUM_WORKERS = 60                              # 并行调用的 CPU 核心数 (当前服务器共有 64 核，设为 60 进行高并发并行加速)
    LEVEL_RATIOS = [0.25, 0.35, 0.25, 0.15]       # Level 1~4 目标采样占比 (L1:单突变25%, L2:双突变35%, L3:多突变25%, L4:高阶突变15%)
    
    # 导出的数据集文件名 (默认自动携带样本数，也可直接在此自定义修改)
    DATASET_NAME = f"mass_{NUM_SAMPLES}_random_dataset_210min_500steps_dual_labels.npz"

    # 导出的数据集目录路径 (自动适配系统环境，也可直接在此自定义写死绝对路径)
    if os.name == 'nt':
        OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    else:
        OUTPUT_DIR = "/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast"
    # =========================================================================

    import argparse
    parser = argparse.ArgumentParser(description="MASS Pure Random Dataset Generator (No-Motif)")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES, help="Number of synthetic samples to generate")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel CPU worker processes")
    parser.add_argument("--level_ratios", type=float, nargs=4, default=LEVEL_RATIOS, help="Proportions for Level 1, 2, 3, 4 tiers")
    parser.add_argument("--output_name", type=str, default=None, help="Export dataset .npz filename (default: auto with sample count)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Export directory path")
    args = parser.parse_args()

    # 动态确定输出文件名：如果命令行显式指定了 --output_name 则使用指定的；否则自动使用基于 args.samples 的文件名
    if args.output_name is not None:
        final_output_name = args.output_name
    else:
        if args.samples == NUM_SAMPLES:
            final_output_name = DATASET_NAME
        else:
            final_output_name = f"mass_{args.samples}_random_dataset_210min_500steps_dual_labels.npz"

    print(f"Executing MASS Pure Random Dataset Generator (Samples: {args.samples}, Workers: {args.workers}, Output: {final_output_name})...", flush=True)
    generate_mass_random_dataset(
        n_synthetic=args.samples,
        level_ratios=args.level_ratios,
        num_workers=args.workers,
        output_name=final_output_name,
        output_dir=args.output_dir
    )
    print("\n[SUCCESS] MASS Random Dataset Generator Execution Finished Cleanly!", flush=True)
