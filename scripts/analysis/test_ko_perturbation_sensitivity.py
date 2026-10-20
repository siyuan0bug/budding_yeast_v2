#!/usr/bin/env python3
"""
test_ko_perturbation_sensitivity.py
====================================
Rigorous Experimental Audit:
Investigating the Quantitative and Qualitative Effects of Replacing Exact KO (0.0)
with Small Interval Sampling [0, epsilon] on the 108 KO Budding Yeast Real Mutants.

Author: Antigravity AI Agent & Research Team
Date: 2026-09-19
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Ensure project paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PARENT_DIR = os.path.dirname(PROJECT_ROOT)

for p in [PROJECT_ROOT, PARENT_DIR, DATA_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import mutant_aware module WITHOUT MODIFICATION
from budding_yeast_v2.data.systems.budding import api as ma

CORE_9_INDICES = [ma.nMASS, ma.nCLN2, ma.nCLB2, ma.nSIC1, ma.nC5, ma.nF2, ma.nF5, ma.nSIC1P, ma.nC2P]
VAR_NAMES_39 = [
    'MASS', 'CLN2', 'CLB2', 'CLB5', 'SIC1', 'CDC6', 'C2', 'C5', 'F2', 'F5',
    'SIC1P', 'C2P', 'C5P', 'CDC6P', 'F2P', 'F5P', 'SWI5T', 'SWI5', 'IEP', 'CDC20T',
    'CDC20', 'CDH1T', 'CDH1', 'CDC14T', 'CDC14', 'NET1T', 'NET1', 'RENT', 'TEM1',
    'CDC15', 'PPX', 'PDS1', 'ESP1', 'ORI', 'BUD', 'SPN', 'Vi20', 'lte1', 'BUB2'
]

EPSILON_LEVELS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05]
MONTE_CARLO_INTERVALS = [1e-4, 1e-3, 1e-2]  # Uniform [0, eps] random draws


def evaluate_single_mutant(mutant_item):
    """
    Evaluates ground truth trajectory (0.0) vs multiple perturbation levels
    for a single real mutant.
    """
    m_id, (m_name, base_ovr) = mutant_item
    
    # Identify zeroed parameters
    zero_keys = [k for k, v in base_ovr.items() if v == 0.0]
    if not zero_keys:
        return None

    # 1. Ground Truth Simulation (Exact 0.0)
    _, y_true, bio_true, pat_true = ma.simulate_mutant(m_name, base_ovr)
    if y_true is None:
        return {
            'mutant_id': m_id,
            'mutant_name': m_name,
            'base_status': 'FAILED_AT_BASE',
            'zero_keys': zero_keys,
            'results': {}
        }

    results = {}

    # 2. Deterministic Absolute Perturbations
    for eps in EPSILON_LEVELS:
        p_pert = dict(base_ovr)
        for k in zero_keys:
            p_pert[k] = eps
        
        _, y_pert, bio_p, pat_p = ma.simulate_mutant(m_name, p_pert)
        if y_pert is None:
            results[f'abs_eps_{eps}'] = {
                'eps': eps,
                'type': 'absolute',
                'status': 'SIM_FAILED',
                'mae': np.nan,
                'rel_l2': np.nan,
                'max_diff': np.nan,
                'core9_mae': np.nan,
                'pattern': 'None',
                'pattern_match': False,
                'bio_label': 'None',
                'bio_match': False
            }
        else:
            diff = np.abs(y_pert - y_true)
            mae = float(np.mean(diff))
            norm_true = np.linalg.norm(y_true)
            rel_l2 = float(np.linalg.norm(y_pert - y_true) / (norm_true + 1e-8))
            max_diff = float(np.max(diff))
            core9_mae = float(np.mean(diff[CORE_9_INDICES, :]))
            pat_match = bool(pat_p == pat_true)
            bio_match = bool(bio_p == bio_true)

            results[f'abs_eps_{eps}'] = {
                'eps': eps,
                'type': 'absolute',
                'status': 'SUCCESS',
                'mae': mae,
                'rel_l2': rel_l2,
                'max_diff': max_diff,
                'core9_mae': core9_mae,
                'pattern': pat_p,
                'pattern_match': pat_match,
                'bio_label': bio_p,
                'bio_match': bio_match
            }

    # 3. Stochastic Uniform Interval Sampling U[0, eps] (3 Monte Carlo draws)
    rng = np.random.RandomState(42 + int(float(m_id.replace('2.1', '21').replace('2.2', '22'))))
    for eps_bound in MONTE_CARLO_INTERVALS:
        mc_maes = []
        mc_rel_l2s = []
        mc_pat_matches = []
        
        for _ in range(3):
            p_mc = dict(base_ovr)
            for k in zero_keys:
                p_mc[k] = float(rng.uniform(0.0, eps_bound))
            _, y_mc, bio_mc, pat_mc = ma.simulate_mutant(m_name, p_mc)
            if y_mc is not None:
                mc_maes.append(float(np.mean(np.abs(y_mc - y_true))))
                mc_rel_l2s.append(float(np.linalg.norm(y_mc - y_true) / (np.linalg.norm(y_true) + 1e-8)))
                mc_pat_matches.append(pat_mc == pat_true)
            else:
                mc_pat_matches.append(False)

        results[f'mc_interval_{eps_bound}'] = {
            'eps_bound': eps_bound,
            'type': 'monte_carlo_uniform',
            'mean_mae': float(np.mean(mc_maes)) if mc_maes else np.nan,
            'mean_rel_l2': float(np.mean(mc_rel_l2s)) if mc_rel_l2s else np.nan,
            'pattern_concordance': float(np.mean(mc_pat_matches)) if mc_pat_matches else 0.0
        }

    return {
        'mutant_id': m_id,
        'mutant_name': m_name,
        'base_status': 'SUCCESS',
        'base_pattern': pat_true,
        'base_bio_label': bio_true,
        'zero_keys': zero_keys,
        'results': results
    }


def plot_representative_mutants(out_dir):
    """
    Generates high-resolution comparative time-series trajectory plots
    for key representative mutants:
    - #3: cln1Δ cln2Δ (G1 cyclin double KO)
    - #13: cln3Δ bck2Δ (Synthetic lethality G1 arrest)
    - #36: sic1Δ cdh1Δ (Viable double exit KO)
    - #75: cdc20Δ pds1Δ (Spindle checkpoint bypass)
    - #124: APC-A GAL-CLB2 (Mitotic arrest with overexpression)
    """
    plot_targets = [
        ('3', '3_cln1Δ_cln2Δ'),
        ('13', '13_cln3Δ_bck2Δ'),
        ('36', '36_sic1Δ_cdh1Δ'),
        ('75', '75_cdc20Δ_pds1Δ'),
        ('124', '124_APC-A_GAL-CLB2')
    ]
    
    key_vars = [
        (ma.nMASS, 'MASS (Cell Size)'),
        (ma.nCLN2, 'CLN2 (G1 Cyclin)'),
        (ma.nCLB2, 'CLB2 (Mitotic Cyclin)'),
        (ma.nSIC1, 'SIC1 (CKI Inhibitor)'),
        (ma.nCDH1, 'CDH1 (APC Activator)'),
        (ma.nBUD, 'BUD (Budding Index)')
    ]

    t_grid = np.linspace(0, 210, 500)
    plot_eps_list = [0.0, 1e-4, 1e-2, 0.05]
    colors = {0.0: '#1f77b4', 1e-4: '#2ca02c', 1e-2: '#ff7f0e', 0.05: '#d62728'}
    linestyles = {0.0: '-', 1e-4: ':', 1e-2: '--', 0.05: '-.'}

    os.makedirs(out_dir, exist_ok=True)

    for mid, mtitle in plot_targets:
        if mid not in ma.REAL_MUTANTS:
            continue
        m_name, base_ovr = ma.REAL_MUTANTS[mid]
        zero_keys = [k for k, v in base_ovr.items() if v == 0.0]
        
        sim_curves = {}
        patterns = {}
        for eps in plot_eps_list:
            p_curr = dict(base_ovr)
            if eps > 0:
                for k in zero_keys:
                    p_curr[k] = eps
            _, y_sol, _, pat_sol = ma.simulate_mutant(m_name, p_curr)
            sim_curves[eps] = y_sol
            patterns[eps] = pat_sol

        # Plot 2x3 grid
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
        fig.suptitle(f"KO Perturbation Trajectory Audit: Mutant #{mid} ({m_name})\nZeroed Params: {zero_keys}", fontsize=14, fontweight='bold')
        
        for ax_idx, (v_idx, v_name) in enumerate(key_vars):
            ax = axes[ax_idx // 3, ax_idx % 3]
            for eps in plot_eps_list:
                y_traj = sim_curves[eps]
                if y_traj is not None:
                    lbl = f"eps={eps} ({patterns[eps].split(':')[0]})" if ax_idx == 0 else None
                    ax.plot(t_grid, y_traj[v_idx], label=lbl, color=colors[eps], linestyle=linestyles[eps], lw=2.0 if eps == 0.0 else 1.8)
            ax.set_title(v_name, fontsize=12, fontweight='bold')
            ax.set_ylabel("Concentration / State", fontsize=10)
            ax.grid(True, alpha=0.3)
            if ax_idx >= 3:
                ax.set_xlabel("Time (min)", fontsize=10)
        
        fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11, framealpha=0.9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        save_path = os.path.join(out_dir, f"mutant_{mid}_{mtitle.replace('/', '_')}_sensitivity.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"  -> Plot saved: {save_path}")


def main():
    print("=" * 85)
    print(" KO PERTURBATION SENSITIVITY AUDIT ENGINE (Chen 2004 Model)")
    print("=" * 85)
    t0 = time.time()

    # Step 1: Collect all KO real mutants
    ko_candidates = []
    for m_id, (m_name, p_dict) in ma.REAL_MUTANTS.items():
        zero_keys = [k for k, v in p_dict.items() if v == 0.0]
        if zero_keys or 'Δ' in m_name:
            ko_candidates.append((m_id, (m_name, p_dict)))

    print(f"Total Real Mutants in Table S5 : {len(ma.REAL_MUTANTS)}")
    print(f"Real Mutants with KO (0.0)     : {len(ko_candidates)}")
    print(f"Testing Perturbation Levels    : {EPSILON_LEVELS}")
    print(f"Testing Stochastic Intervals   : U[0, eps] for {MONTE_CARLO_INTERVALS}")
    print("-" * 85)

    # Step 2: Parallel simulation
    num_workers = min(50, os.cpu_count() or 4)
    print(f"Running simulation with {num_workers} parallel workers...", flush=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        eval_records = list(executor.map(evaluate_single_mutant, ko_candidates))

    eval_records = [r for r in eval_records if r is not None and r['base_status'] == 'SUCCESS']
    print(f"Successfully simulated and audited {len(eval_records)} KO mutants in {time.time() - t0:.2f}s.\n")

    # Step 3: Statistical Analysis & Aggregation
    out_dir = os.path.join(PROJECT_ROOT, "scripts", "analysis", "outputs", "ko_sensitivity_analysis")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for rec in eval_records:
        mid = rec['mutant_id']
        mname = rec['mutant_name']
        b_pat = rec['base_pattern']
        b_bio = rec['base_bio_label']
        z_keys = "|".join(rec['zero_keys'])
        
        row = {
            'Mutant_ID': mid,
            'Mutant_Name': mname,
            'Zero_Params': z_keys,
            'Num_Zero_Params': len(rec['zero_keys']),
            'Base_Pattern': b_pat,
            'Base_Bio': b_bio
        }
        
        for eps in EPSILON_LEVELS:
            res = rec['results'].get(f'abs_eps_{eps}', {})
            row[f'MAE_eps_{eps}'] = res.get('mae', np.nan)
            row[f'RelL2_eps_{eps}'] = res.get('rel_l2', np.nan)
            row[f'MaxDiff_eps_{eps}'] = res.get('max_diff', np.nan)
            row[f'Core9MAE_eps_{eps}'] = res.get('core9_mae', np.nan)
            row[f'PatMatch_eps_{eps}'] = res.get('pattern_match', False)
            row[f'Pat_eps_{eps}'] = res.get('pattern', 'None')

        for eps_bound in MONTE_CARLO_INTERVALS:
            mc_res = rec['results'].get(f'mc_interval_{eps_bound}', {})
            row[f'MC_MAE_eps_{eps_bound}'] = mc_res.get('mean_mae', np.nan)
            row[f'MC_RelL2_eps_{eps_bound}'] = mc_res.get('mean_rel_l2', np.nan)
            row[f'MC_PatConc_eps_{eps_bound}'] = mc_res.get('pattern_concordance', 0.0)

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "per_mutant_ko_sensitivity.csv")
    df.to_csv(csv_path, index=False)
    print(f"Per-mutant CSV report saved to: {csv_path}")

    # Summary Statistics across all 108 mutants
    summary = {
        'total_audited_ko_mutants': len(df),
        'eps_level_metrics': {}
    }

    print("\n" + "=" * 85)
    print(f"{'Epsilon Level':<15} | {'Mean MAE':<10} | {'Median RelL2':<12} | {'Max RelL2':<10} | {'Pattern Flip Count':<20} | {'Flip %':<8}")
    print("-" * 85)

    for eps in EPSILON_LEVELS:
        mean_mae = float(df[f'MAE_eps_{eps}'].mean())
        med_l2 = float(df[f'RelL2_eps_{eps}'].median())
        max_l2 = float(df[f'RelL2_eps_{eps}'].max())
        flips = int((~df[f'PatMatch_eps_{eps}']).sum())
        flip_pct = float(flips / len(df) * 100)

        summary['eps_level_metrics'][str(eps)] = {
            'mean_mae': mean_mae,
            'median_rel_l2': med_l2,
            'max_rel_l2': max_l2,
            'pattern_flips': flips,
            'pattern_flip_pct': flip_pct
        }

        print(f"eps = {eps:<9} | {mean_mae:<10.6f} | {med_l2:<12.6f} | {max_l2:<10.4f} | {flips:3d} / {len(df):3d} mutants     | {flip_pct:5.1f}%")

    print("-" * 85)
    print("Monte Carlo Interval Sampling U[0, eps]:")
    for eps_bound in MONTE_CARLO_INTERVALS:
        mc_mae = float(df[f'MC_MAE_eps_{eps_bound}'].mean())
        mc_l2 = float(df[f'MC_RelL2_eps_{eps_bound}'].median())
        mc_conc = float(df[f'MC_PatConc_eps_{eps_bound}'].mean() * 100)
        print(f"  U[0, {eps_bound:<7}] : Mean MAE={mc_mae:.6f}, Median RelL2={mc_l2:.6f}, Pattern Concordance={mc_conc:.1f}%")

    # Save summary json
    json_path = os.path.join(out_dir, "summary_report.json")
    with open(json_path, 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSummary JSON saved to: {json_path}")

    # Step 4: Generate Representative Visual Plots
    print("\nGenerating visual comparison curves for representative mutants...")
    plot_representative_mutants(os.path.join(out_dir, "plots"))

    # Step 5: Identify the Most Fragile Mutants (Flips even at eps <= 1e-3)
    fragile_df = df[~df['PatMatch_eps_0.001']][['Mutant_ID', 'Mutant_Name', 'Zero_Params', 'Base_Pattern', 'Pat_eps_0.001', 'RelL2_eps_0.001']]
    print("\n" + "=" * 85)
    print(f" BIFURCATION-FRAGILE MUTANTS (Phase flips occurred even at tiny eps <= 0.001): {len(fragile_df)}")
    print("=" * 85)
    for _, r in fragile_df.iterrows():
        print(f"  #{r['Mutant_ID']:<4} {r['Mutant_Name']:<35} | {r['Base_Pattern'].split(':')[0]} -> {r['Pat_eps_0.001'].split(':')[0]} | RelL2={r['RelL2_eps_0.001']:.4f}")

    print("\nAudit completed successfully.")


if __name__ == '__main__':
    main()
