import os
import sys
import numpy as np
import pandas as pd
import json
import time

def classify_trajectory(y):
    """
    Classify a single trajectory y (shape: 39 x 500)
    Variables:
      0: MASS
      2: CLB2
      32: ESP1
      33: ORI
      35: SPN
    """
    mc = y[0, :]
    clb2 = y[2, :]
    esp1 = y[32, :]
    ori = y[33, :]
    spn = y[35, :]

    # 1. Discrete division count v:
    d = np.diff(mc)
    di = np.where(d < -0.1)[0]
    v = 0
    if len(di) > 0:
        v = 1
        ld = di[0]
        for dd in di[1:]:
            if dd - ld > 10:
                v += 1
                ld = dd

    max_clb2 = float(np.max(clb2))
    end_clb2 = float(clb2[-1])
    end_esp1 = float(esp1[-1])
    max_ori = float(np.max(ori))

    # 2. Orthogonal bifurcation critical flag:
    start_critical = (0.15 <= max_clb2 <= 0.40) or (0.5 <= max_ori <= 1.5 and end_clb2 < 0.25)
    finish_critical = (0.25 <= end_clb2 <= 0.80) or (0.08 <= end_esp1 <= 0.15)
    is_critical = bool(start_critical or finish_critical)

    # 3. Six mutually exclusive dynamic phases:
    if v >= 2:
        phase = "P1: Sustained Multi-Cycle Oscillation"
    elif v == 1:
        phase = "P2: Transient Single-Division Exhaustion"
    else:  # v == 0
        is_mitotic_arrest = (end_clb2 >= 0.35) or (end_esp1 < 0.10) or (max_clb2 > 10.0)
        if is_mitotic_arrest:
            if (max_clb2 > 10.0) or (end_clb2 > 5.0):
                phase = "P5b: Hyper-Clb2 Mitotic Singularity"
            else:
                phase = "P5a: Moderate Mitotic Exit Arrest"
        else:
            if max_ori < 0.5:
                phase = "P3: Quiescent G1 Arrest"
            else:
                phase = "P4: S/G2 Phase Uncoupled Arrest"

    return v, phase, is_critical, {
        'max_clb2': max_clb2,
        'end_clb2': end_clb2,
        'end_esp1': end_esp1,
        'max_ori': max_ori,
        'max_mass': float(np.max(mc)),
        'end_mass': float(mc[-1])
    }

def main():
    npz_path = '/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast/mutant_aware_pure_random_100000_210m_500s.npz'
    eval_csv_path = '/home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/experiments/mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/triple_dynamic_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/per_mutant_metrics.csv'
    
    print(f"Loading npz: {npz_path} (mmap mode)...")
    npz_data = np.load(npz_path, mmap_mode='r')
    
    data = npz_data['data']
    mutant_names = npz_data['mutant_names']
    labels = npz_data['labels']
    pattern_labels = npz_data['pattern_labels']
    
    total_samples = len(mutant_names)
    print(f"Total samples: {total_samples}")
    
    # Process 126 Real Mutants first
    print("\n--- Processing 126 Real Mutants ---")
    real_names = mutant_names[:126]
    real_data = data[:126]
    
    real_results = []
    for i in range(126):
        v, phase, is_crit, feats = classify_trajectory(real_data[i])
        real_results.append({
            'index': i,
            'name': str(real_names[i]),
            'v': v,
            'phase': phase,
            'is_bifurcation_critical': is_crit,
            'original_pattern': str(pattern_labels[i]),
            'original_label': str(labels[i]),
            **feats
        })
        
    df_real = pd.DataFrame(real_results)
    
    # Merge with eval metrics
    if os.path.exists(eval_csv_path):
        df_eval = pd.read_csv(eval_csv_path)
        # Match on index or name
        df_real['eval_MAE'] = df_eval['MAE']
        df_real['eval_MSE'] = df_eval['MSE']
        df_real['eval_Relative_L2'] = df_eval['Relative_L2']
        df_real['eval_Correlation'] = df_eval['Correlation']
    
    print(f"Real mutants classified. Distribution of phases:")
    print(df_real['phase'].value_counts())
    print("\nReal mutants bifurcation criticality:")
    print(df_real['is_bifurcation_critical'].value_counts())
    
    print("\nReal mutants Phase x Criticality:")
    print(pd.crosstab(df_real['phase'], df_real['is_bifurcation_critical'], margins=True))
    
    # Process 100,000 Synthetic Mutants
    print("\n--- Processing 100,000 Synthetic Mutants (in chunks) ---")
    synth_start = 126
    n_synth = total_samples - synth_start
    chunk_size = 10000
    
    synth_phases = []
    synth_critical = []
    synth_v = []
    synth_levels = []
    
    t0 = time.time()
    for start_idx in range(synth_start, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        # Slice only required channels: 0 (MASS), 2 (CLB2), 32 (ESP1), 33 (ORI), 35 (SPN)
        # Using [0, 2, 32, 33, 35]
        chunk_data = data[start_idx:end_idx, [0, 2, 32, 33, 35], :]
        chunk_names = mutant_names[start_idx:end_idx]
        
        for k in range(end_idx - start_idx):
            # Create a 39-channel pseudo view or adapt classify_trajectory
            y_sub = np.zeros((39, 500), dtype=np.float32)
            y_sub[0] = chunk_data[k, 0]
            y_sub[2] = chunk_data[k, 1]
            y_sub[32] = chunk_data[k, 2]
            y_sub[33] = chunk_data[k, 3]
            y_sub[35] = chunk_data[k, 4]
            
            v, phase, is_crit, _ = classify_trajectory(y_sub)
            synth_phases.append(phase)
            synth_critical.append(is_crit)
            synth_v.append(v)
            
            # extract level from name: e.g. RANDOM_..._L4_99996
            name_str = str(chunk_names[k])
            if "_L" in name_str:
                try:
                    lvl = int(name_str.split("_L")[1].split("_")[0])
                except Exception:
                    lvl = -1
            else:
                lvl = -1
            synth_levels.append(lvl)
            
        elapsed = time.time() - t0
        processed = end_idx - synth_start
        print(f"Processed {processed}/{n_synth} ({processed/n_synth*100:.1f}%) in {elapsed:.1f}s...")
        
    df_synth = pd.DataFrame({
        'phase': synth_phases,
        'is_bifurcation_critical': synth_critical,
        'v': synth_v,
        'level': synth_levels
    })
    
    print("\n--- Synthetic Mutants Phase Distribution ---")
    synth_phase_counts = df_synth['phase'].value_counts()
    synth_phase_pct = df_synth['phase'].value_counts(normalize=True) * 100
    for p in synth_phase_counts.index:
        print(f"  {p}: {synth_phase_counts[p]:6d} ({synth_phase_pct[p]:6.2f}%)")
        
    print("\n--- Synthetic Mutants Bifurcation Criticality ---")
    synth_crit_counts = df_synth['is_bifurcation_critical'].value_counts()
    synth_crit_pct = df_synth['is_bifurcation_critical'].value_counts(normalize=True) * 100
    for c in synth_crit_counts.index:
        print(f"  {c}: {synth_crit_counts[c]:6d} ({synth_crit_pct[c]:6.2f}%)")
        
    print("\n--- Synthetic Mutants Phase x Criticality ---")
    print(pd.crosstab(df_synth['phase'], df_synth['is_bifurcation_critical'], margins=True))
    
    print("\n--- Synthetic Mutants Phase x Level ---")
    print(pd.crosstab(df_synth['phase'], df_synth['level'], margins=True))
    
    # Save processed results
    out_dir = '/home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/experiments/mutant_aware_pure_random_100000_210m_500s'
    df_real.to_csv(os.path.join(out_dir, 'real_mutants_phases_diagnostics.csv'), index=False)
    
    summary_stats = {
        'real_total': 126,
        'real_phase_counts': df_real['phase'].value_counts().to_dict(),
        'real_phase_pct': (df_real['phase'].value_counts(normalize=True)*100).to_dict(),
        'real_crit_counts': df_real['is_bifurcation_critical'].value_counts().to_dict(),
        'real_crit_pct': (df_real['is_bifurcation_critical'].value_counts(normalize=True)*100).to_dict(),
        'synth_total': n_synth,
        'synth_phase_counts': df_synth['phase'].value_counts().to_dict(),
        'synth_phase_pct': (df_synth['phase'].value_counts(normalize=True)*100).to_dict(),
        'synth_crit_counts': df_synth['is_bifurcation_critical'].value_counts().to_dict(),
        'synth_crit_pct': (df_synth['is_bifurcation_critical'].value_counts(normalize=True)*100).to_dict(),
        'cross_tab_synth': pd.crosstab(df_synth['phase'], df_synth['is_bifurcation_critical']).to_dict(),
        'cross_tab_real': pd.crosstab(df_real['phase'], df_real['is_bifurcation_critical']).to_dict(),
    }
    
    with open(os.path.join(out_dir, 'dynamic_phases_summary.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
        
    print("\nSuccessfully saved diagnostics to:")
    print(f"  - {os.path.join(out_dir, 'real_mutants_phases_diagnostics.csv')}")
    print(f"  - {os.path.join(out_dir, 'dynamic_phases_summary.json')}")

if __name__ == '__main__':
    main()
