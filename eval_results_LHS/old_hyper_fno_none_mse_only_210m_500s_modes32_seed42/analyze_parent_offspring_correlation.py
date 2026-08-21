"""
Statistical analysis of the relationship between parent mutants and offspring LHS samples.
Quantifies correlation between model prediction performance on real mutants vs their LHS offspring.
"""
import numpy as np
import pandas as pd
import re
import json
import os
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# Paths
# ==========================================
EVAL_DIR = "/home/users/zsy/trae_workspace/budding_yeast_v2/eval_result/old_hyper_fno_none_mse_only_210m_500s_modes32_seed42"
LHS_CSV = os.path.join(EVAL_DIR, "per_lhs_test_metrics.csv")
MUTANT_CSV = os.path.join(EVAL_DIR, "per_mutant_metrics.csv")
GLOBAL_JSON = os.path.join(EVAL_DIR, "global_metrics_denorm.json")

SENSITIVITY_ATLAS = "/home/users/zsy/trae_workspace/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels_Sensitivity_Atlas.csv"
PLOTTED_SUMMARY = "/home/users/zsy/trae_workspace/mydata/budding_yeast/MAEgt-1_lhs_cheat_origin_210min_500steps_dual_labels_plots/lhs_cheat_origin_210min_500steps_dual_labels_Plotted_Summary.csv"

OUT_DIR = "/home/users/zsy/trae_workspace/active_learning_analysis"
os.makedirs(OUT_DIR, exist_ok=True)


def extract_parent_id(lhs_name):
    """
    Extract parent mutant ID from LHS sample name.
    e.g. '27_sic1_KO_LHS_030' -> '27_sic1_KO'
         '61.1_CLB2_db_clb5_Gal_LHS_139' -> '61.1_CLB2_db_clb5_Gal'
         '14_cln3_bck2_GAL_CLN2_cln1_cln2_LHS_044' -> '14_cln3_bck2_GAL_CLN2_cln1_cln2'
    """
    # Strip the _LHS_XXX suffix
    m = re.match(r'^(.+?)_LHS_\d+$', lhs_name)
    if m:
        return m.group(1)
    return None


def load_data():
    """Load all CSVs and join LHS samples with their parent mutants."""
    print("=" * 70)
    print("STEP 1: Loading data")
    print("=" * 70)

    lhs_df = pd.read_csv(LHS_CSV)
    mutant_df = pd.read_csv(MUTANT_CSV)

    with open(GLOBAL_JSON) as f:
        global_metrics = json.load(f)

    print(f"  LHS test samples loaded: {len(lhs_df)}")
    print(f"  Real mutants loaded: {len(mutant_df)}")
    print(f"  Global LHS  MAE={global_metrics['lhs_test']['all_39_vars']['MAE']:.4f}  MSE={global_metrics['lhs_test']['all_39_vars']['MSE']:.4f}")
    print(f"  Global Real MAE={global_metrics['real_mutants']['all_39_vars']['MAE']:.4f}  MSE={global_metrics['real_mutants']['all_39_vars']['MSE']:.4f}")

    # Extract parent IDs for LHS samples
    lhs_df['Parent_Mutant'] = lhs_df['Mutant'].apply(extract_parent_id)
    unmatched = lhs_df[lhs_df['Parent_Mutant'].isna()]
    if len(unmatched) > 0:
        print(f"  WARNING: {len(unmatched)} LHS samples could not be matched to a parent.")
        print(f"    Examples: {unmatched['Mutant'].head(5).tolist()}")

    # Build parent lookup
    mutant_lookup = mutant_df.set_index('Mutant')

    # Match LHS samples to parents
    lhs_df['Parent_Found'] = lhs_df['Parent_Mutant'].isin(mutant_lookup.index)
    matched_lhs = lhs_df[lhs_df['Parent_Found']].copy()
    unmatched_lhs = lhs_df[~lhs_df['Parent_Found']].copy()
    print(f"  Matched LHS samples to parents: {len(matched_lhs)} / {len(lhs_df)}")
    if len(unmatched_lhs) > 0:
        missing_parents = sorted(unmatched_lhs['Parent_Mutant'].dropna().unique())
        print(f"  Unmatched parent IDs (not in mutant CSV): {missing_parents[:10]}{'...' if len(missing_parents)>10 else ''}")

    # Attach parent metrics
    matched_lhs['Parent_MAE'] = matched_lhs['Parent_Mutant'].map(mutant_lookup['MAE'])
    matched_lhs['Parent_MSE'] = matched_lhs['Parent_Mutant'].map(mutant_lookup['MSE'])
    matched_lhs['Parent_Relative_L2'] = matched_lhs['Parent_Mutant'].map(mutant_lookup['Relative_L2'])
    matched_lhs['Parent_Correlation'] = matched_lhs['Parent_Mutant'].map(mutant_lookup['Correlation'])
    matched_lhs['Parent_Pattern'] = matched_lhs['Parent_Mutant'].map(mutant_lookup['Pattern'])

    return matched_lhs, mutant_df, global_metrics


def correlation_analysis(matched_lhs, mutant_df):
    """Compute correlations between parent and offspring performance."""
    print("\n" + "=" * 70)
    print("STEP 2: Parent-Offspring Correlation Analysis")
    print("=" * 70)

    # --- 2a. Sample-level correlation (each LHS sample vs its parent) ---
    print("\n--- 2a. Sample-level correlation (each LHS sample vs its parent's metric) ---")
    metrics = ['MAE', 'MSE', 'Relative_L2', 'Correlation']
    parent_cols = ['Parent_MAE', 'Parent_MSE', 'Parent_Relative_L2', 'Parent_Correlation']

    sample_corr_results = []
    for m, pc in zip(metrics, parent_cols):
        valid = matched_lhs[[pc, m]].dropna()
        if len(valid) < 5:
            continue
        # Spearman is more robust to non-linear relationships and outliers
        spearman_r, spearman_p = stats.spearmanr(valid[pc], valid[m])
        pearson_r, pearson_p = stats.pearsonr(valid[pc], valid[m])
        sample_corr_results.append({
            'Metric': m,
            'N_Samples': len(valid),
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p,
        })
        print(f"  {m:14s} | N={len(valid):5d} | Pearson r={pearson_r:+.4f} (p={pearson_p:.2e}) | Spearman rho={spearman_r:+.4f} (p={spearman_p:.2e})")

    sample_corr_df = pd.DataFrame(sample_corr_results)
    sample_corr_df.to_csv(os.path.join(OUT_DIR, "sample_level_correlation.csv"), index=False)

    # --- 2b. Family-level aggregation (parent metric vs mean/median of offspring) ---
    print("\n--- 2b. Family-level correlation (parent vs aggregated offspring) ---")
    family_agg = matched_lhs.groupby('Parent_Mutant').agg(
        LHS_Count=('MAE', 'count'),
        LHS_MAE_mean=('MAE', 'mean'),
        LHS_MAE_median=('MAE', 'median'),
        LHS_MAE_min=('MAE', 'min'),
        LHS_MAE_max=('MAE', 'max'),
        LHS_MAE_std=('MAE', 'std'),
        LHS_MSE_mean=('MSE', 'mean'),
        LHS_MSE_median=('MSE', 'median'),
        LHS_Corr_mean=('Correlation', 'mean'),
    ).reset_index()

    # Merge parent metrics
    mutant_lookup = mutant_df.set_index('Mutant')
    family_agg['Parent_MAE'] = family_agg['Parent_Mutant'].map(mutant_lookup['MAE'])
    family_agg['Parent_MSE'] = family_agg['Parent_Mutant'].map(mutant_lookup['MSE'])
    family_agg['Parent_Correlation'] = family_agg['Parent_Mutant'].map(mutant_lookup['Correlation'])
    family_agg['Parent_Pattern'] = family_agg['Parent_Mutant'].map(mutant_lookup['Pattern'])

    family_corr_results = []
    pairs = [
        ('Parent_MAE', 'LHS_MAE_mean', 'MAE (parent vs LHS mean)'),
        ('Parent_MAE', 'LHS_MAE_median', 'MAE (parent vs LHS median)'),
        ('Parent_MAE', 'LHS_MAE_min', 'MAE (parent vs LHS best)'),
        ('Parent_MAE', 'LHS_MAE_max', 'MAE (parent vs LHS worst)'),
        ('Parent_MSE', 'LHS_MSE_mean', 'MSE (parent vs LHS mean)'),
        ('Parent_MSE', 'LHS_MSE_median', 'MSE (parent vs LHS median)'),
        ('Parent_Correlation', 'LHS_Corr_mean', 'Correlation (parent vs LHS mean)'),
    ]
    for x_col, y_col, label in pairs:
        valid = family_agg[[x_col, y_col]].dropna()
        if len(valid) < 5:
            continue
        spearman_r, spearman_p = stats.spearmanr(valid[x_col], valid[y_col])
        pearson_r, pearson_p = stats.pearsonr(valid[x_col], valid[y_col])
        family_corr_results.append({
            'Comparison': label,
            'N_Families': len(valid),
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p,
        })
        print(f"  {label:42s} | N={len(valid):3d} | Pearson r={pearson_r:+.4f} (p={pearson_p:.2e}) | Spearman rho={spearman_r:+.4f} (p={spearman_p:.2e})")

    family_corr_df = pd.DataFrame(family_corr_results)
    family_corr_df.to_csv(os.path.join(OUT_DIR, "family_level_correlation.csv"), index=False)
    family_agg.to_csv(os.path.join(OUT_DIR, "family_aggregated_metrics.csv"), index=False)

    return family_agg, sample_corr_df, family_corr_df


def gap_analysis(matched_lhs, mutant_df, family_agg):
    """Identify mutants with the largest parent-offspring performance gap."""
    print("\n" + "=" * 70)
    print("STEP 3: Performance Gap Analysis (Parent - Offspring)")
    print("=" * 70)

    # Compute gap = Parent_MAE - LHS_MAE_median (positive = parent is harder)
    family_agg['MAE_Gap'] = family_agg['Parent_MAE'] - family_agg['LHS_MAE_median']
    family_agg['MAE_Ratio'] = family_agg['Parent_MAE'] / (family_agg['LHS_MAE_median'] + 1e-10)

    # Rank by gap (largest positive gap = biggest generalization problem)
    ranked = family_agg.sort_values('MAE_Gap', ascending=False).reset_index(drop=True)

    print("\nTop 15 mutants with LARGEST parent-offspring MAE gap (parent much harder):")
    print(f"  {'Mutant':45s} {'Pattern':35s} {'P_MAE':>8s} {'LHS_med':>8s} {'Gap':>8s} {'Ratio':>7s} {'N':>4s}")
    print("  " + "-" * 120)
    for _, r in ranked.head(15).iterrows():
        print(f"  {r['Parent_Mutant']:45s} {str(r['Parent_Pattern']):35s} {r['Parent_MAE']:8.4f} {r['LHS_MAE_median']:8.4f} {r['MAE_Gap']:8.4f} {r['MAE_Ratio']:7.2f} {int(r['LHS_Count']):4d}")

    print("\nTop 15 mutants with SMALLEST/NEGATIVE gap (LHS harder than parent):")
    print(f"  {'Mutant':45s} {'Pattern':35s} {'P_MAE':>8s} {'LHS_med':>8s} {'Gap':>8s} {'Ratio':>7s} {'N':>4s}")
    print("  " + "-" * 120)
    for _, r in ranked.tail(15).iterrows():
        print(f"  {r['Parent_Mutant']:45s} {str(r['Parent_Pattern']):35s} {r['Parent_MAE']:8.4f} {r['LHS_MAE_median']:8.4f} {r['MAE_Gap']:8.4f} {r['MAE_Ratio']:7.2f} {int(r['LHS_Count']):4d}")

    # Pattern-stratified gap analysis
    print("\nPattern-stratified gap analysis:")
    pattern_gap = family_agg.groupby('Parent_Pattern').agg(
        N_Families=('Parent_Mutant', 'count'),
        Mean_Parent_MAE=('Parent_MAE', 'mean'),
        Mean_LHS_MAE=('LHS_MAE_median', 'mean'),
        Mean_Gap=('MAE_Gap', 'mean'),
        Mean_Ratio=('MAE_Ratio', 'mean'),
    ).reset_index()
    print(pattern_gap.to_string(index=False))
    pattern_gap.to_csv(os.path.join(OUT_DIR, "pattern_stratified_gap.csv"), index=False)

    ranked.to_csv(os.path.join(OUT_DIR, "mutant_gap_ranking.csv"), index=False)
    return ranked


def coverage_analysis(matched_lhs, mutant_df):
    """Analyze LHS sample coverage per parent mutant."""
    print("\n" + "=" * 70)
    print("STEP 4: LHS Coverage Analysis")
    print("=" * 70)

    coverage = matched_lhs.groupby('Parent_Mutant').size().reset_index(name='LHS_Count')
    mutant_lookup = mutant_df.set_index('Mutant')
    coverage['Parent_MAE'] = coverage['Parent_Mutant'].map(mutant_lookup['MAE'])
    coverage['Parent_Pattern'] = coverage['Parent_Mutant'].map(mutant_lookup['Pattern'])

    # Mutants with NO LHS offspring in test set
    all_mutants = set(mutant_df['Mutant'])
    mutants_with_lhs = set(coverage['Parent_Mutant'])
    mutants_without_lhs = all_mutants - mutants_with_lhs
    print(f"  Mutants with LHS offspring in test set: {len(mutants_with_lhs)} / {len(all_mutants)}")
    print(f"  Mutants WITHOUT any LHS offspring in test set: {len(mutants_without_lhs)}")
    if mutants_without_lhs:
        print(f"    Examples: {sorted(mutants_without_lhs)[:15]}")

    print(f"\n  LHS count per parent distribution:")
    print(f"    min={coverage['LHS_Count'].min()}, max={coverage['LHS_Count'].max()}, "
          f"mean={coverage['LHS_Count'].mean():.1f}, median={coverage['LHS_Count'].median():.1f}")

    # Correlation between coverage and parent performance
    valid = coverage[['LHS_Count', 'Parent_MAE']].dropna()
    if len(valid) >= 5:
        r, p = stats.spearmanr(valid['LHS_Count'], valid['Parent_MAE'])
        print(f"\n  Correlation (LHS coverage count vs Parent MAE): Spearman rho={r:+.4f} (p={p:.2e})")
        if p < 0.05:
            print(f"    -> Significant: {'more' if r>0 else 'fewer'} LHS samples associated with {'worse' if r>0 else 'better'} parent performance")

    coverage.to_csv(os.path.join(OUT_DIR, "lhs_coverage_per_mutant.csv"), index=False)
    return coverage, mutants_without_lhs


def sensitivity_atlas_analysis():
    """Analyze the Sensitivity Atlas to understand LHS sampling distribution."""
    print("\n" + "=" * 70)
    print("STEP 5: Sensitivity Atlas Analysis (Ground-Truth Distance Distribution)")
    print("=" * 70)

    if not os.path.exists(SENSITIVITY_ATLAS):
        print(f"  Sensitivity Atlas not found: {SENSITIVITY_ATLAS}")
        return None

    atlas = pd.read_csv(SENSITIVITY_ATLAS)
    print(f"  Loaded Sensitivity Atlas: {len(atlas)} rows")

    # Per-family statistics on ground-truth MAE (distance from parent)
    family_stats = atlas.groupby('Real_Mutant').agg(
        N_LHS=('LHS_Sample_Name', 'count'),
        GT_MAE_min=('MAE', 'min'),
        GT_MAE_median=('MAE', 'median'),
        GT_MAE_mean=('MAE', 'mean'),
        GT_MAE_max=('MAE', 'max'),
        GT_MAE_q10=('MAE', lambda x: np.quantile(x, 0.10)),
        GT_MAE_q90=('MAE', lambda x: np.quantile(x, 0.90)),
    ).reset_index()

    # How many LHS samples are "near-duplicates" (GT MAE < 0.01)?
    near_dup_threshold = 0.01
    near_dups = atlas[atlas['MAE'] < near_dup_threshold]
    print(f"\n  Near-duplicate LHS samples (GT MAE < {near_dup_threshold}): {len(near_dups)} / {len(atlas)} ({100*len(near_dups)/len(atlas):.1f}%)")

    # How many LHS samples are "very different" (GT MAE > 0.5)?
    far_threshold = 0.5
    far_samples = atlas[atlas['MAE'] > far_threshold]
    print(f"  Far LHS samples (GT MAE > {far_threshold}): {len(far_samples)} / {len(atlas)} ({100*len(far_samples)/len(atlas):.1f}%)")

    # Distribution of GT MAE
    print(f"\n  Ground-truth MAE distribution (LHS vs parent):")
    print(f"    min={atlas['MAE'].min():.4f}, q10={np.quantile(atlas['MAE'], 0.10):.4f}, "
          f"median={atlas['MAE'].median():.4f}, q90={np.quantile(atlas['MAE'], 0.90):.4f}, max={atlas['MAE'].max():.4f}")

    # Pattern flip analysis: how often does LHS offspring have different pattern than parent?
    pattern_flip = atlas[atlas['Real_Pattern'] != atlas['LHS_Pattern']]
    print(f"\n  Pattern flips (LHS pattern != parent pattern): {len(pattern_flip)} / {len(atlas)} ({100*len(pattern_flip)/len(atlas):.1f}%)")

    # Per-family pattern flip rate
    flip_rate = pattern_flip.groupby('Real_Mutant').size().reset_index(name='Flip_Count')
    family_stats = family_stats.merge(flip_rate, on='Real_Mutant', how='left')
    family_stats['Flip_Count'] = family_stats['Flip_Count'].fillna(0)
    family_stats['Flip_Rate'] = family_stats['Flip_Count'] / family_stats['N_LHS']

    # Families with highest pattern flip rate (indicates LHS samples crossing decision boundaries)
    print("\n  Top 10 families with HIGHEST pattern flip rate (LHS crosses decision boundary):")
    top_flip = family_stats.sort_values('Flip_Rate', ascending=False).head(10)
    print(f"    {'Mutant':40s} {'N_LHS':>6s} {'Flips':>6s} {'Rate':>6s} {'GT_MAE_med':>11s}")
    for _, r in top_flip.iterrows():
        print(f"    {r['Real_Mutant']:40s} {int(r['N_LHS']):6d} {int(r['Flip_Count']):6d} {r['Flip_Rate']:6.2f} {r['GT_MAE_median']:11.4f}")

    family_stats.to_csv(os.path.join(OUT_DIR, "sensitivity_atlas_family_stats.csv"), index=False)
    return family_stats


def identify_active_learning_targets(family_agg, ranked, coverage, mutants_without_lhs, atlas_stats):
    """Identify priority mutants for active learning acquisition."""
    print("\n" + "=" * 70)
    print("STEP 6: Active Learning Target Identification")
    print("=" * 70)

    # Priority score combines:
    #   1. High parent MAE (model struggles on real mutant)
    #   2. Large parent-offspring gap (LHS doesn't represent parent well)
    #   3. Low LHS coverage (few offspring to learn from)
    #   4. High pattern flip rate (LHS samples cross decision boundaries)

    targets = family_agg.copy()

    # Normalize components to [0,1]
    def norm(s):
        s = s.fillna(s.median())
        rng = s.max() - s.min()
        if rng < 1e-10:
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / rng

    targets['score_parent_mae'] = norm(targets['Parent_MAE'])
    targets['score_gap'] = norm(targets['MAE_Gap'])
    targets['score_low_coverage'] = 1.0 - norm(targets['LHS_Count'])  # fewer = higher priority

    if atlas_stats is not None:
        atlas_lookup = atlas_stats.set_index('Real_Mutant')
        targets['Flip_Rate'] = targets['Parent_Mutant'].map(atlas_lookup['Flip_Rate']).fillna(0)
        targets['score_flip'] = norm(targets['Flip_Rate'])
        # Weighted priority: emphasize parent difficulty and gap
        targets['Priority_Score'] = (
            0.35 * targets['score_parent_mae'] +
            0.30 * targets['score_gap'] +
            0.20 * targets['score_low_coverage'] +
            0.15 * targets['score_flip']
        )
    else:
        targets['Priority_Score'] = (
            0.45 * targets['score_parent_mae'] +
            0.35 * targets['score_gap'] +
            0.20 * targets['score_low_coverage']
        )

    targets = targets.sort_values('Priority_Score', ascending=False).reset_index(drop=True)

    print("\nTop 25 priority mutants for active learning:")
    print(f"  {'Rank':>4s} {'Mutant':45s} {'Pattern':35s} {'P_MAE':>8s} {'Gap':>8s} {'N_LHS':>6s} {'Priority':>9s}")
    print("  " + "-" * 130)
    for i, r in targets.head(25).iterrows():
        print(f"  {i+1:4d} {r['Parent_Mutant']:45s} {str(r['Parent_Pattern']):35s} {r['Parent_MAE']:8.4f} {r['MAE_Gap']:8.4f} {int(r['LHS_Count']):6d} {r['Priority_Score']:9.4f}")

    targets.to_csv(os.path.join(OUT_DIR, "active_learning_priority_targets.csv"), index=False)
    return targets


def main():
    print("\n" + "#" * 70)
    print("# PARENT-OFFSPRING CORRELATION & ACTIVE LEARNING ANALYSIS")
    print("#" * 70)

    matched_lhs, mutant_df, global_metrics = load_data()
    family_agg, sample_corr_df, family_corr_df = correlation_analysis(matched_lhs, mutant_df)
    ranked = gap_analysis(matched_lhs, mutant_df, family_agg)
    coverage, mutants_without_lhs = coverage_analysis(matched_lhs, mutant_df)
    atlas_stats = sensitivity_atlas_analysis()
    targets = identify_active_learning_targets(family_agg, ranked, coverage, mutants_without_lhs, atlas_stats)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n1. Performance gap: Real MAE={global_metrics['real_mutants']['all_39_vars']['MAE']:.4f} vs LHS MAE={global_metrics['lhs_test']['all_39_vars']['MAE']:.4f} ({global_metrics['real_mutants']['all_39_vars']['MAE']/global_metrics['lhs_test']['all_39_vars']['MAE']:.1f}x worse on real)")
    print(f"2. Sample-level correlation (Parent MAE vs LHS MAE): Spearman rho={sample_corr_df.iloc[0]['Spearman_r']:+.4f}")
    print(f"3. Family-level correlation (Parent MAE vs LHS median MAE): Spearman rho={family_corr_df.iloc[1]['Spearman_r']:+.4f}")
    print(f"4. Mutants without LHS coverage in test: {len(mutants_without_lhs)}")
    print(f"5. Top priority mutant for active learning: {targets.iloc[0]['Parent_Mutant']} (score={targets.iloc[0]['Priority_Score']:.4f})")
    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
