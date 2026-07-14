import sys
import math
import collections
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def load_metadata_only(dataset_path):
    print(f"Loading metadata from {dataset_path} ... (This avoids loading the huge Y tensor)")
    t0 = time.time()
    dataset = np.load(dataset_path, allow_pickle=True)
    
    # Extract only what we need
    param_matrix = dataset['params'] if 'params' in dataset else dataset['param_conds']
    pattern_labels = dataset['pattern_labels']
    mutant_names = dataset.get('mutant_names', [f"Sample_{i:05d}" for i in range(param_matrix.shape[0])])
    
    print(f"Loaded {len(mutant_names)} samples in {time.time()-t0:.2f}s.")
    return param_matrix, pattern_labels, mutant_names

def main():
    dataset_path = '/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels.npz'
    param_matrix, pattern_labels, mutant_names = load_metadata_only(dataset_path)

    # Extract LHS indices
    lhs_indices = [i for i, name in enumerate(mutant_names) if '_LHS_' in str(name)]
    if not lhs_indices:
        print("Warning: No _LHS_ names found. Assuming all except last 10 are LHS.")
        lhs_indices = list(range(len(mutant_names) - 10))

    print(f"Total LHS samples to process: {len(lhs_indices)}")

    # -------------------------------------------------------------
    # 1. Stratified Splitting (Simulating yeast_datamodule.py)
    # -------------------------------------------------------------
    print("\n[Method A] Executing DataModule Stratified Splitting (Family + Pattern)...")
    np.random.seed(42)
    
    family_groups = collections.defaultdict(list)
    for idx in lhs_indices:
        name = str(mutant_names[idx])
        family = name.split('_LHS_')[0] if '_LHS_' in name else name
        family_groups[family].append(idx)

    train_idx_stratified = []
    
    for family, family_indices in family_groups.items():
        pattern_subgroups = collections.defaultdict(list)
        for idx in family_indices:
            pattern = str(pattern_labels[idx]).split(':')[0]
            pattern_subgroups[pattern].append(idx)
            
        for pat, pat_indices in pattern_subgroups.items():
            np.random.shuffle(pat_indices)
            n_pat = len(pat_indices)
            val_quota = math.ceil(n_pat * 0.10)
            test_quota = math.ceil(n_pat * 0.10)
            
            # Orphan protection mechanism
            if n_pat <= 2:
                val_quota, test_quota = 0, 0
            elif n_pat <= 4:
                test_quota = 0
                
            if val_quota + test_quota > n_pat:
                test_quota = max(0, n_pat - val_quota)
                
            train_from_pat = pat_indices[val_quota + test_quota:]
            train_idx_stratified.extend(train_from_pat)

    np.random.shuffle(train_idx_stratified)
    print(f"Stratified Train Pool Size: {len(train_idx_stratified)}")

    # -------------------------------------------------------------
    # 2. KMeans Splitting (Simulating al_callback.py Pool-Based)
    # -------------------------------------------------------------
    print("\n[Method B] Executing ActiveLearningCallback KMeans Splitting...")
    # Normalize params using LHS statistics
    p_mean = param_matrix[lhs_indices].mean(axis=0, keepdims=True)
    p_std = param_matrix[lhs_indices].std(axis=0, keepdims=True) + 1e-6
    param_norm = (param_matrix - p_mean) / p_std
    
    # We cluster ON the stratified train pool to pick 5000 initial samples, 
    # exactly as al_callback.py does in on_train_start!
    train_P = param_norm[train_idx_stratified]
    
    n_clusters = min(5000, train_P.shape[0])
    print(f"Running KMeans with {n_clusters} clusters (this may take a minute)...")
    
    # Using MiniBatchKMeans for speed in this test script, mathematically similar distribution
    t0 = time.time()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000, max_iter=50)
    cluster_labels = kmeans.fit_predict(train_P)
    centers = kmeans.cluster_centers_
    print(f"KMeans completed in {time.time()-t0:.2f}s.")

    train_idx_kmeans = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        # Find closest sample to cluster center
        distances = np.linalg.norm(train_P[cluster_indices] - centers[cluster_id], axis=1)
        nearest_idx = cluster_indices[np.argmin(distances)]
        train_idx_kmeans.append(train_idx_stratified[nearest_idx])

    print(f"KMeans Initial Train Pool Size: {len(train_idx_kmeans)}")

    # -------------------------------------------------------------
    # 3. Distribution Comparison
    # -------------------------------------------------------------
    def get_distribution(indices):
        fam_dist = collections.defaultdict(int)
        pat_dist = collections.defaultdict(int)
        for idx in indices:
            name = str(mutant_names[idx])
            family = name.split('_LHS_')[0] if '_LHS_' in name else name
            pattern = str(pattern_labels[idx]).split(':')[0]
            fam_dist[family] += 1
            pat_dist[pattern] += 1
        return fam_dist, pat_dist

    strat_fam, strat_pat = get_distribution(train_idx_stratified)
    kmeans_fam, kmeans_pat = get_distribution(train_idx_kmeans)

    print("\n" + "="*80)
    print(f"{'PATTERN DISTRIBUTION COMPARISON':^80}")
    print("="*80)
    for p in sorted(strat_pat.keys()):
        strat_ratio = strat_pat[p] / len(train_idx_stratified) * 100
        km_ratio = kmeans_pat.get(p, 0) / len(train_idx_kmeans) * 100
        print(f"{p:<15}: Stratified Pool = {strat_ratio:>6.2f}% | KMeans Picked = {km_ratio:>6.2f}%")

    print("\n" + "="*80)
    print(f"{'RAREST FAMILIES COMPARISON (Top 15 most vulnerable families)':^80}")
    print("="*80)
    print(f"{'Family':<35} | {'Stratified Count':<18} | {'KMeans Count':<15}")
    print("-" * 80)
    for f in sorted(strat_fam.keys(), key=lambda x: strat_fam[x])[:15]:
        print(f"{f[:35]:<35} | {strat_fam[f]:<18} | {kmeans_fam.get(f, 0):<15}")
        
    print("\n" + "="*80)
    print(f"{'LARGEST FAMILIES COMPARISON (Top 5 largest families)':^80}")
    print("="*80)
    print(f"{'Family':<35} | {'Stratified Count':<18} | {'KMeans Count':<15}")
    print("-" * 80)
    for f in sorted(strat_fam.keys(), key=lambda x: strat_fam[x], reverse=True)[:5]:
        print(f"{f[:35]:<35} | {strat_fam[f]:<18} | {kmeans_fam.get(f, 0):<15}")

if __name__ == "__main__":
    main()
