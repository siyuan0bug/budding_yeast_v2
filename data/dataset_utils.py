import numpy as np


def load_yeast_dataset_universal(dataset_path):
    print(f"Loading dataset: {dataset_path}")
    dataset = np.load(dataset_path, allow_pickle=True)
    data = dataset['data']
    param_matrix = dataset['params'] if 'params' in dataset else dataset['param_conds']
    names = dataset.get('mutant_names', [f"Sample_{i:05d}" for i in range(data.shape[0])])
    num_params = param_matrix.shape[1]
    pattern_labels = dataset['pattern_labels']
    print(f"Loaded successfully! Valid samples: {data.shape[0]} | Param dim: {num_params} | Time steps: {data.shape[2]}")
    return data, param_matrix, names, num_params, pattern_labels


def stratified_split_subset(global_indices, labels_subset, ratios=(0.7, 0.1, 0.2), seed=42):
    np.random.seed(seed)
    train_idx, val_idx, test_idx = [], [], []
    unique_labels = np.unique(labels_subset)
    for lbl in unique_labels:
        local_lbl_indices = np.where(labels_subset == lbl)[0]
        np.random.shuffle(local_lbl_indices)
        n = len(local_lbl_indices)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train_idx.extend(local_lbl_indices[:n_train])
        val_idx.extend(local_lbl_indices[n_train:n_train + n_val])
        test_idx.extend(local_lbl_indices[n_train + n_val:])
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    return [global_indices[i] for i in train_idx], [global_indices[i] for i in val_idx], [global_indices[i] for i in test_idx]
