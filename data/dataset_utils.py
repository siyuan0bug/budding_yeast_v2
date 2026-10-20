import numpy as np


def load_yeast_dataset_universal(dataset_path, *, return_metadata=False):
    print(f"Loading dataset: {dataset_path}")
    with np.load(dataset_path, allow_pickle=True) as dataset:
        if 'data' not in dataset or str(dataset.get('artifact_kind',''))=='parameter_pool':
            raise ValueError('A parameter pool has no trajectories; simulate it before training')
        data = dataset['data']
        param_matrix = dataset['params'] if 'params' in dataset else dataset['param_conds']
        names = dataset.get('mutant_names', np.array([f"Sample_{i:05d}" for i in range(data.shape[0])]))
        pattern_labels = dataset['pattern_labels']
        from .systems.budding.pattern_groups import PATTERN_GROUP_FIELDS
        metadata={key:dataset[key] for key in ('system_id','schema_version','is_real','time_grid','variable_names','param_names','artifact_kind',
                  'labels','biological_labels_available',*PATTERN_GROUP_FIELDS) if key in dataset}
    if data.ndim!=3 or param_matrix.ndim!=2 or len(param_matrix)!=len(data) or len(names)!=len(data) or len(pattern_labels)!=len(data):
        raise ValueError('Dataset trajectory, parameter and label rows do not agree')
    if not np.isfinite(data).all() or not np.isfinite(param_matrix).all():
        raise ValueError('Dataset contains nonfinite trajectories or parameters')
    if 'is_real' in metadata and metadata['is_real'].shape!=(len(data),):
        raise ValueError('Dataset is_real must have one flag per sample')
    if 'time_grid' in metadata:
        grid=metadata['time_grid']
        if grid.shape!=(data.shape[2],) or not np.isfinite(grid).all() or grid[0]!=0 or (np.diff(grid)<=0).any():
            raise ValueError('Dataset has an invalid physical time grid')
    if return_metadata:
        from .label_metadata import trajectory_label_metadata
        from .systems.budding.pattern_groups import validate_stored_pattern_groups
        system_id=str(metadata.get('system_id','budding' if data.shape[1]==39 else 'fission' if data.shape[1]==16 else 'unspecified'))
        calculated=trajectory_label_metadata(data,metadata.get('labels'),system_id)
        if 'biological_labels_available' in metadata and not np.array_equal(metadata['biological_labels_available'],calculated['biological_labels_available']):
            raise ValueError('Stored biological-label availability disagrees with labels')
        if system_id=='budding':
            validate_stored_pattern_groups(metadata,calculated)
            metadata['pattern_group_source']='stored_verified' if 'pattern_group_version' in metadata else 'derived_from_physical_trajectory'
        metadata.update(calculated)
    num_params = param_matrix.shape[1]
    print(f"Loaded successfully! Valid samples: {data.shape[0]} | Param dim: {num_params} | Time steps: {data.shape[2]}")
    result=(data, param_matrix, names, num_params, pattern_labels)
    return (*result,metadata) if return_metadata else result


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
