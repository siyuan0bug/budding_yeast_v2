"""backends services extracted from the pre-package implementation."""
import numpy as np
import torch


def _bmdal_distance_select(pool_matrix, train_matrix, n_select, device='cpu', *, method):
    """Official distance selectors on predicted features, without posterior."""
    from bmdal_reg.bmdal.feature_data import TensorFeatureData
    from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
    from bmdal_reg.bmdal.features import Features
    n_select = min(int(n_select), len(pool_matrix))
    if n_select <= 0:
        return np.empty(0, dtype=int)
    train_matrix, pool_matrix = train_matrix.to(device), pool_matrix.to(device)
    if not torch.isfinite(train_matrix).all() or not torch.isfinite(pool_matrix).all():
        raise ValueError('Distance selection requires finite features')
    width = train_matrix.shape[1]
    train_features = Features(IdentityFeatureMap(width), TensorFeatureData(train_matrix))
    pool_features = Features(IdentityFeatureMap(width), TensorFeatureData(pool_matrix))
    if method == 'lcmd':
        selector = selection_module().LargestClusterMaxDistSelectionMethod(
            pool_features=pool_features, train_features=train_features,
            sel_with_train=True, dist_weight_mode='sq-dist')
    elif method == 'max_dist':
        selector = selection_module().MaxDistSelectionMethod(
            pool_features=pool_features, train_features=train_features, sel_with_train=True)
    else:
        raise ValueError(f'Unsupported distance method: {method}')
    result = selector.select(n_select).detach().cpu().numpy()
    if len(result) != n_select or len(np.unique(result)) != len(result):
        raise RuntimeError(f'{method} returned an invalid selection')
    return result


def _bmdal_lcmd_select(pool_matrix, train_matrix, n_select, device='cpu'):
    return _bmdal_distance_select(pool_matrix,train_matrix,n_select,device,method='lcmd')


def _bmdal_predicted_max_dist_select(pool_matrix, train_matrix, n_select, device='cpu'):
    return _bmdal_distance_select(pool_matrix,train_matrix,n_select,device,method='max_dist')


def _bmdal_bait_select(pool_matrix, train_matrix, n_select, device='cpu'):
    """Official max_dist_selection posterior pipeline + actual bmdal BAIT."""
    from bmdal_reg.bmdal.feature_data import TensorFeatureData
    from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
    from bmdal_reg.bmdal.features import Features
    n_select = min(int(n_select), len(pool_matrix))
    if n_select <= 0:
        return np.empty(0, dtype=int)
    train_matrix, pool_matrix = train_matrix.to(device), pool_matrix.to(device)
    if not torch.isfinite(train_matrix).all() or not torch.isfinite(pool_matrix).all():
        raise ValueError('BAIT requires finite features')
    energy = train_matrix.square().sum(1).mean()
    if not torch.isfinite(energy) or energy <= 0:
        raise ValueError('BAIT training features have zero energy; official scale transform is undefined')
    width = train_matrix.shape[1]
    train_features = Features(IdentityFeatureMap(width), TensorFeatureData(train_matrix))
    pool_features = Features(IdentityFeatureMap(width), TensorFeatureData(pool_matrix))
    # Exact scale/posterior order and noise from config/acquisition/bait.yaml.
    scale = train_features.scale_tfm()
    train_features, pool_features = scale(train_features), scale(pool_features)
    posterior = train_features.posterior_tfm(sigma=1e-2, allow_kernel_space_posterior=False)
    train_features, pool_features = posterior(train_features), posterior(pool_features)
    selector = selection_module().BaitFeatureSpaceSelectionMethod(
        pool_features=pool_features, train_features=train_features,
        sel_with_train=True, noise_sigma=1e-2)
    result = selector.select(n_select).detach().cpu().numpy()
    if len(result) != n_select or len(np.unique(result)) != len(result):
        raise RuntimeError('BAIT returned an invalid selection')
    return result


def _bmdal_max_dist_select(self, pool_features, train_features, n_select):
    """原生 PyTorch 向量化 MaxDist 贪心最大距离选样 (等价于 bmdal_reg, 耗时由几百秒降至几秒, 兼具 NumPy 2 兼容性)"""
    n_pool = pool_features.shape[0]
    n_select = min(n_select, n_pool)
    if n_select <= 0:
        return np.array([], dtype=int)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pool_feat = pool_features.to(device)
        train_feat = train_features.to(device)
    except RuntimeError:
        device = torch.device('cpu')
        pool_feat = pool_features.cpu()
        train_feat = train_features.cpu()

    # 1. 计算每个候选样本到当前训练集的最小欧氏距离
    chunk_size = 50000
    min_dist_to_train = torch.full((n_pool,), float('inf'), device=device)
    with torch.no_grad():
        for i in range(0, n_pool, chunk_size):
            end_c = min(i + chunk_size, n_pool)
            p_chunk = pool_feat[i:end_c]
            dists = torch.cdist(p_chunk, train_feat)  # (chunk, M)
            min_dist_to_train[i:end_c] = dists.min(dim=1)[0]
            del dists, p_chunk

        # 2. 贪心最大距离选择
        selected_indices = []
        curr_min_dist = min_dist_to_train.clone()

        update_chunk = 100000
        for step in range(n_select):
            best_idx = torch.argmax(curr_min_dist).item()
            selected_indices.append(best_idx)
            curr_min_dist[best_idx] = -float('inf')
            if step == n_select - 1:
                break
            best_point = pool_feat[best_idx:best_idx + 1]  # (1, D)
            for i in range(0, n_pool, update_chunk):
                end_u = min(i + update_chunk, n_pool)
                p_chunk = pool_feat[i:end_u]
                dist_to_new = torch.norm(p_chunk - best_point, dim=1)
                curr_min_dist[i:end_u] = torch.minimum(curr_min_dist[i:end_u], dist_to_new)

    return np.array(selected_indices, dtype=int)



from functools import lru_cache
import importlib.util
import sys

import numpy as np


class _NumpyCompatibilityView:
    Inf = np.inf

    def __getattr__(self, name):
        return getattr(np, name)


@lru_cache(maxsize=1)
def selection_module():
    spec = importlib.util.find_spec('bmdal_reg.bmdal.selection')
    if spec is None or spec.origin is None:
        raise ImportError('BAIT/LCMD/MaxDist require the bmdal_reg dependency used by AL4PDE')
    name = 'bmdal_reg.bmdal._yeast_numpy2_selection'
    private_spec = importlib.util.spec_from_file_location(name, spec.origin)
    module = importlib.util.module_from_spec(private_spec)
    sys.modules[name] = module
    try:
        private_spec.loader.exec_module(module)
        module.np = _NumpyCompatibilityView()
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
