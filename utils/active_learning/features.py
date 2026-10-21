"""features services extracted from the pre-package implementation."""
import torch.nn.functional as F
import numpy as np
import torch

_SHAPE_FEAT_GROUP_SIZES = [32, 3, 39, 39, 39, 39, 141]

_SHAPE_FEAT_GROUP_WEIGHTS = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5]


def _bait_feature_batch(self, predictions):
    """Official spatial_mean + sketch for an ODE with a singleton spatial axis.

    B,V,T -> B,1,T,V; spatial mean removes only the singleton, preserving
    time and variables. This is not a temporal mean or the custom 332D map.
    """
    from bmdal_reg.bmdal.feature_data import TensorFeatureData
    from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
    from bmdal_reg.bmdal.features import Features
    raw = predictions.detach().cpu().transpose(1, 2).flatten(1)
    if not torch.isfinite(raw).all():
        raise ValueError('Nonfinite BAIT prediction features')
    features = Features(IdentityFeatureMap(raw.shape[1]), TensorFeatureData(raw))
    if self._bait_sketch is None:
        # One sketch shared by train/pool and all rounds, as in AL4PDE.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_seed)
            self._bait_sketch = features.sketch_tfm(self.bait_sketch_dim)
    return self._bait_sketch(features).get_feature_matrix()


def _compute_predicted_distance_features(self, pl_module, dm, device, batch_size=128):
    """Predict BOTH train and pool trajectories; never query the ODE oracle."""
    train_x, train_p = dm.train_dataset_synth.tensors[:2]
    if not len(train_p):
        raise ValueError('Distance acquisition requires a nonempty training set')
    training = pl_module.training
    pl_module.eval()
    def collect(count, inputs):
        output = None
        for start in range(0, count, batch_size):
            indices = np.arange(start, min(start + batch_size, count))
            x, p = inputs(indices)
            predictions = pl_module.forward_ic_time(x.to(device), p.to(device))
            features = self._bait_feature_batch(predictions)
            if output is None:
                output = torch.empty(count, features.shape[1], dtype=features.dtype)
            output[indices] = features
        return output
    try:
        with torch.no_grad():
            train_features = collect(len(train_p), lambda idx: (train_x[idx], train_p[idx]))
            pool_features = collect(self._pool_size,
                lambda idx: (self._get_pool_X_batch(idx), self._pool_P[idx]))
    finally:
        pl_module.train(training)
    return pool_features, train_features


def _compute_bait_features(self, pl_module, dm, device, batch_size=128):
    """Compatibility name for the shared BAIT/LCMD prediction features."""
    return self._compute_predicted_distance_features(pl_module, dm, device, batch_size)


def _compute_pool_features_spatial_mean(self, pl_module, dm, device, batch_size=200):
    """对候选池做 forward, 返回轨迹 spatial_mean 特征 (al4pde 默认 traj_transform)

    features(c) = model(X(c), params(c)).mean(time_dim) -> (N, 39)
    """
    pl_module.eval()
    features = []
    with torch.no_grad():
        for i in range(0, self._pool_size, batch_size):
            end_idx = min(i + batch_size, self._pool_size)
            X_batch = self._get_pool_X_batch(slice(i, end_idx)).to(device)
            P_batch = self._pool_P[i:end_idx].to(device)
            # forward: (B, 39, 2, T) + (B, 141) -> (B, 39, T)
            preds = pl_module.forward_ic_time(X_batch, P_batch)
            # spatial_mean: 对时间维取 mean -> (B, 39)
            feat = preds.mean(dim=2).cpu()
            features.append(feat)
    pl_module.train()
    return torch.cat(features, dim=0)


def _compute_train_features_spatial_mean(self, pl_module, dm, device, batch_size=200):
    """对训练集做 forward, 返回轨迹 spatial_mean 特征"""
    train_X, train_P, _ = dm.train_dataset_synth.tensors[:3]
    n_train = len(train_P)
    pl_module.eval()
    features = []
    with torch.no_grad():
        for i in range(0, n_train, batch_size):
            end_idx = min(i + batch_size, n_train)
            X_batch = train_X[i:end_idx].to(device)
            P_batch = train_P[i:end_idx].to(device)
            preds = pl_module.forward_ic_time(X_batch, P_batch)
            feat = preds.mean(dim=2).cpu()
            features.append(feat)
    pl_module.train()
    return torch.cat(features, dim=0)


def _extract_shape_features(self, preds, params):
    """从模型预测轨迹和参数中提取形状感知特征 (raw, 未标准化)

    Args:
        preds: (B, 39, T) 模型预测轨迹
        params: (B, 141) 归一化参数
    Returns:
        features: (B, 332) raw 特征 (拼接, 未标准化)
    """
    B, V, T = preds.shape
    groups = []

    # 1. MASS 轨迹降采样 (变量 0, 用户关键洞察: mass pattern 决定曲线类别)
    mass_traj = preds[:, 0, :]  # (B, T)
    mass_ds = F.interpolate(
        mass_traj.unsqueeze(1), size=32, mode='linear', align_corners=False
    ).squeeze(1)  # (B, 32)
    groups.append(mass_ds)

    # 2. MASS 一阶差分统计 (区分指数/线性/阶跃增长)
    mass_diff = mass_traj[:, 1:] - mass_traj[:, :-1]  # (B, T-1)
    groups.append(torch.stack([
        mass_diff.mean(dim=1),     # 平均增长率
        mass_diff.std(dim=1),      # 增长率波动 (线性≈0, 指数>0)
        mass_diff.max(dim=1)[0],   # 最大增长 (捕捉 GAL 诱导阶跃)
    ], dim=1))  # (B, 3)

    # 3. 全变量轨迹统计 (捕捉整体形状, 不仅是均值)
    groups.append(preds.mean(dim=2))     # (B, 39)
    groups.append(preds.std(dim=2))      # (B, 39)
    groups.append(preds.max(dim=2)[0])   # (B, 39)
    groups.append(preds.min(dim=2)[0])   # (B, 39)

    # 4. 参数空间特征 (辅助: 参数层多样性, 已归一化)
    groups.append(params)  # (B, 141)

    return torch.cat(groups, dim=1)  # (B, 332)


def _standardize_shape_features(self, pool_feat, train_feat):
    """用 train 统计量对 pool+train 特征做分组 z-score 标准化 + 分组权重

    各组独立标准化, 防止高方差组 (如参数) 主导距离。
    用 train 的 mean/std (而非 pool) 避免候选池分布偏移影响标准化基准。

    v2.2 方案 A: 标准化后施加 _SHAPE_FEAT_GROUP_WEIGHTS 权重,
    增强参数空间 (141 维) 敏感度 ×2.5, 降低 MASS 曲线 (32 维) 权重 ×0.5,
    缓解 diversity 过拟合到 LHS 形状分布的问题。

    Args:
        pool_feat: (N, 332) raw pool 特征
        train_feat: (M, 332) raw train 特征
    Returns:
        pool_std, train_std: 标准化+加权后的特征
    """
    assert pool_feat.shape[1] == sum(self._SHAPE_FEAT_GROUP_SIZES)
    assert len(self._SHAPE_FEAT_GROUP_SIZES) == len(self._SHAPE_FEAT_GROUP_WEIGHTS)
    idx = 0
    pool_parts, train_parts = [], []
    for size, weight in zip(self._SHAPE_FEAT_GROUP_SIZES, self._SHAPE_FEAT_GROUP_WEIGHTS):
        train_g = train_feat[:, idx:idx + size]
        mean = train_g.mean(dim=0, keepdim=True)
        std = train_g.std(dim=0, keepdim=True)
        # z-score 标准化后施加分组权重 (v2.2 方案 A)
        pool_std_g = (pool_feat[:, idx:idx + size] - mean) / (std + 1e-8)
        train_std_g = (train_g - mean) / (std + 1e-8)
        pool_parts.append(pool_std_g * weight)
        train_parts.append(train_std_g * weight)
        idx += size
    return torch.cat(pool_parts, dim=1), torch.cat(train_parts, dim=1)


def _compute_pool_features_shape_aware(self, pl_module, dm, device, batch_size=2500, indices=None):
    """返回候选池 (或指定子集) 的轨迹形状感知特征 (raw, 未标准化)
    通过模型 forward_ic_time 快速批量预测生成轨迹特征 (无需前端耗时 ODE 求解)。
    """
    if indices is None:
        if self._pool_shape_features_cached is not None:
            return self._pool_shape_features_cached
        eval_indices = np.arange(self._pool_size)
    else:
        eval_indices = np.asarray(indices, dtype=np.int64)

    n_eval = len(eval_indices)
    pl_module.eval()
    out_features = None
    with torch.no_grad():
        for i in range(0, n_eval, batch_size):
            end_idx = min(i + batch_size, n_eval)
            b_idx = eval_indices[i:end_idx]
            X_batch = self._get_pool_X_batch(b_idx).to(device)
            P_batch = self._pool_P[b_idx].to(device)
            preds = pl_module.forward_ic_time(X_batch, P_batch)
            feat = self._extract_shape_features(preds, P_batch).cpu()
            del X_batch, P_batch, preds
            if out_features is None:
                out_features = torch.empty((n_eval, feat.shape[1]), dtype=torch.float32)
            out_features[i:end_idx] = feat
            del feat
    pl_module.train()
    if indices is None:
        self._pool_shape_features_cached = out_features
    return out_features


def _compute_train_features_shape_aware(self, pl_module, dm, device, batch_size=200):
    """从训练集真实 Y 轨迹提取形状特征 (raw, 未标准化)

    改进: 训练集已有真实轨迹 (数据集中的 Y), 直接用, 不需要模型预测。
    """
    _, train_P, train_Y = dm.train_dataset_synth.tensors[:3]
    n_train = len(train_P)
    features = []
    for i in range(0, n_train, batch_size):
        end_idx = min(i + batch_size, n_train)
        Y_batch = train_Y[i:end_idx]      # (B, 39, T) 真实轨迹 (归一化空间)
        P_batch = train_P[i:end_idx]      # (B, 141) 归一化参数
        # Y 是归一化的, 但形状特征在分组 z-score 后与归一化无关
        feat = self._extract_shape_features(Y_batch, P_batch).cpu()
        features.append(feat)
    return torch.cat(features, dim=0)


def _compute_diversity_scores(self, pool_features, train_features, chunk=50000):
    """计算每个候选样本到训练集的最小欧氏距离 (diversity score), 支持 GPU 高速批处理 (流式传输, 显存 < 100MB)"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_feat = train_features.to(device)
    except RuntimeError:
        device = torch.device('cpu')
        train_feat = train_features.cpu()

    n_pool = pool_features.shape[0]
    min_dists = torch.empty((n_pool,), dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, n_pool, chunk):
            end_i = min(i + chunk, n_pool)
            p_chunk = pool_features[i:end_i].to(device)
            dist = torch.cdist(p_chunk, train_feat)
            min_dists[i:end_i] = dist.min(dim=1)[0].cpu()
            del dist, p_chunk
    return min_dists.numpy()


def _normalize_array(self, arr):
    """归一化数组到 [0, 1]"""
    arr = np.asarray(arr, dtype=np.float64)
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


