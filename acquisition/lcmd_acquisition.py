"""
LCMD (Largest Cluster Max-Dist) active learning acquisition strategy.

Ported from al4pde's distance_pool_based.py, adapted for yeast dataset.
Uses bmdal_reg's LargestClusterMaxDistSelectionMethod for diversity-based
selection: picks pool samples whose predicted-trajectory features are
farthest from the training set (max-dist within the largest cluster).

Feature = rollout N steps from ic, flatten (B, C, N+1) -> (B, C*(N+1)),
then random-sketch to sketch_dim for fast distance computation.
"""

import time
import numpy as np
# bmdal_reg 3.0.2 uses np.Inf which was removed in NumPy 2.0
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

import torch
import wandb
from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
from bmdal_reg.bmdal.features import Features
from bmdal_reg.bmdal.selection import (
    LargestClusterMaxDistSelectionMethod,
    MaxDistSelectionMethod,
    BaitFeatureSpaceSelectionMethod,
    KmeansppSelectionMethod,
    RandomSelectionMethod,
)

# 支持的选择方法映射 (命令行名 -> bmdal_reg 类)
SELECTION_METHODS = {
    'lcmd': LargestClusterMaxDistSelectionMethod,
    'max_dist': MaxDistSelectionMethod,
    'bait': BaitFeatureSpaceSelectionMethod,
    'kmeans++': KmeansppSelectionMethod,
    'random': RandomSelectionMethod,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def lcmd_select(train_mat, pool_mat, n_samples, sketch_dim=512, sel_with_train=True,
                selection_method='lcmd'):
    """
    通用特征多样性选择: sketch降维 + SelectionMethod.
    Ported from al4pde max_dist_selection (distance_pool_based.py:14-31).

    Args:
        selection_method: 选择算法名, 见 SELECTION_METHODS
    """
    train_mat = train_mat.to(device)
    pool_mat = pool_mat.to(device)
    n_features = train_mat.shape[1]

    train_features = Features(IdentityFeatureMap(n_features=n_features), TensorFeatureData(train_mat))
    pool_features = Features(IdentityFeatureMap(n_features=n_features), TensorFeatureData(pool_mat))

    # sketch 随机投影降维, 加速距离计算
    if sketch_dim is not None and sketch_dim < n_features:
        sketch = train_features.sketch_tfm(sketch_dim)
        train_features = sketch(train_features)
        pool_features = sketch(pool_features)

    # BAIT 特殊处理: 需要 scale_tfm + posterior_tfm 变换 (al4pde distance_pool_based.py:22-28)
    if selection_method == 'bait':
        scale_tfm = train_features.scale_tfm()
        train_features = scale_tfm(train_features)
        pool_features = scale_tfm(pool_features)
        posterior_tfm = train_features.posterior_tfm(sigma=1e-2, allow_kernel_space_posterior=False)
        train_features = posterior_tfm(train_features)
        pool_features = posterior_tfm(pool_features)

    sel_method_cls = SELECTION_METHODS.get(selection_method)
    if sel_method_cls is None:
        raise ValueError(f"Unknown selection_method: {selection_method}. "
                         f"Available: {list(SELECTION_METHODS.keys())}")

    # BAIT 需要 noise_sigma 参数 (用于得分计算与增量更新)
    if selection_method == 'bait':
        sel_method = sel_method_cls(
            pool_features=pool_features, train_features=train_features,
            sel_with_train=sel_with_train, noise_sigma=1e-2,
        )
    else:
        sel_method = sel_method_cls(
            pool_features=pool_features, train_features=train_features,
            sel_with_train=sel_with_train,
        )
    return sel_method.select(n_samples).detach().cpu()


class LCMDAcquisition:
    """
    LCMD active learning: 基于预测轨迹特征的多样性选择.

    流程:
    1. 对训练集用真实ic rollout N步, 展平为特征
    2. 对pool用统一默认ic rollout N步, 展平为特征
       (yeast ic 基本固定, 仅 CDH1T/CDH1/BUB2 少数分量随mutant变,
        统一ic足以反映参数对动力学的影响差异)
    3. sketch降维 + LargestClusterMaxDistSelectionMethod选择
    4. ODE模拟选中参数, 保存
    """

    def __init__(self, pool_params, pool_dicts, num_rollout_steps=20,
                 pred_batch_size=1024, sel_with_train=True, sketch_dim=512,
                 available_indices=None, selection_method='lcmd'):
        """
        Args:
            pool_params: (N, 141) numpy array of candidate parameters
            pool_dicts: list of dicts corresponding to pool_params
            num_rollout_steps: 特征提取 rollout 步数 (非完整轨迹, 20步足够区分轨迹形态)
            pred_batch_size: pool 推理 batch size (pool约120万, 调大省时)
            sel_with_train: 距离度量是否纳入训练集参考
            sketch_dim: 随机投影降维维度
            available_indices: 已过滤的可用池索引 (init阶段RandomPoolBased选完后传入剩余池);
                               None 则初始化为全量并shuffle
            selection_method: 选择算法名, 见 SELECTION_METHODS (lcmd/max_dist/bait/kmeans++/random)
        """
        self.pool_params = pool_params
        self.pool_dicts = pool_dicts
        self.num_total = len(pool_dicts)
        if available_indices is not None:
            self.available_indices = list(available_indices)
        else:
            self.available_indices = list(range(self.num_total))
            np.random.shuffle(self.available_indices)

        self.num_rollout_steps = num_rollout_steps
        self.pred_batch_size = pred_batch_size
        self.sel_with_train = sel_with_train
        self.sketch_dim = sketch_dim
        self.selection_method = selection_method

    def _extract_features(self, model, ic, params):
        """
        对给定ic和params做rollout, 返回展平的特征矩阵.
        ic: (B, C), params: (B, P) -> features: (B, C*(N+1))
        复用 model.roll_out (已有 nan_to_num + clamp 保护).
        """
        with torch.no_grad():
            pred = model.roll_out(ic, params, self.num_rollout_steps)  # (B, C, N+1)
        B = pred.shape[0]
        feat = pred.reshape(B, -1)
        # 防止特征中残留的 nan/inf 污染距离计算
        feat = torch.nan_to_num(feat, nan=0.0, posinf=1e3, neginf=-1e3)
        return feat.detach().cpu()

    def generate(self, model, param_gen, simulator, al_iter, save_dir, batch_size, train_loader=None):
        """
        LCMD选择 + ODE模拟 + 保存.

        Args:
            model: ARModelWrapper (用其 roll_out 提取特征)
            param_gen: PDEParamGenerator (未直接使用, 保留接口兼容)
            simulator: YeastSimulator
            al_iter: AL 迭代号
            save_dir: 保存目录
            batch_size: 本轮要选的样本数
            train_loader: 训练集 DataLoader (产出 4元组 ic, traj, params, t_idx)
        """
        t0 = time.time()

        if len(self.available_indices) < batch_size:
            print(f"Warning: Only {len(self.available_indices)} candidates left in pool. Selecting all remaining.")
            batch_size = len(self.available_indices)
        if batch_size == 0:
            return

        if train_loader is None:
            raise ValueError("LCMDAcquisition.generate requires train_loader to extract training features.")

        model.eval()

        # 1. 提取训练集特征 (用真实ic)
        train_features = []
        default_ic = None
        for batch in train_loader:
            ic, traj, params, t_idx = batch[0], batch[1], batch[2], batch[3]
            ic = ic.to(device)
            params = params.to(device)
            if default_ic is None:
                default_ic = ic[0:1].clone()  # 用第一个样本的ic作为pool评估的统一ic
            feat = self._extract_features(model, ic, params)
            train_features.append(feat)
        train_mat = torch.cat(train_features, dim=0)  # (N_train, C*(N+1))

        # 2. 提取pool特征 (用统一默认ic)
        avail_idx = self.available_indices
        pool_params_tensor = torch.tensor(self.pool_params[avail_idx], dtype=torch.float32)
        pool_features = []
        for i in range(0, len(pool_params_tensor), self.pred_batch_size):
            batch_params = pool_params_tensor[i:i + self.pred_batch_size].to(device)
            B = batch_params.shape[0]
            batch_ic = default_ic.expand(B, -1)  # 统一ic
            feat = self._extract_features(model, batch_ic, batch_params)
            pool_features.append(feat)
        pool_mat = torch.cat(pool_features, dim=0)  # (N_avail, C*(N+1))

        t_feat = time.time() - t0
        print(f"LCMD: feature extraction done in {t_feat:.1f}s. "
              f"train={train_mat.shape}, pool={pool_mat.shape}")

        # 3. 选择 (lcmd/max_dist/bait/kmeans++/random)
        t_sel = time.time()
        sel_idx_in_avail = lcmd_select(
            train_mat, pool_mat, batch_size,
            sketch_dim=self.sketch_dim, sel_with_train=self.sel_with_train,
            selection_method=self.selection_method,
        )
        print(f"{self.selection_method}: selection done in {time.time() - t_sel:.1f}s. Selected {len(sel_idx_in_avail)} samples.")

        # 4. 映射回原始pool索引并从可用池移除
        selected_indices = [avail_idx[i] for i in sel_idx_in_avail.tolist()]
        for idx in selected_indices:
            self.available_indices.remove(idx)

        wandb.log({
            "al/al_iter": al_iter,
            "al/sel_time": time.time() - t0,
            "al/pool_remaining": len(self.available_indices),
            "al/feat_time": t_feat,
        })

        # 5. ODE模拟 + 保存 (复用现有逻辑)
        print(f"Running Simulator on {batch_size} selected candidates...")
        sel_dicts = [self.pool_dicts[i] for i in selected_indices]
        sel_params = self.pool_params[selected_indices]
        trajs, valid_indices = simulator.simulate_batch(sel_dicts)
        if len(valid_indices) > 0:
            from budding_yeast_v2.data.param_gen import save_ar_batch
            valid_sel_params = sel_params[valid_indices]
            save_ar_batch(trajs, valid_sel_params, save_dir, al_iter, 0)

        print(f"LCMDAcquisition: Completed iter {al_iter}. Total time {time.time() - t0:.1f}s. "
              f"Remaining in pool: {len(self.available_indices)}")

        model.train()
