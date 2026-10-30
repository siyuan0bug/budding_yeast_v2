"""strategies residual services extracted from the pre-package implementation."""
from budding_yeast_v2.utils.active_learning.oracle import PARAM_KEYS
import numpy as np
from budding_yeast_v2.data.dataset_archive import open_dataset
import os
import time
import torch


def _al4pde_residual_sample(self, trainer, pl_module, dm, device, current_epoch):
    """al4pde Residual (v2.4 优化版): 局部连续生化通道流速自适应相对物理残差 (Flux-Adaptive Relative Residual).

    四大优化彻底攻克物理残差失效瓶颈:
    1. 流速自适应相对物理残差 (Flux-Adaptive Relative Residual):
       分母采用局部流速与底噪保护 (|du/dt| + |f_biochem| + 0.1), 将残差严格约束在 [0, 1] 内,
       杜绝超表达大常数绑架打分, 极大提高对敲除阻滞失败/假振荡 (如 15 号突变体) 的捕捉灵敏度。
    2. 门控 1 (标称通量剪枝 Nominal Flux Pruning):
       提前筛除合成常数超过真实突变体极大值 3x 的病态发散候选。
    3. 门控 2 (模型预测浓度极值惩罚 Model-Predicted Max Penalty):
       若模型预测浓度已突破真实生理极大值 2x, 判定为潜在发散 Monster 予以 0.01x 惩罚。
    4. 配额达标动态缓冲填充 (Quota-Guaranteed Dynamic Filling):
       严格累积达标有效样本 (Round 1: 10,000, Round 2: 20,000), 根除数据饥饿。
    """
    if self._pool_size == 0:
        print("[al4pde_residual] 候选池已空, 跳过本轮 AL")
        return None

    from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE

    print(f"[al4pde_residual] 🚀 启动流速自适应相对物理残差评估 (池大小: {self._pool_size})...")
    t_start = time.time()
    pl_module.eval()

    # 初始化向量化 ODE 引擎
    param_keys = getattr(self, '_param_names_cached', None)
    if param_keys is None:
        pool_ds = self.al_pool_dataset or getattr(dm, 'dataset_path', None)
        if pool_ds and os.path.exists(pool_ds):
            npz_data = open_dataset(pool_ds, allow_pickle=True)
            if 'param_names' in npz_data:
                param_keys = list(npz_data['param_names'])
    if param_keys is None:
        param_keys = list(PARAM_KEYS)
    ode_engine = VectorizedYeastODE(param_keys).to(device)

    batch_size = 2500
    pool_indices = np.arange(self._pool_size)
    all_residual_scores = []
    n_steps = self._al4pde_t_grid.shape[2] if self._al4pde_t_grid is not None else 500
    dt = getattr(dm, 't_max', 210.0) / max(1, (n_steps - 1))

    # 提取真实生理极大值 (用于门控 2 预测浓度惩罚)
    if hasattr(dm, '_Y') and hasattr(dm, '_test_idx_real'):
        real_envelope = dm._Y[dm._test_idx_real].max(dim=-1)[0].max(dim=0)[0].to(device)
    else:
        Y_real_phys = dm.denormalize_y(dm.test_dataset_real.tensors[2].to(device))
        real_envelope = Y_real_phys.max(dim=-1)[0].max(dim=0)[0]
    dyn_max_thresh = torch.maximum(real_envelope * 2.0, torch.tensor(2.0, device=device))

    p_mean = dm.p_mean.to(device)
    p_std = dm.p_std.to(device)

    pruned_count = 0
    with torch.no_grad():
        for i in range(0, len(pool_indices), batch_size):
            end_idx = min(i + batch_size, len(pool_indices))
            b_idx = pool_indices[i:end_idx]

            X_batch = self._get_pool_X_batch(b_idx).to(device)
            P_batch = self._pool_P[b_idx].to(device)

            if self._pool_P_real is not None:
                P_real_batch = self._pool_P_real[b_idx].to(device)
            else:
                P_real_batch = P_batch * p_std + p_mean

            # 门控 1: 标称通量剪枝 (Nominal Flux Pruning)
            flux_mask = ode_engine.check_nominal_flux_mask(P_real_batch, max_individual_ratio=3.0, max_sum_ratio=3.0)
            pruned_count += (~flux_mask).sum().item()

            # 1. 模型预测 (归一化空间)
            preds_norm = pl_module.forward_ic_time(X_batch, P_batch)

            # 2. 真实物理空间反归一化 (浓度与参数)
            preds_real = dm.denormalize_y(preds_norm, device)

            # 3. 门控 2: 模型预测浓度极值先验 (Pred Max Monster Penalty)
            is_pred_monster = (preds_real[:, 1:39, :].max(dim=-1)[0] > dyn_max_thresh[1:39].unsqueeze(0)).any(dim=-1)

            # 4. 矢量化计算流速自适应相对物理残差 (bounded in [0, 1])
            b_scores = ode_engine.compute_residual_scores(preds_real, P_real_batch, dt, relative=True, eps=0.1)

            # 门控 2 施加 0.01x 惩罚
            b_scores = torch.where(is_pred_monster, b_scores * 0.01, b_scores)
            # 门控 1 直接屏蔽 (设为 -1.0)
            b_scores = torch.where(flux_mask, b_scores, torch.full_like(b_scores, -1.0))

            all_residual_scores.append(b_scores.cpu().numpy())

    pl_module.train()
    residual_scores = np.concatenate(all_residual_scores)
    eval_time = time.time() - t_start

    # 排序候选 (得分最高优先)
    ranked_idx = np.argsort(residual_scores)[::-1]
    valid_candidate_mask = residual_scores[ranked_idx] >= 0.0
    ranked_candidates = ranked_idx[valid_candidate_mask]

    n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)

    print(f"[al4pde_residual] ✅ 评估完成 (耗时 {eval_time:.2f}s, 速率 {self._pool_size/eval_time:.1f} 样本/s)")
    print(f"[al4pde_residual] 门控 1 (标称通量剪枝): 剔除 {pruned_count}/{self._pool_size} 极端候选 "
          f"({100*pruned_count/self._pool_size:.2f}%)")
    print(f"[al4pde_residual] 有效相对残差分布: min={residual_scores[ranked_candidates].min():.4f}, "
          f"median={np.median(residual_scores[ranked_candidates]):.4f}, max={residual_scores[ranked_candidates].max():.4f}")

    # 调用配额达标动态缓冲填充器，严格入库 n_select 个有效样本
    return self.make_selection_plan(ranked_candidates, dm, target_valid_num=n_select)


def _al4pde_residual_diversity_sample(self, trainer, pl_module, dm, device, current_epoch):
    """al4pde Residual-Diversity (v1.0): 相对物理残差门控 + 轨迹拓扑多样性混合采样.

    两阶段机制:
    1. 阶段 1 (残差门控): 在全候选池评估流速自适应相对残差与物理门控, 截取残差最高的 Top 50,000 候选子集 (高违规高信息量池);
    2. 阶段 2 (多样性细选): 在此 50,000 子集内提取 332 维轨迹形状感知特征与训练集对比,
       运行 bmdal MaxDist 贪心最大距离选择最正交、最分散的样本排序;
    3. 传入配额达标动态缓冲填充器, 严格入库 K_target 个有效样本。
    """
    if self._pool_size == 0:
        print("[al4pde_residual_diversity] 候选池已空, 跳过本轮 AL")
        return None

    from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE

    print(f"[al4pde_residual_diversity] 🚀 阶段 1: 启动相对物理残差门控粗筛 (池大小: {self._pool_size})...")
    t_start = time.time()
    pl_module.eval()

    param_keys = getattr(self, '_param_names_cached', None)
    if param_keys is None:
        pool_ds = self.al_pool_dataset or getattr(dm, 'dataset_path', None)
        if pool_ds and os.path.exists(pool_ds):
            npz_data = open_dataset(pool_ds, allow_pickle=True)
            if 'param_names' in npz_data:
                param_keys = list(npz_data['param_names'])
    if param_keys is None:
        param_keys = list(PARAM_KEYS)
    ode_engine = VectorizedYeastODE(param_keys).to(device)

    batch_size = 2500
    pool_indices = np.arange(self._pool_size)
    all_residual_scores = []
    n_steps = self._al4pde_t_grid.shape[2] if self._al4pde_t_grid is not None else 500
    dt = getattr(dm, 't_max', 210.0) / max(1, (n_steps - 1))

    if hasattr(dm, '_Y') and hasattr(dm, '_test_idx_real'):
        real_envelope = dm._Y[dm._test_idx_real].max(dim=-1)[0].max(dim=0)[0].to(device)
    else:
        Y_real_phys = dm.denormalize_y(dm.test_dataset_real.tensors[2].to(device))
        real_envelope = Y_real_phys.max(dim=-1)[0].max(dim=0)[0]
    dyn_max_thresh = torch.maximum(real_envelope * 2.0, torch.tensor(2.0, device=device))

    p_mean = dm.p_mean.to(device)
    p_std = dm.p_std.to(device)

    with torch.no_grad():
        for i in range(0, len(pool_indices), batch_size):
            end_idx = min(i + batch_size, len(pool_indices))
            b_idx = pool_indices[i:end_idx]

            X_batch = self._get_pool_X_batch(b_idx).to(device)
            P_batch = self._pool_P[b_idx].to(device)

            if self._pool_P_real is not None:
                P_real_batch = self._pool_P_real[b_idx].to(device)
            else:
                P_real_batch = P_batch * p_std + p_mean

            flux_mask = ode_engine.check_nominal_flux_mask(P_real_batch, max_individual_ratio=3.0, max_sum_ratio=3.0)
            preds_norm = pl_module.forward_ic_time(X_batch, P_batch)
            preds_real = dm.denormalize_y(preds_norm, device)
            is_pred_monster = (preds_real[:, 1:39, :].max(dim=-1)[0] > dyn_max_thresh[1:39].unsqueeze(0)).any(dim=-1)

            b_scores = ode_engine.compute_residual_scores(preds_real, P_real_batch, dt, relative=True, eps=0.1)
            b_scores = torch.where(is_pred_monster, b_scores * 0.01, b_scores)
            b_scores = torch.where(flux_mask, b_scores, torch.full_like(b_scores, -1.0))
            all_residual_scores.append(b_scores.cpu().numpy())

    pl_module.train()
    residual_scores = np.concatenate(all_residual_scores)

    # 提取残差最高的前 50,000 个合格候选子集
    ranked_idx = np.argsort(residual_scores)[::-1]
    valid_candidate_mask = residual_scores[ranked_idx] >= 0.0
    feasible_candidates = ranked_idx[valid_candidate_mask]
    subpool_size = min(50000, len(feasible_candidates))
    subpool_idx = feasible_candidates[:subpool_size]

    print(f"[al4pde_residual_diversity] ✅ 阶段 1 粗筛完成: 提取 Top {subpool_size} 高物理残差候选子集 "
          f"(残差区间: {residual_scores[subpool_idx[-1]]:.4f} ~ {residual_scores[subpool_idx[0]]:.4f})")

    # 阶段 2: 轨迹形状感知特征多样性 MaxDist 选择
    print(f"[al4pde_residual_diversity] 🚀 阶段 2: 在高残差子集中计算形状感知特征并进行 MaxDist 拓扑多样性覆盖...")
    subpool_features = self._compute_pool_features_shape_aware(pl_module, dm, device, indices=subpool_idx)
    train_features = self._compute_train_features_shape_aware(pl_module, dm, device)
    subpool_features, train_features = self._standardize_shape_features(subpool_features, train_features)

    # 在 50,000 个子集中选取全部并获得多样性优先序
    diverse_local_indices = self._bmdal_max_dist_select(subpool_features, train_features, len(subpool_idx))
    ordered_candidates = subpool_idx[diverse_local_indices]

    n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
    print(f"[al4pde_residual_diversity] 多样性排序完成, 进入配额达标动态缓冲填充 (目标有效样本: {n_select})...")
    return self.make_selection_plan(ordered_candidates, dm, target_valid_num=n_select)


