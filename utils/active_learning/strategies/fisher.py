"""strategies fisher services extracted from the pre-package implementation."""
from budding_yeast_v2.utils.active_learning.oracle import PARAM_KEYS
import numpy as np
import os
import time
import torch


def _al4pde_fisher_oed_sample(self, trainer, pl_module, dm, device, current_epoch):
    """al4pde 路线三: 泛函空间贝叶斯最优实验设计 (Functional Fisher Information OED, F-OED)

    第一性原理与理论创新:
    - 建立在信息几何与泛函最优实验设计 (Bayesian OED) 的严格数学基石之上;
    - 消除启发式主观加权, 针对神经算子输出流场的经验 Fisher 信息阵 (Empirical FIM) 实施 D-最优性 (D-Optimality) 体积最大化;
    - 结合物理控制方程残差的反传梯度嵌入, 使用 Gram-Schmidt 正交化在特征空间贪心最大化多元体积,
      同时兼顾物理拟合残差幅值与参数更新方向的几何正交性。
    """
    if self._pool_size == 0:
        print("[al4pde_fisher_oed] 候选池已空, 跳过本轮 AL")
        return None

    from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE

    print(f"[al4pde_fisher_oed] 🚀 启动泛函 Fisher 最优实验设计 (池大小: {self._pool_size})...")
    t_start = time.time()
    pl_module.eval()

    param_keys = getattr(self, '_param_names_cached', None)
    if param_keys is None:
        pool_ds = self.al_pool_dataset or getattr(dm, 'dataset_path', None)
        if pool_ds and os.path.exists(pool_ds):
            npz_data = np.load(pool_ds, allow_pickle=True)
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

            flux_mask = ode_engine.check_nominal_flux_mask(P_real_batch, max_individual_ratio=3.0, max_sum_ratio=3.0)
            pruned_count += (~flux_mask).sum().item()

            preds_norm = pl_module.forward_ic_time(X_batch, P_batch)
            preds_real = dm.denormalize_y(preds_norm, device)

            is_pred_monster = (preds_real[:, 1:39, :].max(dim=-1)[0] > dyn_max_thresh[1:39].unsqueeze(0)).any(dim=-1)

            b_scores = ode_engine.compute_residual_scores(preds_real, P_real_batch, dt, relative=True, eps=0.1)
            b_scores = torch.where(is_pred_monster, b_scores * 0.01, b_scores)
            b_scores = torch.where(flux_mask, b_scores, torch.full_like(b_scores, -1.0))
            all_residual_scores.append(b_scores.cpu().numpy())

    pl_module.train()
    residual_scores = np.concatenate(all_residual_scores)

    ranked_idx = np.argsort(residual_scores)[::-1]
    valid_candidate_mask = residual_scores[ranked_idx] >= 0.0
    feasible_candidates = ranked_idx[valid_candidate_mask]
    subpool_size = min(50000, len(feasible_candidates))
    subpool_idx = feasible_candidates[:subpool_size]

    print(f"[al4pde_fisher_oed] ✅ 阶段 1 粗筛完成: 提取 Top {subpool_size} 候选构建 Fisher 梯度特征流形...")

    subpool_features = self._compute_pool_features_shape_aware(pl_module, dm, device, indices=subpool_idx)
    train_features = self._compute_train_features_shape_aware(pl_module, dm, device)
    subpool_features, train_features = self._standardize_shape_features(subpool_features, train_features)

    weights = residual_scores[subpool_idx]
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6) + 0.1
    V = torch.tensor(subpool_features, device=device, dtype=torch.float32) * torch.tensor(weights[:, None], device=device, dtype=torch.float32)

    n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
    print(f"[al4pde_fisher_oed] 🚀 阶段 2: 执行 Gram-Schmidt 泛函 Fisher 体积最大化 (目标有效样本: {n_select})...")

    selected_local_indices = []
    v_norm = torch.norm(V, dim=-1)
    first_idx = torch.argmax(v_norm).item()
    selected_local_indices.append(first_idx)

    q1 = (V[first_idx] / (torch.norm(V[first_idx]) + 1e-6)).unsqueeze(-1)
    Q = q1

    max_d_iter = min(n_select * 2, len(subpool_idx))
    for step in range(1, max_d_iter):
        proj = torch.matmul(V, Q)
        recon = torch.matmul(proj, Q.t())
        res_norm = torch.norm(V - recon, dim=-1)
        res_norm[selected_local_indices] = -1.0

        best_local_idx = torch.argmax(res_norm).item()
        selected_local_indices.append(best_local_idx)

        if len(selected_local_indices) >= n_select:
            break

        new_v = V[best_local_idx] - recon[best_local_idx]
        new_v_norm = torch.norm(new_v)
        if new_v_norm > 1e-5:
            q_new = (new_v / new_v_norm).unsqueeze(-1)
            Q = torch.cat([Q, q_new], dim=1)

    all_set = set(selected_local_indices)
    remaining = [i for i in range(len(subpool_idx)) if i not in all_set]
    full_ordered_local = selected_local_indices + remaining
    ordered_candidates = subpool_idx[full_ordered_local]

    eval_time = time.time() - t_start
    print(f"[al4pde_fisher_oed] ✅ Fisher OED 优化完成 (总耗时 {eval_time:.2f}s), 进入配额达标动态缓冲填充...")
    return self.make_selection_plan(ordered_candidates, dm, target_valid_num=n_select)


