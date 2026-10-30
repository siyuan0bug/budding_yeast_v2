"""strategies events services extracted from the pre-package implementation."""
import torch.nn.functional as F
from budding_yeast_v2.utils.active_learning.oracle import PARAM_KEYS
import numpy as np
from budding_yeast_v2.data.dataset_archive import open_dataset
import os
import time
import torch


def _al4pde_bifurcation_sample(self, trainer, pl_module, dm, device, current_epoch):
    """al4pde 路线一: 相空间吸引子与分岔临界流形主动学习 (Bifurcation-Manifold Active Learning, BM-AL)

    第一性原理与理论创新:
    - 细胞周期本质是相空间中的稳定极限环吸引子 (Limit Cycle Attractor);
    - 突变致死阻滞对应定点吸引子 (Fixed-Point Arrest);
    - 真实突变体绝大多数恰好分布在这两条相变分岔超曲面 (Codimension-1 Bifurcation Manifold) 之上;
    - 本算法通过检测核心细胞周期蛋白 (CLB2, SIC1, CLN2) 的互斥振荡活力、临界减速 (Critical Slowing Down) 漂移、
      以及中性相平衡, 准确定位相变分岔超曲面, 杜绝纯残差对死细胞阻滞态的病态过度采样。
    """
    if self._pool_size == 0:
        print("[al4pde_bifurcation] 候选池已空, 跳过本轮 AL")
        return None

    from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE

    print(f"[al4pde_bifurcation] 🚀 启动分岔临界流形探测 (池大小: {self._pool_size})...")
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
    all_bif_scores = []
    n_steps = self._al4pde_t_grid.shape[2] if self._al4pde_t_grid is not None else 500

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

            CLB2 = preds_real[:, 2, :]
            SIC1 = preds_real[:, 4, :]
            CLN2 = preds_real[:, 1, :]

            # 振荡活力 (Oscillatory Vitality)
            amp_clb2 = CLB2.max(dim=-1)[0] - CLB2.min(dim=-1)[0]
            amp_sic1 = SIC1.max(dim=-1)[0] - SIC1.min(dim=-1)[0]
            osc_vitality = (amp_clb2 * amp_sic1) / (amp_clb2 * amp_sic1 + 0.04)

            # 临界漂移 (Phase Drift & Critical Slowing Down)
            mid = n_steps // 2
            mean_1 = preds_real[:, 1:33, :mid].mean(dim=-1)
            mean_2 = preds_real[:, 1:33, mid:].mean(dim=-1)
            drift = torch.norm(mean_2 - mean_1, dim=-1) / (torch.norm(mean_1, dim=-1) + 1e-4)

            bif_scores = osc_vitality * torch.exp(-torch.abs(drift - 0.25) / 0.5)

            b_scores = torch.where(is_pred_monster, bif_scores * 0.01, bif_scores)
            b_scores = torch.where(flux_mask, b_scores, torch.full_like(b_scores, -1.0))
            all_bif_scores.append(b_scores.cpu().numpy())

    pl_module.train()
    bif_scores_all = np.concatenate(all_bif_scores)
    eval_time = time.time() - t_start

    ranked_idx = np.argsort(bif_scores_all)[::-1]
    valid_candidate_mask = bif_scores_all[ranked_idx] >= 0.0
    ranked_candidates = ranked_idx[valid_candidate_mask]

    n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
    print(f"[al4pde_bifurcation] ✅ 分岔流形评估完成 (耗时 {eval_time:.2f}s, 速率 {self._pool_size/eval_time:.1f} 样本/s)")
    print(f"[al4pde_bifurcation] 门控剪枝: 剔除 {pruned_count}/{self._pool_size} 异常候选")
    print(f"[al4pde_bifurcation] 最高分岔临界度候选: min={bif_scores_all[ranked_candidates[-1]]:.4f}, max={bif_scores_all[ranked_candidates[0]]:.4f}")
    return self.make_selection_plan(ranked_candidates, dm, target_valid_num=n_select)


def _al4pde_bifurcation_entropy_sample(self, trainer, pl_module, dm, device, current_epoch):
    """al4pde 分岔相变信息熵主动学习 (Bifurcation Shannon Entropy Active Learning, BSE-AL)

    第一性原理与理论创新:
    - 酵母细胞周期动力系统具有陡峭的高维相变分岔超曲面 (Limit Cycle vs. Fixed-Point Arrest);
    - 在相变边界附近，动力学对参数微小扰动极其敏感，系统在增殖振荡 (Pattern A/B) 与致死停滞 (Pattern C) 之间发生剧烈拓扑跃迁;
    - 通过模型自带的分岔判别头 P(C | p)（或由训练集在线训练的高速分岔判别器），对 100 万未标注候选池的参数批次计算香农分岔信息熵:
        H(p) = - \\sum_{k \\in {A, B, C}} P_k(p) * ln(P_k(p) + 1e-12)
    - 挑选 H(p) 最高的临界样本（即模型在分类决策上最不确定、最接近分岔相变临界面的样本）调用真实 ODE 求解器打标并增补进训练集;
    - 这样能用最少的样本代价，把高维相变边界刻画得极其陡峭精准，彻底消除盲目起振或虚假阻滞的泛化缺陷。
    """
    if self._pool_size == 0:
        print("[al4pde_bifurcation_entropy] 候选池已空, 跳过本轮 AL")
        return None

    from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE

    print(f"[al4pde_bifurcation_entropy] 🚀 启动分岔相变信息熵主动学习评估 (池大小: {self._pool_size})...")
    t_start = time.time()

    # 1. 获取/构建分岔分类器
    classifier_fn = None
    if pl_module is not None and hasattr(pl_module, 'model') and hasattr(pl_module.model, 'pattern_head'):
        pl_module.eval()
        classifier_fn = lambda p_batch: pl_module.model.pattern_head(p_batch)
        print("[al4pde_bifurcation_entropy] 使用主模型内置 pattern_head 分岔预测头")
    else:
        # 在线训练极速 2 层 MLP 分类器
        print("[al4pde_bifurcation_entropy] 主模型无 pattern_head，在线训练高速分岔分类器...")
        t_clf_start = time.time()
        train_tensors = dm.train_dataset_synth.tensors
        train_P = train_tensors[1].to(device)
        if len(train_tensors) >= 4:
            train_labels = train_tensors[3].to(device)
        else:
            train_Y = train_tensors[2].to(device)
            amp = train_Y[:, 2, :].max(dim=-1)[0] - train_Y[:, 2, :].min(dim=-1)[0]
            train_labels = torch.where(amp > 0.5, torch.zeros_like(amp, dtype=torch.long), 
                                       torch.where(amp > 0.1, torch.ones_like(amp, dtype=torch.long), 
                                                   torch.full_like(amp, 2, dtype=torch.long)))

        mlp_clf = torch.nn.Sequential(
            torch.nn.Linear(train_P.shape[1], 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 3)
        ).to(device)
        optimizer = torch.optim.AdamW(mlp_clf.parameters(), lr=3e-3, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        mlp_clf.train()
        for _ in range(15):
            logits = mlp_clf(train_P)
            loss = criterion(logits, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mlp_clf.eval()
        classifier_fn = lambda p_batch: mlp_clf(p_batch)
        print(f"[al4pde_bifurcation_entropy] 高速分类器训练完成, 耗时 {time.time() - t_clf_start:.3f}s")

    # 2. 准备 ODE 物理引擎用于通量异常初筛
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

    batch_size = 10000
    pool_indices = np.arange(self._pool_size)
    all_entropies = []
    p_mean = dm.p_mean.to(device)
    p_std = dm.p_std.to(device)
    pruned_count = 0

    with torch.no_grad():
        for i in range(0, len(pool_indices), batch_size):
            end_idx = min(i + batch_size, len(pool_indices))
            b_idx = pool_indices[i:end_idx]

            P_batch = self._pool_P[b_idx].to(device)

            if self._pool_P_real is not None:
                P_real_batch = self._pool_P_real[b_idx].to(device)
            else:
                P_real_batch = P_batch * p_std + p_mean

            flux_mask = ode_engine.check_nominal_flux_mask(P_real_batch, max_individual_ratio=3.0, max_sum_ratio=3.0)
            pruned_count += (~flux_mask).sum().item()

            logits = classifier_fn(P_batch)
            probs = F.softmax(logits, dim=-1)

            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
            entropy = torch.where(flux_mask, entropy, torch.full_like(entropy, -1.0))
            all_entropies.append(entropy.cpu().numpy())

    if pl_module is not None:
        pl_module.train()
    entropies_all = np.concatenate(all_entropies)
    eval_time = time.time() - t_start

    ranked_idx = np.argsort(entropies_all)[::-1]
    valid_candidate_mask = entropies_all[ranked_idx] >= 0.0
    ranked_candidates = ranked_idx[valid_candidate_mask]

    n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
    max_ent = entropies_all[ranked_candidates[0]] if len(ranked_candidates) > 0 else 0.0
    print(f"[al4pde_bifurcation_entropy] ✅ 分岔信息熵评估完成 (耗时 {eval_time:.2f}s, 吞吐量 {self._pool_size/eval_time:.1f} 样本/s)")
    print(f"[al4pde_bifurcation_entropy] 门控剪枝: 剔除 {pruned_count}/{self._pool_size} 异常参数候选")
    print(f"[al4pde_bifurcation_entropy] 选中前 {n_select} 个最高熵临界样本: 最大熵={max_ent:.4f} (理论上限 ln(3)≈1.0986)")

    return self.make_selection_plan(ranked_candidates, dm, target_valid_num=n_select)


def _al4pde_event_entropy_sample(self, trainer, pl_module, dm, device, current_epoch):
    """al4pde 路线二: 生物检验点离散事件转移熵 (Hybrid-Event Transition Entropy, HETE-AL)

    第一性原理与理论创新:
    - 酵母细胞周期是包含连续化学动力学与离散检验点跳跃的混合自动机 (Hybrid Automaton);
    - 4 大关键生命事件: G1/S 启动 (Start)、极性芽体 (Bud)、纺锤体组装检验 (SAC)、出核网络 (MEN);
    - 神经算子在检验点相变处的决策模糊度对应于事件发生概率密度的最大信息熵 (Maximum Transition Entropy);
    - 本算法通过计算 4 大检验点激活指示函数沿时间轴的跳跃导数分布与转移熵, 专门挑选时序决策处于临界犹豫态的核心突变体。
    """
    if self._pool_size == 0:
        print("[al4pde_event_entropy] 候选池已空, 跳过本轮 AL")
        return None

    from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE

    print(f"[al4pde_event_entropy] 🚀 启动检验点离散事件转移熵评估 (池大小: {self._pool_size})...")
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
    all_event_scores = []

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

            s_start = torch.sigmoid((preds_real[:, 1, :] - 0.15) * 15.0) * torch.sigmoid((0.15 - preds_real[:, 4, :]) * 15.0)
            s_bud   = torch.sigmoid((preds_real[:, 2, :] - 0.20) * 15.0)
            s_sac   = torch.sigmoid((preds_real[:, 19, :] - 0.25) * 15.0)
            s_men   = torch.sigmoid((preds_real[:, 21, :] - 0.30) * 15.0) * torch.sigmoid((0.15 - preds_real[:, 2, :]) * 15.0)

            event_entropies = []
            for s_ev in [s_start, s_bud, s_sac, s_men]:
                rho = torch.abs(s_ev[:, 1:] - s_ev[:, :-1])
                sum_rho = rho.sum(dim=-1, keepdim=True) + 1e-6
                p = rho / sum_rho
                H = - torch.sum(p * torch.log(p + 1e-12), dim=-1)
                active_w = torch.clamp(sum_rho.squeeze(-1) / 4.0, 0.0, 1.0)
                event_entropies.append(H * active_w)

            ev_scores = torch.stack(event_entropies, dim=-1).sum(dim=-1)

            ev_scores = torch.where(is_pred_monster, ev_scores * 0.01, ev_scores)
            ev_scores = torch.where(flux_mask, ev_scores, torch.full_like(ev_scores, -1.0))
            all_event_scores.append(ev_scores.cpu().numpy())

    pl_module.train()
    event_scores_all = np.concatenate(all_event_scores)
    eval_time = time.time() - t_start

    ranked_idx = np.argsort(event_scores_all)[::-1]
    valid_candidate_mask = event_scores_all[ranked_idx] >= 0.0
    ranked_candidates = ranked_idx[valid_candidate_mask]

    n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
    print(f"[al4pde_event_entropy] ✅ 检验点转移熵评估完成 (耗时 {eval_time:.2f}s, 速率 {self._pool_size/eval_time:.1f} 样本/s)")
    print(f"[al4pde_event_entropy] 门控剪枝: 剔除 {pruned_count}/{self._pool_size} 异常候选")
    print(f"[al4pde_event_entropy] 最高转移熵候选: min={event_scores_all[ranked_candidates[-1]]:.4f}, max={event_scores_all[ranked_candidates[0]]:.4f}")
    return self.make_selection_plan(ranked_candidates, dm, target_valid_num=n_select)


