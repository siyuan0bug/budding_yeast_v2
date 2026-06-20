import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_eqns_torch import (
    eqns_torch, apply_event_torch, params_tensor_to_dict,
    detect_event_mask_torch, PARAM_KEYS
)


class PhysicsInformedLoss(nn.Module):
    def __init__(self, mean_y, std_y):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.bound_norm = -mean_y / std_y

    def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth):
        loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))
        negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))
        pred_fft = torch.fft.rfft(pred_norm, dim=-1)
        target_fft = torch.fft.rfft(target_norm, dim=-1)
        k = torch.fft.rfftfreq(pred_norm.size(-1), device=pred_norm.device)
        diff_fft = pred_fft - target_fft
        loss_spectral_smooth = torch.mean((torch.abs(diff_fft) ** 2) * (k ** 2).unsqueeze(0).unsqueeze(0), dim=(1, 2))
        total_loss = loss_mse + (lambda_penalty * negative_penalty) + (lambda_smooth * loss_spectral_smooth)
        return (total_loss * batch_weights).mean()


class MSEOnlyLoss(nn.Module):
    def __init__(self, mean_y=None, std_y=None):
        super().__init__()

    def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth):
        return F.mse_loss(pred_norm, target_norm)


class MSENegPenLoss(nn.Module):
    def __init__(self, mean_y, std_y):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.bound_norm = -mean_y / std_y

    def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth):
        loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))
        negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))
        total_loss = loss_mse + (lambda_penalty * negative_penalty)
        return (total_loss * batch_weights).mean()


class MSESmoothLoss(nn.Module):
    def __init__(self, mean_y=None, std_y=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth):
        loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))
        pred_fft = torch.fft.rfft(pred_norm, dim=-1)
        target_fft = torch.fft.rfft(target_norm, dim=-1)
        k = torch.fft.rfftfreq(pred_norm.size(-1), device=pred_norm.device)
        diff_fft = pred_fft - target_fft
        loss_spectral_smooth = torch.mean((torch.abs(diff_fft) ** 2) * (k ** 2).unsqueeze(0).unsqueeze(0), dim=(1, 2))
        total_loss = loss_mse + (lambda_smooth * loss_spectral_smooth)
        return (total_loss * batch_weights).mean()


class DilatedSobolevLoss(nn.Module):
    def __init__(self, mean_y, std_y, dilation_window=5):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.bound_norm = -mean_y / std_y
        self.dilation_window = dilation_window
        self.register_buffer('kernel', torch.ones(1, 1, dilation_window))

    def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty=None, lambda_smooth=None):
        loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))

        diff_target = target_norm[:, :, 1:] - target_norm[:, :, :-1]
        diff_pred = pred_norm[:, :, 1:] - pred_norm[:, :, :-1]

        is_jump_exact = (torch.abs(diff_target) > 0.2).float()

        B, V, T_minus_1 = is_jump_exact.shape
        is_jump_flat = is_jump_exact.reshape(B * V, 1, T_minus_1)
        padding = self.dilation_window // 2

        # 🌟 关键修改：强制将 kernel 转移到输入所在的设备
        device = pred_norm.device
        kernel = self.kernel.to(device)

        is_jump_dilated = F.conv1d(is_jump_flat, kernel, padding=padding)
        is_jump_dilated = (is_jump_dilated > 0).float().reshape(B, V, T_minus_1)

        is_smooth = 1.0 - is_jump_dilated

        jump_error = torch.abs(diff_pred - diff_target) * is_jump_exact
        loss_jump = jump_error.sum(dim=(1, 2)) / (is_jump_exact.sum(dim=(1, 2)) + 1e-8)

        smooth_error_1st = (diff_pred ** 2) * is_smooth
        loss_smooth_1st = smooth_error_1st.sum(dim=(1, 2)) / (is_smooth.sum(dim=(1, 2)) + 1e-8)

        diff2_pred = pred_norm[:, :, 2:] - 2 * pred_norm[:, :, 1:-1] + pred_norm[:, :, :-2]
        is_smooth_2nd = is_smooth[:, :, :-1] * is_smooth[:, :, 1:]
        smooth_error_2nd = (diff2_pred ** 2) * is_smooth_2nd
        loss_smooth_2nd = smooth_error_2nd.sum(dim=(1, 2)) / (is_smooth_2nd.sum(dim=(1, 2)) + 1e-8)

        negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))

        total_loss = 2.0 * loss_mse + 5.0 * loss_jump + 0.5 * loss_smooth_1st + 1.0 * loss_smooth_2nd + 10.0 * negative_penalty

        return (total_loss * batch_weights).mean()


class PINNResidualLoss(nn.Module):
    """
    PINN 残差损失：ODE 物理约束 + Event 跳跃条件。

    损失组成：
        L_total = L_data(MSE) + λ_phys · L_residual(ODE残差) + λ_jump · L_jump(跳跃条件)
                  + λ_pen · L_neg(负值惩罚)

    核心思想（PINN）：
        模型预测 û(t) 代入 ODE 方程 du/dt = f(u,p) 后，残差 r(t)=dû/dt-f(û,p) 应接近 0。
        在 event 突变点用掩码屏蔽残差，改用跳跃条件约束。

    Event 处理（掩码 + 跳跃条件）：
        - 连续区间：L_residual = mean(|| dû/dt - f(û,p) ||²)  (应用时间掩码)
        - Event 点：L_jump = mean(|| û(t⁺) - T_event(û(t⁻)) ||²)  (跳跃条件)
    """

    def __init__(self, mean_y, std_y, p_mean, p_std, t_max=210.0,
                 lambda_phys=0.1, lambda_jump=1.0, lambda_penalty=10.0,
                 event_window=4, event_tol=0.08, jump_ratio=0.3,
                 residual_subsample=1, event_weight=0.1):
        """
        Args:
            mean_y, std_y: (1, 39, 1) 状态变量归一化统计量
            p_mean, p_std: (1, 141) 参数归一化统计量
            t_max: 物理时间上限（分钟），用于计算 dt
            lambda_phys: ODE 残差损失权重
            lambda_jump: 跳跃条件损失权重
            lambda_penalty: 负值惩罚权重
            event_window: event 掩码窗口半径
            event_tol: 事件触发条件容差
            jump_ratio: 跳变检测阈值
            residual_subsample: 残差计算的时间步子采样率（降低计算量，1=全部）
            event_weight: event 区间的残差权重（软掩码，0=完全屏蔽，1=不屏蔽）
        """
        super().__init__()
        self.register_buffer('mean_y', mean_y.reshape(1, -1, 1))
        self.register_buffer('std_y', std_y.reshape(1, -1, 1))
        self.register_buffer('p_mean', p_mean.reshape(1, -1))
        self.register_buffer('p_std', p_std.reshape(1, -1))
        self.register_buffer('bound_norm', -mean_y.reshape(1, -1, 1) / std_y.reshape(1, -1, 1))

        self.t_max = t_max
        self.lambda_phys = lambda_phys
        self.lambda_jump = lambda_jump
        self.lambda_penalty = lambda_penalty
        self.event_window = event_window
        self.event_tol = event_tol
        self.jump_ratio = jump_ratio
        self.residual_subsample = residual_subsample
        self.event_weight = event_weight

        self.mse = nn.MSELoss(reduction='none')

    def _denormalize_y(self, y_norm):
        """反归一化状态变量到真实空间"""
        return y_norm * self.std_y + self.mean_y

    def _denormalize_p(self, p_norm):
        """反归一化参数到真实空间，并构建参数字典"""
        p_real = p_norm * self.p_std + self.p_mean  # (B, 141)
        extra = {'is_mutant_104': torch.zeros(p_real.shape[0], device=p_real.device)}
        # Vi20_active, BUB2_active, init_BUB2 需要从参数向量中取（它们在 PARAM_KEYS 中）
        p_dict = params_tensor_to_dict(p_real, extra)
        return p_dict

    def _compute_ode_residual_loss(self, pred_real, p_dict, mask):
        """
        计算 ODE 残差损失（带 event 软掩码）。

        策略：在连续区间使用完整权重，在 event 区间使用降低的权重（event_weight）。
        向量化实现：将 (B, T, 39) reshape 为 (B*T, 39) 一次调用 eqns_torch，消除 for 循环。

        Args:
            pred_real: (B, 39, T) 真实空间预测轨迹
            p_dict: 参数字典
            mask: (B, T) 事件掩码，True=连续区间

        Returns:
            loss_residual: 标量，软掩码后的 ODE 残差损失
        """
        B, V, T = pred_real.shape
        dt = self.t_max / (T - 1)

        # 中心差分计算 dû/dt
        du_dt = torch.zeros_like(pred_real)
        du_dt[:, :, 1:-1] = (pred_real[:, :, 2:] - pred_real[:, :, :-2]) / (2 * dt)
        du_dt[:, :, 0] = (pred_real[:, :, 1] - pred_real[:, :, 0]) / dt
        du_dt[:, :, -1] = (pred_real[:, :, -1] - pred_real[:, :, -2]) / dt

        # 向量化计算 f(u, p)：将 (B, 39, T) -> (B, T, 39) -> (B*T, 39)
        # eqns_torch 已对 batch 维向量化，只需 reshape 一次调用
        f_u = torch.zeros_like(pred_real)
        t_indices = list(range(0, T, self.residual_subsample))
        t_grid = torch.linspace(0, self.t_max, T, device=pred_real.device)

        # 收集所有需要计算的时间步，批量调用 eqns_torch
        # pred_real: (B, 39, T) -> 选取 t_indices -> (B, 39, len(t_indices))
        # -> transpose -> (B, len(t_indices), 39) -> reshape -> (B*len(t_indices), 39)
        n_t = len(t_indices)
        y_all = pred_real[:, :, t_indices]  # (B, 39, n_t)
        y_all = y_all.transpose(1, 2).reshape(B * n_t, V)  # (B*n_t, 39)

        # 参数字典扩展：每个 (B,) -> (B*n_t,)
        p_dict_expanded = {}
        for k, v in p_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B:
                p_dict_expanded[k] = v.unsqueeze(1).expand(B, n_t).reshape(B * n_t)
            else:
                p_dict_expanded[k] = v

        # 时间扩展
        t_all = t_grid[t_indices].unsqueeze(0).expand(B, n_t).reshape(B * n_t)

        # 一次调用计算所有时间步的 ODE 右端项
        f_all = eqns_torch(y_all, p_dict_expanded, t_all)  # (B*n_t, 39)

        # 填回 f_u: (B*n_t, 39) -> (B, n_t, 39) -> (B, 39, n_t)
        f_u[:, :, t_indices] = f_all.reshape(B, n_t, V).transpose(1, 2)

        # 残差
        residual = du_dt - f_u  # (B, 39, T)
        residual_sq = residual ** 2  # (B, 39, T)

        # 软掩码：连续区间权重=1.0，event 区间权重=event_weight
        weight = torch.ones_like(mask, dtype=pred_real.dtype, device=pred_real.device)
        weight[~mask] = self.event_weight
        weight = weight.unsqueeze(1).expand_as(residual_sq)  # (B, 39, T)

        # 子采样：只在计算了 f_u 的时间步上计算残差
        subsample_mask = torch.zeros(T, dtype=torch.bool, device=pred_real.device)
        subsample_mask[t_indices] = True
        weight = weight * subsample_mask.unsqueeze(0).unsqueeze(0).float()

        # 加权平均
        loss_residual = (residual_sq * weight).sum() / (weight.sum() + 1e-8)

        return loss_residual

    def _compute_jump_loss(self, pred_real, p_dict, mask):
        """
        计算跳跃条件损失：约束 event 区间内的状态变化方向。

        策略：在 event 区间（mask=False）内的所有相邻点对上，约束关键变量的
        变化方向符合 event 重置规则。由于无法精确判断 event 类型，对所有 event
        规则的方向约束取最小值（即至少有一种规则被满足）。

        具体约束（基于 event 重置规则的物理意义）：
        - event1 (分裂): BUD/SPN 应下降 → relu(d_bud), relu(d_spn) 应为 0
        - event2 (S期退出): ORI 应下降 → relu(d_ori) 应为 0
        - event3 (DNA复制): Vi20/BUB2 应上升 → relu(-d_vi20), relu(-d_bub2) 应为 0
        - event4 (纺锤体通过): Vi20 应下降, lte1 应上升 → relu(d_vi20), relu(-d_lte1) 应为 0

        Args:
            pred_real: (B, 39, T) 真实空间预测轨迹
            p_dict: 参数字典（值为 (B,) 张量）
            mask: (B, T) 事件掩码，True=连续区间，False=event 区间

        Returns:
            loss_jump: 标量，方向约束损失
        """
        B, V, T = pred_real.shape

        # 计算所有相邻点对的变化
        d = pred_real[:, :, 1:] - pred_real[:, :, :-1]  # (B, 39, T-1)

        d_bud = d[:, 34]   # (B, T-1)
        d_spn = d[:, 35]
        d_ori = d[:, 33]
        d_vi20 = d[:, 36]
        d_lte1 = d[:, 37]
        d_bub2 = d[:, 38]

        # 各 event 的方向违反量
        viol_e1 = torch.relu(d_bud) + torch.relu(d_spn)  # BUD/SPN 不应上升
        viol_e2 = torch.relu(d_ori)  # ORI 不应上升
        viol_e3 = torch.relu(-d_vi20) + torch.relu(-d_bub2)  # Vi20/BUB2 不应下降
        viol_e4 = torch.relu(d_vi20) + torch.relu(-d_lte1)  # Vi20 不应上升, lte1 不应下降

        # 取最小违反量（最可能匹配的 event 类型）
        all_viols = torch.stack([viol_e1, viol_e2, viol_e3, viol_e4], dim=-1)  # (B, T-1, 4)
        min_viol = all_viols.min(dim=-1)[0]  # (B, T-1)
        min_viol_sq = min_viol ** 2  # (B, T-1)

        # 在 event 区间内应用约束（mask=False 的区间）
        # mask 是 (B, T)，需要对应到 (B, T-1) 的相邻点对
        # 一个相邻点对 (t, t+1) 属于 event 区间，如果 mask[t] 或 mask[t+1] 为 False
        event_mask = (~mask[:, :-1]) | (~mask[:, 1:])  # (B, T-1)

        if event_mask.any():
            masked_viol = min_viol_sq[event_mask]
            loss_jump = masked_viol.mean()
        else:
            loss_jump = torch.tensor(0.0, device=pred_real.device)

        return loss_jump

    def forward(self, pred_norm, target_norm, p_norm, batch_weights,
                lambda_penalty=None, lambda_smooth=None):
        """
        PINN 损失前向传播。

        Args:
            pred_norm: (B, 39, T) 归一化预测轨迹
            target_norm: (B, 39, T) 归一化目标轨迹
            p_norm: (B, 141) 归一化参数向量
            batch_weights: (B,) 批次权重
            lambda_penalty: 负值惩罚权重（覆盖默认值）
            lambda_smooth: 未使用（接口兼容）

        Returns:
            total_loss: 标量
        """
        lam_pen = lambda_penalty if lambda_penalty is not None else self.lambda_penalty

        # 1. 数据损失 (MSE)
        loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))  # (B,)

        # 2. 负值惩罚
        negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))  # (B,)

        # 3. 反归一化到真实空间
        pred_real = self._denormalize_y(pred_norm)
        p_dict = self._denormalize_p(p_norm)

        # 4. 检测 event 掩码
        with torch.no_grad():
            mask = detect_event_mask_torch(
                pred_real, event_window=self.event_window,
                event_tol=self.event_tol, jump_ratio=self.jump_ratio
            )

        # 5. ODE 残差损失（连续区间）
        loss_residual = self._compute_ode_residual_loss(pred_real, p_dict, mask)

        # 6. 跳跃条件损失（event 点）
        loss_jump = self._compute_jump_loss(pred_real, p_dict, mask)

        # 7. 总损失
        total_loss = (loss_mse + lam_pen * negative_penalty) * batch_weights
        total_loss = total_loss.mean()
        total_loss = total_loss + self.lambda_phys * loss_residual + self.lambda_jump * loss_jump

        # 记录分量（供 lit_module 日志）
        self._last_loss_components = {
            'mse': loss_mse.mean().detach(),
            'residual': loss_residual.detach(),
            'jump': loss_jump.detach(),
            'neg_pen': negative_penalty.mean().detach(),
        }

        return total_loss


LOSS_REGISTRY = {
    'physics_informed': PhysicsInformedLoss,
    'mse_only': MSEOnlyLoss,
    'mse_negpen': MSENegPenLoss,
    'mse_smooth': MSESmoothLoss,
    't_smooth': DilatedSobolevLoss,
    'pinn_residual': PINNResidualLoss,
}


def get_curriculum_weights(epoch, max_epochs=120):
    if epoch < 15:
        return 0.0, 0.0
    elif epoch < 60:
        progress = (epoch - 15) / 45.0
        lam_pen = 10.0 * progress
        lam_sm = 0.5 * progress
        return lam_pen, lam_sm
    else:
        return 10.0, 0.5