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
    PINN 残差损失 v2：ODE 物理约束 + Event 跳跃条件。

    v2 修复（v1 全面差于 mse_only，RelL2 0.234 vs 0.128）：
    1. Event 掩码用**真值**检测（v1 用预测，早期掩码错误导致梯度爆炸）
    2. 负值惩罚权重降低（v1 默认 10.0，压制 MSE 主任务）
    3. 跳跃条件用**真值方向**约束（v1 假设 event 类型，约束失效）
    4. Curriculum learning：早期纯 MSE，后期逐步加入物理约束

    损失组成：
        L_total = L_data(MSE) + λ_pen · L_neg + λ_phys · L_residual + λ_jump · L_jump

    核心思想（PINN）：
        模型预测 û(t) 代入 ODE 方程 du/dt = f(u,p) 后，残差 r(t)=dû/dt-f(û,p) 应接近 0。
        在 event 突变点用**真值掩码**屏蔽残差，改用真值方向的跳跃条件约束。
    """

    def __init__(self, mean_y, std_y, p_mean, p_std, t_max=210.0,
                 lambda_phys=0.01, lambda_jump=0.001, lambda_penalty=1.0,
                 event_window=4, event_tol=0.08, jump_ratio=0.3,
                 residual_subsample=1, event_weight=0.1,
                 warmup_epochs=30):
        """
        Args:
            mean_y, std_y: (1, 39, 1) 状态变量归一化统计量
            p_mean, p_std: (1, 141) 参数归一化统计量
            t_max: 物理时间上限（分钟），用于计算 dt
            lambda_phys: ODE 残差损失权重（v2 默认 0.01，弱正则）
            lambda_jump: 跳跃条件损失权重（v2 默认 0.001，弱正则）
            lambda_penalty: 负值惩罚权重（v2 默认 1.0，v1 为 10.0）
            event_window: event 掩码窗口半径
            event_tol: 事件触发条件容差
            jump_ratio: 跳变检测阈值
            residual_subsample: 残差计算的时间步子采样率
            event_weight: event 区间的残差权重（软掩码）
            warmup_epochs: 物理约束 warmup 轮数（前 N 轮纯 MSE）
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
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0

        self.mse = nn.MSELoss(reduction='none')

    def set_epoch(self, epoch):
        """设置当前 epoch，用于 curriculum learning"""
        self._current_epoch = epoch

    def _denormalize_y(self, y_norm):
        """反归一化状态变量到真实空间"""
        return y_norm * self.std_y + self.mean_y

    def _denormalize_p(self, p_norm):
        """反归一化参数到真实空间，并构建参数字典"""
        p_real = p_norm * self.p_std + self.p_mean  # (B, 141)
        extra = {'is_mutant_104': torch.zeros(p_real.shape[0], device=p_real.device)}
        p_dict = params_tensor_to_dict(p_real, extra)
        return p_dict

    def _get_physics_weight(self):
        """
        Curriculum learning: 物理约束权重调度。
        前 warmup_epochs 轮纯 MSE，之后线性增长到目标权重。
        """
        if self._current_epoch < self.warmup_epochs:
            return 0.0, 0.0
        progress = min(1.0, (self._current_epoch - self.warmup_epochs) / 30.0)
        return self.lambda_phys * progress, self.lambda_jump * progress

    def _compute_ode_residual_loss(self, pred_real, p_dict, mask):
        """
        计算 ODE 残差损失（带 event 软掩码，归一化到与 MSE 可比量级）。

        策略：
        1. 在真实空间计算残差 r = dû/dt - f(û, p)
        2. 将残差归一化到归一化空间：r_norm = r / std_y
        3. 连续区间权重=1.0，event 区间权重=event_weight

        Args:
            pred_real: (B, 39, T) 真实空间预测轨迹
            p_dict: 参数字典
            mask: (B, T) 事件掩码（基于真值检测），True=连续区间

        Returns:
            loss_residual: 标量，归一化后的软掩码 ODE 残差损失
        """
        B, V, T = pred_real.shape
        dt = self.t_max / (T - 1)

        # 中心差分计算 dû/dt（真实空间）
        du_dt = torch.zeros_like(pred_real)
        du_dt[:, :, 1:-1] = (pred_real[:, :, 2:] - pred_real[:, :, :-2]) / (2 * dt)
        du_dt[:, :, 0] = (pred_real[:, :, 1] - pred_real[:, :, 0]) / dt
        du_dt[:, :, -1] = (pred_real[:, :, -1] - pred_real[:, :, -2]) / dt

        # 向量化计算 f(u, p)
        f_u = torch.zeros_like(pred_real)
        t_indices = list(range(0, T, self.residual_subsample))
        t_grid = torch.linspace(0, self.t_max, T, device=pred_real.device)

        n_t = len(t_indices)
        y_all = pred_real[:, :, t_indices]  # (B, 39, n_t)
        y_all = y_all.transpose(1, 2).reshape(B * n_t, V)  # (B*n_t, 39)

        p_dict_expanded = {}
        for k, v in p_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B:
                p_dict_expanded[k] = v.unsqueeze(1).expand(B, n_t).reshape(B * n_t)
            else:
                p_dict_expanded[k] = v

        t_all = t_grid[t_indices].unsqueeze(0).expand(B, n_t).reshape(B * n_t)
        f_all = eqns_torch(y_all, p_dict_expanded, t_all)  # (B*n_t, 39)
        f_u[:, :, t_indices] = f_all.reshape(B, n_t, V).transpose(1, 2)

        # 残差（真实空间）归一化到归一化空间
        residual = du_dt - f_u  # (B, 39, T)
        residual_norm = residual / self.std_y  # (B, 39, T)
        residual_sq = residual_norm ** 2

        # 软掩码：连续区间权重=1.0，event 区间权重=event_weight
        weight = torch.ones_like(mask, dtype=pred_real.dtype, device=pred_real.device)
        weight[~mask] = self.event_weight
        weight = weight.unsqueeze(1).expand_as(residual_sq)  # (B, 39, T)

        # 子采样掩码
        subsample_mask = torch.zeros(T, dtype=torch.bool, device=pred_real.device)
        subsample_mask[t_indices] = True
        weight = weight * subsample_mask.unsqueeze(0).unsqueeze(0).float()

        loss_residual = (residual_sq * weight).sum() / (weight.sum() + 1e-8)
        return loss_residual

    def _compute_jump_loss(self, pred_real, target_real, mask):
        """
        计算跳跃条件损失 v2：用**真值方向**约束预测。

        v2 改进（v1 假设 event 类型取最小违反量，约束失效）：
        - 在 event 区间，用真值的变化方向约束预测的变化方向
        - 如果真值下降，预测不应上升；如果真值上升，预测不应下降
        - 损失 = relu(预测变化 × (-真值变化方向))²

        Args:
            pred_real: (B, 39, T) 真实空间预测轨迹
            target_real: (B, 39, T) 真实空间真值轨迹
            mask: (B, T) 事件掩码（基于真值检测），True=连续区间

        Returns:
            loss_jump: 标量，方向约束损失
        """
        B, V, T = pred_real.shape

        # 计算相邻点变化（归一化空间）
        d_pred = (pred_real[:, :, 1:] - pred_real[:, :, :-1]) / self.std_y  # (B, 39, T-1)
        d_target = (target_real[:, :, 1:] - target_real[:, :, :-1]) / self.std_y  # (B, 39, T-1)

        # 真值方向符号：+1 上升, -1 下降, 0 平稳
        target_dir = torch.sign(d_target)  # (B, 39, T-1)

        # 方向违反：预测变化与真值方向相反
        # 如果真值下降 (dir=-1)，预测不应上升 → viol = relu(d_pred)
        # 如果真值上升 (dir=+1)，预测不应下降 → viol = relu(-d_pred)
        viol = torch.relu(-target_dir * d_pred)  # (B, 39, T-1)
        viol_sq = viol ** 2

        # 在 event 区间内应用约束
        event_mask = (~mask[:, :-1]) | (~mask[:, 1:])  # (B, T-1)
        weight = event_mask.unsqueeze(1).expand_as(viol_sq).float()  # (B, 39, T-1)

        if weight.sum() > 0:
            loss_jump = (viol_sq * weight).sum() / (weight.sum() + 1e-8)
        else:
            loss_jump = torch.tensor(0.0, device=pred_real.device)

        return loss_jump

    def forward(self, pred_norm, target_norm, p_norm, batch_weights,
                lambda_penalty=None, lambda_smooth=None):
        """
        PINN 损失 v2 前向传播。

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

        # 1. 数据损失 (MSE) —— 始终是主任务
        loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))  # (B,)

        # 2. 负值惩罚（v2 权重降低到 1.0）
        negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))  # (B,)

        # 基础损失 = MSE + 负值惩罚
        base_loss = (loss_mse + lam_pen * negative_penalty) * batch_weights
        total_loss = base_loss.mean()

        # 3. Curriculum learning: warmup 期纯 MSE，之后逐步加入物理约束
        lam_phys_eff, lam_jump_eff = self._get_physics_weight()

        if lam_phys_eff > 0 or lam_jump_eff > 0:
            # 反归一化到真实空间
            pred_real = self._denormalize_y(pred_norm)
            target_real = self._denormalize_y(target_norm)
            p_dict = self._denormalize_p(p_norm)

            # 4. Event 掩码用**真值**检测（v1 用预测，早期掩码错误）
            with torch.no_grad():
                mask = detect_event_mask_torch(
                    target_real, event_window=self.event_window,
                    event_tol=self.event_tol, jump_ratio=self.jump_ratio
                )

            # 5. ODE 残差损失（连续区间）
            if lam_phys_eff > 0:
                loss_residual = self._compute_ode_residual_loss(pred_real, p_dict, mask)
                total_loss = total_loss + lam_phys_eff * loss_residual

            # 6. 跳跃条件损失（event 点，用真值方向约束）
            if lam_jump_eff > 0:
                loss_jump = self._compute_jump_loss(pred_real, target_real, mask)
                total_loss = total_loss + lam_jump_eff * loss_jump

        # 记录分量
        self._last_loss_components = {
            'mse': loss_mse.mean().detach(),
            'neg_pen': negative_penalty.mean().detach(),
            'phys_weight': torch.tensor(lam_phys_eff),
            'jump_weight': torch.tensor(lam_jump_eff),
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