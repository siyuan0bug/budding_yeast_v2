import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_eqns_torch import (
    eqns_torch, apply_event_torch, params_tensor_to_dict,
    detect_event_mask_torch, PARAM_KEYS
)
from .biological_guard import CONTINUOUS_VARS, EVENT_VARS


class PhysicsInformedLoss(nn.Module):
    def __init__(self, mean_y, std_y):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.bound_norm = -mean_y / std_y

    def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth):
        loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))
        negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))
        pred_fft = torch.fft.rfft(pred_norm.float(), dim=-1)
        target_fft = torch.fft.rfft(target_norm.float(), dim=-1)
        k = torch.fft.rfftfreq(pred_norm.size(-1), device=pred_norm.device)
        diff_fft = pred_fft - target_fft
        loss_spectral_smooth = torch.mean((torch.abs(diff_fft) ** 2) * (k ** 2).unsqueeze(0).unsqueeze(0), dim=(1, 2))
        total_loss = loss_mse + (lambda_penalty * negative_penalty) + (lambda_smooth * loss_spectral_smooth)
        return (total_loss * batch_weights).mean()


class MSEOnlyLoss(nn.Module):
    def __init__(self, mean_y=None, std_y=None):
        super().__init__()

    def forward(self, pred_norm, target_norm, batch_weights=None, lambda_penalty=None, lambda_smooth=None):
        return F.mse_loss(pred_norm, target_norm)


class RelativeL2Loss(nn.Module):
    """
    Safe Relative L2 Loss across all 39 protein channels:
    L = (1 / (B * V)) * sum_{b, v} [ ||Y_{b,v} - Y_pred_{b,v}||_2 / sqrt(||Y_{b,v}||_2^2 + eps^2) ]
    - Guarantees denominator is strictly positive and non-zero (smoothly >= eps).
    - Prevents large-scale variables (e.g. ORI/BUD) from dominating small-concentration cyclins.
    - Clamps relative loss to max_clamp (default 10.0) to eliminate gradient spikes.
    """
    def __init__(self, mean_y=None, std_y=None, eps=0.1, max_clamp=10.0):
        super().__init__()
        self.eps = eps
        self.max_clamp = max_clamp

    def forward(self, pred_norm, target_norm, batch_weights=None, lambda_penalty=None, lambda_smooth=None):
        # pred_norm, target_norm: (B, V, T)
        diff_l2 = torch.norm(pred_norm - target_norm, p=2, dim=-1) # (B, V)
        target_energy = torch.sqrt(torch.sum(target_norm ** 2, dim=-1) + (self.eps ** 2)) # (B, V)
        rel_l2 = diff_l2 / target_energy # (B, V)
        if self.max_clamp is not None:
            rel_l2 = torch.clamp(rel_l2, max=self.max_clamp)
        return rel_l2.mean()


class HybridMSERelLoss(nn.Module):
    """
    Hybrid Loss balancing waveform correlation and cyclin channel fairness:
    L = MSE(pred, target) + lambda_rel * RelativeL2Loss(pred, target)
    - MSE preserves sharp oscillatory peaks and high Pearson correlation (Corr >= 0.88).
    - Relative L2 prevents cyclin starvation, penalizing rescue bifurcation drifts (MAE <= 0.35).
    """
    def __init__(self, mean_y=None, std_y=None, lambda_rel=0.2, eps=0.1, max_clamp=10.0):
        super().__init__()
        self.lambda_rel = lambda_rel
        self.rel_loss = RelativeL2Loss(eps=eps, max_clamp=max_clamp)

    def forward(self, pred_norm, target_norm, batch_weights=None, lambda_penalty=None, lambda_smooth=None, **kwargs):
        loss_mse = F.mse_loss(pred_norm, target_norm)
        loss_rel = self.rel_loss(pred_norm, target_norm)
        return loss_mse + self.lambda_rel * loss_rel


class GradNormHybridLoss(nn.Module):
    """
    39-Channel Full GradNorm Hybrid Loss (方案 B: 39 通道全量微观梯度自适应平衡):
    L_total = (1 / 39) * sum_{v=0}^{38} w_v(t) * [ MSE_v + lambda_rel * RelL2_v ]

    核心数学设计:
    1. 单次 Autograd 极速求导技巧 (Single-Call Decoupled Norm Trick):
       因为通道损失 L_v 仅依赖于 pred[:, v, :]，通道间在输出层完全正交解耦。
       对 channel_losses.sum() 仅需 1 次 autograd.grad 即可解析出全部 39 个通道各自的精确 L2 梯度范数 (<2ms)。
    2. 自适应相对逆训练进度 r_v(t):
       对收敛滞后通道 (r_v > 1) 动态提高目标梯度强度 G_target_v。
    3. 闭式 EMA 权重平滑更新 (Closed-form EMA Update):
       无需额外 optimizer 与二次反向传播，完全保证 Lightning DDP / 混合精度训练安全，
       并严格维持 mean(task_weights) == 1.0，与基准 hybrid_mse_rel 尺度无缝兼容。
    """
    def __init__(self, mean_y=None, std_y=None, num_vars=39, lambda_rel=0.2, eps=0.1, max_clamp=10.0,
                 alpha=1.0, eta=0.025, w_min=0.1, w_max=5.0, warmup_steps=50):
        super().__init__()
        self.num_vars = num_vars
        self.lambda_rel = lambda_rel
        self.eps = eps
        self.max_clamp = max_clamp
        self.alpha = alpha
        self.eta = eta
        self.w_min = w_min
        self.w_max = w_max
        self.warmup_steps = warmup_steps

        # 注册非持久性缓冲区，支持 DDP 与设备自动迁移
        self.register_buffer('task_weights', torch.ones(num_vars, dtype=torch.float32))
        self.register_buffer('initial_losses', torch.zeros(num_vars, dtype=torch.float32))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('is_initialized', torch.tensor(False, dtype=torch.bool))

        # 用于存储最近一个 step 的诊断指标供 Lightning 记录
        self._last_metrics = {}

    def forward(self, pred_norm, target_norm, batch_weights=None, lambda_penalty=None, lambda_smooth=None,
                update_weights=True, **kwargs):
        # pred_norm, target_norm: (B, V, T)
        B, V, T = pred_norm.shape
        diff = pred_norm - target_norm

        # 1. 向量化计算 39 个通道各自的 MSE: (V,)
        mse_v = torch.mean(diff ** 2, dim=(0, 2))

        # 2. 向量化计算 39 个通道各自的 Relative L2: (V,)
        diff_l2 = torch.norm(diff, p=2, dim=-1)  # (B, V)
        target_energy = torch.sqrt(torch.sum(target_norm ** 2, dim=-1) + (self.eps ** 2))  # (B, V)
        rel_l2_b_v = diff_l2 / target_energy  # (B, V)
        if self.max_clamp is not None:
            rel_l2_b_v = torch.clamp(rel_l2_b_v, max=self.max_clamp)
        rel_l2_v = torch.mean(rel_l2_b_v, dim=0)  # (V,)

        # 3. 39 个通道独立的 Hybrid 损失: (V,)
        channel_losses = mse_v + self.lambda_rel * rel_l2_v

        # 4. 训练模式且包含计算图时执行 GradNorm 动态更新
        if self.training and pred_norm.requires_grad and update_weights:
            with torch.enable_grad():
                # 利用通道正交解耦特性，一次 autograd 提取 39 个通道对 pred_norm 的梯度范数
                grads = torch.autograd.grad(channel_losses.sum(), pred_norm, retain_graph=True, create_graph=False)[0]
                g_v = torch.norm(grads, p=2, dim=(0, 2)).detach()  # (V,)
                g_v = torch.clamp(g_v, min=1e-8)

            with torch.no_grad():
                curr_losses = channel_losses.detach()

                # 初始化 / 预热阶段：记录初始损失基线 (非原地操作，杜绝 PyTorch autograd 版本冲突)
                if not self.is_initialized:
                    if self.step_count == 0:
                        self.initial_losses = curr_losses
                    else:
                        # 预热期平滑跟踪初始损失
                        self.initial_losses = 0.9 * self.initial_losses + 0.1 * curr_losses

                    if self.step_count >= self.warmup_steps:
                        self.is_initialized = torch.tensor(True, device=pred_norm.device)

                # 仅在基线非退化状态下执行 GradNorm 更新
                init_loss_safe = torch.clamp(self.initial_losses, min=1e-6)
                loss_ratio = curr_losses / init_loss_safe  # (V,)
                mean_loss_ratio = loss_ratio.mean()
                inverse_rate = loss_ratio / (mean_loss_ratio + 1e-8)  # r_v(t)

                # 当前加权梯度范数 G_v 与平均梯度范数
                curr_w = self.task_weights.detach()
                G_v = curr_w * g_v
                mean_G = G_v.mean()

                # 目标梯度范数 G_target
                G_target = mean_G * (inverse_rate ** self.alpha)

                # 目标权重比例
                w_target = G_target / g_v

                # EMA 平滑更新
                new_w = (1.0 - self.eta) * curr_w + self.eta * w_target

                # 归一化约束: sum(w) == V (等价于 mean(w) == 1.0)
                new_w = new_w * (V / (new_w.sum() + 1e-8))

                # 安全钳制，防止任何单一通道过度膨胀或彻底饿死
                new_w = torch.clamp(new_w, min=self.w_min, max=self.w_max)
                new_w = new_w * (V / (new_w.sum() + 1e-8))

                # 非原地赋值更新缓冲，避免破坏前向计算图中已保存张量的版本计数器
                self.task_weights = new_w
                self.step_count = self.step_count + 1

        # 5. 计算最终加权损失 (除以 V 保证与基线 HybridMSERelLoss 尺度完全一致)
        weighted_loss = torch.sum(self.task_weights * channel_losses) / float(V)

        # 6. 保存诊断数据供 logger 记录
        if self.training:
            w_snap = self.task_weights.detach()
            self._last_metrics = {
                'w_clb2': w_snap[2].item(),  # CLB2 (v=2)
                'w_sic1': w_snap[4].item(),  # SIC1 (v=4)
                'w_cln2': w_snap[1].item(),  # CLN2 (v=1)
                'w_mass': w_snap[0].item(),  # MASS (v=0)
                'w_max': w_snap.max().item(),
                'w_min': w_snap.min().item(),
                'w_std': w_snap.std().item(),
            }

        return weighted_loss


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
        pred_fft = torch.fft.rfft(pred_norm.float(), dim=-1)
        target_fft = torch.fft.rfft(target_norm.float(), dim=-1)
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
                 warmup_epochs=30, norm_type='log1p'):
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
            norm_type: 归一化方案 (asinh_z, log1p, asinh_scale, linear)
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
        self.norm_type = norm_type
        self._current_epoch = 0

        self.mse = nn.MSELoss(reduction='none')

    def set_epoch(self, epoch):
        """设置当前 epoch，用于 curriculum learning"""
        self._current_epoch = epoch

    def _denormalize_y(self, y_norm):
        """反归一化状态变量到真实空间"""
        y_shifted = y_norm * self.std_y + self.mean_y
        if getattr(self, 'norm_type', 'log1p') in ('log1p', 'asinh_z'):
            return torch.clamp(torch.expm1(torch.clamp(y_shifted, min=-10.0, max=15.0)), min=0.0)
        return y_shifted

    def _denormalize_p(self, p_norm):
        """反归一化参数到真实空间，并构建参数字典"""
        nt = getattr(self, 'norm_type', 'log1p')
        if nt == 'asinh_z':
            is_varying = (self.p_std > 1e-5)
            p_std_safe = torch.where(is_varying, self.p_std, torch.ones_like(self.p_std))
            p_real = torch.where(is_varying, torch.sinh(p_norm) * p_std_safe + self.p_mean, self.p_mean)
            p_real = torch.clamp(p_real, min=0.0)
        elif nt == 'asinh_scale':
            is_varying = (self.p_std > 1e-5)
            p_trans = torch.where(is_varying, p_norm * self.p_std + self.p_mean, self.p_mean)
            p_real = torch.clamp(10.0 ** torch.clamp(p_trans, min=-8.0, max=6.0), min=0.0)
        else:
            p_real = p_norm * self.p_std + self.p_mean
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
        # 【关键】浓度非负约束: 防止预测的负值导致 ODE 分母接近 0 产生 NaN
        # (SIC1T = SIC1+C2+C5+SIC1P+C2P+C5P 等聚合变量在分母中, 负值可使分母归零)
        y_all = torch.clamp(y_all, min=0.0)
        y_all = y_all.transpose(1, 2).reshape(B * n_t, V)  # (B*n_t, 39)

        p_dict_expanded = {}
        for k, v in p_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B:
                p_dict_expanded[k] = v.unsqueeze(1).expand(B, n_t).reshape(B * n_t)
            else:
                p_dict_expanded[k] = v

        t_all = t_grid[t_indices].unsqueeze(0).expand(B, n_t).reshape(B * n_t)
        f_all = eqns_torch(y_all, p_dict_expanded, t_all)  # (B*n_t, 39)
        # 安全防护: ODE 函数在极端输入下仍可能产生 NaN/Inf (Goldbeter-Koshland 分母→0)
        f_all = torch.nan_to_num(f_all, nan=0.0, posinf=1e4, neginf=-1e4)
        f_u[:, :, t_indices] = f_all.reshape(B, n_t, V).transpose(1, 2)

        # 残差（真实空间）归一化到归一化空间
        residual = du_dt - f_u  # (B, 39, T)
        residual_norm = residual / self.std_y  # (B, 39, T)
        # 【关键】nan_to_num + clamp 双重防护:
        #   torch.clamp(NaN, max=100) 仍是 NaN! 必须先用 nan_to_num 转为有限值
        #   ODE 函数 f(u,p) 分母→0 时产生 NaN, 传播到残差使模型权重变 NaN
        residual_norm = torch.nan_to_num(residual_norm, nan=0.0, posinf=50.0, neginf=-50.0)
        residual_sq = torch.clamp(residual_norm ** 2, max=100.0)

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


class H1Loss(nn.Module):
    """
    H1 Sobolev 损失: 时域 L2 + 频域梯度 L2

    数学定义:
        ||u-v||_{H1}² = α·||u-v||_{L2}² + β·||du/dt - dv/dt||_{L2}²

    时域部分 (α): MSE(pred, target) — 约束原始值
    频域部分 (β): Σ_k k²·|FFT(pred-target)_k|² — 约束导数/高频分量

    频域实现 (合并两项等价):
        H1 = Σ_k (α + β·k²)·|û_k - v̂_k|²

    适配 budding yeast:
    - 数据形状: (B, 39, T), T=500, FFT 在时间维度 dim=-1
    - 39 个变量有不同量级，在归一化空间计算
    - 时域和频域分别计算，便于观察两部分贡献

    来源: HyperNOs 项目的 H1relLoss_1D (loss_fun.py L336-L401)，
          适配为 budding yeast 时间序列场景。
    原始 H1relLoss_1D 在频域合并计算 (α + β·k²) 权重，
          此处拆分为时域 MSE + 频域梯度 MSE 以明确两部分贡献。
    """

    def __init__(self, mean_y=None, std_y=None, alpha=1.0, beta=1.0):
        """
        Args:
            mean_y, std_y: (1, 39, 1) 归一化统计量 (接口兼容, 此损失在归一化空间计算不直接使用)
            alpha: 时域 L2 权重 (约束原始值)
            beta: 频域梯度 L2 权重 (约束导数/高频)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth):
        B, V, T = pred_norm.shape

        # 时域 L2 损失 (原始值 MSE)
        diff = pred_norm - target_norm  # (B, V, T)
        loss_time = diff.pow(2).mean(dim=(1, 2))  # (B,)

        # 频域梯度 L2 损失 (导数 MSE, 通过 FFT 计算)
        # 导数在频域: du/dt ↔ i·ω·û, 模² = ω²·|û|²
        # 使用 rfft (实数输入), 频率 k = rfftfreq(T)
        # 【关键】norm='ortho' 确保 FFT 能量守恒 (对齐 HyperNOs H1relLoss_1D)
        #   无 norm 时 |FFT|² 正比于 T², T=500 时频域损失比 MSE 大 ~500 倍, 导致梯度爆炸
        diff_fft = torch.fft.rfft(diff.float(), dim=-1, norm='ortho')  # (B, V, T//2+1) complex
        k = torch.fft.rfftfreq(T, device=pred_norm.device)  # (T//2+1,) 归一化频率 [0, 0.5]
        # 频域梯度: |FFT(diff)|² × k², 对高频分量施加更大权重
        loss_freq = (torch.abs(diff_fft) ** 2 * (k ** 2).unsqueeze(0).unsqueeze(0)).mean(dim=(1, 2))  # (B,)

        # H1 = α·时域L2 + β·频域梯度L2
        total_loss = self.alpha * loss_time + self.beta * loss_freq  # (B,)

        # 记录分量
        self._last_loss_components = {
            'h1_time': loss_time.mean().detach(),
            'h1_freq': loss_freq.mean().detach(),
        }

        return (total_loss * batch_weights).mean()


class PINOLoss(PINNResidualLoss):
    """
    PINO (Physics-Informed Neural Operator) 损失函数。

    L = L_data + λ_phys·L_phys + λ_h1·L_h1

    L_data:  MSE 数据损失 (时域, 继承自 PINNResidualLoss)
    L_phys:  ODE 残差损失 (有限差分计算 du/dt, 继承自 PINNResidualLoss)
    L_h1:    H1 频域梯度约束 (新增, FFT 计算高频分量误差)

    与 PINNResidualLoss 的区别:
    1. 加入 H1 频域梯度约束 (Sobolev 范数, 约束高频/导数误差)
    2. 保留 ODE 残差和 event 掩码 (物理约束)
    3. H1 约束只包含频域梯度部分 (时域 L2 已由 MSE 覆盖)

    设计说明:
    - PINO 论文用频域方法计算导数, 但 budding yeast 轨迹含 event 跳跃,
      FFT 导数在跳跃点产生 Gibbs 现象, 故 du/dt 仍用有限差分。
    - PINO 的频域思想体现在 H1 约束上: 通过 FFT 在频域约束高频误差。
    - H1 约束与物理约束同步 curriculum learning (warmup 后逐步启动)。

    来源: PINO 论文 "Physics-Informed Neural Operator for Learning PDEs"
          (Li et al., ICLR 2021), 适配 budding yeast ODE 场景。
    """

    def __init__(self, mean_y, std_y, p_mean, p_std, t_max=210.0,
                 lambda_phys=0.01, lambda_jump=0.001, lambda_penalty=1.0,
                 event_window=4, event_tol=0.08, jump_ratio=0.3,
                 residual_subsample=1, event_weight=0.1,
                 warmup_epochs=30,
                 lambda_h1=0.1, h1_beta=1.0):
        """
        Args:
            ... (继承 PINNResidualLoss 的所有参数)
            lambda_h1: H1 频域约束权重 (warmup 后逐步启动)
            h1_beta: H1 频域梯度 L2 权重 (时域 L2 由 MSE 覆盖, 此处只约束频域梯度)
        """
        super().__init__(mean_y, std_y, p_mean, p_std, t_max,
                         lambda_phys, lambda_jump, lambda_penalty,
                         event_window, event_tol, jump_ratio,
                         residual_subsample, event_weight,
                         warmup_epochs)
        self.lambda_h1 = lambda_h1
        self.h1_beta = h1_beta

    def _compute_h1_freq_loss(self, pred_norm, target_norm):
        """
        H1 频域梯度损失 (仅频域梯度部分, 时域 L2 由 MSE 覆盖)

        计算: mean( |FFT(pred-target)|² × k² )

        高频分量 (大 k) 受到更大惩罚, 约束预测的导数/高频误差。
        不使用 event 掩码: FFT 是全局变换, 无法局部屏蔽;
        通过较小的 lambda_h1 权重避免压制 event 跳跃。
        """
        B, V, T = pred_norm.shape
        diff = pred_norm - target_norm  # (B, V, T)
        # 【关键】norm='ortho' 确保 FFT 能量守恒, 否则 |FFT|² 正比于 T² 导致梯度爆炸
        diff_fft = torch.fft.rfft(diff.float(), dim=-1, norm='ortho')  # (B, V, T//2+1) complex
        k = torch.fft.rfftfreq(T, device=pred_norm.device)  # (T//2+1,)
        # 频域梯度 L2: |FFT(diff)|² × k²
        loss_freq = (torch.abs(diff_fft) ** 2 * (k ** 2).unsqueeze(0).unsqueeze(0)).mean(dim=(1, 2))  # (B,)
        return loss_freq.mean()

    def _get_h1_weight(self):
        """Curriculum: H1 约束与物理约束同步启动"""
        if self._current_epoch < self.warmup_epochs:
            return 0.0
        progress = min(1.0, (self._current_epoch - self.warmup_epochs) / 30.0)
        return self.lambda_h1 * progress

    def forward(self, pred_norm, target_norm, p_norm, batch_weights,
                lambda_penalty=None, lambda_smooth=None):
        """
        PINO 损失前向传播。

        Args:
            pred_norm: (B, 39, T) 归一化预测轨迹
            target_norm: (B, 39, T) 归一化目标轨迹
            p_norm: (B, 141) 归一化参数向量
            batch_weights: (B,) 批次权重
            lambda_penalty: 负值惩罚权重 (覆盖默认值)
            lambda_smooth: 未使用 (接口兼容)

        Returns:
            total_loss: 标量
        """
        # 父类处理: MSE + 负值惩罚 + ODE残差 + 跳跃条件
        total_loss = super().forward(pred_norm, target_norm, p_norm, batch_weights,
                                     lambda_penalty, lambda_smooth)

        # H1 频域约束 (新增)
        lam_h1_eff = self._get_h1_weight()
        loss_h1 = torch.tensor(0.0, device=pred_norm.device)
        if lam_h1_eff > 0:
            loss_h1 = self._compute_h1_freq_loss(pred_norm, target_norm)
            total_loss = total_loss + lam_h1_eff * loss_h1

        # 更新分量记录
        if hasattr(self, '_last_loss_components'):
            self._last_loss_components['h1_freq'] = loss_h1.detach()
            self._last_loss_components['h1_weight'] = torch.tensor(lam_h1_eff)

        return total_loss


class PhaseDerivativeLoss(nn.Module):
    """
    Phase Space & Tangential Derivative Loss:
    L = L_hybrid + lambda_deriv * L_deriv_rel + lambda_tangent * L_tangent
    - L_hybrid: Base MSE + Relative L2 amplitude loss
    - L_deriv_rel: First-order temporal derivative relative error (central differences)
    - L_tangent: Tangential vector alignment in 39-D phase space (1 - CosineSimilarity)
    Forces the neural operator to learn limit cycle phase flow and slope dynamics.
    """
    def __init__(self, mean_y=None, std_y=None, lambda_rel=0.2, lambda_deriv=0.2, lambda_tangent=0.1, eps=0.1):
        super().__init__()
        self.lambda_rel = lambda_rel
        self.lambda_deriv = lambda_deriv
        self.lambda_tangent = lambda_tangent
        self.eps = eps
        self.rel_loss = RelativeL2Loss(eps=eps, max_clamp=10.0)

    def forward(self, pred_norm, target_norm, batch_weights=None, lambda_penalty=None, lambda_smooth=None):
        # 1. Base amplitude loss
        loss_mse = F.mse_loss(pred_norm, target_norm)
        loss_rel = self.rel_loss(pred_norm, target_norm)
        loss_hybrid = loss_mse + self.lambda_rel * loss_rel

        # 2. Derivative field via central difference along time dimension
        d_pred = (pred_norm[:, :, 2:] - pred_norm[:, :, :-2]) * 0.5
        d_target = (target_norm[:, :, 2:] - target_norm[:, :, :-2]) * 0.5

        # Relative L2 on derivative field
        diff_d_l2 = torch.norm(d_pred - d_target, p=2, dim=-1)
        target_d_energy = torch.sqrt(torch.sum(d_target ** 2, dim=-1) + 1e-4)
        loss_deriv = torch.clamp(diff_d_l2 / target_d_energy, max=10.0).mean()

        # Tangential phase alignment in 39-D phase space
        cos_sim = F.cosine_similarity(d_pred, d_target, dim=1, eps=1e-6) # (B, T-2)
        loss_tangent = (1.0 - cos_sim).mean()

        total_loss = loss_hybrid + self.lambda_deriv * loss_deriv + self.lambda_tangent * loss_tangent
        return total_loss


class SoftDTWLoss(nn.Module):
    """
    Differentiable Soft Dynamic Time Warping Loss:
    L = L_hybrid + lambda_dtw * SoftDTW(pred, target)
    - Anti-diagonal vectorized GPU dynamic programming with temperature gamma.
    - Decouples geometric waveform morphology / amplitude fitting from minor temporal phase drift.
    """
    def __init__(self, mean_y=None, std_y=None, gamma=0.1, sub_stride=20, lambda_rel=0.2, lambda_dtw=0.05, use_core_only=True):
        super().__init__()
        self.gamma = gamma
        self.sub_stride = sub_stride
        self.lambda_rel = lambda_rel
        self.lambda_dtw = lambda_dtw
        self.use_core_only = use_core_only
        self.core_indices = [0, 1, 2, 4, 7, 8, 9, 10, 11]
        self.rel_loss = RelativeL2Loss(eps=0.1, max_clamp=10.0)

    def _soft_dtw_anti_diag(self, x, y):
        # x, y: (B*V, N) - compute in float32 for AMP stability and index-put compatibility
        orig_dtype = x.dtype
        x_f = x.float()
        y_f = y.float()
        B_V, N = x_f.shape
        D = (x_f.unsqueeze(2) - y_f.unsqueeze(1)) ** 2  # Pairwise Euclidean distance matrix (B*V, N, N)

        # DP matrix padded with 1e4 in float32
        R = torch.full((B_V, N + 2, N + 2), fill_value=1e4, device=x.device, dtype=torch.float32)
        R[:, 1, 1] = 0.0

        # Vectorized loop across anti-diagonals k = i + j
        for k in range(3, 2 * N + 3):
            i_min = max(2, k - (N + 1))
            i_max = min(N + 1, k - 2)
            if i_min > i_max:
                continue
            i_idx = torch.arange(i_min, i_max + 1, device=x.device)
            j_idx = k - i_idx

            r0 = R[:, i_idx - 1, j_idx - 1]
            r1 = R[:, i_idx - 1, j_idx]
            r2 = R[:, i_idx, j_idx - 1]

            stack_r = torch.stack([r0, r1, r2], dim=-1)
            soft_min = -self.gamma * torch.logsumexp(-stack_r / self.gamma, dim=-1)
            d_val = D[:, i_idx - 2, j_idx - 2]
            R[:, i_idx, j_idx] = d_val + soft_min

        return R[:, N + 1, N + 1].mean().to(orig_dtype)

    def forward(self, pred_norm, target_norm, batch_weights=None, lambda_penalty=None, lambda_smooth=None):
        loss_mse = F.mse_loss(pred_norm, target_norm)
        loss_rel = self.rel_loss(pred_norm, target_norm)
        loss_hybrid = loss_mse + self.lambda_rel * loss_rel

        # Subsample along time (T=500 -> N=25) on Core 9 channels for millisecond GPU execution
        if self.use_core_only:
            p_sub = pred_norm[:, self.core_indices, ::self.sub_stride]
            t_sub = target_norm[:, self.core_indices, ::self.sub_stride]
        else:
            p_sub = pred_norm[:, :, ::self.sub_stride]
            t_sub = target_norm[:, :, ::self.sub_stride]

        B, V, N = p_sub.shape
        dtw_val = self._soft_dtw_anti_diag(p_sub.reshape(B * V, N), t_sub.reshape(B * V, N))

        return loss_hybrid + self.lambda_dtw * dtw_val


class TripleDynamicLoss(nn.Module):
    """
    三元动力学复合损失 (Triple-Dynamic Composite Loss)
    同时联合三大约束维度:
    1. 宏观浓度与通道能量平衡: Hybrid Loss (MSE + lambda_rel * Rel_L2)
    2. 39维相空间微分切向流场对齐: Phase Derivative Loss (一阶中心差分 Rel_L2 + 速度向量余弦相似度)
    3. 核心细胞周期因子可微软动态时间规整: Soft-DTW Loss (解除尖峰时序微小相位偏移的欧氏过度惩罚)
    """
    def __init__(self, lambda_rel=0.5, lambda_deriv=0.5, lambda_cos=0.1, lambda_dtw=0.05,
                 gamma=0.1, sub_stride=20, use_core_only=True):
        super().__init__()
        self.lambda_rel = lambda_rel
        self.lambda_deriv = lambda_deriv
        self.lambda_cos = lambda_cos
        self.lambda_dtw = lambda_dtw
        self.gamma = gamma
        self.sub_stride = sub_stride
        self.use_core_only = use_core_only
        self.core_indices = [0, 1, 2, 4, 7, 8, 9, 10, 11]  # Core 9 variables
        self.rel_loss = RelativeL2Loss(eps=0.1, max_clamp=10.0)

    def _soft_dtw_anti_diag(self, x, y):
        orig_dtype = x.dtype
        x_f = x.float()
        y_f = y.float()
        B_V, N = x_f.shape
        D = (x_f.unsqueeze(2) - y_f.unsqueeze(1)) ** 2  # (B*V, N, N)

        R = torch.full((B_V, N + 2, N + 2), fill_value=1e4, device=x.device, dtype=torch.float32)
        R[:, 1, 1] = 0.0

        for k in range(3, 2 * N + 3):
            i_min = max(2, k - (N + 1))
            i_max = min(N + 1, k - 2)
            if i_min > i_max:
                continue
            i_idx = torch.arange(i_min, i_max + 1, device=x.device)
            j_idx = k - i_idx

            r0 = R[:, i_idx - 1, j_idx - 1]
            r1 = R[:, i_idx - 1, j_idx]
            r2 = R[:, i_idx, j_idx - 1]

            stack_r = torch.stack([r0, r1, r2], dim=-1)
            soft_min = -self.gamma * torch.logsumexp(-stack_r / self.gamma, dim=-1)
            d_val = D[:, i_idx - 2, j_idx - 2]
            R[:, i_idx, j_idx] = d_val + soft_min

        return R[:, N + 1, N + 1].mean().to(orig_dtype)

    def forward(self, pred_norm, target_norm, batch_weights=None, lambda_penalty=None, lambda_smooth=None, **kwargs):
        # 1. Macro Hybrid Loss
        loss_mse = F.mse_loss(pred_norm, target_norm)
        loss_rel = self.rel_loss(pred_norm, target_norm)
        loss_hybrid = loss_mse + self.lambda_rel * loss_rel

        # 2. Micro Phase Derivative & Tangential Alignment Loss
        d_pred = (pred_norm[:, :, 2:] - pred_norm[:, :, :-2]) * 0.5
        d_target = (target_norm[:, :, 2:] - target_norm[:, :, :-2]) * 0.5

        loss_deriv_rel = self.rel_loss(d_pred, d_target)

        eps = 1e-6
        pred_norm_t = torch.norm(d_pred, dim=1, keepdim=True) + eps
        target_norm_t = torch.norm(d_target, dim=1, keepdim=True) + eps
        cos_sim = torch.sum((d_pred / pred_norm_t) * (d_target / target_norm_t), dim=1)  # (B, T-2)
        loss_cos = (1.0 - cos_sim).mean()

        loss_phase = loss_deriv_rel + self.lambda_cos * loss_cos

        # 3. Nonlinear Soft-DTW Phase Decoupling Loss
        if self.use_core_only:
            p_sub = pred_norm[:, self.core_indices, ::self.sub_stride]
            t_sub = target_norm[:, self.core_indices, ::self.sub_stride]
        else:
            p_sub = pred_norm[:, :, ::self.sub_stride]
            t_sub = target_norm[:, :, ::self.sub_stride]

        B, V, N = p_sub.shape
        dtw_val = self._soft_dtw_anti_diag(p_sub.reshape(B * V, N), t_sub.reshape(B * V, N))

        # 4. Total Triple-Dynamic Loss
        total_loss = loss_hybrid + self.lambda_deriv * loss_phase + self.lambda_dtw * dtw_val
        return total_loss


# =========================================================================
# 方案 A, 方案 C, 方案 A+C 损失函数体系
# =========================================================================
class HybridBifurcateLoss(nn.Module):
    """
    方案 A 损失: 基础数据损失 + 表型分类交叉熵 + 阻滞态末端平坦稳态正则
    - loss_data: HybridMSERelLoss(pred, target)
    - loss_cls: CrossEntropy(pattern_logits, pattern_targets)
    - loss_arrest: Penalize late-time temporal derivatives for cells in arrest state (Pattern C)
    """
    def __init__(self, lambda_rel=0.2, lambda_pattern=0.1, lambda_arrest=0.05,
                 eps=0.1, max_clamp=10.0, arrest_window_ratio=0.5):
        super().__init__()
        self.hybrid_data_loss = HybridMSERelLoss(lambda_rel=lambda_rel, eps=eps, max_clamp=max_clamp)
        self.lambda_pattern = lambda_pattern
        self.lambda_arrest = lambda_arrest
        self.arrest_window_ratio = arrest_window_ratio

    def forward(self, pred_norm, target_norm, batch_weights=None, lambda_penalty=None, lambda_smooth=None,
                pattern_logits=None, pattern_targets=None):
        # 1. Base hybrid data loss
        loss_data = self.hybrid_data_loss(pred_norm, target_norm)

        # 2. Pattern classification loss
        loss_cls = torch.tensor(0.0, device=pred_norm.device)
        if pattern_logits is not None and pattern_targets is not None:
            if pattern_targets.dim() > 1:
                pattern_targets = pattern_targets.squeeze()
            loss_cls = F.cross_entropy(pattern_logits, pattern_targets.long())

        # 3. Pattern C arrest flatness penalty (suppresses fake oscillations like #15)
        loss_arrest = torch.tensor(0.0, device=pred_norm.device)
        if self.lambda_arrest > 0:
            B, V, T = pred_norm.shape
            t_start = int(T * (1.0 - self.arrest_window_ratio))
            d_late = pred_norm[:, :, (t_start + 1):] - pred_norm[:, :, t_start:-1]
            d_late_sq = d_late ** 2

            if pattern_targets is not None:
                is_arrest = (pattern_targets == 2).float().unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
                num_arrest = is_arrest.sum()
                if num_arrest > 0:
                    loss_arrest = (d_late_sq * is_arrest).sum() / (num_arrest * V * (T - t_start - 1) + 1e-8)
            elif pattern_logits is not None:
                p_c = F.softmax(pattern_logits, dim=-1)[:, 2].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
                loss_arrest = (d_late_sq * p_c).mean()

        total_loss = loss_data + self.lambda_pattern * loss_cls + self.lambda_arrest * loss_arrest
        return total_loss


class HybridPINOCollocationLoss(PINNResidualLoss):
    """
    方案 C 损失: Hybrid 数据损失 + 无标签/配点 ODE 物理残差 (解决 Event 跳变爆炸)
    - 基础数据损失采用 HybridMSERelLoss 保证宏观波形与微量蛋白能量平衡
    - 3 种 Event 跳变策略:
        1. 'decoupled': 仅在 33 个 strictly continuous 生化通道计算物理残差 (完全免疫跳跃)
        2. 'huber': 在全通道上使用 Huber loss (梯度截断为常数 delta, 防止梯度爆炸)
        3. 'full': 传统平方残差
    - 支持跳跃代数匹配损失 (lambda_jump > 0)
    - Curriculum learning 预热调度
    """
    def __init__(self, mean_y, std_y, p_mean, p_std, t_max=210.0,
                 lambda_rel=0.2, lambda_phys=0.01, lambda_jump=0.001, lambda_penalty=1.0,
                 event_window=4, event_tol=0.08, jump_ratio=0.3,
                 residual_subsample=1, event_weight=0.1, warmup_epochs=30,
                 event_strategy='decoupled', huber_delta=1.0):
        super().__init__(mean_y, std_y, p_mean, p_std, t_max,
                         lambda_phys, lambda_jump, lambda_penalty,
                         event_window, event_tol, jump_ratio,
                         residual_subsample, event_weight,
                         warmup_epochs)
        self.hybrid_data_loss = HybridMSERelLoss(lambda_rel=lambda_rel)
        self.event_strategy = event_strategy
        self.huber_delta = huber_delta

    def _compute_ode_residual_loss(self, pred_real, p_dict, mask):
        B, V, T = pred_real.shape
        dt = self.t_max / (T - 1)

        du_dt = torch.zeros_like(pred_real)
        du_dt[:, :, 1:-1] = (pred_real[:, :, 2:] - pred_real[:, :, :-2]) / (2 * dt)
        du_dt[:, :, 0] = (pred_real[:, :, 1] - pred_real[:, :, 0]) / dt
        du_dt[:, :, -1] = (pred_real[:, :, -1] - pred_real[:, :, -2]) / dt

        f_u = torch.zeros_like(pred_real)
        t_indices = list(range(0, T, self.residual_subsample))
        t_grid = torch.linspace(0, self.t_max, T, device=pred_real.device)

        n_t = len(t_indices)
        y_all = pred_real[:, :, t_indices]
        y_all = torch.clamp(y_all, min=0.0)
        y_all = y_all.transpose(1, 2).reshape(B * n_t, V)

        p_dict_expanded = {}
        for k, v in p_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B:
                p_dict_expanded[k] = v.unsqueeze(1).expand(B, n_t).reshape(B * n_t)
            else:
                p_dict_expanded[k] = v

        t_all = t_grid[t_indices].unsqueeze(0).expand(B, n_t).reshape(B * n_t)
        f_all = eqns_torch(y_all, p_dict_expanded, t_all)
        f_all = torch.nan_to_num(f_all, nan=0.0, posinf=1e4, neginf=-1e4)
        f_u[:, :, t_indices] = f_all.reshape(B, n_t, V).transpose(1, 2)

        residual = du_dt - f_u
        residual_norm = residual / self.std_y
        residual_norm = torch.nan_to_num(residual_norm, nan=0.0, posinf=50.0, neginf=-50.0)

        # 解法 1: 变量解耦 (仅在 33 个连续生化变量上计算残差)
        if self.event_strategy == 'decoupled':
            residual_norm = residual_norm[:, CONTINUOUS_VARS, :]
            residual_val = torch.where(
                torch.abs(residual_norm) <= self.huber_delta,
                0.5 * (residual_norm ** 2),
                self.huber_delta * (torch.abs(residual_norm) - 0.5 * self.huber_delta)
            )
            weight = torch.ones(B, len(CONTINUOUS_VARS), T, dtype=pred_real.dtype, device=pred_real.device)
            subsample_mask = torch.zeros(T, dtype=torch.bool, device=pred_real.device)
            subsample_mask[t_indices] = True
            weight = weight * subsample_mask.unsqueeze(0).unsqueeze(0).float()
            return (residual_val * weight).sum() / (weight.sum() + 1e-8)

        # 解法 2: 鲁棒 Huber 物理残差 (在所有 39 个变量上自适应截断跳变冲击)
        elif self.event_strategy == 'huber':
            residual_val = torch.where(
                torch.abs(residual_norm) <= self.huber_delta,
                0.5 * (residual_norm ** 2),
                self.huber_delta * (torch.abs(residual_norm) - 0.5 * self.huber_delta)
            )
            weight = torch.ones_like(mask, dtype=pred_real.dtype, device=pred_real.device)
            weight[~mask] = self.event_weight
            weight = weight.unsqueeze(1).expand_as(residual_val)
            subsample_mask = torch.zeros(T, dtype=torch.bool, device=pred_real.device)
            subsample_mask[t_indices] = True
            weight = weight * subsample_mask.unsqueeze(0).unsqueeze(0).float()
            return (residual_val * weight).sum() / (weight.sum() + 1e-8)

        # 默认/传统平方残差
        else:
            residual_sq = torch.clamp(residual_norm ** 2, max=100.0)
            weight = torch.ones_like(mask, dtype=pred_real.dtype, device=pred_real.device)
            weight[~mask] = self.event_weight
            weight = weight.unsqueeze(1).expand_as(residual_sq)
            subsample_mask = torch.zeros(T, dtype=torch.bool, device=pred_real.device)
            subsample_mask[t_indices] = True
            weight = weight * subsample_mask.unsqueeze(0).unsqueeze(0).float()
            return (residual_sq * weight).sum() / (weight.sum() + 1e-8)

    def forward(self, pred_norm, target_norm, p_norm, batch_weights=None,
                lambda_penalty=None, lambda_smooth=None):
        lam_pen = lambda_penalty if lambda_penalty is not None else self.lambda_penalty

        # 1. 数据损失用 HybridMSERelLoss
        loss_data = self.hybrid_data_loss(pred_norm, target_norm)

        # 2. 负值惩罚
        negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))
        base_loss = loss_data + (lam_pen * negative_penalty).mean()

        total_loss = base_loss

        # 3. 物理残差与跳跃约束
        lam_phys_eff, lam_jump_eff = self._get_physics_weight()
        if lam_phys_eff > 0 or lam_jump_eff > 0:
            pred_real = self._denormalize_y(pred_norm)
            target_real = self._denormalize_y(target_norm)
            p_dict = self._denormalize_p(p_norm)

            with torch.no_grad():
                mask = detect_event_mask_torch(
                    target_real, event_window=self.event_window,
                    event_tol=self.event_tol, jump_ratio=self.jump_ratio
                )

            if lam_phys_eff > 0:
                loss_residual = self._compute_ode_residual_loss(pred_real, p_dict, mask)
                total_loss = total_loss + lam_phys_eff * loss_residual

            if lam_jump_eff > 0:
                loss_jump = self._compute_jump_loss(pred_real, target_real, mask)
                total_loss = total_loss + lam_jump_eff * loss_jump

        return total_loss


class HybridAllLoss(HybridPINOCollocationLoss):
    """
    方案 A + C 全融合旗舰损失:
    Hybrid 数据损失 + 表型分类交叉熵 + 阻滞稳态约束 + PINO 鲁棒物理残差
    """
    def __init__(self, mean_y, std_y, p_mean, p_std, t_max=210.0,
                 lambda_rel=0.2, lambda_pattern=0.1, lambda_arrest=0.05,
                 lambda_phys=0.01, lambda_jump=0.001, lambda_penalty=1.0,
                 event_window=4, event_tol=0.08, jump_ratio=0.3,
                 residual_subsample=1, event_weight=0.1, warmup_epochs=30,
                 event_strategy='decoupled', huber_delta=1.0):
        super().__init__(mean_y, std_y, p_mean, p_std, t_max=t_max,
                         lambda_rel=lambda_rel, lambda_phys=lambda_phys,
                         lambda_jump=lambda_jump, lambda_penalty=lambda_penalty,
                         event_window=event_window, event_tol=event_tol,
                         jump_ratio=jump_ratio, residual_subsample=residual_subsample,
                         event_weight=event_weight, warmup_epochs=warmup_epochs,
                         event_strategy=event_strategy, huber_delta=huber_delta)
        self.lambda_pattern = lambda_pattern
        self.lambda_arrest = lambda_arrest

    def forward(self, pred_norm, target_norm, p_norm, batch_weights=None,
                lambda_penalty=None, lambda_smooth=None,
                pattern_logits=None, pattern_targets=None):
        # 1. Base PINO collocation forward (hybrid data + negpen + phys + jump)
        total_loss = super().forward(pred_norm, target_norm, p_norm, batch_weights,
                                     lambda_penalty=lambda_penalty, lambda_smooth=lambda_smooth)

        # 2. Add pattern classification loss
        if pattern_logits is not None and pattern_targets is not None:
            if pattern_targets.dim() > 1:
                pattern_targets = pattern_targets.squeeze()
            loss_cls = F.cross_entropy(pattern_logits, pattern_targets.long())
            total_loss = total_loss + self.lambda_pattern * loss_cls

        # 3. Add arrest state flatness penalty
        if self.lambda_arrest > 0 and pattern_targets is not None:
            B, V, T = pred_norm.shape
            t_start = T // 2
            d_late = pred_norm[:, :, (t_start + 1):] - pred_norm[:, :, t_start:-1]
            d_late_sq = d_late ** 2
            is_arrest = (pattern_targets == 2).float().unsqueeze(1).unsqueeze(2)
            num_arrest = is_arrest.sum()
            if num_arrest > 0:
                loss_arrest = (d_late_sq * is_arrest).sum() / (num_arrest * V * (T - t_start - 1) + 1e-8)
                total_loss = total_loss + self.lambda_arrest * loss_arrest

        return total_loss


LOSS_REGISTRY = {
    'physics_informed': PhysicsInformedLoss,
    'mse_only': MSEOnlyLoss,
    'rel_l2': RelativeL2Loss,
    'hybrid_mse_rel': HybridMSERelLoss,
    'gradnorm_hybrid': GradNormHybridLoss,
    'hybrid_bifurcate': HybridBifurcateLoss,
    'hybrid_pino_colloc': HybridPINOCollocationLoss,
    'hybrid_all': HybridAllLoss,
    'mse_negpen': MSENegPenLoss,
    'mse_smooth': MSESmoothLoss,
    't_smooth': DilatedSobolevLoss,
    'pinn_residual': PINNResidualLoss,
    'h1': H1Loss,
    'pino': PINOLoss,
    'phase_derivative': PhaseDerivativeLoss,
    'soft_dtw': SoftDTWLoss,
    'triple_dynamic': TripleDynamicLoss,
    'hybrid_phase_dtw': TripleDynamicLoss,
    'hybrid_phase_deriv': PhaseDerivativeLoss,
    'hybrid_soft_dtw': SoftDTWLoss,
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