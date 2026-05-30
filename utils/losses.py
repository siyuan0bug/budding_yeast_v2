import torch
import torch.nn as nn
import torch.nn.functional as F


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


LOSS_REGISTRY = {
    'physics_informed': PhysicsInformedLoss,
    'mse_only': MSEOnlyLoss,
    'mse_negpen': MSENegPenLoss,
    'mse_smooth': MSESmoothLoss,
    't_smooth': DilatedSobolevLoss,
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