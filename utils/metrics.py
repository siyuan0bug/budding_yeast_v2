import torch
import torch.nn.functional as F

def relative_l2_loss(pred, target, eps=0.1):
    """
    计算 Per-variable Relative L2 Loss (方案 B: 物理基线截断保护)。
    与 calculate_metrics 中的 'Relative L2' 计算完全一致，返回 PyTorch Scalar Tensor。
    
    Args:
        pred: (Batch, Vars, Time) 或 (Vars, Time) 预测张量
        target: (Batch, Vars, Time) 或 (Vars, Time) 真实目标张量
        eps: 物理基底浓度下限 (默认 0.1 a.u.)
        
    Returns:
        rel_l2: scalar tensor, 39 个变量 Relative L2 的平均值
    """
    if pred.dim() == 3:
        V = pred.shape[1]
        pred_v = pred.transpose(0, 1).reshape(V, -1)
        target_v = target.transpose(0, 1).reshape(V, -1)
    elif pred.dim() == 2:
        pred_v = pred
        target_v = target
    else:
        pred_v = pred.view(1, -1)
        target_v = target.view(1, -1)

    numerator = torch.sum((pred_v - target_v) ** 2, dim=1)
    denominator = torch.sum(target_v ** 2, dim=1)

    N_steps = pred_v.size(1)
    floor_energy = (eps ** 2) * N_steps
    safe_denominator = torch.clamp(denominator, min=floor_energy)

    return torch.sqrt(numerator / safe_denominator).mean()

def calculate_metrics(pred, target, eps=0.1):
    # 动态适应维度：正常传入的 pred 和 target 形状是 (Batch, Vars, Time)
    if pred.dim() == 3:
        B, V, T = pred.shape
        # 将维度转换为 (Vars, Batch, Time) 后，展平为 (Vars, Batch * Time)
        # 这样就把同一变量的所有样本和时间步拼在了一起
        pred_v = pred.transpose(0, 1).reshape(V, -1)
        target_v = target.transpose(0, 1).reshape(V, -1)
    elif pred.dim() == 2:
        # 如果传入的是 (Vars, Time)
        pred_v = pred
        target_v = target
    else:
        # 兜底防错
        pred_v = pred.view(1, -1)
        target_v = target.view(1, -1)

    # ==========================================
    # 1. 计算 Per-variable Relative L2 (方案 B: 物理基线截断保护)
    # ==========================================
    rel_l2 = relative_l2_loss(pred, target, eps=eps)

    # ==========================================
    # 2. 计算 Per-variable Correlation (Pearson)
    # ==========================================
    pred_mean = pred_v.mean(dim=1, keepdim=True)
    target_mean = target_v.mean(dim=1, keepdim=True)

    pred_centered = pred_v - pred_mean
    target_centered = target_v - target_mean

    # 协方差与标准差乘积，同样分别得到 39 个值
    cov = torch.sum(pred_centered * target_centered, dim=1)
    std_prod = torch.sqrt(torch.sum(pred_centered ** 2, dim=1) * torch.sum(target_centered ** 2, dim=1))
    
    # 算出 39 个相关系数后求平均
    correlation = (cov / (std_prod + 1e-8)).mean()

    # ==========================================
    # 3. 计算 MAE 和 MSE
    # ==========================================
    # 对于这两个绝对绝对误差指标，不管展平算还是先求变量再求平均，数学上是完全等价的
    mae = torch.mean(torch.abs(pred_v - target_v))
    mse = F.mse_loss(pred_v, target_v)

    return {
        'Relative L2': rel_l2.item(),
        'Correlation': correlation.item(),
        'MAE': mae.item(),
        'MSE': mse.item(),
    }