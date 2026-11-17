import numpy as np
import torch
import torch.nn.functional as F


RELATIVE_L2_METRIC_VERSION = 'physical_sample_channel_rel_l2_smooth_eps001_v3'
PHYSICAL_RELATIVE_L2_EPS = 0.01
SAMPLE_CHANNEL_REL_L2_NAME = 'Relative L2'  # One displayed metric; compatibility constant.


def sample_channel_relative_errors(pred, target, eps=PHYSICAL_RELATIVE_L2_EPS):
    """Uncapped physical relative L2 for each sample/channel, norm over time.

    Same denominator structure as RelativeL2Loss, evaluated on physical values:
    ||prediction-target||_t / sqrt(||target||_t^2 + eps^2).
    eps is an energy-norm protection, not the legacy per-time amplitude floor.
    No log transform, recentering, output cap, or target clipping occurs here.
    """
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('Physical Relative L2 requires a finite positive eps')
    p, y = torch.as_tensor(pred), torch.as_tensor(target)
    if p.shape != y.shape or p.ndim not in (2, 3):
        raise ValueError('Expected aligned V,T or N,V,T physical trajectories')
    if p.ndim == 2:
        p, y = p[None], y[None]
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError('Physical Relative L2 requires finite prediction and truth')
    # Double accumulation also avoids float32 energy overflow for finite inputs.
    numerator = torch.linalg.vector_norm(p-y, dim=-1, dtype=torch.float64)
    energy = torch.linalg.vector_norm(y, dim=-1, dtype=torch.float64)
    return numerator / torch.hypot(energy, energy.new_tensor(eps))


def relative_l2_loss(pred, target, eps=PHYSICAL_RELATIVE_L2_EPS):
    """Canonical physical, uncapped sample/channel Relative L2 scalar."""
    return sample_channel_relative_errors(pred, target, eps).mean()


def calculate_metrics(pred, target, eps=PHYSICAL_RELATIVE_L2_EPS):
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
    # 1. Canonical sample/channel physical RelL2, no ratio cap.
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


def trajectory_relative_errors(predictions, targets):
    """Per-input physical trajectory RelL2, shared by validation and reporting."""
    p=torch.as_tensor(predictions,dtype=torch.float64)
    y=torch.as_tensor(targets,dtype=torch.float64,device=p.device)
    return (torch.linalg.vector_norm((p-y).flatten(1),dim=1)
            /(torch.linalg.vector_norm(y.flatten(1),dim=1)+1e-8))


def per_sample_metrics(prediction, target):
    p,y=torch.as_tensor(prediction),torch.as_tensor(target)
    values=calculate_metrics(p,y)
    return dict(rel_l2=values['Relative L2'],MAE=values['MAE'],MSE=values['MSE'],
                correlation=values['Correlation'],
                trajectory_rel_l2=float(trajectory_relative_errors(p[None],y[None])[0]))


def summarize(predictions, targets, system_id=None):
    """One summary for static, pool-selection and Budding/Fission evaluation."""
    original_p,original_y=torch.as_tensor(predictions),torch.as_tensor(targets)
    p=original_p.detach().cpu().double();y=original_y.detach().cpu().double()
    if p.shape!=y.shape or p.ndim!=3 or not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError('Evaluation requires aligned finite B,V,T physical trajectories')
    if system_id is None:system_id='fission' if p.shape[1]==16 else 'budding' if p.shape[1]==39 else 'unspecified'
    metrics=calculate_metrics(original_p,original_y) if len(y) else None
    result=dict(count=len(y),metric_version=RELATIVE_L2_METRIC_VERSION,
        primary_metric='sample_channel_rel_l2',metric_space='physical',
        sample_channel_rel_l2=None if metrics is None else metrics[SAMPLE_CHANNEL_REL_L2_NAME],
        all_variables=metrics)
    result['all_16_vars' if system_id=='fission' else 'all_39_vars']=metrics
    if system_id=='budding':
        core=[0,1,2,3,4,20,22,33,35]
        result['core_9_vars']=calculate_metrics(original_p[:,core],original_y[:,core]) if len(y) else None
    if not len(y):return result
    rel=trajectory_relative_errors(p,y).numpy()
    a,b=p-p.mean(-1,keepdim=True),y-y.mean(-1,keepdim=True)
    denominator=(a.square().sum(-1)*b.square().sum(-1)).sqrt();valid=denominator>1e-12
    correlation=(a*b).sum(-1)[valid]/denominator[valid]
    result.update(macro_trajectory_rel_l2=float(rel.mean()),median_rel_l2=float(np.median(rel)),
        p90_rel_l2=float(np.quantile(rel,.9)),p95_rel_l2=float(np.quantile(rel,.95)),
        worst_10pct_mean_rel_l2=float(np.sort(rel)[-max(1,int(np.ceil(.1*len(rel)))):].mean()),
        global_frobenius_rel_l2=float(torch.linalg.vector_norm(p-y)/(torch.linalg.vector_norm(y)+1e-8)),
        valid_channel_correlation=float(correlation.mean()) if correlation.numel() else None,
        undefined_channel_correlations=int((~valid).sum()),
        metric_definition='sample_channel_rel_l2: mean_i,v ||prediction_iv-target_iv||_t/'
                          'sqrt(||target_iv||_t^2+0.01^2), physical, no cap; '
                          'macro_trajectory_rel_l2: mean_i ||prediction_i-target_i||_F/(||target_i||_F+1e-8); '
                          'all_variables.Relative L2 is the same canonical sample/channel metric; eps=0.01, no cap')
    if system_id=='budding':
        from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
        truth,pred=pattern_group_metadata(y.numpy()),pattern_group_metadata(p.numpy())
        matrix=np.zeros((6,6),dtype=np.int64)
        np.add.at(matrix,(truth['pattern_group_ids'],pred['pattern_group_ids']),1)
        result.update(six_group_confusion=matrix.tolist(),division_count_mae=float(np.abs(
            truth['observed_division_counts']-pred['observed_division_counts']).mean()))
    return result
