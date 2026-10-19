"""Shared tensor transforms; fitting/split policy remains owned by the DataModule.

``log1p`` describes the *state* transform, not the parameter transform:
Y -> zscore(log1p(Y)), P -> zscore(P). Keep these four historical modes
numerically compatible while using the same inverse everywhere.
"""
import torch


NORM_TYPES = ('log1p', 'linear', 'asinh_z', 'asinh_scale')


def check_norm_type(norm_type):
    if norm_type not in NORM_TYPES:
        raise ValueError(f'Unsupported normalization: {norm_type!r}')
    return norm_type


def _state_stat(stat, value):
    if stat is None:
        raise ValueError('Missing normalization statistic (asinh_scale requires s_c)')
    stat = torch.as_tensor(stat, device=value.device, dtype=value.dtype)
    if value.ndim == 2:  # (B,V), used by physical IC decoders
        return stat.reshape(1, -1)
    return stat.reshape(1, -1, 1)


def transform_y(y, norm_type='log1p', s_c=None):
    check_norm_type(norm_type)
    if norm_type in ('log1p', 'asinh_z'):
        return torch.log1p(y.clamp(min=0.0))
    if norm_type == 'asinh_scale':
        return torch.asinh(y.clamp(min=0.0) / (2.0 * _state_stat(s_c, y)))
    return y


def normalize_y(y, mean_y, std_y, norm_type='log1p', s_c=None):
    z = (transform_y(y, norm_type, s_c) - _state_stat(mean_y, y)) / _state_stat(std_y, y)
    return torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)


def denormalize_y(y, mean_y, std_y, norm_type='log1p', s_c=None):
    check_norm_type(norm_type)
    shifted = y * _state_stat(std_y, y) + _state_stat(mean_y, y)
    if norm_type in ('log1p', 'asinh_z'):
        return torch.expm1(shifted.clamp(min=-10.0, max=15.0)).clamp(min=0.0)
    if norm_type == 'asinh_scale':
        return (2.0 * _state_stat(s_c, y) * torch.sinh(shifted.clamp(-10.0, 10.0))).clamp(min=0.0)
    return shifted


def normalize_p(p, p_mean, p_std, norm_type='log1p'):
    check_norm_type(norm_type)
    mean = torch.as_tensor(p_mean, device=p.device, dtype=p.dtype)
    std = torch.as_tensor(p_std, device=p.device, dtype=p.dtype)
    if norm_type in ('asinh_z', 'asinh_scale'):
        varying = std > 1e-5
        safe_std = torch.where(varying, std, torch.ones_like(std))
        if norm_type == 'asinh_z':
            z = torch.asinh((p - mean) / safe_std)
        else:
            z = (torch.log10(p.clamp(min=1e-8)) - mean) / safe_std
        z = torch.where(varying, z, torch.zeros_like(z))
    else:
        z = (p - mean) / std
    return torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)


def denormalize_p(p, p_mean, p_std, norm_type='log1p'):
    check_norm_type(norm_type)
    mean = torch.as_tensor(p_mean, device=p.device, dtype=p.dtype)
    std = torch.as_tensor(p_std, device=p.device, dtype=p.dtype)
    if norm_type == 'asinh_z':
        varying = std > 1e-5
        safe_std = torch.where(varying, std, torch.ones_like(std))
        return torch.where(varying, torch.sinh(p) * safe_std + mean, mean).clamp(min=0.0)
    if norm_type == 'asinh_scale':
        shifted = torch.where(std > 1e-5, p * std + mean, mean)
        return torch.pow(10.0, shifted.clamp(-8.0, 6.0)).clamp(min=0.0)
    return p * std + mean
