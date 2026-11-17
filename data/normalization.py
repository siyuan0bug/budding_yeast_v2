"""Shared statistics and tensor transforms; split policy stays in the DataModule.

``log1p`` describes the *state* transform, not the parameter transform:
Y -> zscore(log1p(Y)), P -> zscore(P). Keep these four historical modes
numerically compatible while using the same inverse everywhere.
"""
from dataclasses import dataclass
import numpy as np
import torch


NORM_TYPES = ('log1p', 'linear', 'asinh_z', 'asinh_scale')


@dataclass
class NormalizationStats:
    """One train-fitted transform for DataModules, inference and acquisition."""
    mean_y: torch.Tensor
    std_y: torch.Tensor
    p_mean: torch.Tensor
    p_std: torch.Tensor
    norm_type: str = 'log1p'
    s_c: torch.Tensor | None = None

    @classmethod
    def fit(cls, params, trajectories, norm_type='log1p'):
        check_norm_type(norm_type)
        p = torch.as_tensor(np.asarray(params).copy(), dtype=torch.float32)
        y = torch.as_tensor(trajectories, dtype=torch.float32)
        if not len(p) or len(p) != len(y) or not torch.isfinite(p).all() or not torch.isfinite(y).all():
            raise ValueError('Normalization needs aligned finite training rows')
        scale = None
        if norm_type == 'asinh_scale':
            value = np.quantile(np.maximum(y[:20000].numpy(), 0.), .90, axis=(0, 2), keepdims=True)
            scale = torch.tensor(np.maximum(value, .01), dtype=torch.float32)
        z = transform_y(y, norm_type, scale)
        q = torch.log10(p.clamp(min=1e-8)) if norm_type == 'asinh_scale' else p
        p_std = q.std(0, keepdim=True)
        if norm_type in ('log1p', 'linear'):
            p_std = p_std + 1e-6
        return cls(z.mean((0, 2), keepdim=True), z.std((0, 2), keepdim=True) + 1e-6,
                   q.mean(0, keepdim=True), p_std, norm_type, scale)

    @classmethod
    def from_datamodule(cls, dm):
        return cls(dm.mean_y, dm.std_y, dm.p_mean, dm.p_std, dm.norm_type, dm.s_c)

    def bind(self, dm):
        for key in ('mean_y', 'std_y', 'p_mean', 'p_std', 's_c'):
            setattr(dm, key, getattr(self, key))

    def encode(self, params, initial, steps):
        p = torch.as_tensor(np.asarray(params).copy(), dtype=torch.float32)
        y0 = torch.as_tensor(np.asarray(initial).copy(), dtype=torch.float32)
        initial = normalize_y(y0, self.mean_y, self.std_y, self.norm_type, self.s_c)
        grid = torch.linspace(0, 1, steps)[None, None].expand(len(p), y0.shape[1], -1)
        x = torch.stack((initial[:, :, None].expand(-1, -1, steps), grid), dim=2)
        return x, normalize_p(p, self.p_mean, self.p_std, self.norm_type)

    def targets(self, values):
        return normalize_y(torch.as_tensor(values, dtype=torch.float32), self.mean_y, self.std_y, self.norm_type, self.s_c)

    def decode(self, values):
        return denormalize_y(values, self.mean_y, self.std_y, self.norm_type, self.s_c)

    def state(self):
        return {key: None if getattr(self, key) is None else getattr(self, key).detach().cpu()
                for key in ('mean_y', 'std_y', 'p_mean', 'p_std', 's_c')}


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
