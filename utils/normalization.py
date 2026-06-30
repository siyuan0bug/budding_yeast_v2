"""
TaskNormalizer for the AR+AL pipeline.

References AL4PDE's normalization.py — computes per-channel mean/std
from the training data, used for:
1. Normalizing model inputs/outputs
2. Computing channel_std_rel noise levels
"""

import torch


class TaskNormalizer:
    """
    Computes and stores per-channel normalization statistics.
    
    Operates on data in (N, T, C) or (B, C, T) format.
    """

    def __init__(self):
        self.mean_channels = None   # (C,)
        self.std_channels = None    # (C,)
        self.mean_params = None     # (P,)
        self.std_params = None      # (P,)
        self._device = torch.device('cpu')

    def update(self, train_loader):
        """
        Compute normalization stats from a training DataLoader.
        
        Expects DataLoader to yield (ic, traj, params, t_idx) tuples
        where traj is (B, C, traj_len).
        """
        all_vals = []
        all_params = []

        for ic, traj, params, t_idx in train_loader:
            # traj: (B, C, traj_len) — collect all values
            all_vals.append(traj.reshape(-1, traj.shape[1]))  # (B*traj_len, C)
            all_params.append(params)

        all_vals = torch.cat(all_vals, dim=0)     # (total, C)
        all_params = torch.cat(all_params, dim=0)  # (total, P)

        self.mean_channels = all_vals.mean(dim=0)       # (C,)
        self.std_channels = all_vals.std(dim=0) + 1e-6   # (C,) Z-score后方差全≈1.0(见data_distribution_analysis.md §7.1)

        self.mean_params = all_params.mean(dim=0)       # (P,)
        self.std_params = all_params.std(dim=0) + 1e-6   # (P,) 65%参数为固定值(§4.1), std=0需eps保护

        return self

    def norm_state(self, x):
        """Normalize a state tensor. x: (..., C)"""
        res = (x - self.mean_channels) / self.std_channels
        return torch.nan_to_num(res, nan=0.0, posinf=10.0, neginf=-10.0)

    def denorm_state(self, x):
        """Denormalize a state tensor. x: (..., C)"""
        return x * self.std_channels + self.mean_channels

    def norm_traj(self, traj):
        """Normalize a trajectory. traj: (B, C, T) or (B, T, C)"""
        if traj.dim() == 3 and traj.shape[1] == len(self.mean_channels):
            # (B, C, T) format
            mean = self.mean_channels.unsqueeze(0).unsqueeze(-1)
            std = self.std_channels.unsqueeze(0).unsqueeze(-1)
        else:
            # (B, T, C) format
            mean = self.mean_channels.unsqueeze(0).unsqueeze(0)
            std = self.std_channels.unsqueeze(0).unsqueeze(0)
        res = (traj - mean) / std
        return torch.nan_to_num(res, nan=0.0, posinf=10.0, neginf=-10.0)

    def denorm_traj(self, traj):
        """Denormalize a trajectory. traj: (B, C, T) or (B, T, C)"""
        if traj.dim() == 3 and traj.shape[1] == len(self.mean_channels):
            mean = self.mean_channels.unsqueeze(0).unsqueeze(-1)
            std = self.std_channels.unsqueeze(0).unsqueeze(-1)
        else:
            mean = self.mean_channels.unsqueeze(0).unsqueeze(0)
            std = self.std_channels.unsqueeze(0).unsqueeze(0)
        return traj * std + mean

    def norm_param(self, param):
        """Normalize parameter vector. param: (B, P)"""
        res = (param - self.mean_params) / self.std_params
        return torch.nan_to_num(res, nan=0.0, posinf=10.0, neginf=-10.0)

    def denorm_param(self, param):
        """Denormalize parameter vector. param: (B, P)"""
        return param * self.std_params + self.mean_params

    def to(self, device):
        self._device = device
        if self.mean_channels is not None:
            self.mean_channels = self.mean_channels.to(device)
            self.std_channels = self.std_channels.to(device)
        if self.mean_params is not None:
            self.mean_params = self.mean_params.to(device)
            self.std_params = self.std_params.to(device)
        return self

    def state_dict(self):
        return {
            'mean_channels': self.mean_channels.cpu() if self.mean_channels is not None else None,
            'std_channels': self.std_channels.cpu() if self.std_channels is not None else None,
            'mean_params': self.mean_params.cpu() if self.mean_params is not None else None,
            'std_params': self.std_params.cpu() if self.std_params is not None else None,
        }

    def load_state_dict(self, state_dict):
        self.mean_channels = state_dict['mean_channels']
        self.std_channels = state_dict['std_channels']
        self.mean_params = state_dict['mean_params']
        self.std_params = state_dict['std_params']
        return self

    def cpu(self):
        return self.to(torch.device('cpu'))
