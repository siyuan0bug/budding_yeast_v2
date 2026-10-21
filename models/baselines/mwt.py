"""Standard polynomial multiwavelet hierarchy, adapted to joint-state ODE outputs.
Sources: Tau_BNO-main/models/MWNO and RealPDEBench MWT_libs.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import joint_condition_grid
from .fno import SpectralConv1d

class MWTKernel1d(nn.Module):
    """Official MWT_CZ1d recurrence, with power-of-two padding handled outside."""
    def __init__(self, channels, k, modes):
        super().__init__()
        from .multiwavelet_filters import get_filter
        h0, h1, g0, g1, phi0, phi1 = get_filter('legendre', k)
        self.k, self.channels = k, channels
        for name, matrix in (
            ('ec_s', np.concatenate((h0.T, h1.T), axis=0)),
            ('ec_d', np.concatenate((g0.T, g1.T), axis=0)),
            ('rc_e', np.concatenate((h0 @ phi0, g0 @ phi0), axis=0)),
            ('rc_o', np.concatenate((h1 @ phi1, g1 @ phi1), axis=0)),
        ):
            matrix[np.abs(matrix) < 1e-8] = 0
            self.register_buffer(name, torch.tensor(matrix, dtype=torch.float32))
        self.A = SpectralConv1d(channels * k, modes)
        self.B = SpectralConv1d(channels * k, modes)
        self.C = SpectralConv1d(channels * k, modes)
        self.T0 = nn.Linear(k, k)

    def decompose(self, x):
        paired = torch.cat((x[:, ::2], x[:, 1::2]), dim=-1)
        return paired @ self.ec_d.to(x.dtype), paired @ self.ec_s.to(x.dtype)

    def reconstruct(self, x):
        even = x @ self.rc_e.to(x.dtype)
        odd = x @ self.rc_o.to(x.dtype)
        return torch.stack((even, odd), dim=2).flatten(1, 2)

    def _kernel(self, kernel, x):
        b, n, c, k = x.shape
        return kernel(x.reshape(b, n, c*k).transpose(1, 2)).transpose(1, 2).reshape(b, n, c, k)

    def forward(self, x):
        details, skips = [], []
        while x.shape[1] > 1:
            d, x = self.decompose(x)
            details.append(self._kernel(self.A, d) + self._kernel(self.B, x))
            skips.append(self._kernel(self.C, d))
        x = self.T0(x)
        for detail, skip in zip(reversed(details), reversed(skips)):
            x = self.reconstruct(torch.cat((x + skip, detail), dim=-1))
        return x

class MWT(nn.Module):
    """Polynomial multiwavelet operator with joint parameter/initial-state input."""
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4, modes=24, k=4):
        super().__init__()
        if k < 1 or width < k or width % k or n_blocks < 1:
            raise ValueError("width must be a positive multiple of k; n_blocks must be positive")
        self.num_vars, self.param_dim, self.k = num_vars, param_dim, k
        self.channels = width // k
        self.lift = nn.Linear(num_vars + param_dim + 1, width)
        self.blocks = nn.ModuleList([MWTKernel1d(self.channels, k, modes) for _ in range(n_blocks)])
        self.project = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, num_vars))

    def forward(self, x, params):
        features = joint_condition_grid(x, params, self.num_vars, self.param_dim).transpose(1, 2)
        length = features.shape[1]
        h = self.lift(features)
        padded = 2 ** math.ceil(math.log2(length))
        # Periodic extension is an explicit task adaptation for the 500-point grid.
        if padded > length:
            h = torch.cat((h, h[:, :padded-length]), dim=1)
        h = h.reshape(h.shape[0], padded, self.channels, self.k)
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i < len(self.blocks) - 1:
                h = F.relu(h)
        return self.project(h[:, :length].flatten(2)).transpose(1, 2).contiguous()
