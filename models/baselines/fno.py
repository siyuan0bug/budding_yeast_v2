"""Standard temporal FNO adapted by concatenating all initial states and parameters.
Architecture source: Li et al. / RealPDEBench model/fno.py (3D reduced to time).
"""
import torch
from torch import nn
from torch.nn import functional as F
from .common import joint_condition_grid

class SpectralConv1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        if width < 1 or modes < 1:
            raise ValueError("width and modes must be positive")
        self.modes = modes
        self.weight = nn.Parameter(torch.rand(width, width, modes, dtype=torch.cfloat) / width**2)

    def forward(self, x):
        # FFT is unsupported for BF16 and for many FP16 signal lengths.
        dtype = x.dtype
        z = x if dtype == torch.float64 else x.float()
        spectrum = torch.fft.rfft(z)
        count = min(self.modes, spectrum.shape[-1])
        out = torch.zeros_like(spectrum)
        out[..., :count] = torch.einsum(
            'bim,iom->bom', spectrum[..., :count], self.weight[..., :count].to(spectrum.dtype)
        )
        return torch.fft.irfft(out, n=x.shape[-1]).to(dtype)

class FNO(nn.Module):
    """1D FNO with concatenated full initial state and ODE parameters."""
    def __init__(self, num_vars=39, param_dim=141, width=64, modes=24, n_blocks=4, padding=8):
        super().__init__()
        if n_blocks < 1 or padding < 0:
            raise ValueError("n_blocks must be positive and padding nonnegative")
        self.num_vars, self.param_dim, self.padding = num_vars, param_dim, padding
        self.lift = nn.Conv1d(num_vars + param_dim + 1, width, 1)
        self.spectral = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(n_blocks)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_blocks)])
        self.project = nn.Sequential(nn.Conv1d(width, 128, 1), nn.GELU(), nn.Conv1d(128, num_vars, 1))

    def forward(self, x, params):
        h = self.lift(joint_condition_grid(x, params, self.num_vars, self.param_dim))
        if self.padding:
            h = F.pad(h, (0, self.padding))
        for i, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            h = spectral(h) + local(h)
            if i < len(self.spectral) - 1:
                h = F.gelu(h)
        if self.padding:
            h = h[..., :-self.padding]
        return self.project(h)
