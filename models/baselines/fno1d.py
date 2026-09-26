#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard 1D Fourier Neural Operator (FNO1d)
===========================================
Adapted from RealPDEBench and PDEBench (Zongyi Li et al., ICLR 2021).
Channel-mode 1D FNO with:
- Continuous 1D Fourier spectral convolutions in frequency domain.
- 1x1 local convolution spatial skip connections.
- Parameter FiLM (Feature-wise Linear Modulation) conditioning.
- AMP (bfloat16/float32) safe FFT/iFFT operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        orig_dtype = x.dtype
        x_float = x.float()
        x_ft = torch.fft.rfft(x_float, dim=-1)

        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        modes = min(self.modes1, x_ft.shape[-1])
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights1[:, :, :modes])

        x_out = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x_out.to(orig_dtype)


class FNO1dBlock(nn.Module):
    def __init__(self, width, modes, param_dim=141):
        super().__init__()
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(1, width)
        self.film = nn.Linear(param_dim, width * 2)

    def forward(self, x, p):
        gamma, beta = self.film(p).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        x1 = self.conv(x)
        x2 = self.w(x)
        out = self.norm(x1 + x2)
        out = (1.0 + gamma) * out + beta
        return F.gelu(out)


class FNO1d_Standard(nn.Module):
    """
    Standard 1D Fourier Neural Operator for Yeast Dynamics:
    - Input: ic_time_grid (B, 39, 2, 500) and params (B, 141)
    - Output: (B, 39, 500)
    """
    def __init__(self, num_vars=39, param_dim=141, modes=32, width=128, n_blocks=4):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.modes = modes
        self.in_channels = num_vars * 2  # IC + Time grid = 78

        self.p_in = nn.Conv1d(self.in_channels, width, 1)
        self.blocks = nn.ModuleList([FNO1dBlock(width, modes, param_dim) for _ in range(n_blocks)])
        self.q_out = nn.Sequential(
            nn.Conv1d(width, 128, 1),
            nn.GELU(),
            nn.Conv1d(128, num_vars, 1)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)
        x = self.p_in(x)

        for block in self.blocks:
            x = block(x, params)

        out = self.q_out(x)
        return out
