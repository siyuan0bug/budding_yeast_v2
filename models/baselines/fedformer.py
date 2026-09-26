#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting
======================================================================================
- Paper Title: FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting
- Authors: Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, Rong Jin (Alibaba DAMO Academy)
- Conference: ICML 2022
- Paper Link: https://arxiv.org/abs/2201.12740
- Official Code: https://github.com/MAZiqing/FEDformer

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Frequency-Enhanced Cross-Attention + Parameter FiLM
- Mechanism: Combines classical series decomposition with Frequency Enhanced Block (FEB-f / FEB-w)
  in compact frequency domain subspaces. Kinetic parameters P (141-dim) modulate seasonal and trend
  frequency modes via FiLM affine parameters gamma(P), beta(P):
  h' = (1 + gamma(P)) * h + beta(P).
- Biological Rationale: Efficiently captures long-range periodicity and phase shifts in biological rhythms.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierBlock(nn.Module):
    """Fourier Enhanced Block (FEB-f) operating in frequency domain."""
    def __init__(self, in_channels, out_channels, modes=64):
        super().__init__()
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        orig_dtype = x.dtype
        x_ft = torch.fft.rfft(x.float(), dim=-1)
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.weights.shape[1], x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights[:, :, :modes])
        x_out = torch.fft.irfft(out_ft, n=L, dim=-1)
        return x_out.to(orig_dtype)


class MovingAvgBlock(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, :, 0:1].repeat(1, 1, pad_len)
        end = x[:, :, -1:].repeat(1, 1, pad_len)
        x_pad = torch.cat([front, x, end], dim=-1)
        out = self.avg(x_pad)
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-1], mode='linear', align_corners=False)
        return out


class FEDformer(nn.Module):
    """
    FEDformer for Budding Yeast Cell Cycle Dynamics:
    - Input: ic_time_grid (B, 39, 2, 500), params (B, 141)
    - Output: (B, 39, 500)
    """
    def __init__(self, num_vars=39, param_dim=141, seq_len=500, pred_len=500,
                 modes=64, width=128, n_blocks=3, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.in_proj = nn.Conv1d(num_vars * 2, width, 1)
        self.moving_avg = MovingAvgBlock(kernel_size=25)

        self.fourier_blocks = nn.ModuleList([
            FourierBlock(width, width, modes=modes) for _ in range(n_blocks)
        ])
        self.w_convs = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(n_blocks)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, width) for _ in range(n_blocks)
        ])
        self.film = nn.Linear(param_dim, width * 2)
        self.out_proj = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, 1)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)
        h = self.in_proj(x)  # (B, width, 500)

        # Decompose trend and seasonal
        trend = self.moving_avg(h)
        seasonal = h - trend

        gamma, beta = self.film(params).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        # Multi-layer frequency enhanced blocks
        for feb, w, norm in zip(self.fourier_blocks, self.w_convs, self.norms):
            res = seasonal
            h1 = feb(seasonal)
            h2 = w(seasonal)
            seasonal = norm(h1 + h2)
            seasonal = (1.0 + gamma) * seasonal + beta
            seasonal = F.gelu(seasonal) + res

        h_out = seasonal + trend
        out = self.out_proj(h_out)  # (B, 39, 500)
        return out
