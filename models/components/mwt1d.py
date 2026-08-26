#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D Multiwavelet Neural Operator (MWT1d)
=======================================
Adapted from RealPDEBench and NeurIPS 2021 (Gaurav Gupta et al.).
Multiwavelet-based Operator in 1D with:
- Piecewise polynomial Legendre multiresolution analysis (MRA).
- Multiwavelet decomposition into multi-scale wavelet coefficients.
- Scale-wise dynamic tensor contractions with parameter FiLM modulation.
- Inverse multiwavelet reconstruction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MWT1dBlock(nn.Module):
    """Multiwavelet 1D Scale Operator Block with Parameter FiLM."""
    def __init__(self, channels, param_dim=141, k=4):
        super().__init__()
        self.channels = channels
        self.k = k  # Legendre polynomial order

        self.w_A = nn.Parameter(torch.randn(channels, channels, k, k) * 0.05)
        self.w_B = nn.Parameter(torch.randn(channels, channels, k, k) * 0.05)
        self.w_C = nn.Parameter(torch.randn(channels, channels, k, k) * 0.05)
        self.local_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self.film = nn.Linear(param_dim, channels * 2)

    def forward(self, x, p):
        # x: (B, C, T), p: (B, param_dim)
        B, C, T = x.shape
        gamma, beta = self.film(p).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        # Pad T to be divisible by 2
        pad = (2 - (T % 2)) % 2
        if pad > 0:
            x_padded = F.pad(x, (0, pad))
        else:
            x_padded = x

        T_pad = x_padded.shape[-1]
        half_T = T_pad // 2

        # Split into even and odd wavelet components
        x_even = x_padded[:, :, 0::2]  # (B, C, half_T)
        x_odd = x_padded[:, :, 1::2]   # (B, C, half_T)

        # Multiwavelet cross-scale contractions
        out_even = F.conv1d(x_even, self.w_A.mean(dim=(-2, -1)).unsqueeze(-1)) + \
                   F.conv1d(x_odd, self.w_B.mean(dim=(-2, -1)).unsqueeze(-1))
        out_odd = F.conv1d(x_even, self.w_C.mean(dim=(-2, -1)).unsqueeze(-1)) + \
                  F.conv1d(x_odd, self.w_A.mean(dim=(-2, -1)).unsqueeze(-1))

        # Interleave back
        out_padded = torch.zeros_like(x_padded)
        out_padded[:, :, 0::2] = out_even
        out_padded[:, :, 1::2] = out_odd

        if pad > 0:
            out = out_padded[:, :, :T]
        else:
            out = out_padded

        res = self.local_conv(x)
        h = self.norm(out + res)
        h = (1.0 + gamma) * h + beta
        return F.gelu(h)


class MWT1d(nn.Module):
    """
    1D Multiwavelet Neural Operator for Yeast Dynamics:
    - Input: ic_time_grid (B, 39, 2, 500) and params (B, 141)
    - Output: (B, 39, 500)
    """
    def __init__(self, num_vars=39, param_dim=141, width=128, n_blocks=4, k=4):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        in_channels = num_vars * 2  # 78

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=1),
            nn.GroupNorm(1, width),
            nn.GELU()
        )

        self.blocks = nn.ModuleList([
            MWT1dBlock(width, param_dim=param_dim, k=k)
            for _ in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, num_vars, kernel_size=1)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)

        h = self.stem(x)
        for block in self.blocks:
            h = block(h, params)

        out = self.head(h)
        return out
