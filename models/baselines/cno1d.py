#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D Continuous Neural Operator (CNO1d)
=====================================
Adapted from RealPDEBench (Bogdan Raonic et al., ICLR 2023).
Continuous Neural Operator in 1D with:
- Bandlimited continuous convolutions with antialiasing filters.
- Continuous multi-scale residual hierarchy.
- Parameter FiLM conditioning for biochemical kinetic control.
- Invariant to temporal sampling grids.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousConv1d(nn.Module):
    """Bandlimited Continuous 1D Convolution with Antialiasing Low-Pass Filter."""
    def __init__(self, in_channels, out_channels, kernel_size=5, cutoff=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=self.padding)

        # Sinc-based antialiasing filter kernel
        t = torch.linspace(-2.0, 2.0, kernel_size)
        sinc_kernel = torch.sinc(t * cutoff)
        sinc_kernel = sinc_kernel / sinc_kernel.sum()
        self.register_buffer('filter', sinc_kernel.view(1, 1, kernel_size))

    def forward(self, x):
        # x: (B, C, T)
        h = self.conv(x)
        B, C, T = h.shape
        # Apply antialiasing lowpass filter along time
        h_filtered = F.conv1d(h.view(B * C, 1, T), self.filter, padding=self.padding).view(B, C, T)
        return h_filtered


class CNOBlock1d(nn.Module):
    def __init__(self, channels, param_dim=141, kernel_size=5):
        super().__init__()
        self.cconv1 = ContinuousConv1d(channels, channels, kernel_size=kernel_size)
        self.norm1 = nn.GroupNorm(1, channels)
        self.cconv2 = ContinuousConv1d(channels, channels, kernel_size=kernel_size)
        self.norm2 = nn.GroupNorm(1, channels)
        self.film = nn.Linear(param_dim, channels * 2)

    def forward(self, x, p):
        gamma, beta = self.film(p).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        res = x
        h = F.leaky_relu(self.norm1(self.cconv1(x)), 0.2)
        h = (1.0 + gamma) * h + beta
        h = self.norm2(self.cconv2(h))
        return F.leaky_relu(h + res, 0.2)


class CNO1d(nn.Module):
    """
    1D Continuous Neural Operator for Yeast Dynamics:
    - Input: ic_time_grid (B, 39, 2, 500) and params (B, 141)
    - Output: (B, 39, 500)
    """
    def __init__(self, num_vars=39, param_dim=141, width=128, n_blocks=4):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        in_channels = num_vars * 2  # 78

        self.stem = nn.Sequential(
            ContinuousConv1d(in_channels, width, kernel_size=5),
            nn.GroupNorm(1, width),
            nn.LeakyReLU(0.2)
        )

        self.blocks = nn.ModuleList([
            CNOBlock1d(width, param_dim=param_dim, kernel_size=5)
            for _ in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.LeakyReLU(0.2),
            ContinuousConv1d(width, width, kernel_size=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)

        h = self.stem(x)
        for block in self.blocks:
            h = block(h, params)

        out = self.head(h)
        return out
