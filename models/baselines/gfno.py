#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G-FNO: Group Equivariant Fourier Neural Operator
================================================
- Paper Title: Group Equivariant Fourier Neural Operators for Partial Differential Equations
- Authors: Jacob Helwig, Xuan Zhang, Shenglong Xu, Shuiwang Ji
- Conference: ICLR 2023
- Paper Link: https://arxiv.org/abs/2301.12586
- Official Code: https://github.com/divelab/AIRS/tree/main/OpenPDE/G_FNO

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Group-Equivariant Spectral Conditioning + Parameter FiLM
- Mechanism: Implements Fourier operator layers that are equivariant under discrete transformation
  groups (such as temporal reflection Z2 symmetry). Parameters P condition the spectral representations
  in an invariant channel space before equivariant Fourier projection.
- Biological Rationale: Enforces symmetry inductive biases while allowing kinetic parameters to direct the forward flux.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralGConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, T = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)

        modes = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes] = torch.einsum('bct,cot->bot', x_ft[:, :, :modes], self.weights[:, :, :modes])

        out = torch.fft.irfft(out_ft, n=T)
        # Symmetrized group equivariant combination (Z2 reflection invariance)
        x_flip = torch.flip(x, dims=[-1])
        x_flip_ft = torch.fft.rfft(x_flip)
        out_flip_ft = torch.zeros_like(out_ft)
        out_flip_ft[:, :, :modes] = torch.einsum('bct,cot->bot', x_flip_ft[:, :, :modes], self.weights[:, :, :modes])
        out_flip = torch.fft.irfft(out_flip_ft, n=T)

        return 0.5 * (out + torch.flip(out_flip, dims=[-1]))


class G_FNO(nn.Module):
    def __init__(self, num_vars=39, in_dim=2, param_dim=141, width=64, modes=16, n_layers=4):
        super().__init__()
        self.num_vars = num_vars
        self.in_proj = nn.Conv1d(num_vars * in_dim, width, 1)

        self.spectral_layers = nn.ModuleList([
            SpectralGConv1d(width, width, modes=modes) for _ in range(n_layers)
        ])
        self.bypass_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(n_layers)
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
        h = self.in_proj(x)

        gamma, beta = self.film(params).chunk(2, dim=-1)

        for s_layer, b_layer in zip(self.spectral_layers, self.bypass_layers):
            h = s_layer(h) + b_layer(h)
            h = (1.0 + gamma.unsqueeze(-1)) * h + beta.unsqueeze(-1)
            h = F.gelu(h)

        out = self.out_proj(h)
        return out
