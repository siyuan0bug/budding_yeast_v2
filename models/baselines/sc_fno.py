#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity-Constrained Fourier Neural Operator (SC-FNO) & Polynomial/Multipole Operators
========================================================================================
- Paper Title: Sensitivity-Constrained Fourier Neural Operators for Forward and Inverse Problems in Parametric Differential Equations
- Authors: Abdolmehdi Behroozi, Chaopeng Shen, Daniel Kifer
- Conference: ICLR 2025
- Zenodo: https://doi.org/10.5281/zenodo.15537454

Also includes variants from the original repository:
- PNO1d: Polynomial Neural Operator (Orthogonal Legendre basis expansion)
- MPNO1d: Multipole Neural Operator
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SensitivityFourierBlock1d(nn.Module):
    """
    Sensitivity-Constrained Fourier Block:
    Integrates frequency-domain spectral kernel with parameter sensitivity gradients.
    """
    def __init__(self, in_channels, out_channels, modes=24, param_dim=141):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        self.bypass = nn.Conv1d(in_channels, out_channels, 1)

        # Sensitivity modulation layer: maps kinetic parameters into spectral weights adjustment
        self.sensitivity_proj = nn.Sequential(
            nn.Linear(param_dim, out_channels * 2),
            nn.SiLU(),
            nn.Linear(out_channels * 2, out_channels * 2)
        )
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x, params):
        B, C, T = x.shape
        orig_dtype = x.dtype
        x_float = x.float()

        # 1. 1D FFT
        x_ft = torch.fft.rfft(x_float, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)

        modes = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes] = torch.einsum("bix,iox->box", x_ft[:, :, :modes], self.weights[:, :, :modes])

        x_spectral = torch.fft.irfft(out_ft, n=T, dim=-1).to(orig_dtype)
        x_linear = self.bypass(x)

        h = x_spectral + x_linear
        h = self.norm(h)

        # 2. Sensitivity constraint modulation
        gamma, beta = self.sensitivity_proj(params).chunk(2, dim=-1)
        h = (1.0 + gamma.unsqueeze(-1)) * h + beta.unsqueeze(-1)
        return F.gelu(h)


class SC_FNO(nn.Module):
    """Sensitivity-Constrained Fourier Neural Operator (SC-FNO)"""
    def __init__(self, num_vars=39, in_dim=2, param_dim=141, width=192, modes=24, n_layers=4):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.modes = modes

        self.in_proj = nn.Conv1d(num_vars * in_dim, width, 1)
        self.blocks = nn.ModuleList([
            SensitivityFourierBlock1d(width, width, modes=modes, param_dim=param_dim)
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, 1)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)
        h = self.in_proj(x)

        for block in self.blocks:
            h = block(h, params)

        out = self.out_proj(h)
        return out


class PolynomialLayer1D(nn.Module):
    """Orthogonal Legendre Polynomial Basis Layer for PNO"""
    def __init__(self, width, poly_degree=4):
        super().__init__()
        self.width = width
        self.poly_degree = poly_degree
        self.weights_t = nn.Parameter(torch.randn(width, poly_degree + 1) * 0.02)
        self.bias = nn.Parameter(torch.zeros(width, 1))

    def legendre_basis(self, t_grid, degree):
        # t_grid: (B, 1, T) in [-1, 1]
        bases = [torch.ones_like(t_grid)]
        if degree >= 1:
            bases.append(t_grid)
        for n in range(1, degree):
            p_next = ((2 * n + 1) * t_grid * bases[-1] - n * bases[-2]) / (n + 1)
            bases.append(p_next)
        return torch.stack(bases, dim=-1)  # (B, 1, T, degree+1)

    def forward(self, x, t_grid):
        B, C, T = x.shape
        orig_dtype = x.dtype
        # Compute Legendre basis
        P_t = self.legendre_basis(t_grid.float(), self.poly_degree) # (B, 1, T, D)
        # Tensor contraction: (B, 1, T, D) x (C, D) -> (B, C, T)
        out_t = torch.einsum('bitd,cd->bct', P_t, self.weights_t).to(orig_dtype)
        out = x * out_t + self.bias
        return F.gelu(out)


class PNO1d(nn.Module):
    """Polynomial Neural Operator (PNO) using Orthogonal Legendre Bases"""
    def __init__(self, num_vars=39, in_dim=2, param_dim=141, width=192, poly_degree=4, depth=4):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.in_proj = nn.Conv1d(num_vars * in_dim + param_dim, width, 1)
        self.poly_layers = nn.ModuleList([PolynomialLayer1D(width, poly_degree) for _ in range(depth)])
        self.bypass_convs = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.out_proj = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, 1)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)

        # Expand parameters along time
        p_expand = params.unsqueeze(-1).expand(-1, -1, T)
        x_in = torch.cat([x, p_expand], dim=1)
        h = self.in_proj(x_in)

        # Normalized t grid in [-1, 1]
        t_grid = torch.linspace(-1.0, 1.0, T, device=x.device).view(1, 1, T).expand(B, 1, T)

        for poly_layer, bypass in zip(self.poly_layers, self.bypass_convs):
            h = poly_layer(h, t_grid) + bypass(h)

        out = self.out_proj(h)
        return out


class MultipoleLayer1D(nn.Module):
    """Multipole Kernel Interaction Layer for MPNO"""
    def __init__(self, width, kernel_size=15):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, kernel_size=kernel_size, padding=kernel_size // 2, groups=width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = F.gelu(self.conv2(x))
        x = self.norm(x + res)
        return x


class MPNO1d(nn.Module):
    """Multipole Neural Operator (MPNO)"""
    def __init__(self, num_vars=39, in_dim=2, param_dim=141, width=192, depth=4):
        super().__init__()
        self.num_vars = num_vars
        self.in_proj = nn.Conv1d(num_vars * in_dim + param_dim, width, 1)
        self.layers = nn.ModuleList([MultipoleLayer1D(width, kernel_size=7 + 4 * i) for i in range(depth)])
        self.out_proj = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, 1)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)
        p_expand = params.unsqueeze(-1).expand(-1, -1, T)
        h = self.in_proj(torch.cat([x, p_expand], dim=1))

        for layer in self.layers:
            h = layer(h)

        out = self.out_proj(h)
        return out
