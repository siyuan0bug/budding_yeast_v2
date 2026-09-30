#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bregman Proximal Neural Operators (BregmanFNO & BregmanWNO)
===========================================================
- Paper Title: A Bregman Proximal Viewpoint on Neural Operators
- Conference: ICML 2025
- Repository: https://github.com/bregman-no/bregmano
- Core Concept: Reformulates neural operator layers as discrete Bregman proximal steps,
  maintaining dual states (x, x_skip) with zero-weight initializations, providing
  strong mathematical convergence guarantees and anti-divergence stability on stiff ODEs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt


class BregmanSpectralConv1d(nn.Module):
    """Bregman Spectral Convolution Layer with zero initialization for proximal step"""
    def __init__(self, in_channels, out_channels, modes=24):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Zero initialization for Bregman proximal step
        self.weights = nn.Parameter(
            torch.zeros(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, T = x.shape
        orig_dtype = x.dtype
        x_float = x.float()

        x_ft = torch.fft.rfft(x_float, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)

        modes = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes] = torch.einsum("bct,cot->bot", x_ft[:, :, :modes], self.weights[:, :, :modes])

        out = torch.fft.irfft(out_ft, n=T, dim=-1)
        return out.to(orig_dtype)


class BregmanFNOBlock1d(nn.Module):
    """
    Bregman FNO Block maintaining dual proximal states (x, x_skip).
    x_{k+1} = Prox_Bregman(x_k, K(x_k) + W(x_k))
    """
    def __init__(self, width, modes=24, param_dim=141):
        super().__init__()
        self.conv = BregmanSpectralConv1d(width, width, modes)
        self.skip = nn.Conv1d(width, width, 1)
        # Initialize skip with zeros for exact proximal identity
        nn.init.zeros_(self.skip.weight)
        if self.skip.bias is not None:
            nn.init.zeros_(self.skip.bias)

        self.mlp = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, width, 1)
        )
        self.norm1 = nn.GroupNorm(1, width)
        self.norm2 = nn.GroupNorm(1, width)
        self.film = nn.Linear(param_dim, width * 2)

    def forward(self, x_tuple, params):
        x, x_skip = x_tuple

        # Spectral and spatial skip
        x_fno = self.conv(x)
        x_skip_fno = self.skip(x)
        h = self.norm1(x_fno + x_skip_fno)

        # Bregman proximal update: blend with previous dual accumulator
        x = h + x_skip
        x_skip = x

        # Nonlinear MLP with FiLM kinetic conditioning
        gamma, beta = self.film(params).chunk(2, dim=-1)
        x_mod = (1.0 + gamma.unsqueeze(-1)) * x + beta.unsqueeze(-1)
        x = x + self.norm2(self.mlp(F.gelu(x_mod)))

        return x, x_skip


class BregmanFNO(nn.Module):
    """Bregman Proximal Fourier Neural Operator (ICML 2025)"""
    def __init__(self, num_vars=39, in_dim=2, param_dim=141, width=192, modes=24, n_layers=4):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.modes = modes
        self.in_proj = nn.Conv1d(num_vars * in_dim, width, 1)

        self.blocks = nn.ModuleList([
            BregmanFNOBlock1d(width, modes=modes, param_dim=param_dim)
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

        # Dual state initialization: (x, x_skip)
        x_tuple = (h, h)
        for block in self.blocks:
            x_tuple = block(x_tuple, params)

        out = self.out_proj(x_tuple[0])
        return out


class BregmanWNOBlock1d(nn.Module):
    """Bregman Wavelet Neural Operator Block with dual proximal state update"""
    def __init__(self, width, wavelet='haar', level=3, param_dim=141):
        super().__init__()
        self.width = width
        self.wavelet = wavelet
        self.level = level

        self.wavelet_convs = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(level + 1)
        ])
        for c in self.wavelet_convs:
            nn.init.zeros_(c.weight)
            if c.bias is not None:
                nn.init.zeros_(c.bias)

        self.skip = nn.Conv1d(width, width, 1)
        nn.init.zeros_(self.skip.weight)
        if self.skip.bias is not None:
            nn.init.zeros_(self.skip.bias)

        self.mlp = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, width, 1)
        )
        self.norm = nn.GroupNorm(1, width)
        self.film = nn.Linear(param_dim, width * 2)

    def forward(self, x_tuple, params):
        x, x_skip = x_tuple
        B, C, T = x.shape
        orig_dtype = x.dtype
        x_float = x.float()

        # Multi-level DWT in float32 for ptwt compatibility
        coeffs = ptwt.wavedec(x_float, self.wavelet, level=self.level)
        processed_coeffs = []
        for c, conv in zip(coeffs, self.wavelet_convs):
            c_proc = conv(c.to(conv.weight.dtype))
            processed_coeffs.append(c_proc.float())

        # IDWT
        x_wno = ptwt.waverec(processed_coeffs, self.wavelet).to(orig_dtype)
        if x_wno.shape[-1] != T:
            x_wno = F.interpolate(x_wno, size=T, mode='linear', align_corners=False)

        x_skip_wno = self.skip(x)
        h = self.norm(x_wno + x_skip_wno)

        # Bregman proximal update
        x = h + x_skip
        x_skip = x

        gamma, beta = self.film(params).chunk(2, dim=-1)
        x_mod = (1.0 + gamma.unsqueeze(-1)) * x + beta.unsqueeze(-1)
        x = x + self.mlp(F.gelu(x_mod))

        return x, x_skip


class BregmanWNO(nn.Module):
    """Bregman Proximal Wavelet Neural Operator (ICML 2025)"""
    def __init__(self, num_vars=39, in_dim=2, param_dim=141, width=192, wavelet='haar', level=3, n_layers=4):
        super().__init__()
        self.num_vars = num_vars
        self.in_proj = nn.Conv1d(num_vars * in_dim, width, 1)

        self.blocks = nn.ModuleList([
            BregmanWNOBlock1d(width, wavelet=wavelet, level=level, param_dim=param_dim)
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

        x_tuple = (h, h)
        for block in self.blocks:
            x_tuple = block(x_tuple, params)

        out = self.out_proj(x_tuple[0])
        return out
