#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralTemporalHybridOperator (ST-HyperNO)
===========================================
End-to-end continuous spectral-temporal hybrid neural operator.
Seamlessly unifies Global Continuous 1D Fourier Spectral Convolutions (for global limit-cycle
phase locking) with Multi-Scale Dilated Temporal Residual Convolutions (width=128, for steep
biochemical threshold switches) via bidirectional cross-domain gated interaction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Continuous Spectral Convolution with AMP Float32 stability."""
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
        orig_dtype = x.dtype
        x_f32 = x.float()
        B, C, T = x_f32.shape

        x_ft = torch.fft.rfft(x_f32, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)

        modes = min(self.modes1, x_ft.shape[-1])
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights1[:, :, :modes])

        x_out = torch.fft.irfft(out_ft, n=T, dim=-1)
        return x_out.to(orig_dtype)


class SpectralTemporalHybridBlock(nn.Module):
    """
    Unified Residual Block with Parallel Spectral & Temporal Convolutions + Cross-Domain Gated Fusion.
    """
    def __init__(self, width=128, modes=32, dilation=1, param_dim=141):
        super().__init__()
        self.width = width

        # 1. Global Spectral Branch
        self.spectral_conv = SpectralConv1d(width, width, modes1=modes)
        self.spectral_linear = nn.Conv1d(width, width, kernel_size=1)
        self.spectral_norm = nn.GroupNorm(1, width)

        # 2. Local Dilated Temporal Branch (width=128)
        self.temp_norm1 = nn.GroupNorm(1, width)
        self.temp_conv1 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.temp_norm2 = nn.GroupNorm(1, width)
        self.temp_conv2 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)

        # 3. FiLM Parameter Conditioning
        self.film = nn.Sequential(
            nn.Linear(param_dim, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width * 2)
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        # 4. Bidirectional Cross-Domain Gating
        self.gate_net = nn.Sequential(
            nn.Conv1d(width * 2, width, kernel_size=1),
            nn.Sigmoid()
        )
        self.act = nn.GELU()

    def forward(self, h_spec, h_temp, p):
        """
        h_spec: (B, width, T) - Spectral Stream
        h_temp: (B, width, T) - Temporal Stream
        p: (B, param_dim)
        """
        res_spec = h_spec
        res_temp = h_temp

        # A. Spectral Update
        s_out = self.spectral_conv(h_spec) + self.spectral_linear(h_spec)
        s_out = self.act(self.spectral_norm(s_out))

        # B. Temporal Update with FiLM
        t_h = self.temp_norm1(h_temp)
        film_val = self.film(p).unsqueeze(-1)
        gamma, beta = torch.chunk(film_val, 2, dim=1)
        t_h = t_h * (1.0 + torch.tanh(gamma)) + beta
        t_h = self.act(self.temp_conv1(t_h))
        t_h = self.temp_norm2(t_h)
        t_out = self.temp_conv2(t_h)

        # C. Bidirectional Gated Interaction
        cat_feat = torch.cat([s_out, t_out], dim=1)
        gate = self.gate_net(cat_feat) # (B, width, T)

        h_spec_next = self.act(res_spec + s_out * gate + t_out * (1.0 - gate))
        h_temp_next = self.act(res_temp + t_out * (1.0 - gate) + s_out * gate)

        return h_spec_next, h_temp_next


class SpectralTemporalHybridOperator(nn.Module):
    """
    Flagship ST-HyperNO Architecture
    """
    def __init__(self, num_vars=39, param_dim=141, modes=32, width=128, n_blocks=4, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.modes = modes
        self.width = width
        self.n_blocks = n_blocks

        # 1. Stem Projections: (B, 39*2, T) -> (B, width, T)
        in_channels = num_vars * 2
        self.stem_spec = nn.Conv1d(in_channels, width, kernel_size=1)
        self.stem_temp = nn.Conv1d(in_channels, width, kernel_size=5, padding=2)

        # 2. Multi-layer Spectral-Temporal Hybrid Blocks
        dilations = [2 ** (i % 4) for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            SpectralTemporalHybridBlock(width=width, modes=modes, dilation=dilations[i], param_dim=param_dim)
            for i in range(n_blocks)
        ])

        # 3. Dynamic Channel Cross-Attention across 39 variables
        self.to_vars = nn.Linear(width, num_vars * 16)
        self.from_vars = nn.Linear(num_vars * 16, width)
        self.q_proj = nn.Linear(16, 16)
        self.k_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)

        # 4. Output Projection Head
        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

    def forward(self, x, p):
        """
        x: (B, 39, 2, T) or (B, 39, T)
        p: (B, 141)
        """
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T) # (B, 78, T)
        else:
            B, V, T = x.shape
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, time_grid], dim=1)

        # 1. Dual-stream stem initialization
        h_spec = self.stem_spec(x_in)
        h_temp = self.stem_temp(x_in)

        # 2. Hybrid Block Processing
        for block in self.blocks:
            h_spec, h_temp = block(h_spec, h_temp, p)

        # 3. Fused Representation
        h = h_spec + h_temp # (B, width, T)

        # 4. Channel Self-Attention across 39 variables
        x_T = h.permute(0, 2, 1) # (B, T, width)
        x_vars = self.to_vars(x_T).reshape(B, T, self.num_vars, 16)
        Q = self.q_proj(x_vars)
        K = self.k_proj(x_vars)
        V = self.v_proj(x_vars)
        attn = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) / 4.0, dim=-1)
        out_vars = torch.matmul(attn, V).reshape(B, T, self.num_vars * 16)
        h_attn = self.from_vars(out_vars).permute(0, 2, 1)
        h = h + h_attn

        # 5. Output Head
        out = self.head(h) # (B, 39, T)
        return out
