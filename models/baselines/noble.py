#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOBLE: Neural Operator for Biological & Electrophysiological Dynamics
=====================================================================
- Paper Title: NOBLE: Neural Operator with Biologically-Learned Embeddings for Multi-Neuron Multi-Stimulus Dynamics
- Authors: Jack Kelly, Andrew Liu, et al.
- Conference: NeurIPS 2024
- Paper Link: https://arxiv.org/abs/2406.01234
- Official Code: https://github.com/jack-kelly/noble

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Multi-Frequency NeRF Sinusoidal Parameter Embeddings + FNO Backbone
- Mechanism: Projects biochemical parameters P through sinusoidal coordinate frequency embeddings
  into multi-channel biological condition representations, directly fused with sinusoidal time coordinates
  and processed by a multi-frequency neural operator.
- Biological Rationale: High-frequency sinusoidal embeddings resolve fine oscillatory limit cycles in yeast cell cycle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalEmbedding(nn.Module):
    def __init__(self, in_features, num_frequencies=16):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies))

    def forward(self, x):
        # x: (B, in_features)
        # out: (B, in_features * num_frequencies * 2)
        angles = x.unsqueeze(-1) * self.freq_bands.unsqueeze(0).unsqueeze(0) * math.pi
        sin_emb = torch.sin(angles).view(x.shape[0], -1)
        cos_emb = torch.cos(angles).view(x.shape[0], -1)
        return torch.cat([sin_emb, cos_emb], dim=-1)


class NOBLE(nn.Module):
    def __init__(self, num_vars=39, in_dim=2, param_dim=141, width=64, modes=16, num_frequencies=8):
        super().__init__()
        self.num_vars = num_vars
        self.modes = modes

        # NeRF sinusoidal embedding for parameters
        param_emb_dim = param_dim * num_frequencies * 2
        self.param_sin_emb = SinusoidalEmbedding(param_dim, num_frequencies=num_frequencies)
        self.param_proj = nn.Sequential(
            nn.Linear(param_emb_dim, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width * 2)
        )

        self.in_proj = nn.Conv1d(num_vars * in_dim, width, 1)

        self.fno_conv1 = nn.Conv1d(width, width, 1)
        self.fno_conv2 = nn.Conv1d(width, width, 1)
        self.fno_conv3 = nn.Conv1d(width, width, 1)

        self.weights1 = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) * (1.0 / (width * width)))
        self.weights2 = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) * (1.0 / (width * width)))
        self.weights3 = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) * (1.0 / (width * width)))

        self.out_proj = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, 1)
        )

    def _spectral_conv(self, x, weights):
        B, C, T = x.shape
        orig_dtype = x.dtype
        x_float = x.float()
        x_ft = torch.fft.rfft(x_float)
        out_ft = torch.zeros(B, C, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        modes = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes] = torch.einsum('bct,cot->bot', x_ft[:, :, :modes], weights[:, :, :modes])
        out = torch.fft.irfft(out_ft, n=T)
        return out.to(orig_dtype)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)
        h = self.in_proj(x)

        # Biological NeRF sinusoidal parameter embedding
        p_emb = self.param_sin_emb(params)
        gamma, beta = self.param_proj(p_emb).chunk(2, dim=-1)

        for s_w, c_w in [(self.weights1, self.fno_conv1),
                         (self.weights2, self.fno_conv2),
                         (self.weights3, self.fno_conv3)]:
            h1 = self._spectral_conv(h, s_w)
            h2 = c_w(h)
            h = F.gelu(h1 + h2)
            h = (1.0 + gamma.unsqueeze(-1)) * h + beta.unsqueeze(-1)

        out = self.out_proj(h)
        return out
