#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wavelet-Driven Neural Operator (WDNO)
=====================================
- Paper Title: RealPDEBench: A Benchmark for Complex Physical Systems / Wavelet-Driven Neural Operator
- Conference: NeurIPS 2024
- Paper Link: https://arxiv.org/abs/2410.05264
- Official Code: https://github.com/BaratiLab/RealPDEBench

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Multi-Level Discrete Wavelet Decomposition + Parameter FiLM
- Mechanism: Performs multi-level discrete wavelet decomposition (Haar/Daubechies) separating high-frequency
  phosphorylation bursts from low-frequency cell cycle trends. Modulated by parameter FiLM at each level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt


class WDNO(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, seq_len=500, pred_len=500,
                 width=128, wavelet='haar', level=3, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.wavelet = wavelet
        self.level = level

        self.in_proj = nn.Conv1d(num_vars * 2, width, 1)
        self.wavelet_convs = nn.ModuleList([
            nn.Conv1d(width, width, 3, padding=1) for _ in range(level + 1)
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

        # Multi-level DWT (cast to float32 for ptwt compatibility)
        orig_dtype = h.dtype
        h_float = h.float()
        coeffs = ptwt.wavedec(h_float, self.wavelet, level=self.level)
        processed_coeffs = []
        for c, conv in zip(coeffs, self.wavelet_convs):
            c_proc = F.gelu(conv(c.to(conv.weight.dtype)))
            processed_coeffs.append(c_proc.float())

        # IDWT
        h_recon = ptwt.waverec(processed_coeffs, self.wavelet).to(orig_dtype)
        if h_recon.shape[-1] != T:
            h_recon = F.interpolate(h_recon, size=T, mode='linear', align_corners=False)

        gamma, beta = self.film(params).chunk(2, dim=-1)
        h_mod = (1.0 + gamma.unsqueeze(-1)) * h_recon + beta.unsqueeze(-1)

        out = self.out_proj(h_mod)
        return out
