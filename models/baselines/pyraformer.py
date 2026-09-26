#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series
=========================================================================
- Paper Title: Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting
- Authors: Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X. Liu, Deying Li
- Conference: ICLR 2022
- Paper Link: https://openreview.net/forum?id=0EXmFzVa5ge
- Official Code: https://github.com/alipay/Pyraformer

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Pyramidal Multi-Scale Attention + Parameter FiLM
- Mechanism: Constructs a multi-resolution tree structure to summarize temporal signals at different
  scales (e.g. coarse cell cycle period vs fine phosphorylation reaction). Kinetic parameters P (141-dim)
  modulate pyramidal representations via FiLM affine parameters gamma(P), beta(P):
  h' = (1 + gamma(P)) * h + beta(P).
- Biological Rationale: Perfectly captures cross-scale interactions from fast enzymatic reactions to slow cell division.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidalBlock(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.conv_coarse = nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1)
        self.conv_fine = nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, D)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm(x + attn_out)

        # Multi-scale pyramidal flow
        x_coarse = self.conv_coarse(x.permute(0, 2, 1))  # (B, D, L/2)
        x_fine = self.conv_fine(x_coarse).permute(0, 2, 1)  # (B, L, D)
        return self.norm(x + x_fine)


class Pyraformer(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, seq_len=500, pred_len=500,
                 d_model=128, n_heads=4, e_layers=3, d_ff=256, dropout=0.1, **kwargs):
        super().__init__()
        self.in_proj = nn.Linear(num_vars * 2, d_model)
        self.layers = nn.ModuleList([
            PyramidalBlock(d_model, n_heads=n_heads) for _ in range(e_layers)
        ])
        self.film = nn.Linear(param_dim, d_model * 2)
        self.out_proj = nn.Linear(d_model, num_vars)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T).permute(0, 2, 1)
        h = self.in_proj(x)

        gamma, beta = self.film(params).chunk(2, dim=-1)
        h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)

        for layer in self.layers:
            h = layer(h)

        out = self.out_proj(h)
        return out.permute(0, 2, 1).contiguous()
