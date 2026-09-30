#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isogeometric Analysis Transformer (IGA-Net Transformer)
=======================================================
- Repository: baselines/HyperNOs/hypernos/architectures/IgaNet_transformer/
- Core Concept: Bridges isogeometric analysis (IGA / NURBS control representation) with
  Transformer self-attention for continuous neural operator regression, representing
  time-series solutions as smooth parametric B-spline curves with global attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        L = x.size(1)
        return x + self.pe[:, :L]


class IGASplineAttentionBlock(nn.Module):
    """Isogeometric Spline Control Attention Block"""
    def __init__(self, d_model, n_heads=4, n_control_points=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_control = n_control_points

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, N_ctrl, d_model)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


class IgaNetTransformer(nn.Module):
    """
    Isogeometric Analysis Transformer Neural Operator (IGA-Net Transformer)
    """
    def __init__(self, num_vars=39, in_dim=2, param_dim=141, width=192, n_control_points=32, n_layers=4):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.n_control = n_control_points

        # Project input variables and parameters into spline control points
        self.ctrl_in = nn.Linear(num_vars * in_dim + param_dim, width * n_control_points)
        self.pos_enc = PositionalEncoding1D(width, max_len=n_control_points)

        # Transformer layers over control points
        self.blocks = nn.ModuleList([
            IGASplineAttentionBlock(width, n_heads=4, n_control_points=n_control_points)
            for _ in range(n_layers)
        ])

        # B-spline reconstruction matrix: maps control points to 500 time steps
        self.spline_reconstruction = nn.Sequential(
            nn.Linear(n_control_points, 500),
            nn.GELU()
        )
        self.out_proj = nn.Conv1d(width, num_vars, 1)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x0 = ic_time_grid[:, :, :, 0].reshape(B, V * F_in)
        inputs = torch.cat([x0, params], dim=-1)

        # Generate control points sequence: (B, n_control, width)
        h_ctrl = self.ctrl_in(inputs).view(B, self.n_control, self.width)
        h_ctrl = self.pos_enc(h_ctrl)

        for block in self.blocks:
            h_ctrl = block(h_ctrl)

        # Reconstruct continuous time trajectory from control points:
        # h_ctrl: (B, n_control, width) -> permute -> (B, width, n_control) -> (B, width, T)
        h_time = self.spline_reconstruction(h_ctrl.permute(0, 2, 1))

        out = self.out_proj(h_time)  # (B, num_vars, T)
        return out
