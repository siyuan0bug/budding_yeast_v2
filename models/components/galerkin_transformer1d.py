#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D Galerkin Transformer Operator (GalerkinTransformer1d)
========================================================
Adapted from RealPDEBench (Shuhao Cao et al., ICLR 2021).
Linear-complexity continuous Hilbert space attention for 1D dynamical systems:
- Softmax-free Galerkin inner product kernel: Attn(Q, K, V) = Q * (K^T * V) / T.
- Multi-head continuous Galerkin attention with O(T * D_h^2) linear time complexity.
- Parameter FiLM conditioning for biochemical ODE parameters.
- Continuous time positional encoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GalerkinAttention1d(nn.Module):
    """Multi-Head Galerkin Attention in 1D Continuous Function Space."""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)

        # Galerkin continuous inner product: K^T * V / T -> (B, H, Dh, Dh)
        kv = torch.einsum('bhtd, bhte -> bhde', k, v) / float(T)

        # Project back: Q * (K^T V) -> (B, H, T, Dh)
        out = torch.einsum('bhtd, bhde -> bhte', q, kv)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class GalerkinTransformerBlock1d(nn.Module):
    def __init__(self, d_model, n_heads=4, d_ff=256, param_dim=141):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GalerkinAttention1d(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.film1 = nn.Linear(param_dim, d_model * 2)
        self.film2 = nn.Linear(param_dim, d_model * 2)

    def forward(self, x, p):
        # x: (B, T, d_model), p: (B, param_dim)
        g1, b1 = self.film1(p).unsqueeze(1).chunk(2, dim=-1)
        h1 = self.norm1(x)
        h1 = (1.0 + g1) * h1 + b1
        x = x + self.attn(h1)

        g2, b2 = self.film2(p).unsqueeze(1).chunk(2, dim=-1)
        h2 = self.norm2(x)
        h2 = (1.0 + g2) * h2 + b2
        x = x + self.ffn(h2)
        return x


class GalerkinTransformer1d(nn.Module):
    """
    1D Galerkin Transformer Operator for Yeast Dynamics:
    - Input: ic_time_grid (B, 39, 2, 500) and params (B, 141)
    - Output: (B, 39, 500)
    """
    def __init__(self, num_vars=39, param_dim=141, width=128, n_blocks=4, n_heads=4):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        in_channels = num_vars * 2  # 78

        self.stem = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([
            GalerkinTransformerBlock1d(width, n_heads=n_heads, d_ff=width * 2, param_dim=param_dim)
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, num_vars)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        # Permute to (B, T, 78) for Transformer sequence format
        x = ic_time_grid.reshape(B, V * F_in, T).permute(0, 2, 1)

        h = self.stem(x)  # (B, T, width)
        for block in self.blocks:
            h = block(h, params)

        out = self.head(h)  # (B, T, 39)
        return out.permute(0, 2, 1)  # (B, 39, 500)
