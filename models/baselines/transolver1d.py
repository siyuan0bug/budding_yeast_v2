#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D Transolver Operator (Transolver1d)
=====================================
Adapted from RealPDEBench (Haixu Wu et al., 2024).
Physics-Informed Attention with Continuous Physical Slices for 1D dynamical systems:
- Learns M physical reference slices across continuous time space.
- Token-to-Slice physical aggregation with linear complexity O(T * M).
- Slice-to-Token physical field reconstruction.
- Parameter FiLM conditioning for biochemical kinetic constants.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsAttention1d(nn.Module):
    """Physics-informed Slice Attention in 1D Continuous Space."""
    def __init__(self, d_model, n_slices=16, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_slices = n_slices
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable physical reference slice queries
        self.slices = nn.Parameter(torch.randn(1, n_slices, d_model) * 0.02)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, _ = x.shape
        slices = self.slices.expand(B, -1, -1)  # (B, M, d_model)

        # 1. Aggregate tokens into M physical slices: Q = Slices, K = x, V = x
        q_slice = self.q_proj(slices).view(B, self.n_slices, self.n_heads, self.head_dim).transpose(1, 2)
        k_tok = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_tok = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Slice attention weights: (B, H, M, T)
        attn_to_slice = F.softmax(torch.einsum('bhmd, bhtd -> bhmt', q_slice, k_tok) / math.sqrt(self.head_dim), dim=-1)
        slice_repr = torch.einsum('bhmt, bhtd -> bhmd', attn_to_slice, v_tok)  # (B, H, M, Dh)

        # 2. Propagate slices back to continuous tokens: Q = x, K = Slices, V = Slice_Repr
        q_tok = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_slice = self.k_proj(slices).view(B, self.n_slices, self.n_heads, self.head_dim).transpose(1, 2)

        attn_to_tok = F.softmax(torch.einsum('bhtd, bhmd -> bhtm', q_tok, k_slice) / math.sqrt(self.head_dim), dim=-1)
        out = torch.einsum('bhtm, bhmd -> bhtd', attn_to_tok, slice_repr)  # (B, H, T, Dh)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class TransolverBlock1d(nn.Module):
    def __init__(self, d_model, n_slices=16, n_heads=4, d_ff=256, param_dim=141):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhysicsAttention1d(d_model, n_slices=n_slices, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.film1 = nn.Linear(param_dim, d_model * 2)
        self.film2 = nn.Linear(param_dim, d_model * 2)

    def forward(self, x, p):
        g1, b1 = self.film1(p).unsqueeze(1).chunk(2, dim=-1)
        h1 = self.norm1(x)
        h1 = (1.0 + g1) * h1 + b1
        x = x + self.attn(h1)

        g2, b2 = self.film2(p).unsqueeze(1).chunk(2, dim=-1)
        h2 = self.norm2(x)
        h2 = (1.0 + g2) * h2 + b2
        x = x + self.ffn(h2)
        return x


class Transolver1d(nn.Module):
    """
    1D Transolver Physics-Informed Attention Operator for Yeast Dynamics:
    - Input: ic_time_grid (B, 39, 2, 500) and params (B, 141)
    - Output: (B, 39, 500)
    """
    def __init__(self, num_vars=39, param_dim=141, width=128, n_blocks=4, n_slices=16, n_heads=4):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        in_channels = num_vars * 2  # 78

        self.stem = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([
            TransolverBlock1d(width, n_slices=n_slices, n_heads=n_heads, d_ff=width * 2, param_dim=param_dim)
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
        x = ic_time_grid.reshape(B, V * F_in, T).permute(0, 2, 1)  # (B, T, 78)

        h = self.stem(x)
        for block in self.blocks:
            h = block(h, params)

        out = self.head(h)
        return out.permute(0, 2, 1)  # (B, 39, 500)
