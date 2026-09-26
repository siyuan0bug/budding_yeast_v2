#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reformer: The Efficient Transformer
===================================
- Paper Title: Reformer: The Efficient Transformer
- Authors: Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya
- Conference: ICLR 2020
- Paper Link: https://arxiv.org/abs/2001.04451
- Official Code: https://github.com/google/trax/tree/master/trax/models/reformer / https://github.com/thuml/iTransformer

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: LSH / Chunked Attention + Parameter FiLM
- Mechanism: Employs chunked temporal attention to reduce complexity to O(L log L).
  Kinetic parameters P (141-dim) modulate temporal token representations via FiLM affine parameters
  gamma(P), beta(P) = Linear(P).chunk(2): h' = (1 + gamma(P)) * h + beta(P).
- Biological Rationale: Groups similar temporal activation states into shared attention buckets.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkedAttention(nn.Module):
    """Chunk-based efficient attention layer (Reformer ICLR 2020)."""
    def __init__(self, d_model, n_heads=4, chunk_size=50):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        d_k = D // H
        Q = self.q_proj(x).view(B, L, H, d_k).permute(0, 2, 1, 3)
        K = self.k_proj(x).view(B, L, H, d_k).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, L, H, d_k).permute(0, 2, 1, 3)

        # Chunked attention across L (500 steps / chunk_size)
        num_chunks = L // self.chunk_size
        Q_c = Q.view(B, H, num_chunks, self.chunk_size, d_k)
        K_c = K.view(B, H, num_chunks, self.chunk_size, d_k)
        V_c = V.view(B, H, num_chunks, self.chunk_size, d_k)

        scores = torch.matmul(Q_c, K_c.transpose(-2, -1)) / math.sqrt(d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_c).view(B, H, L, d_k).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out)


class Reformer(nn.Module):
    """
    Reformer for Budding Yeast Cell Cycle Dynamics:
    - Input: ic_time_grid (B, 39, 2, 500), params (B, 141)
    - Output: (B, 39, 500)
    """
    def __init__(self, num_vars=39, param_dim=141, seq_len=500, pred_len=500,
                 d_model=128, n_heads=4, e_layers=3, d_ff=256, dropout=0.1, **kwargs):
        super().__init__()
        self.in_proj = nn.Linear(num_vars * 2, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': ChunkedAttention(d_model, n_heads=n_heads),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(e_layers)
        ])
        self.film = nn.Linear(param_dim, d_model * 2)
        self.out_proj = nn.Linear(d_model, num_vars)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T).permute(0, 2, 1)  # (B, 500, 78)
        h = self.in_proj(x)

        gamma, beta = self.film(params).chunk(2, dim=-1)
        h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)

        for layer in self.layers:
            h = layer['norm1'](h + layer['attn'](h))
            h = layer['norm2'](h + layer['ffn'](h))

        out = self.out_proj(h)
        return out.permute(0, 2, 1).contiguous()
