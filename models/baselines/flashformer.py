#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flashformer: Linearized Fast Attention Transformer
==================================================
- Paper Title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- Authors: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford) / Tsinghua iTransformer benchmark
- Conference: NeurIPS 2022
- Paper Link: https://arxiv.org/abs/2205.14135
- Official Code: https://github.com/Dao-AILab/flash-attention / https://github.com/thuml/iTransformer

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Linearized Kernel Attention + Parameter FiLM
- Mechanism: Replaces quadratic softmax attention with kernelized linear attention:
  phi(x) = elu(x) + 1. Features are modulated by FiLM affine parameters gamma(P), beta(P):
  h' = (1 + gamma(P)) * h + beta(P).
- Biological Rationale: Highly scalable continuous attention for high-frequency yeast metabolic oscillations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearizedAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        Q = F.elu(self.q_proj(x)).add(1.0).view(B, L, H, D).permute(0, 2, 1, 3)
        K = F.elu(self.k_proj(x)).add(1.0).view(B, L, H, D).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, L, H, D).permute(0, 2, 1, 3)

        KV = torch.einsum('bhld,bhlm->bhdm', K, V)
        K_sum = K.sum(dim=-2, keepdim=True)
        Z = torch.einsum('bhld,bhid->bhl', Q, K_sum).unsqueeze(-1) + self.eps

        out = torch.einsum('bhld,bhdm->bhlm', Q, KV) / Z
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        return self.out_proj(out)


class Flashformer(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, seq_len=500, pred_len=500,
                 d_model=128, n_heads=4, e_layers=3, d_ff=256, dropout=0.1, **kwargs):
        super().__init__()
        self.in_proj = nn.Linear(num_vars * 2, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': LinearizedAttention(d_model, n_heads=n_heads),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(e_layers)
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
            h = layer['norm1'](h + layer['attn'](h))
            h = layer['norm2'](h + layer['ffn'](h))

        out = self.out_proj(h)
        return out.permute(0, 2, 1).contiguous()
