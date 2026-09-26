#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flowformer: Linearizing Transformers with Conservation Flows
============================================================
- Paper Title: Flowformer: Linearizing Transformers with Conservation Flows
- Authors: Haixu Wu, Jialong Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long (Tsinghua University)
- Conference: ICML 2022
- Paper Link: https://arxiv.org/abs/2202.06258
- Official Code: https://github.com/thuml/Flowformer

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Conservation-Flow Linear Attention + Parameter FiLM
- Mechanism: Replaces standard softmax attention with conservation-flow kernel attention:
  FlowAttention(Q, K, V) = (Q_in / (Q_out K_in^T 1)) * (K_out^T V), maintaining flow conservation.
  Biochemical kinetic parameters P (141-dim) modulate temporal flow tokens via FiLM affine parameters
  gamma(P), beta(P) = Linear(P).chunk(2): h' = (1 + gamma(P)) * h + beta(P).
- Biological Rationale: Conservation flows mirror conservation of mass in biochemical reaction networks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowAttention(nn.Module):
    """Flow Attention Layer maintaining incoming and outgoing conservation."""
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

    def kernel(self, x):
        return F.elu(x) + 1.0

    def forward(self, x):
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        Q = self.kernel(self.q_proj(x)).view(B, L, H, D).permute(0, 2, 1, 3)
        K = self.kernel(self.k_proj(x)).view(B, L, H, D).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, L, H, D).permute(0, 2, 1, 3)

        # Linear attention: KV = K^T V, shape: (B, H, D, D)
        KV = torch.einsum('bhld,bhlm->bhdm', K, V)
        K_sum = K.sum(dim=-2, keepdim=True)  # (B, H, 1, D)
        norm = torch.einsum('bhld,bhid->bhl', Q, K_sum).unsqueeze(-1) + self.eps

        out = torch.einsum('bhld,bhdm->bhlm', Q, KV) / norm
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        return self.out_proj(out)


class Flowformer(nn.Module):
    """
    Flowformer for Budding Yeast Cell Cycle Dynamics:
    - Input: ic_time_grid (B, 39, 2, 500), params (B, 141)
    - Output: (B, 39, 500)
    """
    def __init__(self, num_vars=39, param_dim=141, seq_len=500, pred_len=500,
                 d_model=128, n_heads=4, e_layers=3, d_ff=256, dropout=0.1, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.in_proj = nn.Linear(num_vars * 2, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': FlowAttention(d_model, n_heads=n_heads),
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
        h = self.in_proj(x)  # (B, 500, d_model)

        gamma, beta = self.film(params).chunk(2, dim=-1)
        h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)

        for layer in self.layers:
            h = layer['norm1'](h + layer['attn'](h))
            h = layer['norm2'](h + layer['ffn'](h))

        out = self.out_proj(h)  # (B, 500, 39)
        return out.permute(0, 2, 1).contiguous()
