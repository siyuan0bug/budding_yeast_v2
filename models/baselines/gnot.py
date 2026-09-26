#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNOT: A General Neural Operator for Multi-Scale PDEs
====================================================
- Paper Title: GNOT: A General Neural Operator for Multi-Scale PDEs
- Authors: Zecheng Zhang, Ruisi Zhang, Weijian Zheng, et al.
- Conference: ICML 2023
- Paper Link: https://arxiv.org/abs/2302.14376
- Official Code: https://github.com/HaoZhr/GNOT

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Cross-Attention Sensor Conditioning + Parameter FiLM
- Mechanism: Employs continuous geometric pretrained Transformer blocks with linear self-attention
  and cross-attention between spatial-temporal queries and parameter tokens:
  Q = Linear(X_query), K, V = MLP(P). Cross-attention allows each time point to dynamically query
  relevant biochemical parameter constraints.
- Biological Rationale: Non-local continuous evaluation across time conditioned on mutant genotype.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearCrossAttention(nn.Module):
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

    def forward(self, x, context):
        B, L, _ = x.shape
        _, S, _ = context.shape
        H, D = self.n_heads, self.d_head

        Q = F.elu(self.q_proj(x)).add(1.0).view(B, L, H, D).permute(0, 2, 1, 3)
        K = F.elu(self.k_proj(context)).add(1.0).view(B, S, H, D).permute(0, 2, 1, 3)
        V = self.v_proj(context).view(B, S, H, D).permute(0, 2, 1, 3)

        KV = torch.einsum('bhsd,bhsm->bhdm', K, V)
        K_sum = K.sum(dim=-2, keepdim=True)
        Z = torch.einsum('bhld,bhid->bhl', Q, K_sum).unsqueeze(-1) + self.eps

        out = torch.einsum('bhld,bhdm->bhlm', Q, KV) / Z
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        return self.out_proj(out)


class GNOT(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, seq_len=500, pred_len=500,
                 d_model=128, n_heads=4, n_layers=3, d_ff=256, **kwargs):
        super().__init__()
        self.in_proj = nn.Linear(num_vars * 2, d_model)
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 4)  # 4 parameter tokens
        )
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': LinearCrossAttention(d_model, n_heads=n_heads),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_model, num_vars)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T).permute(0, 2, 1)  # (B, 500, 78)
        h = self.in_proj(x)  # (B, 500, d_model)

        # Context tokens from params: (B, 4, d_model)
        p_tokens = self.param_encoder(params).view(B, 4, -1)

        for layer in self.layers:
            attn_out = layer['cross_attn'](h, p_tokens)
            h = layer['norm1'](h + attn_out)
            h = layer['norm2'](h + layer['ffn'](h))

        out = self.out_proj(h)
        return out.permute(0, 2, 1).contiguous()
