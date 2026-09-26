#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
================================================================================
- Paper Title: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
- Authors: Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang
- Conference: AAAI 2021 (Best Paper Award)
- Paper Link: https://arxiv.org/abs/2012.07436
- Official Code: https://github.com/zhouhaoyi/Informer2020

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: ProbSparse Attention Modulation + Parameter FiLM
- Mechanism: Employs ProbSparse self-attention to select dominant active queries via KL-divergence measurement,
  reducing computational complexity to O(L log L). Features are modulated by FiLM affine parameters
  gamma(P), beta(P) = Linear(P).chunk(2) after sparse attention distillation:
  h' = (1 + gamma(P)) * h + beta(P).
  Outputs entire 500-step trajectory via direct generative non-autoregressive projection.
- Biological Rationale: Focuses attention on critical cell cycle phase transition moments (START checkpoint,
  metaphase-anaphase transition) while ignoring flat inactive phases.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbAttention(nn.Module):
    """ProbSparse Self-Attention mechanism (AAAI 2021 Best Paper)."""
    def __init__(self, factor=5, scale=None, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q: (B, H, L_Q, E), K: (B, H, L_K, E)
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Sample K randomly to calculate query sparsity measurement
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Sparsity measurement: M = max(Q_K) - sum(Q_K)/sample_k
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), sample_k)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use index to get top queries
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # (B, H, n_top, E)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def forward(self, queries, keys, values):
        B, L_Q, H, E = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        u_q = min(int(self.factor * math.log(L_Q)), L_Q)
        u_k = min(int(self.factor * math.log(L_K)), L_K)
        u_q = max(u_q, 1)
        u_k = max(u_k, 1)

        scores_top, index = self._prob_QK(queries, keys, sample_k=u_k, n_top=u_q)
        scale = self.scale or 1.0 / math.sqrt(E)
        scores_top = scores_top * scale

        # Context matrix
        attn = torch.softmax(scores_top, dim=-1)
        context_top = torch.matmul(self.dropout(attn), values)

        # Fill in mean for inactive queries
        context = values.mean(dim=-2, keepdim=True).expand(B, H, L_Q, E).clone()
        context[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = context_top

        return context.permute(0, 2, 1, 3).contiguous(), None


class ConvLayer(nn.Module):
    """Self-attention distilling layer: 1D convolution + ELU + MaxPool."""
    def __init__(self, c_in):
        super().__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: (B, L, D) -> (B, D, L)
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.permute(0, 2, 1)


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = ProbAttention(factor=5)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        d_k = D // H
        Q = self.q_proj(x).view(B, L, H, d_k)
        K = self.k_proj(x).view(B, L, H, d_k)
        V = self.v_proj(x).view(B, L, H, d_k)

        attn_out, _ = self.attn(Q, K, V)
        attn_out = self.out_proj(attn_out.view(B, L, D))
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class Informer(nn.Module):
    """
    Informer for Budding Yeast Cell Cycle Dynamics:
    - Input: ic_time_grid (B, 39, 2, 500), params (B, 141)
    - Output: (B, 39, 500)
    """
    def __init__(self, num_vars=39, param_dim=141, seq_len=500, pred_len=500,
                 d_model=128, n_heads=4, e_layers=2, d_ff=256, dropout=0.1, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.enc_embedding = nn.Linear(num_vars * 2, d_model)
        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(e_layers)
        ])
        self.film = nn.Linear(param_dim, d_model * 2)
        self.projection = nn.Linear(d_model, num_vars)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T).permute(0, 2, 1)  # (B, 500, 78)
        h = self.enc_embedding(x)  # (B, 500, d_model)

        gamma, beta = self.film(params).chunk(2, dim=-1)
        h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)

        for layer in self.layers:
            h = layer(h)

        out = self.projection(h)  # (B, 500, 39)
        return out.permute(0, 2, 1).contiguous()  # (B, 39, 500)
