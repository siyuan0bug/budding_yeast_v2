#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vanilla Transformer for Time Series (VanillaTransformer)
=========================================================
- Paper Title: Attention Is All You Need
- Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- Conference: NeurIPS 2017
- Paper Link: https://arxiv.org/abs/1706.03762
- Official Code: https://github.com/tensorflow/tensor2tensor / https://github.com/thuml/iTransformer

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Parameter FiLM Modulated Full Self-Attention
- Mechanism: Canonical dense multi-head self-attention with scaled dot-product attention:
  Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.
  Biochemical kinetic parameters P (141-dim) modulate temporal token representations via FiLM affine
  transformations gamma(P), beta(P) = Linear(P).chunk(2): h' = (1 + gamma(P)) * h + beta(P).
- Biological Rationale: Canonical dense attention benchmark to evaluate global temporal context modeling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaTransformer(nn.Module):
    """
    Standard Canonical Transformer for Budding Yeast Cell Cycle Dynamics:
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.film = nn.Linear(param_dim, d_model * 2)
        self.out_proj = nn.Linear(d_model, num_vars)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T).permute(0, 2, 1)  # (B, 500, 78)
        h = self.in_proj(x)  # (B, 500, d_model)

        gamma, beta = self.film(params).chunk(2, dim=-1)
        h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)

        h = self.transformer_encoder(h)  # (B, 500, d_model)
        out = self.out_proj(h)  # (B, 500, 39)
        return out.permute(0, 2, 1).contiguous()  # (B, 39, 500)
