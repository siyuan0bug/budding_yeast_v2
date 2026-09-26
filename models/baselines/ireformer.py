#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iReformer: Inverted Reformer for Time Series
=======================================
- Paper Title: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
- Authors: Hao Liu, Chengqing Xu, Jiehui Xu, Jianmin Wang, Mingsheng Long (Tsinghua University)
- Conference: ICLR 2024 (Spotlight)
- Paper Link: https://arxiv.org/abs/2310.06625
- Official Code: https://github.com/thuml/iTransformer

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Inverted Variate Tokenization + Biochemical Parameter Prefix-Token
- Mechanism: Inverts the channel-temporal axes: each protein is treated as an individual token,
  embedding its 500-step trajectory into d_model. Kinetic parameters P (141-dim) form an active
  Biochemical Prefix-Token prepended to the sequence of 39 proteins. Multi-head Reformer attention
  operates across the 40 tokens. A linear projector then maps each protein back to 500 time steps.
- Biological Rationale: Focuses attention mechanism directly on the 39x39 protein-protein interaction topology.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class iReformer(nn.Module):
    """
    iReformer for Budding Yeast Cell Cycle Dynamics:
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
        self.d_model = d_model

        self.enc_embedding = nn.Linear(seq_len * 2, d_model)
        self.param_token_gen = nn.Sequential(
            nn.Linear(param_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, pred_len)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x_flat = ic_time_grid.reshape(B, V, F_in * T)
        var_tokens = self.enc_embedding(x_flat)  # (B, 39, d_model)

        p_token = self.param_token_gen(params).unsqueeze(1)  # (B, 1, d_model)
        tokens = torch.cat([p_token, var_tokens], dim=1)  # (B, 40, d_model)

        tokens = self.norm(self.encoder(tokens))
        protein_tokens = tokens[:, 1:, :]  # (B, 39, d_model)

        out = self.projector(protein_tokens)  # (B, 39, 500)
        return out
