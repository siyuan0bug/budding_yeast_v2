#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
=============================================================================
- Paper Title: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
- Authors: Hao Liu, Chengqing Xu, Jiehui Xu, Jianmin Wang, Mingsheng Long (Tsinghua University)
- Conference: ICLR 2024 (Spotlight)
- Paper Link: https://arxiv.org/abs/2310.06625
- Official Code: https://github.com/thuml/iTransformer

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Biochemical Parameter Prefix-Token & Variate Cross-Modulation
- Mechanism: Inverts typical temporal tokenization: each protein channel's full temporal series (500 steps)
  is embedded into a d_model token. Kinetic parameters P (141-dim) are projected via MLP into a specialized
  Biochemical Control Token [P_token], prepended to the 39 protein variate tokens:
  Tokens = [P_token, V_1, V_2, ..., V_39] in R^(B x 40 x d_model).
  Multi-head self-attention computes global interaction across all protein channels conditioned on P.
  After Transformer blocks, a temporal projector maps each protein token back to 500 continuous time steps.
- Biological Rationale: Perfectly models inter-protein regulatory networks as attention graph, driven by global genotype P.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class iTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        # Self-attention across inverted variate tokens
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.linear2(self.act(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_out))
        return x


class iTransformer(nn.Module):
    """
    iTransformer for Budding Yeast Cell Cycle Dynamics:
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

        # Projection from 2 features (IC + time grid) x seq_len into d_model token
        self.enc_embedding = nn.Linear(seq_len * 2, d_model)

        # Parameter conditioning prefix token generator
        self.param_token_gen = nn.Sequential(
            nn.Linear(param_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

        # Inverted Transformer Encoder
        self.layers = nn.ModuleList([
            iTransformerEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Temporal projector: d_model -> pred_len (500 steps)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, pred_len)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape

        # Inverted tokenization: flatten (2 x 500) for each of the 39 variables
        # (B, 39, 2, 500) -> (B, 39, 1000)
        x_flat = ic_time_grid.reshape(B, V, F_in * T)
        var_tokens = self.enc_embedding(x_flat)  # (B, 39, d_model)

        # Generate parameter conditioning prefix token
        p_token = self.param_token_gen(params).unsqueeze(1)  # (B, 1, d_model)

        # Prepend parameter token: total 40 tokens (B, 40, d_model)
        tokens = torch.cat([p_token, var_tokens], dim=1)

        # Pass through Inverted Transformer layers
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        # Extract only the 39 protein tokens (discard prefix token)
        protein_tokens = tokens[:, 1:, :]  # (B, 39, d_model)

        # Temporal projection: (B, 39, d_model) -> (B, 39, 500)
        out = self.projector(protein_tokens)
        return out
