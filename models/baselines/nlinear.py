#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLinear: Normalized-Linear Model for Time Series
================================================
- Paper Title: Are Transformers Effective for Time Series?
- Authors: Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu
- Conference: AAAI 2023
- Paper Link: https://arxiv.org/abs/2205.13504
- Official Code: https://github.com/cure-lab/LTSF-Linear

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Initial Value Shift Normalization + Parameter Modulation
- Mechanism: Normalizes the input sequence by subtracting the initial/last known baseline state (y0),
  applies a direct linear projection across time horizon, applies FiLM parameter modulation gamma(P), beta(P),
  and adds back the baseline state: y = y0 + (1 + gamma(P)) * W(x - y0) + beta(P).
- Biological Rationale: Perfectly addresses initial concentration level shift across yeast mutant batches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NLinear(nn.Module):
    """
    NLinear for Budding Yeast Cell Cycle Dynamics:
    - Input: ic_time_grid (B, 39, 2, 500), params (B, 141)
    - Output: (B, 39, 500)
    """
    def __init__(self, num_vars=39, param_dim=141, seq_len=500, pred_len=500, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.in_proj = nn.Conv1d(num_vars * 2, num_vars, 1)
        self.linear = nn.Linear(seq_len, pred_len)
        self.film = nn.Linear(param_dim, num_vars * 2)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)
        x = self.in_proj(x)  # (B, 39, 500)

        # Baseline shift normalization (subtract initial value)
        y0 = x[:, :, 0:1]  # (B, 39, 1)
        x_norm = x - y0

        # Linear projection
        out = self.linear(x_norm)  # (B, 39, 500)

        # Parameter conditioning
        gamma, beta = self.film(params).chunk(2, dim=-1)
        out = (1.0 + gamma.unsqueeze(-1)) * out + beta.unsqueeze(-1)

        # Add back initial condition baseline
        out = out + y0
        return out
