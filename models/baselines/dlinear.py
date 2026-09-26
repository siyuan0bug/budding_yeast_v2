#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLinear: Decomposition-Linear Model for Time Series
===================================================
- Paper Title: Are Transformers Effective for Time Series?
- Authors: Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu
- Conference: AAAI 2023
- Paper Link: https://arxiv.org/abs/2205.13504
- Official Code: https://github.com/cure-lab/LTSF-Linear

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Parameter-Modulated Series Decomposition
- Mechanism: Uses a moving average kernel to decompose protein concentration trajectories into
  trend-cyclical (macro-growth/arrest) and seasonal/residual components. Biochemical kinetic
  parameters P (141-dim) modulate the linear mapping weights via FiLM affine transformation:
  y_trend = (1 + gamma_t(P)) * W_t(x_trend) + beta_t(P)
  y_season = (1 + gamma_s(P)) * W_s(x_season) + beta_s(P)
  y = y_trend + y_season
- Biological Rationale: Separates monotonic accumulation/depletion trends from cell cycle periodic oscillations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvg(nn.Module):
    """Moving average block to extract trends from temporal trajectories."""
    def __init__(self, kernel_size=25, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (B, C, T)
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, :, 0:1].repeat(1, 1, pad_len)
        end = x[:, :, -1:].repeat(1, 1, pad_len)
        x_pad = torch.cat([front, x, end], dim=-1)
        out = self.avg(x_pad)
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-1], mode='linear', align_corners=False)
        return out


class SeriesDecomp(nn.Module):
    """Decomposes time series into residual/seasonal and trend components."""
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size=kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear for Budding Yeast Cell Cycle Dynamics:
    - Input: ic_time_grid (B, 39, 2, 500), params (B, 141)
    - Output: (B, 39, 500)
    """
    def __init__(self, num_vars=39, param_dim=141, seq_len=500, pred_len=500, kernel_size=25, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.decompsition = SeriesDecomp(kernel_size)

        # Input projection from (IC + time_grid) to internal temporal representation
        self.in_proj = nn.Conv1d(num_vars * 2, num_vars, 1)

        # Separate linear projections for seasonal and trend components
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

        # FiLM parameter conditioning
        self.film_seasonal = nn.Linear(param_dim, num_vars * 2)
        self.film_trend = nn.Linear(param_dim, num_vars * 2)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)
        x = self.in_proj(x)  # (B, 39, 500)

        # Series decomposition
        seasonal, trend = self.decompsition(x)  # (B, 39, 500), (B, 39, 500)

        # Direct linear mapping
        seasonal_out = self.linear_seasonal(seasonal)  # (B, 39, 500)
        trend_out = self.linear_trend(trend)          # (B, 39, 500)

        # Parameter conditioning
        gamma_s, beta_s = self.film_seasonal(params).chunk(2, dim=-1)
        gamma_t, beta_t = self.film_trend(params).chunk(2, dim=-1)

        seasonal_out = (1.0 + gamma_s.unsqueeze(-1)) * seasonal_out + beta_s.unsqueeze(-1)
        trend_out = (1.0 + gamma_t.unsqueeze(-1)) * trend_out + beta_t.unsqueeze(-1)

        out = seasonal_out + trend_out
        return out
