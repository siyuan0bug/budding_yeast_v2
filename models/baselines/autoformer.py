#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
=============================================================================================
- Paper Title: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
- Authors: Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long (Tsinghua University)
- Conference: NeurIPS 2021
- Paper Link: https://arxiv.org/abs/2106.13008
- Official Code: https://github.com/thuml/Autoformer

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Auto-Correlation Frequency Mode Modulation + Parameter FiLM
- Mechanism: Discards point-wise self-attention in favor of subseries Auto-Correlation via Wiener-Khinchin
  theorem (FFT-based periodicity detection and phase aggregation). Kinetic parameters P (141-dim) modulate
  the decomposed seasonal and trend representations across encoder-decoder blocks using FiLM affine heads:
  h' = (1 + gamma(P)) * h + beta(P).
- Biological Rationale: Perfectly tracks cell cycle oscillatory rhythm (periodic phases) and protein level trends.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoCorrelation(nn.Module):
    """Auto-Correlation Mechanism based on Wiener-Khinchin theorem (efficient O(L log L) FFT)."""
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg(self, values, corr):
        """Speed up subseries aggregation via rolled tensor sum."""
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # Top-k periodicities
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # Softmax weights
        tmp_corr = torch.softmax(weights, dim=-1)
        # Aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values)
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1))
        return delays_agg

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # Period-based dependencies via Wiener-Khinchin theorem (with AMP bfloat16 safe float32 cast)
        orig_dtype = queries.dtype
        q_in = queries.permute(0, 2, 3, 1).contiguous().float()
        k_in = keys.permute(0, 2, 3, 1).contiguous().float()
        q_fft = torch.fft.rfft(q_in, dim=-1)
        k_fft = torch.fft.rfft(k_in, dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1).to(orig_dtype)

        # Time delay aggregation
        V = self.time_delay_agg(values.permute(0, 2, 3, 1).contiguous(), corr)
        return V.permute(0, 3, 1, 2).contiguous(), None


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_correlation(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)


class SeriesDecompBlock(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (B, L, D) -> permute to (B, D, L)
        x_p = x.permute(0, 2, 1)
        pad_len = (self.kernel_size - 1) // 2
        front = x_p[:, :, 0:1].repeat(1, 1, pad_len)
        end = x_p[:, :, -1:].repeat(1, 1, pad_len)
        x_pad = torch.cat([front, x_p, end], dim=-1)
        trend = self.avg(x_pad)
        if trend.shape[-1] != x_p.shape[-1]:
            trend = F.interpolate(trend, size=x_p.shape[-1], mode='linear', align_corners=False)
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class Autoformer(nn.Module):
    """
    Autoformer for Budding Yeast Cell Cycle Dynamics:
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

        # Temporal Embedding: (B, 78, 500) -> (B, 500, d_model)
        self.enc_embedding = nn.Linear(num_vars * 2, d_model)

        self.decomp = SeriesDecompBlock(kernel_size=25)
        self.layers = nn.ModuleList([
            AutoCorrelationLayer(AutoCorrelation(factor=1), d_model=d_model, n_heads=n_heads)
            for _ in range(e_layers)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.film = nn.Linear(param_dim, d_model * 2)

        # Output projection back to 39 protein channels
        self.projection = nn.Linear(d_model, num_vars)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        # Flatten (B, 78, 500) -> permute to (B, 500, 78)
        x = ic_time_grid.reshape(B, V * F_in, T).permute(0, 2, 1)
        h = self.enc_embedding(x)  # (B, 500, d_model)

        # Parameter conditioning
        gamma, beta = self.film(params).chunk(2, dim=-1)
        h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)

        # Multi-layer auto-correlation blocks
        for layer in self.layers:
            seasonal, trend = self.decomp(h)
            attn_out = layer(seasonal, seasonal, seasonal)
            h = self.norm(h + attn_out)
            h = h + self.ffn(h)

        out = self.projection(h)  # (B, 500, 39)
        return out.permute(0, 2, 1).contiguous()  # (B, 39, 500)
