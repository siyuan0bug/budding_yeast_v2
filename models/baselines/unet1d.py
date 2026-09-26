#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D U-Net Neural Operator (UNet1d)
=================================
Adapted from RealPDEBench and PDEBench (Takamoto et al., 2022).
Multi-scale 1D Encoder-Decoder with:
- Multi-resolution hierarchical downsampling and upsampling.
- Residual skip connections preserving both high-frequency phase shifts and low-frequency macro trends.
- Parameter FiLM (Feature-wise Linear Modulation) conditioning at each scale.
- Continuous 1D temporal interpolative upsampling for resolution-invariant dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, param_dim=141):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.film = nn.Linear(param_dim, out_channels * 2)

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, p):
        gamma, beta = self.film(p).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        res = self.skip(x)
        h = F.gelu(self.norm1(self.conv1(x)))
        h = (1.0 + gamma) * h + beta
        h = self.norm2(self.conv2(h))
        return F.gelu(h + res)


class UNet1d(nn.Module):
    """
    1D U-Net Neural Operator for Yeast Dynamics:
    - Input: ic_time_grid (B, 39, 2, 500) and params (B, 141)
    - Output: (B, 39, 500)
    """
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=2):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        in_channels = num_vars * 2  # 78

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU()
        )

        # Level 1 (T=500)
        self.enc1 = nn.ModuleList([
            FiLMResBlock1d(width, width, dilation=1, param_dim=param_dim)
            for _ in range(n_blocks)
        ])
        self.down1 = nn.Conv1d(width, width * 2, kernel_size=4, stride=2, padding=1)  # -> T=250

        # Level 2 (T=250)
        self.enc2 = nn.ModuleList([
            FiLMResBlock1d(width * 2, width * 2, dilation=2, param_dim=param_dim)
            for _ in range(n_blocks)
        ])
        self.down2 = nn.Conv1d(width * 2, width * 4, kernel_size=4, stride=2, padding=1)  # -> T=125

        # Bottleneck (T=125)
        self.bottleneck = nn.ModuleList([
            FiLMResBlock1d(width * 4, width * 4, dilation=4, param_dim=param_dim)
            for _ in range(n_blocks)
        ])

        # Decoder 2 (T=125 -> T=250)
        self.up2_conv = nn.Conv1d(width * 4 + width * 2, width * 2, kernel_size=3, padding=1)
        self.dec2 = nn.ModuleList([
            FiLMResBlock1d(width * 2, width * 2, dilation=2, param_dim=param_dim)
            for _ in range(n_blocks)
        ])

        # Decoder 1 (T=250 -> T=500)
        self.up1_conv = nn.Conv1d(width * 2 + width, width, kernel_size=3, padding=1)
        self.dec1 = nn.ModuleList([
            FiLMResBlock1d(width, width, dilation=1, param_dim=param_dim)
            for _ in range(n_blocks)
        ])

        # Head
        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)

        # Stem
        h0 = self.stem(x)  # (B, width, 500)

        # Enc 1
        h1 = h0
        for block in self.enc1:
            h1 = block(h1, params)
        d1 = self.down1(h1)  # (B, width*2, 250)

        # Enc 2
        h2 = d1
        for block in self.enc2:
            h2 = block(h2, params)
        d2 = self.down2(h2)  # (B, width*4, 125)

        # Bottleneck
        b = d2
        for block in self.bottleneck:
            b = block(b, params)

        # Dec 2
        u2 = F.interpolate(b, size=h2.shape[-1], mode='linear', align_corners=False)
        u2 = self.up2_conv(torch.cat([u2, h2], dim=1))
        for block in self.dec2:
            u2 = block(u2, params)

        # Dec 1
        u1 = F.interpolate(u2, size=h1.shape[-1], mode='linear', align_corners=False)
        u1 = self.up1_conv(torch.cat([u1, h1], dim=1))
        for block in self.dec1:
            u1 = block(u1, params)

        out = self.head(u1)  # (B, 39, 500)
        return out
