#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure 1D Wavelet Neural Operator (PureWNO_2022)
=============================================
- Paper Title: Wavelet Neural Operator: a neural operator for solving partial differential equations
- Authors: Tapas Tripura, Souvik Chakraborty
- Journal / Conference: Computer Methods in Applied Mechanics and Engineering (CMAME) 2023 / Math. Comput. Appl. 2022
- Paper Link: https://arxiv.org/abs/2203.01360
- Official Code: https://github.com/geckowang/WNO

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: None (Unconditioned Multi-resolution Wavelet Baseline)
- Mechanism: Employs discrete wavelet transforms (DWT/IDWT via ptwt/PyWavelets) across 4 decomposition levels,
  evaluating multi-scale temporal localized feature extraction capability without parameter injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt


class WaveletConv1d(nn.Module):
    """
    1D Continuous/Discrete Wavelet Convolution Layer (Tripura & Chakraborty, CMAME 2022).
    Decomposes the 1D input into multi-resolution wavelet coefficients (DWT) and applies
    learned scale-specific linear transformations before reconstructing (IDWT).
    """
    def __init__(self, in_channels, out_channels, level=3, wavelet='haar'):
        super(WaveletConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.wavelet = wavelet

        self.num_scales = level + 1
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(self.num_scales, out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        B, C_in, T = x.shape
        orig_dtype = x.dtype
        x_float = x.float()
        coeffs = ptwt.wavedec(x_float, self.wavelet, level=self.level, mode='zero')

        out_coeffs = []
        for i, c in enumerate(coeffs):
            w = self.weights[i].float()
            c_out = torch.einsum("bit, oi -> bot", c.float(), w)
            out_coeffs.append(c_out.float())

        x_out = ptwt.waverec(out_coeffs, self.wavelet)
        if x_out.shape[-1] > T:
            x_out = x_out[..., :T]
        elif x_out.shape[-1] < T:
            x_out = F.pad(x_out, (0, T - x_out.shape[-1]))

        return (x_out + self.bias.view(1, -1, 1)).to(orig_dtype)


class WNO1dBlock(nn.Module):
    """
    Canonical 1D WNO Layer: Wavelet Integral Operator + Local 1x1 Convolution Residual + GELU.
    """
    def __init__(self, width, level=3, wavelet='haar'):
        super(WNO1dBlock, self).__init__()
        self.conv = WaveletConv1d(width, width, level=level, wavelet=wavelet)
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        return F.gelu(x1 + x2)


class PureWNO_2022(nn.Module):
    """
    Pure 1D Wavelet Neural Operator (Tripura & Chakraborty 2022):
    - Input Lifting P: Maps input IC, time grid, and kinetic parameters to latent width.
    - Multi-layer WNO Blocks: 4 layers of multi-scale Wavelet Integral Transforms.
    - Output Projection Q: 2-layer MLP projecting latent representation back to physical trajectories.
    """
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4, level=3, wavelet='haar'):
        super(PureWNO_2022, self).__init__()
        self.num_vars = num_vars
        self.width = width
        self.p = nn.Linear(2 + param_dim, self.width)
        self.wno_blocks = nn.ModuleList([
            WNO1dBlock(self.width, level=level, wavelet=wavelet) for _ in range(n_blocks)
        ])
        self.q = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, ic_time_grid, params):
        """
        ic_time_grid: (B, V, 2, T) where [:, :, 0, :] is IC and [:, :, 1, :] is time grid
        params: (B, param_dim)
        """
        B, V, _, T = ic_time_grid.shape
        p_expand = params.view(B, 1, -1, 1).expand(-1, V, -1, T)
        x_in = torch.cat([ic_time_grid, p_expand], dim=2) # (B, V, 2 + param_dim, T)
        
        # Reshape each variable as an independent instance channel
        x = x_in.reshape(B * V, -1, T)
        x = x.permute(0, 2, 1)
        x = self.p(x) # (B*V, T, width)
        x = x.permute(0, 2, 1) # (B*V, width, T)

        for block in self.wno_blocks:
            x = block(x)

        x = x.permute(0, 2, 1)
        x = self.q(x) # (B*V, T, 1)
        x = x.permute(0, 2, 1) # (B*V, 1, T)
        
        return x.reshape(B, V, T)
