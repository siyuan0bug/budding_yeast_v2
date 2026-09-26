#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D ResNet with FiLM Conditioning (ResNetFiLM)
=============================================
- Paper Title: FiLM: Visual Reasoning with a General Conditioning Layer / RealPDEBench
- Authors: Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville (AAAI 2018) / RealPDEBench (2024)
- Conference: AAAI 2018 / NeurIPS 2024
- Paper Link: https://arxiv.org/abs/1709.07871
- Official Code: https://github.com/BaratiLab/RealPDEBench

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Feature-wise Linear Modulation (FiLM)
- Mechanism: Direct 1D dilated residual blocks modulated by MLP(P) affine scaling gamma and bias beta.
- Biological Rationale: Direct modulation of residual biochemical fluxes with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM):
    Predicts (gamma, beta) from the parameter latent vector and applies affine transformation:
    x_mod = Norm(x) * (1.0 + tanh(gamma)) + beta
    """
    def __init__(self, d_lambda, width):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_lambda, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width * 2)
        )
        # Initialize final linear layer close to 0 so modulation starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, lambda_prime):
        """
        x: (B, width, T)
        lambda_prime: (B, d_lambda)
        """
        mod = self.mlp(lambda_prime) # (B, width * 2)
        gamma, beta = torch.chunk(mod, 2, dim=-1) # (B, width), (B, width)
        gamma = gamma.unsqueeze(-1) # (B, width, 1)
        beta = beta.unsqueeze(-1)   # (B, width, 1)

        return x * (1.0 + torch.tanh(gamma)) + beta


class ChannelSelfAttention(nn.Module):
    """
    Channel-wise attention across 39 protein variables at each time step.
    """
    def __init__(self, num_vars=39, width=64, d_attn=16, d_lambda=128):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.d_attn = d_attn

        self.to_vars = nn.Linear(width, num_vars * d_attn)
        self.from_vars = nn.Linear(num_vars * d_attn, width)

        self.fc_q = nn.Linear(d_attn, d_attn)
        self.fc_k = nn.Linear(d_attn, d_attn)
        self.fc_v = nn.Linear(d_attn, d_attn)

        self.ppi_bias = nn.Sequential(
            nn.Linear(d_lambda, 64),
            nn.GELU(),
            nn.Linear(64, num_vars * num_vars)
        )

    def forward(self, x, lambda_prime):
        B, W, T = x.shape
        x_T = x.permute(0, 2, 1) # (B, T, W)
        x_vars = self.to_vars(x_T).reshape(B, T, self.num_vars, self.d_attn)

        Q = self.fc_q(x_vars)
        K = self.fc_k(x_vars)
        V_val = self.fc_v(x_vars)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_attn ** 0.5)
        bias = self.ppi_bias(lambda_prime).reshape(B, 1, self.num_vars, self.num_vars)
        scores = scores + bias

        attn = F.softmax(scores, dim=-1)
        out_vars = torch.matmul(attn, V_val)

        out_flat = out_vars.reshape(B, T, self.num_vars * self.d_attn)
        out = self.from_vars(out_flat).permute(0, 2, 1) # (B, W, T)
        return out


class ResNetFiLMBlock(nn.Module):
    """
    1D Dilated Residual Block with FiLM Parameter Conditioning and Channel Self-Attention.
    """
    def __init__(self, width, dilation, d_lambda, num_vars=39, d_attn=16):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, width)
        self.film1 = FiLMLayer(d_lambda, width)
        self.conv1 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)

        self.norm2 = nn.GroupNorm(1, width)
        self.film2 = FiLMLayer(d_lambda, width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)

        self.attn = ChannelSelfAttention(num_vars=num_vars, width=width, d_attn=d_attn, d_lambda=d_lambda)

    def forward(self, x, lambda_prime):
        res = x

        h = self.norm1(x)
        h = self.film1(h, lambda_prime)
        h = F.gelu(self.conv1(h))

        h = self.norm2(h)
        h = self.film2(h, lambda_prime)
        h = self.conv2(h)

        h_attn = self.attn(h, lambda_prime)

        return F.gelu(res + h + h_attn)


class ResNetFiLM(nn.Module):
    """
    1D ResNet with FiLM Parameter Conditioning (One-Shot Time-Domain Neural Operator).
    """
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4, d_lambda=128, d_attn=16):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.n_blocks = n_blocks

        # Parameter conditioning encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_lambda)
        )

        in_channels = num_vars * 2
        self.stem = nn.Conv1d(in_channels, width, kernel_size=5, padding=2)

        # Multi-scale hierarchical dilations: 1, 2, 4, 8, ...
        dilations = [2 ** (i % 4) for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            ResNetFiLMBlock(width=width, dilation=dilations[i], d_lambda=d_lambda,
                            num_vars=num_vars, d_attn=d_attn)
            for i in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

    def forward(self, ic_time_grid, params):
        """
        ic_time_grid: (B, num_vars, 2, T)
        params: (B, param_dim)
        """
        B, V, F_in, T = ic_time_grid.shape

        lambda_prime = self.param_encoder(params)

        x = ic_time_grid.reshape(B, V * F_in, T)
        h = self.stem(x)

        for block in self.blocks:
            h = block(h, lambda_prime)

        out = self.head(h) # (B, num_vars, T)
        return out
