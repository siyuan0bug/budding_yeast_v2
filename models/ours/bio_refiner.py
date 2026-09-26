#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioRefiner (Two-Stage Coarse-to-Fine Neural Operator Refiner)
============================================================
A novel two-stage coarse-to-fine operator architecture for stiff biochemical ODE systems:
- Stage 1: Coarse Macro Operator (BioMechPathway backbone) predicting macro trajectory topology Y^(0).
- Stage 2: Stiff Phase Residual Refiner taking [Y^(0), dY^(0)/dt, X, p] to predict high-frequency phase corrections Delta Y.
- Final Output: Y^(final) = Y^(0) + tanh(gamma) * Delta Y with deep supervision support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bio_mechanistic_operator import BioMechPathwayOnly


class FiLMBlock1d(nn.Module):
    """Dilated 1D Residual Block with Parameter FiLM Modulation."""
    def __init__(self, channels, dilation=1, param_dim=141):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, channels)
        self.film = nn.Linear(param_dim, channels * 2)

    def forward(self, x, p):
        # p: (B, param_dim) -> (B, 2*channels)
        gamma, beta = self.film(p).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        res = x
        out = F.gelu(self.norm1(self.conv1(x)))
        out = (1.0 + gamma) * out + beta
        out = self.norm2(self.conv2(out))
        return F.gelu(out + res)


class StiffPhaseRefiner(nn.Module):
    """
    Stage 2: Stiff Phase-Field Residual Refinement Network.
    Takes concatenated [Y^(0), dY^(0)/dt, X_in] (156 channels) and parameter vector p
    to generate stiff phase-transition corrections Delta Y (39 channels).
    """
    def __init__(self, in_channels=156, num_vars=39, width=128, param_dim=141, n_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU()
        )

        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList([
            FiLMBlock1d(width, dilation=dilations[i % len(dilations)], param_dim=param_dim)
            for i in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

    def forward(self, z, p):
        # z: (B, 156, T), p: (B, 141)
        h = self.stem(z)
        for block in self.blocks:
            h = block(h, p)
        delta_y = self.head(h)
        return delta_y


class BioRefiner(nn.Module):
    """
    BioRefiner: Two-Stage Coarse-to-Fine Neural Operator.
    """
    def __init__(self, num_vars=39, width=128, modes=32, param_dim=141, n_blocks=4,
                 coarse_n_blocks=4, deep_supervision=True):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.deep_supervision = deep_supervision

        # Stage 1: Coarse BioMech Pathway Base Operator
        self.stage1_coarse = BioMechPathwayOnly(
            num_vars=num_vars,
            width=width,
            modes=modes,
            param_dim=param_dim,
            n_blocks=coarse_n_blocks
        )

        # Stage 2: Stiff Phase Residual Refiner
        # In channels = 39 (Y0) + 39 (dY0/dt) + 78 (X_in) = 156
        self.stage2_refiner = StiffPhaseRefiner(
            in_channels=num_vars * 2 + num_vars * 2,  # 156
            num_vars=num_vars,
            width=width,
            param_dim=param_dim,
            n_blocks=n_blocks
        )

        # Learnable gating scale (initialized to 0.1 for gentle progressive intervention)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def _compute_derivative(self, y):
        # y: (B, 39, T)
        # Compute central difference with boundary padding
        d = torch.zeros_like(y)
        d[:, :, 1:-1] = (y[:, :, 2:] - y[:, :, :-2]) * 0.5
        d[:, :, 0] = y[:, :, 1] - y[:, :, 0]
        d[:, :, -1] = y[:, :, -1] - y[:, :, -2]
        return d

    def forward(self, x, p, return_coarse=False):
        """
        x: (B, 39, 2, T) or (B, 39, T)
        p: (B, 141)
        """
        # Format input tensor X_in
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T)  # (B, 78, T)
        else:
            B, V, T = x.shape
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, time_grid], dim=1)  # (B, 78, T)

        # 1. Stage 1 Coarse Trajectory Prediction
        y_coarse = self.stage1_coarse(x, p)  # (B, 39, T)

        # 2. Extract 1st-order temporal derivative features
        dy_coarse = self._compute_derivative(y_coarse)  # (B, 39, T)

        # 3. Concatenate rich phase-field representation
        z_refine = torch.cat([y_coarse, dy_coarse, x_in], dim=1)  # (B, 156, T)

        # 4. Stage 2 Stiff Residual Correction
        delta_y = self.stage2_refiner(z_refine, p)  # (B, 39, T)

        # 5. Gated Residual Composition
        y_final = y_coarse + torch.tanh(self.gamma) * delta_y

        if return_coarse:
            return y_final, y_coarse
        return y_final
