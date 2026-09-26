#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Koopman Operator / Dynamic Mode Decomposition (DeepKoopmanOperator)
========================================================================
- Paper Title: RealPDEBench: A Benchmark for Complex Physical Systems with Pair Data / Deep Koopman
- Conference: NeurIPS 2024 / ICLR 2024
- Paper Link: https://arxiv.org/abs/2410.05264
- Official Code: https://github.com/BaratiLab/RealPDEBench

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Parameter-Conditioned Linear Spectral Koopman Matrix K(P)
- Mechanism: Maps nonlinear state Y into an infinite-dimensional latent Hilbert space via lifting encoder:
  z_0 = Phi(Y_0, P). In latent space, nonlinear dynamics follow linear evolution:
  z_{t+1} = K(P) z_t, where K(P) is parameterized by kinetic parameters P.
  Reconstructed trajectory: Y(t) = Phi^{-1}(z_t).
- Biological Rationale: Koopman operator linearization decomposes complex nonlinear ODE limit cycles into linear eigenmodes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepKoopmanOperator(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, seq_len=500, pred_len=500,
                 latent_dim=64, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.latent_dim = latent_dim

        # Lifting network: (IC, params) -> latent state z0
        self.lifting_net = nn.Sequential(
            nn.Linear(num_vars + param_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )
        # Parameter-dependent transition matrix generator
        self.koopman_gen = nn.Sequential(
            nn.Linear(param_dim, latent_dim * latent_dim)
        )
        # Reconstruction network: latent z(t) -> Y(t)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_vars)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        y0 = ic_time_grid[:, :, 0, 0]  # (B, 39)
        inp = torch.cat([y0, params], dim=-1)
        z0 = self.lifting_net(inp)  # (B, latent_dim)

        # Generate Koopman operator K(P): (B, latent_dim, latent_dim)
        # Scaled near identity to ensure stable discrete propagation
        K = self.koopman_gen(params).view(B, self.latent_dim, self.latent_dim)
        K = torch.eye(self.latent_dim, device=K.device).unsqueeze(0) + 0.01 * torch.tanh(K)

        # Rollout linear Koopman dynamics across 500 steps
        zs = [z0]
        z_curr = z0
        for _ in range(T - 1):
            z_curr = torch.bmm(K, z_curr.unsqueeze(-1)).squeeze(-1)
            zs.append(z_curr)
        z_traj = torch.stack(zs, dim=1)  # (B, 500, latent_dim)

        out = self.decoder(z_traj)  # (B, 500, 39)
        return out.permute(0, 2, 1).contiguous()
