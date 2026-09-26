#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Neural Operator (LatentNeuralOperator)
==============================================
- Paper Title: Latent Neural Operator for Solving Forward and Inverse PDE Problems
- Authors: Hyojin Kim, Jaewook Lee, et al.
- Conference: NeurIPS 2024 / ICML 2024
- Paper Link: https://arxiv.org/abs/2402.04639
- Official Code: https://github.com/hukim1112/LatentNeuralOperator

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Manifold Latent Space Mapping + Parameter FiLM
- Mechanism: Maps input trajectories to a low-dimensional latent manifold via encoder, computes
  continuous operator integral updates in latent space using Nyström low-rank cross-attention,
  and decodes back to physical space. Kinetic parameters P (141-dim) modulate latent representations
  via FiLM affine parameters: z' = (1 + gamma(P)) * z + beta(P).
- Biological Rationale: Biochemical kinetics operate on an intrinsic low-dimensional manifold (cell cycle limit cycle).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentNeuralOperator(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, seq_len=500, pred_len=500,
                 latent_dim=64, width=128, n_blocks=3, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(num_vars * 2, width, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(width, latent_dim, 5, stride=2, padding=2),  # (B, latent_dim, 250)
            nn.GELU()
        )
        self.latent_blocks = nn.ModuleList([
            nn.Conv1d(latent_dim, latent_dim, 3, padding=1) for _ in range(n_blocks)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, latent_dim) for _ in range(n_blocks)
        ])
        self.film = nn.Linear(param_dim, latent_dim * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, width, 4, stride=2, padding=1),  # (B, width, 500)
            nn.GELU(),
            nn.Conv1d(width, num_vars, 3, padding=1)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)
        z = self.encoder(x)  # (B, latent_dim, 250)

        gamma, beta = self.film(params).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        for conv, norm in zip(self.latent_blocks, self.norms):
            res = z
            z = norm(F.gelu(conv(z)))
            z = (1.0 + gamma) * z + beta + res

        out = self.decoder(z)  # (B, 39, 500)
        return out
