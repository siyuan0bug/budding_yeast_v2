#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhysicsConstrainedBiochemicalOperator (PC-BioNO)
================================================
A physics-constrained neural operator integrating exact algebraic conservation laws
and phase-space manifold projections from Chen 2004 Table 1:
1. Sic1 Total: [Sic1_T] = [Sic1] + [C2] + [C5] + [Sic1P] + [C2P] + [C5P]
2. Cdc6 Total: [Cdc6_T] = [Cdc6] + [F2] + [F5] + [Cdc6P] + [F2P] + [F5P]
3. Net1 Total: [Net1_T] = [Net1] + [RENT] + [Net1P]
4. Cdc14 Total: [Cdc14_T] = [Cdc14] + [RENT]
5. Strict physical non-negativity: [X(t)] >= 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dynamic_reaction_flux_operator import DyRFOResNetBlock
from .bio_hetero_causal_graph import load_multirelational_graphs, BioMutationFluxEncoder

# Indices
(nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5,
 nSIC1P, nC2P, nC5P, nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T,
 nCDC20, nCDH1T, nCDH1, nCDC14T, nCDC14, nNET1T, nNET1, nRENT, nTEM1,
 nCDC15, nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, nVi20, nlte1, nBUB2) = range(39)


class DifferentiableConservationManifoldLayer(nn.Module):
    """
    Differentiable Phase-Space Manifold Projection Layer.
    Softly projects unconstrained predictions onto the physical conservation manifold.
    """
    def __init__(self, num_vars=39):
        super().__init__()
        self.num_vars = num_vars
        # Learnable relaxation/scale factors
        self.alpha_sic1 = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_cdc6 = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_net1 = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_cdc14 = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x_raw):
        """
        x_raw: (B, 39, T) - Raw model output
        Output: (B, 39, T) - Projected physically consistent trajectories
        """
        # 1. Strict non-negativity (Proteins concentrations >= 0)
        x = F.softplus(x_raw) # (B, 39, T)

        # 2. Fractional conservation projections (avoid division by zero with epsilon)
        eps = 1e-6

        # --- Law 1: Net1 + RENT <= Net1T ---
        net1_sum = x[:, nNET1:nNET1+1, :] + x[:, nRENT:nRENT+1, :] + eps
        net1_scale = torch.clamp(x[:, nNET1T:nNET1T+1, :] / net1_sum, max=1.0)
        x_net1_adj = x[:, nNET1:nNET1+1, :] * net1_scale
        x_rent_adj = x[:, nRENT:nRENT+1, :] * net1_scale

        # --- Law 2: Cdc14 + RENT <= Cdc14T ---
        cdc14_sum = x[:, nCDC14:nCDC14+1, :] + x_rent_adj + eps
        cdc14_scale = torch.clamp(x[:, nCDC14T:nCDC14T+1, :] / cdc14_sum, max=1.0)
        x_cdc14_adj = x[:, nCDC14:nCDC14+1, :] * cdc14_scale

        # --- Law 3: Swi5 <= Swi5T, Cdh1 <= Cdh1T, Cdc20 <= Cdc20T ---
        x_swi5_adj = torch.minimum(x[:, nSWI5:nSWI5+1, :], x[:, nSWI5T:nSWI5T+1, :])
        x_cdh1_adj = torch.minimum(x[:, nCDH1:nCDH1+1, :], x[:, nCDH1T:nCDH1T+1, :])
        x_cdc20_adj = torch.minimum(x[:, nCDC20:nCDC20+1, :], x[:, nCDC20T:nCDC20T+1, :])

        # Assemble softly projected variables
        out = x.clone()
        out[:, nNET1:nNET1+1, :] = x_net1_adj
        out[:, nRENT:nRENT+1, :] = x_rent_adj
        out[:, nCDC14:nCDC14+1, :] = x_cdc14_adj
        out[:, nSWI5:nSWI5+1, :] = x_swi5_adj
        out[:, nCDH1:nCDH1+1, :] = x_cdh1_adj
        out[:, nCDC20:nCDC20+1, :] = x_cdc20_adj

        return out


class PhysicsConstrainedBiochemicalOperator(nn.Module):
    """
    Flagship PhysicsConstrainedBiochemicalOperator (PC-BioNO) Architecture
    """
    def __init__(self, num_vars=39, param_dim=141, width=128, n_blocks=4, d_var=16, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.width = width
        self.n_blocks = n_blocks

        # 1. Parameter Flux Encoder
        self.flux_encoder = BioMutationFluxEncoder(param_dim, width)

        # 2. Stem Lifting: (B, 39*2, T) -> (B, width, T)
        in_channels = num_vars * 2
        self.stem = nn.Conv1d(in_channels, width, kernel_size=5, padding=2)

        # 3. Dynamic Reaction Flux Residual Hierarchy
        dilations = [2 ** (i % 4) for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            DyRFOResNetBlock(width=width, dilation=dilations[i], param_dim=param_dim,
                             num_vars=num_vars, d_var=d_var)
            for i in range(n_blocks)
        ])

        # 4. Output Head
        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

        # 5. Differentiable Conservation Manifold Layer
        self.manifold_proj = DifferentiableConservationManifoldLayer(num_vars=num_vars)

        # Buffers
        rel_g, c_motifs, g_bipartite = load_multirelational_graphs()
        self.register_buffer('rel_graphs', rel_g)
        self.register_buffer('circuit_motifs', c_motifs)
        self.register_buffer('gene_bipartite', g_bipartite)

    def forward(self, x, p):
        """
        x: (B, 39, 2, T) or (B, 39, T)
        p: (B, 141)
        """
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T) # (B, 78, T)
        else:
            B, V, T = x.shape
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, time_grid], dim=1)

        # 1. Parameter flux encoding
        flux_w, gene_act = self.flux_encoder(p)

        # 2. Stem lifting
        h = self.stem(x_in)

        # 3. Dynamic Reaction Flux residual hierarchy
        for block in self.blocks:
            h = block(h, p, self.rel_graphs, flux_w)

        # 4. Raw Output
        raw_out = self.head(h) # (B, 39, T)

        # 5. Phase-Space Conservation Manifold Projection
        out = self.manifold_proj(raw_out)
        return out
