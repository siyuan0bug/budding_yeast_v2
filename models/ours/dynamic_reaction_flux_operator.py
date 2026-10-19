#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DynamicReactionFluxOperator (DyRFO)
===================================
A novel biological neural operator with continuous-time dynamic reaction flux gating.
Unlike static GCNs, DyRFO computes time-varying, phase-dependent biochemical reaction
fluxes: A_r(t) = Sigmoid(Q_r(X(t)) K_r(X(t))^T / sqrt(d) + FluxBias_r(p)) * A_r^prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .bio_hetero_causal_graph import load_multirelational_graphs, BioMutationFluxEncoder


class DynamicReactionFluxBlock(nn.Module):
    """
    Time-Varying Dynamic Biochemical Reaction Flux Layer.
    Computes instantaneous reaction connectivity conditioned on state X(t) and parameter flux p.
    """
    def __init__(self, width=128, num_vars=39, d_var=16, num_relations=6, param_dim=141):
        super().__init__()
        self.num_vars = num_vars
        self.d_var = d_var
        self.num_relations = num_relations

        self.to_vars = nn.Linear(width, num_vars * d_var)
        self.from_vars = nn.Linear(num_vars * d_var, width)

        # Relation-specific state transformations
        self.w_v = nn.Linear(d_var, d_var)
        self.gate_net = nn.Sequential(
            nn.Linear(d_var, d_var),
            nn.GELU(),
            nn.Linear(d_var, d_var),
            nn.Sigmoid()
        )

        # Parameter flux bias generator
        self.flux_bias_gen = nn.Sequential(
            nn.Linear(param_dim, width),
            nn.GELU(),
            nn.Linear(width, num_relations)
        )

        self.norm = nn.GroupNorm(1, width)
        self.act = nn.GELU()

    def forward(self, x, p, rel_graphs, flux_weights, untyped_graph=None):
        """
        x: (B, width, T)
        p: (B, 141)
        rel_graphs: (6, 39, 39)
        flux_weights: (B, 6)
        """
        B, W, T = x.shape
        x_T = x.permute(0, 2, 1) # (B, T, W)
        x_vars = self.to_vars(x_T).reshape(B, T, self.num_vars, self.d_var) # (B, T, 39, d_var)

        # Parameter flux bias & effective multi-relational adjacency: (B, 39, 39)
        p_bias = torch.sigmoid(self.flux_bias_gen(p)) # (B, 6)
        eff_weights = flux_weights * p_bias # (B, 6)
        A_eff = torch.einsum('br,rij->bij', eff_weights, rel_graphs) # (B, 39, 39)
        if untyped_graph is not None:
            A_eff = A_eff + eff_weights.mean(-1)[:, None, None] * untyped_graph

        # 1. State-dependent dynamic kinetic gating (captures instant enzyme activation)
        gates = self.gate_net(x_vars) # (B, T, 39, d_var)
        V = self.w_v(x_vars) * gates # (B, T, 39, d_var)

        # 2. High-efficiency dynamic message passing: (B, T, 39, d_var)
        # O(B * T * V * d_var) memory complexity instead of O(B * T * V^2)
        out_vars = torch.matmul(A_eff.unsqueeze(1), V) # (B, 1, 39, 39) x (B, T, 39, d_var) -> (B, T, 39, d_var)

        out_flat = out_vars.reshape(B, T, self.num_vars * self.d_var)
        out = self.from_vars(out_flat).permute(0, 2, 1) # (B, width, T)
        return self.act(self.norm(out))


class DyRFOResNetBlock(nn.Module):
    """
    Dilated Residual Block with Dynamic Reaction-Flux Gating and FiLM Parameter Conditioning.
    """
    def __init__(self, width=128, dilation=1, param_dim=141, num_vars=39, d_var=16):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, width)
        self.conv1 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.act = nn.GELU()

        # FiLM Conditioning
        self.film = nn.Sequential(
            nn.Linear(param_dim, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width * 2)
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        # Dynamic Reaction-Flux Layer
        self.dy_flux = DynamicReactionFluxBlock(width=width, num_vars=num_vars, d_var=d_var, param_dim=param_dim)

    def forward(self, x, p, rel_graphs, flux_weights, untyped_graph=None):
        res = x

        # 1. First temporal conv + FiLM
        h = self.norm1(x)
        film1 = self.film(p).unsqueeze(-1)
        g1, b1 = torch.chunk(film1, 2, dim=1)
        h = h * (1.0 + torch.tanh(g1)) + b1
        h = self.act(self.conv1(h))

        # 2. Second temporal conv + FiLM
        h = self.norm2(h)
        film2 = self.film(p).unsqueeze(-1)
        g2, b2 = torch.chunk(film2, 2, dim=1)
        h = h * (1.0 + torch.tanh(g2)) + b2
        h = self.conv2(h)

        # 3. Dynamic Reaction Flux Passing
        h_flux = self.dy_flux(h, p, rel_graphs, flux_weights, untyped_graph)

        return self.act(res + h + h_flux)


class DynamicReactionFluxOperator(nn.Module):
    """
    Flagship DyRFO Neural Operator Architecture (Width=128 by default).
    """
    def __init__(self, num_vars=39, param_dim=141, width=128, n_blocks=4, d_var=16, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.width = width
        self.n_blocks = n_blocks

        # 1. Parameter & Mutation Flux Encoder
        self.flux_encoder = BioMutationFluxEncoder(param_dim, width)

        # 2. Stem Input Lifting: (B, 39*2, T) -> (B, width, T)
        in_channels = num_vars * 2
        self.stem = nn.Conv1d(in_channels, width, kernel_size=5, padding=2)

        # 3. Dynamic Reaction Flux Residual Blocks
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

        # Load & register physical prior graphs
        load_multirelational_graphs().register(self, num_vars)

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
            h = block(h, p, self.rel_graphs, flux_w, self.untyped_graph)

        # 4. Output projection
        out = self.head(h) # (B, 39, T)
        return out
