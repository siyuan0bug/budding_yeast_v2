#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioHeteroCausalGraphNet (Biochemical Multi-Relational Causal Graph Neural Operator)
=================================================================================
Uses 6 ODE-term relation overlays plus untyped source support, 4 circuit motif attentions
(START, SPINDLE, MEN, RESET), dynamic mutation flux modulation, and algebraic conservation laws.
Memory complexity: O(B * width * T), strictly optimized for large batch sizes (512/1024).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .relation_graph_contracts import RelationBundle, untyped_message

# 39 Variable Definitions
(nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5,
 nSIC1P, nC2P, nC5P, nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T,
 nCDC20, nCDH1T, nCDH1, nCDC14T, nCDC14, nNET1T, nNET1, nRENT, nTEM1,
 nCDC15, nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, nVi20, nlte1, nBUB2) = range(39)


def load_multirelational_graphs(device='cpu'):
    """Return a bundle whose register() builds ODE-backed topology.

    The old NPZ supplies ONLY circuit/module metadata, never graph edges.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(current_dir, "../../data/yeast_multirelational_graphs.npz")
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            circuit_motifs = torch.tensor(data['circuit_motifs'], dtype=torch.float32, device=device)
            gene_bipartite = torch.tensor(data['gene_bipartite'], dtype=torch.float32, device=device)
    else:
        from budding_yeast_v2.data.build_bio_multirelational_graphs import (
            build_circuit_motifs, build_gene_bipartite_incidence)
        circuit_motifs = torch.tensor(build_circuit_motifs(), dtype=torch.float32, device=device)
        gene_bipartite = torch.tensor(build_gene_bipartite_incidence(), dtype=torch.float32, device=device)
    for value, shape in ((circuit_motifs, (4, 39)), (gene_bipartite, (27, 39))):
        if value.shape != shape or not torch.isfinite(value).all():
            raise ValueError('Invalid circuit/module metadata')
    return RelationBundle(circuit_motifs, gene_bipartite, device)


class BioMutationFluxEncoder(nn.Module):
    """
    Encodes 141-dim kinetic parameters into 6-relation flux modulation weights
    and 27-gene activity vectors.
    """
    def __init__(self, param_dim=141, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.flux_head = nn.Linear(hidden_dim, 6) # (B, 6) relation weights
        self.gene_head = nn.Linear(hidden_dim, 27) # (B, 27) gene activity weights

    def forward(self, p):
        feat = self.net(p)
        flux_w = F.softplus(self.flux_head(feat)) # (B, 6)
        gene_act = torch.sigmoid(self.gene_head(feat)) # (B, 27)
        return flux_w, gene_act


class MultiRelationalGraphConvBlock(nn.Module):
    """
    Performs 6-relation message passing over the 39 state variables.
    H' = sum_r w_r*A_r*H*W_r + mean(w)*A_untyped*H*mean(W) + H.
    Relation labels are annotations; no fixed physical derivative sign is implied.
    """
    def __init__(self, width=64, num_vars=39, d_var=16, num_relations=6):
        super().__init__()
        self.num_vars = num_vars
        self.d_var = d_var
        self.num_relations = num_relations

        self.to_vars = nn.Linear(width, num_vars * d_var)
        self.rel_weights = nn.Parameter(torch.Tensor(num_relations, d_var, d_var))
        nn.init.kaiming_uniform_(self.rel_weights, a=5**0.5)

        self.from_vars = nn.Linear(num_vars * d_var, width)
        self.norm = nn.GroupNorm(1, width)
        self.act = nn.SiLU()

    def forward(self, x, rel_graphs, flux_weights, untyped_graph=None):
        """
        x: (B, width, T)
        rel_graphs: (6, 39, 39)
        flux_weights: (B, 6)
        """
        B, W, T = x.shape
        x_T = x.permute(0, 2, 1) # (B, T, W)
        x_vars = self.to_vars(x_T).reshape(B, T, self.num_vars, self.d_var)

        out_vars = torch.zeros_like(x_vars)
        for r in range(self.num_relations):
            A_r = rel_graphs[r]
            w_r = flux_weights[:, r].view(B, 1, 1, 1)
            x_w = torch.matmul(x_vars, self.rel_weights[r])
            prop = torch.einsum('nm,btmc->btnc', A_r, x_w)
            out_vars = out_vars + w_r * prop
        if untyped_graph is not None:
            out_vars = out_vars + untyped_message(x_vars, self.rel_weights, flux_weights, untyped_graph)

        out_flat = out_vars.reshape(B, T, self.num_vars * self.d_var)
        out = self.from_vars(out_flat).permute(0, 2, 1) # (B, W, T)
        return self.act(self.norm(out)) + x


class CircuitMotifCrossAttention(nn.Module):
    """
    4-Head Circuit Motif Cross Attention over 39 biochemical variables.
    """
    def __init__(self, width=64, num_vars=39, d_attn=16):
        super().__init__()
        self.num_vars = num_vars
        self.d_attn = d_attn
        self.to_vars = nn.Linear(width, num_vars * d_attn)
        self.from_vars = nn.Linear(num_vars * d_attn, width)

        self.q_proj = nn.Linear(d_attn, d_attn)
        self.k_proj = nn.Linear(d_attn, d_attn)
        self.v_proj = nn.Linear(d_attn, d_attn)

    def forward(self, x, circuit_motifs):
        """
        x: (B, width, T)
        circuit_motifs: (4, 39)
        """
        B, W, T = x.shape
        x_T = x.permute(0, 2, 1)
        x_vars = self.to_vars(x_T).reshape(B, T, self.num_vars, self.d_attn)

        Q = self.q_proj(x_vars)
        K = self.k_proj(x_vars)
        V = self.v_proj(x_vars)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_attn ** 0.5)
        motif_bias = torch.einsum('hn,hm->nm', circuit_motifs, circuit_motifs).unsqueeze(0).unsqueeze(0)
        scores = scores + motif_bias * 2.0

        attn = F.softmax(scores, dim=-1)
        out_vars = torch.matmul(attn, V)

        out_flat = out_vars.reshape(B, T, self.num_vars * self.d_attn)
        out = self.from_vars(out_flat).permute(0, 2, 1)
        return out + x


class BioHeteroCausalGraphNet(nn.Module):
    """
    Flagship BioHetero-CausalGraphNet Architecture
    """
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4, d_var=16, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.width = width
        self.n_blocks = n_blocks

        # 1. Parameter & Dynamic Flux Encoder
        self.flux_encoder = BioMutationFluxEncoder(param_dim, width)

        # 2. Stem Lifting: (B, 39*2, T) -> (B, width, T)
        in_channels = num_vars * 2
        self.stem = nn.Conv1d(in_channels, width, kernel_size=5, padding=2)

        # 3. Multi-Relational Graph Convolution Layers
        self.graph_blocks = nn.ModuleList([
            MultiRelationalGraphConvBlock(width=width, num_vars=num_vars, d_var=d_var, num_relations=6)
            for _ in range(n_blocks)
        ])

        # 4. Circuit Motif Attention Layer
        self.circuit_attn = CircuitMotifCrossAttention(width=width, num_vars=num_vars, d_attn=d_var)

        # 5. Temporal Multi-Scale 1D Conv Inception Backbone
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, kernel_size=k, padding=k//2),
                nn.GroupNorm(4, width),
                nn.SiLU(),
                nn.Conv1d(width, width, kernel_size=1)
            )
            for k in [3, 7, 15, 31]
        ])

        # 6. FiLM Dynamic Modulation from parameters
        self.film_gamma = nn.Linear(param_dim, width)
        self.film_beta = nn.Linear(param_dim, width)

        # 7. Output Projection: (B, width, T) -> (B, 39, T)
        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

        # Buffers
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
            x_in = torch.cat([x, time_grid], dim=1) # (B, 78, T)

        # 1. Encode mutation flux
        flux_w, gene_act = self.flux_encoder(p)

        # 2. Stem lifting
        h = self.stem(x_in) # (B, width, T)

        # 3. Multi-Relational Graph Message Passing
        for block in self.graph_blocks:
            h = block(h, self.rel_graphs, flux_w, self.untyped_graph)

        # 4. Circuit Motif Attention
        h = self.circuit_attn(h, self.circuit_motifs)

        # 5. Temporal Multi-Scale Inception Convolutions
        t_out = sum(conv(h) for conv in self.temporal_convs) # (B, width, T)

        # 6. Dynamic FiLM Modulation
        gamma = self.film_gamma(p).unsqueeze(-1) # (B, width, 1)
        beta = self.film_beta(p).unsqueeze(-1)
        t_out = t_out * (1.0 + gamma) + beta

        # 7. Output Projection
        out = self.head(t_out) # (B, 39, T)
        return out
