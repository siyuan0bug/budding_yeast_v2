#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HeteroGraphResNetFiLM (High-Efficiency Spatial-Temporal Multi-Relational ResNet)
==============================================================================
Infuses the 6 biochemical relational graphs (Chen 2004 Table 1) and 4 functional
circuit motifs into ResNetFiLM with optimal O(B * width * T) memory complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .bio_hetero_causal_graph import load_multirelational_graphs, BioMutationFluxEncoder


class HeteroGraphMessagePassing(nn.Module):
    """
    Memory-efficient 6-relation message passing across 39 biochemical variables.
    Complexity: O(B * T * 39 * d_var), exactly like GraphResNetFiLM.
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

    def forward(self, x, rel_graphs, flux_weights):
        """
        x: (B, width, T)
        rel_graphs: (6, 39, 39)
        flux_weights: (B, 6)
        """
        B, W, T = x.shape
        # (B, T, W) -> (B, T, 39, d_var)
        x_T = x.permute(0, 2, 1)
        x_vars = self.to_vars(x_T).reshape(B, T, self.num_vars, self.d_var)

        out_vars = torch.zeros_like(x_vars)
        for r in range(self.num_relations):
            A_r = rel_graphs[r] # (39, 39)
            w_r = flux_weights[:, r].view(B, 1, 1, 1) # (B, 1, 1, 1)
            # (B, T, 39, d_var) @ (d_var, d_var)
            x_w = torch.matmul(x_vars, self.rel_weights[r])
            # A_r @ x_w across 39 variables
            prop = torch.einsum('nm,btmc->btnc', A_r, x_w)
            out_vars = out_vars + w_r * prop

        out_flat = out_vars.reshape(B, T, self.num_vars * self.d_var)
        out = self.from_vars(out_flat).permute(0, 2, 1) # (B, W, T)
        return self.act(self.norm(out))


class HeteroCircuitAttention(nn.Module):
    """
    4-Head Circuit Motif Attention over 39 biochemical variables.
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

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_attn ** 0.5) # (B, T, 39, 39)
        # Apply biological circuit motif spatial bias: (4, 39) -> (39, 39)
        motif_bias = torch.einsum('hn,hm->nm', circuit_motifs, circuit_motifs).unsqueeze(0).unsqueeze(0)
        scores = scores + motif_bias * 2.0

        attn = F.softmax(scores, dim=-1)
        out_vars = torch.matmul(attn, V) # (B, T, 39, d_attn)

        out_flat = out_vars.reshape(B, T, self.num_vars * self.d_attn)
        out = self.from_vars(out_flat).permute(0, 2, 1)
        return out


class HeteroGraphResNetBlock(nn.Module):
    """
    High-efficiency 1D Dilated Residual Block with Multi-Relational Graph Message Passing and FiLM.
    """
    def __init__(self, width=64, dilation=1, param_dim=141, num_vars=39, d_var=16):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, width)
        self.conv1 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.act = nn.GELU()

        # FiLM generator
        self.film_gen = nn.Sequential(
            nn.Linear(param_dim, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width * 2)
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

        # Multi-relational graph propagation
        self.graph_mp = HeteroGraphMessagePassing(width=width, num_vars=num_vars, d_var=d_var)

    def forward(self, x, p, rel_graphs, flux_weights):
        """
        x: (B, width, T)
        p: (B, 141)
        """
        res = x

        # 1. First temporal conv + FiLM
        h = self.norm1(x)
        film1 = self.film_gen(p).unsqueeze(-1) # (B, 2*W, 1)
        gamma1, beta1 = torch.chunk(film1, 2, dim=1)
        h = h * (1.0 + torch.tanh(gamma1)) + beta1
        h = self.act(self.conv1(h))

        # 2. Second temporal conv + FiLM
        h = self.norm2(h)
        film2 = self.film_gen(p).unsqueeze(-1)
        gamma2, beta2 = torch.chunk(film2, 2, dim=1)
        h = h * (1.0 + torch.tanh(gamma2)) + beta2
        h = self.conv2(h)

        # 3. Multi-relational graph message passing
        h_graph = self.graph_mp(h, rel_graphs, flux_weights)

        return self.act(res + h + h_graph)


class HeteroGraphResNetFiLM(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4, d_var=16, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.width = width
        self.n_blocks = n_blocks

        # 1. Flux & Gene Activity Encoder
        self.flux_encoder = BioMutationFluxEncoder(param_dim, width)

        # 2. Stem Lifting: (B, 39*2, T) -> (B, width, T)
        in_channels = num_vars * 2
        self.stem = nn.Conv1d(in_channels, width, kernel_size=5, padding=2)

        # 3. Multi-Relational Dilated Residual Blocks
        dilations = [2 ** (i % 4) for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            HeteroGraphResNetBlock(width=width, dilation=dilations[i], param_dim=param_dim,
                                   num_vars=num_vars, d_var=d_var)
            for i in range(n_blocks)
        ])

        # 4. Circuit Motif Attention
        self.circuit_attn = HeteroCircuitAttention(width=width, num_vars=num_vars, d_attn=d_var)

        # 5. Output Projection: (B, width, T) -> (B, 39, T)
        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

        # Buffers
        rel_g, c_motifs, g_bipartite = load_multirelational_graphs()
        self.register_buffer('rel_graphs', rel_g)           # (6, 39, 39)
        self.register_buffer('circuit_motifs', c_motifs)   # (4, 39)
        self.register_buffer('gene_bipartite', g_bipartite) # (27, 39)

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

        # 1. Encode parameter flux
        flux_w, gene_act = self.flux_encoder(p)

        # 2. Stem lifting
        h = self.stem(x_in) # (B, width, T)

        # 3. HeteroGraph ResNet Blocks
        for block in self.blocks:
            h = block(h, p, self.rel_graphs, flux_w)

        # 4. Circuit Motif Attention
        h_attn = self.circuit_attn(h, self.circuit_motifs)
        h = h + h_attn

        # 5. Output
        out = self.head(h) # (B, 39, T)
        return out
