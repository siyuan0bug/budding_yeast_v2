#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HeteroGraphDualPathNet (High-Efficiency Dual-Path Spatial-Spectral Operator)
===========================================================================
Dual-path neural operator combining a Multi-Relational Biochemical Graph Stream
(Spatial Biochemical Message Passing) and a Continuous 1D Fourier Spectral Stream
(Frequency & Phase Periodic Locking) with Cross-Domain Gated Fusion.
Memory complexity: O(B * width * T), strictly optimized for large batch sizes (512/1024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .bio_hetero_causal_graph import (
    load_multirelational_graphs,
    BioMutationFluxEncoder,
    MultiRelationalGraphConvBlock,
    CircuitMotifCrossAttention
)


class SpectralConv1dBlock(nn.Module):
    """
    1D Continuous Spectral Convolution Block with AMP Float32 protection.
    """
    def __init__(self, in_channels, out_channels, modes1=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        orig_dtype = x.dtype
        x_f32 = x.float()
        B, C, T = x_f32.shape

        x_ft = torch.fft.rfft(x_f32, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)

        modes = min(self.modes1, x_ft.shape[-1])
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights1[:, :, :modes])

        x_out = torch.fft.irfft(out_ft, n=T, dim=-1)
        return x_out.to(orig_dtype)


class HeteroGraphDualPathNet(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, modes=32, width=64, n_blocks=4, d_var=16, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.modes = modes
        self.width = width
        self.n_blocks = n_blocks

        # 1. Parameter & Flux Encoder
        self.flux_encoder = BioMutationFluxEncoder(param_dim, width)

        # 2. Input Stem Projections: (B, 39*2, T) -> (B, width, T)
        in_channels = num_vars * 2
        self.spatial_in = nn.Conv1d(in_channels, width, kernel_size=1)
        self.spectral_in = nn.Conv1d(in_channels, width, kernel_size=1)

        # 3. Dual-Stream Blocks
        self.graph_blocks = nn.ModuleList([
            MultiRelationalGraphConvBlock(width=width, num_vars=num_vars, d_var=d_var, num_relations=6)
            for _ in range(n_blocks)
        ])
        self.spectral_blocks = nn.ModuleList([
            SpectralConv1dBlock(width, width, modes1=modes)
            for _ in range(n_blocks)
        ])
        self.spectral_w = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=1)
            for _ in range(n_blocks)
        ])
        self.gate_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width * 2, width, kernel_size=1),
                nn.Sigmoid()
            )
            for _ in range(n_blocks)
        ])

        # 4. Circuit Motif Attention
        self.circuit_attn = CircuitMotifCrossAttention(width=width, num_vars=num_vars, d_attn=d_var)

        # 5. Output Projection
        self.out_proj = nn.Sequential(
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

        # 1. Encode parameter flux
        flux_w, gene_act = self.flux_encoder(p)

        # 2. Dual-stream stem initialization: (B, width, T)
        h_spatial = self.spatial_in(x_in)
        h_spectral = self.spectral_in(x_in)

        # 3. Dual-Stream Iterative Processing with Gated Fusion
        for i in range(self.n_blocks):
            # Spatial 6-relation graph step
            h_spatial = self.graph_blocks[i](h_spatial, self.rel_graphs, flux_w, self.untyped_graph)

            # Spectral 1D Fourier step
            h_spec_conv = self.spectral_blocks[i](h_spectral) + self.spectral_w[i](h_spectral)
            h_spectral = F.silu(h_spec_conv)

            # Cross-Domain Gated Fusion
            fusion_cat = torch.cat([h_spatial, h_spectral], dim=1) # (B, 2*width, T)
            gate = self.gate_projs[i](fusion_cat) # (B, width, T)
            h_spectral = h_spectral * gate + h_spatial * (1.0 - gate)
            h_spatial = h_spatial * (1.0 - gate) + h_spectral * gate

        # 4. Circuit Motif Attention
        h_out = self.circuit_attn(h_spectral, self.circuit_motifs)

        # 5. Output
        out = self.out_proj(h_out) # (B, 39, T)
        return out
