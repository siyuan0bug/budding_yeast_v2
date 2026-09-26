#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HeteroGraphChannelHyperFNO
==========================
Hypernetwork-driven Fourier Neural Operator where channel-mixing kernels
are guided by the 27-gene bipartite matrix and 6 biochemical relational graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .bio_hetero_causal_graph import load_multirelational_graphs, BioMutationFluxEncoder


class SpectralConv1dBlock(nn.Module):
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


class BioGuidedChannelHyperBlock(nn.Module):
    """
    FNO block where the channel transformation matrix is dynamically predicted by
    a Hypernetwork guided by 27-gene activities and 6-relation flux modulations.
    """
    def __init__(self, width, modes=32):
        super().__init__()
        self.width = width
        self.spectral = SpectralConv1dBlock(width, width, modes1=modes)
        self.w = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(4, width)
        self.act = nn.SiLU()

    def forward(self, x, channel_mixing_matrix):
        """
        x: (B, width, T)
        channel_mixing_matrix: (B, width, width)
        """
        B, C, T = x.shape
        # 1. Standard spectral + bypass conv
        h_spec = self.spectral(x)
        h_w = self.w(x)
        h = h_spec + h_w

        # 2. Dynamic channel mixing from hypernetwork
        # h: (B, width, T) -> einsum with (B, width, width)
        h_mixed = torch.einsum('bij,bjt->bit', channel_mixing_matrix, h)
        out = self.act(self.norm(h + h_mixed))
        return out


class HeteroGraphChannelHyperFNO(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, modes=32, width=64, n_blocks=4, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.modes = modes
        self.width = width
        self.n_blocks = n_blocks

        # 1. Flux & Gene Activity Encoder
        self.flux_encoder = BioMutationFluxEncoder(param_dim, width)

        # 2. Bio-Guided Hypernetwork: Maps (141 params + 27 gene activities) -> channel mixing matrices
        self.hypernet = nn.Sequential(
            nn.Linear(param_dim + 27, width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, n_blocks * width * width)
        )

        # 3. Input Projection
        self.in_proj = nn.Conv1d(num_vars + 1, width, kernel_size=1)

        # 4. Bio-Guided Channel Hyper Blocks
        self.blocks = nn.ModuleList([
            BioGuidedChannelHyperBlock(width, modes=modes)
            for _ in range(n_blocks)
        ])

        # 5. Output Projection
        self.out_proj = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

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
            B, N, _, T = x.shape
            ic = x[:, :, 0, :]
            time_grid = x[:, :, 1, :]
        else:
            B, N, T = x.shape
            ic = x
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).repeat(B, N, 1)

        # 1. Encode parameter flux & gene activity
        flux_w, gene_act = self.flux_encoder(p) # (B, 6), (B, 27)

        # 2. Generate Bio-Guided channel mixing matrices: (B, n_blocks, width, width)
        hyper_in = torch.cat([p, gene_act], dim=-1) # (B, 141 + 27)
        mixing_matrices = self.hypernet(hyper_in).view(B, self.n_blocks, self.width, self.width)
        # Normalize to prevent explosion
        mixing_matrices = torch.tanh(mixing_matrices) * 0.5

        # 3. Input Projection
        time_1d = time_grid[:, 0:1, :] # (B, 1, T)
        h = self.in_proj(torch.cat([ic, time_1d], dim=1)) # (B, width, T)

        # 4. Forward through HyperFNO Blocks
        for i, block in enumerate(self.blocks):
            h = block(h, mixing_matrices[:, i])

        # 5. Output
        out = self.out_proj(h) # (B, 39, T)
        return F.softplus(out)
