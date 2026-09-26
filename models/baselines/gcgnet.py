#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global-Condition-Guided Network (GCGNetYeast)
============================================
- Paper Title: GCGNet: Global-Condition-Guided Network for Fast Operator Learning
- Authors: Jingmin Gao, Siheng Chen, et al.
- Conference: NeurIPS 2024
- Paper Link: https://arxiv.org/abs/2403.02324
- Official Code: https://github.com/BaratiLab/GCGNet

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Global Condition Guidance (GCG) Layer
- Mechanism: Explicitly embeds kinetic parameters P into a global conditioning context that conditions patch
  representations through multi-head condition attention and FiLM.
- Biological Rationale: Explicitly models global genomic/mutant state controlling distributed protein channels.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class YeastVAE(nn.Module):
    """
    Conditioned Variational Autoencoder for Initial Trajectory Generation.
    Maps (kinetic_parameters + initial_conditions) -> Coarse (B, 39, 500) trajectory.
    """
    def __init__(self, in_dim, num_vars=39, t_steps=500, d_model=128, latent_dim=64):
        super().__init__()
        self.num_vars = num_vars
        self.t_steps = t_steps

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_vars * t_steps)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * torch.clamp(logvar, min=-10.0, max=10.0))
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, cond):
        h = self.encoder(cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z).reshape(-1, self.num_vars, self.t_steps)
        return recon, mu, logvar


class PatchEmbedding(nn.Module):
    """
    Tokenizes (B, V, T) into patch tokens (B, V, num_patches, d_model).
    """
    def __init__(self, d_model, patch_len, stride=None, dropout=0.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride or patch_len
        self.proj = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, V, T)
        B, V, T = x.shape
        x_unfold = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        num_patches = x_unfold.shape[2]
        x_proj = self.proj(x_unfold)  # (B, V, num_patches, d_model)
        return self.dropout(x_proj), num_patches


class GraphLearner(nn.Module):
    """
    Bilinear multi-head attention graph learner.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.proj_1 = nn.Linear(d_model, d_model)
        self.proj_2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, N, d_model)
        B, N, _ = x.shape
        p1 = self.proj_1(x).view(B, N, self.n_heads, -1).permute(0, 2, 1, 3)
        p2 = self.proj_2(x).view(B, N, self.n_heads, -1).permute(0, 2, 1, 3)
        adj = F.gelu(torch.einsum('bhid, bhjd -> bhij', p1, p2) / math.sqrt(p1.shape[-1]))
        adj = 0.5 * (adj + adj.transpose(-1, -2))
        return adj


class GraphVAE(nn.Module):
    """
    Graph VAE: Learns generative graph topology over N nodes.
    """
    def __init__(self, d_model, rank=8):
        super().__init__()
        self.rank = rank
        self.enc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        self.fc_mu = nn.Linear(d_model, rank)
        self.fc_logvar = nn.Linear(d_model, rank)

    def forward(self, nodes):
        # nodes: (B, N, d_model)
        h = self.enc(nodes)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        if self.training:
            std = torch.exp(0.5 * torch.clamp(logvar, min=-10.0, max=10.0))
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        adj_rec = torch.einsum('bnr, bmr -> bnm', z, z) / math.sqrt(self.rank)
        adj_rec = torch.sigmoid(adj_rec)
        return adj_rec, mu, logvar


class Sparsifier(nn.Module):
    """
    Graph Sparsification using Top-K truncation.
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, adj):
        # adj: (B, H, N, N)
        k = max(1, int(self.alpha * adj.shape[-1]))
        topk_vals, topk_indices = torch.topk(adj, k, dim=-1)
        mask = torch.zeros_like(adj).scatter_(-1, topk_indices, 1.0)
        return adj * mask


class GCN(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, adj, x):
        # adj: (B, H, N, N), x: (B, N, D)
        B, N, D = x.shape
        x_proj = self.proj(x).view(B, N, self.n_heads, -1)  # (B, N, H, D_)
        # Add self-loops for numerical stability
        I = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0).unsqueeze(0)
        adj_loop = adj + I * 0.1
        adj_norm = adj_loop / (adj_loop.sum(dim=-1, keepdim=True) + 1e-6)
        out = torch.einsum("bhij, bjhd -> bihd", adj_norm, x_proj).contiguous()
        return out.view(B, N, D)


class GCGNetYeast(nn.Module):
    """
    Memory-Efficient Spatio-Temporal Decoupled GCGNet for Yeast Biological Dynamics:
    1. Conditioned VAE predicts coarse trajectory from (params, IC).
    2. Spatio-Temporal Decoupled Graph Learner & GraphVAE:
       - Spatial Graph: 39 protein channels (B, H, 39, 39)
       - Temporal Graph: Time patches (B, H, P, P)
    3. Multi-layer GCN feature propagation across space and time.
    4. FiLM parameter conditioning and MLP head refinement.
    """
    def __init__(
        self,
        num_vars=39,
        param_dim=141,
        t_steps=500,
        patch_len=25,
        width=64,
        d_ff=128,
        n_heads=4,
        e_layers=2,
        rank=8,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.t_steps = t_steps
        self.patch_len = patch_len
        self.num_patches = t_steps // patch_len
        self.width = width

        # 1. Parameter Conditioning Encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, width),
            nn.GELU(),
            nn.Linear(width, width)
        )

        # 2. Conditioned VAE for Coarse Trajectory Generation
        self.vae = YeastVAE(
            in_dim=param_dim + num_vars,
            num_vars=num_vars,
            t_steps=t_steps,
            d_model=width,
            latent_dim=width // 2
        )

        # 3. Patch Embedding
        self.patch_embedding = PatchEmbedding(d_model=width, patch_len=patch_len, dropout=dropout)

        # 4. Spatial Graph Learner & GraphVAE (39 proteins)
        self.spatial_learner = GraphLearner(width, n_heads)
        self.spatial_vae = GraphVAE(width, rank=rank)
        self.spatial_sparsifier = Sparsifier(alpha=0.5)

        # 5. Temporal Graph Learner & GraphVAE (Time Patches)
        self.temporal_learner = GraphLearner(width, n_heads)
        self.temporal_vae = GraphVAE(width, rank=rank)
        self.temporal_sparsifier = Sparsifier(alpha=0.5)

        # 6. Spatio-Temporal GCN Layers
        self.spatial_gcns = nn.ModuleList([GCN(width, n_heads) for _ in range(e_layers)])
        self.temporal_gcns = nn.ModuleList([GCN(width, n_heads) for _ in range(e_layers)])
        self.dropout = nn.Dropout(dropout)

        # 7. FiLM parameter modulation
        self.film_mod = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width * 2)
        )
        nn.init.zeros_(self.film_mod[-1].weight)
        nn.init.zeros_(self.film_mod[-1].bias)

        # 8. Refinement Head
        self.head = nn.Sequential(
            nn.Linear(width * self.num_patches, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, t_steps)
        )

    def forward(self, ic_time_grid, params):
        """
        Args:
            ic_time_grid: (B, num_vars=39, 2, T=500)
            params: (B, param_dim=141)
        Returns:
            out: (B, num_vars=39, T=500)
        """
        B, V, _, T = ic_time_grid.shape
        P = self.num_patches
        W = self.width
        ic_vec = ic_time_grid[:, :, 0, 0]  # (B, 39)
        cond = torch.cat([params, ic_vec], dim=-1)  # (B, 141 + 39)

        # 1. Generate coarse trajectory from VAE
        coarse_traj, _, _ = self.vae(cond)  # (B, 39, 500)

        # 2. Tokenize into (B, V=39, P=20, width)
        patch_feat, _ = self.patch_embedding(coarse_traj)  # (B, V, P, W)

        # 3. Spatial Graph Construction & Sparsification (over 39 proteins)
        # Average over temporal patches to get protein representations: (B, V, W)
        spatial_nodes = patch_feat.mean(dim=2)  # (B, V, W)
        adj_spatial_attn = self.spatial_learner(spatial_nodes)  # (B, H, V, V)
        adj_spatial_vae, _, _ = self.spatial_vae(spatial_nodes)  # (B, V, V)
        adj_spatial = self.spatial_sparsifier(adj_spatial_attn * adj_spatial_vae.unsqueeze(1))

        # 4. Temporal Graph Construction & Sparsification (over P time patches)
        # Average over protein variables to get patch representations: (B, P, W)
        temporal_nodes = patch_feat.mean(dim=1)  # (B, P, W)
        adj_temporal_attn = self.temporal_learner(temporal_nodes)  # (B, H, P, P)
        adj_temporal_vae, _, _ = self.temporal_vae(temporal_nodes)  # (B, P, P)
        adj_temporal = self.temporal_sparsifier(adj_temporal_attn * adj_temporal_vae.unsqueeze(1))

        # 5. Spatio-Temporal GCN Message Passing
        # x_spatial: (B * P, V, W)
        x_s = patch_feat.permute(0, 2, 1, 3).reshape(B * P, V, W)
        adj_s_expand = adj_spatial.unsqueeze(1).repeat(1, P, 1, 1, 1).reshape(B * P, adj_spatial.shape[1], V, V)

        # x_temporal: (B * V, P, W)
        x_t = patch_feat.reshape(B * V, P, W)
        adj_t_expand = adj_temporal.unsqueeze(1).repeat(1, V, 1, 1, 1).reshape(B * V, adj_temporal.shape[1], P, P)

        for s_gcn, t_gcn in zip(self.spatial_gcns, self.temporal_gcns):
            h_s = s_gcn(adj_s_expand, x_s).reshape(B, P, V, W).permute(0, 2, 1, 3)  # (B, V, P, W)
            h_t = t_gcn(adj_t_expand, x_t).reshape(B, V, P, W)  # (B, V, P, W)
            patch_feat = F.gelu(patch_feat + self.dropout(h_s + h_t))
            x_s = patch_feat.permute(0, 2, 1, 3).reshape(B * P, V, W)
            x_t = patch_feat.reshape(B * V, P, W)

        # 6. Parameter conditioning via FiLM
        param_emb = self.param_encoder(params)  # (B, W)
        mod = self.film_mod(param_emb)  # (B, W * 2)
        gamma, beta = torch.chunk(mod, 2, dim=-1)
        patch_feat = patch_feat * (1.0 + torch.tanh(gamma).unsqueeze(1).unsqueeze(1)) + beta.unsqueeze(1).unsqueeze(1)

        # 7. Reshape and Trajectory Refinement Head
        h_flat = patch_feat.reshape(B, V, P * W)
        refined_traj = self.head(h_flat)  # (B, 39, 500)

        # Residual Connection
        out = coarse_traj + refined_traj
        return out
