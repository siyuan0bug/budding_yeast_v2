#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioRefiner-Plus & Sub-module Ablations Suite
============================================
An upgraded family of two-stage neural operators for stiff biochemical systems:
1. BioRefinerCurvature: Sub-direction 1 (Injects 2nd-order temporal curvature/acceleration tensor d^2Y/dt^2).
2. BioRefinerPyramid: Sub-direction 2 (Multi-scale temporal pyramidal receptive field U-blocks).
3. BioRefinerGraph: Sub-direction 3 (Stoichiometric reaction graph-guided 39-variable cross-attention).
4. BioRefinerPlus: The grand fusion model integrating all 3 sub-directions.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .bio_mechanistic_operator import BioMechPathwayOnly


# =========================================================================
# 1. 基础模块: 参数 FiLM 调制的膨胀卷积块
# =========================================================================
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
        gamma, beta = self.film(p).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        res = x
        out = F.gelu(self.norm1(self.conv1(x)))
        out = (1.0 + gamma) * out + beta
        out = self.norm2(self.conv2(out))
        return F.gelu(out + res)


# =========================================================================
# 2. 子方向 2 模块: 多尺度时间金字塔残差块 (Multi-Scale Temporal Pyramidal Block)
# =========================================================================
class TemporalPyramidBlock(nn.Module):
    """
    Multi-Scale Temporal Pyramidal Block.
    Combines narrow local receptive fields (dilations 1, 2 for stiff spikes)
    and wide macroscopic receptive fields (dilations 4, 8, 16 + temporal pooling for cycle baseline).
    """
    def __init__(self, channels, param_dim=141):
        super().__init__()
        branch_dim = channels // 4
        self.b1 = nn.Conv1d(channels, branch_dim, kernel_size=3, padding=1, dilation=1)
        self.b2 = nn.Conv1d(channels, branch_dim, kernel_size=3, padding=2, dilation=2)
        self.b3 = nn.Conv1d(channels, branch_dim, kernel_size=3, padding=4, dilation=4)
        self.b4 = nn.Sequential(
            nn.AvgPool1d(kernel_size=5, stride=1, padding=2),
            nn.Conv1d(channels, branch_dim, kernel_size=1)
        )
        self.fuse = nn.Conv1d(branch_dim * 4, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self.film = nn.Linear(param_dim, channels * 2)

    def forward(self, x, p):
        gamma, beta = self.film(p).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        c1 = self.b1(x)
        c2 = self.b2(x)
        c3 = self.b3(x)
        c4 = self.b4(x)
        concat = torch.cat([c1, c2, c3, c4], dim=1)
        out = self.norm(self.fuse(concat))
        out = (1.0 + gamma) * out + beta
        return F.gelu(out + x)


# =========================================================================
# 3. 子方向 3 模块: 生化反应拓扑图导向的跨变量注意力 (Stoichiometric Graph Cross-Attention)
# =========================================================================
class StoichiometricGraphAttention(nn.Module):
    """
    Graph-Guided Cross-Variable Attention across 39 Biochemical Species.
    Injects yeast signed reaction adjacency topology as a structural bias in attention weights.
    """
    def __init__(self, channels=128, num_vars=39, head_dim=16, adj_matrix=None):
        super().__init__()
        self.num_vars = num_vars
        self.head_dim = head_dim
        self.to_vars = nn.Linear(channels, num_vars * head_dim)
        self.from_vars = nn.Linear(num_vars * head_dim, channels)
        self.q_proj = nn.Linear(head_dim, head_dim)
        self.k_proj = nn.Linear(head_dim, head_dim)
        self.v_proj = nn.Linear(head_dim, head_dim)

        if adj_matrix is None:
            adj_path = 'budding_yeast_v2/data/yeast_signed_adjacency_matrix.npy'
            if os.path.exists(adj_path):
                adj_matrix = np.load(adj_path)
            else:
                adj_matrix = np.eye(num_vars)

        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).abs()
        # Add self-loop and normalize
        adj_tensor = adj_tensor + torch.eye(num_vars)
        adj_norm = adj_tensor / adj_tensor.sum(dim=-1, keepdim=True).clamp(min=1.0)
        self.register_buffer('graph_bias', adj_norm)

    def forward(self, h):
        # h: (B, channels, T)
        B, C, T = h.shape
        h_t = h.permute(0, 2, 1)  # (B, T, channels)
        vars_feat = self.to_vars(h_t).reshape(B, T, self.num_vars, self.head_dim)  # (B, T, 39, 16)

        Q = self.q_proj(vars_feat)  # (B, T, 39, 16)
        K = self.k_proj(vars_feat)  # (B, T, 39, 16)
        V = self.v_proj(vars_feat)  # (B, T, 39, 16)

        attn_raw = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, T, 39, 39)
        # Add structural biochemical reaction graph bias
        attn = F.softmax(attn_raw + 0.5 * self.graph_bias.unsqueeze(0).unsqueeze(0), dim=-1)
        out_vars = torch.matmul(attn, V).reshape(B, T, self.num_vars * self.head_dim)  # (B, T, 39*16)
        h_attn = self.from_vars(out_vars).permute(0, 2, 1)  # (B, channels, T)
        return h + h_attn


# =========================================================================
# 4. 统一 Stage 2 Refiner: 支持任意子模块消融开关
# =========================================================================
class ModularStiffRefiner(nn.Module):
    """
    Modular Stage 2 Stiff Phase Residual Refiner with ablation toggles:
    - use_curvature: Ingests 2nd-order acceleration/curvature tensor d^2Y/dt^2
    - use_pyramid: Employs multi-scale temporal pyramid blocks
    - use_graph_attn: Applies reaction graph-guided cross-attention
    """
    def __init__(self, in_channels=156, num_vars=39, width=128, param_dim=141, n_blocks=4,
                 use_pyramid=False, use_graph_attn=False, adj_matrix=None):
        super().__init__()
        self.use_pyramid = use_pyramid
        self.use_graph_attn = use_graph_attn

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU()
        )

        dilations = [1, 2, 4, 8]
        if self.use_pyramid:
            self.blocks = nn.ModuleList([
                TemporalPyramidBlock(width, param_dim=param_dim)
                for _ in range(n_blocks)
            ])
        else:
            self.blocks = nn.ModuleList([
                FiLMBlock1d(width, dilation=dilations[i % len(dilations)], param_dim=param_dim)
                for i in range(n_blocks)
            ])

        if self.use_graph_attn:
            self.graph_attn = StoichiometricGraphAttention(channels=width, num_vars=num_vars, adj_matrix=adj_matrix)
        else:
            self.graph_attn = None

        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

    def forward(self, z, p):
        # z: (B, in_channels, T), p: (B, 141)
        h = self.stem(z)
        for block in self.blocks:
            h = block(h, p)
        if self.graph_attn is not None:
            h = self.graph_attn(h)
        delta_y = self.head(h)
        return delta_y


# =========================================================================
# 5. 通用双阶基类 BioRefinerBase (支持独立子模块与复合大模型)
# =========================================================================
class BioRefinerBase(nn.Module):
    """
    Base class for two-stage coarse-to-fine neural operators.
    """
    def __init__(self, num_vars=39, width=128, modes=32, param_dim=141, n_blocks=4,
                 coarse_n_blocks=4, use_curvature=False, use_pyramid=False,
                 use_graph_attn=False, adj_matrix=None):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.use_curvature = use_curvature
        self.use_pyramid = use_pyramid
        self.use_graph_attn = use_graph_attn

        # Stage 1: Coarse BioMech Pathway Base Operator
        self.stage1_coarse = BioMechPathwayOnly(
            num_vars=num_vars,
            width=width,
            modes=modes,
            param_dim=param_dim,
            n_blocks=coarse_n_blocks
        )

        # Compute Stage 2 input channel count:
        # Base: Y0 (39) + dY0/dt (39) + X_in (78) = 156
        # If use_curvature: + d^2Y0/dt^2 (39) = 195
        in_channels = num_vars * 2 + num_vars * 2 + (num_vars if use_curvature else 0)

        # Stage 2: Stiff Phase Residual Refiner
        self.stage2_refiner = ModularStiffRefiner(
            in_channels=in_channels,
            num_vars=num_vars,
            width=width,
            param_dim=param_dim,
            n_blocks=n_blocks,
            use_pyramid=use_pyramid,
            use_graph_attn=use_graph_attn,
            adj_matrix=adj_matrix
        )

        # Learnable gating scale (initialized to 0.1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def _compute_1st_derivative(self, y):
        d = torch.zeros_like(y)
        d[:, :, 1:-1] = (y[:, :, 2:] - y[:, :, :-2]) * 0.5
        d[:, :, 0] = y[:, :, 1] - y[:, :, 0]
        d[:, :, -1] = y[:, :, -1] - y[:, :, -2]
        return d

    def _compute_2nd_derivative(self, y):
        d2 = torch.zeros_like(y)
        d2[:, :, 1:-1] = y[:, :, 2:] - 2.0 * y[:, :, 1:-1] + y[:, :, :-2]
        d2[:, :, 0] = d2[:, :, 1]
        d2[:, :, -1] = d2[:, :, -2]
        return d2

    def forward(self, x, p, return_coarse=False):
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T)  # (B, 78, T)
        else:
            B, V, T = x.shape
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, time_grid], dim=1)  # (B, 78, T)

        # 1. Stage 1 Coarse Trajectory Prediction
        y_coarse = self.stage1_coarse(x, p)  # (B, 39, T)

        # 2. Extract derivative and curvature tensors
        dy_coarse = self._compute_1st_derivative(y_coarse)  # (B, 39, T)

        if self.use_curvature:
            d2y_coarse = self._compute_2nd_derivative(y_coarse)  # (B, 39, T)
            z_refine = torch.cat([y_coarse, dy_coarse, d2y_coarse, x_in], dim=1)  # (B, 195, T)
        else:
            z_refine = torch.cat([y_coarse, dy_coarse, x_in], dim=1)  # (B, 156, T)

        # 3. Stage 2 Stiff Residual Correction
        delta_y = self.stage2_refiner(z_refine, p)  # (B, 39, T)

        # 4. Gated Residual Composition
        y_final = y_coarse + torch.tanh(self.gamma) * delta_y

        if return_coarse:
            return y_final, y_coarse
        return y_final


# =========================================================================
# 6. 四大派生消融与旗舰模型（按独立子方向严格解耦）
# =========================================================================
class BioRefinerCurvature(BioRefinerBase):
    """消融子模型 1: 仅开启二阶相空间曲率/加速度特征 (Curvature Only)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_curvature=True, use_pyramid=False, use_graph_attn=False, **kwargs)


class BioRefinerPyramid(BioRefinerBase):
    """消融子模型 2: 仅开启多尺度时间金字塔残差网络 (Temporal Pyramid Only)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_curvature=False, use_pyramid=True, use_graph_attn=False, **kwargs)


class BioRefinerGraph(BioRefinerBase):
    """消融子模型 3: 仅开启生化反应拓扑图导向的跨变量注意力 (Graph Attention Only)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_curvature=False, use_pyramid=False, use_graph_attn=True, **kwargs)


class BioRefinerPlus(BioRefinerBase):
    """终极大融合旗舰: 三大子方向全开 (Curvature + Pyramid + Graph Attention)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_curvature=True, use_pyramid=True, use_graph_attn=True, **kwargs)
