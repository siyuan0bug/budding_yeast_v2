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
from .bio_hetero_causal_graph import BioMutationFluxEncoder
from .graph_contracts import resolve_variable_graph, row_normalized_support, attach_variable_graph
from .graph_contracts import fixed_graph_is_legacy
from .relation_graph_contracts import register_relation_graph


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
    Injects unsigned ODE/event potential support as a soft attention affinity.
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

        graph = resolve_variable_graph(adj_matrix, num_vars)
        self.register_buffer('graph_bias', row_normalized_support(graph))
        attach_variable_graph(self, graph, ('graph_bias',))

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


class DynamicReactionFluxGraphAttention(StoichiometricGraphAttention):
    """
    Reaction Flux Guided Cross-Variable Attention across 39 Biochemical Species.
    Dynamically weights six ODE-term relation overlays and an untyped route.
    These learned modulation weights are not measured physical reaction fluxes.
    """
    def __init__(self, channels=128, num_vars=39, head_dim=16, adj_matrix=None):
        super().__init__(channels=channels, num_vars=num_vars, head_dim=head_dim, adj_matrix=adj_matrix)
        register_relation_graph(self, num_vars, variable_graph=adj_matrix)

    def forward(self, h, flux_w=None):
        if flux_w is None:
            return super().forward(h)
        return self.forward_flux(h, flux_w)

    def forward_flux(self, h, flux_w):
        B, C, T = h.shape
        h_t = h.permute(0, 2, 1)  # (B, T, channels)
        vars_feat = self.to_vars(h_t).reshape(B, T, self.num_vars, self.head_dim)  # (B, T, 39, 16)

        Q = self.q_proj(vars_feat)  # (B, T, 39, 16)
        K = self.k_proj(vars_feat)  # (B, T, 39, 16)
        V = self.v_proj(vars_feat)  # (B, T, 39, 16)

        attn_raw = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, T, 39, 39)

        # flux_w: (B, 6), rel_graphs: (6, 39, 39)
        dynamic_adj = torch.einsum('br,rvw->bvw', flux_w, self.rel_graphs.to(h.device))  # (B, 39, 39)
        dynamic_adj = dynamic_adj + flux_w.mean(-1)[:, None, None] * self.untyped_graph
        eye = torch.eye(self.num_vars, device=h.device, dtype=dynamic_adj.dtype).unsqueeze(0)
        # Preserve the historical +I convention only in explicit replay mode.
        dynamic_adj = dynamic_adj + eye if fixed_graph_is_legacy(self, 'relations') else dynamic_adj * (1 - eye) + eye
        dynamic_norm = dynamic_adj / dynamic_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)

        attn = F.softmax(attn_raw + 0.5 * dynamic_norm.unsqueeze(1), dim=-1)  # (B, T, 39, 39)
        out_vars = torch.matmul(attn, V).reshape(B, T, self.num_vars * self.head_dim)
        h_attn = self.from_vars(out_vars).permute(0, 2, 1)
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

    def forward(self, z, p, flux_w=None):
        # z: (B, in_channels, T), p: (B, 141)
        h = self.stem(z)
        for block in self.blocks:
            h = block(h, p)
        if self.graph_attn is not None:
            if flux_w is not None and hasattr(self.graph_attn, 'forward_flux'):
                h = self.graph_attn.forward_flux(h, flux_w)
            else:
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


# =========================================================================
# 7. 机制与分岔增强新架构 (方案 A, 方案 B, 方案 A+B)
# =========================================================================
class BioRefinerBifurcate(BioRefinerBase):
    """
    方案 A: 两阶段分岔解耦模型 (表型分类前置 + 分岔表征注入)
    - 增加轻量级 MLP 分类头 pattern_head，从 141 维动力学参数预测 Pattern A/B/C 分岔分类
    - 将分岔概率表征投影注入 Stage 2 Refiner，防止阻滞态产生恶性伪振荡
    - 支持返回 pattern_logits 供多任务表型交叉熵联合训练
    """
    def __init__(self, *args, num_vars=39, width=128, modes=32, param_dim=141, n_blocks=4, **kwargs):
        super().__init__(
            *args, num_vars=num_vars, width=width, modes=modes, param_dim=param_dim,
            n_blocks=n_blocks, use_curvature=True, use_pyramid=True, use_graph_attn=True, **kwargs
        )
        self.pattern_head = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3)
        )
        self.pattern_proj = nn.Linear(3, param_dim)

    def forward(self, x, p, return_coarse=False, return_pattern=False):
        pattern_logits = self.pattern_head(p)  # (B, 3)
        pattern_probs = F.softmax(pattern_logits, dim=-1)
        p_refined = p + self.pattern_proj(pattern_probs)

        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T)
        else:
            B, V, T = x.shape
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, time_grid], dim=1)

        y_coarse = self.stage1_coarse(x, p)
        dy_coarse = self._compute_1st_derivative(y_coarse)

        if self.use_curvature:
            d2y_coarse = self._compute_2nd_derivative(y_coarse)
            z_refine = torch.cat([y_coarse, dy_coarse, d2y_coarse, x_in], dim=1)
        else:
            z_refine = torch.cat([y_coarse, dy_coarse, x_in], dim=1)

        delta_y = self.stage2_refiner(z_refine, p_refined)
        y_final = y_coarse + torch.tanh(self.gamma) * delta_y

        if return_pattern:
            if return_coarse:
                return y_final, y_coarse, pattern_logits
            return y_final, pattern_logits

        if return_coarse:
            return y_final, y_coarse
        return y_final


class BioRefinerGraphFlux(BioRefinerBase):
    """
    方案 B: 机制感知架构 (生化图路由替代全局扁平 FiLM)
    - 引入 BioMutationFluxEncoder 将 141 维动力学参数编码为 6 种反应通量权重与 27 个基因活性
    - 结合 27x39 基因二部图将扰动直接投影至各生化分子通道
    - 采用 DynamicReactionFluxGraphAttention 实现通量自适应局部生化消息路由
    """
    def __init__(self, *args, num_vars=39, width=128, modes=32, param_dim=141, n_blocks=4, **kwargs):
        super().__init__(
            *args, num_vars=num_vars, width=width, modes=modes, param_dim=param_dim,
            n_blocks=n_blocks, use_curvature=True, use_pyramid=True, use_graph_attn=True, **kwargs
        )
        self.flux_encoder = BioMutationFluxEncoder(param_dim=param_dim, hidden_dim=64)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        npz_path = os.path.join(current_dir, "../../data/yeast_multirelational_graphs.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            gene_bipartite = torch.from_numpy(data['gene_bipartite']).float()  # (27, 39)
        else:
            gene_bipartite = torch.zeros(27, 39)
        self.register_buffer('gene_bipartite', gene_bipartite)

        self.stage2_refiner.graph_attn = DynamicReactionFluxGraphAttention(
            channels=width, num_vars=num_vars, adj_matrix=kwargs.get('adj_matrix')
        )
        self.flux_proj = nn.Sequential(
            nn.Linear(6 + 39, 64),
            nn.GELU(),
            nn.Linear(64, param_dim)
        )

    def forward(self, x, p, return_coarse=False):
        flux_w, gene_act = self.flux_encoder(p)
        species_flux = torch.matmul(gene_act, self.gene_bipartite.to(p.device))  # (B, 39)
        mech_feat = torch.cat([flux_w, species_flux], dim=-1)
        p_mech = p + self.flux_proj(mech_feat)

        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T)
        else:
            B, V, T = x.shape
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, time_grid], dim=1)

        y_coarse = self.stage1_coarse(x, p_mech)
        dy_coarse = self._compute_1st_derivative(y_coarse)

        if self.use_curvature:
            d2y_coarse = self._compute_2nd_derivative(y_coarse)
            z_refine = torch.cat([y_coarse, dy_coarse, d2y_coarse, x_in], dim=1)
        else:
            z_refine = torch.cat([y_coarse, dy_coarse, x_in], dim=1)

        delta_y = self.stage2_refiner(z_refine, p_mech, flux_w=flux_w)
        y_final = y_coarse + torch.tanh(self.gamma) * delta_y

        if return_coarse:
            return y_final, y_coarse
        return y_final


class BioRefinerMechanistic(BioRefinerBase):
    """
    方案 A + B 全融合机制旗舰模型:
    1. 分岔先验: 轻量级 Pattern 分类头预测 A/B/C 表型，分岔概率调制 Stage 2 参数
    2. 反应通量路由: BioMutationFluxEncoder 提取 6-relation flux + 27-gene activity + 39-species perturbation
    3. 动态拓扑注意力: DynamicReactionFluxGraphAttention 根据突变通量动态缩放生化图连接
    4. 二阶相空间曲率 + 多尺度时间金字塔 + 双阶粗精细化门控
    """
    def __init__(self, *args, num_vars=39, width=128, modes=32, param_dim=141, n_blocks=4, **kwargs):
        super().__init__(
            *args, num_vars=num_vars, width=width, modes=modes, param_dim=param_dim,
            n_blocks=n_blocks, use_curvature=True, use_pyramid=True, use_graph_attn=True, **kwargs
        )
        self.pattern_head = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3)
        )
        self.pattern_proj = nn.Linear(3, param_dim)

        self.flux_encoder = BioMutationFluxEncoder(param_dim=param_dim, hidden_dim=64)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        npz_path = os.path.join(current_dir, "../../data/yeast_multirelational_graphs.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            gene_bipartite = torch.from_numpy(data['gene_bipartite']).float()
        else:
            gene_bipartite = torch.zeros(27, 39)
        self.register_buffer('gene_bipartite', gene_bipartite)

        self.stage2_refiner.graph_attn = DynamicReactionFluxGraphAttention(
            channels=width, num_vars=num_vars, adj_matrix=kwargs.get('adj_matrix')
        )
        self.flux_proj = nn.Sequential(
            nn.Linear(6 + 39, 64),
            nn.GELU(),
            nn.Linear(64, param_dim)
        )

    def forward(self, x, p, return_coarse=False, return_pattern=False):
        pattern_logits = self.pattern_head(p)
        pattern_probs = F.softmax(pattern_logits, dim=-1)

        flux_w, gene_act = self.flux_encoder(p)
        species_flux = torch.matmul(gene_act, self.gene_bipartite.to(p.device))
        mech_feat = torch.cat([flux_w, species_flux], dim=-1)

        p_refined = p + self.pattern_proj(pattern_probs) + self.flux_proj(mech_feat)

        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T)
        else:
            B, V, T = x.shape
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, time_grid], dim=1)

        y_coarse = self.stage1_coarse(x, p)
        dy_coarse = self._compute_1st_derivative(y_coarse)

        if self.use_curvature:
            d2y_coarse = self._compute_2nd_derivative(y_coarse)
            z_refine = torch.cat([y_coarse, dy_coarse, d2y_coarse, x_in], dim=1)
        else:
            z_refine = torch.cat([y_coarse, dy_coarse, x_in], dim=1)

        delta_y = self.stage2_refiner(z_refine, p_refined, flux_w=flux_w)
        y_final = y_coarse + torch.tanh(self.gamma) * delta_y

        if return_pattern:
            if return_coarse:
                return y_final, y_coarse, pattern_logits
            return y_final, pattern_logits

        if return_coarse:
            return y_final, y_coarse
        return y_final
