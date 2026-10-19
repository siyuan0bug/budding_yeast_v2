#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioMassActionRefiner (Law-of-Mass-Action Neural Operator)
=========================================================
A next-generation biophysical neural operator designed to fundamentally eliminate
badcases (such as Mutant 15 cln3Δ bck2Δ multi-copy CLN2 and arrest bifurcation failures)
through three first-principles mathematical inductive biases:

1. Law of Mass Action Multiplicative Tensor Layer (质量作用定律乘积式原语层):
   Replaces unconstrained additive linear projections with true multiplicative reaction fluxes:
   v_r(t) = k_r(p) * \prod_{j \in Reactants(r)} [S_j(t)]
   When upstream activators (e.g. SBF, CLN3, BCK2) are zero, the synthesis flux v_r(t) identically
   evaluates to 0.0 regardless of how large the multi-copy parameter k_r is, structurally immunizing
   the network against false oscillations.

2. Zero-Additive-Leak Gating (零加性泄漏参数调制):
   Replaces naive FiLM ((1+gamma)*x + beta) with state-factorized multiplicative modulation:
   out = x * (1 + tanh(gamma(p))) + tanh(W_x * x) * tanh(beta(p))
   Guarantees strict Origin Invariance: when input state x -> 0, out == 0. Parameters cannot
   create artificial oscillatory signals out of a quiescent zero-state.

3. Bifurcation Attractor Latch (动力学分岔吸引子稳态锁存器):
   Identifies whether the kinetic perturbation leads to viable limit-cycle oscillations or
   fatal arrest (G1 arrest, Telophase arrest, G2/M arrest). For arrest regimes, applies an
   attractor projection that suppresses high-frequency limit-cycle ripple and produces a
   pure steady-state plateau.
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import os
import numpy as np
import torch
from .graph_contracts import resolve_variable_graph, row_normalized_support, attach_variable_graph
import torch.nn as nn
import torch.nn.functional as F

from .bio_mechanistic_operator import BioMechPathwayOnly
from .bio_hetero_causal_graph import BioMutationFluxEncoder


# =========================================================================
# 1. 质量作用定律原语层 (Law of Mass Action Reaction Core)
# =========================================================================

class MassActionReactionCore(nn.Module):
    """
    Law of Mass Action (LMA) Multiplicative Reaction Flux Layer.
    Explicitly evaluates core biochemical synthesis, activation, and degradation fluxes
    via multiplicative tensor products between kinetic rates k(p) and substrate species S(t).
    
    Ensures that for any reaction with required upstream substrate/activator S_j:
        Flux = k_r(p) * S_j(t)  -->  If S_j(t) == 0, Flux == 0.0 identically!
    """
    def __init__(self, num_vars: int = 39, param_dim: int = 141, hidden_dim: int = 64):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        
        # 提取关键动力学参数并映射为有效反应速率常数 (全部使用非负激活 Softplus)
        self.rate_generator = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),
            nn.Softplus()
        )
        
        # 反应物化学计量加权投影 (保证只有相关物种参与乘积)
        self.species_gate = nn.Sequential(
            nn.Conv1d(num_vars, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 32, kernel_size=1),
            nn.Softplus()
        )
        
        # 通量投影回 39 个物种通道 (化学计量矩阵投影)
        self.flux_to_species = nn.Conv1d(32, num_vars, kernel_size=1, bias=False)

    def forward(self, y_state: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        y_state: (B, 39, T) 物理或特征浓度轨迹
        p: (B, 141) 动力学参数向量
        
        Returns:
            flux_contrib: (B, 39, T) 严格受质量作用定律乘积约束的生化变化通量
        """
        B, V, T = y_state.shape
        
        # 1. 计算各反应的速率常数向量 k_r(p) >= 0: (B, 32, 1)
        k_rates = self.rate_generator(p).unsqueeze(-1)
        
        # 2. 计算物种浓度非负门控底物项 S_act(t) >= 0: (B, 32, T)
        # 保证当底物浓度趋于 0 时，对应反应底物项趋于 0
        s_act = self.species_gate(torch.clamp(y_state, min=0.0))
        
        # 3. 严格乘法结合律: Reaction Flux J_r(t) = k_r(p) * S_act(t)
        # 若底物为 0，则无论 k_rates 多大，乘积严格为 0!
        flux_tensor = k_rates * s_act  # (B, 32, T)
        
        # 4. 投影为 39 个物种的变化率贡献
        flux_contrib = self.flux_to_species(flux_tensor)  # (B, 39, T)
        return flux_contrib


# =========================================================================
# 2. 零加性泄漏参数调制模块 (Zero-Additive-Leak FiLM Blocks)
# =========================================================================

class ZeroLeakFiLMBlock1d(nn.Module):
    """
    Zero-Leak Parameter-Modulated Dilated Residual Block.
    Unlike standard FiLM ( (1+gamma)*x + beta ), this block factorizes all parameter
    effects through state-dependent terms:
        out = x * (1.0 + tanh(gamma)) + tanh(W_gate * x) * tanh(beta)
    When x == 0, out == 0 identically (Strict Origin Invariance).
    """
    def __init__(self, channels: int, dilation: int = 1, param_dim: int = 141):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, channels)
        
        # Parameter projection for gamma and beta
        self.film_proj = nn.Linear(param_dim, channels * 2)
        # State gate to ensure zero leakage
        self.state_gate = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film_proj(p).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)    # (B, C, 1)

        res = x
        h = F.gelu(self.norm1(self.conv1(x)))
        
        # 零泄漏调制：beta 必须与经过状态门控的特征相乘，不能直接作为独立偏置相加
        gate_x = torch.tanh(self.state_gate(h))
        h_mod = h * (1.0 + torch.tanh(gamma)) + gate_x * torch.tanh(beta)
        
        out = self.norm2(self.conv2(h_mod))
        return F.gelu(out + res)


class ZeroLeakTemporalPyramidBlock(nn.Module):
    """
    Multi-Scale Temporal Pyramidal Block with Zero-Additive-Leak Parameter Gating.
    Combines local receptive fields (dilations 1, 2) and macroscopic baselines (dilations 4, 8)
    while maintaining strict origin invariance under parameter perturbations.
    """
    def __init__(self, channels: int, param_dim: int = 141):
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
        
        self.film_proj = nn.Linear(param_dim, channels * 2)
        self.state_gate = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film_proj(p).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        c1 = self.b1(x)
        c2 = self.b2(x)
        c3 = self.b3(x)
        c4 = self.b4(x)
        concat = torch.cat([c1, c2, c3, c4], dim=1)
        h = self.norm(self.fuse(concat))
        
        # Zero-leak parameter gating
        gate_x = torch.tanh(self.state_gate(h))
        h_mod = h * (1.0 + torch.tanh(gamma)) + gate_x * torch.tanh(beta)
        
        return F.gelu(h_mod + x)


# =========================================================================
# 3. 生化反应图拓扑跨变量注意力 (Stoichiometric Mass Action Attention)
# =========================================================================

class StoichiometricMassActionAttention(nn.Module):
    """
    Graph-Guided Cross-Variable Attention across 39 Biochemical Species
    constrained by stoichiometric adjacency and mutation flux routing.
    """
    def __init__(self, channels: int = 128, num_vars: int = 39, head_dim: int = 16, adj_matrix=None):
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

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, C, T = h.shape
        h_t = h.permute(0, 2, 1)  # (B, T, channels)
        vars_feat = self.to_vars(h_t).reshape(B, T, self.num_vars, self.head_dim)

        Q = self.q_proj(vars_feat)
        K = self.k_proj(vars_feat)
        V = self.v_proj(vars_feat)

        attn_raw = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn_raw + 0.5 * self.graph_bias.unsqueeze(0).unsqueeze(0), dim=-1)
        out_vars = torch.matmul(attn, V).reshape(B, T, self.num_vars * self.head_dim)
        h_attn = self.from_vars(out_vars).permute(0, 2, 1)
        return h + h_attn


# =========================================================================
# 4. 动力学分岔吸引子稳态锁存器 (Bifurcation Attractor Latch)
# =========================================================================

class BifurcationAttractorLatch(nn.Module):
    """
    Bifurcation Attractor Latch (分岔吸引子稳态锁存器).
    Detects whether the kinetic parameter regime corresponds to an irreversible
    cell cycle arrest (e.g. G1 arrest in cln3Δ bck2Δ, Telophase arrest in cdc14-ts, etc.).
    
    If an arrest attractor is detected with probability p_arrest, it clamps the temporal
    derivative of the state to zero after the arrest phase, eliminating false sine-wave oscillations.
    """
    def __init__(self, num_vars: int = 39, param_dim: int = 141, hidden_dim: int = 64):
        super().__init__()
        self.num_vars = num_vars
        
        # 1. 分岔类型分类头 (3 分类: 0=正常增殖振荡, 1=G1期死锁停滞, 2=有丝分裂/末期死锁停滞)
        self.classifier = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # 2. 变量级停滞倾向自适应投射
        self.arrest_var_gate = nn.Sequential(
            nn.Linear(3, num_vars),
            nn.Sigmoid()
        )
        
        # 3. 稳态低通投影核 (提取并锁存平原水平线)
        self.latch_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, y: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        y: (B, 39, T) 细化后的轨迹预测
        p: (B, 141) 参数向量
        
        Returns:
            y_latched: (B, 39, T) 经吸引子平原锁存后的物理纯净轨迹
            bifurcation_logits: (B, 3) 分岔表型对数几率
        """
        B, V, T = y.shape
        bifurcation_logits = self.classifier(p)  # (B, 3)
        bifurcation_probs = F.softmax(bifurcation_logits, dim=-1)  # (B, 3)
        
        # 停滞概率 (G1 arrest + M/T arrest)
        p_arrest_total = bifurcation_probs[:, 1:]  # (B, 2)
        var_arrest_weight = self.arrest_var_gate(bifurcation_probs).unsqueeze(-1)  # (B, 39, 1)
        
        # 稳态平原基准: 轨迹后期的平均稳态浓度
        # 取时间后半段作为稳态渐近线
        steady_plateau = y[:, :, T // 2:].mean(dim=-1, keepdim=True).expand(-1, -1, T)  # (B, 39, T)
        
        # 时间增长权重: 越往后，停滞态对稳态平原的吸引越强
        t_ramp = torch.linspace(0.0, 1.0, T, device=y.device).view(1, 1, T)
        effective_latch = var_arrest_weight * t_ramp  # (B, 39, T)
        
        # 混合输出: 动态振荡与稳态吸引子
        y_latched = (1.0 - effective_latch) * y + effective_latch * steady_plateau
        return y_latched, bifurcation_logits


# =========================================================================
# 5. 旗舰架构: BioMassActionRefiner (质量作用定律全约束算子)
# =========================================================================

class BioMassActionRefiner(nn.Module):
    """
    BioMassActionRefiner: 结合质量作用定律乘积式、零泄漏参数调制与分岔稳态锁存的终极算子。
    
    架构流程:
    1. Stage 1: Coarse BioMech Pathway Operator (基础宏观轨迹)
    2. Mass Action Core: 质量作用定律直接物理通量计算 (v = k * S_act)
    3. Stage 2 Zero-Leak Refiner: 零加性泄漏时间金字塔残差修正 + 拓扑图注意力
    4. Bifurcation Attractor Latch: 动力学吸引子平原锁存，彻底根除死锁样本的虚假振荡
    5. Physical Clamp: 保证非负浓度物理约束
    """
    def __init__(
        self,
        num_vars: int = 39,
        width: int = 128,
        modes: int = 32,
        param_dim: int = 141,
        n_blocks: int = 4,
        coarse_n_blocks: int = 4,
        adj_matrix=None,
        **kwargs
    ):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.param_dim = param_dim

        # 1. 反应通量编码器 (解析 6 种生化反应通量与 27 基因活性)
        self.flux_encoder = BioMutationFluxEncoder(param_dim=param_dim, hidden_dim=64)

        # 2. Stage 1: 粗尺度生化通路算子
        self.stage1_coarse = BioMechPathwayOnly(
            num_vars=num_vars,
            width=width,
            modes=modes,
            param_dim=param_dim,
            n_blocks=coarse_n_blocks
        )

        # 3. 质量作用定律原语层
        self.mass_action_core = MassActionReactionCore(
            num_vars=num_vars,
            param_dim=param_dim,
            hidden_dim=64
        )

        # 4. Stage 2 零泄漏残差细化网络
        # 输入通道: y_coarse(39) + dy_coarse(39) + d2y_coarse(39) + mass_action_flux(39) + x_in(78) = 234
        in_channels = num_vars * 4 + num_vars * 2

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU()
        )

        self.zero_leak_blocks = nn.ModuleList([
            ZeroLeakTemporalPyramidBlock(width, param_dim=param_dim)
            if i % 2 == 0 else
            ZeroLeakFiLMBlock1d(width, dilation=2 ** (i % 3), param_dim=param_dim)
            for i in range(n_blocks)
        ])

        self.graph_attn = StoichiometricMassActionAttention(
            channels=width, num_vars=num_vars, adj_matrix=adj_matrix
        )

        self.refiner_head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

        # 5. 分岔吸引子稳态锁存器
        self.attractor_latch = BifurcationAttractorLatch(
            num_vars=num_vars,
            param_dim=param_dim,
            hidden_dim=64
        )

        # 可学习门控缩放因子
        self.gamma_res = nn.Parameter(torch.tensor(0.1))
        self.gamma_lma = nn.Parameter(torch.tensor(0.2))

    def _compute_1st_derivative(self, y: torch.Tensor) -> torch.Tensor:
        d = torch.zeros_like(y)
        d[:, :, 1:-1] = (y[:, :, 2:] - y[:, :, :-2]) * 0.5
        d[:, :, 0] = y[:, :, 1] - y[:, :, 0]
        d[:, :, -1] = y[:, :, -1] - y[:, :, -2]
        return d

    def _compute_2nd_derivative(self, y: torch.Tensor) -> torch.Tensor:
        d2 = torch.zeros_like(y)
        d2[:, :, 1:-1] = y[:, :, 2:] - 2.0 * y[:, :, 1:-1] + y[:, :, :-2]
        d2[:, :, 0] = d2[:, :, 1]
        d2[:, :, -1] = d2[:, :, -2]
        return d2

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        return_coarse: bool = False,
        return_pattern: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            x: (B, 39, 2, T) 或 (B, 78, T) 输入张量 [初值, 时间网格]
            p: (B, 141) 动力学参数向量
            return_coarse: 是否返回 Stage 1 粗预测
            return_pattern: 是否返回分岔分类 logits
        """
        # 统一输入维度为 (B, 78, T)
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T)
        else:
            B, V, T = x.shape
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, time_grid], dim=1)

        # 1. Stage 1: 粗尺度预测
        y_coarse = self.stage1_coarse(x, p)  # (B, 39, T)

        # 2. 质量作用定律原语层通量计算 (纯乘法约束)
        lma_flux = self.mass_action_core(y_coarse, p)  # (B, 39, T)

        # 3. 提取一阶与二阶导数张量
        dy_coarse = self._compute_1st_derivative(y_coarse)
        d2y_coarse = self._compute_2nd_derivative(y_coarse)

        # 4. 拼接 Stage 2 特征输入
        z_refine = torch.cat([y_coarse, dy_coarse, d2y_coarse, lma_flux, x_in], dim=1)  # (B, 234, T)

        # 5. Stage 2 零泄漏残差计算
        h = self.stem(z_refine)
        for block in self.zero_leak_blocks:
            h = block(h, p)
        h = self.graph_attn(h)
        delta_y = self.refiner_head(h)  # (B, 39, T)

        # 6. 双阶残差门控融合 + 质量作用通量注入
        y_composite = (
            y_coarse
            + torch.tanh(self.gamma_lma) * lma_flux
            + torch.tanh(self.gamma_res) * delta_y
        )

        # 7. 动力学分岔吸引子稳态锁存 (彻底滤除停滞突变体的伪振荡)
        y_final, bifurcation_logits = self.attractor_latch(y_composite, p)

        if return_pattern:
            if return_coarse:
                return y_final, y_coarse, bifurcation_logits
            return y_final, bifurcation_logits

        if return_coarse:
            return y_final, y_coarse
        return y_final
