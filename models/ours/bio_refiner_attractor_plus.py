#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioRefinerAttractorPlus: 集大成旗舰级双吸引子因果神经算子
=============================================================================
融合“双吸引子乘法熔断 (Mixture-of-Attractors Gating)”与“54调控边因果可达性容量 (Causal DAG Reachability)”，
辅以多尺度时间金字塔与化学计量拓扑图注意力，全面攻克非线性动力学相变突变体。

核心物理架构:
1. 吸引子分岔判别器 (Attractor Bifurcation Classifier):
   从 141 维动力学参数 p 评估系统分岔状态 P(Pattern A/B/C)。
   - P_arrest (Pattern C 概率): 处于致死停滞吸引子。
   - P_osc = 1 - P_arrest: 处于增殖振荡极限环吸引子。

2. 调控因果可达性容量 (Causal DAG Reachability Gate):
   沿 54 条生物学调控边推导各变量的活化容量 A_i(p) in [0, 1]，对振荡流进行上游因果限制。

3. 双流生成与乘法结合律熔断:
   - 稳态平原流 Y_plat(t): 指数衰减至固定点平衡点 Y*(p)，数学上杜绝任何傅里叶频域波纹。
   - 振荡极限环流 Y_osc(t): 粗尺度生化通路算子 + 时间金字塔 + 化学计量注意力，受 A_i(p) 包络约束。
   - 终态融合: Y(t) = P_osc * Y_osc(t) + P_arrest * Y_plat(t)。
     在死锁状态下，振荡流被严格乘以 0.0，杜绝残差穿透。
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bio_mechanistic_operator import BioMechPathwayOnly
from .bio_refiner_plus import TemporalPyramidBlock, StoichiometricGraphAttention
from .bio_refiner_attractor import EquilibriumRelaxationStream
from .bio_refiner_causal_reach import CausalReachabilityGate
from .input_contracts import extract_initial_state


class BioRefinerAttractorPlus(nn.Module):
    """
    BioRefinerAttractorPlus 旗舰神经算子
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

        # =====================================================================
        # 1. 吸引子分岔判别器
        # =====================================================================
        self.pattern_head = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3)
        )

        # =====================================================================
        # 2. 因果调控图可达性容量门控
        # =====================================================================
        self.reachability_gate = CausalReachabilityGate(
            num_vars=num_vars, param_dim=param_dim, d_gene=16, n_layers=2
        )

        # =====================================================================
        # 3. 稳态平原生成流
        # =====================================================================
        self.plateau_stream = EquilibriumRelaxationStream(
            num_vars=num_vars, param_dim=param_dim, hidden_dim=64
        )

        # =====================================================================
        # 4. 振荡极限环流 (粗阶段 + 细化金字塔 + 化学计量图注意力)
        # =====================================================================
        self.stage1_coarse = BioMechPathwayOnly(
            num_vars=num_vars,
            width=width,
            modes=modes,
            param_dim=param_dim,
            n_blocks=coarse_n_blocks
        )

        in_channels = num_vars * 3 + num_vars * 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU()
        )
        self.pyramid_blocks = nn.ModuleList([
            TemporalPyramidBlock(width, param_dim=param_dim)
            for _ in range(n_blocks)
        ])
        self.graph_attn = StoichiometricGraphAttention(
            channels=width, num_vars=num_vars, adj_matrix=adj_matrix
        )
        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def set_parameter_stats(self, p_mean=None, p_std=None, norm_type="log1p"):
        self.reachability_gate.set_parameter_stats(p_mean, p_std, norm_type)

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
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T)
        else:
            B, V, T = x.shape
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, time_grid], dim=1)

        # 1. 评估吸引子分岔概率与因果可达性容量
        pattern_logits = self.pattern_head(p)  # (B, 3)
        pattern_probs = F.softmax(pattern_logits, dim=-1)  # (B, 3)
        p_arrest = pattern_probs[:, 2:3].unsqueeze(-1)     # (B, 1, 1)
        p_oscillatory = 1.0 - p_arrest
        capacity = self.reachability_gate(p)               # (B, 39, 1)

        # 2. 稳态平原流
        y_plat = self.plateau_stream(x, p)  # (B, 39, T)

        # 3. 振荡极限环流 (受容量调制)
        y_coarse = self.stage1_coarse(x, p)  # (B, 39, T)
        dy_coarse = self._compute_1st_derivative(y_coarse)
        d2y_coarse = self._compute_2nd_derivative(y_coarse)
        z_refine = torch.cat([y_coarse, dy_coarse, d2y_coarse, x_in], dim=1)

        h = self.stem(z_refine)
        for block in self.pyramid_blocks:
            h = block(h, p)
        h = self.graph_attn(h)
        delta_y = self.head(h)

        x_0 = extract_initial_state(x, self.num_vars)
        y_osc_unconstrained = y_coarse + torch.tanh(self.gamma) * delta_y
        y_osc = x_0 + capacity * (y_osc_unconstrained - x_0)

        # 4. 严格乘法双吸引子熔断
        y_final = p_oscillatory * y_osc + p_arrest * y_plat

        if return_pattern:
            if return_coarse:
                return y_final, y_coarse, pattern_logits
            return y_final, pattern_logits

        if return_coarse:
            return y_final, y_coarse
        return y_final
