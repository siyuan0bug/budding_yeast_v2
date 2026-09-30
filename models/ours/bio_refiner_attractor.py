#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioRefinerAttractor: 双吸引子解耦神经算子 (Mixture-of-Attractors Neural Operator)
=============================================================================
彻底解决非线性分岔拓扑奇点与加法残差穿透缺陷的下一代架构:

核心物理机理:
1. 吸引子分岔判别器 (Attractor Bifurcation Classifier):
   从 141 维动力学参数 p 直接预测系统所处的相空间吸引子类别:
   - Pattern A (0): 正常极限环振荡 (Limit Cycle)
   - Pattern B (1): 缓慢/单次分裂极限环
   - Pattern C (2): 致死死锁停滞吸引子 (Fixed-Point Arrest)

2. 双生成流解耦 (Dual Generative Streams):
   - 振荡流 Y_osc(t): 专注于周期性极限环动力学的高频拟合 (多尺度时间金字塔 + 拓扑图注意力)
   - 平原流 Y_plat(t): 专注于死锁稳态的连续物理松弛 (无高频波纹, 从初值平滑弛豫至稳态平衡点)

3. 严格乘法结合律关断 (Multiplicative Attractor Gating):
   Y(t) = (1 - P_arrest(p)) * Y_osc(t) + P_arrest(p) * Y_plat(t)
   当样本落入死锁吸引子 (P_arrest -> 1.0) 时，振荡分支被绝对零置 (乘以 0.0),
   彻底斩断类似 Mutant 15 在 t=90 处因局部多拷贝参数激发的虚假振荡！
"""

from typing import Optional, Tuple, Union
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bio_mechanistic_operator import BioMechPathwayOnly
from .bio_refiner_plus import TemporalPyramidBlock, StoichiometricGraphAttention


class EquilibriumRelaxationStream(nn.Module):
    """
    稳态平原松弛流 (Plateau Stream):
    针对致死停滞突变体 (Pattern C)，生成从初始浓度 x_0 平滑演化至固定点平衡点 Y* 的单调物理稳态，
    计算图中不包含任何傅里叶频域模式或循环震荡原语，数学上天生免疫正弦波纹。
    """
    def __init__(self, num_vars: int = 39, param_dim: int = 141, hidden_dim: int = 64):
        super().__init__()
        self.num_vars = num_vars
        
        # 稳态渐近平衡点预测网络 Y*(p)
        self.equilibrium_net = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_vars)
        )
        
        # 特征松弛速率常数 lambda(p) > 0
        self.rate_net = nn.Sequential(
            nn.Linear(param_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_vars),
            nn.Softplus()
        )

    def forward(self, x_in: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        x_in: (B, 78, T) 包含变量初值与时间网格
        p: (B, 141) 动力学参数
        Returns:
            y_plat: (B, 39, T) 纯净物理平原稳态
        """
        B = p.shape[0]
        T = x_in.shape[-1]
        
        # 提取初始浓度 x_0: (B, 39, 1)
        x_0 = x_in[:, :self.num_vars, 0:1]
        
        # 提取时间网格 t: [0, 1] 归一化网格
        t_grid = torch.linspace(0.0, 1.0, T, device=p.device, dtype=p.dtype).view(1, 1, T).expand(B, -1, -1)
        
        # 计算平衡点与松弛速率
        y_eq = self.equilibrium_net(p).unsqueeze(-1)  # (B, 39, 1)
        lam = self.rate_net(p).unsqueeze(-1) + 0.1     # (B, 39, 1)
        
        # 指数弛豫方程: Y(t) = Y* + (x_0 - Y*) * exp(-lambda * t)
        decay = torch.exp(-lam * t_grid * 5.0)  # (B, 39, T)
        y_plat = y_eq + (x_0 - y_eq) * decay
        return y_plat


class BioRefinerAttractor(nn.Module):
    """
    BioRefinerAttractor: 双吸引子解耦乘法熔断神经算子
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
        # 1. 吸引子分岔判别器 (Attractor Bifurcation Classifier)
        # =====================================================================
        self.pattern_head = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3)  # 输出 Pattern A (0), B (1), C (2) logits
        )

        # =====================================================================
        # 2. 稳态平原生成流 (Plateau Stream: 专司固定点吸引子)
        # =====================================================================
        self.plateau_stream = EquilibriumRelaxationStream(
            num_vars=num_vars, param_dim=param_dim, hidden_dim=64
        )

        # =====================================================================
        # 3. 周期极限环生成流 (Oscillatory Stream: 双阶粗精算子)
        # =====================================================================
        # Stage 1: 粗尺度生化通路算子
        self.stage1_coarse = BioMechPathwayOnly(
            num_vars=num_vars,
            width=width,
            modes=modes,
            param_dim=param_dim,
            n_blocks=coarse_n_blocks
        )

        # Stage 2: 时间金字塔 + 图注意力残差细化
        # 输入通道: y_coarse(39) + dy_coarse(39) + d2y_coarse(39) + x_in(78) = 195
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
        前向传播: 严格执行吸引子乘法熔断结合律
        """
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T)
        else:
            B, V, T = x.shape
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, time_grid], dim=1)

        # 1. 评估吸引子分岔概率
        pattern_logits = self.pattern_head(p)  # (B, 3)
        pattern_probs = F.softmax(pattern_logits, dim=-1)  # (B, 3)
        # P_arrest (Pattern C 概率): (B, 1, 1)
        p_arrest = pattern_probs[:, 2:3].unsqueeze(-1)
        p_oscillatory = 1.0 - p_arrest

        # 2. 生成稳态平原流 (无振荡纯净物理松弛)
        y_plat = self.plateau_stream(x_in, p)  # (B, 39, T)

        # 3. 生成振荡极限环流 (Stage 1 + Stage 2)
        y_coarse = self.stage1_coarse(x, p)  # (B, 39, T)
        dy_coarse = self._compute_1st_derivative(y_coarse)
        d2y_coarse = self._compute_2nd_derivative(y_coarse)
        z_refine = torch.cat([y_coarse, dy_coarse, d2y_coarse, x_in], dim=1)

        h = self.stem(z_refine)
        for block in self.pyramid_blocks:
            h = block(h, p)
        h = self.graph_attn(h)
        delta_y = self.head(h)
        y_osc = y_coarse + torch.tanh(self.gamma) * delta_y  # (B, 39, T)

        # 4. 严格乘法结合律关断熔断
        # 当 p_arrest -> 1.0 时, p_oscillatory -> 0.0, y_osc 被严格清零, 绝无残差穿透!
        y_final = p_oscillatory * y_osc + p_arrest * y_plat

        if return_pattern:
            if return_coarse:
                return y_final, y_coarse, pattern_logits
            return y_final, pattern_logits

        if return_coarse:
            return y_final, y_coarse
        return y_final
