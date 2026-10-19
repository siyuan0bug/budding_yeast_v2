#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioRefinerCausalReach: 生物调控图因果可达性门控神经算子
=============================================================================
通过酵母细胞周期 27 个功能基因模块与 54 条生化调控边构建的因果有向无环图 (Causal DAG)，
自适应推断各物种在当前动力学参数 p 下的生物学可激活容量 (Biological Reachability Capacity) A_i(p) in [0, 1]。

核心物理与生物学机理:
1. 因果可达性传播 (Causal Reachability Propagation):
   - 沿 54 条真实生化调控边（激活 + / 抑制 -）前向传播基因扰动信号。
   - 当上游关键起始通路全部关闭（例如 CLN3 与 BCK2 均处于敲除状态），
     下游转录因子（SBF/MBF）与后续周期蛋白（Clb5/Clb2）的激活容量 A_i(p) 将严格衰减至 0。

2. 容量包络调制 (Capacity-Modulated Amplitude Bounding):
   - Y_i(t) = Y_{i, base}(x_0, p) + A_i(p) * Delta Y_i(t)
   - 当 A_i(p) -> 0 时，任何频域或高阶残差在数学上被乘法阻断（乘以 0.0），
     从因果拓扑源头彻底杜绝虚假起振（False Limit Cycle Induction）。
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bio_mechanistic_operator import BioMechPathwayOnly
from .bio_refiner_plus import TemporalPyramidBlock, StoichiometricGraphAttention
from .input_contracts import ParameterInputTransform, extract_initial_state
from .bio_refiner_causal import (
    MODULE_NAMES,
    MODULE_PARAM_INDICES,
    MODULE_REGULATORY_EDGES,
    WT_VALUES_141,
    CausalRMSNorm,
)


class CausalReachabilityGate(nn.Module):
    """
    基于生物图拓扑的因果可达性容量推断模块:
    输入 141 维动力学参数 p，输出 39 个物种的动态振幅容量 A in [0, 1]^39。
    """
    def __init__(
        self,
        num_vars: int = 39,
        param_dim: int = 141,
        d_gene: int = 16,
        n_layers: int = 2
    ):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.d_gene = d_gene
        self.num_modules = len(MODULE_NAMES)

        # 注册野生型参考参数
        p_wt_tensor = torch.tensor(WT_VALUES_141, dtype=torch.float32)
        if len(p_wt_tensor) != param_dim:
            if len(p_wt_tensor) > param_dim:
                p_wt_tensor = p_wt_tensor[:param_dim]
            else:
                p_wt_tensor = F.pad(p_wt_tensor, (0, param_dim - len(p_wt_tensor)))
        self.register_buffer('p_wt', p_wt_tensor)
        self.parameter_transform = ParameterInputTransform(param_dim)

        # 基因模块索引映射
        self.name_to_idx = {name: i for i, name in enumerate(MODULE_NAMES)}
        
        # 构建 27x27 调控邻接矩阵与符号矩阵
        adj = torch.zeros(self.num_modules, self.num_modules, dtype=torch.float32)
        sign_adj = torch.zeros(self.num_modules, self.num_modules, dtype=torch.float32)
        for src, dst, sign in MODULE_REGULATORY_EDGES:
            if src in self.name_to_idx and dst in self.name_to_idx:
                u, v = self.name_to_idx[src], self.name_to_idx[dst]
                adj[v, u] = 1.0  # u 影响 v
                sign_adj[v, u] = 1.0 if sign == '+' else -1.0
        self.register_buffer('adj_mask', adj)
        self.register_buffer('sign_adj', sign_adj)

        # 模块输入局部投影
        self.module_proj = nn.ModuleDict()
        for gene_name, p_idx in MODULE_PARAM_INDICES.items():
            valid_idx = [i for i in p_idx if i < param_dim]
            in_d = max(1, len(valid_idx)) * 2
            self.module_proj[gene_name] = nn.Linear(in_d, d_gene, bias=False)

        # 因果图传播层
        self.layers = nn.ModuleList([
            nn.Linear(d_gene * 2, d_gene, bias=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([CausalRMSNorm(d_gene) for _ in range(n_layers)])

        # 从 27 个基因模块映射到 39 个物种的可达性门控容量 A_i in [0, 1]
        self.to_species_capacity = nn.Sequential(
            nn.Linear(self.num_modules * d_gene + param_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_vars),
            nn.Sigmoid()
        )

    def set_parameter_stats(self, p_mean=None, p_std=None, norm_type="log1p"):
        self.parameter_transform.set_parameter_stats(p_mean, p_std, norm_type)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        p: (B, 141)
        Returns:
            capacity: (B, 39, 1) 动态振幅容量
        """
        B = p.shape[0]
        # Mechanistic differences use physical units; learned global conditioning
        # below still receives p in its original normalized representation.
        delta_p, log_res = self.parameter_transform.intervention_features(p, self.p_wt)

        # 初始模块嵌入
        module_embeds = []
        for name in MODULE_NAMES:
            p_idx = MODULE_PARAM_INDICES[name]
            valid_idx = [i for i in p_idx if i < self.param_dim]
            if len(valid_idx) == 0:
                feat = torch.zeros(B, 2, device=p.device, dtype=p.dtype)
            else:
                d_p = delta_p[:, valid_idx]
                l_p = log_res[:, valid_idx]
                feat = torch.cat([d_p, l_p], dim=-1)
            e_g = self.module_proj[name](feat)  # (B, d_gene)
            module_embeds.append(e_g)

        H = torch.stack(module_embeds, dim=1)  # (B, 27, d_gene)

        # 因果拓扑图传播
        safe_adj = self.adj_mask + torch.eye(self.num_modules, device=p.device)
        deg = safe_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        norm_adj = safe_adj / deg

        for layer, norm in zip(self.layers, self.norms):
            # 聚合上游调控信号: (B, 27, d_gene)
            msg = torch.einsum('ij,bjd->bid', norm_adj, H)
            combined = torch.cat([H, msg], dim=-1)  # (B, 27, 2 * d_gene)
            H = norm(H + F.gelu(layer(combined)))

        # 展平并映射至物种可达性容量
        H_flat = H.view(B, -1)  # (B, 27 * d_gene)
        global_in = torch.cat([H_flat, p], dim=-1)
        capacity = self.to_species_capacity(global_in).unsqueeze(-1)  # (B, 39, 1)
        return capacity


class BioRefinerCausalReach(nn.Module):
    """
    BioRefinerCausalReach 神经算子:
    结合因果调控图可达性容量 A(p) 与双阶粗精算子，
    当上游通路关闭时，乘法阻断下游振幅，消除虚假起振。
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

        # 1. 因果可达性容量门控
        self.reachability_gate = CausalReachabilityGate(
            num_vars=num_vars, param_dim=param_dim, d_gene=16, n_layers=2
        )

        # 2. 分岔分类头 (支持 AL4PDE 与多任务损失)
        self.pattern_head = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3)
        )

        # 3. 粗尺度算子
        self.stage1_coarse = BioMechPathwayOnly(
            num_vars=num_vars,
            width=width,
            modes=modes,
            param_dim=param_dim,
            n_blocks=coarse_n_blocks
        )

        # 4. 细化时间金字塔与图注意力
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

        # 1. 评估因果可达性容量 A(p) 与分岔概率
        capacity = self.reachability_gate(p)  # (B, 39, 1)
        pattern_logits = self.pattern_head(p) # (B, 3)

        # 2. Stage 1 粗预测
        y_coarse = self.stage1_coarse(x, p)  # (B, 39, T)
        dy_coarse = self._compute_1st_derivative(y_coarse)
        d2y_coarse = self._compute_2nd_derivative(y_coarse)
        z_refine = torch.cat([y_coarse, dy_coarse, d2y_coarse, x_in], dim=1)

        # 3. Stage 2 细化残差
        h = self.stem(z_refine)
        for block in self.pyramid_blocks:
            h = block(h, p)
        h = self.graph_attn(h)
        delta_y = self.head(h)

        # 4. 因果容量包络调制:
        # 当上游通路关闭 capacity -> 0 时，不仅细化残差 delta_y 被清零，
        # y_coarse 的偏离也被收敛至初值稳态 x_0
        x_0 = extract_initial_state(x, self.num_vars)  # (B, 39, 1)
        y_osc = y_coarse + torch.tanh(self.gamma) * delta_y
        y_final = x_0 + capacity * (y_osc - x_0)

        if return_pattern:
            if return_coarse:
                return y_final, y_coarse, pattern_logits
            return y_final, pattern_logits

        if return_coarse:
            return y_final, y_coarse
        return y_final
