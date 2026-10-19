#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioRefinerDynamicPlus: 旗舰动态生物神经网络算子 (Dynamic Mechanistic Neural Operator)
========================================================================================
全面修复并重构前代架构中的结构性缺陷，集大成的新一代旗舰神经算子:
  [1] 物理质量动态特征通道 (Dynamic Mass Event Channel):
      通过解析物理指数生长方程 mass(t) = m_0 * exp((ln2 / mdt) * t) 构建单调过阈特征，
      作为物理特征通道无缝拼入 Stage 2，彻底激活细胞大小穿越机制。
  [2] 机制感知拓扑时变适配器 (TimeVaryingDynamicAdapter):
      保留 27 个功能模块与 54 条生物反应调控拓扑，将硬掩码替换为软结构偏置 (Structural Prior Bias)，
      在保持强反应物理引导的同时，解除信息饥饿，支持多跳生化级联通信。
  [3] 双向全时程时域金字塔细化器 (Bi-directional Time-Varying Stiff Refiner):
      将 Stage 2 局部卷积由单侧因果截断恢复为对称双向多尺度视野，
      消除人工群延迟与相位后移，采用 GroupNorm 杜绝非稳态推理时的统计量偏移。
  [4] 残差稳态自然合成 (Natural Residual Synthesis):
      彻底剥离未监督且导致退化的 state_latch 自锁门控，还给网络完整的瞬态动态波峰表达能力，
      根本性解决 #15 等停滞突变体初期脉冲被削峰抹平的致命缺陷。

Author: Antigravity AI
Date: 2026-09-21
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import json
from hashlib import sha256
from .graph_contracts import attach_fixed_graph
import torch.nn as nn
import torch.nn.functional as F
from .input_contracts import ParameterInputTransform

from .bio_refiner_plus import StoichiometricGraphAttention, TemporalPyramidBlock
from .bio_refiner_causal import (
    PARAM_KEYS_141,
    WT_VALUES_141,
    MODULE_NAMES,
    MODULE_PARAM_INDICES,
    MODULE_REGULATORY_EDGES,
    CausalRMSNorm,
)


# ==============================================================================
# 1. 物理质量动态事件特征通道 (Dynamic Mass Event Channel)
# ==============================================================================

class DynamicMassEventTrigger(nn.Module):
    """
    物理质量连续动态事件触发器:
    酵母细胞质量遵循指数生长方程: d(MASS)/dt = (ln2 / mdt) * MASS
    通过初值 mass_0 和环境倍增时间 mdt，解析构建真实单调递增的质量轨迹 MASS(t)，
    计算穿越临界阈值 theta(p) 的光滑连续时相门控。
    """
    def __init__(self, width: int = 192, param_dim: int = 141, d_event: int = 24):
        super().__init__()
        self.width = width
        self.d_event = d_event

        # 从动力学参数映射事件阈值与陡峭度
        self.threshold_net = nn.Sequential(
            nn.Linear(param_dim, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 2)  # [theta_offset, tau_raw]
        )
        self.event_proj = nn.Sequential(
            nn.Conv1d(2, d_event, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_event, d_event, kernel_size=1)
        )

    def forward(self, x_in: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        x_in: (B, 78, T)
        p: (B, 141) 参数矢量 (包含 mdt 在 index 140)
        Returns:
            event_emb: (B, d_event, T)
        """
        B = p.shape[0]
        T = x_in.shape[-1]

        # 提取初始质量 mass_0 (变量 0 对应 MASS)
        if x_in.dim() == 4:
            m0 = x_in[:, 0, 0, 0:1]  # (B, 1)
        else:
            m0 = x_in[:, 0, 0:1]     # (B, 1)

        # 提取倍增时间 mdt (默认 WT 约 90min, 范围 [30, 300])
        mdt = p[:, 140:141].clamp(min=30.0, max=300.0)  # (B, 1)

        # 物理时间网格 t: [0, 210] min
        t_physical = torch.linspace(0.0, 210.0, T, device=p.device, dtype=p.dtype).unsqueeze(0).expand(B, -1)  # (B, T)

        # 解析物理质量轨迹: MASS(t) = m0 * exp( (ln2 / mdt) * t )
        growth_rate = math.log(2.0) / mdt  # (B, 1)
        mass_t = m0 * torch.exp(growth_rate * t_physical)  # (B, T)

        # 临界阈值与敏感度
        theta_params = self.threshold_net(p)
        theta = theta_params[:, 0:1] + 1.0  # (B, 1)
        tau = F.softplus(theta_params[:, 1:2]) + 0.05  # (B, 1)

        diff = (mass_t - theta) / tau  # (B, T)
        event_mask = torch.sigmoid(diff)  # (B, T)

        t_norm = torch.linspace(0.0, 1.0, T, device=p.device, dtype=p.dtype).unsqueeze(0).expand(B, -1)
        event_input = torch.stack([event_mask, t_norm], dim=1)  # (B, 2, T)
        event_emb = self.event_proj(event_input)  # (B, d_event, T)
        return event_emb


# ==============================================================================
# 2. 软拓扑图注意力层 (Soft Mechanistic Graph Topology GAT)
# ==============================================================================

class SoftTopologyGAT(nn.Module):
    """
    机制感知软拓扑图注意力:
    在 27 个功能模块间进行消息传递。
    将硬掩码替换为软结构偏置 (Structural Prior Bias)，在赋予直连反应更强先验权重的
    同时，保留非直接连接节点之间的注意力通道，允许跨多跳生物分子级联信号传递。
    """
    def __init__(self, num_nodes: int = 27, d_gene: int = 16, num_heads: int = 4):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_gene = d_gene
        self.num_heads = num_heads
        assert d_gene % num_heads == 0
        self.head_dim = d_gene // num_heads

        self.q_proj = nn.Linear(d_gene, d_gene, bias=False)
        self.k_proj = nn.Linear(d_gene, d_gene, bias=False)
        self.v_proj = nn.Linear(d_gene, d_gene, bias=False)
        self.out_proj = nn.Linear(d_gene, d_gene, bias=False)

        self.sign_bias_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.norm = CausalRMSNorm(d_gene)

    def forward(
        self,
        e_time: torch.Tensor,
        adj_mask: torch.Tensor,
        sign_adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        e_time: (B, num_nodes, d_gene, T_sub)
        adj_mask: (num_nodes, num_nodes)
        sign_adj: (num_nodes, num_nodes)
        """
        B, N, D, T = e_time.shape
        e_flat = e_time.permute(0, 3, 1, 2).reshape(B * T, N, D)

        H = self.num_heads
        d_k = self.head_dim

        Q = self.q_proj(e_flat).view(B * T, N, H, d_k).permute(0, 2, 1, 3)
        K = self.k_proj(e_flat).view(B * T, N, H, d_k).permute(0, 2, 1, 3)
        V = self.v_proj(e_flat).view(B * T, N, H, d_k).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)  # (B*T, H, N, N)

        # 软拓扑结构偏置: 自环加上正规化邻接矩阵
        diag_eye = torch.eye(N, device=adj_mask.device, dtype=adj_mask.dtype)
        safe_adj = torch.clamp(adj_mask + diag_eye, 0.0, 1.0)
        adj_norm = safe_adj / (safe_adj.sum(dim=-1, keepdim=True) + 1e-8)

        scores = scores + 0.5 * adj_norm.unsqueeze(0).unsqueeze(0)
        if sign_adj is not None:
            scores = scores + self.sign_bias_weight * sign_adj.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, V)  # (B*T, H, N, d_k)
        out = out.permute(0, 2, 1, 3).contiguous().view(B * T, N, D)
        out = self.out_proj(out)

        out_res = self.norm(e_flat + out)
        out_time = out_res.view(B, T, N, D).permute(0, 2, 3, 1)
        return out_time


# ==============================================================================
# 3. 时变动态调控适配器 (TimeVaryingDynamicAdapter)
# ==============================================================================

class TimeVaryingDynamicAdapter(nn.Module):
    """
    时变动态调控张量生成器:
    利用 Stage 1 粗轨迹推断各时相生理活跃度，结合生化图网络为 141 维动力学参数
    生成时变修正张量 p_dyn(t) in R^{B x 141 x T}。
    """
    def __init__(
        self,
        param_dim: int = 141,
        d_gene: int = 16,
        num_heads: int = 4,
        t_stride: int = 10,
        t_steps: int = 500
    ):
        super().__init__()
        self.param_dim = param_dim
        self.d_gene = d_gene
        self.t_stride = t_stride
        self.t_steps = t_steps
        self.num_nodes = len(MODULE_NAMES)

        self.register_buffer('p_wt', torch.tensor(WT_VALUES_141, dtype=torch.float32))
        self.parameter_transform = ParameterInputTransform(param_dim)

        node_to_idx = {name: i for i, name in enumerate(MODULE_NAMES)}
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        sign = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for src, dst, s in MODULE_REGULATORY_EDGES:
            if src in node_to_idx and dst in node_to_idx:
                u, v = node_to_idx[src], node_to_idx[dst]
                adj[v, u] = 1.0  # row receives, column sends
                sign[v, u] = 1.0 if s == '+' else -1.0
        self.register_buffer('adj_mask', torch.from_numpy(adj))
        self.register_buffer('sign_adj', torch.from_numpy(sign))
        digest = sha256(json.dumps((MODULE_NAMES, MODULE_REGULATORY_EDGES, 'target_source'),
                                  separators=(',', ':')).encode()).hexdigest()
        # This is a manually defined module prior, not an ODE state graph.
        metadata = torch.tensor([1, 2, 0, *bytes.fromhex(digest)], dtype=torch.uint8)
        attach_fixed_graph(self, 'module', metadata, ('adj_mask', 'sign_adj'))

        # 基因级静态突变嵌入投影
        self.module_projections = nn.ModuleList()
        for gene_name in MODULE_NAMES:
            indices = MODULE_PARAM_INDICES.get(gene_name, [])
            in_dim = max(len(indices) * 2, 1)
            self.module_projections.append(
                nn.Sequential(
                    nn.Linear(in_dim, d_gene),
                    nn.GELU(),
                    nn.Linear(d_gene, d_gene)
                )
            )

        # 动态状态节点特征投影
        self.state_node_proj = nn.Sequential(
            nn.Conv1d(self.num_nodes, self.num_nodes * d_gene, kernel_size=1, groups=self.num_nodes),
            nn.GELU()
        )
        self.state_norm = CausalRMSNorm(d_gene)

        # 软拓扑图注意力更新
        self.time_gat = SoftTopologyGAT(num_nodes=self.num_nodes, d_gene=d_gene, num_heads=num_heads)

        # 解码至 141 维时变参数偏移
        self.param_decoders = nn.ModuleList()
        for gene_name in MODULE_NAMES:
            indices = MODULE_PARAM_INDICES.get(gene_name, [])
            out_dim = len(indices)
            if out_dim > 0:
                self.param_decoders.append(
                    nn.Sequential(
                        nn.Linear(d_gene, d_gene),
                        nn.GELU(),
                        nn.Linear(d_gene, out_dim)
                    )
                )
            else:
                self.param_decoders.append(nn.Identity())

        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _build_var_to_gene(self, device, dtype) -> torch.Tensor:
        M = torch.zeros((self.num_nodes, 39), device=device, dtype=dtype)
        gene_to_vars = {
            'CLN2': [1], 'CLB2': [2], 'CLB5': [3], 'SIC1': [4, 10], 'CDC6': [5, 13],
            'SWI5': [16, 17], 'CDC20': [19, 20], 'CDH1': [21, 22], 'CDC14': [23, 24],
            'NET1': [25, 26], 'TEM1': [28], 'CDC15': [29], 'PPX': [30], 'PDS1': [31],
            'ESP1': [32], 'MAD2': [35, 36], 'BUB2': [38], 'SBF_MBF': [1, 3], 'Mcm1': [2, 17],
            'APC_Core': [18, 20, 22], 'ORI_BUD_SPN': [33, 34, 35], 'MDT': [0], 'CDC5_Proxy': [2],
            'CDC28': [1, 2, 3], 'SCC1': [32, 35],
        }
        for g_idx, g_name in enumerate(MODULE_NAMES):
            if g_name in gene_to_vars:
                for v_idx in gene_to_vars[g_name]:
                    M[g_idx, v_idx] = 1.0 / len(gene_to_vars[g_name])
        return M

    def set_parameter_stats(self, p_mean=None, p_std=None, norm_type="log1p"):
        self.parameter_transform.set_parameter_stats(p_mean, p_std, norm_type)

    def forward(self, p: torch.Tensor, y_coarse: torch.Tensor) -> torch.Tensor:
        B, num_vars, T = y_coarse.shape
        diff, log_ratio = self.parameter_transform.intervention_features(
            p, self.p_wt, eps=1e-6, logarithm="ratio")

        gene_embeddings = []
        for i, gene_name in enumerate(MODULE_NAMES):
            indices = MODULE_PARAM_INDICES.get(gene_name, [])
            proj = self.module_projections[i]
            if len(indices) > 0:
                feat = torch.cat([diff[:, indices], log_ratio[:, indices]], dim=-1)
                e_g = proj(feat)
            else:
                e_g = torch.zeros((B, self.d_gene), device=p.device, dtype=p.dtype)
            gene_embeddings.append(e_g)

        e_mut_static = torch.stack(gene_embeddings, dim=1)  # (B, 27, d_gene)

        # 时间子采样提取生理活跃度
        y_sub = y_coarse[:, :, ::self.t_stride]  # (B, 39, T_sub)
        T_sub = y_sub.shape[-1]

        M = self._build_var_to_gene(device=p.device, dtype=p.dtype)  # (27, 39)
        z_gene = torch.einsum('gm,bmt->bgt', M, y_sub)  # (B, 27, T_sub)

        z_dyn = self.state_node_proj(z_gene)  # (B, 27 * d_gene, T_sub)
        z_dyn = z_dyn.view(B, self.num_nodes, self.d_gene, T_sub)
        z_dyn = z_dyn.permute(0, 1, 3, 2)  # (B, 27, T_sub, d_gene)
        z_dyn = self.state_norm(z_dyn).permute(0, 1, 3, 2)  # (B, 27, d_gene, T_sub)

        e_time_in = e_mut_static.unsqueeze(-1) + z_dyn
        e_time_updated = self.time_gat(e_time_in, self.adj_mask, self.sign_adj)

        delta_p_sub_list = [torch.zeros((B, 1, T_sub), device=p.device, dtype=p.dtype) for _ in range(self.param_dim)]
        for i, gene_name in enumerate(MODULE_NAMES):
            indices = MODULE_PARAM_INDICES.get(gene_name, [])
            dec = self.param_decoders[i]
            if len(indices) > 0:
                h_node = e_time_updated[:, i, :, :].permute(0, 2, 1)  # (B, T_sub, d_gene)
                d_p = dec(h_node).permute(0, 2, 1)  # (B, len(indices), T_sub)
                for idx_sub, param_idx in enumerate(indices):
                    delta_p_sub_list[param_idx] = d_p[:, idx_sub:idx_sub+1, :]

        delta_p_t = torch.cat(delta_p_sub_list, dim=1)  # (B, 141, T_sub)
        delta_p_full = F.interpolate(delta_p_t, size=T, mode='linear', align_corners=True)

        p_dynamic_t = p.unsqueeze(-1) + torch.tanh(self.alpha) * delta_p_full
        return p_dynamic_t


# ==============================================================================
# 4. 双向对称时变残差块 (Bi-directional Time-Varying ResConvBlock1D)
# ==============================================================================

class TimeVaryingResConvBlock1D(nn.Module):
    """
    双向时变 FiLM 卷积块:
    采用对称填充 padding = dilation，消除单向因果截断带来的累积群延迟与相位错位；
    采用无状态的 GroupNorm(1, channels) 消除测试阶段由于分布偏离造成的 BatchNorm 漂移。
    """
    def __init__(self, channels: int, param_dim: int = 141, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.film1 = nn.Linear(param_dim, channels * 2)
        self.film2 = nn.Linear(param_dim, channels * 2)

    def forward(self, x: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        p_t: (B, param_dim, T) 或 (B, param_dim)
        """
        res = x

        # FiLM 调制 1
        if p_t.dim() == 3:
            p_t_perm = p_t.permute(0, 2, 1)  # (B, T, param_dim)
            gamma1, beta1 = self.film1(p_t_perm).permute(0, 2, 1).chunk(2, dim=1)  # (B, C, T)
        else:
            gamma1, beta1 = self.film1(p_t).unsqueeze(-1).chunk(2, dim=1)

        out = self.conv1(x)
        out = self.norm1(out)
        out = (1.0 + gamma1) * out + beta1
        out = F.gelu(out)

        # FiLM 调制 2
        if p_t.dim() == 3:
            gamma2, beta2 = self.film2(p_t_perm).permute(0, 2, 1).chunk(2, dim=1)
        else:
            gamma2, beta2 = self.film2(p_t).unsqueeze(-1).chunk(2, dim=1)

        out = self.conv2(out)
        out = self.norm2(out)
        out = (1.0 + gamma2) * out + beta2
        out = F.gelu(out + res)
        return out


# ==============================================================================
# 5. Stage 2 双向刚性残差细化器 (TimeVaryingStiffRefiner)
# ==============================================================================

class TimeVaryingStiffRefiner(nn.Module):
    """
    Stage 2 双向刚性时变细化器:
    利用全时程对称双向感受野与多尺度金字塔对 Stage 1 粗轨迹进行高保真修光。
    """
    def __init__(
        self,
        in_channels: int,
        num_vars: int = 39,
        width: int = 192,
        param_dim: int = 141,
        n_blocks: int = 4,
        use_pyramid: bool = True,
        use_graph_attn: bool = True,
        adj_matrix: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.use_pyramid = use_pyramid
        self.use_graph_attn = use_graph_attn

        self.stem = nn.Conv1d(in_channels, width, kernel_size=1)

        self.blocks = nn.ModuleList([
            TimeVaryingResConvBlock1D(width, param_dim, dilation=2**i)
            for i in range(n_blocks)
        ])

        if use_pyramid:
            self.pyramid = TemporalPyramidBlock(width, param_dim=param_dim)

        if use_graph_attn:
            self.graph_attn = StoichiometricGraphAttention(
                channels=width,
                num_vars=num_vars,
                adj_matrix=adj_matrix
            )

        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width // 2, num_vars, kernel_size=1)
        )

    def forward(self, z: torch.Tensor, p_dynamic_t: torch.Tensor) -> torch.Tensor:
        h = self.stem(z)

        for block in self.blocks:
            h = block(h, p_dynamic_t)

        if self.use_pyramid:
            p_static = p_dynamic_t.mean(dim=-1) if p_dynamic_t.dim() == 3 else p_dynamic_t
            h = self.pyramid(h, p_static)

        if self.use_graph_attn:
            h = self.graph_attn(h)

        delta_y = self.head(h)
        return delta_y


# ==============================================================================
# 6. 旗舰完整模型: BioRefinerDynamicPlus
# ==============================================================================

class BioRefinerDynamicPlus(nn.Module):
    """
    旗舰完整模型 (BioRefinerDynamicPlus):
    1. 动态物理质量事件特征通道 (DynamicMassEventTrigger) 真实注入 Stage 2 计算流；
    2. 机制感知时变调控适配器 (TimeVaryingDynamicAdapter) 基于软拓扑图偏置提供时相动力学特征；
    3. Stage 1: 生化路径增强算子 (BioMechPathwayOnly);
    4. Stage 2: 双向对称全局残差细化器 (TimeVaryingStiffRefiner)；
    5. 残差稳态自然合成: 彻底剥离退化的 state_latch，保留完整的生物学瞬态脉冲与稳态平衡。
    """
    def __init__(
        self,
        num_vars: int = 39,
        width: int = 192,
        modes: int = 24,
        param_dim: int = 141,
        n_blocks: int = 4,
        coarse_n_blocks: int = 4,
        d_gene: int = 16,
        adj_matrix: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.param_dim = param_dim

        # 1. 物理动态质量事件特征通道 (提取 24 维物理事件特征)
        d_event = max(width // 8, 16)
        self.d_event = d_event
        self.mass_event_trigger = DynamicMassEventTrigger(
            width=width,
            param_dim=param_dim,
            d_event=d_event
        )

        # 2. 时变动态调控适配器 (软拓扑偏置)
        self.dynamic_adapter = TimeVaryingDynamicAdapter(
            param_dim=param_dim,
            d_gene=d_gene,
            num_heads=4,
            t_stride=10
        )

        # 3. Stage 1 基础骨干: 增强粗生成
        from .bio_mechanistic_operator import BioMechPathwayOnly
        self.stage1_coarse = BioMechPathwayOnly(
            num_vars=num_vars,
            width=width,
            modes=modes,
            param_dim=param_dim,
            n_blocks=coarse_n_blocks
        )

        # 4. Stage 2 双向刚性残差细化器
        # 输入维度: y_coarse(39) + dy(39) + d2y(39) + x_in(78) + event_emb(d_event)
        in_channels = num_vars * 5 + d_event
        self.stage2_refiner = TimeVaryingStiffRefiner(
            in_channels=in_channels,
            num_vars=num_vars,
            width=width,
            param_dim=param_dim,
            n_blocks=n_blocks,
            use_pyramid=True,
            use_graph_attn=True,
            adj_matrix=adj_matrix
        )

        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def set_parameter_stats(self, p_mean=None, p_std=None, norm_type="log1p"):
        self.dynamic_adapter.set_parameter_stats(p_mean, p_std, norm_type)

    def _compute_1st_derivative(self, y: torch.Tensor) -> torch.Tensor:
        dy = torch.zeros_like(y)
        dy[:, :, 1:-1] = (y[:, :, 2:] - y[:, :, :-2]) * 0.5
        dy[:, :, 0] = y[:, :, 1] - y[:, :, 0]
        dy[:, :, -1] = y[:, :, -1] - y[:, :, -2]
        return dy

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
        return_coarse: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        端到端全时程前向推理
        """
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T)
        else:
            B, V, T = x.shape
            t_grid = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, t_grid], dim=1)

        # 1. 物理动态事件特征提取
        event_emb = self.mass_event_trigger(x_in, p)  # (B, d_event, T)

        # 2. Stage 1 粗生成
        y_coarse = self.stage1_coarse(x, p)  # (B, 39, T)

        # 3. 提取因果导数与二阶导数 (曲率)
        dy_coarse = self._compute_1st_derivative(y_coarse)
        d2y_coarse = self._compute_2nd_derivative(y_coarse)

        # 4. 拼接 Stage 2 输入张量 (物理事件通道真实连接到计算图中)
        z_refine = torch.cat([y_coarse, dy_coarse, d2y_coarse, x_in, event_emb], dim=1)

        # 5. 机制感知时变调控张量
        p_dynamic_t = self.dynamic_adapter(p, y_coarse)

        # 6. Stage 2 双向全局细化器
        delta_y = self.stage2_refiner(z_refine, p_dynamic_t)

        # 7. 残差自然合成 (彻底剔除常数卷积平滑 state_latch)
        y_final = y_coarse + torch.tanh(self.gamma) * delta_y

        if return_coarse:
            return y_final, y_coarse
        return y_final
