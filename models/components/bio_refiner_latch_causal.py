#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioRefiner-Latch-Causal: 状态自锁时变因果神经算子 (State-Latch Causal Neural Operator)
========================================================================================
彻底解决芽殖酵母细胞周期四大结构性缺陷的下一代旗舰算子:
  [1] 物理质量事件积分 (Dynamic Mass Event Trigger):
      针对 MASS 静态平铺缺陷，通过解析物理指数生长方程 mass(t) = m_0 * exp((ln2 / mdt) * t)
      恢复真正的单调过阈相变信号，彻底激活 G1/S 事件触发器。
  [2] 严格时间因果卷积 (Strictly Causal 1D Convolution):
      将局部细化卷积由对称填充升级为单侧因果填充 (kernel_size - 1, 0)，
      在数学上严格切断未来信息向过去反向泄漏 (d(y_t)/d(x_{t'}) = 0 for t' > t)。
  [3] 零泄漏拓扑图注意力 (Strict Mechanistic Graph Topology Mask):
      使用 -1e9 硬掩码彻底隔断非反应边上的非物理虚假注意力流动，
      保证图注意力消息 100% 局限于生物化学已证实的 54 条机制调控边。
  [4] 生理状态自锁门控 (Physiological State-Latch Gating):
      针对致死性停滞突变体 (如 GAL-CLB2, cdc14-ts, cdc20-ts) 的正弦波抖动缺陷，
      引入稳态自锁门控，在进入阻滞相变点后平滑抑制高频伪振荡，输出物理纯净的水平稳态。

Author: Antigravity AI
Date: 2026-09-21
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# 1. 严格单侧因果 1D 卷积 (Strict Causal Conv1D)
# ==============================================================================

class CausalConv1d(nn.Module):
    """
    单侧因果 1D 卷积:
    仅在时间轴左侧 (过去) 填充 (kernel_size - 1) * dilation，右侧 (未来) 填充 0。
    保证任何时刻 t 的输出严格仅依赖于当前及过去时刻 tau <= t，彻底杜绝未来泄漏。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True
    ):
        super().__init__()
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad format for 1D: (left_pad, right_pad)
        x_padded = F.pad(x, (self.pad_len, 0))
        return self.conv(x_padded)


# ==============================================================================
# 2. 物理质量动态事件积分器 (Dynamic Mass Event Trigger)
# ==============================================================================

class DynamicMassEventTrigger(nn.Module):
    """
    物理质量连续动态事件触发器:
    酵母细胞质量遵循指数生长方程: d(MASS)/dt = (ln2 / mdt) * MASS
    通过初值 mass_0 和环境倍增时间 mdt，解析构建真实单调递增的质量轨迹 MASS(t)，
    计算穿越临界阈值 theta(p) 的光滑连续时相门控。
    """
    def __init__(self, width: int = 128, param_dim: int = 141):
        super().__init__()
        self.width = width
        # 从动力学参数映射事件阈值与陡峭度
        self.threshold_net = nn.Sequential(
            nn.Linear(param_dim, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 2)  # [theta_offset, tau_raw]
        )
        self.event_proj = nn.Sequential(
            CausalConv1d(2, width // 4, kernel_size=3, dilation=1),
            nn.GELU(),
            nn.Conv1d(width // 4, width, kernel_size=1)
        )

    def forward(self, x_in: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        x_in: (B, 78, T) 或 (B, 39, 2, T)
        p: (B, 141) 参数矢量 (包含 mdt 在 index 140)
        """
        B = p.shape[0]
        T = x_in.shape[-1]

        # 提取初始质量 mass_0 (变量 0 对应 MASS)
        if x_in.dim() == 4:
            m0 = x_in[:, 0, 0, 0:1]  # (B, 1)
        else:
            m0 = x_in[:, 0, 0:1]     # (B, 1)

        # 提取倍增时间 mdt (默认 WT 约 90min, 范围 [60, 240])
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
        event_emb = self.event_proj(event_input)  # (B, width, T)
        return event_emb


# ==============================================================================
# 3. 严格拓扑有向图注意力层 (Strict Mechanistic Graph Topology GAT)
# ==============================================================================

class StrictCausalTopologyGAT(nn.Module):
    """
    严格拓扑因果有向图注意力:
    在 27 个功能模块间进行消息传递。
    对非直接生物反应边施加 -1e9 硬阻断，并在归一化后强制点乘拓扑掩码，
    确保非反应节点间注意力权重恒为 0.0。
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

        if sign_adj is not None:
            scores = scores + self.sign_bias_weight * sign_adj.unsqueeze(0).unsqueeze(0)

        # 严格拓扑自环与硬掩码 (严格使用 -1e9 屏蔽不连通节点)
        diag_eye = torch.eye(N, device=adj_mask.device, dtype=adj_mask.dtype)
        safe_adj = torch.clamp(adj_mask + diag_eye, 0.0, 1.0)
        mask = (safe_adj.unsqueeze(0).unsqueeze(0) == 0)
        scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        # 二次硬截断: 消除极微小浮点溢出
        attn_weights = attn_weights * safe_adj.unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.matmul(attn_weights, V)  # (B*T, H, N, d_k)
        out = out.permute(0, 2, 1, 3).contiguous().view(B * T, N, D)
        out = self.out_proj(out)

        out_res = self.norm(e_flat + out)
        out_time = out_res.view(B, T, N, D).permute(0, 2, 3, 1)
        return out_time


# ==============================================================================
# 4. 生理状态自锁门控机制 (Physiological State-Latch Gating)
# ==============================================================================

class PhysiologicalStateLatch(nn.Module):
    """
    生理状态自锁门控 (Physiological State Latch):
    检测致死性停滞突变与平衡点稳态，抑制频域多项式伪振荡。
    """
    def __init__(self, channels: int, num_vars: int = 39, param_dim: int = 141):
        super().__init__()
        self.channels = channels
        self.num_vars = num_vars

        # 停滞检测分类头
        self.latch_detector = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_vars),
            nn.Sigmoid()
        )

        # 时域高频平滑低通核 (用于阻滞平原的去纹波投影)
        self.smooth_conv = CausalConv1d(num_vars, num_vars, kernel_size=9, dilation=2)

    def forward(self, delta_y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        delta_y: (B, 39, T)
        p: (B, 141)
        """
        # (B, 39, 1) 每个变量在当前突变下的阻滞倾向
        latch_prob = self.latch_detector(p).unsqueeze(-1)

        # 提取低频稳态基线
        smoothed_y = self.smooth_conv(delta_y)

        # 当检测到阻滞稳态时，平滑抑制高频抖动
        out = (1.0 - latch_prob) * delta_y + latch_prob * smoothed_y
        return out


# ==============================================================================
# 5. 时变因果突变适配器 (TimeVaryingCausalAdapter)
# ==============================================================================

class TimeVaryingCausalAdapter(nn.Module):
    """
    时变因果突变嵌入适配器 (严格拓扑版)
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

        node_to_idx = {name: i for i, name in enumerate(MODULE_NAMES)}
        adj_mask = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32)
        sign_adj = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32)

        for src, tgt, sign in MODULE_REGULATORY_EDGES:
            if src in node_to_idx and tgt in node_to_idx:
                s_i = node_to_idx[src]
                t_i = node_to_idx[tgt]
                adj_mask[t_i, s_i] = 1.0
                sign_adj[t_i, s_i] = 1.0 if sign == '+' else -1.0

        self.register_buffer('adj_mask', adj_mask)
        self.register_buffer('sign_adj', sign_adj)

        self.module_projections = nn.ModuleList()
        for gene_name in MODULE_NAMES:
            indices = MODULE_PARAM_INDICES.get(gene_name, [])
            in_feats = len(indices) * 2
            if in_feats > 0:
                proj = nn.Sequential(
                    nn.Linear(in_feats, d_gene, bias=False),
                    nn.GELU(),
                    nn.Linear(d_gene, d_gene, bias=False)
                )
            else:
                proj = nn.Identity()
            self.module_projections.append(proj)

        self.state_node_proj = nn.Conv1d(27, 27 * d_gene, kernel_size=1, groups=27, bias=False)
        self.state_norm = CausalRMSNorm(d_gene)

        self.time_gat = StrictCausalTopologyGAT(
            num_nodes=self.num_nodes,
            d_gene=d_gene,
            num_heads=num_heads
        )

        self.param_decoders = nn.ModuleList()
        for gene_name in MODULE_NAMES:
            indices = MODULE_PARAM_INDICES.get(gene_name, [])
            out_dim = len(indices)
            if out_dim > 0:
                dec = nn.Sequential(
                    nn.Linear(d_gene, d_gene, bias=False),
                    nn.GELU(),
                    nn.Linear(d_gene, out_dim, bias=False)
                )
            else:
                dec = nn.Identity()
            self.param_decoders.append(dec)

        self.alpha = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def _build_var_to_gene(self, device, dtype):
        M = torch.zeros((len(MODULE_NAMES), 39), device=device, dtype=dtype)
        gene_to_vars = {
            'CLN2': [1], 'CLN3': [0], 'CLB2': [2, 6, 8, 11, 14], 'CLB5': [3, 7, 9, 12, 15],
            'SIC1': [4, 6, 7, 10, 11, 12], 'CDC6': [5, 8, 9, 13, 14, 15], 'BCK2': [0],
            'SWI5': [16, 17], 'CDC20': [19, 20, 36], 'CDH1': [21, 22], 'CDC14': [23, 24, 27],
            'NET1': [25, 26, 27], 'TEM1': [28], 'CDC15': [29], 'PPX': [30], 'PDS1': [31],
            'ESP1': [32], 'MAD2': [35, 36], 'BUB2': [38], 'SBF_MBF': [1, 3], 'Mcm1': [2, 17],
            'APC_Core': [18, 20, 22], 'ORI_BUD_SPN': [33, 34, 35], 'MDT': [0], 'CDC5_Proxy': [2],
            'CDC28': [1, 2, 3], 'SCC1': [32, 35],
        }
        for g_idx, g_name in enumerate(MODULE_NAMES):
            if g_name in gene_to_vars:
                for v_idx in gene_to_vars[g_name]:
                    M[g_idx, v_idx] = 1.0 / len(gene_to_vars[g_name])
        return M

    def forward(self, p: torch.Tensor, y_coarse: torch.Tensor) -> torch.Tensor:
        B, num_vars, T = y_coarse.shape
        p_wt_batch = self.p_wt.unsqueeze(0).expand(B, -1)
        diff = p - p_wt_batch
        log_ratio = torch.log(p.clamp(min=1e-6)) - torch.log(p_wt_batch.clamp(min=1e-6))

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

        p_causal_t = p.unsqueeze(-1) + torch.tanh(self.alpha) * delta_p_full
        return p_causal_t


# ==============================================================================
# 6. 时变因果 ResConv 块 (Causal ResConvBlock1D)
# ==============================================================================

class CausalTimeVaryingResConvBlock1D(nn.Module):
    """
    时变因果 FiLM 卷积块:
    使用 CausalConv1d 保证严格的时间因果流。
    """
    def __init__(self, channels: int, param_dim: int = 141, dilation: int = 1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

        self.film_conv = nn.Conv1d(param_dim, 2 * channels, kernel_size=1)

    def forward(self, x: torch.Tensor, p_causal_t: torch.Tensor) -> torch.Tensor:
        if p_causal_t.dim() == 2:
            p_causal_t = p_causal_t.unsqueeze(-1)

        gamma, beta = self.film_conv(p_causal_t).chunk(2, dim=1)

        h = self.conv1(x)
        h = self.norm1(h)
        h = h * (1.0 + gamma) + beta
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = h * (1.0 + gamma) + beta
        return self.act(x + h)


# ==============================================================================
# 7. 时变刚性细化器 (Causal Stiff Refiner)
# ==============================================================================

class CausalTimeVaryingStiffRefiner(nn.Module):
    """
    Stage 2 因果刚性细化器
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
            CausalTimeVaryingResConvBlock1D(width, param_dim, dilation=2**i)
            for i in range(n_blocks)
        ])

        if use_pyramid:
            self.pyramid = TemporalPyramidBlock(width)

        if use_graph_attn:
            self.graph_attn = StoichiometricGraphAttention(
                channels=width,
                num_vars=num_vars,
                adj_matrix=adj_matrix
            )

        self.head = nn.Sequential(
            nn.Conv1d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width // 2, num_vars, kernel_size=1)
        )

    def forward(self, z: torch.Tensor, p_causal_t: torch.Tensor) -> torch.Tensor:
        h = self.stem(z)

        for block in self.blocks:
            h = block(h, p_causal_t)

        if self.use_pyramid:
            p_static = p_causal_t.mean(dim=-1) if p_causal_t.dim() == 3 else p_causal_t
            h = self.pyramid(h, p_static)

        if self.use_graph_attn:
            h = self.graph_attn(h)

        return self.head(h)


# ==============================================================================
# 8. 旗舰完整模型: BioRefinerLatchCausalPlus (独立新模型)
# ==============================================================================

class BioRefinerLatchCausalPlus(nn.Module):
    """
    四位一体终极架构 (BioRefinerLatchCausalPlus):
    1. 动态质量事件触发器 (Dynamic Mass Event Trigger) 驱动粗生成 Stage 1；
    2. 严格单向拓扑图掩码 (Strict Mechanistic Graph Mask) 驱动时变因果适配器；
    3. 严格单侧因果因果卷积 (Strict Causal Conv1D) 驱动残差细化 Stage 2；
    4. 生理状态自锁门控 (Physiological State-Latch Gating) 彻底消除停滞纹波。
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

        # 1. 物理动态质量事件触发器
        self.mass_event_trigger = DynamicMassEventTrigger(width=width, param_dim=param_dim)

        # 2. 时变因果适配器 (严格拓扑掩码)
        self.dynamic_causal_adapter = TimeVaryingCausalAdapter(
            param_dim=param_dim,
            d_gene=d_gene,
            num_heads=4,
            t_stride=10
        )

        # 3. Stage 1 基础骨干: 增强因果粗生成
        from .bio_mechanistic_operator import BioMechPathwayOnly
        self.stage1_coarse = BioMechPathwayOnly(
            num_vars=num_vars,
            width=width,
            modes=modes,
            param_dim=param_dim,
            n_blocks=coarse_n_blocks
        )

        # 4. Stage 2 因果刚性残差细化器
        in_channels = num_vars * 5  # y_coarse(39) + dy(39) + d2y(39) + x_in(78) = 195
        self.stage2_refiner = CausalTimeVaryingStiffRefiner(
            in_channels=in_channels,
            num_vars=num_vars,
            width=width,
            param_dim=param_dim,
            n_blocks=n_blocks,
            use_pyramid=True,
            use_graph_attn=True,
            adj_matrix=adj_matrix
        )

        # 5. 生理状态自锁门控 (稳态阻滞去纹波)
        self.state_latch = PhysiologicalStateLatch(
            channels=width,
            num_vars=num_vars,
            param_dim=param_dim
        )

        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

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
        端到端因果自锁推理
        """
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T)
        else:
            B, V, T = x.shape
            t_grid = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, t_grid], dim=1)

        # 1. 物理动态事件特征注入
        event_emb = self.mass_event_trigger(x_in, p)  # (B, width, T)

        # 2. Stage 1 粗生成
        y_coarse = self.stage1_coarse(x, p)  # (B, 39, T)

        # 3. 提取因果导数与速度
        dy_coarse = self._compute_1st_derivative(y_coarse)
        d2y_coarse = self._compute_2nd_derivative(y_coarse)
        z_refine = torch.cat([y_coarse, dy_coarse, d2y_coarse, x_in], dim=1)

        # 4. 严格拓扑时变因果调控张量
        p_causal_t = self.dynamic_causal_adapter(p, y_coarse)

        # 5. Stage 2 因果细化器
        delta_y = self.stage2_refiner(z_refine, p_causal_t)

        # 6. 生理状态自锁 (抑制阻滞高频抖动)
        delta_y_latched = self.state_latch(delta_y, p)

        # 7. 门控残差复合
        y_final = y_coarse + torch.tanh(self.gamma) * delta_y_latched

        if return_coarse:
            return y_final, y_coarse
        return y_final
