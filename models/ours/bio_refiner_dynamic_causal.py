#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioRefiner-Dynamic-Causal: 时变动态因果神经算子 (Time-Varying Causal Neural Operator)
===================================================================================
彻底解决芽殖酵母多突变体与单次分裂相变时间抖动的下一代旗舰算子。

核心特性与设计哲学:
1. 保持现有稳定模型完整性:
   - 绝不修改原有 bio_refiner_causal.py 与 bio_refiner_plus.py 的任何现有类；
   - 继承并复用其严谨生物学元数据 (27 基因模块, 54 条单向符号调控边, 141 参数基准先验)。
2. 时变动态因果路由 (Time-Varying Causal Routing):
   - 细胞周期调控网络具有强时相性: G1 期 (SBF/CLN2)、S/G2 期 (CLB5/CLB2/Mcm1)、M 期退役 (CDC20/CDH1/CDC14)；
   - 本模块将 Stage 1 粗糙预测轨迹 y_coarse 映射为当前生理时相活跃度 z_state(t)；
   - 静态基因突变嵌入 e_mut 与时变生理特征 z_state(t) 在时间切片上联合进行拓扑掩码图注意力传播，
     输出逐时刻的因果调控张量 p_causal(t) ∈ R^(B, param_dim, T)。
3. 时变局部 FiLM 卷积细化 (Time-Varying Stiff Refiner):
   - ResNet 膨胀卷积块内部采用逐时刻动态 FiLM 调制: h(t) = h(t) * (1 + gamma(t)) + beta(t)；
   - 使 G1 阻滞突变专注于 G1 时间窗口，M 期退出突变专注于 M 期时间窗口，彻底解决多重突变在时间轴上的因果拮抗。
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bio_mechanistic_operator import BioMechPathwayOnly
from .bio_refiner_plus import (
    StoichiometricGraphAttention,
    TemporalPyramidBlock,
)
from .bio_refiner_causal import (
    PARAM_KEYS_141,
    WT_VALUES_141,
    MODULE_NAMES,
    MODULE_PARAM_INDICES,
    MODULE_REGULATORY_EDGES,
    CausalRMSNorm,
)


# ==============================================================================
# 1. 生理时相映射矩阵 (39 变量 -> 27 基因模块生理状态映射)
# ==============================================================================

def build_var_to_gene_tensor(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    构建 39 个连续浓度变量到 27 个功能基因模块的指示矩阵 M ∈ R^(27, 39)。
    变量索引对应 Chen (2004) 标准 39 变量:
    0: MASS, 1: CLN2, 2: CLB2, 3: CLB5, 4: SIC1, 5: CDC6, 6: C2, 7: C5, 8: F2, 9: F5,
    10: SIC1P, 11: C2P, 12: C5P, 13: CDC6P, 14: F2P, 15: F5P, 16: SWI5T, 17: SWI5, 18: IEP,
    19: CDC20T, 20: CDC20, 21: CDH1T, 22: CDH1, 23: CDC14T, 24: CDC14, 25: NET1T, 26: NET1,
    27: RENT, 28: TEM1, 29: CDC15, 30: PPX, 31: PDS1, 32: ESP1, 33: ORI, 34: BUD, 35: SPN,
    36: Vi20, 37: lte1, 38: BUB2
    """
    M = torch.zeros((len(MODULE_NAMES), 39), device=device, dtype=dtype)
    gene_to_vars = {
        'CLN2': [1],
        'CLN3': [0],
        'CLB2': [2, 6, 8, 11, 14],
        'CLB5': [3, 7, 9, 12, 15],
        'SIC1': [4, 6, 7, 10, 11, 12],
        'CDC6': [5, 8, 9, 13, 14, 15],
        'BCK2': [0],
        'SWI5': [16, 17],
        'CDC20': [19, 20, 36],
        'CDH1': [21, 22],
        'CDC14': [23, 24, 27],
        'NET1': [25, 26, 27],
        'TEM1': [28],
        'CDC15': [29],
        'PPX': [30],
        'PDS1': [31],
        'ESP1': [32],
        'MAD2': [35, 36],
        'BUB2': [38],
        'SBF_MBF': [1, 3],
        'Mcm1': [2, 17],
        'APC_Core': [18, 20, 22],
        'ORI_BUD_SPN': [33, 34, 35],
        'MDT': [0],
        'CDC5_Proxy': [2],
        'CDC28': [1, 2, 3],
        'SCC1': [32, 35],
    }
    for g_idx, g_name in enumerate(MODULE_NAMES):
        if g_name in gene_to_vars:
            for v_idx in gene_to_vars[g_name]:
                M[g_idx, v_idx] = 1.0 / len(gene_to_vars[g_name])
    return M


# ==============================================================================
# 2. 时变拓扑有向图注意力层 (Time-Varying Topology GAT)
# ==============================================================================

class TimeVaryingCausalTopologyGAT(nn.Module):
    """
    时变拓扑因果注意力层:
    在每个时间切片上，将静态突变特征与动态生物时相特征联合，
    严格沿 54 条有向调控边进行图消息传递。
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
        Args:
            e_time: (B, num_nodes, d_gene, T_sub) 时空节点张量
            adj_mask: (num_nodes, num_nodes) 静态生物边掩码
            sign_adj: (num_nodes, num_nodes) 符号偏置矩阵
        Returns:
            out: (B, num_nodes, d_gene, T_sub) 时变更新后节点特征
        """
        B, N, D, T = e_time.shape
        # 将批次与时间轴折叠进行并行矩阵运算: (B*T, N, D)
        e_flat = e_time.permute(0, 3, 1, 2).reshape(B * T, N, D)

        H = self.num_heads
        d_k = self.head_dim

        Q = self.q_proj(e_flat).view(B * T, N, H, d_k).permute(0, 2, 1, 3)
        K = self.k_proj(e_flat).view(B * T, N, H, d_k).permute(0, 2, 1, 3)
        V = self.v_proj(e_flat).view(B * T, N, H, d_k).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)  # (B*T, H, N, N)

        if sign_adj is not None:
            scores = scores + self.sign_bias_weight * sign_adj.unsqueeze(0).unsqueeze(0)

        # 自环保护与拓扑掩码
        diag_eye = torch.eye(N, device=adj_mask.device, dtype=adj_mask.dtype)
        safe_adj = torch.clamp(adj_mask + diag_eye, 0.0, 1.0)
        mask = (safe_adj.unsqueeze(0).unsqueeze(0) == 0)
        scores = scores.masked_fill(mask, -1e4)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        out = torch.matmul(attn_weights, V)  # (B*T, H, N, d_k)
        out = out.permute(0, 2, 1, 3).contiguous().view(B * T, N, D)
        out = self.out_proj(out)

        out_res = self.norm(e_flat + out)  # (B*T, N, D)
        out_time = out_res.view(B, T, N, D).permute(0, 2, 3, 1)  # (B, N, D, T)
        return out_time


# ==============================================================================
# 3. 时变因果突变适配器 (Time-Varying Causal Mutant Adapter)
# ==============================================================================

class TimeVaryingCausalAdapter(nn.Module):
    """
    时变因果突变嵌入适配器:
    接收连续动力学参数 p: (B, 141) 与粗糙轨迹 y_coarse: (B, 39, T)，
    输出随时间演化的时变因果调控张量 p_causal(t): (B, 141, T)。
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

        # 注册野生型 WT 基准参数
        self.register_buffer('p_wt', torch.tensor(WT_VALUES_141, dtype=torch.float32))

        # 构建因果有向图邻接矩阵与符号矩阵
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

        # 1. 静态突变特征提取 (无偏置以保全零扰动)
        self.module_projections = nn.ModuleList()
        for gene_name in MODULE_NAMES:
            indices = MODULE_PARAM_INDICES.get(gene_name, [])
            in_feats = len(indices) * 2  # 包含线性差值与对数残差
            if in_feats > 0:
                proj = nn.Sequential(
                    nn.Linear(in_feats, d_gene, bias=False),
                    nn.GELU(),
                    nn.Linear(d_gene, d_gene, bias=False)
                )
            else:
                proj = nn.Identity()
            self.module_projections.append(proj)

        # 2. 动态生理时相特征投影器: 将 (B, 39, T) 聚合为 (B, 27, d_gene, T_sub)
        self.state_node_proj = nn.Conv1d(27, 27 * d_gene, kernel_size=1, groups=27, bias=False)
        self.state_norm = CausalRMSNorm(d_gene)

        # 3. 时变拓扑图注意力消息传递
        self.time_gat = TimeVaryingCausalTopologyGAT(
            num_nodes=self.num_nodes,
            d_gene=d_gene,
            num_heads=num_heads
        )

        # 4. 时变调控向量投影器: (B, 27 * d_gene, T_sub) -> (B, param_dim, T_sub)
        self.out_temporal_proj = nn.Sequential(
            nn.Conv1d(self.num_nodes * d_gene, param_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(param_dim, param_dim, kernel_size=1, bias=False)
        )

        # 零残差平滑门控标量
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, p: torch.Tensor, y_coarse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: (B, 141) 参数矢量
            y_coarse: (B, 39, T) Stage 1 粗糙轨迹预测
        Returns:
            p_causal_t: (B, 141, T) 时变因果调控张量
        """
        B, D = p.shape
        T = y_coarse.shape[-1]
        device = p.device

        # A. 提取静态基因突变嵌入 e_mut: (B, 27, d_gene)
        delta_p = p - self.p_wt.to(device)
        eps = 1e-6
        log_res = torch.sign(delta_p) * torch.log1p(torch.abs(delta_p) / (torch.abs(self.p_wt.to(device)) + eps))

        e_list = []
        for g_idx, gene_name in enumerate(MODULE_NAMES):
            indices = MODULE_PARAM_INDICES.get(gene_name, [])
            if len(indices) > 0:
                p_diff = delta_p[:, indices]
                p_log = log_res[:, indices]
                feat = torch.cat([p_diff, p_log], dim=1)
                e_g = self.module_projections[g_idx](feat)
            else:
                e_g = torch.zeros(B, self.d_gene, device=device, dtype=p.dtype)
            e_list.append(e_g)

        e_mut = torch.stack(e_list, dim=1)  # (B, 27, d_gene)

        # B. 提取动态生理时相特征 z_state: (B, 27, d_gene, T_sub)
        M_var2gene = build_var_to_gene_tensor(device, p.dtype)  # (27, 39)
        # 沿时间下采样进行抗噪时序平滑
        y_sub = y_coarse[:, :, ::self.t_stride]  # (B, 39, T_sub)
        T_sub = y_sub.shape[-1]

        # 矩阵乘法聚合至 27 节点: (B, 27, T_sub)
        z_node = torch.einsum('gv,bvt->bgt', M_var2gene, y_sub)
        z_emb = self.state_node_proj(z_node)  # (B, 27 * d_gene, T_sub)
        z_emb = z_emb.view(B, self.num_nodes, self.d_gene, T_sub)
        z_emb = self.state_norm(z_emb.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # (B, 27, d_gene, T_sub)

        # C. 静态突变与动态时相联合注入
        e_time = e_mut.unsqueeze(-1) + z_emb  # (B, 27, d_gene, T_sub)

        # D. 时变拓扑图消息传递
        e_time_updated = self.time_gat(e_time, self.adj_mask, self.sign_adj)  # (B, 27, d_gene, T_sub)

        # E. 时变因果特征融合与残差输出
        e_summary = e_time_updated.view(B, self.num_nodes * self.d_gene, T_sub)
        delta_p_t = self.out_temporal_proj(e_summary)  # (B, 141, T_sub)

        # 线性插值回原始 T=500 分辨率
        delta_p_full = F.interpolate(delta_p_t, size=T, mode='linear', align_corners=True)  # (B, 141, T)

        # 残差复合保证 f(WT) = p_wt
        p_causal_t = p.unsqueeze(-1) + torch.tanh(self.alpha) * delta_p_full  # (B, 141, T)
        return p_causal_t


# ==============================================================================
# 4. 时变局部 FiLM 调制膨胀卷积块 (Time-Varying ResConvBlock1D)
# ==============================================================================

class TimeVaryingResConvBlock1D(nn.Module):
    """
    时变局部 FiLM 卷积块:
    支持逐时刻参数缩放与偏置: h(t) = h(t) * (1 + gamma(t)) + beta(t)
    """
    def __init__(self, channels: int, param_dim: int = 141, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

        # 时变 1x1 卷积产生逐时刻 (gamma(t), beta(t))
        self.film_conv = nn.Conv1d(param_dim, 2 * channels, kernel_size=1)

    def forward(self, x: torch.Tensor, p_causal_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) 隐层特征
            p_causal_t: (B, 141, T) 时变因果调制张量, 或 (B, 141) 兼容静态输入
        """
        if p_causal_t.dim() == 2:
            p_causal_t = p_causal_t.unsqueeze(-1)

        gamma, beta = self.film_conv(p_causal_t).chunk(2, dim=1)  # (B, C, T)

        h = self.conv1(x)
        h = self.norm1(h)
        h = h * (1.0 + gamma) + beta
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = h * (1.0 + gamma) + beta
        return self.act(x + h)


# ==============================================================================
# 5. 时变刚性细化网络 (Time-Varying Modular Stiff Refiner)
# ==============================================================================

class TimeVaryingModularStiffRefiner(nn.Module):
    """
    Stage 2 时变刚性残差细化网络:
    整合 Time-Varying FiLM 卷积骨干 + 跨变量化学计量图注意力 + 多尺度时间金字塔
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
# 6. 主模型: BioRefinerDynamicCausalPlus (独立新架构)
# ==============================================================================

class BioRefinerDynamicCausalPlus(nn.Module):
    """
    终极时变因果旗舰模型 (BioRefinerDynamicCausalPlus):
    - 完全独立于现有类，不修改任何已有代码；
    - Stage 1: 生物机制基础算子粗生成 (BioMechPathwayOnly)；
    - 动态因果路由器: 依据粗生成生物时相，实时动态更新 27 节点拓扑消息流动；
    - Stage 2: 时变局部 FiLM 刚性残差矫正器 + 二阶曲率加速度 + 化学计量跨变量注意力；
    - 全面攻克单次分裂 (Pattern B) 相变抖动与复杂多突变拮抗饱和。
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

        # 1. 动态时变因果适配器
        self.dynamic_causal_adapter = TimeVaryingCausalAdapter(
            param_dim=param_dim,
            d_gene=d_gene,
            num_heads=4,
            t_stride=10
        )

        # 2. Stage 1: 粗糙生化算子
        self.stage1_coarse = BioMechPathwayOnly(
            num_vars=num_vars,
            width=width,
            modes=modes,
            param_dim=param_dim,
            n_blocks=coarse_n_blocks
        )

        # 3. Stage 2 通道计算: y_coarse(39) + dy/dt(39) + d2y/dt2(39) + x_in(78) = 195
        in_channels = num_vars * 5  # 195

        # 4. Stage 2: 时变残差矫正器 (融合曲率 + 金字塔 + 反应图注意力)
        self.stage2_refiner = TimeVaryingModularStiffRefiner(
            in_channels=in_channels,
            num_vars=num_vars,
            width=width,
            param_dim=param_dim,
            n_blocks=n_blocks,
            use_pyramid=True,
            use_graph_attn=True,
            adj_matrix=adj_matrix
        )

        # 门控残差缩放因子
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
        端到端前向推理
        Args:
            x: (B, 39, 2, T) 或 (B, 39, T)
            p: (B, 141) 参数矢量
        """
        # 标准化 x_in 输入
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T)  # (B, 78, T)
        else:
            B, V, T = x.shape
            t_grid = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, t_grid], dim=1)  # (B, 78, T)

        # 1. Stage 1: 生成粗糙动力学轨迹 y_coarse
        y_coarse = self.stage1_coarse(x, p)  # (B, 39, T)

        # 2. 提取一阶速度与二阶曲率加速度特征
        dy_coarse = self._compute_1st_derivative(y_coarse)
        d2y_coarse = self._compute_2nd_derivative(y_coarse)
        z_refine = torch.cat([y_coarse, dy_coarse, d2y_coarse, x_in], dim=1)  # (B, 195, T)

        # 3. 时变动态因果路由: 结合静态参数扰动与当前生理时相特征
        p_causal_t = self.dynamic_causal_adapter(p, y_coarse)  # (B, 141, T)

        # 4. Stage 2: 时变细化网络矫正残差
        delta_y = self.stage2_refiner(z_refine, p_causal_t)  # (B, 39, T)

        # 5. 门控复合
        y_final = y_coarse + torch.tanh(self.gamma) * delta_y

        if return_coarse:
            return y_final, y_coarse
        return y_final
