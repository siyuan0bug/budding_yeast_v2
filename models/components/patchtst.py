"""
PatchTST: A Time Series is Worth 64 Words — Long-term Forecasting with Transformers
================================================================================
Reference: Nie Y., Nguyen N., Sinthong P., Kalagnanam J. (ICLR 2023)
           "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
           https://openreview.net/forum?id=Jbdc0vTOcol

核心创新:
1. Patching: 将时间序列分割为 patch（子序列级嵌入），大幅减少 token 数量，
   降低计算复杂度，同时保留局部语义信息。
2. Channel Independence: 每个变量独立处理，避免多变量间的参数耦合，
   提升泛化能力并降低过拟合风险。

本实现适配 YeastLitModule 接口:
    输入: ic_time_grid (B, V, 2, T), params (B, param_dim)
    输出: (B, V, T)

其中 ic_time_grid 的 dim 2 = [IC, t_grid]，IC 为初始条件通道，t_grid 为时间网格。
参数 params 通过 FiLM 调制注入到 Transformer 各层，实现条件生成。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    """便捷的维度转置层"""
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class PositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        # x: (B, N, d_model)
        return x + self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    """
    将 1D 时间序列分割为 patch 并线性投影到 d_model 维。

    输入: (B, T) 单变量时间序列
    输出: (B, N_patches, d_model)
    """
    def __init__(self, patch_len, stride, d_model, in_channels=2):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * in_channels, d_model)

    def forward(self, x):
        # x: (B, C, T) 其中 C 为输入通道数（IC + t_grid）
        B, C, T = x.shape
        # 计算可分的 patch 数
        num_patches = (T - self.patch_len) // self.stride + 1
        # 确保能整除，否则截断
        keep_len = (num_patches - 1) * self.stride + self.patch_len
        x = x[:, :, :keep_len]
        # 展开 patch: (B, C, N, P) via unfold
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # 合并通道和 patch: (B, N, C*P)
        # x: (B, C, N, P) -> (B, N, C, P) -> (B, N, C*P)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(B, num_patches, C * self.patch_len)
        x = self.proj(x)
        return x, num_patches


class FiLMModulation(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) 条件注入层。
    用参数生成 scale (γ) 和 shift (β) 来调制 Transformer 特征。

    参考: Perez E. et al. (AAAI 2018) "FiLM: Visual Reasoning with a General Conditioning Layer"
    """
    def __init__(self, d_model, param_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(param_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )

    def forward(self, x, params):
        # x: (B, N, d_model), params: (B, param_dim)
        film = self.proj(params)  # (B, 2*d_model)
        gamma, beta = film.chunk(2, dim=-1)  # (B, d_model) each
        gamma = gamma.unsqueeze(1)  # (B, 1, d_model)
        beta = beta.unsqueeze(1)
        return x * (1 + gamma) + beta


class MultiHeadAttention(nn.Module):
    """标准多头自注意力（支持可选的 FiLM 调制）"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        # x: (B, N, d_model)
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 缩放点积注意力
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer 编码器块，带 FiLM 参数调制"""
    def __init__(self, d_model, n_heads, d_ff, dropout, param_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.film = FiLMModulation(d_model, param_dim)

    def forward(self, x, params):
        # 自注意力子层
        h = self.norm1(x)
        h = self.attn(h)
        x = x + self.film(h, params)  # FiLM 调制残差
        # 前馈子层
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class PatchTST(nn.Module):
    """
    PatchTST: Patch Time Series Transformer (ICLR 2023)

    通道独立 + Patching + Transformer 编码器，通过 FiLM 注入参数条件。

    Args:
        num_vars: 变量数 V (通道独立处理)
        param_dim: 参数维度 (用于 FiLM 条件调制)
        modes: 兼容接口（未使用，保留以兼容 run_pipeline.sh）
        width: d_model (Transformer 隐藏维度)
        n_blocks: Transformer 层数
        patch_len: patch 长度 P
        stride: patch 步长 S
        n_heads: 多头注意力头数
        d_ff: FFN 中间维度
        dropout: dropout 概率
    """
    def __init__(
        self,
        num_vars=39,
        param_dim=141,
        modes=24,  # 兼容接口，未使用
        width=128,
        n_blocks=4,
        patch_len=16,
        stride=8,
        n_heads=8,
        d_ff=None,
        dropout=0.1,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = width
        self.patch_len = patch_len
        self.stride = stride
        if d_ff is None:
            d_ff = 4 * width

        # Patch 嵌入: 输入 2 通道 (IC + t_grid) × patch_len
        self.patch_embed = PatchEmbedding(
            patch_len=patch_len,
            stride=stride,
            d_model=width,
            in_channels=2,
        )

        # 位置编码
        # 预估最大 patch 数: T=500, P=16, S=8 → (500-16)/8+1 = 61
        max_patches = 256
        self.pos_enc = PositionalEncoding(width, max_len=max_patches)

        # Transformer 编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(width, n_heads, d_ff, dropout, param_dim)
            for _ in range(n_blocks)
        ])

        self.norm = nn.LayerNorm(width)

        # 输出头: 将 patch 嵌束展平并投影到 T
        # 输入维度 = num_patches * d_model，但 num_patches 依赖 T
        # 使用自适应方式: 每个 patch 投影到对应的时间子段
        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, patch_len),
        )

    def forward(self, ic_time_grid, params):
        """
        Args:
            ic_time_grid: (B, V, 2, T), dim 2 = [IC, t_grid]
            params: (B, param_dim)

        Returns:
            (B, V, T) 预测轨迹
        """
        B, V, C, T = ic_time_grid.shape
        assert C == 2, f"期望输入通道数为 2 (IC + t_grid), 实际 {C}"

        # ===== 通道独立处理: 将 V 维视为 batch 维 =====
        # (B, V, 2, T) -> (B*V, 2, T)
        x = ic_time_grid.reshape(B * V, C, T)

        # Patch 嵌入: (B*V, 2, T) -> (B*V, N, d_model)
        x, num_patches = self.patch_embed(x)

        # 位置编码
        x = self.pos_enc(x)

        # 参数广播到每个变量: (B, param_dim) -> (B*V, param_dim)
        params_expanded = params.unsqueeze(1).expand(-1, V, -1).reshape(B * V, -1)

        # Transformer 编码
        for block in self.blocks:
            x = block(x, params_expanded)
        x = self.norm(x)

        # 输出头: 每个 patch 预测 patch_len 个时间步
        # (B*V, N, d_model) -> (B*V, N, patch_len)
        out = self.head(x)

        # 重建完整时间序列: 将重叠的 patch 平均合并
        # (B*V, N, patch_len) -> (B*V, T)
        out = self._overlap_add(out, T, num_patches)

        # 还原形状: (B*V, T) -> (B, V, T)
        out = out.reshape(B, V, T)
        return out

    def _overlap_add(self, patches, target_len, num_patches):
        """
        将重叠的 patch 合并为完整时间序列（使用平均）。

        Args:
            patches: (B*V, N, patch_len)
            target_len: 目标时间长度 T
            num_patches: patch 数量 N

        Returns:
            (B*V, T)
        """
        B = patches.shape[0]
        P = self.patch_len
        S = self.stride

        # 初始化输出和计数
        out = torch.zeros(B, target_len, device=patches.device, dtype=patches.dtype)
        count = torch.zeros(B, target_len, device=patches.device, dtype=patches.dtype)

        # 逐 patch 累加
        for i in range(num_patches):
            start = i * S
            end = start + P
            if end > target_len:
                end = target_len
                out[:, start:end] += patches[:, i, :end - start]
                count[:, start:end] += 1
            else:
                out[:, start:end] += patches[:, i]
                count[:, start:end] += 1

        # 平均
        count = torch.clamp(count, min=1.0)
        return out / count
