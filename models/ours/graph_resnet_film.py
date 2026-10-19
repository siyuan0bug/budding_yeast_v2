import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .graph_contracts import resolve_variable_graph, attach_variable_graph


class FiLMLayer(nn.Module):
    """
    特征线性调制层 (Feature-wise Linear Modulation):
    基于突变参数隐向量 lambda_prime 预测 (gamma, beta) 并进行仿射变换:
    x_mod = x * (1.0 + tanh(gamma)) + beta
    """
    def __init__(self, d_lambda, width):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_lambda, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width * 2)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, lambda_prime):
        # x: (B, width, T), lambda_prime: (B, d_lambda)
        mod = self.mlp(lambda_prime) # (B, width * 2)
        gamma, beta = torch.chunk(mod, 2, dim=-1)
        gamma = gamma.unsqueeze(-1) # (B, width, 1)
        beta = beta.unsqueeze(-1)   # (B, width, 1)
        return x * (1.0 + torch.tanh(gamma)) + beta


class SignedGraphCausalAttention(nn.Module):
    """
    带符号生物反应拓扑图因果注意力交互层:
    - 将时空隐状态投影到 39 个蛋白质通道空间 (B, T, 39, d_attn)
    - 结合带符号物理邻接图 A_bio (分离激活边 A+ 与抑制边 A-)
    - 注入动态突变参数边偏置 Delta A(p)
    - 进行符合生化反应拓扑的因果注意力扩散，计算复杂度与显存极度轻量 (仅 ~0.15GB)
    """
    def __init__(self, num_vars=39, width=64, d_attn=16, d_lambda=128, adj_matrix=None):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.d_attn = d_attn

        # Legacy buffer names retained for checkpoint compatibility. Default +1
        # is an unsigned attention affinity, NOT a biochemical activation sign.
        graph = resolve_variable_graph(adj_matrix, num_vars)
        adj_tensor = graph.matrix.clone()
        self.register_buffer('adj_signed', adj_tensor)
        self.register_buffer('adj_act', torch.clamp(adj_tensor, min=0.0))
        self.register_buffer('adj_inh', torch.clamp(-adj_tensor, min=0.0))
        attach_variable_graph(self, graph, ('adj_signed', 'adj_act', 'adj_inh'))

        # 2. 可学习带符号通路权重
        self.w_act = nn.Parameter(torch.ones(num_vars, num_vars) * 0.5)
        self.w_inh = nn.Parameter(torch.ones(num_vars, num_vars) * 0.5)

        # 3. 动态突变因果偏置生成器
        self.ppi_bias = nn.Sequential(
            nn.Linear(d_lambda, 64),
            nn.GELU(),
            nn.Linear(64, num_vars * num_vars)
        )
        nn.init.zeros_(self.ppi_bias[-1].weight)
        nn.init.zeros_(self.ppi_bias[-1].bias)

        # 4. 投影与注意力线性层
        self.to_vars = nn.Linear(width, num_vars * d_attn)
        self.from_vars = nn.Linear(num_vars * d_attn, width)

        self.fc_q = nn.Linear(d_attn, d_attn)
        self.fc_k = nn.Linear(d_attn, d_attn)
        self.fc_v = nn.Linear(d_attn, d_attn)

    def forward(self, x, lambda_prime):
        """
        x: (B, width, T)
        lambda_prime: (B, d_lambda)
        """
        B, W, T = x.shape
        x_T = x.permute(0, 2, 1) # (B, T, W)
        x_vars = self.to_vars(x_T).reshape(B, T, self.num_vars, self.d_attn)

        Q = self.fc_q(x_vars)
        K = self.fc_k(x_vars)
        V_val = self.fc_v(x_vars)

        # 计算自注意力亲和度 (B, T, V, V)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_attn ** 0.5)

        # 注入带符号生物反应图先验 + 动态突变偏置
        w_act_eff = F.softplus(self.w_act) * self.adj_act
        w_inh_eff = F.softplus(self.w_inh) * self.adj_inh
        a_bio = (w_act_eff - w_inh_eff).unsqueeze(0).unsqueeze(0) # (1, 1, V, V)

        delta_a = self.ppi_bias(lambda_prime).reshape(B, 1, self.num_vars, self.num_vars)
        scores = scores + a_bio + delta_a

        attn = F.softmax(scores, dim=-1)
        out_vars = torch.matmul(attn, V_val) # (B, T, V, d_attn)

        out_flat = out_vars.reshape(B, T, self.num_vars * self.d_attn)
        out = self.from_vars(out_flat).permute(0, 2, 1) # (B, width, T)
        return out


class GraphResNetFiLMBlock(nn.Module):
    """
    结合多尺度空洞时域卷积、FiLM 参数调制与生化反应图因果注意力的核心残差块
    """
    def __init__(self, width, dilation, d_lambda, num_vars=39, d_attn=16, adj_matrix=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, width)
        self.film1 = FiLMLayer(d_lambda, width)
        self.conv1 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)

        self.norm2 = nn.GroupNorm(1, width)
        self.film2 = FiLMLayer(d_lambda, width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)

        self.graph_attn = SignedGraphCausalAttention(num_vars=num_vars, width=width,
                                                     d_attn=d_attn, d_lambda=d_lambda,
                                                     adj_matrix=adj_matrix)

    def forward(self, x, lambda_prime):
        res = x

        h = self.norm1(x)
        h = self.film1(h, lambda_prime)
        h = F.gelu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.film2(h, lambda_prime)
        h = F.gelu(h)
        h = self.conv2(h)

        # 生化反应图因果注意力交互
        h_graph = self.graph_attn(h, lambda_prime)

        return res + h + h_graph


class GraphResNetFiLM(nn.Module):
    """
    GraphResNetFiLM: 融合生化反应拓扑图因果扩散、多尺度空洞卷积与 FiLM 参数调制的超轻量高性能神经算子
    """
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4,
                 d_lambda=128, d_attn=16, adj_matrix=None):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.width = width
        self.d_lambda = d_lambda

        # 1. 突变参数编码器
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 256),
            nn.GELU(),
            nn.Linear(256, d_lambda),
            nn.LayerNorm(d_lambda)
        )

        # 2. 初始条件提升层: [IC, t_grid] (39*2=78 通道) -> width (64/128 通道)
        self.lifting = nn.Conv1d(num_vars * 2, width, kernel_size=1)

        # 3. 多尺度空洞图因果残差块 (dilations: 1, 2, 4, 8, ...)
        dilations = [2 ** (i % 4) for i in range(n_blocks)] # [1, 2, 4, 8]
        self.blocks = nn.ModuleList([
            GraphResNetFiLMBlock(width, dilation=dilations[i], d_lambda=d_lambda,
                                 num_vars=num_vars, d_attn=d_attn, adj_matrix=adj_matrix)
            for i in range(n_blocks)
        ])

        # 4. 投影层: width -> 39
        self.norm_out = nn.GroupNorm(1, width)
        self.projection = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

    def forward(self, x, p):
        """
        Args:
            x: (B, V, 2, T) 输入张量 [IC, t_grid]
            p: (B, param_dim) 突变参数向量
        Returns:
            out: (B, V, T) 预测的 39 个蛋白质全时程浓度轨迹
        """
        B, V, _, T = x.shape

        # 1. 编码参数
        lambda_prime = self.param_encoder(p) # (B, d_lambda)

        # 2. 提升维度 (B, 39*2, T) -> (B, width, T)
        x_flat = x.reshape(B, V * 2, T)
        h = self.lifting(x_flat)

        # 3. 逐层图因果残差扩散
        for block in self.blocks:
            h = block(h, lambda_prime)

        # 4. 投影输出 (B, 39, T)
        h = self.norm_out(h)
        out = self.projection(h)

        return out
