import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class NIPSYeastBlock(nn.Module):
    def __init__(self, num_vars, width, modes, adj_matrix=None):
        super().__init__()
        self.modes = modes
        self.width = width
        self.num_vars = num_vars

        self.norm_fourier = nn.GroupNorm(1, num_vars * width)
        self.norm_attn = nn.GroupNorm(1, num_vars * width)
        self.norm_local = nn.GroupNorm(1, num_vars * width)

        self.fourier_conv = SpectralConv1d(num_vars * width, num_vars * width, modes)

        self.fc_q = nn.Linear(width, width)
        self.fc_k = nn.Linear(width, width)
        self.fc_v = nn.Linear(width, width)

        self.w = nn.Conv1d(num_vars * width, num_vars * width, 1, groups=num_vars)

        if adj_matrix is not None:
            self.register_buffer('adj_matrix', torch.tensor(adj_matrix, dtype=torch.float32))
            self.physics_weights = nn.Parameter(torch.ones(num_vars, num_vars) * 0.1)
        else:
            self.adj_matrix = None

    def forward(self, x, shift):
        B, V, W, T = x.shape
        x_flat = x.reshape(B, V * W, T)

        x_fourier_in = self.norm_fourier(x_flat)
        x_ft = self.fourier_conv(x_fourier_in).reshape(B, V, W, T)

        x_attn_in = self.norm_attn(x_flat).reshape(B, V, W, T)

        x_ft_T = x_ft.permute(0, 3, 1, 2)
        x_attn_in_T = x_attn_in.permute(0, 3, 1, 2)

        Q = self.fc_q(x_ft_T)
        K = self.fc_k(x_attn_in_T)
        V_val = self.fc_v(x_attn_in_T)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.width ** 0.5)

        if self.adj_matrix is not None:
            positive_weights = torch.relu(self.physics_weights)
            phys_bias = positive_weights * self.adj_matrix
            phys_bias = phys_bias.unsqueeze(0).unsqueeze(1)
            attn_scores = attn_scores + phys_bias

        attn_probs = F.softmax(attn_scores, dim=-1)
        x_attn = torch.matmul(attn_probs, V_val)
        x_attn = x_attn.permute(0, 2, 3, 1)

        x_local = self.w(self.norm_local(x_flat)).reshape(B, V, W, T)

        return F.gelu(x_attn + x_ft + x_local + shift + x)


class ICEncoder(nn.Module):
    def __init__(self, num_vars, width, modes, n_encoder_blocks=2, adj_matrix=None):
        super().__init__()
        self.p_layer = nn.Linear(2, width)
        self.blocks = nn.ModuleList([
            NIPSYeastBlock(num_vars, width, modes, adj_matrix)
            for _ in range(n_encoder_blocks)
        ])

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.p_layer(x)
        x = x.permute(0, 1, 3, 2)

        zero_shift = torch.zeros(x.shape[0], self.blocks[0].num_vars, self.blocks[0].width, 1, device=x.device)
        for block in self.blocks:
            x = block(x, zero_shift)
        return x


class ParamToICCrossAttention(nn.Module):
    def __init__(self, param_dim, hidden_dim, num_vars=39, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(param_dim, hidden_dim)
        self.var_embed = nn.Parameter(torch.randn(num_vars, hidden_dim) * 0.02)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, ic_features, params, return_weights=False):
        B, n_v, W, T = ic_features.shape
        ic_seq = ic_features.permute(0, 1, 3, 2).reshape(B, n_v * T, W)

        # 1. 生成 Query (B, n_v, hidden_dim)
        q_global = self.q_proj(params)
        Q = q_global.unsqueeze(1) + self.var_embed.unsqueeze(0)

        # 2. 生成 Key 和 Value (B, n_v*T, hidden_dim)
        K = self.k_proj(ic_seq)
        V_attn = self.v_proj(ic_seq)

        # 🌟 核心优化：利用标准的 Multi-Head Attention 维度进行重塑
        # 彻底砍掉 B * n_v 带来的 39 倍内存灾难
        Q = Q.reshape(B, n_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        # Q shape: (B, num_heads, n_v, head_dim)
        
        K = K.reshape(B, n_v * T, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        # K shape: (B, num_heads, n_v*T, head_dim)
        
        V_attn = V_attn.reshape(B, n_v * T, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        # V_attn shape: (B, num_heads, n_v*T, head_dim)

        # 3. 计算注意力分数 (自动广播机制会处理完美匹配)
        # (B, num_heads, n_v, head_dim) @ (B, num_heads, head_dim, n_v*T) -> (B, num_heads, n_v, n_v*T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 4. 加权求和
        # (B, num_heads, n_v, n_v*T) @ (B, num_heads, n_v*T, head_dim) -> (B, num_heads, n_v, head_dim)
        out = torch.matmul(attn_probs, V_attn)
        
        # 5. 恢复最终形状
        out = out.permute(0, 2, 1, 3).reshape(B, n_v, self.num_heads * self.head_dim)
        out = self.out_proj(out)

        if return_weights:
            weights = attn_probs.mean(dim=1)
            return out, weights
            
        return out


class NIPSYeastFNO(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, modes=24, width=16, n_blocks=4, adj_matrix=None):
        super().__init__()
        self.num_vars = num_vars
        self.width = width

        self.ic_encoder = ICEncoder(num_vars, width, modes, n_encoder_blocks=2, adj_matrix=adj_matrix)
        self.cross_attn = ParamToICCrossAttention(param_dim, width, num_vars=num_vars, num_heads=4)

        self.blocks = nn.ModuleList([
            NIPSYeastBlock(num_vars, width, modes, adj_matrix) for _ in range(n_blocks)
        ])

        self.q_layer1 = nn.Conv1d(num_vars * width, num_vars * 64, 1, groups=num_vars)
        self.q_layer2 = nn.Conv1d(num_vars * 64, num_vars * 1, 1, groups=num_vars)

    def forward(self, ic_time_grid, params):
        B, V, _, T = ic_time_grid.shape
        ic_enc = self.ic_encoder(ic_time_grid)
        context = self.cross_attn(ic_enc, params)
        shift = context.unsqueeze(-1)
        x = ic_enc
        for block in self.blocks:
            x = block(x, shift)
        x_flat = x.reshape(B, V * self.width, T)
        out = F.gelu(self.q_layer1(x_flat))
        out = self.q_layer2(out)
        return out.reshape(B, V, T)