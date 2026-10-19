import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_contracts import resolve_variable_graph, attach_variable_graph


class DynamicProjection(nn.Module):
    def __init__(self, in_dim, out_dim, d_lambda):
        super().__init__()
        self.W0 = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.U0 = nn.Parameter(torch.randn(out_dim, in_dim, d_lambda) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, lambda_prime):
        mod_w = torch.einsum('oid, bd -> boi', self.U0, lambda_prime)
        W_dyn = self.W0.unsqueeze(0) * (1.0 + mod_w)
        out = torch.einsum('boi, bit -> bot', W_dyn, x) + self.bias.unsqueeze(0).unsqueeze(-1)
        return out


class ChannelAttention(nn.Module):
    """
    Channel-wise Attention over the 39 protein channels at each time step,
    with parameter-conditioned dynamic protein-protein interaction (PPI) bias.
    """
    def __init__(self, num_vars=39, width=64, d_attn=16, d_lambda=32, adj_matrix=None):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.d_attn = d_attn

        # Project width -> (num_vars * d_attn)
        self.to_vars = nn.Linear(width, num_vars * d_attn)
        self.from_vars = nn.Linear(num_vars * d_attn, width)

        # Multi-head / single-head Q, K, V projections
        self.fc_q = nn.Linear(d_attn, d_attn)
        self.fc_k = nn.Linear(d_attn, d_attn)
        self.fc_v = nn.Linear(d_attn, d_attn)

        # Dynamic PPI Affinity Generator: predicts (num_vars x num_vars) interaction bias from lambda_prime
        self.ppi_bias_generator = nn.Sequential(
            nn.Linear(d_lambda, 64),
            nn.GELU(),
            nn.Linear(64, num_vars * num_vars)
        )

        graph = resolve_variable_graph(adj_matrix, num_vars)
        self.register_buffer('adj_matrix', graph.matrix.clone())
        self.adj_weight = nn.Parameter(torch.ones(num_vars, num_vars) * 0.1)
        attach_variable_graph(self, graph, ('adj_matrix',), optional_parameters=('adj_weight',))

        self.norm = nn.GroupNorm(1, width)

    def forward(self, x, lambda_prime):
        """
        x: (B, width, T)
        lambda_prime: (B, d_lambda)
        """
        B, W, T = x.shape
        x_norm = self.norm(x) # (B, W, T)

        # 1. Project to (B, T, num_vars, d_attn)
        x_T = x_norm.permute(0, 2, 1) # (B, T, W)
        x_vars = self.to_vars(x_T).reshape(B, T, self.num_vars, self.d_attn) # (B, T, V, d_attn)

        # 2. Compute Q, K, V
        Q = self.fc_q(x_vars) # (B, T, V, d_attn)
        K = self.fc_k(x_vars) # (B, T, V, d_attn)
        V_val = self.fc_v(x_vars) # (B, T, V, d_attn)

        # 3. Scaled dot-product attention scores across V channels
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_attn ** 0.5) # (B, T, V, V)

        # 4. Add dynamic parameter-conditioned interaction bias
        # When CDC6 is high and CLB2 is mutated, this dynamically modulates the affinity between ch 2 and ch 5
        dynamic_ppi_bias = self.ppi_bias_generator(lambda_prime).reshape(B, 1, self.num_vars, self.num_vars)
        attn_scores = attn_scores + dynamic_ppi_bias

        if self.adj_matrix is not None:
            phys_bias = torch.relu(self.adj_weight) * self.adj_matrix
            attn_scores = attn_scores + phys_bias.unsqueeze(0).unsqueeze(1)

        # 5. Softmax attention & value aggregation
        attn_probs = F.softmax(attn_scores, dim=-1) # (B, T, V, V)
        x_out_vars = torch.matmul(attn_probs, V_val) # (B, T, V, d_attn)

        # 6. Project back to (B, width, T)
        x_out_flat = x_out_vars.reshape(B, T, self.num_vars * self.d_attn) # (B, T, V * d_attn)
        x_out = self.from_vars(x_out_flat).permute(0, 2, 1) # (B, W, T)

        return x_out


class ChannelHyperFNOBlock(nn.Module):
    def __init__(self, width, modes, d_lambda, num_vars=39, d_attn=16, adj_matrix=None):
        super().__init__()
        self.width = width
        self.modes = modes

        # Temporal Fourier Spectral Convolution
        self.scale = 1 / (width * width)
        self.R0 = nn.Parameter(self.scale * torch.randn(width, width, modes, dtype=torch.cfloat))
        self.V0 = nn.Parameter(torch.randn(width, d_lambda) * 0.1)
        self.V1 = nn.Parameter(torch.randn(width, d_lambda) * 0.1)

        # Spatial Dense Projection
        self.W0 = nn.Parameter(torch.randn(width, width) * 0.1)
        self.U0 = nn.Parameter(torch.randn(width, width, d_lambda) * 0.1)

        # Channel-wise Attention over 39 protein variables
        self.channel_attn = ChannelAttention(num_vars=num_vars, width=width, d_attn=d_attn,
                                             d_lambda=d_lambda, adj_matrix=adj_matrix)

        self.register_buffer('I', torch.eye(width))

    def forward(self, x, lambda_prime):
        B, _, T = x.shape
        I_b = self.I.unsqueeze(0)

        # 1. Temporal Spectral Conv
        mod_r = torch.einsum('id, bd, jd -> bij', self.V0, lambda_prime, self.V1)
        H_r = I_b + mod_r
        R_dyn = self.R0.unsqueeze(0) * H_r.unsqueeze(-1).to(torch.cfloat)

        orig_dtype = x.dtype
        x_float = x.float()
        x_ft = torch.fft.rfft(x_float, dim=-1)
        out_ft = torch.zeros(B, self.width, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum('bix, boix -> box', x_ft[:, :, :self.modes], R_dyn)
        x_spectral = torch.fft.irfft(out_ft, n=T, dim=-1).to(orig_dtype)

        # 2. Spatial Linear Projection
        mod_w = torch.einsum('oid, bd -> boi', self.U0, lambda_prime)
        W_dyn = self.W0.unsqueeze(0) * (I_b + mod_w)
        x_spatial = torch.einsum('boi, bit -> bot', W_dyn, x)

        # 3. Channel Attention across 39 protein channels
        x_attn = self.channel_attn(x, lambda_prime)

        # 4. Residual connection & activation
        return F.gelu(x_spectral + x_spatial + x_attn + x)


class ChannelHyperFNO(nn.Module):
    """
    Channel-wise Attention HyperFNO:
    Unifies Fourier Spectral Conv1D (temporal dynamics), Dynamic Spatial Projection,
    and Dynamic Channel Self-Attention over 39 protein variables with parameter-conditioned PPI bias.
    """
    def __init__(self, num_vars=39, param_dim=141, modes=24, width=64, n_layers=4, d_lambda=32,
                 d_attn=16, adj_matrix=None):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.n_layers = n_layers
        self.d_lambda = d_lambda

        # Parameter encoder to latent vector lambda_prime
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.GELU(),
            nn.Linear(64, d_lambda)
        )

        in_channels = num_vars * 2
        self.P = DynamicProjection(in_channels, width, d_lambda)

        self.blocks = nn.ModuleList([
            ChannelHyperFNOBlock(width=width, modes=modes, d_lambda=d_lambda,
                                 num_vars=num_vars, d_attn=d_attn, adj_matrix=adj_matrix)
            for _ in range(n_layers)
        ])

        self.Q = DynamicProjection(width, 64, d_lambda)
        self.Q_prime = DynamicProjection(64, num_vars, d_lambda)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape

        # Encode parameters to dynamic conditioning vector
        lambda_prime = self.param_encoder(params)

        # Reshape input grid (B, 39*2, T)
        x = ic_time_grid.reshape(B, V * F_in, T)

        # Initial dynamic projection
        x = self.P(x, lambda_prime)

        # Channel-wise Attention HyperFNO Blocks
        for block in self.blocks:
            x = block(x, lambda_prime)

        # Output dynamic projections
        x = F.gelu(self.Q(x, lambda_prime))
        u = self.Q_prime(x, lambda_prime)

        return u.reshape(B, V, T)
