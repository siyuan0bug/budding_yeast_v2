import torch
import torch.nn as nn
import torch.nn.functional as F


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


class HyperFNOBlock(nn.Module):
    def __init__(self, width, modes, d_lambda):
        super().__init__()
        self.width = width
        self.modes = modes

        self.scale = 1 / (width * width)
        self.R0 = nn.Parameter(self.scale * torch.randn(width, width, modes, dtype=torch.cfloat))
        self.V0 = nn.Parameter(torch.randn(width, d_lambda) * 0.1)
        self.V1 = nn.Parameter(torch.randn(width, d_lambda) * 0.1)

        self.W0 = nn.Parameter(torch.randn(width, width) * 0.1)
        self.U0 = nn.Parameter(torch.randn(width, width, d_lambda) * 0.1)

        self.register_buffer('I', torch.eye(width))

    def forward(self, x, lambda_prime):
        B, _, T = x.shape
        I_b = self.I.unsqueeze(0)

        mod_r = torch.einsum('id, bd, jd -> bij', self.V0, lambda_prime, self.V1)
        H_r = I_b + mod_r
        R_dyn = self.R0.unsqueeze(0) * H_r.unsqueeze(-1).to(torch.cfloat)

        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.width, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum('bix, boix -> box', x_ft[:, :, :self.modes], R_dyn)
        x_spectral = torch.fft.irfft(out_ft, n=T, dim=-1)

        mod_w = torch.einsum('oid, bd -> boi', self.U0, lambda_prime)
        W_dyn = self.W0.unsqueeze(0) * (I_b + mod_w)
        x_spatial = torch.einsum('boi, bit -> bot', W_dyn, x)

        return F.gelu(x_spectral + x_spatial)


class HyperFNO(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, modes=24, width=64, n_layers=4, d_lambda=32):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.n_layers = n_layers
        self.d_lambda = d_lambda

        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.GELU(),
            nn.Linear(64, d_lambda)
        )

        in_channels = num_vars * 2
        self.P = DynamicProjection(in_channels, width, d_lambda)

        self.blocks = nn.ModuleList([HyperFNOBlock(width, modes, d_lambda) for _ in range(n_layers)])

        self.Q = DynamicProjection(width, 64, d_lambda)
        self.Q_prime = DynamicProjection(64, num_vars, d_lambda)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape

        lambda_prime = self.param_encoder(params)

        x = ic_time_grid.reshape(B, V * F_in, T)

        x = self.P(x, lambda_prime)

        for block in self.blocks:
            x = block(x, lambda_prime)

        x = F.gelu(self.Q(x, lambda_prime))
        u = self.Q_prime(x, lambda_prime)

        return u.reshape(B, V, T)