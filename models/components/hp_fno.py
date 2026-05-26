import torch
import torch.nn as nn
import torch.nn.functional as F


class CoupledSpectralConv1d(nn.Module):
    def __init__(self, num_vars, width, modes):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.modes = modes

        mix_channels = num_vars * width

        self.encoder = nn.Conv1d(mix_channels, mix_channels, 1, dtype=torch.cfloat)
        self.weights = nn.Parameter(torch.rand(mix_channels, self.modes, dtype=torch.cfloat) * (1.0 / mix_channels))
        self.decoder = nn.Conv1d(mix_channels, mix_channels, 1, dtype=torch.cfloat)

    def forward(self, x):
        B, V, W, T = x.shape

        x_ft = torch.fft.rfft(x, dim=-1)

        x_ft_flat = x_ft.reshape(B, V * W, -1)
        out_ft_flat = torch.zeros_like(x_ft_flat)

        active_modes = min(self.modes, x_ft_flat.size(-1))

        mixed_ft = self.encoder(x_ft_flat[:, :, :active_modes])
        mixed_ft = mixed_ft * self.weights[:, :active_modes]
        unmixed_ft = self.decoder(mixed_ft)

        out_ft_flat[:, :, :active_modes] = unmixed_ft

        out_ft = out_ft_flat.reshape(B, V, W, -1)
        x_out = torch.fft.irfft(out_ft, n=T, dim=-1)
        return x_out


class hpFNOxBlock(nn.Module):
    def __init__(self, num_vars, width, modes):
        super().__init__()
        self.num_vars = num_vars
        self.width = width

        self.fno_conv = CoupledSpectralConv1d(num_vars, width, modes)

        self.w = nn.Conv1d(num_vars * width, num_vars * width, 1, groups=num_vars)

    def forward(self, x, shift):
        B, V, W, T = x.shape

        x_spectral = self.fno_conv(x)

        x_flat = x.reshape(B, V * W, T)
        x_local = self.w(x_flat).reshape(B, V, W, T)

        return F.gelu(x_spectral + x_local + shift)


class hpYeastFNO(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, modes=24, width=128):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.num_layers = 4

        self.hypernet = nn.Linear(param_dim, self.num_layers * width)

        self.p_layer = nn.Linear(2, width)

        self.blocks = nn.ModuleList([hpFNOxBlock(num_vars, width, modes) for _ in range(self.num_layers)])

        self.q_layer1 = nn.Conv1d(num_vars * width, num_vars * 64, 1, groups=num_vars)
        self.q_layer2 = nn.Conv1d(num_vars * 64, num_vars * 1, 1, groups=num_vars)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape

        shifts = self.hypernet(params)
        shifts = shifts.reshape(B, self.num_layers, 1, self.width, 1)

        x = ic_time_grid.permute(0, 1, 3, 2)
        x = self.p_layer(x)
        x = x.permute(0, 1, 3, 2)

        for i, block in enumerate(self.blocks):
            x = block(x, shifts[:, i])

        x_flat = x.reshape(B, V * self.width, T)
        out = F.gelu(self.q_layer1(x_flat))
        out = self.q_layer2(out)

        return out.reshape(B, V, T)