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

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1dBlock(nn.Module):
    def __init__(self, width, modes):
        super(FNO1dBlock, self).__init__()
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        return F.gelu(x1 + x2)


class PureFNO_ICLR2021(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, modes=24, width=64):
        super(PureFNO_ICLR2021, self).__init__()
        self.width = width
        self.p = nn.Linear(2 + param_dim, self.width)
        self.fno_blocks = nn.ModuleList([FNO1dBlock(self.width, modes) for _ in range(4)])
        self.q = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, ic_time_grid, params):
        B, V, _, T = ic_time_grid.shape
        p_expand = params.view(B, 1, -1, 1).expand(-1, V, -1, T)
        x_in = torch.cat([ic_time_grid, p_expand], dim=2)
        x = x_in.reshape(B * V, -1, T)
        x = x.permute(0, 2, 1)
        x = self.p(x)
        x = x.permute(0, 2, 1)
        for block in self.fno_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x.reshape(B, V, T)
