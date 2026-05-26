import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], self.weights)
        return torch.fft.irfft(out_ft, n=x.size(-1))


class SparseTemporalMixer(nn.Module):
    def __init__(self, causal_matrix, cond_dim=43):
        super().__init__()
        self.num_vars = causal_matrix.shape[0]

        self.register_buffer('mask', (causal_matrix != 0).float())

        init_weight = causal_matrix.float() * 0.5 + torch.randn_like(causal_matrix.float()) * 0.05
        self.static_weight = nn.Parameter(init_weight * self.mask)

        self.attn_net = nn.Sequential(
            nn.Conv1d(self.num_vars + cond_dim, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, self.num_vars * self.num_vars, 1)
        )

    def forward(self, x, cond_expand):
        B, C, T = x.shape
        attn_in = torch.cat([x, cond_expand], dim=1)
        attn_logits = self.attn_net(attn_in)
        attn_logits = attn_logits.view(B, self.num_vars, self.num_vars, T)

        attn = torch.sigmoid(attn_logits) * self.mask.view(1, self.num_vars, self.num_vars, 1)

        W_dynamic = self.static_weight.view(1, self.num_vars, self.num_vars, 1) * attn
        out = torch.einsum('bijt, bjt -> bit', W_dynamic, x)
        return out


class SparseAttnFNOBlock(nn.Module):
    def __init__(self, causal_matrix, modes, cond_dim=43):
        super().__init__()
        num_vars = causal_matrix.shape[0]
        self.fno_conv = SpectralConv1d(num_vars, num_vars, modes)
        self.mixer = SparseTemporalMixer(causal_matrix, cond_dim)

    def forward(self, x, cond_expand):
        x_fft = self.fno_conv(x)
        x_mix = self.mixer(x, cond_expand)
        return F.gelu(x_fft + x_mix)


class UltimateFNO(nn.Module):
    def __init__(self, causal_matrix, param_dim=42, modes=24):
        super().__init__()
        self.num_vars = causal_matrix.shape[0]
        self.cond_dim = param_dim + 1

        self.block1 = SparseAttnFNOBlock(causal_matrix, modes, self.cond_dim)
        self.block2 = SparseAttnFNOBlock(causal_matrix, modes, self.cond_dim)
        self.block3 = SparseAttnFNOBlock(causal_matrix, modes, self.cond_dim)
        self.block4 = SparseAttnFNOBlock(causal_matrix, modes, self.cond_dim)

        self.out_proj = nn.Conv1d(self.num_vars, self.num_vars, 1)

    def forward(self, ic, t_grid, param_cond):
        B, _, T = ic.shape
        p_expand = param_cond.unsqueeze(-1).expand(B, param_cond.shape[1], T)
        cond_expand = torch.cat([p_expand, t_grid], dim=1)

        x = ic
        x = self.block1(x, cond_expand)
        x = self.block2(x, cond_expand)
        x = self.block3(x, cond_expand)
        x = self.block4(x, cond_expand)

        return self.out_proj(x)
