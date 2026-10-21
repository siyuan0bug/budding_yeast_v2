"""Plain residual MLP baseline, with full task inputs concatenated at the stem."""
from torch import nn
from .common import joint_condition_grid


class ResNet(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4):
        super().__init__()
        if n_blocks < 1:
            raise ValueError('n_blocks must be positive')
        self.num_vars, self.param_dim = num_vars, param_dim
        self.stem = nn.Linear(num_vars + param_dim + 1, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
            for _ in range(n_blocks)
        ])
        self.act = nn.GELU()
        self.head = nn.Linear(width, num_vars)

    def forward(self, x, params):
        h = self.stem(joint_condition_grid(x, params, self.num_vars, self.param_dim, uniform=False).transpose(1, 2))
        for block in self.blocks:
            h = self.act(h + block(h))
        return self.head(h).transpose(1, 2).contiguous()
