"""Plain MLP baseline: concatenate full initial state, parameters and query time."""
from torch import nn
from .common import joint_condition_grid


class MLP(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4):
        super().__init__()
        if n_blocks < 1:
            raise ValueError('n_blocks must be positive')
        self.num_vars, self.param_dim = num_vars, param_dim
        layers = [nn.Linear(num_vars + param_dim + 1, width), nn.GELU()]
        for _ in range(n_blocks - 1):
            layers.extend((nn.Linear(width, width), nn.GELU()))
        layers.append(nn.Linear(width, num_vars))
        self.net = nn.Sequential(*layers)

    def forward(self, x, params):
        features = joint_condition_grid(x, params, self.num_vars, self.param_dim, uniform=False)
        return self.net(features.transpose(1, 2)).transpose(1, 2).contiguous()
