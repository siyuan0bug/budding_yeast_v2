"""Efficient-KAN B-spline regression of the conditional solution at query times."""
from torch import nn
from .common import joint_condition_grid
from ._upstream.kan import KAN as EfficientKAN


class KAN(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4):
        super().__init__()
        if width < 1 or n_blocks < 1:
            raise ValueError('KAN requires positive width and n_blocks')
        self.num_vars, self.param_dim = num_vars, param_dim
        # Official default grid_size=5, spline_order=3, fixed grid_range=[-1,1].
        self.net = EfficientKAN([num_vars + param_dim + 1] + [width] * n_blocks + [num_vars])

    def forward(self, x, params):
        features = joint_condition_grid(x, params, self.num_vars, self.param_dim, uniform=False)
        return self.net(features.transpose(1, 2)).transpose(1, 2).contiguous()
