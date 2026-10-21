"""Mamba-1 temporal regression with input-dependent selective state-space dynamics."""
from torch import nn
from .common import joint_condition_grid
from ._upstream.mamba_backbone import MambaPerTokenRegressor


class Mamba(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4):
        super().__init__()
        if width < 1 or n_blocks < 1:
            raise ValueError('Mamba requires positive width and n_blocks')
        self.num_vars, self.param_dim = num_vars, param_dim
        self.net = MambaPerTokenRegressor(
            input_dim=num_vars + param_dim + 1, d_model=width, num_layers=n_blocks,
            d_state=16, d_conv=4, expand=2, mlp_hidden_dim=2 * width,
            dropout=0.0, num_outputs_per_token=num_vars)

    def forward(self, x, params):
        features = joint_condition_grid(x, params, self.num_vars, self.param_dim, uniform=False)
        return self.net(features.transpose(1, 2)).transpose(1, 2).contiguous()
