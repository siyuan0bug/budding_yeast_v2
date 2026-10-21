"""Temporal Transformer encoder regressor; full initial state and parameters per token."""
from torch import nn
from .common import joint_condition_grid
from ._upstream.transformer import TransformerPerTokenRegressor


class Transformer(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4):
        super().__init__()
        if width < 4 or width % 4 or n_blocks < 1:
            raise ValueError('Transformer requires width divisible by 4 and positive n_blocks')
        self.num_vars, self.param_dim = num_vars, param_dim
        self.net = TransformerPerTokenRegressor(
            input_dim=num_vars + param_dim + 1, d_model=width, nhead=4,
            num_layers=n_blocks, dim_feedforward=2 * width, dropout=0.0,
            num_outputs_per_token=num_vars)

    def forward(self, x, params):
        features = joint_condition_grid(x, params, self.num_vars, self.param_dim, uniform=False)
        return self.net(features.transpose(1, 2)).transpose(1, 2).contiguous()
