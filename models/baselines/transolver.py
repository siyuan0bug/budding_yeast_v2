"""Official Transolver physics attention, with joint-state temporal query I/O."""
import torch
from torch import nn
from .common import joint_condition_grid
from ._upstream.transolver import MLP, Transolver_block


class Transolver(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4, n_heads=4, n_slices=32):
        super().__init__()
        if width % n_heads or n_blocks < 1:
            raise ValueError('Invalid attention width or block count')
        self.num_vars, self.param_dim = num_vars, param_dim
        self.preprocess = MLP(num_vars + param_dim + 1, width * 2, width, n_layers=0, res=False)
        self.blocks = nn.ModuleList([
            Transolver_block(n_heads, width, 0.0, mlp_ratio=1, last_layer=(i == n_blocks-1),
                              out_dim=num_vars, slice_num=n_slices)
            for i in range(n_blocks)
        ])
        self.apply(self._init_weights)
        self.placeholder = nn.Parameter(torch.rand(width) / width)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, params):
        grid = joint_condition_grid(x, params, self.num_vars, self.param_dim, uniform=False).transpose(1, 2)
        h = self.preprocess(grid) + self.placeholder
        for block in self.blocks:
            h = block(h)
        return h.transpose(1, 2).contiguous()
