"""GNOT with official heterogeneous normalized attention and geometric experts.

Only graph batching is replaced by dense temporal query tensors. The two source
branches encode all initial states and all parameters; neither is a new module.
"""
from torch import nn
from .common import joint_condition_grid
from ._upstream.gnot import MoEGPTConfig, MIOECrossAttentionBlock


class Projection(nn.Module):
    """The residual projection MLP used by the official GNOT."""
    def __init__(self, input_dim, width, output_dim, layers=2):
        super().__init__()
        self.linear_pre = nn.Linear(input_dim, width)
        self.linears = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.linear_post = nn.Linear(width, output_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.linear_pre(x))
        for layer in self.linears:
            x = x + self.act(layer(x))
        return self.linear_post(x)


class GNOT(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4, n_heads=4, n_experts=2):
        super().__init__()
        if width % n_heads or n_blocks < 1:
            raise ValueError('width must divide into attention heads; n_blocks must be positive')
        self.num_vars, self.param_dim = num_vars, param_dim
        self.trunk = Projection(1 + param_dim, width, width)
        self.branch = nn.ModuleList([Projection(num_vars, width, width), Projection(param_dim, width, width)])
        config = MoEGPTConfig(n_embd=width, n_head=n_heads, n_layer=n_blocks,
                              n_experts=n_experts, space_dim=1, n_inputs=2, n_inner=4)
        self.blocks = nn.ModuleList([MIOECrossAttentionBlock(config) for _ in range(n_blocks)])
        self.out = Projection(width, width, num_vars)

    def forward(self, x, params):
        grid = joint_condition_grid(x, params, self.num_vars, self.param_dim, uniform=False).transpose(1, 2)
        pos = grid[..., -1:]
        h = self.trunk(grid[..., self.num_vars:])
        sources = [self.branch[0](grid[:, :1, :self.num_vars]), self.branch[1](params.unsqueeze(1))]
        for block in self.blocks:
            h = block(h, sources, pos)
        return self.out(h).transpose(1, 2).contiguous()
