"""Official LNO PhCA encoder/decoder and latent attention with ODE task I/O."""
from torch import nn
from .common import joint_condition_grid
from ._upstream.lno import LNO as CoreLNO


class LNO(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4, n_heads=4, n_modes=32):
        super().__init__()
        if width % n_heads or n_blocks < 1:
            raise ValueError('Invalid attention width or block count')
        self.num_vars, self.param_dim = num_vars, param_dim
        self.core = CoreLNO(n_blocks, n_modes, width, n_heads, 2, 1,
                           num_vars + param_dim + 1, num_vars,
                           'Attention_Vanilla', 'GELU', {'time':False})

    def forward(self, x, params):
        grid = joint_condition_grid(x, params, self.num_vars, self.param_dim, uniform=False).transpose(1, 2)
        # Time is part of both the location and input representation: a constant
        # branch replicated at all locations would collapse the PhCA modes.
        return self.core(grid[..., -1:], grid).transpose(1, 2).contiguous()
