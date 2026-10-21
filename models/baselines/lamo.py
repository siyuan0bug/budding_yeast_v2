"""LaMO slice/Hydra/deslice operator using the upstream unfused Hydra core.

The small latent sequence uses exact reference SSD recurrence instead of a
compiled Mamba scan. No approximate SSM replacement or parameter modulation.
"""
import torch
from torch import nn
from .common import joint_condition_grid
from ._upstream.hydra import Hydra


class LatentSSM(nn.Module):
    def __init__(self, width, heads=4, slices=32, state_dim=64):
        super().__init__()
        if width % heads or heads % 2:
            raise ValueError('LaMO width must divide into an even number of heads')
        self.heads, self.dim_head, self.slices = heads, width // heads, slices
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * .5)
        self.in_project_x = nn.Linear(width, width)
        self.in_project_fx = nn.Linear(width, width)
        self.in_project_slice = nn.Linear(self.dim_head, slices)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.hydra = Hydra(width, d_state=state_dim, d_conv=3, expand=1,
                           headdim=self.dim_head, ngroups=2, use_mem_eff_path=False)
        self.to_out = nn.Linear(width, width)

    def forward(self, x):
        b, n, width = x.shape
        values = self.in_project_fx(x).reshape(b,n,self.heads,self.dim_head).transpose(1,2)
        features = self.in_project_x(x).reshape(b,n,self.heads,self.dim_head).transpose(1,2)
        weights = (self.in_project_slice(features) / self.temperature).softmax(-1)
        tokens = torch.einsum('bhnc,bhng->bhgc', values, weights)
        tokens = tokens / (weights.sum(2).unsqueeze(-1) + 1e-5)
        tokens = tokens.transpose(1,2).reshape(b,self.slices,width)
        tokens = self.hydra(tokens).reshape(b,self.slices,self.heads,self.dim_head).transpose(1,2)
        out = torch.einsum('bhgc,bhng->bhnc', tokens, weights).transpose(1,2).reshape(b,n,width)
        return self.to_out(out)


class LaMO(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4, n_heads=4, n_slices=32):
        super().__init__()
        if n_blocks < 1:
            raise ValueError('n_blocks must be positive')
        self.num_vars, self.param_dim = num_vars, param_dim
        self.stem = nn.Sequential(nn.Linear(num_vars+param_dim+1,width*2),nn.GELU(),nn.Linear(width*2,width))
        self.blocks = nn.ModuleList([
            nn.ModuleDict({'norm1':nn.LayerNorm(width),'ssm':LatentSSM(width,n_heads,n_slices),
                           'norm2':nn.LayerNorm(width),
                           'mlp':nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))})
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,num_vars))

    def forward(self, x, params):
        h = self.stem(joint_condition_grid(x,params,self.num_vars,self.param_dim,uniform=False).transpose(1,2))
        for block in self.blocks:
            h = h + block['ssm'](block['norm1'](h))
            h = h + block['mlp'](block['norm2'](h))
        return self.head(h).transpose(1,2).contiguous()
