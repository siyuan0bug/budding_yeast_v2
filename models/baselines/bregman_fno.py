"""Bregman FNO: official dual-state postactivation recurrence, dense 1D kernels.

Configuration: linear zero-initialized spectral/local updates, no normalization
or optional channel MLP. This is the upstream use_mlp=False/norm=None case,
reduced to time and adapted by concatenating initial states and parameters.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .fno import FNO
from .common import joint_condition_grid


class BregmanFNO(FNO):
    def __init__(self, num_vars=39, param_dim=141, width=64, modes=24, n_blocks=4, padding=8):
        super().__init__(num_vars,param_dim,width,modes,n_blocks,padding)
        for spectral, local in zip(self.spectral,self.local):
            nn.init.zeros_(spectral.weight)
            nn.init.zeros_(local.weight)
            nn.init.zeros_(local.bias)

    def forward(self, x, params):
        h = self.lift(joint_condition_grid(x,params,self.num_vars,self.param_dim))
        if self.padding:
            h = F.pad(h,(0,self.padding))
        skip = h
        for i,(spectral,local) in enumerate(zip(self.spectral,self.local)):
            h = spectral(h) + local(h) + skip
            skip = h
            if i < len(self.spectral)-1:
                h = F.gelu(h)
        if self.padding:
            h = h[..., :-self.padding]
        return self.project(h)
