#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Operator Network (DeepONet)
================================
- Paper Title: Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators
- Authors: Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, George Em Karniadakis
- Journal: Nature Machine Intelligence 2021
- Paper Link: https://doi.org/10.1038/s42256-021-00302-5
- Official Code: https://github.com/lululxvi/deeponet

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Branch-Trunk Dual-Network Inner Product (Chen & Chen 1995 Operator Theorem)
- Mechanism: Branch net encodes biochemical kinetic parameters P into p_basis basis coefficients; Trunk net
  encodes temporal coordinates and initial conditions Y0. Output is inner product: y(t) = sum(Branch_k * Trunk_k).
- Biological Rationale: Decouples time trajectory basis functions from mutant-specific parameter amplitude weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet) for Yeast Biological Dynamics:
    - Branch Network: Encodes kinetic parameters and initial conditions into basis expansion coefficients.
    - Trunk Network: Learns continuous time-domain basis functions over the time grid t in [0, 1].
    - Output: One-shot tensor contraction producing the entire (B, 39, 500) trajectory.
    """
    def __init__(self, num_vars=39, param_dim=141, p_basis=128, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.num_vars = num_vars
        self.p_basis = p_basis

        if n_blocks < 1:
            raise ValueError('n_blocks must be positive')
        self.n_blocks = n_blocks
        def dense(input_dim, output_dim):
            layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
            for _ in range(n_blocks - 1):
                layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU()))
            return nn.Sequential(*layers, nn.Linear(hidden_dim, output_dim))
        self.branch_net = dense(param_dim + num_vars, num_vars * p_basis)
        self.trunk_net = dense(1, p_basis)

        # Learnable per-variable bias
        self.bias = nn.Parameter(torch.zeros(num_vars))

    def forward(self, ic_time_grid, params):
        """
        ic_time_grid: (B, num_vars, 2, T) where [:, :, 0, :] is IC and [:, :, 1, :] is time grid
        params: (B, param_dim)
        """
        B, V, F_in, T_steps = ic_time_grid.shape

        # 1. Extract initial condition vector (B, 39)
        ic_vec = ic_time_grid[:, :, 0, 0] # (B, num_vars)
        branch_in = torch.cat([params, ic_vec], dim=-1) # (B, param_dim + num_vars)

        # 2. Compute Branch coefficients (B, num_vars, p_basis)
        branch_out = self.branch_net(branch_in).reshape(B, V, self.p_basis)

        # 3. Extract time coordinates: (B, T_steps, 1)
        t_coords = ic_time_grid[:, 0, 1, :].unsqueeze(-1)  # (B, T_steps, 1)

        # 4. Compute Trunk continuous basis functions: (B, T_steps, p_basis)
        trunk_out = self.trunk_net(t_coords)  # (B, T_steps, p_basis)

        # 5. One-shot contraction: B_{b, v, k} * T_{b, t, k} -> (B, V, T_steps)
        out = torch.einsum('bvk, btk -> bvt', branch_out, trunk_out) + self.bias.unsqueeze(0).unsqueeze(-1)

        return out
