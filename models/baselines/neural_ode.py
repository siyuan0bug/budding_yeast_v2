#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Ordinary Differential Equations (NeuralODEModel)
======================================================
- Paper Title: Neural Ordinary Differential Equations
- Authors: Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
- Conference: NeurIPS 2018 (Best Paper Award)
- Paper Link: https://arxiv.org/abs/1806.07366
- Official Code: https://github.com/rtqichen/torchdiffeq

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: Parameterized Vector Field Concatenation
- Mechanism: Vector field network f_theta(t, y; P) takes current concentration state y concatenated with
  parameter vector P, integrated over [0, T] via adaptive Runge-Kutta / Dormand-Prince (dopri5) ODE solver.
- Biological Rationale: Directly mimics continuous biophysical reaction rate derivative dy/dt = f(y; P).
"""

import torch
import torch.nn as nn
from .common import joint_condition_grid

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None


class DerivativeMLP(nn.Module):
    def __init__(self, state_dim=39, param_dim=141, hidden_dim=128, n_blocks=2):
        super().__init__()
        if n_blocks < 1:
            raise ValueError('n_blocks must be positive')
        layers = [nn.Linear(state_dim + param_dim + 1, hidden_dim), nn.Softplus()]
        for _ in range(n_blocks - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.Softplus()))
        self.net = nn.Sequential(*layers, nn.Linear(hidden_dim, state_dim))
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, y, params):
        if isinstance(t, (float, int)):
            t = torch.full((y.size(0), 1), t, device=y.device)
        elif t.dim() == 0:
            t = t.expand(y.size(0), 1)
        inp = torch.cat([y, params, t], dim=-1)
        return self.net(inp)


class NeuralODEModel(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, hidden_dim=128,
                 method='dopri5', rtol=1e-3, atol=1e-4, n_blocks=2):
        super().__init__()
        if odeint is None:
            raise ImportError("Please install torchdiffeq first: pip install torchdiffeq")

        self.num_vars = num_vars
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.method = method
        self.rtol = rtol
        self.atol = atol

        self.derivative_net = DerivativeMLP(
            state_dim=num_vars, param_dim=param_dim, hidden_dim=hidden_dim, n_blocks=n_blocks
        )

    def forward(self, ic_time_grid, params):
        B, V, _, T = ic_time_grid.shape

        y0 = ic_time_grid[:, :, 0, 0]

        joint_condition_grid(ic_time_grid, params, self.num_vars, self.param_dim, uniform=False)
        times = ic_time_grid[:, 0, 1, :]
        if not torch.equal(times, times[:1].expand_as(times)):
            return torch.cat([self.forward(ic_time_grid[i:i+1], params[i:i+1]) for i in range(B)])
        t_grid = times[0]

        pred_y = odeint(
            lambda t, y: self.derivative_net(t, y, params),
            y0, t_grid,
            method=self.method, rtol=self.rtol, atol=self.atol
        )

        pred_y = pred_y.permute(1, 2, 0)
        return pred_y
