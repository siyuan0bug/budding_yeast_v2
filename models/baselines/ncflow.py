#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Context Flow (NCFlow)
============================
- Paper Title: Neural Context Flows for Meta-Learning of Dynamical Systems
- Authors: Roussel Desmond Nzoyem, David A.W. Barton, Tom Deakin
- Conference: ICLR 2025
- Repository: https://github.com/ddrous/ncflow / https://github.com/ddrous/ncflow-torch
- Core Concept: Continuous ODE vector field with Contextual Self-Modulation (Taylor expansion
  of the vector field about the environment context/parameter vector c), allowing adaptive flow
  prediction across diverse parametric regimes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualSelfModulatedODEFunc(nn.Module):
    """
    Contextual Self-Modulated Vector Field:
    dx/dt = f(x, t, c) + (c - c_base) * df/dc
    """
    def __init__(self, hidden_dim=128, context_dim=141, taylor_order=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.taylor_order = taylor_order

        # Base context anchor
        self.base_context = nn.Parameter(torch.zeros(context_dim))

        # Vector field network: takes (x, c, t) -> dx/dt
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + context_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t, x, context):
        # x: (B, hidden_dim), context: (B, context_dim), t: float or (B, 1)
        B = x.shape[0]
        if isinstance(t, (float, int)):
            t_tensor = torch.full((B, 1), float(t), device=x.device, dtype=x.dtype)
        elif t.ndim == 0:
            t_tensor = t.view(1, 1).expand(B, 1)
        else:
            t_tensor = t.view(B, 1)

        c_base = self.base_context.unsqueeze(0).expand(B, -1)
        delta_c = context - c_base

        if self.taylor_order == 0 or not self.training:
            # 0th-order direct evaluation
            inputs = torch.cat([x, context, t_tensor], dim=-1)
            return self.net(inputs)

        # 1st-order Contextual Taylor Self-Modulation
        c_anchor = c_base.clone().requires_grad_(True)
        inputs_anchor = torch.cat([x, c_anchor, t_tensor], dim=-1)
        vf_anchor = self.net(inputs_anchor)

        # Efficient directional derivative (JVP / Taylor term) via autograd
        v = torch.randn_like(vf_anchor)
        grad_c = torch.autograd.grad(vf_anchor, c_anchor, grad_outputs=v, create_graph=True, retain_graph=True)[0]
        taylor_term = (grad_c * delta_c).sum(dim=-1, keepdim=True).expand_as(vf_anchor)

        return vf_anchor + 0.1 * taylor_term


class NCFlow(nn.Module):
    """
    Neural Context Flow (NCFlow) for Yeast Cell Cycle
    """
    def __init__(self, num_vars=39, in_dim=2, param_dim=141, width=192, taylor_order=1, n_steps=500):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.n_steps = n_steps

        # State projector: maps initial state x0 and parameters p into latent trajectory space
        self.in_proj = nn.Linear(num_vars + param_dim, width)
        self.context_proj = nn.Linear(param_dim, width)

        self.vector_field = ContextualSelfModulatedODEFunc(
            hidden_dim=width,
            context_dim=width,
            taylor_order=taylor_order
        )

        # Multi-scale temporal integration layer (Euler / RK4 unrolled steps)
        self.temporal_refiner = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(width, width, 1)
        )
        self.out_proj = nn.Conv1d(width, num_vars, 1)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x0 = ic_time_grid[:, :, 0, 0]  # (B, num_vars)

        # Context vector
        ctx = F.silu(self.context_proj(params))  # (B, width)
        h0 = F.silu(self.in_proj(torch.cat([x0, params], dim=-1)))  # (B, width)

        # Continuous flow simulation across T steps using discrete integration
        dt = 1.0 / float(T)
        h_traj = [h0]
        h_curr = h0

        # Unrolled flow propagation (5 sub-sampling blocks with residual flow steps)
        sub_steps = 10
        step_stride = T // sub_steps
        coarse_traj = [h0]

        for s in range(sub_steps - 1):
            t_curr = float(s) / float(sub_steps)
            dh = self.vector_field(t_curr, h_curr, ctx)
            h_curr = h_curr + (1.0 / sub_steps) * dh
            coarse_traj.append(h_curr)

        # Interpolate coarse flow to full resolution T
        coarse_tensor = torch.stack(coarse_traj, dim=-1)  # (B, width, sub_steps)
        h_full = F.interpolate(coarse_tensor, size=T, mode='linear', align_corners=False)  # (B, width, T)

        # Refine continuous trajectory with contextual temporal convolutional operator
        h_out = h_full + self.temporal_refiner(h_full)
        out = self.out_proj(h_out)  # (B, num_vars, T)
        return out
