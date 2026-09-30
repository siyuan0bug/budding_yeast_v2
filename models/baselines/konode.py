#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Koopman Neural Ordinary Differential Equations (KoNODE)
=======================================================
- Paper Title: Koopman Neural Ordinary Differential Equations
- Core Concept: Bridges Koopman operator theory and Neural ODEs by embedding non-linear
  biological state space into a linear observable space with continuous spectrum evolution,
  avoiding stiffness singularities while retaining non-linear decoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KoNODE(nn.Module):
    """
    Koopman Neural Ordinary Differential Equation (KoNODE) for Yeast Cell Cycle
    """
    def __init__(self, num_vars=39, in_dim=2, param_dim=141, width=192, koopman_modes=32, n_layers=3):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.koopman_modes = koopman_modes

        # 1. Observable Encoder: maps initial states x0 and parameters p into Koopman coordinates
        self.encoder = nn.Sequential(
            nn.Linear(num_vars + param_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU()
        )

        # Spectral generator: predicts real and imaginary eigenvalues (growth/decay & oscillation frequency)
        self.spec_real = nn.Linear(width, koopman_modes)
        self.spec_imag = nn.Linear(width, koopman_modes)
        self.koopman_init = nn.Linear(width, koopman_modes * 2)

        # 2. Dynamic continuous-time decoder: decodes continuous Koopman trajectory w(t) back to 39 variables
        self.decoder_in = nn.Linear(koopman_modes * 2 + width, width)
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, 5, padding=2),
                nn.SiLU(),
                nn.Conv1d(width, width, 1)
            ) for _ in range(n_layers)
        ])
        self.out_proj = nn.Conv1d(width, num_vars, 1)

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        x0 = ic_time_grid[:, :, 0, 0]  # Initial condition (B, num_vars)

        # Encode (x0, params)
        cond = torch.cat([x0, params], dim=-1)
        h_cond = self.encoder(cond)  # (B, width)

        # Predict continuous Koopman spectrum and initial coordinates
        # Clamp real spectrum for biological Lyapunov stability (negative/zero for bounded limit cycles)
        re = -F.softplus(self.spec_real(h_cond)) * 0.5  # (B, K)
        im = self.spec_imag(h_cond) * 2.0 * math.pi      # (B, K)

        w0 = self.koopman_init(h_cond)  # (B, 2*K)
        w0_re = w0[:, :self.koopman_modes]
        w0_im = w0[:, self.koopman_modes:]

        # Continuous-time evaluation grid t in [0, 1]
        t = torch.linspace(0.0, 1.0, T, device=x0.device).view(1, 1, T)  # (1, 1, T)

        # Exact analytical continuous evolution of Koopman observables:
        # w(t) = exp(re * t) * [cos(im * t) + i * sin(im * t)] * w0
        decay = torch.exp(re.unsqueeze(-1) * t)  # (B, K, T)
        phase = im.unsqueeze(-1) * t             # (B, K, T)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        w_t_re = decay * (w0_re.unsqueeze(-1) * cos_phase - w0_im.unsqueeze(-1) * sin_phase)
        w_t_im = decay * (w0_re.unsqueeze(-1) * sin_phase + w0_im.unsqueeze(-1) * cos_phase)
        w_t = torch.cat([w_t_re, w_t_im], dim=1)  # (B, 2*K, T)

        # Broadcast static context across time
        h_static = h_cond.unsqueeze(-1).expand(-1, -1, T)  # (B, width, T)
        features = torch.cat([w_t, h_static], dim=1)      # (B, 2*K + width, T)

        h = F.silu(self.decoder_in(features.permute(0, 2, 1)).permute(0, 2, 1))

        # Dynamic continuous decoding
        for layer in self.decoder_layers:
            h = h + layer(h)

        out = self.out_proj(h)  # (B, num_vars, T)
        return out
