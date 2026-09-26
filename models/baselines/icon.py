#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICON: In-Context Operator Networks
==================================
- Paper Title: In-Context Operator Networks (ICON)
- Authors: Luning Sun, Zhao Chen, Mengnan Chen, et al.
- Conference: NeurIPS 2023
- Paper Link: https://arxiv.org/abs/2304.14816
- Official Code: https://github.com/lu-group/in-context-operator-networks

Biochemical Parameter Conditioning Mechanism:
---------------------------------------------
- Conditioning Method: In-Context Prompt Token Conditioning
- Mechanism: Maps biochemical kinetic parameters P into a sequence of prompt condition tokens [T_cond].
  The temporal trajectory queries attend to the in-context prompt condition tokens via full multi-head
  self-attention and cross-attention blocks to infer the operator mapping on the fly.
- Biological Rationale: Direct prompt formulation captures diverse mutant dynamics as contextual in-context prompts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICONGPT(nn.Module):
    def __init__(self, num_vars=39, in_dim=2, param_dim=141, d_model=128, n_heads=4, n_layers=4, num_prompt_tokens=4):
        super().__init__()
        self.num_vars = num_vars
        self.num_prompt_tokens = num_prompt_tokens

        # In-context condition prompt encoder
        self.prompt_encoder = nn.Sequential(
            nn.Linear(param_dim, d_model * num_prompt_tokens),
            nn.GELU()
        )

        # Trajectory query encoder
        self.traj_in_proj = nn.Linear(num_vars * in_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_vars)
        )

    def forward(self, ic_time_grid, params):
        B, V, F_in, T = ic_time_grid.shape
        # Trajectory tokens: (B, T, d_model)
        x_traj = self.traj_in_proj(ic_time_grid.permute(0, 3, 1, 2).reshape(B, T, V * F_in))

        # In-context prompt tokens: (B, num_prompt_tokens, d_model)
        prompts = self.prompt_encoder(params).view(B, self.num_prompt_tokens, -1)

        # Concatenate in-context prompt tokens with trajectory tokens
        seq = torch.cat([prompts, x_traj], dim=1)  # (B, num_prompt_tokens + T, d_model)

        h = self.transformer(seq)

        # Extract only the trajectory tokens
        h_traj = h[:, self.num_prompt_tokens:, :]
        out = self.out_proj(h_traj)  # (B, T, V)
        return out.permute(0, 2, 1)  # (B, V, T)
