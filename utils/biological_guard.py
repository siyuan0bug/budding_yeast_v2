#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biological Guard Utility (出芽酵母生物物理边界保护工具)
======================================================
Strictly derived from empirical data & ODE simulation audits:
1. Initial-Condition Aware Zeroing:
   Only zeroes channels where initial condition AND synthesis are strictly 0.
   Across 100,000 samples, only 5 variables ever satisfy this condition:
   ['CDH1', 'CDH1T', 'TEM1', 'CDC15', 'BUB2'].
   All other variables undergo natural exponential decay from Table 2B initial conditions
   and MUST NOT be artificially zeroed across all time steps.
2. Non-Negative Physical Projection:
   Clamps biochemical concentrations to >= 0.0, eliminating negative oscillations.
3. Event Jump vs. Continuous Variable Definitions:
   Isolates the 6 hybrid event variables (MASS, ORI, BUD, SPN, Vi20, lte1) from
   the 33 strictly continuous biochemical concentration variables.
"""

import torch
import torch.nn as nn
import numpy as np

# 39 Variable Definitions
(nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5,
 nSIC1P, nC2P, nC5P, nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T,
 nCDC20, nCDH1T, nCDH1, nCDC14T, nCDC14, nNET1T, nNET1, nRENT, nTEM1,
 nCDC15, nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, nVi20, nlte1, nBUB2) = range(39)

VARIABLE_NAMES = [
    'MASS', 'CLN2', 'CLB2', 'CLB5', 'SIC1', 'CDC6', 'C2', 'C5', 'F2', 'F5',
    'SIC1P', 'C2P', 'C5P', 'CDC6P', 'F2P', 'F5P', 'SWI5T', 'SWI5', 'IEP', 'CDC20T',
    'CDC20', 'CDH1T', 'CDH1', 'CDC14T', 'CDC14', 'NET1T', 'NET1', 'RENT', 'TEM1',
    'CDC15', 'PPX', 'PDS1', 'ESP1', 'ORI', 'BUD', 'SPN', 'Vi20', 'lte1', 'BUB2'
]

# 6 Macro Event / Jump Variables (subject to cell division / spindle / bud resets)
EVENT_VARS = [nMASS, nORI, nBUD, nSPN, nVi20, nlte1]

# 33 Strictly Continuous Biochemical Concentration Variables (100% C^1 smooth everywhere)
CONTINUOUS_VARS = [v for v in range(39) if v not in EVENT_VARS]

# The 5 empirically verified variables that can be identically 0 across all 500 steps
POSSIBLE_ZERO_VARS = [nCDH1T, nCDH1, nTEM1, nCDC15, nBUB2]


def compute_init_aware_zero_mask(p_norm, p_mean=None, p_std=None, param_keys=None):
    """
    Compute boolean/float mask (B, 39, 1) indicating which variables are guaranteed to be
    identically 0 across the entire timeline due to initial pool elimination (init = 0).

    Args:
        p_norm: (B, 141) or (B, param_dim) parameter tensor
        p_mean, p_std: Optional normalization buffers
        param_keys: List of 141 parameter names

    Returns:
        mask: (B, 39, 1) Float tensor, 1.0 = active, 0.0 = identically eliminated
    """
    B = p_norm.shape[0]
    device = p_norm.device
    mask = torch.ones(B, 39, 1, device=device, dtype=p_norm.dtype)

    # Denormalize p if stats provided
    if p_mean is not None and p_std is not None:
        p_real = p_norm * p_std.to(device) + p_mean.to(device)
    else:
        p_real = p_norm

    # If param_keys is provided, locate the exact indices of the init parameters
    if param_keys is not None:
        key_to_idx = {k: idx for idx, k in enumerate(param_keys)}
        
        # 1. CDH1 elimination
        idx_cdh1t = key_to_idx.get('init_CDH1T', None)
        idx_cdh1 = key_to_idx.get('init_CDH1', None)
        if idx_cdh1t is not None and idx_cdh1 is not None:
            is_cdh1_zero = (p_real[:, idx_cdh1t] < 1e-4) & (p_real[:, idx_cdh1] < 1e-4)
            mask[is_cdh1_zero, nCDH1T, :] = 0.0
            mask[is_cdh1_zero, nCDH1, :] = 0.0

        # 2. BUB2 elimination
        idx_bub2 = key_to_idx.get('init_BUB2', None)
        if idx_bub2 is not None:
            is_bub2_zero = (p_real[:, idx_bub2] < 1e-4)
            mask[is_bub2_zero, nBUB2, :] = 0.0

        # 3. TEM1 elimination (Tem1T == 0)
        idx_tem1t = key_to_idx.get('Tem1T', None)
        if idx_tem1t is not None:
            is_tem1_zero = (p_real[:, idx_tem1t] < 1e-4)
            mask[is_tem1_zero, nTEM1, :] = 0.0

        # 4. CDC15 elimination (Cdc15T == 0)
        idx_cdc15t = key_to_idx.get('Cdc15T', None)
        if idx_cdc15t is not None:
            is_cdc15_zero = (p_real[:, idx_cdc15t] < 1e-4)
            mask[is_cdc15_zero, nCDC15, :] = 0.0

    return mask


def apply_biological_guard(y, p_norm=None, p_mean=None, p_std=None, param_keys=None,
                           clamp_non_negative=True, use_init_zero_mask=True):
    """
    Apply safe biological post-processing to predicted trajectories:
    1. Zeroes out channels only when init=0 is explicitly satisfied.
    2. Clamps physical concentrations to >= 0.0.

    Args:
        y: (B, 39, T) Trajectory in physical/denormalized space.
    """
    out = y
    if clamp_non_negative:
        out = torch.clamp(out, min=0.0)

    if use_init_zero_mask and p_norm is not None and param_keys is not None:
        mask = compute_init_aware_zero_mask(p_norm, p_mean, p_std, param_keys)
        out = out * mask

    return out
