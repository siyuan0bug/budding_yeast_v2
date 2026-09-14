#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioRefiner-Causal & Causal Mutant Embedding Suite
=================================================
A causal representation learning (CPA-inspired) neural operator architecture
for the Budding Yeast Cell Cycle (Chen 2004) ODE system.

Key Innovations:
1. CausalMutantAdapter:
   - Built-in Chen (2004) Glucose Wild-Type (WT) baseline prior p_wt: (141,).
   - 27 Functional Gene Modules with sparse parameter index masks.
   - Dual linear & logarithmic intervention residual extraction:
     Delta_p = p - p_wt, log_scale = sign(Delta_p) * log1p(|Delta_p| / (|p_wt| + eps)).
   - Zero-bias modular sparse MLP projections ensuring e_g = 0 for unperturbed wild-type modules.
   - Directed mechanistic message passing along the 54-edge biological gene regulation graph
     (CLN3 -> SBF_MBF, CLB2 -> Mcm1, CDH1 -| CLB2, etc.).
   - Global causal summary fusion & residual parameter conditioning:
     p_causal = p + proj(concat([p, causal_summary])) preserving exact param_dim=141.
2. BioRefinerCausal Suite:
   - BioRefinerCausalBase: Two-stage coarse-to-fine operator with causal mutant conditioning.
   - BioRefinerCausalCurvature: Curvature acceleration + causal adapter.
   - BioRefinerCausalPyramid: Multi-scale temporal pyramid + causal adapter.
   - BioRefinerCausalGraph: Reaction graph cross-attention + causal adapter.
   - BioRefinerCausalPlus: Grand fusion (Curvature + Pyramid + Graph Attention + Causal Adapter).
   - BioRefinerCausal: Primary alias to BioRefinerCausalPlus.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bio_mechanistic_operator import BioMechPathwayOnly
from .bio_refiner_plus import (
    FiLMBlock1d,
    TemporalPyramidBlock,
    StoichiometricGraphAttention,
    ModularStiffRefiner,
)

# ==============================================================================
# 1. Chen (2004) 141-Parameter Metadata Prior & Biological Graph Topology
# ==============================================================================

# Sorted 141 parameter names corresponding to the standard dataset exports
PARAM_KEYS_141: List[str] = [
    'B0', 'BUB2_active', 'C0', 'Cdc15T', 'Dn3', 'ESP1T', 'J20ppx', 'Jacdh',
    'Jaiep', 'Jamcm', 'Jasbf', 'Jd2c1', 'Jd2f6', 'Jicdh', 'Jiiep', 'Jimcm',
    'Jisbf', 'Jitem', 'Jn3', 'Jpds', 'Jspn', 'Tem1T', 'Vi20_active', 'ebudb5',
    'ebudn2', 'ebudn3', 'ec1b2', 'ec1b5', 'ec1k2', 'ec1n2', 'ec1n3', 'ef6b2',
    'ef6b5', 'ef6k2', 'ef6n2', 'ef6n3', 'eicdhb2', 'eicdhb5', 'eicdhn2',
    'eicdhn3', 'eorib2', 'eorib5', 'esbfb5', 'esbfn2', 'esbfn3', 'init_BUB2',
    'init_CDH1', 'init_CDH1T', 'jatem', 'ka15P', 'ka15p', 'ka15pp', 'ka20p',
    'ka20pp', 'kacdhp', 'kacdhpp', 'kaiep', 'kamcm', 'kasb2', 'kasb5',
    'kasbf', 'kasesp', 'kasf2', 'kasf5', 'kasrent', 'kasrentp', 'kaswi',
    'kd14', 'kd1c1', 'kd1f6', 'kd1pdsp', 'kd20', 'kd2c1', 'kd2f6', 'kd2pdspp',
    'kd3c1', 'kd3f6', 'kd3pdspp', 'kdb2P', 'kdb2p', 'kdb2pp', 'kdb5p',
    'kdb5pp', 'kdbud', 'kdcdh', 'kdib2', 'kdib5', 'kdiesp', 'kdif2', 'kdif5',
    'kdirent', 'kdirentp', 'kdn2', 'kdnet', 'kdori', 'kdppxp', 'kdppxpp',
    'kdspn', 'kdswi', 'ki15', 'ki20p', 'ki20pp', 'kicdhp', 'kicdhpp', 'kiiep',
    'kimcm', 'kisbfp', 'kisbfpp', 'kiswi', 'kkpnetp', 'kkpnetpp', 'kppc1',
    'kppf6', 'kppnetp', 'kppnetpp', 'ks14', 'ks1pdspp', 'ks20p', 'ks20pp',
    'ks2pdspp', 'ksb2p', 'ksb2pp', 'ksb5p', 'ksb5pp', 'ksbud', 'ksc1p',
    'ksc1pp', 'kscdh', 'ksf6p', 'ksf6pp', 'ksf6ppp', 'ksn2p', 'ksn2pp',
    'ksnet', 'ksori', 'kspdsp', 'ksppx', 'ksspn', 'ksswip', 'ksswipp', 'mdt'
]

# Exact Chen (2004) Wild-Type (WT on Glucose) values for the 141 parameters
WT_VALUES_141: List[float] = [
    0.054, 1.0, 0.4, 1.0, 1.0, 1.0, 0.15, 0.03,
    0.1, 0.1, 0.01, 0.05, 0.05, 0.03, 0.1, 0.1,
    0.01, 0.1, 6.0, 0.04, 0.14, 1.0, 8.0, 1.0,
    0.25, 0.05, 0.45, 0.1, 0.03, 0.06, 0.3, 0.55,
    0.1, 0.03, 0.06, 0.3, 1.2, 8.0, 0.4,
    0.25, 0.45, 0.9, 2.0, 2.0, 10.0, 0.2,
    0.9304992, 1.0, 0.1, 0.001, 0.002, 1.0, 0.05,
    0.2, 0.01, 0.8, 0.1, 1.0, 50.0, 50.0,
    0.38, 50.0, 15.0, 0.01, 200.0, 1.0, 2.0,
    0.1, 0.01, 0.01, 0.01, 0.3, 1.0, 1.0, 0.2,
    1.0, 1.0, 0.04, 0.15, 0.003, 0.4, 0.01,
    0.16, 0.06, 0.01, 0.05, 0.06, 0.5, 0.5, 0.01,
    1.0, 2.0, 0.12, 0.03, 0.06, 0.17, 2.0,
    0.06, 0.08, 0.5, 0.01, 8.0, 0.001, 0.08, 0.15,
    0.15, 0.6, 8.0, 0.05, 0.01, 0.6, 4.0,
    4.0, 0.05, 3.0, 0.2, 0.03, 0.006, 0.6,
    0.055, 0.001, 0.04, 0.0008, 0.005, 0.2, 0.012,
    0.12, 0.01, 0.024, 0.12, 0.004, 0.0, 0.15,
    0.084, 2.0, 0.0, 0.1, 0.1, 0.005, 0.08, 90.0
]

# 27 Functional Gene Modules
MODULE_NAMES: List[str] = [
    'CLN2', 'CLN3', 'CLB2', 'CLB5', 'SIC1', 'CDC6', 'BCK2', 'SWI5',
    'CDC20', 'CDH1', 'CDC14', 'NET1', 'TEM1', 'CDC15', 'PPX', 'PDS1',
    'ESP1', 'MAD2', 'BUB2', 'SBF_MBF', 'Mcm1', 'APC_Core', 'ORI_BUD_SPN',
    'MDT', 'CDC5_Proxy', 'CDC28', 'SCC1'
]

# Parameter indices controlled by each gene module in PARAM_KEYS_141
MODULE_PARAM_INDICES: Dict[str, List[int]] = {
    'CLN2': [131, 132],
    'CLN3': [2],
    'CLB2': [120, 121, 79, 80, 78, 85],
    'CLB5': [122, 123, 82],
    'SIC1': [125, 126, 75, 72],
    'CDC6': [128, 129, 130],
    'BCK2': [0],
    'SWI5': [138, 139],
    'CDC20': [117, 118, 53],
    'CDH1': [127, 103, 47, 46],
    'CDC14': [115, 111, 112, 66, 55],
    'NET1': [133, 64, 65],
    'TEM1': [21, 51],
    'CDC15': [3, 110],
    'PPX': [136, 96, 114],
    'PDS1': [135, 116, 119, 74, 77],
    'ESP1': [5, 61, 87],
    'MAD2': [22],
    'BUB2': [1, 45],
    'SBF_MBF': [60, 107, 132, 123],
    'Mcm1': [57],
    'APC_Core': [53, 117, 118, 127, 47, 46],
    'ORI_BUD_SPN': [134, 124, 137],
    'MDT': [140],
    'CDC5_Proxy': [136],
    'CDC28': [28, 29, 27, 26, 30, 33, 34, 32, 31, 35, 38, 39, 37, 36, 43, 44, 42, 41, 40, 24, 25, 23],
    'SCC1': [61]
}

# 54 Directed Mechanistic Regulatory Edges: (source_gene, target_gene, sign)
# sign: '+' indicates activation/induction, '-' indicates inhibition/repression
MODULE_REGULATORY_EDGES: List[Tuple[str, str, str]] = [
    ('CLN3', 'SBF_MBF', '+'), ('BCK2', 'SBF_MBF', '+'), ('CLN2', 'SBF_MBF', '+'),
    ('CLB5', 'SBF_MBF', '+'), ('CLB2', 'SBF_MBF', '-'),
    ('SBF_MBF', 'CLN2', '+'), ('SBF_MBF', 'CLB5', '+'), ('SBF_MBF', 'CDC6', '+'),
    ('CLB2', 'Mcm1', '+'), ('Mcm1', 'CLB2', '+'), ('Mcm1', 'SWI5', '+'),
    ('Mcm1', 'CDC20', '+'), ('Mcm1', 'PDS1', '+'),
    ('CLB2', 'CDC20', '+'), ('CDC20', 'CLB2', '-'), ('CDC20', 'CLB5', '-'),
    ('CDC20', 'PDS1', '-'),
    ('CDH1', 'CLB2', '-'), ('CDH1', 'PDS1', '-'),
    ('CDC14', 'CDH1', '+'), ('CDC14', 'SIC1', '+'), ('CDC14', 'CDC6', '+'),
    ('CLB2', 'CDH1', '-'), ('CLB5', 'CDH1', '-'), ('CLN2', 'CDH1', '-'),
    ('CLN3', 'CDH1', '-'),
    ('SIC1', 'CLB2', '-'), ('SIC1', 'CLB5', '-'),
    ('CDC6', 'CLB2', '-'), ('CDC6', 'CLB5', '-'),
    ('SWI5', 'SIC1', '+'), ('SWI5', 'CDC6', '+'),
    ('NET1', 'CDC14', '-'), ('PPX', 'NET1', '+'), ('CDC15', 'NET1', '+'),
    ('TEM1', 'CDC15', '+'), ('BUB2', 'TEM1', '-'), ('MAD2', 'CDC20', '-'),
    ('PDS1', 'ESP1', '-'),
    ('CDC28', 'SIC1', '+'), ('CDC28', 'CDC6', '+'), ('CDC28', 'CDH1', '+'),
    ('CDC28', 'SBF_MBF', '+'), ('CDC28', 'ORI_BUD_SPN', '+'),
    ('CLB2', 'ORI_BUD_SPN', '+'), ('CLB5', 'ORI_BUD_SPN', '+'),
    # APC_Core regulation edges
    ('APC_Core', 'CLB2', '-'), ('APC_Core', 'PDS1', '-'),
    ('CDC20', 'APC_Core', '+'), ('CDH1', 'APC_Core', '+'),
    # Direct ODE kinetic regulation edges (Chen 2004 Eq 17 & 19)
    ('CLB2', 'SWI5', '-'), ('CLB2', 'SIC1', '-'),
    ('CLB5', 'SIC1', '-'), ('CLN2', 'SIC1', '-')
]


# ==============================================================================
# 2. 拓扑掩码多头图注意力层 (Directed Mechanistic Graph Message Passing)
# ==============================================================================

class CausalRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) without learnable bias.
    Strictly preserves zero perturbation (f(0) = 0 identically).
    
    Unlike standard LayerNorm, it does not subtract the feature mean and has no offset bias beta,
    preventing any shift drift during backpropagation and eliminating 0/sqrt(eps) numerical drift.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MechanisticGraphMessagePassing(nn.Module):
    """
    Topology-Constrained Directed Graph Attention Layer across the 27 Functional Gene Modules.
    
    Mutations and interventions at upstream regulators strictly propagate downstream
    only along biologically verified directed regulatory edges (A_target,source > 0).
    A signed topological bias incorporates activating (+) vs inhibitory (-) modulation.
    """
    def __init__(self, num_nodes: int = 27, d_gene: int = 16, num_heads: int = 4):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_gene = d_gene
        self.num_heads = num_heads
        assert d_gene % num_heads == 0, f"d_gene ({d_gene}) must be divisible by num_heads ({num_heads})"
        self.head_dim = d_gene // num_heads

        # Linear projections without bias to preserve zero-perturbation property
        self.q_proj = nn.Linear(d_gene, d_gene, bias=False)
        self.k_proj = nn.Linear(d_gene, d_gene, bias=False)
        self.v_proj = nn.Linear(d_gene, d_gene, bias=False)
        self.out_proj = nn.Linear(d_gene, d_gene, bias=False)

        # Learnable signed edge affinity scale
        self.sign_bias_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        
        # Zero-bias RMSNorm strictly guarantees f(0) = 0 (zero perturbation preservation)
        self.norm = CausalRMSNorm(d_gene)

    def forward(
        self,
        e: torch.Tensor,
        adj_mask: torch.Tensor,
        sign_adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            e: (B, num_nodes, d_gene) node perturbation embeddings
            adj_mask: (num_nodes, num_nodes) binary adjacency matrix (1 if edge exists or self-loop)
            sign_adj: (num_nodes, num_nodes) signed adjacency matrix (+1 for activation, -1 for repression)
        Returns:
            e_out: (B, num_nodes, d_gene) updated node representations
        """
        B, N, D = e.shape
        H = self.num_heads
        d_k = self.head_dim

        # Compute Q, K, V: (B, H, N, d_k)
        Q = self.q_proj(e).view(B, N, H, d_k).permute(0, 2, 1, 3)
        K = self.k_proj(e).view(B, N, H, d_k).permute(0, 2, 1, 3)
        V = self.v_proj(e).view(B, N, H, d_k).permute(0, 2, 1, 3)

        # Scaled dot-product attention scores: (B, H, N, N)
        # Note: row i represents target node (recipient), col j represents source node (transmitter)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)

        # Inject signed regulatory edge prior if available
        if sign_adj is not None:
            scores = scores + self.sign_bias_weight * sign_adj.unsqueeze(0).unsqueeze(0)

        # 1. Enforce diagonal self-loops so every node has at least 1 incoming edge (prevents in-degree=0 NaN)
        diag_eye = torch.eye(N, device=adj_mask.device, dtype=adj_mask.dtype)
        safe_adj = torch.clamp(adj_mask + diag_eye, 0.0, 1.0)

        # 2. Enforce strict biological topology masking:
        # Use -1e4 instead of -1e9 to avoid FP16 overflow/underflow
        mask = (safe_adj.unsqueeze(0).unsqueeze(0) == 0)
        scores = scores.masked_fill(mask, -1e4)

        # 3. Softmax with NaN guard
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, N, N)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Message aggregation: (B, H, N, d_k)
        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)  # (B, N, D)
        out = self.out_proj(out)

        # Residual connection with zero-preserving RMSNorm
        return self.norm(e + out)


# ==============================================================================
# 3. 因果突变嵌入适配器 (Causal Mutant Embedding Adapter)
# ==============================================================================

class CausalMutantAdapter(nn.Module):
    """
    Causal Mutant Embedding Adapter (CPA-Mechanistic Hybrid).
    
    Transforms the continuous 141-dimensional kinetic parameter vector p into a
    mechanism-invariant, compositionally additive causal representation p_causal: (B, 141).
    
    Workflow:
    1. Intervention Extraction:
       Computes both absolute difference Delta_p = p - p_wt and relative logarithmic
       residuals log_res = sign(Delta_p) * log1p(|Delta_p| / (|p_wt| + eps)).
    2. Modular Sparse Projections:
       Projects the intervention features for each of the 27 functional gene modules
       into a modular causal vector e_g in R^{d_gene} using un-biased local MLPs.
       Guarantees e_g = 0 identically whenever Delta_p_g = 0 (unperturbed gene).
    3. Mechanistic Message Passing:
       Propagates mutation signals strictly along the 54 biological regulatory edges
       using topology-constrained multi-head graph attention.
    4. Global Fusion:
       Aggregates updated gene representations, concatenates with raw p, and outputs
       a smooth residual enhancement: p_causal = p + proj([p, causal_summary]).
    """
    def __init__(
        self,
        param_dim: int = 141,
        d_gene: int = 16,
        num_heads: int = 4,
        embed_dim: Optional[int] = None,
        fusion_hidden_dim: int = 256
    ):
        super().__init__()
        self.param_dim = param_dim
        self.d_gene = d_gene
        self.num_modules = len(MODULE_NAMES)

        # 1. Register Chen (2004) Wild-Type on Glucose default parameter buffer
        p_wt_tensor = torch.tensor(WT_VALUES_141, dtype=torch.float32)
        if len(p_wt_tensor) != param_dim:
            if len(p_wt_tensor) > param_dim:
                p_wt_tensor = p_wt_tensor[:param_dim]
            else:
                p_wt_tensor = F.pad(p_wt_tensor, (0, param_dim - len(p_wt_tensor)))
        self.register_buffer('p_wt', p_wt_tensor)

        # 2. Build 27x27 directed regulatory adjacency and signed matrices
        adj = torch.zeros(self.num_modules, self.num_modules, dtype=torch.float32)
        sign_adj = torch.zeros(self.num_modules, self.num_modules, dtype=torch.float32)
        name_to_idx = {name: i for i, name in enumerate(MODULE_NAMES)}

        for src, dst, sign in MODULE_REGULATORY_EDGES:
            if src in name_to_idx and dst in name_to_idx:
                u = name_to_idx[src]
                v = name_to_idx[dst]
                # Directed edge u -> v: target v receives message from source u
                adj[v, u] = 1.0
                sign_adj[v, u] = 1.0 if sign == '+' else -1.0

        # Add self-loops (each gene retains its own primary perturbation state)
        for i in range(self.num_modules):
            adj[i, i] = 1.0
            sign_adj[i, i] = 0.5

        self.register_buffer('adj_topo', adj)
        self.register_buffer('sign_adj', sign_adj)

        # 3. Store parameter index masks for each module
        self.module_indices: List[List[int]] = []
        for name in MODULE_NAMES:
            raw_indices = MODULE_PARAM_INDICES.get(name, [])
            valid_indices = [idx for idx in raw_indices if idx < param_dim]
            self.module_indices.append(valid_indices)

        # 4. Modular Sparse MLP Projections (bias=False ensures f(0) = 0 strictly)
        self.gene_projections = nn.ModuleList()
        for idx_list in self.module_indices:
            in_dim = max(1, len(idx_list) * 2)  # [Delta_p, log_scaled_residual]
            mlp = nn.Sequential(
                nn.Linear(in_dim, d_gene * 2, bias=False),
                nn.GELU(),
                nn.Linear(d_gene * 2, d_gene, bias=False)
            )
            self.gene_projections.append(mlp)

        # 5. Directed Mechanistic Message Passing Layer
        self.message_passing = MechanisticGraphMessagePassing(
            num_nodes=self.num_modules,
            d_gene=d_gene,
            num_heads=num_heads
        )

        # 6. Global Causal Fusion Network
        summary_dim = self.num_modules * d_gene + d_gene * 2
        fusion_in_dim = param_dim + summary_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, param_dim)
        )

        # Zero-initialize final projection layer so p_causal initially equals p smoothly
        nn.init.zeros_(self.fusion_mlp[-1].weight)
        nn.init.zeros_(self.fusion_mlp[-1].bias)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Forward causal parameter adaptation.
        
        Args:
            p: (B, 141) continuous kinetic parameter vector
        Returns:
            p_causal: (B, 141) causally enriched parameter conditioning vector
        """
        B, P = p.shape
        device = p.device
        dtype = p.dtype

        p_wt_ref = self.p_wt.to(device=device, dtype=dtype).unsqueeze(0)  # (1, P)
        eps = 1e-5

        # 1. Dual Linear & Logarithmic Intervention Residuals
        delta_p = p - p_wt_ref  # (B, P)
        log_res = torch.sign(delta_p) * torch.log1p(torch.abs(delta_p) / (torch.abs(p_wt_ref) + eps))  # (B, P)

        # 2. Modular Sparse Projections
        gene_embeddings = []
        for i, idx_list in enumerate(self.module_indices):
            if len(idx_list) == 0:
                e_g = torch.zeros(B, self.d_gene, device=device, dtype=dtype)
            else:
                p_sub = delta_p[:, idx_list]   # (B, len(idx_list))
                log_sub = log_res[:, idx_list] # (B, len(idx_list))
                feat_g = torch.cat([p_sub, log_sub], dim=-1)  # (B, 2 * len(idx_list))
                e_g = self.gene_projections[i](feat_g)        # (B, d_gene)
            gene_embeddings.append(e_g)

        # Stack into graph node tensor: (B, 27, d_gene)
        E_init = torch.stack(gene_embeddings, dim=1)

        # 3. Mechanistic Topological Message Passing
        adj = self.adj_topo.to(device=device, dtype=dtype)
        sign_adj = self.sign_adj.to(device=device, dtype=dtype)
        E_updated = self.message_passing(E_init, adj, sign_adj)  # (B, 27, d_gene)

        # 4. Global Multi-Resolution Causal Aggregation
        e_flat = E_updated.reshape(B, self.num_modules * self.d_gene)  # (B, 27 * d_gene)
        e_mean = E_updated.mean(dim=1)                                  # (B, d_gene)
        e_max = E_updated.max(dim=1)[0]                                 # (B, d_gene)
        causal_summary = torch.cat([e_flat, e_mean, e_max], dim=-1)    # (B, summary_dim)

        # 5. Residual Fusion
        fusion_input = torch.cat([p, causal_summary], dim=-1)          # (B, 141 + summary_dim)
        delta_p_causal = self.fusion_mlp(fusion_input)                 # (B, 141)
        p_causal = p + delta_p_causal                                  # (B, 141)

        return p_causal


# ==============================================================================
# 4. 双阶因果基类 BioRefinerCausalBase
# ==============================================================================

class BioRefinerCausalBase(nn.Module):
    """
    Two-stage Coarse-to-Fine Neural Operator with Causal Mutant Embedding Adapter.
    
    Structure:
    - Frontend: CausalMutantAdapter transforms p -> p_causal (141)
    - Stage 1: BioMechPathwayOnly coarse trajectory predictor (conditioned on p_causal)
    - Intermediate: Computes 1st order derivative dy/dt and (optional) 2nd order curvature d2y/dt2
    - Stage 2: ModularStiffRefiner for stiff phase residual correction (conditioned on p_causal)
    - Output: Gated residual composition: y_final = y_coarse + tanh(gamma) * delta_y
    """
    def __init__(
        self,
        num_vars: int = 39,
        width: int = 128,
        modes: int = 32,
        param_dim: int = 141,
        n_blocks: int = 4,
        coarse_n_blocks: int = 4,
        use_curvature: bool = False,
        use_pyramid: bool = False,
        use_graph_attn: bool = False,
        d_gene: int = 16,
        adj_matrix: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.param_dim = param_dim
        self.use_curvature = use_curvature
        self.use_pyramid = use_pyramid
        self.use_graph_attn = use_graph_attn

        # Causal Mutant Embedding Adapter
        self.causal_adapter = CausalMutantAdapter(
            param_dim=param_dim,
            d_gene=d_gene,
            embed_dim=param_dim
        )

        # Stage 1: Coarse BioMech Pathway Base Operator
        self.stage1_coarse = BioMechPathwayOnly(
            num_vars=num_vars,
            width=width,
            modes=modes,
            param_dim=param_dim,
            n_blocks=coarse_n_blocks
        )

        # Compute Stage 2 input channel count:
        # Base: y_coarse (num_vars) + dy/dt (num_vars) + x_in (num_vars * 2) = num_vars * 4 = 156
        # If use_curvature: + d2y/dt2 (num_vars) = num_vars * 5 = 195
        in_channels = num_vars * 2 + num_vars * 2 + (num_vars if use_curvature else 0)

        # Stage 2: Modular Stiff Phase Residual Refiner
        self.stage2_refiner = ModularStiffRefiner(
            in_channels=in_channels,
            num_vars=num_vars,
            width=width,
            param_dim=param_dim,
            n_blocks=n_blocks,
            use_pyramid=use_pyramid,
            use_graph_attn=use_graph_attn,
            adj_matrix=adj_matrix
        )

        # Learnable gating scale (initialized to 0.1)
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _compute_1st_derivative(self, y: torch.Tensor) -> torch.Tensor:
        """Central finite differences for 1st-order temporal derivative."""
        d = torch.zeros_like(y)
        d[:, :, 1:-1] = (y[:, :, 2:] - y[:, :, :-2]) * 0.5
        d[:, :, 0] = y[:, :, 1] - y[:, :, 0]
        d[:, :, -1] = y[:, :, -1] - y[:, :, -2]
        return d

    def _compute_2nd_derivative(self, y: torch.Tensor) -> torch.Tensor:
        """Second-order central finite differences for curvature acceleration."""
        d2 = torch.zeros_like(y)
        d2[:, :, 1:-1] = y[:, :, 2:] - 2.0 * y[:, :, 1:-1] + y[:, :, :-2]
        d2[:, :, 0] = d2[:, :, 1]
        d2[:, :, -1] = d2[:, :, -2]
        return d2

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        return_coarse: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward prediction pipeline.
        
        Args:
            x: Trajectory input tensor, either (B, 39, T) or (B, 39, 2, T)
            p: (B, 141) continuous parameter vector
            return_coarse: Whether to return the Stage 1 coarse prediction alongside final
        Returns:
            y_final: (B, 39, T) refined biological state trajectory
            (optional) y_coarse: (B, 39, T)
        """
        # 1. Causal Mutant Embedding Adapter
        p_causal = self.causal_adapter(p)

        # Standardize input channels
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T)  # (B, 78, T)
        else:
            B, V, T = x.shape
            time_grid = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, time_grid], dim=1)  # (B, 78, T)

        # 2. Stage 1 Coarse Trajectory Prediction (conditioned on p_causal)
        y_coarse = self.stage1_coarse(x, p_causal)  # (B, 39, T)

        # 3. Extract derivative and curvature tensors
        dy_coarse = self._compute_1st_derivative(y_coarse)  # (B, 39, T)

        if self.use_curvature:
            d2y_coarse = self._compute_2nd_derivative(y_coarse)  # (B, 39, T)
            z_refine = torch.cat([y_coarse, dy_coarse, d2y_coarse, x_in], dim=1)  # (B, 195, T)
        else:
            z_refine = torch.cat([y_coarse, dy_coarse, x_in], dim=1)  # (B, 156, T)

        # 4. Stage 2 Stiff Phase Residual Correction (conditioned on p_causal)
        delta_y = self.stage2_refiner(z_refine, p_causal)  # (B, 39, T)

        # 5. Gated Residual Composition
        y_final = y_coarse + torch.tanh(self.gamma) * delta_y

        if return_coarse:
            return y_final, y_coarse
        return y_final


# ==============================================================================
# 5. 派生消融与旗舰模型 (Derived Ablation & Grand Fusion Models)
# ==============================================================================

class BioRefinerCausalCurvature(BioRefinerCausalBase):
    """消融模型 1: 因果适配器 + 二阶曲率/加速度特征 (Causal + Curvature)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_curvature=True, use_pyramid=False, use_graph_attn=False, **kwargs)


class BioRefinerCausalPyramid(BioRefinerCausalBase):
    """消融模型 2: 因果适配器 + 多尺度时间金字塔残差网络 (Causal + Pyramid)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_curvature=False, use_pyramid=True, use_graph_attn=False, **kwargs)


class BioRefinerCausalGraph(BioRefinerCausalBase):
    """消融模型 3: 因果适配器 + 生化反应拓扑图跨变量注意力 (Causal + Graph Attention)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_curvature=False, use_pyramid=False, use_graph_attn=True, **kwargs)


class BioRefinerCausalPlus(BioRefinerCausalBase):
    """
    终极因果全能旗舰 (Grand Causal Fusion Flagship):
    融合 Causal Adapter + 2nd-Order Curvature + Multi-Scale Temporal Pyramid + Reaction Graph Cross-Attention.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_curvature=True, use_pyramid=True, use_graph_attn=True, **kwargs)


# Primary alias pointing to the grand fusion model
BioRefinerCausal = BioRefinerCausalPlus
