#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Biochemical Reaction Flux Operator (BioFluxOperator)
================================================================================
Implements:
1. UniversalBiochemicalGate: Causal parent-to-child gating based on the 27-gene network.
2. MultiplicativeFluxHead: Strict multiplicative coupling between parameters and activation states:
   Flux_v(t) = Basal_v(p) + Scale_v(p) * (Gate_v(p) * Act_v(t))
Zero-leakage, mathematically general across all parameter vectors p in R^141.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

# 27 核心生化功能模块
MODULE_NAMES = [
    'CLN2', 'CLB2', 'CLB5', 'SIC1', 'CDC6', 'SWI5', 'CDC20', 'CDH1',
    'CDC14', 'NET1', 'TEM1', 'CDC15', 'PPX', 'PDS1', 'ESP1', 'MAD2',
    'BUB2', 'SBF_MBF', 'Mcm1', 'APC_Core', 'ORI_BUD_SPN', 'MDT',
    'CLN3', 'BCK2', 'CDC5_Proxy', 'CDC28', 'SCC1'
]

# 核心生化合成/基础激活参数映射 (索引参考 data/mutant_aware.py 141 维参数字典)
# 这里仅定义通用调控反应法则，对所有 100k 样本和任意突变体统一生效
PARAM_INDICES_MAP = {
    'B0': 0,        # Bck2
    'C0': 2,        # Cln3
    'ks14': 115,    # Cdc14 合成
    'ks20p': 117,   # Cdc20 基础转录
    'ks20pp': 118,  # Cdc20 Mcm1 依赖转录
    'ksb2p': 120,   # Clb2 基础转录
    'ksb2pp': 121,  # Clb2 Mcm1 依赖转录
    'ksb5p': 122,   # Clb5 基础转录
    'ksb5pp': 123,  # Clb5 SBF 依赖转录
    'ksc1p': 125,   # Sic1 基础转录
    'ksc1pp': 126,  # Sic1 Swi5 依赖转录
    'kscdh': 127,   # Cdh1 合成
    'ksf6p': 128,   # Cdc6 基础转录
    'ksf6pp': 129,  # Cdc6 Swi5 依赖转录
    'ksf6ppp': 130, # Cdc6 SBF 依赖转录
    'ksn2p': 131,   # Cln2 基础/外源转录
    'ksn2pp': 132,  # Cln2 SBF 依赖转录
    'ksnet': 133,   # Net1 合成
    'mdt': 140      # 倍增时间
}


class UniversalBiochemicalGate(nn.Module):
    """
    通用生化因果门控:
    根据 27 节点调控因果拓扑，对各主要合成通量进行前驱驱动连续门控。
    当且仅当输入参数中的前驱驱动为 0 时，门控软截断为 0。
    """
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, p: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        p: (B, 141) 连续动力学参数
        Returns:
            gates: Dict of gating scalars in [0, 1] of shape (B, 1)
        """
        gates = {}
        B = p.shape[0]

        # 1. SBF/MBF 必需启动子前驱驱动 (Cln3, Bck2, 以及自身基础泄漏)
        c0 = p[:, PARAM_INDICES_MAP['C0']:PARAM_INDICES_MAP['C0']+1].abs()
        b0 = p[:, PARAM_INDICES_MAP['B0']:PARAM_INDICES_MAP['B0']+1].abs()
        # 若 C0 和 B0 均接近 0，则驱动门控逼近 0
        upstream_sbf_drive = (c0 + b0) / (c0 + b0 + self.eps)
        gates['SBF_drive'] = upstream_sbf_drive.clamp(0.0, 1.0)

        # 2. Cln2 转录通量前驱门控
        # 若外源转录 ksn2p > eps, 或 SBF 上游驱动有效，则允许合成
        ksn2p = p[:, PARAM_INDICES_MAP['ksn2p']:PARAM_INDICES_MAP['ksn2p']+1].abs()
        ksn2pp = p[:, PARAM_INDICES_MAP['ksn2pp']:PARAM_INDICES_MAP['ksn2pp']+1].abs()
        
        # 显式基础通量与可诱导通量尺度
        cln2_inducible_gate = gates['SBF_drive'] * torch.tanh(ksn2pp / self.eps)
        cln2_basal_gate = torch.tanh(ksn2p / self.eps)
        gates['CLN2_total_gate'] = (cln2_basal_gate + cln2_inducible_gate).clamp(0.0, 1.0)

        # 3. Clb5 转录通量门控 (依赖 SBF)
        ksb5p = p[:, PARAM_INDICES_MAP['ksb5p']:PARAM_INDICES_MAP['ksb5p']+1].abs()
        ksb5pp = p[:, PARAM_INDICES_MAP['ksb5pp']:PARAM_INDICES_MAP['ksb5pp']+1].abs()
        clb5_inducible_gate = gates['SBF_drive'] * torch.tanh(ksb5pp / self.eps)
        clb5_basal_gate = torch.tanh(ksb5p / self.eps)
        gates['CLB5_total_gate'] = (clb5_basal_gate + clb5_inducible_gate).clamp(0.0, 1.0)

        # 4. Cdc20 合成通量门控 (依赖 ks20p + ks20pp)
        ks20p = p[:, PARAM_INDICES_MAP['ks20p']:PARAM_INDICES_MAP['ks20p']+1].abs()
        ks20pp = p[:, PARAM_INDICES_MAP['ks20pp']:PARAM_INDICES_MAP['ks20pp']+1].abs()
        cdc20_gate = torch.tanh((ks20p + ks20pp) / self.eps)
        gates['CDC20_gate'] = cdc20_gate.clamp(0.0, 1.0)

        # 5. Cdc14 合成通量门控
        ks14 = p[:, PARAM_INDICES_MAP['ks14']:PARAM_INDICES_MAP['ks14']+1].abs()
        gates['CDC14_gate'] = torch.tanh(ks14 / self.eps).clamp(0.0, 1.0)

        # 6. Net1 合成通量门控
        ksnet = p[:, PARAM_INDICES_MAP['ksnet']:PARAM_INDICES_MAP['ksnet']+1].abs()
        gates['NET1_gate'] = torch.tanh(ksnet / self.eps).clamp(0.0, 1.0)

        return gates


class MultiplicativeFluxHead(nn.Module):
    """
    因果通量乘法耦合输出头:
    对 39 个变量中的核心生化产物施加严格的参数乘法通量约束:
    y_out[v] = Base[v] + Dynamic[v] * Gate[v]
    """
    def __init__(self, num_vars: int = 39, width: int = 192):
        super().__init__()
        self.num_vars = num_vars
        self.gate_engine = UniversalBiochemicalGate()

        # 变量到特定生化门控的映射
        # 1: CLN2, 3: CLB5, 19: CDC20T, 20: CDC20, 23: CDC14T, 24: CDC14, 25: NET1T, 26: NET1
        self.var_gate_keys = {
            1: 'CLN2_total_gate',
            3: 'CLB5_total_gate',
            19: 'CDC20_gate',
            20: 'CDC20_gate',
            23: 'CDC14_gate',
            24: 'CDC14_gate',
            25: 'NET1_gate',
            26: 'NET1_gate'
        }

    def forward(self, y_raw: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        y_raw: (B, 39, T) 神经网络主干输出的粗预测
        p: (B, 141) 参数矢量
        Returns:
            y_modulated: (B, 39, T) 施加生化因果通量门控后的严格物理约束预测
        """
        B, V, T = y_raw.shape
        gates = self.gate_engine(p)  # 各门控 shape (B, 1)

        # 构建全变量因果门控矩阵 (B, 39, 1)
        causal_mask = torch.ones((B, V, 1), device=y_raw.device, dtype=y_raw.dtype)

        for var_idx, gate_key in self.var_gate_keys.items():
            if gate_key in gates:
                causal_mask[:, var_idx:var_idx+1, :] = gates[gate_key].unsqueeze(-1)

        # 施加乘法门控 (保证当门控为0时，变量全时程严格置0)
        y_modulated = y_raw * causal_mask
        return y_modulated
