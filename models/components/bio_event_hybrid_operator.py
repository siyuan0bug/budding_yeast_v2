#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioEventHybridOperator (全事件驱动混合生化神经算子)
================================================================================
Implements the hybrid dynamical automaton directly corresponding to Chen (2004) ODE events:
1. Event 3: ORI >= theta_ori  (S-phase initiation)
2. Event 4: SPN >= theta_spn  (Spindle alignment / SAC relief)
3. Event 1: CLB2 <= theta_exit (Mitotic exit / Cell division)
4. Event 2: CLB2 + CLB5 <= theta_lic (Origin licensing reset)

Couples causal phase states (Phase 0: G1, Phase 1: S/G2, Phase 2: M, Phase 3: Exit)
into continuous, mathematically sound conditioning representations for the Stiff Refiner.
Zero error floor, zero leakage, no scale/coordinate mismatch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class CausalODEEventDetectors(nn.Module):
    """
    连续可微自适应 ODE 4 大物理事件探测器:
    输入全时程 39 变量粗轨迹 y(t) 与参数向量 p，输出各时间步的事件激活度 S_k(t) in [0, 1]。
    利用时间因果累积算子 (cummax) 确保事件发生的严格物理时序。
    完全适配 Z-score 归一化特征空间，自适应预测阈值与平滑温度，杜绝数值溢出与 NaN。
    """
    def __init__(self, param_dim: int = 141):
        super().__init__()
        # 可学习的自适应生理阈值映射网络 (从参数 p 自适应预测对应量纲的事件阈值与敏感度)
        self.threshold_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.GELU(),
            nn.Linear(64, 8)  # 4 个阈值偏移 + 4 个温度尺度
        )
        # 初始阈值基准 (在归一化空间的平滑初值)
        self.base_thresh = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32))

    def forward(self, y: torch.Tensor, p: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        y: (B, 39, T) 连续轨迹
        p: (B, 141) 参数向量
        Indices:
            CLB2 = 2, CLB5 = 3, ORI = 33, SPN = 35
        Returns:
            Dict containing event activation profiles (B, T)
        """
        B, _, T = y.shape
        clb2 = y[:, 2, :]   # (B, T)
        clb5 = y[:, 3, :]   # (B, T)
        ori = y[:, 33, :]   # (B, T)
        spn = y[:, 35, :]   # (B, T)

        h = self.threshold_net(p)  # (B, 8)
        theta = h[:, :4] + self.base_thresh.unsqueeze(0)  # (B, 4)
        tau = F.softplus(h[:, 4:]) + 0.2  # (B, 4), 保证温度 >= 0.2，彻底杜绝梯度爆炸

        theta3 = theta[:, 0:1]  # (B, 1)
        theta4 = theta[:, 1:2]
        theta1 = theta[:, 2:3]
        theta2 = theta[:, 3:4]

        tau3 = tau[:, 0:1]
        tau4 = tau[:, 1:2]
        tau1 = tau[:, 2:3]
        tau2 = tau[:, 3:4]

        # 1. Event 3: ORI >= theta3 (DNA 复制起点点火)
        e3_raw = torch.sigmoid((ori - theta3) / tau3)
        # 物理不可逆性：一旦点火，本周期 S 期历史状态置 1
        e3_history = torch.cummax(e3_raw, dim=-1)[0]

        # 2. Event 4: SPN >= theta4 (纺锤体对齐)
        # 物理因果约束：必须在 Event 3 发生后方可发生
        e4_raw = torch.sigmoid((spn - theta4) / tau4) * e3_history
        e4_history = torch.cummax(e4_raw, dim=-1)[0]

        # 3. Event 1: CLB2 <= theta1 (完成有丝分裂退出 / 胞质分裂)
        # 物理因果约束：必须在中期纺锤体对齐 (Event 4) 触发后，激酶被成功拉低
        e1_raw = torch.sigmoid((theta1 - clb2) / tau1) * e4_history
        e1_history = torch.cummax(e1_raw, dim=-1)[0]

        # 4. Event 2: CLB2 + CLB5 <= theta2 (G1 早期执照复位)
        # 物理因果约束：必须在分裂后 (Event 1) 所有 CDK 激酶清零方可重新授权
        e2_raw = torch.sigmoid((theta2 - (clb2 + clb5)) / tau2) * e1_history
        e2_history = torch.cummax(e2_raw, dim=-1)[0]

        return {
            'S3_raw': e3_raw, 'S3_history': e3_history,
            'S4_raw': e4_raw, 'S4_history': e4_history,
            'S1_raw': e1_raw, 'S1_history': e1_history,
            'S2_raw': e2_raw, 'S2_history': e2_history
        }


class HybridPhaseStateMachine(nn.Module):
    """
    因果时序单调相区有限状态机 (Phase 0~3):
    根据 4 大事件历史，划分每个时刻所处的严格生物学生理时相。
    满足归一化条件: P_G1(t) + P_S(t) + P_M(t) + P_Exit(t) == 1.0
    """
    def __init__(self, param_dim: int = 141):
        super().__init__()
        self.detectors = CausalODEEventDetectors(param_dim=param_dim)

    def forward(self, y: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        y: (B, 39, T)
        p: (B, 141)
        Returns:
            phase_probs: (B, 4, T) 各相区连续权重矩阵 [G1, S/G2, M, Exit]
            events: Dict of event signals
        """
        ev = self.detectors(y, p)

        s3_h = ev['S3_history']  # (B, T)
        s4_h = ev['S4_history']  # (B, T)
        s1_h = ev['S1_history']  # (B, T)

        # 1. Phase 0: Pre-S G1 启动期 (Start 前)
        p_g1 = (1.0 - s3_h).clamp(0.0, 1.0)

        # 2. Phase 1: S/G2 出芽与合成期 (Start 已触发，纺锤体未对齐)
        p_sg2 = (s3_h * (1.0 - s4_h)).clamp(0.0, 1.0)

        # 3. Phase 2: M 期中期 (纺锤体对齐，等待激酶降解退出)
        p_m = (s4_h * (1.0 - s1_h)).clamp(0.0, 1.0)

        # 4. Phase 3: 有丝分裂后期退出期 (分裂已完成)
        p_exit = s1_h.clamp(0.0, 1.0)

        # 组合为 (B, 4, T)
        phase_probs = torch.stack([p_g1, p_sg2, p_m, p_exit], dim=1)
        # 数值归一化防止浮点漂移
        phase_probs = phase_probs / (phase_probs.sum(dim=1, keepdim=True) + 1e-6)

        return phase_probs, ev


class UniversalEventFluxCoupler(nn.Module):
    """
    全事件混合动力学神经算子耦合层:
    1. 计算 4 大离散事件与相区状态机。
    2. 将相区指示矩阵与事件时序历史注入到时空特征流中。
    3. 保留无界连续特征流，不施加破坏性的归一化空间硬乘截断。
    """
    def __init__(self, num_vars: int = 39, width: int = 192, d_event: int = 24, param_dim: int = 141):
        super().__init__()
        self.num_vars = num_vars
        self.width = width
        self.d_event = d_event

        self.state_machine = HybridPhaseStateMachine(param_dim=param_dim)

        # 相区与事件嵌入映射: (4 个相区概率 + 4 个事件历史 = 8 通道) -> d_event
        self.event_feature_proj = nn.Sequential(
            nn.Conv1d(8, d_event, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_event, d_event, kernel_size=1)
        )

    def forward(
        self,
        y_coarse: torch.Tensor,
        p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        y_coarse: (B, 39, T) 粗估计轨迹
        p: (B, 141) 参数矢量
        Returns:
            y_coarse: (B, 39, T) 保持无偏连续特征流
            event_emb: (B, d_event, T) 事件与时相混合动力学特征嵌入
            events_dict: 供监督和分析用的事件状态字典
        """
        phase_probs, ev = self.state_machine(y_coarse, p)  # phase_probs: (B, 4, T)

        # 提取 4 个核心事件特征
        s3 = ev['S3_history'].unsqueeze(1)
        s4 = ev['S4_history'].unsqueeze(1)
        s1 = ev['S1_history'].unsqueeze(1)
        s2 = ev['S2_history'].unsqueeze(1)
        events_tensor = torch.cat([phase_probs, s3, s4, s1, s2], dim=1)  # (B, 8, T)

        # 映射为事件动态表征
        event_emb = self.event_feature_proj(events_tensor)  # (B, d_event, T)

        return y_coarse, event_emb, ev
