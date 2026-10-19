#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HANO: Hybrid-Automaton Neural Operator (混合自动机神经算子)
============================================================

面向刚性、事件驱动的生化 ODE 系统（Chen et al. 2004 出芽酵母细胞周期）的
结构保持神经算子。三大结构先验被直接编码为架构组件:

1. 化学计量通量解码器 (Stoichiometric Flux Decoder)
   ODE 右端项本质是反应网络的线性组合: dy/dt = S·v(y, p)。
   本模型不直接回归状态轨迹，而是预测 R=98 个反应通量速率 v(t)，
   状态由真实空间积分重构: y(t) = y0 + S·∫v dτ (verify_stoichiometry 验证
   S·v ≡ eqns_torch, 逐项误差 < 1e-4)。
   ⇒ 初始条件精确满足; SIC1/CDC6 复合物池总量仅经合成/降解通量改变
     (络合/磷酸化结构性抵消); Vi20/lte1/BUB2 平滑段恒定(S 行为零)。

2. 事件时钟 + 事件跳变层 (Event Clock & Event Jump Layer)
   4 类细胞周期事件的跳变规则(BUD→0, ORI→0, MASS 乘性缩减, Vi20/lte1/BUB2
   重置等)完全已知(apply_event_torch)。网络只需预测事件热图 s_k(t)，
   跳变由已知规则层施加，梯度经热图端到端回传。
   事件标签从训练轨迹免费自动提取(交叉熵辅助监督)。

3. 多重关系图注意力 (Multi-Relational Graph Attention)
   39 个蛋白变量间的 6 类生化关系(复合物/磷酸化/去磷酸化/转录激活/
   泛素化/GK开关)作为跨变量注意力的结构偏置。

归一化约定 (与 yeast_datamodule 一致):
- y: log1p 后标准化; p: 线性标准化
- 模型在真实空间完成通量积分与跳变，最后映射回 log1p 归一化空间
- 统计量 (mean_y/std_y/p_mean/p_std/t_max) 以 persistent buffer 保存，
  由 lit module 在 setup() 注入并随 checkpoint 持久化

消融/性能变体:
- hano            : 完整模型 (纯物理路径)
- hano_dual       : 双路径门控融合 (物理路径 + 直接回归兜底, 性能主推)
- hano_no_event   : 去掉事件时钟与跳变层 (纯通量平滑预测)
- hano_no_flux    : 去掉通量解码器 (直接输出头, 保留事件层)
- hano_no_graph   : 去掉多重关系图注意力
- hano_refine     : 额外增加门控残差精炼头
"""

import math
import numpy as np
import torch
from .relation_graph_contracts import register_relation_graph
from budding_yeast_v2.data.normalization import normalize_y, denormalize_y, denormalize_p, check_norm_type
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 0. 变量索引 (与 utils/physics_eqns_torch.py 完全一致)
# ============================================================
(nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5,
 nSIC1P, nC2P, nC5P, nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T,
 nCDC20, nCDH1T, nCDH1, nCDC14T, nCDC14, nNET1T, nNET1, nRENT, nTEM1, nCDC15,
 nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, nVi20, nlte1, nBUB2) = range(39)

NUM_VARS = 39

# 参数键索引 (延迟导入避免循环依赖, 141 维)
try:
    from budding_yeast_v2.utils.physics_eqns_torch import PARAM_KEYS
    _P_IDX = {k: i for i, k in enumerate(PARAM_KEYS)}
except Exception:  # pragma: no cover
    _P_IDX = {}


# ============================================================
# 1. 反应基表 (Reaction Basis): 从 eqns_torch 逐项精确提取
#    每个反应: (名称, 化学计量 {var_idx: coeff})
#    验证: true_flux_rates() 给出真实速率, verify_stoichiometry()
#          检查 S·v == eqns_torch 逐项一致
# ============================================================
REACTIONS = [
    # --- 质量生长 ---
    ('grow',            {nMASS: +1}),                                  # kg*MASS
    # --- CLN2 ---
    ('synth_CLN2',      {nCLN2: +1}),                                  # (ksn2p+ksn2pp*SBF)*MASS
    ('deg_CLN2',        {nCLN2: -1}),                                  # kdn2*CLN2
    # --- CLB2 / CLB5 合成 ---
    ('synth_CLB2',      {nCLB2: +1}),                                  # (ksb2p+ksb2pp*MCM1)*MASS
    ('synth_CLB5',      {nCLB5: +1}),                                  # (ksb5p+ksb5pp*SBF)*MASS
    # --- SIC1 家族络合/解离 (池守恒) ---
    ('bind_CLB2_SIC1',  {nCLB2: -1, nSIC1: -1, nC2: +1}),              # kasb2*CLB2*SIC1
    ('bind_CLB5_SIC1',  {nCLB5: -1, nSIC1: -1, nC5: +1}),              # kasb5*CLB5*SIC1
    ('bind_CLB2_CDC6',  {nCLB2: -1, nCDC6: -1, nF2: +1}),              # kasf2*CLB2*CDC6
    ('bind_CLB5_CDC6',  {nCLB5: -1, nCDC6: -1, nF5: +1}),              # kasf5*CLB5*CDC6
    ('diss_C2',         {nC2: -1, nCLB2: +1, nSIC1: +1}),              # kdib2*C2
    ('diss_C5',         {nC5: -1, nCLB5: +1, nSIC1: +1}),              # kdib5*C5
    ('diss_F2',         {nF2: -1, nCLB2: +1, nCDC6: +1}),              # kdif2*F2
    ('diss_F5',         {nF5: -1, nCLB5: +1, nCDC6: +1}),              # kdif5*F5
    # --- APC 降解 (整复合物降解, 释放抑制剂) ---
    ('deg_CLB2',        {nCLB2: -1}),                                  # Vdb2*CLB2
    ('deg_CLB5',        {nCLB5: -1}),                                  # Vdb5*CLB5
    ('deg_C2',          {nC2: -1, nSIC1: +1}),                         # Vdb2*C2
    ('deg_C5',          {nC5: -1, nSIC1: +1}),                         # Vdb5*C5
    ('deg_F2',          {nF2: -1, nCDC6: +1}),                         # Vdb2*F2
    ('deg_F5',          {nF5: -1, nCDC6: +1}),                         # Vdb5*F5
    ('deg_C2P',         {nC2P: -1, nCLB2: +1}),                        # kd3c1*C2P
    ('deg_C5P',         {nC5P: -1, nCLB5: +1}),                        # kd3c1*C5P
    ('deg_F2P',         {nF2P: -1, nCLB2: +1}),                        # kd3f6*F2P
    ('deg_F5P',         {nF5P: -1, nCLB5: +1}),                        # kd3f6*F5P
    ('deg_C2P_rel',     {nC2P: -1, nSIC1P: +1}),                       # Vdb2*C2P
    ('deg_C5P_rel',     {nC5P: -1, nSIC1P: +1}),                       # Vdb5*C5P
    ('deg_F2P_rel',     {nF2P: -1, nCDC6P: +1}),                       # Vdb2*F2P
    ('deg_F5P_rel',     {nF5P: -1, nCDC6P: +1}),                       # Vdb5*F5P
    ('deg_SIC1P',       {nSIC1P: -1}),                                 # kd3c1*SIC1P
    ('deg_CDC6P',       {nCDC6P: -1}),                                 # kd3f6*CDC6P
    # --- SIC1/CDC6 磷酸化循环 (池守恒) ---
    ('phos_SIC1',       {nSIC1: -1, nSIC1P: +1}),                      # Vkpc1*SIC1
    ('dephos_SIC1P',    {nSIC1P: -1, nSIC1: +1}),                      # Vppc1*SIC1P
    ('phos_C2',         {nC2: -1, nC2P: +1}),                          # Vkpc1*C2
    ('dephos_C2P',      {nC2P: -1, nC2: +1}),                          # Vppc1*C2P
    ('phos_C5',         {nC5: -1, nC5P: +1}),                          # Vkpc1*C5
    ('dephos_C5P',      {nC5P: -1, nC5: +1}),                          # Vppc1*C5P
    ('phos_CDC6',       {nCDC6: -1, nCDC6P: +1}),                      # Vkpf6*CDC6
    ('dephos_CDC6P',    {nCDC6P: -1, nCDC6: +1}),                      # Vppf6*CDC6P
    ('phos_F2',         {nF2: -1, nF2P: +1}),                          # Vkpf6*F2
    ('dephos_F2P',      {nF2P: -1, nF2: +1}),                          # Vppf6*F2P
    ('phos_F5',         {nF5: -1, nF5P: +1}),                          # Vkpf6*F5
    ('dephos_F5P',      {nF5P: -1, nF5: +1}),                          # Vppf6*F5P
    # --- 合成 (SIC1/CDC6/SWI5 等) ---
    ('synth_SIC1',      {nSIC1: +1}),                                  # ksc1p+ksc1pp*SWI5
    ('synth_CDC6',      {nCDC6: +1}),                                  # ksf6p+ksf6pp*SWI5+ksf6ppp*SBF
    ('synth_SWI5T',     {nSWI5T: +1}),                                 # ksswip+ksswipp*MCM1
    ('deg_SWI5T',       {nSWI5T: -1}),                                 # kdswi*SWI5T
    ('synth_SWI5',      {nSWI5: +1}),                                  # ksswip+ksswipp*MCM1
    ('act_SWI5',        {nSWI5: +1}),                                  # kaswi*CDC14*(SWI5T-SWI5)
    ('degSWI5_a',       {nSWI5: -1}),                                  # kiswi*CLB2*SWI5
    ('degSWI5_b',       {nSWI5: -1}),                                  # kdswi*SWI5
    # --- IEP / CDC20 ---
    ('act_IEP',         {nIEP: +1}),                                   # Vaiep*(1-IEP)/(Jaiep+1-IEP)
    ('deg_IEP',         {nIEP: -1}),                                   # kiiep*IEP/(Jiiep+IEP)
    ('synth_CDC20T',    {nCDC20T: +1}),                                # ks20p+ks20pp*MCM1
    ('deg_CDC20T',      {nCDC20T: -1}),                                # kd20*CDC20T
    ('act_CDC20',       {nCDC20: +1}),                                  # (ka20p+ka20pp*IEP)*(CDC20T-CDC20)
    ('degCDC20_a',      {nCDC20: -1}),                                 # Vi20*CDC20
    ('degCDC20_b',      {nCDC20: -1}),                                 # kd20*CDC20
    # --- CDH1 (注: 本模型中 CDH1T 为平行合成池, 激活/失活不改变 CDH1T, 与 eqns 精确一致) ---
    ('synth_CDH1T',     {nCDH1T: +1}),                                 # kscdh
    ('deg_CDH1T',       {nCDH1T: -1}),                                 # kdcdh*CDH1T
    ('synth_CDH1',      {nCDH1: +1}),                                  # kscdh
    ('deg_CDH1',        {nCDH1: -1}),                                  # kdcdh*CDH1
    ('act_CDH1',        {nCDH1: +1}),                                  # Vacdh*(CDH1T-CDH1)/(Jacdh+CDH1T-CDH1)
    ('inact_CDH1',      {nCDH1: -1}),                                  # Vicdh*CDH1/(Jicdh+CDH1)
    # --- CDC14 / NET1 / RENT (REN 网络核心) ---
    ('synth_CDC14T',    {nCDC14T: +1}),                                # ks14
    ('deg_CDC14T',      {nCDC14T: -1}),                                # kd14*CDC14T
    ('synth_CDC14',     {nCDC14: +1}),                                 # ks14
    ('deg_CDC14',       {nCDC14: -1}),                                 # kd14*CDC14
    ('synth_NET1T',     {nNET1T: +1}),                                 # ksnet
    ('deg_NET1T',       {nNET1T: -1}),                                 # kdnet*NET1T
    ('synth_NET1',      {nNET1: +1}),                                  # ksnet
    ('deg_NET1',        {nNET1: -1}),                                  # kdnet*NET1
    ('bind_NET1_CDC14', {nNET1: -1, nCDC14: -1, nRENT: +1}),           # kasrent*NET1*CDC14
    ('bind_NET1P_CDC14', {nCDC14: -1}),                                # kasrentp*NET1P*CDC14 (不生成 RENT, 与 eqns 一致)
    ('diss_RENT',       {nRENT: -1, nNET1: +1, nCDC14: +1}),           # kdirent*RENT
    ('diss_RENTP',      {nCDC14: +1}),                                 # kdirentp*RENTP
    ('dephos_NET1P',    {nNET1: +1}),                                  # Vppnet*NET1P
    ('dephos_RENTP',    {nRENT: +1}),                                  # Vppnet*RENTP
    ('phos_NET1',       {nNET1: -1}),                                  # Vkpnet*NET1
    ('phos_RENT',       {nRENT: -1}),                                  # Vkpnet*RENT
    ('degRENT_kd14',    {nRENT: -1, nNET1: +1}),                       # kd14*RENT
    ('degRENT_kdnet',   {nRENT: -1, nCDC14: +1}),                      # kdnet*RENT
    ('degRENTP_kdnet',  {nCDC14: +1}),                                 # kdnet*RENTP
    # --- TEM1 / CDC15 (MEN 途径) ---
    ('act_TEM1',        {nTEM1: +1}),                                  # lte1*(Tem1T-TEM1)/(jatem+Tem1T-TEM1)
    ('inact_TEM1',      {nTEM1: -1}),                                  # BUB2*TEM1/(Jitem+TEM1)
    ('act_CDC15',       {nCDC15: +1}),                                 # (ka15p*(Tem1T-TEM1)+ka15pp*TEM1+ka15P*CDC14)*(Cdc15T-CDC15)
    ('inact_CDC15',     {nCDC15: -1}),                                 # ki15*CDC15
    # --- PPX ---
    ('synth_PPX',       {nPPX: +1}),                                   # ksppx
    ('deg_PPX',         {nPPX: -1}),                                   # Vdppx*PPX
    # --- PDS1 / ESP1 (分离酶) ---
    ('synth_PDS1',      {nPDS1: +1}),                                  # kspdsp+ks1pdspp*SBF+ks2pdspp*MCM1
    ('bind_PDS1_ESP1',  {nPDS1: -1, nESP1: -1}),                       # kasesp*PDS1*ESP1
    ('diss_PE',         {nPDS1: +1, nESP1: +1}),                       # kdiesp*PE
    ('deg_PDS1',        {nPDS1: -1}),                                  # Vdpds*PDS1
    ('deg_PE',          {nESP1: +1}),                                  # Vdpds*PE
    # --- ORI / BUD / SPN (形态发生标志) ---
    ('act_ORI',         {nORI: +1}),                                   # ksori*(eorib5*CLB5+eorib2*CLB2)
    ('deg_ORI',         {nORI: -1}),                                   # kdori*ORI
    ('act_BUD',         {nBUD: +1}),                                   # ksbud*(ebudn2*CLN2+ebudn3*CLN3+ebudb5*CLB5)
    ('deg_BUD',         {nBUD: -1}),                                   # kdbud*BUD
    ('act_SPN',         {nSPN: +1}),                                   # ksspn*CLB2/(Jspn+CLB2)
    ('deg_SPN',         {nSPN: -1}),                                   # kdspn*SPN
]

NUM_REACTIONS = len(REACTIONS)


def build_stoichiometry():
    """构建 (39, NUM_REACTIONS) 化学计量符号矩阵 S: dy/dt = S·v."""
    S = np.zeros((NUM_VARS, NUM_REACTIONS), dtype=np.float32)
    for r, (_, stoich) in enumerate(REACTIONS):
        for v, c in stoich.items():
            S[v, r] = c
    return S


# ============================================================
# 2. 真实通量速率 (仅用于化学计量验证与可解释性分析)
# ============================================================
def _gk(Va, Vi, Ja, Ji):
    BB = Vi - Va + Ja * Vi + Ji * Va
    disc = torch.clamp(BB ** 2 - 4 * (Vi - Va) * Ji * Va, min=0.0)
    return 2 * Ji * Va / (BB + torch.sqrt(disc) + 1e-12)


def true_flux_rates(y, p_dict):
    """
    给定 (B,39) 状态与参数字典, 返回 (B, NUM_REACTIONS) 真实反应速率。
    速率表达式与 eqns_torch 逐项对应 (is_mutant_104=False 情形)。
    用途: (1) verify_stoichiometry 验证 S 的正确性;
          (2) 训练后对比预测通量与真实通量 (可解释性)。
    """
    MASS = y[:, nMASS]; CLN2 = y[:, nCLN2]; CLB2 = y[:, nCLB2]; CLB5 = y[:, nCLB5]
    SIC1 = y[:, nSIC1]; CDC6 = y[:, nCDC6]; C2 = y[:, nC2]; C5 = y[:, nC5]
    F2 = y[:, nF2]; F5 = y[:, nF5]
    SIC1P = y[:, nSIC1P]; C2P = y[:, nC2P]; C5P = y[:, nC5P]
    CDC6P = y[:, nCDC6P]; F2P = y[:, nF2P]; F5P = y[:, nF5P]
    SWI5T = y[:, nSWI5T]; SWI5 = y[:, nSWI5]; IEP = y[:, nIEP]
    CDC20T = y[:, nCDC20T]; CDC20 = y[:, nCDC20]
    CDH1T = y[:, nCDH1T]; CDH1 = y[:, nCDH1]
    CDC14T = y[:, nCDC14T]; CDC14 = y[:, nCDC14]
    NET1T = y[:, nNET1T]; NET1 = y[:, nNET1]; RENT = y[:, nRENT]
    TEM1 = y[:, nTEM1]; CDC15 = y[:, nCDC15]
    PPX = y[:, nPPX]; PDS1 = y[:, nPDS1]; ESP1 = y[:, nESP1]
    ORI = y[:, nORI]; BUD = y[:, nBUD]; SPN = y[:, nSPN]
    lte1 = y[:, nlte1]; BUB2 = y[:, nBUB2]

    P = p_dict
    kg = math.log(2) / P['mdt']
    BCK2 = P['B0'] * MASS
    CLN3 = P['C0'] * P['Dn3'] * MASS / (P['Jn3'] + P['Dn3'] * MASS)
    SIC1T = SIC1 + C2 + C5 + SIC1P + C2P + C5P
    CDC6T = CDC6 + F2 + F5 + CDC6P + F2P + F5P
    RENTP = CDC14T - RENT - CDC14
    NET1P = NET1T - NET1 - CDC14T + CDC14
    PE = P['ESP1T'] - ESP1

    Vasbf = P['kasbf'] * (P['esbfn2'] * CLN2 + P['esbfn3'] * (CLN3 + BCK2) + P['esbfb5'] * CLB5)
    Vdb2 = P['kdb2p'] + P['kdb2pp'] * CDH1 + P['kdb2P'] * CDC20
    Vdb5 = P['kdb5p'] + P['kdb5pp'] * CDC20
    Vkpc1 = P['kd1c1'] + P['kd2c1'] * (P['ec1n3'] * CLN3 + P['ec1k2'] * BCK2 + P['ec1n2'] * CLN2 + P['ec1b2'] * CLB2 + P['ec1b5'] * CLB5) / (P['Jd2c1'] + SIC1T)
    Vppc1 = P['kppc1'] * CDC14
    Vkpf6 = P['kd1f6'] + P['kd2f6'] * (P['ef6n3'] * CLN3 + P['ef6k2'] * BCK2 + P['ef6n2'] * CLN2 + P['ef6b2'] * CLB2 + P['ef6b5'] * CLB5) / (P['Jd2f6'] + CDC6T)
    Vppf6 = P['kppf6'] * CDC14
    Vaiep = P['kaiep'] * CLB2
    Vacdh = P['kacdhp'] + P['kacdhpp'] * CDC14
    Vicdh = P['kicdhp'] + P['kicdhpp'] * (P['eicdhn3'] * CLN3 + P['eicdhn2'] * CLN2 + P['eicdhb5'] * CLB5 + P['eicdhb2'] * CLB2)
    Vppnet = P['kppnetp'] + P['kppnetpp'] * PPX
    Vkpnet = (P['kkpnetp'] + P['kkpnetpp'] * CDC15) * MASS
    Vdpds = P['kd1pdsp'] + P['kd2pdspp'] * CDC20 + P['kd3pdspp'] * CDH1
    Vdppx = P['kdppxp'] + P['kdppxpp'] * (P['J20ppx'] + CDC20) * P['Jpds'] / (P['Jpds'] + PDS1)

    SBF = _gk(Vasbf, P['kisbfp'] + P['kisbfpp'] * CLB2, P['Jasbf'], P['Jisbf'])
    MCM1 = _gk(P['kamcm'] * CLB2, P['kimcm'], P['Jamcm'], P['Jimcm'])

    Vi20 = y[:, nVi20]
    rates = [
        kg * MASS,
        (P['ksn2p'] + P['ksn2pp'] * SBF) * MASS,
        P['kdn2'] * CLN2,
        (P['ksb2p'] + P['ksb2pp'] * MCM1) * MASS,
        (P['ksb5p'] + P['ksb5pp'] * SBF) * MASS,
        P['kasb2'] * CLB2 * SIC1,
        P['kasb5'] * CLB5 * SIC1,
        P['kasf2'] * CLB2 * CDC6,
        P['kasf5'] * CLB5 * CDC6,
        P['kdib2'] * C2,
        P['kdib5'] * C5,
        P['kdif2'] * F2,
        P['kdif5'] * F5,
        Vdb2 * CLB2,
        Vdb5 * CLB5,
        Vdb2 * C2,
        Vdb5 * C5,
        Vdb2 * F2,
        Vdb5 * F5,
        P['kd3c1'] * C2P,
        P['kd3c1'] * C5P,
        P['kd3f6'] * F2P,
        P['kd3f6'] * F5P,
        Vdb2 * C2P,
        Vdb5 * C5P,
        Vdb2 * F2P,
        Vdb5 * F5P,
        P['kd3c1'] * SIC1P,
        P['kd3f6'] * CDC6P,
        Vkpc1 * SIC1,
        Vppc1 * SIC1P,
        Vkpc1 * C2,
        Vppc1 * C2P,
        Vkpc1 * C5,
        Vppc1 * C5P,
        Vkpf6 * CDC6,
        Vppf6 * CDC6P,
        Vkpf6 * F2,
        Vppf6 * F2P,
        Vkpf6 * F5,
        Vppf6 * F5P,
        P['ksc1p'] + P['ksc1pp'] * SWI5,
        P['ksf6p'] + P['ksf6pp'] * SWI5 + P['ksf6ppp'] * SBF,
        P['ksswip'] + P['ksswipp'] * MCM1,
        P['kdswi'] * SWI5T,
        P['ksswip'] + P['ksswipp'] * MCM1,
        P['kaswi'] * CDC14 * (SWI5T - SWI5),
        P['kiswi'] * CLB2 * SWI5,
        P['kdswi'] * SWI5,
        Vaiep * (1 - IEP) / (P['Jaiep'] + 1 - IEP),
        P['kiiep'] * IEP / (P['Jiiep'] + IEP),
        P['ks20p'] + P['ks20pp'] * MCM1,
        P['kd20'] * CDC20T,
        (P['ka20p'] + P['ka20pp'] * IEP) * (CDC20T - CDC20),
        Vi20 * CDC20,
        P['kd20'] * CDC20,
        P['kscdh'],
        P['kdcdh'] * CDH1T,
        P['kscdh'],
        P['kdcdh'] * CDH1,
        Vacdh * (CDH1T - CDH1) / (P['Jacdh'] + CDH1T - CDH1),
        Vicdh * CDH1 / (P['Jicdh'] + CDH1),
        P['ks14'],
        P['kd14'] * CDC14T,
        P['ks14'],
        P['kd14'] * CDC14,
        P['ksnet'],
        P['kdnet'] * NET1T,
        P['ksnet'],
        P['kdnet'] * NET1,
        P['kasrent'] * NET1 * CDC14,
        P['kasrentp'] * NET1P * CDC14,
        P['kdirent'] * RENT,
        P['kdirentp'] * RENTP,
        Vppnet * NET1P,
        Vppnet * RENTP,
        Vkpnet * NET1,
        Vkpnet * RENT,
        P['kd14'] * RENT,
        P['kdnet'] * RENT,
        P['kdnet'] * RENTP,
        lte1 * (P['Tem1T'] - TEM1) / (P['jatem'] + P['Tem1T'] - TEM1),
        BUB2 * TEM1 / (P['Jitem'] + TEM1),
        (P['ka15p'] * (P['Tem1T'] - TEM1) + P['ka15pp'] * TEM1 + P['ka15P'] * CDC14) * (P['Cdc15T'] - CDC15),
        P['ki15'] * CDC15,
        P['ksppx'],
        Vdppx * PPX,
        P['kspdsp'] + P['ks1pdspp'] * SBF + P['ks2pdspp'] * MCM1,
        P['kasesp'] * PDS1 * ESP1,
        P['kdiesp'] * PE,
        Vdpds * PDS1,
        Vdpds * PE,
        P['ksori'] * (P['eorib5'] * CLB5 + P['eorib2'] * CLB2),
        P['kdori'] * ORI,
        P['ksbud'] * (P['ebudn2'] * CLN2 + P['ebudn3'] * CLN3 + P['ebudb5'] * CLB5),
        P['kdbud'] * BUD,
        P['ksspn'] * CLB2 / (P['Jspn'] + CLB2),
        P['kdspn'] * SPN,
    ]
    return torch.stack(rates, dim=-1)  # (B, NUM_REACTIONS)


def verify_stoichiometry(n_checks=8, seed=0):
    """
    验证化学计量分解的正确性: 对随机状态/参数,
    S @ true_flux_rates(y, p) 必须与 eqns_torch(y, p) 逐项一致。
    """
    from budding_yeast_v2.utils.physics_eqns_torch import eqns_torch, params_tensor_to_dict
    from budding_yeast_v2.data.mass_dataset_generator import get_default_params

    torch.manual_seed(seed)
    defaults = get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    keys = sorted([k for k in defaults.keys() if k != 'is_mutant_104'])

    S = torch.tensor(build_stoichiometry())
    max_err = 0.0
    for _ in range(n_checks):
        B = 16
        y = torch.rand(B, NUM_VARS) * 1.5
        p_vec = torch.stack([
            torch.full((B,), float(defaults[k])) * (0.5 + torch.rand(B))
            for k in keys
        ], dim=-1)
        p_dict = params_tensor_to_dict(p_vec)
        f_ref = eqns_torch(y, p_dict)                       # (B, 39)
        v = true_flux_rates(y, p_dict)                      # (B, R)
        f_dec = v @ S.t()                                   # (B, 39)
        err = (f_ref - f_dec).abs().max().item()
        max_err = max(max_err, err)
    print(f"[HANO] 化学计量验证: {NUM_REACTIONS} 个反应 | 最大逐项误差 = {max_err:.3e}")
    assert max_err < 1e-4, f"化学计量分解与 eqns_torch 不一致! max_err={max_err}"
    print("[HANO] 化学计量验证通过: S·v ≡ eqns_torch")
    return True


# ============================================================
# 3. 事件定义 (与 apply_event_torch 的跳变规则一致)
# ============================================================
# 事件类型: 1=分裂(CLB2<0.3↓) 2=S期退出(CLB2+CLB5<0.2↓) 3=DNA复制(ORI≥1↑) 4=纺锤体通过(SPN≥1↑)
# 每个事件: (变量, 模式, 参数), 模式 ∈ {'reset0', 'const', 'mass_shrink'}
EVENT_RULES = {
    1: [(nBUD, 'reset0', None),
        (nSPN, 'reset0', None),
        (nMASS, 'mass_shrink', 'mdt'),
        (nlte1, 'const', 0.1)],
    2: [(nORI, 'reset0', None)],
    3: [(nVi20, 'const', 'Vi20_active'),
        (nBUB2, 'const', 'BUB2_active')],
    4: [(nVi20, 'const', 0.01),
        (nlte1, 'const', 1.0),
        (nBUB2, 'const', 'init_BUB2')],
}
NUM_EVENTS = 4


def _mass_shrink_factor(mdt):
    """分裂事件的质量缩减因子 (apply_event_torch event1 规则)."""
    kg = math.log(2.0) / mdt.clamp(min=1.0)
    return torch.abs(1.0 + torch.exp(-kg * ((1.026 / kg) - 32.0)) - 1.0)


# ============================================================
# 4. 模型组件
# ============================================================
class FiLMConvBlock(nn.Module):
    """参数 FiLM 调制的膨胀残差卷积块 (骨干基本单元)."""
    def __init__(self, channels, dilation=1, param_dim=141):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, channels)
        self.film = nn.Linear(param_dim, channels * 2)

    def forward(self, x, p):
        gamma, beta = self.film(p).chunk(2, dim=-1)
        res = x
        out = F.gelu(self.norm1(self.conv1(x)))
        out = (1.0 + gamma.unsqueeze(-1)) * out + beta.unsqueeze(-1)
        out = self.norm2(self.conv2(out))
        return F.gelu(out + res)


class MultiRelGraphAttention(nn.Module):
    """
    多重关系图导向的跨变量注意力。
    ODE 反应项派生的 6 类关系覆盖层，加未分类依赖，作为软注意力偏置。
    分类是显式机制注解，不是永久导数符号或唯一的生物学识别。
    """
    def __init__(self, channels=128, num_vars=39, head_dim=16, rel_graphs=None):
        super().__init__()
        self.num_vars = num_vars
        self.head_dim = head_dim
        self.to_vars = nn.Linear(channels, num_vars * head_dim)
        self.from_vars = nn.Linear(num_vars * head_dim, channels)
        self.q_proj = nn.Linear(head_dim, head_dim)
        self.k_proj = nn.Linear(head_dim, head_dim)
        self.v_proj = nn.Linear(head_dim, head_dim)

        register_relation_graph(self, num_vars, normalized=True, rel_graphs=rel_graphs)
        self.rel_weights = nn.Parameter(torch.zeros(6))   # 可学习关系融合权重

    def forward(self, h):
        # h: (B, C, T)
        B, C, T = h.shape
        h_t = h.permute(0, 2, 1)                          # (B, T, C)
        vars_feat = self.to_vars(h_t).reshape(B, T, self.num_vars, self.head_dim)
        Q, K, V = self.q_proj(vars_feat), self.k_proj(vars_feat), self.v_proj(vars_feat)
        attn_raw = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        # 关系加权图偏置: sum_r softmax(w)_r * A_r
        w = F.softmax(self.rel_weights, dim=0)
        bias = torch.einsum('r,rv->v', w, self.rel_graphs.reshape(6, -1)
                            ).reshape(self.num_vars, self.num_vars)
        bias = bias + w.mean() * self.untyped_graph
        attn = F.softmax(attn_raw + bias.unsqueeze(0).unsqueeze(0), dim=-1)
        out = torch.matmul(attn, V).reshape(B, T, self.num_vars * self.head_dim)
        return h + self.from_vars(out).permute(0, 2, 1)


class GKFluxHead(nn.Module):
    """
    GK 门控通量头: v = softplus(W·h) · GK(Va(h), Vi(h)) · scale。
    Goldbeter-Koshland 超灵敏开关作为每条反应的门控非线性。
    """
    def __init__(self, width=128, num_reactions=NUM_REACTIONS):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
        )
        self.to_flux = nn.Conv1d(width, num_reactions, 1)
        self.to_va = nn.Conv1d(width, num_reactions, 1)
        self.to_vi = nn.Conv1d(width, num_reactions, 1)
        self.Ja = nn.Parameter(torch.full((num_reactions,), 0.5))
        self.Ji = nn.Parameter(torch.full((num_reactions,), 0.5))
        self.log_scale = nn.Parameter(torch.full((num_reactions,), -4.0))

    def forward(self, h):
        # h: (B, C, T) -> v: (B, R, T)  (真实物理单位: 浓度/分钟, 非负)
        z = self.pre(h)
        base = F.softplus(self.to_flux(z))
        Va = F.softplus(self.to_va(z))
        Vi = F.softplus(self.to_vi(z))
        gate = _gk(Va, Vi, F.softplus(self.Ja).view(1, -1, 1),
                   F.softplus(self.Ji).view(1, -1, 1))
        return base * gate * torch.exp(self.log_scale).view(1, -1, 1)


class EventClockHead(nn.Module):
    """事件时钟: 输出 4 类事件热图 logits (B, 4, T)."""
    def __init__(self, width=128, num_events=NUM_EVENTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, num_events, 1),
        )

    def forward(self, h):
        return self.net(h)  # logits


class EventJumpLayer(nn.Module):
    """
    事件跳变层 (真实空间)。已知跳变规则 + 预测事件热图:
      y = y_smooth + Σ_k cumsum_t[ s_k(t) · Δ_k(Y_ref(t)) ]
    两遍精细化: 第一遍以 y_smooth 为参考计算跳变幅度,
    第二遍以含跳变的 Y_ref 为参考 (正确处理同类事件多次发生,
    如两次分裂的 MASS 乘性缩减级联)。
    """
    def __init__(self):
        super().__init__()

    def _event_deltas(self, event_id, y_ref, p_raw):
        """计算事件 event_id 在参考轨迹 y_ref (B,39,T) 下的跳变幅度 Δ (B,39,T)。"""
        B, V, T = y_ref.shape
        delta = torch.zeros_like(y_ref)
        rules = EVENT_RULES[event_id]
        for var, mode, param in rules:
            if mode == 'reset0':
                delta[:, var, :] = -y_ref[:, var, :]
            elif mode == 'const':
                if isinstance(param, str):
                    idx = _P_IDX.get(param)
                    if idx is not None:
                        target = p_raw[:, idx]
                    else:
                        target = torch.zeros(B, device=y_ref.device)
                else:
                    target = torch.full((B,), float(param), device=y_ref.device)
                delta[:, var, :] = target.unsqueeze(-1) - y_ref[:, var, :]
            elif mode == 'mass_shrink':
                mdt = p_raw[:, _P_IDX['mdt']].clamp(min=1.0) if 'mdt' in _P_IDX \
                    else torch.full((B,), 90.0, device=y_ref.device)
                f = _mass_shrink_factor(mdt)               # (B,)
                delta[:, var, :] = (f - 1.0).unsqueeze(-1) * y_ref[:, var, :]
        return delta

    def forward(self, y_smooth, s, p_raw):
        """
        y_smooth: (B,39,T) 平滑轨迹 (真实空间)
        s:        (B,4,T) 事件热图概率 (sigmoid 后)
        p_raw:    (B,141) 真实参数向量
        """
        # 第一遍: 以 y_smooth 为参考
        jumps = torch.zeros_like(y_smooth)
        for k in range(NUM_EVENTS):
            sk = s[:, k:k + 1, :]                          # (B,1,T)
            delta = self._event_deltas(k + 1, y_smooth, p_raw)
            jumps = jumps + sk * delta
        y_ref = y_smooth + torch.cumsum(jumps, dim=-1)
        # 第二遍: 以含跳变参考重算幅度
        jumps = torch.zeros_like(y_smooth)
        for k in range(NUM_EVENTS):
            sk = s[:, k:k + 1, :]
            delta = self._event_deltas(k + 1, y_ref, p_raw)
            jumps = jumps + sk * delta
        return y_smooth + torch.cumsum(jumps, dim=-1)


# ============================================================
# 5. HANO 主模型
# ============================================================
class HANO(nn.Module):
    """
    Hybrid-Automaton Neural Operator.

    参数注入: FiLM; 结构先验: 反应通量解码 + 事件跳变层 + 多重关系图注意力。
    接口: forward(x, p, return_aux=False)
      x: (B, 39, 2, T)  x[:,:,0,:]=归一化IC广播, x[:,:,1,:]=t_grid(0~1)
      p: (B, 141)       标准化参数
      返回: (B, 39, T) log1p 归一化空间轨迹; return_aux=True 时附 (event_logits, flux)
    """
    def __init__(self, num_vars=39, param_dim=141, width=128, n_blocks=4,
                 use_event=True, use_flux=True, use_graph=True,
                 use_refine=False, lambda_event=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.width = width
        self.use_event = use_event
        self.use_flux = use_flux
        self.use_graph = use_graph
        self.use_refine = use_refine
        self.lambda_event = lambda_event

        # ---- 归一化统计量 (persistent: 随 checkpoint 保存, eval 自动恢复) ----
        self.register_buffer('mean_y', torch.zeros(1, num_vars, 1))
        self.register_buffer('std_y', torch.ones(1, num_vars, 1))
        self.register_buffer('p_mean', torch.zeros(1, param_dim))
        self.register_buffer('p_std', torch.ones(1, param_dim))
        self.register_buffer('t_max', torch.tensor(210.0))
        self.register_buffer('s_c', torch.ones(1, num_vars, 1), persistent=False)
        self.norm_type = 'log1p'  # persisted by YeastLitModule's preprocessing payload

        # ---- 化学计量矩阵 ----
        if use_flux:
            self.register_buffer('S', torch.tensor(build_stoichiometry()))  # (39, R)
            self.flux_head = GKFluxHead(width=width)

        # ---- 骨干 ----
        in_ch = num_vars * 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, 3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
        )
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList([
            FiLMConvBlock(width, dilation=dilations[i % len(dilations)], param_dim=param_dim)
            for i in range(n_blocks)
        ])
        self.graph_attn = MultiRelGraphAttention(channels=width, num_vars=num_vars) \
            if use_graph else None

        # ---- 输出头 ----
        if use_event:
            self.event_head = EventClockHead(width=width)
            self.jump_layer = EventJumpLayer()
        if not use_flux:
            # 消融: 直接输出头 (仍保证 IC 精确: 输出残差加回 IC)
            self.direct_head = nn.Sequential(
                nn.Conv1d(width, width, 3, padding=1),
                nn.GroupNorm(1, width),
                nn.GELU(),
                nn.Conv1d(width, num_vars, 1),
            )
        if use_refine:
            self.refine_head = nn.Sequential(
                nn.Conv1d(width + num_vars, width, 3, padding=1),
                nn.GroupNorm(1, width),
                nn.GELU(),
                nn.Conv1d(width, num_vars, 1),
            )
            self.gamma = nn.Parameter(torch.tensor(0.0))

    # ---------- 统计量注入 (lit module setup() 调用) ----------
    def set_stats(self, mean_y, std_y, p_mean, p_std, t_max=210.0, *, norm_type='log1p', s_c=None):
        check_norm_type(norm_type)
        if norm_type == 'asinh_scale' and s_c is None:
            raise ValueError('asinh_scale HANO requires s_c')
        dev = self.mean_y.device
        self.mean_y.copy_(mean_y.reshape(1, -1, 1).to(dev))
        self.std_y.copy_(std_y.reshape(1, -1, 1).to(dev))
        self.p_mean.copy_(p_mean.reshape(1, -1).to(dev))
        self.p_std.copy_(p_std.reshape(1, -1).to(dev))
        self.t_max.fill_(float(t_max))
        self.norm_type = norm_type
        self.s_c.copy_(torch.ones_like(self.s_c) if s_c is None else s_c.reshape_as(self.s_c).to(dev))

    # ---------- 归一化工具 ----------
    def _denorm_y(self, y_norm):
        """log1p 归一化空间 -> 真实空间 (与 datamodule.denormalize_y 一致).
        支持 (B, 39) 与 (B, 39, T)."""
        return denormalize_y(y_norm, self.mean_y, self.std_y, self.norm_type, self.s_c)

    def _norm_y(self, y_real):
        """真实空间 -> log1p 归一化空间. 支持 (B, 39) 与 (B, 39, T)."""
        return normalize_y(y_real, self.mean_y, self.std_y, self.norm_type, self.s_c)

    def _denorm_p(self, p_norm):
        return denormalize_p(p_norm, self.p_mean, self.p_std, self.norm_type)

    # ---------- 事件标签提取 (真实空间, 从目标轨迹) ----------
    def compute_event_labels(self, y_norm):
        """
        从归一化目标轨迹提取 4 类事件标签 (B, 4, T)。
        依据 apply_event_torch 的跳变特征 (与 detect_event_mask_torch 阈值一致):
          ev1 分裂: BUD/SPN 骤降或 MASS 缩减
          ev2 S期退出: ORI 归零
          ev3 DNA复制: Vi20/BUB2 激活
          ev4 纺锤体通过: lte1 升至 1 / Vi20 骤降
        """
        with torch.no_grad():
            y_real = self._denorm_y(y_norm)                # (B, 39, T)
            def d(v):
                return torch.cat([v[:, :1], v[:, 1:] - v[:, :-1]], dim=1)
            dBUD, dSPN, dMASS = d(y_real[:, nBUD]), d(y_real[:, nSPN]), d(y_real[:, nMASS])
            dORI = d(y_real[:, nORI])
            dVi20, dBUB2, dlte1 = d(y_real[:, nVi20]), d(y_real[:, nBUB2]), d(y_real[:, nlte1])
            ev1 = (dBUD < -0.3) | (dSPN < -0.3) | (dMASS < -0.5)
            ev2 = (dORI < -0.5)
            ev3 = (dVi20 > 0.3) | (dBUB2 > 0.3)
            ev4 = (dlte1 > 0.3) | (dVi20 < -0.3)
            labels = torch.stack([ev1, ev2, ev3, ev4], dim=1).float()
        return labels

    # ---------- 物理路径: 通量积分 + 事件跳变 (float32 保护) ----------
    def _physics_path(self, h, ic_norm, p):
        """
        结构化物理路径: GK 通量解码 -> 梯形积分 -> 化学计量重构 -> 事件跳变。
        强制 float32: bf16 AMP 下 cumsum 积分/GK 开关/跳变累加对精度敏感
        (实测 wb 数据集 epoch 75+ 持续 NaN 的根源)。

        h: (B, C, T) 骨干特征; ic_norm: (B, 39) 归一化 IC; p: (B, 141) 标准化参数
        返回 (physics_out, y_real, aux)
        """
        aux = {}
        with torch.autocast(device_type=h.device.type, enabled=False):
            h32 = h.float()
            p_raw = self._denorm_p(p.float())               # (B, 141) 真实参数
            y0_real = self._denorm_y(ic_norm.float())       # (B, 39) 真实初值

            if self.use_flux:
                v = self.flux_head(h32)                      # (B, R, T) 真实速率
                v = torch.clamp(v, max=100.0)                # 速率上限防爆炸
                Tn = v.shape[-1]
                dt = (self.t_max / max(Tn - 1, 1))
                # 梯形积分: Phi(t) = ∫v dτ  (cumulative)
                Phi = torch.cumsum(0.5 * (v[:, :, 1:] + v[:, :, :-1]) * dt, dim=-1)
                Phi = torch.cat([torch.zeros_like(Phi[:, :, :1]), Phi], dim=-1)  # (B, R, T)
                # 化学计量重构: y = y0 + S·Φ (真实空间, IC 精确)
                y_smooth = y0_real.unsqueeze(-1) + torch.einsum('vr,brt->bvt', self.S, Phi)
                aux['flux'] = v
            else:
                y_smooth = y0_real.unsqueeze(-1) + self.direct_head(h32)  # IC 精确 + 残差头

            if self.use_event:
                logits = self.event_head(h32)                # (B, 4, T)
                s = torch.sigmoid(logits)
                y_real = self.jump_layer(y_smooth, s, p_raw)
                aux['event_logits'] = logits
            else:
                y_real = y_smooth

            physics_out = self._norm_y(y_real)               # (B, 39, T) 归一化空间
        return physics_out, y_real, aux

    # ---------- 前向 ----------
    def forward(self, x, p, return_aux=False):
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            ic_norm = x[:, :, 0, 0]                        # (B, 39) 归一化 IC
            xin = x.reshape(B, V * F_in, T)
        else:
            B, V, T = x.shape
            ic_norm = x[:, :, 0]
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).expand(B, V, 1)
            xin = torch.cat([x[:, :, :1].expand(B, V, T), time_grid.expand(B, V, T)], dim=1)

        # ---- 骨干 ----
        h = self.stem(xin)
        for blk in self.blocks:
            h = blk(h, p)
        if self.graph_attn is not None:
            h = self.graph_attn(h)

        out, y_real, aux = self._physics_path(h, ic_norm, p)

        # ---- 可选残差精炼 ( gated, 初始为 0 不破坏结构保证 ) ----
        if self.use_refine:
            r = self.refine_head(torch.cat([h, out], dim=1))
            y_real = y_real + torch.tanh(self.gamma) * r
            out = self._norm_y(y_real)

        if return_aux:
            return out, aux
        return out


# ============================================================
# 6. 消融变体
# ============================================================
class HANO_NoEvent(HANO):
    """消融: 去掉事件时钟与跳变层 (纯通量平滑预测)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_event=False, **kwargs)


class HANO_NoFlux(HANO):
    """消融: 去掉通量解码器 (直接输出头, 保留事件层)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_flux=False, **kwargs)


class HANO_NoGraph(HANO):
    """消融: 去掉多重关系图注意力."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_graph=False, **kwargs)


class HANO_Refine(HANO):
    """HANO + 门控残差精炼头."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_refine=True, **kwargs)


class HANO_Dual(HANO):
    """
    双路径门控融合 (性能主推变体)。

    纯物理路径 (通量积分) 是开环系统: t 时刻的通量误差会累积到之后所有时刻,
    且 98 个数量级各异的反应速率难以直接回归 —— 这是 HANO 原版性能瓶颈的根源。
    本变体解除该瓶颈, 同时保留全部结构先验:

        out = g · 物理路径 (通量积分 + 事件跳变, 结构保证)
            + (1-g) · 直接路径 (归一化空间残差回归, 容量兜底)

    - 直接路径同样 IC 精确 (残差加回归一化 IC)
    - 门控 g 由骨干特征逐变量、逐时刻预测 (零初始化, 起步 g=0.5)
    - 物理路径有独立辅助损失 (lambda_phys, lit module 注入),
      防止门控偏向直接路径后物理分支梯度饿死
    - 下限: 退化为直接回归 (≥ unet 水平); 上限: 物理结构增强
    """
    def __init__(self, *args, lambda_phys=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_phys = lambda_phys
        self.dual_trunk = nn.Sequential(
            nn.Conv1d(self.width, self.width, 3, padding=1),
            nn.GroupNorm(1, self.width),
            nn.GELU(),
            nn.Conv1d(self.width, self.width, 3, padding=1),
            nn.GroupNorm(1, self.width),
            nn.GELU(),
        )
        self.dual_out = nn.Conv1d(self.width, self.num_vars, 1)
        self.gate_head = nn.Conv1d(self.width, self.num_vars, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)   # g 初始 0.5

    def forward(self, x, p, return_aux=False):
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            ic_norm = x[:, :, 0, 0]                        # (B, 39) 归一化 IC
            xin = x.reshape(B, V * F_in, T)
        else:
            B, V, T = x.shape
            ic_norm = x[:, :, 0]
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).expand(B, V, 1)
            xin = torch.cat([x[:, :, :1].expand(B, V, T), time_grid.expand(B, V, T)], dim=1)

        # ---- 骨干 ----
        h = self.stem(xin)
        for blk in self.blocks:
            h = blk(h, p)
        if self.graph_attn is not None:
            h = self.graph_attn(h)

        # ---- 物理路径 (结构先验) ----
        physics_out, _, aux = self._physics_path(h, ic_norm, p)

        # ---- 直接路径 (归一化空间残差, IC 精确) ----
        direct_out = ic_norm.unsqueeze(-1) + self.dual_out(self.dual_trunk(h))

        # ---- 逐点门控融合 ----
        g = torch.sigmoid(self.gate_head(h))               # (B, 39, T)
        out = g * physics_out + (1.0 - g) * direct_out

        aux['physics_out'] = physics_out
        aux['gate'] = g

        if return_aux:
            return out, aux
        return out


if __name__ == '__main__':
    verify_stoichiometry()
