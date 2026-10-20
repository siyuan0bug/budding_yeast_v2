# -*- coding: utf-8 -*-
"""
================================================================================
 Fission Yeast Cell-Cycle Model (Sveiczer et al., PNAS 2000)
 Random-Sampling Mutant Dataset Generator — Server Edition (v2.0)
================================================================================
 定位 : 服务器批量数据生成器 (形态对齐 codes/mutant_aware_server.py)
        单文件自包含 · 命令行驱动 · 多进程并行 · 导出单个 .npz · 无绘图。
 模型 : sveiczer_csikasznagy_gyorffy_tyson_novak_2000.cellml
        论文: papers/Sveiczer 等 - 2000 (PNAS 97:7865-7870) Table 1 / Table 2
 依据 : extracted_markdown_notes/Mutant_Aware_Sampling_Scheme_v1_Sveiczer_2000.md

 采样 (纯随机, 无任何拓扑/结构约束):
   * 真实突变体 : Table 2 全部 20 项 (WT + 19 株; 含两行 "or" 并列的
                  cdc25ts/cdc25D 与 ste9D/cdc13dbD, 两种菌株都各建一条), 逐一仿真。
   * 合成突变体 : 分层配额 L1/L2/L3 = 1 / 2 / 3 个蛋白质模块
                  (默认 45/35/20; 上限 3 与论文 Table 2 的最大突变基因数一致),
                  模块从 11 个模块中均匀随机抽取,
                  L2/L3 模块互斥 (replace=False, 参数天然不冲突)。
   * L1 覆盖     : (模块 x 模式) 网格轮转, 保证每个格子被抽到 (Latin-square 式)。
   * 模式采样    : --mode_freq {class, flat}
                  class = 模式类配额 (默认); flat = 均匀 (1/k)。
   * 强度        : KO=0; ts=base*0.15*Beta(0.5,2); OP=base*LogU[1.5,10];
                  damp=U[0.004,0.025]; scale=base*U[0.5,1.5]。
   * 去重        : 与真实突变体 + 已接受合成样本的参数向量比对 (rtol=1e-5)。

 离散事件 (论文 Table 1 "Switches", 无条件规则, G2 重置为涌现行为):
   (i)   MPFa/PP 上穿 0.65 -> START: RDNA=0, kS=kSmax; RDNA->1 -> S 完成: kS=kSmin
   (ii)  MPFa/PP 下穿 0.50 -> 功能性分裂: mass <- mass/2
   (iii) S 期中 MPF>0.2 -> 复制减速; 未完成复制即分裂 = cut 致死

 导出 .npz 字段:
   data(N,16,500) / params(N,P) / param_names / mutant_names / labels /
   pattern_labels / levels / var_names / time_grid

 运行示例:
   python fission_random_sampler.py --samples 1000 --workers 60
   python fission_random_sampler.py --samples 1000 --output_dir /path/to/dir
================================================================================
"""

import argparse
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ==============================================================================
# Part 0. 全局配置
# ==============================================================================
TF_MIN      = 300.0     # 仿真时长 (min): 2 个 WT 完整周期 (MDT=150)
N_T         = 500       # 均匀时间网格点数
MASS_LIMIT  = 50.0      # 质量爆炸判据 (Discarded_Mass, 原始尺度)
MASS_UNIT   = 1.4048    # WT 极限环出生质量 (原始尺度); 论文 Table 2 相对尺度基准
TOO_SMALL_M = 0.625     # Too Small 阈值 (原始尺度) = 相对 0.445 x MASS_UNIT。
                        # 论文 Table 2 脚注§ 只给定性判据: "显著小于 wee1⁻ 的细胞视为过小
                        # 不可活", 参照点 = wee1⁻ 出生质量 0.48 rel (= 0.674 raw), 无具体数值。
                        # 0.445 是本工程在该空隙内选定的阈值: 不可活株最大 0.41 rel (0.577 raw)
                        # < 0.625 raw < 可活株最小 0.48 rel (0.674 raw)。空隙内任意取值结果相同。
MAX_EVENTS  = 80        # 单次仿真事件数上限
DT_EPS      = 1e-6      # 事件后时间前跳 (min), 防止事件零点重复触发
SEED        = 20000

DEFAULT_SAMPLES      = 1000
DEFAULT_LEVEL_RATIOS = [0.45, 0.35, 0.20]   # L1/L2/L3 配额 (1/2/3 个模块, 上限 3 = Table 2 最大突变基因数)

# 16 个状态变量 (顺序即存储顺序, 下标固定)
VAR_NAMES = ['mass', 'RDNA', 'MPF', 'preMPF', 'Rum1', 'Rum1P', 'CR', 'CRP',
             'Ste9', 'Mik1', 'Wee1', 'Cdc25', 'Slp1', 'Slp1a', 'Inh', 'PI']
(iMASS, iRDNA, iMPF, iPREMPF, iRUM1, iRUM1P, iCR, iCRP,
 iSTE9, iMIK1, iWEE1, iCDC25, iSLP1, iSLP1A, iINH, iPI) = range(16)
N_VAR = 16

# ==============================================================================
# Part 1. 参数 (WT, 论文 Table 1 / CellML 逐项核对)
# ==============================================================================
def WT_parameters():
    p = dict(
        # --- Cdc13 合成与生长 ---
        k1=0.02,            # Cdc13 一阶合成速率
        mu=0.00462,         # 指数生长率 (质量倍增 150 min)
        # --- Cdc13 降解 (Ste9 介导) ---
        V2_=0.02, V2=1.0,   # 主动降解 V2 = V2_ + V2*Ste9
        V2c_=0.02, V2c=0.5, # CR/CRP 三聚体降解 V2c = V2c_ + V2c*Ste9
        # --- Inh (PP 抑制子) 合成 ---
        k3=0.1,
        # --- Inh/PI 被 Slp1a 降解 ---
        V4_=0.01, V4=1.0,
        # --- Rum1 合成与降解 ---
        k5=0.1, k6_=0.1, k6=5.0,
        # --- Rum1/Rum1P 磷酸化 (MPFa 与 SK*mass) ---
        kp=100.0, kpp_=1.0, kpp=100.0,
        # --- 三聚体 CR/CRP 结合与解离 ---
        kJ=400.0, kJR=1.0,
        # --- Ste9 修饰 ---
        kste9=5.0, kste9r_=0.03, kste9r=8.0,
        # --- Mik1 修饰 ---
        kM=1.0, kMR_=0.01, kMR=5.0,
        # --- Wee1 修饰 ---
        kW=2.0, kWR_=0.4, kWR=1.0,
        # --- Cdc25 修饰 ---
        k25=1.0, k25R_=0.4, k25R=2.0,
        # --- Slp1 (APC) ---
        kas=0.1, kad=0.1, kaa_=0.01, kaa=0.1, kai=0.1,
        # --- Inh 磷酸化 / PI 去磷酸化 ---
        ki=50.0, kiR=0.5,
        # --- S 期开关参数 ---
        kSmax=5.0, kSmin=0.1,
        # --- 酪氨酸修饰酶最大活性 ---
        Vwee=10.0, Vwee_=0.08, Vmik=2.0, Vmik_=0.04,
        V25=10.0, V25_=0.05, Vpyp=0.07,
        # --- Michaelis 常数 (超灵敏开关) ---
        Jste9=0.01, Jste9R=0.01, Jmik=0.15, JmikR=0.15,
        Jwee=0.2, JweeR=0.2, J25=0.05, J25R=0.05,
        # --- 起始激酶 SK / MPFa eps / epsP / 复制速率 K / Pyp3 总量 ---
        SK=0.018, eps=0.05, epsP=0.025, K=0.06, Pyp3=1.0,
        # --- 离散开关状态 kS (动态, 随事件切换; 不参与参数向量导出) ---
        kS=0.1,
    )
    return p

# 参数向量导出顺序 (排除动态开关 kS)
EXPORT_KEYS = sorted([k for k in WT_parameters() if k != 'kS'])

# ==============================================================================
# Part 2. ODE 引擎 (16 维, 全部速率函数按论文 Table 1 内联)
# ==============================================================================
def _Y_switch(MPF):
    """规则(iii): S 期中 MPF>0.2 -> 复制减速。用平滑 sigmoid 实现, 数值稳健。"""
    return 100.0 / (1.0 + np.exp(-80.0 * (MPF - 0.2)))

def eqns(t, y, p):
    mass, RDNA, MPF, preMPF, Rum1, Rum1P, CR, CRP, \
        Ste9, Mik1, Wee1, Cdc25, Slp1, Slp1a, Inh, PI = y

    MPFa = MPF + p['eps'] * preMPF
    PP   = max(1.0 - PI, 0.0)
    SKm  = p['SK'] * mass

    k2   = p['V2_']  + p['V2']  * Ste9
    k2c  = p['V2c_'] + p['V2c'] * Ste9
    kWee = p['Vwee'] * Wee1 + p['Vwee_'] * (1.0 - Wee1) \
         + p['Vmik'] * Mik1 + p['Vmik_'] * (1.0 - Mik1)
    kC25 = p['V25'] * Cdc25 + p['V25_'] * (1.0 - Cdc25) + p['Vpyp'] * p['Pyp3']
    k4   = p['V4_'] + p['V4'] * Slp1a

    dRum1   = (p['k5'] - p['k6_'] * Rum1
               - p['kp'] * (MPFa + p['epsP'] * SKm) * Rum1
               + (p['kpp_'] + p['kpp'] * PP) * Rum1P
               - p['kJ'] * MPF * Rum1 + p['kJR'] * CR + k2c * CR)
    dRum1P  = (p['kp'] * (MPFa + p['epsP'] * SKm) * Rum1
               - (p['kpp_'] + p['kpp'] * PP) * Rum1P
               - (p['k6_'] + p['k6']) * Rum1P
               - p['kJ'] * MPF * Rum1P + p['kJR'] * CRP + k2c * CRP)
    dCR     = (p['kJ'] * MPF * Rum1 - p['kJR'] * CR - k2c * CR
               - p['k6_'] * CR
               - p['kp'] * (MPFa + p['epsP'] * SKm) * CR
               + (p['kpp_'] + p['kpp'] * PP) * CRP)
    dCRP    = (p['kp'] * (MPFa + p['epsP'] * SKm) * CR
               - (p['kpp_'] + p['kpp'] * PP) * CRP
               + p['kJ'] * MPF * Rum1P - p['kJR'] * CRP - k2c * CRP
               - (p['k6_'] + p['k6']) * CRP)
    dMPF    = (p['k1'] * mass - k2 * MPF - kWee * MPF + kC25 * preMPF
               - p['kJ'] * MPF * (Rum1 + Rum1P)
               + p['kJR'] * (CR + CRP) + p['k6_'] * CR + (p['k6_'] + p['k6']) * CRP)
    dpreMPF = kWee * MPF - kC25 * preMPF - k2 * preMPF
    dSte9   = ((p['kste9r_'] + p['kste9r'] * PP) * (1.0 - Ste9) / (p['Jste9R'] + 1.0 - Ste9)
               - p['kste9'] * (MPFa + SKm) * Ste9 / (p['Jste9'] + Ste9))
    dMik1   = ((p['kS'] + p['kMR_'] + p['kMR'] * PP) * (1.0 - Mik1) / (p['JmikR'] + 1.0 - Mik1)
               - p['kM'] * MPFa * Mik1 / (p['Jmik'] + Mik1))
    dWee1   = ((p['kWR_'] + p['kWR'] * PP) * (1.0 - Wee1) / (p['JweeR'] + 1.0 - Wee1)
               - p['kW'] * MPFa * Wee1 / (p['Jwee'] + Wee1))
    dCdc25  = (p['k25'] * MPFa * (1.0 - Cdc25) / (p['J25'] + 1.0 - Cdc25)
               - (p['kS'] + p['k25R_'] + p['k25R'] * PP) * Cdc25 / (p['J25R'] + Cdc25))
    dSlp1   = p['kas'] * MPFa - p['kad'] * Slp1
    dSlp1a  = ((p['kaa_'] + p['kaa'] * MPFa) * (Slp1 - Slp1a)
               - p['kai'] * Slp1a - p['kad'] * Slp1a)
    dInh    = p['k3'] - p['ki'] * Inh * PP + p['kiR'] * PI - k4 * Inh
    dPI     = p['ki'] * Inh * PP - p['kiR'] * PI - k4 * PI
    dmass   = p['mu'] * mass
    # DNA 复制: 完成后 (RDNA>=1) 停止, 下次 START 由事件置 0
    dRDNA   = 0.0 if RDNA >= 1.0 else p['K'] / (1.0 + _Y_switch(MPF) * MPFa)

    return np.array([dmass, dRDNA, dMPF, dpreMPF, dRum1, dRum1P, dCR, dCRP,
                     dSte9, dMik1, dWee1, dCdc25, dSlp1, dSlp1a, dInh, dPI])

# ---- 离散事件函数 (论文 Table 1 Switches) ----
def _ratio_MPFa_PP(y, p):
    MPFa = y[iMPF] + p['eps'] * y[iPREMPF]
    PP   = max(1.0 - y[iPI], 1e-6)
    return MPFa / PP

def ev_start(t, y, p):
    # START: MPFa/PP 上穿 0.65 (S 期中的虚假穿越在事件处理阶段按 RDNA 过滤)
    return _ratio_MPFa_PP(y, p) - 0.65
ev_start.terminal  = True
ev_start.direction = 1

def ev_sfinish(t, y, p):
    # S 期完成: RDNA 上穿 1 (阈值取 0.99999 避免与钳位冲突)
    return y[iRDNA] - 0.99999
ev_sfinish.terminal  = True
ev_sfinish.direction = 1

def ev_division(t, y, p):
    # 功能性分裂: MPFa/PP 下穿 0.50
    return _ratio_MPFa_PP(y, p) - 0.50
ev_division.terminal  = True
ev_division.direction = -1

EVENTS = [ev_start, ev_sfinish, ev_division]

# ==============================================================================
# Part 3. 事件驱动仿真
# ==============================================================================
def _default_birth_state():
    """CellML 占位初值 (模型自洽收敛前的启动状态)。"""
    y = np.zeros(N_VAR)
    y[iMASS], y[iRDNA] = 1.0, 1.0
    y[iPREMPF], y[iPI] = 1.0, 0.2
    return y

def run_event_sim(y0, p, tf=TF_MIN):
    """事件驱动积分, 逐步记录状态序列, 返回 (seg_list, info)。"""
    from scipy.integrate import solve_ivp
    y0 = np.array(y0, dtype=float).copy()
    y0[iRDNA] = min(y0[iRDNA], 1.0)
    t_cur, n_ev = 0.0, 0
    div_times, div_masses, birth_masses = [], [], []
    rdna_at_div, post_div_states = [], []
    start_times, sfinish_times = [], []
    max_MPF = float(y0[iMPF])
    status, err = 'OK', ''
    segs = []
    while t_cur < tf and n_ev < MAX_EVENTS:
        sol = solve_ivp(eqns, (t_cur, tf), y0, method='BDF', args=(p,),
                        events=EVENTS, dense_output=True, rtol=1e-6, atol=1e-8)
        if not sol.success:
            status, err = 'NumericalFailure', str(sol.message)
            break
        max_MPF = max(max_MPF, float(np.max(sol.sol(np.linspace(t_cur, sol.t[-1], 200))[iMPF])))
        fired = None
        for k, name in ((2, 'DIV'), (0, 'START'), (1, 'SFINISH')):
            if sol.t_events[k].size > 0:
                fired = (k, name, sol.t_events[k][0], sol.y_events[k][0])
                break
        if fired is None:
            segs.append((t_cur, sol.t[-1], sol))
            y0, t_cur = sol.y[:, -1].copy(), sol.t[-1]
            status = 'End'
            break
        k, name, te, ye = fired
        segs.append((t_cur, te, sol))
        ye = np.maximum(np.array(ye, dtype=float), 0.0)
        ye[iRDNA] = min(ye[iRDNA], 1.0)
        if name == 'DIV':
            rdna_at_div.append(float(ye[iRDNA]))   # 分裂时复制完成度 (cut 判据)
            ye[iMASS] /= 2.0
            birth_masses.append(float(ye[iMASS]))
            div_times.append(float(te))
            div_masses.append(float(2.0 * ye[iMASS]))
            # 防止 ratio 恰在阈值处重复触发: 整体微缩 MPF/preMPF 0.05%
            MPFa = ye[iMPF] + p['eps'] * ye[iPREMPF]
            PP = max(1.0 - ye[iPI], 1e-6)
            if PP > 1e-6 and MPFa / PP >= 0.4995:
                ye[iMPF] *= 0.9995
                ye[iPREMPF] *= 0.9995
            post_div_states.append(ye.copy())
        elif name == 'START':
            # 合法 START (G1) 或 S 期虚假穿越均处理: 把 ratio 微推离阈值
            r = _ratio_MPFa_PP(ye, p)
            if ye[iRDNA] >= 1.0:          # G1 中的合法 START
                start_times.append(float(te))
                ye[iRDNA] = 0.0
                p['kS'] = p['kSmax']
            if r <= 0.6502:
                f = 0.6504 / max(r, 1e-12)
                if f <= 1.05:
                    ye[iMPF] *= f
                    ye[iPREMPF] *= f
        else:  # SFINISH
            sfinish_times.append(float(te))
            ye[iRDNA] = 1.0
            p['kS'] = p['kSmin']
        y0, t_cur = ye, float(te) + DT_EPS
        n_ev += 1
    if status == 'OK' and n_ev >= MAX_EVENTS:
        status = 'MaxEvents'
    info = dict(div_times=div_times, div_masses=div_masses,
                birth_masses=birth_masses, rdna_at_div=rdna_at_div,
                post_div_states=post_div_states,
                start_times=start_times, sfinish_times=sfinish_times,
                max_MPF=max_MPF, status=status, err=err)
    return segs, info

def resample_uniform(segs, tf=TF_MIN, n_t=N_T):
    """将分段 dense-output 解重采样到均匀网格 (正确处理事件跳变)。"""
    t_grid = np.linspace(0.0, tf, n_t)
    Y = np.zeros((N_VAR, n_t))
    for j, tq in enumerate(t_grid):
        chosen = None
        for (t0, t1, sol) in segs:
            if t0 - 1e-9 <= tq <= t1 + 1e-9:
                chosen = (min(max(tq, t0), t1), sol)
                break
            if t0 > tq:
                chosen = (t0, sol)
                break
        if chosen is None:
            t1, sol = segs[-1][1], segs[-1][2]
            chosen = (t1, sol)
        Y[:, j] = chosen[1].sol(chosen[0])
    return np.maximum(Y, 0.0)

# ==============================================================================
# Part 4. 表型标注
# ==============================================================================
def assign_labels(info, y_uniform, births_lab=None):
    div_times  = info['div_times']
    births     = births_lab if births_lab is not None else info['birth_masses']
    n_div      = len(div_times)
    status     = info['status']
    if status in ('NumericalFailure', 'MaxEvents'):
        return 'Inviable: Numerical Failure', None, status
    if n_div == 0:
        if info['max_MPF'] > 0.5:
            return 'Inviable: M-phase Arrest', None, status
        return 'Inviable: G2 Arrest', None, status
    if births and births[-1] < TOO_SMALL_M:   # 稳态 (最后一次) 出生质量, 原始尺度
        return 'Inviable: Too Small', None, status
    if n_div >= 2:
        pattern = 'Pattern_A'
    elif n_div == 1:
        pattern = 'Pattern_B'
    else:
        pattern = 'Pattern_C'
    return 'Viable', pattern, status

# ==============================================================================
# Part 5. 真实突变体 (论文 Table 2, 18 株 + WT)
# ==============================================================================
REAL_MUTANTS = {
    'WT':                      ({}, 'no change'),
    'wee1ts':                  ({'Vwee': 0.08}, 'Vwee=0.08'),
    'wee1D':                   ({'Vwee': 0.0, 'Vwee_': 0.0}, 'Vwee=Vwee_=0'),
    'mik1D':                   ({'Vmik': 0.0, 'Vmik_': 0.0}, 'Vmik=Vmik_=0'),
    'mik1D wee1ts':            ({'Vmik': 0.0, 'Vmik_': 0.0, 'Vwee': 0.08},
                                'Vmik=Vmik_=0, Vwee=0.08 (cut)'),
    'pyp3D':                   ({'Vpyp': 0.0}, 'Vpyp=0'),
    'cdc25ts':                 ({'V25': 0.05, 'V25_': 0.00025},
                                'V25=0.05, V25_=0.00025 (G2 block)'),
    'cdc25D':                  ({'V25': 0.0, 'V25_': 0.0},
                                'V25=V25_=0 (G2 block)'),
    'wee1ts cdc25D':           ({'Vwee': 0.08, 'V25': 0.0, 'V25_': 0.0},
                                'Vwee=0.08, V25=V25_=0 (quantized 112/170)'),
    'wee1ts cdc25D pyp3D':     ({'Vwee': 0.08, 'V25': 0.0, 'V25_': 0.0, 'Vpyp': 0.0},
                                'G2 block (no Cdc25, no Pyp3)'),
    'mik1D wee1ts cdc25D':     ({'Vmik': 0.0, 'Vmik_': 0.0, 'Vwee': 0.08,
                                'V25': 0.0, 'V25_': 0.0}, 'cut lethal'),
    'wee1ts cdc25ts':          ({'Vwee': 0.08, 'V25': 0.05, 'V25_': 0.00025},
                                'Vwee=0.08, V25=0.05, V25_=0.00025'),
    'wee1D cdc25D':            ({'Vwee': 0.0, 'Vwee_': 0.0, 'V25': 0.0, 'V25_': 0.0},
                                'Vwee=Vwee_=0, V25=V25_=0'),
    'rum1D':                   ({'k5': 0.0}, 'k5=0'),
    'rum1D wee1ts':            ({'k5': 0.0, 'Vwee': 0.08}, 'k5=0, Vwee=0.08 (too small)'),
    'ste9D':                   ({'V2': 0.0, 'V2c': 0.0}, 'V2=V2c=0'),
    'cdc13dbD':                ({'V2': 0.0, 'V2c': 0.0},
                                'V2=V2c=0 (same as ste9D; non-degradable Cdc13)'),
    'ste9D wee1ts':            ({'V2': 0.0, 'V2c': 0.0, 'Vwee': 0.08},
                                'V2=V2c=0, Vwee=0.08 (too small)'),
    'ste9D rum1D':             ({'V2': 0.0, 'V2c': 0.0, 'k5': 0.0}, 'V2=V2c=0, k5=0'),
    'ste9D rum1D wee1ts':      ({'V2': 0.0, 'V2c': 0.0, 'k5': 0.0, 'Vwee': 0.08},
                                'V2=V2c=0, k5=0, Vwee=0.08 (too small)'),
}
# 注: 论文 Table 2 中有两行以 "or" 并列两种菌株 (作者因两者模拟表型相同而合并成一行),
#     本脚本对每一行都建全 (两种菌株各自作为独立样本):
#   (1) "cdc25ts or cdc25D"  ->  参数不同 (ts: V25=0.05,V25_=0.00025; D: 0,0),
#                                建 'cdc25ts' 与 'cdc25D' 两条;
#   (2) "ste9D or cdc13dbD"  ->  参数相同 (V2=V2c=0), 建 'ste9D' 与 'cdc13dbD' 两条;
#                                cdc13dbD (单拷贝非降解 Cdc13) 在本 16 变量框架下
#                                与 ste9D 等价, 两条轨迹完全重合, 属预期行为。

# ==============================================================================
# Part 6. 随机采样方案 (纯随机 —— 无拓扑/结构约束)
# ==============================================================================
SAMPLING_MODULES = {
    'Wee1':   {'params': ['Vwee', 'Vwee_'],           'modes': ['KO', 'ts', 'OP']},
    'Mik1':   {'params': ['Vmik', 'Vmik_'],           'modes': ['KO', 'ts', 'OP']},
    'Cdc25':  {'params': ['V25', 'V25_'],             'modes': ['KO', 'ts', 'OP']},
    'Pyp3':   {'params': ['Vpyp'],                    'modes': ['KO', 'ts', 'OP']},
    'Rum1':   {'params': ['k5'],                      'modes': ['KO', 'OP']},
    'Ste9':   {'params': ['V2', 'V2c'],               'modes': ['KO', 'ts', 'OP']},
    'SK':     {'params': ['SK'],                      'modes': ['damp']},
    'Cdc13':  {'params': ['k1'],                      'modes': ['ts', 'OP']},
    'Slp1':   {'params': ['kas', 'kad'],              'modes': ['ts', 'OP']},
    'PPsys':  {'params': ['V4', 'k3'],                'modes': ['ts', 'OP']},
    'Growth': {'params': ['mu'],                      'modes': ['scale']},
}

# 模式 -> 模式类 映射 (对齐 mutant_aware_server 的 MODE_CLASS 词表)
MODE_CLASS = {
    'KO': 'loss',          # 敲除
    'OP': 'expression',    # 过表达
    'ts': 'activity',      # 温度敏感 / 活性降低
    'damp': 'activity',    # SK 阻尼 (Rum1Δ 样表型)
    'scale': 'event',      # 生长速率改变
}
# 模式类配额 (class 臂): loss/expression 主导, activity 次之。
MODE_CLASS_QUOTA = {
    'loss': 0.28, 'expression': 0.28, 'activity': 0.18,
    'stability': 0.20, 'structure': 0.03, 'event': 0.03,
}
MODE_FREQ = 'class'        # 由 --mode_freq 覆盖 (class = 配额; flat = 均匀)

# L1 (模块 x 模式) 网格轮转 —— 保证每个合法格子被抽到 (Latin-square 式覆盖)
_L1_GRID = [(m, mo) for m in SAMPLING_MODULES for mo in SAMPLING_MODULES[m]['modes']]
_L1_IDX = {'i': 0}

def next_L1_cell():
    cell = _L1_GRID[_L1_IDX['i'] % len(_L1_GRID)]
    _L1_IDX['i'] += 1
    return cell

def _sample_mode(module):
    """按 --mode_freq 抽取该模块的一个合法模式。"""
    modes = SAMPLING_MODULES[module]['modes']
    if MODE_FREQ == 'class':
        w = np.array([MODE_CLASS_QUOTA.get(MODE_CLASS.get(m, 'event'), 1.0)
                      for m in modes], dtype=float)
        return str(np.random.choice(modes, p=w / w.sum()))
    return str(np.random.choice(modes))          # flat: 均匀 1/k

def _strength(mode, key, rng):
    """按模式给出参数强度 (相对 WT 的绝对取值)。"""
    base = WT_parameters()[key]
    if mode == 'KO':
        return 0.0                                # 精确敲除 (保持 0, 不设边界缓冲)
    if mode == 'ts':
        return base * (0.15 * rng.beta(0.5, 2.0)) # 残余活性 0~15%
    if mode == 'OP':
        fold = float(np.exp(rng.uniform(np.log(1.5), np.log(10.0))))
        return base * fold                        # 过表达 1.5~10x (log-uniform)
    if mode == 'damp':
        return float(rng.uniform(0.004, 0.025))   # SK 阻尼 (绝对带)
    if mode == 'scale':
        return base * float(rng.uniform(0.5, 1.5))# 生长速率 0.5~1.5x
    return base

def sample_random_mutant(level, rng):
    """按分层 level 抽一个合成突变体, 返回 (overrides, name)。
    level=1 -> 1 个模块 (L1 网格轮转); level=2 -> 2 个; level=3 -> 3 个。
    (模块数上限 3, 与论文 Table 2 的最大突变基因数一致。) 所有模块均均匀随机抽取。"""
    if level == 1:
        m, mode = next_L1_cell()                  # 网格轮转 -> 覆盖保证
        pairs = [(m, mode)]
    else:
        ki = level                                # L2 -> 2, L3 -> 3
        mods = rng.choice(list(SAMPLING_MODULES.keys()), size=ki, replace=False)
        pairs = [(str(m), _sample_mode(str(m))) for m in mods]
    ovr, parts = {}, []
    for m, mode in pairs:
        for key in SAMPLING_MODULES[m]['params']:
            ovr[key] = _strength(mode, key, rng)
        parts.append(f'{m}:{mode}')
    return ovr, '+'.join(parts)

def param_vector(p):
    return np.array([p[k] for k in EXPORT_KEYS], dtype=float)

def is_duplicate(vec, ref_vecs):
    for v in ref_vecs:
        if np.allclose(vec, v, rtol=1e-5, atol=1e-9):
            return True
    return False

# ==============================================================================
# Part 7. 出生状态推导 + 单突变体仿真 (worker)
# ==============================================================================
def derive_wt_birth_state(verbose=True):
    """预热收敛 WT, 返回其出生态 (最后一次分裂后的状态)。"""
    p = WT_parameters()
    p['kS'] = p['kSmin']
    y0 = _default_birth_state()
    segs, info = run_event_sim(y0, p, tf=1200.0)
    if not info['post_div_states']:
        raise RuntimeError('WT warmup failed: no division events. ' + info['err'])
    y_birth = info['post_div_states'][-1].copy()
    cts = np.diff(info['div_times'])[-3:] if len(info['div_times']) >= 2 else []
    if verbose:
        print(f'  [WT warmup] divisions: {len(info["div_times"])}, '
              f'birth mass = {y_birth[iMASS]:.4f} raw '
              f'(= {y_birth[iMASS] / MASS_UNIT:.3f} paper-rel)')
        print(f'  [WT check ] cycle times: '
              f'{[f"{c:.1f}" for c in cts]} min (paper: ~150)')
    return y_birth

def mutant_birth_state(overrides, y_wt_birth, warmup_tf=1000.0, mass_floor=0.05):
    """逐株 warmup: 从严 WT 出生态出发, 把该突变体跑到自身吸引子, 返回其出生态。
    无分裂(G2/M 阻滞)或数值失败/质量塌缩 -> 退回 WT 出生态。"""
    p = WT_parameters(); p['kS'] = p['kSmin']
    for k, v in overrides.items():
        if k in p:
            p[k] = v
    y_wtb = np.array(y_wt_birth, float)
    try:
        _, info = run_event_sim(y_wtb.copy(), p, tf=warmup_tf)
    except Exception:
        return y_wtb.copy()
    if info['status'] in ('NumericalFailure', 'MaxEvents') or not info['post_div_states']:
        return y_wtb.copy()
    y0 = info['post_div_states'][-1].copy()
    if y0[iMASS] < mass_floor:
        return y_wtb.copy()
    return y0

def simulate_mutant(name, overrides, y_wt_birth, level=0, tf=TF_MIN, n_t=N_T):
    """完整仿真一个突变体, 返回紧凑 dict (可 pickle, 供多进程)。"""
    p = WT_parameters()
    p['kS'] = p['kSmin']
    for k, v in overrides.items():
        if k not in p:
            return dict(name=name, level=level, y=None, params_vec=None,
                        label='Invalid Param', pattern=None, n_div=0,
                        birth_rel=float('nan'), status='Invalid',
                        err=f'unknown param {k}')
        p[k] = v
    y0 = mutant_birth_state(overrides, y_wt_birth)
    try:
        segs, info = run_event_sim(y0, p, tf=tf)
    except Exception as e:
        return dict(name=name, level=level, y=None, params_vec=param_vector(p),
                    label='Inviable: Numerical Failure', pattern=None, n_div=0,
                    birth_rel=float('nan'), status='NumericalFailure', err=repr(e))
    if info['status'] == 'NumericalFailure':
        return dict(name=name, level=level, y=None, params_vec=param_vector(p),
                    label='Inviable: Numerical Failure', pattern=None, n_div=0,
                    birth_rel=float('nan'), status='NumericalFailure', err=info['err'])
    y = resample_uniform(segs, tf=tf, n_t=n_t)
    if np.max(y[iMASS]) > MASS_LIMIT:
        return dict(name=name, level=level, y=None, params_vec=param_vector(p),
                    label='Discarded_Mass', pattern=None,
                    n_div=len(info['div_times']), birth_rel=float('nan'),
                    status='Discarded_Mass', err='mass explosion')
    # mass 通道归一到论文相对尺度 (WT 出生 = 1.00, 分裂 = 2.00); 其余通道不变
    y[iMASS] = y[iMASS] / MASS_UNIT
    # cut 表型: 未完成复制 (RDNA<0.99) 即发生分裂
    if any(rd < 0.99 for rd in info.get('rdna_at_div', [])):
        return dict(name=name, level=level, y=y, params_vec=param_vector(p),
                    label='Inviable: cut (premature division)', pattern=None,
                    n_div=len(info['div_times']),
                    birth_rel=(info['birth_masses'][-1] / MASS_UNIT
                               if info['birth_masses'] else float('nan')),
                    status=info['status'], err='')
    # Too Small 判据用稳态出生质量: 从最后出生状态再续跑 1050 min 评估稳态
    births_lab = None
    if info['post_div_states']:
        try:
            _, info_ext = run_event_sim(info['post_div_states'][-1], dict(p), tf=1050.0)
            if info_ext['birth_masses']:
                births_lab = info_ext['birth_masses']
        except Exception:
            pass
    label, pattern, status = assign_labels(info, y, births_lab=births_lab)
    birth_rel = (births_lab[-1] / MASS_UNIT if births_lab
                 else (info['birth_masses'][-1] / MASS_UNIT if info['birth_masses']
                       else float('nan')))
    return dict(name=name, level=level, y=y, params_vec=param_vector(p),
                label=label, pattern=pattern, n_div=len(info['div_times']),
                birth_rel=birth_rel, status=status, err='')

def _sim_worker(payload):
    name, overrides, level, y_wt_birth = payload
    return simulate_mutant(name, overrides, y_wt_birth, level=level)

# ==============================================================================
# Part 8. 数据集生成 + .npz 导出 + CLI
# ==============================================================================
def generate_dataset(n_synthetic=DEFAULT_SAMPLES, level_ratios=None,
                     num_workers=None, output_name=None, output_dir=None,
                     seed=SEED, mode_freq='class', max_attempts=None):
    """生成完整数据集 (18 真实突变体 + n_synthetic 合成随机突变体) 并导出单个 .npz。"""
    global MODE_FREQ
    MODE_FREQ = mode_freq
    if level_ratios is None:
        level_ratios = DEFAULT_LEVEL_RATIOS

    if output_name is None:
        output_name = f"fission_rand_{n_synthetic}_{int(TF_MIN)}m_{N_T}s.npz"
    if output_dir is None:
        output_dir = (os.path.dirname(os.path.abspath(__file__)) if os.name == 'nt'
                      else "/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass
    dataset_path = os.path.join(output_dir, output_name)

    if max_attempts is None:
        max_attempts = max(50000, n_synthetic * 10)

    # 可复现: 固定 RNG + 固定 L1 网格轮转顺序
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    np.random.RandomState(seed).shuffle(_L1_GRID)
    _L1_IDX['i'] = 0

    print('=' * 85)
    print(' Fission Yeast (Sveiczer 2000) — Random-Sampling Dataset Generator (Server Ed.)')
    print(f' Target Output : {dataset_path}')
    print(f' Mode arm      : {mode_freq}   Seed: {seed}')
    print('=' * 85)

    # ---- Phase 1: WT 出生状态 ----
    print('\n[Phase 1/4] Deriving WT birth state (warmup convergence) ...')
    y_wt_birth = derive_wt_birth_state(verbose=True)

    # ---- Phase 2: 组装任务队列 (真实 + 合成) ----
    print(f'\n[Phase 2/4] Sampling tasks: {len(REAL_MUTANTS)} real + {n_synthetic} synthetic ...')
    ordered = []          # [(name, overrides, level)]
    for name, (ovr, desc) in REAL_MUTANTS.items():
        ordered.append((name, dict(ovr), 0))
    n_real = len(ordered)

    real_vecs = [param_vector({**WT_parameters(), **ovr}) for ovr, _ in REAL_MUTANTS.values()]
    syn_vecs = []

    r = np.array(level_ratios, dtype=float); r = r / r.sum()
    n_l = [int(math.floor(x * n_synthetic)) for x in r[:2]]
    n_l.append(n_synthetic - sum(n_l))
    level_assign = [1] * n_l[0] + [2] * n_l[1] + [3] * n_l[2]
    np.random.shuffle(level_assign)
    print(f' -> Level targets: L1={n_l[0]} L2={n_l[1]} L3={n_l[2]}')

    attempts = blocked = 0
    counters = Counter()
    while len(syn_vecs) < n_synthetic and attempts < max_attempts:
        attempts += 1
        level = level_assign[len(syn_vecs)]
        ovr, desc = sample_random_mutant(level, rng)
        pv = param_vector({**WT_parameters(), **ovr})
        if is_duplicate(pv, real_vecs) or is_duplicate(pv, syn_vecs):
            blocked += 1                 # 与真实/已接受样本参数向量重合 -> 丢弃
            continue
        syn_vecs.append(pv)
        idx = len(syn_vecs)
        sname = f'Rand_{desc}_L{level}_{idx}'.replace(' ', '_')
        ordered.append((sname, ovr, level))
        counters[level] += 1
    print(f' -> Sampling done: {len(syn_vecs)} synthetic in {attempts} attempts '
          f'({blocked} blocked by dedup).')

    # ---- Phase 3: 多进程仿真 ----
    if num_workers is None:
        num_workers = min(60, os.cpu_count() or 4)
    print(f'\n[Phase 3/4] Simulating {len(ordered)} trajectories on {num_workers} workers ...')
    t0 = time.time()
    results, sim_counts = {}, Counter()
    log_step = max(50, len(ordered) // 40)
    payloads = [(nm, ovr, lv, y_wt_birth) for nm, ovr, lv in ordered]
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(_sim_worker, pl): pl[0] for pl in payloads}
        for i, fut in enumerate(as_completed(futs)):
            try:
                res = fut.result()
            except Exception as e:
                nm = futs[fut]
                results[nm] = dict(name=nm, level=0, y=None, params_vec=None,
                                   label='Inviable: Numerical Failure', pattern=None,
                                   n_div=0, birth_rel=float('nan'), status='failed', err=repr(e))
                sim_counts['failed'] += 1
                continue
            results[res['name']] = res
            if res['y'] is not None:
                sim_counts['ok_real' if res['level'] == 0 else 'ok_synth'] += 1
            elif res['label'] == 'Discarded_Mass':
                sim_counts['discarded'] += 1
            else:
                sim_counts['failed'] += 1
            if (i + 1) % log_step == 0 or (i + 1) == len(ordered):
                el = max(0.001, time.time() - t0)
                print(f'  |-- [{i+1:5d}/{len(ordered)}] valid(real/synth)='
                      f'{sim_counts["ok_real"]}/{sim_counts["ok_synth"]}  '
                      f'discarded={sim_counts["discarded"]}  failed={sim_counts["failed"]}  '
                      f'{ (i+1)/el:5.1f} samples/s', flush=True)
    print(f' -> Simulation completed in {time.time()-t0:.1f}s.')

    # ---- Phase 4: 打包导出 ----
    print('\n[Phase 4/4] Packaging & exporting .npz ...')
    data, params, names, labels, patterns, levels = [], [], [], [], [], []
    for nm, ovr, lv in ordered:
        res = results.get(nm)
        if res is None or res['y'] is None:
            continue
        data.append(res['y'])
        params.append(res['params_vec'])
        names.append(nm)
        labels.append(res['label'])
        patterns.append(str(res['pattern']))
        levels.append(lv)
    data       = np.array(data)
    params     = np.array(params)
    names      = np.array(names)
    labels     = np.array(labels)
    patterns   = np.array(patterns)
    levels     = np.array(levels, dtype=int)

    np.savez_compressed(dataset_path,
                        data=data, params=params,
                        param_names=np.array(EXPORT_KEYS),
                        mutant_names=names, labels=labels,
                        pattern_labels=patterns, levels=levels,
                        var_names=np.array(VAR_NAMES),
                        time_grid=np.linspace(0.0, TF_MIN, N_T))

    # ---- 摘要报告 ----
    n_ok_real = sum(1 for l in levels if l == 0)
    n_ok_synth = sum(1 for l in levels if l > 0)
    print('\n' + '=' * 85)
    print(' DATASET GENERATION SUMMARY')
    print('=' * 85)
    print(f' Export path        : {dataset_path}')
    print(f' data shape         : {data.shape}   (N, {N_VAR}, {N_T})')
    print(f' params shape       : {params.shape}   param_names: {len(EXPORT_KEYS)}')
    print(f' real saved         : {n_ok_real} / {n_real}')
    print(f' synthetic saved    : {n_ok_synth} / {n_synthetic}')
    print(f' dedup blocked      : {blocked} candidates')
    print(f' discarded (mass)   : {sim_counts["discarded"]}')
    print(f' failed             : {sim_counts["failed"]}')
    print(f' label distribution : {dict(Counter(labels))}')
    print(f' synthetic levels   : {dict(Counter(levels[levels > 0].tolist()))}')
    print('=' * 85)


def main():
    if os.name == 'nt':
        default_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        default_dir = "/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast"

    parser = argparse.ArgumentParser(
        description='Fission Yeast (Sveiczer 2000) Random-Sampling Dataset Generator (Server Edition)')
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES,
                        help='合成突变体数量 (默认 1000)')
    parser.add_argument('--workers', type=int, default=None,
                        help='并行进程数 (默认 min(60, cpu_count))')
    parser.add_argument('--level_ratios', type=float, nargs=3, default=DEFAULT_LEVEL_RATIOS,
                        help='L1/L2/L3 配额比例 (默认 0.45 0.35 0.20; 模块数上限 3)')
    parser.add_argument('--mode_freq', choices=['class', 'flat'], default='class',
                        help='模式采样臂: class=模式类配额(默认); flat=均匀(1/k)')
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子 (默认 20000)')
    parser.add_argument('--output_name', type=str, default=None, help='导出 .npz 文件名')
    parser.add_argument('--output_dir', type=str, default=default_dir, help='导出目录')
    args = parser.parse_args()

    print(f"Executing fission random sampler (Samples={args.samples}, Workers={args.workers}, "
          f"ModeFreq={args.mode_freq}, Seed={args.seed}) ...", flush=True)
    generate_dataset(n_synthetic=args.samples, level_ratios=args.level_ratios,
                     num_workers=args.workers, output_name=args.output_name,
                     output_dir=args.output_dir, seed=args.seed, mode_freq=args.mode_freq)
    print('\n[SUCCESS] Fission random sampler finished cleanly!', flush=True)


if __name__ == '__main__':
    main()
