"""Unchanged 16-state Fission RHS and three event guards."""
import numpy as np
from .constants import iMASS, iRDNA, iMPF, iPREMPF, iPI

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
