"""Fission physical scales, solver window, state order and event limits."""
import numpy as np

TF_MIN      = 300.0     # 仿真时长 (min): 2 个 WT 完整周期 (MDT=150)

N_T         = 500       # 均匀时间网格点数

MASS_LIMIT  = 50.0      # 质量爆炸判据 (Discarded_Mass, 原始尺度)

MASS_UNIT   = 1.4048    # WT 极限环出生质量 (原始尺度); 论文 Table 2 相对尺度基准

TOO_SMALL_M = 0.625     # Too Small 阈值 (原始尺度) = 相对 0.445 x MASS_UNIT。

MAX_EVENTS  = 80        # 单次仿真事件数上限

DT_EPS      = 1e-6      # 事件后时间前跳 (min), 防止事件零点重复触发

SEED        = 20000

DEFAULT_SAMPLES      = 1000

DEFAULT_LEVEL_RATIOS = [0.45, 0.35, 0.20]   # L1/L2/L3 配额 (1/2/3 个模块, 上限 3 = Table 2 最大突变基因数)

VAR_NAMES = ['mass', 'RDNA', 'MPF', 'preMPF', 'Rum1', 'Rum1P', 'CR', 'CRP',
             'Ste9', 'Mik1', 'Wee1', 'Cdc25', 'Slp1', 'Slp1a', 'Inh', 'PI']

(iMASS, iRDNA, iMPF, iPREMPF, iRUM1, iRUM1P, iCR, iCRP,
 iSTE9, iMIK1, iWEE1, iCDC25, iSLP1, iSLP1A, iINH, iPI) = range(16)

N_VAR = 16
