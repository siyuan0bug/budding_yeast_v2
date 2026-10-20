"""Unchanged Fission KO/ts/OP/damp/scale amplitude draws."""
import numpy as np
from .parameters import WT_parameters

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
