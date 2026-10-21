"""Original MA inverse-CDF amplitude mapping, shared by both Sobol strategies."""
import numpy as np
from ..catalog import require_mutation
from scipy.stats import beta, norm
_MC_NS = np.arange(2, 31)
_MC_WEIGHTS = 1.0 / _MC_NS
_MC_PROBS = _MC_WEIGHTS / _MC_WEIGHTS.sum()
_MC_CDF = np.cumsum(_MC_PROBS)

def sample_scaling_factor_sobol(mode, g_name, u):
    """
    Maps a low-discrepancy uniform coordinate u in [0, 1] to the exact physiological
    parameter scaling factor via Inverse-CDF (Quantile Transformation).
    Matches mutant_aware.py ranges and distributions 100% identically.
    """
    if g_name is not None:
        require_mutation(g_name, mode)
    u = np.clip(u, 1e-06, 1.0 - 1e-06)
    if mode == 'GAL-OE':
        if g_name == 'CDC20':
            lo, hi = (np.log10(0.5), np.log10(12.0))
        elif g_name == 'CLB2':
            lo, hi = (np.log10(0.002), np.log10(1.5))
        elif g_name == 'SIC1':
            lo, hi = (np.log10(0.03), np.log10(0.36))
        elif g_name == 'CDC6':
            lo, hi = (np.log10(0.1), np.log10(0.6))
        elif g_name == 'CLB5':
            lo, hi = (np.log10(0.004), np.log10(0.03))
        elif g_name == 'PDS1':
            lo, hi = (np.log10(0.02), np.log10(0.25))
        elif g_name == 'CLN2':
            lo, hi = (np.log10(0.06), np.log10(0.3))
        elif g_name == 'CLN3':
            lo, hi = (np.log10(1.0), np.log10(5.0))
        elif g_name == 'TEM1':
            lo, hi = (np.log10(2.0), np.log10(8.0))
        elif g_name == 'CDC15':
            lo, hi = (np.log10(3.0), np.log10(20.0))
        elif g_name == 'ESP1':
            lo, hi = (np.log10(1.5), np.log10(6.0))
        elif g_name == 'CDC14':
            lo, hi = (np.log10(0.3), np.log10(1.2))
        elif g_name == 'NET1':
            lo, hi = (np.log10(0.15), np.log10(0.6))
        elif g_name == 'SWI5':
            lo, hi = (np.log10(0.02), np.log10(0.1))
        else:
            lo, hi = (np.log10(2.0), np.log10(30.0))
        return float(10.0 ** (lo + u * (hi - lo)))
    elif mode == 'MC':
        idx = np.searchsorted(_MC_CDF, u)
        return float(_MC_NS[min(idx, len(_MC_NS) - 1)])
    elif mode == 'ts':
        if g_name == 'CDC14':
            return float(0.05 + u * (0.35 - 0.05))
        if g_name in ('TEM1', 'ESP1'):
            lo, hi = (np.log10(0.0005), np.log10(0.004))
            return float(10.0 ** (lo + u * (hi - lo)))
        if g_name == 'NET1':
            lo, hi = (np.log10(0.0125), np.log10(0.1))
            return float(10.0 ** (lo + u * (hi - lo)))
        if g_name in ('CDC20', 'CDC15', 'APC_Core'):
            return float(0.0 + u * 0.01)
        val = float(0.2 * beta.ppf(u, 0.5, 2.0))
        return val
    elif mode == 'DAmP':
        val = float(norm.ppf(u, loc=0.2, scale=0.05))
        return float(np.clip(val, 0.05, 0.4))
    elif mode == 'pLOF':
        return float(0.1 + u * (0.5 - 0.1))
    elif mode == 'db':
        if g_name == 'CLB2':
            return float(0.02125 + u * (0.17 - 0.02125))
        elif g_name == 'PDS1':
            return float(0.0 + u * 0.01)
        elif g_name in ('CLB5', 'SIC1'):
            return float(0.0 + u * 0.01)
        else:
            return 0.0
    return 1.0
