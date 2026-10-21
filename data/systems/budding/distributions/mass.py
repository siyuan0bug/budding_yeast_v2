"""mass amplitude distribution; ranges and draw order preserved."""
import numpy as np
from ..catalog import require_mutation

def sample_scaling_factor(mode, g_name=None, *, rng=None):
    if g_name is not None:
        require_mutation(g_name, mode)
    rng = np.random if rng is None else rng
    if mode == 'GAL-OE':
        if g_name == 'CDC20':
            return 10.0 ** float(rng.uniform(np.log10(0.5), np.log10(12.0)))
        elif g_name == 'CLB2':
            return 10.0 ** float(rng.uniform(np.log10(0.002), np.log10(1.5)))
        elif g_name == 'SIC1':
            return 10.0 ** float(rng.uniform(np.log10(0.03), np.log10(0.36)))
        elif g_name == 'CDC6':
            return 10.0 ** float(rng.uniform(np.log10(0.1), np.log10(0.6)))
        elif g_name == 'CLB5':
            return 10.0 ** float(rng.uniform(np.log10(0.004), np.log10(0.03)))
        elif g_name == 'PDS1':
            return 10.0 ** float(rng.uniform(np.log10(0.02), np.log10(0.25)))
        elif g_name == 'CLN2':
            return 10.0 ** float(rng.uniform(np.log10(0.03), np.log10(0.3)))
        elif g_name == 'CLN3':
            return 10.0 ** float(rng.uniform(np.log10(1.0), np.log10(5.0)))
        elif g_name == 'TEM1':
            return 10.0 ** float(rng.uniform(np.log10(2.0), np.log10(8.0)))
        elif g_name == 'CDC15':
            return 10.0 ** float(rng.uniform(np.log10(3.0), np.log10(20.0)))
        elif g_name == 'ESP1':
            return 10.0 ** float(rng.uniform(np.log10(1.5), np.log10(6.0)))
        elif g_name == 'CDC14':
            return 10.0 ** float(rng.uniform(np.log10(0.3), np.log10(1.2)))
        elif g_name == 'NET1':
            return 10.0 ** float(rng.uniform(np.log10(0.15), np.log10(0.6)))
        elif g_name == 'SWI5':
            return 10.0 ** float(rng.uniform(np.log10(0.02), np.log10(0.1)))
        else:
            log_alpha = rng.uniform(np.log10(2.0), np.log10(30.0))
            return 10.0 ** float(log_alpha)
    elif mode == 'MC':
        log_alpha = rng.uniform(np.log10(1.5), np.log10(30.0))
        return 10.0 ** float(log_alpha)
    elif mode == 'ts':
        if g_name == 'CDC14':
            return float(rng.uniform(0.05, 0.35))
        u = rng.beta(a=0.5, b=2.0)
        return float(0.2 * u)
    elif mode == 'DAmP':
        val = rng.normal(loc=0.2, scale=0.05)
        return float(np.clip(val, 0.05, 0.4))
    elif mode == 'pLOF':
        return float(rng.uniform(0.1, 0.5))
    elif mode == 'db':
        if g_name == 'CLB2':
            return float(rng.uniform(0.04, 0.15))
        elif g_name == 'PDS1':
            return float(rng.uniform(0.0, 0.05))
        elif g_name == 'CLB5':
            return 0.0
        elif g_name == 'SIC1':
            return 0.0
        else:
            return 0.0
    return 1.0
