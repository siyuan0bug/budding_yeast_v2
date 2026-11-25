"""Compositional Epistasis & Bifurcation-Aware Sampling (CEBS / R25).

Module 1: Compositional order marginal preservation (exact native L1/L2/L3 prior).
Module 2: 0-ODE virtual single-module nonlinear epistasis magnitude M_epi(S).
Module 3: 0-ODE continuous-dose bifurcation-boundary query synthesis (Track C).
"""
from .selector import select_order_power_pool, select_epistasis_pool
from .synthesis import CEBSSynthesis, CEBS_GEN_METHODS

__all__ = [
    'select_order_power_pool',
    'select_epistasis_pool',
    'CEBSSynthesis',
    'CEBS_GEN_METHODS',
]
