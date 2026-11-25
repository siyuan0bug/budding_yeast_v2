"""Switch-D2: Task-Aligned Epistemic Relative Variance & Stochastic Weighted D^2 Sampling.

Unified framework for both fixed-pool AL (Track A) and level-marginal-preserving
legal query synthesis (Track C).
"""
from .selector import select_switch_d2_pool, select_switch_power
from .synthesis import SwitchD2Synthesis, SWITCH_D2_GEN_METHODS

__all__ = [
    'select_switch_d2_pool',
    'select_switch_power',
    'SwitchD2Synthesis',
    'SWITCH_D2_GEN_METHODS',
]
