"""DMCED: Disagreement-Metric Combinatorial Experimental Design.

Unified multi-channel epistemic disagreement & risk-weighted Voronoi acquisition
for both fixed-pool active learning (Track A) and hypergraph-factorized legal
query synthesis (Track C).
"""
from .selector import select_dmced_pool, select_dmced_power
from .synthesis import DMCEDSynthesis, DMCED_GEN_METHODS

__all__ = [
    'select_dmced_pool',
    'select_dmced_power',
    'DMCEDSynthesis',
    'DMCED_GEN_METHODS',
]
