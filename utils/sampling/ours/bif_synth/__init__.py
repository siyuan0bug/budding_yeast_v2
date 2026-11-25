"""R26/R27: Topology-Preserving & Pool-Anchored Bifurcation / Hybrid Allelic-Series Query Synthesis.

Methods:
- ours_pool_bif_synth: Pool-anchored (zero external generator, zero pool labels) 9-pt continuous
  bifurcation synthesis preserving exact modes and COMPOSITE_REAL_EVIDENCE composites.
- ours_pool_mode_bif_v2_synth: Pool-anchored composite-safe, scale-normalized gain-gated allelic-series
  mode switching (KO/Suppression/Overexpression) + continuous bifurcation synthesis.
- ours_pool_mode_bif_synth: R26 v1 unconstrained two-level mode+dose synthesis.
- ours_bif_power_synth: 24k out-of-pool bank + single-axis 9-point bifurcation synthesis.
- ours_grad_bif_synth: 24k out-of-pool bank + multi-axis steepest coarse-to-fine bifurcation synthesis.
- ours_power_synth: 24k out-of-pool bank + pure Power (strict same-source control).
"""
from .synthesis import BifurcationSynthesis, BIF_SYNTH_GEN_METHODS as _OUT_METHODS
from .pool_mode_synthesis import PoolModeBifSynthesis, POOL_MODE_GEN_METHODS

BIF_SYNTH_GEN_METHODS = _OUT_METHODS + POOL_MODE_GEN_METHODS

__all__ = [
    'BifurcationSynthesis',
    'PoolModeBifSynthesis',
    'POOL_MODE_GEN_METHODS',
    'BIF_SYNTH_GEN_METHODS',
]
