"""Small, picklable contracts shared by independent biological systems."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    model_version: str
    variable_names: tuple[str, ...]
    param_names: tuple[str, ...]
    horizon: float
    steps: int
    benchmark_profile: str
    mass_scale: float = 1.0


@dataclass
class Candidate:
    sample_id: int
    name: str
    overrides: dict
    level: int
    is_real: bool = False
    refinement: tuple | None = None
    pair_id: int = -1
    metadata: dict = field(default_factory=dict)


@dataclass
class SimulationResult:
    candidate: Candidate
    trajectory: np.ndarray | None
    params: np.ndarray
    y_init: np.ndarray
    label: str
    pattern: str | None
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationConfig:
    recipe: str
    n_samples: int | None = None
    seed: int | None = None
    workers: int = 1
    output: str = ""
    mutation_profile: str | None = None
    distribution_profile: str | None = None
    level_ratios: tuple[float, ...] | None = None
    include_real: bool = True
    parameter_only: bool = False
    mode_freq: str | None = None
    temperature: float = 0.8
    curv_thresh: float = 0.01
    flat_prob: float = 0.20
    max_attempts: int | None = None
    ko_interval_max: float = 0.0001
    pilot_samples: int | None = None
    overwrite: bool = False
    paired_output: str = ""
    source: str = ""
    wt_fold_dist: str = "tier2"
    enable_bisection: bool = True
    nan_cap_frac: float = 0.05
    backlog_thresh: float = 0.20
    max_attempts_mult: int = 30
