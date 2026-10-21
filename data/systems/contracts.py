"""Small, picklable contracts shared by independent biological systems."""
from dataclasses import dataclass, field
from typing import Any, Protocol

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


@dataclass(frozen=True)
class Recipe:
    system: str
    sampler: str
    ratios: tuple[float, ...]
    mutation: str = 'mutant_aware'
    distribution: str = 'mutant_aware'
    dtype: str = 'float64'
    quota_rounding: str = 'floor'
    shuffle_levels: bool = True
    queue_multiplier: float = 1.0
    notes: str = ''
    mode_freq: str = 'class'
    default_samples: int = 100000
    default_seed: int = 42
    default_pilot: int = 1000



class SystemAdapter(Protocol):
    system_id: str
    spec: object

    def prepare_context(self, config): ...
    def real_candidates(self, include_real): ...
    def references(self): ...
    def is_collision(self, overrides, references, sampler): ...
    def add_reference(self, references, overrides): ...
    def rescue(self, overrides, description, sampler, rng): ...
    def rescue_rng(self, sampler): ...
    def pairs_from_description(self, description): ...
    def special_candidates(self, recipe, config, sampler, report): ...
    def simulate(self, candidate, config, context): ...
    def parameter_result(self, candidate, context): ...
    def accept_result(self, result, state, config): ...
    def finalize(self, results, config, recipe): ...
    def expected_real_count(self): ...
    def validate_config(self, config, recipe): ...
    def handles_derived(self, recipe): ...
    def generate_derived(self, config): ...

