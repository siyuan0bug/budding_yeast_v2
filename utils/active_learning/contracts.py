"""Small data contracts at the strategy, simulation and training boundaries."""
from dataclasses import dataclass
from typing import Any, Protocol

# Biological simulation contracts remain owned by data.systems.
from budding_yeast_v2.data.systems.contracts import Candidate, SimulationResult


@dataclass(frozen=True)
class ALConfig:
    trigger_every_n_epochs: int = 10
    strategy: str = 'al4pde_random'
    num_add: int = 5000
    perturbation: float = 0.1
    mae_threshold: float = 0.1
    initial_train_size: int = 5000
    initial_selection: str = 'kmeans'
    diversity_weight: float = 0.5
    pool_subset_size: int = 10000
    uncertainty_metric: str = 'variance'
    random_seed: int = 42
    atlas_csv_path: str | None = None
    al_pool_dataset: str | None = None
    target_weight: float = 0.4
    mass_weight: float = 0.5
    novelty_weight: float = 0.6
    family_weight: float = 0.2
    ensemble_uncertainty_mode: str = 'loss_mse'
    power_beta: float = 1.0
    bait_sketch_dim: int = 512


@dataclass(frozen=True)
class SelectionPlan:
    candidate_ids: tuple[int, ...]
    pool_token: int
    pool_revision: int
    strategy: str
    target_valid_num: int | None = None

    @property
    def fill_policy(self):
        return 'single_batch' if self.target_valid_num is None else 'refill_valid'


@dataclass(frozen=True)
class SelectionContext:
    engine: Any
    model: Any
    datamodule: Any
    device: Any


class Acquisition(Protocol):
    def select(self, context: SelectionContext, budget: int) -> SelectionPlan | None: ...


@dataclass
class QueryBatch:
    data: dict | None
    consumed_ids: tuple[int, ...]
    accepted_ids: tuple[int, ...]
    strategy: str

    def __post_init__(self):
        if self.data is None:
            return
        size = len(self.data['X'])
        for key in ('P', 'Y', 'names', 'labels', 'Y_real', 'P_real', 'biological_labels'):
            if key in self.data and len(self.data[key]) != size:
                raise ValueError(f'Query row alignment error: {key}')

    @property
    def n_accepted(self):
        return 0 if self.data is None else len(self.data['X'])
