"""Input-only selection views shared by sampling methods and workflows."""
from dataclasses import dataclass,field
from typing import Callable
import numpy as np

def readonly(value):
    value = np.asarray(value).copy()
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class PoolInputs:
    ids: np.ndarray
    params: np.ndarray
    initial: np.ndarray
    metadata: tuple = ()

    def __post_init__(self):
        for name in ('ids', 'params', 'initial'):
            object.__setattr__(self, name, readonly(getattr(self, name)))
        if len(self.ids) != len(self.params) or len(self.ids) != len(self.initial):
            raise ValueError('Misaligned input-only pool view')


@dataclass(frozen=True)
class QueriedTraining:
    inputs: PoolInputs
    trajectories: np.ndarray


@dataclass(frozen=True)
class SamplingContext:
    pool: PoolInputs
    train: QueriedTraining
    predictor: object
    seed: int
    device: str
    sketch_dim: int = 512
    label_reader: Callable | None = None  # exclusively authorized B methods
    prediction_batch_size: int = 128
    system_id: str = 'budding'
    time_grid: np.ndarray | None = None


@dataclass
class SamplingResult:
    ids: np.ndarray
    weights: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    def validate(self, allowed_ids, budget, *, replacement_slots=False):
        self.ids = np.asarray(self.ids, dtype=np.int64)
        size_valid = 1 < len(self.ids) <= budget if replacement_slots else len(self.ids) == budget
        if (self.ids.ndim != 1 or not size_valid
                or len(set(map(int, self.ids))) != len(self.ids)
                or not set(map(int, self.ids)) <= set(map(int, allowed_ids))):
            raise ValueError('Selector returned duplicate/out-of-domain IDs or wrong budget')
        if replacement_slots and self.diagnostics.get('draw_slots') != budget:
            raise ValueError('Replacement sampling must retain the original draw-slot count')
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)
            if (self.weights.shape != self.ids.shape or not np.isfinite(self.weights).all()
                    or (self.weights < 0).any()):
                raise ValueError('Malformed selection weights')
        return self


def __getattr__(name):
    if name == 'LabelOracle':
        from .query import LabelOracle
        return LabelOracle
    if name in ('ALConfig','SelectionPlan','SelectionContext','Acquisition','QueryBatch','Candidate','SimulationResult'):
        from .online import contracts
        return getattr(contracts,name)
    raise AttributeError(name)
