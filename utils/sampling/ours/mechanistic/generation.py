"""Use the same acquisitions on new canonical proposals, outside the frozen pool."""
from . import METHOD_NAMES
from .selectors import select, prepare_training
from .systems import BiologicalODE
from ...contracts import SamplingContext, QueriedTraining


class MechanisticGeneration:
    def __init__(self, name, initial, truth, seed, *, system_id='budding'):
        if name not in METHOD_NAMES:
            raise ValueError('Unknown portable generation method')
        self.name, self.seed, self.calls = name, int(seed), 0
        self.system_id = system_id
        self.train = QueriedTraining(initial, truth)
        self.predictor = None
        self._prepared = None

    def refresh(self, inputs, truth, predictor):
        # References only: no duplicate trajectory snapshots or model copies.
        self.train = QueriedTraining(inputs, truth)
        self.predictor = predictor
        self._prepared = prepare_training(self.train, BiologicalODE(self.system_id))

    def select(self, bank, count):
        if self.predictor is None:
            raise ValueError('Train the current surrogate before generating mechanism queries')
        context = SamplingContext(bank, self.train, self.predictor, self.seed + self.calls,
                                  self.predictor.device, system_id=self.system_id)
        result = select(context, count, self.name, training_preparation=self._prepared)
        self.calls += 1
        # Domain bank IDs are its positional rows; only existing legal proposals
        # are selected. No continuous interpolation or mutation-rule overrides.
        result.diagnostics.update(query_source='fresh canonical proposal bank', pool_outside_allowed=True)
        return result.ids, result.diagnostics

    def state_dict(self):
        return dict(version='mechanistic_generation_v1', name=self.name, seed=self.seed, calls=self.calls,
                    system_id=self.system_id)

    def load_state_dict(self, state):
        if (state.get('version') != 'mechanistic_generation_v1' or state['name'] != self.name
                or state['seed'] != self.seed or state.get('system_id') != self.system_id):
            raise ValueError('Mechanistic generation identity changed')
        self.calls = int(state['calls'])
