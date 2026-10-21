"""Fission adapter: its own warmup context, dedup index, collision rescue and labels."""
import numpy as np

from .specification import SPEC
from .constants import MASS_UNIT, TF_MIN
from .parameters import WT_parameters, param_vector
from .real_mutants import real_tasks
from .samplers.references import FissionReferenceIndex
from .samplers.rescue import rescue_fission
from ..contracts import Candidate, SimulationResult
from ..registry import resolve_config, candidate_name
from ...simulation_guards import simulation_deadline


# Fission 单样本包含 1000 min 预热 + 300 min 主仿真 + 1050 min 稳态出生质量续跑。
# 5 s 在 32~64 并发下低于实际墙钟时间，会误杀合法样本。
SIMULATION_TIMEOUT = 30.0


class FissionAdapter:
    system_id = 'fission'
    spec = SPEC

    def prepare_context(self, config):
        from .states import derive_wt_birth_state
        with simulation_deadline(SIMULATION_TIMEOUT):
            return derive_wt_birth_state()

    def real_candidates(self, include_real):
        return ([Candidate(-i - 1, n, o, 0, True)
                 for i, (n, o) in enumerate(real_tasks())] if include_real else [])

    def references(self):
        return FissionReferenceIndex(
            [param_vector({**WT_parameters(), **ovr}) for _, ovr in real_tasks()])

    def is_collision(self, overrides, references, sampler):
        return references.collides(param_vector({**WT_parameters(), **overrides}))

    def add_reference(self, references, overrides):
        references.add(param_vector({**WT_parameters(), **overrides}))

    def rescue(self, overrides, description, sampler, rng):
        return rescue_fission(overrides, description, rng)

    def rescue_rng(self, sampler):
        return sampler.rng

    def pairs_from_description(self, description):
        return None

    def parameter_result(self, candidate, context):
        params = param_vector({**WT_parameters(), **candidate.overrides})
        y_init = np.array(context, copy=True)
        y_init[0] /= MASS_UNIT
        return SimulationResult(candidate, None, params, y_init, 'Unlabelled', None, 'parameter_only')

    def simulate(self, candidate, config, context):
        from .simulation import simulate_mutant

        params = param_vector({**WT_parameters(), **candidate.overrides})
        row = SimulationResult(candidate, None, params, np.array(context, copy=True),
                               'Inviable: Numerical Failure', None, 'numerical_failure')
        try:
            with simulation_deadline(SIMULATION_TIMEOUT):
                result = simulate_mutant(candidate.name, candidate.overrides, context, level=candidate.level)
            if result['y'] is not None and result['status'] == 'End' and result.get('completed_time', 0) >= TF_MIN:
                row.trajectory = result['y']
                row.params = result['params_vec']
                # Exported y_init is in the same units as exported trajectories.
                row.y_init = result['y_init'].copy()
                row.y_init[0] /= MASS_UNIT
                row.label = result['label']
                row.pattern = result['pattern']
                row.status = 'success'
                row.diagnostics = {'n_div': result['n_div'], 'birth_rel': result['birth_rel']}
            elif result['label'] == 'Discarded_Mass':
                row.label = result['label']
                row.status = 'discarded_mass'
        except Exception as exc:
            row.diagnostics = {'error': repr(exc)}
        return row

    def accept_result(self, result, state, config):
        return True

    def finalize(self, results, config, recipe):
        return results

    def expected_real_count(self):
        return 20

    def special_candidates(self, recipe, config, sampler, report):
        return None

    def validate_config(self, config, recipe):
        if (config.mutation_profile is not None or config.distribution_profile is not None
                or config.mode_freq not in ('class', 'flat')):
            raise ValueError('Fission uses its own profile and supports class/flat mode frequencies')
        if config.parameter_only:
            raise ValueError('Fission initial states require per-mutant warmup; parameter-only pools are Budding-only')

    def handles_derived(self, recipe):
        return False

    def generate_derived(self, config):
        raise ValueError('Fission has no derived paired/source datasets')


ADAPTER = FissionAdapter()


class FissionSampler:
    """Adapter for systems/fission/samplers/random.py."""

    def __init__(self, recipe, config):
        from .samplers import random as fission

        config = resolve_config(config)
        self.recipe, self.config = recipe, config
        self.backend = None
        self.kernel = None
        self.sobol = None
        self.module = fission
        fission.MODE_FREQ = config.mode_freq
        fission._L1_GRID[:] = [(m, mode) for m in fission.SAMPLING_MODULES
                               for mode in fission.SAMPLING_MODULES[m]['modes']]
        np.random.RandomState(config.seed).shuffle(fission._L1_GRID)
        fission._L1_IDX['i'] = 0
        self.rng = np.random.default_rng(config.seed)

    def sample(self, level):
        overrides, desc = self.module.sample_random_mutant(level, self.rng)
        return overrides, desc, None, None

    def candidate_name(self, description, level, number):
        return candidate_name('fission_random', description, level, number)

    def display_number(self, level_counts, number, level):
        return number

    def schedule_rng(self):
        return np.random

