"""Budding adapter: MA biology, collision rescue, BAMS refinement and result filters."""
from collections import Counter
import importlib
from scipy.stats import qmc
from dataclasses import replace

import numpy as np

from .specification import BuddingSystem, SPEC
from . import real_mutants
from .samplers.rescue import parse_pairs, rescue_budding
from ..contracts import Candidate, SimulationResult
from ..registry import resolve_config, candidate_name


class BuddingAdapter:
    system_id = 'budding'
    spec = SPEC

    def prepare_context(self, config):
        return None

    def real_candidates(self, include_real):
        return ([Candidate(-i - 1, n, o, 0, True)
                 for i, (n, o) in enumerate(real_mutants.real_tasks())] if include_real else [])

    def references(self):
        return None

    def _backend(self, config, recipe):
        return BuddingSystem(config.mutation_profile or recipe.mutation,
                             config.distribution_profile or recipe.distribution)

    def is_collision(self, overrides, references, sampler):
        return real_mutants.is_duplicate_real_mutant(overrides, sampler.backend.defaults())

    def add_reference(self, references, overrides):
        return None

    def rescue(self, overrides, description, sampler, rng):
        return rescue_budding(overrides, description, sampler.backend, rng)

    def rescue_rng(self, sampler):
        # Pure Random threads its explicit RNG; legacy MASS/k-DPP/BAMS callers use numpy.
        if sampler.recipe.sampler in ('random', 'random_ko_interval'):
            return sampler.rng
        return np.random

    def pairs_from_description(self, description):
        return parse_pairs(description)

    def parameter_result(self, candidate, context):
        backend = BuddingSystem()
        defaults = backend.defaults()
        _, y0, _, _ = backend.prepare_initial_conditions(candidate.overrides)
        params = np.array([candidate.overrides.get(k, defaults[k]) for k in backend.PARAM_KEYS], dtype=float)
        return SimulationResult(candidate, None, params, y0, 'Unlabelled', None, 'parameter_only')

    def simulate(self, candidate, config, context):
        from ..generation import ensure_candidate
        from ..registry import get_recipe
        from .samplers.bams import execute_simulation_task

        recipe = get_recipe(config.recipe)
        backend = self._backend(config, recipe)
        if candidate.refinement is not None:
            # Stable per-candidate worker randomness, independent of completion order.
            state = np.random.get_state()
            np.random.seed(np.random.SeedSequence([config.seed, candidate.sample_id]).generate_state(1)[0])
            try:
                result, overrides = execute_simulation_task(
                    (candidate.name, dict(candidate.overrides), candidate.refinement, 210, 500), ma=backend)
            finally:
                np.random.set_state(state)
            candidate = replace(candidate, overrides=overrides)
            if not candidate.is_real and real_mutants.is_duplicate_real_mutant(overrides, backend.defaults()):
                from ..registry import make_sampler
                local = make_sampler(recipe, config)
                local.rng = np.random.RandomState(config.seed + candidate.sample_id)
                candidate = ensure_candidate(candidate, local, Counter(), adapter=self, rescue_rng=local.rng)
                result = backend.simulate_mutant(candidate.name, candidate.overrides)
        else:
            result = backend.simulate_mutant(candidate.name, candidate.overrides)
        row = self.parameter_result(candidate, context)
        _, trajectory, label, pattern = result
        row.trajectory = trajectory
        row.label = label
        row.pattern = pattern
        row.status = ('success' if trajectory is not None
                      else ('discarded_mass' if label == 'Discarded_Mass' else 'numerical_failure'))
        return row

    def accept_result(self, result, state, config):
        recipe = state['recipe']
        if result.candidate.is_real:
            return True
        if recipe.sampler == 'wt_blind' and np.max(np.abs(result.trajectory)) > 1000.:
            state['report']['trajectory_limit'] += 1
            return False
        if recipe.sampler == 'kdpp_curvature':
            from .samplers.kdpp import compute_trajectory_curvature
            level = result.candidate.level
            accepted = sum(state['level_counts'].values())
            if accepted >= config.n_samples:
                return False
            if state['level_counts'][level] >= state['target'][level]:
                return False
            if (compute_trajectory_curvature(result.trajectory) < config.curv_thresh
                    and state['sampler'].rng.rand() > config.flat_prob):
                state['report']['flatline_filtered'] += 1
                return False
        return True

    def finalize(self, results, config, recipe):
        if recipe.sampler == 'wt_blind':
            return ([r for r in results if r.candidate.is_real]
                    + sorted((r for r in results if not r.candidate.is_real), key=lambda r: r.candidate.name))
        return results

    def expected_real_count(self):
        return 126

    def special_candidates(self, recipe, config, sampler, report):
        if recipe.sampler == 'wt_blind':
            from .samplers.wt_pipeline import prepare_wt_candidates
            return prepare_wt_candidates(sampler, config, report)
        return None

    def validate_config(self, config, recipe):
        if not np.isfinite(config.curv_thresh) or config.curv_thresh < 0:
            raise ValueError('Curvature threshold must be finite and nonnegative')
        if not 0 <= config.ko_interval_max <= 1e-4:
            raise ValueError('KO interval must be in [0, 1e-4]')
        if recipe.sampler == 'wt_blind' and config.mode_freq not in ('flat', 's5'):
            raise ValueError('WT-Blind supports flat/s5 mode frequencies')
        if config.pilot_samples < 1 or config.wt_fold_dist not in ('flat', 'tier2'):
            raise ValueError('Invalid WT-Blind pilot or fold policy')
        if not 0 <= config.nan_cap_frac <= 1 or not 0 <= config.backlog_thresh <= 1 or config.max_attempts_mult < 1:
            raise ValueError('Invalid WT-Blind ledger configuration')

    def handles_derived(self, recipe):
        return recipe.sampler in ('twins', 'ko_mirror', 'bams_ablation')

    def generate_derived(self, config):
        from .derived import generate_derived
        return generate_derived(config)


ADAPTER = BuddingAdapter()


class BuddingSampler:
    """Adapter for the algorithms under systems/budding/samplers."""

    def __init__(self, recipe, config):
        from .specification import BuddingSystem
        from .samplers import common

        config = resolve_config(config)
        self.recipe, self.config = recipe, config
        self.rng = np.random.RandomState(config.seed)
        self.kernel = None
        self.sobol = None
        common.MODE_FREQ = config.mode_freq
        common._L1_IDX['i'] = 0
        self.backend = BuddingSystem(config.mutation_profile or recipe.mutation,
                                     config.distribution_profile or recipe.distribution)
        kind = recipe.sampler
        module_name = {'random_ko_interval': 'random', 'mass_random': 'random',
                       'mass_random_dist': 'random', 'mass_motifs': 'mass_motifs',
                       'gibbs_sobol': 'sobol', 'pool': 'sobol',
                       'kdpp_curvature': 'kdpp'}.get(kind, kind)
        self.module = importlib.import_module(f'{__package__}.samplers.{module_name}')
        if kind in ('gibbs', 'gibbs_sobol'):
            self.kernel = self.module.GibbsBoltzmannGraphSampler(temperature=config.temperature, seed=config.seed, ma=self.backend)
        elif kind == 'kdpp':
            self.kernel = self.module.GraphDiffusionKDPP(seed=config.seed, ma=self.backend)
        elif kind == 'kdpp_curvature':
            self.kernel = self.module.SignedDirectedGraphKDPP(seed=config.seed, ma=self.backend)
        elif kind == 'motif':
            self.kernel = self.module.MotifTopologySampler(seed=config.seed, ma=self.backend)
        elif kind == 'mass':
            self.module.MODE_FREQ = config.mode_freq
            self.module._L2_PICKER = common.QuotaPicker(
                list(self.module.STRUCT_L2), list(self.module.STRUCT_L2.values()))
            self.module._L34_PICKER = common.QuotaPicker(
                list(self.module.STRUCT_L34), list(self.module.STRUCT_L34.values()))
        elif kind == 'bams':
            self.module._L1_IDX['i'] = 0
        if kind in ('gibbs_sobol', 'pool'):
            self.sobol = qmc.Sobol(d=16 if kind == 'gibbs_sobol' else 8, scramble=True, seed=config.seed)
            self._sobol_points = None
            self._sobol_index = 0

    def _coordinates(self):
        if self._sobol_points is None or self._sobol_index == len(self._sobol_points):
            self._sobol_points = self.sobol.random(65536)
            self._sobol_index = 0
        row = self._sobol_points[self._sobol_index]
        self._sobol_index += 1
        return row

    def sample(self, level):
        if level not in (1,2,3):raise ValueError('Only L1/L2/L3 are supported')
        m, kind, rng, backend = self.module, self.recipe.sampler, self.rng, self.backend
        refinement = None
        if kind in ('random', 'random_ko_interval'):
            overrides, desc = m.sample_pure_random_mutant(
                level, rng, enable_ko_interval=kind == 'random_ko_interval',
                ko_interval_max=self.config.ko_interval_max, ma=backend)
        elif kind == 'mass_random':
            overrides, desc = m.sample_mass_random(level, ma=backend)
        elif kind == 'mass_random_dist':
            overrides, desc = m.sample_mass_random_dist(level, ma=backend)
        elif kind in ('mass', 'mass_motifs'):
            overrides, desc = m.sample_mass_mutant(level, ma=backend)
        elif kind == 'gibbs':
            overrides, desc = m.sample_gibbs_mutant(level, self.kernel, rng, ma=backend)
        elif kind == 'gibbs_sobol':
            overrides, desc = m.sample_gibbs_sobol_mutant(level, self.kernel, self._coordinates(), rng, ma=backend)
        elif kind == 'pool':
            overrides, desc, _ = m.sample_pure_random_candidate(level, self._coordinates(), rng, ma=backend)
        elif kind == 'kdpp':
            overrides, desc = m.sample_kdpp_mutant(level, self.kernel, rng, ma=backend)
        elif kind == 'kdpp_curvature':
            overrides, desc = m.sample_signed_kdpp_mutant(level, self.kernel, rng, ma=backend)
        elif kind == 'motif':
            genes, modes = self.kernel.sample_motif(level)
            overrides = m.build_perturbed_sample(genes, modes, rng, ma=backend)
            desc = '+'.join(f'{g}_{mode}' for g, mode in zip(genes, modes))
            return overrides, desc, None, [(g, mode) for g, mode in zip(genes, modes)]
        elif kind == 'bams':
            overrides, desc, refinement = m.sample_bams_mutant_task(level, ma=backend)
            if not self.config.enable_bisection:
                refinement = None
        elif kind == 'wt_blind':
            cand = m.build_candidate(level, m.make_samplers(), ma=backend)
            if cand is None:return {}, '', None, None
            return cand['ovr'], cand['desc'], None, cand['genes_modes']
        else:
            raise ValueError(f'Unsupported Budding sampler: {kind}')
        return overrides, desc, refinement, None

    def candidate_name(self, description, level, number):
        return candidate_name(self.recipe.sampler, description, level, number)

    def display_number(self, level_counts, number, level):
        if self.recipe.sampler == 'motif':
            return level_counts[level]
        if self.recipe.sampler == 'mass_random_dist':
            return level_counts[level] + 1
        return number

    def schedule_rng(self):
        if self.recipe.sampler in ('mass', 'mass_motifs', 'mass_random', 'bams'):
            return np.random
        return self.rng

