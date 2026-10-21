"""Fission AL input pools: canonical mutant warmup, without full labelled trajectories.

Run with ``python -m budding_yeast_v2.data.systems.fission.al_pool --help``.
Warmup is real ODE work; a pool is not a zero-simulation parameter draw.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np

from .adapter import ADAPTER, SIMULATION_TIMEOUT
from .constants import MASS_UNIT
from .parameters import WT_parameters, param_vector
from .states import mutant_birth_state
from ..contracts import GenerationConfig, SimulationResult
from ..sampling_policy import VERSION, POOL_RATIOS
from ..dataset_io import write_dataset
from ..generation import prepare_candidates
from ..registry import get_recipe, make_sampler, resolve_config
from ...simulation_guards import bounded_ordered_map, simulation_deadline

INITIAL_STATE_PROTOCOL = 'fission_mutant_warmup_1000_v1'
# Spawn imports can precede the 30 s native solver deadline.
PARENT_STALL_TIMEOUT = SIMULATION_TIMEOUT + 30.0


def warmup_candidate(payload):
    """Spawn-safe worker; all concentrations exported in native dataset units."""
    candidate, wt_birth = payload
    try:
        with simulation_deadline(SIMULATION_TIMEOUT):
            initial = mutant_birth_state(candidate.overrides, wt_birth)
        initial = np.asarray(initial, dtype=float).copy()
        initial[0] /= MASS_UNIT
        if not np.isfinite(initial).all():
            return None
        return SimulationResult(candidate, None,
            param_vector({**WT_parameters(), **candidate.overrides}), initial,
            'Unlabelled', None, 'parameter_only')
    except Exception:
        return None


def generate_pool(output, samples=10000, seed=20000, workers=1, mode_freq='class'):
    """Preserve original sampler, real/pool collision rescue and warmup fallback."""
    if samples < 1 or seed < 0 or workers < 1 or mode_freq not in ('class', 'flat'):
        raise ValueError('Invalid Fission pool count, seed, workers or mode frequency')
    if Path(output).suffix != '.npz':
        raise ValueError('Fission pool output must end in .npz')
    if Path(output).exists():
        raise FileExistsError(output)
    config = resolve_config(GenerationConfig('fission_random', n_samples=samples,
        seed=seed, workers=workers, output=str(output), include_real=False,
        parameter_only=True, mode_freq=mode_freq))
    recipe = get_recipe(config.recipe)
    report = Counter(collisions=0, rescued=0)
    dict.update(report, mutation_profile='fission', distribution_profile='fission',
        rng_contract='legacy_algorithms_scoped_v1', sampling_policy=VERSION, initial_state_protocol=INITIAL_STATE_PROTOCOL,
        full_trajectory_simulations=0, requested_synthetic=samples)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        sampler = make_sampler(recipe, config)
        candidates = prepare_candidates(recipe, config, sampler, report)
        wt_birth = ADAPTER.prepare_context(config)
        tasks = [(candidate, wt_birth) for candidate in candidates]
        if workers == 1:
            results = [warmup_candidate(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as executor:
                with closing(bounded_ordered_map(executor, warmup_candidate, tasks, stall_timeout=PARENT_STALL_TIMEOUT)) as rows:
                    results = list(rows)
        if any(row is None for row in results):
            raise RuntimeError('Fission candidate warmup failed/timed out; refusing a partial input pool')
        report['warmed_candidates'] = len(results)
        return write_dataset(output, ADAPTER.spec, results, config, dict(report), parameter_only=True)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--mode-freq', choices=('class', 'flat'), default='class')
    args = parser.parse_args(argv)
    path = generate_pool(args.output, args.samples, args.seed, args.workers, args.mode_freq)
    print(f'Saved Fission input pool: {path}; mutant warmup only, no trajectory labels')


if __name__ == '__main__':
    main()
