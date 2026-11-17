"""A shared canonical real-mutant simulation reference, isolated from selectors."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from budding_yeast_v2.data.dataset_archive import atomic_json, fingerprint
from budding_yeast_v2.data.dataset_archive import file_sha256
from budding_yeast_v2.data.systems.budding import parameters, states, ode, real_mutants, simulation


def _reference_task(task):
    name, overrides = task
    from budding_yeast_v2.data.systems.budding.offline_pool import simulate
    from budding_yeast_v2.data.systems.contracts import Candidate
    return simulate(Candidate(0, name, overrides, -1))


def prepare_real_reference(output, workers=8):
    import fcntl
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.reference.lock').open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return _prepare_real_reference(root, workers)


def _prepare_real_reference(root, workers):
    if workers < 1:
        raise ValueError('Reference workers must be positive')
    source = {Path(module.__file__).name: file_sha256(module.__file__)
              for module in (parameters, states, ode, real_mutants, simulation)}
    from budding_yeast_v2.data.systems.budding import solver_policy
    source['solver_policy.py'] = file_sha256(solver_policy.__file__)
    identity = dict(canonical_sources=source, horizon=210, points=500, solver='BDF',
                    solver_policy=solver_policy.SOLVER_POLICY,
                    rtol=solver_policy.RTOL, atol=solver_policy.ATOL, rows=len(real_mutants.REAL_MUTANTS))
    path = root / f'real_reference_{fingerprint(identity)[:12]}.npz'
    report = path.with_suffix('.json')
    if path.exists() and report.exists():
        value = json.loads(report.read_text())
        if value['sha256'] != file_sha256(path):
            raise ValueError('Real reference changed')
        return path
    from budding_yeast_v2.data.simulation_guards import bounded_ordered_map
    tasks = real_mutants.real_tasks()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as executor, closing(
            bounded_ordered_map(executor, _reference_task, tasks, stall_timeout=35.)) as ordered:
        results = list(ordered)
    if any(result.trajectory is None for result in results):
        raise RuntimeError('At least one real reference failed; no partial benchmark is published')
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, data=np.asarray([r.trajectory for r in results], dtype=np.float32),
        params=np.asarray([r.params for r in results], dtype=np.float64),
        y_init=np.asarray([r.y_init for r in results], dtype=np.float64),
        mutant_names=np.asarray([r.candidate.name for r in results]),
        is_real=np.ones(len(results), dtype=bool))
    temporary.replace(path)
    atomic_json(report, dict(**identity, sha256=file_sha256(path),
                            role='evaluation_only_literature_defined_mutant_simulations'))
    return path
