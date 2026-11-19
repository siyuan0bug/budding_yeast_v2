"""Bounded, ordered CPU execution of the unchanged per-candidate estimator."""
import atexit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

import numpy as np
import torch
from budding_yeast_v2.utils.sampling.ours.mechanistic.systems import BiologicalODE
from budding_yeast_v2.utils.sampling.ours.mechanistic.transport import estimate

_SYSTEM = None
_POOLS = {}


def _initialize(system_id):
    global _SYSTEM
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _SYSTEM = BiologicalODE(system_id)


def score_row(system, row):
    predicted, p, y0, meta, reference, refp, refy0, refmeta, exact, scale, times, cfg = row
    counts = dict(certified_equivalence_rows=0, linearized_response_rows=0,
                  transported_defect_rows=0, grazing_updates=0, unresolved_event_intervals=0)
    if exact:
        scaled = (predicted - reference) / scale[:, None]
        score = float(np.sqrt(np.mean(scaled ** 2)))
        counts['certified_equivalence_rows'] = 1
    elif reference is not None:
        _, response, grid, details = estimate(system, reference, p, y0, scale,
            metadata=meta, time_grid=times, config=cfg, reference_params=refp,
            reference_initial=refy0, reference_metadata=refmeta)
        scaled = (predicted[:, grid] - reference[:, grid] - response) / scale[:, None]
        score = float(np.sqrt(np.mean(scaled ** 2)))
        counts['linearized_response_rows'] = 1
        for key in ('grazing_updates', 'unresolved_event_intervals'): counts[key] = details[key]
    else:
        score, error, _, details = estimate(system, predicted, p, y0, scale,
            metadata=meta, time_grid=times, config=cfg)
        scaled = error / scale[:, None]
        counts['transported_defect_rows'] = 1
        for key in ('grazing_updates', 'unresolved_event_intervals'): counts[key] = details[key]
    if not np.isfinite(score) or not np.isfinite(scaled).all():
        raise FloatingPointError('Nonfinite mechanism acquisition in CPU worker')
    columns = np.linspace(0, scaled.shape[-1] - 1, 8).round().astype(int)
    return score, scaled[:, columns].ravel(), counts


def _work(row):
    return score_row(_SYSTEM, row)


def score_rows(system, rows, workers):
    if workers == 1:
        return [score_row(system, row) for row in rows]
    if not 1 <= workers <= 16:
        raise ValueError('Mechanism CPU workers must be between 1 and 16')
    key = system.name, workers
    if key not in _POOLS:
        _POOLS[key] = ProcessPoolExecutor(max_workers=workers,
            mp_context=mp.get_context('spawn'), initializer=_initialize, initargs=(system.name,))
    # One predictor batch is the entire in-flight bound. Ordered map preserves
    # the caller's candidate order, including deterministic score tie handling.
    return list(_POOLS[key].map(_work, rows, chunksize=4))


def shutdown():
    for pool in _POOLS.values(): pool.shutdown(wait=True, cancel_futures=True)
    _POOLS.clear()

atexit.register(shutdown)
