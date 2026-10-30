"""Fission-specific input-pool and oracle services for the shared AL engine."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import json
import multiprocessing as mp
import os

import numpy as np
from budding_yeast_v2.data.dataset_archive import open_dataset
import torch

from budding_yeast_v2.data.normalization import normalize_p, normalize_y
from budding_yeast_v2.data.systems.sampling_policy import VERSION, valid_candidate
from budding_yeast_v2.data.systems.contracts import Candidate
from budding_yeast_v2.data.simulation_guards import bounded_ordered_map, simulation_deadline
from budding_yeast_v2.data.systems.fission.adapter import ADAPTER, SIMULATION_TIMEOUT
from budding_yeast_v2.data.systems.fission.al_pool import INITIAL_STATE_PROTOCOL, PARENT_STALL_TIMEOUT
from budding_yeast_v2.data.systems.fission.constants import MASS_UNIT
from budding_yeast_v2.data.systems.fission.parameters import WT_parameters, param_vector
from budding_yeast_v2.data.systems.fission.simulation import simulate_mutant
from budding_yeast_v2.data.systems.fission.specification import SPEC


def load_pool(engine, dm):
    """Reject trajectory archives and un-warmed or incorrectly ordered pools."""
    if not engine.al_pool_dataset or not os.path.isfile(engine.al_pool_dataset):
        raise ValueError('Fission AL requires --al_pool_dataset built by '
                         'python -m budding_yeast_v2.data.systems.fission.al_pool')
    with np.load(engine.al_pool_dataset, allow_pickle=False) as archive:
        required = ('system_id', 'artifact_kind', 'param_names', 'variable_names',
                    'params', 'y_init', 'mutant_names', 'overrides_json', 'time_grid',
                    'generation_report', 'is_real', 'model_version', 'mass_output_scale', 'levels')
        if any(key not in archive for key in required):
            raise ValueError('Fission AL input pool is missing required metadata')
        if str(archive['system_id']) != 'fission' or str(archive['artifact_kind']) != 'parameter_pool':
            raise ValueError('Fission AL requires a Fission parameter_pool, not a trajectory dataset')
        if (tuple(archive['param_names']) != SPEC.param_names
                or tuple(archive['variable_names']) != SPEC.variable_names
                or str(archive['model_version']) != SPEC.model_version
                or float(archive['mass_output_scale']) != SPEC.mass_scale):
            raise ValueError('Fission pool variable/parameter order, model version or mass scale mismatch')
        report = json.loads(str(archive['generation_report']))
        if report.get('initial_state_protocol') != INITIAL_STATE_PROTOCOL:
            raise ValueError('Fission candidate initial states must use per-mutant warmup')
        if str(archive.get('sampling_policy','')) != VERSION:
            raise ValueError('Regenerate Fission pool with the current L1/L2/L3 sampling policy')
        physical, initial = archive['params'].copy(), archive['y_init'].copy()
        names, time_grid = archive['mutant_names'].copy(), archive['time_grid'].copy()
        overrides = [json.loads(str(value)) for value in archive['overrides_json']]
        levels = archive['levels'].copy()
        count = len(names)
        expected_steps = dm.train_dataset_synth.tensors[2].shape[-1]
        if (not count or physical.shape != (count, len(SPEC.param_names))
                or initial.shape != (count, len(SPEC.variable_names))
                or len(overrides) != count or archive['is_real'].shape != (count,)
                or np.any(archive['is_real']) or len(set(map(str, names))) != count
                or not np.isfinite(physical).all() or not np.isfinite(initial).all()):
            raise ValueError('Malformed/nonfinite Fission candidate inputs')
        if (time_grid.shape != (expected_steps,)
                or not np.allclose(time_grid, np.linspace(0, dm.t_max, expected_steps))):
            raise ValueError('Fission candidate pool and training time grids differ')
        for i,name in enumerate(names):
            description = str(name).removeprefix('Rand_').rsplit('_L',1)[0]
            level = int(archive['levels'][i])
            if valid_candidate(Candidate(i,str(name),overrides[i],level,metadata={'description':description}),'fission') is None:
                raise ValueError('Fission pool violates exact L1/L2/L3 module counts')
        defaults = WT_parameters()
        for vector, override in zip(physical, overrides):
            if (not isinstance(override, dict) or any(key not in defaults for key in override)
                    or not np.allclose(vector, param_vector({**defaults, **override}), rtol=1e-12, atol=1e-12)):
                raise ValueError('Fission overrides disagree with exported parameter vectors')

    # Exclude all existing dataset rows (including validation/test) by physical
    # parameters, even if names differ, in addition to the canonical real set.
    references = ADAPTER.references()
    if getattr(dm, 'dataset_path', None):
        with open_dataset(dm.dataset_path, allow_pickle=True) as archive:
            for vector in archive['params']:
                references.add(vector)
    keep = []
    for vector in physical:
        keep.append(not references.collides(vector))
        if keep[-1]:
            references.add(vector)
    keep = np.asarray(keep, dtype=bool)
    if not keep.any():
        raise ValueError('No Fission candidates remain after dataset/real/pool deduplication')
    physical, initial, names = physical[keep], initial[keep], names[keep]
    overrides = [value for value, accepted in zip(overrides, keep) if accepted]
    p = torch.as_tensor(physical, dtype=torch.float32)
    y0 = torch.as_tensor(initial, dtype=torch.float32)
    engine._pool_size = len(names)
    engine._pool_mutant_names = names
    engine._pool_ovr = overrides
    engine._pool_P_real = p
    engine._pool_initial_physical = y0.clone()
    engine._pool_P = normalize_p(p, dm.p_mean.cpu(), dm.p_std.cpu(), dm.norm_type)
    engine._pool_Y_ic = normalize_y(y0, dm.mean_y.cpu(), dm.std_y.cpu(), dm.norm_type,
                                  getattr(dm, 's_c', None))
    engine._pool_X = engine._pool_Y = None
    engine._pool_mass_meta = [{'level':int(level),'genes':[], 'source_candidate_id':int(i)}
                             for level,i in zip(levels[keep], np.flatnonzero(keep))]
    engine._al4pde_t_grid = torch.linspace(0, 1, expected_steps).reshape(1, 1, -1)
    engine._al4pde_pool_loaded = True
    engine._fission_wt_birth = ADAPTER.prepare_context(None)
    engine._fission_initial_by_id = {i: value for i, value in enumerate(initial)}
    print(f'[Fission AL] Loaded {len(names)} warmed candidates; removed {count-len(names)} collisions')


def simulate_task(payload):
    """Full native warmup + 300-min trajectory + viability continuation, spawn-safe."""
    name, overrides, wt_birth, horizon, steps = payload
    try:
        with simulation_deadline(SIMULATION_TIMEOUT):
            result = simulate_mutant(name, overrides, wt_birth, tf=horizon, n_t=steps)
        if (result['y'] is None or result['status'] != 'End'
                or result.get('completed_time', 0) < horizon):
            return None
        y = result['y']
        initial = result['y_init'].copy()
        initial[0] /= MASS_UNIT
        if (y.shape != (len(SPEC.variable_names), steps) or not np.isfinite(y).all()
                or not np.isfinite(result['params_vec']).all() or not np.isfinite(initial).all()):
            return None
        # Preserve None for inviable strains as the original exporter string.
        return y, result['params_vec'], str(result['pattern']), result['label'], initial
    except Exception:
        return None


def query_pool(engine, indices, dm, target_valid_num=None):
    """Use native solver validity and mass ceiling; no Budding real-envelope filter."""
    indices = np.asarray(indices, dtype=np.int64)
    if target_valid_num is not None:
        raise ValueError('Fission generic strategies use single-batch queries')
    pool_ids = engine.pool.ensure_ids().copy()
    steps = engine._al4pde_t_grid.shape[-1]
    tasks = [(str(engine._pool_mutant_names[i]), dict(engine._pool_ovr[i]),
              engine._fission_wt_birth, float(dm.t_max), steps) for i in indices]
    workers = int(os.environ.get('AL_SIM_WORKERS', min(8, os.cpu_count() or 1)))
    if workers < 1:
        raise ValueError('AL_SIM_WORKERS must be positive')
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as executor:
        with closing(bounded_ordered_map(executor, simulate_task, tasks, stall_timeout=PARENT_STALL_TIMEOUT)) as rows:
            results = list(rows)
    accepted, valid = [], []
    for index, result in zip(indices, results):
        if result is None:
            continue
        candidate_id = int(pool_ids[index])
        # Acquisitions must see exactly the initial state later used by the oracle.
        if not np.allclose(result[4], engine._fission_initial_by_id[candidate_id], rtol=1e-8, atol=1e-9):
            raise ValueError('Fission warmup input differs between candidate and oracle')
        accepted.append(index)
        valid.append(result)
    x = engine._get_pool_X_batch(np.asarray(accepted, dtype=np.int64)).clone()
    names = [str(engine._pool_mutant_names[i]) for i in accepted]
    accepted_levels = [int(engine._pool_mass_meta[i]['level']) for i in accepted]
    from .io import pool_provenance
    provenance = pool_provenance(engine, accepted, [row[4] for row in valid])
    keep = np.ones(engine._pool_size, dtype=bool)
    keep[indices] = False
    engine.pool.keep(keep)
    for index in indices:
        engine._fission_initial_by_id.pop(int(pool_ids[index]), None)
    if not valid:
        print('[Fission AL] All selected simulations failed; consumed candidates removed')
        return None
    physical_y = np.stack([row[0] for row in valid])
    physical_p = np.stack([row[1] for row in valid])
    return dict(candidate_ids=pool_ids[accepted], consumed_ids=pool_ids[indices], X=x,
        Y=normalize_y(torch.as_tensor(physical_y, dtype=torch.float32), dm.mean_y.cpu(),
                      dm.std_y.cpu(), dm.norm_type, getattr(dm, 's_c', None)),
        P=normalize_p(torch.as_tensor(physical_p, dtype=torch.float32), dm.p_mean.cpu(),
                      dm.p_std.cpu(), dm.norm_type),
        Y_real=list(physical_y), P_real=list(physical_p), names=names, provenance=provenance,
        levels=accepted_levels, requested_levels=accepted_levels, module_counts=accepted_levels,
        labels=[row[2] for row in valid], biological_labels=[row[3] for row in valid],
        sample_plots={})
