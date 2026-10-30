"""Validated NPZ serialization. No fake labels for failed integrations."""
import json
import os
from dataclasses import asdict
from pathlib import Path
import tempfile

import numpy as np


def write_dataset(path, spec, results, config, report, *, parameter_only=False, dtype='float64'):
    from .registry import resolve_config
    config=resolve_config(config)
    path=Path(path).expanduser().resolve()
    if path.suffix != '.npz':
        raise ValueError('Dataset output must end in .npz')
    if path.exists() and not config.overwrite:
        raise FileExistsError(f'Refusing to overwrite {path}; choose a new output or explicitly use --overwrite')
    if not results:
        raise ValueError('No valid samples; refusing an empty/misleading dataset')
    if not parameter_only and any(r.status!='success' or r.trajectory is None for r in results):
        raise ValueError('Only successful simulation status may enter a trajectory dataset')
    from .sampling_policy import VERSION, valid_candidate
    if report.get('sampling_policy') == VERSION:
        for row in results:
            checked = valid_candidate(row.candidate, spec.system_id)
            if checked is None:
                raise ValueError('Synthetic result violates exact L1/L2/L3 policy')
            row.candidate = checked
    if report.get('sampling_policy') == VERSION:
        from .sampling_policy import ENVIRONMENT_MODULES
        if spec.system_id == 'budding':
            from .budding.catalog import GROUP_BY_NAME
            catalog = GROUP_BY_NAME
        else:
            from .fission.catalog import SAMPLING_MODULES
            catalog = SAMPLING_MODULES
        expected = {(gene,mode) for gene,group in catalog.items() if gene not in ENVIRONMENT_MODULES for mode in group['modes']}
        covered = {(gene,mode) for row in results if not row.candidate.is_real and row.candidate.level==1
                   for gene,mode in row.candidate.metadata['pairs'] if (gene,mode) in expected}
        from collections import Counter
        histogram = Counter(int(row.candidate.metadata['module_count']) for row in results if not row.candidate.is_real)
        report = dict(report,saved_module_count_histogram=dict(histogram),requested_level_ratios=list(config.level_ratios),l1_mode_cells_total=len(expected),l1_mode_cells_covered=len(covered),
                      l1_missing_mode_cells=sorted(f'{gene}[{mode}]' for gene,mode in expected-covered))
    n=len(results)
    if len({r.candidate.sample_id for r in results})!=n:
        raise ValueError('Duplicate sample_id in dataset')
    params=np.asarray([r.params for r in results],dtype=np.float32 if config.recipe=='budding_mass_random_dist' else np.float64)
    y_init=np.asarray([r.y_init for r in results],dtype=float)
    if params.shape!=(n,len(spec.param_names)) or y_init.shape!=(n,len(spec.variable_names)):
        raise ValueError('Parameter/initial-state dimensions do not agree with the system')
    if not np.isfinite(params).all() or not np.isfinite(y_init).all():
        raise ValueError('Nonfinite exported input')
    archive=dict(params=params,param_names=np.array(spec.param_names),
                 mutant_names=np.array([r.candidate.name for r in results]),
                 sample_id=np.array([r.candidate.sample_id for r in results],dtype=np.int64),
                 pair_id=np.array([r.candidate.pair_id for r in results],dtype=np.int64),
                 is_real=np.array([r.candidate.is_real for r in results],dtype=bool),
                 levels=np.array([r.candidate.level for r in results],dtype=np.int16),
                 overrides_json=np.array([json.dumps(r.candidate.overrides,sort_keys=True,default=lambda v:v.item()) for r in results]),
                 candidate_metadata_json=np.array([json.dumps(r.candidate.metadata,sort_keys=True,default=lambda v:v.item()) for r in results]),
                 simulation_diagnostics_json=np.array([json.dumps(r.diagnostics,sort_keys=True,default=lambda v:v.item()) for r in results]),
                 y_init=y_init,variable_names=np.array(spec.variable_names),var_names=np.array(spec.variable_names),
                 time_grid=np.linspace(0,spec.horizon,spec.steps),system_id=np.array(spec.system_id),
                 model_version=np.array(spec.model_version),benchmark_profile=np.array(spec.benchmark_profile),
                 schema_version=np.int64(2),artifact_kind=np.array('parameter_pool' if parameter_only else 'trajectory_dataset'),
                 mass_output_scale=np.float64(spec.mass_scale),
                 mutation_profile=np.array(config.mutation_profile or report['mutation_profile']),
                 distribution_profile=np.array(config.distribution_profile or report['distribution_profile']),
                 sampling_seed=np.int64(config.seed),sampling_rng_contract=np.array(report['rng_contract']),
                 generation_config=np.array(json.dumps(asdict(config),ensure_ascii=False)),
                 generation_report=np.array(json.dumps(report,ensure_ascii=False)))
    from ..sample_metadata import mutation_metadata
    archive.update(mutation_metadata(archive['mutant_names'], archive['candidate_metadata_json'],
                                    archive['is_real'], spec.system_id))
    archive.update(simulation_status=np.asarray([r.status for r in results]),
        initial_state_source=np.full(n, 'system_prepared'),
        overrides_available=np.ones(n, dtype=bool),
        candidate_ids=np.asarray([r.candidate.metadata.get('source_candidate_id',
            r.candidate.sample_id-1 if parameter_only and not r.candidate.is_real else -1)
            for r in results], dtype=np.int64))
    if spec.system_id == 'budding':
        from .budding.catalog import CATALOG_VERSION
        from .budding.distributions.ranges import RANGE_FINGERPRINT
        archive['catalog_version'] = np.array(CATALOG_VERSION)
        archive['sampling_range_fingerprint'] = np.array(RANGE_FINGERPRINT)
    if 'pool_generation_contract' in report:
        archive['pool_generation_contract'] = np.array(report['pool_generation_contract'])
    if report.get('sampling_policy') == VERSION:
        archive.update(sampling_policy=np.array(VERSION),
            requested_levels=np.array([r.candidate.level for r in results],dtype=np.int16),
            module_counts=np.array([-1 if r.candidate.is_real else r.candidate.metadata['module_count'] for r in results],dtype=np.int16))
    if parameter_only:
        archive['p_mean']=params.mean(axis=0)
        std=params.std(axis=0)
        archive['p_std']=np.where(std<1e-8,1.,std)
    else:
        data=np.asarray([r.trajectory for r in results],dtype=dtype)
        if data.shape!=(n,len(spec.variable_names),spec.steps) or not np.isfinite(data).all():
            raise ValueError('Malformed/nonfinite trajectory dataset')
        archive.update(data=data,labels=np.array([r.label for r in results]),
                       pattern_labels=np.array([str(r.pattern) for r in results]),
                       simulation_status=np.array([r.status for r in results]))
        from ..label_metadata import trajectory_label_metadata
        archive.update(trajectory_label_metadata(data,archive['labels'],spec.system_id))
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.npz',delete=False) as handle:
            temporary=Path(handle.name)
            np.savez_compressed(handle,**archive)
        if config.overwrite:
            os.replace(temporary,path)
        else:
            # Atomic no-clobber publication, including concurrent writers.
            os.link(temporary,path)
            temporary.unlink()
        return str(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_dataset(path, *, require_trajectories=True):
    with np.load(path,allow_pickle=True) as archive:
        if require_trajectories and ('data' not in archive or str(archive.get('artifact_kind',''))=='parameter_pool'):
            raise ValueError('A parameter pool has no trajectory labels')
        return {key:archive[key] for key in archive.files}
