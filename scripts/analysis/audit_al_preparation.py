"""Validate a prepared AL pool and its fixed initial split without training."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from budding_yeast_v2.data.systems.al_initial import validate_existing_initial, random_split
from budding_yeast_v2.data.systems.registry import get_system_spec
from budding_yeast_v2.data.systems.sampling_policy import VERSION, POOL_RATIOS


def audit(pool_path, initial_path, expected_pool, train_samples, seed):
    validate_existing_initial(initial_path, pool_path, train_samples, seed)
    with np.load(pool_path, allow_pickle=False) as pool, np.load(initial_path, allow_pickle=False) as initial:
        system = str(pool['system_id'])
        spec = get_system_spec(system)
        expected_real = 126 if system == 'budding' else 20
        assert str(initial['system_id']) == system
        assert str(pool['artifact_kind']) == 'parameter_pool' and 'data' not in pool
        assert str(initial['artifact_kind']) == 'trajectory_dataset'
        for archive in (pool, initial):
            assert str(archive['sampling_policy']) == VERSION
            assert tuple(archive['param_names']) == spec.param_names
            assert tuple(archive['variable_names']) == spec.variable_names
            assert np.allclose(archive['time_grid'], np.linspace(0, spec.horizon, spec.steps))
            assert np.isfinite(archive['params']).all() and np.isfinite(archive['y_init']).all()
            synthetic = ~archive['is_real']
            assert np.isin(archive['module_counts'][synthetic], [1,2,3]).all()
            assert np.array_equal(archive['module_counts'][synthetic], archive['requested_levels'][synthetic])
        assert len(pool['params']) == expected_pool and not pool['is_real'].any()
        assert len(np.unique(pool['mutant_names'])) == expected_pool
        counts = Counter(pool['module_counts'].tolist())
        if system == 'budding':
            from budding_yeast_v2.data.systems.budding.al_pool import validate_pool_contract
            validate_pool_contract(pool)
            allocation = json.loads(str(pool['generation_report']))['module_quotas']
            planned = Counter()
            for row in allocation:
                assert row['accepted'] == row['quota']
                planned[row['level']] += row['quota']
            assert counts == planned  # Pool quotas differ from initial-set 5/40/55.
        else:
            assert [counts[k] for k in (1,2,3)] == [round(expected_pool*r) for r in POOL_RATIOS]
        assert np.allclose(pool['p_mean'], pool['params'].mean(axis=0))
        std = pool['params'].std(axis=0)
        assert np.allclose(pool['p_std'], np.where(std<1e-8,1.,std))
        if system == 'budding':
            assert str(pool['mutation_profile']) == str(pool['distribution_profile']) == 'mutant_aware'
        else:
            from budding_yeast_v2.data.systems.fission.al_pool import INITIAL_STATE_PROTOCOL
            assert json.loads(str(pool['generation_report']))['initial_state_protocol'] == INITIAL_STATE_PROTOCOL
        real = initial['is_real']
        assert real.sum() == expected_real
        splits = [initial[f'split_{key}'] for key in ('train','val','test','real')]
        assert list(map(len,splits)) == [train_samples,train_samples//8,train_samples//8,expected_real]
        assert np.array_equal(np.sort(np.concatenate(splits)), np.arange(len(real)))
        assert np.array_equal(splits[3], np.flatnonzero(real))
        for actual, expected in zip(splits[:3],random_split(np.flatnonzero(~real),seed)):
            assert np.array_equal(actual,expected)
        report = json.loads(str(initial['generation_report']))
        ids = np.asarray(report['selected_candidate_ids'],dtype=np.int64)
        assert len(np.unique(ids)) == train_samples*5//4
        assert np.array_equal(initial['mutant_names'][~real],pool['mutant_names'][ids])
        # NPZ trajectories are float32; exported model parameters retain float64.
        assert np.array_equal(initial['params'][~real],pool['params'][ids])
        assert np.array_equal(initial['y_init'][~real],pool['y_init'][ids])
        assert np.array_equal(initial['candidate_ids'][~real],ids)
        initial_counts = Counter(initial['module_counts'][~real].tolist())
        assert [initial_counts[k] for k in (1,2,3)] == [round(len(ids)*r) for r in POOL_RATIOS]
        for field in ('labels','pattern_labels','mutation_descriptions','mutation_pairs_json',
                      'overrides_json','candidate_metadata_json','y_init'):
            assert len(initial[field]) == len(real), field
        if system == 'budding':
            from budding_yeast_v2.data.systems.budding.pattern_groups import (
                pattern_group_metadata, validate_stored_pattern_groups)
            validate_stored_pattern_groups(initial,pattern_group_metadata(initial['data']))
        data = initial['data']
        assert data.shape == (len(real),len(spec.variable_names),spec.steps)
        assert np.isfinite(data).all()
        assert (initial['simulation_status']=='success').all()
        return dict(status='passed',system=system,pool=str(Path(pool_path).resolve()),
            initial=str(Path(initial_path).resolve()),pool_samples=expected_pool,
            pool_L_counts=dict(counts),initial_synthetic=int((~real).sum()),real_mutants=int(real.sum()),
            splits=dict(zip(('train','val','test','real'),map(len,splits))),
            initial_L_counts=dict(Counter(initial['module_counts'][~real].tolist())),
            simulation_failures=report.get('simulation_failures',0),
            finite_negative_values_preserved=int((data<0).sum()),
            mutation_profile=str(pool['mutation_profile']),distribution_profile=str(pool['distribution_profile']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',required=True)
    parser.add_argument('--initial',required=True)
    parser.add_argument('--pool-samples',required=True,type=int)
    parser.add_argument('--train-samples',required=True,type=int)
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:
        result=audit(args.pool,args.initial,args.pool_samples,args.train_samples,args.seed)
    except Exception as exc:
        result=dict(status='failed',error=repr(exc))
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        raise
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
