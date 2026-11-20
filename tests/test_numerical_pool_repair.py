"""Accuracy retries and immutable corrected views preserve scientific identities."""
import json
from unittest.mock import patch

import numpy as np
import pytest

from budding_yeast_v2.data.dataset_archive import TrajectoryStore, open_dataset, file_sha256
from budding_yeast_v2.data.dataset_repairs import write_row_patch
from budding_yeast_v2.data.systems.budding import simulation as sim
from budding_yeast_v2.data.systems.budding.solver_policy import RTOL, ATOL, RETRY_RTOL
from test_sampling_benchmark import make_pool


def test_same_input_retry_does_not_clip_truth_or_change_api():
    _, initial, _, _ = sim.prepare_initial_conditions({})
    bad = np.repeat(initial[:,None], 5, axis=1)
    bad[28,2] = 100
    good = bad.copy(); good[28,2] = 1
    info = {}
    with patch.object(sim, '_integrate_mutant', side_effect=[('x',bad,'bio','abc'),('x',good,'bio','abc')]) as call:
        result = sim.simulate_mutant('x', {}, ns=5, diagnostics=info)
    assert result[1] is good and len(result) == 4
    assert call.call_args_list[0].kwargs == dict(rtol=RTOL,atol=ATOL)
    assert call.call_args_list[1].kwargs['rtol'] == RETRY_RTOL
    assert call.call_args_list[0].args == call.call_args_list[1].args
    assert len(info['attempts']) == 2 and info['status'] == 'success'


def test_unresolved_total_violation_is_failure_but_negative_values_are_retained():
    _, initial, _, _ = sim.prepare_initial_conditions({})
    bad = np.repeat(initial[:,None], 5, axis=1); bad[28,2] = 100
    with patch.object(sim, '_integrate_mutant', return_value=('x',bad,'bio','abc')):
        assert sim.simulate_mutant('x', {})[1] is None
    bad[28,2] = 1; bad[1,2] = -1e-3
    with patch.object(sim, '_integrate_mutant', return_value=('x',bad,'bio','abc')):
        assert sim.simulate_mutant('x', {})[1] is bad


def test_overlay_exact_arbitrary_reads_and_array_hashes_without_copying_other_rows(tmp_path):
    path, arrays = make_pool(tmp_path/'original')
    digest = file_sha256(path)
    ids = np.array([27,0,9,8])  # unordered, boundaries and last partial block
    changed = arrays['data'][ids].copy(); changed[:,2,1:] *= 2
    labels = np.asarray(['A much longer biological label']*len(ids))
    target = write_row_patch(path,tmp_path/'corrected'/'dataset.json',ids,
                             dict(data=changed,labels=labels),provenance=dict(reason='test'))
    expected = arrays['data'].copy(); expected[ids] = changed
    store = TrajectoryStore(target)
    np.testing.assert_array_equal(store.read([27,2,0,27,8,9])['data'],expected[[27,2,0,27,8,9]])
    with open_dataset(target) as a:
        np.testing.assert_array_equal(a['data'],expected)  # validates full-field SHA too
        np.testing.assert_array_equal(a['params'],arrays['params'])
        np.testing.assert_array_equal(a['labels'][ids],labels)
    assert file_sha256(path) == digest
    assert len(list(target.parent.glob('*.npz'))) == 1
    with np.load(target.with_suffix('.patch.npz')) as patch_data:
        assert patch_data['data'].shape[0] == len(ids)
    with pytest.raises(FileExistsError):
        write_row_patch(path,target,ids,dict(data=changed),provenance={})


def test_production_tight_solver_records_actual_precision():
    info = {}
    name, y, bio, pattern = sim.simulate_mutant('WT',{},diagnostics=info)
    assert y.shape == (39,500) and np.isfinite(y).all()
    assert info['rtol'] == 1e-8 and info['atol'] == 1e-10
    assert not info['tem1_total_violation']


def test_failed_simulation_is_recorded_and_replenished_to_requested_count(tmp_path, monkeypatch):
    from dataclasses import replace
    from budding_yeast_v2.data.systems.budding import offline_pool
    calls = []
    def force_one_failure(executor, function, tasks, **kwargs):
        for task in tasks:
            result = function(task)
            if not calls:
                result = replace(result, trajectory=None, status='numerical_failure_or_timeout')
            calls.append(task.sample_id)
            yield result
    monkeypatch.setattr(offline_pool,'bounded_ordered_map',force_one_failure)
    path = offline_pool.generate('mutation',tmp_path/'pool',count=3,workers=1,block_size=2,seed=22)
    store = TrajectoryStore(path)
    assert store.count == 3 and len(calls) == 4
    report = json.loads(store.scalar('generation_report').item())
    assert report['counts']['simulation_failures'] == 1
    with open_dataset(path) as archive:
        np.testing.assert_array_equal(archive['sample_id'],np.arange(1,4))
