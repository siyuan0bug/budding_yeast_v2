"""Neutral groups supplement biology and preserve existing ABC training contracts."""
import csv
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from budding_yeast_v2.data.dataset_utils import load_yeast_dataset_universal
from budding_yeast_v2.data.systems.contracts import Candidate, GenerationConfig, SimulationResult
from budding_yeast_v2.data.systems.dataset_io import write_dataset
from budding_yeast_v2.data.label_metadata import UNKNOWN_BIOLOGY_LABEL, trajectory_label_metadata
from budding_yeast_v2.data.systems.budding.pattern_groups import (
    PATTERN_GROUP_CODES, PATTERN_GROUP_FIELDS, PATTERN_GROUP_TO_ABC,
    abc_pattern_ids, pattern_group_metadata)
from budding_yeast_v2.data.systems.budding.phenotypes import analyze_mass_pattern
from budding_yeast_v2.data.systems.budding.specification import SPEC
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_six_patterns import features


def six_curves():
    y = np.zeros((6, 39, 60), dtype=np.float64)
    y[:, 0] = 2.; y[:, 2] = .1; y[:, 32] = .5; y[:, 33] = .1
    y[0, 0, 12:] -= .2; y[0, 0, 40:] -= .2
    y[1, 0, 12:] -= .2
    y[3, 33] = .5
    y[4, 2, -1] = .4
    y[5, 2, 10] = 10.01
    return y


def biology():
    # A multi-division trajectory need not be biologically viable.
    return np.array(['Inviable: Unbudded Division', 'Viable',
        'Inviable: G1 Arrest', 'Inviable: Sequence Failure',
        'Inviable: M-phase Arrest', 'Inviable: M-phase Arrest'])


def abc(y):
    return np.array([analyze_mass_pattern(row[0]) for row in y])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_all_groups_match_independent_audit_and_losslessly_coarsen_to_abc(dtype):
    y = six_curves().astype(dtype)
    result = pattern_group_metadata(y, batch_size=2)
    np.testing.assert_array_equal(result['pattern_group_codes'], PATTERN_GROUP_CODES)
    f, _ = features(y)
    np.testing.assert_array_equal(result['pattern_group_codes'], f['pattern_code'])
    np.testing.assert_array_equal(result['observed_division_counts'], f['divisions'])
    expected = [x.split(':')[0] for x in abc(y)]
    np.testing.assert_array_equal(np.asarray(PATTERN_GROUP_TO_ABC)[result['pattern_group_ids']], expected)
    for key, value in result.items():
        np.testing.assert_array_equal(value, pattern_group_metadata(y.swapaxes(1, 2))[key])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_random_boundaries_drop_clustering_and_division_priority_match_audit(dtype):
    rng = np.random.default_rng(147)
    y = rng.uniform(0, .9, (200, 39, 60)).astype(dtype)
    y[:, 2, -1] = rng.choice([.1, .35, 5., 5.01], 200)
    y[:, 32, -1] = rng.choice([.09, .1, .5], 200)
    y[:, 33] = rng.choice([.49, .5, 1.], (200, 60))
    y[:100, 0] = 2  # Exercise zero-division branches as well.
    y[100:, 2, 0] = 11  # Multi-division still has priority over high Clb2.
    for drop in (0, 6, 12):
        y[99, 0, drop+1:] -= .2
    f, _ = features(y)
    out = pattern_group_metadata(y, batch_size=7)
    np.testing.assert_array_equal(out['pattern_group_codes'], f['pattern_code'])
    np.testing.assert_array_equal(out['observed_division_counts'], f['divisions'])
    assert out['pattern_group_codes'][99] == 'P2'


@pytest.mark.parametrize('shape', [(1, 39, 39), (1, 38, 60), (1, 39, 1)])
def test_ambiguous_or_short_trajectory_rejected(shape):
    with pytest.raises(ValueError):
        pattern_group_metadata(np.zeros(shape))


def test_nonfinite_cannot_become_a_biological_or_neutral_group():
    y = six_curves(); y[1, 10, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        pattern_group_metadata(y)


def save_old(path, *, include_biology=True, **extra):
    y = six_curves()
    fields = dict(data=y, params=np.zeros((6, 141)), mutant_names=np.array([f'SYN_{i}' for i in range(6)]),
                  pattern_labels=abc(y))
    if include_biology:
        fields['labels'] = biology()
    fields.update(extra)
    np.savez_compressed(path, **fields)


def test_old_loading_derives_groups_without_rewriting_or_fabricating_biology(tmp_path):
    path = tmp_path/'old.npz'; save_old(path)
    before = path.read_bytes()
    loaded = load_yeast_dataset_universal(path, return_metadata=True)
    np.testing.assert_array_equal(loaded[4], abc(six_curves()))
    np.testing.assert_array_equal(loaded[5]['labels'], biology())
    np.testing.assert_array_equal(loaded[5]['pattern_group_codes'], PATTERN_GROUP_CODES)
    assert loaded[5]['pattern_group_source'] == 'derived_from_physical_trajectory'
    assert path.read_bytes() == before
    missing = tmp_path/'missing.npz'; save_old(missing, include_biology=False)
    meta = load_yeast_dataset_universal(missing, return_metadata=True)[5]
    assert not meta['biological_labels_available'].any()
    assert (meta['labels'] == UNKNOWN_BIOLOGY_LABEL).all()


@pytest.mark.parametrize('tamper', ['code', 'partial', 'version', 'rules'])
def test_stored_supplement_is_checked_against_physical_trajectories(tmp_path, tamper):
    fields = pattern_group_metadata(six_curves())
    if tamper == 'code': fields['pattern_group_codes'][0] = 'P5b'
    if tamper == 'partial': fields.pop('pattern_group_ids')
    if tamper == 'version': fields['pattern_group_version'] = np.array('unsupported')
    if tamper == 'rules': fields['pattern_group_rules_json'] = np.array('{}')
    path = tmp_path/'tampered.npz'; save_old(path, **fields)
    with pytest.raises(ValueError):
        load_yeast_dataset_universal(path, return_metadata=True)


def test_modular_writer_preserves_biology_abc_and_adds_versioned_groups(tmp_path):
    y = six_curves()
    rows = [SimulationResult(Candidate(i, f'SYN_{i}', {}, 1), row, np.zeros(141),
                row[:, 0], biology()[i], abc(y)[i], 'success') for i, row in enumerate(y)]
    path = tmp_path/'new.npz'
    report = dict(mutation_profile='mutant_aware', distribution_profile='mutant_aware', rng_contract='test')
    write_dataset(path, replace(SPEC, steps=60), rows, GenerationConfig('budding_random'), report)
    *_, meta = load_yeast_dataset_universal(path, return_metadata=True)
    assert meta['pattern_group_source'] == 'stored_verified'
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['labels'], biology())
        np.testing.assert_array_equal(saved['pattern_labels'], abc(y))
        np.testing.assert_array_equal(saved['data'], y)
        assert all(k in saved for k in PATTERN_GROUP_FIELDS)
    pool = tmp_path/'pool.npz'
    write_dataset(pool, replace(SPEC, steps=60), rows, GenerationConfig('budding_pool'), report, parameter_only=True)
    with np.load(pool) as saved:
        assert not any(k in saved for k in PATTERN_GROUP_FIELDS)


def test_fission_preserves_own_biology_without_budding_thresholds():
    fields = trajectory_label_metadata(np.ones((2, 16, 60)), ['Viable', 'arrest'], 'fission')
    assert not any(k in fields for k in PATTERN_GROUP_FIELDS)
    np.testing.assert_array_equal(fields['labels'], ['Viable', 'arrest'])


def test_al_append_targets_use_actual_abc_and_snapshots_keep_all_label_layers(tmp_path):
    from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
    y = six_curves(); x = torch.zeros((6, 39, 2, 60)); p = torch.zeros((6, 141))
    dm = SimpleNamespace(system_id='budding', t_max=210,
        train_dataset_synth=TensorDataset(x[:1], p[:1], torch.tensor(y[:1]), torch.tensor([0])),
        _mutant_names=['original'], _train_idx_synth=np.array([0]), pattern_labels=abc(y[:1]),
        biological_labels=biology()[:1], biological_labels_available=np.ones(1, bool))
    for key in ('pattern_group_codes', 'pattern_group_labels', 'pattern_group_ids', 'observed_division_counts'):
        setattr(dm, key, pattern_group_metadata(y[:1])[key])
    bd = dict(X=x, P=p, Y=torch.tensor(y), Y_real=y, P_real=p.numpy(),
              names=[f'added_{i}' for i in range(6)], labels=abc(y), biological_labels=biology(),
              levels=[1,2,2,3,3,3],requested_levels=[1,2,2,3,3,3],module_counts=[1,2,2,3,3,3],
              candidate_ids=np.arange(6,dtype=np.int64))
    cb = ActiveLearningCallback(strategy='al4pde_random')
    cb._pool_size = 6
    cb.select = lambda *args: cb.make_selection_plan(np.arange(6))
    cb._build_broadcast_from_pool = lambda *args, **kwargs: bd
    cb._al4pde_al_iter = 0
    for prefix in ('_cum_synth', '_benchmark_real'):
        for suffix in ('data', 'params', 'names', 'labels', 'biological_labels'):
            setattr(cb, f'{prefix}_{suffix}', [])
    cb._param_names_cached = np.array(SPEC.param_names)
    # Current snapshots retain the original sample and exact-L metadata.
    source=tmp_path/'initial.npz'
    np.savez_compressed(source,data=y[:1],params=p[:1].numpy(),mutant_names=np.array(['original']),
        pattern_labels=abc(y[:1]),labels=biology()[:1],levels=np.array([1]),
        requested_levels=np.array([1]),module_counts=np.array([1]))
    dm.dataset_path=str(source);dm.variable_names=SPEC.variable_names
    dm._test_idx_real=[];dm._val_idx_synth=[];dm._test_idx_synth=[]
    cb._cum_synth_data=[y[0]];cb._cum_synth_params=[p[0].numpy()]
    cb._cum_synth_names=['original'];cb._cum_synth_labels=list(abc(y[:1]))
    cb._cum_synth_biological_labels=list(biology()[:1])
    from budding_yeast_v2.utils.sampling.io import cache_source_metadata,configure_snapshot_storage
    with np.load(source) as archive:cache_source_metadata(cb,dm,archive)
    configure_snapshot_storage(cb,dm,tmp_path)
    assert cb.al4pde_select_and_append(None, dm, device='cpu', exp_dir=str(tmp_path)) == 6
    np.testing.assert_array_equal(dm.train_dataset_synth.tensors[3], [0, 0, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(dm.pattern_labels[1:], abc(y))
    np.testing.assert_array_equal(dm.biological_labels[1:], biology())
    np.testing.assert_array_equal(dm.pattern_group_codes[1:], PATTERN_GROUP_CODES)
    saved = tmp_path/'rounds/round_01/dataset.json'
    meta = load_yeast_dataset_universal(saved, return_metadata=True)[5]
    np.testing.assert_array_equal(meta['labels'][1:], biology())
    np.testing.assert_array_equal(meta['pattern_group_codes'][1:], PATTERN_GROUP_CODES)
    assert not meta['is_real'].any()
    assert meta['time_grid'][-1] == 210


def test_al_worker_retains_biology_with_legacy_bridge_tuple_compatibility(monkeypatch):
    from budding_yeast_v2.utils.sampling import oracle as al
    y = np.pad(six_curves()[0],((0,0),(0,440)),mode='edge')
    monkeypatch.setattr(al, 'simulate_mutant_mass', lambda *args: ('mutant', y, biology()[0], abc(y[None])[0]))
    assert len(al.solve_single_mass_bridge('mutant', {})) == 4
    result = al._solve_single_mass_task_wrapper(('mutant', {}))
    assert len(result) == 5 and result[4] == biology()[0]
    np.testing.assert_array_equal(result[1], y.astype(np.float32))
    assert result[0] is None and result[1].dtype==np.float32


def test_evaluation_sidecar_joins_by_indices_and_does_not_change_metrics(tmp_path):
    from budding_yeast_v2.eval import write_supplementary_label_metrics
    y = six_curves(); groups = pattern_group_metadata(y)
    dm = SimpleNamespace(_mutant_names=[f'm_{i}' for i in range(6)], biological_labels=biology(),
                         pattern_group_version=str(groups['pattern_group_version']), **{
                             k:groups[k] for k in ('pattern_group_codes', 'pattern_group_labels', 'observed_division_counts')})
    indices = [5, 0]
    rows = [dict(Mutant=dm._mutant_names[i], Pattern=abc(y)[i], MAE=i+.1, MSE=i+.2,
                 Relative_L2=i+.3, Correlation=.9) for i in indices]
    original = [dict(r) for r in rows]
    path = tmp_path/'groups.csv'; write_supplementary_label_metrics(rows, indices, dm, path)
    with path.open() as handle: saved = list(csv.DictReader(handle))
    assert rows == original
    assert [r['Pattern_Group'] for r in saved] == ['P5b', 'P1']
    assert [r['Biological_Label'] for r in saved] == list(biology()[indices])
    assert float(saved[0]['MAE']) == rows[0]['MAE']
    with pytest.raises(ValueError, match='order'):
        write_supplementary_label_metrics(rows, indices[::-1], dm, path)


def test_abc_unknown_target_is_not_silently_filled_as_a():
    with pytest.raises(ValueError, match='unknown ABC'):
        abc_pattern_ids(['Pattern_Unknown'])


@pytest.mark.parametrize('include_biology', [True, False])
def test_al_round_zero_preserves_selected_original_labels_and_marks_missing_biology(tmp_path, include_biology):
    from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
    source = tmp_path/'initial.npz'; save_old(source, include_biology=include_biology,
        levels=np.array([0,1,1,1,1,1]),requested_levels=np.array([0,1,1,1,1,1]),
        module_counts=np.array([-1,1,1,1,1,1]),is_real=np.array([True,False,False,False,False,False]))
    y = six_curves(); indices = np.array([2, 3])
    dm = SimpleNamespace(dataset_path=str(source), system_id='budding', t_max=210,
        variable_names=SPEC.variable_names,_train_idx_synth=indices, _test_idx_real=np.array([0]),
        _mutant_names=[f'SYN_{i}' for i in range(6)],
        train_dataset_synth=TensorDataset(torch.zeros((2, 39, 2, 60)),
            torch.zeros((2, 141)), torch.tensor(y[indices]), torch.tensor([2, 2])))
    cb = ActiveLearningCallback(strategy='al4pde_random')
    cb._generate_param_only_pool = lambda dm: None
    cb._pool_size = 1; cb._pool_mutant_names = np.array(['unused_pool'])
    cb.al4pde_setup(dm, device='cpu', exp_dir=str(tmp_path))
    saved = load_yeast_dataset_universal(tmp_path/'rounds/round_00/dataset.json', return_metadata=True)
    meta = saved[5]
    np.testing.assert_array_equal(meta['is_real'], [True, False, False])
    np.testing.assert_array_equal(saved[4], abc(y)[[0, 2, 3]])
    np.testing.assert_array_equal(meta['pattern_group_codes'], ['P1', 'P3', 'P4'])
    if include_biology:
        np.testing.assert_array_equal(meta['labels'], biology()[[0, 2, 3]])
    else:
        assert not meta['biological_labels_available'].any()
        assert (meta['labels'] == UNKNOWN_BIOLOGY_LABEL).all()


def test_al_initial_set_built_from_pool_also_has_actual_abc_targets_and_biology(tmp_path):
    from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
    y = six_curves(); x = torch.zeros((6, 39, 2, 60)); p = torch.zeros((6, 141))
    dm = SimpleNamespace(system_id='budding',t_max=210,variable_names=SPEC.variable_names,_mutant_names=[],
        pattern_labels=np.array([], dtype=str), biological_labels=np.array([], dtype=str),
        biological_labels_available=np.array([], dtype=bool),
        train_dataset_synth=TensorDataset(x[:0], p[:0], torch.tensor(y[:0]), torch.empty(0, dtype=torch.long)))
    for key in ('pattern_group_codes', 'pattern_group_labels', 'pattern_group_ids', 'observed_division_counts'):
        setattr(dm, key, pattern_group_metadata(y[:0])[key])
    bd = dict(X=x, P=p, Y=torch.tensor(y), Y_real=y, P_real=p.numpy(),
              names=[f'new_{i}' for i in range(6)],labels=abc(y),biological_labels=biology(),
              levels=[2,2,2,3,3,3],requested_levels=[2,2,2,3,3,3],module_counts=[2,2,2,3,3,3])
    cb = ActiveLearningCallback(strategy='al4pde_random', initial_train_size=6)
    cb._generate_param_only_pool = lambda dm: None
    cb._pool_size = 6
    from budding_yeast_v2.data.systems.generation import level_schedule
    from budding_yeast_v2.data.systems.sampling_policy import POOL_RATIOS
    levels=level_schedule(6,POOL_RATIOS,np.random.RandomState(cb.random_seed+1))
    cb._pool_mass_meta=[{'level':int(level)} for level in levels]
    for key in ('levels','requested_levels','module_counts'):bd[key]=list(map(int,levels))
    cb._build_broadcast_from_pool = lambda *args: bd
    cb.al4pde_setup(dm, device='cpu', exp_dir=str(tmp_path))
    np.testing.assert_array_equal(dm.train_dataset_synth.tensors[3], [0, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(dm.biological_labels, biology())
    np.testing.assert_array_equal(dm.pattern_group_codes, PATTERN_GROUP_CODES)
    meta = load_yeast_dataset_universal(tmp_path/'rounds/round_00/dataset.json', return_metadata=True)[5]
    np.testing.assert_array_equal(meta['labels'], biology())
