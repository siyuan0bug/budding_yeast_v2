import json
from pathlib import Path

import numpy as np
import pytest

from budding_yeast_v2.data.dataset_archive import (
    open_dataset, shared_initial_archive, verify_dataset_equal, write_dataset_manifest,
    dataset_field_chunks, read_dataset_rows, round_dataset_paths,
)


def fixture_blocks(tmp_path):
    initial = tmp_path / 'initial.npz'
    added = tmp_path / 'added.npz'
    common = dict(time_grid=np.linspace(0, 210, 7), variable_names=np.array(['a', 'b']))
    a = dict(data=np.arange(56, dtype=np.float32).reshape(4, 2, 7),
             params=np.arange(12, dtype=np.float64).reshape(4, 3),
             y_init=np.array([[1+1e-12, 0]]*4, dtype=np.float64),
             mutant_names=np.array(['real', 'train_a', 'val', 'test']),
             labels=np.array(['good']*4), sample_id=np.arange(1, 5), **common)
    b = dict(data=np.arange(28, dtype=np.float32).reshape(2, 2, 7) / 3,
             params=np.arange(6, dtype=np.float64).reshape(2, 3) / 7,
             y_init=np.ones((2, 2), dtype=np.float64),
             mutant_names=np.array(['added_long_a', 'added_b']), labels=np.array(['unknown']*2),
             sample_id=np.arange(1, 3), **common)
    np.savez_compressed(initial, **a)
    np.savez_compressed(added, **b)
    target = {key: np.concatenate([a[key][:2], b[key], a[key][2:]])
              for key in ('data', 'params', 'y_init', 'mutant_names', 'labels')}
    target.update(common, sample_id=np.arange(1, 7), split_train=np.array([1, 2, 3]),
                  split_val=np.array([4]), split_test=np.array([5]), split_real=np.array([0]),
                  empty=np.array([], dtype=np.int16), scalar=np.array('new_metadata'))
    segments = [(initial, [0, 1]), (added, None), (initial, [2, 3])]
    return initial, added, target, segments


def test_exact_reconstruction_alias_and_relocation(tmp_path):
    _, _, target, segments = fixture_blocks(tmp_path)
    old = tmp_path / 'dataset.npz'
    np.savez_compressed(old, **target)
    manifest = write_dataset_manifest(old, target, segments)
    assert verify_dataset_equal(old, manifest)
    spec = json.loads(Path(manifest).read_text())
    assert 'pieces' in spec['fields']['data']
    assert 'pieces' in spec['fields']['params']
    assert 'pieces' in spec['fields']['y_init']
    assert spec['fields']['sample_id']['source'] != spec['fields']['time_grid']['source']
    old.unlink()
    with open_dataset(old) as archive:
        assert set(archive.files) == set(target)
        for key in target:
            assert archive[key].dtype == target[key].dtype
            assert archive[key].tobytes() == target[key].tobytes()
    relocated = tmp_path.parent / (tmp_path.name + '_moved')
    tmp_path.rename(relocated)
    with open_dataset(relocated / 'dataset.json') as archive:
        np.testing.assert_array_equal(archive['data'], target['data'])
    relocated.rename(tmp_path)


def test_changed_or_missing_blocks_fail(tmp_path):
    initial, added, target, segments = fixture_blocks(tmp_path)
    manifest = write_dataset_manifest(tmp_path/'dataset.json', target, segments)
    with added.open('ab') as handle:
        handle.write(b'changed')
    with open_dataset(manifest) as archive, pytest.raises(ValueError, match='missing/changed'):
        archive['data']
    initial.unlink()
    with open_dataset(manifest) as archive, pytest.raises(FileNotFoundError):
        archive['data']


def test_wrong_order_or_rounded_trajectory_rejected(tmp_path):
    _, _, target, segments = fixture_blocks(tmp_path)
    target['data'] = target['data'].copy()
    target['data'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='exact target'):
        write_dataset_manifest(tmp_path/'dataset.json', target, segments)
    assert not (tmp_path/'dataset.json').exists()


def test_shared_initial_is_independent_immutable_copy(tmp_path):
    initial, _, target, _ = fixture_blocks(tmp_path)
    root = tmp_path/'al_budding'
    one = shared_initial_archive(initial, root/'fno/loss/hparams/seed_42')
    two = shared_initial_archive(initial, root/'fno/other/hparams/seed_42')
    assert one == two
    assert Path(one).stat().st_ino != initial.stat().st_ino
    assert not (Path(one).stat().st_mode & 0o222)
    np.savez_compressed(initial, changed=np.array([1]))
    with open_dataset(one) as archive:
        np.testing.assert_array_equal(archive['data'][:2], target['data'][:2])


def test_chunk_and_selected_row_analysis_preserve_order(tmp_path):
    _, _, target, segments = fixture_blocks(tmp_path)
    path = write_dataset_manifest(tmp_path/'dataset.json', target, segments)
    chunks = list(dataset_field_chunks(path, chunk_size=1))
    assert [start for start, _ in chunks] == list(range(6))
    np.testing.assert_array_equal(np.concatenate([chunk for _, chunk in chunks]), target['data'])
    np.testing.assert_array_equal(read_dataset_rows(path, [5, 0, 2, 2]), target['data'][[5, 0, 2, 2]])


def test_existing_run_compaction_is_verified_and_repeatable(tmp_path):
    from budding_yeast_v2.scripts.analysis.compact_al_datasets import compact_run
    import shutil
    initial, added, target, _ = fixture_blocks(tmp_path)
    run = tmp_path/'al_budding/fno/loss/hparams/seed_42'
    (run/'data').mkdir(parents=True)
    shutil.copyfile(initial, run/'data/initial_dataset.npz')
    added_path=run/'acquisitions/after_round_00/added_samples.npz'
    added_path.parent.mkdir(parents=True)
    shutil.copyfile(added, added_path)
    snapshot=run/'rounds/round_01/dataset.npz'
    snapshot.parent.mkdir(parents=True)
    np.savez_compressed(snapshot, **target)
    assert compact_run(run)['candidates']==2
    assert snapshot.exists()
    result=compact_run(run, apply=True)
    assert len(result['files'])==2 and not snapshot.exists()
    assert round_dataset_paths(run)==[snapshot.with_suffix('.json')]
    with open_dataset(snapshot) as archive:
        for key in target:
            assert archive[key].tobytes()==target[key].tobytes()
    assert compact_run(run, apply=True)['candidates']==0
