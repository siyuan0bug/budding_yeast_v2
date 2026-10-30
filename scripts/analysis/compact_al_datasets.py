"""Replace redundant AL snapshots with verified, exact dataset manifests.

Default is a dry run. --apply verifies every field before unlinking each old NPZ.
Only run directories explicitly listed in the launch manifest are considered.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from budding_yeast_v2.data.dataset_archive import (
    open_dataset, shared_initial_archive, write_dataset_manifest,
    verify_dataset_equal, _atomic_json,
)


def ordered_segments(target_names, blocks):
    lookup = {}
    for block in blocks:
        with np.load(block, allow_pickle=True) as archive:
            for i, name in enumerate(archive['mutant_names']):
                name = str(name)
                if name in lookup:
                    raise ValueError(f'Duplicate sample name across blocks: {name}')
                lookup[name] = (str(block), i)
    segments = []
    for name in target_names:
        block, index = lookup[str(name)]
        if not segments or segments[-1][0] != block:
            segments.append((block, []))
        segments[-1][1].append(index)
    return segments


def compact_run(run, *, apply=False):
    run = Path(run).resolve()
    initial = run / 'data' / 'initial_dataset.npz'
    snapshots = sorted(run.glob('rounds/round_*/dataset.npz'))
    candidates = [*snapshots, *([initial] if initial.exists() else [])]
    result = dict(run=str(run), apply=apply, candidates=len(candidates), files=[],
                  old_bytes=sum(p.stat().st_size for p in candidates))
    if not apply or not candidates:
        return result
    if not initial.exists():
        initial_manifest = run / 'data' / 'initial_dataset.json'
        if not initial_manifest.exists():
            raise ValueError(f'Missing immutable initial dataset reference: {run}')
        with initial_manifest.open() as handle:
            sources = json.load(handle)['sources']
        initial = (initial_manifest.parent / next(iter(sources.values()))['path']).resolve()
    shared = Path(shared_initial_archive(initial, run))
    additions = sorted(run.glob('acquisitions/after_round_*/added_samples.npz'))
    for snapshot in snapshots:
        started = time.monotonic()
        round_index = int(snapshot.parent.name.rsplit('_', 1)[1])
        blocks = [shared, *[p for p in additions if int(p.parent.name.rsplit('_', 1)[1]) < round_index]]
        size = snapshot.stat().st_size
        with np.load(snapshot, allow_pickle=True) as target:
            segments = ordered_segments(target['mutant_names'], blocks)
            manifest = write_dataset_manifest(snapshot.with_suffix('.json'), target, segments,
                provenance=dict(compacted_from=snapshot.name, verification='all_fields_bitwise'))
        verify_dataset_equal(snapshot, manifest)
        # All checks pass while the original still exists; only now release it.
        snapshot.unlink()
        record = dict(path=str(snapshot), manifest=manifest, released_bytes=size,
                      verified='all_fields_bitwise', seconds=time.monotonic()-started)
        result['files'].append(record)
        print(json.dumps(record), flush=True)
    initial_copy = run / 'data' / 'initial_dataset.npz'
    if initial_copy.exists():
        size = initial_copy.stat().st_size
        with np.load(initial_copy, allow_pickle=True) as target:
            manifest = write_dataset_manifest(initial_copy.with_suffix('.json'), target, [(shared, None)],
                provenance=dict(role='shared_initial', verification='all_fields_bitwise'))
        verify_dataset_equal(initial_copy, manifest)
        initial_copy.unlink()
        result['files'].append(dict(path=str(initial_copy), manifest=manifest,
                                    released_bytes=size, verified='all_fields_bitwise'))
    # Keep the existing pool provenance and record the now-shared initial path.
    pool_manifest = run / 'data' / 'pool_manifest.json'
    if pool_manifest.exists():
        payload = json.loads(pool_manifest.read_text())
        payload.update(initial_dataset=str(shared), storage_layout='shared_initial_append_only')
        _atomic_json(pool_manifest, payload)
    result['shared_initial'] = str(shared)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, help='Launch manifest with explicit run paths')
    parser.add_argument('--run', type=Path, action='append', default=[])
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--workers', type=int, default=1, help='Independent run conversions; bounded process count')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = list(args.run)
    if args.manifest:
        payload = json.loads(args.manifest.read_text())
        for task in payload['tasks']:
            runs.append(Path(task.get('experiment_dir') or task.get('exp_dir') or task.get('run_dir') or task['path']))
    if not runs:
        parser.error('Supply --run or --manifest')
    if args.workers < 1:
        parser.error('--workers must be positive')
    report = dict(apply=args.apply, runs=[])
    def collect(run, result=None, error=None):
        print(f'[compact] {run}', flush=True)
        report['runs'].append(result if error is None else dict(run=str(run), error=repr(error)))
        _atomic_json(args.output, report)
    runs = list(dict.fromkeys(runs))
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as executor:
            futures = {executor.submit(compact_run, run, apply=args.apply): run for run in runs}
            for future in as_completed(futures):
                run = futures[future]
                try:
                    collect(run, result=future.result())
                except Exception as exc:
                    collect(run, error=exc)
                    raise
    else:
        for run in runs:
            try:
                collect(run, result=compact_run(run, apply=args.apply))
            except Exception as exc:
                collect(run, error=exc)
                raise
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
