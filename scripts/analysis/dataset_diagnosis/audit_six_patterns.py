#!/usr/bin/env python3
"""Read-only, streaming audit of the user-proposed Budding six-pattern v1.

This is an experimental trajectory partition, not a bifurcation detector.
Original NPZ labels, generators, samplers and training splits are never changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import re
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CODES = ('P1', 'P2', 'P3', 'P4', 'P5a', 'P5b')
VERSION = 'user_six_patterns_v1_2026-10-20'
REFERENCE = ROOT / ('experiments/mutant_aware_pure_random_100000_210m_500s/'
    'biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/'
    'seed_42/eval/per_mutant_metrics.csv')
REAL_RE = re.compile(r'^(\d+(?:\.\d+)?)_')


def array_header(path):
    with zipfile.ZipFile(path) as archive, archive.open('data.npy') as stream:
        version = np.lib.format.read_magic(stream)
        return np.lib.format._read_array_header(stream, version)


def chunks(path, chunk_size):
    """Compressed NPZ cannot be memory-mapped: stream the C-order NPY member."""
    with zipfile.ZipFile(path) as archive, archive.open('data.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        if fortran or dtype.hasobject:
            raise ValueError('Expected numeric C-order data.npy')
        stride = int(np.prod(shape[1:])) * dtype.itemsize
        for start in range(0, shape[0], chunk_size):
            n = min(chunk_size, shape[0] - start)
            raw = stream.read(n * stride)
            if len(raw) != n * stride:
                raise EOFError('Truncated data.npy')
            yield start, np.frombuffer(raw, dtype=dtype).reshape(n, *shape[1:])


def decide(f, scale=1.):
    """Exact executable tree supplied by the user; scale tests threshold sensitivity."""
    v, bmax, bend, eend, omax = (f[k] for k in
        ('divisions', 'max_clb2', 'end_clb2', 'end_esp1', 'max_ori'))
    code = np.full(len(v), 'P4', dtype='<U7')
    code[omax < .5 * scale] = 'P3'
    mitotic = (bend >= .35 * scale) | (eend < .10 * scale) | (bmax > 10 * scale)
    hyper = (bmax > 10 * scale) | (bend > 5 * scale)
    code[mitotic] = 'P5a'; code[mitotic & hyper] = 'P5b'
    code[v == 1] = 'P2'; code[v >= 2] = 'P1'
    code[~f['valid']] = 'INVALID'
    start = ((bmax >= .15 * scale) & (bmax <= .40 * scale)) | (
        (omax >= .5 * scale) & (omax <= 1.5 * scale) & (bend < .25 * scale))
    finish = ((bend >= .25 * scale) & (bend <= .80 * scale)) | (
        (eend >= .08 * scale) & (eend <= .15 * scale))
    return code, start & f['valid'], finish & f['valid']


def features(y):
    """Batch input N,39,T or N,T,39. Reject ambiguous axes and nonfinite rows."""
    y = np.asarray(y)
    if y.ndim != 3 or (y.shape[1] == 39) == (y.shape[2] == 39):
        raise ValueError(f'Expected unambiguous N,39,T or N,T,39; got {y.shape}')
    if y.shape[2] == 39:
        y = y.swapaxes(1, 2)
    if y.shape[2] < 2:
        raise ValueError('At least two time points required')
    valid = np.isfinite(y).all(axis=(1, 2))
    drops = np.diff(y[:, 0], axis=1) < -.1
    # Chain clustering: compare to every preceding detected drop, including
    # suppressed drops. This matches the attachment and production phenotype.
    positions = np.where(drops, np.arange(drops.shape[1]), -100000)
    last = np.maximum.accumulate(positions, axis=1)
    previous = np.concatenate((np.full((len(y), 1), -100000), last[:, :-1]), axis=1)
    starts = drops & (positions - previous > 10)
    f = {'valid': valid, 'divisions': starts.sum(1), 'raw_mass_drops': drops.sum(1)}
    for idx, name in ((0, 'mass'), (1, 'cln2'), (2, 'clb2'), (3, 'clb5'),
                      (4, 'sic1'), (32, 'esp1'), (33, 'ori'), (34, 'bud'), (35, 'spn')):
        a = y[:, idx]
        # The supplied reference converts extrema to Python float before
        # comparisons. Promote float32 features to reproduce boundary rounding.
        f['max_' + name] = a.max(1).astype(np.float64)
        f['end_' + name] = a[:, -1].astype(np.float64)
        f['init_' + name] = a[:, 0].astype(np.float64)
    tail = y[:, 2, -max(2, y.shape[2] // 5):]
    f['tail_clb2_range_fraction'] = np.ptp(tail, axis=1) / np.maximum(f['max_clb2'], 1e-8)
    f['negative_below_minus_1e4'] = (y.min(axis=(1, 2)) < -1e-4)
    code, start, finish = decide(f)
    f.update(pattern_code=code, critical_start=start, critical_finish=finish,
             is_bifurcation_critical=start | finish)
    f['hyper_clb2_any_v'] = (f['max_clb2'] > 10) | (f['end_clb2'] > 5)
    f['ori_reaches_one'] = f['max_ori'] >= 1
    f['spn_reaches_one'] = f['max_spn'] >= 1
    f['p3_past_clb2_inconsistent'] = (code == 'P3') & (f['max_clb2'] >= .35)
    f['p4_prose_inconsistent'] = (code == 'P4') & ((f['max_spn'] >= .2) | (f['max_clb2'] >= .35))
    # Literal prose predicates can overlap and leave holes. The executable tree
    # is exhaustive because of its ordered fallback, not because prose is MECE.
    v0 = f['divisions'] == 0
    raw = np.stack((f['divisions'] >= 2, f['divisions'] == 1,
        v0 & (f['max_ori'] < .5) & (f['max_clb2'] < .35),
        v0 & (f['max_ori'] >= .5) & (f['max_spn'] < .2) & (f['max_clb2'] < .35),
        v0 & ((f['end_clb2'] >= .35) | (f['end_esp1'] < .1)),
        v0 & f['hyper_clb2_any_v']), axis=1)
    f['prose_rule_matches'] = raw.sum(1)
    for factor, suffix in ((.9, 'minus10pct'), (1.1, 'plus10pct')):
        other, s, e = decide(f, factor)
        f['pattern_changes_' + suffix] = (other != code) & valid
        f['critical_changes_' + suffix] = ((s | e) != (start | finish)) & valid
    return f, y[:, :, 0].copy()


def make_splits(names, original_patterns, is_real, seed):
    """Current non-LHS stratified split, preserving encountered group order."""
    split = np.full(len(names), 'test_real', dtype='<U11')
    rng = np.random.RandomState(seed)
    groups = {}
    for i in np.flatnonzero(~is_real):
        groups.setdefault(str(original_patterns[i]).split(':')[0], []).append(i)
    for idx in groups.values():
        rng.shuffle(idx); n = len(idx); nv = nt = math.ceil(.1 * n)
        if n <= 2: nv = nt = 0
        elif n <= 4: nt = 0
        nt = min(nt, n - nv)
        split[idx[:nv]] = 'val_synth'; split[idx[nv:nv+nt]] = 'test_synth'
        split[idx[nv+nt:]] = 'train_synth'
    return split


def identity(path):
    st = path.stat()
    return dict(path=str(path.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns)


def write_evaluation(real_frame, eval_csv, output):
    """Refresh evaluation joins without decompressing trajectories a second time."""
    metrics = pd.read_csv(eval_csv)
    if metrics['Mutant'].duplicated().any() or real_frame.mutant_name.duplicated().any():
        raise ValueError('Duplicate real-mutant names in evaluation join')
    joined = real_frame.merge(metrics.rename(columns={'Mutant':'mutant_name'}),
        on='mutant_name', how='outer', validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Evaluation and trajectory real-mutant names differ')
    joined.to_csv(output / 'real_evaluation.csv', index=False)
    numeric = [c for c in ('MAE','MSE','Relative_L2','Correlation') if c in joined]
    joined.groupby(['pattern_code','is_bifurcation_critical'])[numeric].agg(['count','median','mean','max']).to_csv(output / 'evaluation_by_pattern.csv')
    return identity(Path(eval_csv))


def heavy_diagnostics(frame, params, ic, output):
    """Descriptive train support; no test trajectories are used to fit a model."""
    from scipy.spatial import cKDTree
    train = frame['split'].eq('train_synth').to_numpy() & frame['valid'].to_numpy()
    real = frame['is_real'].to_numpy() & frame['valid'].to_numpy()
    cols = ['divisions', 'max_clb2', 'end_clb2', 'end_esp1', 'max_ori', 'max_spn', 'end_mass']
    result = frame.loc[real, ['mutant_name', 'pattern_code', 'is_bifurcation_critical']].copy()
    if not train.any(): return
    x = np.log1p(np.maximum(frame[cols].to_numpy(dtype=np.float64), 0))
    scale = x[train].std(0) + 1e-6
    dist, idx = cKDTree(x[train] / scale).query(x[real] / scale, k=1)
    result['dynamic_feature_nn_distance'] = dist
    result['dynamic_feature_nn_name'] = frame.loc[train, 'mutant_name'].to_numpy()[idx]
    # Full input: all parameters + initial concentrations, standardised using train.
    x = np.concatenate((params.astype(np.float64), ic.astype(np.float64)), axis=1)
    scale = x[train].std(0) + 1e-6
    dist, idx = cKDTree(x[train] / scale).query(x[real] / scale, k=1)
    result['full_input_nn_distance'] = dist
    result['full_input_nn_name'] = frame.loc[train, 'mutant_name'].to_numpy()[idx]
    records = []
    pt = params[train]
    for i in np.flatnonzero(real):
        row = frame.iloc[i]; same = frame['pattern_code'].eq(row['pattern_code']).to_numpy()
        samecrit = frame['is_bifurcation_critical'].eq(row['is_bifurcation_critical']).to_numpy()
        records.append(dict(train_same_pattern=int((train & same).sum()),
            train_same_pattern_and_critical=int((train & same & samecrit).sum()),
            train_same_zero_parameter_face=int(np.all(pt[:, params[i] == 0] == 0, axis=1).sum()),
            train_exact_parameters=int(np.all(pt == params[i], axis=1).sum())))
    result = pd.concat((result.reset_index(drop=True), pd.DataFrame(records)), axis=1)
    result.to_csv(output / 'real_training_support.csv', index=False)


def run_one(job):
    path, outroot, chunk_size, seed, diagnostic_name, eval_csv = job
    path = Path(path); out = Path(outroot) / path.stem; out.mkdir(parents=True, exist_ok=True)
    before = identity(path); start_time = time.monotonic()
    with np.load(path, allow_pickle=True) as z:
        names = z['mutant_names'].astype(str)
        original = z['pattern_labels'].astype(str) if 'pattern_labels' in z else np.full(len(names), 'MISSING')
        labels = z['labels'].astype(str) if 'labels' in z else np.full(len(names), 'MISSING')
        name_real = np.array([bool(REAL_RE.match(n)) for n in names])
        real = z['is_real'].astype(bool) if 'is_real' in z else name_real
        if real.shape != (len(names),): raise ValueError('is_real shape mismatch')
        params = z['params'] if path.name == diagnostic_name else None
        time_grid = z['time_grid'] if 'time_grid' in z else None
    collected, initial, real_hashes = [], [], {}
    for start, block in chunks(path, chunk_size):
        f, ic = features(block); n = len(ic)
        if start + n > len(names): raise ValueError('data/name count mismatch')
        collected.append(pd.DataFrame(f))
        if params is not None: initial.append(ic)
        b = block if block.shape[1] == 39 else block.swapaxes(1, 2)
        for j in np.flatnonzero(real[start:start+n]):
            # Cast permits identical float32/float64 exports to share a digest.
            real_hashes[int(start+j)] = hashlib.sha256(np.asarray(b[j], dtype='<f4').tobytes()).hexdigest()
        if (start // chunk_size) % 80 == 0:
            print(f'{path.stem}: {start+n:,}/{len(names):,}', flush=True)
    frame = pd.concat(collected, ignore_index=True)
    if len(frame) != len(names): raise ValueError('data/name count mismatch')
    frame.insert(0, 'row_index', np.arange(len(names)))
    frame.insert(1, 'mutant_name', names); frame.insert(2, 'is_real', real)
    frame['original_pattern'] = original; frame['original_label'] = labels
    frame['split'] = make_splits(names, original, real, seed)
    frame['level'] = pd.Series(names).str.extract(r'_L(\d+)(?:_|$)', expand=False).fillna('-1').astype(int)
    expected = np.select((frame.divisions >= 2, frame.divisions == 1), ('Pattern_A', 'Pattern_B'), 'Pattern_C')
    frame['old_pattern_mismatch'] = frame.valid & (np.array([s.split(':')[0] for s in original]) != expected)
    # Store independent sidecars; never rewrite the source dataset.
    frame.to_csv(out / 'samples.csv.gz', index=False, compression='gzip')
    real_frame = frame.loc[real].copy()
    real_frame['trajectory_sha256_float32'] = [real_hashes[i] for i in real_frame.row_index]
    real_frame.to_csv(out / 'real_mutants.csv', index=False)
    group_columns = ['is_real', 'split', 'pattern_code', 'is_bifurcation_critical', 'level']
    frame.groupby(group_columns, dropna=False).size().rename('count').reset_index().to_csv(out / 'joint_counts.csv', index=False)
    pd.crosstab(frame.original_pattern, frame.pattern_code).to_csv(out / 'old_to_new_patterns.csv')
    pd.crosstab(frame.original_label, frame.pattern_code).to_csv(out / 'biology_to_new_patterns.csv')
    if params is not None:
        heavy_diagnostics(frame, params, np.concatenate(initial), out)
        if eval_csv and Path(eval_csv).exists():
            write_evaluation(real_frame, eval_csv, out)
    counts = []
    for population, mask in (('all', np.ones(len(frame), bool)), ('real', real),
                             ('synthetic', ~real), ('train_synth', frame.split.eq('train_synth').to_numpy())):
        sub = frame.loc[mask]; total = len(sub)
        for code in (*CODES, 'INVALID'):
            part = sub.loc[sub.pattern_code.eq(code)]
            counts.append(dict(population=population, pattern_code=code, count=len(part),
                percent=100*len(part)/max(total,1), total=total,
                critical_count=int(part.is_bifurcation_critical.sum())))
    diagnostics = ['old_pattern_mismatch','p3_past_clb2_inconsistent','p4_prose_inconsistent',
        'negative_below_minus_1e4','pattern_changes_minus10pct','pattern_changes_plus10pct',
        'critical_changes_minus10pct','critical_changes_plus10pct','hyper_clb2_any_v']
    summary = dict(version=VERSION, identity=before, shape=list(array_header(path)[0]),
        seconds=time.monotonic()-start_time, real_metadata_disagreements=int(np.sum(real != name_real)),
        real_name_duplicates=int(real_frame.mutant_name.duplicated().sum()),
        time_grid_source='stored' if time_grid is not None else 'legacy_filename_210min_500steps',
        split_seed=seed, counts=counts, evaluation_csv=eval_csv if params is not None else None,
        evaluation_identity=identity(Path(eval_csv)) if params is not None and eval_csv and Path(eval_csv).exists() else None,
        diagnostics={p:{k:int(frame.loc[m,k].sum()) for k in diagnostics} | {
            'prose_gaps':int((frame.loc[m,'prose_rule_matches'].eq(0) & frame.loc[m,'valid']).sum()),
            'prose_overlaps':int((frame.loc[m,'prose_rule_matches'].gt(1) & frame.loc[m,'valid']).sum()),
            'critical_total':int(frame.loc[m,'is_bifurcation_critical'].sum())}
            for p,m in (('real',real),('synthetic',~real))})
    if identity(path) != before: raise RuntimeError('Source file changed during audit')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    print(f'DONE {path.name} {len(frame):,} in {summary["seconds"]:.1f}s', flush=True)
    return path.stem, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('/home/users/zsy/mydata/budding_yeast'))
    parser.add_argument('--datasets', type=Path, nargs='*', help='Explicit files; default recursively all non-LHS NPZ')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--diagnostic-dataset', default='mutant_aware_pure_random_100000_210m_500s.npz')
    parser.add_argument('--eval-csv', type=Path, default=REFERENCE)
    parser.add_argument('--resume', action='store_true', help='Reuse summaries only when source identity and version match')
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_size < 1: parser.error('workers and chunk-size must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(set(p.resolve() for p in (args.datasets or args.data_dir.rglob('*.npz'))))
    manifest, jobs, results, errors = [], [], {}, {}
    used_stems = set()
    for p in sources:
        row = dict(path=str(p), **{k:v for k,v in identity(p).items() if k != 'path'})
        if 'lhs' in str(p.relative_to(args.data_dir.resolve()) if p.is_relative_to(args.data_dir.resolve()) else p.name).lower():
            row['status'] = 'excluded_deprecated_lhs'
        else:
            with zipfile.ZipFile(p) as z: has_data = 'data.npy' in z.namelist()
            if not has_data: row['status'] = 'excluded_no_trajectory'
            else:
                shape, ft, dtype = array_header(p); row.update(shape=list(shape), dtype=str(dtype))
                if len(shape) != 3 or (shape[1] == 39) == (shape[2] == 39):
                    row['status'] = 'excluded_not_budding39'
                else:
                    if p.stem in used_stems:
                        parser.error('Included dataset stems must be unique within one output directory')
                    used_stems.add(p.stem)
                    row['status'] = 'scheduled'
                    cached = args.output_dir / p.stem / 'summary.json'
                    if args.resume and cached.exists():
                        s = json.loads(cached.read_text())
                        if s['identity'] == identity(p) and s['version'] == VERSION and s['split_seed'] == args.split_seed:
                            if p.name == args.diagnostic_dataset and args.eval_csv.exists() and s.get('evaluation_identity') != identity(args.eval_csv):
                                real_frame = pd.read_csv(cached.parent / 'real_mutants.csv')
                                s['evaluation_identity'] = write_evaluation(real_frame, str(args.eval_csv), cached.parent)
                                s['evaluation_csv'] = str(args.eval_csv)
                                cached.write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
                            results[p.stem] = s; row['status'] = 'cached'
                    if row['status'] == 'scheduled':
                        jobs.append((str(p), str(args.output_dir), args.chunk_size, args.split_seed,
                                     args.diagnostic_dataset, str(args.eval_csv)))
        manifest.append(row)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(f'Processing {len(jobs)} files, cached {len(results)}, excluded {len(manifest)-len(jobs)-len(results)}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, j): j[0] for j in jobs}
        for future in as_completed(futures):
            try:
                name, result = future.result(); results[name] = result
            except Exception as exc:
                errors[futures[future]] = repr(exc); print(f'FAILED {futures[future]}: {exc!r}', flush=True)
            (args.output_dir / 'progress.json').write_text(json.dumps(dict(completed=list(results),errors=errors),indent=2)+'\n')
    rows = [dict(dataset=name, **row) for name,s in sorted(results.items()) for row in s['counts']]
    pd.DataFrame(rows).to_csv(args.output_dir / 'all_pattern_counts.csv', index=False)
    (args.output_dir / 'errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    if errors: raise SystemExit(1)


if __name__ == '__main__':
    main()
