"""Snapshot AL status, paired real-mutant errors and physical sampling domains.

Reads existing artifacts only; never trains, queries an oracle or alters runs.
Trajectory arrays are streamed from NPZ archives to avoid multi-GB RAM copies.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo
import zipfile

import numpy as np
from budding_yeast_v2.data.dataset_archive import open_dataset, round_dataset_paths, dataset_field_chunks
import pandas as pd

TZ = ZoneInfo('Asia/Shanghai')
METRICS = ('MAE', 'MSE', 'Relative_L2', 'Correlation')


def active_runs():
    text = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    return {line.split('--exp_dir ', 1)[1].split()[0]: int(line.split()[0])
            for line in text.splitlines() if 'train.py ' in line and '--exp_dir ' in line}


def discover(root):
    running = active_runs()
    exits = {}
    # A retry reuses the canonical experiment path after archiving its failed
    # predecessor. Apply timestamped batches in order, never filesystem order.
    for path in sorted((root.parent / 'batch_logs').rglob('*.tsv')):
        for line in path.read_text().splitlines():
            fields = line.replace('\\t', '\t').split('\t')
            if len(fields) == 5 and fields[3].lstrip('-').isdigit():
                exits[fields[4]] = int(fields[3])
    result = []
    now = datetime.now(TZ)
    for config in sorted(root.glob('*/*/seed_*/config.json')):
        run = config.parent
        args = json.loads(config.read_text())['args']
        records_path = run / 'summary/rounds_summary.json'
        records = json.loads(records_path.read_text()) if records_path.exists() else []
        log = run / 'train.log'
        text = log.read_text(errors='replace') if log.exists() else ''
        pid = running.get(str(run))
        age = (now.timestamp() - log.stat().st_mtime) / 3600 if log.exists() else None
        code = exits.get(str(run))
        status = ('running_stalled' if age > 1 else 'running') if pid else (
            'completed' if code == 0 and len(records) == int(args['num_al_rounds'])
            else 'failed' if 'Traceback (most recent call last)' in text or code else 'incomplete')
        signature = run.parent.name
        cohort = signature.rsplit('_refit_', 1)[-1]
        error_lines=(re.sub(r'^\[\d{4}-\d\d-\d\d [^\]]+\]\s*','',line)
                     for line in reversed(text.splitlines()))
        error = next((line for line in error_lines if re.match(r'^\w*(Error|Exception):',line)), '')
        row = dict(cohort=cohort, strategy=args['al_strategy'], seed=args['seed'],
                   status=status, completed_rounds=len(records), pid=pid, exit_code=code,
                   log_idle_hours=age, error=error, run=str(run), args=args)
        result.append((row, records))
    return result, now.isoformat()


def metric_frame(path):
    obj = json.loads(path.read_text())
    frame = pd.DataFrame(obj['per_mutant']).set_index('Mutant')
    return frame, obj


def streamed_extrema(path):
    peaks, minima = [], []
    for _, block in dataset_field_chunks(path, chunk_size=128):
        if not np.isfinite(block).all():raise ValueError(f'Nonfinite physical trajectory in {path}')
        peaks.append(block.max(2));minima.append(block.min(2))
    return np.concatenate(peaks), np.concatenate(minima)


def scan_domains(runs, output):
    stages, variables, groups = [], [], []
    invariants, examples = [], {}
    reference_real = None
    for info, records in runs:
        if info['cohort'] not in ('balanced', 'gene_ranges'):
            continue
        if info['cohort'] == 'balanced' and info['strategy'] != 'al4pde_random':
            continue
        run = Path(info['run'])
        paths = round_dataset_paths(run)
        if not paths:
            continue
        # A snapshot must be fully persisted; partially written active files are
        # deliberately ignored until a later invocation of this audit.
        paths = [p for p in paths if p.suffix == '.json' or zipfile.is_zipfile(p)]
        if not paths:
            continue
        path = paths[-1]
        key = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        cache = output / f'extrema_{key}.npz'
        size = path.stat().st_size
        with open_dataset(path, allow_pickle=False) as z:
            metadata = {k: z[k] for k in ('split_train', 'split_real', 'candidate_ids',
                'module_counts', 'pattern_group_codes', 'variable_names', 'params',
                'param_names', 'overrides_json', 'mutant_names')}
        if cache.exists():
            with open_dataset(cache) as z:
                valid = int(z['source_size']) == size
                if valid:
                    peaks, minima = z['peaks'], z['minima']
        else:
            valid = False
        if not valid:
            print(f'Streaming: {info["cohort"]}/{info["strategy"]} {path.parent.name}', flush=True)
            peaks, minima = streamed_extrema(path)
            np.savez_compressed(cache, peaks=peaks, minima=minima, source_size=size,
                                source_path=str(path))
        real_max = peaks[metadata['split_real']].max(0)
        if reference_real is None:
            reference_real = real_max
        else:
            np.testing.assert_allclose(real_max, reference_real, rtol=1e-6, atol=1e-6)
        high = (peaks[:, 1:] > np.maximum(2 * real_max[1:], 2.)).any(1) | (peaks[:, 0] > 10.)
        for prior in paths:
            with open_dataset(prior, allow_pickle=False) as z:
                count = len(z['split_train'])
                rows = metadata['split_train'][:count]
                np.testing.assert_array_equal(z['candidate_ids'][z['split_train']], metadata['candidate_ids'][rows])
            index = int(prior.parent.name.split('_')[-1])
            common = dict(cohort=info['cohort'], strategy=info['strategy'], round=index, train_samples=count)
            stages.append(dict(**common, high_count=int(high[rows].sum()), high_fraction=float(high[rows].mean()),
                               mass_above_10=int((peaks[rows, 0]>10).sum()),
                               negative_samples=int((minima[rows]<0).any(1).sum())))
            for kind, values in [('L', metadata['module_counts']), ('group', metadata['pattern_group_codes'])]:
                for value in np.unique(values[rows]):
                    local = rows[values[rows] == value]
                    groups.append(dict(**common, kind=kind, value=str(value), count=len(local),
                                       high_fraction=float(high[local].mean())))
        rows = metadata['split_train']
        names = list(metadata['variable_names'])
        keys = list(metadata['param_names'])
        for variable,key in [('TEM1','Tem1T'),('CDC15','Cdc15T'),('ESP1','ESP1T')]:
            j=names.index(variable);total=metadata['params'][:,keys.index(key)]
            bad=rows[peaks[rows,j]>total[rows]+1e-3*np.maximum(1.,total[rows])]
            invariants.append(dict(cohort=info['cohort'],strategy=info['strategy'],variable=variable,
                                   count=len(bad),train_samples=len(rows)))
            if info['cohort']=='gene_ranges' and variable=='TEM1':
                for row in bad:
                    identity=int(metadata['candidate_ids'][row])
                    examples[identity]=dict(candidate_id=identity,name=str(metadata['mutant_names'][row]),
                        peak=float(peaks[row,j]),total=float(total[row]),
                        overrides=json.loads(str(metadata['overrides_json'][row])))
        for j, name in enumerate(metadata['variable_names']):
            variables.append(dict(cohort=info['cohort'], strategy=info['strategy'], train_samples=len(rows),
                variable=str(name), real_max=float(real_max[j]), p95=float(np.quantile(peaks[rows,j], .95)),
                p99=float(np.quantile(peaks[rows,j], .99)), maximum=float(peaks[rows,j].max())))
    for name, rows in [('training_domains', stages), ('variable_peaks', variables), ('group_coverage', groups)]:
        pd.DataFrame(rows).to_csv(output / f'{name}.csv', index=False)
    pd.DataFrame(invariants).to_csv(output/'total_pool_invariants.csv',index=False)
    (output/'invariant_examples.json').write_text(json.dumps(list(examples.values()),indent=2))


def recheck_invariants(output):
    from unittest.mock import patch
    from budding_yeast_v2.data.systems.budding import simulation
    from budding_yeast_v2.data.systems.budding.api import simulate_mutant
    solve=simulation.solve_ivp
    def strict(*args,**kwargs):
        kwargs.update(rtol=1e-7,atol=1e-9,max_step=.25)
        return solve(*args,**kwargs)
    examples=json.loads((output/'invariant_examples.json').read_text())
    results=[]
    with patch.object(simulation,'solve_ivp',strict):
        for example in examples:
            result=simulate_mutant(example['name'],example['overrides'],210,500)
            y=result[1]
            peak=None if y is None else float(np.max(y[28]))
            results.append(dict(candidate_id=example['candidate_id'],name=example['name'],
                original_peak=example['peak'],total=example['total'],strict_peak=peak,
                strict_succeeded=y is not None,
                strict_violates_total=None if y is None else peak>example['total']+1e-3*max(1.,example['total'])))
    (output/'invariant_strict_rechecks.json').write_text(json.dumps(results,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/users/zsy/mydata/budding_yeast/experiments/al_budding/fno'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scan-trajectories', action='store_true')
    parser.add_argument('--strict-recheck-invariants',action='store_true',
                        help='Re-solve saved total-pool violations in this audit process only')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs, timestamp = discover(args.root)
    statuses, rounds, per_mutant, losses = [], [], [], []
    for info, records in runs:
        statuses.append({k:v for k,v in info.items() if k != 'args'})
        for record in records:
            common = dict(cohort=info['cohort'], strategy=info['strategy'], seed=info['seed'])
            rounds.append(dict(**common, **record))
            folder = Path(info['run']) / 'rounds' / f'round_{record["round"]:02d}'
            path = folder / 'eval/metrics_denorm.json'
            if path.exists():
                frame, obj = metric_frame(path)
                for name, row in frame.iterrows():
                    per_mutant.append(dict(**common, round=record['round'], train_samples=record['train_samples'],
                                           Mutant=name, **row.to_dict()))
            history = folder / 'training_history.csv'
            if history.exists():
                h = pd.read_csv(history)
                losses.append(dict(**common, round=record['round'], train_samples=record['train_samples'],
                    final_train_loss=float(h.train_loss.iloc[-1]), final_val_synth=float(h.val_synth.iloc[-1]),
                    best_val_synth=float(h.val_synth.min()),
                    val_first10_median=float(h.val_synth.head(10).median()),
                    val_last10_median=float(h.val_synth.tail(10).median())))
    for name, data in [('status', statuses), ('round_metrics', rounds), ('per_mutant', per_mutant), ('loss_diagnostics', losses)]:
        pd.DataFrame(data).to_csv(args.output / f'{name}.csv', index=False)
    (args.output / 'snapshot.json').write_text(json.dumps(dict(timestamp=timestamp, status=statuses), indent=2))
    frame = pd.DataFrame(per_mutant)
    reference = frame[(frame.cohort == 'balanced') & (frame.strategy == 'al4pde_random') & (frame['round'] == 3)].set_index('Mutant')
    rank = reference[['MAE','Relative_L2']].rank(pct=True).mean(1).sort_values(ascending=False)
    badcases = rank.head(20).index
    reference.loc[badcases].assign(badcase_score=rank.loc[badcases]).to_csv(args.output/'reference_top20.csv')
    comparisons, bad_rows = [], []
    for (cohort, strategy, index), local in frame.groupby(['cohort','strategy','round']):
        part = local.set_index('Mutant')
        matching = frame[(frame.cohort=='balanced') & (frame.strategy==strategy) & (frame['round']==index)].set_index('Mutant')
        if cohort=='gene_ranges' and len(matching):
            if set(matching.index) != set(part.index):
                raise ValueError('Different real benchmark identities')
            matching = matching.loc[part.index]
            for key in METRICS:
                differences = part[key] - matching[key]
                comparisons.append(dict(cohort=cohort, strategy=strategy, round=int(index), metric=key,
                    old_mean=float(matching[key].mean()), new_mean=float(part[key].mean()),
                    better=int(((differences>0) if key=='Correlation' else (differences<0)).sum()),
                    worse=int(((differences<0) if key=='Correlation' else (differences>0)).sum())))
        selected = part.loc[badcases]
        for name, row in selected.iterrows():
            bad_rows.append(dict(cohort=cohort,strategy=strategy,round=int(index),Mutant=name,
                **{key:float(row[key]) for key in METRICS},
                reference_MAE=float(reference.loc[name,'MAE']),
                reference_Relative_L2=float(reference.loc[name,'Relative_L2'])))
    pd.DataFrame(comparisons).to_csv(args.output/'paired_range_comparisons.csv',index=False)
    pd.DataFrame(bad_rows).to_csv(args.output/'fixed_top20_errors.csv',index=False)
    plot(pd.DataFrame(rounds), pd.DataFrame(bad_rows), args.output)
    if args.scan_trajectories:
        scan_domains(runs,args.output)
    if args.strict_recheck_invariants:
        recheck_invariants(args.output)
    print(pd.DataFrame(statuses)[['cohort','strategy','status','completed_rounds']].to_string(index=False),flush=True)


def plot(frame, badcases, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for (cohort,strategy), local in frame.groupby(['cohort','strategy']):
        if cohort!='gene_ranges' and not (cohort=='balanced' and strategy=='al4pde_random'):
            continue
        label=cohort+'/'+strategy.removeprefix('al4pde_')
        for ax,key in zip(axes,('mae','rel_l2','mse')):
            ax.plot(local.train_samples,local['real_mutants_'+key],marker='o',label=label)
            ax.set(xlabel='Synthetic training samples',ylabel='Real '+key)
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Same FNO training protocol; seed 42; unfinished curves stop at last evaluated round')
    for suffix in ('png','pdf'):fig.savefig(output/f'al_range_learning_curves.{suffix}',dpi=180)
    plt.close(fig)
    selected=badcases[(badcases.cohort=='gene_ranges') & (badcases['round']==3)]
    if len(selected):
        matrix=selected.pivot(index='Mutant',columns='strategy',values='Relative_L2')
        fig,ax=plt.subplots(figsize=(8,9),constrained_layout=True)
        heat=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='magma',vmin=0)
        ax.set_yticks(range(len(matrix)),matrix.index,fontsize=7)
        ax.set_xticks(range(len(matrix.columns)),[k.removeprefix('al4pde_') for k in matrix.columns],rotation=25)
        fig.colorbar(heat,ax=ax,label='Relative L2')
        ax.set_title('Fixed top 20 from old balanced Random; new completed final rounds')
        for suffix in ('png','pdf'):fig.savefig(output/f'fixed_top20_heatmap.{suffix}',dpi=180)
        plt.close(fig)


if __name__=='__main__':
    main()
