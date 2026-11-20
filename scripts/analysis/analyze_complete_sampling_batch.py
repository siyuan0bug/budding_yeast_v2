#!/usr/bin/env python3
"""Audit completed offline sampling runs using saved reports; optional CPU inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def analyze(root, output, previous=None):
    output.mkdir(parents=True, exist_ok=True)
    tasks = read(root/'batch/tasks.json')
    timing = {(t['pool'], t['method']): t['seconds'] for t in tasks}
    final, rounds, convergence, badcases, comparisons, issues = [], [], [], [], [], []
    for directory in sorted(root.glob('*/*/seed_42')):
        pool, method = directory.parts[-3:-1]
        state = read(directory/'_state/state.json')
        completion = state['completion']
        if completion['status'] != 'complete' or completion['unique_training_inputs'] != 10000:
            issues.append(str(directory)+': incomplete/budget mismatch')
        if 'Traceback (most recent call last)' in (directory/'train.log').read_text():
            issues.append(str(directory)+': traceback')
        report = read(directory/'eval/global_metrics_denorm.json')
        row = dict(pool=pool, method=method, seconds=timing[(pool, method)], **{
            'unique_labels_accessed': completion['unique_labels_accessed']})
        for domain in ('real_mutants', 'synth_test', 'cross_pool_test'):
            values = report[domain]
            assert values['metric_version'] == 'physical_sample_channel_rel_l2_smooth_eps001_v3'
            row.update({domain+'_'+k: v for k, v in values['all_39_vars'].items()})
            matrix = np.asarray(values['pattern_metrics']['six_group_confusion'])
            row[domain+'_pattern_accuracy'] = float(matrix.trace()/matrix.sum())
        samples = pd.read_csv(directory/'eval/per_mutant_metrics.csv')
        assert len(samples) == 126 and np.isfinite(samples[['MAE','MSE','rel_l2']]).all().all()
        assert np.isclose(samples.rel_l2.mean(),row['real_mutants_Relative L2'])
        for metric in ('MAE','MSE','rel_l2'):
            row[f'real_{metric}_median'] = samples[metric].median()
            row[f'real_{metric}_p90'] = samples[metric].quantile(.9)
        row['rel_worst13_share'] = samples.nlargest(13,'rel_l2').rel_l2.sum()/samples.rel_l2.sum()
        # Rank aggregation is for diagnosis, not a training/selection signal.
        samples['bad_rank'] = (samples.MAE.rank(pct=True)+samples.rel_l2.rank(pct=True))/2
        for _, s in samples.iterrows():
            badcases.append(dict(pool=pool,method=method,**s.to_dict()))
        for i, budget in enumerate((2000,4000,6000,8000,10000)):
            rd = directory/f'rounds/round_{i:02d}'
            metadata = read(directory/f'_state/round_{i:02d}/primary/training.json')
            history = pd.read_csv(rd/'history.csv')
            dataset = read(rd/'dataset.json')
            assert len(dataset['rows']) == len(set(dataset['rows'])) == budget
            assert len(history) == (200 if i == 4 else 100)
            assert metadata['validation_metric'] == 'val_loss_synth'
            assert metadata['best_epoch'] == int(history.loc[history.val_loss_synth.idxmin(),'epoch'])
            assert (rd/'model.ckpt').exists() and Path(metadata['last_epoch_checkpoint']['path']).exists()
            assert len(list((rd/'eval/real_mutants').glob('*.png'))) == 126
            ev = read(rd/'eval/global_metrics_denorm.json')
            rounds.append(dict(pool=pool,method=method,round=i,training_rows=budget,
                best_epoch=metadata['best_epoch'],**ev['real_mutants']['all_39_vars']))
            if i == 4:
                for field in ('train_loss','val_loss_synth'):
                    values = history[field].to_numpy()[-20:]
                    convergence.append(dict(pool=pool,method=method,field=field,
                        first10_mean=float(values[:10].mean()),last10_mean=float(values[10:].mean()),
                        change_percent=float((values[10:].mean()/values[:10].mean()-1)*100),
                        best_epoch=metadata['best_epoch'],last_epoch=200))
                row['best_epoch'] = metadata['best_epoch']
                row['final_val_normalized'] = history.val_loss_synth.min()
        if previous is not None:
            oldpath = previous/pool/method/'seed_42/eval/global_metrics_denorm.json'
            if oldpath.exists():
                old = read(oldpath)['real_mutants']
                assert old['metric_version'] == report['real_mutants']['metric_version']
                for metric, new in report['real_mutants']['all_39_vars'].items():
                    before = old['all_39_vars'][metric]
                    comparisons.append(dict(pool=pool,method=method,metric=metric,
                        old=before,new=new,change_percent=100*(new/before-1)))
        final.append(row)
    assert len(final) == len(tasks) == 16 and all(t['exit_code']==0 for t in tasks)
    assert not issues, issues
    save_csv(output/'final_metrics.csv',final)
    save_csv(output/'round_metrics.csv',rounds)
    save_csv(output/'convergence.csv',convergence)
    save_csv(output/'per_mutant_all_methods.csv',badcases)
    save_csv(output/'previous_comparison.csv',comparisons)
    bad = pd.DataFrame(badcases)
    overall = bad[bad.pool=='mutation'].groupby('mutant_name').agg(
        mean_bad_rank=('bad_rank','mean'),mean_MAE=('MAE','mean'),mean_rel_l2=('rel_l2','mean'))
    overall.sort_values('mean_bad_rank',ascending=False).head(20).to_csv(output/'top20_badcases.csv')
    for pool in ('mutation','lhs'):
        df = pd.DataFrame(final).query('pool == @pool').set_index('method')
        baseline = df.loc['random']
        save_csv(output/f'{pool}_vs_random.csv',[
            dict(method=m,**{k:100*(r[k]/baseline[k]-1) for k in
                ('real_mutants_MAE','real_mutants_MSE','real_mutants_Relative L2')})
            for m,r in df.iterrows()])
    return pd.DataFrame(final), pd.DataFrame(rounds)


def reinfer(root, output, names):
    """Only 126 held-out real rows, CPU; never supplied to a selector."""
    import torch
    from budding_yeast_v2.utils.evaluation.checkpoints import load_sampling_checkpoint, Predictor
    from budding_yeast_v2.utils.sampling.contracts import PoolInputs
    from budding_yeast_v2.utils.evaluation.metrics import sample_channel_relative_errors
    from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
    torch.set_num_threads(2)
    conf = read(root/'mutation/random/seed_42/config.json')
    with np.load(conf['real_reference']['path'],allow_pickle=True) as z:
        truth,params,initial,mutants = (z[k] for k in ('data','params','y_init','mutant_names'))
    inputs = PoolInputs(np.arange(len(truth)),params,initial)
    rows = []
    for key in names:
        pool, method = key.split('/')
        directory = root/pool/method/'seed_42'
        model,norm,_ = load_sampling_checkpoint(directory/'rounds/round_04/model.ckpt','cpu')
        pred = Predictor([model],norm,truth.shape[-1],'cpu',batch_size=16).physical(inputs)
        errors = sample_channel_relative_errors(pred,truth).numpy()
        stored = read(directory/'eval/global_metrics_denorm.json')['real_mutants']['sample_channel_rel_l2']
        # CPU/GPU FFT rounding is amplified by near-zero physical denominators.
        # Saved GPU metrics stay authoritative; this read-only audit allows 0.1%.
        assert np.isclose(errors.mean(),stored,rtol=1e-3,atol=1e-5),(key,errors.mean(),stored)
        energy = np.linalg.norm(truth.astype(np.float64),axis=-1)
        for j,var in enumerate(VARIABLE_NAMES):
            rows.append(dict(pool=pool,method=method,variable=var,
                mean_relative_l2=float(errors[:,j].mean()),
                share_of_relative_sum=float(errors[:,j].sum()/errors.sum()),
                near_zero_targets=int((energy[:,j]<.01).sum()),
                MAE=float(np.abs(pred[:,j]-truth[:,j]).mean()),
                fitted_std=float(norm.std_y.reshape(-1)[j])))
        for idx in (next(i for i,x in enumerate(mutants) if str(x).startswith('15_')),
                    next(i for i,x in enumerate(mutants) if str(x).startswith('37_'))):
            save_csv(output/f'{pool}_{method}_mutant_{str(mutants[idx]).split("_")[0]}_channels.csv',[
                dict(variable=var,relative_l2=errors[idx,j],truth_norm=energy[idx,j],
                    truth_min=float(truth[idx,j].min()),truth_max=float(truth[idx,j].max()),
                    prediction_min=float(pred[idx,j].min()),prediction_max=float(pred[idx,j].max()))
                for j,var in enumerate(VARIABLE_NAMES)])
        print('CPU_RECHECK',key,'cpu=',errors.mean(),'saved_gpu=',stored,flush=True)
    save_csv(output/'channel_diagnosis.csv',rows)


def plots(output, final, rounds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for col,pool in enumerate(('mutation','lhs')):
        for row,metric in enumerate(('MAE','Relative L2')):
            ax=axes[row,col]
            for method, group in rounds[rounds.pool==pool].groupby('method'):
                ax.plot(group.training_rows,group[metric],marker='o',markersize=3,label=method)
            ax.set(title=f'{pool}: real mutant {metric}',xlabel='Queried training inputs',ylabel=metric)
            ax.grid(alpha=.25)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
    fig.savefig(output/'round_comparison.png',dpi=180)
    plt.close(fig)


def initial_coverage(root, output, cache):
    """Read input metadata only; no 100k-trajectory loading or new simulation."""
    with np.load(cache,allow_pickle=True) as z:
        initial=z['y_init']
    rows=[]
    for path in sorted(root.glob('mutation/*/seed_42/rounds/round_*/dataset.json')):
        ids=np.asarray(read(path)['rows'],dtype=int)
        rows.append(dict(method=path.parts[-5],round=path.parts[-2],training_rows=len(ids),
            CDH1T_zero_initial=int((np.abs(initial[ids,21])<1e-12).sum()),
            fraction=float((np.abs(initial[ids,21])<1e-12).mean()),
            pool_fraction=float((np.abs(initial[:,21])<1e-12).mean())))
    save_csv(output/'cdh1_zero_initial_coverage.csv',rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path)
    p.add_argument('--reinfer',nargs='*',default=[])
    p.add_argument('--mutation-input-cache',type=Path)
    args=p.parse_args()
    final,rounds=analyze(args.root,args.output,args.previous)
    plots(args.output,final,rounds)
    if args.mutation_input_cache:initial_coverage(args.root,args.output,args.mutation_input_cache)
    if args.reinfer:reinfer(args.root,args.output,args.reinfer)
    print(final[['pool','method','real_mutants_MAE','real_mutants_MSE','real_mutants_Relative L2']].to_string(index=False))


if __name__=='__main__':
    main()
