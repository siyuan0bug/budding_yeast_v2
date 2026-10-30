"""Audit saved AL learning curves without extending or retraining experiments.

Plateau statistics are descriptive, not a proof of stationarity. Validation
alone informs budget recommendations; real-test curves remain post-hoc plots.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def percent_drop(before,after):
    return float(100*(1-after/before)) if before>0 else None


def describe(frame):
    if frame.epochs.duplicated().any() or not np.all(np.diff(frame.epochs)>0):
        raise ValueError('Epoch rows must be unique and increasing')
    f=frame.set_index('epochs');n=int(f.index[-1])
    if len(f)<21:raise ValueError('Need at least 21 epochs for tail-window diagnostics')
    if not np.isfinite(f[['val_synth','lr']]).all().all():raise ValueError('Nonfinite validation/lr history')
    val=f.val_synth;best=int(val.idxmin());last=float(val.iloc[-1])
    prior=val.iloc[-20:-10];tail=val.iloc[-10:]
    result=dict(epochs=n,best_val_epoch=best,best_val_mse=float(val.min()),final_val_mse=last,
        final_vs_best_gap_pct=float(100*(last/val.min()-1)),
        val_last10_vs_previous10_drop_pct=percent_drop(float(prior.mean()),float(tail.mean())),
        val_last10_cv_pct=float(100*tail.std(ddof=1)/tail.mean()),
        val_last10_linear_drop_per10_pct=float(-100*10*np.polyfit(np.arange(10),tail.to_numpy(),1)[0]/tail.mean()),
        final_train_hybrid=float(f.train_loss.iloc[-1]),
        train_last10_drop_pct=percent_drop(float(f.train_loss.iloc[-11]),float(f.train_loss.iloc[-1])),
        val_last10_endpoint_drop_pct=percent_drop(float(val.iloc[-11]),last),
        val_last20_endpoint_drop_pct=percent_drop(float(val.iloc[-21]),last),
        lr_final=float(f.lr.iloc[-1]),lr_peak=float(f.lr.max()),
        lr_final_to_peak=float(f.lr.iloc[-1]/f.lr.max()))
    for epoch in (50,70,80,90,100):
        if epoch in f.index:
            for field in ('train_loss','val_synth','lr'):
                result[f'{field}_ep{epoch}']=float(f.loc[epoch,field])
    return result


def plot(runs,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['al4pde_random','al4pde_uncertainty','al4pde_power','al4pde_max_dist',
           'al4pde_lcmd','al4pde_bait','ours_response']
    for zoom in (False,True):
        fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
        for ax,strategy in zip(axes.flat,order):
            f=runs[strategy];s=describe(f);short=strategy.removeprefix('al4pde_')
            if zoom:
                local=f[f.epochs>=60];scale=float(f.loc[f.epochs==80,'val_synth'].iloc[0])
                ax.plot(local.epochs,local.val_synth/scale*100,color='seagreen',lw=1.8)
                ax.axvline(s['best_val_epoch'],color='gray',ls=':',lw=1)
                ax.set(xlabel='Epoch',ylabel='Val MSE / epoch-80 value (%)',xlim=(60,100))
                label=f"90 to 100: {s['val_last10_endpoint_drop_pct']:+.2f}% lower\nfinal vs best: +{s['final_vs_best_gap_pct']:.2f}%"
            else:
                ax.plot(f.epochs,f.train_loss,label='Train hybrid loss',color='royalblue')
                ax.plot(f.epochs,f.val_synth,label='Validation MSE',color='seagreen')
                ax.set(xlabel='Epoch',ylabel='Normalized objective',yscale='log')
                label=f"best val: epoch {s['best_val_epoch']}\nfinal val: {s['final_val_mse']:.5f}"
            ax.text(.97,.97,label,transform=ax.transAxes,ha='right',va='top',fontsize=8)
            ax.set_title(short);ax.grid(alpha=.2)
        ax=axes.flat[-1];f=runs['ours_response'];ax.plot(f.epochs,f.lr,color='darkorange')
        ax.set(title='Shared learning-rate schedule',xlabel='Epoch',ylabel='Learning rate',yscale='log');ax.grid(alpha=.2)
        if not zoom:axes.flat[0].legend(fontsize=7,loc='lower left')
        fig.suptitle('Final AL round: 70k training samples, FNO, seed 42\n'+(
            'Within-run tail progress; real/test data do not set stopping decisions' if zoom else
            'Train = MSE + 0.2 relative loss; validation = MSE. Absolute levels differ.'),fontsize=12)
        name='final_round_tail_validation' if zoom else 'final_round_convergence'
        for suffix in ('png','pdf'):fig.savefig(output/f'{name}.{suffix}',dpi=180)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/experiments/al_budding/fno'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[];final={};integrity=[]
    for run in sorted(a.root.glob('*/*_refit_gene_ranges/seed_42')):
        config=json.loads((run/'config.json').read_text())['args'];strategy=config['al_strategy']
        log=(run/'train.log').read_text(errors='replace')
        if 'Traceback (most recent call last)' in log:raise ValueError(f'Failed run: {run}')
        records=json.loads((run/'summary/rounds_summary.json').read_text())
        for folder in sorted(run.glob('rounds/round_*')):
            path=folder/'training_history.csv'
            if not path.exists():continue
            index=int(folder.name.rsplit('_',1)[-1]);frame=pd.read_csv(path)
            result=describe(frame);record=records[index]
            if len(frame)!=record['epochs']:raise ValueError('Partial or mismatched history')
            rows.append(dict(strategy=strategy,round=index,train_samples=record['train_samples'],
                batch_size=config['batch_size'],drop_last=True,
                optimizer_steps=(record['train_samples']//config['batch_size'])*len(frame),
                **result,source=str(path)))
            if index==3:
                final[strategy]=frame
                # Logs print 6 decimals; compare validation endpoint at that precision.
                matches=re.findall(r'\[Epoch\s+100/100\].*?val_synth=([^\s|]+)',log)
                # Epoch summaries sometimes pad the max epoch with zeros but 100 is unchanged.
                if not matches:raise ValueError('Missing final epoch in train.log')
                error=abs(float(matches[-1])-result['final_val_mse'])
                if error>6e-7:raise ValueError('Log/CSV final validation mismatch')
                integrity.append(dict(strategy=strategy,log_csv_final_val_abs_difference=error,
                                      checkpoint_exists=(folder/'model.ckpt').exists()))
    if len(final)!=7:raise ValueError(f'Expected 7 completed methods, got {len(final)}')
    frame=pd.DataFrame(rows);frame.to_csv(a.output/'all_rounds_convergence.csv',index=False)
    frame[frame['round']==3].to_csv(a.output/'final_round_convergence.csv',index=False)
    (a.output/'integrity.json').write_text(json.dumps(dict(status='passed',runs=integrity,
        scope='Descriptive curve audit; no optimization continuation or convergence theorem'),indent=2))
    plot(final,a.output)
    print(frame[frame['round']==3][['strategy','final_val_mse','best_val_epoch',
        'val_last10_endpoint_drop_pct','final_vs_best_gap_pct','lr_final']].to_string(index=False))


if __name__=='__main__':main()
