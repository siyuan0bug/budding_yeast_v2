"""Compare full-budget AL outcomes and diagnose accepted acquisition components.

All benchmark labels are post-hoc only. No models are trained, no ODEs queried,
and no production configuration is modified. Prior and current complete results
are combined by strategy; obsolete event implementations are excluded.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor


def read(path):return json.loads(path.read_text())


def accepted_components(run,strategy):
    budgets=[];tails=[];feedback=[]
    for directory in sorted((run/'acquisitions').glob('after_round_*')):
        path=directory/'query_design.json'
        if not path.exists():continue
        acquisition=int(directory.name.rsplit('_',1)[1]);seen=set();counts=Counter();tail=Counter()
        for batch in read(path)['batches']:
            feedback.append(dict(strategy=strategy,acquisition=acquisition,**batch.get('feedback',{})))
            for choice in batch['choices']:
                for endpoint in choice['endpoints']:
                    if endpoint['kind']!='pool' or endpoint['status']!='accepted':continue
                    cid=int(endpoint['candidate_id'])
                    if cid in seen:continue
                    seen.add(cid);counts[choice['kind']]+=1
                    if choice['kind']=='tail':tail[(choice['variable'],choice['amplitude_band'])]+=1
        report=read(directory/'report.json')
        if len(seen)!=report['accepted']:raise ValueError(f'Accepted-query accounting mismatch: {directory}')
        budgets.extend(dict(strategy=strategy,acquisition=acquisition,kind=key,
            accepted=value,fraction=value/len(seen)) for key,value in counts.items())
        tails.extend(dict(strategy=strategy,acquisition=acquisition,variable=variable,band=band,count=count)
                     for (variable,band),count in tail.items())
    return budgets,tails,feedback


def temporal_edges(run):
    """Prior-model calibration evaluated on the next selected synthetic edges.

    Fixed diagnostic seed, not a bitwise reproduction of acquisition ranking.
    Selected-edge evaluation is conditional, not an unbiased pool estimate.
    """
    result=[];previous=None
    for path in sorted(run.glob('acquisitions/after_round_*/edge_feedback.npz')):
        with np.load(path) as z:
            current={k:z[k] for k in ('features','log_variance','observed_error')}
            acquisition=int(z['acquisition'])
        x=current['features'];u=current['log_variance'];e=current['observed_error'];eps=1e-8
        row=dict(acquisition=acquisition,edges=len(x),raw_u_spearman=float(spearmanr(u,e).statistic),
                 raw_mse=float(np.mean((u-e)**2)),observed_mean=float(e.mean()))
        if previous is not None:
            target=np.clip(np.log(previous['observed_error']+eps)-np.log(previous['log_variance']+eps),-8,8)
            tree=ExtraTreesRegressor(n_estimators=32,max_depth=12,min_samples_leaf=8,n_jobs=2,random_state=42+1009*acquisition)
            tree.fit(previous['features'],target);correction=tree.predict(x)
            predicted=np.exp(np.clip(correction,-8,8))*(u+eps)
            actual_correction=np.log(e+eps)-np.log(u+eps)
            row.update(corrected_u_spearman=float(spearmanr(predicted,e).statistic),
                correction_spearman=float(spearmanr(correction,actual_correction).statistic),
                corrected_mse=float(np.mean((predicted-e)**2)),predicted_mean=float(predicted.mean()))
        result.append(row);previous=current
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,action='append',required=True,help='Old to new priority')
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--top20',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    runs={};pending=[]
    for path in args.manifest:
        for task in read(path)['tasks']:
            strategy=task['strategy'];run=Path(task['experiment_dir']);summary=run/'summary/rounds_summary.json'
            records=read(summary) if summary.exists() else []
            log=(run/'train.log').read_text(errors='replace')
            if len(records)!=4 or records[-1]['train_samples']!=70000 or 'exit=0 log=' not in log:
                pending.append(dict(strategy=strategy,completed_rounds=len(records),run=str(run)));continue
            if strategy=='ours_event' and 'eventfix' not in str(run):continue
            runs[strategy]=run
    with np.load(args.initial,allow_pickle=False) as z:
        mask=z['is_real'];names=z['mutant_names'][mask].astype(str)
        groups=dict(zip(names,z['pattern_group_codes'][mask].astype(str)))
    top20=set(pd.read_csv(args.top20).Mutant)
    if len(top20)!=20:raise ValueError('Expected the fixed prior Random top20')
    all_rounds=[];samples=[];finals=[];budgets=[];tails=[];feedback=[];edge=[]
    for strategy,run in runs.items():
        config=read(run/'config.json')['args']
        for k,value in dict(model='fno',width=192,n_blocks=4,modes=24,batch_size=512,
                            al_round_epochs=100,al_final_epochs=200,min_lr=1e-5,seed=42).items():
            if config[k]!=value:raise ValueError(f'Incompatible {k}: {run}')
        records=read(run/'summary/rounds_summary.json')
        for record in records:
            all_rounds.append(dict(strategy=strategy,**record))
            metrics=read(run/'rounds'/f'round_{record["round"]:02d}'/'eval/metrics_denorm.json')
            for row in metrics['per_mutant']:
                if row['Mutant'] not in groups:raise ValueError('Real benchmark identity changed')
                samples.append(dict(strategy=strategy,round=record['round'],group=groups[row['Mutant']],**row))
        final=dict(strategy=strategy,run=str(run),**records[-1])
        x=np.array([r['train_samples'] for r in records])
        for metric in ('mae','mse','rel_l2'):
            final['aulc_'+metric]=float(trapezoid([r['real_mutants_'+metric] for r in records],x)/(x[-1]-x[0]))
        finals.append(final)
        if strategy.startswith('ours_global_'):
            b,t,f=accepted_components(run,strategy);budgets+=b;tails+=t;feedback+=f
        if strategy=='ours_global_calibrated':edge=temporal_edges(run)
    frame=pd.DataFrame(samples);final=frame[frame['round']==3];summary=pd.DataFrame(finals)
    reference=final[final.strategy=='al4pde_power'].set_index('Mutant');paired=[];group_rows=[]
    if len(reference)!=126:raise ValueError('Complete Power reference required')
    for strategy,local in final.groupby('strategy'):
        local=local.set_index('Mutant').loc[reference.index]
        summary.loc[summary.strategy==strategy,'mean_per_mutant_rel']=local.Relative_L2.mean()
        summary.loc[summary.strategy==strategy,'median_per_mutant_rel']=local.Relative_L2.median()
        summary.loc[summary.strategy==strategy,'q90_per_mutant_rel']=local.Relative_L2.quantile(.9)
        summary.loc[summary.strategy==strategy,'max_per_mutant_rel']=local.Relative_L2.max()
        summary.loc[summary.strategy==strategy,'rel_above1']=int((local.Relative_L2>1).sum())
        for metric in ('MAE','MSE','Relative_L2','Correlation'):
            difference=local[metric]-reference[metric];better=difference>0 if metric=='Correlation' else difference<0
            paired.append(dict(strategy=strategy,metric=metric,better=int(better.sum()),worse=int((difference!=0).sum()-better.sum()),
                all_mean=float(local[metric].mean()),top20_mean=float(local.loc[list(top20),metric].mean()),
                top20_better=int(better.loc[list(top20)].sum())))
        for group,rows in local.groupby('group'):
            group_rows.append(dict(strategy=strategy,group=group,count=len(rows),mean_mae=rows.MAE.mean(),
                mean_mse=rows.MSE.mean(),mean_rel=rows.Relative_L2.mean(),median_rel=rows.Relative_L2.median()))
    order=summary.sort_values('real_mutants_rel_l2').strategy
    final[final.Mutant.isin(top20)].to_csv(args.output/'fixed_top20_per_mutant.csv',index=False)
    selected=final.copy();selected['mutant_id']=selected.Mutant.str.split('_').str[0]
    selected[selected.mutant_id.isin(['15','20','22','42','50','62','124'])].to_csv(args.output/'mechanism_cases.csv',index=False)
    for name,df in [('summary',summary),('rounds',pd.DataFrame(all_rounds)),('per_mutant',frame),
        ('paired_vs_power',pd.DataFrame(paired)),('real_group_metrics',pd.DataFrame(group_rows)),
        ('accepted_components',pd.DataFrame(budgets)),('tail_strata',pd.DataFrame(tails)),
        ('edge_temporal_calibration',pd.DataFrame(edge))]:df.to_csv(args.output/f'{name}.csv',index=False)
    (args.output/'acquisition_feedback.json').write_text(json.dumps(feedback,indent=2))
    (args.output/'status.json').write_text(json.dumps(dict(checked=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        complete={k:str(v) for k,v in runs.items()},excluded_or_pending=pending,
        note='seed42; official global relative L2 differs from mean per-mutant Relative_L2'),indent=2))
    for strategy in ['ours_global_tail','ours_global_response','ours_global_interaction','ours_global_calibrated']:
        local=final[final.strategy==strategy].set_index('Mutant').loc[reference.index]
        out=pd.DataFrame(dict(Mutant=reference.index,delta_mae=local.MAE-reference.MAE,delta_mse=local.MSE-reference.MSE,
                              delta_rel=local.Relative_L2-reference.Relative_L2))
        out.sort_values('delta_mse',ascending=False).to_csv(args.output/f'{strategy}_vs_power_cases.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    focus=['al4pde_random','al4pde_max_dist','al4pde_power','al4pde_lcmd','al4pde_uncertainty',
           'ours_global_response','ours_global_tail','ours_global_calibrated']
    rounds=pd.DataFrame(all_rounds)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for strategy in focus:
        local=rounds[rounds.strategy==strategy].sort_values('train_samples')
        for ax,key in zip(axes,('mae','rel_l2')):
            ax.plot(local.train_samples/1000,local['real_mutants_'+key],marker='o',label=strategy.removeprefix('al4pde_').removeprefix('ours_'))
    for ax,key in zip(axes,('MAE','Relative L2')):ax.set(xlabel='Training samples (thousands)',ylabel=key);ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.savefig(args.output/'learning_curves.png',dpi=170);plt.close(fig)
    table=final[final.Mutant.isin(top20)].pivot(index='Mutant',columns='strategy',values='Relative_L2')
    names=reference.loc[list(top20)].sort_values('Relative_L2',ascending=False).index
    table=table.loc[names,focus]
    fig,ax=plt.subplots(figsize=(11,8),constrained_layout=True)
    im=ax.imshow(np.log10(np.maximum(table.values,1e-3)),aspect='auto',cmap='magma',vmin=-2,vmax=1)
    ax.set_yticks(range(20),[name.split('_',1)[0] for name in names]);ax.set_xticks(range(len(focus)),
        [s.removeprefix('al4pde_').removeprefix('ours_') for s in focus],rotation=35,ha='right')
    ax.set_ylabel('Prior Random top20 mutant ID');fig.colorbar(im,ax=ax,label='log10 per-mutant Relative L2')
    fig.savefig(args.output/'badcase_comparison.png',dpi=170);plt.close(fig)
    print(summary[['strategy','real_mutants_mae','real_mutants_mse','real_mutants_rel_l2','real_mutants_corr','aulc_mae']].sort_values('real_mutants_rel_l2').to_string(index=False))


if __name__=='__main__':main()
