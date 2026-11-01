#!/usr/bin/env python3
"""Reproducible archived-model comparison and frozen marker-closure diagnosis.

No training, no alteration of archived results. Old baselines retain their old
names. AL rounds, datasets, losses and normalization are kept as separate rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
METRICS = ('MAE','MSE','Relative_L2')
COHORT = 'mutant_aware_pure_random_100000_210m_500s'
REFERENCE = '02482ab483cf'


def mutant_id(name):
    match=re.match(r'^(\d+(?:\.\d+)?)_',str(name))
    return match.group(1) if match else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root,output):
    csv_files=sorted(root.rglob('per_mutant_metrics.csv'))
    al_files=sorted((root/'al_budding').rglob('metrics_denorm.json'))
    runs=[];frames=[];skipped=[]
    for source in csv_files+al_files:
        kind='al' if source.name=='metrics_denorm.json' else 'csv'
        run=next((p for p in source.parents if (p/'config.json').exists()),None)
        if run is None:
            skipped.append(dict(source=str(source),reason='no config'));continue
        config=json.loads((run/'config.json').read_text());a=config.get('args',config)
        if a.get('system_id','budding')!='budding' or 'fission' in source.relative_to(root).parts[0]:
            skipped.append(dict(source=str(source),reason='fission'));continue
        payload=json.loads(source.read_text()) if kind=='al' else None
        frame=pd.DataFrame(payload['per_mutant']) if payload is not None else pd.read_csv(source)
        if 'Mutant' not in frame or not all(k in frame for k in METRICS):
            skipped.append(dict(source=str(source),reason='missing columns'));continue
        frame['mutant_id']=frame.Mutant.map(mutant_id)
        if frame.mutant_id.isna().any() or frame.mutant_id.duplicated().any() or not np.isfinite(frame[list(METRICS)]).all().all():
            skipped.append(dict(source=str(source),reason='invalid identity/metrics'));continue
        run_id=hashlib.sha256(str(source).encode()).hexdigest()[:12]
        relative=source.relative_to(root)
        pooled={}
        global_file=source.with_name('global_metrics_denorm.json')
        if global_file.exists():
            global_data=json.loads(global_file.read_text()).get('real_mutants',{})
            pooled=global_data.get('all_39_vars',global_data.get('all_variables',{}))
        record=dict(run_id=run_id,source=str(source),sha256=digest(source),kind=kind,
            dataset=relative.parts[0],model=a.get('model','unknown'),loss=a.get('loss_type','unknown'),
            norm=a.get('norm_type','log1p'),seed=a.get('seed',42),width=a.get('width'),
            n_blocks=a.get('n_blocks'),epochs=a.get('max_epochs'),batch_size=a.get('batch_size'),
            lr=a.get('lr'),strategy=a.get('al_strategy','none'),n_real=len(frame),
            round=int(source.parent.parent.name.split('_')[-1]) if kind=='al' else -1,
            historical_copy=any('旧版本' in part for part in relative.parts),
            pooled_rel=pooled.get('Relative L2'),pooled_corr=pooled.get('Correlation'))
        for metric in METRICS:
            record['mean_'+metric]=frame[metric].mean()
            record['median_'+metric]=frame[metric].median()
            record['p90_'+metric]=frame[metric].quantile(.9)
            record['worst20_'+metric]=frame[metric].nlargest(20).mean()
            frame['rank_'+metric]=frame[metric].rank(ascending=False,method='average')
        record['mean_corr']=frame.Correlation.mean() if 'Correlation' in frame else np.nan
        frame['severity_rank']=frame[['rank_'+m for m in METRICS]].mean(axis=1)
        frame['run_id']=run_id
        runs.append(record);frames.append(frame)
    runs=pd.DataFrame(runs);metrics=pd.concat(frames,ignore_index=True)
    runs.to_csv(output/'runs.csv',index=False)
    metrics.to_csv(output/'per_mutant_all_runs.csv',index=False)
    (output/'skipped.json').write_text(json.dumps(skipped,ensure_ascii=False,indent=2))
    return runs,metrics


def compare(runs,metrics,output):
    cohort=runs[(runs.dataset==COHORT)&(runs.kind=='csv')&(~runs.historical_copy)&(runs.n_real==126)].copy()
    cohort=cohort.sort_values('mean_MAE')
    cohort.to_csv(output/'historical_cohort.csv',index=False)
    ref=cohort[(cohort.model=='biom_refiner_plus')&(cohort.loss=='hybrid_mse_rel')&(cohort.norm=='log1p')]
    if len(ref)!=1:raise ValueError('Reference configuration is ambiguous')
    reference=metrics[metrics.run_id==ref.iloc[0].run_id].copy()
    selected=reference.sort_values(['severity_rank','mutant_id']).head(20)
    selected.to_csv(output/'top20_reference.csv',index=False)
    for metric in METRICS:
        reference.nlargest(20,metric).to_csv(output/f'top20_{metric}.csv',index=False)
    # Identical archived metric files are not independent evidence (old aliases).
    cohort[cohort.sha256.duplicated(False)].to_csv(output/'duplicate_cohort_metrics.csv',index=False)
    keys=cohort.drop_duplicates(['model','loss','norm','seed']).drop_duplicates('sha256').run_id
    consensus=metrics[metrics.run_id.isin(keys)].groupby('mutant_id').agg(
        mean_severity_rank=('severity_rank','mean'),n_runs=('run_id','nunique'),
        median_rel=('Relative_L2','median'),maximum_rel=('Relative_L2','max')).sort_values('mean_severity_rank')
    consensus.to_csv(output/'consensus_badcases.csv')
    paired=[]
    for _,row in cohort.iterrows():
        other=metrics[metrics.run_id==row.run_id].set_index('mutant_id')
        joined=reference.set_index('mutant_id').join(other[list(METRICS)],rsuffix='_other',validate='one_to_one')
        r=dict(run_id=row.run_id,model=row.model,loss=row.loss,norm=row.norm)
        for metric in METRICS:
            difference=joined[metric+'_other']-joined[metric]
            r[metric+'_wins_against_reference']=int((difference < -1e-9).sum())
            r[metric+'_mean_difference']=difference.mean()
        paired.append(r)
    pd.DataFrame(paired).to_csv(output/'paired_against_reference.csv',index=False)
    return cohort,reference,selected


def frozen_diagnosis(audit,reference,selected,output):
    import torch
    from budding_yeast_v2.models.ours.bio_refiner_marker_closure import marker_closure
    from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
    replay=audit/'replay'/REFERENCE
    provenance=json.loads((replay/'provenance.json').read_text())
    if not provenance.get('matched_archived_metrics_within_tolerance'):
        raise ValueError('Frozen replay has not matched its archived reference')
    runs=pd.read_csv(audit/'primary_runs.csv')
    reference_run=runs[runs.run_id==REFERENCE].iloc[0]
    manifest=json.loads((audit/'feature_manifest.json').read_text())
    feature_file=Path(manifest[reference_run.dataset])
    with np.load(replay/'predictions.npz') as z:
        prediction=z['predictions'].astype('float64');truth=z['targets'].astype('float64');names=z['names'].astype(str)
    with np.load(feature_file) as z:
        from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
        if list(z['param_names'].astype(str)) != list(PARAM_KEYS):
            raise ValueError('Frozen cache parameter order differs from canonical PARAM_KEYS')
        real=np.flatnonzero(z['is_real']);cached_names=z['names'][real].astype(str)
        locations={name:i for i,name in enumerate(cached_names)}
        positions=np.array([locations[n] for n in names])
        parameters=z['params'][real][positions].astype('float64')
        initial=z['ic'][real][positions].astype('float64')
    times=np.broadcast_to(np.linspace(0,210,truth.shape[-1]),(len(truth),truth.shape[-1])).copy()
    ids=[mutant_id(n) for n in names]
    closure=marker_closure(torch.tensor(prediction),torch.tensor(parameters),torch.tensor(initial),torch.tensor(times)).numpy()
    true_closure=marker_closure(torch.tensor(truth),torch.tensor(parameters),torch.tensor(initial),torch.tensor(times)).numpy()
    # Factorial oracle: truth drivers / predicted guard schedule and vice versa.
    def schedule(y):
        from budding_yeast_v2.models.ours.bio_refiner_marker_closure import downward_crossings
        y=torch.tensor(y)
        m1,f1=downward_crossings(y[:,2],.3);m2,f2=downward_crossings(y[:,2]+y[:,3],.2)
        return torch.stack((m2,m1,m1),1),torch.stack((f2,f1,f1),1)
    true_schedule=schedule(truth);pred_schedule=schedule(prediction)
    def diagnose(drivers,s):
        return marker_closure(torch.tensor(drivers),torch.tensor(parameters),torch.tensor(initial),torch.tensor(times),reset_override=s).numpy()
    oracle_schedule=diagnose(prediction,true_schedule)
    oracle_drivers=diagnose(truth,pred_schedule)
    def metrics_np(pred,target):
        difference=pred-target
        return dict(MAE=float(np.mean(abs(difference))),MSE=float(np.mean(difference**2)),
            Relative_L2=float(np.mean(np.sqrt(np.mean(difference**2,axis=-1)/np.maximum(np.mean(target**2,axis=-1),.01)))))
    closed=prediction.copy();closed[:,33:36]=closure
    readout_rows=[]
    for i,mid in enumerate(ids):
        before=metrics_np(prediction[i],truth[i]);after=metrics_np(closed[i],truth[i])
        readout_rows.append(dict(mutant_id=mid,**{k+'_before':v for k,v in before.items()},**{k+'_after':v for k,v in after.items()}))
    readout_frame=pd.DataFrame(readout_rows)
    readout_frame.to_csv(output/'frozen_full_metrics_before_after.csv',index=False)
    comparison=reference.set_index('mutant_id').join(readout_frame.set_index('mutant_id'),validate='one_to_one')
    cache_difference={m:float((comparison[m]-comparison[m+'_before']).abs().max()) for m in METRICS}
    for metric in METRICS:
        if cache_difference[metric] > provenance['tolerances'][metric]:
            raise ValueError(f'Frozen replay no longer matches current archived {metric}')
    rows=[];channels=[]
    for i,name in enumerate(names):
        delta=prediction[i]-truth[i];sse=(delta**2).sum(axis=1)
        record=dict(mutant_id=ids[i],Mutant=name,
            dominant_sse_variable=VARIABLE_NAMES[sse.argmax()],dominant_sse_share=sse.max()/max(sse.sum(),1e-30),
            tail10_sse_share=(delta[:,-50:]**2).sum()/max(sse.sum(),1e-30),
            truth_CLN2_peak=truth[i,1].max(),prediction_CLN2_peak=prediction[i,1].max(),
            truth_ORI_peak=truth[i,33].max(),prediction_ORI_peak=prediction[i,33].max())
        for j,channel in enumerate((33,34,35)):
            target=truth[i,channel]
            for label,value in (('original',prediction[i,channel]),('closure',closure[i,j]),
                                ('truth_drivers_truth_schedule',true_closure[i,j]),
                                ('pred_drivers_truth_schedule',oracle_schedule[i,j]),
                                ('truth_drivers_pred_schedule',oracle_drivers[i,j])):
                record[f'{VARIABLE_NAMES[channel]}_{label}_MSE']=np.mean((value-target)**2)
        for j,v in enumerate(VARIABLE_NAMES):
            channels.append(dict(mutant_id=ids[i],variable=v,MSE=np.mean(delta[j]**2),
                MAE=np.mean(abs(delta[j])),sse_share=sse[j]/max(sse.sum(),1e-30)))
        rows.append(record)
    diagnostics=pd.DataFrame(rows)
    diagnostics.to_csv(output/'frozen_closure_diagnosis.csv',index=False)
    pd.DataFrame(channels).to_csv(output/'frozen_channel_errors.csv',index=False)
    selected.merge(diagnostics,on=['mutant_id','Mutant'],validate='one_to_one').to_csv(output/'top20_evidence.csv',index=False)
    np.savez_compressed(output/'diagnostic_marker_trajectories.npz',closure=closure,truth_closure=true_closure,names=names)
    (output/'frozen_provenance.json').write_text(json.dumps(dict(
        upstream=provenance,cache=str(replay/'predictions.npz'),sha256=digest(replay/'predictions.npz'),
        current_archive_max_abs_differences=cache_difference,
        feature_cache=str(feature_file),description='Frozen readout intervention, not retraining; oracle variants never used by the model.'),indent=2))
    return diagnostics,(prediction,truth,closure,names)


def plots(cohort,metrics,selected,arrays,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wanted=['fno1d','bregman_wno','mwt1d','gnot','transolver1d','unet1d','deeponet','konode','lamo','neural_ode','biom_refiner_plus','biom_refiner_event_hybrid','biom_refiner_causal_reach','biom_refiner_attractor_plus']
    subset=cohort[cohort.model.isin(wanted)&(cohort.norm=='log1p')]
    labels=[];columns=[]
    for _,row in subset.iterrows():
        labels.append(row.model+' / '+row.loss)
        m=metrics[metrics.run_id==row.run_id].set_index('mutant_id')
        columns.append(m.loc[selected.mutant_id,'Relative_L2'].values)
    fig,ax=plt.subplots(figsize=(12,9))
    values=np.stack(columns,1)
    im=ax.imshow(np.log10(values.clip(1e-4)),aspect='auto',cmap='magma')
    ax.set_xticks(range(len(labels)),labels,rotation=65,ha='right',fontsize=8)
    ax.set_yticks(range(20),['#'+s for s in selected.mutant_id])
    ax.set_title('Archived per-mutant relative L2; same dataset, different historical implementations')
    fig.colorbar(im,ax=ax,label='log10 relative L2')
    fig.tight_layout();fig.savefig(output/'badcase_model_heatmap.png',dpi=170);plt.close(fig)
    prediction,truth,closure,names=arrays
    locations={mutant_id(n):i for i,n in enumerate(names)};t=np.linspace(0,210,500)
    (output/'cases').mkdir(exist_ok=True)
    for mid in selected.mutant_id:
        i=locations[mid];fig,axes=plt.subplots(2,3,figsize=(13,6))
        for ax,v in zip(axes.flat,(0,1,2,3,33,35)):
            ax.plot(t,truth[i,v],color='black',lw=1.4,label='ODE benchmark')
            ax.plot(t,prediction[i,v],color='#c84d49',lw=1,label='Frozen BioRefinerPlus')
            if v in (33,35):
                ax.plot(t,closure[i,{33:0,35:2}[v]],color='#287eab',lw=1,label='Marker closure (no retraining)')
            ax.set_title({0:'MASS',1:'CLN2',2:'CLB2',3:'CLB5',33:'ORI',35:'SPN'}[v]);ax.set_xlabel('min')
        axes[0,0].legend(fontsize=7)
        fig.suptitle(names[i],fontsize=10);fig.tight_layout()
        fig.savefig(output/'cases'/f'case_{mid}.png',dpi=150);plt.close(fig)


def probe_hidden_intervention(output):
    """Demonstrate the input contract limitation of the scheduled #104 change."""
    from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS, _defaults
    from budding_yeast_v2.data.systems.budding.real_mutants import REAL_MUTANTS
    from budding_yeast_v2.data.systems.budding.simulation import simulate_mutant, prepare_initial_conditions
    treatment=dict(REAL_MUTANTS['104'][1])
    control={k:v for k,v in treatment.items() if k!='is_mutant_104'}
    pa,ica,*_=prepare_initial_conditions(treatment);pb,icb,*_=prepare_initial_conditions(control)
    fa=[pa.get(k,_defaults[k]) for k in PARAM_KEYS];fb=[pb.get(k,_defaults[k]) for k in PARAM_KEYS]
    a=simulate_mutant('with_timed_induction',treatment)[1]
    b=simulate_mutant('without_timed_induction',control)[1]
    record=dict(input_parameters_identical=bool(np.array_equal(fa,fb)),initial_identical=bool(np.array_equal(ica,icb)),
                both_simulations_successful=a is not None and b is not None)
    if a is not None and b is not None:
        delta=a-b
        record.update(MAE=float(abs(delta).mean()),MSE=float((delta**2).mean()),maximum_difference=float(abs(delta).max()),
            mean_abs_difference_before_180=float(abs(delta[:,np.linspace(0,210,500)<180]).mean()))
    (output/'hidden_intervention_104.json').write_text(json.dumps(record,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/experiments'))
    parser.add_argument('--audit',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/analysis/badcase_audit_2026-10-27'))
    parser.add_argument('--output',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/analysis/operator_design_2026-10-30'))
    parser.add_argument('--skip-plots',action='store_true')
    parser.add_argument('--probe-hidden-intervention',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    runs,metrics=inventory(args.experiments,args.output)
    cohort,reference,selected=compare(runs,metrics,args.output)
    diagnostics,arrays=frozen_diagnosis(args.audit,reference,selected,args.output)
    if args.probe_hidden_intervention:probe_hidden_intervention(args.output)
    if not args.skip_plots:plots(cohort,metrics,selected,arrays,args.output)
    summary=dict(n_runs=len(runs),n_cohort=len(cohort),n_csv=int((runs.kind=='csv').sum()),n_al=int((runs.kind=='al').sum()),
                 reference_top20=selected.mutant_id.tolist(),n_frozen=len(diagnostics),
                 closure_ORI_wins=int((diagnostics.ORI_closure_MSE < diagnostics.ORI_original_MSE).sum()),
                 closure_ORI_mean_MSE=float(diagnostics.ORI_closure_MSE.mean()),
                 original_ORI_mean_MSE=float(diagnostics.ORI_original_MSE.mean()))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
