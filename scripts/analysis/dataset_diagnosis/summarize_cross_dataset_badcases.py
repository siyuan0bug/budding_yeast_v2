"""Build evidence tables and scientific plots from frozen badcase audit outputs."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.cross_dataset_badcases import DEFAULT_OUT,mid
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pure_random_evidence import VARIABLES


def markdown(frame):
    columns=list(frame.columns);lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in frame.itertuples(index=False,name=None):
        values=[f'{x:.4g}' if isinstance(x,(float,np.floating)) else str(x) for x in row]
        lines.append('| '+' | '.join(v.replace('|','/') for v in values)+' |')
    return '\n'.join(lines)


def short(name):
    return str(name).replace('_100000_210m_500s','').replace('_50000_210m_500s','_50k').replace('_dataset_210min_500steps_dual_labels','').replace('mutant_aware_pure_random','pure_random').replace('distribution_twin','twin')


def build(args):
    out=args.output;runs=pd.read_csv(out/'primary_runs.csv');all_runs=pd.read_csv(out/'runs.csv')
    metrics=pd.read_csv(out/'all_mutant_metrics.csv',dtype={'mutant_id':str});support=pd.read_csv(out/'support_and_errors.csv',dtype={'mutant_id':str})
    validations=[];replays=[];channels=[];predictions={}
    for _,run in runs.iterrows():
        directory=out/'replay'/run.run_id
        historical=out/'replay_historical'/run.run_id
        if (historical/'provenance.json').exists() and json.loads((historical/'provenance.json').read_text())['matched_archived_metrics_within_tolerance']:directory=historical
        if not (directory/'provenance.json').exists():continue
        p=json.loads((directory/'provenance.json').read_text())
        historical=metrics[metrics.run_id.eq(run.run_id)]
        replayed=pd.read_csv(directory/'replayed_metrics.csv',dtype={'mutant_id':str})
        same_set=set(historical.mutant_id)==set(replayed.mutant_id)
        matched=p['matched_archived_metrics_within_tolerance'] and same_set
        validations.append(dict(run_id=run.run_id,kind=run.kind,dataset=run.dataset_tag,norm=run.norm,matched=matched,
            same_real_condition_set=same_set,**{k+'_max_abs_difference':v for k,v in p['max_abs_differences'].items()}))
        if not matched:continue
        replayed['kind']=run.kind;replayed['dataset_tag']=run.dataset_tag;replays.append(replayed)
        channels.append(pd.read_csv(directory/'channel_errors.csv',dtype={'mutant_id':str}))
        with np.load(directory/'predictions.npz',allow_pickle=False) as z:
            predictions[run.run_id]=dict(pred=z['predictions'],truth=z['targets'],ids=[mid(n) for n in z['names']])
    validation=pd.DataFrame(validations);validation.to_csv(out/'replay_validation.csv',index=False)
    replay=pd.concat(replays,ignore_index=True);channel=pd.concat(channels,ignore_index=True)
    replay.to_csv(out/'validated_replay_diagnostics.csv',index=False);channel.to_csv(out/'validated_channel_errors.csv',index=False)
    valid_ids=set(validation[validation.matched].run_id);static_ids=set(runs[runs.kind.eq('static')].run_id)
    valid_static=valid_ids&static_ids
    relevant=support[support.run_id.isin(valid_ids)].merge(replay.drop(columns=['Mutant','MAE','MSE','Relative_L2','kind','round'],errors='ignore'),on=['run_id','mutant_id'],suffixes=('','_replay'),validate='one_to_one')
    relevant.to_csv(out/'validated_case_evidence.csv',index=False)
    ranking=pd.read_csv(out/'cross_dataset_badcase_ranking.csv',dtype={'mutant_id':str})
    selected=ranking.mutant_id.head(args.top_k).tolist()
    # Strict matched architecture/training cohort, selected without test errors.
    cohort=runs[runs.kind.eq('static')&runs.model.eq('biom_refiner_plus')&runs.norm.eq('log1p')&runs.width.eq(192)&runs.batch_size.eq(256)&runs.epochs.eq(100)&runs.seed.eq(42)&np.isclose(runs.lr,.00088)&runs.run_id.isin(valid_ids)]
    cohort_metrics=metrics[metrics.run_id.isin(cohort.run_id)]
    cohort_metrics.groupby('mutant_id').agg(mean_rank=('severity_rank','mean'),top20_runs=('is_top20','sum'),runs=('run_id','nunique')).sort_values('mean_rank').to_csv(out/'matched_training_cohort_ranking.csv')
    # Per-dataset and per-AL-round top20; every original run also stays in the master CSV.
    per=out/'per_experiment';per.mkdir(exist_ok=True)
    for _,run in all_runs.iterrows():
        frame=metrics[metrics.run_id.eq(run.run_id)]
        directory=per/run.kind/run.dataset_tag;directory.mkdir(parents=True,exist_ok=True)
        for key in ('severity_rank','rank_MAE','rank_Relative_L2','rank_MSE'):
            frame.sort_values(key,kind='stable').head(20).to_csv(directory/f'{run.model}_{run.loss}_seed{run.seed}_round{run["round"]}_{run.run_id}_top20_{key}.csv',index=False)
    # Raw truth versions and parameter/IC differences relative to new AL canonical data.
    ref_run=runs[runs.kind.eq('al')].iloc[0]
    ref_cache=out/'features'/f'{hashlib.sha256(ref_run.dataset.encode()).hexdigest()[:12]}.npz'
    with np.load(ref_cache,allow_pickle=False) as z:
        refnames=z['names'][z['is_real']];refp=z['params'][z['is_real']];refic=z['ic'][z['is_real']];refy=z['real_y'];reference={mid(n):i for i,n in enumerate(refnames)}
    truth_versions=[]
    for _,run in runs.iterrows():
        key=hashlib.sha256(run.dataset.encode()).hexdigest()[:12]
        with np.load(out/'features'/f'{key}.npz',allow_pickle=False) as z:
            idx=np.flatnonzero(z['is_real']);p=z['params'][idx];ic=z['ic'][idx];truth=z['real_y'];names=z['names'][idx]
            for i,name in enumerate(names):
                condition=mid(name);j=reference[condition];diff=truth[i].astype('float64')-refy[j]
                rel=np.sqrt(np.sum(diff*diff,axis=1)/np.maximum(np.sum(refy[j].astype('float64')**2,axis=1),5.)).mean()
                truth_versions.append(dict(run_id=run.run_id,dataset_tag=run.dataset_tag,kind=run.kind,mutant_id=condition,
                    same_parameters=bool(np.array_equal(p[i].astype('float32'),refp[j].astype('float32'))),
                    same_initial=bool(np.array_equal(ic[i],refic[j])),same_truth=bool(np.array_equal(truth[i],refy[j])),
                    truth_Relative_L2_to_current=float(rel),truth_clb2_max=float(truth[i,2].max()),truth_ori_max=float(truth[i,33].max())))
    versions=pd.DataFrame(truth_versions);versions.to_csv(out/'real_truth_versions.csv',index=False)
    # Aggregate evidence for 20 named badcases; archived errors are never replaced by replay values.
    cases=[]
    for condition in selected:
        rank=ranking[ranking.mutant_id.eq(condition)].iloc[0]
        g=relevant[relevant.mutant_id.eq(condition)&relevant.kind.eq('static')]
        dominant=g.dominant_mse_variable.mode().iloc[0] if len(g) else 'unverified'
        local=g.input_near20_same_pattern_fraction.median() if len(g) else np.nan
        modes=g.module_count.mode().iloc[0] if len(g) else np.nan
        if modes>3:reason='L4超出新pool范围；多KO与初值联合外推'
        elif condition=='15':reason='START假激活；低信号多通道相对误差；条件邻域稀少'
        elif condition in ('22','23','20','24','33'):reason='单次分裂/衰竭瞬态与阶段切换；ORI驱动或时序误差'
        elif condition=='62':reason='单模块内GAL+db复合突变；高驱动/稳定性与ORI误差'
        elif condition in ('124','59','58','57','54'):reason='高幅值或稳定性背景下ORI累计量误差；分类正确仍可错'
        elif condition in ('16','35'):reason='START/S-G2解耦条件覆盖与动力学混淆'
        else:reason=f'联合突变条件覆盖不足的关联证据；主要误差通道{dominant}'
        cases.append(dict(mutant_id=condition,Mutant=rank.Mutant,top20_datasets=int(rank.top20_datasets),datasets=int(rank.datasets),
            median_MAE=rank.median_MAE,median_Relative_L2=rank.median_Relative_L2,modules=modes,dominant_error_variable=dominant,
            nearest20_same_pattern_median=local,verified_static_runs=len(g),interpretation=reason))
    case_table=pd.DataFrame(cases)
    base_run=runs[runs.dataset_tag.eq('mutant_aware_pure_random_100000_210m_500s')].iloc[0]
    baseline=metrics[metrics.run_id.eq(base_run.run_id)][['mutant_id','MAE','MSE','Relative_L2']].rename(columns={'MAE':'baseline_MAE','MSE':'baseline_MSE','Relative_L2':'baseline_Relative_L2'})
    case_table=case_table.merge(baseline,on='mutant_id',validate='one_to_one')
    case_table.to_csv(out/'twenty_badcase_evidence.csv',index=False)
    # Keep all static datasets visible without treating heterogeneous runs as a
    # controlled leaderboard. Model/loss/norm and replay status accompany scores.
    overview=[]
    for _,run in runs[runs.kind.eq('static')].iterrows():
        g=metrics[metrics.run_id.eq(run.run_id)]
        item=dict(run_id=run.run_id,dataset=run.dataset_tag,model=run.model,loss=run.loss,norm=run.norm,
                  replay_validated=run.run_id in valid_ids,mean_MAE=g.MAE.mean(),mean_Relative_L2=g.Relative_L2.mean())
        for condition in ('15','20','22','42','124'):
            item[f'relL2_{condition}']=g.loc[g.mutant_id.eq(condition),'Relative_L2'].iloc[0]
        overview.append(item)
    pd.DataFrame(overview).to_csv(out/'static_dataset_overview.csv',index=False)
    # Legacy AL is an explicitly separate archival appendix: no recovered pool
    # or training split is assumed, and these repeated rounds do not reweight
    # the static-data badcase ranking.
    manifest=json.loads((out/'raw_manifest.json').read_text());legacy=[];legacy_runs=[]
    for source in manifest['csv']:
        source=Path(source)
        if 'al_pure_random_1m' not in source.parts:continue
        g=pd.read_csv(source);g['mutant_id']=g.Mutant.map(mid)
        for key in ('MAE','Relative_L2','MSE'):g['rank_'+key]=g[key].rank(ascending=False,method='min')
        g['severity_rank']=(g.rank_MAE+g.rank_Relative_L2)/2
        g['is_top20']=g.severity_rank.rank(method='first')<=20
        g['source']=str(source);legacy.append(g)
        legacy_runs.append(dict(source=str(source),conditions=len(g),mean_MAE=g.MAE.mean(),mean_Relative_L2=g.Relative_L2.mean()))
    if legacy:
        pd.concat(legacy,ignore_index=True).to_csv(out/'legacy_al_archived_metrics.csv',index=False)
        pd.DataFrame(legacy_runs).to_csv(out/'legacy_al_archived_runs.csv',index=False)
    # Quantify whether easy cases include duplicate inputs, without assuming identical targets.
    duplicate=[]
    for rid,g in relevant[relevant.kind.eq('static')].groupby('run_id'):
        for label,sub in [('bad20',g.nsmallest(20,'severity_rank')),('good20',g.nlargest(20,'severity_rank'))]:
            duplicate.append(dict(run_id=rid,group=label,exact_input_matches=int((sub.train_exact_model_inputs>0).sum()),n=len(sub),
                median_input_distance=sub.input_nn_distance.median(),median_local25=sub.train_local25.median()))
    pd.DataFrame(duplicate).to_csv(out/'bad_vs_good_controls.csv',index=False)
    correlation=pd.read_csv(out/'support_error_correlations.csv');correlation=correlation[correlation.run_id.isin(valid_static)].copy()
    order=np.argsort(correlation.pvalue.to_numpy());pvalues=correlation.pvalue.to_numpy()[order];q=np.minimum.accumulate((pvalues*len(pvalues)/np.arange(1,len(pvalues)+1))[::-1])[::-1]
    correlation.loc[correlation.index[order],'qvalue_bh']=np.minimum(q,1);correlation.to_csv(out/'validated_support_correlations.csv',index=False)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    # Two metrics reveal different failure modes; no pooled-correlation ranking.
    ordered=runs[runs.kind.eq('static')].copy();ordered['label']=ordered.dataset_tag.map(short)
    fig,axes=plt.subplots(2,1,figsize=(19,15),layout='constrained')
    for ax,key,title in zip(axes,['Relative_L2','MAE'],['Per-mutant Relative L2 (log10)','Per-mutant MAE (log10)']):
        a=metrics[metrics.run_id.isin(ordered.run_id)].pivot(index='mutant_id',columns='run_id',values=key).reindex(index=selected,columns=ordered.run_id)
        im=ax.imshow(np.log10(np.maximum(a.to_numpy(),1e-4)),aspect='auto',cmap='magma',vmin=-3,vmax=1)
        ax.set_xticks(range(len(ordered)));ax.set_xticklabels(ordered.label,rotation=65,ha='right',fontsize=7)
        ax.set_yticks(range(len(selected)));ax.set_yticklabels(selected);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.8)
    fig.savefig(figures/'badcase_dataset_heatmap.png',dpi=180);plt.close(fig)
    # Cases: same target version and same model/norm in each best/worst pair.
    pairs=[];case_dir=figures/'cases';case_dir.mkdir(exist_ok=True)
    primary_lookup=runs.set_index('run_id');support_lookup=support.set_index(['run_id','mutant_id'])
    for condition in selected:
        options=[]
        for rid in valid_static:
            record=primary_lookup.loc[rid]
            if record.model!='biom_refiner_plus' or record.norm!='log1p':continue
            pred=predictions[rid]
            if condition not in pred['ids']:continue
            pos=pred['ids'].index(condition);target_hash=hashlib.sha256(pred['truth'][pos].tobytes()).hexdigest()
            error=float(metrics[metrics.run_id.eq(rid)&metrics.mutant_id.eq(condition)].Relative_L2.iloc[0])
            options.append(dict(run_id=rid,target=target_hash,error=error,position=pos))
        if not options:continue
        frame=pd.DataFrame(options);groups=[g for _,g in frame.groupby('target')]
        group=max(groups,key=lambda g:((g.error.max()+1e-8)/(g.error.min()+1e-8) if len(g)>1 else 0,len(g)))
        low=group.loc[group.error.idxmin()];high=group.loc[group.error.idxmax()]
        variables=[0,1,2,3,4,19,22,32,33];t=np.linspace(0,210,500)
        fig,axes=plt.subplots(3,3,figsize=(14,9),layout='constrained')
        for ax,v in zip(axes.flat,variables):
            ax.plot(t,predictions[low.run_id]['truth'][int(low.position),v],color='black',lw=1.8,label='Stored ODE target')
            ax.plot(t,predictions[high.run_id]['pred'][int(high.position),v],color='#ce4239',lw=1.2,label='Higher-error dataset')
            ax.plot(t,predictions[low.run_id]['pred'][int(low.position),v],color='#218c74',lw=1.2,ls='--',label='Lower-error dataset')
            ax.set_title(VARIABLES[v]);ax.grid(alpha=.25);ax.set_xlabel('min')
        axes[0,0].legend(fontsize=7)
        title=f'Mutant #{condition} | same target version | higher: {short(primary_lookup.loc[high.run_id].dataset_tag)} ({high.error:.3g}) | lower: {short(primary_lookup.loc[low.run_id].dataset_tag)} ({low.error:.3g})'
        fig.suptitle(title,fontsize=10);fig.savefig(case_dir/f'case_{condition}.png',dpi=160);plt.close(fig)
        pairs.append(dict(mutant_id=condition,lower_run=low.run_id,higher_run=high.run_id,lower_error=low.error,higher_error=high.error))
    pd.DataFrame(pairs).to_csv(out/'same_truth_plot_pairs.csv',index=False)
    # Active learning curves are compared within architecture and at actual budgets.
    al=metrics[metrics.kind.eq('al')&metrics.model.eq('fno')&metrics.mutant_id.isin(['15','20','22','42','124'])]
    fig,axes=plt.subplots(1,5,figsize=(19,4),layout='constrained')
    for ax,condition in zip(axes,['15','20','22','42','124']):
        for strategy,g in al[al.mutant_id.eq(condition)].groupby('strategy'):
            g=g.sort_values('round');n=[10000,20000,40000,70000]
            ax.plot([n[int(r)] for r in g['round']],g.Relative_L2,'o-',label=strategy.replace('al4pde_',''))
        ax.set_title(f'#{condition}');ax.set_xlabel('Training samples');ax.set_ylabel('Relative L2');ax.grid(alpha=.3)
    axes[-1].legend(fontsize=8);fig.savefig(figures/'fno_badcases_across_al_rounds.png',dpi=180);plt.close(fig)
    # ODE probes: discrete phase switches are observations over 210 min.
    probes=pd.read_csv(out/'fixed_background_ode_probes.csv',dtype={'mutant_id':str})
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,axis,title in zip(axes,['CLN2_copy_number','CLB5_copy_number','APCA_residual_fraction'],['#15: fixed cln3/bck2 KO','#22: fixed cln1/2 + cln3 KO','#124: APC-A near zero']):
        g=probes[probes.axis.eq(axis)];ax.plot(g.value,g.max_ori,'o-');ax.set_title(title);ax.set_xlabel(axis);ax.set_ylabel('max ORI');ax.grid(alpha=.3)
    fig.savefig(figures/'fixed_background_ode_probes.png',dpi=180);plt.close(fig)
    summary=dict(metric_files=len(all_runs),static_metric_files=int(all_runs.kind.eq('static').sum()),al_round_metric_files=int(all_runs.kind.eq('al').sum()),
        static_dataset_directories=int(all_runs[all_runs.kind.eq('static')].dataset_tag.nunique()),unique_static_datasets=int(runs.kind.eq('static').sum()),
        primary_runs=len(runs),validated_replays=int(validation.matched.sum()),unmatched_replays=int((~validation.matched).sum()),
        validated_static_runs=len(valid_static),matched_training_cohort=len(cohort),named_badcases=len(cases),case_figures=len(pairs),
        support_rows=len(support),static_split_provenance=support[support.kind.eq('static')].split_provenance.value_counts().to_dict(),
        main_output=str(out),raw_source_data_modified=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print(markdown(case_table[['mutant_id','median_MAE','median_Relative_L2','dominant_error_variable','interpretation']]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--top-k',type=int,default=20);args=parser.parse_args();build(args)


if __name__=='__main__':main()
