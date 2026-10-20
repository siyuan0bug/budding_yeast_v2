#!/usr/bin/env python3
"""Compare fixed trajectory labels using stored real-mutant AL evaluations.

This measures diagnostic/predictive utility of partitions, not a causal gain
from retraining or acquiring data with new labels. No experiments are mutated.
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

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.summarize_six_pattern_audit import select_badcases,markdown_table

METRICS=('MAE','MSE','Relative_L2')
TARGETS=(*METRICS,'composite_error_rank')
SCHEMES=('GLOBAL','ABC','BIO','ABC_BIO','SIX','ABC_CRITICAL','SIX_CRITICAL','SIX_BIO')
PARENTS={'ABC':'GLOBAL','BIO':'GLOBAL','ABC_BIO':'ABC','SIX':'ABC',
         'ABC_CRITICAL':'ABC','SIX_CRITICAL':'SIX','SIX_BIO':'ABC_BIO'}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schemes_from_labels(real):
    abc=real.original_pattern.str.split(':').str[0].to_numpy()
    six=real.pattern_code.astype(str).to_numpy()
    critical=real.is_bifurcation_critical.astype(bool).astype(str).to_numpy()
    bio=real.original_label.astype(str).to_numpy()
    return dict(GLOBAL=np.full(len(real),'all'),ABC=abc,BIO=bio,
        ABC_BIO=np.char.add(np.char.add(abc,'|'),bio),SIX=six,
        ABC_CRITICAL=np.char.add(np.char.add(abc,'|'),critical),
        SIX_CRITICAL=np.char.add(np.char.add(six,'|'),critical),
        SIX_BIO=np.char.add(np.char.add(six,'|'),bio))


def mutant_folds(abc,repeats=10,k=5,seed=42020):
    """Fixed splits stratified ONLY by the old ABC labels, never by error."""
    rng=np.random.RandomState(seed);result=np.empty((repeats,len(abc)),int)
    for r in range(repeats):
        for group in np.unique(abc):
            ids=np.flatnonzero(abc==group);rng.shuffle(ids)
            result[r,ids]=np.arange(len(ids))%k
    return result


def group_prediction(labels,parent_labels,source,train,prior,alpha):
    """Shrink child-group means toward parent, counting mutants not run-rows.

    Unsplit parent groups retain EXACTLY the parent prediction; thus P1/P2
    cannot acquire an artificial benefit from a second regularisation step.
    """
    labels=np.asarray(labels);parent_labels=np.asarray(parent_labels)
    levels,codes=np.unique(labels,return_inverse=True)
    active=np.zeros(len(labels),bool)
    for parent in np.unique(parent_labels):
        mask=parent_labels==parent
        if len(np.unique(labels[mask]))>1:active[mask]=True
    for child in levels:
        if len(np.unique(parent_labels[labels==child]))!=1:
            raise ValueError('Child partition must refine its parent')
    onehot=np.eye(len(levels))[codes[train]]
    counts=onehot.sum(0)
    sums=np.einsum('nk,anm->akm',onehot,source[:,train,:],optimize=True)
    numer=sums[:,codes,:]+alpha*prior
    denom=counts[codes][None,:,None]+alpha
    pred=np.divide(numer,denom,out=prior.copy(),where=denom>0)
    pred[:,~active,:]=prior[:,~active,:]
    return pred


def crossfit(y,schemes,folds,alpha=5.):
    """Hold out the target AL strategy AND every held-out mutant identity.

    y is algorithm x mutant x metric. Target-algorithm errors and all errors of
    test-fold identities are excluded from each group-mean fit.
    """
    if len(y)<2:raise ValueError('At least two independent strategy outcomes required')
    source=(y.sum(0,keepdims=True)-y)/(len(y)-1)
    preds={s:np.empty((len(folds),*y.shape)) for s in SCHEMES}
    for r,fold in enumerate(folds):
        for f in np.unique(fold):
            train=fold!=f;test=~train
            prior=np.broadcast_to(source[:,train,:].mean(1)[:,None,:],y.shape).copy()
            local={'GLOBAL':prior}
            for scheme in SCHEMES[1:]:
                parent=PARENTS[scheme]
                local[scheme]=group_prediction(schemes[scheme],schemes[parent],source,train,local[parent],alpha)
            for scheme in SCHEMES:preds[scheme][r][:,test,:]=local[scheme][:,test,:]
    return preds


def child_crossfit(y,labels,parent_labels,parent_prediction,folds,alpha=5.):
    source=(y.sum(0,keepdims=True)-y)/(len(y)-1)
    pred=np.empty_like(parent_prediction)
    for r,fold in enumerate(folds):
        for f in np.unique(fold):
            train=fold!=f;test=~train
            p=group_prediction(labels,parent_labels,source,train,parent_prediction[r],alpha)
            pred[r][:,test,:]=p[:,test,:]
    return pred


def expected_topk_hits(score,truth,k=20):
    """Expected hits under uniform tie breaking; no mutant-name tie advantage."""
    k=min(k,len(score));cut=np.partition(score,len(score)-k)[len(score)-k]
    above=score>cut+1e-12;ties=np.abs(score-cut)<=1e-12
    return float(truth[above].sum()+(k-int(above.sum()))*truth[ties].mean())


def read_evaluations(root,real,out):
    manifest=[];records=[];baseline_signature=None;run_configs={}
    for config in sorted(root.glob('*/w*/seed*/config.json')):
        run=config.parent;args=json.loads(config.read_text()).get('args',{})
        strategy=args.get('al_strategy',run.parents[1].name)
        run_configs[str(run.resolve())]=args
        for path in sorted((run/'eval').glob('**/per_mutant_metrics.csv')):
            row=dict(path=str(path.resolve()),sha256=fingerprint(path),strategy=strategy,seed=args.get('seed'),
                run_dir=str(run.resolve()),dataset=args.get('dataset_name'),config_sha256=fingerprint(config))
            if 'failed' in run.name.lower():row['status']='excluded_failed_run';manifest.append(row);continue
            if args.get('loss_type')!='hybrid_mse_rel':row['status']='excluded_other_loss';manifest.append(row);continue
            df=pd.read_csv(path)
            if df.Mutant.duplicated().any() or set(df.Mutant)!=set(real.mutant_name):
                raise ValueError(f'Real identity mismatch: {path}')
            if not np.isfinite(df[[*METRICS,'Correlation']].to_numpy()).all():
                raise ValueError(f'Nonfinite metrics: {path}')
            match=re.fullmatch(r'round_(\d+)_train(\d+)',path.parent.name)
            if not match:
                row['status']='endpoint_alias_or_extra';manifest.append(row);continue
            round_id,budget=map(int,match.groups())
            aligned=df.set_index('Mutant').loc[real.mutant_name]
            signature=hashlib.sha256(aligned[[*METRICS,'Correlation']].to_numpy(dtype='<f8').tobytes()).hexdigest()
            row.update(round=round_id,train_samples=budget,aligned_metrics_sha256=signature,status='included')
            if round_id==0:
                if baseline_signature is None:baseline_signature=signature
                elif signature!=baseline_signature:raise ValueError('Initial AL baselines are not identical')
                else:row['status']='duplicate_shared_round0';manifest.append(row);continue
            ranked=select_badcases(df).rename(columns={'Mutant':'mutant_name'})
            join=ranked.merge(real,on='mutant_name',validate='one_to_one')
            if not np.all(join.Pattern.str.split(':').str[0]==join.original_pattern.str.split(':').str[0]):
                raise ValueError('Stored ABC pattern differs from current canonical target')
            for c in METRICS:
                join['risk_'+c]=(join[c].rank(method='average',ascending=True)-1)/(len(join)-1)
            join['risk_composite_error_rank']=join[['risk_'+c for c in METRICS]].mean(1)
            join['strategy']=strategy if round_id else 'shared_round0'
            join['round']=round_id;join['train_samples']=budget
            join['metrics_path']=str(path.resolve());records.append(join);manifest.append(row)
    if not records:raise ValueError('No complete round metrics found')
    frame=pd.concat(records,ignore_index=True)
    frame.to_csv(out/'all_real_evaluations.csv.gz',index=False,compression='gzip')
    (out/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    (out/'run_configs.json').write_text(json.dumps(run_configs,indent=2,ensure_ascii=False)+'\n')
    return frame,manifest


def assess_stage(frame,real,round_id,schemes,args,out):
    stage=frame[frame['round']==round_id];strategies=sorted(stage.strategy.unique())
    if len(strategies)<2:return None
    aligned=[stage[stage.strategy.eq(s)].set_index('mutant_name').loc[real.mutant_name] for s in strategies]
    y=np.stack([d[['risk_'+c for c in TARGETS]].to_numpy() for d in aligned])
    truth=np.stack([d.selected_top_k.to_numpy() for d in aligned])
    folds=mutant_folds(schemes['ABC'],args.repeats,seed=args.seed)
    pred=crossfit(y,schemes,folds,args.shrinkage)
    losses={s:((p-y[None])**2) for s,p in pred.items()}
    rows=[];per_algorithm=[]
    for scheme in SCHEMES:
        for j,metric in enumerate(TARGETS):
            loss=losses[scheme][...,j].mean()
            base=losses['ABC'][...,j].mean();glob=losses['GLOBAL'][...,j].mean()
            rows.append(dict(round=round_id,scheme=scheme,metric=metric,groups=len(np.unique(schemes[scheme])),
                cv_rmse=np.sqrt(loss),cv_r2_vs_global=1-loss/glob,
                rmse_improvement_vs_ABC_percent=100*(1-np.sqrt(loss/base))))
        for i,strategy in enumerate(strategies):
            for j,metric in enumerate(TARGETS):
                a=losses[scheme][:,i,:,j].mean();b=losses['ABC'][:,i,:,j].mean()
                per_algorithm.append(dict(round=round_id,strategy=strategy,scheme=scheme,metric=metric,
                    cv_rmse=np.sqrt(a),rmse_improvement_vs_ABC_percent=100*(1-np.sqrt(a/b))))
    utility=[]
    for s in SCHEMES:
        for i,strategy in enumerate(strategies):
            hits=[expected_topk_hits(p[i,:,3],truth[i]) for p in pred[s]]
            utility.append(dict(round=round_id,strategy=strategy,scheme=s,
                expected_top20_hits=np.mean(hits),expected_top20_recall=np.mean(hits)/20))
    pd.DataFrame(rows).to_csv(out/f'round{round_id}_crossvalidated_utility.csv',index=False)
    pd.DataFrame(per_algorithm).to_csv(out/f'round{round_id}_utility_by_strategy.csv',index=False)
    pd.DataFrame(utility).to_csv(out/f'round{round_id}_top20_retrieval.csv',index=False)
    # Paired mutant-cluster bootstrap: repeated algorithms/folds are not
    # counted as independent biological observations.
    rng=np.random.RandomState(args.seed+round_id)
    draw=rng.randint(0,len(real),size=(args.bootstraps,len(real)))
    bootstrap=[]
    for scheme,parent in (('SIX','ABC'),('SIX_CRITICAL','SIX'),('ABC_BIO','ABC'),('SIX','ABC_BIO'),('SIX_BIO','ABC_BIO')):
        old=losses[parent].mean((0,1));new=losses[scheme].mean((0,1))
        for j,metric in enumerate(TARGETS):
            b=100*(1-np.sqrt(new[draw,j].mean(1)/old[draw,j].mean(1)))
            bootstrap.append(dict(round=round_id,scheme=scheme,parent=parent,metric=metric,
                rmse_improvement_percent=100*(1-np.sqrt(new[:,j].mean()/old[:,j].mean())),
                ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975))))
    pd.DataFrame(bootstrap).to_csv(out/f'round{round_id}_mutant_cluster_bootstrap.csv',index=False)
    # No error-driven choice of shrinkage: alpha=5 is the declared primary.
    sensitivity=[]
    for alpha in (0.,2.,5.,10.):
        pr=pred if alpha==args.shrinkage else crossfit(y,schemes,folds,alpha)
        for scheme in ('ABC','BIO','ABC_BIO','SIX','ABC_CRITICAL','SIX_CRITICAL','SIX_BIO'):
            err=((pr[scheme]-y[None])**2).mean((0,1,2))
            for j,metric in enumerate(TARGETS):sensitivity.append(dict(round=round_id,shrinkage=alpha,scheme=scheme,metric=metric,cv_rmse=np.sqrt(err[j])))
    pd.DataFrame(sensitivity).to_csv(out/f'round{round_id}_shrinkage_sensitivity.csv',index=False)
    # Same-cardinality null: shuffle the new groups within old ABC, retaining
    # group sizes and the unmodified P1/P2 branches. Flags shuffled within SIX.
    null=[];observed=(losses['ABC'].mean((0,1,2))-losses['SIX'].mean((0,1,2)))
    observed_flag=losses['SIX'].mean((0,1,2))-losses['SIX_CRITICAL'].mean((0,1,2))
    observed_bio=losses['ABC_BIO'].mean((0,1,2))-losses['SIX_BIO'].mean((0,1,2))
    rng_bio=np.random.RandomState(args.seed+42+round_id)
    for trial in range(args.permutations if round_id==args.primary_round else 0):
        shuffled=schemes['SIX'].copy();flagged=schemes['SIX_CRITICAL'].copy()
        for group in np.unique(schemes['ABC']):
            ids=np.flatnonzero(schemes['ABC']==group);shuffled[ids]=rng.permutation(shuffled[ids])
        for group in np.unique(schemes['SIX']):
            ids=np.flatnonzero(schemes['SIX']==group);flagged[ids]=rng.permutation(flagged[ids])
        p6=child_crossfit(y,shuffled,schemes['ABC'],pred['ABC'],folds,args.shrinkage)
        pc=child_crossfit(y,flagged,schemes['SIX'],pred['SIX'],folds,args.shrinkage)
        shuffled_bio=schemes['SIX_BIO'].copy()
        for group in np.unique(schemes['ABC_BIO']):
            ids=np.flatnonzero(schemes['ABC_BIO']==group);shuffled_bio[ids]=rng_bio.permutation(shuffled_bio[ids])
        pb=child_crossfit(y,shuffled_bio,schemes['ABC_BIO'],pred['ABC_BIO'],folds,args.shrinkage)
        gain6=losses['ABC'].mean((0,1,2))-((p6-y[None])**2).mean((0,1,2))
        gainc=losses['SIX'].mean((0,1,2))-((pc-y[None])**2).mean((0,1,2))
        gainb=losses['ABC_BIO'].mean((0,1,2))-((pb-y[None])**2).mean((0,1,2))
        for j,metric in enumerate(TARGETS):
            null.append(dict(trial=trial,metric=metric,six_gain=float(gain6[j]),critical_gain=float(gainc[j]),six_bio_gain=float(gainb[j])))
        if (trial+1)%100==0:print(f'Partition null {trial+1}/{args.permutations}',flush=True)
    if null:
        nd=pd.DataFrame(null);nd.to_csv(out/'cardinality_matched_null.csv',index=False)
        tests=[]
        for j,metric in enumerate(TARGETS):
            sub=nd[nd.metric.eq(metric)]
            for label,col,obs in (('SIX_vs_ABC','six_gain',observed[j]),('CRITICAL_increment','critical_gain',observed_flag[j]),('SIX_BIO_vs_ABC_BIO','six_bio_gain',observed_bio[j])):
                tests.append(dict(comparison=label,metric=metric,observed_mse_gain=obs,
                    null_mean=sub[col].mean(),p_value=(1+np.sum(sub[col]>=obs))/(1+len(sub))))
        pd.DataFrame(tests).to_csv(out/'cardinality_matched_permutation_tests.csv',index=False)
    # Store one prediction per held-out identity/strategy, averaged across
    # fold repetitions, with exact targets; used only for diagnosis.
    predictions=[]
    for s in SCHEMES:
        mean=pred[s].mean(0)
        for i,strategy in enumerate(strategies):
            for j,name in enumerate(real.mutant_name):
                predictions.append(dict(strategy=strategy,mutant_name=name,scheme=s,
                    predicted_error_rank=mean[i,j,3],actual_error_rank=y[i,j,3]))
    pd.DataFrame(predictions).to_csv(out/f'round{round_id}_out_of_fold_predictions.csv',index=False)
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root',type=Path,default=ROOT/'experiments/al_pure_random_1m/biom_refiner_plus')
    p.add_argument('--labels',type=Path,default=ROOT/'scripts/analysis/outputs/six_pattern_audit_2026-10-20/mutant_aware_pure_random_100000_210m_500s/real_mutants.csv')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--primary-round',type=int,default=2)
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--shrinkage',type=float,default=5.)
    p.add_argument('--permutations',type=int,default=499)
    p.add_argument('--bootstraps',type=int,default=5000)
    p.add_argument('--seed',type=int,default=42020)
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if args.repeats<1 or args.bootstraps<1 or args.permutations<1 or args.shrinkage<0:p.error('Invalid analysis counts')
    real=pd.read_csv(args.labels).sort_values('row_index').reset_index(drop=True)
    if len(real)!=126 or real.mutant_name.duplicated().any():raise ValueError('Expected canonical 126 conditions')
    schemes=schemes_from_labels(real);frame,manifest=read_evaluations(args.experiment_root,real,out)
    # Per-run lists, group error summaries and persistence use actual errors,
    # never pooled JSON correlation/relative-L2 in place of per-mutant metrics.
    frame[frame.selected_top_k].to_csv(out/'top20_each_evaluation.csv',index=False)
    group_rows=[]
    lookup=real.set_index('mutant_name')
    for (strategy,round_id),sub in frame.groupby(['strategy','round']):
        sub=sub.set_index('mutant_name').loc[real.mutant_name]
        for scheme in SCHEMES[1:]:
            for value in np.unique(schemes[scheme]):
                part=sub.iloc[np.flatnonzero(schemes[scheme]==value)]
                row=dict(strategy=strategy,round=round_id,scheme=scheme,label=value,n=len(part),
                    badcase_count=int(part.selected_top_k.sum()),badcase_rate=float(part.selected_top_k.mean()))
                for metric in (*METRICS,'Correlation'):
                    row[metric+'_median']=part[metric].median();row[metric+'_mean']=part[metric].mean()
                group_rows.append(row)
    pd.DataFrame(group_rows).to_csv(out/'label_group_errors.csv',index=False)
    primary=frame[frame['round']==args.primary_round]
    if primary.strategy.nunique()<2:raise ValueError('Insufficient primary-round strategy outcomes')
    summary=primary.groupby('mutant_name').agg(strategies=('strategy','nunique'),
        top20_frequency=('selected_top_k','sum'),mean_error_rank=('risk_composite_error_rank','mean'),
        median_MAE=('MAE','median'),median_MSE=('MSE','median'),median_Relative_L2=('Relative_L2','median'),
        min_Relative_L2=('Relative_L2','min'),max_Relative_L2=('Relative_L2','max'))
    summary=summary.merge(real.set_index('mutant_name')[['original_pattern','original_label','pattern_code','is_bifurcation_critical']],left_index=True,right_index=True)
    summary=summary.sort_values(['top20_frequency','mean_error_rank'],ascending=False)
    summary.to_csv(out/'persistent_badcases_all126.csv');summary.head(20).to_csv(out/'persistent_top20.csv')
    primary[primary.mutant_name.str.split('_').str[0].isin(['15','22','42','124'])].to_csv(out/'key_badcases_by_strategy.csv',index=False)
    # Explicit loop works across the pandas versions used by this repository.
    flag_rows=[]
    for strategy,g in primary.groupby('strategy'):
        flag_rows.append(dict(strategy=strategy,top20_critical_hits=int((g.selected_top_k&g.is_bifurcation_critical).sum()),total_critical=int(g.is_bifurcation_critical.sum()),badcase_total=int(g.selected_top_k.sum())))
    pd.DataFrame(flag_rows).to_csv(out/'critical_flag_badcase_coverage.csv',index=False)
    overall=[]
    for round_id in sorted(frame['round'].unique()):
        if round_id==0:continue
        result=assess_stage(frame,real,int(round_id),schemes,args,out)
        if result is not None:overall.append(result)
    pd.concat(overall,ignore_index=True).to_csv(out/'all_rounds_label_utility.csv',index=False)
    (out/'methodology.json').write_text(json.dumps(dict(
        primary_round=args.primary_round,primary_strategies=sorted(primary.strategy.unique()),
        targets=list(TARGETS),primary_target='composite_error_rank',
        rank_definition='Within each run, (ascending average rank - 1)/(126 - 1); composite is equal mean of MAE/MSE/Relative_L2 ranks',
        cross_validation='Leave target strategy out; five-fold mutant-identity holdout stratified by ABC, repeated with fixed seed',
        folds=5,repeats=args.repeats,shrinkage_mutant_equivalents=args.shrinkage,
        permutations=args.permutations,bootstrap_mutant_clusters=args.bootstraps,seed=args.seed,
        categories={s:len(np.unique(v)) for s,v in schemes.items()},
        interpretation='Post-hoc diagnostic utility, NOT causal evidence that new-label AL/training improves surrogate accuracy; one seed and a shared benchmark',
        label_source=str(args.labels.resolve()),label_source_sha256=fingerprint(args.labels)),indent=2)+'\n')
    # Detect concurrent rewrites of metrics rather than mixing experiment states.
    for row in manifest:
        if fingerprint(row['path'])!=row['sha256']:raise RuntimeError(f'Evaluation changed: {row["path"]}')
    print(pd.read_csv(out/f'round{args.primary_round}_crossvalidated_utility.csv').query("metric == 'composite_error_rank'").to_string(index=False))
    print(summary.head(20).to_string())


if __name__=='__main__':main()
