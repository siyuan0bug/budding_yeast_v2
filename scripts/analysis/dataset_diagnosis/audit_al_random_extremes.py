"""Audit physical extremes, normalization and errors in an existing Budding AL run.

Read-only: no generator/model/loss/filter changes, no training or new ODE queries.
Counterfactual filtered statistics are diagnostic; they are not retrained results.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from budding_yeast_v2.data.dataset_archive import open_dataset, round_dataset_paths
from budding_yeast_v2.data.normalization import normalize_y,normalize_p,denormalize_y
from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
from budding_yeast_v2.eval import load_model_from_checkpoint
from budding_yeast_v2.utils.metrics import calculate_metrics


def moments(block):
    a=np.asarray(block,dtype=np.float64)
    if not np.isfinite(a).all():raise ValueError('Nonfinite saved trajectory')
    log=np.log1p(np.maximum(a,0.))
    return (a.max(2),a.min(2),a.sum(2),(a*a).sum(2),log.sum(2),(log*log).sum(2))


def fit(sum_,squares,indices,steps):
    n=len(indices)*steps
    mean=sum_[indices].sum(0)/n
    variance=np.maximum((squares[indices].sum(0)-n*mean**2)/(n-1),0.)
    return mean,np.sqrt(variance)+1e-6


def energy(sum_,squares,mean,std,steps):
    return np.maximum(squares-2*sum_*mean+steps*mean**2,0.)/std**2


def load_snapshot(path):
    with open_dataset(path,allow_pickle=False) as z:
        keys=('data','params','y_init','mutant_names','candidate_ids','module_counts',
              'pattern_group_codes','pattern_labels','labels','split_train','split_val',
              'split_test','split_real','variable_names')
        return {k:z[k] for k in keys}


def predict(model,indices,saved,state,device,batch_size):
    pred=[];target=[];normalized_pred=[];normalized_target=[]
    steps=saved['data'].shape[-1]
    for offset in range(0,len(indices),batch_size):
        rows=indices[offset:offset+batch_size]
        y=torch.tensor(saved['data'][rows],dtype=torch.float32)
        y_norm=normalize_y(y,state['mean_y'],state['std_y'],state['norm_type'],state.get('s_c'))
        # Match DataModule exactly: IC is taken from the normalized trajectory t0.
        ic=y_norm[:,:,0]
        x=torch.stack((ic[:,:,None].expand(-1,-1,steps),torch.linspace(0,1,steps)[None,None,:].expand(len(rows),39,-1)),2)
        p=normalize_p(torch.tensor(saved['params'][rows],dtype=torch.float32),state['p_mean'],state['p_std'],state['norm_type'])
        with torch.no_grad():output=model.forward_ic_time(x.to(device),p.to(device)).cpu().float()
        decoded=denormalize_y(output,state['mean_y'],state['std_y'],state['norm_type'],state.get('s_c'))
        truth=denormalize_y(y_norm,state['mean_y'],state['std_y'],state['norm_type'],state.get('s_c'))
        if not torch.isfinite(decoded).all():raise ValueError('Nonfinite checkpoint prediction')
        pred.append(decoded.numpy());target.append(truth.numpy())
        normalized_pred.append(output.numpy());normalized_target.append(y_norm.numpy())
    return tuple(np.concatenate(x) for x in (pred,target,normalized_pred,normalized_target))


def error_rows(pred,target,pred_z,target_z):
    difference=pred-target
    normdiff=pred_z-target_z
    mse=(normdiff**2).mean((1,2))
    relative=np.sqrt((normdiff**2).sum(2))/np.sqrt((target_z**2).sum(2)+.1**2)
    return dict(MAE=np.abs(difference).mean((1,2)),MSE=(difference**2).mean((1,2)),
                normalized_MSE=mse,hybrid=mse+.2*np.minimum(relative,10.).mean(1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--train-stratum-samples',type=int,default=256)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    rounds=round_dataset_paths(args.run)
    latest=rounds[-1]
    print(f'Loading one final cumulative snapshot: {latest}',flush=True)
    saved=load_snapshot(latest)
    data=saved['data'];n,v,steps=data.shape
    if v!=39:raise ValueError('This audit is for Budding only')
    print(f'Loaded {data.shape}; extracting moments with {args.workers} CPU threads',flush=True)
    matrices=[np.empty((n,v),dtype=np.float64) for _ in range(6)]
    slices=[slice(i,min(i+256,n)) for i in range(0,n,256)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for idx,result in enumerate(executor.map(lambda s:moments(data[s]),slices)):
            for dest,source in zip(matrices,result):dest[slices[idx]]=source
            if idx%50==0:print(f'Extrema/moments: {slices[idx].stop}/{n}',flush=True)
    maxima,minima,rawsum,rawsquare,logsum,logsquare=matrices
    real=saved['split_real'];train=saved['split_train'];realmax=maxima[real].max(0)
    floor_limits=np.maximum(2*realmax,2.)
    old_molecular=(maxima[:,1:]>floor_limits[1:]).any(1)
    old_mass=maxima[:,0]>10.
    high=old_molecular|old_mass
    literal=(maxima>2*realmax).any(1)
    ratio=np.divide(maxima,realmax,out=np.full_like(maxima,np.nan),where=realmax>0)
    names=saved['variable_names']
    stages=[];scales=[];vars_rows=[];groups=[]
    for j,name in enumerate(names):
        peaks=maxima[train,j]
        vars_rows.append(dict(variable=str(name),real_max=realmax[j],synthetic_max=peaks.max(),
            synthetic_to_real_max_ratio=peaks.max()/realmax[j] if realmax[j]>0 else None,
            peak_q50=np.quantile(peaks,.5),peak_q95=np.quantile(peaks,.95),
            peak_q99=np.quantile(peaks,.99),peak_q999=np.quantile(peaks,.999),
            above_2real=int((peaks>2*realmax[j]).sum()),above_legacy_floor=int((peaks>floor_limits[j]).sum()),
            legacy_floor_threshold=floor_limits[j],raw_min=minima[train,j].min()))
    pd.DataFrame(vars_rows).to_csv(args.output/'variable_extrema.csv',index=False)
    # Verify cumulative rows are exact prefixes, avoiding duplicate multi-GB reads.
    previous=set();stage_index=np.full(n,-1,dtype=int)
    for r,path in enumerate(rounds):
        with open_dataset(path,allow_pickle=False) as z:
            current=z['split_train'];ids=z['candidate_ids'][current]
        count=len(current);subset=train[:count]
        if not np.array_equal(ids,saved['candidate_ids'][subset]):raise ValueError('Cumulative training order is not a prefix')
        newly=np.array([i for i in subset if int(i) not in previous],dtype=int)
        previous.update(map(int,subset));stage_index[newly]=r
        mean,std=fit(logsum,logsquare,subset,steps)
        clean=subset[~high[subset]]
        cleanmean,cleanstd=fit(logsum,logsquare,clean,steps)
        zenergy=energy(logsum[subset],logsquare[subset],mean,std,steps).sum(1)
        rawmean=rawsum[subset].sum(0)/(count*steps)
        rawenergy=energy(rawsum[subset],rawsquare[subset],rawmean,np.ones(v),steps).sum(1)
        logenergy=energy(logsum[subset],logsquare[subset],mean,np.ones(v),steps).sum(1)
        row=dict(round=r,train_samples=count,added_samples=len(newly),
            legacy_rejected=int(high[subset].sum()),legacy_fraction=float(high[subset].mean()),
            newly_legacy_rejected=int(high[newly].sum()),newly_legacy_fraction=float(high[newly].mean()),
            literal_2real_rejected=int(literal[subset].sum()),mass_above_10=int(old_mass[subset].sum()),
            molecular_rejected=int(old_molecular[subset].sum()),
            centered_raw_energy_high_share=float(rawenergy[high[subset]].sum()/rawenergy.sum()),
            centered_log_energy_high_share=float(logenergy[high[subset]].sum()/logenergy.sum()),
            centered_logz_energy_high_share=float(zenergy[high[subset]].sum()/zenergy.sum()))
        state=torch.load(path.parent/'preprocessing.pt',map_location='cpu',weights_only=False)
        saved_mean=state['mean_y'].numpy().reshape(-1)
        saved_std=state['std_y'].numpy().reshape(-1)
        row['saved_mean_max_abs_difference_float64_reduction']=float(np.max(np.abs(mean-saved_mean)))
        row['saved_std_max_abs_difference_float64_reduction']=float(np.max(np.abs(std-saved_std)))
        row['saved_mean_difference_in_std_max']=float(np.max(np.abs(mean-saved_mean)/saved_std))
        for j,name in enumerate(names):
            scales.append(dict(round=r,variable=str(name),std_all=std[j],std_if_filtered=cleanstd[j],
                saved_mean=float(saved_mean[j]),saved_std=float(saved_std[j]),recomputed_mean64=float(mean[j]),
                std_ratio=std[j]/cleanstd[j],small_log_error_mse_weight_ratio=(cleanstd[j]/std[j])**2,
                unit_range_0to1_z_width=np.log(2)/std[j],
                real_max_z=(np.log1p(max(realmax[j],0))-mean[j])/std[j]))
        for kind,values in (('pattern_group',saved['pattern_group_codes']),('L',saved['module_counts'])):
            for value in np.unique(values[subset]):
                mask=values[subset]==value
                groups.append(dict(round=r,kind=kind,value=str(value),count=int(mask.sum()),
                                   legacy_rejected=int(high[subset][mask].sum()),fraction=float(high[subset][mask].mean())))
        stages.append(row)
    pd.DataFrame(stages).to_csv(args.output/'stage_extremes.csv',index=False)
    pd.DataFrame(scales).to_csv(args.output/'normalization_effects.csv',index=False)
    pd.DataFrame(groups).to_csv(args.output/'group_coverage.csv',index=False)
    rows=pd.DataFrame(dict(row=np.arange(n),name=saved['mutant_names'],candidate_id=saved['candidate_ids'],
        acquisition_round=stage_index,level=saved['module_counts'],group=saved['pattern_group_codes'],
        legacy_rejected=high,literal_2real=literal,min_value=minima.min(1),max_real_ratio=np.nanmax(ratio,axis=1)))
    rows.to_csv(args.output/'sample_extremes.csv.gz',index=False)
    top=[]
    for j in np.argsort(np.nanmax(ratio[train],axis=0))[-12:][::-1]:
        i=train[np.argmax(maxima[train,j])]
        top.append(dict(variable=str(names[j]),row=int(i),name=str(saved['mutant_names'][i]),
                        peak=float(maxima[i,j]),real_max=float(realmax[j]),ratio=float(ratio[i,j])))
    pd.DataFrame(top).to_csv(args.output/'extreme_examples.csv',index=False)
    np.savez_compressed(args.output/'trajectory_moments.npz',maxima=maxima,minima=minima,logsum=logsum,
                        logsquare=logsquare,real_max=realmax,legacy_high=high,train=train,real=real)
    # Freeze analysis to a completed checkpoint; the final round may still train.
    completed=[p for p in rounds if (p.parent/'model.ckpt').exists() and (p.parent/'eval/metrics_denorm.json').exists()]
    checkpoint_path=completed[-1];r=int(checkpoint_path.parent.name.split('_')[-1])
    with open_dataset(checkpoint_path,allow_pickle=False) as z:
        current_train=z['split_train'];current_names=z['mutant_names'][current_train]
    current=train[:len(current_train)]
    assert np.array_equal(current_names,saved['mutant_names'][current])
    print(f'Predicting with completed round {r}, train={len(current)} on {args.device}',flush=True)
    model,_=load_model_from_checkpoint(str(checkpoint_path.parent/'model.ckpt'),device='cpu')
    state=model._preprocessing_state;model.to(args.device)
    model_capacity=dict(parameter_entries=sum(p.numel() for p in model.model.parameters()),
        real_degrees_of_freedom=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.model.parameters()))
    rng=np.random.RandomState(42);sampled=[];weights={}
    for group in np.unique(saved['pattern_group_codes'][current]):
        for value in (False,True):
            available=current[(saved['pattern_group_codes'][current]==group)&(high[current]==value)]
            k=min(len(available),args.train_stratum_samples)
            if k:
                chosen=rng.choice(available,k,replace=False);sampled.extend(chosen)
                weights.update({int(i):len(available)/k for i in chosen})
    infer_sets={'real':real,'synth_test':saved['split_test'],'train_stratified':np.asarray(sampled,dtype=int)}
    inference_rows=[];aggregate={};real_pred=None
    for split,indices in infer_sets.items():
        pred,target,pz,tz=predict(model,indices,saved,state,args.device,args.batch_size)
        local=error_rows(pred,target,pz,tz)
        for k,i in enumerate(indices):
            inference_rows.append(dict(split=split,row=int(i),name=str(saved['mutant_names'][i]),
                group=str(saved['pattern_group_codes'][i]),level=int(saved['module_counts'][i]),
                legacy_rejected=bool(high[i]),weight=weights.get(int(i),1.),**{key:float(value[k]) for key,value in local.items()}))
        aggregate[split]={}
        for tag,mask in (('all',np.ones(len(indices),dtype=bool)),('legacy_inside',~high[indices]),('legacy_outside',high[indices])):
            if mask.any():aggregate[split][tag]=dict(count=int(mask.sum()),**calculate_metrics(torch.tensor(pred[mask]),torch.tensor(target[mask])))
        if split=='real':
            official=json.loads((checkpoint_path.parent/'eval/metrics_denorm.json').read_text())
            expected=official['real_mutants']['all_variables']
            for metric,value in expected.items():
                if not np.isclose(aggregate[split]['all'][metric],value,rtol=3e-4,atol=3e-5):raise ValueError(f'Replayed real metric differs: {metric}')
            real_pred=pred;real_target=target
            variable_mae=np.abs(pred-target).mean((0,2));variable_mse=((pred-target)**2).mean((0,2))
            pd.DataFrame(dict(variable=names,MAE=variable_mae,MSE=variable_mse,
                             target_max=target.max((0,2)),pred_max=pred.max((0,2)))).to_csv(args.output/'real_variable_errors.csv',index=False)
            metrics=pd.DataFrame(official['per_mutant'])
            metrics['rank_score']=metrics.MAE.rank(ascending=False)+metrics.Relative_L2.rank(ascending=False)
            metrics['group']=saved['pattern_group_codes'][real]
            metrics['pred_group']=pattern_group_metadata(pred)['pattern_group_codes']
            metrics['true_clb2_max']=target[:,2].max(1);metrics['pred_clb2_max']=pred[:,2].max(1)
            metrics['true_cln2_max']=target[:,1].max(1);metrics['pred_cln2_max']=pred[:,1].max(1)
            metrics.sort_values('rank_score').to_csv(args.output/'real_badcases.csv',index=False)
        print(f'{split}: {len(indices)} predictions complete',flush=True)
    errors=pd.DataFrame(inference_rows);errors.to_csv(args.output/'prediction_errors.csv',index=False)
    train_errors=errors[errors.split=='train_stratified'];weighted=[]
    for value in (False,True):
        block=train_errors[train_errors.legacy_rejected==value]
        if len(block):weighted.append(dict(legacy_rejected=value,estimated_population=int(round(block.weight.sum())),
             sample_count=len(block),**{metric:float(np.average(block[metric],weights=block.weight)) for metric in ('MAE','MSE','normalized_MSE','hybrid')}))
    pd.DataFrame(weighted).to_csv(args.output/'weighted_training_error_groups.csv',index=False)
    worst=pd.read_csv(args.output/'real_badcases.csv').head(20)
    # Current run metrics are read only; an ongoing round is never called complete.
    performance_rows=[]
    for completed_path in completed:
        rr=int(completed_path.parent.name.split('_')[-1])
        measured=json.loads((completed_path.parent/'eval/metrics_denorm.json').read_text())
        record=dict(round=rr,train_samples=stages[rr]['train_samples'])
        for split in ('real_mutants','synth_test'):
            values=measured[split]['all_variables']
            for key,label in (('MAE','mae'),('MSE','mse'),('Relative L2','rel_l2'),('Correlation','corr')):
                record[split+'_'+label]=values[key]
        performance_rows.append(record)
    performance=pd.DataFrame(performance_rows)
    performance.to_csv(args.output/'observed_performance_curve.csv',index=False)
    report=dict(run=str(args.run.resolve()),snapshot=str(latest.resolve()),
        analyzed_training_samples=len(train),evaluated_checkpoint_round=r,evaluated_training_samples=len(current),
        final_round_evaluation_available=(args.run/'rounds/round_03/eval/metrics_denorm.json').exists(),
        real_samples=len(real),finite_negative_samples=int((minima[train].min(1)<0).sum()),
        minimum=float(minima[train].min()),stages=stages,extreme_examples=top,
        inference=aggregate,weighted_training_groups=weighted,model_capacity=model_capacity,
        top20_real_MAE_share=float(worst.MAE.sum()/pd.read_csv(args.output/'real_badcases.csv').MAE.sum()),
        scope='Observational diagnostics; no filtered retraining or model-capacity ablation; no causal attribution.',
        baseline_energy_note='Centered constant predictor energy shares, not trained-model gradients.',
        thresholds=dict(literal='any variable peak > 2*real maximum',legacy='MASS>10 OR any non-MASS peak>max(2*real maximum,2)'),
        elapsed_seconds=time.monotonic()-start)
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    make_plots(args.output,names,vars_rows,stages,scales,performance,real_pred,real_target,saved['mutant_names'][real])
    print(json.dumps(report,ensure_ascii=False),flush=True)


def make_plots(out,names,variables,stages,scales,performance,pred,target,real_names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
    frame=pd.DataFrame(variables).sort_values('synthetic_to_real_max_ratio',ascending=False).head(15)
    axes[0,0].barh(frame.variable[::-1],frame.synthetic_to_real_max_ratio[::-1]);axes[0,0].set_xscale('log')
    axes[0,0].axvline(2,color='red',ls='--');axes[0,0].set_title('Synthetic / real maximum (cumulative train)')
    axes[0,1].plot([s['train_samples'] for s in stages],[100*s['legacy_fraction'] for s in stages],marker='o')
    axes[0,1].set(xlabel='Cumulative training samples',ylabel='Rows outside old envelope (%)')
    for metric in ('real_mutants_mae','real_mutants_rel_l2'):
        axes[1,0].plot(performance.train_samples,performance[metric],marker='o',label=metric)
    axes[1,0].legend();axes[1,0].set_title('Completed rounds only')
    scale=pd.DataFrame(scales);scale=scale[scale['round']==scale['round'].max()].sort_values('std_ratio',ascending=False).head(15)
    axes[1,1].barh(scale.variable[::-1],scale.std_ratio[::-1]);axes[1,1].axvline(1,color='red',ls='--')
    axes[1,1].set_title('log1p std: all / counterfactual filtered (no retraining)')
    for ax in axes.flat:ax.grid(alpha=.2)
    for suffix in ('png','pdf'):fig.savefig(out/f'extremes_overview.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(18,12),constrained_layout=True)
    for row,mid in enumerate(('15','22','42','124')):
        i=next(i for i,name in enumerate(real_names) if str(name).split('_')[0]==mid)
        for col,j in enumerate((0,1,2,33)):
            ax=axes[row,col];ax.plot(np.linspace(0,210,target.shape[-1]),target[i,j],label='ODE')
            ax.plot(np.linspace(0,210,pred.shape[-1]),pred[i,j],label='FNO',alpha=.8)
            ax.set_title(f'#{mid} | {names[j]}');ax.grid(alpha=.2)
            if row==0 and col==0:ax.legend()
    for suffix in ('png','pdf'):fig.savefig(out/f'badcase_examples.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
