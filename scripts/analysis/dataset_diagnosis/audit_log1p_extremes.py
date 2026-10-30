#!/usr/bin/env python3
"""Quantify scaling and baseline-MSE sensitivity to existing in-support extremes.

This is a preprocessing diagnostic, not a trained-model accuracy experiment.
Statistics are fitted to the synthetic pilot only; real trajectories are used
only for reporting their transformed ranges.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_six_patterns import chunks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',required=True);p.add_argument('--real-dataset',required=True)
    p.add_argument('--real-features',required=True);p.add_argument('--reference-summary',required=True)
    p.add_argument('--output-dir',required=True);args=p.parse_args()
    base=Path(args.pilot);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((base/'manifest.json').read_text());names=manifest['variables']
    rows=sorted([json.loads(x) for x in (base/'diagnostics.jsonl').read_text().splitlines()],key=lambda x:x['index'])
    assert all(r['status']=='success' for r in rows)
    y=np.load(base/'trajectories.npy',mmap_mode='r')
    limits=np.asarray(json.loads(Path(args.reference_summary).read_text())['envelope_limits'])
    maximum=np.array([r['maxima'] for r in rows]);high=(maximum[:,1:]>limits[1:]).any(axis=1)
    stats={key:dict(sum=np.zeros(39),squares=np.zeros(39),raw_sum=np.zeros(39),n=0) for key in ('all','filtered')}
    for start in range(0,len(y),128):
        block=np.maximum(y[start:start+128],0);log=np.log1p(block)
        for key,mask in (('all',np.ones(len(block),bool)),('filtered',~high[start:start+len(block)])):
            z=log[mask];a=block[mask];s=stats[key]
            s['sum']+=z.sum(axis=(0,2));s['squares']+=(z*z).sum(axis=(0,2));s['raw_sum']+=a.sum(axis=(0,2));s['n']+=len(z)*y.shape[-1]
    for s in stats.values():
        s['mean']=s['sum']/s['n'];s['raw_mean']=s['raw_sum']/s['n']
        s['std']=np.sqrt(np.maximum((s['squares']-s['sum']*s['sum']/s['n'])/(s['n']-1),0))+1e-6
    all_stats=stats['all'];filtered=stats['filtered']
    energy={kind:np.zeros(len(y)) for kind in ('raw','log','log_z')}
    baseline_relative=np.zeros(len(y))
    for start in range(0,len(y),128):
        block=np.maximum(y[start:start+128],0);log=np.log1p(block)
        for kind,value in (('raw',block-all_stats['raw_mean'][None,:,None]),('log',log-all_stats['mean'][None,:,None]),('log_z',(log-all_stats['mean'][None,:,None])/all_stats['std'][None,:,None])):
            energy[kind][start:start+len(block)]=(value*value).sum(axis=(1,2))
            if kind=='log_z':
                per_channel=(value*value).sum(axis=2)
                baseline_relative[start:start+len(block)]=(np.sqrt(per_channel)/np.sqrt(per_channel+.1**2)).mean(axis=1)
    real_features=pd.read_csv(args.real_features);is_real=real_features.is_real.to_numpy(bool)
    selected=[]
    for start,block in chunks(args.real_dataset,128):
        mask=is_real[start:start+len(block)]
        if mask.any():selected.append(block[mask].copy())
        if start+len(block)>np.flatnonzero(is_real)[-1]:break
    real=np.maximum(np.concatenate(selected),0)
    real_log=np.log1p(real);table=[]
    for i,name in enumerate(names):
        z=(real_log[:,i]-all_stats['mean'][i])/all_stats['std'][i]
        zf=(real_log[:,i]-filtered['mean'][i])/filtered['std'][i]
        table.append(dict(variable=name,std_all=all_stats['std'][i],std_filtered=filtered['std'][i],
            std_ratio=all_stats['std'][i]/filtered['std'][i],
            same_small_error_mse_weight_ratio=(filtered['std'][i]/all_stats['std'][i])**2,
            real_z_min=float(z.min()),real_z_max=float(z.max()),real_z_abs_p99=float(np.quantile(abs(z),.99)),
            real_filtered_z_abs_p99=float(np.quantile(abs(zf),.99)),
            real_max=float(real[:,i].max()),pilot_max=float(maximum[:,i].max()),
            log_real_max=float(np.log1p(real[:,i].max())),log_pilot_max=float(np.log1p(maximum[:,i].max()))))
    frame=pd.DataFrame(table);frame.to_csv(out/'variable_scaling.csv',index=False)
    baseline_hybrid=energy['log_z']/(39*y.shape[-1])+.2*baseline_relative
    result=dict(baseline_hybrid_high_group_share=float(baseline_hybrid[high].sum()/baseline_hybrid.sum()),pilot_samples=len(y),high_amplitude_rows=int(high.sum()),
        baseline='Predict per-variable transformed pilot mean; diagnostic energy share, not actual trained loss.',
        high_group_energy_share={key:float(values[high].sum()/values.sum()) for key,values in energy.items()},
        max_std_ratio=float(frame.std_ratio.max()),max_std_ratio_variable=str(frame.loc[frame.std_ratio.idxmax(),'variable']),
        clipping_note='Finite negatives are retained in raw files; this diagnostic reproduces current log1p(max(y,0)).',
        loss_note='Current hybrid_mse_rel computes both MSE and relative L2 in standardized log space, not physical space.')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    np.savez(out/'statistics.npz',mean_all=all_stats['mean'],std_all=all_stats['std'],mean_filtered=filtered['mean'],std_filtered=filtered['std'])
    print(json.dumps(result,ensure_ascii=False));print(frame.set_index('variable').loc[['CLB2','SIC1','CDC6','CDC14','CDC15','ORI','BUD'],['std_all','std_filtered','std_ratio','same_small_error_mse_weight_ratio','real_z_abs_p99','real_filtered_z_abs_p99','log_real_max','log_pilot_max']].to_string())


if __name__=='__main__':main()
