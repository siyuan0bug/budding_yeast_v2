"""Compare specified good cases with badcases using one frozen AL checkpoint.

Read-only model/data audit. The optional ODE probes are diagnostic trajectories,
never appended to an experiment's training set.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.cross_dataset_badcases import DEFAULT_OUT,mid
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pool_badcase_coverage import box
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pure_random_evidence import data_chunks
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_six_patterns import features


def decode_pairs(name):
    return tuple(sorted((g.removeprefix('MASS_RAND_'),mode) for g,mode in re.findall(r'([A-Za-z0-9_]+)\[([^\]]+)\]',str(name))))


def probe(job):
    from budding_yeast_v2.data.systems.budding.simulation import simulate_mutant,prepare_initial_conditions
    label,overrides=job
    _,y,bio,pattern=simulate_mutant('diagnostic_'+label,overrides)
    if y is None:raise RuntimeError(f'ODE probe failed: {label}')
    p,ic,_,_=prepare_initial_conditions(overrides)
    # The 141 exported features also include these three initial-state fields;
    # prepare_initial_conditions intentionally keeps them outside the RHS dict.
    p.update(init_CDH1T=float(ic[21]),init_CDH1=float(ic[22]),init_BUB2=float(ic[38]))
    return label,overrides,p,ic,y


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--run-id',default='9c82a7fe845c')
    parser.add_argument('--ids',nargs='+',choices=['7','8','9','10','15','22'],default=['7','8','9','10','15','22'])
    parser.add_argument('--pool',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/budding_pool_1000000_seed42_al.npz'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();out=args.output or args.audit_root/'good_vs_bad_7_10';out.mkdir(parents=True,exist_ok=True)
    runs=pd.read_csv(args.audit_root/'primary_runs.csv');run=runs[runs.run_id.eq(args.run_id)].iloc[0]
    key=hashlib.sha256(run.dataset.encode()).hexdigest()[:12]
    with np.load(args.audit_root/'features'/f'{key}.npz',allow_pickle=False) as z:
        p=z['params'];ic=z['ic'];names=z['names'];train=z['split_train'];real=np.flatnonzero(z['is_real']);keys=z['param_names'];true=z['real_y']
    ids=[mid(n) for n in names[real]];lookup={s:real[ids.index(s)] for s in args.ids};wt=p[real[ids.index('1')]];mdt=int(np.flatnonzero(keys=='mdt')[0])
    desired={
        '7':(('CDH1','KO'),('CLN2','GAL-OE')),
        '8':(('CLN3','KO'),),'9':(('CLN3','GAL-OE'),),'10':(('BCK2','KO'),),
        '15':(('BCK2','KO'),('CLN2','MC'),('CLN3','KO')),
        '22':(('CLB5','MC'),('CLN2','KO'),('CLN3','KO')),
    }
    summary=[]
    with np.load(args.pool,allow_pickle=False) as z:poolp=z['params'];poolnames=z['mutant_names']
    desired_masks={s:np.zeros(len(poolnames),dtype=bool) for s in args.ids}
    gene_masks={s:np.zeros(len(poolnames),dtype=bool) for s in args.ids}
    targets={s:frozenset(g for g,_ in desired[s]) for s in args.ids}
    for j,name in enumerate(poolnames):
        pairs=decode_pairs(name);genes=frozenset(g for g,_ in pairs)
        for s in args.ids:
            gene_masks[s][j]=genes==targets[s];desired_masks[s][j]=pairs==desired[s]
    train_pairs=[decode_pairs(n) for n in names[train]]
    base=pd.read_csv(args.audit_root/'validated_case_evidence.csv',dtype={'mutant_id':str});base=base[base.run_id.eq(args.run_id)].set_index('mutant_id')
    for s in args.ids:
        q=p[lookup[s]];tm=np.array([x==desired[s] for x in train_pairs]);tg=np.array([frozenset(g for g,_ in x)==targets[s] for x in train_pairs]);carbon=np.isclose(poolp[:,mdt],q[mdt],atol=1e-5,rtol=0)
        r=dict(mutant_id=s,Mutant=names[lookup[s]],pool_same_modules=int(gene_masks[s].sum()),pool_same_modes=int(desired_masks[s].sum()),pool_same_modes_carbon=int((desired_masks[s]&carbon).sum()),pool_local25=int(box(poolp,q,wt,mdt,.25).sum()),train_same_modules=int(tg.sum()),train_same_modes=int(tm.sum()),train_same_modes_carbon=int((tm&np.isclose(p[train,mdt],q[mdt],atol=1e-5,rtol=0)).sum()),train_local25=int(box(p[train],q,wt,mdt,.25).sum()))
        contains=np.array([set(desired[s]).issubset(x) for x in train_pairs])
        r['train_contains_required_modes']=int(contains.sum())
        r['train_contains_required_modes_carbon']=int((contains&np.isclose(p[train,mdt],q[mdt],atol=1e-5,rtol=0)).sum())
        for k in ['MAE','MSE','Relative_L2','input_nn_distance','trajectory_feature_nn_distance','input_near20_same_pattern_fraction','train_exact_model_inputs']:r[k]=base.loc[s,k]
        summary.append(r)
    pd.DataFrame(summary).to_csv(out/'coverage_comparison.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    del poolp,poolnames,desired_masks,gene_masks
    neighbors=pd.read_csv(args.audit_root/'support'/args.run_id/'neighbors.csv',dtype={'mutant_id':str})
    selected=neighbors[neighbors.mutant_id.isin(args.ids)&neighbors.metric.isin(['input','trajectory_feature'])&neighbors['rank'].le(3)].copy()
    needed=set(selected.neighbor_row.astype(int));selected_y={}
    excerpt=out/'neighbor_trajectories.npz'
    if excerpt.exists():
        with np.load(excerpt,allow_pickle=False) as z:selected_y={int(i):y for i,y in zip(z['row_indices'],z['data'])}
    if not needed.issubset(selected_y):
        for start,block in data_chunks(run.dataset,256):
            for i in sorted(needed):
                if start<=i<start+len(block):selected_y[i]=block[i-start].copy()
            if needed.issubset(selected_y):break
        np.savez_compressed(excerpt,row_indices=sorted(selected_y),data=np.stack([selected_y[i] for i in sorted(selected_y)]))
    import torch
    from budding_yeast_v2.eval import load_model_from_checkpoint
    from budding_yeast_v2.data.normalization import normalize_y,normalize_p,denormalize_y
    from budding_yeast_v2.utils.metrics import calculate_metrics
    torch.set_num_threads(4)
    checkpoint=torch.load(run.checkpoint,map_location='cpu',weights_only=False,mmap=True);pp=checkpoint['preprocessing_contract']
    model,_=load_model_from_checkpoint(run.checkpoint,device='cpu',graph_load_policy='checkpoint');model.to(args.device).eval()
    def predict(params,initial):
        p0=torch.from_numpy(np.asarray(params,dtype='float32'));y0=torch.from_numpy(np.asarray(initial,dtype='float32'))[:,:,None]
        yn=normalize_y(y0,pp['mean_y'],pp['std_y'],pp['norm_type']).expand(-1,-1,500)
        grid=torch.linspace(0,1,500)[None,None].expand_as(yn);x=torch.stack((yn,grid),2)
        pn=normalize_p(p0,pp['p_mean'],pp['p_std'],pp['norm_type'])
        with torch.inference_mode():output=model(x.to(args.device),pn.to(args.device))
        return denormalize_y(output.cpu(),pp['mean_y'],pp['std_y'],pp['norm_type']).numpy()
    rows=sorted(needed);prediction=predict(p[rows],ic[rows]);pred_lookup=dict(zip(rows,prediction));records=[]
    for _,r in selected.iterrows():
        i=int(r.neighbor_row);target=true[ids.index(r.mutant_id)]
        score=calculate_metrics(torch.from_numpy(np.maximum(selected_y[i],0)),torch.from_numpy(np.maximum(target,0)))
        fit=calculate_metrics(torch.from_numpy(pred_lookup[i]),torch.from_numpy(np.maximum(selected_y[i],0)))
        diff=p[i].astype('float64')-p[lookup[r.mutant_id]].astype('float64');changed=np.abs(diff)>1e-10
        record=r.to_dict();record.update(neighbor_to_real_RelL2=score['Relative L2'] if 'Relative L2' in score else score['Relative_L2'],neighbor_fit_RelL2=fit['Relative L2'] if 'Relative L2' in fit else fit['Relative_L2'],parameter_differences=json.dumps({str(k):{'real':float(a),'train':float(b)} for k,a,b in zip(keys[changed],p[lookup[r.mutant_id],changed],p[i,changed])}))
        records.append(record)
    pd.DataFrame(records).to_csv(out/'neighbor_fidelity.csv',index=False)
    from budding_yeast_v2.data.systems.budding.parameters import get_default_params
    from budding_yeast_v2.data.systems.budding.real_mutants import REAL_MUTANTS
    defaults=get_default_params();jobs=[]
    for n,c,b in itertools.product((1,5),(False,True),(False,True)):
        ovr={'C0':0. if c else defaults['C0'],'B0':0. if b else defaults['B0'],'ksn2pp':defaults['ksn2pp']*n}
        jobs.append((f'CLN2x{n}_cln3KO{int(c)}_bck2KO{int(b)}',ovr))
    for fraction in (.001,.01,.1,.5):
        jobs.append((f'15_restore_BCK2_{fraction}',dict(REAL_MUTANTS['15'][1],B0=defaults['B0']*fraction)))
        jobs.append((f'15_restore_CLN3_{fraction}',dict(REAL_MUTANTS['15'][1],C0=defaults['C0']*fraction)))
    probes=out/'probe_trajectories.npz'
    if probes.exists():
        with np.load(probes,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['labels'],[label for label,_ in jobs])
            probe_data=z['data'];probe_p=z['params'];probe_ic=z['initial']
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:result=list(executor.map(probe,jobs))
        probe_data=np.stack([r[4] for r in result]);probe_p=np.array([[r[2][k] for k in keys] for r in result],dtype='float32');probe_ic=np.stack([r[3] for r in result])
        np.savez_compressed(probes,data=probe_data,params=probe_p,initial=probe_ic,labels=np.array([r[0] for r in result]))
    yp=predict(probe_p,probe_ic);ft,_=features(probe_data);fp,_=features(yp);records=[]
    for i,(label,overrides) in enumerate(jobs):
        score=calculate_metrics(torch.from_numpy(yp[i]),torch.from_numpy(np.maximum(probe_data[i],0)))
        records.append(dict(label=label,overrides=json.dumps(overrides),true_pattern=ft['pattern_code'][i],pred_pattern=fp['pattern_code'][i],true_divisions=int(ft['divisions'][i]),pred_divisions=int(fp['divisions'][i]),true_cln2_max=float(probe_data[i,1].max()),pred_cln2_max=float(yp[i,1].max()),true_ori_max=float(probe_data[i,33].max()),pred_ori_max=float(yp[i,33].max()),**{k.replace(' ','_'):v for k,v in score.items()}))
    pd.DataFrame(records).to_csv(out/'factorial_and_restore_probes.csv',index=False)
    np.savez_compressed(out/'probe_predictions.npz',predictions=yp)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(14,9),layout='constrained')
    panel=[0,1,3,4,5,7,9,11,15]
    for ax,i in zip(axes.flat,panel):
        ax.plot(np.linspace(0,210,500),probe_data[i,1],label='ODE',color='black');ax.plot(np.linspace(0,210,500),yp[i,1],label='Frozen FNO',color='#c43c39',ls='--');ax.set_title(jobs[i][0],fontsize=9);ax.set_xlabel('min');ax.set_ylabel('CLN2');ax.grid(alpha=.2)
    axes[0,0].legend();fig.savefig(out/'cln2_factorial_comparison.png',dpi=170);plt.close(fig)
    print(pd.DataFrame(records).drop(columns='overrides').to_string(index=False),flush=True)
    (out/'provenance.json').write_text(json.dumps(dict(run_id=args.run_id,dataset=run.dataset,checkpoint=run.checkpoint,pool=str(args.pool),probe_count=len(jobs),training_modified=False),indent=2))


if __name__=='__main__':main()
