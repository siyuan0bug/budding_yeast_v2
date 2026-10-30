"""Training-only support diagnostics for already-trained yeast surrogates."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.cross_dataset_badcases import mid,DEFAULT_OUT
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pure_random_evidence import split_indices,VARIABLES
from budding_yeast_v2.scripts.analysis.audit_real_mutant_module_counts import count_rows
from budding_yeast_v2.data.systems.budding.catalog import GENE_NAMES

_tree=ast.parse((ROOT/'data/generate_bams_ablation_datasets.py').read_text())
_skeletons=ast.literal_eval(next(n.value for n in _tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TARGETED_SKELETONS' for t in n.targets)))
TARGETED={s['desc']:frozenset(v if isinstance(v,str) else v[0] for v in s['lofs']+s['gofs']) for s in _skeletons}
HISTORICAL=set(GENE_NAMES)|{'CDC5_Proxy','SCC1','CDC28','SBF_MBF','PPX','Mcm1','ORI_BUD_SPN'}


def genes(name):
    if str(name).startswith(('BAMS_TARGET_HARD_','BAMS_TARGET_HYPO_')):
        return next((value for key,value in TARGETED.items() if str(name).endswith('_'+key)),frozenset())
    result=[]
    for raw,_ in re.findall(r'([A-Za-z0-9_]+)\[([^\]]+)\]',str(name)):
        gene=next((g for g in sorted(HISTORICAL,key=len,reverse=True) if raw==g or raw.endswith('_'+g)),raw)
        if gene!='MDT':result.append(gene)
    return frozenset(result)


def get_split(row,cache):
    import torch
    checkpoint=torch.load(row.checkpoint,map_location='cpu',weights_only=False,mmap=True) if row.checkpoint and Path(row.checkpoint).exists() else {}
    pp=checkpoint.get('preprocessing_contract')
    if pp:
        split={k:np.asarray(pp['split'][k],dtype=int) for k in ('train','val','test','real')}
        return split,'checkpoint_preprocessing',pp
    if 'split_train' in cache:
        split={k:cache['split_'+k] for k in ('train','val','test','real')}
        return split,'stored_npz_split',None
    import subprocess
    config=json.loads((Path(row.run)/'config.json').read_text());commit=config.get('git_commit','')
    source=subprocess.run(['git','show',f'{commit}:data/yeast_datamodule.py'],cwd=ROOT,capture_output=True,text=True).stdout if commit else ''
    integer_only="r'^\\d+_'" in source
    split_names=cache['names'].astype(object).copy()
    if integer_only:
        for i,name in enumerate(split_names):
            condition=re.match(r'^(\d+\.\d+)_',str(name))
            if condition:split_names[i]='SYNTH_LEGACY_DECIMAL_'+str(name)
    real,synth,train,val,test=split_indices(split_names,cache['patterns'],int(row.seed))
    return dict(real=real,train=train,val=val,test=test),('git_confirmed_legacy_integer_prefix' if integer_only else 'reconstructed_legacy_pattern_seed'),None


def nearest(x,q,k=20):
    x=np.asarray(x,dtype='float64');q=np.asarray(q,dtype='float64')
    matrix=np.maximum((x*x).sum(1)[None]+(q*q).sum(1)[:,None]-2*q@x.T,0)
    k=min(k,len(x));idx=np.argpartition(matrix,k-1,axis=1)[:,:k]
    dist=np.take_along_axis(matrix,idx,1);order=np.argsort(dist,axis=1)
    return np.sqrt(np.take_along_axis(dist,order,1)),np.take_along_axis(idx,order,1)


def analyze(row,cache_path,out):
    out=out/'support'/row.run_id;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():return
    with np.load(cache_path,allow_pickle=False) as saved:cache={k:saved[k] for k in saved.files}
    split,provenance,pp=get_split(row,cache)
    train,real=split['train'],split['real'];names=cache['names'];p=cache['params'].astype('float64');ic=cache['ic'].astype('float64')
    assert not set(train)&set(real) and cache['is_real'][real].all() 
    ps=p[train].std(0)+1e-6;pm=p[train].mean(0)
    f=(p-pm)/ps;pdists,pidx=nearest(f[train],f[real])
    # Common audit coordinates: per-parameter Z-score plus log1p/Z initial
    # states. Use checkpoint statistics when present; otherwise reconstruct
    # synthetic statistics. This is not the actual nonlinear encoder geometry
    # of legacy asinh/log-parameter models; their errors use separate strict
    # checkpoint replay. Compare these distances within runs, not across norms.
    if pp:
        model_pm=pp['p_mean'].numpy().reshape(-1);model_ps=pp['p_std'].numpy().reshape(-1)
        ym=pp['mean_y'].numpy().reshape(-1);ys=pp['std_y'].numpy().reshape(-1)
    else:
        syn=np.ones(len(names),dtype=bool);syn[real]=False
        model_pm=p[syn].mean(0);model_ps=p[syn].std(0,ddof=1)+1e-6
        ym,ys=cache['global_y_mean'],cache['global_y_std']
        missed=np.flatnonzero(cache['is_real'] & syn)
        if len(missed):
            allreal=np.flatnonzero(cache['is_real']);local=[int(np.flatnonzero(allreal==i)[0]) for i in missed]
            x=np.log1p(np.maximum(cache['real_y'][local],0)).astype('float64')
            n0=int((~cache['is_real']).sum())*500;n1=len(missed)*500
            total=ym*n0+x.sum((0,2));square=(ys-1e-6)**2*(n0-1)+ym*ym*n0+(x*x).sum((0,2))
            ym=total/(n0+n1);ys=np.sqrt(np.maximum((square-total*total/(n0+n1))/(n0+n1-1),0))+1e-6
    full=np.concatenate(((p-model_pm)/model_ps,(np.log1p(np.maximum(ic,0))-ym)/ys),axis=1)
    idists,iidx=nearest(full[train],full[real])
    emb=cache['embedding'].astype('float64');em=emb[train].mean(0);es=emb[train].std(0)+1e-6
    tdists,tidx=nearest(((emb-em)/es)[train],((emb-em)/es)[real])
    real_definition={r['identity']:r for r in count_rows('budding')[0]}
    gene_sets=[frozenset(real_definition[mid(name)]['modules'].split(';'))-{''} if cache['is_real'][i] else genes(name) for i,name in enumerate(names)];allgenes=sorted(HISTORICAL-{'MDT'});bit={g:1<<i for i,g in enumerate(allgenes)}
    masks=np.array([sum(bit.get(g,0) for g in gs) for gs in gene_sets],dtype='uint64')
    pnames=cache['param_names'];mdt_idx=int(np.flatnonzero(pnames=='mdt')[0]);pt=p[train];pmin,pmax=pt.min(0),pt.max(0)
    zeros=pt==0;pattern_counts=Counter(cache['code'][train]);p32=p.astype('float32');i32=np.log1p(np.maximum(ic.astype('float32'),0))
    baseline_mean=p[~cache['is_real']].mean(0)
    peak99=np.quantile(cache['maxima'][train],.99,axis=0)
    peakmax=cache['maxima'][train].max(0)
    p32train=p32[train];i32train=i32[train]
    rows=[];neighbors=[]
    for pos,idx in enumerate(real):
        condition=mid(names[idx]);realgenes=frozenset(real_definition[condition]['modules'].split(';'))-{''}
        target_mask=sum(bit.get(g,0) for g in realgenes);exact_genes=masks[train]==target_mask
        same_carbon=np.isclose(pt[:,mdt_idx],p[idx,mdt_idx],rtol=0,atol=1e-5)
        zero_mask=p[idx]==0;face=zeros[:,zero_mask].all(1) if zero_mask.any() else np.ones(len(train),bool)
        exact_params=(p32train==p32[idx]).all(1)
        exact_input=exact_params & (i32train==i32[idx]).all(1)
        widths=np.maximum(.25*np.abs(p[idx]),.05*np.maximum(np.abs(baseline_mean),1e-8))
        box25=(np.abs(pt-p[idx])<=widths+1e-10).all(1)&face&same_carbon
        box50=(np.abs(pt-p[idx])<=2*widths+1e-10).all(1)&face&same_carbon
        outside=(p[idx]<pmin-1e-9)|(p[idx]>pmax+1e-9)
        same=cache['code'][train]==cache['code'][idx]
        neighborhood=iidx[pos];same_near=(cache['code'][train[neighborhood]]==cache['code'][idx])
        row_result=dict(run_id=row.run_id,kind=row.kind,dataset=row.dataset,model=row.model,strategy=row.strategy,round=int(row['round']),
            mutant_id=condition,Mutant=names[idx],pattern=cache['code'][idx],divisions=int(cache['divisions'][idx]),
            module_count=len(realgenes),modules=';'.join(sorted(realgenes)),split_provenance=provenance,n_train=len(train),
            train_same_pattern=pattern_counts[cache['code'][idx]],train_same_module_set=int(exact_genes.sum()),
            train_same_modules_carbon=int((exact_genes&same_carbon).sum()),train_same_zero_face=int(face.sum()),
            train_exact_parameters=int(exact_params.sum()),train_exact_model_inputs=int(exact_input.sum()),
            train_local25=int(box25.sum()),train_local50=int(box50.sum()),train_local25_same_pattern=int((box25&same).sum()),
            param_nn_distance=float(pdists[pos,0]),input_nn_distance=float(idists[pos,0]),input_top20_mean=float(idists[pos].mean()),
            trajectory_feature_nn_distance=float(tdists[pos,0]),input_near20_same_pattern_fraction=float(same_near.mean()),
            input_near20_divisions_min=int(cache['divisions'][train[neighborhood]].min()),input_near20_divisions_max=int(cache['divisions'][train[neighborhood]].max()),
            out_of_train_range_parameters=';'.join(pnames[outside]),
            peak_exceeds_train_max=';'.join(np.asarray(VARIABLES)[cache['maxima'][idx]>peakmax+1e-5]),
            peak_exceeds_train_p99=';'.join(np.asarray(VARIABLES)[cache['maxima'][idx]>peak99+1e-5]),
            truth_clb2_max=float(cache['maxima'][idx,2]),truth_ori_max=float(cache['maxima'][idx,33]),
            truth_sha256=hashlib.sha256(cache['real_y'][np.flatnonzero(np.flatnonzero(cache['is_real'])==idx)[0]].astype('<f4').tobytes()).hexdigest(),
            parameter_sha256=hashlib.sha256(p[idx].astype('<f4').tobytes()).hexdigest(),
            initial_sha256=hashlib.sha256(ic[idx].astype('<f4').tobytes()).hexdigest())
        rows.append(row_result)
        for metric,indices,distances in (('input',iidx,idists),('parameter',pidx,pdists),('trajectory_feature',tidx,tdists)):
            for rank,(local,distance) in enumerate(zip(indices[pos,:5],distances[pos,:5]),1):
                j=train[local];neighbors.append(dict(run_id=row.run_id,mutant_id=condition,metric=metric,rank=rank,
                    neighbor_name=names[j],neighbor_row=int(j),distance=float(distance),neighbor_pattern=cache['code'][j],
                    neighbor_divisions=int(cache['divisions'][j]),neighbor_clb2_max=float(cache['maxima'][j,2]),neighbor_ori_max=float(cache['maxima'][j,33])))
    pd.DataFrame(rows).to_csv(out/'real_support.csv',index=False);pd.DataFrame(neighbors).to_csv(out/'neighbors.csv',index=False)
    (out/'summary.json').write_text(json.dumps(dict(run_id=row.run_id,dataset=row.dataset,split_provenance=provenance,n_train=len(train),real=len(real),decimal_benchmark_conditions_in_train=names[train[cache['is_real'][train]]].tolist()),indent=2))
    print('support done',row.dataset_tag,row.model,row.strategy,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--wait-for-features',action='store_true');args=parser.parse_args()
    runs=pd.read_csv(args.output/'primary_runs.csv').fillna('')
    from concurrent.futures import ProcessPoolExecutor
    import time,zipfile
    pending=[row for _,row in runs.iterrows()];futures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        while pending:
            waiting=[]
            for row in pending:
                cache=args.output/'features'/f'{hashlib.sha256(row.dataset.encode()).hexdigest()[:12]}.npz'
                if cache.exists() and zipfile.is_zipfile(cache):futures.append(executor.submit(analyze,row,str(cache),args.output))
                elif args.wait_for_features:waiting.append(row)
                else:raise FileNotFoundError(cache)
            pending=waiting
            for future in futures:
                if future.done():future.result()
            if pending:time.sleep(10)
        for future in futures:future.result()
    support=pd.concat([pd.read_csv(p,dtype={'mutant_id':str}) for p in (args.output/'support').glob('*/real_support.csv')],ignore_index=True)
    support=support[support.run_id.isin(runs.run_id)]
    metrics=pd.read_csv(args.output/'all_mutant_metrics.csv',dtype={'mutant_id':str})
    support.mutant_id=support.mutant_id.astype(str)
    joined=support.merge(metrics[['run_id','mutant_id','MAE','MSE','Relative_L2','severity_rank','is_top20']],on=['run_id','mutant_id'],validate='one_to_one')
    joined.to_csv(args.output/'support_and_errors.csv',index=False)
    records=[]
    for run,group in joined.groupby('run_id'):
        for feature in ('train_local25','train_same_module_set','train_same_pattern','param_nn_distance','input_nn_distance','trajectory_feature_nn_distance','input_near20_same_pattern_fraction'):
            if group[feature].nunique()>1:
                rho,pvalue=spearmanr(group[feature],group.Relative_L2)
                records.append(dict(run_id=run,feature=feature,spearman_rho=rho,pvalue=pvalue,n=len(group)))
    pd.DataFrame(records).to_csv(args.output/'support_error_correlations.csv',index=False)


if __name__=='__main__':main()
