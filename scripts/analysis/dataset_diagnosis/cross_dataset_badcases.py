#!/usr/bin/env python3
"""Read-only cross-dataset badcase inventory, cached dynamics and train support."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.dataset_archive import open_dataset, resolve_dataset_path
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pure_random_evidence import data_chunks,split_indices,VARIABLES
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_six_patterns import features

DEFAULT_OUT=Path('/home/users/zsy/mydata/budding_yeast/analysis/badcase_audit_2026-10-27')


def identity(path):
    path=resolve_dataset_path(path);s=path.stat()
    return dict(path=str(path.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns)


def mid(name):
    m=re.match(r'^(\d+(?:\.\d+)?)_',str(name))
    return m.group(1) if m else None


def locate_config(path):
    for p in path.parents:
        if (p/'config.json').is_file():return p
    return None


def inventory(args):
    manifest_path=args.output/'raw_manifest.json'
    if not manifest_path.exists():
        csvs=[];al=[]
        for base,dirs,files in os.walk(args.experiments):
            dirs[:]=[d for d in dirs if d not in ('wandb','real_mutants','acquisitions','data','checkpoints')]
            base=Path(base)
            if 'per_mutant_metrics.csv' in files:csvs.append(str(base/'per_mutant_metrics.csv'))
            if 'metrics_denorm.json' in files and 'al_budding' in base.parts:al.append(str(base/'metrics_denorm.json'))
        manifest_path.write_text(json.dumps(dict(csv=csvs,al=al),indent=2))
    manifest=json.loads(manifest_path.read_text())
    runs=[];frames=[];skipped=[]
    sources=[(Path(x),'static') for x in manifest['csv'] if Path(x).relative_to(args.experiments).parts[0] not in ('al_pure_random_1m','fission_random_50000_seed20000_modular')]
    sources += [(Path(x),'al') for x in manifest['al']]
    for source,kind in sources:
        run=locate_config(source)
        if run is None:skipped.append(dict(source=str(source),reason='missing config'));continue
        config=json.loads((run/'config.json').read_text());a=config.get('args',config)
        if kind=='static':
            df=pd.read_csv(source)
            path=Path(a.get('dataset_name',''))
            if not path.is_absolute():path=Path(a.get('data_dir',str(args.data)))/path
            if not path.exists():path=args.data/path.name
            round_no=-1
        else:
            content=json.loads(source.read_text());df=pd.DataFrame(content.get('per_mutant',[]))
            path=source.parent.parent/'dataset.npz'
            if not path.exists() and path.with_suffix('.json').exists():path=path.with_suffix('.json')
            round_no=int(source.parent.parent.name.split('_')[-1])
        if 'Mutant' not in df or not path.exists():
            skipped.append(dict(source=str(source),reason='missing metrics/dataset',dataset=str(path)));continue
        df=df.copy();df['mutant_id']=df.Mutant.map(mid)
        if len(df)<20 or df.mutant_id.isna().any() or df.mutant_id.duplicated().any():
            skipped.append(dict(source=str(source),reason='invalid real-condition identity',rows=len(df)));continue
        run_id=hashlib.sha256(str(source).encode()).hexdigest()[:12]
        ckpts=list((run/'checkpoints').glob('*.ckpt')) if kind=='static' else [path.parent/'model.ckpt']
        ckpts=sorted(ckpts,key=lambda p:(not 'final' in p.name,not 'best' in p.name,p.name))
        record=dict(run_id=run_id,kind=kind,dataset_tag=run.relative_to(args.experiments).parts[0],dataset=str(path.resolve()),
            run=str(run),model_directory=run.relative_to(args.experiments).parts[1],source=str(source),model=a.get('model'),loss=a.get('loss_type'),norm=a.get('norm_type','log1p'),
            width=a.get('width'),batch_size=a.get('batch_size'),n_blocks=a.get('n_blocks'),epochs=a.get('max_epochs'),lr=a.get('lr'),seed=int(a.get('seed',42)),round=round_no,strategy=a.get('al_strategy','none'),
            checkpoint=str(ckpts[0]) if ckpts else '',real_conditions=len(df),timestamp=config.get('timestamp',''),
            mean_MAE=float(df.MAE.mean()),mean_Relative_L2=float(df.Relative_L2.mean()),mean_MSE=float(df.MSE.mean()),
            complete_rounds=round_no+1 if kind=='al' else 0)
        runs.append(record)
        for key in ('MAE','Relative_L2','MSE'):df['rank_'+key]=df[key].rank(ascending=False,method='min')
        df['severity_rank']=(df.rank_MAE+df.rank_Relative_L2)/2
        df['is_top20']=df.severity_rank.rank(method='first')<=20
        for key in ('run_id','kind','dataset_tag','model','loss','norm','seed','round','strategy'):df[key]=record[key]
        frames.append(df)
    runs=pd.DataFrame(runs);all_metrics=pd.concat(frames,ignore_index=True)
    runs.to_csv(args.output/'runs.csv',index=False)
    all_metrics.to_csv(args.output/'all_mutant_metrics.csv',index=False)
    all_metrics[all_metrics.is_top20].to_csv(args.output/'top20_per_run.csv',index=False)
    (args.output/'inventory_skipped.json').write_text(json.dumps(skipped,indent=2))
    primary=[]
    for path,group in runs[runs.kind.eq('static')].groupby('dataset'):
        # Fixed comparable architecture/loss/seed; do not select by test error.
        score=(group.model.ne('biom_refiner_plus').astype(int)*100 +group.loss.ne('hybrid_mse_rel').astype(int)*10+group.seed.ne(42).astype(int)*5+group.norm.ne('log1p').astype(int)+group.dataset_tag.ne(Path(path).stem).astype(int)*2+group.model_directory.ne(group.model).astype(int)*3)
        primary.append(group.loc[score.sort_values(kind='stable').index[0]])
    # AL dynamics/support: latest measured round of each model/strategy.
    for _,group in runs[runs.kind.eq('al')].groupby(['run','model','strategy']):primary.append(group.sort_values('round').iloc[-1])
    pd.DataFrame(primary).to_csv(args.output/'primary_runs.csv',index=False)
    comparable_ids=[row['run_id'] for row in primary if row['kind']=='static']
    selected=all_metrics[all_metrics.run_id.isin(comparable_ids)]
    ranked=selected.groupby('mutant_id').agg(Mutant=('Mutant','first'),datasets=('run_id','nunique'),
        top20_datasets=('is_top20','sum'),median_MAE=('MAE','median'),median_Relative_L2=('Relative_L2','median'),
        median_MSE=('MSE','median'),mean_severity_rank=('severity_rank','mean')).sort_values(['mean_severity_rank'])
    ranked.to_csv(args.output/'cross_dataset_badcase_ranking.csv')
    print(f'inventory: {len(runs)} metric files, {len(primary)} primary runs; {len(skipped)} skipped',flush=True)


def feature_cache(job):
    path,out=map(Path,job);before=identity(path);cache=out/'features'/f'{hashlib.sha256(str(path).encode()).hexdigest()[:12]}.npz'
    if cache.exists():
        with open_dataset(cache,allow_pickle=False) as z:
            if json.loads(str(z['identity']))==before:return str(cache)
    start_time=time.monotonic()
    with open_dataset(path,allow_pickle=True) as z:
        names=z['mutant_names'].astype(str);p=z['params'];real=z['is_real'] if 'is_real' in z else np.array([mid(n) is not None for n in names]);patterns=z['pattern_labels'].astype(str)
        keys=z['param_names'].astype(str) if 'param_names' in z else np.array(__import__('budding_yeast_v2.data.systems.budding.parameters',fromlist=['PARAM_KEYS']).PARAM_KEYS)
        split_fields={key:z[key] for key in z.files if key.startswith('split_') and key in ('split_train','split_val','split_test','split_real')}
    n=len(names);ic=np.empty((n,39),dtype='float32');maxima=np.empty_like(ic);ends=np.empty_like(ic)
    emb=np.empty((n,9*21),dtype='float32');div=np.empty(n,dtype='int16');code=np.empty(n,dtype='<U3');critical=np.empty(n,dtype=bool)
    real_y=np.empty((real.sum(),39,500),dtype='float32');real_pos={idx:k for k,idx in enumerate(np.flatnonzero(real))}
    sums=np.zeros(39);squares=np.zeros(39);steps=0;time_indices=np.round(np.linspace(0,499,21)).astype(int)
    for start,block in data_chunks(path,256):
        if block.shape[1]!=39:block=block.swapaxes(1,2)
        if block.shape[2]!=500:raise ValueError(f'Unexpected grid: {path} {block.shape}')
        stop=start+len(block);f,initial=features(block)
        ic[start:stop]=initial;maxima[start:stop]=block.max(2);ends[start:stop]=block[:,:,-1]
        div[start:stop]=f['divisions'];code[start:stop]=f['pattern_code'];critical[start:stop]=f['is_bifurcation_critical']
        transformed=np.log1p(np.maximum(block,0)).astype(np.float32)
        emb[start:stop]=transformed[:,[0,1,2,3,4,22,32,33,35]][:,:,time_indices].reshape(len(block),-1)
        synthetic=~real[start:stop]
        x=transformed[synthetic].astype(np.float64);sums+=x.sum((0,2));squares+=(x*x).sum((0,2));steps+=len(x)*500
        for idx in np.flatnonzero(real[start:stop]):real_y[real_pos[start+idx]]=block[idx]
        if start%25600==0:print(f'features {path.name}: {stop}/{n}',flush=True)
    mean=sums/steps;std=np.sqrt(np.maximum((squares-sums*sums/steps)/(steps-1),0))+1e-6
    assert identity(path)==before
    cache.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(cache,identity=json.dumps(before),names=names,params=p,param_names=keys,is_real=real,patterns=patterns,
        ic=ic,maxima=maxima,ends=ends,embedding=emb,divisions=div,code=code,critical=critical,real_y=real_y,
        global_y_mean=mean,global_y_std=std,**split_fields)
    print(f'features done {path.name}: {time.monotonic()-start_time:.1f}s',flush=True)
    return str(cache)


def caches(args):
    runs=pd.read_csv(args.output/'primary_runs.csv');paths=sorted(runs.dataset.unique())
    results={}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(feature_cache,(path,str(args.output))):path for path in paths}
        for future in as_completed(futures):results[futures[future]]=future.result()
    (args.output/'feature_manifest.json').write_text(json.dumps(results,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/experiments'))
    parser.add_argument('--data',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--phase',choices=['inventory','features'],required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    {'inventory':inventory,'features':caches}[args.phase](args)


if __name__=='__main__':main()
