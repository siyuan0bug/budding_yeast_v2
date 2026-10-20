#!/usr/bin/env python3
"""Inspect full trajectories of feature/input neighbors for selected badcases.

Neighbors were selected from current synthetic TRAIN only by the audit. These
are nearest in the stated feature metrics, not globally nearest trajectories.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_six_patterns import REFERENCE,chunks,identity
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.summarize_six_pattern_audit import select_badcases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/mutant_aware_pure_random_100000_210m_500s.npz'))
    p.add_argument('--audit-dir',type=Path,required=True,help='The per-dataset directory, not audit root')
    p.add_argument('--eval-csv',type=Path,default=REFERENCE)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--top-k',type=int,default=20)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    before=identity(args.dataset)
    import json
    summary=json.loads((args.audit_dir/'summary.json').read_text())
    if summary['identity'] != before:raise ValueError('Audit refers to a different source version')
    frame=pd.read_csv(args.audit_dir/'samples.csv.gz',usecols=['row_index','mutant_name','split'])
    if frame.mutant_name.duplicated().any():raise ValueError('Ambiguous sample names')
    indexed=frame.set_index('mutant_name')
    chosen=select_badcases(pd.read_csv(args.eval_csv),args.top_k).head(args.top_k)
    support=pd.read_csv(args.audit_dir/'real_training_support.csv').set_index('mutant_name')
    requests=[]
    for name in chosen.Mutant:
        for kind in ('dynamic_feature','full_input'):
            nn=support.loc[name,kind+'_nn_name']
            if indexed.loc[nn,'split'] != 'train_synth':raise ValueError('Neighbor is outside train')
            requests.append((name,nn,kind))
    wanted={int(indexed.loc[n,'row_index']) for pair in requests for n in pair[:2]}
    selected={}
    for start,b in chunks(args.dataset,256):
        b=b if b.shape[1]==39 else b.swapaxes(1,2)
        for idx in wanted:
            if start<=idx<start+len(b):selected[idx]=b[idx-start].copy()
        if start%20480==0:print(f'Neighbor extraction {start+len(b):,}',flush=True)
        if len(selected)==len(wanted):break
    if len(selected)!=len(wanted) or identity(args.dataset)!=before:raise ValueError('Incomplete or changing source')
    rows=[];pairs=[];pairnames=[]
    for name,nn,kind in requests:
        a=selected[int(indexed.loc[name,'row_index'])].astype(np.float64)
        b=selected[int(indexed.loc[nn,'row_index'])].astype(np.float64)
        diff=b-a
        rel=np.sqrt(np.sum(diff**2,axis=1)/np.maximum(np.sum(a**2,axis=1),.1**2*a.shape[1]))
        rows.append(dict(mutant_name=name,neighbor_name=nn,neighbor_metric=kind,
            all39_MAE=np.abs(diff).mean(),all39_MSE=(diff**2).mean(),
            all39_per_variable_floor_relativeL2=rel.mean(),
            real_cln2_max=a[1].max(),neighbor_cln2_max=b[1].max(),
            real_ori_max=a[33].max(),neighbor_ori_max=b[33].max()))
        pairs.append(np.stack([a,b]));pairnames.append((name,nn,kind))
    pd.DataFrame(rows).to_csv(args.output_dir/'training_neighbor_trajectories.csv',index=False)
    np.savez_compressed(args.output_dir/'training_neighbor_pairs.npz',data=np.stack(pairs),names=np.array(pairnames))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
