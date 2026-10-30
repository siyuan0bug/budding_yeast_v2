"""Post-hoc pool availability versus acquired TRAIN coverage; no oracle queries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.cross_dataset_badcases import DEFAULT_OUT,mid
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.badcase_support import genes
from budding_yeast_v2.scripts.analysis.audit_real_mutant_module_counts import count_rows


def box(p,query,wt,mdt_col,factor,zero_tolerance=0.):
    width=np.maximum(factor*np.abs(query),.05*np.abs(wt))
    zero=query==0
    width[zero]=zero_tolerance*np.abs(wt[zero])
    result=(np.abs(p-query)<=width+1e-10).all(1)
    result &= (p[:,zero]==0).all(1) if zero_tolerance==0 else (np.abs(p[:,zero])<=zero_tolerance*np.abs(wt[zero])+1e-12).all(1)
    result &= np.isclose(p[:,mdt_col],query[mdt_col],rtol=0,atol=1e-5)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--top-k',type=int,default=30);args=parser.parse_args()
    data=Path('/home/users/zsy/mydata/budding_yeast')
    with np.load(data/'budding_pool_1000000_seed42_al.npz',allow_pickle=False) as pool:
        p=pool['params'];names=pool['mutant_names'];pnames=pool['param_names'];levels=pool['module_counts']
    with np.load(data/'budding_initial_train10000_seed42_al.npz',allow_pickle=False) as initial:
        flags=initial['is_real'];real_p=initial['params'][flags];real_names=initial['mutant_names'][flags]
    definition={r['identity']:r for r in count_rows('budding')[0]}
    lookup={mid(n):i for i,n in enumerate(real_names)};wt=real_p[lookup['1']];mdt=int(np.flatnonzero(pnames=='mdt')[0])
    ranking=pd.read_csv(args.output/'cross_dataset_badcase_ranking.csv',dtype={'mutant_id':str})
    selected=list(dict.fromkeys(ranking.mutant_id.head(args.top_k).tolist()+['15','22','42','124']))
    # The current pool has unique names and exact module metadata; this only
    # decodes those existing annotations, never changes the sampling catalog.
    gene_strings=np.array([';'.join(sorted(genes(n))) for n in names]);runs=pd.read_csv(args.output/'primary_runs.csv');runs=runs[runs.kind.eq('al')]
    acquisitions={}
    for _,row in runs.iterrows():
        key=hashlib.sha256(row.dataset.encode()).hexdigest()[:12]
        with np.load(args.output/'features'/f'{key}.npz',allow_pickle=False) as cache:
            sp=cache['params'];nm=cache['names'];flag=cache['is_real'];train=cache['split_train']
            acquisitions[row.run_id]=(row,sp[train],nm[train])
    records=[];examples=[]
    for condition in selected:
        q=real_p[lookup[condition]];target=definition[condition]['modules'];same=gene_strings==target
        b25=box(p,q,wt,mdt,.25);b50=box(p,q,wt,mdt,.5)
        nearzero=box(p,q,wt,mdt,.25,.01)
        counts=dict(pool_nearzero_local25=int(nearzero.sum()),pool_same_modules=int(same.sum()),pool_same_modules_carbon=int((same&np.isclose(p[:,mdt],q[mdt],atol=1e-5)).sum()),
                    pool_local25=int(b25.sum()),pool_local50=int(b50.sum()),
                    requested_real_modules=definition[condition]['module_count'])
        for rid,(row,train_params,train_names) in acquisitions.items():
            records.append(dict(mutant_id=condition,run_id=rid,model=row.model,strategy=row.strategy,round=int(row['round']),n_train=len(train_params),
                **counts,train_nearzero_local25=int(box(train_params,q,wt,mdt,.25,.01).sum()),train_local25_common_box=int(box(train_params,q,wt,mdt,.25).sum()),train_local50_common_box=int(box(train_params,q,wt,mdt,.5).sum())))
        for idx in np.flatnonzero(b25)[:5]:
            changed=np.abs(p[idx]-q)>1e-10
            examples.append(dict(mutant_id=condition,candidate_name=names[idx],module_count=int(levels[idx]),
                differences_from_real=json.dumps(dict(zip(pnames[changed].tolist(),p[idx,changed].tolist())))))
        print('pool coverage',condition,counts,flush=True)
    pd.DataFrame(records).to_csv(args.output/'pool_vs_acquisition_support.csv',index=False)
    pd.DataFrame(examples).to_csv(args.output/'pool_local_candidates_examples.csv',index=False)


if __name__=='__main__':main()
