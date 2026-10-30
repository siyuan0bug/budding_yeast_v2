"""Reconstruct legacy asinh_scale statistics without fitting on benchmark errors."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.cross_dataset_badcases import DEFAULT_OUT
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pure_random_evidence import data_chunks


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUT);args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    runs=pd.read_csv(args.output/'primary_runs.csv');runs=runs[runs.norm.eq('asinh_scale')]
    for _,row in runs.iterrows():
        cache=args.output/'features'/f'{hashlib.sha256(row.dataset.encode()).hexdigest()[:12]}.npz'
        with np.load(cache,allow_pickle=False) as z:real=z['is_real'];params=torch.from_numpy(z['params'].astype('float32'))
        wanted=np.flatnonzero(~real)[:20000];collected=[]
        for start,block in data_chunks(row.dataset,256):
            local=wanted[(wanted>=start)&(wanted<start+len(block))]-start
            if len(local):collected.append(np.maximum(block[local].astype('float32'),0))
            if start+len(block)>wanted[-1]:break
        first=np.concatenate(collected);sc=np.maximum(np.quantile(first,.9,axis=(0,2),keepdims=True),.01).astype('float32');del first,collected
        total=np.zeros(39);square=np.zeros(39);count=0
        for start,block in data_chunks(row.dataset,256):
            keep=~real[start:start+len(block)];x=np.arcsinh(np.maximum(block[keep].astype('float32'),0)/(2*sc)).astype('float64')
            total+=x.sum((0,2));square+=(x*x).sum((0,2));count+=len(x)*500
        ym=(total/count).astype('float32');ys=(np.sqrt(np.maximum((square-total*total/count)/(count-1),0))+1e-6).astype('float32')
        p=torch.log10(params[np.flatnonzero(~real)].clamp(min=1e-8))
        directory=args.output/'normalization';directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/f'{row.run_id}.npz',p_mean=p.mean(0).numpy(),p_std=p.std(0).numpy(),mean_y=ym,std_y=ys,s_c=sc)
        print('statistics reconstructed',row.dataset,flush=True)


if __name__=='__main__':main()
