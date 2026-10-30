"""Post-hoc ensemble calibration on real benchmarks; never used for acquisition.

Replays saved pre-query checkpoints on CPU. No training or ODE simulation.
"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from budding_yeast_v2.data.dataset_archive import open_dataset, read_dataset_rows
from budding_yeast_v2.data.normalization import normalize_y,normalize_p,denormalize_y
from budding_yeast_v2.eval import load_model_from_checkpoint
from budding_yeast_v2.utils.metrics import calculate_metrics


def real_prefix(path):
    with open_dataset(path,allow_pickle=False) as z:
        real=z['split_real'];np.testing.assert_array_equal(real,np.arange(len(real)))
        params=z['params'][real];names=z['mutant_names'][real]
    data=read_dataset_rows(path,real)
    return torch.tensor(data,dtype=torch.float32),torch.tensor(params,dtype=torch.float32),names


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(a.threads)
    target,params,names=real_prefix(a.run/'rounds/round_00/dataset.npz')
    rows=[];summaries=[]
    for r in range(3):
        folder=a.run/f'rounds/round_{r:02d}'
        primary,_=load_model_from_checkpoint(str(folder/'model.ckpt'),device='cpu')
        paths=sorted((folder/'ensemble').glob('*member_1.ckpt'));assert len(paths)==1
        auxiliary,_=load_model_from_checkpoint(str(paths[0]),device='cpu')
        state=primary._preprocessing_state
        for key in ('mean_y','std_y','p_mean','p_std'):
            torch.testing.assert_close(state[key],auxiliary._preprocessing_state[key],rtol=0,atol=0)
        truth=normalize_y(target,state['mean_y'],state['std_y'],state['norm_type'],state.get('s_c'))
        ic=truth[:,:,0];grid=torch.linspace(0,1,500)[None,None,:].expand(len(ic),39,-1)
        x=torch.stack((ic[:,:,None].expand(-1,-1,500),grid),2)
        normalized_p=normalize_p(params,state['p_mean'],state['p_std'],state['norm_type'])
        primary.eval();auxiliary.eval();pred=[]
        with torch.no_grad():
            for model in (primary,auxiliary):
                pred.append(torch.cat([model.forward_ic_time(x[i:i+32],normalized_p[i:i+32]).float() for i in range(0,len(x),32)]))
        primary_error=(pred[0]-truth).square().mean((1,2)).numpy()
        auxiliary_error=(pred[1]-truth).square().mean((1,2)).numpy()
        unc=((pred[0]-pred[1])/2).square().mean((1,2)).numpy()
        rank=pd.Series(unc).rank(pct=True).to_numpy()
        decoded=[denormalize_y(value,state['mean_y'],state['std_y'],state['norm_type'],state.get('s_c')) for value in pred]
        for i,name in enumerate(names):
            metric=calculate_metrics(decoded[0][i],target[i]);aux=calculate_metrics(decoded[1][i],target[i])
            rows.append(dict(round=r,Mutant=str(name),uncertainty=float(unc[i]),real_uncertainty_percentile=float(rank[i]),
                primary_normalized_mse=float(primary_error[i]),auxiliary_normalized_mse=float(auxiliary_error[i]),
                disagreement_to_primary_error=float(unc[i]/max(primary_error[i],1e-12)),
                primary_MAE=metric['MAE'],primary_Relative_L2=metric['Relative L2'],
                auxiliary_MAE=aux['MAE'],auxiliary_Relative_L2=aux['Relative L2']))
        summaries.append(dict(round=r,spearman_uncertainty_vs_normalized_error=float(spearmanr(unc,primary_error).statistic),
            median_disagreement_to_error=float(np.median(unc/np.maximum(primary_error,1e-12)))))
        print('Round',r,summaries[-1],flush=True)
        del primary,auxiliary,pred,decoded
    pd.DataFrame(rows).to_csv(a.output/'real_ensemble_replay.csv',index=False)
    (a.output/'real_ensemble_replay_summary.json').write_text(json.dumps(dict(
        scope='post-hoc on real benchmarks, not candidate acquisition and not a pool calibration estimate',rounds=summaries),indent=2))


if __name__=='__main__':main()
