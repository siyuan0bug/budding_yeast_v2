"""Post-hoc ensemble calibration on benchmark inputs, never an AL selector."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.dataset_archive import open_dataset
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.cross_dataset_badcases import DEFAULT_OUT,mid
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pure_random_evidence import data_chunks


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--device',default='cuda:5');args=parser.parse_args()
    import torch
    from budding_yeast_v2.eval import load_model_from_checkpoint
    from budding_yeast_v2.data.normalization import normalize_y,normalize_p,denormalize_y
    from budding_yeast_v2.utils.metrics import calculate_metrics
    torch.set_num_threads(4);runs=pd.read_csv(args.output/'primary_runs.csv');rows=[]
    for _,run in runs[runs.kind.eq('al')&runs.model.eq('fno')&runs.strategy.isin(['al4pde_uncertainty','al4pde_power'])].iterrows():
        for r in range(3):
            directory=Path(run.run)/'rounds'/f'round_{r:02d}'
            checkpoints=[directory/'model.ckpt',directory/'ensemble'/f'round_{r}_ensemble_member_1.ckpt']
            if not all(p.exists() for p in checkpoints):continue
            with open_dataset(directory/'dataset.npz',allow_pickle=False) as z:
                flags=z['is_real'];real=np.flatnonzero(flags);params=z['params'][real];names=z['mutant_names'][real]
            selected={}
            for start,block in data_chunks(directory/'dataset.npz',256):
                for idx in real[(real>=start)&(real<start+len(block))]:selected[int(idx)]=block[idx-start]
                if len(selected)==len(real):break
            truth=torch.from_numpy(np.stack([selected[int(i)] for i in real]).astype('float32')).clamp(min=0)
            outputs=[];state=None
            for checkpoint in checkpoints:
                loaded=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);pp=loaded['preprocessing_contract'];state=pp
                x0=normalize_y(truth[:,:,:1],pp['mean_y'],pp['std_y']).expand(-1,-1,500)
                grid=torch.linspace(0,1,500)[None,None].expand_as(x0);x=torch.stack((x0,grid),2)
                p=normalize_p(torch.from_numpy(params.astype('float32')),pp['p_mean'],pp['p_std'])
                model,_=load_model_from_checkpoint(str(checkpoint),device='cpu',graph_load_policy='checkpoint');model.to(args.device).eval();prediction=[]
                with torch.inference_mode():
                    for i in range(0,len(real),32):prediction.append(model(x[i:i+32].to(args.device),p[i:i+32].to(args.device)).cpu())
                outputs.append(torch.cat(prediction));del model;torch.cuda.empty_cache()
            stacked=torch.stack(outputs);score=((stacked-stacked.mean(0))**2).mean((0,2,3)).numpy()
            rank=pd.Series(score).rank(ascending=False,method='min').to_numpy()
            pred=denormalize_y(outputs[0],state['mean_y'],state['std_y'])
            for i,name in enumerate(names):
                metric=calculate_metrics(pred[i],truth[i]);rows.append(dict(strategy=run.strategy,round=r,mutant_id=mid(name),uncertainty=score[i],uncertainty_rank_among126=rank[i],**{k.replace(' ','_'):v for k,v in metric.items()}))
            print(run.strategy,r,'scored',len(real),flush=True)
    pd.DataFrame(rows).to_csv(args.output/'real_ensemble_calibration.csv',index=False)


if __name__=='__main__':main()
