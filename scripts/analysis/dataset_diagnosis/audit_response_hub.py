"""Measure true response diversity around one saved acquisition hub (no new ODE)."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
from budding_yeast_v2.data.dataset_archive import open_dataset, read_dataset_rows
import pandas as pd
import torch


def read_rows(path,rows):
    return read_dataset_rows(path,rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--round',type=int,required=True);p.add_argument('--hub-id',type=int,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    audit=json.loads((a.run/f'acquisitions/after_round_{a.round:02d}/response_pairs.json').read_text())
    edges=[r for r in audit['choices'] if r['kind']=='response' and any(e['candidate_id']==a.hub_id for e in r['endpoints'])]
    wanted=sorted({e['candidate_id'] for r in edges for e in r['endpoints']})
    path=a.run/f'rounds/round_{a.round+1:02d}/dataset.npz'
    with open_dataset(path,allow_pickle=False) as z:
        ids=z['candidate_ids'];lookup={int(cid):i for i,cid in enumerate(ids) if int(cid) in wanted}
        rows=[lookup[cid] for cid in wanted];names=z['mutant_names'][rows]
        codes=z['pattern_group_codes'][rows];params=z['params'][rows]
        param_names=z['param_names']
    print('Reading cached trajectories for',len(rows),'hub and partner inputs',flush=True)
    data=read_rows(path,rows);hub=wanted.index(a.hub_id)
    state=torch.load(a.run/f'rounds/round_{a.round:02d}/preprocessing.pt',map_location='cpu',weights_only=False)
    std=state['std_y'].numpy().reshape(39,1)
    delta=(np.log1p(np.maximum(data,0))-np.log1p(np.maximum(data[hub],0)))/std
    times=np.round(np.linspace(0,499,32)).astype(int)
    actual_mse=np.mean(delta[...,times]**2,axis=(1,2))
    raw_diff=data-data[hub];rows_out=[]
    for cid,name,code,mse,diff in zip(wanted,names,codes,actual_mse,raw_diff):
        matching=[r for r in edges if any(e['candidate_id']==cid for e in r['endpoints'])]
        if cid==a.hub_id:continue
        rows_out.append(dict(candidate_id=cid,name=str(name),group=str(code),
            true_normalized_response_mse32=float(mse),max_absolute_trajectory_difference=float(np.max(np.abs(diff))),
            predicted_response_disagreement_mse32=float(matching[0]['response_variance'])))
    frame=pd.DataFrame(rows_out);frame.to_csv(a.output/'hub_true_response.csv',index=False)
    varying=np.flatnonzero(np.ptp(params,axis=0)>1e-10)
    fractions=[r['response_variance']/(1+r['distance'])/(
        sum(e['kind']=='pool' for e in r['endpoints'])*r['score']) for r in edges]
    report=dict(hub_id=a.hub_id,hub_name=str(names[hub]),response_actions=len(edges),partner_count=len(frame),
        true_response_mse_median=float(frame.true_normalized_response_mse32.median()),
        true_response_mse_max=float(frame.true_normalized_response_mse32.max()),
        predicted_disagreement_median=float(frame.predicted_response_disagreement_mse32.median()),
        response_score_fraction_median=float(np.median(fractions)),
        raw_max_abs_difference=float(frame.max_absolute_trajectory_difference.max()),
        varying_parameters=param_names[varying].tolist(),scope='Saved ODE labels only; no simulation or training')
    (a.output/'hub_true_response_summary.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(a.output/'hub_trajectories.npz',candidate_ids=wanted,data=data,hub_index=hub)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
