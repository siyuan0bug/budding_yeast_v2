"""Fixed-background ODE probes for badcase mechanisms, never training additions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.cross_dataset_badcases import DEFAULT_OUT


def simulate(job):
    from budding_yeast_v2.data.systems.budding.real_mutants import REAL_MUTANTS
    from budding_yeast_v2.data.systems.budding.simulation import simulate_mutant
    from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_six_patterns import features
    condition,axis,value,edit=job;overrides=dict(REAL_MUTANTS[condition][1]);overrides.update(edit)
    _,y,label,pattern=simulate_mutant(f'probe_{condition}_{axis}_{value}',overrides)
    result=dict(mutant_id=condition,axis=axis,value=value,biological_label=label,original_pattern=pattern,overrides=json.dumps(overrides,sort_keys=True))
    if y is not None:
        f,_=features(y[None]);result.update({k:v[0].item() for k,v in f.items() if k in ('divisions','pattern_code','max_clb2','max_ori','end_clb2','end_esp1')})
    return result,y


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    from budding_yeast_v2.data.systems.budding.parameters import get_default_params
    wt=get_default_params();jobs=[]
    for n in (2,3,4,5,6,8,10,15,20,30):jobs.append(('15','CLN2_copy_number',n,dict(ksn2pp=wt['ksn2pp']*n)))
    for n in (2,3,4,5,6,8,10,15,20,30):jobs.append(('22','CLB5_copy_number',n,dict(ksb5p=wt['ksb5p']*n,ksb5pp=wt['ksb5pp']*n)))
    for value in (.03,.06,.09,.12,.15,.18,.24,.36,.48,.60,.72,.96):jobs.append(('124','GAL_CLB2_ksb2p',value,dict(ksb2p=value)))
    for alpha in (0.,.000001,.00001,.0001,.001,.003,.01):jobs.append(('124','APCA_residual_fraction',alpha,dict(ka20pp=wt['ka20pp']*alpha)))
    for alpha in (0.,.0001,.001,.01,.05,.10,.20):jobs.append(('20','CLN2_residual_fraction',alpha,dict(ksn2pp=wt['ksn2pp']*alpha)))
    for value in (.5,1.,2.,3.,4.,5.,6.,8.,12.):jobs.append(('42','GAL_CDC20_ks20p',value,dict(ks20p=value)))
    args.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:results=list(executor.map(simulate,jobs))
    pd.DataFrame([x[0] for x in results]).to_csv(args.output/'fixed_background_ode_probes.csv',index=False)
    valid=[i for i,x in enumerate(results) if x[1] is not None]
    np.savez_compressed(args.output/'fixed_background_ode_probes.npz',data=np.stack([results[i][1] for i in valid]),row_indices=valid)
    print(pd.DataFrame([x[0] for x in results]).drop(columns='overrides').to_string(index=False))


if __name__=='__main__':main()
