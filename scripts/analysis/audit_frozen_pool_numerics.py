#!/usr/bin/env python3
"""Stream a complete frozen pool once for numerical QC; do not modify samples."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from budding_yeast_v2.data.dataset_archive import TrajectoryStore, atomic_json
from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
from budding_yeast_v2.utils.experiments.resources import resource_slot


def audit(path, output):
    store=TrajectoryStore(path,cache_chunks=1)
    pieces=store.fields['data']['pieces']
    count=0
    totals=dict(nonfinite_rows=0,minimum_below_minus_1e_4=0,negative_rows=0,
                nonpositive_mass_rows=0,initial_mismatch_rows=0,TEM1_excess_over_1e_3=0,
                TEM1_excess_over_1pct_or_0_01=0,target_above_decode_ceiling_rows=0)
    extrema=dict(minimum=float('inf'),maximum=float('-inf'),TEM1_excess_maximum=0.,initial_maximum_error=0.)
    max_by_variable=np.full(39,-np.inf)
    flagged=[]
    started=time.monotonic()
    for i,item in enumerate(pieces):
        piece=store.manifest['segments'][item] if isinstance(item,int) else item
        sid=piece['source'];rows=piece.get('rows')
        y=store._array(sid,piece.get('key','data'))
        params=store._array(sid,'params')
        initial=store._array(sid,'y_init')
        if rows is not None:y,params,initial=y[rows],params[rows],initial[rows]
        finite=np.isfinite(y).all((1,2))
        low=y.min((1,2));high=y.max((1,2))
        tem_limit=params[:,PARAM_KEYS.index('Tem1T')]
        excess=y[:,28,:].max(1)-tem_limit
        initial_error=np.abs(y[:,:,0]-initial)
        initial_bad=(initial_error > (1e-5+1e-5*np.abs(initial))).any(1)
        severe=excess>np.maximum(.01,.01*np.abs(tem_limit))
        masks=dict(nonfinite_rows=~finite,minimum_below_minus_1e_4=low < -1e-4,
            negative_rows=low < 0,nonpositive_mass_rows=y[:,0,:].min(1)<=0,
            initial_mismatch_rows=initial_bad,TEM1_excess_over_1e_3=excess>1e-3,
            TEM1_excess_over_1pct_or_0_01=severe,
            target_above_decode_ceiling_rows=high>np.expm1(15.))
        for key,mask in masks.items():totals[key]+=int(mask.sum())
        extrema['minimum']=min(extrema['minimum'],float(low.min()))
        extrema['maximum']=max(extrema['maximum'],float(high.max()))
        extrema['TEM1_excess_maximum']=max(extrema['TEM1_excess_maximum'],float(excess.max()))
        extrema['initial_maximum_error']=max(extrema['initial_maximum_error'],float(initial_error.max()))
        max_by_variable=np.maximum(max_by_variable,y.max((0,2)))
        suspicious=(~finite)|(low < -1e-4)|initial_bad|severe|(high>np.expm1(15.))
        flagged.extend(dict(pool_row=count+int(j),minimum=float(low[j]),maximum=float(high[j]),
                            TEM1_excess=float(excess[j]),initial_error=float(initial_error[j].max()))
                       for j in np.flatnonzero(suspicious))
        count+=len(y)
        if i%25==0 or i==len(pieces)-1:
            atomic_json(output,dict(status='running',rows=count,total_rows=store.count,**totals,**extrema,
                flagged_rows=flagged,source_sha256=store.digest,seconds=time.monotonic()-started))
            print(f'{path.parent.name}: {count}/{store.count}',flush=True)
    if count!=store.count:raise ValueError('Manifest row mismatch')
    result=dict(status='complete',rows=count,**totals,**extrema,
        maxima_by_variable=max_by_variable.tolist(),flagged_rows=flagged,
        source_sha256=store.digest,seconds=time.monotonic()-started,
        thresholds_are_audit_flags_not_rejection_policy=True,source_modified=False)
    atomic_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((args.protocol/'protocol.json').read_text())
    results={}
    for name,source in protocol['pools'].items():
        with resource_slot('data','full frozen-pool numerical audit'):
            results[name]=audit(Path(source['path']),args.output/f'{name}_numerical_qc.json')
    atomic_json(args.output/'full_pool_numerical_qc.json',results)


if __name__=='__main__':main()
