#!/usr/bin/env python3
"""Read-only streaming audit of current Budding AL filters on an existing NPZ."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_six_patterns import chunks
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--features',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--chunk-size',type=int,default=128)
    args=parser.parse_args()
    if args.chunk_size<1:parser.error('chunk-size must be positive')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(args.features)
    with np.load(args.dataset,allow_pickle=True) as z:
        names=z['mutant_names'];params=z['params'];keys=z['param_names']
    assert np.array_equal(names.astype(str),frame.mutant_name.to_numpy(str))
    assert np.array_equal(frame.row_index.to_numpy(),np.arange(len(frame)))
    real=frame.is_real.to_numpy(bool)
    real_last=np.flatnonzero(real)[-1]
    # Read only through the last real row for threshold calibration.
    maxima=np.zeros((len(frame),39));minima=np.zeros_like(maxima)
    realmax=np.full(39,-np.inf)
    for start,y in chunks(args.dataset,args.chunk_size):
        local=real[start:start+len(y)]
        if local.any():realmax=np.maximum(realmax,y[local].max(axis=(0,2)))
        if start+len(y)>real_last:break
    limits=np.maximum(2*realmax,2.)
    envelope=np.zeros((len(frame),39),bool)
    point_negative=np.zeros(39,dtype=np.int64)
    severe_points=np.zeros(39,dtype=np.int64)
    started=time.monotonic()
    for start,y in chunks(args.dataset,args.chunk_size):
        end=start+len(y)
        if not np.isfinite(y).all():
            raise ValueError(f'Nonfinite raw trajectory in rows {start}:{end}')
        maxima[start:end]=y.max(axis=2);minima[start:end]=y.min(axis=2)
        envelope[start:end]=maxima[start:end]>limits
        point_negative+=(y<0).sum(axis=(0,2))
        severe_points+=(y<-1e-4).sum(axis=(0,2))
        if (start//args.chunk_size)%100==0:
            print(f'{end}/{len(frame)} rows, {time.monotonic()-started:.1f}s',flush=True)
    flux_ok=VectorizedYeastODE(list(keys)).check_nominal_flux_mask(torch.from_numpy(params)).numpy()
    frame['mass_above_10']=maxima[:,0]>10
    frame['envelope_rejected']=envelope[:,1:].any(axis=1)
    frame['flux_rejected']=~flux_ok
    frame['al_rejected']=frame.mass_above_10|frame.envelope_rejected
    frame['flux_or_al_rejected']=frame.al_rejected|frame.flux_rejected
    frame['raw_min']=minima.min(axis=1)
    for threshold in (0.,1e-12,1e-8,1e-6,1e-4,1e-3,1e-2):
        frame[f'negative_below_{threshold:g}']=frame.raw_min < -threshold
    def summary(rows):
        n=len(rows)
        return {'count':n,**{key:int(rows[key].sum()) for key in
            ('mass_above_10','envelope_rejected','flux_rejected','al_rejected','flux_or_al_rejected')},
            'negative_counts':{key:int(rows[key].sum()) for key in rows if key.startswith('negative_below_')},
            'minimum':float(rows.raw_min.min()),
            'max_mass':float(rows.max_mass.max())}
    result={'dataset':str(Path(args.dataset).resolve()),'features':args.features,
        'note':'Historical saved survivors only; cannot measure mass-discarded or failed simulations.',
        'rows':len(frame),'seconds':time.monotonic()-started,'real_maxima':realmax.tolist(),
        'envelope_limits':limits.tolist(),
        'synthetic':summary(frame[~real]),'real':summary(frame[real]),
        'synthetic_by_pattern':{str(k):summary(g) for k,g in frame[~real].groupby('pattern_code')},
        'real_negative_rows':frame[real&(frame.raw_min<-1e-4)][['mutant_name','raw_min']].to_dict('records'),
        'real_flux_rejected':frame[real&frame.flux_rejected][['mutant_name','pattern_code']].to_dict('records')}
    variables=pd.DataFrame({'variable':VARIABLE_NAMES,'real_max':realmax,'limit':limits,
        'synthetic_envelope_exceedances':envelope[~real].sum(axis=0),
        'real_envelope_exceedances':envelope[real].sum(axis=0),
        'all_minimum':minima.min(axis=0),'negative_points':point_negative,'points_below_minus_1e4':severe_points})
    frame.to_csv(output/'samples.csv.gz',index=False)
    variables.to_csv(output/'variables.csv',index=False)
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    # Preserve per-variable extrema for follow-up threshold sensitivity without
    # decompressing the large trajectory archive again.
    np.savez_compressed(output/'extrema.npz',maxima=maxima,minima=minima,real=real,thresholds=limits)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
