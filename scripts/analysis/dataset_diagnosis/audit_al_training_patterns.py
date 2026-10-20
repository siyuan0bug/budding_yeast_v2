#!/usr/bin/env python3
"""Read-only six-pattern census of physical AL training snapshots.

Snapshot exports include real benchmark rows; these are always excluded from
synthetic training coverage counts. Repeated round-zero archives are streamed
separately, keeping per-file provenance rather than assuming identical data.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_six_patterns import (
    CODES, REAL_RE, chunks, features, identity)


def run_snapshot(job):
    path,out,canonical_path,canonical_params_path,top_names=job
    path=Path(path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    before=identity(path);started=time.monotonic()
    with np.load(path,allow_pickle=True) as z:
        names=z['mutant_names'].astype(str);patterns=z['pattern_labels'].astype(str)
        params=z['params'];keys=z['param_names'].astype(str)
    real=np.array([bool(REAL_RE.match(n)) for n in names])
    canonical=pd.read_csv(canonical_path).set_index('mutant_name')
    expected=canonical.trajectory_sha256_float32.to_dict()
    records=[];digests={}
    for start,y in chunks(path,256):
        f,_=features(y);records.append(pd.DataFrame(f))
        for j in np.flatnonzero(real[start:start+len(y)]):
            a=y[j] if y.shape[1]==39 else y[j].T
            digests[names[start+j]]=hashlib.sha256(np.asarray(a,dtype='<f4').tobytes()).hexdigest()
    f=pd.concat(records,ignore_index=True)
    if len(f)!=len(names):raise ValueError('Trajectory/name shape mismatch')
    f.insert(0,'row_index',np.arange(len(names)));f.insert(1,'mutant_name',names);f['is_real']=real
    f['original_pattern']=patterns
    old=np.select((f.divisions>=2,f.divisions==1),('Pattern_A','Pattern_B'),'Pattern_C')
    mismatch=(np.array([s.split(':')[0] for s in patterns])!=old)&f.valid.to_numpy()
    columns=['row_index','mutant_name','is_real','original_pattern','pattern_code','is_bifurcation_critical',
             'divisions','max_clb2','max_ori','max_spn','hyper_clb2_any_v','valid','negative_below_minus_1e4']
    f[columns].to_csv(out/'samples.csv.gz',index=False,compression='gzip')
    rr=f.loc[real].copy();rr['trajectory_sha256_float32']=[digests[n] for n in rr.mutant_name]
    rr['matches_canonical']=[digests.get(n)==expected.get(n) for n in rr.mutant_name]
    rr.to_csv(out/'real_verification.csv',index=False)
    if rr.mutant_name.duplicated().any() or set(rr.mutant_name)!=set(expected):
        raise ValueError('Wrong real benchmark identities in training snapshot')
    # Exact-zero support is a descriptive coordinate constraint, not a distance
    # or a proof that all members have similar complete genotypes/trajectories.
    with np.load(canonical_params_path,allow_pickle=False) as z:
        cn=z['mutant_names'].astype(str);ck=z['param_names'].astype(str);cp=z['params']
    key_order=[list(keys).index(k) for k in ck]
    train=params[~real][:,key_order]
    support=[]
    for name in top_names:
        idx=np.flatnonzero(cn==name)
        if len(idx)!=1:raise ValueError('Ambiguous canonical parameter row')
        zero=cp[idx[0]]==0
        support.append(dict(mutant_name=name,train_same_zero_parameter_face=int(np.all(train[:,zero]==0,axis=1).sum())))
    pd.DataFrame(support).to_csv(out/'zero_face_support.csv',index=False)
    counts=[]
    for population,mask in (('real',real),('synthetic',~real)):
        sub=f.loc[mask]
        for code in (*CODES,'INVALID'):
            part=sub[sub.pattern_code.eq(code)]
            counts.append(dict(population=population,pattern_code=code,count=len(part),
                critical_count=int(part.is_bifurcation_critical.sum())))
    result=dict(identity=before,counts=counts,n_real=int(real.sum()),n_synthetic=int((~real).sum()),
        old_pattern_mismatch=int(mismatch.sum()),real_exact_canonical=int(rr.matches_canonical.sum()),
        negative_below_minus_1e_minus4=int(f.loc[~real,'negative_below_minus_1e4'].sum()),
        duplicate_synthetic_names=int(pd.Series(names[~real]).duplicated().sum()),seconds=time.monotonic()-started)
    if identity(path)!=before:raise RuntimeError('Snapshot changed during census')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'DONE {out.parent.name}/{out.name}: {len(f):,} rows, canonical {result["real_exact_canonical"]}/126',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root',type=Path,default=ROOT/'experiments/al_pure_random_1m/biom_refiner_plus')
    p.add_argument('--canonical-labels',type=Path,default=ROOT/'scripts/analysis/outputs/six_pattern_audit_2026-10-20/mutant_aware_pure_random_100000_210m_500s/real_mutants.csv')
    p.add_argument('--canonical-dataset',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/mutant_aware_pure_random_100000_210m_500s.npz'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    canonical=pd.read_csv(args.canonical_labels)
    # All 126 support faces are reported; no badcase outcome enters the census.
    names=canonical.mutant_name.tolist();jobs=[];manifest=[];results=[]
    for path in sorted(args.experiment_root.glob('*/w*/seed_42/datasets/al_iter*_train*.npz')):
        strategy=path.parents[3].name.removeprefix('hybrid_mse_rel_')
        match=re.fullmatch(r'al_iter(\d+)_train(\d+)',path.stem)
        row=dict(strategy=strategy,round=int(match[1]),train_samples=int(match[2]),path=str(path.resolve()))
        out=args.output_dir/strategy/path.stem;cached=out/'summary.json'
        if args.resume and cached.exists() and json.loads(cached.read_text())['identity']==identity(path):
            result=json.loads(cached.read_text());results.append(dict(**row,**result))
        else:jobs.append((path,str(out),str(args.canonical_labels),str(args.canonical_dataset),names))
        manifest.append(row)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    mapping={r['path']:r for r in manifest};errors={}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(run_snapshot,j):j[0] for j in jobs}
        for future in as_completed(futures):
            path=futures[future]
            try:results.append(dict(**mapping[str(path.resolve())],**future.result()))
            except Exception as e:errors[str(path)]=repr(e);print(f'FAILED {path}: {e!r}',flush=True)
    (args.output_dir/'errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    rows=[]
    for s in results:
        if s['n_synthetic']!=s['train_samples']:raise ValueError('Filename budget and actual synthetic count differ')
        for c in s['counts']:rows.append(dict(strategy=s['strategy'],round=s['round'],train_samples=s['train_samples'],**c))
    pd.DataFrame(rows).to_csv(args.output_dir/'training_pattern_counts.csv',index=False)
    pd.DataFrame([{k:v for k,v in s.items() if k not in ('counts','identity')} for s in results]).to_csv(args.output_dir/'snapshot_quality.csv',index=False)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
