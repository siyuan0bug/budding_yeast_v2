"""Recompute a few strongest negative trajectories without modifying source pools."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from budding_yeast_v2.scripts.analysis.compare_zero_residual_trajectories import run,rel_l2
from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--top',type=int,default=12)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.pool/'dataset.json').read_text())
    lowest=[];counts={f'below_{threshold:g}':0 for threshold in [-1e-6,-1e-4,-1e-3,-1e-2]}
    offset=0
    for index,segment in enumerate(manifest['segments']):
        path=args.pool/manifest['sources'][segment['source']]['path']
        with np.load(path,allow_pickle=False) as archive:
            y=archive['data'];minimum=y.min(axis=(1,2))
            for threshold in [-1e-6,-1e-4,-1e-3,-1e-2]:counts[f'below_{threshold:g}']+=int((minimum<threshold).sum())
            candidates=np.argsort(minimum)[:args.top]
            overrides=archive['overrides_json'];ids=archive['candidate_ids']
            for local in candidates:
                channel,t=np.unravel_index(y[local].argmin(),y[local].shape)
                lowest.append(dict(pool_row=offset+int(local),candidate_id=int(ids[local]),minimum=float(minimum[local]),
                    negative_variable=VARIABLE_NAMES[channel],negative_time_min=float(t*210/499),
                    overrides=json.loads(str(overrides[local])),trajectory=y[local].copy()))
        lowest=sorted(lowest,key=lambda r:r['minimum'])[:args.top];offset+=len(y)
        if (index+1)%100==0:print(f'Scanned {offset} trajectories',flush=True)
    tasks=[]
    for row in lowest:
        for precision,rtol,atol,timeout in [('tight',1e-8,1e-10,30.),('ultra',1e-10,1e-12,45.)]:
            tasks.append(dict(pool_row=row['pool_row'],candidate_id=row['candidate_id'],precision=precision,
                rtol=rtol,atol=atol,timeout_s=timeout,horizon=210,steps=500,overrides=row['overrides']))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as executor:
        results=list(executor.map(run,tasks))
    originals={r['pool_row']:r for r in lowest};records=[];saved={}
    for r,y in results:
        old=originals[r['pool_row']];raw=old['trajectory'];entry={k:v for k,v in r.items() if k!='overrides'}
        entry.update(source_minimum=old['minimum'],negative_variable=old['negative_variable'],negative_time_min=old['negative_time_min'])
        if y is not None:
            source_group=pattern_group_metadata(raw)['pattern_group_codes'][0]
            entry.update(recomputed_minimum=float(y.min()),source_vs_recomputed_rel_l2=rel_l2(raw,y),
                source_pattern_group=str(source_group),same_pattern_group=source_group==r['group'])
            saved[f'{r["pool_row"]}_{r["precision"]}']=y
        records.append(entry)
    for row in lowest:saved[f'{row["pool_row"]}_source']=row['trajectory']
    np.savez_compressed(args.output/'trajectories.npz',**saved)
    for r in records:
        if r['precision']!='tight' or not r['success']:continue
        ultra=next(x for x in records if x['pool_row']==r['pool_row'] and x['precision']=='ultra')
        if ultra['success']:r['tight_vs_ultra_rel_l2']=rel_l2(saved[f'{r["pool_row"]}_tight'],saved[f'{r["pool_row"]}_ultra'])
    summary=dict(pool=str(args.pool),rows_scanned=offset,negative_counts=counts,selected=len(lowest),
        recomputations=len(records),successes=sum(r['success'] for r in records),
        maximum_tight_vs_ultra_rel_l2=max((r.get('tight_vs_ultra_rel_l2',0) for r in records),default=0),
        source_dataset_modified=False,records=records)
    (args.output/'diagnosis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    keys=list(dict.fromkeys(k for r in records for k in r))
    with (args.output/'recomputations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()
