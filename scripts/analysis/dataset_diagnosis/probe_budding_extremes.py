#!/usr/bin/env python3
"""Probe in-support Budding dose quantiles without introducing amplitude filters.

Quantile scenarios are diagnostic designs, not estimates of their joint sampling
probabilities. Both registered profiles and only legal current mutation cells are
used. Raw trajectories and exact overrides are saved for independent review.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.calibrate_budding_limits import initialize_worker, run_one
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.budding.catalog import GROUP_BY_NAME, touch_set
from budding_yeast_v2.data.systems.budding.real_mutants import is_duplicate_real_mutant

SKELETONS=[
 [('CLB2','MC'),('CLB2','db')], [('CLB2','GAL-OE'),('CLB2','db')],
 [('SIC1','MC'),('SIC1','db')], [('SIC1','GAL-OE'),('SIC1','db')],
 [('CLB5','MC'),('CLB5','db')], [('CLB5','GAL-OE'),('CLB5','db')],
 [('PDS1','GAL-OE'),('PDS1','db')], [('CDC6','MC')], [('CDC20','MC')],
 [('CDC14','MC')], [('NET1','GAL-OE'),('NET1','ts')], [('TEM1','MC')],
 [('CDC15','MC')], [('CLN2','MC')], [('CLN3','MC')], [('BCK2','MC')],
 [('SWI5','GAL-OE')],
 [('CLB2','MC'),('CLB2','db'),('CDH1','KO')],
 [('CLB2','GAL-OE'),('CLB2','db'),('APC_Core','APC-A')],
 [('CLB2','GAL-OE'),('CLB2','db'),('CDC20','ts')],
 [('CLB2','GAL-OE'),('CLB2','db'),('CDC14','ts')],
 [('CLB2','GAL-OE'),('CLB2','db'),('CDH1','KO'),('CDC20','ts')],
 [('CLB2','GAL-OE'),('CLB2','db'),('APC_Core','APC-A'),('CDC20','ts')],
 [('CDC14','MC'),('NET1','DAmP')],
 [('CDC6','MC'),('SIC1','MC'),('CDH1','KO')],
 [('CDC20','MC'),('CDH1','KO')],
 [('CLN2','MC'),('CLN3','MC'),('BCK2','MC')],
 [('CLN3','KO'),('BCK2','KO'),('CLN2','MC')],
 [('CLN2','KO'),('CLN3','KO'),('CLB5','MC')],
]


def make_rows():
    rows=[]
    scenarios=[('median',.5,False),('one_high_q90',.9,False),
               ('joint_q90',.9,True),('joint_q95',.95,True),('joint_q99',.99,True)]
    for profile in ('mutant_aware','mass'):
        backend=BuddingSystem(profile,profile)
        for skeleton in SKELETONS:
            touched=set()
            for gene,mode in skeleton:
                assert gene in GROUP_BY_NAME and mode in GROUP_BY_NAME[gene]['modes']
                footprint=touch_set(gene,mode)
                assert not touched&footprint,(skeleton,touched&footprint)
                touched |= footprint
            for scenario,q,joint in scenarios:
                defaults=backend.defaults();p=defaults.copy();ovr={};doses=[]
                raised=False
                for gene,mode in skeleton:
                    # Raising one expression coordinate isolates a reasonably
                    # accessible extreme from simultaneous multi-tail corners.
                    if mode in ('GAL-OE','MC'):
                        u=q if joint or not raised else .5
                        raised=True
                    elif mode in ('db','ts','DAmP','pLOF'):
                        u=1-q if joint else .5
                    else:u=.5
                    alpha=backend.sample_scaling_factor_sobol(mode,gene,u)
                    backend.apply_gene_mutation(gene,mode,alpha,defaults,p,ovr,rng=np.random.RandomState(1729))
                    doses.append(dict(gene=gene,mode=mode,cdf_coordinate=u,alpha=alpha))
                # Exactly the existing carbon rule for these constructs.
                ovr['mdt']=150. if any(m=='GAL-OE' for g,m in skeleton) else 90.
                rows.append(dict(index=len(rows),profile=profile,scenario=scenario,
                    skeleton='+'.join(f'{g}[{m}]' for g,m in skeleton),
                    name=f'Probe_{profile}_{len(rows)}',level=len({g for g,m in skeleton}),
                    doses=doses,overrides=ovr,
                    canonical_real_collision=is_duplicate_real_mutant(ovr,defaults)))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--timeout',type=float,default=30.)
    args=parser.parse_args();output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    rows=make_rows()
    protected=list(ROOT.joinpath('data/systems').rglob('*.py'))+list(ROOT.joinpath('utils/active_learning').rglob('*.py'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    with (output/'candidates.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,default=lambda x:x.item())+'\n')
    y=np.lib.format.open_memmap(output/'trajectories.npy',mode='w+',dtype=np.float64,shape=(len(rows),39,500));y[:]=np.nan
    started=time.perf_counter();diagnostics=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
        futures=[pool.submit(run_one,(r['index'],r['name'],r['overrides'],args.timeout)) for r in rows]
        with (output/'diagnostics.jsonl').open('w') as f:
            for future in as_completed(futures):
                result,trajectory=future.result();i=result['index']
                result.update(profile=rows[i]['profile'],scenario=rows[i]['scenario'],skeleton=rows[i]['skeleton'],level=rows[i]['level'])
                if trajectory is not None:y[i]=trajectory
                diagnostics.append(result);f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
    y.flush()
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (output/'completion.json').write_text(json.dumps(dict(samples=len(rows),success=sum(r['status']=='success' for r in diagnostics),
        seconds=time.perf_counter()-started,production_files_unchanged=len(hashes),
        note='Diagnostic quantile combinations; joint occurrence rates are not inferred.'),indent=2)+'\n')
    print((output/'completion.json').read_text(),flush=True)


if __name__=='__main__':main()
