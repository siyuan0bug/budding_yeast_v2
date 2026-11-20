#!/usr/bin/env python3
"""Strictly recompute numerical audit flags and locate their benchmark exposure."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from budding_yeast_v2.data.dataset_archive import TrajectoryStore, atomic_json
from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
from budding_yeast_v2.scripts.analysis.compare_zero_residual_trajectories import run, rel_l2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--source-results',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    protocol=json.loads((args.protocol/'protocol.json').read_text())
    qc=json.loads((args.audit/'full_pool_numerical_qc.json').read_text())
    originals={};tasks=[]
    for name,report in qc.items():
        ids=[r['pool_row'] for r in report['flagged_rows']]
        if not ids:continue
        store=TrajectoryStore(protocol['pools'][name]['path'],cache_chunks=1)
        raw=store.read(ids,('data','params','overrides_json'))
        with np.load(args.protocol/f'{name}_split.npz',allow_pickle=False) as a:
            roles={role:set(map(int,a[role])) for role in ('initial','candidates','validation','test')}
        for i,row in enumerate(ids):
            exposed=[]
            for snapshot in sorted((args.source_results/name).glob('*/seed_42/rounds/round_*/dataset.json')):
                if row in json.loads(snapshot.read_text())['rows']:
                    exposed.append(dict(method=snapshot.parents[3].name,round=snapshot.parent.name))
            originals[name,row]=dict(data=raw['data'][i],Tem1T=float(raw['params'][i,PARAM_KEYS.index('Tem1T')]),
                                    roles=[role for role,values in roles.items() if row in values],training_exposure=exposed)
            tasks.append(dict(pool=name,pool_row=row,precision='tight',rtol=1e-8,atol=1e-10,
                timeout_s=30.,horizon=210,steps=500,overrides=json.loads(str(raw['overrides_json'][i]))))
    results=[];plots=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as executor:
        for entry,trajectory in executor.map(run,tasks):
            old=originals[entry['pool'],entry['pool_row']]
            row={k:v for k,v in entry.items() if k!='overrides'}
            row.update(roles=old['roles'],training_exposure=old['training_exposure'],Tem1T=old['Tem1T'],
                       old_minimum=float(old['data'].min()),old_TEM1_peak=float(old['data'][28].max()))
            if trajectory is not None:
                row.update(tight_TEM1_peak=float(trajectory[28].max()),
                    tight_TEM1_excess=float(trajectory[28].max()-old['Tem1T']),
                    source_vs_tight_rel_l2=rel_l2(old['data'],trajectory),
                    old_pattern=str(pattern_group_metadata(old['data'])['pattern_group_codes'][0]))
                plots.append((row,old['data'],trajectory))
            results.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='training_exposure'}),flush=True)
    atomic_json(args.audit/'strict_solver_rechecks.json',dict(
        cases=len(results),successes=sum(r['success'] for r in results),
        records=results,source_modified=False,sampling_rules_changed=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots((len(plots)+3)//4,4,figsize=(16,3.4*((len(plots)+3)//4)),squeeze=False)
    for ax,(row,original,tight) in zip(axes.flat,plots):
        channel=28 if row['old_TEM1_peak']>row['Tem1T']+.01 else int(np.unravel_index(original.argmin(),original.shape)[0])
        t=np.linspace(0,210,500)
        ax.plot(t,original[channel],label='Stored BDF 1e-4/1e-6',lw=1.3)
        ax.plot(t,tight[channel],label='Tight BDF 1e-8/1e-10',lw=1.3,ls='--')
        if channel==28:ax.axhline(row['Tem1T'],color='gray',ls=':',label='Tem1 total')
        ax.set(title=f"{row['pool']} row {row['pool_row']} / variable {channel}",xlabel='Time (min)',ylabel='Physical value')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(plots):]:ax.axis('off')
    fig.tight_layout();fig.savefig(args.audit/'strict_solver_rechecks.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
