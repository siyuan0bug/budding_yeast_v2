#!/usr/bin/env python3
"""Independent, unfiltered Budding amplitude/runtime pilot; production code is unchanged.

The historical terminal MASS rejection, if present, is removed from an in-memory
copy of the canonical solver. Current solvers already retain finite amplitudes.
Equations, events, initial conditions and tolerances stay the same. Output is a
calibration artifact, not an approved training dataset.
"""
import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import inspect
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.simulation_guards import simulation_deadline
from budding_yeast_v2.data.systems.budding import simulation
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
from budding_yeast_v2.data.systems.contracts import GenerationConfig
from budding_yeast_v2.data.systems.registry import RECIPES, make_sampler, resolve_config
from budding_yeast_v2.data.systems.generation import prepare_candidates

_SOLVE=None


def make_unfiltered_solver():
    tree=ast.parse(inspect.getsource(simulation.simulate_mutant.__wrapped__))
    fn=tree.body[0];fn.decorator_list=[];fn.name='calibration_solver'
    removed=0
    for i,node in reversed(list(enumerate(fn.body))):
        if isinstance(node,ast.If) and ast.unparse(node.test)=='tf <= 210 and max_m > 10.0':
            del fn.body[i];removed+=1
    if removed>1:raise RuntimeError('Canonical solver changed: audit the pilot transformation')
    # Preserve the cause instead of coalescing all exceptions into Numerical Failure.
    for node in ast.walk(tree):
        if isinstance(node,ast.ExceptHandler):node.body=[ast.Raise()]
    namespace=dict(simulation.__dict__)
    exec(compile(ast.fix_missing_locations(tree),'<independent-calibration-solver>','exec'),namespace)
    return namespace['calibration_solver']


def initialize_worker():
    global _SOLVE
    _SOLVE=make_unfiltered_solver()


def run_one(payload):
    index,name,overrides,timeout=payload
    started=time.perf_counter()
    result={'index':index,'name':name}
    y=None
    try:
        with simulation_deadline(timeout):
            _,y,bio,pattern=_SOLVE(name,overrides,210,500)
        if y is None:
            result.update(status='numerical_failure',reason=str(bio))
        elif y.shape!=(39,500) or not np.isfinite(y).all():
            y=None;result.update(status='invalid_output',reason='nonfinite or wrong shape')
        else:
            result.update(status='success',biological_label=bio,pattern_label=pattern,
                maxima=y.max(axis=1).tolist(),minima=y.min(axis=1).tolist(),
                negative_points=int((y<0).sum()))
    except TimeoutError as exc:
        result.update(status='timeout',reason=str(exc));y=None
    except Exception as exc:
        result.update(status='exception',reason=repr(exc));y=None
    result['wall_seconds']=time.perf_counter()-started
    return result,y


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-per-profile',type=int,default=2500)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--timeout',type=float,default=30.)
    parser.add_argument('--seed',type=int,default=20261022)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    if args.samples_per_profile<1 or args.workers<1 or args.timeout<=0:
        parser.error('samples/workers/timeout must be positive')
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    protected=[*ROOT.joinpath('data').glob('*.py'),*ROOT.joinpath('data/systems').rglob('*.py'),
               *ROOT.joinpath('utils/active_learning').rglob('*.py'),ROOT/'utils/ode_residual_engine.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    (output/'production_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    candidates=[];configs=[]
    for i,(recipe,profile) in enumerate((('budding_pool','mutant_aware'),('budding_mass_random','mass'))):
        config=resolve_config(GenerationConfig(recipe,n_samples=args.samples_per_profile,
            seed=args.seed+i,include_real=False,parameter_only=True,level_ratios=(.05,.40,.55)))
        np.random.seed(config.seed)
        sampler=make_sampler(RECIPES[recipe],config)
        rows=prepare_candidates(RECIPES[recipe],config,sampler,Counter())
        for row in rows:
            candidates.append(dict(index=len(candidates),name=row.name,level=row.level,
                                   profile=profile,recipe=recipe,overrides=row.overrides))
        configs.append(asdict(config))
    total=len(candidates)
    with (output/'candidates.jsonl').open('w') as f:
        for row in candidates:f.write(json.dumps(row,default=lambda x:x.item(),ensure_ascii=False)+'\n')
    manifest=dict(artifact_kind='unfiltered_calibration_pilot',args=vars(args),configs=configs,
        variables=VARIABLE_NAMES,param_names=PARAM_KEYS,shape=[total,39,500],
        solver_changes=['terminal MASS rejection disabled','exception causes preserved',
                        'outer wall-clock timeout replaces decorator'],
        negative_rejection=False,real_envelope_rejection=False,ours_gate=False)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    trajectories=np.lib.format.open_memmap(output/'trajectories.npy',mode='w+',dtype=np.float64,shape=(total,39,500))
    trajectories[:]=np.nan
    payload=[(r['index'],r['name'],r['overrides'],args.timeout) for r in candidates]
    completed=[];counts=Counter();started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
        futures={pool.submit(run_one,row):row[0] for row in payload}
        with (output/'diagnostics.jsonl').open('w') as log:
            for future in as_completed(futures):
                row,y=future.result();i=row['index']
                row.update(profile=candidates[i]['profile'],level=candidates[i]['level'])
                if y is not None:trajectories[i]=y
                completed.append(row);counts[row['status']]+=1
                log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                if len(completed)%250==0 or len(completed)==total:
                    print(json.dumps(dict(done=len(completed),total=total,counts=dict(counts),seconds=time.perf_counter()-started)),flush=True)
    trajectories.flush()
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),'Production files changed during calibration'
    (output/'completion.json').write_text(json.dumps(dict(counts=counts,seconds=time.perf_counter()-started,
        production_files_unchanged=len(hashes)),indent=2)+'\n')
    print('PILOT_COMPLETE',str(output),flush=True)


if __name__=='__main__':main()
