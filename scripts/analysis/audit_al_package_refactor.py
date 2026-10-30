#!/usr/bin/env python3
"""Compare all 15 extracted acquisitions with the exact pre-package snapshot.

Artificial candidate data only; no real benchmark tuning or experiment mutation.
The audit checks algorithm ASTs, ordered selection, quotas and RNG streams.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import random
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import TensorDataset

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.utils.active_learning import ActiveLearningEngine
from budding_yeast_v2.utils.active_learning.registry import SPECS
from budding_yeast_v2.utils.active_learning.oracle import PARAM_KEYS, _MASS_DEFAULTS


def load_baseline():
    path=ROOT/'utils/archive/al_callback_before_package.py'
    # Resolve historical relative imports through the preserved archive package.
    name='budding_yeast_v2.utils.archive._al_package_baseline'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


class NormalizeAST(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return None
        return self.generic_visit(node)
    def visit_Attribute(self,node):
        if node.attr=='make_selection_plan':node.attr='_build_broadcast_from_pool'
        return self.generic_visit(node)


def normalized_source(function):
    node=ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(function)))).body[0]
    node.decorator_list=[]
    return ast.dump(NormalizeAST().visit(node),include_attributes=False)


class ArtificialSurrogate(torch.nn.Module):
    def __init__(self,offset=0.):
        super().__init__()
        self.offset=torch.nn.Parameter(torch.tensor(offset))
        self.model=SimpleNamespace()
    def forward_ic_time(self,x,p):
        t=torch.linspace(0,1,x.shape[-1],device=p.device)[None,None,:]
        channels=torch.arange(39,device=p.device)[None,:,None]
        return .15+.04*torch.sin(t*8+channels*.2)+.01*p[:,:1,None]+self.offset


def fixture(cls,strategy,seed):
    cb=cls(strategy=strategy,initial_train_size=2,random_seed=seed,bait_sketch_dim=5)
    count,train_count,steps=8,6,16
    params=np.tile(np.array([_MASS_DEFAULTS[k] for k in PARAM_KEYS],dtype=np.float32),(count,1))
    params[:,PARAM_KEYS.index('ksb2p')]=np.linspace(.002,.01,count)
    cb._pool_size=count
    cb._pool_P=torch.tensor(params)
    cb._pool_P_real=torch.tensor(params)
    cb._pool_X=torch.zeros(count,39,2,steps)
    cb._pool_mutant_names=np.array([f'MASS_RAND_{g}[{m}]_L1_{i}' for i,(g,m) in enumerate(
        [('CLB2','KO'),('CLN2','GAL-OE'),('SIC1','KO'),('CLB2','GAL-OE')]*2)])
    cb._pool_mass_meta=[{'genes':[(g,m)],'level':1} for g,m in
        [('CLB2','KO'),('CLN2','GAL-OE'),('SIC1','KO'),('CLB2','GAL-OE')]*2]
    cb._al4pde_t_grid=torch.linspace(0,1,steps)[None,None,:]
    cb._param_names_cached=np.array(PARAM_KEYS)
    model=ArtificialSurrogate()
    x=cb._pool_X[:train_count].clone();p=cb._pool_P[:train_count].clone()
    y=model.forward_ic_time(x,p).detach()
    names=cb._pool_mutant_names[:train_count].tolist()
    dm=SimpleNamespace(system_id='budding',train_dataset_synth=TensorDataset(x,p,y,torch.tensor([0,1,2,0,1,2])),
        p_mean=torch.zeros(1,141),p_std=torch.ones(1,141),t_max=210.,_Y=torch.full((1,39,steps),100.),
        _test_idx_real=[0],_mutant_names=names,_train_idx_synth=np.arange(train_count),
        denormalize_y=lambda value,*args:value)
    if strategy in SPECS and SPECS[strategy].requires_ensemble:
        cb.set_ensemble_members([model,ArtificialSurrogate(.02)])
    if strategy.startswith('ours_'):
        cb._pool_ovr = [{} for _ in range(count)]
        cb._pool_initial_physical = torch.zeros(count,39,dtype=torch.float64)
    cb._build_broadcast_from_pool=lambda indices,dm,target_valid_num=None:{
        'indices':np.asarray(indices).tolist(),'target_valid_num':None if target_valid_num is None else int(target_valid_num)}
    return cb,model,dm


def random_state():
    state=np.random.get_state()
    return (state[0],state[1].copy(),state[2:],torch.get_rng_state().clone(),random.getstate())


def same_state(left,right):
    return (left[0]==right[0] and np.array_equal(left[1],right[1]) and left[2]==right[2]
        and torch.equal(left[3],right[3]) and left[4]==right[4])


def run_audit(seeds):
    baseline=load_baseline()
    rows=[]
    for strategy,spec in SPECS.items():
        before=getattr(baseline.ActiveLearningCallback,spec.function)
        after=getattr(ActiveLearningEngine,spec.function)
        if normalized_source(before)!=normalized_source(after):
            raise AssertionError(f'Numerical source changed: {strategy}')
        for seed in seeds:
            selections=[];states=[]
            for cls in (baseline.ActiveLearningCallback,ActiveLearningEngine):
                cb,model,dm=fixture(cls,strategy,seed)
                np.random.seed(seed);torch.manual_seed(seed);random.seed(seed)
                result=getattr(cb,spec.function)(None,model,dm,'cpu',0)
                selections.append(result);states.append(random_state())
            if selections[0]!=selections[1] or not same_state(*states):
                raise AssertionError(f'Selection or RNG differs: {strategy}, seed={seed}')
            rows.append({'strategy':strategy,'seed':seed,'selection':selections[1],
                         'source_equal':True,'rng_equal':True})
    return {'strategy_count':len(SPECS),'cases_passed':len(rows),'cases':rows,
        'baseline_sha256':hashlib.sha256((ROOT/'utils/archive/al_callback_before_package.py').read_bytes()).hexdigest(),
        'scope':'artificial inputs, ordered selection and RNG equivalence; no production training'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,7,42])
    args=parser.parse_args()
    with contextlib.redirect_stdout(io.StringIO()),torch.random.fork_rng(devices=[]):
        report=run_audit(args.seeds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
