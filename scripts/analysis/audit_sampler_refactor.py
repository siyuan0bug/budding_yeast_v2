"""Exact old/new candidate and driver-order audit, without ODE or dataset writes.

Legacy files are imported read-only. Collision tests are bypassed only in driver
comparisons, isolating the sampler from the explicitly changed rescue policy.
"""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib
import inspect
import io
import json
import math
from pathlib import Path
import random
import sys
import textwrap
from unittest.mock import patch

import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.systems.contracts import GenerationConfig
from budding_yeast_v2.data.systems.generation import prepare_candidates
from budding_yeast_v2.data.systems.registry import RECIPES
from budding_yeast_v2.data.systems.registry import make_sampler
from budding_yeast_v2.data.systems.budding.samplers import common
from budding_yeast_v2.data.systems.budding import real_mutants
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem

CASES={
 'budding_random':('generate_pure_random_dataset','sample_pure_random_mutant','generate_pure_random_dataset'),
 'budding_random_ko_interval':('generate_pure_random_dataset','sample_pure_random_mutant','generate_pure_random_dataset'),
 'budding_mass_random':('mass_random_sampler','sample_mass_random','generate_mass_random_dataset'),
 'budding_mass_random_dist':('mass_random_mutant_dist','sample_mass_random_dist','generate_mass_random_dist_dataset'),
 'budding_mass':('mass_dataset_generator','sample_mass_mutant','generate_mass_dataset'),
 'budding_mutant_aware':('mutant_aware','sample_mass_mutant','generate_mass_dataset'),
 'budding_gibbs':('gibbs_boltzmann_sampler','sample_gibbs_mutant','generate_gibbs_dataset'),
 'budding_gibbs_sobol':('gibbs_sobol_sampler','sample_gibbs_sobol_mutant','generate_gibbs_sobol_dataset'),
 'budding_kdpp':('generate_kdpp_gated_dataset','sample_kdpp_mutant','generate_kdpp_dataset'),
 'budding_kdpp_curvature':('generate_kdpp_curv_gated_dataset','sample_signed_kdpp_mutant','generate_kdpp_curv_gated_dataset'),
 'budding_motif':('generate_motif_topology_dataset','build_perturbed_sample','generate_motif_dataset'),
 'budding_bams':('bams_dataset_generator','sample_bams_mutant_task','generate_bams_dataset'),
 'budding_pool':('generate_1m_pure_random_pool','sample_pure_random_candidate','generate_1m_pure_random_pool'),
 'fission_random':('fission_random_sampler','sample_random_mutant','generate_dataset'),
}


def module(name):
    return importlib.import_module('budding_yeast_v2.data.'+name)


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value),sort_keys=True,allow_nan=True).encode()).hexdigest()


def rng_state(rng):
    return digest(rng.bit_generator.state if isinstance(rng,np.random.Generator) else rng.get_state())


def reset(m,seed,mode='class'):
    common.MODE_FREQ=mode
    for obj in (m,getattr(m,'ma',None)):
        if obj is None:continue
        if hasattr(obj,'MODE_FREQ'):obj.MODE_FREQ=mode
        if hasattr(obj,'_L1_IDX'):obj._L1_IDX['i']=0
        for picker,quotas in (('_L2_PICKER','STRUCT_L2'),('_L34_PICKER','STRUCT_L34')):
            if hasattr(obj,picker):
                q=getattr(obj,quotas)
                setattr(obj,picker,obj.QuotaPicker(list(q),list(q.values())))
    if m.__name__.endswith('fission_random_sampler'):
        m._L1_GRID[:]=[(g,mo) for g in m.SAMPLING_MODULES for mo in m.SAMPLING_MODULES[g]['modes']]
    np.random.seed(seed);random.seed(seed)


def record(level,result,rng=None,kernel=None):
    state={'global':rng_state(np.random)}
    if rng is not None:state['caller']=rng_state(rng)
    if kernel is not None:state['kernel']=rng_state(kernel.rng)
    return {'level':int(level),'result':plain(result),'rng':state}


def new_result(sampler,level):
    overrides,description,refinement,pairs=sampler.sample(level)
    kind=sampler.recipe.sampler
    if kind=='motif':return overrides
    if kind=='pool':return (overrides,description,'')
    return (overrides,description,refinement) if kind=='bams' else (overrides,description)


def raw_stream(name,seed,count,new):
    recipe=RECIPES[name]
    old=module(CASES[name][0]);reset(old,seed,recipe.mode_freq)
    cfg=GenerationConfig(name,n_samples=count,seed=seed,mode_freq=recipe.mode_freq)
    kernel=None
    if new:
        sampler=make_sampler(recipe,cfg);rng=sampler.rng;kernel=getattr(sampler, 'kernel', None)
    else:
        rng=np.random.default_rng(seed) if recipe.system=='fission' else np.random.RandomState(seed)
        if recipe.sampler in ('gibbs','gibbs_sobol'):kernel=old.GibbsBoltzmannGraphSampler(seed=seed)
        if recipe.sampler=='kdpp':kernel=old.GraphDiffusionKDPP(seed=seed)
        if recipe.sampler=='kdpp_curvature':kernel=old.SignedDirectedGraphKDPP(seed=seed)
        if recipe.sampler=='motif':kernel=old.MotifTopologySampler(seed=seed)
        if recipe.system=='fission':np.random.RandomState(seed).shuffle(old._L1_GRID)
    sobol=None
    if recipe.sampler in ('gibbs_sobol','pool'):
        sobol=qmc.Sobol(d=16 if recipe.sampler=='gibbs_sobol' else 8,scramble=True,seed=seed).random_base2(math.ceil(math.log2(max(1024,count))))
    rows=[]
    for i in range(count):
        level=1+i%len(recipe.ratios)
        if new:
            result=new_result(sampler,level)
        elif recipe.sampler in ('random','random_ko_interval'):
            result=old.sample_pure_random_mutant(level,rng,enable_ko_interval=recipe.sampler=='random_ko_interval')
        elif recipe.sampler=='gibbs':result=old.sample_gibbs_mutant(level,kernel,rng)
        elif recipe.sampler=='gibbs_sobol':result=old.sample_gibbs_sobol_mutant(level,kernel,sobol[i],rng)
        elif recipe.sampler=='pool':
            output=old.sample_pure_random_candidate(level,sobol[i],rng)
            result=(output[0],output[1],'')
        elif recipe.sampler=='kdpp':result=old.sample_kdpp_mutant(level,kernel,rng)
        elif recipe.sampler=='kdpp_curvature':result=old.sample_signed_kdpp_mutant(level,kernel,rng)
        elif recipe.sampler=='motif':
            genes,modes=kernel.sample_motif(level)
            result=old.build_perturbed_sample(genes,modes,rng)
        elif recipe.system=='fission':result=old.sample_random_mutant(level,rng)
        else:result=getattr(old,CASES[name][1])(level=level)
        uses_rng=recipe.sampler in ('random','random_ko_interval','gibbs','gibbs_sobol','kdpp','kdpp_curvature','motif','pool','fission_random')
        rows.append(record(level,result,rng if uses_rng else None,kernel))
    return rows


class Captured(Exception):
    pass


def old_driver(name,seed,count):
    old=module(CASES[name][0]);recipe=RECIPES[name]
    reset(old,seed,recipe.mode_freq)
    fn=getattr(old,CASES[name][1]);sig=inspect.signature(fn)
    rows=[]
    total=math.ceil(count*recipe.queue_multiplier)
    depth=0
    def capture(*args,**kwargs):
        nonlocal depth
        depth+=1
        try:result=fn(*args,**kwargs)
        finally:depth-=1
        if depth:return result
        bound=sig.bind(*args,**kwargs).arguments
        rng=bound.get('rng')
        kernel=next((v for v in bound.values() if hasattr(v,'rng')),None)
        level=bound.get('level')
        if recipe.sampler=='motif':
            level=len(bound['genes'])
            kernel=inspect.currentframe().f_back.f_locals['sampler']
        if recipe.sampler=='pool':result=(result[0],result[1],'')
        rows.append(record(level,result,rng,kernel))
        if len(rows)==total:raise Captured()
        return result
    driver=getattr(old,CASES[name][2]);params=inspect.signature(driver).parameters
    values={'n_synthetic':count,'num_samples':count,'n_total':count,'seed':seed,
            'num_workers':1,'max_workers':1,'output_path':'/dev/null',
            'output_dir':'/tmp','output_name':'sampler_audit_never_written.npz',
            'level_ratios':recipe.ratios[:4] if recipe.sampler=='motif' else recipe.ratios,
            'ratios':recipe.ratios,'mode_freq':recipe.mode_freq,
            'enable_ko_interval':recipe.sampler=='random_ko_interval'}
    kwargs={k:v for k,v in values.items() if k in params}
    if recipe.sampler=='motif':kwargs['level_ratios']=(.1,.3,.3,.3)
    with ExitStack() as stack:
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(patch.object(old,CASES[name][1],capture))
        for obj in (old,getattr(old,'ma',None)):
            if obj is None:continue
            for field in ('is_duplicate_real_mutant','is_duplicate'):
                if hasattr(obj,field):stack.enter_context(patch.object(obj,field,return_value=False))
        if recipe.system=='fission':stack.enter_context(patch.object(old,'derive_wt_birth_state',return_value=np.zeros(16)))
        # Guard any accidental progression past the requested sampling prefix.
        stack.enter_context(patch.object(np,'savez_compressed',side_effect=AssertionError('Audit must not write datasets')))
        try:driver(**kwargs)
        except Captured:pass
    assert len(rows)==total,(name,len(rows),total)
    return rows


def new_driver(name,seed,count):
    recipe=RECIPES[name]
    cfg=GenerationConfig(name,n_samples=count,seed=seed,include_real=False,mode_freq=recipe.mode_freq)
    np.random.seed(seed);random.seed(seed)
    sampler=make_sampler(recipe,cfg)
    rows=[]
    original=sampler.sample
    def capture(level):
        overrides,description,refinement,pairs=original(level)
        kind=recipe.sampler
        result=overrides if kind=='motif' else ((overrides,description,'') if kind=='pool' else ((overrides,description,refinement) if kind=='bams' else (overrides,description)))
        uses_rng=kind in ('random','random_ko_interval','gibbs','gibbs_sobol','kdpp','kdpp_curvature','motif','pool','fission_random')
        rows.append(record(level,result,sampler.rng if uses_rng else None,getattr(sampler, 'kernel', None)))
        return overrides,description,refinement,pairs
    with patch.object(sampler,'sample',capture),patch('budding_yeast_v2.data.systems.generation.ensure_candidate',side_effect=lambda candidate,*args: candidate):
        prepare_candidates(recipe,cfg,sampler,Counter())
    return rows


def compare(left,right):
    diffs=[]
    for i,(a,b) in enumerate(zip(left,right)):
        fields=[key for key in a if digest(a[key])!=digest(b[key])]
        if fields and len(diffs)<3:
            diffs.append({'index':i,'fields':fields,'old':a,'new':b})
    return {'old_count':len(left),'new_count':len(right),'equal':digest(left)==digest(right),
            'matching_rows':sum(digest(a)==digest(b) for a,b in zip(left,right)),
            'old_sha256':digest(left),'new_sha256':digest(right),'first_differences':diffs}


def special_stream(kind,seed,count,new,fold='tier2',mode='flat'):
    source={'wt':'mass_wt_blind_sampler','twins':'generate_distribution_ablation_twins','targeted':'generate_bams_ablation_datasets'}[kind]
    mode=mode or 'class'
    old=module(source);reset(old,seed,mode)
    backend=BuddingSystem('mass','mass') if kind=='wt' else BuddingSystem()
    if kind=='wt':
        target=module('systems.budding.samplers.wt_blind') if new else old
        reset(target,seed,mode);target.FOLD_DIST=fold
        bands=target.make_samplers()
    rows=[]
    if kind=='targeted' and not new:
        with patch.object(old,'is_duplicate_real_mutant',return_value=False),redirect_stdout(io.StringIO()):
            a,b=old.generate_targeted_pairs(n_samples=count,seed=seed)
        # Per-draw state is checked by the independent targeted functions below;
        # the real driver is additionally checked for the complete paired result.
        return [{'result':[plain(x[1]),plain(y[1])]} for x,y in zip(a,b)],rng_state(np.random)
    for i in range(count):
        level=1+i%4
        if kind=='wt':
            candidate=target.build_candidate(level,bands,ma=backend) if new else target.build_candidate(level,bands)
            result=candidate
        elif kind=='twins':
            result=module('systems.budding.samplers.twins').sample_twin_mutant_pair(level,backend.defaults(),ma=backend) if new else old.sample_twin_mutant_pair(level,backend.defaults())
        else:
            a,b,_=module('systems.budding.samplers.targeted_ablation').targeted_pair(i)
            rows.append({'result':[plain(a),plain(b)]});continue
        rows.append({'result':plain(result),'rng':rng_state(np.random)})
    return rows,rng_state(np.random)


def wt_driver(seed,new):
    """Execute the real pilot/ledger loop, stopping before ODE/export."""
    source=module('systems.budding.samplers.wt_pipeline') if new else module('mass_wt_blind_sampler')
    target=module('systems.budding.samplers.wt_blind') if new else source
    reset(target,seed,'flat')
    fn=source.prepare_wt_candidates if new else source.generate_wt_blind_dataset
    lines,start=inspect.getsourcelines(fn)
    body=ast.parse(textwrap.dedent(''.join(lines))).body[0].body
    index=next(i for i,node in enumerate(body) if isinstance(node,ast.While) and 'accepted' in ast.unparse(node.test))
    stop_line=start-1+body[index+1].lineno
    captured={}
    def trace(frame,event,arg):
        if frame.f_code is not fn.__code__:return None
        if event=='line' and frame.f_lineno==stop_line:
            captured['rows']=frame.f_locals['accepted']
            captured['attempts']=frame.f_locals['attempts']
            captured['rng']=rng_state(np.random)
            raise Captured()
        return trace
    cfg=GenerationConfig('budding_wt_blind',n_samples=16,pilot_samples=32,seed=seed,mode_freq='flat',include_real=False)
    sampler=make_sampler(RECIPES[cfg.recipe],cfg) if new else None
    previous=sys.gettrace()
    with patch.object(real_mutants,'is_duplicate_real_mutant',return_value=False),redirect_stdout(io.StringIO()),ExitStack() as stack:
        if not new:stack.enter_context(patch.object(source,'is_duplicate_real_mutant',return_value=False))
        try:
            sys.settrace(trace)
            if new:fn(sampler,cfg,Counter())
            else:fn(n_synthetic=16,n_pilot=32,seed=seed,include_real=False,fold_dist='tier2',output_dir='/tmp')
        except Captured:pass
        finally:sys.settrace(previous)
    assert 'rows' in captured
    return captured


def run(seeds,count,names):
    rows=[]
    for name in names:
        for seed in seeds:
            for stage,left_fn,right_fn in (('raw',lambda:raw_stream(name,seed,count,False),lambda:raw_stream(name,seed,count,True)),
                                           ('driver_no_collisions',lambda:old_driver(name,seed,count),lambda:new_driver(name,seed,count))):
                try:
                    result=compare(left_fn(),right_fn())
                except Exception as exc:
                    result={'equal':False,'error':repr(exc)}
                rows.append(dict(recipe=name,seed=seed,stage=stage,**result))
                print(name,seed,stage,'PASS' if result['equal'] else 'FAIL',flush=True)
    for kind in ('wt','twins','targeted'):
        for seed in seeds:
            for fold,mode in ([('tier2','flat'),('flat','flat'),('tier2','s5'),('flat','s5')] if kind=='wt' else [('','')]):
                try:
                    a,sa=special_stream(kind,seed,count,False,fold,mode)
                    b,sb=special_stream(kind,seed,count,True,fold,mode)
                    result=compare(a,b)
                    result['rng_equal']=sa==sb
                    result['equal'] &= sa==sb
                except Exception as exc:result={'equal':False,'error':repr(exc)}
                rows.append(dict(recipe=kind,seed=seed,stage=f'raw_{fold}_{mode}',**result))
                print(kind,seed,fold,mode,'PASS' if result['equal'] else 'FAIL',flush=True)
    for seed in seeds:
        try:
            a,b=wt_driver(seed,False),wt_driver(seed,True)
            result=compare([dict(result=plain(c)) for c in a['rows']],[dict(result=plain(c)) for c in b['rows']])
            result.update(rng_equal=a['rng']==b['rng'],old_attempts=a['attempts'],new_attempts=b['attempts'])
            result['equal'] &= result['rng_equal'] and a['attempts']==b['attempts']
        except Exception as exc:result={'equal':False,'error':repr(exc)}
        rows.append(dict(recipe='wt',seed=seed,stage='pilot_and_ledger_no_collisions',**result))
        print('wt',seed,'pilot_and_ledger','PASS' if result['equal'] else 'FAIL',flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,7,42])
    parser.add_argument('--samples',type=int,default=256)
    parser.add_argument('--recipes',nargs='+',choices=tuple(CASES),default=list(CASES))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.samples<1:parser.error('--samples must be positive')
    rows=run(args.seeds,args.samples,args.recipes)
    result={'seeds':args.seeds,'samples_per_case':args.samples,'comparisons':len(rows),
            'passed':sum(row['equal'] for row in rows),'results':rows,
            'scope':'Exact candidates and RNG states; driver comparisons bypass collision policy. No ODE or final dataset equivalence claim.'}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
    return 0 if result['passed']==result['comparisons'] else 1


if __name__=='__main__':
    sys.exit(main())
