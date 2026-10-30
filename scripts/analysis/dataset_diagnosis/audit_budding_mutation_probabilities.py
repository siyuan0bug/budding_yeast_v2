"""Audit current mutation probabilities without changing production samplers.

Measures proposal -> conflict/count/real-rescue acceptance and an actual globally
unique parameter pool. No ODE integration, training, or original-file writes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.systems.budding import catalog
from budding_yeast_v2.data.systems.budding.samplers.rescue import parse_pairs
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.contracts import GenerationConfig
from budding_yeast_v2.data.systems.registry import get_recipe,resolve_config,make_sampler,get_adapter
from budding_yeast_v2.data.systems.sampling_policy import draw_candidate
from budding_yeast_v2.data.systems.generation import level_schedule


def theoretical_modes(gene,arm):
    group=catalog.GROUP_BY_NAME[gene];modes=list(group['modes'])
    if arm=='s5':w=[group['modes'][m] for m in modes]
    elif arm=='class':w=[catalog.MODE_CLASS_QUOTA[catalog.MODE_CLASS[m]] for m in modes]
    elif arm=='role':
        aff=catalog.ROLE_AFFINITY.get(catalog.GENE_ROLE[gene],{})
        w=[aff.get(catalog.MODE_CLASS[m],1.) for m in modes]
    else:w=np.ones(len(modes))
    w=np.asarray(w,dtype=float);return dict(zip(modes,w/w.sum()))


def pair_conflicts(pairs):
    touched={};overlap=[]
    for g,m in pairs:
        for k in catalog.touch_set(g,m):
            if k in touched:overlap.append((touched[k],(g,m),k))
            touched[k]=(g,m)
    return overlap


def run_strategy(job):
    name,n,seed=job;np.random.seed(seed);random.seed(seed)
    cfg=resolve_config(GenerationConfig(name,n_samples=n,seed=seed,include_real=False,enable_bisection=False))
    recipe=get_recipe(name);sampler=make_sampler(recipe,cfg)
    schedule=level_schedule(n,cfg.level_ratios,sampler.schedule_rng(),shuffle=True)
    counts=Counter();totals=Counter();stats=Counter();examples=[];error_examples=[];raw=sampler.sample
    def record(stage,level,pairs,ovr):
        totals[(stage,int(level))]+=1
        for g,m in set(pairs):counts[(stage,int(level),g,m)]+=1
        conflicts=pair_conflicts(pairs)
        if conflicts:
            stats[stage+'_overlap_samples']+=1
            if len(examples)<12:examples.append(dict(stage=stage,level=int(level),pairs=pairs,overlaps=conflicts,overrides=ovr))
    def wrapped(level):
        result=raw(level);ovr,desc,_,pairs=result
        record('proposal',level,pairs or parse_pairs(desc),ovr)
        return result
    sampler.sample=wrapped
    for number,level in enumerate(schedule,1):
        try:candidate=draw_candidate(sampler,int(level),number,stats)
        except Exception as exc:
            stats['errors']+=1
            if len(error_examples)<10:error_examples.append(dict(level=int(level),number=number,error=repr(exc)))
            if len(examples)<12:examples.append(dict(level=int(level),error=repr(exc)))
            continue
        record('accepted',level,candidate.metadata['pairs'],candidate.overrides)
    rows=[dict(recipe=name,stage=stage,level=level,gene=g,mode=m,count=counts[stage,level,g,m],samples=totals[stage,level],inclusion_fraction=counts[stage,level,g,m]/max(1,totals[stage,level]))
          for stage in ('proposal','accepted') for level in (1,2,3) for g in catalog.GENE_NAMES for m in catalog.GROUP_BY_NAME[g]['modes']]
    print('strategy done',name,dict(stats),flush=True)
    return rows,dict(recipe=name,samples=n,seed=seed,statistics=dict(stats),stage_totals={f'{s}_L{l}':v for (s,l),v in totals.items()},examples=examples,error_examples=error_examples)


def strengths_and_footprints(out,n):
    rows=[];assignments=[];rng=np.random.RandomState(12345)
    for profile in ('mutant_aware','mass'):
        backend=BuddingSystem(profile,profile);wt=backend.defaults()
        for gene in catalog.GENE_NAMES:
            for mode in catalog.GROUP_BY_NAME[gene]['modes']:
                direct=np.array([backend.sample_scaling_factor(mode,gene,rng=rng) for _ in range(n)])
                quant=np.array([backend.sample_scaling_factor_sobol(mode,gene,(i+.5)/n) for i in range(n)])
                d=float(ks_2samp(direct,quant).statistic)
                rows.append(dict(profile=profile,gene=gene,mode=mode,n=n,direct_min=direct.min(),direct_max=direct.max(),quantile_min=quant.min(),quantile_max=quant.max(),distribution_D=d,finite=bool(np.isfinite(direct).all() and np.isfinite(quant).all())))
                union=set();changed=0
                for alpha in direct[:32]:
                    ovr={};backend.apply_gene_mutation(gene,mode,float(alpha),wt,wt.copy(),ovr,rng=rng);union.update(ovr)
                    _,ic,_,_=backend.prepare_initial_conditions(ovr)
                    changed+=any(v!=wt.get(k) for k,v in ovr.items())
                    if not np.isfinite(ic).all():raise ValueError((profile,gene,mode,'nonfinite initial'))
                assignments.append(dict(profile=profile,gene=gene,mode=mode,changed_assignments=changed,total=32,unexpected_footprint=';'.join(sorted(union-catalog.touch_set(gene,mode))),footprint=';'.join(sorted(union))))
    pd.DataFrame(rows).to_csv(out/'strength_distributions.csv',index=False)
    pd.DataFrame(assignments).to_csv(out/'mutation_assignments.csv',index=False)


def unique_pool(out,n,workers):
    from budding_yeast_v2.data.systems.parallel_pool import prepare_parallel_pool
    config=resolve_config(GenerationConfig('budding_pool',n_samples=n,seed=42,workers=workers,include_real=False,parameter_only=True))
    stats=Counter();candidates=prepare_parallel_pool(get_recipe(config.recipe),config,stats)
    count=Counter();level_counts=Counter();examples=[];adapter=get_adapter('budding')
    for c in candidates:
        level_counts[c.level]+=1
        for pair in set(c.metadata['pairs']):count[(c.level,*pair)]+=1
        if any(g in catalog.RESTORED_FUNCTIONAL_MODULES for g,m in c.metadata['pairs']) and len(examples)<30:
            examples.append(dict(name=c.name,level=c.level,overrides=c.overrides))
    frame=pd.DataFrame([dict(level=l,gene=g,mode=m,count=count[l,g,m],samples=level_counts[l],inclusion_fraction=count[l,g,m]/level_counts[l]) for l in (1,2,3) for g in catalog.GENE_NAMES for m in catalog.GROUP_BY_NAME[g]['modes']])
    frame.to_csv(out/'unique_pool_modes.csv',index=False)
    (out/'unique_pool_report.json').write_text(json.dumps(dict(samples=n,workers=workers,seed=42,level_counts=dict(level_counts),statistics=dict(stats),examples=examples),indent=2))
    print('unique pool done',n,dict(stats),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/analysis/mutation_probability_audit_2026-10-28'))
    parser.add_argument('--samples',type=int,default=3000);parser.add_argument('--random-samples',type=int,default=12000)
    parser.add_argument('--strength-samples',type=int,default=2048);parser.add_argument('--pool-size',type=int,default=20000);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths=sorted((ROOT/'data/systems/budding').rglob('*.py'))+[ROOT/'data/systems/sampling_policy.py',ROOT/'data/systems/parallel_pool.py',ROOT/'data/systems/generation.py']
    original={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    theory=[]
    for g in catalog.GENE_NAMES:
        for arm in ('flat','class','role','s5'):
            base=theoretical_modes(g,arm)
            for m,prob in base.items():
                included=prob
                if g in catalog.COMPOSITE_ELIGIBLE:
                    synth=[x for x in ('GAL-OE','MC') if x in base];struct=[x for x in ('db','PM','CA') if x in base]
                    if m in synth:included+=.25*sum(base[x] for x in struct)/len(synth)
                    if m in struct:included+=.25*sum(base[x] for x in synth)/len(struct)
                theory.append(dict(gene=g,mode=m,arm=arm,base_probability=prob,inclusion_with_composite=included))
    pd.DataFrame(theory).to_csv(args.output/'theoretical_mode_probabilities.csv',index=False)
    recipes=['budding_random','budding_pool','budding_mass_random','budding_mass_random_dist','budding_mutant_aware','budding_gibbs','budding_gibbs_sobol','budding_kdpp','budding_kdpp_curvature','budding_motif','budding_bams','budding_wt_blind']
    jobs=[(r,args.random_samples if r in recipes[:2] else min(args.samples,300) if r.endswith('wt_blind') else args.samples,42+i) for i,r in enumerate(recipes)]
    frames=[];reports=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as executor:
        futures={executor.submit(run_strategy,j):j[0] for j in jobs}
        for f in as_completed(futures):
            try:rows,report=f.result();frames.extend(rows);reports.append(report)
            except Exception as exc:reports.append(dict(recipe=futures[f],fatal_error=repr(exc)));print('FAILED',futures[f],repr(exc),flush=True)
    pd.DataFrame(frames).to_csv(args.output/'proposal_and_accepted_modes.csv',index=False)
    (args.output/'strategy_reports.json').write_text(json.dumps(reports,indent=2))
    strengths_and_footprints(args.output,args.strength_samples)
    unique_pool(args.output,args.pool_size,args.workers)
    assert original=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output/'provenance.json').write_text(json.dumps(dict(production_sha256=original,catalog_version=catalog.CATALOG_VERSION,gene_groups=len(catalog.GENE_NAMES),genetic_modules=len(set(catalog.GENE_NAMES)-{'MDT'}),legal_modes=sum(len(x['modes']) for x in catalog.GROUP_BY_NAME.values()),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},production_modified=False,ode_simulations=0),indent=2))


if __name__=='__main__':main()
