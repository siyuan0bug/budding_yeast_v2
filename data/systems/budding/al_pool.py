"""Capacity-aware module-set / allele quotas for unsolved Budding AL pools.

No real-mutant test errors or target identities enter quota planning. Independent
scrambled Sobol streams live inside fixed allele combinations. Global effective
input deduplication follows deterministic module order, independent of workers.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import multiprocessing as mp

import numpy as np
from scipy.stats import qmc

from ..contracts import Candidate
from ..registry import candidate_name
from ..sampling_policy import valid_candidate
from ...simulation_guards import bounded_ordered_map
from .quotas import enumerate_combinations, waterfill_quotas
from .specification import BuddingSystem
from .real_mutants import effective_inputs, is_duplicate_real_mutant
from .samplers.rescue import parse_pairs, rescue_budding
from .distributions.ranges import copy_support,GAL_MODES,AMPLITUDE_MODES,RANGE_FINGERPRINT

POOL_CONTRACT = 'budding_module_allele_equal_gene_ranges_2026_10_29'


def signature(overrides, backend):
    return hashlib.sha256(effective_inputs(overrides, backend.defaults()).astype('<f8').tobytes()).digest()


def finite_options(cell, backend):
    """Enumerate small finite supports exactly, with original proposal weights."""
    upper = cell['finite_capacity_upper']
    if upper is None or upper > 180:
        return None
    pairs = parse_pairs(cell['alleles'])
    defaults = backend.defaults()
    draws = [copy_support(gene,mode) if mode in ('MC','GAL-MC') else (1.,) for gene,mode in pairs]
    rows = {}
    for alphas in itertools.product(*draws):
        overrides = {}
        for (gene,mode),alpha in zip(pairs,alphas):
            backend.apply_gene_mutation(gene,mode,float(alpha),defaults,defaults.copy(),overrides)
        environments=((150.,1.),) if any(m in GAL_MODES for _,m in pairs) else ((90.,.8),(150.,.15),(160.,.05))
        for mdt,probability in environments:
            value = dict(overrides,mdt=mdt)
            if is_duplicate_real_mutant(value,defaults):
                continue
            key = signature(value,backend)
            weight = probability*np.prod([1./a for (_,m),a in zip(pairs,alphas) if m in ('MC','GAL-MC')])
            if key in rows:
                rows[key][1] += weight
            else:
                rows[key] = [value,float(weight),list(map(float,alphas))]
    return [(key,*value) for key,value in sorted(rows.items())]


def plan_pool(size, seed=42):
    cells, counts = enumerate_combinations()
    groups = defaultdict(list)
    for cell in cells:
        groups[cell['modules']].append(cell)
    backend = BuddingSystem()
    planned = []
    for index,(modules,rows) in enumerate(sorted(groups.items())):
        if any(row['has_continuous_strength'] for row in rows):
            capacity = np.inf
        else:
            supports = [finite_options(row,backend) for row in rows]
            if any(support is None for support in supports):
                raise RuntimeError('Finite module support exceeds enumeration contract')
            capacity = len({option[0] for support in supports for option in support})
        planned.append(dict(index=index,modules=modules,level=rows[0]['level'],cells=rows,capacity=capacity))
    quotas = waterfill_quotas([row['capacity'] for row in planned],size,seed)
    for group,quota in zip(planned,quotas):
        group['quota'] = int(quota)
    return planned,counts


class CellStream:
    def __init__(self, seed):
        self.sobol=qmc.Sobol(16,scramble=True,seed=seed)
        self.points=None;self.position=0
    def next(self):
        if self.points is None or self.position==len(self.points):
            self.points=self.sobol.random(64);self.position=0
        point=self.points[self.position];self.position+=1
        return point


class CoordinateRNG:
    """Uniform/branch draws inside mutation rules use the same cell's Sobol row."""
    def __init__(self, point):self.point=point;self.index=0
    def random(self):
        if self.index>=len(self.point)-1:raise RuntimeError('Sobol coordinates exhausted')
        value=float(self.point[self.index]);self.index+=1
        return value
    def uniform(self,low=0.,high=1.):return low+(high-low)*self.random()


def _propose_group(task):
    group,config,retry,prior_counts,banned = task
    backend=BuddingSystem();defaults=backend.defaults();stats=Counter()
    cells=group['cells'];counts=np.asarray(prior_counts,dtype=int).copy()
    capacities=[];supports=[];streams=[];pairs=[]
    for i,cell in enumerate(cells):
        support=finite_options(cell,backend)
        if support is not None:
            support=[option for option in support if option[0] not in banned]
            cap=int(counts[i])+len(support)
        else:
            cap=np.inf if cell['finite_capacity_upper'] is None else cell['finite_capacity_upper']
        supports.append(support);capacities.append(cap);pairs.append(parse_pairs(cell['alleles']))
        seed=int(np.random.SeedSequence([config.seed,group['index'],i,retry,731]).generate_state(1)[0])
        streams.append([seed,None])
    capacities=np.asarray(capacities,dtype=float)
    rng=np.random.RandomState(np.random.SeedSequence([config.seed,group['index'],retry,912]).generate_state(1)[0])
    ordering=rng.permutation(len(cells));rank=np.argsort(ordering)
    cell_lookup={tuple(sorted(p)):i for i,p in enumerate(pairs)}
    accepted=[];seen=set(banned);attempts=np.zeros(len(cells),dtype=int)
    limit=config.max_attempts or 2000
    while counts.sum()<group['quota']:
        available=np.flatnonzero(counts<capacities)
        if not len(available):break
        smallest=counts[available].min()
        tied=available[counts[available]==smallest]
        i=int(tied[np.argmin(rank[tied])]);attempts[i]+=1
        if attempts[i]>limit+20*group['quota']:
            capacities[i]=counts[i];stats['bounded_cell_exhaustion']+=1;continue
        if streams[i][1] is None:streams[i][1]=CellStream(streams[i][0])
        point=streams[i][1].next();actual=i;description=cells[i]['alleles'].replace(';','+')
        was_rescued=False
        if supports[i] is not None:
            if not supports[i]:capacities[i]=counts[i];continue
            weights=np.array([entry[2] for entry in supports[i]],dtype=float)
            selected=min(int(np.searchsorted(np.cumsum(weights)/weights.sum(),point[0])),len(weights)-1)
            key,overrides,_,alphas=supports[i].pop(selected)
            overrides=dict(overrides)
            # Aliased finite cells can share an effective input within this set.
            if key in seen:stats['local_duplicates']+=1;continue
        else:
            overrides={};alphas=[];coordinates=CoordinateRNG(point)
            for gene,mode in pairs[i]:
                alpha=(backend.sample_scaling_factor_sobol(mode,gene,coordinates.random())
                       if mode in AMPLITUDE_MODES else 1.)
                alphas.append(float(alpha))
                backend.apply_gene_mutation(gene,mode,alpha,defaults,defaults.copy(),overrides,rng=coordinates)
            overrides['mdt']=150. if any(m in GAL_MODES for _,m in pairs[i]) else (90. if point[-1]<.8 else 150. if point[-1]<.95 else 160.)
            if is_duplicate_real_mutant(overrides,defaults):
                stats['collisions']+=1
                for _ in range(30):
                    proposed,new_description=rescue_budding(overrides,description,backend,rng)
                    new_pairs=parse_pairs(new_description)
                    target=cell_lookup.get(tuple(sorted(new_pairs)))
                    if target is None or counts[target]>=capacities[target]:continue
                    if is_duplicate_real_mutant(proposed,defaults):continue
                    # A rescue must fill a currently least represented mode cell.
                    if counts[target]>smallest:continue
                    overrides,description,actual=proposed,new_description,target
                    was_rescued=True;stats['rescued']+=1;break
                else:stats['collision_redraws']+=1;continue
            key=signature(overrides,backend)
            if key in seen:stats['local_duplicates']+=1;continue
        seen.add(key);counts[actual]+=1
        accepted.append((actual,overrides,description,key,dict(
            description=description,pairs=pairs[actual],quota_module_set=group['modules'],
            quota_allele=cells[actual]['alleles'],requested_quota_allele=cells[i]['alleles'],
            proposal_strength_draws=alphas,rescued=was_rescued,
            pool_generation_contract=POOL_CONTRACT,sobol_retry=retry)))
    return group['index'],accepted,counts.tolist(),capacities.tolist(),stats


def prepare_quota_pool(recipe,config,report):
    if recipe.system!='budding' or recipe.sampler!='pool' or not config.parameter_only:
        raise ValueError('Balanced quotas apply only to the Budding AL parameter pool')
    if (config.mutation_profile or recipe.mutation)!='mutant_aware' or (config.distribution_profile or recipe.distribution)!='mutant_aware':
        raise ValueError('Balanced Budding AL pools use the mutant-aware mutation and distribution profiles')
    groups,enumeration=plan_pool(config.n_samples,config.seed)
    counts=[[0]*len(group['cells']) for group in groups]
    accepted=[[] for _ in groups];banned=[set() for _ in groups];seen=set()
    active=[i for i,group in enumerate(groups) if group['quota']]
    executor=ProcessPoolExecutor(max_workers=config.workers,mp_context=mp.get_context('spawn')) if config.workers>1 else None
    try:
        for retry in range(20):
            if not active:break
            print(f'[budding quota pool] pass={retry+1} groups={len(active)} accepted={len(seen)}/{config.n_samples} workers={config.workers}',flush=True)
            tasks=[(groups[i],config,retry,counts[i],banned[i]) for i in active]
            outputs=bounded_ordered_map(executor,_propose_group,tasks,stall_timeout=600.) if executor else map(_propose_group,tasks)
            remaining=[]
            for completed,(i,rows,_,capacities,stats) in enumerate(outputs,1):
                report.update(stats)
                for cell,ovr,description,key,metadata in rows:
                    banned[i].add(key)
                    if key in seen:report['global_duplicates']+=1;continue
                    seen.add(key);counts[i][cell]+=1
                    accepted[i].append((ovr,description,metadata))
                if len(accepted[i])<groups[i]['quota']:
                    remaining.append(i)
                if completed%50==0 or completed==len(active):
                    print(f'[budding quota pool] groups_done={completed}/{len(active)} accepted={len(seen)}/{config.n_samples}',flush=True)
            if remaining==active and retry>2:
                # Do not silently relax a module quota or admit duplicate inputs.
                if all(sum(counts[i])==0 for i in remaining):
                    raise RuntimeError('Balanced pool capacity exhausted for required module sets')
            active=remaining
        if active:raise RuntimeError(f'Balanced pool has unfilled module quotas: {[groups[i]["modules"] for i in active]}')
    finally:
        if executor:executor.shutdown(wait=True,cancel_futures=True)
    candidates=[];allocation=[];mode_allocation=[]
    for group,rows,cell_counts in zip(groups,accepted,counts):
        allocation.append(dict(modules=group['modules'],level=group['level'],quota=group['quota'],accepted=len(rows),
            capacity=None if np.isinf(group['capacity']) else int(group['capacity'])))
        for cell,number in zip(group['cells'],cell_counts):
            mode_allocation.append(dict(modules=group['modules'],alleles=cell['alleles'],accepted=number))
        for overrides,description,metadata in rows:
            number=len(candidates)+1
            candidate=valid_candidate(Candidate(number,candidate_name('pool',description,group['level'],number),
                overrides,group['level'],metadata=metadata),'budding')
            if candidate is None:raise RuntimeError('Balanced pool exported an invalid allele/module assignment')
            candidates.append(candidate)
    # File order is randomized reproducibly, rather than grouping all L1 rows first.
    np.random.RandomState(config.seed).shuffle(candidates)
    from dataclasses import replace
    candidates=[replace(row,sample_id=i+1,name=candidate_name('pool',row.metadata['description'],row.level,i+1)) for i,row in enumerate(candidates)]
    dict.update(report,sampled_synthetic=len(candidates),rng_contract=POOL_CONTRACT,
        sampling_range_fingerprint=RANGE_FINGERPRINT,
        pool_generation_contract=POOL_CONTRACT,quota_enumeration=enumeration,
        module_quotas=allocation,allele_counts=mode_allocation,
        environment_non_gal_probabilities=[.8,.15,.05],proposal_workers=config.workers,
        global_unique_inputs=len(seen),actual_level_counts=dict(Counter(row.level for row in candidates)))
    return candidates


def validate_pool_contract(archive):
    """Reject modular caches made under the superseded random-quota/catalog rules."""
    from .catalog import CATALOG_VERSION
    if str(archive.get('pool_generation_contract',''))!=POOL_CONTRACT:
        raise ValueError('Budding candidate pool uses superseded generation/range rules; regenerate the pool and initial dataset')
    if str(archive.get('catalog_version',''))!=CATALOG_VERSION:
        raise ValueError('Budding candidate pool catalog changed; regenerate the balanced pool')
    if str(archive.get('sampling_range_fingerprint',''))!=RANGE_FINGERPRINT:
        raise ValueError('Budding candidate pool strength ranges changed; regenerate the pool and initial dataset')
