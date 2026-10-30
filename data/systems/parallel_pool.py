"""Deterministic parallel pool proposals with global collision/identity checks.

Chunk seeds depend on root seed, level, chunk and retry, never worker completion
order or worker count. Sampling profiles and rescue predicates stay in adapters.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import multiprocessing as mp
import random

import numpy as np

from .registry import get_adapter, get_recipe, make_sampler
from .sampling_policy import draw_candidate, valid_candidate
from ..simulation_guards import bounded_ordered_map

RNG_CONTRACT = 'pool_seedsequence_chunks256_global_dedup_v1'
CHUNK_SIZE = 256


def _proposals(payload):
    config, batch_id, retry, requests = payload
    seed=int(np.random.SeedSequence([config.seed,batch_id,retry,9381]).generate_state(1)[0])
    local=replace(config,seed=seed)
    np.random.seed(seed);random.seed(seed)
    recipe=get_recipe(config.recipe)
    sampler=make_sampler(recipe,local)
    adapter=get_adapter(recipe.system)
    references=adapter.references()
    report=Counter()
    rows=[draw_candidate(sampler,level,number,report,references) for number,level in requests]
    return rows,report


def prepare_parallel_pool(recipe,config,report):
    from .generation import ensure_candidate,level_schedule
    if recipe.system == 'budding' and recipe.sampler == 'pool':
        from .budding.al_pool import prepare_quota_pool
        return prepare_quota_pool(recipe,config,report)
    if not config.parameter_only or recipe.system not in ('budding','fission'):
        raise ValueError('Parallel proposals are reserved for input pools')
    adapter=get_adapter(recipe.system)
    schedule=level_schedule(config.n_samples,config.level_ratios,np.random.RandomState(config.seed),
                            rounding=recipe.quota_rounding,shuffle=recipe.shuffle_levels)
    pending=[(i+1,int(level)) for i,level in enumerate(schedule)]
    references=adapter.references()
    coordinator=make_sampler(recipe,config)
    accepted={};seen=set()
    executor=ProcessPoolExecutor(max_workers=config.workers,mp_context=mp.get_context('spawn')) if config.workers>1 else None
    try:
        for retry in range(config.max_attempts or 100):
            if not pending:break
            tasks=[(config,i//CHUNK_SIZE,retry,pending[i:i+CHUNK_SIZE]) for i in range(0,len(pending),CHUNK_SIZE)]
            print(f'[{recipe.system} pool] pass={retry+1} pending={len(pending)} workers={config.workers}',flush=True)
            rows=bounded_ordered_map(executor,_proposals,tasks,stall_timeout=300.) if executor else map(_proposals,tasks)
            remaining=[]
            for batch_index,(proposals,stats) in enumerate(rows):
                report.update(stats)
                for candidate in proposals:
                    rng=np.random.default_rng(np.random.SeedSequence([config.seed,candidate.sample_id,retry,5281]))
                    try:
                        candidate=ensure_candidate(candidate,coordinator,report,references,adapter=adapter,rescue_rng=rng)
                    except RuntimeError as exc:
                        if 'Collision rescue exhausted' not in str(exc):raise
                        report['collision_exhausted']+=1
                        remaining.append((candidate.sample_id,candidate.level));continue
                    candidate=valid_candidate(candidate,recipe.system)
                    if candidate is None:
                        raise RuntimeError('Rescue changed requested module count')
                    if recipe.system=='budding':
                        row=adapter.parameter_result(candidate,None)
                        signature=hashlib.sha256(np.r_[row.params,row.y_init].astype(np.float64).tobytes()).digest()
                        if signature in seen:
                            report['pool_duplicates']+=1
                            remaining.append((candidate.sample_id,candidate.level));continue
                        seen.add(signature)
                    adapter.add_reference(references,candidate.overrides)
                    accepted[candidate.sample_id]=candidate
                if (batch_index+1)%20==0 or batch_index+1==len(tasks):
                    print(f'[{recipe.system} pool] accepted={len(accepted)}/{config.n_samples} retry={len(remaining)}',flush=True)
            pending=remaining
        if pending:raise RuntimeError(f'Parallel pool exhausted with {len(pending)} missing rows')
    finally:
        if executor:executor.shutdown(wait=True,cancel_futures=True)
    report.update(sampled_synthetic=len(accepted))
    report['rng_contract']=RNG_CONTRACT
    report['proposal_chunk_size']=CHUNK_SIZE
    report['proposal_workers']=config.workers
    return [accepted[i] for i in sorted(accepted)]
