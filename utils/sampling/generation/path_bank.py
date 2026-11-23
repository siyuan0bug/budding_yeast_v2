"""Shared legal path planner for response ranking and its random-point control."""
from collections import defaultdict
from functools import lru_cache
import hashlib
import numpy as np
from .interventions import intervention_bank
from ..ours.response_v2.paths import path_id,axes


@lru_cache(None)
def mutable_axis(gene,modes):
    """Deterministic native-law probes; fixed atoms remain available globally."""
    from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
    backend=BuddingSystem();defaults=backend.defaults()
    seed=int.from_bytes(hashlib.sha256(repr((gene,modes)).encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed);variants=[]
    for _ in range(8):
        edits={}
        for mode in modes:
            alpha=backend.sample_scaling_factor(mode,gene,rng=rng)
            backend.apply_gene_mutation(gene,mode,alpha,defaults,defaults.copy(),edits,rng=rng)
        variants.append(edits)
    return any(v!=variants[0] for v in variants[1:])


def path_bank(domain,controller,count):
    rng=controller.rng;inputs=controller.inputs;groups=defaultdict(list)
    for j,(p,meta) in enumerate(zip(inputs.params,inputs.metadata)):
        for gene in axes(meta):
            modes=tuple(m for g,m in meta.get('pairs',meta.get('genes',())) if g==gene)
            if not mutable_axis(gene,modes):continue
            key=path_id(p,meta,gene,controller.system.param_names)
            if key is not None:groups[key].append((j,gene))
    # Select path identities, not rows: repeated anchors do not inflate a path's mass.
    existing=[k for k in groups if controller.index.support(k)];single=[k for k in groups if not controller.index.support(k)]
    maximum=min(16,max(2,int(np.ceil(count*.8/6))))
    chosen=[]
    for keys,quota in [(existing,max(0,maximum-2)),(single,maximum)]:
        if keys:
            chosen.extend(rng.choice(keys,min(quota,maximum-len(chosen),len(keys)),replace=False).tolist())
    if not chosen and existing:chosen=rng.choice(existing,min(maximum,len(existing)),replace=False).tolist()
    plan=[groups[k][int(rng.integers(len(groups[k])))] for k in chosen]
    candidates,signatures,bank=intervention_bank(domain,controller,count,path_plan=plan)
    return candidates,signatures,bank
