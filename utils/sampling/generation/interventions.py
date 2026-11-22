"""Single-axis canonical mutation redraws; no synthetic physical-vector interpolation."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import numpy as np

from budding_yeast_v2.data.systems.budding.catalog import touch_set
from budding_yeast_v2.data.systems.budding.real_mutants import effective_inputs
from budding_yeast_v2.data.systems.sampling_policy import valid_candidate
from ..contracts import PoolInputs


def intervention_bank(domain,controller,count):
    if domain.kind!='mutation':raise ValueError('Adaptive intervention generation currently requires mutation proposals')
    domain.verify();backend=domain.proposals.backend;defaults=domain.proposals.defaults
    out=[];signatures=[];vectors=[];local=set();rng=controller.rng
    probability=controller.anchor_probabilities();train=controller.inputs
    exploration=max(1,int(np.ceil(.2*count)))
    path_quota=count-exploration
    for _ in range(max(100,count*30)):
        if len(out)==path_quota:break
        j=int(rng.choice(len(train.ids),p=probability));meta=train.metadata[j]
        tokens=meta.get('pairs',meta.get('genes',()))
        modules=sorted({g for g,_ in tokens});rng.shuffle(modules)
        if not modules:continue
        gene=modules[0];controls=[(g,m) for g,m in tokens if g==gene]
        touched=set().union(*(touch_set(g,m) for g,m in controls))
        # Overlapping controls cannot be independently redrawn.
        if any(touched & touch_set(g,m) for g,m in tokens if g!=gene):continue
        original=meta.get('_effective_overrides',meta.get('original_overrides'))
        if original is None:raise ValueError('Queried anchor lacks actual overrides')
        changes={};draws=[]
        for g,mode in controls:
            alpha=backend.sample_scaling_factor(mode,g,rng=rng)
            backend.apply_gene_mutation(g,mode,alpha,defaults,defaults.copy(),changes,rng=rng)
            draws.append(dict(gene=g,mode=mode,alpha=float(alpha)))
        if not set(changes)<=touched:
            domain.counts['unsupported_axis_footprint']+=1;continue
        overrides={k:v for k,v in original.items() if k not in touched};overrides.update(changes)
        # Environment is never used as a rescue/edit direction.
        if overrides.get('mdt',defaults['mdt'])!=original.get('mdt',defaults['mdt']):
            raise ValueError('Intervention changed environment')
        vector=effective_inputs(overrides,defaults);signature=hashlib.sha256(vector.astype('<f8').tobytes()).hexdigest()
        base=effective_inputs(original,defaults)
        if np.array_equal(vector,base):domain.counts['fixed_axis_no_change']+=1;continue
        if domain.proposals.collides(vector) or signature in domain.forbidden:
            domain.counts['intervention_excluded_collision']+=1;continue
        if signature in domain.seen or signature in local:domain.counts['duplicate_intervention']+=1;continue
        # Build a Candidate through the same type/validity contract, not an ODE fork.
        from budding_yeast_v2.data.systems.contracts import Candidate
        index=domain.next_index;domain.next_index+=1
        metadata=deepcopy(meta);metadata.update(original_overrides=dict(overrides),rescued=False,
            generation_proposal_index=index,source_candidate_id=1_000_000_000+index,
            adaptive_anchor_id=int(train.ids[j]),adaptive_axis=gene,
            sampling_paradigm='single_axis_canonical_mutation_redraw',
            proposal_strength_draws=[r for r in meta.get('proposal_strength_draws',[]) if r['gene']!=gene]+draws)
        metadata.pop('_effective_overrides',None)
        metadata['mutation_parameters']=[dict(gene=g,mode=m,overrides={k:overrides[k]
            for k in sorted(touch_set(g,m)) if k in overrides}) for g,m in tokens]
        candidate=valid_candidate(Candidate(sample_id=index+1,name=f'Adaptive-mutation-{index:09d}',overrides=overrides,level=len(modules),metadata=metadata),'budding')
        if candidate is None:domain.counts['invalid_intervention']+=1;continue
        _,y0,_,_=backend.prepare_initial_conditions(candidate.overrides)
        p=np.array([overrides.get(k,defaults[k]) for k in backend.PARAM_KEYS])
        out.append(candidate);signatures.append(signature);vectors.append((p,y0));local.add(signature)
        domain.counts['adaptive_interventions']+=1
    fallback=max(0,path_quota-len(out))
    global_count=count-len(out)
    # Bounded attempts; fixed KO/db atoms have finite capacity. Explicit exploration.
    for _ in range(max(100,count*100)):
        if len(out)==count:break
        c,s,v=domain.bank(1)
        if s[0] in local:continue
        c[0]=replace(c[0],metadata=dict(c[0].metadata,adaptive_fallback=True))
        out+=c;signatures+=s;vectors.append((v.params[0],v.initial[0]));local.add(s[0])
    if len(out)!=count:raise RuntimeError('Adaptive fallback exhausted bounded unique proposals')
    domain.counts['adaptive_fallback_proposals']+=fallback
    domain.counts['adaptive_global_proposals']+=global_count
    return out,signatures,PoolInputs(np.arange(count),np.asarray([p for p,_ in vectors]),
        np.asarray([y for _,y in vectors]),tuple(dict(c.metadata,_effective_overrides=dict(c.overrides)) for c in out))
