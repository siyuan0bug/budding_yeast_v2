"""Legal proposal and solver adapter. Never edits physical vectors or ODE rules."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from budding_yeast_v2.data.systems.budding.offline_pool import Proposals,simulate,source_hashes
from budding_yeast_v2.data.dataset_archive import file_sha256
from budding_yeast_v2.utils.sampling.contracts import PoolInputs


class FrozenBuddingDomain:
    """Only the canonical Proposals implementation creates legal inputs.

    Policies rank proposals or project their generated targets onto *new* legal
    proposals. They never interpolate physical params/ICs or fabricate mutations.
    LHS proposals use the reviewed frozen specification and its exact atoms.
    """
    def __init__(self,kind,seed,directory,*,lhs_config=None,forbidden=(),seen=(),next_index=0):
        if kind not in ('mutation','lhs'):raise ValueError('Unknown canonical proposal family')
        self.kind=kind;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.sources=source_hashes();self.forbidden=set(forbidden);self.seen=set(seen)
        self.next_index=int(next_index);self.counts=Counter()
        spec=None;spec_hash=None
        if kind=='lhs':
            if lhs_config is None:raise ValueError('LHS generation requires its frozen --lhs-config')
            spec=json.loads(Path(lhs_config).read_text());spec_hash=file_sha256(lhs_config)
            from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
            inventory=Path(lhs_config).parent/'inventory.json'
            if tuple(spec['positive_window_approved'])!=(.5,1.5):raise ValueError('LHS background window changed')
            if not inventory.exists() or json.loads(inventory.read_text())['range_fingerprint']!=RANGE_FINGERPRINT:
                raise ValueError('LHS configuration does not match frozen mutation ranges')
        self.proposals=Proposals(kind,int(seed),100000,self.directory,spec)
        self.proposals.config_sha=spec_hash

    def verify(self):
        if source_hashes()!=self.sources:raise RuntimeError('Canonical biology/generation sources changed during run')

    def bank(self,count,max_attempts=None):
        self.verify();candidates=[];signatures=[];vectors=[];local=set()
        limit=max_attempts or max(100,count*100)
        for _ in range(limit):
            if len(candidates)==count:break
            index=self.next_index;self.next_index+=1;self.counts['proposals']+=1
            proposed,rejection=self.proposals.propose(index)
            if proposed is None:self.counts[rejection]+=1;continue
            candidate,signature=proposed;signature=signature.hex()
            if signature in self.forbidden:self.counts['holdout_collision']+=1;continue
            if signature in self.seen or signature in local:self.counts['duplicate_proposal']+=1;continue
            # Names and provenance only. Overrides, prepared ICs and solver are untouched.
            candidate=replace(candidate,name=f'Generated-{self.kind}-{index:09d}',
                              metadata=dict(candidate.metadata,generation_proposal_index=index,
                                            source_candidate_id=1_000_000_000+index))
            candidates.append(candidate);signatures.append(signature);local.add(signature)
            p,y0,_,_=self.proposals.backend.prepare_initial_conditions(candidate.overrides)
            params=[candidate.overrides.get(k,self.proposals.defaults[k]) for k in self.proposals.backend.spec.param_names]
            vectors.append((params,y0))
            self.counts['rescued']+=int(candidate.metadata.get('rescued',False))
        if len(candidates)!=count:raise RuntimeError(f'Legal proposal bank exhausted: {len(candidates)}/{count}; {dict(self.counts)}')
        view=PoolInputs(np.arange(count),np.asarray([x[0] for x in vectors]),np.asarray([x[1] for x in vectors]))
        return candidates,signatures,view

    def consume(self,signature):
        if signature in self.seen or signature in self.forbidden:raise ValueError('Duplicate/forbidden query')
        self.seen.add(signature)

    def state(self):
        return dict(next_index=self.next_index,seen=sorted(self.seen),counts=dict(self.counts),sources=self.sources)


def project_legal_targets(targets,legal,mean,scale):
    """Unique nearest legal candidates, bounded memory; no biological mutation."""
    targets=np.asarray(targets,dtype=np.float64);legal=np.asarray(legal,dtype=np.float64)
    if targets.ndim!=2 or legal.ndim!=2 or targets.shape[1]!=legal.shape[1] or len(targets)>len(legal):
        raise ValueError('Invalid legal projection shapes')
    if not all(np.isfinite(x).all() for x in (targets,legal,mean,scale)) or (np.asarray(scale)<=0).any():
        raise ValueError('Nonfinite/invalid legal projection inputs')
    chosen=[];available=np.ones(len(legal),bool);z=(legal-mean)/scale
    for target in (targets-mean)/scale:
        distance=np.sum((z-target)**2,axis=1);distance[~available]=np.inf
        index=int(np.argmin(distance));available[index]=False;chosen.append(index)
    return np.asarray(chosen,dtype=int)


def feature_inputs(view):
    return np.column_stack((view.params,view.initial))
