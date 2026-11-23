"""Model-response interpolation and event-gap acquisition on fresh legal paths.

Local queried-data interpolation is a heuristic, not a certified ODE error
bound. Prequential progress is deliberately distinct from held-out gain.
"""
from collections import defaultdict
import numpy as np

from . import LEGACY_METHODS as METHODS
from .features import trajectory_features,event_features,context_key
from .feedback import Feedback
from ..mechanistic.systems import BiologicalODE
from ...contracts import PoolInputs


def prediction(predictor,inputs):
    blocks=[];expected=0
    for start,value in predictor.batches(inputs,physical=True):
        value=value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
        if start!=expected or not np.isfinite(value).all():raise ValueError('Unaligned/nonfinite model predictions')
        blocks.append(value);expected+=len(value)
    if expected!=len(inputs.ids):raise ValueError('Missing prediction rows')
    return np.concatenate(blocks)


class AdaptiveGeneration:
    def __init__(self,name,initial,truth,seed,*,system_id='budding'):
        if name not in METHODS:raise ValueError('Unknown adaptive generation policy')
        self.name=name;self.seed=int(seed);self.rng=np.random.default_rng(seed);self.calls=0
        self.system=BiologicalODE(system_id);self.system_id=system_id;self.feedback=Feedback()
        truth=np.asarray(truth,dtype=np.float64)
        self.scale=np.maximum(np.sqrt(np.mean(truth**2,axis=(0,2))),1e-5)
        raw=np.column_stack((initial.params,initial.initial))
        self.input_scale=raw.std(0);self.input_scale[self.input_scale<1e-8]=1.
        self.predictor=None;self.pending={};self.last_refresh_target=None

    def encode(self,y,p):
        return event_features(y,p,self.system) if self.name=='ours_event_generate' else trajectory_features(y,self.scale)

    def refresh(self,inputs,truth,predictor,target=None):
        self.inputs=inputs;self.truth=np.asarray(truth);self.predictor=predictor
        if len(inputs.metadata)!=len(inputs.ids):raise ValueError('Adaptive mutation generation needs aligned metadata')
        self.encoded=self.encode(truth,inputs.params)
        self.x=np.column_stack((inputs.params,inputs.initial))/self.input_scale
        self.groups=defaultdict(list)
        for j,m in enumerate(inputs.metadata):self.groups[context_key(m)].append(j)
        if target is None or target!=self.last_refresh_target:self.feedback.round+=1
        self.last_refresh_target=target

    def anchor_probabilities(self):
        # Difficulty estimates come only from previous untrained queries.
        keys=[context_key(meta) for meta in self.inputs.metadata]
        regional={key:max(self.feedback.recent_error(key),.05)*(
            self.feedback.weight(key) if self.name=='ours_progress_generate' else 1.) for key in set(keys)}
        weights=[regional[key] for key in keys]
        weights=np.asarray(weights);weights/=weights.sum()
        return .8*weights+.2/len(weights)

    def select(self,bank,count):
        if self.predictor is None:raise ValueError('Refresh trained predictor before querying')
        if not 0<count<=len(bank.ids):raise ValueError('Invalid adaptive budget')
        pred=prediction(self.predictor,bank);encoded=self.encode(pred,bank.params)
        x=np.column_stack((bank.params,bank.initial))/self.input_scale
        score=np.empty(len(bank.ids));available=0
        for i,meta in enumerate(bank.metadata):
            rows=self.groups.get(context_key(meta),[])
            if not rows:score[i]=1.;continue
            # At most 8 queried neighbours: no dense pool-by-pool matrix.
            if 'adaptive_axis' in meta:
                from budding_yeast_v2.data.systems.budding.catalog import touch_set
                tokens=meta.get('pairs',meta.get('genes',()))
                touched=set().union(*(touch_set(g,m) for g,m in tokens if g==meta['adaptive_axis']))
                positions=[j for j,k in enumerate(self.system.param_names) if k not in touched]
                flags=lambda m:tuple(bool(m.get('_effective_overrides',m.get('original_overrides',{})).get(k,False))
                                      for k in ('is_sbf_ca','is_mutant_104'))
                rows=[j for j in rows if np.array_equal(self.inputs.params[j,positions],bank.params[i,positions])
                      and flags(self.inputs.metadata[j])==flags(meta)]
                if not rows:raise ValueError('Legal path has no matching queried anchor')
            rows=np.asarray(rows);dist=((self.x[rows]-x[i])**2).sum(1)
            ids=np.argsort(dist,kind='stable')[:8];idx=rows[ids];d=dist[ids]
            weights=1./np.maximum(d,1e-6);weights/=weights.sum()
            # Conditional local response prediction from REVEALED truth only.
            target=weights@self.encoded[idx]
            if self.name!='ours_event_generate' and len(idx)>=2:
                # Local affine response from known labels; regularized dual solve.
                dx=self.x[idx]-self.x[idx[0]];dy=self.encoded[idx]-self.encoded[idx[0]]
                gram=dx@dx.T;ridge=max(float(np.trace(gram))*1e-4,1e-8)
                alpha=np.linalg.solve(gram+ridge*np.eye(len(idx)),dx@(x[i]-self.x[idx[0]]))
                target=self.encoded[idx[0]]+alpha@dy
            score[i]=np.sqrt(np.mean((encoded[i]-target)**2));available+=int(len(idx)>=2)
            if self.name=='ours_progress_generate':score[i]*=self.feedback.weight(context_key(meta))
        if not np.isfinite(score).all():raise FloatingPointError('Nonfinite adaptive score')
        chosen=[];remaining=np.ones(len(score),bool)
        # Response/input redundancy reduces marginal gain inside this batch.
        priority=score.copy();exploit=count-int(np.ceil(.2*count))
        for _ in range(exploit):
            tie=self.rng.permutation(np.flatnonzero(remaining));j=int(tie[np.argmax(priority[tie])])
            chosen.append(j);remaining[j]=False
            distance=np.sqrt(np.mean((x-x[j])**2,axis=1))
            priority*=1.-.8*np.exp(-distance**2/.01)
        chosen.extend(self.rng.choice(np.flatnonzero(remaining),count-len(chosen),replace=False).tolist())
        for j in chosen:
            key=self._key(bank.params[j],bank.initial[j])
            self.pending[key]=dict(prediction=pred[j].copy(),score=float(score[j]),context=context_key(bank.metadata[j]))
        self.calls+=1
        return np.asarray(chosen),dict(version='adaptive_response_prototype_v1',method=self.name,
            score='queried local interpolation gap, physical asinh RMS features' if self.name!='ours_event_generate'
                  else 'queried guard crossing count/time/order interpolation gap',
            known_interpolation_limit='may reward legitimate response changes; not certified error',
            candidate_truth_used=False,pattern_labels_used=False,local_multiple_anchor_rows=available,
            random_queries=count-exploit,selected_scores=score[chosen].tolist(),**self.feedback.summary())

    @staticmethod
    def _key(p,y):
        import hashlib
        return hashlib.sha256(np.r_[p,y].astype('<f8').tobytes()).hexdigest()

    def observe(self,inputs,truth):
        if len(inputs.ids)!=len(truth):raise ValueError('Unaligned revealed labels')
        pending=[]
        for p,y in zip(inputs.params,inputs.initial):
            key=self._key(p,y)
            if key not in self.pending:raise PermissionError('Feedback lacks a saved pre-query prediction')
            pending.append((key,self.pending[key]))
        # All rows validated before mutation; recomputing with post-query model is forbidden.
        predicted=np.stack([r['prediction'] for _,r in pending])
        error=np.sqrt(np.mean((self.encode(predicted,inputs.params)-self.encode(truth,inputs.params))**2,axis=1))
        for (key,r),e in zip(pending,error):
            self.feedback.observe(r['context'],r['score'],e);del self.pending[key]

    def discard(self,inputs):
        for p,y in zip(inputs.params,inputs.initial):self.pending.pop(self._key(p,y),None)

    def state_dict(self):
        return dict(version='adaptive_response_prototype_v1',name=self.name,seed=self.seed,system=self.system_id,
            scale=self.scale,input_scale=self.input_scale,rng=self.rng.bit_generator.state,calls=self.calls,
            feedback=self.feedback.state_dict(),pending=self.pending,last_refresh_target=self.last_refresh_target)

    def load_state_dict(self,state):
        if (state['version']!='adaptive_response_prototype_v1' or state['name']!=self.name
                or state['seed']!=self.seed or state['system']!=self.system_id):raise ValueError('Adaptive identity changed')
        self.scale=np.asarray(state['scale']);self.input_scale=np.asarray(state['input_scale'])
        self.rng.bit_generator.state=state['rng'];self.calls=int(state['calls']);self.pending=state['pending']
        self.feedback.load_state_dict(state['feedback']);self.last_refresh_target=state['last_refresh_target']
