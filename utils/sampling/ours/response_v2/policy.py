"""Response-v2 and identical-path random control with pre-query task diagnostics."""
from collections import Counter
import numpy as np
import torch

from . import METHODS
from .paths import PathIndex
from .metrics import relative,normalized_prediction,normalization_state,restore_normalization
from .selection import select_actions
from ..adaptive_generation.policy import AdaptiveGeneration
from ...contracts import PoolInputs


class ResponseGeneration:
    version='task_response_paths_v2'
    def __init__(self,name,initial,truth,seed,*,system_id='budding'):
        if name not in METHODS or system_id!='budding':raise ValueError('Response-v2 currently supports Budding mutation only')
        from ..mechanistic.systems import BiologicalODE
        self.name=name;self.seed=int(seed);self.system=BiologicalODE(system_id)
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,202]))
        self.selection_rng=np.random.default_rng(np.random.SeedSequence([seed,303]))
        raw=np.column_stack((initial.params,initial.initial))
        self.input_scale=raw.std(0);self.input_scale[self.input_scale<1e-8]=1.
        self.last_refresh_target=None;self.round=0;self.pending={};self.journal=[]
        self.total_selected=0;self.exploration_selected=0;self.calls=0;self.predictor=None

    def refresh(self,inputs,truth,predictor,target=None):
        if self.pending:raise ValueError('Cannot replace model/statistics with unresolved queries')
        if len(inputs.metadata)!=len(inputs.ids) or len(truth)!=len(inputs.ids):raise ValueError('Missing aligned queried metadata')
        self.inputs=inputs;self.predictor=predictor;self.normalization=predictor.normalization
        self.index=PathIndex(self.system.param_names,self.input_scale)
        for start in range(0,len(inputs.ids),128):
            raw=np.asarray(truth[start:start+128])
            if not np.isfinite(raw).all():raise ValueError('Nonfinite revealed trajectories before normalization')
            block=self.normalization.targets(raw).cpu().numpy()
            if not np.isfinite(block).all():raise ValueError('Nonfinite revealed normalization')
            for j,t in enumerate(block,start):self.index.add(j,inputs.params[j],inputs.initial[j],inputs.metadata[j],t)
        if target is None or target!=self.last_refresh_target:self.round+=1
        self.last_refresh_target=target

    def anchor_probabilities(self):
        # Identical generator for the control: no model-error-driven anchor bias.
        return np.full(len(self.inputs.ids),1/len(self.inputs.ids))

    def select(self,bank,count):
        if self.predictor is None:raise ValueError('Refresh the pre-query model first')
        if not 0<count<=len(bank.ids):raise ValueError('Invalid response-v2 query budget')
        if len(bank.metadata)!=len(bank.ids):raise ValueError('Legal path metadata missing')
        pred=normalized_prediction(self.predictor,bank)
        scores=np.full(len(bank.ids),np.nan);supported=np.zeros(len(bank.ids),bool)
        for i,meta in enumerate(bank.metadata):
            target,anchors=self.index.estimate(bank.params[i],bank.initial[i],meta)
            if target is not None:
                scores[i]=relative(pred[i:i+1],target[None])[0];supported[i]=True
        due=max(0,int(np.floor(.2*(self.total_selected+count)))-self.exploration_selected)
        ids,actions=select_actions(bank,scores,supported,count,self.selection_rng,due,
                                  random_control=self.name=='path_random_generate')
        stats=normalization_state(self.normalization)
        for i,action in zip(ids,actions):
            key=AdaptiveGeneration._key(bank.params[i],bank.initial[i])
            if key in self.pending:raise ValueError('Duplicate outstanding response query')
            self.pending[key]=dict(prediction=pred[i].copy(),score=float(scores[i]) if supported[i] else None,
                action=action,metadata=dict(bank.metadata[i]),normalization=stats,round=self.round)
        self.total_selected+=len(ids);self.exploration_selected+=actions.count('explore');self.calls+=1
        measured=[r for r in self.journal if r['round']==self.round and r['action']=='response']
        correlation=None
        if len(measured)>2:
            from scipy.stats import spearmanr
            a=[r['score'] for r in measured];b=[r['normalized_relative_l2'] for r in measured]
            if np.ptp(a)>0 and np.ptp(b)>0:correlation=float(spearmanr(a,b).statistic)
        return ids,dict(version=self.version,method=self.name,candidate_truth_used=False,pattern_labels_used=False,
            action_counts=dict(Counter(actions)),supported_candidates=int(supported.sum()),
            selected_supported=int(supported[ids].sum()),query_count=len(ids),
            selected_scores=[float(scores[i]) if supported[i] else None for i in ids],
            feedback_count=len(self.journal),round_response_score_task_spearman=correlation,
            score_space='current train-fitted normalized full-time sample/channel Relative L2, eps=.01, uncapped',
            capacity_shortfall=actions.count('capacity_fallback'),
            prequery_task_metric_recorded=True)

    def observe(self,inputs,truth):
        if len(inputs.ids)!=len(truth) or not np.isfinite(truth).all():raise ValueError('Invalid revealed batch')
        rows=[]
        for p,y in zip(inputs.params,inputs.initial):
            key=AdaptiveGeneration._key(p,y)
            if key not in self.pending:raise PermissionError('Missing saved pre-query prediction')
            rows.append((key,self.pending[key]))
        observed=[]
        for i,(key,row) in enumerate(rows):
            norm=restore_normalization(row['normalization'])
            target=norm.targets(truth[i:i+1]).cpu().numpy();pred=row['prediction'][None]
            physical=norm.decode(torch.as_tensor(pred)).cpu().numpy()
            actual=float(relative(pred,target)[0]);physical_error=float(relative(physical,truth[i:i+1])[0])
            observed.append((key,row,target[0],actual,physical_error))
        metadata=[];base=len(self.inputs.ids)
        # Labels are usable immediately, before the next model-training round.
        for i,(key,row,target,actual,physical_error) in enumerate(observed):
            self.index.add(base+i,inputs.params[i],inputs.initial[i],row['metadata'],target)
            self.journal.append(dict(round=row['round'],action=row['action'],score=row['score'],
                path_id=row['metadata'].get('adaptive_path_id'),normalized_relative_l2=actual,
                physical_relative_l2=physical_error,normalization_scope='saved pre-query round statistics'))
            metadata.append(row['metadata']);del self.pending[key]
        next_id=int(np.max(self.inputs.ids))+1
        self.inputs=PoolInputs(np.r_[self.inputs.ids,np.arange(next_id,next_id+len(inputs.ids))],
            np.vstack((self.inputs.params,inputs.params)),np.vstack((self.inputs.initial,inputs.initial)),
            tuple(self.inputs.metadata)+tuple(metadata))

    def discard(self,inputs):
        for p,y in zip(inputs.params,inputs.initial):
            row=self.pending.pop(AdaptiveGeneration._key(p,y),None)
            if row is not None:
                self.total_selected-=1;self.exploration_selected-=int(row['action']=='explore')

    def state_dict(self):
        return dict(version=self.version,name=self.name,seed=self.seed,input_scale=self.input_scale,
            rng=self.rng.bit_generator.state,selection_rng=self.selection_rng.bit_generator.state,
            round=self.round,last_refresh_target=self.last_refresh_target,pending=self.pending,journal=self.journal,
            total_selected=self.total_selected,exploration_selected=self.exploration_selected,calls=self.calls)

    def load_state_dict(self,state):
        if state.get('version')!=self.version or state['name']!=self.name or state['seed']!=self.seed:
            raise ValueError('Response-v2 identity changed')
        for key in ('input_scale','round','last_refresh_target','pending','journal','total_selected','exploration_selected','calls'):
            setattr(self,key,state[key])
        self.rng.bit_generator.state=state['rng'];self.selection_rng.bit_generator.state=state['selection_rng']
