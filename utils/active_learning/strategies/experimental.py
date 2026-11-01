"""Independent AL research variants; fixed pool, common oracle and training.

No real benchmark identity/label or held-out synthetic trajectory is used.
The old ours_response implementation remains a separately registered control.
"""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import query_design as design
from . import query_feedback as feedback
from .response import background_graph, prediction_features, _pairs, SETTINGS as RESPONSE_SETTINGS


VARIANTS = {
    'ours_response_marginal': dict(mechanism='marginal_response', feedback=False),
    'ours_response_feedback': dict(mechanism='marginal_response', feedback=True),
    'ours_interaction': dict(mechanism='exact_mc_ko_interaction', feedback=False),
    'ours_invariance': dict(mechanism='proved_tem1_invariance', feedback=False),
    'ours_event': dict(mechanism='predicted_event_disagreement', feedback=False),
    'ours_error': dict(mechanism='prequery_point_error', feedback=True),
    'ours_coverage': dict(mechanism='feedback_hierarchical_coverage', feedback=True),
}

SETTINGS = dict(strength_bins=4, mass_drop_threshold=.1, debounce_steps=10,
                error_rank_multiplier_max=2.)


def event_disagreement(physical):
    """Bounded event descriptors from member predictions, not true labels.

    Observation-window division counts are neutral groups, not attractor or
    bifurcation claims. Use every output time for mass drops and 10-step debounce.
    """
    values=physical.detach().cpu().numpy();members,n,_,steps=values.shape
    from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
    descriptors=[]
    for member in values:
        canonical=pattern_group_metadata(member)
        rows=[]
        for index,trajectory in enumerate(member):
            hits=np.flatnonzero(-np.diff(trajectory[0])>SETTINGS['mass_drop_threshold'])+1
            divisions=hits[np.r_[True,np.diff(hits)>SETTINGS['debounce_steps']]] if len(hits) else []
            count=int(canonical['observed_division_counts'][index])
            group=int(canonical['pattern_group_ids'][index])
            clb=trajectory[2];esp=trajectory[32];ori=trajectory[33]
            one_hot=np.eye(6)[group]
            amplitude=np.log1p(np.maximum([clb.max(),clb[-1],ori.max(),trajectory[35].max(),esp[-1]],0))
            # Every descriptor lies in [0,1]; no raw amplitude dominance.
            rows.append(np.r_[one_hot,float(count>0),
                divisions[0]/max(1,steps-1) if len(divisions) else 1.,amplitude/(1+amplitude)])
        descriptors.append(rows)
    return np.var(np.asarray(descriptors),axis=0).mean(1).astype(np.float32)


def point_rank(scores,budget,seed,kind):
    if not np.isfinite(scores).all() or (scores<0).any():raise ValueError('Invalid point scores')
    count=budget-int(design.SETTINGS['random_fraction']*budget)
    rng=np.random.RandomState(seed)
    order=rng.permutation(len(scores))
    selected=list(map(int,order[np.argsort(-scores[order],kind='stable')[:count]]))
    records=[dict(kind=kind,endpoints=[i],new_queries=1,score=float(scores[i])) for i in selected]
    remaining=np.setdiff1d(np.arange(len(scores)),selected)
    for i in rng.choice(remaining,budget-len(selected),replace=False):
        selected.append(int(i));records.append(dict(kind='random',endpoints=[int(i)],new_queries=1))
    return np.asarray(selected,dtype=np.int64),records


def coverage_keys(params,metadata):
    """Coarse joint strength regions, recovered from the existing inverse CDF."""
    from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS,get_default_params
    from .response import _strength_curve
    lookup={k:i for i,k in enumerate(PARAM_KEYS)};defaults=get_default_params();keys=[]
    for p,meta in zip(params,metadata):
        coordinates=[]
        for gene,mode in _pairs(meta):
            branch=''
            if (gene,mode)==('CDC14','ts'):
                branch='synthesis' if p[lookup['ks14']]<.1*defaults['ks14'] else 'activity'
            names,values,cdf=_strength_curve(gene,mode,branch)
            if names:
                u=np.interp(p[lookup[names[0]]],values,cdf)
                bins=SETTINGS['strength_bins']
                coordinates.append((gene,mode,branch,min(bins-1,int(bins*u))))
        keys.append((_pairs(meta),float(p[lookup['mdt']]),tuple(coordinates)))
    return keys


def _experimental_sample(self,trainer,pl_module,dm,device,current_epoch):
    if not self._pool_size:return None
    config=VARIANTS[self.strategy];started=time.monotonic()
    params,initials,overrides,metadata=design.input_table(self)
    anchor_count=len(params)-int(self._pool_size)
    if anchor_count and anchor_count!=len(dm.train_dataset_synth):
        raise ValueError('Experimental training-anchor alignment error')
    mechanism=config['mechanism'];needs_graph=mechanism in (
        'marginal_response','exact_mc_ko_interaction','proved_tem1_invariance')
    events=np.zeros(len(params),dtype=np.float32)
    def observer(left,right,predictions):
        members,n,variables,steps=predictions.shape
        physical=dm.denormalize_y(predictions.flatten(0,1),device).reshape(members,n,variables,steps)
        events[left:right]=event_disagreement(physical)
    edges=np.empty((0,2),dtype=np.int64);distances=np.empty(0)
    if needs_graph:
        edges,distances=background_graph(params,initials,overrides,metadata,self._pool_size)
        print(f'[{self.strategy}] legal response edges={len(edges)}',flush=True)
    point,features=prediction_features(self,dm,device,anchor_count,
        observer=observer if mechanism=='predicted_event_disagreement' else None,
        keep_features=mechanism in ('marginal_response','exact_mc_ko_interaction'))
    budget=self._al4pde_scheduled_num_add()
    seed=int(self.random_seed+1009*self._al4pde_al_iter+self.pool.revision)
    keys=design.background_keys(params,metadata)
    feedback_report={};fixed_features=None;error_scores=None
    if config['feedback']:
        fixed_features=feedback.fixed_input_features(self,params,initials)
        target='response_error' if self.strategy=='ours_response_feedback' else 'point_error'
        error_scores,feedback_report=feedback.predict_error(self,fixed_features[:self._pool_size],target)
    if mechanism=='marginal_response':
        scores=None
        if error_scores is not None:
            # Rank-calibrated multiplier in [1,2], keeps uncertainty units and
            # prevents a learned error outlier from arbitrarily dominating.
            from scipy.stats import rankdata
            multiplier=np.ones(len(params),dtype=np.float32)
            multiplier[:self._pool_size]+=(SETTINGS['error_rank_multiplier_max']-1)*rankdata(error_scores,method='average')/max(1,len(error_scores))
            scores=design.action_variance(features,edges)*multiplier[edges].mean(1)
        selected,records=design.marginal_actions(point,features,edges,distances,keys,
            self._pool_size,budget,seed,action_scores=scores)
    elif mechanism=='exact_mc_ko_interaction':
        blocks=design.interaction_blocks(params,initials,overrides,metadata,edges)
        feedback_report['exact_legal_interaction_blocks']=len(blocks)
        selected,records=design.marginal_actions(point,features,blocks,np.zeros(len(blocks)),keys,
            self._pool_size,budget,seed,kind='interaction')
    elif mechanism=='proved_tem1_invariance':
        valid=design.invariance_edges(params,initials,overrides,edges)
        nodes=np.unique(valid);means,_=feedback.predict_nodes(self,dm,device,nodes)
        mapped=np.searchsorted(nodes,valid)
        scores=np.mean((means[mapped[:,1]]-means[mapped[:,0]])**2,axis=(1,2)) if len(valid) else np.empty(0)
        sketch=np.zeros((1,len(params),16),dtype=np.float32)
        if len(nodes):
            flat=means.reshape(len(nodes),-1)
            columns=np.linspace(0,flat.shape[1]-1,16).round().astype(int)
            sketch[0,nodes]=flat[:,columns]
        feedback_report['proved_invariance_edges']=len(valid)
        selected,records=design.marginal_actions(point,sketch,valid,np.zeros(len(valid)),keys,
            self._pool_size,budget,seed,action_scores=scores,kind='invariance')
    elif mechanism=='predicted_event_disagreement':
        selected,records=point_rank(events[:self._pool_size],budget,seed,'predicted_event')
    elif mechanism=='prequery_point_error':
        scores=point[:self._pool_size] if error_scores is None else error_scores
        selected,records=point_rank(scores,budget,seed,'predicted_error' if error_scores is not None else 'uncertainty_cold_start')
    elif mechanism=='feedback_hierarchical_coverage':
        region_keys=coverage_keys(params,metadata)
        scores=point[:self._pool_size] if error_scores is None else error_scores
        selected,records=design.hierarchical_rank(scores,region_keys[:self._pool_size],
            region_keys[self._pool_size:],budget,seed)
    else:raise ValueError(mechanism)
    if len(selected)!=budget or len(np.unique(selected))!=budget:
        raise ValueError('Experimental selector violated unique input budget')
    if config['feedback']:
        feedback.prepare_feedback(self,dm,device,selected,records,fixed_features)
    ids=self.pool.ensure_ids();trace=[]
    provenance=getattr(self,'_cum_synth_provenance',[])
    for row in records:
        record=dict(row)
        record['endpoints']=[dict(kind='pool',candidate_id=int(ids[i])) if i<self._pool_size else
            dict(kind='training_anchor',training_row=int(i-self._pool_size),
                 candidate_id=int(provenance[i-self._pool_size].get('candidate_id',-1))) for i in row['endpoints']]
        trace.append(record)
    if not hasattr(self,'_experimental_audits'):self._experimental_audits=defaultdict(list)
    self._experimental_audits[int(self._al4pde_al_iter)].append(dict(strategy=self.strategy,
        version='query_design_v1',settings=dict(variant=config,design=design.SETTINGS,
            feedback=feedback.SETTINGS,events=SETTINGS,response_geometry=RESPONSE_SETTINGS),
        feedback=feedback_report,choices=trace,
        seconds=time.monotonic()-started,budget=int(budget),pool_size=int(self._pool_size)))
    print(f'[{self.strategy}] selected={len(selected)}, feedback={feedback_report}, seconds={time.monotonic()-started:.1f}',flush=True)
    return self.make_selection_plan(selected,dm)


def write_audit(engine,directory,batch):
    index=int(engine._al4pde_al_iter)-1
    reports=engine._experimental_audits.pop(index,[])
    accepted=set(map(int,batch.get('candidate_ids',())))
    for report in reports:
        for action in report['choices']:
            for endpoint in action['endpoints']:
                endpoint['status']='known_training' if endpoint['kind']=='training_anchor' else (
                    'accepted' if endpoint['candidate_id'] in accepted else 'not_accepted')
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    temporary=path/'query_design.json.tmp'
    temporary.write_text(json.dumps(dict(strategy=engine.strategy,batches=reports),indent=2))
    temporary.replace(path/'query_design.json')
    feedback.write_feedback(engine,directory,index)
    if hasattr(engine,'_hybrid_edge_outputs'):
        from .hybrid_feedback import write_edges
        write_edges(engine,directory,index)
    if hasattr(engine,'_mechanism_artifacts'):
        from ..auxiliary.artifacts import write_artifacts
        write_artifacts(engine,directory,index)
