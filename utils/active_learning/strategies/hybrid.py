"""Global predicted-trajectory coverage with local response queries.

Four independently runnable ablations share the fixed pool and unique-ODE budget.
Only synthetic training inputs, pool inputs and model predictions enter selection.
"""
from collections import defaultdict
import time
import numpy as np
import torch

from . import query_design as design
from . import query_feedback as feedback
from . import hybrid_feedback as calibration
from .hybrid_features import PredictionObserver,select_global,stratified_tail,edge_features
from .response import background_graph,prediction_features,_pairs


VARIANTS={
    'ours_global_interaction':dict(local='interaction',tail_fraction=0.,calibrated=False,feedback=False),
    'ours_global_response':dict(local='response',tail_fraction=0.,calibrated=False,feedback=True),
    'ours_global_tail':dict(local='response',tail_fraction=.2,calibrated=False,feedback=True),
    'ours_global_calibrated':dict(local='response',tail_fraction=0.,calibrated=True,feedback=True),
}
SETTINGS=dict(global_and_tail_fraction=.5,response_fraction=.4,random_fraction=.1,
              global_projection='32 times x all variables, shared CountSketch',sketch_width=128,
              global_predictor='ensemble_mean',fallback='MaxDist with all selected endpoints as anchors')


def _training_features(engine,dm,observer,device):
    """Compatibility for callers without cached training provenance; inputs only."""
    members=engine._ensemble_members;modes=[m.training for m in members]
    devices=[next(m.parameters(),torch.empty(0)).device for m in members]
    x,p=dm.train_dataset_synth.tensors[:2];features=[]
    try:
        for member in members:member.to(device).eval()
        with torch.no_grad():
            for start in range(0,len(p),128):
                predictions=torch.stack([m.forward_ic_time(x[start:start+128].to(device,dtype=torch.float32),
                    p[start:start+128].to(device,dtype=torch.float32)).float() for m in members]).mean(0)
                times=torch.linspace(0,predictions.shape[-1]-1,min(32,predictions.shape[-1]),device=device).round().long()
                features.append(observer.sketch(predictions[...,times].flatten(1)).cpu())
    finally:
        for m,mode,original in zip(members,modes,devices):m.to(original).train(mode)
    return torch.cat(features)


def _hybrid_sample(self,trainer,pl_module,dm,device,current_epoch):
    if not self._pool_size:return None
    started=time.monotonic();config=VARIANTS[self.strategy]
    params,initials,overrides,metadata=design.input_table(self)
    count=len(params);anchors=count-self._pool_size
    if anchors and anchors!=len(dm.train_dataset_synth):raise ValueError('Synthetic anchor alignment mismatch')
    seed=int(self.random_seed+1009*self._al4pde_al_iter+self.pool.revision)
    edges,distances=background_graph(params,initials,overrides,metadata,self._pool_size)
    print(f'[{self.strategy}] legal local edges={len(edges)}',flush=True)
    observer=PredictionObserver(count,self.random_seed,calibrated=config['calibrated'],width=SETTINGS['sketch_width'])
    point,features=prediction_features(self,dm,device,anchors,
        observer=lambda left,right,pred:observer(left,right,pred,dm,device),compact_two_members=True)
    train_features=(observer.global_features[self._pool_size:] if anchors
                    else _training_features(self,dm,observer,device))
    budget=self._al4pde_scheduled_num_add();report={};fixed=None;edge_x=log_variance=None
    if config['local']=='interaction':
        actions=design.interaction_blocks(params,initials,overrides,metadata,edges)
        action_distances=np.zeros(len(actions),dtype=np.float32)
        report['exact_legal_interaction_blocks']=len(actions)
    else:actions,action_distances=edges,distances
    scores=design.action_variance(features,actions)
    if config['feedback']:
        fixed=feedback.fixed_input_features(self,params,initials)
        if config['calibrated']:
            log_variance=design.action_variance(observer.log_contrasts,edges)
            context=[]
            for row in metadata:
                pairs=_pairs(row)
                context.append((len({g for g,m in pairs})/3.,
                    float(any(m in ('db','APC-A') for g,m in pairs)),
                    float(any(m in ('MC','GAL-MC','GAL-OE') for g,m in pairs))))
            edge_x=edge_features(point,observer,edges,distances,scores,log_variance,self._pool_size,
                                 fixed_inputs=fixed,context=np.asarray(context,dtype=np.float32))
            multiplier,fit=calibration.calibrated_multiplier(self,edge_x,seed)
            scores*=multiplier;report['edge_calibration']=fit
        else:
            error,fit=feedback.predict_error(self,fixed[:self._pool_size],'response_error')
            report['node_feedback']=fit
            if error is not None:
                from scipy.stats import rankdata
                multiplier=np.ones(count,dtype=np.float32)
                multiplier[:self._pool_size]+=rankdata(error,method='average')/max(1,len(error))
                scores*=multiplier[actions].mean(1)
    tail_budget=int(budget*config['tail_fraction'])
    tail,records=stratified_tail(observer,self._pool_size,tail_budget,seed)
    global_budget=int(budget*SETTINGS['global_and_tail_fraction'])-len(tail)
    global_rows=select_global(self,observer,train_features,global_budget,tail,device)
    records.extend(dict(kind='global',endpoints=[int(i)],new_queries=1) for i in global_rows)
    preselected=np.r_[tail,global_rows].astype(np.int64)
    def fallback(needed,chosen):return select_global(self,observer,train_features,needed,chosen,device)
    selected,local_records=design.marginal_actions(point,features,actions,action_distances,
        design.background_keys(params,metadata),self._pool_size,budget,seed,action_scores=scores,
        kind=config['local'],preselected=preselected,point_fraction=0.,random_fraction=.1,fallback=fallback)
    records.extend(local_records)
    if len(selected)!=budget or len(np.unique(selected))!=budget or sum(r['new_queries'] for r in records)!=budget:
        raise ValueError('Hybrid selector violated unique-query accounting')
    if config['feedback']:
        feedback.prepare_feedback(self,dm,device,selected,records,fixed)
        if config['calibrated']:calibration.prepare_edges(self,records,edges,edge_x,log_variance,count)
    ids=self.pool.ensure_ids();provenance=getattr(self,'_cum_synth_provenance',[]);trace=[]
    for row in records:
        record=dict(row)
        record['endpoints']=[dict(kind='pool',candidate_id=int(ids[i])) if i<self._pool_size else
            dict(kind='training_anchor',training_row=int(i-self._pool_size),
                 candidate_id=int(provenance[i-self._pool_size].get('candidate_id',-1))) for i in row['endpoints']]
        trace.append(record)
    if not hasattr(self,'_experimental_audits'):self._experimental_audits=defaultdict(list)
    self._experimental_audits[int(self._al4pde_al_iter)].append(dict(strategy=self.strategy,version='global_response_v2',
        settings=dict(variant=config,hybrid=SETTINGS,calibration=calibration.SETTINGS),feedback=report,
        choices=trace,seconds=time.monotonic()-started,budget=int(budget),pool_size=int(self._pool_size)))
    print(f'[{self.strategy}] selected={budget}, global={len(global_rows)}, tail={len(tail)}, feedback={report}, seconds={time.monotonic()-started:.1f}',flush=True)
    return self.make_selection_plan(selected,dm)
