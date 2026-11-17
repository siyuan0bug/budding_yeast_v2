"""Generation decisions/feedback, isolated from simulation and model training."""
from dataclasses import asdict
import numpy as np
import torch

from ..baseline.oed import BayesianOED,OEDConfig,temporal_informativeness
from ..baseline.ogas import ConditionalOGAS,OGASConfig
from ..baseline.hggs import MultigridGenetic,HGGSConfig
from .domain import feature_inputs,project_legal_targets


METHODS=('random_generate','oed_temporal','ogas_round_projected','hggs_projected')


class GenerationPolicy:
    def __init__(self,name,initial,trajectories,seed=42,*,oed=OEDConfig(),ogas=OGASConfig(),hggs=HGGSConfig(),device='cpu'):
        if name not in METHODS:raise ValueError(f'Choose a declared adaptation: {METHODS}')
        self.name=name;self.seed=int(seed);self.rng=np.random.default_rng(seed)
        self.oed_config=oed.validate();self.ogas_config=ogas.validate();self.controller=None
        x=feature_inputs(initial)
        self.channel_scale=np.maximum(np.asarray(trajectories,dtype=np.float64).std((0,2)),1e-3)
        if name=='oed_temporal':
            self.controller=BayesianOED(x,seed,oed)
            self.controller.observe(x,temporal_informativeness(trajectories,channel_scale=self.channel_scale,config=oed))
        elif name=='ogas_round_projected':self.controller=ConditionalOGAS(x,seed,ogas,device)
        elif name=='hggs_projected':
            self.controller=MultigridGenetic(seed,hggs)
            self.mean=x.mean(0);self.scale=x.std(0);self.scale[self.scale<1e-8]=1.
        self.feedback_rows=0;self.refreshed_target=None

    def refresh(self,inputs,truth,predictor,target=None):
        if self.name!='hggs_projected':return
        if target is not None and self.refreshed_target==target:return
        residual=np.empty(len(inputs.ids))
        for start,prediction in predictor.batches(inputs):
            target=predictor.normalization.targets(truth[start:start+len(prediction)])
            residual[start:start+len(prediction)]=torch.linalg.vector_norm((prediction-target).flatten(1),dim=1).numpy()
        self.controller.refresh(feature_inputs(inputs),residual)
        self.refreshed_target=target

    def observe(self,inputs,truth,predictor=None):
        x=feature_inputs(inputs)
        if self.name=='oed_temporal':
            self.controller.observe(x,temporal_informativeness(truth,channel_scale=self.channel_scale,config=self.oed_config))
        elif self.name=='ogas_round_projected':
            if predictor is None:raise ValueError('OGAS requires the pre-query surrogate for difficulty feedback')
            observed=predictor.normalization.targets(truth).double()
            prediction=torch.cat([v for _,v in predictor.batches(inputs)]).double()
            # Same smooth, uncapped sample/channel loss as the main task,
            # reduced per input. Only revealed synthetic trajectories are used.
            difficulty=(torch.linalg.vector_norm(prediction-observed,dim=-1)
                        /torch.hypot(torch.linalg.vector_norm(observed,dim=-1),
                                     observed.new_tensor(.01))).mean(1).numpy()
            self.controller.observe(x,difficulty)
        self.feedback_rows+=len(x)

    def select(self,bank,count):
        if not 0<count<=len(bank.ids):raise ValueError('Invalid generation query size')
        x=feature_inputs(bank)
        if self.name=='random_generate':
            ids=self.rng.choice(len(x),count,replace=False);diagnostics=dict(rule='canonical proposal random reference')
        elif self.name=='oed_temporal':
            scores=self.controller.scores(x);ids=np.argsort(-scores,kind='stable')[:count]
            diagnostics=dict(rule='Xu2026 Eq5/8 posterior UCB',selected_scores=scores[ids].tolist(),
                             observations=self.controller.observations,temporal_metric=asdict(self.oed_config))
        elif self.name=='hggs_projected':
            targets,diagnostics=self.controller.propose(count)
            ids=project_legal_targets(targets,x,self.mean,self.scale)
            diagnostics.update(task_adaptations=['full trajectory residual replaces frequency error',
                'MGS desired coordinates projected onto fresh canonical legal proposals'],legal_projection=True)
        else:
            diagnostics=self.controller.train_cycle(x)
            targets,mixture=self.controller.sample_targets(count,x)
            ids=project_legal_targets(targets,x,self.controller.mean,self.controller.scale)
            diagnostics.update(mixture,rule='OGAS conditional DDPM + density ratio + legal projection',
                task_adaptations=['round-cold-start surrogate','fresh canonical legal proposal projection',
                                  'canonical proposal reference replaces continuous box uniform'],
                projection_dimensions=x.shape[1])
        return np.asarray(ids,dtype=int),diagnostics

    def state_dict(self):
        return dict(name=self.name,rng=self.rng.bit_generator.state,feedback_rows=self.feedback_rows,
                    channel_scale=self.channel_scale,
                    mean=getattr(self,'mean',None),scale=getattr(self,'scale',None),refreshed_target=self.refreshed_target,
                    controller=None if self.controller is None else self.controller.state_dict())

    def load_state_dict(self,state):
        if state['name']!=self.name:raise ValueError('Generation method changed')
        self.rng.bit_generator.state=state['rng'];self.feedback_rows=int(state['feedback_rows'])
        self.channel_scale=np.asarray(state['channel_scale'])
        self.refreshed_target=state.get('refreshed_target')
        if self.name=='hggs_projected':self.mean=np.asarray(state['mean']);self.scale=np.asarray(state['scale'])
        if self.controller is not None:self.controller.load_state_dict(state['controller'])
