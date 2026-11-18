"""OGAS continuous-training adapter; canonical ODE workers run asynchronously.

The generator has one bounded CPU thread. Solver workers are separate CPU
processes. This is an independently implemented, legal-domain adaptation of
OGAS Algorithm 1, not a claim to reproduce Melissa/PDE experiments unchanged.
"""
from dataclasses import dataclass,asdict,replace
import json
import math
import os

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

from ..baseline.ogas import ConditionalOGAS
from ..benchmark.runner import _json_safe
from budding_yeast_v2.data.dataset_archive import file_sha256,atomic_json
from budding_yeast_v2.utils.experiments.logging import log_event
from budding_yeast_v2.utils.experiments.resources import resource_slot
from .domain import feature_inputs,project_legal_targets
from .replay import TrajectoryReplay
from .background import BackgroundOGAS


ONLINE_METHODS=('ogas_async_projected','random_async')


@dataclass(frozen=True)
class OnlineConfig:
    epochs: int = 200
    warmup: int = 5
    generation_epochs: int = 120
    steps_per_epoch: int | None = None
    replay_capacity: int = 20000
    generator_every_steps: int = 10
    bootstrap_steps: int = 10
    mixture_warmup_steps: int = 100
    flush_rows: int = 128
    max_attempts_mult: int = 30
    signal: str = 'scaled_mse'
    signal_eps: float = 1e-8

    def validate(self):
        if (not 0<=self.warmup<self.epochs or not 1<=self.generation_epochs<self.epochs
                or min(self.replay_capacity,self.generator_every_steps,self.bootstrap_steps,
                       self.flush_rows,self.max_attempts_mult,self.mixture_warmup_steps)<1
                or self.steps_per_epoch is not None and self.steps_per_epoch<1
                or self.signal not in ('scaled_mse','relative_l2')
                or not np.isfinite(self.signal_eps) or self.signal_eps<=0):
            raise ValueError('Invalid online training/generation schedule')
        return self


def batch_difficulty(prediction,target,kind,eps):
    """Full-trajectory adaptation of OGAS Eq.4, or declared task loss signal."""
    pred=prediction.detach().double();truth=target.detach().double()
    if pred.shape!=truth.shape or pred.ndim!=3 or not torch.isfinite(pred).all():
        raise FloatingPointError('Invalid training-batch OGAS feedback')
    if kind=='scaled_mse':
        values=(pred-truth).square().mean((1,2))/(torch.linalg.vector_norm(truth.flatten(1),dim=1)+eps)
    elif kind=='relative_l2':
        values=(torch.linalg.vector_norm(pred-truth,dim=-1)
                /torch.hypot(torch.linalg.vector_norm(truth,dim=-1),truth.new_tensor(eps))).mean(1)
    else:raise ValueError('Unknown OGAS feedback signal')
    if not torch.isfinite(values).all():raise FloatingPointError('Nonfinite OGAS training signal')
    return values.cpu().numpy()


class OnlineGeneration(Callback):
    def __init__(self,domain,store,queue,directory,initial,*,target,method,config,ogas,seed,
                 device='cpu',bank_size=512):
        super().__init__()
        self.domain,self.store,self.queue=domain,store,queue
        self.directory=directory;self.target=target;self.initial=initial;self.method=method
        self.config=config.validate();self.seed=seed;self.bank_size=bank_size
        self.controller=(ConditionalOGAS(feature_inputs(initial),seed,ogas,'cpu')
                         if method=='ogas_async_projected' else None)
        self.background=None if self.controller is None else BackgroundOGAS(self.controller,max(1,queue.limit*8))
        self.rng=np.random.default_rng(seed);self.replay=None;self.pending=[]
        self.steps=0;self.last_cycle=0;self.generation_closed=False;self.captured=None;self.hook=None
        self.accepted_by_epoch=[];self.optimizer_steps_with_pending_queries=0
        self.best_score=float('inf');self.best_exposure=None
        self.sampler_not_ready_queries=0;self.exploration_queries=0

    def adapt_data(self,dm,normalization):
        if dm.num_workers!=0:raise ValueError('Streaming replay requires a single loading process')
        steps=self.config.steps_per_epoch or math.ceil(self.target/dm.batch_size_synth)
        self.replay=TrajectoryReplay(normalization,self.initial.params,self.initial.initial,
            self.store.initial['data'],capacity=self.config.replay_capacity,
            draws=steps*dm.batch_size_synth,seed=self.seed)
        dm.train_dataset_synth=self.replay;dm.drop_last_train=False

    @property
    def accepted(self):return self.store.count+len(self.pending)

    def _accept(self,results):
        valid=[r for r in results if r.status=='success']
        if valid:
            flags=(np.asarray([r.candidate.metadata['ogas_is_reference'] for r in valid])
                   if self.controller is not None else np.ones(len(valid),dtype=bool))
            self.replay.add(np.asarray([r.params for r in valid]),np.asarray([r.y_init for r in valid]),
                            np.asarray([r.trajectory for r in valid]),is_reference=flags)
            self.pending.extend(valid)
            if self.accepted>self.target:raise RuntimeError('Online label budget overshot')
            if len(self.pending)>=self.config.flush_rows:self.flush()

    def flush(self):
        if self.pending:
            with resource_slot('data','flush immutable online ODE block'):self.store.append(self.pending)
            self.pending=[]

    def _fill(self):
        count=min(self.queue.limit-len(self.queue.pending),self.target-self.accepted-len(self.queue.pending))
        if count<=0:return
        if self.queue.submitted+count>(self.target-len(self.initial.ids))*self.config.max_attempts_mult:
            raise RuntimeError('Online ODE attempt cap reached; no partial completion')
        with resource_slot('scoring','online canonical proposal and generation'):
            candidates,signatures,view=self.domain.bank(max(self.bank_size,count));x=feature_inputs(view)
            if self.controller is None:
                guided=0
                ids=self.rng.choice(len(x),count,replace=False);diagnostics=dict(rule='random canonical reference')
            else:
                targets=self.background.take(count)
                fraction=(1-self.controller.config.exploration)*min(1.,max(0.,self.steps-self.config.bootstrap_steps)
                                                                  /self.config.mixture_warmup_steps)
                explore=self.rng.random(count)>=fraction
                guided=min(int((~explore).sum()),len(targets))
                selected=project_legal_targets(targets[:guided],x,self.controller.mean,self.controller.scale) if guided else np.array([],int)
                available=np.setdiff1d(np.arange(len(x)),selected,assume_unique=True)
                ids=np.r_[selected,self.rng.choice(available,count-guided,replace=False)]
                unavailable=int((~explore).sum())-guided
                self.sampler_not_ready_queries+=unavailable;self.exploration_queries+=int(explore.sum())
                diagnostics=self.background.status()
                diagnostics.update(rule='OGAS Algorithm1, asynchronous conditional DDPM, legal projection',
                    guided_queries=guided,exploration_queries=int(explore.sum()),sampler_not_ready_queries=unavailable,
                    requested_generator_fraction=fraction)
            diagnostics.update(reference_queries=count-guided,generator_queries=guided)
            for i in ids:self.domain.consume(signatures[i])
        chosen=[replace(candidates[i],metadata=dict(candidates[i].metadata,ogas_is_reference=position>=guided))
                for position,i in enumerate(ids)]
        self.queue.submit(chosen,model_step=self.steps)
        with (self.directory/'_state/acquisition.jsonl').open('a') as log:
            log.write(json.dumps(_json_safe(dict(diagnostics,model_step=self.steps,queries=count)))+'\n')

    def _cycle(self):
        if self.controller is None:return
        with resource_slot('scoring','online OGAS generator update'):
            _,_,reference=self.domain.bank(self.bank_size)
            self.background.request(feature_inputs(reference))
        self.last_cycle=self.steps
        with (self.directory/'_state/generator.jsonl').open('a') as log:
            log.write(json.dumps(dict(event='nonblocking_update_request',model_step=self.steps,
                **self.background.status()))+'\n')

    def _drain(self):
        while self.queue.pending:self._accept(self.queue.poll(block=True))
        self.flush()

    def finish_generation(self):
        # Exact quota at a predeclared training boundary. A fixed post-generation
        # training period follows, so late arrivals are not simply evaluated
        # before having a chance to enter any training minibatch.
        if self.background is not None:
            self._cycle();self.background.synchronize()
        while self.accepted<self.target:
            if self.background is not None and self.background.status()['queued_targets']<self.queue.limit:
                self._cycle();self.background.synchronize()
            self._fill()
            self._accept(self.queue.poll(block=True))
        self._drain();self.generation_closed=True
        log_event('END','online generation quota',accepted=self.accepted,steps=self.steps,
                  attempts=self.queue.submitted,failures=self.queue.failed)

    def on_fit_start(self,trainer,pl_module):
        if self.replay is None:raise RuntimeError('Online data adapter was not prepared')
        if pl_module.hparams.model_name!='fno' or pl_module.hparams.loss_type!='rel_l2':
            raise ValueError('This validated online adapter currently requires registered FNO with rel_l2')
        def capture(module,args,output):
            if pl_module.training:
                if not torch.is_tensor(output):raise ValueError('Need a trajectory tensor for online feedback')
                self.captured=output.detach()
        self.hook=pl_module.model.register_forward_hook(capture)

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        self.steps+=1
        self.replay.record_exposure(batch['online_row'].detach().cpu().numpy())
        if self.queue.pending:self.optimizer_steps_with_pending_queries+=1
        if self.controller is not None:
            if self.captured is None:raise RuntimeError('Missing current training-batch prediction')
            epsilon=pl_module.criterion.eps if self.config.signal=='relative_l2' else self.config.signal_eps
            values=batch_difficulty(self.captured,batch['y'],self.config.signal,epsilon)
            x=np.column_stack((batch['online_params'].detach().cpu().numpy(),
                               batch['online_initial'].detach().cpu().numpy()))
            self.background.observe(x,values,is_reference=batch['online_is_reference'].detach().cpu().numpy())
        self.captured=None
        if self.generation_closed:return
        self._accept(self.queue.poll())
        if self.steps>=self.config.bootstrap_steps:
            if self.controller is not None and (self.last_cycle==0 or self.steps-self.last_cycle>=self.config.generator_every_steps):
                self._cycle()
            self._fill()

    def on_train_epoch_end(self,trainer,pl_module):
        if trainer.current_epoch+1>=self.config.generation_epochs and not self.generation_closed:
            self.finish_generation()
        self.accepted_by_epoch.append(dict(epoch=int(trainer.current_epoch+1),accepted=self.accepted,
            unique_training_inputs=len(self.replay.exposure),training_draws=sum(self.replay.exposure.values())))
        log_event('PROGRESS','online exposure',**self.accepted_by_epoch[-1],pending_queries=len(self.queue.pending))
        value=float(trainer.callback_metrics['val_loss_synth'])
        if not np.isfinite(value):raise FloatingPointError('Nonfinite online validation objective')
        if value<self.best_score:
            self.best_score=value
            self.best_exposure=dict(self.training_contract(),epoch=int(trainer.current_epoch+1),val_loss_synth=value)
            # Lightning weights-only intentionally omits callback state/hooks.
            # Keep exposure next to the best weights, rather than silently
            # expanding each best/last weight file into a sampler checkpoint.
            atomic_json(self.directory/'stream/best_exposure.json',self.best_exposure)

    def state_dict(self):
        # A durable checkpoint contains no live futures. At most queue.limit
        # trajectories are drained; it does not finish the full generation quota.
        self._drain()
        best_path=self.directory/'stream/model.ckpt'
        best=(torch.load(best_path,map_location='cpu',weights_only=False)
              if best_path.exists() and self.best_exposure is not None else None)
        return dict(chunks=self.store.chunks,domain=self.domain.state(),replay=self.replay.state(),
            background=None if self.background is None else self.background.state_dict(),
            rng=self.rng.bit_generator.state,steps=self.steps,last_cycle=self.last_cycle,
            generation_closed=self.generation_closed,accepted_by_epoch=self.accepted_by_epoch,
            best_score=self.best_score,best_exposure=self.best_exposure,
            best_checkpoint=best,
            queue_counts=(self.queue.submitted,self.queue.finished,self.queue.failed),
            sampler_not_ready_queries=self.sampler_not_ready_queries,exploration_queries=self.exploration_queries,
            optimizer_steps_with_pending_queries=self.optimizer_steps_with_pending_queries)

    def load_state_dict(self,state):
        for chunk in state['chunks']:
            if file_sha256(self.store.directory/chunk['path'])!=chunk['sha256']:raise ValueError('Online source block changed')
        self.store.chunks=list(state['chunks']);domain=state['domain']
        if domain['sources']!=self.domain.sources:raise ValueError('Canonical sources changed')
        self.domain.next_index=domain['next_index'];self.domain.seen=set(domain['seen']);self.domain.counts.update(domain['counts'])
        self.replay.restore(state['replay'],self.store.array('params'),self.store.array('y_init'),self.store.array('data'))
        if self.background is not None:self.background.load_state_dict(state['background'])
        self.rng.bit_generator.state=state['rng'];self.steps=state['steps'];self.last_cycle=state['last_cycle']
        self.generation_closed=state['generation_closed'];self.accepted_by_epoch=state['accepted_by_epoch']
        self.queue.submitted,self.queue.finished,self.queue.failed=state['queue_counts']
        self.optimizer_steps_with_pending_queries=state['optimizer_steps_with_pending_queries']
        self.best_score=state['best_score'];self.best_exposure=state['best_exposure']
        self.sampler_not_ready_queries=state['sampler_not_ready_queries'];self.exploration_queries=state['exploration_queries']
        # An interrupted run may have overwritten best after its last durable
        # optimizer/sampler checkpoint. Restore the matching best, not that
        # uncommitted future model, whose training data might also be rolled back.
        if state['best_checkpoint'] is not None:
            path=self.directory/'stream/model.ckpt';temporary=path.with_suffix('.restore.tmp')
            torch.save(state['best_checkpoint'],temporary);os.replace(temporary,path)
            atomic_json(self.directory/'stream/best_exposure.json',self.best_exposure)

    def training_contract(self):
        return dict(normalization='initial training frozen',
            training_exposure='dynamic',accepted=self.accepted,unique_training_inputs=len(self.replay.exposure),
            training_draws=sum(self.replay.exposure.values()),chunks=list(self.store.chunks),
            native_melissa_pde_reproduction=False,settings=asdict(self.config))

    def on_save_checkpoint(self,trainer,pl_module,checkpoint):
        self._drain()
        checkpoint['online_generation_contract']=self.training_contract()

    def on_train_end(self,trainer,pl_module):
        if self.accepted!=self.target or not self.generation_closed:
            raise RuntimeError('Online training ended without complete generation quota')
        path=self.directory/'_state/online_final.pt'
        temporary=path.with_suffix('.tmp')
        torch.save(self.state_dict(),temporary);os.replace(temporary,path)

    def close(self):
        if self.hook is not None:self.hook.remove();self.hook=None
        if self.queue.pending:self.queue.abort()
        if self.background is not None:self.background.close()
