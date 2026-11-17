"""PEAKS streaming acquisition and ERACS reselection, independent of fit/eval."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

from budding_yeast_v2.data.dataset_archive import atomic_json
from budding_yeast_v2.utils.evaluation.checkpoints import Predictor
from budding_yeast_v2.utils.experiments.logging import log_event
from budding_yeast_v2.utils.experiments.resources import resource_slot
from ..baseline.regression import gradient_bank,subset
from ..baseline.peaks import PEAKSStream
from ..baseline.eracs import gradient_snr,weighted_omp
from .dataset import ResidentDataset


DYNAMIC_METHODS=('peaks_regression','eracs_regression','random_stream','random_dynamic')


@dataclass(frozen=True)
class DynamicConfig:
    epochs: int = 200
    warmup: int = 10
    bootstrap_epochs: int = 5
    acquisition_epochs: int = 120
    steps_per_epoch: int | None = None
    sketch_dim: int = 128
    delta: int = 8
    selection_percent: float = 10.
    refresh_steps: int = 500
    cache_capacity: int = 20000
    reference_rows: int = 64
    stream_batch: int = 64
    max_scans_per_addition: int = 10000
    check_every: int = 5
    snr_threshold: float = .1
    omp_group_size: int = 100

    def validate(self):
        values=(self.epochs,self.sketch_dim,self.delta,self.refresh_steps,self.cache_capacity,self.reference_rows,
                self.stream_batch,self.max_scans_per_addition,self.check_every,self.omp_group_size)
        if min(values)<1 or not 0<=self.warmup<self.epochs or not 0<self.acquisition_epochs<self.epochs:
            raise ValueError('Invalid dynamic training/selection schedule')
        if not 0<=self.bootstrap_epochs<self.acquisition_epochs:raise ValueError('Initial training must finish before acquisition closes')
        if not 0<self.selection_percent<=100 or not np.isfinite(self.snr_threshold) or self.snr_threshold<0:
            raise ValueError('Invalid dynamic thresholds')
        if self.steps_per_epoch is not None and self.steps_per_epoch<1:raise ValueError('Invalid dynamic step budget')
        return self


class DynamicSelection(Callback):
    def __init__(self,inputs,oracle,initial,target,method,config,seed,directory,device,batch_size):
        self.inputs=inputs;self.oracle=oracle;self.initial=list(map(int,initial));self.target=target
        self.method=method;self.config=config.validate();self.seed=seed;self.directory=Path(directory);self.device=device
        self.batch_size=batch_size;self.rng=np.random.default_rng(seed);self.steps=0;self.events=[];self.best_score=float('inf')
        self.best_exposure=None;self.dataset=None;self.pool_lookup={int(r):i for i,r in enumerate(inputs.ids)}
        self.stream=PEAKSStream(config.selection_percent,config.refresh_steps,config.cache_capacity)
        if config.delta>batch_size:raise ValueError('PEAKS delta cannot exceed actual training batch')

    def adapt_data(self,dm,normalization):
        self.norm=normalization;steps=self.config.steps_per_epoch or int(np.ceil(self.target/self.batch_size))
        initial=self.initial
        if self.method in ('eracs_regression','random_dynamic'):
            remaining=np.setdiff1d(self.inputs.ids,initial)
            initial=initial+self.rng.choice(remaining,self.target-len(initial),replace=False).tolist()
        self.dataset=ResidentDataset(normalization,self.inputs,
            lambda rows:self.oracle.reveal(rows),initial,capacity=self.target,draws=steps*self.batch_size,seed=self.seed)
        dm.train_dataset_synth=self.dataset;dm.num_workers=0

    def predictor(self,model):return Predictor([model],self.norm,self.dataset.truth.shape[-1],self.device,
        self.config.stream_batch,sketch_seed=self.seed)

    def gradients(self,model,rows):
        view=subset(self.inputs,[self.pool_lookup[int(r)] for r in rows])
        return gradient_bank(self.predictor(model),view,self.oracle.scoring_reader,
            dimension=self.config.sketch_dim,eps=float(model.criterion.eps))

    def acquire(self,model,count):
        if count<=0:return []
        with resource_slot('scoring','dynamic streaming selection'):
            if self.method=='peaks_regression' and (self.stream.anchor is None or self.steps%self.config.refresh_steps==0):
                refs=self.rng.choice(self.dataset.rows,min(self.config.reference_rows,len(self.dataset.rows)),replace=False)
                self.stream.set_reference(self.gradients(model,refs),self.steps)
            selected=[];scanned=0
            while len(selected)<count:
                remaining=np.setdiff1d(self.inputs.ids,np.r_[self.dataset.rows,selected])
                if not len(remaining):raise RuntimeError('PEAKS source exhausted before budget')
                rows=self.rng.choice(remaining,min(self.config.stream_batch,len(remaining)),replace=False)
                gradients=self.gradients(model,rows) if self.method=='peaks_regression' else None
                for i,row in enumerate(rows):
                    scanned+=1
                    accept=True if gradients is None else self.stream.decide(gradients[i])[0]
                    if accept:selected.append(int(row))
                    if len(selected)==count:break
                    if scanned>=count*self.config.max_scans_per_addition:
                        raise RuntimeError('PEAKS scan cap reached; no silent random fill')
            self.dataset.append(selected)
            self.events.append(dict(event='stream_accept',step=self.steps,rows=selected,scans=scanned))
            return selected

    def on_train_epoch_start(self,trainer,pl_module):
        epoch=int(trainer.current_epoch)
        if self.method in ('peaks_regression','random_stream'):
            if epoch==self.config.acquisition_epochs:
                # Explicit fixed-budget/fine-tune boundary; normal steps above
                # follow Algorithm1 new+old batches and persistent optimization.
                while len(self.dataset.rows)<self.target:
                    self.acquire(pl_module,min(self.config.delta,self.target-len(self.dataset.rows)))
            return
        if epoch==0 or epoch%self.config.check_every:return
        with resource_slot('scoring','ERACS SNR / OMP reselection'):
            current=self.gradients(pl_module,self.dataset.rows)
            snr=gradient_snr(current,self.dataset.weights[:len(current)])
            event=dict(event='snr_check',epoch=epoch,snr=snr,trigger=snr<self.config.snr_threshold)
            if self.method=='random_dynamic':
                rows=self.rng.choice(self.inputs.ids,self.target,replace=False);weights=None;event['trigger']=True
            elif event['trigger']:
                full=self.gradients(pl_module,self.inputs.ids)
                indices,weights,details=weighted_omp(full,self.target,group_size=self.config.omp_group_size,
                                                    seed=self.seed+epoch)
                rows=self.inputs.ids[indices];event.update(details)
            else:rows=None
            if rows is not None:
                self.dataset.set_rows(rows,weights);event['resident_rows']=list(map(int,rows))
                self.oracle.store.snapshot(self.directory/f'_state/selection_epoch_{epoch:04d}.json',rows,
                    weights=weights,provenance=dict(role='dynamic_resident',epoch=epoch))
            self.events.append(event);log_event('SELECT','dynamic coreset check',epoch=epoch,snr=snr,reselected=rows is not None)

    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        if (self.method not in ('peaks_regression','random_stream')
                or not self.config.bootstrap_epochs<=trainer.current_epoch<self.config.acquisition_epochs):return
        new=self.acquire(pl_module,min(self.config.delta,self.target-len(self.dataset.rows)))
        if new:
            # Replace actual batch after Lightning prefetch: exactly delta new
            # + batch_size-delta existing draws, not a delayed sampler hint.
            rows=new+batch['source_row'][len(new):].detach().cpu().tolist()
            replacement=self.dataset.batch(rows)
            for key,value in replacement.items():batch[key]=value.to(pl_module.device)

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        self.dataset.exposure.update(map(int,batch['source_row'].detach().cpu().tolist()));self.steps+=1

    def on_train_epoch_end(self,trainer,pl_module):
        value=trainer.callback_metrics.get('val_loss_synth')
        if value is not None and float(value)<self.best_score:
            self.best_score=float(value);self.best_exposure=dict(epoch=int(trainer.current_epoch)+1,
                unique_training_inputs=len(self.dataset.exposure),training_draws=sum(self.dataset.exposure.values()),
                resident_rows=list(self.dataset.rows),optimizer_steps=self.steps)
            atomic_json(self.directory/'stream/best_exposure.json',self.best_exposure)

    def state_dict(self):
        best=self.directory/'stream/model.ckpt'
        return dict(dataset=self.dataset.state(),steps=self.steps,events=self.events,rng=self.rng.bit_generator.state,
            sampler=self.stream.state_dict(),best_score=self.best_score,best_exposure=self.best_exposure,
            matching_best=None if not best.exists() else torch.load(best,map_location='cpu',weights_only=False))

    def load_state_dict(self,state):
        self.dataset.restore(state['dataset']);self.steps=state['steps'];self.events=state['events']
        self.rng.bit_generator.state=state['rng'];self.stream.load_state_dict(state['sampler'])
        self.best_score=state['best_score'];self.best_exposure=state['best_exposure']
        if state['matching_best'] is not None:
            path=self.directory/'stream/model.ckpt';temporary=path.with_suffix('.restore')
            torch.save(state['matching_best'],temporary);temporary.replace(path)
        if self.best_exposure is not None:atomic_json(self.directory/'stream/best_exposure.json',self.best_exposure)

    def on_train_end(self,trainer,pl_module):
        if len(self.dataset.rows)!=self.target:raise RuntimeError('Incomplete dynamic resident budget')
        self.oracle.store.snapshot(self.directory/'stream/dataset.json',self.dataset.rows,
            weights=self.dataset.weights[:len(self.dataset.rows)],provenance=dict(training_exposure='dynamic'))
        path=self.directory/'_state/dynamic_final.pt';temporary=path.with_suffix('.tmp')
        torch.save(self.state_dict(),temporary);temporary.replace(path)
