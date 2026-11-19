"""HGGS growing cohorts in one optimizer session, with transactional boundaries."""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Dataset

from budding_yeast_v2.data.dataset_archive import atomic_json,file_sha256,fingerprint
from budding_yeast_v2.utils.experiments.logging import log_event
from budding_yeast_v2.utils.training.artifacts import write_history
from budding_yeast_v2.utils.training.runner import EpochSummaryCallback


def checkpoint(trainer,path,*,weights_only=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.next')
    trainer.save_checkpoint(temporary,weights_only=weights_only)
    os.replace(temporary,path)


class GrowingDataset(Dataset):
    """One shuffled pass per epoch, with raw arrays refreshed only at stages."""
    def __init__(self,normalization):self.norm=normalization;self.training=None
    def refresh(self,training):
        if not len(training.inputs.ids) or not np.isfinite(training.trajectories).all():
            raise ValueError('HGGS needs finite, nonempty queried training data')
        self.training=training
    def __len__(self):return len(self.training.inputs.ids)
    def __getitems__(self,indices):
        rows=np.asarray(indices,dtype=int);a=self.training
        x,p=self.norm.encode(a.inputs.params[rows],a.inputs.initial[rows],a.trajectories.shape[-1])
        y=self.norm.targets(a.trajectories[rows])
        return [dict(x=x[i],p=p[i],y=y[i]) for i in range(len(rows))]
    def __getitem__(self,index):return self.__getitems__([index])[0]


class HGGSStages(Callback):
    """Feedback always uses live current weights; best is only a saved report.

    The shared recorder runs before this callback. A full checkpoint is written
    before acquisition, with pending=True, and again after committing the next
    cohort. Recovery can finish a partly committed ODE acquisition without
    rolling the surrogate back to validation-best or duplicating queries.
    """
    def __init__(self,provider,budgets,epochs,directory,reference,reference_ids,
                 validation,validation_metric,*,keep_phase_weights,resume_path):
        self.provider=provider;self.budgets=tuple(map(int,budgets));self.epochs=tuple(map(int,epochs))
        if len(self.budgets)!=len(self.epochs) or min(self.epochs)<1:raise ValueError('HGGS stage schedule mismatch')
        self.ends=np.cumsum(self.epochs);self.directory=Path(directory)
        self.reference=reference;self.reference_ids=list(map(int,reference_ids))
        self.validation=validation;self.validation_metric=validation_metric
        self.keep_phase_weights=keep_phase_weights;self.stage=0;self.best=float('inf');self.best_epoch=None
        self.resume_path=Path(resume_path)
        self.pending=False;self.records=[];self.dataset=None;self.optimizer=None
        self.identity=fingerprint(dict(budgets=self.budgets,epochs=self.epochs,reference=self.reference_ids))

    def adapt_data(self,dm,normalization):
        for key in ('mean_y','std_y','p_mean','p_std','s_c'):
            value=getattr(self.reference,key)
            setattr(normalization,key,None if value is None else value.detach().cpu().clone())
        normalization.bind(dm);self.norm=normalization;self.dm=dm
        dm._dataset_contract=dict(dm._dataset_contract,normalization_reference=dict(
            scope='already_revealed_shared_initial_training',count=len(self.reference_ids),
            rows_sha256=fingerprint(self.reference_ids),version='hggs_continuous_v2'))
        dm.set_validation_data(self.validation,metric='normalized_loss' if self.validation_metric=='val_loss_synth' else self.validation_metric)
        self.dataset=GrowingDataset(normalization);self.dataset.refresh(self.provider.training(self.budgets[self.stage]))
        dm.train_dataset_synth=self.dataset;dm.num_workers=0;dm.drop_last_train=False

    def phase_dir(self,index=None):return self.directory/f'round_{self.stage if index is None else index:02d}'

    def state_dict(self):
        path=self.phase_dir()/'model.ckpt'
        return dict(identity=self.identity,stage=self.stage,best=self.best,best_epoch=self.best_epoch,
            pending=self.pending,records=self.records,provider=self.provider.state(),
            best_bytes=path.read_bytes() if self.keep_phase_weights and path.exists() and self.best_epoch is not None else None)

    def load_state_dict(self,state):
        if state['identity']!=self.identity:raise ValueError('HGGS stage identity changed')
        for name in ('stage','best','best_epoch','pending','records'):setattr(self,name,state[name])
        self.provider.restore(state['provider'])
        if state.get('best_bytes') is not None:
            path=self.phase_dir()/'model.ckpt';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(state['best_bytes'])
        if self.dataset is not None:self.dataset.refresh(self.provider.training(self.budgets[self.stage]))

    def on_train_start(self,trainer,pl_module):
        self.optimizer=trainer.optimizers[0]
        if self.pending:self.advance(trainer,pl_module)

    def advance(self,trainer,model):
        if trainer.optimizers[0] is not self.optimizer:raise RuntimeError('HGGS optimizer was reset between stages')
        target=self.budgets[self.stage+1]
        self.provider.acquire(target,model,self.norm)
        self.stage+=1;self.pending=False;self.best=float('inf');self.best_epoch=None
        self.dataset.refresh(self.provider.training(target))
        self.provider.snapshot(self.phase_dir()/'dataset.json',target,self.stage)
        checkpoint(trainer,self.resume_path)
        log_event('STAGE','HGGS continued training',stage=self.stage,training_rows=target,
                  optimizer_steps=int(trainer.global_step),normalization='shared_initial_training_frozen')

    def on_train_epoch_end(self,trainer,pl_module):
        epoch=int(trainer.current_epoch)+1
        value=float(trainer.callback_metrics[self.validation_metric])
        if not math.isfinite(value):raise FloatingPointError('Nonfinite HGGS validation loss')
        if value<self.best:
            self.best=value;self.best_epoch=epoch
            if self.keep_phase_weights:checkpoint(trainer,self.phase_dir()/'model.ckpt',weights_only=True)
        if epoch!=self.ends[self.stage]:return
        recorder=next(c for c in trainer.callbacks if isinstance(c,EpochSummaryCallback))
        start=0 if self.stage==0 else int(self.ends[self.stage-1])
        history=[dict(r,epoch=int(r['epoch'])-start,global_epoch=int(r['epoch'])) for r in recorder.history[start:epoch]]
        if len(history)!=self.epochs[self.stage]:raise RuntimeError('HGGS boundary history missing recorded epochs')
        rd=self.phase_dir();rd.mkdir(parents=True,exist_ok=True);write_history(rd/'history.csv',history)
        steps=int(trainer.global_step);previous=0 if not self.records else self.records[-1]['optimizer_steps_total']
        adam=[float(s['step']) for s in self.optimizer.state.values() if 'step' in s]
        record=dict(stage=self.stage,training_rows=self.budgets[self.stage],epochs=self.epochs[self.stage],
            global_epoch=epoch,best_epoch=self.best_epoch-start,best_global_epoch=self.best_epoch,
            best_validation_error=self.best,optimizer_steps_total=steps,optimizer_steps_delta=steps-previous,
            adam_step=max(adam,default=0),feedback_checkpoint='current_model',normalization='shared_initial_training_frozen')
        self.records.append(record);atomic_json(rd/'stage.json',record)
        self.pending=self.stage+1<len(self.budgets)
        atomic_json(self.directory/'stages.json',dict(records=self.records,provider=self.provider.state(),
            complete=not self.pending,identity=self.identity))
        # Save before any acquisition: a crash can resume from these exact current weights.
        checkpoint(trainer,self.resume_path)
        if self.pending:self.advance(trainer,pl_module)


class GFProvider:
    def __init__(self,source,inputs,coarse,truth,boundary,batch_size,device):
        self.source=source;self.inputs=inputs;self.coarse=np.asarray(coarse);self.truth=truth
        self.rows=list(map(int,boundary));self.lookup={int(row):i for i,row in enumerate(coarse)}
        self.batch_size=batch_size;self.device=device
    def state(self):return dict(rows=self.rows)
    def restore(self,state):
        rows=state['rows']
        if len(set(rows))!=len(rows) or not set(rows)<=self.lookup.keys():raise ValueError('HGGS GF recovery rows invalid')
        self.rows=list(map(int,rows))
    def training(self,count):
        from ..benchmark.runner import input_view
        from ..contracts import QueriedTraining
        if len(self.rows)!=count:raise ValueError('HGGS GF row quota mismatch')
        return QueriedTraining(input_view(self.inputs,self.rows),self.truth[[self.lookup[i] for i in self.rows]])
    def snapshot(self,path,count,index):
        if not path.exists():self.source.snapshot(path,self.rows,provenance=dict(role='HGGS GF current-model stage',stage=index))
    def acquire(self,target,model,norm):
        from budding_yeast_v2.utils.evaluation.checkpoints import Predictor
        from budding_yeast_v2.utils.training.monitoring import evaluation_state
        from ..benchmark.runner import input_view
        selected=set(self.rows);remaining=np.asarray([r for r in self.coarse if int(r) not in selected])
        residual=np.empty(len(remaining));predictor=Predictor([model],norm,self.truth.shape[-1],self.device,self.batch_size)
        with evaluation_state(model):
            for start,prediction in predictor.batches(input_view(self.inputs,remaining)):
                ids=remaining[start:start+len(prediction)]
                truth=norm.targets(self.truth[[self.lookup[int(i)] for i in ids]])
                residual[start:start+len(prediction)]=torch.linalg.vector_norm((prediction-truth).flatten(1),dim=1).numpy()
        if not np.isfinite(residual).all():raise FloatingPointError('Nonfinite HGGS GF ranking')
        selected=remaining[np.argsort(-residual,kind='stable')[:target-len(self.rows)]]
        self.rows.extend(map(int,selected))
