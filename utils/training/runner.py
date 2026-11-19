"""Shared Lightning execution, logging and checkpoint services for all models.

This module has no pool, acquisition method, round budget or biological generator.
Workflow adapters prepare a DataModule and provide a model factory.
"""
from pathlib import Path
import json
import random
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from budding_yeast_v2.utils.experiments.logging import log_event,format_epoch_line
from budding_yeast_v2.data.dataset_archive import atomic_json,fingerprint,file_sha256
from .convergence import loss_rates
from .artifacts import TrainingArtifacts, write_history, read_history

class EpochSummaryCallback(Callback):
    """每个 epoch 结束后用 print 输出训练摘要, 确保日志文件中可见 (不依赖 tqdm 终端渲染)."""

    def __init__(self, *, output=None, identity=None, generator=None, wandb_run=None,
                 round_index=0, member=0, monitor='val_loss_synth',test_monitor=None):
        super().__init__()
        self.epoch_start_time = None
        self.epoch_lr = None
        self.output = None if output is None else Path(output)
        self.identity = identity
        self.generator = generator
        self.wandb_run = wandb_run
        self.round_index, self.member = round_index, member
        self.monitor = monitor
        self.test_monitor = test_monitor
        self.history, self.previous_seconds, self.started, self.restored = [], 0., None, None

    def on_fit_start(self, trainer, pl_module):
        self.started = time.monotonic()

    def on_train_start(self, trainer, pl_module):
        if self.restored is not None:
            state = self.restored
            torch.set_rng_state(state['torch_rng'])
            np.random.set_state(state['numpy_rng'])
            random.setstate(state['python_rng'])
            self.generator.set_state(state['loader_rng'])
            if pl_module.device.type == 'cuda' and state.get('cuda_rng') is not None:
                torch.cuda.set_rng_state(state['cuda_rng'],pl_module.device)

    def state_dict(self):
        if self.output is None:
            return {}
        return dict(history=self.history, seconds=self.seconds,
                    torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                    loader_rng=self.generator.get_state(),
                    cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)

    def load_state_dict(self, state):
        if state:
            self.history, self.previous_seconds = state['history'], state['seconds']
            self.restored = state

    @property
    def seconds(self):
        return self.previous_seconds + (time.monotonic()-self.started if self.started is not None else 0.)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.output is not None:
            checkpoint['training_run'] = dict(identity=self.identity,history=self.history,seconds=self.seconds)

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        self.epoch_lr = float(trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_start_time is None:
            return
        if self.output is not None:
            from budding_yeast_v2.data.dataset_archive import atomic_json
            row = dict(epoch=trainer.current_epoch+1,
                train_loss=float(trainer.callback_metrics['train_loss']),
                val_error=float(trainer.callback_metrics[self.monitor]),validation_metric=self.monitor,
                lr=(self.epoch_lr if self.epoch_lr is not None else float(trainer.optimizers[0].param_groups[0]['lr'])),
                lr_next=float(trainer.optimizers[0].param_groups[0]['lr']),
                optimizer_steps=int(trainer.global_step))
            for key in ('val_macro_rel_l2','val_sample_channel_rel_l2'):
                if key in trainer.callback_metrics:
                    row[key]=float(trainer.callback_metrics[key])
            row['loss_type']=pl_module.hparams.loss_type
            if self.test_monitor is not None or 'val_sample_channel_rel_l2' in row:
                from budding_yeast_v2.utils.evaluation.metrics import RELATIVE_L2_METRIC_VERSION
                row['physical_rel_l2_version']=RELATIVE_L2_METRIC_VERSION
            if 'val_mse_norm' in trainer.callback_metrics:row['val_mse_norm']=float(trainer.callback_metrics['val_mse_norm'])
            if 'val_loss_synth' in trainer.callback_metrics:
                row['val_loss_synth']=float(trainer.callback_metrics['val_loss_synth'])
            for key in ('test_loss_synth','test_loss_real','test_real_denorm_rel_l2','test_synth_mse_norm','test_real_mse_norm'):
                if self.test_monitor is not None and key in self.test_monitor.latest:
                    row[key]=self.test_monitor.latest[key]
            self.history.append(row)
            write_history(self.output/'history.csv',self.history)
            print(format_epoch_line(row,round_index=self.round_index,member=self.member,
                epochs=trainer.max_epochs,elapsed_s=time.time()-self.epoch_start_time),flush=True)
            if self.wandb_run is not None:
                self.wandb_run.log({f'round_{self.round_index}/member_{self.member}/{k}':v for k,v in row.items()})
            return
        # Lightning finalizes epoch-reduced train_loss after validation. Reading
        # it in the validation hook prints NaN first, then the previous epoch.
        epoch = trainer.current_epoch
        m = trainer.callback_metrics
        train_loss = float(m.get('train_loss', float('nan')))
        val_synth = float(m.get('val_loss_synth', float('nan')))
        test_synth = float(m.get('test_loss_synth', float('nan')))
        test_real_mse = float(m.get('test_loss_real', float('nan')))
        test_real_denorm = float(m.get('test_real_denorm_rel_l2', m.get('test_loss_real_denorm', float('nan'))))
        lr = (self.epoch_lr if self.epoch_lr is not None else
              trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else 0.0)
        elapsed = time.time() - self.epoch_start_time
        gn_str = ""
        if 'train_w_clb2' in m:
            gn_str = f" | GN[clb2={float(m['train_w_clb2']):.2f}, sic1={float(m['train_w_sic1']):.2f}, max={float(m.get('train_w_max', 0.0)):.2f}]"
        print(f"[Epoch {epoch+1:03d}/{trainer.max_epochs:03d}] "
              f"train_loss={train_loss:.6f} | val_synth={val_synth:.6f} | "
              f"test_synth={test_synth:.6f} | test_real={test_real_mse:.6f} | "
              f"test_real_denorm={test_real_denorm:.6f}"
              f"{gn_str} | "
              f"lr={lr:.2e} | {elapsed:.1f}s", flush=True)


def make_trainer(**kwargs):
    """Common Lightning execution settings for static, online and complete-pool runs."""
    accelerator = kwargs.get('accelerator','cpu')
    defaults = dict(gradient_clip_val=1.,num_sanity_val_steps=0,
                    reload_dataloaders_every_n_epochs=0,
                    precision='bf16-mixed' if accelerator in ('gpu','cuda') else '32-true')
    defaults.update(kwargs)
    return pl.Trainer(**defaults)


class FinalEpochWeights(Callback):
    """Retain the final epoch once, before the recovery checkpoint is retired."""
    def __init__(self, path):
        self.path=Path(path)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch+1 == trainer.max_epochs:
            trainer.save_checkpoint(self.path,weights_only=True)



def fit_model_round(model_factory, datamodule, output, model_seed, *, epochs,
                    identity, device='cpu', monitor='val_loss_synth', round_index=0, member=0,
                    wandb_run=None, state_dir=None, plot_context=None, monitoring=None,
                    extra_callbacks=(),after_epoch_callbacks=(),reload_dataloaders_every_n_epochs=0):
    """Train one prepared cohort; data acquisition and split policy stay outside.

    The factory returns a registered LightningModule. Its optimizer/scheduler and
    training_step are the same ones used by train.py; no second gradient loop.
    Diagnostic callbacks may observe training or retain predefined snapshots;
    validation checkpoint selection remains controlled by ``monitor``.
    """
    from budding_yeast_v2.utils.evaluation.checkpoints import load_model_from_checkpoint
    from budding_yeast_v2.utils.experiments.curves import loss_figure
    identity=dict(identity,validation_metric=monitor,final_weights_policy='retain_last_epoch_weights_only_v1',
        epoch_monitoring={name:dict(rows_sha256=fingerprint(inputs.ids.tolist()),
        count=len(truth),shape=list(truth.shape[1:])) for name,(inputs,truth) in (monitoring or {}).items()})
    if reload_dataloaders_every_n_epochs:
        identity['dataloader_reload_epochs']=int(reload_dataloaders_every_n_epochs)
    artifacts=TrainingArtifacts.create(output,state_dir)
    output=artifacts.output
    pl.seed_everything(model_seed,workers=True)
    datamodule.loader_generator=torch.Generator().manual_seed(model_seed)
    model=model_factory()
    complete=json.loads(artifacts.metadata.read_text()) if artifacts.metadata.exists() else None
    if complete is not None:
        if complete.get('identity')!=identity or complete.get('epochs')!=epochs:
            raise ValueError('Completed training identity changed')
        if not artifacts.model.exists():raise RuntimeError('Missing completed model checkpoint')
        final=complete.get('last_epoch_checkpoint')
        if (final is None or not artifacts.last_weights.exists()
                or file_sha256(artifacts.last_weights)!=final['sha256']):
            raise RuntimeError('Missing or changed retained last-epoch checkpoint')
        history=read_history(artifacts.history)
        if len(history)!=epochs:raise ValueError('Completed history length changed')
        model,_=load_model_from_checkpoint(artifacts.model,device=device)
        return model,history
    from .monitoring import EpochTestMonitor
    test_monitor=EpochTestMonitor(monitoring,batch_size=datamodule.val_batch_size) if monitoring else None
    recorder=EpochSummaryCallback(output=output,identity=identity,generator=datamodule.loader_generator,
        wandb_run=wandb_run,round_index=round_index,member=member,monitor=monitor,test_monitor=test_monitor)
    last=artifacts.resume
    saved=torch.load(last,map_location='cpu',weights_only=False) if last.exists() else None
    metadata=None if saved is None else saved.get('training_run',saved.get('sampling_training'))
    if saved is not None and (metadata is None or metadata['identity']!=identity):
        raise ValueError('Round checkpoint identity changed')
    if saved is None or int(saved['epoch'])+1<epochs:
        best=ModelCheckpoint(dirpath=output,filename='model',monitor=monitor,mode='min',save_top_k=1,
                             save_weights_only=True,enable_version_counter=False)
        periodic=ModelCheckpoint(dirpath=artifacts.state,filename='resume',save_top_k=1,every_n_epochs=5,
                                 enable_version_counter=False)
        accelerator='gpu' if str(device).startswith('cuda') else 'cpu'
        index=int(str(device).split(':')[1]) if ':' in str(device) else 0
        trainer=make_trainer(max_epochs=epochs,logger=False,callbacks=([test_monitor] if test_monitor is not None else [])+list(extra_callbacks)+[recorder]+list(after_epoch_callbacks)+[FinalEpochWeights(artifacts.last_weights),best,periodic],
            accelerator=accelerator,devices=[index] if accelerator=='gpu' else 1,
            enable_progress_bar=False,enable_model_summary=False,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs)
        trainer.fit(model,datamodule=datamodule,ckpt_path=str(last) if saved is not None else None,
                    weights_only=False)
        trainer.save_checkpoint(last)
        history,seconds,steps=recorder.history,recorder.seconds,int(trainer.global_step)
        model.cpu()
        del trainer
    else:
        history,seconds,steps=metadata['history'],metadata['seconds'],int(saved['global_step'])
    if not artifacts.model.exists():raise RuntimeError('Missing best validation checkpoint')
    if not artifacts.last_weights.exists():raise RuntimeError('Missing last-epoch weights checkpoint')
    write_history(artifacts.history,history)
    convergence={f'{key}_last_{window}':loss_rates([r[key] for r in history[-window:]])
                 for window in (10,20) for key in ('train_loss','val_error')}
    loss_figure(history,output/'loss.png',title=f'Round {round_index:02d}, member {member}',
                context=dict(plot_context or {},model=model.hparams.model_name,
                             loss=model.hparams.loss_type,seed=identity.get('experiment_seed',model_seed),
                             training_rows=len(datamodule.train_dataset_synth),
                             warmup=model.hparams.get('warmup_epochs',0)))
    # Commit completion only after checkpoint, history and figure exist.
    atomic_json(artifacts.metadata,dict(status='complete',identity=identity,epochs=epochs,member=member,
        model=model.hparams.model_name,training_rows=len(datamodule.train_dataset_synth),
        optimizer_steps=steps,seed=identity.get('experiment_seed',model_seed),derived_model_seed=model_seed,
        seconds=seconds,convergence=convergence,
        validation_metric=monitor,best_validation_error=min(r['val_error'] for r in history),
        best_epoch=min(history,key=lambda r:r['val_error'])['epoch'],
        last_epoch_checkpoint=dict(path=str(artifacts.last_weights.resolve()),epoch=epochs,
            weights_only=True,sha256=file_sha256(artifacts.last_weights))))
    artifacts.resume.unlink(missing_ok=True)
    model,_=load_model_from_checkpoint(artifacts.model,device=device)
    return model,history
