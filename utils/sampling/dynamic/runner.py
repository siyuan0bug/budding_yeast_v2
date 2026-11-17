"""Continuous labelled-pool orchestration using the shared fit/eval services."""
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import numpy as np
import torch

from budding_yeast_v2.data.dataset_archive import atomic_json,file_sha256,fingerprint
from budding_yeast_v2.utils.evaluation.checkpoints import Predictor
from budding_yeast_v2.utils.experiments.logging import tee_experiment_log,log_stage
from budding_yeast_v2.utils.experiments.resources import resource_slot,exclusive_gpu_slot
from ..benchmark.protocol import BenchmarkConfig,child_seed
from ..benchmark.runner import experiment_lock,source_inventory,input_view
from ..benchmark.rounds import fit_round
from ..benchmark.reports import save_progress_reports
from ..generation.cohorts import FrozenCohorts
from ..query import LabelOracle
from ..contracts import QueriedTraining
from .callback import DynamicConfig,DynamicSelection,DYNAMIC_METHODS


def run_dynamic(protocol,pool,method,output,*,device='cpu',dynamic_config=None,lhs_config=None,
                diagnostic_only=False,plots=False,wandb_enabled=False):
    dynamic=DynamicConfig(**(dynamic_config or {})).validate()
    root=Path(protocol).resolve();root=root if root.is_dir() else root.parent
    definition=json.loads((root/'protocol.json').read_text());cfg=BenchmarkConfig(**definition['config']).validate()
    if method not in DYNAMIC_METHODS:raise ValueError('Unknown dynamic method')
    if cfg.model!='fno' or cfg.loss!='rel_l2' or cfg.validation_metric!='normalized_loss':
        raise ValueError('Current dynamic gradient adapter requires FNO / smooth uncapped Relative L2 / normalized validation best')
    if cfg.budgets[-1]<=cfg.budgets[0]:raise ValueError('Dynamic experiment requires a larger final budget')
    settings=dict(workflow='labelled_pool_continuous_regression_v1',protocol=definition['fingerprint'],pool=pool,method=method,
        seed=cfg.seed,training=asdict(cfg),dynamic=asdict(dynamic),normalization='initial_train_frozen',
        access='training_side_labels_for_scoring' if method in ('peaks_regression','eracs_regression') else 'selected_training_labels_only',
        training_exposure='dynamic',task_adaptation=True,
        sources=source_inventory(),diagnostic_only=diagnostic_only,
        lhs_config_sha256=None if lhs_config is None else file_sha256(lhs_config))
    directory=Path(output)/f"{definition['protocol_id']}_dynamic{fingerprint(settings)[:8]}"/pool/method/f'seed_{cfg.seed}'
    with experiment_lock(directory),resource_slot('experiment','dynamic labelled-pool experiment'),exclusive_gpu_slot(device),tee_experiment_log(directory/'train.log'):
        state_dir=directory/'_state';state_dir.mkdir(exist_ok=True)
        old=directory/'config.json'
        if old.exists() and json.loads(old.read_text())!=settings:raise ValueError('Dynamic source/configuration changed')
        atomic_json(old,settings);done=state_dir/'completion.json'
        if done.exists():
            for relative,digest in json.loads(done.read_text())['artifacts'].items():
                if file_sha256(directory/relative)!=digest:raise ValueError('Dynamic artifact changed')
            for relative,digest in definition['artifacts'].items():
                if file_sha256(root/relative)!=digest:raise ValueError('Frozen protocol changed')
            return directory
        with resource_slot('data','dynamic frozen input cohorts'):
            # Read-only pool selection never synthesizes LHS inputs. The exact
            # frozen source/split SHA is sufficient; no new LHS spec is needed.
            cohorts=FrozenCohorts(root,pool,cfg,diagnostic_only=diagnostic_only,lhs_config=lhs_config,verify_generation=False)
            with np.load(root/f'{pool}_inputs.npz',allow_pickle=False) as z:inputs={k:z[k] for k in z.files}
            with np.load(root/f'{pool}_split.npz') as z:allowed=z['candidates']
            view=input_view(inputs,allowed);initial=cohorts.initial_rows
            if cfg.budgets[-1]>len(allowed):raise ValueError('Dynamic target exceeds training-side universe')
            oracle=LabelOracle(cohorts.source,allowed,state_dir/'label_access.json',track='B')
            stream=directory/'stream';stream.mkdir(exist_ok=True)
            snapshot=stream/'initial_dataset.json'
            cohorts.source.snapshot(snapshot,initial,provenance=dict(role='initial_normalization_fit',source_trajectories_copied=False))
        # Oracle owns its data slot; retaining the cohort-loading slot here can
        # deadlock a full set of concurrently initializing dynamic jobs.
        training=QueriedTraining(cohorts.initial_inputs,oracle.reveal(initial)['data'])
        callback=DynamicSelection(view,oracle,initial,cfg.budgets[-1],method,dynamic,
            child_seed(cfg.seed,'dynamic_selection'),directory,device,cfg.batch_size)
        train_cfg=replace(cfg,budgets=(cfg.budgets[0],),epochs=(dynamic.epochs,),warmup=(dynamic.warmup,))
        context=dict(model=cfg.model,loss=cfg.loss,method=method,pool=pool,seed=cfg.seed,
                     training_rows=cfg.budgets[-1],warmup=dynamic.warmup)
        started=time.monotonic();wandb_run=None
        try:
            if wandb_enabled:
                import wandb
                wandb_run=wandb.init(project='budding_sampling_dynamic',mode='offline',dir=str(state_dir),config=settings)
            with log_stage('continuous dynamic coreset training',method=method):
                model,norm,history=fit_round(train_cfg,0,training,cohorts.validation,stream,
                    child_seed(cfg.seed,'model',0,0),device=device,dataset_path=snapshot,
                    identity=dict(workflow=settings['workflow'],configuration=fingerprint(settings)),
                    state_dir=state_dir/'primary',plot_context=context,monitoring=cohorts.monitoring,
                    wandb_run=wandb_run,extra_callbacks=(callback,),data_adapter=callback.adapt_data)
            if callback.steps==0:callback.load_state_dict(torch.load(state_dir/'dynamic_final.pt',map_location='cpu',weights_only=False))
            predictor=Predictor([model],norm,training.trajectories.shape[-1],device,cfg.prediction_batch_size)
            with log_stage('dynamic independent evaluation'):
                metrics=cohorts.evaluate(predictor,stream/'eval',0,len(callback.dataset.rows),plots=plots,context=context)
            save_progress_reports(directory,[dict(round=0,training_rows=len(callback.dataset.rows),metrics=metrics)],
                [(0,history)],{0:context},stream)
            if wandb_run is not None:wandb_run.log({f'evaluation/{k}/relative_l2':v['sample_channel_rel_l2'] for k,v in metrics.items()})
            report=dict(status='complete',resident_training_inputs=len(callback.dataset.rows),
                unique_training_inputs=len(callback.dataset.exposure),training_draws=sum(callback.dataset.exposure.values()),
                unique_labels_accessed=len(oracle.revealed),logical_label_reads=oracle.accesses,new_ode_calls=0,
                optimizer_steps=callback.steps,best_checkpoint_exposure=callback.best_exposure,
                training_exposure='dynamic',normalization='initial_train_frozen',events=callback.events,
                seconds=time.monotonic()-started,source_trajectories_copied=False,
                native_classification_reproduction=False)
            artifacts=[*stream.rglob('*'),*state_dir.glob('selection_epoch_*.json'),state_dir/'dynamic_final.pt',
                state_dir/'label_access.json',state_dir/'primary/last_weights.ckpt',state_dir/'primary/training.json',
                * (directory/'eval').glob('*.json'),* (directory/'eval').glob('*.csv'),directory/'performance.csv']
            report['artifacts']={str(p.relative_to(directory)):file_sha256(p) for p in artifacts if p.is_file()}
            atomic_json(done,report)
            return directory
        finally:
            if wandb_run is not None:wandb_run.finish()
