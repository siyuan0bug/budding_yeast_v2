"""Continuous online OGAS orchestration using the shared training/evaluation code.

This workflow deliberately has its own identity: persistent surrogate/optimizer,
initial normalization frozen, dynamic replay, concurrent new ODE queries. It is
not silently inserted into the five cold-start, fixed-pool ranking.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import torch

from budding_yeast_v2.data.dataset_archive import TrajectoryStore,atomic_json,file_sha256,fingerprint
from ..baseline.ogas import OGASConfig
from ..benchmark.protocol import BenchmarkConfig,child_seed,effective_signatures
from ..benchmark.rounds import fit_round
from ..benchmark.runner import experiment_lock,source_inventory
from ..benchmark.reports import save_progress_reports
from ..contracts import PoolInputs,QueriedTraining
from budding_yeast_v2.utils.evaluation.checkpoints import Predictor
from budding_yeast_v2.utils.experiments.logging import log_event,log_stage,tee_experiment_log
from budding_yeast_v2.utils.experiments.resources import exclusive_gpu_slot,resource_slot
from .async_queries import AsyncQueries,query_costs
from .cohorts import FrozenCohorts
from .domain import FrozenBuddingDomain
from .online import ONLINE_METHODS,OnlineConfig,OnlineGeneration
from .runner import worker_lifecycle
from .storage import GeneratedTrainingStore


def _online_artifacts(directory):
    patterns=('data/chunks/*.npz','data/training_source_*.json','data/training_source_*.npz',
        'stream/*','stream/eval/*.json','stream/eval/*.csv','_state/online_final.pt',
        '_state/primary/last_weights.ckpt','_state/primary/training.json','eval/*.json','eval/*.csv',
        '_state/ode_attempts.jsonl','_state/generator.jsonl','_state/acquisition.jsonl','performance.csv')
    return {str(p.relative_to(directory)):file_sha256(p) for pattern in patterns
            for p in sorted(directory.glob(pattern)) if p.is_file()}


def run_online_generation(protocol,pool,method,output,*,device='cpu',workers=4,bank_size=512,
                          lhs_config=None,ogas_config=None,online_config=None,
                          plots=False,wandb_enabled=False,diagnostic_only=False):
    online=OnlineConfig(**(online_config or {})).validate()
    # Latest v2 Table1 numerical recipe; width128 is an explicit CPU/resource
    # adaptation from the paper's width1024. The older round entry keeps v1.
    ogas_defaults=dict(paper_version='arxiv_v2',diffusion_steps=100,sampling_steps=100,
                       learning_rate=2e-4,exploration=.3)
    ogas_defaults.update(ogas_config or {});ogas=OGASConfig(**ogas_defaults).validate()
    if method not in ONLINE_METHODS or min(workers,bank_size)<1:raise ValueError('Invalid online generation arguments')
    root=Path(protocol).resolve();root=root if root.is_dir() else root.parent
    definition=json.loads((root/'protocol.json').read_text());cfg=BenchmarkConfig(**definition['config']).validate()
    if cfg.model!='fno' or cfg.loss!='rel_l2' or cfg.validation_metric!='normalized_loss':
        raise ValueError('Online adapter requires FNO/rel_l2 with normalized validation best')
    if pool not in definition['pools']:raise ValueError('Unknown source pool')
    if pool=='lhs' and lhs_config is None:raise ValueError('LHS generation requires its frozen specification')
    if bank_size<workers:raise ValueError('Bank must cover all solver workers')
    if cfg.budgets[-1]<=cfg.budgets[0]:raise ValueError('Online generation requires a nonempty new-label budget')
    settings=dict(workflow='ogas_async_legal_domain_v1',protocol=definition['fingerprint'],pool=pool,
        method=method,seed=cfg.seed,training=asdict(cfg),online=asdict(online),ogas=asdict(ogas),
        bank_size=bank_size,workers=workers,max_inflight=workers,
        lhs_config_sha256=None if lhs_config is None else file_sha256(lhs_config),
        diagnostic_only=diagnostic_only,sources=source_inventory(),normalization='initial_train_frozen',
        checkpoint_selection='same_source_normalized_validation_best',training_exposure='dynamic',
        native_melissa_pde_reproduction=False,generator_execution='bounded asynchronous CPU thread')
    run_id=f"{definition['protocol_id']}_online{fingerprint(settings)[:8]}"
    from ..layout import experiment_directory
    directory=experiment_directory(output,cfg,pool,method,'generate')
    with experiment_lock(directory),resource_slot('experiment','online generation experiment'), \
            exclusive_gpu_slot(device),tee_experiment_log(directory/'train.log'):
        (directory/'_state').mkdir(exist_ok=True)
        config_path=directory/'config.json'
        if config_path.exists() and json.loads(config_path.read_text())!=settings:raise ValueError('Online configuration changed')
        atomic_json(config_path,settings)
        done=directory/'_state/completion.json'
        if done.exists():
            result=json.loads(done.read_text())
            if result['accepted']!=cfg.budgets[-1]:raise ValueError('Wrong completed online budget')
            with resource_slot('data','verify completed online generation'):
                for relative,digest in result['artifacts'].items():
                    path=directory/relative
                    if not path.exists() or file_sha256(path)!=digest:raise ValueError(f'Online artifact changed: {relative}')
                for relative,digest in definition['artifacts'].items():
                    if file_sha256(root/relative)!=digest:raise ValueError('Frozen protocol artifact changed')
            return directory
        with resource_slot('data','online initial/frozen evaluation cohorts'):
            cohorts=FrozenCohorts(root,pool,cfg,diagnostic_only=diagnostic_only,lhs_config=lhs_config)
            store=GeneratedTrainingStore(cohorts.source.path,cohorts.initial_rows,directory/'data',pool,cfg.seed,
                generation_metadata=dict(method=method,workflow=settings['workflow'],configuration=fingerprint(settings)))
            initial=cohorts.initial_inputs
            stream=directory/'stream';stream.mkdir(exist_ok=True)
            snapshot=stream/'initial_dataset.json'
            if not snapshot.exists():
                source=TrajectoryStore(store.publish(0));source.snapshot(snapshot,np.arange(len(initial.ids)),
                    provenance=dict(role='normalization_initial_training_only',source_trajectories_copied=False))
        domain=FrozenBuddingDomain(pool,child_seed(cfg.seed,'generated_proposals'),directory/'_state/proposals',
            lhs_config=lhs_config,forbidden=cohorts.forbidden,
            seen=effective_signatures(initial.params,initial.initial))
        training=QueriedTraining(PoolInputs(np.arange(len(initial.ids)),initial.params,initial.initial),store.initial['data'])
        train_cfg=replace(cfg,budgets=(cfg.budgets[0],),epochs=(online.epochs,),warmup=(online.warmup,))
        context=dict(model=cfg.model,loss=cfg.loss,pool=pool,method=method,training_rows=cfg.budgets[-1],
                     seed=cfg.seed,warmup=online.warmup)
        wandb_run=None;callback=None;started=time.monotonic()
        try:
            if wandb_enabled:
                import wandb
                os.environ['WANDB_MODE']='offline'
                wandb_run=wandb.init(project='budding_sampling_generation',group=run_id,
                    name=f'{pool}/{method}/seed{cfg.seed}',dir=str(directory/'_state'),mode='offline',config=settings)
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
                                     initializer=worker_lifecycle,initargs=(os.getpid(),)) as executor:
                queue=AsyncQueries(executor,directory/'_state/ode_attempts.jsonl',workers)
                callback=OnlineGeneration(domain,store,queue,directory,initial,target=cfg.budgets[-1],method=method,
                    config=online,ogas=ogas,seed=child_seed(cfg.seed,'online_sampler'),device=device,bank_size=bank_size)
                try:
                    with log_stage('continuous online surrogate training',method=method,epochs=online.epochs):
                        model,norm,history=fit_round(train_cfg,0,training,cohorts.validation,stream,
                            child_seed(cfg.seed,'model',0,0),device=device,dataset_path=snapshot,
                            identity=dict(workflow=settings['workflow'],configuration=fingerprint(settings)),
                            state_dir=directory/'_state/primary',plot_context=context,
                            monitoring=cohorts.monitoring,wandb_run=wandb_run,
                            extra_callbacks=(callback,),data_adapter=callback.adapt_data)
                    # Completed training can be reused after interruption before
                    # evaluation; sampler/replay recovery is separate from weights.
                    final=directory/'_state/online_final.pt'
                    if store.count<cfg.budgets[-1]:
                        callback.load_state_dict(torch.load(final,map_location=device,weights_only=False))
                    if store.count!=cfg.budgets[-1]:raise RuntimeError('Incomplete online label quota')
                except BaseException:
                    callback.close();raise
            domain.verify()
            with resource_slot('data','final online source/selection manifest'):
                source=TrajectoryStore(store.publish(1))
                source.snapshot(stream/'dataset.json',np.arange(store.count),
                    provenance=dict(training_exposure='dynamic',source_trajectories_copied=False))
            predictor=Predictor([model],norm,store.initial['data'].shape[-1],device,cfg.prediction_batch_size)
            with log_stage('online independent evaluation',samples=store.count):
                metrics=cohorts.evaluate(predictor,stream/'eval',0,store.count,plots=plots,
                    context={k:v for k,v in context.items() if k not in ('warmup','loss')})
            save_progress_reports(directory,[dict(round=0,training_rows=store.count,metrics=metrics)],
                [(0,history)],{0:context},stream)
            if wandb_run is not None:
                wandb_run.log({f'evaluation/{k}/sample_channel_rel_l2_physical':v['sample_channel_rel_l2']
                               for k,v in metrics.items()})
            selected=callback.best_exposure
            if selected is None:raise RuntimeError('Missing best model training exposure record')
            report=dict(status='complete',accepted=store.count,shared_initial_labels=len(initial.ids),
                newly_accepted=store.count-len(initial.ids),**query_costs(queue.path),
                training_exposure='dynamic',normalization='initial_train_frozen',
                unique_training_inputs=len(callback.replay.exposure),training_draws=sum(callback.replay.exposure.values()),
                optimizer_steps=callback.steps,optimizer_steps_with_pending_queries=callback.optimizer_steps_with_pending_queries,
                best_checkpoint_exposure=selected,accepted_by_epoch=callback.accepted_by_epoch,
                seconds=time.monotonic()-started,proposal_counts=dict(domain.counts),sources_unchanged=True,
                asynchronous_solver_training=True,generator_execution='bounded asynchronous CPU thread',
                sampler_not_ready_queries=callback.sampler_not_ready_queries,
                exploration_queries=callback.exploration_queries,
                generator_status=None if callback.background is None else callback.background.status(),
                native_melissa_pde_reproduction=False,source_trajectories_copied=False)
            report['artifacts']=_online_artifacts(directory);atomic_json(done,report)
            log_event('COMPLETE','online generation experiment',path=directory,samples=store.count)
            return directory
        finally:
            if callback is not None:callback.close()
            if wandb_run is not None:wandb_run.finish()
