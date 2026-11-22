"""HGGS GF, one main-model reset, then continuous legal projected MGS."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch

from budding_yeast_v2.data.dataset_archive import TrajectoryStore,atomic_json,file_sha256,fingerprint
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.utils.evaluation.checkpoints import Predictor,load_sampling_checkpoint
from budding_yeast_v2.utils.training.artifacts import read_history
from budding_yeast_v2.utils.training.monitoring import evaluation_state
from budding_yeast_v2.utils.experiments.curves import loss_figure
from budding_yeast_v2.utils.experiments.logging import tee_experiment_log,log_stage,log_event
from budding_yeast_v2.utils.experiments.resources import resource_slot,exclusive_gpu_slot
from ..baseline.hggs import HGGSConfig
from ..benchmark.protocol import BenchmarkConfig,child_seed,effective_signatures
from ..benchmark.runner import experiment_lock,source_inventory,training_costs
from ..benchmark.rounds import fit_round
from ..benchmark.reports import save_progress_reports
from ..contracts import QueriedTraining
from .cohorts import FrozenCohorts
from .domain import FrozenBuddingDomain
from .hggs_initial import prepare_hggs_initial
from .hggs_stages import HGGSStages
from .policies import GenerationPolicy
from .storage import GeneratedTrainingStore
from .runner import _view,_acquire,_artifacts,worker_lifecycle


class MGSProvider:
    def __init__(self,store,domain,policy,executor,directory,cfg,device,options):
        self.store,self.domain,self.policy,self.executor=store,domain,policy,executor
        self.directory,self.cfg,self.device,self.options=directory,cfg,device,options
        self.cache=None;self.active=cfg.budgets[0]
    def state(self):return dict(active=self.active)
    def restore(self,state):
        self.active=int(state['active']);self.cache=None
        if not 0<self.active<=self.store.count:raise ValueError('HGGS MGS recovery count exceeds committed data')
    def training(self,count):
        if count>self.store.count:raise ValueError('Missing HGGS generated training quota')
        self.active=count
        if self.cache is None or len(self.cache.inputs.ids)!=count:
            self.cache=QueriedTraining(_view(self.store,count),self.store.array('data')[:count])
        return self.cache
    def snapshot(self,path,count,index):
        if path.exists():return
        path.parent.mkdir(parents=True,exist_ok=True)
        with resource_slot('data','HGGS source-only stage manifest'):
            source=TrajectoryStore(self.store.publish(index))
            source.snapshot(path,np.arange(count),provenance=dict(method='hggs_projected',stage=index,
                training='continuous_current_model',source_trajectories_copied=False))
    def acquire(self,target,model,norm):
        if self.store.count>target:raise ValueError('HGGS generation is ahead of the recovering stage')
        if self.store.count<target:
            predictor=Predictor([model],norm,self.store.initial['data'].shape[-1],self.device,self.cfg.prediction_batch_size)
            with evaluation_state(model):
                _acquire(target,self.domain,self.store,self.policy,predictor,self.executor,self.directory,**self.options)
        self.cache=None


def run_hggs(protocol,pool,output,*,device='cpu',workers=4,bank_size=512,query_batch_size=32,
             lhs_config=None,hggs_config=None,max_attempts_mult=30,plots=False,wandb_enabled=False,diagnostic_only=False):
    root=Path(protocol).resolve();root=root if root.is_dir() else root.parent
    definition=json.loads((root/'protocol.json').read_text());cfg=BenchmarkConfig(**definition['config']).validate()
    settings_hggs=HGGSConfig(**(hggs_config or {})).validate()
    if pool not in definition['pools'] or cfg.validation_metric!='normalized_loss':raise ValueError('HGGS requires a source pool and normalized validation')
    if pool=='lhs' and lhs_config is None:raise ValueError('Provide the frozen LHS specification')
    if min(workers,bank_size,query_batch_size,max_attempts_mult)<1 or query_batch_size>bank_size:raise ValueError('Invalid HGGS query limits')
    settings=dict(workflow='hggs_continuous_gf_mgs_v2',protocol=definition['fingerprint'],pool=pool,
        method='hggs_projected',seed=cfg.seed,training=asdict(cfg),hggs=asdict(settings_hggs),
        workers=workers,bank_size=bank_size,query_batch_size=query_batch_size,max_attempts_mult=max_attempts_mult,
        lhs_config_sha256=None if lhs_config is None else file_sha256(lhs_config),diagnostic_only=diagnostic_only,
        normalization='shared_initial_training_frozen',feedback_checkpoint='current_model',
        checkpoint_selection='per_stage_same_source_normalized_validation_best_for_reporting_only',
        main_model_resets=1,sources=source_inventory())
    from ..layout import experiment_directory
    directory=experiment_directory(output,cfg,pool,'hggs_projected','generate')
    with experiment_lock(directory),resource_slot('experiment','HGGS continuous experiment'),exclusive_gpu_slot(device),tee_experiment_log(directory/'train.log'):
        old=directory/'config.json'
        if old.exists() and json.loads(old.read_text())!=settings:raise ValueError('HGGS configuration/source changed')
        atomic_json(old,settings);state_dir=directory/'_state';state_dir.mkdir(exist_ok=True)
        done=state_dir/'completion.json'
        if done.exists():
            result=json.loads(done.read_text())
            if result['accepted']!=cfg.budgets[-1]:raise ValueError('HGGS completed budget mismatch')
            for path,digest in result['artifacts'].items():
                if file_sha256(directory/path)!=digest:raise ValueError('Completed HGGS artifact changed')
            for path,digest in definition['artifacts'].items():
                if file_sha256(root/path)!=digest:raise ValueError('Frozen HGGS protocol changed')
            return directory
        with resource_slot('data','HGGS frozen cohorts and reference scales'):
            cohorts=FrozenCohorts(root,pool,cfg,diagnostic_only=diagnostic_only,lhs_config=lhs_config)
        reference_ids=cohorts.initial_rows.copy()
        reference=NormalizationStats.fit(cohorts.initial_inputs.params,cohorts.source.read(reference_ids)['data'])
        gf=prepare_hggs_initial(cohorts,root,cfg,directory,settings_hggs,device=device)
        committed=state_dir/'generation.json';state=json.loads(committed.read_text()) if committed.exists() else None
        with resource_slot('data','HGGS generated source recovery'):
            store=GeneratedTrainingStore(cohorts.source.path,cohorts.initial_rows,directory/'data',pool,cfg.seed,
                chunks=() if state is None else state['chunks'],generation_metadata=settings)
        # GF has already queried every coarse input, including discarded rows.
        # Do not spend a new ODE query on those known inputs during MGS.
        with np.load(root/f'{pool}_inputs.npz',allow_pickle=False) as inputs:
            coarse_rows=np.asarray(gf['coarse_source_ids'],dtype=int)
            known_inputs=effective_signatures(inputs['params'][coarse_rows],inputs['y_init'][coarse_rows])
        previous={} if state is None else state['domain']
        domain=FrozenBuddingDomain(pool,child_seed(cfg.seed,'generated_proposals'),state_dir/'proposals',
            lhs_config=lhs_config,forbidden=cohorts.forbidden,
            seen=previous.get('seen',known_inputs),
            next_index=previous.get('next_index',0))
        if state is not None:
            if domain.sources!=previous['sources']:raise ValueError('Canonical HGGS sources changed')
            domain.counts.update(previous['counts'])
        policy=GenerationPolicy('hggs_projected',cohorts.initial_inputs,store.initial['data'],
            child_seed(cfg.seed,'generation_policy'),hggs=settings_hggs,device=device)
        if state is not None:
            if file_sha256(state['policy_path'])!=state['policy_sha256']:raise ValueError('HGGS policy checkpoint changed')
            policy.load_state_dict(torch.load(state['policy_path'],map_location='cpu',weights_only=False))
        wandb_run=None
        try:
            if wandb_enabled:
                import wandb
                os.environ['WANDB_MODE']='offline'
                wandb_run=wandb.init(project='budding_sampling_generation',mode='offline',dir=str(state_dir),config=settings)
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
                    initializer=worker_lifecycle,initargs=(os.getpid(),)) as executor:
                provider=MGSProvider(store,domain,policy,executor,directory,cfg,device,dict(bank_size=bank_size,
                    query_batch_size=query_batch_size,max_attempts_mult=max_attempts_mult))
                stages=HGGSStages(provider,cfg.budgets,cfg.epochs,directory/'rounds',reference,reference_ids,
                    cohorts.validation,'val_loss_synth',keep_phase_weights=True,resume_path=state_dir/'primary/resume.ckpt')
                snapshot=directory/'rounds/round_00/dataset.json';provider.snapshot(snapshot,cfg.budgets[0],0)
                train_cfg=replace(cfg,budgets=(cfg.budgets[0],),epochs=(sum(cfg.epochs),),warmup=(cfg.warmup[0],))
                with log_stage('HGGS one-reset continuous MGS training',epochs=sum(cfg.epochs)):
                    model,_,_=fit_round(train_cfg,0,provider.training(cfg.budgets[0]),cohorts.validation,
                        state_dir/'continuous',child_seed(cfg.seed,'hggs_mgs_model'),device=device,dataset_path=snapshot,
                        state_dir=state_dir/'primary',identity=dict(workflow=settings['workflow'],configuration=fingerprint(settings)),
                        data_adapter=stages.adapt_data,after_epoch_callbacks=(stages,),reload_dataloaders_every_n_epochs=1,
                        monitoring=cohorts.monitoring,wandb_run=wandb_run)
                model.cpu()
            progress=json.loads((directory/'rounds/stages.json').read_text())
            if not progress['complete'] or store.count!=cfg.budgets[-1]:raise RuntimeError('Incomplete HGGS continuous stages')
            records=[];histories=[];contexts={}
            for i,budget in enumerate(cfg.budgets):
                rd=directory/f'rounds/round_{i:02d}';model,norm,_=load_sampling_checkpoint(rd/'model.ckpt',device)
                predictor=Predictor([model],norm,store.initial['data'].shape[-1],device,cfg.prediction_batch_size)
                context=dict(model=cfg.model,loss=cfg.loss,pool=pool,method='hggs_projected',training_rows=budget,seed=cfg.seed)
                metrics=cohorts.evaluate(predictor,rd/'eval',i,budget,
                    plots=plots and (i == len(cfg.budgets) - 1),context={k:v for k,v in context.items() if k!='loss'})
                records.append(dict(round=i,training_rows=budget,metrics=metrics));contexts[i]=context
                history=read_history(rd/'history.csv');histories.append((i,history))
                loss_figure(history,rd/'loss.png',title=f'HGGS stage {i}',context=context)
                save_progress_reports(directory,records,histories,contexts,rd);model.cpu()
            domain.verify()
            from .async_queries import query_costs
            costs=query_costs(state_dir/'ode_attempts.jsonl')
            gf_access=json.loads((state_dir/'hggs_label_access.json').read_text())
            result=dict(status='complete',accepted=store.count,shared_initial_labels=gf['shared_random_initial_overlap'],
                newly_accepted=store.count-len(store.initial_rows),**costs,
                training_schedule='GF continuous; one reset; MGS continuous',main_model_resets=1,
                normalization='shared_initial_training_frozen',training_costs=training_costs(directory),
                hggs_gf=dict(coarse_labels=gf['coarse_labels'],unique_labels_accessed=gf_access['unique_labels_accessed'],
                    initial_cohort_matches_shared_random=False,
                    pilot_training=json.loads((state_dir/'hggs_pilot/primary/_state/training.json').read_text())),
                stage_records=progress['records'],sources_unchanged=True,source_trajectories_copied=False)
            result['artifacts']=_artifacts(directory)
            for p in [*directory.glob('rounds/round_*/stage.json'),directory/'rounds/stages.json',
                      state_dir/'primary/training.json',state_dir/'primary/last_weights.ckpt',state_dir/'continuous/history.csv']:
                result['artifacts'][str(p.relative_to(directory))]=file_sha256(p)
            atomic_json(done,result);log_event('COMPLETE','HGGS continuous experiment',path=directory,samples=store.count)
            return directory
        finally:
            if wandb_run is not None:wandb_run.finish()
