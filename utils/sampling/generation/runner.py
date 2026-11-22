"""Round-controlled generation studies using shared training/evaluation services.

OGAS here is explicitly a round-based, legal-projection adaptation. It does not
claim Melissa's continuously coupled online surrogate-training schedule.
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
from budding_yeast_v2.data.simulation_guards import bounded_ordered_map
from budding_yeast_v2.data.systems.budding.offline_pool import simulate
from ..benchmark.protocol import BenchmarkConfig,child_seed,effective_signatures
from ..benchmark.runner import experiment_lock,source_inventory,_json_safe
from ..benchmark.rounds import fit_round
from ..benchmark.reports import save_progress_reports
from ..contracts import PoolInputs,QueriedTraining
from ..baseline.oed import OEDConfig
from ..baseline.ogas import OGASConfig
from ..baseline.hggs import HGGSConfig
from budding_yeast_v2.utils.evaluation.checkpoints import Predictor
from budding_yeast_v2.utils.experiments.logging import log_stage,log_event,tee_experiment_log
from budding_yeast_v2.utils.experiments.resources import resource_slot,exclusive_gpu_slot
from .cohorts import FrozenCohorts
from .domain import FrozenBuddingDomain
from .policies import GenerationPolicy,METHODS,OUR_METHODS
from .storage import GeneratedTrainingStore
from ..ours.adaptive_generation import METHODS as ADAPTIVE_METHODS


def worker_lifecycle(parent):
    """Prevent solver orphans after abrupt supervisor death; no recursive launch."""
    import ctypes
    import signal
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM)!=0:
        raise OSError('Cannot arm solver parent-death signal')
    if os.getppid()!=parent:raise RuntimeError('Solver supervisor already exited')


def _view(store,count=None,*,metadata=False):
    n=store.count if count is None else count
    rows=()
    if metadata:
        rows=[json.loads(str(value)) for value in store.array('candidate_metadata_json')[:n]]
        for row,value in zip(rows,store.array('overrides_json')[:n]):
            row['_effective_overrides']=json.loads(str(value))
    return PoolInputs(np.arange(n),store.array('params')[:n],store.array('y_init')[:n],tuple(rows))


def _artifacts(directory):
    patterns=('data/chunks/*.npz','data/training_source_*.json','data/training_source_*.npz',
              'rounds/round_*/model.ckpt','rounds/round_*/dataset.json','rounds/round_*/history.csv',
              'rounds/round_*/eval/global_metrics_denorm.json','_state/generation.json',
              '_state/hggs_gf.json','_state/hggs_label_access.json','_state/hggs_pilot/**/*.json',
              '_state/hggs_pilot/**/*.ckpt','_state/hggs_pilot/**/*.csv',
              '_state/round_*/primary/last_weights.ckpt','_state/policy_*.pt','performance.csv')
    return {str(p.relative_to(directory)):file_sha256(p)
            for pattern in patterns for p in sorted(directory.glob(pattern))}


def _commit(directory,domain,store,policy):
    # Versioned policy files: interruption between writes cannot invalidate an
    # older committed JSON pointer. Orphan files are not treated as completed.
    state_dir=directory/'_state';state_dir.mkdir(exist_ok=True)
    path=state_dir/f'policy_{len(store.chunks):06d}_{domain.next_index:09d}.pt'
    temporary=path.with_suffix('.tmp')
    torch.save(policy.state_dict(),temporary);os.replace(temporary,path)
    atomic_json(state_dir/'generation.json',dict(chunks=store.chunks,domain=domain.state(),
        policy_path=str(path),policy_sha256=file_sha256(path),accepted=store.count))
    # Keep the current and previous committed recovery point, not a DDPM copy
    # for every acquisition. Historical training best/last remain unchanged.
    checkpoints=sorted(state_dir.glob('policy_*.pt'))
    for obsolete in checkpoints[:-2]:obsolete.unlink()


def _acquire(target,domain,store,policy,predictor,executor,directory,*,bank_size,query_batch_size,max_attempts_mult):
    policy.refresh(_view(store,metadata=policy.name in OUR_METHODS),store.array('data'),predictor,target)
    attempts=0;needed=target-store.count;pending=[];last_commit=0
    while store.count+len(pending)<target:
        if attempts>=needed*max_attempts_mult:
            if pending:
                store.append(pending);_commit(directory,domain,store,policy)
            raise RuntimeError('Generation query cap reached; no partial completion')
        count=min(query_batch_size,target-store.count-len(pending),needed*max_attempts_mult-attempts)
        scoring_resource='cpu_scoring' if policy.name in OUR_METHODS else 'scoring'
        with resource_slot(scoring_resource,'legal proposal generation and scoring'):
            if policy.name in ADAPTIVE_METHODS:
                from .interventions import intervention_bank
                candidates,signatures,bank=intervention_bank(domain,policy.controller,max(bank_size,count))
            elif policy.name=='ours_intervention_closure':
                from .controlled import controlled_bank
                candidates,signatures,bank=controlled_bank(domain,max(bank_size,count))
            else:
                candidates,signatures,bank=domain.bank(max(bank_size,count))
            if policy.name in OUR_METHODS:
                bank=PoolInputs(bank.ids,bank.params,bank.initial,tuple(dict(c.metadata,
                    _effective_overrides=dict(c.overrides)) for c in candidates))
            ids,diagnostics=policy.select(bank,count)
        chosen=[candidates[i] for i in ids]
        if policy.name=='ogas_round_projected':
            flags=diagnostics['reference_flags']
            if len(flags)!=len(chosen):raise ValueError('OGAS proposal/reference flags are unaligned')
            chosen=[replace(candidate,metadata=dict(candidate.metadata,ogas_is_reference=bool(reference)))
                    for candidate,reference in zip(chosen,flags)]
        for i in ids:domain.consume(signatures[i])
        with (directory/'_state/ode_attempts.jsonl').open('a') as log:
            for candidate in chosen:
                log.write(json.dumps(_json_safe(dict(event='start',
                    proposal_id=candidate.metadata['generation_proposal_index'],overrides=candidate.overrides)))+'\n')
            log.flush();os.fsync(log.fileno())
        with log_stage('new canonical ODE queries',queries=len(chosen),target=target):
            results=list(bounded_ordered_map(executor,simulate,chosen,stall_timeout=35.))
        successful=[]
        with (directory/'_state/ode_attempts.jsonl').open('a') as log:
            for result in results:
                attempts+=1
                record=dict(event='finish',proposal_id=result.candidate.metadata['generation_proposal_index'],status=result.status,
                            overrides=result.candidate.overrides,diagnostics=result.diagnostics)
                log.write(json.dumps(_json_safe(record),allow_nan=False)+'\n')
                if result.status=='success':successful.append(result)
            log.flush();os.fsync(log.fileno())
        if policy.name in ADAPTIVE_METHODS:
            failed=[r for r in results if r.status!='success']
            if failed:
                policy.controller.discard(PoolInputs(np.arange(len(failed)),np.asarray([r.params for r in failed]),
                                                     np.asarray([r.y_init for r in failed])))
        if successful:
            view=PoolInputs(np.arange(len(successful)),np.asarray([r.params for r in successful]),
                            np.asarray([r.y_init for r in successful]))
            # Feedback uses the model from BEFORE these labels were revealed.
            flags=(np.asarray([r.candidate.metadata['ogas_is_reference'] for r in successful])
                   if policy.name=='ogas_round_projected' else None)
            policy.observe(view,np.asarray([r.trajectory for r in successful]),predictor,is_reference=flags)
            pending.extend(successful)
        if len(pending)>=128 or store.count+len(pending)==target or attempts-last_commit>=128:
            store.append(pending);pending=[]
            _commit(directory,domain,store,policy);last_commit=attempts
        diagnostics.update(valid=len(successful),attempts=len(results),accepted_total=store.count+len(pending))
        with (directory/'_state/acquisition.jsonl').open('a') as log:
            log.write(json.dumps(_json_safe(diagnostics),allow_nan=False)+'\n')
        log_event('PROGRESS','generated training data',accepted=store.count+len(pending),target=target,
                  successful=len(successful),failed=len(results)-len(successful))


def run_generation(protocol,pool,method,output,*,device='cpu',workers=4,bank_size=512,
                   query_batch_size=None,lhs_config=None,oed_config=None,ogas_config=None,hggs_config=None,
                   max_attempts_mult=30,plots=False,wandb_enabled=False,diagnostic_only=False):
    if query_batch_size is None:query_batch_size=1 if method=='oed_temporal' else 32
    if method not in METHODS or min(workers,bank_size,query_batch_size,max_attempts_mult)<1:
        raise ValueError('Invalid generation method or positive resource limits')
    if query_batch_size>bank_size:raise ValueError('Query batch cannot exceed legal proposal bank')
    if method=='hggs_projected':
        from .hggs_runner import run_hggs
        return run_hggs(protocol,pool,output,device=device,workers=workers,bank_size=bank_size,
            query_batch_size=query_batch_size,lhs_config=lhs_config,hggs_config=hggs_config,
            max_attempts_mult=max_attempts_mult,plots=plots,wandb_enabled=wandb_enabled,diagnostic_only=diagnostic_only)
    root=Path(protocol).resolve();root=root if root.is_dir() else root.parent
    definition=json.loads((root/'protocol.json').read_text());cfg=BenchmarkConfig(**definition['config']).validate()
    if pool not in definition['pools']:raise ValueError('Unknown source pool')
    if pool!='mutation' and method in ADAPTIVE_METHODS:
        raise ValueError('Adaptive intervention prototypes currently support mutation generation only')
    if pool=='lhs' and lhs_config is None:raise ValueError('Provide the frozen LHS specification')
    if pool=='lhs' and method=='ours_intervention_closure':
        raise ValueError('Intervention closure requires legal mutation controls, not an unlabelled LHS design')
    oed=OEDConfig(**(oed_config or {})).validate();ogas=OGASConfig(**(ogas_config or {})).validate()
    hggs=HGGSConfig(**(hggs_config or {})).validate()
    # A versioned namespace prevents source changes from overwriting old results.
    settings=dict(workflow='legal_open_generation_round_adaptations_v1',protocol=definition['fingerprint'],
        seed=cfg.seed,pool=pool,method=method,training=asdict(cfg),bank_size=bank_size,
        query_batch_size=query_batch_size,max_attempts_mult=max_attempts_mult,
        oed=asdict(oed),ogas=asdict(ogas),hggs=asdict(hggs),lhs_config_sha256=None if lhs_config is None else file_sha256(lhs_config),
        diagnostic_only=diagnostic_only,sources=source_inventory())
    run_id=f"{definition['protocol_id']}_gen{fingerprint(settings)[:8]}"
    from ..layout import experiment_directory
    directory=experiment_directory(output,cfg,pool,method,'generate')
    with experiment_lock(directory),resource_slot('experiment','generation experiment'),exclusive_gpu_slot(device), \
            tee_experiment_log(directory/'train.log'):
        started=time.monotonic()
        config_path=directory/'config.json'
        if config_path.exists() and json.loads(config_path.read_text())!=settings:raise ValueError('Generation configuration changed')
        atomic_json(config_path,settings)
        done=directory/'_state/completion.json'
        if done.exists():
            result=json.loads(done.read_text())
            if result['accepted']!=cfg.budgets[-1]:raise ValueError('Wrong completed generation budget')
            with resource_slot('data','verify completed generation artifacts'):
                for relative,digest in result['artifacts'].items():
                    path=directory/relative
                    if not path.exists() or file_sha256(path)!=digest:raise ValueError(f'Completed generation artifact changed: {relative}')
                for relative,digest in definition['artifacts'].items():
                    if file_sha256(root/relative)!=digest:raise ValueError('Frozen protocol artifact changed')
            return directory
        with resource_slot('data','generation frozen cohorts'):
            cohorts=FrozenCohorts(root,pool,cfg,diagnostic_only=diagnostic_only,lhs_config=lhs_config)
        with resource_slot('data','generated source initialization'):
            committed=directory/'_state/generation.json'
            state=json.loads(committed.read_text()) if committed.exists() else None
            store=GeneratedTrainingStore(cohorts.source.path,cohorts.initial_rows,directory/'data',pool,cfg.seed,
                chunks=() if state is None else state['chunks'],
                generation_metadata=dict(method=method,workflow=settings['workflow'],configuration=fingerprint(settings)))
        if len(cohorts.initial_rows)!=cfg.budgets[0]:raise ValueError('Initial source cohort disagrees with budget')
        initial=cohorts.initial_inputs
        seen=set(effective_signatures(initial.params,initial.initial))
        domain_state={} if state is None else state['domain']
        domain=FrozenBuddingDomain(pool,child_seed(cfg.seed,'generated_proposals'),directory/'_state/proposals',
            lhs_config=lhs_config,forbidden=cohorts.forbidden,seen=domain_state.get('seen',seen),
            next_index=domain_state.get('next_index',0))
        if state is not None:
            if domain.sources!=domain_state['sources']:raise ValueError('Biological source identity changed')
            domain.counts.update(domain_state['counts'])
        policy=GenerationPolicy(method,initial,store.initial['data'],child_seed(cfg.seed,'generation_policy'),
                                oed=oed,ogas=ogas,hggs=hggs,device=device)
        if state is not None:
            if file_sha256(state['policy_path'])!=state['policy_sha256']:raise ValueError('Generation policy changed')
            policy.load_state_dict(torch.load(state['policy_path'],map_location=device,weights_only=False))
        records=[];histories=[];contexts={};wandb_run=None
        try:
            if wandb_enabled:
                import wandb
                os.environ['WANDB_MODE']='offline'
                wandb_run=wandb.init(project='budding_sampling_generation',group=run_id,
                    name=f'{pool}/{method}/seed{cfg.seed}',dir=str(directory/'_state'),mode='offline',config=settings)
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
                                     initializer=worker_lifecycle,initargs=(os.getpid(),)) as executor:
                for i,budget in enumerate(cfg.budgets):
                    if store.count<budget:raise RuntimeError('Missing generated training quota')
                    rd=directory/f'rounds/round_{i:02d}';rd.mkdir(parents=True,exist_ok=True)
                    snapshot=rd/'dataset.json'
                    if not snapshot.exists():
                        if store.count!=budget:raise ValueError('Missing historical snapshot; cannot reconstruct altered sources')
                        with resource_slot('data','generated training manifest'):
                            source=TrajectoryStore(store.publish(i));source.snapshot(snapshot,np.arange(budget),
                                provenance=dict(method=method,seed=cfg.seed,round=i,source_trajectories_copied=False))
                    training=QueriedTraining(_view(store,budget),store.array('data')[:budget])
                    context=dict(model=cfg.model,loss=cfg.loss,pool=pool,method=method,training_rows=budget,
                                 seed=cfg.seed,warmup=cfg.warmup[i]);contexts[i]=context
                    with log_stage('generation round training',round=i,method=method,samples=budget):
                        model,norm,history=fit_round(cfg,i,training,cohorts.validation,rd,
                            child_seed(cfg.seed,'model',i,0),device=device,dataset_path=snapshot,
                            identity=dict(workflow=settings['workflow'],configuration=fingerprint(settings),
                                          rows_sha256=fingerprint(training.inputs.ids.tolist())),
                            state_dir=directory/f'_state/round_{i:02d}/primary',plot_context=context,
                            monitoring=cohorts.monitoring,wandb_run=wandb_run)
                    predictor=Predictor([model],norm,training.trajectories.shape[-1],device,cfg.prediction_batch_size)
                    histories.append((i,history))
                    with log_stage('generation independent evaluation',round=i):
                        metrics=cohorts.evaluate(predictor,rd/'eval',i,budget,
                            plots=plots and (i == len(cfg.budgets) - 1),
                            context={k:v for k,v in context.items() if k not in ('warmup','loss')})
                    records.append(dict(round=i,training_rows=budget,metrics=metrics))
                    save_progress_reports(directory,records,histories,contexts,rd)
                    if wandb_run is not None:
                        wandb_run.log({f'evaluation/{name}/sample_channel_rel_l2_physical':v['sample_channel_rel_l2']
                                       for name,v in metrics.items()}|dict(round=i,training_rows=budget))
                    if i==len(cfg.budgets)-1:continue
                    if state is None and policy.feedback_rows==0 and method=='ogas_round_projected':
                        policy.observe(initial,store.initial['data'],predictor,
                                       is_reference=np.ones(len(initial.ids),dtype=bool))
                    if store.count<cfg.budgets[i+1]:
                        _acquire(cfg.budgets[i+1],domain,store,policy,predictor,executor,directory,
                            bank_size=bank_size,query_batch_size=query_batch_size,max_attempts_mult=max_attempts_mult)
                    model.cpu()
                    if str(device).startswith('cuda'):torch.cuda.empty_cache()
            domain.verify()
            attempts=0;finished=0;failures=0;ode_calls=0
            path=directory/'_state/ode_attempts.jsonl'
            if path.exists():
                for line in path.read_text().splitlines():
                    row=json.loads(line)
                    if row['event']=='start':attempts+=1;continue
                    finished+=1;failures+=int(row['status']!='success')
                    ode_calls+=len(row['diagnostics'].get('attempts',[None]))
            gf_path=directory/'_state/hggs_gf.json'
            gf=None if not gf_path.exists() else json.loads(gf_path.read_text())
            gf_access=directory/'_state/hggs_label_access.json'
            gf_access=None if not gf_access.exists() else json.loads(gf_access.read_text())
            atomic_json(done,dict(status='complete',accepted=store.count,
                shared_initial_labels=len(initial.ids) if gf is None else gf['shared_random_initial_overlap'],
                hggs_gf=None if gf is None else dict(coarse_labels=gf['coarse_labels'],
                    unique_labels_accessed=gf_access['unique_labels_accessed'],initial_cohort_matches_shared_random=False,
                    pilot_training=json.loads((directory/'_state/hggs_pilot/primary/_state/training.json').read_text())
                    if (directory/'_state/hggs_pilot/primary/_state/training.json').exists() else None),
                newly_accepted=store.count-len(initial.ids),attempted_inputs=attempts,failed_inputs=failures,
                solver_attempts=ode_calls+attempts-finished,solver_attempts_exact=attempts==finished,
                incomplete_query_attempts=attempts-finished,
                seconds=time.monotonic()-started,proposal_counts=dict(domain.counts),
                sources_unchanged=True,native_online_ogas=False,training_schedule='shared cold-start rounds',
                conditional_reference='canonical legal proposal distribution',source_trajectories_copied=False,
                artifacts=_artifacts(directory)))
            log_event('COMPLETE','generation experiment',path=directory,samples=store.count)
            return directory
        finally:
            if wandb_run is not None:wandb_run.finish()
