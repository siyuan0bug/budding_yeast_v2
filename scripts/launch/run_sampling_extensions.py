#!/usr/bin/env python3
"""Seed42 completion queue for independent fixed-pool/dynamic/generative baselines.

No model or selection implementation lives here. Every child uses the public
run_sampling_budding.sh CLI. Only Popen-owned isolated groups are monitored/signalled.
"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.dataset_archive import atomic_json,file_sha256,fingerprint,TrajectoryStore
from budding_yeast_v2.utils.sampling.benchmark.runner import source_inventory,experiment_lock
from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig
from budding_yeast_v2.utils.experiments.resources import resource_state,memory_status
from budding_yeast_v2.utils.experiments.memory_guard import BatchMemoryGuard
from budding_yeast_v2.scripts.launch import queue_handover as handovers

DATA=Path('/home/users/zsy/mydata/budding_yeast')
PROTOCOL=DATA/'experiments/budding_sampling/shared/pool100k/seed_42'


def tasks(suite='extensions'):
    if suite=='all':
        primary=[('run','random'),('run','power'),('run','lcmd_ll'),('run','physics_pre_power'),
                 ('generate','hggs_projected'),('run','low_rank'),('dynamic','peaks_regression'),
                 ('generate','ogas_async_projected')]
        rows=[dict(kind=k,method=m,pool=p,task_id=f'{k}_{p}_{m}',status='pending')
              for k,m in primary for p in ('mutation','lhs')]
        ours=[r for r in tasks('ours') if r['method'] in ('ours_sensitivity','ours_dynamics_transport','ours_intervention_closure')]
        return rows+ours
    if suite=='ours':
        from budding_yeast_v2.utils.sampling.registry import METHODS
        from budding_yeast_v2.utils.sampling.ours.mechanistic import METHOD_NAMES
        # Start new kernels alongside two established methods, then continue
        # through every supported complete-pool comparison and open generation.
        order=('ours_mechanism_witness','ours_dynamics_transport',
               'ours_pattern_context','ours_pattern_support',
               'ours_intervention_closure','ours_sensitivity','ours_pattern_balance')
        rows=[dict(kind='run',method=name,pool=pool)
              for pool in ('mutation','lhs') for name in order
              if not METHODS[name].mutation_only or pool=='mutation']
        rows += [dict(kind='generate',method=name,pool=pool,bank_size=64,
                      query_batch_size=8,workers=2)
                 for pool in ('mutation','lhs') for name in METHOD_NAMES
                 if not METHODS[name].mutation_only or pool=='mutation']
        return [dict(row,task_id=f"{row['kind']}_{row['pool']}_{row['method']}",status='pending')
                for row in rows]
    if suite!='extensions':raise ValueError('Unknown completion queue suite')
    # First four exercise the newly implemented kernels on the real workload.
    primary=[('run','low_rank'),('dynamic','peaks_regression'),('dynamic','eracs_regression'),
             ('generate','hggs_projected'),('run','lev_input'),('run','clustering_sensitivity'),
             ('run','clustering_sensitivity_wor'),('run','physics_pre_topk'),('run','physics_pre_power'),
             ('generate','oed_temporal'),('generate','ogas_async_projected')]
    controls=[('generate','random_generate'),('generate','random_async'),
              ('dynamic','random_stream'),('dynamic','random_dynamic')]
    rows=[]
    for pool in ('mutation','lhs'):
        for kind,name in primary[:4]:rows.append(dict(kind=kind,method=name,pool=pool))
    for kind,name in primary[4:]+controls:
        for pool in ('mutation','lhs'):rows.append(dict(kind=kind,method=name,pool=pool))
    return [dict(row,task_id=f"{row['pool']}_{row['method']}",status='pending') for row in rows]


def failed_tasks(rows,previous):
    """Select failed task identities only; never inherit old output/state files."""
    if not isinstance(previous,list) or len({t['task_id'] for t in previous})!=len(previous):
        raise ValueError('Invalid previous task ledger')
    known={t['task_id']:t for t in rows};selected=[]
    for old in previous:
        if old['status']!='failed':continue
        current=known.get(old['task_id'])
        if current is None or any(old[k]!=current[k] for k in ('kind','pool','method')):
            raise ValueError(f'Unknown/changed failed task: {old["task_id"]}')
        selected.append(dict(current,rerun_from_result_directory=old.get('result_directory'),
                             previous_exit_code=old.get('exit_code')))
    if not selected:raise ValueError('Previous queue has no failed tasks to rerun')
    return selected


def next_pending(rows, active_tasks, mechanism_limit):
    """Keep long CPU acquisitions from occupying every GPU job position."""
    from budding_yeast_v2.utils.sampling.ours.mechanistic import METHOD_NAMES
    occupied=sum(t['method'] in METHOD_NAMES for t in active_tasks)
    return next((t for t in rows if t['status']=='pending' and
                 (t['method'] not in METHOD_NAMES or occupied < mechanism_limit)),None)


def gpu_memory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
    return {a.strip():float(b.strip())/1024 for a,b in (line.split(',') for line in raw.splitlines())}


def find_result(task,protocol_id,started=0.):
    root=Path(task['output'])
    from budding_yeast_v2.utils.sampling.layout import method_name
    paths=list(root.glob(f"*/{task['pool']}/{method_name(task['method'],task['kind'])}/*/seed_{task.get('seed',42)}/config.json"))
    eligible=[p for p in paths if p.stat().st_mtime>=started-2]
    return str(max(eligible,key=lambda p:p.stat().st_mtime).parent) if eligible else task.get('result_directory')


def validate_result(task):
    if not task.get('result_directory'):raise ValueError('Child exited without a result directory')
    directory=Path(task['result_directory']);evaluation=directory/'eval'
    marker=directory/('_state/state.json' if task['kind']=='run' else '_state/completion.json')
    state=json.loads(marker.read_text());state=state.get('completion',state)
    if state.get('status')!='complete':raise ValueError('Missing completed workflow state')
    metrics=json.loads((evaluation/'global_metrics_denorm.json').read_text())
    if metrics['real_mutants']['count']!=126:raise ValueError('Incomplete real-mutant evaluation')
    for name,count in [('per_mutant_metrics.csv',126),('per_synth_test_metrics.csv',metrics['synth_test']['count'])]:
        with (evaluation/name).open() as handle:actual=sum(1 for _ in csv.DictReader(handle))
        if actual!=count:raise ValueError(f'Incomplete sample metrics: {name}')
    if len(list((evaluation/'real_mutants').glob('*.png')))!=126:raise ValueError('Incomplete real-mutant figures')
    return dict(real_mutants=126,synth_test=metrics['synth_test']['count'],real_figures=126)


def verified_spec_bundle(protocol,directory):
    original=DATA/'offline_pools/lhs_matched_100000_seed42/lhs_specification.json'
    source=TrajectoryStore(protocol['pools']['lhs']['path'])
    config=json.loads(str(source.scalar('generation_config').item()))
    from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
    if (file_sha256(original)!=config['lhs_spec_sha256']
            or str(source.scalar('sampling_range_fingerprint').item())!=RANGE_FINGERPRINT):
        raise ValueError('Reviewed LHS specification/range does not match the source pool')
    bundle=directory/'frozen_lhs';bundle.mkdir(exist_ok=True)
    destination=bundle/'lhs_specification.json'
    if not destination.exists():shutil.copyfile(original,destination)
    if file_sha256(destination)!=file_sha256(original):raise ValueError('Frozen LHS spec copy changed')
    # Original pool copy no longer has its inventory sidecar. Reconstruct only
    # the provenance needed by the existing read-only domain validator, using
    # the pool's verified fingerprint; no marginal or biological value changes.
    atomic_json(bundle/'inventory.json',dict(range_fingerprint=RANGE_FINGERPRINT,
        specification_sha256=file_sha256(destination),verified_against_pool=str(source.path),
        verified_pool_sha256=source.digest,original_spec=str(original),provenance_only=True))
    return destination


def command(task,protocol,lhs_spec):
    cmd=['bash',str(ROOT/'run_sampling_budding.sh'),task['kind'],'--protocol',str(protocol),
         '--pool',task['pool'],'--method',task['method'],'--output',task['output'],
         '--device','cuda:0','--plots','--wandb']
    if task['kind']=='run':cmd+=['--seed',str(task.get('seed',42))]
    elif task['kind']=='generate':
        cmd+=['--workers',str(task.get('workers',2)),'--bank-size',str(task.get('bank_size',512))]
        if 'query_batch_size' in task:cmd+=['--query-batch-size',str(task['query_batch_size'])]
        if task['pool']=='lhs':cmd+=['--lhs-config',str(lhs_spec)]
    return cmd


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--directory',type=Path,required=True,help='Queue status/logs only; results retain public workflow roots')
    parser.add_argument('--gpus',nargs='+',default=['0','1','2','3'])
    parser.add_argument('--initial-jobs',type=int,default=4)
    parser.add_argument('--max-jobs',type=int,default=8)
    parser.add_argument('--jobs-per-gpu',type=int,default=2)
    parser.add_argument('--ramp-after',type=float,default=900.)
    parser.add_argument('--ramp-interval',type=float,default=90.)
    parser.add_argument('--gpu-free-gib',type=float,default=12.)
    parser.add_argument('--cache-chunks',type=int,default=1024)
    parser.add_argument('--retry-failed-from',type=Path,help='Previous tasks.json; rerun failed identities in fresh results')
    parser.add_argument('--results-root',type=Path,help='Fresh parent for fixed_pool/generation/dynamic result roots')
    parser.add_argument('--suite',choices=('extensions','ours','all'),default='extensions')
    parser.add_argument('--methods',nargs='+',choices=sorted({t['method'] for suite in ('extensions','ours','all') for t in tasks(suite)}),
                        help='Run only these methods, on both pools; completed runs may be rerun in a fresh root')
    parser.add_argument('--mechanism-workers',type=int,default=8)
    parser.add_argument('--mechanism-concurrency',type=int,default=2)
    parser.add_argument('--io-concurrency',type=int,default=4)
    parser.add_argument('--ramp-step',type=int,default=1)
    parser.add_argument('--adopt-from',type=Path,help='Explicit detached supervisor receipt; keep live jobs running')
    parser.add_argument('--admit-all-pending',action='store_true',
                        help='Admit all experiment types; retain per-phase CPU/IO semaphores')
    parser.add_argument('--added-resource-dir',type=Path,help='Reuse the verified previous added-job resource domain')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    args.protocol=args.protocol or PROTOCOL.parent/f'seed_{args.seed}'
    if (set(args.gpus)-{'0','1','2','3'} or len(set(args.gpus))!=len(args.gpus)
            or not 1<=args.initial_jobs<=args.max_jobs<=16 or not 1<=args.jobs_per_gpu<=8
            or min(args.ramp_after,args.ramp_interval,args.gpu_free_gib)<=0
            or not 1<=args.cache_chunks<=4096
            or not 1<=args.mechanism_workers<=min(16,len(os.sched_getaffinity(0)))
            or not 1<=args.mechanism_concurrency<=16
            or not 1<=args.io_concurrency<=8 or not 1<=args.ramp_step<=4):raise ValueError('Invalid bounded shared-GPU queue settings')
    protocol=json.loads((args.protocol/'protocol.json').read_text());cfg=BenchmarkConfig(**protocol['config']).validate()
    if cfg.seed!=args.seed:raise ValueError('Queue seed disagrees with shared protocol')
    rows=tasks(args.suite)
    retry_source=None
    if args.retry_failed_from is not None:
        if args.results_root is None:raise ValueError('Failed-task reruns require a fresh --results-root')
        retry_source=dict(path=str(args.retry_failed_from.resolve()),sha256=file_sha256(args.retry_failed_from))
        previous_manifest=args.retry_failed_from.parent/'manifest.json'
        if previous_manifest.exists():
            previous=json.loads(previous_manifest.read_text())
            if previous['protocol_fingerprint']!=protocol['fingerprint']:
                raise ValueError('Retry protocol differs from the original queue')
        rows=failed_tasks(rows,json.loads(args.retry_failed_from.read_text()))
    if args.methods is not None:
        rows=[row for row in rows if row['method'] in args.methods]
        if not rows:raise ValueError('No tasks match the requested methods')
    outputs=dict(run=DATA/'experiments/budding_sampling',generate=DATA/'experiments/budding_sampling',
                 dynamic=DATA/'experiments/budding_sampling')
    if args.results_root is not None:
        outputs={kind:args.results_root for kind in ('run','generate','dynamic')}
    for row in rows:
        row['output']=str(outputs[row['kind']]);row['seed']=cfg.seed
    if args.dry_run:
        print(json.dumps(dict(seed=cfg.seed,protocol=protocol['protocol_id'],tasks=rows,count=len(rows),
            initial_jobs=args.initial_jobs,maximum_jobs=args.max_jobs,gpus=args.gpus),ensure_ascii=False,indent=2));return
    if args.adopt_from is not None and args.directory.resolve()==args.adopt_from.resolve():
        raise ValueError('Live adoption requires a fresh supervisor directory')
    args.directory.mkdir(parents=True,exist_ok=True)
    with experiment_lock(args.directory):
        manifest=args.directory/'manifest.json';sources=source_inventory()
        handover=None if args.adopt_from is None else handovers.load_handover(args.adopt_from,sources)
        if handover is not None:
            if manifest.exists():raise ValueError('Adoption is one-time; use a new receipt for another transfer')
            if handover['protocol_fingerprint']!=protocol['fingerprint']:
                raise ValueError('Live adoption protocol changed')
            expected={t['task_id']:t for t in rows}
            if set(expected)!={t['task_id'] for t in handover['tasks']}:
                raise ValueError('Live adoption task set changed')
            for old in handover['tasks']:
                if any(old[k]!=expected[old['task_id']][k] for k in ('kind','method','pool','output')):
                    raise ValueError('Live adoption task definition/output changed')
                if old['status']=='running' and old['gpu'] not in args.gpus:
                    raise ValueError('Adopted GPU is not authorized for this supervisor')
        added_resource_root=(args.directory/'.resource_locks').resolve()
        if args.added_resource_dir is not None:
            if handover is None:raise ValueError('Explicit resource-domain reuse requires live adoption')
            previous_manifest=json.loads((args.adopt_from/'manifest.json').read_text())
            previous_root=previous_manifest.get('live_adoption',{}).get('added_job_resource_root')
            if previous_root is None or args.added_resource_dir.resolve()!=Path(previous_root).resolve():
                raise ValueError('Resource domain must match the verified previous supervisor')
            added_resource_root=args.added_resource_dir.resolve()
        settings=dict(protocol=str(args.protocol.resolve()),protocol_fingerprint=protocol['fingerprint'],seed=cfg.seed,
            suite=args.suite,
            training=asdict(cfg),sources=sources,gpus=args.gpus,initial_jobs=args.initial_jobs,
            max_jobs=args.max_jobs,jobs_per_gpu=args.jobs_per_gpu,ramp_after=args.ramp_after,
            ramp_interval=args.ramp_interval,minimum_gpu_free_gib=args.gpu_free_gib,oracle_cache_chunks=args.cache_chunks,
            system_available_reserve_gib=100.,wandb_mode='offline',launcher_sha256=file_sha256(__file__),
            tasks=[{k:v for k,v in t.items() if k!='status'} for t in rows],
            execution_limits=dict(mechanism_workers=args.mechanism_workers,
                mechanism_concurrency=args.mechanism_concurrency,ordinary_scoring_concurrency=2,
                io_concurrency=args.io_concurrency,worker_threads=1),ramp_step=args.ramp_step,
            handover_helper_sha256=file_sha256(handovers.__file__),admit_all_pending=args.admit_all_pending)
        if handover is not None:
            settings['live_adoption']=dict(directory=str(args.adopt_from.resolve()),
                receipt_sha256=file_sha256(args.adopt_from/'handover.json'),
                added_job_resource_root=str(added_resource_root),
                inherited_jobs_keep_original_limits=True)
        if retry_source is not None:settings['rerun_task_ledger']=retry_source
        if manifest.exists() and json.loads(manifest.read_text())!=settings:raise ValueError('Queue configuration/source changed; use a fresh queue directory')
        atomic_json(manifest,settings)
        status_path=args.directory/'tasks.json'
        if handover is not None:
            saved_rows={t['task_id']:t for t in handover['tasks']}
            for row in rows:
                old=saved_rows[row['task_id']]
                if old['status'] in ('complete','running','failed'):
                    row.update(old)
                    if old['status']=='running':row.update(adopted=True,exit_code_source='verified_completion_after_adoption')
        elif status_path.exists():
            old={t['task_id']:t for t in json.loads(status_path.read_text())}
            for row in rows:
                if row['task_id'] in old:
                    saved=old[row['task_id']]
                    if saved['status']=='complete':row.update(saved)
                    elif saved['status']=='running':row['previous_pid']=saved.get('pid')
        lhs_spec=verified_spec_bundle(protocol,args.directory)
        for relative,digest in protocol['artifacts'].items():
            if file_sha256(args.protocol/relative)!=digest:raise ValueError('Shared protocol changed')
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline',
            PYTHONUNBUFFERED='1',SAMPLING_MAX_TOTAL_JOBS=str(args.max_jobs),SAMPLING_IO_CONCURRENCY=str(args.io_concurrency),
            SAMPLING_SCORING_CONCURRENCY='2',SAMPLING_MECHANISM_CONCURRENCY=str(args.mechanism_concurrency),
            SAMPLING_MECHANISM_WORKERS=str(args.mechanism_workers),SAMPLING_MIN_AVAILABLE_GIB='100',SAMPLING_MEMORY_PRESSURE_POLICY='available',
            SAMPLING_IO_GUARD_POLICY='disk_correlated',SAMPLING_IO_ADMISSION='warn',SAMPLING_PLOT_WORKERS='2',
            SAMPLING_ORACLE_CACHE_CHUNKS=str(args.cache_chunks),PYTHONPATH=str(ROOT.parent))
        if handover is not None:
            # Inherited jobs retain their old CPU/IO domain. Added jobs have a
            # separate domain so old 2-slot scorers cannot starve behind new
            # 5-slot scorers. Parent admission/guard still covers ALL jobs.
            env['SAMPLING_RESOURCE_DIR']=str(added_resource_root)
        # Parent guard uses the exact same remaining-memory and IO policies.
        for key in ('SAMPLING_MIN_AVAILABLE_GIB','SAMPLING_MEMORY_PRESSURE_POLICY','SAMPLING_IO_GUARD_POLICY',
                    'SAMPLING_IO_ADMISSION'):os.environ[key]=env[key]
        (args.directory/'logs').mkdir(exist_ok=True)
        active={};started=time.monotonic();capacity=args.initial_jobs;last_ramp=started;last_notice=0.
        with BatchMemoryGuard(args.directory/'resources.jsonl') as guard:
            if handover is not None:
                for task in rows:
                    if task['status']!='running':continue
                    child=handovers.AdoptedProcess(task,handover['processes'][task['task_id']],validate_result)
                    handle=Path(task['launcher_log']).open('a')
                    tracking=guard.track(child);tracking.__enter__()
                    active[child.pid]=(child,task,handle,tracking)
                    print(f"ADOPT {task['task_id']} GPU={task['gpu']} pid={child.pid} (no worker restart)",flush=True)
            while any(t['status'] in ('pending','running') for t in rows):
                now=time.monotonic()
                for pid,(child,task,handle,tracking) in list(active.items()):
                    result=find_result(task,protocol['protocol_id'],task['started_wall_time'])
                    if result:task['result_directory']=result
                    code=child.poll()
                    if code is not None:
                        tracking.__exit__(None,None,None);handle.close();active.pop(pid)
                        valid=code==0
                        if valid:
                            try:task['verified_outputs']=validate_result(task)
                            except (OSError,ValueError,KeyError) as exc:valid=False;task['failure_reason']=str(exc)
                        task.update(status='complete' if valid else 'failed',exit_code=code,
                            seconds=time.time()-task['started_wall_time'],finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
                        print(f"END {task['task_id']} exit={code} result={task.get('result_directory')}",flush=True)
                if guard.aborted:
                    for t in rows:
                        if t['status']=='running':t['status']='guard_stopped'
                    atomic_json(status_path,rows);raise RuntimeError(guard.reason)
                pending=next_pending(rows,[t for _,t,_,_ in active.values()],
                    args.max_jobs if args.admit_all_pending else args.mechanism_concurrency)
                blocked,health=resource_state()
                free=gpu_memory() if pending is not None else {}
                if now-started>=args.ramp_after and now-last_ramp>=args.ramp_interval and capacity<args.max_jobs:
                    # Ramp only after real training has started in at least two
                    # current jobs, with ample system headroom. No owned-PSS cap.
                    ready=sum('[Epoch ' in Path(t['launcher_log']).read_text(errors='replace')[-20000:]
                              for _,t,_,_ in active.values())
                    if ready>=min(2,len(active)) and not blocked and health['available_gib']>=200:
                        capacity=min(args.max_jobs,capacity+args.ramp_step);last_ramp=now;print(f'RAMP capacity={capacity} available={health["available_gib"]:.1f}GiB',flush=True)
                counts={g:sum(t['gpu']==g for _,t,_,_ in active.values()) for g in args.gpus}
                usable=[g for g in args.gpus if counts[g]<args.jobs_per_gpu and free.get(g,0)>=args.gpu_free_gib]
                if pending and len(active)<capacity and usable and not blocked and health['available_gib']>=124:
                    if source_inventory()!=sources:raise RuntimeError('Source changed while queue is active; refuse mixed-version jobs')
                    if shutil.disk_usage(DATA).free<50*2**30:raise RuntimeError('Less than 50GiB available on data volume')
                    gpu=min(usable,key=lambda g:(counts[g],-free[g]))
                    log=args.directory/'logs'/f"{pending['task_id']}.log";handle=log.open('a')
                    cmd=command(pending,args.protocol,lhs_spec)
                    if shutil.which('ionice'):cmd=['ionice','-c','2','-n','7',*cmd]
                    if shutil.which('nice'):cmd=['nice','-n','10',*cmd]
                    child=subprocess.Popen(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                    tracking=guard.track(child);tracking.__enter__()
                    pending.update(status='running',pid=child.pid,gpu=gpu,started_wall_time=time.time(),
                        started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),launcher_log=str(log),command=cmd)
                    active[child.pid]=(child,pending,handle,tracking)
                    print(f"START {pending['task_id']} GPU={gpu} pid={child.pid}",flush=True)
                atomic_json(status_path,rows)
                atomic_json(args.directory/'status.json',dict(active=len(active),capacity=capacity,
                    pending=sum(t['status']=='pending' for t in rows),completed=sum(t['status']=='complete' for t in rows),
                    failed=sum(t['status']=='failed' for t in rows),available_gib=health['available_gib'],
                    seconds=now-started,gpu_free_gib=free,queue_pid=os.getpid(),memory_guard_aborted=guard.aborted))
                if now-last_notice>60:
                    print(f"QUEUE active={len(active)} pending={sum(t['status']=='pending' for t in rows)} available={health['available_gib']:.1f}GiB",flush=True);last_notice=now
                time.sleep(5)
        failures=[t['task_id'] for t in rows if t['status']=='failed']
        print(json.dumps(dict(status='complete' if not failures else 'completed_with_failures',failures=failures)),flush=True)
        return int(bool(failures))


if __name__=='__main__':raise SystemExit(main())
