#!/usr/bin/env python3
"""Gradually admit full-size resource probes, then overlap training/evaluation.

Uses real production loading, normalization, FNO and evaluation. Each worker is
diagnostic-only (two epochs, no acquisition/accuracy comparison). No trajectory
copies are stored. The parent samples its owned process trees and terminates
only those workers if memory limits are exceeded.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT.parent))
from budding_yeast_v2.scripts.analysis.profile_sampling_resources import counters,write_json
from budding_yeast_v2.utils.experiments.resources import memory_status,resource_state,disk_counters
from budding_yeast_v2.utils.experiments.memory_guard import PressureWatch


def stop_owned(children):
    """Only Popen objects explicitly launched in a new process group are eligible."""
    active=[child for child in children if child.poll() is None]
    for child in active:
        try:
            if os.getpgid(child.pid)!=child.pid:
                raise RuntimeError('Refusing to signal an unexpected process group')
            os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass  # A worker may finish between poll() and group lookup.
    deadline=time.monotonic()+10
    for child in active:
        try:child.wait(max(.1,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()


def log(message):
    print(f'[{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}] {message}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--real-reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',nargs='+',default=['2','3'])
    parser.add_argument('--stages',nargs='+',type=int,default=[2,4,6])
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--selectors',action='store_true',help='Also exercise 90k LCMD/BAIT/ensemble scoring in each worker')
    parser.add_argument('--io-concurrency',type=int,choices=range(1,17),default=2)
    parser.add_argument('--scoring-concurrency',type=int,choices=range(1,17),
        help='Concurrent candidate-scoring processes; defaults to worker count')
    parser.add_argument('--loading-only',action='store_true',help='Measure preprocessing only; stop workers before training')
    parser.add_argument('--pressure-grace-seconds',type=float,default=20.)
    parser.add_argument('--io-policy',choices=('strict','disk_correlated'),default='strict',
        help='Diagnostic only: correlate IO PSI with measured device congestion')
    parser.add_argument('--memory-policy',choices=('strict','available'),default='strict',
        help='Use effective available memory as the limit; PSI stays diagnostic')
    parser.add_argument('--disk-pressure-grace-seconds',type=float,default=60.)
    parser.add_argument('--max-pss-gib',type=float,help='Optional diagnostic cap; default uses available memory only')
    parser.add_argument('--minimum-available-gib',type=float,default=100.)
    parser.add_argument('--timeout',type=float,default=2700.)
    args=parser.parse_args()
    if args.pressure_grace_seconds<=0 or args.disk_pressure_grace_seconds<=0:
        parser.error('Pressure grace period must be positive')
    if sorted(set(args.stages))!=args.stages or args.stages[0]<1 or args.stages[-1]>16:
        parser.error('Use increasing stages, maximum sixteen workers for the current batch')
    if any(g not in ('0','1','2','3') for g in args.gpus):
        parser.error('This concurrent probe uses only explicitly authorized nonexclusive GPUs 0--3')
    # Parent admission and worker preprocessing must use the same reserve.
    os.environ['SAMPLING_MIN_AVAILABLE_GIB']=str(args.minimum_available_gib)
    # Only this bounded diagnostic delegates IO admission to its device monitor.
    os.environ['SAMPLING_IO_ADMISSION']='warn' if args.io_policy=='disk_correlated' else 'strict'
    os.environ['SAMPLING_MEMORY_PRESSURE_POLICY']=args.memory_policy
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'config.json').exists():raise ValueError('Use a fresh diagnostic output directory')
    identity=dict(stages=args.stages,gpus=args.gpus,train_rows=10000,epochs=args.epochs,
        max_pss_gib=args.max_pss_gib,minimum_available_gib=args.minimum_available_gib,
        io_concurrency=args.io_concurrency,scoring_concurrency=args.scoring_concurrency or args.stages[-1],
        diagnostic_only=True,source_protocol=str(args.protocol),
        selectors=args.selectors,loading_only=args.loading_only,io_policy=args.io_policy,
        memory_policy=args.memory_policy,
        pressure_grace_seconds=args.pressure_grace_seconds,
        disk_pressure_grace_seconds=args.disk_pressure_grace_seconds,
        diagnostic_disk_limits=dict(busy_percent=90,queue=16,await_ms=250))
    write_json(args.output/'config.json',identity)
    children=[];handles=[];jobs=[];snapshots=[];peak_pss=0.;min_available=float('inf')
    release=args.output/'release_training'
    began=time.monotonic();last_log=0.;failure=None;peak_active=0
    pressure_watch=PressureWatch(io_policy=args.io_policy,memory_policy=args.memory_policy,
        grace=args.pressure_grace_seconds,
        disk_grace=args.disk_pressure_grace_seconds,
        io_limit=float(os.environ.get('SAMPLING_MAX_IO_PSI','5')),
        memory_limit=float(os.environ.get('SAMPLING_MAX_MEMORY_PSI','1')))
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        WANDB_MODE='offline',PYTHONUNBUFFERED='1',SAMPLING_MAX_TOTAL_JOBS=str(args.stages[-1]),
        SAMPLING_IO_CONCURRENCY=str(args.io_concurrency),SAMPLING_MIN_AVAILABLE_GIB=str(args.minimum_available_gib),
        SAMPLING_SCORING_CONCURRENCY=str(args.scoring_concurrency or args.stages[-1]),
        SAMPLING_PROBE_SUPERVISOR_PID=str(os.getpid()),
        SAMPLING_PLOT_WORKERS='1')

    def sample(stream):
        nonlocal peak_pss,min_available,last_log,peak_active
        phases={};per_job={}
        for item,child in zip(jobs,children):
            if child.poll() is None:per_job[str(child.pid)]=counters(child.pid)
            p=Path(item['directory'])/'phase.json'
            if p.exists():phases[str(child.pid)]=json.loads(p.read_text())['phase']
            if child.poll() not in (None,0):raise RuntimeError(f'Worker failed: {item}')
        available=memory_status()[1]/2**30
        pss=sum(v['pss_gib'] for v in per_job.values())
        peak_pss=max(peak_pss,pss);min_available=min(min_available,available)
        peak_active=max(peak_active,len(per_job))
        blocked,health=resource_state()
        now=time.monotonic();disks=disk_counters()
        pressure=pressure_watch.update(now,health,disks,active=bool(per_job))
        row=dict(seconds=time.monotonic()-began,wall_time=time.time(),pss_gib=pss,available_gib=available,
            jobs=per_job,phases=phases,health=health,diskstats=disks,**pressure,admission_blocked=blocked)
        stream.write(json.dumps(row)+'\n');stream.flush()
        if time.monotonic()-last_log>=30:
            log(json.dumps({k:row[k] for k in ['seconds','pss_gib','available_gib','phases']}));last_log=time.monotonic()
        if ((args.max_pss_gib is not None and pss>args.max_pss_gib)
                or available<args.minimum_available_gib):
            raise RuntimeError(f'Memory guard: owned PSS={pss:.2f}GiB available={available:.2f}GiB')
        if pressure['sustained_pressure']:
            raise RuntimeError(f'Sustained host pressure; stopping owned probes: {health}; disk_congestion={pressure["severe_disk_congestion"]}')
        if time.monotonic()-began>args.timeout:raise TimeoutError('Bounded concurrency probe exceeded its time limit')
        return row

    try:
        with (args.output/'samples.jsonl').open('w') as stream:
            for target in args.stages:
                while len(children)<target:
                    row=sample(stream)
                    if row['admission_blocked']:
                        time.sleep(2);continue
                    index=len(children);gpu=args.gpus[index%len(args.gpus)]
                    folder=args.output/f'worker_{index:02d}'
                    folder.mkdir()
                    pool='mutation' if index%2==0 else 'lhs'
                    command=[sys.executable,str(PROJECT/'scripts/analysis/profile_sampling_resources.py'),
                        '--worker','--protocol',str(args.protocol),'--output',str(folder),'--pool',pool,
                        '--gpu',gpu,'--threads','2','--epochs',str(args.epochs),
                        '--train-size','10000','--real-reference',str(args.real_reference),
                        '--release-file',str(release)]
                    if args.selectors:command.append('--selectors')
                    if shutil.which('ionice'):command=['ionice','-c','2','-n','7',*command]
                    if shutil.which('nice'):command=['nice','-n','10',*command]
                    handle=(folder/'worker.log').open('w');handles.append(handle)
                    child=subprocess.Popen(command,cwd=PROJECT,stdout=handle,stderr=subprocess.STDOUT,
                        env=dict(env,CUDA_VISIBLE_DEVICES=gpu),start_new_session=True)
                    children.append(child);jobs.append(dict(pid=child.pid,gpu=gpu,pool=pool,directory=str(folder)))
                    log(f'Admitted worker {index}, GPU {gpu}, pool {pool}; stage target={target}')
                    time.sleep(2)
                # Full-size preprocessing must finish before we raise concurrency.
                while True:
                    row=sample(stream)
                    if all(row['phases'].get(str(c.pid))=='ready for concurrent training' for c in children):break
                    time.sleep(2)
                log(f'{target} workers ready after full preprocessing; observing a ten-second plateau')
                plateau=[]
                for _ in range(5):
                    time.sleep(2);plateau.append(sample(stream))
                snapshots.append(dict(workers=target,prepared_pss_gib=max(r['pss_gib'] for r in plateau),
                    minimum_available_gib=min(r['available_gib'] for r in plateau)))
                write_json(args.output/'prepared_stages.json',snapshots)
            loading_seconds=time.monotonic()-began
            if args.loading_only:
                log('Loading-only measurement completed; stopping workers before training')
                stop_owned(children)
            else:
                release.touch();log('Released all workers for concurrent full-size training/evaluation')
                while any(c.poll() is None for c in children):
                    sample(stream);time.sleep(2)
                sample(stream)
    except BaseException as exc:
        failure=str(exc);log(f'Aborting owned resource probes: {failure}')
        stop_owned(children)
    finally:
        for handle in handles:handle.close()
    workers=[]
    for job,child in zip(jobs,children):
        result=Path(job['directory'])/'worker_result.json'
        workers.append(dict(job,exit_code=child.returncode,
            result=json.loads(result.read_text()) if result.exists() else None))
    report=dict(status='complete' if failure is None else 'failed',failure=failure,
        seconds=time.monotonic()-began,peak_owned_pss_gib=peak_pss,
        minimum_available_gib=min_available,peak_active_workers=peak_active,
        prepared_stages=snapshots,workers=workers,identity=identity,
        loading_seconds=locals().get('loading_seconds'),
        completed_resource_jobs_per_hour=(len(workers)*3600/(time.monotonic()-began)
            if failure is None and not args.loading_only else None),
        limitations=f'Resource-only {args.epochs}-epoch runs; no convergence/accuracy ranking. Warm filesystem cache; '
        f'{args.io_concurrency} loaders remain limited by project locks. '
        + ('Includes 90k LCMD/BAIT/ensemble scoring, but not auxiliary member training or complete AL rounds.'
           if args.selectors else 'No 90k acquisition or ensemble auxiliary training.'))
    write_json(args.output/'profile.json',report)
    log(json.dumps(report))
    if failure:raise SystemExit(1)


if __name__=='__main__':main()
