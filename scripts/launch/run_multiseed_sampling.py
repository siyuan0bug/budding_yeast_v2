#!/usr/bin/env python3
"""One bounded multi-seed supervisor; existing complete pools are never regenerated."""
import argparse,json,os,subprocess,sys,time
from dataclasses import replace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.dataset_archive import atomic_json
from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig,prepare_protocol,child_seed
from budding_yeast_v2.utils.sampling.benchmark.runner import source_inventory
from budding_yeast_v2.utils.sampling.layout import RESULTS,SHARED,experiment_directory

METHODS=['power','random','lcmd_ll','physics_pre_power','low_rank','peaks_regression','ogas_async_projected']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seeds',nargs='+',type=int,default=[123,1024])
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--io-concurrency',type=int,default=2)
    p.add_argument('--adopt-first-queue',type=Path)
    p.add_argument('--gpu',choices=['0'],default='0');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();assert len(set(args.seeds))==len(args.seeds) and 1<=args.jobs<=4 and 1<=args.io_concurrency<=4
    definition=json.loads((SHARED/'seed_42/protocol.json').read_text());pools={k:v['path'] for k,v in definition['pools'].items()}
    rows=[dict(seed=seed,pool=pool,method=method,directory=str(experiment_directory(RESULTS,
        dict(definition['config'],seed=seed),pool,method,'generate' if method=='ogas_async_projected' else 'run')))
        for seed in args.seeds for method in METHODS for pool in ('mutation','lhs')]
    manifest=dict(seeds=args.seeds,methods=METHODS,tasks=rows,count=len(rows),gpu='0',max_experiments=args.jobs,
        io_concurrency=args.io_concurrency,adopt_first_queue=str(args.adopt_first_queue) if args.adopt_first_queue else None,ensemble_execution='independent_spawn_parallel',generation_seed=42,
        sources=source_inventory(),configuration=definition['config'])
    if args.dry_run:print(json.dumps(manifest,indent=2));return
    args.directory.mkdir(parents=True,exist_ok=True)
    old=args.directory/'manifest.json'
    if old.exists() and json.loads(old.read_text())!=manifest:raise ValueError('Batch configuration/source changed')
    atomic_json(old,manifest)
    atomic_json(args.directory/'status.json',dict(phase='preparing_shared',seeds=args.seeds))
    for seed in args.seeds:
        cfg=replace(BenchmarkConfig(**definition['config']),seed=seed)
        root=prepare_protocol(pools,SHARED,cfg);value=json.loads((root/'protocol.json').read_text())
        if value.get('real_reference') not in (None,definition['real_reference']):raise ValueError('Real reference mismatch')
        value['real_reference']=definition['real_reference'];atomic_json(root/'protocol.json',value)
        atomic_json(root/'seed_manifest.json',dict(generation_seed=42,experiment_seed=seed,
            split=child_seed(seed,'split'),initial={name:child_seed(seed,'initial',i) for i,name in enumerate(sorted(pools))},
            model=[[child_seed(seed,'model',r,m) for m in range(cfg.ensemble_members)] for r in range(len(cfg.epochs))],
            selection=[child_seed(seed,'selection',r) for r in range(len(cfg.epochs))]))
    env=dict(os.environ,SAMPLING_ENSEMBLE_PARALLEL='1',SAMPLING_RESOURCE_DIR=str(args.directory/'locks'),
        WANDB_MODE='offline',SAMPLING_PLOT_WORKERS='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    failures=[]
    for seed in args.seeds:
        if source_inventory()!=manifest['sources']:raise RuntimeError('Sources changed after batch freeze')
        command=['bash',str(ROOT/'run_sampling_budding.sh'),'extensions','--suite','all','--methods',*METHODS,
            '--seed',str(seed),'--protocol',str(SHARED/f'seed_{seed}'),'--directory',str(args.directory/f'seed_{seed}'),
            '--gpus','0','--initial-jobs',str(args.jobs),'--max-jobs',str(args.jobs),'--jobs-per-gpu',str(args.jobs),
            '--io-concurrency',str(1 if args.adopt_first_queue and seed==args.seeds[0] else args.io_concurrency),'--mechanism-workers','2','--mechanism-concurrency','1','--cache-chunks','128']
        if args.adopt_first_queue and seed==args.seeds[0]:
            command+=['--adopt-from',str(args.adopt_first_queue)]
        atomic_json(args.directory/'status.json',dict(phase='running',seed=seed,queue=str(args.directory/f'seed_{seed}')))
        with (args.directory/f'seed_{seed}_supervisor.log').open('a') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(args.directory/f'seed_{seed}_exit.json',dict(exit_code=result.returncode))
        if result.returncode:failures.append(seed)
    atomic_json(args.directory/'status.json',dict(phase='complete' if not failures else 'completed_with_failures',seeds=args.seeds,failed_seed_queues=failures))


if __name__=='__main__':main()
