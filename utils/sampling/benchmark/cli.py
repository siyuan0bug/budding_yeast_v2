"""Prepare, inspect and run complete-pool sampling benchmarks (no online simulation)."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import shutil
import sys
import time

from ..registry import METHODS, method_spec
from .protocol import BenchmarkConfig, prepare_protocol
from budding_yeast_v2.data.dataset_archive import atomic_json
from budding_yeast_v2.utils.experiments.logging import install_timestamped_streams, log_event
from budding_yeast_v2.utils.experiments.resources import wait_for_resources, resource_slot
from budding_yeast_v2.utils.experiments.memory_guard import BatchMemoryGuard


DATA = Path('/home/users/zsy/mydata/budding_yeast')
DEFAULT_METHODS = ('random','topk','power','max_dist_pred','coreset_ll','lcmd_ll','bait_ll','vessal_penultimate')


def _shared(parser):
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SEED', 42)))
    parser.add_argument('--model', help='Registered model name; defaults to the shared config (fno)')
    parser.add_argument('--mutation-pool', type=Path, default=DATA / 'offline_pools/mutation_100000_seed42_numerical_qc/dataset.json')
    parser.add_argument('--lhs-pool', type=Path, default=DATA / 'offline_pools/lhs_matched_100000_seed42_numerical_qc/dataset.json')
    parser.add_argument('--protocol-root', type=Path, default=DATA / 'experiments/budding_sampling/shared/pool100k')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--config', type=Path, help='JSON overrides for BenchmarkConfig; not per-method tuning')
    parser.add_argument('--real-reference', type=Path)
    parser.add_argument('--reference-workers', type=int, default=8)


def _prepare(args, with_real=True):
    if args.protocol:
        root = args.protocol if args.protocol.is_dir() else args.protocol.parent
        value = json.loads((root / 'protocol.json').read_text())
        if value['config']['seed'] != args.seed:
            raise ValueError('Requested experiment seed disagrees with stored protocol')
        if args.config is not None or (args.model is not None and value['config'].get('model','fno') != args.model):
            from .protocol import derive_training_protocol
            configuration=dict(value['config'])
            if args.config is not None:configuration.update(json.loads(args.config.read_text()))
            if args.model is not None:configuration['model']=args.model
            configuration['seed']=args.seed
            root=derive_training_protocol(root,args.protocol_root,BenchmarkConfig(**configuration).validate())
    else:
        overrides = {} if args.config is None else json.loads(args.config.read_text())
        overrides['seed'] = args.seed
        if args.model is not None:
            overrides['model'] = args.model
        config = BenchmarkConfig(**overrides).validate()
        if os.environ.get('SKIP_PREPARE') == '1':
            from .protocol import protocol_location
            root = protocol_location(dict(mutation=args.mutation_pool, lhs=args.lhs_pool), args.protocol_root, config)
            if not (root / 'protocol.json').exists():
                raise FileNotFoundError(f'SKIP_PREPARE requested but protocol is missing: {root}')
        else:
            root = prepare_protocol(dict(mutation=args.mutation_pool, lhs=args.lhs_pool), args.protocol_root, config)
    reference = args.real_reference
    if reference is None and with_real:
        stored = json.loads((root / 'protocol.json').read_text()).get('real_reference')
        if stored:
            from budding_yeast_v2.data.dataset_archive import file_sha256
            reference = Path(stored['path'])
            if file_sha256(reference) != stored['sha256']:
                raise ValueError('Frozen real evaluation reference changed')
        else:
            from .references import prepare_real_reference
            reference = prepare_real_reference(args.protocol_root / 'references', args.reference_workers)
    return root, reference


def _batch(args):
    names = args.methods or DEFAULT_METHODS
    names = tuple(names)
    invalid = [name for name in names if method_spec(name).status != 'implemented']
    if invalid:
        raise NotImplementedError({name: method_spec(name).note for name in invalid})
    if any(method_spec(name).mode=='dynamic' for name in names):
        raise ValueError('Dynamic methods use the explicit dynamic CLI; profile resources before batch launch')
    gpus = args.gpus
    if not gpus or args.jobs_per_gpu < 1:
        raise ValueError('Supply usable GPU IDs and a positive per-GPU concurrency')
    if len(set(gpus)) != len(gpus):
        raise ValueError('GPU IDs must not be repeated')
    # GPUs 4--7 are exclusive cards, regardless of jobs-per-gpu.
    gpu_slots = [gpu for repetition in range(args.jobs_per_gpu) for gpu in gpus
                 if repetition == 0 or gpu not in ('4', '5', '6', '7')]
    tasks = [(pool, name) for pool in args.pools for name in names
             if not METHODS[name].mutation_only or pool == 'mutation']
    maximum = int(os.environ.get('SAMPLING_MAX_TOTAL_JOBS', '16'))
    if maximum < 1:
        raise ValueError('SAMPLING_MAX_TOTAL_JOBS must be positive')
    slots = min(len(gpu_slots), maximum)
    # DRY_RUN must not create protocols, labels, result directories or checkpoints.
    if args.dry_run:
        for i, (pool, name) in enumerate(tasks):
            print(json.dumps(dict(pool=pool, method=name, track=METHODS[name].track,
                seed=args.seed, gpu=gpu_slots[i % slots],
                jobs_per_gpu=1 if gpu_slots[i % slots] in ('4','5','6','7') else args.jobs_per_gpu,
                max_total_jobs=maximum)))
        return
    with resource_slot('data', 'shared protocol preparation'):
        root, reference = _prepare(args)
    protocol = json.loads((root / 'protocol.json').read_text())
    from .runner import experiment_lock
    batch_dir = args.output / 'batches' / f'seed_{args.seed}'
    with experiment_lock(batch_dir):
        # Free GPU slots pull the next task instead of waiting behind a slow
        # experiment in a permanently assigned queue. Admission limits remain.
        pending = Queue()
        for task in tasks:
            pending.put(task)
        status = []

        def worker(slot, guard):
            gpu = gpu_slots[slot]
            rows = []
            while True:
                if guard.aborted:break
                try:
                    pool, name = pending.get_nowait()
                except Empty:
                    break
                from ..layout import experiment_directory
                destination = experiment_directory(args.output,protocol['config'],pool,name)
                destination.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, '-m', 'budding_yeast_v2.utils.sampling.benchmark.cli', 'run',
                       '--protocol', str(root), '--seed', str(args.seed), '--pool', pool, '--method', name,
                       '--output', str(args.output), '--device', 'cuda:0', '--real-reference', str(reference)]
                if args.plots:
                    cmd.append('--plots')
                if args.wandb:
                    cmd.append('--wandb')
                if shutil.which('ionice'):
                    cmd = ['ionice', '-c', '2', '-n', '7', *cmd]
                if shutil.which('nice'):
                    cmd = ['nice', '-n', '10', *cmd]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WANDB_MODE='offline',
                           PYTHONUNBUFFERED='1', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '2'),
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS', '2'))
                started = time.monotonic()
                try:
                    wait_for_resources('batch task admission',cancel=lambda:guard.aborted)
                except RuntimeError:
                    if guard.aborted:break
                    raise
                if guard.aborted:break
                log_event('START', 'sampling batch task', gpu=gpu, pool=pool, method=name)
                with (destination / 'train.log').open('a') as log:
                    child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,start_new_session=True)
                    with guard.track(child):
                        code = child.wait()
                row = dict(pool=pool, method=name, gpu=gpu, exit_code=code,
                           seconds=time.monotonic() - started, directory=str(destination))
                rows.append(row)
                log_event('END' if code == 0 else 'FAIL', 'sampling batch task', **row)
                if guard.aborted or (code and os.environ.get('CONTINUE_ON_ERROR', '1') == '0'):
                    break
            return rows

        with BatchMemoryGuard(batch_dir/'resource_samples.jsonl') as guard:
            with ThreadPoolExecutor(max_workers=slots) as executor:
                futures = [executor.submit(worker, slot, guard) for slot in range(slots)]
                try:
                    for future in as_completed(futures):
                        status.extend(future.result())
                        atomic_json(batch_dir / 'tasks.json', status)
                except BaseException as exc:
                    guard.stop(f'Batch interrupted: {exc}')
                    raise
        if guard.aborted:raise RuntimeError(guard.reason)
        if any(row['exit_code'] for row in status):
            raise SystemExit(1)


def main(argv=None):
    install_timestamped_streams()
    os.environ['WANDB_MODE'] = 'offline'
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    from ..generation.cli import add_parser as add_generation_parser
    add_generation_parser(sub)
    from ..dynamic.cli import add_parser as add_dynamic_parser
    add_dynamic_parser(sub)
    sub.add_parser('list', help='Implemented methods and explicit adaptation blockers')
    prepare = sub.add_parser('prepare', help='Shared splits/input caches/reference only; no training')
    _shared(prepare)
    prepare.add_argument('--no-real', action='store_true')
    run = sub.add_parser('run', help='One complete sampling experiment, resumable')
    _shared(run)
    run.add_argument('--pool', choices=('mutation', 'lhs'), required=True)
    run.add_argument('--method', choices=tuple(METHODS), required=True)
    run.add_argument('--output', type=Path, default=DATA / 'experiments/budding_sampling')
    run.add_argument('--device', default='cpu')
    run.add_argument('--plots', action='store_true')
    run.add_argument('--wandb', action='store_true')
    run.add_argument('--diagnostic-only', action='store_true', help='No-real test run; marked non-benchmark')
    batch = sub.add_parser('batch', help='Both pools, shared GPU task queue and initialization')
    _shared(batch)
    batch.add_argument('--output', type=Path, default=DATA / 'experiments/budding_sampling')
    batch.add_argument('--pools', nargs='+', choices=('mutation', 'lhs'), default=('mutation', 'lhs'))
    batch.add_argument('--methods', nargs='+', choices=tuple(METHODS))
    batch.add_argument('--gpus', nargs='+', default=os.environ.get('GPU_IDS', '0 1 2 3').split())
    batch.add_argument('--jobs-per-gpu', type=int, default=int(os.environ.get('MAX_JOBS_PER_GPU', 4)))
    batch.add_argument('--dry-run', action='store_true', default=os.environ.get('DRY_RUN') == '1')
    batch.add_argument('--plots', action=argparse.BooleanOptionalAction, default=True)
    batch.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.command == 'generate':
        from ..generation.cli import execute
        execute(args)
    elif args.command == 'dynamic':
        from ..dynamic.cli import execute
        execute(args)
    elif args.command == 'list':
        for spec in METHODS.values():
            print(json.dumps(asdict(spec), ensure_ascii=False))
        from ..generation.policies import METHODS as GENERATION_METHODS
        for name in GENERATION_METHODS:
            print(json.dumps(dict(name=name,family='ours' if name.startswith('ours_') else 'baseline',mode='generation',command='generate',
                status='implemented_task_adaptation',native_online_ogas=False)))
        from ..generation.online import ONLINE_METHODS
        for name in ONLINE_METHODS:
            print(json.dumps(dict(name=name,family='baseline',mode='online_generation',command='generate',
                status='implemented_task_adaptation',asynchronous_solver_training=True,
                native_melissa_pde_reproduction=False)))
    elif args.command == 'prepare':
        root, reference = _prepare(args, not args.no_real)
        log_event('COMPLETE', 'shared benchmark preparation', protocol=root, real_reference=reference)
    elif args.command == 'run':
        root, reference = _prepare(args, not args.diagnostic_only)
        from .runner import run_experiment
        run_experiment(root, args.pool, args.method, args.output, device=args.device,
                       real_reference=reference, plots=args.plots, wandb_enabled=args.wandb,
                       diagnostic_only=args.diagnostic_only)
    else:
        os.environ.setdefault('SAMPLING_IO_GUARD_POLICY','disk_correlated')
        os.environ.setdefault('SAMPLING_MEMORY_PRESSURE_POLICY','available')
        os.environ.setdefault('SAMPLING_SCORING_CONCURRENCY','4')
        os.environ.setdefault('SAMPLING_IO_ADMISSION',
            'warn' if os.environ['SAMPLING_IO_GUARD_POLICY']=='disk_correlated' else 'strict')
        _batch(args)


if __name__ == '__main__':
    main()
