#!/usr/bin/env python3
"""One bounded production-size sampling resource probe, never a benchmark result."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT.parent))
from budding_yeast_v2.utils.experiments.resources import process_tree, process_counters as counters, resource_slot


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def arm_resource_supervisor():
    # Linux workers must not survive a vanished resource watchdog, including
    # while waiting at the preprocessing barrier after a disconnected terminal.
    supervisor = os.environ.get('SAMPLING_PROBE_SUPERVISOR_PID')
    if supervisor:
        import ctypes
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot arm parent-death signal')
        if os.getppid() != int(supervisor):
            raise RuntimeError('Resource supervisor exited before worker initialization')


def worker(args):
    arm_resource_supervisor()
    import numpy as np
    import torch
    from budding_yeast_v2.data.dataset_archive import atomic_json, file_sha256, fingerprint
    from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig
    from budding_yeast_v2.utils.sampling.benchmark import runner, rounds
    from budding_yeast_v2.utils.sampling import query
    from budding_yeast_v2.utils.experiments.logging import install_timestamped_streams
    install_timestamped_streams()
    torch.set_num_threads(args.threads)
    root = args.output / 'protocol';root.mkdir(exist_ok=True)
    source = json.loads((args.protocol / 'protocol.json').read_text())
    config = replace(BenchmarkConfig(**source['config']), budgets=(args.train_size,),
                     epochs=(args.epochs,), warmup=(0,))
    artifacts = {}
    for name in source['pools']:
        cache = root / f'{name}_inputs.npz'
        if not cache.exists():
            cache.symlink_to((args.protocol / cache.name).resolve())
        with np.load(args.protocol / f'{name}_split.npz') as original:
            split = {key: original[key] for key in original.files}
        split['initial'] = split['random_order'][:args.train_size]
        path = root / f'{name}_split.npz';np.savez_compressed(path, **split)
        artifacts[path.name], artifacts[cache.name] = file_sha256(path), file_sha256(cache)
    identity = dict(source_protocol=source['fingerprint'], diagnostic_only=True, config=asdict(config))
    protocol = dict(source, config=asdict(config), artifacts=artifacts,
                    protocol_id=f'resource_probe_n{args.train_size}', fingerprint=fingerprint(identity))
    atomic_json(root / 'protocol.json', protocol)
    phase_file = args.output / 'phase.json'
    current_phase = ['initialization']
    def phase(value):
        current_phase[0] = value
        atomic_json(phase_file, dict(phase=value, time=time.time()))
        print(f'[RESOURCE PHASE] {value}', flush=True)
    old_stage, old_slot = runner.log_stage, runner.resource_slot
    @contextmanager
    def stage(name, **kwargs):
        previous = current_phase[0];phase(name)
        with old_stage(name, **kwargs):
            yield
        phase(previous)
    @contextmanager
    def slot(kind, name):
        previous = current_phase[0];phase(name)
        with old_slot(kind, name):
            yield
        phase(previous)
    runner.log_stage, runner.resource_slot = stage, slot
    rounds.resource_slot = slot
    query.resource_slot = slot
    if args.release_file is not None:
        original_fit = rounds.fit_model_round
        def gated_fit(*values, **kwargs):
            phase('ready for concurrent training')
            while not args.release_file.exists():
                time.sleep(.5)
            phase('concurrent training')
            return original_fit(*values, **kwargs)
        # Data preprocessing has released its IO slot before this barrier.
        rounds.fit_model_round = gated_fit
    phase('single experiment')
    result = runner.run_experiment(root, args.pool, 'random', args.output / 'diagnostic_runs',
        device='cuda:0', real_reference=args.real_reference, plots=args.plots,
        wandb_enabled=True, diagnostic_only=True)
    if args.selectors:
        phase('waiting for candidate scoring slot')
        with resource_slot('scoring', 'resource probe candidate scoring'):
            from budding_yeast_v2.utils.evaluation.checkpoints import Predictor, load_sampling_checkpoint
            from budding_yeast_v2.utils.sampling.contracts import PoolInputs
            from budding_yeast_v2.utils.sampling.baseline.geometry import distance_select
            from budding_yeast_v2.utils.sampling.baseline.bait import bait_select
            from budding_yeast_v2.models.baselines.fno import FNO
            import gc
            phase('reload trained checkpoint for resource-only scoring')
            model, normalization, checkpoint = load_sampling_checkpoint(result / 'rounds/round_00/model.ckpt')
            del checkpoint
            with np.load(root / f'{args.pool}_inputs.npz', allow_pickle=False) as a:
                inputs = {key: a[key] for key in ('params', 'y_init')}
            with np.load(root / f'{args.pool}_split.npz') as a:
                candidate_ids, anchor_ids = a['candidates'], a['initial'][:8000]
            pool = PoolInputs(candidate_ids, inputs['params'][candidate_ids], inputs['y_init'][candidate_ids])
            anchors = PoolInputs(anchor_ids, inputs['params'][anchor_ids], inputs['y_init'][anchor_ids])
            predictor = Predictor([model], normalization, 500, 'cuda:0', config.prediction_batch_size)
            phase('90k candidate LL feature extraction')
            features = predictor.features(pool, config.sketch_dim, latent=True)
            train_features = predictor.features(anchors, config.sketch_dim, latent=True)
            phase('LCMD 90k candidates, 8k anchors, select 2k')
            selected = distance_select(features, train_features, 2000, 'cuda:0', 'lcmd')
            assert len(set(selected)) == 2000
            phase('BAIT 90k candidates, 8k anchors, select 2k')
            selected = bait_select(features, train_features, 2000, 'cuda:0')
            assert len(set(selected)) == 2000
            del features, train_features
            gc.collect();torch.cuda.empty_cache()
            phase('90k candidate two-member uncertainty inference')
            torch.manual_seed(43)
            secondary = FNO(width=config.width, modes=config.modes, n_blocks=config.n_blocks)
            predictor.members = [model, secondary]
            scores = predictor.uncertainty(pool)
            assert np.isfinite(scores).all()
    phase('complete')
    atomic_json(args.output / 'worker_result.json', dict(status='complete', diagnostic_only=True,
        results=str(result), train_size=args.train_size, epochs=args.epochs,
        source_trajectories_copied=False, gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))


def monitor(args):
    from budding_yeast_v2.utils.experiments.resources import memory_status
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS=str(args.threads),
               SAMPLING_PROBE_SUPERVISOR_PID=str(os.getpid()),
               MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
               WANDB_MODE='offline', PYTHONUNBUFFERED='1', SAMPLING_MAX_LOAD_PER_CPU='0')
    # Only this single bounded probe bypasses the load-average admission rule.
    # Memory/IO pressure checks and the configured loader locks remain active.
    command = [sys.executable, str(script), *sys.argv[1:], '--worker']
    with (args.output / 'worker.log').open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, cwd=PROJECT)
        began = time.monotonic();peaks = {};peak_rss = peak_pss = 0.;last = 0.;abort = None
        with (args.output / 'samples.jsonl').open('w') as samples:
            while child.poll() is None:
                row = counters(child.pid)
                phase = 'startup'
                try:
                    phase = json.loads((args.output / 'phase.json').read_text())['phase']
                except (OSError, ValueError):
                    pass
                available = memory_status()[1]/2**30
                row.update(seconds=time.monotonic()-began, phase=phase, available_gib=available)
                samples.write(json.dumps(row)+'\n');samples.flush()
                peak_rss=max(peak_rss,row['rss_gib']);peak_pss=max(peak_pss,row['pss_gib'])
                bucket=peaks.setdefault(phase,dict(rss_gib=0.,pss_gib=0.))
                for key in ('rss_gib','pss_gib'):bucket[key]=max(bucket[key],row[key])
                if time.monotonic()-last>=30:
                    print(json.dumps(row),flush=True);last=time.monotonic()
                if ((args.max_pss_gib is not None and row['pss_gib']>args.max_pss_gib)
                        or available<args.minimum_available_gib):
                    abort=f'probe resource limit: {row}'
                if row['seconds']>args.timeout:
                    abort='probe wall-clock limit'
                if abort:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
                time.sleep(.5)
        child.wait()
    result=dict(status='complete' if child.returncode==0 else 'failed', exit_code=child.returncode,
        aborted=abort,seconds=time.monotonic()-began,peak_tree_rss_gib=peak_rss,
        peak_tree_pss_gib=peak_pss,phases=peaks,diagnostic_only=True,
        note='Tree RSS double-counts shared pages; PSS apportions them. Short training is not a convergence test.')
    write_json(args.output/'profile.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if child.returncode:raise SystemExit(child.returncode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pool',choices=('mutation','lhs'),default='mutation')
    parser.add_argument('--gpu',type=int,default=6)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--train-size',type=int,default=10000)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--selectors',action='store_true')
    parser.add_argument('--plots',action='store_true')
    parser.add_argument('--max-pss-gib',type=float,help='Optional diagnostic cap; default uses available memory only')
    parser.add_argument('--minimum-available-gib',type=float,default=100.)
    parser.add_argument('--timeout',type=float,default=1800.)
    parser.add_argument('--real-reference',type=Path,required=True)
    parser.add_argument('--release-file',type=Path,help='Resource-only barrier after full training preprocessing')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.train_size<2 or args.epochs<1 or args.threads<1:parser.error('Positive sizes required')
    if args.worker:worker(args)
    else:monitor(args)


if __name__=='__main__':main()
