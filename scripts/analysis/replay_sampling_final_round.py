#!/usr/bin/env python3
"""Replay a frozen sampling round and observe physical real metrics by epoch.

The existing round adapter and shared Trainer own training. Real observations
never choose weights; only synthetic validation selects best checkpoints.
"""
import argparse
import csv
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from budding_yeast_v2.data.dataset_archive import atomic_json, file_sha256
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.scripts.analysis.probe_relative_l2_training_cap import (
    load_probe_cohorts, model_weights_sha256, save_csv)
from budding_yeast_v2.utils.evaluation.checkpoints import load_model_from_checkpoint
from budding_yeast_v2.utils.evaluation.metrics import calculate_metrics, RELATIVE_L2_METRIC_VERSION
from budding_yeast_v2.utils.evaluation.reporting import save_real_evaluation
from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig
from budding_yeast_v2.utils.sampling.benchmark.rounds import fit_round
from budding_yeast_v2.utils.sampling.contracts import PoolInputs, QueriedTraining
from budding_yeast_v2.utils.training.monitoring import evaluation_state


def predict_real(model, stats, cohort, batch_size):
    values = []
    with evaluation_state(model):
        for offset in range(0, len(cohort['data']), batch_size):
            end = min(offset+batch_size, len(cohort['data']))
            x, p = stats.encode(cohort['params'][offset:end], cohort['y_init'][offset:end],
                                cohort['data'].shape[-1])
            prediction = model.forward_ic_time(x.to(model.device), p.to(model.device)).float()
            physical = stats.decode(prediction)
            if not torch.isfinite(physical).all():
                raise FloatingPointError('Nonfinite real prediction in diagnostic replay')
            values.append(physical.cpu())
    return torch.cat(values)


class RealEpochTrace(Callback):
    """Read-only observer, with snapshots fixed before opening real results."""
    def __init__(self, output, cohort, batch_size, snapshots, expected_stats, expected_weights):
        self.output, self.cohort = output, cohort
        self.batch_size, self.snapshots = batch_size, set(snapshots)
        self.expected_stats, self.expected_weights = expected_stats, expected_weights
        self.rows, self.snapshot_audit = [], []
        self.truth = torch.as_tensor(cohort['data'])

    def on_fit_start(self, trainer, model):
        actual = NormalizationStats.from_datamodule(trainer.datamodule)
        checks = {name: torch.equal(getattr(actual, name).cpu(), getattr(self.expected_stats, name).cpu())
                  for name in ('mean_y', 'std_y', 'p_mean', 'p_std')}
        atomic_json(self.output/'normalization_audit.json', checks)
        if not all(checks.values()):
            raise ValueError('Refitted normalization differs from the original frozen round')

    def on_validation_epoch_end(self, trainer, model):
        if trainer.sanity_checking:
            return
        stats = NormalizationStats.from_datamodule(trainer.datamodule)
        predictions = predict_real(model, stats, self.cohort, self.batch_size)
        metrics = calculate_metrics(predictions, self.truth)
        row = dict(epoch=trainer.current_epoch+1, MAE=metrics['MAE'], MSE=metrics['MSE'],
                   Relative_L2=metrics['Relative L2'], Correlation=metrics['Correlation'],
                   metric_space='physical', metric_version=RELATIVE_L2_METRIC_VERSION)
        self.rows.append(row)
        save_csv(self.output/'real_epoch_metrics.csv', self.rows)
        print(f'[REAL PHYSICAL] epoch={row["epoch"]} MAE={row["MAE"]:.6f} '
              f'MSE={row["MSE"]:.6f} RelL2={row["Relative_L2"]:.6f} '
              f'Corr={row["Correlation"]:.6f} purpose=diagnostic_only', flush=True)

    def on_train_epoch_end(self, trainer, model):
        epoch = trainer.current_epoch+1
        if epoch in self.snapshots:
            trainer.save_checkpoint(self.output/f'epoch_{epoch:03d}.ckpt', weights_only=True)
            sha = model_weights_sha256(model)
            self.snapshot_audit.append(dict(epoch=epoch, weights_sha256=sha,
                original_best_weights_match=sha == self.expected_weights))
            atomic_json(self.output/'snapshot_audit.json', self.snapshot_audit)


def report(output, config, comparison, trace, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, key in zip(axes.flat, ('MAE', 'MSE', 'Relative_L2', 'Correlation')):
        ax.plot([r['epoch'] for r in trace], [r[key] for r in trace])
        ax.axvline(config['original_best_epoch'], ls='--', color='#d95f02', label='Original best epoch')
        ax.set(xlabel='Epoch', ylabel=key, title=f'Real mutants: physical {key}')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Fixed final-round cohort; real metrics are diagnostic only')
    fig.tight_layout()
    fig.savefig(output/'real_epoch_metrics.png', dpi=180)
    plt.close(fig)
    fields = ('MAE', 'MSE', 'Relative L2', 'Correlation')
    lines = ['# Power最终轮：物理多指标checkpoint诊断', '',
        '仅重跑固定最终轮，原run与数据保持原样；epsilon0.01、未截断Relative L2。',
        '完整训练/验证/合成test复用原行和顺序；相同派生模型seed、模型容量、batch、LR与warmup。',
        '逐epoch real只读监控，不能据real挑选正式checkpoint。',
        f'原物理validation-best：epoch{config["original_best_epoch"]}。', '',
        '|模型状态|MAE↓|MSE↓|物理RelL2↓|Correlation↑|', '|---|---:|---:|---:|---:|']
    for row in comparison:
        lines.append('|'+row['state']+'|'+ '|'.join(f'{row[k]:.6f}' for k in fields)+'|')
    lines += ['', '重跑epoch36是否与原权重逐字节相同，以snapshot_audit.json为准；',
        '若不相同，后续epoch指标只能称为同配置重跑结果，不能冒充原末轮权重的直接重评。',
        '固定epoch及validation-best的逐株CSV/指标见对应eval目录；只给原best对应epoch与末轮绘制全部126株图。',
        'real_epoch_metrics.csv/png记录200轮物理指标；history.csv保留原训练/验证/测试loss监控。',
        f'诊断耗时{config["seconds"]:.1f}秒，RSS峰值{config["peak_rss_gib"]:.3f}GiB，'
        f'CUDA allocated峰值{config["cuda_peak_allocated_gib"]:.3f}GiB。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    save_csv(output/'checkpoint_comparison.csv', comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--round', type=int, default=4)
    parser.add_argument('--snapshots', type=int, nargs='+', default=[36, 50, 100, 150, 200])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--plot-workers', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'config.json').exists():
        parser.error('Choose a fresh output directory; preserve prior diagnostics')
    torch.set_num_threads(2)
    started = time.monotonic()
    source_config = json.loads((args.run/'config.json').read_text())
    cfg = BenchmarkConfig(**source_config['training']).validate()
    metadata = json.loads((args.run/f'_state/round_{args.round:02d}/primary/training.json').read_text())
    original_history = list(csv.DictReader((args.run/f'rounds/round_{args.round:02d}/history.csv').open()))
    original_best_epoch = int(min(original_history, key=lambda r: float(r['val_error']))['epoch'])
    if any(e < 1 or e > cfg.epochs[args.round] for e in args.snapshots):
        parser.error('Snapshot epoch is outside the original round')
    for path, sha in source_config['sources'].items():
        if '/models/' in path or path.endswith(('data/normalization.py', 'utils/training/losses.py')):
            if file_sha256(Path(path)) != sha:
                raise ValueError(f'Original model/loss/normalization code changed: {path}')
    checkpoint, old_stats, contract, cohorts, selection = load_probe_cohorts(
        args.run, args.protocol, args.round, 5000, cfg.seed)
    original_model, _ = load_model_from_checkpoint(checkpoint, device='cpu')
    original_weights = model_weights_sha256(original_model)
    # Use the same inference backend for the original and replayed snapshots.
    # CPU/CUDA FFT roundoff can otherwise perturb even identical FNO weights.
    original_model.to(args.device)
    original_predictions = predict_real(original_model, old_stats, cohorts['real'], cfg.prediction_batch_size)
    original_metrics = calculate_metrics(original_predictions, torch.as_tensor(cohorts['real']['data']))
    del original_model, original_predictions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    config = dict(source_run=str(args.run), round=args.round, training=source_config['training'],
        original_checkpoint_sha256=file_sha256(checkpoint), original_best_epoch=original_best_epoch,
        derived_model_seed=metadata['derived_model_seed'], snapshots=args.snapshots,
        real_metrics_for_selection=False, source_history_sha256=file_sha256(args.run/f'rounds/round_{args.round:02d}/history.csv'),
        source_code={name:file_sha256(Path(__file__).parents[2]/name) for name in (
            'scripts/analysis/replay_sampling_final_round.py', 'utils/training/runner.py',
            'utils/sampling/benchmark/rounds.py', 'utils/training/monitoring.py')})
    atomic_json(args.output/'config.json', config)
    atomic_json(args.output/'selection.json', selection)
    atomic_json(args.output/'original_best_metrics.json', original_metrics)
    training = QueriedTraining(PoolInputs(np.asarray(selection['train']), cohorts['train']['params'],
        cohorts['train']['y_init']), cohorts['train']['data'])
    validation = [(PoolInputs(np.asarray(selection['validation']), cohorts['validation']['params'],
        cohorts['validation']['y_init']), cohorts['validation']['data'])]
    monitoring = {name: (PoolInputs(np.arange(len(cohorts[key]['data'])), cohorts[key]['params'],
        cohorts[key]['y_init']), cohorts[key]['data']) for name, key in [('real', 'real'), ('synth', 'synth_test')]}
    observer = RealEpochTrace(args.output, cohorts['real'], cfg.prediction_batch_size,
                              args.snapshots, old_stats, original_weights)
    norm_best = ModelCheckpoint(dirpath=args.output, filename='best_normalized', monitor='val_loss_synth',
        mode='min', save_top_k=1, save_weights_only=True, enable_version_counter=False)
    model, stats, history = fit_round(cfg, args.round, training, validation, args.output,
        metadata['derived_model_seed'], device=args.device, dataset_path=args.run/f'rounds/round_{args.round:02d}/dataset.json',
        state_dir=args.output/'_state', identity=dict(diagnostic='checkpoint_choice_replay', source_sha256=config['original_checkpoint_sha256']),
        monitoring=monitoring, extra_callbacks=[observer, norm_best])
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    with np.load(source_config['real_reference']['path'], allow_pickle=False) as archive:
        names = archive['mutant_names']
    comparison = [dict(state=f'original_epoch_{original_best_epoch:03d}', epoch=original_best_epoch, **original_metrics)]
    states = [(f'epoch_{e:03d}', args.output/f'epoch_{e:03d}.ckpt') for e in args.snapshots]
    states += [('best_physical', args.output/'model.ckpt'), ('best_normalized', args.output/'best_normalized.ckpt')]
    for label, path in states:
        model, _ = load_model_from_checkpoint(path, device=args.device)
        predictions = predict_real(model, stats, cohorts['real'], cfg.prediction_batch_size)
        summary = save_real_evaluation(predictions, torch.as_tensor(cohorts['real']['data']), names,
            args.output/label/'eval', plots=label in (f'epoch_{original_best_epoch:03d}', f'epoch_{cfg.epochs[args.round]:03d}'),
            plot_workers=args.plot_workers, context=dict(model=cfg.model, diagnostic=label))
        saved = torch.load(path, map_location='cpu', weights_only=False)
        comparison.append(dict(state=label, epoch=int(saved['epoch'])+1, **summary['all_variables']))
        model.cpu()
        del model, predictions, saved
        gc.collect()
        torch.cuda.empty_cache()
    config.update(seconds=time.monotonic()-started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    atomic_json(args.output/'results.json', dict(config=config, comparison=comparison,
        snapshot_audit=observer.snapshot_audit, real_epoch_metrics=observer.rows))
    report(args.output, config, comparison, observer.rows, history)
    print('FINAL ROUND PHYSICAL METRIC REPLAY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
