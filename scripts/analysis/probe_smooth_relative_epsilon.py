#!/usr/bin/env python3
"""Paired smooth-epsilon diagnostic using the project's shared model/trainer.

Only training epsilon changes. All physical scores use the same fixed epsilon;
synthetic validation selects checkpoints. No formal experiment is overwritten.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from budding_yeast_v2.data.dataset_archive import atomic_json, file_sha256
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
from budding_yeast_v2.scripts.analysis.probe_relative_l2_training_cap import (
    FixedCohort, evaluate, initialize_probe_model, load_probe_cohorts,
    model_weights_sha256, save_csv)
from budding_yeast_v2.scripts.analysis.probe_relative_l2_epsilon import GradientDiagnostics
from budding_yeast_v2.utils.evaluation.checkpoints import load_model_from_checkpoint
from budding_yeast_v2.utils.evaluation.metrics import (
    PHYSICAL_RELATIVE_L2_EPS, RELATIVE_L2_METRIC_VERSION)
from budding_yeast_v2.utils.training.losses import SMOOTH_RELATIVE_L2_RULE
from budding_yeast_v2.utils.training.runner import EpochSummaryCallback, make_trainer


def audit_denominators(target, eps_values, output):
    """Quantify deviation from pure Relative L2 independently of model quality."""
    norms = target.double().norm(dim=-1).numpy()
    rows, summary = [], {}
    for eps in eps_values:
        shrink = 1 - norms / np.hypot(norms, eps)
        summary[str(eps)] = dict(pairs=int(norms.size), zero_pairs=int((norms == 0).sum()),
            min_norm=float(norms.min()), below_epsilon=int((norms < eps).sum()),
            shrink_max_pct=float(100 * shrink.max()), shrink_mean_pct=float(100 * shrink.mean()),
            shrink_over_1pct_pairs=int((shrink > .01).sum()),
            shrink_over_10pct_pairs=int((shrink > .1).sum()),
            maximum_inverse_denominator=float(1 / np.hypot(norms, eps).min()))
        for i, name in enumerate(VARIABLE_NAMES):
            rows.append(dict(eps=eps, variable=name, min_norm=float(norms[:, i].min()),
                median_norm=float(np.median(norms[:, i])),
                below_epsilon=int((norms[:, i] < eps).sum()),
                shrink_max_pct=float(100 * shrink[:, i].max()),
                shrink_mean_pct=float(100 * shrink[:, i].mean())))
    save_csv(output / 'training_denominators.csv', rows)
    atomic_json(output / 'denominator_summary.json', summary)
    return summary


def write_report(output, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for arm in results['arms']:
        label = f'train eps={arm["eps"]:g}'
        h = arm['history']
        epochs = [r['epoch'] for r in h]
        axes[0].plot(epochs, [r['val_loss_synth'] for r in h], label=label)
        axes[1].plot(epochs, [r['val_sample_channel_rel_l2'] for r in h], label=label)
        axes[2].plot(epochs, [r['gradient_norm_max'] for r in arm['gradients']], label=label)
        for stage, cohorts in arm['evaluations'].items():
            for name, metrics in cohorts.items():
                rows.append(dict(eps=arm['eps'], stage=stage, cohort=name, **metrics))
    axes[0].set(title='Normalized validation loss (training epsilon)', xlabel='Epoch', ylabel='Relative L2')
    axes[1].set(title=f'Physical validation (fixed eps={PHYSICAL_RELATIVE_L2_EPS})',
                xlabel='Epoch', ylabel='Sample/channel Relative L2')
    axes[2].set(title='Gradient norm before clipping', xlabel='Epoch', ylabel='Gradient norm', yscale='log')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Same data, initialization and batch order; test/real excluded from selection')
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=180)
    plt.close(fig)
    save_csv(output / 'comparison.csv', rows)
    lines = ['# 平滑 Relative L2：较小 epsilon 对照', '',
        '训练分母 sqrt(||z||²+eps²)，不截断；梯度范数裁剪1。',
        '各臂共用相同源行、train-only统计、初始权重和批次顺序；单seed42。',
        'FNO width192/blocks4/modes24；60epoch、batch128、LR7.5e-4、warmup4、最低LR1e-5。',
        '同时记录按归一化验证loss与按物理验证指标选出的checkpoint；只访问合成validation进行选择。',
        '全部物理比较固定eps=0.01；各臂训练eps不同，不能按训练loss数值直接排名。',
        'test/real仅在validation结果落盘后推理。本诊断不自动改生产epsilon。', '',
        '|训练eps|末轮validation物理RelL2|末轮validation MAE|最大梯度|训练分母对原比值最大减小|步数|',
        '|---:|---:|---:|---:|---:|---:|']
    for arm in results['arms']:
        val = arm['evaluations']['last']['validation']
        distortion = results['denominators'][str(arm['eps'])]['shrink_max_pct']
        lines.append(f'|{arm["eps"]}|{val["sample_channel_rel_l2"]:.6f}|{val["MAE"]:.6f}|'
            f'{max(r["gradient_norm_max"] for r in arm["gradients"]):.4f}|{distortion:.6f}%|'
            f'{arm["optimizer_steps"]}|')
    lines += ['', '对于严格零目标，未保护的Relative L2没有定义，减小epsilon会提高该项最大权重。',
        '本数据子集若没有近零归一化范数，只能验证常见尺度下的稳定性，不能证明所有零目标下稳定。',
        f'进程RSS峰值 {results["peak_rss_gib"]:.3f} GiB；CUDA allocated峰值 '
        f'{results["cuda_peak_allocated_gib"]:.3f} GiB。',
        '不同checkpoint的验证指标、test/real结果见comparison.csv；源ID见selection.json，不复制轨迹。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eps', type=float, nargs='+', default=[.01, .003, .001])
    parser.add_argument('--holdout-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if len(set(args.eps)) != len(args.eps) or any(not np.isfinite(e) or e <= 0 for e in args.eps):
        parser.error('epsilon must be distinct, finite and positive')
    if PHYSICAL_RELATIVE_L2_EPS != .01 or not RELATIVE_L2_METRIC_VERSION.endswith('smooth_eps001_v3'):
        parser.error('Recorded diagnostic requires fixed physical smooth eps=0.01')
    if args.epochs < 5:
        parser.error('use at least five epochs')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'config.json').exists():
        parser.error('Use a fresh output directory; preserve completed diagnostics')
    torch.set_num_threads(2)
    checkpoint, stats, contract, cohorts, selection = load_probe_cohorts(
        args.run, args.protocol, args.round, args.holdout_size, args.seed)
    identity = dict(source=str(checkpoint), source_sha256=file_sha256(checkpoint),
        seed=args.seed, train_rows=len(selection['train']), validation_rows=len(selection['validation']),
        epochs=args.epochs, eps_values=args.eps, batch_size=128, lr=7.5e-4, warmup=4, min_lr=1e-5,
        denominator_rule=SMOOTH_RELATIVE_L2_RULE, loss_cap=None, gradient_clip_norm=1.,
        physical_eval_eps=PHYSICAL_RELATIVE_L2_EPS, physical_metric_version=RELATIVE_L2_METRIC_VERSION,
        selection='synthetic validation only; both normalized and physical checkpoints; no default change',
        source_files={p: file_sha256(Path(__file__).parents[2] / p) for p in (
            'scripts/analysis/probe_smooth_relative_epsilon.py', 'scripts/analysis/probe_relative_l2_epsilon.py',
            'scripts/analysis/probe_relative_l2_training_cap.py', 'utils/training/losses.py',
            'utils/evaluation/metrics.py', 'utils/training/runner.py', 'models/yeast_lit_module.py')})
    atomic_json(args.output / 'config.json', identity)
    atomic_json(args.output / 'selection.json', selection)
    dm = FixedCohort(cohorts['train'], cohorts['validation'], stats, contract, 128, args.seed)
    dm.validation_metric = 'sample_channel_rel_l2'
    audit = audit_denominators(dm.train_dataset_synth.tensors[2], args.eps, args.output)
    print('DENOMINATOR AUDIT ' + json.dumps(audit), flush=True)
    arms, initial_sha = [], None
    for eps in args.eps:
        out = args.output / f'eps_{eps:g}'
        out.mkdir()
        pl.seed_everything(args.seed, workers=True)
        model = initialize_probe_model(checkpoint, 'scratch')
        model.criterion.eps, model.criterion.max_clamp = eps, None
        model.criterion.denominator_rule = SMOOTH_RELATIVE_L2_RULE
        model.lr, model.max_epochs = 7.5e-4, args.epochs
        model.hparams.update(lr=7.5e-4, max_epochs=args.epochs, warmup_epochs=4, min_lr=1e-5)
        weights_sha = model_weights_sha256(model)
        if initial_sha is None:
            initial_sha = weights_sha
        if weights_sha != initial_sha:
            raise ValueError('Initialization mismatch')
        dm.loader_generator = torch.Generator().manual_seed(args.seed)
        recorder = EpochSummaryCallback(output=out, identity=identity, generator=dm.loader_generator,
            round_index=args.round, monitor='val_sample_channel_rel_l2')
        gradients = GradientDiagnostics(out)
        checkpoints = {label: ModelCheckpoint(dirpath=out, filename=label, monitor=monitor,
            mode='min', save_top_k=1, save_weights_only=True, enable_version_counter=False)
            for label, monitor in [('best_physical', 'val_sample_channel_rel_l2'),
                                   ('best_normalized', 'val_loss_synth')]}
        trainer = make_trainer(max_epochs=args.epochs, accelerator='gpu',
            devices=[int(args.device.split(':')[-1])], logger=False,
            callbacks=[gradients, recorder, *checkpoints.values()],
            enable_progress_bar=False, enable_model_summary=False)
        started = time.monotonic()
        model.train()
        trainer.fit(model, datamodule=dm)
        model.to(args.device)
        validation = evaluate(model, stats, cohorts['validation'], 128)
        trainer.save_checkpoint(out / 'last.ckpt', weights_only=True)
        expected = int(np.ceil(len(selection['train']) / 128)) * args.epochs
        if recorder.history[-1]['optimizer_steps'] != expected:
            raise ValueError('Unexpected update budget')
        arm = dict(eps=eps, initial_weights_sha256=weights_sha, seconds=time.monotonic()-started,
            optimizer_steps=expected, history=recorder.history, gradients=gradients.rows,
            last_validation=validation,
            selected_epochs={name: int(torch.load(callback.best_model_path, map_location='cpu',
                weights_only=False)['epoch'])+1 for name, callback in checkpoints.items()})
        atomic_json(out / 'validation_result.json', arm)
        arms.append(arm)
        print(f'ARM COMPLETE eps={eps} validation={validation}', flush=True)
        model.cpu()
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(args.output / 'validation_comparison.json', dict(used_test_or_real=False,
        scores={str(a['eps']): a['last_validation'] for a in arms}, production_defaults_changed=False))
    # Holdout inference begins only after validation summaries are persisted.
    for arm in arms:
        arm['evaluations'] = {}
        for stage in ('last', 'best_physical', 'best_normalized'):
            model, _ = load_model_from_checkpoint(args.output/f'eps_{arm["eps"]:g}'/f'{stage}.ckpt',
                                                device=args.device)
            if model.criterion.eps != arm['eps']:
                raise ValueError('Checkpoint epsilon not restored')
            arm['evaluations'][stage] = {name: evaluate(model, stats, cohort, 128)
                for name, cohort in cohorts.items() if name != 'train'}
            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        atomic_json(args.output/f'eps_{arm["eps"]:g}'/'result.json', arm)
    result = dict(identity=identity, denominators=audit, arms=arms,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    atomic_json(args.output/'results.json', result)
    write_report(args.output, result)
    print('SMOOTH EPSILON PROBE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
