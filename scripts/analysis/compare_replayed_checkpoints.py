#!/usr/bin/env python3
"""Compare frozen replay checkpoints on synthetic validation and real mutants.

No training or checkpoint selection; retain only metrics and paired CSVs.
"""
import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from budding_yeast_v2.data.dataset_archive import TrajectoryStore, atomic_json
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
from budding_yeast_v2.scripts.analysis.probe_relative_l2_training_cap import save_csv
from budding_yeast_v2.scripts.analysis.replay_sampling_final_round import predict_real
from budding_yeast_v2.utils.evaluation.checkpoints import load_model_from_checkpoint
from budding_yeast_v2.utils.evaluation.metrics import (
    calculate_metrics, per_sample_metrics, sample_channel_relative_errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads((args.replay/'config.json').read_text())
    # Plotting may still be running; all immutable training snapshots must exist.
    if not all((args.replay/f'epoch_{e:03d}.ckpt').exists() for e in config['snapshots']):
        parser.error('All fixed training snapshots must be complete first')
    source = Path(config['source_run'])
    source_config = json.loads((source/'config.json').read_text())
    ids = json.loads((args.replay/'selection.json').read_text())['validation']
    manifest = json.loads((source/f'rounds/round_{config["round"]:02d}/dataset.json').read_text())
    store = TrajectoryStore(manifest['source'], cache_chunks=1)
    if store.digest != manifest['source_sha256']:
        raise ValueError('Frozen source changed')
    validation = {'data': store.read(np.asarray(ids), ('data',))['data']}
    with np.load(args.protocol/f'{source_config["pool"]}_inputs.npz', allow_pickle=False) as a:
        validation.update(params=a['params'][ids], y_init=a['y_init'][ids])
    with np.load(source_config['real_reference']['path'], allow_pickle=False) as a:
        real = {k: a[k] for k in ('params', 'y_init', 'data')}
        names = a['mutant_names'].astype(str)
    summaries, variables, cases = [], [], {}
    for epoch in config['snapshots']:
        model, _ = load_model_from_checkpoint(args.replay/f'epoch_{epoch:03d}.ckpt', device=args.device)
        stats = NormalizationStats.from_datamodule(model)
        for name, cohort in [('validation', validation), ('real', real)]:
            pred = predict_real(model, stats, cohort, 128)
            truth = torch.as_tensor(cohort['data'])
            summaries.append(dict(epoch=epoch, cohort=name, **calculate_metrics(pred, truth)))
            if name == 'real':
                cases[epoch] = {name: per_sample_metrics(p, y)
                    for name, p, y in zip(names, pred, truth)}
                errors = pred.double()-truth.double()
                mae = errors.abs().mean((0, 2)).tolist()
                mse = errors.square().mean((0, 2)).tolist()
                rel = sample_channel_relative_errors(pred, truth).mean(0).tolist()
                variables.extend(dict(epoch=epoch, variable=v, MAE=mae[j], MSE=mse[j],
                    Relative_L2=rel[j]) for j, v in enumerate(VARIABLE_NAMES))
            del pred
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save_csv(args.replay/'validation_and_real_checkpoints.csv', summaries)
    save_csv(args.replay/'real_per_variable_checkpoints.csv', variables)
    first, last = config['original_best_epoch'], max(config['snapshots'])
    earlier, later = cases[first], cases[last]
    if earlier.keys() != later.keys() or len(earlier) != 126:
        raise ValueError('Real reference cases changed')
    paired = []
    for name in earlier:
        row = dict(mutant_name=name)
        for metric in ('MAE', 'MSE', 'rel_l2', 'correlation'):
            a, b = float(earlier[name][metric]), float(later[name][metric])
            row.update({f'{metric}_epoch{first}': a, f'{metric}_epoch{last}': b,
                        f'{metric}_delta': b-a})
        paired.append(row)
    save_csv(args.replay/'per_mutant_checkpoint_changes.csv', paired)
    changes = {metric: sum(r[f'{metric}_delta'] < 0 for r in paired) for metric in ('MAE', 'MSE', 'rel_l2')}
    changes['correlation'] = sum(r['correlation_delta'] > 0 for r in paired)
    atomic_json(args.replay/'paired_improved_cases.json', dict(count=126, earlier_epoch=first,
        later_epoch=last, improved=changes, diagnostic_only=True))
    lines = ['', '## 同池合成validation的物理指标复核', '',
        '|Epoch|validation MAE↓|validation MSE↓|validation物理RelL2↓|', '|---:|---:|---:|---:|']
    for row in summaries:
        if row['cohort'] == 'validation':
            lines.append(f'|{row["epoch"]}|{row["MAE"]:.6f}|{row["MSE"]:.6f}|{row["Relative L2"]:.6f}|')
    lines += ['', '此处只作固定checkpoint诊断，没有用real结果改选模型。',
        f'126株中从epoch{first}到{last}，MAE/MSE/RelL2/Corr改善数量为'
        f'{changes["MAE"]}/{changes["MSE"]}/{changes["rel_l2"]}/{changes["correlation"]}。',
        '逐株及逐变量变化见per_mutant_checkpoint_changes.csv与real_per_variable_checkpoints.csv。']
    (args.replay/'validation_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summaries=summaries, improved=changes), indent=2), flush=True)


if __name__ == '__main__':
    main()
