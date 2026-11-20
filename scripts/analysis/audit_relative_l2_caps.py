#!/usr/bin/env python3
"""Replay saved checkpoints to audit normalized Relative L2 caps, without training.

One process/model/device at a time. Per-pool training unions are loaded once;
inference is batched and predictions/trajectories are never written to reports.
This is a checkpoint-state audit, not a reconstruction of historical gradients.
"""
import argparse
import csv
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from budding_yeast_v2.data.dataset_archive import TrajectoryStore, atomic_json, file_sha256
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
from budding_yeast_v2.utils.evaluation.checkpoints import load_sampling_checkpoint
from budding_yeast_v2.utils.training.losses import RelativeL2Loss


def load_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError('Cannot determine available host memory')


def checkpoint_identity(checkpoint, config, cohort, row_ids, device):
    return dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=file_sha256(checkpoint),
                loss_source_sha256=file_sha256(RelativeL2Loss.forward.__code__.co_filename),
                protocol=config['protocol_fingerprint'], cohort=cohort, device=str(device),
                row_ids_sha256=__import__('hashlib').sha256(np.asarray(row_ids, dtype=np.int64).tobytes()).hexdigest())


def audit_cohort(model, stats, arrays, truth, truth_indices, row_ids, names, batch_size):
    """Return the exact saved loss plus uncapped diagnostics, per sample/channel."""
    criterion = model.criterion
    if not isinstance(criterion, RelativeL2Loss):
        raise ValueError('Audit requires the existing RelativeL2Loss training criterion')
    if criterion.max_clamp is None:
        raise ValueError('Checkpoint criterion has no finite cap')
    raw_parts, case_rows = [], []
    with torch.inference_mode(), torch.autocast(device_type=model.device.type, enabled=False):
        for start in range(0, len(row_ids), batch_size):
            ids = np.asarray(row_ids[start:start+batch_size], dtype=np.int64)
            lookup = truth_indices[start:start+len(ids)]
            target = torch.as_tensor(truth[lookup], dtype=torch.float32, device=model.device)
            x, p = stats.encode(arrays['params'][ids], arrays['y_init'][ids], target.shape[-1])
            prediction = model.forward_ic_time(x.to(model.device), p.to(model.device)).float()
            normalized_target = stats.targets(target)
            if prediction.shape != target.shape or not torch.isfinite(prediction).all() or not torch.isfinite(normalized_target).all():
                raise FloatingPointError('Nonfinite or misaligned prediction/target')
            ratio = torch.linalg.vector_norm(prediction-normalized_target, dim=-1) / (
                normalized_target.square().sum(-1) + criterion.eps**2).sqrt()
            capped = ratio.clamp(max=criterion.max_clamp)
            torch.testing.assert_close(capped.mean(1), criterion.per_sample(prediction, normalized_target))
            torch.testing.assert_close(capped.mean(), criterion(prediction, normalized_target))
            raw = ratio.cpu().numpy()
            raw_parts.append(raw)
            for k, candidate_id in enumerate(ids):
                case_rows.append(dict(sample_id=int(candidate_id), sample_name=str(names[candidate_id]),
                    clipped_channels=int((raw[k] > criterion.max_clamp).sum()),
                    relative_l2_norm=float(capped[k].mean()),
                    relative_l2_norm_uncapped=float(raw[k].mean()), max_channel_ratio=float(raw[k].max())))
    raw = np.concatenate(raw_parts)
    if not np.isfinite(raw).all():
        raise FloatingPointError('Nonfinite relative error')
    capped = np.minimum(raw, criterion.max_clamp)
    clipped = raw > criterion.max_clamp
    summary = dict(samples=len(raw), variables=raw.shape[1], sample_variable_pairs=int(raw.size),
        eps=float(criterion.eps), max_clamp=float(criterion.max_clamp),
        relative_l2_norm=float(capped.astype(np.float64).mean()),
        relative_l2_norm_uncapped=float(raw.astype(np.float64).mean()),
        clipped_pairs=int(clipped.sum()), clipped_pair_pct=float(100*clipped.mean()),
        samples_with_clipping=int(clipped.any(1).sum()), sample_clip_pct=float(100*clipped.any(1).mean()),
        pairs_exactly_at_cap=int((raw == criterion.max_clamp).sum()),
        max_uncapped_channel_ratio=float(raw.max()),
        p99_uncapped_channel_ratio=float(np.quantile(raw, .99)),
        channels=[dict(variable=VARIABLE_NAMES[v], clipped_pairs=int(clipped[:, v].sum()),
            clipped_pair_pct=float(100*clipped[:, v].mean()),
            normalized_relative_mean=float(capped[:, v].astype(np.float64).mean()),
            uncapped_relative_mean=float(raw[:, v].astype(np.float64).mean()),
            max_uncapped_channel_ratio=float(raw[:, v].max())) for v in range(raw.shape[1])])
    return summary, case_rows


def summarize_output(output):
    rows, channels = [], []
    for path in sorted((output/'checkpoints').glob('*/*/*/*.json')):
        value = load_json(path)
        base = {k:value[k] for k in ('pool','method','round','checkpoint_epoch','cohort')}
        summary = value['summary']
        rows.append(dict(**base, **{k:v for k,v in summary.items() if k != 'channels'}))
        channels.extend(dict(**base, **row) for row in summary['channels'])
    if rows:
        write_csv(output/'summary.csv', rows)
        write_csv(output/'per_variable.csv', channels)
    return rows


def write_report(output, experiments_root):
    """Render the audit independently, without rerunning inference or changing eval."""
    output, experiments_root = Path(output), Path(experiments_root)
    details = [load_json(p) for p in sorted((output/'checkpoints').glob('*/*/*/*.json'))]
    rows = summarize_output(output)
    validation = []
    for d in details:
        if d['cohort'] not in ('real', 'synth_test'):
            continue
        history = experiments_root/d['pool']/d['method']/'seed_42/rounds'/f'round_{d["round"]:02d}'/'history.csv'
        with history.open() as handle:
            recorded = list(csv.DictReader(handle))[d['checkpoint_epoch']-1]
        key = 'test_loss_real' if d['cohort']=='real' else 'test_loss_synth'
        validation.append(dict(pool=d['pool'], method=d['method'], round=d['round'], cohort=d['cohort'],
            saved_epoch=d['checkpoint_epoch'], recorded_loss=float(recorded[key]),
            replayed_loss=d['summary']['relative_l2_norm'],
            absolute_difference=abs(float(recorded[key])-d['summary']['relative_l2_norm'])))
    if validation:
        write_csv(output/'history_crosscheck.csv', validation)
    final=[]
    for d in details:
        if d['round'] != 4 or d['cohort'] != 'real':
            continue
        peers={q['cohort']:q['summary'] for q in details if
            (q['pool'],q['method'],q['round'])==(d['pool'],d['method'],d['round'])}
        run=experiments_root/d['pool']/d['method']/'seed_42'
        with (run/'eval/per_mutant_metrics.csv').open() as handle:
            cases=list(csv.DictReader(handle))
        final.append(dict(pool=d['pool'],method=d['method'],
            relative_l2_norm=peers['real']['relative_l2_norm'],
            relative_l2_norm_uncapped=peers['real']['relative_l2_norm_uncapped'],
            real_clipped_pairs=peers['real']['clipped_pairs'],
            train_clipped_pairs=peers.get('train',{}).get('clipped_pairs'),
            synth_test_clipped_pairs=peers.get('synth_test',{}).get('clipped_pairs'),
            macro_sample_channel_rel_l2_physical=float(np.mean([float(r['rel_l2']) for r in cases]))))
    if final:
        write_csv(output/'final_models.csv', final)
    train=[r for r in rows if r['cohort']=='train']
    real=[r for r in rows if r['cohort']=='real']
    flagged=[r for r in train if r['clipped_pairs']]
    lines=['# Relative L2 cap10 检查（2026-11-15）','',
        f'源实验：`{experiments_root}`。仅重新推理，不修改模型、损失、训练数据或正式 eval 文件。','',
        f'覆盖 {len({(r["pool"],r["method"]) for r in real})} 项实验、{len(real)} 个已保存主模型 checkpoint；'
        f'{len(train)} 组训练集回放、{len(real)} 组 real 评测，合计 {len(rows)} 组 cohort 回放。',
        'checkpoint 是各轮 validation 选择的 best，并不一定是最后一个 epoch；ensemble 辅助成员不在本次主模型检查范围。','',
        '## 结论与历史边界','',
        f'训练集回放中 {len(flagged)}/{len(train)} 个主模型出现 ratio>10；'
        f'real 回放中 {sum(r["clipped_pairs"]>0 for r in real)}/{len(real)} 个出现。',
        '每项统计先对每个样本的每个变量沿时间求相对范数，再检查是否超过10；“pair比例”分母是样本数×39，“样本比例”是至少一个变量超限的样本比例。',
        '本次报告回答保存状态下是否触发。历史 history.csv 未记录逐 batch/逐 step cap 计数，故不能还原训练全程频率、持续时间或累计零梯度项数；未触发的保存状态也不能证明中间从未触发。',
        '以 eval/no-autocast FP32 回放，训练本身为 bf16 mixed；不能把回放比率冒充历史训练步精确计数。',
        f'与对应保存 epoch 的日志监控核对：{len(validation)} 组，最大 loss 绝对差 '
        f'{max((r["absolute_difference"] for r in validation),default=0):.3g}。训练日志的 train_loss 是沿训练过程累计，不能要求其等于固定 checkpoint 的训练集重算值。','',
        '## 存在截断的训练集回放','',
        '|池|方法|round（从0计）|epoch|超限pair|pair比例|至少一项超限的样本比例|cap后loss|cap前loss|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in flagged:
        lines.append(f'|{r["pool"]}|{r["method"]}|{r["round"]}|{r["checkpoint_epoch"]}|'
            f'{r["clipped_pairs"]}/{r["sample_variable_pairs"]}|{r["clipped_pair_pct"]:.6f}%|'
            f'{r["sample_clip_pct"]:.3f}%|{r["relative_l2_norm"]:.6f}|{r["relative_l2_norm_uncapped"]:.6f}|')
    lines+=['','## 最后一轮主模型','',
        '|池|方法|训练pair超限数|real pair超限数|合成test pair超限数|real归一化loss|real归一化uncapped|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in final:
        lines.append(f'|{r["pool"]}|{r["method"]}|{r["train_clipped_pairs"]}|{r["real_clipped_pairs"]}|'
            f'{r["synth_test_clipped_pairs"]}|{r["relative_l2_norm"]:.6f}|{r["relative_l2_norm_uncapped"]:.6f}|')
    lines+=['','## eval 建议（未改生产实现）','',
        '训练继续当前 RelativeL2Loss。eval 增加与训练完全一致的 relative_l2_norm，同时报告 uncapped 和 cap比例，保留物理空间 MAE/MSE/Correlation/原历史RelL2。',
        '若目标是每株、每变量等权，物理空间可对现有 per_mutant_metrics.csv 的 rel_l2 求算术平均；本报告 final_models.csv 的 macro_sample_channel_rel_l2_physical 就是该值，沿用原有物理分母保护，不是新的无保护百分比误差。',
        '不同方法/轮次训练集重新拟合的归一化 stats 不同，因此归一化 loss 即使公式相同也不是完全一致的跨模型物理标尺。不能把 cap10 下的 loss 作为唯一的 badcase 严重程度指标。','',
        'GNOT 与当前代码的归约结构相同，但没有本项目 eps=.1、cap10，也没有相同的 log1p 编码；GNOT 本身也将归一化训练loss与逆变换后的物理metrics分开计算。','',
        '完整数据：summary.csv、per_variable.csv、final_models.csv、history_crosscheck.csv；checkpoints/ 中保留每模型每cohort的摘要与逐样本表，无重复轨迹。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pools', nargs='+', default=['mutation','lhs'])
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--rounds', nargs='+', type=int, default=[4])
    parser.add_argument('--cohorts', nargs='+', choices=['train','real','synth_test'], default=['train','real'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch size and threads must be positive')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = load_json(args.protocol/'protocol.json')
    began = time.monotonic()
    processed = 0
    for pool in args.pools:
        runs = sorted((args.experiments_root/pool).glob('*/seed_42'))
        if args.methods:
            runs = [p for p in runs if p.parent.name in args.methods]
        if not runs:
            raise ValueError(f'No matching experiments: {pool}')
        jobs = []
        for run in runs:
            config = load_json(run/'config.json')
            completion = load_json(run/'_state/state.json')['completion']
            if completion['status'] != 'complete' or config['training']['loss'] != 'rel_l2':
                raise ValueError('Only completed rel_l2 runs may be audited')
            loss_path = Path(RelativeL2Loss.forward.__code__.co_filename).resolve()
            if config['sources'].get(str(loss_path)) != file_sha256(loss_path):
                raise ValueError('Training loss source differs from the completed run')
            for ri in args.rounds:
                rd = run/'rounds'/f'round_{ri:02d}'
                with (rd/'history.csv').open() as handle:
                    header = next(csv.reader(handle))
                jobs.append(dict(run=run,config=config,round=ri,rd=rd,
                    rows=np.asarray(load_json(rd/'dataset.json')['rows'],dtype=np.int64),
                    historical_cap_counter_present=any('cap' in k or 'clipp' in k for k in header)))
        real_path = Path(jobs[0]['config']['real_reference']['path'])
        if any(j['config']['real_reference'] != jobs[0]['config']['real_reference'] for j in jobs):
            raise ValueError('Real references disagree')
        if file_sha256(real_path) != jobs[0]['config']['real_reference']['sha256']:
            raise ValueError('Real reference changed')
        with np.load(real_path, allow_pickle=False) as a:
            real = {key:a[key] for key in ('params','y_init','data','mutant_names')}
        arrays = union_truth = union_rows = store = None
        if any(c != 'real' for c in args.cohorts):
            if available_gib() < 128:
                raise RuntimeError('Host memory reserve below 128 GiB; stopping audit')
            with np.load(args.protocol/f'{pool}_inputs.npz', allow_pickle=False) as a:
                arrays = {key:a[key] for key in ('params','y_init','mutant_names')}
            with np.load(args.protocol/f'{pool}_split.npz', allow_pickle=False) as a:
                synth_test_ids = a['test'].copy()
            union_rows = np.unique(np.concatenate(
                ([j['rows'] for j in jobs] if 'train' in args.cohorts else []) +
                ([synth_test_ids] if 'synth_test' in args.cohorts else [])))
            store = TrajectoryStore(protocol['pools'][pool]['path'], cache_chunks=1)
            if store.digest != protocol['pools'][pool]['sha256']:
                raise ValueError('Pool source changed')
            print(f'Loading {pool}: {len(union_rows)} unique selected/holdout rows once', flush=True)
            union_truth = store.read(union_rows, ('data',))['data']
        arr = truth = lookup = None
        for job in jobs:
            rd, config = job['rd'], job['config']
            checkpoint = rd/'model.ckpt'
            model = stats = payload = None
            for cohort in args.cohorts:
                ids = np.arange(len(real['data'])) if cohort == 'real' else (
                    job['rows'] if cohort == 'train' else synth_test_ids)
                identity = checkpoint_identity(checkpoint, config, cohort, ids, args.device)
                dest = args.output/'checkpoints'/pool/job['run'].parent.name/f'round_{job["round"]:02d}'/f'{cohort}.json'
                if dest.exists():
                    if load_json(dest)['identity'] != identity:
                        raise ValueError(f'Audit identity changed: {dest}')
                    continue
                if model is None:
                    model, stats, payload = load_sampling_checkpoint(checkpoint, device=args.device)
                arr, truth = (real, real['data']) if cohort == 'real' else (arrays, union_truth)
                lookup = ids if cohort == 'real' else np.searchsorted(union_rows, ids)
                t0 = time.monotonic()
                summary, cases = audit_cohort(model, stats, arr, truth, lookup, ids,
                    arr['mutant_names'], args.batch_size)
                result = dict(identity=identity, pool=pool, method=job['run'].parent.name,
                    round=job['round'], checkpoint_epoch=int(payload['epoch'])+1, cohort=cohort,
                    historical_cap_counter_present=job['historical_cap_counter_present'],
                    replay_precision='float32/no-autocast/eval-mode', summary=summary,
                    seconds=time.monotonic()-t0)
                dest.parent.mkdir(parents=True, exist_ok=True)
                write_csv(dest.with_suffix('.csv'), cases)
                atomic_json(dest, result)
                processed += 1
                print(f'{pool}/{result["method"]}/r{job["round"]:02d}/{cohort}: '
                    f'cap pairs={summary["clipped_pairs"]}/{summary["sample_variable_pairs"]} '
                    f'({summary["clipped_pair_pct"]:.5f}%), loss={summary["relative_l2_norm"]:.6f}, '
                    f'uncapped={summary["relative_l2_norm_uncapped"]:.6f}, {result["seconds"]:.1f}s', flush=True)
            del model, stats, payload
            gc.collect()
        del jobs, arrays, union_truth, union_rows, real, store, arr, truth, lookup
        gc.collect()
    rows = summarize_output(args.output)
    atomic_json(args.output/'audit.json', dict(status='complete', checkpoints=len({
        (r['pool'],r['method'],r['round']) for r in rows}),cohort_replays=len(rows),
        processed_this_invocation=processed, seconds=time.monotonic()-began,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if str(args.device).startswith('cuda') else 0.,
        note='Saved best-checkpoint inference only; no per-step historical cap counters. No optimization or test-based tuning.'))
    write_report(args.output, args.experiments_root)
    print(json.dumps(load_json(args.output/'audit.json')), flush=True)


if __name__ == '__main__':
    main()
