#!/usr/bin/env python3
"""Export a verified, complete subset of the sampling result inventory."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile


NAMES = {
    'random': 'Random', 'power': 'Power/SBAL', 'lcmd_ll': 'LCMD-LL',
    'physics_pre_power': 'Physics-PRE-Power', 'physics_pre_topk': 'Physics-PRE-TopK [补充]',
    'hggs_projected': 'HGGS [旧训练流程]', 'low_rank': 'Low-Rank-GradientEnergy',
    'peaks_regression': 'PEAKS-Regression', 'ogas_async_projected': 'OGAS [公式修正后]',
    'ours_sensitivity': 'Ours: Sensitivity', 'ours_dynamics_transport': 'Ours: Transport',
    'ours_intervention_closure': 'Ours: Closure',
}
COHORTS = ('real_mutants', 'synth_test', 'cross_pool_test')


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)


def flatten(value, prefix=''):
    if isinstance(value, (dict, list)):
        items = value.items() if isinstance(value, dict) else enumerate(value)
        return {key: leaf for name, child in items
                for key, leaf in flatten(child, f'{prefix}.{name}' if prefix else str(name)).items()}
    return {prefix: value}


def label(record):
    name = NAMES[record['method']]
    if record['method'].startswith('ours_'):
        name += ' [生成]' if record['workflow'] == 'round_generation' else ' [池选]'
    return name


def number(value):
    if value is None: return '未定义'
    return f'{value:.6g}' if isinstance(value, float) else str(value)


def table(lines, headers, rows):
    lines += ['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|']
    lines += ['|' + '|'.join(map(number, row)) + '|' for row in rows]
    lines.append('')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((args.source / 'all_metrics.json').read_text())
    records = [r for r in inventory['experiments'] if r['method'] in NAMES and
               (r['method'] != 'ogas_async_projected' or r['version'] == 'ogas_aligned_20261118')]
    records.sort(key=lambda r: (r['pool'] != 'mutation', list(NAMES).index(r['method']), r['workflow']))
    ids = {r['run_id'] for r in records}; assert len(ids) == len(records)
    final = []; hashes = {}
    for r in records:
        source = Path(r['metrics_file']); raw = source.read_bytes()
        assert json.loads(raw) == r['metrics'], f'Source metrics changed: {source}'
        hashes[str(source)] = hashlib.sha256(raw).hexdigest()
        for cohort in COHORTS:
            values = r['metrics'][cohort]; matrix = values['pattern_metrics']['six_group_confusion']
            assert sum(map(sum, matrix)) == values['count']
            final.append(dict(run_id=r['run_id'], display=label(r), pool=r['pool'], method=r['method'],
                workflow=r['workflow'], version=r['version'], cohort=cohort,
                round=r['metrics']['round'], train_samples=r['train_samples'], metrics_file=str(source),
                **flatten(values), **{'derived.six_group_accuracy': sum(matrix[i][i] for i in range(6)) / values['count']}))
    write_csv(out / 'all_final_metrics.csv', final)
    for name in ('all_round_metrics.csv', 'per_mutant_all_methods.csv', 'synth_sample_csv_index.csv',
                 'training_history_index.csv', 'experiment_index.csv'):
        rows = [r for r in read_csv(args.source / name) if r['run_id'] in ids]
        write_csv(out / name, rows)
    payload = dict(protocol=inventory['protocol'], experiments=records)
    (out / 'all_metrics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    lines = ['# 三个Ours与八个Baseline：全部评测指标', '',
        '来源：冻结协议seed42_a866aef5d988，FNO/seed42，所有数据直接核对原始终评JSON。', '',
        f'共{len(records)}项已完成实验、{len(final)}组终评。主清单23项，另含PRE-TopK两池补充2项。', '',
        '- HGGS为修正前GF/MGS训练流程的历史结果；持续训练修正版尚未正式重跑，不能将旧成绩标为修正后。',
        '- OGAS只选公式修正后版本。Low-Rank、PEAKS、PRE、HGGS与OGAS均保留任务适配名称。',
        '- Sensitivity仅mutation池选；Closure仅mutation池选/生成；Transport两池均有池选/生成。不存在的组合没有填0。',
        '- 不同持续/生成/一次性方法的训练、标签访问和初始GF成本不同，不是严格等总算力比较。', '',
        '## 指标说明', '',
        '全部下表均在逆归一化后的物理空间评测。主Relative为逐样本逐变量时间L2比值的均值，分母sqrt(||truth||²+0.01²)，不截断。',
        '39变量与核心9变量各保留MAE/MSE/Relative/Correlation。轨迹Mean/Median/P90/P95/Worst10%/GlobalFrob为完整轨迹诊断，权重不同于主Relative。',
        'ValidCorr仅平均有定义的逐样本逐变量时间相关系数；Undefined记录未定义项数。六组准确率由原混淆矩阵派生，DivisionMAE是分裂次数误差。',
        'real为126株；同池test为来源池固定5000条，cross_pool_test为另一个池的固定5000条。',
        '跨池预测可能极大，原值完整保留，没有截断或混入同池成绩。所有原始叶字段及定义文本见CSV/JSON。', '']
    for pool in ('mutation', 'lhs'):
        for cohort in COHORTS:
            selected = [r for r in records if r['pool'] == pool]
            lines += [f'## {pool}训练 → {cohort}', '', '### 39变量及核心9变量', '']
            table(lines, ['方法', '39-MAE↓', '39-MSE↓', '39-Relative↓', '39-Corr↑',
                          '9-MAE↓', '9-MSE↓', '9-Relative↓', '9-Corr↑'],
                  [[label(r)] + [r['metrics'][cohort][group][k] for group in ('all_39_vars', 'core_9_vars')
                                for k in ('MAE', 'MSE', 'Relative L2', 'Correlation')] for r in selected])
            lines += ['### 轨迹、分裂及Pattern诊断', '']
            keys = ('macro_trajectory_rel_l2', 'median_rel_l2', 'p90_rel_l2', 'p95_rel_l2',
                    'worst_10pct_mean_rel_l2', 'global_frobenius_rel_l2', 'valid_channel_correlation',
                    'undefined_channel_correlations')
            table(lines, ['方法', 'Mean↓', 'Median↓', 'P90↓', 'P95↓', 'Worst10%↓', 'GlobalFrob↓',
                          'ValidCorr↑', 'Undefined', 'DivisionMAE↓', '六组准确率↑'],
                  [[label(r)] + [r['metrics'][cohort]['trajectory_metrics'][k] for k in keys] +
                   [r['metrics'][cohort]['pattern_metrics']['division_count_mae'],
                    f"{sum(r['metrics'][cohort]['pattern_metrics']['six_group_confusion'][i][i] for i in range(6))/r['metrics'][cohort]['count']:.2%}"]
                   for r in selected])
            lines += ['### 完整六分类混淆矩阵', '', '行是真值，列是预测；顺序P1/P2/P3/P4/P5a/P5b。', '']
            for r in selected:
                lines += [f'**{label(r)}**', '']
                table(lines, ['真值/预测', 'P1', 'P2', 'P3', 'P4', 'P5a', 'P5b'],
                      [[name] + row for name, row in zip(('P1', 'P2', 'P3', 'P4', 'P5a', 'P5b'),
                       r['metrics'][cohort]['pattern_metrics']['six_group_confusion'])])
    lines += ['## 精确实验目录、预测图与日志', '']
    for r in records:
        lines += [f"- {r['pool']} / {label(r)}：[目录]({r['directory']})；[126株预测图]({r['images']})；[原始JSON]({r['metrics_file']})；[日志]({r['log']})。"]
    lines += ['', '## 导出文件', '',
        '- all_final_metrics.csv：全部终评字段，不舍弃矩阵或指标定义。',
        '- all_metrics.json：完整原始嵌套JSON及原目录/配置/图像路径。',
        '- all_round_metrics.csv：各轮已保存的全部指标。',
        '- per_mutant_all_methods.csv：所选实验全部real逐株结果。',
        '- synth_sample_csv_index.csv：同池test逐样本CSV的原文件位置，不重复复制轨迹。',
        '- experiment_index.csv / training_history_index.csv：实验及训练历史索引。', '']
    (out / 'report.md').write_text('\n'.join(lines))
    audit = dict(experiments=len(records), terminal_cohorts=len(final), unique_methods=len({r['method'] for r in records}),
                 source_json_verified=True, source_sha256=hashes, missing_combinations_not_fabricated=True)
    (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(out / '汇报用_三个Ours与八个Baseline_完整指标.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.suffix != '.zip': z.write(path, path.name)
    print(json.dumps({k: v for k, v in audit.items() if k != 'source_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
