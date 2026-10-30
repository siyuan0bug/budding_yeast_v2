"""Read-only audit of a manifest-selected AL batch, including incomplete runs.

Never fits models, loads large datasets, changes live runs or uses real metrics
to select checkpoints. Historical protocol comparisons are descriptive only.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--health', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, health = read_json(args.manifest), read_json(args.health)
    status_map = {r['strategy']: r for r in health['tasks']}
    states, rounds, mutants, convergence, historical, integrity = [], [], [], [], [], []
    query_rows = []
    final_frames = {}
    for task in manifest['tasks']:
        strategy = task['strategy']
        run = Path(task['experiment_dir'])
        state = status_map[strategy]
        records_path = run / 'summary/rounds_summary.json'
        records = read_json(records_path) if records_path.exists() else []
        log = (run / 'train.log').read_text(errors='replace')
        state=dict(state)
        if state.get('pid') and state['stage'] not in ('error','finished'):
            starts=re.findall(r'\[START\] ([^|\r\n]+)',log)
            latest=starts[-1].strip() if starts else ''
            for fragment,label in [('ODE','simulating'),('ensemble','ensemble'),
                ('candidate scoring','scoring'),('evaluation','evaluating'),
                ('snapshot','saving'),('summary','saving'),('dataset append','appending'),
                ('normalization','preparing'),('main model training','training')]:
                if fragment in latest:
                    state['stage']=label
                    break
        elapsed = re.findall(r'\[END\] experiment \| elapsed_s=([\d.]+)', log)
        exits = re.findall(r'END GPU=.*?exit=(\d+)', log)
        states.append(dict(strategy=strategy, gpu=state['gpu'], stage=state['stage'],
                           round=state['round'], epoch=state['epoch'], pid=state['pid'],
                           completed_rounds=len(records),
                           exit_code=int(exits[-1]) if exits else None,
                           elapsed_seconds=float(elapsed[-1]) if elapsed else None,
                           error='; '.join(state['errors']), run=str(run)))
        for path in sorted(run.glob('acquisitions/after_round_*/query_design.json')):
            # An actively written artifact is excluded until a later snapshot.
            try:
                design=read_json(path)
            except json.JSONDecodeError:
                continue
            for batch_index,batch in enumerate(design.get('batches',[])):
                choices=batch.get('choices',[])
                counts=Counter(x['kind'] for x in choices)
                costs=Counter()
                for choice in choices:
                    costs[choice['kind']]+=choice.get('new_queries',0)
                query_rows.append(dict(strategy=strategy, acquisition=path.parent.name,
                    batch=batch_index, action_counts=json.dumps(counts),
                    planned_endpoint_costs=json.dumps(costs),
                    feedback=json.dumps(batch.get('feedback',{}))))
        for record in records:
            rounds.append(dict(strategy=strategy, **record))
            folder = run / 'rounds' / f'round_{record["round"]:02d}'
            metrics = read_json(folder / 'eval/metrics_denorm.json')
            for item in metrics['per_mutant']:
                mutants.append(dict(strategy=strategy, round=record['round'],
                                    train_samples=record['train_samples'], **item))
            report = read_json(folder / 'convergence.json')
            history = pd.read_csv(folder / 'training_history.csv')
            final = report.get('final') or {}
            if len(history) != record['epochs']:
                raise ValueError(f'Completed round has partial history: {folder}')
            if not np.isclose(metrics['real_mutants']['all_variables']['MAE'],
                              record['real_mutants_mae']):
                raise ValueError(f'Metric summary mismatch: {folder}')
            convergence.append(dict(strategy=strategy, round=record['round'],
                epochs=len(history), status=report['status'],
                best_val_epoch=int(history.loc[history.val_synth.idxmin(), 'epochs']),
                final_vs_best_val_pct=100*(history.val_synth.iloc[-1]/history.val_synth.min()-1),
                **final))
            integrity.append(dict(strategy=strategy, round=record['round'],
                                  history_rows=len(history), real_count=len(metrics['per_mutant']),
                                  checkpoint_exists=(folder/'model.ckpt').exists()))
        if state['stage'] == 'finished':
            if len(records) != 4 or not exits or exits[-1] != '0':
                raise ValueError(f'Completion not confirmed: {run}')
            final_frames[strategy] = pd.read_csv(run/'rounds/round_03/training_history.csv')
            # Exact former short-budget cohort; do not mix balanced/range runs.
            old = run.parent.parent / 'w192_n4_m24_b512_e50-100_lr1e-3_wm4-8_al4_refit_gene_ranges' / 'seed_42'
            old_path = old / 'summary/rounds_summary.json'
            if old_path.exists():
                previous = read_json(old_path)[-1]
                for metric in ('mae', 'mse', 'rel_l2', 'corr'):
                    key = 'real_mutants_'+metric
                    historical.append(dict(strategy=strategy, metric=metric,
                        old=previous[key], current=records[-1][key],
                        change_pct=100*(records[-1][key]/previous[key]-1),
                        old_run=str(old), current_run=str(run),
                        caveat='Epochs, LR floor and round RNG policy changed; acquired data may differ'))

    state_frame, frame = pd.DataFrame(states), pd.DataFrame(rounds)
    per_mutant, conv = pd.DataFrame(mutants), pd.DataFrame(convergence)
    paired = []
    if final_frames:
        reference = per_mutant[(per_mutant.strategy == 'al4pde_random') & (per_mutant['round'] == 3)].set_index('Mutant')
        for strategy in final_frames:
            local = per_mutant[(per_mutant.strategy == strategy) & (per_mutant['round'] == 3)].set_index('Mutant')
            if set(local.index) != set(reference.index):
                raise ValueError('Different real benchmark identities')
            local = local.loc[reference.index]
            for metric in ('MAE','MSE','Relative_L2','Correlation'):
                difference = local[metric] - reference[metric]
                better = difference > 0 if metric == 'Correlation' else difference < 0
                paired.append(dict(strategy=strategy, metric=metric, better=int(better.sum()),
                    worse=int((difference != 0).sum()-better.sum()),
                    mean=float(local[metric].mean()), median=float(local[metric].median()),
                    maximum=float(local[metric].max())))
        rank = reference[['MAE', 'Relative_L2']].rank(pct=True).mean(axis=1)
        top20 = rank.nlargest(20).index
        per_mutant[per_mutant.Mutant.isin(top20)].to_csv(args.output/'random_final_top20_all_methods.csv',index=False)
        final = per_mutant[per_mutant['round']==3].pivot(index='Mutant',columns='strategy',values=['MAE','MSE','Relative_L2'])
        final.to_csv(args.output/'completed_final_mutant_comparison.csv')

    for name, data in [('status',state_frame), ('round_metrics',frame), ('per_mutant',per_mutant),
                       ('convergence',conv), ('historical_protocol_comparison',pd.DataFrame(historical)),
                       ('query_design_summary',pd.DataFrame(query_rows)),
                       ('paired_completed_vs_random',pd.DataFrame(paired))]:
        data.to_csv(args.output/(name+'.csv'),index=False)
    now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    (args.output/'snapshot.json').write_text(json.dumps(dict(checked=now,
        health_checked=health['checked'], stages=dict(Counter(x['stage'] for x in states)), status=states,
        integrity=integrity, caveat='Single seed; incomplete runs are not final rankings'),indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(15,6),constrained_layout=True)
    for strategy, local in frame.groupby('strategy',sort=False):
        local=local.sort_values('train_samples')
        for ax,key in zip(axes,('real_mutants_mae','real_mutants_rel_l2')):
            ax.plot(local.train_samples/1000,local[key],marker='o',ms=4,
                    label=strategy.removeprefix('al4pde_'))
    for ax,title in zip(axes,('Real MAE (physical units)','Real relative L2 (existing pooled metric)')):
        ax.set(xlabel='Synthetic training samples (thousands)',ylabel=title)
        ax.grid(alpha=.25)
    axes[1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
    fig.suptitle('Long-budget batch: completed rounds only; truncated curves are unfinished')
    fig.savefig(args.output/'round_comparison.png',dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(len(final_frames),2,figsize=(11,3*len(final_frames)),squeeze=False,constrained_layout=True)
    for row,(strategy,history) in enumerate(final_frames.items()):
        for col,(key,label) in enumerate((('train_loss','Training hybrid loss'),('val_synth','Validation MSE'))):
            tail=history[history.epochs>=150]
            axes[row,col].plot(tail.epochs,100*tail[key]/tail[key].iloc[0])
            axes[row,col].set(title=strategy+' — '+label,xlabel='Epoch',ylabel='Epoch 150 = 100')
            axes[row,col].grid(alpha=.25)
    fig.savefig(args.output/'completed_loss_tails.png',dpi=170)
    plt.close(fig)
    lines=[f'# 长程 AL 进度审计：{now}', '',
        f'状态：{health["stages"]}。读取既有日志、JSON和小型CSV，未改动训练或正式数据。', '',
        '## 进度', '', state_frame[['strategy','stage','round','epoch','completed_rounds','exit_code']].to_markdown(index=False),
        '', '## 相同训练预算的逐轮指标', '',
        frame[['strategy','train_samples','real_mutants_mae','real_mutants_mse','real_mutants_rel_l2','real_mutants_corr']].to_markdown(index=False,floatfmt='.6f'),
        '', '## 已完成轮次的 loss 变化率', '',
        conv[['strategy','round','train_trend_pct','val_trend_pct','status']].to_markdown(index=False,floatfmt='.6f'),
        '', '## 与历史短预算结果的描述性比较', '',
        pd.DataFrame(historical).drop(columns=['old_run','current_run','caveat'],errors='ignore').to_markdown(index=False,floatfmt='.6f'),
        '', '变更同时包含训练轮数、最低学习率、固定round seed和不同后续采集数据，不能把全部变化归因于epoch。',
        'real指标用于事后报告；未据此选择checkpoint、调参或停止训练。单seed不能支持显著性结论。',
        '最后10轮变化率小于0.1%只满足一次检查；当前监控要求连续三次，且小学习率下的平台不证明收敛。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(state_frame[['strategy','stage','completed_rounds']].to_string(index=False))
    print(frame[frame['round']==3][['strategy','real_mutants_mae','real_mutants_mse','real_mutants_rel_l2']].to_string(index=False))
    print(args.output/'REPORT.md')


if __name__ == '__main__':
    main()
