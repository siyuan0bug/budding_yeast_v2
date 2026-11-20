#!/usr/bin/env python3
"""Audit completed sampling runs and compare loss protocols without loading trajectories."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

NAMES={'random':'Random','topk':'Uncertainty','power':'Power','max_dist_pred':'MaxDist',
       'coreset_ll':'CoreSet','lcmd_ll':'LCMD','bait_ll':'BAIT','vessal_penultimate':'VeSSAL'}
KEYS=('MAE','MSE','RelL2','Corr','macro_RelL2','tail_RelL2','valid_channel_corr','group_accuracy')


def load(path):return json.loads(Path(path).read_text())


def flatten(block):
    if block.get('metric_version','').startswith('physical_sample_channel_rel_l2_'):
        raise ValueError('This historical comparison expects the original pooled Relative L2; '
                         'do not mix it with the new sample/channel definition')
    a,t=block['all_39_vars'],block['trajectory_metrics']
    matrix=block['pattern_metrics']['six_group_confusion']
    return dict(MAE=a['MAE'],MSE=a['MSE'],RelL2=a['Relative L2'],Corr=a['Correlation'],
        macro_RelL2=t['macro_trajectory_rel_l2'],tail_RelL2=t['worst_10pct_mean_rel_l2'],
        valid_channel_corr=t['valid_channel_correlation'],
        group_accuracy=sum(matrix[i][i] for i in range(6))/block['count'])


def csv_write(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def read_cases(path):
    with Path(path).open(newline='') as handle:return {r['mutant_name']:r for r in csv.DictReader(handle)}


def audit_run(directory):
    config=load(directory/'config.json');state=load(directory/'_state/state.json')
    if (state.get('completion') or {}).get('status')!='complete':raise ValueError(f'Incomplete run: {directory}')
    if state['completion']['unique_training_inputs']!=10000:raise ValueError('Wrong final budget')
    if state['completion']['new_ode_calls']!=0:raise ValueError('Unexpected online ODE calls')
    metrics=load(directory/'eval/global_metrics_denorm.json')
    if (metrics['round'],metrics['train_samples'],metrics['real_mutants']['count'],metrics['synth_test']['count'])!=(4,10000,126,5000):
        raise ValueError('Unexpected final evaluation cohort')
    if not all(math.isfinite(v) for v in flatten(metrics['real_mutants']).values()):raise ValueError('Nonfinite metrics')
    checked_images=0
    for i,epochs in enumerate(config['training']['epochs']):
        rd=directory/f'rounds/round_{i:02d}'
        if not (rd/'model.ckpt').is_file() or not (rd/'loss.png').is_file():raise ValueError('Missing round checkpoint/figure')
        with (rd/'history.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        if [int(r['epoch']) for r in rows]!=list(range(1,epochs+1)):raise ValueError('Epoch history is incomplete')
        if len(load(rd/'dataset.json')['rows'])!=config['training']['budgets'][i]:raise ValueError('Wrong round cohort')
        for name,count in (('per_mutant_metrics.csv',126),('per_synth_test_metrics.csv',5000)):
            with (rd/'eval'/name).open(newline='') as handle:
                if sum(1 for _ in csv.DictReader(handle))!=count:raise ValueError(f'Incomplete {name}')
        images=len(list((rd/'eval/real_mutants').glob('*.png')))
        if images!=126:raise ValueError('Incomplete real-mutant figures')
        checked_images+=images
    text=(directory/'train.log').read_text(errors='replace')
    if 'Traceback (most recent call last)' in text:raise ValueError(f'Traceback in {directory}')
    return config,metrics,dict(status='complete',rounds=5,images=checked_images,tracebacks=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root',type=Path,required=True)
    parser.add_argument('--new-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[];pairs=[];case_changes=[];audits=[];fairness=[];grouped={}
    for pool in ('mutation','lhs'):
        for directory in sorted((args.new_root/pool).glob('*/seed_42')):
            method=directory.parent.name
            config,metrics,audit=audit_run(directory)
            audits.append(dict(pool=pool,method=method,**audit))
            current=flatten(metrics['real_mutants'])
            completion=load(directory/'_state/state.json')['completion']
            rows.append(dict(pool=pool,method=method,**current,
                synth_MAE=metrics['synth_test']['all_39_vars']['MAE'],
                synth_RelL2=metrics['synth_test']['all_39_vars']['Relative L2'],
                invocation_seconds=completion['invocation_seconds']))
            old_dir=args.old_root/pool/method/'seed_42'
            if not old_dir.exists():continue
            previous_config,previous,_=audit_run(old_dir)
            diffs={key:[previous_config['training'][key],value] for key,value in config['training'].items()
                   if previous_config['training'][key]!=value}
            if set(diffs)!={'loss'}:raise ValueError(f'Uncontrolled training config changes: {diffs}')
            if previous_config['real_reference']!=config['real_reference']:raise ValueError('Real reference changed')
            same=[];jaccard=[]
            for i in range(5):
                a=load(old_dir/f'rounds/round_{i:02d}/dataset.json')['rows']
                b=load(directory/f'rounds/round_{i:02d}/dataset.json')['rows']
                same.append(a==b);jaccard.append(len(set(a)&set(b))/len(set(a)|set(b)))
            if not same[0] or (method=='random' and not all(same)):raise ValueError('Shared initial/random data changed')
            fairness.append(dict(pool=pool,method=method,training_config_diff=diffs,
                identical_order_by_round=same,jaccard_by_round=jaccard))
            old=flatten(previous['real_mutants'])
            paired=dict(pool=pool,method=method)
            for key in KEYS:
                paired[key+'_old']=old[key];paired[key+'_new']=current[key]
                paired[key+'_delta']=current[key]-old[key]
                paired[key+'_pct']=100*(current[key]/old[key]-1) if old[key] else None
            pairs.append(paired);grouped.setdefault(pool,[]).append((old,current))
            a,b=read_cases(old_dir/'eval/per_mutant_metrics.csv'),read_cases(directory/'eval/per_mutant_metrics.csv')
            if set(a)!=set(b) or len(a)!=126:raise ValueError('Real mutant identities changed')
            for name in a:
                item=dict(pool=pool,method=method,mutant_name=name)
                for key in ('MAE','MSE','trajectory_rel_l2','correlation'):
                    item[key+'_old']=float(a[name][key]);item[key+'_new']=float(b[name][key])
                    item[key+'_delta']=item[key+'_new']-item[key+'_old']
                case_changes.append(item)
    tasks=load(args.new_root/'batch/tasks.json')
    if len(tasks)!=16 or any(v['exit_code'] for v in tasks):raise ValueError('Batch contains missing/failed tasks')
    if len(rows)!=16 or len(pairs)!=8:raise ValueError('Unexpected comparison population')
    averages={}
    for pool,values in grouped.items():
        averages[pool]={key:dict(old=statistics.mean(a[key] for a,b in values),
            new=statistics.mean(b[key] for a,b in values),
            improved=sum((b[key]>a[key] if key in ('Corr','valid_channel_corr','group_accuracy') else b[key]<a[key]) for a,b in values)) for key in KEYS}
    csv_write(args.output/'final_metrics.csv',rows);csv_write(args.output/'paired_comparison.csv',pairs)
    csv_write(args.output/'per_mutant_changes.csv',case_changes)
    resource=dict(peak_pss_gib=0.,minimum_available_gib=float('inf'),actions=set())
    with (args.new_root/'batch/resource_samples.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line);resource['peak_pss_gib']=max(resource['peak_pss_gib'],r['pss_gib'])
            resource['minimum_available_gib']=min(resource['minimum_available_gib'],r['available_gib'])
            resource['actions'].add(r['watchdog_action'])
    resource['actions']=sorted(resource['actions'])
    audit=dict(runs=audits,paired_training_checks=fairness,resource=resource,averages=averages,
               completed=16,exit_codes_all_zero=True,seed=42,figure_check='counts/existence; PNG bytes not decoded',
               note='Adaptive methods change selected data after round 0; Random is the fixed-data loss comparison.')
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2))
    lines=['# Relative L2 与 Hybrid 损失对比（2026-11-15）','',
        f'旧结果：`{args.old_root}`；新结果：`{args.new_root}`。',
        '','16项全部完成，80轮评测与10080幅real图的数量检查通过，16项退出码均0，无Traceback。',
        f"进程树PSS峰值{resource['peak_pss_gib']:.2f}GiB，未触发watchdog（动作：{resource['actions']}）。",'',
        '所有表均为最后轮10000训练样本、126株real、seed42。旧损失MSE+0.2×RelativeL2，新损失纯RelativeL2。',
        '只配对原来存在的4方法×2池。新增4方法无旧结果，不混入损失前后均值。',
        '配置差异仅loss；两池Random五轮训练ID及顺序全部相同。其他自适应方法D0相同，后续选样随模型变化；它们衡量整条采样训练流程的结果。',
        '新流程还新增逐epoch测试监控并保存/恢复RNG；不影响梯度或checkpoint选择，但墙钟时间不能当作单纯loss计算速度比较。','',
        '## 配对结果：旧 → 新','', '| 池 | 方法 | MAE↓ | MSE↓ | 历史RelL2↓ | 历史Corr↑ | macro轨迹RelL2↓ |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in pairs:
        cells=[f"{r[k+'_old']:.4f} → {r[k+'_new']:.4f}" for k in ('MAE','MSE','RelL2','Corr','macro_RelL2')]
        lines.append('| '+r['pool']+' | '+NAMES[r['method']]+' | '+' | '.join(cells)+' |')
    lines+=['','## 四个配对方法的均值（描述统计）','',
            '| 池 | 指标 | 旧 | 新 | 相对变化 | 改善组数 |','|---|---|---:|---:|---:|---:|']
    for pool,values in averages.items():
        for key,v in values.items():lines.append(f"| {pool} | {key} | {v['old']:.5f} | {v['new']:.5f} | {100*(v['new']/v['old']-1):+.2f}% | {v['improved']}/4 |")
    lines+=['','## 新批次全部方法','', '| 池 | 方法 | MAE↓ | MSE↓ | 历史RelL2↓ | 历史Corr↑ | macro轨迹RelL2↓ |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:(r['pool'],r['MAE'])):
        lines.append('| '+r['pool']+' | '+NAMES[r['method']]+' | '+' | '.join(f'{r[k]:.5f}' for k in ('MAE','MSE','RelL2','Corr','macro_RelL2'))+' |')
    lines+=['','## 解释及结论','',
        '1. mutation池纯RelativeL2有MAE/MSE/macro改善，但历史RelL2与历史Correlation全部退化；LHS没有稳定综合收益。不能称为全面优于Hybrid。',
        '2. 训练RelativeL2是归一化空间的逐样本逐变量比值并截到10；历史物理评测RelL2是逐变量拼接全部样本/时间再计算，采用不同分母保护；macro则逐样本汇总完整39变量。名称相似但目标不同。',
        '3. 历史Correlation把同一变量的所有样本/时间拼接；逐轨迹非退化通道Correlation反而两池都改善。不能仅由前者下降推断每条波形都变差；跨样本幅值关系与通道权重也在变化。',
        '4. 去掉MSE改变了优化权重，可能影响幅值校准，这是机制解释而非已经完成因果证明。自适应方法还改变后续查询。',
        '5. mutation+LCMD的MSE明显下降，主要改善包括#124/#50/#62；但#22仍退化，不能宣称badcase已消除。',
        '6. 只有seed42；均值为四个不同方法的描述汇总，不是多seed统计显著性。根据预先确定的主要指标选择，并保留两组结果；本分析不修改默认损失或启动新实验。','']
    (args.output/'report.md').write_text('\n'.join(lines))
    plot(pairs,args.output/'comparison.png')
    print(json.dumps(dict(completed=16,paired=8,resource=resource,averages=averages,output=str(args.output)),indent=2))


def plot(pairs,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    methods=('random','topk','lcmd_ll','bait_ll')
    for row,pool in enumerate(('mutation','lhs')):
        lookup={r['method']:r for r in pairs if r['pool']==pool}
        for col,key in enumerate(('MAE','MSE','RelL2','Corr')):
            ax=axes[row,col];x=np.arange(len(methods))
            a=[lookup[m][key+'_old'] for m in methods];b=[lookup[m][key+'_new'] for m in methods]
            ax.bar(x-.19,a,.38,label='Hybrid: MSE + 0.2 RelL2',color='#1565c0')
            ax.bar(x+.19,b,.38,label='Pure Relative L2',color='#ef6c00')
            ax.set_xticks(x,[NAMES[m] for m in methods],rotation=18)
            ax.set_title(f'{pool}: {key}'+(' (higher better)' if key=='Corr' else ' (lower better)'))
            ax.set_ylim(0,1 if key=='Corr' else max(a+b)*1.2);ax.grid(axis='y',alpha=.2)
            if row==0 and col==0:ax.legend(fontsize=8)
    fig.suptitle('Same pool/split/seed42/FNO/final 10k: real-mutant physical metrics',fontsize=14)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


if __name__=='__main__':main()
