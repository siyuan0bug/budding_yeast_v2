#!/usr/bin/env python3
"""Export completed sampling evaluations without loading trajectories or models."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo


COHORTS=('real_mutants','synth_test','cross_pool_test')
VERSION='physical_sample_channel_rel_l2_smooth_eps001_v3'
ROOTS=('budding_sampling','budding_dynamic','budding_generation',
       'budding_sampling_repairs','budding_ogas_alignment','budding_ours')
METRICS=('MAE','MSE','Relative L2','Correlation')
MAIN=('random','topk','power','lcmd_ll','bait_ll','low_rank','peaks_regression',
      'physics_pre_power','hggs_projected','ogas_async_projected')


def read(path):return json.loads(path.read_text())


def flatten(value,prefix=''):
    if isinstance(value,dict):
        return {key:v for name,item in value.items()
                for key,v in flatten(item,f'{prefix}.{name}' if prefix else name).items()}
    if isinstance(value,list):
        return {key:v for i,item in enumerate(value)
                for key,v in flatten(item,f'{prefix}.{i}').items()}
    return {prefix:value}


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fields);writer.writeheader();writer.writerows(rows)


def cohort_values(value):
    flat=flatten(value)
    matrix=value['pattern_metrics']['six_group_confusion']
    flat['derived.six_group_accuracy']=sum(matrix[i][i] for i in range(6))/value['count']
    return flat


def classify(directory,config):
    method=config.get('method',directory.parts[-2])
    if isinstance(method,dict):method=method['name']
    if config.get('workflow')=='labelled_pool_continuous_regression_v1':return 'continuous_pool'
    if config.get('workflow')=='ogas_async_legal_domain_v1':return 'continuous_generation'
    if config.get('workflow')=='legal_open_generation_round_adaptations_v1':return 'round_generation'
    if config.get('workflow')=='hggs_continuous_gf_mgs_v2':return 'hggs_continuous_generation'
    return 'weighted_one_shot' if method in ('low_rank','clustering_sensitivity') else (
        'one_shot' if method=='lev_input' else 'fixed_pool_rounds')


def completion(directory):
    path=directory/'_state/completion.json'
    if path.exists():return read(path)
    path=directory/'_state/state.json'
    return read(path).get('completion',{}) if path.exists() else {}


def validation_history(directory):
    histories=sorted(directory.glob('rounds/round_*/history.csv')) or sorted(directory.glob('stream/history.csv'))
    rows=[]
    for history in histories:
        data=list(csv.DictReader(history.open()))
        if not data:raise ValueError(f'Empty history: {history}')
        best=min(data,key=lambda row:float(row['val_loss_synth']))
        candidates=([history.parent/'stage.json'] if (history.parent/'stage.json').exists() else
                    [directory/'_state/primary/training.json'] if history.parent.name=='stream' else
                    [directory/'_state'/history.parent.name/'primary/training.json'])
        metadata=read(candidates[0])
        if metadata['best_epoch']!=int(best['epoch']):raise ValueError(f'Best mismatch: {history}')
        if len(data)!=metadata['epochs']:raise ValueError(f'History epoch mismatch: {history}')
        if metadata.get('validation_metric','val_loss_synth')!='val_loss_synth':raise ValueError(f'Unexpected best criterion: {history}')
        rows.append(dict(round=history.parent.name,epochs=len(data),best_epoch=metadata['best_epoch'],
            optimizer_steps=int(metadata.get('optimizer_steps_delta',data[-1]['optimizer_steps'])),history=str(history)))
    return rows


def fmt(v):
    if v is None:return '未定义'
    if isinstance(v,int):return str(v)
    return f'{v:.6g}'


def display(record):
    name=record['method']
    if record['family']=='ours':name+=' [生成]' if record['workflow']=='round_generation' else ' [池选]'
    if name=='ogas_async_projected':name+=' [公式修正后]' if record['version']=='ogas_aligned_20261118' else ' [修正前]'
    if name=='hggs_projected':name+=' [持续训练修正版]' if record['workflow']=='hggs_continuous_generation' else ' [旧训练流程]'
    return name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast'))
    parser.add_argument('--protocol',default='seed42_a866aef5d988')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plots',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    base=args.data_root/'experiments';protocol=read(args.data_root/'benchmarks/budding_sampling'/args.protocol/'protocol.json')
    records=[];inventory=[];final=[];rounds=[];per_mutant=[];history_index=[];issues=[];checks=Counter()
    synth_sources=[];canonical_names=None
    migration=base/'budding_sampling/batches/layout_migration_20261120/plan.json'
    migrated=read(migration) if migration.exists() else []
    by_destination={str(Path(r['destination'])):r for r in migrated}
    canonical={Path(r['destination']).resolve() for r in migrated}
    for root in ROOTS:
        configs={cf for cf in (base/root).glob('**/seed_42/config.json') if cf.parent.resolve() not in canonical}
        configs.update(Path(r['destination'])/'config.json' for r in migrated if Path(r['source']).relative_to(base).parts[0]==root)
        for cf in sorted(configs):
            config=read(cf)
            if args.protocol not in str(cf) and config.get('protocol_fingerprint',config.get('protocol'))!=protocol['fingerprint']:continue
            d=cf.parent;done=completion(d)
            pool=config.get('pool',d.parts[-3]);method=config.get('method',d.parts[-2])
            if isinstance(method,dict):method=method['name']
            workflow=classify(d,config);version='current_protocol'
            if root=='budding_ours':version='ours_20261119'
            elif root=='budding_ogas_alignment':version='ogas_aligned_20261118'
            elif root=='budding_sampling_repairs':version='bugfix_20261118_old_training'
            elif method=='ogas_async_projected':version='ogas_pre_alignment'
            origin=Path(by_destination[str(d)]['source']) if str(d) in by_destination else d
            run_id='__'.join((root,origin.parent.parent.parent.name,pool,method))
            info=dict(run_id=run_id,pool=pool,method=method,version=version,workflow=workflow,
                family='ours' if method.startswith('ours_') else 'baseline',directory=str(d),
                config=str(cf),log=str(d/'train.log'),metrics_file=str(d/'eval/global_metrics_denorm.json'),
                real_csv=str(d/'eval/per_mutant_metrics.csv'),synth_csv=str(d/'eval/per_synth_test_metrics.csv'),
                images=str(d/'eval/real_mutants'),status=done.get('status','incomplete_historical'))
            inventory.append(info)
            if done.get('status')!='complete':continue
            metrics=read(Path(info['metrics_file']))
            train=config.get('training',{})
            assert train.get('model')=='fno' and train.get('relative_l2_eps')==.01
            raw=read(d/'_state/state.json') if (d/'_state/state.json').exists() else {}
            oracle=raw.get('oracle',{})
            for name in ('label_access.json','hggs_label_access.json'):
                f=d/'_state'/name
                if f.exists():oracle=read(f)
            if oracle:
                import numpy as np
                with np.load(args.data_root/'benchmarks/budding_sampling'/args.protocol/f'{pool}_split.npz') as z:
                    allowed=set(map(int,z['candidates']))
                assert set(map(int,oracle['revealed_ids']))<=allowed
                checks['training_side_label_ledgers']+=1
            histories=validation_history(d)
            assert histories and histories[-1]['epochs']==200
            history_index.extend(dict(run_id=run_id,pool=pool,method=method,**row) for row in histories)
            report=dict(info,train_samples=metrics['train_samples'],
                primary_training_steps=sum(x['optimizer_steps'] for x in histories),
                primary_training_epochs=sum(x['epochs'] for x in histories),
                best_epoch=histories[-1]['best_epoch'],
                unique_labels_accessed=done.get('unique_labels_accessed',oracle.get('unique_labels_accessed')),
                new_ode_calls=done.get('new_ode_calls',done.get('solver_attempts')),
                accepted=done.get('accepted'),failed_inputs=done.get('failed_inputs'),
                completion_scalars={k:v for k,v in done.items() if not isinstance(v,(dict,list))},metrics=metrics)
            records.append(report)
            for cohort in COHORTS:
                value=metrics[cohort];flat=cohort_values(value)
                assert value['metric_version']==VERSION and value['metric_space']=='physical'
                assert value['count']==(126 if cohort=='real_mutants' else 5000)
                assert math.isclose(value['sample_channel_rel_l2'],value['all_39_vars']['Relative L2'])
                assert all(math.isfinite(v) for v in flat.values() if isinstance(v,(int,float)))
                matrix=value['pattern_metrics']['six_group_confusion'];assert sum(map(sum,matrix))==value['count']
                row=dict(run_id=run_id,pool=pool,method=method,version=version,workflow=workflow,family=info['family'],
                    cohort=cohort,round=metrics['round'],train_samples=metrics['train_samples'],
                    metrics_file=info['metrics_file'],**flat)
                assert all(row[k]==v for k,v in flatten(value).items())
                final.append(row);checks['final_cohorts']+=1
            for path in sorted(d.glob('rounds/round_*/eval/global_metrics_denorm.json')):
                metric=read(path)
                for cohort in COHORTS:
                    rounds.append(dict(run_id=run_id,pool=pool,method=method,version=version,
                        round=metric['round'],train_samples=metric['train_samples'],cohort=cohort,
                        metrics_file=str(path),**cohort_values(metric[cohort])))
            if (d/'stream/history.csv').exists():
                for cohort in COHORTS:
                    rounds.append(dict(run_id=run_id,pool=pool,method=method,version=version,
                        round='continuous_final',train_samples=metrics['train_samples'],cohort=cohort,
                        metrics_file=info['metrics_file'],**cohort_values(metrics[cohort])))
            samples=list(csv.DictReader(Path(info['real_csv']).open()))
            assert len(samples)==126 and len({r['mutant_name'] for r in samples})==126
            names={r['mutant_name'] for r in samples}
            if canonical_names is None:canonical_names=names
            assert names==canonical_names
            for column,key in [('MAE','MAE'),('MSE','MSE'),('rel_l2','Relative L2')]:
                assert all(math.isfinite(float(s[column])) for s in samples)
                mean=statistics.mean(float(s[column]) for s in samples)
                assert math.isclose(mean,metrics['real_mutants']['all_39_vars'][key],rel_tol=2e-5,abs_tol=1e-7),(d,column)
            per_mutant.extend(dict(run_id=run_id,pool=pool,method=method,version=version,
                workflow=workflow,family=info['family'],**row) for row in samples)
            assert len(list(Path(info['images']).glob('*.png')))==126
            sums=defaultdict(float);count=0
            with Path(info['synth_csv']).open() as handle:
                for row in csv.DictReader(handle):
                    for key in ('MAE','MSE','rel_l2'):
                        value=float(row[key]);assert math.isfinite(value);sums[key]+=value
                    count+=1
            assert count==5000
            for key,column in [('MAE','MAE'),('MSE','MSE'),('rel_l2','Relative L2')]:
                assert math.isclose(sums[key]/count,metrics['synth_test']['all_39_vars'][column],rel_tol=2e-5,abs_tol=1e-7),(d,key)
            synth_sources.append(dict(run_id=run_id,pool=pool,method=method,path=info['synth_csv'],rows=count))
            checks['csv_json_pairs']+=2;checks['real_figures']+=126
    assert len({r['run_id'] for r in records})==len(records)
    # Preserve pre/post alignment variants, but use corrected OGAS for main comparisons.
    preferred=[r for r in records if r['version']!='ogas_pre_alignment']
    write_csv(out/'experiment_index.csv',inventory)
    write_csv(out/'all_final_metrics.csv',final)
    write_csv(out/'all_round_metrics.csv',rounds)
    write_csv(out/'per_mutant_all_methods.csv',per_mutant)
    write_csv(out/'training_history_index.csv',history_index)
    write_csv(out/'synth_sample_csv_index.csv',synth_sources)
    (out/'all_metrics.json').write_text(json.dumps(dict(protocol=protocol,experiments=records),ensure_ascii=False,indent=2)+'\n')
    comparisons=[];case_diagnostics=[];matching=[];paired=[]
    byid={r['run_id']:r for r in records}
    lookup={(r['pool'],r['method'],r['workflow']):r for r in preferred}
    for r in preferred:
        for reference in ('random','power','lcmd_ll'):
            ref=lookup[(r['pool'],reference,'fixed_pool_rounds')]
            for cohort in COHORTS:
                a,b=r['metrics'][cohort]['all_39_vars'],ref['metrics'][cohort]['all_39_vars']
                comparisons.append(dict(run_id=r['run_id'],pool=r['pool'],method=r['method'],workflow=r['workflow'],
                    cohort=cohort,reference=reference,**{f'{m}_change_pct':100*(a[m]/b[m]-1) for m in METRICS}))
            if r['family']=='ours':
                aa={x['mutant_name']:x for x in per_mutant if x['run_id']==r['run_id']}
                bb={x['mutant_name']:x for x in per_mutant if x['run_id']==ref['run_id']}
                gaps=sorted(((name,float(aa[name]['MSE'])-float(bb[name]['MSE'])) for name in aa),key=lambda x:-x[1])
                net=sum(v for _,v in gaps)
                paired.append(dict(run_id=r['run_id'],pool=r['pool'],method=r['method'],workflow=r['workflow'],reference=reference,
                    **{f'{metric}_wins':sum(float(aa[n][metric])<float(bb[n][metric]) for n in aa)
                       for metric in ('MAE','MSE','rel_l2')},
                    top3_mse_regressions=json.dumps(gaps[:3],ensure_ascii=False),
                    top3_share_of_net_mse_regression=None if net<=0 else sum(v for _,v in gaps[:3])/net))
        refname=('random_generate' if r['workflow']=='round_generation' else
                 'random_async' if r['workflow']=='continuous_generation' else
                 'random_stream' if r['method']=='peaks_regression' else None)
        if refname:
            ref=lookup[(r['pool'],refname,r['workflow'])]
            a,b=r['metrics']['real_mutants']['all_39_vars'],ref['metrics']['real_mutants']['all_39_vars']
            matching.append(dict(run_id=r['run_id'],pool=r['pool'],method=r['method'],reference=refname,
                **{f'{m}_change_pct':100*(a[m]/b[m]-1) for m in METRICS}))
        samples=[s for s in per_mutant if s['run_id']==r['run_id']]
        mses=sum(float(s['MSE']) for s in samples);rels=sum(float(s['rel_l2']) for s in samples)
        worst_mse=max(samples,key=lambda s:float(s['MSE']));worst_rel=max(samples,key=lambda s:float(s['rel_l2']))
        case_diagnostics.append(dict(run_id=r['run_id'],pool=r['pool'],method=r['method'],workflow=r['workflow'],
            largest_mse_mutant=worst_mse['mutant_name'],largest_mse_share=float(worst_mse['MSE'])/mses,
            largest_relative_mutant=worst_rel['mutant_name'],largest_relative_share=float(worst_rel['rel_l2'])/rels,
            worst13_relative_share=sum(sorted(float(s['rel_l2']) for s in samples)[-13:])/rels))
    write_csv(out/'vs_fixed_baselines.csv',comparisons)
    write_csv(out/'matched_workflow_controls.csv',matching)
    write_csv(out/'ours_paired_mutant_wins.csv',paired)
    write_csv(out/'badcase_concentration.csv',case_diagnostics)
    preferred_ids={r['run_id'] for r in preferred}
    cases=[r for r in per_mutant if r['run_id'] in preferred_ids]
    case_scores=defaultdict(list)
    for r in preferred:
        subset=[x for x in cases if x['run_id']==r['run_id']]
        for metric in ('MAE','rel_l2'):
            order=sorted(subset,key=lambda x:float(x[metric]))
            for rank,x in enumerate(order,1):case_scores[(x['pool'],x['mutant_name'])].append(rank/126)
    top20=[]
    for pool in ('mutation','lhs'):
        values=[dict(pool=p,mutant_name=name,mean_posthoc_rank=statistics.mean(scores))
                for (p,name),scores in case_scores.items() if p==pool]
        top20.extend(sorted(values,key=lambda x:-x['mean_posthoc_rank'])[:20])
    write_csv(out/'top20_badcases.csv',top20)
    if args.plots:plot_comparison(records,out)
    lines=['# seed42全部采样基线与ours评测汇总（2026-11-19）','',
        f'快照北京时间：{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}。', '',
        f'共找到{len(inventory)}份当前协议实验配置，{len(records)}项完整终评，{len(inventory)-len(records)}项历史未完成记录。本批ours17项全部完成、0失败。最终评测覆盖{len(final)}组队列/测试域；所有源JSON叶字段和混淆矩阵逐项保留。', '',
        '本工具只读标量JSON、逐样本CSV、history与标签ID账本；不加载轨迹或checkpoint，不训练、不生成ODE、不改原结果。', '',
        '## 结论与比较条件','',
        '- mutation综合仍以Power最强：MAE0.15030、MSE2.25805、主Relative1.05698；ours没有全面超过它。',
        '- ours固定池Context在mutation的相关系数0.91469和完整轨迹均值0.28381有优势，但MAE0.18173/MSE7.55667明显较差；完整轨迹指标不是主逐样本逐变量Relative。',
        '- ours固定池Closure是当前新机理方法中较均衡的一项：mutation MAE0.16641/MSE2.73847/Relative1.37550，优于Random的误差，但仍落后Power。',
        '- LHS固定池Pattern Balance的real MAE0.40791略优于Power0.41464，但MSE28.84106和Relative2.61005较差；不能称全面胜出。',
        '- 新生成式ours未表现出相对同流程Random-Generate的稳定综合收益。逐项百分比见matched_workflow_controls.csv；不能用动态/生成的标签和训练成本冒充固定池相同预算。',
        '- Low-Rank、PEAKS、HGGS以及OGAS旧版按原结果提取。HGGS GF/checkpoint/归一化及MGS冷启动问题尚未在生产修复；OGAS同时保留修正前/公式修正后，两者不混名。',
        '- 只有seed42，没有多seed误差条；结论为描述性比较。', '',
        '## 冻结设置','',
        f'协议：`{args.data_root / "benchmarks/budding_sampling" / args.protocol / "protocol.json"}`。',
        '每池10万完整轨迹，9万训练侧、5000 validation、5000同池test；相同来源的普通方法共同随机初始2000。FNO192/4blocks/modes24/batch128，LR5e-4，minLR1e-5，Relative L2 eps0.01、无截断，物理空间终评。',
        '固定池和分轮生成：2k/4k/6k/8k/10k，100/100/100/100/200epoch，逐轮冷启动/refit；Low-Rank和带权Clustering一次性选样；PEAKS/ERACS/异步OGAS为连续200epoch、初始统计冻结。HGGS初始GF不同；Low-Rank不是10000唯一输入。详见配置索引。', '',
        '## 全部指标与文件','',
        '- all_final_metrics.csv：每实验×real/同池test/跨池test，共全部原始叶字段；含39变量及核心9变量四指标、轨迹分位数、相关性、分裂误差和6×6矩阵36格。',
        '- all_metrics.json：保留所有终评原嵌套JSON及完整来源路径，未舍弃字段。',
        '- all_round_metrics.csv：所有已完成实验各轮终评；持续训练仅记录其final，没有伪造五轮。',
        '- experiment_index.csv：包含完成和未完成历史项、版本、流程、config/train.log、JSON/CSV/126幅图片的精确路径。',
        '- per_mutant_all_methods.csv：全部real逐株指标；synth_sample_csv_index.csv指向全部同池test逐样本CSV，不复制轨迹或重复保存源样本。',
        '- training_history_index.csv：逐轮epochs、best epoch、优化步数和原history路径。',
        '- vs_fixed_baselines.csv与matched_workflow_controls.csv：相对Random/Power/LCMD及同流程随机对照的变化百分比。',
        '- badcase_concentration.csv与top20_badcases.csv：仅事后诊断的误差集中情况及top20名单；不进入采样代码。', '',
        '主Relative：逐样本逐变量沿时间求L2比值，再平均，分母sqrt(||truth||²+0.01²)。轨迹均值/Median/P90/P95/Worst10%是完整39×T轨迹相对Frobenius误差的诊断量；与主Relative的权重不同。Corr是每变量拼接全部样本/时间后的Pearson再平均，Valid-Corr则只平均有定义的逐样本逐通道时间Pearson。六组准确率是由原混淆矩阵计算的派生量，矩阵行是真值、列是预测，P1/P2/P3/P4/P5a/P5b。',
        'cross_pool_test仅为另一个池的固定test，不参与best。mutation模型对LHS的跨域值可能很大；原值完整保留，不裁剪、不与同池test混为一项。','']
    lines += ['## ours收益与退步具体发生在哪里','',
        '以下均为事后比较，不改变训练、归一化、checkpoint或采样评分。', '',
        '- mutation固定池Context相对Power有79/126株MAE改善、72/126株MSE改善，但#62、#124、#50的MSE增量分别约255.19、251.45、118.21，合计解释约93.6%的净MSE差额。多数样本改善与整体MSE变差可以同时发生。',
        '- mutation固定池Closure的#15 MAE由Power的0.1377降至0.06317，主Relative由7.420降至0.8513；#124 MSE由58.68降至24.99。但#78、#75、#59等退步，MAE仅59/126株优于Power，整体MAE/MSE/Relative仍落后。',
        '- 固定池Sensitivity的real MAE0.16173较好，但#124的MSE约179.16，约占自身总MSE45.9%，尚未控制高幅值尾部。Pattern Balance在mutation的#15和#124均较差，Pattern总体配额本身不足以保证关键输入响应学对。',
        '- 生成式Transport在LHS的real MAE/MSE相对Random-Generate降低10.18%/14.78%，但主Relative升高20.69%；mutation则MSE升高138.77%。生成并未稳定解决误差分配。',
        '- 同池mutation合成test上，Pattern Support的Relative1.27052、Corr0.84984均为当前固定池方法最好，Context的Relative1.31220也好于Power1.38114；但real上Power更好。因此候选分布内收益不能自动等同于real泛化。',
        '- Witness在mutation同池test的MSE61.6402好于Power73.7357，但real MSE4.1164较差；物理评分/覆盖目标与最终real损失一致性的假设仍缺少验证，当前结果不能证明唯一根因。',
        '- 旧Low-Rank整体较弱；PEAKS对同流程Random-Stream有小幅收益，mutation/LHS MAE改善3.41%/2.96%，但访问约6.8万标签。公式修正后OGAS在LHS相对Random-Async的MAE改善4.38%，mutation MAE变差4.40%；不能据修正后分数更差就恢复错误公式。', '',
        '逐株胜负和最大退步保存为ours_paired_mutant_wins.csv；完整逐株值见per_mutant_all_methods.csv。下一步更有价值的是在合成train/validation上核查采样分数与实际查询误差的关联、权重集中、响应组冗余和极端幅值支撑，避免继续盲目叠加选样项；不把上述real编号直接写入采样算法。','']
    if args.plots:lines += ['![real四指标对比](real_metrics_comparison.png)','']
    for pool in ('mutation','lhs'):
        group=[r for r in records if r['pool']==pool]
        group.sort(key=lambda r:(r['family']=='ours',r['workflow'],r['method'],r['version']))
        for cohort in COHORTS:
            lines += [f'## {pool}训练 / {cohort}','',
                '每项都展示，保留旧版与修正版名称；不同流程并非相同总计算量或标签访问量。', '',
                '|方法|流程|39-MAE↓|39-MSE↓|39-Relative↓|39-Corr↑|9-MAE↓|9-MSE↓|9-Relative↓|9-Corr↑|',
                '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for r in group:
                m=r['metrics'][cohort]
                lines.append('|'+display(r)+'|'+r['workflow']+'|'+'|'.join(fmt(m[k][metric])
                    for k in ('all_39_vars','core_9_vars') for metric in METRICS)+'|')
            lines += ['', '|方法|轨迹均值↓|Median↓|P90↓|P95↓|Worst10%↓|全局Frob↓|Valid-Corr↑|未定义相关项数|分裂MAE↓|六组准确率↑|',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            keys=('macro_trajectory_rel_l2','median_rel_l2','p90_rel_l2','p95_rel_l2',
                'worst_10pct_mean_rel_l2','global_frobenius_rel_l2','valid_channel_correlation','undefined_channel_correlations')
            for r in group:
                m=r['metrics'][cohort];t=m['trajectory_metrics'];p=m['pattern_metrics']
                acc=sum(p['six_group_confusion'][i][i] for i in range(6))/m['count']
                lines.append('|'+display(r)+'|'+'|'.join(fmt(t[k]) for k in keys)+
                    f'|{fmt(p["division_count_mae"])}|{acc:.2%}|')
            lines += ['', '<details><summary>全部六组混淆矩阵（行真值、列预测）</summary>','']
            for r in group:
                lines.extend([f'**{display(r)}**','', '|真值/预测|P1|P2|P3|P4|P5a|P5b|','|---|---:|---:|---:|---:|---:|---:|'])
                for code,row in zip(('P1','P2','P3','P4','P5a','P5b'),r['metrics'][cohort]['pattern_metrics']['six_group_confusion']):
                    lines.append('|'+code+'|'+'|'.join(map(str,row))+'|')
                lines.append('')
            lines += ['</details>','']
    lines += ['## 精确目录与图片索引','', '|池|方法|版本|完整实验目录|预测图片|','|---|---|---|---|---|']
    for r in records:
        lines.append(f'|{r["pool"]}|{display(r)}|{r["version"]}|[实验]({r["directory"]})|[126株图片]({r["images"]})|')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    checks.update(completed_experiments=len(records),incomplete_historical=len(inventory)-len(records),
        final_metric_leaves_preserved=1,round_cohorts=len(rounds),per_mutant_rows=len(per_mutant))
    (out/'audit.json').write_text(json.dumps(dict(checks=dict(checks),issues=issues,
        trajectories_loaded=False,checkpoints_loaded=False),indent=2)+'\n')
    print(json.dumps(dict(output=str(out),**dict(checks)),indent=2))


def plot_comparison(records,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator,FormatStrFormatter,NullLocator
    import numpy as np
    colors={'fixed_baseline':'#627d98','other_baseline':'#a1acb6','fixed_ours':'#1a9988','generated_ours':'#db8c32'}
    fig,axes=plt.subplots(2,4,figsize=(18,14))
    for row,pool in enumerate(('mutation','lhs')):
        group=[r for r in records if r['pool']==pool and (r['method'] in MAIN or r['family']=='ours')]
        group.sort(key=lambda r:(r['family']=='ours',r['workflow']=='round_generation',r['method'],r['version']))
        labels=[r['method'].replace('ours_','')+(' / gen' if r['family']=='ours' and r['workflow']=='round_generation' else '')+
            (' / old' if r['version']=='ogas_pre_alignment' else '') for r in group]
        for col,metric in enumerate(METRICS):
            ax=axes[row,col];vals=[r['metrics']['real_mutants']['all_39_vars'][metric] for r in group]
            barcolors=[colors['generated_ours' if r['family']=='ours' and r['workflow']=='round_generation' else
                'fixed_ours' if r['family']=='ours' else 'fixed_baseline' if r['workflow']=='fixed_pool_rounds' else 'other_baseline'] for r in group]
            ax.barh(np.arange(len(group)),vals,color=barcolors,height=.72);ax.invert_yaxis()
            ax.set_yticks(np.arange(len(group)),labels if col==0 else ['']*len(group),fontsize=8)
            if metric in ('MSE','Relative L2'):
                ax.set_xscale('log')
                lower,upper=min(vals)*.8,max(vals)*1.1
                ax.set_xlim(lower,upper)
                ax.xaxis.set_major_locator(FixedLocator(np.geomspace(lower,upper,4)))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
                ax.xaxis.set_minor_locator(NullLocator())
            ax.set_title(f'{pool} / {metric}'+(' (log scale)' if metric in ('MSE','Relative L2') else ''),fontsize=11)
            ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('FNO, seed 42: physical real-mutant evaluation\nDifferent workflows and label-access costs are shown together for diagnosis',fontsize=15)
    fig.legend(handles=[Patch(color=v,label=k.replace('_',' ')) for k,v in colors.items()],loc='lower center',ncol=4)
    fig.tight_layout(rect=(0,.03,1,.95));fig.savefig(out/'real_metrics_comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
