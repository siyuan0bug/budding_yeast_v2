"""Audit terminal FNO metrics, aggregate paired seeds and diagnose ours without training."""
import argparse,csv,json,hashlib,statistics,math
from pathlib import Path

BASELINES=['random','power','lcmd','physics_pre','low_rank','peaks','ogas']
KEYS=['MAE','MSE','Relative L2','Correlation']

def flatten(x,prefix=''):
    if isinstance(x,dict):
        return {k:v for a,b in x.items() for k,v in flatten(b,f'{prefix}.{a}' if prefix else a).items()}
    return {prefix:json.dumps(x,ensure_ascii=False) if isinstance(x,list) else x}

def csvout(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records=[];flat=[];samples=[];diagnostics=[]
    for p in sorted((a.root/'fno').glob('*/*/*/seed_*/eval/global_metrics_denorm.json')):
        d=p.parent.parent;pool,method,profile,seed=d.relative_to(a.root/'fno').parts
        v=json.loads(p.read_text());c=json.loads((d/'config.json').read_text());comp=d/'_state/completion.json'
        done=json.loads(comp.read_text()) if comp.exists() else {}
        identity=dict(pool=pool,method=method,profile=profile,seed=int(seed[5:]),directory=str(d),metrics_file=str(p))
        assert v['real_mutants']['count']==126
        for cohort in ('real_mutants','synth_test','cross_pool_test'):
            vals=v[cohort]
            assert all(math.isfinite(vals['all_39_vars'][k]) for k in KEYS)
            flat.append(dict(**identity,cohort=cohort,**flatten(vals)))
        r=dict(**identity,round=v.get('round'),train_samples=v.get('train_samples'),metrics=v,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),training=c.get('training'),
            completion={k:x for k,x in done.items() if not isinstance(x,(dict,list))},
            workflow=c.get('workflow','round_pool'),config=c)
        records.append(r)
        sample=d/'eval/per_mutant_metrics.csv'
        if sample.exists():
            with sample.open(encoding='utf-8-sig',newline='') as f:
                samples.extend(dict(**identity,**x) for x in csv.DictReader(f))
        for sp in sorted(d.rglob('selection.json')):
            sv=json.loads(sp.read_text());diag=sv.get('diagnostics',{})
            diagnostics.append({**identity,'selection':str(sp),'selection_seconds':sv.get('selection_seconds'),
                **{('diagnostic_'+k):x for k,x in diag.items() if isinstance(x,(str,int,float,bool))}})
    csvout(a.output/'all_final_metrics.csv',flat);csvout(a.output/'per_mutant_metrics.csv',samples);csvout(a.output/'selection_diagnostics.csv',diagnostics)
    # Export original nested metrics, configurations and hashes for exact traceability.
    (a.output/'inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    aggregates=[]
    for pool in ('mutation','lhs'):
        for method in sorted({r['method'] for r in records if r['pool']==pool}):
            group=[r for r in records if r['pool']==pool and r['method']==method]
            assert len({r['seed'] for r in group})==len(group),'Multiple profiles require explicit stratification'
            for cohort in ('real_mutants','synth_test','cross_pool_test'):
                item=dict(pool=pool,method=method,cohort=cohort,n=len(group),seeds=','.join(str(r['seed']) for r in group))
                for k in KEYS:
                    x=[r['metrics'][cohort]['all_39_vars'][k] for r in group];item[k+'_mean']=statistics.mean(x);item[k+'_std']=statistics.stdev(x) if len(x)>1 else None
                aggregates.append(item)
    csvout(a.output/'mean_std.csv',aggregates)
    # Paired mutant comparisons at seed42 only: no unseen-seed extrapolation for ours.
    pairs=[]
    for pool in ('mutation','lhs'):
        power={r['mutant_name']:r for r in samples if r['pool']==pool and r['method']=='power' and r['seed']==42}
        for method in sorted({r['method'] for r in samples if r['method'].startswith('ours_')}):
            ours=[r for r in samples if r['pool']==pool and r['method']==method and r['seed']==42]
            for r in ours:
                b=power.get(r['mutant_name'])
                if b:
                    pairs.append(dict(pool=pool,method=method,mutant_name=r['mutant_name'],pattern_group=r.get('pattern_group'),
                        **{k+'_ours':float(r[k]) for k in ('MAE','MSE','rel_l2')},
                        **{k+'_power':float(b[k]) for k in ('MAE','MSE','rel_l2')},
                        **{k+'_delta':float(r[k])-float(b[k]) for k in ('MAE','MSE','rel_l2')}))
    csvout(a.output/'ours_vs_power_per_mutant_seed42.csv',pairs)
    lines=['# FNO采样多seed终评分析（2026-11-22）','','所有数值读取各实验eval/global_metrics_denorm.json；只取根eval的终评，不把中间轮当独立实验。mean±sample std，不是标准误。原始所有指标见CSV/JSON。', '',
        '## 完整性及可比性','','七类baseline × 两池 × 3seed均存在终评JSON；ours仅seed42。Low-Rank按带放回抽样权重训练，唯一输入可能不足10000；PEAKS及OGAS为持续训练，不能视为相同算力的五轮冷启动。',
        'c4队列seed1024中PEAKS/OGAS四项exit1为目录锁拒绝重复启动；目标目录另有complete凭据及终评。不能将该失败账本当作成功任务账本，但已有完整结果无需据此重训。', '',
        '评测均为物理空间；Relative是逐样本逐变量时间Relative L2、eps0.01、无截断。不同seed改变split/初始集/模型/采样随机流，固定pool生成seed42。测试集上的差异仅作事后诊断，不能据此调参选epoch。', '']
    for pool in ('mutation','lhs'):
        lines += [f'## {pool}：baseline real mutant三seed统计','','|方法|n|MAE↓|MSE↓|Relative L2↓|Corr↑|','|---|---|---|---|---|---|']
        for r in aggregates:
            if r['pool']==pool and r['cohort']=='real_mutants' and r['method'] in BASELINES:
                lines.append('|'+ '|'.join([r['method'],str(r['n'])]+[f"{r[k+'_mean']:.5f} ± {r[k+'_std']:.5f}" for k in KEYS])+'|')
        lines+=['',f'## {pool}：各单次实验real终评（含其它历史baseline/ours）','','|方法|seed|训练唯一样本/行数报告值|MAE|MSE|Relative|Corr|','|---|---|---|---|---|---|---|']
        for r in records:
            if r['pool']==pool:
                lines.append('|'+ '|'.join([r['method'],str(r['seed']),str(r['train_samples'])]+[f"{r['metrics']['real_mutants']['all_39_vars'][k]:.5f}" for k in KEYS])+'|')
        lines+=['']
    lines+=['## Ours同seed逐株比较（对Power，seed42）','','|池|方法|MAE胜/126|Relative胜/126|平均MAE差(ours-Power)|','|---|---|---|---|---|']
    for pool,method in sorted({(r['pool'],r['method']) for r in pairs}):
        group=[r for r in pairs if r['pool']==pool and r['method']==method]
        lines.append(f"|{pool}|{method}|{sum(r['MAE_delta']<0 for r in group)}/{len(group)}|{sum(r['rel_l2_delta']<0 for r in group)}/{len(group)}|{statistics.mean(r['MAE_delta'] for r in group):.5f}|")
    lines+=['','## 输出文件','','- all_final_metrics.csv：全部93项实验、三种测试集合的全部原始字段（含39/9变量、轨迹、Pattern矩阵）。','- inventory.json：原始metrics、config、完成账本、文件SHA及完整目录。','- mean_std.csv：全部方法按池/测试集合的均值和标准差；n=1不伪造std。','- per_mutant_metrics.csv、ours_vs_power_per_mutant_seed42.csv：逐株原值和配对差。','- selection_diagnostics.csv：选样耗时与机制实际触发情况。','']
    (a.output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(experiments=len(records),baseline_runs=sum(r['method'] in BASELINES for r in records),all_metrics_rows=len(flat),per_mutant_rows=len(samples)),ensure_ascii=False))
    for r in aggregates:
        if r['method'] in BASELINES and r['cohort']=='real_mutants':print(r['pool'],r['method'],r['n'],*[f"{r[k+'_mean']:.4f}±{r[k+'_std']:.4f}" for k in KEYS])

if __name__=='__main__':main()
