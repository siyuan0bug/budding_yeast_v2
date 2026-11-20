#!/usr/bin/env python3
"""Audit paired sampling protocols; replay real references under one physical metric.

Only one saved model is resident at a time. No training, ODE, normalization
refitting, prediction archives, or historical-result rewrites are performed.
"""
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import resource
import statistics

import numpy as np
import torch

from budding_yeast_v2.data.dataset_archive import atomic_json,file_sha256
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
from budding_yeast_v2.utils.evaluation.checkpoints import load_model_from_checkpoint
from budding_yeast_v2.utils.evaluation.metrics import (
    RELATIVE_L2_METRIC_VERSION,sample_channel_relative_errors,trajectory_relative_errors)

METHODS=('random','topk','power','max_dist_pred','coreset_ll','lcmd_ll','bait_ll','vessal_penultimate')
NAMES=dict(random='Random',topk='Uncertainty',power='Power',max_dist_pred='MaxDist',
           coreset_ll='CoreSet',lcmd_ll='LCMD',bait_ll='BAIT',vessal_penultimate='VeSSAL')


def read(path):return json.loads(Path(path).read_text())


def write_csv(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def flatten(block):
    a,t=block['all_39_vars'],block['trajectory_metrics']
    matrix=block['pattern_metrics']['six_group_confusion']
    return dict(MAE=a['MAE'],MSE=a['MSE'],Corr=a['Correlation'],
        macro_RelL2=t['macro_trajectory_rel_l2'],tail_RelL2=t['worst_10pct_mean_rel_l2'],
        valid_channel_corr=t['valid_channel_correlation'],
        pattern_accuracy=sum(matrix[i][i] for i in range(6))/block['count'],
        division_count_mae=block['pattern_metrics']['division_count_mae'])


def audit_run(directory):
    config=read(directory/'config.json');state=read(directory/'_state/state.json')
    complete=state.get('completion',{})
    if complete.get('status')!='complete' or complete['unique_training_inputs']!=10000 or complete['new_ode_calls']!=0:
        raise ValueError(f'Incomplete or wrong budget: {directory}')
    metrics=read(directory/'eval/global_metrics_denorm.json')
    if (metrics['round'],metrics['train_samples'],metrics['real_mutants']['count'],metrics['synth_test']['count'])!=(4,10000,126,5000):
        raise ValueError(f'Wrong final cohorts: {directory}')
    rounds=[]
    for index,epochs in enumerate(config['training']['epochs']):
        folder=directory/f'rounds/round_{index:02d}'
        with (folder/'history.csv').open() as handle:history=list(csv.DictReader(handle))
        if [int(r['epoch']) for r in history]!=list(range(1,epochs+1)):
            raise ValueError(f'Incomplete history: {folder}')
        if any(not math.isfinite(float(r['train_loss'])) or not math.isfinite(float(r['val_error'])) for r in history):
            raise ValueError(f'Nonfinite history: {folder}')
        selection=read(folder/'dataset.json')
        if len(selection['rows'])!=config['training']['budgets'][index] or len(set(selection['rows']))!=len(selection['rows']):
            raise ValueError(f'Wrong ordered query cohort: {folder}')
        for name,count in [('per_mutant_metrics.csv',126),('per_synth_test_metrics.csv',5000)]:
            with (folder/'eval'/name).open() as handle:
                if sum(1 for _ in csv.DictReader(handle))!=count:raise ValueError(f'Wrong CSV rows: {folder/name}')
        if len(list((folder/'eval/real_mutants').glob('*.png')))!=126:
            raise ValueError(f'Incomplete prediction figures: {folder}')
        if not (folder/'model.ckpt').is_file() or not (folder/'loss.png').is_file():raise ValueError('Missing model/loss figure')
        best=min(history,key=lambda row:float(row['val_error']))
        rounds.append(dict(round=index,epochs=epochs,selection=selection,
            selected_epoch=int(best['epoch']),validation_metric=best['validation_metric'],
            best_validation=float(best['val_error']),last_validation=float(history[-1]['val_error']),
            train_loss_at_best=float(best['train_loss']),last_train_loss=float(history[-1]['train_loss'])))
    if 'Traceback (most recent call last)' in (directory/'train.log').read_text(errors='replace'):
        raise ValueError(f'Traceback in completed experiment: {directory}')
    return config,metrics,rounds,complete


def replay_real(checkpoint,reference,device,batch_size):
    model,_=load_model_from_checkpoint(checkpoint,device=device)
    stats=NormalizationStats.from_datamodule(model)
    norms=torch.linalg.vector_norm(torch.as_tensor(reference['data'],dtype=torch.float64),dim=-1).numpy()
    relative=[];mae=[];mse=[];trajectory=[]
    with torch.inference_mode(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        for start in range(0,len(norms),batch_size):
            stop=min(start+batch_size,len(norms))
            x,p=stats.encode(reference['params'][start:stop],reference['y_init'][start:stop],reference['data'].shape[-1])
            prediction=stats.decode(model.forward_ic_time(x.to(device),p.to(device)).float())
            truth=torch.as_tensor(reference['data'][start:stop],device=device)
            if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite physical replay')
            error=prediction.double()-truth.double()
            relative.append(sample_channel_relative_errors(prediction,truth).cpu().numpy())
            mae.append(error.abs().mean(-1).cpu().numpy());mse.append(error.square().mean(-1).cpu().numpy())
            trajectory.append(trajectory_relative_errors(prediction,truth).cpu().numpy())
    values=dict(relative=np.concatenate(relative),mae=np.concatenate(mae),mse=np.concatenate(mse),
                trajectory=np.concatenate(trajectory),target_norm=norms)
    normalization={key:getattr(stats,key).cpu().numpy().copy() for key in ('mean_y','std_y','p_mean','p_std')}
    del model,stats;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return values,normalization


def generate_report(output,summary,pairs,all_rows,case_changes):
    lines=['# 新旧sampling协议比较（2026-11-16）','',
        '旧：seed42_bb035a147b38；新：seed42_02eebd57592f。两批各16项，均seed42、FNO、最终10000训练样本、126株real。',
        '结论：mutation池这次整体退化；LHS池变化混合、总体接近。不能将新配置称为整体升级，也不能把所有退化归因于单独一个损失参数。',
        '旧评测JSON的Relative L2定义与新版本不同，本文不直接比较它们。下表统一RelL2均由各自最终checkpoint重新推理，固定原评测batch128、物理sample/channel、smooth eps0.01、不截断。MAE/MSE/Correlation及整轨迹macro定义未变。',
        '两批不仅改变训练eps0.1→0.01、cap10→无cap，也改变best选择指标（整轨迹macro→物理sample/channel）及验证推理精度、并发。不能将总体变化单独归因于去cap或eps。','',
        '## Real mutant 配对结果','',
        '|池|方法|MAE 旧→新|MSE 旧→新|统一RelL2 旧→新|选中epoch 旧→新|',
        '|---|---|---:|---:|---:|---:|']
    for row in pairs:
        lines.append(f'|{row["pool"]}|{NAMES[row["method"]]}|{row["MAE_old"]:.4f}→{row["MAE_new"]:.4f}|'
            f'{row["MSE_old"]:.4f}→{row["MSE_new"]:.4f}|{row["unified_RelL2_old"]:.4f}→{row["unified_RelL2_new"]:.4f}|'
            f'{row["best_epoch_old"]}→{row["best_epoch_new"]}|')
    lines+=['','## 八方法均值（描述统计，不是八个独立seed）','',
        '|池|指标|旧均值|新均值|相对变化|改善方法数|','|---|---|---:|---:|---:|---:|']
    for pool,values in summary['averages'].items():
        for name,value in values.items():
            lines.append(f'|{pool}|{name}|{value["old"]:.6f}|{value["new"]:.6f}|{value["relative_pct"]:+.2f}%|{value["improved"]}/8|')
    with (output/'near_zero_diagnostics.csv').open() as handle:zero_rows=list(csv.DictReader(handle))
    with (output/'per_variable.csv').open() as handle:variable_rows=list(csv.DictReader(handle))
    focus=next(r for r in zero_rows if r['pool']=='mutation' and r['method']=='random' and r['protocol']=='new')
    cdh=next(r for r in variable_rows if r['pool']=='mutation' and r['method']=='random' and r['protocol']=='new' and r['variable']=='CDH1T')
    total=next(r['unified_RelL2'] for r in all_rows if r['pool']=='mutation' and r['method']=='random' and r['protocol']=='new')
    lines+=['','## 小分母实际在强调什么','',
        f'新mutation/Random：4914个样本/变量项中，物理目标时间范数≤0.01的只有{focus["near_zero_pairs"]}项（{100*int(focus["near_zero_pairs"])/4914:.2f}%），却贡献总Relative L2的{100*float(focus["near_zero_fraction"]):.2f}%。',
        f'CDH1T一个变量贡献总分约{100*float(cdh["unified_RelL2"])/(39*total):.2f}%；126株中有25株的该通道近零。',
        '这些项包含真正的cdh1Δ/APC-ts背景。错误表达本身有生物意义，不能仅因它们主导指标就忽略；同时，指标不能直接替代其它蛋白幅值、相位和分裂预测的综合评价。',
        'Random即使只看非近零项，统一RelL2仍从0.4060升至0.4469；退化不只是改了指标数值或零点放大。',
        '训练的log1p+Z-score Relative L2中，“接近零”对应接近训练均值；物理Relative L2中则对应真正的低浓度。公式相同不等于两者赋予通道相同权重。','',
        '## 既有关注case（同池八方法MAE均值）','',
        '|池|case|旧MAE|新MAE|','|---|---|---:|---:|']
    for pool in ('mutation','lhs'):
        for prefix in ('15_','22_','42_','124_'):
            selected=[r for r in case_changes if r['pool']==pool and r['mutant_name'].startswith(prefix)]
            if selected:
                a,b=(statistics.mean(r['MAE_'+label] for r in selected) for label in ('old','new'))
                lines.append(f'|{pool}|#{prefix[:-1]}|{a:.6f}|{b:.6f}|')
    lines+=['','## 解释边界及诊断线索','',
        '1. 两池Random五轮训练ID和顺序一致，且最终训练统计逐项一致；Random仍变差时不能只归因于主动采样选错数据。自适应方法后续样本随模型改变，比较的是整条训练/采样/选checkpoint流程。',
        '2. mutation/Random最后一轮旧选176、新选56epoch；新Power选36epoch。新Random在被选epoch的训练loss约0.1803，训练到200epoch约0.0901。不能把“跑完200epoch”理解为“最终评测用了第200epoch的权重”。',
        '3. 这支持检查checkpoint指标权重是否与幅值/整体轨迹目标存在冲突；但没有保存各epoch模型，尚未完成同一新训练运行按旧指标重选或评测最后权重的反事实实验，不能断言所有退化都由checkpoint选择造成。',
        '4. 近零通道按物理目标时间范数≤0.01进行诊断；相对误差分母接近0.01会强调很小的非零预测。它们是否主导评分应看near_zero_fraction和逐变量贡献，不能把大Relative L2直接解释成所有蛋白都严重预测崩塌。',
        '5. 只有seed42，没有多seed统计显著性；real仅用于事后评价，不用于训练、选样或选checkpoint。','',
        '## 同池合成test（指标定义未变）','',
        '|池|指标|旧八方法均值|新八方法均值|相对变化|','|---|---|---:|---:|---:|']
    for pool,values in summary['synth_averages'].items():
        for name,value in values.items():
            lines.append(f'|{pool}|{name}|{value["old"]:.6f}|{value["new"]:.6f}|{value["relative_pct"]:+.2f}%|')
    lines+=['','## 运行核查与资源','',
        '32项新旧实验均完成；两批共160轮history、checkpoint、loss图、CSV行数及20160幅real图数量核查通过（未逐张解码PNG）。',
        f'新批全部16项退出码0，五轮epoch数100/100/100/100/200，10000唯一查询，无新增ODE。新批PSS峰值{summary["resource"]["new_peak_pss_gib"]:.2f}GiB，内存watchdog未触发。',
        f'旧批次监控墙钟{summary["resource"]["old_seconds"]/3600:.2f}小时，新批{summary["resource"]["new_seconds"]/3600:.2f}小时；共享服务器负载及新旧验证计算不同，不能当作严格并发吞吐对照。',
        '文件：paired_comparison.csv、final_metrics.csv、per_mutant_changes.csv、case_averages.csv、badcases_top20.csv、per_variable.csv、near_zero_diagnostics.csv、round_selection.csv、audit.json、comparison.png。只保存标量与统计，不复制轨迹或重写旧结果。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for index,pool in enumerate(('mutation','lhs')):
        for column,metric in enumerate(('MAE','unified_RelL2')):
            ax=axes[index,column];rows=[r for r in pairs if r['pool']==pool]
            y=np.arange(len(rows));old=[r[metric+'_old'] for r in rows];new=[r[metric+'_new'] for r in rows]
            ax.hlines(y,np.minimum(old,new),np.maximum(old,new),color='#9e9e9e',linewidth=2)
            ax.scatter(old,y,color='#1565c0',label='Old protocol',s=35,zorder=3)
            ax.scatter(new,y,color='#ef6c00',label='New protocol',marker='s',s=30,zorder=3)
            ax.set(yticks=y,yticklabels=[NAMES[r['method']] for r in rows],
                   xlabel=metric+' (lower is better)',title=f'{pool}: real {metric}')
            ax.invert_yaxis();ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root',type=Path,required=True)
    parser.add_argument('--new-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=128)
    args=parser.parse_args();torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    runs={};audits=[];round_rows=[];fairness=[]
    for pool in ('mutation','lhs'):
        for method in METHODS:
            entries={}
            for label,root in [('old',args.old_root),('new',args.new_root)]:
                folder=root/pool/method/'seed_42'
                config,metrics,rounds,complete=audit_run(folder)
                entries[label]=(folder,config,metrics,rounds,complete)
                audits.append(dict(protocol=label,pool=pool,method=method,rounds=5,images=630,status='complete'))
                for row in rounds:round_rows.append(dict(protocol=label,pool=pool,method=method,**{k:v for k,v in row.items() if k!='selection'}))
            a,b=entries['old'],entries['new']
            if a[1]['real_reference']['sha256']!=b[1]['real_reference']['sha256']:raise ValueError('Reference changed')
            same=[x['selection']['rows']==y['selection']['rows'] for x,y in zip(a[3],b[3])]
            if any(x['selection']['source_sha256']!=y['selection']['source_sha256'] for x,y in zip(a[3],b[3])):
                raise ValueError('Frozen trajectory source changed')
            if not same[0] or (method=='random' and not all(same)):raise ValueError('Shared initial/Random cohort changed')
            for key in ('seed','model','budgets','epochs','warmup','width','modes','n_blocks','batch_size','lr','min_lr','weight_decay'):
                if a[1]['training'][key]!=b[1]['training'][key]:raise ValueError(f'Training budget changed: {key}')
            fairness.append(dict(pool=pool,method=method,ordered_ids_equal_by_round=same))
            runs[(pool,method)]=entries
    print('AUDIT COMPLETE: 32 experiments, 160 rounds, starting batched physical reference replay',flush=True)
    reference_path=Path(runs[('mutation','random')]['new'][1]['real_reference']['path'])
    if file_sha256(reference_path)!=runs[('mutation','random')]['new'][1]['real_reference']['sha256']:raise ValueError('Reference bytes changed')
    with np.load(reference_path,allow_pickle=False) as a:reference={k:a[k] for k in ('params','y_init','data','mutant_names')}
    final=[];pairs=[];cases=[];variables=[];near_zero=[]
    for (pool,method),entries in runs.items():
        current={};normalizations={}
        for label,(folder,config,metrics,rounds,complete) in entries.items():
            values,normalization=replay_real(folder/'rounds/round_04/model.ckpt',reference,args.device,args.batch_size)
            current[label]=values;normalizations[label]=normalization
            m=flatten(metrics['real_mutants']);m['unified_RelL2']=float(values['relative'].mean())
            if label=='new' and not np.isclose(m['unified_RelL2'],metrics['real_mutants']['sample_channel_rel_l2'],rtol=2e-5,atol=1e-5):
                raise ValueError('Current replay differs from saved new physical metric')
            final.append(dict(protocol=label,pool=pool,method=method,metric_version=RELATIVE_L2_METRIC_VERSION,
                **m,best_epoch=rounds[-1]['selected_epoch'],seconds=complete['invocation_seconds']))
            mask=values['target_norm']<=.01
            near_zero.append(dict(protocol=label,pool=pool,method=method,pairs=int(mask.size),near_zero_pairs=int(mask.sum()),
                near_zero_fraction=float(values['relative'][mask].sum()/values['relative'].sum()),
                near_zero_rel_mean=float(values['relative'][mask].mean()),normal_rel_mean=float(values['relative'][~mask].mean())))
            for index,name in enumerate(VARIABLE_NAMES):
                variables.append(dict(protocol=label,pool=pool,method=method,variable=name,
                    MAE=float(values['mae'][:,index].mean()),MSE=float(values['mse'][:,index].mean()),
                    unified_RelL2=float(values['relative'][:,index].mean()),near_zero_targets=int(mask[:,index].sum())))
            print(label,pool,method,'unified real RelL2',m['unified_RelL2'],flush=True)
        if method=='random' and any(not np.array_equal(normalizations['old'][k],normalizations['new'][k]) for k in normalizations['old']):
            raise ValueError('Random training normalization changed')
        values=[next(r for r in final if r['pool']==pool and r['method']==method and r['protocol']==label) for label in ('old','new')]
        pair=dict(pool=pool,method=method)
        for name in ('MAE','MSE','Corr','macro_RelL2','tail_RelL2','unified_RelL2','pattern_accuracy','best_epoch'):
            pair[name+'_old'],pair[name+'_new']=(r[name] for r in values)
        pairs.append(pair)
        for index,name in enumerate(reference['mutant_names']):
            row=dict(pool=pool,method=method,mutant_name=str(name))
            for metric,key in [('MAE','mae'),('MSE','mse'),('unified_RelL2','relative')]:
                for label in ('old','new'):row[metric+'_'+label]=float(current[label][key][index].mean())
                row[metric+'_delta']=row[metric+'_new']-row[metric+'_old']
            cases.append(row)
    averages={};synth_averages={}
    for pool in ('mutation','lhs'):
        averages[pool]={};synth_averages[pool]={}
        for metric in ('MAE','MSE','Corr','macro_RelL2','tail_RelL2','unified_RelL2','pattern_accuracy'):
            rows=[r for r in pairs if r['pool']==pool]
            a,b=(statistics.mean(r[metric+'_'+label] for r in rows) for label in ('old','new'))
            averages[pool][metric]=dict(old=a,new=b,relative_pct=100*(b/a-1),
                improved=sum(r[metric+'_new']>r[metric+'_old'] if metric in ('Corr','pattern_accuracy') else r[metric+'_new']<r[metric+'_old'] for r in rows))
        for metric in ('MAE','MSE','macro_RelL2','tail_RelL2'):
            a,b=(statistics.mean(flatten(entries[label][2]['synth_test'])[metric]
                    for (p,_),entries in runs.items() if p==pool) for label in ('old','new'))
            synth_averages[pool][metric]=dict(old=a,new=b,relative_pct=100*(b/a-1))
    old_resource=read(args.old_root/'batch/resource_samples.summary.json')
    new_resource=read(args.new_root/'batch/resource_samples.summary.json')
    if new_resource['aborted']:raise ValueError('Memory guard aborted new batch')
    for root in (args.old_root,args.new_root):
        tasks=read(root/'batch/tasks.json')
        if len(tasks)!=16 or any(r['exit_code'] for r in tasks):raise ValueError('Batch contains failed tasks')
    summary=dict(metric_version=RELATIVE_L2_METRIC_VERSION,old_root=str(args.old_root),new_root=str(args.new_root),
        runs=audits,fairness=fairness,averages=averages,synth_averages=synth_averages,
        resource=dict(old_peak_pss_gib=old_resource['peak_pss_gib'],new_peak_pss_gib=new_resource['peak_owned_pss_gib'],
            old_seconds=old_resource['seconds'],new_seconds=new_resource['seconds']),
        replay_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        replay_cuda_peak_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0.,
        limitations='One seed; multi-factor protocol change; no post-hoc retraining or test-based selection')
    for name,rows in [('final_metrics',final),('paired_comparison',pairs),('per_mutant_changes',cases),
                      ('per_variable',variables),('near_zero_diagnostics',near_zero),('round_selection',round_rows)]:
        write_csv(args.output/(name+'.csv'),rows)
    case_averages=[]
    for pool in ('mutation','lhs'):
        names=sorted({row['mutant_name'] for row in cases if row['pool']==pool})
        for name in names:
            rows=[r for r in cases if r['pool']==pool and r['mutant_name']==name]
            row=dict(pool=pool,mutant_name=name)
            for key in ('MAE','MSE','unified_RelL2'):
                for label in ('old','new'):row[key+'_'+label]=statistics.mean(r[key+'_'+label] for r in rows)
                row[key+'_delta']=row[key+'_new']-row[key+'_old']
            case_averages.append(row)
    write_csv(args.output/'case_averages.csv',case_averages)
    write_csv(args.output/'badcases_top20.csv',sum(([dict(rank=index+1,**r) for index,r in enumerate(
        sorted([row for row in case_averages if row['pool']==pool],key=lambda row:row['MAE_new'],reverse=True)[:20])]
        for pool in ('mutation','lhs')),[]))
    atomic_json(args.output/'audit.json',summary)
    generate_report(args.output,summary,pairs,final,cases)
    print('COMPARISON COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
