#!/usr/bin/env python3
"""Paired cold-start FNO epsilon probe; synthetic validation alone chooses eps.

Shared model/trainer, frozen train-only normalization, identical initialization
and minibatch order. Physical evaluation always uses floor eps=.1. Test/real
outputs are opened for inference only after the validation decision is saved.
One GPU process trains all arms sequentially; no trajectory snapshots are saved.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from budding_yeast_v2.data.dataset_archive import atomic_json, file_sha256
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
from budding_yeast_v2.scripts.analysis.probe_relative_l2_training_cap import (
    FixedCohort, evaluate, initialize_probe_model, load_probe_cohorts,
    model_weights_sha256, save_csv)
from budding_yeast_v2.utils.evaluation.checkpoints import load_model_from_checkpoint
from budding_yeast_v2.utils.evaluation.metrics import RELATIVE_L2_METRIC_VERSION
from budding_yeast_v2.utils.experiments.curves import loss_figure
from budding_yeast_v2.utils.training.runner import EpochSummaryCallback, make_trainer


class GradientDiagnostics(Callback):
    """Record finite gradients before the shared Trainer's norm clipping."""
    def __init__(self, output):
        self.output=output
        self.rows=[]

    def on_train_epoch_start(self, trainer, model):
        self.norms=[]

    def on_after_backward(self, trainer, model):
        norms=torch.stack([p.grad.detach().norm().float() for p in model.parameters() if p.grad is not None])
        norm=norms.norm()
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite epsilon-probe gradient')
        self.norms.append(norm)

    def on_train_epoch_end(self, trainer, model):
        values=torch.stack(self.norms)
        row=dict(epoch=trainer.current_epoch+1,gradient_norm_max=float(values.max()),
                 gradient_norm_mean=float(values.mean()),steps=len(values),
                 steps_above_clip=int((values>1.).sum()))
        self.rows.append(row)
        save_csv(self.output/'gradient_diagnostics.csv',self.rows)


def denominator_audit(targets, eps_values, output):
    """Only the queried training targets determine these scale diagnostics."""
    norms=targets.norm(dim=-1).numpy()
    rows=[]
    for eps in eps_values:
        for index,name in enumerate(VARIABLE_NAMES):
            values=norms[:,index]
            rows.append(dict(eps=eps,variable=name,samples=len(values),
                below_floor=int((values<eps).sum()),below_floor_pct=float(100*(values<eps).mean()),
                norm_min=float(values.min()),norm_p01=float(np.quantile(values,.01)),
                norm_median=float(np.median(values)),norm_p99=float(np.quantile(values,.99))))
    save_csv(output/'training_denominators.csv',rows)
    return {str(eps):dict(pairs=int(norms.size),below_floor=int((norms<eps).sum()),
                        below_floor_pct=float(100*(norms<eps).mean())) for eps in eps_values}


def choose_epsilon(arms):
    """Predeclared decision: fixed-last physical sample/channel validation loss."""
    return min(arms,key=lambda arm:arm['last_validation']['sample_channel_rel_l2'])['eps']


def report(output, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=results['arms']
    rows=[]
    for arm in arms:
        for stage in ('last','best'):
            for cohort,metrics in arm[stage].items():
                rows.append(dict(eps=arm['eps'],stage=stage,cohort=cohort,**metrics))
    save_csv(output/'comparison.csv',rows)
    fig,axes=plt.subplots(1,3,figsize=(17,4.8))
    for arm in arms:
        label=f'train eps={arm["eps"]}'
        history=arm['history']
        axes[0].plot([r['epoch'] for r in history],[r['val_macro_rel_l2'] for r in history],label=label)
        axes[1].plot([r['epoch'] for r in arm['gradients']],
                     [r['gradient_norm_max'] for r in arm['gradients']],label=label)
    axes[0].set(title='Physical validation macro RelL2',xlabel='Epoch',ylabel='Validation error')
    axes[1].set(title='Gradient norm before clipping',xlabel='Epoch',ylabel='Global gradient norm',yscale='log')
    roles=['validation','synth_test','real'];x=np.arange(len(roles))
    for index,arm in enumerate(arms):
        axes[2].bar(x+(index-(len(arms)-1)/2)*.24,
            [arm['last'][role]['sample_channel_rel_l2'] for role in roles],width=.24,label=f'eps={arm["eps"]}')
    axes[2].set(xticks=x,xticklabels=roles,title='Same physical metric: eval eps=0.1',ylabel='Sample/channel RelL2')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Paired cold starts; epsilon chosen using synthetic validation only')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    lines=['# Relative L2 分母下限对照（2026-11-16）','',
        '训练与评测均用 max(||target||_time, eps)，loss 不截断；梯度范数裁剪1。',
        '训练epsilon分别为0.03、0.1、0.3；所有物理评测固定epsilon=0.1，不能用各自不同的评测分母排名。',
        '单一mutation训练集合、单seed42、完全相同随机初始权重及批顺序，归一化仅来自这4000条合成训练样本。',
        '每臂60epoch，batch128，LR7.5e-4，warmup4，cosine最低1e-5；既有FNO192/4blocks/modes24。',
        '512条固定合成validation；512条合成test与126株real仅在validation决策文件写入后评测。',
        '预先固定：用第60epoch的validation物理逐样本/变量Relative L2选择epsilon；best checkpoint另按既有validation macro规则记录。',
        f'验证集选择：epsilon={results["decision"]["chosen_eps"]}。不据real或test结果改选。','',
        '|训练epsilon|训练末loss*|验证RelL2|合成test RelL2|real RelL2|real MAE|real MSE|裁剪前最大梯度|秒|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        last=arm['last']
        lines.append(f'|{arm["eps"]}|{arm["history"][-1]["train_loss"]:.6f}|'
            f'{last["validation"]["sample_channel_rel_l2"]:.6f}|{last["synth_test"]["sample_channel_rel_l2"]:.6f}|'
            f'{last["real"]["sample_channel_rel_l2"]:.6f}|{last["real"]["MAE"]:.6f}|{last["real"]["MSE"]:.6f}|'
            f'{max(r["gradient_norm_max"] for r in arm["gradients"]):.4f}|{arm["seconds"]:.1f}|')
    lines+=['','*训练loss分母不同，不用其数值跨epsilon排名。',
        '这次对照支持稳定性和局部选择，不代表跨初始化/不同训练集合的最优epsilon，也没有完成完整五轮采样实验。',
        '范数较小也可能来自Z-score中心化，不能把所有受保护项解释为物理零浓度。',
        f'进程RSS峰值{results["peak_rss_gib"]:.3f}GiB，GPU allocated峰值{results["cuda_peak_allocated_gib"]:.3f}GiB。',
        'training_denominators.csv记录每个变量的范数分位数及下限触发比例；comparison.csv包含best/last分组完整指标。',
        '仅保存源行ID和checkpoint，不保存轨迹副本；历史正式结果保持原样。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--eps',type=float,nargs='+',default=[.03,.1,.3])
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--holdout-size',type=int,default=512)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if RELATIVE_L2_METRIC_VERSION != 'physical_sample_channel_rel_l2_floor_v2':
        parser.error('This historical probe requires max-floor training and physical eps=0.1. '
                     'Current defaults changed; use its recorded source version to reproduce it. '
                     'Existing results are unchanged.')
    if any(not np.isfinite(e) or e<=0 for e in args.eps) or len(set(args.eps))!=len(args.eps):
        parser.error('epsilon values must be distinct, finite, positive')
    if args.epochs<5:parser.error('use at least five epochs')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'config.json').exists():
        raise ValueError('Use a fresh output directory; completed diagnostics must not be overwritten')
    checkpoint,stats,contract,cohorts,selection=load_probe_cohorts(
        args.run,args.protocol,args.round,args.holdout_size,args.seed)
    identity=dict(seed=args.seed,source=str(checkpoint),source_sha256=file_sha256(checkpoint),
        train_rows=len(selection['train']),validation_rows=len(selection['validation']),
        eps_values=args.eps,epochs=args.epochs,batch_size=128,lr=7.5e-4,min_lr=1e-5,warmup=4,
        denominator_rule='max_target_time_l2_eps',loss_cap=None,gradient_clip_norm=1.,
        physical_metric_version=RELATIVE_L2_METRIC_VERSION,physical_evaluation_eps=.1,
        validation_decision='fixed_last_physical_sample_channel_rel_l2',
        source_files={name:file_sha256(Path(__file__).parents[2]/name) for name in
            ['utils/training/losses.py','utils/evaluation/metrics.py','models/yeast_lit_module.py',
             'scripts/analysis/probe_relative_l2_epsilon.py','scripts/analysis/probe_relative_l2_training_cap.py']})
    atomic_json(args.output/'config.json',identity)
    atomic_json(args.output/'selection.json',selection)
    dm=FixedCohort(cohorts['train'],cohorts['validation'],stats,contract,128,args.seed)
    audit=denominator_audit(dm.train_dataset_synth.tensors[2],args.eps,args.output)
    atomic_json(args.output/'denominator_summary.json',audit)
    results=[];common_sha=None
    for eps in args.eps:
        out=args.output/f'eps_{eps:g}';out.mkdir()
        pl.seed_everything(args.seed,workers=True)
        model=initialize_probe_model(checkpoint,'scratch')
        model.criterion.eps=eps;model.criterion.max_clamp=None
        model.lr=7.5e-4;model.max_epochs=args.epochs
        model.hparams.update(lr=7.5e-4,max_epochs=args.epochs,warmup_epochs=4,min_lr=1e-5)
        weights_sha=model_weights_sha256(model)
        if common_sha is None:common_sha=weights_sha
        if common_sha!=weights_sha:raise ValueError('Paired initialization mismatch')
        dm.loader_generator=torch.Generator().manual_seed(args.seed)
        recorder=EpochSummaryCallback(output=out,identity=identity,generator=dm.loader_generator,
            round_index=args.round,member=0,monitor='val_macro_rel_l2')
        diagnostics=GradientDiagnostics(out)
        best=ModelCheckpoint(dirpath=out,filename='best',monitor='val_macro_rel_l2',mode='min',
            save_top_k=1,save_weights_only=True,enable_version_counter=False)
        trainer=make_trainer(max_epochs=args.epochs,accelerator='gpu',devices=[int(args.device.split(':')[-1])],
            logger=False,callbacks=[diagnostics,recorder,best],enable_progress_bar=False,enable_model_summary=False)
        started=time.monotonic()
        model.train();trainer.fit(model,datamodule=dm)
        model.to(args.device)
        last=evaluate(model,stats,cohorts['validation'],128)
        trainer.save_checkpoint(out/'last.ckpt',weights_only=True)
        if recorder.history[-1]['optimizer_steps']!=int(np.ceil(len(selection['train'])/128))*args.epochs:
            raise ValueError('Paired optimizer-step count mismatch')
        result=dict(eps=eps,last_validation=last,initial_weights_sha256=weights_sha,
            optimizer_steps=recorder.history[-1]['optimizer_steps'],history=recorder.history,
            gradients=diagnostics.rows,seconds=time.monotonic()-started)
        results.append(result);atomic_json(out/'validation_result.json',result)
        loss_figure(recorder.history,out/'loss.png',title=f'Uncapped Relative L2, train eps={eps}',
            context=dict(model='fno',loss='rel_l2',training_rows=len(selection['train']),warmup=4,seed=args.seed))
        print(f'VALIDATION ARM COMPLETE eps={eps}: {json.dumps(last)}',flush=True)
        model.cpu();del model,trainer;gc.collect();torch.cuda.empty_cache()
    decision=dict(chosen_eps=choose_epsilon(results),selection_metric='last_validation.sample_channel_rel_l2',
        physical_eval_eps=.1,used_real_or_test=False,
        scores={str(r['eps']):r['last_validation']['sample_channel_rel_l2'] for r in results})
    atomic_json(args.output/'validation_decision.json',decision)
    print(f'VALIDATION DECISION (test/real not evaluated yet): {json.dumps(decision)}',flush=True)
    for arm in results:
        out=args.output/f'eps_{arm["eps"]:g}'
        for stage in ('last','best'):
            model,_=load_model_from_checkpoint(out/f'{stage}.ckpt',device=args.device)
            if model.criterion.eps!=arm['eps']:raise ValueError('Checkpoint lost training epsilon')
            arm[stage]={name:evaluate(model,stats,data,128) for name,data in cohorts.items() if name!='train'}
            model.cpu();del model;gc.collect();torch.cuda.empty_cache()
        atomic_json(out/'result.json',arm)
    output=dict(identity=identity,decision=decision,denominators=audit,arms=results,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    atomic_json(args.output/'results.json',output)
    report(args.output,output)
    print('EPSILON PROBE COMPLETE',flush=True)


if __name__=='__main__':
    main()
