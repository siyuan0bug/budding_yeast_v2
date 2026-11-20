#!/usr/bin/env python3
"""Controlled cap10/no-cap FNO probe with shared initialization and data.

Uses the shared Lightning model/optimizer/trainer, identical queried training
rows and frozen normalization. Start from the same checkpoint or the same
random weights. Synthetic validation selects checkpoints; test
and real are evaluated only before/after optimization. Not a new AL benchmark.
"""
import argparse
import copy
import csv
import hashlib
import inspect
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from budding_yeast_v2.data.dataset_archive import TrajectoryStore, atomic_json, file_sha256
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.utils.evaluation.checkpoints import load_model_from_checkpoint
from budding_yeast_v2.utils.evaluation.metrics import sample_channel_relative_errors, trajectory_relative_errors
from budding_yeast_v2.utils.training.runner import make_trainer, EpochSummaryCallback
from budding_yeast_v2.utils.training.monitoring import evaluation_state
from budding_yeast_v2.utils.experiments.curves import loss_figure
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule


def model_weights_sha256(model):
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_probe_model(path, start):
    if start=='checkpoint':
        return load_model_from_checkpoint(path,device='cpu')[0]
    payload=torch.load(path,map_location='cpu',weights_only=False)
    accepted=inspect.signature(YeastLitModule.__init__).parameters
    kwargs={k:v for k,v in payload['hyper_parameters'].items() if k in accepted and k!='self'}
    model=YeastLitModule(**kwargs)
    # Restore only the frozen preprocessing contract, never the trained weights.
    preprocessing_payload=dict(payload)
    preprocessing_payload.pop('training_objective_definition',None)
    model.on_load_checkpoint(preprocessing_payload)
    return model


def save_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class FixedCohort(pl.LightningDataModule):
    """Small probe adapter; the full model/optimization logic remains shared."""
    def __init__(self, training, validation, stats, contract, batch_size, seed):
        super().__init__()
        self.contract=copy.deepcopy(contract)
        self.system_id='budding';self.t_max=210.;self.norm_type=stats.norm_type
        self.num_vars,self.num_params,self.T_STEPS=39,141,training['data'].shape[-1]
        stats.bind(self)
        self.batch_size=self.val_batch_size=batch_size
        self.validation_metric='macro_trajectory_rel_l2'
        self.seed=seed;self.loader_generator=torch.Generator().manual_seed(seed)
        x,p=stats.encode(training['params'],training['y_init'],self.T_STEPS)
        self.train_dataset_synth=TensorDataset(x,p,stats.targets(training['data']))
        x,p=stats.encode(validation['params'],validation['y_init'],self.T_STEPS)
        physical=torch.as_tensor(validation['data'],dtype=torch.float32)
        self.validation=TensorDataset(x,p,stats.targets(physical),physical)

    def preprocessing_state(self):
        return copy.deepcopy(self.contract)

    def train_dataloader(self):
        return DataLoader(self.train_dataset_synth,batch_size=self.batch_size,shuffle=True,
                          generator=self.loader_generator,num_workers=0,drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.validation,batch_size=self.val_batch_size,shuffle=False,num_workers=0)


class CapDiagnostics(Callback):
    """Observe actual training forward terms and gradients before norm clipping."""
    def __init__(self, output):
        self.output=Path(output);self.rows=[]

    def on_fit_start(self, trainer, model):
        def hook(criterion,args,result):
            if not criterion.training:
                return
            with torch.no_grad():
                p,y=args[:2]
                ratio=criterion.channel_errors(p,y)
                self.sums+=torch.stack(((ratio>10).sum(),ratio.new_tensor(ratio.numel()),
                    ratio.sum(),ratio.clamp(max=10).sum())).double()
                self.max_ratio=torch.maximum(self.max_ratio,ratio.max().double())
        self.handle=model.criterion.register_forward_hook(hook)

    def on_train_epoch_start(self, trainer, model):
        self.sums=torch.zeros(4,dtype=torch.float64,device=model.device)
        self.max_ratio=torch.zeros((),dtype=torch.float64,device=model.device)
        self.gradients=[]
        self.output_gradients=[]

    def on_after_backward(self, trainer, model):
        with torch.no_grad():
            values=[p.grad.norm().float() for p in model.parameters() if p.grad is not None]
            norm=torch.stack(values).norm().detach()
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite probe gradient norm')
            self.gradients.append(norm)
            self.output_gradients.append(model.model.project[-1].weight.grad[23].norm().detach())

    def on_train_epoch_end(self, trainer, model):
        a,b,c,d=self.sums.cpu().tolist()
        if b == 0:
            raise RuntimeError('Training cap observer saw no forward terms; check model training mode')
        gradients=torch.stack(self.gradients)
        row=dict(epoch=trainer.current_epoch+1,clipped_pairs=int(a),sample_variable_pairs=int(b),
            clipped_pair_pct=100*a/b,uncapped_train_terms=c/b,capped_train_terms=d/b,
            max_ratio=float(self.max_ratio),gradient_norm_mean=float(gradients.mean()),
            gradient_norm_max=float(gradients.max()),
            CDC14T_output_weight_gradient_mean=float(torch.stack(self.output_gradients).mean()))
        self.rows.append(row);save_csv(self.output/'cap_diagnostics.csv',self.rows)
        print(f'[CAP] epoch={row["epoch"]} over10={a:.0f}/{b:.0f} '
              f'({row["clipped_pair_pct"]:.5f}%) grad_max={row["gradient_norm_max"]:.4g}',flush=True)

    def on_fit_end(self, trainer, model):
        self.handle.remove()


def evaluate(model, stats, cohort, batch_size):
    n=len(cohort['data']);v,t=cohort['data'].shape[1:]
    totals=torch.zeros(9,dtype=torch.float64,device=model.device)
    with evaluation_state(model):
        for start in range(0,n,batch_size):
            stop=min(start+batch_size,n)
            x,p=stats.encode(cohort['params'][start:stop],cohort['y_init'][start:stop],t)
            target=torch.as_tensor(cohort['data'][start:stop],device=model.device)
            z=stats.targets(target)
            pred=model.forward_ic_time(x.to(model.device),p.to(model.device)).float()
            ratio=model.criterion.channel_errors(pred,z)
            physical=stats.decode(pred).double();target=target.double()
            if not torch.isfinite(ratio).all() or not torch.isfinite(physical).all():
                raise FloatingPointError('Nonfinite probe evaluation')
            error=physical-target
            totals+=torch.stack((sample_channel_relative_errors(physical,target).sum(),
                error.abs().sum(),error.square().sum(),error[:,23].abs().sum(),
                ratio.double().sum(),ratio.clamp(max=10).double().sum(),
                (ratio>10).double().sum(),trajectory_relative_errors(physical,target).sum(),
                (ratio>10).any(1).double().sum()))
    a,b,c,d,e,f,g,h,j=totals.cpu().tolist()
    return dict(count=n,sample_channel_rel_l2=a/(n*v),MAE=b/(n*v*t),MSE=c/(n*v*t),
        CDC14T_MAE=d/(n*t),uncapped_norm_loss=e/(n*v),capped_norm_loss=f/(n*v),
        over10_pairs=int(g),over10_pair_pct=100*g/(n*v),macro_trajectory_rel_l2=h/n,
        samples_over10=int(j))


def write_probe_report(output):
    """Physical diagnostics/plots from completed arms; never selects a new model."""
    output=Path(output)
    results=json.loads((output/'results.json').read_text())
    arms=results['arms'];lookup={r['arm']:r for r in arms}
    if lookup['cap10']['before']!=lookup['uncapped']['before']:
        raise ValueError('Paired arms do not start from identical predictions')
    cfg=results['identity']
    expected_steps=((cfg['training_rows']+cfg['batch_size']-1)//cfg['batch_size'])*cfg['epochs']
    if any(r['optimizer_steps']!=expected_steps for r in arms):
        raise ValueError('Unexpected paired update budget')
    rows=[]
    for arm in arms:
        for stage in ('before','last','best'):
            for role,metrics in arm[stage].items():
                rows.append(dict(arm=arm['arm'],stage=stage,cohort=role,**metrics))
    save_csv(output/'comparison.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
    source=Path(results['identity']['source'])
    config=json.loads((source.parents[2]/'config.json').read_text())
    with np.load(config['real_reference']['path'],allow_pickle=False) as a:
        reference={k:a[k] for k in ('params','y_init','data','mutant_names')}
    focus=next(i for i,n in enumerate(reference['mutant_names']) if str(n).startswith('98_'))
    curves={};by_variable=[]
    torch.set_num_threads(2)
    before_checkpoint=output/'initial.ckpt' if (output/'initial.ckpt').exists() else source
    for label,path in [('before',before_checkpoint),('cap10',output/'cap10/last.ckpt'),('uncapped',output/'uncapped/last.ckpt')]:
        model,_=load_model_from_checkpoint(path,device='cpu')
        stats=NormalizationStats.from_datamodule(model)
        sums=torch.zeros(3,39,dtype=torch.float64)
        with torch.inference_mode():
            for start in range(0,len(reference['data']),16):
                stop=min(start+16,len(reference['data']))
                x,p=stats.encode(reference['params'][start:stop],reference['y_init'][start:stop],500)
                pred=stats.decode(model.forward_ic_time(x,p)).double()
                truth=torch.as_tensor(reference['data'][start:stop],dtype=torch.float64)
                error=pred-truth
                sums+=torch.stack((error.abs().sum((0,2)),error.square().sum((0,2)),
                                   sample_channel_relative_errors(pred,truth).sum(0)))
                if start<=focus<stop:curves[label]=pred[focus-start,23].numpy()
        for v,name in enumerate(VARIABLE_NAMES):
            by_variable.append(dict(arm=label,variable=name,MAE=float(sums[0,v]/(126*500)),
                MSE=float(sums[1,v]/(126*500)),sample_channel_rel_l2=float(sums[2,v]/126)))
        del model
    save_csv(output/'real_per_variable.csv',by_variable)
    diag={}
    for label in ('cap10','uncapped'):
        with (output/label/'cap_diagnostics.csv').open() as handle:diag[label]=list(csv.DictReader(handle))
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for label,color in [('cap10','#1565c0'),('uncapped','#ef6c00')]:
        epochs=[int(r['epoch']) for r in diag[label]]
        axes[0,0].plot(epochs,[float(r['uncapped_train_terms']) for r in diag[label]],label=label,color=color)
        axes[0,1].plot(epochs,[float(r['clipped_pair_pct']) for r in diag[label]],label=label,color=color)
    axes[0,0].set(title='Same uncapped normalized training diagnostic',xlabel='Epoch',ylabel='Relative L2 (uncapped)')
    axes[0,1].set(title='Actual training terms above 10',xlabel='Epoch',ylabel='Sample-channel terms (%)')
    roles=('validation','synth_test','real');x=np.arange(3)
    for label,delta,color in [('cap10',-.18,'#1565c0'),('uncapped',.18,'#ef6c00')]:
        axes[1,0].bar(x+delta,[lookup[label]['last'][role]['sample_channel_rel_l2'] for role in roles],.36,label=label,color=color)
    axes[1,0].set(xticks=x,xticklabels=[f'Validation ({cfg["validation_rows"]})',f'Synthetic test ({cfg["test_rows"]})','Real (126)'],
                  title=f'Physical sample/channel RelL2 after {cfg["epochs"]} epochs',ylabel='Relative L2 (no cap)')
    t=np.linspace(0,210,500)
    axes[1,1].plot(t,reference['data'][focus,23],color='black',label='ODE truth',linewidth=2)
    axes[1,1].plot(t,curves['before'],'--',color='gray',label='Before')
    axes[1,1].plot(t,curves['cap10'],color='#1565c0',label='cap10')
    axes[1,1].plot(t,curves['uncapped'],color='#ef6c00',label='uncapped')
    axes[1,1].set(title='Diagnostic example #98: CDC14T',xlabel='Time (min)',ylabel='Physical concentration')
    for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle(f'Same FNO initialization / {cfg["training_rows"]:,} training rows / frozen normalization / {expected_steps} updates per arm')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    lines=['# 训练 Relative L2 cap10 对照（2026-11-15）','',
        f'起点模式：{cfg.get("start","checkpoint")}；单seed受控诊断，不是多seed总体优越性结论。',
        f'固定{cfg["training_rows"]}训练行、原训练归一化统计；seed={cfg["seed"]}，两分支各{cfg["epochs"]}epoch/{expected_steps}步。',
        f'两边均重置AdamW、LR={cfg["lr"]}、warmup={cfg["warmup"]}、cosine到{cfg["min_lr"]}、batch={cfg["batch_size"]}、梯度范数裁剪1。仅loss比值cap10或None不同。',
        f'{cfg["validation_rows"]}条固定合成validation选择best，另{cfg["test_rows"]}条固定合成test和126 real仅事后报告；不作选样或checkpoint挑选。',
        '两分支before四组预测指标逐项完全一致，训练ID、数据顺序seed和步数相同；正式训练默认未变。','',
        f'## 固定{cfg["epochs"]}epoch后的结果（last，避免只挑有利checkpoint）','',
        '|集合|指标|cap10|不截断|','|---|---|---:|---:|']
    for role in ('validation','synth_test','real'):
        for key in ('sample_channel_rel_l2','MAE','MSE','CDC14T_MAE','macro_trajectory_rel_l2'):
            lines.append(f'|{role}|{key}|{lookup["cap10"]["last"][role][key]:.6f}|{lookup["uncapped"]["last"][role][key]:.6f}|')
    lines+=['','## 稳定性和边界','']
    for label in ('cap10','uncapped'):
        history=diag[label]
        gradients=max(float(row['gradient_norm_max']) for row in history)
        lines.append(f'{label}：完整{len(history)}epoch；末epoch超10项={history[-1]["clipped_pairs"]}/{history[-1]["sample_variable_pairs"]}；裁剪前最大梯度范数={gradients:.6g}。')
    delta=lookup['uncapped']['last']['validation']['sample_channel_rel_l2']-lookup['cap10']['last']['validation']['sample_channel_rel_l2']
    lines += [f'不截断相对cap10的末epoch合成validation物理RelL2差值={delta:+.6f}（负值更好）。',
        '两个分支均正常完成，未出现非有限训练loss/梯度；finite大loss与NaN/Inf必须区分。',
        '这符合“中心化分母接近零时，个别通道得到很大优化权重”的机制解释，但没有单独隔离共享主干、Adam状态和所有通道间的因果通路。',
        '正式训练默认未改；评测始终不cap、保留零分母保护。不能根据单seed和单一训练集合宣称已找到最优损失。',
        'best由固定合成validation的原完整物理轨迹macro规则选择，完整best/last均保留在results.json/comparison.csv。','',
        '图：comparison.png（统一uncapped训练诊断、实际超限比例、物理评测、CDC14T曲线）；逐变量诊断real_per_variable.csv；两分支各有history.csv/cap_diagnostics.csv/loss.png和best/last checkpoint。',
        '原正式实验和既有结果未改动。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def load_probe_cohorts(run, protocol_dir, round_index, holdout_size, seed):
    """Read selected source rows once; shared by cap and epsilon diagnostics."""
    source=Path(run)/'rounds'/f'round_{round_index:02d}'
    checkpoint=source/'model.ckpt'
    config=json.loads((Path(run)/'config.json').read_text())
    manifest=json.loads((source/'dataset.json').read_text())
    pool=config['pool']
    with np.load(Path(protocol_dir)/f'{pool}_split.npz',allow_pickle=False) as a:
        rng=np.random.default_rng(seed)
        val_ids=np.sort(rng.choice(a['validation'],holdout_size,replace=False))
        test_ids=np.sort(rng.choice(a['test'],holdout_size,replace=False))
    train_ids=np.asarray(manifest['rows'],dtype=np.int64)
    if set(train_ids)&(set(val_ids)|set(test_ids)) or set(val_ids)&set(test_ids):
        raise ValueError('Probe train/validation/test overlap')
    store=TrajectoryStore(manifest['source'],cache_chunks=1)
    if store.digest!=manifest['source_sha256']:
        raise ValueError('Training source changed')
    union=np.unique(np.concatenate((train_ids,val_ids,test_ids)))
    print(f'Loading {len(union)} unique source rows once',flush=True)
    physical=store.read(union,('data',))['data']
    with np.load(Path(protocol_dir)/f'{pool}_inputs.npz',allow_pickle=False) as a:
        inputs={k:a[k] for k in ('params','y_init')}
    cohorts={}
    for name,ids in [('train',train_ids),('validation',val_ids),('synth_test',test_ids)]:
        cohorts[name]=dict(params=inputs['params'][ids],y_init=inputs['y_init'][ids],
                          data=physical[np.searchsorted(union,ids)])
    with np.load(config['real_reference']['path'],allow_pickle=False) as a:
        cohorts['real']={k:a[k] for k in ('params','y_init','data')}
    base,_=load_model_from_checkpoint(checkpoint,device='cpu')
    stats=NormalizationStats.from_datamodule(base)
    contract=copy.deepcopy(base._preprocessing_state)
    return checkpoint,stats,contract,cohorts,dict(train=train_ids.tolist(),validation=val_ids.tolist(),test=test_ids.tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--start',choices=['checkpoint','scratch'],default='checkpoint')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--holdout-size',type=int,default=512)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--min-lr',type=float,default=1e-5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((args.protocol/'protocol.json').read_text())
    checkpoint,stats,contract,cohorts,selection=load_probe_cohorts(
        args.run,args.protocol,args.round,args.holdout_size,args.seed)
    train_ids,val_ids,test_ids=(selection[k] for k in ('train','validation','test'))
    template=initialize_probe_model(checkpoint,args.start)
    objective_rule=template.criterion.denominator_rule
    objective_eps=template.criterion.eps
    del template
    from budding_yeast_v2.utils.evaluation.metrics import RELATIVE_L2_METRIC_VERSION, PHYSICAL_RELATIVE_L2_EPS
    identity=dict(source=str(checkpoint),sha256=file_sha256(checkpoint),round=args.round,
        training_rows=len(train_ids),validation_rows=len(val_ids),test_rows=len(test_ids),
        epochs=args.epochs,lr=args.lr,min_lr=args.min_lr,warmup=args.warmup,seed=args.seed,batch_size=args.batch_size,
        start=args.start,
        gradient_clip_norm=1.,precision='bf16-mixed',
        denominator_rule=objective_rule,eps=objective_eps,
        physical_metric_version=RELATIVE_L2_METRIC_VERSION,physical_eval_eps=PHYSICAL_RELATIVE_L2_EPS,
        protocol='shared initialization, reset optimizer, frozen stats, paired data order; recorded denominator diagnostic')
    if (args.output/'config.json').exists() and json.loads((args.output/'config.json').read_text())!=identity:
        raise ValueError('Probe configuration changed; use a separate output directory')
    atomic_json(args.output/'config.json',identity)
    atomic_json(args.output/'selection.json',selection)
    dm=FixedCohort(cohorts['train'],cohorts['validation'],stats,contract,args.batch_size,args.seed)
    results=[]
    for label,cap in [('cap10',10.),('uncapped',None)]:
        out=args.output/label;out.mkdir(exist_ok=True)
        if (out/'result.json').exists():
            results.append(json.loads((out/'result.json').read_text()));continue
        pl.seed_everything(args.seed,workers=True)
        model=initialize_probe_model(checkpoint,args.start)
        model.criterion.max_clamp=cap
        model.lr=args.lr;model.max_epochs=args.epochs
        model.hparams.update(lr=args.lr,max_epochs=args.epochs,warmup_epochs=args.warmup,min_lr=args.min_lr)
        initial_weights_sha256=model_weights_sha256(model)
        if not (args.output/'initial.ckpt').exists():
            torch.save(dict(hyper_parameters=dict(model.hparams),preprocessing_contract=copy.deepcopy(contract),
                state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=-1),args.output/'initial.ckpt')
        model.to(args.device)
        before={name:evaluate(model,stats,data,args.batch_size) for name,data in cohorts.items()}
        atomic_json(out/'before.json',before)
        dm.loader_generator=torch.Generator().manual_seed(args.seed)
        diagnostics=CapDiagnostics(out)
        recorder=EpochSummaryCallback(output=out,identity=identity,generator=dm.loader_generator,
            round_index=args.round,member=0,monitor='val_macro_rel_l2')
        best=ModelCheckpoint(dirpath=out,filename='best',monitor='val_macro_rel_l2',mode='min',
            save_top_k=1,save_weights_only=True,enable_version_counter=False)
        trainer=make_trainer(max_epochs=args.epochs,accelerator='gpu',devices=[int(args.device.split(':')[-1])],
            logger=False,callbacks=[diagnostics,recorder,best],enable_progress_bar=False,enable_model_summary=False)
        started=time.monotonic()
        # Checkpoint loading returns eval(); continuation must explicitly restore
        # train mode before Lightning captures/restores module modes.
        model.train()
        trainer.fit(model,datamodule=dm)
        # Lightning's strategy teardown may move the module back to CPU.
        model.to(args.device)
        print(f'Evaluating last checkpoint: {label}',flush=True)
        last={name:evaluate(model,stats,data,args.batch_size) for name,data in cohorts.items()}
        trainer.save_checkpoint(out/'last.ckpt',weights_only=True)
        model.cpu();del model,trainer
        selected,_=load_model_from_checkpoint(best.best_model_path,device=args.device)
        selected.criterion.max_clamp=cap
        after={name:evaluate(selected,stats,data,args.batch_size) for name,data in cohorts.items()}
        result=dict(arm=label,cap=cap,before=before,last=last,best=after,
            best_validation_macro=float(best.best_model_score),seconds=time.monotonic()-started,
            initial_weights_sha256=initial_weights_sha256,
            initial_checkpoint_sha256=identity['sha256'],optimizer_steps=int(recorder.history[-1]['optimizer_steps']))
        atomic_json(out/'result.json',result);results.append(result)
        loss_figure(recorder.history,out/'loss.png',title=f'Cap probe: {label}',
            context=dict(model='fno',loss='rel_l2',training_rows=len(train_ids),warmup=args.warmup,seed=args.seed))
        selected.cpu();del selected
        print(f'ARM COMPLETE {label}: '+json.dumps(result['last']),flush=True)
    atomic_json(args.output/'results.json',dict(identity=identity,arms=results,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        note='Post-hoc targeted recovery probe, not a cold-start/multiseed generalization claim. No AL selection or real-based checkpoint selection.'))
    write_probe_report(args.output)


if __name__=='__main__':
    main()
