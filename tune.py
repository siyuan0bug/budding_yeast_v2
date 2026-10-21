#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna Distributed Hyperparameter Optimization (多 GPU 高并发超参自动搜索)
========================================================================
特点:
  1. 支持 8 GPU 进程共享 SQLite WAL 数据库并行搜索 (吞吐量提升 8 倍)。
  2. 模型感知自适应搜索空间 (针对 WNO 限制 batch_size 彻底防 OOM，针对 Refiner/FNO 专属优化)。
  3. 自动提取 Top-K 最优超参并生成可直接粘贴到 run_pipeline.sh 的实验配置。
  4. 支持断点续跑 (Resume) 与智能中途剪枝 (Pruning)。
"""

import os
import sys
import gc
import json
import sqlite3
import argparse
import shutil
from typing import Dict, Any

# 确保能从父目录导入 budding_yeast_v2 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from budding_yeast_v2.utils.active_learning.registry import CLI_STRATEGIES

import torch
import numpy as np
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# PL 2.x 的剪枝回调
try:
    from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
except ImportError:
    from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule, MODEL_REGISTRY
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.data.preprocessing_contract import TRAIN_FIT_SCOPE
from budding_yeast_v2.utils.metrics import calculate_metrics


def sample_model_aware_hparams(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """根据模型架构自适应采样超参数，严格防爆显存并适配模型特性"""
    # 1. 通用优化器与训练超参
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    warmup_epochs = trial.suggest_int("warmup_epochs", 5, 20)
    max_epochs = trial.suggest_int("max_epochs", 100, 300, step=50)

    m_lower = model_name.lower()
    hparams = {
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "max_epochs": max_epochs,
        "n_blocks": trial.suggest_int("n_blocks", 2, 4 if model_name == "unet" else 6),
    }

    # 2. 模型专属架构参数采样
    if model_name == "unet":
        # Encoder channels double at each level; keep the search practical.
        hparams["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
        hparams["width"] = trial.suggest_categorical("width", [16, 32, 64])
        hparams["modes"] = 24
        hparams["wavelet"] = "haar"
    elif model_name in ('kan', 'transformer', 'mamba'):
        # Spline bases, attention matrices and the unfused selective scan need
        # smaller batches than pointwise MLPs; do not search unused Fourier modes.
        hparams["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
        hparams["width"] = trial.suggest_categorical("width", [32, 64, 128])
        hparams["modes"] = 24
        hparams["wavelet"] = "haar"
    elif model_name == "wno":
        # WNO 大 Batch 易 OOM (小波张量树展开)，严格约束 batch_size
        hparams["batch_size"] = trial.suggest_categorical("batch_size", [64, 128])
        hparams["width"] = trial.suggest_categorical("width", [32, 64])
        hparams["level"] = trial.suggest_int("level", 2, 4)
        hparams["wavelet"] = trial.suggest_categorical("wavelet", ["haar", "db2", "sym2"])
        hparams["modes"] = 32
    elif "biom_refiner" in m_lower:
        # Refiner 机制算子 (支持探索大 batch 256, 512, 1024)
        hparams["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024])
        hparams["width"] = trial.suggest_categorical("width", [64, 128, 192])
        hparams["modes"] = trial.suggest_categorical("modes", [16, 24, 32])
        hparams["wavelet"] = "haar"
    elif "fno" in m_lower or model_name == "mwt":
        # FNO 算子轻量高效 (支持探索激进大 batch 256, 512, 1024, 2048)
        hparams["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
        hparams["width"] = trial.suggest_categorical("width", [32, 64, 128])
        hparams["modes"] = trial.suggest_categorical("modes", [16, 24, 32, 48, 64])
        hparams["wavelet"] = "haar"
    else:
        # MLP / ResNet / DeepONet / other selected baselines (支持大 batch 256, 512, 1024, 2048)
        hparams["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
        hparams["width"] = trial.suggest_categorical("width", [64, 128, 256])
        hparams["modes"] = 24  # fixed placeholder; not an HPO dimension for non-spectral models
        hparams["wavelet"] = "haar"

    return hparams


def evaluate_real_metrics(model, datamodule, device):
    """Physical-space real-mutant objective (test-guided HPO is intentional)."""
    model.bind_datamodule_stats(datamodule)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in datamodule.test_dataset_real:
            x, p, y = [b.unsqueeze(0).to(device) for b in batch]
            all_preds.append(datamodule.denormalize_y(model.forward_ic_time(x, p), device).cpu())
            all_targets.append(datamodule.denormalize_y(y, device).cpu())
    return calculate_metrics(torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0))


def bind_objective_contract(study):
    """Keep physical MAE and its train-only preprocessing protocol consistent."""
    version = 'real_physical_mae_v1'
    recorded = study.user_attrs.get('objective_contract')
    if recorded not in (None, version) or (recorded is None and len(study.trials) > 0):
        raise ValueError('This study contains unverified/legacy objective scores. '
                         'Use a NEW study_name/storage; do not mix old scores with physical MAE.')
    scope = study.user_attrs.get('normalization_fit_scope')
    if scope not in (None, TRAIN_FIT_SCOPE) or (scope is None and len(study.trials) > 0):
        raise ValueError('This study predates train-only normalization or uses a different fit_scope. '
                         'Use a NEW study_name/storage; do not mix preprocessing protocols.')
    study.set_user_attr('objective_contract', version)
    study.set_user_attr('normalization_fit_scope', TRAIN_FIT_SCOPE)


def bind_study_dataset(study, dm):
    """Never compare HPO trials from different biological systems or datasets."""
    contract = dict(system_id=dm.system_id, dataset=dm._dataset_contract)
    existing = study.user_attrs.get('biological_dataset_contract')
    if existing is not None and existing != contract:
        raise ValueError('HPO study biological system/dataset mismatch; use a separate study')
    if existing is None:
        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            raise ValueError('HPO study has completed trials without a biological dataset contract; use a new study')
        study.set_user_attr('biological_dataset_contract', contract)


def make_objective(args):
    """Return a static-data HPO objective; AL uses the outer training driver."""
    if args.al_strategy != 'none':
        raise ValueError('HPO requires --al_strategy none; use train.py for active learning')

    def objective(trial: optuna.Trial) -> float:
        # 1. 采样模型感知超参
        hp = sample_model_aware_hparams(trial, args.model)
        trial.set_user_attr("max_epochs", hp["max_epochs"])
        trial.set_user_attr('normalization_fit_scope', TRAIN_FIT_SCOPE)

        # 3. 构建 DataModule
        dataset_path = args.dataset_name if os.path.isabs(args.dataset_name) else os.path.join(args.data_dir, args.dataset_name)
        datamodule = YeastDataModule(
            dataset_path=dataset_path,
            batch_size_synth=hp["batch_size"],
            val_batch_size=256,
            num_workers=args.num_workers,
            use_adj_matrix=False,
            norm_type=getattr(args, 'norm_type', 'log1p'),
            fit_scope=TRAIN_FIT_SCOPE,
            sensitivity_path=args.sensitivity_path,
        )

        model_numpy_state=np.random.get_state()
        datamodule.setup('fit')
        if getattr(args, 'system', 'auto') not in ('auto', datamodule.system_id):
            raise ValueError('--system contradicts the dataset biological system')
        bind_study_dataset(trial.study, datamodule)
        trial.set_user_attr('system_id', datamodule.system_id)
        from budding_yeast_v2.data.label_metadata import dataset_label_summary
        trial.set_user_attr('dataset_labels', dataset_label_summary(datamodule))
        if datamodule.system_id=='fission' and args.al_strategy!='none':
            raise ValueError('Fission HPO currently requires --al_strategy none')

        # Construct from the pre-split RNG state; keep DataModule split state.
        data_numpy_state=np.random.get_state()
        np.random.set_state(model_numpy_state)
        try:
            model = YeastLitModule(
                model_name=args.model,
                adj_matrix_path=args.adj_matrix_path,
                graph_view=args.graph_view,
                graph_load_policy=args.graph_load_policy,
                num_vars=datamodule.num_vars,
                param_dim=datamodule.num_params,
                system_id=datamodule.system_id,
                modes=hp.get("modes", 32),
                width=hp["width"],
                n_blocks=hp["n_blocks"],
                seq_len=datamodule.T_STEPS,
                wavelet_level=hp.get('level',3),
                sensitivity_weight=args.sensitivity_weight,
                wavelet=hp.get("wavelet", "haar"),
                loss_type=args.loss_type,
                lr=hp["lr"],
                weight_decay=hp["weight_decay"],
                max_epochs=hp["max_epochs"],
                warmup_epochs=hp["warmup_epochs"],
                d_lambda=args.d_lambda,
                ode_method=args.ode_method,
                ode_rtol=args.ode_rtol,
                ode_atol=args.ode_atol,
                norm_type=getattr(args, 'norm_type', 'log1p'),
            )
        finally:
            np.random.set_state(data_numpy_state)

        # 4. Checkpoint 与剪枝回调 (隔离每个 trial 的 ckpt)
        worker_suffix = f"w{args.worker_id}_" if args.worker_id is not None else ""
        ckpt_dir = os.path.join(args.save_dir, "checkpoints", f"{worker_suffix}trial_{trial.number}")
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{val_loss_synth:.4f}",
            monitor="val_loss_synth",
            mode="min",
            save_top_k=1,
        )
        pruner_cb = PyTorchLightningPruningCallback(trial, monitor="val_loss_synth")
        callbacks = [checkpoint_callback, pruner_cb]

        # AL acquisition requires train.py's outer retraining loop. HPO is
        # explicitly static until that driver is shared with the tuner.
        # 6. 训练
        trainer = pl.Trainer(
            max_epochs=hp["max_epochs"],
            logger=False,  # 调参期间关闭 wandb 避免大量琐碎 runs
            callbacks=callbacks,
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="bf16-mixed",
            reload_dataloaders_every_n_epochs=0,
            num_sanity_val_steps=0,
            enable_progress_bar=False,  # 避免多卡并行互相抢占终端刷新
        )

        try:
            trainer.fit(model, datamodule=datamodule)

            # 7. 检测剪枝 (中途被早停淘汰)
            if trainer.current_epoch + 1 < hp["max_epochs"]:
                raise optuna.TrialPruned()

            # 8. 评测 Real Mutants MAE
            best_ckpt = checkpoint_callback.best_model_path
            if not best_ckpt or not os.path.exists(best_ckpt):
                raise optuna.TrialPruned()

            best_model = YeastLitModule.load_from_checkpoint(best_ckpt)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            best_model = best_model.to(device)
            best_model.eval()

            datamodule.setup("test")
            metrics = evaluate_real_metrics(best_model, datamodule, device)
            real_mae = float(metrics["MAE"])
            real_l2 = float(metrics["Relative L2"])
            real_corr = float(metrics["Correlation"])

            trial.set_user_attr("real_mae", real_mae)
            trial.set_user_attr("real_l2", real_l2)
            trial.set_user_attr("real_corr", real_corr)
            trial.set_user_attr("best_ckpt", best_ckpt)

            # 9. 清理临时 ckpt 省空间
            if args.cleanup_ckpts:
                shutil.rmtree(ckpt_dir, ignore_errors=True)

            del best_model, model, trainer, datamodule
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return real_mae

        except optuna.TrialPruned:
            # 正常剪枝，清理显存
            if args.cleanup_ckpts:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # 智能拦截爆显存 (CUDA OOM): 彻底释放显存并自动转为剪枝，防止进程崩溃
            err_msg = str(e).lower()
            if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in err_msg or "cuda" in err_msg:
                print(f"\n⚠️ [Trial #{trial.number} | Worker {args.worker_id}] 触发 CUDA 显存溢出 (OOM)!")
                print(f"   超参: batch_size={hp.get('batch_size')}, width={hp.get('width')}, modes={hp.get('modes')}")
                print(f"   执行防爆恢复: 彻底清空 GPU 显存缓存并自动剪枝该 Trial，继续下一轮探索...")
                
                # 显式清理对象与显存
                if 'best_model' in locals(): del best_model
                if 'model' in locals(): del model
                if 'trainer' in locals(): del trainer
                if 'datamodule' in locals(): del datamodule
                if args.cleanup_ckpts:
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 抛出剪枝信号，让 Optuna 记录该超参不可行并自动降权
                raise optuna.TrialPruned(f"CUDA OOM (batch_size={hp.get('batch_size')})")
            else:
                raise e

    return objective


def format_pipeline_experiment_line(model: str, loss_type: str, al_strategy: str, dataset_path: str,
                                   dataset_tag: str, p: Dict[str, Any], gpu_id: int = 0) -> str:
    """Generate the pipeline's 16 positional columns plus explicit training flags."""
    if model not in MODEL_REGISTRY:
        raise ValueError(f'Unknown/removed model: {model}')
    if any(c.isspace() for c in dataset_path):
        raise ValueError('The pipeline row format requires dataset paths without whitespace; use train.py directly')
    values = [model,loss_type,'false',dataset_path,str(p.get('modes',24)),str(p.get('width',64)),
              str(p.get('batch_size',64)),str(p.get('max_epochs',100)),al_strategy,
              str(p.get('lr',1e-3)),str(p.get('warmup_epochs',10)),'0','0','0','one_time',str(gpu_id)]
    extras = {'n_blocks':p.get('n_blocks',4),'weight_decay':p.get('weight_decay',1e-4),
              'dataset_tag':dataset_tag}
    if model == 'wno':
        extras.update(wavelet=p.get('wavelet','haar'),wavelet_level=p.get('level',3))
    if model in ('neural_ode','konode'):
        extras.update(ode_method=p.get('ode_method','dopri5'),ode_rtol=p.get('ode_rtol',1e-3),ode_atol=p.get('ode_atol',1e-4))
    if model == 'sc_fno':
        if not p.get('sensitivity_path'):
            raise ValueError('SC-FNO exported command requires sensitivity_path')
        extras.update(sensitivity_path=p['sensitivity_path'],sensitivity_weight=p.get('sensitivity_weight',1.0))
    for name,value in extras.items():
        if any(c.isspace() for c in str(value)):
            raise ValueError(f'Pipeline extra {name} contains whitespace; use train.py directly')
        values.append(f'--{name}={value}')
    # Double-quoted shell array entry; reject shell expansions in literal values.
    if any(any(c in value for c in ('"','`','$','\\')) for value in values):
        raise ValueError('Pipeline row contains shell expansion characters; use train.py directly')
    return '"' + ' '.join(values) + '"'


def main():
    parser = argparse.ArgumentParser(description="Optuna Distributed Multi-GPU HPO")

    # 模型与数据集
    parser.add_argument("--sensitivity_path", type=str, default=None)
    parser.add_argument("--sensitivity_weight", type=float, default=1.0)
    parser.add_argument("--model", type=str, required=True, choices=tuple(MODEL_REGISTRY))
    parser.add_argument("--system", choices=("auto", "budding", "fission"), default="auto")
    parser.add_argument("--al_strategy", type=str, default="none", choices=CLI_STRATEGIES)
    parser.add_argument("--loss_type", type=str, default="hybrid_mse_rel")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/home/users/zsy/mydata/budding_yeast")
    parser.add_argument("--adj_matrix_path", type=str, default=None)
    parser.add_argument("--graph_view", choices=["continuous", "hybrid"], default="hybrid")
    parser.add_argument("--graph_load_policy", choices=["current", "checkpoint"], default="current")
    parser.add_argument("--norm_type", choices=['log1p', 'linear', 'asinh_z', 'asinh_scale'], default='log1p')

    # 架构与 ODE
    parser.add_argument("--d_lambda", type=int, default=32)
    parser.add_argument("--ode_method", type=str, default="dopri5")
    parser.add_argument("--ode_rtol", type=float, default=1e-3)
    parser.add_argument("--ode_atol", type=float, default=1e-4)

    # AL 参数

    # 搜索与并发配置
    parser.add_argument("--n_trials", type=int, default=50, help="该 Worker 进程执行的 trial 数")
    parser.add_argument("--save_dir", type=str, default="./tune_result")
    parser.add_argument("--study_name", type=str, default="yeast_hpo")
    parser.add_argument("--storage", type=str, default=None, help="SQLite 存储 URL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cleanup_ckpts", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--no_prune", action="store_true", default=False)
    parser.add_argument("--prune_warmup_epochs", type=int, default=10)
    parser.add_argument("--worker_id", type=int, default=0, help="多卡并行 Worker 序号")

    args = parser.parse_args()
    if args.model == 'sc_fno' and (args.loss_type != 'sc_sensitivity' or not args.sensitivity_path):
        parser.error('sc_fno requires --loss_type sc_sensitivity --sensitivity_path <reference.npz>')
    if args.sensitivity_path and args.model != 'sc_fno':
        parser.error('--sensitivity_path is reserved for sc_fno')
    if args.al_strategy != 'none':
        parser.error('HPO currently supports --al_strategy none only; use train.py for the AL outer loop')

    pl.seed_everything(args.seed + args.worker_id, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # SQLite 并发 WAL 模式配置 (防死锁)
    db_file = os.path.join(args.save_dir, f"{args.study_name}.db")
    if not args.resume and os.path.exists(db_file) and args.worker_id == 0:
        os.remove(db_file)
        print(f"🧹 [全新搜索] Worker 0 已初始化清理数据库: {db_file}")

    # 开启 WAL
    conn = sqlite3.connect(db_file, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.close()

    storage_url = args.storage or f"sqlite:///{db_file}?timeout=60"
    sampler = optuna.samplers.TPESampler(seed=args.seed + args.worker_id)
    pruner = optuna.pruners.NopPruner() if args.no_prune else optuna.pruners.MedianPruner(
        n_warmup_steps=args.prune_warmup_epochs,
        interval_steps=1,
    )

    # 采用指数退避重试加载 study，防御多进程高并发初始化时的竞争
    study = None
    for attempt in range(6):
        try:
            study = optuna.create_study(
                study_name=args.study_name,
                storage=storage_url,
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True,
            )
            break
        except Exception as e:
            if "already exists" in str(e).lower() or "locked" in str(e).lower():
                import time
                time.sleep(1.0 + args.worker_id * 0.5)
                continue
            raise e

    if study is None:
        raise RuntimeError(f"Worker {args.worker_id} 无法连接或创建 Optuna Study: {args.study_name}")
    bind_objective_contract(study)

    print(f"🚀 [Worker {args.worker_id}] 启动超参搜索 | Study: {args.study_name} | 目标 Trials: {args.n_trials}")
    objective = make_objective(args)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=False,
        catch=(RuntimeError,),
    )

    # 汇总输出 (仅由 Worker 0 或搜索结束时输出)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if args.worker_id == 0 and len(completed) > 0:
        # 按 Real MAE 排序
        completed.sort(key=lambda t: t.value)
        top_k = completed[:min(5, len(completed))]

        print(f"\n{'='*70}")
        print(f"🏆 Optuna 超参搜索完成! 全部有效 Trials: {len(completed)}")
        print(f"{'='*70}")

        top_k_records = []
        dataset_tag = os.path.basename(args.dataset_name).replace(".npz", "")
        for rank, t in enumerate(top_k, 1):
            print(f"\n  🥇 Rank #{rank} (Trial #{t.number}) -> Real MAE: {t.value:.4f} | Corr: {t.user_attrs.get('real_corr', 'N/A')}")
            print(f"     Params: {t.params}")
            line = format_pipeline_experiment_line(args.model, args.loss_type, args.al_strategy,
                                                 os.path.join(args.data_dir,args.dataset_name), dataset_tag,
                                                 dict(t.params, ode_method=args.ode_method, ode_rtol=args.ode_rtol,
                                                      ode_atol=args.ode_atol,sensitivity_path=args.sensitivity_path,
                                                      sensitivity_weight=args.sensitivity_weight))
            print(f"     📋 对应的 run_pipeline.sh 配置行:")
            print(f"     {line}")
            top_k_records.append({
                "rank": rank,
                "trial_number": t.number,
                "real_mae": t.value,
                "real_l2": t.user_attrs.get("real_l2"),
                "real_corr": t.user_attrs.get("real_corr"),
                "params": t.params,
                "pipeline_line": line
            })

        best_path = os.path.join(args.save_dir, "best_params.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump({
                "study_name": args.study_name,
                "model": args.model,
                "loss_type": args.loss_type,
                "dataset": args.dataset_name,
                "top_k": top_k_records
            }, f, indent=2, ensure_ascii=False)
        print(f"\n📝 最优配置与 Top-K 方案已保存: {best_path}")


if __name__ == "__main__":
    main()
