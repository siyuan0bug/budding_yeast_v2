"""
Optuna 超参搜索脚本
==================
用法示例:
    # 搜索 hyper_fno + 无 AL 的训练超参（最常见）
    python tune.py --model hyper_fno --al_strategy none \
        --dataset_name lhs_v1_210m_500s --n_trials 100

    # 搜索 hyper_fno + RGS 主动学习的训练超参（每轮 trial 较慢）
    python tune.py --model hyper_fno --al_strategy rgs \
        --dataset_name lhs_v1_210m_500s --n_trials 50 \
        --al_trigger_epoch 10 --al_num_add 5000

    # 中断后恢复（自动从 SQLite 续跑）
    python tune.py --model hyper_fno --al_strategy none \
        --dataset_name lhs_v1_210m_500s --n_trials 100 --resume

说明:
    - 模型名(--model)和 AL 策略(--al_strategy)由你固定，不参与搜索
    - 搜索空间: lr, weight_decay, warmup_epochs, batch_size, max_epochs,
                modes, width, n_blocks
    - batch_size 范围 256~4096（针对大显存优化），lr/warmup 范围已相应放大
    - 优化目标: Real Mutants MAE（越小越好）
    - 中期剪枝: 训练中监控 val_loss_lhs，差 trial 提前终止
    - 结果: save_dir/best_params.json + optuna_study.db（可可视化/恢复）
"""
import os
import sys
import gc
import json
import argparse
import shutil

# 确保能从父目录导入 budding_yeast_v2 包（与 run_pipeline.sh 的 PYTHONPATH 逻辑一致）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# PL 2.x 的剪枝回调从 optuna_integration 导入（optuna 4.9+ 新路径）
try:
    from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
except ImportError:
    from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.utils.metrics import calculate_metrics


def make_objective(args):
    """返回 objective(trial) 闭包，捕获用户固定的配置。"""

    def objective(trial: optuna.Trial) -> float:
        # ===== 1. 采样超参（搜索空间）=====
        # --- 训练超参（针对大 batch 显存优化）---
        batch_size = trial.suggest_categorical("batch_size", [64,128,256, 512, 1024, 2048, 4096])
        # 大 batch 需更大 lr（线性缩放规则），范围相应放大
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        # 大 batch 正则化需求降低，下限放宽
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        # 大 batch 每 epoch 迭代少，warmup 需更长
        warmup_epochs = trial.suggest_int("warmup_epochs", 5, 40)
        # 大 batch 单 epoch 快，可跑更多 epoch
        max_epochs = trial.suggest_int("max_epochs", 100, 500, step=50)

        # --- 模型结构超参 ---
        modes = trial.suggest_categorical("modes", [16, 24, 32, 48, 64])
        width = trial.suggest_categorical("width", [16, 32, 48, 64, 96])
        n_blocks = trial.suggest_int("n_blocks", 2, 6)

        trial.set_user_attr("max_epochs", max_epochs)

        # ===== 2. 构建模型（结构参数由 trial 采样）=====
        model = YeastLitModule(
            model_name=args.model,
            num_vars=39,
            param_dim=141,
            modes=modes,
            width=width,
            n_blocks=n_blocks,
            loss_type=args.loss_type,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
            d_lambda=args.d_lambda,
            ode_method=args.ode_method,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
        )

        # ===== 3. 构建数据模块 =====
        datamodule = YeastDataModule(
            dataset_path=os.path.join(args.data_dir, args.dataset_name),
            adj_matrix_path=args.adj_matrix_path if args.adj_matrix_path else "mydata/yeast_signed_adjacency_matrix.npy",
            batch_size_lhs=batch_size,
            val_batch_size=256,
            num_workers=args.num_workers,
            use_adj_matrix=(args.adj_matrix_path is not None),
        )

        # ===== 4. 回调：每 trial 独立 ckpt 目录 + 剪枝 =====
        ckpt_dir = os.path.join(args.save_dir, "checkpoints", f"trial_{trial.number}")
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{val_loss_lhs:.4f}",
            monitor="val_loss_lhs",
            mode="min",
            save_top_k=1,
        )
        pruner_cb = PyTorchLightningPruningCallback(trial, monitor="val_loss_lhs")
        # 注意: 不使用 LearningRateMonitor，因为它需要 logger（HPO 关闭了 wandb）
        callbacks = [checkpoint_callback, pruner_cb]

        # ===== 5. 主动学习回调（用户固定策略）=====
        if args.al_strategy != "none":
            from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
            callbacks.append(
                ActiveLearningCallback(
                    trigger_every_n_epochs=args.al_trigger_epoch,
                    strategy=args.al_strategy,
                    num_add=args.al_num_add,
                    perturbation=args.al_perturbation,
                    mae_threshold=args.al_mae_threshold,
                    # 标准池化 AL 参数 (使用默认值，HPO 不搜索这些)
                    initial_train_size=args.al_initial_train_size,
                    initial_selection=args.al_initial_selection,
                    diversity_weight=args.al_diversity_weight,
                    pool_subset_size=args.al_pool_subset_size,
                    uncertainty_metric=args.al_uncertainty_metric,
                    random_seed=args.seed,
                )
            )

        # ===== 6. 训练 =====
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=False,  # HPO 期间关闭 wandb，避免大量 run 干扰
            callbacks=callbacks,
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=args.devices,
            precision="bf16-mixed",
            strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
            reload_dataloaders_every_n_epochs=args.al_trigger_epoch if args.al_strategy != "none" else 0,
            num_sanity_val_steps=0,
            enable_progress_bar=True,
        )

        trainer.fit(model, datamodule=datamodule)

        # ===== 7. 检测是否被剪枝（训练提前结束）=====
        # 若训练在 max_epochs 之前终止，说明被 MedianPruner 剪枝
        if trainer.current_epoch + 1 < max_epochs:
            raise optuna.TrialPruned()

        # ===== 8. 加载最优 ckpt，评估 Real Mutants MAE =====
        best_ckpt = checkpoint_callback.best_model_path
        if not best_ckpt or not os.path.exists(best_ckpt):
            raise optuna.TrialPruned()

        best_model = YeastLitModule.load_from_checkpoint(best_ckpt)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        best_model = best_model.to(device)
        best_model.eval()

        datamodule.setup("test")
        mean_y = datamodule.mean_y.to(device)
        std_y = datamodule.std_y.to(device)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in datamodule.test_dataset_real:
                x, p, y = [b.unsqueeze(0).to(device) for b in batch]
                pred_denorm = (best_model.forward_ic_time(x, p) * std_y + mean_y).cpu()
                target_denorm = (y * std_y + mean_y).cpu()
                all_preds.append(pred_denorm)
                all_targets.append(target_denorm)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_preds, all_targets)
        real_mae = metrics["MAE"]

        trial.set_user_attr("real_mae", real_mae)
        trial.set_user_attr("real_corr", metrics["Correlation"])
        trial.set_user_attr("best_ckpt", best_ckpt)

        # ===== 9. 清理 ckpt 省磁盘（可选）=====
        if args.cleanup_ckpts:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

        # 释放显存
        del best_model, model, trainer, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return real_mae

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna 超参搜索")

    # ---- 用户固定：模型与 AL 策略 ----
    parser.add_argument("--model", type=str, required=True,
                        choices=["pure_fno", "cross_fno", "ultimate_fno", "wno",
                                 "lno", "hyper_fno", "hp_fno", "neural_ode"],
                        help="模型名（固定，不搜索）")
    parser.add_argument("--al_strategy", type=str, default="none",
                        choices=["none", "random", "us", "is", "wrs", "vessal", "hggs", "rgs", "pial"],
                        help="主动学习策略（固定，不搜索）。pial=Physics-Informed AL (ODE残差+轨迹多样性)")

    # ---- 用户固定：模型结构与 ODE 参数（modes/width/n_blocks 已加入搜索空间）----
    parser.add_argument("--loss_type", type=str, default="mse_only")
    parser.add_argument("--d_lambda", type=int, default=32)
    parser.add_argument("--ode_method", type=str, default="dopri5")
    parser.add_argument("--ode_rtol", type=float, default=1e-3)
    parser.add_argument("--ode_atol", type=float, default=1e-4)

    # ---- 用户固定：AL 超参 ----
    parser.add_argument("--al_trigger_epoch", type=int, default=10)
    parser.add_argument("--al_num_add", type=int, default=5000)
    parser.add_argument("--al_perturbation", type=float, default=0.1)
    parser.add_argument("--al_mae_threshold", type=float, default=0.1)
    # 标准池化 AL 参数 (HPO 不搜索，使用默认值)
    parser.add_argument("--al_initial_train_size", type=int, default=5000,
                        help="标准AL初始训练集大小 (默认5000)")
    parser.add_argument("--al_initial_selection", type=str, default='kmeans',
                        choices=['kmeans', 'random'],
                        help="标准AL初始训练集选择方法")
    parser.add_argument("--al_diversity_weight", type=float, default=0.5,
                        help="Importance Sampling中多样性权重")
    parser.add_argument("--al_pool_subset_size", type=int, default=10000,
                        help="每轮AL从池中采样的子集大小")
    parser.add_argument("--al_uncertainty_metric", type=str, default='variance',
                        choices=['variance', 'entropy'],
                        help="不确定性度量方法")

    # ---- 数据 ----
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--adj_matrix_path", type=str, default=None)

    # ---- 搜索配置 ----
    parser.add_argument("--n_trials", type=int, default=100, help="总 trial 数")
    parser.add_argument("--save_dir", type=str, default="./tune_result")
    parser.add_argument("--study_name", type=str, default="yeast_hpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cleanup_ckpts", action="store_true",
                        help="评估后删除该 trial 的 ckpt 省磁盘（默认保留）")
    parser.add_argument("--resume", action="store_true",
                        help="从已有 study 续跑（需相同 study_name + save_dir）")
    # 剪枝配置
    parser.add_argument("--no_prune", action="store_true",
                        help="禁用中期剪枝（不推荐，会浪费算力）")
    parser.add_argument("--prune_warmup_epochs", type=int, default=10,
                        help="前 N 个 epoch 不剪枝，让模型先热身")

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # ===== 创建/恢复 Study =====
    storage = f"sqlite:///{os.path.join(args.save_dir, 'optuna_study.db')}"
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.NopPruner() if args.no_prune else optuna.pruners.MedianPruner(
        n_warmup_steps=args.prune_warmup_epochs,
        interval_steps=1,
    )

    # 全新搜索时删除旧 db，避免 DuplicatedStudyError；续跑时保留
    db_path = os.path.join(args.save_dir, "optuna_study.db")
    if not args.resume and os.path.exists(db_path):
        os.remove(db_path)
        print(f"  [全新搜索] 已删除旧的 study 数据库: {db_path}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",  # 最小化 Real MAE
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume,
    )

    n_existing = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"{'='*60}")
    print(f"  Optuna 超参搜索")
    print(f"  模型: {args.model} | AL 策略: {args.al_strategy}")
    print(f"  搜索空间: lr, weight_decay, warmup_epochs, batch_size, max_epochs, modes, width, n_blocks")
    print(f"  优化目标: Real Mutants MAE (越小越好)")
    print(f"  总 trials: {args.n_trials} | 已完成: {n_existing} | 剪枝: {'关闭' if args.no_prune else '开启'}")
    print(f"  存储路径: {storage}")
    print(f"{'='*60}")

    # ===== 运行搜索 =====
    objective = make_objective(args)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
        catch=(RuntimeError,),
    )

    # ===== 输出最优结果 =====
    print(f"\n{'='*60}")
    print(f"  搜索完成!")
    print(f"{'='*60}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  完成: {len(completed)} | 剪枝: {len(pruned)} | 总计: {len(study.trials)}")

    if study.best_trial is not None:
        best = study.best_trial
        print(f"\n  🏆 最优 Trial #{best.number}")
        print(f"     Real MAE: {best.value:.4f}")
        print(f"     Real Corr: {best.user_attrs.get('real_corr', 'N/A')}")
        print(f"     最优 ckpt: {best.user_attrs.get('best_ckpt', 'N/A')}")
        print(f"\n     最优超参:")
        for k, v in best.params.items():
            print(f"       {k}: {v}")

        best_result = {
            "trial_number": best.number,
            "real_mae": best.value,
            "real_corr": best.user_attrs.get("real_corr"),
            "best_ckpt": best.user_attrs.get("best_ckpt"),
            "params": best.params,
            "fixed_config": {
                "model": args.model,
                "al_strategy": args.al_strategy,
                "loss_type": args.loss_type,
            },
        }
        best_path = os.path.join(args.save_dir, "best_params.json")
        with open(best_path, "w") as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
        print(f"\n  📝 最优参数已保存: {best_path}")

    # ===== 导出全部 trial 历史 CSV =====
    csv_path = os.path.join(args.save_dir, "all_trials.csv")
    with open(csv_path, "w") as f:
        f.write("trial,state,real_mae,lr,weight_decay,warmup_epochs,batch_size,max_epochs,modes,width,n_blocks\n")
        for t in study.trials:
            state = t.state.name
            mae = t.value if t.value is not None else t.user_attrs.get("real_mae", "")
            p = t.params
            f.write(f"{t.number},{state},{mae},"
                    f"{p.get('lr','')},{p.get('weight_decay','')},"
                    f"{p.get('warmup_epochs','')},{p.get('batch_size','')},"
                    f"{p.get('max_epochs','')},{p.get('modes','')},"
                    f"{p.get('width','')},{p.get('n_blocks','')}\n")
    print(f"  📝 全部历史已保存: {csv_path}")

    print(f"\n  💡 可视化: optuna-dashboard {storage}")
    print(f"     或运行: python -c \"import optuna; optuna.visualization.plot_optimization_history(study).show()\"")


if __name__ == "__main__":
    main()
