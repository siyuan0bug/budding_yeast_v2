import os
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.utils.metrics import calculate_metrics

VARIABLE_NAMES = [
    "MASS", "CLN2", "CLB2", "CLB5", "SIC1", "CDC6", "C2", "C5", "F2", "F5",
    "SIC1P", "C2P", "C5P", "CDC6P", "F2P", "F5P", "SWI5T", "SWI5", "IEP", "CDC20T",
    "CDC20", "CDH1T", "CDH1", "CDC14T", "CDC14", "NET1T", "NET1", "RENT", "TEM1", "CDC15",
    "PPX", "PDS1", "ESP1", "ORI", "BUD", "SPN", "Vi20", "lte1", "BUB2"
]

class SimToRealVisualizerCallback(Callback):
    def __init__(self, plot_every_n_epochs=10, seed=42):
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.seed = seed

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.plot_every_n_epochs != 0:
            return

        dm = trainer.datamodule
        if dm is None or dm.fixed_samples_for_plot is None:
            return

        device = pl_module.device
        pl_module.eval()

        mean_y = dm.mean_y.to(device)
        std_y = dm.std_y.to(device)
        log_dict = {'epoch': trainer.current_epoch + 1}

        with torch.no_grad():
            for key, (x, p, y, name, pat_label) in dm.fixed_samples_for_plot.items():
                x, p, y = x.to(device), p.to(device), y.to(device)
                pred = pl_module.forward_ic_time(x, p)

                pred_denorm = pred * std_y + mean_y
                target_denorm = y * std_y + mean_y

                metrics = calculate_metrics(pred_denorm.cpu(), target_denorm.cpu())
                m_str = f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4e} | Rel L2: {metrics['Relative L2']:.4f} | Corr: {metrics['Correlation']:.4f}"

                fig, axes = plt.subplots(5, 8, figsize=(28, 16))
                axes = axes.flatten()

                for var_idx in range(39):
                    ax = axes[var_idx]
                    pred_np = pred_denorm[0, var_idx, :].cpu().numpy()
                    target_np = target_denorm[0, var_idx, :].cpu().numpy()

                    T_len = len(pred_np)
                    t_max = getattr(dm, 't_max', 210)
                    t = np.linspace(0, t_max, T_len)

                    ax.plot(t, target_np, 'b-', label='Ground Truth', linewidth=2)
                    ax.plot(t, pred_np, 'r--', label='Prediction', linewidth=2)

                    var_name = VARIABLE_NAMES[var_idx] if var_idx < len(VARIABLE_NAMES) else f"Var {var_idx}"
                    ax.set_title(f'{var_idx}: {var_name}', fontweight='bold', fontsize=10)
                    ax.set_xlabel('Time (min)')
                    if var_idx == 0: ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                for i in range(39, len(axes)):
                    axes[i].set_visible(False)

                title_str = f"[{pat_label}] {name} (Seed: {self.seed} | Epoch {trainer.current_epoch + 1})\n{m_str}"
                plt.suptitle(title_str, fontsize=22, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                # key 格式: "LHS_train/Pattern_A_0", "LHS_val/Pattern_B_1", "real_mutants/Pattern_C_5"
                # group = key 的第一段, 样本名 = key 的第二段
                parts = key.split('/', 1)
                group_name = parts[0]  # LHS_train, LHS_val, LHS_test, real_mutants
                sample_name = parts[1] if len(parts) > 1 else key
                if trainer.logger and hasattr(trainer.logger, 'experiment'):
                    log_dict[f'{group_name}/{sample_name}'] = wandb.Image(fig)

                plt.close(fig)

        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log(log_dict)
        pl_module.train()

def main():
    parser = argparse.ArgumentParser(description='Train Yeast Model')
    # 基础参数
    parser.add_argument('--model', type=str, default='pure_fno')
    parser.add_argument('--wavelet', type=str, default='haar')
    parser.add_argument('--loss_type', type=str, default='physics_informed')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--adj_matrix_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./train_result')
    
    # 超参数
    parser.add_argument('--modes', type=int, default=24)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Cosine+Warmup 的线性预热 epoch 数')
    parser.add_argument('--seed', type=int, default=42)
    
    # Ode 参数
    parser.add_argument('--d_lambda', type=int, default=32)
    parser.add_argument('--ode_method', type=str, default='dopri5')
    parser.add_argument('--ode_rtol', type=float, default=1e-3)
    parser.add_argument('--ode_atol', type=float, default=1e-4)
    
    # 训练配置
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help='验证/测试的 batch size（不影响训练，设大加速评估，默认 256）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--project', type=str, default='budding_yeast_v2_active')
    parser.add_argument('--name', type=str, default=None)
    
    # 🌟🌟🌟 新增 1：主动学习专属传参 🌟🌟🌟
    parser.add_argument('--al_strategy', type=str, default='none',
                          choices=['none', 'random', 'us', 'is', 'wrs', 'vessal', 'hggs', 'rgs'],
                          help="主动学习采样策略 (默认 none 代表锁定数据集; rgs=Real-Guided局部密集采样)")
    parser.add_argument('--al_trigger_epoch', type=int, default=10, help="触发主动学习的 Epoch 间隔")
    parser.add_argument('--al_num_add', type=int, default=5000, help="每轮 AL 新增样本数")
    parser.add_argument('--al_perturbation', type=float, default=0.1, help="RGS专用: 连续参数局部扰动幅度 (0.1=±10%%)")
    parser.add_argument('--al_mae_threshold', type=float, default=0.1, help="RGS专用: MAE大于此阈值的Real Mutant优先密集采样")

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

    model = YeastLitModule(
        model_name=args.model,
        wavelet=args.wavelet,
        num_vars=39,
        param_dim=141,
        modes=args.modes,
        width=args.width,
        n_blocks=args.n_blocks,
        loss_type=args.loss_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        d_lambda=args.d_lambda,
        ode_method=args.ode_method,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
    )

    datamodule = YeastDataModule(
        dataset_path=os.path.join(args.data_dir, args.dataset_name),
        adj_matrix_path=args.adj_matrix_path if args.adj_matrix_path else 'mydata/yeast_signed_adjacency_matrix.npy',
        batch_size_lhs=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        use_adj_matrix=(args.adj_matrix_path is not None),
    )

    run_name = args.name or f"{args.model}_{args.loss_type}_seed{args.seed}"
    logger = WandbLogger(project=args.project, name=run_name)
    logger.log_hyperparams(vars(args))

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, run_name),
        filename='best-{epoch:02d}-{val_loss_lhs:.4f}',
        monitor='val_loss_lhs',
        mode='min',
        save_top_k=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    visualizer = SimToRealVisualizerCallback(plot_every_n_epochs=10, seed=args.seed)

    # 🌟🌟🌟 新增 2：挂载回调拦截器 🌟🌟🌟
    callbacks = [checkpoint_callback, lr_monitor, visualizer]
    
    if args.al_strategy != 'none':
        from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
        print(f"🚀 [Active Learning] 已挂载主动学习引擎，当前策略: {args.al_strategy.upper()}")
        al_cb = ActiveLearningCallback(
              trigger_every_n_epochs=args.al_trigger_epoch,
              strategy=args.al_strategy,
              num_add=args.al_num_add,
              perturbation=args.al_perturbation,
              mae_threshold=args.al_mae_threshold,
          )
        callbacks.append(al_cb)
    else:
        print("🌱 [Baseline Mode] 未启用主动学习，数据集在全生命周期锁定。")

    # 🌟🌟🌟 新增 3：修改 Trainer，强制刷新 DataLoader 🌟🌟🌟
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator=args.device if args.device != 'cpu' else 'auto',
        devices=args.devices,

        # 🌟🌟🌟 新增这一行：开启 BF16 混合精度加速 🌟🌟🌟
        precision="bf16-mixed",

        strategy='ddp_find_unused_parameters_true' if args.devices > 1 else 'auto',
        # 【关键】开启后，每次触发主动学习添加新数据后，底层数据集才会立刻生效
        reload_dataloaders_every_n_epochs=args.al_trigger_epoch if args.al_strategy != 'none' else 0,
        num_sanity_val_steps=0 # 建议设为0，防止 AL 框架在第0步引发多余逻辑
    )

    trainer.fit(model, datamodule=datamodule)
    
    #print("\n🔬 开始最终测试阶段...")
    #trainer.test(model, datamodule=datamodule, ckpt_path='best')
    
    wandb.finish()

if __name__ == '__main__':
    main()