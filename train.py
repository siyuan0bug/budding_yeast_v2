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
import swanlab  # 🌟 新增导入

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.utils.metrics import calculate_metrics

CORE_9_INDICES = [0, 1, 2, 3, 4, 20, 22, 33, 35]
CORE_9_NAMES = ['MASS (0)', 'CLN2 (1)', 'CLB2 (2)', 'CLB5 (3)', 'SIC1 (4)', 'CDC20 (20)', 'CDH1 (22)', 'ORI (33)', 'SPN (35)']
# 🌟 新增变量名列表
VARIABLE_NAMES = [
    "MASS", "CLN2", "CLB2", "CLB5", "SIC1", "CDC6", "C2", "C5", "F2", "F5",
    "SIC1P", "C2P", "C5P", "CDC6P", "F2P", "F5P", "SWI5T", "SWI5", "IEP", "CDC20T",
    "CDC20", "CDH1T", "CDH1", "CDC14T", "CDC14", "NET1T", "NET1", "RENT", "TEM1", "CDC15",
    "PPX", "PDS1", "ESP1", "ORI", "BUD", "SPN", "Vi20", "lte1", "BUB2"
]

class SimToRealVisualizerCallback(Callback):
    def __init__(self, plot_every_n_epochs=5, seed=42):
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.seed = seed  # 🌟 保存 seed


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

                # 反归一化
                pred_denorm = pred * std_y + mean_y
                target_denorm = y * std_y + mean_y

                # 🌟 提取全部 4 个评测指标
                metrics = calculate_metrics(pred_denorm.cpu(), target_denorm.cpu())
                m_str = f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4e} | Rel L2: {metrics['Relative L2']:.4f} | Corr: {metrics['Correlation']:.4f}"

                # 5x8 全变量网格
                fig, axes = plt.subplots(5, 8, figsize=(28, 16))
                axes = axes.flatten()

                for var_idx in range(39):
                    ax = axes[var_idx]
                    pred_np = pred_denorm[0, var_idx, :].cpu().numpy()
                    target_np = target_denorm[0, var_idx, :].cpu().numpy()

                    # 🌟 恢复真实的生物学时间轴 (210 分钟)
                    T_len = len(pred_np)
                    t_max = getattr(dm, 't_max', 210) # 获取动态时间，获取不到默认 210
                    t = np.linspace(0, t_max, T_len)
                    
                    ax.plot(t, target_np, 'b-', label='Ground Truth', linewidth=2)
                    ax.plot(t, pred_np, 'r--', label='Prediction', linewidth=2)
                    
                    # 🌟 修改点：替换原来的 ax.set_title
                    var_name = VARIABLE_NAMES[var_idx] if var_idx < len(VARIABLE_NAMES) else f"Var {var_idx}"
                    ax.set_title(f'{var_idx}: {var_name}', fontweight='bold', fontsize=10)

                    ax.set_xlabel('Time (min)')
                    if var_idx == 0: ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                for i in range(39, len(axes)):
                    axes[i].set_visible(False)

                # 🌟 富文本标题
                title_str = f"[{pat_label}] {name} (Seed: {self.seed} | Epoch {trainer.current_epoch + 1})\n{m_str}"
                plt.suptitle(title_str, fontsize=22, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.96]) # 防止 suptitle 重叠

                group_name = "Diagnostics_LHS" if "LHS" in key else "Diagnostics_Real"
                if trainer.logger and hasattr(trainer.logger, 'experiment'):
                    log_dict[f'{group_name}/{key}'] = wandb.Image(fig)

                plt.close(fig)

        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log(log_dict)
        pl_module.train()


def main():
    #swanlab.sync_wandb(wandb_run=False)
    parser = argparse.ArgumentParser(description='Train Yeast Dynamics Models')

    parser.add_argument('--model', type=str, default='pure_fno',
                        choices=['pure_fno', 'cross_fno', 'ultimate_fno',
                                 'wno', 'lno', 'hyper_fno', 'hp_fno', 'neural_ode'],
                        help='Model architecture')
    parser.add_argument('--loss_type', type=str, default='physics_informed',
                        choices=['physics_informed', 'mse_only', 'mse_negpen', 'mse_smooth', 't_smooth'],
                        help='Loss function type')

    parser.add_argument('--num_vars', type=int, default=39)
    parser.add_argument('--param_dim', type=int, default=141)
    parser.add_argument('--modes', type=int, default=24)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--n_blocks', type=int, default=4)

    parser.add_argument('--wavelet', type=str, default='haar',
                        choices=['haar', 'db1', 'db2', 'db3', 'db4', 'sym2', 'sym3', 'coif1'],
                        help='Wavelet type for WNO model')
    parser.add_argument('--d_lambda', type=int, default=32,
                        help='Hyper latent dimension for HyperFNO')
    parser.add_argument('--ode_method', type=str, default='dopri5',
                        choices=['dopri5', 'rk4', 'euler', 'midpoint', 'adaptive_heun'],
                        help='ODE solver method for Neural ODE')
    parser.add_argument('--ode_rtol', type=float, default=1e-3)
    parser.add_argument('--ode_atol', type=float, default=1e-4)

    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--max_epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use')

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str, default='lhs_massive_210min_500steps_dual_labels.npz',
                        help='Dataset filename for different time-length experiments')
    parser.add_argument('--adj_matrix_path', type=str, default=None)
    parser.add_argument('--causal_matrix_path', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--project', type=str, default='yeast-dynamics-v2')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./train_result')

    args = parser.parse_args()

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')

    adj_matrix = None
    if args.adj_matrix_path and os.path.exists(args.adj_matrix_path):
        if args.adj_matrix_path.endswith('.npy'):
            adj_matrix = np.load(args.adj_matrix_path)
        else:
            adj_matrix = torch.load(args.adj_matrix_path)
    else:
        print("当前未传入邻接矩阵，模型将运行在纯数据驱动模式下。")

    causal_matrix = None
    if args.causal_matrix_path and os.path.exists(args.causal_matrix_path):
        causal_matrix = torch.load(args.causal_matrix_path)

    model = YeastLitModule(
        model_name=args.model,
        num_vars=args.num_vars,
        param_dim=args.param_dim,
        modes=args.modes,
        width=args.width,
        n_blocks=args.n_blocks,
        adj_matrix=adj_matrix,
        causal_matrix=causal_matrix,
        loss_type=args.loss_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        wavelet=args.wavelet,
        d_lambda=args.d_lambda,
        ode_method=args.ode_method,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
    )

    datamodule = YeastDataModule(
        dataset_path=os.path.join(args.data_dir, args.dataset_name),
        adj_matrix_path=args.adj_matrix_path if args.adj_matrix_path else 'mydata/yeast_signed_adjacency_matrix.npy',
        batch_size_lhs=args.batch_size,
        val_batch_size=args.batch_size,
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
    visualizer = SimToRealVisualizerCallback(plot_every_n_epochs=10, seed=args.seed) # 🌟 传入当前的 seed

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, visualizer],
        accelerator='gpu',                 # 🌟 指定使用 GPU
        devices=args.devices,              # 🌟 动态接收显卡数量
        strategy='ddp' if args.devices > 1 else 'auto', # 🌟 多卡时自动启用 DDP 分布式训练
        log_every_n_steps=10,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
    )


if __name__ == '__main__':
    main()