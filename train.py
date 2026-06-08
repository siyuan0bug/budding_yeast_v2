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
        current_epoch = trainer.current_epoch
        if current_epoch % self.plot_every_n_epochs != 0 and current_epoch != trainer.max_epochs - 1:
            return

        dm = trainer.datamodule
        fixed_samples = dm.fixed_samples_for_plot

        if not fixed_samples:
            return

        pl_module.eval()
        fig, axes = plt.subplots(6, 4, figsize=(24, 24))
        axes = axes.flatten()
        plot_idx = 0

        with torch.no_grad():
            for key, (x, p, y, m_name, p_label) in fixed_samples.items():
                x = x.to(pl_module.device)
                p = p.to(pl_module.device)
                y = y.to(pl_module.device)  # 🌟 加上这一行，把真实的 y 也送到显卡上
                
                mean_y = dm.mean_y.to(pl_module.device)
                std_y = dm.std_y.to(pl_module.device)

                pred_norm = pl_module.forward_ic_time(x, p)
                
                # 反归一化
                pred = (pred_norm * std_y + mean_y).cpu()
                target = (y * std_y + mean_y).cpu()
                
                # 指标计算使用原始张量 (1, 39, 500)
                metrics = calculate_metrics(pred, target)
                
                # 绘图转换为 numpy，并提取第一个样本
                p_np = pred[0].numpy()
                t_np = target[0].numpy()
                
                t = np.linspace(0, 210, 500)
                
                m_str = f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4f} | RelL2: {metrics['Relative L2']:.4f} | Corr: {metrics['Correlation']:.4f}"
                title = f"{m_name} ({p_label}) [S:{self.seed}]\n{m_str}"
                
                ax = axes[plot_idx]
                ax.set_title(title, fontsize=9, fontweight='bold')
                
                lines_truth, lines_pred = [], []
                for i, v_idx in enumerate(CORE_9_INDICES):
                    line_t, = ax.plot(t, t_np[v_idx], '-', linewidth=1.5, label=f'{CORE_9_NAMES[i]} (T)')
                    line_p, = ax.plot(t, p_np[v_idx], '--', linewidth=1.5, label=f'{CORE_9_NAMES[i]} (P)')
                    lines_truth.append(line_t)
                    lines_pred.append(line_p)
                    
                ax.set_xlabel('Time (min)')
                ax.set_ylabel('Concentration')
                ax.grid(True, alpha=0.3)
                plot_idx += 1
                
                if plot_idx >= len(axes): break

        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
            
        handles = [l for pair in zip(lines_truth, lines_pred) for l in pair]
        labels = [h.get_label() for h in handles]
        fig.legend(handles, labels, loc='lower center', ncol=9, bbox_to_anchor=(0.5, 0.0), fontsize=10)
            
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        # 记录到 Wandb
        trainer.logger.experiment.log({"Diagnostics_Real": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)
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
    parser.add_argument('--seed', type=int, default=42)
    
    # Ode 参数
    parser.add_argument('--d_lambda', type=int, default=32)
    parser.add_argument('--ode_method', type=str, default='dopri5')
    parser.add_argument('--ode_rtol', type=float, default=1e-3)
    parser.add_argument('--ode_atol', type=float, default=1e-4)
    
    # 训练配置
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--project', type=str, default='budding_yeast_v2_active')
    parser.add_argument('--name', type=str, default=None)
    
    # 🌟🌟🌟 新增 1：主动学习专属传参 🌟🌟🌟
    parser.add_argument('--al_strategy', type=str, default='none', 
                        choices=['none', 'random', 'us', 'is', 'wrs', 'vessal', 'hggs'],
                        help="主动学习采样策略 (默认 none 代表锁定数据集)")
    parser.add_argument('--al_trigger_epoch', type=int, default=10, help="触发主动学习的 Epoch 间隔")

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
    visualizer = SimToRealVisualizerCallback(plot_every_n_epochs=10, seed=args.seed)

    # 🌟🌟🌟 新增 2：挂载回调拦截器 🌟🌟🌟
    callbacks = [checkpoint_callback, lr_monitor, visualizer]
    
    if args.al_strategy != 'none':
        from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
        print(f"🚀 [Active Learning] 已挂载主动学习引擎，当前策略: {args.al_strategy.upper()}")
        al_cb = ActiveLearningCallback(trigger_every_n_epochs=args.al_trigger_epoch, strategy=args.al_strategy)
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