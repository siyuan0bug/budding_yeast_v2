import os
import time
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, EarlyStopping
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

                pred_denorm = dm.denormalize_y(pred, device)
                target_denorm = dm.denormalize_y(y, device)

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
                if isinstance(trainer.logger, WandbLogger):
                    log_dict[f'{group_name}/{sample_name}'] = wandb.Image(fig)

                plt.close(fig)

        if isinstance(trainer.logger, WandbLogger) and log_dict:
            trainer.logger.experiment.log(log_dict)
        pl_module.train()

class EpochSummaryCallback(Callback):
    """每个 epoch 结束后用 print 输出训练摘要, 确保日志文件中可见 (不依赖 tqdm 终端渲染)."""

    def __init__(self):
        super().__init__()
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epoch_start_time is None:
            return
        epoch = trainer.current_epoch
        m = trainer.callback_metrics
        train_loss = float(m.get('train_loss', float('nan')))
        val_lhs = float(m.get('val_loss_lhs', float('nan')))
        test_lhs = float(m.get('test_loss_lhs', float('nan')))
        test_real = float(m.get('test_loss_real', float('nan')))
        lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else 0.0
        elapsed = time.time() - self.epoch_start_time
        print(f"[Epoch {epoch+1:03d}/{trainer.max_epochs:03d}] "
              f"train_loss={train_loss:.6f} | val_lhs={val_lhs:.6f} | "
              f"test_lhs={test_lhs:.6f} | test_real={test_real:.6f} | "
              f"lr={lr:.2e} | {elapsed:.1f}s", flush=True)

def main():
    parser = argparse.ArgumentParser(description='Train Yeast Model')
    # 基础参数
    parser.add_argument('--model', type=str, default='pure_fno')
    parser.add_argument('--wavelet', type=str, default='haar')
    parser.add_argument('--loss_type', type=str, default='physics_informed')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--adj_matrix_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='/home/users/zsy/mydata/budding_yeast')
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

    # PINN 残差损失参数 (loss_type='pinn_residual' 时生效)
    parser.add_argument('--lambda_phys', type=float, default=0.1,
                        help='PINN: ODE 残差损失权重')
    parser.add_argument('--lambda_jump', type=float, default=1.0,
                        help='PINN: Event 跳跃条件损失权重')
    parser.add_argument('--event_window', type=int, default=4,
                        help='PINN: Event 掩码窗口半径（时间步）')
    parser.add_argument('--event_tol', type=float, default=0.08,
                        help='PINN: 事件触发条件容差')
    parser.add_argument('--jump_ratio', type=float, default=0.3,
                        help='PINN: 跳变检测阈值（占变量值域比例）')
    parser.add_argument('--residual_subsample', type=int, default=1,
                        help='PINN: 残差计算时间步子采样率（1=全部，2=隔步，降低计算量）')
    parser.add_argument('--event_weight', type=float, default=0.1,
                        help='PINN: event 区间残差权重（软掩码，0=完全屏蔽，1=不屏蔽）')
    parser.add_argument('--lambda_penalty', type=float, default=1.0,
                        help='PINN v2: 负值惩罚权重（v1=10.0, v2=1.0）')

    # PINO 损失参数 (loss_type='pino' 时生效)
    # H1 Sobolev 损失: ||u-v||_{H1}² = α·||u-v||_{L2}² + β·||du/dt-dv/dt||_{L2}²
    # 时域 L2 (α) 已含在 MSE 主损失中，这里 β 控制频域梯度约束强度
    parser.add_argument('--lambda_h1', type=float, default=0.1,
                        help='PINO: H1 频域梯度损失权重（warmup 后线性增长到该值）')
    parser.add_argument('--h1_beta', type=float, default=1.0,
                        help='PINO: H1 频域项中 k² 的缩放系数（β in weight=α+β·k²）')

    # 自回归滚动选项
    parser.add_argument('--rollout_type', type=str, default='one_time',
                        choices=['one_time', 'autoregressive'],
                        help='预测方式: one_time=一次性预测(默认), autoregressive=自回归滚动')
    
    parser.add_argument('--norm_type', type=str, default='log1p', choices=['log1p', 'linear'],
                        help='数据归一化类型: log1p (平滑对数归一化, 默认), linear (传统线性Z-score)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help='验证/测试的 batch size（不影响训练，设大加速评估，默认 256）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--project', type=str, default='budding_yeast_v2_mass')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true', default=False,
                        help='禁用 W&B 上传，仅本地日志')
    
    # 🌟🌟🌟 新增 1：主动学习专属传参 🌟🌟🌟
    parser.add_argument('--al_strategy', type=str, default='none',
                          choices=['none', 'random', 'us', 'is', 'wrs', 'vessal', 'hggs', 'rgs', 'pial', 'mutant_aware',
                                   'al4pde_random', 'al4pde_uncertainty', 'al4pde_diversity', 'maal', 'al4pde_mass',
                                   'al4pde_novelty'],
                          help="主动学习采样策略 (默认 none 代表锁定数据集; "
                               "us/is/random=标准池化AL; rgs=Real-Guided局部密集采样; "
                               "pial=Physics-Informed AL (ODE残差+轨迹多样性); "
                               "mutant_aware=基于atlas数据的优先级加权采样, 需配合 --atlas_csv; "
                               "al4pde_random/uncertainty/diversity=al4pde风格池化AL; "
                               "maal=Mutant-Aware AL (diversity+target priority); "
                               "al4pde_mass=借鉴mass_dataset_generator的分层采样, 按通路频率×机制频率优先选择生物学高频典型突变; "
                               "al4pde_novelty=纯随机候选池(无定向采样)+组合新颖性选择: 初始集与候选池均由mass_random_sampler完全随机生成(保留Level分层), "
                               "acquisition=novelty(突变单元全子集组合覆盖)+family_unc(训练集家族不确定性)+diversity(ODE轨迹形状), 对real mutants零泄露. "
                               "注: 全部al4pde策略的候选池与初始集统一为sample_mass_random随机生成 (v2.6)")
    parser.add_argument('--atlas_csv', type=str, default=None,
                        help="mutant_aware 策略专用: mutant_atlas_merged.csv 路径 (含 lhs_MAE_min, lhs_n_samples 列)")
    # al4pde 框架专用参数
    parser.add_argument('--al_pool_dataset', type=str, default=None,
                        help="al4pde/maal 策略专用: 候选池数据集路径 (v5), 含 target 样本 (_LHS_T 后缀)")
    parser.add_argument('--al_target_weight', type=float, default=0.4,
                        help="MAAL 的 target priority 权重 (默认 0.4; diversity 权重 = 1 - target_weight)")
    parser.add_argument('--al_mass_weight', type=float, default=0.5,
                        help="al4pde_mass 的 MASS 生物学分层优先级权重 (默认 0.5; diversity 权重 = 1 - mass_weight)")
    parser.add_argument('--al_novelty_weight', type=float, default=0.6,
                        help="al4pde_novelty 的组合新颖性权重 (默认 0.6; 突变单元(基因,突变类型)组合在训练集中的"
                             "覆盖稀缺度, 单元/对/三元/.../完整组合全子集评估, rescue motif 价值的主要代理, 主导选样)")
    parser.add_argument('--al_family_weight', type=float, default=0.2,
                        help="al4pde_novelty 的家族不确定性权重 (默认 0.2; diversity 权重 = 1 - novelty - family 自动归一化)")
    parser.add_argument('--al_trigger_epoch', type=int, default=10, help="触发主动学习的 Epoch 间隔")
    parser.add_argument('--al_num_add', type=int, default=500, help="每轮 AL 新增样本数 (减小以避免引入噪声)")
    parser.add_argument('--al_perturbation', type=float, default=0.05, help="RGS专用: 连续参数局部扰动幅度 (0.05=±5%%, 减小以提升参数合理性)")
    parser.add_argument('--al_mae_threshold', type=float, default=0.1, help="RGS专用: MAE大于此阈值的Real Mutant优先密集采样")
    # 标准池化 AL 专属参数
    parser.add_argument('--al_initial_train_size', type=int, default=2000,
                        help="AL初始训练集大小 (默认2000; us/is/random=池化AL让AL有发挥空间; "
                             "mutant_aware/rgs=查询合成AL建议设为50000+, 先在全量数据上预训练再触发AL定向采样)")
    parser.add_argument('--al_initial_selection', type=str, default='kmeans',
                        choices=['kmeans', 'random'],
                        help="标准AL初始训练集选择方法 (kmeans=聚类多样性, random=随机)")
    parser.add_argument('--al_diversity_weight', type=float, default=0.5,
                        help="Importance Sampling中多样性权重 (0~1, 默认0.5)")
    parser.add_argument('--al_pool_subset_size', type=int, default=10000,
                        help="每轮AL从池中采样的子集大小 (平衡效率, 默认10000)")
    parser.add_argument('--al_uncertainty_metric', type=str, default='variance',
                        choices=['variance', 'entropy'],
                        help="不确定性度量方法 (variance=预测方差, entropy=预测熵)")

    # 机制与分岔增强超参数 (方案 A, 方案 C, 方案 A+C)
    parser.add_argument('--lambda_pattern', type=float, default=0.1,
                        help="Pattern 表型分类交叉熵损失权重 (默认 0.1)")
    parser.add_argument('--lambda_arrest', type=float, default=0.05,
                        help="Pattern C 阻滞态末端平坦稳态正则权重 (默认 0.05)")
    parser.add_argument('--pino_event_strategy', type=str, default='decoupled',
                        choices=['decoupled', 'huber', 'full'],
                        help="PINO ODE 残差处理 Event 跳变策略 (decoupled=33连续变量, huber=Huber截断, full=传统平方)")
    parser.add_argument('--huber_delta', type=float, default=1.0,
                        help="PINO 鲁棒 Huber 损失截断阈值 delta (默认 1.0)")

    # 实验管理与目录参数
    parser.add_argument('--exp_root', type=str, default='./experiments',
                        help='统一分层实验根目录 (默认 ./experiments)')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='显式指定实验目录 (若指定则优先使用该目录作为输出路径)')
    parser.add_argument('--dataset_tag', type=str, default=None,
                        help='数据集简短标签 (如 massv3, v5, 用于构建分层路径)')
    parser.add_argument('--hparam_tag', type=str, default=None,
                        help='超参附加标签 (用于构建分层路径)')
    parser.add_argument('--use_hierarchical_dir', action='store_true', default=False,
                        help='启用分层语义化目录结构 (experiments/dataset/model/...)')

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    # 导入实验管理工具库
    from budding_yeast_v2.utils.exp_utils import (
        build_experiment_dir,
        generate_hparams_signature,
        save_metadata,
        setup_wandb_logger,
        clean_dataset_tag
    )

    # 确定实际实验输出目录
    run_name = args.name or f"{args.model}_{args.loss_type}_seed{args.seed}"
    if args.exp_dir:
        actual_exp_dir = os.path.abspath(args.exp_dir)
    elif args.use_hierarchical_dir:
        ds_tag = clean_dataset_tag(args.dataset_tag or args.dataset_name)
        hparams_sig = generate_hparams_signature(
            width=args.width, modes=args.modes, batch_size=args.batch_size, lr=args.lr,
            epochs=args.max_epochs,
            model_name=args.model, extra_tag=args.hparam_tag, rollout_type=args.rollout_type,
            wavelet=args.wavelet,
            use_adj=(args.adj_matrix_path is not None),
            warmup=args.warmup_epochs,
            norm_type=args.norm_type
        )
        actual_exp_dir = build_experiment_dir(
            exp_root=args.exp_root, dataset_tag=ds_tag, model_name=args.model,
            loss_type=args.loss_type, al_strategy=args.al_strategy,
            hparams_sig=hparams_sig, seed=args.seed
        )
    else:
        # 兼容旧版扁平目录
        actual_exp_dir = os.path.join(args.save_dir, run_name)

    ckpt_dir = os.path.join(actual_exp_dir, 'checkpoints') if (args.exp_dir or args.use_hierarchical_dir) else actual_exp_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(actual_exp_dir, exist_ok=True)

    # 保存完整运行配置与元数据快照
    save_metadata(actual_exp_dir, args, extra_info={"exp_dir": actual_exp_dir, "ckpt_dir": ckpt_dir})

    def build_fresh_model():
        """构造一个全新随机初始化的模型 (v2.3 al4pde 外层重训: 每轮冷启动)。"""
        return YeastLitModule(
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
            lambda_phys=args.lambda_phys,
            lambda_jump=args.lambda_jump,
            lambda_penalty=args.lambda_penalty,
            event_window=args.event_window,
            event_tol=args.event_tol,
            jump_ratio=args.jump_ratio,
            residual_subsample=args.residual_subsample,
            event_weight=args.event_weight,
            lambda_h1=args.lambda_h1,
            h1_beta=args.h1_beta,
            lambda_pattern=args.lambda_pattern,
            lambda_arrest=args.lambda_arrest,
            pino_event_strategy=args.pino_event_strategy,
            huber_delta=args.huber_delta,
            rollout_type=args.rollout_type,
        )

    model = build_fresh_model()

    datamodule = YeastDataModule(
        dataset_path=os.path.join(args.data_dir, args.dataset_name),
        adj_matrix_path=args.adj_matrix_path if args.adj_matrix_path else 'mydata/yeast_signed_adjacency_matrix.npy',
        batch_size_lhs=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        use_adj_matrix=(args.adj_matrix_path is not None),
        norm_type=args.norm_type,
    )

    # W&B Logger 结构化接入
    logger = setup_wandb_logger(args, exp_dir=actual_exp_dir, project=args.project)

    AL4PDE_STRATEGIES = ('al4pde_random', 'al4pde_uncertainty', 'al4pde_diversity',
                         'maal', 'al4pde_mass', 'al4pde_novelty')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    visualizer = SimToRealVisualizerCallback(plot_every_n_epochs=10, seed=args.seed)

    if args.al_strategy in AL4PDE_STRATEGIES:
        # ================= v2.3 al4pde 外层重训式 AL (方案 A) =================
        from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
        NUM_AL_ROUNDS = 5
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 [al4pde v2.3 外层重训] 策略={args.al_strategy.upper()} | "
              f"{NUM_AL_ROUNDS} 轮 × {args.max_epochs} epoch/轮 | 每轮冷启动 + 全新余弦 | 取最后一轮最后 epoch")

        datamodule.setup('fit')  # 构建数据集 (setup 幂等, 后续 fit 不会重建)

        al_cb = ActiveLearningCallback(
            trigger_every_n_epochs=args.al_trigger_epoch,
            strategy=args.al_strategy,
            num_add=args.al_num_add,
            perturbation=args.al_perturbation,
            mae_threshold=args.al_mae_threshold,
            initial_train_size=args.al_initial_train_size,
            initial_selection=args.al_initial_selection,
            diversity_weight=args.al_diversity_weight,
            pool_subset_size=args.al_pool_subset_size,
            uncertainty_metric=args.al_uncertainty_metric,
            random_seed=args.seed,
            atlas_csv_path=args.atlas_csv,
            al_pool_dataset=args.al_pool_dataset,
            target_weight=args.al_target_weight,
            mass_weight=args.al_mass_weight,
            novelty_weight=args.al_novelty_weight,
            family_weight=args.family_weight if hasattr(args, 'family_weight') else args.al_family_weight,
        )
        al_cb.enable_outer_loop_mode()
        # 一次性: 缩至初始子集 + 生成候选池 + (diversity/maal/boundary) ODE 预计算
        al_cb.al4pde_setup(datamodule, device=device_str)

        trainer = None
        for r in range(NUM_AL_ROUNDS):
            n_train = len(datamodule.train_dataset_lhs)
            print(f"\n===== [al4pde v2.3] AL 轮 {r + 1}/{NUM_AL_ROUNDS} | 当前训练集: {n_train} =====")
            model = build_fresh_model()  # 冷启动: 全新随机初始化 + 全新余弦周期
            trainer = pl.Trainer(
                max_epochs=args.max_epochs,
                logger=logger,
                callbacks=[lr_monitor, visualizer, EpochSummaryCallback()],
                accelerator=args.device if args.device != 'cpu' else 'auto',
                devices=args.devices,
                gradient_clip_val=1.0,
                precision="bf16-mixed",
                strategy='ddp_find_unused_parameters_true' if args.devices > 1 else 'auto',
                reload_dataloaders_every_n_epochs=0,
                num_sanity_val_steps=0,
            )
            trainer.fit(model, datamodule=datamodule)
            # 前 4 轮: 用当前(本轮训练完)模型选样并追加; 末轮只训练不选 (对齐 al4pde is_last)
            if r < NUM_AL_ROUNDS - 1:
                al_cb.al4pde_select_and_append(model, datamodule, device=device_str)

        # 取最后一轮最后 epoch 权重保存 (唯一 ckpt, run_pipeline.sh 用 ls *.ckpt|head -1 发现)
        final_ckpt = os.path.join(ckpt_dir, 'final-last-epoch.ckpt')
        trainer.save_checkpoint(final_ckpt)
        print(f"✅ [al4pde v2.3] 已保存最终模型 (最后一轮最后 epoch): {final_ckpt}")
    else:
        # ================= 非 al4pde: 单次 fit (最优权重依据 val_loss_lhs) =================
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='best-{epoch:02d}-{val_loss_lhs:.4f}',
            monitor='val_loss_lhs',
            mode='min',
            save_top_k=1,
        )
        callbacks = [checkpoint_callback, lr_monitor, visualizer, EpochSummaryCallback()]

        if args.al_strategy != 'none':
            from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
            print(f"🚀 [Active Learning] 已挂载主动学习引擎，当前策略: {args.al_strategy.upper()}")
            al_cb = ActiveLearningCallback(
                  trigger_every_n_epochs=args.al_trigger_epoch,
                  strategy=args.al_strategy,
                  num_add=args.al_num_add,
                  perturbation=args.al_perturbation,
                  mae_threshold=args.al_mae_threshold,
                  initial_train_size=args.al_initial_train_size,
                  initial_selection=args.al_initial_selection,
                  diversity_weight=args.al_diversity_weight,
                  pool_subset_size=args.al_pool_subset_size,
                  uncertainty_metric=args.al_uncertainty_metric,
                  random_seed=args.seed,
                  atlas_csv_path=args.atlas_csv,
                  al_pool_dataset=args.al_pool_dataset,
                  target_weight=args.al_target_weight,
                  mass_weight=args.al_mass_weight,
                  novelty_weight=args.al_novelty_weight,
                  family_weight=args.family_weight if hasattr(args, 'family_weight') else args.al_family_weight,
              )
            callbacks.append(al_cb)
        else:
            print("🌱 [Baseline Mode] 未启用主动学习，数据集在全生命周期锁定。")

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger=logger,
            callbacks=callbacks,
            accelerator=args.device if args.device != 'cpu' else 'auto',
            devices=args.devices,
            gradient_clip_val=1.0,
            precision="bf16-mixed",
            strategy='ddp_find_unused_parameters_true' if args.devices > 1 else 'auto',
            reload_dataloaders_every_n_epochs=args.al_trigger_epoch if args.al_strategy != 'none' else 0,
            num_sanity_val_steps=0
        )
        trainer.fit(model, datamodule=datamodule)
    
    wandb.finish()

if __name__ == '__main__':
    main()