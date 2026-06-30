

"""
过滤训练集实验：只保留 MAE > 阈值的真实突变体母体的 LHS 子样本作为训练集，
验证集和测试集保持不变。用于验证真实 mutant 指标是否与数据集分布有关。

参数来自 best_params.json，输出文件夹带特殊 tag 区分。

用法:
    cd /home/users/zsy/trae_workspace/budding_yeast_v2
    python train_eval_filtered.py \
        --csv_path eval_result/hyper_fno_none_mse_only_210m_500s_modes32_seed42/per_mutant_metrics.csv \
        --mae_threshold 0.35 \
        --best_params tune_result/best_params.json \
        --dataset_name /home/users/zsy/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels.npz
"""

import os
import sys
import csv
import json
import shutil
import argparse

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

# 确保能从父目录导入 budding_yeast_v2 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.utils.metrics import calculate_metrics


VARIABLE_NAMES = [
    "MASS", "CLN2", "CLB2", "CLB5", "SIC1", "CDC6", "C2", "C5", "F2", "F5",
    "SIC1P", "C2P", "C5P", "CDC6P", "F2P", "F5P", "SWI5T", "SWI5", "IEP", "CDC20T",
    "CDC20", "CDH1T", "CDH1", "CDC14T", "CDC14", "NET1T", "NET1", "RENT", "TEM1", "CDC15",
    "PPX", "PDS1", "ESP1", "ORI", "BUD", "SPN", "Vi20", "lte1", "BUB2"
]

CORE_9_INDICES = [0, 1, 2, 3, 4, 20, 22, 33, 35]


# ============================================================
# 防过拟合措施 4: train/val gap 监控回调
# ============================================================
class TrainValGapLogger(pl.Callback):
    """每个 epoch 结束后记录 train_loss 和 val_loss，用于诊断过拟合。

    过拟合判定标准:
      - gap/val > 0.3: 明显过拟合 (train 远低于 val)
      - gap/val < 0.1: 正常
      - 0.1 ~ 0.3: 轻微过拟合
    """

    def __init__(self):
        super().__init__()
        self.history = []
        self._last_train_loss = None

    def on_train_epoch_end(self, trainer, pl_module):
        # 从 Lightning module 的输出中提取 train_loss
        # YeastLitModule 在 training_step 中会 log 'train_loss'
        try:
            metrics = trainer.callback_metrics
            train_loss = metrics.get('train_loss', None)
            if train_loss is not None:
                self._last_train_loss = float(train_loss)
        except Exception:
            pass

    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            metrics = trainer.callback_metrics
            val_loss = metrics.get('val_loss_lhs', None)
            if val_loss is not None and self._last_train_loss is not None:
                rec = {
                    'epoch': trainer.current_epoch,
                    'train_loss': self._last_train_loss,
                    'val_loss': float(val_loss),
                }
                self.history.append(rec)
                gap = rec['val_loss'] - rec['train_loss']
                ratio = gap / rec['val_loss'] if rec['val_loss'] > 0 else 0
                status = "⚠️ 过拟合" if ratio > 0.3 else ("✓ 正常" if ratio < 0.1 else "~ 轻微")
                print(f"  [Epoch {rec['epoch']:>3d}] train={rec['train_loss']:.6f} "
                      f"val={rec['val_loss']:.6f} gap={gap:.6f} ({ratio:.2%}) {status}")
        except Exception:
            pass


# ============================================================
# 1. 从 CSV 读取 MAE > 阈值的真实突变体
# ============================================================
def load_poor_mutants(csv_path, mae_threshold):
    """读取 per_mutant_metrics.csv，返回 MAE > 阈值的 [(name, mae), ...] 列表（按 MAE 降序）"""
    poor_mutants = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mae = float(row['MAE'])
            if mae > mae_threshold:
                poor_mutants.append((row['Mutant'], mae))
    poor_mutants.sort(key=lambda x: x[1], reverse=True)
    return poor_mutants


# ============================================================
# 2. 过滤训练集：只保留 poor_families 中的 LHS 子样本
# ============================================================
def filter_train_dataset(dm, poor_families):
    """
    过滤 dm.train_dataset_lhs，只保留母体属于 poor_families 的 LHS 子样本。
    LHS 样本命名格式: "{母体名}_LHS_{xxx}"，母体名 = split('_LHS_')[0]
    """
    X_train, P_train, Y_train = dm.train_dataset_lhs.tensors

    # 获取训练集每个样本的母体名
    train_names = [str(dm._mutant_names[idx]) for idx in dm._train_idx_lhs]
    train_families = []
    for name in train_names:
        family = name.split('_LHS_')[0] if '_LHS_' in name else name
        train_families.append(family)

    keep_mask = np.array([fam in poor_families for fam in train_families])
    keep_indices = np.where(keep_mask)[0]

    # 统计保留的母体
    kept_families = set(np.array(train_families)[keep_mask])
    missing_families = poor_families - kept_families

    print(f"\n{'='*60}")
    print(f"训练集过滤结果:")
    print(f"  原始训练集: {len(train_families)} 样本")
    print(f"  过滤后保留: {len(keep_indices)} 样本 (来自 {len(kept_families)} 个母体)")
    if missing_families:
        print(f"  ⚠️ {len(missing_families)} 个差母体在训练集中无 LHS 子样本:")
        for fam in sorted(missing_families):
            print(f"     - {fam}")
    print(f"{'='*60}\n")

    # 重建训练集 TensorDataset
    dm.train_dataset_lhs = TensorDataset(
        X_train[keep_indices].clone(),
        P_train[keep_indices].clone(),
        Y_train[keep_indices].clone(),
    )
    return dm, kept_families, missing_families


# ============================================================
# 3. 评估函数（复用 eval.py 逻辑）
# ============================================================
def plot_variable_grid(pred, target, var_indices, title, save_path, rows, cols, t_max=210):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    t = np.linspace(0, t_max, pred.shape[-1])
    for i, var_idx in enumerate(var_indices):
        ax = axes[i]
        ax.plot(t, target[var_idx], 'b-', label='Truth', linewidth=1.5)
        ax.plot(t, pred[var_idx], 'r--', label='Pred', linewidth=1.5)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Concentration')
        var_name = VARIABLE_NAMES[var_idx] if var_idx < len(VARIABLE_NAMES) else f"Var {var_idx}"
        ax.set_title(f'{var_idx}: {var_name}', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for i in range(len(var_indices), len(axes)):
        axes[i].set_visible(False)
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def evaluate_dataset(model, dm, dataset, indices, device, mean_y, std_y, seed, save_dir,
                     sub_dir, max_plots=None):
    """评估一个 dataset，生成图片和 CSV 行"""
    plot_dir = os.path.join(save_dir, sub_dir)
    os.makedirs(plot_dir, exist_ok=True)

    all_preds_denorm, all_targets_denorm = [], []
    csv_rows = []
    plot_count = 0

    with torch.no_grad():
        for local_idx, batch in enumerate(dataset):
            global_idx = indices[local_idx]
            m_name = dm._mutant_names[global_idx]
            p_label = dm.pattern_labels[global_idx]

            x, p, y = [b.unsqueeze(0).to(device) for b in batch]
            pred_denorm = (model.forward_ic_time(x, p) * std_y + mean_y).cpu()
            target_denorm = (y * std_y + mean_y).cpu()

            all_preds_denorm.append(pred_denorm)
            all_targets_denorm.append(target_denorm)

            metrics = calculate_metrics(pred_denorm, target_denorm)
            csv_rows.append({
                'Mutant': m_name,
                'Pattern': p_label,
                'MAE': metrics['MAE'],
                'MSE': metrics['MSE'],
                'Relative_L2': metrics['Relative L2'],
                'Correlation': metrics['Correlation']
            })

            if max_plots is None or plot_count < max_plots:
                m_str = f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4e} | Rel L2: {metrics['Relative L2']:.4f} | Corr: {metrics['Correlation']:.4f}"
                title_base = f"[{p_label}] {m_name} (Seed: {seed})\n{m_str}"
                p_np, t_np = pred_denorm[0].numpy(), target_denorm[0].numpy()
                save_plot_path = os.path.join(plot_dir, f"{m_name}.png")
                plot_variable_grid(p_np, t_np, range(39), f"{title_base}\nAll 39 Variables",
                                   save_plot_path, 5, 8, t_max=dm.t_max)
                plot_count += 1

    if all_preds_denorm:
        all_preds = torch.cat(all_preds_denorm, dim=0)
        all_targets = torch.cat(all_targets_denorm, dim=0)
        summary = {
            "all_39_vars": calculate_metrics(all_preds, all_targets),
            "core_9_vars": calculate_metrics(all_preds[:, CORE_9_INDICES, :],
                                             all_targets[:, CORE_9_INDICES, :]),
            "count": len(all_preds_denorm)
        }
    else:
        summary = {"all_39_vars": {}, "core_9_vars": {}, "count": 0}

    return summary, csv_rows, all_preds_denorm, all_targets_denorm


def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Mutant', 'Pattern', 'MAE', 'MSE', 'Relative_L2', 'Correlation']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"CSV saved to: {path}")


# ============================================================
# 4. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='过滤训练集实验: 只用差母体的 LHS 子样本训练')
    # 实验配置
    parser.add_argument('--csv_path', type=str, required=True,
                        help='per_mutant_metrics.csv 路径（用于提取差母体名单）')
    parser.add_argument('--mae_threshold', type=float, default=0.35,
                        help='MAE 阈值，大于此值的真实突变体母体将被保留')
    parser.add_argument('--best_params', type=str, required=True,
                        help='best_params.json 路径（训练超参来源）')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='数据集路径 (npz 文件)')
    parser.add_argument('--tag', type=str, default=None,
                        help='实验 tag，用于区分输出文件夹 (默认: filtered_poor_mae{threshold})')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--project', type=str, default='budding_yeast_v2_filtered')
    # 防过拟合措施参数
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='EarlyStopping patience: val_loss_lhs 连续 N epoch 不改善则停止 (0=禁用)')
    parser.add_argument('--weight_decay_override', type=float, default=1e-3,
                        help='覆盖 best_params 中的 weight_decay (默认 1e-3, 原值 1.72e-6 太小)')
    parser.add_argument('--max_epochs_override', type=int, default=100,
                        help='覆盖 best_params 中的 max_epochs (默认 100, 训练集小不需要 200)')
    parser.add_argument('--batch_size_override', type=int, default=32,
                        help='覆盖 best_params 中的 batch_size (默认 32, 小数据集用小 batch)')
    parser.add_argument('--lr_override', type=float, default=5e-4,
                        help='覆盖 best_params 中的 lr (默认 5e-4, 小数据集降低学习率)')
    parser.add_argument('--warmup_epochs_override', type=int, default=5,
                        help='覆盖 best_params 中的 warmup_epochs (默认 5, 小数据集不需要 10 轮预热)')
    args = parser.parse_args()

    # ---- 加载 best_params.json ----
    with open(args.best_params, 'r') as f:
        best = json.load(f)
    params = best['params']
    fixed = best['fixed_config']
    print(f"\n加载最优超参 (trial {best['trial_number']}):")
    print(f"  模型: {fixed['model']} | 损失: {fixed['loss_type']} | AL: {fixed['al_strategy']}")
    print(f"  原始: batch_size={params['batch_size']}, lr={params['lr']:.6f}, "
          f"wd={params['weight_decay']:.2e}, warmup={params['warmup_epochs']}, "
          f"epochs={params['max_epochs']}")
    print(f"  modes={params['modes']}, width={params['width']}, n_blocks={params['n_blocks']}")
    print(f"  原始 Real MAE: {best['real_mae']:.4f}, Corr: {best['real_corr']:.4f}")

    # ---- 防过拟合: 覆盖小数据集不适用的超参 ----
    original_bs = params['batch_size']
    original_lr = params['lr']
    original_wd = params['weight_decay']
    original_warmup = params['warmup_epochs']
    original_epochs = params['max_epochs']
    params['batch_size'] = args.batch_size_override
    params['lr'] = args.lr_override
    params['weight_decay'] = args.weight_decay_override
    params['warmup_epochs'] = args.warmup_epochs_override
    params['max_epochs'] = args.max_epochs_override
    print(f"\n🛡️ 小数据集超参覆盖:")
    print(f"  batch_size:    {original_bs} → {params['batch_size']}")
    print(f"  lr:            {original_lr:.6f} → {params['lr']:.6f}")
    print(f"  weight_decay:  {original_wd:.2e} → {params['weight_decay']:.2e}")
    print(f"  warmup_epochs: {original_warmup} → {params['warmup_epochs']}")
    print(f"  max_epochs:    {original_epochs} → {params['max_epochs']}")
    print(f"  early_stopping_patience: {args.early_stopping_patience}"
          f"{' (禁用)' if args.early_stopping_patience == 0 else ''}")

    # ---- 加载差母体名单 ----
    poor_mutants = load_poor_mutants(args.csv_path, args.mae_threshold)
    poor_families = set(m[0] for m in poor_mutants)
    print(f"\nMAE > {args.mae_threshold} 的真实突变体: {len(poor_families)} 个")
    print("Top-10 最差:")
    for name, mae in poor_mutants[:10]:
        print(f"  {name}: MAE={mae:.4f}")

    # ---- 实验 tag ----
    tag = args.tag or f"filtered_poor_mae{args.mae_threshold}"
    run_name = f"{fixed['model']}_{tag}_seed{args.seed}"
    print(f"\n实验名称: {run_name}")

    pl.seed_everything(args.seed, workers=True)

    # ---- 创建数据模块并 setup ----
    datamodule = YeastDataModule(
        dataset_path=args.dataset_name,
        batch_size_lhs=params['batch_size'],
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        use_adj_matrix=False,
        seed=args.seed,
    )
    datamodule.setup()

    # ---- 过滤训练集 ----
    datamodule, kept_families, missing_families = filter_train_dataset(datamodule, poor_families)

    # ---- 关键修复: 不传 datamodule 给 trainer.fit()，改为直接传 dataloader ----
    # 原因: trainer.fit(model, datamodule=dm) 会在 _call_setup_hook 中
    # 无条件调用 dm.setup()，覆盖掉过滤后的 train_dataset_lhs。
    # 改为直接传 dataloader 可以跳过 setup() 调用，保留过滤结果。
    # 同时需要手动将 datamodule 绑定到 trainer，使 checkpoint 等功能正常。

    # ---- 创建模型 ----
    model = YeastLitModule(
        model_name=fixed['model'],
        num_vars=39,
        param_dim=141,
        modes=params['modes'],
        width=params['width'],
        n_blocks=params['n_blocks'],
        loss_type=fixed['loss_type'],
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        max_epochs=params['max_epochs'],
        warmup_epochs=params['warmup_epochs'],
        rollout_type='one_time',
    )

    # ---- 训练 ----
    train_save_dir = os.path.join('./train_result', run_name)
    eval_dir = os.path.join('./eval_result', run_name)

    # 清理旧结果，防止残留文件混淆
    for old_dir in [train_save_dir, eval_dir]:
        if os.path.exists(old_dir):
            print(f"⚠️ 清理旧结果目录: {old_dir}")
            shutil.rmtree(old_dir, ignore_errors=True)
    os.makedirs(train_save_dir, exist_ok=True)

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "offline"
    logger = WandbLogger(project=args.project, name=run_name)
    logger.log_hyperparams({
        **vars(args), **params, **fixed,
        'tag': tag, 'experiment': 'filtered_poor_mutants',
        'n_poor_families': len(poor_families),
        'n_kept_families': len(kept_families),
    })

    checkpoint_callback = ModelCheckpoint(
        dirpath=train_save_dir,
        filename='best-{epoch:02d}-{val_loss_lhs:.4f}',
        monitor='val_loss_lhs',
        mode='min',
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ---- 防过拟合措施 1: EarlyStopping ----
    callbacks_list = [checkpoint_callback, lr_monitor]
    if args.early_stopping_patience > 0:
        early_stop_cb = EarlyStopping(
            monitor='val_loss_lhs',
            patience=args.early_stopping_patience,
            mode='min',
            verbose=True,
            check_finite=True,
            stopping_threshold=1e-6,
        )
        callbacks_list.append(early_stop_cb)
        print(f"  EarlyStopping 已启用 (patience={args.early_stopping_patience})")

    # ---- 防过拟合措施 4: train/val gap 监控回调 ----
    train_val_gap_cb = TrainValGapLogger()
    callbacks_list.append(train_val_gap_cb)

    trainer = pl.Trainer(
        max_epochs=params['max_epochs'],
        logger=logger,
        callbacks=callbacks_list,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=args.devices,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
    )

    print(f"\n开始训练 (max_epochs={params['max_epochs']})...")
    print(f"  训练集大小: {len(datamodule.train_dataset_lhs)} (过滤后)")

    # ---- 关键修复: 直接传 dataloader 而非 datamodule，防止 setup() 覆盖过滤 ----
    train_dl = datamodule.train_dataloader()
    val_dls = datamodule.val_dataloader()
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dls)

    # ---- 防过拟合措施 4: 打印 train/val gap 汇总 ----
    print(f"\n{'='*60}")
    print(f"训练过程 train/val gap 汇总 (过拟合诊断):")
    print(f"{'='*60}")
    history = train_val_gap_cb.history
    if history:
        print(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'Gap':<14} {'Gap/Val':<10} {'状态'}")
        print("-" * 75)
        for rec in history:
            gap = rec['val_loss'] - rec['train_loss']
            ratio = gap / rec['val_loss'] if rec['val_loss'] > 0 else 0
            # 过拟合判定: gap/val > 0.3 且 val 不再下降
            status = "⚠️ 过拟合" if ratio > 0.3 else ("✓ 正常" if ratio < 0.1 else "~ 轻微")
            print(f"  {rec['epoch']:<6} {rec['train_loss']:<14.6f} {rec['val_loss']:<14.6f} "
                  f"{gap:<14.6f} {ratio:<10.2%} {status}")
        print("-" * 75)
        final_rec = history[-1]
        best_val = min(r['val_loss'] for r in history)
        best_epoch = [r['epoch'] for r in history if r['val_loss'] == best_val][0]
        print(f"最佳 val_loss: {best_val:.6f} (epoch {best_epoch})")
        print(f"最终 train_loss: {final_rec['train_loss']:.6f}, val_loss: {final_rec['val_loss']:.6f}")
        final_gap = final_rec['val_loss'] - final_rec['train_loss']
        print(f"最终 gap: {final_gap:.6f} (gap/val={final_gap/final_rec['val_loss']:.2%})")
        if final_rec['val_loss'] > best_val * 1.1:
            print(f"⚠️ 警告: 最终 val_loss 比最佳值高 {(final_rec['val_loss']/best_val - 1)*100:.1f}%, 可能过拟合!")
        else:
            print(f"✓ val_loss 保持接近最佳值, 未明显过拟合")
    else:
        print("  (无训练历史记录)")
    print(f"{'='*60}\n")

    # ---- 加载最佳 checkpoint ----
    best_ckpt = checkpoint_callback.best_model_path
    print(f"\n最佳 checkpoint: {best_ckpt}")
    print(f"最佳 val_loss_lhs: {checkpoint_callback.best_model_score:.6f}")

    ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    mean_y = datamodule.mean_y.to(device)
    std_y = datamodule.std_y.to(device)

    # ---- 评估 ----
    os.makedirs(eval_dir, exist_ok=True)

    print("\n评估 Real Mutants...")
    real_summary, real_csv, real_preds, real_targets = evaluate_dataset(
        model, datamodule, datamodule.test_dataset_real, datamodule._test_idx_real,
        device, mean_y, std_y, args.seed, eval_dir,
        sub_dir='real_mutants', max_plots=None
    )

    print("评估 LHS 测试集...")
    lhs_summary, lhs_csv, lhs_preds, lhs_targets = evaluate_dataset(
        model, datamodule, datamodule.test_dataset_lhs, datamodule._test_idx_lhs,
        device, mean_y, std_y, args.seed, eval_dir,
        sub_dir='lhs_test', max_plots=100
    )

    # ---- 汇总 JSON ----
    all_preds_list = real_preds + lhs_preds
    all_targets_list = real_targets + lhs_targets
    if all_preds_list:
        all_preds = torch.cat(all_preds_list, dim=0)
        all_targets = torch.cat(all_targets_list, dim=0)
        all_test_summary = {
            "all_39_vars": calculate_metrics(all_preds, all_targets),
            "core_9_vars": calculate_metrics(all_preds[:, CORE_9_INDICES, :],
                                             all_targets[:, CORE_9_INDICES, :]),
            "count": len(all_preds_list)
        }
    else:
        all_test_summary = {"all_39_vars": {}, "core_9_vars": {}, "count": 0}

    global_metrics = {
        "experiment": "filtered_poor_mutants",
        "tag": tag,
        "mae_threshold": args.mae_threshold,
        "n_poor_families": len(poor_families),
        "n_kept_in_train": len(kept_families),
        "n_missing_in_train": len(missing_families),
        "train_size_after_filter": len(datamodule.train_dataset_lhs),
        "all_test": all_test_summary,
        "real_mutants": real_summary,
        "lhs_test": lhs_summary,
    }

    json_path = os.path.join(eval_dir, 'global_metrics_denorm.json')
    with open(json_path, 'w') as f:
        json.dump(global_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"评估结果汇总:")
    print(f"{'='*60}")
    print(f"  all_test:     n={all_test_summary['count']}, "
          f"MAE={all_test_summary['all_39_vars']['MAE']:.4f}, "
          f"Corr={all_test_summary['all_39_vars']['Correlation']:.4f}")
    print(f"  real_mutants: n={real_summary['count']}, "
          f"MAE={real_summary['all_39_vars']['MAE']:.4f}, "
          f"Corr={real_summary['all_39_vars']['Correlation']:.4f}")
    print(f"  lhs_test:     n={lhs_summary['count']}, "
          f"MAE={lhs_summary['all_39_vars']['MAE']:.4f}, "
          f"Corr={lhs_summary['all_39_vars']['Correlation']:.4f}")
    print(f"{'='*60}")

    # ---- 保存 CSV ----
    write_csv(real_csv, os.path.join(eval_dir, 'per_mutant_metrics.csv'))
    write_csv(lhs_csv, os.path.join(eval_dir, 'per_lhs_test_metrics.csv'))

    # ---- 对比：差母体训练前后的 MAE ----
    print(f"\n{'='*60}")
    print(f"差母体 (MAE > {args.mae_threshold}) 训练前后对比:")
    print(f"{'='*60}")
    print(f"{'Mutant':<45} {'Before':<10} {'After':<10} {'Delta':<10}")
    print("-" * 75)

    # 构建 before/after 映射
    before_map = {name: mae for name, mae in poor_mutants}
    after_map = {row['Mutant']: float(row['MAE']) for row in real_csv}

    improved, degraded = 0, 0
    for name, before_mae in poor_mutants:
        after_mae = after_map.get(name, float('nan'))
        delta = after_mae - before_mae if not np.isnan(after_mae) else float('nan')
        marker = ""
        if not np.isnan(delta):
            if delta < -0.01:
                marker = " ↓"
                improved += 1
            elif delta > 0.01:
                marker = " ↑"
                degraded += 1
        print(f"  {name:<43} {before_mae:<10.4f} {after_mae:<10.4f} {delta:<10.4f}{marker}")

    print("-" * 75)
    print(f"改善: {improved} | 恶化: {degraded} | 持平: {len(poor_mutants) - improved - degraded}")
    print(f"{'='*60}\n")

    # 保存对比结果
    comparison_path = os.path.join(eval_dir, 'poor_mutants_comparison.csv')
    with open(comparison_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Mutant', 'MAE_Before', 'MAE_After', 'Delta', 'Improved'])
        for name, before_mae in poor_mutants:
            after_mae = after_map.get(name, float('nan'))
            delta = after_mae - before_mae if not np.isnan(after_mae) else float('nan')
            improved_flag = "YES" if (not np.isnan(delta) and delta < -0.01) else "NO"
            writer.writerow([name, f"{before_mae:.6f}", f"{after_mae:.6f}",
                             f"{delta:.6f}", improved_flag])
    print(f"对比结果保存至: {comparison_path}")

    wandb.finish()


if __name__ == '__main__':
    main()
