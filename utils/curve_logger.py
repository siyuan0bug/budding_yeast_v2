"""
Training Curve Logger and Visualization Utility for Budding Yeast Experiments.

Features:
1. Automated tracking of training and evaluation metrics across epochs.
2. Single-consolidated publication-quality Loss Curve figure generation (PNG + PDF).
3. Explicit loss function subtitle display, without hardcoding 'MSE'.
4. Clear separation: Normalized space losses vs. Denormalized physical relative error.
5. Dual-saving mechanism (Local eval/ directory + W&B media logging).
6. Robust log/config file parser to generate curves from existing/historical runs.
"""

import os
import re
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.loggers import WandbLogger
    HAS_PL = True
except ImportError:
    HAS_PL = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class TrainingCurveLogger:
    """In-memory collector and visualizer for epoch-level training metrics."""

    def __init__(self, exp_name: str = "", loss_type: str = "", model_name: str = ""):
        self.exp_name = exp_name
        self.loss_type = loss_type
        self.model_name = model_name
        self.epochs: List[int] = []
        self.train_loss: List[float] = []
        self.val_synth: List[float] = []
        self.test_synth: List[float] = []
        self.test_real_norm: List[float] = []
        self.test_real_denorm_rel_l2: List[float] = []
        self.lr: List[float] = []

    def record(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_synth: Optional[float] = None,
        test_synth: Optional[float] = None,
        test_real_norm: Optional[float] = None,
        test_real_denorm_rel_l2: Optional[float] = None,
        lr: Optional[float] = None,
    ):
        self.epochs.append(epoch)
        self.train_loss.append(float(train_loss) if train_loss is not None else float('nan'))
        self.val_synth.append(float(val_synth) if val_synth is not None else float('nan'))
        self.test_synth.append(float(test_synth) if test_synth is not None else float('nan'))
        self.test_real_norm.append(float(test_real_norm) if test_real_norm is not None else float('nan'))
        self.test_real_denorm_rel_l2.append(
            float(test_real_denorm_rel_l2) if test_real_denorm_rel_l2 is not None else float('nan')
        )
        self.lr.append(float(lr) if lr is not None else float('nan'))

    def has_data(self) -> bool:
        return len(self.epochs) > 0

    def plot(
        self,
        save_path_png: Optional[str] = None,
        save_path_pdf: Optional[str] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        dpi: int = 300,
        ax=None,
    ) -> Optional[plt.Figure]:
        """Generate a single-panel consolidated loss curve figure."""
        if not self.has_data():
            return None

        epochs = np.array(self.epochs)
        train_loss = np.array(self.train_loss)
        val_synth = np.array(self.val_synth)
        test_synth = np.array(self.test_synth)
        test_real_norm = np.array(self.test_real_norm)
        test_real_denorm = np.array(self.test_real_denorm_rel_l2)

        embedded = ax is not None
        if embedded:
            fig,ax1=ax.figure,ax
        else:
            fig,ax1=plt.subplots(figsize=(11,7),dpi=dpi)

        # Style palette (Modern, clear contrast)
        c_train = '#1f77b4'       # Blue
        c_val = '#2ca02c'         # Green
        c_test_synth = '#bcbd22'  # Olive/Yellow-Green
        c_test_real = '#ff7f0e'   # Orange
        c_denorm_l2 = '#d62728'   # Deep Red

        # Determine marker spacing
        n_pts = len(epochs)
        mark_every = max(1, n_pts // 15)

        # 1. Left Axis: Normalized Space Losses (Log Scale)
        lines = []
        labels = []

        valid_train = ~np.isnan(train_loss)
        if np.any(valid_train):
            l1 = ax1.plot(epochs[valid_train], train_loss[valid_train], color=c_train, linewidth=2.2,
                          linestyle='-', label='Train Synth Loss')
            lines.extend(l1)
            labels.append(l1[0].get_label())

        valid_val = ~np.isnan(val_synth)
        if np.any(valid_val):
            l2 = ax1.plot(epochs[valid_val], val_synth[valid_val], color=c_val, linewidth=2.4,
                          linestyle='-', marker='o', markersize=5, markevery=mark_every,
                          label='Val Synth Loss')
            lines.extend(l2)
            labels.append(l2[0].get_label())

        valid_test_synth = ~np.isnan(test_synth)
        if np.any(valid_test_synth):
            l3 = ax1.plot(epochs[valid_test_synth], test_synth[valid_test_synth], color=c_test_synth,
                          linewidth=2.0, linestyle='--', marker='^', markersize=5, markevery=mark_every,
                          label='Test Synth Loss')
            lines.extend(l3)
            labels.append(l3[0].get_label())

        valid_test_real = ~np.isnan(test_real_norm)
        if np.any(valid_test_real):
            l4 = ax1.plot(epochs[valid_test_real], test_real_norm[valid_test_real], color=c_test_real,
                          linewidth=2.2, linestyle='-.', marker='v', markersize=5, markevery=mark_every,
                          label='Test Real Mutants Loss')
            lines.extend(l4)
            labels.append(l4[0].get_label())

        ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Normalized Loss (Log Scale)', fontsize=13, fontweight='bold', color='#222222')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='#222222', labelsize=11)
        ax1.tick_params(axis='x', labelsize=11)
        ax1.grid(True, which="both", ls="--", color='#dddddd', alpha=0.7)

        # 2. Right Axis: Real Mutants Physical Denormalized Relative L2 Loss
        valid_denorm = ~np.isnan(test_real_denorm)
        if np.any(valid_denorm):
            ax2 = ax1.twinx()
            l5 = ax2.plot(epochs[valid_denorm], test_real_denorm[valid_denorm], color=c_denorm_l2,
                          linewidth=2.6, linestyle='-', marker='s', markersize=6, markevery=mark_every,
                          label=r'Test Real Mutants $\mathbf{Rel\text{-}L_2}$ (Denorm Physical)')
            lines.extend(l5)
            labels.append(l5[0].get_label())

            ax2.set_ylabel(r'Physical Space Relative $L_2$ Loss ($\mu\mathrm{M}$ / a.u., Denorm)',
                           fontsize=13, fontweight='bold', color=c_denorm_l2)
            ax2.tick_params(axis='y', labelcolor=c_denorm_l2, labelsize=11)
            min_rel = np.nanmin(test_real_denorm[valid_denorm])
            max_rel = np.nanmax(test_real_denorm[valid_denorm])
            ax2.set_ylim(bottom=max(0.0, min_rel * 0.8), top=max_rel * 1.15 if max_rel > 0 else 1.0)
        else:
            ax2 = None

        # 3. Legend (Combined from both axes)
        ax1.legend(lines, labels, loc='upper right', framealpha=0.92, fontsize=10.5, fancybox=True, edgecolor='#cccccc')

        # 4. Main Title and Subtitle (Specifying Loss Function and Model/Experiment)
        main_title = title or "Training Convergence & Real Mutants Generalization"
        if not embedded:fig.suptitle(main_title, fontsize=15, fontweight='bold', y=0.98)

        sub_parts = []
        if self.exp_name:
            sub_parts.append(f"Exp: {self.exp_name}")
        if self.model_name:
            sub_parts.append(f"Model: {self.model_name}")
        if self.loss_type:
            sub_parts.append(f"Loss Function: {self.loss_type}")

        if subtitle:
            sub_str = subtitle
        elif sub_parts:
            sub_str = " | ".join(sub_parts)
        else:
            sub_str = ""

        if embedded:
            ax1.set_title(main_title+'\n'+sub_str,fontsize=11,color='#444444',pad=10)
        elif sub_str:
            ax1.set_title(sub_str, fontsize=11, color='#444444', pad=10)

        # 5. Key Metrics Summary Box (Bottom Left)
        summary_lines = []
        if np.any(valid_val):
            best_val_idx = np.nanargmin(val_synth)
            summary_lines.append(f"Best Val Loss: {val_synth[best_val_idx]:.6f} (Ep {epochs[best_val_idx]})")
        if np.any(valid_train):
            summary_lines.append(f"Final Train Loss: {train_loss[-1]:.6f}")
        if np.any(valid_test_synth):
            summary_lines.append(f"Final Synth Test Loss: {test_synth[-1]:.6f}")
        if np.any(valid_test_real):
            summary_lines.append(f"Final Real Mutants Loss: {test_real_norm[-1]:.6f}")
        if np.any(valid_denorm):
            summary_lines.append(f"Final Real Rel-L2 (Denorm): {test_real_denorm[-1]:.4f}")

        if summary_lines:
            summary_text = "\n".join(summary_lines)
            props = dict(boxstyle='round,pad=0.6', facecolor='#ffffff', alpha=0.96, edgecolor='#b0bec5')
            target_ax = ax2 if ax2 is not None else ax1
            target_ax.text(0.03, 0.05, summary_text, transform=ax1.transAxes, fontsize=10,
                           verticalalignment='bottom', bbox=props, family='monospace', zorder=100)

        # Set x limits neatly
        ax1.set_xlim(left=1, right=max(epochs[-1], 1))

        if not embedded:fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Save files
        if save_path_png:
            os.makedirs(os.path.dirname(os.path.abspath(save_path_png)), exist_ok=True)
            fig.savefig(save_path_png, dpi=dpi, bbox_inches='tight')
        if save_path_pdf:
            os.makedirs(os.path.dirname(os.path.abspath(save_path_pdf)), exist_ok=True)
            fig.savefig(save_path_pdf, bbox_inches='tight')

        return fig


def extract_and_plot_from_log(
    log_path: str,
    output_dir: Optional[str] = None,
    custom_title: Optional[str] = None,
    loss_type: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Optional[str]:
    """Parse a train.log file and generate a loss curve plot."""
    if not os.path.exists(log_path):
        return None

    run_dir = os.path.dirname(os.path.abspath(log_path))
    detected_loss_type = loss_type
    detected_model = model_name

    # Try reading config.json in the same directory if available
    config_path = os.path.join(run_dir, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as cf:
                cfg = json.load(cf)
                if not detected_loss_type and 'loss_type' in cfg:
                    detected_loss_type = cfg['loss_type']
                if not detected_model and 'model' in cfg:
                    detected_model = cfg['model']
        except Exception:
            pass

    # Read train.log to parse loss_type and metrics
    logger = TrainingCurveLogger(
        exp_name=os.path.basename(run_dir),
        loss_type=detected_loss_type or "",
        model_name=detected_model or ""
    )

    epoch_regex = re.compile(
        r'\[Epoch\s+(\d+)/\d+\]\s+train_loss=([^\s\|]+)\s+\|\s+(?:val_synth|val_lhs)=([^\s\|]+)\s+\|\s+(?:test_synth|test_lhs)=([^\s\|]+)\s+\|\s+(?:test_real_mse|test_real)=([^\s\|]+)(?:\s+\|\s+(?:test_real_denorm_rel_l2|test_real_denorm|test_loss_real_denorm)=([^\s\|]+))?'
    )
    lr_regex = re.compile(r'lr=([^\s\|]+)')
    header_loss_regex = re.compile(r'开始训练.*:\s*([^\s]+)')

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        epoch_offset = 0
        last_epoch = 0
        for line in f:
            if not logger.loss_type:
                hm = header_loss_regex.search(line)
                if hm:
                    task_str = hm.group(1)
                    # Extract loss pattern if present
                    for lt in ['hybrid_mse_rel', 'triple_dynamic', 'hybrid_bifurcate', 'hybrid_pino_colloc', 'hybrid_all', 'physics_informed', 'mse_only', 'rel_l2']:
                        if lt in task_str:
                            logger.loss_type = lt
                            break

            m = epoch_regex.search(line)
            if m:
                raw_epoch = int(m.group(1))
                if raw_epoch < last_epoch:
                    epoch_offset += last_epoch
                last_epoch = raw_epoch
                epoch = epoch_offset + raw_epoch

                def parse_val(s):
                    try:
                        v = float(s)
                        return v if not math.isnan(v) else None
                    except (ValueError, TypeError):
                        return None

                train_loss = parse_val(m.group(2))
                val_synth = parse_val(m.group(3))
                test_synth = parse_val(m.group(4))
                test_real_norm = parse_val(m.group(5))
                test_real_denorm = parse_val(m.group(6)) if m.group(6) is not None else None

                lr_m = lr_regex.search(line)
                lr = parse_val(lr_m.group(1)) if lr_m else None

                logger.record(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_synth=val_synth,
                    test_synth=test_synth,
                    test_real_norm=test_real_norm,
                    test_real_denorm_rel_l2=test_real_denorm,
                    lr=lr,
                )

    if not logger.has_data():
        return None

    target_dir = output_dir or run_dir
    png_path = os.path.join(target_dir, "loss_curves.png")
    pdf_path = os.path.join(target_dir, "loss_curves.pdf")

    fig = logger.plot(save_path_png=png_path, save_path_pdf=pdf_path, title=custom_title)
    if fig is not None:
        plt.close(fig)
        return png_path
    return None


if HAS_PL:
    class TrainingCurveCallback(Callback):
        """Lightning Callback to automatically record metrics and plot curves on training finish."""

        def __init__(self, save_dir: str, exp_name: str = "", loss_type: str = "", model_name: str = "", round_output=False):
            super().__init__()
            self.save_dir = save_dir
            self.round_output = round_output
            self.curve_logger = TrainingCurveLogger(exp_name=exp_name, loss_type=loss_type, model_name=model_name)

        def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            if trainer.sanity_checking:return
            epoch = trainer.current_epoch + 1
            m = trainer.callback_metrics

            def get_f(keys: List[str]):
                for k in keys:
                    if k in m:
                        try:
                            v = float(m[k])
                            if not math.isnan(v):
                                return v
                        except (ValueError, TypeError):
                            pass
                return None

            train_loss = get_f(['train_loss', 'train_loss_epoch'])
            val_synth = get_f(['val_loss_synth', 'val_synth'])
            test_synth = get_f(['test_loss_synth', 'test_synth'])
            test_real_norm = get_f(['test_loss_real', 'test_real', 'test_real_mse'])
            test_real_denorm = get_f(['test_real_denorm_rel_l2', 'test_loss_real_denorm', 'test_real_denorm'])

            lr = None
            if trainer.optimizers:
                lr = trainer.optimizers[0].param_groups[0]['lr']

            self.curve_logger.record(
                epoch=epoch,
                train_loss=train_loss,
                val_synth=val_synth,
                test_synth=test_synth,
                test_real_norm=test_real_norm,
                test_real_denorm_rel_l2=test_real_denorm,
                lr=lr,
            )

        def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            # Lightning finalizes train epoch reductions after validation. Read
            # them here so epoch 1 is present and later points are not one epoch late.
            epoch=trainer.current_epoch+1
            value=trainer.callback_metrics.get('train_loss',trainer.callback_metrics.get('train_loss_epoch'))
            if value is None:return
            if self.curve_logger.epochs and self.curve_logger.epochs[-1]==epoch:
                self.curve_logger.train_loss[-1]=float(value)
            else:
                self.curve_logger.record(epoch=epoch,train_loss=float(value),
                    lr=trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else None)

        def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            """Automatically render the loss curve figure on train finish."""
            if not self.curve_logger.has_data():
                return

            if self.round_output:
                import csv
                os.makedirs(self.save_dir,exist_ok=True)
                keys=('epochs','train_loss','val_synth','test_synth','test_real_norm','test_real_denorm_rel_l2','lr')
                with open(os.path.join(self.save_dir,'training_history.csv'),'w',newline='') as handle:
                    writer=csv.writer(handle);writer.writerow(keys)
                    writer.writerows(zip(*(getattr(self.curve_logger,key) for key in keys)))
                fig=self.curve_logger.plot(save_path_png=os.path.join(self.save_dir,'loss_curves.png'),
                                           save_path_pdf=os.path.join(self.save_dir,'loss_curves.pdf'))
                if fig is not None:plt.close(fig)
                return

            eval_dir = os.path.join(self.save_dir, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            png_eval = os.path.join(eval_dir, "loss_curves.png")
            pdf_eval = os.path.join(eval_dir, "loss_curves.pdf")
            png_root = os.path.join(self.save_dir, "loss_curves.png")

            fig = self.curve_logger.plot(save_path_png=png_eval, save_path_pdf=pdf_eval)
            if fig is not None:
                fig.savefig(png_root, dpi=300, bbox_inches='tight')

                if isinstance(trainer.logger, WandbLogger) and HAS_WANDB:
                    trainer.logger.experiment.log({'Loss_Curves/overall_convergence': wandb.Image(fig)})

                plt.close(fig)
                print(f"📈 [Loss Curves] 全生命周期 5 合 1 收敛曲线已自动保存至:\n   - {png_eval}\n   - {png_root}")
