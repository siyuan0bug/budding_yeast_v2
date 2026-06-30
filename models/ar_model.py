"""
AR Model Wrapper for autoregressive training and inference.

Wraps any existing model (HyperFNO, etc.) to provide:
1. Single-step forward: state(B,C) + params(B,P) -> next_state(B,C)
2. Multi-step rollout: ic(B,C) + params(B,P) -> trajectory(B,C,T)
3. Training loop with noise injection (AL4PDE style)

References AL4PDE's wrapper.py and arena_wrapper.py.
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from budding_yeast_v2.models.components.hyper_fno import HyperFNO
from budding_yeast_v2.models.ar_loss import get_ar_loss
from budding_yeast_v2.utils.normalization import TaskNormalizer
from budding_yeast_v2.utils.metrics import calculate_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VARIABLE_NAMES = [
    "MASS", "CLN2", "CLB2", "CLB5", "SIC1", "CDC6", "C2", "C5", "F2", "F5",
    "SIC1P", "C2P", "C5P", "CDC6P", "F2P", "F5P", "SWI5T", "SWI5", "IEP", "CDC20T",
    "CDC20", "CDH1T", "CDH1", "CDC14T", "CDC14", "NET1T", "NET1", "RENT", "TEM1", "CDC15",
    "PPX", "PDS1", "ESP1", "ORI", "BUD", "SPN", "Vi20", "lte1", "BUB2"
]


# ======== Model Registry (same as yeast_lit_module.py) ========
def build_model(cfg):
    """Build a model from config dict."""
    name = cfg.get('name', 'hyper_fno')
    if name == 'hyper_fno':
        return HyperFNO(
            num_vars=cfg.get('num_vars', 39),
            param_dim=cfg.get('param_dim', 141),
            modes=cfg.get('modes', 32),
            width=cfg.get('width', 32),
            n_layers=cfg.get('n_layers', 4),
            d_lambda=cfg.get('d_lambda', 32),
        )
    elif name == 'unet1d':
        from budding_yeast_v2.models.components.unet1d_adapter import Unet1DAdapter
        return Unet1DAdapter(
            num_vars=cfg.get('num_vars', 39),
            param_dim=cfg.get('param_dim', 141),
            hidden_channels=cfg.get('hidden_channels', 64),
            ch_mults=cfg.get('ch_mults', (1, 2, 2, 4)),
            n_blocks=cfg.get('n_blocks', 2),
        )
    elif name == 'fno1d':
        from budding_yeast_v2.models.components.fno1d_adapter import FNO1DAdapter
        return FNO1DAdapter(
            num_vars=cfg.get('num_vars', 39),
            param_dim=cfg.get('param_dim', 141),
            modes=cfg.get('modes', 16),
            width=cfg.get('width', 64),
            t_steps=cfg.get('t_steps', 500),
        )
    else:
        raise ValueError(f"Unknown model: {name}. "
                         f"AR mode currently supports: hyper_fno, unet1d, fno1d")


class ARModelWrapper(nn.Module):
    """
    Wraps a backbone model (HyperFNO) for autoregressive single-step prediction.
    
    The backbone expects input (B, V, 2, T) and outputs (B, V, T).
    This wrapper:
    1. Takes single-step state (B, C) + params (B, P) 
    2. Constructs full-size input tensor (B, C, 2, T_model)
    3. Runs backbone
    4. Extracts next-step prediction from output[:, :, 1]
    5. Optionally applies residual prediction (predict_delta)
    """

    def __init__(self, model_cfg, training_cfg, t_steps=500):
        super().__init__()
        self.model_cfg = model_cfg
        self.model = build_model(model_cfg)
        self.t_steps = t_steps
        self.num_vars = model_cfg.get('num_vars', 39)
        self.param_dim = model_cfg.get('param_dim', 141)
        
        # Residual prediction settings (from AL4PDE)
        self.predict_delta = model_cfg.get('predict_delta', True)
        self.difference_weight = model_cfg.get('difference_weight', 0.3)
        
        # Training config
        self.training_type = training_cfg.get('training_type', 'autoregressive')
        self.noise = training_cfg.get('noise', 0.1)
        self.noise_mode = training_cfg.get('noise_mode', 'channel_std_rel')
        self.batch_size = training_cfg.get('batch_size', 64)
        self.lr = training_cfg.get('lr', 1e-3)
        self.weight_decay = training_cfg.get('weight_decay', 1e-4)
        self.max_grad_norm = training_cfg.get('max_grad_norm', 50.0)
        self.val_period = training_cfg.get('val_period', 20)
        self.vis_period = training_cfg.get('vis_period', 100)
        self.warmup_epochs = training_cfg.get('warmup_epochs', 10)
        self.block_grad = training_cfg.get('block_grad', False)
        self.num_epochs = training_cfg.get('num_epochs', 500)  # 用于 cosine schedule 上限
        
        # State
        self.task_norm = None
        self.optimizer = None
        self.scheduler = None
        self._train_loader = None
        self._train_loader_full = None
        self._val_loader = None
        
        # Pre-compute t_grid for model input construction
        self.register_buffer('t_grid', torch.linspace(0, 1, t_steps))

    def forward_single_step(self, state, params):
        """
        Single-step prediction: state(B, C) + params(B, P) -> next_state(B, C)
        
        Constructs the (B, C, 2, T) input that HyperFNO expects:
        - Channel 0 (IC): current state broadcast to all T positions
        - Channel 1 (t_grid): normalized time grid [0, 1]
        """
        B = state.shape[0]
        C = self.num_vars
        T = self.t_steps
        
        # Normalize state and params
        if self.task_norm is not None:
            state_norm = self.task_norm.norm_state(state)
            params_norm = self.task_norm.norm_param(params)
        else:
            state_norm = state
            params_norm = params
        
        # Construct (B, C, 2, T) input
        # IC channel: state broadcast across time
        ic_channel = state_norm.unsqueeze(-1).expand(B, C, T)  # (B, C, T)
        # t_grid channel: same grid for all variables
        t_channel = self.t_grid.unsqueeze(0).unsqueeze(0).expand(B, C, T)  # (B, C, T)
        
        model_input = torch.stack([ic_channel, t_channel], dim=2)  # (B, C, 2, T)
        
        # Forward through backbone
        output = self.model(model_input, params_norm)  # (B, C, T)

        # 防止 HyperFNO 的 FFT 谱卷积在训练早期产生 NaN/Inf 污染 loss 和梯度
        output = torch.nan_to_num(output, nan=0.0, posinf=1e3, neginf=-1e3)

        # Extract next step (index 1)
        next_pred = output[:, :, 1]  # (B, C)
        
        # Residual prediction
        if self.predict_delta:
            next_pred = state_norm + self.difference_weight * next_pred
        
        # Denormalize
        if self.task_norm is not None:
            next_pred = self.task_norm.denorm_state(next_pred)
        
        return next_pred

    def roll_out(self, ic, params, num_steps):
        """
        Autoregressive rollout from initial condition.

        Args:
            ic:     (B, C) initial state
            params: (B, P) parameters
            num_steps: number of steps to predict

        Returns:
            pred: (B, C, num_steps+1) trajectory including IC
        """
        pred = [ic.unsqueeze(-1)]  # [(B, C, 1)]
        xx = ic

        for t in range(num_steps):
            with torch.no_grad():
                xx = self.forward_single_step(xx, params)
            # 防止 499 步 rollout 累积误差导致数值溢出 (未收敛模型早期常见)
            # nan_to_num 处理 inf/nan; clamp 防止大有限值在 MSE 计算时平方溢出为 inf
            xx = torch.nan_to_num(xx, nan=0.0, posinf=10.0, neginf=-10.0)
            xx = torch.clamp(xx, -1e3, 1e3)
            pred.append(xx.unsqueeze(-1))

        return torch.cat(pred, dim=-1)  # (B, C, num_steps+1)

    def init_training(self, al_iter, train_loader, train_loader_full, val_loader=None):
        """
        Initialize model, optimizer, scheduler for a new AL iteration.
        Model weights are completely rebuilt from scratch each AL iter (AL4PDE style).
        """
        # Fully re-initialize model weights
        self.model = build_model(self.model_cfg)
        
        self.to(device)
        self.model.train()
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[AR Model] Total trainable parameters: {total_params:,}")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        # Cosine annealing with warmup
        warmup = self.warmup_epochs
        eta_min = 1e-6
        
        def lr_lambda(epoch):
            if epoch < warmup:
                return float(epoch) / float(max(1, warmup))
            progress = float(epoch - warmup) / float(max(1, self.num_epochs - warmup))
            return (eta_min / self.lr) + 0.5 * (1.0 - eta_min / self.lr) * (1.0 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        
        self._train_loader = train_loader
        self._train_loader_full = train_loader_full
        self._val_loader = val_loader
        
        # Compute normalization stats
        if al_iter == 0:
            self.task_norm = TaskNormalizer()
            self.task_norm.update(train_loader)
            self.task_norm.to(device)

    def train_single_epoch(self, current_epoch, total_epoch, num_epoch, last_metrics=None):
        """Train for one epoch. Follows AL4PDE's wrapper.py pattern."""
        self.model.train()
        
        # Compute noise std
        if self.noise_mode == 'channel_std_rel':
            noise_std = self.noise * self.task_norm.std_channels  # (C,)
        else:
            noise_std = self.noise
        
        train_loss_sum = 0.0
        n = 0
        
        if last_metrics is None:
            last_metrics = {}
            
        desc = f"Epoch {current_epoch+1}/{num_epoch}"
        pbar = tqdm(self._train_loader, desc=desc, leave=False, dynamic_ncols=True, mininterval=0.5)
        for batch_idx, (ic, traj, params, t_idx) in enumerate(pbar):
            ic = ic.to(device)
            traj = traj.to(device)
            params = params.to(device)
            B = ic.shape[0]
            
            # Multi-step AR loss
            pred, loss = get_ar_loss(
                ic, traj, params, self,
                noise_std=noise_std,
                training_type=self.training_type,
                current_epoch=current_epoch,
                num_epoch=num_epoch,
                block_grad=self.block_grad,
            )
            
            self.optimizer.zero_grad()
            loss.backward()

            # NaN/Inf 梯度保护: clip_grad_norm_ 无法处理 NaN, 会导致权重被永久污染
            # (HyperFNO 的 FFT 谱卷积在训练早期易产生数值溢出 → NaN 梯度 → 权重 NaN → 预测全 0)
            if torch.isnan(loss) or torch.isinf(loss):
                pbar.set_postfix_str(f"train_loss: {train_loss_sum/max(n,1):.3f} | SKIP(NaN loss)")
                continue
            for p in self.model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1e3, neginf=-1e3)

            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            
            train_loss_sum += loss.detach().item() * B
            n += B
            
            # Format postfix string matching user's requested style
            postfix_str = f"train_loss: {train_loss_sum/n:.3f}"
            for k, v in last_metrics.items():
                postfix_str += f" | {k}: {v:.3f}"
            pbar.set_postfix_str(postfix_str)
        
        self.scheduler.step()
        
        train_avg_loss = train_loss_sum / max(n, 1)
        
        if current_epoch % self.val_period == 0:
            lr_current = self.optimizer.param_groups[0]['lr']
            wandb.log({
                'train_loss': train_avg_loss,
                'lr': lr_current,
                'total_epoch': total_epoch,
            })
        
        return train_avg_loss

    @torch.no_grad()
    def validate(self, step, val_loader=None, prefix='val/', t_max=210.0):
        """
        Validate on full trajectory rollout.
        Computes MAE, MSE, Rel L2, Correlation — same metrics as eval.py.
        """
        self.model.eval()
        loader = val_loader or self._val_loader
        if loader is None:
            return
        
        all_preds = []
        all_targets = []
        
        for batch in loader:
            if len(batch) >= 6:
                ic, traj, params, t_idx, names, labels = batch
            else:
                ic, traj, params, t_idx = batch
                
            ic = ic.to(device)
            traj = traj.to(device)  # (B, C, T_full)
            params = params.to(device)
            
            num_steps = traj.shape[-1] - 1
            pred = self.roll_out(ic, params, num_steps)  # (B, C, T_full)
            
            all_preds.append(pred.cpu())
            all_targets.append(traj.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 未收敛模型 499 步 rollout 可能产生 nan/inf/大有限值, 用 nan_to_num + clamp 保护避免 MSE 平方溢出为 inf
        all_preds = torch.nan_to_num(all_preds, nan=0.0, posinf=10.0, neginf=-10.0)
        all_preds = torch.clamp(all_preds, -1e3, 1e3)

        # Compute MSE
        mse = F.mse_loss(all_preds, all_targets).item()
        
        # Compute full metrics using existing calculate_metrics
        metrics = calculate_metrics(all_preds, all_targets)
        
        log_dict = {
            f'{prefix}mse': mse,
            f'{prefix}mae': metrics['MAE'],
            f'{prefix}rel_l2': metrics['Relative L2'],
            f'{prefix}correlation': metrics['Correlation'],
            'total_epoch': step,
        }
        wandb.log(log_dict)
        
        self.model.train()
        return mse

    @torch.no_grad()
    def visualize(self, step, val_loader=None, save_dir=None, t_max=210.0):
        """
        Generate trajectory comparison plots (same format as SimToRealVisualizerCallback).
        Logs to wandb as images.
        """
        self.model.eval()
        loader = val_loader or self._val_loader
        if loader is None:
            return
        
        # Get first batch
        batch = next(iter(loader))
        if len(batch) >= 6:
            ic_b, traj_b, params_b, t_idx_b, names_b, labels_b = batch
        else:
            ic_b, traj_b, params_b, t_idx_b = batch
            names_b = [f"Sample_{i}" for i in range(ic_b.shape[0])]
            labels_b = ["Pattern_Unknown" for _ in range(ic_b.shape[0])]
            
        ic_b = ic_b.to(device)
        traj_b = traj_b.to(device)
        params_b = params_b.to(device)
        
        # Roll out all at once
        num_steps = traj_b.shape[-1] - 1
        pred_b = self.roll_out(ic_b, params_b, num_steps)
        
        # Select samples to plot: max 2 per pattern
        pattern_counts = {}
        selected_indices = []
        for i, label in enumerate(labels_b):
            if pattern_counts.get(label, 0) < 2:
                selected_indices.append(i)
                pattern_counts[label] = pattern_counts.get(label, 0) + 1
                
        log_dict = {'total_epoch': step}
        
        for i in selected_indices:
            pred_np = pred_b[i].cpu().numpy()
            target_np = traj_b[i].cpu().numpy()
            name = names_b[i]
            label = labels_b[i]
            
            T_len = pred_np.shape[1]
            t = np.linspace(0, t_max, T_len)
            
            fig, axes = plt.subplots(5, 8, figsize=(28, 16))
            axes_flat = axes.flatten()
            
            for var_idx in range(39):
                ax = axes_flat[var_idx]
                ax.plot(t, target_np[var_idx], 'b-', label='Ground Truth', linewidth=2)
                ax.plot(t, pred_np[var_idx], 'r--', label='Prediction', linewidth=2)
                var_name = VARIABLE_NAMES[var_idx] if var_idx < len(VARIABLE_NAMES) else f"Var {var_idx}"
                ax.set_title(f'{var_idx}: {var_name}', fontweight='bold', fontsize=10)
                ax.set_xlabel('Time (min)')
                if var_idx == 0:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            for j in range(39, len(axes_flat)):
                axes_flat[j].set_visible(False)
            
            metrics = calculate_metrics(pred_b[i:i+1].cpu(), traj_b[i:i+1].cpu())
            m_str = (f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4e} | "
                     f"Rel L2: {metrics['Relative L2']:.4f} | Corr: {metrics['Correlation']:.4f}")
            
            title_str = f"[{label}] {name} (Epoch {step})\n{m_str}"
            plt.suptitle(title_str, fontsize=22, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            group_name = "real_mutants" if not str(name).startswith("LHS") else "LHS_samples"
            log_dict[f'{group_name}/{name}'] = wandb.Image(fig)
            
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f'{name}_epoch{step}.png'), dpi=150)
            
            plt.close(fig)
            
        wandb.log(log_dict)

    def train_n_epochs(self, al_iter, num_epoch, step_offset, vis=True):
        """
        Train for num_epoch epochs within one AL iteration.
        Follows AL4PDE's Model.train_n_epoch pattern.
        """
        total_time = 0.0
        last_metrics = {}
        
        # Initial evaluation before training this AL iteration (optional, but good for progress bar)
        if self._val_loader is not None:
            last_metrics['test_loss_real'] = self.validate(step_offset, prefix='test_real/')
        if self._train_loader_full is not None:
            last_metrics['train_full'] = self.validate(step_offset, val_loader=self._train_loader_full, prefix='train/')
            
        for i in range(num_epoch):
            t = time.time()
            train_avg_loss = self.train_single_epoch(i, step_offset + i, num_epoch, last_metrics=last_metrics)
            total_time += time.time() - t
            
            if i % self.val_period == 0:
                if self._val_loader is not None:
                    last_metrics['test_loss_real'] = self.validate(step_offset + i, prefix='test_real/')
                if self._train_loader_full is not None:
                    last_metrics['train_full'] = self.validate(step_offset + i, val_loader=self._train_loader_full, prefix='train/')
            
            if vis and ((i > 0 and i % self.vis_period == 0) or i == num_epoch - 1):
                self.visualize(step_offset + i)
        
        return total_time
