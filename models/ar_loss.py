"""
Multi-step autoregressive loss with noise injection.

Direct adaptation of AL4PDE's pdebench_loss.py for 1D time-series data.
Supports autoregressive, teacher forcing, and scheduled teacher forcing.
"""

import torch
import torch.nn.functional as F
import math


def get_ar_loss(ic, traj, params, model, noise_std, training_type='autoregressive',
                current_epoch=0, num_epoch=500, block_grad=False):
    """
    Compute multi-step rollout loss with noise injection.
    
    This is a direct adaptation of AL4PDE's get_loss function.
    
    Args:
        ic:     (B, C)          — initial state (first frame of sub-trajectory)
        traj:   (B, C, traj_len) — target sub-trajectory (C=39, traj_len=K+1)
        params: (B, P)          — PDE parameters
        model:  ARModelWrapper  — the model with forward(state, params) -> next_state
        noise_std: float or (C,) — noise standard deviation
        training_type: 'autoregressive' / 'full_teacher_forcing' / 'teacher_forcing_schedule'
        current_epoch: current epoch (for scheduled teacher forcing)
        num_epoch: total epochs (for scheduled teacher forcing)
        block_grad: if True, detach between AR steps
        
    Returns:
        pred:  (B, C, traj_len) — predicted trajectory including IC
        loss:  scalar           — accumulated loss over all steps
    """
    B, C, traj_len = traj.shape
    
    # pred collects the full predicted trajectory
    pred = traj[:, :, :1]  # start with IC, shape (B, C, 1)
    
    # xx is the current input state, shape (B, C)
    xx = ic.clone()
    
    loss = 0.0
    
    for t in range(1, traj_len):
        y = traj[:, :, t]  # target at step t, shape (B, C)
        
        # Add noise to input (AL4PDE: every step gets noise)
        xx = xx + torch.normal(torch.zeros_like(xx), noise_std)
        
        if block_grad:
            xx = xx.detach()
        
        # Model prediction: (B, C) -> (B, C)
        model_pred = model.forward_single_step(xx, params)
        
        # Accumulate loss (compute MSE on normalized space if normalizer is available)
        if getattr(model, 'task_norm', None) is not None:
            y_norm = model.task_norm.norm_state(y)
            pred_norm = model.task_norm.norm_state(model_pred)
            loss = loss + F.mse_loss(pred_norm, y_norm)
        else:
            loss = loss + F.mse_loss(model_pred, y)
        
        # Build prediction trajectory
        pred = torch.cat((pred, model_pred.unsqueeze(-1)), dim=-1)  # (B, C, t+1)
        
        # Update xx for next step
        if training_type == 'autoregressive':
            xx = model_pred
        elif training_type == 'full_teacher_forcing':
            xx = y
        elif training_type == 'teacher_forcing_schedule':
            # Sigmoid schedule: gradually shift from teacher forcing to AR
            ep_norm = current_epoch / num_epoch
            t_norm = 0.5 * (1.0 + math.tanh((ep_norm - 0.5) / 0.2))
            t_ar = int(traj_len * t_norm)
            t_ar = min(traj_len, max(1, t_ar))
            if t < t_ar:
                xx = model_pred  # autoregressive
            else:
                xx = y  # teacher forcing
        else:
            raise ValueError(f"Unknown training_type: {training_type}")
    
    return pred, loss
