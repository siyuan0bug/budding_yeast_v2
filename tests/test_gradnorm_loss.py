#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for GradNormHybridLoss (39-Channel Full GradNorm Hybrid Loss).
"""

import pytest
import torch
import torch.nn as nn
from budding_yeast_v2.utils.training.losses import LOSS_REGISTRY, GradNormHybridLoss, HybridMSERelLoss

def test_registry_registration():
    assert 'gradnorm_hybrid' in LOSS_REGISTRY
    loss_cls = LOSS_REGISTRY['gradnorm_hybrid']
    assert loss_cls is GradNormHybridLoss

def test_scale_identity_with_hybrid_mse_rel():
    """Verify that when task_weights are all 1.0, GradNormHybridLoss exactly matches HybridMSERelLoss."""
    torch.manual_seed(42)
    B, V, T = 8, 39, 100
    pred = torch.randn(B, V, T)
    target = torch.randn(B, V, T)

    criterion_base = HybridMSERelLoss(lambda_rel=0.2, eps=0.1)
    criterion_gn = GradNormHybridLoss(lambda_rel=0.2, eps=0.1)

    loss_base = criterion_base(pred, target)
    # In eval mode, no weight updates occur
    criterion_gn.eval()
    loss_gn = criterion_gn(pred, target)

    diff = torch.abs(loss_base - loss_gn).item()
    print(f"Scale difference: {diff:.8e}")
    assert diff < 1e-6, f"Scale mismatch between GradNormHybridLoss and HybridMSERelLoss: {diff}"

def test_gradnorm_weight_update_in_training():
    """Verify that task_weights update dynamically during training and remain frozen in eval mode."""
    torch.manual_seed(42)
    B, V, T = 8, 39, 100

    criterion = GradNormHybridLoss(num_vars=39, lambda_rel=0.2, alpha=1.0, eta=0.05, warmup_steps=5)
    criterion.train()

    initial_weights = criterion.task_weights.clone()
    assert torch.allclose(initial_weights, torch.ones(39)), "Initial weights must be all 1.0"

    # Simulate 10 training steps
    for step in range(10):
        pred = torch.randn(B, V, T, requires_grad=True)
        # Create non-uniform target so some channels are harder than others
        target = torch.randn(B, V, T)
        target[:, 2, :] *= 5.0  # CLB2 has higher variance / harder
        target[:, 4, :] *= 0.1  # SIC1 has lower variance / easier

        loss = criterion(pred, target)
        loss.backward()

    updated_weights = criterion.task_weights.clone()

    # Verify weights have moved from 1.0
    assert not torch.allclose(initial_weights, updated_weights), "Weights should dynamically adapt"

    # Verify mean(task_weights) == 1.0 constraint
    mean_w = updated_weights.mean().item()
    print(f"Mean task weight after 10 steps: {mean_w:.6f}")
    assert abs(mean_w - 1.0) < 1e-4, f"Mean task weight must be 1.0, got {mean_w}"

    # Verify bounds [0.1, 5.0]
    assert updated_weights.min().item() >= 0.1, f"Weight lower bound violated: {updated_weights.min().item()}"
    assert updated_weights.max().item() <= 5.0, f"Weight upper bound violated: {updated_weights.max().item()}"

    # Verify frozen in eval mode
    criterion.eval()
    pred_eval = torch.randn(B, V, T, requires_grad=True)
    target_eval = torch.randn(B, V, T)
    loss_eval = criterion(pred_eval, target_eval)

    assert torch.allclose(criterion.task_weights, updated_weights), "Weights must NOT change during eval"

def test_backward_gradient_flow():
    """Verify that gradients properly flow back to the model parameters through GradNormHybridLoss."""
    torch.manual_seed(42)
    B, V, T = 4, 39, 50

    model = nn.Sequential(
        nn.Conv1d(39, 64, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv1d(64, 39, kernel_size=1)
    )

    criterion = GradNormHybridLoss(num_vars=39)
    criterion.train()

    x = torch.randn(B, 39, T)
    target = torch.randn(B, 39, T)

    pred = model(x)
    loss = criterion(pred, target)
    loss.backward()

    # Verify all model parameters have non-zero, valid finite gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"
        assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradient!"
        assert param.grad.abs().sum().item() > 0.0, f"Parameter {name} has strictly zero gradient!"

    print("Backward gradient flow test PASSED!")

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
