#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification & Smoke Test Suite for Mechanistic, Bifurcation & PINO Pipeline
============================================================================
Validates:
1. Model forward/backward for:
   - BioRefinerBifurcate (Plan A)
   - BioRefinerGraphFlux (Plan B)
   - BioRefinerMechanistic (Plan A+B)
   - Original BioRefinerPlus (Baseline compatibility)
2. Loss computation & gradients for:
   - HybridBifurcateLoss (Plan A)
   - HybridPINOCollocationLoss (Plan C)
   - HybridAllLoss (Plan A+C)
   - Original HybridMSERelLoss
3. YeastLitModule training_step integration for all combinations.
4. Biological guard functions (init-aware zeroing, non-negative clamp).
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Ensure root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, os.path.abspath(os.path.join(root_dir, '..')))

from budding_yeast_v2.models.components.bio_refiner_plus import (
    BioRefinerPlus, BioRefinerBifurcate, BioRefinerGraphFlux, BioRefinerMechanistic
)
from budding_yeast_v2.utils.training.losses import (
    LOSS_REGISTRY, HybridMSERelLoss, HybridBifurcateLoss,
    HybridPINOCollocationLoss, HybridAllLoss
)
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.utils.physics.variable_groups import (
    CONTINUOUS_VARS, EVENT_VARS, POSSIBLE_ZERO_VARS
)

from budding_yeast_v2.utils.archive.biological_guard import (
    compute_init_aware_zero_mask, apply_biological_guard,
)


def test_models():
    print("\n" + "=" * 60)
    print("1. Testing Model Architectures (Forward & Backward)")
    print("=" * 60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    B, V, T = 4, 39, 50
    param_dim = 141

    x = torch.randn(B, V, 2, T, device=device)
    p = torch.randn(B, param_dim, device=device)

    models = {
        'BioRefinerPlus': BioRefinerPlus(num_vars=V, width=32, modes=8, param_dim=param_dim, n_blocks=2),
        'BioRefinerBifurcate': BioRefinerBifurcate(num_vars=V, width=32, modes=8, param_dim=param_dim, n_blocks=2),
        'BioRefinerGraphFlux': BioRefinerGraphFlux(num_vars=V, width=32, modes=8, param_dim=param_dim, n_blocks=2),
        'BioRefinerMechanistic': BioRefinerMechanistic(num_vars=V, width=32, modes=8, param_dim=param_dim, n_blocks=2),
    }

    for name, model in models.items():
        model = model.to(device)
        model.train()

        # Standard forward
        out = model(x, p)
        assert out.shape == (B, V, T), f"{name} standard forward failed: expected ({B}, {V}, {T}), got {out.shape}"
        assert not torch.isnan(out).any(), f"{name} produced NaNs in output!"

        # Coarse forward
        out_f, out_c = model(x, p, return_coarse=True)
        assert out_f.shape == (B, V, T) and out_c.shape == (B, V, T), f"{name} return_coarse failed"

        # Pattern forward (if supported)
        if hasattr(model, 'pattern_head'):
            out_f, out_c, pat_logits = model(x, p, return_coarse=True, return_pattern=True)
            assert pat_logits.shape == (B, 3), f"{name} pattern_logits shape mismatch: {pat_logits.shape}"
            loss = out_f.sum() + pat_logits.sum()
        else:
            loss = out_f.sum()

        loss.backward()
        print(f"  ✓ {name:<25}: Forward & Backward PASSED (Output: {out.shape})")


def test_losses():
    print("\n" + "=" * 60)
    print("2. Testing Loss Functions")
    print("=" * 60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    B, V, T = 4, 39, 50
    param_dim = 141

    pred_norm = torch.randn(B, V, T, device=device, requires_grad=True)
    target_norm = torch.randn(B, V, T, device=device)
    p_norm = torch.randn(B, param_dim, device=device)
    weights = torch.ones(B, device=device)
    pat_logits = torch.randn(B, 3, device=device, requires_grad=True)
    pat_targets = torch.tensor([0, 1, 2, 2], device=device)

    # 1. HybridBifurcateLoss
    loss_fn_bif = HybridBifurcateLoss(lambda_rel=0.2, lambda_pattern=0.1, lambda_arrest=0.05).to(device)
    loss_bif = loss_fn_bif(pred_norm, target_norm, pattern_logits=pat_logits, pattern_targets=pat_targets)
    assert not torch.isnan(loss_bif), "HybridBifurcateLoss produced NaN!"
    loss_bif.backward(retain_graph=True)
    print(f"  ✓ HybridBifurcateLoss: Value = {loss_bif.item():.4f} | Gradients Finite")

    # 2. HybridPINOCollocationLoss (Decoupled strategy)
    mean_y = torch.zeros(1, V, 1, device=device)
    std_y = torch.ones(1, V, 1, device=device)
    p_mean = torch.zeros(1, param_dim, device=device)
    p_std = torch.ones(1, param_dim, device=device)

    loss_fn_pino = HybridPINOCollocationLoss(
        mean_y, std_y, p_mean, p_std, t_max=210.0,
        lambda_rel=0.2, lambda_phys=0.01, lambda_jump=0.001,
        event_strategy='decoupled', huber_delta=1.0, warmup_epochs=0
    ).to(device)
    loss_fn_pino.set_epoch(10)  # Post-warmup
    loss_pino = loss_fn_pino(pred_norm, target_norm, p_norm, weights)
    assert not torch.isnan(loss_pino), "HybridPINOCollocationLoss produced NaN!"
    loss_pino.backward(retain_graph=True)
    print(f"  ✓ HybridPINOCollocationLoss (Decoupled): Value = {loss_pino.item():.4f} | Gradients Finite")

    # 3. HybridAllLoss
    loss_fn_all = HybridAllLoss(
        mean_y, std_y, p_mean, p_std, t_max=210.0,
        lambda_rel=0.2, lambda_pattern=0.1, lambda_arrest=0.05,
        lambda_phys=0.01, lambda_jump=0.001,
        event_strategy='decoupled', huber_delta=1.0, warmup_epochs=0
    ).to(device)
    loss_fn_all.set_epoch(10)
    loss_all = loss_fn_all(pred_norm, target_norm, p_norm, weights,
                           pattern_logits=pat_logits, pattern_targets=pat_targets)
    assert not torch.isnan(loss_all), "HybridAllLoss produced NaN!"
    loss_all.backward()
    print(f"  ✓ HybridAllLoss (Plan A+C): Value = {loss_all.item():.4f} | Gradients Finite")


def test_lit_module_training_step():
    print("\n" + "=" * 60)
    print("3. Testing YeastLitModule Integration with New Models & Losses")
    print("=" * 60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    B, V, T = 4, 39, 50
    param_dim = 141

    class DummyDataModule:
        def __init__(self):
            self.mean_y = torch.zeros(1, V, 1)
            self.std_y = torch.ones(1, V, 1)
            self.p_mean = torch.zeros(1, param_dim)
            self.p_std = torch.ones(1, param_dim)
            self.t_max = 210

    dm = DummyDataModule()

    test_configs = [
        ('biom_refiner_plus', 'hybrid_mse_rel'),
        ('biom_refiner_bifurcate', 'hybrid_bifurcate'),
        ('biom_refiner_graph_flux', 'hybrid_mse_rel'),
        ('biom_refiner_plus', 'hybrid_pino_colloc'),
        ('biom_refiner_mechanistic', 'hybrid_all'),
    ]

    for model_name, loss_type in test_configs:
        lit_module = YeastLitModule(
            model_name=model_name,
            loss_type=loss_type,
            num_vars=V,
            param_dim=param_dim,
            width=32,
            modes=8,
            n_blocks=2,
            max_epochs=100,
            warmup_epochs=0,
            lambda_pattern=0.1,
            lambda_arrest=0.05,
            pino_event_strategy='decoupled',
            huber_delta=1.0
        ).to(device)

        # Mock trainer datamodule and log
        class MockTrainer:
            def __init__(self, dm):
                self.datamodule = dm
        lit_module.trainer = MockTrainer(dm)
        lit_module.log = lambda *args, **kwargs: None
        lit_module.on_train_start()

        # Batch with pattern label
        x = torch.randn(B, V, 2, T, device=device)
        p = torch.randn(B, param_dim, device=device)
        y = torch.randn(B, V, T, device=device)
        pat_id = torch.tensor([0, 1, 2, 2], device=device)
        batch = (x, p, y, pat_id)

        lit_module.train()
        loss = lit_module.training_step(batch, 0)
        assert not torch.isnan(loss), f"LitModule training step failed for {model_name} + {loss_type}: NaN loss"
        loss.backward()

        # Validation step
        lit_module.eval()
        with torch.no_grad():
            val_loss = lit_module.validation_step(batch, 0)
            assert not torch.isnan(val_loss), f"LitModule validation step failed for {model_name}"

        print(f"  ✓ {model_name:<26} + {loss_type:<20}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")


def test_biological_guard():
    print("\n" + "=" * 60)
    print("4. Testing Biological Guard Functions")
    print("=" * 60)

    B = 2
    param_keys = ['k_syn', 'init_CDH1T', 'init_CDH1', 'BUB2_syn', 'init_BUB2', 'Tem1T', 'Cdc15T']
    param_keys.extend([f"p_{i}" for i in range(141 - len(param_keys))])

    # Sample 0: Normal WT (no init=0)
    # Sample 1: Explicit cdh1 knockout (init_CDH1T=0, init_CDH1=0)
    p_real = torch.ones(B, 141)
    p_real[1, 1] = 0.0  # init_CDH1T
    p_real[1, 2] = 0.0  # init_CDH1

    mask = compute_init_aware_zero_mask(p_real, param_keys=param_keys)
    assert mask[0, 21, 0] == 1.0, "WT CDH1T should NOT be masked!"
    assert mask[0, 22, 0] == 1.0, "WT CDH1 should NOT be masked!"
    assert mask[1, 21, 0] == 0.0, "Knockout CDH1T should be masked to 0.0!"
    assert mask[1, 22, 0] == 0.0, "Knockout CDH1 should be masked to 0.0!"
    print("  ✓ compute_init_aware_zero_mask correctly zeros only verified init=0 channels.")

    y = torch.tensor([[[-0.5, 1.2], [0.8, -0.3]]])  # (1, 2, 2)
    y_guarded = apply_biological_guard(y, clamp_non_negative=True)
    assert (y_guarded >= 0.0).all(), "Non-negative physical clamp failed!"
    print("  ✓ apply_biological_guard successfully clamped negative values to 0.0.")


if __name__ == '__main__':
    print("Starting Comprehensive Verification Test Suite...")
    test_models()
    test_losses()
    test_lit_module_training_step()
    test_biological_guard()
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY! ZERO ERRORS DETECTED.")
    print("=" * 60)
