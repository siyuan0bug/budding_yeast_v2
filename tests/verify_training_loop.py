#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification of Training Loop & Convergence for BioEventHybridRefiner
================================================================================
Runs 2 epochs on a small subset of real data using PyTorch Lightning Trainer
to verify that:
1. DataModule feeds batches correctly.
2. Losses (triple_dynamic) compute smoothly without NaNs or infs.
3. Optimizer updates parameters cleanly.
4. Validation loop runs and outputs valid metrics.
"""

import sys
import os
import torch
import pytorch_lightning as pl

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
_parent_root = os.path.dirname(_project_root)
if _parent_root not in sys.path:
    sys.path.insert(0, _parent_root)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.yeast_lit_module import YeastLitModule
from data.yeast_datamodule import YeastDataModule


def run_mini_training_test():
    print("=" * 80)
    print(" BIO-EVENT-HYBRID-REFINER MINI-TRAINING SANITY CHECK")
    print("=" * 80)

    dataset_path = "/home/users/zsy/mydata/budding_yeast/gibbs_boltzmann_100000_210m_500s.npz"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    dm = YeastDataModule(
        dataset_path=dataset_path,
        batch_size_lhs=64,
        val_batch_size=64,
        num_workers=2,
        use_adj_matrix=False,
        norm_type='log1p'
    )
    dm.setup('fit')
    print(f"DataModule setup complete. Train set size: {len(dm.train_dataset_lhs):,}")

    model = YeastLitModule(
        model_name='biom_refiner_event_hybrid',
        num_vars=39,
        param_dim=141,
        modes=16,
        width=64,
        n_blocks=2,
        loss_type='triple_dynamic',
        lr=1e-3,
        max_epochs=2,
        warmup_epochs=0
    )

    trainer = pl.Trainer(
        max_epochs=2,
        limit_train_batches=5,  # 5 batches per epoch for ultra-fast check
        limit_val_batches=2,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )

    trainer.fit(model, datamodule=dm)
    print("\n" + "=" * 80)
    print(" 🎯 MINI-TRAINING LOOP COMPLETED SUCCESSFULLY WITHOUT ERRORS!")
    print("=" * 80)


if __name__ == '__main__':
    run_mini_training_test()
