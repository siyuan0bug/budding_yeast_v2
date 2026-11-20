#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试: BioRefinerDynamicCausalPlus 与 TripleDynamicLoss 集成验证
"""

import os
import sys
import unittest
import torch
import numpy as np

# Ensure root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, os.path.abspath(os.path.join(root_dir, '..')))

from budding_yeast_v2.models.components.bio_refiner_dynamic_causal import (
    BioRefinerDynamicCausalPlus,
    TimeVaryingCausalAdapter,
    build_var_to_gene_tensor
)
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.utils.training.losses import TripleDynamicLoss


class TestDynamicCausalRefiner(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.B = 2
        self.V = 39
        self.T = 500
        self.param_dim = 141

        # 构建虚拟输入
        ic = torch.randn(self.B, self.V, 1, self.T, device=self.device)
        t_grid = torch.linspace(0, 1, self.T, device=self.device).view(1, 1, 1, self.T).expand(self.B, self.V, 1, self.T)
        self.x = torch.cat([ic, t_grid], dim=2)  # (B, 39, 2, T)
        self.p = torch.randn(self.B, self.param_dim, device=self.device)

    def test_var_to_gene_matrix(self):
        M = build_var_to_gene_tensor(self.device)
        self.assertEqual(M.shape, (27, 39))
        self.assertTrue(torch.all(M >= 0))
        # 验证每一行至少有一个非零项
        row_sums = M.sum(dim=1)
        self.assertTrue(torch.all(row_sums > 0))

    def test_forward_and_backward(self):
        model = BioRefinerDynamicCausalPlus(
            num_vars=self.V,
            width=64,
            modes=16,
            param_dim=self.param_dim,
            n_blocks=2,
            coarse_n_blocks=2,
            d_gene=16
        ).to(self.device)

        y_final, y_coarse = model(self.x, self.p, return_coarse=True)
        self.assertEqual(y_final.shape, (self.B, self.V, self.T))
        self.assertEqual(y_coarse.shape, (self.B, self.V, self.T))
        self.assertFalse(torch.isnan(y_final).any(), "NaN found in y_final")

        # 反向传播测试
        loss = y_final.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN in grad of {name}")

    def test_triple_dynamic_loss_integration(self):
        model = BioRefinerDynamicCausalPlus(
            num_vars=self.V,
            width=64,
            modes=16,
            param_dim=self.param_dim,
            n_blocks=2,
            coarse_n_blocks=2,
            d_gene=16
        ).to(self.device)

        criterion = TripleDynamicLoss(
            lambda_rel=0.5,
            lambda_deriv=0.5,
            lambda_cos=0.1,
            lambda_dtw=0.05,
            sub_stride=20
        ).to(self.device)

        pred = model(self.x, self.p)
        target = torch.randn_like(pred)

        loss = criterion(pred, target)
        self.assertTrue(loss.item() > 0)
        self.assertFalse(torch.isnan(loss).any())

        loss.backward()
        self.assertIsNotNone(model.gamma.grad)

    def test_lit_module_registration(self):
        lit_mod = YeastLitModule(
            model_name='biom_refiner_dynamic_causal_plus',
            loss_type='triple_dynamic',
            width=64,
            modes=16,
            n_blocks=2
        ).to(self.device)
        self.assertIsInstance(lit_mod.model, BioRefinerDynamicCausalPlus)


if __name__ == '__main__':
    unittest.main()
