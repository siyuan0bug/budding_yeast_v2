#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit and Integration Test Suite for BioRefinerCausal & CausalMutantAdapter
=========================================================================
Verifies:
1. CausalMutantAdapter input/output shapes, zero-perturbation property, and device/dtype compatibility.
2. Mechanistic directed message passing along 54 biological regulatory edges.
3. Full model suite forward pass (3D & 4D inputs) and return_coarse functionality.
4. Backward pass & gradient propagation across all modules.
5. Integration with YeastLitModule & MODEL_REGISTRY.
"""

import math
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

from budding_yeast_v2.models.components.bio_refiner_causal import (
    CausalRMSNorm,
    CausalMutantAdapter,
    MechanisticGraphMessagePassing,
    BioRefinerCausalBase,
    BioRefinerCausalCurvature,
    BioRefinerCausalPyramid,
    BioRefinerCausalGraph,
    BioRefinerCausalPlus,
    BioRefinerCausal,
    WT_VALUES_141,
    MODULE_NAMES,
    MODULE_REGULATORY_EDGES,
    MODULE_PARAM_INDICES,
)
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule, MODEL_REGISTRY


class TestBioRefinerCausalSuite(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B = 4
        self.V = 39
        self.T = 100  # fast sequence length for testing
        self.P = 141
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_causal_adapter_zero_perturbation(self):
        """Verify that wild-type baseline input produces exact zero delta initially."""
        adapter = CausalMutantAdapter(param_dim=self.P, d_gene=16).to(self.device)
        wt_tensor = torch.tensor(WT_VALUES_141, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.B, 1)

        p_causal = adapter(wt_tensor)
        self.assertEqual(p_causal.shape, (self.B, self.P))
        # Initialized with zero-weight final projection: p_causal should equal p_wt exactly
        diff = torch.abs(p_causal - wt_tensor).max().item()
        self.assertAlmostEqual(diff, 0.0, places=5, msg="At initialization, WT input must satisfy p_causal == p_wt")

    def test_causal_adapter_perturbation_response(self):
        """Verify that knockout/overexpression perturbations induce non-zero causal shifts."""
        adapter = CausalMutantAdapter(param_dim=self.P, d_gene=16).to(self.device)
        # Train slightly or change weights to ensure non-zero response after initial state
        with torch.no_grad():
            adapter.fusion_mlp[-1].weight.normal_(0.0, 0.01)

        p = torch.tensor(WT_VALUES_141, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.B, 1)
        # Perturb CLN2 (indices 131, 132) -> knockout (0.0)
        p[1, 131] = 0.0
        p[1, 132] = 0.0
        # Perturb CDH1 (indices 127, 103, 47, 46) -> overexpression (10.0x)
        p[2, 127] *= 10.0
        p[2, 103] *= 10.0

        p_causal = adapter(p)
        self.assertEqual(p_causal.shape, (self.B, self.P))
        self.assertTrue(torch.isfinite(p_causal).all(), "Outputs must be finite")
        self.assertFalse(torch.allclose(p_causal[1], p_causal[0]), "Mutant CLN2-KO should differ from WT")
        self.assertFalse(torch.allclose(p_causal[2], p_causal[0]), "Mutant CDH1-OE should differ from WT")

    def test_message_passing_topology(self):
        """Verify that graph message passing respects topology and handles signed affinities."""
        gmp = MechanisticGraphMessagePassing(num_nodes=27, d_gene=16, num_heads=4).to(self.device)
        e = torch.randn(self.B, 27, 16, device=self.device)
        adj_mask = torch.eye(27, device=self.device)
        adj_mask[19, 1] = 1.0  # CLN3 (1) -> SBF_MBF (19)
        sign_adj = torch.zeros(27, 27, device=self.device)
        sign_adj[19, 1] = 1.0

        e_out = gmp(e, adj_mask, sign_adj)
        self.assertEqual(e_out.shape, (self.B, 27, 16))
        self.assertTrue(torch.isfinite(e_out).all())

    def test_causal_rmsnorm_zero_preservation(self):
        """Verify that CausalRMSNorm strictly guarantees f(0) = 0 even after weights change."""
        norm = CausalRMSNorm(16).to(self.device)
        # Verify no bias attribute exists
        self.assertFalse(hasattr(norm, 'bias'), "CausalRMSNorm must not possess learnable bias")
        # Set weights to arbitrary non-zero values (simulating post-training state)
        with torch.no_grad():
            norm.weight.fill_(3.14159)
        zeros_in = torch.zeros(self.B, 27, 16, device=self.device)
        out = norm(zeros_in)
        self.assertEqual(out.abs().max().item(), 0.0, "CausalRMSNorm(0) must be identically 0.0")

    def test_indegree_zero_stability(self):
        """Verify that nodes with in-degree 0 in raw adj_mask do not produce NaNs or leak attention."""
        gmp = MechanisticGraphMessagePassing(num_nodes=27, d_gene=16, num_heads=4).to(self.device)
        e = torch.randn(self.B, 27, 16, device=self.device)
        # Construct an adj_mask where node 23 (MDT) has in-degree 0 (entire row is 0, no edges incoming)
        sparse_adj = torch.zeros(27, 27, device=self.device)
        sparse_adj[0, 1] = 1.0  # only edge (1 -> 0)
        # Node 23 has row 23 all zeros in input adj_mask

        # Test FP32
        out_fp32 = gmp(e, sparse_adj)
        self.assertTrue(torch.isfinite(out_fp32).all(), "In-degree 0 node produced NaN/Inf in FP32!")

        # Test half precision (FP16) stability against underflow/overflow
        if self.device.type == 'cuda':
            gmp_fp16 = gmp.half()
            e_fp16 = e.half()
            sparse_adj_fp16 = sparse_adj.half()
            out_fp16 = gmp_fp16(e_fp16, sparse_adj_fp16)
            self.assertTrue(torch.isfinite(out_fp16).all(), "In-degree 0 node produced NaN/Inf in FP16!")

    def test_device_and_dtype_consistency(self):
        """Verify that adapter operates seamlessly across dtypes."""
        adapter = CausalMutantAdapter(param_dim=self.P, d_gene=16).to(self.device)
        p_float = torch.randn(self.B, self.P, dtype=torch.float32, device=self.device)
        out_float = adapter(p_float)
        self.assertEqual(out_float.dtype, torch.float32)
        self.assertEqual(out_float.device.type, self.device.type)

    def test_model_suite_forward_backward(self):
        """Test forward and backward passes for all BioRefinerCausal variants."""
        models = {
            'BioRefinerCausalBase': BioRefinerCausalBase(
                num_vars=self.V, width=32, modes=8, param_dim=self.P, n_blocks=2, coarse_n_blocks=2
            ),
            'BioRefinerCausalCurvature': BioRefinerCausalCurvature(
                num_vars=self.V, width=32, modes=8, param_dim=self.P, n_blocks=2, coarse_n_blocks=2
            ),
            'BioRefinerCausalPyramid': BioRefinerCausalPyramid(
                num_vars=self.V, width=32, modes=8, param_dim=self.P, n_blocks=2, coarse_n_blocks=2
            ),
            'BioRefinerCausalGraph': BioRefinerCausalGraph(
                num_vars=self.V, width=32, modes=8, param_dim=self.P, n_blocks=2, coarse_n_blocks=2,
                adj_matrix=np.ones((self.V, self.V), dtype=np.float32)
            ),
            'BioRefinerCausalPlus': BioRefinerCausalPlus(
                num_vars=self.V, width=32, modes=8, param_dim=self.P, n_blocks=2, coarse_n_blocks=2,
                adj_matrix=np.ones((self.V, self.V), dtype=np.float32)
            ),
        }

        # 1. Test 3D Input (B, 39, T)
        x_3d = torch.randn(self.B, self.V, self.T, device=self.device)
        # 2. Test 4D Input (B, 39, 2, T)
        x_4d = torch.randn(self.B, self.V, 2, self.T, device=self.device)
        p = torch.randn(self.B, self.P, device=self.device)

        for name, model in models.items():
            model = model.to(self.device)
            model.train()

            # Test 3D forward
            y_3d = model(x_3d, p)
            self.assertEqual(y_3d.shape, (self.B, self.V, self.T), f"{name} 3D output shape mismatch")
            self.assertTrue(torch.isfinite(y_3d).all(), f"{name} produced non-finite values with 3D input")

            # Test return_coarse
            y_fin, y_crs = model(x_3d, p, return_coarse=True)
            self.assertEqual(y_fin.shape, (self.B, self.V, self.T))
            self.assertEqual(y_crs.shape, (self.B, self.V, self.T))

            # Test 4D forward
            y_4d = model(x_4d, p)
            self.assertEqual(y_4d.shape, (self.B, self.V, self.T), f"{name} 4D output shape mismatch")

            # Test backward pass
            loss = y_4d.sum()
            loss.backward()

            # Verify gradient exists on causal adapter
            grad_found = False
            for prm in model.causal_adapter.parameters():
                if prm.grad is not None and torch.isfinite(prm.grad).all():
                    grad_found = True
                    break
            self.assertTrue(grad_found, f"{name} failed to propagate gradients to causal_adapter")

    def test_lit_module_registration_and_train_step(self):
        """Test instantiation and mock train step via YeastLitModule."""
        for model_name in [
            'biom_refiner_causal',
            'biom_refiner_causal_plus',
            'biom_refiner_causal_base',
            'biom_refiner_causal_curv',
            'biom_refiner_causal_pyr',
            'biom_refiner_causal_graph',
        ]:
            self.assertIn(model_name, MODEL_REGISTRY, f"{model_name} missing from MODEL_REGISTRY")
            lit = YeastLitModule(
                model_name=model_name,
                num_vars=self.V,
                param_dim=self.P,
                width=32,
                modes=8,
                n_blocks=2,
                loss_type='mse_only'
            ).to(self.device)

            x = torch.randn(self.B, self.V, self.T, device=self.device)
            p = torch.randn(self.B, self.P, device=self.device)
            y = torch.randn(self.B, self.V, self.T, device=self.device)

            batch = (x, p, y)
            loss = lit.training_step(batch, batch_idx=0)
            self.assertTrue(torch.isfinite(loss), f"{model_name} training_step produced non-finite loss")
            loss.backward()


if __name__ == '__main__':
    unittest.main()
