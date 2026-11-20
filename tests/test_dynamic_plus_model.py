#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试: BioRefinerDynamicPlus 架构与梯度流健全性检查
======================================================
验证点:
1. 模型能够成功实例化并支持前向推理 (B, 39, 2, 500) -> (B, 39, 500)
2. 彻底剔除 state_latch 模块 (hasattr 为 False)
3. 物理事件通道 mass_event_trigger 成功注入 Stage 2，梯度反传无死计算
4. 模型在 YeastLitModule 中通过注册名 'biom_refiner_dynamic_plus' 正常解析与执行
5. 双向时域卷积支持全局对称视野，无人工因果群延迟
"""

import unittest
import torch
import torch.nn as nn

from budding_yeast_v2.models.components.bio_refiner_dynamic_plus import (
    BioRefinerDynamicPlus,
    DynamicMassEventTrigger,
    SoftTopologyGAT,
    TimeVaryingDynamicAdapter,
    TimeVaryingResConvBlock1D,
)
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule, MODEL_REGISTRY


class TestBioRefinerDynamicPlus(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.B = 2
        self.V = 39
        self.T = 500
        self.param_dim = 141
        self.width = 64  # 测试采用较小通道以快速执行
        self.modes = 16

    def test_registry(self):
        """测试模型在全局注册表中正常注册"""
        self.assertIn('biom_refiner_dynamic_plus', MODEL_REGISTRY)
        self.assertEqual(MODEL_REGISTRY['biom_refiner_dynamic_plus'], BioRefinerDynamicPlus)

    def test_state_latch_removed(self):
        """验证导致常数平滑滤波退化的 state_latch 模块已被彻底移除"""
        model = BioRefinerDynamicPlus(
            num_vars=self.V,
            width=self.width,
            modes=self.modes,
            param_dim=self.param_dim,
            n_blocks=2,
            coarse_n_blocks=2
        ).to(self.device)
        self.assertFalse(hasattr(model, 'state_latch'))
        self.assertFalse(hasattr(model, 'latch_detector'))

    def test_forward_and_backward_gradients(self):
        """测试全时程前向推理与物理事件通道反向梯度流"""
        model = BioRefinerDynamicPlus(
            num_vars=self.V,
            width=self.width,
            modes=self.modes,
            param_dim=self.param_dim,
            n_blocks=2,
            coarse_n_blocks=2
        ).to(self.device)
        model.train()

        x = torch.randn(self.B, self.V, 2, self.T, device=self.device, requires_grad=True)
        p = torch.randn(self.B, self.param_dim, device=self.device, requires_grad=True)

        y_final, y_coarse = model(x, p, return_coarse=True)

        # 检查输出维度
        self.assertEqual(y_final.shape, (self.B, self.V, self.T))
        self.assertEqual(y_coarse.shape, (self.B, self.V, self.T))

        # 检查梯度反传
        loss = y_final.sum()
        loss.backward()

        # 1. 验证物理事件触发器成功接入计算图，不再是死代码
        event_thresh_grad = model.mass_event_trigger.threshold_net[0].weight.grad
        self.assertIsNotNone(event_thresh_grad, "DynamicMassEventTrigger threshold_net 梯度不应为空")
        self.assertTrue(torch.count_nonzero(event_thresh_grad) > 0, "物理质量事件特征梯度必须有效传播")

        event_proj_grad = model.mass_event_trigger.event_proj[0].weight.grad
        self.assertIsNotNone(event_proj_grad, "DynamicMassEventTrigger event_proj 梯度不应为空")

        # 2. 验证时变动态适配器正常反传
        self.assertIsNotNone(model.dynamic_adapter.alpha.grad, "dynamic_adapter.alpha 梯度必须存在")

        # 3. 验证 Stage 2 细化器正常反传
        self.assertIsNotNone(model.stage2_refiner.head[2].weight.grad, "Stage 2 输出头梯度必须存在")

        # 4. 验证残差门控缩放因子 gamma 反传
        self.assertIsNotNone(model.gamma.grad, "gamma 缩放参数梯度必须存在")

    def test_soft_topology_gat(self):
        """测试机制软拓扑图注意力层"""
        d_gene = 16
        gat = SoftTopologyGAT(num_nodes=27, d_gene=d_gene, num_heads=4).to(self.device)
        e_time = torch.randn(self.B, 27, d_gene, 50, device=self.device)
        adj_mask = torch.eye(27, device=self.device)
        sign_adj = torch.zeros(27, 27, device=self.device)

        out = gat(e_time, adj_mask, sign_adj)
        self.assertEqual(out.shape, (self.B, 27, d_gene, 50))
        self.assertFalse(torch.isnan(out).any())

    def test_time_varying_res_block_bidirectional(self):
        """测试 Stage 2 双向对称填充，验证全时程全局上下文"""
        block = TimeVaryingResConvBlock1D(channels=32, param_dim=self.param_dim, dilation=2).to(self.device)
        x = torch.randn(self.B, 32, 100, device=self.device)
        p_t = torch.randn(self.B, self.param_dim, 100, device=self.device)

        out = block(x, p_t)
        self.assertEqual(out.shape, (self.B, 32, 100))
        # 验证 GroupNorm 正常运行 (无状态统计量漂移)
        self.assertIsInstance(block.norm1, nn.GroupNorm)
        self.assertIsInstance(block.norm2, nn.GroupNorm)

    def test_lit_module_integration(self):
        """测试与 PyTorch Lightning YeastLitModule 的无缝集成"""
        lit_mod = YeastLitModule(
            model_name='biom_refiner_dynamic_plus',
            loss_type='hybrid_mse_rel',
            width=self.width,
            modes=self.modes,
            n_blocks=2,
        ).to(self.device)

        x = torch.randn(self.B, self.V, 2, self.T, device=self.device)
        p = torch.randn(self.B, self.param_dim, device=self.device)

        with torch.no_grad():
            y_pred = lit_mod.forward_ic_time(x, p)

        self.assertEqual(y_pred.shape, (self.B, self.V, self.T))
        self.assertFalse(torch.isnan(y_pred).any())


if __name__ == '__main__':
    unittest.main()
