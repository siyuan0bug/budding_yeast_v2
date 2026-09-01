#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HANO / HANO_Dual 短程训练冒烟测试 (CPU, 2 个 batch):
验证真实数据全链路: 数据加载 → 统计量注入 → forward(通量积分+事件跳变[+双路径融合])
→ triple_dynamic 损失 + 事件 BCE 辅助损失 [+ 物理路径辅助损失] → 反向传播 → 参数更新。
"""
import torch
import torch.nn.functional as F

from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule

DATASET = '/home/users/zsy/mydata/budding_yeast/wb_mass_100000_dataset_210min_500steps_dual_labels.npz'

torch.manual_seed(42)

print('=== 1. 加载真实数据 (WT-Blind 100k) ===')
dm = YeastDataModule(dataset_path=DATASET, batch_size_lhs=16, num_workers=2)
dm.setup('fit')
dl = dm.train_dataloader()
batches = [next(iter(dl)), next(iter(dl))]
x, p, y = batches[0]
print(f'   batch: x={tuple(x.shape)}, p={tuple(p.shape)}, y={tuple(y.shape)}')


def run_smoke(model_name):
    print(f'\n=== 2. 实例化 YeastLitModule({model_name}) 并注入统计量 ===')
    lit = YeastLitModule(model_name=model_name, loss_type='triple_dynamic',
                         width=64, n_blocks=2, lr=1e-3)
    lit.model.set_stats(dm.mean_y, dm.std_y, dm.p_mean, dm.p_std, float(dm.t_max))

    opt = torch.optim.AdamW(lit.model.parameters(), lr=1e-3)

    print('=== 3. 训练步 (forward + 损失 + 辅助损失 + backward) ===')
    for step, (x, p, y) in enumerate(batches):
        lit.model.train()
        out, aux = lit.model(x, p, return_aux=True)
        assert out.shape == y.shape, f'shape mismatch: {out.shape} vs {y.shape}'
        assert torch.isfinite(out).all(), '输出含 NaN/Inf!'

        w = torch.ones(len(x))
        loss_main = lit.criterion(out, y, w, 1.0, 1.0)
        loss = loss_main

        # 物理路径辅助损失 (仅 hano_dual)
        if 'physics_out' in aux:
            phys_loss = lit.criterion(aux['physics_out'], y, w, 1.0, 1.0)
            loss = loss + lit.model.lambda_phys * phys_loss

        # 事件 BCE 辅助损失
        labels = lit.model.compute_event_labels(y)                    # (B, 4, T)
        ev_loss = F.binary_cross_entropy_with_logits(aux['event_logits'], labels)
        loss = loss + lit.model.lambda_event * ev_loss

        opt.zero_grad()
        loss.backward()
        gnorm = torch.sqrt(sum((g ** 2).sum() for g in lit.model.parameters()
                               if g.grad is not None))
        n_no_grad = sum(1 for prm in lit.model.parameters() if prm.grad is None)
        opt.step()

        msg = (f'   step {step}: loss_main={loss_main.item():.4f}, '
               f'ev_loss={ev_loss.item():.4f}, total={loss.item():.4f}, '
               f'grad_norm={gnorm.item():.3f}, 无梯度参数={n_no_grad}')
        if 'physics_out' in aux:
            msg += f', phys_loss={phys_loss.item():.4f}'
        if 'gate' in aux:
            msg += f', gate_mean={aux["gate"].mean().item():.3f}'
        print(msg)
        assert torch.isfinite(loss), '损失为 NaN/Inf!'
        assert gnorm > 0, '梯度全为零!'

    print('=== 4. 验证模式 forward (eval) ===')
    lit.model.eval()
    with torch.no_grad():
        pred = lit.model(x, p)
        mse = F.mse_loss(pred, y).item()
    print(f'   eval MSE={mse:.4f}, pred finite={torch.isfinite(pred).all().item()}')

    print('=== 5. 通量可解释性检查: 预测通量速率 ===')
    with torch.no_grad():
        v = aux['flux']
    print(f'   flux shape={tuple(v.shape)}, 非负={bool((v >= 0).all())}, '
          f'mean={v.mean().item():.3e}, max={v.max().item():.3e}')


run_smoke('hano')
run_smoke('hano_dual')

print('\n✅ HANO / HANO_Dual 冒烟测试全部通过: 真实数据训练/验证链路正常')
