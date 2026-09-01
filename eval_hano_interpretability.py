#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HANO 可解释性评估 (论文核心证据生成器)
=====================================
对训练好的 HANO checkpoint 量化三大结构组件的行为:

1. 通量保真度 (Flux Fidelity)
   预测的 R=98 个反应通量 v_pred(t) vs 真值通量 true_flux_rates(y_true, p)
   → 逐反应 Pearson 相关系数 + 相对误差, 输出 top/bottom 反应表与图。
   证据: 网络确实学到了化学计量分解下的真实反应动力学。

2. 事件检测 (Event Detection)
   事件热图 sigmoid(logits) vs 从真值轨迹提取的事件标签
   → Precision / Recall / F1 (阈值 0.5), 逐事件类型 + 样本热图可视化。
   证据: 事件时钟学会了定位 4 类细胞周期事件。

3. 关系权重 (Multi-Relational Attention)
   softmax(rel_weights) → 6 类生化关系的注意力贡献占比。
   证据: 图注意力自适应利用了先验生化拓扑。

用法:
  python eval_hano_interpretability.py --ckpt <path/to/best-*.ckpt> \
      --dataset_name <npz> --save_dir ./eval_result/hano_interp [--n_samples 64] [--device cuda]
"""
import os
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.models.components.hano import REACTIONS, NUM_REACTIONS
from budding_yeast_v2.utils.physics_eqns_torch import params_tensor_to_dict

EVENT_NAMES = ['ev1_division', 'ev2_S_exit', 'ev3_DNA_replication', 'ev4_spindle']
REL_NAMES = ['complex', 'kinase', 'phosphatase', 'transcription', 'ubiquitination', 'gk_switch']
REACTION_NAMES = [r[0] for r in REACTIONS]


def load_hano(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt.get('hyper_parameters', {})
    model = YeastLitModule(
        model_name=hp.get('model_name', 'hano'), num_vars=hp.get('num_vars', 39),
        param_dim=hp.get('param_dim', 141), modes=hp.get('modes', 32),
        width=hp.get('width', 64), n_blocks=hp.get('n_blocks', 4),
        loss_type=hp.get('loss_type', 'triple_dynamic'),
    )
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    model = model.to(device).eval()
    return model, hp


def collect_batches(dm, loader, model, device, n_samples):
    """收集前 n_samples 个样本的预测/真值/通量/事件热图/真实参数."""
    out = {'x': [], 'p': [], 'y': [], 'flux': [], 'event_prob': [], 'labels': []}
    got = 0
    with torch.no_grad():
        for x, p, y in loader:
            x, p, y = x.to(device), p.to(device), y.to(device)
            pred, aux = model.model(x, p, return_aux=True)
            out['x'].append(x.cpu())
            out['p'].append(p.cpu())
            out['y'].append(y.cpu())
            out['flux'].append(aux['flux'].cpu())
            out['event_prob'].append(torch.sigmoid(aux['event_logits']).cpu())
            out['labels'].append(model.model.compute_event_labels(y).cpu())
            got += len(x)
            if got >= n_samples:
                break
    return {k: torch.cat(v, dim=0)[:n_samples] for k, v in out.items()}


@torch.no_grad()
def eval_flux_fidelity(data, model, device, batch=8):
    """逐反应 Pearson 相关 + 相对误差 (真实空间)."""
    B = data['y'].shape[0]
    cors, relerrs, mags = [], [], []
    for i in range(0, B, batch):
        y_norm = data['y'][i:i + batch].to(device)                # (b,39,T)
        p_norm = data['p'][i:i + batch].to(device)
        m = model.model
        y_real = m._denorm_y(y_norm)                              # (b,39,T)
        p_real = m._denorm_p(p_norm)                              # (b,141)
        b, V, T = y_real.shape
        # 真值通量: (b*T, 39) -> (b, R, T)
        y_flat = y_real.permute(0, 2, 1).reshape(-1, V)
        p_rep = p_real.unsqueeze(1).expand(-1, T, -1).reshape(-1, p_real.shape[-1])
        from budding_yeast_v2.models.components.hano import true_flux_rates
        v_true = true_flux_rates(y_flat, params_tensor_to_dict(p_rep))
        v_true = v_true.reshape(b, T, -1).permute(0, 2, 1)        # (b,R,T)
        v_pred = data['flux'][i:i + batch].to(device)
        # 逐反应统计
        vt = v_true.reshape(b, -1)
        vp = v_pred.reshape(b, -1)
        for r in range(NUM_REACTIONS):
            a, c = vt[:, r], vp[:, r]
            va, vc = a.var(), c.var()
            if va > 1e-12 and vc > 1e-12:
                cor = torch.corrcoef(torch.stack([a, c]))[0, 1].item()
            else:
                cor = 0.0
            scale = a.abs().mean().item() + 1e-9
            cors.append(cor)
            relerrs.append(((c - a).abs().mean() / scale).item())
            mags.append(a.abs().mean().item())
    cors = np.array(cors).reshape(-1, NUM_REACTIONS).mean(axis=0)
    relerrs = np.array(relerrs).reshape(-1, NUM_REACTIONS).mean(axis=0)
    mags = np.array(mags).reshape(-1, NUM_REACTIONS).mean(axis=0)
    return cors, relerrs, mags


def eval_event_detection(data, thr=0.5):
    """事件热图 vs 标签: Precision/Recall/F1 (逐事件类型 + 总体)."""
    prob = data['event_prob']                                    # (B,4,T)
    lab = data['labels']
    pred = (prob >= thr).float()
    metrics = {}
    for k, name in enumerate(EVENT_NAMES):
        tp = (pred[:, k] * lab[:, k]).sum().item()
        fp = (pred[:, k] * (1 - lab[:, k])).sum().item()
        fn = ((1 - pred[:, k]) * lab[:, k]).sum().item()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        metrics[name] = {'precision': prec, 'recall': rec, 'f1': f1,
                         'n_pos': lab[:, k].sum().item()}
    tp = (pred * lab).sum().item()
    fp = (pred * (1 - lab)).sum().item()
    fn = ((1 - pred) * lab).sum().item()
    metrics['overall'] = {
        'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
        'f1': 2 * (tp / max(tp + fp, 1)) * (tp / max(tp + fn, 1)) /
              max(tp / max(tp + fp, 1) + tp / max(tp + fn, 1), 1e-9)}
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--dataset_name', type=str,
                    default='/home/users/zsy/mydata/budding_yeast/wb_mass_100000_dataset_210min_500steps_dual_labels.npz')
    ap.add_argument('--save_dir', type=str, default='./eval_result/hano_interp')
    ap.add_argument('--n_samples', type=int, default=64)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device(args.device)

    print(f'=== 加载模型: {args.ckpt} ===')
    model, hp = load_hano(args.ckpt, device)
    print(f'    model={hp.get("model_name")}, width={hp.get("width")}, loss={hp.get("loss_type")}')

    print('=== 加载数据 ===')
    dm = YeastDataModule(dataset_path=args.dataset_name, batch_size_lhs=8, num_workers=2)
    dm.setup('test')
    model.model.set_stats(dm.mean_y, dm.std_y, dm.p_mean, dm.p_std, float(dm.t_max))

    # 用 LHS 测试集 (真值通量可由 ODE 参数精确计算; real mutant 参数有突变覆盖,同样可算)
    # datamodule: val_dataloader() 返回 [val_lhs, test_lhs, test_real] (dataloader_idx 0/1/2)
    loader = dm.val_dataloader()[1]
    data = collect_batches(dm, loader, model, device, args.n_samples)
    print(f'    收集样本: {data["y"].shape[0]}')

    # ---------- 1. 通量保真度 ----------
    print('=== 1. 通量保真度 (pred flux vs true flux) ===')
    cors, relerrs, mags = eval_flux_fidelity(data, model, device)
    order = np.argsort(-cors)
    summary = {
        'flux_mean_correlation': float(cors.mean()),
        'flux_median_correlation': float(np.median(cors)),
        'flux_frac_cor_gt_0.7': float((cors > 0.7).mean()),
        'top10_reactions': [
            {'name': REACTION_NAMES[r], 'cor': float(cors[r]),
             'relerr': float(relerrs[r]), 'mag': float(mags[r])} for r in order[:10]],
        'bottom10_reactions': [
            {'name': REACTION_NAMES[r], 'cor': float(cors[r]),
             'relerr': float(relerrs[r]), 'mag': float(mags[r])} for r in order[-10:]],
    }
    print(f"    平均 Pearson r = {summary['flux_mean_correlation']:.3f} | "
          f"median = {summary['flux_median_correlation']:.3f} | "
          f"r>0.7 占比 = {summary['flux_frac_cor_gt_0.7']:.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(cors, bins=30, color='steelblue', edgecolor='k')
    axes[0].set_title(f'Per-reaction flux correlation (mean={cors.mean():.3f})')
    axes[0].set_xlabel('Pearson r'); axes[0].set_ylabel('#reactions')
    top = order[:15][::-1]
    axes[1].barh(range(15), cors[top], color='seagreen')
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels([REACTION_NAMES[r] for r in top], fontsize=7)
    axes[1].set_title('Top-15 reactions by correlation')
    plt.tight_layout(); plt.savefig(f'{args.save_dir}/flux_fidelity.png', dpi=150); plt.close()

    # ---------- 2. 事件检测 ----------
    print('=== 2. 事件检测 (heatmap vs labels) ===')
    ev_metrics = eval_event_detection(data)
    for name in EVENT_NAMES:
        m = ev_metrics[name]
        print(f"    {name:20s} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} (pos={m['n_pos']:.0f})")
    print(f"    overall             F1={ev_metrics['overall']['f1']:.3f}")

    fig, axes = plt.subplots(2, 4, figsize=(18, 5))
    for k, name in enumerate(EVENT_NAMES):
        axes[0, k].imshow(data['labels'][:4, k].numpy(), aspect='auto', cmap='Reds')
        axes[0, k].set_title(f'{name} label', fontsize=9)
        axes[1, k].imshow(data['event_prob'][:4, k].numpy(), aspect='auto',
                          cmap='Blues', vmin=0, vmax=1)
        axes[1, k].set_title(f'{name} prob (F1={ev_metrics[name]["f1"]:.2f})', fontsize=9)
    plt.tight_layout(); plt.savefig(f'{args.save_dir}/event_detection.png', dpi=150); plt.close()

    # ---------- 3. 关系权重 ----------
    print('=== 3. 多重关系注意力权重 ===')
    graph_attn = getattr(model.model, 'graph_attn', None)
    if graph_attn is not None and hasattr(graph_attn, 'rel_weights'):
        w = torch.softmax(graph_attn.rel_weights.detach().cpu(), dim=0).numpy()
        rel_summary = {n: float(x) for n, x in zip(REL_NAMES, w)}
        print('    ' + ' | '.join(f'{n}={x:.3f}' for n, x in zip(REL_NAMES, w)))
        plt.figure(figsize=(7, 4))
        plt.bar(REL_NAMES, w, color='darkorange', edgecolor='k')
        plt.ylabel('softmax weight'); plt.title('Multi-relational attention weights')
        plt.xticks(rotation=20)
        plt.tight_layout(); plt.savefig(f'{args.save_dir}/relation_weights.png', dpi=150); plt.close()
    else:
        rel_summary = {'note': 'model variant has no graph attention (use_graph=False)'}
        print('    (无图注意力组件)')

    # ---------- 4. 样本通量轨迹对比 (幅度最大的 6 个反应) ----------
    fig, axes = plt.subplots(2, 3, figsize=(16, 6))
    i0 = 0
    from budding_yeast_v2.models.components.hano import true_flux_rates
    m = model.model
    with torch.no_grad():
        y_real = m._denorm_y(data['y'][i0:i0 + 1].to(device))
        p_real = m._denorm_p(data['p'][i0:i0 + 1].to(device))
        b, V, T = y_real.shape
        y_flat = y_real.permute(0, 2, 1).reshape(-1, V)
        p_rep = p_real.unsqueeze(1).expand(-1, T, -1).reshape(-1, p_real.shape[-1])
        v_true0 = true_flux_rates(y_flat, params_tensor_to_dict(p_rep)).reshape(b, T, -1).permute(0, 2, 1)
    v_pred0 = data['flux'][i0:i0 + 1]
    top_mag = np.argsort(-mags)[:6]
    for ax, r in zip(axes.flat, top_mag):
        ax.plot(v_true0[0, r].cpu().numpy(), 'b-', label='true', lw=1.5)
        ax.plot(v_pred0[0, r].numpy(), 'r--', label='pred', lw=1.5)
        ax.set_title(f'{REACTION_NAMES[r]} (r={cors[r]:.2f})', fontsize=9)
        ax.legend(fontsize=7)
    plt.suptitle('Sample-0 reaction flux trajectories', fontsize=11)
    plt.tight_layout(); plt.savefig(f'{args.save_dir}/flux_trajectories_sample0.png', dpi=150); plt.close()

    # ---------- 汇总 ----------
    result = {'ckpt': args.ckpt, 'n_samples': int(data['y'].shape[0]),
              'flux_fidelity': summary, 'event_detection': ev_metrics,
              'relation_weights': rel_summary}
    with open(f'{args.save_dir}/interpretability_summary.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\n✅ 结果已保存至 {args.save_dir}/ (json + 4 张图)')


if __name__ == '__main__':
    main()
