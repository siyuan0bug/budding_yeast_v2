import os
import argparse
import json
import csv
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.data.lhs_v1_origin import get_default_params, mutant_rules
from budding_yeast_v2.utils.metrics import calculate_metrics

CORE_9_INDICES = [0, 1, 2, 3, 4, 20, 22, 33, 35]

VARIABLE_NAMES = [
    "MASS", "CLN2", "CLB2", "CLB5", "SIC1", "CDC6", "C2", "C5", "F2", "F5",
    "SIC1P", "C2P", "C5P", "CDC6P", "F2P", "F5P", "SWI5T", "SWI5", "IEP", "CDC20T",
    "CDC20", "CDH1T", "CDH1", "CDC14T", "CDC14", "NET1T", "NET1", "RENT", "TEM1", "CDC15",
    "PPX", "PDS1", "ESP1", "ORI", "BUD", "SPN", "Vi20", "lte1", "BUB2"
]

# ======== 突变参数信息提取 ========
_DEFAULTS_DICT = get_default_params()
_DEFAULTS_DICT.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
PARAM_KEYS = sorted([k for k in _DEFAULTS_DICT.keys() if k != 'is_mutant_104'])
_DEFAULT_PARAM_VEC = np.array([_DEFAULTS_DICT[k] for k in PARAM_KEYS])

def get_real_mutations_str(mutant_name):
    rules = mutant_rules.get(mutant_name, {})
    if not rules:
        return "WT (Default)"
    return ", ".join(f"{k}={v:.4g}" for k, v in rules.items())

def get_lhs_perturbation_str(lhs_name, lhs_param_vec_real):
    mother_name = None
    for rule_name in sorted(mutant_rules.keys(), key=len, reverse=True):
        if lhs_name.startswith(rule_name + '_LHS'):
            mother_name = rule_name
            break

    if mother_name is None:
        return "Unknown", "Unknown mother mutant"

    mother_rules = mutant_rules.get(mother_name, {})
    mother_defaults = dict(_DEFAULTS_DICT)
    mother_defaults.update(mother_rules)
    mother_param_vec = np.array([mother_defaults[k] for k in PARAM_KEYS])

    mutated_info = []
    for lhs_val, mother_val, name in zip(lhs_param_vec_real, mother_param_vec, PARAM_KEYS):
        if not np.isclose(lhs_val, mother_val, rtol=1e-4, atol=1e-6):
            if abs(mother_val) > 1e-8:
                ratio = lhs_val / mother_val
                mutated_info.append(f"{name}={lhs_val:.4g} (mother={mother_val:.4g}, {ratio:.2f}x)")
            else:
                mutated_info.append(f"{name}={lhs_val:.4g} (mother=0)")

    if not mutated_info:
        return mother_name, "No Perturbations (Identical to Base Mutant)"

    chunks = [" | ".join(mutated_info[i:i+5]) for i in range(0, len(mutated_info), 5)]
    return mother_name, "\n".join(chunks)

def load_model_from_checkpoint(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt.get('hyper_parameters', {})
    model = YeastLitModule(
        model_name=hp.get('model_name', 'pure_fno'), num_vars=hp.get('num_vars', 39),
        param_dim=hp.get('param_dim', 141), modes=hp.get('modes', 24),
        width=hp.get('width', 64), n_blocks=hp.get('n_blocks', 4),
        adj_matrix=hp.get('adj_matrix', None), causal_matrix=hp.get('causal_matrix', None),
        loss_type=hp.get('loss_type', 'physics_informed'), wavelet=hp.get('wavelet', 'haar'),
        d_lambda=hp.get('d_lambda', 32), ode_method=hp.get('ode_method', 'dopri5')
    )
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    model.eval()
    return model, hp

def _render_and_save_plot(task):
    """独立的多进程画图函数，充分利用多核 CPU"""
    pred, target, title, save_path, rows, cols, t_max = task
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    t = np.linspace(0, t_max, pred.shape[-1])
    for i in range(pred.shape[0]):
        ax = axes[i]
        ax.plot(t, target[i], 'b-', label='Truth', linewidth=1.5)
        ax.plot(t, pred[i], 'r--', label='Pred', linewidth=1.5)
        ax.set_xlabel('Time (min)', fontsize=8)
        ax.set_ylabel('Concentration', fontsize=8)
        var_name = VARIABLE_NAMES[i] if i < len(VARIABLE_NAMES) else f"Var {i}"
        ax.set_title(f'{i}: {var_name}', fontweight='bold', fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for i in range(pred.shape[0], len(axes)):
        axes[i].set_visible(False)
    plt.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def evaluate_dataset(model, dm, dataset, indices, device, mean_y, std_y, seed, save_dir, 
                     sub_dir, batch_size=512, max_plots=None, max_workers=16):
    """
    高度优化的通用评估函数：
    1. 使用 GPU 大 Batch 并行推理，将千级样本前向时间压至 < 0.2 秒；
    2. 向量化计算 Per-mutant 指标；
    3. 使用多进程进程池并行绘制 matplotlib 图片。
    """
    plot_dir = os.path.join(save_dir, sub_dir)
    os.makedirs(plot_dir, exist_ok=True)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    all_preds_denorm_list = []
    all_targets_denorm_list = []
    all_p_real_list = []

    # 1. GPU 大 Batch 高速推理
    with torch.no_grad():
        for batch in loader:
            x, p, y = [b.to(device, non_blocking=True) for b in batch]
            pred = model.forward_ic_time(x, p)
            
            pred_denorm = pred * std_y + mean_y
            target_denorm = y * std_y + mean_y
            p_real = p * dm.p_std.to(device) + dm.p_mean.to(device)
            
            all_preds_denorm_list.append(pred_denorm.cpu())
            all_targets_denorm_list.append(target_denorm.cpu())
            all_p_real_list.append(p_real.cpu())

    all_preds = torch.cat(all_preds_denorm_list, dim=0)    # (N, V, T)
    all_targets = torch.cat(all_targets_denorm_list, dim=0) # (N, V, T)
    all_p_real = torch.cat(all_p_real_list, dim=0)          # (N, param_dim)

    # 2. 向量化构建 CSV 记录与画图任务
    csv_rows = []
    plot_tasks = []
    N = len(indices)

    for i in range(N):
        global_idx = indices[i]
        m_name = dm._mutant_names[global_idx]
        p_label = dm.pattern_labels[global_idx]
        
        pred_i = all_preds[i]     # (V, T)
        target_i = all_targets[i] # (V, T)
        p_real_i = all_p_real[i].numpy()
        
        metrics = calculate_metrics(pred_i, target_i)
        csv_rows.append({
            'Mutant': m_name,
            'Pattern': p_label,
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'Relative_L2': metrics['Relative L2'],
            'Correlation': metrics['Correlation']
        })

        if max_plots is None or i < max_plots:
            m_str = f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4e} | Rel L2: {metrics['Relative L2']:.4f} | Corr: {metrics['Correlation']:.4f}"
            title_base = f"[{p_label}] {m_name} (Seed: {seed})\n{m_str}"
            
            if '_LHS_' in m_name:
                mother_name, perturb_str = get_lhs_perturbation_str(m_name, p_real_i)
                title_base += f"\nMother: {mother_name} | Perturbations (mother_val, ratio):\n{perturb_str}"
            elif str(m_name).startswith('MASS_'):
                title_base += f"\nMASS synthetic sample (mutant-aware stratified sampling)"
            else:
                mut_str = get_real_mutations_str(m_name)
                title_base += f"\nMutations: {mut_str}"

            save_plot_path = os.path.join(plot_dir, f"{m_name}.png")
            plot_tasks.append((pred_i.numpy(), target_i.numpy(), f"{title_base}\nAll 39 Variables", save_plot_path, 5, 8, dm.t_max))

    # 3. 多进程并发画图
    if plot_tasks:
        workers = min(max_workers, os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_render_and_save_plot, plot_tasks))

    # 4. 全局汇总指标
    summary = {
        "all_39_vars": calculate_metrics(all_preds, all_targets),
        "core_9_vars": calculate_metrics(all_preds[:, CORE_9_INDICES, :], all_targets[:, CORE_9_INDICES, :]),
        "count": N
    }

    return summary, csv_rows, [all_preds], [all_targets]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='/home/users/zsy/mydata/budding_yeast')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1024, help='GPU 推理 Batch Size (A6000 48GB 推荐 1024)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./eval_result')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lhs_test_plots', type=int, default=50, help='LHS 测试集最多画多少张图')
    parser.add_argument('--plot_workers', type=int, default=16, help='CPU 多进程画图核数')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    model, _ = load_model_from_checkpoint(args.ckpt, device=args.device)
    model = model.to(args.device)

    dm = YeastDataModule(
        dataset_path=os.path.join(args.data_dir, args.dataset_name),
        val_batch_size=args.batch_size, num_workers=args.num_workers, use_adj_matrix=False
    )
    dm.setup('test')
    mean_y, std_y = dm.mean_y.to(args.device), dm.std_y.to(args.device)

    # ======== 1. 评估 Real Mutants ========
    real_summary, real_csv, real_preds, real_targets = evaluate_dataset(
        model, dm, dm.test_dataset_real, dm._test_idx_real,
        args.device, mean_y, std_y, args.seed, args.save_dir,
        sub_dir='real_mutants', batch_size=args.batch_size, max_plots=None, max_workers=args.plot_workers
    )

    # ======== 2. 评估 LHS 测试集 ========
    lhs_summary, lhs_csv, lhs_preds, lhs_targets = evaluate_dataset(
        model, dm, dm.test_dataset_lhs, dm._test_idx_lhs,
        args.device, mean_y, std_y, args.seed, args.save_dir,
        sub_dir='lhs_test', batch_size=args.batch_size, max_plots=args.lhs_test_plots, max_workers=args.plot_workers
    )

    # ======== 3. 汇总 JSON ========
    all_preds = torch.cat(real_preds + lhs_preds, dim=0)
    all_targets = torch.cat(real_targets + lhs_targets, dim=0)

    all_test_summary = {
        "all_39_vars": calculate_metrics(all_preds, all_targets),
        "core_9_vars": calculate_metrics(all_preds[:, CORE_9_INDICES, :], all_targets[:, CORE_9_INDICES, :]),
        "count": len(all_preds)
    }

    global_metrics = {
        "all_test": all_test_summary,
        "real_mutants": real_summary,
        "lhs_test": lhs_summary,
    }

    json_path = os.path.join(args.save_dir, 'global_metrics_denorm.json')
    with open(json_path, 'w') as f:
        json.dump(global_metrics, f, indent=2)
    print(f"📊 Global metrics JSON saved to: {json_path}")
    print(f"   all_test:     n={all_test_summary['count']}, MAE={all_test_summary['all_39_vars']['MAE']:.4f}, Corr={all_test_summary['all_39_vars']['Correlation']:.4f}")
    print(f"   real_mutants: n={real_summary['count']}, MAE={real_summary['all_39_vars']['MAE']:.4f}, Corr={real_summary['all_39_vars']['Correlation']:.4f}")
    print(f"   lhs_test:     n={lhs_summary['count']}, MAE={lhs_summary['all_39_vars']['MAE']:.4f}, Corr={lhs_summary['all_39_vars']['Correlation']:.4f}")

    # ======== 4. 输出 CSV ========
    def write_csv(rows, path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Mutant', 'Pattern', 'MAE', 'MSE', 'Relative_L2', 'Correlation']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"📝 CSV saved to: {path}")

    write_csv(real_csv, os.path.join(args.save_dir, 'per_mutant_metrics.csv'))
    write_csv(lhs_csv, os.path.join(args.save_dir, 'per_lhs_test_metrics.csv'))

if __name__ == '__main__':
    main()
