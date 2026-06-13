import os
import argparse
import json
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.utils.metrics import calculate_metrics

CORE_9_INDICES = [0, 1, 2, 3, 4, 20, 22, 33, 35]

VARIABLE_NAMES = [
    "MASS", "CLN2", "CLB2", "CLB5", "SIC1", "CDC6", "C2", "C5", "F2", "F5",
    "SIC1P", "C2P", "C5P", "CDC6P", "F2P", "F5P", "SWI5T", "SWI5", "IEP", "CDC20T",
    "CDC20", "CDH1T", "CDH1", "CDC14T", "CDC14", "NET1T", "NET1", "RENT", "TEM1", "CDC15",
    "PPX", "PDS1", "ESP1", "ORI", "BUD", "SPN", "Vi20", "lte1", "BUB2"
]

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

def plot_variable_grid(pred, target, var_indices, title, save_path, rows, cols, t_max=210):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    t = np.linspace(0, t_max, pred.shape[-1])
    for i, var_idx in enumerate(var_indices):
        ax = axes[i]
        ax.plot(t, target[var_idx], 'b-', label='Truth', linewidth=1.5)
        ax.plot(t, pred[var_idx], 'r--', label='Pred', linewidth=1.5)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Concentration')
        var_name = VARIABLE_NAMES[var_idx] if var_idx < len(VARIABLE_NAMES) else f"Var {var_idx}"
        ax.set_title(f'{var_idx}: {var_name}', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=8)
    for i in range(len(var_indices), len(axes)): axes[i].set_visible(False)
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def evaluate_dataset(model, dm, dataset, indices, device, mean_y, std_y, seed, save_dir, 
                     sub_dir, max_plots=None):
    """通用评估函数，评估一个 dataset 并生成图片和 CSV 行。"""
    plot_dir = os.path.join(save_dir, sub_dir)
    os.makedirs(plot_dir, exist_ok=True)
    
    all_preds_denorm, all_targets_denorm = [], []
    csv_rows = []
    plot_count = 0

    with torch.no_grad():
        for local_idx, batch in enumerate(dataset):
            global_idx = indices[local_idx]
            m_name = dm._mutant_names[global_idx]
            p_label = dm.pattern_labels[global_idx]

            x, p, y = [b.unsqueeze(0).to(device) for b in batch]
            pred_denorm = (model.forward_ic_time(x, p) * std_y + mean_y).cpu()
            target_denorm = (y * std_y + mean_y).cpu()

            all_preds_denorm.append(pred_denorm)
            all_targets_denorm.append(target_denorm)

            metrics = calculate_metrics(pred_denorm, target_denorm)
            csv_rows.append({
                'Mutant': m_name,
                'Pattern': p_label,
                'MAE': metrics['MAE'],
                'MSE': metrics['MSE'],
                'Relative_L2': metrics['Relative L2'],
                'Correlation': metrics['Correlation']
            })

            # 生成图片（受 max_plots 限制）
            if max_plots is None or plot_count < max_plots:
                m_str = f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4e} | Rel L2: {metrics['Relative L2']:.4f} | Corr: {metrics['Correlation']:.4f}"
                title_base = f"[{p_label}] {m_name} (Seed: {seed})\n{m_str}"
                p_np, t_np = pred_denorm[0].numpy(), target_denorm[0].numpy()
                save_plot_path = os.path.join(plot_dir, f"{m_name}.png")
                plot_variable_grid(p_np, t_np, range(39), f"{title_base}\nAll 39 Variables",
                                   save_plot_path, 5, 8, t_max=dm.t_max)
                plot_count += 1

    # 汇总指标
    if all_preds_denorm:
        all_preds = torch.cat(all_preds_denorm, dim=0)
        all_targets = torch.cat(all_targets_denorm, dim=0)
        summary = {
            "all_39_vars": calculate_metrics(all_preds, all_targets),
            "core_9_vars": calculate_metrics(all_preds[:, CORE_9_INDICES, :], all_targets[:, CORE_9_INDICES, :]),
            "count": len(all_preds_denorm)
        }
    else:
        summary = {"all_39_vars": {}, "core_9_vars": {}, "count": 0}

    return summary, csv_rows, all_preds_denorm, all_targets_denorm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./eval_result')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lhs_test_plots', type=int, default=100, help='LHS 测试集最多画多少张图')
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
        sub_dir='real_mutants', max_plots=None  # 全部画图
    )

    # ======== 2. 评估 LHS 测试集 ========
    lhs_summary, lhs_csv, lhs_preds, lhs_targets = evaluate_dataset(
        model, dm, dm.test_dataset_lhs, dm._test_idx_lhs,
        args.device, mean_y, std_y, args.seed, args.save_dir,
        sub_dir='lhs_test', max_plots=args.lhs_test_plots
    )

    # ======== 3. 汇总 JSON ========
    all_preds_list = real_preds + lhs_preds
    all_targets_list = real_targets + lhs_targets

    if all_preds_list:
        all_preds = torch.cat(all_preds_list, dim=0)
        all_targets = torch.cat(all_targets_list, dim=0)
        all_test_summary = {
            "all_39_vars": calculate_metrics(all_preds, all_targets),
            "core_9_vars": calculate_metrics(all_preds[:, CORE_9_INDICES, :], all_targets[:, CORE_9_INDICES, :]),
            "count": len(all_preds_list)
        }
    else:
        all_test_summary = {"all_39_vars": {}, "core_9_vars": {}, "count": 0}

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
