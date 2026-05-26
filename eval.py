import os
import argparse
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.utils.metrics import calculate_metrics

CORE_9_INDICES = [0, 1, 2, 3, 4, 20, 22, 33, 35]

# 🌟 新增：从 lhs_v1.py 提取的 39 个变量严格对应名称
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

# 🌟 保持画图逻辑不变（遵照你不修改刻度的要求）
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./eval_result')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
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

    all_preds_denorm, all_targets_denorm = [], []

    with torch.no_grad():
        for local_idx, batch in enumerate(dm.test_dataset_real):
            global_idx = dm._test_idx_real[local_idx]
            m_name = dm._mutant_names[global_idx]
            p_label = dm.pattern_labels[global_idx]
            
            # 🌟 删除了创建子文件夹 m_dir 的代码

            x, p, y = [b.unsqueeze(0).to(args.device) for b in batch]
            pred_denorm = (model.forward_ic_time(x, p) * std_y + mean_y).cpu()
            target_denorm = (y * std_y + mean_y).cpu()

            all_preds_denorm.append(pred_denorm)
            all_targets_denorm.append(target_denorm)

            metrics = calculate_metrics(pred_denorm, target_denorm)
            # 🌟 删除了生成独立 mutant_metrics.json 的代码

            m_str = f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4e} | Rel L2: {metrics['Relative L2']:.4f} | Corr: {metrics['Correlation']:.4f}"
            title_base = f"[{p_label}] {m_name} (Seed: {args.seed})\n{m_str}"
            
            p_np, t_np = pred_denorm[0].numpy(), target_denorm[0].numpy()

            # 🌟 删除了局部变量网格图，只保留 39 变量全景图
            # 🌟 直接保存在 args.save_dir 下，文件名为突变体的真实名字
            save_plot_path = os.path.join(args.save_dir, f"{m_name}.png")
            plot_variable_grid(p_np, t_np, range(39), f"{title_base}\nAll 39 Variables", 
                               save_plot_path, 5, 8, t_max=dm.t_max)

    # 🌟 依然在主目录生成唯一的 global_metrics_denorm.json
    all_preds = torch.cat(all_preds_denorm, dim=0)
    all_targets = torch.cat(all_targets_denorm, dim=0)
    with open(os.path.join(args.save_dir, 'global_metrics_denorm.json'), 'w') as f:
        json.dump({"all_39_vars": calculate_metrics(all_preds, all_targets), "core_9_vars": calculate_metrics(all_preds[:, CORE_9_INDICES, :], all_targets[:, CORE_9_INDICES, :])}, f, indent=2)

if __name__ == '__main__':
    main()