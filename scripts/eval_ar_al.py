import os
import argparse
import json
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from budding_yeast_v2.models.ar_model import ARModelWrapper
from budding_yeast_v2.utils.normalization import TaskNormalizer
from budding_yeast_v2.utils.metrics import calculate_metrics

CORE_9_INDICES = [0, 1, 2, 3, 4, 20, 22, 33, 35]
VARIABLE_NAMES = [
    "MASS", "CLN2", "CLB2", "CLB5", "SIC1", "CDC6", "C2", "C5", "F2", "F5",
    "SIC1P", "C2P", "C5P", "CDC6P", "F2P", "F5P", "SWI5T", "SWI5", "IEP", "CDC20T",
    "CDC20", "CDH1T", "CDH1", "CDC14T", "CDC14", "NET1T", "NET1", "RENT", "TEM1", "CDC15",
    "PPX", "PDS1", "ESP1", "ORI", "BUD", "SPN", "Vi20", "lte1", "BUB2"
]

def load_ar_model(ckpt_path, device='cpu'):
    # 先加载到 CPU, 避免 GPU 被训练进程占用时 "CUDA-capable device(s) is/are busy"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    
    # Init Wrapper
    model = ARModelWrapper(cfg['model'], cfg['training'], t_steps=cfg['data']['t_steps'])
    
    # Init and load normalizer
    task_norm = TaskNormalizer()
    if 'task_norm_state' in ckpt:
        task_norm.load_state_dict(ckpt['task_norm_state'])
    task_norm.to(device)
    model.task_norm = task_norm
    
    # Load model weights
    # checkpoint 由 ARModelWrapper.state_dict() 保存, 键名带 'model.' 前缀;
    # 加载到内部 HyperFNO 时需去除前缀
    if 'model_state' in ckpt:
        state = ckpt['model_state']
        if any(k.startswith('model.') for k in state):
            state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
        model.model.load_state_dict(state)
    else:
        model.model.load_state_dict(ckpt)
        
    # 移动整个 ARModelWrapper (含 t_grid buffer + model.model + task_norm) 到目标设备
    model.to(device)
    model.eval()
    return model, cfg

def plot_variable_grid(pred_np, target_np, indices, title, save_path, rows=5, cols=8, t_max=210):
    T_len = pred_np.shape[1]
    t = np.linspace(0, t_max, T_len)
    fig, axes = plt.subplots(rows, cols, figsize=(28, 16))
    axes = axes.flatten()

    for i, var_idx in enumerate(indices):
        ax = axes[i]
        ax.plot(t, target_np[var_idx], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(t, pred_np[var_idx], 'r--', label='Prediction', linewidth=2)
        var_name = VARIABLE_NAMES[var_idx] if var_idx < len(VARIABLE_NAMES) else f"Var {var_idx}"
        ax.set_title(f'{var_idx}: {var_name}', fontweight='bold', fontsize=10)
        ax.set_xlabel('Time (min)')
        if i == 0: ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def extract_pattern_label(name):
    """Simple heuristic to assign a pattern string based on mutant name."""
    if 'WT' in name: return 'Pattern_WT'
    if 'cln1' in name or 'cln2' in name or 'cln3' in name: return 'Pattern_G1'
    if 'clb' in name: return 'Pattern_S_G2_M'
    if 'cdc' in name: return 'Pattern_Arrest'
    return 'Pattern_Other'

def evaluate_raw_dataset(model, Y_raw, P_raw, names, is_real, device, seed, save_dir, sub_dir, max_plots=None):
    """
    Evaluates dataset. 
    Y_raw: (N, C, T), P_raw: (N, P).
    """
    plot_dir = os.path.join(save_dir, sub_dir)
    os.makedirs(plot_dir, exist_ok=True)
    
    csv_rows = []
    all_preds = []
    all_targets = []
    plot_count = 0
    
    for i in range(len(Y_raw)):
        name = str(names[i])
        
        # Only process corresponding split
        is_this_real = not name.startswith("LHS_")
        if is_this_real != is_real:
            continue
            
        p_label = extract_pattern_label(name)
        
        ic = torch.tensor(Y_raw[i, :, 0], dtype=torch.float32).unsqueeze(0).to(device)  # (1, C)
        p = torch.tensor(P_raw[i], dtype=torch.float32).unsqueeze(0).to(device)         # (1, P)
        y = torch.tensor(Y_raw[i], dtype=torch.float32).unsqueeze(0)                    # (1, C, T)
        
        # Roll out
        num_steps = y.shape[-1] - 1
        with torch.no_grad():
            # ARModelWrapper roll_out denormalizes the trajectory internally!
            pred = model.roll_out(ic, p, num_steps).cpu()
            
        all_preds.append(pred)
        all_targets.append(y)
        
        metrics = calculate_metrics(pred, y)
        csv_rows.append({
            'Mutant': name,
            'Pattern': p_label,
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'Relative_L2': metrics['Relative L2'],
            'Correlation': metrics['Correlation']
        })
        
        if max_plots is None or plot_count < max_plots:
            m_str = f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4e} | Rel L2: {metrics['Relative L2']:.4f} | Corr: {metrics['Correlation']:.4f}"
            title_base = f"[{p_label}] {name} (Seed: {seed})\n{m_str}"
            save_plot_path = os.path.join(plot_dir, f"{name}.png")
            
            p_np, t_np = pred[0].numpy(), y[0].numpy()
            plot_variable_grid(p_np, t_np, range(39), f"{title_base}\nAll 39 Variables", save_plot_path, t_max=210)
            plot_count += 1
            
    if all_preds:
        all_preds_t = torch.cat(all_preds, dim=0)
        all_targets_t = torch.cat(all_targets, dim=0)
        summary = {
            "all_39_vars": calculate_metrics(all_preds_t, all_targets_t),
            "core_9_vars": calculate_metrics(all_preds_t[:, CORE_9_INDICES, :], all_targets_t[:, CORE_9_INDICES, :]),
            "count": len(all_preds)
        }
    else:
        summary = {"all_39_vars": {}, "core_9_vars": {}, "count": 0}
        
    return summary, csv_rows, all_preds, all_targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./eval_result')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lhs_test_plots', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Loading AR model from {args.ckpt}...")
    model, cfg = load_ar_model(args.ckpt, device=args.device)
    
    print(f"Loading raw dataset from {args.data_path}...")
    npz_data = np.load(args.data_path, allow_pickle=True)
    names = npz_data['mutant_names']
    Y_raw = npz_data['data']
    P_raw = npz_data['params']

    # 1. Real Mutants
    print("Evaluating Real Mutants...")
    real_summary, real_csv, real_preds, real_targets = evaluate_raw_dataset(
        model, Y_raw, P_raw, names, is_real=True, device=args.device,
        seed=args.seed, save_dir=args.save_dir, sub_dir='real_mutants', max_plots=None
    )

    # 2. LHS Test
    print("Evaluating LHS Test...")
    lhs_summary, lhs_csv, lhs_preds, lhs_targets = evaluate_raw_dataset(
        model, Y_raw, P_raw, names, is_real=False, device=args.device,
        seed=args.seed, save_dir=args.save_dir, sub_dir='lhs_test', max_plots=args.lhs_test_plots
    )

    # 3. Aggregate JSON
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
    print(f"   all_test:     n={all_test_summary['count']}, MAE={all_test_summary.get('all_39_vars', {}).get('MAE', 0):.4f}")
    print(f"   real_mutants: n={real_summary['count']}, MAE={real_summary.get('all_39_vars', {}).get('MAE', 0):.4f}")
    print(f"   lhs_test:     n={lhs_summary['count']}, MAE={lhs_summary.get('all_39_vars', {}).get('MAE', 0):.4f}")

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
    print("Done!")

if __name__ == '__main__':
    main()
