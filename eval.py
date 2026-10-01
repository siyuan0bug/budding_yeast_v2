import os
import sys

# Ensure repository root is in sys.path regardless of where eval.py is launched from
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_current_dir)
for _p in [_repo_root, _current_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import csv
import glob
import re
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.data.mass_dataset_generator import get_default_params, mutant_rules
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

def get_synth_perturbation_str(synth_name, synth_param_vec_real):
    mother_name = None
    for rule_name in sorted(mutant_rules.keys(), key=len, reverse=True):
        if synth_name.startswith(rule_name + '_LHS') or synth_name.startswith(rule_name + '_SYNTH'):
            mother_name = rule_name
            break

    if mother_name is None:
        return "Unknown", "Unknown mother mutant"

    mother_rules = mutant_rules.get(mother_name, {})
    mother_defaults = dict(_DEFAULTS_DICT)
    mother_defaults.update(mother_rules)
    mother_param_vec = np.array([mother_defaults[k] for k in PARAM_KEYS])

    mutated_info = []
    for s_val, mother_val, name in zip(synth_param_vec_real, mother_param_vec, PARAM_KEYS):
        if not np.isclose(s_val, mother_val, rtol=1e-4, atol=1e-6):
            if abs(mother_val) > 1e-8:
                ratio = s_val / mother_val
                mutated_info.append(f"{name}={s_val:.4g} (mother={mother_val:.4g}, {ratio:.2f}x)")
            else:
                mutated_info.append(f"{name}={s_val:.4g} (mother=0)")

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
    try:
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
    except Exception as e:
        print(f"⚠️ 画图失败 ({save_path}): {e}")
        try:
            plt.close('all')
        except Exception:
            pass

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
            x, p, y = [b.to(device, non_blocking=True) for b in batch[:3]]
            pred = model.forward_ic_time(x, p)
            
            pred_denorm = dm.denormalize_y(pred, device)
            target_denorm = dm.denormalize_y(y, device)
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
            
            if '_LHS_' in m_name or '_SYNTH_' in m_name:
                mother_name, perturb_str = get_synth_perturbation_str(m_name, p_real_i)
                title_base += f"\nMother: {mother_name} | Perturbations (mother_val, ratio):\n{perturb_str}"
            elif str(m_name).startswith('MASS_'):
                title_base += f"\nMASS synthetic sample (mutant-aware stratified sampling)"
            else:
                mut_str = get_real_mutations_str(m_name)
                title_base += f"\nMutations: {mut_str}"

            save_plot_path = os.path.join(plot_dir, f"{m_name}.png")
            plot_tasks.append((pred_i.numpy(), target_i.numpy(), f"{title_base}\nAll 39 Variables", save_plot_path, 5, 8, dm.t_max))

    # 3. 全局汇总指标
    summary = {
        "all_39_vars": calculate_metrics(all_preds, all_targets),
        "core_9_vars": calculate_metrics(all_preds[:, CORE_9_INDICES, :], all_targets[:, CORE_9_INDICES, :]),
        "count": N
    }

    return summary, csv_rows, [all_preds], [all_targets], plot_tasks

def evaluate_model_on_datamodule(model, dm, device=None, save_dir=None, seed=42, batch_size=512, 
                                 max_plots=None, max_real_plots=None, synth_test_plots=50, plot_workers=16):
    """
    对给定的 model 和 dm 执行全量评测，生成 global_metrics_denorm.json, per_mutant_metrics.csv 等标准成果。
    """
    if save_dir is None:
        save_dir = './eval_result'
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    model = model.to(device)
    model.eval()

    mean_y, std_y = dm.mean_y.to(device), dm.std_y.to(device)

    # 确定画图数量限制：
    # 真实突变体 (126株) 是最重要的生物学验证基准，除非明确指定为 0，否则默认必须全量保存所有图片 (None)。
    if max_plots == 0:
        actual_real_plots = 0
        actual_synth_plots = 0
    else:
        actual_real_plots = max_real_plots  # None 即全量保存全部 126 张真实突变体图片
        actual_synth_plots = synth_test_plots

    # ======== 1. 评估 Real Mutants ========
    real_summary, real_csv, real_preds, real_targets, real_plot_tasks = evaluate_dataset(
        model, dm, dm.test_dataset_real, dm._test_idx_real,
        device, mean_y, std_y, seed, save_dir,
        sub_dir='real_mutants', batch_size=batch_size, max_plots=actual_real_plots, max_workers=plot_workers
    )

    # ======== 2. 评估 Synthetic 测试集 ========
    synth_dataset = dm.test_dataset_synth
    synth_idx = dm._test_idx_synth
    synth_summary, synth_csv, synth_preds, synth_targets, synth_plot_tasks = evaluate_dataset(
        model, dm, synth_dataset, synth_idx,
        device, mean_y, std_y, seed, save_dir,
        sub_dir='synth_test', batch_size=batch_size, max_plots=actual_synth_plots, max_workers=plot_workers
    )

    # ======== 3. 汇总 JSON (立即落盘) ========
    all_preds = torch.cat(real_preds + synth_preds, dim=0)
    all_targets = torch.cat(real_targets + synth_targets, dim=0)

    all_test_summary = {
        "all_39_vars": calculate_metrics(all_preds, all_targets),
        "core_9_vars": calculate_metrics(all_preds[:, CORE_9_INDICES, :], all_targets[:, CORE_9_INDICES, :]),
        "count": len(all_preds)
    }

    global_metrics = {
        "all_test": all_test_summary,
        "real_mutants": real_summary,
        "synth_test": synth_summary,
    }

    json_path = os.path.join(save_dir, 'global_metrics_denorm.json')
    with open(json_path, 'w') as f:
        json.dump(global_metrics, f, indent=2)
    print(f"📊 Global metrics JSON saved to: {json_path}")
    print(f"   all_test:     n={all_test_summary['count']}, MAE={all_test_summary['all_39_vars']['MAE']:.4f}, Corr={all_test_summary['all_39_vars']['Correlation']:.4f}")
    print(f"   real_mutants: n={real_summary['count']}, MAE={real_summary['all_39_vars']['MAE']:.4f}, Corr={real_summary['all_39_vars']['Correlation']:.4f}")
    print(f"   synth_test:   n={synth_summary['count']}, MAE={synth_summary['all_39_vars']['MAE']:.4f}, Corr={synth_summary['all_39_vars']['Correlation']:.4f}")

    # ======== 4. 输出 CSV 记录 (立即落盘) ========
    def write_csv(rows, path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Mutant', 'Pattern', 'MAE', 'MSE', 'Relative_L2', 'Correlation']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"📝 CSV saved to: {path}")

    write_csv(real_csv, os.path.join(save_dir, 'per_mutant_metrics.csv'))
    write_csv(synth_csv, os.path.join(save_dir, 'per_synth_test_metrics.csv'))

    # ======== 5. 多进程并发画图 (若有请求) ========
    all_plot_tasks = real_plot_tasks + synth_plot_tasks
    if all_plot_tasks:
        print(f"🎨 正在多进程并发绘制曲线图 (共 {len(all_plot_tasks)} 幅)...")
        workers = min(plot_workers, os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_render_and_save_plot, all_plot_tasks))
        print(f"✅ 曲线图已成功保存至: {save_dir}")

    return global_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint 路径 (若提供 exp_dir 则可自动查找)')
    parser.add_argument('--exp_dir', type=str, default=None, help='分层实验目录 (如 ./experiments/.../seed_42)')
    parser.add_argument('--data_dir', type=str, default='/home/users/zsy/mydata/budding_yeast')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1024, help='GPU 推理 Batch Size (A6000 48GB 推荐 1024)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default=None, help='评估输出目录 (默认保存在 exp_dir/eval 或 ./eval_result)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--real_mutant_plots', type=int, default=-1, help='真实突变体最多画多少张图 (-1 表示完整保存全部 126 个真实突变体图片)')
    parser.add_argument('--synth_test_plots', type=int, default=50, help='合成测试集最多画多少张图 (0 表示不画合成测试集)')
    parser.add_argument('--plot_workers', type=int, default=16, help='CPU 多进程画图核数')
    parser.add_argument('--norm_type', type=str, default='log1p', choices=['log1p', 'asinh_z', 'linear', 'asinh_scale'],
                        help='数据归一化类型: log1p (平滑对数归一化, 默认), asinh_z, asinh_scale, linear')
    parser.add_argument('--seed', type=int, default=None, help='随机种子 (默认自动从 ckpt 或 exp_dir 解析，兜底 42)')
    args = parser.parse_args()

    # 路径与 Checkpoint 解析
    if args.exp_dir:
        args.exp_dir = os.path.abspath(args.exp_dir)
        if args.save_dir is None:
            args.save_dir = os.path.join(args.exp_dir, 'eval')
        if args.ckpt is None:
            # 自动在 exp_dir/checkpoints 或 exp_dir 寻找 ckpt
            ckpt_search_dirs = [os.path.join(args.exp_dir, 'checkpoints'), args.exp_dir]
            for d in ckpt_search_dirs:
                if os.path.isdir(d):
                    ckpts = glob.glob(os.path.join(d, '*.ckpt'))
                    if ckpts:
                        # 优先选 final-last 或 best
                        args.ckpt = sorted(ckpts, key=lambda x: (not 'final' in x, not 'best' in x))[0]
                        break

    if args.save_dir is None:
        args.save_dir = './eval_result'

    if not args.ckpt or not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"未找到有效的 Checkpoint 文件: {args.ckpt}")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"🎯 加载 Checkpoint: {args.ckpt}")
    print(f"📁 评估结果输出目录: {args.save_dir}")
    model, hp = load_model_from_checkpoint(args.ckpt, device=args.device)
    model = model.to(args.device)

    # 自动解析 seed: 优先级 args.seed -> checkpoint hyper_parameters -> 路径中的 seed_<N> -> 默认 42
    seed = args.seed
    if seed is None:
        seed = hp.get('seed', None)
    if seed is None and args.ckpt:
        m = re.search(r'seed_(\d+)', args.ckpt)
        if m:
            seed = int(m.group(1))
    if seed is None and args.exp_dir:
        m = re.search(r'seed_(\d+)', args.exp_dir)
        if m:
            seed = int(m.group(1))
    if seed is None:
        seed = 42
    args.seed = seed

    # 自动解析 norm_type: 优先从 checkpoint hyper_parameters 读取 (若可用)
    resolved_norm_type = args.norm_type
    if hp and 'norm_type' in hp:
        resolved_norm_type = hp['norm_type']
        print(f"📋 从 Checkpoint 自动继承 norm_type: {resolved_norm_type}")

    dm = YeastDataModule(
        dataset_path=os.path.join(args.data_dir, args.dataset_name),
        val_batch_size=args.batch_size, num_workers=args.num_workers, use_adj_matrix=False,
        norm_type=resolved_norm_type
    )
    dm.setup('test')

    max_real_plots = None if args.real_mutant_plots < 0 else args.real_mutant_plots
    evaluate_model_on_datamodule(
        model=model,
        dm=dm,
        device=args.device,
        save_dir=args.save_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        max_real_plots=max_real_plots,
        synth_test_plots=args.synth_test_plots,
        plot_workers=args.plot_workers
    )

    # 自动检查并补充生成 5 合 1 loss 曲线图 (若训练日志存在)
    try:
        from budding_yeast_v2.utils.curve_logger import extract_and_plot_from_log
        exp_root = os.path.dirname(os.path.abspath(args.save_dir)) if os.path.basename(os.path.abspath(args.save_dir)) == 'eval' else args.save_dir
        candidate_logs = [
            os.path.join(exp_root, 'train.log'),
            os.path.join(os.path.dirname(exp_root), 'train.log'),
        ]
        for clog in candidate_logs:
            if os.path.exists(clog):
                extract_and_plot_from_log(clog, output_dir=args.save_dir)
                break
    except Exception as e:
        print(f"⚠️ [Loss Curves] 自动生成曲线提示: {e}")

if __name__ == '__main__':
    main()
