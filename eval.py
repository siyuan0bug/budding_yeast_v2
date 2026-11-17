import os
import sys

# Ensure repository root is in sys.path regardless of where eval.py is launched from
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_current_dir)
for _p in [_repo_root, _current_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pathlib import Path
from budding_yeast_v2.utils.evaluation.checkpoints import load_model_from_checkpoint
from budding_yeast_v2.utils.evaluation.metrics import summarize, per_sample_metrics, RELATIVE_L2_METRIC_VERSION
from budding_yeast_v2.utils.evaluation.reporting import _render_and_save_plot, render_prediction_plots, save_real_evaluation
from budding_yeast_v2.data.dataset_archive import atomic_json

import argparse
import json
import csv
import glob
import re
import torch
from torch.utils.data import DataLoader
import numpy as np

from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.data.systems.budding.api import get_default_params, mutant_rules
from budding_yeast_v2.utils.evaluation.metrics import calculate_metrics
from budding_yeast_v2.data.preprocessing_contract import resolve_checkpoint_norm, resolve_checkpoint_fit_scope

CORE_9_INDICES = [0, 1, 2, 3, 4, 20, 22, 33, 35]

from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES

# ======== 突变参数信息提取 ========
_DEFAULTS_DICT = get_default_params()
_DEFAULTS_DICT.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
PARAM_KEYS = sorted([k for k in _DEFAULTS_DICT.keys() if k != 'is_mutant_104'])
_DEFAULT_PARAM_VEC = np.array([_DEFAULTS_DICT[k] for k in PARAM_KEYS])

from budding_yeast_v2.utils.evaluation.labels import get_real_mutations_str

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



_metric_summary = summarize  # Historical analysis imports use this public name.


def evaluate_dataset(model, dm, dataset, indices, device, mean_y, std_y, seed, save_dir, 
                     sub_dir, batch_size=512, max_plots=None, max_workers=16):
    """
    高度优化的通用评估函数：
    1. 使用 GPU 大 Batch 并行推理，将千级样本前向时间压至 < 0.2 秒；
    2. 向量化计算 Per-mutant 指标；
    3. 使用多进程进程池并行绘制 matplotlib 图片。
    """
    plot_dir = os.path.join(save_dir, sub_dir)
    if max_plots is None or max_plots > 0:
        os.makedirs(plot_dir, exist_ok=True)
    
    system_id=getattr(dm,'system_id','budding')
    if len(dataset)==0:
        empty=torch.empty(0,dm.num_vars,dm.T_STEPS)
        return _metric_summary(empty,empty,system_id),[],[empty],[empty],[]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=getattr(dm,'num_workers',4), pin_memory=device.type=='cuda')
    if len(indices) != len(dataset):
        raise ValueError('Evaluation row IDs and dataset are not aligned')
    
    all_preds_denorm_list = []
    all_targets_denorm_list = []
    all_p_real_list = []
    offset = 0

    # 1. GPU 大 Batch 高速推理
    with torch.no_grad():
        for batch in loader:
            x, p, y = [b.to(device, non_blocking=True) for b in batch[:3]]
            pred = model.forward_ic_time(x, p)
            
            pred_denorm = dm.denormalize_y(pred, device)
            source_rows=np.asarray(indices[offset:offset+len(x)],dtype=np.int64)
            # Use exact saved physical truth, not an encode/decode round trip
            # that loses finite negative values and introduces float32 rounding.
            target_denorm = torch.as_tensor(dm._raw_data[source_rows])
            p_real = torch.as_tensor(dm._param_conds[source_rows])
            
            all_preds_denorm_list.append(pred_denorm.cpu())
            all_targets_denorm_list.append(target_denorm.cpu())
            all_p_real_list.append(p_real.cpu())
            offset += len(x)

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
        
        metrics = per_sample_metrics(pred_i, target_i)
        csv_rows.append({
            'Mutant': m_name,
            'Pattern': p_label,
            'Relative_L2_Version': RELATIVE_L2_METRIC_VERSION,
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'Relative_L2': metrics['rel_l2'],
            'Correlation': metrics['correlation']
        })

        if max_plots is None or i < max_plots:
            m_str = f"Rel L2 (sample/channel): {metrics['rel_l2']:.4f} | MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4e} | Corr: {metrics['correlation']:.4f}"
            title_base = f"[{p_label}] {m_name} (Seed: {seed})\n{m_str}"
            
            if system_id=='budding' and ('_LHS_' in m_name or '_SYNTH_' in m_name):
                mother_name, perturb_str = get_synth_perturbation_str(m_name, p_real_i)
                title_base += f"\nMother: {mother_name} | Perturbations (mother_val, ratio):\n{perturb_str}"
            elif global_idx not in set(dm._test_idx_real):
                title_base += f"\nMASS synthetic sample (mutant-aware stratified sampling)"
            else:
                mut_str = get_real_mutations_str(m_name,system_id)
                title_base += f"\nMutations: {mut_str}"

            save_plot_path = os.path.join(plot_dir, f"{m_name}.png")
            cols=4 if system_id=='fission' else 8
            rows=(pred_i.shape[0]+cols-1)//cols
            names=getattr(dm,'variable_names',VARIABLE_NAMES)
            plot_tasks.append((pred_i.numpy(), target_i.numpy(), f"{title_base}\nAll {pred_i.shape[0]} Variables", save_plot_path, rows, cols, dm.t_max, names))

    # 3. 全局汇总指标
    summary = _metric_summary(all_preds,all_targets,system_id)

    return summary, csv_rows, [all_preds], [all_targets], plot_tasks

def write_supplementary_label_metrics(rows, indices, dm, path):
    """Join observed labels to metrics; preserve the historical metric CSV contract."""
    from budding_yeast_v2.data.label_metadata import UNKNOWN_BIOLOGY_LABEL
    fields = ['Mutant', 'Pattern', 'Relative_L2_Version', 'MAE', 'MSE', 'Relative_L2', 'Correlation',
              'Biological_Label', 'Biological_Label_Available', 'Pattern_Group',
              'Pattern_Group_Label', 'Pattern_Group_Version', 'Observed_Divisions']
    if len(rows) != len(indices):
        raise ValueError('Evaluation labels and metric rows do not agree')
    biology = getattr(dm, 'biological_labels', None)
    codes = getattr(dm, 'pattern_group_codes', None)
    labels = getattr(dm, 'pattern_group_labels', None)
    divisions = getattr(dm, 'observed_division_counts', None)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, idx in zip(rows, indices):
            if str(row['Mutant']) != str(dm._mutant_names[idx]):
                raise ValueError('Evaluation mutant order disagrees with label indices')
            bio = str(biology[idx]) if biology is not None else UNKNOWN_BIOLOGY_LABEL
            writer.writerow(dict(row, Biological_Label=bio,
                Biological_Label_Available=bio != UNKNOWN_BIOLOGY_LABEL,
                Pattern_Group=str(codes[idx]) if codes is not None else '',
                Pattern_Group_Label=str(labels[idx]) if labels is not None else '',
                Pattern_Group_Version=getattr(dm, 'pattern_group_version', None) or '',
                Observed_Divisions=int(divisions[idx]) if divisions is not None else ''))


def evaluate_model_on_datamodule(model, dm, device=None, save_dir=None, seed=42, batch_size=512, 
                                 max_plots=None, max_real_plots=None, synth_test_plots=50, plot_workers=16,
                                 compact=False):
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
    # Standalone evaluation has no Trainer.setup hook. Bind the same parameter
    # transform as training before causal modules compare inputs with physical WT.
    if hasattr(model, 'bind_datamodule_stats'):
        model.bind_datamodule_stats(dm)
    elif hasattr(model, 'set_parameter_stats'):
        model.set_parameter_stats(dm.p_mean, dm.p_std, norm_type=dm.norm_type)

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

    all_test_summary = _metric_summary(all_preds,all_targets,getattr(dm,'system_id','budding'))

    global_metrics = {
        'system_id':getattr(dm,'system_id','budding'),
        'variable_names':list(getattr(dm,'variable_names',VARIABLE_NAMES)),
        "real_mutants": real_summary,
        "synth_test": synth_summary,
        "all_test": all_test_summary,
    }

    if compact:
        global_metrics['per_mutant'] = real_csv
        global_metrics['per_synth_test'] = synth_csv
    json_path = os.path.join(save_dir, 'metrics_denorm.json' if compact else 'global_metrics_denorm.json')
    with open(json_path, 'w') as f:
        json.dump(global_metrics, f, indent=2)
    if not compact and hasattr(model, 'graph_report'):
        with open(os.path.join(save_dir, 'graph_provenance.json'), 'w') as f:
            json.dump(model.graph_report(), f, indent=2)
    print(f"📊 Global metrics JSON saved to: {json_path}")
    for name,summary in (('all_test',all_test_summary),('real_mutants',real_summary),('synth_test',synth_summary)):
        metrics=summary.get('all_variables',summary.get('all_39_vars'))
        detail=f"MAE={metrics['MAE']:.4f}, Corr={metrics['Correlation']:.4f}" if metrics else 'empty split'
        print(f"   {name}: n={summary['count']}, {detail}")

    # ======== 4. 输出 CSV 记录 (立即落盘) ========
    def write_csv(rows, path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Mutant', 'Pattern', 'Relative_L2_Version', 'MAE', 'MSE', 'Relative_L2', 'Correlation']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"📝 CSV saved to: {path}")

    if not compact:
        write_csv(real_csv, os.path.join(save_dir, 'per_mutant_metrics.csv'))
        write_csv(synth_csv, os.path.join(save_dir, 'per_synth_test_metrics.csv'))
        write_supplementary_label_metrics(real_csv, dm._test_idx_real, dm,
            os.path.join(save_dir, 'per_mutant_label_groups.csv'))
        write_supplementary_label_metrics(synth_csv, dm._test_idx_synth, dm,
            os.path.join(save_dir, 'per_synth_test_label_groups.csv'))

    # ======== 5. 多进程并发画图 (若有请求) ========
    all_plot_tasks = real_plot_tasks + synth_plot_tasks
    if all_plot_tasks:
        print(f"🎨 正在多进程并发绘制曲线图 (共 {len(all_plot_tasks)} 幅)...")
        workers = min(plot_workers, os.cpu_count() or 4)
        render_prediction_plots(all_plot_tasks,workers)
        print(f"✅ 曲线图已成功保存至: {save_dir}")

    return global_metrics



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint 路径 (若提供 exp_dir 则可自动查找)')
    parser.add_argument('--real_reference', help='Evaluate a frozen physical reference using checkpoint statistics, without refitting')
    parser.add_argument('--exp_dir', type=str, default=None, help='分层实验目录 (如 ./experiments/.../seed_42)')
    parser.add_argument('--data_dir', type=str, default='/home/users/zsy/mydata/budding_yeast')
    parser.add_argument('--dataset_name', type=str, default=None,help='AL rounds infer dataset.json or legacy dataset.npz next to the checkpoint')
    parser.add_argument('--batch_size', type=int, default=1024, help='GPU 推理 Batch Size (A6000 48GB 推荐 1024)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='评估输出目录 (默认 eval_graph_<policy>，与历史 eval 分开)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--graph_load_policy', choices=['current', 'checkpoint'], default='current',
                        help='current: migrate fixed variable graph; checkpoint: reproduce saved graph')
    parser.add_argument('--graph_view', choices=['continuous', 'hybrid'], default=None)
    parser.add_argument('--adj_matrix_path', default=None, help='Explicit target/source graph override')
    parser.add_argument('--real_mutant_plots', type=int, default=-1, help='真实突变体最多画多少张图 (-1 表示完整保存全部 126 个真实突变体图片)')
    parser.add_argument('--synth_test_plots', type=int, default=50, help='合成测试集最多画多少张图 (0 表示不画合成测试集)')
    parser.add_argument('--plot_workers', type=int, default=16, help='CPU 多进程画图核数')
    parser.add_argument('--norm_type', type=str, default=None, choices=['log1p', 'asinh_z', 'linear', 'asinh_scale'],
                        help='数据归一化类型: log1p (平滑对数归一化, 默认), asinh_z, asinh_scale, linear')
    parser.add_argument('--seed', type=int, default=None, help='随机种子 (默认自动从 ckpt 或 exp_dir 解析，兜底 42)')
    args = parser.parse_args()

    # 路径与 Checkpoint 解析
    if args.exp_dir:
        args.exp_dir = os.path.abspath(args.exp_dir)
        if args.save_dir is None:
            args.save_dir = os.path.join(args.exp_dir, f'eval_graph_{args.graph_load_policy}')
        if args.ckpt is None:
            round_models=sorted(glob.glob(os.path.join(args.exp_dir,'rounds','round_*','model.ckpt'))+
                                glob.glob(os.path.join(args.exp_dir,'rounds','round_*','primary','best.ckpt')))
            if round_models:args.ckpt=round_models[-1]
            # 自动在 exp_dir/checkpoints 或 exp_dir 寻找 ckpt
            ckpt_search_dirs = [os.path.join(args.exp_dir,'primary'),os.path.join(args.exp_dir, 'checkpoints'), args.exp_dir]
            for d in ckpt_search_dirs:
                if args.ckpt is not None:break
                if os.path.isdir(d):
                    ckpts = glob.glob(os.path.join(d, '*.ckpt'))
                    if ckpts:
                        # 优先选 final-last 或 best
                        args.ckpt = sorted(ckpts, key=lambda x: (not 'final' in x, not 'best' in x))[0]
                        break

    if args.save_dir is None:
        args.save_dir = f'./eval_result_graph_{args.graph_load_policy}'

    if not args.ckpt or not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"未找到有效的 Checkpoint 文件: {args.ckpt}")
    if args.dataset_name is None and args.real_reference is None:
        directory=os.path.dirname(os.path.abspath(args.ckpt))
        candidates=[os.path.join(root,name) for root in (directory,os.path.dirname(directory))
                    for name in ('dataset.json','dataset.npz')]
        args.dataset_name=next((path for path in candidates if os.path.isfile(path)),None)
        if args.dataset_name is None:raise ValueError('Provide --dataset_name for a checkpoint without an adjacent AL dataset.json/dataset.npz')

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"🎯 加载 Checkpoint: {args.ckpt}")
    print(f"📁 评估结果输出目录: {args.save_dir}")
    model, hp = load_model_from_checkpoint(
        args.ckpt, device=args.device, graph_load_policy=args.graph_load_policy,
        graph_view=args.graph_view, adj_matrix_path=args.adj_matrix_path,
        norm_type=args.norm_type,
    )
    print(f'Graph loading policy: {args.graph_load_policy}; provenance saved with metrics.')
    model = model.to(args.device)
    if args.real_reference is not None:
        from budding_yeast_v2.data.normalization import NormalizationStats
        from budding_yeast_v2.utils.evaluation.checkpoints import Predictor
        from budding_yeast_v2.utils.sampling.contracts import PoolInputs
        if not model._stats_bound:
            raise ValueError('Reference evaluation requires recorded checkpoint statistics')
        with np.load(args.real_reference,allow_pickle=False) as archive:
            truth=archive['data'];names=archive['mutant_names']
            inputs=PoolInputs(np.arange(len(truth)),archive['params'],archive['y_init'])
        predictor=Predictor([model],NormalizationStats.from_datamodule(model),truth.shape[-1],args.device,args.batch_size)
        save_real_evaluation(predictor.physical(inputs),truth,names,args.save_dir,
                             plots=args.real_mutant_plots!=0,plot_workers=args.plot_workers,
                             context=dict(model=hp.get('model_name','unknown'),
                                          seed=args.seed if args.seed is not None else hp.get('data_seed',42)))
        return

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
        norm_type=resolved_norm_type, seed=hp.get('data_seed', 42),
        preprocessing_state=model._preprocessing_state,
        fit_scope=hp['normalization_fit_scope'],
    )
    dm.setup('test')
    if dm.system_id!=model.system_id:
        raise ValueError('Checkpoint biological system does not match the evaluation dataset')

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
        from budding_yeast_v2.utils.experiments.curves import extract_and_plot_from_log
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
