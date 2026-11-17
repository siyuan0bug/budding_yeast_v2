"""Physical prediction reports and checked trajectory plots, independent of CLI."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
from budding_yeast_v2.data.dataset_archive import atomic_json
from .metrics import summarize, per_sample_metrics, RELATIVE_L2_METRIC_VERSION

def _render_and_save_plot(task):
    """独立的多进程画图函数，充分利用多核 CPU"""
    import os
    import multiprocessing
    # Spawned plotting workers are CPU-only, including on exclusive GPUs.
    if multiprocessing.current_process().name != 'MainProcess':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        pred, target, title, save_path, rows, cols, t_max = task[:7]
        names=task[7] if len(task)>7 else VARIABLE_NAMES
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten()
        t = np.linspace(0, t_max, pred.shape[-1])
        for i in range(pred.shape[0]):
            ax = axes[i]
            ax.plot(t, target[i], 'b-', label='Truth', linewidth=1.5)
            ax.plot(t, pred[i], 'r--', label='Pred', linewidth=1.5)
            ax.set_xlabel('Time (min)', fontsize=8)
            ax.set_ylabel('Concentration', fontsize=8)
            var_name = names[i] if i < len(names) else f"Var {i}"
            ax.set_title(f'{i}: {var_name}', fontweight='bold', fontsize=9)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        for i in range(pred.shape[0], len(axes)):
            axes[i].set_visible(False)
        plt.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig(save_path, dpi=120, bbox_inches='tight', metadata={'Description':title})
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ 画图失败 ({save_path}): {e}")
        try:
            plt.close('all')
        except Exception:
            pass


def render_prediction_plots(tasks, workers=4):
    """One checked renderer for static and sampling evaluations."""
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    if not tasks:
        return
    with ProcessPoolExecutor(max_workers=min(workers,len(tasks)),mp_context=mp.get_context('spawn')) as executor:
        list(executor.map(_render_and_save_plot,tasks))
    missing=[task[3] for task in tasks if not Path(task[3]).is_file()]
    if missing:
        raise RuntimeError(f'Missing prediction figures: {missing[:3]}')


def export_sample_metrics(predictions,targets,names,path,*,source=None,pool_rows=None,
                          candidate_ids=None,metadata=None):
    """Stable sample IDs, physical errors and available source/trajectory labels."""
    from .labels import trajectory_labels
    labels=trajectory_labels(targets)
    if len(names)!=len(targets) or len(predictions)!=len(targets):
        raise ValueError('Sample report arrays are not aligned')
    rows=[]
    for i,(p,y,name,label) in enumerate(zip(predictions,targets,names,labels)):
        row=dict(mutant_name=str(name),metric_version=RELATIVE_L2_METRIC_VERSION)
        if source is not None:row['source']=source
        if pool_rows is not None:row['pool_row']=int(pool_rows[i])
        if candidate_ids is not None:row['source_candidate_id']=int(candidate_ids[i])
        row.update(label)
        if metadata:
            for key,values in metadata.items():
                value=values[i]
                row[key]=value.item() if hasattr(value,'item') else value
        row.update(per_sample_metrics(p,y));rows.append(row)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def organized_metrics(summary,*,source=None):
    """Compact serialization only: preserve shared metric computations unchanged."""
    result=dict(count=summary['count'])
    if source is not None:result['source']=source
    for key in ('metric_version','primary_metric','metric_space','sample_channel_rel_l2'):
        if key in summary:result[key]=summary[key]
    for key in ('all_39_vars','all_16_vars','core_9_vars'):
        if key in summary:result[key]=summary[key]
    patterns=('six_group_confusion','division_count_mae')
    result['trajectory_metrics']={k:v for k,v in summary.items()
        if k not in ('count','metric_version','primary_metric','metric_space','sample_channel_rel_l2',
                     'all_variables','all_39_vars','all_16_vars','core_9_vars',*patterns)}
    if any(k in summary for k in patterns):
        result['pattern_metrics']={k:summary[k] for k in patterns if k in summary}
    return result


def ordered_global_metrics(value):
    """Keep metadata first, then real, same-source and cross-pool tests."""
    priority=('round','train_pool','train_samples','real_mutants','synth_test','cross_pool_test')
    return {key:value[key] for key in priority if key in value} | {key:item for key,item in value.items() if key not in priority}


def save_global_metrics(output,metrics,*,pool_name,round_index,training_rows):
    value=dict(round=round_index,train_pool=pool_name,train_samples=training_rows)
    for key,summary in metrics.items():
        if key=='real_mutants':value[key]=organized_metrics(summary)
        else:
            source=key.removesuffix('_test')
            value['synth_test' if source==pool_name else 'cross_pool_test']=organized_metrics(summary,source=source)
    value=ordered_global_metrics(value)
    atomic_json(Path(output)/'global_metrics_denorm.json',value)
    return value


def save_real_evaluation(predictions,targets,names,output,*,plots=False,plot_workers=4,
                         context=None,write_summary=True):
    import textwrap
    from .labels import get_real_mutations_str
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not len(names):raise ValueError('Empty real reference')
    summary=summarize(predictions,targets)
    rows=export_sample_metrics(predictions,targets,names,output/'per_mutant_metrics.csv')
    if write_summary:
        atomic_json(output/'global_metrics_denorm.json',dict(real_mutants=organized_metrics(summary)))
    if plots:
        from budding_yeast_v2.data.systems.registry import resolve_system
        system=resolve_system('fission' if targets.shape[1]==16 else 'budding')
        cols=4 if targets.shape[1]==16 else 8
        nrows=(targets.shape[1]+cols-1)//cols
        images=output/'real_mutants';images.mkdir(exist_ok=True)
        context=context or {}
        description=' | '.join(f'{k}: {v}' for k,v in context.items())
        tasks=[]
        for p,y,name,row in zip(predictions,targets,names,rows):
            label=' | '.join(str(row[k]) for k in ('pattern_label','pattern_group') if k in row)
            title=f"{name} | {label} | {description}\n"
            title+=f"Rel-L2={row['rel_l2']:.5f} (sample/channel, physical) | MAE={row['MAE']:.5f} | MSE={row['MSE']:.5f} | Corr={row['correlation']:.5f} | All {targets.shape[1]} variables\n"
            title+='\n'.join(textwrap.wrap('Mutations: '+get_real_mutations_str(name,system.system_id),width=160))
            tasks.append((p,y,title,str(images/(str(name).replace('/','_')+'.png')),
                          nrows,cols,system.horizon,system.variable_names))
        render_prediction_plots(tasks,plot_workers)
    return summary
