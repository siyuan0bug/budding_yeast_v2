"""One summary per AL run; round curves retain the existing plot style."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_summary(root, records, histories, final_metrics):
    output=Path(root)/'summary';output.mkdir(parents=True,exist_ok=True)
    with (output/'al_performance_curve.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (output/'rounds_summary.json').write_text(json.dumps(records,indent=2))
    (output/'global_metrics_denorm.json').write_text(json.dumps(final_metrics,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,key,title in zip(axes,('mae','rel_l2','corr'),('MAE','Relative L2','Correlation')):
        for split in ('real_mutants','synth_test'):
            ax.plot([r['train_samples'] for r in records],
                    [np.nan if r.get(split+'_'+key) is None else r[split+'_'+key] for r in records],
                    marker='o',label=split)
        ax.set(xlabel='Synthetic training samples',ylabel=title);ax.grid(alpha=.25);ax.legend()
    fig.savefig(output/'al_performance_curve.png',dpi=200);plt.close(fig)
    # Independent panels: normalization is refitted, so normalized losses do not
    # share a physical scale across rounds. Physical RelL2 remains comparable.
    fig,axes=plt.subplots(len(histories),1,figsize=(11,7*len(histories)),squeeze=False)
    for i,history in enumerate(histories):
        history.plot(ax=axes[i,0],title=f'Round {i:02d} | {records[i]["train_samples"]} synthetic training samples',
            subtitle='Normalization fitted on this round: compare physical Rel-L2 across rounds')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(output/f'all_rounds_loss.{suffix}',dpi=200)
    plt.close(fig)
