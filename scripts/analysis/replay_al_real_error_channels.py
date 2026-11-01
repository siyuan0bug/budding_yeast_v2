"""CPU replay of completed AL checkpoints for physical per-channel diagnostics."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
import torch
from budding_yeast_v2.data.dataset_archive import read_dataset_rows
from budding_yeast_v2.data.normalization import normalize_p, normalize_y, denormalize_y
from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
from budding_yeast_v2.eval import load_model_from_checkpoint
from budding_yeast_v2.utils.metrics import calculate_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--initial',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--strategies',nargs='+',default=None)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads)
    manifest=json.loads(args.manifest.read_text())
    with np.load(args.initial,allow_pickle=False) as archive:
        real=archive['split_real'];params=archive['params'][real];names=archive['mutant_names'][real]
        variables=archive['variable_names']
    target_raw=read_dataset_rows(args.initial,real)
    truth_group=pattern_group_metadata(target_raw)
    target=torch.as_tensor(target_raw,dtype=torch.float32)
    p=torch.as_tensor(params,dtype=torch.float32)
    variable_rows,group_rows,verification=[],[],[];selected_curves={}
    for task in manifest['tasks']:
        run=Path(task['experiment_dir']);strategy=task['strategy']
        if args.strategies and strategy not in args.strategies:continue
        summary=json.loads((run/'summary/rounds_summary.json').read_text())
        if len(summary)!=4:continue
        model,hp=load_model_from_checkpoint(str(run/'rounds/round_03/model.ckpt'),device='cpu')
        model.eval();state=model._preprocessing_state
        y=normalize_y(target,state['mean_y'],state['std_y'],state['norm_type'],state.get('s_c'))
        pp=normalize_p(p,state['p_mean'],state['p_std'],state['norm_type'])
        ic=y[:,:,0];t=torch.linspace(0,1,target.shape[-1])[None,None,:].expand(len(target),39,-1)
        x=torch.stack((ic[:,:,None].expand(-1,-1,target.shape[-1]),t),2)
        with torch.no_grad():
            prediction=torch.cat([model.forward_ic_time(x[i:i+16],pp[i:i+16]).float() for i in range(0,len(x),16)])
            physical=denormalize_y(prediction,state['mean_y'],state['std_y'],state['norm_type'],state.get('s_c'))
            truth=denormalize_y(y,state['mean_y'],state['std_y'],state['norm_type'],state.get('s_c'))
        if not torch.isfinite(physical).all():raise ValueError('Nonfinite real replay')
        metric=calculate_metrics(physical,truth)
        official=json.loads((run/'rounds/round_03/eval/metrics_denorm.json').read_text())['real_mutants']['all_variables']
        verification.append(dict(strategy=strategy,**{'replay_'+k:v for k,v in metric.items()},
            **{'official_'+k:v for k,v in official.items()},
            mae_relative_difference=abs(metric['MAE']/official['MAE']-1),
            mse_relative_difference=abs(metric['MSE']/official['MSE']-1)))
        if abs(metric['MAE']/official['MAE']-1)>.01:
            raise ValueError(f'Replay differs by >1% from saved MAE for {strategy}')
        actual, pred=truth.numpy(),physical.numpy();difference=pred-actual
        mse=np.mean(difference.astype(np.float64)**2,axis=2);mae=np.mean(np.abs(difference),axis=2)
        rel=np.sqrt(np.sum(difference.astype(np.float64)**2,axis=2)/np.maximum(np.sum(actual.astype(np.float64)**2,axis=2),.01*actual.shape[-1]))
        pred_group=pattern_group_metadata(pred)
        for i,name in enumerate(names):
            for j,variable in enumerate(variables):
                variable_rows.append(dict(strategy=strategy,Mutant=str(name),variable=str(variable),
                    MAE=float(mae[i,j]),MSE=float(mse[i,j]),Relative_L2=float(rel[i,j]),
                    truth_peak=float(actual[i,j].max()),predicted_peak=float(pred[i,j].max()),
                    truth_end=float(actual[i,j,-1]),predicted_end=float(pred[i,j,-1])))
            group_rows.append(dict(strategy=strategy,Mutant=str(name),
                truth_group=str(truth_group['pattern_group_codes'][i]),predicted_group=str(pred_group['pattern_group_codes'][i]),
                truth_divisions=int(truth_group['observed_division_counts'][i]),
                predicted_divisions=int(pred_group['observed_division_counts'][i]),
                MAE=float(mae[i].mean()),MSE=float(mse[i].mean()),Relative_L2=float(rel[i].mean())))
        chosen=[i for i,n in enumerate(names) if str(n).split('_',1)[0] in ('15','124')]
        selected_curves[strategy]=pred[chosen]
        print(strategy,metric,'replay verified',flush=True)
        del model,prediction,physical,x,y
    frame=pd.DataFrame(variable_rows);frame.to_csv(args.output/'per_mutant_variable_errors.csv',index=False)
    pd.DataFrame(group_rows).to_csv(args.output/'predicted_window_groups.csv',index=False)
    pd.DataFrame(verification).to_csv(args.output/'replay_verification.csv',index=False)
    total=frame.groupby(['strategy','variable'],as_index=False)[['MAE','MSE']].mean()
    total['global_mse_fraction']=total.MSE/total.groupby('strategy').MSE.transform('sum')
    total.to_csv(args.output/'global_variable_error_contributions.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids=[i for i,n in enumerate(names) if str(n).split('_',1)[0] in ('15','124')]
    columns=[('CLN2',1),('CLB2',2),('SIC1',4),('ORI',33)]
    fig,axes=plt.subplots(2,4,figsize=(17,7),constrained_layout=True)
    for row,i in enumerate(ids):
        for col,(label,index) in enumerate(columns):
            ax=axes[row,col];ax.plot(np.linspace(0,210,500),target_raw[i,index],color='black',lw=2,label='ODE')
            for method in (args.strategies or ('al4pde_random','al4pde_max_dist','ours_interaction','ours_response_feedback')):
                if method in selected_curves:
                    ax.plot(np.linspace(0,210,500),selected_curves[method][row,index],lw=1,label=method)
            ax.set(title=str(names[i])+' / '+label,xlabel='Time (min)');ax.grid(alpha=.2)
    axes[0,-1].legend(fontsize=7)
    fig.savefig(args.output/'mutants15_124_comparison.png',dpi=180);plt.close(fig)
    print(args.output,flush=True)


if __name__=='__main__':main()
