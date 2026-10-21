#!/usr/bin/env python3
"""Rank per-mutant errors from explicitly supplied current-model checkpoints."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import pandas as pd
from budding_yeast_v2.utils.checkpoint_evaluation import predict_real_checkpoints
from budding_yeast_v2.utils.metrics import calculate_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',action='append',required=True,help='LABEL=PATH; repeat for multiple models')
    parser.add_argument('--dataset_name',required=True)
    parser.add_argument('--output',type=Path,default=Path('eval_result/real_mutants_badcases.csv'))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch_size',type=int,default=32)
    args=parser.parse_args()
    predictions,target,names=predict_real_checkpoints(args.checkpoint,args.dataset_name,args.device,args.batch_size)
    records=[]
    for i,name in enumerate(names):
        row={'Mutant_Name':name}
        for label,values in predictions.items():
            metrics=calculate_metrics(values[i:i+1],target[i:i+1])
            for key in ('MAE','Relative L2','Correlation'):row[f'{label}_{key}']=metrics[key]
        row['Avg_RelL2']=sum(row[f'{label}_Relative L2'] for label in predictions)/len(predictions)
        records.append(row)
    frame=pd.DataFrame(records).sort_values('Avg_RelL2',ascending=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(args.output,index=False)
    print(frame.head(25).to_string(index=False))


if __name__=='__main__':main()
