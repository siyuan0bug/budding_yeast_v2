#!/usr/bin/env python3
"""Evaluate explicit checkpoints and a fixed equal-weight ensemble on real mutants."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import torch
from budding_yeast_v2.utils.checkpoint_evaluation import predict_real_checkpoints
from budding_yeast_v2.utils.metrics import calculate_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',action='append',required=True,help='LABEL=PATH; repeat for ensemble members')
    parser.add_argument('--dataset_name',required=True)
    parser.add_argument('--output',type=Path,default=Path('eval_result/ensemble_metrics.json'))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch_size',type=int,default=32)
    args=parser.parse_args()
    predictions,target,names=predict_real_checkpoints(args.checkpoint,args.dataset_name,args.device,args.batch_size)
    result={label:calculate_metrics(values,target) for label,values in predictions.items()}
    ensemble=torch.stack(list(predictions.values())).mean(0)
    result['equal_weight_ensemble']=calculate_metrics(ensemble,target)
    result['sample_count']=len(names)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,default=float))
    print(json.dumps(result,indent=2,default=float))


if __name__=='__main__':main()
