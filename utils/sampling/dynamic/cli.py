"""Explicit continuous workflows; never change the fixed-pool five-round default."""
import json
from pathlib import Path
from .callback import DYNAMIC_METHODS,DynamicConfig


def add_parser(subparsers):
    p=subparsers.add_parser('dynamic',help='PEAKS streaming / ERACS SNR-triggered coreset training')
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--pool',choices=('mutation','lhs'),required=True)
    p.add_argument('--method',choices=DYNAMIC_METHODS,required=True)
    p.add_argument('--output',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/experiments/budding_dynamic'))
    p.add_argument('--device',default='cpu');p.add_argument('--config',type=Path)
    p.add_argument('--lhs-config',type=Path);p.add_argument('--plots',action='store_true')
    p.add_argument('--wandb',action='store_true');p.add_argument('--diagnostic-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')


def execute(args):
    settings={} if args.config is None else json.loads(args.config.read_text())
    configuration=DynamicConfig(**settings).validate()
    if args.dry_run:
        print(json.dumps(dict(method=args.method,pool=args.pool,configuration=configuration.__dict__,
            training='continuous / persistent optimizer',normalization='initial training frozen',
            unselected_training_truth_access=True,heldout_truth_for_selection=False,trajectories_copied=False)))
        return
    from .runner import run_dynamic
    print(run_dynamic(args.protocol,args.pool,args.method,args.output,device=args.device,dynamic_config=settings,
        lhs_config=args.lhs_config,diagnostic_only=args.diagnostic_only,plots=args.plots,wandb_enabled=args.wandb))
