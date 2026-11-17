"""Arguments/dispatch for declared generative adaptations, no implicit training."""
import json
from pathlib import Path

from .policies import METHODS
from .online import ONLINE_METHODS,OnlineConfig


def add_parser(subparsers):
    p=subparsers.add_parser('generate',help='New legal ODE queries; round-based or asynchronous online adapters')
    p.add_argument('--protocol',type=Path,required=True,help='Frozen shared splits, initial set and real reference')
    p.add_argument('--pool',choices=('mutation','lhs'),required=True)
    p.add_argument('--method',choices=METHODS+ONLINE_METHODS,required=True)
    p.add_argument('--output',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/experiments/budding_generation'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--bank-size',type=int,default=512)
    p.add_argument('--query-batch-size',type=int,help='Default OED=1, OGAS/random=32; >1 is a declared batch adaptation')
    p.add_argument('--lhs-config',type=Path,help='Reviewed frozen mixed-marginal LHS configuration')
    p.add_argument('--oed-config',type=Path,help='Algorithm-only JSON settings; no biological overrides')
    p.add_argument('--ogas-config',type=Path,help='Algorithm-only JSON settings; no biological overrides')
    p.add_argument('--hggs-config',type=Path,help='GF/MGS settings only; no biological overrides')
    p.add_argument('--online-config',type=Path,help='Online-only epochs/replay/update frequency JSON, no biological overrides')
    p.add_argument('--max-attempts-mult',type=int,default=30)
    p.add_argument('--plots',action='store_true')
    p.add_argument('--wandb',action='store_true')
    p.add_argument('--diagnostic-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')


def execute(args):
    online=args.method in ONLINE_METHODS
    if online and args.query_batch_size is not None:
        raise ValueError('Online queue is bounded by --workers, not --query-batch-size')
    if not online and args.online_config is not None:
        raise ValueError('--online-config requires an online method')
    if online and args.oed_config is not None:raise ValueError('OED settings do not apply to online OGAS/random')
    if args.hggs_config is not None and args.method!='hggs_projected':raise ValueError('HGGS settings require hggs_projected')
    batch=args.query_batch_size
    if batch is None:batch=args.workers if online else 1 if args.method=='oed_temporal' else 32
    if min(args.workers,args.bank_size,batch,args.max_attempts_mult)<1 or batch>args.bank_size:
        raise ValueError('Invalid positive generation limits or bank smaller than query batch')
    if args.dry_run:
        if online:
            OnlineConfig(**({} if args.online_config is None else json.loads(args.online_config.read_text()))).validate()
        print(json.dumps(dict(method=args.method,pool=args.pool,protocol=str(args.protocol),workers=args.workers,
            legal_bank_size=args.bank_size,query_batch_size=batch,
            source_rules='unchanged canonical proposal/initialization/ODE',
            native_online_ogas=False,asynchronous_solver_training=online,
            training_schedule='continuous surrogate, frozen initial normalization' if online else 'shared cold-start rounds'),ensure_ascii=False))
        return
    if online:
        from .online_runner import run_online_generation
        configuration={} if args.online_config is None else json.loads(args.online_config.read_text())
        configuration.setdefault('max_attempts_mult',args.max_attempts_mult)
        path=run_online_generation(args.protocol,args.pool,args.method,args.output,device=args.device,
            workers=args.workers,bank_size=args.bank_size,lhs_config=args.lhs_config,
            ogas_config=None if args.ogas_config is None else json.loads(args.ogas_config.read_text()),
            online_config=configuration,plots=args.plots,wandb_enabled=args.wandb,
            diagnostic_only=args.diagnostic_only)
        print(path);return
    from .runner import run_generation
    path=run_generation(args.protocol,args.pool,args.method,args.output,device=args.device,workers=args.workers,
        bank_size=args.bank_size,query_batch_size=batch,lhs_config=args.lhs_config,
        oed_config=None if args.oed_config is None else json.loads(args.oed_config.read_text()),
        ogas_config=None if args.ogas_config is None else json.loads(args.ogas_config.read_text()),
        hggs_config=None if args.hggs_config is None else json.loads(args.hggs_config.read_text()),
        max_attempts_mult=args.max_attempts_mult,plots=args.plots,wandb_enabled=args.wandb,
        diagnostic_only=args.diagnostic_only)
    print(path)
