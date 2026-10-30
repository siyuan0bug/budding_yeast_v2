"""Replay saved Budding FNO acquisition paths on real pool inputs, without training.

Validates the post-refit float64-provenance / float32-model boundary. A small
pool slice tests integration; it is not a million-candidate memory/time benchmark.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from budding_yeast_v2.eval import load_model_from_checkpoint
from budding_yeast_v2.data.normalization import normalize_p, normalize_y
from budding_yeast_v2.utils.active_learning import ActiveLearningEngine
from budding_yeast_v2.utils.active_learning.registry import strategy_spec

STRATEGIES = ('al4pde_random','al4pde_uncertainty','al4pde_power',
              'al4pde_max_dist','al4pde_lcmd','al4pde_bait','ours_response')


def audit(args):
    if not 0 < args.query_budget <= args.pool_samples or args.train_samples < 1:
        raise ValueError('Invalid sample/query counts')
    model,_ = load_model_from_checkpoint(args.checkpoint,device='cpu')
    state = model._preprocessing_state
    if state is None:
        raise ValueError('Use a checkpoint with an explicit preprocessing contract')
    member,_ = load_model_from_checkpoint(args.ensemble_checkpoint,device='cpu')
    for key in ('p_mean','p_std','mean_y','std_y'):
        torch.testing.assert_close(state[key],member._preprocessing_state[key],rtol=0,atol=0)
    model.to(args.device)
    print('Loading real pool slice (cached inputs, no ODE)',flush=True)
    with np.load(args.pool,allow_pickle=False) as saved:
        count=len(saved['params'])
        ids=np.sort(np.random.RandomState(42).choice(count,args.pool_samples,replace=False))
        params=torch.tensor(saved['params'][ids],dtype=torch.float32)
        initial=torch.tensor(saved['y_init'][ids],dtype=torch.float64)
        overrides=[json.loads(str(x)) for x in saved['overrides_json'][ids]]
        metadata=[json.loads(str(x)) for x in saved['candidate_metadata_json'][ids]]
        names=saved['mutant_names'][ids]
        grid=torch.tensor(saved['time_grid'],dtype=torch.float32)
    with np.load(args.round_dataset,allow_pickle=False) as saved:
        train=saved['split_train'][:args.train_samples]
        train_p=torch.tensor(saved['params'][train],dtype=torch.float32)
        train_ic=torch.tensor(saved['y_init'][train],dtype=torch.float32)
        train_ic64=saved['y_init'][train]
        train_ovr=saved['overrides_json'][train]
        train_meta=saved['candidate_metadata_json'][train]
        train_ids=saved['candidate_ids'][train]
    dm=SimpleNamespace(system_id='budding',norm_type=state['norm_type'],s_c=state.get('s_c'),
                       t_max=float(grid[-1]))
    for key in ('p_mean','p_std','mean_y','std_y'):setattr(dm,key,state[key].clone())
    normalized_ic=normalize_y(train_ic,dm.mean_y,dm.std_y,dm.norm_type,dm.s_c)
    normalized_p=normalize_p(train_p,dm.p_mean,dm.p_std,dm.norm_type)
    train_x=torch.stack((normalized_ic[:,:,None].expand(-1,-1,len(grid)),
                         torch.linspace(0,1,len(grid))[None,None,:].expand(len(train),39,-1)),2)
    dm.train_dataset_synth=TensorDataset(train_x,normalized_p,
                                        torch.full((len(train),39,len(grid)),float('nan')))
    # Reproduce the reported failure using the old exact round-loop expression.
    old_ic=normalize_y(initial[:1],dm.mean_y,dm.std_y,dm.norm_type,dm.s_c)
    old_x=torch.stack((old_ic[:,:,None].expand(-1,-1,len(grid)),
                      torch.linspace(0,1,len(grid))[None,None,:].expand(1,39,-1)),2)
    reproduced=False
    try:
        with torch.no_grad():
            model.forward_ic_time(old_x.to(args.device),normalized_p[:1].to(args.device))
    except RuntimeError as exc:
        reproduced='double' in str(exc) and 'float' in str(exc)
        if not reproduced:raise
    assert reproduced,'Expected the original dtype failure to reproduce'
    rows=[]
    for strategy in args.strategies:
        engine=ActiveLearningEngine(strategy=strategy,initial_train_size=args.query_budget)
        engine._pool_size=len(ids);engine._pool_P_real=params.clone()
        engine._pool_initial_physical=initial.clone();engine._pool_ovr=overrides
        engine._pool_mass_meta=metadata;engine._pool_mutant_names=names
        engine._al4pde_t_grid=torch.linspace(0,1,len(grid)).reshape(1,1,-1)
        engine.pool.candidate_ids=ids.copy()
        engine.refresh_model_inputs(dm)
        torch.testing.assert_close(engine._pool_initial_physical,initial,rtol=0,atol=0)
        x=engine._get_pool_X_batch(np.arange(len(ids)))
        assert x.dtype==engine._pool_P.dtype==torch.float32
        with torch.no_grad():
            y=model.forward_ic_time(x.to(args.device),engine._pool_P.to(args.device))
            assert torch.isfinite(y).all()
        if strategy_spec(strategy).requires_ensemble:engine.set_ensemble_members((model,member))
        if strategy=='ours_response':
            engine._cum_synth_params=list(train_p.numpy())
            engine._cum_synth_provenance=[dict(y_init=y0,overrides_json=str(ovr),candidate_metadata_json=str(meta),candidate_id=int(cid))
                for y0,ovr,meta,cid in zip(train_ic64,train_ovr,train_meta,train_ids)]
        with patch.object(engine,'_build_broadcast_from_pool',side_effect=AssertionError('Unexpected ODE query')):
            plan=engine.select(model,dm,args.device)
        assert len(plan.candidate_ids)==args.query_budget
        assert len(engine.pool.resolve(plan))==args.query_budget
        rows.append(dict(strategy=strategy,status='passed',selected=list(plan.candidate_ids),
                         input_dtype=str(x.dtype),physical_initial_dtype=str(initial.dtype)))
        print(f'{strategy}: passed',flush=True)
    return dict(status='passed',checkpoint=str(Path(args.checkpoint).resolve()),
                ensemble_checkpoint=str(Path(args.ensemble_checkpoint).resolve()),
                pool=str(Path(args.pool).resolve()),pool_samples=len(ids),train_anchors=len(train),
                original_failure_reproduced=reproduced,strategies=rows,device=args.device,
                scope='saved FNO prediction and selection only; no training/ODE; small slice, not a full-pool benchmark')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pool','checkpoint','ensemble-checkpoint','round-dataset','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--pool-samples',type=int,default=128)
    parser.add_argument('--train-samples',type=int,default=16)
    parser.add_argument('--query-budget',type=int,default=4)
    parser.add_argument('--strategies',nargs='+',choices=STRATEGIES,default=list(STRATEGIES))
    args=parser.parse_args()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    try:report=audit(args)
    except Exception as exc:
        output.write_text(json.dumps(dict(status='failed',error=repr(exc)),indent=2)+'\n')
        raise
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
