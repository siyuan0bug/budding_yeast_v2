"""Validate new acquisitions with saved FNOs, native inputs and a small ODE query.

No training, real-target scoring or modification of the shared dataset/pool.
"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.data.dataset_archive import open_dataset
from budding_yeast_v2.eval import load_model_from_checkpoint
from budding_yeast_v2.utils.active_learning import ActiveLearningEngine
from budding_yeast_v2.utils.active_learning.registry import build_strategy
from budding_yeast_v2.utils.active_learning.strategies.hybrid import VARIANTS
from budding_yeast_v2.utils.active_learning.strategies.hybrid_feedback import calibrated_multiplier


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',type=Path,required=True);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--primary',type=Path,required=True);p.add_argument('--auxiliary',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--candidates',type=int,default=4096)
    p.add_argument('--budget',type=int,default=256);p.add_argument('--query',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    primary,_=load_model_from_checkpoint(str(a.primary),device='cpu')
    auxiliary,_=load_model_from_checkpoint(str(a.auxiliary),device='cpu')
    dm=YeastDataModule(str(a.primary.parent/'dataset.json'),num_workers=0,use_adj_matrix=False,norm_type='log1p',
                       preprocessing_state=primary._preprocessing_state)
    dm.setup('fit')
    base=ActiveLearningEngine(strategy='ours_global_calibrated',initial_train_size=10000,al_pool_dataset=str(a.pool))
    base.setup(dm,a.device)
    # The checkpoint uses the canonical round ordering. Reference the same
    # physical initial NPZ through an exact name/index permutation for test I/O.
    with np.load(a.initial,allow_pickle=False) as archive:
        lookup={str(name):i for i,name in enumerate(archive['mutant_names'])}
    base._snapshot_initial_path=str(a.initial)
    base._snapshot_real_rows=np.array([lookup[str(dm._mutant_names[i])] for i in dm._test_idx_real])
    base._snapshot_train_rows=np.array([lookup[str(dm._mutant_names[i])] for i in dm._train_idx_synth])
    base._snapshot_heldout_rows=np.array([lookup[str(dm._mutant_names[i])]
        for i in list(dm._val_idx_synth)+list(dm._test_idx_synth)])
    ids=base.pool.ensure_ids();chosen=set()
    with np.load(a.geometry,allow_pickle=False) as z:
        for row,known in zip(z['candidate_ids'][:64],z['known_training'][:64]):
            chosen.update(int(cid) for cid,is_known in zip(row,known) if not is_known)
    rng=np.random.RandomState(20261030)
    for cid in rng.permutation(ids):
        if len(chosen)>=a.candidates:break
        chosen.add(int(cid))
    base.pool.keep(np.isin(ids,list(chosen)))
    base.set_ensemble_members([primary,auxiliary])
    reports=[]
    for name in VARIANTS:
        engine=copy.copy(base);engine.pool=copy.deepcopy(base.pool)
        engine.strategy=name;engine.config=replace(base.config,strategy=name);engine.acquisition=build_strategy(name)
        engine._snapshot_added_paths=[];engine._pending_add_budget=a.budget
        plan=engine.select(primary,dm,a.device)
        assert len(plan.candidate_ids)==a.budget==len(set(plan.candidate_ids))
        trace=engine._experimental_audits[0][0]
        kinds={}
        for row in trace['choices']:kinds[row['kind']]=kinds.get(row['kind'],0)+row['new_queries']
        reports.append(dict(strategy=name,selected=len(plan.candidate_ids),budget_by_kind=kinds,
                            pool_unchanged=engine._pool_size==base._pool_size))
        print(name,kinds,flush=True)
    if a.query:
        engine._experimental_audits.clear()
        engine.fill_acquisition_budget=True
        with tempfile.TemporaryDirectory(prefix='native_hybrid_',dir=a.output) as directory:
            added=engine.select_and_append(primary,dm,a.device,directory)
            assert added==a.budget
            path=Path(directory)/'rounds/round_01/dataset.json'
            with open_dataset(path) as archive:
                assert len(archive['split_train'])==10000+a.budget
                assert archive['y_init'].dtype==archive['params'].dtype==np.dtype('float64')
                assert archive['data'].dtype==np.dtype('float32')
                assert all(k in archive for k in ('mutation_pairs_json','levels','pattern_group_codes','labels'))
            replay=YeastDataModule(str(path),num_workers=0,use_adj_matrix=False,norm_type='log1p')
            replay.setup('fit');engine.refresh_model_inputs(replay)
            assert engine._pool_P.dtype==engine._pool_Y_ic.dtype==torch.float32
            edge_path=Path(directory)/'acquisitions/after_round_00/edge_feedback.npz'
            with np.load(edge_path) as archive:
                assert archive['features'].shape[1]==17
                multiplier,fit=calibrated_multiplier(engine,archive['features'],42)
                assert np.isfinite(multiplier).all() and (multiplier>=1).all() and (multiplier<=2).all()
            reports.append(dict(native_query=added,round_manifest_replay=True,edge_calibration=fit))
    (a.output/'verification.json').write_text(json.dumps(dict(no_training=True,reports=reports),indent=2))
    print(a.output/'verification.json',flush=True)


if __name__=='__main__':main()
