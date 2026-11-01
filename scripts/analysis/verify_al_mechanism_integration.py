"""Saved-FNO validation of train-only Pattern and canonical response acquisitions.

No surrogate training or benchmark-target scoring. Optional native oracle checks
only query selected unlabelled synthetic candidates from the existing pool.
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
from budding_yeast_v2.eval import load_model_from_checkpoint
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.data.dataset_archive import open_dataset
from budding_yeast_v2.utils.active_learning import ActiveLearningEngine
from budding_yeast_v2.utils.active_learning.registry import build_strategy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',type=Path,required=True);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--member',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--candidates',type=int,default=2048)
    p.add_argument('--budget',type=int,default=64);p.add_argument('--query',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    model,_=load_model_from_checkpoint(str(a.checkpoint),device='cpu')
    member,_=load_model_from_checkpoint(str(a.member),device='cpu')
    dm=YeastDataModule(str(a.checkpoint.parent/'dataset.json'),num_workers=0,use_adj_matrix=False,
        norm_type='log1p',preprocessing_state=model._preprocessing_state)
    dm.setup('fit')
    base=ActiveLearningEngine(strategy='ours_pattern_support',initial_train_size=10000,al_pool_dataset=str(a.pool))
    base.setup(dm,'cuda')
    # Canonical checkpoint row order differs from the root initial NPZ; storage
    # maps its exact sample names back to original source indices.
    with np.load(a.initial,allow_pickle=False) as z:lookup={str(name):i for i,name in enumerate(z['mutant_names'])}
    base._snapshot_initial_path=str(a.initial)
    base._snapshot_real_rows=np.array([lookup[str(dm._mutant_names[i])] for i in dm._test_idx_real])
    base._snapshot_train_rows=np.array([lookup[str(dm._mutant_names[i])] for i in dm._train_idx_synth])
    base._snapshot_heldout_rows=np.array([lookup[str(dm._mutant_names[i])] for i in list(dm._val_idx_synth)+list(dm._test_idx_synth)])
    ids=base.pool.ensure_ids();keep=np.zeros(len(ids),dtype=bool)
    keep[np.random.RandomState(20261031).choice(len(ids),a.candidates,replace=False)]=True;base.pool.keep(keep)
    base.set_ensemble_members([model,member]);reports=[];engines={}
    for strategy in ('ours_pattern_balance','ours_pattern_context','ours_pattern_support','ours_sensitivity'):
        engine=copy.copy(base);engine.pool=copy.deepcopy(base.pool)
        engine.strategy=strategy;engine.config=replace(base.config,strategy=strategy);engine.acquisition=build_strategy(strategy)
        engine._pending_add_budget=a.budget;engine._snapshot_added_paths=[]
        before=torch.get_rng_state().clone();plan=engine.select(model,dm,'cuda')
        assert len(plan.candidate_ids)==a.budget==len(set(plan.candidate_ids))
        torch.testing.assert_close(before,torch.get_rng_state())
        reports.append(dict(strategy=strategy,selected=a.budget,oracle_called_during_selection=False,
            pool_unchanged=engine._pool_size==a.candidates))
        engines[strategy]=(engine,plan)
        if strategy=='ours_sensitivity':
            audit=engine._experimental_audits[0][0]['feedback'];reports[-1]['physics']=audit
            if a.query:
                batch=engine.query(plan,dm)
                assert batch.data is not None and len(batch.data['names'])>0
                reports[-1]['native_oracle_accepted']=len(batch.data['names'])
    if a.query:
        engine,plan=engines['ours_pattern_support']
        engine._experimental_audits.clear();engine._mechanism_artifacts.clear();engine.fill_acquisition_budget=True
        with tempfile.TemporaryDirectory(prefix='native_mechanism_',dir=a.output) as folder:
            added=engine.select_and_append(model,dm,'cuda',folder);assert added==a.budget
            root=Path(folder);acq=root/'acquisitions/after_round_00'
            assert (acq/'pattern_classifier.joblib').is_file()
            assert json.loads((acq/'pattern_classifier.json').read_text())['training_rows']==10000
            with np.load(acq/'mechanism_predictions.npz') as z:assert z['pattern_probability'].shape[1]==6
            with open_dataset(root/'rounds/round_01/dataset.json') as z:
                assert len(z['split_train'])==10000+a.budget
                assert z['y_init'].dtype==z['params'].dtype==np.float64
                assert z.manifest['fields']['data']['dtype']=='<f4'
                assert all(k in z for k in ('labels','pattern_labels','pattern_group_codes','mutation_pairs_json'))
            reports.append(dict(native_snapshot_append=added,classifier_and_prequery_artifacts_saved=True))
    (a.output/'verification.json').write_text(json.dumps(dict(no_training=True,reports=reports),indent=2))
    print(json.dumps(reports,indent=2),flush=True)


if __name__=='__main__':main()
