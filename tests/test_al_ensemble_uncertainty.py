"""Independent trained ensemble acquisition, official variance and top-K semantics."""
import os
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from budding_yeast_v2.utils.al_callback import ActiveLearningCallback, train_acquisition_ensemble
from budding_yeast_v2.scripts.analysis.audit_al4pde_alignment import function_from_source

ROOT = Path(__file__).resolve().parents[1]


class Member(torch.nn.Module):
    def __init__(self, slope=1., offset=0.):
        super().__init__()
        self.slope=torch.nn.Parameter(torch.tensor(slope))
        self.offset=torch.nn.Parameter(torch.tensor(offset))
        self.seen=[]
    def forward_ic_time(self,x,p):
        self.seen.append((x.clone(),p.clone()))
        return (p[:,:1,None]*self.slope+self.offset).expand(-1,39,8)
    def forward(self,x,grid,p):
        return self.forward_ic_time(x,p)


def callback(members, mode='loss_mse'):
    cb=ActiveLearningCallback(strategy='al4pde_uncertainty',initial_train_size=2,ensemble_uncertainty_mode=mode)
    cb._pool_size=4
    cb._pool_P=torch.arange(4.).reshape(4,1).expand(-1,141).clone()
    cb._get_pool_X_batch=lambda indices:torch.ones(len(indices),39,2,8)
    cb.set_ensemble_members(members)
    return cb


@pytest.mark.parametrize('batch_size',[1,3,128])
@pytest.mark.parametrize('mode',['loss_mse','sample_variance'])
def test_scores_and_order_match_official_source(batch_size,mode):
    members=[Member(1.,0.),Member(3.,1.)]
    cb=callback(members,mode)
    official=function_from_source(ROOT.parent/'al4pde/al4pde/prob_models/ensemble.py','uncertainty','Ensemble')
    expected=official(SimpleNamespace(base_models=members),torch.ones(4,39,2,8),None,cb._pool_P)
    if mode=='loss_mse':
        rollout=function_from_source(ROOT.parent/'al4pde/al4pde/prob_models/ensemble.py','unc_independent_rollout','Ensemble')
        outputs=torch.stack([m.forward_ic_time(torch.ones(4,39,2,8),cb._pool_P) for m in members])
        state=SimpleNamespace(_roll_out_all=lambda *args:outputs,adapt_unc_to_loss=True,
            loss=lambda pred,target,**kwargs:(pred-target)**2,first_model_prediction=True)
        _,expected=rollout(state,None,None,8)
    scores=cb._compute_ensemble_uncertainty('cpu',np.arange(4),batch_size=batch_size)
    torch.testing.assert_close(scores,expected.reshape(4,-1).mean(1))
    cb._build_broadcast_from_pool=lambda selected,dm:selected
    selection=cb._al4pde_uncertainty_sample(None,members[0],None,'cpu',0)
    top_k=function_from_source(ROOT.parent/'al4pde/al4pde/acquisition/uncertainty_based.py','top_k')
    np.testing.assert_array_equal(selection,top_k(scores,2).numpy())


def test_no_noise_identical_inputs_and_preserved_modes_and_rng():
    members=[Member(),Member()]
    members[0].train();members[1].eval()
    cb=callback(members)
    state=torch.get_rng_state().clone()
    with patch('torch.randn_like',side_effect=AssertionError('input noise forbidden')):
        scores=cb._compute_ensemble_uncertainty('cpu',np.arange(4))
    assert torch.equal(state,torch.get_rng_state())
    assert scores.tolist()==[0.]*4
    assert members[0].training and not members[1].training
    for member in members:
        torch.testing.assert_close(member.seen[-1][1],cb._pool_P)
    torch.testing.assert_close(members[0].seen[-1][0],members[1].seen[-1][0])


def test_missing_or_reused_members_fail_instead_of_silent_noise_fallback():
    cb=ActiveLearningCallback(strategy='al4pde_uncertainty')
    with pytest.raises(RuntimeError,match='trained ensemble'):
        cb._compute_ensemble_uncertainty('cpu',np.arange(1))
    member=Member()
    for members in ([member],[member,member]):
        with pytest.raises(ValueError,match='distinct'):
            cb.set_ensemble_members(members)


def test_prediction_error_restores_member_training_modes():
    class Broken(Member):
        def forward_ic_time(self,x,p):
            return super().forward_ic_time(x,p)*float('nan')
    members=[Member(),Broken()]
    members[1].eval()
    cb=callback(members)
    with pytest.raises(ValueError,match='Nonfinite'):
        cb._compute_ensemble_uncertainty('cpu',np.arange(4))
    assert members[0].training and not members[1].training


class TinyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer=torch.nn.Linear(2,2)
    def forward_ic_time(self,x,p):
        return self.layer(p).unsqueeze(-1).expand(-1,2,4)
    def training_step(self,batch,batch_idx):
        p,y=batch
        return torch.nn.functional.mse_loss(self.layer(p),y)
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),lr=.01)


class TinyData(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(4,2),torch.zeros(4,2)),batch_size=2)


def test_two_members_really_train_save_and_restore_random_streams(tmp_path):
    def trainer_factory(index):
        return pl.Trainer(accelerator='cpu',devices=1,max_epochs=1,logger=False,
            enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
            limit_val_batches=0,num_sanity_val_steps=0)
    pl.seed_everything(42,workers=True)
    primary=TinyModel()
    initial=primary.layer.weight.detach().clone()
    primary_trainer=trainer_factory(0)
    primary_trainer.fit(primary,datamodule=TinyData())
    assert not torch.equal(initial,primary.layer.weight)
    primary_state={k:v.clone() for k,v in primary.state_dict().items()}
    torch_state=torch.get_rng_state().clone()
    np_state=np.random.get_state(); py_state=random.getstate()
    env_state={k:os.environ.get(k) for k in ('PL_GLOBAL_SEED','PL_SEED_WORKERS')}
    members,records=train_acquisition_ensemble(primary,TinyModel,trainer_factory,TinyData(),
        n_members=2,seed=42,round_index=0,checkpoint_dir=tmp_path)
    assert len(members)==2 and members[0] is primary
    assert records[1]['seed']!=42
    assert Path(records[1]['checkpoint']).exists()
    assert not torch.equal(members[0].layer.weight,members[1].layer.weight)
    for key,value in primary.state_dict().items():torch.testing.assert_close(value,primary_state[key])
    assert torch.equal(torch_state,torch.get_rng_state())
    assert np_state[0]==np.random.get_state()[0]
    np.testing.assert_array_equal(np_state[1],np.random.get_state()[1])
    assert py_state==random.getstate()
    assert env_state=={k:os.environ.get(k) for k in env_state}
    checkpoint=torch.load(records[1]['checkpoint'],map_location='cpu',weights_only=False)
    for key,value in members[1].state_dict().items():torch.testing.assert_close(value,checkpoint['state_dict'][key])
