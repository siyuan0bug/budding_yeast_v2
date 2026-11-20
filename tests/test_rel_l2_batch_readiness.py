"""Pure relative-L2 overnight configuration, frozen splits and all chosen baselines."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import pytest
import torch
from test_sampling_benchmark import make_pool,config
from budding_yeast_v2.utils.training.losses import RelativeL2Loss
from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig,prepare_protocol,derive_training_protocol

METHODS=('random','topk','power','max_dist_pred','coreset_ll','lcmd_ll','bait_ll','vessal_penultimate')


def test_existing_relative_loss_formula_and_default():
    assert BenchmarkConfig().loss=='rel_l2'
    torch.manual_seed(42)
    p=torch.randn(3,39,12,requires_grad=True);y=torch.randn_like(p)
    criterion=RelativeL2Loss()
    assert criterion.max_clamp is None
    expected=(torch.linalg.vector_norm(p-y,dim=-1)/(y.square().sum(-1)+.01**2).sqrt()).mean()
    torch.testing.assert_close(criterion(p,y),expected,rtol=0,atol=0)
    criterion(p,y).backward();assert torch.isfinite(p.grad).all()


def test_training_only_protocol_reuses_exact_same_immutable_artifacts(tmp_path):
    source,_=make_pool(tmp_path/'pool')
    previous_cfg=replace(config(),loss='hybrid_mse_rel')
    previous=prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',previous_cfg)
    cfg=replace(previous_cfg,loss='rel_l2')
    with patch('budding_yeast_v2.data.dataset_archive.TrajectoryStore.inputs',side_effect=AssertionError('decoded pool inputs')):
        current=derive_training_protocol(previous,tmp_path/'protocol',cfg)
    assert current!=previous
    assert prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',cfg)==current
    before=json.loads((previous/'protocol.json').read_text());after=json.loads((current/'protocol.json').read_text())
    assert after['config']['loss']=='rel_l2' and before['artifacts']==after['artifacts']
    for name in before['artifacts']:
        assert (previous/name).stat().st_ino==(current/name).stat().st_ino
    with pytest.raises(ValueError,match='Data/split setting changed: seed'):
        derive_training_protocol(previous,tmp_path/'protocol',replace(cfg,seed=123))


@pytest.mark.parametrize('method',METHODS)
def test_selected_baselines_complete_five_rounds_with_relative_loss(tmp_path,method):
    from budding_yeast_v2.utils.sampling.benchmark.runner import run_experiment
    from budding_yeast_v2.utils.training.artifacts import read_history
    source,values=make_pool(tmp_path/'pool')
    cfg=replace(config(),loss='rel_l2',budgets=(2,4,6,8,10),epochs=(1,1,1,1,2),warmup=(0,)*5)
    protocol=prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',cfg)
    reference=tmp_path/'real_fixture.npz'
    np.savez(reference,**{key:values[key][20:23] for key in ('params','y_init','data','mutant_names')})
    with patch('budding_yeast_v2.data.systems.budding.simulation.simulate_mutant',side_effect=AssertionError('offline called ODE')):
        run=run_experiment(protocol,'mutation',method,tmp_path/'runs',diagnostic_only=True,real_reference=reference)
    state=json.loads((run/'_state/state.json').read_text())
    assert state['completion']['unique_training_inputs']==10
    assert state['completion']['new_ode_calls']==0
    assert len(list(run.glob('rounds/round_*/model.ckpt')))==5
    assert not list(run.rglob('*.npz'))
    for i in range(5):
        rd=run/f'rounds/round_{i:02d}'
        ckpt=torch.load(rd/'model.ckpt',map_location='cpu',weights_only=False)
        assert ckpt['hyper_parameters']['loss_type']=='rel_l2'
        assert ckpt['training_objective_definition']['eps']==cfg.relative_l2_eps
        history=read_history(rd/'history.csv')
        assert all(np.isfinite(row['train_loss']) and row['train_loss']>=0 for row in history)
        assert all(np.isfinite(row[key]) for row in history
                   for key in ('test_loss_synth','test_loss_real','test_real_denorm_rel_l2','val_sample_channel_rel_l2'))
        assert all(row['loss_type']=='rel_l2' for row in history)
        assert all(np.isfinite(row[key]) and row[key]>=0 for row in history for key in ('val_loss_synth','test_loss_synth','test_loss_real'))
        assert all('val_mse_norm' in row and 'test_real_mse_norm' in row for row in history)
