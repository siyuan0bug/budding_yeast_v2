"""Per-epoch monitoring is exact, batched, read-only and leaves training unchanged."""
from dataclasses import replace
from pathlib import Path
import json,random
import numpy as np
import pytest
import torch
from torch import nn
from test_sampling_benchmark import make_pool,config
from test_shared_training_cleanup import selected_fixture
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.utils.sampling.contracts import PoolInputs
from budding_yeast_v2.utils.training.monitoring import evaluate_cohort
from budding_yeast_v2.utils.evaluation.metrics import relative_l2_loss
from budding_yeast_v2.utils.sampling.benchmark.rounds import fit_round


class Dummy(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(1.1))
    @property
    def device(self):return self.weight.device
    def forward_ic_time(self,x,p):
        assert not self.training and not torch.is_grad_enabled()
        torch.rand(2);np.random.rand();random.random()
        return x[:,:,0,:]*self.weight
    def denormalize_y(self,y):return y


@pytest.mark.parametrize('variables',[16,39])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_batch_pooling_matches_full_metric_and_preserves_rng_and_mode(variables,dtype):
    param_dim=58 if variables==16 else 141
    p=np.zeros((5,param_dim));initial=np.full((5,variables),.5,dtype=np.float32)
    initial[::2,0]=.001;initial[:,2]=np.arange(5)*.3
    y=np.broadcast_to(initial[:,:,None],(5,variables,13)).astype(dtype);y[0,3,2]=-.234567891234
    inputs=PoolInputs(np.arange(5),p,initial)
    stats=NormalizationStats(torch.zeros(1,variables,1),torch.ones(1,variables,1),torch.zeros(1,param_dim),torch.ones(1,param_dim),'linear')
    model=Dummy().train()
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    before=(torch.get_rng_state(),np.random.get_state(),random.getstate())
    expected=torch.as_tensor(initial)[:,:,None].expand(-1,-1,13)*model.weight.detach()
    mse=float((expected.double()-stats.targets(y).double()).square().mean())
    rel=float(relative_l2_loss(expected.double(),torch.tensor(y).double()))
    for batch_size in (1,2,3,5):
        result=evaluate_cohort(model,stats,inputs,y,batch_size=batch_size)
        assert result['mse_norm']==pytest.approx(mse,abs=1e-14)
        assert result['rel_l2_physical']==pytest.approx(rel,abs=1e-14)
        assert model.training
        assert torch.equal(before[0],torch.get_rng_state())
        assert np.array_equal(before[1][1],np.random.get_state()[1])
        assert before[1][2:]==np.random.get_state()[2:] and before[2]==random.getstate()
        assert model.weight.grad is None


def test_monitoring_adds_all_test_fields_without_changing_updates_or_best_checkpoint(tmp_path):
    _,values,snapshot,training,validation=selected_fixture(tmp_path)
    cfg=replace(config(),epochs=(3,3,3,3),batch_size=2)
    cohorts={name:(PoolInputs(np.arange(start,start+3),values['params'][start:start+3],values['y_init'][start:start+3]),
                    values['data'][start:start+3]) for name,start in (('synth',12),('real',20))}
    plain,stats,baseline=fit_round(cfg,0,training,validation,tmp_path/'plain',77,dataset_path=snapshot)
    monitored,stats2,history=fit_round(cfg,0,training,validation,tmp_path/'monitored',77,
                                    dataset_path=snapshot,monitoring=cohorts)
    for key,value in plain.state_dict().items():
        torch.testing.assert_close(value,monitored.state_dict()[key],rtol=0,atol=0)
    for before,after in zip(baseline,history):
        assert before=={key:after[key] for key in before}
        assert all(np.isfinite(after[key]) for key in ('test_loss_synth','test_loss_real','test_real_denorm_rel_l2'))
    torch.testing.assert_close(stats.p_mean,stats2.p_mean,rtol=0,atol=0)
    summary=json.loads((tmp_path/'monitored/_state/training.json').read_text())
    assert summary['validation_metric']=='val_loss_synth'
    assert summary['best_validation_error']==min(r['val_error'] for r in history)


def test_benchmark_wires_only_own_pool_test_and_real_into_primary_history(tmp_path):
    from budding_yeast_v2.utils.sampling.benchmark.protocol import prepare_protocol
    from budding_yeast_v2.utils.sampling.benchmark.runner import run_experiment
    source,values=make_pool(tmp_path/'pool')
    cfg=replace(config(),budgets=(2,4),epochs=(2,2),warmup=(0,0))
    protocol=prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',cfg)
    reference=tmp_path/'real.npz'
    np.savez(reference,**{key:values[key][20:23] for key in ('params','y_init','data','mutant_names')})
    run=run_experiment(protocol,'mutation','random',tmp_path/'runs',real_reference=reference)
    from budding_yeast_v2.utils.training.artifacts import read_history
    for i in range(2):
        rows=read_history(run/f'rounds/round_{i:02d}/history.csv')
        assert all('test_loss_synth' in row and 'test_loss_real' in row and 'test_real_denorm_rel_l2' in row for row in rows)
        best=min(rows,key=lambda row:row['val_error'])
        evaluation=json.loads((run/f'rounds/round_{i:02d}/eval/global_metrics_denorm.json').read_text())
        assert best['test_real_denorm_rel_l2']==pytest.approx(evaluation['real_mutants']['all_39_vars']['Relative L2'],rel=2e-6)
        assert not any('cross' in key for row in rows for key in row)


def test_pure_relative_monitor_reports_training_objective_and_keeps_mse_separate():
    from budding_yeast_v2.utils.training.losses import RelativeL2Loss
    inputs=PoolInputs(np.arange(3),np.zeros((3,141)),np.full((3,39),.5,dtype=np.float32))
    y=np.full((3,39,12),.5,dtype=np.float32)
    stats=NormalizationStats(torch.zeros(1,39,1),torch.ones(1,39,1),torch.zeros(1,141),torch.ones(1,141),'linear')
    model=Dummy()
    result=evaluate_cohort(model,stats,inputs,y,batch_size=2,criterion=RelativeL2Loss())
    prediction=torch.full_like(torch.tensor(y),.5)*model.weight.detach()
    assert result['loss_norm']==pytest.approx(float(RelativeL2Loss()(prediction,torch.tensor(y))),rel=1e-6)
    assert result['mse_norm']==pytest.approx(float((prediction-torch.tensor(y)).square().mean()),rel=1e-6)
    assert result['loss_norm']!=pytest.approx(result['mse_norm'])
