"""Cross-entry regression for shared training, statistics, checkpoint and metrics."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from test_sampling_benchmark import make_pool,config
from budding_yeast_v2.data.dataset_archive import TrajectoryStore
from budding_yeast_v2.data.normalization import NormalizationStats,NORM_TYPES
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.utils.sampling.contracts import PoolInputs,QueriedTraining
from budding_yeast_v2.utils.sampling.benchmark.protocol import prepare_protocol
from budding_yeast_v2.utils.sampling.benchmark.runner import run_experiment
from budding_yeast_v2.utils.evaluation.checkpoints import load_model_from_checkpoint,Predictor
from budding_yeast_v2.utils.evaluation.metrics import summarize,calculate_metrics
from budding_yeast_v2.utils.sampling.benchmark.rounds import fit_round


def selected_fixture(tmp_path):
    source,values=make_pool(tmp_path/'pool')
    ids=np.arange(4)
    snapshot=tmp_path/'selected.json'
    TrajectoryStore(source).snapshot(snapshot,ids,provenance={'seed':42})
    training=QueriedTraining(PoolInputs(ids,values['params'][ids],values['y_init'][ids]),values['data'][ids])
    validation=[(PoolInputs(np.arange(8,10),values['params'][8:10],values['y_init'][8:10]),values['data'][8:10])]
    return source,values,snapshot,training,validation


@pytest.mark.parametrize('norm_type',NORM_TYPES)
def test_data_module_and_acquisition_fit_one_statistics_implementation(tmp_path,norm_type):
    _,values,snapshot,training,_=selected_fixture(tmp_path)
    dm=YeastDataModule(dataset_path=str(snapshot),norm_type=norm_type,num_workers=0,use_adj_matrix=False)
    dm.setup('fit')
    stats=NormalizationStats.fit(training.inputs.params,training.trajectories,norm_type)
    for name in ('mean_y','std_y','p_mean','p_std','s_c'):
        left,right=getattr(stats,name),getattr(dm,name)
        if left is None:assert right is None
        else:torch.testing.assert_close(left,right,rtol=0,atol=0)
    x,p=stats.encode(training.inputs.params,training.inputs.initial,500)
    torch.testing.assert_close(x,dm.train_dataset_synth.tensors[0],rtol=0,atol=0)
    torch.testing.assert_close(p,dm.train_dataset_synth.tensors[1],rtol=0,atol=0)


def test_shared_summary_preserves_legacy_metrics_and_fission_has_no_budding_labels():
    rng=np.random.default_rng(42)
    y=torch.tensor(rng.uniform(0,2,(3,16,25)),dtype=torch.float32)
    p=y*.8
    result=summarize(p,y,'fission')
    assert result['all_variables']==calculate_metrics(p,y)
    assert result['all_16_vars']==result['all_variables']
    assert 'six_group_confusion' not in result and 'core_9_vars' not in result
    assert result['macro_trajectory_rel_l2']==pytest.approx(.2,rel=1e-6)


def test_common_training_supports_deeponet_and_standard_eval_checkpoint(tmp_path):
    source,values=make_pool(tmp_path/'pool')
    cfg=replace(config(),model='deeponet',width=8)
    protocol=prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',cfg)
    run=run_experiment(protocol,'mutation','random',tmp_path/'runs',diagnostic_only=True)
    checkpoint=run/'rounds/round_03/model.ckpt'
    model,hp=load_model_from_checkpoint(checkpoint)
    assert hp['model_name']=='deeponet' and model._trainer is None
    assert model._preprocessing_state['fit_scope']=='train_synthetic'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert 'pytorch-lightning_version' in saved and 'state_dict' in saved
    reference=tmp_path/'reference.npz'
    np.savez(reference,**{k:values[k][:2] for k in ('params','y_init','data','mutant_names')})
    out=tmp_path/'standalone_eval'
    args=[sys.executable,'eval.py','--ckpt',str(checkpoint),'--real_reference',str(reference),
          '--device','cpu','--real_mutant_plots','0','--save_dir',str(out)]
    completed=subprocess.run(args,capture_output=True,text=True,cwd=Path(__file__).resolve().parents[1])
    assert completed.returncode==0,completed.stdout+completed.stderr
    assert json.loads((out/'global_metrics_denorm.json').read_text())['real_mutants']['count']==2


def test_round_resume_restores_optimizer_scheduler_and_rng_exactly(tmp_path,monkeypatch):
    _,_,snapshot,training,validation=selected_fixture(tmp_path)
    cfg=replace(config(),epochs=(6,6,6,6),batch_size=2)
    captured={};original_unlink=Path.unlink
    def remove(path,*args,**kwargs):
        if path.name=='resume.ckpt' and path.exists():
            captured[str(path)]=torch.load(path,map_location='cpu',weights_only=False)
        return original_unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',remove)
    baseline=tmp_path/'baseline'
    fit_round(cfg,0,training,validation,baseline,77,dataset_path=snapshot)
    import budding_yeast_v2.utils.training.runner as entry
    real_factory=entry.make_trainer
    class CrashAfterSavedEpoch(pl.Callback):
        def on_train_epoch_start(self,trainer,pl_module):
            if trainer.current_epoch==5:raise RuntimeError('intentional interruption')
    def interrupted_factory(**kwargs):
        kwargs['callbacks']=[*kwargs['callbacks'],CrashAfterSavedEpoch()]
        return real_factory(**kwargs)
    interrupted=tmp_path/'interrupted'
    with patch.object(entry,'make_trainer',side_effect=interrupted_factory):
        with pytest.raises(RuntimeError,match='intentional interruption'):
            fit_round(cfg,0,training,validation,interrupted,77,dataset_path=snapshot)
    assert (interrupted/'_state/resume.ckpt').exists()
    fit_round(cfg,0,training,validation,interrupted,77,dataset_path=snapshot)
    before=torch.load(baseline/'model.ckpt',map_location='cpu',weights_only=False)
    after=torch.load(interrupted/'model.ckpt',map_location='cpu',weights_only=False)
    assert json.loads((baseline/'_state/training.json').read_text())['optimizer_steps']==12
    assert json.loads((interrupted/'_state/training.json').read_text())['optimizer_steps']==12
    assert not (interrupted/'_state/resume.ckpt').exists()
    assert 'optimizer_states' not in after
    assert 'preprocessing_contract' in after
    from budding_yeast_v2.utils.training.artifacts import read_history
    assert read_history(baseline/'history.csv')==read_history(interrupted/'history.csv')
    full_before=captured[str(baseline/'_state/resume.ckpt')]
    full_after=captured[str(interrupted/'_state/resume.ckpt')]
    assert full_before['global_step']==full_after['global_step']==12
    for key in ('state_dict','optimizer_states','lr_schedulers'):
        torch.testing.assert_close(full_before[key],full_after[key],rtol=0,atol=0)
    # A completed rerun loads without fitting, even though resume was removed.
    with patch.object(entry,'make_trainer',side_effect=AssertionError('completed round retrained')):
        fit_round(cfg,0,training,validation,interrupted,77,dataset_path=snapshot)
    for key in before['state_dict']:
        torch.testing.assert_close(before['state_dict'][key],after['state_dict'][key],rtol=0,atol=0)


def test_new_workflow_can_use_shared_training_with_a_different_validation_metric(tmp_path):
    _,values,snapshot,training,validation=selected_fixture(tmp_path)
    dm=YeastDataModule(dataset_path=str(snapshot),num_workers=0,use_adj_matrix=False,batch_size_synth=4)
    dm.setup('fit');dm.drop_last_train=False
    dm.set_validation_data(validation);dm.validation_metric='normalized_mse'
    from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
    from budding_yeast_v2.utils.training.runner import fit_model_round
    model,history=fit_model_round(lambda:YeastLitModule(model_name='fno',width=4,modes=2,n_blocks=1,
        max_epochs=2,warmup_epochs=0,loss_type='hybrid_mse_rel'),dm,tmp_path/'alternative_flow',77,
        epochs=2,identity={'purpose':'independent_workflow'},monitor='val_loss_synth')
    assert len(history)==2 and all(row['validation_metric']=='val_loss_synth' for row in history)
    assert model._trainer is None
    assert json.loads((tmp_path/'alternative_flow/_state/training.json').read_text())['validation_metric']=='val_loss_synth'


def test_new_selection_rule_requires_no_workflow_or_trainer_changes(monkeypatch):
    import types
    from budding_yeast_v2.utils.sampling.registry import METHODS,Method,select
    from budding_yeast_v2.utils.sampling.contracts import SamplingContext,SamplingResult
    module_name='budding_yeast_v2.utils.sampling.ours._test_rule'
    module=types.ModuleType(module_name)
    module.select=lambda context,budget:SamplingResult(context.pool.ids[-budget:])
    monkeypatch.setitem(sys.modules,module_name,module)
    monkeypatch.setitem(METHODS,'_test_rule',Method('_test_rule',family='ours',implementation='_test_rule:select'))
    inputs=PoolInputs(np.arange(6),np.ones((6,141)),np.ones((6,39)))
    train=QueriedTraining(PoolInputs(np.array([],dtype=int),np.empty((0,141)),np.empty((0,39))),np.empty((0,39,500)))
    result=select('_test_rule',SamplingContext(inputs,train,None,42,'cpu'),2)
    assert result.ids.tolist()==[4,5]


def test_round_schedule_changes_only_configuration(tmp_path):
    source,_=make_pool(tmp_path/'pool')
    cfg=replace(config(),budgets=(2,6),epochs=(1,2),warmup=(0,0))
    protocol=prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',cfg)
    run=run_experiment(protocol,'mutation','random',tmp_path/'runs',diagnostic_only=True)
    import csv
    records=list(csv.DictReader((run/'performance.csv').open()))
    assert sorted({int(row['training_rows']) for row in records})==[2,6]
    assert (run/'rounds/round_01/model.ckpt').exists()
