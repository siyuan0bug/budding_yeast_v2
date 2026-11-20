"""Schedule compatibility, shared Budding budgets, and observable loss trends."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest
import torch
import pytorch_lightning as pl

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.utils.sampling.convergence import LossChangeMonitor,loss_rates


def model(**kwargs):
    return YeastLitModule(model_name='fno',width=4,modes=2,n_blocks=1,
        loss_type='hybrid_mse_rel',lr=1e-3,**kwargs)


def test_legacy_schedule_is_identical_through_original_training_horizon():
    m=model(max_epochs=100,warmup_epochs=8)
    lam=m.configure_optimizers()['lr_scheduler']['scheduler'].lr_lambdas[0]
    for epoch in range(101):
        expected=epoch/8 if epoch<8 else .001+.5*(1-.001)*(1+math.cos(math.pi*((epoch-8)/92)))
        assert lam(epoch)==expected
    del m.hparams['min_lr'] # Legacy checkpoint hparams have no explicit floor.
    assert m.configure_optimizers()['lr_scheduler']['scheduler'].lr_lambdas[0](100)==.001


def test_new_schedule_has_longer_useful_phase_and_no_rebound_after_horizon():
    m=model(max_epochs=200,warmup_epochs=10,min_lr=1e-5)
    lam=m.configure_optimizers()['lr_scheduler']['scheduler'].lr_lambdas[0]
    assert lam(0)==0 and lam(10)==1
    assert lam(100)*m.lr>5e-4
    assert lam(200)*m.lr==1e-5 and lam(250)*m.lr==1e-5
    values=[lam(t)*m.lr for t in range(10,251)]
    assert np.all(np.diff(values)<=0) and min(values)>=1e-5
    assert m.hparams.min_lr==1e-5


@pytest.mark.parametrize('floor',[-1.,float('nan'),float('inf'),.01])
def test_invalid_floor_rejected_before_training(floor):
    with pytest.raises(ValueError,match='min_lr'):model(min_lr=floor).configure_optimizers()


def test_loss_change_rates_match_known_exponential_and_handle_zeros_nan():
    values=np.exp(-.002*np.arange(11))
    rate=loss_rates(values);expected=100*(1-np.exp(-.002))
    assert rate['net_pct']==pytest.approx(expected)
    assert rate['trend_pct']==pytest.approx(expected)
    assert loss_rates(np.zeros(11))['trend_pct']==0
    assert loss_rates([1.,0.]) is None and loss_rates([1.,float('nan')]) is None


def fake_trainer(module):
    return SimpleNamespace(current_epoch=0,global_step=0,max_epochs=40,callback_metrics={},
        optimizers=[torch.optim.SGD(module.parameters(),lr=.001)],logger=None,should_stop=False)


def test_monitor_labels_plateau_but_never_stops_and_logs_no_test_labels(tmp_path):
    module=model(min_lr=1e-5);trainer=fake_trainer(module);monitor=LossChangeMonitor(tmp_path)
    for epoch in range(40):
        trainer.current_epoch=epoch
        trainer.callback_metrics=dict(train_loss=torch.tensor(1.),val_loss_synth=torch.tensor(2.),
            test_loss_real=torch.tensor(float('nan')),test_loss_synth=torch.tensor(float('nan')))
        monitor.on_train_epoch_end(trainer,module)
    monitor.on_train_end(trainer,module)
    report=json.loads((tmp_path/'convergence.json').read_text())
    assert report['status']=='plateau_candidate' and report['consecutive_checks']==3
    assert report['epochs_observed']==40 and report['early_stopping'] is False
    assert trainer.should_stop is False
    assert report['final']['train_net_pct']==0
    assert 'test_loss_real' not in (tmp_path/'convergence.csv').read_text()


def test_loss_still_falling_or_missing_validation_cannot_be_declared_plateau(tmp_path):
    module=model();trainer=fake_trainer(module);monitor=LossChangeMonitor(tmp_path)
    for epoch in range(40):
        trainer.current_epoch=epoch
        trainer.callback_metrics=dict(train_loss=torch.tensor(math.exp(-.01*epoch)),val_loss_synth=torch.tensor(1.))
        monitor.on_train_epoch_end(trainer,module)
    monitor.on_train_end(trainer,module)
    assert json.loads((tmp_path/'convergence.json').read_text())['status']=='still_changing'
    trainer.current_epoch=40;trainer.callback_metrics=dict(train_loss=torch.tensor(1.))
    monitor.on_train_epoch_end(trainer,module);monitor.on_train_end(trainer,module)
    report=json.loads((tmp_path/'convergence.json').read_text())
    assert report['status']=='invalid_or_missing_loss' and report['final']['val_mse'] is None


def test_real_lightning_monitor_reads_current_epoch_loss_and_saves_summary(tmp_path,capsys):
    from torch.utils.data import DataLoader,TensorDataset
    from budding_yeast_v2.train import EpochSummaryCallback
    class Tiny(pl.LightningModule):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.));self.lr=.001
            self.save_hyperparameters(dict(min_lr=1e-5))
        def training_step(self,batch,index):
            loss=(self.weight-batch[0]).square().mean()
            self.log('train_loss',loss,on_step=False,on_epoch=True);return loss
        def validation_step(self,batch,index):self.log('val_loss_synth',torch.tensor(1.))
        def configure_optimizers(self):return torch.optim.SGD(self.parameters(),lr=0.)
    loader=DataLoader(TensorDataset(torch.tensor([1.,3.])),batch_size=1)
    monitor=LossChangeMonitor(tmp_path,window=2,required_checks=1)
    trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=4,logger=False,
        callbacks=[EpochSummaryCallback(),monitor],enable_checkpointing=False,enable_model_summary=False,
        enable_progress_bar=False,num_sanity_val_steps=0)
    trainer.fit(Tiny(),train_dataloaders=loader,val_dataloaders=loader)
    assert [r['train'] for r in monitor.history]==[5.,5.,5.,5.]
    summaries=[line for line in capsys.readouterr().out.splitlines() if '[Epoch ' in line]
    assert len(summaries)==4 and all('train_loss=5.000000' in line for line in summaries)
    assert json.loads((tmp_path/'convergence.json').read_text())['status']=='plateau_candidate'


@pytest.mark.parametrize('profile,epochs,warmup',[('screen','50/100','4/8'),('full','100/200','5/10')])
def test_launcher_applies_protocol_to_all_fourteen_methods_and_preserves_pool(tmp_path,profile,epochs,warmup):
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'AL_SUITE':'research','DRY_RUN':'1','TRAINING_PROFILE':profile,'EXP_ROOT':str(tmp_path/'results')}
    output=subprocess.check_output(['bash',str(root/'run_sampling_budding.sh')],env=env,text=True)
    tasks=[r for r in output.splitlines() if r.startswith('GPU=')]
    assert len(tasks)==14 and all(f'epochs={epochs}' in r and f'warmup={warmup} min_lr=1e-5' in r for r in tasks)
    assert all('width=192 blocks=4 batch=512 lr=1e-3 modes=24' in r for r in tasks)
    assert 'budding_pool_1000000_seed42_gene_ranges.npz' in output
    assert f'_refit_gene_ranges_research_v1_{profile}/' in output
    assert not (tmp_path/'results').exists()
    rejected=subprocess.run(['bash',str(root/'run_sampling_budding.sh')],env={**env,'MIN_LR':'1'},capture_output=True)
    assert rejected.returncode==2


def test_default_is_full_mechanism_with_safe_new_namespace(tmp_path):
    root=Path(__file__).resolve().parents[1]
    env={k:v for k,v in os.environ.items() if k not in ('AL_SUITE','TRAINING_PROFILE','STRATEGIES','MODEL','MODELS','EXPERIMENTS_TEXT','GPU_ID','GPU_IDS')}
    env.update(DRY_RUN='1',EXP_ROOT=str(tmp_path/'results'))
    output=subprocess.check_output(['bash',str(root/'run_sampling_budding.sh')],env=env,text=True)
    tasks=[line for line in output.splitlines() if line.startswith('GPU=')]
    assert len(tasks)==4 and all('epochs=100/200' in line for line in tasks)
    assert [sum(line.startswith(f'GPU={gpu} ') for line in tasks) for gpu in range(4)]==[1,1,1,1]
    assert 'Suite=mechanism' in output and 'Training profile=full' in output
    assert 'max 1 experiments per GPU' in output and 'mechanism_v1_full' in output
    assert not (tmp_path/'results').exists()


def test_optimized_eleven_remain_explicitly_reproducible(tmp_path):
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'AL_SUITE':'optimized','DRY_RUN':'1','EXP_ROOT':str(tmp_path/'results')}
    output=subprocess.check_output(['bash',str(root/'run_sampling_budding.sh')],env=env,text=True)
    assert sum(line.startswith('GPU=') for line in output.splitlines())==11
    assert 'optimized_v2_full' in output
