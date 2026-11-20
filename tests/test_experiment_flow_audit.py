"""Invalid predictions cannot hide behind decode clamps or checkpoint monitors."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from pytorch_lightning import Callback

from test_shared_training_cleanup import selected_fixture
from test_sampling_benchmark import config
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.utils.evaluation.checkpoints import Predictor
from budding_yeast_v2.utils.sampling.contracts import PoolInputs
from budding_yeast_v2.utils.sampling.benchmark.rounds import fit_round


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_validation_aborts_before_logging_nonfinite_checkpoint_score(bad):
    model=YeastLitModule(model_name='fno',width=4,modes=2,n_blocks=1,loss_type='rel_l2')
    model._trainer=SimpleNamespace(datamodule=SimpleNamespace(validation_metric='normalized_loss'),logger=None)
    model.forward_ic_time=lambda x,p:torch.full((len(x),39,11),bad)
    model.log=lambda *a,**k:pytest.fail('Invalid prediction was logged for checkpoint selection')
    with pytest.raises(FloatingPointError,match='normalized validation'):
        model.validation_step((torch.zeros(2,39,2,11),torch.zeros(2,141),torch.ones(2,39,11),torch.ones(2,39,11)),0)


def test_physical_evaluator_rejects_inf_before_inverse_clamp_can_hide_it():
    class Model(torch.nn.Module):
        def forward_ic_time(self,x,p):return torch.full((len(x),39,11),float('inf'))
    stats=NormalizationStats(torch.zeros(1,39,1),torch.ones(1,39,1),
                             torch.zeros(1,141),torch.ones(1,141))
    # This finite decode is exactly why the raw prediction must be checked.
    assert torch.isfinite(stats.decode(torch.full((1,39,11),float('inf')))).all()
    predictor=Predictor([Model()],stats,11,'cpu')
    inputs=PoolInputs(np.arange(2),np.ones((2,141)),np.ones((2,39)))
    with pytest.raises(FloatingPointError,match='before decoding'):
        predictor.physical(inputs)


def test_history_records_learning_rate_used_not_only_next_scheduled_rate(tmp_path):
    _,_,snapshot,training,validation=selected_fixture(tmp_path)
    cfg=replace(config(),epochs=(3,)*4,warmup=(1,)*4)
    observed=[]
    class ActualLR(Callback):
        def on_train_batch_start(self,trainer,model,batch,index):
            if index==0:observed.append(float(trainer.optimizers[0].param_groups[0]['lr']))
    _,_,history=fit_round(cfg,0,training,validation,tmp_path/'round',77,dataset_path=snapshot,
                          extra_callbacks=[ActualLR()])
    assert [r['lr'] for r in history]==observed
    assert history[0]['lr']==0 and history[0]['lr_next']==cfg.lr
