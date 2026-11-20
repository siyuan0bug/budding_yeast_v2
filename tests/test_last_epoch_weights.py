"""Best remains the selection model; last is a cheap, standard diagnostic checkpoint."""
from dataclasses import replace
import json
from unittest.mock import patch

import pytest
import torch

from test_sampling_benchmark import config
from test_shared_training_cleanup import selected_fixture
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.utils.evaluation.checkpoints import load_model_from_checkpoint
from budding_yeast_v2.utils.sampling.benchmark.rounds import fit_round


def fixture(tmp_path,monkeypatch):
    _,_,snapshot,training,validation=selected_fixture(tmp_path)
    cfg=replace(config(),epochs=(3,)*4,batch_size=2)
    def validation_step(self,batch,index,dataloader_idx=0):
        self.log('val_loss_synth',torch.tensor([3.,1.,2.][self.current_epoch],device=self.device),
                 on_step=False,on_epoch=True,batch_size=len(batch[0]))
    monkeypatch.setattr(YeastLitModule,'validation_step',validation_step)
    return cfg,snapshot,training,validation


def test_last_checkpoint_is_final_weights_only_and_best_stays_validation_best(tmp_path,monkeypatch):
    cfg,snapshot,training,validation=fixture(tmp_path,monkeypatch)
    output=tmp_path/'round'
    best,stats,_=fit_round(cfg,0,training,validation,output,77,dataset_path=snapshot)
    first=torch.load(output/'model.ckpt',map_location='cpu',weights_only=False)
    final_path=output/'_state/last_weights.ckpt'
    final=torch.load(final_path,map_location='cpu',weights_only=False)
    assert first['epoch']+1==2 and final['epoch']+1==3
    assert not {'optimizer_states','lr_schedulers','callbacks'} & final.keys()
    restored,_=load_model_from_checkpoint(final_path)
    final_stats=NormalizationStats.from_datamodule(restored)
    for key in ('mean_y','std_y','p_mean','p_std'):
        torch.testing.assert_close(getattr(stats,key),getattr(final_stats,key),rtol=0,atol=0)
    assert any(not torch.equal(first['state_dict'][key],final['state_dict'][key]) for key in first['state_dict'])
    meta=json.loads((output/'_state/training.json').read_text())
    assert meta['best_epoch']==2 and meta['last_epoch_checkpoint']['epoch']==3
    assert meta['last_epoch_checkpoint']['weights_only']
    assert not (output/'_state/resume.ckpt').exists()
    final_path.unlink()
    with pytest.raises(RuntimeError,match='retained last-epoch'):
        fit_round(cfg,0,training,validation,output,77,dataset_path=snapshot)


def test_completion_after_plot_interruption_reuses_final_weights_without_retraining(tmp_path,monkeypatch):
    cfg,snapshot,training,validation=fixture(tmp_path,monkeypatch)
    output=tmp_path/'round'
    with patch('budding_yeast_v2.utils.experiments.curves.loss_figure',side_effect=RuntimeError('plot interruption')):
        with pytest.raises(RuntimeError,match='plot interruption'):
            fit_round(cfg,0,training,validation,output,77,dataset_path=snapshot)
    assert (output/'_state/resume.ckpt').exists() and (output/'_state/last_weights.ckpt').exists()
    with patch('budding_yeast_v2.utils.training.runner.make_trainer',side_effect=AssertionError('unexpected retraining')):
        fit_round(cfg,0,training,validation,output,77,dataset_path=snapshot)
    assert not (output/'_state/resume.ckpt').exists()
    assert json.loads((output/'_state/training.json').read_text())['last_epoch_checkpoint']['epoch']==3
