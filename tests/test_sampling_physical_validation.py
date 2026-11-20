"""Checkpoint selection uses the canonical physical metric, not normalized loss."""
from types import SimpleNamespace
import pytest
import torch

from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.utils.evaluation.metrics import relative_l2_loss


@pytest.mark.parametrize('batch_size',[1,2,3])
def test_validation_matches_physical_sample_channel_metric_independent_of_batches(batch_size):
    model=YeastLitModule(model_name='fno',width=4,modes=2,n_blocks=1,loss_type='rel_l2')
    model._trainer=SimpleNamespace(datamodule=SimpleNamespace(validation_metric='sample_channel_rel_l2'),logger=None)
    logged={}
    model.log=lambda name,value,**kwargs:logged.update({name:float(value)})
    def forward(x,p):
        assert not torch.is_autocast_enabled(x.device.type)
        return x[:,:,0,:]
    model.forward_ic_time=forward
    model.denormalize_y=lambda value:value
    truth=torch.tensor([.001,100.,0.])[:,None,None].expand(-1,39,4).clone()
    truth[0,5,0]=-1e-6
    prediction=truth+torch.tensor([.01,1.,.02])[:,None,None]
    model.on_validation_epoch_start()
    for offset in range(0,3,batch_size):
        y=truth[offset:offset+batch_size];p=prediction[offset:offset+batch_size]
        # Deliberately different normalized targets: physical scoring must use y.
        with torch.autocast('cpu',dtype=torch.bfloat16):
            model.validation_step((p[:,:,None,:],torch.zeros(len(y),141),y+999,y),0)
    model.on_validation_epoch_end()
    assert logged['val_sample_channel_rel_l2']==pytest.approx(float(relative_l2_loss(prediction,truth)))
    assert 'val_macro_rel_l2' not in logged
    assert logged['val_loss_synth']!=pytest.approx(logged['val_sample_channel_rel_l2'])
