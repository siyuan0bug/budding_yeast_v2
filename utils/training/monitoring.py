"""Read-only per-epoch test monitoring, separate from validation and acquisition.

Inputs/targets are shared CPU references. Only each inference batch is encoded;
no trajectory copies, refitting, gradients, random sampling or model selection.
"""
from contextlib import contextmanager
import random
import numpy as np
import torch
from pytorch_lightning import Callback
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.utils.evaluation.metrics import sample_channel_relative_errors


@contextmanager
def evaluation_state(model):
    """Monitoring cannot change subsequent shuffling, dropout or acquisition RNG."""
    numpy_state,python_state=np.random.get_state(),random.getstate()
    mode=model.training
    device=model.device
    try:
        with torch.random.fork_rng(devices=[device] if device.type=='cuda' else []):
            model.eval()
            with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
                yield
    finally:
        model.train(mode)
        np.random.set_state(numpy_state);random.setstate(python_state)


def evaluate_cohort(model,stats,inputs,truth,*,batch_size,criterion=None):
    """Exactly aggregated normalized objectives and canonical physical RelL2.

    Per-sample/channel ratios accumulate across batches before averaging, so the
    last short batch cannot reweight samples or change the metric formula.
    Physical truth stays raw (including small negatives), not inverse-encoded.
    """
    if batch_size<1 or len(truth)==0 or len(inputs.ids)!=len(truth):
        raise ValueError('Monitoring needs a nonempty aligned cohort and positive batch size')
    if truth.ndim!=3 or len(inputs.params)!=len(truth) or len(inputs.initial)!=len(truth):
        raise ValueError('Monitoring expects aligned N,V,T targets and physical inputs')
    device=model.device
    physical_relative_sum=torch.zeros((),dtype=torch.float64,device=device)
    normalized_error=torch.zeros((),dtype=torch.float64,device=device)
    objective_sum=torch.zeros_like(normalized_error)
    with evaluation_state(model):
        for offset in range(0,len(truth),batch_size):
            stop=min(len(truth),offset+batch_size)
            x,p=stats.encode(inputs.params[offset:stop],inputs.initial[offset:stop],truth.shape[-1])
            target=torch.as_tensor(truth[offset:stop],device=device)
            normalized_target=stats.targets(target)
            prediction=model.forward_ic_time(x.to(device),p.to(device)).float()
            if prediction.shape!=target.shape or not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise FloatingPointError('Nonfinite/misaligned epoch monitoring trajectories')
            normalized_error+=(prediction.double()-normalized_target.double()).square().sum()
            if criterion is not None:
                objective_sum+=criterion.per_sample(prediction,normalized_target).double().sum()
            physical=model.denormalize_y(prediction).double()
            physical_relative_sum+=sample_channel_relative_errors(physical,target.double()).sum()
    mse=float(normalized_error/(len(truth)*truth.shape[1]*truth.shape[2]))
    return dict(mse_norm=mse,loss_norm=float(objective_sum/len(truth)) if criterion is not None else mse,
                rel_l2_physical=float(physical_relative_sum/(len(truth)*truth.shape[1])))


class EpochTestMonitor(Callback):
    """Test metrics are log-only; no test dataloaders enter validation selection."""
    def __init__(self,cohorts,*,batch_size):
        super().__init__()
        if set(cohorts)-{'synth','real'}:raise ValueError('Only same-source synth and real test monitoring are supported')
        self.cohorts=cohorts
        self.batch_size=batch_size
        self.latest={}

    def on_fit_start(self,trainer,pl_module):
        from budding_yeast_v2.utils.experiments.logging import log_event
        log_event('INFO','epoch test monitoring',
                  **{name+'_samples':len(truth) for name,(_,truth) in self.cohorts.items()},
                  purpose='log_only',checkpoint_metric='unchanged_validation_metric')

    def on_validation_epoch_end(self,trainer,pl_module):
        if trainer.sanity_checking:return
        stats=NormalizationStats.from_datamodule(trainer.datamodule)
        self.latest={}
        from .losses import RelativeL2Loss
        criterion=pl_module.criterion if isinstance(pl_module.criterion,RelativeL2Loss) else None
        for name,(inputs,truth) in self.cohorts.items():
            result=evaluate_cohort(pl_module,stats,inputs,truth,batch_size=self.batch_size,criterion=criterion)
            self.latest['test_loss_synth' if name=='synth' else 'test_loss_real']=result['loss_norm']
            if criterion is not None:self.latest['test_'+name+'_mse_norm']=result['mse_norm']
            if name=='real':self.latest['test_real_denorm_rel_l2']=result['rel_l2_physical']
        for name,value in self.latest.items():
            pl_module.log(name,value,on_step=False,on_epoch=True,logger=trainer.logger is not None,
                          add_dataloader_idx=False,sync_dist=False)
