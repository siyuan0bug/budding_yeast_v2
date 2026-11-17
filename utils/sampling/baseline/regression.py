"""Exact per-example FNO output-head gradients for trajectory regression.

No classification pseudo-labels, physical basis queries or held-out data. The
head derivative is verified against autograd; optional sketches are explicit.
"""
import numpy as np
import torch

from ..contracts import PoolInputs
from .geometry import gaussian_projection
from budding_yeast_v2.utils.experiments.logging import log_event


def relative_head_gradients(prediction, target, hidden, eps=.01):
    if prediction.shape!=target.shape or prediction.shape[::2]!=(hidden.shape[0],hidden.shape[2]):
        raise ValueError('Misaligned regression head tensors')
    error=prediction.double()-target.double()
    denominator=torch.hypot(torch.linalg.vector_norm(target.double(),dim=-1),target.new_tensor(eps).double())
    length=torch.linalg.vector_norm(error,dim=-1)
    derivative=torch.where(length[...,None]>0,
        error/(length.clamp_min(1e-300)*denominator)[...,None],0.)/prediction.shape[1]
    weight=torch.einsum('bct,bht->bch',derivative,hidden.double())
    bias=derivative.sum(-1)
    gradients=torch.cat((weight.flatten(1),bias),1)
    if not torch.isfinite(gradients).all():raise FloatingPointError('Nonfinite regression head gradient')
    return gradients


def gradient_bank(predictor, inputs, label_reader, *, dimension=128, eps=.01):
    if label_reader is None:raise PermissionError('Regression gradient selection requires authorized training-side truth')
    model=predictor.members[0];backbone=getattr(model,'model',model)
    if not hasattr(backbone,'project') or not isinstance(backbone.project[-1],torch.nn.Conv1d):
        raise ValueError('The current regression gradient adapter requires the FNO linear output head')
    output=None;projection=None;training=model.training;model.eval()
    try:
        for start in range(0,len(inputs.ids),predictor.batch_size):
            stop=min(start+predictor.batch_size,len(inputs.ids));rows=inputs.ids[start:stop]
            truth=label_reader(rows)['data']
            x,p=predictor.normalization.encode(inputs.params[start:stop],inputs.initial[start:stop],predictor.steps)
            with torch.no_grad():
                prediction,hidden=backbone(x.to(predictor.device),p.to(predictor.device),return_features=True)
                target=predictor.normalization.targets(truth).to(prediction.device)
                full=relative_head_gradients(prediction,target,hidden,eps).cpu().float()
            if output is None:
                width=min(dimension,full.shape[1]) if dimension else full.shape[1]
                output=np.empty((len(inputs.ids),width),np.float64)
                if width<full.shape[1]:projection=gaussian_projection(full.shape[1],width,predictor.sketch_seed)
            output[start:stop]=(full if projection is None else full@projection).numpy()
            if len(inputs.ids)>=1024 and (start==0 or stop==len(inputs.ids) or stop//8192!=start//8192):
                log_event('PROGRESS','regression gradient features',completed=stop,total=len(inputs.ids),dimension=output.shape[1])
    finally:model.train(training)
    if output is None:raise ValueError('Empty gradient candidate bank')
    return output


def subset(inputs,indices):
    indices=np.asarray(indices,dtype=np.int64)
    return PoolInputs(inputs.ids[indices],inputs.params[indices],inputs.initial[indices])
