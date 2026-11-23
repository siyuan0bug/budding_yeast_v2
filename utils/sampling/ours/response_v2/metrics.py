"""Full-time task Relative metrics using the shared uncapped criterion."""
import numpy as np
import torch
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.utils.training.losses import RelativeL2Loss


def relative(prediction,target):
    p=torch.as_tensor(np.asarray(prediction).copy(),dtype=torch.float32)
    t=torch.as_tensor(np.asarray(target).copy(),dtype=torch.float32)
    if p.shape!=t.shape or p.ndim!=3 or not torch.isfinite(p).all() or not torch.isfinite(t).all():
        raise ValueError('Relative diagnostics require aligned finite B,V,T arrays')
    value=RelativeL2Loss(eps=.01).per_sample(p,t).numpy()
    if not np.isfinite(value).all():raise FloatingPointError('Nonfinite task-relative diagnostic')
    return value


def normalized_prediction(predictor,inputs):
    result=[];expected=0
    for start,value in predictor.batches(inputs,physical=False):
        value=value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
        if start!=expected or not np.isfinite(value).all():raise ValueError('Unaligned/nonfinite pre-query prediction')
        result.append(value.copy());expected+=len(value)
    if expected!=len(inputs.ids):raise ValueError('Missing pre-query predictions')
    return np.concatenate(result)


def normalization_state(norm):
    return dict({k:None if v is None else v.clone() for k,v in norm.state().items()},norm_type=norm.norm_type)


def restore_normalization(value):return NormalizationStats(**value)
