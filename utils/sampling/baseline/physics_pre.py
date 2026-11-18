"""Polanska Eq.5 and nearest-queried-reference normalization, adapted to events."""
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from budding_yeast_v2.utils.physics.canonical_rhs import rhs
from .uncertainty import _power_sample_indices
from ..contracts import SamplingResult


def continuous_mask(y, window=4):
    """Bracket directed event crossings and observed resets, not whole threshold bands.

    A slowly varying prediction near CLB2=.3 is not an event at every point.
    No true candidate event times are needed or accepted by this function.
    """
    clb2, clb5, ori, spn = y[:, 2], y[:, 3], y[:, 33], y[:, 35]
    events = torch.zeros((len(y), y.shape[-1]), dtype=torch.bool, device=y.device)
    for signal, downward in ((clb2 - .3, True), (clb2 + clb5 - .2, True),
                             (ori - 1., False), (spn - 1., False)):
        a, b = signal[:, :-1], signal[:, 1:]
        events[:, 1:] |= ((a >= 0) & (b < 0)) if downward else ((a <= 0) & (b > 0))
    # MASS grows continuously; only downward drops are division evidence.
    events[:, 1:] |= y[:, 0, :-1] - y[:, 0, 1:] > .1
    for channel, threshold in ((33, .5), (34, .3), (35, .3)):
        a, b = y[:, channel, :-1], y[:, channel, 1:]
        events[:, 1:] |= ((a - b) > threshold) & (b < .25 * a)
    # Event flags have zero between-event derivative in the canonical model.
    for channel in (36, 38):
        events[:, 1:] |= (y[:, channel, 1:] - y[:, channel, :-1]).abs() > .3
    near = torch.nn.functional.max_pool1d(events[:, None].to(y.dtype),
        kernel_size=2 * window + 1, stride=1, padding=window)[:, 0].bool()
    return ~near


def residual_scores(trajectories, params, horizon=210., *, sample_ids=None, return_diagnostics=False):
    y = torch.as_tensor(np.asarray(trajectories).copy(), dtype=torch.float64)
    p = torch.as_tensor(np.asarray(params).copy(), dtype=torch.float64)
    if (y.ndim!=3 or y.shape[1]!=39 or y.shape[-1]<5 or p.ndim!=2 or len(p)!=len(y)
            or not torch.isfinite(y).all() or not torch.isfinite(p).all()
            or not np.isfinite(horizon) or horizon<=0):
        raise ValueError('PRE requires aligned finite 39D physical trajectories and positive horizon')
    ids=np.arange(len(y)) if sample_ids is None else np.asarray(sample_ids)
    if ids.shape!=(len(y),):raise ValueError('PRE sample IDs must align with trajectories')
    dt = horizon / (y.shape[-1] - 1)
    continuous = continuous_mask(y)
    # Every point in a central-difference stencil must be continuous.
    mask = continuous[:, :-2] & continuous[:, 1:-1] & continuous[:, 2:]
    # Four-point padding is conservative, not part of the event itself. A
    # noisy surrogate may make these neighborhoods overlap over the full
    # horizon. Shrink only the extra padding, retaining every detected event
    # and requiring all three points of each difference to be event-free.
    windows=torch.full((len(y),),4,dtype=torch.int64)
    minimum=min(8,y.shape[-1]-2)
    for window in (3,2,1,0):
        sparse=mask.sum(1)<minimum
        if not sparse.any():break
        safe=continuous_mask(y[sparse],window)
        mask[sparse]=safe[:,:-2]&safe[:,1:-1]&safe[:,2:]
        windows[sparse]=window
    empty=~mask.any(1)
    if empty.any():
        bad=ids[empty.numpy()].tolist()
        raise ValueError(f'Physics-PRE has no event-free three-point stencil even without padding; sample_ids={bad}')
    derivative = (y[:, :, 2:] - y[:, :, :-2]) / (2 * dt)
    time = torch.linspace(dt, horizon - dt, y.shape[-1] - 2, dtype=y.dtype)
    predicted_rhs = rhs(y[:, :, 1:-1].transpose(1, 2), p, time).transpose(1, 2)
    error = (derivative - predicted_rhs).abs().mean(1)
    invalid=mask&~torch.isfinite(error)
    if invalid.any():
        bad=ids[invalid.any(1).numpy()].tolist()
        raise FloatingPointError(f'Physics-PRE nonfinite ODE residual on continuous stencils; sample_ids={bad}')
    # Multiplication would propagate NaN from a masked singular event point.
    scores=(torch.where(mask,error,0.).sum(1)/mask.sum(1)).numpy()
    diagnostics=dict(adaptive_event_padding=[dict(sample_id=int(ids[i]),window=int(windows[i]),
        continuous_stencils=int(mask[i].sum())) for i in torch.where(windows<4)[0].tolist()],
        minimum_continuous_stencils=int(mask.sum(1).min()),
        masked_nonfinite_points=int((~mask&~torch.isfinite(error)).sum()))
    return (scores,diagnostics) if return_diagnostics else scores


def select(context, budget, name):
    audit=dict(training_reference=[],candidates=[],masked_nonfinite_points=0)
    def audited_scores(y,p,ids,role):
        values,details=residual_scores(y,p,sample_ids=ids,return_diagnostics=True)
        audit[role].extend(details['adaptive_event_padding'])
        audit['masked_nonfinite_points']+=details['masked_nonfinite_points']
        return values
    refs = np.concatenate([audited_scores(context.train.trajectories[i:i + 64],
        context.train.inputs.params[i:i + 64],context.train.inputs.ids[i:i + 64],'training_reference')
        for i in range(0, len(context.train.inputs.ids), 64)])
    scale = context.predictor.normalization.p_std.detach().cpu().numpy()
    nearest = NearestNeighbors(n_neighbors=1).fit(context.train.inputs.params / scale)
    scores = np.empty(len(context.pool.ids), dtype=np.float64)
    floor = 1e-6
    for start, predicted in context.predictor.batches(context.pool, physical=True):
        end = start + len(predicted)
        params = context.pool.params[start:end]
        reference_ids = nearest.kneighbors(params / scale, return_distance=False)[:, 0]
        scores[start:end] = audited_scores(predicted.numpy(),params,context.pool.ids[start:end],'candidates') / np.maximum(refs[reference_ids], floor)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite normalized PRE acquisition score')
    if name == 'physics_pre_topk':
        indices = np.argsort(-scores, kind='stable')[:budget]
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(context.seed)
            indices = _power_sample_indices(scores, budget).numpy()
    return SamplingResult(context.pool.ids[indices], diagnostics=dict(
        source='Polanska2026 Eq.5 and nearest training truth residual normalization',
        task_adaptation='canonical 39D ODE, central difference, predicted-event mask',
        reference_floor=floor, parameter_distance='training-Z-score Euclidean',
        event_padding_audit=audit,
        selected_scores=scores[indices].tolist(), training_reference_labels_only=True))
