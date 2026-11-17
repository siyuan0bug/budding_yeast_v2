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


def residual_scores(trajectories, params, horizon=210.):
    y = torch.as_tensor(np.asarray(trajectories).copy(), dtype=torch.float64)
    p = torch.as_tensor(np.asarray(params).copy(), dtype=torch.float64)
    if (y.ndim!=3 or y.shape[1]!=39 or y.shape[-1]<5 or p.ndim!=2 or len(p)!=len(y)
            or not torch.isfinite(y).all() or not torch.isfinite(p).all()
            or not np.isfinite(horizon) or horizon<=0):
        raise ValueError('PRE requires aligned finite 39D physical trajectories and positive horizon')
    dt = horizon / (y.shape[-1] - 1)
    continuous = continuous_mask(y)
    # Every point in a central-difference stencil must be continuous.
    mask = continuous[:, :-2] & continuous[:, 1:-1] & continuous[:, 2:]
    derivative = (y[:, :, 2:] - y[:, :, :-2]) / (2 * dt)
    time = torch.linspace(dt, horizon - dt, y.shape[-1] - 2, dtype=y.dtype)
    predicted_rhs = rhs(y[:, :, 1:-1].transpose(1, 2), p, time).transpose(1, 2)
    error = (derivative - predicted_rhs).abs().mean(1)
    if not mask.any(1).all() or not torch.isfinite(error).all():
        raise ValueError('Physics-PRE has no finite continuous evaluation interval')
    return ((error * mask).sum(1) / mask.sum(1)).numpy()


def select(context, budget, name):
    refs = np.concatenate([residual_scores(context.train.trajectories[i:i + 64],
                                           context.train.inputs.params[i:i + 64])
                           for i in range(0, len(context.train.inputs.ids), 64)])
    scale = context.predictor.normalization.p_std.numpy()
    nearest = NearestNeighbors(n_neighbors=1).fit(context.train.inputs.params / scale)
    scores = np.empty(len(context.pool.ids), dtype=np.float64)
    floor = 1e-6
    for start, predicted in context.predictor.batches(context.pool, physical=True):
        end = start + len(predicted)
        params = context.pool.params[start:end]
        reference_ids = nearest.kneighbors(params / scale, return_distance=False)[:, 0]
        scores[start:end] = residual_scores(predicted.numpy(), params) / np.maximum(refs[reference_ids], floor)
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
        selected_scores=scores[indices].tolist(), training_reference_labels_only=True))
