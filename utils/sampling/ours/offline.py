"""Reuse original research algorithms with a queried-only complete-pool view."""
import json

import numpy as np
import torch

from ..contracts import SamplingResult
from .auxiliary.views import Inputs, Training, AcquisitionData, Normalization


def _inputs(value):
    metadata = tuple(value.metadata) if value.metadata else tuple({} for _ in value.ids)
    from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
    overrides = []
    for row, meta in zip(value.params, metadata):
        override = dict(meta.get('_effective_overrides', {}))
        override.update(zip(PARAM_KEYS, map(float, row)))
        overrides.append(override)
    overrides = tuple(overrides)
    return Inputs(value.params, value.initial, metadata, overrides, value.ids)


def select(context, budget, name):
    if getattr(context, 'system_id', 'budding') != 'budding':
        raise ValueError('Historical Pattern/sensitivity methods are Budding-only; use an explicit portable mechanism adapter')
    norm = context.predictor.normalization
    view = AcquisitionData(_inputs(context.pool),
        Training(_inputs(context.train.inputs), tuple(context.train.trajectories)),
        Normalization(norm.mean_y, norm.std_y, norm.p_mean, norm.p_std, 'log1p'),
        context.predictor.steps, 210.)
    if name in ('ours_pattern_balance', 'ours_pattern_context', 'ours_pattern_support'):
        from .pattern import choose
        indices, report, predictions, classifier = choose(view, context.predictor.members,
            name, budget, context.seed, context.device)
        # The fitted estimator/selected predictions are persisted by the runner.
        result = SamplingResult(context.pool.ids[indices], diagnostics=report)
        result.artifacts = dict(predictions=predictions, classifier=classifier)
        return result
    if name == 'ours_sensitivity':
        from .auxiliary.sensitivity import score_pool, select_scores
        if not view.pool.metadata or any(not row.get('pairs', row.get('genes')) for row in view.pool.metadata):
            raise ValueError('Mutation-response sensitivity requires real mutation metadata')
        scores, details = score_pool(view, context.predictor.members[0], context.device, 0)
        indices, exploit = select_scores(scores, budget, context.seed)
        return SamplingResult(context.pool.ids[indices], diagnostics=dict(
            eligible_positive_scores=int((scores > 0).sum()), exploit=int(exploit),
            selected_scores=scores[indices].tolist()))
    raise ValueError(f'Offline adaptation has not been validated for {name}')
