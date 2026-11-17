"""Axiotis2024 Algorithm 1: representative loss plus squared-distance sensitivity.

No author-specific implementation was located: this is a paper reimplementation.
Native replacement/importance weights and unweighted WOR adaptation are distinct.
"""
import numpy as np
import torch
from sklearn.cluster import KMeans

from ..contracts import PoolInputs, SamplingResult


def select(context, budget, name):
    if context.label_reader is None:
        raise PermissionError('Center-loss sensitivity requires explicit label queries')
    x = np.concatenate((context.pool.params, context.pool.initial), axis=1)
    scale = x.std(0)
    scale[scale < 1e-8] = 1.
    x = ((x - x.mean(0)) / scale).astype(np.float32)
    clusters = min(128, max(2, int(np.sqrt(len(x)))), len(x))
    estimator = KMeans(n_clusters=clusters, random_state=context.seed,
                       n_init=10, max_iter=300).fit(x)
    representatives, distance = [], np.zeros(len(x), dtype=np.float64)
    assignments = estimator.labels_
    loss_by_cluster = np.zeros(clusters, dtype=np.float64)
    present = np.unique(assignments)
    for cluster in present:
        members = np.flatnonzero(assignments == cluster)
        distances = ((x[members] - estimator.cluster_centers_[cluster]) ** 2).sum(1)
        representative = members[np.argmin(distances)]
        representatives.append(representative)
        # Use the actual queried representative as c_i, not an unqueryable mean.
        distance[members] = ((x[members] - x[representative]) ** 2).sum(1)
    representative_view = PoolInputs(context.pool.ids[representatives],
        context.pool.params[representatives], context.pool.initial[representatives])
    targets = context.predictor.normalization.targets(context.label_reader(representative_view.ids)['data'])
    prediction = torch.cat([value for _, value in context.predictor.batches(representative_view)])
    center_loss = (prediction - targets).square().flatten(1).mean(1).numpy()
    loss_by_cluster[present] = center_loss
    holder = .1
    scores = loss_by_cluster[assignments] + holder * distance
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Invalid clustering-sensitivity scores')
    if scores.sum() == 0:
        probabilities = np.full(len(scores), 1 / len(scores))
        degeneracy = 'all scores zero; uniform proposal'
    else:
        probabilities = scores / scores.sum()
        degeneracy = None
    rng = np.random.default_rng(context.seed)
    native = name == 'clustering_sensitivity'
    if native:
        draws = rng.choice(len(scores), size=budget, replace=True, p=probabilities)
        indices, multiplicity = np.unique(draws, return_counts=True)
        # Uniform minibatches over unique rows estimate the native weighted
        # full-universe MEAN risk: (1/(m*n))*sum_draw loss_i/p_i.
        weights = len(indices) * multiplicity / (budget * len(scores) * probabilities[indices])
    else:
        # This is explicitly an unweighted, without-replacement task adaptation;
        # marginal draw probabilities are NOT inclusion probabilities.
        positive = np.flatnonzero(probabilities > 0)
        take = min(budget, len(positive))
        indices = rng.choice(len(scores), size=take, replace=False, p=probabilities)
        if take < budget:
            zeros = np.flatnonzero(probabilities == 0)
            indices = np.r_[indices, rng.choice(zeros, budget - take, replace=False)]
        weights = None
        draws = indices
        multiplicity = np.ones(len(indices), dtype=int)
    return SamplingResult(context.pool.ids[indices], weights=weights,
        probabilities=probabilities[indices], diagnostics=dict(
            source='Axiotis2024 Algorithm1 and experimental representative-center adaptation',
            source_type='paper_reimplementation', embedding='candidate input Z-score',
            z=2, holder_lambda=holder, representative_queries=context.pool.ids[representatives].tolist(),
            draw_slots=int(budget), unique_draw_inputs=len(indices), replacement=native,
            draw_ids=context.pool.ids[draws].tolist(), multiplicity=multiplicity.tolist(),
            draw_probability=probabilities[indices].tolist(), degeneracy=degeneracy,
            weights='linear native importance weights' if native else 'unweighted WOR adaptation; no unbiasedness claim'))
