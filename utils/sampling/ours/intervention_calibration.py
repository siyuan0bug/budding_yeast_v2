"""Input-only intervention-order diagnostic, not a claimed novel sampler.

Fixed quota controls test whether the source pool's high-order mass is useful.
They preserve common initial IDs and sample uniformly inside each actual L.
No true trajectory, real identity, error score, or named phenotype is accepted.
"""
from collections import defaultdict
import numpy as np


METHODS = ('calibration_pool_random_v1', 'calibration_order_balanced_v1',
           'calibration_low_order_first_v1')


def reserve_roles(rows, levels, contexts, initial, *, seed=42, fraction=.15, max_contexts=64):
    rows, levels, initial = map(np.asarray, (rows, levels, initial))
    if (len(rows) != len(levels) or len(contexts) != len(rows) or len(set(rows)) != len(rows)
            or len(set(initial)) != len(initial) or not set(initial) <= set(rows)
            or not set(levels) <= {1, 2, 3} or not 0 < fraction < .5):
        raise ValueError('Malformed intervention-order role inputs')
    initial_set = set(initial)
    known = {key for row, key in zip(rows, contexts) if row in initial_set}
    grouped = defaultdict(lambda: defaultdict(list))
    for row, level, key in zip(rows, levels, contexts):
        if key not in known:
            grouped[int(level)][key].append(int(row))
    rng = np.random.default_rng(seed)
    roles = {r: [] for r in ('feedback', 'audit')}
    roles.update({r+'_contexts': [] for r in ('feedback', 'audit')})
    for level in (1, 2, 3):
        groups = grouped[level]
        if len(groups) < 8:
            raise ValueError('Need >=8 unseen full contexts per intervention order')
        keys = sorted(groups); rng.shuffle(keys)
        count = min(max_contexts, max(4, int(len(keys)*fraction)), len(keys)//2)
        for j, role in enumerate(('feedback', 'audit')):
            for key in keys[j*count:(j+1)*count]:
                values = groups[key]
                chosen = rng.choice(values, min(8, len(values)), replace=False)
                roles[role].extend(map(int, chosen)); roles[role+'_contexts'].append(key)
    forbidden = set(roles['feedback_contexts']) | set(roles['audit_contexts'])
    roles['training'] = [int(r) for r, k in zip(rows, contexts) if k not in forbidden]
    roles['initial'] = list(map(int, initial))
    return roles


def quota_targets(capacity, lower, total, balanced=True):
    """Capacity-capped equal final counts, or lexicographic low-order priority."""
    cap, target = np.asarray(capacity, dtype=int), np.asarray(lower, dtype=int).copy()
    if (cap.shape != (3,) or target.shape != (3,) or (target < 0).any()
            or (target > cap).any() or not target.sum() <= total <= cap.sum()):
        raise ValueError('Impossible unique-input quota')
    remaining = int(total-target.sum())
    if balanced:
        while remaining:
            active = np.flatnonzero(target < cap)
            j = int(active[np.argmin(target[active])])
            target[j] += 1; remaining -= 1
    else:
        for j in range(3):
            take = min(remaining, int(cap[j]-target[j]))
            target[j] += take; remaining -= take
    if remaining:
        raise RuntimeError('Quota allocation did not exhaust its budget')
    return target


def select_rows(rows, levels, initial, count, method, *, seed=42):
    rows, levels, initial = map(np.asarray, (rows, levels, initial))
    if method not in METHODS or len(rows) != len(levels) or len(set(rows)) != len(rows):
        raise ValueError('Unknown method or malformed candidate IDs')
    lookup = dict(zip(map(int, rows), map(int, levels)))
    if not set(initial) <= set(rows) or not len(initial) <= count <= len(rows):
        raise ValueError('Initial/budget outside candidate universe')
    initial_set = set(initial)
    available = rows[~np.isin(rows, initial)]
    capacity = np.array([(levels == j).sum() for j in (1, 2, 3)])
    lower = np.array([sum(lookup[int(i)] == j for i in initial) for j in (1, 2, 3)])
    if method == METHODS[0]:
        chosen = np.random.default_rng(seed).choice(available, count-len(initial), replace=False)
        targets = np.array([sum(lookup[int(i)] == j for i in np.r_[initial, chosen]) for j in (1, 2, 3)])
    else:
        targets = quota_targets(capacity, lower, count, balanced=method == METHODS[1])
        chosen = []
        for j in (1, 2, 3):
            members = np.array([int(i) for i in available if lookup[int(i)] == j])
            # Shared within-level ordering across the two quota controls.
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), j]))
            chosen.extend(rng.permutation(members)[:targets[j-1]-lower[j-1]].tolist())
        chosen = np.asarray(chosen, dtype=np.int64)
    chosen = np.sort(chosen)
    np.random.default_rng(np.random.SeedSequence([int(seed), 777])).shuffle(chosen)
    selected = np.r_[initial, chosen].astype(np.int64)
    if len(set(selected)) != count:
        raise RuntimeError('Duplicate selected inputs')
    return selected, dict(method=method, initial_counts=lower.tolist(), capacity=capacity.tolist(),
        final_counts=targets.tolist(), biological_distribution_unchanged=True,
        within_level='uniform_without_replacement', output_labels_used=False,
        diagnostic_only=True, not_a_claimed_novel_algorithm=True)


def balanced_metrics(per_sample, levels, contexts):
    """Each order has weight 1/3; contexts retained for later paired audits."""
    levels = np.asarray(levels)
    by_level = {}
    keys = ('normalized_relative', 'physical_mae', 'physical_relative', 'physical_mse')
    for level in (1, 2, 3):
        mask = levels == level
        if not mask.any():
            raise ValueError('Every order must have a nonempty independent cohort')
        by_level[str(level)] = dict(count=int(mask.sum()), contexts=len(set(np.asarray(contexts)[mask])),
            **{key: float(np.asarray(per_sample[key])[mask].mean()) for key in keys})
    return dict(by_level=by_level, balanced={key: float(np.mean([x[key] for x in by_level.values()])) for key in keys},
        per_sample=per_sample, levels=levels.tolist(), contexts=list(contexts))


def decision(candidate, control, *, min_gain=.05, slack=.05):
    """Predeclared synthetic-only screen/confirmation, never real metrics."""
    if not candidate or len(candidate) != len(control):
        raise ValueError('Need aligned nonempty independent seeds')
    checked = [float(row['balanced'][key]) for row in candidate+control
               for key in ('normalized_relative', 'physical_mae', 'physical_relative')]
    if not np.isfinite(checked).all() or min(checked) < 0:
        raise ValueError('Invalid synthetic diagnostic metrics')
    if any(row['balanced'][key] == 0 for row in control
           for key in ('normalized_relative', 'physical_mae', 'physical_relative')):
        return dict(passed=False, reason='zero control risk; no relative improvement claim')
    gains = [1-a['balanced']['normalized_relative']/b['balanced']['normalized_relative']
             for a, b in zip(candidate, control)]
    mean_a = np.mean([a['balanced']['normalized_relative'] for a in candidate])
    mean_b = np.mean([b['balanced']['normalized_relative'] for b in control])
    gain = float(1-mean_a/mean_b)
    guard = {}
    for key in ('physical_mae', 'physical_relative'):
        guard[key] = float(np.mean([a['balanced'][key] for a in candidate])/
                           np.mean([b['balanced'][key] for b in control]))
    levels = {level: float(np.mean([a['by_level'][str(level)]['normalized_relative'] for a in candidate])/
                          np.mean([b['by_level'][str(level)]['normalized_relative'] for b in control]))
              for level in (1, 2, 3)}
    passed = (gain >= min_gain and sum(x > 0 for x in gains) >= (1 if len(gains) == 1 else 2)
              and max(guard.values()) <= 1+slack and max(levels.values()) <= 1.10)
    return dict(passed=bool(passed), normalized_gain=gain, per_seed_gain=list(map(float, gains)),
        physical_ratios=guard, level_ratios=levels, threshold=min_gain, physical_slack=slack,
        information='synthetic feedback or independent audit only')
