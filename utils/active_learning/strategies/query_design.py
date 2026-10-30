"""Legal, pool-only query geometry and diminishing-return batch selection.

All row indices are local to one SelectionPlan. Stable candidate IDs are used
at the query/feedback boundary. No oracle labels enter these functions.
"""
from collections import defaultdict
from functools import lru_cache
import hashlib
import heapq
import json

import numpy as np

from .response import _pairs, _strength_curve

SETTINGS = dict(point_fraction=.1, random_fraction=.1, direction_sketch_dim=16,
                sketch_seed=1729, distance_penalty='1+RMS_CDF_distance')


def input_table(engine):
    if engine._pool_initial_physical is None:
        raise ValueError('Experimental AL requires physical pool initial states; call setup with a current pool')
    params = engine._pool_P_real.detach().cpu().numpy()
    initials = engine._pool_initial_physical.detach().cpu().numpy()
    overrides, metadata = list(engine._pool_ovr), list(engine._pool_mass_meta)
    provenance = getattr(engine, '_cum_synth_provenance', [])
    if provenance:
        params = np.concatenate((params, np.asarray(engine._cum_synth_params, dtype=np.float32)))
        initials = np.concatenate((initials, np.asarray([r['y_init'] for r in provenance])))
        overrides += [json.loads(r['overrides_json']) for r in provenance]
        metadata += [json.loads(r['candidate_metadata_json']) for r in provenance]
    if any(not _pairs(row) for row in metadata):
        raise ValueError('Experimental AL requires structured synthetic mutation metadata')
    if not (len(params) == len(initials) == len(overrides) == len(metadata)):
        raise ValueError('Synthetic input/provenance alignment error')
    return params, initials, overrides, metadata


def background_keys(params, metadata):
    from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
    mdt = PARAM_KEYS.index('mdt')
    return [(_pairs(row), float(p[mdt])) for p, row in zip(params, metadata)]


def action_variance(features, actions):
    """Variance of b-a, or (d-c)-(b-a), evaluated in bounded chunks."""
    result = np.empty(len(actions), dtype=np.float32)
    for start in range(0, len(actions), 256):
        part = actions[start:start+256]
        delta = features[:, part[:, 1]] - features[:, part[:, 0]]
        if part.shape[1] == 4:
            delta = (features[:, part[:, 3]] - features[:, part[:, 2]]) - delta
        result[start:start+len(part)] = np.mean(delta**2, axis=(0, 2))
    return result


def marginal_actions(point, features, actions, distances, keys, pool_size, budget,
                     seed, action_scores=None, kind='response', *, preselected=(),
                     point_fraction=None, random_fraction=None, fallback=None):
    """10% point exploration, 80% contrasts, 10% random; unique-query costs.

    Contrasts receive only response reward, never endpoint point reward.
    Costs update when an endpoint is selected. Direction novelty uses a PSD
    accumulator in a deterministic 16-dimensional CountSketch of ensemble responses.
    Sparse adjacency updates handle cost *increases in utility* correctly;
    lazy updates handle monotonically decreasing directional novelty.
    """
    if not 0 <= budget <= pool_size:
        raise ValueError('Invalid acquisition budget')
    score = action_variance(features, actions) if action_scores is None else np.asarray(action_scores)
    if (not np.isfinite(point).all() or not np.isfinite(score).all()
            or (score < 0).any() or len(score) != len(actions)):
        raise ValueError('Nonfinite or negative experimental acquisition score')
    selected = list(map(int,preselected))
    if len(set(selected))!=len(selected) or any(i<0 or i>=pool_size for i in selected):
        raise ValueError('Invalid preselected pool endpoints')
    chosen, records = set(selected), []
    rng = np.random.RandomState(seed)
    random_count = int(budget * (SETTINGS['random_fraction'] if random_fraction is None else random_fraction))
    point_count = (max(1,int(budget*SETTINGS['point_fraction'])) if point_fraction is None
                   else int(budget*point_fraction))
    point_count = min(budget-random_count-len(selected),point_count)
    if point_count<0:raise ValueError('Preselected endpoints exceed nonrandom budget')
    point_candidates=[]
    if point_count:
        order=np.argsort(-np.asarray(point[:pool_size]),kind='stable')
        for i in order:
            if int(i) not in chosen:point_candidates.append(int(i))
            if len(point_candidates)==point_count:break
    for i in point_candidates:
        i = int(i); chosen.add(i); selected.append(i)
        records.append(dict(kind='point', endpoints=[i], new_queries=1, score=float(point[i])))
    limit = budget-random_count
    # Every response coordinate contributes; subsampling individual coordinates
    # could entirely miss a short event or one of the 39 channels.
    dimension=features.shape[0]*features.shape[-1]
    width=min(SETTINGS['direction_sketch_dim'],dimension)
    sketch_rng=np.random.RandomState(SETTINGS['sketch_seed'])
    buckets=sketch_rng.randint(0,width,size=dimension)
    signs=sketch_rng.choice([-1.,1.],size=dimension)
    covariance, versions = {}, defaultdict(int)
    adjacency = defaultdict(list)
    groups = [keys[int(row[-1])] for row in actions]
    for j, row in enumerate(actions):
        for node in set(map(int, row)):
            if node < pool_size:
                adjacency[node].append(j)

    def direction(j):
        row = actions[j]
        v = features[:,row[1]]-features[:,row[0]]
        if len(row) == 4:
            v = features[:,row[3]]-features[:,row[2]]-v
        v=np.bincount(buckets,weights=v.ravel()*signs,minlength=width)
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-12 else np.zeros_like(v)

    def utility(j):
        new = [i for i in set(map(int, actions[j])) if i < pool_size and i not in chosen]
        if not new or score[j] <= 0:
            return 0., new
        group = groups[j]
        novelty = 1.
        if group in covariance:
            v = direction(j)
            novelty /= 1.+float(v @ covariance[group] @ v)
        return float(score[j])*novelty/(1.+float(distances[j]))/len(new), new

    heap = []
    tickets = np.zeros(len(actions), dtype=np.int64)
    def enqueue(j):
        value, _ = utility(j)
        tickets[j] += 1
        if value > 0:
            heapq.heappush(heap, (-value, j, int(tickets[j]), versions[groups[j]]))
    for j in range(len(actions)):
        enqueue(j)
    while heap and len(selected) < limit:
        negative, j, ticket, version = heapq.heappop(heap)
        if ticket != tickets[j]:
            continue
        if version != versions[groups[j]]:
            enqueue(j); continue
        value, new = utility(j)
        if not new or len(new)+len(selected) > limit:
            continue
        selected.extend(new); chosen.update(new)
        records.append(dict(kind=kind, endpoints=list(map(int, actions[j])), new_queries=len(new),
                            score=value, response_variance=float(score[j]), distance=float(distances[j])))
        group = groups[j]; v = direction(j)
        if group not in covariance:
            covariance[group] = np.zeros((len(v), len(v)), dtype=np.float32)
        covariance[group] += np.outer(v, v)
        versions[group] += 1
        for affected in {a for node in new for a in adjacency[node]}:
            enqueue(affected)
    # Sparse/no contrasts: explicitly label the non-response fallback.
    fallback_candidates = (np.argsort(-np.asarray(point[:pool_size]),kind='stable') if fallback is None
                           else fallback(limit-len(selected),np.asarray(selected,dtype=np.int64)))
    for i in fallback_candidates:
        if len(selected) >= limit:
            break
        i = int(i)
        if i not in chosen:
            chosen.add(i); selected.append(i)
            records.append(dict(kind='point_fallback' if fallback is None else 'global_fallback',
                                endpoints=[i],new_queries=1,score=float(point[i])))
    if len(selected)!=limit:raise ValueError('Fallback did not fill the nonrandom budget')
    remaining = np.setdiff1d(np.arange(pool_size), selected)
    for i in rng.choice(remaining, budget-len(selected), replace=False):
        i = int(i); selected.append(i)
        records.append(dict(kind='random', endpoints=[i], new_queries=1))
    return np.asarray(selected, dtype=np.int64), records


@lru_cache(maxsize=1)
def validate_invariance_rule():
    """Fail closed if the equation/event implementation changes after this proof."""
    from pathlib import Path
    from budding_yeast_v2.data.systems.budding import ode, simulation
    expected=((ode, 'cc58e4efa1ee62fc617cae74c0c170059e72eac46cd0a194eccc6e35bfc6b647'),
              (simulation, '9156a2f8d5ef34b15dd9e6daa68dda767df77ba0034313ef872460e0a2662ec1'))
    for module,digest in expected:
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Canonical ODE/events changed: re-audit TEM1 invariance before enabling this selector')


def invariance_edges(params, initials, overrides, edges):
    """Only a proved condition: Tem1T=TEM1(0)=0, change ka15pp alone.

    Current events do not reset TEM1; dTEM1/dt vanishes at TEM1=0.
    ka15pp appears only multiplied by TEM1 in the canonical ODE.
    """
    from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
    validate_invariance_rule()
    target, total = PARAM_KEYS.index('ka15pp'), PARAM_KEYS.index('Tem1T')
    valid = []
    for a, b in edges:
        if params[a, total] != 0 or params[b, total] != 0 or initials[a, 28] != 0 or initials[b, 28] != 0:
            continue
        changed = np.flatnonzero(params[a] != params[b])
        if changed.tolist() != [target] or not np.array_equal(initials[a], initials[b]):
            continue
        # background_graph has already checked event switches and legal modes.
        valid.append((int(a), int(b)))
    return np.asarray(valid, dtype=np.int64).reshape(-1, 2)


def interaction_blocks(params, initials, overrides, metadata, edges):
    """Exact MC contrasts with/without one additional KO, all endpoints existing.

    No tolerance matching, fabricated candidate or extra oracle call. Conservative
    first implementation: exclude overlapping override controls and fixed/event
    switches, and require identical ICs within each dose contrast.
    """
    from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
    from budding_yeast_v2.data.systems.budding.states import INIT_Y0_MAP, initial_conds
    backend = BuddingSystem(); defaults = backend.defaults(); lookup = {k:i for i,k in enumerate(backend.PARAM_KEYS)}
    def digest(p, y, pairs, extras):
        raw = np.asarray(p, dtype='<f4').tobytes()+np.asarray(y, dtype='<f8').tobytes()
        return hashlib.sha256(raw+json.dumps([pairs, extras], sort_keys=True).encode()).digest()
    def extras(row):
        return {k:v for k,v in row.items() if k not in lookup and not k.startswith('init_')}
    pairs = [_pairs(row) for row in metadata]
    extra = [extras(row) for row in overrides]
    index = {digest(p,y,g,e):i for i,(p,y,g,e) in enumerate(zip(params,initials,pairs,extra))}
    rules = {}
    for gene, mode in sorted({pair for row in pairs for pair in row if pair[1]=='KO'}):
        rule = {}
        backend.apply_gene_mutation(gene, mode, 0., defaults, defaults.copy(), rule, rng=np.random.RandomState(0))
        rules[gene] = rule
    lower = {}
    for i, row in enumerate(pairs):
        if not any(mode in ('MC','GAL-MC') for _,mode in row):
            continue
        for gene, mode in row:
            if mode != 'KO' or sum(g==gene for g,_ in row) != 1:
                continue
            rule = rules[gene]
            if any(k not in lookup and k not in INIT_Y0_MAP for k in rule):
                continue
            # A different mutation touching the removed control would invalidate
            # the counterfactual. Be conservative, even for coincident WT values.
            remaining = tuple(pair for pair in row if pair[0]!=gene)
            from budding_yeast_v2.data.systems.budding.catalog import GROUP_BY_NAME
            other_keys = {k for g,_ in remaining for k in GROUP_BY_NAME[g]['params']}
            if set(rule) & other_keys:
                continue
            p, y = params[i].copy(), initials[i].copy()
            for key in rule:
                if key in lookup:
                    p[lookup[key]] = defaults[key]
                if key in INIT_Y0_MAP:
                    y[INIT_Y0_MAP[key]] = initial_conds()[INIT_Y0_MAP[key]]
            # The canonical initializer forces these totals' initial states.
            for key, variable in (('Tem1T',28),('Cdc15T',29)):
                if key in rule:
                    y[variable] = initial_conds()[variable]
            match = index.get(digest(p,y,remaining,extra[i]))
            if match is not None:
                lower[i,gene] = match
    blocks = set()
    for a, b in edges:
        changed = set(np.flatnonzero(params[a] != params[b]))
        controls = []
        for gene, mode in pairs[a]:
            if mode in ('MC','GAL-MC'):
                keys, _, _ = _strength_curve(gene, mode, '')
                if changed and changed <= {lookup[k] for k in keys}:
                    controls.append(gene)
        if len(controls) != 1:
            continue
        for gene, mode in pairs[a]:
            if mode == 'KO' and (a,gene) in lower and (b,gene) in lower:
                c, d = lower[a,gene], lower[b,gene]
                if len({int(a),int(b),c,d}) == 4 and np.array_equal(initials[c],initials[d]):
                    blocks.add((c,d,int(a),int(b)))
    return np.asarray(sorted(blocks), dtype=np.int64).reshape(-1,4)


def hierarchical_rank(scores, pool_keys, training_keys, budget, seed):
    """Feedback-weighted diminishing allocation: module set, mode, strength bin."""
    counts = [defaultdict(int) for _ in range(3)]
    def groups(key):
        pairs, environment, region = key
        modules = tuple(sorted({g for g,_ in pairs}))
        return (modules, (pairs, environment), (pairs, environment, region))
    for key in training_keys:
        for count, group in zip(counts, groups(key)):
            count[group] += 1
    def value(i):
        penalty = np.prod([1.+count[group] for count,group in zip(counts,groups(pool_keys[i]))])
        return float(scores[i])/np.sqrt(penalty)
    heap = [(-value(i), i) for i in range(len(scores))]; heapq.heapify(heap)
    selected = []; limit = budget-int(budget*SETTINGS['random_fraction'])
    while len(selected)<limit and heap:
        negative, i = heapq.heappop(heap)
        current = value(i)
        if heap and current < -heap[0][0]-1e-15:
            heapq.heappush(heap, (-current,i)); continue
        selected.append(i)
        for count, group in zip(counts,groups(pool_keys[i])):
            count[group] += 1
    records = [dict(kind='coverage',endpoints=[int(i)],new_queries=1) for i in selected]
    remaining = np.setdiff1d(np.arange(len(scores)),selected)
    for i in np.random.RandomState(seed).choice(remaining,budget-len(selected),replace=False):
        selected.append(int(i));records.append(dict(kind='random',endpoints=[int(i)],new_queries=1))
    return np.asarray(selected,dtype=np.int64),records
