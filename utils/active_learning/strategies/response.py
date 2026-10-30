"""Ensemble disagreement about within-background mutation responses.

Only pool inputs and predictions enter acquisition. Synthetic training inputs can
serve as free anchors; real benchmarks and true trajectories are never scored.
"""
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

SETTINGS = dict(version='response_uncertainty_v1', neighbors=3, time_points=32,
                exploration_fraction=.1, distance_weight='1/(1+RMS_CDF_distance)',
                point_uncertainty='full_trajectory_loss_mse', cdf_grid_size=4097)


class _CoordinateRNG:
    def __init__(self, u, branch):
        self.u, self.branch = u, branch
    def random(self):
        return .25 if self.branch == 'synthesis' else .75
    def uniform(self, low=0., high=1.):
        return low + (high-low)*self.u


@lru_cache(maxsize=None)
def _strength_curve(gene, mode, branch):
    """Invert the actual mutation rule, including internally drawn alleles.

    One coordinate per mutation control, rather than counting linked rates twice.
    No duplicate list of biological rate/range constants is maintained here.
    """
    from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
    backend = BuddingSystem()
    defaults = backend.defaults()
    u = np.linspace(0., 1., SETTINGS['cdf_grid_size'])
    # Fixed alleles have no strength coordinate.
    if mode in ('KO', 'CA', 'PM', 'CSL', 'CKO', 'pDEL'):
        return (), None, None
    rows = []
    for coordinate in u:
        alpha = backend.sample_scaling_factor_sobol(mode, gene, float(coordinate))
        overrides = {}
        backend.apply_gene_mutation(gene, mode, alpha, defaults, defaults.copy(),
                                    overrides, rng=_CoordinateRNG(coordinate, branch))
        rows.append(overrides)
    keys = sorted(key for key in rows[0] if key in backend.PARAM_KEYS
                  and rows[0][key] != rows[-1][key])
    if not keys:
        return (), None, None
    key = keys[0]
    values = np.array([row[key] for row in rows], dtype=np.float64)
    if mode in ('MC','GAL-MC'):
        # Mid-CDF for discrete 1/N copies, including the endpoint atoms.
        from budding_yeast_v2.data.systems.budding.distributions.ranges import copy_support,GAL_MC_BASE
        copies = copy_support(gene,mode)
        probability = 1./copies
        probability /= probability.sum()
        scale=GAL_MC_BASE if mode=='GAL-MC' else defaults[key]
        return tuple(keys), (copies * scale), (np.cumsum(probability)-probability/2)
    unique, first, counts = np.unique(values, return_index=True, return_counts=True)
    if len(unique) < 2 or not (np.all(np.diff(values) >= 0) or np.all(np.diff(values) <= 0)):
        raise ValueError(f'Nonmonotone response strength rule: {gene}/{mode}/{branch}')
    # Plateaus (clipped normal tails) receive their probability midpoint.
    coordinates = np.array([u[start:start+count].mean() for start,count in zip(first,counts)])
    return tuple(keys), unique, coordinates


def _pairs(meta):
    pairs = meta.get('pairs', meta.get('genes', ()))
    return tuple(sorted((str(g), str(m)) for g,m in pairs))


def background_graph(params, initials, overrides, metadata, pool_size):
    """Sparse near + far contrasts; exact fixed inputs form a hard background.

    Float64 ICs are compared exactly (conservative); all parameters not changed
    by a strength control, and all nonparameter event switches, must also match.
    """
    from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS, get_default_params
    lookup = {key:i for i,key in enumerate(PARAM_KEYS)}
    defaults = get_default_params()
    if (len(params) != len(initials) or len(params) != len(overrides)
            or len(params) != len(metadata) or not 0 <= pool_size <= len(params)
            or not np.isfinite(params).all() or not np.isfinite(initials).all()):
        raise ValueError('Response background inputs must be finite and aligned')
    groups = defaultdict(list)
    coordinates = []
    for i,(p,y,ovr,meta) in enumerate(zip(params,initials,overrides,metadata)):
        pairs = _pairs(meta)
        if not pairs:
            raise ValueError('Response acquisition requires structured mutation pairs')
        branches, dims, active = [], [], set()
        for gene,mode in pairs:
            branch = ''
            if gene == 'CDC14' and mode == 'ts':
                branch = 'synthesis' if float(p[lookup['ks14']]) < .1*defaults['ks14'] else 'activity'
            keys, values, probability = _strength_curve(gene, mode, branch)
            branches.append(branch)
            active.update(keys)
            if keys:
                dims.append(float(np.interp(p[lookup[keys[0]]], values, probability)))
        fixed = np.asarray(p, dtype='<f4').copy()
        for key in active:
            fixed[lookup[key]] = 0.
        extras = {key:value for key,value in ovr.items()
                  if key not in lookup and not key.startswith('init_')}
        digest = hashlib.sha256(fixed.tobytes()+np.asarray(y,dtype='<f8').tobytes()
                                 +json.dumps(extras,sort_keys=True).encode()).digest()
        key = (pairs, tuple(branches), digest)
        groups[key].append(i)
        coordinates.append(dims)
    edges = set()
    for rows in groups.values():
        if len(rows) < 2 or not coordinates[rows[0]]:
            continue
        values = np.asarray([coordinates[i] for i in rows])
        tree = cKDTree(values)
        k = min(SETTINGS['neighbors']+1, len(rows))
        _, neighbors = tree.query(values, k=k)
        # Include opposite extremes on every control: far contrasts remain legal.
        extremes = np.unique(np.r_[values.argmin(0), values.argmax(0)])
        for local,i in enumerate(rows):
            if i >= pool_size:
                continue  # No anchor-anchor query.
            candidates = set(np.atleast_1d(neighbors[local]).tolist()) | set(extremes.tolist())
            for other in candidates:
                j = rows[other]
                if i != j:
                    edges.add((min(i,j),max(i,j)))
    edges = np.array(sorted(edges), dtype=np.int64).reshape(-1,2)
    distances = np.array([np.sqrt(np.mean((np.asarray(coordinates[a])-coordinates[b])**2))
                          for a,b in edges], dtype=np.float32)
    return edges, distances


def prediction_features(engine, dm, device, anchor_count, batch_size=128,
                        observer=None, keep_features=True, compact_two_members=False):
    """Stream full point variance and 32-time-point centered ensemble features."""
    members = engine._ensemble_members
    if members is None or len(members) < 2:
        raise RuntimeError('ours_response requires independently trained ensemble members')
    modes = [m.training for m in members]
    devices = [next(m.parameters(),torch.empty(0)).device for m in members]
    n = engine._pool_size + anchor_count
    point = np.empty(n,dtype=np.float32)
    feature = None
    train_x,train_p = dm.train_dataset_synth.tensors[:2]
    try:
        for member in members:
            member.to(device).eval()
        with torch.no_grad():
            for start in range(0,n,batch_size):
                stop = min(start+batch_size,n)
                # Keep the pool/train boundary out of an individual batch.
                for left,right in ((start,min(stop,engine._pool_size)),
                                   (max(start,engine._pool_size),stop)):
                    if left >= right:
                        continue
                    if left < engine._pool_size:
                        ids = np.arange(left,right)
                        x,p = engine._get_pool_X_batch(ids),engine._pool_P[ids]
                    else:
                        ids = np.arange(left-engine._pool_size,right-engine._pool_size)
                        x,p = train_x[ids],train_p[ids]
                    outputs = torch.stack([m.forward_ic_time(x.to(device),p.to(device)).float() for m in members])
                    if not torch.isfinite(outputs).all():
                        raise ValueError('Nonfinite response prediction')
                    if observer is not None:
                        observer(left, right, outputs)
                    outputs -= outputs.mean(0,keepdim=True)
                    point[left:right] = outputs.square().flatten(2).mean((0,2)).cpu().numpy()
                    times = torch.linspace(0,outputs.shape[-1]-1,min(SETTINGS['time_points'],outputs.shape[-1]),device=device).round().long()
                    sampled = outputs[...,times].flatten(2).cpu().numpy()
                    if compact_two_members and len(members)==2:
                        # Two centered members carry one independent contrast.
                        sampled = (sampled[:1]-sampled[1:])*.5
                    if keep_features:
                        if feature is None:
                            feature = np.empty((len(sampled),n,sampled.shape[-1]),dtype=np.float32)
                        feature[:,left:right] = sampled
                if (start//batch_size) % 100 == 0 or stop == n:
                    print(f'[ours_response] ensemble predictions {stop}/{n}',flush=True)
    finally:
        for member,mode,original_device in zip(members,modes,devices):
            member.to(original_device).train(mode)
    return point,feature


def rank_queries(point, features, edges, distances, pool_size, budget, seed):
    """Compare points and pairs per new oracle call; count shared endpoints once."""
    if (not 0 <= budget <= pool_size or not np.isfinite(point).all()
            or not np.isfinite(features).all() or not np.isfinite(distances).all()):
        raise ValueError('Invalid response scores or query budget')
    pair_variance = np.empty(len(edges),dtype=np.float32)
    for start in range(0,len(edges),512):
        part = edges[start:start+512]
        delta = features[:,part[:,1]]-features[:,part[:,0]]
        pair_variance[start:start+len(part)] = np.mean(delta**2,axis=(0,2))
    cost = (edges < pool_size).sum(1)
    # A known anchor contributes response information, not a paid point reward.
    pair_score = ((point[edges[:,0]]*(edges[:,0] < pool_size)
                   + point[edges[:,1]]*(edges[:,1] < pool_size)
                   + pair_variance/(1+distances))/cost)
    # Point utility includes only points that still need an oracle query.
    utilities = np.r_[point[:pool_size],pair_score]
    order = np.argsort(-utilities,kind='stable')
    random_count = min(budget,int(np.floor(budget*SETTINGS['exploration_fraction'])))
    selected, chosen, audit = [], set(), []
    for action in order:
        if len(selected) >= budget-random_count:
            break
        endpoints = (int(action),) if action < pool_size else tuple(map(int,edges[action-pool_size]))
        new = [i for i in endpoints if i < pool_size and i not in chosen]
        if not new or len(selected)+len(new) > budget-random_count:
            continue
        selected.extend(new);chosen.update(new)
        record = dict(kind='point' if action < pool_size else 'response',endpoints=endpoints,
                      score=float(utilities[action]),new_queries=len(new))
        if action >= pool_size:
            j = action-pool_size
            record.update(distance=float(distances[j]),response_variance=float(pair_variance[j]))
        audit.append(record)
    remaining = np.setdiff1d(np.arange(pool_size),selected)
    extra = np.random.RandomState(seed).choice(remaining,budget-len(selected),replace=False)
    for i in extra:
        selected.append(int(i));audit.append(dict(kind='random',endpoints=(int(i),),new_queries=1))
    return np.asarray(selected,dtype=np.int64),audit


def _ours_response_sample(self, trainer, pl_module, dm, device, current_epoch):
    if not self._pool_size:
        return None
    # The provenance cache contains synthetic TRAINING rows only, never held-outs.
    provenance = getattr(self,'_cum_synth_provenance',[])
    anchor_count = len(provenance)
    if anchor_count and anchor_count != len(dm.train_dataset_synth):
        raise ValueError('Response training-anchor provenance alignment error')
    params = self._pool_P_real.detach().cpu().numpy()
    initials = self._pool_initial_physical.detach().cpu().numpy()
    overrides,metadata = list(self._pool_ovr),list(self._pool_mass_meta)
    if anchor_count:
        params = np.concatenate((params,np.asarray(self._cum_synth_params,dtype=np.float32)))
        initials = np.concatenate((initials,np.asarray([r['y_init'] for r in provenance])))
        overrides += [json.loads(r['overrides_json']) for r in provenance]
        metadata += [json.loads(r['candidate_metadata_json']) for r in provenance]
    print(f'[ours_response] building contrasts for {self._pool_size} candidates + {anchor_count} training anchors',flush=True)
    started = time.monotonic()
    edges,distances = background_graph(params,initials,overrides,metadata,self._pool_size)
    graph_seconds = time.monotonic()-started
    print(f'[ours_response] {len(edges)} contrasts in {graph_seconds:.1f}s; predicting ensemble',flush=True)
    started = time.monotonic()
    point,features = prediction_features(self,dm,device,anchor_count)
    prediction_seconds = time.monotonic()-started
    budget = self._al4pde_scheduled_num_add()
    seed = int(self.random_seed+1009*getattr(self,'_al4pde_al_iter',0)+self.pool.revision)
    started = time.monotonic()
    selected,records = rank_queries(point,features,edges,distances,self._pool_size,budget,seed)
    ranking_seconds = time.monotonic()-started
    ids = self.pool.ensure_ids()
    for record in records:
        record['endpoints'] = [dict(kind='pool',candidate_id=int(ids[i])) if i < self._pool_size
                               else dict(kind='training_anchor',training_row=int(i-self._pool_size),
                                         candidate_id=int(provenance[i-self._pool_size].get('candidate_id',-1)))
                               for i in record['endpoints']]
    if not hasattr(self,'_response_acquisitions'):
        self._response_acquisitions = defaultdict(list)
        self._response_timings = defaultdict(list)
    round_index = getattr(self,'_al4pde_al_iter',0)
    self._response_acquisitions[round_index].extend(records)
    self._response_timings[round_index].append(dict(graph_seconds=graph_seconds,
        prediction_seconds=prediction_seconds,ranking_seconds=ranking_seconds,
        pool_size=int(self._pool_size),anchors=int(anchor_count),contrasts=len(edges),budget=int(budget)))
    print(f'[ours_response] backgrounds: {len(edges)} contrasts, {anchor_count} train anchors; query {len(selected)}')
    return self.make_selection_plan(selected,dm)


def write_response_audit(engine, directory, batch):
    records = getattr(engine,'_response_acquisitions',{}).get(engine._al4pde_al_iter-1)
    if records is None:
        return
    accepted = set(map(int,batch.get('candidate_ids',())))
    for record in records:
        for endpoint in record['endpoints']:
            endpoint['status'] = ('known_training' if endpoint['kind']=='training_anchor' else
                                  'accepted' if endpoint['candidate_id'] in accepted else 'not_accepted')
    report = dict(settings=SETTINGS,choices=records,
                  timings=getattr(engine,'_response_timings',{}).get(engine._al4pde_al_iter-1,[]),
                  response_actions=sum(r['kind']=='response' for r in records))
    # Serialize before touching the destination: an audit failure must not leave
    # a truncated JSON file that looks like a completed acquisition report.
    encoded = json.dumps(report,indent=2)
    destination = Path(directory)/'response_pairs.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(encoded,encoding='utf-8')
    temporary.replace(destination)
