"""Seeded random splits and per-L KMeans initialization for both yeast systems."""
import argparse
from collections import Counter
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .contracts import Candidate, GenerationConfig
from .dataset_io import write_dataset
from .generation import level_schedule, run_simulations
from .registry import get_adapter, resolve_config
from .sampling_policy import POOL_RATIOS, VERSION, valid_candidate


SPLIT_POLICY = 'seeded_random_80_10_10_v1'


def pool_identity(path):
    path=Path(path).resolve();stat=path.stat()
    with np.load(path,allow_pickle=False) as pool:
        if str(pool.get('system_id',''))=='budding':
            from .budding.al_pool import validate_pool_contract
            validate_pool_contract(pool)
        return dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                    contract=str(pool.get('pool_generation_contract','')),
                    catalog=str(pool.get('catalog_version','')))


def random_split(indices, seed):
    """No Pattern, BIO or L information enters the permutation."""
    order = np.random.RandomState(seed).permutation(np.asarray(indices, dtype=np.int64))
    n_train, n_val = int(len(order)*.8), int(len(order)*.1)
    return order[:n_train], order[n_train:n_train+n_val], order[n_train+n_val:]


def select_representatives(params, count, seed, mean=None, std=None):
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count > len(params):
        raise ValueError('Insufficient candidates for KMeans quota')
    # Every actual candidate in a disjoint cluster has a unique representative.
    std = params.std(axis=0) if std is None else std
    mean = params.mean(axis=0) if mean is None else mean
    features = (params-mean)/np.where(std < 1e-8, 1., std)
    km = MiniBatchKMeans(n_clusters=count, random_state=seed, n_init=3,
                        batch_size=max(count, min(10000,len(params))), max_iter=100)
    labels = km.fit_predict(features)
    chosen = []
    for cluster in range(count):
        rows = np.flatnonzero(labels == cluster)
        if len(rows):
            chosen.append(int(rows[np.argmin(np.sum((features[rows]-km.cluster_centers_[cluster])**2,axis=1))]))
    if len(chosen) < count:
        remaining = np.setdiff1d(np.arange(len(params)), chosen)
        chosen.extend(np.random.RandomState(seed).choice(remaining,count-len(chosen),replace=False).tolist())
    return np.asarray(chosen, dtype=np.int64)


def validate_existing_initial(path, pool_path, train_samples, seed):
    """Refuse stale initialization rules before a launcher reuses solved data."""
    with np.load(path, allow_pickle=False) as saved:
        report = json.loads(str(saved['generation_report']))
        expected = dict(initial_parameter_transform='zscore',
                        initial_parameter_fit_scope='full_source_pool',
                        split_policy=SPLIT_POLICY, split_seed=seed)
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError('Existing initial dataset uses different initialization rules; '
                             'choose a new INIT_DATASET path to regenerate it')
        if Path(report['source_pool']).resolve() != Path(pool_path).resolve():
            raise ValueError('Existing initial dataset belongs to a different source pool')
        if Path(pool_path).exists() and report.get('source_pool_identity')!=pool_identity(pool_path):
            raise ValueError('Existing initial dataset source pool changed; choose a new INIT_DATASET path')
        if len(saved['split_train']) != train_samples or int(saved['split_seed']) != seed:
            raise ValueError('Existing initial dataset train size/split seed differs from this run')
    return str(path)


def generate_initial(pool_path, output, train_samples=None, seed=42, workers=1):
    """Select all initial trajectories first, simulate/refill, then split once."""
    output = Path(output)
    if output.exists():raise FileExistsError(output)
    with np.load(pool_path,allow_pickle=False) as pool:
        system = str(pool['system_id'])
        if system=='budding':
            from .budding.al_pool import validate_pool_contract
            validate_pool_contract(pool)
        if str(pool['artifact_kind']) != 'parameter_pool' or str(pool['sampling_policy']) != VERSION:
            raise ValueError('Use a current exact-L parameter pool')
        params, levels = pool['params'], pool['module_counts']
        pool_mean,pool_std = pool['p_mean'],pool['p_std']
        names, edits = pool['mutant_names'], pool['overrides_json']
        metadata = pool['candidate_metadata_json']
        param_names, grid = pool['param_names'], pool['time_grid']
        mutation,distribution = str(pool['mutation_profile']),str(pool['distribution_profile'])
        if system=='fission':
            from .fission.al_pool import INITIAL_STATE_PROTOCOL
            if json.loads(str(pool['generation_report'])).get('initial_state_protocol')!=INITIAL_STATE_PROTOCOL:
                raise ValueError('Fission initialization requires a native warmed input pool')
        if 'data' in pool:raise ValueError('Initialization expects unsolved candidates')
    adapter = get_adapter(system)
    if tuple(param_names) != adapter.spec.param_names or not np.allclose(grid,np.linspace(0,adapter.spec.horizon,adapter.spec.steps)):
        raise ValueError('Pool schema/horizon disagrees with biological model')
    if (params.shape!=(len(names),len(param_names)) or levels.shape!=(len(names),)
            or len(edits)!=len(names) or len(metadata)!=len(names) or not np.isin(levels,[1,2,3]).all()
            or not np.isfinite(params).all()):
        raise ValueError('Malformed initial candidate pool rows')
    train_samples = (10000 if system=='budding' else 5000) if train_samples is None else train_samples
    if train_samples < 1 or train_samples % 4:
        raise ValueError('Initial train size must be a positive multiple of 4 for exact 80/10/10 counts')
    total = train_samples*5//4
    config = resolve_config(GenerationConfig('budding_random' if system=='budding' else 'fission_random',
        n_samples=total,seed=seed,workers=workers,output=str(output),level_ratios=POOL_RATIOS,
        mutation_profile=mutation if system=='budding' else None,
        distribution_profile=distribution if system=='budding' else None))
    target = Counter(level_schedule(total,POOL_RATIOS,np.random.RandomState(seed)).tolist())
    selected = []
    for level, count in sorted(target.items()):
        print(f"[{system} initial] KMeans L{level}: selecting {count}",flush=True)
        indices = np.flatnonzero(levels==level)
        selected.extend(indices[select_representatives(params[indices],count,seed+level,pool_mean,pool_std)].tolist())
    used, accepted, counts = set(), [], Counter()
    report = Counter()
    context = adapter.prepare_context(config)
    rng = np.random.RandomState(seed)
    while selected:
        candidates = []
        for i in selected:
            meta = json.loads(str(metadata[i]))
            meta.update(source_candidate_id=int(i), source_pool=str(Path(pool_path).resolve()))
            candidate = valid_candidate(Candidate(int(i)+1,str(names[i]),json.loads(str(edits[i])),int(levels[i]),metadata=meta),system)
            if candidate is None:raise ValueError('Pool contains an invalid mutation/module assignment')
            # Reject historical real collisions; do not change scored pool inputs.
            if system == 'budding':
                from .budding.real_mutants import is_duplicate_real_mutant
                from .budding.specification import BuddingSystem
                if is_duplicate_real_mutant(candidate.overrides,BuddingSystem().defaults()):
                    raise ValueError('Pool contains a real collision; regenerate it')
            candidates.append(candidate)
        print(f"[{system} initial] simulating {len(candidates)} on {workers} workers",flush=True)
        used.update(selected)
        for row in run_simulations(candidates,config,context):
            if (row.status=='success' and row.trajectory is not None
                    and row.trajectory.shape==(len(adapter.spec.variable_names),adapter.spec.steps)
                    and np.isfinite(row.trajectory).all()):
                accepted.append(row);counts[row.candidate.level]+=1
            else:report['simulation_failures']+=1
        missing = {level:count-counts[level] for level,count in target.items() if count>counts[level]}
        if not missing:break
        if len(used)>=30*total:raise RuntimeError(f'Initial same-L quota exhausted: {missing}')
        selected = []
        for level,count in sorted(missing.items()):
            available = np.array([i for i in np.flatnonzero(levels==level) if i not in used],dtype=np.int64)
            if len(available)<count:raise RuntimeError(f'Initial candidate pool exhausted at L{level}')
            selected.extend(rng.choice(available,count,replace=False).tolist())
    accepted.sort(key=lambda row:row.candidate.sample_id)
    real = list(run_simulations(adapter.real_candidates(True),config,context))
    if len(real)!=adapter.expected_real_count() or any(r.status!='success' for r in real):
        raise RuntimeError('Incomplete real benchmark; refusing initial dataset')
    dict.update(report,recipe=config.recipe,mutation_profile=mutation,
        distribution_profile=distribution,
        rng_contract='initial_kmeans_seeded_split_v1',sampling_policy=VERSION,
        initial_selection='per_L_kmeans',initial_parameter_transform='zscore',
        initial_parameter_fit_scope='full_source_pool',
        split_policy=SPLIT_POLICY,split_seed=seed,
        source_pool=str(Path(pool_path).resolve()),selected_candidate_ids=[r.candidate.sample_id-1 for r in accepted])
    report['source_pool_identity']=pool_identity(pool_path)
    # The regular writer preserves canonical biology, L metadata and labels.
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.npz',delete=False) as handle:
        temporary=Path(handle.name)
    temporary.unlink()
    try:
        write_dataset(temporary,adapter.spec,real+accepted,config,dict(report),dtype='float32')
        with np.load(temporary,allow_pickle=False) as saved:fields={k:saved[k] for k in saved.files}
        train,val,test = random_split(np.arange(len(real),len(real)+len(accepted)),seed)
        fields.update(split_policy=np.array(SPLIT_POLICY),split_seed=np.int64(seed),
            split_train=train,split_val=val,split_test=test,split_real=np.arange(len(real),dtype=np.int64),
            initial_consumed_names=np.asarray(names[sorted(used)]))
        np.savez_compressed(temporary,**fields)
        os.link(temporary,output)  # Atomic publication; never overwrite a run.
    finally:temporary.unlink(missing_ok=True)
    return str(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--train-samples',type=int)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--check-existing',action='store_true',
                        help='Validate and reuse an existing initial dataset; generate if absent')
    args=parser.parse_args()
    if args.check_existing and Path(args.output).exists():
        if args.train_samples is None:parser.error('--check-existing requires --train-samples')
        print(validate_existing_initial(args.output,args.pool,args.train_samples,args.seed))
        return
    print(generate_initial(args.pool,args.output,args.train_samples,args.seed,args.workers))


if __name__=='__main__':main()
