"""One experiment seed, grouped cross-pool splits, and shared random initialization."""
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

from budding_yeast_v2.data.dataset_archive import TrajectoryStore, atomic_json, fingerprint
from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS, get_default_params


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 42
    model: str = 'fno'
    budgets: tuple = (2000, 4000, 6000, 8000, 10000)
    holdout_fraction: float = .05
    epochs: tuple = (100, 100, 100, 100, 200)
    warmup: tuple = (5, 5, 5, 5, 10)
    width: int = 192
    modes: int = 24
    n_blocks: int = 4
    batch_size: int = 128
    lr: float = 5e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    loss: str = 'rel_l2'
    relative_l2_eps: float = 0.01
    sketch_dim: int = 512
    prediction_batch_size: int = 128
    ensemble_members: int = 2
    validation_scope: str = 'source'
    validation_metric: str = 'normalized_loss'

    def validate(self):
        if self.validation_scope not in ('source', 'both'):
            raise ValueError('validation_scope must be source or both')
        if self.validation_metric not in ('normalized_loss','sample_channel_rel_l2','macro_trajectory_rel_l2'):
            raise ValueError('Unknown validation metric')
        if not np.isfinite(self.relative_l2_eps) or self.relative_l2_eps<=0:
            raise ValueError('Relative L2 eps must be finite and positive')
        if (len(self.budgets) != len(self.epochs) or len(self.epochs) != len(self.warmup)
                or not self.budgets or self.budgets[0] < 1
                or np.any(np.diff(self.budgets) <= 0)):
            raise ValueError('Budgets, epochs and warmup must describe increasing rounds')
        if not 0 < self.holdout_fraction < .5:
            raise ValueError('Invalid holdout fraction')
        if any(e < 1 or w < 0 or w >= e for e, w in zip(self.epochs, self.warmup)):
            raise ValueError('Each round needs positive training epochs beyond warmup')
        if min(self.width, self.modes, self.n_blocks, self.batch_size, self.sketch_dim) < 1:
            raise ValueError('Model and batch dimensions must be positive')
        if not 0 <= self.min_lr <= self.lr or self.ensemble_members < 2:
            raise ValueError('Invalid learning rate or ensemble size')
        return self


def child_seed(seed, purpose, *parts):
    """Stable independent streams; Python's process-randomized hash is not used."""
    code = int.from_bytes(hashlib.sha256(str(purpose).encode()).digest()[:4], 'little')
    return int(np.random.SeedSequence([int(seed), code, *map(int, parts)]).generate_state(1)[0])


def effective_signatures(params, initial):
    """Canonical complete-pool effective input identity, excluding init aliases.

    The complete-pool contract never generates #104's timed intervention. Actual
    protein IC overrides are represented by initial; BUB2 reset is separate.
    """
    keys = sorted(get_default_params())
    positions = {name: PARAM_KEYS.index(name) for name in keys}
    core = params[:, [positions[k] for k in keys]]
    reset = np.where(params[:, positions['BUB2_active']] == 0, 0,
                     params[:, PARAM_KEYS.index('init_BUB2')])
    vectors = np.column_stack((core, initial, reset, np.zeros(len(params))))
    return np.asarray([hashlib.sha256(row.astype('<f8').tobytes()).hexdigest()
                       for row in vectors])


def _definition(pools, config):
    config.validate()
    stores = {name: TrajectoryStore(path) for name, path in sorted(pools.items())}
    definitions = {name: dict(path=str(store.path), sha256=store.digest, rows=store.count)
                   for name, store in stores.items()}
    identity = dict(version='offline_complete_pools_v1', pools=definitions, config=asdict(config),
                    identity='canonical_complete_pool_effective_input_sha256_v1')
    return stores, definitions, identity


def protocol_location(pools, output, config=BenchmarkConfig()):
    _, _, identity = _definition(pools, config)
    return Path(output) / f'seed{config.seed}_{fingerprint(identity)[:12]}'


def prepare_protocol(pools, output, config=BenchmarkConfig()):
    import fcntl
    root = protocol_location(pools, output, config)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.prepare.lock').open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return _prepare_protocol(pools, output, config)


def _prepare_protocol(pools, output, config):
    stores, definitions, identity = _definition(pools, config)
    protocol_id = f'seed{config.seed}_{fingerprint(identity)[:12]}'
    root = Path(output) / protocol_id
    if (root / 'protocol.json').exists():
        import json
        existing = json.loads((root / 'protocol.json').read_text())
        if existing['fingerprint'] != fingerprint(identity):
            raise ValueError('Protocol identity changed')
        for path, digest in existing['artifacts'].items():
            from budding_yeast_v2.data.dataset_archive import file_sha256
            if file_sha256(root / path) != digest:
                raise ValueError(f'Protocol artifact changed: {path}')
        return root
    root.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    inputs = {}
    for name, store in stores.items():
        inputs[name] = store.inputs()
        signatures = effective_signatures(inputs[name]['params'], inputs[name]['y_init'])
        for row, signature in enumerate(signatures):
            groups[str(signature)].append((name, row))
    ordered = sorted(groups)
    rng = np.random.default_rng(child_seed(config.seed, 'split'))
    ordered = [ordered[i] for i in rng.permutation(len(ordered))]
    targets = {name: int(store.count * config.holdout_fraction) for name, store in stores.items()}
    roles = {name: np.zeros(store.count, dtype=np.int8) for name, store in stores.items()}
    assigned = set()
    for role in (1, 2):  # validation, test; all occurrences of one input share a role
        counts = {name: 0 for name in stores}
        for signature in ordered:
            if signature in assigned:
                continue
            members = groups[signature]
            increments = {name: sum(n == name for n, _ in members) for name in stores}
            if any(counts[name] + increments[name] > targets[name] for name in stores):
                continue
            for name, row in members:
                roles[name][row] = role
                counts[name] += 1
            assigned.add(signature)
    artifacts = {}
    from budding_yeast_v2.data.dataset_archive import file_sha256
    for pool_index, (name, store) in enumerate(stores.items()):
        candidate = np.flatnonzero(roles[name] == 0)
        if len(candidate) < config.budgets[-1]:
            raise ValueError('Selection universe cannot satisfy final budget')
        order = np.random.default_rng(child_seed(config.seed, 'initial', pool_index)).permutation(candidate)
        split = root / f'{name}_split.npz'
        np.savez_compressed(split, candidates=candidate, validation=np.flatnonzero(roles[name] == 1),
                            test=np.flatnonzero(roles[name] == 2), initial=order[:config.budgets[0]],
                            random_order=order)
        cache = root / f'{name}_inputs.npz'
        np.savez_compressed(cache, **inputs[name])
        artifacts[split.name] = file_sha256(split)
        artifacts[cache.name] = file_sha256(cache)
    report = dict(protocol_id=protocol_id, fingerprint=fingerprint(identity), **identity,
                  cross_pool_duplicate_groups=sum(len(v) > 1 for v in groups.values()),
                  actual_split_counts={name: dict(candidates=int((roles[name] == 0).sum()),
                    validation=int((roles[name] == 1).sum()), test=int((roles[name] == 2).sum())) for name in stores},
                  source_trajectories_copied=False, artifacts=artifacts)
    atomic_json(root / 'protocol.json', report)
    return root


def derive_training_protocol(source,output,config):
    """Reuse immutable inputs/splits for a training-only configuration change.

    Hard links are safe because protocol caches are verified and never mutated.
    No full-pool decompression, trajectory duplication or new random split.
    """
    import fcntl,json,os
    from budding_yeast_v2.data.dataset_archive import file_sha256
    source=Path(source).resolve()
    if source.is_file():source=source.parent
    previous=json.loads((source/'protocol.json').read_text())
    config.validate()
    for key in ('seed','budgets','holdout_fraction','validation_scope'):
        a,b=previous['config'][key],asdict(config)[key]
        if (list(a) if isinstance(a,(list,tuple)) else a)!=(list(b) if isinstance(b,(list,tuple)) else b):
            raise ValueError(f'Data/split setting changed: {key}; create a new protocol instead')
    pools={name:value['path'] for name,value in previous['pools'].items()}
    _,definitions,identity=_definition(pools,config)
    if definitions!=previous['pools']:raise ValueError('Frozen pool definitions changed')
    protocol_id=f'seed{config.seed}_{fingerprint(identity)[:12]}'
    root=Path(output)/protocol_id;root.mkdir(parents=True,exist_ok=True)
    with (root/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name,digest in previous['artifacts'].items():
            if Path(name).name!=name or file_sha256(source/name)!=digest:
                raise ValueError(f'Frozen source artifact changed: {name}')
            target=root/name
            if target.exists():
                if file_sha256(target)!=digest:raise ValueError(f'Derived artifact changed: {target}')
            else:os.link(source/name,target)
        report=dict(previous,**identity,protocol_id=protocol_id,fingerprint=fingerprint(identity),
                    derived_from=dict(path=str(source),protocol_sha256=file_sha256(source/'protocol.json')))
        if (root/'protocol.json').exists():
            existing=json.loads((root/'protocol.json').read_text())
            if existing['fingerprint']!=report['fingerprint']:raise ValueError('Derived protocol identity changed')
        else:atomic_json(root/'protocol.json',report)
    return root


def derive_repaired_protocol(source, pools, output, *, real_reference):
    """Rebind corrected truth views; require identical ordered inputs and splits."""
    import fcntl, json, os
    from budding_yeast_v2.data.dataset_archive import file_sha256
    source = Path(source).resolve()
    previous = json.loads((source / 'protocol.json').read_text())
    config = BenchmarkConfig(**previous['config']).validate()
    stores, definitions, identity = _definition(pools, config)
    if set(stores) != set(previous['pools']):
        raise ValueError('Pool set changed')
    for name, store in stores.items():
        old = TrajectoryStore(previous['pools'][name]['path'])
        if old.digest != previous['pools'][name]['sha256'] or old.count != store.count:
            raise ValueError('Original pool changed')
        for key in ('params','y_init','candidate_ids','mutant_names','sample_id','is_real','overrides_json'):
            a, b = old.fields[key], store.fields[key]
            if any(a[item] != b[item] for item in ('sha256','shape','dtype')):
                raise ValueError(f'Repair changed ordered input field: {name}/{key}')
    protocol_id = f'seed{config.seed}_{fingerprint(identity)[:12]}'
    root = Path(output) / protocol_id
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, digest in previous['artifacts'].items():
            if Path(name).name != name or file_sha256(source / name) != digest:
                raise ValueError(f'Changed input/split cache: {name}')
            target = root / name
            if target.exists():
                if file_sha256(target) != digest:
                    raise ValueError(f'Changed derived cache: {target}')
            else:
                os.link(source / name, target)
        report = dict(previous, **identity, protocol_id=protocol_id, fingerprint=fingerprint(identity),
            derived_from=dict(path=str(source), protocol_sha256=file_sha256(source/'protocol.json'),
                              reason='corrected_trajectories_identical_inputs_and_splits'),
            real_reference=dict(path=str(Path(real_reference).resolve()), sha256=file_sha256(real_reference)))
        path = root / 'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != report:
                raise ValueError('Repaired protocol already exists with different provenance')
        else:
            atomic_json(path, report)
    return root
