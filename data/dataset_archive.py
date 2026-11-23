"""Lossless dataset views over immutable NPZ blocks.

The manifest is a storage format only: consumers still see the same named numpy
arrays, in the same order and dtype, as a materialized round dataset.
"""
from collections.abc import Mapping
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np


FORMAT = 'yeast_dataset_manifest'
_HASH_CACHE = {}


def file_sha256(path):
    path = Path(path).resolve()
    st = path.stat()
    identity = (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if identity not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b''):
                digest.update(block)
        _HASH_CACHE[identity] = digest.hexdigest()
    return _HASH_CACHE[identity]


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def resolve_dataset_path(path):
    """Old config/checkpoint paths ending in dataset.npz also resolve after compaction."""
    path = Path(path)
    if path.is_file():
        return path
    if path.suffix == '.npz' and path.with_suffix('.json').is_file():
        return path.with_suffix('.json')
    raise FileNotFoundError(path)


def open_dataset(path, *, allow_pickle=True):
    path = resolve_dataset_path(path)
    if path.suffix == '.json':
        with path.open() as handle:
            kind = json.load(handle).get('format')
        if kind == 'yeast_sampling_selection':
            return SelectionArchive(path)
        return ManifestArchive(path, allow_pickle=allow_pickle)
    return np.load(path, allow_pickle=allow_pickle)


def round_dataset_paths(run):
    """One dataset per round, including compacted historical experiments."""
    result = []
    for directory in sorted((Path(run) / 'rounds').glob('round_*')):
        for name in ('dataset.npz', 'dataset.json'):
            if (directory / name).is_file():
                result.append(directory / name)
                break
    return result


def dataset_field_chunks(path, key='data', chunk_size=512):
    """Read trajectories for analysis without materializing a cumulative dataset."""
    path = resolve_dataset_path(path)
    if path.suffix == '.json':
        with open_dataset(path) as archive:
            if isinstance(archive, SelectionArchive):
                for start in range(0, len(archive.selection_rows), chunk_size):
                    yield start, archive.store.read(archive.selection_rows[start:start + chunk_size], (key,))[key]
                return
            spec = archive.manifest['fields'][key]
            pieces = spec.get('pieces')
            if pieces is None:
                pieces = [dict(source=spec['source'], rows=None)]
            offset = 0
            for piece in pieces:
                if isinstance(piece, int):
                    piece = archive.manifest['segments'][piece]
                values = archive._source(piece['source'])[piece.get('key', key)]
                indices = _rows(piece.get('rows'), len(values))
                for start in range(0, len(indices), chunk_size):
                    yield offset+start, values[indices[start:start+chunk_size]].astype(spec['dtype'], copy=False)
                offset += len(indices)
                del values
        return
    with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = reader(stream)
        if not shape or fortran or dtype.hasobject:
            raise ValueError('Chunk reading requires a non-object C-order row array')
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        for start in range(0, shape[0], chunk_size):
            count = min(chunk_size, shape[0]-start)
            encoded = stream.read(count*row_bytes)
            if len(encoded) != count*row_bytes:
                raise ValueError('Truncated dataset array')
            yield start, np.frombuffer(encoded, dtype=dtype).reshape(count, *shape[1:])


def read_dataset_rows(path, rows, key='data'):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or (rows < 0).any():
        raise ValueError('Supply a nonempty one-dimensional set of row indices')
    source=resolve_dataset_path(path)
    if source.suffix=='.json' and json.loads(source.read_text()).get('format')==FORMAT:
        return TrajectoryStore(source).read(rows,(key,))[key]
    result, copied = None, 0
    for start, block in dataset_field_chunks(path, key):
        if result is None:
            result = np.empty((len(rows), *block.shape[1:]), dtype=block.dtype)
        selected = np.flatnonzero((rows >= start) & (rows < start+len(block)))
        result[selected] = block[rows[selected]-start]
        copied += len(selected)
        if copied == len(rows):
            return result
    raise ValueError('Requested row is outside the dataset')


def shared_initial_archive(source, experiment_dir):
    """One immutable, content-addressed initial block per experiment family."""
    source = resolve_dataset_path(source).resolve()
    if source.suffix != '.npz':
        raise ValueError('AL initial input must be a materialized trajectory NPZ')
    experiment_dir = Path(experiment_dir).resolve()
    family = next((p for p in experiment_dir.parents if p.name in ('al_budding', 'al_fission')),
                  experiment_dir.parent)
    root = Path(os.environ.get('AL_SHARED_DATA_DIR', family / 'shared' / 'datasets'))
    root.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(source)
    target = root / (digest + '.npz')
    if target.exists():
        if file_sha256(target) != digest:
            raise ValueError(f'Corrupt shared initial dataset: {target}')
        return str(target.resolve())
    fd, temporary = tempfile.mkstemp(prefix='.initial-', suffix='.npz', dir=root)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        with open(temporary, 'rb') as handle:
            os.fsync(handle.fileno())
        if file_sha256(temporary) != digest:
            raise ValueError('Initial dataset changed while copying')
        os.chmod(temporary, 0o444)
        # Atomic publish without overwriting a concurrent producer's block.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if file_sha256(target) != digest:
                raise ValueError(f'Corrupt shared initial dataset: {target}')
        _sync_directory(root)
    finally:
        os.unlink(temporary)
    return str(target.resolve())


def _array_digest(array):
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(str(array.shape).encode())
    if array.dtype.hasobject:
        digest.update(json.dumps(array.tolist(), ensure_ascii=False, default=str).encode())
    elif array.size:
        digest.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return digest.hexdigest()


def _equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.hasobject:
        return np.array_equal(a, b)
    if a.ndim == 0:
        return a.tobytes() == b.tobytes()
    return all(a[i:i+256].tobytes() == b[i:i+256].tobytes()
               for i in range(0, len(a), 256))


def _rows(selector, count):
    if selector is None:
        return np.arange(count, dtype=np.int64)
    result = np.asarray(selector, dtype=np.int64)
    if result.ndim != 1 or ((result < 0) | (result >= count)).any():
        raise ValueError('Manifest row indices are outside their source block')
    return result


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ManifestArchive(Mapping):
    def __init__(self, path, *, allow_pickle=True):
        self.path = Path(path).resolve()
        with self.path.open() as handle:
            self.manifest = json.load(handle)
        if (self.manifest.get('format') != FORMAT or self.manifest.get('version') != 1):
            raise ValueError(f'Unsupported dataset manifest: {self.path}')
        self.files = list(self.manifest['fields'])
        self._stack = ExitStack()
        self._archives = {}
        self._allow_pickle = allow_pickle

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __contains__(self, key):
        return key in self.manifest['fields']

    def _source(self, source_id):
        if source_id not in self._archives:
            source = self.manifest['sources'][source_id]
            path = (self.path.parent / source['path']).resolve()
            if path.suffix != '.npz' or file_sha256(path) != source['sha256']:
                raise ValueError(f'Dataset block is missing/changed: {path}')
            self._archives[source_id] = self._stack.enter_context(
                np.load(path, allow_pickle=self._allow_pickle))
        return self._archives[source_id]

    def __getitem__(self, key):
        spec = self.manifest['fields'][key]
        dtype, shape = np.dtype(spec['dtype']), tuple(spec['shape'])
        if 'source' in spec:
            result = self._source(spec['source'])[spec.get('key', key)]
        else:
            result = np.empty(shape, dtype=dtype)
            offset = 0
            for piece in spec['pieces']:
                if isinstance(piece, int):
                    piece = self.manifest['segments'][piece]
                array = self._source(piece['source'])[piece.get('key', key)]
                indices = _rows(piece.get('rows'), len(array))
                for start in range(0, len(indices), 256):
                    chunk = indices[start:start+256]
                    result[offset+start:offset+start+len(chunk)] = array[chunk]
                offset += len(indices)
                del array
            if offset != shape[0]:
                raise ValueError(f'Manifest row count mismatch for {key}')
        if result.shape != shape or result.dtype != dtype or _array_digest(result) != spec['sha256']:
            raise ValueError(f'Dataset field checksum/schema mismatch: {key}')
        return result

    def close(self):
        self._stack.close()
        self._archives.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SelectionArchive(Mapping):
    """Ordered complete-pool row references, readable without dataset duplication.

    Selected training snapshots explicitly keep all referenced rows in train;
    legacy DataModule consumers must not silently split them a second time.
    """
    def __init__(self, path):
        self.path = Path(path).resolve()
        value = json.loads(self.path.read_text())
        if value.get('version') != 1:
            raise ValueError('Unsupported sampling selection version')
        self.store = TrajectoryStore((self.path.parent / value['source']).resolve())
        if self.store.digest != value['source_sha256']:
            raise ValueError('Sampling selection source has changed')
        self.selection_rows = np.asarray(value['rows'], dtype=np.int64)
        if (self.selection_rows.ndim != 1 or (self.selection_rows < 0).any()
                or (self.selection_rows >= self.store.count).any()
                or len(np.unique(self.selection_rows)) != len(self.selection_rows)):
            raise ValueError('Invalid sampling selection row identities')
        n = len(self.selection_rows)
        self.overrides = dict(dataset_role=np.array('selected_training_reference'),
            split_policy=np.array('offline_selected_training'),
            split_seed=np.array(value.get('provenance', {}).get('seed', 42)),
            split_train=np.arange(n, dtype=np.int64),
            split_val=np.array([], dtype=np.int64), split_test=np.array([], dtype=np.int64),
            split_real=np.array([], dtype=np.int64))
        if value.get('weights') is not None:
            weights = np.asarray(value['weights'], dtype=np.float64)
            if weights.shape != (n,) or not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError('Invalid sampling selection weights')
            self.overrides['sample_weights'] = weights
        self.files = list(dict.fromkeys([*self.store.fields, *self.overrides]))

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __contains__(self, key):
        return key in self.files

    def __getitem__(self, key):
        if key in self.overrides:
            return self.overrides[key]
        if key not in self.store.fields:
            raise KeyError(key)
        return self.store.read(self.selection_rows, (key,))[key]

    def close(self):
        self.store._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_dataset_manifest(path, arrays, segments, *, provenance=None):
    """Reference matching fields; store only differing metadata locally.

    segments: ordered (NPZ path, row indices or None) covering all target rows.
    Every reference is checked against the target, including dtype and row order.
    A large unmatched trajectory is an error, never a silent full-data copy.
    """
    path = Path(path).resolve().with_suffix('.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    sources, parts, descriptors = {}, [], {}
    with ExitStack() as stack:
        for source_path, selector in segments:
            source_path = Path(source_path).resolve()
            if source_path.suffix != '.npz':
                raise ValueError('Dataset blocks must be materialized NPZ archives')
            sid = str(source_path)
            if sid not in sources:
                sources[sid] = stack.enter_context(np.load(source_path, allow_pickle=True))
                descriptors[sid] = dict(path=os.path.relpath(source_path, path.parent),
                                        sha256=file_sha256(source_path))
            # The tiny sample-name array supplies the source row count.
            count = len(sources[sid]['mutant_names'])
            indices = _rows(selector, count)
            parts.append(dict(source=sid, rows=None if selector is None else indices.tolist(), count=len(indices)))
        row_count = sum(part['count'] for part in parts)
        fields, overrides = {}, {}
        for key in arrays.keys():
            target = np.asarray(arrays[key])
            spec = dict(dtype=target.dtype.str, shape=list(target.shape), sha256=_array_digest(target))
            matched = False
            if target.ndim and len(target) == row_count:
                offset, matched = 0, True
                for part in parts:
                    source = sources[part['source']]
                    if key not in source:
                        matched = False
                        break
                    value = source[key]
                    indices = _rows(part['rows'], len(value)) if value.ndim else np.array([], dtype=np.int64)
                    if value.ndim == 0 or value.shape[1:] != target.shape[1:] or (len(indices) and len(value) <= indices.max()):
                        matched = False
                    else:
                        for start in range(0, len(indices), 256):
                            selected = value[indices[start:start+256]].astype(target.dtype, copy=False)
                            if not _equal(target[offset+start:offset+start+len(selected)], selected):
                                matched = False
                                break
                    del value
                    if not matched:
                        break
                    offset += len(indices)
                if matched:
                    spec['pieces'] = parts
            if not matched and key != 'data':
                for sid, source in sources.items():
                    if key in source and _equal(target, source[key]):
                        spec['source'] = sid
                        matched = True
                        break
            if not matched:
                if key == 'data':
                    raise ValueError('Trajectory blocks do not reconstruct the exact target dataset')
                overrides[key] = target
                spec['source'] = '__metadata__'
            fields[key] = spec
            del target
        if overrides:
            fd, temporary = tempfile.mkstemp(prefix='.metadata-', suffix='.npz', dir=path.parent)
            os.close(fd)
            try:
                np.savez_compressed(temporary, **overrides)
                with open(temporary, 'rb') as handle:
                    os.fsync(handle.fileno())
                digest = file_sha256(temporary)
                metadata_path = path.parent / (path.stem + '.metadata.' + digest[:16] + '.npz')
                os.replace(temporary, metadata_path)
                descriptors['__metadata__'] = dict(path=metadata_path.name, sha256=digest)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        # Compact source ids keep the JSON small and independently relocatable.
        ids = {sid: str(i) for i, sid in enumerate(descriptors)}
        encoded_fields = {}
        for key, spec in fields.items():
            spec = dict(spec)
            if 'source' in spec:
                spec['source'] = ids[spec['source']]
            else:
                spec['pieces'] = list(range(len(parts)))
            encoded_fields[key] = spec
        manifest = dict(format=FORMAT, version=1, rows=row_count,
                        segments=[dict(p, source=ids[p['source']]) for p in parts],
                        sources={ids[sid]: desc for sid, desc in descriptors.items()},
                        fields=encoded_fields, provenance=provenance or {})
        _atomic_json(path, manifest)
    return str(path)


def verify_dataset_equal(original, manifest):
    """Bounded to one field at a time, exact comparison before removing a snapshot."""
    with open_dataset(original) as before, open_dataset(manifest) as after:
        if set(before.files) != set(after.files):
            raise ValueError('Dataset field set changed during compaction')
        for key in before.files:
            if not _equal(before[key], after[key]):
                raise ValueError(f'Dataset changed during compaction: {key}')
    return True


from collections import OrderedDict

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(encoded + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class TrajectoryStore:
    """Read requested rows only; no full-pool trajectory tensor or label cache.

    This is owned by the runner/oracle, never attached to PoolInputs passed to a
    selector. Chunk hashes are checked on first access and cached by file identity.
    """
    def __init__(self, path, cache_chunks=2):
        path = Path(path)
        self.path = (path / 'dataset.json' if path.is_dir() else path).resolve()
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get('format') != FORMAT:
            raise ValueError('A complete yeast dataset manifest is required')
        self.digest = file_sha256(self.path)
        self.count = int(self.manifest['rows'])
        self.cache_chunks = int(cache_chunks)
        self._cache = OrderedDict()
        self._verified = {}
        self._source_sizes = {}

    @property
    def fields(self):
        return self.manifest['fields']

    def _array(self, source_id, key):
        identity = (source_id, key)
        if identity in self._cache:
            self._cache.move_to_end(identity)
            return self._cache[identity]
        source = self.manifest['sources'][source_id]
        path = (self.path.parent / source['path']).resolve()
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._verified.get(source_id) != stamp:
            if file_sha256(path) != source['sha256']:
                raise ValueError(f'Changed pool block: {path}')
            self._verified[source_id] = stamp
        with np.load(path, allow_pickle=False) as archive:
            value = archive[key]
        self._cache[identity] = value
        while len(self._cache) > max(1, self.cache_chunks):
            self._cache.popitem(last=False)
        return value

    def scalar(self, key):
        spec = self.fields[key]
        if 'source' not in spec:
            raise ValueError(f'{key} is a row field')
        return self._array(spec['source'], spec.get('key', key)).copy()

    def read(self, rows, keys=('data',)):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or (rows < 0).any() or (rows >= self.count).any():
            raise ValueError('Invalid pool row indices')
        output = {}
        for key in keys:
            spec = self.fields[key]
            if 'pieces' not in spec:
                output[key] = self.scalar(key)
                continue
            result = np.empty((len(rows), *spec['shape'][1:]), dtype=spec['dtype'])
            offset = 0
            grouped = {}
            for item in spec['pieces']:
                piece = self.manifest['segments'][item] if isinstance(item, int) else item
                if 'count' in piece:
                    n = int(piece['count'])
                elif piece.get('rows') is not None:
                    n = len(piece['rows'])
                else:
                    sid = piece['source']
                    if sid not in self._source_sizes:
                        # Production complete-pool manifests omit the optional
                        # segment count. Read tiny input IDs, never trajectories.
                        self._source_sizes[sid] = len(self._array(sid, 'candidate_ids'))
                    n = self._source_sizes[sid]
                hits = np.flatnonzero((rows >= offset) & (rows < offset + n))
                if len(hits):
                    local = rows[hits] - offset
                    if piece.get('rows') is not None:
                        local = np.asarray(piece['rows'], dtype=np.int64)[local]
                    grouped.setdefault((piece['source'], piece.get('key', key)), []).append((hits, local))
                offset += n
            if offset != self.count:
                raise ValueError('Pool manifest row count mismatch')
            # A generated view may reference shuffled original rows through
            # thousands of tiny pieces. Decode each source/field once rather
            # than cycling those pieces through the small block LRU.
            for (sid, field), selections in grouped.items():
                value = self._array(sid, field)
                for hits, local in selections:
                    result[hits] = value[local]
            output[key] = result
        return output

    def inputs(self):
        keys = ('params', 'y_init', 'candidate_ids', 'mutant_names', 'is_real')
        values = self.read(np.arange(self.count), keys)
        if values['is_real'].any():
            raise ValueError('Complete selection pools must not contain real references')
        if not np.isfinite(values['params']).all() or not np.isfinite(values['y_init']).all():
            raise ValueError('Nonfinite pool inputs')
        return values

    def snapshot(self, path, rows, *, weights=None, provenance=None):
        rows = np.asarray(rows, dtype=np.int64)
        if len(np.unique(rows)) != len(rows):
            raise ValueError('Snapshot rows must be unique; aggregate draw multiplicities first')
        # Validation without reading true trajectories.
        if rows.ndim != 1 or (rows < 0).any() or (rows >= self.count).any():
            raise ValueError('Invalid snapshot rows')
        value = dict(format='yeast_sampling_selection', version=1,
                     source=str(self.path), source_sha256=self.digest,
                     rows=rows.tolist(), provenance=provenance or {})
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != rows.shape or not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError('Malformed selection weights')
            value['weights'] = weights.tolist()
        atomic_json(path, value)


def read_selection(path, keys=('data',)):
    value = json.loads(Path(path).read_text())
    if value.get('format') != 'yeast_sampling_selection':
        raise ValueError('Unsupported selection manifest')
    store = TrajectoryStore(value['source'])
    if store.digest != value['source_sha256']:
        raise ValueError('Selection source manifest changed')
    return store.read(value['rows'], keys)
