"""Immutable row corrections over existing trajectory blocks, without duplication."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

import numpy as np

from .dataset_archive import TrajectoryStore, atomic_json, file_sha256


def write_row_patch(source, destination, rows, changes, *, provenance, progress=None):
    """Publish an exact ordered view; only corrected rows occupy new NPZ space.

    All supplied fields must already be row fields. Unchanged fields, source
    hashes and scalar metadata are preserved. Changed field hashes are rebuilt
    in one bounded pass over the original blocks, before publishing the view.
    """
    source = TrajectoryStore(source, cache_chunks=1)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    ids = np.asarray(rows, dtype=np.int64)
    if (ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids)
            or (ids < 0).any() or (ids >= source.count).any()):
        raise ValueError('Patch requires unique valid row IDs')
    row_fields = {k for k, v in source.fields.items() if 'pieces' in v}
    if not changes or not set(changes) <= row_fields:
        raise ValueError('Patch may only replace existing row fields')
    manifest = deepcopy(source.manifest)
    canonical = source.fields['data']['pieces']
    if any(source.fields[k]['pieces'] != canonical for k in row_fields):
        raise ValueError('Repair requires aligned row-field pieces')
    patch = source.read(ids, tuple(sorted(row_fields)))
    for key, value in changes.items():
        value = np.asarray(value)
        old = patch[key]
        if value.shape != old.shape or value.dtype.kind != old.dtype.kind:
            raise ValueError(f'Changed row shape/type: {key}')
        dtype = np.promote_types(old.dtype, value.dtype)
        if dtype.kind not in 'US' and dtype != old.dtype:
            raise ValueError(f'Numeric patch must retain dtype: {key}')
        patch[key] = value.astype(dtype, copy=False)
        manifest['fields'][key]['dtype'] = dtype.str
    if not np.isfinite(patch['data']).all():
        raise ValueError('Nonfinite repaired trajectory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    patch_path = destination.with_suffix('.patch.npz')
    if patch_path.exists():
        raise FileExistsError(patch_path)
    temporary = patch_path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **patch)
    os.link(temporary, patch_path)
    temporary.unlink()
    for sid, descriptor in manifest['sources'].items():
        descriptor['path'] = os.path.relpath((source.path.parent / descriptor['path']).resolve(), destination.parent)
    patch_sid = '__row_corrections__'
    if patch_sid in manifest['sources']:
        raise ValueError('Nested row repairs require a fresh source ID')
    manifest['sources'][patch_sid] = dict(path=patch_path.name, sha256=file_sha256(patch_path))
    segments, offset = [], 0
    digests = {}
    for key in changes:
        spec = manifest['fields'][key]
        h = hashlib.sha256()
        h.update(spec['dtype'].encode()); h.update(str(tuple(spec['shape'])).encode())
        digests[key] = h
    # A block at a time, sharing one open compressed source across changed fields.
    for block_index, item in enumerate(canonical):
        piece = source.manifest['segments'][item] if isinstance(item, int) else item
        descriptor = source.manifest['sources'][piece['source']]
        path = (source.path.parent / descriptor['path']).resolve()
        if file_sha256(path) != descriptor['sha256']:
            raise ValueError(f'Changed original source: {path}')
        with np.load(path, allow_pickle=False) as archive:
            local = (np.arange(len(archive['candidate_ids'])) if piece.get('rows') is None
                     else np.asarray(piece['rows'], dtype=np.int64))
            n = len(local)
            hits = np.flatnonzero((ids >= offset) & (ids < offset + n))
            ordered = hits[np.argsort(ids[hits])]
            start = 0
            for hit in ordered:
                position = int(ids[hit] - offset)
                if position > start:
                    segments.append(dict(source=piece['source'], rows=local[start:position].tolist(), count=position-start))
                segments.append(dict(source=patch_sid, rows=[int(hit)], count=1))
                start = position + 1
            if start < n:
                if start == 0:
                    segments.append(dict(piece, count=n))
                else:
                    segments.append(dict(source=piece['source'], rows=local[start:].tolist(), count=n-start))
            for key, digest in digests.items():
                values = archive[key][local].astype(manifest['fields'][key]['dtype'], copy=True)
                values[ids[hits] - offset] = patch[key][hits]
                digest.update(memoryview(np.ascontiguousarray(values)).cast('B'))
        offset += n
        if progress is not None:
            progress(block_index + 1, offset, source.count)
    if offset != source.count:
        raise ValueError('Source row count changed')
    manifest['segments'] = segments
    for key in row_fields:
        manifest['fields'][key]['pieces'] = list(range(len(segments)))
    for key, digest in digests.items():
        manifest['fields'][key]['sha256'] = digest.hexdigest()
    manifest['provenance'] = dict(manifest.get('provenance', {}), numerical_repair=dict(
        source=str(source.path), source_sha256=source.digest, corrected_rows=ids.tolist(),
        changed_fields=sorted(changes), inputs_and_row_order_unchanged=True, **provenance))
    atomic_json(destination, manifest)
    return destination
