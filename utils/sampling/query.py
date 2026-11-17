"""Authorized label access; no training, model or selection-score code."""
import numpy as np
import os
from budding_yeast_v2.data.dataset_archive import atomic_json
from budding_yeast_v2.utils.experiments.resources import resource_slot

class LabelOracle:
    """Runner-owned exact stored output service with persistent unique-read counts."""
    def __init__(self, store, allowed_ids, ledger_path, track='A',state_key=None):
        self.store = store
        # Optional bounded RAM reuse for random/streaming queries. NPZ chunks
        # remain immutable; no trajectory copy is written to disk. The batch
        # supervisor still gates admission by system MemAvailable, not owned PSS.
        chunks=int(os.environ.get('SAMPLING_ORACLE_CACHE_CHUNKS',store.cache_chunks))
        if not 1<=chunks<=4096:raise ValueError('Oracle RAM cache must use 1..4096 chunk fields')
        self.store.cache_chunks=chunks
        self.allowed = set(map(int, allowed_ids))
        self.path = ledger_path
        self.track = track
        self.state_key = state_key
        self.revealed = set()
        self.accesses = 0
        self.events = []
        from pathlib import Path
        import json
        if Path(ledger_path).exists():
            value = json.loads(Path(ledger_path).read_text())
            if state_key is not None:value=value.get(state_key)
            if value is None:return
            if value['pool_sha256'] != store.digest or value['track'] != track:
                raise ValueError('Oracle source/track changed')
            self.revealed = set(value['revealed_ids'])
            self.accesses = value['logical_label_reads']
            self.events = value['events']

    def reveal(self, ids, purpose='selected_training', keys=('data',)):
        ids = np.asarray(ids, dtype=np.int64)
        if not set(map(int, ids)) <= self.allowed:
            raise PermissionError('Validation/test/foreign IDs cannot be queried for training')
        if purpose == 'candidate_scoring' and self.track != 'B':
            raise PermissionError('Unqueried candidate truth is forbidden in AL scoring')
        with resource_slot('data', 'queried trajectory loading'):
            values = self.store.read(ids, keys)
        newly = len(set(map(int, ids)) - self.revealed)
        self.revealed.update(map(int, ids))
        self.accesses += len(ids)
        self.events.append(dict(purpose=purpose, rows=len(ids), new_unique_labels=newly))
        value=dict(pool_sha256=self.store.digest, track=self.track,
            revealed_ids=sorted(self.revealed), unique_labels_accessed=len(self.revealed),
            logical_label_reads=self.accesses, new_ode_calls=0, events=self.events)
        if self.state_key is None:atomic_json(self.path,value)
        else:
            from budding_yeast_v2.utils.experiments.state import update_section
            update_section(self.path,self.state_key,value)
        return values

    def scoring_reader(self, ids):
        return self.reveal(ids, purpose='candidate_scoring',
                           keys=('data', 'pattern_group_ids'))

