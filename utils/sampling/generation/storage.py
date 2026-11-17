"""Immutable generated blocks plus manifests referencing original initial rows.

No initial trajectories or cumulative snapshots are copied. Field-specific source
pieces also preserve numerical-QC overlays in the frozen source pool.
"""
import os
from pathlib import Path
import zipfile

import numpy as np

from budding_yeast_v2.data.dataset_archive import (FORMAT,TrajectoryStore,atomic_json,
                                                  file_sha256,_array_digest,fingerprint)
from budding_yeast_v2.data.systems.budding.offline_pool import ROW_FIELDS,extra_fields
from budding_yeast_v2.data.systems.budding.specification import SPEC
from budding_yeast_v2.data.systems.contracts import GenerationConfig
from budding_yeast_v2.data.systems.dataset_io import write_dataset


def array_header(path,key):
    with zipfile.ZipFile(path) as archive,archive.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,_,dtype=reader(stream)
    return shape,dtype


class GeneratedTrainingStore:
    def __init__(self,source,initial_rows,directory,kind,seed,chunks=(),generation_metadata=None):
        self.source=TrajectoryStore(source);self.initial_rows=np.asarray(initial_rows,dtype=np.int64)
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'chunks').mkdir(exist_ok=True)
        self.kind,self.seed=kind,int(seed);self.chunks=list(chunks)
        self.generation_metadata=dict(generation_metadata or {})
        keys=sorted(ROW_FIELDS & self.source.fields.keys())
        self.initial=self.source.read(self.initial_rows,keys)
        for item in self.chunks:
            if file_sha256(self.directory/item['path'])!=item['sha256']:raise ValueError('Generated block changed')
        self._header_cache={}

    @property
    def count(self):
        return len(self.initial_rows)+sum(x['rows'] for x in self.chunks)

    def array(self,key):
        values=[self.initial[key]]
        for chunk in self.chunks:
            with np.load(self.directory/chunk['path'],allow_pickle=False) as z:values.append(z[key])
        return values[0] if len(values)==1 else np.concatenate(values)

    def append(self,results):
        if not results:return
        name=f'chunks/block_{len(self.chunks):06d}.npz';path=self.directory/name
        if path.exists():
            # Only a deterministic, uncommitted block in this run may be retried.
            # Committed entries were verified in __init__ and are never replaced.
            path.unlink()
        for i,result in enumerate(results):
            result.candidate.sample_id=self.source.count+self.count+i+1
        cfg=GenerationConfig('budding_random',n_samples=len(results),seed=self.seed,include_real=False,
                             mutation_profile='mutant_aware',distribution_profile='mutant_aware')
        report=dict(mutation_profile='mutant_aware',distribution_profile='mutant_aware',
                    rng_contract='canonical_proposals_no_rule_changes',workflow='open_generation')
        write_dataset(path,SPEC,results,cfg,report,dtype='float32',extra_fields=extra_fields(results,self.kind))
        self.chunks.append(dict(path=name,sha256=file_sha256(path),rows=len(results)))

    def _header(self,path,key):
        identity=(str(path),key)
        if identity not in self._header_cache:self._header_cache[identity]=array_header(path,key)
        return self._header_cache[identity]

    def publish(self,round_index):
        path=self.directory/f'training_source_{round_index:02d}.json'
        sources={};segments=[];source_ids={};piece_ids={}
        def source_id(source_path):
            source_path=Path(source_path).resolve();key=str(source_path)
            if key not in source_ids:
                sid=f's{len(sources)}';source_ids[key]=sid
                sources[sid]=dict(path=os.path.relpath(source_path,path.parent),sha256=file_sha256(source_path))
            return source_ids[key]
        def piece(source_path,rows,key=None):
            value=dict(source=source_id(source_path),rows=None if rows is None else list(map(int,rows)))
            if key is not None:value['key']=key
            token=fingerprint(value)
            if token not in piece_ids:piece_ids[token]=len(segments);segments.append(value)
            return piece_ids[token]
        def initial_pieces(key):
            spec=self.source.fields[key];original=self.source.manifest
            entries=[original['segments'][p] if isinstance(p,int) else p for p in spec['pieces']]
            counts=[];resolved=[]
            for entry in entries:
                source=original['sources'][entry['source']]
                src=(self.source.path.parent/source['path']).resolve()
                field=entry.get('key',key);rows=entry.get('rows')
                count=self._header(src,field)[0][0] if rows is None else len(rows)
                counts.append(count);resolved.append((src,rows,entry.get('key')))
            ends=np.cumsum(counts);groups=np.searchsorted(ends,self.initial_rows,side='right')
            starts=np.r_[0,ends[:-1]];answer=[];offset=0
            while offset<len(groups):
                stop=offset+1
                while stop<len(groups) and groups[stop]==groups[offset]:stop+=1
                g=groups[offset];src,selection,field=resolved[g]
                local=self.initial_rows[offset:stop]-starts[g]
                if selection is not None:local=np.asarray(selection)[local]
                answer.append(piece(src,local,field));offset=stop
            return answer
        fields={}
        for key,spec in self.source.fields.items():
            if key in ROW_FIELDS and 'pieces' in spec:
                values=self.array(key)
                refs=initial_pieces(key)
                refs.extend(piece(self.directory/c['path'],None) for c in self.chunks)
                fields[key]=dict(dtype=values.dtype.str,shape=list(values.shape),sha256=_array_digest(values),pieces=refs)
            elif 'source' in spec:
                original=self.source.manifest['sources'][spec['source']]
                sid=source_id(self.source.path.parent/original['path'])
                fields[key]=dict(spec,source=sid)
        metadata=path.with_suffix('.npz')
        import json
        np.savez_compressed(metadata,dataset_role=np.array('generated_training_source'),
            generation_workflow=np.array('canonical_legal_proposal_adapter_v1'),
            generation_config=np.array(json.dumps(dict(self.generation_metadata,kind=self.kind,seed=self.seed),sort_keys=True)),
            generation_report=np.array(json.dumps(dict(initial_rows=len(self.initial_rows),
                newly_generated=self.count-len(self.initial_rows),total_rows=self.count,
                source_trajectories_copied=False),sort_keys=True)))
        sid=source_id(metadata)
        for key in ('dataset_role','generation_workflow','generation_config','generation_report'):
            with np.load(metadata,allow_pickle=False) as z:value=z[key]
            fields[key]=dict(source=sid,dtype=value.dtype.str,shape=list(value.shape),sha256=_array_digest(value))
        atomic_json(path,dict(format=FORMAT,version=1,rows=self.count,sources=sources,segments=segments,
            fields=fields,provenance=dict(workflow='open_generation',initial_source=str(self.source.path),
                                        initial_source_sha256=self.source.digest,initial_rows=self.initial_rows.tolist(),
                                        source_trajectories_copied=False,generated_chunks=self.chunks)))
        return path
