"""Generate resumable complete mutation/LHS pools with canonical Budding biology.

All pool rows are synthetic. Real definitions are used only for input collision
exclusion, never for trajectory filtering or fitting proposal probabilities.
"""
import argparse
import fcntl
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import scipy

from ..contracts import Candidate, GenerationConfig, SimulationResult
from ..dataset_io import write_dataset
from ..sampling_policy import valid_candidate
from ...dataset_archive import FORMAT, _atomic_json, _array_digest, file_sha256
from ...simulation_guards import bounded_ordered_map
from .catalog import CATALOG_VERSION, touch_set
from .distributions.ranges import RANGE_FINGERPRINT, GAL_MODES
from .lhs import sample_design
from .quotas import enumerate_combinations
from .real_mutants import effective_inputs, get_cached_real_params
from .samplers.rescue import rescue_budding, parse_pairs
from .specification import BuddingSystem, SPEC
from .solver_policy import SOLVER_POLICY, RTOL, ATOL, RETRY_RTOL, RETRY_ATOL

CONTRACT='budding_complete_pools_random_and_mixed_lhs_v1'
ROW_FIELDS={'params','mutant_names','sample_id','pair_id','is_real','levels','overrides_json',
    'candidate_metadata_json','simulation_diagnostics_json','y_init','mutation_descriptions',
    'mutation_pairs_json','mutation_pairs_available','simulation_status','initial_state_source',
    'overrides_available','candidate_ids','data','labels','biological_labels_available',
    'pattern_labels','pattern_group_codes','pattern_group_labels','pattern_group_ids',
    'observed_division_counts','requested_levels','module_counts','mutation_available',
    'lhs_design_ids','lhs_design_rows'}


def stamp(message):print(time.strftime('[%Y-%m-%d %H:%M:%S] ')+message,flush=True)


def stream_seed(seed,kind,index,stage=0):
    return int(np.random.SeedSequence([seed,101 if kind=='mutation' else 202,index,stage]).generate_state(1)[0])


def source_hashes():
    root=Path(__file__).resolve().parents[2]
    files=list((root/'systems'/'budding').rglob('*.py'))
    files += [root/name for name in ['simulation_guards.py','dataset_archive.py','sample_metadata.py','label_metadata.py',
                                     'systems/dataset_io.py','systems/contracts.py','systems/registry.py','systems/sampling_policy.py']]
    return {str(p):file_sha256(p) for p in sorted(files)}


class Proposals:
    def __init__(self,kind,seed,count,directory,lhs_spec):
        self.kind,self.seed,self.count,self.directory,self.lhs_spec=kind,seed,count,directory,lhs_spec
        self.backend=BuddingSystem();self.defaults=self.backend.defaults();self.reals=get_cached_real_params(self.defaults)
        self.design_id=-1;self.design=None
        if kind=='mutation':
            groups=defaultdict(list)
            for cell in enumerate_combinations()[0]:groups[cell['modules']].append(cell['alleles'])
            self.groups=[(modules,sorted(rows)) for modules,rows in sorted(groups.items())]
            if len(self.groups)!=1159:raise ValueError('Review changed module-combination policy')

    def collides(self,vector):
        return bool(np.any(np.all(np.abs(self.reals-vector)<=1e-8+1e-5*np.abs(self.reals),axis=1)))

    def get_design(self,index):
        design_id,row=divmod(index,self.count)
        if self.design_id!=design_id:
            path=self.directory/'designs'/f'design_{design_id:04d}.npz';path.parent.mkdir(exist_ok=True)
            if path.exists():
                with np.load(path,allow_pickle=False) as archive:
                    if str(archive['config_sha256'])!=self.config_sha:raise ValueError('LHS design configuration changed')
                    self.design=archive['params']
            else:
                stamp(f'Creating LHS probability design {design_id}: {self.count} rows, 141 coordinates')
                self.design,coordinates=sample_design(self.lhs_spec,SPEC.param_names,self.count,
                                                     stream_seed(self.seed,self.kind,design_id,11))
                temporary=path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary,params=self.design,param_names=SPEC.param_names,
                    config_sha256=self.config_sha,design_seed=stream_seed(self.seed,self.kind,design_id,11),
                    latent_design_sha256=hashlib.sha256(coordinates.astype('<f8').tobytes()).hexdigest())
                os.replace(temporary,path)
                stamp(f'LHS design {design_id} saved')
            self.design_id=design_id
        return design_id,row,self.design[row]

    def propose(self,index):
        entropy=[self.seed,101 if self.kind=='mutation' else 202,index,7]
        rng=np.random.default_rng(np.random.SeedSequence(entropy))
        if self.kind=='lhs':
            design,row,params=self.get_design(index)
            overrides=dict(zip(SPEC.param_names,map(float,params)))
            metadata=dict(description='Independent mixed-marginal parameter design',pairs=[],module_names=[],
                module_count=-1,requested_level=-1,lhs_design_id=design,lhs_design_row=row)
            level=-1;name=f'MixedLHS-{index:09d}'
        else:
            modules,options=self.groups[int(rng.integers(len(self.groups)))];allele=options[int(rng.integers(len(options)))]
            pairs=parse_pairs(allele);overrides={};draws=[]
            for gene,mode in pairs:
                alpha=self.backend.sample_scaling_factor(mode,gene,rng=rng)
                self.backend.apply_gene_mutation(gene,mode,alpha,self.defaults,self.defaults.copy(),overrides,rng=rng)
                draws.append(dict(gene=gene,mode=mode,alpha=float(alpha)))
            overrides['mdt']=150. if any(m in GAL_MODES for _,m in pairs) else float(rng.choice([90.,150.,160.],p=[.8,.15,.05]))
            level=len(modules.split(';'));description='+'.join(f'{g}[{m}]' for g,m in pairs)
            metadata=dict(description=description,pairs=pairs,proposal_modules=modules,proposal_alleles=allele,
                proposal_strength_draws=draws,original_overrides=dict(overrides),rescued=False,
                module_names=modules.split(';'),module_count=level,requested_level=level)
            name=f'OfflineMutation-{index:09d}'
        metadata.update(source_candidate_id=index,sampling_paradigm='mutation_random' if self.kind=='mutation' else 'lhs_mixed',
                        proposal_seed_entropy=entropy)
        vector=effective_inputs(overrides,self.defaults)
        if self.collides(vector):
            metadata['real_collision']=True
            if self.kind=='lhs':return None,'real_collision'
            for attempt in range(30):
                proposed,description=rescue_budding(overrides,metadata['description'],self.backend,rng)
                checked=valid_candidate(Candidate(index+1,name,proposed,level,
                    metadata=dict(metadata,description=description,pairs=parse_pairs(description))),'budding')
                if checked is None:continue
                trial=effective_inputs(proposed,self.defaults)
                if not self.collides(trial):
                    overrides=proposed;vector=trial;metadata=dict(checked.metadata,rescued=True,rescue_attempts=attempt+1);break
            else:return None,'real_collision_exhausted'
        candidate=Candidate(index+1,name,overrides,level,metadata=metadata)
        if self.kind=='mutation':
            candidate=valid_candidate(candidate,'budding')
            if candidate is None:raise ValueError('Random proposal violates mutation contract')
            candidate.metadata['mutation_parameters']=[dict(gene=g,mode=m,
                overrides={k:overrides[k] for k in sorted(touch_set(g,m)) if k in overrides})
                for g,m in candidate.metadata['pairs']]
        return (candidate,hashlib.sha256(vector.astype('<f8').tobytes()).digest()),None


def simulate(candidate):
    """No stochastic solver or mutation draws inside workers."""
    backend=BuddingSystem();started=time.monotonic()
    diagnostics={}
    _,trajectory,bio,pattern=backend.simulate_mutant(candidate.name,candidate.overrides,diagnostics=diagnostics)
    _,initial,_,_=backend.prepare_initial_conditions(candidate.overrides)
    defaults=backend.defaults()
    params=np.array([candidate.overrides.get(k,defaults[k]) for k in SPEC.param_names],dtype=np.float64)
    diagnostics['seconds']=time.monotonic()-started
    if trajectory is not None:
        diagnostics.update(minimum_raw=float(trajectory.min()),maximum_raw=float(trajectory.max()),
            negative_value_count=int((trajectory<0).sum()),negative_variables=np.flatnonzero((trajectory<0).any(axis=1)).tolist())
        with np.errstate(over='ignore'):
            finite32=np.isfinite(trajectory.astype(np.float32)).all()
        if trajectory.shape!=(39,500) or not np.isfinite(trajectory).all() or not finite32:trajectory=None
    return SimulationResult(candidate,trajectory,params,initial,bio,pattern,
                            'success' if trajectory is not None else 'numerical_failure_or_timeout',diagnostics)


def extra_fields(results,kind):
    n=len(results);mutation=kind=='mutation'
    return dict(sampling_paradigm=np.array('mutation_random' if mutation else 'lhs_mixed'),
        dataset_role=np.array('complete_candidate_pool'),
        requested_levels=np.array([r.candidate.level for r in results],dtype=np.int16),
        module_counts=np.array([r.candidate.level for r in results],dtype=np.int16),
        mutation_available=np.full(n,mutation,dtype=bool),
        lhs_design_ids=np.array([r.candidate.metadata.get('lhs_design_id',-1) for r in results],dtype=np.int32),
        lhs_design_rows=np.array([r.candidate.metadata.get('lhs_design_row',-1) for r in results],dtype=np.int64))


def publish_manifest(directory,chunks,report,config):
    """Reference immutable chunks once; do not copy cumulative trajectory arrays."""
    if not chunks:raise ValueError('Cannot publish empty pool')
    sources={str(i):dict(path=c['path'],sha256=c['sha256']) for i,c in enumerate(chunks)}
    segments=[dict(source=str(i),rows=None) for i in range(len(chunks))]
    fields={};total=sum(c['rows'] for c in chunks)
    with np.load(directory/chunks[0]['path'],allow_pickle=False) as archive:
        for key in archive.files:
            value=archive[key]
            if key in ROW_FIELDS:
                shape=list(value.shape);shape[0]=total;dtype=value.dtype
                # String widths can differ between chunks; never truncate names,
                # metadata, genotype lists or labels to the first chunk width.
                if dtype.kind in 'US':
                    for c in chunks[1:]:dtype=np.promote_types(dtype,np.dtype(c['row_dtypes'][key]))
                fields[key]=dict(dtype=dtype.str,shape=shape,pieces=list(range(len(chunks))))
            else:fields[key]=dict(dtype=value.dtype.str,shape=list(value.shape),source='0',sha256=_array_digest(value))
    # Match ManifestArchive's exact array hashes without allocating a second
    # cumulative copy of the trajectories. Hash includes final dtype and shape.
    digests={}
    for key in ROW_FIELDS:
        spec=fields[key];h=hashlib.sha256()
        h.update(spec['dtype'].encode());h.update(str(tuple(spec['shape'])).encode());digests[key]=h
    stamp(f'Hashing {total} stored rows across {len(chunks)} immutable chunks')
    for chunk in chunks:
        with np.load(directory/chunk['path'],allow_pickle=False) as archive:
            for key,h in digests.items():
                value=np.ascontiguousarray(archive[key].astype(fields[key]['dtype'],copy=False))
                h.update(memoryview(value).cast('B'))
    for key,h in digests.items():fields[key]['sha256']=h.hexdigest()
    meta_path=directory/'pool_metadata.npz'
    meta=dict(generation_report=np.array(json.dumps(report,ensure_ascii=False)),
              generation_config=np.array(json.dumps(config,ensure_ascii=False)),
              pool_generation_contract=np.array(CONTRACT),artifact_kind=np.array('trajectory_dataset'))
    temporary=meta_path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**meta);os.replace(temporary,meta_path)
    sources['metadata']=dict(path=meta_path.name,sha256=file_sha256(meta_path))
    for key,value in meta.items():fields[key]=dict(dtype=value.dtype.str,shape=list(value.shape),source='metadata',sha256=_array_digest(value))
    _atomic_json(directory/'dataset.json',dict(format=FORMAT,version=1,rows=total,
        segments=segments,sources=sources,fields=fields,provenance=dict(contract=CONTRACT,kind=config['kind'],seed=config['seed'])))


def _generate_locked(kind,output,count=100000,seed=42,workers=24,block_size=256,lhs_config=None,resume=False,max_attempts_mult=30):
    directory=Path(output).expanduser().resolve()
    directory.mkdir(parents=True,exist_ok=True);(directory/'chunks').mkdir(exist_ok=True)
    if kind not in ('mutation','lhs') or min(count,workers,block_size)<1 or seed<0:raise ValueError('Invalid generation settings')
    lhs_spec=None;spec_sha=None
    if kind=='lhs':
        if lhs_config is None:raise ValueError('--lhs-config must name the reviewed mixed marginal JSON')
        path=Path(lhs_config);lhs_spec=json.loads(path.read_text());spec_sha=file_sha256(path)
        if set(lhs_spec['marginals'])!=set(SPEC.param_names):raise ValueError('LHS parameter schema mismatch')
        if tuple(lhs_spec['positive_window_approved'])!=(.5,1.5):raise ValueError('Preserve approved ±50% background window')
        inventory=path.parent/'inventory.json'
        if not inventory.exists() or json.loads(inventory.read_text())['range_fingerprint']!=RANGE_FINGERPRINT:
            raise ValueError('LHS configuration uses obsolete mutation ranges')
        existing_spec=directory/'lhs_specification.json'
        if existing_spec.exists() and json.loads(existing_spec.read_text())!=lhs_spec:
            raise ValueError('Existing run uses a different LHS specification')
        _atomic_json(existing_spec,lhs_spec)
    config=dict(contract=CONTRACT,kind=kind,count=count,seed=seed,block_size=block_size,
        lhs_spec_sha256=spec_sha,catalog_version=CATALOG_VERSION,range_fingerprint=RANGE_FINGERPRINT,
        horizon=210.,steps=500,trajectory_dtype='float32',parameter_dtype='float64',
        source_hashes=source_hashes(),background_rule='U(0.5WT,1.5WT)' if kind=='lhs' else 'WT',
        mutation_proposal='uniform module sets then uniform legal complete alleles; all amplitudes pseudorandom',
        rng_contract='SeedSequence(kind, generation_seed, proposal_id); ordered acceptance',
        simulation_timeout_s=5.,candidate_pool_contains_real=False,
        solver_policy=SOLVER_POLICY,solver_rtol=RTOL,solver_atol=ATOL,
        retry_rtol=RETRY_RTOL,retry_atol=RETRY_ATOL,max_solver_attempts=2)
    config['software_versions']=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__)
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    existing_config=directory/'run_config.json'
    if existing_config.exists() and json.loads(existing_config.read_text())['config_sha256']!=config_hash:
        raise ValueError('Existing run settings or production sources changed')
    _atomic_json(directory/'run_config.json',dict(config,workers=workers,config_sha256=config_hash))
    with closing(sqlite3.connect(directory/'progress.sqlite',timeout=30.)) as database:
        database.execute('PRAGMA journal_mode=WAL');database.execute('PRAGMA synchronous=FULL')
        database.execute('CREATE TABLE IF NOT EXISTS seen (signature BLOB PRIMARY KEY)')
        database.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)');database.commit()
        previous=database.execute("SELECT value FROM metadata WHERE key='state'").fetchone()
        if previous:
            state=json.loads(previous[0])
            if state['config_sha256']!=config_hash:raise ValueError('Resume settings or production sources changed')
            for chunk in state['chunks']:
                if file_sha256(directory/chunk['path'])!=chunk['sha256']:raise ValueError('Committed trajectory chunk changed')
        else:state=dict(config_sha256=config_hash,next_proposal=0,accepted=0,batches=0,chunks=[],counts={})
        if state['accepted']==count:
            if (directory/'completion.json').exists() and (directory/'dataset.json').exists():
                completed=json.loads((directory/'completion.json').read_text())
                if completed['manifest_sha256']!=file_sha256(directory/'dataset.json'):raise ValueError('Completed manifest changed')
                database.close();stamp('Pool already complete and verified');return str(directory/'dataset.json')
            stamp('Pool already complete; checking/publishing manifest')
            publish_manifest(directory,state['chunks'],state,config)
            if not (directory/'completion.json').exists():
                _atomic_json(directory/'completion.json',dict(samples=count,kind=kind,seed=seed,
                    completed_at=state['updated_at'],manifest_sha256=file_sha256(directory/'dataset.json'),counts=state['counts']))
            database.close();return str(directory/'dataset.json')
        proposals=Proposals(kind,seed,count,directory,lhs_spec);proposals.config_sha=spec_sha
        settings=GenerationConfig('budding_random',n_samples=count,seed=seed,workers=workers,include_real=False,
                                  mutation_profile='mutant_aware',distribution_profile='mutant_aware')
        counters=Counter(state['counts']);started=time.monotonic()
        stamp(f'START {kind}: accepted {state["accepted"]}/{count}; workers={workers}, seed={seed}')
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as executor:
            while state['accepted']<count:
                database.execute('BEGIN');batch=[]
                while len(batch)<min(block_size,count-state['accepted']):
                    index=state['next_proposal'];state['next_proposal']+=1
                    if state['next_proposal']>count*max_attempts_mult:raise RuntimeError('Proposal limit exceeded; inspect rejection report')
                    proposed,rejection=proposals.propose(index);counters['proposals']+=1
                    if proposed is None:counters[rejection]+=1;continue
                    candidate,signature=proposed
                    inserted=database.execute('INSERT OR IGNORE INTO seen(signature) VALUES (?)',(signature,)).rowcount
                    if not inserted:counters['duplicates']+=1;continue
                    if candidate.metadata.get('real_collision'):counters['real_collisions']+=1
                    if candidate.metadata.get('rescued'):counters['rescued']+=1
                    batch.append(candidate)
                results=[];attempts=[]
                for result in bounded_ordered_map(executor,simulate,batch,stall_timeout=35.):
                    candidate=result.candidate
                    attempts.append(dict(proposal_id=candidate.metadata['source_candidate_id'],status=result.status,diagnostics=result.diagnostics))
                    counters['simulated']+=1
                    if result.status!='success':counters['simulation_failures']+=1;continue
                    result.candidate=replace(candidate,sample_id=state['accepted']+len(results)+1)
                    results.append(result)
                    counters['negative_samples']+=int(result.diagnostics['negative_value_count']>0)
                    counters['level_'+str(candidate.level)]+=1
                    counters['bio_'+result.label]+=1;counters['abc_'+str(result.pattern)]+=1
                    for gene,mode in candidate.metadata.get('pairs',[]):counters[f'mode_{gene}[{mode}]']+=1
                batch_id=state['batches'];chunk=None
                if results:
                    relative=f'chunks/chunk_{batch_id:06d}.npz';path=directory/relative
                    # An interrupted process can leave an uncommitted orphan file.
                    # Only this run's deterministic, uncommitted chunk is replaced.
                    if path.exists():path.unlink()
                    report=dict(mutation_profile='mutant_aware',distribution_profile='mutant_aware',
                        rng_contract=config['rng_contract'],pool_generation_contract=CONTRACT,
                        sampling_paradigm='mutation_random' if kind=='mutation' else 'lhs_mixed',
                        proposal_batch=batch_id,requested_complete_pool_size=count)
                    write_dataset(path,SPEC,results,settings,report,dtype='float32',extra_fields=extra_fields(results,kind))
                    with np.load(path,allow_pickle=False) as archive:
                        dtypes={key:archive[key].dtype.str for key in ROW_FIELDS}
                    chunk=dict(path=relative,sha256=file_sha256(path),rows=len(results),row_dtypes=dtypes)
                _atomic_json(directory/'chunks'/f'attempts_{batch_id:06d}.json',attempts)
                if chunk:state['chunks'].append(chunk)
                state['accepted']+=len(results);state['batches']+=1;state['counts']=dict(counters)
                state['updated_at']=time.strftime('%Y-%m-%d %H:%M:%S');state['workers']=workers
                database.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('state',?)",(json.dumps(state),));database.commit()
                _atomic_json(directory/'progress.json',state)
                stamp(f'{kind} {state["accepted"]}/{count}; proposals={counters["proposals"]}, failures={counters["simulation_failures"]}, '
                      f'duplicates={counters["duplicates"]}, elapsed={time.monotonic()-started:.1f}s')
        state['completed_at']=time.strftime('%Y-%m-%d %H:%M:%S')
        publish_manifest(directory,state['chunks'],state,config)
        _atomic_json(directory/'completion.json',dict(samples=count,kind=kind,seed=seed,completed_at=state['completed_at'],
            manifest_sha256=file_sha256(directory/'dataset.json'),counts=dict(counters)))
        database.close();stamp(f'COMPLETE {directory / "dataset.json"}')
        return str(directory/'dataset.json')

def generate(kind,output,count=100000,seed=42,workers=24,block_size=256,lhs_config=None,resume=False,max_attempts_mult=30):
    directory=Path(output).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise FileExistsError(f'Use a fresh output directory or --resume: {directory}')
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.generation.lock').open('a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'Another generator owns {directory}')
        return _generate_locked(kind,output,count,seed,workers,block_size,lhs_config,resume,max_attempts_mult)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['mutation','lhs'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=100000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--block-size',type=int,default=256)
    parser.add_argument('--lhs-config',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-attempts-mult',type=int,default=30)
    args=parser.parse_args()
    generate(args.kind,args.output,args.samples,args.seed,args.workers,args.block_size,args.lhs_config,args.resume,args.max_attempts_mult)


if __name__=='__main__':main()
