"""Full, streamed integrity/distribution audit of a completed Budding pool.

No simulation, training, filtering or pool mutation. Includes all archive hashes,
effective-input uniqueness/real exclusion, initial states, labels and LHS design.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import qmc

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from budding_yeast_v2.data.dataset_archive import file_sha256,open_dataset
from budding_yeast_v2.data.systems.budding.offline_pool import ROW_FIELDS
from budding_yeast_v2.data.systems.budding.catalog import first_mutation_conflict,require_mutation
from budding_yeast_v2.data.systems.budding.lhs import mixture_atoms,mixture_cdf
from budding_yeast_v2.data.systems.budding.parameters import get_default_params
from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
from budding_yeast_v2.data.systems.budding.phenotypes import analyze_mass_pattern
from budding_yeast_v2.data.systems.budding.real_mutants import effective_inputs,get_cached_real_params
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem,SPEC


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit_lhs(directory,spec,names):
    rows=[];design_count=0
    for path in sorted((directory/'designs').glob('design_*.npz')):
        with np.load(path,allow_pickle=False) as archive:
            values=archive['params'];seed=int(archive['design_seed']);latent_hash=str(archive['latent_design_sha256'])
        u=qmc.LatinHypercube(d=len(names),seed=seed).random(len(values))
        for source,target in [('init_CDH1T','init_CDH1'),('BUB2_active','init_BUB2')]:
            u[:,names.index(target)]=u[:,names.index(source)]
        assert hashlib.sha256(u.astype('<f8').tobytes()).hexdigest()==latent_hash
        for j,name in enumerate(names):
            strata=(u[:,j]*len(values)).astype(np.int64)
            assert np.array_equal(np.bincount(strata,minlength=len(values)),np.ones(len(values),dtype=int)),name
            x=values[:,j];parts=spec['marginals'][name];upper=mixture_cdf(x,parts);lower=upper.copy()
            for atom,mass in mixture_atoms(parts):lower[x==atom]-=mass
            # Generalized quantile property: F(x-) <= u <= F(x).
            violation=float(max(np.max(lower-u[:,j]),np.max(u[:,j]-upper),0.))
            assert violation<2e-10,(name,violation)
            rows.append(dict(design=path.name,parameter=name,count=len(values),latin_strata_valid=True,
                             quantile_probability_violation=violation))
        design_count+=1
    return rows,design_count


def audit(directory,output):
    directory=directory.resolve();output.mkdir(parents=True,exist_ok=True)
    config=json.loads((directory/'run_config.json').read_text())
    completion=json.loads((directory/'completion.json').read_text())
    manifest=json.loads((directory/'dataset.json').read_text())
    assert file_sha256(directory/'dataset.json')==completion['manifest_sha256']
    assert manifest['rows']==config['count']==completion['samples']
    for path,digest in config['source_hashes'].items():assert file_sha256(path)==digest,path
    kind=config['kind'];names=list(SPEC.param_names);defaults=BuddingSystem().defaults()
    default_vector=np.array([defaults[k] for k in names]);reals=get_cached_real_params(defaults)
    digests={}
    for key in ROW_FIELDS:
        field=manifest['fields'][key];h=hashlib.sha256()
        h.update(field['dtype'].encode());h.update(str(tuple(field['shape'])).encode());digests[key]=h
    seen=set();candidate_ids=set();counts=Counter();groups=Counter();modes=Counter();sets=Counter()
    pmin=np.full(141,np.inf);pmax=-pmin.copy();wt=np.zeros(141,dtype=np.int64);zero=wt.copy()
    ymin=np.full(39,np.inf);ymax=-ymin.copy();peaks=[];negative_rows=0;raw_negative_rows=0
    retained_rescued=0;cursor=0;design_cache={}
    for chunk_index,segment in enumerate(manifest['segments']):
        source=manifest['sources'][segment['source']];path=directory/source['path']
        assert file_sha256(path)==source['sha256'],path
        with np.load(path,allow_pickle=False) as archive:
            block={key:archive[key] for key in ROW_FIELDS}
        data=block['data'];params=block['params'];initial=block['y_init'];n=len(data)
        assert data.shape==(n,39,500) and params.shape==(n,141) and initial.shape==(n,39)
        assert data.dtype==np.float32 and params.dtype==initial.dtype==np.float64
        assert np.isfinite(data).all() and np.isfinite(params).all() and np.isfinite(initial).all()
        assert not block['is_real'].any() and np.all(block['simulation_status']=='success')
        np.testing.assert_array_equal(block['sample_id'],np.arange(cursor+1,cursor+n+1))
        np.testing.assert_allclose(data[:,:,0],initial,atol=1e-7,rtol=1e-6)
        calculated=pattern_group_metadata(data)
        for key in ['pattern_group_codes','pattern_group_labels','pattern_group_ids','observed_division_counts']:
            np.testing.assert_array_equal(block[key],calculated[key],err_msg=key)
        np.testing.assert_array_equal(block['pattern_labels'],[analyze_mass_pattern(y[0]) for y in data])
        vectors=[]
        for i in range(n):
            overrides=json.loads(str(block['overrides_json'][i]));meta=json.loads(str(block['candidate_metadata_json'][i]))
            pairs=json.loads(str(block['mutation_pairs_json'][i]));diag=json.loads(str(block['simulation_diagnostics_json'][i]))
            expected=np.array([overrides.get(k,defaults[k]) for k in names])
            np.testing.assert_array_equal(params[i],expected)
            np.testing.assert_array_equal(initial[i],BuddingSystem.prepare_initial_conditions(overrides)[1])
            identity=int(block['candidate_ids'][i]);assert identity not in candidate_ids;candidate_ids.add(identity)
            assert identity==meta['source_candidate_id']
            vector=effective_inputs(overrides,defaults);signature=hashlib.sha256(vector.astype('<f8').tobytes()).digest()
            assert signature not in seen;seen.add(signature);vectors.append(vector)
            if kind=='mutation':
                assert block['mutation_available'][i] and block['mutation_pairs_available'][i] and pairs
                for gene,mode in pairs:require_mutation(gene,mode);assert gene!='MDT';modes[f'{gene}[{mode}]']+=1
                assert first_mutation_conflict(pairs) is None
                modules=sorted({g for g,m in pairs});level=len(modules)
                assert level==block['levels'][i]==block['requested_levels'][i]==block['module_counts'][i] and 1<=level<=3
                sets[';'.join(modules)]+=1;counts[f'L{level}']+=1
                assert overrides['mdt'] in [90.,150.,160.]
                if any(m in ('GAL-OE','GAL-MC') for g,m in pairs):assert overrides['mdt']==150.
                retained_rescued+=int(meta.get('rescued',False))
            else:
                assert not pairs and not block['mutation_available'][i] and not block['mutation_pairs_available'][i]
                assert block['levels'][i]==block['requested_levels'][i]==block['module_counts'][i]==-1
                design_id=int(block['lhs_design_ids'][i]);row=int(block['lhs_design_rows'][i])
                assert divmod(identity,config['count'])==(design_id,row)
                if design_id not in design_cache:
                    with np.load(directory/'designs'/f'design_{design_id:04d}.npz') as z:design_cache[design_id]=z['params']
                np.testing.assert_array_equal(params[i],design_cache[design_id][row])
            raw_negative_rows+=int(diag['negative_value_count']>0)
        vectors=np.array(vectors)
        for start in range(0,n,32):
            differences=np.abs(vectors[start:start+32,None,:]-reals[None,:,:])
            assert not np.all(differences<=1e-8+1e-5*np.abs(reals[None,:,:]),axis=2).any(),'real input in pool'
        for key,h in digests.items():
            value=np.ascontiguousarray(block[key].astype(manifest['fields'][key]['dtype'],copy=False))
            h.update(memoryview(value).cast('B'))
        pmin=np.minimum(pmin,params.min(0));pmax=np.maximum(pmax,params.max(0))
        wt+=(params==default_vector).sum(0);zero+=(params==0).sum(0)
        ymin=np.minimum(ymin,data.min(axis=(0,2)));ymax=np.maximum(ymax,data.max(axis=(0,2)))
        peaks.append(data.max(axis=2));negative_rows+=int((data<0).any(axis=(1,2)).sum())
        groups.update(map(str,block['pattern_group_codes']));counts.update(map(str,block['labels']));cursor+=n
        if (chunk_index+1)%50==0:print(f'{kind}: audited {cursor}/{config["count"]}',flush=True)
    assert cursor==config['count']==len(seen)==len(candidate_ids)
    for key,h in digests.items():assert h.hexdigest()==manifest['fields'][key]['sha256'],key
    # Also use the public reader for every scalar/schema field, verifying its
    # source hash and array digest with the same contract as training consumers.
    with open_dataset(directory/'dataset.json',allow_pickle=False) as archive:
        for key in archive.files:
            if key not in ROW_FIELDS:archive[key]
    peaks=np.concatenate(peaks)
    parameter_rows=[dict(parameter=k,WT=default_vector[j],minimum=pmin[j],maximum=pmax[j],
        WT_count=int(wt[j]),WT_fraction=float(wt[j]/cursor),zero_count=int(zero[j]),zero_fraction=float(zero[j]/cursor)) for j,k in enumerate(names)]
    variable_rows=[dict(variable=k,minimum=float(ymin[j]),maximum=float(ymax[j]),
        peak_median=float(np.median(peaks[:,j])),peak_p95=float(np.quantile(peaks[:,j],.95)),
        peak_p99=float(np.quantile(peaks[:,j],.99))) for j,k in enumerate(SPEC.variable_names)]
    write_csv(output/'parameter_statistics.csv',parameter_rows);write_csv(output/'variable_statistics.csv',variable_rows)
    if sets:write_csv(output/'module_set_counts.csv',[dict(modules=k,count=v) for k,v in sorted(sets.items())])
    if modes:write_csv(output/'mode_counts.csv',[dict(mode=k,count=v) for k,v in sorted(modes.items())])
    lhs_report=None
    if kind=='lhs':
        spec=json.loads((directory/'lhs_specification.json').read_text())
        rows,designs=audit_lhs(directory,spec,names);write_csv(output/'lhs_design_checks.csv',rows)
        lhs_report=dict(designs=designs,coordinates_checked=len(rows)*config['count'],
            maximum_cdf_probability_violation=max(r['quantile_probability_violation'] for r in rows),
            latent_design_hashes_valid=True,all_univariate_strata_valid=True)
    size=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    summary=dict(audited_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),pool=str(directory),kind=kind,
        rows=cursor,chunks=len(manifest['segments']),seed=config['seed'],status='passed',
        all_chunk_and_field_hashes_valid=True,unique_effective_inputs=len(seen),real_collisions_in_saved_pool=0,
        actual_initials_and_labels_valid=True,pattern_groups=dict(groups),biological_and_level_counts=dict(counts),
        retained_rescued=retained_rescued,represented_module_sets=len(sets),
        raw_negative_samples=raw_negative_rows,stored_float32_negative_samples=negative_rows,
        directory_bytes=size,directory_GiB=size/1024**3,lhs_design=lhs_report,
        simulation_counters=completion['counts'],
        note='completed_at in the generation record precedes final manifest hashing; publication is recorded by COMPLETE log lines')
    (output/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.pool,args.output)


if __name__=='__main__':main()
