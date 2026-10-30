"""Compare a fixed historical Top20 against old/new unsolved pools, in blocks.

Counts reproduce audit_pool_badcase_coverage.box exactly. Supplementary IC/joint
counts and nearest distances do not claim trajectory or prediction similarity.
No test identities enter the generator or its quota allocation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pool_badcase_coverage import box
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.badcase_support import genes
from budding_yeast_v2.scripts.analysis.audit_real_mutant_module_counts import count_rows
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.cross_dataset_badcases import mid
from budding_yeast_v2.data.systems.budding.al_pool import POOL_CONTRACT
from budding_yeast_v2.data.systems.budding.catalog import CATALOG_VERSION


def identity(path):
    stat=path.stat()
    return dict(path=str(path.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    data=Path('/home/users/zsy/mydata/budding_yeast')
    oldaudit=data/'analysis/badcase_audit_2026-10-27'
    parser.add_argument('--old-pool',type=Path,default=data/'budding_pool_1000000_seed42_al.npz')
    parser.add_argument('--new-pool',type=Path,default=data/'budding_pool_1000000_seed42_balanced.npz')
    parser.add_argument('--real-dataset',type=Path,default=data/'budding_initial_train10000_seed42_al.npz')
    parser.add_argument('--badcases',type=Path,default=oldaudit/'twenty_badcase_evidence.csv')
    parser.add_argument('--output',type=Path,default=data/'analysis/balanced_pool_2026-10-28')
    parser.add_argument('--block-size',type=int,default=32768)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases=pd.read_csv(args.badcases,dtype={'mutant_id':str}).head(20)
    if len(cases)!=20 or cases.mutant_id.nunique()!=20:raise ValueError('A fixed unique Top20 is required')
    definitions={row['identity']:row for row in count_rows('budding')[0]}
    with np.load(args.real_dataset,allow_pickle=False) as z:
        flags=z['is_real'];realp=z['params'][flags];realnames=z['mutant_names'][flags]
        realic=z['y_init'][flags] if 'y_init' in z else z['data'][flags,:,0]
        keys=z['param_names']
    lookup={mid(name):i for i,name in enumerate(realnames)}
    wt=realp[lookup['1']];mdt=int(np.flatnonzero(keys=='mdt')[0])
    allrows=[];examples=[];pool_reports={}
    reference_std=None
    for tag,path in [('old',args.old_pool),('balanced',args.new_pool)]:
        print('loading',tag,path,flush=True)
        with np.load(path,allow_pickle=False) as z:
            p=z['params'];initial=z['y_init'];names=z['mutant_names'];levels=z['module_counts']
            if not np.array_equal(z['param_names'],keys):raise ValueError('Parameter order mismatch')
            if tag=='old':reference_std=np.where(z['p_std']<1e-8,1.,z['p_std'])
            else:
                if str(z['pool_generation_contract'])!=POOL_CONTRACT or str(z['catalog_version'])!=CATALOG_VERSION:raise ValueError('Wrong balanced contract/catalog')
                report=json.loads(str(z['generation_report']))
                pool_reports[tag]=report
                pd.DataFrame(report['module_quotas']).to_csv(args.output/'module_quotas.csv',index=False)
                pd.DataFrame(report['allele_counts']).to_csv(args.output/'allele_counts.csv',index=False)
                assert 'data' not in z and 'labels' not in z and 'pattern_labels' not in z
                assert p.shape==(1000000,141) and initial.shape==(1000000,39)
                assert np.isfinite(p).all() and np.isfinite(initial).all()
                assert all(row['quota']==row['accepted'] for row in report['module_quotas'])
                assert report['global_unique_inputs']==len(p)
                required={'mutation_descriptions','mutation_pairs_json','overrides_json','candidate_metadata_json',
                    'levels','requested_levels','module_counts','sample_id','simulation_status'}
                assert required<=set(z.files)
                # Independent raw P+IC uniqueness audit; no trajectory simulation.
                raw=np.ascontiguousarray(np.concatenate((p,initial),axis=1)).view(np.dtype((np.void,(141+39)*8))).reshape(-1)
                if len(np.unique(raw))!=len(p):raise ValueError('Duplicate exported P+IC rows')
                del raw
        print('decoding module sets',tag,flush=True)
        modulekeys=np.asarray([';'.join(sorted(genes(name))) for name in names])
        if tag=='balanced':
            observed=Counter(modulekeys)
            if any(observed[row['modules']]!=row['accepted'] for row in report['module_quotas']):
                raise ValueError('Exported module counts differ from quota report')
            if not np.array_equal(np.bincount(levels,minlength=4)[1:],[9815,126225,863960]):
                raise ValueError('Unexpected million-pool L totals')
        for _,case in cases.iterrows():
            condition=case.mutant_id;index=lookup[condition];q=realp[index];ic=realic[index]
            target=definitions[condition]['modules'];same=modulekeys==target
            count=Counter(pool_same_modules=int(same.sum()))
            nearest=(float('inf'),-1);nearest_carbon=(float('inf'),-1)
            for start in range(0,len(p),args.block_size):
                end=min(start+args.block_size,len(p));block=p[start:end]
                carbon=np.isclose(block[:,mdt],q[mdt],rtol=0,atol=1e-5)
                strict=box(block,q,wt,mdt,.25);wide=box(block,q,wt,mdt,.5);relaxed=box(block,q,wt,mdt,.25,.01)
                icmatch=np.isclose(initial[start:end],ic,rtol=1e-6,atol=1e-8).all(1)
                count.update(pool_same_modules_carbon=int((same[start:end]&carbon).sum()),
                    pool_local25=int(strict.sum()),pool_local50=int(wide.sum()),
                    pool_nearzero_local25=int(relaxed.sum()),
                    pool_local25_with_IC=int((strict&icmatch).sum()),
                    pool_nearzero_local25_with_IC=int((relaxed&icmatch).sum()))
                distance=np.sqrt(np.mean(((block-q)/reference_std)**2,axis=1))
                j=int(np.argmin(distance))
                if distance[j]<nearest[0]:nearest=(float(distance[j]),start+j)
                distance[~carbon]=np.inf;j=int(np.argmin(distance))
                if distance[j]<nearest_carbon[0]:nearest_carbon=(float(distance[j]),start+j)
            row=dict(pool=tag,mutant_id=condition,Mutant=case.Mutant,modules=target,
                real_module_count=definitions[condition]['module_count'],pool_size=len(p),**dict(count),
                nearest_common_z_rms=nearest[0],nearest_carbon_common_z_rms=nearest_carbon[0])
            allrows.append(row)
            for scope,(distance,i) in [('all',nearest),('same_carbon',nearest_carbon)]:
                changed=np.abs(p[i]-q)>1e-10
                examples.append(dict(pool=tag,mutant_id=condition,scope=scope,candidate_index=i,
                    candidate_name=str(names[i]),distance=distance,
                    initial_max_abs_difference=float(np.abs(initial[i]-ic).max()),
                    parameter_differences_json=json.dumps({str(k):dict(candidate=float(v),real=float(t)) for k,v,t in zip(keys[changed],p[i,changed],q[changed])})))
            print(tag,condition,dict(count),flush=True)
        del p,initial,names,levels,modulekeys
    frame=pd.DataFrame(allrows);frame.to_csv(args.output/'coverage_by_pool.csv',index=False)
    comparison=frame[frame.pool.eq('old')].merge(frame[frame.pool.eq('balanced')],on='mutant_id',suffixes=('_old','_balanced'))
    comparison.to_csv(args.output/'top20_comparison.csv',index=False)
    pd.DataFrame(examples).to_csv(args.output/'nearest_candidates.csv',index=False)
    fields=['pool_same_modules','pool_same_modules_carbon','pool_local25','pool_local50','pool_nearzero_local25','pool_local25_with_IC','pool_nearzero_local25_with_IC']
    outcomes={field:dict(improved=int((comparison[field+'_balanced']>comparison[field+'_old']).sum()),
        unchanged=int((comparison[field+'_balanced']==comparison[field+'_old']).sum()),
        decreased=int((comparison[field+'_balanced']<comparison[field+'_old']).sum()),
        old_total=int(comparison[field+'_old'].sum()),balanced_total=int(comparison[field+'_balanced'].sum())) for field in fields}
    provenance=dict(inputs=[identity(p) for p in [args.old_pool,args.new_pool,args.real_dataset,args.badcases]],
        cases=cases.mutant_id.tolist(),historical_badcase_sha256=hashlib.sha256(args.badcases.read_bytes()).hexdigest(),
        parameter_box='historical max(factor*abs(real), .05*abs(WT)); exact zeros and same mdt; relaxed zeros allow .01*abs(WT)',
        nearest_metric='RMS parameter distance standardized by OLD pool p_std for both pools',
        IC_match='rtol=1e-6, atol=1e-8, all 39 raw initial states',
        pool_generation_contract=POOL_CONTRACT,catalog_version=CATALOG_VERSION,
        outcomes=outcomes,trajectories_simulated=False,training_started=False)
    (args.output/'coverage_summary.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False))
    print(json.dumps(outcomes,indent=2),flush=True)


if __name__=='__main__':main()
