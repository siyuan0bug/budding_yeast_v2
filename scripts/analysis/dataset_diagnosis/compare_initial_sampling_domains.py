"""Compare old/new physical AL initialization domains without training or filtering."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd


def block_statistics(data):
    values=np.asarray(data,dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError('Nonfinite trajectory')
    log=np.log1p(np.maximum(values,0.))
    return values.max(2),values.min(2),log.sum(2),(log*log).sum(2)


def inspect(path,workers):
    print(f'Reading trajectories: {path}',flush=True)
    with np.load(path,allow_pickle=False) as z:
        result={key:z[key] for key in ('is_real','split_train','split_val','split_test','split_real',
            'module_counts','params','param_names','variable_names','pattern_group_codes')}
        data=z['data'];result['steps']=data.shape[-1]
        result['fingerprint']=str(z.get('sampling_range_fingerprint','legacy_unrecorded'))
    blocks=[slice(i,min(i+128,len(data))) for i in range(0,len(data),128)]
    matrices=[np.empty((len(data),39),dtype=np.float64) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index,values in enumerate(executor.map(lambda rows:block_statistics(data[rows]),blocks)):
            for target,block in zip(matrices,values):target[blocks[index]]=block
    result.update(zip(('peaks','minima','log_sum','log_square_sum'),matrices))
    return result


def sigma(saved,rows):
    size=len(rows)*saved['steps']
    total=saved['log_sum'][rows].sum(0);square=saved['log_square_sum'][rows].sum(0)
    return np.sqrt(np.maximum((square-total**2/size)/(size-1),0.))+1e-6


def json_value(value):
    if isinstance(value,(np.integer,np.floating)):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new',type=Path,required=True);p.add_argument('--old',type=Path,required=True)
    p.add_argument('--pool',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.workers<1:p.error('--workers must be positive')
    datasets={tag:inspect(path,a.workers) for tag,path in [('old',a.old),('new',a.new)]}
    names=list(datasets['new']['variable_names'])
    assert names==list(datasets['old']['variable_names'])
    real_max=datasets['new']['peaks'][datasets['new']['split_real']].max(0)
    np.testing.assert_allclose(real_max,datasets['old']['peaks'][datasets['old']['split_real']].max(0),rtol=1e-6,atol=1e-6)
    stages=[];variables=[];levels=[];summaries={};bounds=[]
    for tag,saved in datasets.items():
        peaks=saved['peaks'];legacy=(peaks[:,1:]>np.maximum(2*real_max[1:],2.)).any(1)|(peaks[:,0]>10.)
        literal=(peaks>2*real_max).any(1)
        for scope,indices in [('train',saved['split_train']),('all_synthetic',np.flatnonzero(~saved['is_real'])),
                              ('val',saved['split_val']),('test',saved['split_test'])]:
            stages.append(dict(cohort=tag,scope=scope,samples=len(indices),legacy_high=int(legacy[indices].sum()),
                legacy_high_fraction=float(legacy[indices].mean()),literal_two_real=int(literal[indices].sum()),
                mass_above_10=int((peaks[indices,0]>10).sum()),
                negative_samples=int((saved['minima'][indices]<0).any(1).sum()),minimum=float(saved['minima'][indices].min())))
        rows=saved['split_train'];std=sigma(saved,rows)
        summaries[tag]=dict(train_samples=len(rows),legacy_high=int(legacy[rows].sum()),
            legacy_high_fraction=float(legacy[rows].mean()),range_fingerprint=saved['fingerprint'])
        for j,name in enumerate(names):
            values=peaks[rows,j]
            variables.append(dict(cohort=tag,variable=name,real_max=real_max[j],p50=np.quantile(values,.5),
                p90=np.quantile(values,.9),p95=np.quantile(values,.95),p99=np.quantile(values,.99),maximum=values.max(),
                above_two_real=int((values>2*real_max[j]).sum()),
                above_legacy_variable_limit=int((values>(10. if j==0 else max(2*real_max[j],2.))).sum()),
                log_std=std[j]))
        for level in (1,2,3):
            chosen=rows[saved['module_counts'][rows]==level]
            levels.append(dict(cohort=tag,level=level,count=len(chosen),legacy_high=int(legacy[chosen].sum()),
                legacy_high_fraction=float(legacy[chosen].mean()) if len(chosen) else None))
        synth=np.flatnonzero(~saved['is_real'])
        for name,key in [('TEM1','Tem1T'),('CDC15','Cdc15T'),('ESP1','ESP1T')]:
            j=names.index(name);total=saved['params'][:,list(saved['param_names']).index(key)]
            bad=synth[peaks[synth,j]>total[synth]+1e-3*np.maximum(1.,total[synth])]
            bounds.append(dict(cohort=tag,variable=name,count=len(bad),rows=bad.tolist()))
        np.savez_compressed(a.output/f'{tag}_moments.npz',peaks=peaks,minima=saved['minima'],
            split_train=rows,legacy_high=legacy,is_real=saved['is_real'],log_std=std)
    print('Reading GAL-MC pool accounting',flush=True)
    with np.load(a.pool,allow_pickle=False) as z:
        report=json.loads(str(z['generation_report']));pairs=z['mutation_pairs_json']
        selected=np.flatnonzero(np.char.find(pairs,'"GAL-MC"')>=0)
        params=z['params'];keys=list(z['param_names'])
        copies=params[selected,keys.index('ksb2p')]/.12
        assert np.allclose(copies,np.round(copies),atol=1e-10,rtol=0)
        assert np.isin(np.round(copies),np.arange(2,11)).all()
        assert np.all(params[selected,keys.index('mdt')]==150.)
        assert np.all(params[selected,keys.index('ksb2pp')]==.04)
        singles=[r for r in report['allele_counts'] if r['modules']=='CLB2' and r['alleles']=='CLB2[GAL-MC]']
        assert sum(r['accepted'] for r in singles)==8
        gal=dict(total_pool_samples=len(params),gal_mc_samples=len(selected),
            copies_histogram=dict(Counter(map(int,np.round(copies)))),single_module_GAL_MC=singles,
            module_level_counts=report['actual_level_counts'])
    pd.DataFrame(stages).to_csv(a.output/'stage_extremes.csv',index=False)
    pd.DataFrame(variables).to_csv(a.output/'variable_extrema.csv',index=False)
    pd.DataFrame(levels).to_csv(a.output/'level_extremes.csv',index=False)
    summary=dict(datasets=summaries,bound_checks=bounds,gal_mc=gal,
        note='Historical old/new initial domains; no filtering and no model training. Multiple input-domain changes occur together.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,default=json_value)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    frame=pd.DataFrame(variables)
    for ax,name in zip(axes.flat,['CLN2','CLB2','SIC1','CDC6','ORI','BUD']):
        group=frame[frame.variable==name].set_index('cohort')
        for i,tag in enumerate(['old','new']):
            ax.bar(np.arange(3)+(i-.5)*.34,[group.loc[tag,k] for k in ('p95','p99','maximum')],width=.34,label=tag)
        ax.axhline(group.real_max.iloc[0],c='black',ls='--',lw=1,label='real maximum')
        ax.set_xticks(range(3),['P95','P99','Maximum']);ax.set_yscale('log');ax.set_title(name);ax.grid(axis='y',alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Initial synthetic training set: physical trajectory peaks (10,000 samples each)')
    fig.savefig(a.output/'initial_extremes_comparison.png',dpi=180);fig.savefig(a.output/'initial_extremes_comparison.pdf');plt.close(fig)
    old,new=summaries['old'],summaries['new']
    lines=['# Initial sampling domain comparison','',
        f"Old train high: {old['legacy_high']}/10000 ({old['legacy_high_fraction']:.2%}).",
        f"New train high: {new['legacy_high']}/10000 ({new['legacy_high_fraction']:.2%}).",'',
        'Legacy high: any non-MASS peak > max(2 * real maximum, 2), or MASS > 10.',
        'This is a diagnostic flag. No high-amplitude samples were removed.',
        'Both initial sets use the same size and split seed, but different candidate pools; this is not a single-factor ablation.',
        'No new FNO accuracy result is claimed.','',
        f"GAL-MC in full pool: {len(selected)}; single-module unique non-real inputs: 8."]
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2,default=json_value),flush=True)
    if any(r['count'] for r in bounds if r['cohort']=='new'):
        raise ValueError('New trajectories violate total-pool bounds; review before formal training')


if __name__=='__main__':main()
