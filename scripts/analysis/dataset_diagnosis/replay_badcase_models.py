"""Frozen real-mutant replay, metric agreement checks and channel attribution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.cross_dataset_badcases import DEFAULT_OUT,mid
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pure_random_evidence import VARIABLES


def replay(row,cache_path,out,device,batch_size,legacy_all_rows=False,historical_code=False):
    import torch
    from budding_yeast_v2.eval import load_model_from_checkpoint
    from budding_yeast_v2.data.normalization import normalize_y,normalize_p,denormalize_y
    from budding_yeast_v2.utils.metrics import calculate_metrics
    from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_six_patterns import features
    directory=out/('replay_historical' if historical_code else 'replay_all_rows' if legacy_all_rows else 'replay')/row.run_id;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'provenance.json').exists():return
    with np.load(cache_path,allow_pickle=False) as c:
        names=c['names'];real=np.flatnonzero(c['is_real']);synthetic=np.flatnonzero(~c['is_real'])
        truth=torch.from_numpy(c['real_y'].copy()).clamp(min=0)
        params=torch.from_numpy(c['params'].astype('float32'));ic=torch.from_numpy(c['ic'][real].copy())
        ym=c['global_y_mean'].astype('float32');ys=c['global_y_std'].astype('float32')
    checkpoint=torch.load(row.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    pp=checkpoint.get('preprocessing_contract')
    if pp:
        pm,ps=pp['p_mean'],pp['p_std'];ym,ys=pp['mean_y'],pp['std_y'];sc=pp.get('s_c');norm=pp['norm_type']
    else:
        pm=params[synthetic].mean(0,keepdim=True);ps=params[synthetic].std(0,keepdim=True)+1e-6;sc=None;norm=row.norm
    if legacy_all_rows and pp is None and norm in ('log1p','asinh_z'):
        transformed=torch.log1p(truth).double().numpy()
        n0=len(synthetic)*500;n1=len(real)*500
        m0=np.asarray(ym).reshape(-1).astype('float64');s0=np.asarray(ys).reshape(-1).astype('float64')-1e-6
        total=m0*n0+transformed.sum((0,2));square=s0*s0*(n0-1)+m0*m0*n0+(transformed*transformed).sum((0,2))
        ym=(total/(n0+n1)).astype('float32');ys=(np.sqrt(np.maximum((square-total*total/(n0+n1))/(n0+n1-1),0))+1e-6).astype('float32')
        pm=params.mean(0,keepdim=True);ps=params.std(0,keepdim=True)+(1e-6 if norm=='log1p' else 0)
    if historical_code and pp is None:
        archived=pd.read_csv(row.source);requested={mid(n) for n in archived.Mutant}
        keep=np.array([mid(names[i]) in requested for i in real]);missing=np.flatnonzero(~keep)
        if len(missing):
            transform=torch.log1p(truth[missing]).double().numpy()
            n0=len(synthetic)*500;n1=len(missing)*500;m0=np.asarray(ym).reshape(-1).astype('float64');s0=np.asarray(ys).reshape(-1).astype('float64')-1e-6
            total=m0*n0+transform.sum((0,2));square=s0*s0*(n0-1)+m0*m0*n0+(transform*transform).sum((0,2))
            ym=(total/(n0+n1)).astype('float32');ys=(np.sqrt(np.maximum((square-total*total/(n0+n1))/(n0+n1-1),0))+1e-6).astype('float32')
            truth=truth[keep];ic=ic[keep];real=real[keep];synthetic=np.setdiff1d(np.arange(len(params)),real)
            pm=params[synthetic].mean(0,keepdim=True);ps=params[synthetic].std(0,keepdim=True)+1e-6
    if norm=='asinh_z' and pp is None and not legacy_all_rows:ps=params[synthetic].std(0,keepdim=True)
    if norm=='asinh_scale' and pp is None:
        stat_path=out/'normalization'/f'{row.run_id}.npz'
        if not stat_path.exists():raise ValueError('Legacy asinh_scale statistics need explicit reconstruction')
        with np.load(stat_path,allow_pickle=False) as stats:
            pm=torch.from_numpy(stats['p_mean']);ps=torch.from_numpy(stats['p_std']);ym=stats['mean_y'];ys=stats['std_y'];sc=stats['s_c']
    if historical_code:
        import subprocess,types,importlib.util
        config=json.loads((Path(row.run)/'config.json').read_text());commit=config['git_commit']
        package_name='frozen_bio_'+commit;source_dir=out/'historical_code'/commit;source_dir.mkdir(exist_ok=True,parents=True)
        package=types.ModuleType(package_name);package.__path__=[str(source_dir)];sys.modules[package_name]=package
        for module in ('bio_mechanistic_operator','bio_refiner_plus'):
            source=subprocess.check_output(['git','show',f'{commit}:models/components/{module}.py'],cwd=ROOT)
            path=source_dir/f'{module}.py';path.write_bytes(source)
            spec=importlib.util.spec_from_file_location(package_name+'.'+module,path)
            loaded=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loaded;spec.loader.exec_module(loaded)
        hp=checkpoint['hyper_parameters'];constructor=sys.modules[package_name+'.bio_refiner_plus'].BioRefinerPlus
        network=constructor(num_vars=39,param_dim=141,width=hp['width'],modes=hp['modes'],n_blocks=hp.get('n_blocks',4))
        state={k.removeprefix('model.'):v for k,v in checkpoint['state_dict'].items() if k.startswith('model.')}
        network.load_state_dict(state,strict=True)
        model=types.SimpleNamespace(model=network,to=lambda device:None)
        network.to(device).eval()
    else:
        model,hp=load_model_from_checkpoint(row.checkpoint,device='cpu',graph_load_policy='checkpoint')
        model=model.to(device);model.eval()
    yn=normalize_y(ic[:,:,None],ym,ys,norm,sc).expand(-1,-1,500)
    grid=torch.linspace(0,1,500)[None,None].expand_as(yn)
    inputs=torch.stack((yn,grid),dim=2);pn=normalize_p(params[real],pm,ps,norm)
    predictions=[];coarse=[]
    with torch.inference_mode():
        for start in range(0,len(real),batch_size):
            x=inputs[start:start+batch_size].to(device);p=pn[start:start+batch_size].to(device)
            if row.model in ('biom_refiner_plus','biom_refiner_curvature'):
                y,c=model.model(x,p,return_coarse=True)
                coarse.append(denormalize_y(c,ym,ys,norm,sc).cpu())
            else:y=model(x,p)
            predictions.append(denormalize_y(y,ym,ys,norm,sc).cpu())
    prediction=torch.cat(predictions)
    predicted_features,_=features(prediction.numpy());true_features,_=features(truth.numpy())
    records=[];channels=[]
    for pos,idx in enumerate(real):
        metrics=calculate_metrics(prediction[pos],truth[pos]);delta=prediction[pos].double().numpy()-truth[pos].double().numpy()
        t=truth[pos].double().numpy();p=prediction[pos].double().numpy()
        mse=np.mean(delta*delta,axis=1);mae=np.mean(np.abs(delta),axis=1)
        rel=np.sqrt(np.sum(delta*delta,axis=1)/np.maximum(np.sum(t*t,axis=1),5.))
        low=np.sqrt(np.mean(t*t,axis=1))<.1
        top_mse=int(np.argmax(mse));top_rel=int(np.argmax(rel))
        record=dict(Mutant=names[idx],mutant_id=mid(names[idx]),run_id=row.run_id,**{k.replace(' ','_'):v for k,v in metrics.items()},
            dominant_mse_variable=VARIABLES[top_mse],dominant_mse_fraction=float(mse[top_mse]/max(mse.sum(),1e-30)),
            dominant_rel_variable=VARIABLES[top_rel],dominant_rel_fraction=float(rel[top_rel]/max(rel.sum(),1e-30)),
            low_amplitude_rel_fraction=float(rel[low].sum()/max(rel.sum(),1e-30)),
            tail_10pct_sse_fraction=float(np.sum(delta[:,-50:]**2)/max(np.sum(delta**2),1e-30)),
            log1p_MSE=float(np.mean((np.log1p(p)-np.log1p(t))**2)),
            true_divisions=int(true_features['divisions'][pos]),predicted_divisions=int(predicted_features['divisions'][pos]),
            true_pattern=true_features['pattern_code'][pos],predicted_pattern=predicted_features['pattern_code'][pos],
            truth_clb2_max=float(t[2].max()),prediction_clb2_max=float(p[2].max()),
            truth_ori_max=float(t[33].max()),prediction_ori_max=float(p[33].max()))
        if coarse:
            cp=torch.cat(coarse)[pos];record['coarse_MAE']=float((cp-truth[pos]).abs().mean());record['coarse_MSE']=float(((cp-truth[pos])**2).mean())
        records.append(record)
        for v,name in enumerate(VARIABLES):
            channels.append(dict(run_id=row.run_id,mutant_id=mid(names[idx]),variable=name,MAE=mae[v],MSE=mse[v],Relative_L2=rel[v],
                mse_share=mse[v]/max(mse.sum(),1e-30),rel_share=rel[v]/max(rel.sum(),1e-30),
                truth_rms=np.sqrt(np.mean(t[v]*t[v])),truth_max=t[v].max(),pred_max=p[v].max(),
                truth_end=t[v,-1],pred_end=p[v,-1]))
    frame=pd.DataFrame(records)
    historical=pd.read_csv(row.source) if row.kind=='static' else pd.DataFrame(json.loads(Path(row.source).read_text())['per_mutant'])
    historical['mutant_id']=historical.Mutant.map(mid)
    joined=frame.merge(historical[['mutant_id','MAE','MSE','Relative_L2','Correlation']],on='mutant_id',suffixes=('_replay','_saved'),validate='one_to_one')
    differences={key:float(np.max(np.abs(joined[key+'_replay']-joined[key+'_saved']))) for key in ('MAE','MSE','Relative_L2','Correlation')}
    limits=dict(MAE=max(.005,.01*historical.MAE.max()),MSE=max(.02,.01*historical.MSE.max()),Relative_L2=max(.005,.01*historical.Relative_L2.max()),Correlation=.02)
    same_conditions=set(frame.mutant_id)==set(historical.mutant_id)
    matched=same_conditions and all(differences[k]<=limits[k] for k in differences)
    frame.to_csv(directory/'replayed_metrics.csv',index=False);pd.DataFrame(channels).to_csv(directory/'channel_errors.csv',index=False)
    np.savez_compressed(directory/'predictions.npz',predictions=prediction.numpy(),targets=truth.numpy(),names=names[real])
    provenance=dict(run_id=row.run_id,checkpoint=row.checkpoint,source_metrics=row.source,device=device,
        statistics_source='checkpoint_preprocessing' if pp else ('git_version_and_archived_real_partition' if historical_code else 'candidate_legacy_all_rows' if legacy_all_rows else 'reconstructed_legacy_all_synthetic'),
        graph_policy='checkpoint',same_real_condition_set=same_conditions,matched_archived_metrics_within_tolerance=matched,max_abs_differences=differences,tolerances=limits)
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('replay',row.dataset_tag,row.model,row.strategy,'matched',matched,'differences',differences,flush=True)
    del model,prediction;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--device',default='cuda:5');parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--historical-model-code',action='store_true');parser.add_argument('--legacy-all-rows',action='store_true');parser.add_argument('--only',default='');parser.add_argument('--wait-for-features',action='store_true');args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    runs=pd.read_csv(args.output/'primary_runs.csv').fillna('')
    if args.only:runs=runs[runs.dataset_tag.str.contains(args.only,regex=False)]
    waiting=[row for _,row in runs.iterrows()];errors=[]
    while waiting:
        later=[]
        for row in waiting:
            path=args.output/'features'/f'{hashlib.sha256(row.dataset.encode()).hexdigest()[:12]}.npz'
            if not path.exists() or not zipfile.is_zipfile(path):
                if args.wait_for_features:later.append(row);continue
                raise FileNotFoundError(path)
            try:replay(row,str(path),args.output,args.device,args.batch_size,args.legacy_all_rows,args.historical_model_code)
            except Exception as exc:
                import traceback
                traceback.print_exc();errors.append(dict(run_id=row.run_id,error=repr(exc)))
        waiting=later
        if waiting:time.sleep(10)
    (args.output/'replay_errors.json').write_text(json.dumps(errors,indent=2))


if __name__=='__main__':main()
