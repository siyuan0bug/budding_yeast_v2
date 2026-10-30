"""Recent, complete synthetic-edge feedback calibrates response uncertainty."""
from pathlib import Path
import json
import numpy as np

SETTINGS=dict(min_edges=32,n_estimators=32,max_depth=12,min_samples_leaf=8,
              max_threads=2,history='most_recent_acquisition',epsilon=1e-8,
              max_log_correction=8.,rank_multiplier=(1.,2.))


def calibrated_multiplier(engine,features,seed):
    if not len(features):return np.ones(0,dtype=np.float32),dict(status='no_edges',complete_edges=0)
    history=getattr(engine,'_hybrid_edge_history',[])
    if not history or len(history[-1]['features'])<SETTINGS['min_edges']:
        return np.ones(len(features),dtype=np.float32),dict(status='cold_start',complete_edges=0)
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_squared_error
    from scipy.stats import rankdata
    last=history[-1];x=last['features'];variance=last['log_variance'];error=last['observed_error']
    eps=SETTINGS['epsilon']
    target=np.clip(np.log(error+eps)-np.log(variance+eps),
                   -SETTINGS['max_log_correction'],SETTINGS['max_log_correction'])
    rng=np.random.RandomState(seed);order=rng.permutation(len(x));cut=max(1,int(.8*len(x)))
    model=ExtraTreesRegressor(n_estimators=SETTINGS['n_estimators'],max_depth=SETTINGS['max_depth'],
        min_samples_leaf=SETTINGS['min_samples_leaf'],n_jobs=SETTINGS['max_threads'],random_state=seed)
    model.fit(x[order[:cut]],target[order[:cut]])
    test=order[cut:]
    corrected=np.exp(np.clip(model.predict(x[test]),-8.,8.))*(variance[test]+eps)
    report=dict(status='fitted',complete_edges=len(x),history_acquisition=int(last['acquisition']),
        calibration_mse=float(mean_squared_error(error[test],corrected)),
        uncalibrated_mse=float(mean_squared_error(error[test],variance[test])),
        note='Prior complete synthetic edges only; random prior holdout is not a pool-wide estimate')
    model.fit(x,target)
    predicted=np.concatenate([model.predict(features[i:i+8192]) for i in range(0,len(features),8192)])
    multiplier=1.+rankdata(predicted,method='average')/max(1,len(predicted))
    return multiplier.astype(np.float32),report


def prepare_edges(engine,records,edges,features,log_variance,node_count):
    """Capture acquisition-time features; target labels arrive only after query."""
    encoded=edges[:,0].astype(np.int64)*node_count+edges[:,1]
    ids=engine.pool.ensure_ids();pool_size=engine._pool_size;rows={}
    for record in records:
        if record['kind']!='response':continue
        a,b=map(int,record['endpoints']);position=int(np.searchsorted(encoded,a*node_count+b))
        if position>=len(edges) or tuple(edges[position])!=(a,b):
            raise ValueError('Selected response does not match its legal edge')
        stable=tuple(int(ids[i]) if i<pool_size else -1-(i-pool_size) for i in (a,b))
        rows[stable]=(features[position].copy(),float(log_variance[position]))
    engine._pending_hybrid_edges=dict(acquisition=int(engine._al4pde_al_iter),rows=rows)


def observe_edges(engine,batch):
    pending=getattr(engine,'_pending_hybrid_edges',None)
    if pending is None:return
    del engine._pending_hybrid_edges
    if batch.data is None:return
    index=pending['acquisition']
    outputs=getattr(engine,'_query_feedback_outputs',{}).get(index,[])
    if not outputs:return
    features=[];variance=[];errors=[];endpoints=[]
    for action in outputs[-1]['action_errors']:
        key=tuple(map(int,action['endpoints']))
        if action['kind']!='response' or key not in pending['rows']:continue
        x,u=pending['rows'][key]
        features.append(x);variance.append(u);errors.append(action['observed_response_error']);endpoints.append(key)
    if not features:return
    row=dict(acquisition=index,features=np.asarray(features,dtype=np.float32),
        log_variance=np.asarray(variance,dtype=np.float32),observed_error=np.asarray(errors,dtype=np.float32),
        endpoints=np.asarray(endpoints,dtype=np.int64))
    if not hasattr(engine,'_hybrid_edge_outputs'):engine._hybrid_edge_outputs={}
    engine._hybrid_edge_outputs.setdefault(index,[]).append(row)
    # Multiple refill batches belong to ONE acquisition, not separate histories.
    batches=engine._hybrid_edge_outputs[index]
    merged={key:np.concatenate([b[key] for b in batches]) for key in ('features','log_variance','observed_error','endpoints')}
    merged['acquisition']=index
    if not hasattr(engine,'_hybrid_edge_history'):engine._hybrid_edge_history=[]
    if engine._hybrid_edge_history and engine._hybrid_edge_history[-1]['acquisition']==index:
        engine._hybrid_edge_history[-1]=merged
    else:engine._hybrid_edge_history.append(merged)
    engine._hybrid_edge_history=engine._hybrid_edge_history[-1:]


def write_edges(engine,directory,index):
    output=getattr(engine,'_hybrid_edge_outputs',{}).pop(index,[])
    if not output:return
    path=Path(directory);fields={key:np.concatenate([b[key] for b in output])
        for key in ('features','log_variance','observed_error','endpoints')}
    fields['acquisition']=np.int64(index)
    temporary=path/'edge_feedback.npz.tmp'
    with temporary.open('wb') as handle:np.savez_compressed(handle,**fields)
    temporary.replace(path/'edge_feedback.npz')
    schema=json.dumps(dict(version='hybrid_edges_v1',
        settings=SETTINGS,source='complete previously queried synthetic edges; no single-point fallback labels',
        feature_count=int(fields['features'].shape[1]),edges=int(len(fields['features']))),indent=2)
    temporary=path/'edge_feedback_schema.json.tmp'
    temporary.write_text(schema)
    temporary.replace(path/'edge_feedback_schema.json')
