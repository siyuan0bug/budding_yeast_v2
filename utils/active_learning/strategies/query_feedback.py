"""Pre-query synthetic error feedback, stable across normalization refits.

Fixed physical-input Z-score features and log1p physical trajectory errors are
used across rounds. Predictions are recorded BEFORE the oracle or dataset append.
Only accepted synthetic query labels and explicitly identified training anchors
can produce feedback. Held-out and real labels have no entry point here.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

SETTINGS = dict(time_points=32, min_feedback_rows=32, n_estimators=32, max_depth=12,
                min_samples_leaf=8, max_threads=2, feedback_fit_fraction=.8,
                trajectory_units='physical_log1p', feature_fit='first_remaining_pool_inputs')


def fixed_input_features(engine, params, initials):
    values = np.concatenate((params, initials), axis=1).astype(np.float32)
    if not hasattr(engine, '_query_feature_stats'):
        # Fit once on remaining pool inputs, not output labels or real inputs.
        pool = values[:int(engine._pool_size)]
        mean = pool.mean(0, dtype=np.float64)
        scale = pool.std(0, dtype=np.float64)
        scale[scale < 1e-8] = 1.
        engine._query_feature_stats = mean.astype(np.float32), scale.astype(np.float32)
    mean, scale = engine._query_feature_stats
    return (values-mean)/scale


def predict_nodes(engine, dm, device, nodes, time_indices=None):
    """Ensemble mean in physical log1p space; no labels, restore model state."""
    nodes = np.asarray(nodes,dtype=np.int64)
    members = engine._ensemble_members
    if members is None or len(members)<2:
        raise RuntimeError('Experimental acquisition requires trained independent members')
    modes = [m.training for m in members]
    devices = [next(m.parameters(),torch.empty(0)).device for m in members]
    train_x, train_p = dm.train_dataset_synth.tensors[:2]
    result = None
    try:
        for member in members:
            member.to(device).eval()
        with torch.no_grad():
            for start in range(0,len(nodes),128):
                local = nodes[start:start+128]
                # Build each domain separately, then restore the caller's order.
                out = None
                for mask in (local<engine._pool_size, local>=engine._pool_size):
                    positions = np.flatnonzero(mask)
                    if not len(positions):continue
                    rows = local[mask]
                    if rows[0]<engine._pool_size:
                        x,p = engine._get_pool_X_batch(rows),engine._pool_P[rows]
                    else:
                        rows = rows-engine._pool_size
                        x,p = train_x[rows],train_p[rows]
                    predictions = torch.stack([m.forward_ic_time(x.to(device=device,dtype=torch.float32),
                                                  p.to(device=device,dtype=torch.float32)).float() for m in members])
                    if not torch.isfinite(predictions).all():
                        raise ValueError('Nonfinite pre-query prediction')
                    if time_indices is None:
                        time_indices = np.linspace(0,predictions.shape[-1]-1,min(SETTINGS['time_points'],predictions.shape[-1])).round().astype(int)
                    # Transform each member, THEN average; same units as feedback targets.
                    physical = dm.denormalize_y(predictions.flatten(0,1),device)
                    physical = physical.reshape(*predictions.shape)
                    logs = torch.log1p(physical.clamp(min=0)).mean(0)[...,time_indices].cpu().numpy()
                    if out is None:out=np.empty((len(local),*logs.shape[1:]),dtype=np.float32)
                    out[positions]=logs
                if result is None:result=np.empty((len(nodes),*out.shape[1:]),dtype=np.float32)
                result[start:start+len(local)] = out
    finally:
        for member,mode,original in zip(members,modes,devices):
            member.to(original).train(mode)
    if result is None:
        result=np.empty((0,39,0),dtype=np.float32)
    return result, np.asarray(time_indices if time_indices is not None else [],dtype=np.int64)


def predict_error(engine, features, target='point_error'):
    """Fit only previous pre-query outcomes; return None on cold start."""
    history = getattr(engine,'_query_feedback_history',[])
    if not history:return None,dict(status='cold_start',feedback_rows=0)
    x = np.concatenate([r['input_features'] for r in history])
    y = np.concatenate([r[target] for r in history])
    if len(y)<SETTINGS['min_feedback_rows']:return None,dict(status='insufficient_feedback',feedback_rows=len(y))
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_squared_error
    seed=int(engine.random_seed+97*engine._al4pde_al_iter)
    rng=np.random.RandomState(seed);order=rng.permutation(len(y));cut=max(1,int(SETTINGS['feedback_fit_fraction']*len(y)))
    def regressor():
        return ExtraTreesRegressor(n_estimators=SETTINGS['n_estimators'],max_depth=SETTINGS['max_depth'],
            min_samples_leaf=SETTINGS['min_samples_leaf'],
            n_jobs=max(1,min(SETTINGS['max_threads'],int(os.environ.get('AL_SIM_WORKERS','2')))),random_state=seed)
    model=regressor();model.fit(x[order[:cut]],np.log1p(y[order[:cut]]))
    held=order[cut:];prediction=np.maximum(np.expm1(model.predict(x[held])),0)
    report=dict(status='fitted',feedback_rows=len(y),target=target,
        feedback_holdout_mse=float(mean_squared_error(y[held],prediction)),
        constant_holdout_mse=float(mean_squared_error(y[held],np.full(len(held),y[order[:cut]].mean()))),
        note='Random held-out prior synthetic queries; not an unbiased pool error estimate')
    model.fit(x,np.log1p(y))
    result=np.concatenate([np.maximum(np.expm1(model.predict(features[i:i+8192])),0)
                           for i in range(0,len(features),8192)])
    return result,report


def prepare_feedback(engine, dm, device, selected, records, features):
    """Capture stable IDs and predictions before engine.query prunes the pool."""
    endpoints=sorted(set(i for r in records for i in r['endpoints']))
    predictions,times=predict_nodes(engine,dm,device,endpoints)
    lookup={node:i for i,node in enumerate(endpoints)}
    ids=engine.pool.ensure_ids()
    node_ids={i:int(ids[i]) if i<engine._pool_size else -1-(i-engine._pool_size) for i in endpoints}
    actual_anchors={}
    for node in endpoints:
        if node>=engine._pool_size:
            value=dm.train_dataset_synth.tensors[2][node-engine._pool_size:node-engine._pool_size+1]
            physical=dm.denormalize_y(value,'cpu')
            actual_anchors[node_ids[node]]=torch.log1p(physical.clamp(min=0))[0,:,times].cpu().numpy()
    engine._pending_query_feedback=dict(candidate_ids=np.asarray([int(ids[i]) for i in selected]),
        input_features=features[selected].copy(),predictions={node_ids[i]:predictions[lookup[i]] for i in endpoints},
        anchors=actual_anchors,times=times,
        actions=[dict(kind=r['kind'],endpoints=[node_ids[i] for i in r['endpoints']]) for r in records])


def observe_query(engine,batch):
    """Called immediately after the oracle, before append/refit. Consume once."""
    pending=getattr(engine,'_pending_query_feedback',None)
    if pending is None:return
    del engine._pending_query_feedback
    if batch.data is None:return
    data=batch.data;accepted=np.asarray(data['candidate_ids'],dtype=np.int64)
    if len(np.unique(accepted))!=len(accepted):raise ValueError('Duplicate feedback candidate IDs')
    selected_lookup={int(i):j for j,i in enumerate(pending['candidate_ids'])}
    positions=np.asarray([selected_lookup[int(i)] for i in accepted])
    actual=np.log1p(np.maximum(np.asarray(data['Y_real'],dtype=np.float32)[...,pending['times']],0))
    predicted=np.asarray([pending['predictions'][int(i)] for i in accepted])
    point_error=np.mean((predicted-actual)**2,axis=(1,2))
    actual_by_id=dict(pending['anchors']);actual_by_id.update(zip(map(int,accepted),actual))
    sums=np.zeros(len(accepted));counts=np.zeros(len(accepted),dtype=int)
    accepted_lookup={int(i):j for j,i in enumerate(accepted)};action_errors=[]
    for action in pending['actions']:
        nodes=action['endpoints']
        if len(nodes) not in (2,4) or not all(i in actual_by_id for i in nodes):continue
        def delta(values):
            response=values[nodes[1]]-values[nodes[0]]
            return values[nodes[3]]-values[nodes[2]]-response if len(nodes)==4 else response
        error=float(np.mean((delta(pending['predictions'])-delta(actual_by_id))**2))
        action_errors.append(dict(**action,observed_response_error=error))
        for i in set(nodes)&accepted_lookup.keys():
            j=accepted_lookup[i];sums[j]+=error;counts[j]+=1
    response_error=np.divide(sums,counts,out=point_error.astype(float).copy(),where=counts>0).astype(np.float32)
    row=dict(candidate_ids=accepted,input_features=pending['input_features'][positions],
             point_error=point_error,response_error=response_error,response_observation_count=counts)
    if not hasattr(engine,'_query_feedback_history'):engine._query_feedback_history=[]
    engine._query_feedback_history.append(row)
    if not hasattr(engine,'_query_feedback_outputs'):engine._query_feedback_outputs={}
    acquisition=int(engine._al4pde_al_iter)
    engine._query_feedback_outputs.setdefault(acquisition,[]).append(dict(**row,
        prediction_log1p=predicted,actual_log1p=actual,time_indices=pending['times'],action_errors=action_errors))


def write_feedback(engine,directory,round_index):
    output=getattr(engine,'_query_feedback_outputs',{}).pop(round_index,[])
    if not output:return
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    fields={key:np.concatenate([r[key] for r in output]) for key in (
        'candidate_ids','input_features','point_error','response_error','response_observation_count',
        'prediction_log1p','actual_log1p')}
    fields['time_indices']=output[0]['time_indices']
    fields['input_mean'],fields['input_std']=engine._query_feature_stats
    # File handle prevents numpy silently appending a second .npz suffix.
    temporary=path/'prequery_feedback.npz.tmp'
    with temporary.open('wb') as handle:np.savez_compressed(handle,**fields)
    temporary.replace(path/'prequery_feedback.npz')
    actions=[a for row in output for a in row['action_errors']]
    temporary=path/'response_feedback.json.tmp'
    temporary.write_text(json.dumps(dict(units='MSE in physical log1p space, up to 32 sampled times',
        anchor_encoding='negative ID = -1 - synthetic training row; never a real mutant ID',
        accepted=int(len(fields['candidate_ids'])),actions=actions),indent=2))
    temporary.replace(path/'response_feedback.json')
