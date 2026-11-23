"""R19: frozen contrast scoring, followed by actual equal-budget cold starts."""
from dataclasses import replace,asdict
from collections import Counter
from pathlib import Path
import gc,json,time
import numpy as np
import torch
from budding_yeast_v2.data.dataset_archive import TrajectoryStore,atomic_json,file_sha256
from budding_yeast_v2.data.systems.budding.specification import SPEC
from budding_yeast_v2.utils.sampling.layout import RESULTS,SHARED,experiment_directory
from budding_yeast_v2.utils.sampling.contracts import PoolInputs,QueriedTraining
from budding_yeast_v2.utils.sampling.ours.mutation_contrasts import exact_contrasts,split_contrasts
from budding_yeast_v2.utils.sampling.ours.conditioning_gradients import fno_gradient_rows,response_feedback_gradient
from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig,child_seed
from budding_yeast_v2.utils.sampling.benchmark.diagnostics import context_key,evaluate_reference_cohort
from budding_yeast_v2.utils.sampling.benchmark.rounds import fit_round
from budding_yeast_v2.utils.sampling.benchmark.runner import source_inventory,experiment_lock
from budding_yeast_v2.utils.evaluation.checkpoints import Predictor,load_sampling_checkpoint
from budding_yeast_v2.utils.training.losses import RelativeL2Loss
from budding_yeast_v2.utils.experiments.resources import resource_slot
from budding_yeast_v2.utils.experiments.logging import tee_experiment_log

WORK=SHARED/'seed_42/studies/contrast_coldstart_v1'
METHODS=('condition','head','random','error')

def config():
    protocol=json.loads((SHARED/'seed_42/protocol.json').read_text())
    return replace(BenchmarkConfig(**protocol['config']),budgets=(2256,),epochs=(100,),warmup=(5,)).validate()

def directory(method):
    return experiment_directory(RESULTS,config(),'mutation','contrast_coldstart_'+method+'_v1')

def read(source,rows,fields):
    with resource_slot('data','R19 bounded source read'):return source.read(rows,fields)

def protected_contexts():
    protected=set()
    paths=[RESULTS/'summary/conditioning_gradient_probe_v1/protocol.json',RESULTS/'summary/mechanism_transfer_screen_v1/protocol.json',SHARED/'seed_42/studies/intervention_order_calibration_v1/study.json',Path('/home/users/zsy/mydata/budding_yeast/resource_profiles/sampling_utility_probe_20261122/protocol.json')]
    for path in paths:
        v=json.loads(path.read_text());roles=v.get('roles',v)
        for key in ['feedback_contexts','audit_contexts']:protected.update(roles.get(key,[]))
        for role in ['candidate','feedback','check']:
            if role in roles and roles[role] and isinstance(roles[role][0],dict):
                for edge in roles[role]:protected.update([edge['left_context'],edge['right_context']])
    extra=directory('condition').parents[2]/'ours_response_omission_probe/diagnostic_2k_plus256_steps400/seed_42/protocol.json'
    if extra.exists():
        v=json.loads(extra.read_text());protected.update(v.get('probe_contexts',[]));protected.update(v.get('protected_contexts',[]))
    return protected

def prepare(device):
    WORK.mkdir(parents=True,exist_ok=False);started=time.monotonic();sources=source_inventory()
    shared=SHARED/'seed_42';protocol=json.loads((shared/'protocol.json').read_text())
    for name,digest in protocol['artifacts'].items():
        if file_sha256(shared/name)!=digest:raise ValueError('Shared artifact changed: '+name)
    source=TrajectoryStore(protocol['pools']['mutation']['path'])
    if source.digest!=protocol['pools']['mutation']['sha256']:raise ValueError('Source changed')
    with np.load(shared/'mutation_split.npz') as s:initial=s['initial'].copy();candidate=s['candidates'].copy();validation=s['validation'].copy()
    inp=read(source,candidate,['params','y_init','candidate_metadata_json']);meta=tuple(json.loads(str(v)) for v in inp['candidate_metadata_json'])
    keys=[context_key(m,p[SPEC.param_names.index('mdt')]) for m,p in zip(meta,inp['params'])];lookup=dict(zip(map(int,candidate),keys));old=protected_contexts();known={lookup[int(i)] for i in initial}
    allowed=np.array([i for i,k in enumerate(keys) if k not in old and k not in known]);rng=np.random.default_rng(child_seed(42,'R19_new_input_only_roles'))
    by={}
    for i in allowed:by.setdefault(keys[i],[]).append(int(i))
    contexts=sorted(by);rng.shuffle(contexts);probe_positions=[];reserved=set()
    for key in contexts:
        take=min(2,len(by[key]),512-len(probe_positions));probe_positions.extend(map(int,rng.choice(by[key],take,replace=False)));reserved.add(key)
        if len(probe_positions)==512:break
    if len(probe_positions)!=512:raise ValueError('Insufficient new probe backgrounds')
    positions=np.array([i for i in allowed if keys[i] not in reserved]);view=PoolInputs(candidate[positions],inp['params'][positions],inp['y_init'][positions],tuple(meta[i] for i in positions))
    edges=exact_contrasts(view,per_gene=256,seed=child_seed(42,'R19_contrasts'));roles=split_contrasts(edges,per_gene_role=32,seed=child_seed(42,'R19_roles'))
    feedback=[];counts=Counter()
    for e in roles['feedback']:
        if counts[e['gene']]<8:feedback.append(e);counts[e['gene']]+=1
    roles['feedback']=feedback;probe_rows=candidate[probe_positions].tolist();response_reserved={e[k] for e in roles['feedback']+roles['check'] for k in ['left_context','right_context']}
    val=read(source,validation,['params','candidate_metadata_json']);val_keys=[context_key(json.loads(str(m)),p[SPEC.param_names.index('mdt')]) for m,p in zip(val['candidate_metadata_json'],val['params'])]
    val_rows=[int(i) for i,k in zip(validation,val_keys) if k not in reserved|response_reserved|old]
    if len(val_rows)<4000:raise ValueError('Too few shared validation rows')
    prior=json.loads((RESULTS/'summary/conditioning_gradient_probe_v1/protocol.json').read_text());checkpoint=Path(prior['checkpoint'])
    original=json.loads((RESULTS/'summary/mechanism_transfer_screen_v1/protocol.json').read_text())
    if initial.tolist()!=original['initial_rows']:raise ValueError('Initial training IDs changed')
    if file_sha256(checkpoint)!=prior['checkpoint_sha256']:raise ValueError('Scoring checkpoint changed')
    settings=dict(status='roles_declared_before_scoring',config=asdict(config()),source=str(source.path),source_sha256=source.digest,
      initial=initial.tolist(),probe=probe_rows,validation=val_rows,roles=roles,sources=sources,checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
      primary='new synthetic probe physical sample/channel Relative L2; require >=5% gain over matched pair random and MAE/MSE not worse by >5%',
      model_seed=child_seed(42,'R19_common_cold_model'),new_ODE_calls=0,real_test_prior_audit_read=False,
      interpretation='cold 2256/100epoch actual training, not full five rounds; scoring response MSE unchanged from R18; final normalization independently refit per selected training set',
      probe_status='input-only reservation; labels inaccessible until all selections locked')
    atomic_json(WORK/'protocol.json',settings);print('PAIRS',{k:len(v) for k,v in roles.items()},'validation',len(val_rows),flush=True)
    rows=sorted({e[k] for role in ['candidate','feedback'] for e in roles[role] for k in ['left','right']});a=read(source,rows,['params','y_init','data']);loc=dict(zip(rows,range(len(rows))))
    model,norm,_=load_sampling_checkpoint(checkpoint,device);model.eval();x,p=norm.encode(a['params'],a['y_init'],500);y=norm.targets(a['data']);x=x.to(device);p=p.to(device);y=y.to(device)
    pairs=[(loc[e['left']],loc[e['right']]) for e in roles['feedback']];reference=response_feedback_gradient(model,x,p,y,pairs);criterion=RelativeL2Loss(eps=.01)
    loss=[];condition=[];head=[]
    for start in range(0,len(rows),32):
        stop=min(start+32,len(rows));l,g,h=fno_gradient_rows(model,x[start:stop],p[start:stop],y[start:stop],criterion)
        loss.append(l.cpu().numpy());condition.append(g.cpu().numpy());head.append(h.cpu().numpy())
        if start%320==0:print('GRADIENT_ROWS',stop,'/',len(rows),flush=True)
    values={'condition':np.concatenate(condition),'head':np.concatenate(head)};loss=np.concatenate(loss);scores={k:[] for k in ['condition','head','error']}
    for e in roles['candidate']:
        left,right=loc[e['left']],loc[e['right']];scores['error'].append(float((loss[left]+loss[right])/2))
        for index,block in enumerate(['condition','head']):
            g=(values[block][left]+values[block][right])/2;ref=reference[index].detach().cpu().numpy();denom=np.linalg.norm(g)*np.linalg.norm(ref)
            scores[block].append(float(g@ref/denom) if denom>0 else None)
    def pick(order):
        chosen=[];used=set()
        for i in order:
            e=roles['candidate'][int(i)]
            if {e['left'],e['right']}&used:continue
            chosen.append(int(i));used.update([e['left'],e['right']])
            if len(chosen)==128:return chosen
        raise ValueError(f'Insufficient independent paired endpoints: {len(chosen)}/128')
    eligible=[i for i in range(len(roles['candidate'])) if scores['condition'][i] is not None and scores['head'][i] is not None]
    orders={k:sorted(eligible,key=lambda i:-scores[k][i]) for k in scores};orders['random']=rng.permutation(eligible)
    settings['zero_gradient_unscorable_edges']=len(roles['candidate'])-len(eligible)
    selections={}
    for method in METHODS:
        chosen=pick(orders[method]);added=[roles['candidate'][i][side] for i in chosen for side in ['left','right']];selected=initial.tolist()+added
        if len(set(selected))!=2256 or set(added)&set(probe_rows):raise ValueError('Budget/role violation')
        target=WORK/f'{method}_dataset.json';source.snapshot(target,selected,provenance=dict(study='R19',method=method))
        selections[method]=dict(path=str(target),sha256=file_sha256(target),rows=selected,chosen_edges=chosen)
    atomic_json(WORK/'selection.json',dict(locked_before_probe_labels=True,selections=selections,scores=scores));settings['selection_sha256']=file_sha256(WORK/'selection.json');settings['status']='prepared';settings['unique_selection_labels_read']=len(rows);settings['preparation_seconds']=time.monotonic()-started
    if sources!=source_inventory():raise ValueError('Sources changed')
    atomic_json(WORK/'protocol.json',settings);print('PREPARED',WORK,flush=True)

def train(method,device):
    settings=json.loads((WORK/'protocol.json').read_text());selection=json.loads((WORK/'selection.json').read_text());desc=selection['selections'][method]
    if settings['sources']!=source_inventory() or file_sha256(WORK/'selection.json')!=settings['selection_sha256'] or file_sha256(desc['path'])!=desc['sha256']:raise ValueError('Identity changed')
    source=TrajectoryStore(settings['source']);path=directory(method);cfg=config();started=time.monotonic()
    if source.digest!=settings['source_sha256']:raise ValueError('Source changed')
    with experiment_lock(path),tee_experiment_log(path/'train.log'):
        atomic_json(path/'config.json',dict(workflow='R19_coldstart_pilot',method=method,training=asdict(cfg),model_seed=settings['model_seed'],sources=settings['sources']))
        a=read(source,desc['rows'],['params','y_init','data']);v=read(source,settings['validation'],['params','y_init','data'])
        def inputs(data,ids):return PoolInputs(np.asarray(ids),data['params'],data['y_init'])
        import wandb
        wb=wandb.init(project='budding_sampling_benchmark',group='contrast_coldstart_v1',name=method+'/seed42',dir=str(path),mode='offline')
        try:
            model,norm,history=fit_round(cfg,0,QueriedTraining(inputs(a,desc['rows']),a['data']),[(inputs(v,settings['validation']),v['data'])],path/'training',settings['model_seed'],device=device,dataset_path=desc['path'],state_dir=path/'_state/primary',identity=dict(study='R19',selection_sha256=desc['sha256']),wandb_run=wb)
            common_a=read(source,settings['initial'],['params','data'])
            from budding_yeast_v2.data.normalization import NormalizationStats
            common=NormalizationStats.fit(common_a['params'],common_a['data']);del a,v,common_a;gc.collect()
            check=read(source,settings['probe'],['params','y_init','data'])
            metric=evaluate_reference_cohort(Predictor([model],norm,500,device,128),inputs(check,settings['probe']),check['data'],common,path/'eval/synthetic_probe',variable_names=SPEC.variable_names,sample_ids=settings['probe'],return_per_sample=True)
            if settings['sources']!=source_inventory() or history[-1]['optimizer_steps']!=1800:raise ValueError('Source/update mismatch')
            atomic_json(path/'completion.json',dict(status='complete',method=method,metrics=metric,optimizer_steps=1800,epochs=100,training_inputs=2256,seconds=time.monotonic()-started,real_test_prior_audit_read=False))
        finally:wb.finish()

def summarize():
    output=RESULTS/'summary/contrast_coldstart_v1';output.mkdir(parents=True,exist_ok=False);results={m:json.loads((directory(m)/'completion.json').read_text()) for m in METHODS}
    baseline=results['random']['metrics']['physical']['all_variables'];condition=results['condition']['metrics']['physical']['all_variables']
    gain=1-condition['Relative L2']/baseline['Relative L2'];passed=gain>=.05 and condition['MAE']<=1.05*baseline['MAE'] and condition['MSE']<=1.05*baseline['MSE']
    atomic_json(output/'report.json',dict(status='complete',results=results,condition_vs_random_relative_gain=gain,screen_passed=passed,real_test_prior_audit_read=False))
    lines=['# R19：实际冷启动训练对照','', '四组各2256条、100epoch、1800更新，相同初始化/正式单轮超参数，各自按训练集重拟合统计。来源合成验证归一化loss选best。非正式五轮榜单，不评real。','', '|方法|物理Relative↓|物理MAE↓|物理MSE↓|Corr↑|','|---|---:|---:|---:|---:|']
    for method,result in results.items():
        p=result['metrics']['physical']['all_variables'];lines.append('|'+method+'|'+'|'.join(f'{p[k]:.6f}' for k in ['Relative L2','MAE','MSE','Correlation'])+'|')
    lines+=['',f'条件评分对随机Relative改善{gain:.3%}，预定门槛通过={passed}。', '只使用预留的新训练侧合成探针；旧检查/保留审计/正式test/real不用于选择、调参或模型选择。单seed试训不证明稳定胜过Power/LCMD。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
