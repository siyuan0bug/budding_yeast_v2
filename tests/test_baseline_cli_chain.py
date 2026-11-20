"""Actual CLI parsers -> production constructors -> one-step fit -> checkpoint evaluation."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest
import torch
import pytorch_lightning as pl

from budding_yeast_v2.models.baselines import BASELINE_REGISTRY
from budding_yeast_v2.models.yeast_lit_module import MODEL_REGISTRY,YeastLitModule
from budding_yeast_v2.utils.sampling.registry import CLI_STRATEGIES
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.data.sensitivity import attach_sensitivities,preprocessing_fingerprint
from budding_yeast_v2.eval import load_model_from_checkpoint
from budding_yeast_v2 import tune

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module',autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def source_main(name):
    return next(n for n in ast.parse((ROOT/name).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')


def train_parser():
    statements=[]
    for node in source_main('train.py').body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in node.targets):break
        statements.append(node)
    ns=dict(argparse=argparse,torch=torch,MODEL_REGISTRY=MODEL_REGISTRY,CLI_STRATEGIES=CLI_STRATEGIES)
    exec(compile(ast.Module(body=statements,type_ignores=[]),'actual_train_parser','exec'),ns)
    return ns['parser']


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    path=tmp_path_factory.mktemp('baseline_cli')/'artificial_210min.npz'
    rng=np.random.RandomState(17)
    # Artificial pipeline data, not an ODE reference or a performance experiment.
    np.savez(path,data=rng.uniform(.1,1.,(62,39,32)).astype(np.float32),
             params=rng.uniform(.1,2.,(62,141)),param_names=np.array([f'p{i}' for i in range(141)]),
             mutant_names=np.array(['1_artificial','2_artificial']+[f'RANDOM_{i}' for i in range(60)]),
             pattern_labels=np.array(['Pattern_B: Single-Division (==1)']*62))
    return path


@pytest.mark.parametrize('name',sorted(BASELINE_REGISTRY))
def test_cli_to_real_training_checkpoint_and_evaluation(name,dataset,tmp_path):
    loss='sc_sensitivity' if name=='sc_fno' else 'mse_only'
    argv=['--model',name,'--dataset_name',str(dataset),'--device','cpu','--width','8',
          '--n_blocks','2','--modes','4','--wavelet','db2','--wavelet_level','2',
          '--ode_method','rk4','--ode_rtol','0.002','--ode_atol','0.0002',
          '--sensitivity_weight','0.7','--loss_type',loss,'--max_epochs','1',
          '--warmup_epochs','0','--num_workers','0','--batch_size','2','--no_wandb']
    args=train_parser().parse_args(argv)
    dm=YeastDataModule(dataset_path=str(dataset),num_workers=0,use_adj_matrix=False,
                      batch_size_synth=2,val_batch_size=2,norm_type=args.norm_type)
    dm.setup('fit')
    if name=='sc_fno':
        sidecar=tmp_path/'sens.npz'
        np.savez(sidecar,dy_dp=np.zeros((*dm._raw_data.shape,1),dtype=np.float32),
                 parameter_indices=np.array([0]),initial_conditions_held_fixed=True,
                 data_sha256=dm._dataset_contract['data_sha256'],params_sha256=dm._dataset_contract['params_sha256'],
                 preprocessing_sha256=preprocessing_fingerprint(dm))
        attach_sensitivities(dm,sidecar)
    # Execute the actual nested production factory, not a test-side reconstruction.
    factory=next(n for n in ast.walk(source_main('train.py')) if isinstance(n,ast.FunctionDef) and n.name=='build_fresh_model')
    ns=dict(args=args,datamodule=dm,YeastLitModule=YeastLitModule)
    exec(compile(ast.Module(body=[factory],type_ignores=[]),'actual_train_factory','exec'),ns)
    model=ns['build_fresh_model']()
    assert model.hparams.n_blocks==2 and model.hparams.seq_len==32
    assert model.hparams.wavelet_level==2 and model.hparams.sensitivity_weight==.7
    before={k:v.detach().clone() for k,v in model.model.named_parameters()}
    trainer=pl.Trainer(accelerator='cpu',devices=1,fast_dev_run=True,precision='32-true',logger=False,
        enable_checkpointing=False,enable_model_summary=False,enable_progress_bar=False)
    trainer.fit(model,datamodule=dm)
    assert any(not torch.equal(before[k],v) for k,v in model.model.named_parameters())
    path=tmp_path/f'{name}.ckpt';trainer.save_checkpoint(path)
    restored,hp=load_model_from_checkpoint(path,device='cpu')
    assert hp['n_blocks']==2 and hp['seq_len']==32 and hp['wavelet_level']==2
    assert hp['ode_method']=='rk4' and hp['ode_rtol']==.002 and hp['ode_atol']==.0002
    model.eval();x,p,_=next(iter(dm.val_dataloader()[0]))
    with torch.no_grad():torch.testing.assert_close(model(x,p),restored(x,p))
    assert 'constructor_contract_version' not in hp


@pytest.mark.parametrize('name',sorted(BASELINE_REGISTRY))
def test_hpo_export_matches_actual_shell_columns_and_train_parser(name):
    hp=dict(width=8,n_blocks=2,modes=4,batch_size=2,max_epochs=1,lr=.00123456789,
            weight_decay=.00037,warmup_epochs=0,wavelet='db2',level=2,
            ode_method='rk4',ode_rtol=.002,ode_atol=.0002,
            sensitivity_path='/tmp/sensitivity.npz',sensitivity_weight=.7)
    loss='sc_sensitivity' if name=='sc_fno' else 'mse_only'
    line=tune.format_pipeline_experiment_line(name,loss,'none','/tmp/artificial.npz','artificial',hp,gpu_id=3)
    # Use Bash read exactly as the production row reader does; never source
    # run_pipeline.sh, which would start the user's real experiments.
    script='''read -r MODEL LOSS USE_ADJ DATASET MODES WIDTH BS EP AL LR WARMUP ADD PERT THRESH ROLLOUT GPU EXTRA <<< "$1"
read -r -a ARGS <<< "$EXTRA"
printf '%s\\n' "$MODEL" "$DATASET" "$MODES" "$WIDTH" "$BS" "$EP" "$LR" "$WARMUP" "$GPU" "${ARGS[@]}"
'''
    fields=subprocess.check_output(['bash','-c',script,'bash',line[1:-1]],text=True).splitlines()
    assert fields[:2]==[name,'/tmp/artificial.npz'] and fields[8]=='3'
    args=train_parser().parse_args(['--model',fields[0],'--dataset_name',fields[1],
        '--modes',fields[2],'--width',fields[3],'--batch_size',fields[4],'--max_epochs',fields[5],
        '--lr',fields[6],'--warmup_epochs',fields[7],*fields[9:]])
    assert args.n_blocks==2 and args.weight_decay==.00037 and args.lr==hp['lr']
    if name=='wno':assert (args.wavelet,args.wavelet_level)==('db2',2)
    if name in ('neural_ode','konode'):assert (args.ode_method,args.ode_rtol,args.ode_atol)==('rk4',.002,.0002)
    if name=='sc_fno':assert args.sensitivity_path==hp['sensitivity_path'] and args.sensitivity_weight==.7


def test_depth_flags_change_deeponet_and_neural_ode():
    for name in ('deeponet','neural_ode','unet'):
        counts=[]
        for depth in (2,3):
            model=YeastLitModule(model_name=name,width=8,n_blocks=depth,loss_type='mse_only')
            counts.append(sum(p.numel() for p in model.parameters()))
        assert counts[0] != counts[1]


def test_removed_cli_names_and_dead_constructor_argument_are_rejected():
    parser=train_parser()
    for old in ('pure_fno','fno1d','fno_ref','wno_ref','mwt_ref','resnet_film','unet1d'):
        with pytest.raises(SystemExit):parser.parse_args(['--model',old,'--dataset_name','unused.npz'])
    with pytest.raises(SystemExit):parser.parse_args(['--model','unet','--dataset_name','unused.npz','--constructor_contract_version','1'])
    assert 'constructor_contract_version' not in __import__('inspect').signature(YeastLitModule).parameters


@pytest.mark.parametrize('name',sorted(BASELINE_REGISTRY))
def test_hpo_constructor_receives_effective_arguments(name):
    args=SimpleNamespace(model=name,adj_matrix_path=None,graph_view='hybrid',graph_load_policy='current',
        loss_type='sc_sensitivity' if name=='sc_fno' else 'mse_only',sensitivity_weight=.7,
        d_lambda=32,ode_method='rk4',ode_rtol=.002,ode_atol=.0002,norm_type='log1p')
    hp=dict(width=8,n_blocks=2,modes=4,wavelet='db2',level=2,lr=.001,weight_decay=.00037,max_epochs=1,warmup_epochs=0)
    dm=SimpleNamespace(num_vars=39,num_params=141,system_id='budding',T_STEPS=32)
    tree=ast.parse((ROOT/'tune.py').read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='YeastLitModule')
    model=eval(compile(ast.Expression(body=call),'actual_hpo_constructor','eval'),dict(YeastLitModule=YeastLitModule,args=args,hp=hp,datamodule=dm))
    assert model.hparams.seq_len==32 and model.hparams.n_blocks==2 and model.hparams.wavelet_level==2
    if name=='wno':assert model.model.wavelet[0].wavelet=='db2' and model.model.wavelet[0].level==2
    if name in ('neural_ode','konode'):assert model.model.method=='rk4' and model.model.rtol==.002


def test_hpo_does_not_search_ignored_model_knobs():
    class Trial:
        def __init__(self):self.searched=set()
        def suggest_float(self,name,lo,hi,**kwargs):self.searched.add(name);return lo
        def suggest_int(self,name,lo,hi,**kwargs):self.searched.add(name);return lo
        def suggest_categorical(self,name,choices):self.searched.add(name);return choices[0]
    for name in BASELINE_REGISTRY:
        trial=Trial();hp=tune.sample_model_aware_hparams(trial,name)
        assert 'n_blocks' in trial.searched
        assert ('modes' in trial.searched)==(name in ('fno','sc_fno','bregman_fno','mwt'))
        assert ('level' in trial.searched)==(name=='wno')


@pytest.mark.parametrize('length',[31,500])
def test_plain_unet_odd_lengths_and_bfloat16(length):
    # Fix initialization: tiny ReLU networks can otherwise randomly be inactive.
    torch.manual_seed(73)
    model=YeastLitModule(model_name='unet',width=4,n_blocks=2,num_vars=3,param_dim=5,loss_type='mse_only')
    initial=torch.randn(2,3,requires_grad=True)
    time=torch.linspace(0,1,length).expand(2,3,-1)
    x=torch.stack((initial[...,None].expand_as(time),time),2);p=torch.randn(2,5,requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=model(x,p);loss=out.float().square().mean()
    loss.backward()
    assert out.shape==(2,3,length) and torch.isfinite(out).all()
    assert initial.grad.abs().sum()>0 and p.grad.abs().sum()>0


def test_downstream_tools_use_checkpoint_contract_without_model_fallback(dataset,tmp_path):
    from budding_yeast_v2.utils.evaluation.checkpoints import predict_real_checkpoints
    dm=YeastDataModule(dataset_path=str(dataset),num_workers=0,use_adj_matrix=False)
    dm.setup('fit')
    model=YeastLitModule(model_name='unet',width=4,n_blocks=2,seq_len=32,loss_type='mse_only')
    model.bind_datamodule_stats(dm)
    checkpoint={'hyper_parameters':dict(model.hparams),'state_dict':model.state_dict(),
                'preprocessing_contract':dm.preprocessing_state()}
    path=tmp_path/'unet.ckpt';torch.save(checkpoint,path)
    predictions,target,names=predict_real_checkpoints([f'unet={path}'],str(dataset),batch_size=2)
    assert len(names)==2 and predictions['unet'].shape==target.shape==(2,39,32)
    with pytest.raises(ValueError,match='LABEL=PATH'):
        predict_real_checkpoints([str(path)],str(dataset))
    changed=dict(checkpoint);changed['preprocessing_contract']=dict(checkpoint['preprocessing_contract'])
    changed['preprocessing_contract']['std_y']=checkpoint['preprocessing_contract']['std_y']*2
    second=tmp_path/'different.ckpt';torch.save(changed,second)
    with pytest.raises(ValueError,match='normalization differs'):
        predict_real_checkpoints([f'first={path}',f'second={second}'],str(dataset),batch_size=2)
