"""Measure isolated synthetic training-step CUDA peaks; never run AL or W&B."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CASES = [('budding','fno',192,4),('budding','deeponet',256,3),
         ('budding','unet',32,4),('budding','transformer',192,4),
         ('budding','biom_refiner_plus',192,4),('budding','biom_refiner',192,4),
         ('budding','hyper_fno',192,4),('fission','fno',128,4),
         ('fission','biom_refiner_plus',128,4),('fission','graph_resnet_film',128,4)]


def measure(args):
    import torch
    from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
    torch.set_num_threads(4)
    torch.manual_seed(42)
    v,p = (39,141) if args.system=='budding' else (16,58)
    result=dict(system=args.system,model=args.model,width=args.width,n_blocks=args.blocks,
                modes=24,batch=args.batch,steps=args.steps,time_steps=500,precision='bf16-mixed',
                loss='hybrid_mse_rel',input='synthetic_normalized',
                coarse_auxiliary_loss=args.model.startswith(('biom_refiner','bio_refiner')),
                device=torch.cuda.get_device_name(0))
    try:
        torch.cuda.reset_peak_memory_stats()
        module=YeastLitModule(model_name=args.model,num_vars=v,param_dim=p,system_id=args.system,
                             width=args.width,modes=24,n_blocks=args.blocks,loss_type='hybrid_mse_rel',
                             norm_type='log1p',max_epochs=50,lr=1e-3,weight_decay=1e-4).cuda().train()
        module.log=lambda *a,**kw:None
        optimizer=torch.optim.AdamW(module.parameters(),lr=1e-3,weight_decay=1e-4)
        result['real_scalar_parameters']=sum(x.numel()*(2 if x.is_complex() else 1) for x in module.parameters())
        x=torch.empty(args.batch,v,2,500,device='cuda')
        x[:,:,0,:]=torch.randn(args.batch,v,1,device='cuda')
        x[:,:,1,:]=torch.linspace(0,1,500,device='cuda')
        parameters=torch.randn(args.batch,p,device='cuda')
        target=torch.randn(args.batch,v,500,device='cuda')
        result['peaks_by_stage_gib']={}
        start=time.monotonic()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=module.training_step((x,parameters,target),step)
            torch.cuda.synchronize()
            result['peaks_by_stage_gib'][f'{step}_forward']=torch.cuda.max_memory_allocated()/2**30
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite synthetic training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(),1.)
            torch.cuda.synchronize()
            result['peaks_by_stage_gib'][f'{step}_backward']=torch.cuda.max_memory_allocated()/2**30
            optimizer.step()
            torch.cuda.synchronize()
            result['peaks_by_stage_gib'][f'{step}_optimizer']=torch.cuda.max_memory_allocated()/2**30
            del loss
        result.update(status='ok',seconds=time.monotonic()-start)
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status='oom',error=str(exc))
    except Exception as exc:
        result.update(status='error',error=repr(exc))
    result.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    Path(args.output).write_text(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',default='5')
    parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--output',required=True)
    parser.add_argument('--timeout',type=int,default=240)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--system',choices=('budding','fission'))
    parser.add_argument('--model')
    parser.add_argument('--width',type=int)
    parser.add_argument('--blocks',type=int)
    args=parser.parse_args()
    if args.batch<1 or args.steps<2:parser.error('positive batch and at least two steps required')
    if args.worker:
        measure(args);return
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    root=str(Path(__file__).resolve().parents[3])
    env['PYTHONPATH']=root+os.pathsep+env.get('PYTHONPATH','')
    results=[]
    cases=[case for case in CASES if (args.system is None or case[0]==args.system)
           and (args.model is None or case[1]==args.model)]
    if not cases:parser.error('No profiling cases match --system/--model')
    for system,model,width,blocks in cases:
        width=args.width if args.width is not None else width
        blocks=args.blocks if args.blocks is not None else blocks
        file=output/f'{system}_{model}.json'
        file.unlink(missing_ok=True)  # A crashed rerun must not reuse stale measurements.
        cmd=[sys.executable,__file__,'--worker','--system',system,'--model',model,'--width',str(width),
             '--blocks',str(blocks),'--batch',str(args.batch),'--steps',str(args.steps),'--output',str(file)]
        with (output/f'{system}_{model}.log').open('w') as log:
            try:
                done=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
                result=json.loads(file.read_text()) if file.exists() else dict(system=system,model=model,status='crash',exit_code=done.returncode)
            except subprocess.TimeoutExpired:
                result=dict(system=system,model=model,status='timeout')
        results.append(result)
        (output/'summary.json').write_text(json.dumps(dict(physical_gpu=args.gpu,batch=args.batch,steps=args.steps,
            note='Synthetic normalized inputs; two AdamW updates, actual training_step and coarse loss. Excludes AL acquisition, validation, ensemble and full-run variation.',results=results),indent=2))
        print(json.dumps({k:result.get(k) for k in ('system','model','status','peak_allocated_gib','peak_reserved_gib','seconds')}),flush=True)

if __name__=='__main__':main()
