"""Exact per-example FNO input/head gradients, without per-example backprop loops."""
import torch
from budding_yeast_v2.models.baselines.common import joint_condition_grid


def fno_gradient_rows(model, x, p, target, criterion):
    backbone=getattr(model,'model',model)
    if not hasattr(backbone,'lift') or not hasattr(backbone,'project'):
        raise ValueError('This diagnostic explicitly requires the standard FNO')
    cache={}
    def lift_hook(_,args,output):cache['lift']=output
    def head_hook(_,args):cache['latent']=args[0]
    handles=[backbone.lift.register_forward_hook(lift_hook),backbone.project[-1].register_forward_pre_hook(head_hook)]
    try:
        pred=model.forward_ic_time(x,p)
        losses=criterion.per_sample(pred,target)
        if not torch.isfinite(pred).all() or not torch.isfinite(losses).all():raise FloatingPointError('Nonfinite candidate prediction/loss')
        delta,output_delta=torch.autograd.grad(losses.sum(),(cache['lift'],pred))
        cond=joint_condition_grid(x,p,backbone.num_vars,backbone.param_dim)[:,:-1]
        input_grad=torch.einsum('bot,bit->boi',delta,cond).flatten(1)
        head_grad=torch.cat((torch.einsum('bot,bit->boi',output_delta,cache['latent']).flatten(1),output_delta.sum(-1)),1)
        if not torch.isfinite(input_grad).all() or not torch.isfinite(head_grad).all():raise FloatingPointError('Nonfinite example gradient')
        return losses.detach(),input_grad.detach(),head_grad.detach()
    finally:
        for handle in handles:handle.remove()


def response_feedback_gradient(model, x, p, target, pairs, *, batch_pairs=4):
    """Exact mean full-trajectory response-MSE gradient on declared feedback pairs."""
    backbone=getattr(model,'model',model)
    params=[backbone.lift.weight,backbone.project[-1].weight,backbone.project[-1].bias]
    total=[torch.zeros_like(v) for v in params]
    for start in range(0,len(pairs),batch_pairs):
        batch=pairs[start:start+batch_pairs]
        ids=[i for pair in batch for i in pair]
        pred=model.forward_ic_time(x[ids],p[ids]);y=target[ids]
        loss=torch.nn.functional.mse_loss(pred[1::2]-pred[::2],y[1::2]-y[::2])
        gradients=torch.autograd.grad(loss,params)
        for result,value in zip(total,gradients):result.add_(value.detach(),alpha=len(batch)/len(pairs))
    condition=total[0][:,:-1,:].flatten()
    head=torch.cat([total[1].flatten(),total[2].flatten()])
    if not torch.isfinite(condition).all() or not torch.isfinite(head).all():raise FloatingPointError('Nonfinite feedback gradient')
    return condition,head
