"""Cohen-Addad2026 Algorithm2, explicit gradient-energy regression instantiation.

The appendix uses gradient norms as a loss proxy. Here ell(v)=||v||² is defined
on *gradient embeddings*, including every basis direction, never on biological
inputs. The surrogate is still trained with its original Relative L2 objective.
"""
import numpy as np
from sklearn.utils.extmath import randomized_svd

from ..contracts import SamplingResult
from .regression import gradient_bank


def sensitivity_scores(embedding,rank=32,gamma=5.,lam=1.,xi=1e-6,seed=42):
    x=np.asarray(embedding,dtype=np.float64)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid low-rank embedding')
    if rank<1 or gamma<0 or lam<=0 or xi<0:raise ValueError('Invalid low-rank constants')
    rank=min(rank,*x.shape)
    _,_,basis=randomized_svd(x,n_components=rank,random_state=seed)
    coefficients=x@basis.T
    basis_loss=np.sum(basis*basis,axis=1)  # Valid ell(v), no fabricated ODE labels.
    residual=np.maximum(np.sum(x*x,1)-np.sum(coefficients*coefficients,1),0.)
    scores=(gamma+1)*(coefficients*coefficients@basis_loss)+gamma*rank*xi+lam*residual
    if not np.isfinite(scores).all() or (scores<0).any():raise FloatingPointError('Invalid low-rank sensitivity')
    return scores,dict(rank=rank,basis_loss=basis_loss.tolist(),gamma=gamma,lambda_=lam,xi=xi,
                      residual_energy=float(residual.sum()),basis_queries_to_ode=0)


def select(context,budget,name='low_rank'):
    eps=float(getattr(context.predictor.members[0].criterion,'eps',.01))
    gradients=gradient_bank(context.predictor,context.pool,context.label_reader,
                            dimension=context.sketch_dim,eps=eps)
    scores,diagnostics=sensitivity_scores(gradients,seed=context.seed)
    probability=scores/scores.sum() if scores.sum()>0 else np.full(len(scores),1/len(scores))
    draws=np.random.default_rng(context.seed).choice(len(scores),budget,replace=True,p=probability)
    indices,multiplicity=np.unique(draws,return_counts=True)
    weights=len(indices)*multiplicity/(budget*len(scores)*probability[indices])
    diagnostics.update(source='Cohen-Addad2026 Algorithm2',representation='sketched FNO output-head Relative-L2 gradients',
        selection_loss='gradient energy ell(v)=squared norm; not surrogate training loss',
        task_adaptation=True,draw_slots=int(budget),draw_ids=context.pool.ids[draws].tolist(),
        unique_draw_inputs=len(indices),multiplicity=multiplicity.tolist(),replacement=True,
        weights='native inverse-probability weights for universe mean; duplicate aggregation',
        no_original_relative_loss_preservation_claim=True)
    return SamplingResult(context.pool.ids[indices],weights,probability[indices],diagnostics)
