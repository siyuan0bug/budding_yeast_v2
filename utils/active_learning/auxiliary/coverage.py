"""Sparse, frozen-round conditional evidence objective and greedy batches."""
from collections import defaultdict
import heapq
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from .views import contexts

SETTINGS=dict(pool_neighbors=16,train_neighbors=8,epsilon=.05,
              geometry='same module set/environment/event flags; soft rates, IC and mode tokens',
              descriptor='mean of per-member physical descriptors; train-scaled RBF')


def balanced_select(probabilities,labels,budget,seed):
    """Microbatch label-balancing control, without real target ratios.

    Refresh a global shortlist every 128 picks; update counts after every pick.
    A lazy global heap can rescore the entire pool after each pick when class
    probabilities are similar, making million-point acquisition impractical.
    This control deliberately uses a documented approximation instead.
    """
    if not 0<=budget<=len(probabilities):raise ValueError('Balance budget outside pool')
    counts=np.bincount(labels,minlength=6).astype(float)+1.
    rng=np.random.RandomState(seed);order=rng.permutation(len(probabilities))
    values=np.asarray(probabilities,dtype=np.float32)[order];available=np.ones(len(values),dtype=bool)
    selected=[];gains=[];refresh=128;shortlist=1024
    while len(selected)<budget:
        scores=values@np.asarray(1./counts,dtype=np.float32);scores[~available]=-np.inf
        count=min(shortlist,int(available.sum()))
        local=np.argpartition(-scores,count-1)[:count]
        # Stable random-seed order resolves equal scores independently of BLAS.
        local.sort();block=values[local];active=np.ones(count,dtype=bool)
        for _ in range(min(refresh,budget-len(selected),count)):
            scores=block@np.asarray(1./counts,dtype=np.float32);scores[~active]=-np.inf
            j=int(np.argmax(scores));i=int(local[j])
            selected.append(int(order[i]));gains.append(float(scores[j]));counts+=block[j]
            active[j]=False;available[i]=False
    return np.asarray(selected,dtype=np.int64),np.asarray(gains)


def sparse_geometry(view,pool_inputs,train_inputs,pool_p,labels,pool_phi=None,train_phi=None):
    """Npool x (Npool+Ntrain), with separate train-neighbor queries.

    Pool neighbors alone would almost never contain a train point in a million
    pool, incorrectly declaring nearly everything unsupported.
    """
    n=len(pool_inputs);train_p=np.eye(6,dtype=np.float32)[labels]
    phi_mean=phi_scale=None
    if pool_phi is not None:
        phi_mean=train_phi.mean(0);phi_scale=train_phi.std(0);phi_scale[phi_scale<1e-5]=1.
        pool_phi=(pool_phi-phi_mean)/phi_scale;train_phi=(train_phi-phi_mean)/phi_scale
    pool_groups=defaultdict(list);train_groups=defaultdict(list)
    for i,key in enumerate(contexts(view.pool)):pool_groups[key].append(i)
    for i,key in enumerate(contexts(view.train.inputs)):train_groups[key].append(i)
    row_parts=[];col_parts=[];value_parts=[]
    for key,indices in pool_groups.items():
        indices=np.asarray(indices,dtype=np.int64);known=np.asarray(train_groups[key],dtype=np.int64)
        values=pool_inputs[indices];active=np.std(values,axis=0)>1e-6
        if len(known):active|=np.std(np.concatenate((values,train_inputs[known])),axis=0)>1e-6
        if not active.any():active[0]=True
        values=values[:,active]
        for local,k,is_train in ((indices,SETTINGS['pool_neighbors'],False),(known,SETTINGS['train_neighbors'],True)):
            if not len(local):continue
            reference=train_inputs[local][:,active] if is_train else values
            distances,neighbors=cKDTree(reference).query(values,k=min(k,len(local)),workers=1)
            neighbors=np.asarray(neighbors).reshape(len(indices),-1);distances=np.asarray(distances).reshape(neighbors.shape)
            rows=np.repeat(indices,neighbors.shape[1]);columns=local[neighbors.ravel()]
            # Mean squared input distance avoids scaling with constant feature dimensions.
            weights=np.exp(-.5*distances.ravel()**2/max(1,values.shape[1]))
            probabilities=train_p if is_train else pool_p
            weights*=np.einsum('ij,ij->i',pool_p[rows],probabilities[columns])
            if pool_phi is not None:
                reference_phi=train_phi if is_train else pool_phi
                for start in range(0,len(rows),8192):
                    delta=pool_phi[rows[start:start+8192]]-reference_phi[columns[start:start+8192]]
                    weights[start:start+8192]*=np.exp(-.5*np.mean(delta**2,axis=1))
            row_parts.append(rows);col_parts.append(columns+n if is_train else columns);value_parts.append(weights.astype(np.float32))
    matrix=coo_matrix((np.concatenate(value_parts),(np.concatenate(row_parts),np.concatenate(col_parts))),
                      shape=(n,n+len(train_inputs))).tocsc()
    matrix.eliminate_zeros()
    support=SETTINGS['epsilon']+np.asarray(matrix[:,n:].sum(1)).ravel().astype(float)
    return matrix[:,:n],support,dict(edges=int(matrix.nnz),contexts=len(pool_groups),
        input_dimension=int(pool_inputs.shape[1]),descriptor_dimension=0 if pool_phi is None else int(pool_phi.shape[1]),settings=SETTINGS)


def coverage_select(kernel,support,budget,seed):
    """Exact lazy greedy for this frozen sparse log-coverage proxy, unique nodes."""
    if not 0<=budget<=kernel.shape[1]:raise ValueError('Coverage budget outside pool')
    support=np.asarray(support,dtype=np.float64).copy();kernel=kernel.tocsc()
    if (kernel.data<0).any() or not np.isfinite(kernel.data).all() or (support<=0).any():raise ValueError('Invalid evidence kernel')
    def gain(i):
        a,b=kernel.indptr[i:i+2];rows=kernel.indices[a:b];values=kernel.data[a:b]
        return float(np.log1p(values/support[rows]).sum())
    rng=np.random.RandomState(seed);tie=rng.random(kernel.shape[1])
    heap=[(-gain(i),tie[i],i) for i in range(kernel.shape[1])];heapq.heapify(heap)
    selected=[];gains=[]
    while len(selected)<budget:
        _,random,i=heapq.heappop(heap);value=gain(i)
        if heap and value < -heap[0][0]-1e-12:
            heapq.heappush(heap,(-value,random,i));continue
        selected.append(i);gains.append(value)
        a,b=kernel.indptr[i:i+2];support[kernel.indices[a:b]]+=kernel.data[a:b]
    return np.asarray(selected,dtype=np.int64),np.asarray(gains)
