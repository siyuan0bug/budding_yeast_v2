"""Input-only exact native-intervention contrasts for research diagnostics."""
from collections import defaultdict
import hashlib,json
import numpy as np
from budding_yeast_v2.data.systems.budding.catalog import TOUCH
from budding_yeast_v2.data.systems.budding.specification import SPEC
from budding_yeast_v2.utils.sampling.benchmark.diagnostics import context_key


def exact_contrasts(inputs, *, per_gene=48, seed=42):
    if not inputs.metadata:raise ValueError('Native mutation metadata required')
    vector=np.c_[inputs.params,inputs.initial].astype('<f8')
    genes=sorted({g for g,_ in TOUCH if g!='MDT'})
    contexts=[context_key(m,p[SPEC.param_names.index('mdt')]) for m,p in zip(inputs.metadata,inputs.params)]
    tokens=[sorted(map(tuple,m.get('pairs',m.get('genes',())))) for m in inputs.metadata]
    rng=np.random.default_rng(seed);edges=[]
    for gene in genes:
        footprint=set().union(*(v for (g,_),v in TOUCH.items() if g==gene))
        editable={j for j,k in enumerate(SPEC.param_names) if k in footprint}
        init_keys={k[5:].casefold() for k in footprint if k.startswith('init_')}
        editable|={len(SPEC.param_names)+j for j,k in enumerate(SPEC.variable_names) if k.casefold() in init_keys}
        keep=[j for j in range(vector.shape[1]) if j not in editable]
        groups=defaultdict(list)
        for i in range(len(inputs.ids)):
            other=tuple(t for t in tokens[i] if t[0]!=gene)
            key=(vector[i,keep].tobytes(),other)
            groups[key].append(i)
        order=list(groups);rng.shuffle(order);count=0
        for key in order:
            group=groups[key]
            if len(group)<2:continue
            # At most four pairs per exact background, bounded rather than n^2.
            group=rng.permutation(group);used=0
            for left in group:
                for right in group:
                    if left>=right or np.array_equal(vector[left],vector[right]):continue
                    if not np.array_equal(vector[left,keep],vector[right,keep]):raise ValueError('Nonfocal input changed')
                    background=json.dumps([key[1],float(inputs.params[left,SPEC.param_names.index('mdt')])],separators=(',',':'))
                    edges.append(dict(gene=gene,left=int(inputs.ids[left]),right=int(inputs.ids[right]),left_context=contexts[left],right_context=contexts[right],background=background))
                    count+=1;used+=1
                    if used==4 or count==per_gene:break
                if used==4 or count==per_gene:break
            if count==per_gene:break
    return edges


def split_contrasts(edges, *, per_gene_role=3, seed=42):
    """Whole endpoint contexts AND focal-gene/background roles are disjoint."""
    rng=np.random.default_rng(seed);roles={k:[] for k in ('candidate','feedback','check')}
    assigned={};backgrounds={};counts=defaultdict(int)
    for index in rng.permutation(len(edges)):
        e=edges[index];contexts={e['left_context'],e['right_context']};bg=(e['gene'],e['background'])
        for role in sorted(roles,key=lambda r:(counts[e['gene'],r],r)):
            if counts[e['gene'],role]>=per_gene_role:continue
            if any(c in assigned and assigned[c]!=role for c in contexts):continue
            if bg in backgrounds and backgrounds[bg]!=role:continue
            if any((e['left'],e['right'])==(q['left'],q['right']) for q in roles[role]):continue
            roles[role].append(e);counts[e['gene'],role]+=1
            assigned.update({c:role for c in contexts});backgrounds[bg]=role;break
    if any(len(v)<4 for v in roles.values()):raise ValueError('Insufficient exact independent contrasts')
    return roles


def safe_cosine(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite gradient')
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator>0 else None
