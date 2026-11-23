"""Budgeted action selection: missing response estimates never get a numeric score."""
from collections import Counter
import numpy as np
from ...baseline.path_random import choose


def select_actions(bank,scores,supported,count,rng,exploration_due,random_control=False):
    metadata=bank.metadata;local=[i for i,m in enumerate(metadata) if m.get('adaptive_path_id')]
    global_ids=[i for i,m in enumerate(metadata) if not m.get('adaptive_path_id')]
    selected=[];actions=[];used=set();per_path=Counter();supported=np.asarray(supported,bool)
    def take(i,action):
        selected.append(i);actions.append(action);used.add(i)
        if metadata[i].get('adaptive_path_id'):per_path[metadata[i]['adaptive_path_id']]+=1
    # Independent exploration allowance; not ranked alongside measured response errors.
    for i in rng.permutation(global_ids)[:min(exploration_due,count)]:take(int(i),'explore')
    single={metadata[i]['adaptive_path_id'] for i in local if not supported[i]}
    bootstrap_limit=count-len(selected) if not supported.any() else min(1,count-len(selected))
    for key in rng.permutation(sorted(single))[:bootstrap_limit]:
        candidates=[i for i in local if metadata[i]['adaptive_path_id']==key and i not in used]
        if candidates:take(choose(candidates,rng),'support')
    while len(selected)<count:
        candidates=[i for i in local if supported[i] and i not in used and per_path[metadata[i]['adaptive_path_id']]<4]
        if not candidates:break
        if random_control:i=choose(candidates,rng)
        else:
            order=rng.permutation(candidates)
            i=int(order[np.argmax([scores[j]/(1+per_path[metadata[j]['adaptive_path_id']]) for j in order])])
        take(i,'response' if not random_control else 'path_random')
    # Capacity shortage is explicit, never disguised as estimated response gain.
    for i in rng.permutation([i for i in range(len(metadata)) if i not in used]):
        if len(selected)==count:break
        take(int(i),'capacity_fallback')
    return np.asarray(selected),actions
