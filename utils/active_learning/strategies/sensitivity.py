"""Independent, ODE-derived mutation-response consistency acquisition."""
from collections import defaultdict,Counter
import time
from ..auxiliary.views import build_view
from ..auxiliary.sensitivity import score_pool,select_scores,SETTINGS


def _sensitivity_sample(self,trainer,pl_module,dm,device,current_epoch):
    if not self._pool_size:return None
    started=time.monotonic();view=build_view(self,dm)
    scores,details=score_pool(view,pl_module,device,int(self._al4pde_al_iter))
    selected,exploit=select_scores(scores,self._al4pde_scheduled_num_add(),int(self.random_seed+1009*self._al4pde_al_iter))
    index=int(self._al4pde_al_iter)
    if not hasattr(self,'_experimental_audits'):self._experimental_audits=defaultdict(list)
    self._experimental_audits[index].append(dict(strategy=self.strategy,version='canonical_sensitivity_v1',
        seconds=time.monotonic()-started,budget=len(selected),pool_size=self._pool_size,
        feedback=dict(settings=SETTINGS,direction_counts=dict(Counter(details['direction_kind'])),
                      eligible_positive_scores=int((scores>0).sum()),continuous_interval_only=True),
        choices=[dict(kind='sensitivity' if j<exploit else 'random',
            endpoints=[dict(kind='pool',candidate_id=int(view.pool.ids[i]))],new_queries=1,score=float(scores[i]))
            for j,i in enumerate(selected)]))
    if not hasattr(self,'_mechanism_artifacts'):self._mechanism_artifacts={}
    self._mechanism_artifacts.setdefault(index,[]).append(dict(predictions=dict(candidate_ids=view.pool.ids[selected],
        score=scores[selected],**{k:v[selected] for k,v in details.items()})))
    print(f'[{self.strategy}] selected={len(selected)} elapsed_s={time.monotonic()-started:.1f}',flush=True)
    return self.make_selection_plan(selected,dm)
