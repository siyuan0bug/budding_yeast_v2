"""PEAKS-Regression-Reference: first-order kernel influence and streaming IDS.

The classification class prototype has no canonical continuous-label analogue.
Use the mean gradient on selected training references instead: g_ref·g_candidate
= output-error derivative × output-head NTK × reference-error derivative. This
is an explicit regression/reference adaptation, not the original class-logit
proxy. Quantile acceptance, cache refresh and new+old training batches remain.
"""
from bisect import insort,bisect_left,bisect_right
import numpy as np


class PEAKSStream:
    def __init__(self,selection_percent=10.,refresh=500,cache_capacity=20000):
        if not 0<selection_percent<=100 or min(refresh,cache_capacity)<1:raise ValueError('Invalid PEAKS stream settings')
        self.percent=float(selection_percent);self.refresh=int(refresh);self.capacity=int(cache_capacity)
        self.cache=[];self.history=[];self.anchor=None;self.decisions=0;self.accepted=0;self.refreshes=0

    def set_reference(self,gradients,step):
        g=np.asarray(gradients,dtype=np.float64)
        if g.ndim!=2 or not len(g) or not np.isfinite(g).all():raise ValueError('Invalid PEAKS reference gradients')
        self.anchor=g.mean(0);self.cache=[];self.history=[];self.refreshes+=1;self.reference_step=int(step)

    def decide(self,gradient):
        if self.anchor is None:raise RuntimeError('PEAKS reference has not been refreshed')
        score=float(np.asarray(gradient)@self.anchor)
        if not np.isfinite(score):raise FloatingPointError('Nonfinite PEAKS influence')
        if len(self.history)==self.capacity:
            oldest=self.history.pop(0);self.cache.pop(bisect_left(self.cache,oldest))
        self.history.append(score);insort(self.cache,score)
        # scipy.stats.percentileofscore(kind='rank'), with current score included.
        left,right=bisect_left(self.cache,score),bisect_right(self.cache,score)
        percentile=50.*(left+right+1)/len(self.cache)
        accept=percentile>=100.-self.percent
        self.decisions+=1;self.accepted+=int(accept)
        return accept,dict(score=score,percentile=percentile)

    def state_dict(self):return dict(self.__dict__)
    def load_state_dict(self,state):self.__dict__.update(state)
