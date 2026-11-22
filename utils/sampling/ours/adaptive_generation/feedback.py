"""Pre-query observations only: calibration diagnostics and regional progress."""
import numpy as np


class Feedback:
    def __init__(self):self.round=0;self.rows=[];self._groups=None

    def observe(self,key,predicted_error,actual_error):
        if not np.isfinite([predicted_error,actual_error]).all() or min(predicted_error,actual_error)<0:
            raise ValueError('Invalid prequential feedback')
        self.rows.append(dict(key=str(key),round=int(self.round),score=float(predicted_error),error=float(actual_error)))
        self._groups=None

    def grouped(self):
        if self._groups is None:
            from collections import defaultdict
            self._groups=defaultdict(list)
            for r in self.rows:self._groups[r['key']].append(r)
        return self._groups

    def recent_error(self,key):
        rows=self.grouped().get(key,[])
        return float(np.mean([r['error'] for r in rows[-32:]])) if rows else 1.

    def weight(self,key):
        rows=self.grouped().get(key,[]);epochs=sorted({r['round'] for r in rows})
        if len(epochs)<2:return 1.  # unknown, do not silently suppress exploration
        old=np.mean([r['error'] for r in rows if r['round']==epochs[-2]])
        new=np.mean([r['error'] for r in rows if r['round']==epochs[-1]])
        return float(np.clip(1.+(old-new)/max(old,1e-8),.25,2.))

    def summary(self):
        correlation=None
        if len(self.rows)>2:
            from scipy.stats import spearmanr
            score=np.array([r['score'] for r in self.rows]);error=np.array([r['error'] for r in self.rows])
            if np.ptp(score)>0 and np.ptp(error)>0:correlation=float(spearmanr(score,error).statistic)
        return dict(prequery_score_error_spearman=correlation,prequery_feedback_count=len(self.rows),feedback_round=self.round,
            progress_estimator='regional prequential error change; different inputs, not causal learning gain',
            mean_prequery_error=float(np.mean([r['error'] for r in self.rows])) if self.rows else None)

    def state_dict(self):return dict(round=self.round,rows=self.rows)
    def load_state_dict(self,value):
        self.round=int(value['round']);self.rows=list(value['rows']);self._groups=None
        for r in self.rows:
            if not np.isfinite([r['score'],r['error']]).all():raise ValueError('Nonfinite restored feedback')
