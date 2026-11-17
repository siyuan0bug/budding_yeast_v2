"""One bounded CPU generator thread; surrogate updates never wait for a cycle.

Only explicit checkpoint/quota barriers wait. There is no second GPU process,
no unbounded queue, and no access to the surrogate, oracle or test cohorts.
"""
from collections import deque
from copy import deepcopy
import threading
import time

import numpy as np


class BackgroundOGAS:
    def __init__(self,controller,target_capacity):
        if controller.device.type!='cpu':raise ValueError('Background generator currently validates CPU execution only')
        if target_capacity<1:raise ValueError('Positive generator cache capacity required')
        self.controller=controller;self.capacity=target_capacity
        self.condition=threading.Condition();self.feedback=deque();self.feedback_rows=0
        self.targets=deque(maxlen=target_capacity);self.reference=None
        self.requested=False;self.busy=False;self.stopping=False;self.error=None
        self.dropped_feedback_rows=0;self.completed_cycles=0;self.latest={}
        self.thread=threading.Thread(target=self._run,name='ogas-generator',daemon=True);self.thread.start()

    def observe(self,x,signal):
        with self.condition:
            self._check()
            self.feedback.append((np.asarray(x).copy(),np.asarray(signal).copy()));self.feedback_rows+=len(x)
            # Same recent-feedback principle as B; bounded even if generation
            # cannot keep up. Evictions are explicitly counted in the report.
            while self.feedback_rows>self.controller.config.capacity and len(self.feedback)>1:
                old,_=self.feedback.popleft();self.feedback_rows-=len(old);self.dropped_feedback_rows+=len(old)
            if self.feedback_rows>self.controller.config.capacity:
                old,signals=self.feedback.pop();extra=len(old)-self.controller.config.capacity
                self.feedback.append((old[extra:],signals[extra:]))
                self.feedback_rows-=extra;self.dropped_feedback_rows+=extra

    def request(self,reference):
        with self.condition:
            self._check()
            self.reference=np.asarray(reference).copy();self.requested=True;self.condition.notify_all()

    def take(self,count):
        with self.condition:
            self._check()
            rows=[self.targets.popleft() for _ in range(min(count,len(self.targets)))]
            return np.asarray(rows) if rows else np.empty((0,self.controller.dimension),np.float32)

    def status(self):
        with self.condition:
            self._check()
            return dict(self.latest,busy=self.busy,queued_targets=len(self.targets),
                completed_cycles=self.completed_cycles,dropped_feedback_rows=self.dropped_feedback_rows)

    def _check(self):
        if self.error is not None:raise RuntimeError('Background OGAS generator failed') from self.error
        if self.stopping:raise RuntimeError('Background OGAS generator is closed')

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.stopping or self.requested)
                    if self.stopping:return
                    feedback=list(self.feedback);self.feedback.clear();self.feedback_rows=0
                    reference=self.reference.copy();self.requested=False;self.busy=True
                for x,signal in feedback:self.controller.observe(x,signal)
                if feedback:
                    diagnostics=self.controller.train_cycle(reference)
                elif self.controller.cycles:diagnostics=dict(rule='sample existing conditional generator, no stale-buffer retraining')
                else:raise ValueError('Background OGAS requires revealed training feedback')
                targets,_=self.controller.sample_targets(self.capacity,reference,exploration=False)
                with self.condition:
                    self.targets.extend(targets);self.latest=diagnostics
                    self.completed_cycles+=1;self.busy=False;self.condition.notify_all()
        except BaseException as exc:
            with self.condition:self.error=exc;self.busy=False;self.requested=False;self.condition.notify_all()

    def synchronize(self,timeout=35.):
        deadline=time.monotonic()+timeout
        with self.condition:
            while self.busy or self.requested:
                self._check();remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Background OGAS did not reach checkpoint barrier')
                self.condition.wait(timeout=min(remaining,1.))
            self._check()

    def state_dict(self):
        self.synchronize()
        with self.condition:
            return deepcopy(dict(controller=self.controller.state_dict(),feedback=list(self.feedback),
                feedback_rows=self.feedback_rows,targets=list(self.targets),reference=self.reference,
                dropped_feedback_rows=self.dropped_feedback_rows,completed_cycles=self.completed_cycles,latest=self.latest))

    def load_state_dict(self,state):
        self.synchronize()
        with self.condition:
            self.controller.load_state_dict(state['controller']);self.feedback=deque(state['feedback'])
            self.feedback_rows=state['feedback_rows'];self.targets=deque(state['targets'],maxlen=self.capacity)
            self.reference=state['reference'];self.dropped_feedback_rows=state['dropped_feedback_rows']
            self.completed_cycles=state['completed_cycles'];self.latest=state['latest']

    def close(self):
        with self.condition:self.stopping=True;self.condition.notify_all()
        self.thread.join(timeout=35.)
        if self.thread.is_alive():raise TimeoutError('Background OGAS thread failed to stop')
