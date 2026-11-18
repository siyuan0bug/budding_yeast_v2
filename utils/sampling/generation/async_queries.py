"""Bounded asynchronous ODE futures and an append-only cost ledger.

Only the supplied executor's children may be terminated. There is no recursive
process launch and no unbounded submission queue or unbounded result buffer.
"""
from concurrent.futures import FIRST_COMPLETED,wait
import json
import os
import time

from ..benchmark.runner import _json_safe
from budding_yeast_v2.data.systems.budding.offline_pool import simulate


class AsyncQueries:
    def __init__(self,executor,ledger,limit,*,timeout=35.,function=simulate):
        if limit<1 or timeout<=0:raise ValueError('Invalid asynchronous queue bounds')
        self.executor=executor;self.path=ledger;self.limit=int(limit);self.timeout=timeout;self.function=function
        self.pending={};self.submitted=0;self.finished=0;self.failed=0

    def _log(self,records):
        with self.path.open('a') as log:
            for row in records:log.write(json.dumps(_json_safe(row),allow_nan=False)+'\n')
            log.flush();os.fsync(log.fileno())

    def submit(self,candidates,*,model_step):
        if len(candidates)+len(self.pending)>self.limit:raise ValueError('Asynchronous queue is full')
        for candidate in candidates:
            self._log([dict(event='start',proposal_id=candidate.metadata['generation_proposal_index'],
                            model_step=int(model_step),overrides=candidate.overrides,
                            is_reference=candidate.metadata.get('ogas_is_reference'))])
            future=self.executor.submit(self.function,candidate)
            self.pending[future]=(candidate,time.monotonic());self.submitted+=1

    def poll(self,*,block=False):
        if not self.pending:return []
        done={f for f in self.pending if f.done()}
        if not done and block:
            done,_=wait(self.pending,timeout=min(self.timeout,1.),return_when=FIRST_COMPLETED)
        if not done:
            if time.monotonic()-min(v[1] for v in self.pending.values())>self.timeout:
                self.abort();raise TimeoutError('Asynchronous canonical ODE worker exceeded parent deadline')
            return []
        # Completion order is intentionally online, and is preserved by the
        # ledger/source manifests. Seeds alone cannot reproduce OS timing.
        results=[]
        for future in sorted(done,key=lambda f:self.pending[f][0].metadata['generation_proposal_index']):
            candidate,_=self.pending.pop(future);result=future.result();self.finished+=1
            self.failed+=int(result.status!='success')
            self._log([dict(event='finish',proposal_id=candidate.metadata['generation_proposal_index'],
                status=result.status,overrides=candidate.overrides,diagnostics=result.diagnostics,
                is_reference=candidate.metadata.get('ogas_is_reference'))])
            results.append(result)
        return results

    def abort(self):
        for future in self.pending:future.cancel()
        processes=list((getattr(self.executor,'_processes',None) or {}).values())
        for process in processes:
            if process.is_alive():process.terminate()
        for process in processes:
            process.join(timeout=1.)
            if process.is_alive():process.kill();process.join(timeout=1.)
        self.pending.clear()


def query_costs(path):
    starts=finishes=failures=solver_calls=0
    if path.exists():
        with path.open() as log:
            for line in log:
                row=json.loads(line)
                if row['event']=='start':starts+=1
                else:
                    finishes+=1;failures+=int(row['status']!='success')
                    solver_calls+=len(row['diagnostics'].get('attempts',[None]))
    return dict(attempted_inputs=starts,finished_inputs=finishes,failed_inputs=failures,
                solver_attempts=solver_calls+starts-finishes,solver_attempts_exact=starts==finishes,
                incomplete_query_attempts=starts-finishes)
