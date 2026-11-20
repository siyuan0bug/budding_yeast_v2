"""Large AL query batches must not eagerly flood the process executor."""
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import closing
import multiprocessing as mp

import pytest

from budding_yeast_v2.data.simulation_guards import bounded_ordered_map


def test_submission_window_also_bounds_completed_out_of_order_results():
    state = {'submitted':0, 'consumed':0, 'largest':0}
    class Executor:
        _max_workers = 3
        def submit(self, function, task):
            state['submitted'] += 1
            outstanding = state['submitted'] - state['consumed']
            state['largest'] = max(state['largest'],outstanding)
            assert outstanding <= 6
            future = Future();future.set_result(function(task));return future
    rows=[]
    for value in bounded_ordered_map(Executor(),abs,range(-30000,0)):
        rows.append(value);state['consumed'] += 1
    assert rows == list(range(30000,0,-1))
    assert state['largest'] == 6


def test_early_quota_close_does_not_submit_remaining_candidates():
    submitted=[]
    class Executor:
        _max_workers=2
        def submit(self,function,task):
            submitted.append(task);future=Future();future.set_result(function(task));return future
    with closing(bounded_ordered_map(Executor(),abs,range(30000))) as results:
        assert next(results)==0
    assert len(submitted)==4


def test_spawn_executor_returns_ordered_real_results():
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as executor:
        assert list(bounded_ordered_map(executor,abs,range(-100,0),stall_timeout=35))==list(range(100,0,-1))


def test_task_error_propagates():
    class Executor:
        def submit(self,function,task):
            future=Future();future.set_exception(ValueError('solver wrapper error'));return future
    with pytest.raises(ValueError,match='solver wrapper error'):
        list(bounded_ordered_map(Executor(),abs,[1]))
