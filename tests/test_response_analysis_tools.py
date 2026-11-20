"""Verify query accounting/score decomposition and streamed saved-label reads."""
import json
import numpy as np

from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_response_acquisitions import trace_statistics
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_response_hub import read_rows


def test_trace_decomposes_reused_point_reward_without_double_charging(tmp_path):
    source=tmp_path/'after_round_00';source.mkdir()
    path=source/'response_pairs.json'
    path.write_text(json.dumps(dict(response_actions=1,timings=[],choices=[
        dict(kind='point',endpoints=[dict(kind='pool',candidate_id=0)],score=4,new_queries=1),
        dict(kind='response',endpoints=[dict(kind='pool',candidate_id=0),dict(kind='pool',candidate_id=1)],
             score=3,response_variance=1,distance=0,new_queries=1),
        dict(kind='random',endpoints=[dict(kind='pool',candidate_id=2)],new_queries=1)])))
    report=trace_statistics(path,tmp_path)
    assert report['unique_queries']==3
    assert report['reused_endpoint_new_queries']==1
    assert report['reconstructed_point_scores']==2
    np.testing.assert_allclose(report['response_reward_p10_p50_p90'],[1/6]*3)
    np.testing.assert_allclose(report['reused_old_point_reward_p10_p50_p90'],[2/3]*3)


def test_streamed_rows_preserve_requested_order(tmp_path):
    data=np.arange(10*39*500,dtype=np.float32).reshape(10,39,500)
    path=tmp_path/'fixture.npz';np.savez_compressed(path,data=data)
    np.testing.assert_array_equal(read_rows(path,[5,0,9]),data[[5,0,9]])
