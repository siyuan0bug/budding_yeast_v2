"""Parallel members preserve independent seeded training and read-only validation."""
from dataclasses import replace
import json
import torch
from test_sampling_benchmark import make_pool,config
from budding_yeast_v2.utils.sampling.benchmark.protocol import prepare_protocol
from budding_yeast_v2.utils.sampling.benchmark.runner import run_experiment


def test_parallel_matches_serial_members_and_selection(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path/'locks'))
    monkeypatch.setenv('SAMPLING_IO_CONCURRENCY','1')
    monkeypatch.setenv('SAMPLING_IO_ADMISSION','warn')
    source,_=make_pool(tmp_path/'pool')
    cfg=replace(config(),budgets=(2,4),epochs=(2,2),warmup=(0,0))
    protocol=prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',cfg)
    runs=[]
    for mode in ('0','1'):
        monkeypatch.setenv('SAMPLING_ENSEMBLE_PARALLEL',mode)
        runs.append(run_experiment(protocol,'mutation','power',tmp_path/f'mode{mode}',diagnostic_only=True))
    for relative in ['rounds/round_00/model.ckpt','rounds/round_01/model.ckpt','_state/round_00/ensemble_member_1/model.ckpt']:
        a,b=[torch.load(r/relative,map_location='cpu',weights_only=False) for r in runs]
        for name,value in a['state_dict'].items():torch.testing.assert_close(value,b['state_dict'][name],rtol=0,atol=0)
    a,b=[json.loads((r/'rounds/round_00/selection.json').read_text()) for r in runs]
    assert a['selected_ids']==b['selected_ids']
    assert (runs[1]/'_state/round_00/ensemble_member_1/train.log').exists()


def test_exclusive_cards_reject_parallel_children(monkeypatch,tmp_path):
    import pytest
    from budding_yeast_v2.utils.sampling.benchmark.ensemble import parallel_members
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','4')
    with pytest.raises(ValueError,match='exclusive GPUs'):
        with parallel_members(config(),0,tmp_path,'mutation',tmp_path/'data.json',tmp_path,{},'cuda:0'):pass
