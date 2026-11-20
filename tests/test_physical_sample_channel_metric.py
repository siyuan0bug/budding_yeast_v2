"""Physical, uncapped Relative L2 and exact-source evaluation contracts."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from budding_yeast_v2.utils.evaluation.metrics import (
    calculate_metrics, per_sample_metrics, sample_channel_relative_errors, summarize)
from budding_yeast_v2.utils.evaluation.reporting import organized_metrics
from budding_yeast_v2.utils.training.losses import RelativeL2Loss


def test_same_formula_as_uncapped_training_and_equal_sample_channel_reduction():
    y=torch.tensor([[[1.,1.],[0.,0.]],[[100.,100.],[2.,2.]]],dtype=torch.float64)
    p=y+torch.tensor([[[1.],[2.]],[[10.],[.2]]],dtype=torch.float64)
    errors=sample_channel_relative_errors(p,y)
    assert errors.shape==(2,2)
    torch.testing.assert_close(errors.mean(),RelativeL2Loss(max_clamp=None)(p,y))
    torch.testing.assert_close(errors.mean(),torch.cat([
        sample_channel_relative_errors(p[i:i+1],y[i:i+1]) for i in range(2)]).mean())
    assert errors[0,1]>10  # genuine large errors remain visible in evaluation
    m=calculate_metrics(p,y)
    assert list(m)[0]=='Relative L2'
    assert m['Relative L2']==pytest.approx(float(errors.mean()))
    assert 'Sample-channel Relative L2' not in m


def test_report_metric_is_physical_no_log_centering_or_cap():
    y=torch.zeros(1,16,4,dtype=torch.float32);p=torch.ones_like(y)
    out=organized_metrics(summarize(p,y,'fission'))
    assert out['metric_space']=='physical' and out['primary_metric']=='sample_channel_rel_l2'
    assert out['sample_channel_rel_l2']==pytest.approx(200.)
    assert list(out).index('sample_channel_rel_l2') < list(out).index('all_16_vars')
    assert per_sample_metrics(p[0],y[0])['rel_l2']==pytest.approx(200.)
    assert 'sample_channel_rel_l2' not in out['trajectory_metrics']


def test_static_evaluation_uses_exact_source_truth_and_parameters(tmp_path):
    from budding_yeast_v2.eval import evaluate_dataset
    raw=np.full((3,16,4),-1e-6,dtype=np.float32)
    params=np.arange(3*141,dtype=np.float64).reshape(3,141)+1e-10
    dm=SimpleNamespace(system_id='fission',num_vars=16,T_STEPS=4,num_workers=0,
        _raw_data=raw,_param_conds=params,_mutant_names=np.array(['a','b','c']),
        pattern_labels=np.array(['NA']*3),denormalize_y=lambda y,device:y)
    dataset=TensorDataset(torch.zeros(2,16,2,4),torch.zeros(2,141),torch.zeros(2,16,4))
    model=SimpleNamespace(forward_ic_time=lambda x,p:torch.zeros(len(x),16,4))
    summary,rows,_,targets,_=evaluate_dataset(model,dm,dataset,[2,0],torch.device('cpu'),
        None,None,42,str(tmp_path),'test',batch_size=1,max_plots=0)
    np.testing.assert_array_equal(targets[0].numpy(),raw[[2,0]])
    assert [r['Mutant'] for r in rows]==['c','a']
    assert summary['all_variables']['MSE']==pytest.approx(1e-12)
    assert rows[0]['Relative_L2_Version']=='physical_sample_channel_rel_l2_smooth_eps001_v3'


def test_plot_and_csv_prioritize_uncapped_new_physical_metric(tmp_path, monkeypatch):
    from budding_yeast_v2.utils.evaluation import reporting
    tasks=[]
    monkeypatch.setattr(reporting,'render_prediction_plots',lambda rows,workers:tasks.extend(rows))
    y=np.zeros((1,16,4),dtype=np.float32);p=np.ones_like(y)
    reporting.save_real_evaluation(p,y,['WT'],tmp_path,plots=True)
    assert 'Rel-L2=200.00000 (sample/channel, physical)' in tasks[0][2]
    assert 'Legacy' not in tasks[0][2]
    import csv,json
    with (tmp_path/'per_mutant_metrics.csv').open() as handle:
        row=next(csv.DictReader(handle))
    assert float(row['rel_l2'])==pytest.approx(200.)
    assert 'sample_channel_rel_l2' not in row
    assert row['metric_version']=='physical_sample_channel_rel_l2_smooth_eps001_v3'
    result=json.loads((tmp_path/'global_metrics_denorm.json').read_text())['real_mutants']
    assert result['sample_channel_rel_l2']==pytest.approx(200.)


def test_canonical_examples_and_zero_target_protection():
    from budding_yeast_v2.utils.evaluation.metrics import relative_l2_loss
    y=torch.tensor([1.,100.],dtype=torch.float64)[:,None,None].expand(-1,1,500)
    p=torch.tensor([2.,110.],dtype=torch.float64)[:,None,None].expand(-1,1,500)
    expected=(1/(1+.01**2/500)**.5 + .1/(1+.01**2/(500*100**2))**.5)/2
    assert float(relative_l2_loss(p,y))==pytest.approx(expected,rel=1e-12)
    y=torch.zeros(1,1,500,dtype=torch.float64);p=torch.full_like(y,.01)
    assert float(relative_l2_loss(p,y))==pytest.approx(22.360679775)


def test_aggregate_rejects_different_relative_definitions():
    from budding_yeast_v2.scripts.analysis.analyze_experiments import aggregate_group_metrics
    block={'all_39_vars':{'MAE':1.,'Relative L2':.2,'MSE':1.,'Correlation':.5}}
    old={'real_mutants':dict(block)}
    new={'real_mutants':dict(block,metric_version='physical_sample_channel_rel_l2_smooth_eps001_v3')}
    with pytest.raises(ValueError,match='different Relative L2'):
        aggregate_group_metrics({'seeds':[{'seed':42,'metrics':old},{'seed':2026,'metrics':new}]})
