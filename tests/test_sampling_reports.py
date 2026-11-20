"""Output contracts: physical per-row reports, informative plots, no duplicate data."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from budding_yeast_v2.utils.evaluation import reporting
from budding_yeast_v2.utils.evaluation.labels import get_real_mutations_str
from budding_yeast_v2.utils.experiments.curves import loss_figure,all_rounds_figure,performance_figure


def test_canonical_real_names_survive_sanitization_and_unknown_is_not_wt():
    message=get_real_mutations_str('15_cln3Δ_bck2Δ_multi-copy_CLN2')
    assert 'WT' not in message and 'ksn2pp=0.75' in message
    assert get_real_mutations_str('unknown')=='Canonical mutation definition unavailable'
    assert get_real_mutations_str('1_Wild-type')=='mdt=90'
    assert 'Vmik=0' in get_real_mutations_str('mik1D_wee1ts','fission')


def test_real_and_source_synth_exports_and_full_plot_header(tmp_path,monkeypatch):
    y=np.ones((2,39,50),dtype=np.float32);p=y*.9
    names=['15_cln3Δ_bck2Δ_multi-copy_CLN2','1_Wild-type']
    tasks=[]
    monkeypatch.setattr(reporting,'render_prediction_plots',lambda items,workers:tasks.extend(items))
    summary=reporting.save_real_evaluation(p,y,names,tmp_path,plots=True,
        context=dict(model='fno',method='random',round=0,seed=42))
    assert len(tasks)==2 and tasks[0][4:6]==(5,8)
    assert all(key in tasks[0][2] for key in ('MAE=','MSE=','Rel-L2=','Corr=','Pattern_','P5a','seed: 42','ksn2pp='))
    reporting._render_and_save_plot(tasks[0])
    image=Image.open(tasks[0][3]);assert image.width>image.height
    assert image.info['Description']==tasks[0][2]
    real=list(csv.DictReader((tmp_path/'per_mutant_metrics.csv').open()))
    assert len(real)==2 and float(real[0]['MAE'])==pytest.approx(.1,abs=1e-6)
    reporting.export_sample_metrics(p,y,['a','b'],tmp_path/'per_synth_test_metrics.csv',
        source='lhs',pool_rows=[11,25],candidate_ids=[100,300],metadata={'labels':['x','y']})
    synth=list(csv.DictReader((tmp_path/'per_synth_test_metrics.csv').open()))
    assert [(r['source'],r['pool_row'],r['source_candidate_id']) for r in synth]==[('lhs','11','100'),('lhs','25','300')]
    assert synth[0]['labels']=='x' and synth[0]['pattern_group']=='P5a'
    result=reporting.save_global_metrics(tmp_path,{'lhs_test':summary,'mutation_test':summary,'real_mutants':summary},
                                       pool_name='lhs',round_index=0,training_rows=2000)
    assert list(result)==['round','train_pool','train_samples','real_mutants','synth_test','cross_pool_test']
    assert result['synth_test']['source']=='lhs' and result['cross_pool_test']['source']=='mutation'
    assert result['real_mutants']['all_39_vars']==summary['all_variables']
    assert 'all_variables' not in result['real_mutants']
    assert not (tmp_path/'metrics_denorm.json').exists()


def test_fission_exports_do_not_invent_budding_patterns(tmp_path):
    y=np.ones((1,16,30),dtype=np.float32)
    reporting.save_real_evaluation(y,y,['WT'],tmp_path)
    data=json.loads((tmp_path/'global_metrics_denorm.json').read_text())['real_mutants']
    assert 'all_16_vars' in data and 'core_9_vars' not in data and 'pattern_metrics' not in data
    row=next(csv.DictReader((tmp_path/'per_mutant_metrics.csv').open()))
    assert not any('pattern' in key for key in row)


def test_original_curve_style_all_rounds_and_no_single_round_performance(tmp_path,monkeypatch):
    import matplotlib.figure
    captured=[];save=matplotlib.figure.Figure.savefig
    def capture(fig,*args,**kwargs):
        captured.append(fig)
        return save(fig,*args,**kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure,'savefig',capture)
    history=[dict(epoch=i,train_loss=1/i,val_error=.5/i,val_loss_synth=.8/i,lr=.001/i,
                  validation_metric='val_macro_rel_l2') for i in range(1,6)]
    loss_figure(history,tmp_path/'loss.png',context=dict(model='fno',method='random',loss='hybrid_mse_rel',warmup=1))
    labels=[line.get_label() for ax in captured[-1].axes for line in ax.lines]
    assert 'Train Synth Loss' in labels and 'Val Synth Loss' in labels
    assert 'Validation macro trajectory RelL2' in labels
    assert 'dimensionless' in captured[-1].axes[1].get_ylabel()
    assert not any('Test Real' in label for label in labels)
    all_rounds_figure([(i,history) for i in range(5)],tmp_path/'all.png')
    assert len(captured[-1].axes)==10 # five shared dual-axis panels, LR insets attached
    record=dict(training_rows=2000,metrics={})
    performance_figure([record],tmp_path/'performance.png')
    assert not (tmp_path/'performance.png').exists()


def test_local_wandb_and_single_round_artifacts(tmp_path,monkeypatch):
    from dataclasses import replace
    from test_sampling_benchmark import make_pool,config
    from budding_yeast_v2.utils.sampling.benchmark.protocol import prepare_protocol
    from budding_yeast_v2.utils.sampling.benchmark.runner import run_experiment
    monkeypatch.setenv('WANDB_MODE','online')
    monkeypatch.setenv('WANDB_SILENT','true')
    source,_=make_pool(tmp_path/'pool')
    cfg=replace(config(),budgets=(2,),epochs=(1,),warmup=(0,))
    protocol=prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',cfg)
    run=run_experiment(protocol,'lhs','random',tmp_path/'runs',diagnostic_only=True,wandb_enabled=True)
    assert (run/'eval').is_symlink()
    assert (run/'eval/per_synth_test_metrics.csv').is_file()
    assert (run/'loss.png').is_file() and not (run/'performance.png').exists()
    assert len(list((run/'_state/wandb').glob('offline-run-*')))==1
    assert not (run/'wandb').exists()
    assert not list(run.rglob('*.npz'))
    assert not list(run.rglob('history.json')) and not list(run.rglob('last.ckpt'))
    summary=json.loads((run/'eval/global_metrics_denorm.json').read_text())
    assert summary['synth_test']['source']=='lhs' and summary['cross_pool_test']['source']=='mutation'


@pytest.mark.parametrize('monitor,checkpoint_label',[
    ('val_macro_rel_l2','validation macro RelL2'),
    ('val_sample_channel_rel_l2','validation sample/channel RelL2'),
    ('val_loss_synth','normalized validation loss')])
def test_complete_five_curve_panels_use_real_not_validation_on_red_axis(tmp_path,monkeypatch,monitor,checkpoint_label):
    import matplotlib.figure
    captured=[];save=matplotlib.figure.Figure.savefig
    def capture(fig,*args,**kwargs):
        captured.append(fig);return save(fig,*args,**kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure,'savefig',capture)
    rows=[dict(epoch=i,train_loss=1/i,val_error=abs(i-3)+.1,val_loss_synth=.8/i,
        test_loss_synth=.9/i,test_loss_real=1.2/i,test_real_denorm_rel_l2=.6/i,
        lr=.001/i,validation_metric=monitor) for i in range(1,6)]
    if monitor=='val_loss_synth':
        for row in rows:row['val_loss_synth']=row['val_error']
    all_rounds_figure([(i,rows) for i in range(5)],tmp_path/'five.png',
        {i:dict(model='fno',loss='hybrid_mse_rel',training_rows=2000*(i+1),seed=42) for i in range(5)})
    fig=captured[-1]
    assert len(fig.axes)==10
    for i in range(5):
        ax=fig.axes[i];legend=[t.get_text() for t in ax.get_legend().get_texts()]
        assert len(legend)==5
        assert all(name in legend for name in ('Train Synth Loss','Val Synth Loss','Test Synth Loss','Test Real Mutants Loss'))
        right=fig.axes[5+i]
        np.testing.assert_array_equal(right.lines[0].get_ydata(),[r['test_real_denorm_rel_l2'] for r in rows])
        assert 'dimensionless' in right.get_ylabel() and 'validation' not in right.get_ylabel()
        text=' '.join(t.get_text() for t in right.texts)
        assert 'Final Synth Test Loss' in text and 'Final Real Mutants Loss' in text
        assert f'Checkpoint: {checkpoint_label}=0.10000 (Ep 3)' in text
