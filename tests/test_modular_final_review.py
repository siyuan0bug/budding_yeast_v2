"""Cross-stage regression cases found during the final refactor review."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.systems import generation as runner
from budding_yeast_v2.data.systems.budding import derived
from budding_yeast_v2.data.systems.contracts import Candidate,GenerationConfig
from budding_yeast_v2.data.systems.dataset_io import read_dataset,write_dataset
from budding_yeast_v2.data.systems.registry import RECIPES
from budding_yeast_v2.data.systems.registry import make_sampler
from budding_yeast_v2.data.systems.budding import real_mutants
from budding_yeast_v2.data.systems.budding.adapter import BuddingAdapter
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem,SPEC

REPORT={'mutation_profile':'mutant_aware','distribution_profile':'mutant_aware','rng_contract':'test'}


def row(candidate):
    result=BuddingAdapter().parameter_result(candidate,None)
    result.trajectory=np.repeat(result.y_init[:,None],500,axis=1)
    result.label='Viable';result.pattern='test';result.status='success'
    return result


def save_source(path,rows):
    config=GenerationConfig('budding_random',output=str(path))
    return write_dataset(path,SPEC,rows,config,REPORT)


def test_real_mutants_modules_are_named_and_copy_rules():
    from budding_yeast_v2.data.systems.fission import real_mutants as fission
    assert len(real_mutants.real_tasks())==126 and len(fission.real_tasks())==20
    for module in (real_mutants,fission):
        name,overrides=module.real_tasks()[0]
        overrides['unexpected']=1
        assert 'unexpected' not in module.real_tasks()[0][1]
        assert Path(module.__file__).name=='real_mutants.py'


def test_fission_warmup_does_not_swallow_timeout(monkeypatch):
    from budding_yeast_v2.data.systems.fission import states,simulation
    def timeout(*args,**kwargs):raise TimeoutError('deadline')
    monkeypatch.setattr(simulation,'run_event_sim',timeout)
    with pytest.raises(TimeoutError):states.mutant_birth_state({},np.zeros(16))


@pytest.mark.parametrize('label_extension',[False,True])
def test_fission_simulation_and_label_extension_propagate_timeout(monkeypatch,label_extension):
    from budding_yeast_v2.data.systems.fission import simulation
    monkeypatch.setattr(simulation,'mutant_birth_state',lambda *args:np.zeros(16))
    info={'status':'End','rdna_at_div':[],'post_div_states':[np.zeros(16)],
          'birth_masses':[1.4048],'div_times':[150.],'completed_time':300.}
    calls=[]
    def timed_run(*args,**kwargs):
        calls.append(1)
        if label_extension and len(calls)==1:return [],info
        raise TimeoutError('deadline')
    monkeypatch.setattr(simulation,'run_event_sim',timed_run)
    monkeypatch.setattr(simulation,'resample_uniform',lambda *args,**kwargs:np.ones((16,500)))
    with pytest.raises(TimeoutError):simulation.simulate_mutant('test',{},np.zeros(16))


def test_rescue_updates_machine_readable_modes(monkeypatch):
    sampler=make_sampler(RECIPES['budding_motif'],GenerationConfig('budding_motif'))
    checks=iter([True,False])
    monkeypatch.setattr(real_mutants,'is_duplicate_real_mutant',lambda *args:next(checks))
    from budding_yeast_v2.data.systems.budding import adapter as budding_adapter
    monkeypatch.setattr(budding_adapter,'rescue_budding',lambda *args:({'mdt':150.,'ks20pp':.01},'CDC20[DAmP]'))
    candidate=Candidate(1,'old',{'mdt':90.},1,metadata={'description':'CDC20_KO','pairs':[('CDC20','KO')]})
    fixed=runner.ensure_candidate(candidate,sampler,Counter())
    assert fixed.metadata['pairs']==[('CDC20','DAmP')]


def test_twin_rejects_carbon_rescue(monkeypatch):
    b={'mdt':90.,'kasrent':2.5,'kasrentp':.01}
    a={'mdt':90.,'kasrent':10.,'kasrentp':.05}
    monkeypatch.setattr(real_mutants,'is_duplicate_real_mutant',lambda ovr,*args:ovr==a)
    monkeypatch.setattr(derived,'ensure_candidate',lambda candidate,*args:replace(candidate,
                        overrides=dict(candidate.overrides,mdt=150.),
                        metadata={'description':'NET1[TAB6-1]_MDT[150]'}))
    sampler=make_sampler(RECIPES['budding_random'],GenerationConfig('budding_twins'))
    with pytest.raises(RuntimeError,match='preserve mdt'):
        derived._twin_rescue(b,a,'NET1[TAB6-1]',1,1,sampler,Counter())


def test_source_preserves_pair_ids_and_metadata(tmp_path):
    candidate=Candidate(1,'RANDOM_CLB2[ts]_L1_1',{'mdt':150.},1,pair_id=17,metadata={'source_tag':'keep'})
    path=save_source(tmp_path/'source.npz',[row(candidate)])
    _,rows=derived._source_rows(GenerationConfig('budding_ko_mirror',source=path))
    assert rows[0].candidate.pair_id==17
    assert rows[0].candidate.metadata['source_tag']=='keep'


def test_source_real_flag_cannot_redefine_canonical_real(tmp_path):
    name,overrides=real_mutants.real_tasks()[0]
    forged=row(Candidate(-1,name,dict(overrides,mdt=160.),0,True))
    path=save_source(tmp_path/'forged.npz',[forged])
    with pytest.raises(ValueError,match='canonical'):
        derived._source_rows(GenerationConfig('budding_ko_mirror',source=path))


def test_source_rejects_conflicting_parameter_metadata(tmp_path):
    candidate=Candidate(1,'RANDOM_CLB2[ts]_L1_1',{'mdt':150.},1)
    source=row(candidate)
    source.params[list(SPEC.param_names).index('mdt')]=90.
    path=save_source(tmp_path/'source.npz',[source])
    with pytest.raises(ValueError,match='parameter'):
        derived._source_rows(GenerationConfig('budding_ko_mirror',source=path))


def test_failed_status_cannot_be_exported_as_finite_trajectory(tmp_path):
    result=row(Candidate(1,'candidate',{},1))
    result.status='numerical_failure'
    path=tmp_path/'failure.npz'
    with pytest.raises(ValueError,match='status'):
        write_dataset(path,SPEC,[result],GenerationConfig('budding_random'),REPORT)
    assert not path.exists()


def test_pair_result_count_cannot_silently_shrink(monkeypatch):
    candidate=Candidate(1,'pair',{},1,pair_id=1)
    calls=iter([[row(candidate)],[]])
    monkeypatch.setattr(derived,'run_simulations',lambda *args:iter(next(calls)))
    with pytest.raises(ValueError,match='paired'):
        derived._paired_results([candidate],[candidate],GenerationConfig('budding_twins'))


def test_paired_rows_cannot_be_matched_by_position_after_reordering(monkeypatch):
    a=Candidate(1,'left',{},1,pair_id=1)
    b=Candidate(2,'right',{},1,pair_id=2)
    calls=iter([[row(a)],[row(b)]])
    monkeypatch.setattr(derived,'run_simulations',lambda *args:iter(next(calls)))
    with pytest.raises(ValueError,match='paired'):
        derived._paired_results([a],[a],GenerationConfig('budding_twins'))


def test_output_suffix_is_checked_before_sampling(tmp_path,monkeypatch):
    def unexpected(*args,**kwargs):raise AssertionError('Sampling ran before validating output')
    monkeypatch.setattr(runner,'prepare_candidates',unexpected)
    with pytest.raises(ValueError,match='.npz'):
        runner.generate_dataset(GenerationConfig('budding_random',output=str(tmp_path/'bad.npy')))


def test_ko_mirror_honors_no_real_and_preserves_pair_metadata(tmp_path,monkeypatch):
    name,overrides=real_mutants.real_tasks()[0]
    real=row(Candidate(-1,name,overrides,0,True))
    synthetic=row(Candidate(1,'RANDOM_CLB2[ts]_L1_1',{'mdt':150.},1,pair_id=7,metadata={'tag':'keep'}))
    source=save_source(tmp_path/'source.npz',[real,synthetic])
    monkeypatch.setattr(derived,'run_simulations',lambda *args:iter([]))
    output=runner.generate_dataset(GenerationConfig('budding_ko_mirror',include_real=False,
                                   source=source,output=str(tmp_path/'mirror.npz')))
    archive=read_dataset(output)
    assert archive['sample_id'].tolist()==[1]
    assert archive['pair_id'].tolist()==[7]
    assert json.loads(str(archive['candidate_metadata_json'][0]))['tag']=='keep'


def test_targeted_ids_do_not_overlap_base_rows(tmp_path,monkeypatch):
    rows=[row(Candidate(i,f"SYNTH_{desc}_L{level}_{i}",{'mdt':150.},level,pair_id=i))
          for i,level in enumerate((1,2,2,3,3,3,3,3),1)
          for desc in ['+'.join(f'{gene}[ts]' for gene in ('CLB2','SIC1','CDC20')[:level])]]
    monkeypatch.setattr(derived,'_source_rows',lambda config:({},rows))
    monkeypatch.setattr(derived,'run_simulations',lambda candidates,*args:iter(map(row,candidates)))
    outputs=runner.generate_dataset(GenerationConfig('budding_bams_ablation',n_samples=5,include_real=False,
                                   source='unused.npz',output=str(tmp_path/'control.npz')))
    for name in ('hard','hypo'):
        archive=read_dataset(outputs[name])
        assert len(set(archive['sample_id']))==len(archive['sample_id'])
        assert len(set(archive['pair_id']))==len(archive['pair_id'])


def test_mirror_preserves_source_profile_provenance(tmp_path,monkeypatch):
    candidate=Candidate(1,'MASS_CLB2[ts]_L1_1',{'mdt':150.},1)
    source=tmp_path/'mass.npz'
    write_dataset(source,SPEC,[row(candidate)],GenerationConfig('budding_mass',mutation_profile='mass',distribution_profile='mass'),REPORT)
    monkeypatch.setattr(derived,'run_simulations',lambda *args:iter([]))
    path=runner.generate_dataset(GenerationConfig('budding_ko_mirror',source=str(source),output=str(tmp_path/'mirror.npz')))
    archive=read_dataset(path)
    assert str(archive['mutation_profile'])==str(archive['distribution_profile'])=='mass'


def test_targeted_ablation_distinguishes_source_and_mixed_profiles(tmp_path,monkeypatch):
    rows=[row(Candidate(i,f"SYNTH_{desc}_L{level}_{i}",{'mdt':150.},level,pair_id=i))
          for i,level in enumerate((1,2,2,3,3,3,3,3),1)
          for desc in ['+'.join(f'{gene}[ts]' for gene in ('CLB2','SIC1','CDC20')[:level])]]
    monkeypatch.setattr(derived,'_source_rows',lambda config:({'mutation_profile':np.array('mass'),'distribution_profile':np.array('mass')},rows))
    monkeypatch.setattr(derived,'run_simulations',lambda candidates,*args:iter(map(row,candidates)))
    outputs=runner.generate_dataset(GenerationConfig('budding_bams_ablation',n_samples=5,include_real=False,
                                   source='unused.npz',output=str(tmp_path/'control.npz')))
    assert str(read_dataset(outputs['control'])['mutation_profile'])=='mass'
    hard=read_dataset(outputs['hard'])
    assert str(hard['mutation_profile'])==str(hard['distribution_profile'])=='mixed'
    report=json.loads(str(hard['generation_report']))
    assert report['source_mutation_profile']=='mass'
