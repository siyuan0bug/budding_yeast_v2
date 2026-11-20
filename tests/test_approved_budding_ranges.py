import json
import numpy as np
import pytest
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
from budding_yeast_v2.data.systems.budding.real_mutants import is_duplicate_real_mutant
from budding_yeast_v2.data.systems.contracts import Candidate
from budding_yeast_v2.data.systems.sampling_policy import valid_candidate

CAPS=dict(CLN2=10,TEM1=10,CDC15=10,CLB5=10,CDC14=10,CDC6=15,BCK2=15,SIC1=20,CDC20=30,CLN3=8,CLB2=8)

@pytest.mark.parametrize('profile',['mutant_aware','mass'])
def test_every_gene_MC_domain_and_both_draw_interfaces(profile):
    b=BuddingSystem(profile,profile);rng=np.random.RandomState(52)
    for gene,cap in CAPS.items():
        random=np.array([b.sample_scaling_factor('MC',gene,rng=rng) for _ in range(256)])
        sobol=np.array([b.sample_scaling_factor_sobol('MC',gene,u) for u in np.linspace(0,1,257)])
        assert random.min()>=2-1e-12 and random.max()<=cap
        assert sobol.min()>=2-1e-12 and sobol.max()<=cap
        if profile=='mutant_aware':
            assert set(sobol)==set(range(2,cap+1))
            assert np.equal(random,np.round(random)).all()

@pytest.mark.parametrize('profile',['mutant_aware','mass'])
def test_approved_log_ranges_are_consistent_and_do_not_change_unapproved_GAL(profile):
    b=BuddingSystem(profile,profile);rng=np.random.RandomState(8)
    for gene,mode,low,high in [('CLB2','GAL-OE',.002,.6),('CLB2','db',.017/.4,.068/.4),
                              ('TEM1','ts',.001,.004),('ESP1','ts',.001,.004),('NET1','ts',.025,.1)]:
        median=b.sample_scaling_factor_sobol(mode,gene,.5)
        assert np.isclose(median,np.sqrt(low*high))
        for _ in range(10):
            alpha=b.sample_scaling_factor(mode,gene,rng=rng);ov={}
            b.apply_gene_mutation(gene,mode,alpha,b.defaults(),{},ov,rng=rng)
            assert low<=alpha<=high
            if mode=='db':assert .017<=ov['kdb2pp']<=.068 and ov['kdb2P']==0
            if gene=='TEM1':assert ov['Tem1T']==ov['init_Tem1']==0 and ov['ka15pp']==alpha
            if gene=='ESP1':assert np.isclose(ov['kasesp']/50,ov['kdiesp']/.5)
        assert low<=b.sample_scaling_factor_sobol(mode,gene,0)<=high
    # No unsupported adoption of the proposed SWI5/CDC20 GAL changes.
    assert np.isclose(b.sample_scaling_factor_sobol('GAL-OE','SWI5',.5),np.sqrt(.02*.1))
    assert np.isclose(b.sample_scaling_factor_sobol('GAL-OE','CDC20',.5),np.sqrt(.5*12))

@pytest.mark.parametrize('profile',['mutant_aware','mass'])
def test_GAL_MC_is_not_native_MC_and_preserves_real_collision(profile):
    b=BuddingSystem(profile,profile);ov={}
    b.apply_gene_mutation('CLB2','GAL-MC',8,b.defaults(),{},ov)
    assert ov=={'ksb2p':.96}
    assert is_duplicate_real_mutant(dict(ov,mdt=150.),b.defaults())
    native={};b.apply_gene_mutation('CLB2','MC',8,b.defaults(),{},native)
    assert native=={'ksb2p':.008,'ksb2pp':.32}
    row=valid_candidate(Candidate(1,'x',{'ksb2p':.84,'mdt':150.},1,
        metadata={'description':'CLB2[GAL-MC]'}),'budding')
    assert row.metadata['module_count']==1
    assert row.metadata['mutation_components'][0]['copy_number']==7
    assert valid_candidate(Candidate(1,'x',{'ksb2p':.84,'mdt':90.},1,
        metadata={'description':'CLB2[GAL-MC]'}),'budding') is None
    assert valid_candidate(Candidate(1,'x',{'ksb2p':.84,'mdt':150.},1,
        metadata={'description':'CLB2[GAL-MC+db]'}),'budding') is None

def test_GAL_MC_finite_capacity_copies_and_model_features_agree():
    from budding_yeast_v2.data.systems.budding.quotas import enumerate_combinations
    from budding_yeast_v2.data.systems.budding.al_pool import finite_options
    from budding_yeast_v2.utils.sampling.ours.response import _strength_curve
    rows,_=enumerate_combinations()
    cell=next(x for x in rows if x['alleles']=='CLB2[GAL-MC]')
    assert cell['finite_capacity_upper']==9
    supports=finite_options(cell,BuddingSystem())
    assert len(supports)==8
    assert all(o[1]['mdt']==150. for o in supports)
    assert not any(np.isclose(o[1]['ksb2p'],.96) for o in supports)
    _,values,_=_strength_curve('CLB2','GAL-MC','')
    np.testing.assert_allclose(values,.12*np.arange(2,11))
    _,values,_=_strength_curve('CDC14','MC','')
    np.testing.assert_allclose(values,.2*np.arange(2,11))

def test_old_pool_rejected_even_if_old_initial_still_exists(tmp_path):
    from budding_yeast_v2.data.systems.al_initial import pool_identity
    old=tmp_path/'old.npz';np.savez(old,system_id='budding',pool_generation_contract='old')
    with pytest.raises(ValueError,match='superseded'):pool_identity(old)
