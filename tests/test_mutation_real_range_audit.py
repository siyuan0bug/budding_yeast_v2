"""Guard named-allele matching and effective (rather than proposal) controls."""
import numpy as np
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.compare_mutation_real_ranges import reference_rows,nominal,support
from budding_yeast_v2.data.systems.budding.distributions.ranges import EXACT_ZERO_TARGETS

def strengths(gene,mode,branch=''):
    return {r['strength'] for r in reference_rows() if (r['gene'],r['mode'],r['branch'])==(gene,mode,branch)}

def test_multi_copy_does_not_confuse_GAL_boost_with_native_MC():
    assert strengths('CLN2','MC')=={5.}
    assert strengths('CLB5','MC')=={4.}
    assert strengths('CDC14','MC')=={2.}
    assert strengths('CDC20','MC')=={25.}
    assert strengths('CLB2','MC')==set()
    rows=reference_rows()
    boosted=[r for r in rows if r['identity']=='50']
    assert len(boosted)==1 and boosted[0]['kind']=='GAL_with_8copy_boost'
    assert boosted[0]['strength']==.96

def test_timed_induction_and_residual_branches():
    assert strengths('SIC1','GAL-OE')=={.12,.24}
    assert not any(r['identity']=='104' and r['mode']=='GAL-OE' for r in reference_rows())
    assert strengths('CDC14','ts','synthesis')=={0.}
    assert np.allclose(sorted(strengths('CDC14','ts','activity')),[.1,.16,.3])
    assert strengths('SIC1','PM')==set()

def test_fixed_MAD2_alias_matches_real_effective_control():
    refs=[r for r in reference_rows() if (r['gene'],r['mode'])==('MAD2','CSL')]
    assert {r['identity'] for r in refs}=={'78','112','114'}

def test_effective_APC_and_CDC14_controls_include_internal_draws():
    a,_=nominal('APC_Core','APC-A','',size=32)
    s,_=nominal('CDC14','ts','synthesis',size=32)
    activity,_=nominal('CDC14','ts','activity',size=32)
    assert np.all(a==0)
    assert 0<s.min()<s.max()<.01
    assert .05<activity.min()<activity.max()<.35

def test_effective_supports_follow_approved_exact_zero_channels():
    for gene,mode in EXACT_ZERO_TARGETS:
        assert support(gene,mode)==(0.,0.)
        values,_=nominal(gene,mode,'',size=32)
        assert np.all(values==0)
