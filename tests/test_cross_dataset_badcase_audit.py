"""Audit metadata compatibility and the explicitly defined diagnostic regions."""
from pathlib import Path
import numpy as np

from budding_yeast_v2.scripts.analysis.dataset_diagnosis.cross_dataset_badcases import feature_cache,mid
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.badcase_support import genes,nearest
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pool_badcase_coverage import box


def test_legacy_object_metadata_and_streamed_statistics(tmp_path):
    rng=np.random.default_rng(9);data=rng.uniform(0,2,size=(7,39,500)).astype('float32')
    data[4,1,0]=-1e-7
    names=np.array(['1_WT','2.1_Gal','2.2_Raff']+[f'RANDOM_CLN2[MC]_L1_{i}' for i in range(4)],dtype=object)
    path=tmp_path/'source.npz'
    np.savez_compressed(path,data=data,params=rng.uniform(size=(7,141)),mutant_names=names,
                        pattern_labels=np.array(['Pattern_A']*7,dtype=object))
    cache=feature_cache((str(path),str(tmp_path)))
    with np.load(cache,allow_pickle=False) as saved:
        transformed=np.log1p(np.maximum(data[3:],0)).astype('float64')
        np.testing.assert_allclose(saved['global_y_mean'],transformed.mean((0,2)),rtol=1e-6)
        np.testing.assert_allclose(saved['global_y_std'],transformed.std((0,2),ddof=1)+1e-6,rtol=1e-6)
        np.testing.assert_array_equal(saved['ic'],data[:,:,0])
        assert saved['real_y'].shape==(3,39,500)
        assert saved['names'].dtype.kind=='U'
    assert mid('2.1_WT_on_Gal')=='2.1'


def test_targeted_skeleton_names_are_not_counted_as_zero_modules():
    assert genes('BAMS_TARGET_HARD_991_M15_mc')=={'CLN3','BCK2','CLN2'}
    assert genes('BAMS_TARGET_HYPO_12_M42')=={'SIC1','CDC6','CDH1','CDC20'}
    assert genes('MASS_RAND_APC_Core[APC-A]+CLB2[GAL-OE]_L2_12')=={'APC_Core','CLB2'}


def test_nearzero_box_does_not_confuse_exact_endpoint_absence_with_neighborhood_absence():
    p=np.array([[0.,1.,90.],[.005,1.1,90.],[.05,1.,90.],[.001,1.,150.]])
    query=np.array([0.,1.,90.]);wt=np.array([1.,1.,90.])
    np.testing.assert_array_equal(box(p,query,wt,2,.25),[True,False,False,False])
    np.testing.assert_array_equal(box(p,query,wt,2,.25,.01),[True,True,False,False])
    distances,indices=nearest(np.array([[0.,0.],[3.,4.]]),np.array([[0.,0.]]),k=2)
    np.testing.assert_array_equal(indices,[[0,1]]);np.testing.assert_allclose(distances,[[0.,5.]])


def test_historical_integer_only_regex_keeps_decimal_benchmarks_out_of_real_test(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from budding_yeast_v2.scripts.analysis.dataset_diagnosis.badcase_support import get_split
    (tmp_path/'config.json').write_text(json.dumps({'git_commit':'legacy'}))
    monkeypatch.setattr('subprocess.run',lambda *a,**kw:SimpleNamespace(stdout="return bool(re.match(r'^\\d+_',s))"))
    cache=dict(names=np.array(['1_WT','2.1_Gal','2.2_Raff','3_del','SYNTH_a','SYNTH_b','SYNTH_c','SYNTH_d']),patterns=np.array(['Pattern_A']*8))
    split,provenance,_=get_split(SimpleNamespace(checkpoint='',run=tmp_path,seed=42),cache)
    assert provenance=='git_confirmed_legacy_integer_prefix'
    np.testing.assert_array_equal(split['real'],[0,3])


def test_probe_exports_complete_model_input_including_initial_features():
    from budding_yeast_v2.scripts.analysis.dataset_diagnosis.compare_good_bad_cases import probe
    from budding_yeast_v2.data.systems.budding.api import PARAM_KEYS
    _,_,parameters,initial,trajectory=probe(('cln3KO',{'C0':0.}))
    assert set(PARAM_KEYS).issubset(parameters)
    assert parameters['C0']==0.
    np.testing.assert_allclose([parameters[k] for k in ['init_CDH1T','init_CDH1','init_BUB2']],initial[[21,22,38]])
    np.testing.assert_allclose(trajectory[:,0],initial,rtol=1e-6,atol=1e-7)


def test_same_modes_comparison_is_order_independent():
    from budding_yeast_v2.scripts.analysis.dataset_diagnosis.compare_good_bad_cases import decode_pairs
    assert decode_pairs('MASS_RAND_CLN2[GAL-OE]+CDH1[KO]_L2_554109')==decode_pairs('MASS_RAND_CDH1[KO]+CLN2[GAL-OE]_L2_1')
    assert decode_pairs('MASS_RAND_CLN3[KO]+BCK2[KO]+CLN2[MC]_L3_1')==( ('BCK2','KO'),('CLN2','MC'),('CLN3','KO'))
