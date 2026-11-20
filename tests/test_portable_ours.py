"""Mechanism semantics, both canonical systems, oracle isolation and workflows."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from budding_yeast_v2.utils.sampling.contracts import PoolInputs, QueriedTraining, SamplingContext
from budding_yeast_v2.utils.sampling.registry import select
from budding_yeast_v2.utils.sampling.ours.mechanistic.systems import BiologicalODE
from budding_yeast_v2.utils.sampling.ours.mechanistic.transport import estimate, jump_matrix, jacobians, TransportConfig
from budding_yeast_v2.utils.sampling.ours.mechanistic.actions import rectangles, choose_faces


@pytest.mark.parametrize('replication',[.5, 1.])
def test_fission_rhs_matches_native_with_dynamic_mode_and_mass_units(replication):
    from budding_yeast_v2.data.systems.fission.parameters import WT_parameters, param_vector
    from budding_yeast_v2.data.systems.fission.ode import eqns
    system = BiologicalODE('fission')
    p = WT_parameters()
    p['kS'] = p['kSmax'] if replication < 1 else p['kSmin']
    raw = np.array([2., replication, .15, .3, .1, .1, .1, .1, .3, .3, .3, .3, .1, .1, .1, .2])
    output = raw.copy()
    output[0] /= system.mass_unit
    actual = system.rhs(torch.tensor(output)[None], torch.tensor(param_vector(p))[None], .5)[0].numpy()
    expected = eqns(.5, raw, p)
    expected[0] /= system.mass_unit
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(system.initials(raw), output)


class LinearSystem:
    n, horizon = 1, 1.
    def initials(self, y): return np.asarray(y)
    def rhs(self, y, p, t, metadata=()): return y * p[:, :1]
    def guards(self, y, p): return (y[..., :1] - 100.), (1,)
    def event(self, a, b, p): return None, 0


def test_transport_identifies_error_and_is_stable_for_stiff_decay():
    system = LinearSystem()
    score, _, _, details = estimate(system, np.ones((1, 100)), [-100.], [0.], [1.])
    assert score == pytest.approx(1., abs=1e-10)
    exact = np.exp(-np.linspace(0, 1, 1000))[None]
    good, _, _, _ = estimate(system, exact, [-1.], [1.], [1.], config=TransportConfig(grid_points=1000))
    assert good < .001 and not details['certified_error_bound']


def test_response_includes_initial_shift_and_matches_linear_closed_form():
    system = LinearSystem()
    _, response, indices, _ = estimate(system, np.ones((1, 400)), [.2], [1.1], [1.],
        reference_params=[0.], reference_initial=[1.], config=TransportConfig(grid_points=400))
    expected = 1.1 * np.exp(.2 * np.linspace(0, 1, 400)[indices]) - 1.
    np.testing.assert_allclose(response[0], expected, atol=.001)


def test_saltation_captures_time_shift_which_reset_jacobian_misses():
    class Jump(LinearSystem):
        def rhs(self, y, p, t, metadata=()): return torch.ones_like(y)
        def guards(self, y, p): return y - 1., (1,)
        def reset(self, y, p, event, metadata=None): return y * 0.
    matrix, transverse = jump_matrix(Jump(), torch.tensor([1.], dtype=torch.float64),
        torch.tensor([0.], dtype=torch.float64), 0, .5, {}, TransportConfig())
    assert transverse
    assert float(matrix[0, 0]) == pytest.approx(1., abs=1e-7)


@pytest.mark.parametrize('name',['budding','fission'])
def test_batched_jacobians_equal_independent_directional_finite_differences(name):
    system = BiologicalODE(name)
    if name == 'budding':
        inputs = budding_inputs([2.])
        p = torch.tensor(inputs.params[0])
        state = torch.tensor(inputs.initial[0])
    else:
        from budding_yeast_v2.data.systems.fission.parameters import WT_parameters, param_vector
        p = torch.tensor(param_vector(WT_parameters()))
        state = torch.tensor([1.5,.5,.1,.3,.1,.1,.1,.1,.3,.3,.3,.3,.1,.1,.1,.2], dtype=torch.float64)
    states = torch.stack((state, state * 1.01))
    matrices = jacobians(system, states, p, torch.tensor([0., 1.]), {}, 1e-5)
    for j in (0, 1, system.n - 1):
        eps = 1e-5 * max(1., abs(float(state[j])))
        shift = torch.zeros_like(state); shift[j] = eps
        plus = system.rhs((state + shift)[None], p[None], 0., ({},))[0]
        minus = system.rhs((state - shift)[None], p[None], 0., ({},))[0]
        torch.testing.assert_close(matrices[0,:,j], (plus - minus) / (2 * eps), rtol=1e-9, atol=1e-9)


def budding_inputs(values, offset=0, gene='CLN3', mode='MC'):
    from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
    b = BuddingSystem()
    rows, initials, metas = [], [], []
    for value in values:
        overrides = {}
        b.apply_gene_mutation(gene, mode, value, b.defaults(), b.defaults().copy(), overrides,
                              rng=np.random.default_rng(42))
        p, y0, _, _ = b.prepare_initial_conditions(overrides)
        rows.append([overrides.get(k, b.defaults()[k]) for k in b.PARAM_KEYS])
        initials.append(y0)
        metas.append(dict(pairs=[[gene, mode]], _effective_overrides=overrides, pattern_group_ids=999))
    return PoolInputs(np.arange(offset, offset + len(values)), np.asarray(rows), np.asarray(initials), tuple(metas))


class PhysicalPredictor:
    device = 'cpu'
    steps = 9
    def __init__(self, system_id): self.system_id = system_id
    def batches(self, inputs, physical=False):
        assert physical
        y0 = BiologicalODE(self.system_id).initials(inputs.initial)
        y = np.repeat(y0[:, :, None], self.steps, 2)
        y[:, 1] += np.linspace(0, .002, self.steps)
        for start in range(0, len(y), 2):
            yield start, torch.tensor(y[start:start + 2], dtype=torch.float32)


def complete_test_pool(root):
    import json
    from test_sampling_benchmark import make_pool
    from budding_yeast_v2.data.dataset_archive import write_dataset_manifest
    from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
    path, arrays = make_pool(root, mode='MC')
    arrays['overrides_json'] = np.array([json.dumps(dict(zip(PARAM_KEYS, p))) for p in arrays['params']])
    chunk = root/'complete.npz'
    np.savez_compressed(chunk, **arrays)
    write_dataset_manifest(path, arrays, [(chunk, None)])
    return path


@pytest.mark.parametrize('method',['ours_mechanism_witness','ours_dynamics_transport','ours_intervention_closure'])
def test_new_selectors_ignore_pattern_and_reject_candidate_label_reader(method):
    pool, train = budding_inputs([2., 4.]), budding_inputs([3.], offset=20)
    predictor = PhysicalPredictor('budding')
    truth = np.repeat(train.initial[:, :, None], predictor.steps, 2)
    context = SamplingContext(pool, QueriedTraining(train, truth), predictor, 42, 'cpu')
    a = select(method, context, 1)
    changed = tuple(dict(m, pattern_group_ids=-7, pattern_labels='poison') for m in pool.metadata)
    b = select(method, replace(context, pool=replace(pool, metadata=changed)), 1)
    np.testing.assert_array_equal(a.ids, b.ids)
    assert a.diagnostics['pattern_labels_used'] is False
    assert not a.diagnostics['candidate_truth_used']
    with pytest.raises(PermissionError):
        select(method, replace(context, label_reader=lambda _: (_ for _ in ()).throw(AssertionError())), 1)


def test_exact_tem1_certificate_is_conditional_and_uses_queried_anchor():
    pool, train = budding_inputs([.001], gene='TEM1', mode='ts'), budding_inputs([.003], 10, 'TEM1', 'ts')
    system = BiologicalODE('budding')
    a = system.equivalence_key(pool.params[0], pool.initial[0], pool.metadata[0])
    b = system.equivalence_key(train.params[0], train.initial[0], train.metadata[0])
    assert a is not None and a == b
    initial = pool.initial[0].copy()
    initial[28] = .001
    assert system.equivalence_key(pool.params[0], initial) is None
    predictor = PhysicalPredictor('budding')
    truth = np.repeat(train.initial[:, :, None], predictor.steps, 2)
    result = select('ours_mechanism_witness', SamplingContext(pool, QueriedTraining(train, truth),
                    predictor, 42, 'cpu'), 1)
    assert result.diagnostics['certified_equivalence_rows'] == 1


def test_rectangles_require_exact_other_inputs_and_charge_only_missing_corner():
    from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
    base = budding_inputs([2., 2., 4., 4.])
    p = base.params.copy()
    p[:, PARAM_KEYS.index('B0')] = [.01, .02, .01, .02]
    meta = tuple(dict(pairs=[['CLN3','MC'], ['BCK2','MC']]) for _ in range(4))
    combined = PoolInputs(np.arange(4), p, base.initial, meta)
    faces = rectangles(combined, BiologicalODE('budding'), 1)
    assert faces.shape == (1, 4) and set(faces[0]) == set(range(4))
    features = np.zeros((4, 3)); features[0] = 1.
    selected, details = choose_faces(faces, features, 1, 1, [1.], 42)
    assert selected.tolist() == [0] and details['face_query_endpoints'] == 1
    p = p.copy(); p[0, PARAM_KEYS.index('mdt')] += 1.
    assert len(rectangles(replace(combined, params=p), BiologicalODE('budding'), 1)) == 0


def test_closure_audit_maps_local_corners_to_stable_pool_and_queried_ids():
    from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
    base = budding_inputs([2., 2., 4., 4.])
    p = base.params.copy()
    p[:, PARAM_KEYS.index('B0')] = [.108, .216, .108, .216]
    meta = tuple(dict(pairs=[['CLN3','MC'], ['BCK2','MC']]) for _ in range(4))
    pool = PoolInputs(np.array([7]), p[:1], base.initial[:1], meta[:1])
    train = PoolInputs(np.array([20,21,22]), p[1:], base.initial[1:], meta[1:])
    predictor = PhysicalPredictor('budding')
    truth = np.repeat(train.initial[:,:,None], predictor.steps, 2)
    context = SamplingContext(pool, QueriedTraining(train, truth), predictor, 42, 'cpu')
    result = select('ours_intervention_closure', context, 1)
    record = result.diagnostics['completed_faces'][0]
    assert {e['candidate_id'] for e in record['endpoints']} == {7,20,21,22}
    assert sum(e['kind']=='queried_train' for e in record['endpoints']) == 3
    assert record['new_candidate_ids'] == [7]


def test_fission_acquisition_core_is_not_hardcoded_to_budding_dimensions():
    from budding_yeast_v2.data.systems.fission.parameters import WT_parameters, param_vector
    p = param_vector(WT_parameters())
    y = np.array([2., .5, .15, .3, .1, .1, .1, .1, .3, .3, .3, .3, .1, .1, .1, .2])
    pool = PoolInputs(np.array([0]), p[None], y[None])
    train = PoolInputs(np.array([20]), p[None], y[None])
    predictor = PhysicalPredictor('fission')
    truth = np.repeat(BiologicalODE('fission').initials(y)[None, :, None], predictor.steps, 2)
    context = SamplingContext(pool, QueriedTraining(train, truth), predictor, 42, 'cpu', system_id='fission')
    for method in ('ours_mechanism_witness', 'ours_dynamics_transport'):
        result = select(method, context, 1)
        assert result.ids.tolist() == [0] and result.diagnostics['system'] == 'fission'


def test_controlled_bank_is_canonical_reproducible_and_resumable(tmp_path):
    from budding_yeast_v2.utils.sampling.generation.controlled import controlled_bank
    from budding_yeast_v2.utils.sampling.generation.domain import FrozenBuddingDomain
    a = FrozenBuddingDomain('mutation', 42, tmp_path/'a')
    candidates, signatures, view = controlled_bank(a, 8)
    b = FrozenBuddingDomain('mutation', 42, tmp_path/'b')
    other, other_signatures, repeated = controlled_bank(b, 8)
    assert signatures == other_signatures and len(set(signatures)) == 8
    np.testing.assert_array_equal(view.params, repeated.params)
    assert len(rectangles(view, BiologicalODE('budding'), 8)) >= 2
    for i, candidate in enumerate(candidates):
        _, expected, _, _ = a.proposals.backend.prepare_initial_conditions(candidate.overrides)
        np.testing.assert_array_equal(view.initial[i], expected)
        assert not candidate.metadata['real_collision']
        assert isinstance(candidate.metadata['source_base_candidate_id'],int)
        assert not a.proposals.collides(__import__('budding_yeast_v2.data.systems.budding.real_mutants',
            fromlist=['effective_inputs']).effective_inputs(candidate.overrides, a.proposals.defaults))
    a.consume(signatures[0])
    state = a.state()
    resumed = FrozenBuddingDomain('mutation', 42, tmp_path/'resumed', seen=state['seen'], next_index=state['next_index'])
    assert controlled_bank(a, 4)[1] == controlled_bank(resumed, 4)[1]


@pytest.mark.parametrize('method',['ours_mechanism_witness','ours_dynamics_transport','ours_intervention_closure'])
def test_two_round_complete_pool_training_acquisition_evaluation_and_resume(tmp_path, method):
    import json
    from test_sampling_benchmark import make_pool, config
    from budding_yeast_v2.utils.sampling.benchmark.protocol import prepare_protocol
    from budding_yeast_v2.utils.sampling.benchmark.runner import run_experiment
    path = complete_test_pool(tmp_path/'source')
    cfg = replace(config(), budgets=(2,4), epochs=(1,1), warmup=(0,0),
                  width=4, modes=2, n_blocks=1, batch_size=2)
    protocol = prepare_protocol(dict(mutation=path, lhs=path), tmp_path/'protocol', cfg)
    run = run_experiment(protocol, 'mutation', method, tmp_path/'runs', diagnostic_only=True)
    assert run_experiment(protocol, 'mutation', method, tmp_path/'runs', diagnostic_only=True) == run
    state = json.loads((run/'_state/state.json').read_text())
    assert state['completion']['unique_training_inputs'] == 4
    assert (run/'rounds/round_01/eval/global_metrics_denorm.json').exists()
    assert not list(run.glob('rounds/*/dataset.npz'))


@pytest.mark.parametrize('method',['ours_mechanism_witness','ours_dynamics_transport','ours_intervention_closure'])
def test_generated_ours_queries_canonical_ode_and_resume_without_duplicate_labels(tmp_path, method):
    import json
    from test_sampling_benchmark import make_pool, config
    from budding_yeast_v2.utils.sampling.benchmark.protocol import prepare_protocol
    from budding_yeast_v2.utils.sampling.generation.runner import run_generation
    path = complete_test_pool(tmp_path/'source')
    cfg = replace(config(), budgets=(2,4), epochs=(1,1), warmup=(0,0),
                  width=4, modes=2, n_blocks=1, batch_size=2)
    protocol = prepare_protocol(dict(mutation=path, lhs=path), tmp_path/'protocol', cfg)
    kwargs = dict(workers=1, bank_size=4, query_batch_size=2, diagnostic_only=True)
    run = run_generation(protocol, 'mutation', method, tmp_path/'runs', **kwargs)
    assert run_generation(protocol, 'mutation', method, tmp_path/'runs', **kwargs) == run
    state = json.loads((run/'_state/completion.json').read_text())
    assert state['accepted'] == 4 and state['newly_accepted'] == 2
    assert state['attempted_inputs'] >= 2
    assert (run/'eval/global_metrics_denorm.json').exists()
