"""ODE relation/module repairs: no dataset access, training or real-mutant inputs."""
import copy
from hashlib import sha256
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from budding_yeast_v2.data.yeast_graph import build_yeast_graph
from budding_yeast_v2.data.yeast_relations import build_yeast_relations, RELATION_NAMES
from budding_yeast_v2.data.build_bio_multirelational_graphs import build_relational_graphs
from budding_yeast_v2.models.ours.graph_contracts import set_graph_load_policy, fixed_graph_report
from budding_yeast_v2.models.ours.bio_hetero_causal_graph import MultiRelationalGraphConvBlock
from budding_yeast_v2.models.ours.hano import MultiRelGraphAttention
from budding_yeast_v2.models.ours.bio_refiner_plus import DynamicReactionFluxGraphAttention, StoichiometricGraphAttention
from budding_yeast_v2.models.ours.bio_refiner_dynamic_plus import TimeVaryingDynamicAdapter, SoftTopologyGAT
from budding_yeast_v2.models.ours.bio_refiner_causal import MODULE_NAMES, MODULE_REGULATORY_EDGES
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule, RELATION_GRAPH_MODELS


@pytest.fixture(scope='module', autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('view,edges', [('continuous', 218), ('hybrid', 226)])
def test_complete_exact_support_no_fabricated_or_reversed_edges(view, edges):
    source = build_yeast_relations()
    rel, untyped = source.matrices(view, self_loops='none')
    expected = source.graph.adjacency(view, self_loops='none')
    np.testing.assert_array_equal(rel.any(0) | untyped, expected)
    assert int(expected.sum()) == edges
    assert not (rel.any(0) & untyped).any()
    assert source.summary(view)['typed_union_nonself'] == 217
    idx = {n: i for i, n in enumerate(source.graph.variable_names)}
    assert not rel[:, idx['CDC20T'], idx['CDH1']].any()
    assert not rel[:, idx['SWI5T'], idx['CLB5']].any()
    assert rel[5, idx['CDC20'], idx['IEP']]
    assert rel[5, idx['CDC20'], idx['CDC20T']]
    assert rel[1, idx['IEP'], idx['CLB2']]
    assert not rel[:, idx['CLB2'], idx['IEP']].any()


def test_fresh_matrices_self_loops_provenance_and_untyped_semantics():
    source = build_yeast_relations()
    rel, untyped = source.matrices()
    assert not rel[:, np.arange(39), np.arange(39)].any()
    assert untyped.diagonal().all()
    assert len(source.fingerprint) == 64 and source.fingerprint != source.graph.source_sha256
    idx = {n: i for i, n in enumerate(source.graph.variable_names)}
    assert untyped[idx['CDC20'], idx['Vi20']]  # No forced biochemical tag.
    assert untyped[idx['Vi20'], idx['SPN']]  # Event, not a continuous reaction.
    rel[:] = False
    untyped[:] = False
    assert source.matrices()[0].any() and source.matrices()[1].any()
    assert all(record.expression and record.line > 0 for record in source.records)


TOY = '''
raise RuntimeError('must not execute')
(nA, nB) = range(2)
REAL_MUTANTS = {'hidden': 123}
def eqns(t, y, p):
    dydt = np.zeros(2)
    (A, B) = y
    dydt[0] = p['unknown_rate'] * B
    dydt[1] = p['ks20p'] * A - p['kd20'] * B
    return dydt
def event1(t, y, p): return y[nA]-1
event1.direction = 1
event1.terminal = True
def simulate_mutant(name, ovr, tf=210, ns=500):
    sol = solve_ivp(eqns, (0,tf), y0, args=(p,), events=[event1])
    ie = next((i+1 for i, te in enumerate(sol.t_events) if len(te)>0), -1)
    if ie == 1:
        y0[nB] = 0
'''


def test_unknown_and_ambiguous_flux_terms_stay_untyped_source_not_executed(tmp_path):
    path = tmp_path / 'toy.py'
    path.write_text(TOY)
    first = build_yeast_relations(path)
    rel, untyped = first.matrices('continuous', self_loops='none')
    assert untyped[0, 1] and rel[3, 1, 0]
    path.write_text(TOY.replace("p['unknown_rate'] * B", "p['ks20p'] * p['kd20'] * B"))
    ambiguous = build_yeast_relations(path)
    assert ambiguous.matrices('continuous', self_loops='none')[1][0, 1]
    assert first.fingerprint != ambiguous.fingerprint
    path.write_text(TOY.replace("'hidden': 123", "'hidden': 999"))
    assert first.fingerprint == build_yeast_relations(path).fingerprint


@pytest.mark.parametrize('name', sorted(RELATION_GRAPH_MODELS))
@pytest.mark.parametrize('view,edges', [('continuous', 218), ('hybrid', 226)])
def test_all_registry_consumers_route_complete_support_and_finite_backward(name, view, edges):
    lit = YeastLitModule(model_name=name, width=16, modes=2, n_blocks=1,
                         loss_type='mse_only', graph_view=view)
    reports = lit.graph_report()['relation_graphs']
    assert reports and all(r['total_nonself'] == edges for r in reports)
    assert all(r['orientation'] == 'target_source' for r in reports)
    assert not lit.graph_report()['legacy_six_relation_branch']
    if name == 'hetero_graph_channel_hyper_fno':
        assert not reports[0]['used_in_forward']
    lit.set_parameter_stats(norm_type='physical')
    ic = torch.full((1, 39, 32), .1)
    time = torch.linspace(0, 1, 32)[None, None].expand_as(ic)
    output = lit(torch.stack((ic, time), 2), torch.ones(1, 141))
    assert output.shape == (1, 39, 32) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in lit.parameters() if p.grad is not None)


@pytest.mark.parametrize('name', ['bio_hetero_causal_graph', 'hetero_graph_resnet_film',
                                 'dyrfo', 'hano', 'biom_refiner_graph_flux'])
@pytest.mark.parametrize('policy', ['current', 'checkpoint'])
def test_old_six_channel_checkpoint_and_learned_weights(name, policy):
    lit = YeastLitModule(model_name=name, width=16, modes=2, n_blocks=1, loss_type='mse_only')
    state = copy.deepcopy(lit.state_dict())
    old = torch.tensor(build_relational_graphs())  # As saved, source/target.
    if name == 'hano':
        old = old / old.sum(-1, keepdim=True).clamp(min=1.0)
    for key in list(state):
        if key.endswith('untyped_graph') or key.endswith('_relations_graph_metadata'):
            del state[key]
        elif key.endswith('rel_graphs'):
            state[key] = old.clone()
    target = YeastLitModule(model_name=name, width=16, modes=2, n_blocks=1,
                            loss_type='mse_only', graph_load_policy=policy)
    target.load_state_dict(state, strict=True)
    for key, parameter in target.named_parameters():
        assert torch.equal(parameter, state[key])
    for module in target.modules():
        if hasattr(module, 'untyped_graph'):
            if policy == 'checkpoint':
                assert torch.equal(module.rel_graphs, old) and not module.untyped_graph.any()
            else:
                assert (module.rel_graphs.ne(0).any(0) | module.untyped_graph.ne(0)).sum() == 265
    assert target.graph_report()['legacy_six_relation_branch'] == (policy == 'checkpoint')


@pytest.mark.parametrize('policy', ['current', 'checkpoint'])
def test_module_direction_and_saved_buffer_migration(policy):
    adapter = TimeVaryingDynamicAdapter(d_gene=4)
    idx = {n: i for i, n in enumerate(MODULE_NAMES)}
    for src, dst, sign in MODULE_REGULATORY_EDGES:
        assert adapter.adj_mask[idx[dst], idx[src]] == 1
        assert adapter.sign_adj[idx[dst], idx[src]] == (1 if sign == '+' else -1)
    state = copy.deepcopy(adapter.state_dict())
    state['adj_mask'] = state['adj_mask'].T.contiguous()
    state['sign_adj'] = state['sign_adj'].T.contiguous()
    del state['_module_graph_metadata']
    target = TimeVaryingDynamicAdapter(d_gene=4)
    set_graph_load_policy(target, policy)
    target.load_state_dict(state, strict=True)
    expected = adapter.adj_mask if policy == 'current' else state['adj_mask']
    assert torch.equal(target.adj_mask, expected)
    for key, parameter in target.named_parameters():
        assert torch.equal(parameter, state[key])
    assert fixed_graph_report(target)[0]['orientation'] == ('target_source' if policy == 'current' else 'legacy_as_saved')


def test_gcn_one_way_edge_and_untyped_route_not_reversed():
    block = MultiRelationalGraphConvBlock(width=3, num_vars=3, d_var=1)
    with torch.no_grad():
        block.to_vars.weight.copy_(torch.eye(3))
        block.to_vars.bias.zero_()
        block.from_vars.weight.copy_(torch.eye(3))
        block.from_vars.bias.zero_()
        block.rel_weights.fill_(1)
    block.norm = torch.nn.Identity()
    block.act = torch.nn.Identity()
    x = torch.zeros(1, 3, 1)
    x[:, 0] = 1
    rel = torch.zeros(6, 3, 3)
    rel[0, 1, 0] = 1
    untyped = torch.zeros(3, 3)
    untyped[2, 0] = 1
    actual = block(x, rel, torch.ones(1, 6), untyped)
    assert torch.equal(actual.flatten(), torch.tensor([1., 1., 1.]))


@pytest.mark.parametrize('factory', [StoichiometricGraphAttention, MultiRelGraphAttention])
def test_attention_still_allows_communication_without_a_direct_edge(factory):
    kwargs = {'adj_matrix': np.zeros((3, 3))} if factory is StoichiometricGraphAttention else {
        'rel_graphs': np.zeros((6, 3, 3))}
    layer = factory(channels=3, num_vars=3, head_dim=1, **kwargs)
    with torch.no_grad():
        layer.to_vars.weight.copy_(torch.eye(3))
        layer.to_vars.bias.zero_()
        layer.from_vars.weight.copy_(torch.eye(3))
        layer.from_vars.bias.zero_()
        for projection in (layer.q_proj, layer.k_proj):
            projection.weight.zero_()
            projection.bias.zero_()
        layer.v_proj.weight.fill_(1)
        layer.v_proj.bias.zero_()
    x = torch.zeros(1, 3, 1, requires_grad=True)
    gradient = torch.autograd.grad(layer(x)[0, 1, 0], x)[0]
    assert gradient[0, 0, 0] > 0  # No hard mask was introduced.


def test_dynamic_module_soft_attention_bias_direction_and_no_hard_mask():
    layer = SoftTopologyGAT(d_gene=4)
    adapter = TimeVaryingDynamicAdapter(d_gene=4)
    with torch.no_grad():
        layer.q_proj.weight.zero_()
        layer.k_proj.weight.zero_()
    captured = []
    def softmax(value, dim):
        captured.append(value.detach().clone())
        return torch.softmax(value, dim=dim)
    with patch('torch.nn.functional.softmax', side_effect=softmax):
        layer(torch.randn(1, 27, 4, 2), adapter.adj_mask, adapter.sign_adj)
    idx = {n: i for i, n in enumerate(MODULE_NAMES)}
    scores = captured[0]
    assert scores[0, 0, idx['SBF_MBF'], idx['CLN3']] > 0
    assert scores[0, 0, idx['CLN3'], idx['SBF_MBF']] == 0
    assert torch.isfinite(scores).all() and (torch.softmax(scores, -1) > 0).all()


@pytest.mark.parametrize('factory', [MultiRelGraphAttention, DynamicReactionFluxGraphAttention])
def test_legacy_attention_exact_replay_formula_and_new_roundtrip(factory):
    layer = factory(channels=8, head_dim=2)
    old = torch.tensor(build_relational_graphs())
    if factory is MultiRelGraphAttention:
        old = old / old.sum(-1, keepdim=True).clamp(min=1.)
    state = copy.deepcopy(layer.state_dict())
    del state['_relations_graph_metadata']
    del state['untyped_graph']
    state['rel_graphs'] = old
    set_graph_load_policy(layer, 'checkpoint')
    layer.load_state_dict(state, strict=True)
    h = torch.randn(2, 8, 5)
    flux = torch.rand(2, 6)
    variables = layer.to_vars(h.permute(0, 2, 1)).reshape(2, 5, 39, 2)
    q, k, v = layer.q_proj(variables), layer.k_proj(variables), layer.v_proj(variables)
    scores = q @ k.transpose(-1, -2) / 2**.5
    if factory is MultiRelGraphAttention:
        weights = torch.softmax(layer.rel_weights, 0)
        bias = torch.einsum('r,rij->ij', weights, old)
        expected_attention = torch.softmax(scores + bias[None, None], -1)
        actual = layer(h)
    else:
        adjacency = torch.einsum('br,rij->bij', flux, old) + torch.eye(39)[None]
        normalized = adjacency / adjacency.sum(-1, keepdim=True).clamp(min=1.)
        expected_attention = torch.softmax(scores + .5 * normalized[:, None], -1)
        actual = layer(h, flux)
    expected = h + layer.from_vars((expected_attention @ v).reshape(2, 5, 78)).permute(0, 2, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored = factory(channels=8, head_dim=2)
    set_graph_load_policy(restored, 'checkpoint')
    restored.load_state_dict(layer.state_dict(), strict=True)
    assert fixed_graph_report(restored)[0]['origin'] == 'legacy_unknown'


def test_malformed_relation_state_fails_even_non_strict():
    layer = MultiRelGraphAttention(channels=8)
    for key, invalid in [('_relations_graph_metadata', torch.zeros(2)),
                         ('untyped_graph', torch.full((39, 39), float('nan')))]:
        state = copy.deepcopy(layer.state_dict())
        state[key] = invalid
        with pytest.raises(RuntimeError):
            layer.load_state_dict(state, strict=False)
    state = copy.deepcopy(layer.state_dict())
    del state['rel_graphs']
    with pytest.raises(RuntimeError, match='Incomplete'):
        layer.load_state_dict(state, strict=False)


def test_old_binary_assets_are_only_metadata_not_topology(tmp_path):
    from budding_yeast_v2.models.ours.bio_hetero_causal_graph import load_multirelational_graphs
    asset = ROOT / 'data/yeast_multirelational_graphs.npz'
    before = sha256(asset.read_bytes()).hexdigest()
    # With NPZ absent, named circuit/module builders are used, not identity graphs.
    with patch('budding_yeast_v2.models.ours.bio_hetero_causal_graph.os.path.exists', return_value=False):
        module = torch.nn.Module()
        load_multirelational_graphs().register(module)
    assert fixed_graph_report(module)[0]['total_nonself'] == 226
    assert sha256(asset.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('name', ['bio_hetero_causal_graph', 'hano', 'biom_refiner_graph_flux',
                                 'biom_refiner_dynamic_plus'])
def test_new_checkpoint_roundtrip_retains_actual_contracts(name):
    source = YeastLitModule(model_name=name, width=16, modes=2, n_blocks=1,
                            loss_type='mse_only', graph_view='continuous')
    target = YeastLitModule(model_name=name, width=16, modes=2, n_blocks=1,
                            loss_type='mse_only', graph_load_policy='checkpoint')
    target.load_state_dict(source.state_dict(), strict=True)
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])
    report = target.graph_report()
    assert not report['legacy_six_relation_branch']
    assert all(r['view'] == 'continuous' and r['total_nonself'] == 218 for r in report['relation_graphs'])
    if name == 'biom_refiner_dynamic_plus':
        assert report['module_graph_migrated'] and report['module_graphs'][0]['orientation'] == 'target_source'


def test_audit_cli_and_training_tags_are_distinct(tmp_path):
    from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_yeast_graph import main
    from budding_yeast_v2.utils.experiments.paths import generate_hparams_signature
    summary = main(['--relations', '--view', 'continuous'])
    assert summary['relations']['total_nonself'] == 218
    assert summary['relations']['typed_union_nonself'] + summary['relations']['untyped_nonself'] == 218
    kwargs = dict(width=16, modes=2, batch_size=4, lr=.001, graph_view='hybrid')
    previous = generate_hparams_signature(**kwargs)
    typed = generate_hparams_signature(**kwargs, relation_graph_version=1)
    module = generate_hparams_signature(**kwargs, module_graph_version=1)
    assert len({previous, typed, module}) == 3
    assert 'rterms1' in typed and 'mdir1' in module


@pytest.mark.parametrize('factory', [MultiRelGraphAttention, DynamicReactionFluxGraphAttention])
def test_cpu_autocast_and_double_graph_forward_backward(factory):
    module = factory(channels=8, head_dim=2)
    h = torch.randn(2, 8, 8, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = module(h) if factory is MultiRelGraphAttention else module(h, torch.ones(2, 6))
    output.square().mean().backward()
    assert torch.isfinite(output).all() and torch.isfinite(h.grad).all()
    module.double()
    h = h.detach().double().requires_grad_()
    output = module(h) if factory is MultiRelGraphAttention else module(h, torch.ones(2, 6).double())
    output.square().mean().backward()
    assert output.dtype == torch.float64 and torch.isfinite(h.grad).all()
