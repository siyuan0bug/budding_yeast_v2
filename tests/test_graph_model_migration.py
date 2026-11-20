"""Source graph routing/checkpoint tests; artificial inputs, no dataset or training."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from budding_yeast_v2.data.yeast_graph import build_yeast_graph
from budding_yeast_v2.models.ours.graph_contracts import (
    resolve_variable_graph, row_normalized_support, set_graph_load_policy, variable_graph_report,
)
from budding_yeast_v2.models.ours.bio_refiner_plus import StoichiometricGraphAttention
from budding_yeast_v2.models.ours.cross_fno import NIPSYeastBlock
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule, SINGLE_VARIABLE_GRAPH_MODELS


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def offdiag_count(matrix):
    a = matrix.bool().clone()
    a.fill_diagonal_(False)
    return int(a.sum())


@pytest.mark.parametrize("view,edges", [("hybrid", 226), ("continuous", 218)])
def test_cwd_independent_default_and_single_self_loop(tmp_path, monkeypatch, view, edges):
    monkeypatch.chdir(tmp_path)
    graph = resolve_variable_graph(view=view)
    assert offdiag_count(graph.matrix) == edges
    assert torch.all(graph.matrix.diag() == 1)
    normalized = row_normalized_support(graph)
    assert torch.allclose(normalized.sum(-1), torch.ones(39))
    assert torch.allclose(normalized.diag(), 1 / graph.matrix.sum(-1))
    assert graph.origin == 1 and graph.digest == build_yeast_graph().source_sha256


def test_override_validation_and_path(tmp_path):
    a = np.eye(39, dtype=np.float32)
    a[3, 7] = -2
    path = tmp_path / "override.npy"
    np.save(path, a)
    graph = resolve_variable_graph(path=path)
    assert graph.origin == 2 and graph.view_code == 0
    assert graph.matrix[3, 7] == -2 and graph.matrix[7, 3] == 0
    for value in (np.eye(38), np.full((39, 39), np.nan)):
        with pytest.raises(ValueError):
            resolve_variable_graph(value)
    with pytest.raises(ValueError):
        resolve_variable_graph(num_vars=27)
    with pytest.raises(FileNotFoundError):
        resolve_variable_graph(path=tmp_path / "missing.npy")
    with pytest.raises(ValueError):
        resolve_variable_graph(a, path=path)


@pytest.mark.parametrize("policy", ["current", "checkpoint"])
def test_old_identity_checkpoint_policy_preserves_learned_weights(policy):
    source = StoichiometricGraphAttention(channels=8, head_dim=4)
    state = copy.deepcopy(source.state_dict())
    del state["_variable_graph_metadata"]
    state["graph_bias"] = torch.eye(39)
    target = StoichiometricGraphAttention(channels=8, head_dim=4)
    set_graph_load_policy(target, policy)
    target.load_state_dict(state, strict=True)
    for name, parameter in target.named_parameters():
        assert torch.equal(parameter, state[name])
    if policy == "current":
        assert offdiag_count(target.graph_bias) == 226
        assert variable_graph_report(target)[0]["origin"] == "ode_source"
    else:
        assert torch.equal(target.graph_bias, torch.eye(39))
        assert variable_graph_report(target)[0]["origin"] == "legacy_unknown"


def test_new_roundtrip_and_invalid_provenance():
    graph = resolve_variable_graph(view="continuous")
    source = StoichiometricGraphAttention(channels=8, head_dim=4, adj_matrix=graph)
    target = StoichiometricGraphAttention(channels=8, head_dim=4, adj_matrix=graph)
    target.load_state_dict(source.state_dict(), strict=True)
    assert torch.equal(source.graph_bias, target.graph_bias)
    set_graph_load_policy(target, "checkpoint")
    target.load_state_dict(source.state_dict(), strict=True)
    assert variable_graph_report(target)[0]["view"] == "continuous"
    for strict in (True, False):
        state = copy.deepcopy(source.state_dict())
        state["_variable_graph_metadata"] = torch.zeros(4)
        with pytest.raises(RuntimeError, match="provenance"):
            target.load_state_dict(state, strict=strict)
        state = copy.deepcopy(source.state_dict())
        state["graph_bias"][0, 0] = float("nan")
        with pytest.raises(RuntimeError, match="graph buffer"):
            target.load_state_dict(state, strict=strict)


@pytest.mark.parametrize("policy", ["current", "checkpoint"])
def test_historical_missing_optional_graph_branch(policy):
    source = NIPSYeastBlock(39, 2, 2)
    state = copy.deepcopy(source.state_dict())
    for key in ("adj_matrix", "physics_weights", "_variable_graph_metadata"):
        del state[key]
    target = NIPSYeastBlock(39, 2, 2)
    set_graph_load_policy(target, policy)
    target.load_state_dict(state, strict=True)
    if policy == "current":
        assert offdiag_count(target.adj_matrix) == 226
        assert torch.all(target.physics_weights == .1)
    else:
        assert not target.adj_matrix.any() and not target.physics_weights.any()
        assert variable_graph_report(target)[0]["origin"] == "legacy_disabled"
        # A zero structural bias reproduces the historical None branch.
        x = torch.randn(1, 39, 2, 16)
        shift = torch.zeros(1, 39, 2, 1)
        actual = target(x, shift)
        target.adj_matrix = None
        assert torch.equal(actual, target(x, shift))


@pytest.mark.parametrize("name", sorted(SINGLE_VARIABLE_GRAPH_MODELS))
def test_registry_routes_canonical_graph_and_finite_forward_backward(name):
    lit = YeastLitModule(model_name=name, width=16 if name != "cross_fno" else 4,
                         modes=2, n_blocks=1, loss_type="mse_only", graph_view="continuous")
    lit.set_parameter_stats(norm_type="physical")
    report = lit.graph_report()
    assert report["variable_graphs"]
    assert all(item["origin"] == "ode_source" and item["view"] == "continuous"
               for item in report["variable_graphs"])
    # The flux branch now has its own source-derived typed + untyped contract.
    if name in ("biom_refiner_graph_flux", "biom_refiner_mechanistic"):
        assert not report["legacy_six_relation_branch"]
        assert report['relation_graphs'][0]['total_nonself'] == 218
    ic = torch.full((1, 39, 32), .1)
    time = torch.linspace(0, 1, 32)[None, None].expand_as(ic)
    p = torch.ones(1, 141)
    output = lit(torch.stack((ic, time), 2), p)
    assert output.shape == (1, 39, 32) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(torch.isfinite(par.grad).all() for par in lit.parameters() if par.grad is not None)


def test_no_graph_baseline_is_unchanged_and_wrong_interface_fails():
    lit = YeastLitModule(model_name="fno", width=8, modes=2, loss_type="mse_only")
    assert lit.graph_report()["variable_graphs"] == []
    with pytest.raises(ValueError, match="does not accept"):
        YeastLitModule(model_name="fno", adj_matrix=np.eye(39))


def test_explicit_path_is_routed_and_contents_checkpointed(tmp_path):
    path = tmp_path / "override.npy"
    a = np.eye(39, dtype=np.float32)
    a[4, 2] = 1
    np.save(path, a)
    lit = YeastLitModule(model_name="biom_refiner_plus", width=16, modes=2, n_blocks=1,
                         loss_type="mse_only", adj_matrix_path=str(path))
    assert lit.graph_report()["variable_graphs"][0]["origin"] == "explicit_override"
    assert lit.hparams.adj_matrix_path is None
    assert torch.equal(lit.hparams.adj_matrix, torch.from_numpy(a))
    restored = YeastLitModule(**dict(lit.hparams))
    restored.load_state_dict(lit.state_dict(), strict=True)
    assert restored.graph_report()["variable_graphs"][0]["origin"] == "explicit_override"


def test_eval_checkpoint_policy_and_file_immutability(tmp_path):
    from budding_yeast_v2.eval import load_model_from_checkpoint
    from hashlib import sha256
    lit = YeastLitModule(model_name="biom_refiner_plus", width=16, modes=2,
                         n_blocks=1, loss_type="mse_only")
    state = copy.deepcopy(lit.state_dict())
    graph_key = "model.stage2_refiner.graph_attn.graph_bias"
    state[graph_key] = torch.eye(39)
    del state["model.stage2_refiner.graph_attn._variable_graph_metadata"]
    path = tmp_path / "legacy.ckpt"
    torch.save({"hyper_parameters": dict(lit.hparams), "state_dict": state}, path)
    before = sha256(path.read_bytes()).hexdigest()
    current, _ = load_model_from_checkpoint(path)
    legacy, _ = load_model_from_checkpoint(path, graph_load_policy="checkpoint")
    assert offdiag_count(current.state_dict()[graph_key]) == 226
    assert torch.equal(legacy.state_dict()[graph_key], torch.eye(39))
    assert sha256(path.read_bytes()).hexdigest() == before
    state.pop("model.stage2_refiner.graph_attn.q_proj.weight")
    torch.save({"hyper_parameters": dict(lit.hparams), "state_dict": state}, path)
    with pytest.raises(RuntimeError, match="Incomplete model checkpoint"):
        load_model_from_checkpoint(path)


def test_hierarchical_paths_distinguish_source_graph_views():
    from budding_yeast_v2.utils.experiments.paths import generate_hparams_signature
    kwargs = dict(width=16, modes=2, batch_size=4, lr=.001, model_name="biom_refiner_plus")
    old = generate_hparams_signature(**kwargs)
    hybrid = generate_hparams_signature(**kwargs, use_adj=True, graph_view="hybrid")
    continuous = generate_hparams_signature(**kwargs, use_adj=True, graph_view="continuous")
    assert len({old, hybrid, continuous}) == 3
    assert "gsrc1-hybrid" in hybrid and "gsrc1-continuous" in continuous
