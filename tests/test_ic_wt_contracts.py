"""Regression tests using artificial ICs/stats and the public ODE WT only.

No real-mutant parameters/trajectories, dataset generation, or training run.
"""
import io
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from budding_yeast_v2.data.mutant_aware import PARAM_KEYS, get_default_params
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.models.ours.input_contracts import ParameterInputTransform, extract_initial_state
from budding_yeast_v2.models.ours.bio_refiner_attractor import BioRefinerAttractor, EquilibriumRelaxationStream
from budding_yeast_v2.models.ours.bio_refiner_attractor_plus import BioRefinerAttractorPlus
from budding_yeast_v2.models.ours.bio_refiner_causal import CausalMutantAdapter, WT_VALUES_141
from budding_yeast_v2.models.ours.bio_refiner_causal_reach import BioRefinerCausalReach, CausalReachabilityGate
from budding_yeast_v2.models.ours.bio_refiner_dynamic_causal import TimeVaryingCausalAdapter as DynamicAdapter
from budding_yeast_v2.models.ours.bio_refiner_latch_causal import TimeVaryingCausalAdapter as LatchAdapter
from budding_yeast_v2.models.ours.bio_refiner_dynamic_plus import TimeVaryingDynamicAdapter
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule

NORMS = ("physical", "log1p", "linear", "asinh_z", "asinh_scale")
ADAPTERS = (CausalMutantAdapter, CausalReachabilityGate, DynamicAdapter, LatchAdapter, TimeVaryingDynamicAdapter)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def wt():
    return torch.tensor(WT_VALUES_141)[None]


def stats(norm):
    # Artificial, nonidentity statistics, independent of every real mutant.
    mean = .7 * wt() + .2
    std = torch.linspace(.4, 2., 141)[None]
    if norm == "asinh_scale":
        mean = torch.log10(wt().clamp(min=1e-8)) + .3
    if norm in ("asinh_z", "asinh_scale"):
        std[:, 10] = 0.  # Match DataModule's constant-coordinate handling.
        mean[:, 10] = wt()[:, 10] if norm == "asinh_z" else torch.log10(wt()[:, 10])
    return mean, std


def independent_encode(p, mean, std, norm):
    if norm == "physical":
        return p
    if norm in ("log1p", "linear"):
        return (p - mean) / std
    varying = std > 1e-5
    safe = torch.where(varying, std, torch.ones_like(std))
    transformed = torch.asinh((p - mean) / safe) if norm == "asinh_z" else (
        torch.log10(p.clamp(min=1e-8)) - mean) / safe
    return torch.where(varying, transformed, 0.)


def ic_input(layout="4d"):
    initial = torch.arange(1., 40.)[None, :, None].expand(2, -1, -1)
    ic = initial.expand(-1, -1, 32).clone()
    time = torch.linspace(0, 1, 32)[None, None].expand_as(ic)
    x = torch.stack((ic, time), 2)
    if layout == "3d":
        return ic, initial
    if layout == "interleaved":
        return x.reshape(2, 78, 32), initial
    return x, initial


@pytest.mark.parametrize("layout", ("4d", "3d", "interleaved"))
def test_ic_and_plateau_preserve_every_channel(layout):
    x, initial = ic_input(layout)
    x.requires_grad_()
    assert torch.equal(extract_initial_state(x, 39), initial)
    stream = EquilibriumRelaxationStream()
    result = stream(x, torch.zeros(2, 141))
    assert torch.equal(result[:, :, :1], initial)
    result.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_ic_noncontiguous_and_invalid_layouts():
    x, initial = ic_input()
    noncontiguous = x.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not noncontiguous.is_contiguous()
    assert torch.equal(extract_initial_state(noncontiguous, 39), initial)
    for invalid in (torch.zeros(()), torch.zeros(2, 39), torch.zeros(2, 38, 2, 4),
                    torch.zeros(2, 39, 3, 4), torch.zeros(2, 39, 2, 0)):
        with pytest.raises(ValueError):
            extract_initial_state(invalid, 39)


class FixedPattern(nn.Module):
    def __init__(self, arrest):
        super().__init__()
        self.arrest = arrest

    def forward(self, p):
        logits = p.new_full((len(p), 3), -100.)
        logits[:, 2 if self.arrest else 0] = 100.
        return logits


class ZeroCapacity(nn.Module):
    def forward(self, p):
        return p.new_zeros(len(p), 39, 1)


@pytest.mark.parametrize("layout", ("4d", "3d"))
@pytest.mark.parametrize("kind", ("attractor", "plus_plateau", "plus_capacity", "reach_capacity"))
def test_actual_model_branches_use_original_ic(layout, kind):
    factory = BioRefinerAttractor if kind == "attractor" else (
        BioRefinerCausalReach if kind == "reach_capacity" else BioRefinerAttractorPlus)
    model = factory(width=16, modes=4, n_blocks=1, coarse_n_blocks=1)
    if hasattr(model, "set_parameter_stats"):
        model.set_parameter_stats(norm_type="physical")
    if kind != "attractor":
        model.reachability_gate = ZeroCapacity()
    if kind != "reach_capacity":
        model.pattern_head = FixedPattern(arrest=kind in ("attractor", "plus_plateau"))
    x, initial = ic_input(layout)
    output, coarse, logits = model(x, wt().expand(2, -1), return_coarse=True, return_pattern=True)
    assert output.shape == coarse.shape == (2, 39, 32)
    assert logits.shape == (2, 3)
    if kind in ("plus_capacity", "reach_capacity"):
        assert torch.equal(output, initial.expand_as(output))
    else:
        assert torch.equal(output[:, :, :1], initial)


def test_wt_reference_matches_public_ode_order():
    defaults = get_default_params()
    defaults.update(init_CDH1T=1., init_CDH1=.9304992, init_BUB2=.2)
    assert torch.equal(wt()[0], torch.tensor([defaults[key] for key in PARAM_KEYS]))


def test_existing_ic_audit_now_reports_no_misplaced_channels():
    from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pure_random_evidence import audit_attractor_input
    assert audit_attractor_input()["wrong_initial_channels"] == 0


def test_existing_wt_audit_distinguishes_legacy_and_fixed_features(tmp_path):
    from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pure_random_evidence import audit_causal_reference
    synthetic = torch.cat((wt() * .7 + .1, wt() * 1.1 + .2, wt() * 1.4 + .3)).numpy()
    result = audit_causal_reference(synthetic, np.arange(3), np.array(PARAM_KEYS), tmp_path)
    assert result["implemented_wt_delta_l2"] == 0
    assert result["implemented_wt_log_res_l2"] == 0
    assert result["legacy_normalized_wt_minus_physical_reference_l2"] > 1


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("logarithm", ("signed", "ratio"))
def test_wt_features_zero_in_every_supported_parameter_space(norm, logarithm):
    mean, std = stats(norm)
    transform = ParameterInputTransform(141)
    transform.set_parameter_stats(mean, std, norm)
    encoded_wt = independent_encode(wt(), mean, std, norm)
    delta, log = transform.intervention_features(encoded_wt, wt()[0], logarithm=logarithm)
    assert torch.count_nonzero(delta) == 0
    assert torch.count_nonzero(log) == 0


@pytest.mark.parametrize("norm", ("log1p", "linear", "asinh_z", "asinh_scale"))
def test_decoder_matches_datamodule_and_physical_mutation_features(norm):
    mean, std = stats(norm)
    transform = ParameterInputTransform(141)
    transform.set_parameter_stats(mean, std, norm)
    physical = wt().clone()
    physical[:, 64] *= .4  # kasrent: public WT 200 -> arbitrary synthetic 80.
    physical[:, 132] *= 3.
    encoded = independent_encode(physical, mean, std, norm).requires_grad_()
    dm = YeastDataModule(dataset_path="unused.npz", norm_type=norm)
    dm.p_mean, dm.p_std = mean, std
    torch.testing.assert_close(transform.to_physical(encoded), dm.denormalize_p(encoded))
    delta, log = transform.intervention_features(encoded, wt()[0])
    expected = physical - wt()
    torch.testing.assert_close(delta[:, [64, 132]], expected[:, [64, 132]], rtol=1e-5, atol=3e-5)
    expected_log = torch.sign(expected) * torch.log1p(expected.abs() / (wt().abs() + 1e-5))
    torch.testing.assert_close(log[:, [64, 132]], expected_log[:, [64, 132]], rtol=1e-5, atol=3e-5)
    _, ratio = transform.intervention_features(encoded, wt()[0], eps=1e-6, logarithm="ratio")
    expected_ratio = torch.log(physical.clamp(min=1e-6)) - torch.log(wt().clamp(min=1e-6))
    torch.testing.assert_close(ratio[:, [64, 132]], expected_ratio[:, [64, 132]], rtol=1e-5, atol=3e-5)
    (delta.square().sum() + log.square().sum()).backward()
    assert encoded.grad is not None and torch.isfinite(encoded.grad).all()
    assert encoded.grad[:, 64].abs().sum() > 0


@pytest.mark.parametrize("factory", ADAPTERS)
@pytest.mark.parametrize("norm", ("log1p", "asinh_z", "asinh_scale"))
def test_all_adapters_feed_zero_wt_features_to_gene_projections(factory, norm):
    module = factory(d_gene=4)
    mean, std = stats(norm)
    module.set_parameter_stats(mean, std, norm)
    encoded_wt = independent_encode(wt(), mean, std, norm)
    captured = []
    projections = (module.gene_projections if isinstance(module, CausalMutantAdapter) else
                   module.module_proj.values() if isinstance(module, CausalReachabilityGate) else
                   module.module_projections)
    handles = [layer.register_forward_pre_hook(lambda layer, args: captured.append(args[0].detach().clone()))
               for layer in projections]
    try:
        module(encoded_wt) if factory in (CausalMutantAdapter, CausalReachabilityGate) else module(
            encoded_wt, torch.ones(1, 39, 32))
    finally:
        for handle in handles:
            handle.remove()
    assert captured and all(torch.count_nonzero(feature) == 0 for feature in captured)


def test_learned_conditioning_stays_normalized():
    mean, std = stats("log1p")
    p = independent_encode(wt(), mean, std, "log1p")
    for module in (CausalMutantAdapter(d_gene=4), CausalReachabilityGate(d_gene=4)):
        module.set_parameter_stats(mean, std)
        captured = []
        layer = module.fusion_mlp if isinstance(module, CausalMutantAdapter) else module.to_species_capacity
        handle = layer.register_forward_pre_hook(lambda layer, args: captured.append(args[0].detach().clone()))
        try:
            module(p)
        finally:
            handle.remove()
        actual = captured[0][:, :141] if isinstance(module, CausalMutantAdapter) else captured[0][:, -141:]
        assert torch.equal(actual, p)


def test_missing_or_invalid_statistics_fail_explicitly():
    transform = ParameterInputTransform(141)
    with pytest.raises(RuntimeError, match="set_parameter_stats"):
        transform.intervention_features(wt(), wt()[0])
    for mean, std, mode in ((None, None, "log1p"), (torch.zeros(3), torch.ones(3), "log1p"),
                            (torch.zeros(141), torch.zeros(141), "linear"),
                            (torch.full((141,), float("nan")), torch.ones(141), "log1p"),
                            (torch.zeros(141), -torch.ones(141), "asinh_z"),
                            (None, None, "unknown")):
        with pytest.raises(ValueError):
            transform.set_parameter_stats(mean, std, mode)
    transform.set_parameter_stats(norm_type="physical")
    delta, log = transform.intervention_features(wt(), wt()[0])
    assert not delta.any() and not log.any()


@pytest.mark.parametrize("norm", ("log1p", "asinh_z", "asinh_scale"))
def test_checkpoint_stats_roundtrip_and_legacy_strict_load(norm):
    source, target = CausalReachabilityGate(d_gene=4), CausalReachabilityGate(d_gene=4)
    mean, std = stats(norm)
    source.set_parameter_stats(mean, std, norm)
    p = independent_encode(wt(), mean, std, norm)
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    target.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    torch.testing.assert_close(target(p), source(p), rtol=0, atol=0)
    legacy = {key: value for key, value in source.state_dict().items() if not key.startswith("parameter_transform.")}
    target = CausalReachabilityGate(d_gene=4)
    target.load_state_dict(legacy, strict=True)
    with pytest.raises(RuntimeError, match="set_parameter_stats"):
        target(p)
    target.set_parameter_stats(mean, std, norm)
    torch.testing.assert_close(target(p), source(p), rtol=0, atol=0)
    # Lightning may inject stats in setup before loading a legacy checkpoint.
    injected_first = CausalReachabilityGate(d_gene=4)
    injected_first.set_parameter_stats(mean, std, norm)
    injected_first.load_state_dict(legacy, strict=True)
    torch.testing.assert_close(injected_first(p), source(p), rtol=0, atol=0)
    # A partial/malformed new-format state is not silently accepted as legacy.
    partial = source.state_dict()
    partial.pop("parameter_transform.p_std")
    for strict in (True, False):
        with pytest.raises(RuntimeError, match="Incomplete parameter-normalization state"):
            CausalReachabilityGate(d_gene=4).load_state_dict(partial, strict=strict)


@pytest.mark.parametrize("name", ("biom_refiner_causal", "biom_refiner_causal_reach", "biom_refiner_attractor_plus",
                                 "biom_refiner_dynamic_causal_plus", "biom_refiner_latch_causal_plus", "biom_refiner_dynamic_plus"))
def test_lightning_setup_injects_stats_and_forward_backward(name):
    lit = YeastLitModule(model_name=name, width=16, modes=4, n_blocks=1, loss_type="mse_only")
    mean, std = stats("log1p")
    dm = SimpleNamespace(mean_y=torch.zeros(1, 39, 1), std_y=torch.ones(1, 39, 1),
                         p_mean=mean, p_std=std, norm_type="log1p", t_max=210., s_c=None)
    lit._trainer = SimpleNamespace(datamodule=dm)
    lit.setup("fit")
    lit._trainer = None
    lit.log = lambda *args, **kwargs: None
    x, _ = ic_input()
    p = independent_encode(wt().expand(2, -1), mean, std, "log1p")
    loss = lit.training_step((x, p, torch.zeros(2, 39, 32)), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in lit.parameters() if parameter.grad is not None)


def test_double_precision_module_and_input_gradients():
    module = CausalReachabilityGate(d_gene=4).double()
    mean, std = (tensor.double() for tensor in stats("log1p"))
    module.set_parameter_stats(mean, std)
    p = independent_encode(wt().double(), mean, std, "log1p").requires_grad_()
    output = module(p)
    assert output.dtype == torch.float64
    output.sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_cpu_autocast_does_not_break_physical_features():
    module = CausalReachabilityGate(d_gene=4)
    mean, std = stats("log1p")
    module.set_parameter_stats(mean, std)
    p = independent_encode(wt(), mean, std, "log1p").requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = module(p)
    assert torch.isfinite(output).all()
    output.float().sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


@pytest.mark.parametrize("name", ("biom_refiner_causal_reach", "biom_refiner_attractor", "biom_refiner_attractor_plus"))
def test_existing_legacy_checkpoint_loads_strictly_without_reading_real_data(name):
    root = Path(__file__).resolve().parents[1]
    directory = root / "experiments/mutant_aware_pure_random_100000_210m_500s" / name
    candidates = sorted(p for p in directory.glob("**/checkpoints/*.ckpt") if p.name != "last.ckpt")
    if not candidates:
        pytest.skip("Historical checkpoint is not installed in this workspace")
    saved = torch.load(candidates[0], map_location="cpu", weights_only=False)
    parameters = inspect.signature(YeastLitModule).parameters
    lit = YeastLitModule(**{key: value for key, value in saved["hyper_parameters"].items() if key in parameters})
    lit.load_state_dict(saved["state_dict"], strict=True)
    mean, std = stats("log1p")
    lit.set_parameter_stats(mean, std, norm_type="log1p")
    ic = torch.ones(1, 39, 64)
    time = torch.linspace(0, 1, 64)[None, None].expand_as(ic)
    p = independent_encode(wt(), mean, std, "log1p")
    with torch.inference_mode():
        result = lit(torch.stack((ic, time), 2), p)
    assert result.shape == (1, 39, 64) and torch.isfinite(result).all()


def test_standalone_eval_binds_stats_before_forward(tmp_path):
    from budding_yeast_v2.eval import evaluate_model_on_datamodule
    from budding_yeast_v2.utils.evaluation.metrics import calculate_metrics

    class EvaluationProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = CausalReachabilityGate(d_gene=4)

        def set_parameter_stats(self, *args, **kwargs):
            self.gate.set_parameter_stats(*args, **kwargs)

    model = EvaluationProbe()
    mean, std = stats("log1p")
    dm = SimpleNamespace(mean_y=torch.zeros(1, 39, 1), std_y=torch.ones(1, 39, 1),
                         p_mean=mean, p_std=std, norm_type="log1p", test_dataset_real=object(),
                         test_dataset_synth=object(), _test_idx_real=[0], _test_idx_synth=[1],
                         _mutant_names=["m0", "m1"])

    names = iter(("m0", "m1"))
    def fake_evaluate(model, dm, *args, **kwargs):
        p = independent_encode(wt(), mean, std, "log1p")
        delta, log = model.gate.parameter_transform.intervention_features(p, model.gate.p_wt)
        assert not delta.any() and not log.any()
        output = torch.zeros(1, 39, 4)
        summary = {"all_39_vars": calculate_metrics(output, output),
                   "core_9_vars": calculate_metrics(output, output), "count": 1}
        row = dict(Mutant=next(names), Pattern="Pattern_A", MAE=0., MSE=0.,
                   Relative_L2=0., Correlation=1.)
        return summary, [row], [output], [output], []

    with patch("budding_yeast_v2.eval.evaluate_dataset", side_effect=fake_evaluate) as evaluate:
        evaluate_model_on_datamodule(model, dm, device="cpu", save_dir=str(tmp_path), max_plots=0)
        assert evaluate.call_count == 2
