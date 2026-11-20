"""Scientific input-contract and upstream-recursion checks for paper baselines."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from budding_yeast_v2.models.baselines.fno import FNO
from budding_yeast_v2.models.baselines.wno import WNO, WaveletConv1d
from budding_yeast_v2.models.baselines.mwt import MWT, MWTKernel1d
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule, MODEL_REGISTRY


@pytest.fixture(autouse=True)
def deterministic_small_cpu_test():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        yield
    torch.set_num_threads(threads)


def inputs(length, batch=2, num_vars=3, param_dim=5):
    initial = torch.randn(batch, num_vars, requires_grad=True)
    time = torch.linspace(0, 1, length).expand(batch, num_vars, -1)
    x = torch.stack((initial.unsqueeze(-1).expand_as(time), time), dim=2)
    params = torch.randn(batch, param_dim, requires_grad=True)
    return x, params, initial


def build(kind, length):
    kwargs = dict(num_vars=3, param_dim=5, width=8, n_blocks=2)
    if kind is WNO:
        return kind(**kwargs, seq_len=length)
    return kind(**kwargs, modes=4)


@pytest.mark.parametrize('kind', [FNO, WNO, MWT])
@pytest.mark.parametrize('length', [31, 500])
def test_full_input_dependencies_and_backward(kind, length):
    model = build(kind, length)
    x, p, initial = inputs(length)
    out = model(x, p)
    assert out.shape == (2, 3, length)
    assert torch.isfinite(out).all()
    # One output species must be able to depend on every initial species and
    # every ODE parameter, including the other species' initial conditions.
    grad_y0, grad_p = torch.autograd.grad(out[:, 0].square().mean(), (initial, p), retain_graph=True)
    assert torch.all(grad_y0.abs().sum(0) > 0)
    assert torch.all(grad_p.abs().sum(0) > 0)
    out.square().mean().backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())


@pytest.mark.parametrize('kind', [FNO, WNO, MWT])
def test_equal_initial_species_remain_distinguishable(kind):
    model = build(kind, 32).eval()
    x, p, _ = inputs(32)
    x = x.detach()
    x[:, :, 0, :] = 0
    with torch.no_grad():
        out = model(x, p)
        assert not torch.equal(out[:, 0], out[:, 1])
        changed = x.clone()
        changed[:, 2, 0, :] = 1
        assert not torch.equal(out[:, 0], model(changed, p)[:, 0])
        # An initial-value task must not consume the other feature-0 entries
        # as a historical trajectory, even if callers supply different ones.
        changed = x.clone()
        changed[:, :, 0, 1:] = 999
        torch.testing.assert_close(out, model(changed, p))


@pytest.mark.parametrize('kind', [FNO, WNO, MWT])
def test_bfloat16_and_reload(kind):
    model = build(kind, 32)
    x, p, _ = inputs(32)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = model(x, p)
        loss = out.float().square().mean()
    loss.backward()
    assert torch.isfinite(out).all()
    restored = build(kind, 32)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x, p), model(x, p))


@pytest.mark.parametrize('name', ['fno', 'wno', 'mwt'])
@pytest.mark.parametrize('num_vars,param_dim', [(39, 141), (16, 58)])
def test_production_registry(name, num_vars, param_dim):
    model = YeastLitModule(model_name=name, num_vars=num_vars, param_dim=param_dim,
                           width=8, n_blocks=2, modes=4, loss_type='mse_only')
    x, p, _ = inputs(500, batch=1, num_vars=num_vars, param_dim=param_dim)
    with torch.no_grad():
        assert model.forward_ic_time(x, p).shape == (1, num_vars, 500)
    assert model.model.__class__ is MODEL_REGISTRY[name]


def test_wavelet_analysis_synthesis_and_coefficient_kernels():
    # At one level, retaining both low and detail with identity channel
    # kernels must recover the signal, including odd-length cropping.
    layer = WaveletConv1d(2, seq_len=31, level=1, wavelet='db2')
    with torch.no_grad():
        layer.low_weight.zero_()
        layer.detail_weight.zero_()
        for i in range(2):
            layer.low_weight[i, i].fill_(1)
            layer.detail_weight[i, i].fill_(1)
    x = torch.randn(2, 2, 31)
    torch.testing.assert_close(layer(x), x, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match='configured'):
        layer(x[..., :-1])


def test_multiwavelet_filter_inverse():
    block = MWTKernel1d(channels=2, k=4, modes=4)
    x = torch.randn(2, 16, 2, 4)
    detail, smooth = block.decompose(x)
    torch.testing.assert_close(block.reconstruct(torch.cat((smooth, detail), -1)), x,
                               atol=2e-6, rtol=1e-5)


def test_mwt_recursion_matches_upstream(monkeypatch):
    # Compare against the actual source, not a second copy of our recurrence.
    upstream = Path(__file__).resolve().parents[2] / 'baselines/Tau_BNO-main/models/MWNO'
    if not upstream.exists():
        pytest.skip('Upstream reference checkout unavailable')
    package = types.ModuleType('_mwt_test_upstream')
    package.__path__ = [str(upstream)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name in ('utils', 'models'):
        spec = importlib.util.spec_from_file_location(f'{package.__name__}.{name}', upstream / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    original = module.MWT_CZ1d(k=4, alpha=4, c=2)
    adapted = MWTKernel1d(channels=2, k=4, modes=4)
    with torch.no_grad():
        for key in ('ec_s', 'ec_d', 'rc_e', 'rc_o'):
            torch.testing.assert_close(getattr(adapted, key), getattr(original, key))
        adapted.T0.load_state_dict(original.T0.state_dict())
        for key in ('A', 'B', 'C'):
            getattr(adapted, key).weight.copy_(getattr(original, key).weights1)
    x = torch.randn(2, 32, 2, 4)
    torch.testing.assert_close(adapted(x), original(x), atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('kind', [FNO, WNO, MWT])
def test_bad_query_grids_rejected(kind):
    model = build(kind, 32)
    x, p, _ = inputs(32)
    broken = x.detach().clone()
    broken[:, :, 1, 8] += .01
    with pytest.raises(ValueError, match='uniformly'):
        model(broken, p)
    broken[:, 0, 1, :] = x[:, 0, 1, :] + .01
    with pytest.raises(ValueError, match='share'):
        model(broken, p)
