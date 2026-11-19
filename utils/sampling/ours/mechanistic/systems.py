"""Read-only ODE adapters. No real-mutant tables or trajectory labels are read."""
import ast
from functools import lru_cache
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


@lru_cache(None)
def _fission_rhs():
    from budding_yeast_v2.data.systems.fission import ode
    from budding_yeast_v2.utils.physics.canonical_rhs import _Vectorize, _maximum, _where

    class Vectorize(_Vectorize):
        def visit_Call(self, node):
            node = super().visit_Call(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'max':
                node.func.id = '_maximum'
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'np' and node.func.attr == 'array'):
                node.func = ast.Attribute(ast.Name('torch', ast.Load()), 'stack', ast.Load())
                node.keywords = [ast.keyword('dim', ast.Constant(-1))]
            return node

    source = inspect.getsource(ode._Y_switch) + '\n' + inspect.getsource(ode.eqns)
    namespace = dict(torch=torch, _maximum=_maximum, _where=_where,
                     np=SimpleNamespace(exp=torch.exp))
    tree = ast.fix_missing_locations(Vectorize().visit(ast.parse(source)))
    exec(compile(tree, '<canonical_fission_sampling_rhs>', 'exec'), namespace)
    return namespace['eqns'], hashlib.sha256(source.encode()).hexdigest()


class BiologicalODE:
    """Physical trajectories B,V,T; supplied initials retain their native units.

    Fission exported MASS trajectories are relative; its stored birth initials
    and canonical RHS use raw mass. Its omitted kS mode is reconstructed from
    replication completion for each continuous segment, never held constant.
    """
    def __init__(self, name):
        if name not in ('budding', 'fission'):
            raise ValueError('Supply an explicit supported biological system')
        self.name = name
        if name == 'budding':
            from budding_yeast_v2.data.systems.budding.specification import SPEC
            from budding_yeast_v2.utils.physics.canonical_rhs import SOURCE_SHA256
            self.mass_unit = 1.
            self.source_sha = SOURCE_SHA256
            from budding_yeast_v2.data.systems.budding import simulation
            self.certificate_supported = (
                self.source_sha == '1a7cfe09f61176b427cad4af2494d5cd85bbd16bf4d2ca4f6ce64cbc5c3f8af5'
                and hashlib.sha256(Path(simulation.__file__).read_bytes()).hexdigest()
                == '33acd9e7ea59e5c18bb08a17e8e8f125ec7857c0dcb851c1042f53b63de43b88')
        else:
            from budding_yeast_v2.data.systems.fission.specification import SPEC
            from budding_yeast_v2.data.systems.fission.constants import MASS_UNIT
            self.mass_unit = MASS_UNIT
            _, self.source_sha = _fission_rhs()
            self.certificate_supported = False
        self.spec = SPEC
        self.param_names = tuple(SPEC.param_names)
        self.variable_names = tuple(SPEC.variable_names)
        self.n = len(self.variable_names)
        self.horizon = float(SPEC.horizon)

    def initials(self, initial):
        y = np.asarray(initial, dtype=np.float64).copy()
        if y.shape[-1] != self.n:
            raise ValueError('Initial-state dimension does not match system')
        y[..., 0] /= self.mass_unit
        return y

    def rhs(self, y, p, t, metadata=()):
        if y.shape[-1] != self.n or p.shape[-1] != len(self.param_names):
            raise ValueError('ODE input dimension does not match declared system')
        if self.name == 'budding':
            from budding_yeast_v2.utils.physics.canonical_rhs import rhs
            flags = {key: torch.tensor([bool(m.get('_effective_overrides', m.get('original_overrides', {})).get(key, False))
                                       for m in metadata], device=y.device)
                     for key in ('is_sbf_ca', 'is_mutant_104')} if metadata else None
            return rhs(y, p, t, flags)
        raw = y.clone()
        raw[..., 0] *= self.mass_unit
        values = {key: p[:, j].reshape(len(p), *([1] * (y.ndim - 2)))
                  for j, key in enumerate(self.param_names)}
        values['kS'] = torch.where(raw[..., 1] >= 1., values['kSmin'], values['kSmax'])
        result = _fission_rhs()[0](t, raw, values)
        result[..., 0] /= self.mass_unit
        return result

    def guards(self, y, p):
        if self.name == 'budding':
            return torch.stack((y[..., 2] - .3, y[..., 2] + y[..., 3] - .2,
                                y[..., 33] - 1., y[..., 35] - 1.), -1), (-1, -1, 1, 1)
        lookup = dict(zip(self.param_names, p.unbind(-1)))
        eps = lookup['eps'].reshape(len(p), *([1] * (y.ndim - 2)))
        ratio = (y[..., 2] + eps * y[..., 3]) / (1. - y[..., 15]).clamp(min=1e-6)
        return torch.stack((ratio - .65, y[..., 1] - .99999, ratio - .50), -1), (1, 1, -1)

    def event(self, left, right, p):
        """Earliest directed crossing; only a valid Fission START changes mode."""
        values, directions = self.guards(torch.stack((left, right))[None], p[None])
        a, b = values[0]
        found = []
        for k, direction in enumerate(directions):
            crosses = (a[k] >= 0 and b[k] < 0) if direction < 0 else (a[k] <= 0 and b[k] > 0)
            if crosses:
                fraction = float((a[k] / (a[k] - b[k])).clamp(0., 1.))
                if self.name == 'fission' and k == 0 and float(left[1]) < 1. - 1e-6:
                    continue
                found.append((fraction, k))
        return (min(found) if found else None), len(found)

    def reset(self, state, p, event, metadata=None):
        """Original reset algebra, expressed for differentiation in output units."""
        values = dict(zip(self.param_names, p.unbind(-1)))
        y = state.clone()
        if self.name == 'budding':
            if event == 0:
                y[34] = y[35] = 0.
                kg = np.log(2.) / values['mdt']
                y[0] = torch.abs(torch.exp(-kg * (1.026 / kg - 32.))) * state[0]
                y[37] = .1
                y[2] = state[2] - 1e-6
            elif event == 1:
                y[33] = 0.
                y[3] = state[3] - 1e-6
            elif event == 2:
                y[36] = values['Vi20_active']
                y[38] = values['BUB2_active']
                y[33] = state[33] + 1e-6
            else:
                y[36] = .01
                y[37] = 1.
                overrides = (metadata or {}).get('_effective_overrides', {})
                y[38] = 0. if float(values['BUB2_active']) == 0. else overrides.get('init_BUB2', float(values['init_BUB2']))
                y[35] = state[35] + 1e-6
            return y
        # Fission event handler first enforces its original state constraints.
        y = y.clamp(min=0.)
        y[1] = y[1].clamp(max=1.)
        if event == 2:
            y[0] = y[0] / 2.
            ratio = (y[2] + values['eps'] * y[3]) / (1. - y[15]).clamp(min=1e-6)
            if float((1. - y[15]).clamp(min=1e-6)) > 1e-6 and float(ratio) >= .4995:
                y[2:4] = y[2:4] * .9995
        elif event == 0:
            ratio = (y[2] + values['eps'] * y[3]) / (1. - y[15]).clamp(min=1e-6)
            y[1] = 0.
            if float(ratio) <= .6502:
                factor = .6504 / ratio.clamp(min=1e-12)
                if float(factor) <= 1.05:
                    y[2:4] = y[2:4] * factor
        else:
            y[1] = 1.
        return y

    def module_parameters(self):
        if self.name == 'budding':
            from budding_yeast_v2.data.systems.budding.catalog import GROUP_BY_NAME
            return {name: tuple(row['params']) for name, row in GROUP_BY_NAME.items()}
        from budding_yeast_v2.data.systems.fission.catalog import SAMPLING_MODULES
        return {name: tuple(row['params']) for name, row in SAMPLING_MODULES.items()}

    def equivalence_key(self, p, y0, metadata=None):
        """One conservative, analytic inactive-channel certificate, not a graph guess.

        With Tem1T=TEM1(0)=0, TEM1 stays zero; ka15pp only multiplies
        TEM1 and occurs in no guard/reset. All other effective inputs must match.
        Fission returns no certificate rather than fabricating an invariance.
        """
        if not self.certificate_supported:
            return None
        positions = {key: i for i, key in enumerate(self.param_names)}
        if (p[positions['Tem1T']] != 0. or y0[28] != 0.
                or p[positions['jatem']] <= 0. or p[positions['Jitem']] <= 0.):
            return None
        values = np.asarray(p, dtype='<f8').copy()
        values[positions['ka15pp']] = 0.
        flags = tuple(bool((metadata or {}).get('_effective_overrides',
                           (metadata or {}).get('original_overrides', {})).get(k, False))
                      for k in ('is_sbf_ca', 'is_mutant_104'))
        return values.tobytes(), np.asarray(y0, dtype='<f8').tobytes(), flags
