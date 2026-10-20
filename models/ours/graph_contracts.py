"""Variable/relation/module graph inputs and explicit checkpoint semantics.

All matrices use A[target, source].  The default is unsigned potential support,
not a globally activating/inhibiting Jacobian and not a finite-horizon mask.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from functools import partial
import warnings

import numpy as np
import torch

from budding_yeast_v2.data.yeast_graph import build_yeast_graph


@dataclass(frozen=True)
class VariableGraphInput:
    matrix: torch.Tensor
    origin: int  # 1: ODE source; 2: explicit override
    view_code: int  # 1: hybrid; 2: continuous; 0: unspecified override
    digest: str

    @property
    def shape(self):
        return self.matrix.shape

    def metadata(self):
        return torch.tensor(
            [1, self.origin, self.view_code, *bytes.fromhex(self.digest)],
            dtype=torch.uint8,
        )


def resolve_variable_graph(value=None, num_vars=39, *, view="hybrid", path=None, system_id=None):
    """Resolve once, without cwd-relative files or an identity fallback.

    An explicit matrix/path is an ablation override, not certified ODE support.
    Overrides must already follow the target/source convention.
    """
    if view not in ("continuous", "hybrid"):
        raise ValueError(f"Unsupported variable graph view: {view}")
    if path is not None:
        if value is not None:
            raise ValueError("Specify either a graph matrix or adj_matrix_path, not both")
        value = np.load(Path(path).expanduser(), allow_pickle=False)
    if isinstance(value, VariableGraphInput):
        spec = value
    elif value is None:
        graph = build_yeast_graph(system_id=system_id or ('fission' if num_vars==16 else 'budding'))
        if num_vars != len(graph.variable_names):
            raise ValueError("The source-derived yeast graph has 39 variables; supply an explicit matrix for other sizes")
        matrix = torch.tensor(graph.adjacency(view, self_loops="all"), dtype=torch.float32)
        spec = VariableGraphInput(matrix, 1, 1 if view == "hybrid" else 2, graph.source_sha256)
    else:
        matrix = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32).clone()
        digest = sha256(matrix.contiguous().numpy().tobytes()).hexdigest()
        spec = VariableGraphInput(matrix, 2, 0, digest)
    if spec.matrix.shape != (num_vars, num_vars):
        raise ValueError(f"Expected graph shape {(num_vars, num_vars)}, got {tuple(spec.matrix.shape)}")
    if not torch.isfinite(spec.matrix).all():
        raise ValueError("Variable graph contains non-finite entries")
    return spec


def row_normalized_support(spec):
    """Unsigned affinity, with exactly one unit self-loop (never A + I)."""
    matrix = spec.matrix.abs().clone()
    matrix.fill_diagonal_(1.0)
    return matrix / matrix.sum(-1, keepdim=True).clamp(min=1.0)


def _validate_metadata(value):
    if (not isinstance(value, torch.Tensor) or value.shape != (35,)
            or value.dtype != torch.uint8 or int(value[0]) != 1
            or int(value[1]) not in (0, 1, 2, 3)
            or int(value[2]) not in (0, 1, 2)):
        raise RuntimeError("Invalid variable graph provenance metadata")


def _graph_load_hook(module, state_dict, prefix, local_metadata, strict,
                     missing_keys, unexpected_keys, error_msgs):
    """Keep trainable weights; refresh only declared, deterministic graph buffers."""
    policy = module._variable_graph_load_policy
    expected = module._variable_graph_expected
    meta_key = prefix + "_variable_graph_metadata"
    saved_meta = state_dict.get(meta_key)
    if saved_meta is not None:
        _validate_metadata(saved_meta)
    for name, value in expected.items():
        saved = state_dict.get(prefix + name)
        if saved is not None and (not isinstance(saved, torch.Tensor)
                                  or saved.shape != value.shape
                                  or not torch.isfinite(saved).all()):
            raise RuntimeError(f"Invalid graph buffer {prefix + name}")

    # Some historical cross/WNO/LNO/channel checkpoints had no graph branch.
    # Only this documented pair of absent graph buffers/parameters is allowed.
    optional = module._variable_graph_optional_parameters
    absent_branch = (bool(optional) and saved_meta is None
                     and all(prefix + k not in state_dict for k in expected)
                     and all(prefix + k not in state_dict for k in optional))
    if absent_branch:
        for name in optional:
            initial = module._variable_graph_initial_parameters[name]
            state_dict[prefix + name] = initial.clone() if policy == "current" else torch.zeros_like(initial)
        warnings.warn(
            "Historical checkpoint has no variable-graph branch: current policy initializes "
            "only its graph affinity parameters; checkpoint policy keeps the branch disabled.",
            UserWarning,
        )

    changed = []
    if policy == "current":
        for name, value in expected.items():
            key = prefix + name
            if key not in state_dict or not torch.equal(state_dict[key].detach().cpu(), value):
                changed.append(key)
            state_dict[key] = value.clone()
        state_dict[meta_key] = module._variable_graph_expected_metadata.clone()
        if changed:
            warnings.warn(
                "Checkpoint variable graph replaced by the configured current graph; learned "
                "weights are retained. This is a migrated model, not a historical reproduction.",
                UserWarning,
            )
    else:
        if absent_branch:
            for name, value in expected.items():
                state_dict[prefix + name] = torch.zeros_like(value)
        elif any(prefix + name not in state_dict for name in expected):
            raise RuntimeError(f"Incomplete checkpoint variable graph at {prefix}")
        if saved_meta is None:
            # Do not mislabel an old identity/signed graph as source-derived.
            meta = torch.zeros(35, dtype=torch.uint8)
            meta[0] = 1
            meta[1] = 3 if absent_branch else 0
            state_dict[meta_key] = meta
    module._variable_graph_last_load = {"policy": policy, "replaced_buffers": changed,
                                        "initialized_graph_parameters": list(optional) if absent_branch else []}


def attach_variable_graph(module, spec, buffer_names, *, optional_parameters=()):
    """Declare the exact fixed buffers covered by the graph checkpoint contract."""
    module.register_buffer("_variable_graph_metadata", spec.metadata())
    module._variable_graph_expected_metadata = spec.metadata()
    module._variable_graph_expected = {
        name: getattr(module, name).detach().cpu().clone() for name in buffer_names
    }
    module._variable_graph_optional_parameters = tuple(optional_parameters)
    module._variable_graph_initial_parameters = {
        name: getattr(module, name).detach().cpu().clone() for name in optional_parameters
    }
    module._variable_graph_load_policy = "current"
    module._variable_graph_last_load = None
    module.register_load_state_dict_pre_hook(_graph_load_hook)


def set_graph_load_policy(model, policy):
    if policy not in ("current", "checkpoint"):
        raise ValueError("graph_load_policy must be current or checkpoint")
    for module in model.modules():
        if hasattr(module, "_variable_graph_load_policy"):
            module._variable_graph_load_policy = policy
        if hasattr(module, '_fixed_graph_load_policy'):
            module._fixed_graph_load_policy = policy


def variable_graph_report(model):
    """JSON-safe actual graph provenance, including unlabelled legacy graphs."""
    origins = {0: "legacy_unknown", 1: "ode_source", 2: "explicit_override", 3: "legacy_disabled"}
    views = {0: None, 1: "hybrid", 2: "continuous"}
    report = []
    for name, module in model.named_modules():
        if not hasattr(module, "_variable_graph_metadata"):
            continue
        meta = module._variable_graph_metadata.detach().cpu()
        _validate_metadata(meta)
        report.append({
            "module": name, "origin": origins[int(meta[1])], "view": views[int(meta[2])],
            "source_or_matrix_sha256": bytes(meta[3:].tolist()).hex() if int(meta[1]) in (1, 2) else None,
            "orientation": "target_source", "policy": module._variable_graph_load_policy,
            "last_load": module._variable_graph_last_load,
        })
    return report


def _fixed_graph_load_hook(module, state_dict, prefix, local_metadata, strict,
                           missing_keys, unexpected_keys, error_msgs, *, kind):
    contract = module._fixed_graph_contracts[kind]
    policy = module._fixed_graph_load_policy
    meta_name = f'_{kind}_graph_metadata'
    saved_meta = state_dict.get(prefix + meta_name)
    if prefix + meta_name in state_dict and saved_meta is None:
        raise RuntimeError(f'Invalid {kind} graph provenance metadata')
    if saved_meta is not None:
        _validate_metadata(saved_meta)
    legacy = saved_meta is None or int(saved_meta[1]) == 0
    changed = []
    for name, expected in contract['buffers'].items():
        key = prefix + name
        saved = state_dict.get(key)
        if saved is not None and (not isinstance(saved, torch.Tensor) or saved.shape != expected.shape
                                  or not torch.isfinite(saved).all()):
            raise RuntimeError(f'Invalid {kind} graph buffer {key}')
        if saved is None and not (saved_meta is None and name in contract['optional_legacy_buffers']):
            raise RuntimeError(f'Incomplete {kind} graph checkpoint: {key}')
        if policy == 'current':
            if saved is None or not torch.equal(saved.detach().cpu(), expected):
                changed.append(key)
            state_dict[key] = expected.clone()
        elif saved is None:
            state_dict[key] = torch.zeros_like(expected)
    if policy == 'current':
        state_dict[prefix + meta_name] = contract['metadata'].clone()
        module._fixed_graph_legacy_runtime[kind] = False
        if changed:
            warnings.warn(f'Checkpoint {kind} graph refreshed from the current contract; '
                          'learned weights retained. Not a historical reproduction.', UserWarning)
    else:
        module._fixed_graph_legacy_runtime[kind] = legacy
        if saved_meta is None:
            meta = torch.zeros(35, dtype=torch.uint8)
            meta[0] = 1
            state_dict[prefix + meta_name] = meta
    contract['last_load'] = {'policy': policy, 'replaced_buffers': changed,
                             'legacy_as_saved': legacy if policy == 'checkpoint' else False}


def attach_fixed_graph(module, kind, metadata, buffer_names, *, optional_legacy_buffers=()):
    """Fixed relation/module buffers; no new trainable state or silent fallbacks."""
    _validate_metadata(metadata)
    module.register_buffer(f'_{kind}_graph_metadata', metadata.clone())
    if not hasattr(module, '_fixed_graph_contracts'):
        module._fixed_graph_contracts = {}
        module._fixed_graph_legacy_runtime = {}
        module._fixed_graph_load_policy = 'current'
    module._fixed_graph_contracts[kind] = {
        'metadata': metadata.clone(),
        'buffers': {n: getattr(module, n).detach().cpu().clone() for n in buffer_names},
        'optional_legacy_buffers': tuple(optional_legacy_buffers), 'last_load': None,
    }
    module._fixed_graph_legacy_runtime[kind] = False
    module.register_load_state_dict_pre_hook(partial(_fixed_graph_load_hook, kind=kind))


def fixed_graph_is_legacy(module, kind):
    return module._fixed_graph_legacy_runtime.get(kind, False)


def fixed_graph_report(model):
    result = []
    for name, module in model.named_modules():
        for kind, contract in getattr(module, '_fixed_graph_contracts', {}).items():
            meta = getattr(module, f'_{kind}_graph_metadata').detach().cpu()
            _validate_metadata(meta)
            legacy = int(meta[1]) == 0
            entry = {'module': name, 'kind': kind,
                     'origin': 'legacy_unknown' if legacy else ('ode_source' if int(meta[1]) == 1 else 'explicit_prior_or_override'),
                     'orientation': 'legacy_as_saved' if legacy else 'target_source',
                     'view': {0: None, 1: 'hybrid', 2: 'continuous'}[int(meta[2])],
                     'contract_sha256': None if legacy else bytes(meta[3:].tolist()).hex(),
                     'policy': module._fixed_graph_load_policy, 'last_load': contract['last_load']}
            if kind == 'relations':
                typed = module.rel_graphs.detach().cpu().ne(0).any(0)
                total = typed | module.untyped_graph.detach().cpu().ne(0)
                typed.fill_diagonal_(False)
                total.fill_diagonal_(False)
                entry.update(typed_nonself=int(typed.sum()), total_nonself=int(total.sum()),
                             untyped_nonself=int((total & ~typed).sum()),
                             used_in_forward=getattr(module, '_source_relations_used_in_forward', True),
                             classification='legacy_as_saved' if legacy else 'source_term_annotations_or_explicit_override')
            result.append(entry)
    return result
