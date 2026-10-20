"""ODE-backed six relation overlays plus a parameter-free untyped route."""
from dataclasses import dataclass
from hashlib import sha256

import torch

from budding_yeast_v2.data.yeast_relations import build_yeast_relations
from .graph_contracts import (VariableGraphInput, attach_fixed_graph,
                              resolve_variable_graph)


def relation_inputs(num_vars=39, *, view='hybrid', variable_graph=None, rel_graphs=None):
    if rel_graphs is not None:
        rel = torch.as_tensor(rel_graphs, dtype=torch.float32).detach().cpu().clone()
        if rel.shape != (6, num_vars, num_vars) or not torch.isfinite(rel).all():
            raise ValueError('Expected six finite target/source relation override matrices')
        untyped = torch.zeros(num_vars, num_vars)
        digest = sha256(rel.contiguous().numpy().tobytes()).hexdigest()
        return rel, untyped, torch.tensor([1, 2, 0, *bytes.fromhex(digest)], dtype=torch.uint8)
    if isinstance(variable_graph, VariableGraphInput) and variable_graph.origin == 1:
        view = {1: 'hybrid', 2: 'continuous'}[variable_graph.view_code]
    graph = resolve_variable_graph(variable_graph, num_vars, view=view)
    source = build_yeast_relations(system_id='fission' if num_vars==16 else 'budding')
    rel_np, _ = source.matrices(view, self_loops='none')
    support = graph.matrix.ne(0)
    rel = torch.tensor(rel_np, dtype=torch.float32) * support[None]
    # All source dependencies absent from the typed overlays remain explicit.
    untyped = (support & ~rel.bool().any(0)).float()
    digest = source.fingerprint if graph.origin == 1 else sha256(
        (source.fingerprint + graph.digest).encode()).hexdigest()
    metadata = torch.tensor([1, graph.origin, graph.view_code, *bytes.fromhex(digest)], dtype=torch.uint8)
    return rel, untyped, metadata


def register_relation_graph(module, num_vars=39, *, view='hybrid', normalized=False,
                            variable_graph=None, rel_graphs=None):
    rel, untyped, metadata = relation_inputs(num_vars, view=view, variable_graph=variable_graph,
                                            rel_graphs=rel_graphs)
    if normalized:
        rel = rel / rel.sum(-1, keepdim=True).clamp(min=1.0)
        untyped = untyped / untyped.sum(-1, keepdim=True).clamp(min=1.0)
    module.register_buffer('rel_graphs', rel)
    module.register_buffer('untyped_graph', untyped)
    attach_fixed_graph(module, 'relations', metadata, ('rel_graphs', 'untyped_graph'),
                       optional_legacy_buffers=('untyped_graph',))
    module._source_relations_normalized = normalized
    module._source_relations_num_vars = num_vars
    module._source_relations_explicit_override = int(metadata[1]) == 2


def configure_relation_view(model, view):
    """Bind the registry CLI view without changing every public constructor."""
    for module in model.modules():
        if not hasattr(module, '_source_relations_normalized') or module._source_relations_explicit_override:
            continue
        rel, untyped, metadata = relation_inputs(module._source_relations_num_vars, view=view)
        if module._source_relations_normalized:
            rel = rel / rel.sum(-1, keepdim=True).clamp(min=1.0)
            untyped = untyped / untyped.sum(-1, keepdim=True).clamp(min=1.0)
        module.rel_graphs.copy_(rel.to(module.rel_graphs))
        module.untyped_graph.copy_(untyped.to(module.untyped_graph))
        module._relations_graph_metadata.copy_(metadata.to(module._relations_graph_metadata))
        contract = module._fixed_graph_contracts['relations']
        contract['metadata'] = metadata
        contract['buffers'] = {'rel_graphs': rel.clone(), 'untyped_graph': untyped.clone()}


def untyped_message(features, relation_weights, flux_weights, untyped_graph):
    """Class-agnostic mean transform/gate, reusing the six learned transforms."""
    transformed = torch.matmul(features, relation_weights.mean(0))
    weight = flux_weights.mean(-1).view(len(features), 1, 1, 1)
    return weight * torch.matmul(untyped_graph, transformed)


@dataclass
class RelationBundle:
    circuit_motifs: torch.Tensor
    gene_bipartite: torch.Tensor
    device: str = 'cpu'

    def register(self, module, num_vars=39):
        register_relation_graph(module, num_vars)
        for name in ('rel_graphs', 'untyped_graph', '_relations_graph_metadata'):
            setattr(module, name, getattr(module, name).to(self.device))
        module.register_buffer('circuit_motifs', self.circuit_motifs)
        module.register_buffer('gene_bipartite', self.gene_bipartite)
