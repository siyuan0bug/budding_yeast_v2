"""ODE-backed relation overlays; topology is never supplied by the old NPZ.

Biochemical labels are explicit, reviewable annotations on source flux terms,
not a consequence of graph support or a claim of uniquely identifiable biology.
Dependencies not covered by classified terms have an untyped route. No
trajectory, parameter default, real-mutant table, or legacy edge list is read.
"""
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json

import numpy as np

from .yeast_graph import build_yeast_graph, YeastGraph


RELATION_NAMES = ('complex_exchange', 'phosphorylation', 'dephosphorylation',
                  'production', 'degradation', 'activity_switch')
# Annotate the rate/parameter explicitly present in a source flux term. Do not
# classify by the names of the source/target species or by derivative sign.
RATE_ANCHORS = (
    (), ('Vkpc1', 'Vkpf6', 'Vkpnet', 'Vaiep'), ('Vppc1', 'Vppf6', 'Vppnet', 'Vacdh'),
    ('actual_ksc1p',), ('Vdb2', 'Vdb5', 'Vdpds', 'Vdppx'), ('Vicdh',),
)
PARAMETER_ANCHORS = (
    ('kasb2', 'kasb5', 'kasf2', 'kasf5', 'kdib2', 'kdib5', 'kdif2', 'kdif5',
     'kasrent', 'kasrentp', 'kdirent', 'kdirentp', 'kasesp', 'kdiesp'),
    ('kiswi',), ('kaswi',),
    ('ksn2p', 'ksn2pp', 'ksb2p', 'ksb2pp', 'ksb5p', 'ksb5pp', 'ksc1p', 'ksc1pp',
     'ksf6p', 'ksf6pp', 'ksf6ppp', 'ksswip', 'ksswipp', 'ks20p', 'ks20pp',
     'kscdh', 'ks14', 'ksnet', 'ksppx', 'kspdsp', 'ks1pdspp', 'ks2pdspp',
     'ksori', 'ksbud', 'ksspn'),
    ('kdn2', 'kd3c1', 'kd3f6', 'kdswi', 'kd20', 'kdcdh', 'kd14', 'kdnet',
     'kdori', 'kdbud', 'kdspn'),
    ('kiiep', 'ka20p', 'ka20pp', 'ka15p', 'ka15pp', 'ka15P', 'ki15', 'jatem', 'Jitem'),
)
RULE_VERSION = 1


@dataclass(frozen=True)
class RelationTermRecord:
    target: int
    sources: tuple[int, ...]
    relation: str | None
    anchors: tuple[str, ...]
    expression: str
    line: int


@dataclass(frozen=True)
class YeastRelations:
    graph: YeastGraph
    records: tuple[RelationTermRecord, ...]
    fingerprint: str

    def matrices(self, view='hybrid', *, self_loops='all'):
        """Fresh six overlays + untyped matrix, always [target,source].

        Their union is EXACTLY the selected source graph. Neural self-loops are
        placed only in the untyped route, never six duplicated identity graphs.
        """
        support = self.graph.adjacency(view, self_loops=self_loops)
        continuous = self.graph.adjacency('continuous', self_loops='physical')
        rel = np.zeros((6, len(support), len(support)), dtype=bool)
        for record in self.records:
            if record.relation is not None:
                rel[RELATION_NAMES.index(record.relation), record.target, list(record.sources)] = True
        rel &= continuous[None] & support[None]
        if self_loops == 'all':
            for matrix in rel:
                np.fill_diagonal(matrix, False)
        untyped = support & ~rel.any(0)
        return rel, untyped

    def summary(self, view='hybrid'):
        rel, untyped = self.matrices(view, self_loops='none')
        return {'source_sha256': self.graph.source_sha256, 'contract_sha256': self.fingerprint,
                'orientation': 'target_source', 'view': view, 'relation_names': list(RELATION_NAMES),
                'typed_nonself_counts': rel.sum((1, 2)).tolist(),
                'typed_union_nonself': int(rel.any(0).sum()),
                'untyped_nonself': int(untyped.sum()),
                'total_nonself': int((rel.any(0) | untyped).sum()),
                'classification': 'explicit source-term annotations; not unique biological identification'}


@lru_cache(maxsize=4)
def _build(graph, rules, rate_anchors=RATE_ANCHORS, parameter_anchors=PARAMETER_ANCHORS):
    records = []
    for term in graph.terms:
        names, parameters = set(term.names), set(term.parameters)
        matches = [(i, tuple(sorted((names & set(rate_anchors[i])) |
                                   (parameters & set(parameter_anchors[i]))))) for i in range(6)]
        matches = [(i, anchors) for i, anchors in matches if anchors]
        # A product involving two different mechanisms stays untyped. Nothing
        # is assigned on the basis of an arbitrary priority between labels.
        relation = RELATION_NAMES[matches[0][0]] if len(matches) == 1 else None
        anchors = tuple(a for _, group in matches for a in group)
        records.append(RelationTermRecord(term.target, term.sources, relation, anchors,
                                         term.expression, term.line))
    ledger = [(r.target, r.sources, r.relation, r.anchors, r.expression) for r in records]
    digest = sha256(json.dumps((graph.source_sha256, rules, ledger),
                              separators=(',', ':')).encode()).hexdigest()
    return YeastRelations(graph, tuple(records), digest)


def build_yeast_relations(source_path=None, *, system_id='budding'):
    if system_id=='fission':
        rates=((),(),(),(),('k2','k2c','k4'),('kWee','kC25'))
        parameters=(('kJ','kJR'),('kp','ki'),('kpp_','kpp','kiR'),('k1','k3','k5','kas','mu','K'),
                    ('k6_','k6','kad'),('kste9r_','kste9r','kste9','kMR_','kMR','kM','kWR_','kWR','kW',
                                      'k25','k25R_','k25R','kaa_','kaa','kai','kS'))
        rules=json.dumps(('fission_v1',RELATION_NAMES,rates,parameters),separators=(',',':'))
        return _build(build_yeast_graph(source_path,system_id=system_id),rules,rates,parameters)
    if system_id!='budding':raise ValueError(f'Unknown biological system: {system_id}')
    rules = json.dumps((RULE_VERSION, RELATION_NAMES, RATE_ANCHORS, PARAMETER_ANCHORS), separators=(',', ':'))
    return _build(build_yeast_graph(source_path), rules)
