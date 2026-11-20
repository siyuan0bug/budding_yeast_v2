"""Source topology regressions: synthetic fixtures/public WT, never test mutants."""
import ast
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from budding_yeast_v2.data.systems.budding.graph import read_budding_source

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from budding_yeast_v2.data.yeast_graph import (
    DEFAULT_SOURCE, GraphExtractionError, VIEWS, build_yeast_graph,
)
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_yeast_graph import compare_legacy, main


TOY_SOURCE = '''
raise RuntimeError("source must never be executed")
(nA, nB, nC) = range(3)
REAL_MUTANTS = {"hidden": 999}
def helper(x):
    z = x * 2
    return z
def eqns(t, y, p):
    dydt = np.zeros(3)
    (A, B, C) = y
    if p.get('mode'):
        rate = helper(A)
    else:
        rate = B
    dydt[0] = rate - A
    dydt[1] = p['k'] * C
    dydt[2] = -C
    return dydt
def event1(t, y, p): return y[nA] - 1
event1.terminal = True
event1.direction = 1
def simulate_mutant(name, ovr, tf=210, ns=500):
    sol = solve_ivp(eqns, (0, tf), y0, args=(p,), events=[event1])
    ie = next((i+1 for i, te in enumerate(sol.t_events) if len(te) > 0), -1)
    if ie == 1:
        y0[nB] = y0[nC]
        y0[nC] = y0[nB]
'''


@pytest.fixture(scope="module")
def graph():
    return build_yeast_graph()


def toy(tmp_path, old=None, new=None):
    source = TOY_SOURCE
    if old is not None:
        assert old in source
        source = source.replace(old, new)
    path = tmp_path / "toy_source.py"
    path.write_text(source, encoding="utf-8")
    return build_yeast_graph(path)


def test_default_counts_and_potential_not_signed(graph):
    summary = graph.summary()
    assert summary["num_variables"] == 39
    assert summary["nonself_edges"] == {"continuous": 218, "event_trigger": 12,
                                       "reset_value": 0, "event": 12, "hybrid": 226}
    assert summary["event_added_nonself_edges"] == 8
    assert summary["orientation"] == "target_source"
    assert "not_encoded" in summary["sign_policy"]
    assert len(summary["source_sha256"]) == 64
    assert len(graph.equations) == 39
    assert graph.adjacency("continuous").sum() == 254  # 36 physical self-dependencies.


@pytest.mark.parametrize("view", VIEWS)
def test_matrix_views_are_fresh_boolean_derivations(graph, view):
    a = graph.adjacency(view)
    assert a.shape == (39, 39) and a.dtype == np.bool_
    expected = a.copy()
    a[:] = False
    np.testing.assert_array_equal(graph.adjacency(view), expected)
    assert not graph.adjacency(view, self_loops="none").diagonal().any()
    assert graph.adjacency(view, self_loops="all").diagonal().all()


def test_views_preserve_edge_types_and_no_forced_checkpoint_loops(graph):
    a = graph.adjacency("continuous")
    e = graph.adjacency("event")
    np.testing.assert_array_equal(graph.adjacency("hybrid"), a | e)
    np.testing.assert_array_equal(e, graph.adjacency("event_trigger") | graph.adjacency("reset_value"))
    assert not a[36:39].any()  # These states have zero continuous derivatives.
    assert not graph.adjacency("hybrid").diagonal()[36:39].any()
    assert graph.adjacency("hybrid", self_loops="all").diagonal()[36:39].all()
    with pytest.raises(ValueError):
        graph.adjacency("signed")
    with pytest.raises(ValueError):
        graph.adjacency("continuous", self_loops="sometimes")
    with pytest.raises(FrozenInstanceError):
        graph.source_sha256 = "changed"


def test_guards_resets_and_no_indirect_edges(graph):
    ix = {s: i for i, s in enumerate(graph.variable_names)}
    expected = [(['CLB2'], ['MASS', 'CLB2', 'BUD', 'SPN', 'lte1'], -1),
                (['CLB2', 'CLB5'], ['CLB5', 'ORI'], -1),
                (['ORI'], ['ORI', 'Vi20', 'BUB2'], 1),
                (['SPN'], ['SPN', 'Vi20', 'lte1', 'BUB2'], 1)]
    for record, (sources, targets, direction) in zip(graph.events, expected):
        assert [graph.variable_names[i] for i in record.guard_sources] == sources
        assert [graph.variable_names[r.target] for r in record.resets] == targets
        assert record.direction == direction and record.terminal
    a, e = graph.adjacency("continuous"), graph.adjacency("event")
    assert a[ix['ORI'], ix['CLB2']] and not a[ix['CLB2'], ix['ORI']]
    assert not a[ix['CDC20'], ix['CLB2']]
    assert a[ix['IEP'], ix['CLB2']] and a[ix['CDC20'], ix['IEP']]
    assert e[ix['Vi20'], ix['ORI']] and not a[ix['Vi20'], ix['ORI']]
    # Event timing can affect CDC20 without a direct continuous/guard-reset edge.
    assert not graph.adjacency("hybrid")[ix['CDC20'], ix['ORI']]


def test_provenance_expands_auxiliary_functions_and_excludes_test_table(graph):
    lines = dict(graph.source_statements)
    equation = graph.equations[17]
    assert graph.variable_names[equation.target] == 'SWI5'
    snippets = '\n'.join(lines[i] for i in equation.lines)
    assert 'MCM1 = GK' in snippets and 'BB = Vi - Va' in snippets
    assert 'dydt[17]' in snippets
    assert all('REAL_MUTANTS' not in text for text in lines.values())
    # Solver guards can shift source lines without changing the ODE/reset kernel.
    tree = ast.parse(read_budding_source(DEFAULT_SOURCE))
    assert all(i > 0 for i in lines)
    assert 'REAL_MUTANTS' not in read_budding_source(DEFAULT_SOURCE)
    assert 'Vi20_active' in graph.events[2].resets[1].expression
    assert graph.events[2].resets[1].sources == ()  # Reset value is parameter-only.


@pytest.mark.parametrize("term", [' + C - C', ' + p["k"]*C - C*p["k"]',
                                   ' + (C + A) - A - C', ' + 0*C'])
def test_simple_algebraic_cancellations_are_not_edges(tmp_path, term):
    g = toy(tmp_path, 'dydt[0] = rate - A', 'dydt[0] = rate - A' + term)
    assert not g.adjacency("continuous")[0, 2]


def test_parameter_branches_union_and_sequential_reset_reads(tmp_path):
    g = toy(tmp_path)
    a = g.adjacency("continuous")
    assert a[0, 0] and a[0, 1] and not a[0, 2]
    assert not a[1, 0] and a[1, 2]
    # Both final B and C read ORIGINAL C, not ORIGINAL B after B was overwritten.
    assert g.events[0].resets[0].sources == (2,)
    assert g.events[0].resets[1].sources == (2,)
    assert g.adjacency("event_trigger")[1, 0]
    assert g.adjacency("reset_value")[1, 2]
    assert not g.adjacency("reset_value")[2, 1]


def test_irrelevant_test_metadata_does_not_change_fingerprint(tmp_path):
    a = toy(tmp_path)
    b = toy(tmp_path, 'REAL_MUTANTS = {"hidden": 999}', 'REAL_MUTANTS = {"different": -123456}')
    assert a.source_sha256 == b.source_sha256
    np.testing.assert_array_equal(a.adjacency("hybrid"), b.adjacency("hybrid"))


@pytest.mark.parametrize("old,new", [
    ('return z', 'return z + 1'),
    ('dydt[1] = p[\'k\'] * C', 'dydt[1] = p[\'k\'] * A'),
    ('y[nA] - 1', 'y[nB] - 1'),
    ('event1.direction = 1', 'event1.direction = -1'),
    ('y0[nB] = y0[nC]', 'y0[nB] = y0[nA]'),
])
def test_relevant_source_changes_invalidate_fingerprint(tmp_path, old, new):
    a = toy(tmp_path)
    b = toy(tmp_path, old, new)
    assert a.source_sha256 != b.source_sha256


@pytest.mark.parametrize("old,new", [
    ('dydt[1] = p[\'k\'] * C', 'dydt[1] = np.sin(C)'),
    ('dydt[1] = p[\'k\'] * C', 'dydt[1] = unknown_name'),
    ('dydt[1] = p[\'k\'] * C', 'for j in range(3):\n        dydt[j] = C'),
    ('dydt[1] = p[\'k\'] * C', 'dydt[3] = C'),
    ('if p.get(\'mode\'):', 'if A > 0:'),
    ("p.get('mode')", "p.get('mode', A)"),
    ('dydt[1] = p[\'k\'] * C', 'dydt[1] = C if A > 0 else B'),
    ('dydt[1] = p[\'k\'] * C', 'dydt[1] = np.maximum(A, B, C)'),
    ('dydt[1] = p[\'k\'] * C', 'dydt[1] = __import__("os")'),
    ('(nA, nB, nC) = range(3)', '(nB, nA, nC) = range(3)'),
    ('event1.terminal = True', 'event1.terminal = False'),
    ('event1.direction = 1', 'event1.direction = 2'),
    ('if ie == 1:', 'if ie == 2:'),
    ('events=[event1]', 'events=[event1,event1]'),
    ('i+1 for i, te', 'i+2 for i, te'),
    ('enumerate(sol.t_events)', 'enumerate(reversed(sol.t_events))'),
    ('if len(te) > 0', 'if i == 0'),
    ('y0[nB] = y0[nC]', 'y0.sort()'),
    ('y0[nB] = y0[nC]', 'bd.append(y0.sort())'),
    ('return z', 'return z\nnA = 2'),
    ('return dydt', 'return y'),
    ('return dydt', 'return dydt\n    dydt[1] = A'),
])
def test_unsupported_or_inconsistent_source_fails_closed(tmp_path, old, new):
    with pytest.raises(GraphExtractionError):
        toy(tmp_path, old, new)


def test_boolean_condition_is_not_silently_always_true(tmp_path):
    g = toy(tmp_path, "p.get('mode')", "p.get('mode') and t >= 1")
    assert g.adjacency("continuous")[0, 1]  # The false branch must survive.


def test_default_source_not_imported_and_cwd_independent(tmp_path):
    code = '''
import sys
from budding_yeast_v2.data.yeast_graph import build_yeast_graph
g = build_yeast_graph()
assert "budding_yeast_v2.data.mutant_aware" not in sys.modules
assert g.adjacency("continuous", self_loops="none").sum() == 218
print(g.source_sha256)
'''
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT.parent))
    outputs = [subprocess.check_output([sys.executable, '-c', code], cwd=d, env=env, text=True).strip()
               for d in (ROOT, ROOT.parent, tmp_path)]
    assert len(set(outputs)) == 1


@pytest.fixture(scope="module")
def public_ode_only():
    """Compile ONLY public ODE/default functions, never the source module/test table."""
    tree = ast.parse(read_budding_source(DEFAULT_SOURCE))
    eq = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eqns')
    unpack = next(n for n in eq.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple))
    names = [n.id for n in unpack.targets[0].elts]
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in {'get_default_params', 'initial_conds', 'GK', 'eqns'}]
    namespace = {'np': np, **{'n' + s: i for i, s in enumerate(names)}}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'public_ode_only', 'exec'), namespace)
    return namespace


def numerical_jacobian(ode, y, p):
    j = np.zeros((39, 39))
    for k in range(39):
        step = 1e-6 * max(1., abs(y[k]))
        a, b = y.copy(), y.copy()
        a[k] += step
        b[k] -= step
        j[:, k] = (ode(0., a, p) - ode(0., b, p)) / (2 * step)
    return j


def test_source_support_matches_independent_numerical_jacobians(graph, public_ode_only):
    p = public_ode_only['get_default_params']()
    y = public_ode_only['initial_conds']()
    observed = np.zeros((39, 39), dtype=bool)
    rng = np.random.default_rng(9102)
    for scale in [np.ones(39), *rng.uniform(.8, 1.2, (4, 39))]:
        j = numerical_jacobian(public_ode_only['eqns'], y * scale, p)
        assert np.isfinite(j).all()
        observed |= np.abs(j) > 1e-7
    np.testing.assert_array_equal(observed, graph.adjacency("continuous"))


def test_same_edge_can_change_sign_without_changing_structure(graph, public_ode_only):
    p = public_ode_only['get_default_params']()
    y = public_ode_only['initial_conds']()
    positive = numerical_jacobian(public_ode_only['eqns'], y, p)[17, 2]
    y[2] = .5
    negative = numerical_jacobian(public_ode_only['eqns'], y, p)[17, 2]
    assert positive == pytest.approx(.7610492276, rel=1e-6)
    assert negative == pytest.approx(-.0396495712, rel=1e-6)
    assert graph.adjacency("continuous")[17, 2]


def test_legacy_assets_untouched_and_diagnostic_only(graph):
    paths = [ROOT / 'data' / n for n in ('yeast_signed_adjacency_matrix.npy',
             'yeast_boolean_adjacency_matrix.npy', 'yeast_multirelational_graphs.npz')]
    if not all(p.exists() for p in paths):
        pytest.skip('Optional ignored legacy graph assets are not installed')
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    result = compare_legacy(graph, ROOT / 'data')
    assert all(result[p.name]['support_matches_with_neural_self_loops'] for p in paths[:2])
    r = result[paths[2].name]
    assert (r['unique_nonself_edges'], r['matches_continuous'],
            r['missing_continuous'], r['outside_continuous']) == (124, 114, 104, 10)
    assert 'gene_bipartite' in r['other_fields_preserved']
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


def test_cli_default_no_writes_and_optional_record_export(tmp_path, capsys):
    source = tmp_path / 'source.py'
    source.write_text(TOY_SOURCE, encoding='utf-8')
    before = set(tmp_path.iterdir())
    summary = main(['--source', str(source)])
    assert before == set(tmp_path.iterdir())
    assert summary['num_variables'] == 3
    output = tmp_path / 'graph.json'
    main(['--source', str(source), '--output', str(output)])
    saved = json.loads(output.read_text())
    assert saved['source_sha256'] == summary['source_sha256']
    assert 'equations' in saved and 'event_records' in saved
    assert 'adjacency' not in saved and 'signs' not in saved
    with pytest.raises(FileExistsError):
        main(['--source', str(source), '--output', str(output)])
    with pytest.raises(SystemExit):
        main(['--source', str(source), '--output', str(tmp_path / 'old.npy')])
    capsys.readouterr()


def test_missing_legacy_assets_are_not_replaced_with_identity(graph, tmp_path):
    result = compare_legacy(graph, tmp_path)
    assert all(entry['status'] == 'missing' for entry in result.values())
    assert not list(tmp_path.iterdir())


def test_legacy_export_requires_explicit_new_file(tmp_path):
    from budding_yeast_v2.data.build_bio_multirelational_graphs import (
        main as legacy_main, build_relational_graphs,
    )
    with pytest.raises(SystemExit):
        legacy_main([])
    with pytest.raises(SystemExit):
        legacy_main(['--output', str(tmp_path / 'graph.npz')])
    output = tmp_path / 'legacy.npz'
    legacy_main(['--legacy-export', '--output', str(output)])
    before = hashlib.sha256(output.read_bytes()).hexdigest()
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['rel_graphs'], build_relational_graphs())
        assert data['gene_bipartite'].shape == (27, 39)
    with pytest.raises(FileExistsError):
        legacy_main(['--legacy-export', '--output', str(output)])
    assert before == hashlib.sha256(output.read_bytes()).hexdigest()
