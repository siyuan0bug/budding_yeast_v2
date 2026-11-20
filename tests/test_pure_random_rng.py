"""Single-stream sampler regressions; WT/artificial fixtures, no real ODE runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import Future

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from budding_yeast_v2.data import generate_pure_random_dataset as pure

ma = pure.ma
FACTORIES = [np.random.RandomState, np.random.default_rng]
ALL_MODES = [(gene, mode) for gene in ma.GENE_NAMES for mode in ma.GROUP_BY_NAME[gene]['modes']]


@pytest.fixture(autouse=True)
def preserve_global_state():
    original = np.random.get_state()
    yield
    np.random.set_state(original)


def wt_params():
    values = ma.get_default_params()
    values.update({'init_CDH1T': 1., 'init_CDH1': .9304992, 'init_BUB2': .2})
    return values


def candidates(factory, seed, ko_interval=False, count=256):
    rng = factory(seed)
    return [pure.sample_pure_random_mutant(1 + i % 4, rng, enable_ko_interval=ko_interval)
            for i in range(count)]


@pytest.mark.parametrize('factory', FACTORIES)
@pytest.mark.parametrize('ko_interval', [False, True])
def test_same_seed_is_independent_of_global_random_state(factory, ko_interval):
    np.random.seed(10)
    first = candidates(factory, 42, ko_interval)
    np.random.seed(11)
    np.random.uniform(size=157)
    second = candidates(factory, 42, ko_interval)
    assert first == second
    assert len(set(desc for _, desc in first)) > 30


@pytest.mark.parametrize('factory', FACTORIES)
def test_changing_local_seed_changes_candidates(factory):
    assert candidates(factory, 42) != candidates(factory, 43)


@pytest.mark.parametrize('factory', FACTORIES)
def test_explicit_rng_never_draws_from_global_helpers(monkeypatch, factory):
    rng = factory(42)
    def forbidden(*args, **kwargs):
        raise AssertionError('A downstream draw escaped to global np.random')
    for name in ('uniform', 'normal', 'choice', 'beta', 'rand', 'random', 'randint'):
        monkeypatch.setattr(np.random, name, forbidden)
    wt = wt_params()
    for gene, mode in ALL_MODES:
        alpha = ma.sample_scaling_factor(mode, gene, rng=rng)
        overrides = {}
        ma.apply_gene_mutation(gene, mode, alpha, wt, wt.copy(), overrides, rng=rng)
        ma._sample_mode(gene, rng=rng)
    # Explicitly exercise the custom stochastic branch not listed in every table.
    ma.apply_gene_mutation('ORI_BUD_SPN', 'perturb', 1., wt, wt.copy(), {}, rng=rng)
    ma._resolve_conflicts([('SBF_MBF', 'KO'), ('CLN2', 'KO'), ('CLB5', 'KO')], rng=rng)
    ma.rescue_collision_candidate({'ks20p': 0., 'ks20pp': 0., 'mdt': 90.},
                                  'CDC20[KO]+CDC14[ts]', wt, rng=rng)
    for i in range(128):
        pure.sample_pure_random_mutant(1 + i % 4, rng, enable_ko_interval=bool(i % 2))


@pytest.mark.parametrize('mode_freq', ['class', 'flat', 'role', 's5'])
def test_conflict_redraw_receives_the_same_stream(monkeypatch, mode_freq):
    monkeypatch.setattr(ma, 'MODE_FREQ', mode_freq)
    rng = np.random.RandomState(42)
    calls = []
    original = ma._sample_mode
    def tracked(gene, *, rng=None):
        calls.append(rng)
        return original(gene, rng=rng)
    monkeypatch.setattr(ma, '_sample_mode', tracked)
    result = ma._resolve_conflicts([('SBF_MBF', 'KO'), ('CLN2', 'KO'), ('CLB5', 'KO')], rng=rng)
    assert calls and all(item is rng for item in calls)
    touched = set()
    for gene, mode in result:
        keys = ma.touch_set(gene, mode)
        assert not (touched & keys)
        touched.update(keys)


@pytest.mark.parametrize('seed,continuous', [(0, True), (4, False)])
def test_rescue_branches_and_nested_helpers_use_one_stream(monkeypatch, seed, continuous):
    rng = np.random.RandomState(seed)
    scale_calls, mutation_calls = [], []
    scale, mutation = ma.sample_scaling_factor, ma.apply_gene_mutation
    def tracked_scale(mode, gene=None, *, rng=None):
        scale_calls.append(rng)
        return scale(mode, gene, rng=rng)
    def tracked_mutation(*args, rng=None):
        mutation_calls.append(rng)
        return mutation(*args, rng=rng)
    monkeypatch.setattr(ma, 'sample_scaling_factor', tracked_scale)
    monkeypatch.setattr(ma, 'apply_gene_mutation', tracked_mutation)
    overrides, desc = ma.rescue_collision_candidate({'mdt': 90.}, 'CDC20[KO]+CDC14[ts]', wt_params(), rng=rng)
    if continuous:
        assert scale_calls and mutation_calls
        assert all(item is rng for item in scale_calls + mutation_calls)
        assert overrides['mdt'] == 90. and 'CDC20[ts]' in desc
    else:
        assert not scale_calls and not mutation_calls
        assert overrides['mdt'] in (150., 160.) and '_MDT[' in desc


def test_omitted_rng_preserves_global_stream_calling_convention():
    wt = wt_params()
    def run(explicit):
        np.random.seed(29)
        kwargs = {'rng': np.random} if explicit else {}
        results = []
        for gene, mode in ALL_MODES:
            alpha = ma.sample_scaling_factor(mode, gene, **kwargs)
            overrides = {}
            ma.apply_gene_mutation(gene, mode, alpha, wt, wt.copy(), overrides, **kwargs)
            results.append((alpha, overrides, ma._sample_mode(gene, **kwargs)))
        for _ in range(20):
            results.append(ma.rescue_collision_candidate({'mdt': 90.}, 'CDC20[KO]+CDC14[ts]', wt, **kwargs))
        return results, np.random.get_state()
    omitted, state1 = run(False)
    explicit, state2 = run(True)
    assert omitted == explicit
    assert state1[0] == state2[0] and np.array_equal(state1[1], state2[1]) and state1[2:] == state2[2:]


def test_generator_tasks_and_rng_metadata_replay_without_real_ode(monkeypatch, tmp_path):
    submissions = []
    class ArtificialExecutor:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def submit(self, function, *args):
            name, overrides, horizon, steps = args
            submissions.append((name, dict(overrides)))
            future = Future()
            future.set_result((name, np.zeros((39, steps), dtype=np.float32), 'Artificial',
                               'Pattern_C: Exponential-Arrest (==0)'))
            return future
    # Never read test-mutant parameter rules or integrate their trajectories.
    monkeypatch.setattr(ma, 'REAL_MUTANTS', {1000: ('artificial_public_wt', {})})
    monkeypatch.setattr(pure.concurrent.futures, 'ProcessPoolExecutor', ArtificialExecutor)
    run_tasks, archives = [], []
    for i, global_seed in enumerate((10, 11)):
        calls = [0]
        def artificial_collision(overrides, defaults):
            calls[0] += 1
            return calls[0] == 1  # Exercise driver's rescue forwarding once.
        monkeypatch.setattr(ma, 'is_duplicate_real_mutant', artificial_collision)
        submissions.clear()
        np.random.seed(global_seed)
        name = f'artificial_run_{i}.npz'
        pure.generate_pure_random_dataset(num_samples=8, num_workers=1, seed=42,
                                         output_dir=str(tmp_path), output_name=name)
        run_tasks.append(list(submissions))
        with np.load(tmp_path / name) as archive:
            assert int(archive['sampling_seed']) == 42
            assert str(archive['sampling_rng_contract']) == pure.SAMPLING_RNG_CONTRACT
            archives.append({key: archive[key] for key in archive.files})
    assert run_tasks[0] == run_tasks[1]
    for key in archives[0]:
        assert np.array_equal(archives[0][key], archives[1][key]), key


@pytest.mark.parametrize('ko_interval', [False, True])
def test_fresh_process_hash_seeds_do_not_change_candidates(ko_interval):
    script = """
import hashlib, json, numpy as np
from budding_yeast_v2.data import generate_pure_random_dataset as pure
rng = np.random.RandomState(42)
values = [pure.sample_pure_random_mutant(1+i%4, rng, enable_ko_interval=KO_INTERVAL) for i in range(400)]
values += [pure.ma._resolve_conflicts([('SBF_MBF','KO'),('CLN2','KO'),('CLB5','KO')], rng=rng) for _ in range(40)]
print(hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest())
""".replace('KO_INTERVAL', repr(ko_interval))
    digests = []
    for hash_seed in ('1', '22', '321'):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT.parent, env=env,
                                text=True, capture_output=True, check=True, timeout=20)
        digest = result.stdout.strip()
        assert len(digest) == hashlib.sha256().digest_size * 2
        digests.append(digest)
    assert len(set(digests)) == 1
