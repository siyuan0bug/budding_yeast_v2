"""No inactive modules or unsupported same-gene compounds reach new samples."""
import itertools
import numpy as np
import pytest

from budding_yeast_v2.data.systems.budding import catalog
from budding_yeast_v2.data.systems.budding.quotas import allele_options
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.budding.samplers import common, bams, wt_blind
from budding_yeast_v2.data.systems.budding.samplers.rescue import rescue_budding, parse_pairs
from budding_yeast_v2.data.systems.contracts import Candidate
from budding_yeast_v2.data.systems.sampling_policy import valid_candidate


def test_disabled_modules_reject_before_rng_or_parameter_edit():
    assert len(set(catalog.GENE_NAMES) - {'MDT'}) == 19
    for profile in ('mass', 'mutant_aware'):
        backend = BuddingSystem(profile, profile)
        for group in catalog.DISABLED_BINDING_GROUPS.values():
            for mode in group['modes']:
                rng = np.random.RandomState(42)
                before = rng.get_state()
                defaults = backend.defaults()
                params, overrides = defaults.copy(), {}
                with pytest.raises(ValueError, match='Unsupported Budding'):
                    backend.apply_gene_mutation(group['name'], mode, .5, defaults, params, overrides, rng=rng)
                with pytest.raises(ValueError, match='Unsupported Budding'):
                    backend.sample_scaling_factor(mode, group['name'], rng=rng)
                assert params == defaults and not overrides
                assert np.array_equal(before[1], rng.get_state()[1]) and before[2:] == rng.get_state()[2:]


def test_only_real_supported_compounds_are_enumerated_or_expanded():
    actual = {(g, a) for g in catalog.GENE_NAMES for a in allele_options(g) if len(a) > 1}
    assert actual == set(catalog.COMPOSITE_REAL_EVIDENCE)
    assert len(actual) == 4
    for gene in catalog.GENE_NAMES:
        for mode in catalog.GROUP_BY_NAME[gene]['modes']:
            for seed in range(40):
                expanded = catalog.expand_composites([(gene, mode)], np.random.RandomState(seed))
                assert catalog.first_mutation_conflict(expanded) is None


@pytest.mark.parametrize('gene,modes', [
    ('CLB2', ('MC', 'db')), ('CLB5', ('MC', 'db')),
    ('SIC1', ('GAL-OE', 'PM')), ('SIC1', ('MC', 'db')), ('SIC1', ('MC', 'PM')),
    ('CLB2', ('GAL-MC', 'db')),
])
def test_generation_boundary_and_conflict_resolution_reject_bad_compounds(gene, modes):
    pairs = [(gene, m) for m in modes]
    desc = '+'.join(f'{g}[{m}]' for g, m in pairs)
    assert valid_candidate(Candidate(1, 'bad', {}, 1, metadata={'description': desc}), 'budding') is None
    backend = BuddingSystem()
    for seed in range(10):
        np.random.seed(seed)
        for resolved in (common._resolve_conflicts(pairs), wt_blind._resolve_conflicts(pairs),
                         bams.resolve_pair_conflicts(pairs, ma=backend)):
            assert catalog.first_mutation_conflict(resolved) is None


def test_rescue_keeps_same_gene_alleles_in_allowlist():
    backend = BuddingSystem()
    pairs = [('CLB2', 'GAL-OE'), ('CLB2', 'db')]
    overrides = {'mdt': 150.}
    for gene, mode in pairs:
        alpha = backend.sample_scaling_factor(mode, gene, rng=np.random.RandomState(1))
        backend.apply_gene_mutation(gene, mode, alpha, backend.defaults(), {}, overrides)
    for seed in range(80):
        _, desc = rescue_budding(overrides, 'CLB2[GAL-OE]+CLB2[db]', backend, np.random.RandomState(seed))
        assert catalog.first_mutation_conflict(parse_pairs(desc)) is None
