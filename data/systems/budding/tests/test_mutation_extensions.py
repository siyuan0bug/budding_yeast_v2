"""Mechanism and sampler regressions; run with unittest from the project root."""
import unittest
from types import SimpleNamespace

import numpy as np

from .. import catalog, ode, parameters, states, topology
from ..distributions import mass as mass_distribution
from ..distributions import mutant_aware as ma_distribution
from ..distributions.quantiles import sample_scaling_factor_sobol
from ..mutations import mass, mutant_aware
from ..samplers import common, mass as structured, random, wt_blind


def edit(gene, mode, alpha, profile=mutant_aware, seed=42):
    params = parameters.get_default_params()
    overrides = {}
    profile.apply_gene_mutation(gene, mode, alpha, params, params.copy(), overrides,
                                rng=np.random.RandomState(seed))
    return overrides


class MutationExtensions(unittest.TestCase):
    def test_all_legal_controls_have_complete_conflict_footprints(self):
        pairs = {(g, m) for g in catalog.GENE_NAMES
                 for m in catalog.GROUP_BY_NAME[g]['modes']}
        self.assertEqual(pairs, set(catalog.TOUCH))
        for profile, distribution in ((mass, mass_distribution), (mutant_aware, ma_distribution)):
            for gene, mode in sorted(pairs):
                alpha = distribution.sample_scaling_factor(mode, gene, rng=np.random.RandomState(123))
                with self.subTest(profile=profile.__name__, gene=gene, mode=mode):
                    # CDC14-ts has two legal stochastic branches; TOUCH is their union.
                    footprints = [set(edit(gene, mode, alpha, profile, seed)) for seed in [0, 42]]
                    for actual in footprints:
                        self.assertLessEqual(actual, catalog.touch_set(gene, mode))
                    self.assertEqual(set.union(*footprints), catalog.touch_set(gene, mode))

    def test_invalid_mechanisms_are_rejected_before_edits(self):
        for gene, mode in [('CDC5_Proxy', 'KO'), ('SCC1', 'scc1-RR'),
                           ('CDC28', 'ts'), ('CLB2', 'CA'), ('SBF_MBF', 'CA'),
                           ('PPX', 'MC')]:
            for profile in (mass, mutant_aware):
                params, overrides = parameters.get_default_params(), {}
                before = params.copy()
                with self.assertRaises(ValueError):
                    profile.apply_gene_mutation(gene, mode, 1., params, params, overrides)
                self.assertEqual(params, before)
                self.assertEqual(overrides, {})

    def test_men_damp_is_unavailable_in_every_profile(self):
        for profile in (mass, mutant_aware):
            for gene in ['TEM1', 'CDC15']:
                self.assertNotIn('DAmP', catalog.GROUP_BY_NAME[gene]['modes'])
                self.assertNotIn((gene, 'DAmP'), catalog.TOUCH)
                for alpha in [.05, .2, .4]:
                    with self.assertRaises(ValueError):
                        edit(gene, 'DAmP', alpha, profile)
                for distribution in (mass_distribution, ma_distribution):
                    with self.assertRaises(ValueError):
                        distribution.sample_scaling_factor('DAmP', gene)
                with self.assertRaises(ValueError):
                    sample_scaling_factor_sobol('DAmP', gene, .5)

    def test_overexpression_retains_benchmark_initial_protocol(self):
        for profile in (mass, mutant_aware):
            for gene, initial in [('TEM1', 'init_Tem1'), ('CDC15', 'init_Cdc15')]:
                for mode in ['MC', 'GAL-OE']:
                    self.assertNotIn(initial, edit(gene, mode, 5., profile))

    def test_ppx_has_phosphatase_not_cdc5_kinase_semantics(self):
        params, y = parameters.get_default_params(), states.initial_conds()
        base = ode.eqns(0., y, params)
        ko = ode.eqns(0., y, dict(params, **edit('PPX', 'KO', 1.)))
        oe = ode.eqns(0., y, dict(params, **edit('PPX', 'GAL-OE', .3)))
        activity = ode.eqns(0., y, dict(params, **edit('PPX', 'ts', .1)))
        self.assertLess(ko[states.nPPX], base[states.nPPX])
        self.assertGreater(oe[states.nPPX], base[states.nPPX])
        # Less PPX catalytic activity reduces formation of dephosphorylated Net1.
        self.assertLess(activity[states.nNET1], base[states.nNET1])

    def test_transcription_controls_reduce_their_actual_targets(self):
        params, y = parameters.get_default_params(), states.initial_conds()
        base = ode.eqns(0., y, params)
        for gene, targets in [('Mcm1', [states.nCLB2, states.nCDC20T, states.nSWI5T, states.nPDS1]),
                              ('SBF_MBF', [states.nCLN2, states.nCLB5, states.nPDS1])]:
            changed = ode.eqns(0., y, dict(params, **edit(gene, 'KO', 1.)))
            for target in targets:
                self.assertLess(changed[target], base[target])

    def test_restored_quantiles_and_wt_blind_footprints(self):
        self.assertAlmostEqual(sample_scaling_factor_sobol('GAL-OE', 'PPX', .5), np.sqrt(.2 * .6))
        self.assertAlmostEqual(sample_scaling_factor_sobol('db', 'PPX', .5), .005)
        self.assertGreater(sample_scaling_factor_sobol('ts', 'Mcm1', .5), 0.)
        self.assertIs(wt_blind.TOUCH, catalog.TOUCH)
        self.assertEqual(wt_blind.GAL_ALPHA_BIO['PPX'], (.2, .6))
        for gene in ['TEM1', 'CDC15']:
            self.assertNotIn((gene, 'DAmP'), wt_blind.FOLD_MODES)
        for gene in catalog.RESTORED_FUNCTIONAL_MODULES:
            self.assertTrue(topology.ADJ[gene])

    def test_db_is_not_in_the_loss_biased_input_pool(self):
        saved = np.random.get_state()
        try:
            np.random.seed(42)
            for _ in range(300):
                result = structured._assign_convergent_modes(['PPX', 'CLB2', 'CDC14'])
                self.assertNotIn('db', [mode for _, mode in result[:2]])
        finally:
            np.random.set_state(saved)

    def test_wt_blind_builds_restored_controls_with_consistent_pool_initials(self):
        backend = SimpleNamespace(distribution_profile='mutant_aware',
            apply_gene_mutation=mutant_aware.apply_gene_mutation,
            sample_scaling_factor=ma_distribution.sample_scaling_factor)
        saved = np.random.get_state()
        old_index = wt_blind._L1_IDX['i']
        try:
            np.random.seed(2810)
            wt_blind._L1_IDX['i'] = 0
            seen = set()
            samplers = wt_blind.make_samplers()
            for _ in wt_blind._L1_GRID:
                row = wt_blind.build_candidate(1, samplers, ma=backend)
                self.assertIsNotNone(row)
                self.assertTrue(np.isfinite(list(row['ovr'].values())).all())
                seen.update(row['genes_modes'])
                for total, initial in [('Tem1T', 'init_Tem1'), ('Cdc15T', 'init_Cdc15')]:
                    if initial in row['ovr']:
                        self.assertLessEqual(row['ovr'][initial], row['ovr'][total])
            self.assertEqual(seen, set(catalog.TOUCH))
        finally:
            np.random.set_state(saved)
            wt_blind._L1_IDX['i'] = old_index

    def test_random_and_graph_samplers_emit_only_legal_nonconflicting_controls(self):
        def apply(*args, **kwargs):
            return mutant_aware.apply_gene_mutation(*args, **kwargs)
        backend = SimpleNamespace(
            get_default_params=parameters.get_default_params, apply_gene_mutation=apply,
            sample_scaling_factor=ma_distribution.sample_scaling_factor,
            GENE_NAMES=catalog.GENE_NAMES, GROUP_BY_NAME=catalog.GROUP_BY_NAME,
            COMPOSITE_ELIGIBLE=catalog.COMPOSITE_ELIGIBLE,
            _resolve_conflicts=common._resolve_conflicts)
        from ..samplers.rescue import parse_pairs
        saved = np.random.get_state()
        try:
            rng = np.random.RandomState(2801)
            np.random.seed(2801)
            for level in [1, 2, 3]:
                for _ in range(100):
                    for overrides, desc in [random.sample_pure_random_mutant(level, rng, ma=backend),
                                            structured.sample_mass_mutant(level, ma=backend)]:
                        touched = set()
                        pairs = parse_pairs(desc)
                        self.assertTrue(pairs)
                        for gene, mode in pairs:
                            catalog.require_mutation(gene, mode)
                            footprint = catalog.touch_set(gene, mode)
                            self.assertFalse(touched & footprint)
                            touched.update(footprint)
                        if any(mode == 'GAL-OE' for _, mode in pairs):
                            self.assertEqual(overrides['mdt'], 150.)
        finally:
            np.random.set_state(saved)


if __name__ == '__main__':
    unittest.main()
