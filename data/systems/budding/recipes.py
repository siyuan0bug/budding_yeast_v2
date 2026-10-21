"""Budding dataset recipes; values preserve the original defaults."""
from ..contracts import Recipe


RECIPES = {
    'budding_random': Recipe('budding','random',(.25,.25,.25,.25),dtype='float32',quota_rounding='round',shuffle_levels=False),
    'budding_random_ko_interval': Recipe('budding','random_ko_interval',(.25,.25,.25,.25),dtype='float32',quota_rounding='round',shuffle_levels=False),
    'budding_mass_random': Recipe('budding','mass_random',(.40,.30,.20,.10),'mass','mass'),
    'budding_mass_random_dist': Recipe('budding','mass_random_dist',(.25,.35,.25,.15),'mass','mutant_aware',dtype='float32',quota_rounding='second_remainder',shuffle_levels=False),
    'budding_mass': Recipe('budding','mass_motifs',(.25,.35,.25,.15),'mass','mass',notes='Historical targeted rescue motifs retained explicitly.'),
    'budding_mutant_aware': Recipe('budding','mass',(.25,.35,.25,.15)),
    'budding_gibbs': Recipe('budding','gibbs',(.30,.30,.20,.12,.08),dtype='float32'),
    'budding_gibbs_sobol': Recipe('budding','gibbs_sobol',(.30,.30,.20,.12,.08),dtype='float32',quota_rounding='first_remainder',shuffle_levels=False),
    'budding_kdpp': Recipe('budding','kdpp',(.25,.25,.25,.25)),
    'budding_kdpp_curvature': Recipe('budding','kdpp_curvature',(.20,.40,.25,.15),queue_multiplier=1.35),
    'budding_motif': Recipe('budding','motif',(.10,.30,.30,.21,.09),dtype='float32',quota_rounding='motif',shuffle_levels=False),
    'budding_bams': Recipe('budding','bams',(.35,.30,.22,.13),dtype='float32'),
    'budding_wt_blind': Recipe('budding','wt_blind',(.40,.30,.20,.10),'mass','mass',quota_rounding='round',notes='Special QSS, inversion and ledger policy retains its original bands/topology.',mode_freq='flat',default_pilot=20000),
    'budding_pool': Recipe('budding','pool',(.05,.25,.40,.30),quota_rounding='round',default_samples=1000000),
    'budding_twins': Recipe('budding','twins',(.25,.25,.25,.25),dtype='float32'),
    'budding_ko_mirror': Recipe('budding','ko_mirror',(.25,.25,.25,.25),dtype='float32'),
    'budding_bams_ablation': Recipe('budding','bams_ablation',(.35,.30,.22,.13),dtype='float32',notes='Targeted diagnostic ablation; requires a base BAMS dataset.',default_samples=50000,default_seed=20260915),
}
