"""Budding dataset recipes; current exact L1/L2/L3 quotas; original generators remain at data root."""
from ..contracts import Recipe
from ..sampling_policy import STATIC_RATIOS, POOL_RATIOS


RECIPES = {
    'budding_random': Recipe('budding','random',STATIC_RATIOS,dtype='float32',quota_rounding='round',shuffle_levels=False),
    'budding_random_ko_interval': Recipe('budding','random_ko_interval',STATIC_RATIOS,dtype='float32',quota_rounding='round',shuffle_levels=False),
    'budding_mass_random': Recipe('budding','mass_random',STATIC_RATIOS,'mass','mass'),
    'budding_mass_random_dist': Recipe('budding','mass_random_dist',STATIC_RATIOS,'mass','mutant_aware',dtype='float32',quota_rounding='second_remainder',shuffle_levels=False),
    'budding_mass': Recipe('budding','mass_motifs',STATIC_RATIOS,'mass','mass',notes='Historical targeted rescue motifs retained explicitly.'),
    'budding_mutant_aware': Recipe('budding','mass',STATIC_RATIOS),
    'budding_gibbs': Recipe('budding','gibbs',STATIC_RATIOS,dtype='float32'),
    'budding_gibbs_sobol': Recipe('budding','gibbs_sobol',STATIC_RATIOS,dtype='float32',quota_rounding='first_remainder',shuffle_levels=False),
    'budding_kdpp': Recipe('budding','kdpp',STATIC_RATIOS),
    'budding_kdpp_curvature': Recipe('budding','kdpp_curvature',STATIC_RATIOS,queue_multiplier=1.35),
    'budding_motif': Recipe('budding','motif',STATIC_RATIOS,dtype='float32',quota_rounding='floor',shuffle_levels=False),
    'budding_bams': Recipe('budding','bams',STATIC_RATIOS,dtype='float32'),
    'budding_wt_blind': Recipe('budding','wt_blind',STATIC_RATIOS,'mass','mass',quota_rounding='round',notes='Special QSS, inversion and ledger policy retains its original bands/topology.',mode_freq='flat',default_pilot=20000),
    'budding_pool': Recipe('budding','pool',POOL_RATIOS,quota_rounding='round',default_samples=1000000),
    'budding_twins': Recipe('budding','twins',STATIC_RATIOS,dtype='float32'),
    'budding_ko_mirror': Recipe('budding','ko_mirror',STATIC_RATIOS,dtype='float32'),
    'budding_bams_ablation': Recipe('budding','bams_ablation',STATIC_RATIOS,dtype='float32',notes='Targeted diagnostic ablation; requires a base BAMS dataset.',default_samples=50000,default_seed=20260915),
}
