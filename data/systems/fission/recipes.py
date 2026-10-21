"""Fission dataset recipes; current exact L1/L2/L3 quotas; original generators remain at data root."""
from ..contracts import Recipe
from ..sampling_policy import STATIC_RATIOS, POOL_RATIOS


RECIPES = {
    'fission_random': Recipe('fission','fission_random',STATIC_RATIOS,'fission','fission',default_samples=1000,default_seed=20000),
}
