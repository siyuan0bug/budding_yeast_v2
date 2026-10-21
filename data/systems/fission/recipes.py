"""Fission dataset recipes; values preserve the original defaults."""
from ..contracts import Recipe


RECIPES = {
    'fission_random': Recipe('fission','fission_random',(.45,.35,.20),'fission','fission',default_samples=1000,default_seed=20000),
}
