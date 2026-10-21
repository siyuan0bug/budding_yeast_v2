"""One canonical biological model, independently selectable synthetic profiles."""
import importlib

from ..contracts import SystemSpec
from .parameters import PARAM_KEYS, get_default_params
from .states import VARIABLE_NAMES, initial_conds
from .simulation import simulate_mutant, prepare_initial_conditions
from . import catalog, topology, real_mutants

SPEC = SystemSpec('budding', 'mutant_aware_2026_10_19', tuple(VARIABLE_NAMES),
                  tuple(PARAM_KEYS), 210.0, 500, 'mutant_aware_real_v1')


class BuddingSystem:
    """Explicit backend passed to samplers; never imports an old generator."""
    get_default_params = staticmethod(get_default_params)
    initial_conds = staticmethod(initial_conds)
    simulate_mutant = staticmethod(simulate_mutant)
    prepare_initial_conditions = staticmethod(prepare_initial_conditions)
    PARAM_KEYS = PARAM_KEYS
    REAL_MUTANTS = real_mutants.REAL_MUTANTS
    spec = SPEC

    def __init__(self, mutation_profile='mutant_aware', distribution_profile='mutant_aware'):
        profiles = ('mass', 'mutant_aware')
        if mutation_profile not in profiles or distribution_profile not in profiles:
            raise ValueError(f'Budding profiles must belong to {profiles}')
        self.mutation_profile = mutation_profile
        self.distribution_profile = distribution_profile
        self._mutations = importlib.import_module(f'{__package__}.mutations.{mutation_profile}')
        self._distributions = importlib.import_module(f'{__package__}.distributions.{distribution_profile}')

    def sample_scaling_factor(self, mode, g_name=None, *, rng=None):
        return self._distributions.sample_scaling_factor(mode, g_name, rng=rng)

    def apply_gene_mutation(self, *args, rng=None):
        return self._mutations.apply_gene_mutation(*args, rng=rng)

    def sample_scaling_factor_sobol(self, mode, gene, u):
        from .distributions.quantiles import sample_scaling_factor_sobol
        if self.distribution_profile == 'mutant_aware':
            return sample_scaling_factor_sobol(mode,gene,u)
        import numpy as np
        from scipy.stats import beta
        u = np.clip(u,1e-6,1.-1e-6)
        if mode=='MC':
            return float(10.**(np.log10(1.5)+u*(np.log10(30.)-np.log10(1.5))))
        if mode=='GAL-OE' and gene=='CLN2':
            return float(10.**(np.log10(.03)+u*(np.log10(.30)-np.log10(.03))))
        if mode=='ts' and gene!='CDC14':
            return float(.2*beta.ppf(u,.5,2.))
        if mode=='db':
            if gene=='CLB2': return float(.04+u*(.15-.04))
            if gene=='PDS1': return float(u*.05)
            return 0.
        return sample_scaling_factor_sobol(mode,gene,u)

    def __getattr__(self, name):
        # Shared selection utilities have per-run reset managed by the driver.
        from .samplers import common
        if name=='MODE_FREQ':
            return common.MODE_FREQ
        for module in (catalog, topology, common):
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(name)

    def defaults(self):
        defaults=get_default_params()
        defaults.update({'init_CDH1T':1.,'init_CDH1':.9304992,'init_BUB2':.2})
        return defaults
