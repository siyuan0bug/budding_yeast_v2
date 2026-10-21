"""Public modular Budding API for training, evaluation, AL and analysis.

Biology and real mutants always use canonical MA. Synthetic MASS and MA rules
are selected explicitly; importing this API never imports an archived generator.
"""
from . import parameters,states,ode,simulation,phenotypes,real_mutants,catalog,topology
from .specification import BuddingSystem,SPEC

get_default_params=parameters.get_default_params
PARAM_KEYS=parameters.PARAM_KEYS
initial_conds=states.initial_conds
VARIABLE_NAMES=states.VARIABLE_NAMES
eqns=ode.eqns
GK=ode.GK
simulate_mutant=simulation.simulate_mutant
prepare_initial_conditions=simulation.prepare_initial_conditions
REAL_MUTANTS=real_mutants.REAL_MUTANTS
mutant_rules=real_mutants.mutant_rules
is_duplicate_real_mutant=real_mutants.is_duplicate_real_mutant
_SYSTEM=BuddingSystem()
_MASS_SYSTEM=BuddingSystem('mass','mass')


def sample_scaling_factor(mode,g_name=None,*,rng=None):
    return _SYSTEM.sample_scaling_factor(mode,g_name,rng=rng)


def apply_gene_mutation(*args,rng=None):
    return _SYSTEM.apply_gene_mutation(*args,rng=rng)


def sample_mass_random(level=1):
    if level not in (1,2,3):raise ValueError('Only L1/L2/L3 are supported')
    from .samplers.random import sample_mass_random as sample
    return sample(level,ma=_MASS_SYSTEM)


def sample_mass_mutant(level=1):
    if level not in (1,2,3):raise ValueError('Only L1/L2/L3 are supported')
    from .samplers.mass import sample_mass_mutant as sample
    return sample(level,ma=_SYSTEM)


def __getattr__(name):
    from .samplers import common, mass
    if name=='MODE_FREQ':return common.MODE_FREQ
    for module in (parameters,states,ode,simulation,phenotypes,real_mutants,catalog,topology,common,mass):
        if hasattr(module,name):return getattr(module,name)
    raise AttributeError(name)
