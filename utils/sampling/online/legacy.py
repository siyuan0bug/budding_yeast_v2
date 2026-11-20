"""Compatibility imports for the modular active-learning package.

Production entry points use utils.sampling. Old import/pickle paths remain
resolvable; numerical implementations and patch targets live in their modules.
"""
from budding_yeast_v2.utils.sampling import ActiveLearningEngine, train_acquisition_ensemble
from budding_yeast_v2.utils.sampling.online.oracle import (
    PARAM_KEYS, VARIABLE_NAMES, MASS_SIM_TIMEOUT, _MASS_DEFAULTS,
    simulate_mutant_mass, solve_single_mass_bridge, _solve_single_mass_task_wrapper,
)
from budding_yeast_v2.utils.sampling.online.pool import _parse_mass_gene_modes, _mass_sample_genes, _mass_desc_of

ActiveLearningCallback = ActiveLearningEngine

__all__ = ['ActiveLearningCallback', 'ActiveLearningEngine', 'train_acquisition_ensemble',
           'solve_single_mass_bridge', 'PARAM_KEYS', 'VARIABLE_NAMES']
