"""Single source of strategy names, aliases, capabilities and constructors."""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class StrategySpec:
    name: str
    module: str
    function: str
    requires_ensemble: bool = False
    fill_policy: str = 'single_batch'
    systems: tuple[str, ...] = ('budding',)


SPECS = {
    **{name: StrategySpec(name, 'strategies.hybrid', '_hybrid_sample', True)
       for name in ('ours_global_interaction','ours_global_response','ours_global_tail','ours_global_calibrated')},
    **{name: StrategySpec(name, 'strategies.experimental', '_experimental_sample', True)
       for name in ('ours_response_marginal', 'ours_response_feedback', 'ours_interaction',
                    'ours_invariance', 'ours_event', 'ours_error', 'ours_coverage')},
    'ours_response': StrategySpec('ours_response', 'strategies.response', '_ours_response_sample', True),
    'al4pde_random': StrategySpec('al4pde_random', 'strategies.random', '_al4pde_random_sample', False, 'single_batch', systems=('budding', 'fission')),
    'al4pde_uncertainty': StrategySpec('al4pde_uncertainty', 'strategies.uncertainty', '_al4pde_uncertainty_sample', True, 'single_batch', systems=('budding', 'fission')),
    'al4pde_power': StrategySpec('al4pde_power', 'strategies.uncertainty', '_al4pde_power_sample', True, 'single_batch', systems=('budding', 'fission')),
    'al4pde_lcmd': StrategySpec('al4pde_lcmd', 'strategies.diversity', '_al4pde_lcmd_sample', False, 'single_batch', systems=('budding', 'fission')),
    'al4pde_bait': StrategySpec('al4pde_bait', 'strategies.diversity', '_al4pde_bait_sample', False, 'single_batch', systems=('budding', 'fission')),
    'al4pde_max_dist': StrategySpec('al4pde_max_dist', 'strategies.diversity', '_al4pde_max_dist_sample'),
    'al4pde_diversity': StrategySpec('al4pde_diversity', 'strategies.diversity', '_al4pde_diversity_sample', False, 'single_batch'),
    'maal': StrategySpec('maal', 'strategies.biology', '_maal_sample', False, 'single_batch'),
    'al4pde_mass': StrategySpec('al4pde_mass', 'strategies.biology', '_al4pde_mass_sample', False, 'single_batch'),
    'al4pde_novelty': StrategySpec('al4pde_novelty', 'strategies.biology', '_al4pde_novelty_sample', False, 'single_batch'),
    'al4pde_residual': StrategySpec('al4pde_residual', 'strategies.residual', '_al4pde_residual_sample', False, 'refill_valid'),
    'al4pde_residual_diversity': StrategySpec('al4pde_residual_diversity', 'strategies.residual', '_al4pde_residual_diversity_sample', False, 'refill_valid'),
    'al4pde_bifurcation': StrategySpec('al4pde_bifurcation', 'strategies.events', '_al4pde_bifurcation_sample', False, 'refill_valid'),
    'al4pde_bifurcation_entropy': StrategySpec('al4pde_bifurcation_entropy', 'strategies.events', '_al4pde_bifurcation_entropy_sample', False, 'refill_valid'),
    'al4pde_event_entropy': StrategySpec('al4pde_event_entropy', 'strategies.events', '_al4pde_event_entropy_sample', False, 'refill_valid'),
    'al4pde_fisher_oed': StrategySpec('al4pde_fisher_oed', 'strategies.fisher', '_al4pde_fisher_oed_sample', False, 'refill_valid'),
}
ALIASES = {
    'al4pde_res_div': 'al4pde_residual_diversity',
    'al4pde_bif_entropy': 'al4pde_bifurcation_entropy',
    'al4pde_event': 'al4pde_event_entropy',
    'al4pde_foed': 'al4pde_fisher_oed',
}
STRATEGY_NAMES = tuple(SPECS) + tuple(ALIASES)
CLI_STRATEGIES = ('none',) + STRATEGY_NAMES


def canonical_name(name):
    name = str(name).lower()
    return ALIASES.get(name, name)


def is_strategy(name):
    return canonical_name(name) in SPECS


def strategy_spec(name):
    key = canonical_name(name)
    if key not in SPECS:
        raise ValueError(f'Unsupported or archived AL strategy {name!r}; see utils/archive/al_callback_legacy.py')
    return SPECS[key]


class RegisteredStrategy:
    def __init__(self, spec):
        self.spec = spec

    def select(self, context, budget):
        if getattr(context.datamodule, 'system_id', 'budding') not in self.spec.systems:
            raise ValueError(f'{self.spec.name} does not support {context.datamodule.system_id}; supported systems: {self.spec.systems}')
        if budget != context.engine._al4pde_scheduled_num_add():
            raise ValueError('Selection budget disagrees with the configured schedule')
        module = import_module('.' + self.spec.module, __package__)
        function = getattr(module, self.spec.function)
        return function(context.engine, None, context.model, context.datamodule, context.device, 0)


def build_strategy(name):
    return RegisteredStrategy(strategy_spec(name))


def validate_system(name, system_id):
    spec = strategy_spec(name)
    if system_id not in spec.systems:
        supported = ', '.join(key for key, value in SPECS.items() if system_id in value.systems)
        raise ValueError(f'{name} is not supported for {system_id}. Available strategies: {supported}')
    return spec
