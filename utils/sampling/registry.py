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
    **{name: StrategySpec(name,'ours.pattern','_pattern_sample',True)
       for name in ('ours_pattern_balance','ours_pattern_context','ours_pattern_support')},
    'ours_sensitivity': StrategySpec('ours_sensitivity','ours.sensitivity','_sensitivity_sample'),
    **{name: StrategySpec(name, 'ours.hybrid', '_hybrid_sample', True)
       for name in ('ours_global_interaction','ours_global_response','ours_global_tail','ours_global_calibrated')},
    **{name: StrategySpec(name, 'ours.experimental', '_experimental_sample', True)
       for name in ('ours_response_marginal', 'ours_response_feedback', 'ours_interaction',
                    'ours_invariance', 'ours_event', 'ours_error', 'ours_coverage')},
    'ours_response': StrategySpec('ours_response', 'ours.response', '_ours_response_sample', True),
    'al4pde_random': StrategySpec('al4pde_random', 'baseline.random', '_al4pde_random_sample', False, 'single_batch', systems=('budding', 'fission')),
    'al4pde_uncertainty': StrategySpec('al4pde_uncertainty', 'baseline.uncertainty', '_al4pde_uncertainty_sample', True, 'single_batch', systems=('budding', 'fission')),
    'al4pde_power': StrategySpec('al4pde_power', 'baseline.uncertainty', '_al4pde_power_sample', True, 'single_batch', systems=('budding', 'fission')),
    'al4pde_lcmd': StrategySpec('al4pde_lcmd', 'baseline.diversity', '_al4pde_lcmd_sample', False, 'single_batch', systems=('budding', 'fission')),
    'al4pde_bait': StrategySpec('al4pde_bait', 'baseline.diversity', '_al4pde_bait_sample', False, 'single_batch', systems=('budding', 'fission')),
    'al4pde_max_dist': StrategySpec('al4pde_max_dist', 'baseline.diversity', '_al4pde_max_dist_sample'),
    'al4pde_diversity': StrategySpec('al4pde_diversity', 'ours.diversity', '_al4pde_diversity_sample', False, 'single_batch'),
    'maal': StrategySpec('maal', 'ours.biology', '_maal_sample', False, 'single_batch'),
    'al4pde_mass': StrategySpec('al4pde_mass', 'ours.biology', '_al4pde_mass_sample', False, 'single_batch'),
    'al4pde_novelty': StrategySpec('al4pde_novelty', 'ours.biology', '_al4pde_novelty_sample', False, 'single_batch'),
    'al4pde_residual': StrategySpec('al4pde_residual', 'ours.residual', '_al4pde_residual_sample', False, 'refill_valid'),
    'al4pde_residual_diversity': StrategySpec('al4pde_residual_diversity', 'ours.residual', '_al4pde_residual_diversity_sample', False, 'refill_valid'),
    'al4pde_bifurcation': StrategySpec('al4pde_bifurcation', 'ours.events', '_al4pde_bifurcation_sample', False, 'refill_valid'),
    'al4pde_bifurcation_entropy': StrategySpec('al4pde_bifurcation_entropy', 'ours.events', '_al4pde_bifurcation_entropy_sample', False, 'refill_valid'),
    'al4pde_event_entropy': StrategySpec('al4pde_event_entropy', 'ours.events', '_al4pde_event_entropy_sample', False, 'refill_valid'),
    'al4pde_fisher_oed': StrategySpec('al4pde_fisher_oed', 'ours.fisher', '_al4pde_fisher_oed_sample', False, 'refill_valid'),
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


@dataclass(frozen=True)
class Method:
    name: str
    family: str = 'baseline'
    track: str = 'A'
    ensemble: bool = False
    mode: str = 'incremental'
    mutation_only: bool = False
    status: str = 'implemented'
    note: str = ''
    implementation: str | None = None  # Optional "module:function", under family/.


METHODS = {m.name: m for m in [
    Method('random'), Method('topk', ensemble=True), Method('power', ensemble=True),
    Method('max_dist_pred'), Method('coreset_ll'), Method('lcmd_ll'), Method('lcmd_pred'),
    Method('bait_ll'), Method('bait_pred'), Method('vessal_penultimate'),
    Method('lev_input', mode='one_shot'), Method('residual_topk', track='B'),
    Method('true_pattern_random', track='B'),
    *[Method(name, family='ours', ensemble=True) for name in
      ('ours_pattern_balance', 'ours_pattern_context', 'ours_pattern_support')],
    Method('ours_sensitivity', family='ours', mutation_only=True),
    Method('ours_mechanism_witness', family='ours', implementation='mechanistic.selectors:select_witness',
           note='Pattern-free queried-anchor response estimator; exact certificates only where explicitly validated'),
    Method('ours_dynamics_transport', family='ours', implementation='mechanistic.selectors:select_transport',
           note='First-order implicit physical defect transport with declared event linearization limits'),
    Method('ours_intervention_closure', family='ours', mutation_only=True,
           implementation='mechanistic.selectors:select_closure',
           note='Exact legal control rectangles; query-costed estimated interaction errors, explicit point fill'),
    Method('physics_pre_topk'), Method('physics_pre_power'),
    Method('clustering_sensitivity', track='B', mode='weighted_one_shot',
           note='Native draw-slot budget, possible duplicate draws aggregated with weights; not unique-10k AL'),
    Method('clustering_sensitivity_wor', track='B',
           note='Explicit unweighted WOR10k adaptation, not the original unbiased replacement estimator'),
    Method('low_rank', track='B', mode='weighted_one_shot',
           implementation='low_rank:select',
           note='Algorithm2 gradient-energy regression instantiation; weighted replacement draws, not unique-10k AL'),
    Method('peaks_regression',track='B',mode='dynamic',
           note='Continuous streaming reference-kernel regression adaptation; use dynamic CLI'),
    Method('eracs_regression',track='B',mode='dynamic',
           note='Continuous SNR-triggered weighted Batch-OMP regression adaptation; use dynamic CLI'),
    Method('random_stream',mode='dynamic',note='Random selection with the same continuous new+old PEAKS training schedule'),
    Method('random_dynamic',mode='dynamic',note='Random replacement with the same ERACS continuous schedule'),
    Method('peaks', track='B', status='workflow_required',
           note='Use explicit peaks_regression dynamic workflow; original class-logit proxy is not used'),
    Method('hggs', track='C', status='workflow_required',
           note='Use generate --method hggs_projected for GF + legal projected MGS trajectory adaptation'),
    Method('ogas', track='C', status='workflow_required',
           note='Use generate --method ogas_async_projected or ogas_round_projected; not native Melissa reproduction'),
]}


def method_spec(name):
    if name not in METHODS:
        raise ValueError(f'Unknown benchmark method {name!r}; choose from {tuple(METHODS)}')
    return METHODS[name]


def select(method, context, budget):
    spec = method_spec(method)
    if spec.status != 'implemented':
        raise NotImplementedError(f'{method}: {spec.note}')
    if spec.mode=='dynamic':raise ValueError('Dynamic selection runs through dynamic CLI, not one batch select()')
    if spec.track == 'A' and context.label_reader is not None:
        raise PermissionError('AL selectors must receive an input-only candidate view')
    if spec.implementation is None:
        module = import_module(f'budding_yeast_v2.utils.sampling.{spec.family}.offline')
        result = module.select(context,int(budget),spec.name)
    else:
        module_name,function_name=spec.implementation.split(':',1)
        module=import_module(f'budding_yeast_v2.utils.sampling.{spec.family}.{module_name}')
        result=getattr(module,function_name)(context,int(budget))
    return result.validate(context.pool.ids, int(budget),
        replacement_slots=spec.mode == 'weighted_one_shot')
