"""Single registry for biological systems, dataset recipes and sampler creation."""
from dataclasses import replace
from importlib import import_module
from pathlib import Path

from .contracts import Recipe, SystemAdapter

# Adding a system requires its spec/adapter/recipes and one registration here.
_SYSTEMS = {
    'budding': ('budding', 'BuddingSampler'),
    'fission': ('fission', 'FissionSampler'),
}


def _system_module(system_id, module):
    try:
        package, _ = _SYSTEMS[system_id]
    except KeyError as exc:
        raise ValueError(f'Unknown biological system: {system_id!r}') from exc
    return import_module(f'{__package__}.{package}.{module}')


RECIPES = {
    name: recipe
    for system_id in _SYSTEMS
    for name, recipe in _system_module(system_id, 'recipes').RECIPES.items()
}


def get_system_spec(system_id):
    return _system_module(system_id, 'specification').SPEC


def resolve_system(system_id=None, *, num_vars=None, param_dim=None):
    if system_id is None:
        system_id = 'fission' if (num_vars, param_dim) == (16, 58) else 'budding'
    spec = get_system_spec(system_id)
    if num_vars is not None and num_vars != len(spec.variable_names):
        raise ValueError(f'{system_id}: variable dimension/order does not match its biological model')
    if param_dim is not None and param_dim != len(spec.param_names):
        raise ValueError(f'{system_id}: parameter dimension/order does not match its biological model')
    return spec


def get_adapter(system_id) -> SystemAdapter:
    return _system_module(system_id, 'adapter').ADAPTER


def real_mutation_rules(system_id):
    module = _system_module(system_id, 'real_mutants')
    if hasattr(module, 'mutant_rules'):
        return module.mutant_rules
    return {name: dict(value[0]) for name, value in module.REAL_MUTANTS.items()}


def get_recipe(name):
    if name not in RECIPES:
        raise ValueError(f'Unknown recipe {name!r}; available: {", ".join(RECIPES)}')
    return RECIPES[name]


def resolve_config(config):
    recipe = get_recipe(config.recipe)
    return replace(config,
        n_samples=recipe.default_samples if config.n_samples is None else config.n_samples,
        seed=recipe.default_seed if config.seed is None else config.seed,
        pilot_samples=recipe.default_pilot if config.pilot_samples is None else config.pilot_samples,
        mode_freq=config.mode_freq or recipe.mode_freq)


def output_filename(config):
    config = resolve_config(config)
    if config.recipe == 'budding_ko_mirror' and config.source:
        return f'{Path(config.source).stem}_ko_interval_seed{config.seed}_modular.npz'
    return f'{config.recipe}_{config.n_samples}_seed{config.seed}_modular.npz'


def make_sampler(recipe, config):
    config = resolve_config(config)
    module = _system_module(recipe.system, 'adapter')
    return getattr(module, _SYSTEMS[recipe.system][1])(recipe, config)


def candidate_name(kind, description, level, number):
    """Preserve existing external sample names."""
    if kind == 'fission_random':
        return f'Rand_{description}_L{level}_{number}'.replace(' ', '_')
    if kind == 'motif':
        return f'Synth_L{level}_{number:05d}_{description}'
    prefix = {'random': 'RANDOM', 'random_ko_interval': 'RANDOM', 'pool': 'MASS_RAND',
              'gibbs': 'GIBBS', 'gibbs_sobol': 'SOBOL', 'kdpp': 'KDPP', 'bams': 'BAMS',
              'wt_blind': 'WB-MASS'}.get(kind, 'MASS')
    return f'{prefix}_{description}_L{level}_{number}'.replace(' ', '_')


# Legacy factory imports construct the registered sampler, with no second implementation.
Sampler = make_sampler


def __getattr__(name):
    if name in ('BuddingSampler', 'FissionSampler'):
        system_id = 'budding' if name == 'BuddingSampler' else 'fission'
        return getattr(_system_module(system_id, 'adapter'), name)
    if name == 'common':
        return import_module(f'{__package__}.budding.samplers.common')
    raise AttributeError(name)
