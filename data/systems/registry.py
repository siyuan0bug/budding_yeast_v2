"""Biological system selection shared by datasets, graphs, models and evaluation."""


def get_system_spec(system_id):
    if system_id=='budding':
        from .budding.specification import SPEC
    elif system_id=='fission':
        from .fission.specification import SPEC
    else:
        raise ValueError(f'Unknown biological system: {system_id!r}')
    return SPEC


def resolve_system(system_id=None, *, num_vars=None, param_dim=None):
    if system_id is None:
        system_id='fission' if (num_vars,param_dim)==(16,58) else 'budding'
    spec=get_system_spec(system_id)
    if num_vars is not None and num_vars!=len(spec.variable_names):
        raise ValueError(f'{system_id}: variable dimension/order does not match its biological model')
    if param_dim is not None and param_dim!=len(spec.param_names):
        raise ValueError(f'{system_id}: parameter dimension/order does not match its biological model')
    return spec


def real_mutation_rules(system_id):
    if system_id=='budding':
        from .budding.real_mutants import mutant_rules
        return mutant_rules
    if system_id=='fission':
        from .fission.real_mutants import REAL_MUTANTS
        return {name:dict(value[0]) for name,value in REAL_MUTANTS.items()}
    return get_system_spec(system_id)
