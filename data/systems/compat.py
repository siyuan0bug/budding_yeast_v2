"""Lazy aliases for retired framework paths; all implementations live in systems.

Aliases resolve to the *same module object*, so state assignment, monkeypatching
and legacy pickle imports work without copying module globals. No retired source
folders or eagerly imported sampler implementations are needed.
"""
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import sys

_DATA = __package__.rsplit('.', 1)[0]
_SYSTEMS = __package__
_SAMPLERS = ('random', 'kdpp', 'targeted_ablation', 'sobol', 'mass_motifs',
             'twins', 'motif', 'mass', 'wt_pipeline', 'bams', 'common', 'wt_blind', 'gibbs')
_ALIASES = {
    f'{_DATA}.recipes': f'{_SYSTEMS}.registry',
    f'{_SYSTEMS}.base': f'{_SYSTEMS}.registry',
    f'{_SYSTEMS}.factory': f'{_SYSTEMS}.registry',
    f'{_DATA}.sampling.factory': f'{_SYSTEMS}.registry',
    f'{_DATA}.sampling.fission_random': f'{_SYSTEMS}.fission.samplers.random',
    f'{_DATA}.generation.contracts': f'{_SYSTEMS}.contracts',
    f'{_DATA}.generation.runner': f'{_SYSTEMS}.generation',
    f'{_DATA}.generation.policies': f'{_SYSTEMS}.generation',
    f'{_DATA}.generation.io': f'{_SYSTEMS}.dataset_io',
    f'{_DATA}.generation.derived': f'{_SYSTEMS}.budding.derived',
    **{f'{_DATA}.sampling.{name}': f'{_SYSTEMS}.budding.samplers.{name}' for name in _SAMPLERS},
}
_PACKAGES = (f'{_DATA}.sampling', f'{_DATA}.generation')


class _LegacyLoader(Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__
        if name in _PACKAGES:
            def lookup(attribute):
                child = f'{name}.{attribute}'
                if child in _ALIASES:
                    return import_module(child)
                raise AttributeError(attribute)
            module.__getattr__ = lookup
            module.__doc__ = 'Deprecated import namespace; use data.systems instead.'
            return
        target = import_module(_ALIASES[name])
        if name == f'{_DATA}.generation.policies':
            # Older modular callers imported rescue helpers from policies.
            rescue = import_module(f'{_SYSTEMS}.budding.samplers.rescue')
            target.parse_pairs = rescue.parse_pairs
            target.rescue_budding = rescue.rescue_budding
            target.rescue_fission = import_module(f'{_SYSTEMS}.fission.samplers.rescue').rescue_fission
        sys.modules[name] = target


class _LegacyFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _ALIASES or fullname in _PACKAGES:
            return spec_from_loader(fullname, _LegacyLoader(), is_package=fullname in _PACKAGES)
        return None


def install_legacy_imports():
    if not any(isinstance(finder, _LegacyFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyFinder())
