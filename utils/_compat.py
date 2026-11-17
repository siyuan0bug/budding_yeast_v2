"""Lazy aliases for historical imports/checkpoints; all implementations are canonical.

There are no duplicated legacy packages or eager Torch imports in spawn workers.
The aliases return the same module object so historical patch targets still work.
"""
import importlib
import importlib.abc
import importlib.util
import sys

ALIASES = {'budding_yeast_v2.utils.active_learning': 'budding_yeast_v2.utils.sampling',
 'budding_yeast_v2.utils.active_learning.auxiliary': 'budding_yeast_v2.utils.sampling.ours.auxiliary',
 'budding_yeast_v2.utils.active_learning.auxiliary.artifacts': 'budding_yeast_v2.utils.sampling.ours.auxiliary.artifacts',
 'budding_yeast_v2.utils.active_learning.auxiliary.canonical_rhs': 'budding_yeast_v2.utils.physics.canonical_rhs',
 'budding_yeast_v2.utils.active_learning.auxiliary.coverage': 'budding_yeast_v2.utils.sampling.ours.auxiliary.coverage',
 'budding_yeast_v2.utils.active_learning.auxiliary.mutation_directions': 'budding_yeast_v2.utils.sampling.ours.auxiliary.mutation_directions',
 'budding_yeast_v2.utils.active_learning.auxiliary.pattern_classifier': 'budding_yeast_v2.utils.sampling.ours.auxiliary.pattern_classifier',
 'budding_yeast_v2.utils.active_learning.auxiliary.phenotype': 'budding_yeast_v2.utils.sampling.ours.auxiliary.phenotype',
 'budding_yeast_v2.utils.active_learning.auxiliary.sensitivity': 'budding_yeast_v2.utils.sampling.ours.auxiliary.sensitivity',
 'budding_yeast_v2.utils.active_learning.auxiliary.views': 'budding_yeast_v2.utils.sampling.ours.auxiliary.views',
 'budding_yeast_v2.utils.active_learning.backends': 'budding_yeast_v2.utils.sampling.backends',
 'budding_yeast_v2.utils.active_learning.contracts': 'budding_yeast_v2.utils.sampling.contracts',
 'budding_yeast_v2.utils.active_learning.convergence': 'budding_yeast_v2.utils.training.convergence',
 'budding_yeast_v2.utils.active_learning.dataset': 'budding_yeast_v2.utils.archive.online.dataset',
 'budding_yeast_v2.utils.active_learning.engine': 'budding_yeast_v2.utils.archive.online.engine',
 'budding_yeast_v2.utils.active_learning.features': 'budding_yeast_v2.utils.archive.online.features',
 'budding_yeast_v2.utils.active_learning.fission': 'budding_yeast_v2.utils.archive.online.fission',
 'budding_yeast_v2.utils.active_learning.io': 'budding_yeast_v2.utils.archive.online.io',
 'budding_yeast_v2.utils.active_learning.logging': 'budding_yeast_v2.utils.archive.online.logging',
 'budding_yeast_v2.utils.active_learning.oracle': 'budding_yeast_v2.utils.archive.online.oracle',
 'budding_yeast_v2.utils.active_learning.pool': 'budding_yeast_v2.utils.archive.online.pool',
 'budding_yeast_v2.utils.active_learning.prediction': 'budding_yeast_v2.utils.archive.online.prediction',
 'budding_yeast_v2.utils.active_learning.registry': 'budding_yeast_v2.utils.sampling.registry',
 'budding_yeast_v2.utils.active_learning.reporting': 'budding_yeast_v2.utils.archive.online.reporting',
 'budding_yeast_v2.utils.active_learning.strategies': 'budding_yeast_v2.utils.sampling.ours',
 'budding_yeast_v2.utils.active_learning.strategies.biology': 'budding_yeast_v2.utils.sampling.ours.biology',
 'budding_yeast_v2.utils.active_learning.strategies.diversity': 'budding_yeast_v2.utils.archive.online.legacy_diversity',
 'budding_yeast_v2.utils.active_learning.strategies.events': 'budding_yeast_v2.utils.sampling.ours.events',
 'budding_yeast_v2.utils.active_learning.strategies.experimental': 'budding_yeast_v2.utils.sampling.ours.experimental',
 'budding_yeast_v2.utils.active_learning.strategies.fisher': 'budding_yeast_v2.utils.sampling.ours.fisher',
 'budding_yeast_v2.utils.active_learning.strategies.hybrid': 'budding_yeast_v2.utils.sampling.ours.hybrid',
 'budding_yeast_v2.utils.active_learning.strategies.hybrid_features': 'budding_yeast_v2.utils.sampling.ours.hybrid_features',
 'budding_yeast_v2.utils.active_learning.strategies.hybrid_feedback': 'budding_yeast_v2.utils.sampling.ours.hybrid_feedback',
 'budding_yeast_v2.utils.active_learning.strategies.pattern': 'budding_yeast_v2.utils.sampling.ours.pattern',
 'budding_yeast_v2.utils.active_learning.strategies.query_design': 'budding_yeast_v2.utils.sampling.ours.query_design',
 'budding_yeast_v2.utils.active_learning.strategies.query_feedback': 'budding_yeast_v2.utils.sampling.ours.query_feedback',
 'budding_yeast_v2.utils.active_learning.strategies.random': 'budding_yeast_v2.utils.sampling.baseline.random',
 'budding_yeast_v2.utils.active_learning.strategies.residual': 'budding_yeast_v2.utils.sampling.ours.residual',
 'budding_yeast_v2.utils.active_learning.strategies.response': 'budding_yeast_v2.utils.sampling.ours.response',
 'budding_yeast_v2.utils.active_learning.strategies.sensitivity': 'budding_yeast_v2.utils.sampling.ours.sensitivity',
 'budding_yeast_v2.utils.active_learning.strategies.uncertainty': 'budding_yeast_v2.utils.sampling.baseline.uncertainty',
 'budding_yeast_v2.utils.al_callback': 'budding_yeast_v2.utils.archive.online.legacy',
 'budding_yeast_v2.utils.biological_guard': 'budding_yeast_v2.utils.physics.variable_groups',
 'budding_yeast_v2.utils.checkpoint_evaluation': 'budding_yeast_v2.utils.evaluation.checkpoints',
 'budding_yeast_v2.utils.curve_logger': 'budding_yeast_v2.utils.experiments.curves',
 'budding_yeast_v2.utils.evaluation.sampling': 'budding_yeast_v2.eval',
 'budding_yeast_v2.utils.exp_utils': 'budding_yeast_v2.utils.experiments.paths',
 'budding_yeast_v2.utils.experiments.sampling_plots': 'budding_yeast_v2.utils.experiments.curves',
 'budding_yeast_v2.utils.fission_physics': 'budding_yeast_v2.utils.physics.fission',
 'budding_yeast_v2.utils.losses': 'budding_yeast_v2.utils.training.losses',
 'budding_yeast_v2.utils.metrics': 'budding_yeast_v2.utils.evaluation.metrics',
 'budding_yeast_v2.utils.ode_residual_engine': 'budding_yeast_v2.utils.physics.residuals',
 'budding_yeast_v2.utils.physics_eqns_torch': 'budding_yeast_v2.utils.physics.budding',
 'budding_yeast_v2.utils.runtime_logging': 'budding_yeast_v2.utils.experiments.logging',
 'budding_yeast_v2.utils.sampling.benchmark.contracts': 'budding_yeast_v2.utils.sampling.contracts',
 'budding_yeast_v2.utils.sampling.benchmark.methods': 'budding_yeast_v2.utils.sampling.registry',
 'budding_yeast_v2.utils.sampling.benchmark.model': 'budding_yeast_v2.utils.evaluation.checkpoints',
 'budding_yeast_v2.utils.sampling.benchmark.storage': 'budding_yeast_v2.data.dataset_archive',
 'budding_yeast_v2.utils.sampling.convergence': 'budding_yeast_v2.utils.training.convergence',
 'budding_yeast_v2.utils.sampling.dataset': 'budding_yeast_v2.utils.archive.online.dataset',
 'budding_yeast_v2.utils.sampling.engine': 'budding_yeast_v2.utils.archive.online.engine',
 'budding_yeast_v2.utils.sampling.features': 'budding_yeast_v2.utils.archive.online.features',
 'budding_yeast_v2.utils.sampling.fission': 'budding_yeast_v2.utils.archive.online.fission',
 'budding_yeast_v2.utils.sampling.io': 'budding_yeast_v2.utils.archive.online.io',
 'budding_yeast_v2.utils.sampling.legacy': 'budding_yeast_v2.utils.archive.online.legacy',
 'budding_yeast_v2.utils.sampling.legacy_diversity': 'budding_yeast_v2.utils.archive.online.legacy_diversity',
 'budding_yeast_v2.utils.sampling.logging': 'budding_yeast_v2.utils.archive.online.logging',
 'budding_yeast_v2.utils.sampling.oracle': 'budding_yeast_v2.utils.archive.online.oracle',
 'budding_yeast_v2.utils.sampling.ours.auxiliary.canonical_rhs': 'budding_yeast_v2.utils.physics.canonical_rhs',
 'budding_yeast_v2.utils.sampling.pool': 'budding_yeast_v2.utils.archive.online.pool',
 'budding_yeast_v2.utils.sampling.prediction': 'budding_yeast_v2.utils.archive.online.prediction',
 'budding_yeast_v2.utils.sampling.reporting': 'budding_yeast_v2.utils.archive.online.reporting'}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.canonical_spec = module.__spec__
        self.canonical_loader = module.__loader__
        return module

    def exec_module(self, module):
        # Import machinery temporarily assigns the alias spec to this existing
        # object. Restore its canonical identity for pickle and multiprocessing.
        module.__spec__ = self.canonical_spec
        module.__loader__ = self.canonical_loader


class _LegacyUtilsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        canonical = ALIASES.get(fullname)
        if canonical is None and fullname.startswith('utils.'):
            # Historical scripts sometimes put budding_yeast_v2 itself on
            # sys.path. Route those short imports to the same canonical object.
            qualified = 'budding_yeast_v2.' + fullname
            canonical = ALIASES.get(qualified, qualified)
        if canonical is None:
            return None
        spec = importlib.util.find_spec(canonical)
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(canonical),
            is_package=spec.submodule_search_locations is not None)


def install():
    if not any(isinstance(finder, _LegacyUtilsFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyUtilsFinder())
