"""Public active-learning API; heavy training imports are lazy for spawn workers."""
__all__ = ['ActiveLearningEngine', 'ActiveLearningCallback', 'ALConfig', 'SelectionPlan',
           'QueryBatch', 'build_strategy', 'train_acquisition_ensemble']


def __getattr__(name):
    if name in ('ActiveLearningEngine', 'ActiveLearningCallback'):
        from .engine import ActiveLearningEngine
        return ActiveLearningEngine
    if name == 'train_acquisition_ensemble':
        from .prediction import train_acquisition_ensemble
        return train_acquisition_ensemble
    if name == 'build_strategy':
        from .registry import build_strategy
        return build_strategy
    if name in ('ALConfig', 'SelectionPlan', 'QueryBatch'):
        from . import contracts
        return getattr(contracts, name)
    raise AttributeError(name)
