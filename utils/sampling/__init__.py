"""Sampling API and historical online aliases; imports stay lazy for spawn workers."""
__all__ = ['ActiveLearningEngine', 'ActiveLearningCallback', 'ALConfig', 'SelectionPlan',
           'QueryBatch', 'build_strategy', 'train_acquisition_ensemble',
           'PoolInputs', 'QueriedTraining', 'SamplingContext', 'SamplingResult',
           'BenchmarkConfig', 'select_samples', 'run_sampling_experiment']


def __getattr__(name):
    if name in ('PoolInputs', 'QueriedTraining', 'SamplingContext', 'SamplingResult'):
        from . import contracts as benchmark_contracts
        return getattr(benchmark_contracts, name)
    if name == 'BenchmarkConfig':
        from .benchmark.protocol import BenchmarkConfig
        return BenchmarkConfig
    if name == 'select_samples':
        from .registry import select
        return select
    if name == 'run_sampling_experiment':
        from .benchmark.runner import run_experiment
        return run_experiment
    if name in ('ActiveLearningEngine', 'ActiveLearningCallback'):
        from budding_yeast_v2.utils.archive.online.engine import ActiveLearningEngine
        return ActiveLearningEngine
    if name == 'train_acquisition_ensemble':
        from budding_yeast_v2.utils.archive.online.prediction import train_acquisition_ensemble
        return train_acquisition_ensemble
    if name == 'build_strategy':
        from budding_yeast_v2.utils.sampling.registry import build_strategy
        return build_strategy
    if name in ('ALConfig', 'SelectionPlan', 'QueryBatch'):
        from budding_yeast_v2.utils.sampling import contracts
        return getattr(contracts, name)
    raise AttributeError(name)
