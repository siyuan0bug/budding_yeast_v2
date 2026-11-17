"""Round metrics share one W&B run and use matching slash-delimited axes."""
from pytorch_lightning.loggers import WandbLogger


class RoundWandbLogger(WandbLogger):
    LOGGER_JOIN_CHAR = '/'


def make_round_logger(experiment, save_dir, round_index):
    prefix = f'round_{round_index:02d}'
    experiment.define_metric(f'{prefix}/epoch')
    experiment.define_metric(f'{prefix}/*', step_metric=f'{prefix}/epoch', step_sync=True)
    return RoundWandbLogger(experiment=experiment, save_dir=save_dir, prefix=prefix)


def log_round_evaluation(experiment, record):
    experiment.define_metric('al/train_samples')
    experiment.define_metric('al/*', step_metric='al/train_samples', step_sync=True)
    experiment.log({f'al/{key}': value for key, value in record.items() if value is not None})
