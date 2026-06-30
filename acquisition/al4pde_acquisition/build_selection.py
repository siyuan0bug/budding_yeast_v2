import hydra
from budding_yeast_v2.acquisition.al4pde_acquisition.distance_pool_based import build_distance_pool_based


def build_strategy(task, cfg):
    if cfg._target_ == "budding_yeast_v2.acquisition.al4pde_acquisition.distance_pool_based.DistancePoolBased":
        return build_distance_pool_based(task, cfg)
    else:
        return hydra.utils.instantiate(cfg, task=task)
