"""Compatibility names for independent project selectors; no author imports."""
from types import SimpleNamespace
import torch
from .baseline.geometry import distance_select
from .baseline.bait import bait_select


def _bmdal_distance_select(pool_matrix, train_matrix, n_select, device='cpu', *, method):
    return distance_select(pool_matrix, train_matrix, n_select, device, method)


def _bmdal_lcmd_select(pool_matrix, train_matrix, n_select, device='cpu'):
    return distance_select(pool_matrix, train_matrix, n_select, device, 'lcmd')


def _bmdal_predicted_max_dist_select(pool_matrix, train_matrix, n_select, device='cpu'):
    return distance_select(pool_matrix, train_matrix, n_select, device, 'max_dist')


def _bmdal_bait_select(pool_matrix, train_matrix, n_select, device='cpu'):
    return bait_select(pool_matrix, train_matrix, n_select, device)


def _bmdal_max_dist_select(self, pool_features, train_features, n_select):
    return distance_select(pool_features, train_features, n_select,
                           'cuda' if torch.cuda.is_available() else 'cpu')


class _FeatureSelector:
    """Legacy feature-object protocol, independent of its source package."""
    method = 'max_dist'
    def __init__(self, pool_features, train_features, sel_with_train=True, **kwargs):
        self.pool = pool_features.get_feature_matrix()
        train = train_features.get_feature_matrix()
        self.train = train if sel_with_train else train[:0]
        self.options = kwargs
    def select(self, budget):
        return torch.as_tensor(distance_select(self.pool,self.train,budget,self.pool.device,self.method),
                               device=self.pool.device)


class _LCMD(_FeatureSelector):
    method = 'lcmd'


class _BAIT(_FeatureSelector):
    def select(self, budget):
        result=bait_select(self.pool,self.train,budget,self.pool.device,posterior=False,
            noise=self.options.get('noise_sigma',1e-2),
            overselection_factor=self.options.get('overselection_factor',1.))
        return torch.as_tensor(result,device=self.pool.device)


def selection_module():
    """Historical factory API implemented locally."""
    return SimpleNamespace(MaxDistSelectionMethod=_FeatureSelector,
        LargestClusterMaxDistSelectionMethod=_LCMD,BaitFeatureSpaceSelectionMethod=_BAIT)
