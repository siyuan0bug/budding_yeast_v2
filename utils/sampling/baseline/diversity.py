"""strategies diversity services extracted from the pre-package implementation."""


def _al4pde_max_dist_sample(self, trainer, pl_module, dm, device, current_epoch):
    """AL4PDE output-trajectory MaxDist; separate from custom shape diversity."""
    if self._pool_size == 0:
        return None
    pool_features, train_features = self._compute_predicted_distance_features(pl_module,dm,device)
    selected = self._bmdal_predicted_max_dist_select(pool_features,train_features,
        self._al4pde_scheduled_num_add(),device)
    return self.make_selection_plan(selected,dm)



def _al4pde_lcmd_sample(self, trainer, pl_module, dm, device, current_epoch):
    if self._pool_size == 0:
        return None
    pool_features, train_features = self._compute_predicted_distance_features(pl_module, dm, device)
    selected = self._bmdal_lcmd_select(pool_features, train_features,
                                      self._al4pde_scheduled_num_add(), device)
    return self.make_selection_plan(selected, dm)


def _al4pde_bait_sample(self, trainer, pl_module, dm, device, current_epoch):
    if self._pool_size == 0:
        return None
    pool_features, train_features = self._compute_bait_features(pl_module, dm, device)
    selected = self._bmdal_bait_select(pool_features, train_features,
                                      self._al4pde_scheduled_num_add(), device)
    return self.make_selection_plan(selected, dm)
