"""
AR Trajectory Dataset for autoregressive training.

References AL4PDE's TrajDataset (dataloader.py) but adapted for
1D time-series data: (num_samples, num_vars=39, T=500).
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from budding_yeast_v2.data.param_gen import load_ar_batches


class ARTrajDataset(Dataset):
    """
    Slices full trajectories into sub-sequences for AR training.
    
    Given num_train_steps=K, each trajectory of length T is sliced into
    (T - K) sub-sequences of length (K+1). The first frame is the input IC,
    the remaining K frames are targets.
    
    Data layout (internal):
        data:       (N, T, C)    — N trajectories, T timesteps, C=39 channels
        pde_params: (N, P)       — N parameter vectors, P=141
    
    __getitem__ returns:
        ic:         (C,)         — state at start_t
        traj:       (C, K+1)     — sub-trajectory [start_t : start_t+K+1] in (C, T) format
        params:     (P,)         — PDE parameters
        t_idx:      scalar       — starting time index
    """

    def __init__(self, data, pde_params, num_steps=None):
        """
        Args:
            data: (N, C, T) or (N, T, C) numpy/tensor — trajectory data
            pde_params: (N, P) numpy/tensor — parameter vectors
            num_steps: int or None — number of AR steps for training.
                       None = use full trajectory length - 1
        """
        # Ensure data is (N, T, C) internally
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if isinstance(pde_params, np.ndarray):
            pde_params = torch.from_numpy(pde_params).float()

        # Detect layout: if shape is (N, 39, 500), transpose to (N, 500, 39)
        if data.dim() == 3 and data.shape[1] < data.shape[2]:
            # Likely (N, C, T) format — transpose to (N, T, C)
            data = data.permute(0, 2, 1)

        self.data = data            # (N, T, C)
        self.pde_params = pde_params  # (N, P)
        self.T = data.shape[1]
        self.C = data.shape[2]

        if num_steps is not None:
            self._set_num_steps(num_steps)
        else:
            self._set_num_steps(self.T - 1)

    def _set_num_steps(self, num_steps):
        """Set the number of AR steps, which determines sub-sequence length."""
        self.num_steps = num_steps
        self.traj_len = num_steps + 1  # includes IC
        self.num_sub_per_traj = self.T - num_steps
        self.num_sub_trajs = self.num_sub_per_traj * self.data.shape[0]

    def set_num_steps(self, num_steps):
        """Return a new dataset view with different num_steps."""
        ds = ARTrajDataset.__new__(ARTrajDataset)
        ds.data = self.data
        ds.pde_params = self.pde_params
        ds.T = self.T
        ds.C = self.C
        if num_steps is None:
            ds._set_num_steps(ds.T - 1)
        else:
            ds._set_num_steps(num_steps)
        return ds

    def __len__(self):
        return self.num_sub_trajs

    def __getitem__(self, idx):
        traj_idx = idx // self.num_sub_per_traj
        start_t = idx % self.num_sub_per_traj

        # ic: state at start_t, shape (C,)
        ic = self.data[traj_idx, start_t, :]

        # traj: sub-trajectory from start_t to start_t + traj_len, shape (traj_len, C)
        # Transpose to (C, traj_len) for consistency with model output format
        traj = self.data[traj_idx, start_t: start_t + self.traj_len, :].T  # (C, traj_len)

        params = self.pde_params[traj_idx]
        t_idx = torch.tensor(start_t, dtype=torch.float32)

        return ic, traj, params, t_idx


class ARDataModule:
    """
    Manages data loading for the AR+AL pipeline.
    
    Unlike the Lightning DataModule, this is a plain Python class that
    creates DataLoaders on demand (since the dataset changes each AL iter).
    """

    def __init__(self, num_train_steps=2, batch_size=64, num_workers=4,
                 reduced_resolution_t=1):
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.reduced_resolution_t = reduced_resolution_t

    def build_train_loader(self, data_folders):
        """
        Build a DataLoader from all trajectory batches in data_folders.
        
        Args:
            data_folders: list of directories containing npy batch files
            
        Returns:
            train_loader: DataLoader with AR-sliced sub-sequences
            train_loader_full: DataLoader with full trajectories (for eval)
            dataset: the underlying ARTrajDataset
        """
        trajs, params = load_ar_batches(data_folders)
        print(f"[ARDataModule] Loaded {len(trajs)} trajectories from {data_folders}")

        # Time downsampling
        if self.reduced_resolution_t > 1:
            trajs = trajs[:, :, ::self.reduced_resolution_t]

        dataset = ARTrajDataset(trajs, params, num_steps=self.num_train_steps)

        train_loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, drop_last=True
        )

        # Full trajectory loader (for validation rollout)
        # Full sequence loader (for evaluation/metrics)
        train_loader_full = DataLoader(
            dataset.set_num_steps(None), batch_size=512, shuffle=False,
            num_workers=self.num_workers
        )

        return train_loader, train_loader_full, dataset

    def build_eval_loader(self, data_folders):
        """Build a DataLoader for evaluation (full trajectories)."""
        trajs, params = load_ar_batches(data_folders)

        if self.reduced_resolution_t > 1:
            trajs = trajs[:, :, ::self.reduced_resolution_t]

        dataset = ARTrajDataset(trajs, params, num_steps=None)
        loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers
        )
        return loader, dataset
