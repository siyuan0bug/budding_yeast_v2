"""
Parameter space generator for the AR+AL pipeline.

Wraps the existing LHS engine (lhs_v1_origin.py) to provide an AL4PDE-style
interface for generating PDE parameters and running ODE simulations.
"""

import os
import time
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import qmc

from budding_yeast_v2.data.lhs_v1_origin import (
    simulate_mutant,
    get_default_params,
    generate_lhs_tasks,
    mutant_rules,
    DISCRETE_STRICT_PARAMS,
)


# Canonical sorted parameter keys (must match lhs_v1_origin.py line 864)
_defaults = get_default_params()
_defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
PARAM_KEYS = sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])
NUM_PARAMS = len(PARAM_KEYS)  # should be 141


def _default_param_vector():
    """Return the default 141-dim parameter vector."""
    return np.array([_defaults[k] for k in PARAM_KEYS], dtype=np.float64)


class PDEParamGenerator:
    """
    Generates random PDE parameter vectors for the yeast cell cycle ODE.
    
    Two modes:
    1. LHS-style (default): Use the existing generate_lhs_tasks engine, 
       which respects discrete bindings and biological constraints.
    2. Uniform perturbation: Perturb the default parameters uniformly within
       a relative range (simpler but less biologically faithful).
    """

    def __init__(self, mode='lhs', perturbation_range=0.3, seed=None):
        """
        Args:
            mode: 'lhs' or 'uniform'
            perturbation_range: For uniform mode, relative perturbation range (0.3 = ±30%)
            seed: Random seed
        """
        self.mode = mode
        self.perturbation_range = perturbation_range
        self.num_pde_params = NUM_PARAMS
        self.rng = np.random.default_rng(seed)
        self._default_vec = _default_param_vector()

        # Precompute parameter bounds for uniform mode
        self._min_vals = self._default_vec * (1 - perturbation_range)
        self._max_vals = self._default_vec * (1 + perturbation_range)
        # Handle zero-valued defaults
        zero_mask = np.abs(self._default_vec) < 1e-12
        self._min_vals[zero_mask] = 0.0
        self._max_vals[zero_mask] = 0.01

    def set_rng(self, seed):
        self.rng = np.random.default_rng(seed)

    def get_pde_params(self, n):
        """
        Generate n parameter vectors.
        
        Returns:
            params: (n, 141) numpy array of parameter vectors
            param_dicts: list of n dicts (for simulate_mutant)
        """
        if self.mode == 'lhs':
            return self._generate_lhs(n)
        else:
            return self._generate_uniform(n)

    def generate_massive_candidate_pool(self, samples_per_mutant=10000):
        """
        Generate a massive candidate pool by requesting samples_per_mutant
        for EACH mutant. The underlying generate_lhs_tasks already handles
        deduplication and strictly discrete combination limits.
        
        Returns:
            params: (Total, 141) numpy array
            param_dicts: list of dicts
        """
        lhs_tasks = generate_lhs_tasks(samples_per_mutant)
        
        params_list = []
        dicts_list = []
        for name, overrides in lhs_tasks.items():
            p_dict = dict(_defaults)
            p_dict.update(overrides)
            p_vec = np.array([p_dict.get(k, _defaults[k]) for k in PARAM_KEYS])
            params_list.append(p_vec)
            dicts_list.append(overrides)
            
        return np.stack(params_list), dicts_list

    def _generate_lhs(self, n):
        """Generate parameters using the existing LHS engine."""
        # Use the existing engine to generate tasks
        lhs_tasks = generate_lhs_tasks(n)

        params_list = []
        dicts_list = []
        for name, overrides in list(lhs_tasks.items())[:n]:
            p_dict = dict(_defaults)
            p_dict.update(overrides)
            p_vec = np.array([p_dict.get(k, _defaults[k]) for k in PARAM_KEYS])
            params_list.append(p_vec)
            dicts_list.append(overrides)

        # If LHS engine returned fewer than n, pad with uniform samples
        while len(params_list) < n:
            p_vec = self._random_uniform_single()
            p_dict = {k: v for k, v in zip(PARAM_KEYS, p_vec)}
            params_list.append(p_vec)
            dicts_list.append(p_dict)

        return np.array(params_list, dtype=np.float64), dicts_list

    def _generate_uniform(self, n):
        """Generate parameters by uniformly perturbing defaults."""
        params_list = []
        dicts_list = []
        for _ in range(n):
            p_vec = self._random_uniform_single()
            p_dict = {k: v for k, v in zip(PARAM_KEYS, p_vec)}
            params_list.append(p_vec)
            dicts_list.append(p_dict)
        return np.array(params_list, dtype=np.float64), dicts_list

    def _random_uniform_single(self):
        return self.rng.uniform(self._min_vals, self._max_vals)


class YeastSimulator:
    """
    Wraps the ODE solver to provide an AL4PDE-style simulation interface.
    
    Calls simulate_mutant from lhs_v1_origin.py to generate full trajectories.
    """

    def __init__(self, t_max=210, t_steps=500, max_workers=None):
        self.t_max = t_max
        self.t_steps = t_steps
        self.num_channels = 39
        self.max_workers = max_workers or max(1, os.cpu_count() // 4)

    def simulate_batch(self, param_dicts, names=None):
        """
        Simulate a batch of ODE trajectories.
        
        Args:
            param_dicts: list of dicts, each mapping param_name -> value
            names: list of mutant names (for logging)
            
        Returns:
            trajectories: (N_success, 39, t_steps) numpy array
            valid_params: (N_success, 141) numpy array  
            valid_indices: list of indices that succeeded
        """
        if names is None:
            names = [f"AR_AL_sample_{i:05d}" for i in range(len(param_dicts))]

        results = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    simulate_mutant, name, p_dict, self.t_max, self.t_steps
                ): (i, name)
                for i, (name, p_dict) in enumerate(zip(names, param_dicts))
            }

            for future in futures:
                idx, name = futures[future]
                try:
                    m_name, y_uniform, bio_label, pattern_label = future.result(timeout=15.0)
                    if y_uniform is not None:
                        results[idx] = y_uniform
                except Exception:
                    pass

        # Collect successful results
        valid_indices = sorted(results.keys())
        if len(valid_indices) == 0:
            return np.zeros((0, 39, self.t_steps)), np.zeros((0, NUM_PARAMS)), []

        trajectories = np.stack([results[i] for i in valid_indices])
        return trajectories, valid_indices


def save_ar_batch(trajectories, params, save_path, al_iter, batch_idx):
    """
    Save a batch of trajectories and parameters as separate npy files.
    Follows AL4PDE naming convention: <pde_name>_alstp_<iter>_btch<batch>_traj.npy
    
    Args:
        trajectories: (N, 39, T) numpy array
        params: (N, 141) numpy array
        save_path: directory to save to
        al_iter: AL iteration identifier (int or str like 'init')
        batch_idx: batch index within this iteration
    """
    os.makedirs(save_path, exist_ok=True)
    prefix = f"yeast_alstp_{al_iter}_btch{batch_idx}"
    np.save(os.path.join(save_path, f"{prefix}_traj.npy"), trajectories)
    np.save(os.path.join(save_path, f"{prefix}_param.npy"), params)


def load_ar_batches(folders):
    """
    Load all trajectory batches from a list of folders.
    Returns concatenated (N, 39, T) trajectories and (N, 141) params.
    """
    import glob
    all_trajs = []
    all_params = []

    for folder in folders:
        traj_files = sorted(glob.glob(os.path.join(folder, "*_traj.npy")))
        for traj_file in traj_files:
            param_file = traj_file.replace("_traj.npy", "_param.npy")
            if os.path.exists(param_file):
                trajs = np.load(traj_file)
                params = np.load(param_file)
                all_trajs.append(trajs)
                all_params.append(params)

    if len(all_trajs) == 0:
        raise IOError(f"No trajectory data found in {folders}")

    return np.concatenate(all_trajs, axis=0), np.concatenate(all_params, axis=0)
