"""Uniform query-point control on the shared legal-path action candidates."""
import numpy as np


def choose(available,rng):
    ids=np.asarray(available,dtype=np.int64)
    if not len(ids):raise ValueError('No path-control query candidates')
    return int(rng.choice(ids))
