"""Numerical accuracy policy, independent of mutation and phenotype rules."""
import numpy as np

SOLVER_POLICY = 'budding_bdf_tight_checked_v1'
RTOL, ATOL = 1e-8, 1e-10
RETRY_RTOL, RETRY_ATOL = 1e-10, 1e-12


def numerical_flags(trajectory, params, initial):
    """Retry known numerical artifacts; finite negative values are not a filter.

    TEM1's active amount cannot exceed its fixed total when the initial state
    starts inside that interval. Tolerance admits roundoff, not large excursions.
    No empirical real-mutant amplitude envelope is imposed.
    """
    y = np.asarray(trajectory)
    total = float(params['Tem1T'])
    limit = max(.01, .01 * abs(total))
    excess = float(np.max(y[28]) - total)
    total_violation = bool(0 <= initial[28] <= total and excess > limit)
    return dict(tem1_total_violation=total_violation, tem1_excess=excess,
                minimum_raw=float(y.min()), maximum_raw=float(y.max()),
                significant_negative=bool(y.min() < -1e-4),
                negative_value_count=int((y < 0).sum()),
                negative_variables=np.flatnonzero((y < 0).any(axis=1)).tolist())
