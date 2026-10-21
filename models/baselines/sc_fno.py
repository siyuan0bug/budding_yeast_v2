"""SC-FNO is a sensitivity-regularized standard FNO, not a new modulation layer.

Forward architecture is exactly FNO. The trainer must use SensitivityLoss with
reference dy/dp targets (normalized consistently with trajectory and P).
"""
from .fno import FNO


class SCFNO(FNO):
    """Identical FNO architecture; distinction is the supervised sensitivity loss."""
    requires_sensitivity_targets = True
