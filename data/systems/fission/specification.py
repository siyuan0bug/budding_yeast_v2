"""Fission dimensions/scales are independent of Budding's 39/141 contract."""
from ...generation.contracts import SystemSpec
from .constants import VAR_NAMES, TF_MIN, N_T, MASS_UNIT
from .parameters import EXPORT_KEYS

SPEC = SystemSpec('fission', 'sveiczer_server_v2', tuple(VAR_NAMES), tuple(EXPORT_KEYS),
                  TF_MIN, N_T, 'sveiczer_table2_20', MASS_UNIT)
