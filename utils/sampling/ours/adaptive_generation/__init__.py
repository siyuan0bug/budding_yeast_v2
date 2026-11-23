"""Feedback-driven legal generation prototypes; no test/real/Pattern access."""
LEGACY_METHODS=('ours_response_generate','ours_event_generate','ours_progress_generate')
from ..response_v2 import METHODS as V2_METHODS
METHODS=LEGACY_METHODS+V2_METHODS
