"""
Behavioral Biometric Authentication — Phase 3 preprocessing package.

Converts a raw Phase 2 session (browser-collected keyboard/mouse event JSON)
into machine-learning-ready behavioral sequences.

Public API (re-exported from ``ml.preprocessing``):

- :func:`process_session` — one-call pipeline: validate → preprocess →
  extract features → return the processed representation. Deterministic and
  side-effect free (the raw input is never mutated).
- :func:`validate_session` — returns a list of human-readable error strings
  (empty list means the session is valid).
- :class:`SessionValidationError` — raised by ``process_session`` when a raw
  session fails validation; carries the individual error messages.
- :class:`FeatureScaler` — reusable per-feature standardization with
  fit / transform / fit_transform and JSON-serializable state for reuse on
  validation/test/inference data later.
- :func:`preprocess_keyboard` / :func:`preprocess_mouse` — lower-level
  preprocessing entry points.

Privacy invariant: keyboard and mouse sequences contain NUMERICAL timing /
movement features only. No key names, typed characters, passwords, input
values, or screen identity information are ever emitted.
"""

from .validation import (
    SessionValidationError,
    validate_session,
)
from .keyboard import (
    KEYBOARD_FEATURE_COLUMNS,
    preprocess_keyboard,
)
from .mouse import (
    MOUSE_ACTION_EVENTS,
    MOUSE_FEATURE_COLUMNS,
    preprocess_mouse,
)
from .normalization import (
    FeatureScaler,
    rows_to_sequence,
    sequence_to_rows,
)
from .pipeline import process_session

__version__ = "1.0.0"

__all__ = [
    "KEYBOARD_FEATURE_COLUMNS",
    "MOUSE_ACTION_EVENTS",
    "MOUSE_FEATURE_COLUMNS",
    "FeatureScaler",
    "SessionValidationError",
    "preprocess_keyboard",
    "preprocess_mouse",
    "process_session",
    "rows_to_sequence",
    "sequence_to_rows",
    "validate_session",
]