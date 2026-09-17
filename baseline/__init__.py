"""
Behavioral Biometric Authentication — Phase 4 baseline package.

A simple, deterministic, **non-neural** statistical baseline for identity
verification. No machine-learning frameworks, tensors, or training loops.

For each user an **enrollment profile** is built from that user's training
sessions (per-feature keyboard/mouse mean + std). Query sessions are scored
against a profile as an average standardized deviation (a *distance*), kept
separate for keyboard and mouse before being combined.

Public API (re-exported from ``baseline``):

- :func:`build_profile` — enrollment profile from a user's training sessions.
- :class:`UserProfile` — the stored profile (mean/std + counts).
- :func:`score_session` — per-modality + combined score of one session vs a profile.
- :func:`calibrate_threshold` — threshold calibration on training/dev data only.
- :class:`BaselineEvaluator` — genuine/impostor evaluation + metrics.
"""

from .profile import UserProfile, build_profile
from .scoring import score_session
from .evaluator import (
    BaselineEvaluator,
    EvaluationReport,
    ThresholdResult,
    calibrate_threshold,
)

__version__ = "1.0.0"

__all__ = [
    "BaselineEvaluator",
    "EvaluationReport",
    "ThresholdResult",
    "UserProfile",
    "build_profile",
    "calibrate_threshold",
    "score_session",
]