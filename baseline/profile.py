"""
Enrollment profiles for the baseline.

A :class:`UserProfile` summarizes one user's behavioural fingerprint using
only simple statistics of their *training* sessions:

- :attr:`keyboard` — per feature in ``KEYBOARD_FEATURE_COLUMNS``:
  ``(mean, std)`` over ALL pooled training samples of that feature.
- :attr:`mouse` — per feature in ``MOUSE_FEATURE_COLUMNS``: ``(mean, std)``.

Pooling all samples across the user's training sessions (rather than per
session) keeps the estimator stable for the small session counts we have and
mirrors how the Phase 3 ``FeatureScaler`` is later reused for training.

Constant features (std == 0) are stored as-is; scoring treats them as
carrying no discriminating signal (deviation term = 0) so the code never
divides by zero.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from dataset.schema import KEYBOARD_FEATURE_COLUMNS, MOUSE_FEATURE_COLUMNS

FeatureStats = Dict[str, Tuple[float, float]]  # feature -> (mean, std)


class UserProfile:
    """Statistical enrollment profile for one user."""

    def __init__(
        self,
        user_id: str,
        keyboard: FeatureStats,
        mouse: FeatureStats,
        n_train_sessions: int,
        keyboard_samples: int,
        mouse_samples: int,
    ) -> None:
        self.user_id = user_id
        self.keyboard = keyboard
        self.mouse = mouse
        self.n_train_sessions = n_train_sessions
        self.keyboard_samples = keyboard_samples
        self.mouse_samples = mouse_samples

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "keyboard": self.keyboard,
            "mouse": self.mouse,
            "n_train_sessions": self.n_train_sessions,
            "keyboard_samples": self.keyboard_samples,
            "mouse_samples": self.mouse_samples,
        }

    def __repr__(self) -> str:
        return "<UserProfile {} ({} kbd, {} mouse samples)>".format(
            self.user_id, self.keyboard_samples, self.mouse_samples
        )


def _pooled_stats(values: List[float]) -> Tuple[float, float] | None:
    """Return (mean, population-std) or None if there are no values."""
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(variance)


def _feature_stat(rows: Sequence[Dict], feature: str) -> Tuple[float, float] | None:
    """Pool one feature across a list of feature dicts -> (mean, std) or None."""
    values = [float(row[feature]) for row in rows if feature in row]
    return _pooled_stats(values)


def build_profile(user_id: str, train_entries: Sequence[Dict]) -> UserProfile:
    """Build an enrollment profile from one user's training sessions.

    Args:
        user_id: dataset-level identity label.
        train_entries: this user's training sessions (dataset entries). Each
            entry must have ``keyboard_sequence`` and ``mouse_sequence``.

    Returns:
        :class:`UserProfile` with per-feature pooled mean/std. Features with
        no samples at all in training are omitted from the profile (scoring
        ignores them for that user).
    """
    keyboard_rows: List[Dict] = []
    mouse_rows: List[Dict] = []

    for entry in train_entries:
        keyboard_rows.extend(entry.get("keyboard_sequence") or [])
        mouse_rows.extend(entry.get("mouse_sequence") or [])

    keyboard: FeatureStats = {}
    for feature in KEYBOARD_FEATURE_COLUMNS:
        stats = _feature_stat(keyboard_rows, feature)
        if stats is not None:
            keyboard[feature] = stats

    mouse: FeatureStats = {}
    for feature in MOUSE_FEATURE_COLUMNS:
        stats = _feature_stat(mouse_rows, feature)
        if stats is not None:
            mouse[feature] = stats

    return UserProfile(
        user_id=user_id,
        keyboard=keyboard,
        mouse=mouse,
        n_train_sessions=len(train_entries),
        keyboard_samples=len(keyboard_rows),
        mouse_samples=len(mouse_rows),
    )


def build_profiles(train_entries: Sequence[Dict]) -> Dict[str, UserProfile]:
    """Build one profile per user present in the training entries."""
    by_user: Dict[str, List[Dict]] = {}
    for entry in train_entries:
        by_user.setdefault(entry["user_id"], []).append(entry)

    profiles: Dict[str, UserProfile] = {}
    for user_id in sorted(by_user):
        profiles[user_id] = build_profile(user_id, by_user[user_id])
    return profiles


__all__ = ["UserProfile", "build_profile", "build_profiles"]