"""
Scoring a query session against an enrollment profile.

**Score direction**: scores are *distances* — the closer to ``0``, the more
similar the query is to the claimed user's profile. Larger values = worse
match. The verification decision is ``accept if combined <= threshold``.

Per-modality score (``score_modality``)
    For feature ``f`` the session's pooled values are compared to the
    profile's ``(mean, std)`` via the mean absolute standardized deviation::

        d_f = mean( |v - mean_f| / std_f   for v in session values of f )

    If ``std_f <= 0`` (constant feature in training) the term contributes
    ``0`` (no signal) — the code never divides by zero.

    ``keyboard_score`` = mean of ``d_f`` over keyboard features present in
    both the profile and the session.
    ``mouse_score``    = same for mouse trajectory features.

    Because each term is an average over the samples that actually exist,
    sessions with different lengths are handled safely and naturally.

Combined score
    Weighted mean of the two modality scores (default 0.5/0.5). If one
    modality has no data, the combined score falls back to the available
    modality alone (weights are renormalized); if both are empty the combined
    score is ``None``.

Absent data
    A modality with zero samples returns ``None`` for that modality's score
    instead of crashing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from dataset.schema import KEYBOARD_FEATURE_COLUMNS, MOUSE_FEATURE_COLUMNS

from .profile import UserProfile

DEFAULT_COMBINED_WEIGHTS = {"keyboard": 0.5, "mouse": 0.5}


def _pool_feature_values(rows: Sequence[Dict], feature: str) -> List[float]:
    return [float(row[feature]) for row in rows if feature in row]


def _feature_deviation(values: Sequence[float], mean: float, std: float) -> float:
    """Mean absolute standardized deviation of ``values`` from ``(mean, std)``."""
    if not values:
        return 0.0
    if std <= 0.0:
        return 0.0  # constant feature: no discriminating signal
    return sum(abs(v - mean) / std for v in values) / len(values)


def score_modality(
    rows: Sequence[Dict], feature_columns: Sequence[str], stats: Dict[str, Tuple[float, float]]
) -> Tuple[Optional[float], Dict[str, float]]:
    """Score one modality; returns ``(score | None, per-feature breakdown)``.

    ``score`` is ``None`` when the session has no sample rows for this
    modality. Features with no profile stats are ignored.
    """
    if not rows:
        return None, {}

    deviations: Dict[str, float] = {}
    for feature in feature_columns:
        if feature not in stats:
            continue
        values = _pool_feature_values(rows, feature)
        if not values:
            continue
        mean, std = stats[feature]
        deviations[feature] = _feature_deviation(values, mean, std)

    if not deviations:
        return None, deviations

    score = sum(deviations.values()) / len(deviations)
    return score, deviations


def score_session(
    profile: UserProfile,
    session: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Score one query session against a claimed user's profile.

    Args:
        profile: the claimed user's enrollment profile.
        session: one dataset entry (a query session).
        weights: optional ``{"keyboard": .., "mouse": ..}`` for the combined
            score (must sum such that == 1 when renormalized; defaults 0.5/0.5).

    Returns:
        A dict with ``keyboard``, ``mouse``, ``combined`` scores (``None``
        where a modality has no data), sample counts, and per-feature
        breakdowns.
    """
    keyboard_rows = session.get("keyboard_sequence") or []
    mouse_rows = session.get("mouse_sequence") or []

    keyboard_score, keyboard_breakdown = score_modality(
        keyboard_rows, KEYBOARD_FEATURE_COLUMNS, profile.keyboard
    )
    mouse_score, mouse_breakdown = score_modality(
        mouse_rows, MOUSE_FEATURE_COLUMNS, profile.mouse
    )

    combined = _combine(keyboard_score, mouse_score, weights)

    return {
        "keyboard": keyboard_score,
        "mouse": mouse_score,
        "combined": combined,
        "keyboard_samples": len(keyboard_rows),
        "mouse_samples": len(mouse_rows),
        "keyboard_features": keyboard_breakdown,
        "mouse_features": mouse_breakdown,
    }


def _combine(
    keyboard_score: Optional[float],
    mouse_score: Optional[float],
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    w = dict(DEFAULT_COMBINED_WEIGHTS)
    if weights:
        if "keyboard" in weights:
            w["keyboard"] = float(weights["keyboard"])
        if "mouse" in weights:
            w["mouse"] = float(weights["mouse"])

    if keyboard_score is not None and mouse_score is not None:
        total = w["keyboard"] + w["mouse"]
        if total <= 0:
            raise ValueError("combined weights must be positive")
        return (w["keyboard"] * keyboard_score + w["mouse"] * mouse_score) / total
    if keyboard_score is not None:
        return keyboard_score
    if mouse_score is not None:
        return mouse_score
    return None


__all__ = ["score_modality", "score_session"]