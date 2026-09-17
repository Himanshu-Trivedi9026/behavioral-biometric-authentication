"""
Deterministic distance→decision scoring for Phase 6 (small, non-production).

Purpose
-------
Translate a Siamese pair distance into a simple binary verification decision
using a **fixed, caller-supplied** threshold::

    distance <= threshold  ->  accept  (genuine)
    distance >  threshold  ->  reject  (impostor)

The relationship is intentionally simple: *smaller distance = more similar
behavioural pattern*.

Non-goals (Phase 6 scope boundary)
----------------------------------
* No adaptive thresholding.
* No ROC/AUC/EER evaluation (Phase 10).
* No claims that any threshold is production-ready — thresholds here are
  arbitrary analysis values; a production-grade threshold requires Phase 7+
  training / calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Union

import torch

DistanceLike = Union[float, int, torch.Tensor]


@dataclass(frozen=True)
class VerificationDecision:
    """Result of comparing one pair distance against a threshold.

    Attributes:
        distance:   the pair distance (as a Python float).
        threshold:  the threshold it was compared with.
        accepted:   ``True`` if ``distance <= threshold`` (genuine decision).
    """

    distance: float
    threshold: float
    accepted: bool

    def to_dict(self) -> dict:
        return {
            "distance": self.distance,
            "threshold": self.threshold,
            "accepted": self.accepted,
        }


def _as_float(value: DistanceLike, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("{} must be a scalar tensor, got shape {}".format(name, tuple(value.shape)))
        number = float(value.detach().cpu().item())
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{} must be a finite number or scalar tensor, got {!r}".format(name, type(value).__name__))
    else:
        number = float(value)
    if not math.isfinite(number):
        raise ValueError("{} must be finite, got {!r}".format(name, number))
    return number


def decide(distance: DistanceLike, threshold: DistanceLike) -> bool:
    """Return ``True`` (accept) iff ``distance <= threshold``."""
    d = _as_float(distance, "distance")
    t = _as_float(threshold, "threshold")
    if d < 0.0:
        raise ValueError("distance must be non-negative (L2 distance), got {!r}".format(d))
    if t < 0.0:
        raise ValueError("threshold must be non-negative, got {!r}".format(t))
    return d <= t


def decision(distance: DistanceLike, threshold: DistanceLike) -> VerificationDecision:
    """Full decision record for a single pair distance."""
    d = _as_float(distance, "distance")
    t = _as_float(threshold, "threshold")
    if d < 0.0:
        raise ValueError("distance must be non-negative (L2 distance), got {!r}".format(d))
    if t < 0.0:
        raise ValueError("threshold must be non-negative, got {!r}".format(t))
    return VerificationDecision(distance=d, threshold=t, accepted=d <= t)


def decide_batch(distances: Sequence[DistanceLike], threshold: DistanceLike) -> List[bool]:
    """Apply :func:`decide` to a sequence of distances (batched decision)."""
    t = _as_float(threshold, "threshold")
    if t < 0.0:
        raise ValueError("threshold must be non-negative, got {!r}".format(t))
    return [decide(d, t) for d in distances]


def default_threshold() -> float:
    """A neutral placeholder threshold.

    .. warning::
       This is NOT production-ready, NOT calibrated, and carries no accuracy
       guarantee. It exists only so Phase 6 analysis code has a well-defined,
       documented default. Production thresholds belong to Phase 7+.
    """
    return 1.0


__all__ = [
    "VerificationDecision",
    "decide",
    "decide_batch",
    "decision",
    "default_threshold",
]