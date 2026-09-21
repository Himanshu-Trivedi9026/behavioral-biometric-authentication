"""
Phase 8 — Threshold calibration (development- / validation-partition only).

Picks a single decision threshold ``t`` such that the **impostor acceptance
FAR is kept at or below a target budget**, as closely as the available discrete
thresholds permit.

Rule (mirrors Phase 4 baseline): scan candidate thresholds in ascending
distance and keep the *largest* threshold whose achieved FAR <= target — that
maximises TAR under the FAR budget, because both rates are monotone in
``t``. If no candidate meets the budget, the strictest observed threshold is
returned as the closest feasible fallback (FAR there is the smallest observed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Union

import numpy as np

Number = Union[int, float]


@dataclass(frozen=True)
class CalibrationResult:
    """Result of threshold calibration (development data only)."""

    threshold: float
    target_far: float
    achieved_far: float
    achieved_tar: float
    achieved_frr: float
    fallback: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "threshold": self.threshold,
            "target_far": self.target_far,
            "achieved_far": self.achieved_far,
            "achieved_tar": self.achieved_tar,
            "achieved_frr": self.achieved_frr,
            "fallback": self.fallback,
        }


def calibrate_threshold(
    genuine_distances: Sequence[Number],
    impostor_distances: Sequence[Number],
    target_far: float = 0.05,
) -> CalibrationResult:
    """Calibrate against a target FAR from development distances only."""
    gen = np.asarray(genuine_distances, dtype=np.float64)
    imp = np.asarray(impostor_distances, dtype=np.float64)
    if gen.ndim != 1 or imp.ndim != 1:
        raise ValueError("distances must be 1-D sequences")
    if gen.size == 0 or imp.size == 0:
        raise ValueError(
            "calibration requires both genuine and impostor distances "
            "(got genuine={}, impostor={})".format(gen.size, imp.size)
        )
    if not (0.0 <= target_far <= 1.0):
        raise ValueError("target_far must be in [0, 1]")

    candidates = np.unique(np.concatenate([gen, imp]))
    chosen = None
    for t in candidates:
        far = float((imp <= t).mean())
        if far <= target_far:
            chosen = float(t)  # keep the largest threshold meeting the budget
    fallback = chosen is None
    if fallback:
        chosen = float(np.min(candidates))
    far = float((imp <= chosen).mean())
    tar = float((gen <= chosen).mean())
    return CalibrationResult(
        threshold=chosen,
        target_far=float(target_far),
        achieved_far=far,
        achieved_tar=tar,
        achieved_frr=1.0 - tar,
        fallback=bool(fallback),
    )


__all__ = ["CalibrationResult", "calibrate_threshold"]