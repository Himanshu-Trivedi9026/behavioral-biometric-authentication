"""
Baseline evaluation: genuine/impostor attempts, metrics, and threshold.

Workflow
--------
1. ``build_profiles(train_entries)`` -> per-user enrollment profiles
   (enrollment/training data only).
2. ``collect_scores(profiles, test_entries)`` -> two distance lists:
   * **genuine** — each test session scored against its TRUE owner's profile.
   * **impostor** — each test session scored against every OTHER user's
     profile (i.e. the session *claims* a different identity).
3. A threshold turns distances into a decision: ``accept if combined <= t``.
4. Metrics at a threshold: TAR, FAR, FRR, accuracy, confusion-matrix counts.

Threshold
---------
- Configurable (never hard-coded).
- ``calibrate_threshold`` picks it on **training/development data only** —
  the test set is never used to choose a threshold.

The structure is intentionally model-agnostic: the same ``collect_scores`` /
``evaluate`` machinery can later be reused with a neural similarity model by
providing a different score callable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .profile import UserProfile, build_profile, build_profiles
from .scoring import score_session


# ---------------------------------------------------------------------------
# Attempt collection
# ---------------------------------------------------------------------------

def collect_scores(
    profiles: Dict[str, UserProfile],
    test_entries: Sequence[Dict],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Score all genuine + impostor attempts over the test sessions.

    Returns ``(genuine, impostor)`` lists of attempt dicts:

    .. code-block:: python

        {
          "true_user": str, "claimed_user": str, "session_id": str,
          "keyboard": float|None, "mouse": float|None, "combined": float|None,
          "is_genuine": bool,
        }
    """
    genuine: List[Dict[str, Any]] = []
    impostor: List[Dict[str, Any]] = []

    for entry in test_entries:
        owner = entry["user_id"]
        if owner not in profiles:
            raise ValueError(
                "test session '{}' belongs to user '{}' which has no trained "
                "profile".format(entry["session_id"], owner)
            )
        owner_profile = profiles[owner]
        genuine.append(_make_attempt(entry, owner_profile, owner, True))

        for claimed in sorted(profiles):
            if claimed == owner:
                continue
            impostor.append(
                _make_attempt(entry, profiles[claimed], claimed, False)
            )

    return genuine, impostor


def _make_attempt(
    entry: Dict[str, Any],
    profile: UserProfile,
    claimed_user: str,
    is_genuine: bool,
) -> Dict[str, Any]:
    result = score_session(profile, entry)
    return {
        "true_user": entry["user_id"],
        "claimed_user": claimed_user,
        "session_id": entry["session_id"],
        "keyboard": result["keyboard"],
        "mouse": result["mouse"],
        "combined": result["combined"],
        "is_genuine": is_genuine,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_at_threshold(
    genuine: Sequence[Dict[str, Any]],
    impostor: Sequence[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    """Compute verification metrics for a given distance threshold.

    Decision rule: accept if ``combined <= threshold``.

    Attempts whose combined score is ``None`` (session had no usable
    keyboard OR mouse data) are excluded from TAR/FAR/FPR/accuracy and
    reported separately as "unscorable".
    """
    gen_scores = [a["combined"] for a in genuine if a["combined"] is not None]
    imp_scores = [a["combined"] for a in impostor if a["combined"] is not None]

    tp = sum(1 for s in gen_scores if s <= threshold)
    fp = sum(1 for s in imp_scores if s <= threshold)
    fn = len(gen_scores) - tp
    tn = len(imp_scores) - fp

    tar = tp / len(gen_scores) if gen_scores else 0.0
    far = fp / len(imp_scores) if imp_scores else 0.0
    frr = 1.0 - tar
    denom = len(gen_scores) + len(imp_scores)
    accuracy = (tp + tn) / denom if denom else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tar": tar,
        "far": far,
        "frr": frr,
        "accuracy": accuracy,
        "n_genuine": len(gen_scores),
        "n_impostor": len(imp_scores),
        "unscorable_genuine": len(genuine) - len(gen_scores),
        "unscorable_impostor": len(impostor) - len(imp_scores),
    }


def sweep_thresholds(
    genuine: Sequence[Dict[str, Any]],
    impostor: Sequence[Dict[str, Any]],
    n_points: int = 40,
) -> List[Dict[str, Any]]:
    """ROC-style sweep over candidate thresholds (distance domain).

    Thresholds are sampled between the observed min and max combined distance
    plus the two extreme decision points (reject-all / accept-all). Each point
    is ``{"threshold", "far", "tar", "accuracy"}``.
    """
    gen = [a["combined"] for a in genuine if a["combined"] is not None]
    imp = [a["combined"] for a in impostor if a["combined"] is not None]
    all_scores = gen + imp

    if not all_scores:
        return []

    lo = min(all_scores)
    hi = max(all_scores)

    if math.isclose(lo, hi):
        candidates = [lo]
    else:
        step = (hi - lo) / max(n_points - 2, 1)
        candidates = [lo + step * i for i in range(n_points - 2)] + [hi]

    # Extrema: -inf accepts everything, +inf rejects everything.
    points = []
    for t in [-float("inf")] + candidates + [float("inf")]:
        metrics = evaluate_at_threshold(genuine, impostor, t)
        points.append(
            {
                "threshold": t,
                "far": metrics["far"],
                "tar": metrics["tar"],
                "accuracy": metrics["accuracy"],
            }
        )
    return points


# ---------------------------------------------------------------------------
# Calibration (training/development data only)
# ---------------------------------------------------------------------------

def _development_scores(train_entries: Sequence[Dict]) -> Tuple[List[float], List[float]]:
    """Leave-one-out genuine + cross-user impostor distances on training data.

    Genuine: a train session scored against its own user's profile built from
    the user's OTHER train sessions (leave-one-out) to avoid self-bias.
    Impostor: the same train session scored against every other user's full
    profile.

    Returns ``(genuine_dev, impostor_dev)`` distance lists.
    """
    by_user: Dict[str, List[Dict]] = {}
    for entry in train_entries:
        by_user.setdefault(entry["user_id"], []).append(entry)

    genuine_dev: List[float] = []
    impostor_dev: List[float] = []

    impostor_profiles = build_profiles(train_entries)

    for user_id in sorted(by_user):
        sessions = by_user[user_id]
        for excluded in sessions:
            others = [s for s in sessions if s["session_id"] != excluded["session_id"]]
            # Impostor side: score against every other user's full profile.
            for claimed in sorted(impostor_profiles):
                if claimed == user_id:
                    continue
                result = score_session(impostor_profiles[claimed], excluded)
                if result["combined"] is not None:
                    impostor_dev.append(result["combined"])
            # Genuine side: leave-one-out (needs >= 2 train sessions).
            if len(others) < 1:
                continue
            loo_profile = build_profile(user_id, others)
            result = score_session(loo_profile, excluded)
            if result["combined"] is not None:
                genuine_dev.append(result["combined"])

    return genuine_dev, impostor_dev


def _far_at(impostor_dev: Sequence[float], threshold: float) -> float:
    if not impostor_dev:
        return 0.0
    return sum(1 for s in impostor_dev if s <= threshold) / len(impostor_dev)


def _tar_at(genuine_dev: Sequence[float], threshold: float) -> float:
    if not genuine_dev:
        return 0.0
    return sum(1 for s in genuine_dev if s <= threshold) / len(genuine_dev)


def _as_float_threshold(value: Optional[float]) -> float:
    """Coerce a possible ``None`` candidate to +inf (reject-all sentinel)."""
    return value if value is not None else float("inf")


class ThresholdResult:
    """Result of calibrating a decision threshold on development data."""

    def __init__(
        self,
        threshold: float,
        method: str,
        far_target: Optional[float],
        genuine_dev: Sequence[float],
        impostor_dev: Sequence[float],
    ) -> None:
        self.threshold = threshold
        self.method = method
        self.far_target = far_target
        self.genuine_dev = list(genuine_dev)
        self.impostor_dev = list(impostor_dev)

    @property
    def dev_tar(self) -> float:
        return _tar_at(self.genuine_dev, self.threshold)

    @property
    def dev_far(self) -> float:
        return _far_at(self.impostor_dev, self.threshold)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "method": self.method,
            "far_target": self.far_target,
            "dev_tar": self.dev_tar,
            "dev_far": self.dev_far,
            "n_genuine_dev": len(self.genuine_dev),
            "n_impostor_dev": len(self.impostor_dev),
        }

    def __repr__(self) -> str:
        return "<ThresholdResult t={:.4f} method={} dev_far={:.3f}>".format(
            self.threshold, self.method, self.dev_far
        )


def calibrate_threshold(
    train_entries: Sequence[Dict],
    *,
    far_target: float = 0.05,
    method: str = "target-far",
) -> ThresholdResult:
    """Choose a combined-score threshold using training data ONLY.

    Score direction is distance: lower = more similar. ``accept iff
    combined <= threshold``, so a larger threshold = more permissive.

    Methods (documented; both use only development data):
      * ``"target-far"`` — most-permissive threshold whose development
        impostor FAR is still <= ``far_target``. If the FAR budget cannot be
        met by any threshold, falls back to the strictest observed threshold
        (rejects everything on dev data).
      * ``"max-tar-far"`` — threshold maximizing ``TAR - FAR`` (Youden index)
        on development data; ties resolved toward the stricter threshold.
    """
    if method not in ("target-far", "max-tar-far"):
        raise ValueError("unknown calibration method {!r}".format(method))
    if not (0.0 <= far_target <= 1.0):
        raise ValueError("far_target must be in [0, 1]")

    genuine_dev, impostor_dev = _development_scores(train_entries)

    if not impostor_dev:
        raise ValueError(
            "cannot calibrate: no impostor development distances were produced "
            "(need at least 2 users in the training data)"
        )

    if method == "target-far":
        # Scanning from most-permissive (highest threshold) downwards, pick the
        # first threshold that meets the FAR budget.
        candidate = sorted({s for s in impostor_dev + genuine_dev})
        chosen = None
        for t in reversed(candidate):
            if _far_at(impostor_dev, t) <= far_target:
                chosen = t
                break
        if chosen is None:
            # Budget not achievable: fall back to strictest observed threshold.
            chosen = min(candidate)
        threshold = float(chosen)
    else:  # max-tar-far
        candidate = sorted({s for s in impostor_dev + genuine_dev})
        best = None
        best_score = None
        for t in candidate:
            score = _tar_at(genuine_dev, t) - _far_at(impostor_dev, t)
            # Strictness tie-break: prefer smaller threshold when scores tie.
            if best_score is None or score > best_score or (
                abs(score - best_score) < 1e-12
                and t < _as_float_threshold(best)
            ):
                best = t
                best_score = score
        threshold = float(_as_float_threshold(best))

    return ThresholdResult(
        threshold=threshold,
        method=method,
        far_target=far_target,
        genuine_dev=genuine_dev,
        impostor_dev=impostor_dev,
    )


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

class EvaluationReport:
    """Full baseline evaluation summary (JSON-serializable)."""

    def __init__(
        self,
        profiles: Dict[str, UserProfile],
        genuine: Sequence[Dict[str, Any]],
        impostor: Sequence[Dict[str, Any]],
        threshold_result: ThresholdResult,
        threshold_override: Optional[float],
    ) -> None:
        self.profiles = profiles
        self.genuine = list(genuine)
        self.impostor = list(impostor)

        threshold = (
            threshold_override
            if threshold_override is not None
            else threshold_result.threshold
        )
        self.threshold = threshold
        self.threshold_used = threshold_override is None  # True if calibrated
        self.threshold_result = threshold_result

        self.metrics = evaluate_at_threshold(self.genuine, self.impostor, self.threshold)
        self.roc_points = sweep_thresholds(self.genuine, self.impostor)

    @property
    def threshold_source(self) -> str:
        return "calibrated" if self.threshold_used else "explicit"

    def as_dict(self) -> Dict[str, Any]:
        gen_scores = [a["combined"] for a in self.genuine if a["combined"] is not None]
        imp_scores = [a["combined"] for a in self.impostor if a["combined"] is not None]
        return {
            "schema_version": "1.0.0",
            "n_profiles": len(self.profiles),
            "profiles": {
                uid: profile.to_dict() for uid, profile in sorted(self.profiles.items())
            },
            "threshold": self.threshold,
            "threshold_source": "calibrated" if self.threshold_used else "explicit",
            "calibration": self.threshold_result.to_dict(),
            "metrics": self.metrics,
            "attempts": {
                "genuine": self.genuine,
                "impostor": self.impostor,
            },
            "genuine_distances": gen_scores,
            "impostor_distances": imp_scores,
            "roc_points": self.roc_points,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class BaselineEvaluator:
    """Fit on training entries, then evaluate held-out test sessions.

    Generic design: ``fit`` builds the user profiles, ``evaluate`` scores
    genuine + impostor attempts and produces an :class:`EvaluationReport`.
    """

    def __init__(self) -> None:
        self.profiles: Dict[str, UserProfile] = {}

    def fit(self, train_entries: Sequence[Dict]) -> "BaselineEvaluator":
        """Build one enrollment profile per user from the training entries."""
        if not train_entries:
            raise ValueError("cannot fit a baseline with an empty training set")
        self.profiles = build_profiles(train_entries)
        if not self.profiles:
            raise ValueError("training set produced no user profiles")
        return self

    def evaluate(
        self,
        test_entries: Sequence[Dict],
        *,
        threshold: Optional[float] = None,
        calibrate: bool = True,
        far_target: float = 0.05,
        method: str = "target-far",
        train_entries: Optional[Sequence[Dict]] = None,
    ) -> EvaluationReport:
        """Evaluate held-out sessions at a threshold.

        Args:
            test_entries: held-out sessions to verify.
            threshold: explicit combined-score threshold. If ``None`` and
                ``calibrate`` is True, calibrated from ``train_entries``
                (must be supplied or else the fitted profile's training
                data is unavailable — pass the training set explicitly).
            calibrate: whether to auto-calibrate when no threshold is given.
            far_target / method: passed to :func:`calibrate_threshold`.
            train_entries: training data used for calibration (test data is
                NEVER used to pick a threshold).

        Raises:
            ValueError: if neither ``threshold`` nor (``calibrate`` + valid
                training data) is available.
        """
        if not self.profiles:
            raise ValueError("call fit() with training entries before evaluate()")

        if threshold is None:
            if not calibrate:
                raise ValueError(
                    "no threshold given and calibration disabled; pass threshold=..."
                )
            if train_entries is None or len(train_entries) == 0:
                raise ValueError(
                    "calibration requires train_entries (the same data used "
                    "for fitting)"
                )
            threshold_result = calibrate_threshold(
                train_entries, far_target=far_target, method=method
            )
            threshold_result_or_override = threshold_result
            threshold_override = None
        else:
            # Explicit user-supplied threshold: keep it; still report a
            # reference calibration result if training data is available.
            if train_entries and len(train_entries) > 0:
                threshold_result = calibrate_threshold(
                    train_entries, far_target=far_target, method=method
                )
            else:
                threshold_result = ThresholdResult(
                    threshold=threshold,
                    method="explicit",
                    far_target=far_target,
                    genuine_dev=[],
                    impostor_dev=[],
                )
            threshold_result_or_override = threshold_result
            threshold_override = threshold

        genuine, impostor = collect_scores(self.profiles, test_entries)
        return EvaluationReport(
            profiles=self.profiles,
            genuine=genuine,
            impostor=impostor,
            threshold_result=threshold_result_or_override,
            threshold_override=threshold_override,
        )


__all__ = [
    "BaselineEvaluator",
    "EvaluationReport",
    "ThresholdResult",
    "calibrate_threshold",
    "collect_scores",
    "evaluate_at_threshold",
    "sweep_thresholds",
]