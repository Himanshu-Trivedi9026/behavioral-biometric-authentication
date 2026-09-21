"""
Phase 8 — Enrollment + Verification evaluation package.

Built on the Phase 7 trained checkpoint, this package covers the full
enrollment/verification evaluation stack:

* :mod:`ml.evaluation.model` — checkpoint loading + inference-only embedding
  extraction (:class:`~ml.evaluation.model.LoadedVerifier`).
* :mod:`ml.evaluation.enrollment` — enrollment profiles (centroid embeddings)
  and matching (:class:`~ml.evaluation.enrollment.EnrollmentProfile`,
  :class:`~ml.evaluation.enrollment.VerificationResult`).
* :mod:`ml.evaluation.metrics` — FAR / TAR / FRR, confusion-matrix counts,
  ROC points, ROC AUC, EER.
* :mod:`ml.evaluation.calibration` — dev-data threshold calibration against a
  target FAR.
* :mod:`ml.evaluation.evaluator` — end-to-end evaluation returning a
  JSON-serialisable report.

The Phase 7 checkpoint is never modified and the model is never trained here;
scalers fitted on the training partition are reused for dist / test sessions.
"""

from .calibration import CalibrationResult, calibrate_threshold
from .enrollment import EnrollmentProfile, VerificationResult, enroll, verify
from .evaluator import (
    TARGET_FAR,
    build_partition_pairs,
    evaluate_verification,
    save_evaluation_report,
)
from .metrics import auc, eer, evaluate_at_threshold, roc_points
from .model import LoadedVerifier, load_verifier

__version__ = "8.0.0"

__all__ = [
    "CalibrationResult",
    "EnrollmentProfile",
    "LoadedVerifier",
    "TARGET_FAR",
    "VerificationResult",
    "auc",
    "build_partition_pairs",
    "calibrate_threshold",
    "eer",
    "enroll",
    "evaluate_at_threshold",
    "evaluate_verification",
    "load_verifier",
    "roc_points",
    "save_evaluation_report",
    "verify",
]