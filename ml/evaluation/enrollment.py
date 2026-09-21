"""
Phase 8 — Enrollment + Verification module.

An **enrollment profile** summarises the behavioural snapshots a person has
registered (their reference sessions) as a single centroid embedding:

* ``enroll(sessions, …)``             ->  :class:`EnrollmentProfile`
* ``verify(profile, probe, …)``       ->  :class:`VerificationResult`

Verification is a *matching* problem

    distance(profile_centroid, probe_embedding)  <=  threshold  ->  VERIFIED

Distance is Phase 6 L2 (lower = more similar). ``user_id`` / ``session_id``
are metadata only — they are never part of any network input, so the profile
carries just a ``user_ref`` label for reporting.

Consistency rules
-----------------
* Profiles/probes must be embedded with the **same** keyboard/mouse scalers
  that were fitted on the training partition; mismatched scalers silently
  change distances, so callers are expected to thread the scalers through.
* The threshold is supplied by the caller (produced by
  :mod:`ml.evaluation.calibration`); verification never invents its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence

import torch

from ml.encoder.input import has_behavior


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of matching one probe against one enrollment profile."""

    user_id: str
    session_id: str
    distance: float
    threshold: float
    decision: str  # "VERIFIED" (d <= threshold) or "SUSPICIOUS"
    n_enrolled_sessions: int
    embedding_dim: int


@dataclass
class EnrollmentProfile:
    """Reference summary of one user's registered behaviour.

    Attributes:
        user_id: human-readable label only (never a model input).
        centroid: ``[embedding_dim]`` mean embedding of the reference sessions.
        n_sessions: number of reference sessions averaged in.
        embedding_dim: dimensionality of the embedding space.
    """

    user_id: str
    centroid: torch.Tensor
    n_sessions: int
    embedding_dim: int
    note: str = field(default="")


def enroll(
    sessions: Sequence[Mapping[str, Any]],
    verifier: Any,
    keyboard_scaler: Any = None,
    mouse_scaler: Any = None,
    user_ref: Optional[str] = None,
) -> EnrollmentProfile:
    """Build an enrollment profile from one user's reference sessions.

    Embeddings are averaged (centroid). Requires at least one session that has
    actual behavioural content.
    """
    if not isinstance(sessions, (list, tuple)) or not sessions:
        raise ValueError("enrollment requires at least one session")
    if any(not has_behavior(s) for s in sessions):
        raise ValueError("every enrolled session must have behavioural content")

    embed_fn = getattr(verifier, "embed_sessions", None)
    if embed_fn is None:
        raise TypeError("verifier must expose embed_sessions(...)")

    embeddings = embed_fn(sessions, keyboard_scaler, mouse_scaler)
    if embeddings.ndim != 2:
        raise RuntimeError("expected [N, D] embeddings, got shape {}".format(tuple(embeddings.shape)))
    if not bool(torch.isfinite(embeddings).all()):
        raise RuntimeError("enrollment embeddings contain non-finite values")

    centroid = embeddings.mean(dim=0, keepdim=False)
    embedding_dim = embeddings.shape[1]
    if user_ref is None:
        first = sessions[0]
        user_ref = str(first.get("user_id", "")) or first.get("session_id", "<unknown>")

    return EnrollmentProfile(
        user_id=user_ref,
        centroid=centroid,
        n_sessions=len(sessions),
        embedding_dim=embedding_dim,
        note="centroid of {} enrolled session(s)".format(len(sessions)),
    )


def verify(
    profile: EnrollmentProfile,
    probe: Mapping[str, Any],
    verifier: Any,
    threshold: float,
    keyboard_scaler: Any = None,
    mouse_scaler: Any = None,
) -> VerificationResult:
    """Match an actual probe session against an enrollment profile.

    Decision rule: ``distance(profile.centroid, probe_embedding) <= threshold``.
    """
    if not isinstance(profile, EnrollmentProfile):
        raise TypeError("profile must be an EnrollmentProfile")
    if not isinstance(probe, Mapping) or not has_behavior(probe):
        raise ValueError("probe must be a behavioural session dict")
    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise ValueError("threshold must be a non-negative number")

    probe_embedding = verifier.embed_session(probe, keyboard_scaler, mouse_scaler)
    if tuple(probe_embedding.shape) != (profile.embedding_dim,):
        raise RuntimeError(
            "probe embedding dim {} != profile dim {}".format(
                tuple(probe_embedding.shape), profile.embedding_dim
            )
        )
    d = float(verifier.pair_distance(profile.centroid[None, :], probe_embedding[None, :])[0])

    return VerificationResult(
        user_id=profile.user_id,
        session_id=str(probe.get("session_id", "<unknown>")),
        distance=d,
        threshold=float(threshold),
        decision="VERIFIED" if d <= threshold else "SUSPICIOUS",
        n_enrolled_sessions=profile.n_sessions,
        embedding_dim=profile.embedding_dim,
    )


__all__ = ["EnrollmentProfile", "VerificationResult", "enroll", "verify"]