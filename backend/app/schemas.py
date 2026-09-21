"""
Phase 9B — Pydantic request/response schemas.

The API accepts the SAME raw Phase 2 session structure produced by the browser
collector (``keyboard_events`` / ``mouse_events``), so the payload models here
mirror :mod:`ml.preprocessing.validation` exactly:

* ``KeyboardEvent`` / ``MouseEvent`` mirror the raw collector events.
* ``RawSession`` mirrors the raw session export (identifier, wall-clock
  reference, and the two event lists).
* :class:`EnrollmentRequest` wraps one or more raw sessions for a ``user_ref``.
* :class:`VerificationRequest` wraps exactly one probe session.

Validation policy
-----------------
* Numbers are finite floats (``NaN``/``±inf`` are rejected at the schema
  boundary with a 422 before any ML code runs).
* ``user_ref`` is a non-blank string (identity metadata only — never a model
  feature and never authentication).
* Sensible payload limits prevent unbounded request bodies.

Response models never expose raw events, embeddings, centroids, or scaler
internals — only the small, documented fields below.
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_ENROLLMENT_SESSIONS = 16
MAX_SESSION_EVENTS = 20000
MAX_SESSION_ID_LENGTH = 128
MAX_TIMESTAMP_STRING_LENGTH = 256
MAX_USER_REF_LENGTH = 64

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


def validate_user_ref(value: str) -> str:
    """Non-blank, reasonable-length ``user_ref`` (metadata only)."""
    if not value or not value.strip():
        raise ValueError("user_ref must be a non-empty string")
    if len(value) > MAX_USER_REF_LENGTH:
        raise ValueError(
            "user_ref must be at most {} characters".format(MAX_USER_REF_LENGTH)
        )
    return value


class KeyboardEvent(BaseModel):
    """A raw keyboard event produced by the Phase 2 collector."""

    event_type: Literal["keyboard"] = Field(
        description="Collector tag; always 'keyboard'"
    )
    event: Literal["keydown", "keyup"]
    timestamp: FiniteFloat


class MouseEvent(BaseModel):
    """A raw mouse event produced by the Phase 2 collector."""

    event_type: Literal["mouse"] = Field(description="Collector tag; always 'mouse'")
    event: Literal["mousemove", "mousedown", "mouseup"]
    x: FiniteFloat
    y: FiniteFloat
    timestamp: FiniteFloat


class RawSession(BaseModel):
    """A raw Phase 2 behavioral session (browser collector export)."""

    session_id: str = Field(min_length=1, max_length=MAX_SESSION_ID_LENGTH)
    started_at: str = Field(min_length=1, max_length=MAX_TIMESTAMP_STRING_LENGTH)
    ended_at: str = Field(min_length=1, max_length=MAX_TIMESTAMP_STRING_LENGTH)
    timestamp_source: Optional[str] = Field(
        default=None, max_length=MAX_TIMESTAMP_STRING_LENGTH
    )
    keyboard_events: List[KeyboardEvent] = Field(max_length=MAX_SESSION_EVENTS)
    mouse_events: List[MouseEvent] = Field(max_length=MAX_SESSION_EVENTS)

    @model_validator(mode="after")
    def _enforce_total_event_budget(self) -> "RawSession":
        total = len(self.keyboard_events) + len(self.mouse_events)
        if total > MAX_SESSION_EVENTS:
            raise ValueError(
                "a session may contain at most {} events".format(MAX_SESSION_EVENTS)
            )
        return self


class EnrollmentRequest(BaseModel):
    """Enrollment payload: one user, one or more raw behavioral sessions.

    ``user_ref`` is optional (Phase 11): when omitted the authenticated
    identity is used. When present it is validated for shape here and MUST
    equal the authenticated user's identity (enforced by the route, 403
    otherwise). It is never trusted as an authorization source.
    """

    user_ref: Optional[str] = Field(default=None, max_length=MAX_USER_REF_LENGTH)
    sessions: List[RawSession] = Field(
        min_length=1, max_length=MAX_ENROLLMENT_SESSIONS
    )

    @model_validator(mode="after")
    def _validate_user_ref_shape(self) -> "EnrollmentRequest":
        if self.user_ref is not None:
            validate_user_ref(self.user_ref)
        return self


class VerificationRequest(BaseModel):
    """Verification payload: one user, exactly one raw probe session.

    As with enrollment, ``user_ref`` is optional and only ever checked against
    the authenticated identity; it cannot select another user's profile.
    """

    user_ref: Optional[str] = Field(default=None, max_length=MAX_USER_REF_LENGTH)
    session: RawSession

    @model_validator(mode="after")
    def _validate_user_ref_shape(self) -> "VerificationRequest":
        if self.user_ref is not None:
            validate_user_ref(self.user_ref)
        return self


class EnrollmentResponse(BaseModel):
    """Outcome of a successful enrollment (no raw data / embeddings)."""

    user_ref: str
    status: Literal["enrolled"] = "enrolled"
    embedding_dimension: int
    session_count: int


class VerificationResponse(BaseModel):
    """Outcome of a verification attempt (no raw data / embeddings)."""

    user_ref: str
    decision: Literal["VERIFIED", "SUSPICIOUS"]
    distance: float
    threshold: float


class ContinuousVerificationRequest(RawSession):
    """A single bounded behavioral window (Phase 14A) for continuous
    verification.

    Contains behavioral data ONLY — the exact Phase 2 session event shape.
    Identity fields are intentionally absent and extra fields are rejected
    (``user_ref``/``username`` in the body produce a 422), so a client can
    never influence which behavioral profile is selected: the authenticated
    JWT identity is the only source of authorization.
    """

    model_config = ConfigDict(extra="forbid")


class ContinuousVerificationResponse(BaseModel):
    """Outcome of one continuous verification window.

    Minimal and privacy-safe: no identity, centroid, embedding, raw events,
    credentials or internal details are ever included. The three Phase 14A
    fields are unchanged; the Phase 14B additions are the server-authoritative
    session state after this window's decision.
    """

    decision: Literal["VERIFIED", "SUSPICIOUS"]
    distance: float
    threshold: float
    session_state: Literal["verified", "reverification_required"]
    consecutive_suspicious: int
    last_verified_at: Optional[str] = None


class SessionStateResponse(BaseModel):
    """Server-authoritative behavioral state for the authenticated session.

    Read via ``GET /api/v1/continuous-verification/state`` so the frontend can
    restore its UI after a refresh. Aggregate flags/timestamps only — never raw
    data, embeddings or identity fields.
    """

    session_state: Literal["verified", "reverification_required"]
    consecutive_suspicious: int
    last_verified_at: Optional[str] = None


__all__ = [
    "ContinuousVerificationRequest",
    "ContinuousVerificationResponse",
    "MAX_ENROLLMENT_SESSIONS",
    "MAX_SESSION_EVENTS",
    "EnrollmentRequest",
    "EnrollmentResponse",
    "KeyboardEvent",
    "MouseEvent",
    "RawSession",
    "SessionStateResponse",
    "VerificationRequest",
    "VerificationResponse",
    "validate_user_ref",
]