"""
Phase 14B — per-session behavioral verification state (domain + contract).

A login session (one bearer JWT) carries a small, server-authoritative
behavioral state that lives for the duration of that session:

    SessionVerificationState
        state                'verified' | 'reverification_required'
        consecutive_suspicious  count of consecutive SUSPICIOUS continuous
                                windows, bounded to >= 0
        last_verified_at        when the session last produced a VERIFIED
                                continuous window (None until it has)

Only AGGREGATE, non-sensitive flags/timestamps are stored: never raw events,
embeddings, centroids, characters, passwords, tokens or secrets.

Contract semantics (implementations MUST agree):
    get_or_create  — return the row, creating a clean one (verified, 0,
                     last_verified_at=None) on first touch. Creating is
                     race-safe: concurrent first-touches must converge on a
                     single row.
    mark_verified  — a VERIFIED continuous window: state -> 'verified',
                     consecutive_suspicious -> 0, last_verified_at -> now.
                     The row is created if absent.
    mark_suspicious— a SUSPICIOUS continuous window: state ->
                     'reverification_required', consecutive_suspicious +1
                     (1 for a fresh row), last_verified_at preserved.

A fresh login derives a NEW session_id (from the JWT ``iat`` claim), so it
inserts a clean row rather than inheriting the previous login's state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from backend.app.services.profile_store import utc_now

SESSION_STATE_VERIFIED = "verified"
SESSION_STATE_REVERIFICATION_REQUIRED = "reverification_required"
VALID_SESSION_STATES = frozenset(
    {SESSION_STATE_VERIFIED, SESSION_STATE_REVERIFICATION_REQUIRED}
)


def default_session_state(
    user_ref: str, session_id: str, *, now: str
) -> "SessionVerificationState":
    """A clean, unblocked state for a brand-new session (never gated)."""
    return SessionVerificationState(
        user_ref=user_ref,
        session_id=session_id,
        state=SESSION_STATE_VERIFIED,
        consecutive_suspicious=0,
        last_verified_at=None,
        created_at=now,
        updated_at=now,
    )


@dataclass(frozen=True)
class SessionVerificationState:
    """Immutable aggregate behavioral state for one login session."""

    user_ref: str
    session_id: str
    state: str
    consecutive_suspicious: int
    last_verified_at: Optional[str]
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        if self.state not in VALID_SESSION_STATES:
            raise ValueError(
                "state must be one of {}".format(sorted(VALID_SESSION_STATES))
            )
        if self.consecutive_suspicious < 0:
            raise ValueError("consecutive_suspicious must be >= 0")


class SessionStateRepository(ABC):
    """Storage contract for per-session behavioral verification state."""

    @abstractmethod
    def get_or_create(
        self, user_ref: str, session_id: str, *, now: Optional[str] = None
    ) -> SessionVerificationState:
        """Return the session state, creating a clean one on first touch."""

    @abstractmethod
    def get(
        self, user_ref: str, session_id: str
    ) -> Optional[SessionVerificationState]:
        """Return the session state, or ``None`` when never touched."""

    @abstractmethod
    def mark_verified(
        self, user_ref: str, session_id: str, *, now: Optional[str] = None
    ) -> SessionVerificationState:
        """Apply a VERIFIED window decision and return the new state."""

    @abstractmethod
    def mark_suspicious(
        self, user_ref: str, session_id: str, *, now: Optional[str] = None
    ) -> SessionVerificationState:
        """Apply a SUSPICIOUS window decision and return the new state."""


__all__ = [
    "SESSION_STATE_REVERIFICATION_REQUIRED",
    "SESSION_STATE_VERIFIED",
    "VALID_SESSION_STATES",
    "SessionStateRepository",
    "SessionVerificationState",
    "default_session_state",
]