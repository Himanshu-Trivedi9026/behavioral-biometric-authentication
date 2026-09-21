"""
Phase 14B — in-memory per-session behavioral verification state store.

Mirrors :class:`~backend.app.repositories.session_state_repository.SessionStateRepository`
in process memory for the no-database fallback (and for the Phase 9B-style test
harnesses). State is keyed by ``(user_ref, session_id)`` so a fresh login
(a brand-new session_id) automatically starts clean.

Thread safety & race-safety
---------------------------
All reads/writes are guarded by a single ``threading.RLock``. ``get_or_create``
is atomic: two concurrent first-touches for the same key always converge on one
row (create-if-absent under the lock).

Storage & privacy
-----------------
Only aggregate flags and timestamps are kept. Raw keyboard/mouse events,
embeddings, centroids, characters, passwords, JWT tokens and secrets are never
stored here. State never survives a process restart (by design; PostgreSQL is
the persistent Phase 14B path).
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

from backend.app.repositories.session_state_repository import (
    SESSION_STATE_VERIFIED,
    SessionStateRepository,
    SessionVerificationState,
    default_session_state,
)
from backend.app.services.profile_store import utc_now


class InMemorySessionStateStore(SessionStateRepository):
    """Thread-safe, non-persistent per-session verification state store."""

    def __init__(self) -> None:
        self._states: Dict[Tuple[str, str], SessionVerificationState] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(user_ref: str, session_id: str) -> Tuple[str, str]:
        return (user_ref, session_id)

    def get_or_create(
        self, user_ref: str, session_id: str, *, now: Optional[str] = None
    ) -> SessionVerificationState:
        now = now or utc_now()
        key = self._key(user_ref, session_id)
        with self._lock:
            existing = self._states.get(key)
            if existing is not None:
                return existing
            state = default_session_state(user_ref, session_id, now=now)
            self._states[key] = state
            return state

    def get(
        self, user_ref: str, session_id: str
    ) -> Optional[SessionVerificationState]:
        with self._lock:
            return self._states.get(self._key(user_ref, session_id))

    def mark_verified(
        self, user_ref: str, session_id: str, *, now: Optional[str] = None
    ) -> SessionVerificationState:
        now = now or utc_now()
        key = self._key(user_ref, session_id)
        with self._lock:
            existing = self._states.get(key)
            state = SessionVerificationState(
                user_ref=user_ref,
                session_id=session_id,
                state=SESSION_STATE_VERIFIED,
                consecutive_suspicious=0,
                last_verified_at=now,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            self._states[key] = state
            return state

    def mark_suspicious(
        self, user_ref: str, session_id: str, *, now: Optional[str] = None
    ) -> SessionVerificationState:
        now = now or utc_now()
        key = self._key(user_ref, session_id)
        with self._lock:
            existing = self._states.get(key)
            state = SessionVerificationState(
                user_ref=user_ref,
                session_id=session_id,
                state="reverification_required",
                consecutive_suspicious=(existing.consecutive_suspicious + 1) if existing else 1,
                last_verified_at=existing.last_verified_at if existing else None,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            self._states[key] = state
            return state

    def count(self) -> int:
        """Return the number of tracked sessions (test/ops helper)."""
        with self._lock:
            return len(self._states)

    def clear(self) -> int:
        """Remove all sessions; returns the number removed (test helper)."""
        with self._lock:
            n = len(self._states)
            self._states.clear()
            return n


__all__ = ["InMemorySessionStateStore"]