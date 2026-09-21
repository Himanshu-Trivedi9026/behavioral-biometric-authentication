"""
Phase 9B — in-memory enrollment profile store.

Stores one enrollment summary per ``user_ref`` for the lifetime of the process:

* Only aggregate behavioral summary data is kept: the **centroid embedding**
  (an immutable tuple of floats) plus small metadata (``embedding_dim``,
  ``session_count``, timestamps).
* Raw keyboard/mouse events, key identity, characters, passwords/credentials
  and per-session traces are **never** stored here.
* The store is non-persistent by design: everything is lost when the process
  restarts. Persistence (e.g. a database) is a later phase.

Thread safety
-------------
All mutating/reading operations are guarded by a single ``threading.RLock`` so
concurrent FastAPI requests cannot corrupt state or race on duplicate
enrollment (``save`` atomically rejects a profile that already exists).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.app.errors import AppError


class ProfileExistsError(AppError):
    """A profile for this ``user_ref`` already exists (do not silently overwrite)."""

    def __init__(self, user_ref: str) -> None:
        super().__init__(
            code="profile_exists",
            message="a profile already exists for user '{}'".format(user_ref),
            status_code=409,
        )


class ProfileNotFoundError(AppError):
    """No enrollment profile exists for this ``user_ref``."""

    def __init__(self, user_ref: str) -> None:
        super().__init__(
            code="profile_not_found",
            message="no enrollment profile found for user '{}'".format(user_ref),
            status_code=404,
        )


@dataclass(frozen=True)
class StoredProfile:
    """Immutable enrollment summary kept in memory for one ``user_ref``."""

    user_ref: str
    centroid: Tuple[float, ...]
    embedding_dim: int
    session_count: int
    created_at: str
    updated_at: str


def utc_now() -> str:
    """ISO-8601 UTC timestamp (metadata, not a privacy risk)."""
    return datetime.now(timezone.utc).isoformat()


def stored_fields() -> Tuple[str, ...]:
    """Names of the dataclass fields (used by privacy tests)."""
    return tuple(f.name for f in fields(StoredProfile))


class InMemoryProfileStore:
    """Thread-safe, non-persistent enrollment profile store."""

    def __init__(self) -> None:
        self._profiles: Dict[str, StoredProfile] = {}
        self._lock = threading.RLock()

    def save(self, profile: StoredProfile) -> StoredProfile:
        """Insert ``profile``; raise :class:`ProfileExistsError` on duplicates."""
        if not isinstance(profile, StoredProfile):
            raise TypeError("profile must be a StoredProfile")
        with self._lock:
            if profile.user_ref in self._profiles:
                raise ProfileExistsError(profile.user_ref)
            self._profiles[profile.user_ref] = profile
            return profile

    def get(self, user_ref: str) -> Optional[StoredProfile]:
        """Return the stored profile, or ``None`` when the user is unknown."""
        with self._lock:
            return self._profiles.get(user_ref)

    def contains(self, user_ref: str) -> bool:
        with self._lock:
            return user_ref in self._profiles

    def user_refs(self) -> List[str]:
        """Sorted list of stored user references (metadata only)."""
        with self._lock:
            return sorted(self._profiles)

    def count(self) -> int:
        with self._lock:
            return len(self._profiles)

    def list_profiles(self) -> List[Dict[str, Any]]:
        """Non-sensitive metadata per profile (NEVER includes the centroid)."""
        with self._lock:
            return [
                {
                    "user_ref": p.user_ref,
                    "embedding_dim": p.embedding_dim,
                    "session_count": p.session_count,
                    "created_at": p.created_at,
                    "updated_at": p.updated_at,
                }
                for p in sorted(self._profiles.values(), key=lambda p: p.user_ref)
            ]

    def clear(self) -> int:
        """Remove all profiles; returns the number removed (test/ops helper)."""
        with self._lock:
            n = len(self._profiles)
            self._profiles.clear()
            return n


__all__ = [
    "InMemoryProfileStore",
    "ProfileExistsError",
    "ProfileNotFoundError",
    "StoredProfile",
    "stored_fields",
    "utc_now",
]