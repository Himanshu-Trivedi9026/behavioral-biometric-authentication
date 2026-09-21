"""
Phase 10 — profile repository abstraction.

The route handlers depend on :class:`ProfileRepository` — an abstract contract,
not on PostgreSQL SQL. Both the Phase 9B in-memory store and the Phase 10
PostgreSQL repository implement it, so FastAPI tests can swap in a fake and the
ML service / API behavior stays identical.

Contract methods
----------------
* :meth:`ProfileRepository.save` — insert a profile; raise
  :class:`ProfileExistsError` when the ``user_ref`` already exists. The
  implementation must make this race-safe (e.g. a database unique constraint),
  so callers should NOT rely on a pre-check.
* :meth:`ProfileRepository.get` — ``None`` when the user is unknown.
* :meth:`ProfileRepository.contains` — cheap existence check.
* :meth:`ProfileRepository.count` / :meth:`list_profiles` — introspection
  helpers (metadata only, never the centroid).

Only enrollment aggregates flow through this contract:

    user_ref, centroid, embedding_dim, session_count, created_at, updated_at

Raw keyboard/mouse events are never stored or returned.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from backend.app.services.profile_store import (
    ProfileExistsError,
    ProfileNotFoundError,
    StoredProfile,
)


class ProfileRepository(ABC):
    """Abstract persistent store for enrollment profiles."""

    @abstractmethod
    def save(self, profile: StoredProfile) -> StoredProfile:
        """Insert ``profile`` atomically; raise on duplicates."""

    @abstractmethod
    def get(self, user_ref: str) -> Optional[StoredProfile]:
        """Return the stored profile for ``user_ref`` or ``None``."""

    @abstractmethod
    def contains(self, user_ref: str) -> bool:
        """Return whether a profile exists for ``user_ref``."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored profiles."""

    @abstractmethod
    def list_profiles(self) -> List[Dict[str, Any]]:
        """Return non-sensitive metadata per profile (never the centroid)."""


__all__ = [
    "ProfileExistsError",
    "ProfileNotFoundError",
    "ProfileRepository",
    "StoredProfile",
]