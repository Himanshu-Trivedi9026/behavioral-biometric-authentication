"""
Phase 11 — in-memory user repository.

Thread-safe, non-persistent fallback used when no ``database_url`` is
configured. It mirrors the real :class:`PostgresUserRepository` contract
closely enough for development and tests:

* ``create`` is atomic and rejects a duplicate ``username`` with 409
  :class:`UsernameExistsError` (the stand-in for the DB unique constraint).
* lookups return ``None`` for unknown users.
* Only identity records (id, username, password_hash, timestamps) are stored;
  behavioural data and secrets never touch this store.

Like everything in-memory, data is lost on restart. Registration therefore
only "persists" for the lifetime of the process — exactly like the Phase 9B
:class:`InMemoryProfileStore`.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from backend.app.repositories.user_repository import (
    UserRecord,
    UserRepository,
    UsernameExistsError,
)


class InMemoryUserRepository(UserRepository):
    """Thread-safe, non-persistent :class:`UserRepository`."""

    def __init__(self) -> None:
        self._by_username: Dict[str, UserRecord] = {}
        self._by_id: Dict[str, UserRecord] = {}
        self._lock = threading.RLock()

    def create(self, user: UserRecord) -> UserRecord:
        if not isinstance(user, UserRecord):
            raise TypeError("user must be a UserRecord")
        with self._lock:
            if user.username in self._by_username:
                raise UsernameExistsError(user.username)
            if user.id in self._by_id:
                raise ValueError("duplicate user id {!r}".format(user.id))
            self._by_username[user.username] = user
            self._by_id[user.id] = user
            return user

    def get_by_username(self, username: str) -> Optional[UserRecord]:
        with self._lock:
            return self._by_username.get(username)

    def get_by_id(self, user_id: str) -> Optional[UserRecord]:
        with self._lock:
            return self._by_id.get(user_id)

    def contains_username(self, username: str) -> bool:
        with self._lock:
            return username in self._by_username

    def count(self) -> int:
        """Number of stored users (test/ops introspection)."""
        with self._lock:
            return len(self._by_username)

    def clear(self) -> int:
        """Remove all users; returns how many were removed (test/ops helper)."""
        with self._lock:
            n = len(self._by_username)
            self._by_username.clear()
            self._by_id.clear()
            return n


__all__ = ["InMemoryUserRepository"]