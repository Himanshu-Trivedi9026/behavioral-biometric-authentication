"""
Phase 11 — user repository abstraction.

Route handlers depend on :class:`UserRepository` — an abstract contract, not on
PostgreSQL SQL. FastAPI tests swap in the in-memory implementation or a fake,
while the persistent :class:`~backend.app.repositories.postgres_user_repository.PostgresUserRepository`
implements the same contract on PostgreSQL.

Contract methods
----------------
* :meth:`UserRepository.create` — insert ``user``; raise
  :class:`UsernameExistsError` when the ``username`` already exists. The
  implementation must make this race-safe (a database unique constraint), so
  callers MUST NOT rely on a pre-check.
* :meth:`UserRepository.get_by_username` / :meth:`get_by_id` — ``None`` when
  the user is unknown.
* :meth:`UserRepository.contains_username` — cheap existence check.

Only identity records flow through this contract:

    id, username, password_hash, created_at, updated_at

``password_hash`` is an Argon2id hash and is treated as a secret: it is only
ever read by the login flow and is never returned by the API. Raw behavioral
events, embeddings and JWT secrets are never stored here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from backend.app.errors import AppError


@dataclass(frozen=True)
class UserRecord:
    """Immutable persisted user record (identity + Argon2id hash).

    ``password_hash`` is internal-only. The API layer maps this into a safe,
    hash-free response schema before anything leaves the process.
    """

    id: str
    username: str
    password_hash: str
    created_at: str
    updated_at: str


class UsernameExistsError(AppError):
    """A user with this username already exists (database-faithful 409)."""

    def __init__(self, username: str) -> None:
        super().__init__(
            code="username_exists",
            message="a user with username '{}' already exists".format(username),
            status_code=409,
        )


class UserRepository(ABC):
    """Abstract persistent store for application users."""

    @abstractmethod
    def create(self, user: UserRecord) -> UserRecord:
        """Insert ``user`` atomically; raise :class:`UsernameExistsError` on duplicates."""

    @abstractmethod
    def get_by_username(self, username: str) -> Optional[UserRecord]:
        """Return the user for ``username`` or ``None``."""

    @abstractmethod
    def get_by_id(self, user_id: str) -> Optional[UserRecord]:
        """Return the user for ``id`` (UUID string) or ``None``."""

    @abstractmethod
    def contains_username(self, username: str) -> bool:
        """Return whether a user exists for ``username``."""


__all__ = ["UserRecord", "UserRepository", "UsernameExistsError"]