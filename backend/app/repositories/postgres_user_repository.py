"""
Phase 11 — PostgreSQL-backed user repository.

Implements the :class:`~backend.app.repositories.user_repository.UserRepository`
contract on PostgreSQL. It reuses the :class:`UserRecord` value object
unchanged, so the auth routes behave the same whether the user lives in
PostgreSQL or the in-memory fallback — only WHERE the record lives changes.

Storage & privacy
-----------------
The repository persists ONLY identity records:

    id, username, password_hash, created_at, updated_at

``password_hash`` is an Argon2id hash (never plaintext). Raw behavioural
events, embeddings, JWT secrets and tokens are never written here.

Connection strategy
-------------------
Same as the Phase 10 profile repository: a small psycopg3 ``ConnectionPool``
is created lazily on the first database operation — never at import time and
never during ``create_app()`` or dependency resolution. Connections are
borrowed with ``with pool.connection():`` and always returned. The pool runs
``autocommit=True``; INSERTs are wrapped in an explicit transaction block.

Failure behaviour
-----------------
* Missing database URL                 -> ``DatabaseUnavailableError`` (503).
* Connect failure / pool timeout       -> ``DatabaseUnavailableError`` (503).
* Unique-violation on insert           -> ``UsernameExistsError`` (409).
* Any other database error             -> ``DatabaseOperationError`` (500).

No SQL, connection strings, hostnames, credentials, hashes or behavioural
data are ever exposed in these errors.

Lazy driver import
------------------
``psycopg`` / ``psycopg_pool`` are imported inside method bodies so that
importing the application, the routes, or serving ``/health`` never touches
the database stack.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional, Tuple

from backend.app.repositories.errors import (
    DatabaseOperationError,
    DatabaseUnavailableError,
)
from backend.app.repositories.postgres_profile_repository import (
    is_unique_violation,
    raise_for_database_error,
)
from backend.app.repositories.user_repository import (
    UserRecord,
    UserRepository,
    UsernameExistsError,
)

# ---------------------------------------------------------------------------
# SQL (module-level constants; deterministic, non-destructive, inspectable)
# ---------------------------------------------------------------------------

USER_COLUMNS: Tuple[str, ...] = (
    "id",
    "username",
    "password_hash",
    "created_at",
    "updated_at",
)

INSERT_USER_SQL = """
INSERT INTO users
    (id, username, password_hash, created_at, updated_at)
VALUES (%s, %s, %s, %s, %s)
"""

SELECT_USER_BY_USERNAME_SQL = """
SELECT id, username, password_hash, created_at, updated_at
FROM users
WHERE username = %s
"""

SELECT_USER_BY_ID_SQL = """
SELECT id, username, password_hash, created_at, updated_at
FROM users
WHERE id = %s
"""

SELECT_USER_EXISTS_SQL = "SELECT 1 FROM users WHERE username = %s"


# ---------------------------------------------------------------------------
# Pure mapping helpers (fully unit-testable without a database)
# ---------------------------------------------------------------------------


def user_to_values(user: UserRecord) -> Tuple[Any, ...]:
    """Convert a :class:`UserRecord` into INSERT values (no behavioural data)."""
    if not isinstance(user, UserRecord):
        raise TypeError("user must be a UserRecord")
    return (
        user.id,
        user.username,
        user.password_hash,
        user.created_at,
        user.updated_at,
    )


def _to_iso(value: Any) -> str:
    """Render a driver timestamp (or ISO string) as ISO-8601 text."""
    if value is None:
        raise DatabaseOperationError("user timestamps are missing")
    iso = getattr(value, "isoformat", None)
    return iso() if iso else str(value)


def row_to_user(row: Tuple[Any, ...]) -> UserRecord:
    """Convert a SELECT row (in :data:`USER_COLUMNS` order) back to a user."""
    user_id, username, password_hash, created_at, updated_at = row
    return UserRecord(
        id=str(user_id),
        username=str(username),
        password_hash=str(password_hash),
        created_at=_to_iso(created_at),
        updated_at=_to_iso(updated_at),
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class PostgresUserRepository(UserRepository):
    """PostgreSQL-backed :class:`UserRepository` (lazy pool, race-safe create)."""

    def __init__(
        self,
        database_url: str = "",
        *,
        max_pool_size: int = 4,
        connect_timeout: float = 5.0,
        pool_timeout: float = 5.0,
    ) -> None:
        self._conninfo = database_url
        self._max_pool_size = max(1, int(max_pool_size))
        self._connect_timeout = float(connect_timeout)
        self._pool_timeout = float(pool_timeout)
        self._pool: Any = None
        self._lock = threading.RLock()

    @classmethod
    def from_settings(cls, settings: Any) -> "PostgresUserRepository":
        return cls(getattr(settings, "database_url", "") or "")

    # -- pool lifecycle ------------------------------------------------------

    def _ensure_pool(self) -> Any:
        if not self._conninfo:
            raise DatabaseUnavailableError()
        if self._pool is not None:
            return self._pool
        with self._lock:
            if self._pool is not None:
                return self._pool
            from psycopg_pool import ConnectionPool  # lazy

            self._pool = ConnectionPool(
                self._conninfo,
                min_size=0,
                max_size=self._max_pool_size,
                open=False,
                timeout=self._pool_timeout,
                kwargs={
                    "connect_timeout": self._connect_timeout,
                    "autocommit": True,
                },
            )
            self._pool.open()
        return self._pool

    def close(self) -> None:
        """Close the pool (if created). Safe to call multiple times."""
        with self._lock:
            if self._pool is not None:
                try:
                    self._pool.close()
                except Exception:  # noqa: BLE001 - nothing to leak to callers
                    pass
                self._pool = None

    @property
    def pool(self) -> Any:
        """The underlying connection pool (created lazily); tests inspect it."""
        return self._pool

    # -- shared execution ----------------------------------------------------

    def _run(
        self,
        operation: Callable[[Any], Any],
        *,
        unique_username: Optional[str] = None,
    ) -> Any:
        pool = self._ensure_pool()
        try:
            with pool.connection() as conn:
                return operation(conn)
        except Exception as exc:  # noqa: BLE001 - translated to domain errors
            if is_unique_violation(exc):
                if unique_username is not None:
                    raise UsernameExistsError(unique_username) from None
                raise DatabaseOperationError() from None
            raise_for_database_error(exc)  # noqa: B904 - already inside except
        raise AssertionError("raise_for_database_error must always raise")  # pragma: no cover

    # -- repository contract -------------------------------------------------

    def create(self, user: UserRecord) -> UserRecord:
        """Insert ``user``; the DB unique constraint guards duplicates.

        The repository does NOT pre-check the username — the INSERT raises a
        unique-violation on a duplicate, which is translated into the
        structured 409 :class:`UsernameExistsError`.
        """
        values = user_to_values(user)
        self._run(
            lambda conn: _insert_values(conn, values),
            unique_username=user.username,
        )
        return user

    def get_by_username(self, username: str) -> Optional[UserRecord]:
        row = self._run(
            lambda conn: conn.execute(SELECT_USER_BY_USERNAME_SQL, (username,)).fetchone()
        )
        return row_to_user(row) if row is not None else None

    def get_by_id(self, user_id: str) -> Optional[UserRecord]:
        row = self._run(
            lambda conn: conn.execute(SELECT_USER_BY_ID_SQL, (user_id,)).fetchone()
        )
        return row_to_user(row) if row is not None else None

    def contains_username(self, username: str) -> bool:
        row = self._run(
            lambda conn: conn.execute(SELECT_USER_EXISTS_SQL, (username,)).fetchone()
        )
        return row is not None


def _insert_values(conn: Any, values: Tuple[Any, ...]) -> None:
    with conn.transaction():
        conn.execute(INSERT_USER_SQL, values)


__all__ = [
    "INSERT_USER_SQL",
    "PostgresUserRepository",
    "SELECT_USER_BY_ID_SQL",
    "SELECT_USER_BY_USERNAME_SQL",
    "SELECT_USER_EXISTS_SQL",
    "USER_COLUMNS",
    "row_to_user",
    "user_to_values",
]