"""
Phase 10 — PostgreSQL-backed enrollment profile repository.

Implements the :class:`~backend.app.repositories.profile_repository.ProfileRepository`
contract on PostgreSQL. It reuses the Phase 9B :class:`StoredProfile` value
object unchanged, so the ML service and routes behave exactly as before — only
WHERE the profile lives changes (PostgreSQL instead of process-local memory).

Storage & privacy
-----------------
The repository persists ONLY enrollment aggregates:

    user_ref, centroid (DOUBLE PRECISION[]), embedding_dim, session_count,
    created_at, updated_at

Raw keyboard/mouse events, key identity, characters, passwords, tokens and
secrets are never written to the database. Ordinary L2 distance happens inside
the existing ML service; PostgreSQL only supplies the centroid.

Connection strategy
-------------------
A small psycopg3 ``ConnectionPool`` (``psycopg_pool``) is created lazily on the
first database operation — never at import time and never during
``create_app()``. Connections are borrowed with ``with pool.connection():`` and
always returned, so no connection leaks. The pool uses ``autocommit=True``;
writes are wrapped in an explicit ``conn.transaction()`` block.

Failure behaviour
-----------------
* Missing database URL               -> ``DatabaseUnavailableError`` (503).
* Connect failure / pool timeout     -> ``DatabaseUnavailableError`` (503).
* Unique-violation on insert         -> ``ProfileExistsError`` (409).
* Any other database error           -> ``DatabaseOperationError`` (500).

No SQL, connection strings, hostnames, passwords or behavioural data are ever
exposed in these errors (the API handler only uses ``code`` + ``message``).

Lazy driver import
------------------
``psycopg`` / ``psycopg_pool`` are imported inside method bodies so that
importing the application, the routes, or serving ``/health`` never touches
the database stack.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.app.repositories.errors import (
    DatabaseOperationError,
    DatabaseUnavailableError,
)
from backend.app.repositories.profile_repository import ProfileRepository
from backend.app.services.profile_store import ProfileExistsError, StoredProfile

# ---------------------------------------------------------------------------
# SQL (module-level constants; deterministic, non-destructive, inspectable)
# ---------------------------------------------------------------------------

PROFILE_COLUMNS: Tuple[str, ...] = (
    "user_ref",
    "centroid",
    "embedding_dim",
    "session_count",
    "created_at",
    "updated_at",
)

INSERT_PROFILE_SQL = """
INSERT INTO behavioral_profiles
    (user_ref, centroid, embedding_dim, session_count, created_at, updated_at)
VALUES (%s, %s, %s, %s, %s, %s)
"""

SELECT_PROFILE_SQL = """
SELECT user_ref, centroid, embedding_dim, session_count, created_at, updated_at
FROM behavioral_profiles
WHERE user_ref = %s
"""

SELECT_EXISTS_SQL = "SELECT 1 FROM behavioral_profiles WHERE user_ref = %s"

SELECT_COUNT_SQL = "SELECT count(*) FROM behavioral_profiles"

SELECT_METADATA_SQL = """
SELECT user_ref, embedding_dim, session_count, created_at, updated_at
FROM behavioral_profiles
ORDER BY user_ref
"""


# ---------------------------------------------------------------------------
# Pure mapping helpers (fully unit-testable without a database)
# ---------------------------------------------------------------------------


def profile_to_values(profile: StoredProfile) -> Tuple[Any, ...]:
    """Convert a :class:`StoredProfile` into INSERT values (no raw data)."""
    if not isinstance(profile, StoredProfile):
        raise TypeError("profile must be a StoredProfile")
    return (
        profile.user_ref,
        list(profile.centroid),
        profile.embedding_dim,
        profile.session_count,
        profile.created_at,
        profile.updated_at,
    )


def _to_iso(value: Any) -> str:
    """Render a driver timestamp (or ISO string) as ISO-8601 text."""
    if value is None:
        raise DatabaseOperationError("profile timestamps are missing")
    iso = getattr(value, "isoformat", None)
    return iso() if iso else str(value)


def row_to_profile(row: Tuple[Any, ...]) -> StoredProfile:
    """Convert a SELECT row (in :data:`PROFILE_COLUMNS` order) back to a profile."""
    user_ref, centroid, embedding_dim, session_count, created_at, updated_at = row

    return StoredProfile(
        user_ref=str(user_ref),
        centroid=tuple(float(v) for v in centroid),
        embedding_dim=int(embedding_dim),
        session_count=int(session_count),
        created_at=_to_iso(created_at),
        updated_at=_to_iso(updated_at),
    )


def row_to_metadata(row: Tuple[Any, ...]) -> Dict[str, Any]:
    """Convert a metadata row (no centroid) for :meth:`list_profiles`."""
    user_ref, embedding_dim, session_count, created_at, updated_at = row
    return {
        "user_ref": str(user_ref),
        "embedding_dim": int(embedding_dim),
        "session_count": int(session_count),
        "created_at": _to_iso(created_at),
        "updated_at": _to_iso(updated_at),
    }


def is_unique_violation(exc: BaseException) -> bool:
    """True when the exception is a PostgreSQL unique-constraint violation."""
    from psycopg.errors import UniqueViolation  # lazy

    return isinstance(exc, UniqueViolation)


def raise_for_database_error(exc: BaseException) -> None:
    """Translate a storage/driver exception into a structured domain error."""
    import psycopg  # lazy

    if isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError)):
        raise DatabaseUnavailableError() from None
    if isinstance(exc, psycopg.DatabaseError):
        raise DatabaseOperationError() from None
    raise DatabaseOperationError() from None


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class PostgresProfileRepository(ProfileRepository):
    """PostgreSQL-backed :class:`ProfileRepository` (lazy pool, race-safe save)."""

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
    def from_settings(cls, settings: Any) -> "PostgresProfileRepository":
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
        self, operation: Callable[[Any], Any], *, unique_user: Optional[str] = None
    ) -> Any:
        pool = self._ensure_pool()
        try:
            with pool.connection() as conn:
                return operation(conn)
        except Exception as exc:  # noqa: BLE001 - translated to domain errors
            if is_unique_violation(exc):
                if unique_user is not None:
                    raise ProfileExistsError(unique_user) from None
                raise DatabaseOperationError() from None
            raise_for_database_error(exc)  # noqa: B904 - already inside except
        raise AssertionError("raise_for_database_error must always raise")  # pragma: no cover

    # -- repository contract -------------------------------------------------

    def save(self, profile: StoredProfile) -> StoredProfile:
        """Insert ``profile``; the DB unique constraint guards duplicates."""
        values = profile_to_values(profile)
        self._run(
            lambda conn: _insert_values(conn, values),
            unique_user=profile.user_ref,
        )
        return profile

    def get(self, user_ref: str) -> Optional[StoredProfile]:
        row = self._run(
            lambda conn: conn.execute(SELECT_PROFILE_SQL, (user_ref,)).fetchone()
        )
        return row_to_profile(row) if row is not None else None

    def contains(self, user_ref: str) -> bool:
        row = self._run(
            lambda conn: conn.execute(SELECT_EXISTS_SQL, (user_ref,)).fetchone()
        )
        return row is not None

    def count(self) -> int:
        row = self._run(lambda conn: conn.execute(SELECT_COUNT_SQL).fetchone())
        return int(row[0])

    def list_profiles(self) -> List[Dict[str, Any]]:
        rows = self._run(lambda conn: conn.execute(SELECT_METADATA_SQL).fetchall())
        return [row_to_metadata(row) for row in rows]


def _insert_values(conn: Any, values: Tuple[Any, ...]) -> None:
    with conn.transaction():
        conn.execute(INSERT_PROFILE_SQL, values)


__all__ = [
    "PostgresProfileRepository",
    "is_unique_violation",
    "profile_to_values",
    "raise_for_database_error",
    "row_to_metadata",
    "row_to_profile",
]