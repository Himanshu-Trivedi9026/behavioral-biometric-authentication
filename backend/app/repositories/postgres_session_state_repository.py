"""
Phase 14B — PostgreSQL-backed per-session behavioral verification state.

Implements the
:class:`~backend.app.repositories.session_state_repository.SessionStateRepository`
contract on PostgreSQL (migration ``003_create_session_states.sql``). Only
AGGREGATE state is stored:

    user_ref, session_id, state, consecutive_suspicious, last_verified_at,
    created_at, updated_at

Raw keyboard/mouse events, behavioural embeddings, centroids, characters,
passwords, JWT tokens and secrets are never written here.

Concurrency
-----------
All writes use atomic ``INSERT ... ON CONFLICT`` upserts:

* ``get_or_create``     — ``ON CONFLICT DO NOTHING`` then ``SELECT``, so a
  concurrent first-touch converges on a single row (the PK is
  (user_ref, session_id) and the pool is ``autocommit=True``).
* ``mark_verified/…``   — ``ON CONFLICT DO UPDATE`` with ``RETURNING``, so a
  state transition is a single atomic statement (no lost updates).

Failure behaviour
-----------------
Mirrors the Phase 10 PostgreSQL repository: missing/unreachable database ->
``DatabaseUnavailableError`` (503), any other database error ->
``DatabaseOperationError`` (500). No SQL, connection strings, hostnames,
passwords or behavioural data ever surface in errors. The driver is imported
lazily so building the app never touches the DB stack.
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
from backend.app.repositories.session_state_repository import (
    SESSION_STATE_VERIFIED,
    SessionStateRepository,
    SessionVerificationState,
    default_session_state,
)
from backend.app.services.profile_store import utc_now

# ---------------------------------------------------------------------------
# SQL (module-level constants; deterministic, non-destructive, inspectable)
# ---------------------------------------------------------------------------

SESSION_STATE_COLUMNS: Tuple[str, ...] = (
    "user_ref",
    "session_id",
    "state",
    "consecutive_suspicious",
    "last_verified_at",
    "created_at",
    "updated_at",
)

SESSION_STATE_SELECT_LIST = ", ".join(SESSION_STATE_COLUMNS)

SESSION_STATE_COLUMN_NAMES = "user_ref, session_id, state, consecutive_suspicious, last_verified_at, created_at, updated_at"

INSERT_SESSION_STATE_SQL = (
    "INSERT INTO behavioral_session_states ("
    + SESSION_STATE_COLUMN_NAMES
    + ") VALUES (%s, %s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (user_ref, session_id) DO NOTHING "
    "RETURNING " + SESSION_STATE_SELECT_LIST
)

SELECT_SESSION_STATE_SQL = (
    "SELECT " + SESSION_STATE_SELECT_LIST + " "
    "FROM behavioral_session_states "
    "WHERE user_ref = %s AND session_id = %s"
)

SELECT_SESSION_STATE_COUNT_SQL = "SELECT count(*) FROM behavioral_session_states"

UPSERT_VERIFIED_SQL = (
    "INSERT INTO behavioral_session_states ("
    + SESSION_STATE_COLUMN_NAMES
    + ") VALUES (%s, %s, 'verified', 0, %s, %s, %s) "
    "ON CONFLICT (user_ref, session_id) DO UPDATE SET "
    "state = 'verified', "
    "consecutive_suspicious = 0, "
    "last_verified_at = EXCLUDED.last_verified_at, "
    "updated_at = EXCLUDED.updated_at "
    "RETURNING " + SESSION_STATE_SELECT_LIST
)

UPSERT_SUSPICIOUS_SQL = (
    "INSERT INTO behavioral_session_states ("
    + SESSION_STATE_COLUMN_NAMES
    + ") VALUES (%s, %s, 'reverification_required', 1, %s, %s, %s) "
    "ON CONFLICT (user_ref, session_id) DO UPDATE SET "
    "state = 'reverification_required', "
    "consecutive_suspicious = behavioral_session_states.consecutive_suspicious + 1, "
    "updated_at = EXCLUDED.updated_at "
    "RETURNING " + SESSION_STATE_SELECT_LIST
)


# ---------------------------------------------------------------------------
# Pure mapping helpers (fully unit-testable without a database)
# ---------------------------------------------------------------------------


def _to_iso(value: Any) -> str:
    """Render a driver timestamp (or ISO string) as ISO-8601 text."""
    iso = getattr(value, "isoformat", None)
    return iso() if iso else str(value)


def row_to_session_state(row: Tuple[Any, ...]) -> SessionVerificationState:
    """Convert a SELECT row (in :data:`SESSION_STATE_COLUMNS` order) to a state."""
    (
        user_ref,
        session_id,
        state,
        consecutive_suspicious,
        last_verified_at,
        created_at,
        updated_at,
    ) = row
    last = _to_iso(last_verified_at) if last_verified_at is not None else None
    return SessionVerificationState(
        user_ref=str(user_ref),
        session_id=str(session_id),
        state=str(state),
        consecutive_suspicious=int(consecutive_suspicious),
        last_verified_at=last,
        created_at=_to_iso(created_at),
        updated_at=_to_iso(updated_at),
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class PostgresSessionStateRepository(SessionStateRepository):
    """PostgreSQL-backed :class:`SessionStateRepository` (lazy pool, atomic upserts)."""

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
    def from_settings(cls, settings: Any) -> "PostgresSessionStateRepository":
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

    def _run(self, operation: Callable[[Any], Any]) -> Any:
        pool = self._ensure_pool()
        try:
            with pool.connection() as conn:
                return operation(conn)
        except Exception as exc:  # noqa: BLE001 - translated to domain errors
            if is_unique_violation(exc):
                raise DatabaseOperationError() from None
            raise_for_database_error(exc)  # noqa: B904 - already inside except
        raise AssertionError("raise_for_database_error must always raise")  # pragma: no cover

    # -- repository contract -------------------------------------------------

    def get_or_create(
        self, user_ref: str, session_id: str, *, now: Optional[str] = None
    ) -> SessionVerificationState:
        now = now or utc_now()
        row = self._run(
            lambda conn: conn.execute(
                INSERT_SESSION_STATE_SQL,
                (user_ref, session_id, SESSION_STATE_VERIFIED, 0, None, now, now),
            ).fetchone()
        )
        if row is not None:
            return row_to_session_state(row)
        existing = self._run(
            lambda conn: conn.execute(
                SELECT_SESSION_STATE_SQL, (user_ref, session_id)
            ).fetchone()
        )
        if existing is not None:
            return row_to_session_state(existing)
        # Defensive fallback for the (never observed) narrow race between the
        # DO NOTHING and the SELECT: return a clean state for this session.
        return default_session_state(user_ref, session_id, now=now)

    def get(
        self, user_ref: str, session_id: str
    ) -> Optional[SessionVerificationState]:
        row = self._run(
            lambda conn: conn.execute(
                SELECT_SESSION_STATE_SQL, (user_ref, session_id)
            ).fetchone()
        )
        return row_to_session_state(row) if row is not None else None

    def mark_verified(
        self, user_ref: str, session_id: str, *, now: Optional[str] = None
    ) -> SessionVerificationState:
        now = now or utc_now()
        row = self._run(
            lambda conn: conn.execute(
                UPSERT_VERIFIED_SQL, (user_ref, session_id, now, now, now)
            ).fetchone()
        )
        if row is None:  # pragma: no cover - ON CONFLICT always returns a row
            raise DatabaseOperationError("session state insert returned no row")
        return row_to_session_state(row)

    def mark_suspicious(
        self, user_ref: str, session_id: str, *, now: Optional[str] = None
    ) -> SessionVerificationState:
        now = now or utc_now()
        row = self._run(
            lambda conn: conn.execute(
                UPSERT_SUSPICIOUS_SQL, (user_ref, session_id, None, now, now)
            ).fetchone()
        )
        if row is None:  # pragma: no cover - ON CONFLICT always returns a row
            raise DatabaseOperationError("session state insert returned no row")
        return row_to_session_state(row)

    def count(self) -> int:
        """Return the number of tracked sessions (test/ops helper)."""
        row = self._run(
            lambda conn: conn.execute(SELECT_SESSION_STATE_COUNT_SQL).fetchone()
        )
        return int(row[0]) if row is not None else 0


__all__ = [
    "PostgresSessionStateRepository",
    "row_to_session_state",
]