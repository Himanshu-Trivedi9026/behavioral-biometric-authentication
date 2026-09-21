"""
Phase 14B — real PostgreSQL per-session verification state integration tests.

Requires a live PostgreSQL server, pointed at via the ``TEST_DATABASE_URL``
environment variable. Every test in this module is SKIPPED when the variable is
unset, so the normal test run never touches a database:

    TEST_DATABASE_URL=postgresql://user:pass@host:5432/testdb \
        python -m pytest tests/test_postgres_session_state_integration.py -q

What is exercised against the real server:

* schema creation via ``db/migrations/003_create_session_states.sql`` and the
  exact aggregate-only column set + composite PRIMARY KEY;
* repository contract: clean get_or_create, atomic upserts, counter rules,
  last_verified_at semantics;
* persistence across repository instances (write via A, close A, read via B);
* durability + connection cleanup (leases returned, close() idempotent);
* unavailable routing (missing URL / unreachable host -> DatabaseUnavailableError);
* the aggregates-only privacy guarantee (never raw columns).

The module manages the ``behavioral_session_states`` table itself: it is
dropped and recreated from the migration SQL at module start and dropped again
at module end. Point it only at a dedicated TEST database.
"""

from __future__ import annotations

import os

import pytest

from backend.app.repositories.errors import DatabaseUnavailableError
from backend.app.repositories.postgres_session_state_repository import (
    PostgresSessionStateRepository,
    row_to_session_state,
)
from backend.app.repositories.session_state_repository import (
    SESSION_STATE_REVERIFICATION_REQUIRED,
    SESSION_STATE_VERIFIED,
    SessionVerificationState,
)
from backend.app.services.profile_store import utc_now

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIGRATION_PATH = os.path.join(REPO_ROOT, "db", "migrations", "003_create_session_states.sql")

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")

requires_database = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is not set; real-PostgreSQL integration tests skipped",
)

pytestmark = requires_database


@pytest.fixture(scope="module")
def database():
    """A dedicated test database: schema re-applied and dropped per module run."""
    import psycopg

    with open(MIGRATION_PATH, "r", encoding="utf-8") as handle:
        migration_sql = handle.read()

    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS behavioral_session_states CASCADE")
        conn.execute(migration_sql)
        conn.commit()

    try:
        yield TEST_DATABASE_URL
    finally:
        try:
            with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
                conn.execute("DROP TABLE IF EXISTS behavioral_session_states CASCADE")
        except Exception:  # noqa: BLE001 - teardown best-effort only
            pass


def _make_repo(url, **kwargs):
    return PostgresSessionStateRepository(url, **kwargs)


class TestSchemaCreation:
    def test_migration_creates_expected_schema(self, database):
        import psycopg

        with psycopg.connect(database) as conn:
            columns = {
                (row[0], row[1], row[2])
                for row in conn.execute(
                    """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'behavioral_session_states'
                    ORDER BY ordinal_position
                    """
                ).fetchall()
            }
        expected = {
            ("user_ref", "text", "NO"),
            ("session_id", "text", "NO"),
            ("state", "text", "NO"),
            ("consecutive_suspicious", "integer", "NO"),
            ("last_verified_at", "timestamp with time zone", "YES"),
            ("created_at", "timestamp with time zone", "NO"),
            ("updated_at", "timestamp with time zone", "NO"),
        }
        assert expected.issubset(columns)

    def test_composite_primary_key_covers_user_and_session(self, database):
        import psycopg

        with psycopg.connect(database) as conn:
            rows = conn.execute(
                """
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = 'behavioral_session_states'
                  AND tc.constraint_type = 'PRIMARY KEY'
                ORDER BY kcu.ordinal_position
                """
            ).fetchall()
        assert [r[0] for r in rows] == ["user_ref", "session_id"]

    def test_state_check_constraint_rejects_unknown_values(self, database):
        import psycopg

        with psycopg.connect(database) as conn:
            with pytest.raises(psycopg.errors.CheckViolation):
                conn.execute(
                    "INSERT INTO behavioral_session_states "
                    "(user_ref, session_id, state, consecutive_suspicious, last_verified_at, created_at, updated_at) "
                    "VALUES (%s, %s, 'expired', 0, NULL, now(), now())",
                    ("nobody", "n1"),
                )


class TestRepositoryContract:
    def test_get_or_create_creates_clean_row_and_is_idempotent(self, database):
        repo = _make_repo(database)
        try:
            before = repo.count()
            now = utc_now()
            first = repo.get_or_create("alice", "log1", now=now)
            assert first.state == SESSION_STATE_VERIFIED
            assert first.consecutive_suspicious == 0
            assert first.last_verified_at is None
            assert first.created_at == now
            assert repo.count() == before + 1  # created exactly one row

            second = repo.get_or_create("alice", "log1", now=utc_now())
            assert second == first  # untouched by a second get_or_create
            assert repo.count() == before + 1  # still exactly one row
        finally:
            repo.close()

    def test_get_returns_none_for_untouched_session(self, database):
        repo = _make_repo(database)
        try:
            assert repo.get("ghost", "never") is None
        finally:
            repo.close()

    def test_mark_suspicious_blocks_and_increments_per_session(self, database):
        repo = _make_repo(database)
        try:
            one = repo.mark_suspicious("bob", "log1", now=utc_now())
            assert one.state == SESSION_STATE_REVERIFICATION_REQUIRED
            assert one.consecutive_suspicious == 1
            two = repo.mark_suspicious("bob", "log1", now=utc_now())
            assert two.consecutive_suspicious == 2
            # a different session for the same user counts independently
            fresh = repo.mark_suspicious("bob", "log2", now=utc_now())
            assert fresh.consecutive_suspicious == 1
        finally:
            repo.close()

    def test_mark_verified_clears_counter_and_stamps_time(self, database):
        repo = _make_repo(database)
        try:
            repo.mark_suspicious("carol", "log1", now=utc_now())
            repo.mark_suspicious("carol", "log1", now=utc_now())
            verified_at = utc_now()
            state = repo.mark_verified("carol", "log1", now=verified_at)
            assert state.state == SESSION_STATE_VERIFIED
            assert state.consecutive_suspicious == 0
            assert state.last_verified_at is not None
            restored = repo.get("carol", "log1")
            assert restored == state
            assert restored.last_verified_at == verified_at
        finally:
            repo.close()

    def test_mark_suspicious_preserves_last_verified_at(self, database):
        repo = _make_repo(database)
        try:
            repo.mark_verified("dave", "log1", now=utc_now())
            state = repo.mark_suspicious("dave", "log1", now=utc_now())
            assert state.last_verified_at is not None
            assert state.consecutive_suspicious == 1
        finally:
            repo.close()

    def test_mark_on_fresh_session_inserts(self, database):
        repo = _make_repo(database)
        try:
            before = repo.count()
            verified = repo.mark_verified("erin", "log1", now=utc_now())
            assert verified.state == SESSION_STATE_VERIFIED
            assert verified.consecutive_suspicious == 0
            assert repo.count() == before + 1
            assert repo.get("erin", "log1") is not None
        finally:
            repo.close()

    def test_row_to_session_state_round_trip_isolation(self):
        state = SessionVerificationState(
            user_ref="alice",
            session_id="log1",
            state=SESSION_STATE_REVERIFICATION_REQUIRED,
            consecutive_suspicious=3,
            last_verified_at="2025-06-01T00:00:00+00:00",
            created_at="2025-06-01T00:00:00+00:00",
            updated_at="2025-06-01T00:00:10+00:00",
        )
        row = (
            state.user_ref,
            state.session_id,
            state.state,
            state.consecutive_suspicious,
            state.last_verified_at,
            state.created_at,
            state.updated_at,
        )
        restored = row_to_session_state(row)
        assert restored == state


class TestPersistenceAcrossInstances:
    def test_state_survives_repository_restart(self, database):
        repo_a = _make_repo(database)
        repo_a.mark_suspicious("frank", "log1", now=utc_now())
        repo_a.close()
        assert repo_a.pool is None

        repo_b = _make_repo(database)
        try:
            restored = repo_b.get("frank", "log1")
            assert restored is not None
            assert restored.state == SESSION_STATE_REVERIFICATION_REQUIRED
            assert restored.consecutive_suspicious == 1
        finally:
            repo_b.close()


class TestTransactionAndCleanup:
    def test_commit_is_durable_across_connections(self, database):
        import psycopg

        repo = _make_repo(database)
        try:
            repo.mark_suspicious("grace", "log1", now=utc_now())
        finally:
            repo.close()
        with psycopg.connect(database) as conn:
            row = conn.execute(
                "SELECT state, consecutive_suspicious FROM behavioral_session_states WHERE user_ref = %s AND session_id = %s",
                ("grace", "log1"),
            ).fetchone()
        assert row == ("reverification_required", 1)

    def test_pool_reusable_after_operations(self, database):
        repo = _make_repo(database, max_pool_size=4)
        try:
            repo.get_or_create("henry", "log1", now=utc_now())
            repo.mark_suspicious("henry", "log1", now=utc_now())
            repo.get("henry", "log1")
            repo.count()
            stats = repo.pool.get_stats()
            assert stats["pool_available"] == stats["pool_size"]
            assert stats["requests_waiting"] == 0
        finally:
            repo.close()

    def test_close_releases_pool_and_is_idempotent(self, database):
        repo = _make_repo(database)
        repo.get_or_create("ida", "log1", now=utc_now())
        assert repo.pool is not None
        repo.close()
        assert repo.pool is None
        repo.close()
        assert repo.pool is None


class TestUnavailableRouting:
    def test_missing_url_is_unavailable_without_touching_db(self):
        repo = _make_repo("")
        with pytest.raises(DatabaseUnavailableError):
            repo.mark_verified("nobody", "log1", now=utc_now())
        assert repo.pool is None

    def test_unreachable_host_is_unavailable(self):
        repo = _make_repo(
            "postgresql://user:pass@127.0.0.1:1/behavioral",
            connect_timeout=1.0,
            pool_timeout=1.0,
        )
        with pytest.raises(DatabaseUnavailableError):
            repo.get_or_create("oscar", "log1", now=utc_now())
        repo.close()


class TestPrivacy:
    def test_aggregates_only_no_raw_columns(self, database):
        """The table exposes ONLY the seven aggregate columns."""
        import psycopg

        with psycopg.connect(database) as conn:
            columns = {
                row[0]
                for row in conn.execute(
                    "SELECT column_name FROM information_schema.columns"
                    " WHERE table_name = 'behavioral_session_states'"
                ).fetchall()
            }
        assert columns == {
            "user_ref",
            "session_id",
            "state",
            "consecutive_suspicious",
            "last_verified_at",
            "created_at",
            "updated_at",
        }
        for forbidden in ("keyboard_events", "mouse_events", "centroid", "embedding", "password", "token"):
            assert forbidden not in columns