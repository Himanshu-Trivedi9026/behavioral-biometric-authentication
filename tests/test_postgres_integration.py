"""
Phase 10 — real PostgreSQL integration tests.

Requires a live PostgreSQL server, pointed at via the ``TEST_DATABASE_URL``
environment variable. Every test in this module is SKIPPED when the variable is
unset, so the normal test run never touches a database:

    TEST_DATABASE_URL=postgresql://user:pass@host:5432/testdb \
        python -m pytest tests/test_postgres_integration.py -q

What is exercised against the real server:

* schema creation via ``db/migrations/001_create_behavioral_profiles.sql``
* insert + retrieve + centroid round-trip (float precision, dim, timestamps)
* duplicate insert -> race-safe 409 ``ProfileExistsError`` (primary key)
* missing profile -> ``None``
* persistence across repository instances (save via A, close A, read via B)
* transaction/durability behaviour (committed rows survive; a failed insert
  leaves the table consistent and the pool connection reusable)
* connection cleanup (every leased connection returned to the pool; close())

The module exercises an isolated table named ``behavioral_profiles`` (the same
name the migration creates). It DROPs the table at start and at the end of the
run, so it must only ever be pointed at a dedicated TEST database — never at a
database holding real application data.
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from backend.app.repositories.errors import (
    DatabaseOperationError,
    DatabaseUnavailableError,
)
from backend.app.repositories.postgres_profile_repository import (
    PostgresProfileRepository,
)
from backend.app.services.profile_store import ProfileExistsError, StoredProfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIGRATION_PATH = os.path.join(REPO_ROOT, "db", "migrations", "001_create_behavioral_profiles.sql")

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")

requires_database = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is not set; real-PostgreSQL integration tests skipped",
)

pytestmark = requires_database


def make_profile(user_ref: str = "alice", *, dim: int = 8, sessions: int = 3) -> StoredProfile:
    timestamp = "2025-01-02T03:04:05.123456+00:00"
    return StoredProfile(
        user_ref=user_ref,
        centroid=tuple(float(i) + 0.25 for i in range(dim)),
        embedding_dim=dim,
        session_count=sessions,
        created_at=timestamp,
        updated_at=timestamp,
    )


@pytest.fixture(scope="module")
def database():
    """A dedicated test database: schema re-applied and dropped per module run."""
    import psycopg

    with open(MIGRATION_PATH, "r", encoding="utf-8") as handle:
        migration_sql = handle.read()

    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS behavioral_profiles CASCADE")
        conn.execute(migration_sql)
        conn.commit()

    created = True
    try:
        yield TEST_DATABASE_URL
    finally:
        if created:
            try:
                with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
                    conn.execute("DROP TABLE IF EXISTS behavioral_profiles CASCADE")
            except Exception:  # noqa: BLE001 - teardown best-effort only
                pass


def _schema_columns(database_url: str) -> set:
    import psycopg

    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'behavioral_profiles'
            ORDER BY ordinal_position
            """
        ).fetchall()
    return {(row[0], row[1], row[2]) for row in rows}


class TestSchemaCreation:
    def test_migration_creates_expected_schema(self, database):
        columns = _schema_columns(database)
        expected = {
            ("user_ref", "text", "NO"),
            ("centroid", "ARRAY", "NO"),
            ("embedding_dim", "integer", "NO"),
            ("session_count", "integer", "NO"),
            ("created_at", "timestamp with time zone", "NO"),
            ("updated_at", "timestamp with time zone", "NO"),
        }
        assert expected.issubset(columns)

    def test_primary_key_guards_duplicates(self, database):
        import psycopg

        with psycopg.connect(database) as conn:
            keys = conn.execute(
                """
                SELECT tc.table_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = 'behavioral_profiles'
                  AND tc.constraint_type = 'PRIMARY KEY'
                """
            ).fetchall()
        assert ("behavioral_profiles", "user_ref") in keys


class TestRepositoryOperations:
    def test_insert_and_retrieve_round_trip(self, database):
        repo = PostgresProfileRepository(database)
        try:
            profile = make_profile("alice", dim=8, sessions=2)
            assert repo.save(profile) is profile
            restored = repo.get("alice")
            assert restored == profile
        finally:
            repo.close()

    def test_centroid_round_trip_preserves_values(self, database):
        repo = PostgresProfileRepository(database)
        try:
            profile = make_profile("bob", dim=128, sessions=5)
            repo.save(profile)
            restored = repo.get("bob")
            assert restored.centroid == profile.centroid
            assert restored.embedding_dim == 128
            assert restored.session_count == 5
            assert restored.created_at == profile.created_at
            assert restored.updated_at == profile.updated_at
        finally:
            repo.close()

    def test_duplicate_raises_profile_exists(self, database):
        repo = PostgresProfileRepository(database)
        try:
            repo.save(make_profile("carol", sessions=1))
            with pytest.raises(ProfileExistsError) as excinfo:
                repo.save(make_profile("carol", sessions=9))
            error = excinfo.value
            assert error.code == "profile_exists"
            assert error.status_code == 409
            assert "carol" in error.message
            assert repo.get("carol").session_count == 1  # never overwritten
        finally:
            repo.close()

    def test_missing_profile_is_none(self, database):
        repo = PostgresProfileRepository(database)
        try:
            assert repo.get("ghost") is None
            assert repo.contains("ghost") is False
        finally:
            repo.close()

    def test_contains_count_list_profiles(self, database):
        repo = PostgresProfileRepository(database)
        try:
            before = repo.count()
            repo.save(make_profile("dave", sessions=3))
            repo.save(make_profile("erin", sessions=1))
            assert repo.contains("dave")
            assert repo.count() == before + 2
            metadata = repo.list_profiles()
            assert {"dave", "erin"} <= {m["user_ref"] for m in metadata}
            by_ref = {m["user_ref"]: m for m in metadata}
            assert by_ref["dave"]["session_count"] == 3
            assert by_ref["erin"]["session_count"] == 1
            assert all("centroid" not in m for m in metadata)
        finally:
            repo.close()


class TestPersistenceAcrossInstances:
    def test_profile_survives_repository_restart(self, database):
        # --- repository instance A -------------------------------------------------
        repo_a = PostgresProfileRepository(database)
        profile = make_profile("frank", dim=16, sessions=4)
        repo_a.save(profile)
        repo_a.close()
        assert repo_a.pool is None

        # --- repository instance B (new pool, new process-like instance) -----------
        repo_b = PostgresProfileRepository(database)
        try:
            restored = repo_b.get("frank")
            assert restored is not None
            assert restored.user_ref == "frank"
            assert restored.centroid == profile.centroid
            assert restored.embedding_dim == 16
            assert restored.session_count == 4
            assert restored.created_at == profile.created_at
            assert restored.updated_at == profile.updated_at
        finally:
            repo_b.close()


class TestTransactionBehaviour:
    def test_commit_is_durable_across_connections(self, database):
        import psycopg

        repo = PostgresProfileRepository(database)
        try:
            repo.save(make_profile("grace", sessions=2))
        finally:
            repo.close()

        with psycopg.connect(database) as conn:
            row = conn.execute(
                "SELECT user_ref, embedding_dim, session_count FROM behavioral_profiles WHERE user_ref = %s",
                ("grace",),
            ).fetchone()
        assert row == ("grace", 8, 2)

    def test_pool_reusable_after_aborted_transaction(self, database):
        """A failed (unique-violation) insert must not poison the pooled connection."""
        repo = PostgresProfileRepository(database)
        try:
            repo.save(make_profile("henry", sessions=1))
            with pytest.raises(ProfileExistsError):
                repo.save(make_profile("henry", sessions=5))
            # the same session/repository must still be able to write fresh rows
            repo.save(make_profile("ida", sessions=7))
            assert repo.get("henry").session_count == 1
            assert repo.get("ida").session_count == 7
        finally:
            repo.close()

    def test_failed_operation_leaves_table_consistent(self, database):
        repo = PostgresProfileRepository(database)
        try:
            repo.save(make_profile("jane", sessions=3))
            with pytest.raises(ProfileExistsError):
                repo.save(make_profile("jane", sessions=4))
            rows = repo.list_profiles()
            counts = [m["session_count"] for m in rows if m["user_ref"] == "jane"]
            assert counts == [3]
        finally:
            repo.close()


class TestConnectionCleanup:
    def test_every_connection_returned_to_pool(self, database):
        repo = PostgresProfileRepository(database, max_pool_size=4)
        try:
            repo.save(make_profile("kim", sessions=1))
            repo.get("kim")
            repo.count()
            stats = repo.pool.get_stats()
            assert stats["pool_available"] == stats["pool_size"]
            assert stats["requests_waiting"] == 0
        finally:
            repo.close()

    def test_close_releases_pool_and_is_idempotent(self, database):
        repo = PostgresProfileRepository(database)
        repo.save(make_profile("leo", sessions=1))
        assert repo.pool is not None
        repo.close()
        assert repo.pool is None
        repo.close()  # idempotent
        assert repo.pool is None


class TestUnavailableRouting:
    def test_missing_url_is_unavailable_without_touching_db(self):
        repo = PostgresProfileRepository("")
        with pytest.raises(DatabaseUnavailableError):
            repo.save(make_profile("nina"))
        assert repo.pool is None

    def test_unreachable_host_is_unavailable(self):
        repo = PostgresProfileRepository(
            "postgresql://user:pass@127.0.0.1:1/behavioral",
            connect_timeout=1.0,
            pool_timeout=1.0,
        )
        with pytest.raises(DatabaseUnavailableError):
            repo.save(make_profile("oscar"))
        repo.close()


class TestDurabilitySmoke:
    def test_aggregates_only_no_raw_columns(self, database):
        """The persistence layer must only expose the six aggregate columns."""
        import psycopg

        with psycopg.connect(database) as conn:
            columns = {row[0] for row in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'behavioral_profiles'"
            ).fetchall()}
        assert columns == {
            "user_ref",
            "centroid",
            "embedding_dim",
            "session_count",
            "created_at",
            "updated_at",
        }
        assert "keyboard_events" not in columns
        assert "raw_data" not in columns