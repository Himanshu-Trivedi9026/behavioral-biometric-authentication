"""
Phase 11 — real PostgreSQL user-repository integration tests.

Requires a live PostgreSQL server, pointed at via the ``TEST_DATABASE_URL``
environment variable. Every test in this module is SKIPPED when the variable
is unset, so the normal test run never touches a database:

    TEST_DATABASE_URL=postgresql://user:pass@host:5432/testdb \
        python -m pytest tests/test_postgres_user_integration.py -q

What is exercised against the real server:

* schema creation via ``db/migrations/002_create_users.sql``
* insert + retrieve round-trip (UUID id, username, Argon2id hash, timestamps)
* duplicate username -> race-safe 409 :class:`UsernameExistsError`
* unknown-username / unknown-id lookups -> ``None``
* persistence across repository instances (create via A, close A, read via B)
* connection cleanup (every leased connection returned to the pool; close())

The module exercises an isolated table named ``users`` (the same name the
migration creates). It DROPs the table at start and at the end of the run, so
it must only ever be pointed at a dedicated TEST database — never at a
database holding real application data.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from backend.app.repositories.errors import (
    DatabaseOperationError,
    DatabaseUnavailableError,
)
from backend.app.repositories.postgres_user_repository import (
    PostgresUserRepository,
)
from backend.app.repositories.user_repository import UserRecord, UsernameExistsError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIGRATION_PATH = os.path.join(REPO_ROOT, "db", "migrations", "002_create_users.sql")

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")

requires_database = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is not set; real-PostgreSQL integration tests skipped",
)

pytestmark = requires_database


def make_user(username: str, *, user_id: str = "") -> UserRecord:
    timestamp = "2025-01-02T03:04:05.123456+00:00"
    return UserRecord(
        id=user_id or "22222222-2222-4222-8222-222222222222",
        username=username,
        password_hash="$argon2id$v=19$m=65536,t=3,p=4$TEST$onlyed",
        created_at=timestamp,
        updated_at=timestamp,
    )


def _unique_user_id(seed: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, "test://postgres/users/" + seed))


@pytest.fixture(scope="module")
def database():
    """A dedicated users table: schema re-applied and dropped per module run."""
    import psycopg

    with open(MIGRATION_PATH, "r", encoding="utf-8") as handle:
        migration_sql = handle.read()

    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS users CASCADE")
        conn.execute(migration_sql)
        conn.commit()

    try:
        yield TEST_DATABASE_URL
    finally:
        try:
            with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
                conn.execute("DROP TABLE IF EXISTS users CASCADE")
        except Exception:  # noqa: BLE001 - teardown best-effort only
            pass


def _schema_columns(database_url: str) -> set:
    import psycopg

    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'users'
            ORDER BY ordinal_position
            """
        ).fetchall()
    return {(row[0], row[1], row[2]) for row in rows}


class TestSchemaCreation:
    def test_migration_creates_expected_schema(self, database):
        columns = _schema_columns(database)
        expected = {
            ("id", "uuid", "NO"),
            ("username", "text", "NO"),
            ("password_hash", "text", "NO"),
            ("created_at", "timestamp with time zone", "NO"),
            ("updated_at", "timestamp with time zone", "NO"),
        }
        assert expected.issubset(columns)
        # Private/behavioural columns must never exist.
        names = {col[0] for col in columns}
        assert not {"keyboard_events", "mouse_events", "centroid", "jwt"}.intersection(names)

    def test_username_unique_constraint_guards_duplicates(self, database):
        import psycopg

        with psycopg.connect(database) as conn:
            constraints = set(
                conn.execute(
                    """
                    SELECT conname, contype
                    FROM pg_constraint
                    WHERE conrelid = 'users'::regclass
                    """
                ).fetchall()
            )
        assert constraints.intersection({("users_username_key", "u"), ("users_username_key", "p")}) or any(
            name for name, kind in constraints if name == "users_username_key" and kind in ("u", "p")
        )


class TestRepositoryOperations:
    def test_insert_and_retrieve_round_trip(self, database):
        repo = PostgresUserRepository(database)
        try:
            user = make_user("alice", user_id=_unique_user_id("alice"))
            assert repo.create(user) is user
            restored = repo.get_by_username("alice")
            assert restored == user
            assert restored.id == user.id
            assert restored.password_hash == user.password_hash
        finally:
            repo.close()

    def test_duplicate_raises_username_exists(self, database):
        repo = PostgresUserRepository(database)
        try:
            user = make_user("carol", user_id=_unique_user_id("carol"))
            repo.create(user)
            with pytest.raises(UsernameExistsError) as excinfo:
                repo.create(make_user("carol", user_id=_unique_user_id("carol2")))
            error = excinfo.value
            assert error.code == "username_exists"
            assert error.status_code == 409
            assert "carol" in error.message
            assert repo.get_by_username("carol").id == user.id  # never overwritten
        finally:
            repo.close()

    def test_unknown_user_is_none(self, database):
        repo = PostgresUserRepository(database)
        try:
            assert repo.get_by_username("ghost") is None
            assert repo.get_by_id("00000000-0000-4000-8000-000000000000") is None
            assert repo.contains_username("ghost") is False
        finally:
            repo.close()

    def test_get_by_id_round_trip(self, database):
        repo = PostgresUserRepository(database)
        try:
            user = make_user("dave", user_id=_unique_user_id("dave"))
            repo.create(user)
            assert repo.get_by_id(user.id) == user
        finally:
            repo.close()


class TestPersistenceAcrossInstances:
    def test_user_survives_repository_restart(self, database):
        user_id = _unique_user_id("frank")
        repo_a = PostgresUserRepository(database)
        repo_a.create(make_user("frank", user_id=user_id))
        repo_a.close()
        assert repo_a.pool is None

        repo_b = PostgresUserRepository(database)
        try:
            restored = repo_b.get_by_username("frank")
            assert restored is not None
            assert restored.id == user_id
            assert restored.username == "frank"
            assert restored.created_at == "2025-01-02T03:04:05.123456+00:00"
            assert restored.updated_at == "2025-01-02T03:04:05.123456+00:00"
        finally:
            repo_b.close()


class TestTransactionBehaviour:
    def test_pool_reusable_after_aborted_duplicate(self, database):
        repo = PostgresUserRepository(database)
        try:
            repo.create(make_user("grace", user_id=_unique_user_id("grace")))
            with pytest.raises(UsernameExistsError):
                repo.create(make_user("grace", user_id=_unique_user_id("grace2")))
            # The same repository must still be able to write fresh rows.
            repo.create(make_user("ida", user_id=_unique_user_id("ida")))
            assert repo.get_by_username("ida") is not None
        finally:
            repo.close()

    def test_committed_rows_are_durable(self, database):
        import psycopg

        repo = PostgresUserRepository(database)
        try:
            repo.create(make_user("henry", user_id=_unique_user_id("henry")))
        finally:
            repo.close()
        with psycopg.connect(database) as conn:
            row = conn.execute(
                "SELECT username FROM users WHERE username = %s", ("henry",)
            ).fetchone()
        assert row == ("henry",)


class TestConnectionCleanup:
    def test_every_connection_returned_to_pool(self, database):
        repo = PostgresUserRepository(database, max_pool_size=4)
        try:
            repo.create(make_user("kim", user_id=_unique_user_id("kim")))
            repo.get_by_username("kim")
            repo.contains_username("kim")
            stats = repo.pool.get_stats()
            assert stats["pool_available"] == stats["pool_size"]
            assert stats["requests_waiting"] == 0
        finally:
            repo.close()

    def test_close_releases_pool_and_is_idempotent(self, database):
        repo = PostgresUserRepository(database)
        repo.create(make_user("leo", user_id=_unique_user_id("leo")))
        assert repo.pool is not None
        repo.close()
        assert repo.pool is None
        repo.close()
        assert repo.pool is None


class TestUnavailableRouting:
    def test_missing_url_is_unavailable_without_touching_db(self):
        repo = PostgresUserRepository("")
        with pytest.raises(DatabaseUnavailableError):
            repo.create(make_user("nina"))
        assert repo.pool is None

    def test_unreachable_host_is_unavailable(self):
        repo = PostgresUserRepository(
            "postgresql://user:pass@127.0.0.1:1/behavioral",
            connect_timeout=1.0,
            pool_timeout=1.0,
        )
        with pytest.raises(DatabaseUnavailableError):
            repo.create(make_user("oscar"))
        repo.close()


class TestStorageSupportsArgon2idHashes:
    def test_long_hash_and_timestamps_round_trip(self, database):
        repo = PostgresUserRepository(database)
        long_hash = "$argon2id$v=19$m=65536,t=3,p=4$" + "A" * 64 + "=="
        user = UserRecord(
            id=_unique_user_id("long"),
            username="longhash",
            password_hash=long_hash,
            created_at="2025-06-01T12:30:45.123456+00:00",
            updated_at="2025-06-01T12:30:45.123456+00:00",
        )
        try:
            repo.create(user)
            restored = repo.get_by_username("longhash")
            assert restored.password_hash == long_hash
            assert restored.created_at == user.created_at
            assert restored.updated_at == user.updated_at
        finally:
            repo.close()

    def test_username_whitespace_check_enforced_by_database(self, database):
        import psycopg

        with psycopg.connect(database) as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO users (id, username, password_hash, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (_unique_user_id("blank"), "   ", "x", datetime.now(timezone.utc), datetime.now(timezone.utc)),
                )
            except psycopg.errors.CheckViolation:
                ok = True
            else:
                ok = False
        assert ok, "the users_username_non_blank CHECK constraint must reject blank usernames"