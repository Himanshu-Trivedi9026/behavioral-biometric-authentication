"""
Phase 11 — user repository tests (no live database required).

Covers two layers without any live PostgreSQL server:

1. **Pure mapping helpers** — ``user_to_values`` / ``row_to_user``: exact SQL
   value order and driver-timestamp round-trips.

2. **Repository behaviour over fakes** — a scripted in-memory PostgreSQL
   stand-in drives the real :class:`PostgresUserRepository` so ``create`` /
   ``get_by_username`` / ``get_by_id`` / ``contains_username`` run the
   expected statements with the real values, the UNIQUE constraint guards
   duplicates (409 :class:`UsernameExistsError`), and DB failures translate
   into the structured 503/500 domain errors with no SQL/URL/credential
   leakage.

Also covers the in-memory implementation and the Phase 11 dependency wiring
(in-memory without ``database_url``, persistent repository with one, never
connecting on resolution).

Real-PostgreSQL integration lives in ``tests/test_postgres_user_integration.py``
and is gated by ``TEST_DATABASE_URL``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest
from psycopg.errors import CheckViolation, UniqueViolation

from backend.app.config import Settings
from backend.app.dependencies import get_user_repository
from backend.app.repositories.errors import (
    DatabaseOperationError,
    DatabaseUnavailableError,
)
from backend.app.repositories.memory_user_repository import InMemoryUserRepository
from backend.app.repositories.postgres_user_repository import (
    INSERT_USER_SQL,
    SELECT_USER_BY_ID_SQL,
    SELECT_USER_BY_USERNAME_SQL,
    SELECT_USER_EXISTS_SQL,
    PostgresUserRepository,
    row_to_user,
    user_to_values,
)
from backend.app.repositories.user_repository import (
    UserRecord,
    UserRepository,
    UsernameExistsError,
)
from tests.test_postgres_profile_repository import (
    _FakeCursor,
    _FakePool,
    _RecordedConnection,
)

FAKE_DB_URL = "postgresql://user:super-secret@db.example.com:5432/behavioral"


def make_user(username="alice", *, user_id="11111111-1111-4111-8111-111111111111"):
    return UserRecord(
        id=user_id,
        username=username,
        password_hash="$argon2id$v=19$m=65536,t=3,p=4$test$testsalt",
        created_at="2025-01-02T03:04:05.123456+00:00",
        updated_at="2025-01-02T03:04:05.123456+00:00",
    )


class _StoredUsers:
    """Stateful fake database: rows keyed by username, unique on insert.

    Mirrors the real ``users`` table closely enough for repository tests:
    INSERT raises :class:`UniqueViolation` on a duplicate username (the
    race-safe guard) and SELECTs return rows in real column order with
    driver-typed timestamps.
    """

    def __init__(self) -> None:
        self._rows: Dict[str, Tuple[Any, ...]] = {}
        self._by_id: Dict[str, Tuple[Any, ...]] = {}
        self.duplicates = 0

    def _insert(self, values: Tuple[Any, ...]) -> None:
        user_id, username, password_hash, created, updated = values
        if username in self._rows:
            self.duplicates += 1
            raise UniqueViolation('duplicate key value violates unique constraint "users_username_key"')
        row = (
            user_id,
            username,
            password_hash,
            datetime.fromisoformat(created),
            datetime.fromisoformat(updated),
        )
        self._rows[username] = row
        self._by_id[user_id] = row

    def execute(self, statement: str, params: Optional[Tuple[Any, ...]] = None):
        if statement == INSERT_USER_SQL:
            self._insert(tuple(params))
            return _FakeCursor([])
        if statement == SELECT_USER_BY_USERNAME_SQL:
            row = self._rows.get(params[0])
            return _FakeCursor([row] if row is not None else [])
        if statement == SELECT_USER_BY_ID_SQL:
            row = self._by_id.get(params[0])
            return _FakeCursor([row] if row is not None else [])
        if statement == SELECT_USER_EXISTS_SQL:
            return _FakeCursor([(1,)] if params[0] in self._rows else [])
        raise AssertionError("unexpected statement in fake users db: {!r}".format(statement))  # pragma: no cover

    def make_connection(self) -> _RecordedConnection:
        conn = _RecordedConnection()
        conn.execute = self.execute  # type: ignore[method-assign]
        return conn


def _repo_with(pool: _FakePool, *, url: str = FAKE_DB_URL) -> PostgresUserRepository:
    repo = PostgresUserRepository(url)
    repo._pool = pool
    return repo


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


class TestUserToValues:
    def test_ordered_values_match_insert_columns(self):
        user = make_user("alice")
        assert user_to_values(user) == (
            "11111111-1111-4111-8111-111111111111",
            "alice",
            "$argon2id$v=19$m=65536,t=3,p=4$test$testsalt",
            "2025-01-02T03:04:05.123456+00:00",
            "2025-01-02T03:04:05.123456+00:00",
        )

    def test_value_count_matches_insert_placeholders(self):
        assert len(user_to_values(make_user())) == INSERT_USER_SQL.count("%s")

    def test_rejects_non_user_record(self):
        with pytest.raises(TypeError):
            user_to_values({"username": "nope"})  # type: ignore[arg-type]


class TestRowToUser:
    def test_round_trip_with_driver_timestamps(self):
        user = make_user("carol")
        values = user_to_values(user)
        row = (
            values[0],
            values[1],
            values[2],
            datetime.fromisoformat(values[3]),
            datetime.fromisoformat(values[4]),
        )
        restored = row_to_user(row)
        assert restored == user

    def test_ids_are_always_strings(self):
        values = user_to_values(make_user("dave"))
        row = (values[0], values[1], values[2], datetime.now(timezone.utc), datetime.now(timezone.utc))
        restored = row_to_user(row)
        assert isinstance(restored.id, str)
        assert isinstance(restored.username, str)
        assert isinstance(restored.password_hash, str)

    def test_missing_timestamp_raises_structured_error(self):
        values = user_to_values(make_user("erin"))
        row = (values[0], values[1], values[2], None, None)
        with pytest.raises(DatabaseOperationError):
            row_to_user(row)


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


class TestErrorTranslation:
    def test_unique_violation_exposed_via_username_exists(self):
        state = _StoredUsers()
        repo = _repo_with(_FakePool(state.make_connection()))
        repo.create(make_user("alice"))
        with pytest.raises(UsernameExistsError) as excinfo:
            repo.create(make_user("alice"))
        error = excinfo.value
        assert error.code == "username_exists"
        assert error.status_code == 409
        assert "alice" in error.message
        assert INSERT_USER_SQL not in error.message
        assert state.duplicates == 1
        assert repo.pool.checked_out == 0

    def test_operational_error_is_unavailable(self):
        import psycopg

        conn = _RecordedConnection(error_for=INSERT_USER_SQL, error=psycopg.OperationalError("down"))
        repo = _repo_with(_FakePool(conn))
        with pytest.raises(DatabaseUnavailableError):
            repo.create(make_user("bob"))

    def test_other_database_error_is_operation_error(self):
        import psycopg

        conn = _RecordedConnection(error_for=INSERT_USER_SQL, error=psycopg.errors.SyntaxError("bad"))
        repo = _repo_with(_FakePool(conn))
        with pytest.raises(DatabaseOperationError) as excinfo:
            repo.create(make_user("bob"))
        assert excinfo.value.status_code == 500

    def test_messages_never_leak_sql_host_or_credentials(self):
        for factory in (DatabaseOperationError, DatabaseUnavailableError):
            error = factory()
            assert error.status_code in (500, 503)
            for secret in ("SELECT", "INSERT", "postgres", "db.example.com", "super-secret", "$argon2id", FAKE_DB_URL):
                assert secret not in error.message
                assert secret not in error.code


# ---------------------------------------------------------------------------
# Duplicate protection (race-safe, no pre-check)
# ---------------------------------------------------------------------------


class TestDuplicateUsernameProtection:
    def test_duplicate_is_race_safe_via_unique_constraint(self):
        state = _StoredUsers()
        repo = _repo_with(_FakePool(state.make_connection()))
        repo.create(make_user("alice"))
        with pytest.raises(UsernameExistsError):
            repo.create(make_user("alice"))
        # original row untouched
        assert repo.get_by_username("alice").password_hash == make_user("alice").password_hash

    def test_in_memory_repository_guards_duplicates(self):
        repo = InMemoryUserRepository()
        repo.create(make_user("bob"))
        with pytest.raises(UsernameExistsError):
            repo.create(make_user("bob"))
        assert repo.count() == 1


# ---------------------------------------------------------------------------
# Repository behavior over the in-memory + fake PostgreSQL implementations
# ---------------------------------------------------------------------------


class TestInMemoryUserRepository:
    def test_full_contract(self):
        repo = InMemoryUserRepository()
        assert isinstance(repo, UserRepository)
        assert not repo.contains_username("alice")
        assert repo.get_by_username("alice") is None
        assert repo.get_by_id("nope") is None

        user = make_user("alice")
        assert repo.create(user) is user
        assert repo.contains_username("alice")
        assert repo.get_by_username("alice") is user
        assert repo.get_by_id(user.id) is user

    def test_clear_and_count(self):
        repo = InMemoryUserRepository()
        repo.create(make_user("a", user_id="11111111-1111-4111-8111-111111111111"))
        repo.create(make_user("b", user_id="22222222-2222-4222-8222-222222222222"))
        assert repo.count() == 2
        assert repo.clear() == 2
        assert repo.count() == 0

    def test_rejects_non_user_record(self):
        repo = InMemoryUserRepository()
        with pytest.raises(TypeError):
            repo.create({"username": "x"})  # type: ignore[arg-type]


class TestPostgresUserRepositoryOverFakes:
    def _repo(self) -> PostgresUserRepository:
        return _repo_with(_FakePool(_StoredUsers().make_connection()))

    def test_follows_same_contract_as_in_memory(self):
        repo = self._repo()
        assert not repo.contains_username("alice")
        assert repo.get_by_username("alice") is None
        assert repo.get_by_id("ghost") is None

        user = make_user("alice")
        saved = repo.create(user)
        assert saved is user
        assert repo.contains_username("alice")
        assert repo.get_by_username("alice") == user
        assert repo.get_by_id(user.id) == user

        with pytest.raises(UsernameExistsError):
            repo.create(make_user("alice"))
        assert repo.get_by_username("alice") == user

    def test_get_by_id_uses_id_sql(self):
        user = make_user("bob")
        values = user_to_values(user)
        row = (values[0], values[1], values[2], datetime.fromisoformat(values[3]), datetime.fromisoformat(values[4]))
        conn = _RecordedConnection(rows_for={SELECT_USER_BY_ID_SQL: [row]})
        repo = _repo_with(_FakePool(conn))
        assert repo.get_by_id(user.id) == user
        assert conn.calls == [(SELECT_USER_BY_ID_SQL, (user.id,))]

    def test_missing_database_url_is_unavailable(self):
        repo = PostgresUserRepository("")
        with pytest.raises(DatabaseUnavailableError):
            repo.create(make_user("alice"))
        assert repo.pool is None


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, settings: Settings) -> None:
        self.app = SimpleNamespace(state=SimpleNamespace(settings=settings))


class TestDependencyWiring:
    def test_in_memory_without_database_url(self):
        request = _FakeRequest(Settings(environment="test"))
        repo = get_user_repository(request)
        assert isinstance(repo, InMemoryUserRepository)
        assert repo is get_user_repository(request)  # cached

    def test_postgres_repository_when_database_url_configured(self):
        request = _FakeRequest(Settings(environment="test", database_url=FAKE_DB_URL))
        repo = get_user_repository(request)
        assert isinstance(repo, PostgresUserRepository)
        assert repo.pool is None  # lazy: no connection on resolution
        assert repo is get_user_repository(request)  # cached