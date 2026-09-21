"""
Phase 10 — PostgreSQL profile repository tests (no database required).

Covers two layers without any live PostgreSQL server:

1. **Pure mapping helpers** — ``profile_to_values`` / ``row_to_profile`` /
   ``row_to_metadata`` / ``is_unique_violation`` / ``raise_for_database_error``:
   exact SQL value order, metadata never includes the centroid, DB errors are
   translated into the structured 409/500/503 domain errors with no SQL, URLs,
   hostnames or credentials leaking into messages.

2. **Repository behaviour over fakes** — a scripted in-memory PostgreSQL
   stand-in (fake pool + fake connection) drives the real
   :class:`PostgresProfileRepository` so ``save``/``get``/``contains``/``count``/
   ``list_profiles`` are verified to run the expected statements with the
   real profile values, to honour the primary-key duplicate guard (409
   :class:`ProfileExistsError`), and to return every connection to the pool.

Also covers the Phase 10 dependency wiring (in-memory fallback when no
``database_url``, persistent repository when one is configured, never
connecting on resolution) and API behaviour over a PostgreSQL-backed store.

Real-PostgreSQL integration (schema, transactions, persistence across
instances, connection cleanup) lives in ``tests/test_postgres_integration.py``
and is gated by ``TEST_DATABASE_URL``.
"""

from __future__ import annotations

import zlib
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from fastapi.testclient import TestClient
from psycopg.errors import CheckViolation, IntegrityError, UniqueViolation

from backend.app.config import Settings
from backend.app.dependencies import get_ml_service, get_profile_store
from backend.app.errors import AppError
from backend.app.main import create_app
from backend.app.repositories.errors import (
    DatabaseOperationError,
    DatabaseUnavailableError,
)
from backend.app.repositories.postgres_profile_repository import (
    INSERT_PROFILE_SQL,
    PostgresProfileRepository,
    SELECT_COUNT_SQL,
    SELECT_EXISTS_SQL,
    SELECT_METADATA_SQL,
    SELECT_PROFILE_SQL,
    is_unique_violation,
    profile_to_values,
    raise_for_database_error,
    row_to_metadata,
    row_to_profile,
)
from backend.app.repositories.profile_repository import ProfileRepository
from backend.app.services.ml_service import BehavioralMLService
from backend.app.services.profile_store import (
    InMemoryProfileStore,
    ProfileExistsError,
    StoredProfile,
)
from tests.auth_testing import set_current_user

FAKE_DB_URL = "postgresql://user:super-secret@db.example.com:5432/behavioral"


def make_profile(user_ref: str = "alice", *, dim: int = 8, sessions: int = 3) -> StoredProfile:
    return StoredProfile(
        user_ref=user_ref,
        centroid=tuple(float(i) for i in range(dim)),
        embedding_dim=dim,
        session_count=sessions,
        created_at="2025-01-02T03:04:05.123456+00:00",
        updated_at="2025-01-02T03:04:05.123456+00:00",
    )


# ---------------------------------------------------------------------------
# Fakes (scripted and stateful PostgreSQL stand-ins)
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows: List[Tuple[Any, ...]]):
        self._rows = list(rows)
        self._index = 0

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return list(self._rows)


class _RecordedConnection:
    """A minimal connection that records statements and can be scripted."""

    def __init__(
        self,
        rows_for: Optional[Dict[str, List[Tuple[Any, ...]]]] = None,
        *,
        error_for: Optional[str] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        self.calls: List[Tuple[str, Any]] = []
        self.commits = 0
        self._rows_for = rows_for or {}
        self._error_for = error_for
        self._error = error
        self.transactions = 0

    def transaction(self):
        return _FakeTransaction(self)

    def execute(self, statement: str, params: Optional[Tuple[Any, ...]] = None) -> _FakeCursor:
        self.calls.append((statement, params))
        if self._error_for == statement and self._error is not None:
            raise self._error
        return _FakeCursor(self._rows_for.get(statement, []))


class _FakeTransaction:
    def __init__(self, conn: _RecordedConnection) -> None:
        self.conn = conn

    def __enter__(self):
        self.conn.transactions += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.conn.commits += 1
        return False


class _FakePool:
    """A pool whose connection() context manager tracks outstanding leases."""

    def __init__(self, conn: _RecordedConnection) -> None:
        self.conn = conn
        self.closed = False
        self.checked_out = 0

    def connection(self):
        return _ConnectionLease(self)

    def close(self) -> None:
        self.closed = True


class _ConnectionLease:
    def __init__(self, pool: _FakePool) -> None:
        self.pool = pool

    def __enter__(self):
        self.pool.checked_out += 1
        return self.pool.conn

    def __exit__(self, *exc) -> bool:
        self.pool.checked_out -= 1
        return False


def _repo(pool: _FakePool, *, url: str = FAKE_DB_URL) -> PostgresProfileRepository:
    repo = PostgresProfileRepository(url)
    repo._pool = pool  # inject the fake; _ensure_pool() then returns it
    return repo


class _StoredPostgres:
    """Stateful fake database: rows keyed by user_ref, unique on insert.

    Mirrors the real behavioural_profiles table closely enough for repository
    and API tests: INSERT raises :class:`UniqueViolation` on a duplicate
    ``user_ref`` (the race-safe guard) and SELECTs return rows in real column
    order with driver-typed timestamps.
    """

    def __init__(self) -> None:
        self._rows: Dict[str, Tuple[Any, ...]] = {}
        self.duplicates = 0

    def _insert(self, values: Tuple[Any, ...]) -> None:
        user_ref = values[0]
        if user_ref in self._rows:
            self.duplicates += 1
            raise UniqueViolation('duplicate key value violates unique constraint "behavioral_profiles_pkey"')
        self._rows[user_ref] = (
            user_ref,
            list(values[1]),
            values[2],
            values[3],
            datetime.fromisoformat(values[4]),
            datetime.fromisoformat(values[5]),
        )

    def execute(self, statement: str, params: Optional[Tuple[Any, ...]] = None):
        if statement == INSERT_PROFILE_SQL:
            self._insert(tuple(params))
            return _FakeCursor([])
        if statement == SELECT_PROFILE_SQL:
            row = self._rows.get(params[0])
            return _FakeCursor([row] if row is not None else [])
        if statement == SELECT_EXISTS_SQL:
            return _FakeCursor([(1,)] if params[0] in self._rows else [])
        if statement == SELECT_COUNT_SQL:
            return _FakeCursor([(len(self._rows),)])
        if statement == SELECT_METADATA_SQL:
            rows = [(r[0], r[2], r[3], r[4], r[5]) for r in self._rows.values()]
            return _FakeCursor(sorted(rows))
        raise AssertionError("unexpected statement in fake postgres: {!r}".format(statement))  # pragma: no cover

    def make_connection(self) -> _RecordedConnection:
        conn = _RecordedConnection()
        conn.execute = self.execute  # type: ignore[method-assign]
        return conn


def _inject_pool(repo: PostgresProfileRepository, conn: _RecordedConnection) -> _FakePool:
    pool = _FakePool(conn)
    repo._pool = pool
    return pool


# ---------------------------------------------------------------------------
# Value / row mapping
# ---------------------------------------------------------------------------


class TestProfileToValues:
    def test_ordered_values_match_insert_columns(self):
        profile = make_profile("alice", dim=4, sessions=2)
        values = profile_to_values(profile)
        assert values == (
            "alice",
            [0.0, 1.0, 2.0, 3.0],
            4,
            2,
            "2025-01-02T03:04:05.123456+00:00",
            "2025-01-02T03:04:05.123456+00:00",
        )

    def test_value_count_matches_insert_placeholders(self):
        profile = make_profile("bob")
        assert len(profile_to_values(profile)) == INSERT_PROFILE_SQL.count("%s")

    def test_rejects_non_stored_profile(self):
        with pytest.raises(TypeError):
            profile_to_values({"user_ref": "nope"})  # type: ignore[arg-type]


class TestRowToProfile:
    def test_round_trip_with_driver_timestamps(self):
        profile = make_profile("carol", dim=8, sessions=5)
        values = profile_to_values(profile)
        row = (
            values[0],
            values[1],
            values[2],
            values[3],
            datetime.fromisoformat(values[4]),
            datetime.fromisoformat(values[5]),
        )
        restored = row_to_profile(row)
        assert restored == profile

    def test_centroid_always_float_tuple(self):
        row = ("dave", [0, 1, 2], 3, 1, datetime.now(timezone.utc), datetime.now(timezone.utc))
        restored = row_to_profile(row)
        assert isinstance(restored.centroid, tuple)
        assert all(isinstance(v, float) for v in restored.centroid)
        assert restored.embedding_dim == 3

    def test_missing_timestamp_raises_structured_error(self):
        row = ("dave", [0.0], 1, 1, None, None)
        with pytest.raises(DatabaseOperationError) as excinfo:
            row_to_profile(row)
        assert excinfo.value.code == "database_error"
        assert "unavailable" not in str(excinfo.value.message)


class TestRowToMetadata:
    def test_never_includes_centroid(self):
        profile = make_profile("erin", dim=16, sessions=9)
        values = profile_to_values(profile)
        now = datetime.now(timezone.utc)
        row = (values[0], values[2], values[3], now, now)
        metadata = row_to_metadata(row)
        assert metadata["user_ref"] == "erin"
        assert metadata["embedding_dim"] == 16
        assert metadata["session_count"] == 9
        assert "centroid" not in metadata
        assert set(metadata) == {"user_ref", "embedding_dim", "session_count", "created_at", "updated_at"}


# ---------------------------------------------------------------------------
# Error handling / translation
# ---------------------------------------------------------------------------


class TestErrorTranslation:
    def test_unique_violation_detected(self):
        assert is_unique_violation(UniqueViolation("dup"))
        assert not is_unique_violation(CheckViolation("check"))
        assert not is_unique_violation(IntegrityError("integrity"))
        assert not is_unique_violation(ValueError("nope"))

    def test_operational_error_is_unavailable(self):
        import psycopg

        with pytest.raises(DatabaseUnavailableError):
            raise_for_database_error(psycopg.OperationalError("boom"))
        with pytest.raises(DatabaseUnavailableError):
            raise_for_database_error(psycopg.InterfaceError("boom"))

    def test_pool_timeout_and_closed_are_unavailable(self):
        from psycopg_pool import PoolClosed, PoolTimeout

        with pytest.raises(DatabaseUnavailableError):
            raise_for_database_error(PoolTimeout())
        with pytest.raises(DatabaseUnavailableError):
            raise_for_database_error(PoolClosed())

    def test_other_database_errors_are_operation_errors(self):
        import psycopg

        with pytest.raises(DatabaseOperationError):
            raise_for_database_error(psycopg.errors.SyntaxError("bad sql"))

    def test_unknown_errors_are_operation_errors(self):
        with pytest.raises(DatabaseOperationError):
            raise_for_database_error(ValueError("surprise"))

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: DatabaseOperationError(),
            lambda: DatabaseUnavailableError(),
        ],
    )
    def test_messages_never_leak_sql_host_or_credentials(self, factory):
        error = factory()
        assert isinstance(error, AppError)
        assert error.status_code in (500, 503)
        for secret in ("SELECT", "postgres", "db.example.com", "super-secret", FAKE_DB_URL):
            assert secret not in error.message
            assert secret not in error.code


# ---------------------------------------------------------------------------
# Repository behaviour over fakes
# ---------------------------------------------------------------------------


class TestSave:
    def test_insert_runs_with_exact_profile_values(self):
        conn = _RecordedConnection()
        pool = _FakePool(conn)
        repo = _repo(pool)
        profile = make_profile("alice", dim=8, sessions=2)

        result = repo.save(profile)

        assert result is profile
        assert conn.calls == [(INSERT_PROFILE_SQL, profile_to_values(profile))]
        assert conn.commits == 1
        assert conn.transactions == 1
        assert pool.checked_out == 0  # lease returned

    def test_duplicate_insert_raises_profile_exists(self):
        conn = _RecordedConnection(error_for=INSERT_PROFILE_SQL, error=UniqueViolation("dup"))
        pool = _FakePool(conn)
        repo = _repo(pool)

        with pytest.raises(ProfileExistsError) as excinfo:
            repo.save(make_profile("alice"))

        error = excinfo.value
        assert error.code == "profile_exists"
        assert error.status_code == 409
        assert "alice" in error.message
        assert INSERT_PROFILE_SQL not in error.message
        assert pool.checked_out == 0

    def test_duplicate_is_race_safe_via_primary_key(self):
        # The repository must NOT pre-check; the second save must fail with
        # the unique-violation translated to 409 rather than overwriting.
        state = _StoredPostgres()
        repo = _repo(_FakePool(state.make_connection()))
        repo.save(make_profile("alice", sessions=2))
        with pytest.raises(ProfileExistsError):
            repo.save(make_profile("alice", sessions=9))
        assert state._rows["alice"][3] == 2  # original row untouched
        assert state.duplicates == 1

    def test_missing_database_url_is_unavailable(self):
        repo = PostgresProfileRepository("")
        with pytest.raises(DatabaseUnavailableError):
            repo.save(make_profile("alice"))
        assert repo.pool is None

    def test_pool_connection_failure_is_unavailable(self):
        import psycopg

        conn = _RecordedConnection(error_for=INSERT_PROFILE_SQL, error=psycopg.OperationalError("down"))
        repo = _repo(_FakePool(conn))
        with pytest.raises(DatabaseUnavailableError):
            repo.save(make_profile("alice"))

    def test_other_database_error_is_operation_error(self):
        import psycopg

        conn = _RecordedConnection(error_for=INSERT_PROFILE_SQL, error=psycopg.errors.SyntaxError("bad"))
        repo = _repo(_FakePool(conn))
        with pytest.raises(DatabaseOperationError) as excinfo:
            repo.save(make_profile("alice"))
        assert excinfo.value.status_code == 500


class TestQueryMethods:
    def test_get_returns_restored_profile(self):
        profile = make_profile("alice", dim=8, sessions=3)
        values = profile_to_values(profile)
        row = (
            values[0],
            values[1],
            values[2],
            values[3],
            datetime.fromisoformat(values[4]),
            datetime.fromisoformat(values[5]),
        )
        conn = _RecordedConnection(rows_for={SELECT_PROFILE_SQL: [row]})
        repo = _repo(_FakePool(conn))

        assert repo.get("alice") == profile
        assert conn.calls == [(SELECT_PROFILE_SQL, ("alice",))]

    def test_get_missing_returns_none(self):
        conn = _RecordedConnection(rows_for={SELECT_PROFILE_SQL: []})
        repo = _repo(_FakePool(conn))
        assert repo.get("ghost") is None

    def test_contains(self):
        conn = _RecordedConnection(rows_for={SELECT_EXISTS_SQL: [(1,)]})
        repo = _repo(_FakePool(conn))
        assert repo.contains("alice") is True
        conn2 = _RecordedConnection(rows_for={SELECT_EXISTS_SQL: []})
        repo2 = _repo(_FakePool(conn2))
        assert repo2.contains("ghost") is False

    def test_count(self):
        conn = _RecordedConnection(rows_for={SELECT_COUNT_SQL: [(7,)]})
        repo = _repo(_FakePool(conn))
        assert repo.count() == 7

    def test_list_profiles_metadata_only(self):
        now = datetime.now(timezone.utc)
        conn = _RecordedConnection(
            rows_for={
                SELECT_METADATA_SQL: [
                    ("alice", 8, 3, now, now),
                    ("bob", 8, 1, now, now),
                ]
            }
        )
        repo = _repo(_FakePool(conn))
        result = repo.list_profiles()
        assert len(result) == 2
        assert all("centroid" not in metadata for metadata in result)
        assert result[0]["user_ref"] == "alice"
        assert result[0]["session_count"] == 3


class TestPoolLifecycle:
    def test_pool_property_is_none_before_use(self):
        repo = PostgresProfileRepository(FAKE_DB_URL)
        assert repo.pool is None

    def test_pool_created_lazily_and_closed(self):
        from psycopg_pool import ConnectionPool

        repo = PostgresProfileRepository("postgresql://user:pass@localhost:5432/db")
        assert repo.pool is None
        pool = repo._ensure_pool()
        assert isinstance(pool, ConnectionPool)
        assert repo.pool is pool
        repo.close()
        assert repo.pool is None

    def test_missing_url_never_creates_a_pool(self):
        repo = PostgresProfileRepository("")
        with pytest.raises(DatabaseUnavailableError):
            repo._ensure_pool()
        assert repo.pool is None

    def test_close_is_idempotent_and_closes_pool(self):
        conn = _RecordedConnection()
        pool = _FakePool(conn)
        repo = _repo(pool, url=FAKE_DB_URL)
        assert repo.pool is pool
        repo.close()
        assert pool.closed
        assert repo.pool is None
        repo.close()  # second close must not raise
        assert repo.pool is None

    def test_operations_return_every_connection(self):
        state = _StoredPostgres()
        repo = _repo(_FakePool(state.make_connection()))
        repo.save(make_profile("alice"))
        repo.get("alice")
        repo.contains("alice")
        repo.count()
        repo.list_profiles()
        assert repo.pool.checked_out == 0
        assert not repo.pool.closed


# ---------------------------------------------------------------------------
# Phase 10 dependency wiring
# ---------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, settings: Settings) -> None:
        self.app = SimpleNamespace(state=SimpleNamespace(settings=settings))


class TestDependencyWiring:
    def test_in_memory_fallback_without_database_url(self):
        request = _FakeRequest(Settings(environment="test"))
        store = get_profile_store(request)
        assert isinstance(store, InMemoryProfileStore)
        assert store is get_profile_store(request)  # cached on app.state
        assert not hasattr(request.app.state.profile_store, "pool")

    def test_postgres_repository_when_database_url_configured(self):
        request = _FakeRequest(Settings(environment="test", database_url=FAKE_DB_URL))
        store = get_profile_store(request)
        assert isinstance(store, PostgresProfileRepository)
        assert store.pool is None  # lazy: no connection on resolution
        assert store is get_profile_store(request)


# ---------------------------------------------------------------------------
# ProfileRepository contract over the stateful fake database
# ---------------------------------------------------------------------------


class TestRepositoryContractAgainstFakeDatabase:
    def _repo(self) -> PostgresProfileRepository:
        state = _StoredPostgres()
        pool = _FakePool(state.make_connection())
        repo = _repo(pool)
        return repo

    def test_follows_same_contract_as_in_memory_store(self):
        repo = self._repo()
        assert isinstance(repo, ProfileRepository)
        assert not repo.contains("alice")
        assert repo.get("alice") is None
        assert repo.count() == 0

        saved = repo.save(make_profile("alice", dim=8, sessions=2))
        assert repo.contains("alice")
        assert repo.count() == 1
        assert repo.get("alice") == saved
        assert repo.list_profiles()[0]["user_ref"] == "alice"

        with pytest.raises(ProfileExistsError):
            repo.save(make_profile("alice", sessions=9))
        assert repo.get("alice").session_count == 2


# ---------------------------------------------------------------------------
# API behaviour over the persistent repository
# ---------------------------------------------------------------------------


class _FakeVerifier:
    def __init__(self, dim=8):
        self.dim = dim

    @staticmethod
    def _vec(key):
        generator = torch.Generator()
        generator.manual_seed(zlib.crc32(key.encode("utf-8")))
        return torch.randn((8,), generator=generator)

    def embed_session(self, session, keyboard_scaler=None, mouse_scaler=None):
        return self._vec(str(session.get("session_id", "unknown")))

    def embed_sessions(self, sessions, keyboard_scaler=None, mouse_scaler=None, batch_size=16):
        return torch.stack([self.embed_session(s) for s in sessions])

    def pair_distance(self, a, b):
        return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-8)


def _raw_session(session_id, *, kb=3, mouse=3):
    t = 0.0
    keyboard_events = []
    for i in range(kb):
        keyboard_events.append({"event_type": "keyboard", "event": "keydown", "timestamp": t})
        t += 50.0
        keyboard_events.append({"event_type": "keyboard", "event": "keyup", "timestamp": t})
        t += 100.0
    x = y = mt = 0.0
    mouse_events = []
    for i in range(mouse):
        x += 4.0 + i
        y += 2.0
        mt += 16.0
        mouse_events.append(
            {"event_type": "mouse", "event": "mousemove", "x": x, "y": y, "timestamp": mt}
        )
    return {
        "session_id": session_id,
        "started_at": "2025-01-01T00:00:00Z",
        "ended_at": "2025-01-01T00:00:05Z",
        "timestamp_source": "test",
        "keyboard_events": keyboard_events,
        "mouse_events": mouse_events,
    }


@pytest.fixture(scope="function")
def postgres_client():
    state = _StoredPostgres()
    store = PostgresProfileRepository(FAKE_DB_URL)
    store._pool = _FakePool(state.make_connection())
    verifier = _FakeVerifier()
    service = BehavioralMLService(
        Settings(environment="test", debug=False),
        verifier_factory=lambda s: verifier,
        preprocessing_loader=lambda s: (None, None, {"artifact": "behavioral-preprocessing", "artifact_version": 1}),
        verification_config_loader=lambda s: {"artifact": "verification-config", "calibration": {"threshold": 1.0}},
    )
    app = create_app(Settings(environment="test", debug=False))
    app.dependency_overrides[get_ml_service] = lambda: service
    app.dependency_overrides[get_profile_store] = lambda: store
    set_current_user(app, username="alice")
    with TestClient(app) as client:
        yield client, store, state


class TestPostgresBackedAPI:
    def test_enrollment_persists_through_repository(self, postgres_client):
        client, store, state = postgres_client
        response = client.post(
            "/api/v1/enrollment",
            json={"user_ref": "alice", "sessions": [_raw_session("alice_s1")]},
        )
        assert response.status_code == 201
        assert response.json() == {
            "user_ref": "alice",
            "status": "enrolled",
            "embedding_dimension": 8,
            "session_count": 1,
        }
        assert store.contains("alice")
        assert store.get("alice").embedding_dim == 8

    def test_duplicate_enrollment_409_via_repository(self, postgres_client):
        client, _, _ = postgres_client
        payload = {"user_ref": "alice", "sessions": [_raw_session("alice_s1")]}
        assert client.post("/api/v1/enrollment", json=payload).status_code == 201
        response = client.post("/api/v1/enrollment", json=payload)
        assert response.status_code == 409
        assert response.json() == {
            "error": {
                "code": "profile_exists",
                "message": "a profile already exists for user 'alice'",
            }
        }

    def test_verification_against_persisted_profile(self, postgres_client):
        client, _, _ = postgres_client
        client.post(
            "/api/v1/enrollment",
            json={"user_ref": "alice", "sessions": [_raw_session("alice_s1")]},
        )
        response = client.post(
            "/api/v1/verification",
            json={"user_ref": "alice", "session": _raw_session("alice_s1")},
        )
        assert response.status_code == 200
        assert response.json()["decision"] == "VERIFIED"

    def test_unknown_user_still_404(self, postgres_client):
        client, _, _ = postgres_client
        set_current_user(client.app, username="ghost")
        response = client.post(
            "/api/v1/verification",
            json={"user_ref": "ghost", "session": _raw_session("ghost_s1")},
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "profile_not_found"