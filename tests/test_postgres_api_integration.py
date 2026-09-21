"""
Phase 13 — full-flow application persistence over real PostgreSQL.

Requires a live PostgreSQL server, pointed at via ``TEST_DATABASE_URL``. Every
test is SKIPPED when the variable is unset, so the normal test run never
touches a database:

    TEST_DATABASE_URL=postgresql://user:pass@host:5432/testdb \
        .venv/bin/python -m pytest tests/test_postgres_api_integration.py -q

Each test builds a real :func:`create_app` whose settings carry the test
database URL, so FastAPI's factory wiring itself selects the PostgreSQL
repositories (never the in-memory fallbacks). The ML pipeline is a
deterministic stand-in (the persistence contract under test is the database,
not the model) while account/profile storage always uses real PostgreSQL.

Scenarios exercised end-to-end through the HTTP API:

* register + login + enroll on application instance A, then login + verify on
  a brand-new instance B (account and behavioural profile survive the app
  restart/repository teardown);
* duplicate username -> structured 409 from the DB unique constraint;
* duplicate enrollment  -> structured 409 from the profile primary key;
* unreachable database -> structured 503 ``database_unavailable`` with no
  connection details leaked, while ``/health`` stays green (no DB I/O).

The module manages the ``users``, ``behavioral_profiles`` and
``behavioral_session_states`` (Phase 14B) tables itself: they are dropped and
recreated from the migration SQL at module start and dropped again at module
end. Point it only at a dedicated TEST database.
"""

from __future__ import annotations

import os
import uuid
import zlib

import pytest
import torch
from fastapi.testclient import TestClient
from psycopg.errors import UniqueViolation

from backend.app.config import Settings
from backend.app.dependencies import get_ml_service
from backend.app.main import create_app
from backend.app.repositories.postgres_profile_repository import (
    PostgresProfileRepository,
)
from backend.app.repositories.postgres_user_repository import PostgresUserRepository
from backend.app.services.ml_service import BehavioralMLService

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIGRATIONS = [
    os.path.join(REPO_ROOT, "db", "migrations", "001_create_behavioral_profiles.sql"),
    os.path.join(REPO_ROOT, "db", "migrations", "002_create_users.sql"),
    os.path.join(REPO_ROOT, "db", "migrations", "003_create_session_states.sql"),
]

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")

requires_database = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is not set; real-PostgreSQL integration tests skipped",
)

pytestmark = requires_database

# Test-only JWT secret (>= 32 chars). Never a real secret.
JWT_SECRET = "phase13-api-test-secret-key-0123456789abcdef"
PASSWORD = "PersistUser!1x"


class _FakeVerifier:
    def __init__(self, dim=8):
        self.dim = dim
        self.embedding_dim = dim
        self.checkpoint_id = "phase13-fake-verifier"

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


def _app(database_url):
    settings = Settings(
        environment="test",
        database_url=database_url,
        jwt_secret_key=JWT_SECRET,
    )
    service = BehavioralMLService(
        settings,
        verifier_factory=lambda s: _FakeVerifier(),
        preprocessing_loader=lambda s: (None, None, {"artifact": "behavioral-preprocessing", "artifact_version": 1}),
        verification_config_loader=lambda s: {"artifact": "verification-config", "calibration": {"threshold": 1.0}},
    )
    app = create_app(settings)
    app.dependency_overrides[get_ml_service] = lambda: service
    return app


def _close_app_repos(app):
    for name in ("user_repository", "profile_store", "session_state_store"):
        repository = getattr(app.state, name, None)
        close = getattr(repository, "close", None)
        if callable(close):
            close()


def _query(sql, params=()):
    import psycopg

    with psycopg.connect(TEST_DATABASE_URL) as conn:
        return conn.execute(sql, params).fetchall()


@pytest.fixture(scope="module")
def database():
    """Fresh users + behavioral_profiles schema per module run."""
    import psycopg

    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS behavioral_session_states CASCADE")
        conn.execute("DROP TABLE IF EXISTS behavioral_profiles CASCADE")
        conn.execute("DROP TABLE IF EXISTS users CASCADE")
        for path in MIGRATIONS:
            with open(path, "r", encoding="utf-8") as handle:
                conn.execute(handle.read())
        conn.commit()

    try:
        yield TEST_DATABASE_URL
    finally:
        try:
            with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
                conn.execute("DROP TABLE IF EXISTS behavioral_session_states CASCADE")
                conn.execute("DROP TABLE IF EXISTS behavioral_profiles CASCADE")
                conn.execute("DROP TABLE IF EXISTS users CASCADE")
        except Exception:  # noqa: BLE001 - teardown best-effort only
            pass


def _register(client, username):
    return client.post("/api/v1/auth/register", json={"username": username, "password": PASSWORD})


def _login(client, username):
    result = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert result.status_code == 200
    return result.json()["access_token"]


def _enroll(client, token, username):
    return client.post(
        "/api/v1/enrollment",
        headers={"Authorization": "Bearer " + token},
        json={"user_ref": username, "sessions": [_raw_session(username + "_s1")]},
    )


def _unique_username(seed):
    return "p13_{}_{}".format(seed, str(uuid.uuid4())[:8])


class TestPersistenceAcrossApplicationInstances:
    def test_account_and_profile_persist_across_app_restart(self, database):
        username = _unique_username("flow")

        # --- application instance A -------------------------------------------------
        app_a = _app(TEST_DATABASE_URL)
        with TestClient(app_a) as client_a:
            response = _register(client_a, username)
            assert response.status_code == 201
            token = _login(client_a, username)
            enrollment = _enroll(client_a, token, username)
            assert enrollment.status_code == 201
        _close_app_repos(app_a)

        # Rows really live in PostgreSQL before instance B is created.
        users = _query("SELECT username FROM users WHERE username = %s", (username,))
        profiles = _query(
            "SELECT user_ref, embedding_dim, session_count FROM behavioral_profiles WHERE user_ref = %s",
            (username,),
        )
        assert len(users) == 1
        assert len(profiles) == 1
        assert profiles[0][1] == 8  # fake verifier dimension
        assert profiles[0][2] == 1  # one enrolled session

        # --- application instance B (fresh app state, fresh repository pools) -------
        app_b = _app(TEST_DATABASE_URL)
        with TestClient(app_b) as client_b:
            token = _login(client_b, username)
            verification = client_b.post(
                "/api/v1/verification",
                headers={"Authorization": "Bearer " + token},
                json={"user_ref": username, "session": _raw_session(username + "_s2")},
            )
            assert verification.status_code == 200
            body = verification.json()
            assert body["user_ref"] == username
            assert body["decision"] in ("VERIFIED", "SUSPICIOUS")
            assert "distance" in body
            assert "threshold" in body

        user_repo_a = getattr(app_a.state, "user_repository", None)
        user_repo_b = getattr(app_b.state, "user_repository", None)
        profile_store_a = getattr(app_a.state, "profile_store", None)
        profile_store_b = getattr(app_b.state, "profile_store", None)
        assert isinstance(user_repo_a, PostgresUserRepository)
        assert isinstance(user_repo_b, PostgresUserRepository)
        assert isinstance(profile_store_a, PostgresProfileRepository)
        assert isinstance(profile_store_b, PostgresProfileRepository)
        assert user_repo_b is not user_repo_a
        assert profile_store_b is not profile_store_a
        _close_app_repos(app_b)


class TestDuplicateGuardsOverRealPostgres:
    def test_duplicate_username_is_409_and_keeps_one_row(self, database):
        username = _unique_username("dupuser")
        app = _app(TEST_DATABASE_URL)
        with TestClient(app) as client:
            assert _register(client, username).status_code == 201
            duplicate = _register(client, username)
            assert duplicate.status_code == 409
            assert duplicate.json()["error"]["code"] == "username_exists"
        _close_app_repos(app)
        rows = _query("SELECT username FROM users WHERE username = %s", (username,))
        assert len(rows) == 1

    def test_duplicate_enrollment_is_409_and_keeps_one_profile(self, database):
        username = _unique_username("dupprofile")
        app = _app(TEST_DATABASE_URL)
        with TestClient(app) as client:
            assert _register(client, username).status_code == 201
            token = _login(client, username)
            assert _enroll(client, token, username).status_code == 201
            duplicate = _enroll(client, token, username)
            assert duplicate.status_code == 409
            assert duplicate.json()["error"]["code"] == "profile_exists"
        _close_app_repos(app)
        rows = _query(
            "SELECT session_count FROM behavioral_profiles WHERE user_ref = %s", (username,)
        )
        assert len(rows) == 1
        assert rows[0][0] == 1  # never overwritten


class TestDatabaseUnavailableBehaviour:
    UNREACHABLE_URL = "postgresql://user:phantom-secret@127.0.0.1:1/behavioral"

    def test_health_stays_independent_of_database(self):
        app = _app(self.UNREACHABLE_URL)
        with TestClient(app) as client:
            assert client.get("/health").json() == {"status": "ok"}
            assert client.get("/api/v1/health").json() == {"status": "ok"}
            assert self.UNREACHABLE_URL not in client.get("/health").text
            assert "phantom-secret" not in client.get("/health").text

    def test_unreachable_database_returns_structured_503_without_leaks(self):
        app = _app(self.UNREACHABLE_URL)
        with TestClient(app) as client:
            response = _register(client, _unique_username("unreachable"))
            assert response.status_code == 503
            payload = response.json()["error"]
            assert payload["code"] == "database_unavailable"
            for leaked in ("127.0.0.1", "phantom-secret", self.UNREACHABLE_URL, "psycopg", "SELECT"):
                assert leaked not in payload["message"]
            assert "stack" not in payload["message"].lower()