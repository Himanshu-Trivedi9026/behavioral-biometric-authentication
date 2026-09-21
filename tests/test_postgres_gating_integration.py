"""
Phase 14B — full-flow session-state gating over real PostgreSQL.

Requires a live PostgreSQL server, pointed at via ``TEST_DATABASE_URL``. Every
test is SKIPPED when the variable is unset. Mirror of
``tests/test_postgres_api_integration.py``: each test builds a real
:func:`create_app` whose settings carry the test database URL, so FastAPI's
factory wiring selects the PostgreSQL repositories (profiles, users AND the new
session-state store) via migration 003.

Scenarios exercised end-to-end through the HTTP API:

* a fresh JWT login starts clean: enrollment/verification are allowed, and the
  new ``GET /continuous-verification/state`` endpoint reports ``verified``;
* a SUSPICIOUS continuous window flips the session state to
  ``reverification_required`` PERSISTENTLY (survives repository restart);
* enrollment + verification then fail closed with 403 ``reverification_required``
  while continuous verification stays available and a VERIFIED window clears
  the gate;
* a fresh login (new JWT ``iat`` -> new session id) is unaffected by the old
  session's block (per-session, never account-wide);
* an unreachable database yields the structured 503 (fail closed), with
  ``/health`` still green and no connection details leaked.

The module manages the ``users``, ``behavioral_profiles`` and
``behavioral_session_states`` tables itself (drop + recreate from migrations at
start, drop at end). Point it only at a dedicated TEST database.
"""

from __future__ import annotations

import os
import time
import uuid
import zlib

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.dependencies import get_ml_service
from backend.app.main import create_app
from backend.app.repositories.postgres_session_state_repository import (
    PostgresSessionStateRepository,
)
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

JWT_SECRET = "phase14b-pg-test-secret-key-0123456789abcd"
PASSWORD = "PersistGate!1x"

CONTINUOUS_ENDPOINT = "/api/v1/continuous-verification"
STATE_ENDPOINT = "/api/v1/continuous-verification/state"
ENROLLMENT_ENDPOINT = "/api/v1/enrollment"
VERIFICATION_ENDPOINT = "/api/v1/verification"


class _FakeVerifier:
    def __init__(self, dim=8):
        self.dim = dim
        self.embedding_dim = dim
        self.checkpoint_id = "phase14b-pg-fake-verifier"

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
    """Fresh users + profiles + session-states schema per module run."""
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
    return client.post(
        "/api/v1/auth/register", json={"username": username, "password": PASSWORD}
    )


def _login(client, username):
    result = client.post(
        "/api/v1/auth/login", json={"username": username, "password": PASSWORD}
    )
    assert result.status_code == 200
    return result.json()["access_token"]


def _enroll(client, token, username):
    return client.post(
        ENROLLMENT_ENDPOINT,
        headers={"Authorization": "Bearer " + token},
        json={"user_ref": username, "sessions": [_raw_session(username + "_s1")]},
    )


def _unique_username(seed):
    return "p14b_{}_{}".format(seed, str(uuid.uuid4())[:8])


class TestFreshLoginStartsClean:
    def test_fresh_login_state_is_verified_and_actions_allowed(self, database):
        username = _unique_username("fresh")
        app = _app(TEST_DATABASE_URL)
        with TestClient(app) as client:
            assert _register(client, username).status_code == 201
            token = _login(client, username)

            assert client.get(STATE_ENDPOINT, headers=auth(token)).json() == {
                "session_state": "verified",
                "consecutive_suspicious": 0,
                "last_verified_at": None,
            }
            assert _enroll(client, token, username).status_code == 201
            verification = client.post(
                VERIFICATION_ENDPOINT,
                headers=auth(token),
                json={"user_ref": username, "session": _raw_session(username + "_s1")},
            )
            assert verification.status_code == 200
        _close_app_repos(app)


class TestGatingPersistsAndRecovers:
    def test_suspicious_window_gates_then_verified_window_recovers(self, database):
        username = _unique_username("gated")
        app = _app(TEST_DATABASE_URL)
        with TestClient(app) as client:
            assert _register(client, username).status_code == 201
            token = _login(client, username)
            assert _enroll(client, token, username).status_code == 201

            # a far window is SUSPICIOUS -> the session becomes gated
            suspicious = client.post(
                CONTINUOUS_ENDPOINT,
                headers=auth(token),
                json=_raw_session("far_from_" + username),
            )
            assert suspicious.status_code == 200
            assert suspicious.json()["decision"] == "SUSPICIOUS"
            assert suspicious.json()["session_state"] == "reverification_required"

            # protected actions now fail CLOSED
            blocked_verify = client.post(
                VERIFICATION_ENDPOINT,
                headers=auth(token),
                json={"user_ref": username, "session": _raw_session(username + "_s2")},
            )
            assert blocked_verify.status_code == 403
            assert blocked_verify.json()["error"]["code"] == "reverification_required"
            assert _enroll(client, token, username).status_code == 403

            # continuous verification stays available and is the recovery path
            recovery = client.post(
                CONTINUOUS_ENDPOINT,
                headers=auth(token),
                json=_raw_session(username + "_s1"),
            )
            assert recovery.status_code == 200
            assert recovery.json()["decision"] == "VERIFIED"
            assert recovery.json()["session_state"] == "verified"

            unblocked = client.post(
                VERIFICATION_ENDPOINT,
                headers=auth(token),
                json={"user_ref": username, "session": _raw_session(username + "_s3")},
            )
            assert unblocked.status_code == 200
            assert client.get(STATE_ENDPOINT, headers=auth(token)).json()[
                "session_state"
            ] == "verified"
        _close_app_repos(app)

    def test_gate_survives_repository_restart(self, database):
        username = _unique_username("restart")
        app = _app(TEST_DATABASE_URL)
        with TestClient(app) as client:
            assert _register(client, username).status_code == 201
            token = _login(client, username)
            assert _enroll(client, token, username).status_code == 201
            assert client.post(
                CONTINUOUS_ENDPOINT, headers=auth(token), json=_raw_session("far_x")
            ).json()["decision"] == "SUSPICIOUS"
        _close_app_repos(app)

        # Replaying the SAME token (its SHA-256 digest is the key) on a
        # brand-new app must still observe the persisted block — it really
        # lives in PostgreSQL and survives repository teardown.
        app_b = _app(TEST_DATABASE_URL)
        with TestClient(app_b) as client_b:
            assert client_b.get(STATE_ENDPOINT, headers=auth(token)).json()[
                "session_state"
            ] == "reverification_required"
            verification = client_b.post(
                VERIFICATION_ENDPOINT,
                headers=auth(token),
                json={"user_ref": username, "session": _raw_session(username + "_s4")},
            )
            assert verification.status_code == 403
            assert verification.json()["error"]["code"] == "reverification_required"
        _close_app_repos(app_b)


class TestFreshLoginBypassesOldSessionBlock:
    def test_new_login_is_not_gated_by_previous_session(self, database):
        username = _unique_username("relogin")
        app = _app(TEST_DATABASE_URL)
        with TestClient(app) as client:
            assert _register(client, username).status_code == 201
            token = _login(client, username)
            assert _enroll(client, token, username).status_code == 201
            assert client.post(
                CONTINUOUS_ENDPOINT, headers=auth(token), json=_raw_session("far_y")
            ).json()["decision"] == "SUSPICIOUS"
            assert client.post(
                VERIFICATION_ENDPOINT,
                headers=auth(token),
                json={"user_ref": username, "session": _raw_session(username + "_s5")},
            ).status_code == 403

            # a fresh login has a NEW token -> new session_id -> clean state
            # (the header nonce guarantees a distinct token even within the
            # same iat second; the sleep additionally separates the second).
            time.sleep(1.05)
            fresh_token = _login(client, username)
            assert client.get(STATE_ENDPOINT, headers=auth(fresh_token)).json()[
                "session_state"
            ] == "verified"
            verification = client.post(
                VERIFICATION_ENDPOINT,
                headers=auth(fresh_token),
                json={"user_ref": username, "session": _raw_session(username + "_s6")},
            )
            assert verification.status_code == 200
        _close_app_repos(app)


class TestDatabaseUnavailableFailsClosed:
    UNREACHABLE_URL = "postgresql://user:phantom-secret@127.0.0.1:1/behavioral"

    def test_health_stays_independent_of_database(self):
        app = _app(self.UNREACHABLE_URL)
        with TestClient(app) as client:
            assert client.get("/health").json() == {"status": "ok"}
            assert "phantom-secret" not in client.get("/health").text
        _close_app_repos(app)

    def test_protected_actions_return_structured_503(self):
        username = _unique_username("down")
        app = _app(self.UNREACHABLE_URL)
        with TestClient(app) as client:
            register = _register(client, username)
            assert register.status_code == 503
            payload = register.json()["error"]
            assert payload["code"] == "database_unavailable"
            for leaked in ("127.0.0.1", "phantom-secret", self.UNREACHABLE_URL, "psycopg", "SELECT"):
                assert leaked not in payload["message"]
        _close_app_repos(app)


class TestPersistenceWiring:
    def test_session_state_store_uses_postgres_repository(self, database):
        username = _unique_username("wiring")
        app = _app(TEST_DATABASE_URL)
        with TestClient(app) as client:
            assert _register(client, username).status_code == 201
            token = _login(client, username)
            client.get(STATE_ENDPOINT, headers=auth(token))
        store = getattr(app.state, "session_state_store", None)
        assert isinstance(store, PostgresSessionStateRepository)
        rows = _query(
            "SELECT user_ref, session_id, state FROM behavioral_session_states WHERE user_ref = %s",
            (username,),
        )
        assert len(rows) == 1
        assert rows[0][0] == username
        assert rows[0][2] == "verified"
        _close_app_repos(app)


def auth(token):
    return {"Authorization": "Bearer " + token}