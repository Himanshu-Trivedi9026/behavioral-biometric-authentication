"""
Phase 15B — DB-aware readiness endpoint (``GET /health/ready``).

Readiness checks PostgreSQL availability with a lightweight ``SELECT 1`` and
reports the ML service's current load state WITHOUT forcing a load. PostgreSQL
being configured-but-unreachable produces the structured ``503
database_unavailable`` envelope. ``/health`` (liveness) stays a zero-I/O
constant ``200`` and is regressed here.

The database probe is a seam (``health.default_database_probe``); unit tests
monkeypatch it so no real database is needed. A gated live-PostgreSQL test
uses the real probe against ``TEST_DATABASE_URL``.
"""

import os

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.routes import health as health_module
from backend.app.services.ml_service import BehavioralMLService

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")

requires_database = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is not set; real-PostgreSQL readiness tests skipped",
)


class _StubVerifier:
    embedding_dim = 8


def _machine(probe=True, database_url="postgresql://u:p@db:5432/db", *, loaded=False, monkeypatch=None):
    """Build an app whose readiness probe returns ``probe``."""
    if monkeypatch is not None:
        monkeypatch.setattr(health_module, "default_database_probe", lambda url: bool(probe))
    settings = Settings(environment="test", database_url=database_url)
    app = create_app(settings)
    service = BehavioralMLService(
        settings,
        verifier_factory=lambda s: _StubVerifier(),
        preprocessing_loader=lambda s: (None, None, {"artifact": "behavioral-preprocessing", "artifact_version": 1}),
        verification_config_loader=lambda s: {"artifact": "verification-config", "calibration": {"threshold": 1.0}},
    )
    if loaded:
        service.load()
    app.state.ml_service = service
    return app


class TestReadinessHealthy:
    def test_ready_ok_with_database_up(self, monkeypatch):
        app = _machine(probe=True, monkeypatch=monkeypatch)
        with TestClient(app) as client:
            response = client.get("/health/ready")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "ok"
            assert body["database"] == "ok"
            assert body["ml"] == "not_loaded"

    def test_ready_ok_with_ml_loaded(self, monkeypatch):
        app = _machine(probe=True, loaded=True, monkeypatch=monkeypatch)
        with TestClient(app) as client:
            body = client.get("/health/ready").json()
            assert body["database"] == "ok"
            assert body["ml"] == "loaded"

    def test_ready_not_configured_database_is_ok(self, monkeypatch):
        app = _machine(probe=True, database_url="", monkeypatch=monkeypatch)
        with TestClient(app) as client:
            response = client.get("/health/ready")
            assert response.status_code == 200
            assert response.json()["database"] == "not_configured"
            assert response.json()["ml"] == "not_loaded"


class TestReadinessDatabaseUnavailable:
    def test_ready_503_when_database_down(self, monkeypatch):
        app = _machine(probe=False, monkeypatch=monkeypatch)
        with TestClient(app) as client:
            response = client.get("/health/ready")
            assert response.status_code == 503

    def test_readiness_503_uses_structured_envelope(self, monkeypatch):
        app = _machine(probe=False, monkeypatch=monkeypatch)
        with TestClient(app) as client:
            body = client.get("/health/ready").json()
            assert body["error"]["code"] == "database_unavailable"
            assert "message" in body["error"]

    def test_readiness_503_leaks_no_connection_details(self, monkeypatch):
        secret_url = "postgresql://phantom-user:phantom-secret@secret-host:5432/behavioral"
        app = _machine(probe=False, database_url=secret_url, monkeypatch=monkeypatch)
        with TestClient(app) as client:
            text = client.get("/health/ready").text
            assert "phantom-user" not in text
            assert "phantom-secret" not in text
            assert "secret-host" not in text
            assert "postgresql://" not in text
            assert "5432" not in text


class TestHealthLivenessRegression:
    def test_health_unchanged_liveness(self):
        app = _machine(probe=True, monkeypatch=None)
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/health").json() == {"status": "ok"}
            assert client.get("/api/v1/health").json() == {"status": "ok"}

    def test_health_stays_200_when_database_down(self, monkeypatch):
        app = _machine(probe=False, monkeypatch=monkeypatch)
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/health/ready").status_code == 503

    def test_health_does_not_touch_database_probe(self, monkeypatch):
        app = _machine(probe=True, monkeypatch=monkeypatch)
        with TestClient(app) as client:
            assert client.get("/health").json() == {"status": "ok"}


class TestReadinessVersioned:
    def test_versioned_ready_mirrors_canonical(self, monkeypatch):
        app = _machine(probe=True, monkeypatch=monkeypatch)
        with TestClient(app) as client:
            assert client.get("/api/v1/health/ready").status_code == 200
            assert client.get("/api/v1/health/ready").json()["database"] == "ok"
            assert client.get("/api/v1/health/ready").json()["ml"] == "not_loaded"

    def test_versioned_ready_503_on_database_down(self, monkeypatch):
        app = _machine(probe=False, monkeypatch=monkeypatch)
        with TestClient(app) as client:
            response = client.get("/api/v1/health/ready")
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "database_unavailable"


@requires_database
class TestReadinessRealPostgres:
    def test_ready_ok_against_live_database(self):
        settings = Settings(environment="test", database_url=TEST_DATABASE_URL)
        app = create_app(settings)
        with TestClient(app) as client:
            response = client.get("/health/ready")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "ok"
            assert body["database"] == "ok"

    def test_ready_503_against_unreachable_port(self):
        unreachable = TEST_DATABASE_URL.replace(":5432/", ":1/")
        settings = Settings(environment="test", database_url=unreachable)
        app = create_app(settings)
        with TestClient(app) as client:
            response = client.get("/health/ready")
            assert response.status_code == 503
            body = response.json()
            assert body["error"]["code"] == "database_unavailable"
            assert "5432" not in response.text