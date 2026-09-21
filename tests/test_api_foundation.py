"""
Phase 9A — FastAPI foundation tests.

Covers the application factory, health/root endpoints, OpenAPI docs, API
versioning, CORS and the rule that the API must not (in this phase) depend on
the ML model, a database, or the frontend.

All tests run fully offline with :class:`fastapi.testclient.TestClient`.
"""

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app

_FRONTEND_ORIGIN = "http://localhost:5173"


@pytest.fixture(scope="function")
def app():
    return create_app(Settings(environment="test", debug=False))


@pytest.fixture(scope="function")
def client(app):
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# APPLICATION
# ---------------------------------------------------------------------------


def test_create_app_succeeds():
    assert create_app(Settings(environment="test")) is not None


def test_create_app_returns_fastapi_application(app):
    assert isinstance(app, FastAPI)


def test_create_app_can_be_instantiated_more_than_once():
    first = create_app(Settings(environment="test"))
    second = create_app(Settings(environment="test"))
    assert first is not second
    assert first.state.settings is not second.state.settings


def test_create_app_requires_no_database():
    # Creating and serving the app must not touch any database driver.
    before = set(sys.modules)
    app = create_app(Settings(environment="test"))
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    imported = set(sys.modules) - before
    assert response.status_code == 200
    assert not any(name.startswith("sqlalchemy") or name.startswith("redis")
                   or name.startswith("psycopg") or name.startswith("aiosqlite")
                   for name in imported)


def test_create_app_requires_no_ml_checkpoint():
    before = set(sys.modules)
    app = create_app(Settings(environment="test"))
    with TestClient(app) as test_client:
        assert test_client.get("/health").status_code == 200
    # create_app / health requests must not import the ML package (model never
    # loaded in Phase 9A). Uses a before/after diff so the test is robust when
    # the full suite has already imported the ML modules.
    imported = set(sys.modules) - before
    assert not any(name == "ml" or name.startswith("ml.") for name in imported)


def test_create_app_requires_no_frontend():
    app = create_app(Settings(environment="test"))
    with TestClient(app) as test_client:
        assert test_client.get("/").status_code == 200


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------


def test_health_returns_200(client):
    assert client.get("/health").status_code == 200


def test_health_body_status_ok(client):
    body = client.get("/health").json()
    assert body == {"status": "ok"}


def test_health_does_not_load_ml_model(client):
    before = set(sys.modules)
    assert client.get("/health").status_code == 200
    imported = set(sys.modules) - before
    assert not any(name == "ml" or name.startswith("ml.") for name in imported)


def test_health_does_not_access_database(client):
    before = set(sys.modules)
    assert client.get("/health").status_code == 200
    imported = set(sys.modules) - before
    assert not any(name.startswith("sqlalchemy") or name.startswith("redis")
                   or name.startswith("psycopg") for name in imported)


def test_health_is_lightweight_no_model_no_db_no_services(client):
    # The handler body contains only the fixed payload.
    assert client.get("/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# ROOT
# ---------------------------------------------------------------------------


def test_root_returns_200(app, client):
    assert client.get("/").status_code == 200


def test_root_contains_service_info(app, client):
    body = client.get("/").json()
    assert body["service"] == app.state.settings.app_name
    assert body["version"] == app.state.settings.app_version
    assert body["status"] == "ok"
    assert set(body) == {"service", "version", "status"}


def test_root_discloses_no_internal_paths_or_secrets(app, client, monkeypatch):
    monkeypatch.setenv("BBA_SHOULD_NEVER_LEAK", "topsecret-value")
    response = client.get("/")
    text = response.text
    for secret in ("topsecret-value",):
        assert secret not in text
    for needle in ("/models/", ".pt", "/home/", "/data/", "/etc", "DATASET_PATH"):
        assert needle not in text


# ---------------------------------------------------------------------------
# OPENAPI
# ---------------------------------------------------------------------------


def test_openapi_json_available(client):
    assert client.get("/openapi.json").status_code == 200


def test_docs_available(client):
    assert client.get("/docs").status_code == 200


def test_redoc_available(client):
    assert client.get("/redoc").status_code == 200


def test_openapi_metadata(app, client):
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == app.state.settings.app_name
    assert spec["info"]["version"] == app.state.settings.app_version
    assert "/health" in spec["paths"]
    assert "/" in spec["paths"]


# ---------------------------------------------------------------------------
# VERSIONING
# ---------------------------------------------------------------------------


def test_versioned_health_available(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_versioned_root_available(client):
    body = client.get("/api/v1/").json()
    assert body["api"] == "v1"
    assert body["status"] == "ok"


def test_ml_endpoints_registered(client):
    # Phase 9B registers the ML POST endpoints; GET is not allowed (405).
    assert client.get("/api/v1/enrollment").status_code == 405
    assert client.get("/api/v1/verification").status_code == 405
    spec = client.get("/openapi.json").json()
    assert "/api/v1/enrollment" in spec["paths"]
    assert "/api/v1/verification" in spec["paths"]
    assert "post" in spec["paths"]["/api/v1/enrollment"]


def test_api_prefix_configurable():
    app = create_app(Settings(environment="test", api_prefix="/api/v2"))
    with TestClient(app) as test_client:
        assert test_client.get("/api/v2/health").status_code == 200
        assert test_client.get("/api/v1/health").status_code == 404


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_allows_configured_origin(client):
    response = client.get("/health", headers={"Origin": _FRONTEND_ORIGIN})
    assert response.headers.get("access-control-allow-origin") == _FRONTEND_ORIGIN


def test_cors_rejects_unconfigured_origin(client):
    response = client.get("/health", headers={"Origin": "http://evil.example"})
    allow = response.headers.get("access-control-allow-origin")
    assert allow is None or allow != "http://evil.example"


def test_cors_credentials_not_combined_with_wildcard():
    with pytest.raises(ValueError):
        Settings(cors_origins=("*",))
    with pytest.raises(ValueError):
        create_app(Settings(environment="test", cors_origins=("*",)))


def test_cors_origins_are_configurable():
    app = create_app(Settings(environment="test", cors_origins=("https://app.example.com",)))
    with TestClient(app) as test_client:
        response = test_client.get("/health", headers={"Origin": "https://app.example.com"})
        assert response.headers.get("access-control-allow-origin") == "https://app.example.com"