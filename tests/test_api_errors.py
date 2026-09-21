"""
Phase 9A — centralized error-handling tests.

Verifies the structured ``{"error": {...}}`` envelope for validation errors,
HTTP exceptions, unknown routes and unexpected internal errors, and confirms
internal details (stack traces, paths, exception messages) never leak into
responses.

All tests run offline against freshly built applications.
"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from tests.auth_testing import settings as auth_settings


def _app_with_error_routes():
    """Application plus a few deliberately failing routes used by the tests."""
    app = create_app(Settings(environment="test", debug=False))

    @app.get("/echo")
    def _echo(count: int):
        return {"count": count}

    @app.get("/forbidden")
    def _forbidden():
        raise HTTPException(status_code=403, detail="forbidden by test")

    @app.get("/boom")
    def _boom():
        raise RuntimeError("TOP-SECRET-INTERNAL-DETAIL")

    return app


@pytest.fixture(scope="function")
def client():
    app = _app_with_error_routes()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# UNKNOWN ROUTE
# ---------------------------------------------------------------------------


def test_unknown_route_is_structured_404(client):
    response = client.get("/definitely-not-a-route")
    assert response.status_code == 404
    body = response.json()
    assert body == {"error": {"code": "not_found", "message": "Not Found"}}


def test_unknown_route_envelope_shape(client):
    body = client.get("/nope").json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}


# ---------------------------------------------------------------------------
# VALIDATION ERRORS
# ---------------------------------------------------------------------------


def test_validation_error_is_structured_422(client):
    response = client.get("/echo?count=abc")
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["message"] == "request validation failed"
    assert isinstance(body["error"]["details"], list)
    assert body["error"]["details"]


def test_validation_error_details_are_plain(client):
    details = client.get("/echo?count=abc").json()["error"]["details"]
    entry = details[0]
    assert set(entry) >= {"loc", "msg", "type"}
    assert isinstance(entry["loc"], list)
    assert isinstance(entry["msg"], str)
    assert isinstance(entry["type"], str)


# ---------------------------------------------------------------------------
# HTTP EXCEPTIONS
# ---------------------------------------------------------------------------


def test_http_exception_structured(client):
    response = client.get("/forbidden")
    assert response.status_code == 403
    body = response.json()
    assert body == {"error": {"code": "forbidden", "message": "forbidden by test"}}


# ---------------------------------------------------------------------------
# INTERNAL ERRORS
# ---------------------------------------------------------------------------


def test_internal_error_is_safe_generic_500(client):
    response = client.get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body == {
        "error": {
            "code": "internal_error",
            "message": "An unexpected internal error occurred.",
        }
    }


def test_internal_details_never_leaked(client):
    response = client.get("/boom")
    text = response.text
    assert "TOP-SECRET-INTERNAL-DETAIL" not in text
    assert "Traceback" not in text
    assert "/home/" not in text
    assert "RuntimeError" not in text
    assert "line " not in text.lower()


# ---------------------------------------------------------------------------
# POST error paths (foundation only)
# ---------------------------------------------------------------------------


def test_method_not_allowed_is_structured(client):
    response = client.post("/health")
    assert response.status_code == 405
    body = response.json()
    assert body["error"]["code"] == "method_not_allowed"
    assert "error" in body


def test_protected_post_routes_exist(client):
    # The ML routes exist; authentication is evaluated before request-body
    # validation, so an unauthenticated empty-body POST is a structured 401
    # (never a 404).
    auth_app = create_app(auth_settings())
    with TestClient(auth_app) as auth_client:
        for endpoint in ("/api/v1/enrollment", "/api/v1/verification"):
            response = auth_client.post(endpoint)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "authentication_required"