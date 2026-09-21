"""
Shared helpers for Phase 11 authentication/authorization tests (and the Phase
9B/10 tests that now run behind auth).

Everything here is test-only. In particular :data:`TEST_JWT_SECRET` is a
deterministic, sufficiently long secret used ONLY by the test suite — it never
appears in application code or configuration and is meaningless in production.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.auth.dependencies import CurrentUser, get_current_user
from backend.app.config import Settings

# Deterministic test-only secret (>= 32 chars, not a known placeholder).
TEST_JWT_SECRET = "phase11-test-secret-" + "X9" * 24

# A password that satisfies the registration policy (>= 8 chars).
VALID_PASSWORD = "CorrectHorseBatteryStaple!1"
VALID_PASSWORD_2 = "Another.Completely-Different.Pass!42"

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"


def settings(**overrides) -> Settings:
    """Test ``Settings`` with auth enabled via :data:`TEST_JWT_SECRET`."""
    overrides.setdefault("environment", "test")
    overrides.setdefault("jwt_secret_key", TEST_JWT_SECRET)
    return Settings(**overrides)


def register(
    client: TestClient,
    username: str,
    password: str = VALID_PASSWORD,
    endpoint: str = REGISTER,
):
    """POST ``endpoint`` with register-shaped credentials (no asserts)."""
    return client.post(endpoint, json={"username": username, "password": password})


def login(
    client: TestClient,
    username: str,
    password: str = VALID_PASSWORD,
    endpoint: str = LOGIN,
):
    """POST ``endpoint`` with login-shaped credentials (no asserts)."""
    return client.post(endpoint, json={"username": username, "password": password})


def access_token(client: TestClient, username: str, password: str = VALID_PASSWORD) -> str:
    """Register-independent login; returns the access token (asserts 200)."""
    response = login(client, username, password)
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def auth_headers(
    client: TestClient, username: str, password: str = VALID_PASSWORD
) -> dict:
    """Login and return ``Authorization`` headers for ``username``."""
    return {"Authorization": "Bearer " + access_token(client, username, password)}


def set_current_user(
    app,
    username: str = "alice",
    user_id: str = "11111111-1111-4111-8111-111111111111",
) -> None:
    """Override :func:`get_current_user` so requests run as ``username``.

    Used by the Phase 9B/10 route tests whose concern is the ML behavior, not
    the token mechanics (the token mechanics are covered by the Phase 11 auth
    test modules).
    """
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        id=user_id, username=username
    )


__all__ = [
    "LOGIN",
    "REGISTER",
    "TEST_JWT_SECRET",
    "VALID_PASSWORD",
    "VALID_PASSWORD_2",
    "access_token",
    "auth_headers",
    "login",
    "register",
    "set_current_user",
    "settings",
]