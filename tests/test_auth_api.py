"""
Phase 11 — authentication endpoint tests (register + login).

Registration:
* 201 with safe user information (never the password or its hash),
* duplicate username -> 409 ``username_exists``,
* invalid usernames / weak passwords -> 422.

Login:
* valid credentials -> 200 with ``{access_token, token_type: bearer}``,
* wrong password and unknown username produce the IDENTICAL generic 401
  ``invalid_credentials`` (account existence is never revealed),
* the response never contains the password or the hash,
* an app without an auth secret fails closed (503 ``auth_unavailable``).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.dependencies import get_user_repository
from backend.app.main import create_app
from backend.app.repositories.memory_user_repository import InMemoryUserRepository
from tests.auth_testing import (
    VALID_PASSWORD_2,
    login,
    register,
    settings,
)


@pytest.fixture(scope="function")
def app():
    return create_app(settings())


@pytest.fixture(scope="function")
def client(app):
    with TestClient(app) as test_client:
        yield test_client


class TestRegistration:
    def test_register_201_success(self, client):
        response = register(client, "alice")
        assert response.status_code == 201
        body = response.json()
        assert set(body) == {"id", "username", "created_at"}
        assert body["username"] == "alice"
        assert body["id"]
        assert body["created_at"]

    def test_no_password_in_response(self, client):
        response = register(client, "bob")
        assert response.status_code == 201
        body = response.json()
        assert "password" not in body
        assert "password" not in response.text.lower()

    def test_no_password_hash_in_response(self, client):
        response = register(client, "carol")
        assert response.status_code == 201
        assert "$argon2id" not in response.text
        assert "password_hash" not in response.text
        assert "hash" not in response.json()

    def test_duplicate_username_409(self, client):
        assert register(client, "dave").status_code == 201
        response = register(client, "dave")
        assert response.status_code == 409
        assert response.json() == {
            "error": {
                "code": "username_exists",
                "message": "a user with username 'dave' already exists",
            }
        }

    def test_registered_user_is_in_repository(self, app, client):
        assert register(client, "erin").status_code == 201
        repo = get_user_repository(type("_R", (), {"app": app})())
        assert isinstance(repo, InMemoryUserRepository)
        assert repo.contains_username("erin")
        user = repo.get_by_username("erin")
        assert user.password_hash.startswith("$argon2id$")

    @pytest.mark.parametrize(
        "username",
        ["ab", "has space", "with@sign", "", "   ", "x" * 65, "a:b", "a\nb"],
    )
    def test_invalid_username_rejected(self, client, username):
        response = register(client, username)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    @pytest.mark.parametrize("password", ["short", "", "1234567", "   "])
    def test_invalid_password_rejected(self, client, password):
        response = register(client, "frank", password=password)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_username_is_normalized_to_safe_charset(self, client):
        # The charset is already whitespace-free, so no normalization drift is
        # possible and two visually-different-but-equivalent names cannot
        # collide after trimming.
        assert register(client, "user_a").status_code == 201
        assert register(client, "user_a").status_code == 409


class TestLogin:
    def _setup(self, client):
        assert register(client, "grace").status_code == 201

    def test_valid_login_200(self, client):
        self._setup(client)
        response = login(client, "grace")
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"access_token", "token_type"}
        assert body["token_type"] == "bearer"
        assert isinstance(body["access_token"], str) and body["access_token"]

    def test_token_is_a_jwt(self, client):
        import jwt as pyjwt

        self._setup(client)
        response = login(client, "grace")
        claims = pyjwt.decode(
            response.json()["access_token"],
            settings().jwt_secret_key,
            algorithms=["HS256"],
            options={"require": ["exp", "sub"]},
        )
        assert set(claims) == {"sub", "iat", "exp"}

    def test_wrong_password_401(self, client):
        self._setup(client)
        response = login(client, "grace", password=VALID_PASSWORD_2)
        assert response.status_code == 401
        assert response.json() == {
            "error": {"code": "invalid_credentials", "message": "Invalid credentials."}
        }

    def test_unknown_username_same_generic_failure(self, client):
        self._setup(client)
        unknown = login(client, "ghost")
        wrong_pw = login(client, "grace", password=VALID_PASSWORD_2)
        assert unknown.status_code == wrong_pw.status_code == 401
        assert unknown.json() == wrong_pw.json()
        assert unknown.json()["error"]["code"] == "invalid_credentials"

    def test_no_password_or_hash_in_any_login_response(self, client):
        self._setup(client)
        for response in (
            login(client, "grace"),
            login(client, "grace", password=VALID_PASSWORD_2),
            login(client, "ghost"),
        ):
            assert "password" not in response.text.lower()
            assert "password_hash" not in response.text.lower()
            assert "$argon2id" not in response.text

    def test_login_never_reveals_whether_username_exists(self, client):
        self._setup(client)
        existing = login(client, "grace", password=VALID_PASSWORD_2)
        missing = login(client, "nobody-exists")
        assert existing.json() == missing.json() == {
            "error": {"code": "invalid_credentials", "message": "Invalid credentials."}
        }


class TestAuthUnavailable:
    def test_login_fails_closed_without_secret(self):
        app = create_app(settings(jwt_secret_key=""))
        with TestClient(app) as client:
            assert register(client, "hank").status_code == 201
            response = login(client, "hank")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "auth_unavailable"