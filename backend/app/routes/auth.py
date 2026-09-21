"""
Phase 11 — authentication routes.

``build_versioned_router(prefix)`` registers two endpoints under the API
prefix (e.g. ``/api/v1/auth/...``):

* ``POST <prefix>/auth/register`` — create an account; returns 201 with safe
  user information only (id, username, created_at). Duplicate usernames are
  rejected by the repository (database unique constraint) with 409.
* ``POST <prefix>/auth/login`` — verify credentials and return a signed bearer
  access token. A wrong password and an unknown username both produce the
  identical 401 ``invalid_credentials`` response, so account existence is
  never disclosed.

Neither endpoint ever returns or logs the password or its Argon2id hash, and
neither places sensitive data in the token (only ``sub``/``iat``/``exp``).

These endpoints are public by necessity (a client cannot authenticate before
it has credentials). The protected resources are enrollment and verification.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from backend.app.auth.errors import InvalidCredentialsError
from backend.app.auth.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from backend.app.auth.security import create_access_token, hash_password, verify_password
from backend.app.dependencies import get_settings, get_user_repository
from backend.app.repositories.user_repository import UserRecord, UserRepository
from backend.app.services.profile_store import utc_now


def build_versioned_router(prefix: str) -> APIRouter:
    """Build the authentication router under ``prefix`` (e.g. ``/api/v1``)."""
    api = APIRouter(prefix=prefix, tags=["auth"])

    @api.post(
        "/auth/register",
        response_model=UserResponse,
        status_code=201,
        summary="Register a new user account",
        operation_id="register_user",
    )
    def _register(
        payload: RegisterRequest,
        repository: UserRepository = Depends(get_user_repository),  # noqa: B008
    ) -> UserResponse:
        timestamp = utc_now()
        user = UserRecord(
            id=str(uuid.uuid4()),
            username=payload.username,
            password_hash=hash_password(payload.password),
            created_at=timestamp,
            updated_at=timestamp,
        )
        # The repository's unique constraint is the duplicate guard; no
        # pre-check is performed here.
        repository.create(user)
        return UserResponse(
            id=user.id,
            username=user.username,
            created_at=user.created_at,
        )

    @api.post(
        "/auth/login",
        response_model=TokenResponse,
        summary="Log in and obtain a bearer access token",
        operation_id="login_user",
    )
    def _login(
        payload: LoginRequest,
        repository: UserRepository = Depends(get_user_repository),  # noqa: B008
        settings=Depends(get_settings),  # noqa: B008
    ) -> TokenResponse:
        user = repository.get_by_username(payload.username)
        # Constant response for both branches; do not reveal which failed.
        if user is None or not verify_password(payload.password, user.password_hash):
            raise InvalidCredentialsError()
        token = create_access_token(settings, subject=user.id)
        return TokenResponse(access_token=token, token_type="bearer")

    return api


__all__ = ["build_versioned_router"]