"""
Phase 11 — authentication dependencies.

:func:`get_current_user` is the single source of authorization for protected
routes. It:

1. reads ``Authorization: Bearer <JWT>`` (via ``HTTPBearer`` with
   ``auto_error=False`` so the error shape stays ours);
2. validates the JWT signature, expiration and ``sub`` claim;
3. loads the user identified by ``sub`` from the user repository;
4. returns a hash-free :class:`CurrentUser` value.

Missing/malformed credentials and invalid/expired/unknown-subject tokens all
produce structured 401 responses. The returned object intentionally carries no
``password_hash`` — identity is control-plane metadata and is never a model
feature.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.auth.errors import (
    AuthenticationRequiredError,
    InvalidCredentialsError,
)
from backend.app.auth.security import decode_access_token
from backend.app.dependencies import get_settings, get_user_repository

# auto_error=False: we raise our own AppError so the response uses the
# project's {"error": {...}} envelope instead of Starlette's default body.
bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    """Safe authenticated identity (no password hash, no raw data).

    ``session_id`` is a stable per-login id derived ONLY from the validated
    bearer token itself (a SHA-256 digest; ``sub@iat`` is gone). It is never
    returned by the API and never contains behavioural data — it only keys the
    server-authoritative, per-session behavioral verification state.
    """

    id: str
    username: str
    session_id: str = ""


def _session_id_from_token(raw_token: str) -> str:
    """Derive a stable per-login session id from the validated bearer token.

    The JWT payload stays the documented minimal ``{sub, iat, exp}`` (asserted
    by the Phase 11 tests) — no new claim (no ``jti``) is added. The session id
    is a SHA-256 digest of the RAW token bytes, giving:

    * replay of the SAME token => the SAME session id (and therefore the same
      behavioral state) — state cannot be forged by swapping tokens;
    * a fresh login => a NEW token => a NEW session id, even for the same user
      with identical ``sub``/``iat``/``exp`` claims, because each issuance
      signs a distinct header ``nonce`` (see ``create_access_token``). The
      same-second ``iat`` collision of the old ``sub@iat`` derivation is gone.

    The digest is control-plane identity for keying
    ``behavioral_session_states`` rows only; it is never returned by the API
    and never enters preprocessing or model features.
    """
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),  # noqa: B008
) -> CurrentUser:
    """Resolve the authenticated user or raise a structured 401.

    ``credentials`` is ``None`` when the Authorization header is missing or
    does not use the Bearer scheme; a syntactically valid Bearer token is then
    validated cryptographically.
    """
    if credentials is None:
        raise AuthenticationRequiredError()

    settings = get_settings(request)
    claims = decode_access_token(settings, credentials.credentials)

    user = get_user_repository(request).get_by_id(claims["sub"])
    if user is None:
        # A correctly signed token for a user that no longer exists is not a
        # valid identity.
        raise InvalidCredentialsError()

    return CurrentUser(
        id=user.id,
        username=user.username,
        session_id=_session_id_from_token(credentials.credentials),
    )


__all__ = ["CurrentUser", "bearer_scheme", "get_current_user"]