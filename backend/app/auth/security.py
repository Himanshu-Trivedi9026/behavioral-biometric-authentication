"""
Phase 11 — password hashing (Argon2id) and JWT access tokens.

Password hashing
----------------
Uses :class:`argon2.PasswordHasher` (argon2-cffi) with its default, current
Argon2id parameters (memory 64 MiB, time cost 3, parallelism 4). Salts are
generated per-hash by the library, so hashing the same password twice yields
two different hashes. No custom cryptography is implemented anywhere.

Public API (matching the project's documented contract)::

    hash_password(password) -> str
    verify_password(password, password_hash) -> bool

``verify_password`` returns ``False`` for a wrong password and for a
malformed/hash-shaped-but-invalid value instead of raising, so callers cannot
accidentally leak parsing detail. Password hashes are secrets: they are never
logged and never returned by the API.

JWT
---
Access tokens are signed with PyJWT using HMAC-SHA256 (``HS256``). The payload
is intentionally minimal::

    {"sub": "<user id>", "iat": <now>, "exp": <now + configured minutes>}

No password, password hash, behavioural embeddings, raw keyboard/mouse events,
database credentials or secrets are ever placed in the token. Decoding
validates the signature, the expiration and the required ``sub``/``exp``
claims; every failure collapses to the same structured 401 so token internals
are never disclosed.

Each issuance signs a per-token ``nonce`` into the JOSE *header* (Phase 14B
session identity). The header is not a claim — the payload stays exactly
``{sub, iat, exp}`` — but it makes every freshly issued token a distinct byte
string even when two logins for the same user land in the same ``iat`` second
with the same ``exp``. The Phase 14B ``session_id`` is a digest of that token,
so identical-claim tokens STILL map to two independent sessions.

All PyJWT/argon2 imports are lightweight (pure Python / small CFFI binding),
so importing this module — and therefore building the application — does not
touch a database or the ML stack.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from backend.app.auth.errors import AuthUnavailableError, InvalidCredentialsError
from backend.app.config import Settings

# A single, module-level hasher with the library's secure defaults. Password
# hashing is expensive by design; the instance is reused because hashing
# parameters (not per-hash state) are what matter.
_PASSWORD_HASHER = PasswordHasher()


def hash_password(password: str) -> str:
    """Return a salted Argon2id hash of ``password``.

    Raises:
        ValueError: if ``password`` is not a non-empty string.
    """
    if not isinstance(password, str) or not password or not password.strip():
        raise ValueError("password must be a non-empty string")
    return _PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Return whether ``password`` matches ``password_hash``.

    A wrong password or a malformed/unknown hash format both return ``False``
    (fail safe); this function never raises for bad input and never logs the
    password or the hash.
    """
    if not isinstance(password, str) or not password or not password.strip():
        return False
    if not isinstance(password_hash, str) or not password_hash:
        return False
    # Only Argon2id hashes are ever stored/verified here. A syntactically
    # valid hash of a different type (e.g. argon2i) is rejected up front.
    if not password_hash.startswith("$argon2id$"):
        return False
    try:
        # argon2-cffi signature is verify(hash, password).
        return bool(_PASSWORD_HASHER.verify(password_hash, password))
    except (VerificationError, InvalidHashError, ValueError):
        return False
    except Exception:  # noqa: BLE001 - any failure means "not verified"
        return False


def create_access_token(settings: Settings, subject: str) -> str:
    """Sign a short-lived access token whose ``sub`` is ``subject``.

    Raises:
        AuthUnavailableError: when no strong JWT secret is configured.
        ValueError: when ``subject`` is not a non-empty string.
    """
    if not isinstance(settings, Settings):
        raise TypeError("settings must be a Settings")
    if not settings.jwt_secret_key:
        raise AuthUnavailableError()
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("subject must be a non-empty string")

    issued_at = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": issued_at,
        "exp": issued_at + timedelta(minutes=settings.jwt_access_token_expire_minutes),
    }
    # A header nonce (NOT a payload claim) guarantees a distinct token string
    # per issuance even for two logins in the same iat second. Payload shape
    # is untouched: {sub, iat, exp}.
    nonce = secrets.token_urlsafe(18)
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        headers={"nonce": nonce},
    )


def decode_access_token(settings: Settings, token: str) -> Dict[str, Any]:
    """Validate ``token`` and return its claims.

    Validates the signature, the expiration and the required ``sub``/``exp``
    claims. Any failure raises :class:`InvalidCredentialsError` with a single,
    generic message (no token internals are exposed).
    """
    if not isinstance(settings, Settings):
        raise TypeError("settings must be a Settings")
    if not settings.jwt_secret_key:
        raise InvalidCredentialsError()
    if not isinstance(token, str) or not token.strip():
        raise InvalidCredentialsError()

    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]},
        )
    except jwt.exceptions.InvalidTokenError:
        raise InvalidCredentialsError() from None

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise InvalidCredentialsError()
    return claims


__all__ = [
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "verify_password",
]