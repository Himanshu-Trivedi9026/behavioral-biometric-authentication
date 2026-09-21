"""
Phase 11 — authentication request/response schemas.

The register and login endpoints share the same ``{username, password}`` shape
but validate it differently:

* Registration enforces the project's account policy (username charset,
  minimum password length) — a 422 here is an input error.
* Login stays permissive on input and always answers 401 for a bad username
  or password, so the API never reveals which part was wrong or whether the
  account exists.

Response schemas never carry ``password`` or ``password_hash``; the login
response only carries the signed token.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MIN_USERNAME_LENGTH = 3
MAX_USERNAME_LENGTH = 64
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128

# Account identity is also the behavioral ``user_ref``. Restricting it to a
# safe, whitespace-free charset keeps it usable in SQL, URLs and logs and
# avoids Unicode confusable/whitespace normalization issues.
USERNAME_PATTERN = r"^[A-Za-z0-9._-]+$"


class RegisterRequest(BaseModel):
    """Registration payload: a new username and a policy-compliant password."""

    username: str = Field(
        min_length=MIN_USERNAME_LENGTH,
        max_length=MAX_USERNAME_LENGTH,
        pattern=USERNAME_PATTERN,
        description="Account username (also the behavioral user_ref).",
    )
    password: str = Field(
        min_length=MIN_PASSWORD_LENGTH,
        max_length=MAX_PASSWORD_LENGTH,
        description="Plaintext password (hashed with Argon2id server-side).",
    )


class LoginRequest(BaseModel):
    """Login payload. Input is validated only for shape and sane bounds."""

    username: str = Field(min_length=1, max_length=MAX_USERNAME_LENGTH)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class UserResponse(BaseModel):
    """Safe user information returned on registration (never the hash)."""

    id: str
    username: str
    created_at: str


class TokenResponse(BaseModel):
    """Successful login result: the bearer access token and its type."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"


__all__ = [
    "MAX_PASSWORD_LENGTH",
    "MAX_USERNAME_LENGTH",
    "MIN_PASSWORD_LENGTH",
    "MIN_USERNAME_LENGTH",
    "LoginRequest",
    "RegisterRequest",
    "TokenResponse",
    "UserResponse",
]