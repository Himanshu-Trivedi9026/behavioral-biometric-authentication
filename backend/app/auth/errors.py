"""
Phase 11 — structured authentication/authorization errors.

All errors are :class:`~backend.app.errors.AppError` subclasses, so the
centralized FastAPI handler renders them as the standard envelope:

    {"error": {"code": "...", "message": "..."}}

Every message is deliberately generic and static. In particular the login
failure is identical for "unknown username" and "wrong password" so the API
never reveals whether an account exists. No SQL, connection strings,
credentials, password hashes, JWT secrets, stack traces or internal database
details ever appear here.
"""

from __future__ import annotations

from backend.app.errors import AppError


class AuthenticationRequiredError(AppError):
    """No (or malformed) ``Authorization: Bearer`` credentials were supplied."""

    def __init__(self, message: str = "Authentication is required.") -> None:
        super().__init__(code="authentication_required", message=message, status_code=401)


class InvalidCredentialsError(AppError):
    """Credentials/token are invalid, expired or otherwise unusable."""

    def __init__(self, message: str = "Invalid credentials.") -> None:
        super().__init__(code="invalid_credentials", message=message, status_code=401)


class ForbiddenError(AppError):
    """Authenticated, but not authorized to access the requested resource."""

    def __init__(
        self, message: str = "You are not authorized to access this profile."
    ) -> None:
        super().__init__(code="forbidden", message=message, status_code=403)


class ReverificationRequiredError(AppError):
    """The login session's behavioral state requires re-verification.

    Raised (403, ``reverification_required``) by the Phase 14B fail-closed gate
    on protected actions when the session is not currently VERIFIED by a
    continuous-verification window. The continuous-verification endpoint itself
    stays available so the user can re-verify and clear the gate — this is a
    per-session, behavior-driven block, not an account lockout.
    """

    def __init__(
        self,
        message: str = "Behavioral re-verification is required before this action.",
    ) -> None:
        super().__init__(
            code="reverification_required", message=message, status_code=403
        )


class AuthUnavailableError(AppError):
    """Authentication is not configured (no strong JWT secret available)."""

    def __init__(self, message: str = "Authentication is not configured.") -> None:
        super().__init__(code="auth_unavailable", message=message, status_code=503)


__all__ = [
    "AuthenticationRequiredError",
    "AuthUnavailableError",
    "ForbiddenError",
    "InvalidCredentialsError",
    "ReverificationRequiredError",
]