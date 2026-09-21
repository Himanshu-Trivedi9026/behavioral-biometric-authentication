"""
Phase 10 — storage-layer errors for the profile repository.

These are structured ``AppError`` subclasses so FastAPI's centralized error
handler turns them into a safe JSON envelope. No SQL, connection strings,
hostnames, passwords or behavioral data ever appear in the messages.
"""

from __future__ import annotations

from backend.app.errors import AppError


class DatabaseUnavailableError(AppError):
    """The profile database cannot be reached or is not configured."""

    def __init__(self, message: str = "Profile storage is temporarily unavailable.") -> None:
        super().__init__(code="database_unavailable", message=message, status_code=503)


class DatabaseOperationError(AppError):
    """A database operation failed for an unknown reason."""

    def __init__(self, message: str = "Profile storage failed.") -> None:
        super().__init__(code="database_error", message=message, status_code=500)


__all__ = ["DatabaseOperationError", "DatabaseUnavailableError"]