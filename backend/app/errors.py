"""
Phase 9A — centralized API error handling.

Every error response uses the same JSON envelope:

    {"error": {"code": "...", "message": "..."}}

Partial validation errors additionally carry a ``details`` list.  Internal
exceptions are logged server-side and replaced with a generic, safe 500
response — stack traces, filesystem paths, environment variables and internal
exception details are never included in API responses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Union

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("backend.app.errors")

_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "validation_error",
    429: "too_many_requests",
    500: "internal_error",
    502: "bad_gateway",
    503: "service_unavailable",
}


class AppError(Exception):
    """A domain error with a machine-readable code and HTTP status."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        if not isinstance(code, str) or not code:
            raise ValueError("AppError code must be a non-empty string")
        if not isinstance(message, str) or not message:
            raise ValueError("AppError message must be a non-empty string")
        if not isinstance(status_code, int) or not 400 <= status_code < 600:
            raise ValueError("AppError status_code must be in [400, 600)")
        self.code = code
        self.message = message
        self.status_code = status_code


def error_payload(code: str, message: str, details: Optional[List[Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return payload


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: Optional[List[Any]] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_payload(code, message, details),
    )


def _code_for_status(status_code: int) -> str:
    return _STATUS_CODES.get(status_code, "http_error")


async def _handle_validation_error(
    request: Request, exc: RequestValidationError  # pylint: disable=unused-argument
) -> JSONResponse:
    details = [
        {
            "loc": [str(part) for part in getattr(error, "loc", ())],
            "msg": error.get("msg", "validation error"),
            "type": error.get("type", "unknown"),
        }
        for error in exc.errors()
    ]
    message = "request validation failed" if details else "invalid request"
    return error_response(status_code=422, code="validation_error", message=message, details=details)


async def _handle_http_exception(
    request: Request, exc: StarletteHTTPException  # pylint: disable=unused-argument
) -> JSONResponse:
    code = _code_for_status(exc.status_code)
    detail = exc.detail
    message = detail if isinstance(detail, str) and detail else code.replace("_", " ").title()
    return error_response(status_code=exc.status_code, code=code, message=message)


async def _handle_app_error(
    request: Request, exc: AppError  # pylint: disable=unused-argument
) -> JSONResponse:
    """Structured response for domain errors raised by services/routes."""
    return error_response(status_code=exc.status_code, code=exc.code, message=exc.message)


async def _handle_unhandled_exception(
    request: Request, exc: Exception  # pylint: disable=unused-argument
) -> JSONResponse:
    logger.exception("unhandled exception during request handling")
    return error_response(
        status_code=500,
        code="internal_error",
        message="An unexpected internal error occurred.",
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Register the centralized error handlers on ``app``."""
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(Exception, _handle_unhandled_exception)


__all__ = [
    "AppError",
    "error_payload",
    "error_response",
    "install_exception_handlers",
]