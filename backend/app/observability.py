"""
Phase 15B — structured JSON logging and request correlation.

* :class:`JsonFormatter` renders log records as single-line JSON with a fixed
  whitelist of fields. It never includes free-form ``record.__dict__`` extras;
  only the explicitly approved keys below are emitted, so credentials, request
  bodies, headers, raw behavioural events and derived embeddings can never
  appear in a log line.
* :func:`configure_logging` installs process-wide logging (used by the
  application factory so production logging does not depend on the old
  development-only ``logging.basicConfig`` path). It is idempotent: it adds a
  handler at most once and never removes handlers installed by the test
  runner / tooling.
* :class:`RequestIdMiddleware` accepts a caller-supplied ``X-Request-Id`` (when
  safe), otherwise generates one, echoes it on the response, and emits a
  structured access-log record with ``method`` / ``path`` / ``status`` /
  ``duration_ms`` / ``request_id``. It never logs request or response bodies,
  never reads ``Authorization`` and never logs query strings.

All stack-trace text that a handler would normally write is kept out of the
JSON body for records without ``exc_info``; coloured/sensitive exception detail
appears only when an explicit ``exc_info`` is attached (e.g. the centralised
unhandled-exception handler), still without request payloads.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from typing import Any, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQUEST_ID_HEADER = "X-Request-Id"

# Header values are echoed verbatim onto responses, so constrain the accepted
# charset to RFC-safe, non-newline characters and a sane length.
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:*+\-]{1,128}$")

# The complete set of fields a JSON log line may carry. Anything not listed
# here is deliberately omitted (no request body, no headers, no query, no
# free-form record attributes).
_JSON_FIELDS = (
    "timestamp",
    "level",
    "logger",
    "message",
    "request_id",
    "method",
    "path",
    "status",
    "duration_ms",
    "exception",
)

_ACCESS_LOGGER = logging.getLogger("backend.app.http")
_DEV_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _utc_timestamp(record: logging.LogRecord) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created))


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line with a fixed field whitelist."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "timestamp": _utc_timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in (
            "request_id",
            "method",
            "path",
            "status",
            "duration_ms",
        ):
            value = getattr(record, field, None)
            if value is not None:
                entry[field] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        try:
            return json.dumps(entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive only
            return json.dumps(
                {
                    "timestamp": _utc_timestamp(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": "log record could not be serialized fully",
                },
                ensure_ascii=False,
            )

    @classmethod
    def json_fields(cls) -> tuple:
        return _JSON_FIELDS


def _safe_request_id(raw: Optional[str]) -> str:
    """Return ``raw`` when it is a safe ID, otherwise a generated one."""
    if raw and _SAFE_ID_PATTERN.match(raw):
        return raw
    return secrets.token_hex(16)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Per-request correlation ID: accept-safe-or-generate, echo, and log."""

    async def dispatch(self, request: Request, call_next: Any):  # noqa: ANN401
        request_id = _safe_request_id(request.headers.get(REQUEST_ID_HEADER.lower()))
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        response.headers[REQUEST_ID_HEADER] = request_id
        _ACCESS_LOGGER.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response


_configured = False


def configure_logging(level: str, *, json_format: bool = False) -> None:
    """Configure root logging once per process; never disturbs existing hooks.

    Args:
        level: a log level name (``INFO``, ``DEBUG``, ...).
        json_format: when True, emits structured JSON lines (production) and
            disables uvicorn's own access logger in favour of the middleware's
            structured records.
    """
    global _configured  # noqa: PLW0603 - process-wide idempotent install
    root = logging.getLogger()
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = bool(json_format)
    if _configured:
        return
    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_DEV_FORMAT))
    root.addHandler(handler)
    _configured = True


__all__ = [
    "JsonFormatter",
    "REQUEST_ID_HEADER",
    "RequestIdMiddleware",
    "configure_logging",
]