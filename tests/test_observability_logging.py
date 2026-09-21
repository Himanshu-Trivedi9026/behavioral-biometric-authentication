"""
Phase 15B — structured JSON logging and X-Request-Id correlation.

Covers the JSON formatter's field whitelist, configure_logging idempotency,
request-id generation/accept/echo, the structured access-record shape, and the
privacy guarantees: no request/response bodies, no Authorization headers, no
passwords, no raw behavioural events are ever logged.
"""

import json
import logging
import re

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.observability import JsonFormatter, configure_logging

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _app(environment="test"):
    return create_app(Settings(environment=environment, database_url=""))


def _record(**extras):
    record = logging.LogRecord(
        name="backend.app.http",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request",
        args=(),
        exc_info=None,
    )
    for key, value in extras.items():
        setattr(record, key, value)
    return record


class TestJsonFormatter:
    def test_formatter_whitelists_fields_only(self):
        record = _record(
            request_id="rid-1",
            method="GET",
            path="/health",
            status=200,
            duration_ms=1.25,
            body={"password": "SecretOne", "keyboard_events": [{"key": "a"}]},
            authorization="Bearer s3cret-jwt",
        )
        rendered = json.loads(JsonFormatter().format(record))
        assert set(rendered) <= set(JsonFormatter.json_fields())
        assert "body" not in rendered
        assert "authorization" not in rendered
        assert "headers" not in rendered

    def test_formatter_emits_expected_fields(self):
        record = _record(
            request_id="rid-9", method="POST", path="/api/v1/auth/register",
            status=500, duration_ms=3.5,
        )
        rendered = json.loads(JsonFormatter().format(record))
        assert rendered["request_id"] == "rid-9"
        assert rendered["method"] == "POST"
        assert rendered["path"] == "/api/v1/auth/register"
        assert rendered["status"] == 500
        assert rendered["duration_ms"] == 3.5
        assert rendered["level"] == "INFO"
        assert rendered["message"] == "request"

    def test_formatter_includes_exception_text_when_explicit(self):
        import sys

        try:
            raise RuntimeError("boom-internal-only")
        except RuntimeError:
            record = _record()
            record.exc_info = sys.exc_info()
        rendered = json.loads(JsonFormatter().format(record))
        assert "boom-internal-only" in rendered["exception"]


class TestConfigureLogging:
    def test_log_level_applied_to_root(self):
        configure_logging("INFO", json_format=False)
        assert logging.getLogger().level == logging.INFO

    def test_json_format_disables_uvicorn_access_logger(self):
        configure_logging("INFO", json_format=True)
        assert logging.getLogger("uvicorn.access").disabled is True

    def test_dev_format_reenables_uvicorn_access_logger(self):
        configure_logging("INFO", json_format=False)
        assert logging.getLogger("uvicorn.access").disabled is False


class TestRequestIdMiddleware:
    def test_generates_id_when_absent(self, caplog):
        app = _app()
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.headers.get("X-Request-Id")
        assert _ID_PATTERN.match(response.headers["X-Request-Id"])

    def test_accepts_safe_supplied_id(self, caplog):
        app = _app()
        with TestClient(app) as client:
            response = client.get("/health", headers={"X-Request-Id": "my-test-correlation-42"})
            assert response.headers["X-Request-Id"] == "my-test-correlation-42"

    def test_rejects_unsafe_supplied_id(self, caplog):
        app = _app()
        unsafe = "bad id with spaces'quote"
        with TestClient(app) as client:
            response = client.get("/health", headers={"X-Request-Id": unsafe})
            value = response.headers["X-Request-Id"]
            assert value != unsafe
            assert _ID_PATTERN.match(value)

    def test_structured_access_record_shape(self, caplog):
        caplog.set_level(logging.INFO)
        app = _app()
        with TestClient(app) as client:
            client.get("/health")
        records = [r for r in caplog.records if r.name == "backend.app.http"]
        assert records
        latest = records[-1]
        assert latest.method == "GET"
        assert latest.path == "/health"
        assert latest.status == 200
        assert isinstance(latest.duration_ms, float)
        assert _ID_PATTERN.match(latest.request_id)

    def test_status_and_duration_present_on_error(self, caplog):
        caplog.set_level(logging.INFO)
        app = _app()
        with TestClient(app) as client:
            response = client.get("/not-a-route")
            assert response.status_code == 404
        records = [r for r in caplog.records if r.name == "backend.app.http"]
        assert records[-1].status == 404
        assert records[-1].duration_ms > 0


class TestLoggingPrivacy:
    def test_password_never_logged(self, caplog):
        caplog.set_level(logging.INFO)
        app = _app()
        with TestClient(app) as client:
            client.post(
                "/api/v1/auth/register",
                json={"username": "alice", "password": "SuperSecretPass42!"},
            )
        assert "SuperSecretPass42!" not in caplog.text

    def test_authorization_header_never_logged(self, caplog):
        caplog.set_level(logging.INFO)
        app = _app()
        token = "Bearer this-is-a-secret-jwt-token-0000"
        with TestClient(app) as client:
            client.post("/api/v1/enrollment", headers={"Authorization": token}, json={})
        assert "this-is-a-secret-jwt-token-0000" not in caplog.text

    def test_raw_behavioural_events_never_logged(self, caplog):
        caplog.set_level(logging.INFO)
        app = _app()
        payload = {
            "sessions": [
                {
                    "session_id": "s1",
                    "started_at": "2025-01-01T00:00:00Z",
                    "ended_at": "2025-01-01T00:00:05Z",
                    "timestamp_source": "test",
                    "keyboard_events": [{"event_type": "keyboard", "event": "keydown", "key": "a", "timestamp": 0}],
                    "mouse_events": [{"event_type": "mouse", "event": "mousemove", "x": 12.5, "y": 9.25, "timestamp": 1}],
                }
            ]
        }
        with TestClient(app) as client:
            client.post("/api/v1/enrollment", json=payload)
        for marker in ("keydown", "mousemove", '"key": "a"', "12.5"):
            assert marker not in caplog.text

    def test_query_string_never_logged(self, caplog):
        caplog.set_level(logging.INFO)
        app = _app()
        with TestClient(app) as client:
            client.get("/api/v1/health?token=ultra-secret-query")
        records = [r for r in caplog.records if r.name == "backend.app.http"]
        assert records
        for record in records:
            assert "ultra-secret-query" not in str(record.path)
            assert "?token=" not in str(record.path)


class TestUnhandledExceptionLogging:
    def test_response_hides_exception_and_logs_without_secrets(self, caplog):
        caplog.set_level(logging.INFO)
        app = _app()

        def _boom():
            raise RuntimeError("confidential-stack-detail")

        app.get("/_boom")(_boom)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/_boom")
        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "internal_error"
        assert "confidential-stack-detail" not in response.text
        # Server-side logs keep the diagnostic but never request payload/secrets.
        assert "confidential-stack-detail" in caplog.text
        assert "Bearer" not in caplog.text


class TestLogLevelValidation:
    def test_valid_levels_accepted(self):
        for level in ("INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"):
            settings = Settings(environment="test", log_level=level)
            assert settings.log_level == level

    def test_case_insensitive_level_normalized_for_validation(self):
        assert Settings(environment="test", log_level="debug").log_level == "debug"

    def test_invalid_level_rejected(self):
        for level in ("verbose", "TRACE", "", "info,debug", " WARNINGIT "):
            with pytest.raises(ValueError):
                Settings(environment="test", log_level=level)

    def test_default_is_info(self):
        assert Settings(environment="test").log_level == "INFO"